/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_runtime.h>
#include <ecache/embedding_cache_combined.h>
#include <ecache/embedding_cache_combined.cuh>
#include "kernels_common.cuh"
#include "cuda_ops/cuda_common.h"

namespace nve {

// Gather one embedding row per key and write it out dequantized (no pooling). This is the
// hotness-1 / no-accumulate sibling of FindAndCombine: every key produces one output row of
// QuantizationHelper<DTYPE>::ParamType values, reconstructed as float(q) * scale [+ offset].
//
// A quantized row is laid out as [ q[row_size_in_elements] ][ scale ][ offset? ] where q is
// DTYPE's storage type and scale/offset are ParamType, matching get_row_size_in_bytes<DTYPE>().
//
// Each warp (SUBWARP_WIDTH lanes) resolves SUBWARP_WIDTH row addresses (one per lane), then walks
// s = 0..SUBWARP_WIDTH-1 broadcasting lane s's address to the whole warp so the warp cooperatively
// streams that one row out, vectorized over InputVecType/OutputVecType.
template<DataType_t DTYPE,
         typename IndexT,
         typename CacheDataT,
         typename InputVecType,
         typename OutputVecType,
         uint32_t SUBWARP_WIDTH,
         uint32_t BLOCK_Y,
         bool LOAD_INDICES>
__global__ void find_and_dequant(const IndexT* d_keys, const size_t len,
    int8_t* d_values, const int8_t* __restrict__ d_table,
    CacheDataT data, uint32_t curr_table, size_t stride, uint32_t row_size_in_elements)
{
    using ParamType = typename QuantizationHelper<DTYPE>::ParamType;
    constexpr uint32_t ELEMENT_SIZE = QuantizationHelper<DTYPE>::element_size;
    constexpr uint32_t VEC_SIZE_IN_ELEMENTS = sizeof(OutputVecType) / sizeof(ParamType);
    const uint32_t row_size_in_vecs = row_size_in_elements / VEC_SIZE_IN_ELEMENTS;

    const uint32_t tid_batch = blockIdx.x * SUBWARP_WIDTH * BLOCK_Y + threadIdx.y * SUBWARP_WIDTH;
    const uint32_t tid = tid_batch + threadIdx.x; // each lane resolves one key's row address

    uint64_t laneptr;
    if (tid >= len)
    {
        laneptr = 0;
    }
    else
    {
        IndexT lane_idx = LOAD_INDICES ? d_keys[tid] : static_cast<IndexT>(tid);
        laneptr = AddressFunctor<IndexT, CacheDataT>::get_address(lane_idx, d_table, curr_table, data);
    }

    __syncwarp();

    for (uint32_t s = 0; s < SUBWARP_WIDTH; s++)
    {
        const uint64_t src_ptr = __shfl_sync(0xffffffff, laneptr, s, SUBWARP_WIDTH);
        const uint32_t out_row = tid_batch + s;
        // laneptr is 0 only for out-of-range lanes (a resolved address is table + offset, never 0),
        // so src_ptr != 0 already bounds out_row < len; the explicit check keeps that obvious.
        if (src_ptr != 0 && out_row < len)
        {
            ParamType qscale = ParamType(1);
            ParamType qoffset = ParamType(0);
            if constexpr (QuantizationHelper<DTYPE>::has_scale)
            {
                memcpy(&qscale, (const char*)src_ptr + row_size_in_elements * ELEMENT_SIZE, sizeof(ParamType));
            }
            if constexpr (QuantizationHelper<DTYPE>::has_offset)
            {
                memcpy(&qoffset, (const char*)src_ptr + row_size_in_elements * ELEMENT_SIZE + sizeof(ParamType), sizeof(ParamType));
            }
            // Unroll the copy explicitly: issue UNROLL_FACTOR independent loads up front so they
            // stay in flight together (ILP / memory-level parallelism), then dequantize and store.
            constexpr uint32_t UNROLL_FACTOR = 4;
            for (uint32_t base = 0; base < row_size_in_vecs; base += SUBWARP_WIDTH * UNROLL_FACTOR)
            {
                InputVecType in[UNROLL_FACTOR];
                #pragma unroll
                for (uint32_t u = 0; u < UNROLL_FACTOR; u++)
                {
                    uint32_t offset = base + threadIdx.x + u * SUBWARP_WIDTH;
                    if (offset < row_size_in_vecs)
                    {
                        in[u] = *((const InputVecType*)src_ptr + offset);
                    }
                }
                #pragma unroll
                for (uint32_t u = 0; u < UNROLL_FACTOR; u++)
                {
                    uint32_t offset = base + threadIdx.x + u * SUBWARP_WIDTH;
                    if (offset < row_size_in_vecs)
                    {
                        OutputVecType out = dequantize<QuantizationHelper<DTYPE>, OutputVecType, InputVecType>(in[u], qscale, qoffset);
                        *(OutputVecType*)(d_values + out_row * stride + offset * sizeof(OutputVecType)) = out;
                    }
                }
            }
        }
    }
}

// Picks the widest vector load whose row layout stays aligned (Vec4 -> Vec2 -> Vec1) and launches.
template<DataType_t DTYPE, typename IndexT, typename CacheDataT>
cudaError_t call_find_and_dequant_resolved(const IndexT* d_keys, const size_t len,
    int8_t* d_values, const int8_t* d_table, uint32_t row_size_in_elements,
    CacheDataT data, cudaStream_t stream, size_t stride, bool load_indices, uint32_t curr_table)
{
    constexpr uint32_t blockX = 32;
    constexpr uint32_t blockY = 4;
    constexpr uint32_t blockSize = blockX * blockY;
    if (len == 0)
    {
        return cudaSuccess;
    }
    const uint32_t nBlock = static_cast<uint32_t>((len + blockSize - 1) / blockSize);
    dim3 gridDims(nBlock);
    dim3 blockDims(blockX, blockY);

    using ParamType = typename QuantizationHelper<DTYPE>::ParamType;
    const uint32_t row_size_in_bytes = get_row_size_in_bytes<DTYPE>(row_size_in_elements);

    #define NVE_FAD_DISPATCH(INPUT_VEC_TYPE, OUTPUT_VEC_TYPE) \
        do { \
            if (load_indices) \
            { \
                find_and_dequant<DTYPE, IndexT, CacheDataT, INPUT_VEC_TYPE, OUTPUT_VEC_TYPE, blockX, blockY, true> \
                    <<<gridDims, blockDims, 0, stream>>>(d_keys, len, d_values, d_table, data, curr_table, stride, row_size_in_elements); \
            } \
            else \
            { \
                find_and_dequant<DTYPE, IndexT, CacheDataT, INPUT_VEC_TYPE, OUTPUT_VEC_TYPE, blockX, blockY, false> \
                    <<<gridDims, blockDims, 0, stream>>>(d_keys, len, d_values, d_table, data, curr_table, stride, row_size_in_elements); \
            } \
        } while (0)

    if (row_size_in_bytes % sizeof(typename QuantizationHelper<DTYPE>::Vec4) == 0 && row_size_in_elements % 4 == 0)
    {
        NVE_FAD_DISPATCH(typename QuantizationHelper<DTYPE>::Vec4, typename VecWidthHelper<ParamType>::Vec4);
    }
    else if (row_size_in_bytes % sizeof(typename QuantizationHelper<DTYPE>::Vec2) == 0 && row_size_in_elements % 2 == 0)
    {
        NVE_FAD_DISPATCH(typename QuantizationHelper<DTYPE>::Vec2, typename VecWidthHelper<ParamType>::Vec2);
    }
    else
    {
        NVE_FAD_DISPATCH(typename QuantizationHelper<DTYPE>::Vec1, typename VecWidthHelper<ParamType>::Vec1);
    }
    #undef NVE_FAD_DISPATCH
    return cudaGetLastError();
}

template<typename IndexT, typename CacheDataT>
cudaError_t call_find_and_dequant(const IndexT* d_keys, const size_t len,
    int8_t* d_values, const int8_t* d_table, DataType_t dtype, uint32_t row_size_in_elements,
    CacheDataT data, cudaStream_t stream, size_t stride, bool load_indices, uint32_t curr_table = 0)
{
    switch (dtype) {
        case DataType_t::Float32:
            return call_find_and_dequant_resolved<DataType_t::Float32, IndexT, CacheDataT>(d_keys, len, d_values, d_table, row_size_in_elements, data, stream, stride, load_indices, curr_table);
        case DataType_t::Float16:
            return call_find_and_dequant_resolved<DataType_t::Float16, IndexT, CacheDataT>(d_keys, len, d_values, d_table, row_size_in_elements, data, stream, stride, load_indices, curr_table);
        case DataType_t::QInt8RowwiseF32:
            return call_find_and_dequant_resolved<DataType_t::QInt8RowwiseF32, IndexT, CacheDataT>(d_keys, len, d_values, d_table, row_size_in_elements, data, stream, stride, load_indices, curr_table);
        case DataType_t::QInt8RowwiseF16:
            return call_find_and_dequant_resolved<DataType_t::QInt8RowwiseF16, IndexT, CacheDataT>(d_keys, len, d_values, d_table, row_size_in_elements, data, stream, stride, load_indices, curr_table);
        case DataType_t::QUint8RowwiseF32:
            return call_find_and_dequant_resolved<DataType_t::QUint8RowwiseF32, IndexT, CacheDataT>(d_keys, len, d_values, d_table, row_size_in_elements, data, stream, stride, load_indices, curr_table);
        case DataType_t::QUint8RowwiseF16:
            return call_find_and_dequant_resolved<DataType_t::QUint8RowwiseF16, IndexT, CacheDataT>(d_keys, len, d_values, d_table, row_size_in_elements, data, stream, stride, load_indices, curr_table);
        default:
            NVE_THROW_("Unsupported data type for find_and_dequant ", dtype);
    }
    return cudaErrorInvalidValue;
}

} // namespace nve
