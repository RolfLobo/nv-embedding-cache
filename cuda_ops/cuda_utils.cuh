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
#include "cuda_ops/cuda_common.h"
#include "cuda_ops/kernels_common.cuh"
#include "cuda_ops/deduper.cuh"

template<typename IndexT, uint32_t SUBWARP_WIDTH, typename DataType>
__global__ void UnpackDedupKernel(const IndexT* d_inverse_buffer, const IndexT* d_offsets, const IndexT* d_counts_out, const int8_t* uniq_buffer, int8_t* out_buffer, uint64_t rowSizeInBytes)
{
    DataType localBuffer[4];
    const uint32_t rl_idx = blockIdx.x; 
    auto intra_warp_idx = threadIdx.x;
    constexpr auto ELEMENT_SIZE = sizeof(DataType);
    // load input
    auto src_ptr = uniq_buffer + rl_idx * rowSizeInBytes;
    for (uint32_t k = 0, h = 0; k < rowSizeInBytes; k += ELEMENT_SIZE*SUBWARP_WIDTH, h++)
    {
        uint32_t offset = k + intra_warp_idx * ELEMENT_SIZE;
        if (offset < rowSizeInBytes)
        {
            DataType d = *(DataType*)(src_ptr + offset);
            localBuffer[h] = d;
        }
    }

    auto num_outputs = d_counts_out[rl_idx];
    for (uint32_t i = 0; i < num_outputs; i++)
    {
        for (uint32_t k = 0, h = 0; k < rowSizeInBytes; k += ELEMENT_SIZE*SUBWARP_WIDTH, h++)
        {
            uint32_t offset = k + intra_warp_idx * ELEMENT_SIZE;
            auto outputIdx = d_inverse_buffer[d_offsets[rl_idx] + i];
            if (offset < rowSizeInBytes)
            {
                DataType* dst_ptr = (DataType*)(out_buffer + outputIdx * rowSizeInBytes + offset);
                *dst_ptr = localBuffer[h];
            }
        }
    }
}

template<typename IndexT>
void UnpackDedup(const IndexT* d_inverse_buffer, const IndexT* d_offsets, const IndexT* d_counts_out, const int8_t* d_uniq_buffer, int8_t* d_out_buffer, uint64_t num_counts, uint64_t rowSizeInBytes, cudaStream_t stream)
{
    const uint32_t blockX = 32;
    const uint32_t blockY = 1;
    const uint32_t nBlock = static_cast<uint32_t>(num_counts);
    dim3 gridDims(nBlock);
    dim3 blockDims(blockX, blockY);
    if (rowSizeInBytes % sizeof(uint4) == 0)
    {
        UnpackDedupKernel<IndexT, 32, uint4><<<gridDims, blockDims, 0, stream>>>(d_inverse_buffer, d_offsets, d_counts_out, d_uniq_buffer, d_out_buffer, rowSizeInBytes);
    }
    else if (rowSizeInBytes % sizeof(uint32_t) == 0)
    {
        UnpackDedupKernel<IndexT, 32, uint32_t><<<gridDims, blockDims, 0, stream>>>(d_inverse_buffer, d_offsets, d_counts_out, d_uniq_buffer, d_out_buffer, rowSizeInBytes);
    }
    else
    {
        UnpackDedupKernel<IndexT, 32, uint8_t><<<gridDims, blockDims, 0, stream>>>(d_inverse_buffer, d_offsets, d_counts_out, d_uniq_buffer, d_out_buffer, rowSizeInBytes);
    }
    NVE_CHECK_(cudaGetLastError()); // Check kernel launch didn't generate an error
}

template<uint32_t SubwarpWidth, typename DataType>
__global__ void EmbedPooling(const DataType* __restrict__ src,
                             DataType* __restrict__ dst,
                             const uint32_t num_elements,
                             const uint32_t embed_src_stride,
                             const uint32_t embed_dst_stride,
                             const uint32_t hotness,
                             const uint32_t num_bags)
{
    const int bag = blockIdx.x * blockDim.y + threadIdx.y;

    if (bag >= num_bags) {
      return;
    }

    for (uint32_t el = threadIdx.x; el < num_elements; el += SubwarpWidth) {
      DataType acc;
      InitAcc(acc);
      for (uint32_t i = 0; i < hotness; i++) {
        const DataType* embed_src = src + (bag * hotness + i) * embed_src_stride + el;
        nve::Accumulate(acc, *embed_src);
      }
      DataType* embed_dst = dst + bag * embed_dst_stride + el;
      *embed_dst = acc;
    }
}

template<uint32_t SubwarpWidth, typename DataType>
void CallPoolingKernelVecTypeSubwarp(
                              const DataType* __restrict__ src,
                              DataType* __restrict__ dst,
                              const uint32_t num_elements,
                              const uint32_t embed_src_stride,
                              const uint32_t embed_dst_stride,
                              const uint32_t hotness,
                              const uint32_t num_bags,
                              const cudaStream_t stream = 0)
{
    uint32_t bags_per_warp = 32 / SubwarpWidth;
    uint32_t bags_per_sm = bags_per_warp * 4;
    dim3 grid_size ((num_bags + bags_per_sm - 1) / bags_per_sm, 1);
    dim3 block_size (SubwarpWidth, bags_per_sm);
    EmbedPooling<SubwarpWidth, DataType><<<grid_size, block_size, 0, stream>>>(
        src, dst, num_elements, embed_src_stride, embed_dst_stride, hotness, num_bags);
    NVE_CHECK_(cudaGetLastError()); // Check kernel launch didn't generate an error
}
