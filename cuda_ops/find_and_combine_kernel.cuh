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

#include <algorithm>
#include <ecache/embedding_cache_combined.h>
#include <ecache/embedding_cache_combined.cuh>
#include "kernels_common.cuh"
#include "cuda_ops/cuda_common.h"

using namespace nve;

// Bit layout for the packed FindAndCombine MASK template parameter. The kernel unpacks these and
// get_mask() packs them; cmake/gen_find_and_combine_instantiations.py enumerates MASK over
// [0, 16) with this same layout, so all three must stay in sync.
constexpr uint32_t NVE_FAC_MASK_FIXED_HOTNESS = 1u << 0;
constexpr uint32_t NVE_FAC_MASK_SUM_POOLING   = 1u << 1;
constexpr uint32_t NVE_FAC_MASK_IS_WEIGHTED   = 1u << 2;
constexpr uint32_t NVE_FAC_MASK_LOAD_INDICES  = 1u << 3;

template<DataType_t ELEMENT_TYPE_ID, typename INDEX_TYPE, typename ACC_TYPE, typename WEIGHT_TYPE,
         typename INPUT_VEC_TYPE,
         typename ELEMENT_VEC_TYPE, typename ACC_VEC_TYPE, typename CacheDataT,
         uint32_t SZ_ACCUM, uint32_t MASK>
__launch_bounds__(128, 1)
__global__ void FindAndCombine(const uint32_t batchSz,
                               const int8_t* __restrict__ table,
                               const INDEX_TYPE* __restrict__ indices,
                               const INDEX_TYPE* __restrict__ offsets,
                               const WEIGHT_TYPE* __restrict__ weights,
                               int32_t num_hot,  CacheDataT cache,
                               int32_t rowSizeInElements,
                               typename QuantizationHelper<ELEMENT_TYPE_ID>::ParamType* pOutput)
{
    constexpr bool FIXED_HOTNESS = (MASK & NVE_FAC_MASK_FIXED_HOTNESS) != 0u;
    constexpr bool SUM_POOLING   = (MASK & NVE_FAC_MASK_SUM_POOLING)   != 0u;
    constexpr bool IS_WEIGHTED   = (MASK & NVE_FAC_MASK_IS_WEIGHTED)   != 0u;
    constexpr bool LOAD_INDICES  = (MASK & NVE_FAC_MASK_LOAD_INDICES)  != 0u;
    using ELEMENT_TYPE = typename QuantizationHelper<ELEMENT_TYPE_ID>::ParamType;
    const int32_t sampleId = blockIdx.x;
    constexpr int32_t SUBWARP_WIDTH = 32;

    uint32_t hotness = num_hot;
    if constexpr (!FIXED_HOTNESS) {
        hotness = offsets[sampleId + 1] - offsets[sampleId];
    }

    const uint32_t hotnessTid = threadIdx.x;
    
    ACC_VEC_TYPE acc[SZ_ACCUM];
    ACC_TYPE acc_weight[SZ_ACCUM];

    constexpr auto vecSizeInElements = sizeof(ELEMENT_VEC_TYPE) / sizeof(ELEMENT_TYPE);
    int32_t rowSizeInVecs = rowSizeInElements / vecSizeInElements;
    const INDEX_TYPE sampleStart = FIXED_HOTNESS ? sampleId * hotness :  offsets[sampleId];
    const INDEX_TYPE sampleEnd = sampleStart + hotness;
    const INDEX_TYPE* currIdx = indices + sampleStart;
    for (int base_offset = 0; base_offset < rowSizeInVecs; base_offset += SUBWARP_WIDTH * SZ_ACCUM) {
      for (uint32_t k = 0; k < SZ_ACCUM; k++)
      {
          InitAcc(acc[k]);
          if (!SUM_POOLING) {
            InitAcc(acc_weight[k]);
          }
      }
      for (int i = 0; i < hotness; i += SUBWARP_WIDTH)
      {
          // TODO: should be conditional load
          INDEX_TYPE laneIdx = ((i + hotnessTid) < hotness) ? (LOAD_INDICES ? __ldcv(currIdx + i + hotnessTid) : (sampleStart + i + hotnessTid)) : 0;
          uint64_t lanePtr = AddressFunctor<INDEX_TYPE, CacheDataT>::get_address(laneIdx, table, 0, cache);
          #pragma unroll 4
          for (int s = 0; s < SUBWARP_WIDTH; s++)
          {
              uint64_t ptr = __shfl_sync(0xffffffff, lanePtr, s, SUBWARP_WIDTH);
              ELEMENT_TYPE qscale = 1;
              ELEMENT_TYPE qoffset = 0;
              if (QuantizationHelper<ELEMENT_TYPE_ID>::has_scale) {
                  memcpy(&qscale, (const char*)(ptr) + rowSizeInElements * QuantizationHelper<ELEMENT_TYPE_ID>::element_size, sizeof(ELEMENT_TYPE));
              }
              if (QuantizationHelper<ELEMENT_TYPE_ID>::has_offset) {
                  memcpy(&qoffset, (const char*)(ptr) + rowSizeInElements * QuantizationHelper<ELEMENT_TYPE_ID>::element_size + sizeof(ELEMENT_TYPE), sizeof(ELEMENT_TYPE));
              }

              if ((sampleStart + i + s) < sampleEnd) {
                  for (uint32_t k = 0; k < SZ_ACCUM; k++)
                  {
                      uint32_t offset = base_offset + hotnessTid + k * SUBWARP_WIDTH;
                      if (offset < static_cast<uint32_t>(rowSizeInVecs))
                      {
                          INPUT_VEC_TYPE el_quantized = *((INPUT_VEC_TYPE*)ptr + offset);
                          ELEMENT_VEC_TYPE el_raw = dequantize<nve::QuantizationHelper<ELEMENT_TYPE_ID>, ELEMENT_VEC_TYPE, INPUT_VEC_TYPE>(el_quantized, qscale, qoffset);
                          ACC_VEC_TYPE el_cast = Cast<ELEMENT_VEC_TYPE, ACC_VEC_TYPE>(el_raw);
                          if (SUM_POOLING) {
                              if (IS_WEIGHTED) {
                                  WEIGHT_TYPE weight_raw = weights[sampleStart + i + s];
                                  ACC_TYPE weight_cast = weight_raw;
                                  
                          MulAccumulate(acc[k], el_cast, weight_cast);
                              } else {
                                  Accumulate(acc[k], el_cast);
                              }
                          } else {
                              if (IS_WEIGHTED) {
                                  ACC_TYPE weight_cast = weights[sampleStart + i + s];
                                  MulAccumulate(acc[k], el_cast, weight_cast);
                                  Accumulate(acc_weight[k], weight_cast);
                              } else {
                                  Accumulate(acc[k], el_cast);
                                  Accumulate(acc_weight[k], ACC_TYPE(1.0));
                              }
                          }         
                      }
                  }
              }
          }
      }

      ELEMENT_VEC_TYPE* currOut = reinterpret_cast<ELEMENT_VEC_TYPE*>(pOutput + sampleId * rowSizeInElements);

      // TODO: try loop on acc size
      for (int32_t j = hotnessTid + base_offset, k = 0; k < SZ_ACCUM; j += SUBWARP_WIDTH, ++k)
      {
          if (j >= rowSizeInVecs) {
            break;
          }
          if (!SUM_POOLING) {
              // Match the CPU pooling kernels: a zero averaging denominator (an empty bag, or
              // weights summing to zero) yields a zero output row instead of dividing by zero.
              if (IsZero(acc_weight[k])) {
                  InitAcc(acc[k]);
              } else {
                  // TODO: replace by inverse and mul?
                  Div(acc[k], acc_weight[k]);
              }
          }
          currOut[j] = Cast<ACC_VEC_TYPE, ELEMENT_VEC_TYPE>(acc[k]);
      }
    }
}

// Host launch wrapper for one FindAndCombine combination. Kept as a plain (non-__global__) host
// function template so it can be explicitly instantiated in a separate translation unit and linked
// without relocatable device code: the kernel launch (and hence the kernel's device-code
// instantiation) stays co-located with this wrapper in whichever TU instantiates it. The dispatch
// in callFindAndCombineKernelTypesResolved routes every launch through this wrapper; the generated
// inst_*.cu files provide the definitions and gpu_table.cu sees only the extern declarations below.
template<DataType_t ELEMENT_TYPE_ID, typename INDEX_TYPE, typename ACC_TYPE, typename WEIGHT_TYPE,
         typename INPUT_VEC_TYPE, typename ELEMENT_VEC_TYPE, typename ACC_VEC_TYPE, typename CacheDataT,
         uint32_t SZ_ACCUM, uint32_t MASK>
void nve_fac_launch(const uint32_t batchSz,
                    const int8_t* table,
                    const INDEX_TYPE* indices,
                    const INDEX_TYPE* offsets,
                    const WEIGHT_TYPE* weights,
                    int32_t num_hot,
                    CacheDataT cache,
                    int32_t rowSizeInElements,
                    void* output,
                    cudaStream_t stream)
{
    dim3 gridSize(batchSz, 1);
    dim3 blockSize(32, 1);
    FindAndCombine<ELEMENT_TYPE_ID, INDEX_TYPE, ACC_TYPE, WEIGHT_TYPE, INPUT_VEC_TYPE,
                   ELEMENT_VEC_TYPE, ACC_VEC_TYPE, CacheDataT,
                   SZ_ACCUM, MASK>
        <<<gridSize, blockSize, 0, stream>>>(batchSz, table, indices, offsets, weights,
                                             num_hot, cache, rowSizeInElements,
                                             reinterpret_cast<typename QuantizationHelper<ELEMENT_TYPE_ID>::ParamType*>(output));
}

// extern template declarations of nve_fac_launch for every dispatched combination, generated at
// build time (see cmake/gen_find_and_combine_instantiations.py). Including this before the launcher
// templates below suppresses implicit instantiation of the launch wrappers (and their kernels) in
// every TU; the definitions are compiled once, split across the generated inst_*.cu files linked
// into nve-common.
//
// The generated inst_*.cu TUs define NVE_FAC_SKIP_EXTERN_DECLS before including this header: they
// *provide* the definitions and never call the launcher, so they don't need (and shouldn't pay the
// parse cost of) the ~tens-of-thousands of extern declarations. Only TUs that *use* the kernel
// (gpu_table.cu, tests) include them to suppress local instantiation.
#ifndef NVE_FAC_SKIP_EXTERN_DECLS
#include "find_and_combine_kernel_extern.inc"
#endif

inline uint32_t get_mask(SparseType_t hot_type, PoolingType_t pooling_type, bool load_indices) {
    uint32_t mask = 0;
    switch (hot_type) {
      case SparseType_t::Fixed:
        mask |= NVE_FAC_MASK_FIXED_HOTNESS;
        break;
      case SparseType_t::CSR:
        break;
      default:
        NVE_THROW_("Unsupported sparse type ", static_cast<uint32_t>(hot_type));
    }
    switch (pooling_type) {
      case PoolingType_t::Sum:
        mask |= NVE_FAC_MASK_SUM_POOLING;
        break;
      case PoolingType_t::Mean:
        break;
      case PoolingType_t::WeightedSum:
        mask |= (NVE_FAC_MASK_SUM_POOLING | NVE_FAC_MASK_IS_WEIGHTED);
        break;
      case PoolingType_t::WeightedMean:
        mask |= NVE_FAC_MASK_IS_WEIGHTED;
        break;
      default:
        NVE_THROW_("Unsupported pooling type ", static_cast<uint32_t>(pooling_type));
    }
    if (load_indices) {
        mask |= NVE_FAC_MASK_LOAD_INDICES;
    }
    return mask;
}

template<DataType_t ELEMENT_TYPE_ID, typename INDEX_TYPE, typename ACC_TYPE, typename WEIGHT_TYPE, typename CacheDataT>
void callFindAndCombineKernelTypesResolved(const uint32_t batchSz,
                             const int8_t* __restrict__ table,
                             const INDEX_TYPE* __restrict__ indices,
                             const INDEX_TYPE* __restrict__ offsets,
                             const WEIGHT_TYPE* __restrict__ weights,
                             int32_t num_hot,  
                             CacheDataT cache,
                             int32_t rowSizeInElements,
                             SparseType_t hot_type,
                             PoolingType_t pooling_type,
                             bool load_indices,
                             void* output,
                             cudaStream_t stream)
{
    uint32_t mask = get_mask(hot_type, pooling_type, load_indices);
    constexpr int32_t SUBWARP_WIDTH = 32;

    
#define NVE_FAC_LAUNCH(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, ELEMENT_VEC_T, ACC_VEC_T, MASK) \
    nve_fac_launch<ELEMENT_TYPE_ID, INDEX_TYPE, ACC_TYPE, WEIGHT_TYPE, INPUT_VEC_TYPE, \
                   ELEMENT_VEC_T, ACC_VEC_T, CacheDataT, \
                   (SZ_ACC_RUNTIME), MASK>( \
        batchSz, table, indices, offsets, weights, num_hot, cache, rowSizeInElements, output, stream)
    
    // Each thread holds SZ_ACCUM vector accumulators; with SUBWARP_WIDTH (=32) threads per row this
    // covers SUBWARP_WIDTH * SZ_ACCUM * vecSizeInElements row elements per pass. The kernel loops
    // over the row in chunks of that size, so any row length is supported regardless of SZ_ACCUM.
    // We pick the smallest SZ_ACCUM that fits the row in a single pass, clamped to 4, so small rows
    // don't waste registers while large rows fall back to multiple passes. The clamp (4) must match
    // the SZ_ACCUM range instantiated by cmake/gen_find_and_combine_instantiations.py.
#define NVE_FAC_LAUNCH_MASK(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, ELEMENT_VEC_T, ACC_VEC_T)                                                                            \
    do {                                                                                                                                                          \
        switch (mask) {                                                                                                                                           \
            case 0b0000: NVE_FAC_LAUNCH(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, ELEMENT_VEC_T, ACC_VEC_T, 0b0000); break;                                            \
            case 0b0001: NVE_FAC_LAUNCH(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, ELEMENT_VEC_T, ACC_VEC_T, 0b0001); break;                                            \
            case 0b0010: NVE_FAC_LAUNCH(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, ELEMENT_VEC_T, ACC_VEC_T, 0b0010); break;                                            \
            case 0b0011: NVE_FAC_LAUNCH(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, ELEMENT_VEC_T, ACC_VEC_T, 0b0011); break;                                            \
            case 0b0100: NVE_FAC_LAUNCH(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, ELEMENT_VEC_T, ACC_VEC_T, 0b0100); break;                                            \
            case 0b0101: NVE_FAC_LAUNCH(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, ELEMENT_VEC_T, ACC_VEC_T, 0b0101); break;                                            \
            case 0b0110: NVE_FAC_LAUNCH(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, ELEMENT_VEC_T, ACC_VEC_T, 0b0110); break;                                            \
            case 0b0111: NVE_FAC_LAUNCH(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, ELEMENT_VEC_T, ACC_VEC_T, 0b0111); break;                                            \
            case 0b1000: NVE_FAC_LAUNCH(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, ELEMENT_VEC_T, ACC_VEC_T, 0b1000); break;                                            \
            case 0b1001: NVE_FAC_LAUNCH(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, ELEMENT_VEC_T, ACC_VEC_T, 0b1001); break;                                            \
            case 0b1010: NVE_FAC_LAUNCH(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, ELEMENT_VEC_T, ACC_VEC_T, 0b1010); break;                                            \
            case 0b1011: NVE_FAC_LAUNCH(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, ELEMENT_VEC_T, ACC_VEC_T, 0b1011); break;                                            \
            case 0b1100: NVE_FAC_LAUNCH(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, ELEMENT_VEC_T, ACC_VEC_T, 0b1100); break;                                            \
            case 0b1101: NVE_FAC_LAUNCH(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, ELEMENT_VEC_T, ACC_VEC_T, 0b1101); break;                                            \
            case 0b1110: NVE_FAC_LAUNCH(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, ELEMENT_VEC_T, ACC_VEC_T, 0b1110); break;                                            \
            case 0b1111: NVE_FAC_LAUNCH(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, ELEMENT_VEC_T, ACC_VEC_T, 0b1111); break;                                            \
            default:                                                                                                                                               \
                NVE_THROW_("Unsupported mask ", mask);                                                                                                             \
        }                                                                                                                                                         \
    } while (0)

#define NVE_FAC_DISPATCH(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, ELEMENT_VEC_T, ACC_VEC_T)               \
    do {                                                                                        \
        switch (SZ_ACC_RUNTIME) {                                                                \
          case 1: NVE_FAC_LAUNCH_MASK(1, INPUT_VEC_TYPE, ELEMENT_VEC_T, ACC_VEC_T); break;       \
          case 2: NVE_FAC_LAUNCH_MASK(2, INPUT_VEC_TYPE, ELEMENT_VEC_T, ACC_VEC_T); break;       \
          case 3: NVE_FAC_LAUNCH_MASK(3, INPUT_VEC_TYPE, ELEMENT_VEC_T, ACC_VEC_T); break;       \
          case 4: NVE_FAC_LAUNCH_MASK(4, INPUT_VEC_TYPE, ELEMENT_VEC_T, ACC_VEC_T); break;       \
          default: NVE_THROW_("Unsupported kernel dimensions ", SZ_ACC_RUNTIME);                 \
        }                                                                                       \
    } while (0)
    
    constexpr int32_t MAX_SZ_ACCUM = 4;
    if (get_row_size_in_bytes<ELEMENT_TYPE_ID>(rowSizeInElements) % sizeof(typename QuantizationHelper<ELEMENT_TYPE_ID>::Vec4) == 0 && rowSizeInElements % 4 == 0)
    {
        const uint32_t SZ_ACCUM = std::min<uint32_t>(DivRoundUp(rowSizeInElements, SUBWARP_WIDTH * 4), MAX_SZ_ACCUM);
        using INPUT_VEC_TYPE = typename QuantizationHelper<ELEMENT_TYPE_ID>::Vec4;
        using ELEMENT_VEC_TYPE = typename VecWidthHelper<typename QuantizationHelper<ELEMENT_TYPE_ID>::ParamType>::Vec4;
        using ACC_VEC_TYPE = typename VecWidthHelper<ACC_TYPE>::Vec4;
        NVE_FAC_DISPATCH(SZ_ACCUM, INPUT_VEC_TYPE, ELEMENT_VEC_TYPE, ACC_VEC_TYPE);
    }
    else if (get_row_size_in_bytes<ELEMENT_TYPE_ID>(rowSizeInElements) % sizeof(typename QuantizationHelper<ELEMENT_TYPE_ID>::Vec2) == 0 && rowSizeInElements % 2 == 0)
    {
      const uint32_t SZ_ACCUM = std::min<uint32_t>(DivRoundUp(rowSizeInElements, SUBWARP_WIDTH * 2), MAX_SZ_ACCUM);
      using INPUT_VEC_TYPE = typename QuantizationHelper<ELEMENT_TYPE_ID>::Vec2;
      using ELEMENT_VEC_TYPE = typename VecWidthHelper<typename QuantizationHelper<ELEMENT_TYPE_ID>::ParamType>::Vec2;
      using ACC_VEC_TYPE = typename VecWidthHelper<ACC_TYPE>::Vec2;
      NVE_FAC_DISPATCH(SZ_ACCUM, INPUT_VEC_TYPE, ELEMENT_VEC_TYPE, ACC_VEC_TYPE);
    }
    else
    {
      const uint32_t SZ_ACCUM = std::min<uint32_t>(DivRoundUp(rowSizeInElements, SUBWARP_WIDTH * 1), MAX_SZ_ACCUM);
      using INPUT_VEC_TYPE = typename QuantizationHelper<ELEMENT_TYPE_ID>::Vec1;
      using ELEMENT_VEC_TYPE = typename VecWidthHelper<typename QuantizationHelper<ELEMENT_TYPE_ID>::ParamType>::Vec1;
      using ACC_VEC_TYPE = typename VecWidthHelper<ACC_TYPE>::Vec1;
      NVE_FAC_DISPATCH(SZ_ACCUM, INPUT_VEC_TYPE, ELEMENT_VEC_TYPE, ACC_VEC_TYPE);
    }
#undef NVE_FAC_DISPATCH
#undef NVE_FAC_LAUNCH
#undef NVE_FAC_LAUNCH_MASK
    NVE_CHECK_(cudaGetLastError()); // Check kernel launch didn't generate an error
}

template<typename ACC_TYPE,typename WEIGHT_TYPE,typename INDEX_TYPE, typename CacheDataT>
void callFindAndCombineKernelAccWeightType(const uint32_t batchSz,
                              const int8_t* __restrict__ table,
                              const INDEX_TYPE* __restrict__ indices,
                              const INDEX_TYPE* __restrict__ offsets,
                              const WEIGHT_TYPE* __restrict__ weights,
                              int32_t num_hot,  
                              CacheDataT cache,
                              int32_t rowSizeInElements,
                              SparseType_t hot_type,
                              PoolingType_t pooling_type,
                              bool load_indices,
                              DataType_t element_type,
                              void* output,
                              cudaStream_t stream)
{
    switch (element_type) {
      case DataType_t::Float32:
        callFindAndCombineKernelTypesResolved<DataType_t::Float32, INDEX_TYPE, ACC_TYPE, WEIGHT_TYPE, CacheDataT>(batchSz, table, indices, offsets, weights, num_hot, cache, rowSizeInElements, hot_type, pooling_type, load_indices, output, stream);
        break;
      case DataType_t::Float16:
        callFindAndCombineKernelTypesResolved<DataType_t::Float16, INDEX_TYPE, ACC_TYPE, WEIGHT_TYPE, CacheDataT>(batchSz, table, indices, offsets, weights, num_hot, cache, rowSizeInElements, hot_type, pooling_type, load_indices, output, stream);
        break;
      case DataType_t::QUint8RowwiseF32:
        callFindAndCombineKernelTypesResolved<DataType_t::QUint8RowwiseF32, INDEX_TYPE, ACC_TYPE, WEIGHT_TYPE, CacheDataT>(batchSz, table, indices, offsets, weights, num_hot, cache, rowSizeInElements, hot_type, pooling_type, load_indices, output, stream);
        break;
      case DataType_t::QInt8RowwiseF32:
        callFindAndCombineKernelTypesResolved<DataType_t::QInt8RowwiseF32, INDEX_TYPE, ACC_TYPE, WEIGHT_TYPE, CacheDataT>(batchSz, table, indices, offsets, weights, num_hot, cache, rowSizeInElements, hot_type, pooling_type, load_indices, output, stream);
        break;
      case DataType_t::QUint8RowwiseF16:
        callFindAndCombineKernelTypesResolved<DataType_t::QUint8RowwiseF16, INDEX_TYPE, ACC_TYPE, WEIGHT_TYPE, CacheDataT>(batchSz, table, indices, offsets, weights, num_hot, cache, rowSizeInElements, hot_type, pooling_type, load_indices, output, stream);
        break;
      case DataType_t::QInt8RowwiseF16:
        callFindAndCombineKernelTypesResolved<DataType_t::QInt8RowwiseF16, INDEX_TYPE, ACC_TYPE, WEIGHT_TYPE, CacheDataT>(batchSz, table, indices, offsets, weights, num_hot, cache, rowSizeInElements, hot_type, pooling_type, load_indices, output, stream);
        break;
      default:
        NVE_THROW_("Unsupported element type ", element_type);
    }
}

template<typename INDEX_TYPE, typename CacheDataT>
void callFindAndCombineKernel(const uint32_t num_keys,
                              const uint32_t num_offsets,
                              const int8_t* __restrict__ table,
                              const INDEX_TYPE* __restrict__ indices,
                              const INDEX_TYPE* __restrict__ offsets,
                              const void* __restrict__ weights,
                              int32_t num_hot,  
                              CacheDataT cache,
                              int32_t rowSizeInElements,
                              SparseType_t hot_type,
                              PoolingType_t pooling_type,
                              bool load_indices,
                              DataType_t element_type,
                              DataType_t weight_type,
                              DataType_t acc_type,
                              DataType_t /*output_type*/,
                              void* output,
                              cudaStream_t stream)
{
    uint32_t batchSz = 0;
    switch (hot_type) {
      case SparseType_t::Fixed:
        NVE_CHECK_(num_keys % num_hot == 0);
        batchSz = num_keys / num_hot;
        break;
      case SparseType_t::CSR:
        NVE_CHECK_(num_offsets > 0 && offsets != nullptr);
        batchSz = num_offsets;
        break;
      default:
        NVE_THROW_("Unsupported sparse type ", static_cast<uint32_t>(hot_type));
    }
    switch (acc_type) {
      case DataType_t::Float32:
        switch (weight_type) {
          case DataType_t::Float32:
            callFindAndCombineKernelAccWeightType<float, float, INDEX_TYPE, CacheDataT>(batchSz, table, indices, offsets, static_cast<const float*>(weights), num_hot, cache, rowSizeInElements, hot_type, pooling_type, load_indices, element_type, output, stream);
            break;
          case DataType_t::Float16:
            callFindAndCombineKernelAccWeightType<float, __half, INDEX_TYPE, CacheDataT>(batchSz, table, indices, offsets, static_cast<const __half*>(weights), num_hot, cache, rowSizeInElements, hot_type, pooling_type, load_indices, element_type, output, stream);
            break;
          default:
            NVE_THROW_("Unsupported weight type ", weight_type);
            break;
        }
        break;
      case DataType_t::Float16:
        switch (weight_type) {
          case DataType_t::Float32:
            NVE_THROW_("Unsupported weight type precision < acc type precision ", weight_type, " < ", acc_type);
          case DataType_t::Float16:
            callFindAndCombineKernelAccWeightType<__half, __half, INDEX_TYPE, CacheDataT>(batchSz, table, indices, offsets, static_cast<const __half*>(weights), num_hot, cache, rowSizeInElements, hot_type, pooling_type, load_indices, element_type, output, stream);
            break;
          default:
            NVE_THROW_("Unsupported weight type ", weight_type);
        }
        break;
      default:
        NVE_THROW_("Unsupported acc type ", acc_type);
    }
}
