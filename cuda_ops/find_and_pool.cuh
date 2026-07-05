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

#include "cuda_ops/find_and_dequant.cuh"
#include "cuda_ops/find_and_combine_kernel.cuh"
#include "cuda_ops/kernels_common.cuh"

namespace nve {

// Validate the runtime data-type combination before dispatching a CUDA kernel. The output type is
// fixed by QuantizationHelper<value_dtype>::ParamType: non-quantized values preserve their type,
// while rowwise-quantized values are dequantized to the precision encoded in the dtype suffix.
inline void validate_find_and_pool_data_types(PoolingType_t pooling_type, DataType_t value_dtype,
                                              DataType_t output_dtype,
                                              DataType_t weight_dtype,
                                              DataType_t acc_dtype) {
  DataType_t required_output_dtype;
  switch (value_dtype) {
    case DataType_t::Float32:
    case DataType_t::QInt8RowwiseF32:
    case DataType_t::QUint8RowwiseF32:
      required_output_dtype = DataType_t::Float32;
      break;
    case DataType_t::Float16:
    case DataType_t::QInt8RowwiseF16:
    case DataType_t::QUint8RowwiseF16:
      required_output_dtype = DataType_t::Float16;
      break;
    default:
      NVE_THROW_("Unsupported find_and_pool value type ", value_dtype);
  }

  NVE_CHECK_ARG_(output_dtype == required_output_dtype,
                 "Unsupported find_and_pool value/output type combination: value type ", value_dtype,
                 " requires output type ", required_output_dtype, ", got ", output_dtype);

  // Concatenate does not dispatch the combine kernel, so weight and accumulator types are ignored.
  if (pooling_type == PoolingType_t::Concatenate) {
    return;
  }

  NVE_CHECK_(acc_dtype == DataType_t::Float32 || acc_dtype == DataType_t::Float16,
             "Unsupported find_and_pool accumulator type ", acc_dtype);
  NVE_CHECK_ARG_(weight_dtype == DataType_t::Float32 || weight_dtype == DataType_t::Float16,
                 "Unsupported find_and_pool weight type ", weight_dtype);
  NVE_CHECK_(acc_dtype != DataType_t::Float16 || weight_dtype == DataType_t::Float16,
             "Unsupported find_and_pool accumulator/weight type combination: Float16 accumulator "
             "requires Float16 weights");
}

// Validate the output row stride the CUDA pool/dequant kernels will use against the stored row
// layout. The kernels derive the per-row value count from the output stride (they can neither pad
// nor truncate the output layout), so the stride must decode to a whole number of dequantized
// elements whose stored footprint — value bytes plus any trailing scale/offset metadata — fits
// within one stored row. Stored rows themselves may carry trailing alignment padding, so the
// stride is bounded by the row rather than required to match it exactly. The CPU pooling path
// (pool_gathered_host) takes the width from the stored row instead and accepts a padded output
// stride.
inline void validate_pool_output_stride(DataType_t value_dtype, int64_t row_size_in_bytes,
                                        int64_t output_stride) {
  const int64_t out_elem_bytes = static_cast<int64_t>(value_size_in_bytes(value_dtype));
  NVE_CHECK_ARG_(output_stride > 0 && output_stride % out_elem_bytes == 0,
                 "Pooled/dequantized output stride must be a whole number of output elements");
  const int64_t value_count = output_stride / out_elem_bytes;
  const bool quant = is_quant_rowwise(value_dtype);
  const int64_t meta_bytes = quant ? quant_rowwise_meta_bytes(value_dtype) : 0;
  const int64_t stored_elem_bytes = quant ? dtype_size(value_dtype) : out_elem_bytes;
  NVE_CHECK_ARG_(value_count * stored_elem_bytes + meta_bytes <= row_size_in_bytes,
                 "Pooled/dequantized output stride implies more value elements than the stored row "
                 "holds: stride ", output_stride, " bytes decodes to ", value_count,
                 " elements, stored row is ", row_size_in_bytes, " bytes");
}

// Single entry point that gathers one row per key (resolved through the AddressFunctor for
// CacheDataT over `table`) and writes the pooled / dequantized result. It owns the
// Concatenate-vs-pooling routing:
//   * PoolingType_t::Concatenate -> find_and_dequant: one dequantized output row per key.
//   * any other pooling type     -> find_and_combine: one output row per bag.
//
// CacheDataT selects how each lane's row address is resolved:
//   * a GPU cache's CacheData (with table = the UVM fallback) for a cached table lookup, or
//   * ECNoCache<IndexT>::CacheData (with table = a contiguous gathered buffer and
//     load_indices = false) to pool/dequant an already-gathered buffer in place.
//
// value_dtype is the stored (input / quantized) row layout; output_dtype is the dequantized output
// element type; acc_dtype is the accumulator type (always Float32). offsets/num_offsets are used for
// SparseType_t::CSR; fixed_hotness for SparseType_t::Fixed. weights may be null unless the pooling
// type is weighted. Concatenate ignores all sparse-layout and weight arguments.
template <typename IndexT, typename CacheDataT>
void find_and_pool(uint32_t num_keys, const IndexT* keys, const int8_t* table, CacheDataT cache,
                   bool load_indices, SparseType_t sparse_type, PoolingType_t pooling_type,
                   const IndexT* offsets, uint32_t num_offsets, int32_t fixed_hotness,
                   const void* weights, DataType_t value_dtype, DataType_t output_dtype,
                   DataType_t weight_dtype, DataType_t acc_dtype, int64_t value_stride, void* output,
                   cudaStream_t stream) {
  validate_find_and_pool_data_types(pooling_type, value_dtype, output_dtype, weight_dtype,
                                    acc_dtype);
  if (num_keys == 0) {
    return;
  }
  if (pooling_type == PoolingType_t::Concatenate) {
    // No pooling: one dequantized row per key. The number of value elements per row is derived from
    // the output stride and the stored (input) element size, matching GpuTable::find_and_dequant.
    const uint32_t row_size_in_elements =
        static_cast<uint32_t>(value_stride / value_size_in_bytes(value_dtype));
    const cudaError_t status = call_find_and_dequant<IndexT, CacheDataT>(
        keys, static_cast<size_t>(num_keys), static_cast<int8_t*>(output), table, value_dtype,
        row_size_in_elements, cache, stream, static_cast<size_t>(value_stride), load_indices,
        0 /*curr_table*/);
    NVE_CHECK_(status, "find_and_dequant failed");
  } else {
    const int32_t num_elements =
        static_cast<int32_t>(value_stride / value_size_in_bytes(output_dtype));
    callFindAndCombineKernel<IndexT, CacheDataT>(
        num_keys, num_offsets, table, keys, offsets, weights, fixed_hotness, cache, num_elements,
        sparse_type, pooling_type, load_indices, value_dtype, weight_dtype, acc_dtype, output_dtype,
        output, stream);
  }
}

}  // namespace nve
