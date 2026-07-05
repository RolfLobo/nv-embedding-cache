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

#include <buffer_wrapper.hpp>
#include <embedding_layer.hpp>
#include <ecache/ec_no_cache.cuh>

#include "cuda_ops/find_and_pool.cuh"

namespace nve {

// Validate the sparse-layout fields of a non-Concatenate pooling request.
inline void validate_pool_bag_layout(
    const EmbeddingLayerBase::PoolingParams& pool_params, int64_t num_keys) {
  switch (pool_params.sparse_type) {
    case SparseType_t::Fixed: {
      const int64_t hotness = pool_params.fixed_hotness;
      NVE_CHECK_ARG_(hotness > 0 && hotness <= INT32_MAX, "Invalid fixed hotness");
      NVE_CHECK_ARG_(num_keys % hotness == 0, "Number of keys must divide evenly by fixed hotness");
      break;
    }
    case SparseType_t::CSR: {
      NVE_CHECK_ARG_(pool_params.csr_offsets != nullptr, "Invalid CSR offsets buffer");
      NVE_CHECK_ARG_(pool_params.num_csr_offsets >= 2,
                     "CSR pooling requires at least two offsets");
      const int64_t num_output_rows = pool_params.num_csr_offsets - 1;
      NVE_CHECK_ARG_(num_output_rows <= INT32_MAX,
                     "Number of CSR output rows exceeds the CUDA pooling launcher limit");
      break;
    }
    default:
      NVE_THROW_ARG_("Unsupported sparse type ", static_cast<uint32_t>(pool_params.sparse_type));
  }
}

// Run the CUDA pool/dequant kernels over one contiguous raw stored row per key. This is shared by
// the hierarchical layer's cross-tier gather and the GPU layer's quantized-table gather path.
// gather_stride must equal the stored row width: it doubles as the row layout the kernels
// dequantize from.
template <typename KeyType>
void pool_gathered_device(context_ptr_t& ctx,
                          const EmbeddingLayerBase::PoolingParams& pool_params,
                          DataType_t value_dtype, const int8_t* gathered_rows,
                          int64_t gather_stride, int64_t num_keys, void* output,
                          int64_t output_stride, cudaStream_t stream) {
  NVE_CHECK_(num_keys >= 0 && num_keys <= UINT32_MAX,
             "Number of keys exceeds the CUDA pooling launcher limit");
  NVE_CHECK_(gather_stride > 0 && gather_stride <= UINT32_MAX,
             "Gather stride exceeds the CUDA pooling launcher limit");
  // The pool/dequant kernels derive the embedding width from the output stride, so output padding
  // is not supported here (unlike the CPU pooling path): the stride must decode to a value count
  // that fits within one stored row.
  validate_pool_output_stride(value_dtype, gather_stride, output_stride);

  const DataType_t output_dtype = pool_params.output_type;
  const bool concatenate = (pool_params.pooling_type == PoolingType_t::Concatenate);
  const DataType_t weight_dtype =
      (!concatenate && pool_params.weights) ? pool_params.weight_type : output_dtype;

  // Offsets (CSR) and hotness (Fixed) drive bag boundaries; Concatenate ignores both.
  const KeyType* offsets_dev = nullptr;
  std::shared_ptr<BufferWrapper<const KeyType>> offsets_bw;
  int64_t hotness = 0;
  uint32_t num_offsets = 0;
  int64_t num_output_rows = num_keys;
  if (!concatenate) {
    validate_pool_bag_layout(pool_params, num_keys);
    // CSR needs the offsets on device; Fixed only needs the (already validated) hotness scalar.
    if (pool_params.sparse_type == SparseType_t::Fixed) {
      hotness = pool_params.fixed_hotness;
      num_output_rows = num_keys / hotness;
    } else {
      num_output_rows = pool_params.num_csr_offsets - 1;
      num_offsets = static_cast<uint32_t>(num_output_rows);
      const auto offsets_buffer_size =
          static_cast<size_t>(pool_params.num_csr_offsets) * sizeof(KeyType);
      offsets_bw = std::make_shared<BufferWrapper<const KeyType>>(
          ctx, "offsets", static_cast<const KeyType*>(pool_params.csr_offsets),
          offsets_buffer_size);
      offsets_dev = offsets_bw->access_buffer(cudaMemoryTypeDevice, true /*copy_content*/, stream);
    }
  }

  const void* weights_dev = nullptr;
  std::shared_ptr<BufferWrapper<const void>> weights_bw;
  if (!concatenate && pool_params.weights) {
    const auto weights_buffer_size =
        static_cast<size_t>(num_keys) * static_cast<size_t>(dtype_size(pool_params.weight_type));
    weights_bw = std::make_shared<BufferWrapper<const void>>(
        ctx, "weights", pool_params.weights, weights_buffer_size);
    weights_dev = weights_bw->access_buffer(cudaMemoryTypeDevice, true /*copy_content*/, stream);
  }

  const auto output_buffer_size =
      static_cast<size_t>(num_output_rows) * static_cast<size_t>(output_stride);
  auto output_bw =
      std::make_shared<BufferWrapper<void>>(ctx, "pool_output", output, output_buffer_size);
  void* output_dev =
      output_bw->access_buffer(cudaMemoryTypeDevice, false /*copy_content*/, stream);

  typename ECNoCache<KeyType>::CacheData cache{};
  cache.row_size_in_bytes = static_cast<uint32_t>(gather_stride);

  find_and_pool<KeyType, typename ECNoCache<KeyType>::CacheData>(
      static_cast<uint32_t>(num_keys), nullptr /*keys*/, gathered_rows, cache,
      false /*load_indices*/, concatenate ? SparseType_t::Fixed : pool_params.sparse_type,
      pool_params.pooling_type, offsets_dev, num_offsets, static_cast<int32_t>(hotness), weights_dev,
      value_dtype, output_dtype, weight_dtype, DataType_t::Float32 /*acc_dtype*/, output_stride,
      output_dev, stream);

  if (output_dev != output) {
    NVE_CHECK_(
        cudaMemcpyAsync(output, output_dev, output_buffer_size, cudaMemcpyDefault, stream));
  }
}

}  // namespace nve
