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

#include <linear_embedding_layer.hpp>
#include <layer_utils.hpp>
#include <limits>
#include <default_allocator.hpp>
#include <gpu_table.hpp>
#include <insert_heuristic.hpp>
#include "cuda_ops/cuda_common.h"
#include "cuda_ops/sanitize_keys.cuh"
#include <buffer_wrapper.hpp>

namespace nve {

template <typename KeyType>
context_ptr_t LinearUVMEmbeddingLayer<KeyType>::create_execution_context(
  cudaStream_t lookup_stream,
  cudaStream_t modify_stream,
  thread_pool_ptr_t thread_pool,
  allocator_ptr_t allocator) {
  allocator_ptr_t actual_allocator = allocator ? allocator : allocator_;
  auto gpu_ctx = gpu_table_ ? gpu_table_->create_execution_context(lookup_stream, modify_stream, thread_pool, actual_allocator) : nullptr;
  std::vector<context_ptr_t> contexts{std::move(gpu_ctx)};
  return std::make_shared<LayerExecutionContext>(
    lookup_stream,
    modify_stream,
    thread_pool,
    actual_allocator,
    contexts);
}

template <typename KeyType>
LinearUVMEmbeddingLayer<KeyType>::LinearUVMEmbeddingLayer(const Config& cfg, gpu_table_ptr_t gpu_table, allocator_ptr_t allocator)
  : config_(cfg), gpu_table_(std::move(gpu_table)) {
    allocator_ = allocator ? allocator : GetDefaultAllocator();
    NVE_CHECK_(allocator_ != nullptr, "Failed to get default allocator");
    NVE_CHECK_(gpu_table_ != nullptr, "Invalid GPU table");
    NVE_CHECK_(gpu_table_->config().uvm_table != nullptr, "GPU Table must be backed by a UVM buffer");
    NVE_CHECK_(gpu_table_->config().uvm_num_rows > 0, "uvm_num_rows must be >0");
    NVE_CHECK_(static_cast<uint64_t>(gpu_table_->config().uvm_num_rows) <=
                   static_cast<uint64_t>(std::numeric_limits<KeyType>::max()),
               "Number of UVM rows cannot be represented by the layer key type");
    NVE_CHECK_(config_.default_row_index < gpu_table_->config().uvm_num_rows,
               "Default row index must be less than uvm_num_rows");

    auto heuristic = config_.insert_heuristic;
    if (!heuristic) {
      heuristic = std::make_shared<DefaultInsertHeuristic>(std::vector<float>{DefaultInsertHeuristic::DEFAULT_THRESHOLD});
    }
    KeyType invalid_key = static_cast<KeyType>(gpu_table_->get_invalid_key());
    // Callback used on the pooling path, where no reusable per-key row data is produced: promote the
    // collected keys by reading their rows directly from the backing UVM table (see insert_from_uvm).
    AutoInsertHandler::uvm_insert_fn_t uvm_insert_fn =
      [gpu_tbl = gpu_table_](context_ptr_t& ctx, int64_t n, std::shared_ptr<BufferWrapper<const void>> keys_bw) {
        gpu_tbl->insert_from_uvm(ctx, n, std::move(keys_bw));
      };
    auto_insert_handler_ = std::make_shared<AutoInsertHandler>(
      heuristic,
      gpu_table_,
      0 /* table_id */,
      allocator_,
      config_.min_insert_freq_gpu,
      config_.min_insert_size_gpu,
      sizeof(KeyType),
      gpu_table_->get_device_id(),
      &invalid_key,
      std::move(uvm_insert_fn)
    );
}

template <typename KeyType>
LinearUVMEmbeddingLayer<KeyType>::~LinearUVMEmbeddingLayer() {
}

template <typename KeyType>
void LinearUVMEmbeddingLayer<KeyType>::lookup(context_ptr_t& ctx, const int64_t num_keys, const void* keys, void* output,
                    const int64_t output_stride, bitmask64_t* hitmask,
                    const PoolingParams* pool_params, float* hitrates) {
  NVE_NVTX_SCOPED_FUNCTION_COL1_();
  ScopedDevice scope_device(gpu_table_->config().device_id);
  NVE_CHECK_ARG_(num_keys >= 0, "Invalid num_keys");
  NVE_CHECK_ARG_(keys != nullptr, "Invalid keys buffer");
  NVE_CHECK_ARG_(output != nullptr, "Invalid output buffer");
  NVE_CHECK_ARG_(hitmask == nullptr, "Hitmask is not supported for the UVM layer");

  auto layer_ctx = std::dynamic_pointer_cast<LayerExecutionContext>(ctx);
  NVE_CHECK_(layer_ctx != nullptr, "Invalid layer context");
  const cudaStream_t lookup_stream = layer_ctx->get_lookup_stream();
  if (pool_params) {
    validate_pool_params(*pool_params);
  }
  const bool raw_concatenate = is_pooling_raw_concat(pool_params, gpu_table_->config().value_dtype);
  if (raw_concatenate) {
    NVE_CHECK_ARG_(output_stride >= gpu_table_->config().row_size_in_bytes,
               "Raw Concatenate output stride is smaller than the stored row width");
  }
  const int64_t output_rows = get_lookup_output_rows(num_keys, pool_params);
  const auto output_buffer_size =
      static_cast<size_t>(output_rows) * static_cast<size_t>(output_stride);
  const auto key_buffer_size = sizeof(KeyType) * num_keys;

  auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys, key_buffer_size);
  auto output_bw = std::make_shared<BufferWrapper<void>>(ctx, "output", output, output_buffer_size);

  // Keys outside the UVM table are replaced with the default row, so both the lookup below and the
  // auto-insert of these keys stay in bounds (no-op when no default row is configured).
  keys_bw = sanitize_lookup_keys<KeyType>(ctx, std::move(keys_bw), num_keys,
                                          gpu_table_->config().uvm_num_rows,
                                          config_.default_row_index, lookup_stream);

  const bool collect_misses = (hitrates || !std::dynamic_pointer_cast<NeverInsertHeuristic>(config_.insert_heuristic));
  if (collect_misses) {
    NVE_CHECK_(gpu_table_->config().count_misses, "GPU table was not configured to report hit counts");
    gpu_table_->reset_lookup_counter(layer_ctx->table_contexts_.at(0));
  }

  // Lookup
  if (pool_params && !raw_concatenate) {
    // find_and_pool routes internally: Concatenate (one dequantized row per key) goes to
    // find_and_dequant; any other pooling type combines bags via find_and_combine. Same-type
    // Concatenate bypasses this block and uses the raw find path below.
    const bool concatenate = pool_params->pooling_type == PoolingType_t::Concatenate;
    SparseType_t sparse_type = SparseType_t::Fixed;
    int64_t num_offsets = 0;
    int64_t hotness = 0;
    std::shared_ptr<BufferWrapper<const KeyType>> offsets_bw;
    std::shared_ptr<BufferWrapper<const void>> weights_bw;

    const DataType_t output_dtype = pool_params->output_type;
    DataType_t weight_dtype = output_dtype;
    if (!concatenate) {
      sparse_type = pool_params->sparse_type;
      if (sparse_type == SparseType_t::Fixed) {
        hotness = pool_params->fixed_hotness;
      } else {
        num_offsets = pool_params->num_csr_offsets - 1;
        const auto offsets_buffer_size = pool_params->num_csr_offsets * sizeof(KeyType);
        offsets_bw = std::make_shared<BufferWrapper<const KeyType>>(
            ctx, "offsets", static_cast<const KeyType*>(pool_params->csr_offsets),
            offsets_buffer_size);
      }

      if (pool_params->weights) {
        const auto weights_buffer_size =
            static_cast<size_t>(num_keys) * static_cast<size_t>(dtype_size(pool_params->weight_type));
        weights_bw = std::make_shared<BufferWrapper<const void>>(
            ctx, "weights", pool_params->weights, weights_buffer_size);
        weight_dtype = pool_params->weight_type;
      }
    }

    gpu_table_->find_and_pool(
      layer_ctx->table_contexts_.at(0),
      num_keys,
      keys_bw,
      sparse_type,
      num_offsets,
      std::move(offsets_bw),
      hotness,
      pool_params->pooling_type,
      std::move(weights_bw),
      output_dtype, weight_dtype,
      output_stride, output_bw);
  } else {
    // Plain lookup and same-type Concatenate both return complete stored rows, including any
    // rowwise-quantization metadata.
    gpu_table_->find(layer_ctx->table_contexts_.at(0), num_keys, keys_bw, nullptr, output_stride, output_bw, nullptr);
  }

  auto d_output = output_bw->get_buffer(cudaMemoryTypeDevice);
  if (output != d_output) {
    NVE_CHECK_(cudaMemcpyAsync(output, d_output, output_buffer_size, cudaMemcpyDefault, lookup_stream));
  }

  // Handle hitrates
  if (collect_misses) {
    int64_t misses = 0;
    gpu_table_->get_lookup_counter(layer_ctx->table_contexts_.at(0), &misses);
    NVE_CHECK_(cudaStreamSynchronize(lookup_stream)); // Need to wait for lookup to end before checking misses
    const float hitrate = 1.f - (static_cast<float>(misses) / static_cast<float>(num_keys));
    if (hitrates) {
      *hitrates = hitrate;
    }
    // Handle automatic inserts.
    // Real pooling and Concatenate-mode dequant produce output that cannot be reused as raw per-key
    // cache rows, so those paths collect keys only and promote from UVM. Plain lookup and same-type
    // Concatenate both produce reusable raw rows.
    if (auto_insert_handler_) {
      auto_insert_handler_->auto_insert(std::move(layer_ctx), keys_bw, output_bw, hitrate, num_keys, output_stride,
                                        nullptr /*hitmask_bw*/,
                                        pool_params != nullptr && !raw_concatenate /*insert_from_uvm*/);
    }
  }
}

template <typename KeyType>
void LinearUVMEmbeddingLayer<KeyType>::insert(context_ptr_t& ctx, const int64_t num_keys, const void* keys,
                    const int64_t value_stride, const int64_t value_size, const void* values,
                    const int64_t table_id) {
  NVE_NVTX_SCOPED_FUNCTION_COL2_();
  NVE_CHECK_(table_id < get_num_tables(), "Invalid table_id");

  ScopedDevice scope_device(gpu_table_->config().device_id);
  NVE_CHECK_(num_keys >= 0, "Invalid num_keys");
  NVE_CHECK_(keys != nullptr, "Invalid keys buffer");
  NVE_CHECK_(values != nullptr, "Invalid values buffer");
  NVE_CHECK_(value_size == gpu_table_->config().row_size_in_bytes, "Invalid value size");

  auto layer_ctx = std::dynamic_pointer_cast<LayerExecutionContext>(ctx);
  NVE_CHECK_(layer_ctx != nullptr, "Invalid layer context");

  const auto keys_buffer_size = sizeof(KeyType) * num_keys;
  const auto values_buffer_size = num_keys * value_stride;
 
  auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys, keys_buffer_size);
  auto values_bw = std::make_shared<BufferWrapper<const void>>(ctx, "values", values, values_buffer_size);

  gpu_table_->insert(layer_ctx->table_contexts_.at(0), num_keys, std::move(keys_bw), value_stride, value_size, std::move(values_bw));
}

template <typename KeyType>
void LinearUVMEmbeddingLayer<KeyType>::update(context_ptr_t& ctx, const int64_t num_keys, const void* keys,
                    const int64_t value_stride, const int64_t value_size,
                    const void* values, const int64_t table_id) {
  NVE_NVTX_SCOPED_FUNCTION_COL3_();
  ScopedDevice scope_device(gpu_table_->config().device_id);
  NVE_CHECK_(num_keys >= 0, "Invalid num_keys");
  NVE_CHECK_(keys != nullptr, "Invalid keys buffer");
  NVE_CHECK_(values != nullptr, "Invalid values buffer");
  NVE_CHECK_(value_size == gpu_table_->config().row_size_in_bytes, "Invalid value size");
  NVE_CHECK_(table_id < get_num_tables(), "Invalid table_id");

  auto layer_ctx = std::dynamic_pointer_cast<LayerExecutionContext>(ctx);
  NVE_CHECK_(layer_ctx != nullptr, "Invalid layer context");

  const auto keys_buffer_size = sizeof(KeyType) * num_keys;
  const auto values_buffer_size = num_keys * value_stride;
 
  auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys, keys_buffer_size);
  auto values_bw = std::make_shared<BufferWrapper<const void>>(ctx, "values", values, values_buffer_size);

  ScopedModifyLock modify_lock{auto_insert_handler_};
  gpu_table_->update(layer_ctx->table_contexts_.at(0), num_keys, std::move(keys_bw), value_stride, value_size, std::move(values_bw));
}

template <typename KeyType>
void LinearUVMEmbeddingLayer<KeyType>::accumulate(context_ptr_t& ctx, const int64_t num_keys, const void* keys,
                        const int64_t value_stride, const int64_t value_size, const void* values,
                        DataType_t value_type, const int64_t table_id) {
  NVE_NVTX_SCOPED_FUNCTION_COL4_();
  ScopedDevice scope_device(gpu_table_->config().device_id);
  NVE_CHECK_(num_keys >= 0, "Invalid num_keys");
  NVE_CHECK_(keys != nullptr, "Invalid keys buffer");
  NVE_CHECK_(values != nullptr, "Invalid values buffer");
  NVE_CHECK_(value_size == gpu_table_->config().row_size_in_bytes, "Invalid value size");
  NVE_CHECK_(table_id < get_num_tables(), "Invalid table_id");

  auto layer_ctx = std::dynamic_pointer_cast<LayerExecutionContext>(ctx);
  NVE_CHECK_(layer_ctx != nullptr, "Invalid layer context");

  const auto keys_buffer_size = sizeof(KeyType) * num_keys;
  const auto values_buffer_size = num_keys * value_stride;
 
  auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys, keys_buffer_size);
  auto values_bw = std::make_shared<BufferWrapper<const void>>(ctx, "values", values, values_buffer_size);

  ScopedModifyLock modify_lock{auto_insert_handler_};
  gpu_table_->update_accumulate(layer_ctx->table_contexts_.at(0), num_keys, std::move(keys_bw), value_stride, value_size, std::move(values_bw), value_type);
}

template <typename KeyType>
void LinearUVMEmbeddingLayer<KeyType>::clear(context_ptr_t& ctx) { 
  NVE_NVTX_SCOPED_FUNCTION_COL5_();
  ScopedDevice scope_device(gpu_table_->config().device_id);
  auto layer_ctx = std::dynamic_pointer_cast<LayerExecutionContext>(ctx);
  NVE_CHECK_(layer_ctx != nullptr, "Invalid layer context");

  ScopedModifyLock modify_lock{auto_insert_handler_};
  gpu_table_->clear(layer_ctx->table_contexts_.at(0));
}

template <typename KeyType>
void LinearUVMEmbeddingLayer<KeyType>::erase(context_ptr_t& ctx, const int64_t num_keys, const void* keys,
                    const int64_t table_id) {
  NVE_NVTX_SCOPED_FUNCTION_COL6_();
  NVE_CHECK_(table_id < get_num_tables(), "Invalid table_id");
  ScopedDevice scope_device(gpu_table_->config().device_id);

  NVE_CHECK_(num_keys >= 0, "Invalid num_keys");
  NVE_CHECK_(keys != nullptr, "Invalid keys buffer");

  auto layer_ctx = std::dynamic_pointer_cast<LayerExecutionContext>(ctx);
  NVE_CHECK_(layer_ctx != nullptr, "Invalid layer context");

  auto keys_buffer_size = sizeof(KeyType) * num_keys;
  auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys, keys_buffer_size);

  ScopedModifyLock modify_lock{auto_insert_handler_};
  gpu_table_->erase(layer_ctx->table_contexts_.at(0), num_keys, std::move(keys_bw));
}

template class LinearUVMEmbeddingLayer<int32_t>;
template class LinearUVMEmbeddingLayer<int64_t>;

}  // namespace nve
