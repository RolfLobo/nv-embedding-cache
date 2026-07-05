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

#include <unordered_set>
#include <vector>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <memory>
#include <cuda_support.hpp>
#include <thread_pool.hpp>
#include <execution_context.hpp>
#include <ecache/embed_cache.h>
#include <bit_ops.hpp>
#include <chrono>
#include "include/embedding_layer.hpp"  // for EmbeddingLayerBase::PoolingParams

namespace nve {

class DefaultECEvent;

template <>
constexpr bool is_success(const ECError& result) noexcept {
  return result == ECERROR_SUCCESS;
}

template <>
class RuntimeError<ECError> : public Exception {
 public:
  using base_type = Exception;

  RuntimeError() = delete;

  inline RuntimeError(const char file[], const int line, const char expr[],
                      const ECError& error, const std::string& hint) noexcept
      : base_type(file, line, expr, hint), error_{error} {}

  inline RuntimeError(const RuntimeError& that) noexcept : base_type(that), error_{that.error_} {}

  inline RuntimeError& operator=(const RuntimeError& that) noexcept {
    base_type::operator=(that);
    error_ = that.error_;
    return *this;
  }

  inline ECError error() const noexcept { return error_; }

  virtual std::string to_string() const override;

 private:
  ECError error_;
};

/**
 * Auxiliary class to keep track of all existing execution contexts so modify ops can sync with them.
 */
class ContextRegistry {
public:
  ContextRegistry() = default;
  NVE_PREVENT_COPY_AND_MOVE_(ContextRegistry);
  void add_context(const ExecutionContext* ctx);

  void remove_context(const ExecutionContext* ctx);

  std::shared_ptr<DefaultECEvent> create_sync_event();

  bool empty() const;
private:
  mutable std::mutex mutex_;
  std::unordered_set<cudaStream_t> lookup_streams_;
  std::unordered_set<const ExecutionContext*> contexts_; // Holding raw pointers instead of shared_ptr to avoid circular dependency
  void update_streams();
};

/**
 * StreamCoordinator object will set the correct cuda event dependencies between cuda streams during construction and destruction.
 * In between c'tor and d'tor, queue kernels to queue_stream
 */
class StreamCoordinator {
public:
  NVE_PREVENT_COPY_AND_MOVE_(StreamCoordinator);
  StreamCoordinator(const cudaStream_t& caller_stream, const cudaStream_t& run_stream) :
    queue_stream(run_stream ? run_stream : caller_stream), caller_stream_(caller_stream)
  {
    create_stream_dependency(caller_stream_, queue_stream);
  }
  ~StreamCoordinator() {
    create_stream_dependency(queue_stream, caller_stream_);
  }

  /**
   * Create a cuda event dependency between src and dst.
   * Specifically, commands queued for dst after this call will wait for all commands queued for src before this call.
   * An ad-hoc cudaEvent will be created for this call (this will cause a small overhead, measured 0.21us on my machine)
   */
  static inline void create_stream_dependency(const cudaStream_t& src, const cudaStream_t& dst) {
    if (src != dst) {
      cudaEvent_t e;
      NVE_CHECK_(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
      NVE_CHECK_(cudaEventRecord(e, src));
      NVE_CHECK_(cudaStreamWaitEvent(dst, e));
      NVE_CHECK_(cudaEventDestroy(e));
    }
  }

  const cudaStream_t queue_stream;
private:
  const cudaStream_t caller_stream_;
};

class LayerExecutionContext: public ExecutionContext {
 public:
  NVE_PREVENT_COPY_AND_MOVE_(LayerExecutionContext);
  
  LayerExecutionContext(
    cudaStream_t lookup_stream,
    cudaStream_t modify_stream,
    thread_pool_ptr_t thread_pool,
    allocator_ptr_t allocator,
    std::vector<context_ptr_t> table_contexts)
    : ExecutionContext(lookup_stream, modify_stream, std::move(thread_pool), std::move(allocator)),
      table_contexts_(std::move(table_contexts)), parallel_task_res_(table_contexts_.size()) {}

  virtual ~LayerExecutionContext() { wait(); }

  virtual void wait() override {
    for (auto& res : parallel_task_res_) {
      if (res.valid()) {
        res.get(); // This needs to precede the stream sync since it queues work on the streams
      }
    }
    ExecutionContext::wait();
  }

  void submit_parallel_task(ThreadPool::task_type task, int64_t table_id) {
    auto& fut{parallel_task_res_.at(static_cast<size_t>(table_id))};

    if (fut.valid()) {
      if (fut.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
        NVE_LOG_PERF_("Waiting for parallel task"); // This shouldn't happen - just a precaution
      }
      fut.get(); // Also propagates any exception from the previous task before it is overwritten below.
    }
    fut = thread_pool_->submit(std::move(task));
  }

  // Public members since only this file can access them (the actual class isn't exposed)
  std::vector<context_ptr_t> table_contexts_;
  // For now only auto-insert can be offloaded and only one can exist at a given time for every table
  // todo: extend to multiple futures when needed.
  std::vector<ThreadPool::result_type> parallel_task_res_;
};

/**
 * Synchronization primitivaes to allow for try_lock w/o the wierd restrictions of c++;
 * mutex can fail on try lock w/o reason, while the wrapper class have same restriction as mutex while 
 * throwing exception when try_lock a locked object
 */
class TestableMutex {
public:
    TestableMutex() : flag(false) {}

    // check whether the lock is free if true, performs a lock
    bool try_lock() {
        std::unique_lock l(m);
        if (!flag)
        {
            flag = true;
            return true;
        }
        else
        {
          return false;
        }
    }

    bool is_locked() {
      std::unique_lock l(m);
      return flag;
    }

    void lock() {
        std::unique_lock l(m);
        cv.wait(l, [this]{ return !this->flag; });
        flag = true;
    }

    void unlock() {
        std::unique_lock l(m);
        flag = false;
        cv.notify_all();
    }
private:
    bool flag;
    std::mutex m;
    std::condition_variable cv;
};

class InsertHeuristic;
class Table;
template<typename T>
class BufferWrapper;

class AutoInsertHandler final {
public:
  // Callback that promotes keys into the table by reading their rows directly from a backing UVM table
  // (see GpuTable::insert_from_uvm). Supplied by layers whose table supports it (LinearUVM); empty
  // otherwise. Kept as a callback so this generic handler doesn't depend on the concrete table type.
  using uvm_insert_fn_t = std::function<void(context_ptr_t&, int64_t, std::shared_ptr<BufferWrapper<const void>>)>;

  AutoInsertHandler(
    std::shared_ptr<InsertHeuristic> heuristic,
    table_ptr_t table,
    const int64_t table_id,
    allocator_ptr_t allocator,
    const int64_t min_insert_wait,
    const int64_t min_insert_size,
    const int64_t key_size,
    const int32_t layer_gpu_device,
    const void* invalid_key_bytes,
    uvm_insert_fn_t uvm_insert_fn = {});

  ~AutoInsertHandler();

  // insert_from_uvm: when true, this lookup produced no reusable per-key row data (e.g. a pooling
  // lookup), so only keys are collected and the eventual insert promotes them via uvm_insert_fn_. The
  // mode is sticky for the duration of a collection - if a non-pooled collection is later joined by a
  // pooled lookup it switches to UVM mode and any partial data already collected is discarded.
  void auto_insert(
    std::shared_ptr<LayerExecutionContext> layer_ctx,
    std::shared_ptr<BufferWrapper<const void>>& keys_bw,
    std::shared_ptr<BufferWrapper<void>>& output_bw,
    const float hitrate,
    const int64_t num_keys,
    const int64_t output_stride,
    std::shared_ptr<BufferWrapper<max_bitmask_repr_t>> hitmask_bw = nullptr,
    const bool insert_from_uvm = false);
  void lock_modify();
  void unlock_modify();

private:
  TestableMutex insert_lock_; // lock to prevent parallel auto-insert handling and modify parallel to auto-insert
  std::shared_ptr<InsertHeuristic> heuristic_;
  table_ptr_t table_;
  const int64_t table_id_;
  allocator_ptr_t allocator_;
  const int64_t min_insert_wait_;
  const int64_t min_insert_size_;
  const int64_t key_size_;
  const int32_t layer_gpu_device_;
  std::vector<uint8_t> invalid_key_bytes_; // empty when no sentinel is configured (no rewriting)
  uvm_insert_fn_t uvm_insert_fn_; // promotes keys from the backing UVM table; empty when unsupported

  int64_t insert_freq_cnt_{0};
  int64_t collected_keys_{0};
  int64_t collected_output_stride_{0};
  bool collection_is_uvm_{false}; // sticky per-collection: insert the collected keys via uvm_insert_fn_

  // Handler keeps it own keys/data buffers for accumulation of small key sets
  std::unique_ptr<ResizeableBuffer> insert_keys_;
  std::unique_ptr<ResizeableBuffer> insert_data_;

  void collect_keys_and_data(
    std::shared_ptr<BufferWrapper<const void>>& keys_bw,
    std::shared_ptr<BufferWrapper<void>>& output_bw,
    std::shared_ptr<BufferWrapper<max_bitmask_repr_t>>& hitmask_bw,
    cudaStream_t lookup_stream,
    const int64_t num_keys);
  void launch_insert(
    std::shared_ptr<LayerExecutionContext> layer_ctx,
    const int64_t output_stride);
};

void validate_pool_params(const EmbeddingLayerBase::PoolingParams& pool_params);

/**
 * Return the number of rows produced by a layer lookup.
 *
 * Plain lookup and Concatenate produce one row per key. Fixed pooling produces
 * `num_keys / fixed_hotness` rows, while CSR pooling produces
 * `num_csr_offsets - 1` rows. Invalid or non-divisible layouts are rejected.
 */
int64_t get_lookup_output_rows(
    int64_t num_keys, const EmbeddingLayerBase::PoolingParams* pool_params);

/**
 * Pool/dequantize a host-resident gather buffer into a host-resident output buffer using the
 * CPU pooling kernels (cpu_ops/cpu_pooling.h). Shared by the host layer and by the hierarchical
 * layer's no-GPU-tier pooling path.
 *
 * `gather_host` holds `num_keys` raw stored rows at `gather_stride` (each `row_size` bytes,
 * possibly carrying quantized scale/offset metadata); `value_dtype` is the stored value type.
 * Derives the pooling metadata from `pool_params` (sparse layout, hotness/offsets, in/out dtype,
 * element width) and reduces into `output` at `output_stride`. Reads `csr_offsets` and
 * `weights` through `ctx`-wrapped buffers so device-resident inputs are handled, and wraps
 * `output` so a device/managed user buffer is written via a host scratch and copied back.
 *
 * The caller owns the gather buffer. Same-type Concatenate copies each full raw row, including
 * rowwise-quantization metadata, without conversion or reduction.
 *
 * @tparam KeyType key/offset element type (int32_t or int64_t).
 */
template <typename KeyType>
void pool_gathered_host(context_ptr_t& ctx,
                        const EmbeddingLayerBase::PoolingParams& pool_params,
                        DataType_t value_dtype,
                        const int8_t* gather_host, int64_t gather_stride,
                        int64_t row_size, int64_t num_keys,
                        void* output, int64_t output_stride);

// Check if the pooling can be resolved with only copying (equivalent to lookup)
bool is_pooling_raw_concat(const EmbeddingLayerBase::PoolingParams* pool_params, DataType_t value_dtype);

}  // namespace nve
