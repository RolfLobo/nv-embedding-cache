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

// Boundary tests for gather_flow_pipeline (cuda_ops/pipeline_gather.cuh).

#include <gtest/gtest.h>
#include <buffer_wrapper.hpp>
#include <common.hpp>
#include <cuda_ops/pipeline_gather.cuh>
#include <include/ecache/ec_set_associative.cuh>
#include <include/execution_context.hpp>
#include <default_allocator.hpp>
#include <vector>
#include "cuda_ptr.hpp"
#include "include/gpu_table.hpp"

namespace nve {

template <typename IndexT>
class PipelineGatherBoundaryTest : public ::testing::Test {
 public:
  static constexpr size_t kRowSizeInBytes = 32;
  static constexpr uint64_t kNumEmbeddings = 100000;
  // Sized generously (~32K entries) relative to the small key counts used
  // below so the all-hits boundary test has negligible collision-driven eviction risk.
  static constexpr size_t kCacheSizeInBytes = 1ull << 20;

  void SetUp() override {
    NVE_CHECK_(cudaMallocHost(&uvm_table_ptr_, kNumEmbeddings * kRowSizeInBytes));
    for (uint64_t i = 0; i < kNumEmbeddings; i++) {
      for (size_t j = 0; j < kRowSizeInBytes; j++) {
        uvm_table_ptr_[i * kRowSizeInBytes + j] = static_cast<int8_t>((i + j) % 256);
      }
    }
    NVE_CHECK_(cudaStreamCreate(&stream_));

    pipeline_params_ = GatherKernelPipelineParams{4096 /*task_size*/, 4 /*num_aux_streams*/};

    nve::GPUTableConfig cfg;
    cfg.device_id = 0;
    cfg.cache_size = kCacheSizeInBytes;
    cfg.max_modify_size = (1l << 20);
    cfg.row_size_in_bytes = kRowSizeInBytes;
    cfg.count_misses = true;
    cfg.uvm_table = uvm_table_ptr_;
    cfg.uvm_num_rows = kNumEmbeddings;
    cfg.kernel_mode_type = static_cast<uint64_t>(KernelType::PipelineGather); // routes into gather_flow_pipeline
    cfg.kernel_mode_value = reinterpret_cast<uintptr_t>(&pipeline_params_);
    cfg.modify_on_gpu = false;

    gpu_table_ = std::make_shared<nve::GpuTable<IndexT>>(cfg, nullptr);
    ctx_ = gpu_table_->create_execution_context(stream_, stream_, nullptr, nullptr);
  }

  void TearDown() override {
    NVE_CHECK_(cudaStreamSynchronize(stream_));
    ctx_.reset();
    gpu_table_.reset();
    if (uvm_table_ptr_) {
      NVE_CHECK_(cudaFreeHost(uvm_table_ptr_));
      uvm_table_ptr_ = nullptr;
    }
    NVE_CHECK_(cudaStreamDestroy(stream_));
  }

  void InsertAll(const std::vector<IndexT>& keys) {
    if (keys.empty()) return;
    std::vector<int8_t> h_values(keys.size() * kRowSizeInBytes);
    for (size_t i = 0; i < keys.size(); i++) {
      for (size_t j = 0; j < kRowSizeInBytes; j++) {
        h_values[i * kRowSizeInBytes + j] =
            uvm_table_ptr_[static_cast<uint64_t>(keys[i]) * kRowSizeInBytes + j];
      }
    }
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx_, "keys", keys.data(),
                                                                keys.size() * sizeof(IndexT));
    auto values_bw = std::make_shared<BufferWrapper<const void>>(ctx_, "values", h_values.data(),
                                                                  h_values.size());
    gpu_table_->insert(ctx_, static_cast<int64_t>(keys.size()), std::move(keys_bw),
                       kRowSizeInBytes, kRowSizeInBytes, std::move(values_bw));
    NVE_CHECK_(cudaDeviceSynchronize());
  }

  // Buffers are always allocated for at least 1 element (even when keys is empty) so
  // the n == 0 test exercises gather_flow_pipeline's early-return with valid, if
  // unused, device pointers rather than nullptrs of unclear support.
  std::vector<int8_t> Lookup(const std::vector<IndexT>& keys) {
    const size_t n = keys.size();
    void* raw_keys = nullptr;
    NVE_CHECK_(cudaMalloc(&raw_keys, std::max<size_t>(n, 1) * sizeof(IndexT)));
    DevicePtr<void> d_keys(raw_keys);

    void* raw_values = nullptr;
    NVE_CHECK_(cudaMalloc(&raw_values, std::max<size_t>(n, 1) * kRowSizeInBytes));
    DevicePtr<void> d_values(raw_values);

    if (n > 0) {
      NVE_CHECK_(cudaMemcpy(d_keys.get(), keys.data(), n * sizeof(IndexT), cudaMemcpyHostToDevice));
    }

    auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx_, "keys", d_keys.get(),
                                                               n * sizeof(IndexT));
    auto values_bw = std::make_shared<BufferWrapper<void>>(ctx_, "values", d_values.get(),
                                                           n * kRowSizeInBytes);
    gpu_table_->find(ctx_, static_cast<int64_t>(n), std::move(keys_bw), nullptr, kRowSizeInBytes,
                     std::move(values_bw), nullptr);
    NVE_CHECK_(cudaStreamSynchronize(stream_));

    std::vector<int8_t> h_values(n * kRowSizeInBytes);
    if (n > 0) {
      NVE_CHECK_(
          cudaMemcpy(h_values.data(), d_values.get(), h_values.size(), cudaMemcpyDeviceToHost));
    }
    return h_values;
  }

  int64_t GetLookupCounter() {
    int64_t counter = 0;
    gpu_table_->get_lookup_counter(ctx_, &counter);
    return counter;
  }

  int8_t* uvm_table_ptr_ = nullptr;
  cudaStream_t stream_{};
  context_ptr_t ctx_;
  std::shared_ptr<nve::GpuTable<IndexT>> gpu_table_;
  GatherKernelPipelineParams pipeline_params_{};
};

using PipelineGatherBoundaryTestInt32 = PipelineGatherBoundaryTest<int32_t>;
using PipelineGatherBoundaryTestInt64 = PipelineGatherBoundaryTest<int64_t>;

TEST_F(PipelineGatherBoundaryTestInt32, EmptyLookupDoesNotCrash) {
  const std::vector<int32_t> keys;
  EXPECT_NO_THROW({
    const auto values = Lookup(keys);
    EXPECT_TRUE(values.empty());
  });
}
TEST_F(PipelineGatherBoundaryTestInt64, EmptyLookupDoesNotCrash) {
  const std::vector<int64_t> keys;
  EXPECT_NO_THROW({
    const auto values = Lookup(keys);
    EXPECT_TRUE(values.empty());
  });
}

template <typename IndexT>
void RunAllMissesBoundary(PipelineGatherBoundaryTest<IndexT>& fixture) {
  constexpr int64_t n = 500;
  std::vector<IndexT> keys(n);
  for (int64_t i = 0; i < n; i++) keys[i] = static_cast<IndexT>(i);

  fixture.gpu_table_->reset_lookup_counter(fixture.ctx_);
  const auto values = fixture.Lookup(keys);
  EXPECT_EQ(fixture.GetLookupCounter(), n) << "no key was ever inserted, so all must miss";

  for (int64_t i = 0; i < n; i++) {
    for (size_t j = 0; j < fixture.kRowSizeInBytes; j++) {
      EXPECT_EQ(values[static_cast<size_t>(i) * fixture.kRowSizeInBytes + j],
               fixture.uvm_table_ptr_[static_cast<uint64_t>(keys[i]) * fixture.kRowSizeInBytes + j])
          << "key " << i << " byte " << j;
    }
  }
}
TEST_F(PipelineGatherBoundaryTestInt32, AllMissesBoundary) { RunAllMissesBoundary(*this); }
TEST_F(PipelineGatherBoundaryTestInt64, AllMissesBoundary) { RunAllMissesBoundary(*this); }

template <typename IndexT>
void RunAllHitsBoundary(PipelineGatherBoundaryTest<IndexT>& fixture) {
  // Far below the ~32K-entry cache capacity, so collision-driven eviction between
  // insert and lookup is effectively impossible.
  constexpr int64_t n = 32;
  std::vector<IndexT> keys(n);
  for (int64_t i = 0; i < n; i++) keys[i] = static_cast<IndexT>(i);

  fixture.InsertAll(keys);
  fixture.gpu_table_->reset_lookup_counter(fixture.ctx_);
  const auto values = fixture.Lookup(keys);
  EXPECT_EQ(fixture.GetLookupCounter(), 0) << "every key was pre-inserted, so none should miss";

  for (int64_t i = 0; i < n; i++) {
    for (size_t j = 0; j < fixture.kRowSizeInBytes; j++) {
      EXPECT_EQ(values[static_cast<size_t>(i) * fixture.kRowSizeInBytes + j],
               fixture.uvm_table_ptr_[static_cast<uint64_t>(keys[i]) * fixture.kRowSizeInBytes + j])
          << "key " << i << " byte " << j;
    }
  }
}
TEST_F(PipelineGatherBoundaryTestInt32, AllHitsBoundary) { RunAllHitsBoundary(*this); }
TEST_F(PipelineGatherBoundaryTestInt64, AllHitsBoundary) { RunAllHitsBoundary(*this); }

}  // namespace nve
