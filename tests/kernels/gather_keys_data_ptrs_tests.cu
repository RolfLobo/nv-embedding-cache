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

// Tests for GatherKeysAndDataPtrs / DefaultGPUHistogram (cuda_ops/gather_keys_data_ptrs.cuh),

#include "kernel_test_common.cuh"
#include <set>
#include <vector>

namespace {

// Result of one CallGatherKeysAndDataPtrs launch, downloaded to the host so both the address
// and priority checks can be verified against the same run without relaunching the kernel twice.
template <typename KeyType>
struct GatherKeysResult {
  std::vector<KeyType> unique_keys;
  std::vector<KeyType> mapping;    // entry chosen for each output slot id
  std::vector<KeyType> keys;       // full keys array, indexed by entry
  std::vector<const int8_t*> data_ptrs;
  std::vector<float> priorities;   // priority_out[id], already normalized
  const int8_t* base = nullptr;
  int64_t embed_width_in_bytes = 0;
  float norm_factor = 0.0f;
};

// `data` is only ever used for pointer arithmetic inside GatherKeysAndDataPtrs (never
// dereferenced), so an arbitrary non-null device address is enough to check the
// computed offsets without needing a real backing allocation.
template <typename KeyType, bool LoadIndices>
GatherKeysResult<KeyType> RunGatherKeysAndDataPtrs() {
  constexpr int64_t num_unique = 37;
  constexpr int64_t num_entries = 100;
  constexpr int64_t embed_width_in_bytes = 64;
  constexpr float norm_factor = 0.5f;
  const int8_t* const base = reinterpret_cast<const int8_t*>(uintptr_t{0x100000});

  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

  TestBuffer<KeyType> mapping(num_unique * sizeof(KeyType));
  TestBuffer<KeyType> keys(num_entries * sizeof(KeyType));
  TestBuffer<int> priorities(num_unique * sizeof(int));
  TestBuffer<KeyType> unique_keys_out(num_unique * sizeof(KeyType));
  TestBuffer<int8_t*> data_ptrs_out(num_unique * sizeof(int8_t*));

  // keys[entry] deliberately differs from `entry` itself so the LoadIndices=true
  // (address-by-key-value) and LoadIndices=false (address-by-entry-position) cases
  // produce distinguishable, independently-checkable addresses.
  for (int64_t e = 0; e < num_entries; e++) {
    keys.ph[e] = static_cast<KeyType>((e * 7 + 3) % 1000);
  }
  for (int64_t id = 0; id < num_unique; id++) {
    mapping.ph[id] = static_cast<KeyType>(id * 2);
    priorities.ph[id] = static_cast<int>(id + 1);
  }
  keys.HtoD(stream);
  mapping.HtoD(stream);
  priorities.HtoD(stream);

  nve::CallGatherKeysAndDataPtrs<KeyType, LoadIndices>(
      base, mapping.pd, priorities.pd, keys.pd, norm_factor, num_unique,
      embed_width_in_bytes, unique_keys_out.pd, data_ptrs_out.pd, stream);

  unique_keys_out.DtoH(stream);
  data_ptrs_out.DtoH(stream);
  priorities.DtoH(stream);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

  GatherKeysResult<KeyType> result;
  result.base = base;
  result.embed_width_in_bytes = embed_width_in_bytes;
  result.norm_factor = norm_factor;
  result.unique_keys.assign(unique_keys_out.ph, unique_keys_out.ph + num_unique);
  result.mapping.assign(mapping.ph, mapping.ph + num_unique);
  result.keys.assign(keys.ph, keys.ph + num_entries);
  result.data_ptrs.assign(data_ptrs_out.ph, data_ptrs_out.ph + num_unique);
  result.priorities.resize(num_unique);
  for (int64_t id = 0; id < num_unique; id++) {
    result.priorities[id] = *reinterpret_cast<float*>(&priorities.ph[id]);
  }
  return result;
}

}  // namespace

// ---------------------------------------------------------------------------------
// LoadIndices=true vs LoadIndices=false: verify data_ptrs[id] uses
// key*stride (dense table indexed directly by key value) vs entry*stride
// (position of first occurrence), respectively (gather_keys_data_ptrs.cuh:51).
// ---------------------------------------------------------------------------------
template <typename KeyType, bool LoadIndices>
void CheckDataPtrs() {
  const auto r = RunGatherKeysAndDataPtrs<KeyType, LoadIndices>();
  for (size_t id = 0; id < r.unique_keys.size(); id++) {
    const KeyType entry = r.mapping[id];
    const KeyType key = r.keys[static_cast<size_t>(entry)];
    EXPECT_EQ(r.unique_keys[id], key) << "id=" << id;

    const int64_t offset_index =
        LoadIndices ? static_cast<int64_t>(key) : static_cast<int64_t>(entry);
    const int8_t* expected_ptr = r.base + offset_index * r.embed_width_in_bytes;
    EXPECT_EQ(r.data_ptrs[id], expected_ptr) << "id=" << id;
  }
}
TEST(GatherKeysDataPtrs, Int32LoadIndicesTrue) { CheckDataPtrs<int32_t, true>(); }
TEST(GatherKeysDataPtrs, Int32LoadIndicesFalse) { CheckDataPtrs<int32_t, false>(); }
TEST(GatherKeysDataPtrs, Int64LoadIndicesTrue) { CheckDataPtrs<int64_t, true>(); }
TEST(GatherKeysDataPtrs, Int64LoadIndicesFalse) { CheckDataPtrs<int64_t, false>(); }

// ---------------------------------------------------------------------------------
// Priority normalization: priority_out[id] == priorities_in[id] * norm_factor.
// ---------------------------------------------------------------------------------
template <typename KeyType>
void CheckPriorityNormalization() {
  // LoadIndices does not affect priority computation; either instantiation exercises it.
  const auto r = RunGatherKeysAndDataPtrs<KeyType, false>();
  for (size_t id = 0; id < r.priorities.size(); id++) {
    const float priority_in = static_cast<float>(id + 1);  // matches the harness's seeding
    const float expected = priority_in * r.norm_factor;
    EXPECT_FLOAT_EQ(r.priorities[id], expected) << "id=" << id;
  }
}
TEST(GatherKeysPriorityNormalization, Int32) { CheckPriorityNormalization<int32_t>(); }
TEST(GatherKeysPriorityNormalization, Int64) { CheckPriorityNormalization<int64_t>(); }

// ---------------------------------------------------------------------------------
// DefaultGPUHistogram with all-duplicate keys (num_unique_keys_ == 1) vs.
// all-unique keys (num_unique_keys_ == num_keys): boundary sizes for the internal
// buffer layout computed in compute_alloc_size()/compute_histogram().
// ---------------------------------------------------------------------------------
template <typename KeyType>
void RunHistogramAllDuplicateTest() {
  constexpr int64_t num_keys = 300;
  constexpr KeyType the_key = static_cast<KeyType>(77);
  constexpr size_t embedding_size = 16;

  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

  TestBuffer<KeyType> keys(num_keys * sizeof(KeyType));
  for (int64_t i = 0; i < num_keys; i++) keys.ph[i] = the_key;
  keys.HtoD(stream);

  nve::DefaultGPUHistogram<KeyType> hist(num_keys);
  int8_t* tmp_storage = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc(&tmp_storage, hist.get_alloc_size()));

  hist.compute_histogram(keys.pd, num_keys, reinterpret_cast<const int8_t*>(keys.ph),
                         embedding_size, tmp_storage, stream);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

  ASSERT_EQ(hist.get_num_bins(), 1);
  KeyType key_out{};
  float priority_out = 0.0f;
  CHECK_CUDA_ERROR(cudaMemcpy(&key_out, hist.get_keys(), sizeof(KeyType), cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(
      cudaMemcpy(&priority_out, hist.get_priority(), sizeof(float), cudaMemcpyDeviceToHost));
  EXPECT_EQ(key_out, the_key);
  EXPECT_FLOAT_EQ(priority_out, 1.0f);  // the single bin holds every key: count/total == 1

  CHECK_CUDA_ERROR(cudaFree(tmp_storage));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}
TEST(HistogramAllDuplicate, Int32) { RunHistogramAllDuplicateTest<int32_t>(); }
TEST(HistogramAllDuplicate, Int64) { RunHistogramAllDuplicateTest<int64_t>(); }

template <typename KeyType>
void RunHistogramAllUniqueTest() {
  constexpr int64_t num_keys = 300;
  constexpr size_t embedding_size = 16;

  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

  TestBuffer<KeyType> keys(num_keys * sizeof(KeyType));
  std::set<KeyType> expected_keys;
  for (int64_t i = 0; i < num_keys; i++) {
    keys.ph[i] = static_cast<KeyType>(i + 1);
    expected_keys.insert(keys.ph[i]);
  }
  keys.HtoD(stream);

  nve::DefaultGPUHistogram<KeyType> hist(num_keys);
  int8_t* tmp_storage = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc(&tmp_storage, hist.get_alloc_size()));

  hist.compute_histogram(keys.pd, num_keys, reinterpret_cast<const int8_t*>(keys.ph),
                         embedding_size, tmp_storage, stream);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

  ASSERT_EQ(hist.get_num_bins(), num_keys);
  std::vector<KeyType> keys_out(static_cast<size_t>(num_keys));
  std::vector<float> priority_out(static_cast<size_t>(num_keys));
  CHECK_CUDA_ERROR(cudaMemcpy(keys_out.data(), hist.get_keys(), keys_out.size() * sizeof(KeyType),
                             cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaMemcpy(priority_out.data(), hist.get_priority(),
                             priority_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

  const std::set<KeyType> actual_keys(keys_out.begin(), keys_out.end());
  EXPECT_EQ(actual_keys, expected_keys);
  // Every bin holds exactly one key out of num_keys total.
  const float expected_priority = 1.0f / static_cast<float>(num_keys);
  for (const auto p : priority_out) {
    EXPECT_FLOAT_EQ(p, expected_priority);
  }

  CHECK_CUDA_ERROR(cudaFree(tmp_storage));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}
TEST(HistogramAllUnique, Int32) { RunHistogramAllUniqueTest<int32_t>(); }
TEST(HistogramAllUnique, Int64) { RunHistogramAllUniqueTest<int64_t>(); }

// ---------------------------------------------------------------------------------
// "histogram object was allocated for fewer keys than provided": the guard
// at gather_keys_data_ptrs.cuh:122 must fire when compute_histogram() is called with
// more keys than the object's constructor capacity, before touching the (too-small)
// tmpStorage buffer.
// ---------------------------------------------------------------------------------
template <typename KeyType>
void RunHistogramCapacityGuardTest() {
  constexpr int64_t capacity = 4;
  constexpr int64_t provided_num_keys = 40;
  constexpr size_t embedding_size = 16;

  TestBuffer<KeyType> keys(provided_num_keys * sizeof(KeyType));
  for (int64_t i = 0; i < provided_num_keys; i++) keys.ph[i] = static_cast<KeyType>(i);
  keys.HtoD(0);

  nve::DefaultGPUHistogram<KeyType> hist(capacity);
  int8_t* tmp_storage = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc(&tmp_storage, hist.get_alloc_size()));

  EXPECT_THROW(hist.compute_histogram(keys.pd, provided_num_keys,
                                      reinterpret_cast<const int8_t*>(keys.ph), embedding_size,
                                      tmp_storage, 0),
               nve::Exception);

  CHECK_CUDA_ERROR(cudaFree(tmp_storage));
}
TEST(HistogramCapacityGuard, Int32) { RunHistogramCapacityGuardTest<int32_t>(); }
TEST(HistogramCapacityGuard, Int64) { RunHistogramCapacityGuardTest<int64_t>(); }
