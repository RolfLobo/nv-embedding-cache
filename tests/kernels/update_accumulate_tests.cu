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

// Tests for UpdateTable/UpdateAccumulateTable (cuda_ops/update_accumulate.cuh).

#include "kernel_test_common.cuh"
#include <random>

// ---------------------------------------------------------------------------------
// Concurrent accumulate to the same key from multiple indices in one call:
// verify atomicAdd correctness (no lost updates), since UpdateAccumulateTableKernel
// uses atomicAdd per element (update_accumulate.cuh:89).
// ---------------------------------------------------------------------------------
template <typename KeyT>
void RunConcurrentAccumulateTest() {
  constexpr uint64_t num_rows = 4;
  constexpr uint32_t embed_width = 128;  // elements
  constexpr int32_t num_indices = 4096;  // heavy contention: all indices hit the same key
  constexpr KeyT target_key = static_cast<KeyT>(2);

  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

  TestBuffer<float> table(num_rows * embed_width * sizeof(float));
  TestBuffer<float> updates(static_cast<size_t>(num_indices) * embed_width * sizeof(float));
  TestBuffer<KeyT> keys(num_indices * sizeof(KeyT));

  std::memset(table.ph, 0, table.size_);

  std::mt19937 gen(0xACC0FFEE);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<double> expected(embed_width, 0.0);
  for (int32_t i = 0; i < num_indices; i++) {
    keys.ph[i] = target_key;
    for (uint32_t e = 0; e < embed_width; e++) {
      const float v = dist(gen);
      updates.ph[i * embed_width + e] = v;
      expected[e] += v;
    }
  }
  table.HtoD(stream);
  updates.HtoD(stream);
  keys.HtoD(stream);

  nve::UpdateAccumulateTable(updates.pd, keys.pd, table.pd, embed_width, embed_width, embed_width,
                             num_indices, num_rows, stream);

  table.DtoH(stream);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

  // Bound is far looser than float rounding error over 4096 additions (~5e-4 worst
  // case); a real lost-update bug would miss at least one term of magnitude ~O(1).
  constexpr float tol = 0.05f;
  for (uint32_t e = 0; e < embed_width; e++) {
    EXPECT_NEAR(table.ph[static_cast<size_t>(target_key) * embed_width + e],
               static_cast<float>(expected[e]), tol)
        << "elem " << e;
  }
  for (uint64_t r = 0; r < num_rows; r++) {
    if (static_cast<KeyT>(r) == target_key) continue;
    for (uint32_t e = 0; e < embed_width; e++) {
      EXPECT_FLOAT_EQ(table.ph[r * embed_width + e], 0.0f) << "row " << r << " elem " << e
                                                            << " was written unexpectedly";
    }
  }
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}
TEST(UpdateAccumulateConcurrency, Int32SameKeyManyIndices) {
  RunConcurrentAccumulateTest<int32_t>();
}
TEST(UpdateAccumulateConcurrency, Int64SameKeyManyIndices) {
  RunConcurrentAccumulateTest<int64_t>();
}

// ---------------------------------------------------------------------------------
// Each subgroupWidth switch case (32/16/8/4/2/1) in CallUpdatKernelVecType
// (update_accumulate.cuh:124-148, called via UpdateTable's dispatch) and in
// UpdateAccumulateTable's own switch (update_accumulate.cuh:215-251).
// subgroupWidth = min(nextPow2(num_elements), 32); picking num_elements as an exact
// power of two selects that case directly.
// ---------------------------------------------------------------------------------
template <typename KeyT>
void RunUpdateSubgroupWidthTest(uint32_t embed_width_in_bytes) {
  constexpr uint64_t num_rows = 3;
  constexpr int32_t num_indices = 3;
  const uint32_t num_elements = embed_width_in_bytes / sizeof(float);

  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

  TestBuffer<float> table(static_cast<size_t>(num_rows) * num_elements * sizeof(float));
  TestBuffer<float> updates(static_cast<size_t>(num_indices) * num_elements * sizeof(float));
  TestBuffer<KeyT> keys(num_indices * sizeof(KeyT));

  for (uint64_t r = 0; r < num_rows; r++) {
    for (uint32_t e = 0; e < num_elements; e++) table.ph[r * num_elements + e] = -1.0f;
  }
  for (int32_t i = 0; i < num_indices; i++) {
    keys.ph[i] = static_cast<KeyT>(i);
    for (uint32_t e = 0; e < num_elements; e++) {
      updates.ph[i * num_elements + e] = static_cast<float>(i * 100 + static_cast<int32_t>(e));
    }
  }
  table.HtoD(stream);
  updates.HtoD(stream);
  keys.HtoD(stream);

  nve::CallUpdatKernelVecType<KeyT, float>(
      reinterpret_cast<const int8_t*>(updates.pd), keys.pd, reinterpret_cast<int8_t*>(table.pd),
      embed_width_in_bytes, embed_width_in_bytes, embed_width_in_bytes, num_indices, num_rows,
      stream);

  table.DtoH(stream);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

  for (int32_t i = 0; i < num_indices; i++) {
    for (uint32_t e = 0; e < num_elements; e++) {
      EXPECT_FLOAT_EQ(table.ph[static_cast<size_t>(i) * num_elements + e],
                      static_cast<float>(i * 100 + static_cast<int32_t>(e)))
          << "row " << i << " elem " << e;
    }
  }
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}
// embed_width_in_bytes = num_elements * sizeof(float); num_elements chosen as the
// exact power of two so nextPow2(num_elements) == num_elements == target subgroupWidth.
TEST(UpdateSubgroupWidth, Width32) { RunUpdateSubgroupWidthTest<int32_t>(32 * 4); }
TEST(UpdateSubgroupWidth, Width16) { RunUpdateSubgroupWidthTest<int32_t>(16 * 4); }
TEST(UpdateSubgroupWidth, Width8) { RunUpdateSubgroupWidthTest<int32_t>(8 * 4); }
TEST(UpdateSubgroupWidth, Width4) { RunUpdateSubgroupWidthTest<int32_t>(4 * 4); }
TEST(UpdateSubgroupWidth, Width2) { RunUpdateSubgroupWidthTest<int32_t>(2 * 4); }
TEST(UpdateSubgroupWidth, Width1) { RunUpdateSubgroupWidthTest<int32_t>(1 * 4); }
TEST(UpdateSubgroupWidth, Width32Int64Keys) { RunUpdateSubgroupWidthTest<int64_t>(32 * 4); }
TEST(UpdateSubgroupWidth, Width1Int64Keys) { RunUpdateSubgroupWidthTest<int64_t>(1 * 4); }

template <typename KeyT>
void RunUpdateAccumulateSubgroupWidthTest(uint32_t embed_width /* elements */) {
  constexpr uint64_t num_rows = 3;
  constexpr int32_t num_indices = 3;
  constexpr float baseline = 10.0f;

  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

  TestBuffer<float> table(static_cast<size_t>(num_rows) * embed_width * sizeof(float));
  TestBuffer<float> updates(static_cast<size_t>(num_indices) * embed_width * sizeof(float));
  TestBuffer<KeyT> keys(num_indices * sizeof(KeyT));

  for (uint64_t r = 0; r < num_rows; r++) {
    for (uint32_t e = 0; e < embed_width; e++) table.ph[r * embed_width + e] = baseline;
  }
  for (int32_t i = 0; i < num_indices; i++) {
    keys.ph[i] = static_cast<KeyT>(i);
    for (uint32_t e = 0; e < embed_width; e++) {
      updates.ph[i * embed_width + e] = static_cast<float>(i + 1) * 0.5f;
    }
  }
  table.HtoD(stream);
  updates.HtoD(stream);
  keys.HtoD(stream);

  nve::UpdateAccumulateTable(updates.pd, keys.pd, table.pd, embed_width, embed_width, embed_width,
                             num_indices, num_rows, stream);

  table.DtoH(stream);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

  for (int32_t i = 0; i < num_indices; i++) {
    const float expected = baseline + static_cast<float>(i + 1) * 0.5f;
    for (uint32_t e = 0; e < embed_width; e++) {
      EXPECT_FLOAT_EQ(table.ph[static_cast<size_t>(i) * embed_width + e], expected)
          << "row " << i << " elem " << e;
    }
  }
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}
TEST(UpdateAccumulateSubgroupWidth, Width32) { RunUpdateAccumulateSubgroupWidthTest<int32_t>(32); }
TEST(UpdateAccumulateSubgroupWidth, Width16) { RunUpdateAccumulateSubgroupWidthTest<int32_t>(16); }
TEST(UpdateAccumulateSubgroupWidth, Width8) { RunUpdateAccumulateSubgroupWidthTest<int32_t>(8); }
TEST(UpdateAccumulateSubgroupWidth, Width4) { RunUpdateAccumulateSubgroupWidthTest<int32_t>(4); }
TEST(UpdateAccumulateSubgroupWidth, Width2) { RunUpdateAccumulateSubgroupWidthTest<int32_t>(2); }
TEST(UpdateAccumulateSubgroupWidth, Width1) { RunUpdateAccumulateSubgroupWidthTest<int32_t>(1); }
TEST(UpdateAccumulateSubgroupWidth, Width32Int64Keys) {
  RunUpdateAccumulateSubgroupWidthTest<int64_t>(32);
}
TEST(UpdateAccumulateSubgroupWidth, Width1Int64Keys) {
  RunUpdateAccumulateSubgroupWidthTest<int64_t>(1);
}

// ---------------------------------------------------------------------------------
// Unsupported subgroupWidth: the switch statements in CallUpdatKernelVecType
// and UpdateAccumulateTable (update_accumulate.cuh:146, 249) have a `default:
// NVE_THROW_(...)` branch for an "unsupported" subgroupWidth.
//
// subgroupWidth is always computed as std::min(nve::nextPow2(num_elements), 32u).
// nve::nextPow2() (cuda_ops/kernels_common.cuh:41) floors any x >= 1 to the nearest
// power of two, so its output -- and therefore subgroupWidth after the min() cap --
// is always exactly one of {1, 2, 4, 8, 16, 32}: precisely the switch's non-default
// cases. That makes the `default:` branch dead code for every valid (num_elements >=
// 1) input; the only way to reach it is num_elements == 0, where nextPow2(0) invokes
// __builtin_clz(0), itself undefined behavior. We do not exercise that path here (it
// risks crashing the test binary rather than throwing); this test instead documents
// the invariant that makes the throw unreachable in practice, as a pointer for anyone
// auditing that branch.
// ---------------------------------------------------------------------------------
TEST(UpdateSubgroupWidthDefaultBranch, IsUnreachableForValidElementCounts) {
  for (const uint32_t elements :
       {1u, 2u, 3u, 4u, 5u, 7u, 8u, 15u, 16u, 17u, 31u, 32u, 33u, 63u, 64u, 100u, 1000u}) {
    const uint32_t subgroup_width = std::min(nve::nextPow2(elements), 32u);
    EXPECT_TRUE(subgroup_width == 1 || subgroup_width == 2 || subgroup_width == 4 ||
               subgroup_width == 8 || subgroup_width == 16 || subgroup_width == 32)
        << "elements=" << elements << " produced subgroupWidth=" << subgroup_width
        << ", which would hit the (presumed unreachable) default: NVE_THROW_ branch";
  }
}
