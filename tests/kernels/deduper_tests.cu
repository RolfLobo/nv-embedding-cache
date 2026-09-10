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

// Tests for nve::Deduper (cuda_ops/deduper.cuh).

#include "kernel_test_common.cuh"
#include <map>
#include <random>
#include <vector>

namespace {

// Owns a Deduper instance plus all device/host buffers it needs, sized generously
// (num_keys + 1) so both the MAX_RUN_SIZE == -1 and MAX_RUN_SIZE != -1 code paths are
// safe to exercise (the split path reads one element past the RLE-unique count, see
// deduper.cuh's ExclusiveSum calls).
template <typename IndexT, int32_t MAX_RUN_SIZE = -1>
struct DeduperHarness {
  explicit DeduperHarness(uint64_t capacity)
      : keys(std::max<uint64_t>(capacity, 1) * sizeof(IndexT)),
        unique_out((capacity + 1) * sizeof(IndexT)),
        counts_out((capacity + 1) * sizeof(IndexT)),
        loc_map_out((capacity + 1) * sizeof(IndexT)),
        inverse_buffer((capacity + 1) * sizeof(IndexT)),
        offsets((capacity + 1) * sizeof(IndexT)),
        deduper(nve::GetDefaultAllocator()) {
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_num_runs_out, sizeof(IndexT) * 2));
    size_t tmp_dev = 0, tmp_host = 0;
    deduper.get_alloc_requirements(capacity, tmp_dev, tmp_host);
    CHECK_CUDA_ERROR(cudaMalloc(&tmp_device_mem, tmp_dev));
    CHECK_CUDA_ERROR(cudaMallocHost(&tmp_host_mem, tmp_host));
    deduper.set_and_init_buffers(capacity, tmp_device_mem, tmp_host_mem);
  }

  ~DeduperHarness() {
    CHECK_CUDA_ERROR(cudaFreeHost(h_num_runs_out));
    CHECK_CUDA_ERROR(cudaFree(tmp_device_mem));
    CHECK_CUDA_ERROR(cudaFreeHost(tmp_host_mem));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
  }

  void Run(uint64_t n, int end_bit = sizeof(IndexT) * 8) {
    deduper.dedup(keys.pd, n, unique_out.pd, counts_out.pd,
                  (MAX_RUN_SIZE == -1) ? nullptr : loc_map_out.pd, h_num_runs_out,
                  inverse_buffer.pd, offsets.pd, stream, end_bit);
  }

  void SyncAllToHost() {
    unique_out.DtoH(stream);
    counts_out.DtoH(stream);
    loc_map_out.DtoH(stream);
    inverse_buffer.DtoH(stream);
    offsets.DtoH(stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
  }

  cudaStream_t stream;
  TestBuffer<IndexT> keys;
  TestBuffer<IndexT> unique_out;
  TestBuffer<IndexT> counts_out;
  TestBuffer<IndexT> loc_map_out;
  TestBuffer<IndexT> inverse_buffer;
  TestBuffer<IndexT> offsets;
  IndexT* h_num_runs_out = nullptr;
  char* tmp_device_mem = nullptr;
  char* tmp_host_mem = nullptr;
  nve::Deduper<IndexT, MAX_RUN_SIZE> deduper;
};

// Shared round-trip verifier for the MAX_RUN_SIZE == -1 path
// walks d_inverse_buffer via d_offsets/d_counts_out for every unique
// key and confirms every original index maps back to that key, and
// that every original index in [0, num_keys) is covered exactly once.
template <typename IndexT>
void VerifyRoundTrip(DeduperHarness<IndexT, -1>& h, const std::vector<IndexT>& original_keys) {
  h.SyncAllToHost();
  const auto num_runs = h.h_num_runs_out[0];
  std::vector<bool> covered(original_keys.size(), false);
  for (IndexT i = 0; i < num_runs; i++) {
    const IndexT key = h.unique_out.ph[i];
    const IndexT count = h.counts_out.ph[i];
    const IndexT offset = h.offsets.ph[i];
    for (IndexT j = 0; j < count; j++) {
      const IndexT orig_idx = h.inverse_buffer.ph[offset + j];
      ASSERT_GE(orig_idx, 0);
      ASSERT_LT(static_cast<size_t>(orig_idx), original_keys.size());
      EXPECT_EQ(original_keys[static_cast<size_t>(orig_idx)], key)
          << "unique key #" << i << " run element " << j;
      EXPECT_FALSE(covered[static_cast<size_t>(orig_idx)]) << "index " << orig_idx << " covered twice";
      covered[static_cast<size_t>(orig_idx)] = true;
    }
  }
  EXPECT_TRUE(std::all_of(covered.begin(), covered.end(), [](bool b) { return b; }))
      << "not every original index was covered by some run";
}

}  // namespace

// ---------------------------------------------------------------------------------
// All-unique keys: num_unique_out == num_keys, offsets[i] == i.
// ---------------------------------------------------------------------------------
template <typename IndexT>
void RunAllUniqueTest() {
  constexpr uint64_t num_keys = 1000;
  DeduperHarness<IndexT, -1> h(num_keys);
  std::vector<IndexT> ref_keys(num_keys);
  for (uint64_t i = 0; i < num_keys; i++) {
    ref_keys[i] = static_cast<IndexT>(i);
    h.keys.ph[i] = ref_keys[i];
  }
  h.keys.HtoD(h.stream);
  h.Run(num_keys);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(h.stream));

  EXPECT_EQ(h.h_num_runs_out[0], static_cast<IndexT>(num_keys));
  VerifyRoundTrip(h, ref_keys);

  // Ascending unique keys sort to themselves, so offsets[i] == i exactly.
  for (uint64_t i = 0; i < num_keys; i++) {
    EXPECT_EQ(h.offsets.ph[i], static_cast<IndexT>(i));
    EXPECT_EQ(h.counts_out.ph[i], static_cast<IndexT>(1));
  }
}
TEST(DeduperAllUnique, Int32) { RunAllUniqueTest<int32_t>(); }
TEST(DeduperAllUnique, Int64) { RunAllUniqueTest<int64_t>(); }

// ---------------------------------------------------------------------------------
// All-identical keys (single run): one unique key, counts_out[0] == num_keys.
// ---------------------------------------------------------------------------------
template <typename IndexT>
void RunAllIdenticalTest() {
  constexpr uint64_t num_keys = 777;
  constexpr IndexT the_key = static_cast<IndexT>(42);
  DeduperHarness<IndexT, -1> h(num_keys);
  std::vector<IndexT> ref_keys(num_keys, the_key);
  for (uint64_t i = 0; i < num_keys; i++) h.keys.ph[i] = the_key;
  h.keys.HtoD(h.stream);
  h.Run(num_keys);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(h.stream));

  ASSERT_EQ(h.h_num_runs_out[0], static_cast<IndexT>(1));
  VerifyRoundTrip(h, ref_keys);
  EXPECT_EQ(h.unique_out.ph[0], the_key);
  EXPECT_EQ(h.counts_out.ph[0], static_cast<IndexT>(num_keys));
  EXPECT_EQ(h.offsets.ph[0], static_cast<IndexT>(0));
}
TEST(DeduperAllIdentical, Int32) { RunAllIdenticalTest<int32_t>(); }
TEST(DeduperAllIdentical, Int64) { RunAllIdenticalTest<int64_t>(); }

// ---------------------------------------------------------------------------------
// num_keys == 0: completes without launching degenerate grids, no runs.
// ---------------------------------------------------------------------------------
template <typename IndexT>
void RunEmptyTest() {
  DeduperHarness<IndexT, -1> h(16);  // buffers sized for capacity, but we dedup 0 keys
  h.Run(0);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(h.stream));
  EXPECT_EQ(h.h_num_runs_out[0], static_cast<IndexT>(0));
}
TEST(DeduperEmpty, Int32) { RunEmptyTest<int32_t>(); }
TEST(DeduperEmpty, Int64) { RunEmptyTest<int64_t>(); }

// ---------------------------------------------------------------------------------
// num_keys == 1: smallest non-trivial case, boundary of the SPLIT_SIZE/warp
// loops in compute_split_counts / compute_location_mapping.
// ---------------------------------------------------------------------------------
template <typename IndexT>
void RunSingleKeyTest() {
  DeduperHarness<IndexT, -1> h(1);
  const std::vector<IndexT> ref_keys{static_cast<IndexT>(12345)};
  h.keys.ph[0] = ref_keys[0];
  h.keys.HtoD(h.stream);
  h.Run(1);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(h.stream));

  ASSERT_EQ(h.h_num_runs_out[0], static_cast<IndexT>(1));
  VerifyRoundTrip(h, ref_keys);
  EXPECT_EQ(h.unique_out.ph[0], ref_keys[0]);
  EXPECT_EQ(h.counts_out.ph[0], static_cast<IndexT>(1));
  EXPECT_EQ(h.offsets.ph[0], static_cast<IndexT>(0));
}
TEST(DeduperSingleKey, Int32) { RunSingleKeyTest<int32_t>(); }
TEST(DeduperSingleKey, Int64) { RunSingleKeyTest<int64_t>(); }

// ---------------------------------------------------------------------------------
// MAX_RUN_SIZE != -1 with a run longer than MAX_RUN_SIZE: verify
// compute_location_mapping splits the run into the correct number/size of chunks and
// d_loc_map_out/d_offsets agree with the manual chunk math (deduper.cuh:194-214).
// ---------------------------------------------------------------------------------
template <typename IndexT>
void RunSplitLongRunTest() {
  constexpr int32_t MAX_RUN_SIZE = 8;
  constexpr IndexT long_key = static_cast<IndexT>(500);
  constexpr uint64_t long_key_count = 20;  // ceil(20/8) == 3 chunks: 8, 8, 4

  std::vector<IndexT> ref_keys;
  for (uint64_t i = 0; i < long_key_count; i++) ref_keys.push_back(long_key);
  // A handful of single-occurrence keys, each its own 1-element chunk.
  for (IndexT k = 0; k < 5; k++) ref_keys.push_back(static_cast<IndexT>(1000 + k));

  DeduperHarness<IndexT, MAX_RUN_SIZE> h(ref_keys.size());
  for (size_t i = 0; i < ref_keys.size(); i++) h.keys.ph[i] = ref_keys[i];
  h.keys.HtoD(h.stream);
  h.Run(ref_keys.size());
  h.SyncAllToHost();

  const auto num_unique = h.h_num_runs_out[0];
  const auto num_chunks = h.h_num_runs_out[1];
  ASSERT_EQ(num_unique, static_cast<IndexT>(6));  // long_key + 5 singles

  // Expected chunk count per unique key: ceil(counts_out[k] / MAX_RUN_SIZE).
  IndexT expected_total_chunks = 0;
  std::vector<IndexT> expected_chunks_for_key(static_cast<size_t>(num_unique));
  for (IndexT k = 0; k < num_unique; k++) {
    const IndexT count = h.counts_out.ph[k];
    const IndexT chunks = static_cast<IndexT>(nve::ceil_div<IndexT>(count, MAX_RUN_SIZE));
    expected_chunks_for_key[static_cast<size_t>(k)] = chunks;
    expected_total_chunks += chunks;
  }
  EXPECT_EQ(num_chunks, expected_total_chunks);

  // Walk every chunk: verify it stays within MAX_RUN_SIZE, belongs to a plausible key,
  // and that its inverse-buffer elements all resolve to that key's value.
  std::vector<IndexT> actual_chunks_for_key(static_cast<size_t>(num_unique), 0);
  for (IndexT c = 0; c < num_chunks; c++) {
    const IndexT key_id = h.loc_map_out.ph[c];
    ASSERT_GE(key_id, 0);
    ASSERT_LT(key_id, num_unique);
    actual_chunks_for_key[static_cast<size_t>(key_id)]++;

    const IndexT start = h.offsets.ph[c];
    const IndexT end = h.offsets.ph[c + 1];
    const IndexT chunk_size = end - start;
    EXPECT_GT(chunk_size, static_cast<IndexT>(0));
    EXPECT_LE(chunk_size, static_cast<IndexT>(MAX_RUN_SIZE));

    const IndexT key = h.unique_out.ph[key_id];
    for (IndexT j = start; j < end; j++) {
      const IndexT orig_idx = h.inverse_buffer.ph[j];
      ASSERT_GE(orig_idx, 0);
      ASSERT_LT(static_cast<size_t>(orig_idx), ref_keys.size());
      EXPECT_EQ(ref_keys[static_cast<size_t>(orig_idx)], key);
    }
  }
  for (IndexT k = 0; k < num_unique; k++) {
    EXPECT_EQ(actual_chunks_for_key[static_cast<size_t>(k)], expected_chunks_for_key[static_cast<size_t>(k)])
        << "unique key #" << k;
  }
}
TEST(DeduperSplitLongRun, Int32) { RunSplitLongRunTest<int32_t>(); }
TEST(DeduperSplitLongRun, Int64) { RunSplitLongRunTest<int64_t>(); }

// ---------------------------------------------------------------------------------
// dedup() called with num_keys > init_num_keys_ (the size set_and_init_buffers
// was sized/seeded for): the NVE_CHECK_ guard at deduper.cuh:153 must fire before any
// buffer is touched (buffers here are intentionally sized only for the small capacity).
// ---------------------------------------------------------------------------------
template <typename IndexT>
void RunExceedsCapacityTest() {
  DeduperHarness<IndexT, -1> h(10);
  EXPECT_THROW(h.Run(20), nve::Exception);
}
TEST(DeduperExceedsCapacity, Int32) { RunExceedsCapacityTest<int32_t>(); }
TEST(DeduperExceedsCapacity, Int64) { RunExceedsCapacityTest<int64_t>(); }

// ---------------------------------------------------------------------------------
// Reduced end_bit: sorting only enough bits to distinguish the key range (as
// documented at deduper.cuh:132-135 for e.g. bucket indices) must still produce
// correct dedup grouping, since RLE compares full key equality, not just the sorted
// prefix bits.
// ---------------------------------------------------------------------------------
template <typename IndexT>
void RunReducedEndBitTest() {
  constexpr uint64_t num_keys = 4000;
  constexpr int reduced_end_bit = 12;  // covers [0, 4096), enough for all key values below
  constexpr IndexT key_range = 1 << 10;  // force plenty of duplicates within 12 bits

  std::mt19937 gen(0xE2DB17);
  std::uniform_int_distribution<int64_t> dist(0, key_range - 1);

  DeduperHarness<IndexT, -1> h(num_keys);
  std::vector<IndexT> ref_keys(num_keys);
  for (uint64_t i = 0; i < num_keys; i++) {
    ref_keys[i] = static_cast<IndexT>(dist(gen));
    h.keys.ph[i] = ref_keys[i];
  }
  h.keys.HtoD(h.stream);
  h.Run(num_keys, reduced_end_bit);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(h.stream));

  std::map<IndexT, IndexT> ref_histogram;
  for (const auto k : ref_keys) ref_histogram[k]++;

  EXPECT_EQ(h.h_num_runs_out[0], static_cast<IndexT>(ref_histogram.size()));
  VerifyRoundTrip(h, ref_keys);

  std::map<IndexT, IndexT> actual_histogram;
  for (IndexT i = 0; i < h.h_num_runs_out[0]; i++) {
    actual_histogram[h.unique_out.ph[i]] = h.counts_out.ph[i];
  }
  EXPECT_EQ(actual_histogram, ref_histogram);
}
TEST(DeduperReducedEndBit, Int32) { RunReducedEndBitTest<int32_t>(); }
TEST(DeduperReducedEndBit, Int64) { RunReducedEndBitTest<int64_t>(); }

// ---------------------------------------------------------------------------------
// Round-trip check with random duplicate-heavy input: for every unique key,
// every original index reachable via d_inverse_buffer[d_offsets[i]..+d_counts_out[i])
// must map back to that key, and every original index must be covered exactly once.
// This is the Deduper's core contract and was previously unverified end-to-end.
// ---------------------------------------------------------------------------------
template <typename IndexT>
void RunRoundTripRandomTest() {
  constexpr uint64_t num_keys = 5000;
  constexpr IndexT key_range = 500;  // small range relative to num_keys forces heavy duplication

  std::mt19937 gen(0x600D5EED);
  std::uniform_int_distribution<int64_t> dist(0, key_range - 1);

  DeduperHarness<IndexT, -1> h(num_keys);
  std::vector<IndexT> ref_keys(num_keys);
  for (uint64_t i = 0; i < num_keys; i++) {
    ref_keys[i] = static_cast<IndexT>(dist(gen));
    h.keys.ph[i] = ref_keys[i];
  }
  h.keys.HtoD(h.stream);
  h.Run(num_keys);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(h.stream));

  VerifyRoundTrip(h, ref_keys);
}
TEST(DeduperRoundTrip, Int32) { RunRoundTripRandomTest<int32_t>(); }
TEST(DeduperRoundTrip, Int64) { RunRoundTripRandomTest<int64_t>(); }
