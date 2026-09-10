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

// Tests for EmbeddingForwardScatter (cuda_ops/scatter.cuh).

#include "kernel_test_common.cuh"
#include <cstring>
#include <functional>

namespace {

constexpr int8_t kDstSentinel = 0x55;

// Fills src with a per-(index,byte) distinguishable pattern, dst with a sentinel that
// never collides with the src pattern, runs EmbeddingForwardScatter, and verifies:
// a hit_mask bit set for an index means scatter must skip it (dst stays the sentinel),
// a bit unset means dst must equal src for that row.
void RunScatterHitMaskTest(int32_t num_indices, uint32_t embed_width_in_bytes,
                            const std::function<bool(int32_t)>& is_hit) {
  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

  TestBuffer<int8_t> src(static_cast<size_t>(num_indices) * embed_width_in_bytes);
  TestBuffer<int8_t> dst(static_cast<size_t>(num_indices) * embed_width_in_bytes);
  const int64_t hitmask_elems = nve::ceil_div<int64_t>(num_indices, 64);
  TestBuffer<uint64_t> hit_mask(static_cast<size_t>(hitmask_elems) * sizeof(uint64_t));
  std::memset(hit_mask.ph, 0, hit_mask.size_);

  for (int32_t i = 0; i < num_indices; i++) {
    for (uint32_t b = 0; b < embed_width_in_bytes; b++) {
      src.ph[i * embed_width_in_bytes + b] = static_cast<int8_t>((i * 31 + b) & 0x7f);
      dst.ph[i * embed_width_in_bytes + b] = kDstSentinel;
    }
    if (is_hit(i)) {
      hit_mask.ph[i / 64] |= (uint64_t{1} << (i % 64));
    }
  }
  src.HtoD(stream);
  dst.HtoD(stream);
  hit_mask.HtoD(stream);

  nve::EmbeddingForwardScatter(src.pd, dst.pd, embed_width_in_bytes, embed_width_in_bytes,
                               embed_width_in_bytes, hit_mask.pd, num_indices, stream);

  dst.DtoH(stream);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

  for (int32_t i = 0; i < num_indices; i++) {
    const int8_t* src_row = src.ph + i * embed_width_in_bytes;
    const int8_t* dst_row = dst.ph + i * embed_width_in_bytes;
    if (is_hit(i)) {
      for (uint32_t b = 0; b < embed_width_in_bytes; b++) {
        EXPECT_EQ(dst_row[b], kDstSentinel) << "hit index " << i << " byte " << b
                                            << " should not have been scattered";
      }
    } else {
      EXPECT_TRUE(std::equal(src_row, src_row + embed_width_in_bytes, dst_row))
          << "miss index " << i << " was not scattered correctly";
    }
  }
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

}  // namespace

// ---------------------------------------------------------------------------------
// hit_mask bit set vs unset: scatter must skip writing when the corresponding
// bit is 1 (scatter.cuh:47-49), and write when 0.
// ---------------------------------------------------------------------------------
TEST(ScatterHitMask, AlternatingBitsAreSkipped) {
  RunScatterHitMaskTest(200, 64, [](int32_t i) { return (i % 2) == 0; });
}

TEST(ScatterHitMask, AllHitsProduceNoWrites) {
  RunScatterHitMaskTest(80, 32, [](int32_t) { return true; });
}

TEST(ScatterHitMask, AllMissesScatterEverything) {
  RunScatterHitMaskTest(80, 32, [](int32_t) { return false; });
}

// ---------------------------------------------------------------------------------
// Mask boundary indices: embed values at mask_entry boundaries (63, 64, 65) to
// catch off-by-one errors in `embed/64` / `embed%64` (scatter.cuh:42-45).
// ---------------------------------------------------------------------------------
TEST(ScatterHitMask, MaskWordBoundaryIndices) {
  // Spans two uint64_t hit_mask words; only the three indices straddling the boundary
  // between word 0 (bits 0-63) and word 1 (bits 0-63, entry 64) are hits.
  RunScatterHitMaskTest(130, 16, [](int32_t i) { return i == 63 || i == 64 || i == 65; });
}

TEST(ScatterHitMask, LastIndexInWordIsHit) {
  RunScatterHitMaskTest(65, 16, [](int32_t i) { return i == 63; });
}

TEST(ScatterHitMask, FirstIndexOfSecondWordIsHit) {
  RunScatterHitMaskTest(65, 16, [](int32_t i) { return i == 64; });
}

// ---------------------------------------------------------------------------------
// Each embed_width_in_bytes branch in EmbeddingForwardScatter (mod 16/8/4/2,
// scatter.cuh:122-134): confirm correct vector width is selected and data is
// correctly copied for each. All-miss so every row must be scattered.
// ---------------------------------------------------------------------------------
TEST(ScatterWidthBranch, Width16UsesVec4Float) {
  RunScatterHitMaskTest(50, 16, [](int32_t) { return false; });
}

TEST(ScatterWidthBranch, Width8UsesVec2Float) {
  RunScatterHitMaskTest(50, 8, [](int32_t) { return false; });
}

TEST(ScatterWidthBranch, Width4UsesVec1Float) {
  RunScatterHitMaskTest(50, 4, [](int32_t) { return false; });
}

TEST(ScatterWidthBranch, Width2UsesVec1Half) {
  RunScatterHitMaskTest(50, 2, [](int32_t) { return false; });
}

// Widths that are multiples of a larger branch but not the largest one should still
// take the coarsest matching branch (e.g. 48 is a multiple of both 16 and 8; the %16
// check comes first in EmbeddingForwardScatter so it wins).
TEST(ScatterWidthBranch, Width48PrefersVec4Branch) {
  RunScatterHitMaskTest(50, 48, [](int32_t) { return false; });
}

// ---------------------------------------------------------------------------------
// embed_width_in_bytes odd (not a multiple of 2): EmbeddingForwardScatter
// currently falls through every branch with no `else` (scatter.cuh:122-134), so it
// silently performs no scatter and raises no error. This test documents the current
// behavior rather than asserting it is correct; if the no-op is fixed (e.g. to throw
// on unsupported widths), this test's expectation should be updated accordingly.
// ---------------------------------------------------------------------------------
TEST(ScatterOddWidth, SilentlyPerformsNoScatter) {
  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

  constexpr int32_t num_indices = 10;
  constexpr uint32_t width = 5;  // odd: matches none of the %16/%8/%4/%2 branches

  TestBuffer<int8_t> src(static_cast<size_t>(num_indices) * width);
  TestBuffer<int8_t> dst(static_cast<size_t>(num_indices) * width);
  TestBuffer<uint64_t> hit_mask(sizeof(uint64_t));
  std::memset(hit_mask.ph, 0, hit_mask.size_);  // all misses: a working scatter would copy all rows

  for (int32_t i = 0; i < num_indices; i++) {
    for (uint32_t b = 0; b < width; b++) {
      src.ph[i * width + b] = static_cast<int8_t>((i * 17 + b) & 0x7f);
      dst.ph[i * width + b] = kDstSentinel;
    }
  }
  src.HtoD(stream);
  dst.HtoD(stream);
  hit_mask.HtoD(stream);

  nve::EmbeddingForwardScatter(src.pd, dst.pd, width, width, width, hit_mask.pd, num_indices,
                               stream);

  dst.DtoH(stream);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

  for (int32_t i = 0; i < num_indices; i++) {
    for (uint32_t b = 0; b < width; b++) {
      EXPECT_EQ(dst.ph[i * width + b], kDstSentinel)
          << "index " << i << " byte " << b
          << ": odd-width scatter unexpectedly wrote data (no-op behavior changed)";
    }
  }
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}
