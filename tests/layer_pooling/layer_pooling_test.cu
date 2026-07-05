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

// Layer-level pooling & dequantization suite.
//
// One value-parameterized test (LookupVsReference) drives every generated LayerPoolCase through a
// real layer's lookup() with PoolingParams and compares the result against MockHostTable::combine().
// Cases are grouped into one INSTANTIATE_TEST_SUITE_P per layer so CI can shard with
// --gtest_filter (e.g. 'LinearUVM/*') or GTEST_TOTAL_SHARDS/GTEST_SHARD_INDEX. Planned-but-
// unimplemented cross-precision CUDA dequant output combinations live under DISABLED_
// instantiations; run them with --gtest_also_run_disabled_tests as support lands.

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <vector>

#include "layer_pooling_harness.hpp"

namespace nve {
namespace layer_pooling {
namespace {

std::vector<LayerPoolCase> CasesFor(LayerKind layer, CaseStatus status) {
  const GeneratedCases g = GenerateCases();
  const std::vector<LayerPoolCase>& src = (status == CaseStatus::Enabled) ? g.enabled : g.disabled;
  std::vector<LayerPoolCase> out;
  for (const auto& c : src) {
    if (c.layer == layer) out.push_back(c);
  }
  return out;
}

class LayerPoolingTest : public ::testing::TestWithParam<LayerPoolCase> {};

TEST(LayerPoolingValidationTest, UnknownOutputTypeRejected) {
  EmbeddingLayerBase::PoolingParams pp;
  // Argument validation throws the dedicated InvalidArgumentError so API boundaries (e.g. the C
  // API) can classify caller errors separately from internal runtime failures.
  EXPECT_THROW(validate_pool_params(pp), InvalidArgumentError);
}

TEST(LayerPoolingValidationTest, MalformedPoolParamsThrowInvalidArgument) {
  EmbeddingLayerBase::PoolingParams pp;
  pp.pooling_type = PoolingType_t::Sum;
  pp.output_type = DataType_t::Float32;
  pp.sparse_type = SparseType_t::Fixed;
  pp.fixed_hotness = 0;
  EXPECT_THROW(validate_pool_params(pp), InvalidArgumentError);

  pp.fixed_hotness = 3;
  EXPECT_THROW(get_lookup_output_rows(11, &pp), InvalidArgumentError);

  pp.pooling_type = PoolingType_t::WeightedSum;
  pp.weights = nullptr;
  EXPECT_THROW(validate_pool_params(pp), InvalidArgumentError);

  pp.pooling_type = PoolingType_t::Sum;
  pp.sparse_type = SparseType_t::CSR;
  pp.csr_offsets = nullptr;
  EXPECT_THROW(validate_pool_params(pp), InvalidArgumentError);
}

TEST(LayerPoolingValidationTest, ConcatenateIgnoresBagAndWeightMetadata) {
  double ignored_weight = 0.0;
  EmbeddingLayerBase::PoolingParams pp;
  pp.pooling_type = PoolingType_t::Concatenate;
  pp.sparse_type = static_cast<SparseType_t>(2);
  pp.csr_offsets = nullptr;
  pp.num_csr_offsets = -1;
  pp.fixed_hotness = -1;
  pp.weights = &ignored_weight;
  pp.weight_type = DataType_t::Float64;
  pp.output_type = DataType_t::Float32;

  EXPECT_NO_THROW(validate_pool_params(pp));
}

TEST(LayerPoolingValidationTest, FixedHotnessDoesNotRequireCsrOffsets) {
  EmbeddingLayerBase::PoolingParams pp;
  pp.pooling_type = PoolingType_t::Sum;
  pp.sparse_type = SparseType_t::Fixed;
  pp.fixed_hotness = 2;
  pp.output_type = DataType_t::Float32;

  EXPECT_NO_THROW(validate_pool_params(pp));
}

TEST(LayerPoolingValidationTest, LookupOutputRowsFollowPoolingLayout) {
  EXPECT_EQ(get_lookup_output_rows(12, nullptr), 12);

  EmbeddingLayerBase::PoolingParams pp;
  pp.pooling_type = PoolingType_t::Concatenate;
  pp.sparse_type = static_cast<SparseType_t>(2);
  pp.fixed_hotness = -1;
  pp.num_csr_offsets = -1;
  EXPECT_EQ(get_lookup_output_rows(12, &pp), 12);

  pp.pooling_type = PoolingType_t::Sum;
  pp.sparse_type = SparseType_t::Fixed;
  pp.fixed_hotness = 3;
  EXPECT_EQ(get_lookup_output_rows(12, &pp), 4);
  EXPECT_THROW(get_lookup_output_rows(11, &pp), Exception);

  pp.sparse_type = SparseType_t::CSR;
  const int64_t offsets[] = {0, 3, 6, 9, 12};
  pp.csr_offsets = offsets;
  pp.num_csr_offsets = 5;
  EXPECT_EQ(get_lookup_output_rows(12, &pp), 4);
}

// Empty CSR bags (and WeightedMean bags whose weights sum to zero) must produce zero output rows
// on every layer, matching the CPU kernels and torch's EmbeddingBag, instead of a division-by-zero
// NaN on the CUDA combine path.
TEST(LayerPoolingValidationTest, EmptyCsrBagsZeroFill) {
  const LayerKind layers[] = {LayerKind::GPU, LayerKind::LinearUVM, LayerKind::HierWithGPU,
                              LayerKind::Host, LayerKind::HierWithoutGPU};
  const PoolingType_t pools[] = {PoolingType_t::Mean, PoolingType_t::WeightedMean};
  const DataType_t in_dtypes[] = {DataType_t::Float32, DataType_t::QInt8RowwiseF32};

  for (LayerKind layer : layers) {
    if (layer != LayerKind::Host && layer != LayerKind::HierWithoutGPU && !gpu_available()) {
      continue;
    }
#ifndef NVE_FEATURE_PHMAP_PLUGIN
    if (layer == LayerKind::HierWithoutGPU) {
      continue;
    }
#endif
    for (PoolingType_t pool : pools) {
      for (DataType_t in_dtype : in_dtypes) {
        LayerPoolCase c;
        c.layer        = layer;
        c.pooling      = pool;
        c.sparse       = SparseType_t::CSR;
        c.in_dtype     = in_dtype;
        c.out_dtype    = DataType_t::Float32;
        c.weight_dtype = DataType_t::Float32;
        c.key_is_int64 = true;
        c.value_count  = 16;
        c.hotness      = 3;
        c.num_keys     = 12;
        c.num_rows     = 256;
        SCOPED_TRACE(CaseLabel(c));
        RunEmptyBagCsrCase(c);
      }
    }
  }
}

// A padded output stride is a hard error on the GPU-backed layers (their pool/dequant kernels
// derive the embedding width from the stride) and a supported layout on the CPU pooling path.
// Covers both the float and the rowwise-quantized dequant paths.
TEST(LayerPoolingValidationTest, PaddedOutputStridePerLayer) {
  const LayerKind layers[] = {LayerKind::GPU, LayerKind::LinearUVM, LayerKind::HierWithGPU,
                              LayerKind::Host, LayerKind::HierWithoutGPU};
  const DataType_t in_dtypes[] = {DataType_t::Float32, DataType_t::QInt8RowwiseF32};

  for (LayerKind layer : layers) {
    if (layer != LayerKind::Host && layer != LayerKind::HierWithoutGPU && !gpu_available()) {
      continue;
    }
#ifndef NVE_FEATURE_PHMAP_PLUGIN
    if (layer == LayerKind::HierWithoutGPU) {
      continue;
    }
#endif
    for (DataType_t in_dtype : in_dtypes) {
      LayerPoolCase c;
      c.layer        = layer;
      c.pooling      = PoolingType_t::Sum;
      c.sparse       = SparseType_t::Fixed;
      c.in_dtype     = in_dtype;
      c.out_dtype    = DataType_t::Float32;
      c.weight_dtype = DataType_t::Unknown;
      c.key_is_int64 = true;
      c.value_count  = 16;
      c.hotness      = 4;
      c.num_keys     = 64;
      c.num_rows     = 256;
      SCOPED_TRACE(CaseLabel(c));
      RunPaddedStrideCase(c);
    }
  }
}

TEST_P(LayerPoolingTest, LookupVsReference) {
  cudaGetLastError();  // Clear any sticky error left by a previous test.
  const LayerPoolCase& c = GetParam();
  // Host and HierWithoutGPU run entirely on the CPU pooling path (no GPU kernels), so they are not
  // gated on a CUDA device beyond the harness's pinned-host allocations.
  if (c.layer != LayerKind::Host && c.layer != LayerKind::HierWithoutGPU && !gpu_available()) {
    GTEST_SKIP() << "No CUDA device available for " << c;
  }
#ifndef NVE_FEATURE_PHMAP_PLUGIN
  if (c.layer == LayerKind::HierWithoutGPU) {
    GTEST_SKIP() << "The HierWithoutGPU test topology requires the PHMap plugin";
  }
#endif
  if (const char* reason = DisabledReason(c)) {
    // Reached only via --gtest_also_run_disabled_tests. Surface the documented capability gap.
    GTEST_LOG_(INFO) << "Known gap (DISABLED): " << reason;
  }
  RunCase(c);
}

// Enabled cases: one instantiation per layer so CI shards by prefix (GPU/*, LinearUVM/*,
// HierWithGPU/*, Host/*, HierWithoutGPU/*) or via GTEST_TOTAL_SHARDS/GTEST_SHARD_INDEX.
INSTANTIATE_TEST_SUITE_P(GPU, LayerPoolingTest,
                         ::testing::ValuesIn(CasesFor(LayerKind::GPU, CaseStatus::Enabled)),
                         CaseName());
INSTANTIATE_TEST_SUITE_P(LinearUVM, LayerPoolingTest,
                         ::testing::ValuesIn(CasesFor(LayerKind::LinearUVM, CaseStatus::Enabled)),
                         CaseName());
INSTANTIATE_TEST_SUITE_P(HierWithGPU, LayerPoolingTest,
                         ::testing::ValuesIn(CasesFor(LayerKind::HierWithGPU, CaseStatus::Enabled)),
                         CaseName());
INSTANTIATE_TEST_SUITE_P(Host, LayerPoolingTest,
                         ::testing::ValuesIn(CasesFor(LayerKind::Host, CaseStatus::Enabled)),
                         CaseName());
INSTANTIATE_TEST_SUITE_P(HierWithoutGPU, LayerPoolingTest,
                         ::testing::ValuesIn(CasesFor(LayerKind::HierWithoutGPU, CaseStatus::Enabled)),
                         CaseName());

// Known capability gaps: compiled and listed, but disabled (the DISABLED_ prefix on each
// instantiation disables the whole group) so they do not fail CI. Run with
// --gtest_also_run_disabled_tests to track them; promote to Enabled in Classify() when support
// lands. See README.md for the per-layer reasons. (Host has no gaps, so no DISABLED_Host group.)
INSTANTIATE_TEST_SUITE_P(DISABLED_GPU, LayerPoolingTest,
                         ::testing::ValuesIn(CasesFor(LayerKind::GPU, CaseStatus::Disabled)),
                         CaseName());
INSTANTIATE_TEST_SUITE_P(DISABLED_LinearUVM, LayerPoolingTest,
                         ::testing::ValuesIn(CasesFor(LayerKind::LinearUVM, CaseStatus::Disabled)),
                         CaseName());
INSTANTIATE_TEST_SUITE_P(DISABLED_HierWithGPU, LayerPoolingTest,
                         ::testing::ValuesIn(CasesFor(LayerKind::HierWithGPU, CaseStatus::Disabled)),
                         CaseName());

}  // namespace
}  // namespace layer_pooling
}  // namespace nve
