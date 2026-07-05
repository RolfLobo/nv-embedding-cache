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

// Driverless host-layer smoke test — the C++ analogue of tests/python_binding/test_host_cpu.py.
//
// Exercises a HostEmbeddingLayer over a LinearHostTable using plain malloc-backed storage
// (std::vector, NOT cudaMallocHost) and the default allocator/thread pool. On a driverless host
// GetDefaultAllocator() falls back to malloc and the execution context skips cudaStreamSynchronize,
// so this runs end-to-end with no CUDA driver / GPU. Built into its own executable that links only
// nve-common (driver-free in a NVE_DRIVERLESS_BUILD), so it stays loadable on a driverless machine.

#include <gtest/gtest.h>

#include "include/common.hpp"
#include "include/host_embedding_layer.hpp"
#include "include/linear_host_table.hpp"
#include "include/nve_types.hpp"

#include <memory>
#include <numeric>
#include <vector>

namespace nve {
namespace {

using KeyType = int64_t;
constexpr int64_t kRowBytes = 16;  // 4 x float32
constexpr int64_t kNumRows = 256;
constexpr size_t kElems = static_cast<size_t>(kRowBytes) / sizeof(float);

// Build a HostEmbeddingLayer backed by a caller-owned malloc buffer.
struct HostLayerFixture {
  std::vector<float> backing;
  std::shared_ptr<LinearHostTable<KeyType>> table;
  std::shared_ptr<HostEmbeddingLayer<KeyType>> layer;
  context_ptr_t ctx;

  HostLayerFixture() : backing(static_cast<size_t>(kNumRows) * kElems) {
    // Distinct, recognizable per-element values for ground-truth checks.
    std::iota(backing.begin(), backing.end(), 1000.0f);

    LinearHostTableConfig cfg;
    cfg.value_dtype = DataType_t::Float32;
    cfg.max_threads = 8;
    cfg.max_value_size = kRowBytes;
    cfg.emb_table = backing.data();

    table = std::make_shared<LinearHostTable<KeyType>>(cfg);

    HostEmbeddingLayer<KeyType>::Config layer_cfg;
    layer_cfg.layer_name = "driverless_host_layer";
    layer = std::make_shared<HostEmbeddingLayer<KeyType>>(layer_cfg, table);

    // Null stream + default allocator/thread pool: driverless-safe path.
    ctx = layer->create_execution_context(/*lookup_stream=*/0, /*modify_stream=*/0,
                                          /*thread_pool=*/nullptr, /*allocator=*/nullptr);
  }
};

TEST(DriverlessHostLayer, LookupReturnsBackingRows) {
  HostLayerFixture f;
  const std::vector<KeyType> keys{0, 5, 42, 200};
  std::vector<float> out(keys.size() * kElems, 0.0f);

  f.layer->lookup(f.ctx, static_cast<int64_t>(keys.size()), keys.data(), out.data(),
                  kRowBytes, /*hitmask=*/nullptr, /*pool_params=*/nullptr, /*hitrates=*/nullptr);

  for (size_t k = 0; k < keys.size(); ++k) {
    for (size_t e = 0; e < kElems; ++e) {
      const float expected = f.backing[static_cast<size_t>(keys[k]) * kElems + e];
      EXPECT_FLOAT_EQ(out[k * kElems + e], expected) << "key=" << keys[k] << " elem=" << e;
    }
  }
}

TEST(DriverlessHostLayer, UpdateThenLookup) {
  HostLayerFixture f;
  constexpr KeyType key = 77;
  std::vector<float> row(kElems);
  std::iota(row.begin(), row.end(), -5.0f);  // {-5, -4, -3, -2}

  f.layer->update(f.ctx, /*num_keys=*/1, &key, /*value_stride=*/kRowBytes,
                  /*value_size=*/kRowBytes, row.data(), /*table_id=*/0);

  std::vector<float> out(kElems, 0.0f);
  f.layer->lookup(f.ctx, /*num_keys=*/1, &key, out.data(), kRowBytes,
                  /*hitmask=*/nullptr, /*pool_params=*/nullptr, /*hitrates=*/nullptr);

  for (size_t e = 0; e < kElems; ++e) {
    EXPECT_FLOAT_EQ(out[e], row[e]) << "elem=" << e;
  }
}

}  // namespace
}  // namespace nve
