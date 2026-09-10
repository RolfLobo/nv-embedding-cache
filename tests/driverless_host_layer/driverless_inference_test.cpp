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

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "tests/inference_test_utils.hpp"

namespace {

constexpr char kExportScript[] = R"PY(
import sys

import torch

import pynve.torch.nve_layers as nve_layers
from pynve.torch.nve_export import export_aot

save_dir, layer_kind = sys.argv[1:3]
num_embeddings = 1024
embedding_size = 8
weights = torch.arange(num_embeddings, dtype=torch.float32) \
    .unsqueeze(1).expand(num_embeddings, embedding_size).contiguous()
kwargs = dict(
    layer_type=nve_layers.LayerType.HostLayer,
    weight_init=weights,
    optimize_for_training=False,
    device=torch.device("cpu"),
)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nve_layers.NVEmbedding(
            num_embeddings, embedding_size, torch.float32, **kwargs)

    def forward(self, keys):
        return self.emb(keys)


class BagModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nve_layers.NVEmbeddingBag(
            num_embeddings, embedding_size, torch.float32, mode="sum", **kwargs)

    def forward(self, keys, offsets):
        return self.emb(keys, offsets)


keys = torch.tensor([0, 1, 5, 10], dtype=torch.int64)
if layer_kind == "embedding":
    export_aot(Model(), (keys,), save_dir)
elif layer_kind == "bag":
    offsets = torch.tensor([0, 2, 4], dtype=torch.int64)
    export_aot(BagModel(), (keys, offsets), save_dir)
else:
    raise ValueError(f"Unknown layer kind: {layer_kind}")
)PY";

struct DriverlessCase {
  const char* layer_kind;
  const char* inference_flag;  // nullptr when the model takes keys only
  std::vector<const char*> expected_output;
};

// keys {0,1,5,10}; the bag model pools {0,1} and {5,10} with sum.
const std::vector<DriverlessCase> kCases{
    {"embedding", nullptr,
     {"Output shape: [4, 8]", "key=0 -> [0, 0, 0, 0", "key=1 -> [1, 1, 1, 1",
      "key=5 -> [5, 5, 5, 5", "key=10 -> [10, 10, 10, 10"}},
    {"bag", "--pooled",
     {"Output shape: [2, 8]", "bag=0 -> [1, 1, 1, 1", "bag=1 -> [15, 15, 15, 15"}},
};

class NVEDriverlessInferenceTest : public ::testing::TestWithParam<DriverlessCase> {};

TEST_P(NVEDriverlessInferenceTest, ExportAndRunCppInference) {
  const DriverlessCase& c = GetParam();
  const nve_test::TemporaryDirectory export_directory("nve-driverless-inference-test");
  const std::string save_dir = export_directory.path().string();

  const nve_test::ProcessResult export_result = nve_test::run_process(
      {NVE_TEST_PYTHON_EXECUTABLE, "-c", kExportScript, save_dir, c.layer_kind});
  ASSERT_EQ(export_result.exit_code, 0)
      << "Python export failed:\n" << export_result.output;

  std::vector<std::string> inference_args{NVE_TEST_INFERENCE_EXECUTABLE, save_dir};
  if (c.inference_flag != nullptr) {
    inference_args.push_back(c.inference_flag);
  }
  const nve_test::ProcessResult inference_result = nve_test::run_process(inference_args);
  ASSERT_EQ(inference_result.exit_code, 0)
      << "C++ inference failed:\n" << inference_result.output;

  nve_test::expect_output_contains(inference_result.output, c.expected_output);
}

INSTANTIATE_TEST_SUITE_P(LayerKinds, NVEDriverlessInferenceTest, ::testing::ValuesIn(kCases),
                         [](const ::testing::TestParamInfo<DriverlessCase>& info) {
                           return std::string(info.param.layer_kind);
                         });

}  // namespace
