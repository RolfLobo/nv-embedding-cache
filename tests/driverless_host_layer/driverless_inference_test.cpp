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

#include <array>
#include <string>

#include "tests/inference_test_utils.hpp"

namespace {

constexpr char kExportScript[] = R"PY(
import sys

import torch

import pynve.torch.nve_layers as nve_layers
from pynve.torch.nve_export import export_aot

save_dir = sys.argv[1]
num_embeddings = 1024
embedding_size = 8
weights = torch.arange(num_embeddings, dtype=torch.float32) \
    .unsqueeze(1).expand(num_embeddings, embedding_size).contiguous()


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nve_layers.NVEmbedding(
            num_embeddings,
            embedding_size,
            torch.float32,
            layer_type=nve_layers.LayerType.HostLayer,
            weight_init=weights,
            optimize_for_training=False,
            device=torch.device("cpu"),
        )

    def forward(self, keys):
        return self.emb(keys)


keys = torch.tensor([0, 1, 5, 10], dtype=torch.int64)
export_aot(Model(), (keys,), save_dir)
)PY";

TEST(NVEDriverlessInferenceTest, ExportAndRunCppInference) {
  const nve_test::TemporaryDirectory export_directory("nve-driverless-inference-test");
  const std::string save_dir = export_directory.path().string();

  const nve_test::ProcessResult export_result = nve_test::run_process(
      {NVE_TEST_PYTHON_EXECUTABLE, "-c", kExportScript, save_dir});
  ASSERT_EQ(export_result.exit_code, 0)
      << "Python export failed:\n" << export_result.output;

  const nve_test::ProcessResult inference_result =
      nve_test::run_process({NVE_TEST_INFERENCE_EXECUTABLE, save_dir});
  ASSERT_EQ(inference_result.exit_code, 0)
      << "C++ inference failed:\n" << inference_result.output;

  constexpr std::array<const char*, 5> kExpectedOutput{
      "Output shape: [4, 8]", "key=0 -> [0, 0, 0, 0", "key=1 -> [1, 1, 1, 1",
      "key=5 -> [5, 5, 5, 5", "key=10 -> [10, 10, 10, 10"};
  for (const char* expected : kExpectedOutput) {
    EXPECT_NE(inference_result.output.find(expected), std::string::npos)
        << "Missing output: " << expected
        << "\nProcess output:\n" << inference_result.output;
  }
}

}  // namespace
