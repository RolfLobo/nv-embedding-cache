#!/usr/bin/python
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# End-to-end driverless inference test: exports a CPU HostLayer model from Python
# via AOTInductor, then runs the nve_inference_driverless C++ binary (built from
# driverless_inference.cu in this folder) against it. device_index=-1 throughout,
# no CUDA driver touched. C++ analogue of the Python-side
# test_host_layer_cpu_aot_export_load_roundtrip in tests/python_binding.

import pynve.torch.nve_layers as nve_layers
import pynve.torch.nve_export as nve_export
import os
import pytest
import subprocess
import tempfile
import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# Defaults to ./build/bin (where `PYNVE_BUILD_SAMPLES=1 pip install .` and a plain
# cmake build both drop it); override NVE_INFERENCE_DRIVERLESS_BIN for other layouts.
NVE_INFERENCE_DRIVERLESS_BIN = os.environ.get(
    "NVE_INFERENCE_DRIVERLESS_BIN",
    os.path.join(REPO_ROOT, "build/bin/nve_inference_driverless"))


def test_cpp_inference_driverless():
    """Export a CPU HostLayer model with known weights via AOTInductor, run the
    driverless C++ binary (device_index=-1 throughout, no CUDA driver touched),
    verify output."""
    if not os.path.exists(NVE_INFERENCE_DRIVERLESS_BIN):
        pytest.skip(f"C++ binary not found: {NVE_INFERENCE_DRIVERLESS_BIN}")

    num_embeddings = 1024
    embed_size = 8
    weight = (torch.arange(num_embeddings, dtype=torch.float32)
              .unsqueeze(1).expand(num_embeddings, embed_size).contiguous())

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nve_layers.NVEmbedding(
                num_embeddings, embed_size, torch.float32,
                layer_type=nve_layers.LayerType.HostLayer,
                weight_init=weight, optimize_for_training=False,
                device=torch.device("cpu"))

        def forward(self, keys):
            return self.emb(keys)

    keys = torch.tensor([0, 1, 5, 10], dtype=torch.int64)

    with tempfile.TemporaryDirectory() as save_dir:
        nve_export.export_aot(M(), (keys,), save_dir)

        env = os.environ.copy()
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        env["LD_LIBRARY_PATH"] = torch_lib + ":" + env.get("LD_LIBRARY_PATH", "")
        result = subprocess.run(
            [NVE_INFERENCE_DRIVERLESS_BIN, save_dir],
            capture_output=True, text=True, timeout=30,
            cwd=REPO_ROOT,
            env=env,
        )
        assert result.returncode == 0, f"C++ binary failed:\n{result.stderr}"

        stdout = result.stdout
        assert "Output shape: [4, 8]" in stdout
        assert "key=0 -> [0, 0, 0, 0" in stdout
        assert "key=1 -> [1, 1, 1, 1" in stdout
        assert "key=5 -> [5, 5, 5, 5" in stdout
        assert "key=10 -> [10, 10, 10, 10" in stdout
        print("PASS: driverless C++ inference output matches expected values")
