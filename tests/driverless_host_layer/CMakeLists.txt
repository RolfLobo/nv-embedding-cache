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

# Host-only embedding smoke test that must build and run on a driverless host. Links only
# nve-common (driver-free under NVE_DRIVERLESS_BUILD); uses malloc-backed host storage so it
# never touches the CUDA driver.
add_executable(nve_driverless_test host_layer_driverless_test.cpp)
target_link_libraries(nve_driverless_test PUBLIC gtest_main nve-common)

# End-to-end driverless C++ inference: loads a Python-exported AOTInductor
# NVEmbedding(HostLayer) package and runs it entirely at device_index=-1.
# Needs the torch custom-op bindings (nve-torch-ops), so it is only built when
# they are enabled. The `torch` umbrella target forces libtorch_cuda.so NEEDED via
# -Wl,--no-as-needed even when unused (see python/pynve/torch_bindings/CMakeLists.txt),
# so under NVE_DRIVERLESS_BUILD link only the CPU torch interface — nve-torch-ops /
# nve-common already drop their CUDA driver dependency in that configuration —
# giving a binary with no libcuda / libtorch_cuda dependency that loads on a
# driverless host.
if (NOT NVE_DISABLE_TORCH_BINDINGS AND TORCH_FOUND)
    add_executable(nve_inference_driverless driverless_inference.cu)

    target_include_directories(nve_inference_driverless SYSTEM PRIVATE
        ${TORCH_INCLUDE_DIRS}
    )

    target_link_libraries(nve_inference_driverless
        torch_cpu_library
        c10
        nve-torch-ops
        nve-common
    )

    add_executable(driverless_inference_test driverless_inference_test.cpp)
    target_link_libraries(driverless_inference_test PRIVATE gtest_main)
    target_compile_definitions(driverless_inference_test PRIVATE
        NVE_TEST_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}"
        NVE_TEST_INFERENCE_EXECUTABLE="$<TARGET_FILE:nve_inference_driverless>"
    )
    add_dependencies(driverless_inference_test nve_inference_driverless)
endif()
