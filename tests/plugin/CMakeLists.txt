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

# The ABI headers must compile standalone as C11 and C++17 — the layout static
# assertions inside them are the actual test.
add_library(plugin_abi_c_conformance OBJECT abi_header_conformance.c)
set_target_properties(plugin_abi_c_conformance PROPERTIES C_STANDARD 11)
add_library(plugin_abi_cpp_conformance OBJECT abi_header_conformance.cpp)
set_target_properties(plugin_abi_cpp_conformance PROPERTIES CXX_STANDARD 17)

# Test-only external plugin with JSON-driven failure injection (pure C,
# dlopen-only, no NVE libraries — same contract as a real external plugin).
add_library(nve-test-plugin-fault MODULE fault_plugin.c)
set_target_properties(nve-test-plugin-fault PROPERTIES C_STANDARD 11)

add_executable(plugin_loader_tests plugin_loader_tests.cpp)
target_link_libraries(plugin_loader_tests PUBLIC
  gtest_main
  nve-common
)
add_dependencies(plugin_loader_tests
  nve-plugin-stl-map nve-plugin-external-map nve-test-plugin-fault)
target_compile_definitions(plugin_loader_tests PRIVATE
  NVE_STL_MAP_PLUGIN_PATH="$<TARGET_FILE:nve-plugin-stl-map>"
  NVE_EXTERNAL_MAP_PLUGIN_PATH="$<TARGET_FILE:nve-plugin-external-map>"
  NVE_FAULT_PLUGIN_PATH="$<TARGET_FILE:nve-test-plugin-fault>")
