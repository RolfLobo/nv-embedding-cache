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

# Functional tests for the standalone SPH engine (sph), driving its public
# API directly with no plugin layer in between.
add_executable(sph_engine_tests sph_engine_tests.cu)
target_link_libraries(sph_engine_tests PUBLIC
  gtest_main
  sph
  CUDA::cudart
)
# sph is a -rdc static lib; the consuming executable does the device link.
set_target_properties(sph_engine_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(sph_plugin_tests sph_plugin_tests.cpp)
target_link_libraries(sph_plugin_tests PUBLIC
  gtest_main
  nve-plugin-sph
  nve-common
  CUDA::cudart
)

add_executable(sph_plugin_loader_tests sph_plugin_loader_tests.cpp)
target_link_libraries(sph_plugin_loader_tests PUBLIC
  gtest_main
  nve-common
  CUDA::cudart
)
target_compile_definitions(sph_plugin_loader_tests PRIVATE
  NVE_SPH_PLUGIN_PATH="$<TARGET_FILE:nve-plugin-sph>")
add_dependencies(sph_plugin_loader_tests nve-plugin-sph)
