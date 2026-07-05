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

# Unified layer-level pooling & dequantization suite. Reuses the embedding_layer test helpers
# (emb_layer_utils.cpp is recompiled per target, mirroring the kernels target) and the header-only
# MockHostTable reference.
add_executable(layer_pooling_test
    layer_pooling_test.cu
    ../embedding_layer/emb_layer_utils.cpp)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/tests/embedding_layer)  # emb_layer_utils.hpp, mock_host_table.hpp

target_link_libraries(layer_pooling_test PUBLIC gtest_main nve-common)

# HierWithoutGPU dynamically loads the PHMap plugin for its first-level host cache. Make building
# this test target sufficient to produce the plugin whenever that optional feature is enabled.
if(TARGET nve-plugin-phmap)
  add_dependencies(layer_pooling_test nve-plugin-phmap)
endif()
