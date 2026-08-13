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

# The plugin is plain C and only includes the NVE C headers — it must not
# link nve-common (that is the external, unknown-toolchain contract).
add_library(nve-plugin-external-map SHARED src/plugin.c)
set_target_properties(nve-plugin-external-map PROPERTIES C_STANDARD 11)

# The consumer talks to the plugin exclusively through the NVE C API.
add_executable(external_plugin_sample external_plugin_sample.cpp)
target_link_libraries(external_plugin_sample nve-common)
add_dependencies(external_plugin_sample nve-plugin-external-map)
