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

message("NV Embedding Cache Plugin: SPH (GPU hash table)")

# The SPH engine is a STATIC lib (target `sph`) linked into this SHARED
# plugin, so its objects must be position-independent. Set this before pulling
# in the engine subdirectory so the property propagates to its targets.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(engine)  # -> sph (STATIC); exports engine/{include,src,support}

file(GLOB nve_plugin_sph_src
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(nve-plugin-sph SHARED ${nve_plugin_sph_src})
target_include_directories(nve-plugin-sph PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(nve-plugin-sph PUBLIC nve-common sph)
target_compile_options(nve-plugin-sph PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>)
# The library is named libnve-plugin-sph-experimental.so: SPH is not yet a
# stable backend, and the filename is what consumers pass to the plugin loader,
# so the marker is visible at the point of use. The CMake target keeps its plain
# name so `if(TARGET nve-plugin-sph)` and `$<TARGET_FILE:...>` references are
# unaffected.
set_target_properties(nve-plugin-sph PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  OUTPUT_NAME nve-plugin-sph-experimental)

if (NOT NVE_DISABLE_TESTS_AND_SAMPLES)
  add_subdirectory(tests)
endif()
