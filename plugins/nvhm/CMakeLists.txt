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

message("NV Embedding Cache Plugin: nvHashMap")

include_directories(
  ${NVE_THIRD_PARTY_DIR}/nvhashmap/include
  ${CMAKE_BINARY_DIR}/include  # Needed for nvhashmap_conf.hpp, which is created during CMake.
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src  # Needed for nvhm_map_table_impl.hpp in the generated TUs.
)

# Generate the split NvhmMapTable explicit instantiations (mirrors the FindAndCombine pattern in
# src/CMakeLists.txt). See cmake/gen_nvhm_map_table_instantiations.py for details.
set(NVHM_GEN_SCRIPT ${PROJECT_SOURCE_DIR}/cmake/gen_nvhm_map_table_instantiations.py)
set(NVHM_GEN_DIR ${CMAKE_BINARY_DIR}/plugins/nvhm/generated)
set(NVHM_NUM_TUS "" CACHE STRING "Override number of NvhmMapTable generated .cpp files (blank = script default)")
if(NVHM_NUM_TUS)
  set(_nvhm_num_tus_arg --num-tus ${NVHM_NUM_TUS})
endif()

execute_process(
  COMMAND ${Python3_EXECUTABLE} ${NVHM_GEN_SCRIPT} --out-dir ${NVHM_GEN_DIR}
          --features "${NVE_FEATURES}" ${_nvhm_num_tus_arg}
  RESULT_VARIABLE _nvhm_rc)
if(NOT _nvhm_rc EQUAL 0)
  message(FATAL_ERROR "NvhmMapTable instantiation generation failed (${_nvhm_rc})")
endif()
include(${NVHM_GEN_DIR}/nvhm_map_table_generated.cmake)  # -> NVHM_GENERATED_SOURCES

# Re-run the generator (via reconfigure) when the script changes. Feature-set changes already
# trigger a reconfigure through the NVE_FEATURES cache variable.
set_property(DIRECTORY ${CMAKE_SOURCE_DIR} APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  ${NVHM_GEN_SCRIPT})

# The generated sources already exist on disk (written by execute_process above), so they are
# deliberately NOT marked GENERATED: that property makes the Ninja generator emit a no-op build
# edge per file, which `ninja -t clean` would delete without being able to regenerate.

file(GLOB nve_plugin_nvhm_src "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(nve-plugin-nvhm SHARED ${nve_plugin_nvhm_src} ${NVHM_GENERATED_SOURCES})
target_link_libraries(nve-plugin-nvhm PUBLIC nve-common)

if(NOT NVE_DISABLE_TESTS_AND_SAMPLES)
  add_subdirectory(tests)
endif()
