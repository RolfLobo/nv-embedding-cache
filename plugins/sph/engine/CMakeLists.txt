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

# ============================================================================
# The SPH hash-map engine.
#
# Public target:  sph  (STATIC)
# Public header:  #include <sph/sph_hash_map.h>   ->  sph::ShardedPerfectHashMap
#
#
# External deps: CUDA runtime + driver + cub (cub ships with CTK >= 11).
# ============================================================================
find_package(CUDAToolkit REQUIRED)

set(SPH_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

add_library(sph STATIC
    ${SPH_ROOT}/src/sph_hash_map.cu
)
# include/  -> PUBLIC: <sph/sph_hash_map.h>, <sph/error.h>, <sph/sph_geometry.h>.
#              This is the whole consumer-visible surface.
# src/      -> PRIVATE: internal .cuh/.h + sph_types.h.
# support/  -> PRIVATE: sph_check.h, deduper.cuh, resizeable_buffer.hpp.
target_include_directories(sph
    PUBLIC  ${SPH_ROOT}/include
    PRIVATE ${SPH_ROOT}/src
            ${SPH_ROOT}/support
)
target_compile_options(sph PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -lineinfo>
)
# curand is not used by the engine — only by the benchmark, which links it
# itself. cuda_driver is needed for the VMM calls in src/sph_types.h.
target_link_libraries(sph PUBLIC
    CUDA::cuda_driver
)
set_target_properties(sph PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
