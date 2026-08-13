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

#pragma once

#include <stdint.h>
#include <type_traits>

#ifdef __CUDACC__
#define NVE_KEY_UTILS_CUDA_CALLABLE __host__ __device__
#else
#define NVE_KEY_UTILS_CUDA_CALLABLE
#endif

namespace nve {

// A linear (dense) table is indexed directly by key, so only keys in [0, num_rows) address a real
// row. Ops on such tables use this to drop out of range keys instead of touching foreign memory.
template <typename KeyT>
NVE_KEY_UTILS_CUDA_CALLABLE inline bool key_in_range(const KeyT key, const uint64_t num_rows) {
  static_assert(std::is_integral_v<KeyT>);
  if constexpr (std::is_signed_v<KeyT>) {
    if (key < 0) {
      return false;
    }
  }
  return static_cast<uint64_t>(key) < num_rows;
}

}  // namespace nve

#undef NVE_KEY_UTILS_CUDA_CALLABLE
