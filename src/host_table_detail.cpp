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

#include "host_table_detail.hpp"

namespace nve {

template <typename V, typename U>
static void cpu_update_kernel(
  void* const values_dst, const void* const values_src, int64_t value_size,
  const void* const updates, int64_t update_size) noexcept {
  using tmp_type =
      std::conditional_t<(sizeof(V) > sizeof(float) || sizeof(U) > sizeof(float)), double, float>;

  V* __restrict const dst{static_cast<V*>(values_dst)};
  const V* __restrict const src{static_cast<const V*>(values_src)};
  const U* __restrict const upd{static_cast<const U*>(updates)};

  NVE_ASSERT_(reinterpret_cast<uintptr_t>(dst) % alignof(V) == 0);
  NVE_ASSERT_(reinterpret_cast<uintptr_t>(src) % alignof(V) == 0);
  NVE_ASSERT_(reinterpret_cast<uintptr_t>(upd) % alignof(U) == 0);

  NVE_ASSERT_(to_uint(value_size) % sizeof(V) == 0);
  NVE_ASSERT_(to_uint(update_size) % sizeof(U) == 0);
  value_size /= static_cast<int64_t>(sizeof(V));
  update_size /= static_cast<int64_t>(sizeof(U));

  const int64_t n{std::min(value_size, update_size)};

  // Avoid undefined behavior, but retain auto vectorizations from restrict.
  NVE_ASSERT_(
    static_cast<const void*>(&upd[update_size]) <= static_cast<void*>(dst) ||
    static_cast<const void*>(upd) >= static_cast<const void*>(&dst[value_size]));
  if NVE_LIKELY_(src == dst) {
    for (int64_t i{}; i < n; ++i) {
      const tmp_type s{dst[i]};
      const tmp_type u{upd[i]};
      dst[i] = static_cast<V>(s + u);
    }
  } else {
    NVE_ASSERT_(
      static_cast<const void*>(&src[value_size]) <= static_cast<void*>(dst) ||
      static_cast<const void*>(src) >= static_cast<const void*>(&dst[value_size]));

    for (int64_t i{}; i < n; ++i) {
      const tmp_type s{src[i]};
      const tmp_type u{upd[i]};
      dst[i] = static_cast<V>(s + u);
    }

    if (n < value_size) {
      std::memcpy(&dst[n], &src[n], to_uint(value_size - n) * sizeof(V));
    }
  }
}

template <typename Value>
static inline update_kernel_t pick_cpu_update_kernel(const DataType_t update_dtype) {
  // TODO: Add specializations for x86 and ARM low precision instruction sets?
  switch (update_dtype) {
    case DataType_t::Float32:
      return cpu_update_kernel<Value, type_t<DataType_t::Float32>>;
    case DataType_t::Float16:
      return cpu_update_kernel<Value, type_t<DataType_t::Float16>>;
    case DataType_t::BFloat:
      return cpu_update_kernel<Value, type_t<DataType_t::BFloat>>;
    case DataType_t::E4M3:
      return cpu_update_kernel<Value, type_t<DataType_t::E4M3>>;
    case DataType_t::E5M2:
      return cpu_update_kernel<Value, type_t<DataType_t::E5M2>>;
    case DataType_t::Float64:
      return cpu_update_kernel<Value, type_t<DataType_t::Float64>>;
    default:
      NVE_THROW_("Combining data-types (table = ", data_type<Value>(),
                 " and update =", update_dtype, ") is currently not supported!");
  }
}

update_kernel_t pick_cpu_update_kernel(const DataType_t value_dtype, const DataType_t update_dtype) {
  switch (value_dtype) {
    case DataType_t::Float32:
      return pick_cpu_update_kernel<type_t<DataType_t::Float32>>(update_dtype);
    case DataType_t::Float16:
      return pick_cpu_update_kernel<type_t<DataType_t::Float16>>(update_dtype);
    case DataType_t::BFloat:
      return pick_cpu_update_kernel<type_t<DataType_t::BFloat>>(update_dtype);
    case DataType_t::E4M3:
      return pick_cpu_update_kernel<type_t<DataType_t::E4M3>>(update_dtype);
    case DataType_t::E5M2:
      return pick_cpu_update_kernel<type_t<DataType_t::E5M2>>(update_dtype);
    case DataType_t::Float64:
      return pick_cpu_update_kernel<type_t<DataType_t::Float64>>(update_dtype);
    default:
      NVE_THROW_("Combining data-types (value = ", value_dtype, " and update =", update_dtype,
                 ") is currently not supported!");
  }
}

}  // namespace nve
