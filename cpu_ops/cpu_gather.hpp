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

#include "include/key_utils.hpp"
#include "include/common.hpp"
#include "include/thread_pool.hpp"

#include <atomic>

namespace nve {

template<typename IndexT>
int64_t cpu_kernel_gather(thread_pool_ptr_t thread_pool,
              uint64_t n,
              const IndexT* keys, 
              bitmask64_t* hit_mask,
              size_t value_stride, 
              void* values, 
              int8_t* uvm_table_ptr,
              size_t row_size_in_bytes,
              uint64_t num_rows,
              uint64_t num_threads)
{
    constexpr uint64_t num_bits_in_hit_mask = sizeof(bitmask64_t) * 8;
    auto keys_per_task{ceil_div(n, num_threads)};
    // Align to 64. Relied by the task below that accesses 64bit mask parts concurrently without synchronizing.
    keys_per_task = round_up(keys_per_task, num_bits_in_hit_mask);
    std::atomic<int64_t> resolved_hits{0};

    const auto gather_task = [=, &resolved_hits] (const size_t idx) {
        const auto base_key = (idx) * keys_per_task;
        const auto base_mask_idx = base_key / num_bits_in_hit_mask;
        int64_t task_hits = 0;

        for (uint64_t i = 0; i < keys_per_task; i++) {
            if (base_key + i >= n) {
                break;
            }
            const auto mask_idx = base_mask_idx + (i / num_bits_in_hit_mask);
            const bitmask64_t key_bit{bitmask64::single(static_cast<int64_t>(i) % bitmask64::num_bits)};

            // A null hit mask means there are no pre-existing hits to skip.
            if ((hit_mask != nullptr) && ((hit_mask[mask_idx] & key_bit) != 0)) {
                continue;
            }
            IndexT key = reinterpret_cast<const IndexT*>(keys)[base_key + i];
            if (!key_in_range(key, num_rows)) {
                continue;
            }
            int8_t* src_ptr = uvm_table_ptr + (static_cast<size_t>(key) * row_size_in_bytes);
            int8_t* dst_ptr = reinterpret_cast<int8_t*>(values) + (base_key + i) * value_stride;
            std::memcpy(dst_ptr, src_ptr, row_size_in_bytes);
            ++task_hits;
            if (hit_mask != nullptr) {
                // No synchronization needed, mask words are partitioned between tasks (see above).
                hit_mask[mask_idx] |= key_bit;
            }
        }
        resolved_hits.fetch_add(task_hits, std::memory_order_relaxed);
    };

    thread_pool->execute_n(0, static_cast<int64_t>(num_threads), gather_task);
    return resolved_hits.load(std::memory_order_relaxed);
}

template<typename IndexT>
int64_t cpu_kernel_gather_dispatch(thread_pool_ptr_t thread_pool,
              uint64_t n,
              const IndexT* keys, 
              bitmask64_t* hit_mask,
              size_t value_stride, 
              void* values, 
              int8_t* uvm_table_ptr,
              size_t row_size_in_bytes,
              uint64_t num_rows,
              uint64_t num_threads)
{
    return cpu_kernel_gather<IndexT>(thread_pool, n, keys, hit_mask, value_stride, values,
                                     uvm_table_ptr, row_size_in_bytes, num_rows, num_threads);
}

}  // namespace nve
