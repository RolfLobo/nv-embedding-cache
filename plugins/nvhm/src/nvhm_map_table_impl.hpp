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

// Template member definitions of NvhmMapTable. Included only by the generated
// nvhm_map_table_inst_*.cpp translation units (see cmake/gen_nvhm_map_table_instantiations.py),
// which explicitly instantiate the class so the instantiation matrix compiles across many TUs
// instead of one. Keep the generator's combination space in sync with the factory dispatch chain
// in nvhm_map_table.cpp.

#pragma once

#include <random>
#include <execution_context.hpp>
#include <host_table_detail.hpp>
#include <nvhm_map_table.hpp>
#include <buffer_wrapper.hpp>
#include <thread_pool.hpp>

namespace nve {
namespace plugin {

// TODO: Should we enable optimistic prefetch?
inline constexpr bool use_optimistic_prefetch{false};

template <typename MapType, typename PartitionerType>
NvhmMapTable<MapType, PartitionerType>::NvhmMapTable(
    const table_id_t id, const NvhmMapTableConfig& config)
    : base_type(id, config), parts_(static_cast<uint64_t>(config.num_partitions)) {
  NVE_CHECK_(config.key_size == sizeof(key_type));
  for (auto& part : parts_) {
    part.map = map_type(static_cast<uint64_t>(config.initial_capacity),
                        static_cast<uint64_t>(config.max_value_size),
                        static_cast<uint64_t>(config.value_alignment));
  }
}

template <typename MapType, typename PartitionerType>
void NvhmMapTable<MapType, PartitionerType>::clear(context_ptr_t& ctx) {
  const auto& __restrict config{config_};

  std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};

  const auto f{[&parts](const int64_t task_idx) {
    Partition& __restrict part{parts[static_cast<uint64_t>(task_idx)]};
    std::lock_guard lock(part.read_write);

    part.map.clear();
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config.workgroups, 1);
}

template <typename MapType, typename PartitionerType>
void NvhmMapTable<MapType, PartitionerType>::erase(context_ptr_t& ctx, const int64_t n,
                                                             buffer_ptr<const void> keys_bw) {
  if (n <= 0) return;
  const auto& __restrict config{config_};

  const void* const keys_vptr{
      keys_bw ? keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/,
                                       ctx->get_modify_stream())
              : nullptr};
  const key_type* const __restrict keys{reinterpret_cast<const key_type*>(keys_vptr)};

  std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};
  const int64_t num_parts_mask{num_parts - 1};

  const auto f{[n, keys, &parts, num_parts_mask](const int64_t task_idx) {
    Partition& __restrict part{parts[static_cast<uint64_t>(task_idx)]};
    std::lock_guard lock(part.read_write);

    for (int64_t i{}; i != n; ++i) {
      const key_type key{keys[i]};
      if (partitioner(key, num_parts_mask) != task_idx) continue;

      part.map.erase(key);
    }
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config.workgroups, 1);
}

template <typename MapType, typename PartitionerType>
void NvhmMapTable<MapType, PartitionerType>::find(
    context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys_bw,
    buffer_ptr<bitmask64_t> hit_mask_bw, const int64_t value_stride,
    buffer_ptr<void> values_bw, buffer_ptr<int64_t> value_sizes_bw) const {
  if (n <= 0) return;
  const auto& __restrict config{config_};

  auto lookup_stream = ctx->get_lookup_stream();
  const void* const keys_vptr{
      keys_bw ? keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, lookup_stream)
              : nullptr};
  bitmask64_t* const hit_mask{
      hit_mask_bw ? hit_mask_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/,
                                               lookup_stream)
                  : nullptr};
  void* const values_vptr{
      values_bw ? values_bw->access_buffer(cudaMemoryTypeUnregistered, false /*copy_content*/,
                                           lookup_stream)
                : nullptr};
  int64_t* const value_sizes{
      value_sizes_bw ? value_sizes_bw->access_buffer(cudaMemoryTypeUnregistered,
                                                     false /*copy_content*/, lookup_stream)
                     : nullptr};
  const key_type* const keys{reinterpret_cast<const key_type*>(keys_vptr)};
  bitmask64_t* const hm{reinterpret_cast<bitmask64_t*>(hit_mask)};
  char* const values{reinterpret_cast<char*>(values_vptr)};

  if (config.prefetch_values) {
    switch (config.key_fetch_queue_length) {
      case 0:
        n = find_<0, true>(ctx, n, keys, hm, value_stride, values, value_sizes);
        break;
      case 1:
        n = find_<1, true>(ctx, n, keys, hm, value_stride, values, value_sizes);
        break;
      case 2:
        n = find_<2, true>(ctx, n, keys, hm, value_stride, values, value_sizes);
        break;
      case 4:
        n = find_<4, true>(ctx, n, keys, hm, value_stride, values, value_sizes);
        break;
      case 8:
        n = find_<8, true>(ctx, n, keys, hm, value_stride, values, value_sizes);
        break;
      default:
        NVE_THROW_("`config.key_fetch_queue_length` (", config.key_fetch_queue_length, ") is out of bounds!");
    }
  } else {
    switch (config.key_fetch_queue_length) {
      case 0:
        n = find_<0, false>(ctx, n, keys, hm, value_stride, values, value_sizes);
        break;
      case 1:
        n = find_<1, false>(ctx, n, keys, hm, value_stride, values, value_sizes);
        break;
      case 2:
        n = find_<2, false>(ctx, n, keys, hm, value_stride, values, value_sizes);
        break;
      case 4:
        n = find_<4, false>(ctx, n, keys, hm, value_stride, values, value_sizes);
        break;
      case 8:
        n = find_<8, false>(ctx, n, keys, hm, value_stride, values, value_sizes);
        break;
      default:
        NVE_THROW_("`config.key_fetch_queue_length` (", config.key_fetch_queue_length, ") is out of bounds!");
    }
  }
  auto counter = this->lookup_counter_storage(ctx);
  NVE_CHECK_(counter != nullptr, "Invalid key counter");
  *counter += n;
}

template <typename MapType, typename PartitionerType>
void NvhmMapTable<MapType, PartitionerType>::insert(
    context_ptr_t& ctx, const int64_t n, buffer_ptr<const void> keys_bw,
    const int64_t value_stride, const int64_t value_size, buffer_ptr<const void> values_bw) {
  if (n <= 0) return;
  const auto& __restrict config{config_};

  auto modify_stream = ctx->get_modify_stream();
  const void* const keys_vptr{
      keys_bw ? keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, modify_stream)
              : nullptr};
  const void* const values_vptr{
      values_bw ? values_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, modify_stream)
                : nullptr};
  const key_type* const __restrict keys{reinterpret_cast<const key_type*>(keys_vptr)};
  const char* const __restrict values{reinterpret_cast<const char*>(values_vptr)};

  // TODO: Allow dynamically sized vectors.
  NVE_CHECK_(value_size >= 0 && value_size <= config.max_value_size);
  const int64_t overflow_margin{config.overflow_policy.overflow_margin};
  const int64_t resolution_margin{static_cast<int64_t>(static_cast<double>(overflow_margin) *
                                                       config.overflow_policy.resolution_margin)};

  std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};
  const int64_t num_parts_mask{num_parts - 1};

  const table_id_t table_id{this->id};
  const auto f{[n, keys, value_stride, value_size, values, table_id, overflow_margin,
                resolution_margin, &parts, num_parts_mask](const int64_t task_idx) {
    Partition& __restrict part{parts[static_cast<uint64_t>(task_idx)]};
    std::lock_guard lock(part.read_write);
    map_type& __restrict map{part.map};

    lru_meta_type lru_value;
    if constexpr (std::is_same_v<meta_type, no_meta_type>) {
    } else if constexpr (std::is_same_v<meta_type, lru_meta_type>) {
      lru_value = lru_meta_value();
    } else if constexpr (std::is_same_v<meta_type, lfu_meta_type>) {
    } else {
      static_assert(dependent_false_v<meta_type>, "Overflow handler not implemented.");
    }

    int64_t part_size{static_cast<int64_t>(map.size())};
    for (int64_t i{}; i != n; ++i) {
      const key_type key{keys[i]};
      if (partitioner(key, num_parts_mask) != task_idx) continue;

      int64_t new_part_size{part_size};
      const write_pos_type pos{map.upsert(key, new_part_size)};
      map.set_raw_values_at(pos, &values[i * value_stride], static_cast<uint64_t>(value_size));

      if (new_part_size == part_size) continue;
      part_size = new_part_size;

      if constexpr (std::is_same_v<meta_type, no_meta_type>) {
      } else if constexpr (std::is_same_v<meta_type, lru_meta_type>) {
        map.value_at(pos) = lru_value;
      } else if constexpr (std::is_same_v<meta_type, lfu_meta_type>) {
        map.value_at(pos) = 1;
      } else {
        static_assert(dependent_false_v<meta_type>, "Overflow handler not implemented.");
      }

      // Handle overflows.
      if NVE_LIKELY_(part_size < overflow_margin) continue;
      NVE_LOG_VERBOSE_("NV map table ", table_id, " part ", task_idx,
                       " is overflowing. Attempting to resolve...");

      if constexpr (std::is_same_v<meta_type, no_meta_type>) {
        // Fetch all keys and shuffle them randomly.
        std::vector<key_type> keys(map.size());
        auto it{keys.begin()};
        it = map.keys(it);
        NVE_CHECK_(it == keys.end());
        std::shuffle(keys.begin(), keys.end(), std::default_random_engine{random_seed()});

        // Reclaim slots until the resolution margin is reached.
        const auto margin_it{keys.end() - resolution_margin};
        for (it = keys.begin(); it < margin_it; ++it) {
          map.erase(*it);
        }
      } else {
        // Fetch all keys and sort them by the ASCENDING by the associated meta-values.
        std::vector<std::pair<key_type, meta_type>> keys_metas(map.size());
        auto it{keys_metas.begin()};
        it = map.keys_and_values(it);
        NVE_CHECK_(it == keys_metas.end());
        std::sort(keys_metas.begin(), keys_metas.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });

        // Reclaim slots until the resolution margin is reached.
        const auto margin_it{keys_metas.end() - resolution_margin};
        for (it = keys_metas.begin(); it < margin_it; ++it) {
          map.erase(it->first);
        }

        if constexpr (std::is_same_v<meta_type, lru_meta_type>) {
        } else if constexpr (std::is_same_v<meta_type, lfu_meta_type>) {
          // Realign remaining frequency values to avoid entering a steady state.
          map.transform_values([](meta_type& meta) { meta >>= 1; });
        } else {
          static_assert(dependent_false_v<meta_type>, "Overflow handler not implemented.");
        }
      }

      // Update part_size after erasing
      part_size = static_cast<int64_t>(map.size());
    }
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config.workgroups, 1);
}

template <typename MapType, typename PartitionerType>
int64_t NvhmMapTable<MapType, PartitionerType>::size(context_ptr_t& ctx,
                                                                    const bool) const {
  const std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};

  std::atomic_int64_t total_n{0};
  const auto f{[&parts, &total_n](const int64_t task_idx) {
    int64_t n;
    {
      const Partition& __restrict part{parts[static_cast<uint64_t>(task_idx)]};
      std::shared_lock lock(part.read_write);

      n = static_cast<int64_t>(part.map.size());
    }
    total_n.fetch_add(n, std::memory_order_relaxed);
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config_.workgroups, 1);
  return total_n.load(std::memory_order_relaxed);
}

template <typename MapType, typename PartitionerType>
void NvhmMapTable<MapType, PartitionerType>::update(
    context_ptr_t& ctx, const int64_t n, buffer_ptr<const void> keys_bw,
    const int64_t value_stride, const int64_t value_size, buffer_ptr<const void> values_bw) {
  if (n <= 0) return;
  const auto& __restrict config{config_};

  auto modify_stream = ctx->get_modify_stream();
  const void* const keys_vptr{
      keys_bw ? keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, modify_stream)
              : nullptr};
  const void* const values_vptr{
      values_bw ? values_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, modify_stream)
                : nullptr};
  const key_type* const __restrict keys{reinterpret_cast<const key_type*>(keys_vptr)};
  const char* const __restrict values{reinterpret_cast<const char*>(values_vptr)};

  NVE_CHECK_(value_size >= 0 && value_size <= config.max_value_size);

  std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};
  const int64_t num_parts_mask{num_parts - 1};

  const auto f{[n, keys, value_stride, value_size, values, &parts, num_parts_mask](const int64_t task_idx) {
    Partition& __restrict part{parts[static_cast<uint64_t>(task_idx)]};
    std::lock_guard lock(part.read_write);
    map_type& __restrict map{part.map};

    for (int64_t i{}; i != n; ++i) {
      const key_type key{keys[i]};
      if (partitioner(key, num_parts_mask) != task_idx) continue;

      const write_pos_type pos{map.update(key)};
      if (pos == nvhm::npos) continue;

      map.set_raw_values_at(pos, &values[i * value_stride], static_cast<uint64_t>(value_size));
    }
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config.workgroups, 1);
}

template <typename MapType, typename PartitionerType>
void NvhmMapTable<MapType, PartitionerType>::update_accumulate(
    context_ptr_t& ctx, const int64_t n, buffer_ptr<const void> keys_bw,
    const int64_t update_stride, const int64_t update_size, buffer_ptr<const void> updates_bw,
    const DataType_t update_dtype) {
  if (n <= 0) return;
  const auto& __restrict config{config_};

  auto modify_stream = ctx->get_modify_stream();
  const void* const keys_vptr{
      keys_bw ? keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, modify_stream)
              : nullptr};
  const void* const updates_vptr{
      updates_bw ? updates_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, modify_stream)
                 : nullptr};
  const key_type* const __restrict keys{reinterpret_cast<const key_type*>(keys_vptr)};
  const char* const __restrict updates{reinterpret_cast<const char*>(updates_vptr)};

  NVE_CHECK_(update_size >= 0 && update_size <= config.max_value_size);
  const update_kernel_t update_kernel{pick_cpu_update_kernel(config.value_dtype, update_dtype)};

  std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};
  const int64_t num_parts_mask{num_parts - 1};

  const auto f{[n, keys, update_stride, update_size, updates, &update_kernel, &parts, num_parts_mask](const int64_t task_idx) {
    Partition& __restrict part{parts[static_cast<uint64_t>(task_idx)]};
    std::lock_guard lock(part.read_write);
    map_type& __restrict map{part.map};

    for (int64_t i{}; i != n; ++i) {
      const key_type key{keys[i]};
      if (partitioner(key, num_parts_mask) != task_idx) continue;

      const write_pos_type pos{map.update(key)};
      if (pos == nvhm::npos) continue;

      update_kernel(map.raw_values_at(pos), &updates[i * update_stride], update_size);
    }
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config.workgroups, 1);
}

template <typename MapType, typename PartitionerType>
template <size_t KeyFetchQueueLength, bool PrefetchValues>
int64_t NvhmMapTable<MapType, PartitionerType>::find_(
  context_ptr_t& ctx, const int64_t n, const key_type* const __restrict keys,
  bitmask64_t* const __restrict hit_mask, const int64_t value_stride, char* const __restrict values,
  int64_t* const __restrict value_sizes) const {
  if (values) {
    if (value_sizes) {
      return find_<KeyFetchQueueLength, PrefetchValues, true, true>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
    } else {
      return find_<KeyFetchQueueLength, PrefetchValues, true, false>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
    }
  } else {
    if (value_sizes) {
      return find_<KeyFetchQueueLength, PrefetchValues, false, true>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
    } else {
      return find_<KeyFetchQueueLength, PrefetchValues, false, false>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
    }
  }
}

template <typename MapType, typename PartitionerType>
template <size_t KeyFetchQueueLength, bool PrefetchValues, bool WithValues, bool WithValueSizes>
int64_t NvhmMapTable<MapType, PartitionerType>::find_(
    context_ptr_t& ctx, const int64_t n, const key_type* const __restrict keys,
    bitmask64_t* const __restrict hit_mask, const int64_t value_stride, char* const __restrict values,
    int64_t* const __restrict value_sizes) const {
  const auto& __restrict config{config_};

  const int64_t hm_size{ceil_div(n, bitmask64::num_bits)};

  const int64_t max_value_size{config.max_value_size};
  NVE_CHECK_(value_stride >= max_value_size);
  const int64_t max_find_task_size{config.max_find_task_size};

  const std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};
  const int64_t num_parts_mask{num_parts - 1};

  const int64_t num_tasks_per_part{(hm_size + max_find_task_size - 1) / max_find_task_size};
  const int64_t num_tasks{num_parts * num_tasks_per_part};

  std::atomic_int64_t total_num_hits{0};
  const auto f{[n, keys, hit_mask, value_stride, values, value_sizes, hm_size,
                max_value_size, max_find_task_size, &parts, num_parts, num_parts_mask,
                num_tasks_per_part, &total_num_hits](int64_t task_idx) {
    int64_t num_hits{};

    const int64_t part_idx{task_idx / num_tasks_per_part};
    task_idx %= num_tasks_per_part;

    int64_t hm_off0{max_find_task_size * task_idx};
    const int64_t task_size{std::min(max_find_task_size, hm_size - hm_off0)};

    // Scatter parts accross input domain.
    hm_off0 += hm_size * part_idx / num_parts;
    {
      const Partition& __restrict part{parts[static_cast<uint64_t>(part_idx)]};
      std::shared_lock lock(part.read_write);
      const map_type& __restrict map{part.map};

      const char* __restrict prev_src{};
      char* __restrict prev_dst{};
      int64_t prev_value_size{max_value_size};
      const lru_meta_type lru_time{lru_meta_value()};

      for (int64_t hm_idx{}; hm_idx != task_size; ++hm_idx) {
        const int64_t hm_off{(hm_off0 + hm_idx) % hm_size};
        bitmask64_t mask{bitmask64::load(&hit_mask[hm_off])};
        const int64_t i{hm_off * bitmask64::num_bits};
        num_hits -= bitmask64::count(mask);

        auto it_head{bitmask64::clip(~mask, n - i)};
        if constexpr (KeyFetchQueueLength) {
          bitmask64_t it_tail{};
          const auto process_tail{[&](const int j, const read_pos_type& pos) {
            const int64_t ij{i + j};

            // TODO: Support variable length vector sizes.
            const int64_t value_size{max_value_size};
            if constexpr (WithValueSizes) {
              value_sizes[ij] = value_size;
            } else {
              (void)value_sizes;
            }
            if constexpr (WithValues) {
              const char* const __restrict src{map.raw_values_at(pos)};
              char* const __restrict dst{&values[ij * value_stride]};
              if constexpr (PrefetchValues) {
                l1_prefetch(src, dst, std::min(value_size, 8 * cpu_cache_line_size));
                if (prev_src) {
                  nvhm::fast_copy(prev_dst, prev_src, static_cast<uint64_t>(prev_value_size));
                }
                prev_src = src;
                prev_dst = dst;
                prev_value_size = value_size;
              } else {
                nvhm::fast_copy(dst, src, static_cast<uint64_t>(value_size));
              }
            } else {
              (void)value_stride;
              (void)values;
            }

            if constexpr (std::is_same_v<meta_type, no_meta_type>) {
            } else if constexpr (std::is_same_v<meta_type, lru_meta_type>) {
              const_cast<lru_meta_type&>(map.value_at(pos)) = lru_time;
            } else if constexpr (std::is_same_v<meta_type, lfu_meta_type>) {
              ++const_cast<lfu_meta_type&>(map.value_at(pos));
            } else {
              static_assert(dependent_false_v<meta_type>, "Overflow handler not implemented.");
            }
            mask |= bitmask64::single(j);
          }};

          nvhm::experimental::ring_prefetch_queue<key_type, prefetch_type, KeyFetchQueueLength> queue;
          while (it_head) {
            const int64_t j{bitmask64::next(it_head)};
            it_head = bitmask64::skip(it_head);
            const int64_t ij{i + j};

            const key_type key{keys[ij]};
            if NVE_LIKELY_(partitioner(key, num_parts_mask) != part_idx) continue;
            it_tail |= bitmask64::single(j);

            queue.prepare_lookup(map, key, use_optimistic_prefetch);
            if (queue.full()) break;
          }

          while (it_head) {
            const int64_t j{bitmask64::next(it_head)};
            it_head = bitmask64::skip(it_head);
            const int64_t ij{i + j};

            const key_type key{keys[ij]};
            if NVE_LIKELY_(partitioner(key, num_parts_mask) != part_idx) continue;
            it_tail |= bitmask64::single(j);

            const read_pos_type pos{queue.pop_and_lookup(map)};
            if (pos != nvhm::npos) {
              process_tail(bitmask64::next(it_tail), pos);
            }
            it_tail = bitmask64::skip(it_tail);

            queue.prepare_lookup(map, key, use_optimistic_prefetch);
          }

          while (it_tail) {
            const read_pos_type pos{queue.pop_and_lookup(map)};
            if (pos != nvhm::npos) {
              process_tail(bitmask64::next(it_tail), pos);
            }
            it_tail = bitmask64::skip(it_tail);
          }
        } else {
          for (; it_head; it_head = bitmask64::skip(it_head)) {
            const int64_t j{bitmask64::next(it_head)};
            const int64_t ij{i + j};

            const key_type key{keys[ij]};
            if NVE_LIKELY_(partitioner(key, num_parts_mask) != part_idx) continue;

            const read_pos_type pos{map.lookup(key)};
            if (pos == nvhm::npos) continue;

            const int64_t value_size{max_value_size};  // TODO: Support variable length vector sizes.
            if constexpr (WithValueSizes) {
              value_sizes[ij] = value_size;
            } else {
              (void)value_sizes;
            }
            if constexpr (WithValues) {
              const char* const __restrict src{map.raw_values_at(pos)};
              char* const __restrict dst{&values[ij * value_stride]};
              if constexpr (PrefetchValues) {
                l1_prefetch(src, dst, std::min(value_size, 8 * cpu_cache_line_size));
                if (prev_src) {
                  nvhm::fast_copy(prev_dst, prev_src, static_cast<uint64_t>(prev_value_size));
                }
                prev_src = src;
                prev_dst = dst;
                prev_value_size = value_size;
              } else {
                nvhm::fast_copy(dst, src, static_cast<uint64_t>(value_size));
              }
            } else {
              (void)value_stride;
              (void)values;
            }

            if constexpr (std::is_same_v<meta_type, no_meta_type>) {
            } else if constexpr (std::is_same_v<meta_type, lru_meta_type>) {
              const_cast<lru_meta_type&>(map.value_at(pos)) = lru_time;
            } else if constexpr (std::is_same_v<meta_type, lfu_meta_type>) {
              ++const_cast<lfu_meta_type&>(map.value_at(pos));
            } else {
              static_assert(dependent_false_v<meta_type>, "Overflow handler not implemented.");
            }
            mask |= bitmask64::single(j);
          }
        }

        num_hits += bitmask64::count(mask);
        bitmask64::atomic_join(&hit_mask[hm_off], mask);
      }

      if constexpr (WithValues && PrefetchValues) {
        if (prev_src) {
          nvhm::fast_copy(prev_dst, prev_src, static_cast<uint64_t>(prev_value_size));
        }
      }
    }

    total_num_hits.fetch_add(num_hits, std::memory_order_relaxed);
  }};

  ctx->get_thread_pool()->execute_n(0, num_tasks, f, config.workgroups, num_tasks_per_part);
  return total_num_hits.load(std::memory_order_relaxed);
}

}  // namespace plugin
}  // namespace nve
