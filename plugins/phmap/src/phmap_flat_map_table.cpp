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

#include <execution_context.hpp>
#include <host_table_detail.hpp>
#include <phmap_flat_map_table.hpp>
#include <stl_map_backed_table_detail.hpp>

namespace nve {
namespace plugin {

void PHMapFlatMapTableConfig::check() const {
  base_type::check();

  NVE_CHECK_(initial_capacity >= 0);
}

void from_json(const nlohmann::json& json, PHMapFlatMapTableConfig& conf) {
  using base_type = PHMapFlatMapTableConfig::base_type;
  from_json(json, static_cast<base_type&>(conf));

  NVE_READ_JSON_FIELD_(initial_capacity);
}

void to_json(nlohmann::json& json, const PHMapFlatMapTableConfig& conf) {
  using base_type = PHMapFlatMapTableConfig::base_type;
  to_json(json, static_cast<const base_type&>(conf));

  NVE_WRITE_JSON_FIELD_(initial_capacity);
}

template <typename KeyType, typename MetaType, typename PartitionerType>
PHMapFlatMapTable<KeyType, MetaType, PartitionerType>::PHMapFlatMapTable(
    const table_id_t id, const PHMapFlatMapTableConfig& config)
    : base_type(id, config) {
  for (auto& part : this->parts_) {
    part.slot_map.reserve(to_uint(config.initial_capacity));
  }
}

void PHMapFlatMapTableFactoryConfig::check() const { base_type::check(); }

void from_json(const nlohmann::json& json, PHMapFlatMapTableFactoryConfig& conf) {
  using base_type = PHMapFlatMapTableFactoryConfig::base_type;
  from_json(json, static_cast<base_type&>(conf));
}

void to_json(nlohmann::json& json, const PHMapFlatMapTableFactoryConfig& conf) {
  using base_type = PHMapFlatMapTableFactoryConfig::base_type;
  to_json(json, static_cast<const base_type&>(conf));
}

PHMapFlatMapTableFactory::PHMapFlatMapTableFactory(const config_type& config) : base_type(config) {}

template <typename KeyType, typename MetaType>
inline static host_table_ptr_t make_phmap_flat_map_table_3(const table_id_t id,
                                                           const PHMapFlatMapTableConfig& config) {
  if (config.num_partitions == 1) {
    if (config.partitioner != Partitioner_t::AlwaysZero) {
      NVE_LOG_VERBOSE_("Selected ", config.partitioner, " partitioner was disabled because table has only 1 partition.");
    }
    return std::make_shared<PHMapFlatMapTable<KeyType, MetaType, AlwaysZeroPartitioner>>(id, config);
  }

  switch (config.partitioner) {
#if defined(NVE_FEATURE_HT_PART_FNV1A)
    case Partitioner_t::FowlerNollVo:
      return std::make_shared<PHMapFlatMapTable<KeyType, MetaType, FowlerNollVoPartitioner>>(id, config);
#endif
#if defined(NVE_FEATURE_HT_PART_MURMUR3)
    case Partitioner_t::Murmur3:
      return std::make_shared<PHMapFlatMapTable<KeyType, MetaType, Murmur3Partitioner>>(id, config);
#endif
#if defined(NVE_FEATURE_HT_PART_RRXMRRXMSX0)
    case Partitioner_t::Rrxmrrxmsx0:
      return std::make_shared<PHMapFlatMapTable<KeyType, MetaType, Rrxmrrxmsx0Partitioner>>(id, config);
#endif
#if defined(NVE_FEATURE_HT_PART_STD_HASH)
    case Partitioner_t::StdHash:
      return std::make_shared<PHMapFlatMapTable<KeyType, MetaType, StdHashPartitioner>>(id, config);
#endif
    default:
      NVE_THROW_("`config.partitioner` (", config.partitioner, ") is out of bounds!");
  }
}

template <typename KeyType>
inline static host_table_ptr_t make_phmap_flat_map_table_2(const table_id_t id,
                                                           const PHMapFlatMapTableConfig& config) {
  switch (config.overflow_policy.handler) {
    case OverflowHandler_t::EvictRandom:
      return make_phmap_flat_map_table_3<KeyType, no_meta_t>(id, config);
    case OverflowHandler_t::EvictLRU:
      return make_phmap_flat_map_table_3<KeyType, lru_meta_t>(id, config);
    case OverflowHandler_t::EvictLFU:
      return make_phmap_flat_map_table_3<KeyType, lfu_meta_t>(id, config);
  }
  NVE_THROW_("`config.overflow_policy.handler` (", config.overflow_policy.handler,
             ") is out of bounds!");
}

inline static host_table_ptr_t make_phmap_flat_map_table_1(const table_id_t id,
                                                           const PHMapFlatMapTableConfig& config) {
  switch (config.key_size) {
#if defined(NVE_FEATURE_HT_KEY_8)
    case sizeof(int8_t):
      return make_phmap_flat_map_table_2<int8_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_KEY_16)
    case sizeof(int16_t):
      return make_phmap_flat_map_table_2<int16_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_KEY_32)
    case sizeof(int32_t):
      return make_phmap_flat_map_table_2<int32_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_KEY_64)
    case sizeof(int64_t):
      return make_phmap_flat_map_table_2<int64_t>(id, config);
#endif
  }
  NVE_THROW_("`config.key_size` (", config.key_size, ") is out of bounds!");
}

host_table_ptr_t PHMapFlatMapTableFactory::produce(const table_id_t id,
                                                   const PHMapFlatMapTableConfig& config) {
  return make_phmap_flat_map_table_1(id, config);
}

}  // namespace plugin
}  // namespace nve
