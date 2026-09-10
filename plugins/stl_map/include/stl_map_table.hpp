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

#include <stl_map_backed_table.hpp>

namespace nve {
namespace plugin {

struct STLMapTableConfig : public STLContainerTableConfig {
  using base_type = STLContainerTableConfig;

  template <typename KeyType>
  using map_type = std::unordered_map<KeyType, std::byte*>;

  void check() const;
};

void from_json(const nlohmann::json& json, STLMapTableConfig& conf);

void to_json(nlohmann::json& json, const STLMapTableConfig& conf);

template <typename KeyType, typename MetaType, typename PartitionerType>
using STLMapTable = STLContainerTable<STLMapTableConfig, KeyType, MetaType, PartitionerType>;

struct STLMapTableFactoryConfig final : public STLContainerTableFactoryConfig {
  using base_type = STLContainerTableFactoryConfig;

  void check() const;
};

void from_json(const nlohmann::json& json, STLMapTableFactoryConfig& conf);

void to_json(nlohmann::json& json, const STLMapTableFactoryConfig& conf);

class STLMapTableFactory final
    : public STLContainerTableFactory<STLMapTableFactoryConfig, STLMapTableConfig> {
 public:
  using base_type = STLContainerTableFactory<STLMapTableFactoryConfig, STLMapTableConfig>;

  NVE_PREVENT_COPY_AND_MOVE_(STLMapTableFactory);

  STLMapTableFactory() = delete;

  STLMapTableFactory(const config_type& config);

  virtual ~STLMapTableFactory() = default;

  virtual host_table_ptr_t produce(table_id_t id, const table_config_type& config) override;
};

}  // namespace plugin
}  // namespace nve
