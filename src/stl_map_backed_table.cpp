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
#include <stl_map_backed_table_detail.hpp>

namespace nve {

void STLContainerTableConfig::check() const {
  base_type::check();

  NVE_CHECK_(num_partitions >= 0 && has_single_bit(to_uint(num_partitions)));
  NVE_CHECK_(!workgroups.empty());
  NVE_CHECK_(max_find_task_size > 0);

  NVE_CHECK_(value_alignment >= 0 && has_single_bit(to_uint(value_alignment)));
  NVE_CHECK_(allocation_rate >= slot_stride());

  overflow_policy.check();
}

void from_json(const nlohmann::json& json, STLContainerTableConfig& conf) {
  using base_type = STLContainerTableConfig::base_type;
  from_json(json, static_cast<base_type&>(conf));

  NVE_READ_JSON_FIELD_(num_partitions);
  NVE_READ_JSON_FIELD_(partitioner);
  NVE_READ_JSON_FIELD_(workgroups);
  NVE_READ_JSON_FIELD_(max_find_task_size);

  NVE_READ_JSON_FIELD_(value_alignment);
  NVE_READ_JSON_FIELD_(allocation_rate);

  NVE_READ_JSON_FIELD_(prefetch_values);

  NVE_READ_JSON_FIELD_(overflow_policy);
}

void to_json(nlohmann::json& json, const STLContainerTableConfig& conf) {
  using base_type = STLContainerTableConfig::base_type;
  to_json(json, static_cast<const base_type&>(conf));

  NVE_WRITE_JSON_FIELD_(num_partitions);
  NVE_WRITE_JSON_FIELD_(partitioner);
  NVE_WRITE_JSON_FIELD_(workgroups);
  NVE_WRITE_JSON_FIELD_(max_find_task_size);

  NVE_WRITE_JSON_FIELD_(value_alignment);
  NVE_WRITE_JSON_FIELD_(allocation_rate);

  NVE_WRITE_JSON_FIELD_(prefetch_values);

  NVE_WRITE_JSON_FIELD_(overflow_policy);
}

void STLContainerTableFactoryConfig::check() const { base_type::check(); }

void from_json(const nlohmann::json& json, STLContainerTableFactoryConfig& conf) {
  using base_type = typename STLContainerTableFactoryConfig::base_type;
  from_json(json, static_cast<base_type&>(conf));
}

void to_json(nlohmann::json& json, const STLContainerTableFactoryConfig& conf) {
  using base_type = typename STLContainerTableFactoryConfig::base_type;
  to_json(json, static_cast<const base_type&>(conf));
}

}  // namespace nve
