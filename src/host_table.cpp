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

#include <bit_ops.hpp>
#include <host_table.hpp>
#include <json_support.hpp>
#include <limits>

namespace nve {

NVE_DEFINE_JSON_ENUM_CONVERSION_(Partitioner_t, Partitioner_t::AlwaysZero,
                                 Partitioner_t::FowlerNollVo, Partitioner_t::Murmur3,
                                 Partitioner_t::Rrxmrrxmsx0, Partitioner_t::StdHash)

NVE_DEFINE_JSON_ENUM_CONVERSION_(OverflowHandler_t, OverflowHandler_t::EvictRandom,
                                 OverflowHandler_t::EvictLRU, OverflowHandler_t::EvictLFU)

void OverflowPolicyConfig::check() const {
  NVE_CHECK_(overflow_margin >= 0 && overflow_margin <= max_overflow_margin);
  NVE_CHECK_(resolution_margin >= 0.0 && resolution_margin <= 1.0);
}

void from_json(const nlohmann::json& json, OverflowPolicyConfig& conf) {
  NVE_READ_JSON_FIELD_(overflow_margin);
  NVE_READ_JSON_FIELD_(handler);
  NVE_READ_JSON_FIELD_(resolution_margin);
}

void to_json(nlohmann::json& json, const OverflowPolicyConfig& conf) {
  json = json.object();

  NVE_WRITE_JSON_FIELD_(overflow_margin);
  NVE_WRITE_JSON_FIELD_(handler);
  NVE_WRITE_JSON_FIELD_(resolution_margin);
}

void HostTableConfig::check() const {
  static const auto key_sizes{
    make_array(sizeof(int8_t), sizeof(int16_t), sizeof(int32_t), sizeof(int64_t))};
  NVE_CHECK_(std::find(key_sizes.begin(), key_sizes.end(), key_size) != key_sizes.end());

  NVE_CHECK_(max_value_size > 0 &&
             max_value_size <=
                 static_cast<int64_t>(std::numeric_limits<int32_t>::max() - sizeof(int32_t)));
  NVE_CHECK_(max_value_size % value_dtype_size() == 0);
}

void from_json(const nlohmann::json& json, HostTableConfig& conf) {
  NVE_READ_JSON_FIELD_(key_size);
  NVE_READ_JSON_FIELD_(max_value_size);
  NVE_READ_JSON_FIELD_(value_dtype);
  NVE_READ_JSON_FIELD_(invalid_key);
}

void to_json(nlohmann::json& json, const HostTableConfig& conf) {
  json = json.object();

  NVE_WRITE_JSON_FIELD_(key_size);
  NVE_WRITE_JSON_FIELD_(max_value_size);
  NVE_WRITE_JSON_FIELD_(value_dtype);
  NVE_WRITE_JSON_FIELD_(invalid_key);
}

void HostTableFactoryConfig::check() const {}

void from_json(const nlohmann::json&, HostTableFactoryConfig&) {}

void to_json(nlohmann::json& json, const HostTableFactoryConfig&) { json = json.object(); }

HostTableLike::HostTableLike(const table_id_t id) : id{id} {}

}  // namespace nve
