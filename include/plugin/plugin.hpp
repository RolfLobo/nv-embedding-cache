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

#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <plugin/nve_plugin_common.h>
#include <table.hpp>

namespace nve {

/**
 * General factory interface implemented by plugins. Produces any `Table`
 * implementation from a JSON description, so a plugin is not restricted to host
 * tables.
 */
class TableFactory {
 public:
  NVE_PREVENT_COPY_AND_MOVE_(TableFactory);

  TableFactory() = default;

  virtual ~TableFactory() = default;

  virtual table_ptr_t produce(table_id_t id, const nlohmann::json& json) = 0;
};

using table_factory_ptr_t = std::shared_ptr<TableFactory>;

/** ABI mode reported by a plugin SO via `nve_plugin_mode()`. */
enum class PluginMode : nve_plugin_mode_t {
  Internal = NVE_PLUGIN_MODE_INTERNAL, /* C++ ABI, same toolchain       */
  External = NVE_PLUGIN_MODE_EXTERNAL, /* pure-C ABI, unknown toolchain */
};

}  // namespace nve
