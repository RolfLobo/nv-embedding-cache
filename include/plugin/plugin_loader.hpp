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

#include <filesystem>
#include <memory>
#include <string>
#include <utility>

#include <nlohmann/json_fwd.hpp>
#include <plugin/nve_external_plugin.h>
#include <plugin/plugin.hpp>

namespace nve {

/**
 * One host-side lease for one successful dlopen call. Copies of the shared_ptr
 * do not call dlopen again; the last copy performs the one matching dlclose.
 *
 * Every object whose vtable, destructor, or shared_ptr control block contains
 * code from a plugin SO must transitively retain one of these leases, so the
 * dlclose cannot race object destruction.
 */
class PluginLibrary final {
 public:
  ~PluginLibrary() noexcept;  // dlclose(handle_); logs errors, never throws

  NVE_PREVENT_COPY_AND_MOVE_(PluginLibrary);

 private:
  friend class Plugin;
  explicit PluginLibrary(void* handle) : handle_(handle) {}
  void* handle_{nullptr};
};

using plugin_library_ptr_t = std::shared_ptr<PluginLibrary>;

/**
 * Loads one plugin SO (dlopen with RTLD_LOCAL) and validates its metadata,
 * ABI version and mode. May live on the stack: `create_table_factory` copies
 * the library lease into the returned factory, and the factory copies it into
 * every produced table, so destroying the Plugin never unloads code that a
 * live factory or table still references.
 */
class Plugin {
 public:
  /**
   * @param so_path Shared object name/path. Bare sonames are resolved by the
   * dynamic linker (RUNPATH/LD_LIBRARY_PATH) with a fallback beside the
   * nve-common library; explicit paths are loaded directly.
   */
  explicit Plugin(const std::filesystem::path& so_path);
  ~Plugin() = default;  // releases this object's lease copy

  NVE_PREVENT_COPY_AND_MOVE_(Plugin);

  const std::string& name() const { return name_; }
  const std::string& author() const { return author_; }
  PluginMode mode() const { return mode_; }
  std::pair<uint32_t, uint32_t> abi_version() const { return {abi_major_, abi_minor_}; }

  /**
   * Creates a table factory from the plugin. Internal plugins return their
   * real factory wrapped with a library lease; external plugins are wrapped
   * behind the C-vtable adapter. Either way the result (and every table it
   * produces) keeps the SO loaded for its own lifetime.
   *
   * @param config Plugin-specific factory configuration (JSON object).
   */
  table_factory_ptr_t create_table_factory(const nlohmann::json& config) const;

 private:
  /* Creation symbol signatures from the external and internal plugin
   * contracts, stored typed so calls need no cast. */
  using create_internal_factory_t = table_factory_ptr_t (*)(const nlohmann::json&);
  using create_external_factory_t = nve_status_t (*)(const char*, nve_ext_table_factory_t*,
                                                     const char**);

  plugin_library_ptr_t library_;
  std::string name_;
  std::string author_;
  PluginMode mode_{PluginMode::Internal};
  uint32_t abi_major_{0};
  uint32_t abi_minor_{0};
  create_internal_factory_t create_internal_{nullptr};  // nve_plugin_create_internal_factory
  create_external_factory_t create_external_{nullptr};  // nve_plugin_create_external_factory
};

}  // namespace nve
