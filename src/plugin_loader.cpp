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

#include <dlfcn.h>

#include <common.hpp>
#include <json_support.hpp>
#include <plugin/external_plugin_table.hpp>
#include <plugin/plugin_loader.hpp>

namespace nve {

namespace {

/* Metadata symbol types (declared in nve_plugin_common.h). */
using set_nve_services_t = void (*)(const nve_ext_nve_services_t*);
using plugin_string_t = const char* (*)();
using plugin_u32_t = uint32_t (*)();
using plugin_mode_fn_t = nve_plugin_mode_t (*)();

/* POSIX protocol: clear the stale dlerror() state before the lookup so a
 * failure below reports its own message (checked lazily on failure only). */
template <typename Fn>
Fn resolve_symbol(void* handle, const char* name) {
  dlerror();
  Fn fn{reinterpret_cast<Fn>(dlsym(handle, name))};
  NVE_CHECK_(fn != nullptr, "Plugin is missing required symbol '", name, "': ", dlerror());
  return fn;
}

/* Like resolve_symbol, but a missing symbol is a valid null result. */
template <typename Fn>
Fn optional_symbol(void* handle, const char* name) {
  dlerror();
  return reinterpret_cast<Fn>(dlsym(handle, name));
}

/**
 * Wraps an internal plugin's real factory. Holds a copy of the library lease,
 * and gives each produced table a host-owned control block containing both
 * the real table and another copy of that lease, so plugin code stays loaded
 * until the last dependent object dies.
 */
struct OwnedTableState {
  plugin_library_ptr_t library;  // declared first, destroyed last
  table_ptr_t table;             // destroyed before library / dlclose
};

class OwningTableFactory final : public TableFactory {
 public:
  OwningTableFactory(plugin_library_ptr_t library, table_factory_ptr_t inner)
      : library_(std::move(library)), inner_(std::move(inner)) {}

  table_ptr_t produce(table_id_t id, const nlohmann::json& json) override {
    auto state = std::make_shared<OwnedTableState>(
        OwnedTableState{library_, inner_->produce(id, json)});
    NVE_CHECK_(state->table != nullptr, "Plugin returned a null table");
    return table_ptr_t(state, state->table.get());  // aliases the combined state
  }

 private:
  plugin_library_ptr_t library_;  // inner_ is destroyed first (reverse member order)
  table_factory_ptr_t inner_;
};

}  // namespace

PluginLibrary::~PluginLibrary() noexcept {
  if (handle_ == nullptr) return;
  if (dlclose(handle_) != 0) {
    const char* error = dlerror();
    NVE_LOG_ERROR_("dlclose failed for plugin library: ", error ? error : "unknown error");
  }
}

Plugin::Plugin(const std::filesystem::path& so_path) {
  const std::string plugin_path{so_path.string()};
  NVE_CHECK_(!plugin_path.empty(), "Table plugin shared object must not be empty.");
  NVE_LOG_INFO_("Attempting to load table plugin '", plugin_path, "'.");

  // RTLD_LOCAL: external SOs may carry conflicting symbols from unknown toolchains.
  void* handle{dlopen(plugin_path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE)};
  if (handle == nullptr) {
    const char* error{dlerror()};
    const std::string first_error{error ? error : "unknown dlopen error"};
    // An absolute path has no distinct fallback location to retry.
    NVE_CHECK_(!so_path.is_absolute(),
               "Failed to load table plugin '", plugin_path, "': ", first_error);
    // A relative name that failed to load retries beside nve-common.
    Dl_info dli;
    NVE_CHECK_(dladdr(reinterpret_cast<const void*>(&resolve_symbol<plugin_string_t>), &dli) != 0,
               "Failed to resolve the nve-common library path for the plugin fallback.");
    const std::filesystem::path fallback{
        std::filesystem::path{dli.dli_fname}.parent_path() / plugin_path};
    handle = dlopen(fallback.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
    NVE_CHECK_(handle != nullptr, "Failed to load table plugin '", plugin_path, "': ", first_error,
               "; fallback '", fallback.string(), "': ", dlerror());
  }

  // Failure-atomic: install the raw handle in the lease before any validation,
  // so every throw below releases the loader reference. If even the lease
  // allocation fails, release the reference directly.
  try {
    library_ = plugin_library_ptr_t(new PluginLibrary(handle));
  } catch (...) {
    dlclose(handle);
    throw;
  }

  abi_major_ = resolve_symbol<plugin_u32_t>(handle, "nve_plugin_abi_version_major")();
  abi_minor_ = resolve_symbol<plugin_u32_t>(handle, "nve_plugin_abi_version_minor")();
  NVE_CHECK_(abi_major_ == NVE_PLUGIN_ABI_VERSION_MAJOR,
             "Plugin '", plugin_path, "' was built against incompatible ABI major version ",
             abi_major_, " (host: ", NVE_PLUGIN_ABI_VERSION_MAJOR, ").");
  NVE_CHECK_(abi_minor_ <= NVE_PLUGIN_ABI_VERSION_MINOR,
             "Plugin '", plugin_path, "' was built against newer ABI minor version ",
             abi_major_, ".", abi_minor_, " (host: ", NVE_PLUGIN_ABI_VERSION_MAJOR, ".",
             NVE_PLUGIN_ABI_VERSION_MINOR, ").");

  const char* name{resolve_symbol<plugin_string_t>(handle, "nve_plugin_name")()};
  const char* author{resolve_symbol<plugin_string_t>(handle, "nve_plugin_author")()};
  NVE_CHECK_(name != nullptr && author != nullptr,
             "Plugin '", plugin_path, "' returned a null name or author.");
  name_ = name;
  author_ = author;

  const nve_plugin_mode_t mode{resolve_symbol<plugin_mode_fn_t>(handle, "nve_plugin_mode")()};
  switch (mode) {
    case NVE_PLUGIN_MODE_INTERNAL:
      mode_ = PluginMode::Internal;
      create_internal_ =
          resolve_symbol<create_internal_factory_t>(handle, "nve_plugin_create_internal_factory");
      break;
    case NVE_PLUGIN_MODE_EXTERNAL: {
      mode_ = PluginMode::External;
      create_external_ =
          resolve_symbol<create_external_factory_t>(handle, "nve_plugin_create_external_factory");
      // Optional one-time handshake; must be idempotent on the plugin side
      // because constructing Plugin again for an already loaded SO repeats it.
      auto set_services{optional_symbol<set_nve_services_t>(handle, "nve_plugin_set_nve_services")};
      if (set_services != nullptr) {
        set_services(external_nve_services());
      }
      break;
    }
    default:
      NVE_THROW_("Plugin '", plugin_path, "' reports unsupported mode ", mode, ".");
  }

  NVE_LOG_INFO_("Loaded plugin '", name_, "' by '", author_, "' (ABI ", abi_major_, ".",
                abi_minor_, ", mode ", mode == NVE_PLUGIN_MODE_INTERNAL ? "internal" : "external",
                ").");
}

table_factory_ptr_t Plugin::create_table_factory(const nlohmann::json& config) const {
  if (mode_ == PluginMode::Internal) {
    table_factory_ptr_t inner{create_internal_(config)};
    NVE_CHECK_(inner != nullptr, "Plugin '", name_, "' returned a null table factory.");
    return std::make_shared<OwningTableFactory>(library_, std::move(inner));
  }

  const std::string cfg{config.dump()};
  nve_ext_table_factory_t handle{};
  const char* error{nullptr};
  const nve_status_t status{create_external_(cfg.c_str(), &handle, &error)};
  if (status != NVE_SUCCESS) {
    // Copy immediately: the text is plugin-owned thread-local storage.
    check_external_status(status, error);
  }
  return adopt_external_factory(library_, handle);
}

}  // namespace nve
