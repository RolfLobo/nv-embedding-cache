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

/*
 * Interface for internal-mode (same-toolchain C++) plugins. A plugin's
 * plugin.cpp collapses to a single NVE_DEFINE_INTERNAL_PLUGIN(...) invocation
 * that emits the five metadata symbols plus the C++ creation symbol
 * `nve_plugin_create_internal_factory`.
 */

#include <memory>
#include <nlohmann/json.hpp>
#include <plugin/nve_plugin_common.h>
#include <plugin/plugin.hpp>

/* The creation symbol intentionally returns a C++ type with C linkage; that is
 * the internal-mode contract (same toolchain on both sides). */
#if defined(__clang__)
#define NVE_PLUGIN_SUPPRESS_C_LINKAGE_WARNING_ \
  _Pragma("GCC diagnostic push")               \
  _Pragma("GCC diagnostic ignored \"-Wreturn-type-c-linkage\"")
#define NVE_PLUGIN_RESTORE_C_LINKAGE_WARNING_ _Pragma("GCC diagnostic pop")
#else
#define NVE_PLUGIN_SUPPRESS_C_LINKAGE_WARNING_
#define NVE_PLUGIN_RESTORE_C_LINKAGE_WARNING_
#endif

/* Shared metadata block for internal-mode plugins. */
#define NVE_DEFINE_INTERNAL_PLUGIN_METADATA_(NAME_STR, AUTHOR_STR)                        \
  extern "C" NVE_C_API uint32_t nve_plugin_abi_version_major(void) {                      \
    return NVE_PLUGIN_ABI_VERSION_MAJOR; }                                                \
  extern "C" NVE_C_API uint32_t nve_plugin_abi_version_minor(void) {                      \
    return NVE_PLUGIN_ABI_VERSION_MINOR; }                                                \
  extern "C" NVE_C_API const char* nve_plugin_name(void) { return NAME_STR; }             \
  extern "C" NVE_C_API const char* nve_plugin_author(void) { return AUTHOR_STR; }         \
  extern "C" NVE_C_API nve_plugin_mode_t nve_plugin_mode(void) {                          \
    return NVE_PLUGIN_MODE_INTERNAL; }

/**
 * Defines an internal plugin whose factory is constructed from the factory
 * JSON via `static_cast<CONFIG_TYPE>(config)`.
 */
#define NVE_DEFINE_INTERNAL_PLUGIN(NAME_STR, AUTHOR_STR, FACTORY_TYPE, CONFIG_TYPE)       \
  NVE_DEFINE_INTERNAL_PLUGIN_METADATA_(NAME_STR, AUTHOR_STR)                              \
  NVE_PLUGIN_SUPPRESS_C_LINKAGE_WARNING_                                                  \
  extern "C" NVE_C_API nve::table_factory_ptr_t                                           \
  nve_plugin_create_internal_factory(const nlohmann::json& config) {                      \
    return std::make_shared<FACTORY_TYPE>(static_cast<CONFIG_TYPE>(config)); }            \
  NVE_PLUGIN_RESTORE_C_LINKAGE_WARNING_

/**
 * Defines an internal plugin whose factory is default-constructed and
 * intentionally ignores the factory JSON.
 */
#define NVE_DEFINE_INTERNAL_PLUGIN_NO_CONFIG(NAME_STR, AUTHOR_STR, FACTORY_TYPE)          \
  NVE_DEFINE_INTERNAL_PLUGIN_METADATA_(NAME_STR, AUTHOR_STR)                              \
  NVE_PLUGIN_SUPPRESS_C_LINKAGE_WARNING_                                                  \
  extern "C" NVE_C_API nve::table_factory_ptr_t                                           \
  nve_plugin_create_internal_factory(const nlohmann::json& /*config*/) {                  \
    return std::make_shared<FACTORY_TYPE>(); }                                            \
  NVE_PLUGIN_RESTORE_C_LINKAGE_WARNING_
