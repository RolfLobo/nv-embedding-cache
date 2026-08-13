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

/*
 * Header-conformance test: compiling this translation unit as C++17 evaluates
 * every published static_assert in the external plugin header: scalar widths,
 * struct sizes, alignment, and field offsets. This mirrors the
 * C11 conformance TU. External plugins may be written in C++, so both
 * languages must agree on the published layout.
 */

#include <plugin/nve_plugin_common.h>
#include <plugin/nve_external_plugin.h>

/* The metadata + NVE-services macro must compile in C++ too. */
NVE_DEFINE_EXTERNAL_PLUGIN("ABI conformance test plugin (C++)", "NVIDIA Corporation")

/* Size-prefixed ABI structs and NVE_ABI_HAS_FIELD must be usable as plain
 * C++17 (which has no designated initializers, so value-init then prefix). */
int nve_abi_conformance_probe_cpp();
int nve_abi_conformance_probe_cpp() {
  (void)nve_plugin_nve_services();
  nve_ext_table_ops_t table_ops{};
  table_ops.struct_size = sizeof(nve_ext_table_ops_t);
  nve_ext_table_factory_ops_t factory_ops{};
  factory_ops.struct_size = sizeof(nve_ext_table_factory_ops_t);
  nve_ext_nve_services_t services{};
  services.struct_size = sizeof(nve_ext_nve_services_t);
  return NVE_ABI_HAS_FIELD(&table_ops, nve_ext_table_ops_t, last_error) +
         NVE_ABI_HAS_FIELD(&factory_ops, nve_ext_table_factory_ops_t, produce) +
         NVE_ABI_HAS_FIELD(&services, nve_ext_nve_services_t, get_scratch);
}
