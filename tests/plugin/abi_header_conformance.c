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
 * Header-conformance test: compiling this translation unit as C11 evaluates
 * every published _Static_assert in the external plugin header: scalar widths,
 * struct sizes, alignment, and field offsets. Incompatible packing options
 * fail here at compile time instead of producing an incompatible SO.
 */

#include <plugin/nve_plugin_common.h>
#include <plugin/nve_external_plugin.h>

/* The metadata + NVE-services macro must compile in plain C. */
NVE_DEFINE_EXTERNAL_PLUGIN("ABI conformance test plugin", "NVIDIA Corporation")

/* Size-prefixed designated initializers must compile for every ABI struct. */
static const nve_ext_table_ops_t k_test_table_ops = {
    .struct_size = sizeof(nve_ext_table_ops_t),
};

static const nve_ext_table_factory_ops_t k_test_factory_ops = {
    .struct_size = sizeof(nve_ext_table_factory_ops_t),
};

static const nve_ext_nve_services_t k_test_services = {
    .struct_size = sizeof(nve_ext_nve_services_t),
};

/* NVE_ABI_HAS_FIELD must be usable as a plain C expression. */
int nve_abi_conformance_probe(void);
int nve_abi_conformance_probe(void) {
  (void)nve_plugin_nve_services();
  return NVE_ABI_HAS_FIELD(&k_test_table_ops, nve_ext_table_ops_t, last_error) +
         NVE_ABI_HAS_FIELD(&k_test_factory_ops, nve_ext_table_factory_ops_t, produce) +
         NVE_ABI_HAS_FIELD(&k_test_services, nve_ext_nve_services_t, get_scratch);
}
