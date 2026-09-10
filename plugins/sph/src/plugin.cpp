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

#include <plugin/nve_internal_plugin.hpp>
#include <sph_table.hpp>

/* SphTableFactory is not default-constructible, so the empty factory config
 * still goes through the regular configured macro. */
NVE_DEFINE_INTERNAL_PLUGIN("SPH GPU hash table plugin", "NVIDIA Corporation",
                           nve::plugin::SphTableFactory,
                           nve::plugin::SphTableFactoryConfig)
