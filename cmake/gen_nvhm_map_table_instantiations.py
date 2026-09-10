# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate explicit-instantiation sources for the nvhm plugin's NvhmMapTable.

The factory dispatch chain in plugins/nvhm/src/nvhm_map_table.cpp (make_nvhm_map_table_1..6)
instantiates NvhmMapTable across a large combination space (key x meta x kernel x
auto_shrink x partitioner). Compiling the whole matrix in that single translation
unit dominates the build time of the nvhm plugin (and the whole build's critical path).

This script enumerates every combination the factory can produce for the enabled NVE feature set
and emits:

  * nvhm_map_table_inst_NNN.cpp   - explicit `template class` instantiation definitions, split
                                    across NUM_TUS files so the matrix compiles in parallel. Each
                                    TU includes nvhm_map_table_impl.hpp (the template member
                                    definitions), so the explicit class instantiation also pulls
                                    in the find_<...> member-template fan-out.
  * nvhm_map_table_generated.cmake - sets NVHM_GENERATED_SOURCES to the list of .cpp files.

nvhm_map_table.cpp itself deliberately does NOT include the impl header, so its factory chain
cannot re-instantiate the member bodies; they resolve at link time against these TUs. A factory
dimension added without updating this script therefore fails loudly at link (undefined
NvhmMapTable symbols). Keep the enumeration in sync with the factory chain.
"""

import argparse
import os

# --- tunables -----------------------------------------------------------------------------------
# Number of generated .cpp translation units the instantiation matrix is split across. Override
# with --num-tus (or -DNVHM_NUM_TUS) if needed.
NUM_TUS = 32

# --- combination space (mirrors the factory chain in plugins/nvhm/src/nvhm_map_table.cpp) --------
# Feature-gated dimensions map the NVE feature name (lowercase entry of NVE_FEATURES) to the type
# used by the corresponding factory switch case.
KEYS = [
    ("ht_key_8", "int8_t"),
    ("ht_key_16", "int16_t"),
    ("ht_key_32", "int32_t"),
    ("ht_key_64", "int64_t"),
]

KERNELS = [
    ("ht_kernel_1", "nvhm::default_kernel1_t"),
    ("ht_kernel_2", "nvhm::default_kernel2_t"),
    ("ht_kernel_4", "nvhm::default_kernel4_t"),
    ("ht_kernel_8", "nvhm::default_kernel8_t"),
    ("ht_kernel_16", "nvhm::default_kernel16_t"),
    ("ht_kernel_32", "nvhm::default_kernel32_t"),
    ("ht_kernel_64", "nvhm::default_kernel64_t"),
    ("ht_kernel_128", "nvhm::default_kernel128_t"),
    ("ht_kernel_256", "nvhm::default_kernel256_t"),
    ("ht_kernel_512", "nvhm::default_kernel512_t"),
]

# AlwaysZeroPartitioner is unconditional: make_nvhm_map_table_6 uses it whenever
# config.num_partitions == 1, regardless of the enabled partitioner features.
PARTITIONERS = [
    (None, "nve::AlwaysZeroPartitioner"),
    ("ht_part_fnv1a", "nve::FowlerNollVoPartitioner"),
    ("ht_part_murmur3", "nve::Murmur3Partitioner"),
    ("ht_part_rrxmrrxmsx0", "nve::Rrxmrrxmsx0Partitioner"),
    ("ht_part_std_hash", "nve::StdHashPartitioner"),
]

# Unconditional dimensions: every OverflowHandler_t meta type and both values of the runtime
# config bools auto_shrink.
METAS = ["nve::no_meta_t", "nve::lru_meta_t", "nve::lfu_meta_t"]
FLAGS = ["nvhm::flags_t::blobs | nvhm::flags_t::aggressive_prefetch", "nvhm::flags_t::blobs | nvhm::flags_t::aggressive_prefetch | nvhm::flags_t::auto_shrink"]

LICENSE_HEADER = """\
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

// GENERATED FILE - DO NOT EDIT.
// Produced by cmake/gen_nvhm_map_table_instantiations.py.
"""


def enabled(table, features):
    """Return the type names of a feature-gated dimension that are enabled."""
    return [type_name for feature, type_name in table if feature is None or feature in features]


def combinations(features):
    """Yield (key, meta, kernel, auto_shrink, partitioner) per factory combo."""
    for key in enabled(KEYS, features):
        for meta in METAS:
            for flags in FLAGS:
                for kernel in enabled(KERNELS, features):
                    for part in enabled(PARTITIONERS, features):
                        yield (key, meta, flags, kernel, part)


def instantiation(combo):
    """Return one explicit class instantiation definition for a combination."""
    key, meta, flags, kernel, part = combo
    return (
        f"template class nve::plugin::NvhmMapTable<\n"
        f"    nvhm::map<{key}, {meta}, {flags}, {kernel}, nvhm::default_seq_t>,\n"
        f"    {part}>;"
    )


def write_if_different(path, content):
    """Write content only when it differs, to avoid triggering needless recompiles."""
    if os.path.exists(path):
        with open(path, "r") as f:
            if f.read() == content:
                return
    with open(path, "w") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, help="Directory for generated files.")
    parser.add_argument("--features", default="",
                        help="Semicolon-separated NVE_FEATURES list (lowercase).")
    parser.add_argument("--num-tus", type=int, default=NUM_TUS,
                        help=f"Number of generated .cpp files to split the matrix across (default {NUM_TUS}).")
    args = parser.parse_args()

    if args.num_tus < 1:
        parser.error("--num-tus must be >= 1")

    os.makedirs(args.out_dir, exist_ok=True)
    features = {f.strip() for f in args.features.replace(",", ";").split(";") if f.strip()}
    combos = list(combinations(features))
    if not combos:
        print("Warning: no NvhmMapTable combinations for the enabled feature set; "
              "the factory will throw at runtime for every config.")

    # Split the matrix across (at most) num_tus files as evenly as possible. per_file is the
    # ceiling so no file exceeds it; num_files is then recomputed from per_file so we never emit
    # empty trailing TUs.
    num_files = min(args.num_tus, len(combos))
    per_file = (len(combos) + num_files - 1) // num_files if combos else 0
    num_files = (len(combos) + per_file - 1) // per_file if combos else 0

    width = max(3, len(str(num_files - 1))) if num_files else 3
    sources = []
    for fidx in range(num_files):
        chunk = combos[fidx * per_file:(fidx + 1) * per_file]
        name = f"nvhm_map_table_inst_{fidx:0{width}d}.cpp"
        sources.append(name)
        body = [LICENSE_HEADER,
                '#include "nvhm_map_table_impl.hpp"',
                ""]
        body += [instantiation(c) for c in chunk]
        body.append("")
        write_if_different(os.path.join(args.out_dir, name), "\n".join(body))

    # Remove any stale .cpp files from a previous run with a different per-file count.
    for existing in os.listdir(args.out_dir):
        if existing.startswith("nvhm_map_table_inst_") and existing.endswith(".cpp") \
                and existing not in sources:
            os.remove(os.path.join(args.out_dir, existing))

    # CMake fragment listing the generated sources.
    abs_sources = [os.path.join(args.out_dir, s).replace("\\", "/") for s in sources]
    cmake_lines = ["# GENERATED FILE - DO NOT EDIT.",
                   "set(NVHM_GENERATED_SOURCES"]
    cmake_lines += [f"  {s}" for s in abs_sources]
    cmake_lines += [")", ""]
    write_if_different(os.path.join(args.out_dir, "nvhm_map_table_generated.cmake"),
                       "\n".join(cmake_lines))

    print(f"Generated {len(combos)} NvhmMapTable instantiations across {num_files} TUs "
          f"(<= {per_file}/file) in {args.out_dir}")


if __name__ == "__main__":
    main()
