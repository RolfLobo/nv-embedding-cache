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

"""Generate explicit-instantiation sources for the find_and_combine kernel.

The find_and_combine kernel (cuda_ops/find_and_combine_kernel.cuh) is instantiated across a large
combination space. Compiling the whole matrix in the single translation unit that uses it
(src/gpu_table.cu) dominates the build time of nve-common.

This script enumerates every combination the dispatch logic can launch and emits:

  * find_and_combine_kernel_extern.inc  - `extern template` declaration per combination. Included
                                          by the kernel header so every TU stops implicitly
                                          instantiating the kernels.
  * find_and_combine_kernel_inst_NNN.cu - explicit instantiation definitions, INSTANTIATIONS_PER_FILE
                                          per file, so the matrix compiles across many TUs in
                                          parallel instead of one.
  * find_and_combine_generated.cmake    - sets FAC_GENERATED_SOURCES to the list of .cu files.

The enumeration mirrors call_find_and_combine_kernel_types_resolved / NVE_FAC_DISPATCH /
NVE_FAC_LAUNCH_MASK; keep it in sync if those change.
"""

import argparse
import os

# --- tunables -----------------------------------------------------------------------------------
# Number of generated .cu translation units the instantiation matrix is split across. Override with
# --num-tus (or -DFAC_NUM_TUS) if needed.
NUM_TUS = 32

# --- combination space (mirrors the dispatch logic) ---------------------------------------------
ELEMENT_TYPES = [
    "Float32", "Float16",
    "QUint8RowwiseF32", "QInt8RowwiseF32",
    "QUint8RowwiseF16", "QInt8RowwiseF16",
]

# (ACC_TYPE, WEIGHT_TYPE) dispatched in call_find_and_combine_kernel. Only Float32 accumulation is
# supported; the accumulator pairs with either a float or a __half weight.
ACC_WEIGHT = [("float", "float"), ("float", "__half")]

OUTPUT_TYPES = ["float", "__half"]

# (INDEX_TYPE, CacheDataT) combinations instantiated in src/gpu_table.cu.
IDX_CACHE = [
    ("int32_t", "typename nve::EmbedCacheSA<int32_t, int32_t>::CacheData"),
    ("int64_t", "typename nve::EmbedCacheSA<int64_t, int64_t>::CacheData"),
    ("int32_t", "typename nve::ECNoCache<int32_t>::CacheData"),
    ("int64_t", "typename nve::ECNoCache<int64_t>::CacheData"),
]

# SZ_ACCUM values instantiated per vector width (Vec1 / Vec2 / Vec4, selected by rowSizeInElements
# alignment), dispatched by call_find_and_combine_kernel_types_resolved.
#
# The kernel loops over the row in chunks of SUBWARP_WIDTH * SZ_ACCUM vectors, so any row length is
# handled regardless of SZ_ACCUM: SZ_ACCUM only trades register pressure against the number of
# passes. The dispatch picks the smallest SZ_ACCUM that covers the row in a single pass, clamped per
# width to min(4, 8 // width) so that vec_width * SZ_ACCUM <= 8 (bounded register pressure) and
# SZ_ACCUM <= 4. We instantiate exactly that clamped range per width; keep this in sync with the
# per-width clamp in the kernel header (call_find_and_combine_kernel_types_resolved).
MAX_SZ_ACCUM = 4
WIDTH_ACCUM = [(width, sz)
               for width in (4, 2, 1)
               for sz in range(1, min(MAX_SZ_ACCUM, 8 // width) + 1)]

# All 16 packed MASK values. The kernel unpacks the bits as:
#   bit0 = FIXED_HOTNESS, bit1 = SUM_POOLING, bit2 = IS_WEIGHTED, bit3 = LOAD_INDICES
# (see get_mask / NVE_FAC_MASK_* in cuda_ops/find_and_combine_kernel.cuh).
MASKS = list(range(16))

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
// Produced by cmake/gen_find_and_combine_instantiations.py.
"""


def combinations():
    """Yield (element, idx, acc, wgt, out, cache, width, sz, mask) for every dispatched kernel."""
    for element in ELEMENT_TYPES:
        for acc, wgt in ACC_WEIGHT:
            for out in OUTPUT_TYPES:
                for idx, cache in IDX_CACHE:
                    for width, sz in WIDTH_ACCUM:
                        for mask in MASKS:
                            yield (element, idx, acc, wgt, out, cache, width, sz, mask)


def template_args(combo):
    """Return the angle-bracket template argument list for nve_fac_launch<...>."""
    element, idx, acc, wgt, out, cache, width, sz, mask = combo
    elem_id = f"DataType_t::{element}"
    input_vec = f"typename QuantizationHelper<{elem_id}>::Vec{width}"
    out_vec = f"typename VecWidthHelper<{out}>::Vec{width}"
    elem_vec = f"typename VecWidthHelper<typename QuantizationHelper<{elem_id}>::ParamType>::Vec{width}"
    acc_vec = f"typename VecWidthHelper<{acc}>::Vec{width}"
    return (f"{elem_id}, {idx}, {acc}, {wgt}, {out}, {input_vec}, {out_vec}, {elem_vec}, {acc_vec}, "
            f"{cache}, {sz}u, {mask}u")


def instantiation(combo, extern):
    """Return one explicit-instantiation (or extern declaration) statement for a combination.

    Instantiates the nve_fac_launch host wrapper (not the __global__ kernel directly): a plain host
    function links across TUs without relocatable device code, and instantiating it co-locates the
    kernel launch / device-code instantiation in the same TU.
    """
    _element, idx, _acc, wgt, _out, cache, _width, _sz, _mask = combo
    prefix = "extern template" if extern else "template"
    return (
        f"{prefix} void nve_fac_launch<{template_args(combo)}>(\n"
        f"    const uint32_t, const int8_t*, const {idx}*, const {idx}*, const {wgt}*, int32_t,\n"
        f"    {cache}, int32_t, void*, cudaStream_t);"
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
    parser.add_argument("--num-tus", type=int, default=NUM_TUS,
                        help=f"Number of generated .cu files to split the matrix across (default {NUM_TUS}).")
    args = parser.parse_args()

    if args.num_tus < 1:
        parser.error("--num-tus must be >= 1")

    os.makedirs(args.out_dir, exist_ok=True)
    combos = list(combinations())

    # Split the matrix across (at most) num_tus files as evenly as possible. per_file is the ceiling
    # so no file exceeds it; num_files is then recomputed from per_file so we never emit empty
    # trailing TUs (e.g. when num_tus exceeds the combination count).
    num_files = min(args.num_tus, len(combos))
    per_file = (len(combos) + num_files - 1) // num_files
    num_files = (len(combos) + per_file - 1) // per_file

    # extern template declaration header.
    extern_lines = [LICENSE_HEADER,
                    "#pragma once",
                    "",
                    "// extern template declarations of the nve_fac_launch host wrapper for every",
                    "// dispatched find_and_combine combination.",
                    ""]
    extern_lines += [instantiation(c, extern=True) for c in combos]
    extern_lines.append("")
    write_if_different(os.path.join(args.out_dir, "find_and_combine_kernel_extern.inc"),
                       "\n".join(extern_lines))

    # Explicit instantiation .cu files, up to per_file instantiations each.
    width = max(3, len(str(num_files - 1)))
    sources = []
    for fidx in range(num_files):
        chunk = combos[fidx * per_file:(fidx + 1) * per_file]
        name = f"find_and_combine_kernel_inst_{fidx:0{width}d}.cu"
        sources.append(name)
        body = [LICENSE_HEADER,
                "// This TU provides nve_fac_launch definitions; skip the (large) extern-declaration",
                "// list in the kernel header to avoid re-parsing it in every generated TU.",
                "#define NVE_FAC_SKIP_EXTERN_DECLS",
                '#include "cuda_ops/find_and_combine_kernel.cuh"',
                "",
                "using namespace nve;",
                ""]
        body += [instantiation(c, extern=False) for c in chunk]
        body.append("")
        write_if_different(os.path.join(args.out_dir, name), "\n".join(body))

    # Remove any stale .cu files from a previous run with a different per-file count.
    for existing in os.listdir(args.out_dir):
        if existing.startswith("find_and_combine_kernel_inst_") and existing.endswith(".cu") \
                and existing not in sources:
            os.remove(os.path.join(args.out_dir, existing))

    # CMake fragment listing the generated sources.
    abs_sources = [os.path.join(args.out_dir, s).replace("\\", "/") for s in sources]
    cmake_lines = ["# GENERATED FILE - DO NOT EDIT.",
                   "set(FAC_GENERATED_SOURCES"]
    cmake_lines += [f"  {s}" for s in abs_sources]
    cmake_lines += [")", ""]
    write_if_different(os.path.join(args.out_dir, "find_and_combine_generated.cmake"),
                       "\n".join(cmake_lines))

    print(f"Generated {len(combos)} find_and_combine instantiations across {num_files} TUs "
          f"(<= {per_file}/file) in {args.out_dir}")


if __name__ == "__main__":
    main()
