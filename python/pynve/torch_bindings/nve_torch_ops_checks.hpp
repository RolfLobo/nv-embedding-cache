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

// nve_torch_ops_checks.hpp — input validation shared by the CPU and CUDA op
// dispatch TUs (nve_torch_ops_cpu.cpp / nve_torch_ops.cu).
//
// The op schemas declare untyped Tensors and the kernels hand raw data_ptr()s
// to an int64-keyed binding, so dtype and contiguity must be enforced here
// before any pointer reinterpretation.

#pragma once

#include <torch/csrc/stable/tensor.h>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "nve_registry.hpp"

namespace nve {

inline void check_index_tensor(const torch::stable::Tensor& t, const char* name) {
    if (t.scalar_type() != torch::stable::ScalarType::Long) {
        throw std::runtime_error(std::string("nve-torch-ops: ") + name +
                                 " must be an int64 tensor");
    }
    if (!t.is_contiguous()) {
        throw std::runtime_error(std::string("nve-torch-ops: ") + name +
                                 " must be contiguous");
    }
}

// Validates CSR bag offsets at the shape level. The bag ops use torch's
// include_last_offset=True convention: num_bags + 1 entries, so at least two.
// Value-level checks (start at 0, non-decreasing, trailing sentinel equal to
// the number of keys) run in pool_gathered_host, where the offsets are
// host-readable on every entry path.
inline void check_bag_offsets(const torch::stable::Tensor& offsets) {
    check_index_tensor(offsets, "offsets");
    if (offsets.numel() < 2) {
        throw std::runtime_error(
            "nve-torch-ops: offsets must have at least two entries "
            "(num_bags + 1, include_last_offset=True convention)");
    }
}

// Validates per-sample pooling weights and returns their binding dtype tag.
inline int check_pooling_weights(const torch::stable::Tensor& weights,
                                 const int64_t num_keys) {
    if (!weights.is_contiguous()) {
        throw std::runtime_error("nve-torch-ops: weights must be contiguous");
    }
    if (weights.numel() != num_keys) {
        throw std::runtime_error(
            "nve-torch-ops: weights must have one element per key");
    }
    const auto st = weights.scalar_type();
    if (st == torch::stable::ScalarType::Float) {
        return kBindingDtypeFloat32;
    }
    if (st == torch::stable::ScalarType::Half) {
        return kBindingDtypeFloat16;
    }
    throw std::runtime_error(
        "nve-torch-ops: weights must be a float32 or float16 tensor");
}

}  // namespace nve
