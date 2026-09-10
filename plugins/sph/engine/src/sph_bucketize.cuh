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
// ============================================================================
// sph bucket-index kernel + dispatcher.
// ============================================================================
#include "sph_common.cuh"

namespace sph {

inline __global__ void calculate_bucket_index_kernel(
    const int64_t* d_indices, int64_t num_indices, int64_t* d_bucket_indices,
    int64_t num_buckets, int64_t global_seed)
{
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_indices) return;
    uint64_t index = static_cast<uint64_t>(d_indices[tid]);
    d_bucket_indices[tid] = bucket_hash(index, global_seed, num_buckets);
}

inline void call_calculate_bucket_index(
    const int64_t* d_indices, int64_t num_indices, int64_t* d_bucket_indices,
    int64_t num_buckets, int64_t global_seed, cudaStream_t stream)
{
    constexpr uint32_t block_size = 128;
    dim3 block(block_size);
    dim3 grid(static_cast<uint32_t>(ceil_div(num_indices, static_cast<int64_t>(block_size))));
    calculate_bucket_index_kernel<<<grid, block, 0, stream>>>(
        d_indices, num_indices, d_bucket_indices, num_buckets, global_seed);
    SPH_CHECK(cudaGetLastError());
}

}  // namespace sph
