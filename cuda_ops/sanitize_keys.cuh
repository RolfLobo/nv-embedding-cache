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
#include <stdint.h>
#include <cuda_runtime.h>
#include <memory>
#include "cuda_ops/cuda_common.h"
#include "include/buffer_wrapper.hpp"
#include "include/common.hpp"
#include "include/execution_context.hpp"
#include "include/key_utils.hpp"

namespace nve {

template<typename KeyType>
__global__ void sanitize_keys_kernel(const KeyType* __restrict__ keys,
                                   KeyType* __restrict__ out,
                                   const int64_t num_keys,
                                   const uint64_t num_rows,
                                   const KeyType default_row)
{
    const int64_t id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (id >= num_keys) {
      return;
    }

    const KeyType key = keys[id];
    out[id] = key_in_range(key, num_rows) ? key : default_row;
}

// Copies keys into out, replacing every key outside [0, num_rows) with default_row. Lookups on a
// linear (dense) table index it directly by key, so this keeps out of range keys inside the table
// and resolves them to the default row instead of addressing foreign memory.
template<typename KeyType>
void launch_sanitize_keys_kernel(const KeyType* keys,
                                 KeyType* out,
                                 const int64_t num_keys,
                                 const uint64_t num_rows,
                                 const KeyType default_row,
                                 const cudaStream_t stream)
{
    if (num_keys <= 0) {
      return;
    }
    NVE_CHECK_(key_in_range(default_row, num_rows), "Default row is outside the table");

    constexpr uint32_t block_size = 256;
    const auto grid_size = static_cast<uint32_t>((num_keys + block_size - 1) / block_size);
    sanitize_keys_kernel<KeyType><<<grid_size, block_size, 0, stream>>>(
        keys, out, num_keys, num_rows, default_row);
    NVE_CHECK_(cudaGetLastError()); // Check kernel launch didn't generate an error
}

// Sanitizes device resident lookup keys into a context owned buffer and returns it. A negative
// default_row_index disables sanitization, in which case the input keys are returned as is.
template<typename KeyType>
const void* sanitize_lookup_keys(context_ptr_t& ctx,
                                 const void* d_keys,
                                 const int64_t num_keys,
                                 const int64_t num_rows,
                                 const int64_t default_row_index,
                                 const cudaStream_t stream)
{
    if ((default_row_index < 0) || (num_keys <= 0)) {
      return d_keys;
    }
    const auto keys_buffer_size = static_cast<size_t>(num_keys) * sizeof(KeyType);
    auto* sanitized = static_cast<KeyType*>(
        ctx->get_buffer("sanitized_keys", keys_buffer_size, false /*host_alloc*/));
    NVE_CHECK_(sanitized != nullptr, "Failed to allocate sanitized keys buffer");
    launch_sanitize_keys_kernel<KeyType>(static_cast<const KeyType*>(d_keys), sanitized, num_keys,
                                         static_cast<uint64_t>(num_rows),
                                         static_cast<KeyType>(default_row_index), stream);
    return sanitized;
}

// BufferWrapper flavor of the above, for layers that pass keys around as wrappers. The returned
// wrapper is the input one when sanitization is disabled.
template<typename KeyType>
std::shared_ptr<BufferWrapper<const void>> sanitize_lookup_keys(
    context_ptr_t& ctx,
    std::shared_ptr<BufferWrapper<const void>> keys_bw,
    const int64_t num_keys,
    const int64_t num_rows,
    const int64_t default_row_index,
    const cudaStream_t stream)
{
    if ((default_row_index < 0) || (num_keys <= 0)) {
      return keys_bw;
    }
    const auto keys_buffer_size = static_cast<size_t>(num_keys) * sizeof(KeyType);
    const void* d_keys = keys_bw->access_buffer(cudaMemoryTypeDevice, true /*copy_content*/, stream);
    const void* sanitized = sanitize_lookup_keys<KeyType>(ctx, d_keys, num_keys, num_rows,
                                                          default_row_index, stream);
    return std::make_shared<BufferWrapper<const void>>(ctx, "sanitized_keys_wrapper", sanitized,
                                                       keys_buffer_size);
}

}  // namespace nve
