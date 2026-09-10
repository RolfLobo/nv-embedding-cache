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

#include "sph_common.cuh"
#include <cassert>

namespace sph {

struct alignas(8) DefragInfo {
    SPHV2PDE old_pde;
    SPHV2PDE new_pde;
    int32_t  bucket_idx;
    int32_t  num_active_slots;
    int32_t  dst_page_offset_base;
    int32_t  _pad;
};
static_assert(sizeof(DefragInfo) == 32, "DefragInfo size unexpected");

__device__ __forceinline__ int64_t inverse_shard_hash(int32_t mega, int32_t tid_in_mega, int32_t num_shards) {
    return (int64_t)tid_in_mega * (int64_t)num_shards + (int64_t)mega;
}

template<int64_t NUM_PAGES_PER_SHARD, int64_t PAGE_SIZE>
__device__ __forceinline__ void defrag_alloc_page_slot(
    int64_t* d_shard_page_offset_global_counter,
    int64_t  shard_idx,
    int64_t  bucket_size,
    int64_t& out_page_index,
    int64_t& out_page_offset)
{
    uint64_t start_page = 0;
    uint64_t end_page   = 1;
    uint64_t global_offset = 0;
    while (start_page != end_page) {
        global_offset = atomicAdd(
            (unsigned long long*)(d_shard_page_offset_global_counter + shard_idx),
            (unsigned long long)bucket_size);
        start_page = global_offset / PAGE_SIZE;
        end_page   = (global_offset + bucket_size) / PAGE_SIZE;
    }
    out_page_index  = static_cast<int64_t>(start_page);
    out_page_offset = static_cast<int64_t>(global_offset % PAGE_SIZE);
}

template<int64_t NUM_PAGES_PER_SHARD, int64_t PAGE_SIZE>
__global__ void defrag_allocate_kernel(
    int32_t  src_mega,
    int32_t  num_shards,
    int32_t  new_num_shards,
    int64_t  num_buckets,
    const SPHV2PDE* d_pdes,
    const SPHV3PTE* d_ptes,
    int64_t* d_shard_page_offset_global_counter,
    int64_t* d_shard_page_offset_counter,
    int64_t* d_bucket_offsets,
    DefragInfo* d_defrag_info,
    int32_t* d_active_count)
{
    int32_t tid_in_mega = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t bucket_idx  = inverse_shard_hash(src_mega, tid_in_mega, num_shards);
    if (bucket_idx >= num_buckets) return;
    SPHV2PDE old_pde = d_pdes[bucket_idx];
    if (!valid_pde(old_pde)) return;

    int32_t target_mega    = (int32_t)shard_hash(bucket_idx, new_num_shards);
    int32_t flip           = (target_mega == src_mega) ? 1 : 0;
    uint32_t new_shard_bit = (uint32_t)((int32_t)old_pde.shard_bit ^ flip);

    int32_t src_phys = compute_phys_shard(bucket_idx, num_shards,     old_pde.shard_bit);
    int32_t dst_phys = compute_phys_shard(bucket_idx, new_num_shards, new_shard_bit);

    int32_t alpha             = (int32_t)expand_bucket_size(old_pde.bucket_size);
    int32_t old_bucket_offset = (int32_t)expand_bucket_offset(old_pde.bucket_offset);

    int32_t num_active = 0;
    for (int32_t i = 0; i < alpha; i++) {
        uint8_t po = (uint8_t)d_ptes[pte_slot_index(src_phys, old_bucket_offset, i)].page_offset;
        if (po != 0xff) num_active++;
    }

    int64_t bucket_offset_new = atomicAdd(
        (unsigned long long*)(d_bucket_offsets + dst_phys),
        (unsigned long long)alpha);
    int64_t alloc_page = -1, alloc_offset = -1;
    defrag_alloc_page_slot<NUM_PAGES_PER_SHARD, PAGE_SIZE>(
        d_shard_page_offset_global_counter, dst_phys, num_active,
        alloc_page, alloc_offset);

    // Publish the per-page fill high-water.
    atomicMax(
        (unsigned long long*)(d_shard_page_offset_counter + dst_phys * NUM_PAGES_PER_SHARD + alloc_page),
        (unsigned long long)(alloc_offset + num_active));

    SPHV2PDE new_pde = old_pde;
    new_pde.shard_bit     = new_shard_bit;
    new_pde.page_index    = (uint32_t)alloc_page;
    new_pde.bucket_offset = (uint32_t)bucket_offset_new;

    int32_t dst = atomicAdd(d_active_count, 1);
    DefragInfo info;
    info.old_pde              = old_pde;
    info.new_pde              = new_pde;
    info.bucket_idx           = (int32_t)bucket_idx;
    info.num_active_slots     = num_active;
    info.dst_page_offset_base = (int32_t)alloc_offset;
    info._pad                 = 0;
    d_defrag_info[dst] = info;
}

template<int64_t NUM_PAGES_PER_SHARD, int64_t PAGE_SIZE, typename VALUE_TYPE>
__global__ void defrag_commit_kernel(
    int32_t   num_shards,
    int32_t   new_num_shards,
    int64_t   value_size,
    int64_t   payload_shard_stride_bytes,
    int32_t   active_count,
    SPHV2PDE* d_pdes,
    SPHV3PTE* d_ptes,
    int8_t*   d_value_payload,
    const DefragInfo* d_defrag_info)
{
    int32_t tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t lane = threadIdx.x & 31;

    DefragInfo info{};
    int32_t my_active = 0;
    // One entry per active slot in the bucket. Stage 1 caps a bucket at
    // max_keys_per_bucket keys, so that constant is the bound here too.
    uint8_t my_src_offsets[max_keys_per_bucket];
    bool active = (tid < active_count);

    int32_t src_phys = 0;
    int32_t dst_phys = 0;

    if (active) {
        info = d_defrag_info[tid];
        src_phys = compute_phys_shard(info.bucket_idx, num_shards,     info.old_pde.shard_bit);
        dst_phys = compute_phys_shard(info.bucket_idx, new_num_shards, info.new_pde.shard_bit);

        int32_t alpha             = (int32_t)expand_bucket_size(info.old_pde.bucket_size);
        int32_t old_bucket_offset = (int32_t)expand_bucket_offset(info.old_pde.bucket_offset);
        int32_t new_bucket_offset = (int32_t)expand_bucket_offset(info.new_pde.bucket_offset);

        int32_t j = 0;
        for (int32_t i = 0; i < alpha; i++) {
            SPHV3PTE src_pte = d_ptes[pte_slot_index(src_phys, old_bucket_offset, i)];
            uint8_t po  = (uint8_t)src_pte.page_offset;
            if (po == 0xff) {
                d_ptes[pte_slot_index(dst_phys, new_bucket_offset, i)].page_offset = 0xff;
                continue;
            }
            uint8_t new_off = (uint8_t)(info.dst_page_offset_base + j);
            // Direct uint64 manipulation: bits 0-7 = page_offset, 8-63 = fingerprint.
            uint64_t src_word = *reinterpret_cast<const uint64_t*>(&src_pte);
            uint64_t dst_word = (src_word & ~0xFFULL) | (uint64_t)new_off;
            *reinterpret_cast<uint64_t*>(&d_ptes[pte_slot_index(dst_phys, new_bucket_offset, i)]) = dst_word;
            // Debug-only backstop: alpha runs to 128, but stage 1 guarantees at
            // most max_keys_per_bucket of those slots are occupied.
            assert(j < max_keys_per_bucket);
            my_src_offsets[j] = po;
            j++;
        }
        my_active = j;
    }

    const uint32_t mask = 0xffffffffu;

    auto pde_to_u64 = [](SPHV2PDE p) -> uint64_t {
        uint64_t u; __builtin_memcpy(&u, &p, sizeof(u)); return u;
    };
    auto u64_to_pde = [](uint64_t u) -> SPHV2PDE {
        SPHV2PDE p; __builtin_memcpy(&p, &u, sizeof(p)); return p;
    };
    uint64_t old_pde_bits = pde_to_u64(info.old_pde);
    uint64_t new_pde_bits = pde_to_u64(info.new_pde);

    #pragma unroll 1
    for (int32_t owner = 0; owner < 32; owner++) {
        int32_t  own_active = __shfl_sync(mask, my_active, owner);
        if (own_active == 0) continue;
        int32_t  own_bucket   = __shfl_sync(mask, info.bucket_idx, owner);
        int32_t  own_dst_base = __shfl_sync(mask, info.dst_page_offset_base, owner);
        uint32_t old_lo = __shfl_sync(mask, (uint32_t)(old_pde_bits & 0xffffffffu), owner);
        uint32_t old_hi = __shfl_sync(mask, (uint32_t)(old_pde_bits >> 32),         owner);
        uint32_t new_lo = __shfl_sync(mask, (uint32_t)(new_pde_bits & 0xffffffffu), owner);
        uint32_t new_hi = __shfl_sync(mask, (uint32_t)(new_pde_bits >> 32),         owner);
        SPHV2PDE own_old = u64_to_pde(((uint64_t)old_hi << 32) | old_lo);
        SPHV2PDE own_new = u64_to_pde(((uint64_t)new_hi << 32) | new_lo);

        int32_t own_src_phys = compute_phys_shard(own_bucket, num_shards,     own_old.shard_bit);
        int32_t own_dst_phys = compute_phys_shard(own_bucket, new_num_shards, own_new.shard_bit);
        // Byte base of each shard's payload region (real VMM shard stride) plus
        // the in-shard page offset scaled by value_size.
        int64_t src_page_base = (int64_t)own_src_phys * payload_shard_stride_bytes
                              + (int64_t)own_old.page_index * PAGE_SIZE * value_size;
        int64_t dst_page_base = (int64_t)own_dst_phys * payload_shard_stride_bytes
                              + (int64_t)own_new.page_index * PAGE_SIZE * value_size;

        for (int32_t k = 0; k < own_active; k++) {
            uint32_t src_off = (uint32_t)__shfl_sync(mask, (uint32_t)my_src_offsets[k], owner);
            int64_t src_byte = src_page_base + (int64_t)src_off * value_size;
            int64_t dst_byte = dst_page_base + ((int64_t)own_dst_base + (int64_t)k) * value_size;

            const VALUE_TYPE* sp = reinterpret_cast<const VALUE_TYPE*>(d_value_payload + src_byte);
            VALUE_TYPE*       dp = reinterpret_cast<VALUE_TYPE*>      (d_value_payload + dst_byte);
            const int32_t elems = (int32_t)(value_size / (int64_t)sizeof(VALUE_TYPE));
            for (int32_t e = lane; e < elems; e += 32) dp[e] = sp[e];
        }
    }

    __syncwarp(mask);

    if (active) {
        d_pdes[info.bucket_idx] = info.new_pde;
    }
}

// ============================================================================
// Dispatchers
// ============================================================================

template<int64_t NUM_PAGES_PER_SHARD, int64_t PAGE_SIZE>
inline void call_defrag_allocate(
    int64_t buckets_in_mega,
    int32_t src_mega, int32_t num_shards, int32_t new_num_shards, int64_t num_buckets,
    const SPHV2PDE* d_pdes, const SPHV3PTE* d_ptes,
    int64_t* d_shard_page_offset_global_counter, int64_t* d_shard_page_offset_counter,
    int64_t* d_bucket_offsets,
    DefragInfo* d_defrag_info, int32_t* d_active_count, cudaStream_t stream)
{
    constexpr int32_t kAllocBlock = 128;
    dim3 block(kAllocBlock);
    dim3 grid(static_cast<uint32_t>(ceil_div(buckets_in_mega, static_cast<int64_t>(kAllocBlock))));
    defrag_allocate_kernel<NUM_PAGES_PER_SHARD, PAGE_SIZE>
        <<<grid, block, 0, stream>>>(
            src_mega, num_shards, new_num_shards, num_buckets,
            d_pdes, d_ptes, d_shard_page_offset_global_counter,
            d_shard_page_offset_counter, d_bucket_offsets,
            d_defrag_info, d_active_count);
    SPH_CHECK(cudaGetLastError());
}

template<int64_t NUM_PAGES_PER_SHARD, int64_t PAGE_SIZE>
inline void call_defrag_commit(
    int32_t h_active_count,
    int32_t num_shards, int32_t new_num_shards, int64_t value_size,
    int64_t payload_shard_stride_bytes,
    SPHV2PDE* d_pdes, SPHV3PTE* d_ptes, int8_t* d_value_payload,
    const DefragInfo* d_defrag_info, cudaStream_t stream)
{
    constexpr int32_t kCommitBlock = 128;
    dim3 block(kCommitBlock);
    dim3 grid(static_cast<uint32_t>(ceil_div(h_active_count, kCommitBlock)));
    // Vectorized payload-copy element chosen by value_size divisibility.
    SPH_DISPATCH_COPY_T(value_size, VALUE_TYPE, {
        defrag_commit_kernel<NUM_PAGES_PER_SHARD, PAGE_SIZE, VALUE_TYPE>
            <<<grid, block, 0, stream>>>(
                num_shards, new_num_shards, value_size, payload_shard_stride_bytes,
                h_active_count, d_pdes, d_ptes, d_value_payload, d_defrag_info);
    });
    SPH_CHECK(cudaGetLastError());
}

}  // namespace sph
