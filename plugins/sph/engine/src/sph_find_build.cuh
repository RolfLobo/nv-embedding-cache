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

// ---------------------------------------------------------------------------
// find — fingerprint compare.
//
// ---------------------------------------------------------------------------
// HAS_MASK threads an in/out hit mask (roundup(num_keys/64) uint64 words, bit i for
// key i) through the lookup: a bit already set on entry means the key was resolved by
// an earlier tier, so it is skipped entirely; a bit is OR-ed in for every key this
// kernel resolves. HAS_MASK=false is codegen-identical to the unmasked kernel.
template<typename HASH_TYPE, typename DATA_TYPE, int32_t SUBWARP_WIDTH, int32_t KEYS_PER_THREAD,
         bool HAS_MASK>
__global__ void gather_keys_sph_v3_kernel(
    const int64_t* keys,
    int64_t num_keys,
    const SPHV2PDE* d_pdes,
    const SPHV3PTE* d_ptes,
    int64_t num_buckets,
    int64_t num_shards,
    int64_t global_seed,
    int64_t num_pages_per_shard,
    int64_t page_size,
    const int8_t* d_value_payload,
    int64_t value_size,
    int64_t payload_shard_stride_bytes,
    const int8_t* d_default_value,
    int8_t* out_values,
    int64_t out_stride,
    uint64_t* d_hit_mask)
{
    const int32_t lane_in_sw   = threadIdx.x % SUBWARP_WIDTH;
    const int32_t sw_in_block  = threadIdx.x / SUBWARP_WIDTH;
    const int32_t sw_per_block = blockDim.x / SUBWARP_WIDTH;
    const int32_t my_tid_base  = ((int32_t)blockIdx.x * sw_per_block + sw_in_block)
                                 * SUBWARP_WIDTH * KEYS_PER_THREAD
                                 + lane_in_sw * KEYS_PER_THREAD;

    const int8_t* local_src[KEYS_PER_THREAD];
    DATA_TYPE local_values[KEYS_PER_THREAD];
    bool is_hit[KEYS_PER_THREAD];
    // skip[j] folds "past the end of the batch", "already resolved upstream" and
    // "bucket has no valid PDE" 
    bool skip[KEYS_PER_THREAD];
    {
        HASH_TYPE   akeys[KEYS_PER_THREAD];
        SPHV2PDE pdes[KEYS_PER_THREAD];
        SPHV3PTE ptes[KEYS_PER_THREAD];

        #pragma unroll
        for (int32_t j = 0; j < KEYS_PER_THREAD; j++) {
            int32_t tid = my_tid_base + j;
            local_src[j] = nullptr;
            is_hit[j]    = false;
            skip[j]      = (tid >= num_keys);
            if (skip[j]) continue;
            if (HAS_MASK && ((d_hit_mask[tid >> 6] >> (tid & 63)) & 1ull)) {
                skip[j] = true;
                continue;
            }
            akeys[j] = (HASH_TYPE)keys[tid];
        }

        int32_t bucket_indices[KEYS_PER_THREAD];
        #pragma unroll
        for (int32_t j = 0; j < KEYS_PER_THREAD; j++) {
            if (skip[j]) continue;
            bucket_indices[j] = (int32_t)bucket_hash(akeys[j], global_seed, num_buckets);
            pdes[j] = d_pdes[bucket_indices[j]];
            if (!valid_pde(pdes[j])) {
                local_src[j] = d_default_value;
                skip[j] = true;
                continue;
            }
            akeys[j] = avalanche_key(akeys[j]);
        }

        #pragma unroll
        for (int32_t j = 0; j < KEYS_PER_THREAD; j++) {
            if (skip[j]) continue;
            int32_t phys_shard = compute_phys_shard(bucket_indices[j], num_shards, pdes[j].shard_bit);
            auto bucket_offset     = expand_bucket_offset(pdes[j].bucket_offset);
            auto alpha_bucket_size = expand_bucket_size(pdes[j].bucket_size);
            auto pilot             = expand_pilot(pdes[j].pilot);
            auto slot              = hash_avalanched(akeys[j], pilot, alpha_bucket_size);
            int64_t page_in_shard = (int64_t)pdes[j].page_index * page_size;
            local_src[j] = d_value_payload + (int64_t)phys_shard * payload_shard_stride_bytes
                         + page_in_shard * value_size;
            ptes[j]      = d_ptes[pte_slot_index(phys_shard, bucket_offset, slot)];
        }

        #pragma unroll
        for (int32_t j = 0; j < KEYS_PER_THREAD; j++) {
            int32_t tid = my_tid_base + j;
            if (skip[j]) continue;
            uint8_t po = (uint8_t)ptes[j].page_offset;
            if (po == 0xff) {
                local_src[j] = d_default_value;
                continue;
            }
            local_src[j] += (int64_t)po * value_size;
            uint64_t expected_fp = pte_fingerprint_of((uint64_t)keys[tid], num_buckets);
            if ((uint64_t)ptes[j].fingerprint != expected_fp) {
                local_src[j] = d_default_value;
                continue;
            }
            is_hit[j] = true;
        }
    }

    const uint32_t mask = __activemask();

    if (HAS_MASK) {
        // A thread's KEYS_PER_THREAD keys are contiguous and my_tid_base is a multiple of
        // KEYS_PER_THREAD, so they always land in one 64-bit word. MASK_GROUP adjacent lanes
        // cover one word between them; OR-reduce across the group and let its leader issue a
        // single atomic instead of one per thread.
        constexpr int32_t MASK_GROUP = (SUBWARP_WIDTH < 64 / KEYS_PER_THREAD)
                                       ? SUBWARP_WIDTH : 64 / KEYS_PER_THREAD;
        unsigned long long bits = 0;
        #pragma unroll
        for (int32_t j = 0; j < KEYS_PER_THREAD; j++) {
            if (is_hit[j]) bits |= 1ull << ((my_tid_base + j) & 63);
        }
        #pragma unroll
        for (int32_t off = 1; off < MASK_GROUP; off <<= 1) {
            bits |= __shfl_xor_sync(mask, bits, off, SUBWARP_WIDTH);
        }
        if ((lane_in_sw % MASK_GROUP) == 0 && bits != 0) {
            atomicOr(reinterpret_cast<unsigned long long*>(&d_hit_mask[my_tid_base >> 6]), bits);
        }
    }

    const int32_t elems_per_value = (int32_t)(value_size / sizeof(DATA_TYPE));

    #pragma unroll
    for (int32_t owner = 0; owner < SUBWARP_WIDTH; owner++) {
        int64_t owner_tid_base = __shfl_sync(mask, my_tid_base, owner, SUBWARP_WIDTH);
        uint64_t owner_srcs[KEYS_PER_THREAD];
        #pragma unroll
        for (int32_t j = 0; j < KEYS_PER_THREAD; j++) {
            owner_srcs[j] = __shfl_sync(
                mask, reinterpret_cast<uint64_t>(local_src[j]), owner, SUBWARP_WIDTH);
        }
        for (int32_t k = lane_in_sw; k < elems_per_value; k += SUBWARP_WIDTH) {
            #pragma unroll
            for (int32_t j = 0; j < KEYS_PER_THREAD; j++) {
                if (!owner_srcs[j]) continue;
                const DATA_TYPE* src = reinterpret_cast<const DATA_TYPE*>(owner_srcs[j]);
                local_values[j] = src[k];
            }
            #pragma unroll
            for (int32_t j = 0; j < KEYS_PER_THREAD; j++) {
                if (!owner_srcs[j]) continue;
                int8_t* dst = out_values + (owner_tid_base + j) * out_stride;
                reinterpret_cast<DATA_TYPE*>(dst)[k] = local_values[j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// hash — contains() proxy; fingerprint compare, no payload load.
// ---------------------------------------------------------------------------
template<typename HASH_TYPE, int64_t PAGE_SIZE>
__global__ void hash_sph_v3_kernel(
    const int64_t* d_keys, int64_t num_keys,
    const SPHV2PDE* d_pdes,
    const SPHV3PTE* d_ptes,
    int64_t num_buckets,
    int64_t num_shards,
    int64_t global_seed,
    int64_t payload_shard_stride_bytes,
    int64_t value_size,
    int64_t* d_hashes)
{
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_keys) return;

    int64_t key  = d_keys[tid];
    HASH_TYPE  akey = (HASH_TYPE)key;
    int64_t bucket_idx = bucket_hash(akey, global_seed, num_buckets);

    SPHV2PDE pde       = d_pdes[bucket_idx];
    // Before the PTE address is formed: an invalid PDE has no PTE range, and the
    // slot it would point at may be on VMM this shard never committed.
    if (!valid_pde(pde)) {
        d_hashes[tid] = -1;
        return;
    }
    int32_t  phys_shard = compute_phys_shard(bucket_idx, num_shards, pde.shard_bit);
    int64_t  bucket_offset     = expand_bucket_offset(pde.bucket_offset);
    int64_t  alpha_bucket_size = expand_bucket_size(pde.bucket_size);
    uint64_t pilot             = expand_pilot(pde.pilot);

    HASH_TYPE  akey_av = avalanche_key(akey);
    int64_t slot    = hash_avalanched(akey_av, pilot, alpha_bucket_size);

    SPHV3PTE pte = d_ptes[pte_slot_index(phys_shard, bucket_offset, slot)];
    uint8_t  po  = (uint8_t)pte.page_offset;
    if (po == 0xff) {
        d_hashes[tid] = -1;
        return;
    }
    uint64_t expected_fp = pte_fingerprint_of((uint64_t)key, num_buckets);
    if ((uint64_t)pte.fingerprint != expected_fp) {
        d_hashes[tid] = -1;
        return;
    }
    
    int64_t payload_location =
        (int64_t)phys_shard * payload_shard_stride_bytes
      + ((int64_t)pde.page_index * PAGE_SIZE + (int64_t)po) * value_size;
    d_hashes[tid] = payload_location;
}

// ---------------------------------------------------------------------------
// assign — value scatter (SPHV3PTE stride). Copy of assign_keys_sph_v2b_kernel.
// ---------------------------------------------------------------------------
template<typename HASH_TYPE, typename DATA_TYPE, int32_t SUBWARP_WIDTH, int32_t KEYS_PER_THREAD>
__global__ void assign_keys_sph_kernel(
    // values is base + index*values_stride: the caller's rows with
    // values_stride == value_size, or a single broadcast row with stride 0.
    const int64_t* keys, const int8_t* values, int64_t values_stride, int64_t num_keys,
    const SPHV2PDE* d_pdes, const SPHV3PTE* d_ptes,
    int64_t num_buckets, int64_t num_shards, int64_t global_seed,
    int64_t num_pages_per_shard, int64_t page_size,
    int8_t* d_value_payload, int64_t value_size, int64_t payload_shard_stride_bytes)
{
    const int32_t lane_in_sw   = threadIdx.x % SUBWARP_WIDTH;
    const int32_t sw_in_block  = threadIdx.x / SUBWARP_WIDTH;
    const int32_t sw_per_block = blockDim.x / SUBWARP_WIDTH;
    const int64_t my_tid_base  = ((int64_t)blockIdx.x * sw_per_block + sw_in_block)
                                 * SUBWARP_WIDTH * KEYS_PER_THREAD
                                 + lane_in_sw * KEYS_PER_THREAD;

    int8_t* local_dst[KEYS_PER_THREAD];
    #pragma unroll
    for (int32_t j = 0; j < KEYS_PER_THREAD; j++) {
        int64_t tid = my_tid_base + j;
        local_dst[j] = nullptr;
        if (tid >= num_keys) continue;
        int64_t raw_key = keys[tid];
        HASH_TYPE key = (HASH_TYPE)raw_key;
        int64_t bidx = bucket_hash(key, global_seed, num_buckets);
        auto pde    = d_pdes[bidx];
        if (!valid_pde(pde)) continue;
        int32_t sidx = compute_phys_shard(bidx, num_shards, pde.shard_bit);
        auto bucket_offset = expand_bucket_offset(pde.bucket_offset);
        auto alpha_bsz     = expand_bucket_size(pde.bucket_size);
        auto coeff         = expand_pilot(pde.pilot);
        auto slot          = hash_index(key, coeff, alpha_bsz);
        SPHV3PTE pte       = d_ptes[pte_slot_index(sidx, bucket_offset, slot)];
        uint8_t  page_off  = (uint8_t)pte.page_offset;
        if (page_off == 0xff) continue;
        if ((uint64_t)pte.fingerprint != pte_fingerprint_of((uint64_t)raw_key, num_buckets)) continue;
        local_dst[j] = d_value_payload + (int64_t)sidx * payload_shard_stride_bytes
                     + ((int64_t)pde.page_index * page_size + page_off) * value_size;
    }

    const uint32_t mask = __activemask();
    const int32_t elems_per_value = (int32_t)(value_size / sizeof(DATA_TYPE));

    #pragma unroll
    for (int32_t owner = 0; owner < SUBWARP_WIDTH; owner++) {
        #pragma unroll
        for (int32_t j = 0; j < KEYS_PER_THREAD; j++) {
            uint64_t owner_dst_u = __shfl_sync(
                mask, reinterpret_cast<uint64_t>(local_dst[j]), owner, SUBWARP_WIDTH);
            int8_t* dst = reinterpret_cast<int8_t*>(owner_dst_u);
            if (!dst) continue;
            int64_t owner_tid_base = __shfl_sync(mask, my_tid_base, owner, SUBWARP_WIDTH);
            const int8_t* src = values + (owner_tid_base + j) * values_stride;
            #pragma unroll
            for (int32_t k = lane_in_sw; k < elems_per_value; k += SUBWARP_WIDTH) {
                reinterpret_cast<DATA_TYPE*>(dst)[k] =
                    reinterpret_cast<const DATA_TYPE*>(src)[k];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// build-path helpers.
// ---------------------------------------------------------------------------
template<int64_t NUM_PAGES_PER_SHARD, int64_t PAGE_SIZE>
__device__ __noinline__ void allocate_page_slot_global(
    int64_t* d_shard_page_offset_global_counter,
    int64_t shard_idx,
    int64_t bucket_size,
    int64_t& out_page_index,
    int64_t& out_page_offset)
{
    uint64_t start_page = 0;
    uint64_t end_page = 1;
    uint64_t global_offset = 0;
    while (start_page != end_page) {
        global_offset = atomicAdd((unsigned long long*)(d_shard_page_offset_global_counter + shard_idx), (unsigned long long)bucket_size);
        start_page = global_offset / PAGE_SIZE;
        end_page = (global_offset + bucket_size) / PAGE_SIZE;
    }
    out_page_index = static_cast<int64_t>(start_page);
    out_page_offset = static_cast<int64_t>(global_offset % PAGE_SIZE);
}

__global__ inline void calculate_buckets_sph_for_build_kernel(
    const int64_t* d_indices, int64_t num_indices, int64_t num_buckets, int64_t global_seed,
    int64_t max_bucket_size, int64_t* d_buckets, int64_t* d_bucket_sizes,
    int64_t* d_failure_counters)
{
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_indices) return;
    auto key = d_indices[tid];
    uint64_t index = static_cast<uint64_t>(key);
    int64_t bucket_index = bucket_hash(index, global_seed, num_buckets);
    auto next_size = atomicAdd((unsigned long long*)(d_bucket_sizes + bucket_index), 1ull);
    if ((int64_t)next_size >= max_bucket_size) {
        if (d_failure_counters) {
            atomicAdd((unsigned long long*)&d_failure_counters[FC_DROPPED_KEYS], 1ull);
            // One bucket-level report for the bucket that first overflowed.
            if ((int64_t)next_size == max_bucket_size)
                atomicAdd((unsigned long long*)&d_failure_counters[FC_MAX_BUCKET_REACHED], 1ull);
        }
        return;
    }
    d_buckets[bucket_index * max_bucket_size + next_size] = key;
}

inline void call_calculate_buckets_sph_for_build(
    const int64_t* d_indices, int64_t num_indices,
    int64_t num_buckets, int64_t max_bucket_size, int64_t global_seed,
    int64_t* d_buckets, int64_t* d_bucket_sizes, int64_t* d_failure_counters,
    cudaStream_t stream)
{
    cudaMemsetAsync(d_bucket_sizes, 0x0, num_buckets * sizeof(int64_t), stream);
    constexpr uint32_t block_size = 128;
    dim3 block(block_size);
    dim3 grid(static_cast<uint32_t>(ceil_div(num_indices, static_cast<int64_t>(block_size))));
    calculate_buckets_sph_for_build_kernel<<<grid, block, 0, stream>>>(
        d_indices, num_indices, num_buckets, global_seed, max_bucket_size, d_buckets,
        d_bucket_sizes, d_failure_counters);
    SPH_CHECK(cudaGetLastError());
}

template<typename HASH_TYPE, int64_t NUM_PAGES_PER_SHARD, int64_t PAGE_SIZE>
__global__ void calculate_pilots_sph_for_build_kernel(const int64_t* d_buckets,
    // In/out: read as the raw arrival count, written back as the count that
    // actually landed (clamped, or 0 when pilot search failed). Each bucket index
    // is owned by exactly one warp, so the in-place update cannot race.
    int64_t* d_bucket_sizes,
    int64_t num_buckets,
    int64_t num_shards,
    int64_t max_bucket_size,
    int64_t target_exponent,
    SPHV2PDE* d_pdes,
    int64_t* d_page_offsets,
    int64_t* d_bucket_offsets,
    int64_t* d_shard_page_counter,
    int64_t* d_shard_page_offset_global_counter,
    int64_t* d_shard_page_offset_counter,
    int64_t* d_failure_counters)
{
    extern __shared__ int8_t shmem[];
    HASH_TYPE* index_pool = reinterpret_cast<HASH_TYPE*>(shmem + threadIdx.y * (max_bucket_size * sizeof(HASH_TYPE) + sizeof(int64_t)));
    uint32_t bucket_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (bucket_idx >= num_buckets) return;
    // Clamped: calculate_buckets leaves the raw arrival count, which can exceed
    // the max_bucket_size slots the bucket actually has. Only that many keys were
    // stored, and the overflow was already counted as dropped.
    auto bucket_size = d_bucket_sizes[bucket_idx];
    if (bucket_size > max_bucket_size) bucket_size = max_bucket_size;
    if (bucket_size == 0) return;
    auto alpha_bucket_size = compute_alpha_bucket_size(bucket_size, target_exponent, alpha_bucket_size_factor);
    auto shard_idx = shard_hash(bucket_idx, num_shards);
    for (int32_t i = threadIdx.x; i < bucket_size; i += blockDim.x) {
        index_pool[i] = static_cast<HASH_TYPE>(d_buckets[bucket_idx * max_bucket_size + i]);
    }

    uint32_t* flag = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(index_pool) + max_bucket_size * sizeof(HASH_TYPE));
    if (threadIdx.x == 0) {
        *flag = 0;
        d_pdes[bucket_idx].bucket_offset = atomicAdd((unsigned long long*)(&d_bucket_offsets[shard_idx]), compress_bucket_offset(alpha_bucket_size));
        d_pdes[bucket_idx].bucket_size = compress_bucket_size(bucket_size);
        int64_t alloc_page = -1;
        int64_t alloc_offset = -1;
        allocate_page_slot_global<NUM_PAGES_PER_SHARD, PAGE_SIZE>(
            d_shard_page_offset_global_counter,
            shard_idx, bucket_size, alloc_page, alloc_offset);
        d_pdes[bucket_idx].page_index = alloc_page;
        d_page_offsets[bucket_idx] = alloc_offset;
        atomicMax((unsigned long long*)(d_shard_page_offset_counter + shard_idx * NUM_PAGES_PER_SHARD + alloc_page), (unsigned long long)(alloc_offset + bucket_size));
    }
    __syncwarp();
    auto iter_idx = threadIdx.x + 1;
    int64_t i = 0;
    constexpr int64_t max_iterations = (1ll << pilot_bits) / 32;
    while (*flag == 0 && i < max_iterations) {
        uint64_t bitmask[4] = {0,0,0,0};
        bool unique = true;
        uint64_t random_coeff_val = expand_pilot(iter_idx);
        for (int32_t j = 0; j < bucket_size; j++) {
            HASH_TYPE hash = hash_index(index_pool[j], random_coeff_val, alpha_bucket_size);
            uint64_t bit = 1ull << (hash % 64);
            unique &= !(bitmask[hash/64] & bit);
            bitmask[hash/64] |= bit;
        }
        auto mask = __ballot_sync(0xffffffff, unique);
        if (mask != 0) {
            auto found_thread = __ffs(mask) - 1;
            auto pilot = __shfl_sync(0xffffffff, iter_idx, found_thread, 32);
            if (threadIdx.x == 0) {
                d_pdes[bucket_idx].pilot = pilot;
                d_bucket_sizes[bucket_idx] = bucket_size;
            }
            return;
        }
        __syncwarp();
        iter_idx += blockDim.x;
        i++;
    }

    // Every pilot exhausted. The PDE keeps pilot == 0, so valid_pde() is false
    // and the bucket is invisible to every reader — which is safe but used to be
    // silent, losing the whole bucket with no signal. Report it, and zero the
    // size so build_ptes publishes no PTEs for a bucket that has no usable pilot.
    // Duplicate keys land here too: two copies always collide, so no pilot can
    // ever separate them.
    if (threadIdx.x == 0) {
        d_bucket_sizes[bucket_idx] = 0;
        if (d_failure_counters) {
            atomicAdd((unsigned long long*)&d_failure_counters[FC_PILOT_FAILURE], 1ull);
            atomicAdd((unsigned long long*)&d_failure_counters[FC_DROPPED_KEYS],
                      (unsigned long long)bucket_size);
        }
    }
}

template<int64_t NUM_PAGES_PER_SHARD, int64_t PAGE_SIZE>
inline void call_calculate_pilots_sph_for_build(int64_t* d_buckets,
    int64_t* d_bucket_sizes,
    int64_t num_buckets,
    int64_t num_shards,
    int64_t max_bucket_size,
    int64_t target_exponent,
    SPHV2PDE* d_pdes,
    int64_t* d_page_offsets,
    int64_t* d_bucket_offsets,
    int64_t* d_shard_page_counter,
    int64_t* d_shard_page_offset_counter,
    int64_t* d_shard_page_offset_global_counter,
    int64_t* d_failure_counters,
    cudaStream_t stream)
{
    dim3 block(32,4);
    dim3 grid(static_cast<uint32_t>((num_buckets + block.y - 1) / block.y));
    size_t shared_memory_size = max_bucket_size * (sizeof(int64_t)) * block.y + sizeof(int64_t) * block.y;
    calculate_pilots_sph_for_build_kernel<uint64_t, NUM_PAGES_PER_SHARD, PAGE_SIZE><<<grid, block, shared_memory_size, stream>>>(d_buckets,
        d_bucket_sizes, num_buckets, num_shards, max_bucket_size, target_exponent,
        d_pdes, d_page_offsets, d_bucket_offsets,
        d_shard_page_counter, d_shard_page_offset_global_counter, d_shard_page_offset_counter,
        d_failure_counters);
    SPH_CHECK(cudaGetLastError());
}

// build path PTE write with fingerprint (no d_payload_keys).
template<typename HASH_TYPE, int64_t NUM_PAGES_PER_SHARD, int64_t PAGE_SIZE>
__global__ void build_ptes_sph_v3_for_build_kernel(
    const int64_t* d_buckets,
    const int64_t* d_bucket_sizes,
    const int64_t* d_page_offsets,
    const SPHV2PDE* d_pdes,
    int64_t num_buckets,
    int64_t num_shards,
    int64_t max_bucket_size,
    SPHV3PTE* d_ptes)
{
    uint32_t bucket_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket_idx >= num_buckets) return;
    auto bucket_size = d_bucket_sizes[bucket_idx];
    auto page_offset = d_page_offsets[bucket_idx];
    auto pde = d_pdes[bucket_idx];
    int32_t shard_idx = compute_phys_shard(bucket_idx, num_shards, pde.shard_bit);
    auto bucket_offset = expand_bucket_offset(pde.bucket_offset);
    auto alpha_bucket_size = expand_bucket_size(pde.bucket_size);
    auto pilot = expand_pilot(pde.pilot);
    for (int32_t i = 0; i < bucket_size; i++) {
        auto key = d_buckets[bucket_idx * max_bucket_size + i];
        auto hash = hash_index(static_cast<HASH_TYPE>(key), pilot, alpha_bucket_size);
        SPHV3PTE pte;
        pte.page_offset = static_cast<uint8_t>(page_offset + i);
        pte.fingerprint = pte_fingerprint_of((uint64_t)key, num_buckets);
        d_ptes[pte_slot_index(shard_idx, bucket_offset, hash)] = pte;
    }
}

// ============================================================================
// Dispatchers
// ============================================================================

template<typename HASH_TYPE, typename DATA_TYPE, int32_t KEYS_PER_THREAD, bool HAS_MASK>
inline void call_gather_keys_sph_v3_typed(
    uint32_t subwarp_width,
    const int64_t* keys, int64_t num_keys,
    const SPHV2PDE* d_pdes, const SPHV3PTE* d_ptes,
    int64_t num_buckets, int64_t num_shards, int64_t global_seed,
    int64_t num_pages_per_shard, int64_t page_size,
    const int8_t* d_value_payload, int64_t value_size,
    int64_t payload_shard_stride_bytes,
    const int8_t* d_default_value,
    int8_t* out_values, int64_t out_stride, uint64_t* d_hit_mask, cudaStream_t stream)
{
    constexpr int32_t BLOCK_SIZE = 128;
    dim3 block(BLOCK_SIZE);
    dim3 grid(static_cast<uint32_t>((num_keys + (int64_t)BLOCK_SIZE * KEYS_PER_THREAD - 1)
                                    / ((int64_t)BLOCK_SIZE * KEYS_PER_THREAD)));
    switch (subwarp_width) {
        case  2: gather_keys_sph_v3_kernel<HASH_TYPE, DATA_TYPE,  2, KEYS_PER_THREAD, HAS_MASK><<<grid, block, 0, stream>>>(keys, num_keys, d_pdes, d_ptes, num_buckets, num_shards, global_seed, num_pages_per_shard, page_size, d_value_payload, value_size, payload_shard_stride_bytes, d_default_value, out_values, out_stride, d_hit_mask); break;
        case  4: gather_keys_sph_v3_kernel<HASH_TYPE, DATA_TYPE,  4, KEYS_PER_THREAD, HAS_MASK><<<grid, block, 0, stream>>>(keys, num_keys, d_pdes, d_ptes, num_buckets, num_shards, global_seed, num_pages_per_shard, page_size, d_value_payload, value_size, payload_shard_stride_bytes, d_default_value, out_values, out_stride, d_hit_mask); break;
        case  8: gather_keys_sph_v3_kernel<HASH_TYPE, DATA_TYPE,  8, KEYS_PER_THREAD, HAS_MASK><<<grid, block, 0, stream>>>(keys, num_keys, d_pdes, d_ptes, num_buckets, num_shards, global_seed, num_pages_per_shard, page_size, d_value_payload, value_size, payload_shard_stride_bytes, d_default_value, out_values, out_stride, d_hit_mask); break;
        case 16: gather_keys_sph_v3_kernel<HASH_TYPE, DATA_TYPE, 16, KEYS_PER_THREAD, HAS_MASK><<<grid, block, 0, stream>>>(keys, num_keys, d_pdes, d_ptes, num_buckets, num_shards, global_seed, num_pages_per_shard, page_size, d_value_payload, value_size, payload_shard_stride_bytes, d_default_value, out_values, out_stride, d_hit_mask); break;
        case 32: gather_keys_sph_v3_kernel<HASH_TYPE, DATA_TYPE, 32, KEYS_PER_THREAD, HAS_MASK><<<grid, block, 0, stream>>>(keys, num_keys, d_pdes, d_ptes, num_buckets, num_shards, global_seed, num_pages_per_shard, page_size, d_value_payload, value_size, payload_shard_stride_bytes, d_default_value, out_values, out_stride, d_hit_mask); break;
        // Unreachable via sph_copy_subwarp_width (clamps to a power of two in
        // [2,32]), but checked in release: falling through launched no kernel
        // at all and left the output buffer uninitialized.
        default: SPH_CHECK(false, "unsupported subwarp width ", subwarp_width);
    }
    SPH_CHECK(cudaGetLastError());
}

template<typename HASH_TYPE, int32_t KEYS_PER_THREAD>
inline void call_gather_keys_sph_v3(
    const int64_t* keys, int64_t num_keys,
    const SPHV2PDE* d_pdes, const SPHV3PTE* d_ptes,
    int64_t num_buckets, int64_t num_shards, int64_t global_seed,
    int64_t num_pages_per_shard, int64_t page_size,
    const int8_t* d_value_payload, int64_t value_size,
    int64_t payload_shard_stride_bytes,
    const int8_t* d_default_value,
    int8_t* out_values, int64_t out_stride, uint64_t* d_hit_mask, cudaStream_t stream)
{
    // The vectorized store writes at out_stride, so the copy type must divide both sizes.
    // For the power-of-two divisors SPH_DISPATCH_COPY_T tests, (a|b) % n == 0 iff both are.
    SPH_DISPATCH_COPY_T(value_size | out_stride, CopyT, {
        uint32_t subwarp_width = sph_copy_subwarp_width(value_size / (int64_t)sizeof(CopyT));
        if (d_hit_mask != nullptr) {
            call_gather_keys_sph_v3_typed<HASH_TYPE, CopyT, KEYS_PER_THREAD, true>(
                subwarp_width, keys, num_keys, d_pdes, d_ptes, num_buckets, num_shards,
                global_seed, num_pages_per_shard, page_size, d_value_payload, value_size,
                payload_shard_stride_bytes, d_default_value, out_values, out_stride, d_hit_mask, stream);
        } else {
            call_gather_keys_sph_v3_typed<HASH_TYPE, CopyT, KEYS_PER_THREAD, false>(
                subwarp_width, keys, num_keys, d_pdes, d_ptes, num_buckets, num_shards,
                global_seed, num_pages_per_shard, page_size, d_value_payload, value_size,
                payload_shard_stride_bytes, d_default_value, out_values, out_stride, nullptr, stream);
        }
    });
}

template<typename HASH_TYPE, int64_t PAGE_SIZE>
inline void call_hash_sph_v3(
    const int64_t* d_keys, int64_t num_keys,
    const SPHV2PDE* d_pdes, const SPHV3PTE* d_ptes,
    int64_t num_buckets, int64_t num_shards, int64_t global_seed,
    int64_t payload_shard_stride_bytes, int64_t value_size,
    int64_t* d_hashes, cudaStream_t stream)
{
    constexpr uint32_t block_size = 128;
    dim3 block(block_size);
    dim3 grid(static_cast<uint32_t>(ceil_div(num_keys, static_cast<int64_t>(block_size))));
    hash_sph_v3_kernel<HASH_TYPE, PAGE_SIZE><<<grid, block, 0, stream>>>(
        d_keys, num_keys, d_pdes, d_ptes, num_buckets, num_shards, global_seed,
        payload_shard_stride_bytes, value_size, d_hashes);
    SPH_CHECK(cudaGetLastError());
}

template<typename HASH_TYPE, typename DATA_TYPE, int32_t KEYS_PER_THREAD>
inline void call_assign_keys_sph_typed(
    uint32_t subwarp_width,
    const int64_t* keys, const int8_t* values, int64_t values_stride, int64_t num_keys,
    const SPHV2PDE* d_pdes, const SPHV3PTE* d_ptes,
    int64_t num_buckets, int64_t num_shards, int64_t global_seed,
    int64_t num_pages_per_shard, int64_t page_size,
    int8_t* d_value_payload, int64_t value_size,
    int64_t payload_shard_stride_bytes, cudaStream_t stream)
{
    constexpr int32_t BLOCK_SIZE = 128;
    dim3 block(BLOCK_SIZE);
    dim3 grid(static_cast<uint32_t>((num_keys + (int64_t)BLOCK_SIZE * KEYS_PER_THREAD - 1)
                                    / ((int64_t)BLOCK_SIZE * KEYS_PER_THREAD)));
    switch (subwarp_width) {
        case  2: assign_keys_sph_kernel<HASH_TYPE, DATA_TYPE,  2, KEYS_PER_THREAD><<<grid, block, 0, stream>>>(keys, values, values_stride, num_keys, d_pdes, d_ptes, num_buckets, num_shards, global_seed, num_pages_per_shard, page_size, d_value_payload, value_size, payload_shard_stride_bytes); break;
        case  4: assign_keys_sph_kernel<HASH_TYPE, DATA_TYPE,  4, KEYS_PER_THREAD><<<grid, block, 0, stream>>>(keys, values, values_stride, num_keys, d_pdes, d_ptes, num_buckets, num_shards, global_seed, num_pages_per_shard, page_size, d_value_payload, value_size, payload_shard_stride_bytes); break;
        case  8: assign_keys_sph_kernel<HASH_TYPE, DATA_TYPE,  8, KEYS_PER_THREAD><<<grid, block, 0, stream>>>(keys, values, values_stride, num_keys, d_pdes, d_ptes, num_buckets, num_shards, global_seed, num_pages_per_shard, page_size, d_value_payload, value_size, payload_shard_stride_bytes); break;
        case 16: assign_keys_sph_kernel<HASH_TYPE, DATA_TYPE, 16, KEYS_PER_THREAD><<<grid, block, 0, stream>>>(keys, values, values_stride, num_keys, d_pdes, d_ptes, num_buckets, num_shards, global_seed, num_pages_per_shard, page_size, d_value_payload, value_size, payload_shard_stride_bytes); break;
        case 32: assign_keys_sph_kernel<HASH_TYPE, DATA_TYPE, 32, KEYS_PER_THREAD><<<grid, block, 0, stream>>>(keys, values, values_stride, num_keys, d_pdes, d_ptes, num_buckets, num_shards, global_seed, num_pages_per_shard, page_size, d_value_payload, value_size, payload_shard_stride_bytes); break;
        // Unreachable via sph_copy_subwarp_width (clamps to a power of two in
        // [2,32]), but checked in release: falling through launched no kernel
        // at all and left the output buffer uninitialized.
        default: SPH_CHECK(false, "unsupported subwarp width ", subwarp_width);
    }
    SPH_CHECK(cudaGetLastError());
}

// Picks the copy element type by value_size divisibility, then dispatches.
template<typename HASH_TYPE, int32_t KEYS_PER_THREAD>
inline void call_assign_keys_sph(
    const int64_t* keys, const int8_t* values, int64_t values_stride, int64_t num_keys,
    const SPHV2PDE* d_pdes, const SPHV3PTE* d_ptes,
    int64_t num_buckets, int64_t num_shards, int64_t global_seed,
    int64_t num_pages_per_shard, int64_t page_size,
    int8_t* d_value_payload, int64_t value_size,
    int64_t payload_shard_stride_bytes, cudaStream_t stream)
{
    SPH_DISPATCH_COPY_T(value_size, CopyT, {
        uint32_t subwarp_width = sph_copy_subwarp_width(value_size / (int64_t)sizeof(CopyT));
        call_assign_keys_sph_typed<HASH_TYPE, CopyT, KEYS_PER_THREAD>(
            subwarp_width, keys, values, values_stride, num_keys, d_pdes, d_ptes, num_buckets, num_shards,
            global_seed, num_pages_per_shard, page_size, d_value_payload, value_size,
            payload_shard_stride_bytes, stream);
    });
}

template<typename HASH_TYPE, int64_t NUM_PAGES_PER_SHARD, int64_t PAGE_SIZE>
inline void call_build_ptes_sph_v3_for_build(
    int64_t num_buckets,
    const int64_t* d_buckets, const int64_t* d_bucket_sizes, const int64_t* d_page_offsets,
    const SPHV2PDE* d_pdes, int64_t num_shards, int64_t max_bucket_size,
    SPHV3PTE* d_ptes, cudaStream_t stream)
{
    constexpr uint32_t block_size = 128;
    dim3 block(block_size);
    dim3 grid(static_cast<uint32_t>(ceil_div(num_buckets, static_cast<int64_t>(block_size))));
    build_ptes_sph_v3_for_build_kernel<HASH_TYPE, NUM_PAGES_PER_SHARD, PAGE_SIZE>
        <<<grid, block, 0, stream>>>(
            d_buckets, d_bucket_sizes, d_page_offsets, d_pdes,
            num_buckets, num_shards, max_bucket_size, d_ptes);
    SPH_CHECK(cudaGetLastError());
}

}  // namespace sph
