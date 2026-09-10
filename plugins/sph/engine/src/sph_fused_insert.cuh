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
#include "sph_insert_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cassert>

namespace sph {

// Per-bucket gather entry.
struct alignas(8) V3bAux {
    int64_t key;       // raw key
    int32_t src_idx;   // d_inverse_buffer index (NEW); 0 otherwise
    uint8_t new_off;   // dst offset in new page
    uint8_t old_off;   // src offset in old page (OLD reloc)
    uint8_t kind;      // 0=in-place no-copy, 1=old-reloc, 2=new
    uint8_t _pad;
};
static_assert(sizeof(V3bAux) == 16, "V3bAux must be 16 bytes");

enum : uint8_t {
    kV3bInPlace  = 0,
    kV3bOldReloc = 1,
    kV3bNew      = 2,
};

struct alignas(16) V3bRelocPair {
    int8_t* src;
    int8_t* dst;
};
static_assert(sizeof(V3bRelocPair) == 16, "V3bRelocPair must be 16 bytes");

// Subwarp-cooperative value copy of a shmem-buffered (src,dst) list. VALUE_TYPE is the
// vectorized copy element chosen by value_size divisibility (uint4/uint32_t/int8_t).
template<int32_t SUB_WARP_DRAIN_SIZE, typename VALUE_TYPE>
__device__ __forceinline__ void v3b_drain_reloc_list(
    const V3bRelocPair* __restrict__ rl, int32_t count, int64_t value_size, int32_t lane)
{
    constexpr int64_t E = (int64_t)sizeof(VALUE_TYPE);
    constexpr int32_t SUBWARPS = 32 / SUB_WARP_DRAIN_SIZE;
    const int32_t sw         = lane / SUB_WARP_DRAIN_SIZE;
    const int32_t lane_in_sw = lane & (SUB_WARP_DRAIN_SIZE - 1);
    for (int32_t k = sw; k < count; k += SUBWARPS) {
        const int8_t* src = rl[k].src;
        int8_t*       dst = rl[k].dst;
        for (int64_t b = (int64_t)lane_in_sw * E; b < value_size; b += (int64_t)SUB_WARP_DRAIN_SIZE * E) {
            VALUE_TYPE v = *reinterpret_cast<const VALUE_TYPE*>(src + b);
            __stcs(reinterpret_cast<VALUE_TYPE*>(dst + b), v);
        }
    }
}

template<typename HASH_TYPE,
         int64_t NUM_PAGES_PER_SHARD, int64_t PAGE_SIZE,
         int64_t MAX_BUCKET_SIZE,
         int32_t NUM_BUCKETS_PER_BLOCK,
         int32_t WARPS_PER_BLOCK,
         int32_t SUB_WARP_DRAIN_SIZE,
         int32_t RL_CAP,
         int32_t S_CAP,
         typename PTE_TYPE,
         typename VALUE_TYPE,
         // The batch carried no values, so every new key takes the map's default
         // row. Compile-time rather than a runtime stride: this kernel is pinned
         // to 48 registers by __launch_bounds__, and one more live 64-bit value
         // across the write loop measurably increased spill traffic.
         bool NEW_VALUES_ARE_DEFAULT>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32, 5) kernel_v3b_fused(
    int64_t num_buckets,
    int64_t num_total_buckets,
    int64_t num_shards,
    const int64_t* __restrict__ d_keys,
    // Source rows for this batch's NEW keys: the caller's values, or the map's
    // default row when NEW_VALUES_ARE_DEFAULT (in which case the stride below
    // folds to 0 and every new key reads that one row).
    const int8_t*  __restrict__ d_new_values,
    const int64_t* __restrict__ d_active_buckets,
    const int64_t* __restrict__ d_bucket_sizes,
    const int64_t* __restrict__ d_offsets,
    const int64_t* __restrict__ d_inverse_buffer,
    const SPHV2PDE* __restrict__ d_pdes,
    PTE_TYPE*           d_ptes,
    int8_t*        d_value_payload,
    int64_t        value_size,
    int64_t        payload_shard_stride_bytes,
    BucketStateV2G* d_bucket_state,
    int64_t*       d_pilot_counters,
    int64_t*       d_failure_counters)
{
    static_assert(RL_CAP >= 32, "RL_CAP must hold a full 32-wide wave");

    // aux/avk hold one entry per gathered key: the surviving existing keys
    // (Phase 1) plus this batch's new keys (Phase 2). Stage 1 guarantees
    // existing_count + bucket_size <= max_keys_per_bucket == S_CAP and drops
    // the bucket as max_bucket_reached otherwise, so the gather cannot overrun.
    // The asserts below are debug-only backstops; they used to be __trap(),
    // which killed the process for a condition that can no longer occur.
    __shared__ V3bAux       aux_pool[WARPS_PER_BLOCK][S_CAP];
    __shared__ HASH_TYPE       avk_pool[WARPS_PER_BLOCK][S_CAP];
    __shared__ V3bRelocPair rl_pool [WARPS_PER_BLOCK][RL_CAP];
    __shared__ int32_t      aux_count_s[WARPS_PER_BLOCK];
    __shared__ int32_t      reloc_cnt_s[WARPS_PER_BLOCK];
    __shared__ int32_t      rl_count_s [WARPS_PER_BLOCK];
    __shared__ int32_t      block_work_counter;

    const int32_t lane    = (int32_t)(threadIdx.x & 31);
    const int32_t wid     = (int32_t)(threadIdx.x / 32);

    if (threadIdx.x == 0) block_work_counter = 0;
    __syncthreads();

    V3bAux*       aux = aux_pool[wid];
    HASH_TYPE*       avk = avk_pool[wid];
    V3bRelocPair* rl  = rl_pool[wid];
    int32_t*      aux_count = &aux_count_s[wid];
    int32_t*      reloc_cnt = &reloc_cnt_s[wid];
    int32_t*      rl_count  = &rl_count_s[wid];

    while (true) {
        int32_t local;
        if (lane == 0) local = atomicAdd(&block_work_counter, 1);
        local = __shfl_sync(0xffffffffu, local, 0);
        if (local >= NUM_BUCKETS_PER_BLOCK) break;

        const int64_t work_idx = (int64_t)blockIdx.x * NUM_BUCKETS_PER_BLOCK + local;
        if (work_idx >= num_buckets) continue;

        const BucketStateV2G st = d_bucket_state[work_idx];
        uint32_t packed = st.packed;
        uint32_t mode   = packed & PK_MODE_MASK;
        if (mode != BUCKET_MODE_V2G_PILOT_SEARCH &&
            mode != BUCKET_MODE_V2G_CLEAR_PTE &&
            mode != BUCKET_MODE_V2G_WRITE_BUCKET_PTE) {
            continue;
        }

        SPHV2PDE new_pde = st.new_pde;
        const int32_t bucket_idx     = (int32_t)d_active_buckets[work_idx];
        const int32_t bucket_size    = (int32_t)d_bucket_sizes[work_idx];
        const int32_t keys_offs_base = (int32_t)d_offsets[work_idx];
        const SPHV2PDE old_pde       = d_pdes[bucket_idx];
        const BucketStatePacked p    = unpack_state(packed);

        const bool full_rehash = (mode != BUCKET_MODE_V2G_WRITE_BUCKET_PTE);
        const bool in_place    = p.is_curr_page_valid;
        const int32_t new_phys = compute_phys_shard(bucket_idx, num_shards, new_pde.shard_bit);
        const int32_t old_phys = compute_phys_shard(bucket_idx, num_shards, old_pde.shard_bit);
        const int64_t pte_base_new = (int64_t)new_phys * pte_entries_per_shard
                                     + (int64_t)expand_bucket_offset(new_pde.bucket_offset);
        const int64_t pte_base_old = (int64_t)old_phys * pte_entries_per_shard
                                     + (int64_t)expand_bucket_offset(old_pde.bucket_offset);
        // Byte base of each shard's payload region (uses the VMM's real,
        // granularity-aligned shard stride, not the packed slot product).
        const int64_t payload_base_new = (int64_t)new_phys * payload_shard_stride_bytes;
        const int64_t payload_base_old = (int64_t)old_phys * payload_shard_stride_bytes;
        uint64_t coeff_new       = expand_pilot(new_pde.pilot);
        const int32_t old_alpha  = p.is_valid_pde ? (int32_t)expand_bucket_size(old_pde.bucket_size) : 0;
        const int32_t new_alpha  = (int32_t)expand_bucket_size(new_pde.bucket_size);
        const uint32_t local_page_offset_base = p.local_page_offset;
        const int32_t new_page_index = (int32_t)new_pde.page_index;
        const int32_t old_page_index = (int32_t)old_pde.page_index;

        if (lane == 0) { *aux_count = 0; *reloc_cnt = 0; *rl_count = 0; }
        __syncwarp();

        // ---- Phase 1: gather surviving OLD PTEs (the NEW range may alias this) ----
        if (full_rehash && p.is_valid_pde && p.existing_count > 0) {
            for (int32_t i = lane; i < old_alpha; i += 32) {
                PTE_TYPE old = d_ptes[pte_base_old + i];
                uint8_t old_off = (uint8_t)old.page_offset;
                if (old_off == 0xff) continue;
                int64_t key = reconstruct_key_from_pte(old, (int64_t)bucket_idx, num_total_buckets);
                uint8_t new_off, kind;
                if (in_place) {
                    new_off = old_off;
                    kind    = kV3bInPlace;
                } else {
                    int32_t r = atomicAdd(reloc_cnt, 1);
                    new_off = (uint8_t)(local_page_offset_base + (uint32_t)r);
                    kind    = kV3bOldReloc;
                }
                int32_t pos = atomicAdd(aux_count, 1);
                assert(pos < S_CAP);
                V3bAux e;
                e.key = key; e.src_idx = 0; e.new_off = new_off; e.old_off = old_off; e.kind = kind; e._pad = 0;
                aux[pos] = e;
                avk[pos] = avalanche_key((HASH_TYPE)key);
            }
        }
        __syncwarp();
        const int32_t reloc_total = *reloc_cnt;

        // ---- Phase 2: gather NEW keys ----
        const uint32_t new_keys_off_base = local_page_offset_base + (uint32_t)reloc_total;
        for (int32_t i = lane; i < bucket_size; i += 32) {
            int32_t inv = (int32_t)d_inverse_buffer[keys_offs_base + i];
            int64_t key = d_keys[inv];
            int32_t pos = atomicAdd(aux_count, 1);
            assert(pos < S_CAP);
            V3bAux e;
            e.key = key; e.src_idx = inv;
            e.new_off = (uint8_t)(new_keys_off_base + (uint32_t)i);
            e.old_off = 0; e.kind = kV3bNew; e._pad = 0;
            aux[pos] = e;
            avk[pos] = avalanche_key((HASH_TYPE)key);
        }
        __syncwarp();
        const int32_t total = *aux_count;

        // ---- Phase 3: pilot search (only PILOT_SEARCH buckets) ----
        if (mode == BUCKET_MODE_V2G_PILOT_SEARCH) {
            uint32_t iter_idx = (uint32_t)lane + 1;
            int64_t  iter = 0;
            constexpr int64_t max_iterations = (1ll << pilot_bits) / 32;
            constexpr int32_t BITMASK_WORDS  = static_cast<int32_t>(ceil_div<int64_t>(MAX_BUCKET_SIZE, 64));
            uint32_t found;
            if (new_alpha <= 32) {
                found = pilot_search_loop_single<32, uint32_t, HASH_TYPE>(
                    avk, total, new_alpha, iter_idx, iter, 0xffffffffu, max_iterations);
            } else if (new_alpha <= 64) {
                found = pilot_search_loop_single<32, uint64_t, HASH_TYPE>(
                    avk, total, new_alpha, iter_idx, iter, 0xffffffffu, max_iterations);
            } else {
                found = pilot_search_loop_multi<32, BITMASK_WORDS, HASH_TYPE>(
                    avk, total, new_alpha, iter_idx, iter, 0xffffffffu, max_iterations);
            }
            uint32_t pilot = 0;
            if (found) {
                int found_thread = __ffs(found) - 1;
                pilot = __shfl_sync(0xffffffffu, iter_idx, found_thread) - 32;
            }
            __syncwarp();
            if (d_pilot_counters && lane == 0) {
                atomicAdd((unsigned long long*)&d_pilot_counters[2], (unsigned long long)iter);
                if (found) atomicAdd((unsigned long long*)&d_pilot_counters[3], 1ull);
            }
            if (!found) {
                if (lane == 0) {
                    d_bucket_state[work_idx].packed = pk_set_mode(packed, BUCKET_MODE_V2G_PILOT_FAILURE);
                    if (d_failure_counters) {
                        atomicAdd((unsigned long long*)&d_failure_counters[FC_PILOT_FAILURE], 1ull);
                        atomicAdd((unsigned long long*)&d_failure_counters[FC_DROPPED_KEYS],
                                  (unsigned long long)bucket_size);
                    }
                }
                continue;  // warp-uniform — skip write
            }
            new_pde.pilot = pilot;
            coeff_new = expand_pilot(pilot);
            packed = pk_set_mode(packed, BUCKET_MODE_V2G_CLEAR_PTE);
        }

        // ---- Clear NEW PTE range to 0xff, then write ----
        // After pilot search to allow for pilots retry
        if (full_rehash) {
            PTE_TYPE empty;
            empty.page_offset = 0xff;
            if constexpr (std::is_same_v<PTE_TYPE, SPHV3PTE>) empty.fingerprint = 0;
            for (int32_t i = lane; i < new_alpha; i += 32) {
                stcs_store_pte<PTE_TYPE>(&d_ptes[pte_base_new + i], empty);
            }
        }
        __syncwarp();

        // ---- Phase 4: write PTEs + inline value-reloc drain ----
        for (int32_t base = 0; base < total; base += 32) {
            int32_t e = base + lane;
            bool active = e < total;

            if (active) {
                int32_t hash = (int32_t)hash_avalanched(avk[e], coeff_new, new_alpha);
                PTE_TYPE npte = build_pte<PTE_TYPE, HASH_TYPE>((HASH_TYPE)aux[e].key, aux[e].new_off, num_total_buckets);
                stcs_store_pte<PTE_TYPE>(&d_ptes[pte_base_new + hash], npte);
            }

            bool need = active && (aux[e].kind != kV3bInPlace);
            uint32_t bal = __ballot_sync(0xffffffffu, need);
            int32_t cnt = __popc(bal);
            if (cnt > 0) {
                if (*rl_count + cnt > RL_CAP) {
                    v3b_drain_reloc_list<SUB_WARP_DRAIN_SIZE, VALUE_TYPE>(rl, *rl_count, value_size, lane);
                    __syncwarp();
                    if (lane == 0) *rl_count = 0;
                    __syncwarp();
                }
                int32_t prefix   = __popc(bal & ((1u << lane) - 1u));
                int32_t base_idx = *rl_count;
                if (need) {
                    int8_t* dst = d_value_payload + payload_base_new
                        + ((int64_t)new_page_index * PAGE_SIZE + (int64_t)aux[e].new_off) * value_size;
                    int8_t* src;
                    if (aux[e].kind == kV3bNew) {
                        src = const_cast<int8_t*>(d_new_values)
                            + (int64_t)aux[e].src_idx
                              * (NEW_VALUES_ARE_DEFAULT ? (int64_t)0 : value_size);
                    } else {
                        src = d_value_payload + payload_base_old
                            + ((int64_t)old_page_index * PAGE_SIZE + (int64_t)aux[e].old_off) * value_size;
                    }
                    rl[base_idx + prefix] = V3bRelocPair{src, dst};
                }
                __syncwarp();
                if (lane == 0) *rl_count = base_idx + cnt;
                __syncwarp();
            }
        }
        v3b_drain_reloc_list<SUB_WARP_DRAIN_SIZE, VALUE_TYPE>(rl, *rl_count, value_size, lane);
        __syncwarp();

        // ---- State write-back so kernel_v2h_commit_pde publishes new_pde ----
        if (lane == 0) {
            BucketStateV2G upd;
            upd.new_pde = new_pde;
            upd.packed  = packed;
            upd._pad    = 0;
            d_bucket_state[work_idx] = upd;
        }
    }
}

// ============================================================================
// Dispatcher
// ============================================================================

template<typename HASH_TYPE,
         int64_t NUM_PAGES_PER_SHARD, int64_t PAGE_SIZE,
         int64_t MAX_BUCKET_SIZE,
         typename PTE_TYPE>
inline void call_kernel_v3b_fused(
    int64_t num_buckets, int64_t num_total_buckets, int64_t num_shards,
    const int64_t* d_keys, const int8_t* d_values, const int8_t* d_default_value,
    const int64_t* d_active_buckets, const int64_t* d_bucket_sizes,
    const int64_t* d_offsets, const int64_t* d_inverse_buffer,
    const SPHV2PDE* d_pdes, PTE_TYPE* d_ptes, int8_t* d_value_payload,
    int64_t value_size, int64_t payload_shard_stride_bytes,
    BucketStateV2G* d_bucket_state,
    int64_t* d_pilot_counters, int64_t* d_failure_counters,
    cudaStream_t stream)
{
    SPH_CHECK(value_size > 0, "call_kernel_v3b_fused: value_size must be > 0, got ", value_size);
    SPH_CHECK(d_value_payload != nullptr, "call_kernel_v3b_fused: d_value_payload is null");
    SPH_CHECK(d_default_value != nullptr, "call_kernel_v3b_fused: d_default_value is null");

    const int8_t* d_new_values = d_values ? d_values : d_default_value;

    constexpr int32_t WARPS_PER_BLOCK     = 8;
    constexpr int32_t SUB_WARP_DRAIN_SIZE = 8;
    constexpr int32_t RL_CAP              = 32;
    constexpr int32_t S_CAP               = (int32_t)max_keys_per_bucket;
    constexpr int32_t BPB                 = 128;
    dim3 block(WARPS_PER_BLOCK * 32);
    dim3 grid((uint32_t)ceil_div(num_buckets, static_cast<int64_t>(BPB)));

#define SPH_LAUNCH_FUSED(DEFAULTS, VT)                                              \
    kernel_v3b_fused<HASH_TYPE, NUM_PAGES_PER_SHARD, PAGE_SIZE, MAX_BUCKET_SIZE,    \
                     BPB, WARPS_PER_BLOCK, SUB_WARP_DRAIN_SIZE, RL_CAP, S_CAP,      \
                     PTE_TYPE, VT, DEFAULTS>                                        \
        <<<grid, block, 0, stream>>>(                                               \
            num_buckets, num_total_buckets, num_shards,                             \
            d_keys, d_new_values,                                                   \
            d_active_buckets, d_bucket_sizes, d_offsets,                            \
            d_inverse_buffer, d_pdes, d_ptes, d_value_payload, value_size,          \
            payload_shard_stride_bytes,                                             \
            d_bucket_state, d_pilot_counters, d_failure_counters)

    // Vectorized payload-copy element chosen by value_size divisibility.
    if (d_values != nullptr) {
        SPH_DISPATCH_COPY_T(value_size, VALUE_TYPE, { SPH_LAUNCH_FUSED(false, VALUE_TYPE); });
    } else {
        SPH_DISPATCH_COPY_T(value_size, VALUE_TYPE, { SPH_LAUNCH_FUSED(true, VALUE_TYPE); });
    }
#undef SPH_LAUNCH_FUSED

    SPH_CHECK(cudaGetLastError());
}

}  // namespace sph
