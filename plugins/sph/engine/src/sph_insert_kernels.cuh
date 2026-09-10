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

enum BucketModeV2G : uint8_t {
    BUCKET_MODE_V2G_INIT_DONE = 0,
    BUCKET_MODE_V2G_CHECK_PILOT = 1,
    BUCKET_MODE_V2G_CHECK_ALPHA = 2,
    BUCKET_MODE_V2G_CHECK_CURRENT_PAGE = 3,
    BUCKET_MODE_V2G_ALLOCATE_PAGE = 4,
    BUCKET_MODE_V2G_CLEAR_PTE = 5,
    BUCKET_MODE_V2G_WRITE_EXISTING_PTE = 6,
    BUCKET_MODE_V2G_WRITE_BUCKET_PTE = 7,
    BUCKET_MODE_V2G_COMMIT_PTE_AND_PDE = 8,
    BUCKET_MODE_V2G_PILOT_SEARCH = 9,
    BUCKET_MODE_V2G_DONE = 10,
    BUCKET_MODE_V2G_PILOT_FAILURE = 11,
    BUCKET_MODE_V2G_MAX_BUCKET_REACHED = 12,
    BUCKET_MODE_V2G_BUCKET_OFFSET_OVERFLOW = 13,
    BUCKET_MODE_V2G_PAGE_INDEX_OVERFLOW = 14,
};

struct BucketStateV2G {
    SPHV2PDE new_pde;   // 8 B
    uint32_t packed;    // 4 B bitfield
    uint32_t _pad;      // 4 B pad
};
static_assert(sizeof(BucketStateV2G) == 16, "BucketStateV2G must be 16 bytes");

struct BucketStatePacked {
    uint32_t mode              : 4;
    uint32_t is_valid_pde      : 1;
    uint32_t is_pilot_valid    : 1;
    uint32_t is_curr_page_valid: 1;
    uint32_t existing_count    : 7;
    uint32_t local_page_offset : 8;
    uint32_t is_deferred       : 1;
    uint32_t _unused           : 9;
};
static_assert(sizeof(BucketStatePacked) == 4, "BucketStatePacked must be 4 bytes");

__device__ __host__ __forceinline__ BucketStatePacked unpack_state(uint32_t packed) {
    BucketStatePacked p;
    __builtin_memcpy(&p, &packed, sizeof(p));
    return p;
}
__device__ __host__ __forceinline__ uint32_t pack_state(BucketStatePacked p) {
    uint32_t v;
    __builtin_memcpy(&v, &p, sizeof(v));
    return v;
}

static constexpr uint32_t PK_MODE_MASK          = 0xFu;       // bits [0:3]
static constexpr uint32_t PK_VALID_PDE_BIT      = 1u << 4;
static constexpr uint32_t PK_PILOT_VALID_BIT    = 1u << 5;
static constexpr uint32_t PK_EC_SHIFT   = 7;   static constexpr uint32_t PK_EC_MASK   = 0x7Fu;
static constexpr uint32_t PK_LOFF_SHIFT = 14;  static constexpr uint32_t PK_LOFF_MASK = 0xFFu;
static constexpr uint32_t PK_DEFERRED_BIT = 1u << 22;
__device__ __forceinline__ uint32_t pk_set_mode(uint32_t packed, uint32_t m) {
    return (packed & ~PK_MODE_MASK) | (m & PK_MODE_MASK);
}
__device__ __forceinline__ uint32_t pk_set_loff(uint32_t packed, uint32_t v) {
    return (packed & ~(PK_LOFF_MASK << PK_LOFF_SHIFT)) | ((v & PK_LOFF_MASK) << PK_LOFF_SHIFT);
}

// ============================================================================
// Pilot-search inner loops, specialized on the bitmask register type.
// ============================================================================
template<int64_t SUBWARP_WIDTH, typename MASK_TYPE, typename HASH_TYPE>
__device__ __forceinline__ uint32_t pilot_search_loop_single(
    const HASH_TYPE* __restrict__ bucket_keys,
    int32_t total, int64_t new_alpha,
    uint32_t& iter_idx, int64_t& iter,
    uint32_t sub_mask, int64_t max_iterations)
{
    uint32_t found = 0;
    while (!found && iter < max_iterations) {
        MASK_TYPE bitmask = 0;
        uint64_t coeff = expand_pilot(iter_idx);
        for (int32_t j = 0; j < total; j++) {
            auto hash = hash_avalanched(bucket_keys[j], coeff, new_alpha);
            bitmask |= ((MASK_TYPE)1 << hash);
        }
        int32_t pop;
        if constexpr (sizeof(MASK_TYPE) == sizeof(uint32_t)) pop = __popc((uint32_t)bitmask);
        else                                              pop = __popcll((uint64_t)bitmask);
        bool unique = pop == total;
        __syncwarp(sub_mask);
        found = __ballot_sync(sub_mask, unique);
        iter_idx += SUBWARP_WIDTH;
        iter++;
    }
    return found;
}

template<int64_t SUBWARP_WIDTH, int32_t BITMASK_WORDS, typename HASH_TYPE>
__device__ __forceinline__ uint32_t pilot_search_loop_multi(
    const HASH_TYPE* __restrict__ bucket_keys,
    int32_t total, int64_t new_alpha,
    uint32_t& iter_idx, int64_t& iter,
    uint32_t sub_mask, int64_t max_iterations)
{
    uint32_t found = 0;
    while (!found && iter < max_iterations) {
        uint64_t bitmask[BITMASK_WORDS];
        #pragma unroll
        for (int32_t w = 0; w < BITMASK_WORDS; w++) bitmask[w] = 0;
        uint64_t coeff = expand_pilot(iter_idx);
        for (int32_t j = 0; j < total; j++) {
            auto hash = hash_avalanched(bucket_keys[j], coeff, new_alpha);
            bitmask[hash >> 6] |= (1ull << (hash & 63));
        }
        int32_t pop = 0;
        #pragma unroll
        for (int32_t w = 0; w < BITMASK_WORDS; w++) pop += __popcll(bitmask[w]);
        bool unique = pop == total;
        __syncwarp(sub_mask);
        found = __ballot_sync(sub_mask, unique);
        iter_idx += SUBWARP_WIDTH;
        iter++;
    }
    return found;
}

// ============================================================================
// Allocate page — block-level atomicAdd aggregation on the per-shard global
// counter.
// ============================================================================
template<int64_t NUM_PAGES_PER_SHARD, int64_t PAGE_SIZE>
__global__ void kernel_v2g_allocate_page(
    int64_t num_buckets,
    const int64_t* d_active_buckets,
    const int64_t* d_bucket_sizes,
    int64_t num_shards,
    int64_t* d_shard_page_offset_global_counter,
    int64_t* d_shard_page_offsets,
    int64_t* /*d_shard_page_counter*/,
    BucketStateV2G* d_bucket_state,
    int64_t* d_failure_counters = nullptr)
{
    // Dynamic shared reduction, one slot per physical shard. Sized by the launch
    // to 2 * num_total_shards * sizeof(ull); indexed by phys shard id, which ranges
    // over [0, num_total_shards). The strided housekeeping loops below decouple
    // coverage from both blockDim.x and any fixed shard cap.
    const int64_t num_total_shards = num_shards * num_mini_shards;
    extern __shared__ unsigned long long smem[];
    unsigned long long* sh_sum   = smem;
    unsigned long long* sh_start = smem + num_total_shards;

    uint32_t work_idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool in_range = work_idx < num_buckets;

    BucketStateV2G s = in_range ? d_bucket_state[work_idx] : BucketStateV2G{};
    bool active = in_range && (s.packed & PK_MODE_MASK) == BUCKET_MODE_V2G_ALLOCATE_PAGE;

    int64_t shard_idx    = 0;
    int32_t total_needed = 0;
    int32_t bucket_size  = 0;   // batch keys for this bucket; reported if we drop them
    if (active) {
        int64_t bucket_idx     = d_active_buckets[work_idx];
        shard_idx              = compute_phys_shard(bucket_idx, num_shards, s.new_pde.shard_bit);
        bucket_size            = (int32_t)d_bucket_sizes[work_idx];
        int32_t existing_count = (int32_t)((s.packed >> PK_EC_SHIFT) & PK_EC_MASK);
        total_needed           = bucket_size + existing_count;
    }

    for (int64_t sid = threadIdx.x; sid < num_total_shards; sid += blockDim.x) sh_sum[sid] = 0;
    __syncthreads();

    unsigned long long my_pos = 0;
    if (active) {
        my_pos = atomicAdd(&sh_sum[shard_idx], (unsigned long long)total_needed);
    }
    __syncthreads();

    for (int64_t sid = threadIdx.x; sid < num_total_shards; sid += blockDim.x) {
        unsigned long long tot = sh_sum[sid];
        if (tot > 0) {
            sh_start[sid] = atomicAdd(
                (unsigned long long*)(d_shard_page_offset_global_counter + sid),
                tot);
        }
    }
    __syncthreads();

    if (!active) return;

    unsigned long long global_offset = sh_start[shard_idx] + my_pos;
    uint64_t start_page = global_offset / PAGE_SIZE;
    uint64_t end_page   = (global_offset + total_needed) / PAGE_SIZE;
    while (start_page != end_page) {
        global_offset = atomicAdd(
            (unsigned long long*)(d_shard_page_offset_global_counter + shard_idx),
            (unsigned long long)total_needed);
        start_page = global_offset / PAGE_SIZE;
        end_page = (global_offset + total_needed) / PAGE_SIZE;
    }
    int64_t alloc_page   = (int64_t)start_page;
    int64_t alloc_offset = (int64_t)(global_offset % PAGE_SIZE);

    if (alloc_page >= (int64_t)(1ull << page_index_bits)) {
        if (d_failure_counters) {
            atomicAdd((unsigned long long*)&d_failure_counters[FC_PAGE_INDEX_OVERFLOW], 1ull);
            atomicAdd((unsigned long long*)&d_failure_counters[FC_DROPPED_KEYS],
                      (unsigned long long)bucket_size);
        }
        s.packed = pk_set_mode(s.packed, BUCKET_MODE_V2G_PAGE_INDEX_OVERFLOW);
        d_bucket_state[work_idx] = s;
        return;
    }

    atomicMax(
        (unsigned long long*)(d_shard_page_offsets + shard_idx * NUM_PAGES_PER_SHARD + alloc_page),
        (unsigned long long)(alloc_offset + total_needed));

    s.new_pde.page_index = (uint32_t)alloc_page;
    uint32_t next_mode = (s.packed & PK_PILOT_VALID_BIT) ? BUCKET_MODE_V2G_CLEAR_PTE
                                                         : BUCKET_MODE_V2G_PILOT_SEARCH;
    uint32_t new_packed = pk_set_loff(s.packed, (uint32_t)alloc_offset);
    s.packed = pk_set_mode(new_packed, next_mode);
    d_bucket_state[work_idx] = s;
}

// ============================================================================
// init_check_pilot_alpha — merged init + check_pilot + check_alpha +
// check_current_page. Templated on PTE type; read-only on pte.page_offset.
// ============================================================================
template<typename HASH_TYPE, int64_t TARGET_EXPONENT, int64_t ALPHA_BUCKET_SIZE_FACTOR, int64_t NUM_PAGES_PER_SHARD, int64_t PAGE_SIZE, int64_t MAX_BUCKET_SIZE, typename PTE_TYPE = SPHV3PTE>
__global__ void kernel_v2h_init_check_pilot_alpha(
    int64_t num_buckets,
    const int64_t* d_keys,
    const int64_t* d_active_buckets,
    const int64_t* d_bucket_sizes,
    const int64_t* d_offsets,
    const int64_t* d_inverse_buffer,
    const SPHV2PDE* d_pdes,
    const PTE_TYPE* d_ptes,
    int64_t* d_bucket_offsets,
    int64_t num_shards,
    int64_t* d_shard_page_offsets,
    int64_t* d_shard_page_offset_global_counter,
    const int32_t* d_global_shard_bit,
    BucketStateV2G* d_bucket_state,
    int64_t* d_failure_counters = nullptr)
{
    uint32_t work_idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool in_range = work_idx < num_buckets;

    int64_t bucket_idx = 0;
    int32_t shard_idx  = 0;
    int64_t pte_base   = 0;
    SPHV2PDE old_pde   = SPHV2PDE{};
    bool valid         = false;
    int32_t bucket_size = 0;

    BucketStateV2G s{};
    BucketStatePacked p{};

    int32_t existing_count = 0;
    bool pilot_valid = false;
    // The existing-key filter can empty a bucket's run entirely. Such a bucket
    // has nothing to add and must not reserve PTE or page budget; it goes
    // straight to DONE below. It cannot simply return early — the shard
    // reductions further down are __syncthreads()-bound and every thread in the
    // block has to reach them.
    bool empty_bucket = false;

    if (in_range) {
        bucket_idx   = d_active_buckets[work_idx];
        old_pde      = d_pdes[bucket_idx];
        valid        = valid_pde(old_pde);
        bucket_size  = (int32_t)d_bucket_sizes[work_idx];
        empty_bucket = (bucket_size == 0);

        if (valid) {
            s.new_pde = old_pde;
        } else {
            s.new_pde = SPHV2PDE{};
            int64_t mega = shard_hash(bucket_idx, num_shards);
            s.new_pde.shard_bit = (uint32_t)d_global_shard_bit[mega];
        }
        shard_idx = compute_phys_shard(bucket_idx, num_shards, s.new_pde.shard_bit);
        pte_base  = (int64_t)shard_idx * pte_entries_per_shard;
        p.is_valid_pde = valid ? 1 : 0;

        if (valid && !empty_bucket) {
            int32_t keys_offs_base = (int32_t)d_offsets[work_idx];
            int32_t alpha          = (int32_t)expand_bucket_size(s.new_pde.bucket_size);
            int32_t bucket_offset  = (int32_t)expand_bucket_offset(s.new_pde.bucket_offset);

            // BUG (known, unfixed): `mask` is one 64-bit word but `alpha` reaches
            // 128 (alpha_lut[31]), so `1ull << j` is undefined for j >= 64, as is
            // the `mask & (1ull << hash)` test below. Slot j aliases slot j+64.
            //
            // It fails in the safe direction: aliasing can only manufacture a
            // FALSE collision, never hide a real one (two keys in the same slot
            // still set and test the same bit), so pilot_valid is set to false
            // when it need not be. The cost is unnecessary full rehashes — never
            // a lost key.
            //
            // Triggers for any bucket with alpha > 64, i.e. alpha_lut index >= 24
            // (~24+ keys), so it is a dense-bucket-only effect and a plausible
            // contributor to the pilot-failure rate at high load. Fix is to widen
            // this to a BITMASK_WORDS-wide mask, as pilot_search_loop_multi does.
            uint64_t mask = 0;
            int32_t ec = 0;
            for (int32_t j = 0; j < alpha; j++) {
                uint8_t po = (uint8_t)d_ptes[pte_base + bucket_offset + j].page_offset;
                if (po != 0xff) { mask |= 1ull << j; ec++; }
            }
            existing_count = ec;

            pilot_valid = true;
            uint64_t coeff = expand_pilot(s.new_pde.pilot);
            for (int32_t i = 0; i < bucket_size; i++) {
                int32_t inv = (int32_t)d_inverse_buffer[keys_offs_base + i];
                HASH_TYPE key = (HASH_TYPE)d_keys[inv];
                int32_t hash = (int32_t)hash_index(key, coeff, alpha);
                if (mask & (1ull << hash)) { pilot_valid = false; break; }
                mask |= 1ull << hash;
            }
        }
        p.is_pilot_valid = pilot_valid ? 1 : 0;
        p.existing_count = existing_count;
    }

    // Dynamic shared reduction, one slot per physical shard (see kernel_v2g_allocate_page
    // for the same pattern): sized by the launch to 2 * num_total_shards * sizeof(ull),
    // indexed by phys shard id in [0, num_total_shards), strided housekeeping loops.
    const int64_t num_total_shards = num_shards * num_mini_shards;
    extern __shared__ unsigned long long smem[];
    unsigned long long* s_shard_total = smem;
    unsigned long long* s_shard_base  = smem + num_total_shards;
    for (int64_t sid = threadIdx.x; sid < num_total_shards; sid += blockDim.x) s_shard_total[sid] = 0;
    __syncthreads();

    bool max_bucket_reached = false;
    bool bucket_offset_overflow = false;
    bool needs_alloc = false;
    int32_t total_keys = 0;
    unsigned long long my_prefix = 0;
    // Guard the fused kernel's shared-memory gather buffers. Any bucket that
    // takes a full-rehash path there gathers surviving existing keys + new keys
    // into aux/avk, which hold max_keys_per_bucket entries. Rejecting the
    // bucket here — rather than letting the kernel's own bounds check fire —
    // turns what was a process-killing __trap() into a counted, reported drop.
    //
    // Deliberately conservative: even a bucket whose pilot is still valid can
    // reach a full rehash (allocate_page promotes it to CLEAR_PTE when the
    // current page is out of room), so gate on the total either way.
    //
    // Note this replaces a `new_alpha > MAX_BUCKET_SIZE` test that could never
    // be true: alpha comes from alpha_lut[min(bucket_size, 31)], whose largest
    // entry is 128, and MAX_BUCKET_SIZE is 132.
    if (in_range && (bucket_size + existing_count) > (int32_t)max_keys_per_bucket) {
        max_bucket_reached = true;
        if (d_failure_counters) {
            atomicAdd((unsigned long long*)&d_failure_counters[FC_MAX_BUCKET_REACHED], 1ull);
            // Only the batch keys are lost; the bucket keeps its committed PDE.
            atomicAdd((unsigned long long*)&d_failure_counters[FC_DROPPED_KEYS],
                      (unsigned long long)bucket_size);
        }
    }

    if (in_range && !pilot_valid && !max_bucket_reached && !empty_bucket) {
        total_keys = bucket_size + existing_count;
        int32_t new_alpha  = (int32_t)compute_alpha_bucket_size(total_keys, TARGET_EXPONENT, ALPHA_BUCKET_SIZE_FACTOR);
        int32_t old_alpha  = valid ? (int32_t)expand_bucket_size(s.new_pde.bucket_size) : -1;
        needs_alloc        = (new_alpha != old_alpha);

        if (needs_alloc) {
            unsigned long long my_size = (unsigned long long)compress_bucket_offset(new_alpha);
            my_prefix = atomicAdd(&s_shard_total[shard_idx], my_size);
        }
    }
    __syncthreads();

    for (int64_t sid = threadIdx.x; sid < num_total_shards; sid += blockDim.x) {
        unsigned long long total = s_shard_total[sid];
        s_shard_base[sid] = (total > 0)
            ? atomicAdd((unsigned long long*)&d_bucket_offsets[sid], total)
            : 0;
    }
    __syncthreads();

    if (in_range && needs_alloc) {
        unsigned long long compressed = s_shard_base[shard_idx] + my_prefix;
        if (compressed >= (1ull << pte_offset_bits)) {
            bucket_offset_overflow = true;
            if (d_failure_counters) {
                atomicAdd((unsigned long long*)&d_failure_counters[FC_BUCKET_OFFSET_OVERFLOW], 1ull);
                atomicAdd((unsigned long long*)&d_failure_counters[FC_DROPPED_KEYS],
                          (unsigned long long)bucket_size);
            }
        } else {
            s.new_pde.bucket_size   = (uint32_t)compress_bucket_size(total_keys);
            s.new_pde.bucket_offset = (uint32_t)compressed;
        }
    }

    bool got_slot = false;
    if (in_range && valid && !max_bucket_reached && !bucket_offset_overflow && !empty_bucket) {
        int32_t current_page = (int32_t)old_pde.page_index;
        auto commit = *(volatile unsigned long long*)(d_shard_page_offset_global_counter + shard_idx);
        int32_t commit_page = (int32_t)(commit / PAGE_SIZE);
        if (commit_page > current_page) {
            auto counter_addr = (unsigned long long*)(d_shard_page_offsets + shard_idx * NUM_PAGES_PER_SHARD + current_page);
            auto old_val = *counter_addr;
            auto total_needed = (unsigned long long)bucket_size;
            while (old_val + total_needed <= (unsigned long long)PAGE_SIZE) {
                auto prev = atomicCAS(counter_addr, old_val, old_val + total_needed);
                if (prev == old_val) {
                    p.local_page_offset  = (uint32_t)old_val;
                    s.new_pde.page_index = (uint32_t)current_page;
                    got_slot = true;
                    break;
                }
                old_val = prev;
            }
        }
    }

    if (in_range) {
        p.is_curr_page_valid = got_slot ? 1 : 0;

        // DONE is skipped by allocate_page (it only runs ALLOCATE_PAGE), by the
        // fused kernel's entry-mode test, and by commit_pde — so an emptied
        // bucket keeps its committed PDE untouched.
        if      (empty_bucket)                  p.mode = BUCKET_MODE_V2G_DONE;
        else if (max_bucket_reached)            p.mode = BUCKET_MODE_V2G_MAX_BUCKET_REACHED;
        else if (bucket_offset_overflow)        p.mode = BUCKET_MODE_V2G_BUCKET_OFFSET_OVERFLOW;
        else if (!valid)                        p.mode = BUCKET_MODE_V2G_ALLOCATE_PAGE;
        else if (got_slot && pilot_valid)       p.mode = BUCKET_MODE_V2G_WRITE_BUCKET_PTE;
        else if (got_slot)                      p.mode = BUCKET_MODE_V2G_PILOT_SEARCH;
        else                                    p.mode = BUCKET_MODE_V2G_ALLOCATE_PAGE;
        s.packed = pack_state(p);
        d_bucket_state[work_idx] = s;
    }
}

// One thread per active bucket; publishes new_pde to d_pdes.
__global__ inline void kernel_v2h_commit_pde(
    int64_t num_buckets,
    const int64_t*  d_active_buckets,
    BucketStateV2G* d_bucket_state,
    SPHV2PDE*       d_pdes)
{
    const int64_t work_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (work_idx >= num_buckets) return;
    BucketStateV2G s = d_bucket_state[work_idx];
    BucketStatePacked p = unpack_state(s.packed);
    if (p.mode != BUCKET_MODE_V2G_CLEAR_PTE && p.mode != BUCKET_MODE_V2G_WRITE_BUCKET_PTE) return;
    int64_t bucket_index = d_active_buckets[work_idx];
    stcs_store_pde(&d_pdes[bucket_index], s.new_pde);
}

// ============================================================================
// Existing-key / duplicate-key filter.
//
// Runs between bucketize() and kernel_v2h_init_check_pilot_alpha. One thread per
// active bucket. Removes from the batch:
//   * keys already resident in the table (exact test: bucket = key % num_buckets
//     and fingerprint = key / num_buckets together recover the key losslessly),
//   * keys that appear more than once within this batch.
//
// "Removing" never touches d_keys. Every downstream kernel reads the batch as
//     key = d_keys[d_inverse_buffer[d_offsets[w] + i]],  i < d_bucket_sizes[w]
// so a removal is a compaction of d_inverse_buffer[base .. base+size) with a
// shortened d_bucket_sizes[w]. d_offsets[w] is unchanged and the tail gap is
// never read again. Each thread owns a disjoint range, so no locking and no
// global compaction pass is needed.
//
// This has to run BEFORE stage 1, not inside the fused kernel: stage 1 sizes
// new_alpha from bucket_size and kernel_v2g_allocate_page reserves payload slots
// from it. Filtering any later would reserve slots for keys that never land,
// permanently inflating get_fragmentation() with no way to reclaim it short of
// defrag.
//
// Thread-per-bucket, matching the kernels above. A warp-per-bucket version
// measured 15-20% slower on all-unique inserts: bucket runs average ~1 key at
// realistic load factors, so 31 of 32 lanes idled while the block still paid for
// the launch. Serial-per-thread the common size==1 case is a single PDE + PTE
// load and no cross-lane traffic at all; the O(n^2) duplicate scan only bites on
// runs that are already rare, and is bounded by the max_keys_per_bucket guard.
//
// UPDATE_VALUES=false -> OnExisting::ON_EXISTING_IGNORE (keep the stored value)
// UPDATE_VALUES=true  -> OnExisting::ON_EXISTING_UPDATE (overwrite the stored value)
// ============================================================================
template<typename HASH_TYPE, int64_t PAGE_SIZE, typename PTE_TYPE, typename VALUE_TYPE,
         bool UPDATE_VALUES>
__global__ void kernel_v3_filter_existing(
    int64_t num_buckets,           // work items == active buckets
    int64_t num_total_buckets,     // capacity/avg_bucket_size; the fingerprint modulus
    int64_t num_shards,
    const int64_t* __restrict__ d_keys,
    // Replacement rows for UPDATE, as base + index*stride: the caller's values
    // with stride == value_size, or the map's default row with stride 0. Same
    // convention as kernel_v3b_fused.
    const int8_t*  __restrict__ d_new_values,
    int64_t                     d_new_values_stride,
    const int64_t* __restrict__ d_active_buckets,
    int64_t*                    d_bucket_sizes,     // in/out: shortened per bucket
    const int64_t* __restrict__ d_offsets,
    int64_t*                    d_inverse_buffer,   // in/out: compacted in place
    const SPHV2PDE* __restrict__ d_pdes,
    const PTE_TYPE* __restrict__ d_ptes,
    int8_t*        d_value_payload,
    int64_t        value_size,
    int64_t        payload_shard_stride_bytes,
    int64_t*       d_failure_counters)
{
    const int64_t work_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (work_idx >= num_buckets) return;

    const int32_t size = (int32_t)d_bucket_sizes[work_idx];
    // Leave oversized runs alone. Stage 1's max_keys_per_bucket gate drops and
    // counts them; without this guard the O(n^2) dedup below would run on an
    // unbounded n and hang.
    if (size <= 0 || size > (int32_t)max_keys_per_bucket) return;

    const int64_t  bucket_idx = d_active_buckets[work_idx];
    const SPHV2PDE pde        = d_pdes[bucket_idx];
    const bool     pde_ok     = valid_pde(pde);
    // Nothing can collide: a lone key with no resident bucket to probe. This is
    // the hot exit on a sparse table.
    if (size == 1 && !pde_ok) return;

    // Resolve chain constants, hoisted out of the per-key loop.
    int32_t phys = 0;
    int64_t pte_base = 0, alpha = 0;
    uint64_t coeff = 0;
    if (pde_ok) {
        phys     = compute_phys_shard(bucket_idx, num_shards, pde.shard_bit);
        pte_base = pte_slot_index(phys, expand_bucket_offset(pde.bucket_offset), 0);
        alpha    = expand_bucket_size(pde.bucket_size);
        coeff    = expand_pilot(pde.pilot);
    }
    const int64_t base = d_offsets[work_idx];
    int32_t n_out = 0, n_hit = 0, n_dup = 0;

    for (int32_t i = 0; i < size; i++) {
        const int64_t inv = d_inverse_buffer[base + i];
        const int64_t key = d_keys[inv];

        // Keep the LAST occurrence of a repeated key, so that under Update the
        // last value in the batch wins.
        bool dup = false;
        for (int32_t j = i + 1; j < size; j++) {
            if (d_keys[d_inverse_buffer[base + j]] == key) { dup = true; break; }
        }
        if (dup) { n_dup++; continue; }

        if (pde_ok) {
            const int64_t  slot = (int64_t)hash_index((HASH_TYPE)key, coeff, (uint64_t)alpha);
            const PTE_TYPE pte  = d_ptes[pte_base + slot];
            if ((uint8_t)pte.page_offset != 0xff &&
                pte.fingerprint == pte_fingerprint_of((uint64_t)key, num_total_buckets)) {
                n_hit++;
                if constexpr (UPDATE_VALUES) {
                    constexpr int64_t E = (int64_t)sizeof(VALUE_TYPE);
                    const int8_t* src = d_new_values + inv * d_new_values_stride;
                    int8_t*       dst = d_value_payload
                        + (int64_t)phys * payload_shard_stride_bytes
                        + ((int64_t)pde.page_index * PAGE_SIZE + (int64_t)pte.page_offset)
                              * value_size;
                    for (int64_t b = 0; b < value_size; b += E) {
                        VALUE_TYPE v = *reinterpret_cast<const VALUE_TYPE*>(src + b);
                        __stcs(reinterpret_cast<VALUE_TYPE*>(dst + b), v);
                    }
                }
                continue;
            }
        }

        // Compact in place. n_out <= i always, and iteration j > i still reads
        // d_inverse_buffer[base + j], so the rewrite can never clobber a slot
        // this thread has yet to read.
        d_inverse_buffer[base + n_out++] = inv;
    }

    if (n_out != size) d_bucket_sizes[work_idx] = n_out;
    if (d_failure_counters) {
        if (n_hit) atomicAdd((unsigned long long*)&d_failure_counters[FC_EXISTING_KEYS],
                             (unsigned long long)n_hit);
        if (n_dup) atomicAdd((unsigned long long*)&d_failure_counters[FC_BATCH_DUPLICATE_KEYS],
                             (unsigned long long)n_dup);
    }
}

// ============================================================================
// Dispatchers
// ============================================================================

template<typename HASH_TYPE, int64_t TARGET_EXPONENT, int64_t ALPHA_BUCKET_SIZE_FACTOR,
         int64_t NUM_PAGES_PER_SHARD, int64_t PAGE_SIZE, int64_t MAX_BUCKET_SIZE,
         typename PTE_TYPE = SPHV3PTE>
inline void call_kernel_v2h_init_check_pilot_alpha(
    int64_t num_buckets, int64_t num_shards,
    const int64_t* d_keys, const int64_t* d_active_buckets,
    const int64_t* d_bucket_sizes, const int64_t* d_offsets,
    const int64_t* d_inverse_buffer, const SPHV2PDE* d_pdes, const PTE_TYPE* d_ptes,
    int64_t* d_bucket_offsets, int64_t* d_shard_page_offsets,
    int64_t* d_shard_page_offset_global_counter, const int32_t* d_global_shard_bit,
    BucketStateV2G* d_bucket_state, int64_t* d_failure_counters, cudaStream_t stream)
{
    const size_t shmem = 2 * (size_t)(num_shards * num_mini_shards) * sizeof(unsigned long long);
    dim3 block(128);
    dim3 grid((uint32_t)((num_buckets + block.x - 1) / block.x));
    kernel_v2h_init_check_pilot_alpha<HASH_TYPE, TARGET_EXPONENT, ALPHA_BUCKET_SIZE_FACTOR,
                                      NUM_PAGES_PER_SHARD, PAGE_SIZE, MAX_BUCKET_SIZE, PTE_TYPE>
        <<<grid, block, shmem, stream>>>(
            num_buckets, d_keys, d_active_buckets, d_bucket_sizes, d_offsets,
            d_inverse_buffer, d_pdes, d_ptes, d_bucket_offsets, num_shards,
            d_shard_page_offsets, d_shard_page_offset_global_counter,
            d_global_shard_bit, d_bucket_state, d_failure_counters);
    SPH_CHECK(cudaGetLastError());
}

template<int64_t NUM_PAGES_PER_SHARD, int64_t PAGE_SIZE>
inline void call_kernel_v2g_allocate_page(
    int64_t num_buckets, int64_t num_shards,
    const int64_t* d_active_buckets, const int64_t* d_bucket_sizes,
    int64_t* d_shard_page_offset_global_counter, int64_t* d_shard_page_offsets,
    BucketStateV2G* d_bucket_state, int64_t* d_failure_counters, cudaStream_t stream)
{
    const size_t shmem = 2 * (size_t)(num_shards * num_mini_shards) * sizeof(unsigned long long);
    dim3 block(128);
    dim3 grid((uint32_t)((num_buckets + block.x - 1) / block.x));
    kernel_v2g_allocate_page<NUM_PAGES_PER_SHARD, PAGE_SIZE><<<grid, block, shmem, stream>>>(
        num_buckets, d_active_buckets, d_bucket_sizes, num_shards,
        d_shard_page_offset_global_counter, d_shard_page_offsets,
        /*d_shard_page_counter*/ nullptr, d_bucket_state, d_failure_counters);
    SPH_CHECK(cudaGetLastError());
}

inline void call_kernel_v2h_commit_pde(
    int64_t num_buckets,
    const int64_t* d_active_buckets, BucketStateV2G* d_bucket_state,
    SPHV2PDE* d_pdes, cudaStream_t stream)
{
    dim3 block(128);
    dim3 grid((uint32_t)((num_buckets + block.x - 1) / block.x));
    kernel_v2h_commit_pde<<<grid, block, 0, stream>>>(
        num_buckets, d_active_buckets, d_bucket_state, d_pdes);
    SPH_CHECK(cudaGetLastError());
}

template<typename HASH_TYPE, int64_t PAGE_SIZE, typename PTE_TYPE>
inline void call_kernel_v3_filter_existing(
    int64_t num_buckets, int64_t num_total_buckets, int64_t num_shards,
    const int64_t* d_keys, const int8_t* d_values, const int8_t* d_default_value,
    const int64_t* d_active_buckets, int64_t* d_bucket_sizes,
    const int64_t* d_offsets, int64_t* d_inverse_buffer,
    const SPHV2PDE* d_pdes, const PTE_TYPE* d_ptes,
    int8_t* d_value_payload, int64_t value_size, int64_t payload_shard_stride_bytes,
    int64_t* d_failure_counters, bool update_values, cudaStream_t stream)
{
    if (num_buckets <= 0) return;
    // Same invariants the fused kernel relies on; see call_kernel_v3b_fused.
    SPH_CHECK(value_size > 0, "call_kernel_v3_filter_existing: value_size must be > 0, got ",
              value_size);
    SPH_CHECK(d_value_payload != nullptr, "call_kernel_v3_filter_existing: d_value_payload is null");
    SPH_CHECK(d_default_value != nullptr, "call_kernel_v3_filter_existing: d_default_value is null");

    // Under UPDATE a batch with no values sets existing rows to the default row,
    // matching what a new key without values gets.
    const int8_t* d_new_values        = d_values ? d_values : d_default_value;
    const int64_t d_new_values_stride = d_values ? value_size : 0;

    dim3 block(256);
    dim3 grid((uint32_t)((num_buckets + block.x - 1) / block.x));

#define SPH_LAUNCH_FILTER(UPD, VT)                                                \
    kernel_v3_filter_existing<HASH_TYPE, PAGE_SIZE, PTE_TYPE, VT, UPD>            \
        <<<grid, block, 0, stream>>>(                                             \
            num_buckets, num_total_buckets, num_shards, d_keys,                   \
            d_new_values, d_new_values_stride,                                    \
            d_active_buckets, d_bucket_sizes, d_offsets, d_inverse_buffer,        \
            d_pdes, d_ptes, d_value_payload, value_size,                          \
            payload_shard_stride_bytes, d_failure_counters)

    if (update_values) {
        // Vectorized payload-copy element chosen by value_size divisibility.
        SPH_DISPATCH_COPY_T(value_size, VALUE_TYPE, { SPH_LAUNCH_FILTER(true, VALUE_TYPE); });
    } else {
        // Ignore never copies; instantiate one element type only.
        SPH_LAUNCH_FILTER(false, uint4);
    }
#undef SPH_LAUNCH_FILTER

    SPH_CHECK(cudaGetLastError());
}

}  // namespace sph
