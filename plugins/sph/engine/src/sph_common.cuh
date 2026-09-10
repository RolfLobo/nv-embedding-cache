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

#include "sph_types.h"

namespace sph {

// Cooperation width (lanes per value) for the subwarp copy kernels.
// Power-of-two, clamped to [2, 32].
__host__ __device__ inline uint32_t sph_copy_subwarp_width(int64_t elems_per_value) {
    uint32_t n = (elems_per_value <= 1) ? 1u : (uint32_t)elems_per_value;
    n--; n |= n >> 1; n |= n >> 2; n |= n >> 4; n |= n >> 8; n |= n >> 16; n++;
    if (n < 2)  n = 2;
    if (n > 32) n = 32;
    return n;
}

__host__ __device__ __forceinline__ int64_t alpha_lut(int64_t clamped_k) {
    constexpr uint8_t kAlphaLut[32] = {
          8,   8,   8,   8,   8,   8,   8,   9,
         11,  14,  16,  19,  21,  24,  27,  31,
         34,  37,  41,  45,  49,  53,  57,  61,
         66,  70,  75,  80,  84,  90,  95, 128,
    };
    return static_cast<int64_t>(kAlphaLut[clamped_k]);
}

__host__ __device__ inline int64_t expand_bucket_size(int64_t compressed) {
    return alpha_lut(compressed);
}

__host__ __device__ inline int64_t compress_bucket_size(int64_t k) {
    constexpr int64_t kClamp = (1 << bucket_size_bits) - 1;  // 31
    return k < kClamp ? k : kClamp;
}

__host__ __device__ inline uint64_t compute_alpha_bucket_size(
    int64_t bucket_size,
    [[maybe_unused]] int64_t target_exponent,
    [[maybe_unused]] int64_t abf = alpha_bucket_size_factor,
    [[maybe_unused]] float min_alpha = 1.25f)
{
    constexpr int64_t kEntries = 1 << bucket_size_bits;  // 32
    int64_t k = bucket_size < kEntries ? bucket_size : kEntries - 1;
    return static_cast<uint64_t>(alpha_lut(k));
}

__host__ __device__ inline int64_t expand_bucket_offset(int64_t compressed) {
    return compressed;
}

__host__ __device__ inline int64_t compress_bucket_offset(int64_t expanded) {
    return expanded;
}

__host__ __device__ __forceinline__ uint64_t pte_fingerprint_of(uint64_t key, int64_t num_buckets) {
    return key / (uint64_t)num_buckets;
}

__host__ __device__ __forceinline__ int64_t reconstruct_key_from_pte(
    const SPHV3PTE& pte, int64_t bucket_index, int64_t num_buckets)
{
    return (int64_t)((uint64_t)pte.fingerprint * (uint64_t)num_buckets
                   + (uint64_t)bucket_index);
}

template<typename PTE_TYPE>
__device__ __forceinline__ void stcs_store_pte(PTE_TYPE* dst, const PTE_TYPE& val) {
    if constexpr (sizeof(PTE_TYPE) == 8) {
        unsigned long long u; __builtin_memcpy(&u, &val, sizeof(u));
        __stcs(reinterpret_cast<unsigned long long*>(dst), u);
    } else {
        __stcs(reinterpret_cast<char*>(dst),
               *reinterpret_cast<const char*>(&val));
    }
}

__device__ __forceinline__ void stcs_store_pde(SPHV2PDE* dst, const SPHV2PDE& val) {
    unsigned long long u; __builtin_memcpy(&u, &val, sizeof(u));
    __stcs(reinterpret_cast<unsigned long long*>(dst), u);
}

template<typename PTE_TYPE, typename HASH_TYPE>
__host__ __device__ __forceinline__ PTE_TYPE build_pte(
    HASH_TYPE key, uint8_t page_offset, int64_t num_buckets)
{
    if constexpr (std::is_same_v<PTE_TYPE, SPHV3PTE>) {
        SPHV3PTE p;
        p.page_offset = page_offset;
        p.fingerprint = pte_fingerprint_of((uint64_t)key, num_buckets);
        return p;
    } else {
        PTE_TYPE p;
        p.page_offset = page_offset;
        return p;
    }
}

__host__ __device__ inline bool valid_pde(const SPHV2PDE& pde) {
    return pde.pilot != 0;
}

// Universal "High 64-bit" multiplier
__host__ __device__ inline uint64_t mul_hi(uint64_t x, uint64_t y) {
    #ifdef __CUDA_ARCH__
        return __umul64hi(x, y);
    #else
        #if defined(__SIZEOF_INT128__)
            return (uint64_t)(((unsigned __int128)x * y) >> 64);
        #else
            uint64_t a_lo = (uint32_t)x, a_hi = x >> 32;
            uint64_t b_lo = (uint32_t)y, b_hi = y >> 32;
            uint64_t mid = a_lo * b_hi + (a_hi * b_lo);
            return a_hi * b_hi + (mid >> 32);
        #endif
    #endif
}

template<typename HASH_TYPE>
__device__ __host__ __forceinline__ HASH_TYPE avalanche_key(HASH_TYPE index)
{
    HASH_TYPE x = index;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    return x;
}

template<typename HASH_TYPE>
__device__ __host__ __forceinline__ HASH_TYPE hash_avalanched(HASH_TYPE avalanched, uint64_t random_coeff, uint64_t prime_modulo)
{
    return mul_hi(avalanched * random_coeff, prime_modulo);
}

template<typename HASH_TYPE>
__device__ __host__ HASH_TYPE hash_index(HASH_TYPE index, uint64_t random_coeff, uint64_t prime_modulo)
{
    return hash_avalanched(avalanche_key(index), random_coeff, prime_modulo);
}

__device__ __host__ __forceinline__ uint64_t expand_pilot(uint32_t pilot) {
    uint64_t z = pilot + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

template<typename HASH_TYPE>
__device__ __host__ __forceinline__ int64_t bucket_hash(HASH_TYPE key, uint64_t global_seed, int64_t num_buckets)
{
    return (int64_t)((global_seed * (uint64_t)key) % num_buckets);
}

__device__ __host__ __forceinline__ int64_t shard_hash(int64_t bucket_idx, int64_t num_shards)
{
    return bucket_idx % num_shards;
}

__device__ __host__ __forceinline__ int32_t compute_phys_shard(
    int64_t bucket_idx, int64_t num_shards, uint64_t shard_bit)
{
    return (int32_t)(shard_hash(bucket_idx, num_shards) * num_mini_shards
                   + (int64_t)shard_bit);
}

__device__ __host__ __forceinline__ int64_t pte_slot_index(
    int32_t phys_shard, int64_t bucket_offset, int64_t slot)
{
    return (int64_t)phys_shard * pte_entries_per_shard + bucket_offset + slot;
}

template <typename PTE_TYPE>
__device__ __forceinline__ int64_t load_existing_key(
    const PTE_TYPE& pte, int64_t bucket_idx, int64_t num_buckets,
    const int64_t* d_payload_keys, int64_t key_location)
{
    if constexpr (std::is_same_v<PTE_TYPE, SPHV3PTE>) {
        (void)d_payload_keys; (void)key_location;
        return reconstruct_key_from_pte(pte, bucket_idx, num_buckets);
    } else {
        (void)pte; (void)bucket_idx; (void)num_buckets;
        return d_payload_keys[key_location];
    }
}

}  // namespace sph
