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
// sph compile-time geometry constants.
//
// Pure constexpr — no CUDA headers, no device code. Split out of sph_types.h so
// the public sph_hash_map.h can pull in just the geometry it needs (e.g.
// page_index_bits for num_pages_per_shard_) without dragging in the CUDA
// device-code headers. sph_types.h includes this and remains the single home
// for the SPHV3PTE/PDE types, hash helpers, and VMM support classes.
// ============================================================================
#include <cstdint>

namespace sph {

static constexpr int64_t alpha_bucket_size_factor = 4;
static constexpr int64_t bucket_size_bits = 5;
static constexpr int64_t min_bucket_size = 8;
static constexpr int64_t max_alpha_bucket_size = ((1 << bucket_size_bits) - 1) * alpha_bucket_size_factor + min_bucket_size;
static constexpr int64_t pte_offset_bits = 28;
static constexpr int64_t pilot_bits = 12;
static constexpr int64_t shard_bit_bits = 1;
static constexpr int64_t page_index_bits = 64 - pilot_bits - pte_offset_bits - bucket_size_bits - shard_bit_bits;

static constexpr int64_t num_mini_shards = 1ll << shard_bit_bits;
static constexpr int64_t pte_chunk_bytes = 1ll << 21;                          // 2 MiB
static constexpr int64_t pte_entries_per_shard = 1ll << pte_offset_bits;       // 2^28 PTEs
static constexpr int64_t pte_pages_per_shard = pte_entries_per_shard / pte_chunk_bytes;
static_assert(pte_pages_per_shard * pte_chunk_bytes == pte_entries_per_shard,
              "pte_entries_per_shard must be a multiple of pte_chunk_bytes");

// V3: 8-byte PTE with embedded fingerprint.
//
// The fingerprint is key / num_buckets (see pte_fingerprint_of), stored in
// pte_fingerprint_bits bits alongside the 8-bit page_offset. A key spans the
// full 64-bit range, so the quotient only fits when
//   num_buckets >= 2^(64 - pte_fingerprint_bits)
// Below that the fingerprint silently truncates: find()/hash() compare against
// a wrapped value (false hits and misses) and defrag's
// reconstruct_key_from_pte() rebuilds the wrong key. Hence min_num_buckets,
// which ShardedPerfectHashMap enforces on every capacity it accepts.
static constexpr int64_t pte_fingerprint_bits    = 56;
static constexpr int64_t min_num_buckets         = 1ll << (64 - pte_fingerprint_bits);  // 256

static constexpr int64_t max_keys_per_bucket     = 64;
static constexpr int64_t pte_v3_chunk_bytes      = pte_chunk_bytes;            // 2 MiB
static constexpr int64_t pte_v3_page_slots       = pte_chunk_bytes / 8;        // sizeof(SPHV3PTE) = 8
static constexpr int64_t pte_v3_pages_per_shard  = pte_entries_per_shard / pte_v3_page_slots;
static_assert(pte_v3_pages_per_shard * pte_v3_page_slots == pte_entries_per_shard,
              "v3 PTE per-shard layout must be exact");

}  // namespace sph
