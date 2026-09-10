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
// sph::ShardedPerfectHashMap — flattened, productized SPH hash map.
//
// One class, no inheritance. Keeps ONLY the best-performing path: the SPHV3PTE
// fingerprint layout + the fused single-pass insert (kernel_v3b_fused). The
// class consolidates the KEEP-set members from the DynamicMap8b class chain
// (DynamicMap : DynamicMap6 : DynamicMap7 : DynamicMap8 : DynamicMap8b) and
// drops everything the fused-fingerprint path never reads: d_payload_keys,
// the reloc double-buffer, pilot-debug counters, and all SPHV2PTE debug paths.
// ============================================================================
#include <memory>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>       // cudaStream_t / cudaEvent_t appear in the public API
#include "sph/sph_geometry.h"   // compile-time geometry (page_index_bits, num_mini_shards, ...)
#include "sph/error.h"          // SphException hierarchy + InsertResult / SphInsertFailed

// Forward declarations. Full definitions live in sph_types.h, which the .cu
// pulls in; keeping them out of this public header avoids dragging the CUDA
// device-code headers into every consumer.
namespace nve { namespace sph_support { class ResizeableBuffer; } }
template<typename IndexT, int32_t MAX_RUN_SIZE> class SphDeduper;

namespace sph {

class ScratchPool;
class ShardedVmmBuffer;

// ----------------------------------------------------------------------------
// Concurrency contract — enforced by the CALLER, not by this class.
//
// ShardedPerfectHashMap does no locking of its own. The embedder (today: nve's SphTable)
// is responsible for serializing calls according to:
//
//   * modify vs modify  -> MUTUALLY EXCLUSIVE.
//         insert / build / assign / reserve / reset / defrag never overlap each
//         other. The per-call scratch pools (aux_dev_ / aux_host_), the pinned
//         readback buffers and the failure counters are single-instance state;
//         two concurrent modifies on the same map WILL corrupt them.
//   * read vs insert    -> ALLOWED to overlap.
//         find / find_locations may run while an insert is in flight.
//   * read vs defrag    -> MUTUALLY EXCLUSIVE.
//         defrag moves the payload and releases the source shard's VMM, so a
//         concurrent find would read unmapped memory.
//
// Nothing here is synchronized; the exclusion above is the whole contract.
// ----------------------------------------------------------------------------
class ShardedPerfectHashMap
{
public:
    using KeyT  = int64_t;
    using HashT = uint64_t;

    ShardedPerfectHashMap(int64_t initial_capacity, int64_t value_size = 512,
               const int8_t* default_value = nullptr);
    ~ShardedPerfectHashMap();

    ShardedPerfectHashMap(const ShardedPerfectHashMap&)            = delete;
    ShardedPerfectHashMap& operator=(const ShardedPerfectHashMap&) = delete;

    // What insert() does with a batch key that is already resident in the table,
    // or that appears more than once within the same batch.
    //
    // IGNORE / UPDATE both filter the key out of the batch before the insert
    // state machine sizes anything, so the bucket is never asked to hold the key
    // twice. FAIL skips the filter entirely and preserves the pre-filter
    // behaviour: a repeated key makes pilot search unsatisfiable
    // (popcount(mask) == total can never hold), so the bucket reports
    // pilot_failure and every batch key in it is dropped.
    //
    // For duplicates *within* a batch the last occurrence wins under UPDATE.
    enum class OnExisting {
        ON_EXISTING_IGNORE,   // keep the stored value, drop the incoming key
        ON_EXISTING_UPDATE,   // overwrite the stored value in place, drop the incoming key
        ON_EXISTING_FAIL,     // no filtering (legacy behaviour)
    };

    //
    // Throws SphInsertFailed on any dropped key; the exception carries the
    // InsertResult.
    void insert(const KeyT* d_keys, const int8_t* d_values, int64_t num_keys, cudaStream_t stream,
                OnExisting on_existing = OnExisting::ON_EXISTING_IGNORE);
    void build(const KeyT* d_keys, int64_t num_keys, cudaStream_t stream,
               const int8_t* d_values = nullptr);
    void find_locations(const KeyT* d_keys, int64_t num_keys, int64_t* d_locations,
                        cudaStream_t stream);
    // hit_mask is an optional in/out bitmask of roundup(num_keys/64) uint64 words, bit[i] == 1 will skip fetching this key, on return set all found keys coresponding bits to 1, nullptr means all keys are fetched.    
    // value_stride is the byte spacing between output rows; 0 means tightly packed (value_size).
    void find(const int64_t* keys, int64_t num_keys, int8_t* values, cudaStream_t stream,
              uint64_t* hit_mask = nullptr, int64_t value_stride = 0);
    void assign(const int64_t* keys, const int8_t* values, int64_t num_keys, cudaStream_t stream);
    void reset(cudaStream_t stream, int64_t capacity);

    // Pre-commit VMM backing so subsequent inserts skip the per-insert grow.
    void reserve(int64_t num_keys, cudaStream_t stream = 0);

    // Defrag a single mega-shard / all mega-shards (SPHV3PTE path).
    void defrag(int32_t mega_shard, cudaStream_t stream);
    void defrag_all(cudaStream_t stream);

    inline int64_t get_num_buckets()   const { return target_capacity_ / avg_bucket_size_; }
    inline int64_t get_num_shards()    const { return num_shards_; }
    inline int64_t get_global_seed()   const { return global_seed_; }
    // Live keys currently in the table.
    inline int64_t get_size() const { return size_; }
    // The capacity the geometry was sized for. A TARGET, not a hard limit:
    // get_size() is allowed to exceed it, at a rising cost in pilot failures.
    inline int64_t get_target_capacity() const { return target_capacity_; }


    // Defined out-of-line in sph_hash_map.cu: these dereference d_value_payload_
    // / d_bucket_offsets_, whose full types are only visible there.
    //
    // FRESHNESS: both report the table as of the last COMPLETED modify, and
    // neither takes a stream. insert / build / reserve / defrag all synchronize
    // before returning, so calling these after any of them returns is exact.
    // They must NOT overlap a modify in flight — unlike find / find_locations,
    // which the concurrency contract above does allow to overlap an insert.
    // get_num_ptes() is the only accessor that touches the device: it reads
    // device memory with a blocking copy on the default stream, which does not
    // synchronize with a caller's cudaStreamNonBlocking stream, so an
    // overlapping call can read a torn or stale count. That applies equally to
    // get_bits_per_key() below, which is built on it. It is meant for
    // diagnostics, not for a hot path.
    //
    // get_total_payload_slots() — and get_fragmentation() below, which is built
    // on it — read host-side commit bookkeeping only. No device access and no
    // stream involvement, so the default-stream concern above does not apply.
    int64_t get_total_payload_slots() const;
    int64_t get_num_ptes() const;

    inline double get_bits_per_key() const {
        if (size_ == 0) return 0.0;
        int64_t num_ptes = get_num_ptes();
        int64_t num_buckets = get_num_buckets();
        double pte_bits = static_cast<double>(num_ptes) * 8.0;
        double pde_bits = static_cast<double>(num_buckets) * 64.0;
        return (pte_bits + pde_bits) / static_cast<double>(size_);
    }

    inline double get_fragmentation() const {
        if (size_ == 0) return 0.0;
        return static_cast<double>(get_total_payload_slots()) / static_cast<double>(size_);
    }

    const int8_t* get_value_payload_ptr() const;  // out-of-line: derefs d_value_payload_
    inline int64_t get_value_size()        const { return value_size_; }

private:
    void ensure_aux_pools();
    void bucketize(const int64_t* d_indices, int64_t num_indices, int64_t num_buckets,
        int64_t* d_active_buckets, int64_t* d_bucket_sizes,
        int64_t* h_num_active_buckets, int64_t* d_inverse_buffer, int64_t* d_offsets,
        int64_t* d_bucket_indices, int64_t global_seed, cudaStream_t stream);

    // Compile-time geometry.
    constexpr static int64_t target_exponent_    = 4;
    constexpr static int64_t avg_bucket_size_     = 8;
    constexpr static int64_t page_size_           = (1 << 8) - 1;               // 255
    constexpr static int64_t num_pages_per_shard_ = (1 << page_index_bits);

    int64_t value_size_          = 0;
    int64_t target_capacity_             = 0;
    int64_t size_        = 0;
    int64_t num_shards_          = 0;
    int64_t num_buckets_         = 0;
    int64_t global_seed_         = 0;
    size_t  tmp_mem_size_device_ = 0;
    size_t  tmp_mem_size_host_   = 0;
    int64_t* h_num_active_buckets_ = nullptr;

    std::shared_ptr<SphDeduper<KeyT, -1>>   deduper_;
    std::shared_ptr<ShardedVmmBuffer>     d_ptes_;           // SPHV3PTE layout
    std::shared_ptr<ShardedVmmBuffer>     d_value_payload_;
    std::shared_ptr<nve::sph_support::ResizeableBuffer> d_pdes_;
    std::shared_ptr<nve::sph_support::ResizeableBuffer> d_shard_page_counter_;
    std::shared_ptr<nve::sph_support::ResizeableBuffer> d_shard_page_offset_counter_;
    std::shared_ptr<nve::sph_support::ResizeableBuffer> d_shard_page_offset_global_counter_;
    std::shared_ptr<nve::sph_support::ResizeableBuffer> d_shard_locks_;
    std::shared_ptr<nve::sph_support::ResizeableBuffer> d_bucket_offsets_;
    std::shared_ptr<nve::sph_support::ResizeableBuffer> d_global_shard_bit_;
    std::vector<int32_t>                   h_global_shard_bit_;
    std::shared_ptr<nve::sph_support::ResizeableBuffer> deduper_d_tmp_device_mem_;
    std::shared_ptr<nve::sph_support::ResizeableBuffer> deduper_h_tmp_host_mem_;

    std::unique_ptr<ScratchPool> aux_dev_;
    std::unique_ptr<ScratchPool> aux_host_;

    cudaStream_t aux_stream_ = nullptr;
    cudaEvent_t  alloc_done_event_ = nullptr;

    int8_t* d_default_value_ = nullptr;

    int64_t* d_failure_counters_ = nullptr;
    int64_t* h_failure_counters_ = nullptr;
    // Per-insert scratch pointers, wired at the top of insert() and passed by
    // value into run_v3b_phases. Instance state (not static): each map owns its
    // own so concurrent inserts on different tables can't clobber each other.
    int64_t* active_failure_counters_ = nullptr;
    int64_t* active_pilot_counters_   = nullptr;   // stays nullptr on this path
    int32_t* active_global_shard_bit_ = nullptr;
};

}  // namespace sph
