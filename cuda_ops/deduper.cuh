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
// Deduper: CUB-based key deduplication (sort -> run-length-encode -> scan).
//
// ============================================================================

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <string>
#include <utility>

#include "include/allocator.hpp"
#include "include/common.hpp"
#include "include/nve_types.hpp"

namespace nve {

template<typename IndexType, int SPLIT_SIZE = 1024>
__global__ void compute_split_counts(const IndexType* __restrict__ key_counts,
                                     IndexType* __restrict__ num_split_chunks,
                                     const uint64_t num_unique_keys)
{
    const int key_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (key_id < num_unique_keys) {
        num_split_chunks[key_id] = (key_counts[key_id] + SPLIT_SIZE - 1) / SPLIT_SIZE;
    }
}

template<typename IndexT, int SPLIT_SIZE = 1024>
void call_compute_split_counts(const IndexT* key_counts, IndexT* num_split_chunks, uint64_t num_counts, cudaStream_t stream)
{
    const uint32_t THREADS_PER_SM = 128;
    dim3 grid_size (static_cast<uint32_t>((num_counts + THREADS_PER_SM - 1) / THREADS_PER_SM), 1);
    dim3 block_size (THREADS_PER_SM, 1);
    compute_split_counts<IndexT, SPLIT_SIZE><<<grid_size, block_size, 0, stream>>>(key_counts, num_split_chunks, num_counts);
    NVE_CHECK_(cudaGetLastError()); // Check kernel launch didn't generate an error
}

template<typename IndexType, int SPLIT_SIZE = 1024>
__global__ void compute_location_mapping(const IndexType* __restrict__ chunk_counts,
                                         const IndexType* __restrict__ chunk_offsets,
                                         const IndexType* __restrict__ key_offsets,
                                         IndexType* __restrict__ output_location_mapping,
                                         IndexType* __restrict__ split_key_offsets,
                                         const uint64_t num_unique_keys)
{
    const int key_id = blockIdx.x * blockDim.y + threadIdx.y;
    if (key_id < num_unique_keys) {
        IndexType count = chunk_counts[key_id];
        auto location_map_p = output_location_mapping + chunk_offsets[key_id];
        auto split_offsets_p = split_key_offsets + chunk_offsets[key_id];
        for (IndexType i = threadIdx.x; i < count; i += warpSize) {
            location_map_p[i] = key_id;
            split_offsets_p[i] = key_offsets[key_id] + i*SPLIT_SIZE;
        }
        if ((key_id + 1) == num_unique_keys) {
            split_offsets_p[count] = key_offsets[key_id + 1];
        }
    }
}

template<typename IndexT, int SPLIT_SIZE = 1024>
void call_compute_location_mapping(const IndexT* chunk_counts,
                                   const IndexT* chunk_offsets,
                                   const IndexT* __restrict__ key_offsets,
                                   IndexT* output_location_mapping,
                                   IndexT* __restrict__ split_key_offsets,
                                   uint64_t num_counts, cudaStream_t stream)
{
    const uint32_t WARPS_PER_SM = 4;
    dim3 grid_size (static_cast<uint32_t>((num_counts + WARPS_PER_SM - 1) / WARPS_PER_SM), 1);
    dim3 block_size (32, WARPS_PER_SM);
    compute_location_mapping<IndexT, SPLIT_SIZE><<<grid_size, block_size, 0, stream>>>(
        chunk_counts, chunk_offsets, key_offsets, output_location_mapping, split_key_offsets, num_counts);
    NVE_CHECK_(cudaGetLastError()); // Check kernel launch didn't generate an error
}

template<typename IndexT, int32_t MAX_RUN_SIZE = -1>
class Deduper
{
public:
    explicit Deduper(nve::allocator_ptr_t allocator) : allocator_(std::move(allocator))
    {
        NVE_CHECK_(allocator_ != nullptr, "Deduper requires a non-null allocator");
        NVE_CHECK_(allocator_->device_allocate(reinterpret_cast<void**>(&d_num_runs_out_), sizeof(IndexT)));
    }

    ~Deduper()
    {
        // No check here: throwing from a destructor calls std::terminate, and
        // there is nothing to recover during teardown.
        allocator_->device_free(d_num_runs_out_);
    }

    // input d_keys, an array of size num_keys with keys to de duplicate
    // return -
    //  1. d_unique_out - device buffer of unique keys,
    //  2. d_counts_out - device buffer of number of duplicates per key
    //  3. d_loc_map_out - device buffer required when maximal run size is limited
    //     for each sub-run this buffer will hold the unique key id it belongs to
    //  4. h_num_runs_out - size of array for d_unique_out (and d_counts_out)
    //  5. d_inverse_buffer - device buffer to map unique keys to their original location
    //  6. d_offsets buffer - device buffer to record the where each inverse of each key starts
    //     offsets can be used to compute key counters, as d_counts_out[i] = d_offsets[i+1] - d_offsets[i]
    //  inverse example: find all locations of d_unique_out[i]:
    //      inverse_locations[d_counts[i]]
    //      for j < d_counts[i]
    //          inverse_locations[j] = d_inverse_buffer[d_offsets[i]+j]
    // NOTE: for MAX_RUN_SIZE == -1 this writes h_num_runs_out[0] AND
    //   h_num_runs_out[1]; callers must size that host buffer for 2 elements.
    // end_bit: cap for cub::DeviceRadixSort. Defaults to full key width but
    //   callers should pass ceil_log2(num_buckets) when sorting bucket
    //   indices — radix sort cost is linear in (end_bit - begin_bit), so
    //   sorting 22 bits of a 64-bit key is ~3x faster than sorting all 64.
    void dedup(const IndexT* d_keys,
        const uint64_t num_keys,
        IndexT* d_unique_out,
        IndexT* d_counts_out,
        IndexT* d_loc_map_out,
        IndexT* h_num_runs_out,
        IndexT* d_inverse_buffer,
        IndexT* d_offsets,
        cudaStream_t stream,
        int end_bit = sizeof(IndexT)*8)
    {
        // Deduping using CUB operands
        // 1. sort enumrate(d_keys)
        // 2. Perform run length encoding on the sorted array
        // 3. perform exclusive sum on the length array to get the offsets for each key inverse mapping

        // aux_buffers are seeded in set_and_init_buffers() caller must recall those when num_keys are growing)
        NVE_CHECK_(num_keys <= init_num_keys_,
                   "dedup called with more keys than set_and_init_buffers was sized for; "
                   "re-run get_alloc_requirements()/set_and_init_buffers() with the larger num_keys");

        NVE_CHECK_(cub::DeviceRadixSort::SortPairs(d_temp_storage_sort_, temp_storage_bytes_sort_,
            d_keys, d_sorted_buffer_, d_location_buffer_, d_inverse_buffer, num_keys, 0, end_bit, stream));

        NVE_CHECK_(cub::DeviceRunLengthEncode::Encode(
            d_temp_storage_encode_, temp_storage_bytes_encode_,
            d_sorted_buffer_, d_unique_out, d_counts_out, d_num_runs_out_, static_cast<int>(num_keys), stream));

        NVE_CHECK_(cudaMemcpyAsync(h_num_runs_out, d_num_runs_out_, sizeof(IndexT), cudaMemcpyDefault, stream));

        // split runs to chunks of limited length, if required
        if (MAX_RUN_SIZE == -1) {
            // Over-scan: scan num_keys items (the upper bound on unique runs)
            // instead of num_unique_out+1, so this scan no longer depends on
            // the host-side unique count and we can drop the sync that used to
            // precede it. d_offsets[k] = sum(d_counts_out[0..k)); for k in
            // [0, num_unique_out] that sums only valid RLE counts, so
            // d_offsets[0..num_unique_out] are exact. Consumers of this (-1)
            // path read only d_offsets[0..num_unique_out); the junk tail beyond
            // (d_counts_out was not zeroed) is never read.
            NVE_CHECK_(cub::DeviceScan::ExclusiveSum(
                d_temp_storage_ex_sum_, temp_storage_bytes_ex_sum_,
                d_counts_out, d_offsets, static_cast<int>(num_keys), stream));
        } else {
            // This path needs num_unique_out on the host to bound the split
            // kernels, so it syncs here (the -1 path syncs only once, below).
            // Because it syncs anyway, the offset scan uses the exact size
            // num_unique_out+1 rather than over-scanning: compute_location_mapping
            // reads d_tmp_offset_buffer_[num_unique_out] for the last run, which
            // an over-scan of num_keys would leave unwritten when every key is
            // unique (num_unique_out == num_keys).
            NVE_CHECK_(cudaStreamSynchronize(stream));
            auto num_unique_out = static_cast<int>(*h_num_runs_out);

            NVE_CHECK_(cub::DeviceScan::ExclusiveSum(
                d_temp_storage_ex_sum_, temp_storage_bytes_ex_sum_,
                d_counts_out, d_tmp_offset_buffer_, num_unique_out + 1, stream));

             // Run split count compute
             call_compute_split_counts<IndexT, MAX_RUN_SIZE>(d_counts_out, d_split_count_buffer_, num_unique_out, stream);

            // Run exclusive prefix sum on split data
            NVE_CHECK_(cub::DeviceScan::ExclusiveSum(
                d_temp_storage_ex_sum_, temp_storage_bytes_ex_sum_,
                d_split_count_buffer_, d_split_offset_buffer_, num_unique_out + 1, stream));

            // The number of chunks is the last element in ExclusiveSum output
            // Return both numbers, because both are needed for backward path
            NVE_CHECK_(cudaMemcpyAsync(h_num_runs_out + 1, d_split_offset_buffer_ + num_unique_out, sizeof(IndexT), cudaMemcpyDefault, stream));

            // Compute final input/output offsets compute
            call_compute_location_mapping<IndexT, MAX_RUN_SIZE>(
                d_split_count_buffer_,
                d_split_offset_buffer_,
                d_tmp_offset_buffer_,
                d_loc_map_out,
                d_offsets,
                num_unique_out,
                stream);
        }
        NVE_CHECK_(cudaStreamSynchronize(stream));
        if (MAX_RUN_SIZE == -1) {
            // h_num_runs_out[0] is host-visible only after the sync above.
            h_num_runs_out[1] = h_num_runs_out[0];
        }
    }

    IndexT* get_sorted() const
    {
        return d_sorted_buffer_;
    }

    void get_alloc_requirements(uint64_t num_keys, size_t& tmp_mem_size_device, size_t& tmp_mem_size_host) {
        tmp_mem_size_host = 0;
        tmp_mem_size_device = 0;

        IndexT* p = nullptr;
        {
            // determing and allocate temporary storage for sort
            NVE_CHECK_(cub::DeviceRadixSort::SortPairs(p, temp_storage_bytes_sort_,
                    p, p, p, p, num_keys, 0, sizeof(IndexT)*8));
            // align everyrhing to 512B
            temp_storage_bytes_sort_ = ((temp_storage_bytes_sort_ + 511) >> 9) << 9;
            tmp_mem_size_device += temp_storage_bytes_sort_;
        }
        {
            // determing and allocate temporary storage for encode
            NVE_CHECK_(cub::DeviceRunLengthEncode::Encode(
                    p, temp_storage_bytes_encode_,
                    p, p, p, p, static_cast<int>(num_keys)));

            temp_storage_bytes_encode_ = ((temp_storage_bytes_encode_ + 511) >> 9) << 9;
            tmp_mem_size_device += temp_storage_bytes_encode_;
        }
        {
            // determing and allocate temporary storage for ex sum
            // ExclusiveSum will be called on one extra input (0)
            // in order to get last offset equal to number of input keys
            int num_keys_ = static_cast<int>(num_keys) + 1;

            NVE_CHECK_(cub::DeviceScan::ExclusiveSum(
                    p, temp_storage_bytes_ex_sum_,
                    p, p, num_keys_));
            temp_storage_bytes_ex_sum_ = ((temp_storage_bytes_ex_sum_ + 511) >> 9) << 9;
            tmp_mem_size_device += temp_storage_bytes_ex_sum_;
        }
        size_t index_buffer_size = sizeof(IndexT) * (num_keys + 1);
        index_buffer_size = ((index_buffer_size + 1023) >> 10) << 10;

        tmp_mem_size_device += index_buffer_size; //d_sorted_buffer_
        tmp_mem_size_device += index_buffer_size; //d_location_buffer_

        if (MAX_RUN_SIZE != -1) {
            tmp_mem_size_device += index_buffer_size; // d_split_count_buffer_
            tmp_mem_size_device += index_buffer_size; //d_tmp_offset_buffer_
            tmp_mem_size_device += index_buffer_size; //d_split_offset_buffer_
        }
        tmp_mem_size_host += index_buffer_size; //h_location_buffer_
    }

    void set_and_init_buffers(uint64_t num_keys, char* tmp_mem_device, char* tmp_mem_host) {
        init_num_keys_ = num_keys;
        char* curr_device_ptr = tmp_mem_device;
        size_t index_buffer_size = sizeof(IndexT) * (num_keys + 1);
        index_buffer_size = ((index_buffer_size + 511) >> 9) << 9;

        d_temp_storage_sort_ = curr_device_ptr;
        curr_device_ptr += temp_storage_bytes_sort_;
        d_temp_storage_encode_ = curr_device_ptr;
        curr_device_ptr += temp_storage_bytes_encode_;
        d_temp_storage_ex_sum_ = curr_device_ptr;
        curr_device_ptr += temp_storage_bytes_ex_sum_;
        d_sorted_buffer_ = reinterpret_cast<IndexT*>(curr_device_ptr);
        curr_device_ptr += index_buffer_size;
        d_location_buffer_ = reinterpret_cast<IndexT*>(curr_device_ptr);
        curr_device_ptr += index_buffer_size;

        if (MAX_RUN_SIZE != -1) {
            d_split_count_buffer_ = reinterpret_cast<IndexT*>(curr_device_ptr);
            curr_device_ptr += index_buffer_size;
            d_tmp_offset_buffer_ = reinterpret_cast<IndexT*>(curr_device_ptr);
            curr_device_ptr += index_buffer_size;
            d_split_offset_buffer_ = reinterpret_cast<IndexT*>(curr_device_ptr);
            curr_device_ptr += index_buffer_size;
        }

        h_location_buffer_ = reinterpret_cast<IndexT*>(tmp_mem_host);
        for (IndexT i = 0; i < static_cast<IndexT>(num_keys); i++) {
            h_location_buffer_[i] = i;
        }
        // Seed the device location buffer once. It is a read-only values-in to
        // SortPairs in dedup() and is never mutated, so this replaces the
        // per-call HtoD copy that dedup() used to do on every invocation.
        NVE_CHECK_(cudaMemcpy(d_location_buffer_, h_location_buffer_,
                              sizeof(IndexT) * num_keys, cudaMemcpyDefault));
    }

private:
    nve::allocator_ptr_t allocator_;

    size_t   temp_storage_bytes_sort_ = 0;
    size_t   temp_storage_bytes_encode_ = 0;
    size_t   temp_storage_bytes_ex_sum_ = 0;

    void     *d_temp_storage_sort_ = nullptr;
    void     *d_temp_storage_encode_ = nullptr;
    void     *d_temp_storage_ex_sum_ = nullptr;

    uint64_t init_num_keys_ = 0; // num_keys the buffers were sized/seeded for in set_and_init_buffers

    IndexT   *d_location_buffer_ = nullptr; // [1, 2, ..., ] helper buffer to build the reverse map
    IndexT   *h_location_buffer_ = nullptr;
    IndexT   *d_sorted_buffer_ = nullptr; // holds the sorted output
    IndexT   *d_num_runs_out_ = nullptr;    // e.g., [ ]
    IndexT   *d_split_count_buffer_ = nullptr;
    IndexT   *d_tmp_offset_buffer_ = nullptr;
    IndexT   *d_split_offset_buffer_ = nullptr;
};

}  // namespace nve
