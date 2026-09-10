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

#include <gtest/gtest.h>
#include <default_allocator.hpp>
#include <cuda_runtime_api.h>
#include <cstring>
#include <thread>
#include <vector>

namespace nve {

struct AllocatorTestParams {
    size_t host_alloc_threshold;
    size_t alloc_size;
};

class AllocatorTest : public testing::TestWithParam<AllocatorTestParams> {
protected:
    void SetUp() override {
        const auto& params = GetParam();
        alloc_size_ = params.alloc_size;
        allocator_ = std::make_shared<DefaultAllocator>(params.host_alloc_threshold);
        ASSERT_EQ(cudaSuccess, cudaGetDevice(&current_device_));
    }

    void TearDown() override {
        // Ensure we restore the device if changed during tests
        cudaSetDevice(current_device_);
    }

    allocator_ptr_t allocator_;
    int current_device_;
    size_t alloc_size_;
};

TEST_P(AllocatorTest, DeviceAllocateAndFree) {
    void* ptr = nullptr;

    // Test allocation
    EXPECT_EQ(cudaSuccess, allocator_->device_allocate(&ptr, alloc_size_));
    EXPECT_NE(nullptr, ptr);

    // Test we can write to the allocated memory
    cudaError_t err = cudaMemset(ptr, 0xFF, alloc_size_);
    EXPECT_EQ(cudaSuccess, err);

    // Test deallocation
    EXPECT_EQ(cudaSuccess, allocator_->device_free(ptr));
}

TEST_P(AllocatorTest, HostAllocateAndFree) {
    void* ptr = nullptr;

    // Test allocation
    EXPECT_EQ(cudaSuccess, allocator_->host_allocate(&ptr, alloc_size_));
    EXPECT_NE(nullptr, ptr);

    // Test we can write to the allocated memory
    memset(ptr, 0xFF, alloc_size_);

    // Test deallocation
    EXPECT_EQ(cudaSuccess, allocator_->host_free(ptr));
}

INSTANTIATE_TEST_SUITE_P(
    AllocatorTestSuite,
    AllocatorTest,
    testing::Values(
        AllocatorTestParams{0, 4 * 1024},  // 0 threshold, 4KB allocation
        AllocatorTestParams{2 * 1024 * 1024, 3 * 1024 * 1024},  // 2MB threshold, 3MB allocation (huge page in host allocation)
        AllocatorTestParams{2 * 1024 * 1024, 1024}  // 2MB threshold, 1KB allocation (regular allocation in host)
    )
);

// The probe runs inside the noexcept allocator, so it must never throw - not even when the sysfs
// hugepage directory is absent or the request can't be satisfied.
TEST(DefaultAllocatorProbe, HugepageProbeNeverThrows) {
    size_t bits = 1;
    EXPECT_NO_THROW(bits = get_largest_hugepage_bits(size_t(1) << 62));
    EXPECT_EQ(bits, 0u);  // nobody has that many huge pages
    EXPECT_NO_THROW(get_largest_hugepage_bits(4096));
}

// The default allocator is shared by every context: its bookkeeping must survive concurrent use.
TEST(DefaultAllocatorConcurrency, ParallelAllocateAndFree) {
    constexpr int num_threads = 8;
    constexpr int iterations = 64;
    // Threshold 0 sends every host allocation through the hugepage probe + mmap bookkeeping path.
    auto allocator = std::make_shared<DefaultAllocator>(0);
    std::vector<int> failures(num_threads, 0);
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; t++) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < iterations; i++) {
                void* h = nullptr;
                void* d = nullptr;
                failures[t] += (allocator->host_allocate(&h, 4096) != cudaSuccess) || (h == nullptr);
                failures[t] += (allocator->device_allocate_async(&d, 4096, 0) != cudaSuccess) || (d == nullptr);
                failures[t] += (allocator->host_free(h) != cudaSuccess);
                failures[t] += (allocator->device_free_async(d, 0) != cudaSuccess);
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    for (size_t t = 0; t < num_threads; t++) {
        EXPECT_EQ(failures[t], 0) << "thread " << t;
    }
}

}  // namespace nve
