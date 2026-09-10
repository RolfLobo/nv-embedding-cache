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

#include <memory>
#include <type_traits>
#include <cuda_runtime_api.h>

// Owning handles for CUDA resources in tests. ASSERT_* macros return from the enclosing
// test body and NVE_CHECK_ throws, either of which would bypass a manual free, so every
// test allocation should be held by one of these. Tests that assert on the free status
// call the CUDA API directly on .release() instead.
namespace nve {

struct CudaFreeDeleter {
  void operator()(void* ptr) const noexcept {
    if (ptr) cudaFree(ptr);
  }
};

struct CudaFreeHostDeleter {
  void operator()(void* ptr) const noexcept {
    if (ptr) cudaFreeHost(ptr);
  }
};

struct CudaStreamDeleter {
  void operator()(cudaStream_t stream) const noexcept {
    if (stream) cudaStreamDestroy(stream);
  }
};

// cudaMalloc / cudaMallocManaged memory
template <typename T>
using DevicePtr = std::unique_ptr<T, CudaFreeDeleter>;

// cudaMallocHost (pinned) memory
template <typename T>
using HostPtr = std::unique_ptr<T, CudaFreeHostDeleter>;

using StreamPtr = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, CudaStreamDeleter>;

}  // namespace nve
