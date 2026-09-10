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

#include <common.hpp>
#include <cuda_support.hpp>
#include <memory>
#include <allocator.hpp>

namespace nve {

// Internal class for managing scratch buffers
class ResizeableBuffer {
 public:
  ResizeableBuffer(allocator_ptr_t& allocator, bool host_alloc)
      : host_alloc_(host_alloc), buffer_(nullptr), size_(0), allocator_(allocator) {
    NVE_ASSERT_(allocator);
  }
  ~ResizeableBuffer() {
    // Skip the free entirely when nothing was ever allocated: Safer for driverless deployments.
    if (buffer_) {
      const cudaError_t res = free_buffer();
      if (!cuda_runtime_unloading(res, "buffer free")) {
        NVE_CHECK_(res, "buffer free failed");
      }
    }
  }
  // If requested size is larger than existing size, a new buffer will be allocated (old data is lost)
  void* get_ptr(size_t buffer_size) {
    if (buffer_size > size_) {
      // Release the old buffer and forget it before allocating: if the allocation below throws, the
      // destructor must not free the old pointer a second time.
      if (buffer_) {
        NVE_CHECK_(free_buffer(), "buffer free failed");
      }
      void* buffer = nullptr;
      if (host_alloc_) {
        NVE_CHECK_(allocator_->host_allocate(&buffer, buffer_size));
      } else {
        // Remember the device the buffer lives on so the free targets the same device even if the
        // caller's current device has changed by then.
        NVE_CHECK_(cudaGetDevice(&device_id_));
        NVE_CHECK_(allocator_->device_allocate(&buffer, buffer_size, device_id_));
      }
      buffer_ = buffer;
      size_ = buffer_size;
    }
    return buffer_;
  }
  size_t get_size() const { return size_; }
 private:
  // Frees the current buffer and resets the bookkeeping regardless of the outcome, so no later
  // path can free the same pointer again.
  cudaError_t free_buffer() {
    void* const buffer = buffer_;
    buffer_ = nullptr;
    size_ = 0;
    return host_alloc_ ? allocator_->host_free(buffer) : allocator_->device_free(buffer, device_id_);
  }

  bool host_alloc_;
  void* buffer_;
  size_t size_;
  int device_id_{-1}; // device of the current device buffer; -1 = current device / host buffer
  allocator_ptr_t allocator_;
};

}  // namespace nve
