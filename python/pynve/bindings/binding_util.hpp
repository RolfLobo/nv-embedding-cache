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

#include "third_party/pybind11/include/pybind11/pybind11.h"

#include <algorithm>
#include <cstddef>
#include <cstring>

namespace nve {

// Python may retain the memoryview (or a slice of it) after an override
// returns, so a view directly over the caller's storage could dangle.
// Stage through a Python-owned bytearray and copy changes back while the
// destination is still alive. Both copies are unconditional: the buffers
// are small and init-only, so pure-input/output uses aren't special-cased.
class StagedWritableMemoryView {
 public:
  StagedWritableMemoryView(void* destination, size_t size)
      : destination_(destination),
        size_(size),
        storage_(reinterpret_cast<const char*>(destination), size),
        view_(pybind11::reinterpret_steal<pybind11::memoryview>(
            PyMemoryView_FromObject(storage_.ptr()))) {
    if (!view_) {
      throw pybind11::error_already_set();
    }
  }

  ~StagedWritableMemoryView() {
    // An override can release the view and resize view.obj.
    // Never read beyond the resized Python allocation during cleanup.
    const size_t copy_size = std::min(size_, storage_.size());
    if (copy_size != 0) {
      std::memcpy(destination_, PyByteArray_AS_STRING(storage_.ptr()), copy_size);
    }
  }

  const pybind11::memoryview& view() const { return view_; }

 private:
  void* destination_;
  size_t size_;
  pybind11::bytearray storage_;
  pybind11::memoryview view_;
};

}  // namespace nve
