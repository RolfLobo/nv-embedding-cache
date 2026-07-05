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
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cuda_support.hpp>

#ifndef EC_CHECK
#define EC_CHECK(ans) { ECAssert_((ans), __FILE__, __LINE__); }
template<typename ErrType>
inline void ECAssert_(ErrType code, const char *file, int line, bool abort=true) {
    if (code != ErrType(0)) {
        fprintf(stderr, "EC_CHECK: %d %s %d\n", code, file, line);
        if (abort) exit(code);
    }
}
#endif

// helper class to store and recover device when use multi gpu
// this class assumes single context and always restore to the default context
class ScopedDevice
{
public:
    ScopedDevice(int device_id) : device_id_(device_id), curr_device_(0), swap_device_(false)
    {
        // device_id < 0 is the host-only sentinel — do not touch the CUDA runtime.
        if (device_id_ < 0) {
            return;
        }
        NVE_CHECK_(cudaGetDevice(&curr_device_));
        swap_device_ = curr_device_ != device_id_;
        if (swap_device_) {
            NVE_CHECK_(cudaSetDevice(device_id_));
        }

    }
    ~ScopedDevice()
    {
        if (swap_device_) {
            NVE_CHECK_(cudaSetDevice(curr_device_));
        }
    }
private:
    int device_id_;
    int curr_device_;
    bool swap_device_;
};

// True iff a usable CUDA driver and at least one device are present in this process.
// Code paths that may run without a GPU/driver (e.g. host-only inference) use this to gate CUDA calls.
inline bool driver_available()
{
    static const bool available = []() {
        int device_count = 0;
        const bool ok = (cudaGetDeviceCount(&device_count) == cudaSuccess) && (device_count > 0);
        cudaGetLastError();  // clear the sticky error from a failed probe
        return ok;
    }();
    return available;
}
