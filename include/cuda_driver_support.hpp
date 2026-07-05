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

// CUDA Driver API (cu*) error machinery. Pulled out of cuda_support.hpp so that only the
// TUs that actually NVE_CHECK_ a driver-API call carry the cuGetErrorName/String symbols
// (and the <cuda.h> include). Everything else includes cuda_support.hpp and stays free of
// any latent libcuda dependency — that is what lets the driverless build (NVE_DRIVERLESS_BUILD
// ON) produce a binary with no NEEDED libcuda.so.1.

#pragma once

#include <cuda.h>
#include <common.hpp>
#include <cuda_support.hpp>

namespace nve {

template <>
constexpr bool is_success(const CUresult& result) noexcept {
  return result == CUDA_SUCCESS;
}

/**
 * Thrown if a CUDA driver API call fails. Don't use this directly. Use the `NVE_THROW_` and
 * `NVE_CHECK_` macros instead.
 */
template <>
class RuntimeError<CUresult> : public Exception {
 public:
  using base_type = Exception;

  RuntimeError() = delete;

  inline RuntimeError(const char file[], const int line, const char expr[], const CUresult& result,
                      const std::string& hint) noexcept
      : base_type(file, line, expr, hint), result_{result} {}

  inline RuntimeError(const RuntimeError& that) noexcept : base_type(that), result_{that.result_} {}

  inline RuntimeError& operator=(const RuntimeError& that) noexcept {
    base_type::operator=(that);
    result_ = that.result_;
    return *this;
  }

  inline CUresult result() const noexcept { return result_; }

  inline const char* errorName() const noexcept {
    const char* name;
    if (cuGetErrorName(result_, &name) != CUDA_SUCCESS) {
      name = "Call to `cuGetErrorName` failed!";
    }
    return name;
  }

  inline const char* errorString() const noexcept {
    const char* str;
    if (cuGetErrorString(result_, &str) != CUDA_SUCCESS) {
      str = "Call to `cuGetErrorString` failed!";
    }
    return str;
  }

  virtual const char* what() const noexcept override {
    return hint().empty() ? errorString() : hint().c_str();
  }

  virtual std::string to_string() const override {
    std::ostringstream o;

    const char* const what{this->what()};
    o << "CUDA driver error " << errorName() << '[' << result() << "] = '" << what << "' @ " << file()
      << ':' << line();
    const std::string& thread{thread_name()};
    if (!thread.empty()) {
      o << " in thread: '" << thread << '\'';
    }
    const char* const expr{expression()};
    if (what != expr) {
      o << "', expression: '" << expr << '\'';
    }
    const char* const estr{errorString()};
    if (what != estr) {
      o << "', description: '" << estr << '\'';
    }
    o << '.';

    return o.str();
  }

 private:
  CUresult result_;
};

}  // namespace nve
