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

#include <common.hpp>
#include <iostream>
#include <random>

namespace nve {

static thread_local std::string this_thread_name_;

std::string& this_thread_name() { return this_thread_name_; }

std::string Exception::to_string() const {
  std::ostringstream o;

  const char* const what{this->what()};
  o << "Exception '" << what << "' @ " << file_ << ':' << line_;
  const std::string& thread{thread_};
  if (!thread.empty()) {
    o << " in thread: '" << thread << '\'';
  }
  const char* const expr{expr_};
  if (what != expr) {
    o << ", expression: '" << expr << '\'';
  }
  o << '.';

  return o.str();
}

std::string RuntimeError<bool>::to_string() const {
  std::ostringstream o;

  const char* const what{this->what()};
  o << "Runtime error (result=false) '" << what << "' @ " << file() << ':' << line();
  const std::string& thread{thread_name()};
  if (!thread.empty()) {
    o << " in thread: '" << thread << '\'';
  }
  const char* const expr{expression()};
  if (what != expr) {
    o << ", expression: '" << expr << '\'';
  }
  o << '.';

  return o.str();
}

Logger* GetGlobalLogger() {
  static Logger global_logger_;
  return &global_logger_;
}

static thread_local std::random_device rand_dev_;

uint32_t random_seed() { return rand_dev_(); }

}  // namespace nve
