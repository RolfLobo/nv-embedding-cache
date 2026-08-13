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

#include <bit_ops.hpp>

namespace nve {

TEST(bit_ops_test, bitmask64_count) {
  EXPECT_EQ(bitmask64::count(bitmask64::empty), 0);
  EXPECT_EQ(bitmask64::count(bitmask64::full), bitmask64::num_bits);
}

TEST(bit_ops_test, bitmask64_set_get_next) {
  for (int64_t i{}; i < bitmask64::num_bits; ++i) {
    bitmask64_t m{bitmask64::single(i)};

    EXPECT_EQ(bitmask64::count(m), 1);
    for (int64_t j{}; j < bitmask64::num_bits; ++j) {
      EXPECT_EQ(bitmask64::get(m, j), j == i);
    }
    EXPECT_EQ(bitmask64::next(m), i);
  }
}

TEST(bit_ops_test, bitmask64_skip) {
  bitmask64_t m{bitmask64::full};
  for (int64_t i{}; i < bitmask64::num_bits; ++i) {
    EXPECT_TRUE(bitmask64::get(m, i));
    m = bitmask64::skip(m);
    EXPECT_FALSE(bitmask64::get(m, i));
  }
  EXPECT_EQ(m, bitmask64::empty);
}

TEST(bit_ops_test, bitmask64_until) {
  for (int64_t n{}; n < 1024; ++n) {
    bitmask64_t m{bitmask64::until(n)};
    if (n < bitmask64::num_bits) {
      EXPECT_EQ(bitmask64::count(m), n);
    } else {
      EXPECT_EQ(bitmask64::count(m), bitmask64::num_bits);
    }
  }
}

}  // namespace nve
