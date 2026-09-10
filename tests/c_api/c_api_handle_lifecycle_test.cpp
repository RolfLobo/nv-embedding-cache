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

// Handle-lifecycle / invalid-handle-error tests for the C API's table functions

#include <gtest/gtest.h>
#include <nve_c_api.h>

#include <sys/resource.h>

#include "test_utils.hpp"

namespace {

// The two death tests below crash their child process on purpose. Disable core dumps
// for this process before that happens, so the intentional crash doesn't leave a
// core.* file behind. RLIMIT_CORE is inherited across both fork() and execve(), so
// this covers the child regardless of which death-test style (fast/threadsafe) is in
// effect.
void DisableCoreDumps() {
  const struct rlimit no_core{0, 0};
  setrlimit(RLIMIT_CORE, &no_core);
}

// Produces a minimal host table via the in-tree stl-map plugin. Used by the tests
// below that need *some* valid table handle without pulling in CUDA/GPU setup.
nve_table_t MakeStlMapTable(nve_table_factory_t* out_factory) {
  nve_table_factory_t factory = nullptr;
  EXPECT_EQ(NVE_SUCCESS,
            nve_create_table_factory(&factory, "libnve-plugin-stl-map.so", "{}"));
  nve_table_t table = nullptr;
  EXPECT_EQ(NVE_SUCCESS, nve_table_factory_produce(factory, 0, "{}", &table));
  *out_factory = factory;
  return table;
}

}  // namespace

// ---------------------------------------------------------------------------------
// nve_table_find/insert/update/update_accumulate/clear/erase each called with
// table == NULL: confirm NVE_ERROR_INVALID_ARGUMENT per function.
// ---------------------------------------------------------------------------------
TEST(TableHandleLifecycle, NullTableRejectedByFind) {
  int64_t key = 0;
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT,
            nve_table_find(nullptr, nullptr, 1, &key, nullptr, 8, nullptr, nullptr));
}
TEST(TableHandleLifecycle, NullTableRejectedByInsert) {
  int64_t key = 0;
  double value = 0.0;
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT,
            nve_table_insert(nullptr, nullptr, 1, &key, 8, 8, &value));
}
TEST(TableHandleLifecycle, NullTableRejectedByUpdate) {
  int64_t key = 0;
  double value = 0.0;
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT,
            nve_table_update(nullptr, nullptr, 1, &key, 8, 8, &value));
}
TEST(TableHandleLifecycle, NullTableRejectedByUpdateAccumulate) {
  int64_t key = 0;
  float update = 0.0f;
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT,
            nve_table_update_accumulate(nullptr, nullptr, 1, &key, 4, 4, &update,
                                        NVE_DTYPE_FLOAT32));
}
TEST(TableHandleLifecycle, NullTableRejectedByClear) {
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_table_clear(nullptr, nullptr));
}
TEST(TableHandleLifecycle, NullTableRejectedByErase) {
  int64_t key = 0;
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_table_erase(nullptr, nullptr, 1, &key));
}

// ---------------------------------------------------------------------------------
// Double nve_table_destroy on the same handle (use-after-free).
//
// "threadsafe" death-test style: the default "fast" style just fork()s, which is
// unsafe once the process has other threads running (e.g. from earlier GPU/thread-
// pool tests in this binary -- only the calling thread survives into the child, so
// any lock held by another thread at fork time can leave the child deadlocked).
// "threadsafe" instead fork()s and immediately re-execs the test binary to run just
// this one death-test statement in a fresh process image, avoiding that hazard.
// ---------------------------------------------------------------------------------
TEST(TableHandleLifecycle, DoubleDestroyCrashes) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  DisableCoreDumps();
  EXPECT_DEATH(
      {
        nve_table_factory_t factory = nullptr;
        nve_table_t table = MakeStlMapTable(&factory);
        nve_table_destroy(table);
        nve_table_destroy(table);  // second destroy of the same handle: UB
      },
      "");
}

// ---------------------------------------------------------------------------------
// Using a table handle after nve_table_destroy() (dangling pointer). See the
// "threadsafe" note on DoubleDestroyCrashes above.
// ---------------------------------------------------------------------------------
TEST(TableHandleLifecycle, UseAfterDestroyCrashes) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  DisableCoreDumps();
  EXPECT_DEATH(
      {
        nve_table_factory_t factory = nullptr;
        nve_table_t table = MakeStlMapTable(&factory);
        nve_table_destroy(table);
        int64_t key_size = -1;
        nve_table_get_key_size(table, &key_size);  // use after free: UB
      },
      "");
}

// ---------------------------------------------------------------------------------
// nve_table_create_execution_context with table == NULL, or with a valid table but
// out == NULL.
// ---------------------------------------------------------------------------------
TEST(TableHandleLifecycle, CreateExecutionContextNullTable) {
  nve_context_t ctx = nullptr;
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT,
            nve_table_create_execution_context(nullptr, &ctx, nullptr, nullptr, nullptr, nullptr));
}
TEST(TableHandleLifecycle, CreateExecutionContextNullOut) {
  nve_table_factory_t factory = nullptr;
  nve_table_t table = MakeStlMapTable(&factory);
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT,
            nve_table_create_execution_context(table, nullptr, nullptr, nullptr, nullptr, nullptr));
  nve_table_destroy(table);
  nve_table_factory_destroy(factory);
}

// ---------------------------------------------------------------------------------
// nve_table_factory_produce / nve_create_table_factory: *out is reset to NULL before
// any failure path returns, so callers never observe a stale/partial handle after an
// error.
// ---------------------------------------------------------------------------------
TEST(TableHandleLifecycle, CreateTableFactoryResetsOutOnFailure) {
  // A non-null sentinel: if the failure path forgot to clear *out, this pointer
  // value would still be observed after the call.
  nve_table_factory_t sentinel = reinterpret_cast<nve_table_factory_t>(0x1);
  nve_table_factory_t factory = sentinel;
  EXPECT_NE(NVE_SUCCESS, nve_create_table_factory(&factory, "nonexistent_plugin_xyz.so", "{}"));
  EXPECT_EQ(nullptr, factory);
}
TEST(TableHandleLifecycle, TableFactoryProduceResetsOutOnFailure) {
  nve_table_factory_t factory = nullptr;
  ASSERT_EQ(NVE_SUCCESS,
            nve_create_table_factory(&factory, "libnve-plugin-stl-map.so", "{}"));

  nve_table_t sentinel = reinterpret_cast<nve_table_t>(0x1);
  nve_table_t table = sentinel;
  // Malformed JSON: fails during nlohmann::json::parse, well before a table exists.
  EXPECT_NE(NVE_SUCCESS, nve_table_factory_produce(factory, 0, "not valid json", &table));
  EXPECT_EQ(nullptr, table);

  nve_table_factory_destroy(factory);
}
