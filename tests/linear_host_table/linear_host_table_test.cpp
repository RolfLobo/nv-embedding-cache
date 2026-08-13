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
#include <buffer_wrapper.hpp>
#include <common.hpp>
#include <cstring>
#include <linear_host_table.hpp>
#include <default_allocator.hpp>
#include <vector>

namespace nve {

#define TEST_KEY (1337)
#define TEST_ROW_SIZE (128)

struct LinearHostTableTestParams {
    int64_t row_size_bytes;
    int64_t num_rows;
    DataType_t value_dtype;
};

template <typename KeyType>
class LinearHostTableTest : public testing::TestWithParam<LinearHostTableTestParams> {
public:
    using TableType = LinearHostTable<KeyType>;
    using DataType = float;

    LinearHostTableTest() : h_table_(nullptr) {
        Init();
    }

    ~LinearHostTableTest() {
        if (h_table_) {
            NVE_CHECK_(cudaFreeHost(h_table_));
        }
    }

    void test_find() {
        const auto& params = GetParam();
        const size_t data_elements = static_cast<size_t>(params.row_size_bytes) / sizeof(DataType);
        std::vector<DataType> h_output(data_elements, static_cast<DataType>(0.0f));
        KeyType h_keys[] = {TEST_KEY};
        find(h_keys, 1, h_output.data(), params.row_size_bytes);

        // validate results
        for (size_t i = 0; i < data_elements; i++) {
            EXPECT_FLOAT_EQ(h_table_[static_cast<size_t>(TEST_KEY) * data_elements + i], h_output[i]);
        }
    }

    void test_update() {
        const auto& params = GetParam();
        const size_t data_elements = static_cast<size_t>(params.row_size_bytes) / sizeof(DataType);
        std::vector<DataType> h_data(data_elements);
        for (size_t i = 0; i < data_elements; i++) {
            h_data[i] = static_cast<DataType>(-i);
        }
        update(TEST_KEY, h_data.data(), params.row_size_bytes);

        std::vector<DataType> h_output(data_elements, static_cast<DataType>(0.0f));
        KeyType h_keys[] = {TEST_KEY};
        find(h_keys, 1, h_output.data(), params.row_size_bytes);

        // validate results
        for (size_t i = 0; i < data_elements; i++) {
            EXPECT_FLOAT_EQ(h_data[i], h_output[i]);
        }
    }

    void test_update_accumulate() {
        const auto& params = GetParam();
        const size_t data_elements = static_cast<size_t>(params.row_size_bytes) / sizeof(DataType);
        std::vector<DataType> h_data(data_elements);
        for (size_t i = 0; i < data_elements; i++) {
            h_data[i] = static_cast<DataType>(i);
        }
        // make a copy of original data
        std::vector<DataType> h_data_copy(h_table_ + TEST_KEY * data_elements, h_table_ + (TEST_KEY + 1) *  data_elements);
        update_accumulate(TEST_KEY, h_data.data(), params.row_size_bytes, params.value_dtype);

        std::vector<DataType> h_output(data_elements, static_cast<DataType>(0.0f));
        KeyType h_keys[] = {TEST_KEY};
        find(h_keys, 1, h_output.data(), params.row_size_bytes);

        for (size_t i = 0; i < data_elements; i++) {
            EXPECT_FLOAT_EQ(h_data_copy[i] + h_data[i], h_output[i]); // Original value + accumulated value
        }
    }

    void test_find_counts_only_new_hits() {
        const auto& params = GetParam();
        constexpr int64_t num_keys = 70;  // Exercises one full and one partial hitmask word.
        std::vector<KeyType> keys(static_cast<size_t>(num_keys));
        for (int64_t i = 0; i < num_keys; ++i) {
            keys[static_cast<size_t>(i)] = static_cast<KeyType>(i);
        }

        std::vector<DataType> output(
            static_cast<size_t>(num_keys) * static_cast<size_t>(params.row_size_bytes) / sizeof(DataType),
            0.0f);
        std::vector<bitmask64_t> hit_mask(to_uint(ceil_div(num_keys, bitmask64::num_bits)), 0);
        constexpr int64_t existing_hit_indices[] = {0, 2, 63, 69};
        for (const int64_t index : existing_hit_indices) {
            hit_mask[static_cast<size_t>(index / bitmask64::num_bits)] |=
                bitmask64::single(index % bitmask64::num_bits);
        }
        // Padding bits outside num_keys must not be counted as existing hits.
        hit_mask.back() |= bitmask64::single(bitmask64::num_bits - 1);

        auto keys_bw = std::make_shared<BufferWrapper<const void>>(
            ctx_, "keys", keys.data(), keys.size() * sizeof(KeyType));
        auto hit_mask_bw = std::make_shared<BufferWrapper<bitmask64_t>>(
            ctx_, "hit_mask", hit_mask.data(), hit_mask.size() * sizeof(bitmask64_t));
        auto values_bw = std::make_shared<BufferWrapper<void>>(
            ctx_, "values", output.data(),
            static_cast<size_t>(num_keys * params.row_size_bytes));

        tb_->reset_lookup_counter(ctx_);
        tb_->find(ctx_, num_keys, std::move(keys_bw), std::move(hit_mask_bw),
                  params.row_size_bytes, std::move(values_bw), nullptr);

        int64_t hits = 0;
        tb_->get_lookup_counter(ctx_, &hits);
        EXPECT_EQ(num_keys - static_cast<int64_t>(std::size(existing_hit_indices)), hits);
    }

    void test_find_reports_out_of_range_keys_as_misses() {
        const auto& params = GetParam();
        const size_t row_elements = static_cast<size_t>(params.row_size_bytes) / sizeof(DataType);
        const std::vector<KeyType> keys{
            static_cast<KeyType>(-1), static_cast<KeyType>(0),
            static_cast<KeyType>(params.num_rows - 1), static_cast<KeyType>(params.num_rows),
            static_cast<KeyType>(params.num_rows + 1)};
        const int64_t num_keys = static_cast<int64_t>(keys.size());
        constexpr DataType untouched = -123.0f;
        std::vector<DataType> output(keys.size() * row_elements, untouched);
        std::vector<bitmask64_t> hit_mask(to_uint(ceil_div(num_keys, bitmask64::num_bits)), 0);

        auto keys_bw = std::make_shared<BufferWrapper<const void>>(
            ctx_, "keys", keys.data(), keys.size() * sizeof(KeyType));
        auto hit_mask_bw = std::make_shared<BufferWrapper<bitmask64_t>>(
            ctx_, "hit_mask", hit_mask.data(), hit_mask.size() * sizeof(bitmask64_t));
        auto values_bw = std::make_shared<BufferWrapper<void>>(
            ctx_, "values", output.data(), output.size() * sizeof(DataType));

        tb_->reset_lookup_counter(ctx_);
        tb_->find(ctx_, num_keys, std::move(keys_bw), std::move(hit_mask_bw),
                  params.row_size_bytes, std::move(values_bw), nullptr);

        EXPECT_EQ(hit_mask[0], bitmask64::single(1) | bitmask64::single(2));
        for (size_t key_index = 0; key_index < keys.size(); ++key_index) {
            for (size_t element = 0; element < row_elements; ++element) {
                const DataType expected = key_index == 1 || key_index == 2
                                              ? h_table_[static_cast<size_t>(keys[key_index]) *
                                                             row_elements +
                                                         element]
                                              : untouched;
                EXPECT_FLOAT_EQ(output[key_index * row_elements + element], expected);
            }
        }

        int64_t hits = 0;
        tb_->get_lookup_counter(ctx_, &hits);
        EXPECT_EQ(hits, 2);
        EXPECT_EQ(tb_->size(ctx_, true), params.num_rows);

        auto no_mask_keys_bw = std::make_shared<BufferWrapper<const void>>(
            ctx_, "keys", keys.data(), keys.size() * sizeof(KeyType));
        auto no_mask_values_bw = std::make_shared<BufferWrapper<void>>(
            ctx_, "values", output.data(), output.size() * sizeof(DataType));
        tb_->reset_lookup_counter(ctx_);
        tb_->find(ctx_, num_keys, std::move(no_mask_keys_bw), nullptr, params.row_size_bytes,
                  std::move(no_mask_values_bw), nullptr);

        tb_->get_lookup_counter(ctx_, &hits);
        EXPECT_EQ(hits, 2);
    }

    void test_update_ignores_out_of_range_keys() {
        const auto& params = GetParam();
        const size_t row_elements = static_cast<size_t>(params.row_size_bytes) / sizeof(DataType);
        const std::vector<KeyType> keys{static_cast<KeyType>(-1), static_cast<KeyType>(1),
                                        static_cast<KeyType>(params.num_rows)};
        std::vector<DataType> values(keys.size() * row_elements);
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = -1000.0f - static_cast<DataType>(i);
        }
        const std::vector<DataType> original(
            h_table_, h_table_ + static_cast<size_t>(params.num_rows) * row_elements);

        auto keys_bw = std::make_shared<BufferWrapper<const void>>(
            ctx_, "keys", keys.data(), keys.size() * sizeof(KeyType));
        auto values_bw = std::make_shared<BufferWrapper<const void>>(
            ctx_, "values", values.data(), values.size() * sizeof(DataType));
        tb_->update(ctx_, static_cast<int64_t>(keys.size()), std::move(keys_bw),
                    params.row_size_bytes, params.row_size_bytes, std::move(values_bw));

        for (size_t row = 0; row < static_cast<size_t>(params.num_rows); ++row) {
            for (size_t element = 0; element < row_elements; ++element) {
                const DataType expected = row == 1 ? values[row_elements + element]
                                                   : original[row * row_elements + element];
                EXPECT_FLOAT_EQ(h_table_[row * row_elements + element], expected);
            }
        }
    }

    void test_update_accumulate_ignores_out_of_range_keys() {
        const auto& params = GetParam();
        const size_t row_elements = static_cast<size_t>(params.row_size_bytes) / sizeof(DataType);
        const std::vector<KeyType> keys{static_cast<KeyType>(-1), static_cast<KeyType>(1),
                                        static_cast<KeyType>(params.num_rows)};
        std::vector<DataType> updates(keys.size() * row_elements);
        for (size_t i = 0; i < updates.size(); ++i) {
            updates[i] = 1.0f + static_cast<DataType>(i);
        }
        const std::vector<DataType> original(
            h_table_, h_table_ + static_cast<size_t>(params.num_rows) * row_elements);

        auto keys_bw = std::make_shared<BufferWrapper<const void>>(
            ctx_, "keys", keys.data(), keys.size() * sizeof(KeyType));
        auto updates_bw = std::make_shared<BufferWrapper<const void>>(
            ctx_, "updates", updates.data(), updates.size() * sizeof(DataType));
        tb_->update_accumulate(ctx_, static_cast<int64_t>(keys.size()), std::move(keys_bw),
                               params.row_size_bytes, params.row_size_bytes,
                               std::move(updates_bw), params.value_dtype);

        for (size_t row = 0; row < static_cast<size_t>(params.num_rows); ++row) {
            for (size_t element = 0; element < row_elements; ++element) {
                const size_t index = row * row_elements + element;
                const DataType expected =
                    row == 1 ? original[index] + updates[row_elements + element] : original[index];
                EXPECT_FLOAT_EQ(h_table_[index], expected);
            }
        }
    }

    std::shared_ptr<TableType> tb_;
    context_ptr_t ctx_;
    DataType* h_table_;

private:
    void Init() {
        const auto& params = GetParam();
        LinearHostTableConfig cfg;
        cfg.value_dtype = params.value_dtype;
        cfg.max_threads = 64;
        cfg.max_value_size = params.row_size_bytes;
        cfg.num_rows = params.num_rows;

        const size_t table_size = static_cast<size_t>(params.row_size_bytes * params.num_rows);
        NVE_CHECK_(cudaMallocHost(&h_table_, table_size));
        const int64_t table_elements = table_size / sizeof(DataType);
        for (int64_t i = 0; i < table_elements; i++) {
            h_table_[i] = static_cast<DataType>(10000 + i);
        }
        cfg.emb_table = h_table_;

        tb_ = std::make_shared<TableType>(cfg);
        ctx_ = tb_->create_execution_context(0, 0, nullptr, nullptr);
    }

    void find(KeyType* h_keys, int64_t num_keys, DataType* h_data, int64_t row_size) {
        std::vector<bitmask64_t> hit_mask(((static_cast<size_t>(num_keys) + 63) / 64), 0);
        auto keys_bw = std::make_shared<BufferWrapper<const void>>(
            ctx_, "keys", h_keys, static_cast<size_t>(num_keys) * sizeof(KeyType));
        auto hit_mask_bw = std::make_shared<BufferWrapper<bitmask64_t>>(
            ctx_, "hit_mask", hit_mask.data(), hit_mask.size() * sizeof(bitmask64_t));
        auto values_bw = std::make_shared<BufferWrapper<void>>(
            ctx_, "values", h_data, static_cast<size_t>(num_keys) * static_cast<size_t>(row_size));
        tb_->find(ctx_, num_keys, std::move(keys_bw), std::move(hit_mask_bw), row_size,
                     std::move(values_bw), nullptr);
    }

    void update(KeyType h_key, const DataType* h_data, int64_t row_size) {
        auto keys_bw = std::make_shared<BufferWrapper<const void>>(
            ctx_, "keys", &h_key, sizeof(KeyType));
        auto values_bw = std::make_shared<BufferWrapper<const void>>(
            ctx_, "values", h_data, static_cast<size_t>(row_size));
        tb_->update(ctx_, 1, std::move(keys_bw), row_size, row_size, std::move(values_bw));
    }

    void update_accumulate(KeyType h_key, const DataType* h_data, int64_t row_size, DataType_t dtype) {
        auto keys_bw = std::make_shared<BufferWrapper<const void>>(
            ctx_, "keys", &h_key, sizeof(KeyType));
        auto updates_bw = std::make_shared<BufferWrapper<const void>>(
            ctx_, "updates", h_data, static_cast<size_t>(row_size));
        tb_->update_accumulate(ctx_, 1, std::move(keys_bw), row_size, row_size,
                                  std::move(updates_bw), dtype);
    }
};

// Define the test parameters
const LinearHostTableTestParams test_params[] = {
    // row_size_bytes, num_rows, value_dtype
    { 128, 10000, DataType_t::Float32}, 
};

using LHTFixture_INT32_T = LinearHostTableTest<int32_t>;
using LHTFixture_INT64_T = LinearHostTableTest<int64_t>;

#define TEST_FORMAT(test_name, test_func) \
TEST_P(LHTFixture_INT32_T, test_name)     \
{                                         \
    test_func();                          \
}                                         \
TEST_P(LHTFixture_INT64_T, test_name)     \
{                                         \
    test_func();                          \
}                                         \

TEST_FORMAT(find, test_find);
TEST_FORMAT(find_counts_only_new_hits, test_find_counts_only_new_hits);
TEST_FORMAT(find_reports_out_of_range_keys_as_misses, test_find_reports_out_of_range_keys_as_misses);
TEST_FORMAT(update, test_update);
TEST_FORMAT(update_ignores_out_of_range_keys, test_update_ignores_out_of_range_keys);
TEST_FORMAT(update_accumulate, test_update_accumulate);
TEST_FORMAT(update_accumulate_ignores_out_of_range_keys,
            test_update_accumulate_ignores_out_of_range_keys);
#undef TEST_FORMAT

// Instantiate the tests with both type and value parameters
INSTANTIATE_TEST_SUITE_P(
    LinearHostTableTestInt32,
    LHTFixture_INT32_T,
    testing::ValuesIn(test_params));

INSTANTIATE_TEST_SUITE_P(
    LinearHostTableTestInt64,
    LHTFixture_INT64_T,
    testing::ValuesIn(test_params));

} // namespace nve 
