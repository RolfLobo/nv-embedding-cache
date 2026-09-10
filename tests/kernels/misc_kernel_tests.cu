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

#include "kernel_test_common.cuh"

TEST(FindAndPoolValidation, AcceptsSupportedDataTypeCombinations) {
  const DataType_t value_output_types[][2] = {
      {DataType_t::Float32, DataType_t::Float32},
      {DataType_t::Float16, DataType_t::Float16},
      {DataType_t::QInt8RowwiseF32, DataType_t::Float32},
      {DataType_t::QUint8RowwiseF32, DataType_t::Float32},
      {DataType_t::QInt8RowwiseF16, DataType_t::Float16},
      {DataType_t::QUint8RowwiseF16, DataType_t::Float16},
  };
  // Only Float32 accumulation is supported; the accumulator pairs with a Float32 or Float16 weight.
  const DataType_t acc_weight_types[][2] = {
      {DataType_t::Float32, DataType_t::Float32},
      {DataType_t::Float32, DataType_t::Float16},
  };

  for (const auto& value_output : value_output_types) {
    for (const auto& acc_weight : acc_weight_types) {
      EXPECT_NO_THROW(nve::validate_find_and_pool_data_types(
          PoolingType_t::Sum, value_output[0], value_output[1], acc_weight[1], acc_weight[0]));
    }
  }

  // Concatenate does not use the combine kernel's accumulator or weights.
  EXPECT_NO_THROW(nve::validate_find_and_pool_data_types(
      PoolingType_t::Concatenate, DataType_t::Float32, DataType_t::Float32,
      DataType_t::Unknown, DataType_t::Unknown));
}

TEST(FindAndPoolValidation, RejectsUnsupportedDataTypeCombinations) {
  EXPECT_THROW(nve::validate_find_and_pool_data_types(
                   PoolingType_t::Sum, DataType_t::Float64, DataType_t::Float32,
                   DataType_t::Float32, DataType_t::Float32),
               nve::Exception);
  EXPECT_THROW(nve::validate_find_and_pool_data_types(
                   PoolingType_t::Sum, DataType_t::Float32, DataType_t::Float16,
                   DataType_t::Float32, DataType_t::Float32),
               nve::Exception);
  EXPECT_THROW(nve::validate_find_and_pool_data_types(
                   PoolingType_t::Sum, DataType_t::QInt8RowwiseF16, DataType_t::Float32,
                   DataType_t::Float32, DataType_t::Float32),
               nve::Exception);
  EXPECT_THROW(nve::validate_find_and_pool_data_types(
                   PoolingType_t::Sum, DataType_t::Float32, DataType_t::Float32,
                   DataType_t::Float64, DataType_t::Float32),
               nve::Exception);
  EXPECT_THROW(nve::validate_find_and_pool_data_types(
                   PoolingType_t::Sum, DataType_t::Float32, DataType_t::Float32,
                   DataType_t::Float32, DataType_t::Float64),
               nve::Exception);
  EXPECT_THROW(nve::validate_find_and_pool_data_types(
                   PoolingType_t::Sum, DataType_t::Float32, DataType_t::Float32,
                   DataType_t::Float32, DataType_t::Float16),
               nve::Exception);
}

template <typename T>
class EmbeddingCacheLookupHitMaskRefTest : public EmbeddingCacheRefTest<T> {
  using ElemType = typename EmbeddingCacheRefTest<T>::ElemType;
  using IndexType = typename EmbeddingCacheRefTest<T>::IndexType;
  using CacheType = nve::CacheSAHostModify<IndexType, IndexType>;
  using CacheDataType = typename CacheType::CacheData;

  private:
    void ComputeRefResults(const ElemType* table,
                           const IndexType* indices,
                           const std::vector<IndexType>& cache_data) {

        NVE_DEBUG_PRINTF_("compute lookup with hitmask reference results\n");

        uint32_t mask = 0;
        ref_result_.resize(this->batch_ * this->hotness_ * this->num_elements_);
        ref_hitmask_.resize(0);
        int64_t i = 0;
        for (; i < static_cast<int64_t>(this->batch_) * static_cast<int64_t>(this->hotness_); i++) {
            IndexType idx = indices[i];

            bool in_cache = std::binary_search(cache_data.begin(), cache_data.end(), idx);
            if (in_cache) {
               for (uint32_t el = 0; el < this->num_elements_; el++) {
                    ref_result_[i * this->num_elements_ + el] = table[idx * this->num_elements_ + el];
               }
               mask |= 0x80000000;
            }

            if (((i + 1) % 32) == 0) {
                ref_hitmask_.push_back(mask);
                mask = 0;
            } else {
                mask >>= 1;
            }

        }
        uint64_t set_bits = i % 32;
        if (set_bits != 0) {
            mask >>= (32-set_bits-1);
            ref_hitmask_.push_back(mask);
        }
    }

    void LaunchKernel(const ElemType* /*table*/, 
                      const IndexType* indices,
                      CacheDataType& cache_data) {
        int64_t num_indices = static_cast<int64_t>(this->batch_) * static_cast<int64_t>(this->hotness_);
        int64_t num_hitmask_elems = nve::ceil_div(num_indices, static_cast<int64_t>(32));

        result_ = std::make_shared<TestBuffer<ElemType>>(num_indices *this->num_elements_ * sizeof(ElemType));
        hitmask_ = std::make_shared<TestBuffer<uint32_t>>(num_hitmask_elems * sizeof(uint32_t));

        std::memset(hitmask_->ph, 0, num_hitmask_elems * sizeof(uint32_t));
        hitmask_->HtoD(this->stream_);

        NVE_DEBUG_PRINTF_("launch lookup with hitmask kernel\n");

        CHECK_CUDA_ERROR((nve::call_cache_query_hit_mask<IndexType, IndexType>(
            indices, this->batch_ * this->hotness_,
            reinterpret_cast<int8_t*>(result_->pd), hitmask_->pd, cache_data,
            this->stream_, 0, this->num_elements_ * sizeof(ElemType))));
    }

    void CheckResult() {
        NVE_DEBUG_PRINTF_("check lookup with hitmask results\n");
        result_->DtoH(this->stream_);
        hitmask_->DtoH(this->stream_);

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        EXPECT_TRUE(std::equal(ref_hitmask_.begin(), ref_hitmask_.end(), hitmask_->ph));

        int64_t curr_mask_idx = 0;
        int64_t curr_mask_el = 1;

        int64_t num_indices = static_cast<int64_t>(this->batch_) * static_cast<int64_t>(this->hotness_);
        for (int64_t i = 0; i < num_indices; i++) {
            uint32_t curr_mask = ref_hitmask_[curr_mask_idx];
            if ((curr_mask & curr_mask_el) == 0x1) {
                EXPECT_TRUE(std::equal(result_->ph + i * this->num_elements_,
                                       result_->ph + (i + 1) * this->num_elements_,
                                       ref_result_.begin() + i * this->num_elements_));
            }
            if (((i + 1) % 32) == 0) {
                ++curr_mask_idx;
                curr_mask_el = 1;
            } else {
                curr_mask_el <<= 1;
            }
        }
    }

    std::shared_ptr<TestBuffer<ElemType>> result_ = nullptr;
    std::shared_ptr<TestBuffer<uint32_t>> hitmask_ = nullptr;
    std::vector<ElemType> ref_result_;
    std::vector<uint32_t> ref_hitmask_;
};

TYPED_TEST_SUITE_P(EmbeddingCacheLookupHitMaskRefTest);

TYPED_TEST_P(EmbeddingCacheLookupHitMaskRefTest, TestFixedHotnessAgainstRefCpu) {
    for (const auto batch : {17, 2048}) {
        for (const auto hotness : {13, 32}) {
            for (const auto num_elements : {53, 512}) {
                for (const auto allocOnHost: {false, true}){
                    NVE_DEBUG_PRINTF_("running test on batch %d hotness %d num_elements %d allocateDataOn: %s\n", batch, hotness, num_elements, allocOnHost?"host":"device");
                    this->LaunchTest(262144, num_elements, batch, hotness, allocOnHost);
                }
            }
        }
    }
}

REGISTER_TYPED_TEST_SUITE_P(EmbeddingCacheLookupHitMaskRefTest, TestFixedHotnessAgainstRefCpu);

template <typename T>
class ScatterRefTest : public EmbeddingCacheRefTest<T> {
  using ElemType = typename EmbeddingCacheRefTest<T>::ElemType;
  using IndexType = typename EmbeddingCacheRefTest<T>::IndexType;
  using CacheType = nve::CacheSAHostModify<IndexType, IndexType>;
  using CacheDataType = typename CacheType::CacheData;

  private:
    void ComputeRefResults(const ElemType* table,
                           const IndexType* indices,
                           const std::vector<IndexType>& cache_data) {
        NVE_DEBUG_PRINTF_("compute scatter reference results\n");

        // reference - just do gather of all indices
        int64_t num_indices = static_cast<int64_t>(this->batch_) * static_cast<int64_t>(this->hotness_);
        int64_t num_hitmask_elems = nve::ceil_div(num_indices, static_cast<int64_t>(64));
        hitmask_ = std::make_shared<TestBuffer<uint64_t>>(num_hitmask_elems * sizeof(uint64_t));
        input_ = std::make_shared<TestBuffer<ElemType>>(this->batch_ * this->hotness_ * this->num_elements_ * sizeof(ElemType));

        uint64_t mask = 0;

        int64_t i = 0;
        for (; i < num_indices; i++) {
            IndexType idx = indices[i];
            for (uint32_t el = 0; el < this->num_elements_; el++) {
                input_->ph[i * this->num_elements_ + el] = table[idx * this->num_elements_ + el];
            }
            bool in_cache = std::binary_search(cache_data.begin(), cache_data.end(), idx);
            if (in_cache) {
               mask |= 0x80000000;
            }

            if (((i + 1) % 64) == 0) {
                hitmask_->ph[i / 64] = mask;
                mask = 0;
            } else {
                mask >>= 1;
            }

        }
        if (( i % 64) != 0) {
            hitmask_->ph[i / 64] = mask;
        }
    }

    void LaunchKernel(const ElemType* /*table*/,
                      const IndexType* /*indices*/,
                      CacheDataType& /*cache_data*/) {
        NVE_DEBUG_PRINTF_("launch scatter kernel\n");

        hitmask_->HtoD(this->stream_);
        input_->HtoD(this->stream_);
        result_ = std::make_shared<TestBuffer<ElemType>>(this->batch_ * this->hotness_ * this->num_elements_ * sizeof(ElemType));
        uint64_t num_indices = static_cast<uint64_t>(this->batch_) * static_cast<uint64_t>(this->hotness_);
        for (uint64_t i = 0; i < num_indices; i++) {
            for (uint32_t el = 0; el < this->num_elements_; el++) {
                result_->ph[i * this->num_elements_ + el] = ElemType(8);
            }
        }
        result_->HtoD(this->stream_);

        uint32_t embed_width_in_bytes = static_cast<uint32_t>(this->num_elements_ * sizeof(ElemType));
        nve::EmbeddingForwardScatter(input_->pd, result_->pd, embed_width_in_bytes,
                                            embed_width_in_bytes, embed_width_in_bytes,
                                            reinterpret_cast<uint64_t*>(hitmask_->pd),
                                            static_cast<uint32_t>(num_indices), this->stream_);
    }

    void CheckResult() {
        NVE_DEBUG_PRINTF_("check scatter results\n");
        result_->DtoH(this->stream_);

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        int64_t curr_mask_idx = 0;
        int64_t curr_mask_el = 1;

        int64_t num_indices = static_cast<int64_t>(this->batch_) * static_cast<int64_t>(this->hotness_);
        for (int64_t i = 0; i < num_indices; i++) {
            if ((hitmask_->ph[curr_mask_idx] & curr_mask_el) == 0) {
                EXPECT_TRUE(std::equal(result_->ph + i * this->num_elements_,
                                       result_->ph + (i + 1) * this->num_elements_,
                                       input_->ph + i * this->num_elements_));
            } else {
                EXPECT_FALSE(std::equal(result_->ph + i * this->num_elements_,
                                       result_->ph + (i + 1) * this->num_elements_,
                                       input_->ph + i * this->num_elements_));
            }
            if (((i + 1) % 64) == 0) {
                ++curr_mask_idx;
                curr_mask_el = 1;
            } else {
                curr_mask_el <<= 1;
            }
        }
    }

    std::shared_ptr<TestBuffer<ElemType>> result_ = nullptr;
    std::shared_ptr<TestBuffer<ElemType>> input_ = nullptr;
    std::shared_ptr<TestBuffer<uint64_t>> hitmask_ = nullptr;
    std::vector<ElemType> ref_result_;
};

TYPED_TEST_SUITE_P(ScatterRefTest);

TYPED_TEST_P(ScatterRefTest, TestScatterAgainstRefCpu) {
    for (const auto batch : {17, 2048}) {
        for (const auto hotness : {13, 32}) {
            for (const auto num_elements : {13, 512}) {
                for (const auto allocOnHost: {true, false}){
                    NVE_DEBUG_PRINTF_("running test on batch %d hotness %d num_elements %d allocateDataOn: %s\n", batch, hotness, num_elements, allocOnHost?"host":"device");
                    this->LaunchTest(262144, num_elements, batch, hotness, allocOnHost);
                }
            }
        }
    }
}

REGISTER_TYPED_TEST_SUITE_P(ScatterRefTest, TestScatterAgainstRefCpu);

template <typename T>
class UpdateRefTest : public EmbeddingCacheRefTest<T> {
  using ElemType = typename EmbeddingCacheRefTest<T>::ElemType;
  using IndexType = typename EmbeddingCacheRefTest<T>::IndexType;
  using CacheType = nve::CacheSAHostModify<IndexType, IndexType>;
  using CacheDataType = typename CacheType::CacheData;

  private:
    void ComputeRefResults(const ElemType* table,
                           const IndexType* indices,
                           const std::vector<IndexType>& /*cache_data*/) {
        NVE_DEBUG_PRINTF_("compute update reference results\n");

        // reference - just do gather of all indices
        int64_t num_indices = static_cast<int64_t>(this->batch_) * static_cast<int64_t>(this->hotness_);
        input_ = std::make_shared<TestBuffer<ElemType>>(this->batch_ * this->hotness_ * this->num_elements_ * sizeof(ElemType));

        ref_result_.resize(this->num_elements_ * this->num_rows_);
        memcpy(&ref_result_[0], table, this->num_elements_ * this->num_rows_ * sizeof(ElemType));

        // generate inputs to update and update ref
        int64_t i = 0;
        for (; i < num_indices; i++) {
            IndexType idx = indices[i];
            for (uint32_t el = 0; el < this->num_elements_; el++) {
                input_->ph[i * this->num_elements_ + el] = table[idx * this->num_elements_ + el] + ElemType(i);
                ref_result_[idx * this->num_elements_ + el] += ElemType(i);
            }
        }
    }

    void LaunchKernel(const ElemType* table,
                      const IndexType* indices,
                      CacheDataType& /*cache_data*/) {
        NVE_DEBUG_PRINTF_("launch update kernel\n");

        input_->HtoD(this->stream_);

        uint32_t embed_width_in_bytes = static_cast<uint32_t>(this->num_elements_ * sizeof(ElemType));
        uint64_t num_indices = static_cast<uint64_t>(this->batch_) * static_cast<uint64_t>(this->hotness_);
        nve::UpdateTable(input_->pd, indices, const_cast<ElemType*>(table),
                         embed_width_in_bytes, embed_width_in_bytes, embed_width_in_bytes,
                         static_cast<uint32_t>(num_indices), this->num_rows_, this->stream_);
    }

    void CheckResult() {
        NVE_DEBUG_PRINTF_("check update results\n");

        ElemType* curr_table = this->syncTable();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        EXPECT_TRUE(std::equal(ref_result_.begin(), ref_result_.end(), curr_table));
    }

    std::shared_ptr<TestBuffer<ElemType>> input_ = nullptr;
    std::vector<ElemType> ref_result_;
};

TYPED_TEST_SUITE_P(UpdateRefTest);

// this test should get unique keys, therefore using batch size of 1
TYPED_TEST_P(UpdateRefTest, TestUpdateAgainstRefCpu) {
    for (const auto hotness : {17, 2048}) {
        for (const auto num_elements : {53, 512}) {
            for (const auto allocOnHost: {true, false}){
                    NVE_DEBUG_PRINTF_("running test on batch %d hotness %d num_elements %d allocateDataOn: %s\n", 1, hotness, num_elements, allocOnHost?"host":"device");
                    this->LaunchTest(262144, num_elements, 1, hotness, allocOnHost);
                }
        }
    }
}

REGISTER_TYPED_TEST_SUITE_P(UpdateRefTest, TestUpdateAgainstRefCpu);

template <typename T>
class UpdateAccumulateRefTest : public EmbeddingCacheRefTest<T> {
  using ElemType = typename EmbeddingCacheRefTest<T>::ElemType;
  using IndexType = typename EmbeddingCacheRefTest<T>::IndexType;
  using CacheType = nve::CacheSAHostModify<IndexType, IndexType>;
  using CacheDataType = typename CacheType::CacheData;

  private:
    void ComputeRefResults(const ElemType* table,
                           const IndexType* indices,
                           const std::vector<IndexType>& /*cache_data*/) {
        NVE_DEBUG_PRINTF_("compute update accumulate reference results\n");

        // reference - just do gather of all indices
        int64_t num_indices = static_cast<int64_t>(this->batch_) * static_cast<int64_t>(this->hotness_);
        input_ = std::make_shared<TestBuffer<ElemType>>(this->batch_ * this->hotness_ * this->num_elements_ * sizeof(ElemType));

        ref_result_.resize(this->num_elements_ * this->num_rows_);
        memcpy(&ref_result_[0], table, this->num_elements_ * this->num_rows_ * sizeof(ElemType));

        // generate inputs to update and update ref
        std::mt19937 genr(0X753812);
        std::uniform_int_distribution<int32_t> dist_nom(-3, 3);
        std::uniform_int_distribution<uint32_t> dist_denom(5, 9);

        for (int64_t i = 0; i < num_indices; i++) {
            IndexType idx = indices[i];
            for (uint32_t el = 0; el < this->num_elements_; el++) {
                float val = float(dist_nom(genr)) / float(1 << dist_denom(genr));
                input_->ph[i * this->num_elements_ + el] = ElemType(val);
                ref_result_[idx * this->num_elements_ + el] += ElemType(val);
            }
        }
    }

    void LaunchKernel(const ElemType* table,
                      const IndexType* indices,
                      CacheDataType& /*cache_data*/) {
        NVE_DEBUG_PRINTF_("launch update accumulate kernel\n");

        input_->HtoD(this->stream_);

        uint64_t num_indices = static_cast<uint64_t>(this->batch_) * static_cast<uint64_t>(this->hotness_);
        nve::UpdateAccumulateTable(input_->pd, indices, const_cast<ElemType*>(table),
                                   this->num_elements_, this->num_elements_, this->num_elements_,
                                   static_cast<uint32_t>(num_indices), this->num_rows_, this->stream_);
    }

    void CheckResult() {
        NVE_DEBUG_PRINTF_("check update accumulate results\n");

        ElemType* curr_table = this->syncTable();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        const float tolerance = 1e-5f;
        EXPECT_TRUE(std::equal(ref_result_.begin(), ref_result_.end(), curr_table, Near(tolerance)));
    }

    std::shared_ptr<TestBuffer<ElemType>> input_ = nullptr;
    std::vector<ElemType> ref_result_;
};

TYPED_TEST_SUITE_P(UpdateAccumulateRefTest);

TYPED_TEST_P(UpdateAccumulateRefTest, TestUpdateAccumulateAgainstRefCpu) {
    for (const auto batch : {171}) {
        for (const auto hotness : {16, 111}) {
            for (const auto num_elements : {53, 512}) {
                for (const auto allocOnHost: {true, false}){
                    NVE_DEBUG_PRINTF_("running test on batch %d hotness %d num_elements %d allocateDataOn: %s\n", batch, hotness, num_elements, allocOnHost?"host":"device");
                        this->LaunchTest(262144, num_elements, batch, hotness, allocOnHost);
            }
                }
        }
    }
}

REGISTER_TYPED_TEST_SUITE_P(UpdateAccumulateRefTest, TestUpdateAccumulateAgainstRefCpu);

// Keys outside [0, num_rows) must be ignored instead of writing outside the table.
template <typename KeyType>
static void RunOutOfRangeKeysTest(bool accumulate) {
    constexpr uint64_t num_rows = 8;
    constexpr uint32_t num_elements = 16;
    constexpr uint32_t row_bytes = num_elements * sizeof(float);
    // -1, 8 and 100 fall outside the table, the rest address real rows
    const std::vector<KeyType> keys{3, -1, 8, 0, 100, 7};
    const auto num_keys = static_cast<int32_t>(keys.size());

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    TestBuffer<float> table(num_rows * row_bytes);
    TestBuffer<float> updates(keys.size() * row_bytes);
    TestBuffer<KeyType> d_keys(keys.size() * sizeof(KeyType));

    for (uint64_t row = 0; row < num_rows; row++) {
        for (uint32_t el = 0; el < num_elements; el++) {
            table.ph[row * num_elements + el] = float(row * num_elements + el);
        }
    }
    for (size_t i = 0; i < keys.size(); i++) {
        for (uint32_t el = 0; el < num_elements; el++) {
            updates.ph[i * num_elements + el] = 1000.0f + float(i);
        }
        d_keys.ph[i] = keys[i];
    }

    // Reference: apply only the in-range keys, everything else stays untouched
    std::vector<float> ref_result(table.ph, table.ph + num_rows * num_elements);
    for (size_t i = 0; i < keys.size(); i++) {
        if (keys[i] < 0 || static_cast<uint64_t>(keys[i]) >= num_rows) {
            continue;
        }
        for (uint32_t el = 0; el < num_elements; el++) {
            float& dst = ref_result[static_cast<size_t>(keys[i]) * num_elements + el];
            const float src = updates.ph[i * num_elements + el];
            dst = accumulate ? (dst + src) : src;
        }
    }

    table.HtoD(stream);
    updates.HtoD(stream);
    d_keys.HtoD(stream);

    if (accumulate) {
        nve::UpdateAccumulateTable(updates.pd, d_keys.pd, table.pd,
                                   num_elements, num_elements, num_elements,
                                   num_keys, num_rows, stream);
    } else {
        nve::UpdateTable(updates.pd, d_keys.pd, table.pd,
                         row_bytes, row_bytes, row_bytes,
                         num_keys, num_rows, stream);
    }

    table.DtoH(stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    EXPECT_TRUE(std::equal(ref_result.begin(), ref_result.end(), table.ph));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

TEST(UpdateOutOfRangeKeys, UpdateIgnoresInvalidKeysInt32) {
    RunOutOfRangeKeysTest<int32_t>(false /*accumulate*/);
}

TEST(UpdateOutOfRangeKeys, UpdateIgnoresInvalidKeysInt64) {
    RunOutOfRangeKeysTest<int64_t>(false /*accumulate*/);
}

TEST(UpdateOutOfRangeKeys, UpdateAccumulateIgnoresInvalidKeysInt32) {
    RunOutOfRangeKeysTest<int32_t>(true /*accumulate*/);
}

TEST(UpdateOutOfRangeKeys, UpdateAccumulateIgnoresInvalidKeysInt64) {
    RunOutOfRangeKeysTest<int64_t>(true /*accumulate*/);
}

// Keys outside [0, num_rows) must come out as the default row, the rest untouched.
template <typename KeyType>
static void RunSanitizeKeysTest() {
    constexpr uint64_t num_rows = 8;
    constexpr KeyType default_row = 3;
    const std::vector<KeyType> keys{5, -1, 8, 0, 100, 7, -100, 3};
    const auto num_keys = static_cast<int64_t>(keys.size());

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    TestBuffer<KeyType> d_keys(keys.size() * sizeof(KeyType));
    TestBuffer<KeyType> d_out(keys.size() * sizeof(KeyType));
    std::copy(keys.begin(), keys.end(), d_keys.ph);
    d_keys.HtoD(stream);

    nve::launch_sanitize_keys_kernel<KeyType>(d_keys.pd, d_out.pd, num_keys, num_rows, default_row,
                                              stream);

    d_out.DtoH(stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < keys.size(); i++) {
        const KeyType expected = nve::key_in_range(keys[i], num_rows) ? keys[i] : default_row;
        EXPECT_EQ(expected, d_out.ph[i]) << "key slot " << i;
    }
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

TEST(SanitizeKeys, ReplacesInvalidKeysInt32) {
    RunSanitizeKeysTest<int32_t>();
}

TEST(SanitizeKeys, ReplacesInvalidKeysInt64) {
    RunSanitizeKeysTest<int64_t>();
}

TEST(SanitizeKeys, RejectsDefaultRowOutsideTable) {
    TestBuffer<int64_t> buf(sizeof(int64_t));
    EXPECT_THROW(nve::launch_sanitize_keys_kernel<int64_t>(buf.pd, buf.pd, 1, 8 /*num_rows*/,
                                                           8 /*default_row*/, 0),
                 nve::Exception);
}

template <typename T>
class EmbeddingCacheFusedUpdateAccumlateTest : public EmbeddingCacheRefTest<T> {
  public:

  using ElemType = typename EmbeddingCacheRefTest<T>::ElemType;
  using IndexType = typename EmbeddingCacheRefTest<T>::IndexType;
  using CacheType = nve::CacheSAHostModify<IndexType, IndexType>;
  using CacheDataType = typename CacheType::CacheData;

  private:
    void ComputeRefResults(const ElemType* table,
                           const IndexType* indices,    
                           const std::vector<IndexType>& /*cache_data*/) {
        NVE_DEBUG_PRINTF_("compute table and cache results\n");
        NVE_DEBUG_PRINTF_("compute update reference results\n");

        // reference - just do gather of all indices
        IndexType num_keys = this->batch_ * this->hotness_;

        // first compute cache content
        ref_cache_ = this->GetCacheContent();
        
        // second dedup keys
        unique_keys_ = std::make_shared<TestBuffer<IndexType>>(num_keys * sizeof(IndexType));
        // allocate 1 extra element for counts, because we call cub exclusive prefix sum kernel
        // to compute n + 1 outputs, and it reads the last input element even though it is not used
        // in the output
        auto counts_out = std::make_shared<TestBuffer<IndexType>>((num_keys + 1) * sizeof(IndexType));
        
        auto inverse_buffer = std::make_shared<TestBuffer<IndexType>>(num_keys * sizeof(IndexType));
        auto offsets = std::make_shared<TestBuffer<IndexType>>((num_keys + 1) * sizeof(IndexType));
        
        std::shared_ptr<nve::Deduper<IndexType, -1>> deduper_ = std::make_shared<nve::Deduper<IndexType, -1>>(nve::GetDefaultAllocator());
        size_t tmp_mem_size_device, tmp_mem_size_host;
        deduper_->get_alloc_requirements(num_keys, tmp_mem_size_device, tmp_mem_size_host);
        char* tmp_device_mem;
        CHECK_CUDA_ERROR(cudaMalloc(&tmp_device_mem, tmp_mem_size_device));
        char* tmp_host_mem;
        CHECK_CUDA_ERROR(cudaMallocHost(&tmp_host_mem, tmp_mem_size_host));
        deduper_->set_and_init_buffers(num_keys, tmp_device_mem, tmp_host_mem);

        CHECK_CUDA_ERROR(cudaMallocHost(&h_num_runs_out_, sizeof(IndexType) * 2));

        deduper_->dedup(reinterpret_cast<const IndexType*>(indices),
                        num_keys, unique_keys_->pd,
                        counts_out->pd,
                        nullptr, 
                        h_num_runs_out_,
                        inverse_buffer->pd, 
                        offsets->pd,
                        this->stream_);

        unique_keys_->DtoH(this->stream_);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        input_ = std::make_shared<TestBuffer<ElemType>>(*h_num_runs_out_ * this->num_elements_ * sizeof(ElemType));

        ref_result_.resize(this->num_elements_ * this->num_rows_);
        memcpy(&ref_result_[0], table, this->num_elements_ * this->num_rows_ * sizeof(ElemType));

        // generate inputs to update and update ref
        for (int64_t i = 0; i < *h_num_runs_out_; i++) {
            IndexType idx = unique_keys_->ph[i];
            for (uint32_t el = 0; el < this->num_elements_; el++) {
                input_->ph[i * this->num_elements_ + el] = ElemType(i);
                ref_result_[idx * this->num_elements_ + el] += ElemType(i);
                if (ref_cache_.count(idx))
                    ref_cache_[idx][el] += ElemType(i);
            }
        }

        CHECK_CUDA_ERROR(cudaFree(tmp_device_mem));
        CHECK_CUDA_ERROR(cudaFreeHost(tmp_host_mem));
    }

    void LaunchKernel(const ElemType* table, 
                      const IndexType* /*indices*/,
                      CacheDataType& cache_data) {
        NVE_DEBUG_PRINTF_("launch update kernel\n");

        input_->HtoD(this->stream_);

        uint64_t num_indices = *h_num_runs_out_;
        CHECK_CUDA_ERROR((nve::call_update_accumulate_no_sync_fused_with_pipeline<IndexType,IndexType>(reinterpret_cast<int8_t*>(input_->pd),
                                   (int8_t*)(table), 
                                   unique_keys_->pd,
                                   static_cast<IndexType>(num_indices),
                                   this->num_elements_ * sizeof(ElemType),
                                   cache_data,
                                   this->stream_)));
    }

    void CheckResult() {
        NVE_DEBUG_PRINTF_("check update results\n");

        ElemType* curr_table = this->syncTable();
        auto test_cache = this->GetCacheContent();

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        for (uint32_t j = 0; j < this->num_rows_; j ++ ) {
            for (uint32_t i = 0; i < this->num_elements_; i++)
            {
                auto ref_val = ref_result_[j*this->num_elements_ + i];
                auto test_val = curr_table[j*this->num_elements_ + i];
                ASSERT_FLOAT_EQ(ref_val, test_val);
            }
        }

        // compare cache content
        ASSERT_EQ(test_cache.size(), ref_cache_.size());
        for (const auto& ent : test_cache)
        {
            auto key = ent.first;
            auto test_vec = ent.second;
            auto ref_it = ref_cache_.find(key);
            ASSERT_TRUE(ref_it != ref_cache_.end());
            auto ref_vec = ref_it->second;
            for (uint32_t i = 0; i < this->num_elements_; i++)
            {
                auto ref_val = ref_vec[i];
                auto test_val = test_vec[i];
                ASSERT_FLOAT_EQ(ref_val, test_val);
            }

        }

    }

    std::shared_ptr<TestBuffer<ElemType>> input_ = nullptr;
    std::vector<ElemType> ref_result_;
    std::shared_ptr<TestBuffer<IndexType>> unique_keys_ = nullptr;
    IndexType* h_num_runs_out_ = nullptr;
    std::map<IndexType, std::vector<ElemType>> ref_cache_;
};

TYPED_TEST_SUITE_P(EmbeddingCacheFusedUpdateAccumlateTest);

TYPED_TEST_P(EmbeddingCacheFusedUpdateAccumlateTest, TestFusedAccumulateRefCpu) {
    for (const auto batch : {32*15*8, 12, 348989}) {
        for (const auto hotness : {1}) {
            for (const auto num_elements : {128}) {
                for (const auto allocOnHost: {true, false}){
                    NVE_DEBUG_PRINTF_("running test on batch %d hotness %d num_elements %d allocateDataOn: %s\n", batch, hotness, num_elements, allocOnHost?"host":"device");
                    this->LaunchTest(262144, num_elements, batch, hotness, allocOnHost);
                }
            }
        }
    }
}

REGISTER_TYPED_TEST_SUITE_P(EmbeddingCacheFusedUpdateAccumlateTest, TestFusedAccumulateRefCpu);

template <typename T>
class EmbeddingCacheLookupSortGatherRefTest : public EmbeddingCacheRefTest<T> {
  public:

  using ElemType = typename EmbeddingCacheRefTest<T>::ElemType;
  using IndexType = typename EmbeddingCacheRefTest<T>::IndexType;
  using CacheType = nve::CacheSAHostModify<IndexType, IndexType>;
  using CacheDataType = typename CacheType::CacheData;

  private:
    void ComputeRefResults(const ElemType* table,
                           const IndexType* indices,    
                           const std::vector<IndexType>& /*cache_data*/) {
        NVE_DEBUG_PRINTF_("compute lookup reference results\n");
        ref_result_.resize(0);
        ASSERT_TRUE(this->hotness_ == 1);
        for (uint32_t b = 0; b < this->batch_; b++) {
            for (uint32_t el = 0; el < this->num_elements_; el++) {
                ElemType acc = 0;
                acc = table[indices[b] * this->num_elements_ + el];
                ref_result_.push_back(acc);
            }
        }
    }

    void LaunchKernel(const ElemType* table, 
                      const IndexType* indices,
                      CacheDataType& cache_data) {
        result_ = std::make_shared<TestBuffer<ElemType>>(this->batch_ * this->hotness_ * this->num_elements_ * sizeof(ElemType));
        NVE_DEBUG_PRINTF_("launch sort gather lookup kernel\n");

        // first calc required aux size
        size_t bytes = 0;
        CHECK_CUDA_ERROR((nve::call_sort_gather<IndexType, IndexType>(
            reinterpret_cast<const int8_t*>(table), 
            reinterpret_cast<int8_t*>(result_->pd),
            indices, 
            nullptr, 
            bytes,
            this->batch_ * this->hotness_,
            this->num_elements_ * sizeof(ElemType),
            this->kernel_param_,
            cache_data, 
            this->stream_)));
        ASSERT_TRUE(bytes > 0 && bytes != static_cast<size_t>(-1));

        // alloc aux
        int8_t* aux = nullptr;
        CHECK_CUDA_ERROR(cudaMalloc(&aux, bytes));

        // launch kernel
        CHECK_CUDA_ERROR((nve::call_sort_gather<IndexType, IndexType>(
            reinterpret_cast<const int8_t*>(table), 
            reinterpret_cast<int8_t*>(result_->pd),
            indices, 
            aux, 
            bytes,
            this->batch_ * this->hotness_,
            this->num_elements_ * sizeof(ElemType),
            1024,
            cache_data, 
            this->stream_)));

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        //clean up
        CHECK_CUDA_ERROR(cudaFree(aux));
    }

    void CheckResult() {
        NVE_DEBUG_PRINTF_("check lookup results\n");
        result_->DtoH(this->stream_);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        for (uint32_t j = 0; j < this->hotness_ * this->batch_; j ++ ) {
            for (uint32_t i = 0; i < this->num_elements_; i++)
            {
                auto ref_val = ref_result_[j*this->num_elements_ + i];
                auto test_val = result_->ph[j*this->num_elements_ + i];
                ASSERT_FLOAT_EQ(ref_val, test_val);
            }
        }
    }

    std::shared_ptr<TestBuffer<ElemType>> result_ = nullptr;
    std::vector<ElemType> ref_result_;
};

TYPED_TEST_SUITE_P(EmbeddingCacheLookupSortGatherRefTest);

TYPED_TEST_P(EmbeddingCacheLookupSortGatherRefTest, TestSortGather) {
    for (const auto batch : {4096, 1, 33, 4097, 16*1024, 3*4096+1, 2*4096+1}) {
        for (const auto hotness : {1}) {
            for (const auto num_elements : {128, 1, 31}) {
                for (const auto allocOnHost: {false}){
                    for (const auto kernel_param : {1, 17, 32, 4096}) {
                        NVE_DEBUG_PRINTF_("running test on batch %d hotness %d num_elements %d allocateDataOn: %s kernel_param %d\n", batch, hotness, num_elements, allocOnHost?"host":"device", kernel_param);
                        this->LaunchTest(262144, num_elements, batch, hotness, allocOnHost, kernel_param);
                    }
                }
            }
        }
    }
}

REGISTER_TYPED_TEST_SUITE_P(EmbeddingCacheLookupSortGatherRefTest, TestSortGather);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ElemT, typename IndexT>
struct EmbedTestTypeCombo {
  typedef ElemT ElemType;
  typedef IndexT IndexType;
};

typedef ::testing::Types<EmbedTestTypeCombo<float, int32_t>,
                         EmbedTestTypeCombo<float, int64_t>,
                         EmbedTestTypeCombo<__half, int32_t>,
                         EmbedTestTypeCombo<__half, int64_t>>
    EmbedTestTypes;

typedef ::testing::Types<EmbedTestTypeCombo<float, int32_t>,
                         EmbedTestTypeCombo<__half, int64_t>>
    EmbedElementAgnosticTestTypes;

typedef ::testing::Types<EmbedTestTypeCombo<float, int32_t>>
    EmbedBasicTestTypes;

class EmbeddingCacheLookupHitMaskRefTestNames {
 public:
  template <typename T>
  static std::string GetName(int i) {
    return EmbeddingCacheRefTestNames::GetName<T>("EmbeddingCacheLookupHitMaskRefTest_", i);
  }
};

class ScatterRefTestNames {
 public:
  template <typename T>
  static std::string GetName(int i) {
    return EmbeddingCacheRefTestNames::GetName<T>("ScatterRefTest_", i);
  }
};

class UpdateRefTestNames {
 public:
  template <typename T>
  static std::string GetName(int i) {
    return EmbeddingCacheRefTestNames::GetName<T>("UpdateRefTest_", i);
  }
};

class UpdateAccumulateRefTestNames {
 public:
  template <typename T>
  static std::string GetName(int i) {
    return EmbeddingCacheRefTestNames::GetName<T>("UpdateAccumulateRefTest_", i);
  }
};

class FusedUpdateAccumulateRefTestNames {
 public:
  template <typename T>
  static std::string GetName(int i) {
    return EmbeddingCacheRefTestNames::GetName<T>("FusedUpdateAccumulateRefTest_", i);
  }
};

class EmbeddingCacheLookupSortGatherRefTestNames {
 public:
  template <typename T>
  static std::string GetName(int i) {
    return EmbeddingCacheRefTestNames::GetName<T>("LookupSortGatherRefTest_", i);
  }
};

INSTANTIATE_TYPED_TEST_SUITE_P(EmbeddingLookupHitMask,
                               EmbeddingCacheLookupHitMaskRefTest,
                               EmbedTestTypes,
                               EmbeddingCacheLookupHitMaskRefTestNames);

INSTANTIATE_TYPED_TEST_SUITE_P(Scatter,
                               ScatterRefTest,
                               EmbedElementAgnosticTestTypes,
                               ScatterRefTestNames);

INSTANTIATE_TYPED_TEST_SUITE_P(Update,
                               UpdateRefTest,
                               EmbedElementAgnosticTestTypes,
                               UpdateRefTestNames);

INSTANTIATE_TYPED_TEST_SUITE_P(UpdateAccumulate,
                               UpdateAccumulateRefTest,
                               EmbedElementAgnosticTestTypes,
                               UpdateAccumulateRefTestNames);

INSTANTIATE_TYPED_TEST_SUITE_P(FusedUpdateAccumulate,
                               EmbeddingCacheFusedUpdateAccumlateTest,
                               EmbedBasicTestTypes,
                               FusedUpdateAccumulateRefTestNames);

INSTANTIATE_TYPED_TEST_SUITE_P(CacheLookupSortGather,
                               EmbeddingCacheLookupSortGatherRefTest,
                               EmbedBasicTestTypes,
                               EmbeddingCacheLookupSortGatherRefTestNames);

TEST(dedupgrad, dedupgrad)
{
    using IndexType = int64_t; //int32_t;
    using DataType = float;

    const size_t embedding_size = 212;
    const size_t num_rows = 1000000;
    const size_t row_size = sizeof(DataType)*embedding_size;
    
    const size_t batch = 512;
    const size_t hotness = 4096;
    const auto num_keys = batch * hotness;
    auto keys = std::make_shared<TestBuffer<IndexType>>(num_keys * sizeof(IndexType));
    auto unique_keys = std::make_shared<TestBuffer<IndexType>>(num_keys * sizeof(IndexType));
    // allocate an extra element for loc_map, because it is reused for counters, and the
    // way we use counters in Dedup may lead to a read of extra element in one of cub kernels 
    auto loc_map = std::make_shared<TestBuffer<IndexType>>((num_keys + 1) * sizeof(IndexType));
    IndexType* h_num_runs_out;
    auto inverse_buffer = std::make_shared<TestBuffer<IndexType>>(num_keys * sizeof(IndexType));
    auto offsets = std::make_shared<TestBuffer<IndexType>>((num_keys + 1) * sizeof(IndexType));
    auto grads = std::make_shared<TestBuffer<DataType>>(num_keys * row_size);
    auto unique_grads = std::make_shared<TestBuffer<DataType>>(num_keys * row_size);
    auto ref = std::make_shared<TestBuffer<DataType>>(num_keys * row_size);

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    
    const float alpha = 1.05f;
    auto sg = getSampleGenerator<IndexType>(alpha, static_cast<IndexType>(num_rows), hotness, 283982);
    
    // make sure we have keys appearing more than SPLIT_LIMIT times
    // change approx half of batches to contain approx half identical keys
    std::mt19937 genr(0X753812);
    std::bernoulli_distribution distrib(0.5);
    std::uniform_int_distribution<uint32_t> dist_index(0, hotness-1);

    for (uint32_t b = 0; b < batch; b++)
    {
        auto sample = sg->getCategoryIndices();
        if (distrib(genr)) {
            auto duplicate_idx = dist_index(genr);
            IndexType key_to_duplicate = sample[duplicate_idx];
            for (uint32_t h = 0; h < hotness; h++) {
                if (distrib(genr)) sample[h] = key_to_duplicate;
            }
        }
        std::copy(sample.begin(), sample.end(), keys->ph + b*hotness);
    }

    keys->HtoD(stream);
    
    constexpr int32_t MAX_RUN_SIZE = 1024;
    std::shared_ptr<nve::Deduper<IndexType, MAX_RUN_SIZE>> deduper_ = std::make_shared<nve::Deduper<IndexType, MAX_RUN_SIZE>>(nve::GetDefaultAllocator());
    size_t tmp_mem_size_device, tmp_mem_size_host;
    deduper_->get_alloc_requirements(num_keys, tmp_mem_size_device, tmp_mem_size_host);
    char* tmp_device_mem;
    CHECK_CUDA_ERROR(cudaMalloc(&tmp_device_mem, tmp_mem_size_device));
    char* tmp_host_mem;
    CHECK_CUDA_ERROR(cudaMallocHost(&tmp_host_mem, tmp_mem_size_host));
    deduper_->set_and_init_buffers(num_keys, tmp_device_mem, tmp_host_mem);

    CHECK_CUDA_ERROR(cudaMallocHost(&h_num_runs_out, sizeof(IndexType) * 2));

    memset(grads->ph, 0, grads->size_);
    std::uniform_real_distribution<float> dist_float(-1.0f, 1.0f);
    for (uint32_t i = 0; i < num_keys; i++) {
        for (uint32_t j = 0; j < embedding_size; j++) {
            (grads->ph)[i*embedding_size + j] = static_cast<DataType>(dist_float(genr));
        }
    }
    grads->HtoD(stream);

    memset(unique_grads->ph, 0, unique_grads->size_);
    unique_grads->HtoD(stream);

    deduper_->dedup(reinterpret_cast<const IndexType*>(keys->pd),
                    num_keys, unique_keys->pd, 
                    loc_map->pd, // reuse loc_map buffer for counters
                    loc_map->pd, 
                    h_num_runs_out, 
                    inverse_buffer->pd, 
                    offsets->pd,
                    stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    
    DedupGradients<IndexType>(reinterpret_cast<const void*>(grads->pd), (void*)unique_grads->pd, 
                    DataType_t::Float32,
                    unique_keys->pd, 
                    (MAX_RUN_SIZE == -1) ? nullptr : loc_map->pd,
                    offsets->pd,
                    inverse_buffer->pd,
                    row_size,
                    h_num_runs_out,
                    stream);

    // calculate ref
    std::map<IndexType, DataType*> ref_grads;
    for (uint32_t i = 0; i < num_keys; i++) 
    {
        auto key = keys->ph[i];
        if (ref_grads.count(key) == 0) {
            DataType* p = new DataType[embedding_size];
            memset(p, 0, row_size);
            ref_grads[key] = p;
        }
        auto ref_grad = ref_grads[key];
        for (uint32_t j = 0; j < embedding_size; j++) {
            ref_grad[j] += grads->ph[i * embedding_size + j];
        }
    }

    // check results
    unique_grads->DtoH(stream);
    unique_keys->DtoH(stream);
    inverse_buffer->DtoH(stream);
    loc_map->DtoH(stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    for (IndexType i = 0; i < h_num_runs_out[0]; i++) 
    {
        auto unq_key = unique_keys->ph[i];
        DataType* kernel_ptr = unique_grads->ph + i * embedding_size;
        DataType* ref_ptr = ref_grads[unq_key];
        const float tolerance = (MAX_RUN_SIZE == -1) ? 0 : 1e-3f;
        EXPECT_TRUE(std::equal(ref_ptr, ref_ptr + embedding_size, kernel_ptr, Near(tolerance)));
    }

    CHECK_CUDA_ERROR(cudaFree(tmp_device_mem));
    CHECK_CUDA_ERROR(cudaFreeHost(tmp_host_mem));
}

template <typename KeyT, bool gpu_histogram_flag>
struct HistogramTestType {
  typedef KeyT KeyType;
  static bool constexpr runOnGPU = gpu_histogram_flag;
};

template <typename T>
class HistogramTest : public ::testing::Test {
  public:
    using KeyType = typename T::KeyType;

    void launchTest(KeyType num_keys) {
        const size_t embedding_size = 512;

        std::mt19937 genr(0X753812);
        std::uniform_int_distribution<KeyType> dist_keys(1, num_keys * 200);

        std::set<KeyType> unique_keys;
        while (unique_keys.size() < static_cast<size_t>(num_keys)) {
            unique_keys.insert(dist_keys(genr));
        }

        auto keys = std::make_shared<TestBuffer<KeyType>>(num_keys * (num_keys + 1) * sizeof(KeyType) / 2);
        KeyType idx = 0;
        std::vector<KeyType> keys_(unique_keys.begin(), unique_keys.end());

        for (KeyType i = 0 ; i < num_keys; i++) {
            for (int32_t j = 0; j <= i; j++) {
                keys->ph[idx++] = keys_[j];
            }
        }
        auto total_num_keys = idx;
        ASSERT_TRUE(total_num_keys > 0);

        std::map<KeyType, int32_t> histogram;
        for (KeyType i = 0 ; i < total_num_keys; i++) {
            KeyType key = keys->ph[i];
            if (histogram.find(key) != histogram.end()) {
                ++histogram[key];
            } else {
                histogram[key] = 1;
            }
        }

        std::vector<KeyType> unique_keys_h;
        std::vector<float> priority_h;
        std::vector<int8_t*> data_ptrs_h;
        KeyType* d_hist_storage = nullptr;
        KeyType num_unique_keys = 0;

        if (T::runOnGPU) {
            keys->HtoD(0);

            // run the test
            nve::DefaultGPUHistogram<KeyType> histogramGPU(total_num_keys);
            size_t histAllocSize = histogramGPU.get_alloc_size();
            CHECK_CUDA_ERROR(cudaMalloc(&d_hist_storage, histAllocSize));

            histogramGPU.compute_histogram(reinterpret_cast<const KeyType*>(keys->pd), total_num_keys,
                                        reinterpret_cast<const int8_t*>(keys->ph),
                                        embedding_size, d_hist_storage, 0);

            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            num_unique_keys = static_cast<KeyType>(histogramGPU.get_num_bins());

            unique_keys_h.resize(num_unique_keys);
            priority_h.resize(num_unique_keys);
            data_ptrs_h.resize(num_unique_keys);

            KeyType* unique_keys_d = histogramGPU.get_keys();
            float* priority_d = histogramGPU.get_priority();
            const int8_t* const* data_ptrs_d = histogramGPU.get_data();

            CHECK_CUDA_ERROR(cudaMemcpy(&unique_keys_h[0], unique_keys_d, num_unique_keys * sizeof(KeyType), cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaMemcpy(&priority_h[0], priority_d, num_unique_keys * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaMemcpy(&data_ptrs_h[0], data_ptrs_d, num_unique_keys * sizeof(int8_t*), cudaMemcpyDeviceToHost));
        } else {
            nve::DefaultHistogram<KeyType> histogramCPU(
                reinterpret_cast<const KeyType*>(keys->ph),
                static_cast<size_t>(total_num_keys),
                reinterpret_cast<const int8_t*>(keys->ph),
                static_cast<size_t>(embedding_size),
                false);

            num_unique_keys = static_cast<KeyType>(histogramCPU.get_num_bins());
            unique_keys_h.resize(num_unique_keys);
            priority_h.resize(num_unique_keys);
            data_ptrs_h.resize(num_unique_keys);

            memcpy(&unique_keys_h[0], histogramCPU.get_keys(), num_unique_keys * sizeof(KeyType));
            memcpy(&priority_h[0], histogramCPU.get_priority(), num_unique_keys * sizeof(float));
            memcpy(&data_ptrs_h[0], histogramCPU.get_data(), num_unique_keys * sizeof(int8_t*));
        }

        // get ref histogram sorted by priority
        std::map<int32_t, KeyType> histogram_by_count;
        for (typename std::map<KeyType, int32_t>::iterator itr = histogram.begin(); itr != histogram.end(); itr++) {
            histogram_by_count[(*itr).second] = (*itr).first;
        }

        EXPECT_TRUE (histogram_by_count.size() == num_unique_keys);
        EXPECT_TRUE (num_keys == num_unique_keys);
        ASSERT_TRUE (total_num_keys != 0);

        idx = num_unique_keys - 1;
        for (typename std::map<int32_t, KeyType>::iterator itr = histogram_by_count.begin(); itr != histogram_by_count.end(); itr++) {
            KeyType key = (*itr).second;
            float priority = float((*itr).first) / float(total_num_keys);
            EXPECT_TRUE (unique_keys_h[idx] == key);
            EXPECT_TRUE (priority_h[idx] == priority);
            for (KeyType i = 0; i < num_keys; i++) {
                if (keys->ph[i] == key) {
                    const int8_t* data_ptr_ref = reinterpret_cast<const int8_t*>(keys->ph) + i * embedding_size;
                    const int8_t* data_ptr = data_ptrs_h[idx];
                    EXPECT_TRUE (data_ptr == data_ptr_ref);
                    break;
                }
            }
            --idx;
        }

        if (T::runOnGPU) {
            CHECK_CUDA_ERROR(cudaFree(d_hist_storage));
        }
    }
};

typedef ::testing::Types<
                         HistogramTestType<int32_t, true>,
                         HistogramTestType<int64_t, true>,
                         HistogramTestType<int32_t, false>,
                         HistogramTestType<int64_t, false>> HistogramTestTypes;

class HistogramTestNames {
 public:
  template <typename T>
  static std::string GetName(int i) {
    std::string test_name = "HistogramTest_";
    if (std::is_same_v<typename T::KeyType, int32_t>) {
      test_name += std::string("Keys[int32]_");
    }
    if (std::is_same_v<typename T::KeyType, int64_t>) {
      test_name += std::string("Keys[int64]_");
    }
    if (T::runOnGPU) {
      test_name += std::string("GPU_");
    } else {
      test_name += std::string("CPU_");
    }
    test_name += ::testing::PrintToString(i);
    return test_name;
  }
};

TYPED_TEST_SUITE_P(HistogramTest);

TYPED_TEST_P(HistogramTest, HistogramTestSuite) {
    this->launchTest(17);
}

REGISTER_TYPED_TEST_SUITE_P(HistogramTest, HistogramTestSuite);

INSTANTIATE_TYPED_TEST_SUITE_P(Histogram,
                               HistogramTest,
                               HistogramTestTypes,
                               HistogramTestNames);
