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

template <typename T>
class EmbeddingCacheLookupRefTest : public EmbeddingCacheRefTest<T> {
  public:

  using ElemType = typename EmbeddingCacheRefTest<T>::ElemType;
  using IndexType = typename EmbeddingCacheRefTest<T>::IndexType;
  using AccumType = typename T::AccumType;
  using OutputType = typename T::OutputType;
  using CacheType = nve::CacheSAHostModify<IndexType, IndexType>;
  using CacheDataType = typename CacheType::CacheData;

  private:

    // Host-side float<->element conversions so the reference can model the kernel's final
    // Cast<AccumType, OutputType> regardless of whether the types are float or __half.
    static float to_float(float v) { return v; }
    static float to_float(__half v) { return __half2float(v); }
    static OutputType to_output(float v) {
        if constexpr (std::is_same_v<OutputType, __half>) return __float2half(v);
        else return static_cast<OutputType>(v);
    }

    void InitTestExtras() {
        if (T::FIXED_HOTNESS_FLAG) {
            this->m_num_keys = this->m_batch * this->m_hotness;
        } else {
            // allocate and init offsets
            m_offsets = std::make_shared<TestBuffer<IndexType>>((this->m_batch + 1) * sizeof(IndexType));
            std::mt19937 gen(0X475381);
            std::uniform_int_distribution<uint32_t> dist_offset(1, 31);

            IndexType curr_offset = 0;
            for (uint32_t b = 0; b < this->m_batch; b++) {
                m_offsets->ph[b] = curr_offset;
                curr_offset += dist_offset(gen);
            }
            this->m_num_keys = curr_offset;
            m_offsets->ph[this->m_batch] = curr_offset;
            m_offsets->HtoD(this->m_stream);
        }

        if (T::IS_WEIGHTED_FLAG) {
            // allocate and init weights
            m_weights = std::make_shared<TestBuffer<ElemType>>(this->m_num_keys * sizeof(ElemType));
            std::mt19937 genr(0X753812);
            std::uniform_real_distribution<float> dist_float(0.1f, 1.0f);
            std::bernoulli_distribution distrib(0.5);
            for (IndexType i=0 ; i < this->m_num_keys; i++) {
                m_weights->ph[i] = distrib(genr) ? ElemType(0.5f) : ElemType(0.25f);
            }

            m_weights->HtoD(this->m_stream);
        }
    }

    void AllocateKeys() {
        this->m_keys = std::make_shared<TestBuffer<IndexType>>(this->m_num_keys * sizeof(IndexType));
        const float alpha = 1.05f;

        const size_t seed = 283982;
        auto sg = getSampleGenerator<IndexType>(alpha, static_cast<IndexType>(this->m_num_rows),
                                                T::FIXED_HOTNESS_FLAG ? this->m_hotness : 32, seed);
        for (uint32_t b = 0; b < this->m_batch; b++)
        {
            auto sample = sg->getCategoryIndices();
            IndexType start_pos = T::FIXED_HOTNESS_FLAG ? b * this->m_hotness : this->m_offsets->ph[b];
            IndexType keys_to_copy = T::FIXED_HOTNESS_FLAG ? this->m_hotness :
                                                             (this->m_offsets->ph[b + 1] - this->m_offsets->ph[b]);
            std::copy(sample.begin(), sample.begin() + keys_to_copy, this->m_keys->ph + start_pos);
        }
        this->m_keys->HtoD(this->m_stream);
    }

    void ComputeRefResults(const ElemType* table,
                           const IndexType* indices,    
                           const std::vector<IndexType>& /*cache_data*/) {
        NVE_DEBUG_PRINTF_("compute lookup reference results\n");
        m_ref_result.resize(0);
        for (uint32_t b = 0; b < this->m_batch; b++) {
            IndexType hotness;
            IndexType start_idx;
            if (T::FIXED_HOTNESS_FLAG) {
                hotness = this->m_hotness;
                start_idx = b * hotness;
            } else {
                hotness = m_offsets->ph[b+1] - m_offsets->ph[b];
                start_idx = m_offsets->ph[b];
            }

            for (uint32_t el = 0; el < this->m_num_elements; el++) {
                AccumType acc = 0;
                AccumType weight_acc = 0;
                
                for (IndexType h = 0; h < hotness; h++) {
                    // load_indices=true resolves the key from the indices buffer; false reads the
                    // row positionally (sampleStart + h), matching the kernel's laneIdx logic.
                    IndexType row = T::LOAD_INDICES_FLAG ? indices[start_idx + h]
                                                         : static_cast<IndexType>(start_idx + h);
                    AccumType el_cast = table[row * this->m_num_elements + el];
                    if (T::IS_WEIGHTED_FLAG) {
                        AccumType weight_cast = m_weights->ph[start_idx + h];
                        acc += el_cast * weight_cast;
                        weight_acc += weight_cast;
                    } else {
                        acc += el_cast;
                    }
                }
                if (T::SUM_POOLING_FLAG) {
                    m_ref_result.push_back(to_float(acc));
                } else {
                    if (T::IS_WEIGHTED_FLAG) {
                        m_ref_result.push_back(to_float(weight_acc != AccumType(0) ? acc / weight_acc : AccumType(1)));
                    } else {
                        m_ref_result.push_back(to_float(acc / AccumType(hotness)));
                    }
                }
            }
        }
    }

    void LaunchKernel(const ElemType* table,
                      const IndexType* indices,
                      CacheDataType& cache_data) {
        // find_and_combine's load_indices=false path indexes rows positionally over [0, m_num_keys);
        // load_indices=true uses keys in [1, m_num_rows). Either way every accessed row must be < m_num_rows.
        ASSERT_LT(static_cast<uint64_t>(this->m_num_keys), this->m_num_rows)
            << "num_keys must be < table rows so positional (load_indices=false) reads stay in-bounds";
        m_result = std::make_shared<TestBuffer<OutputType>>(this->m_batch * this->m_hotness * this->m_num_elements * sizeof(OutputType));
        NVE_DEBUG_PRINTF_("launch lookup kernel\n");

        int8_t** tables_d;
        std::vector<const int8_t*> tables_h;
        tables_h.push_back(reinterpret_cast <const int8_t*>(table));
        CHECK_CUDA_ERROR(cudaMalloc(&tables_d, sizeof(int8_t*)));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(tables_d, tables_h.data(), sizeof(int8_t*), cudaMemcpyDefault, this->m_stream));

        SparseType_t hot_type = T::FIXED_HOTNESS_FLAG ? SparseType_t::Fixed : SparseType_t::CSR;
        PoolingType_t pooling_type = T::IS_WEIGHTED_FLAG
            ? (T::SUM_POOLING_FLAG ? PoolingType_t::WeightedSum : PoolingType_t::WeightedMean)
            : (T::SUM_POOLING_FLAG ? PoolingType_t::Sum        : PoolingType_t::Mean);
        call_find_and_combine_kernel<IndexType, CacheDataType>(
            static_cast<uint32_t>(this->m_num_keys), this->m_batch, reinterpret_cast<const int8_t*>(table), indices,
            m_offsets == nullptr ? nullptr : m_offsets->pd,
            m_weights == nullptr ? nullptr : m_weights->pd,
            this->m_hotness, cache_data, this->m_num_elements,
            hot_type, pooling_type, /*load_indices=*/T::LOAD_INDICES_FLAG,
            data_type<ElemType>(),    // element_type
            data_type<ElemType>(),    // weight_type
            data_type<AccumType>(),   // acc_type
            data_type<OutputType>(),  // output_type
            m_result->pd, this->m_stream);
        CHECK_CUDA_ERROR(cudaStreamSynchronize(this->m_stream));
        CHECK_CUDA_ERROR(cudaFree(tables_d));
    }

    void CheckResult() {
        NVE_DEBUG_PRINTF_("check lookup results\n");
        m_result->DtoH(this->m_stream);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Model the kernel's final Cast<AccumType, OutputType> by rounding the (float) reference
        // through OutputType before comparing, so a __half output is checked against a __half-rounded
        // reference. The table holds exact dyadic fractions, so float accumulation is order-exact and
        // the only precision loss is this final rounding, matched on both sides.
        const float tolerance = ((!T::IS_WEIGHTED_FLAG) && T::SUM_POOLING_FLAG) ? 0.f : 1e-7f;
        Near near(tolerance);
        bool all_near = true;
        for (size_t i = 0; i < m_ref_result.size(); ++i) {
            const OutputType expected = to_output(m_ref_result[i]);
            all_near = all_near && near(expected, m_result->ph[i]);
        }
        EXPECT_TRUE(all_near);
    }

    std::shared_ptr<TestBuffer<OutputType>> m_result = nullptr;
    std::vector<float> m_ref_result;
    std::shared_ptr<TestBuffer<ElemType>> m_weights = nullptr;
    std::shared_ptr<TestBuffer<IndexType>> m_offsets = nullptr;
};

TYPED_TEST_SUITE_P(EmbeddingCacheLookupRefTest);

TYPED_TEST_P(EmbeddingCacheLookupRefTest, TestPoolingAgainstRefCpu) {
    // Removed "nice" hotness and num elements values in favour of multiple pooling modes
    // Try multiples of 32 in case of test failures
    for (const auto batch : {17, 2048}) {
        for (const auto hotness : {59}) {
            for (const auto num_elements : {132, 30, 63}) {
                for(const auto allocOnHost: {true, false}){
                    NVE_DEBUG_PRINTF_("running test on batch %d hotness %d num_elements %d allocateDataOn: %s\n", batch, hotness, num_elements, allocOnHost?"host":"device");
                    this->LaunchTest(262144, num_elements, batch, hotness, allocOnHost);
                }
            }
        }
    }
}

REGISTER_TYPED_TEST_SUITE_P(EmbeddingCacheLookupRefTest, TestPoolingAgainstRefCpu);

template <typename ElemT, typename IndexT, typename AccumT,
          bool FIXED_HOTNESS, bool SUM_POOILING, bool IS_WEIGHTED, bool LOAD_INDICES = false,
          typename OutT = ElemT>
struct EmbedTestTypePoolingCombo {
  typedef ElemT ElemType;
  typedef IndexT IndexType;
  typedef AccumT AccumType;
  // Output element type written by find_and_combine. Defaults to ElemType (storage == output, the
  // common case); a distinct OutT exercises the OUTPUT_TYPE template parameter (e.g. float storage
  // with __half output, or __half storage with float output).
  typedef OutT OutputType;
  static bool constexpr FIXED_HOTNESS_FLAG = FIXED_HOTNESS;
  static bool constexpr SUM_POOLING_FLAG = SUM_POOILING;
  static bool constexpr IS_WEIGHTED_FLAG = IS_WEIGHTED;
  // false: the kernel indexes rows positionally (sampleStart+i, production/post-gather path);
  // true: the kernel reads the key value from the indices buffer (new indirect-gather feature).
  static bool constexpr LOAD_INDICES_FLAG = LOAD_INDICES;
};

typedef ::testing::Types<
                        // SUM
                         EmbedTestTypePoolingCombo<float, int32_t, float, true, true, false>,
                         EmbedTestTypePoolingCombo<float, int64_t, float, true, true, false>,
                         EmbedTestTypePoolingCombo<__half, int32_t, float, true, true, false>,
                         EmbedTestTypePoolingCombo<__half, int64_t, float, true, true, false>,
                         // MEAN
                         EmbedTestTypePoolingCombo<float, int32_t, float, true, false, false>,
                         EmbedTestTypePoolingCombo<float, int64_t, float, true, false, false>,
                         EmbedTestTypePoolingCombo<__half, int32_t, float, true, false, false>,
                         EmbedTestTypePoolingCombo<__half, int64_t, float, true, false, false>,
                         // CSR
                         EmbedTestTypePoolingCombo<float, int32_t, float, false, true, false>,
                         EmbedTestTypePoolingCombo<float, int64_t, float, false, true, false>,
                         EmbedTestTypePoolingCombo<__half, int32_t, float, false, true, false>,
                         EmbedTestTypePoolingCombo<__half, int64_t, float, false, true, false>,
                         EmbedTestTypePoolingCombo<float, int32_t, float, false, false, false>,
                         EmbedTestTypePoolingCombo<float, int64_t, float, false, false, false>,
                         EmbedTestTypePoolingCombo<__half, int32_t, float, false, false, false>,
                         EmbedTestTypePoolingCombo<__half, int64_t, float, false, false, false>,
                         // weighted
                         EmbedTestTypePoolingCombo<float, int32_t, float, true, true, true>,
                         EmbedTestTypePoolingCombo<float, int64_t, float, true, true, true>,
                         EmbedTestTypePoolingCombo<__half, int32_t, float, true, true, true>,
                         EmbedTestTypePoolingCombo<__half, int64_t, float, true, true, true>,
                         EmbedTestTypePoolingCombo<float, int32_t, float, true, false, true>,
                         EmbedTestTypePoolingCombo<float, int64_t, float, true, false, true>,
                         EmbedTestTypePoolingCombo<__half, int32_t, float, true, false, true>,
                         EmbedTestTypePoolingCombo<__half, int64_t, float, true, false, true>,
                         EmbedTestTypePoolingCombo<float, int32_t, float, false, true, true>,
                         EmbedTestTypePoolingCombo<float, int64_t, float, false, true, true>,
                         EmbedTestTypePoolingCombo<__half, int32_t, float, false, true, true>,
                         EmbedTestTypePoolingCombo<__half, int64_t, float, false, true, true>,
                         EmbedTestTypePoolingCombo<float, int32_t, float, false, false, true>,
                         EmbedTestTypePoolingCombo<float, int64_t, float, false, false, true>,
                         EmbedTestTypePoolingCombo<__half, int32_t, float, false, false, true>,
                         EmbedTestTypePoolingCombo<__half, int64_t, float, false, false, true>,
                         // load_indices=true (indirect key-gather) representative subset:
                         EmbedTestTypePoolingCombo<float, int32_t, float, true,  true,  false, /*LOAD_INDICES=*/true>,   // Fixed sum
                         EmbedTestTypePoolingCombo<float, int32_t, float, true,  false, false, /*LOAD_INDICES=*/true>,   // Fixed mean
                         EmbedTestTypePoolingCombo<float, int32_t, float, false, true,  false, /*LOAD_INDICES=*/true>,   // CSR sum
                         EmbedTestTypePoolingCombo<float, int32_t, float, false, false, false, /*LOAD_INDICES=*/true>,   // CSR mean
                         // Mismatched OUTPUT_TYPE (storage/output differ): exercise the OUTPUT_TYPE
                         // template parameter of find_and_combine. Trailing arg is the output type.
                         // float storage/acc -> __half output:
                         EmbedTestTypePoolingCombo<float, int32_t, float, true,  true,  false, false, __half>,   // Fixed sum
                         EmbedTestTypePoolingCombo<float, int32_t, float, true,  false, false, false, __half>,   // Fixed mean
                         EmbedTestTypePoolingCombo<float, int64_t, float, false, true,  false, false, __half>,   // CSR sum
                         EmbedTestTypePoolingCombo<float, int32_t, float, true,  true,  true,  false, __half>,   // Fixed weighted sum
                         // __half storage, float acc -> float output:
                         EmbedTestTypePoolingCombo<__half, int32_t, float, true,  true,  false, false, float>,   // Fixed sum
                         EmbedTestTypePoolingCombo<__half, int64_t, float, true,  false, false, false, float>,   // Fixed mean
                         EmbedTestTypePoolingCombo<__half, int32_t, float, false, false, false, false, float>,  // CSR mean
                         EmbedTestTypePoolingCombo<__half, int64_t, float, true,  false, true,  false, float>>   // Fixed weighted mean
    EmbedTestPoolingTypes;

class EmbeddingCacheLookupRefTestNames {
 public:
  template <typename T>
  static std::string GetName(int i) {
    return EmbeddingCacheRefTestNames::GetName<T>("EmbeddingCacheLookupRefTest_", i);
  }
};

INSTANTIATE_TYPED_TEST_SUITE_P(EmbeddingLookup,    // TRTREC-44
                               EmbeddingCacheLookupRefTest,
                               EmbedTestPoolingTypes,
                               EmbeddingCacheLookupRefTestNames);
