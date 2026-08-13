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

template <typename ElemT, typename IndexT,
          bool FIXED_HOTNESS, PoolingType_t POOILING_TYPE>
struct EmbedTestTypeBackpropCombo {
  typedef ElemT ElemType;
  typedef IndexT IndexType;
  
  static bool constexpr FIXED_HOTNESS_FLAG = FIXED_HOTNESS;
  static PoolingType_t constexpr POOLING_TYPE = POOILING_TYPE;
};

// Testing only float so far to avoid accuracy issues with weight normalization 
typedef ::testing::Types<
                         // Concat
                         EmbedTestTypeBackpropCombo<float, int32_t, true, PoolingType_t::Concatenate>,
                         EmbedTestTypeBackpropCombo<float, int64_t, true, PoolingType_t::Concatenate>,
                         // SUM
                         EmbedTestTypeBackpropCombo<float, int32_t, true, PoolingType_t::Sum>,
                         EmbedTestTypeBackpropCombo<float, int64_t, true, PoolingType_t::Sum>,
                         // MEAN
                         EmbedTestTypeBackpropCombo<float, int32_t, true, PoolingType_t::Mean>,
                         EmbedTestTypeBackpropCombo<float, int64_t, true, PoolingType_t::Mean>,
                         // CSR 
                         EmbedTestTypeBackpropCombo<float, int32_t, false, PoolingType_t::Sum>,
                         EmbedTestTypeBackpropCombo<float, int64_t, false, PoolingType_t::Sum>,
                         EmbedTestTypeBackpropCombo<float, int32_t, false, PoolingType_t::Mean>,
                         EmbedTestTypeBackpropCombo<float, int64_t, false, PoolingType_t::Mean>,
                         // weighted
                         EmbedTestTypeBackpropCombo<float, int32_t, true, PoolingType_t::WeightedSum>,
                         EmbedTestTypeBackpropCombo<float, int64_t, true, PoolingType_t::WeightedSum>,
                         EmbedTestTypeBackpropCombo<float, int32_t, true, PoolingType_t::WeightedMean>,
                         EmbedTestTypeBackpropCombo<float, int64_t, true, PoolingType_t::WeightedMean>,
                         EmbedTestTypeBackpropCombo<float, int32_t, false, PoolingType_t::WeightedSum>,
                         EmbedTestTypeBackpropCombo<float, int64_t, false, PoolingType_t::WeightedSum>,
                         EmbedTestTypeBackpropCombo<float, int32_t, false, PoolingType_t::WeightedMean>,
                         EmbedTestTypeBackpropCombo<float, int64_t, false, PoolingType_t::WeightedMean>>
    EmbedTestBackpropTypes;

template <typename T>
class PoolingBackPropRefTest : public EmbeddingCacheRefTest<T> {
  public:

  using ElemType = typename EmbeddingCacheRefTest<T>::ElemType;
  using IndexType = typename EmbeddingCacheRefTest<T>::IndexType;
  using CacheType = nve::CacheSAHostModify<IndexType, IndexType>;
  using CacheDataType = typename CacheType::CacheData;

  private:

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

        // allocate and init weights
        m_weights = std::make_shared<TestBuffer<ElemType>>(this->m_num_keys * sizeof(ElemType));
        const bool is_weighted = is_weighted_pooling(T::POOLING_TYPE);
        if (is_weighted) {
            std::mt19937 genr(0X753812);
            std::uniform_real_distribution<float> dist_float(0.1f, 1.0f);
            std::bernoulli_distribution distrib(0.5);
            for (IndexType i=0 ; i < this->m_num_keys; i++) {
                m_weights->ph[i] = distrib(genr) ? ElemType(0.5f) : ElemType(0.25f);
            }
        } else {
            for (IndexType i=0 ; i < this->m_num_keys; i++) {
                m_weights->ph[i] = ElemType(1.0f);
            }
        }
        m_weights->HtoD(this->m_stream);

        uint32_t batch_ = this->m_batch;
        if (T::POOLING_TYPE == PoolingType_t::Concatenate) {
            batch_ *= this->m_hotness;
        }
        m_grads_in = std::make_shared<TestBuffer<ElemType>>(batch_ * this->m_num_elements * sizeof(ElemType));
        
        std::mt19937 genr(0X387512);
        std::uniform_int_distribution<int32_t> dist_nom(-8, 8);
        std::uniform_int_distribution<uint32_t> dist_denom(3, 9);
        for (uint32_t b = 0; b < batch_; b++) {
            for (uint32_t el = 0; el < this->m_num_elements; el++) {
                float nom = float(dist_nom(genr));
                float denom = float(1 << dist_denom(genr));
                m_grads_in->ph[b * this->m_num_elements + el] = nom / denom;
            }
        }
        m_grads_in->HtoD(this->m_stream);
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

    void ComputeRefResults(const ElemType* /*table*/,
                           const IndexType* indices,    
                           const std::vector<IndexType>& /*cache_data*/) {
        NVE_DEBUG_PRINTF_("compute pooling backprop reference results\n");

        m_ref_grads.clear();
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

            // Using constexpr if-else instead of switch to avoid Coverity false positive (PW.SWITCH_SELECTOR_EXPR_IS_CONSTANT)
            ElemType sum_weights = 1.0f;
            if constexpr (T::POOLING_TYPE == PoolingType_t::WeightedMean) {
                sum_weights = 0.f;
                for (IndexType h = 0; h < hotness; h++) {
                    sum_weights += ElemType(m_weights->ph[start_idx + h]);
                }
            } else if constexpr (T::POOLING_TYPE == PoolingType_t::Mean) {
                sum_weights = ElemType(hotness);
            }

            for (IndexType h = 0; h < hotness; h++) {
                auto key = indices[start_idx + h];
                if (m_ref_grads.count(key) == 0) {
                    ElemType* p = new ElemType[this->m_num_elements];
                    memset(p, 0, this->m_num_elements * sizeof(ElemType));
                    m_ref_grads[key] = p;
                }
                auto ref_grad = m_ref_grads[key];
                ElemType* input_grad = m_grads_in->ph + b * this->m_num_elements;

                if (T::POOLING_TYPE == PoolingType_t::Concatenate) {
                    input_grad = m_grads_in->ph + b * hotness * this->m_num_elements + h * this->m_num_elements;
                }

                for (uint32_t el = 0; el < this->m_num_elements; el++) {
                    ElemType el_cast = input_grad[el];
                    ElemType weight_cast = 1.0f;

                    if (is_weighted_pooling(T::POOLING_TYPE)) {
                        weight_cast = m_weights->ph[start_idx + h];
                    }
                    ref_grad[el] += el_cast * (weight_cast / sum_weights);
                }
            }
        }
    }

    void LaunchKernel(const ElemType* /*table*/, 
                      const IndexType* indices,
                      CacheDataType& /*cache_data*/) {
        NVE_DEBUG_PRINTF_("launch pooling backprop kernel %lu keys\n", static_cast<uint64_t>(this->m_num_keys));

        m_unique_keys = std::make_shared<TestBuffer<IndexType>>(this->m_num_keys * sizeof(IndexType));
        m_grads_out = std::make_shared<TestBuffer<ElemType>>(this->m_num_keys * this->m_num_elements * sizeof(ElemType));
    
        const int32_t MAX_RUN_SIZE = 32;
        m_split_grads = MAX_RUN_SIZE != -1;

        nve::GradientCalculator<IndexType, MAX_RUN_SIZE> pool_backprop(nve::GetDefaultAllocator());

        size_t tmp_mem_size_device, tmp_mem_size_host;
        size_t element_size = sizeof(ElemType);
        pool_backprop.get_alloc_requirements(this->m_num_keys, element_size, tmp_mem_size_device, tmp_mem_size_host);
        char* tmp_device_mem;
        CHECK_CUDA_ERROR(cudaMalloc(&tmp_device_mem, tmp_mem_size_device));
        char* tmp_host_mem;
        CHECK_CUDA_ERROR(cudaMallocHost(&tmp_host_mem, tmp_mem_size_host));
        pool_backprop.set_and_init_buffers(this->m_num_keys, element_size, tmp_device_mem, tmp_host_mem);

        PoolingType_t pooling_type = T::POOLING_TYPE;

        pool_backprop.template compute_gradients<ElemType, T::FIXED_HOTNESS_FLAG>(
            indices,
            T::FIXED_HOTNESS_FLAG ? nullptr : m_offsets->pd,
            m_grads_in->pd,
            m_weights->pd,
            m_grads_out->pd,
            m_unique_keys->pd,
            this->m_batch,
            this->m_num_keys,
            this->m_hotness,
            this->m_num_elements,
            pooling_type,
            this->m_stream);

        CHECK_CUDA_ERROR(cudaStreamSynchronize(this->m_stream));

        CHECK_CUDA_ERROR(cudaFree(tmp_device_mem));
        CHECK_CUDA_ERROR(cudaFreeHost(tmp_host_mem));
    }

    void CheckResult() {
        NVE_DEBUG_PRINTF_("check pooling backprop results\n");
        
        // check results
        m_unique_keys->DtoH(this->m_stream);
        m_grads_out->DtoH(this->m_stream);
        CHECK_CUDA_ERROR(cudaStreamSynchronize(this->m_stream));

        const float tolerance = (m_split_grads ||
                                (T::POOLING_TYPE == PoolingType_t::Mean) ||
                                (T::POOLING_TYPE == PoolingType_t::WeightedMean)) ? 1e-5f : 0;

        for (unsigned long i = 0; i < m_ref_grads.size(); i++) 
        {
            auto unq_key = m_unique_keys->ph[i];
            ElemType* kernel_ptr = m_grads_out->ph + i * this->m_num_elements;
            ElemType* ref_ptr = m_ref_grads[unq_key];
            EXPECT_TRUE(std::equal(ref_ptr, ref_ptr + this->m_num_elements, kernel_ptr, Near(tolerance)));
        }
    }

    std::vector<ElemType> m_ref_result;
    std::shared_ptr<TestBuffer<ElemType>> m_weights = nullptr;
    std::shared_ptr<TestBuffer<IndexType>> m_offsets = nullptr;
    std::shared_ptr<TestBuffer<ElemType>> m_grads_in = nullptr;
    std::shared_ptr<TestBuffer<ElemType>> m_grads_out = nullptr;
    std::shared_ptr<TestBuffer<IndexType>> m_unique_keys = nullptr;

    std::map<IndexType, ElemType*> m_ref_grads;

    bool m_split_grads = false;
};

TYPED_TEST_SUITE_P(PoolingBackPropRefTest);

TYPED_TEST_P(PoolingBackPropRefTest, TestPoolingBackpropAgainstRefCpu) {
    // Removed "nice" hotness and num elements values in favour of multiple pooling modes
    // Try multiples of 32 in case of test failures
    for (const auto batch : {171}) {
        for (const auto hotness : {61}) {
            for (const auto num_elements : {53, 128, 510}) {
                for(const auto allocOnHost: {true, false}){
                    NVE_DEBUG_PRINTF_("running test on batch %d hotness %d num_elements %d allocateDataOn: %s\n", batch, hotness, num_elements, allocOnHost?"host":"device");
                    this->LaunchTest(262144, num_elements, batch, hotness, allocOnHost);
                }
            }
        }
    }
}

REGISTER_TYPED_TEST_SUITE_P(PoolingBackPropRefTest, TestPoolingBackpropAgainstRefCpu);

class PoolingBackpropRefTestNames {
 public:
  template <typename T>
  static std::string GetName(int i) {
    return EmbeddingCacheRefTestNames::GetName<T>("PoolingBackpropRefTest_", i);
  }
};

INSTANTIATE_TYPED_TEST_SUITE_P(PoolingBackprop,    // TRTREC-44
                               PoolingBackPropRefTest,
                               EmbedTestBackpropTypes,
                               PoolingBackpropRefTestNames);

