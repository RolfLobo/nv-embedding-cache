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

#include "gtest/gtest.h"
#include <datagen.h>
#include <memory>
#include <algorithm>
#include <thread>
#include <numeric>
#include <cmath>
#include <type_traits>
#include "embedding_cache_combined.cuh"
#include "cuda_ops/scatter.cuh"
#include "cuda_ops/update_accumulate.cuh"
#include "cuda_ops/cuda_utils.cuh"
#include "cuda_ops/dedup_grads_kernel.cuh"
#include "cuda_ops/find_and_combine_kernel.cuh"
#include "cuda_ops/find_and_dequant.cuh"
#include "cuda_ops/find_and_pool.cuh"
#include "cuda_ops/gradient_calculator.cuh"
#include "cuda_ops/gather_keys_data_ptrs.cuh"
#include "../common/test_buffer.h"
#include <default_allocator.hpp>

struct Near {
  float tolerance_;

  explicit Near(const float tol) : tolerance_(tol) {}

  __host__ __device__ bool operator()(const float& a, const float& b) const {
      return fabsf(a - b) <= tolerance_;
  }

  __host__ __device__ bool operator()(const __half& a, const __half& b) const {
      return fabsf(__half2float(a) - __half2float(b)) <= tolerance_;
  }
};

TEST(FindAndPoolValidation, AcceptsSupportedDataTypeCombinations) {
  const DataType_t value_output_types[][2] = {
      {DataType_t::Float32, DataType_t::Float32},
      {DataType_t::Float16, DataType_t::Float16},
      {DataType_t::QInt8RowwiseF32, DataType_t::Float32},
      {DataType_t::QUint8RowwiseF32, DataType_t::Float32},
      {DataType_t::QInt8RowwiseF16, DataType_t::Float16},
      {DataType_t::QUint8RowwiseF16, DataType_t::Float16},
  };
  const DataType_t acc_weight_types[][2] = {
      {DataType_t::Float32, DataType_t::Float32},
      {DataType_t::Float32, DataType_t::Float16},
      {DataType_t::Float16, DataType_t::Float16},
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
class EmbeddingCacheRefTest : public ::testing::Test {
  public:
    typedef typename T::ElemType ElemType;
    typedef typename T::IndexType IndexType;

    using CacheType = nve::CacheSAHostModify<IndexType, IndexType>;
    using CacheDataType = typename CacheType::CacheData;

    EmbeddingCacheRefTest(): m_cache_allocator(nve::DefaultAllocator::DEFAULT_HOST_ALLOC_THRESHOLD) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&m_stream));
    }

    ~EmbeddingCacheRefTest() {
        CHECK_CUDA_ERROR(m_cache_ptr->lookup_context_destroy(m_handle_lookup));
        CHECK_CUDA_ERROR(m_cache_ptr->modify_context_destroy(m_handle_modify));
    }

    void LaunchTest(uint64_t num_rows, uint32_t num_elements, uint32_t batch, uint32_t hotness, bool allocDataOnHost, int64_t kernel_param = 0) {
        m_num_rows = num_rows;
        m_num_elements = num_elements;
        m_batch = batch;
        m_hotness = hotness;
        m_num_keys = static_cast<IndexType>(m_batch) * static_cast<IndexType>(m_hotness);
        m_allocDataOnHost = allocDataOnHost;
        m_kernel_param = kernel_param;
        AllocateTable();
        InitCache();
        InitTestExtras();
        AllocateKeys();
        CacheDataType cache_data = m_cache_ptr->get_cache_data(m_handle_lookup);
        std::vector<IndexType> cached_indices = 
            ComputeCachedIndices(cache_data.num_sets, CacheType::NUM_WAYS);
        PopulateCache(cached_indices);
        ComputeRefResults(m_table->ph, m_keys->ph, cached_indices);
        LaunchKernel(m_table->pd, m_keys->pd, cache_data);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CheckResult();
    }

    cudaStream_t m_stream;
    uint64_t m_num_rows{0};
    uint32_t m_num_elements{0}; 
    uint32_t m_batch{0};
    uint32_t m_hotness{0};
    IndexType m_num_keys{0};
    bool m_allocDataOnHost{false};
    int64_t m_kernel_param{0};
  protected:
  
    ElemType* syncTable() {
      m_table->DtoH(m_stream);
      return m_table->ph;
    }

    void GetCacheContent(ElemType* d_cache_content, std::shared_ptr<TestBuffer<IndexType>> keys_in_c, size_t& num_keys) 
    {
        
        m_cache_ptr->get_keys_stored_in_cache(m_handle_lookup, keys_in_c->ph, num_keys);
        keys_in_c->HtoD(m_stream);
        int64_t num_hitmask_elems = (num_keys + 31) / 32;
        auto hitmask = std::make_shared<TestBuffer<uint32_t>>(num_hitmask_elems * sizeof(uint32_t));
        CHECK_CUDA_ERROR(cudaMemset(hitmask->pd, 0, num_hitmask_elems * sizeof(uint32_t)));

        NVE_DEBUG_PRINTF_("launch lookup with hitmask kernel\n");

        m_cache_ptr->lookup(m_handle_lookup, keys_in_c->pd, num_keys, (int8_t*)d_cache_content, (uint64_t*)hitmask->pd, 0, m_num_elements*sizeof(ElemType), m_stream);
    }

    std::map<IndexType, std::vector<ElemType>> GetCacheContent()
    {
        auto num_vec_in_c = m_cache_ptr->get_max_num_embedding_vectors_in_cache();
        auto key_in_c = std::make_shared<TestBuffer<IndexType>>(sizeof(IndexType)*num_vec_in_c);
        auto cache_content = std::make_shared<TestBuffer<ElemType>>(sizeof(ElemType)*this->m_num_elements*num_vec_in_c);
        size_t num_keys_in_c = 0;
        this->GetCacheContent(cache_content->pd, key_in_c, num_keys_in_c);
        cache_content->DtoH(this->m_stream);
        CHECK_CUDA_ERROR(cudaStreamSynchronize(this->m_stream));
        std::map<IndexType, std::vector<ElemType>> mock_cache;
        for (size_t i = 0; i < num_keys_in_c; i++)
        {
            auto key = key_in_c->ph[i];
            mock_cache[key].resize(this->m_num_elements);
            memcpy(mock_cache[key].data(), cache_content->ph + i*this->m_num_elements, this->m_num_elements* sizeof(ElemType));
        }
        return mock_cache;
    }

    std::shared_ptr<TestBuffer<IndexType>> m_keys = nullptr;

  private:
    virtual void ComputeRefResults(const ElemType* /*table*/,
                                   const IndexType* /*indices*/,
                                   const std::vector<IndexType>& /*cache_data*/) {
        NVE_DEBUG_PRINTF_("compute baseline reference results\n");
    }

    virtual void LaunchKernel(const ElemType* /*table*/, 
                              const IndexType* /*indices*/,
                              CacheDataType& /*cache_data*/) {
    }

    virtual void CheckResult() {
    }
  
    virtual void InitTestExtras() {
    }

    void InitCache() {
        const float cache_ratio = 0.15f;
        const uint32_t num_rows_in_cache = static_cast<uint32_t>(cache_ratio * static_cast<float>(this->m_num_rows));

        typename CacheType::CacheConfig cfg;
        cfg.embed_width_in_bytes = this->m_num_elements * sizeof(ElemType);

        NVE_DEBUG_PRINTF_("cache of %d entries\n", int(cache_ratio * float(this->m_num_rows)));
        cfg.cache_sz_in_bytes = num_rows_in_cache * cfg.embed_width_in_bytes;
        cfg.num_tables = 1;
        cfg.allocate_data_on_host = m_allocDataOnHost;

        m_cache_ptr = std::make_shared<CacheType>(&m_cache_allocator, &m_cache_logger, cfg);
        m_cache_ptr->init();

        m_cache_ptr->lookup_context_create(m_handle_lookup, nullptr, 0);
    }

    void AllocateTable() {
        m_table = std::make_shared<TestBuffer<ElemType>>(this->m_num_rows * this->m_num_elements * sizeof(ElemType));
        std::mt19937 gen(0X814753);
        // init to [a / 2^b] when a and b are in specific range,
        // to minimize arithmetic error in half when doing weigthed math
        // inputs in the range of [-1.0f, 1.0f] accumulate larger errors
        // (tolerance of 1e-3f is required to pass the tests)
        std::uniform_int_distribution<int32_t> dist_nom(-8, 8);
        std::uniform_int_distribution<uint32_t> dist_denom(3, 9);

        for (uint64_t i = 0; i < this->m_num_rows; i++)
        {
            ElemType* curr_row = reinterpret_cast<ElemType*>(m_table->ph + i * this->m_num_elements);
            for (uint64_t j = 0; j < this->m_num_elements; j++)
            {
                float nom = float(dist_nom(gen));
                float denom = float(1 << dist_denom(gen));
                curr_row[j] = nom / denom;
            }
        }

        m_table->HtoD(m_stream);
    }

    virtual void AllocateKeys() {
        m_keys = std::make_shared<TestBuffer<IndexType>>(m_num_keys * sizeof(IndexType));
        const float alpha = 1.05f;

        auto sg = getSampleGenerator<IndexType>(alpha, static_cast<IndexType>(this->m_num_rows), this->m_hotness, 283982);
        for (uint32_t b = 0; b < this->m_batch; b++)
        {
            auto sample = sg->getCategoryIndices();
            std::copy(sample.begin(), sample.end(), m_keys->ph + b*this->m_hotness);
        }
        m_keys->HtoD(m_stream);
    }

    std::vector<IndexType> ComputeCachedIndices(const int num_sets, const int num_ways) {
        std::set<IndexType> cached_indices;
        std::vector<int> counters(num_sets, 0);

        uint64_t cache_capacity = static_cast<uint64_t>(static_cast<double>(this->m_num_rows) * 0.15);

        while (cache_capacity > static_cast<uint64_t>(this->m_num_keys)) {
            cache_capacity /= 2;
        }

        uint32_t step = static_cast<uint32_t>(this->m_num_keys / cache_capacity);
        EXPECT_TRUE(step > 0); // batch_size * hotness > cache_capcitiy 
        // for now: random indices
        for (IndexType i=0; i < this->m_num_keys; i+=step) {
            IndexType idx = m_keys->ph[i];
            if (++counters[idx % num_sets] <= num_ways) {
                cached_indices.insert(idx);
            }
        }

        std::vector<IndexType> idx_vec;
        for (typename std::set<IndexType>::iterator itr = cached_indices.begin(); itr != cached_indices.end(); ++itr) {
            idx_vec.push_back(*itr);
        }
        return idx_vec;
    }

    void PopulateCache(const std::vector<IndexType>& cached_indices) {
        NVE_DEBUG_PRINTF_("inserting %lu embeds to cache\n", cached_indices.size());

        std::vector<float> priorities(cached_indices.size(), 1.0f);
        m_cache_ptr->modify_context_create(m_handle_modify, static_cast<uint32_t>(cached_indices.size()));

        nve::DefaultHistogram<IndexType> hist(cached_indices.data(), cached_indices.size(),
                                              reinterpret_cast<const int8_t*>(m_table->ph),
                                              this->m_num_elements * sizeof(ElemType), true);
        cudaEvent_t wait;
        CHECK_CUDA_ERROR(cudaEventCreate(&wait));
        nve::DefaultECEvent ec_event(std::vector<cudaStream_t>{});

        m_cache_ptr->insert(
            m_handle_modify, hist.get_keys(), hist.get_priority(), hist.get_data(), hist.get_num_bins(), 0, &ec_event, m_stream);

        CHECK_CUDA_ERROR(cudaEventRecord(wait));
        CHECK_CUDA_ERROR(cudaEventSynchronize(wait));

    }

    std::shared_ptr<TestBuffer<ElemType>> m_table = nullptr;

    nve::DefaultAllocator m_cache_allocator;
    nve::Logger m_cache_logger;
    nve::PerformanceMetric m_miss_count;
    std::shared_ptr<CacheType> m_cache_ptr;
    nve::LookupContextHandle m_handle_lookup;
    nve::ModifyContextHandle m_handle_modify;
};

template <typename T>
class EmbeddingCacheLookupRefTest : public EmbeddingCacheRefTest<T> {
  public:

  using ElemType = typename EmbeddingCacheRefTest<T>::ElemType;
  using IndexType = typename EmbeddingCacheRefTest<T>::IndexType;
  using AccumType = typename T::AccumType;
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
                    m_ref_result.push_back(acc);
                } else {
                    if (T::IS_WEIGHTED_FLAG) {
                        m_ref_result.push_back(weight_acc != AccumType(0) ? acc / weight_acc : AccumType(1));
                    } else {
                        m_ref_result.push_back(acc / AccumType(hotness));
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
        m_result = std::make_shared<TestBuffer<ElemType>>(this->m_batch * this->m_hotness * this->m_num_elements * sizeof(ElemType));
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
        callFindAndCombineKernel<IndexType, CacheDataType>(
            static_cast<uint32_t>(this->m_num_keys), this->m_batch, reinterpret_cast<const int8_t*>(table), indices,
            m_offsets == nullptr ? nullptr : m_offsets->pd,
            m_weights == nullptr ? nullptr : m_weights->pd,
            this->m_hotness, cache_data, this->m_num_elements,
            hot_type, pooling_type, /*load_indices=*/T::LOAD_INDICES_FLAG,
            data_type<ElemType>(),   // element_type
            data_type<ElemType>(),   // weight_type
            data_type<AccumType>(),  // acc_type
            data_type<ElemType>(),   // output_type
            m_result->pd, this->m_stream);
        CHECK_CUDA_ERROR(cudaStreamSynchronize(this->m_stream));
        CHECK_CUDA_ERROR(cudaFree(tables_d));
    }

    void CheckResult() {
        NVE_DEBUG_PRINTF_("check lookup results\n");
        m_result->DtoH(this->m_stream);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        if ((!T::IS_WEIGHTED_FLAG) && T::SUM_POOLING_FLAG) {
            const float tolerance = 0;
            EXPECT_TRUE(std::equal(m_ref_result.begin(), m_ref_result.end(), m_result->ph, Near(tolerance)));
        } else {
            const float tolerance = 1e-7f;
            EXPECT_TRUE(std::equal(m_ref_result.begin(), m_ref_result.end(), m_result->ph, Near(tolerance)));
        }
    }

    std::shared_ptr<TestBuffer<ElemType>> m_result = nullptr;
    std::vector<ElemType> m_ref_result;
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

// ---------------------------------------------------------------------------
// Rowwise-quantized FindAndCombine tests (QInt8RowwiseF32 / QUint8RowwiseF32).
//
// A quantized embedding row is stored as:
//   [ StorageType q[num_elements] ][ float scale ][ float offset ]
// and the kernel reconstructs each element as float(q) * scale + offset
// (see kernels_common.cuh). This fixture builds such rows, populates the cache,
// computes a CPU reference by dequantizing with the identical formula and then
// QInt8RowwiseF32 uses symmetric quantization (no stored offset,
// scale = max|v|/127, signed q in [-127,127]); QUint8RowwiseF32 uses
// min/max affine quantization (offset = min, scale = (max-min)/255, unsigned q
// in [0,255]). This fixture is standalone (the float/half fixture above assumes
// one ElemType for storage+output, which a quantized row breaks).
// pooling/dequant arithmetic is under test, not quantization fidelity).
//
// QInt8RowwiseF32 uses symmetric-around-mean quantization (offset = row mean,
// scale = max|v-mean|/127, signed q in [-127,127]); QUint8RowwiseF32 uses
// min/max affine quantization (offset = min, scale = (max-min)/255, unsigned q
// in [0,255]). This fixture is standalone (the float/half fixture above assumes
// one ElemType for storage+output, which a quantized row breaks).
// ---------------------------------------------------------------------------
template <DataType_t VALUE_TYPE_ID_, typename StorageT, typename IndexT,
          bool FIXED_HOTNESS, bool SUM_POOLING, bool IS_WEIGHTED, bool LOAD_INDICES = false>
struct EmbedTestTypeQuantPoolingCombo {
  static constexpr DataType_t VALUE_TYPE_ID = VALUE_TYPE_ID_;
  using StorageType = StorageT;
  using IndexType = IndexT;
  static constexpr bool FIXED_HOTNESS_FLAG = FIXED_HOTNESS;
  static constexpr bool SUM_POOLING_FLAG = SUM_POOLING;
  static constexpr bool IS_WEIGHTED_FLAG = IS_WEIGHTED;
  // false: positional row indexing (sampleStart+i); true: read key from indices buffer.
  static constexpr bool LOAD_INDICES_FLAG = LOAD_INDICES;
};

template <typename T>
class EmbeddingCacheQuantLookupRefTest : public ::testing::Test {
 public:
  using StorageType = typename T::StorageType;
  using IndexType = typename T::IndexType;
  using CacheType = nve::CacheSAHostModify<IndexType, IndexType>;
  using CacheDataType = typename CacheType::CacheData;
  static constexpr DataType_t VALUE_TYPE_ID = T::VALUE_TYPE_ID;
  // ParamType is the scale/offset and output type (float for *F32, __half for *F16).
  using ParamType = typename QuantizationHelper<VALUE_TYPE_ID>::ParamType;
  static constexpr bool HAS_OFFSET = QuantizationHelper<VALUE_TYPE_ID>::has_offset;

  static ParamType from_float(float v) {
    if constexpr (std::is_same_v<ParamType, __half>) return __float2half(v);
    else return static_cast<ParamType>(v);
  }

  EmbeddingCacheQuantLookupRefTest()
      : m_cache_allocator(nve::DefaultAllocator::DEFAULT_HOST_ALLOC_THRESHOLD) {
    CHECK_CUDA_ERROR(cudaStreamCreate(&m_stream));
  }
  ~EmbeddingCacheQuantLookupRefTest() {
    if (m_cache_ptr) {
      CHECK_CUDA_ERROR(m_cache_ptr->lookup_context_destroy(m_handle_lookup));
      CHECK_CUDA_ERROR(m_cache_ptr->modify_context_destroy(m_handle_modify));
    }
  }

  void LaunchTest(uint64_t num_rows, uint32_t num_elements, uint32_t batch, uint32_t hotness,
                  bool allocOnHost) {
    m_num_rows = num_rows;
    m_num_elements = num_elements;
    m_batch = batch;
    m_hotness = hotness;
    m_allocOnHost = allocOnHost;
    m_row_bytes = m_num_elements * sizeof(StorageType)
                + (HAS_OFFSET ? 2 : 1) * sizeof(ParamType);  // q + scale [+ offset]
    // Round the row stride up to 4 bytes so every row base stays aligned for the Vec4 (char4)
    // load path; scale/offset stay at offset num_elements*element_size, padding goes after them.
    m_row_bytes = (m_row_bytes + 3u) & ~static_cast<uint64_t>(3u);
    AllocateTable();
    InitCache();
    InitExtras();  // offsets + weights, finalizes m_num_keys (CSR sets it from the offsets prefix sum)
    // The load_indices=false path indexes rows positionally over [0, m_num_keys); load_indices=true
    // uses keys in [1, m_num_rows). Either way every accessed row must be < m_num_rows.
    ASSERT_LT(static_cast<uint64_t>(m_num_keys), m_num_rows)
        << "num_keys must be < table rows so positional (load_indices=false) reads stay in-bounds";
    AllocateKeys();
    CacheDataType cache_data = m_cache_ptr->get_cache_data(m_handle_lookup);
    std::vector<IndexType> cached = ComputeCachedIndices(cache_data.num_sets, CacheType::NUM_WAYS);
    PopulateCache(cached);
    ComputeRefResults();
    LaunchKernel(cache_data);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CheckResult();
  }

 private:
  // Quantize one row of floats into StorageType q[], producing scale/offset such that
  // float(q[j]) * scale + offset reconstructs v[j].
  void Quantize(const std::vector<float>& v, StorageType* q, ParamType& scale_out, ParamType& offset_out) {
    float scale = 1.0f, offset = 0.0f;
    if constexpr (VALUE_TYPE_ID == DataType_t::QInt8RowwiseF32 ||
                  VALUE_TYPE_ID == DataType_t::QInt8RowwiseF16) {
      // Symmetric. F32 keeps a per-row mean offset; F16 has no offset (pure scale).
      if constexpr (HAS_OFFSET) {
        double sum = 0.0;
        for (float x : v) sum += x;
        offset = static_cast<float>(sum / static_cast<double>(v.size()));
      }
      float amax = 0.0f;
      for (float x : v) amax = std::max(amax, std::fabs(x - offset));
      scale = amax > 0.0f ? amax / 127.0f : 1.0f;
      for (size_t j = 0; j < v.size(); ++j) {
        long qi = std::lround((v[j] - offset) / scale);
        qi = std::min<long>(127, std::max<long>(-127, qi));
        q[j] = static_cast<StorageType>(qi);
      }
    } else {  // QUint8RowwiseF32 : min/max affine
      float lo = v[0], hi = v[0];
      for (float x : v) { lo = std::min(lo, x); hi = std::max(hi, x); }
      scale = hi > lo ? (hi - lo) / 255.0f : 1.0f;
      offset = lo;
      for (size_t j = 0; j < v.size(); ++j) {
        long qi = std::lround((v[j] - offset) / scale);
        qi = std::min<long>(255, std::max<long>(0, qi));
        q[j] = static_cast<StorageType>(qi);
      }
    }
    scale_out = from_float(scale);
    offset_out = from_float(offset);
  }

  // Dequantize one element, mirroring the kernel's per-type arithmetic so the only divergence is
  // float accumulation/rounding (kept inside the tolerance).
  float Dequantize(StorageType q, ParamType scale, ParamType offset) const {
    if constexpr (std::is_same_v<ParamType, __half>) {
      float prod = static_cast<float>(q) * __half2float(scale);
      if constexpr (HAS_OFFSET) prod += __half2float(offset);
      return __half2float(__float2half(prod));  // kernel stores the dequantized value as __half
    } else {
      float r = static_cast<float>(q) * scale;
      if constexpr (HAS_OFFSET) r += offset;
      return r;
    }
  }

  void AllocateTable() {
    m_table = std::make_shared<TestBuffer<int8_t>>(m_num_rows * m_row_bytes);
    std::mt19937 gen(0X814753);
    // Values in [-1, 1] expressed as a / 2^b to keep the float accumulation error small.
    std::uniform_int_distribution<int32_t> dist_nom(-8, 8);
    std::uniform_int_distribution<uint32_t> dist_denom(3, 9);

    std::vector<float> v(m_num_elements);
    for (uint64_t i = 0; i < m_num_rows; i++) {
      for (uint32_t j = 0; j < m_num_elements; j++) {
        v[j] = float(dist_nom(gen)) / float(1 << dist_denom(gen));
      }
      int8_t* row = m_table->ph + i * m_row_bytes;
      ParamType scale{}, offset{};
      Quantize(v, reinterpret_cast<StorageType*>(row), scale, offset);
      int8_t* meta = row + m_num_elements * sizeof(StorageType);
      memcpy(meta, static_cast<const void*>(&scale), sizeof(ParamType));
      if (HAS_OFFSET) memcpy(meta + sizeof(ParamType), static_cast<const void*>(&offset), sizeof(ParamType));
    }
    m_table->HtoD(m_stream);
  }

  void InitCache() {
    const float cache_ratio = 0.15f;
    const uint32_t num_rows_in_cache = static_cast<uint32_t>(cache_ratio * static_cast<float>(m_num_rows));

    typename CacheType::CacheConfig cfg;
    cfg.embed_width_in_bytes = m_row_bytes;
    cfg.cache_sz_in_bytes = num_rows_in_cache * cfg.embed_width_in_bytes;
    cfg.num_tables = 1;
    cfg.allocate_data_on_host = m_allocOnHost;

    m_cache_ptr = std::make_shared<CacheType>(&m_cache_allocator, &m_cache_logger, cfg);
    m_cache_ptr->init();
    m_cache_ptr->lookup_context_create(m_handle_lookup, nullptr, 0);
  }

  void InitExtras() {
    if (T::FIXED_HOTNESS_FLAG) {
      m_num_keys = m_batch * m_hotness;
    } else {
      m_offsets = std::make_shared<TestBuffer<IndexType>>((m_batch + 1) * sizeof(IndexType));
      std::mt19937 gen(0X475381);
      std::uniform_int_distribution<uint32_t> dist_offset(1, 31);
      IndexType curr_offset = 0;
      for (uint32_t b = 0; b < m_batch; b++) {
        m_offsets->ph[b] = curr_offset;
        curr_offset += dist_offset(gen);
      }
      m_num_keys = curr_offset;
      m_offsets->ph[m_batch] = curr_offset;
      m_offsets->HtoD(m_stream);
    }

    if (T::IS_WEIGHTED_FLAG) {
      m_weights = std::make_shared<TestBuffer<ParamType>>(m_num_keys * sizeof(ParamType));
      std::mt19937 genr(0X753812);
      std::bernoulli_distribution distrib(0.5);
      for (IndexType i = 0; i < m_num_keys; i++) {
        m_weights->ph[i] = from_float(distrib(genr) ? 0.5f : 0.25f);
      }
      m_weights->HtoD(m_stream);
    }
  }

  void AllocateKeys() {
    m_keys = std::make_shared<TestBuffer<IndexType>>(m_num_keys * sizeof(IndexType));
    const float alpha = 1.05f;
    const size_t seed = 283982;
    auto sg = getSampleGenerator<IndexType>(alpha, static_cast<IndexType>(m_num_rows),
                                            T::FIXED_HOTNESS_FLAG ? m_hotness : 32, seed);
    for (uint32_t b = 0; b < m_batch; b++) {
      auto sample = sg->getCategoryIndices();
      IndexType start_pos = T::FIXED_HOTNESS_FLAG ? b * m_hotness : m_offsets->ph[b];
      IndexType keys_to_copy = T::FIXED_HOTNESS_FLAG ? m_hotness
                                                     : (m_offsets->ph[b + 1] - m_offsets->ph[b]);
      std::copy(sample.begin(), sample.begin() + keys_to_copy, m_keys->ph + start_pos);
    }
    m_keys->HtoD(m_stream);
  }

  std::vector<IndexType> ComputeCachedIndices(const int num_sets, const int num_ways) {
    std::set<IndexType> cached_indices;
    std::vector<int> counters(num_sets, 0);
    uint64_t cache_capacity = static_cast<uint64_t>(static_cast<double>(m_num_rows) * 0.15);
    while (cache_capacity > static_cast<uint64_t>(m_num_keys)) {
      cache_capacity /= 2;
    }
    uint32_t step = static_cast<uint32_t>(m_num_keys / cache_capacity);
    EXPECT_TRUE(step > 0);
    for (IndexType i = 0; i < m_num_keys; i += step) {
      IndexType idx = m_keys->ph[i];
      if (++counters[idx % num_sets] <= num_ways) {
        cached_indices.insert(idx);
      }
    }
    return std::vector<IndexType>(cached_indices.begin(), cached_indices.end());
  }

  void PopulateCache(const std::vector<IndexType>& cached_indices) {
    m_cache_ptr->modify_context_create(m_handle_modify, static_cast<uint32_t>(cached_indices.size()));
    nve::DefaultHistogram<IndexType> hist(cached_indices.data(), cached_indices.size(),
                                          m_table->ph, m_row_bytes, true);
    cudaEvent_t wait;
    CHECK_CUDA_ERROR(cudaEventCreate(&wait));
    nve::DefaultECEvent ec_event(std::vector<cudaStream_t>{});
    m_cache_ptr->insert(m_handle_modify, hist.get_keys(), hist.get_priority(), hist.get_data(),
                        hist.get_num_bins(), 0, &ec_event, m_stream);
    CHECK_CUDA_ERROR(cudaEventRecord(wait));
    CHECK_CUDA_ERROR(cudaEventSynchronize(wait));
  }

  void ComputeRefResults() {
    m_ref_result.assign(m_batch * m_num_elements, 0.0f);
    for (uint32_t b = 0; b < m_batch; b++) {
      IndexType hotness = T::FIXED_HOTNESS_FLAG ? m_hotness : (m_offsets->ph[b + 1] - m_offsets->ph[b]);
      IndexType start_idx = T::FIXED_HOTNESS_FLAG ? b * m_hotness : m_offsets->ph[b];
      for (uint32_t el = 0; el < m_num_elements; el++) {
        float acc = 0.0f;
        float weight_acc = 0.0f;
        for (IndexType h = 0; h < hotness; h++) {
          // load_indices=true resolves the key from the indices buffer; false reads the row
          // positionally (sampleStart + h), matching the kernel's laneIdx logic.
          IndexType idx = T::LOAD_INDICES_FLAG ? m_keys->ph[start_idx + h]
                                               : static_cast<IndexType>(start_idx + h);
          const int8_t* row = m_table->ph + static_cast<uint64_t>(idx) * m_row_bytes;
          const StorageType* q = reinterpret_cast<const StorageType*>(row);
          const int8_t* meta = row + m_num_elements * sizeof(StorageType);
          ParamType scale{}, offset{};
          memcpy(static_cast<void*>(&scale), meta, sizeof(ParamType));
          if (HAS_OFFSET) memcpy(static_cast<void*>(&offset), meta + sizeof(ParamType), sizeof(ParamType));
          float deq = Dequantize(q[el], scale, offset);
          if (T::IS_WEIGHTED_FLAG) {
            float w = m_weights->ph[start_idx + h];
            acc += deq * w;
            weight_acc += w;
          } else {
            acc += deq;
          }
        }
        float ref;
        if (T::SUM_POOLING_FLAG) {
          ref = acc;
        } else if (T::IS_WEIGHTED_FLAG) {
          ref = weight_acc != 0.0f ? acc / weight_acc : 1.0f;
        } else {
          ref = acc / static_cast<float>(hotness);
        }
        m_ref_result[b * m_num_elements + el] = from_float(ref);
      }
    }
  }

  void LaunchKernel(CacheDataType& cache_data) {
    m_result = std::make_shared<TestBuffer<ParamType>>(m_batch * m_num_elements * sizeof(ParamType));
    SparseType_t hot_type = T::FIXED_HOTNESS_FLAG ? SparseType_t::Fixed : SparseType_t::CSR;
    PoolingType_t pooling_type = T::IS_WEIGHTED_FLAG
        ? (T::SUM_POOLING_FLAG ? PoolingType_t::WeightedSum : PoolingType_t::WeightedMean)
        : (T::SUM_POOLING_FLAG ? PoolingType_t::Sum : PoolingType_t::Mean);
    // Weights are ParamType (Float32/Float16); accumulate in float.
    callFindAndCombineKernel<IndexType, CacheDataType>(
        static_cast<uint32_t>(m_num_keys), m_batch, m_table->pd, m_keys->pd,
        m_offsets == nullptr ? nullptr : m_offsets->pd,
        m_weights == nullptr ? nullptr : m_weights->pd,
        m_hotness, cache_data, m_num_elements,
        hot_type, pooling_type, /*load_indices=*/T::LOAD_INDICES_FLAG,
        VALUE_TYPE_ID, data_type<ParamType>(), DataType_t::Float32,
        data_type<ParamType>(),
        m_result->pd, m_stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(m_stream));
  }

  void CheckResult() {
    m_result->DtoH(m_stream);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    // __half output is limited by half granularity (coarser for sum pooling at larger magnitudes);
    // float output matches the dequant-mirroring reference tightly.
    float tolerance;
    if constexpr (std::is_same_v<ParamType, __half>) {
      tolerance = T::SUM_POOLING_FLAG ? 5e-2f : 1e-2f;
    } else {
      tolerance = 1e-3f;
    }
    EXPECT_TRUE(std::equal(m_ref_result.begin(), m_ref_result.end(), m_result->ph, Near(tolerance)));
  }

  cudaStream_t m_stream;
  uint64_t m_num_rows{0};
  uint32_t m_num_elements{0};
  uint32_t m_batch{0};
  uint32_t m_hotness{0};
  bool m_allocOnHost{false};
  uint64_t m_row_bytes{0};
  IndexType m_num_keys{0};

  std::shared_ptr<TestBuffer<int8_t>> m_table = nullptr;
  std::shared_ptr<TestBuffer<IndexType>> m_keys = nullptr;
  std::shared_ptr<TestBuffer<IndexType>> m_offsets = nullptr;
  std::shared_ptr<TestBuffer<ParamType>> m_weights = nullptr;
  std::shared_ptr<TestBuffer<ParamType>> m_result = nullptr;
  std::vector<ParamType> m_ref_result;

  nve::DefaultAllocator m_cache_allocator;
  nve::Logger m_cache_logger;
  std::shared_ptr<CacheType> m_cache_ptr;
  nve::LookupContextHandle m_handle_lookup;
  nve::ModifyContextHandle m_handle_modify;
};

TYPED_TEST_SUITE_P(EmbeddingCacheQuantLookupRefTest);

TYPED_TEST_P(EmbeddingCacheQuantLookupRefTest, TestPoolingAgainstRefCpu) {
  // num_elements 132 / 30 / 63 select the Vec4 / Vec2 / Vec1 dequant paths (rowSize % 4).
  for (const auto batch : {17, 2048}) {
    for (const auto hotness : {59}) {
      for (const auto num_elements : {132, 30, 63}) {
        for (const auto allocOnHost : {true, false}) {
          this->LaunchTest(262144, num_elements, batch, hotness, allocOnHost);
        }
      }
    }
  }
}

REGISTER_TYPED_TEST_SUITE_P(EmbeddingCacheQuantLookupRefTest, TestPoolingAgainstRefCpu);

// QInt8 storage is signed char; QUint8 storage is unsigned char (matching QuantizationHelper::Vec1).
typedef ::testing::Types<
    // QInt8RowwiseF32 (symmetric-around-mean), covering the four pooling modes over fixed and CSR.
    EmbedTestTypeQuantPoolingCombo<DataType_t::QInt8RowwiseF32, char, int32_t, true,  true,  false>,
    EmbedTestTypeQuantPoolingCombo<DataType_t::QInt8RowwiseF32, char, int64_t, false, true,  false>,
    EmbedTestTypeQuantPoolingCombo<DataType_t::QInt8RowwiseF32, char, int32_t, true,  false, false>,
    EmbedTestTypeQuantPoolingCombo<DataType_t::QInt8RowwiseF32, char, int64_t, false, false, false>,
    EmbedTestTypeQuantPoolingCombo<DataType_t::QInt8RowwiseF32, char, int32_t, true,  true,  true>,
    EmbedTestTypeQuantPoolingCombo<DataType_t::QInt8RowwiseF32, char, int64_t, false, false, true>,
    // QUint8RowwiseF32 (min/max affine).
    EmbedTestTypeQuantPoolingCombo<DataType_t::QUint8RowwiseF32, unsigned char, int32_t, true,  true,  false>,
    EmbedTestTypeQuantPoolingCombo<DataType_t::QUint8RowwiseF32, unsigned char, int64_t, false, true,  false>,
    EmbedTestTypeQuantPoolingCombo<DataType_t::QUint8RowwiseF32, unsigned char, int32_t, true,  false, false>,
    EmbedTestTypeQuantPoolingCombo<DataType_t::QUint8RowwiseF32, unsigned char, int64_t, false, false, false>,
    EmbedTestTypeQuantPoolingCombo<DataType_t::QUint8RowwiseF32, unsigned char, int32_t, true,  true,  true>,
    EmbedTestTypeQuantPoolingCombo<DataType_t::QUint8RowwiseF32, unsigned char, int64_t, false, false, true>,
    // QInt8RowwiseF16 (symmetric, pure-scale / no offset; __half scale + output).
    EmbedTestTypeQuantPoolingCombo<DataType_t::QInt8RowwiseF16, char, int32_t, true,  true,  false>,
    EmbedTestTypeQuantPoolingCombo<DataType_t::QInt8RowwiseF16, char, int64_t, false, true,  false>,
    EmbedTestTypeQuantPoolingCombo<DataType_t::QInt8RowwiseF16, char, int32_t, true,  false, false>,
    EmbedTestTypeQuantPoolingCombo<DataType_t::QInt8RowwiseF16, char, int64_t, false, false, false>,
    EmbedTestTypeQuantPoolingCombo<DataType_t::QInt8RowwiseF16, char, int32_t, true,  true,  true>,
    EmbedTestTypeQuantPoolingCombo<DataType_t::QInt8RowwiseF16, char, int64_t, false, false, true>,
    // load_indices=true (indirect key-gather) representative subset:
    EmbedTestTypeQuantPoolingCombo<DataType_t::QInt8RowwiseF32,  char,          int32_t, true,  true,  false, /*LOAD_INDICES=*/true>,  // Fixed sum
    EmbedTestTypeQuantPoolingCombo<DataType_t::QInt8RowwiseF32,  char,          int32_t, false, false, false, /*LOAD_INDICES=*/true>,  // CSR mean
    EmbedTestTypeQuantPoolingCombo<DataType_t::QUint8RowwiseF32, unsigned char, int32_t, false, true,  false, /*LOAD_INDICES=*/true>,  // CSR sum
    EmbedTestTypeQuantPoolingCombo<DataType_t::QInt8RowwiseF16,  char,          int32_t, true,  false, false, /*LOAD_INDICES=*/true>>  // Fixed mean (F16)
    QuantPoolingTypes;

INSTANTIATE_TYPED_TEST_SUITE_P(EmbeddingLookupQuant,
                               EmbeddingCacheQuantLookupRefTest,
                               QuantPoolingTypes);

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
        m_ref_result.resize(this->m_batch * this->m_hotness * this->m_num_elements);
        m_ref_hitmask.resize(0);
        int64_t i = 0;
        for (; i < static_cast<int64_t>(this->m_batch) * static_cast<int64_t>(this->m_hotness); i++) {
            IndexType idx = indices[i];

            bool in_cache = std::binary_search(cache_data.begin(), cache_data.end(), idx);
            if (in_cache) {
               for (uint32_t el = 0; el < this->m_num_elements; el++) {
                    m_ref_result[i * this->m_num_elements + el] = table[idx * this->m_num_elements + el];
               }
               mask |= 0x80000000;
            }

            if (((i + 1) % 32) == 0) {
                m_ref_hitmask.push_back(mask);
                mask = 0;
            } else {
                mask >>= 1;
            }

        }
        uint64_t set_bits = i % 32;
        if (set_bits != 0) {
            mask >>= (32-set_bits-1);
            m_ref_hitmask.push_back(mask);
        }
    }

    void LaunchKernel(const ElemType* /*table*/, 
                      const IndexType* indices,
                      CacheDataType& cache_data) {
        int64_t num_indices = static_cast<int64_t>(this->m_batch) * static_cast<int64_t>(this->m_hotness);
        int64_t num_hitmask_elems = (num_indices + 31) / 32;

        m_result = std::make_shared<TestBuffer<ElemType>>(num_indices *this->m_num_elements * sizeof(ElemType));
        m_hitmask = std::make_shared<TestBuffer<uint32_t>>(num_hitmask_elems * sizeof(uint32_t));

        std::memset(m_hitmask->ph, 0, num_hitmask_elems * sizeof(uint32_t));
        m_hitmask->HtoD(this->m_stream);

        NVE_DEBUG_PRINTF_("launch lookup with hitmask kernel\n");

        CHECK_CUDA_ERROR((nve::call_cache_query_hit_mask<IndexType, IndexType>(
            indices, this->m_batch * this->m_hotness,
            reinterpret_cast<int8_t*>(m_result->pd), m_hitmask->pd, cache_data,
            this->m_stream, 0, this->m_num_elements * sizeof(ElemType))));                         
    }

    void CheckResult() {
        NVE_DEBUG_PRINTF_("check lookup with hitmask results\n");
        m_result->DtoH(this->m_stream);
        m_hitmask->DtoH(this->m_stream);

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        EXPECT_TRUE(std::equal(m_ref_hitmask.begin(), m_ref_hitmask.end(), m_hitmask->ph)); 

        int64_t curr_mask_idx = 0;
        int64_t curr_mask_el = 1;

        int64_t num_indices = static_cast<int64_t>(this->m_batch) * static_cast<int64_t>(this->m_hotness);
        for (int64_t i = 0; i < num_indices; i++) {
            uint32_t curr_mask = m_ref_hitmask[curr_mask_idx];
            if ((curr_mask & curr_mask_el) == 0x1) {
                EXPECT_TRUE(std::equal(m_result->ph + i * this->m_num_elements,
                                       m_result->ph + (i + 1) * this->m_num_elements,
                                       m_ref_result.begin() + i * this->m_num_elements)); 
            }
            if (((i + 1) % 32) == 0) {
                ++curr_mask_idx;
                curr_mask_el = 1;
            } else {
                curr_mask_el <<= 1;
            }
        }
    }

    std::shared_ptr<TestBuffer<ElemType>> m_result = nullptr;
    std::shared_ptr<TestBuffer<uint32_t>> m_hitmask = nullptr;
    std::vector<ElemType> m_ref_result;
    std::vector<uint32_t> m_ref_hitmask;
};

TYPED_TEST_SUITE_P(EmbeddingCacheLookupHitMaskRefTest);

TYPED_TEST_P(EmbeddingCacheLookupHitMaskRefTest, TestFixedHotnessAgainstRefCpu) {
    for (const auto batch : {17, 2048}) {
        for (const auto hotness : {13, 32}) {
            for (const auto num_elements : {53, 512}) {
                for(const auto allocOnHost: {false, true}){
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
        int64_t num_indices = static_cast<int64_t>(this->m_batch) * static_cast<int64_t>(this->m_hotness);
        int64_t num_hitmask_elems = (num_indices + 63) / 64;
        m_hitmask = std::make_shared<TestBuffer<uint64_t>>(num_hitmask_elems * sizeof(uint64_t));
        m_input = std::make_shared<TestBuffer<ElemType>>(this->m_batch * this->m_hotness * this->m_num_elements * sizeof(ElemType));

        uint64_t mask = 0;

        int64_t i = 0;
        for (; i < num_indices; i++) {
            IndexType idx = indices[i];
            for (uint32_t el = 0; el < this->m_num_elements; el++) {
                m_input->ph[i * this->m_num_elements + el] = table[idx * this->m_num_elements + el];
            }
            bool in_cache = std::binary_search(cache_data.begin(), cache_data.end(), idx);
            if (in_cache) {
               mask |= 0x80000000;
            }

            if (((i + 1) % 64) == 0) {
                m_hitmask->ph[i / 64] = mask;
                mask = 0;
            } else {
                mask >>= 1;
            }

        }
        if (( i % 64) != 0) {
            m_hitmask->ph[i / 64] = mask;
        }
    }

    void LaunchKernel(const ElemType* /*table*/,
                      const IndexType* /*indices*/,
                      CacheDataType& /*cache_data*/) {
        NVE_DEBUG_PRINTF_("launch scatter kernel\n");

        m_hitmask->HtoD(this->m_stream);
        m_input->HtoD(this->m_stream);
        m_result = std::make_shared<TestBuffer<ElemType>>(this->m_batch * this->m_hotness * this->m_num_elements * sizeof(ElemType));
        uint64_t num_indices = static_cast<uint64_t>(this->m_batch) * static_cast<uint64_t>(this->m_hotness);
        for (uint64_t i = 0; i < num_indices; i++) {
            for (uint32_t el = 0; el < this->m_num_elements; el++) {
                m_result->ph[i * this->m_num_elements + el] = ElemType(8);
            }
        }
        m_result->HtoD(this->m_stream);

        uint32_t embed_width_in_bytes = static_cast<uint32_t>(this->m_num_elements * sizeof(ElemType));
        nve::EmbeddingForwardScatter(m_input->pd, m_result->pd, embed_width_in_bytes,
                                            embed_width_in_bytes, embed_width_in_bytes,
                                            reinterpret_cast<uint64_t*>(m_hitmask->pd),
                                            static_cast<uint32_t>(num_indices), this->m_stream);
    }

    void CheckResult() {
        NVE_DEBUG_PRINTF_("check scatter results\n");
        m_result->DtoH(this->m_stream);

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        int64_t curr_mask_idx = 0;
        int64_t curr_mask_el = 1;

        int64_t num_indices = static_cast<int64_t>(this->m_batch) * static_cast<int64_t>(this->m_hotness);
        for (int64_t i = 0; i < num_indices; i++) {
            if ((m_hitmask->ph[curr_mask_idx] & curr_mask_el) == 0) {                
                EXPECT_TRUE(std::equal(m_result->ph + i * this->m_num_elements,
                                       m_result->ph + (i + 1) * this->m_num_elements,
                                       m_input->ph + i * this->m_num_elements));
            } else {
                EXPECT_FALSE(std::equal(m_result->ph + i * this->m_num_elements,
                                       m_result->ph + (i + 1) * this->m_num_elements,
                                       m_input->ph + i * this->m_num_elements)); 
            }
            if (((i + 1) % 64) == 0) {
                ++curr_mask_idx;
                curr_mask_el = 1;
            } else {
                curr_mask_el <<= 1;
            }
        }
    }

    std::shared_ptr<TestBuffer<ElemType>> m_result = nullptr;
    std::shared_ptr<TestBuffer<ElemType>> m_input = nullptr;
    std::shared_ptr<TestBuffer<uint64_t>> m_hitmask = nullptr;
    std::vector<ElemType> m_ref_result;
};

TYPED_TEST_SUITE_P(ScatterRefTest);

TYPED_TEST_P(ScatterRefTest, TestScatterAgainstRefCpu) {
    for (const auto batch : {17, 2048}) {
        for (const auto hotness : {13, 32}) {
            for (const auto num_elements : {13, 512}) {
                for(const auto allocOnHost: {true, false}){
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
        int64_t num_indices = static_cast<int64_t>(this->m_batch) * static_cast<int64_t>(this->m_hotness);
        m_input = std::make_shared<TestBuffer<ElemType>>(this->m_batch * this->m_hotness * this->m_num_elements * sizeof(ElemType));

        m_ref_result.resize(this->m_num_elements * this->m_num_rows);
        memcpy(&m_ref_result[0], table, this->m_num_elements * this->m_num_rows * sizeof(ElemType));

        // generate inputs to update and update ref
        int64_t i = 0;
        for (; i < num_indices; i++) {
            IndexType idx = indices[i];
            for (uint32_t el = 0; el < this->m_num_elements; el++) {
                m_input->ph[i * this->m_num_elements + el] = table[idx * this->m_num_elements + el] + ElemType(i);
                m_ref_result[idx * this->m_num_elements + el] += ElemType(i);
            }
        }
    }

    void LaunchKernel(const ElemType* table,
                      const IndexType* indices,
                      CacheDataType& /*cache_data*/) {
        NVE_DEBUG_PRINTF_("launch update kernel\n");

        m_input->HtoD(this->m_stream);

        uint32_t embed_width_in_bytes = static_cast<uint32_t>(this->m_num_elements * sizeof(ElemType));
        uint64_t num_indices = static_cast<uint64_t>(this->m_batch) * static_cast<uint64_t>(this->m_hotness);
        nve::UpdateTable(m_input->pd, indices, const_cast<ElemType*>(table), 
                         embed_width_in_bytes, embed_width_in_bytes, embed_width_in_bytes,
                         static_cast<uint32_t>(num_indices), this->m_stream);
    }

    void CheckResult() {
        NVE_DEBUG_PRINTF_("check update results\n");

        ElemType* curr_table = this->syncTable();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        EXPECT_TRUE(std::equal(m_ref_result.begin(), m_ref_result.end(), curr_table));
    }

    std::shared_ptr<TestBuffer<ElemType>> m_input = nullptr;
    std::vector<ElemType> m_ref_result;
};

TYPED_TEST_SUITE_P(UpdateRefTest);

// this test should get unique keys, therefore using batch size of 1
TYPED_TEST_P(UpdateRefTest, TestUpdateAgainstRefCpu) {
    for (const auto hotness : {17, 2048}) {
        for (const auto num_elements : {53, 512}) {
            for(const auto allocOnHost: {true, false}){
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
        int64_t num_indices = static_cast<int64_t>(this->m_batch) * static_cast<int64_t>(this->m_hotness);
        m_input = std::make_shared<TestBuffer<ElemType>>(this->m_batch * this->m_hotness * this->m_num_elements * sizeof(ElemType));

        m_ref_result.resize(this->m_num_elements * this->m_num_rows);
        memcpy(&m_ref_result[0], table, this->m_num_elements * this->m_num_rows * sizeof(ElemType));

        // generate inputs to update and update ref
        std::mt19937 genr(0X753812);
        std::uniform_int_distribution<int32_t> dist_nom(-3, 3);
        std::uniform_int_distribution<uint32_t> dist_denom(5, 9);

        for (int64_t i = 0; i < num_indices; i++) {
            IndexType idx = indices[i];
            for (uint32_t el = 0; el < this->m_num_elements; el++) {
                float val = float(dist_nom(genr)) / float(1 << dist_denom(genr));
                m_input->ph[i * this->m_num_elements + el] = ElemType(val);
                m_ref_result[idx * this->m_num_elements + el] += ElemType(val);
            }
        }
    }

    void LaunchKernel(const ElemType* table,
                      const IndexType* indices,
                      CacheDataType& /*cache_data*/) {
        NVE_DEBUG_PRINTF_("launch update accumulate kernel\n");

        m_input->HtoD(this->m_stream);

        uint64_t num_indices = static_cast<uint64_t>(this->m_batch) * static_cast<uint64_t>(this->m_hotness);
        nve::UpdateAccumulateTable(m_input->pd, indices, const_cast<ElemType*>(table), 
                                   this->m_num_elements, this->m_num_elements, this->m_num_elements,
                                   static_cast<uint32_t>(num_indices), this->m_stream);
    }

    void CheckResult() {
        NVE_DEBUG_PRINTF_("check update accumulate results\n");

        ElemType* curr_table = this->syncTable();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        const float tolerance = 1e-5f;
        EXPECT_TRUE(std::equal(m_ref_result.begin(), m_ref_result.end(), curr_table, Near(tolerance)));
    }

    std::shared_ptr<TestBuffer<ElemType>> m_input = nullptr;
    std::vector<ElemType> m_ref_result;
};

TYPED_TEST_SUITE_P(UpdateAccumulateRefTest);

TYPED_TEST_P(UpdateAccumulateRefTest, TestUpdateAccumulateAgainstRefCpu) {
    for (const auto batch : {171}) {
        for (const auto hotness : {16, 111}) {
            for (const auto num_elements : {53, 512}) {
                for(const auto allocOnHost: {true, false}){
                    NVE_DEBUG_PRINTF_("running test on batch %d hotness %d num_elements %d allocateDataOn: %s\n", batch, hotness, num_elements, allocOnHost?"host":"device");
                        this->LaunchTest(262144, num_elements, batch, hotness, allocOnHost);
            }
                }
        }
    }
}

REGISTER_TYPED_TEST_SUITE_P(UpdateAccumulateRefTest, TestUpdateAccumulateAgainstRefCpu);

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
        IndexType num_keys = this->m_batch * this->m_hotness;

        // first compute cache content
        m_ref_cache = this->GetCacheContent();
        
        // second dedup keys
        m_unique_keys = std::make_shared<TestBuffer<IndexType>>(num_keys * sizeof(IndexType));
        // allocate 1 extra element for counts, because we call cub exclusive prefix sum kernel
        // to compute n + 1 outputs, and it reads the last input element even though it is not used
        // in the output
        auto counts_out = std::make_shared<TestBuffer<IndexType>>((num_keys + 1) * sizeof(IndexType));
        
        auto inverse_buffer = std::make_shared<TestBuffer<IndexType>>(num_keys * sizeof(IndexType));
        auto offsets = std::make_shared<TestBuffer<IndexType>>((num_keys + 1) * sizeof(IndexType));
        
        std::shared_ptr<Deduper<IndexType, -1>> deduper_ = std::make_shared<Deduper<IndexType, -1>>();
        size_t tmp_mem_size_device, tmp_mem_size_host;
        deduper_->get_alloc_requirements(num_keys, tmp_mem_size_device, tmp_mem_size_host);
        char* tmp_device_mem;
        CHECK_CUDA_ERROR(cudaMalloc(&tmp_device_mem, tmp_mem_size_device));
        char* tmp_host_mem;
        CHECK_CUDA_ERROR(cudaMallocHost(&tmp_host_mem, tmp_mem_size_host));
        deduper_->set_and_init_buffers(num_keys, tmp_device_mem, tmp_host_mem);

        CHECK_CUDA_ERROR(cudaMallocHost(&m_h_num_runs_out, sizeof(IndexType) * 2));

        deduper_->dedup(reinterpret_cast<const IndexType*>(indices),
                        num_keys, m_unique_keys->pd, 
                        counts_out->pd,
                        nullptr, 
                        m_h_num_runs_out, 
                        inverse_buffer->pd, 
                        offsets->pd,
                        this->m_stream);

        m_unique_keys->DtoH(this->m_stream);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        m_input = std::make_shared<TestBuffer<ElemType>>(*m_h_num_runs_out * this->m_num_elements * sizeof(ElemType));

        m_ref_result.resize(this->m_num_elements * this->m_num_rows);
        memcpy(&m_ref_result[0], table, this->m_num_elements * this->m_num_rows * sizeof(ElemType));

        // generate inputs to update and update ref
        for (int64_t i = 0; i < *m_h_num_runs_out; i++) {
            IndexType idx = m_unique_keys->ph[i];
            for (uint32_t el = 0; el < this->m_num_elements; el++) {
                m_input->ph[i * this->m_num_elements + el] = ElemType(i);
                m_ref_result[idx * this->m_num_elements + el] += ElemType(i);
                if (m_ref_cache.count(idx))
                    m_ref_cache[idx][el] += ElemType(i);
            }
        }

        CHECK_CUDA_ERROR(cudaFree(tmp_device_mem));
        CHECK_CUDA_ERROR(cudaFreeHost(tmp_host_mem));
    }

    void LaunchKernel(const ElemType* table, 
                      const IndexType* /*indices*/,
                      CacheDataType& cache_data) {
        NVE_DEBUG_PRINTF_("launch update kernel\n");

        m_input->HtoD(this->m_stream);

        uint64_t num_indices = *m_h_num_runs_out;
        CHECK_CUDA_ERROR((nve::call_update_accumulate_no_sync_fused_with_pipeline<IndexType,IndexType>(reinterpret_cast<int8_t*>(m_input->pd), 
                                   (int8_t*)(table), 
                                   m_unique_keys->pd, 
                                   static_cast<IndexType>(num_indices),
                                   this->m_num_elements * sizeof(ElemType),
                                   cache_data,
                                   this->m_stream)));
    }

    void CheckResult() {
        NVE_DEBUG_PRINTF_("check update results\n");

        ElemType* curr_table = this->syncTable();
        auto test_cache = this->GetCacheContent();

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        for (uint32_t j = 0; j < this->m_num_rows; j ++ ) {
            for (uint32_t i = 0; i < this->m_num_elements; i++)
            {
                auto ref_val = m_ref_result[j*this->m_num_elements + i];
                auto test_val = curr_table[j*this->m_num_elements + i];
                ASSERT_FLOAT_EQ(ref_val, test_val);
            }
        }

        // compare cache content
        ASSERT_EQ(test_cache.size(), m_ref_cache.size());
        for (const auto& ent : test_cache)
        {
            auto key = ent.first;
            auto test_vec = ent.second;
            auto ref_it = m_ref_cache.find(key);
            ASSERT_TRUE(ref_it != m_ref_cache.end());
            auto ref_vec = ref_it->second;
            for (uint32_t i = 0; i < this->m_num_elements; i++)
            {
                auto ref_val = ref_vec[i];
                auto test_val = test_vec[i];
                ASSERT_FLOAT_EQ(ref_val, test_val);
            }

        }

    }

    std::shared_ptr<TestBuffer<ElemType>> m_input = nullptr;
    std::vector<ElemType> m_ref_result;
    std::shared_ptr<TestBuffer<IndexType>> m_unique_keys = nullptr;
    IndexType* m_h_num_runs_out = nullptr;
    std::map<IndexType, std::vector<ElemType>> m_ref_cache;
};

TYPED_TEST_SUITE_P(EmbeddingCacheFusedUpdateAccumlateTest);

TYPED_TEST_P(EmbeddingCacheFusedUpdateAccumlateTest, TestFusedAccumulateRefCpu) {
    for (const auto batch : {32*15*8, 12, 348989}) {
        for (const auto hotness : {1}) {
            for (const auto num_elements : {128}) {
                for(const auto allocOnHost: {true, false}){
                    NVE_DEBUG_PRINTF_("running test on batch %d hotness %d num_elements %d allocateDataOn: %s\n", batch, hotness, num_elements, allocOnHost?"host":"device");
                    this->LaunchTest(262144, num_elements, batch, hotness, allocOnHost);
                }
            }
        }
    }
}

REGISTER_TYPED_TEST_SUITE_P(EmbeddingCacheFusedUpdateAccumlateTest, TestFusedAccumulateRefCpu);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
        m_ref_result.resize(0);
        ASSERT_TRUE(this->m_hotness == 1);
        for (uint32_t b = 0; b < this->m_batch; b++) {
            for (uint32_t el = 0; el < this->m_num_elements; el++) {
                ElemType acc = 0;
                acc = table[indices[b] * this->m_num_elements + el];
                m_ref_result.push_back(acc);
            }
        }
    }

    void LaunchKernel(const ElemType* table, 
                      const IndexType* indices,
                      CacheDataType& cache_data) {
        m_result = std::make_shared<TestBuffer<ElemType>>(this->m_batch * this->m_hotness * this->m_num_elements * sizeof(ElemType));
        NVE_DEBUG_PRINTF_("launch sort gather lookup kernel\n");

        // first calc required aux size
        size_t bytes = 0;
        CHECK_CUDA_ERROR((nve::call_sort_gather<IndexType, IndexType>(
            reinterpret_cast<const int8_t*>(table), 
            reinterpret_cast<int8_t*>(m_result->pd), 
            indices, 
            nullptr, 
            bytes,
            this->m_batch * this->m_hotness, 
            this->m_num_elements * sizeof(ElemType), 
            this->m_kernel_param,
            cache_data, 
            this->m_stream)));
        ASSERT_TRUE(bytes > 0 && bytes != static_cast<size_t>(-1));

        // alloc aux
        int8_t* aux = nullptr;
        CHECK_CUDA_ERROR(cudaMalloc(&aux, bytes));

        // launch kernel
        CHECK_CUDA_ERROR((nve::call_sort_gather<IndexType, IndexType>(
            reinterpret_cast<const int8_t*>(table), 
            reinterpret_cast<int8_t*>(m_result->pd), 
            indices, 
            aux, 
            bytes,
            this->m_batch * this->m_hotness, 
            this->m_num_elements * sizeof(ElemType), 
            1024,
            cache_data, 
            this->m_stream)));

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        //clean up
        CHECK_CUDA_ERROR(cudaFree(aux));
    }

    void CheckResult() {
        NVE_DEBUG_PRINTF_("check lookup results\n");
        m_result->DtoH(this->m_stream);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        for (uint32_t j = 0; j < this->m_hotness * this->m_batch; j ++ ) {
            for (uint32_t i = 0; i < this->m_num_elements; i++)
            {
                auto ref_val = m_ref_result[j*this->m_num_elements + i];
                auto test_val = m_result->ph[j*this->m_num_elements + i];
                ASSERT_FLOAT_EQ(ref_val, test_val);
            }
        }
    }

    std::shared_ptr<TestBuffer<ElemType>> m_result = nullptr;
    std::vector<ElemType> m_ref_result;
};

TYPED_TEST_SUITE_P(EmbeddingCacheLookupSortGatherRefTest);

TYPED_TEST_P(EmbeddingCacheLookupSortGatherRefTest, TestSortGather) {
    for (const auto batch : {4096, 1, 33, 4097, 16*1024, 3*4096+1, 2*4096+1}) {
        for (const auto hotness : {1}) {
            for (const auto num_elements : {128, 1, 31}) {
                for(const auto allocOnHost: {false}){
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

template <typename ElemT, typename IndexT, typename AccumT,
          bool FIXED_HOTNESS, bool SUM_POOILING, bool IS_WEIGHTED, bool LOAD_INDICES = false>
struct EmbedTestTypePoolingCombo {
  typedef ElemT ElemType;
  typedef IndexT IndexType;
  typedef AccumT AccumType;
  static bool constexpr FIXED_HOTNESS_FLAG = FIXED_HOTNESS;
  static bool constexpr SUM_POOLING_FLAG = SUM_POOILING;
  static bool constexpr IS_WEIGHTED_FLAG = IS_WEIGHTED;
  // false: the kernel indexes rows positionally (sampleStart+i, production/post-gather path);
  // true: the kernel reads the key value from the indices buffer (new indirect-gather feature).
  static bool constexpr LOAD_INDICES_FLAG = LOAD_INDICES;
};

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

typedef ::testing::Types<
                        // SUM
                         EmbedTestTypePoolingCombo<float, int32_t, float, true, true, false>,
                         EmbedTestTypePoolingCombo<float, int64_t, float, true, true, false>,
                         EmbedTestTypePoolingCombo<__half, int32_t, __half, true, true, false>,
                         EmbedTestTypePoolingCombo<__half, int64_t, __half, true, true, false>,
                         EmbedTestTypePoolingCombo<__half, int32_t, float, true, true, false>,
                         EmbedTestTypePoolingCombo<__half, int64_t, float, true, true, false>,
                         // MEAN
                         EmbedTestTypePoolingCombo<float, int32_t, float, true, false, false>,
                         EmbedTestTypePoolingCombo<float, int64_t, float, true, false, false>,
                         EmbedTestTypePoolingCombo<__half, int32_t, __half, true, false, false>,
                         EmbedTestTypePoolingCombo<__half, int64_t, __half, true, false, false>,
                         EmbedTestTypePoolingCombo<__half, int32_t, float, true, false, false>,
                         EmbedTestTypePoolingCombo<__half, int64_t, float, true, false, false>,
                         // CSR 
                         EmbedTestTypePoolingCombo<float, int32_t, float, false, true, false>,
                         EmbedTestTypePoolingCombo<float, int64_t, float, false, true, false>,
                         EmbedTestTypePoolingCombo<__half, int32_t, __half, false, true, false>,
                         EmbedTestTypePoolingCombo<__half, int64_t, __half, false, true, false>,
                         EmbedTestTypePoolingCombo<__half, int32_t, float, false, true, false>,
                         EmbedTestTypePoolingCombo<__half, int64_t, float, false, true, false>,
                         EmbedTestTypePoolingCombo<float, int32_t, float, false, false, false>,
                         EmbedTestTypePoolingCombo<float, int64_t, float, false, false, false>,
                         EmbedTestTypePoolingCombo<__half, int32_t, __half, false, false, false>,
                         EmbedTestTypePoolingCombo<__half, int64_t, __half, false, false, false>,
                         EmbedTestTypePoolingCombo<__half, int32_t, float, false, false, false>,
                         EmbedTestTypePoolingCombo<__half, int64_t, float, false, false, false>,
                         // weighted
                         EmbedTestTypePoolingCombo<float, int32_t, float, true, true, true>,
                         EmbedTestTypePoolingCombo<float, int64_t, float, true, true, true>,
                         EmbedTestTypePoolingCombo<__half, int32_t, __half, true, true, true>,
                         EmbedTestTypePoolingCombo<__half, int64_t, __half, true, true, true>,
                         EmbedTestTypePoolingCombo<__half, int32_t, float, true, true, true>,
                         EmbedTestTypePoolingCombo<__half, int64_t, float, true, true, true>,
                         EmbedTestTypePoolingCombo<float, int32_t, float, true, false, true>,
                         EmbedTestTypePoolingCombo<float, int64_t, float, true, false, true>,
                         EmbedTestTypePoolingCombo<__half, int32_t, __half, true, false, true>,
                         EmbedTestTypePoolingCombo<__half, int64_t, __half, true, false, true>,
                         EmbedTestTypePoolingCombo<__half, int32_t, float, true, false, true>,
                         EmbedTestTypePoolingCombo<__half, int64_t, float, true, false, true>,
                         EmbedTestTypePoolingCombo<float, int32_t, float, false, true, true>,
                         EmbedTestTypePoolingCombo<float, int64_t, float, false, true, true>,
                         EmbedTestTypePoolingCombo<__half, int32_t, __half, false, true, true>,
                         EmbedTestTypePoolingCombo<__half, int64_t, __half, false, true, true>,
                         EmbedTestTypePoolingCombo<__half, int32_t, float, false, true, true>,
                         EmbedTestTypePoolingCombo<__half, int64_t, float, false, true, true>,
                         EmbedTestTypePoolingCombo<float, int32_t, float, false, false, true>,
                         EmbedTestTypePoolingCombo<float, int64_t, float, false, false, true>,
                         EmbedTestTypePoolingCombo<__half, int32_t, __half, false, false, true>,
                         EmbedTestTypePoolingCombo<__half, int64_t, __half, false, false, true>,
                         EmbedTestTypePoolingCombo<__half, int32_t, float, false, false, true>,
                         EmbedTestTypePoolingCombo<__half, int64_t, float, false, false, true>,
                         // load_indices=true (indirect key-gather) representative subset:
                         EmbedTestTypePoolingCombo<float, int32_t, float, true,  true,  false, /*LOAD_INDICES=*/true>,   // Fixed sum
                         EmbedTestTypePoolingCombo<float, int32_t, float, true,  false, false, /*LOAD_INDICES=*/true>,   // Fixed mean
                         EmbedTestTypePoolingCombo<float, int32_t, float, false, true,  false, /*LOAD_INDICES=*/true>,   // CSR sum
                         EmbedTestTypePoolingCombo<float, int32_t, float, false, false, false, /*LOAD_INDICES=*/true>>   // CSR mean
    EmbedTestPoolingTypes;

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

class EmbeddingCacheRefTestNames {
 public:
  template <typename T>
  static std::string GetName(const std::string& base_name, int i) {
    typedef typename T::ElemType ElemType;
    typedef typename T::IndexType IndexType;

    std::string test_name = base_name;
    if (std::is_same_v<ElemType, float>) {
      test_name += std::string("Elem[float]_");
    }
    if (std::is_same_v<ElemType, __half>) {
      test_name += std::string("Elem[half]_");
    }
    if (std::is_same_v<IndexType, int32_t>) {
      test_name += std::string("Index[int32]_");
    }
    if (std::is_same_v<IndexType, int64_t>) {
      test_name += std::string("Index[int64]_");
    }
    test_name += ::testing::PrintToString(i);
    return test_name;
  }
};

class EmbeddingCacheLookupRefTestNames {
 public:
  template <typename T>
  static std::string GetName(int i) {
    return EmbeddingCacheRefTestNames::GetName<T>("EmbeddingCacheLookupRefTest_", i);
  }
};

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

INSTANTIATE_TYPED_TEST_SUITE_P(EmbeddingLookup,    // TRTREC-44
                               EmbeddingCacheLookupRefTest,
                               EmbedTestPoolingTypes,
                               EmbeddingCacheLookupRefTestNames);

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
    const size_t m_num_rows = 1000000;
    const size_t m_row_size = sizeof(DataType)*embedding_size;
    
    const size_t m_batch = 512;
    const size_t m_hotness = 4096;
    const auto num_keys = m_batch * m_hotness;
    auto keys = std::make_shared<TestBuffer<IndexType>>(num_keys * sizeof(IndexType));
    auto unique_keys = std::make_shared<TestBuffer<IndexType>>(num_keys * sizeof(IndexType));
    // allocate an extra element for loc_map, because it is reused for counters, and the
    // way we use counters in Dedup may lead to a read of extra element in one of cub kernels 
    auto loc_map = std::make_shared<TestBuffer<IndexType>>((num_keys + 1) * sizeof(IndexType));
    IndexType* h_num_runs_out;
    auto inverse_buffer = std::make_shared<TestBuffer<IndexType>>(num_keys * sizeof(IndexType));
    auto offsets = std::make_shared<TestBuffer<IndexType>>((num_keys + 1) * sizeof(IndexType));
    auto grads = std::make_shared<TestBuffer<DataType>>(num_keys * m_row_size);
    auto unique_grads = std::make_shared<TestBuffer<DataType>>(num_keys * m_row_size);
    auto ref = std::make_shared<TestBuffer<DataType>>(num_keys * m_row_size);

    cudaStream_t m_stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&m_stream));
    
    const float alpha = 1.05f;
    auto sg = getSampleGenerator<IndexType>(alpha, static_cast<IndexType>(m_num_rows), m_hotness, 283982);
    
    // make sure we have keys appearing more than SPLIT_LIMIT times
    // change approx half of batches to contain approx half identical keys
    std::mt19937 genr(0X753812);
    std::bernoulli_distribution distrib(0.5);
    std::uniform_int_distribution<uint32_t> dist_index(0, m_hotness-1);

    for (uint32_t b = 0; b < m_batch; b++)
    {
        auto sample = sg->getCategoryIndices();
        if (distrib(genr)) {
            auto duplicate_idx = dist_index(genr);
            IndexType key_to_duplicate = sample[duplicate_idx];
            for (uint32_t h = 0; h < m_hotness; h++) {
                if (distrib(genr)) sample[h] = key_to_duplicate;
            }
        }
        std::copy(sample.begin(), sample.end(), keys->ph + b*m_hotness);
    }

    keys->HtoD(m_stream);
    
    constexpr int32_t MAX_RUN_SIZE = 1024;
    std::shared_ptr<Deduper<IndexType, MAX_RUN_SIZE>> deduper_ = std::make_shared<Deduper<IndexType, MAX_RUN_SIZE>>();
    size_t tmp_mem_size_device, tmp_mem_size_host;
    deduper_->get_alloc_requirements(num_keys, tmp_mem_size_device, tmp_mem_size_host);
    char* tmp_device_mem;
    CHECK_CUDA_ERROR(cudaMalloc(&tmp_device_mem, tmp_mem_size_device));
    char* tmp_host_mem;
    CHECK_CUDA_ERROR(cudaMallocHost(&tmp_host_mem, tmp_mem_size_host));
    deduper_->set_and_init_buffers(num_keys, tmp_device_mem, tmp_host_mem);

    CHECK_CUDA_ERROR(cudaMallocHost(&h_num_runs_out, sizeof(IndexType) * 2));

    memset(grads->ph, 0, grads->m_size);
    std::uniform_real_distribution<float> dist_float(-1.0f, 1.0f);
    for (uint32_t i = 0; i < num_keys; i++) {
        for (uint32_t j = 0; j < embedding_size; j++) {
            (grads->ph)[i*embedding_size + j] = static_cast<DataType>(dist_float(genr));
        }
    }
    grads->HtoD(m_stream);

    memset(unique_grads->ph, 0, unique_grads->m_size);
    unique_grads->HtoD(m_stream);

    deduper_->dedup(reinterpret_cast<const IndexType*>(keys->pd),
                    num_keys, unique_keys->pd, 
                    loc_map->pd, // reuse loc_map buffer for counters
                    loc_map->pd, 
                    h_num_runs_out, 
                    inverse_buffer->pd, 
                    offsets->pd,
                    m_stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(m_stream));
    
    DedupGradients<IndexType>(reinterpret_cast<const void*>(grads->pd), (void*)unique_grads->pd, 
                    DataType_t::Float32,
                    unique_keys->pd, 
                    (MAX_RUN_SIZE == -1) ? nullptr : loc_map->pd,
                    offsets->pd,
                    inverse_buffer->pd,
                    m_row_size,
                    h_num_runs_out,
                    m_stream);

    // calculate ref
    std::map<IndexType, DataType*> ref_grads;
    for (uint32_t i = 0; i < num_keys; i++) 
    {
        auto key = keys->ph[i];
        if (ref_grads.count(key) == 0) {
            DataType* p = new DataType[embedding_size];
            memset(p, 0, m_row_size);
            ref_grads[key] = p;
        }
        auto ref_grad = ref_grads[key];
        for (uint32_t j = 0; j < embedding_size; j++) {
            ref_grad[j] += grads->ph[i * embedding_size + j];
        }
    }

    // check results
    unique_grads->DtoH(m_stream);
    unique_keys->DtoH(m_stream);
    inverse_buffer->DtoH(m_stream);
    loc_map->DtoH(m_stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(m_stream));

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

        GradientCalculator<IndexType, MAX_RUN_SIZE> pool_backprop;

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

// ---------------------------------------------------------------------------
// find_and_dequant tests (cuda_ops/find_and_dequant.cuh).
//
// find_and_dequant is the hotness-1 / no-pooling gather: each key produces one output row of
// dequantized QuantizationHelper<DTYPE>::ParamType values (float for *F32, __half for *F16),
// reconstructed as float(q) * scale [+ offset]. Rows are stored exactly like the FindAndCombine
// quant tests above: [ q[num_elements] ][ scale ][ offset? ], with the stride padded to 4 bytes.
//
// This fixture mirrors EmbeddingCacheQuantLookupRefTest's data generation but exercises the
// dequant-only kernel across:
//   * both CacheData variants -- the set-associative cache (USE_CACHE) and ECNoCache (direct
//     table reads), to cover both AddressFunctor<IndexT, CacheDataT> specializations; and
//   * load_indices=true (resolve the row from the keys buffer) and false (positional row = tid).
// The dequantized bytes are identical whether a lookup hits the cache or falls through to the
// table, so the CPU reference is the same for every variant.
// ---------------------------------------------------------------------------
template <DataType_t VALUE_TYPE_ID_, typename StorageT, typename IndexT,
          bool USE_CACHE, bool LOAD_INDICES>
struct DequantTestType {
  static constexpr DataType_t VALUE_TYPE_ID = VALUE_TYPE_ID_;
  using StorageType = StorageT;
  using IndexType = IndexT;
  static constexpr bool USE_CACHE_FLAG = USE_CACHE;
  static constexpr bool LOAD_INDICES_FLAG = LOAD_INDICES;
};

template <typename T>
class FindAndDequantRefTest : public ::testing::Test {
 public:
  using StorageType = typename T::StorageType;
  using IndexType = typename T::IndexType;
  static constexpr DataType_t VALUE_TYPE_ID = T::VALUE_TYPE_ID;
  // ParamType is the scale/offset and dequantized output type (float for *F32, __half for *F16).
  using ParamType = typename QuantizationHelper<VALUE_TYPE_ID>::ParamType;
  static constexpr bool HAS_OFFSET = QuantizationHelper<VALUE_TYPE_ID>::has_offset;
  static constexpr bool USE_CACHE = T::USE_CACHE_FLAG;
  static constexpr bool LOAD_INDICES = T::LOAD_INDICES_FLAG;

  using CacheType = nve::CacheSAHostModify<IndexType, IndexType>;
  using SACacheData = typename CacheType::CacheData;
  using NoCacheData = typename nve::ECNoCache<IndexType>::CacheData;
  using CacheDataType = std::conditional_t<USE_CACHE, SACacheData, NoCacheData>;

  static ParamType from_float(float v) {
    if constexpr (std::is_same_v<ParamType, __half>) return __float2half(v);
    else return static_cast<ParamType>(v);
  }

  FindAndDequantRefTest()
      : m_cache_allocator(nve::DefaultAllocator::DEFAULT_HOST_ALLOC_THRESHOLD) {
    CHECK_CUDA_ERROR(cudaStreamCreate(&m_stream));
  }
  ~FindAndDequantRefTest() {
    if constexpr (USE_CACHE) {
      if (m_cache_ptr) {
        CHECK_CUDA_ERROR(m_cache_ptr->lookup_context_destroy(m_handle_lookup));
        CHECK_CUDA_ERROR(m_cache_ptr->modify_context_destroy(m_handle_modify));
      }
    }
  }

  void LaunchTest(uint64_t num_rows, uint32_t num_elements, uint64_t num_keys, bool allocOnHost) {
    m_num_rows = num_rows;
    m_num_elements = num_elements;
    m_num_keys = static_cast<IndexType>(num_keys);
    m_allocOnHost = allocOnHost;
    m_row_bytes = m_num_elements * sizeof(StorageType)
                + (HAS_OFFSET ? 2 : 1) * sizeof(ParamType);  // q + scale [+ offset]
    // Pad the row stride to 4 bytes so every row base stays aligned for the Vec4 (char4) load path.
    m_row_bytes = (m_row_bytes + 3u) & ~static_cast<uint64_t>(3u);
    // load_indices=false reads rows positionally over [0, num_keys); load_indices=true uses keys in
    // [1, num_rows). Either way every accessed row must be < num_rows.
    ASSERT_LT(static_cast<uint64_t>(m_num_keys), m_num_rows)
        << "num_keys must be < table rows so positional (load_indices=false) reads stay in-bounds";
    AllocateTable();
    AllocateKeys();

    CacheDataType cache_data = MakeCacheData();
    ComputeRefResults();
    LaunchKernel(cache_data);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CheckResult();
  }

 private:
  // Quantize one row of floats into StorageType q[], producing scale/offset such that
  // float(q[j]) * scale + offset reconstructs v[j]. Mirrors EmbeddingCacheQuantLookupRefTest.
  void Quantize(const std::vector<float>& v, StorageType* q, ParamType& scale_out, ParamType& offset_out) {
    float scale = 1.0f, offset = 0.0f;
    if constexpr (VALUE_TYPE_ID == DataType_t::QInt8RowwiseF32 ||
                  VALUE_TYPE_ID == DataType_t::QInt8RowwiseF16) {
      // Symmetric, pure scale (QInt8 variants set has_offset=false). The HAS_OFFSET branch is
      // dead for these types but kept structurally identical to the FindAndCombine quant fixture.
      if constexpr (HAS_OFFSET) {
        double sum = 0.0;
        for (float x : v) sum += x;
        offset = static_cast<float>(sum / static_cast<double>(v.size()));
      }
      float amax = 0.0f;
      for (float x : v) amax = std::max(amax, std::fabs(x - offset));
      scale = amax > 0.0f ? amax / 127.0f : 1.0f;
      for (size_t j = 0; j < v.size(); ++j) {
        long qi = std::lround((v[j] - offset) / scale);
        qi = std::min<long>(127, std::max<long>(-127, qi));
        q[j] = static_cast<StorageType>(qi);
      }
    } else {  // QUint8Rowwise* : min/max affine
      float lo = v[0], hi = v[0];
      for (float x : v) { lo = std::min(lo, x); hi = std::max(hi, x); }
      scale = hi > lo ? (hi - lo) / 255.0f : 1.0f;
      offset = lo;
      for (size_t j = 0; j < v.size(); ++j) {
        long qi = std::lround((v[j] - offset) / scale);
        qi = std::min<long>(255, std::max<long>(0, qi));
        q[j] = static_cast<StorageType>(qi);
      }
    }
    scale_out = from_float(scale);
    offset_out = from_float(offset);
  }

  // Dequantize one element, mirroring the kernel's per-type arithmetic so the only divergence is
  // float rounding (kept inside the tolerance).
  float Dequantize(StorageType q, ParamType scale, ParamType offset) const {
    if constexpr (std::is_same_v<ParamType, __half>) {
      float prod = static_cast<float>(q) * __half2float(scale);
      if constexpr (HAS_OFFSET) prod += __half2float(offset);
      return __half2float(__float2half(prod));  // kernel stores the dequantized value as __half
    } else {
      float r = static_cast<float>(q) * scale;
      if constexpr (HAS_OFFSET) r += offset;
      return r;
    }
  }

  void AllocateTable() {
    m_table = std::make_shared<TestBuffer<int8_t>>(m_num_rows * m_row_bytes);
    std::mt19937 gen(0X814753);
    // Values in [-1, 1] expressed as a / 2^b to keep float rounding small.
    std::uniform_int_distribution<int32_t> dist_nom(-8, 8);
    std::uniform_int_distribution<uint32_t> dist_denom(3, 9);

    std::vector<float> v(m_num_elements);
    for (uint64_t i = 0; i < m_num_rows; i++) {
      for (uint32_t j = 0; j < m_num_elements; j++) {
        v[j] = float(dist_nom(gen)) / float(1 << dist_denom(gen));
      }
      int8_t* row = m_table->ph + i * m_row_bytes;
      ParamType scale{}, offset{};
      Quantize(v, reinterpret_cast<StorageType*>(row), scale, offset);
      int8_t* meta = row + m_num_elements * sizeof(StorageType);
      memcpy(meta, static_cast<const void*>(&scale), sizeof(ParamType));
      if (HAS_OFFSET) memcpy(meta + sizeof(ParamType), static_cast<const void*>(&offset), sizeof(ParamType));
    }
    m_table->HtoD(m_stream);
  }

  void AllocateKeys() {
    m_keys = std::make_shared<TestBuffer<IndexType>>(m_num_keys * sizeof(IndexType));
    const float alpha = 1.05f;
    const size_t seed = 283982;
    auto sg = getSampleGenerator<IndexType>(alpha, static_cast<IndexType>(m_num_rows),
                                            static_cast<uint32_t>(m_num_keys), seed);
    auto sample = sg->getCategoryIndices();
    std::copy(sample.begin(), sample.begin() + m_num_keys, m_keys->ph);
    m_keys->HtoD(m_stream);
  }

  // Builds the CacheData passed to the kernel: a populated set-associative cache when USE_CACHE,
  // otherwise a trivial ECNoCache descriptor that makes the kernel read straight from the table.
  CacheDataType MakeCacheData() {
    if constexpr (USE_CACHE) {
      InitCache();
      std::vector<IndexType> cached = ComputeCachedIndices(m_cache_data.num_sets, CacheType::NUM_WAYS);
      PopulateCache(cached);
      m_cache_data = m_cache_ptr->get_cache_data(m_handle_lookup);
      return m_cache_data;
    } else {
      NoCacheData data{};
      data.row_size_in_bytes = static_cast<uint32_t>(m_row_bytes);
      data.count_misses = false;
      data.misses = nullptr;
      return data;
    }
  }

  void InitCache() {
    const float cache_ratio = 0.15f;
    const uint32_t num_rows_in_cache = static_cast<uint32_t>(cache_ratio * static_cast<float>(m_num_rows));
    typename CacheType::CacheConfig cfg;
    cfg.embed_width_in_bytes = m_row_bytes;
    cfg.cache_sz_in_bytes = num_rows_in_cache * cfg.embed_width_in_bytes;
    cfg.num_tables = 1;
    cfg.allocate_data_on_host = m_allocOnHost;
    m_cache_ptr = std::make_shared<CacheType>(&m_cache_allocator, &m_cache_logger, cfg);
    m_cache_ptr->init();
    m_cache_ptr->lookup_context_create(m_handle_lookup, nullptr, 0);
    m_cache_data = m_cache_ptr->get_cache_data(m_handle_lookup);
  }

  std::vector<IndexType> ComputeCachedIndices(const int num_sets, const int num_ways) {
    std::set<IndexType> cached_indices;
    std::vector<int> counters(num_sets, 0);
    uint64_t cache_capacity = static_cast<uint64_t>(static_cast<double>(m_num_rows) * 0.15);
    while (cache_capacity > static_cast<uint64_t>(m_num_keys)) {
      cache_capacity /= 2;
    }
    uint32_t step = static_cast<uint32_t>(m_num_keys / cache_capacity);
    EXPECT_TRUE(step > 0);
    for (IndexType i = 0; i < m_num_keys; i += step) {
      IndexType idx = m_keys->ph[i];
      if (++counters[idx % num_sets] <= num_ways) {
        cached_indices.insert(idx);
      }
    }
    return std::vector<IndexType>(cached_indices.begin(), cached_indices.end());
  }

  void PopulateCache(const std::vector<IndexType>& cached_indices) {
    m_cache_ptr->modify_context_create(m_handle_modify, static_cast<uint32_t>(cached_indices.size()));
    nve::DefaultHistogram<IndexType> hist(cached_indices.data(), cached_indices.size(),
                                          m_table->ph, m_row_bytes, true);
    cudaEvent_t wait;
    CHECK_CUDA_ERROR(cudaEventCreate(&wait));
    nve::DefaultECEvent ec_event(std::vector<cudaStream_t>{});
    m_cache_ptr->insert(m_handle_modify, hist.get_keys(), hist.get_priority(), hist.get_data(),
                        hist.get_num_bins(), 0, &ec_event, m_stream);
    CHECK_CUDA_ERROR(cudaEventRecord(wait));
    CHECK_CUDA_ERROR(cudaEventSynchronize(wait));
  }

  void ComputeRefResults() {
    m_ref_result.assign(static_cast<size_t>(m_num_keys) * m_num_elements, from_float(0.0f));
    for (uint64_t i = 0; i < static_cast<uint64_t>(m_num_keys); i++) {
      // load_indices=true resolves the row from the keys buffer; false reads positionally (tid).
      IndexType idx = LOAD_INDICES ? m_keys->ph[i] : static_cast<IndexType>(i);
      const int8_t* row = m_table->ph + static_cast<uint64_t>(idx) * m_row_bytes;
      const StorageType* q = reinterpret_cast<const StorageType*>(row);
      const int8_t* meta = row + m_num_elements * sizeof(StorageType);
      ParamType scale{}, offset{};
      memcpy(static_cast<void*>(&scale), meta, sizeof(ParamType));
      if (HAS_OFFSET) memcpy(static_cast<void*>(&offset), meta + sizeof(ParamType), sizeof(ParamType));
      for (uint32_t el = 0; el < m_num_elements; el++) {
        m_ref_result[i * m_num_elements + el] = from_float(Dequantize(q[el], scale, offset));
      }
    }
  }

  void LaunchKernel(CacheDataType& cache_data) {
    m_result = std::make_shared<TestBuffer<ParamType>>(
        static_cast<size_t>(m_num_keys) * m_num_elements * sizeof(ParamType));
    const size_t stride = m_num_elements * sizeof(ParamType);
    CHECK_CUDA_ERROR((nve::call_find_and_dequant<IndexType, CacheDataType>(
        m_keys->pd, static_cast<size_t>(m_num_keys), reinterpret_cast<int8_t*>(m_result->pd),
        m_table->pd, VALUE_TYPE_ID, m_num_elements, cache_data, m_stream, stride,
        /*load_indices=*/LOAD_INDICES, /*curr_table=*/0)));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(m_stream));
  }

  void CheckResult() {
    m_result->DtoH(m_stream);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    // Single dequant per element (no accumulation): float matches the reference exactly bar
    // rounding; __half is limited by half granularity.
    const float tolerance = std::is_same_v<ParamType, __half> ? 1e-2f : 1e-4f;
    EXPECT_TRUE(std::equal(m_ref_result.begin(), m_ref_result.end(), m_result->ph, Near(tolerance)));
  }

  cudaStream_t m_stream;
  uint64_t m_num_rows{0};
  uint32_t m_num_elements{0};
  bool m_allocOnHost{false};
  uint64_t m_row_bytes{0};
  IndexType m_num_keys{0};

  std::shared_ptr<TestBuffer<int8_t>> m_table = nullptr;
  std::shared_ptr<TestBuffer<IndexType>> m_keys = nullptr;
  std::shared_ptr<TestBuffer<ParamType>> m_result = nullptr;
  std::vector<ParamType> m_ref_result;

  nve::DefaultAllocator m_cache_allocator;
  nve::Logger m_cache_logger;
  std::shared_ptr<CacheType> m_cache_ptr;
  nve::LookupContextHandle m_handle_lookup;
  nve::ModifyContextHandle m_handle_modify;
  SACacheData m_cache_data{};
};

TYPED_TEST_SUITE_P(FindAndDequantRefTest);

TYPED_TEST_P(FindAndDequantRefTest, DequantAgainstRefCpu) {
  // num_elements 132 / 30 / 63 select the Vec4 / Vec2 / Vec1 dequant paths (row layout % 4).
  // num_keys 1000 / 4099 cover a non-multiple-of-block tail.
  for (const auto num_keys : {1000, 4099}) {
    for (const auto num_elements : {132, 30, 63}) {
      for (const auto allocOnHost : {true, false}) {
        this->LaunchTest(262144, num_elements, num_keys, allocOnHost);
      }
    }
  }
}

REGISTER_TYPED_TEST_SUITE_P(FindAndDequantRefTest, DequantAgainstRefCpu);

// 4 quant dtypes x {set-associative cache, no-cache} x {load_indices true, false}.
// QInt8 storage is signed char; QUint8 storage is unsigned char (matching QuantizationHelper::Vec1).
typedef ::testing::Types<
    // QInt8RowwiseF32 (symmetric, pure float scale, no offset).
    DequantTestType<DataType_t::QInt8RowwiseF32,  char,          int32_t, /*cache=*/true,  /*load_idx=*/true>,
    DequantTestType<DataType_t::QInt8RowwiseF32,  char,          int64_t, /*cache=*/true,  /*load_idx=*/false>,
    DequantTestType<DataType_t::QInt8RowwiseF32,  char,          int32_t, /*cache=*/false, /*load_idx=*/true>,
    DequantTestType<DataType_t::QInt8RowwiseF32,  char,          int64_t, /*cache=*/false, /*load_idx=*/false>,
    // QUint8RowwiseF32 (min/max affine, float scale + offset).
    DequantTestType<DataType_t::QUint8RowwiseF32, unsigned char, int32_t, /*cache=*/true,  /*load_idx=*/true>,
    DequantTestType<DataType_t::QUint8RowwiseF32, unsigned char, int64_t, /*cache=*/true,  /*load_idx=*/false>,
    DequantTestType<DataType_t::QUint8RowwiseF32, unsigned char, int32_t, /*cache=*/false, /*load_idx=*/true>,
    DequantTestType<DataType_t::QUint8RowwiseF32, unsigned char, int64_t, /*cache=*/false, /*load_idx=*/false>,
    // QInt8RowwiseF16 (symmetric, pure-scale / no offset; __half scale + output).
    DequantTestType<DataType_t::QInt8RowwiseF16,  char,          int32_t, /*cache=*/true,  /*load_idx=*/true>,
    DequantTestType<DataType_t::QInt8RowwiseF16,  char,          int64_t, /*cache=*/true,  /*load_idx=*/false>,
    DequantTestType<DataType_t::QInt8RowwiseF16,  char,          int32_t, /*cache=*/false, /*load_idx=*/true>,
    DequantTestType<DataType_t::QInt8RowwiseF16,  char,          int64_t, /*cache=*/false, /*load_idx=*/false>,
    // QUint8RowwiseF16 (affine; __half scale + offset + output).
    DequantTestType<DataType_t::QUint8RowwiseF16, unsigned char, int32_t, /*cache=*/true,  /*load_idx=*/true>,
    DequantTestType<DataType_t::QUint8RowwiseF16, unsigned char, int64_t, /*cache=*/true,  /*load_idx=*/false>,
    DequantTestType<DataType_t::QUint8RowwiseF16, unsigned char, int32_t, /*cache=*/false, /*load_idx=*/true>,
    DequantTestType<DataType_t::QUint8RowwiseF16, unsigned char, int64_t, /*cache=*/false, /*load_idx=*/false>>
    DequantTypes;

INSTANTIATE_TYPED_TEST_SUITE_P(FindAndDequant,
                               FindAndDequantRefTest,
                               DequantTypes);
