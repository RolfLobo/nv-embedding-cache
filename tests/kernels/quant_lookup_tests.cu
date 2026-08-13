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
    call_find_and_combine_kernel<IndexType, CacheDataType>(
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
