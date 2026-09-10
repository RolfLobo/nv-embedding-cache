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

#include "emb_layer_utils.hpp"
#include <buffer_wrapper.hpp>
#include <embedding_layer.hpp>
#include <linear_embedding_layer.hpp>
#include <execution_context.hpp>
#include <gpu_table.hpp>
#include <host_table.hpp>
#include <key_utils.hpp>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <cuda_support.hpp>

#include "mock_host_table.hpp"
#include "cuda_ops/cuda_common.h"

namespace nve {

enum class PrivateStreamMode : uint64_t {
  None,
  Optimized,
  NonOptimized,
};

// Fixture for functional tests
struct UVMTestCase {
  int64_t uvm_table_size;
  int64_t gpu_table_size;
  int64_t row_size;
  size_t test_keys;
  DataType_t data_type; // for now using the same type for table and accumulate
  PrivateStreamMode private_stream_mode;
  bool modify_on_gpu;
  int64_t hotness = 1;
  PoolingType_t pooling_type = PoolingType_t::Concatenate;
  SparseType_t  offsets_layout = SparseType_t::Fixed;
};
class UVM : public ::testing::TestWithParam<UVMTestCase> {};
class UVMPooling : public ::testing::TestWithParam<UVMTestCase> {};

static bool IsCpu(const std::string cpu_name) {
  FILE* pipe = popen("lscpu|grep 'Model name:'|cut -d':' -f2|sed -e 's/^[[:space:]]*//'", "r");
  if (!pipe) {
    throw std::runtime_error("Failed to check CPU model (popen)");
  }
  char out[1024] = {0};
  if (fgets(out, 1024, pipe) == NULL) {
    throw std::runtime_error("Failed to check CPU model (fgets)");
  }
  auto ret = pclose(pipe);
  if (ret != 0) {
    throw std::runtime_error("Failed to check CPU model (pclose)");
  }

  std::string cpu_str(out);
  return (cpu_str.find(cpu_name) != std::string::npos);
}

static bool IsLargeTestAnd7xxx(const nve::UVMTestCase& tc) {
  static const bool is7xxx = IsCpu("AMD EPYC 7313P") || IsCpu("AMD EPYC 7232P"); // static so we only check this once instead of every test
  return (is7xxx && (tc.uvm_table_size > int64_t(1) << 30));
}

template <typename IndexT>
class UVMLayerTest {
 public:
  using layer_type = LinearUVMEmbeddingLayer<IndexT>;
  static constexpr int64_t MAX_MODIFY_SIZE = (1l << 20);

  UVMLayerTest( int64_t row_size, int64_t gpu_table_size, int64_t uvm_table_size, DataType_t data_type,
                PrivateStreamMode private_stream_mode, bool modify_on_gpu, int device_id, bool insert_heuristic = false,
                std::function<void(GPUTableConfig&)> gpu_cfg_overwrite = nullptr,
                size_t seed = 31337)
    : row_size_(row_size), max_rows_(uvm_table_size / row_size), device_id_(device_id) {
    ScopedDevice dev(device_id);
    // init linear table
    {
      NVE_CHECK_(cudaMallocHost(&linear_table_, static_cast<size_t>(row_size_ * max_rows_)));
      NVE_CHECK_(static_cast<size_t>(row_size_) % sizeof(float) == 0); // cannot use ASSERT_EQ in c'tor

      // Init table values with multiple threads to save on test time
      std::vector<std::shared_ptr<std::thread>> input_gen_threads;
      const int64_t num_threads = static_cast<int64_t>(std::thread::hardware_concurrency());
      for (int64_t t=0 ; t<num_threads ; t++) {
        auto start_row = t * max_rows_ / num_threads;
        auto end_row = std::min<int64_t>((t + 1) * max_rows_ / num_threads, max_rows_);

        input_gen_threads.push_back(std::make_shared<std::thread>(
          InitTableRows, linear_table_, row_size_, start_row, end_row, data_type, seed + static_cast<size_t>(t)
        ));
      }
      for (auto& t : input_gen_threads) {
          t->join();
      }
    }
    // init private stream
    if (private_stream_mode != PrivateStreamMode::None) {
      NVE_CHECK_(cudaStreamCreate(&private_stream));
    }

    // init gpu table
    {
      GPUTableConfig cfg;
      cfg.device_id = device_id;
      cfg.cache_size = static_cast<size_t>(gpu_table_size);
      cfg.max_modify_size = MAX_MODIFY_SIZE;
      cfg.row_size_in_bytes = row_size;
      cfg.uvm_table = linear_table_;
      cfg.uvm_num_rows = max_rows_;
      cfg.count_misses = true;
      cfg.value_dtype = data_type;
      cfg.private_stream = private_stream;
      cfg.modify_on_gpu = modify_on_gpu;
      if (gpu_cfg_overwrite != nullptr) {
        gpu_cfg_overwrite(cfg);
      }
      gpu_tab_ = std::make_shared<GpuTable<IndexT>>(cfg);
    }

    // init layer
    {
      typename LinearUVMEmbeddingLayer<IndexT>::Config cfg;
      if (insert_heuristic) {
        cfg.insert_heuristic = std::make_shared<TestInsertHeuristic>();
        cfg.min_insert_freq_gpu = 0;
        cfg.min_insert_size_gpu = 1024;
      } else {
        cfg.insert_heuristic = std::make_shared<NeverInsertHeuristic>();
      }
      layer_ = std::make_shared<LinearUVMEmbeddingLayer<IndexT>>(cfg, gpu_tab_);
    }
    // init context
    cudaStream_t ctx_stream = (private_stream_mode == PrivateStreamMode::Optimized) ? private_stream : 0;
    ctx_ = layer_->create_execution_context(ctx_stream, ctx_stream, nullptr, nullptr);

    // create ref (using single mock table)
    {
      HostTableConfig mock_cfg;
      mock_cfg.value_dtype = data_type;
      mock_cfg.max_value_size = row_size_;
      ref_tab_ = std::make_shared<MockHostTable<IndexT>>(mock_cfg, true /*functional_ref*/);
    }
  }

  UVMLayerTest(UVMTestCase tc, int device_id = 0) : UVMLayerTest<IndexT>(tc.row_size, tc.gpu_table_size, tc.uvm_table_size, tc.data_type, tc.private_stream_mode, tc.modify_on_gpu, device_id) {}
  ~UVMLayerTest() {
    ScopedDevice dev(device_id_);
    ctx_->wait();
    ctx_.reset();  // explicitly destroy the context before the stream since the context synchronizes on the stream during d'tor
                    // (which may be the private stream about to be destroyed)
    if (private_stream) {
      NVE_CHECK_(cudaStreamSynchronize(private_stream)); // Make sure all work for the stream is complete before destruction
      NVE_CHECK_(cudaStreamDestroy(private_stream));
    }
    NVE_CHECK_(cudaFreeHost(linear_table_));
  }

  void LookupAndCheck(std::vector<IndexT>& keys, uint64_t start_key = 0,
                      uint64_t end_key = uint64_t(-1), float hit_tol = 0.01f, float* hitrate = nullptr) {
    if (keys.empty()) {
      return;
    }
    SetupKeys setup(keys, start_key, end_key);
    auto num_keys = setup.num_keys;
    auto keys_buffer = setup.keys_buffer;

    auto output_size = static_cast<size_t>(num_keys * row_size_);
    int8_t* output{nullptr};
    NVE_CHECK_(cudaMallocHost(&output, output_size));
    NVE_CHECK_(output != 0);
    std::vector<int8_t> ref_output(output_size);

    std::vector<bitmask64_t> ref_hitmask(to_uint(ceil_div(num_keys, bitmask64::num_bits)));

    std::vector<float> hitrates(3);

    layer_->lookup(ctx_, num_keys, keys_buffer, output, row_size_, nullptr /*hitmask*/,
                    nullptr /*pool_params*/, hitrates.data());
    NVE_CHECK_(cudaDeviceSynchronize());
    int64_t ref_hits;
    ref_tab_->reset_lookup_counter(ctx_);
    {
      auto keys_bw = std::make_shared<BufferWrapper<const void>>(
          ctx_, "keys", keys_buffer, static_cast<size_t>(num_keys) * sizeof(IndexT));
      auto hit_mask_bw = std::make_shared<BufferWrapper<bitmask64_t>>(
          ctx_, "hit_mask", ref_hitmask.data(), ref_hitmask.size() * sizeof(bitmask64_t));
      auto values_bw = std::make_shared<BufferWrapper<void>>(
          ctx_, "values", ref_output.data(),
          static_cast<size_t>(num_keys) * static_cast<size_t>(row_size_));
      ref_tab_->find(ctx_, num_keys, std::move(keys_bw), std::move(hit_mask_bw), row_size_,
                         std::move(values_bw), nullptr /*value_sizes*/);
    }
    ref_tab_->get_lookup_counter(ctx_, &ref_hits);


    // compare hitrates
    ASSERT_NEAR(hitrates[0], float(ref_hits) / float(num_keys), hit_tol);

    // compare outputs for hits
    for (int64_t i = 0; i < num_keys; i++) {
      bool ref_hit{bitmask64::get(ref_hitmask.at(to_uint(i / bitmask64::num_bits)), i % bitmask64::num_bits)};
      const int8_t* ref_vec = ref_hit ? ref_output.data() + (i * row_size_) : linear_table_ + (keys.at(static_cast<size_t>(i)) * row_size_);
      for (int64_t j = 0; j < row_size_; j++) {
        ASSERT_EQ(output[(i * row_size_) + j], *(ref_vec + j));
      }
    }
    NVE_CHECK_(cudaFreeHost(output));
    if (hitrate != nullptr) {
      *hitrate = hitrates[0];
    }
  }

  void LookupAndCheckPooling(const UVMTestCase& tc, std::vector<IndexT>& keys,
                             uint64_t start_key = 0, uint64_t end_key = uint64_t(-1)) {
    if (keys.empty()) {
      return;
    }
    SetupKeys setup(keys, start_key, end_key);
    auto num_keys = setup.num_keys;
    auto keys_buffer = setup.keys_buffer;
    auto output_bags = num_keys;

    EmbeddingLayerBase::PoolingParams pp;
    pp.pooling_type = tc.pooling_type;
    pp.sparse_type = tc.offsets_layout;
    // Output dtype matches the table's stored value type (the layer derives the actual output type
    // from config().value_dtype); set it so validate_pool_params() accepts the request.
    pp.output_type = tc.data_type;

    SetupCSROffsets<int64_t> offsets_setup(
        tc.offsets_layout == SparseType_t::CSR ? static_cast<int64_t>(num_keys) : 1,
        std::max<int64_t>(tc.hotness, 1));
    const int64_t fixed_hotness = (num_keys == 1) ? 1 : tc.hotness;
    if (tc.offsets_layout == SparseType_t::CSR) {
      pp.csr_offsets = offsets_setup.offsets_buffer;
      pp.num_csr_offsets = static_cast<int64_t>(offsets_setup.num_offsets);
      output_bags = static_cast<int64_t>(offsets_setup.num_offsets) - 1;
    } else {
      pp.fixed_hotness = fixed_hotness;
      if (tc.pooling_type != PoolingType_t::Concatenate) {
        output_bags = num_keys / fixed_hotness;
      }
    }

    std::vector<int8_t> weights;
    if (is_weighted_pooling(tc.pooling_type)) {
      GenerateWeights(weights, static_cast<uint64_t>(num_keys), tc.data_type);
      pp.weights = weights.data();
      pp.weight_type = tc.data_type;
    } else {
      pp.weights = nullptr;
    }

    // Allocate based on num_keys (matches the BufferWrapper size used by the layer internally),
    // even though only output_bags * row_size_ bytes are written for Sum/Mean pooling.
    auto output_size = static_cast<size_t>(num_keys * row_size_);
    int8_t* output{nullptr};
    NVE_CHECK_(cudaMallocHost(&output, output_size));
    NVE_CHECK_(output != 0);

    std::vector<float> hitrates(3);

    layer_->lookup(ctx_, num_keys, keys_buffer, output, row_size_, nullptr /*hitmask*/,
                    &pp /*pool_params*/, hitrates.data());

    std::vector<int8_t> find_output(static_cast<size_t>(num_keys * row_size_));
    {
      auto keys_bw = std::make_shared<BufferWrapper<const void>>(
          ctx_, "keys", keys_buffer, static_cast<size_t>(num_keys) * sizeof(IndexT));
      auto values_bw = std::make_shared<BufferWrapper<void>>(
          ctx_, "values", find_output.data(),
          static_cast<size_t>(num_keys) * static_cast<size_t>(row_size_));
      ref_tab_->find(ctx_, num_keys, std::move(keys_bw), nullptr /* hitmask */, row_size_,
                      std::move(values_bw), nullptr /*value_sizes*/);
    }

    std::vector<int8_t> ref_output(static_cast<size_t>(output_bags * row_size_));
    auto* mock_ref = static_cast<MockHostTable<IndexT>*>(ref_tab_.get());
    mock_ref->combine(find_output.data(), num_keys, pp.pooling_type, pp.sparse_type,
                      pp.csr_offsets, pp.num_csr_offsets, pp.fixed_hotness, pp.weights,
                      pp.weight_type, ref_output.data());

    NVE_CHECK_(cudaDeviceSynchronize());

    float tolerance = 0.f;
    if (tc.pooling_type != PoolingType_t::Concatenate) {
      switch (tc.data_type) {
        case DataType_t::Float32:
          tolerance = 1e-5f;
          break;
        case DataType_t::Float16:
          // The UVM find_and_combine kernel uses an fp16 accumulator, so summing `hotness` random
          // fp16 values in [0,1) yields a result up to ~hotness with fp16 ulp ~hotness/1024.
          // Scale tolerance with hotness so Sum/WeightedSum at hotness>1 stays within ~1 ulp of the
          // accumulated magnitude; Mean/WeightedMean divides by the count so a smaller floor suffices.
          tolerance = (tc.pooling_type == PoolingType_t::Mean ||
                       tc.pooling_type == PoolingType_t::WeightedMean)
                          ? 5e-3f
                          : 5e-3f * static_cast<float>(std::max<int64_t>(tc.hotness, 1));
          break;
        default:
          throw std::runtime_error("Invalid datatype");
      }
    }

    const int64_t output_elements = static_cast<int64_t>(ref_output.size()) / dtype_size(tc.data_type);
    for (int64_t i = 0; i < output_elements; i++) {
      ASSERT_NEAR(
        load_as_float(output, i, tc.data_type),
        load_as_float(ref_output.data(), i, tc.data_type),
        tolerance
      );
    }
    NVE_CHECK_(cudaFreeHost(output));
  }

  void Insert(std::vector<IndexT>& keys, std::vector<uint8_t>& datavectors, uint64_t start_key = 0, uint64_t end_key = uint64_t(-1)) {
    if (keys.empty()) {
      return;
    }
    SetupKeys setup(keys, start_key, end_key, datavectors, row_size_);
    auto num_keys = setup.num_keys;
    auto keys_buffer = setup.keys_buffer;
    auto data_buffer = setup.data_buffer;

    // "insert" to the linear table
    for (int64_t i=0 ; i<num_keys ; i++) {
      auto pos = start_key + static_cast<size_t>(i);
      auto key = keys[pos];
      auto row_size = static_cast<size_t>(row_size_);
      ASSERT_LE(key, max_rows_); // Not handling keys too big for the linear table
      
      std::memcpy(linear_table_ + (key * row_size_), datavectors.data() + (pos * row_size), row_size);
    }

    layer_->insert(ctx_, num_keys, keys_buffer, row_size_, row_size_, data_buffer, 0);
    bool ref_insert = (gpu_tab_ != nullptr);
    if (ref_insert) {
      auto keys_bw = std::make_shared<BufferWrapper<const void>>(
          ctx_, "keys", keys_buffer, static_cast<size_t>(num_keys) * sizeof(IndexT));
      auto values_bw = std::make_shared<BufferWrapper<const void>>(
          ctx_, "values", data_buffer,
          static_cast<size_t>(num_keys) * static_cast<size_t>(row_size_));
      ref_tab_->insert(ctx_, num_keys, std::move(keys_bw), row_size_, row_size_,
                           std::move(values_bw));
    }
  }

  void Update(std::vector<IndexT>& keys, std::vector<uint8_t>& datavectors, uint64_t start_key = 0,
              uint64_t end_key = uint64_t(-1)) {
    if (keys.empty()) {
      return;
    }
    SetupKeys setup(keys, start_key, end_key, datavectors, row_size_);
    auto num_keys = setup.num_keys;
    auto keys_buffer = setup.keys_buffer;
    auto data_buffer = setup.data_buffer;

    layer_->update(ctx_, num_keys, keys_buffer, row_size_, row_size_, data_buffer, -1);
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(
        ctx_, "keys", keys_buffer, static_cast<size_t>(num_keys) * sizeof(IndexT));
    auto values_bw = std::make_shared<BufferWrapper<const void>>(
        ctx_, "values", data_buffer,
        static_cast<size_t>(num_keys) * static_cast<size_t>(row_size_));
    ref_tab_->update(ctx_, num_keys, std::move(keys_bw), row_size_, row_size_,
                         std::move(values_bw));
  }

  void Accumulate(std::vector<IndexT>& keys, std::vector<uint8_t>& datavectors,
                  DataType_t value_type, uint64_t start_key = 0, uint64_t end_key = uint64_t(-1)) {
    if (keys.empty()) {
      return;
    }
    SetupKeys setup(keys, start_key, end_key, datavectors, row_size_);
    auto num_keys = setup.num_keys;
    auto keys_buffer = setup.keys_buffer;
    auto data_buffer = setup.data_buffer;

    layer_->accumulate(ctx_, num_keys, keys_buffer, row_size_, row_size_, data_buffer, value_type, -1);
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(
        ctx_, "keys", keys_buffer, static_cast<size_t>(num_keys) * sizeof(IndexT));
    auto updates_bw = std::make_shared<BufferWrapper<const void>>(
        ctx_, "updates", data_buffer,
        static_cast<size_t>(num_keys) * static_cast<size_t>(row_size_));
    ref_tab_->update_accumulate(ctx_, num_keys, std::move(keys_bw), row_size_, row_size_,
                                    std::move(updates_bw), value_type);
  }

  void Erase(std::vector<IndexT>& keys, uint64_t start_key = 0, uint64_t end_key = uint64_t(-1)) {
    if (keys.empty()) {
      return;
    }
    SetupKeys setup(keys, start_key, end_key);
    auto num_keys = setup.num_keys;
    auto keys_buffer = setup.keys_buffer;
    layer_->erase(ctx_, num_keys, keys_buffer, 0);
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(
        ctx_, "keys", keys_buffer, static_cast<size_t>(num_keys) * sizeof(IndexT));
    ref_tab_->erase(ctx_, num_keys, std::move(keys_bw));
  }

  void Clear(bool clear_ref = true) {
    layer_->clear(ctx_);
    if (clear_ref) {
      ref_tab_->clear(ctx_);
    }
  }

  void Wait() { ctx_->wait(); }

  typename LinearUVMEmbeddingLayer<IndexT>::Config Config() const {
    return layer_->get_config();
  }

  // Direct access for tests that call the layer / reference with non-default arguments.
  layer_type& layer() { return *layer_; }
  context_ptr_t& ctx() { return ctx_; }
  HostTableLike& ref() { return *ref_tab_; }

 public:
  const int64_t row_size_;
  const int64_t max_rows_;
 private:
  typename layer_type::gpu_table_ptr_t gpu_tab_{nullptr};
  host_table_ptr_t ref_tab_;
  int8_t* linear_table_{nullptr};
  std::shared_ptr<layer_type> layer_;
  context_ptr_t ctx_;
  cudaStream_t private_stream{0};
  int device_id_;
};

// [Sanity] Init the layer
TEST_P(UVM, Init) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto& tc = GetParam();
  if (IsLargeTestAnd7xxx(tc)) {
    GTEST_SKIP() << "Skipping large UVM test";
    return;
  }
  UVMLayerTest<int64_t> ult(tc);
}

// [Sanity] insert 1key, lookup 1key
TEST_P(UVM, SingleLookup) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto& tc = GetParam();
  if (IsLargeTestAnd7xxx(tc)) {
    GTEST_SKIP() << "Skipping large UVM test";
    return;
  }
  UVMLayerTest<int64_t> ult(tc);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, 1, tc.row_size, 0, ult.max_rows_, tc.data_type);
  ult.Insert(keys, data);
  ult.LookupAndCheck(keys);
}

// [Lookup] insert k, lookup 2k, get k hits, k misses (per table, k small enough to avoid partial
// inserts), clear, lookup
TEST_P(UVM, Lookup) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto& tc = GetParam();
  if (IsLargeTestAnd7xxx(tc)) {
    GTEST_SKIP() << "Skipping large UVM test";
    return;
  }
  UVMLayerTest<int64_t> ult(tc);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, ult.max_rows_, tc.data_type);
  ult.Insert(keys, data, 0, tc.test_keys);
  NVE_CHECK_(cudaDeviceSynchronize());
  ult.LookupAndCheck(keys);
  ult.Clear();
  NVE_CHECK_(cudaDeviceSynchronize());
  ult.LookupAndCheck(keys);
}

// [Update] insert k, update 2k, lookup 2k,...
TEST_P(UVM, Update) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto& tc = GetParam();
  if (IsLargeTestAnd7xxx(tc)) {
    GTEST_SKIP() << "Skipping large UVM test";
    return;
  }
  UVMLayerTest<int64_t> ult(tc);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, ult.max_rows_, tc.data_type);
  ult.Insert(keys, data, 0, tc.test_keys / 2); // insert half so the rest will be read from UVM
  NVE_CHECK_(cudaDeviceSynchronize());

  // change some data so there's something to update
  const auto num_rows = static_cast<int64_t>(data.size()) / tc.row_size;
  for (int64_t i = 0; i < num_rows; i+=2) {  // only updating half the rows
    for (int64_t j = 0; j < tc.row_size; j++) {
      data.at(static_cast<size_t>((i * tc.row_size) + j)) *= 3;
    }
  }

  ult.Update(keys, data);
  NVE_CHECK_(cudaDeviceSynchronize());
  ult.LookupAndCheck(keys);
}

// [Insert] insert k, lookup k+k`, insert k`, lookup k+k`
TEST_P(UVM, Insert) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto& tc = GetParam();
  if (IsLargeTestAnd7xxx(tc)) {
    GTEST_SKIP() << "Skipping large UVM test";
    return;
  }
  UVMLayerTest<int64_t> ult(tc);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, ult.max_rows_, tc.data_type);
  ult.Insert(keys, data, 0, tc.test_keys / 6);
  NVE_CHECK_(cudaDeviceSynchronize());
  ult.LookupAndCheck(keys);

  ult.Insert(keys, data, tc.test_keys * 3 / 6, tc.test_keys * 4 / 6);
  NVE_CHECK_(cudaDeviceSynchronize());
  ult.LookupAndCheck(keys);
}

// [Accumulate] insert k+k`, accumulate k+k``
TEST_P(UVM, Accumulate) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto& tc = GetParam();
  if (IsLargeTestAnd7xxx(tc)) {
    GTEST_SKIP() << "Skipping large UVM test";
    return;
  }
  UVMLayerTest<int64_t> ult(tc);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, ult.max_rows_, tc.data_type);
  ult.Insert(keys, data, 0, tc.test_keys / 2);
  NVE_CHECK_(cudaDeviceSynchronize());
  ult.LookupAndCheck(keys);

  // Accumulate some keys that were inserted and some that weren't (reusing original data as
  // gradients)
  ult.Accumulate(keys, data, tc.data_type, tc.test_keys / 16, tc.test_keys * 3 / 16);
  ult.Accumulate(keys, data, tc.data_type, tc.test_keys * 5 / 16, tc.test_keys * 12 / 16);

  NVE_CHECK_(cudaDeviceSynchronize());
  ult.LookupAndCheck(keys);
}

//  [Erase] inserk k+k`, lookup k+k`+k``, erase k, lookup k+k`+k``
TEST_P(UVM, Erase) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto& tc = GetParam();
  if (IsLargeTestAnd7xxx(tc)) {
    GTEST_SKIP() << "Skipping large UVM test";
    return;
  }
  UVMLayerTest<int64_t> ult(tc);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, ult.max_rows_, tc.data_type);
  ult.Insert(keys, data, 0, tc.test_keys / 2);
  NVE_CHECK_(cudaDeviceSynchronize());
  ult.LookupAndCheck(keys);

  ult.Erase(keys, tc.test_keys * 2 / 6, tc.test_keys * 3 / 6);
  ult.Erase(keys, tc.test_keys * 4 / 6, tc.test_keys * 5 / 6);

  NVE_CHECK_(cudaDeviceSynchronize());
  ult.LookupAndCheck(keys);
}

TEST_P(UVM, LookupNonDefaultDevice) {
  cudaGetLastError();  // Clear potential errors left by previous tests.

  // check we have at least 2 devices
  int num_devices = 0;
  NVE_CHECK_(cudaGetDeviceCount(&num_devices));
  if (num_devices < 2) {
    GTEST_SKIP() << "Skipping multi-GPU test since only " << num_devices << " device(s) found";
    return;
  }
  const auto& tc = GetParam();
  if (IsLargeTestAnd7xxx(tc)) {
    GTEST_SKIP() << "Skipping large UVM test";
    return;
  }
  UVMLayerTest<int64_t> ult(tc, 1 /*device_id*/);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, ult.max_rows_, tc.data_type);
  ult.Insert(keys, data, 0, tc.test_keys / 2);
  NVE_CHECK_(cudaDeviceSynchronize());
  ult.LookupAndCheck(keys);
}

TEST_P(UVM, LookupMultiDevice) {
  cudaGetLastError();  // Clear potential errors left by previous tests.

  // check we have at least 2 devices
  int num_devices = 0;
  NVE_CHECK_(cudaGetDeviceCount(&num_devices));
  if (num_devices < 2) {
    GTEST_SKIP() << "Skipping multi-GPU test since only " << num_devices << " device(s) found";
    return;
  }

  const auto& tc = GetParam();
  if (IsLargeTestAnd7xxx(tc)) {
    GTEST_SKIP() << "Skipping large UVM test";
    return;
  }
  UVMLayerTest<int64_t> ult_0(tc, 0 /*device_id*/);
  UVMLayerTest<int64_t> ult_1(tc, 1 /*device_id*/);
  std::thread thread_0([&ult_0, tc]() {
      ScopedDevice dev_0(0);
      for (int i = 0; i < 10; i++) {
        std::vector<int64_t> keys;
        std::vector<uint8_t> data;
        GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, ult_0.max_rows_, tc.data_type);
        ult_0.Insert(keys, data, 0, tc.test_keys / 2);
        NVE_CHECK_(cudaDeviceSynchronize());
        ult_0.LookupAndCheck(keys);
      }
    });
  std::thread thread_1([&ult_1, tc]() {
      ScopedDevice dev_1(1);    
      for (int i = 0; i < 10; i++) {
        std::vector<int64_t> keys;
        std::vector<uint8_t> data;
        GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, ult_1.max_rows_, tc.data_type);
        ult_1.Insert(keys, data, 0, tc.test_keys / 2);
        NVE_CHECK_(cudaDeviceSynchronize());
        ult_1.LookupAndCheck(keys);
      }
    });
  thread_0.join();
  thread_1.join();
}

INSTANTIATE_TEST_SUITE_P(
    EmbLayer,
    UVM,
    ::testing::Values(
        //  TestCase: uvm_table_size,   gpu_table_Size,   row_size,         test_keys         data_type            private_stream                   modify_on_gpu
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 20, int64_t(1) << 10, int64_t(1) << 0,  DataType_t::Float16, PrivateStreamMode::None,         false}),  // Single key
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 20, int64_t(1) << 10, int64_t(1) << 0,  DataType_t::Float32, PrivateStreamMode::None,         false}),  // Single key
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float16, PrivateStreamMode::None,         false}), // Medium
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float32, PrivateStreamMode::None,         false}), // Medium
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float16, PrivateStreamMode::NonOptimized, false}), // Medium, private stream non optimized
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float32, PrivateStreamMode::NonOptimized, false}), // Medium, private stream non optimized
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float16, PrivateStreamMode::Optimized,    false}), // Medium, private stream optimized
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float32, PrivateStreamMode::Optimized,    false}), // Medium, private stream optimized
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 20, int64_t(1) << 10, int64_t(1) << 0,  DataType_t::Float16, PrivateStreamMode::None,         true}),  // Single key
        // Reduced set for modify_on_gpu
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 20, int64_t(1) << 10, int64_t(1) << 0,  DataType_t::Float16, PrivateStreamMode::None,         true}),  // Single key
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 20, int64_t(1) << 10, int64_t(1) << 0,  DataType_t::Float32, PrivateStreamMode::None,         true}),  // Single key
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float32, PrivateStreamMode::None,         true}), // Medium
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float32, PrivateStreamMode::Optimized,    true}) // Medium, private stream optimized
      ));

INSTANTIATE_TEST_SUITE_P(
    EmbLayer_Large,
    UVM,
    ::testing::Values(
        //  TestCase: uvm_table_size,   gpu_table_Size,   row_size,         test_keys         data_Type            private_stream           modify_on_gpu
        UVMTestCase({ int64_t(1) << 32, int64_t(1) << 30, int64_t(1) << 10, int64_t(1) << 16, DataType_t::Float16, PrivateStreamMode::None, false}),  // Large 
        UVMTestCase({ int64_t(1) << 32, int64_t(1) << 30, int64_t(1) << 10, int64_t(1) << 16, DataType_t::Float32, PrivateStreamMode::None, false})  // Large 
      ));

// [Pooling - Sanity] insert 1 key, lookup 1 key with pooling
TEST_P(UVMPooling, SingleLookup) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto& tc = GetParam();
  if (IsLargeTestAnd7xxx(tc)) {
    GTEST_SKIP() << "Skipping large UVM test";
    return;
  }
  UVMLayerTest<int64_t> ult(tc);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, 1, tc.row_size, 0, ult.max_rows_, tc.data_type);
  ult.Insert(keys, data);
  NVE_CHECK_(cudaDeviceSynchronize());
  ult.LookupAndCheckPooling(tc, keys);
}

// [Pooling] insert k, lookup k with pooling and compare against reference combine
TEST_P(UVMPooling, Lookup) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto& tc = GetParam();
  if (IsLargeTestAnd7xxx(tc)) {
    GTEST_SKIP() << "Skipping large UVM test";
    return;
  }
  UVMLayerTest<int64_t> ult(tc);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  // For Fixed offsets layout, ensure test_keys is divisible by hotness
  size_t num_keys = tc.test_keys;
  if (tc.offsets_layout == SparseType_t::Fixed && tc.hotness > 0) {
    auto h = static_cast<size_t>(tc.hotness);
    num_keys = (num_keys / h) * h;
    if (num_keys == 0) {
      num_keys = h;
    }
  }

  GenerateData<int64_t>(keys, data, num_keys, tc.row_size, 0, ult.max_rows_, tc.data_type);
  ult.Insert(keys, data);
  NVE_CHECK_(cudaDeviceSynchronize());
  ult.LookupAndCheckPooling(tc, keys);
}

// Note: PoolingType_t::Concatenate is not supported by the GpuTable::find_and_combine path used by
// LinearUVMEmbeddingLayer (NVE_THROW_NOT_IMPLEMENTED_), so it is not exercised here.
INSTANTIATE_TEST_SUITE_P(
    EmbLayer,
    UVMPooling,
    ::testing::Values(
        //  TestCase: uvm_table_size,   gpu_table_Size,   row_size,         test_keys         data_type            private_stream                modify_on_gpu  hotness  pooling_type                    offsets_layout
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float32, PrivateStreamMode::None,      false,         32,      PoolingType_t::Sum,             SparseType_t::Fixed}),  // fp32 fixed sum
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float32, PrivateStreamMode::None,      false,         32,      PoolingType_t::Sum,             SparseType_t::CSR}),    // fp32 CSR sum
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float32, PrivateStreamMode::None,      false,         32,      PoolingType_t::Mean,            SparseType_t::Fixed}),  // fp32 fixed mean
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float32, PrivateStreamMode::None,      false,         32,      PoolingType_t::Mean,            SparseType_t::CSR}),    // fp32 CSR mean
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float32, PrivateStreamMode::None,      false,         32,      PoolingType_t::WeightedSum,     SparseType_t::Fixed}),  // fp32 fixed weighted sum
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float32, PrivateStreamMode::None,      false,         32,      PoolingType_t::WeightedSum,     SparseType_t::CSR}),    // fp32 CSR weighted sum
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float32, PrivateStreamMode::None,      false,         32,      PoolingType_t::WeightedMean,    SparseType_t::Fixed}),  // fp32 fixed weighted mean
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float32, PrivateStreamMode::None,      false,         32,      PoolingType_t::WeightedMean,    SparseType_t::CSR}),    // fp32 CSR weighted mean

        // fp16 coverage
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float16, PrivateStreamMode::None,      false,         32,      PoolingType_t::Sum,             SparseType_t::Fixed}),  // fp16 fixed sum
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float16, PrivateStreamMode::None,      false,         32,      PoolingType_t::Sum,             SparseType_t::CSR}),    // fp16 CSR sum
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float16, PrivateStreamMode::None,      false,         32,      PoolingType_t::Mean,            SparseType_t::Fixed}),  // fp16 fixed mean
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float16, PrivateStreamMode::None,      false,         32,      PoolingType_t::Mean,            SparseType_t::CSR}),    // fp16 CSR mean
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float16, PrivateStreamMode::None,      false,         32,      PoolingType_t::WeightedSum,     SparseType_t::Fixed}),  // fp16 fixed weighted sum
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float16, PrivateStreamMode::None,      false,         32,      PoolingType_t::WeightedSum,     SparseType_t::CSR}),    // fp16 CSR weighted sum
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float16, PrivateStreamMode::None,      false,         32,      PoolingType_t::WeightedMean,    SparseType_t::Fixed}),  // fp16 fixed weighted mean
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float16, PrivateStreamMode::None,      false,         32,      PoolingType_t::WeightedMean,    SparseType_t::CSR})     // fp16 CSR weighted mean
      ));

// ---------------------------------------------------------------------------
// Rowwise-quantized lookup through LinearUVMEmbeddingLayer.
//
// Builds a UVM table of per-row quantized rows and drives the layer's lookup() with pool_params:
//   * pooling_type == Concatenate  -> routes to GpuTable::find_and_dequant (one dequantized row per
//     key, no pooling), the path piped in for this work.
//   * pooling_type == Sum (Fixed)  -> routes to find_and_combine, which dequantizes each row before
//     summing the bag.
// The dequantized float/half output is checked against a CPU reference computed straight from the
// quantized table via load_quant_row_element_as_float. Keys are never inserted (NeverInsert), so
// every lookup resolves through the UVM table.
// ---------------------------------------------------------------------------
template <typename IndexT>
static void RunQuantUVMLookup(DataType_t dtype, PoolingType_t pooling, int64_t hotness,
                              bool raw_concatenate = false) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const bool fp32_meta = (dtype == DataType_t::QInt8RowwiseF32 || dtype == DataType_t::QUint8RowwiseF32);
  const bool has_offset = (dtype == DataType_t::QUint8RowwiseF32 || dtype == DataType_t::QUint8RowwiseF16);
  const DataType_t out_dt =
      raw_concatenate ? dtype : (fp32_meta ? DataType_t::Float32 : DataType_t::Float16);
  const int64_t param_size = fp32_meta ? static_cast<int64_t>(sizeof(float)) : 2;  // float vs __half
  const bool concat = (pooling == PoolingType_t::Concatenate);

  constexpr int device_id = 0;
  ScopedDevice dev(device_id);

  constexpr int64_t value_count = 64;  // % 4 == 0 -> char4 (Vec4) dequant path
  int64_t row_bytes = value_count + (has_offset ? 2 : 1) * param_size;
  row_bytes = (row_bytes + 3) & ~static_cast<int64_t>(3);  // pad stride for char4 alignment
  constexpr int64_t num_rows = 4096;

  int8_t* h_table = nullptr;
  NVE_CHECK_(cudaMallocHost(&h_table, static_cast<size_t>(num_rows * row_bytes)));
  InitTableRowsQuant(h_table, row_bytes, value_count, 0, num_rows, dtype, 4242);

  GPUTableConfig cfg;
  cfg.device_id = device_id;
  cfg.cache_size = 1l << 20;
  cfg.row_size_in_bytes = row_bytes;
  cfg.value_dtype = dtype;
  cfg.uvm_table = h_table;
  cfg.uvm_num_rows = num_rows;
  cfg.count_misses = true;
  auto gpu_tab = std::make_shared<GpuTable<IndexT>>(cfg);

  typename LinearUVMEmbeddingLayer<IndexT>::Config lcfg;
  lcfg.insert_heuristic = std::make_shared<NeverInsertHeuristic>();
  auto layer = std::make_shared<LinearUVMEmbeddingLayer<IndexT>>(lcfg, gpu_tab);
  auto ctx = layer->create_execution_context(0, 0, nullptr, nullptr);

  const int64_t num_keys = concat ? 100 : (hotness * 16);
  std::vector<IndexT> keys(static_cast<size_t>(num_keys));
  std::mt19937 gen(987);
  std::uniform_int_distribution<int64_t> kd(0, num_rows - 1);
  for (auto& k : keys) k = static_cast<IndexT>(kd(gen));

  const int64_t out_stride = raw_concatenate ? row_bytes : value_count * param_size;

  EmbeddingLayerBase::PoolingParams pp;
  pp.pooling_type = pooling;
  pp.sparse_type = SparseType_t::Fixed;
  pp.output_type = out_dt;
  pp.fixed_hotness = concat ? int64_t(1) : hotness;
  const int64_t output_bags = concat ? num_keys : (num_keys / hotness);

  // Allocate exactly the number of rows produced by the pooling layout.
  int8_t* output = nullptr;
  const size_t output_bytes = static_cast<size_t>(output_bags * out_stride);
  NVE_CHECK_(cudaMallocHost(&output, output_bytes));
  std::memset(output, 0, output_bytes);

  std::vector<float> hitrates(3);
  layer->lookup(ctx, num_keys, keys.data(), output, out_stride, nullptr /*hitmask*/, &pp, hitrates.data());
  NVE_CHECK_(cudaDeviceSynchronize());

  const float tol = fp32_meta ? 1e-3f
                              : (concat ? 2e-2f : 1e-2f * static_cast<float>(std::max<int64_t>(hotness, 1)));
  if (raw_concatenate) {
    for (int64_t i = 0; i < num_keys; ++i) {
      const int8_t* expected =
          h_table + static_cast<int64_t>(keys[static_cast<size_t>(i)]) * row_bytes;
      EXPECT_EQ(std::memcmp(output + i * out_stride, expected, static_cast<size_t>(row_bytes)), 0)
          << "raw row mismatch at key slot " << i;
    }
  } else if (concat) {
    for (int64_t i = 0; i < num_keys; ++i) {
      const int8_t* row = h_table + static_cast<int64_t>(keys[static_cast<size_t>(i)]) * row_bytes;
      for (int64_t e = 0; e < value_count; ++e) {
        const float ref = load_quant_row_element_as_float(row, e, value_count, dtype);
        const float got = load_as_float(output, i * value_count + e, out_dt);
        ASSERT_NEAR(got, ref, tol) << "key slot " << i << " element " << e;
      }
    }
  } else {
    for (int64_t b = 0; b < output_bags; ++b) {
      for (int64_t e = 0; e < value_count; ++e) {
        float ref = 0.0f;
        for (int64_t h = 0; h < hotness; ++h) {
          const int8_t* row = h_table + static_cast<int64_t>(keys[static_cast<size_t>(b * hotness + h)]) * row_bytes;
          ref += load_quant_row_element_as_float(row, e, value_count, dtype);
        }
        const float got = load_as_float(output, b * value_count + e, out_dt);
        ASSERT_NEAR(got, ref, tol) << "bag " << b << " element " << e;
      }
    }
  }

  NVE_CHECK_(cudaFreeHost(output));
  ctx->wait();
  ctx.reset();
  NVE_CHECK_(cudaFreeHost(h_table));
}

// Lookups of keys outside [0, uvm_num_rows) must resolve to the configured default row instead of
// reading past the UVM table. `guard_rows` extra rows past uvm_num_rows hold values no valid row
// can produce, so out of range keys aimed at them are visible in the output.
// With auto-inserts enabled the same batch is looked up twice: the second lookup exercises the keys
// promoted into the GPU cache by the first, which must not turn an out of range key into a hit on
// stale data.
template <typename IndexT>
static void RunUVMDefaultRowLookup(bool pooled, uint64_t kernel_mode, bool auto_insert) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  constexpr int device_id = 0;
  ScopedDevice dev(device_id);

  constexpr int64_t num_rows = 4096;
  constexpr int64_t guard_rows = 8;
  constexpr int64_t default_row_index = 11;
  constexpr int64_t hotness = 4;
  constexpr int64_t value_count = 32;
  constexpr int64_t row_bytes = value_count * static_cast<int64_t>(sizeof(float));

  float* h_table = nullptr;
  NVE_CHECK_(cudaMallocHost(&h_table, static_cast<size_t>((num_rows + guard_rows) * row_bytes)));
  for (int64_t r = 0; r < num_rows + guard_rows; ++r) {
    for (int64_t e = 0; e < value_count; ++e) {
      const float value = static_cast<float>(r * 100) + static_cast<float>(e);
      h_table[r * value_count + e] = (r < num_rows) ? value : -value;
    }
  }

  GPUTableConfig cfg;
  cfg.device_id = device_id;
  cfg.cache_size = 1l << 20;
  cfg.row_size_in_bytes = row_bytes;
  cfg.value_dtype = DataType_t::Float32;
  cfg.uvm_table = h_table;
  cfg.uvm_num_rows = num_rows;
  cfg.count_misses = true;
  cfg.kernel_mode_type = kernel_mode;
  auto gpu_tab = std::make_shared<GpuTable<IndexT>>(cfg);

  typename LinearUVMEmbeddingLayer<IndexT>::Config lcfg;
  if (auto_insert) {
    lcfg.insert_heuristic = std::make_shared<TestInsertHeuristic>();
    lcfg.min_insert_freq_gpu = 0;
    lcfg.min_insert_size_gpu = 1;
  } else {
    lcfg.insert_heuristic = std::make_shared<NeverInsertHeuristic>();
  }
  lcfg.default_row_index = default_row_index;
  auto layer = std::make_shared<LinearUVMEmbeddingLayer<IndexT>>(lcfg, gpu_tab);
  auto ctx = layer->create_execution_context(0, 0, nullptr, nullptr);

  // Every 3rd key is out of range: negative, or aimed at a guard row past the end of the table.
  constexpr int64_t num_keys = 64;
  std::vector<IndexT> keys(static_cast<size_t>(num_keys));
  std::vector<IndexT> resolved(static_cast<size_t>(num_keys));  // row each key must read
  for (int64_t i = 0; i < num_keys; ++i) {
    const auto idx = static_cast<size_t>(i);
    if ((i % 3) == 1) {
      keys[idx] = static_cast<IndexT>(-(i + 1));
    } else if ((i % 3) == 2) {
      keys[idx] = static_cast<IndexT>(num_rows + (i % guard_rows));
    } else {
      keys[idx] = static_cast<IndexT>(i);
    }
    resolved[idx] = key_in_range(keys[idx], num_rows) ? keys[idx]
                                                      : static_cast<IndexT>(default_row_index);
  }

  const int64_t output_bags = pooled ? (num_keys / hotness) : num_keys;
  const size_t output_bytes = static_cast<size_t>(output_bags * row_bytes);
  float* output = nullptr;
  NVE_CHECK_(cudaMallocHost(&output, output_bytes));

  EmbeddingLayerBase::PoolingParams pp;
  pp.pooling_type = PoolingType_t::Sum;
  pp.sparse_type = SparseType_t::Fixed;
  pp.fixed_hotness = hotness;
  pp.output_type = DataType_t::Float32;

  const int num_lookups = auto_insert ? 2 : 1;
  for (int attempt = 0; attempt < num_lookups; ++attempt) {
    std::memset(output, 0, output_bytes);
    std::vector<float> hitrates(3);
    layer->lookup(ctx, num_keys, keys.data(), output, row_bytes, nullptr /*hitmask*/,
                  pooled ? &pp : nullptr, hitrates.data());
    ctx->wait();  // auto-inserts run on the thread pool, wait for the promotion to land
    NVE_CHECK_(cudaDeviceSynchronize());
    if (auto_insert && (attempt > 0)) {
      // The first lookup promotes the keys it resolved, which includes the default row - so the
      // repeat lookup is fully served by the cache. A miss here means the out of range keys were
      // promoted as themselves instead of as the default row.
      EXPECT_FLOAT_EQ(1.0f, hitrates[0]);
    }

    // Reference: out of range keys contribute the default row, in a bag like anywhere else
    for (int64_t b = 0; b < output_bags; ++b) {
      for (int64_t e = 0; e < value_count; ++e) {
        float expected = 0.0f;
        if (pooled) {
          for (int64_t h = 0; h < hotness; ++h) {
            expected += h_table[resolved[static_cast<size_t>(b * hotness + h)] * value_count + e];
          }
        } else {
          expected = h_table[resolved[static_cast<size_t>(b)] * value_count + e];
        }
        ASSERT_FLOAT_EQ(expected, output[b * value_count + e])
            << "lookup " << attempt << " output row " << b << " element " << e;
      }
    }
  }

  NVE_CHECK_(cudaFreeHost(output));
  ctx->wait();
  ctx.reset();
  NVE_CHECK_(cudaFreeHost(h_table));
}

TEST(UVMOutOfRangeKeys, LookupReturnsDefaultRow) {
  RunUVMDefaultRowLookup<int64_t>(false /*pooled*/, 0 /*kernel_mode: default*/, false /*auto_insert*/);
}

TEST(UVMOutOfRangeKeys, LookupReturnsDefaultRowInt32) {
  RunUVMDefaultRowLookup<int32_t>(false /*pooled*/, 0 /*kernel_mode: default*/, false /*auto_insert*/);
}

TEST(UVMOutOfRangeKeys, LookupReturnsDefaultRowSortGather) {
  RunUVMDefaultRowLookup<int64_t>(false /*pooled*/,
                                  static_cast<uint64_t>(KernelType::SortGather), false /*auto_insert*/);
}

TEST(UVMOutOfRangeKeys, PooledLookupPoolsDefaultRow) {
  RunUVMDefaultRowLookup<int64_t>(true /*pooled*/, 0 /*kernel_mode: default*/, false /*auto_insert*/);
}

// Auto-inserts promote the looked up keys into the GPU cache, so a repeated lookup must keep
// returning the default row for out of range keys rather than whatever the promotion cached.
TEST(UVMOutOfRangeKeys, LookupWithAutoInsertStaysConsistent) {
  RunUVMDefaultRowLookup<int64_t>(false /*pooled*/, 0 /*kernel_mode: default*/, true /*auto_insert*/);
}

TEST(UVMOutOfRangeKeys, PooledLookupWithAutoInsertStaysConsistent) {
  RunUVMDefaultRowLookup<int64_t>(true /*pooled*/, 0 /*kernel_mode: default*/, true /*auto_insert*/);
}

TEST(UVMValidation, RejectsDefaultRowOutsideTable) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  constexpr int64_t num_rows = 64;
  constexpr int64_t row_bytes = 32;
  std::vector<int8_t> table(static_cast<size_t>(num_rows * row_bytes));

  GPUTableConfig cfg;
  cfg.device_id = 0;
  cfg.cache_size = 1l << 16;
  cfg.row_size_in_bytes = row_bytes;
  cfg.value_dtype = DataType_t::Float32;
  cfg.uvm_table = table.data();
  cfg.uvm_num_rows = num_rows;
  auto gpu_tab = std::make_shared<GpuTable<int64_t>>(cfg);

  typename LinearUVMEmbeddingLayer<int64_t>::Config lcfg;
  lcfg.insert_heuristic = std::make_shared<NeverInsertHeuristic>();
  lcfg.default_row_index = num_rows;
  EXPECT_THROW((void)std::make_shared<LinearUVMEmbeddingLayer<int64_t>>(lcfg, gpu_tab), Exception);
}

TEST(UVMValidation, RejectsNumRowsOutsideKeyType) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  constexpr int64_t row_bytes = 32;
  std::vector<int8_t> table(static_cast<size_t>(row_bytes));

  GPUTableConfig cfg;
  cfg.device_id = 0;
  cfg.cache_size = 1l << 16;
  cfg.row_size_in_bytes = row_bytes;
  cfg.value_dtype = DataType_t::Float32;
  cfg.uvm_table = table.data();
  cfg.uvm_num_rows = int64_t{1} << 31;
  auto gpu_tab = std::make_shared<GpuTable<int32_t>>(cfg);

  typename LinearUVMEmbeddingLayer<int32_t>::Config lcfg;
  lcfg.insert_heuristic = std::make_shared<NeverInsertHeuristic>();
  EXPECT_THROW((void)std::make_shared<LinearUVMEmbeddingLayer<int32_t>>(lcfg, gpu_tab), Exception);
}

// Concatenate (no pooling) -> find_and_dequant.
TEST(UVMQuant, DequantQInt8F32_Int64)  { RunQuantUVMLookup<int64_t>(DataType_t::QInt8RowwiseF32,  PoolingType_t::Concatenate, 1); }
TEST(UVMQuant, DequantQUint8F32_Int64) { RunQuantUVMLookup<int64_t>(DataType_t::QUint8RowwiseF32, PoolingType_t::Concatenate, 1); }
TEST(UVMQuant, DequantQInt8F16_Int64)  { RunQuantUVMLookup<int64_t>(DataType_t::QInt8RowwiseF16,  PoolingType_t::Concatenate, 1); }
TEST(UVMQuant, DequantQUint8F16_Int64) { RunQuantUVMLookup<int64_t>(DataType_t::QUint8RowwiseF16, PoolingType_t::Concatenate, 1); }
TEST(UVMQuant, DequantQInt8F32_Int32)  { RunQuantUVMLookup<int32_t>(DataType_t::QInt8RowwiseF32,  PoolingType_t::Concatenate, 1); }
TEST(UVMQuant, DequantQUint8F16_Int32) { RunQuantUVMLookup<int32_t>(DataType_t::QUint8RowwiseF16, PoolingType_t::Concatenate, 1); }
// Same-type Concatenate -> raw row passthrough through the ordinary find path.
TEST(UVMQuant, RawConcatQInt8F32_Int64)  { RunQuantUVMLookup<int64_t>(DataType_t::QInt8RowwiseF32,  PoolingType_t::Concatenate, 1, true); }
TEST(UVMQuant, RawConcatQUint8F16_Int32) { RunQuantUVMLookup<int32_t>(DataType_t::QUint8RowwiseF16, PoolingType_t::Concatenate, 1, true); }
// Fixed-hotness Sum pooling over quantized rows -> find_and_combine.
TEST(UVMQuant, SumQInt8F32_Int64)  { RunQuantUVMLookup<int64_t>(DataType_t::QInt8RowwiseF32,  PoolingType_t::Sum, 8); }
TEST(UVMQuant, SumQUint8F32_Int64) { RunQuantUVMLookup<int64_t>(DataType_t::QUint8RowwiseF32, PoolingType_t::Sum, 8); }
TEST(UVMQuant, SumQInt8F16_Int64)  { RunQuantUVMLookup<int64_t>(DataType_t::QInt8RowwiseF16,  PoolingType_t::Sum, 8); }
TEST(UVMQuant, SumQUint8F16_Int64) { RunQuantUVMLookup<int64_t>(DataType_t::QUint8RowwiseF16, PoolingType_t::Sum, 8); }

class UVMSpecialConfig : public ::testing::TestWithParam<UVMTestCase> {};

TEST_P(UVMSpecialConfig, InflightInsert) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto& tc = GetParam();
  UVMLayerTest<int64_t> ult(tc.row_size, tc.gpu_table_size, tc.uvm_table_size, tc.data_type, tc.private_stream_mode, false, 0 /*device_id*/,
                            true /*insert_heuristic*/); // override insert_heuristic
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, ult.max_rows_, tc.data_type);

  // Insert keys to ref and linear (insert to all, the remove from gpu table)
  ult.Insert(keys, data);
  ult.Clear(false /* clear_ref */); // remove from gpu_table, data still left in linear table and ref

  auto first_insert_keys = static_cast<uint64_t>(ult.Config().min_insert_size_gpu / 2);
  // Test relies on having enough keys to trigger auto insert
  ASSERT_LE(first_insert_keys, tc.test_keys); 

  // First lookup: hitrate should be 0 since we cleared the gpu table, too few keys to trigger auto-insert
  float hitrate{0.f};
  ult.LookupAndCheck(keys, 0, first_insert_keys, 1.f, &hitrate);
  ASSERT_EQ(hitrate, 0.f);
  ult.Wait(); // Wait in case auto insert was triggereed prematurely

  // Second lookup: hitrate should still be 0, but auto insert should be triggered
  ult.LookupAndCheck(keys, first_insert_keys, uint64_t(-1), 1.f, &hitrate);
  ASSERT_EQ(hitrate, 0.f);
  ult.Wait(); // Wait for auto insert to finish

  // Third lookup: hitrate should be just for auto insert size, and trigger second auto-insert for all keys
  ult.LookupAndCheck(keys, 0, uint64_t(-1), 1.f, &hitrate);
  ASSERT_NEAR(hitrate, float(ult.Config().min_insert_size_gpu) / float(keys.size()), 1e-2);
  ult.Wait(); // Wait for auto insert to finish

  // Last lookup - now hitrate should be (almost) 100%
  ult.LookupAndCheck(keys);
}

// Auto-insert on the pooling path: a pooling lookup produces no reusable per-key row data, so the layer
// collects keys only and promotes them by reading their rows directly from the UVM table
// (GpuTable::insert_from_uvm). Verifies the promoted rows become resolvable from the GPU cache.
// tc.modify_on_gpu exercises both the GPU (key-indexed) and CPU histogram paths of insert_from_uvm.
TEST_P(UVMSpecialConfig, InflightInsertPooling) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto& tc = GetParam();
  UVMLayerTest<int64_t> ult(tc.row_size, tc.gpu_table_size, tc.uvm_table_size, tc.data_type,
                            tc.private_stream_mode, tc.modify_on_gpu, 0 /*device_id*/, true /*insert_heuristic*/);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;
  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, ult.max_rows_, tc.data_type);

  // Populate linear table + ref with known data, then drop the GPU cache so lookups start cold.
  ult.Insert(keys, data);
  ult.Clear(false /* clear_ref */);

  // Pooling lookup over all keys: resolves via UVM (cache empty) and triggers insert_from_uvm.
  auto tcp = tc;
  tcp.pooling_type = PoolingType_t::Sum;
  tcp.offsets_layout = SparseType_t::Fixed;
  tcp.hotness = 8;
  ult.LookupAndCheckPooling(tcp, keys);
  ult.Wait();  // wait for the async auto-insert to finish

  // The keys were promoted straight from the UVM table; a plain lookup should now hit in the GPU cache
  // (cold before the pooling lookup) and return rows matching the table. LookupAndCheck also verifies
  // the values for both hits and misses. hit_tol=1 ignores the hitrate comparison since the exact
  // residency depends on set-associative eviction at this cache load; we assert it separately.
  float hitrate{0.f};
  ult.LookupAndCheck(keys, 0, uint64_t(-1), 1.f /*ignore hitrate tol*/, &hitrate);
  ASSERT_GT(hitrate, 0.9f);
}

// Mixed collection: non-pooled lookups begin a collection (below the insert threshold), then a pooled
// lookup joins it. The whole pending insert must switch to UVM mode and promote all collected keys from
// the UVM table (the partial per-key data gathered before the switch is discarded).
TEST_P(UVMSpecialConfig, InflightInsertPoolingMixed) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto& tc = GetParam();
  UVMLayerTest<int64_t> ult(tc.row_size, tc.gpu_table_size, tc.uvm_table_size, tc.data_type,
                            tc.private_stream_mode, tc.modify_on_gpu, 0 /*device_id*/, true /*insert_heuristic*/);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;
  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, ult.max_rows_, tc.data_type);

  ult.Insert(keys, data);
  ult.Clear(false /* clear_ref */);

  const auto threshold = static_cast<uint64_t>(ult.Config().min_insert_size_gpu); // 1024
  const auto first = threshold / 2; // 512: below threshold, starts a non-pooled collection (with data)
  ASSERT_LE(threshold, tc.test_keys);

  // Non-pooled lookup starts the collection but does not cross the threshold (no insert launched yet).
  float hitrate{0.f};
  ult.LookupAndCheck(keys, 0, first, 1.f /*ignore hitrate tol*/, &hitrate);
  ASSERT_EQ(hitrate, 0.f);

  // Pooled lookup over the remaining keys: flips the in-flight collection to UVM mode and crosses the
  // threshold, launching insert_from_uvm for all collected keys (the first `first` keys + more).
  auto tcp = tc;
  tcp.pooling_type = PoolingType_t::Sum;
  tcp.offsets_layout = SparseType_t::Fixed;
  tcp.hotness = 8;
  ult.LookupAndCheckPooling(tcp, keys, first, uint64_t(-1));
  ult.Wait();  // wait for the async auto-insert to finish

  // The collection grows to `threshold` keys (keys[0, threshold)); all should now resolve from the GPU
  // cache, including the prefix that was collected before the switch to UVM mode.
  ult.LookupAndCheck(keys, 0, threshold, 1.f /*ignore hitrate tol*/, &hitrate);
  ASSERT_GT(hitrate, 0.9f);
}

// Reverse of InflightInsertPoolingMixed: a pooled lookup starts the collection (collection_is_uvm_=true
// from the outset), then non-pooled lookups join and add more keys. The UVM mode must remain sticky so
// that data is never collected for any of the joining lookups, and the final insert_from_uvm promotes
// all collected keys correctly.
TEST_P(UVMSpecialConfig, InflightInsertPoolingMixedReverse) {
  cudaGetLastError();
  const auto& tc = GetParam();
  UVMLayerTest<int64_t> ult(tc.row_size, tc.gpu_table_size, tc.uvm_table_size, tc.data_type,
                            tc.private_stream_mode, tc.modify_on_gpu, 0 /*device_id*/, true /*insert_heuristic*/);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;
  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, ult.max_rows_, tc.data_type);

  ult.Insert(keys, data);
  ult.Clear(false /* clear_ref */);

  const auto threshold = static_cast<uint64_t>(ult.Config().min_insert_size_gpu);
  const auto first = threshold / 2; // below threshold: pooling lookup starts collection in UVM mode
  ASSERT_LE(threshold, tc.test_keys);

  // Pooled lookup below threshold: starts the collection in UVM mode (collection_is_uvm_ = true).
  auto tcp = tc;
  tcp.pooling_type = PoolingType_t::Sum;
  tcp.offsets_layout = SparseType_t::Fixed;
  tcp.hotness = 8;
  ult.LookupAndCheckPooling(tcp, keys, 0, first);
  ult.Wait();

  // Non-pooled lookups join the in-flight collection. Even though these are non-pooled, UVM mode is
  // sticky: keys are added but no per-key row data is collected.
  float hitrate{0.f};
  ult.LookupAndCheck(keys, first, uint64_t(-1), 1.f /*ignore hitrate tol*/, &hitrate);
  ASSERT_EQ(hitrate, 0.f); // still cold before auto-insert completes
  ult.Wait(); // wait for the async insert_from_uvm to finish

  // All collected keys (first `threshold` keys from both lookups) must now be in the GPU cache.
  ult.LookupAndCheck(keys, 0, threshold, 1.f /*ignore hitrate tol*/, &hitrate);
  ASSERT_GT(hitrate, 0.9f);
}

TEST_P(UVMSpecialConfig, LargeModify) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto& tc = GetParam();
  UVMLayerTest<int64_t> ult(tc.row_size, tc.gpu_table_size, tc.uvm_table_size, tc.data_type,
                            tc.private_stream_mode, tc.modify_on_gpu, 0 /*device_id*/, false /*insert_heuristic*/,
                            [&tc](GPUTableConfig& cfg){ cfg.max_modify_size = tc.test_keys / 3; }); // override config to reduce max_modify_size
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;
  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, ult.max_rows_, tc.data_type);

  // Test insert in parts
  ult.Insert(keys, data);
  ult.Wait();

  // since insert can only process 1/3 of the keys - we expect 0.33 hitrate vs. the ref with 1.0 (ref doesn't model modify size)  
  ult.LookupAndCheck(keys, 0, uint64_t(-1), 0.7f);

  // Now do smaller inserts to make sure the baseline state is the same compared to the ref.
  const auto num_keys = keys.size();
  for (uint64_t k_start = 0; k_start < num_keys ; k_start += (num_keys/4)) {
    uint64_t k_end = std::min(num_keys, k_start + (num_keys/4));
    ult.Insert(keys, data, k_start, k_end);
    ult.Wait();
  }
  ult.LookupAndCheck(keys);

  // Test update in parts
  // Change some data so there's something to update
  const auto num_rows = static_cast<int64_t>(data.size()) / tc.row_size;
  for (int64_t i = 0; i < num_rows; i+=2) {  // only updating half the rows
    for (int64_t j = 0; j < tc.row_size; j++) {
      data.at(static_cast<size_t>((i * tc.row_size) + j)) *= 3;
    }
  }
  ult.Update(keys, data);
  ult.Wait();
  ult.LookupAndCheck(keys);

  // Test accumulate in parts
  ult.Accumulate(keys, data, tc.data_type);
  ult.Wait();
  ult.LookupAndCheck(keys);
}

TEST_P(UVMSpecialConfig, UVMUpdate) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto& tc = GetParam();
  UVMLayerTest<int64_t> ult(tc);

  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, ult.max_rows_, tc.data_type);

  // Insert only half the keys
  ult.Insert(keys, data, tc.test_keys / 4 , tc.test_keys * 3 / 4);

  // Update all keys (half will update in the cache, and all will update the table)
  ult.Update(keys, data);
  ult.Wait(); // Wait for auto insert to finish
  
  // Lookup and compare
  ult.LookupAndCheck(keys);
}

TEST_P(UVMSpecialConfig, UVMCPUAccumulate) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto& tc = GetParam();
  UVMLayerTest<int64_t> ult(tc);

  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, ult.max_rows_, tc.data_type);
  // Insert only a small set of keys to guarantee hit rate with a small cache
  ult.Insert(keys, data, 0, 100);

  // Accumulate all keys (a few will update in the cache, and all will update the table)
  ult.Accumulate(keys, data, tc.data_type);
  ult.Wait(); // Wait for auto insert to finish
  
  // Lookup and compare
  ult.LookupAndCheck(keys);
}

TEST_P(UVMSpecialConfig, UVMGPUAccumulate) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto& tc = GetParam();
  UVMLayerTest<int64_t> ult(tc.row_size, tc.gpu_table_size, tc.uvm_table_size, tc.data_type, tc.private_stream_mode, tc.modify_on_gpu, 0 /*device_id*/,
                            false /*insert_heuristic*/,
                            [](GPUTableConfig& cfg){ cfg.uvm_cpu_accumulate = false; }); // override config to disable cpu uvm update

  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, ult.max_rows_, tc.data_type);
  // Insert only a small set of keys to guarantee hit rate with a small cache
  ult.Insert(keys, data, 0, 100);

  // Accumulate all keys (a few will update in the cache, and all will update the table)
  ult.Accumulate(keys, data, tc.data_type);
  ult.Wait(); // Wait for auto insert to finish
  
  // Lookup and compare
  ult.LookupAndCheck(keys);
}

INSTANTIATE_TEST_SUITE_P(
    EmbLayer,
    UVMSpecialConfig,
    ::testing::Values(
        //  TestCase: uvm_table_size,   gpu_table_Size,   row_size,         test_keys         data_type            private_stream                modify_on_gpu
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 15, DataType_t::Float16, PrivateStreamMode::None,      false}),
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 15, DataType_t::Float32, PrivateStreamMode::None,      false}),
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 15, DataType_t::Float16, PrivateStreamMode::Optimized, false}),
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 15, DataType_t::Float32, PrivateStreamMode::Optimized, false}),
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 15, DataType_t::Float32, PrivateStreamMode::None,      true}),
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 15, DataType_t::Float32, PrivateStreamMode::Optimized, true})

    ));


// Modify ops take a value_stride that may exceed the row size (padded rows). The layer must pass the
// row size, not the stride, as the value size to the GPU table.
TEST(UVMPaddedStride, InsertUpdateAccumulateWithPaddedValueStride) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  constexpr int64_t row_size = 256;
  constexpr int64_t stride = row_size + 64;
  constexpr size_t num_keys = 512;
  // The CPU accumulate path requires tightly packed updates; use the GPU kernel path instead.
  UVMLayerTest<int64_t> ult(row_size, int64_t(1) << 24, int64_t(1) << 26, DataType_t::Float32,
                            PrivateStreamMode::None, true /*modify_on_gpu*/, 0 /*device*/,
                            false /*insert_heuristic*/,
                            [](GPUTableConfig& cfg) { cfg.uvm_cpu_accumulate = false; });
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;
  GenerateData<int64_t>(keys, data, num_keys, row_size, 0, ult.max_rows_, DataType_t::Float32,
                        true /*unique*/);
  const int64_t n = static_cast<int64_t>(num_keys);

  // Spread the tightly packed rows out to the padded stride.
  std::vector<uint8_t> padded(num_keys * stride, 0xEE);
  const auto pad = [&]() {
    for (size_t i = 0; i < num_keys; i++) {
      std::memcpy(&padded[i * stride], &data[i * row_size], row_size);
    }
  };
  const auto ref_op = [&](auto&& op) {
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(ult.ctx(), "keys", keys.data(), num_keys * sizeof(int64_t));
    auto values_bw = std::make_shared<BufferWrapper<const void>>(ult.ctx(), "values", data.data(), data.size());
    op(std::move(keys_bw), std::move(values_bw));
  };

  pad();
  ult.layer().insert(ult.ctx(), n, keys.data(), stride, row_size, padded.data(), 0);
  ref_op([&](auto k, auto v) { ult.ref().insert(ult.ctx(), n, std::move(k), row_size, row_size, std::move(v)); });
  NVE_CHECK_(cudaDeviceSynchronize());
  ult.LookupAndCheck(keys);

  for (auto& b : data) { b ^= 0x5A; }  // change every row so the update is observable
  pad();
  ult.layer().update(ult.ctx(), n, keys.data(), stride, row_size, padded.data(), 0);
  ref_op([&](auto k, auto v) { ult.ref().update(ult.ctx(), n, std::move(k), row_size, row_size, std::move(v)); });
  NVE_CHECK_(cudaDeviceSynchronize());
  ult.LookupAndCheck(keys);

  ult.layer().accumulate(ult.ctx(), n, keys.data(), stride, row_size, padded.data(), DataType_t::Float32, 0);
  ref_op([&](auto k, auto v) {
    ult.ref().update_accumulate(ult.ctx(), n, std::move(k), row_size, row_size, std::move(v), DataType_t::Float32);
  });
  NVE_CHECK_(cudaDeviceSynchronize());
  ult.LookupAndCheck(keys);
}

}  // namespace nve
