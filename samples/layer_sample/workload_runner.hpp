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

#pragma once
#include <engine_harness.h>

#include <embedding_layer.hpp>
#include <execution_context.hpp>
#include <input_gen.cuh>
#include <memory>
#include <thread>
#include <vector>

// todo: template classes on IndexT
using IndexT = int64_t;
using layer_type = nve::EmbeddingLayerBase;

class ColocatedBuffer {
 public:
  ColocatedBuffer(uint64_t size_, int device_id = 0) : size(size_) {
    ScopedDevice sc(device_id);
    NVE_CHECK_(cudaMalloc(&d_ptr_, size));
    NVE_CHECK_(cudaMallocHost(&h_ptr_, size));
  }
  ~ColocatedBuffer() {
    NVE_CHECK_(cudaFree(d_ptr_));
    NVE_CHECK_(cudaFreeHost(h_ptr_));
  }
  const uint64_t size;
  int8_t* h_ptr() const { return h_ptr_; }
  int8_t* d_ptr() const { return d_ptr_; }

 private:
  int8_t* h_ptr_;
  int8_t* d_ptr_;
};

class LayerData {
 public:
  LayerData(uint64_t num_inputs, uint64_t num_warmups, uint64_t num_rows, uint64_t hotness,
            uint64_t batch_size, float alpha, uint64_t row_size, int device_id)
      : num_inputs_(num_inputs),
        num_warmups_(num_warmups),
        num_rows_(num_rows),
        hotness_(hotness),
        batch_size_(batch_size),
        alpha_(alpha),
        row_size_(row_size),
        device_id_(device_id) {
    ScopedDevice scope_device(device_id_);
    {  // Generate inference inputs
      std::vector<std::shared_ptr<std::thread>> input_gen_threads;
      host_inputs_.resize(num_inputs);
      for (uint64_t s = 0; s < num_inputs_; s++) {
        input_gen_threads.push_back(
            std::make_shared<std::thread>(GenerateInput<IndexT>, num_rows, hotness, batch_size,
                                          alpha, device_id_, std::ref(host_inputs_.at(s))));
      }
      for (auto& t : input_gen_threads) {
        t->join();
      }
    }
    {  // Generate warmup inputs
      std::vector<std::shared_ptr<std::thread>> warmup_gen_threads;
      host_warmups_.resize(num_warmups_);
      for (uint64_t s = 0; s < num_warmups_; s++) {
        warmup_gen_threads.push_back(
            std::make_shared<std::thread>(GenerateInput<IndexT>, num_rows, hotness, batch_size,
                                          alpha, device_id_, std::ref(host_warmups_.at(s))));
      }
      for (auto& t : warmup_gen_threads) {
        t->join();
      }
    }

    const auto num_keys = hotness_ * batch_size_;
    const uint64_t input_size = num_keys * sizeof(IndexT);
    input_ = std::make_shared<ColocatedBuffer>(input_size);
  }

  ~LayerData() {
    for (auto& input : host_inputs_) {
      NVE_CHECK_(cudaHostUnregister(input.data()));
    }
  }

  const uint64_t num_inputs_;
  const uint64_t num_warmups_;
  const uint64_t num_rows_;
  const uint64_t hotness_;
  const uint64_t batch_size_;
  const float alpha_;
  const uint64_t row_size_;
  const int device_id_;
  std::vector<std::vector<IndexT>> host_inputs_;
  std::vector<std::vector<IndexT>> host_warmups_;
  std::shared_ptr<ColocatedBuffer> input_;
};

class WorkloadRunner {
 public:
  WorkloadRunner(uint64_t num_inputs, uint64_t num_warmups, uint64_t num_rows, uint64_t hotness,
                 uint64_t batch_size, float alpha, uint64_t row_size,
                 uint64_t /*max_iterations*/,  // todo used for allocating metrics data
                 std::vector<std::shared_ptr<layer_type>>& layers,
                 std::shared_ptr<EngineHarness> trt_engine, uint64_t engine_ctx_id)
      : layers_(layers), trt_engine_(trt_engine), engine_ctx_id_(engine_ctx_id) {
    // generate input
    const int device_id(0);  // todo multi device
    const auto num_layers = layers.size();
    for (uint64_t i = 0; i < num_layers; i++) {
      data_.emplace_back(std::make_shared<LayerData>(num_inputs, num_warmups, num_rows, hotness,
                                                      batch_size, alpha, row_size, device_id));
    }

    // generate contexts
    for (uint64_t i = 0; i < num_layers; i++) {
      cudaStream_t lookup_stream;
      cudaStream_t modify_stream;
      NVE_CHECK_(cudaStreamCreate(&lookup_stream));
      lookup_streams_.push_back(lookup_stream);
      NVE_CHECK_(cudaStreamCreate(&modify_stream));
      modify_streams_.push_back(modify_stream);
      contexts_.emplace_back(layers_.at(i)->create_execution_context(lookup_stream, modify_stream, nullptr, nullptr));
    }

    // generate cuda events to signal each layer completion
    for (auto& l : layers_) {
      cudaEvent_t e;
      NVE_CHECK_(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
      layer_events_.push_back(e);
    }

    // allocate output buffer for the embeddding layers (assuming all are concatenated)
    uint64_t emb_output_size = 0;
    for (auto& data : data_) {
      emb_output_size += data->batch_size_ * data->hotness_ * data->row_size_;
    }
    emb_output_ = std::make_shared<ColocatedBuffer>(emb_output_size);
    if (trt_engine) {
      auto bindings = trt_engine->GetIOBindings();
      if ((bindings.size() != 2) ||
          (bindings.begin()->isInput == (bindings.begin() + 1)->isInput)) {
        throw std::runtime_error("Invalid TRT engine - must have single input and output");
      }
      for (auto& b : bindings) {
        if (b.isInput) {
          // Use embedding output as engine input
          if (b.sizeInBytes > emb_output_->size) {
            throw std::runtime_error(
                "Invalid TRT engine - engine input is larger than embedding output");
          } else if (b.sizeInBytes < emb_output_->size) {
            std::cout << "[W] "
                      << "TRT engine input is smaller than the embedding output." << std::endl;
          }
          engine_buffers_.push_back(emb_output_->d_ptr());
        } else {
          // Create output buffer for the engine
          engine_output_ = std::make_shared<ColocatedBuffer>(b.sizeInBytes);
          engine_buffers_.push_back(engine_output_->d_ptr());
        }
      }
    }

    // todo: allocate metrics data
    // max_iterations
  }
  ~WorkloadRunner() {
    for (auto& c : contexts_) {
      c->wait();
    }
    contexts_.clear();
    for (auto& e : layer_events_) {
      NVE_CHECK_(cudaEventDestroy(e));
    }
    for (auto& s : lookup_streams_) {
      NVE_CHECK_(cudaStreamDestroy(s));
    }
    for (auto& s : modify_streams_) {
      NVE_CHECK_(cudaStreamDestroy(s));
    }
  }

  void WarmupLayerCache(uint64_t num_iterations) {
    // todo: don't store warmups in layer data, instead create inputs on the fly to support large
    // amount of warmup iterations
    const auto num_layers = layers_.size();
    const uint64_t num_warmups =
        data_.at(0)
            ->host_warmups_.size();  // assuming all layers generated the same amount of inputs
    for (uint64_t i = 0; i < num_iterations; i++) {
      for (uint64_t j = 0; j < num_layers; j++) {
        auto layer = layers_.at(j);
        auto data = data_.at(j);
        auto ctx = contexts_.at(j);
        auto h_input = data->host_warmups_.at(i % num_warmups);
        // using emb_output_ for insert sincwe it's large enough and we don't care about the values
        layer->insert(ctx, h_input.size(), h_input.data(), data->row_size_, data->row_size_,
                      emb_output_->d_ptr(), 0);
        layer->insert(ctx, h_input.size(), h_input.data(), data->row_size_, data->row_size_,
                      emb_output_->h_ptr(), 1);
      }
      NVE_CHECK_(cudaDeviceSynchronize());  // Just in case
    }
  }

  struct Metrics {
    std::vector<float> hitrates_gpu;
    std::vector<float> hitrates_host;
    std::vector<float> hitrates_remote;
    
    void resize(uint64_t num_iterations, uint64_t num_layers) {
      const uint64_t total_samples = num_iterations * num_layers;
      hitrates_gpu.resize(total_samples);
      hitrates_host.resize(total_samples);
      hitrates_remote.resize(total_samples);
    }
  };

  void Run(
    uint64_t num_iterations,
    Metrics* metrics,
    bool disable_copy_out = false
  ) {
    const uint64_t num_layers = layers_.size();
    const cudaStream_t first_stream = contexts_.at(0)->get_lookup_stream();  // Using the first layer's lookup stream for the dense
    const uint64_t num_inputs =
        data_.at(0)
            ->host_inputs_.size();  // assuming all layers generated the same amount of inputs
    float hitrates[3] = {0.f};
    for (uint64_t i = 0; i < num_iterations; i++) {
      // start with calling all lookups
      uint64_t output_offset = 0;
      for (uint64_t j = 0; j < num_layers; j++) {
        // todo: multithread here?
        auto& data = data_.at(j);
        auto ctx = contexts_.at(j);
        auto h_input = data->host_inputs_.at(i % num_inputs);
        layers_.at(j)->lookup(ctx, h_input.size(), h_input.data(),
                              emb_output_->d_ptr() + output_offset, data->row_size_,
                              nullptr, /* hitmask */
                              nullptr, /* pool_params*/
                              hitrates);
        // Collect hitrate metrics using direct indexing
        if (metrics != nullptr) {
        const uint64_t idx = i * num_layers + j;
        metrics->hitrates_gpu[idx] = hitrates[0];
          metrics->hitrates_host[idx] = hitrates[1];
          metrics->hitrates_remote[idx] = hitrates[2];
        }
        // update output offset
        const auto output_size = h_input.size() * data->row_size_;
        output_offset += output_size;
      }
      // create dependency from all lookup streams to the first stream (which we'll reuse for the
      // dense part)
      for (uint64_t j = 1; j < num_layers; j++) {
        auto lookup_stream = contexts_.at(j)->get_lookup_stream();
        auto event = layer_events_.at(j);
        NVE_CHECK_(cudaEventRecord(event, lookup_stream));
        NVE_CHECK_(cudaStreamWaitEvent(first_stream, event));
      }

      if (trt_engine_) {
        // Call trt engine for dense
        trt_engine_->Enqueue(first_stream, static_cast<unsigned>(engine_ctx_id_), engine_buffers_);
        // Copy engine output to host
        if (!disable_copy_out) {
          NVE_CHECK_(cudaMemcpyAsync(engine_output_->h_ptr(), engine_output_->d_ptr(),
                                     engine_output_->size, cudaMemcpyDefault, first_stream));
        }
      } else {
        // No TRT engine so copying embedding output to host
        if (!disable_copy_out) {
          NVE_CHECK_(cudaMemcpyAsync(emb_output_->h_ptr(), emb_output_->d_ptr(), emb_output_->size,
                                     cudaMemcpyDefault, first_stream));
        }
      }
    }
  }

 private:
  std::vector<std::shared_ptr<layer_type>> layers_;
  std::vector<std::shared_ptr<LayerData>> data_;
  std::vector<std::shared_ptr<nve::ExecutionContext>> contexts_;
  std::vector<cudaStream_t> lookup_streams_;
  std::vector<cudaStream_t> modify_streams_;
  std::vector<cudaEvent_t> layer_events_;
  std::shared_ptr<ColocatedBuffer> emb_output_;
  std::shared_ptr<ColocatedBuffer> engine_output_;
  std::shared_ptr<EngineHarness> trt_engine_;
  const uint64_t engine_ctx_id_;
  std::vector<void*> engine_buffers_;
};
