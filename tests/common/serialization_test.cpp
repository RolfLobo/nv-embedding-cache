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
#include <serialization.hpp>
#include <common.hpp>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

// Build a numpy v1.0 file image: magic, version, uint16 header length, header dict, raw data.
std::string make_npy(const std::string& header_dict, const std::string& data)
{
    std::string out("\x93NUMPY\x01\x00", 8);
    // v1.0 stores header_len as uint16; a larger dict would silently wrap.
    EXPECT_LE(header_dict.size(), std::numeric_limits<uint16_t>::max());
    const uint16_t header_len = static_cast<uint16_t>(header_dict.size());
    out.append(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    out += header_dict;
    out += data;
    return out;
}

std::string write_npy(const std::string& name, const std::string& header_dict, const std::string& data)
{
    const std::string path = testing::TempDir() + name;
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    const std::string contents = make_npy(header_dict, data);
    file.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    EXPECT_TRUE(file.good());
    return path;
}

std::string header_for_shape(const std::string& shape, const std::string& descr = "<f4")
{
    return "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shape + ", }";
}

nve::NumpyTensorFileFormat open_npy(const std::string& path)
{
    return nve::NumpyTensorFileFormat(std::make_shared<nve::InputFileStreamWrapper>(path));
}

TEST(NumpyTensorFileFormat, Loads2DFile)
{
    const std::vector<float> data = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
    const std::string path = write_npy("npy_2d.npy", header_for_shape("(2, 3)"),
                                       std::string(reinterpret_cast<const char*>(data.data()),
                                                   data.size() * sizeof(float)));
    auto reader = open_npy(path);
    EXPECT_EQ(reader.get_num_rows(), 2u);
    EXPECT_EQ(reader.get_row_size_in_bytes(), 3 * sizeof(float));

    std::vector<float> loaded(6, -1.f);
    reader.reset();
    reader.load_batch(2, loaded.data());
    EXPECT_EQ(loaded, data);
}

TEST(NumpyTensorFileFormat, Loads1DFile)
{
    const std::vector<int64_t> keys = {10, 20, 30};
    const std::string path = write_npy("npy_1d.npy", header_for_shape("(3,)", "<i8"),
                                       std::string(reinterpret_cast<const char*>(keys.data()),
                                                   keys.size() * sizeof(int64_t)));
    auto reader = open_npy(path);
    EXPECT_EQ(reader.get_num_rows(), 3u);
    EXPECT_EQ(reader.get_shape(), std::vector<uint64_t>{3});
    EXPECT_EQ(reader.get_row_size_in_bytes(), sizeof(int64_t));
}

TEST(NumpyTensorFileFormat, RejectsTruncatedData)
{
    // Header declares 4x3 floats (48 bytes) but the file only holds one row.
    const std::string path =
        write_npy("npy_truncated.npy", header_for_shape("(4, 3)"), std::string(12, '\0'));
    EXPECT_THROW(open_npy(path), nve::Exception);
}

TEST(NumpyTensorFileFormat, RejectsGarbageShape)
{
    const std::string path =
        write_npy("npy_bad_shape.npy", header_for_shape("(abc,)"), std::string(64, '\0'));
    EXPECT_THROW(open_npy(path), nve::Exception);
}

TEST(NumpyTensorFileFormat, RejectsNegativeShape)
{
    const std::string path =
        write_npy("npy_neg_shape.npy", header_for_shape("(-1,)"), std::string(64, '\0'));
    EXPECT_THROW(open_npy(path), nve::Exception);
}

TEST(NumpyTensorFileFormat, RejectsEmptyShape)
{
    const std::string path =
        write_npy("npy_empty_shape.npy", header_for_shape("()"), std::string(64, '\0'));
    EXPECT_THROW(open_npy(path), nve::Exception);
}

TEST(NumpyTensorFileFormat, RejectsRowSizeOverflow)
{
    // 2^32 * 2^32 elements per row wraps a uint64 element count.
    const std::string path = write_npy(
        "npy_row_overflow.npy",
        header_for_shape("(2, 4294967296, 4294967296)", "<f8"), std::string(64, '\0'));
    EXPECT_THROW(open_npy(path), nve::Exception);
}

TEST(NumpyTensorFileFormat, RejectsDataSizeOverflow)
{
    // Row size is valid (2^35 bytes) but rows * row size wraps a uint64.
    const std::string path = write_npy(
        "npy_data_overflow.npy",
        header_for_shape("(4294967296, 4294967296)", "<f8"), std::string(64, '\0'));
    EXPECT_THROW(open_npy(path), nve::Exception);
}

}  // namespace
