/*
 * Copyright 2022 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "fcp/aggregation/core/tensor_data.h"

#include <cstdint>
#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fcp/testing/testing.h"

namespace fcp::aggregation {
namespace {

using testing::Eq;
using testing::Invoke;
using testing::Return;

class MockTensorData : public TensorData {
 public:
  explicit MockTensorData(size_t size);

  void Add(size_t byte_offset, size_t byte_size);
  void Add(size_t data_pointer_offset, size_t byte_offset, size_t byte_size);

  MOCK_METHOD(int, num_slices, (), (const override));
  MOCK_METHOD(Slice, get_slice, (int), (const override));
  MOCK_METHOD(size_t, byte_size, (), (const override));

 private:
  int num_slices_ = 0;
};

MockTensorData::MockTensorData(size_t size) {
  EXPECT_CALL(*this, byte_size()).WillRepeatedly(Return(size));
  EXPECT_CALL(*this, num_slices()).WillRepeatedly(Invoke([&] {
    return num_slices_;
  }));
}

void MockTensorData::Add(size_t byte_offset, size_t byte_size) {
  Add(byte_offset, byte_offset, byte_size);
}

void MockTensorData::Add(size_t data_pointer_offset, size_t byte_offset,
                         size_t byte_size) {
  TensorData::Slice slice = {byte_offset, byte_size,
                             reinterpret_cast<void*>(data_pointer_offset)};
  EXPECT_CALL(*this, get_slice(Eq(num_slices_))).WillRepeatedly(Return(slice));
  num_slices_++;
}

TEST(TensorDataTest, CheckValid_ZeroByteSize) {
  MockTensorData tensor_data(0);
  EXPECT_THAT(tensor_data.CheckValid(1), IsCode(FAILED_PRECONDITION));
}

TEST(TensorDataTest, CheckValid_NegativeNumberOfSlices) {
  MockTensorData tensor_data(1);
  EXPECT_CALL(tensor_data, num_slices()).WillRepeatedly(Return(-1));
  EXPECT_THAT(tensor_data.CheckValid(1), IsCode(FAILED_PRECONDITION));
}

TEST(TensorDataTest, CheckValid_ByteSizeNotAligned) {
  MockTensorData tensor_data(33);
  tensor_data.Add(0, 32);
  EXPECT_THAT(tensor_data.CheckValid(4), IsCode(FAILED_PRECONDITION));
}

TEST(TensorDataTest, CheckValid_SliceOutsideByteSize) {
  MockTensorData tensor_data(100);
  tensor_data.Add(0, 101);
  EXPECT_THAT(tensor_data.CheckValid(1), IsCode(FAILED_PRECONDITION));
}

TEST(TensorDataTest, CheckValid_SlicesOverlapping) {
  MockTensorData tensor_data(100);
  tensor_data.Add(0, 60);
  tensor_data.Add(50, 50);
  EXPECT_THAT(tensor_data.CheckValid(1), IsCode(FAILED_PRECONDITION));
}

TEST(TensorDataTest, CheckValid_SlicesNotOrdered) {
  MockTensorData tensor_data(100);
  tensor_data.Add(50, 50);
  tensor_data.Add(0, 50);
  EXPECT_THAT(tensor_data.CheckValid(1), IsCode(FAILED_PRECONDITION));
}

TEST(TensorDataTest, CheckValid_SliceAddressNotAligned) {
  MockTensorData tensor_data(100);
  tensor_data.Add(13, 12, 40);
  EXPECT_THAT(tensor_data.CheckValid(4), IsCode(FAILED_PRECONDITION));
}

TEST(TensorDataTest, CheckValid_SliceOffsetNotAligned) {
  MockTensorData tensor_data(100);
  tensor_data.Add(12, 13, 40);
  EXPECT_THAT(tensor_data.CheckValid(4), IsCode(FAILED_PRECONDITION));
}

TEST(TensorDataTest, CheckValid_SliceSizeNotAligned) {
  MockTensorData tensor_data(100);
  tensor_data.Add(12, 43);
  EXPECT_THAT(tensor_data.CheckValid(4), IsCode(FAILED_PRECONDITION));
}

TEST(TensorDataTest, CheckValid_SliceSizeOverflow) {
  MockTensorData tensor_data(100);
  tensor_data.Add(12, static_cast<size_t>(-1LL));
  EXPECT_THAT(tensor_data.CheckValid(4), IsCode(FAILED_PRECONDITION));
}

TEST(TensorDataTest, CheckValid_Success) {
  MockTensorData tensor_data(96);
  tensor_data.Add(8, 32);
  tensor_data.Add(48, 24);
  tensor_data.Add(72, 24);
  EXPECT_THAT(tensor_data.CheckValid(1), IsOk());
  EXPECT_THAT(tensor_data.CheckValid(2), IsOk());
  EXPECT_THAT(tensor_data.CheckValid(4), IsOk());
  EXPECT_THAT(tensor_data.CheckValid(8), IsOk());
}

}  // namespace
}  // namespace fcp::aggregation
