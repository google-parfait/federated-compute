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

namespace fcp {
namespace aggregation {
namespace {

using testing::Return;

class MockTensorData : public TensorData {
 public:
  MockTensorData(size_t data_pointer_offset, size_t size);

  MOCK_METHOD(const void*, data, (), (const override));
  MOCK_METHOD(size_t, byte_size, (), (const override));
};

MockTensorData::MockTensorData(size_t data_pointer_offset, size_t size) {
  EXPECT_CALL(*this, byte_size()).WillRepeatedly(Return(size));
  EXPECT_CALL(*this, data())
      .WillRepeatedly(Return(reinterpret_cast<void*>(data_pointer_offset)));
}

TEST(TensorDataTest, CheckValid_ZeroByteSize) {
  MockTensorData tensor_data(0, 0);
  EXPECT_THAT(tensor_data.CheckValid(1), IsCode(FAILED_PRECONDITION));
}

TEST(TensorDataTest, CheckValid_ByteSizeNotAligned) {
  MockTensorData tensor_data(0, 33);
  EXPECT_THAT(tensor_data.CheckValid(4), IsCode(FAILED_PRECONDITION));
}

TEST(TensorDataTest, CheckValid_AddressNotAligned) {
  MockTensorData tensor_data(3, 100);
  EXPECT_THAT(tensor_data.CheckValid(4), IsCode(FAILED_PRECONDITION));
}

TEST(TensorDataTest, CheckValid_Success) {
  MockTensorData tensor_data(0, 96);
  EXPECT_THAT(tensor_data.CheckValid(1), IsOk());
  EXPECT_THAT(tensor_data.CheckValid(2), IsOk());
  EXPECT_THAT(tensor_data.CheckValid(4), IsOk());
  EXPECT_THAT(tensor_data.CheckValid(8), IsOk());
}

}  // namespace
}  // namespace aggregation
}  // namespace fcp
