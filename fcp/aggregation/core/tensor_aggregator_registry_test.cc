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

#include "fcp/aggregation/core/tensor_aggregator_registry.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fcp/aggregation/core/tensor_aggregator_factory.h"
#include "fcp/base/monitoring.h"
#include "fcp/testing/testing.h"

namespace fcp {
namespace aggregation {
namespace {

class MockFactory : public TensorAggregatorFactory {
  MOCK_METHOD(StatusOr<std::unique_ptr<TensorAggregator>>, Create,
              (DataType dtype, TensorShape shape), (const override));
};

REGISTER_AGGREGATOR_FACTORY("foobar", MockFactory);

TEST(TensorAggregatorRegistryTest, FactoryRegistrationSuccessful) {
  EXPECT_THAT(GetAggregatorFactory("foobar"), IsOk());
  EXPECT_THAT(GetAggregatorFactory("xyz"), IsCode(NOT_FOUND));
}

TEST(TensorAggregatorRegistryTest, RepeatedRegistrationUnsuccessful) {
  MockFactory factory2;
  EXPECT_DEATH(RegisterAggregatorFactory("foobar", &factory2),
               "already registered");
}

}  // namespace
}  // namespace aggregation
}  // namespace fcp
