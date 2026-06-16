/*
 * Copyright 2020 Google LLC
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

#include "fcp/secagg/shared/rlwe/secagg_rlwe_vector.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fcp/secagg/shared/math.h"
#include "google/protobuf/util/message_differencer.h"

namespace fcp {
namespace secagg {
namespace {

using testing::Eq;

TEST(SecAggRlweVectorTest, GettersReturnValuesFromConstructedVector) {
  auto modulus_params_status =
      internal::rlwe_mont_int::Params::Create(rlwe::kModulus29);
  ASSERT_THAT(modulus_params_status.ok(), Eq(true));
  auto modulus_params = std::move(modulus_params_status.value());

  auto four = internal::rlwe_mont_int::ImportInt(4, modulus_params.get());
  ASSERT_THAT(four.ok(), Eq(true));

  auto five = internal::rlwe_mont_int::ImportInt(5, modulus_params.get());
  ASSERT_THAT(five.ok(), Eq(true));
  std::vector<internal::rlwe_mont_int> raw_uintm_vector = {four.value(),
                                                           five.value()};
  std::vector<rlwe::Polynomial<internal::rlwe_mont_int>> raw_vector = {
      rlwe::Polynomial<internal::rlwe_mont_int>(raw_uintm_vector)};
  SecAggRlweVector vector(raw_vector, modulus_params.get(), 32,
                          rlwe::kLogDegreeBound29);

  auto returned_vector = vector.GetAsPolynomialVector();
  ASSERT_THAT(returned_vector.ok(), Eq(true));
  EXPECT_THAT(returned_vector.value(), Eq(raw_vector));

  auto serialized_raw_polynomial =
      raw_vector[0].Serialize(modulus_params.get());
  ASSERT_THAT(serialized_raw_polynomial.ok(), Eq(true));
  auto serialized = vector.GetAsSerializedVector();
  std::vector<rlwe::SerializedNttPolynomial> serialized_expected = {
      serialized_raw_polynomial.value()};
  EXPECT_THAT(serialized.size(), Eq(1));
  EXPECT_THAT(google::protobuf::util::MessageDifferencer::Equals(serialized[0],
                                                       serialized_expected[0]),
              Eq(true));
}

}  // namespace
}  // namespace secagg
}  // namespace fcp
