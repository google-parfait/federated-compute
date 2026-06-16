/*
 * Copyright 2021 Google LLC
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

#include "fcp/secagg/shared/rlwe/input_vector_rlwe_specification.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fcp/secagg/shared/secagg_vector.h"

namespace fcp {
namespace secagg {
namespace {

TEST(InputVectorRlweSpecificationTest, ConstructorDiesOnSmallModulus) {
  EXPECT_DEATH(InputVectorRlweSpecification vec_spec("test", 16, 8, 2, 3, 1),
               "The specified modulus is not valid");
}

TEST(InputVectorRlweSpecificationTest, ConstructorDiesOnModulusGreatorThanMax) {
  EXPECT_DEATH(InputVectorRlweSpecification vec_spec(
                   "test", 16, 8, 2, 3, SecAggVector::kMaxModulus + 1),
               "The specified modulus is not valid");
}

TEST(InputVectorRlweSpecificationTest, ConstructorDiesOnNegativeLength) {
  EXPECT_DEATH(InputVectorRlweSpecification vec_spec("test", -16, 8, 2, 3, 2),
               "Length must be >= 0");
}

TEST(InputVectorRlweSpecificationTest,
     ConstructorDiesOnLengthNotMultipleOfDegree) {
  EXPECT_DEATH(InputVectorRlweSpecification vec_spec("test", 11, 7, 2, 3, 2),
               "Length must be a multiple of polynomial degree");
}

TEST(InputVectorRlweSpecificationTest, ConstructorDiesOnNegativeDegree) {
  EXPECT_DEATH(InputVectorRlweSpecification vec_spec("test", 14, -7, 2, 3, 2),
               "Polynomial degree must be greater than 0");
}

TEST(InputVectorRlweSpecificationTest, ConstructorDiesOnNegativeRlweModulus) {
  EXPECT_DEATH(InputVectorRlweSpecification vec_spec("test", 14, 7, 0, 3, 2),
               "RlweModulus must be greater than 0");
}

TEST(InputVectorRlweSpecificationTest,
     ConstructorDiesOnNegativeOrZeroLogDegree) {
  EXPECT_DEATH(InputVectorRlweSpecification vec_spec("test", 14, 7, 2, 0, 2),
               "Log degree must be greater than 0");
  EXPECT_DEATH(InputVectorRlweSpecification vec_spec("test", 14, 7, 2, -1, 2),
               "Log degree must be greater than 0");
}

TEST(InputVectorRlweSpecificationTest, ConstructorDiesOnBadLogDegree) {
  EXPECT_DEATH(InputVectorRlweSpecification vec_spec("test", 14, 7, 2, 2, 2),
               "Log degree must be the bit-width of degree but is too small");
}

}  // namespace
}  // namespace secagg
}  // namespace fcp
