/*
 * Copyright 2019 Google LLC
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

#include "fcp/tensorflow/status.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace fcp {

using ::testing::Eq;

TEST(StatusTest, ToTensorFlow_Ok) {
  EXPECT_THAT(ConvertToTensorFlowStatus(FCP_STATUS(OK)),
              Eq(tensorflow::OkStatus()));
}

TEST(StatusTest, ToTensorFlow_Error) {
  Status error = FCP_STATUS(NOT_FOUND) << "Where is my mind?";
  EXPECT_THAT(ConvertToTensorFlowStatus(error),
              // Remove the cast after TF 2.12 is released and used in FCP.
              Eq(tensorflow::Status(
                  static_cast<tsl::errors::Code>(absl::StatusCode::kNotFound),
                  error.message())));
}

}  // namespace fcp
