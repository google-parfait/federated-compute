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
#include "fcp/client/federated_protocol_util.h"

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fcp/client/diag_codes.pb.h"
#include "fcp/client/test_helpers.h"
#include "fcp/testing/testing.h"

namespace fcp::client {
namespace {

using ::testing::StrictMock;

TEST(ExtractTaskNameFromAggregationSessionIdTest, ExtractSuccessfully) {
  StrictMock<MockLogManager> mock_log_manager_;
  EXPECT_EQ(ExtractTaskNameFromAggregationSessionId(
                "population_name/task_name#foo.bar", "population_name",
                mock_log_manager_),
            "task_name");
  EXPECT_EQ(
      ExtractTaskNameFromAggregationSessionId(
          "population_name/task_name#", "population_name", mock_log_manager_),
      "task_name");
  EXPECT_EQ(ExtractTaskNameFromAggregationSessionId(
                "population_name/task_name#foobar", "population_name",
                mock_log_manager_),
            "task_name");
  EXPECT_EQ(ExtractTaskNameFromAggregationSessionId(
                "population/name/task_name#foo.bar", "population/name",
                mock_log_manager_),
            "task_name");
  EXPECT_EQ(ExtractTaskNameFromAggregationSessionId(
                "population/name/task/name#foo.bar", "population/name",
                mock_log_manager_),
            "task/name");
  EXPECT_EQ(ExtractTaskNameFromAggregationSessionId(
                "population_name/task/name#foo.bar", "population_name",
                mock_log_manager_),
            "task/name");
}

TEST(ExtractTaskNameFromAggregationSessionIdTest, ExtractUnsuccessfully) {
  {
    StrictMock<MockLogManager> mock_log_manager_;
    EXPECT_CALL(mock_log_manager_,
                LogDiag(ProdDiagCode::OPSTATS_TASK_NAME_EXTRACTION_FAILED));
    EXPECT_EQ(ExtractTaskNameFromAggregationSessionId("foo", "population_name",
                                                      mock_log_manager_),
              "foo");
  }
  {
    StrictMock<MockLogManager> mock_log_manager_;
    EXPECT_CALL(mock_log_manager_,
                LogDiag(ProdDiagCode::OPSTATS_TASK_NAME_EXTRACTION_FAILED));
    EXPECT_EQ(
        ExtractTaskNameFromAggregationSessionId(
            "population_name2/foo#bar", "population_name", mock_log_manager_),
        "population_name2/foo#bar");
  }
}

}  // namespace
}  // namespace fcp::client
