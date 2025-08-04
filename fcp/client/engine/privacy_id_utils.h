/*
 * Copyright 2025 Google LLC
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
#ifndef FCP_CLIENT_ENGINE_PRIVACY_ID_UTILS_H_
#define FCP_CLIENT_ENGINE_PRIVACY_ID_UTILS_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/civil_time.h"
#include "fcp/client/example_query_result.pb.h"
#include "fcp/protos/confidentialcompute/selection_criteria.pb.h"
#include "fcp/protos/plan.pb.h"

namespace fcp {
namespace client {
namespace engine {

// Get the start of the time window that the event time falls into, given a time
// window schedule specified in the privacy ID config. Only supports IGNORE time
// zone scheme and civil time tumbling windows. event_time should be in the
// format YYYY-MM-DDTHH:MM:SS[+-]HH:MM
absl::StatusOr<absl::CivilSecond> GetPrivacyIdTimeWindowStart(
    const fedsql::PrivacyIdConfig& privacy_id_config,
    absl::string_view event_time);

// Deterministically derive a 16 byte privacy ID from the source ID and window
// start.
absl::StatusOr<std::string> GetPrivacyId(absl::string_view source_id,
                                         absl::CivilSecond window_start);

// ExampleQueryResult corresponding to a specific privacy ID.
struct PerPrivacyIdResult {
  std::string privacy_id;
  fcp::client::ExampleQueryResult example_query_result;
};

// An ExampleQuery and its results split by privacy ID.
struct SplitResults {
  google::internal::federated::plan::ExampleQuerySpec::ExampleQuery
      example_query;
  std::vector<PerPrivacyIdResult> per_privacy_id_results;
};

// Split each ExampleQueryResult within the input vector into multiple
// ExampleQueryResults, one for each unique privacy ID. Add the privacy ID as a
// column with a single value to each ExampleQueryResult.
absl::StatusOr<std::vector<SplitResults>> SplitResultsByPrivacyId(
    const std::vector<std::pair<
        google::internal::federated::plan::ExampleQuerySpec::ExampleQuery,
        fcp::client::ExampleQueryResult>>& structured_example_query_results,
    const fedsql::PrivacyIdConfig& privacy_id_config,
    absl::string_view source_id);

}  // namespace engine
}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_ENGINE_PRIVACY_ID_UTILS_H_
