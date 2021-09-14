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


#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/strings/str_split.h"
#include "fcp/base/monitoring.h"
#include "fcp/client/client_runner.h"
#include "fcp/client/federated_task_environment.h"
#include "fcp/client/fl_runner.h"

ABSL_FLAG(std::string, server, "",
          "Federated Server URI (supports https+test:// and https:// URIs");
ABSL_FLAG(std::string, api_key, "", "API Key");
ABSL_FLAG(std::string, test_cert, "",
          "Path to test CA certificate PEM file; used for https+test:// URIs");
ABSL_FLAG(std::string, session, "", "Session name");
ABSL_FLAG(std::string, population, "", "Population name");
ABSL_FLAG(std::string, retry_token, "", "Retry token");
ABSL_FLAG(std::string, client_version, "", "Client version");
ABSL_FLAG(std::string, attestation_string, "", "Attestation string");
ABSL_FLAG(int, num_examples, 0,
          "Number of (empty) examples each created iterator serves");
ABSL_FLAG(int, num_rounds, 1, "Number of rounds to train");

static constexpr char kUsageString[] =
    "Stand-alone Federated Client Executable.\n\n"
    "Connects to the specified server, tries to retrieve a plan, run the\n"
    "plan (feeding the specified number of empty examples), and report the\n"
    "results of the computation back to the server.";

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage(kUsageString);
  absl::ParseCommandLine(argc, argv);
  FCP_LOG(INFO) << absl::GetFlag(FLAGS_server);

  for (auto i = 0; i < absl::GetFlag(FLAGS_num_rounds); ++i) {
    fcp::client::FederatedTaskEnvDepsImpl federated_task_env_deps_impl(
        absl::GetFlag(FLAGS_num_examples));
    fcp::client::LoggingEventPublisher logging_event_publisher;
    fcp::client::FilesImpl files_impl;
    fcp::client::LogManagerImpl log_manager_impl;
    const fcp::client::FlagsImpl flags;

    auto fl_runner_result = RunFederatedComputation(
        &federated_task_env_deps_impl, &logging_event_publisher, &files_impl,
        &log_manager_impl, &flags, absl::GetFlag(FLAGS_server),
        absl::GetFlag(FLAGS_api_key), absl::GetFlag(FLAGS_test_cert),
        absl::GetFlag(FLAGS_session), absl::GetFlag(FLAGS_population),
        absl::GetFlag(FLAGS_retry_token), absl::GetFlag(FLAGS_client_version),
        absl::GetFlag(FLAGS_attestation_string));
    if (fl_runner_result.ok()) {
      FCP_LOG(INFO) << "Run finished successfully; result: "
                    << fl_runner_result.value().DebugString();
    } else {
      FCP_LOG(ERROR) << "Error during run: " << fl_runner_result.status();
      return 1;
    }
  }
  return 0;
}
