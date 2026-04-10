# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["//fcp:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "private_logger_internal",
    hdrs = ["private_logger_internal.h"],
    copts = ["-std=c++20"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "private_logger",
    hdrs = ["private_logger.h"],
    copts = ["-std=c++20"],
    deps = [
        ":private_logger_internal",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_protobuf//:protobuf_lite",
    ],
)

cc_test(
    name = "private_logger_internal_test",
    srcs = ["private_logger_internal_test.cc"],
    copts = ["-std=c++20"],
    deps = [
        "//fcp/client/privatelogger/tests:mock_private_logger_internal_setup",
        "//fcp/client/privatelogger/tests:private_logger_internal_mocks",
        "//fcp/client/privatelogger/tests:private_logger_internal_test_scenarios",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "log_entry",
    hdrs = ["log_entry.h"],
    copts = ["-std=c++20"],
)

cc_library(
    name = "data_provider",
    hdrs = ["data_provider.h"],
    copts = ["-std=c++20"],
    deps = [
        ":log_entry",
        "//fcp/client:selector_context_cc_proto",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "task_contribution_tracker",
    hdrs = ["task_contribution_tracker.h"],
    copts = ["-std=c++20"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "upload_outcome",
    hdrs = ["upload_outcome.h"],
    copts = ["-std=c++20"],
)

cc_library(
    name = "private_logger_connector",
    hdrs = ["private_logger_connector.h"],
    copts = ["-std=c++20"],
    deps = [
        ":data_provider",
        ":upload_outcome",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "private_logger_test",
    srcs = ["private_logger_test.cc"],
    copts = ["-std=c++20"],
    deps = [
        "//fcp/client:selector_context_cc_proto",
        "//fcp/client/privatelogger/tests:mock_private_logger_setup",
        "//fcp/client/privatelogger/tests:private_logger_mocks",
        "//fcp/client/privatelogger/tests:private_logger_test_scenarios",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "data_provider_test",
    srcs = ["data_provider_test.cc"],
    copts = ["-std=c++20"],
    deps = [
        "//fcp/client:selector_context_cc_proto",
        "//fcp/client/privatelogger/tests:data_provider_mocks",
        "//fcp/client/privatelogger/tests:data_provider_test_scenarios",
        "//fcp/client/privatelogger/tests:mock_data_provider_setup",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "task_contribution_tracker_test",
    srcs = ["task_contribution_tracker_test.cc"],
    copts = ["-std=c++20"],
    deps = [
        "//fcp/client/privatelogger/tests:mock_task_contribution_tracker_setup",
        "//fcp/client/privatelogger/tests:task_contribution_tracker_mocks",
        "//fcp/client/privatelogger/tests:task_contribution_tracker_test_scenarios",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "private_logger_connector_test",
    srcs = ["private_logger_connector_test.cc"],
    copts = ["-std=c++20"],
    deps = [
        ":upload_outcome",
        "//fcp/client/privatelogger/tests:data_provider_mocks",
        "//fcp/client/privatelogger/tests:mock_private_logger_connector_setup",
        "//fcp/client/privatelogger/tests:private_logger_connector_mocks",
        "//fcp/client/privatelogger/tests:private_logger_connector_test_scenarios",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)
