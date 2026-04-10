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

package(
    default_applicable_licenses = ["//fcp:license"],
    default_visibility = [
        "//fcp:internal",
    ],
)

cc_library(
    name = "private_logger_internal_test_scenarios",
    hdrs = ["private_logger_internal_test_scenarios.h"],
    copts = ["-std=c++20"],
    deps = [
        "//fcp/client/privatelogger:private_logger_internal",
    ],
)

cc_library(
    name = "mock_private_logger_internal_setup",
    testonly = True,
    srcs = ["mock_private_logger_internal_setup.cc"],
    hdrs = ["mock_private_logger_internal_setup.h"],
    copts = ["-std=c++20"],
    deps = [
        ":private_logger_internal_mocks",
        ":private_logger_internal_test_scenarios",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_for_library",
    ],
)

cc_library(
    name = "private_logger_test_scenarios",
    hdrs = ["private_logger_test_scenarios.h"],
    copts = ["-std=c++20"],
    deps = [
        "//fcp/client/privatelogger:private_logger",
    ],
)

cc_library(
    name = "private_logger_internal_mocks",
    testonly = True,
    hdrs = ["private_logger_internal_mocks.h"],
    copts = ["-std=c++20"],
    deps = [
        "//fcp/client/privatelogger:private_logger_internal",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_for_library",
    ],
)

cc_library(
    name = "private_logger_mocks",
    testonly = True,
    hdrs = ["private_logger_mocks.h"],
    copts = ["-std=c++20"],
    deps = [
        "//fcp/client/privatelogger:private_logger",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_for_library",
        "@com_google_protobuf//:protobuf_lite",
    ],
)

cc_library(
    name = "mock_private_logger_setup",
    testonly = True,
    srcs = ["mock_private_logger_setup.cc"],
    hdrs = ["mock_private_logger_setup.h"],
    copts = ["-std=c++20"],
    deps = [
        ":private_logger_mocks",
        ":private_logger_test_scenarios",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_for_library",
    ],
)

cc_library(
    name = "task_contribution_tracker_mocks",
    testonly = True,
    hdrs = ["task_contribution_tracker_mocks.h"],
    copts = ["-std=c++20"],
    deps = [
        "//fcp/client/privatelogger:task_contribution_tracker",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_for_library",
    ],
)

cc_library(
    name = "task_contribution_tracker_test_scenarios",
    hdrs = ["task_contribution_tracker_test_scenarios.h"],
    copts = ["-std=c++20"],
    deps = [
        "//fcp/client/privatelogger:task_contribution_tracker",
    ],
)

cc_library(
    name = "mock_task_contribution_tracker_setup",
    testonly = True,
    srcs = ["mock_task_contribution_tracker_setup.cc"],
    hdrs = ["mock_task_contribution_tracker_setup.h"],
    copts = ["-std=c++20"],
    deps = [
        ":task_contribution_tracker_mocks",
        ":task_contribution_tracker_test_scenarios",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_for_library",
    ],
)

cc_library(
    name = "data_provider_test_scenarios",
    hdrs = ["data_provider_test_scenarios.h"],
    copts = ["-std=c++20"],
    deps = [
        "//fcp/client/privatelogger:data_provider",
    ],
)

cc_library(
    name = "data_provider_mocks",
    testonly = True,
    hdrs = ["data_provider_mocks.h"],
    copts = ["-std=c++20"],
    deps = [
        "//fcp/client:selector_context_cc_proto",
        "//fcp/client/privatelogger:data_provider",
        "//fcp/client/privatelogger:log_entry",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_for_library",
    ],
)

cc_library(
    name = "mock_data_provider_setup",
    testonly = True,
    srcs = ["mock_data_provider_setup.cc"],
    hdrs = ["mock_data_provider_setup.h"],
    copts = ["-std=c++20"],
    deps = [
        ":data_provider_mocks",
        ":data_provider_test_scenarios",
        "//fcp/client:selector_context_cc_proto",
        "//fcp/client/privatelogger:log_entry",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_for_library",
    ],
)

cc_library(
    name = "private_logger_connector_test_scenarios",
    hdrs = ["private_logger_connector_test_scenarios.h"],
    copts = ["-std=c++20"],
    deps = [
        "//fcp/client/privatelogger:private_logger_connector",
    ],
)

cc_library(
    name = "private_logger_connector_mocks",
    testonly = True,
    hdrs = ["private_logger_connector_mocks.h"],
    copts = ["-std=c++20"],
    deps = [
        "//fcp/client/privatelogger:data_provider",
        "//fcp/client/privatelogger:private_logger_connector",
        "//fcp/client/privatelogger:upload_outcome",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_for_library",
    ],
)

cc_library(
    name = "mock_private_logger_connector_setup",
    testonly = True,
    srcs = ["mock_private_logger_connector_setup.cc"],
    hdrs = ["mock_private_logger_connector_setup.h"],
    copts = ["-std=c++20"],
    deps = [
        ":private_logger_connector_mocks",
        ":private_logger_connector_test_scenarios",
        "//fcp/client/privatelogger:upload_outcome",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_for_library",
    ],
)
