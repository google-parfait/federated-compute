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
    name = "test_scenarios",
    hdrs = ["test_scenarios.h"],
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
        ":test_scenarios",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_for_library",
    ],
)
