# Copyright 2022 Google LLC
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

# Aggregation Protocol Package

load("//fcp:config.bzl", "FCP_COPTS")

package(
    default_visibility = ["//fcp/aggregation:internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "simple_aggregation",
    srcs = [
        "simple_aggregation_protocol.cc",
    ],
    hdrs = [
        "simple_aggregation_protocol.h",
    ],
    copts = FCP_COPTS,
    deps = [
        "//fcp/aggregation/core:aggregator",
        "//fcp/aggregation/core:federated_sum",
        "//fcp/aggregation/core:tensor",
        "//fcp/aggregation/protocol:aggregation_protocol",
        "//fcp/aggregation/protocol:cc_proto",
        "//fcp/aggregation/protocol:checkpoint_builder",
        "//fcp/aggregation/protocol:checkpoint_parser",
        "//fcp/aggregation/protocol:configuration_cc_proto",
        "//fcp/aggregation/protocol:resource_resolver",
        "//fcp/aggregation/tensorflow:converters",
        "//fcp/base",
        "//fcp/protos:plan_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = 1,
)

cc_test(
    name = "simple_aggregation_test",
    srcs = ["simple_aggregation_protocol_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":simple_aggregation",
        "//fcp/aggregation/core:aggregator",
        "//fcp/aggregation/core:tensor",
        "//fcp/aggregation/protocol:cc_proto",
        "//fcp/aggregation/protocol:checkpoint_builder",
        "//fcp/aggregation/protocol:checkpoint_parser",
        "//fcp/aggregation/protocol:configuration_cc_proto",
        "//fcp/aggregation/protocol/testing:mock_callback",
        "//fcp/aggregation/testing",
        "//fcp/aggregation/testing:test_data",
        "//fcp/base",
        "//fcp/base:scheduler",
        "//fcp/testing",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
    ],
)
