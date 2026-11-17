# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Checkpoint aggregation and checkpoint format libraries.

load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//fcp:config.bzl", "FCP_COPTS")

package(
    default_applicable_licenses = ["//fcp:license"],
    default_visibility = ["//visibility:public"],
)

proto_library(
    name = "checkpoint_aggregator_proto",
    srcs = ["checkpoint_aggregator.proto"],
)

cc_proto_library(
    name = "checkpoint_aggregator_cc_proto",
    deps = [":checkpoint_aggregator_proto"],
)

cc_library(
    name = "checkpoint_builder",
    hdrs = ["checkpoint_builder.h"],
    copts = FCP_COPTS,
    deps = [
        "//fcp/aggregation/core:tensor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:cord",
    ],
)

cc_library(
    name = "checkpoint_parser",
    hdrs = ["checkpoint_parser.h"],
    copts = FCP_COPTS,
    deps = [
        "//fcp/aggregation/core:tensor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:cord",
    ],
)

cc_library(
    name = "checkpoint_header",
    hdrs = ["checkpoint_header.h"],
    copts = FCP_COPTS,
)

cc_library(
    name = "checkpoint_aggregator",
    srcs = ["checkpoint_aggregator.cc"],
    hdrs = ["checkpoint_aggregator.h"],
    copts = FCP_COPTS,
    deps = [
        ":checkpoint_aggregator_cc_proto",
        ":checkpoint_builder",
        ":checkpoint_parser",
        "//fcp/aggregation/core:aggregator",
        "//fcp/aggregation/core:intrinsic",
        "//fcp/aggregation/core:tensor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_protobuf//:protobuf_lite",
    ],
)

cc_library(
    name = "federated_compute_checkpoint_builder",
    srcs = ["federated_compute_checkpoint_builder.cc"],
    hdrs = ["federated_compute_checkpoint_builder.h"],
    copts = FCP_COPTS,
    deps = [
        ":checkpoint_builder",
        ":checkpoint_header",
        "//fcp/aggregation/core:tensor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_protobuf//src/google/protobuf/io",
    ],
)

cc_library(
    name = "federated_compute_checkpoint_parser",
    srcs = ["federated_compute_checkpoint_parser.cc"],
    hdrs = ["federated_compute_checkpoint_parser.h"],
    copts = FCP_COPTS,
    deps = [
        ":checkpoint_header",
        ":checkpoint_parser",
        "//fcp/aggregation/core:tensor",
        "//fcp/aggregation/core:tensor_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_protobuf//src/google/protobuf/io",
    ],
)

cc_library(
    name = "in_memory_checkpoint_parser",
    hdrs = ["in_memory_checkpoint_parser.h"],
    copts = FCP_COPTS,
    deps = [
        ":checkpoint_parser",
        "//fcp/aggregation/core:tensor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "checkpoint_aggregator_test",
    srcs = ["checkpoint_aggregator_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":checkpoint_aggregator",
        "//fcp/aggregation/core:aggregator",
        "//fcp/aggregation/core:federated_state_cc_proto",
        "//fcp/aggregation/core:federated_sum",
        "//fcp/aggregation/core:intrinsic",
        "//fcp/aggregation/core:tensor",
        "//fcp/aggregation/group_by:group_by_aggregator",
        "//fcp/aggregation/group_by:grouping_federated_sum",
        "//fcp/aggregation/testing",
        "//fcp/aggregation/testing:mocks",
        "//fcp/aggregation/testing:test_data",
        "//fcp/base:scheduler",
        "//fcp/testing",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "federated_compute_checkpoint_builder_test",
    srcs = ["federated_compute_checkpoint_builder_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":checkpoint_builder",
        ":checkpoint_header",
        ":federated_compute_checkpoint_builder",
        "//fcp/aggregation/core:tensor",
        "//fcp/aggregation/core:tensor_cc_proto",
        "//fcp/aggregation/testing:test_data",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//src/google/protobuf/io",
    ],
)

cc_test(
    name = "federated_compute_checkpoint_parser_test",
    srcs = ["federated_compute_checkpoint_parser_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":checkpoint_builder",
        ":checkpoint_parser",
        ":federated_compute_checkpoint_builder",
        ":federated_compute_checkpoint_parser",
        "//fcp/aggregation/core:tensor",
        "//fcp/aggregation/core:tensor_cc_proto",
        "//fcp/aggregation/testing",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "in_memory_checkpoint_parser_test",
    srcs = ["in_memory_checkpoint_parser_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":in_memory_checkpoint_parser",
        "//fcp/aggregation/core:tensor",
        "//fcp/aggregation/testing",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
