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

# Non-DP GroupBy aggregation libraries.

load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//fcp:config.bzl", "FCP_COPTS")

package(
    default_applicable_licenses = ["//fcp:license"],
    default_visibility = ["//visibility:public"],
)

proto_library(
    name = "group_by_state_proto",
    srcs = ["group_by_state.proto"],
    deps = ["//fcp/aggregation/core:tensor_proto"],
)

cc_proto_library(
    name = "group_by_state_cc_proto",
    deps = [":group_by_state_proto"],
)

cc_library(
    name = "fedsql_constants",
    hdrs = ["fedsql_constants.h"],
    copts = FCP_COPTS,
)

cc_library(
    name = "contributors_to_groups",
    srcs = ["contributors_to_groups.cc"],
    hdrs = ["contributors_to_groups.h"],
    copts = FCP_COPTS,
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "partitioner",
    srcs = ["partitioner.cc"],
    hdrs = ["partitioner.h"],
    copts = FCP_COPTS,
    deps = [
        "//fcp/aggregation/core:tensor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "composite_key_combiner",
    srcs = ["composite_key_combiner.cc"],
    hdrs = ["composite_key_combiner.h"],
    copts = FCP_COPTS,
    deps = [
        "//fcp/aggregation/core:aggregator",
        "//fcp/aggregation/core:tensor",
        "//fcp/aggregation/core:tensor_cc_proto",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "one_dim_grouping_aggregator",
    srcs = ["one_dim_grouping_aggregator.cc"],
    hdrs = ["one_dim_grouping_aggregator.h"],
    copts = FCP_COPTS,
    deps = [
        ":group_by_state_cc_proto",
        ":partitioner",
        "//fcp/aggregation/core:aggregator",
        "//fcp/aggregation/core:intrinsic",
        "//fcp/aggregation/core:tensor",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "grouping_federated_sum",
    srcs = ["grouping_federated_sum.cc"],
    copts = FCP_COPTS,
    deps = [
        ":group_by_state_cc_proto",
        ":one_dim_grouping_aggregator",
        "//fcp/aggregation/core:aggregator",
        "//fcp/aggregation/core:intrinsic",
        "//fcp/aggregation/core:tensor",
        "//fcp/aggregation/core:tensor_cc_proto",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library(
    name = "group_by_aggregator",
    srcs = ["group_by_aggregator.cc"],
    hdrs = ["group_by_aggregator.h"],
    copts = FCP_COPTS,
    deps = [
        ":composite_key_combiner",
        ":contributors_to_groups",
        ":fedsql_constants",
        ":group_by_state_cc_proto",
        ":one_dim_grouping_aggregator",
        ":partitioner",
        "//fcp/aggregation/core:aggregator",
        "//fcp/aggregation/core:intrinsic",
        "//fcp/aggregation/core:tensor",
        "//fcp/aggregation/core:tensor_cc_proto",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:protobuf_lite",
    ],
    alwayslink = 1,
)

cc_test(
    name = "contributors_to_groups_test",
    srcs = ["contributors_to_groups_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":contributors_to_groups",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "partitioner_test",
    srcs = ["partitioner_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":partitioner",
        "//fcp/aggregation/core:tensor",
        "//fcp/aggregation/testing",
        "//fcp/aggregation/testing:test_data",
        "//fcp/testing",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "composite_key_combiner_test",
    srcs = ["composite_key_combiner_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":composite_key_combiner",
        "//fcp/aggregation/core:aggregator",
        "//fcp/aggregation/core:tensor",
        "//fcp/aggregation/core:tensor_cc_proto",
        "//fcp/aggregation/testing",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "one_dim_grouping_aggregator_test",
    srcs = ["one_dim_grouping_aggregator_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":group_by_state_cc_proto",
        ":one_dim_grouping_aggregator",
        ":partitioner",
        "//fcp/aggregation/core:tensor",
        "//fcp/aggregation/testing",
        "//fcp/aggregation/testing:test_data",
        "//fcp/testing",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "grouping_federated_sum_test",
    srcs = ["grouping_federated_sum_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":grouping_federated_sum",
        ":one_dim_grouping_aggregator",
        "//fcp/aggregation/core:aggregator",
        "//fcp/aggregation/core:intrinsic",
        "//fcp/aggregation/core:tensor",
        "//fcp/aggregation/testing",
        "//fcp/aggregation/testing:test_data",
        "//fcp/testing",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "group_by_aggregator_test",
    srcs = ["group_by_aggregator_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":group_by_aggregator",
        ":grouping_federated_sum",
        "//fcp/aggregation/core:aggregator",
        "//fcp/aggregation/core:intrinsic",
        "//fcp/aggregation/core:tensor",
        "//fcp/aggregation/core:tensor_cc_proto",
        "//fcp/aggregation/testing",
        "//fcp/aggregation/testing:test_data",
        "//fcp/testing",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "group_by_bench",
    testonly = 1,
    srcs = ["group_by_bench.cc"],
    copts = FCP_COPTS,
    linkstatic = 1,
    deps = [
        ":group_by_aggregator",
        ":grouping_federated_sum",
        "//fcp/aggregation/core:aggregator",
        "//fcp/aggregation/core:intrinsic",
        "//fcp/aggregation/core:mutable_string_data",
        "//fcp/aggregation/core:tensor",
        "//fcp/aggregation/core:tensor_cc_proto",
        "@com_google_benchmark//:benchmark_main",
    ],
)

cc_binary(
    name = "composite_key_combiner_bench",
    testonly = 1,
    srcs = ["composite_key_combiner_bench.cc"],
    copts = FCP_COPTS,
    linkstatic = 1,
    deps = [
        ":composite_key_combiner",
        "//fcp/aggregation/core:aggregator",
        "//fcp/aggregation/core:mutable_string_data",
        "//fcp/aggregation/core:tensor",
        "//fcp/aggregation/core:tensor_cc_proto",
        "@com_google_benchmark//:benchmark_main",
    ],
)
