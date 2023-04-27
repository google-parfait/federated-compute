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

# Aggregation Core API

load("//fcp:config.bzl", "FCP_COPTS")

package(
    default_visibility = ["//fcp/aggregation:internal"],
    licenses = ["notice"],  # Apache 2.0
)

# TODO(team): Create a "core" library that bundles all core libraries together.

TENSOR_SRCS = [
    "tensor.cc",
    "tensor_data.cc",
    "tensor_shape.cc",
    "input_tensor_list.cc",
]

TENSOR_HDRS = [
    "agg_vector.h",
    "agg_vector_iterator.h",
    "datatype.h",
    "tensor.h",
    "tensor_data.h",
    "tensor_shape.h",
    "tensor_spec.h",
    "mutable_vector_data.h",
    "input_tensor_list.h",
]

proto_library(
    name = "tensor_proto",
    srcs = ["tensor.proto"],
)

cc_proto_library(
    name = "tensor_cc_proto",
    deps = [":tensor_proto"],
)

cc_library(
    name = "tensor",
    srcs = TENSOR_SRCS,
    hdrs = TENSOR_HDRS,
    copts = FCP_COPTS,
    deps = [
        ":tensor_cc_proto",
        "//fcp/base",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
    ],
)

AGGREGATOR_SRCS = [
    "tensor_aggregator.cc",
    "tensor_aggregator_registry.cc",
]

AGGREGATOR_HDRS = [
    "agg_vector_aggregator.h",
    "one_dim_grouping_aggregator.h",
    "aggregator.h",
    "tensor_aggregator.h",
    "tensor_aggregator_factory.h",
    "tensor_aggregator_registry.h",
]

cc_library(
    name = "aggregator",
    srcs = AGGREGATOR_SRCS,
    hdrs = AGGREGATOR_HDRS,
    copts = FCP_COPTS,
    deps = [
        ":tensor",
        "//fcp/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "tensor_test",
    srcs = [
        "tensor_data_test.cc",
        "tensor_shape_test.cc",
        "tensor_test.cc",
    ],
    copts = FCP_COPTS,
    deps = [
        ":tensor",
        ":tensor_cc_proto",
        "//fcp/aggregation/testing",
        "//fcp/aggregation/testing:test_data",
        "//fcp/base",
        "//fcp/testing",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "agg_vector_test",
    srcs = ["agg_vector_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":tensor",
        "//fcp/aggregation/testing:test_data",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "aggregator_test",
    srcs = [
        "agg_vector_aggregator_test.cc",
    ],
    copts = FCP_COPTS,
    deps = [
        ":aggregator",
        ":tensor",
        ":tensor_cc_proto",
        "//fcp/aggregation/testing",
        "//fcp/aggregation/testing:test_data",
        "//fcp/base",
        "//fcp/testing",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "tensor_aggregator_registry_test",
    srcs = [
        "tensor_aggregator_registry_test.cc",
    ],
    copts = FCP_COPTS,
    deps = [
        ":aggregator",
        "//fcp/base",
        "//fcp/testing",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "federated_sum",
    srcs = [
        "federated_sum.cc",
    ],
    copts = FCP_COPTS,
    deps = [
        ":aggregator",
        ":tensor",
        "//fcp/base",
    ],
    alwayslink = 1,
)

cc_library(
    name = "composite_key_combiner",
    srcs = ["composite_key_combiner.cc"],
    hdrs = ["composite_key_combiner.h"],
    deps = [
        ":tensor",
        ":tensor_cc_proto",
        ":vector_string_data",
        "//fcp/base",
    ],
)

# Separate target from :tensor is required for vector_string_data as string_view is not yet
# supported for nanolibc.
cc_library(
    name = "vector_string_data",
    hdrs = ["vector_string_data.h"],
    deps = [":tensor"],
)

cc_test(
    name = "federated_sum_test",
    srcs = ["federated_sum_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":aggregator",
        ":federated_sum",
        ":tensor",
        "//fcp/aggregation/testing",
        "//fcp/aggregation/testing:test_data",
        "//fcp/testing",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "input_tensor_list_test",
    srcs = ["input_tensor_list_test.cc"],
    deps = [
        ":tensor",
        "//fcp/aggregation/testing:test_data",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "composite_key_combiner_test",
    srcs = ["composite_key_combiner_test.cc"],
    deps = [
        ":composite_key_combiner",
        ":tensor",
        ":tensor_cc_proto",
        "//fcp/aggregation/testing",
        "//fcp/aggregation/testing:test_data",
        "//fcp/base",
        "//fcp/testing",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "mutable_vector_data_test",
    srcs = ["mutable_vector_data_test.cc"],
    deps = [
        ":tensor",
        "//fcp/testing",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "vector_string_data_test",
    srcs = ["vector_string_data_test.cc"],
    deps = [
        ":tensor",
        ":vector_string_data",
        "//fcp/testing",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "one_dim_grouping_aggregator_test",
    srcs = ["one_dim_grouping_aggregator_test.cc"],
    deps = [
        ":aggregator",
        ":tensor",
        "//fcp/aggregation/testing",
        "//fcp/aggregation/testing:test_data",
        "//fcp/base",
        "//fcp/testing",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "federated_sum_bench",
    testonly = 1,
    srcs = ["federated_sum_bench.cc"],
    copts = FCP_COPTS,
    linkstatic = 1,
    deps = [
        ":aggregator",
        ":federated_sum",
        ":tensor",
        "@com_google_benchmark//:benchmark_main",
    ],
)
