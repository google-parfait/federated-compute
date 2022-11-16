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

load("//fcp:config.bzl", "FCP_BAREMETAL_COPTS", "FCP_COPTS")

package(
    default_visibility = ["//fcp/aggregation:internal"],
    licenses = ["notice"],  # Apache 2.0
)

# TODO(team): Create a "core" library that bundles all core libraries together.

TENSOR_SRCS = [
    "tensor.cc",
    "tensor_data.cc",
    "tensor_shape.cc",
]

TENSOR_HDRS = [
    "agg_vector.h",
    "agg_vector_iterator.h",
    "datatype.h",
    "tensor.h",
    "tensor_data.h",
    "tensor_shape.h",
    "tensor_spec.h",
]

cc_library(
    name = "tensor",
    srcs = TENSOR_SRCS,
    hdrs = TENSOR_HDRS,
    copts = FCP_COPTS,
    deps = [
        "//fcp/base",
    ],
)

cc_library(
    name = "baremetal_tensor",
    srcs = TENSOR_SRCS,
    hdrs = TENSOR_HDRS,
    copts = FCP_BAREMETAL_COPTS,
    features = ["-use_header_modules"],
    linkstatic = True,
    deps = [
        "//fcp/base:baremetal_base",
    ],
)

AGGREGATOR_SRCS = [
    "tensor_aggregator.cc",
    "tensor_aggregator_registry.cc",
]

AGGREGATOR_HDRS = [
    "agg_vector_aggregator.h",
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

cc_library(
    name = "baremetal_aggregator",
    srcs = AGGREGATOR_SRCS,
    hdrs = AGGREGATOR_HDRS,
    copts = FCP_BAREMETAL_COPTS,
    features = ["-use_header_modules"],
    linkstatic = True,
    deps = [
        ":baremetal_tensor",
        "//fcp/base:baremetal_base",
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
        "//fcp/aggregation/testing:test_data",
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
        "//fcp/aggregation/testing",
        "//fcp/aggregation/testing:test_data",
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
        "//fcp/base",
    ],
    alwayslink = 1,
)

cc_test(
    name = "federated_sum_test",
    srcs = [
        "federated_sum_test.cc",
    ],
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
