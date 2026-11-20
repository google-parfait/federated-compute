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

load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:java_proto_library.bzl", "java_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_python//python:proto.bzl", "py_proto_library")
load("//fcp:config.bzl", "FCP_COPTS")

package(
    default_applicable_licenses = ["//fcp:license"],
    default_visibility = ["//visibility:public"],
)

proto_library(
    name = "configuration_proto",
    srcs = ["configuration.proto"],
    deps = ["//fcp/aggregation/core:tensor_proto"],
)

cc_proto_library(
    name = "configuration_cc_proto",
    deps = [":configuration_proto"],
)

java_proto_library(
    name = "configuration_java_proto",
    deps = [":configuration_proto"],
)

py_proto_library(
    name = "configuration_py_pb2",
    deps = [":configuration_proto"],
)

# Parses a Configuration proto into intrinsics. Does not depend on any specific
# aggregator; binaries must link the aggregator libraries they need.
cc_library(
    name = "configuration",
    srcs = ["configuration.cc"],
    hdrs = ["configuration.h"],
    copts = FCP_COPTS,
    deps = [
        ":configuration_cc_proto",
        "//fcp/aggregation/core:aggregator",
        "//fcp/aggregation/core:intrinsic",
        "//fcp/aggregation/core:tensor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_protobuf//:protobuf_lite",
    ],
)

cc_test(
    name = "configuration_test",
    srcs = ["configuration_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":configuration",
        ":configuration_cc_proto",
        "//fcp/aggregation/core:aggregator",
        "//fcp/aggregation/core:intrinsic",
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
