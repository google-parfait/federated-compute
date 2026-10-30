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
load("@rules_python//python:proto.bzl", "py_proto_library")
load("//fcp:config.bzl", "FCP_COPTS")

package(
    default_applicable_licenses = ["//fcp:license"],
    default_visibility = ["//visibility:public"],
)

proto_library(
    name = "tensor_proto",
    srcs = ["tensor.proto"],
)

cc_proto_library(
    name = "tensor_cc_proto",
    deps = [":tensor_proto"],
)

java_proto_library(
    name = "tensor_java_proto",
    deps = [":tensor_proto"],
)

py_proto_library(
    name = "tensor_py_pb2",
    deps = [":tensor_proto"],
)

cc_library(
    name = "tensor",
    srcs = [
        "datatype.cc",
        "tensor.cc",
        "tensor_data.cc",
        "tensor_shape.cc",
        "tensor_slice_data.cc",
        "tensor_spec.cc",
    ],
    hdrs = [
        "agg_vector.h",
        "agg_vector_iterator.h",
        "datatype.h",
        "mutable_vector_data.h",
        "scalar_data.h",
        "tensor.h",
        "tensor_data.h",
        "tensor_shape.h",
        "tensor_slice_data.h",
        "tensor_spec.h",
        "vector_data.h",
    ],
    copts = FCP_COPTS,
    deps = [
        ":tensor_cc_proto",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//src/google/protobuf/io",
    ],
)
