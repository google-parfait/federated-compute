# Copyright 2023 Google LLC
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

load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_grpc_library", "py_proto_library")
load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")

package(
    default_visibility = ["//visibility:public"],
)

proto_library(
    name = "access_policy_proto",
    srcs = ["access_policy.proto"],
)

cc_proto_library(
    name = "access_policy_cc_proto",
    deps = [":access_policy_proto"],
)

proto_library(
    name = "blob_header_proto",
    srcs = ["blob_header.proto"],
)

cc_proto_library(
    name = "blob_header_cc_proto",
    deps = [":blob_header_proto"],
)

proto_library(
    name = "ledger_proto",
    srcs = ["ledger.proto"],
    deps = [
        "@com_google_protobuf//:duration_proto",
        "@com_google_protobuf//:timestamp_proto",
    ],
)

cc_proto_library(
    name = "ledger_cc_proto",
    deps = [":ledger_proto"],
)

proto_library(
    name = "pipeline_transform_proto",
    srcs = ["pipeline_transform.proto"],
    deps = [
        "@com_google_protobuf//:any_proto",
    ],
)

proto_library(
    name = "tff_worker_configuration_proto",
    srcs = ["tff_worker_configuration.proto"],
)

cc_proto_library(
    name = "tff_worker_configuration_cc_proto",
    deps = [":tff_worker_configuration_proto"],
)

py_proto_library(
    name = "tff_worker_configuration_py_proto",
    deps = [":tff_worker_configuration_proto"],
)

cc_proto_library(
    name = "pipeline_transform_cc_proto",
    deps = [":pipeline_transform_proto"],
)

cc_grpc_library(
    name = "pipeline_transform_cc_grpc",
    srcs = [":pipeline_transform_proto"],
    grpc_only = True,
    deps = [":pipeline_transform_cc_proto"],
)

py_proto_library(
    name = "pipeline_transform_py_proto",
    deps = [":pipeline_transform_proto"],
)

py_grpc_library(
    name = "pipeline_transform_py_grpc",
    srcs = [":pipeline_transform_proto"],
    deps = [":pipeline_transform_py_proto"],
)
