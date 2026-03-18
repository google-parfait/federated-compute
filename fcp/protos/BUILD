# Copyright 2019 Google LLC
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

load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:java_lite_proto_library.bzl", "java_lite_proto_library")
load("@com_google_protobuf//bazel:java_proto_library.bzl", "java_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@com_google_protobuf//bazel:py_proto_library.bzl", "py_proto_library")

default_visibility = [
    "//visibility:public",
]

package(
    default_visibility = default_visibility,
    licenses = ["notice"],  # Apache 2.0
)

# --------------------------------------------------------------------
# federated_api.proto

# The api protos.
proto_library(
    name = "federated_api_proto",
    srcs = ["federated_api.proto"],
    deps = [
        "//fcp/secagg/shared:proto",
        "@com_google_googleapis//google/rpc:code_proto",
        "@com_google_protobuf//:any_proto",
        "@com_google_protobuf//:duration_proto",
    ],
)

py_proto_library(
    name = "federated_api_py_pb2",
    deps = [":federated_api_proto"],
)

java_proto_library(
    name = "federated_api_java_proto",
    deps = [":federated_api_proto"],
)

cc_proto_library(
    name = "federated_api_cc_proto",
    deps = [":federated_api_proto"],
)

cc_grpc_library(
    name = "cc_grpc",
    srcs = [":federated_api_proto"],
    generate_mocks = True,
    grpc_only = True,
    deps = [":federated_api_cc_proto"],
)

# --------------------------------------------------------------------
# plan.proto

proto_library(
    name = "plan_proto",
    srcs = ["plan.proto"],
    visibility = default_visibility,
    deps = [
        ":data_type_proto",
        "//fcp/protos/confidentialcompute:windowing_schedule_proto",
        "@com_google_protobuf//:any_proto",
    ],
)

cc_proto_library(
    name = "plan_cc_proto",
    visibility = default_visibility + [
    ],
    deps = [":plan_proto"],
)

py_proto_library(
    name = "plan_py_pb2",
    deps = [":plan_proto"],
)

java_proto_library(
    name = "plan_java_proto",
    deps = [":plan_proto"],
)

# --------------------------------------------------------------------
# opstats.proto

proto_library(
    name = "opstats_proto",
    srcs = ["opstats.proto"],
    deps = [
        ":federated_api_proto",
        "@com_google_protobuf//:duration_proto",
        "@com_google_protobuf//:timestamp_proto",
    ],
)

cc_proto_library(
    name = "opstats_cc_proto",
    deps = [":opstats_proto"],
)

# --------------------------------------------------------------------
# population_eligibility_spec.proto

proto_library(
    name = "population_eligibility_spec_proto",
    srcs = ["population_eligibility_spec.proto"],
    deps = [
        ":plan_proto",
        "@com_google_protobuf//:duration_proto",
    ],
)

cc_proto_library(
    name = "population_eligibility_spec_cc_proto",
    deps = [":population_eligibility_spec_proto"],
)

java_proto_library(
    name = "population_eligibility_spec_java_proto",
    deps = [":population_eligibility_spec_proto"],
)

py_proto_library(
    name = "population_eligibility_spec_py_pb2",
    deps = [":population_eligibility_spec_proto"],
)

# --------------------------------------------------------------------
# task_eligibility_context.proto

proto_library(
    name = "task_eligibility_context_proto",
    srcs = ["task_eligibility_context.proto"],
)

java_proto_library(
    name = "task_eligibility_context_java_proto",
    deps = [":task_eligibility_context_proto"],
)

# --------------------------------------------------------------------
# data_type.proto

proto_library(
    name = "data_type_proto",
    srcs = ["data_type.proto"],
)

cc_proto_library(
    name = "data_type_cc_proto",
    deps = [":data_type_proto"],
)

py_proto_library(
    name = "data_type_py_pb2",
    deps = [":data_type_proto"],
)

java_lite_proto_library(
    name = "data_type_java_proto_lite",
    deps = [":data_type_proto"],
)
