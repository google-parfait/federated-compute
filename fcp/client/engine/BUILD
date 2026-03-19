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

load("@com_google_protobuf//bazel:java_proto_library.bzl", "java_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//fcp:config.bzl", "FCP_COPTS")

default_visibility = [
    "//fcp:internal",
]

package(
    default_applicable_licenses = ["//fcp:license"],
    default_visibility = default_visibility,
    licenses = ["notice"],
)

cc_library(
    name = "privacy_id_utils",
    srcs = [
        "privacy_id_utils.cc",
    ],
    hdrs = [
        "privacy_id_utils.h",
    ],
    copts = FCP_COPTS,
    visibility = default_visibility,
    deps = [
        "//fcp/base",
        "//fcp/base:digest",
        "//fcp/client:event_time_range_cc_proto",
        "//fcp/client:example_query_result_cc_proto",
        "//fcp/confidentialcompute:constants",
        "//fcp/confidentialcompute:time_window_utilities",
        "//fcp/protos:plan_cc_proto",
        "//fcp/protos/confidentialcompute:windowing_schedule_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_googleapis//google/type:datetime_cc_proto",
    ],
)

cc_test(
    name = "privacy_id_utils_test",
    srcs = [
        "privacy_id_utils_test.cc",
    ],
    copts = FCP_COPTS,
    visibility = default_visibility,
    deps = [
        ":privacy_id_utils",
        "//fcp/client:event_time_range_cc_proto",
        "//fcp/client:example_query_result_cc_proto",
        "//fcp/confidentialcompute:constants",
        "//fcp/protos:plan_cc_proto",
        "//fcp/protos/confidentialcompute:windowing_schedule_cc_proto",
        "//fcp/testing",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googleapis//google/type:datetime_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "example_query_plan_engine",
    srcs = [
        "example_query_plan_engine.cc",
    ],
    hdrs = [
        "example_query_plan_engine.h",
    ],
    copts = FCP_COPTS,
    visibility = default_visibility,
    deps = [
        ":common",
        ":example_iterator_factory",
        ":privacy_id_utils",
        "//fcp/base",
        "//fcp/client:converters",
        "//fcp/client:event_time_range_cc_proto",
        "//fcp/client:example_iterator_query_recorder",
        "//fcp/client:example_query_result_cc_proto",
        "//fcp/client:simple_task_environment",
        "//fcp/client/opstats:opstats_logger",
        "//fcp/client/tensorflow:tensorflow_runner",
        "//fcp/confidentialcompute:constants",
        "//fcp/protos:data_type_cc_proto",
        "//fcp/protos:plan_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/time",
        "@com_google_googleapis//google/type:datetime_cc_proto",
        "@com_google_protobuf//:protobuf",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:checkpoint_builder",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_builder",
    ],
)

cc_test(
    name = "example_query_plan_engine_test",
    srcs = ["example_query_plan_engine_test.cc"],
    deps = [
        ":common",
        ":example_iterator_factory",
        ":example_query_plan_engine",
        "//fcp/client:client_runner",
        "//fcp/client:event_time_range_cc_proto",
        "//fcp/client:example_query_result_cc_proto",
        "//fcp/client:federated_protocol",
        "//fcp/client:simple_task_environment",
        "//fcp/client:test_helpers",
        "//fcp/client/tensorflow:tensorflow_runner",
        "//fcp/confidentialcompute:constants",
        "//fcp/protos:data_type_cc_proto",
        "//fcp/protos:plan_cc_proto",
        "//fcp/protos/confidentialcompute:windowing_schedule_cc_proto",
        "//fcp/testing",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googleapis//google/type:datetime_cc_proto",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:checkpoint_header",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
    ],
)

cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = ["common.h"],
    deps = [
        ":engine_cc_proto",
        ":example_iterator_factory",
        ":external_dataset_iterator",
        "//fcp/base",
        "//fcp/client:example_iterator_query_recorder",
        "//fcp/client:federated_protocol",
        "//fcp/client:interfaces",
        "//fcp/client:simple_task_environment",
        "//fcp/client/opstats:opstats_logger",
        "//fcp/protos:data_type_cc_proto",
        "//fcp/protos:federated_api_cc_proto",
        "//fcp/protos:plan_cc_proto",
        "//fcp/protos/confidentialcompute:payload_metadata_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "example_iterator_factory",
    hdrs = ["example_iterator_factory.h"],
    copts = FCP_COPTS,
    visibility = default_visibility,
    deps = [
        "//fcp/client:selector_context_cc_proto",
        "//fcp/client:simple_task_environment",
        "//fcp/protos:plan_cc_proto",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "external_dataset_iterator",
    hdrs = ["external_dataset_iterator.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
    ],
)

# Runtime protos. Those do not typically get serialized.
proto_library(
    name = "engine_proto",
    srcs = ["engine.proto"],
    visibility = ["//visibility:public"],
)

cc_proto_library(
    name = "engine_cc_proto",
    visibility = ["//visibility:public"],
    deps = [":engine_proto"],
)

java_proto_library(
    name = "engine_java_proto",
    deps = [":engine_proto"],
)
