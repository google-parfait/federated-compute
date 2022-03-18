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

load("//fcp:config.bzl", "FCP_COPTS")
load("@org_tensorflow//tensorflow/core/platform:build_config.bzl", "tf_proto_library")

package(
    default_visibility = [
        "//fcp:internal",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "grpc_bidi_channel",
    hdrs = ["grpc_bidi_channel.h"],
    copts = FCP_COPTS,
    deps = [
        "//fcp/base",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "grpc_bidi_channel_test",
    srcs = ["grpc_bidi_channel_test.cc"],
    copts = FCP_COPTS,
    tags = ["local"],  # The certificate path is not accessible from a sandbox.
    deps = [
        ":grpc_bidi_stream",
        "//fcp/protos:cc_grpc",
        "//fcp/testing",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "grpc_bidi_stream",
    srcs = ["grpc_bidi_stream.cc"],
    hdrs = ["grpc_bidi_stream.h"],
    copts = FCP_COPTS,
    deps = [
        ":grpc_bidi_channel",
        "//fcp/base",
        "//fcp/base:status_converters",
        "//fcp/protocol:grpc_chunked_bidi_stream",
        "//fcp/protos:cc_grpc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_github_grpc_grpc//:grpc++_codegen_base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "grpc_bidi_stream_test",
    srcs = ["grpc_bidi_stream_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":fake_server",
        ":grpc_bidi_stream",
        ":test_helpers",
        "//fcp/base",
        "//fcp/base:scheduler",
        "//fcp/testing",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "fake_server",
    testonly = 1,
    srcs = ["fake_server.cc"],
    hdrs = ["fake_server.h"],
    copts = FCP_COPTS,
    visibility = ["//fcp:public_api_visibility"],
    deps = [
        ":grpc_bidi_stream",
        "//fcp/base",
        "//fcp/base:status_converters",
        "//fcp/protocol:grpc_chunked_bidi_stream",
        "//fcp/protos:cc_grpc",
        "//fcp/protos:federated_api_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_github_grpc_grpc//:grpc++_codegen_base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest",
    ],
)

# Interfaces used by the engine & protocol. These require platform specific
# implementations.
cc_library(
    name = "interfaces",
    srcs = [],
    hdrs = [
        "event_publisher.h",
        "files.h",
        "flags.h",
        "log_manager.h",
        "secagg_event_publisher.h",
        "task_environment.h",
    ],
    visibility = ["//fcp:public_api_visibility"],
    deps = [
        ":diag_codes_cc_proto",
        ":histogram_counters_cc_proto",
        ":simple_task_environment",
        "//fcp/client/engine:engine_cc_proto",
        "//fcp/protos:plan_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "federated_protocol",
    hdrs = ["federated_protocol.h"],
    deps = [
        "//fcp/client/engine:engine_cc_proto",
        "//fcp/protos:federated_api_cc_proto",
        "//fcp/protos:plan_cc_proto",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "federated_protocol_util",
    srcs = ["federated_protocol_util.cc"],
    hdrs = ["federated_protocol_util.h"],
    deps = [
        ":diag_codes_cc_proto",
        ":interfaces",
        "//fcp/protos:federated_api_cc_proto",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/time",
        "@com_google_protobuf//:cc_wkt_protos",
    ],
)

cc_test(
    name = "federated_protocol_util_test",
    srcs = ["federated_protocol_util_test.cc"],
    deps = [
        ":diag_codes_cc_proto",
        ":federated_protocol_util",
        ":test_helpers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "grpc_federated_protocol",
    srcs = ["grpc_federated_protocol.cc"],
    hdrs = [
        "grpc_federated_protocol.h",
    ],
    deps = [
        ":diag_codes_cc_proto",
        ":federated_protocol",
        ":federated_protocol_util",
        ":fl_runner_cc_proto",
        ":grpc_bidi_stream",
        ":interfaces",
        ":interruptible_runner",
        ":selector_context_cc_proto",
        "//fcp/base",
        "//fcp/client/engine:engine_cc_proto",
        "//fcp/client/http:http_client",
        "//fcp/client/http:in_memory_request_response",
        "//fcp/client/opstats:opstats_logger",
        "//fcp/protocol:grpc_chunked_bidi_stream",
        "//fcp/protos:federated_api_cc_proto",
        "//fcp/protos:plan_cc_proto",
        "//fcp/secagg/client",
        "//fcp/secagg/client:state_transition_listener",
        "//fcp/secagg/shared",
        "//fcp/secagg/shared:cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:cc_wkt_protos",
    ],
)

cc_test(
    name = "grpc_federated_protocol_test",
    srcs = ["grpc_federated_protocol_test.cc"],
    deps = [
        ":diag_codes_cc_proto",
        ":grpc_bidi_stream",
        ":grpc_federated_protocol",
        ":interfaces",
        ":interruptible_runner",
        ":test_helpers",
        "//fcp/base",
        "//fcp/client/engine:engine_cc_proto",
        "//fcp/client/http:http_client",
        "//fcp/client/http:test_helpers",
        "//fcp/protos:federated_api_cc_proto",
        "//fcp/protos:plan_cc_proto",
        "//fcp/secagg/client",
        "//fcp/secagg/shared",
        "//fcp/secagg/testing:client_mocks",
        "//fcp/secagg/testing:common_mocks",
        "//fcp/testing",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "simple_task_environment",
    srcs = ["simple_task_environment.cc"],
    hdrs = ["simple_task_environment.h"],
    copts = FCP_COPTS,
    visibility = ["//fcp:public_api_visibility"],
    deps = [
        ":selector_context_cc_proto",
        "//fcp/base",
        "//fcp/client/http:http_client",
        "//fcp/protos:plan_cc_proto",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "simple_task_environment_test",
    srcs = ["simple_task_environment_test.cc"],
    deps = [
        ":simple_task_environment",
        ":test_helpers",
        "//fcp/testing",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "federated_task_environment",
    srcs = ["federated_task_environment.cc"],
    hdrs = [
        "federated_task_environment.h",
    ],
    copts = FCP_COPTS,
    deps = [
        ":federated_protocol",
        ":interfaces",
        ":simple_task_environment",
        "//fcp/base",
        "//fcp/client/engine:engine_cc_proto",
        "//fcp/client/opstats:opstats_logger",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "federated_task_environment_test",
    srcs = ["federated_task_environment_test.cc"],
    deps = [
        ":federated_task_environment",
        ":test_helpers",
        "//fcp/base",
        "//fcp/protos:federated_api_cc_proto",
        "//fcp/protos:plan_cc_proto",
        "//fcp/testing",
        "@com_google_googletest//:gtest_main",
    ],
)

OPTIONAL_DEPS = select({
    "//conditions:default": [],
})

OPTIONAL_DEFINES = select({
    "//conditions:default": [],
})

cc_library(
    name = "fl_runner",
    srcs = ["fl_runner.cc"],
    hdrs = [
        "fl_runner.h",
    ],
    copts = FCP_COPTS,
    defines = OPTIONAL_DEFINES,
    visibility = ["//fcp:public_api_visibility"],
    deps = [
        ":federated_protocol",
        ":federated_protocol_util",
        ":grpc_federated_protocol",
        ":federated_task_environment",
        ":fl_runner_internal_cc_proto",
        ":fl_runner_cc_proto",
        ":histogram_counters_cc_proto",
        ":interfaces",
        ":interruptible_runner",
        ":simple_task_environment",
        ":selector_context_cc_proto",
        ":phase_logger",
        ":phase_logger_impl",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "//fcp/base",
        "//fcp/client/engine:engine_cc_proto",
        "//fcp/client/engine:plan_engine",
        "//fcp/client/engine:plan_engine_helpers",
        "//fcp/client/opstats:opstats_logger",
        "//fcp/protos:federated_api_cc_proto",
        "//fcp/protos:opstats_cc_proto",
        "//fcp/protos:plan_cc_proto",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ] + OPTIONAL_DEPS,
)

cc_library(
    name = "lc_runner",
    srcs = ["lc_runner.cc"],
    hdrs = [
        "lc_runner.h",
    ],
    copts = FCP_COPTS,
    defines = OPTIONAL_DEFINES,
    visibility = ["//fcp:public_api_visibility"],
    deps = [
        ":interfaces",
        ":selector_context_cc_proto",
        ":simple_task_environment",
        ":phase_logger",
        ":phase_logger_impl",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "//fcp/base",
        "//fcp/client/engine:plan_engine",
        "//fcp/client/engine:plan_engine_helpers",
        "//fcp/client/opstats:opstats_logger",
        "//fcp/protos:plan_cc_proto",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ] + OPTIONAL_DEPS,
)

cc_library(
    name = "phase_logger",
    hdrs = ["phase_logger.h"],
    deps = [
        "//fcp/client/opstats:opstats_logger",
        "//fcp/protos:federated_api_cc_proto",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "phase_logger_impl",
    srcs = ["phase_logger_impl.cc"],
    hdrs = ["phase_logger_impl.h"],
    deps = [
        ":interfaces",
        ":phase_logger",
        "//fcp/base",
        "//fcp/client/opstats:opstats_logger",
        "//fcp/protos:federated_api_cc_proto",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "phase_logger_impl_test",
    srcs = ["phase_logger_impl_test.cc"],
    deps = [
        ":phase_logger_impl",
        ":test_helpers",
        "//fcp/testing",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "interruptible_runner",
    srcs = ["interruptible_runner.cc"],
    hdrs = ["interruptible_runner.h"],
    visibility = ["//fcp:public_api_visibility"],
    deps = [
        ":diag_codes_cc_proto",
        ":interfaces",
        "//fcp/base",
        "//fcp/base:future",
        "//fcp/base:scheduler",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "fake_log_manager",
    hdrs = ["fake_log_manager.h"],
    visibility = ["//fcp:public_api_visibility"],
    deps = [
        ":diag_codes_cc_proto",
        ":interfaces",
        "//fcp/base",
    ],
)

cc_library(
    name = "fake_event_publisher",
    hdrs = ["fake_event_publisher.h"],
    visibility = ["//fcp:public_api_visibility"],
    deps = [
        ":interfaces",
        "//fcp/base",
    ],
)

cc_test(
    name = "interruptible_runner_test",
    srcs = ["interruptible_runner_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":diag_codes_cc_proto",
        ":interruptible_runner",
        ":test_helpers",
        "//fcp/testing",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

# Misc. classes to support embedding a fake client in unit or integration tests.
cc_library(
    name = "client_runner",
    testonly = True,
    hdrs = ["client_runner.h"],
    deps = [
        ":diag_codes_cc_proto",
        ":federated_task_environment",
        ":histogram_counters_cc_proto",
        ":interfaces",
        "//fcp/base",
        "//fcp/client/http:http_client",
        "//fcp/protos:plan_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
        "@com_google_protobuf//:cc_wkt_protos",
    ],
)

# A command line executable running most of the client side code to
# use for debugging and illustrating an example integration.
cc_binary(
    name = "client_runner_main",
    testonly = True,
    srcs = ["client_runner_main.cc"],
    deps = [
        ":client_runner",
        ":federated_task_environment",
        ":fl_runner",
        "//fcp/base",
        "//fcp/tensorflow:external_dataset_op_lib",
        "//fcp/tensorflow:task_eligibility_info_ops_lib",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/flags:usage",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:tensorflow_opensource",
    ],
)

cc_library(
    name = "test_helpers",
    testonly = True,
    srcs = ["test_helpers.cc"],
    hdrs = ["test_helpers.h"],
    deps = [
        ":federated_protocol",
        ":interfaces",
        ":phase_logger",
        ":simple_task_environment",
        "//fcp/base",
        "//fcp/client/http:http_client",
        "//fcp/client/opstats:opstats_db",
        "//fcp/client/opstats:opstats_logger",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

# Protocol buffers for logging. Those get serialized.
proto_library(
    name = "histogram_counters_proto",
    srcs = ["histogram_counters.proto"],
)

java_proto_library(
    name = "histogram_counters_java_proto",
    deps = [":histogram_counters_proto"],
)

cc_proto_library(
    name = "histogram_counters_cc_proto",
    deps = [":histogram_counters_proto"],
)

# Protocol buffers for FL Runner. These do not get serialized to disk.
proto_library(
    name = "fl_runner_proto",
    srcs = ["fl_runner.proto"],
    visibility = ["//fcp:public_api_visibility"],
    deps = [
        "//fcp/protos:federated_api_proto",
    ],
)

java_proto_library(
    name = "fl_runner_java_proto",
    visibility = ["//fcp:public_api_visibility"],
    deps = [":fl_runner_proto"],
)

cc_proto_library(
    name = "fl_runner_cc_proto",
    visibility = ["//fcp:public_api_visibility"],
    deps = [":fl_runner_proto"],
)

# Internal-only protocol buffers for FL Runner. These do not get serialized to disk.
tf_proto_library(
    name = "fl_runner_internal_proto",
    srcs = ["fl_runner_internal.proto"],
    protodeps = [
        "//fcp/client/engine:engine_proto",
        "@org_tensorflow//tensorflow/core:protos_all",
    ],
)

java_proto_library(
    name = "fl_runner_internal_java_proto",
    deps = [":fl_runner_internal_proto"],
)

# Allowing to refer to the cc library generated by the rule above in usual way:
alias(
    name = "fl_runner_internal_cc_proto",
    actual = "fl_runner_internal_proto_cc",
)

# Protocol buffers for logging. Those get serialized.
proto_library(
    name = "diag_codes_proto",
    srcs = ["diag_codes.proto"],
)

java_proto_library(
    name = "diag_codes_java_proto",
    deps = [":diag_codes_proto"],
)

cc_proto_library(
    name = "diag_codes_cc_proto",
    deps = [":diag_codes_proto"],
)

# --------------------------------------------------------------------
# selector_context.proto

proto_library(
    name = "selector_context_proto",
    srcs = ["selector_context.proto"],
)

cc_proto_library(
    name = "selector_context_cc_proto",
    visibility = ["//fcp:public_api_visibility"],
    deps = [":selector_context_proto"],
)

java_proto_library(
    name = "selector_context_java_proto",
    visibility = ["//fcp:public_api_visibility"],
    deps = [":selector_context_proto"],
)
