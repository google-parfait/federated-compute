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

load("@org_tensorflow//tensorflow/core/platform:build_config.bzl", "tf_proto_library")
load("//fcp:config.bzl", "FCP_COPTS")

default_visibility = [
    "//fcp:internal",
]

package(
    default_visibility = default_visibility,
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "plan_engine",
    srcs = [
        "simple_plan_engine.cc",
    ],
    hdrs = [
        "simple_plan_engine.h",
    ],
    copts = FCP_COPTS,
    visibility = default_visibility + [
    ],
    deps = [
        ":common",
        ":plan_engine_helpers",
        ":tf_wrapper",
        "//fcp/base",
        "//fcp/client:histogram_counters_cc_proto",
        "//fcp/client:interfaces",
        "//fcp/client:interruptible_runner",
        "//fcp/client:simple_task_environment",
        "//fcp/client/opstats:opstats_logger",
        "//fcp/protos:plan_cc_proto",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

# A plan-engine independent wrapper around TF that supports cancellation.
cc_library(
    name = "tf_wrapper",
    srcs = ["tf_wrapper.cc"],
    hdrs = ["tf_wrapper.h"],
    copts = FCP_COPTS,
    visibility = ["//fcp:internal"],
    deps = [
        ":plan_engine_helpers",
        "//fcp/base",
        "//fcp/base:future",
        "//fcp/base:scheduler",
        "//fcp/client:diag_codes_cc_proto",
        "//fcp/client:interfaces",
        "//fcp/client:interruptible_runner",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_tensorflow//tensorflow/core:core_cpu",
    ],
)

cc_test(
    name = "tf_wrapper_test",
    srcs = ["tf_wrapper_test.cc"],
    deps = [
        ":tf_wrapper",
        "//fcp/testing",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_protobuf//:cc_wkt_protos",
    ],
)

cc_library(
    name = "plan_engine_helpers",
    srcs = ["plan_engine_helpers.cc"],
    hdrs = ["plan_engine_helpers.h"],
    copts = FCP_COPTS,
    visibility = default_visibility + [
    ],
    deps = [
        ":common",
        "//fcp/base",
        "//fcp/client:diag_codes_cc_proto",
        "//fcp/client:interfaces",
        "//fcp/client:simple_task_environment",
        "//fcp/client/opstats:opstats_example_store",
        "//fcp/client/opstats:opstats_logger",
        "//fcp/client/opstats:opstats_logger_impl",
        "//fcp/client/opstats:pds_backed_opstats_db",
        "//fcp/tensorflow:external_dataset",
        "//fcp/tensorflow:host_object",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@org_tensorflow//tensorflow/core:framework",
    ],
)

cc_library(
    name = "tflite_wrapper",
    srcs = ["tflite_wrapper.cc"],
    hdrs = ["tflite_wrapper.h"],
    deps = [
        ":caching_error_reporter",
        "//fcp/base",
        "//fcp/client:interfaces",
        "//fcp/client:interruptible_runner",
        "//fcp/client:simple_task_environment",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_tensorflow//tensorflow/lite:framework_stable",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/delegates/flex:delegate_only_runtime",
        "@org_tensorflow//tensorflow/lite/delegates/flex:util",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_test(
    name = "tflite_wrapper_test",
    srcs = ["tflite_wrapper_test.cc"],
    data = [
        "//fcp/client/engine/data:join_model.flatbuffer",
        "//fcp/client/engine/data:length_model.flatbuffer",
    ],
    deps = [
        ":tflite_wrapper",
        "//fcp/client:interruptible_runner",
        "//fcp/client:test_helpers",
        "//fcp/testing",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core/kernels:string_join_op",
    ],
)

cc_library(
    name = "tflite_plan_engine",
    srcs = [
        "tflite_plan_engine.cc",
    ],
    hdrs = [
        "tflite_plan_engine.h",
    ],
    copts = FCP_COPTS,
    deps = [
        ":common",
        ":plan_engine_helpers",
        ":tflite_wrapper",
        "//fcp/client:interfaces",
        "//fcp/client:interruptible_runner",
        "//fcp/client:simple_task_environment",
        "//fcp/client/opstats:opstats_logger",
        "//fcp/protos:plan_cc_proto",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "caching_error_reporter",
    srcs = ["caching_error_reporter.cc"],
    hdrs = ["caching_error_reporter.h"],
    deps = [
        "@com_google_absl//absl/synchronization",
        "@org_tensorflow//tensorflow/lite/core/api:error_reporter",
    ],
)

cc_test(
    name = "caching_error_reporter_test",
    srcs = ["caching_error_reporter_test.cc"],
    deps = [
        ":caching_error_reporter",
        "//fcp/testing",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/lite/core/api:error_reporter",
    ],
)

cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = ["common.h"],
    deps = [
        ":engine_cc_proto",
        "//fcp/base",
        "//fcp/protos:plan_cc_proto",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

# Runtime protos. Those do not typically get serialized.
tf_proto_library(
    name = "engine_proto",
    srcs = ["engine.proto"],
    visibility = default_visibility,
)

java_proto_library(
    name = "engine_java_proto",
    deps = [":engine_proto"],
)

# Allowing to refer to the cc library generated by the rule above in usual way:
alias(
    name = "engine_cc_proto",
    actual = "engine_proto_cc",
    visibility = default_visibility + [
    ],
)
