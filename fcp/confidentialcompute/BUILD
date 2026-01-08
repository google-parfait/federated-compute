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
load("@bazel_skylib//lib:selects.bzl", "selects")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//fcp:config.bzl", "FCP_COPTS")

package(
    default_visibility = ["//fcp:internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "client_payload",
    srcs = ["client_payload.cc"],
    hdrs = ["client_payload.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//fcp/protos/confidentialcompute:aggregation_client_payload_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "private_state",
    hdrs = ["private_state.h"],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "client_payload_test",
    srcs = ["client_payload_test.cc"],
    deps = [
        ":client_payload",
        "//fcp/testing",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "time_window_utilities",
    srcs = [
        "time_window_utilities.cc",
    ],
    hdrs = [
        "time_window_utilities.h",
    ],
    copts = FCP_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        "//fcp/base",
        "//fcp/protos/confidentialcompute:windowing_schedule_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "time_window_utilities_test",
    srcs = [
        "time_window_utilities_test.cc",
    ],
    copts = FCP_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":time_window_utilities",
        "//fcp/protos/confidentialcompute:windowing_schedule_proto_cc",
        "//fcp/testing",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

# TODO: b/361182982 - Clean up these flags once the iOS toolchain supports C++20, or a better
# solution towards C++20 compatibility is found.
bool_flag(
    name = "disable_confidential_agg",
    build_setting_default = False,
)

config_setting(
    name = "disable_confidential_agg_flag_setting",
    flag_values = {":disable_confidential_agg": "True"},
)

config_setting(
    name = "is_ios",
    constraint_values = [
        "@platforms//os:ios",
    ],
)

selects.config_setting_group(
    name = "disable_confidential_agg_setting",
    match_any = [
        ":disable_confidential_agg_flag_setting",
        ":is_ios",
    ],
)

CA_OPTIONAL_DEFINES = select({
    ":disable_confidential_agg_setting": [],
    "//conditions:default": [
        "FCP_CLIENT_SUPPORT_CONFIDENTIAL_AGG",
    ],
})

cc_library(
    name = "cose",
    srcs = ["cose.cc"],
    hdrs = ["cose.h"],
    defines = CA_OPTIONAL_DEFINES,
    visibility = ["//visibility:public"],
    deps = [
        "//fcp/base",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ] + select({
        ":disable_confidential_agg_setting": [],
        "//conditions:default": [
            "@libcppbor",
        ],
    }),
)

cc_test(
    name = "cose_test",
    srcs = ["cose_test.cc"],
    deps = [
        ":cose",
        "//fcp/testing",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "crypto",
    srcs = ["crypto.cc"],
    hdrs = ["crypto.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cose",
        "//fcp/base",
        "//fcp/base:digest",
        "//fcp/protos/confidentialcompute:key_cc_proto",
        "@boringssl//:crypto",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "crypto_test",
    srcs = ["crypto_test.cc"],
    deps = [
        ":cose",
        ":crypto",
        "//fcp/base",
        "//fcp/protos/confidentialcompute:key_cc_proto",
        "//fcp/testing",
        "@boringssl//:crypto",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "constants",
    hdrs = ["constants.h"],
    visibility = ["//visibility:public"],
)
