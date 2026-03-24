# Copyright 2021 Google LLC
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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//fcp:config.bzl", "FCP_COPTS")

package(
    default_applicable_licenses = ["//fcp:license"],
    default_visibility = [
        "//fcp:internal",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "opstats_db",
    hdrs = ["opstats_db.h"],
    deps = [
        "//fcp/protos:opstats_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "opstats_logger",
    hdrs = ["opstats_logger.h"],
    deps = [
        ":opstats_db",
        "//fcp/client:interfaces",
        "//fcp/protos:federated_api_cc_proto",
        "//fcp/protos:opstats_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "opstats_logger_impl",
    srcs = ["opstats_logger_impl.cc"],
    hdrs = ["opstats_logger_impl.h"],
    copts = FCP_COPTS,
    deps = [
        ":opstats_db",
        ":opstats_logger",
        ":pds_backed_opstats_db",
        "//fcp/base",
        "//fcp/base:time_util",
        "//fcp/client:diag_codes_cc_proto",
        "//fcp/client:histogram_counters_cc_proto",
        "//fcp/client:interfaces",
        "//fcp/protos:federated_api_cc_proto",
        "//fcp/protos:opstats_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_protobuf//:time_util",
    ],
)

cc_test(
    name = "opstats_logger_impl_test",
    srcs = ["opstats_logger_impl_test.cc"],
    deps = [
        ":opstats_logger",
        ":opstats_logger_impl",
        ":pds_backed_opstats_db",
        "//fcp/base:time_util",
        "//fcp/client:diag_codes_cc_proto",
        "//fcp/client:histogram_counters_cc_proto",
        "//fcp/client:interfaces",
        "//fcp/client:test_helpers",
        "//fcp/client/engine:engine_cc_proto",
        "//fcp/protos:opstats_cc_proto",
        "//fcp/testing",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:time_util",
        "@protodatastore_cpp//protostore:file-storage",
    ],
)

cc_library(
    name = "opstats_example_store",
    srcs = ["opstats_example_store.cc"],
    hdrs = ["opstats_example_store.h"],
    deps = [
        "//fcp/client:diag_codes_cc_proto",
        "//fcp/client:interfaces",
        "//fcp/client/engine:example_iterator_factory",
        "//fcp/protos:plan_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "opstats_utils",
    srcs = ["opstats_utils.cc"],
    hdrs = ["opstats_utils.h"],
    deps = [
        "//fcp/base",
        "//fcp/protos:federated_api_cc_proto",
        "//fcp/protos:opstats_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_protobuf//:duration_cc_proto",
        "@com_google_protobuf//:time_util",
        "@com_google_protobuf//:timestamp_cc_proto",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_test(
    name = "opstats_utils_test",
    srcs = ["opstats_utils_test.cc"],
    deps = [
        ":opstats_utils",
        "//fcp/protos:opstats_cc_proto",
        "//fcp/testing",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:timestamp_cc_proto",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_library(
    name = "pds_backed_opstats_db",
    srcs = ["pds_backed_opstats_db.cc"],
    hdrs = ["pds_backed_opstats_db.h"],
    deps = [
        ":opstats_db",
        "//fcp/base",
        "//fcp/client:diag_codes_cc_proto",
        "//fcp/client:histogram_counters_cc_proto",
        "//fcp/client:interfaces",
        "//fcp/protos:opstats_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_protobuf//:protobuf_lite",
        "@com_google_protobuf//:time_util",
        "@com_google_protobuf//:timestamp_cc_proto",
        "@protodatastore_cpp//protostore:file-storage",
        "@protodatastore_cpp//protostore:proto-data-store",
    ],
)

cc_test(
    name = "pds_backed_opstats_db_test",
    srcs = ["pds_backed_opstats_db_test.cc"],
    deps = [
        ":opstats_db",
        ":pds_backed_opstats_db",
        "//fcp/base",
        "//fcp/client:diag_codes_cc_proto",
        "//fcp/client:histogram_counters_cc_proto",
        "//fcp/client:test_helpers",
        "//fcp/client/engine:engine_cc_proto",
        "//fcp/protos:opstats_cc_proto",
        "//fcp/testing",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:time_util",
        "@protodatastore_cpp//protostore:file-storage",
    ],
)
