# Copyright 2026 Google LLC
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
    default_visibility = ["//fcp:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "log_database",
    hdrs = ["log_database.h"],
    copts = FCP_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        "//fcp/client/diagnostics:diagnostic_logger",
        "//fcp/client/privatelogger/common:log_entry",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "sqlite_log_database",
    srcs = ["sqlite_log_database.cc"],
    hdrs = ["sqlite_log_database.h"],
    copts = FCP_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":log_database",
        "//fcp/client/diagnostics:diagnostic_logger",
        "//fcp/client/privatelogger/common:log_entry",
        "//fcp/client/privatelogger/common:sqlite_connection_mutex",
        "//fcp/client/privatelogger/common:sqlite_util",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@sqlite3",
    ],
)

cc_test(
    name = "sqlite_log_database_test",
    srcs = ["sqlite_log_database_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":log_database",
        ":sqlite_log_database",
        "//fcp/client/diagnostics:diagnostic_logger",
        "//fcp/client/privatelogger/common:log_entry",
        "//fcp/testing",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@sqlite3",
    ],
)

cc_library(
    name = "log_storage",
    hdrs = ["log_storage.h"],
    copts = FCP_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        "//fcp/client/diagnostics:diagnostic_logger",
        "//fcp/client/privatelogger/common:log_entry",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "database_log_storage",
    srcs = ["database_log_storage.cc"],
    hdrs = ["database_log_storage.h"],
    copts = FCP_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":log_database",
        ":log_storage",
        "//fcp/base",
        "//fcp/client/diagnostics:diagnostic_logger",
        "//fcp/client/privatelogger/common:log_entry",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "database_log_storage_test",
    srcs = ["database_log_storage_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":database_log_storage",
        ":log_database",
        ":log_storage",
        ":sqlite_log_database",
        "//fcp/client/diagnostics:diagnostic_logger",
        "//fcp/client/privatelogger/common:log_entry",
        "//fcp/testing",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@sqlite3",
    ],
)
