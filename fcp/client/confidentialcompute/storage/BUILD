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

# SQLite helpers shared by the SQLite-backed stores.
cc_library(
    name = "sqlite_util",
    srcs = ["sqlite_util.cc"],
    hdrs = ["sqlite_util.h"],
    copts = FCP_COPTS,
    visibility = [
        "//fcp/client/confidentialcompute:__subpackages__",
        "//fcp/client/privatelogger:__subpackages__",
    ],
    deps = [
        "//fcp/client/diagnostics:diagnostic_logger",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@sqlite3",
    ],
)

cc_test(
    name = "sqlite_util_test",
    srcs = ["sqlite_util_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":sqlite_test_util",
        ":sqlite_util",
        "//fcp/client/diagnostics:diagnostic_logger",
        "//fcp/client/diagnostics:test_diagnostic_loggers",
        "//fcp/testing",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@sqlite3",
    ],
)

# Per-connection locking shared by the SQLite-backed stores.
cc_library(
    name = "sqlite_connection_mutex",
    srcs = ["sqlite_connection_mutex.cc"],
    hdrs = ["sqlite_connection_mutex.h"],
    copts = FCP_COPTS,
    visibility = [
        "//fcp/client/confidentialcompute:__subpackages__",
        "//fcp/client/privatelogger:__subpackages__",
    ],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
        "@sqlite3",
    ],
)

cc_test(
    name = "sqlite_connection_mutex_test",
    srcs = ["sqlite_connection_mutex_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":sqlite_connection_mutex",
        ":sqlite_test_util",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
        "@sqlite3",
    ],
)

# SQLite helpers for the tests of the SQLite-backed stores.
cc_library(
    name = "sqlite_test_util",
    testonly = True,
    srcs = ["sqlite_test_util.cc"],
    hdrs = ["sqlite_test_util.h"],
    copts = FCP_COPTS,
    visibility = [
        "//fcp/client/confidentialcompute:__subpackages__",
        "//fcp/client/privatelogger:__subpackages__",
    ],
    deps = [
        ":sqlite_util",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_for_library",
        "@sqlite3",
    ],
)
