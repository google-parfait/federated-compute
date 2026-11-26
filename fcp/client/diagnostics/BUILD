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
load("//fcp:config.bzl", "FCP_COPTS")

package(
    default_applicable_licenses = ["//fcp:license"],
    default_visibility = ["//fcp:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "diagnostic_logger",
    hdrs = ["diagnostic_logger.h"],
    copts = FCP_COPTS,
    visibility = ["//visibility:public"],
)

# Diagnostic loggers for tests: a recording one and a shared no-op one.
cc_library(
    name = "test_diagnostic_loggers",
    testonly = True,
    hdrs = ["test_diagnostic_loggers.h"],
    copts = FCP_COPTS,
    deps = [
        ":diagnostic_logger",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)
