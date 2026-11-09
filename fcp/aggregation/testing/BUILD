# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "testing",
    testonly = True,
    srcs = ["testing.cc"],
    hdrs = ["testing.h"],
    copts = FCP_COPTS,
    deps = [
        "//fcp/aggregation/core:intrinsic",
        "//fcp/aggregation/core:tensor",
        "//fcp/base",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_for_library",
    ],
)

cc_library(
    name = "test_data",
    testonly = True,
    hdrs = ["test_data.h"],
    copts = FCP_COPTS,
    deps = ["//fcp/aggregation/core:tensor"],
)
