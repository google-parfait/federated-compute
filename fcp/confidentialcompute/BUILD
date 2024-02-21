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
package(
    default_visibility = ["//fcp:internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "cose",
    srcs = ["cose.cc"],
    hdrs = ["cose.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//fcp/base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@libcppbor",
    ],
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
        "@boringssl//:crypto",
        "@com_google_absl//absl/cleanup",
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
        "//fcp/testing",
        "@boringssl//:crypto",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
