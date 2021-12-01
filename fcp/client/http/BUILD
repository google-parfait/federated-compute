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

package(
    default_visibility = [
        "//fcp:internal",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "http_client",
    srcs = ["http_client.cc"],
    hdrs = ["http_client.h"],
    visibility = ["//fcp:public_api_visibility"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "http_client_test",
    srcs = ["http_client_test.cc"],
    deps = [
        ":http_client",
        "//fcp/base",
        "//fcp/testing",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "in_memory_request_response",
    srcs = ["in_memory_request_response.cc"],
    hdrs = ["in_memory_request_response.h"],
    deps = [
        ":http_client",
        "//fcp/base",
        "//fcp/client:interruptible_runner",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "in_memory_request_response_test",
    srcs = ["in_memory_request_response_test.cc"],
    deps = [
        ":http_client",
        ":in_memory_request_response",
        ":test_helpers",
        "//fcp/base",
        "//fcp/client:diag_codes_cc_proto",
        "//fcp/client:interruptible_runner",
        "//fcp/client:test_helpers",
        "//fcp/testing",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "test_helpers",
    testonly = 1,
    srcs = ["test_helpers.cc"],
    hdrs = ["test_helpers.h"],
    deps = [
        ":http_client",
        "//fcp/base",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)
