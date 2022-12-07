# Copyright 2022 Google LLC
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

# Secure Aggregation Protocol Package

load("//fcp:config.bzl", "FCP_COPTS")

package(
    default_visibility = ["//fcp/aggregation:internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "secure_aggregation",
    srcs = [
        "secure_aggregation_protocol.cc",
    ],
    hdrs = [
        "secure_aggregation_protocol.h",
    ],
    copts = FCP_COPTS,
    deps = [
        "//fcp/aggregation/protocol:aggregation_protocol",
        "//fcp/aggregation/protocol:cc_proto",
        "//fcp/aggregation/protocol:configuration_cc_proto",
        "//fcp/base",
        "//fcp/base:scheduler",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:cord",
    ],
)

cc_test(
    name = "secure_aggregation_test",
    srcs = ["secure_aggregation_protocol_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":secure_aggregation",
        "//fcp/aggregation/protocol:configuration_cc_proto",
        "//fcp/aggregation/protocol/testing:mock_callback",
        "//fcp/aggregation/testing",
        "//fcp/testing",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)
