# Copyright 2025 Google LLC
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

# This directory contains libraries for working with messages in
# //fcp/protos/confidentialcompute/payload_transparency.proto.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "rekor",
    srcs = ["rekor.cc"],
    hdrs = ["rekor.h"],
    deps = [
        ":signatures",
        "//fcp/base",
        "//fcp/protos/confidentialcompute:key_cc_proto",
        "//fcp/protos/confidentialcompute:payload_transparency_cc_proto",
        "@boringssl//:crypto",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "rekor_test",
    srcs = ["rekor_test.cc"],
    deps = [
        ":rekor",
        ":signatures",
        "//fcp/confidentialcompute:crypto",
        "//fcp/protos/confidentialcompute:key_cc_proto",
        "//fcp/protos/confidentialcompute:payload_transparency_cc_proto",
        "//fcp/testing",
        "@boringssl//:crypto",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "signatures",
    srcs = ["signatures.cc"],
    hdrs = ["signatures.h"],
    deps = [
        "//fcp/base",
        "//fcp/protos/confidentialcompute:key_cc_proto",
        "@boringssl//:crypto",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "signatures_test",
    srcs = ["signatures_test.cc"],
    deps = [
        ":signatures",
        "//fcp/base:digest",
        "//fcp/confidentialcompute:crypto",
        "//fcp/protos/confidentialcompute:key_cc_proto",
        "//fcp/testing",
        "@boringssl//:crypto",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
