# Copyright 2019 Google LLC
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

load("//fcp:config.bzl", "FCP_COPTS")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "grpc_chunked_bidi_stream",
    hdrs = ["grpc_chunked_bidi_stream.h"],
    copts = FCP_COPTS,
    deps = [
        "//fcp/base",
        "//fcp/protos:cc_grpc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "grpc_chunked_bidi_stream_test",
    timeout = "long",
    srcs = ["grpc_chunked_bidi_stream_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":grpc_chunked_bidi_stream",
        "//fcp/base",
        "//fcp/client:fake_server",
        "//fcp/client:grpc_bidi_stream",
        "//fcp/protos:federated_api_cc_proto",
        "//fcp/testing",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
