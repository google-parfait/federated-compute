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

# Aggregation Protocol Package

load("//fcp:config.bzl", "FCP_COPTS")

package(
    default_visibility = ["//fcp/aggregation:internal"],
    licenses = ["notice"],  # Apache 2.0
)

proto_library(
    name = "proto",
    srcs = ["aggregation_protocol_messages.proto"],
    deps = [
        "//fcp/secagg/shared:proto",
    ],
)

cc_proto_library(
    name = "cc_proto",
    visibility = ["//visibility:public"],
    deps = [
        ":proto",
    ],
)

cc_library(
    name = "aggregation_protocol",
    hdrs = [
        "aggregation_protocol.h",
        "aggregation_protocol_factory.h",
    ],
    copts = FCP_COPTS,
    deps = [
        ":cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
    ],
)

cc_library(
    name = "simple_aggregation_protocol",
    srcs = ["simple_aggregation_protocol.cc"],
    hdrs = ["simple_aggregation_protocol.h"],
    copts = FCP_COPTS,
    deps = [
        ":aggregation_protocol",
        ":cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:cord",
    ],
)
