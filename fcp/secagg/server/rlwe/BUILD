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

# Description:
#   RLWE SecAgg server-specific components.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//fcp:config.bzl", "FCP_COPTS")

package(
    default_applicable_licenses = ["//fcp:license"],
    licenses = ["notice"],
)

cc_library(
    name = "rlwe",
    srcs = ["rlwe_secagg_server_protocol_impl.cc"],
    hdrs = [
        "rlwe_secagg_server_protocol_impl.h",
    ],
    copts = FCP_COPTS,
    deps = [
        "//fcp/base",
        "//fcp/secagg/server:secagg_scheduler",
        "//fcp/secagg/server:secagg_server_protocol_impl",
        "//fcp/secagg/shared",
        "//fcp/secagg/shared/rlwe",
        "//third_party/rlwe:montgomery",
        "//third_party/rlwe:polynomial",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_protobuf//:any_cc_proto",
    ],
)

cc_test(
    name = "state-tests",
    size = "small",
    srcs = [
        "secagg_server_prng_running_state_rlwe_test.cc",
        "secagg_server_r0_advertise_keys_state_rlwe_test.cc",
        "secagg_server_r2_masked_input_coll_state_rlwe_test.cc",
    ],
    copts = FCP_COPTS,
    deps = [
        ":rlwe",
        "//fcp/base",
        "//fcp/base:scheduler",
        "//fcp/secagg/server",
        "//fcp/secagg/server:secagg_scheduler",
        "//fcp/secagg/server:secret_sharing_graph_factory",
        "//fcp/secagg/server:send_to_clients_interface",
        "//fcp/secagg/server:server_cc_proto",
        "//fcp/secagg/shared",
        "//fcp/secagg/shared:cc_proto",
        "//fcp/secagg/shared/rlwe",
        "//fcp/secagg/testing",
        "//fcp/secagg/testing:common_mocks",
        "//fcp/secagg/testing/server:async_runner",
        "//fcp/secagg/testing/server:server_mocks",
        "//fcp/testing",
        "//fcp/tracing:test_tracing_recorder",
        "//third_party/rlwe:constants",
        "//third_party/rlwe:montgomery",
        "//third_party/rlwe:polynomial",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
