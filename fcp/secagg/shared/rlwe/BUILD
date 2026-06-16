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
#   SecAgg RLWE components

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//fcp:config.bzl", "FCP_COPTS")

package(
    default_applicable_licenses = ["//fcp:license"],
    default_visibility = ["//fcp:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "rlwe",
    srcs = [
        "input_vector_rlwe_specification.cc",
        "map_of_rlwe_masks.cc",
        "secagg_rlwe_vector.cc",
    ],
    hdrs = [
        "input_vector_rlwe_specification.h",
        "map_of_rlwe_masks.h",
        "rlwe_prng_adapter.h",
        "secagg_rlwe_params.h",
        "secagg_rlwe_vector.h",
    ],
    deps = [
        "//fcp/base",
        "//fcp/secagg/shared",
        "//third_party/rlwe:error_params",
        "//third_party/rlwe:montgomery",
        "//third_party/rlwe:polynomial",
        "//third_party/rlwe:serialization_cc_proto",
        "//third_party/rlwe:statusor_fork",
        "//third_party/rlwe:symmetric_encryption",
        "//third_party/rlwe/prng",
        "//third_party/rlwe/prng:single_thread_chacha_prng",
        "@boringssl//:crypto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
    ],
)

cc_test(
    name = "map_of_rlwe_masks_bench",
    size = "large",
    srcs = [
        "map_of_rlwe_masks_bench.cc",
    ],
    copts = FCP_COPTS,
    linkstatic = 1,
    deps = [
        ":rlwe",
        "//fcp/secagg/shared",
        "//third_party/rlwe:symmetric_encryption",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "map_of_rlwe_masks_test",
    size = "small",
    srcs = [
        "map_of_rlwe_masks_test.cc",
    ],
    copts = FCP_COPTS,
    deps = [
        ":rlwe",
        "//fcp/secagg/shared",
        "//third_party/rlwe:symmetric_encryption",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "secagg_rlwe_vector_test",
    size = "small",
    srcs = ["secagg_rlwe_vector_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":rlwe",
        "//fcp/secagg/shared",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:differencer",
    ],
)

cc_test(
    name = "input_vector_rlwe_specification_test",
    size = "small",
    srcs = [
        "input_vector_rlwe_specification_test.cc",
    ],
    copts = FCP_COPTS,
    deps = [
        ":rlwe",
        "//fcp/secagg/shared",
        "@com_google_googletest//:gtest_main",
    ],
)
