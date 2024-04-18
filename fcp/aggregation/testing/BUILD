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

load("//fcp:config.bzl", "FCP_COPTS")

package(
    default_visibility = [
        "//fcp/aggregation:internal",
        "//fcp/client:__pkg__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "testing",
    testonly = True,
    srcs = [
        "testing.cc",
    ],
    hdrs = [
        "testing.h",
    ],
    copts = FCP_COPTS,
    deps = [
        "//fcp/aggregation/core:intrinsic",
        "//fcp/aggregation/core:tensor",
        "//fcp/base",
        "//fcp/testing",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/c:checkpoint_reader",
        "@org_tensorflow//tensorflow/c:tf_status_headers",
        "@org_tensorflow//tensorflow/c:tf_status_helper",
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/cc:ops",
        "@org_tensorflow//tensorflow/cc:scope",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core/platform:status",
        "@org_tensorflow//tensorflow/core/platform:tstring",
    ],
)

cc_library(
    name = "mocks",
    testonly = True,
    hdrs = ["mocks.h"],
    copts = FCP_COPTS,
    deps = [
        "//fcp/aggregation/core:tensor",
        "//fcp/aggregation/protocol:checkpoint_builder",
        "//fcp/aggregation/protocol:checkpoint_parser",
        "//fcp/aggregation/protocol:resource_resolver",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "test_data",
    testonly = True,
    hdrs = ["test_data.h"],
    copts = FCP_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        "//fcp/aggregation/core:tensor",
    ],
)
