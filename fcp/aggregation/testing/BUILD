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
    default_visibility = ["//fcp/aggregation:internal"],
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
        "//fcp/aggregation/core:tensor",
        "//fcp/base",
        "//fcp/tensorflow:status",
        "//fcp/testing",
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
    ],
)

cc_library(
    name = "test_data",
    testonly = True,
    hdrs = [
        "test_data.h",
    ],
    copts = FCP_COPTS,
    deps = [
        "//fcp/aggregation/core:tensor",
    ],
)
