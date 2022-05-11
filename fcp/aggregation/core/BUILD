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

# Aggregation Core API

load("//fcp:config.bzl", "FCP_COPTS")

package(
    default_visibility = ["//fcp:internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "tensor",
    srcs = [
        "tensor_shape.cc",
    ],
    hdrs = [
        "datatype.h",
        "tensor_shape.h",
    ],
    copts = FCP_COPTS + ["-nostdlib"],
    deps = [],
)

cc_test(
    name = "tensor_test",
    srcs = [
        "tensor_shape_test.cc",
    ],
    copts = FCP_COPTS,
    deps = [
        ":tensor",
        "@com_google_googletest//:gtest_main",
    ],
)
