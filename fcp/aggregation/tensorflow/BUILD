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
    name = "converters",
    srcs = ["converters.cc"],
    hdrs = ["converters.h"],
    copts = FCP_COPTS,
    deps = [
        "//fcp/aggregation/core:tensor",
        "//fcp/base",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "converters_test",
    srcs = ["converters_test.cc"],
    deps = [
        ":converters",
        "//fcp/aggregation/core:tensor",
        "//fcp/aggregation/testing",
        "//fcp/base",
        "//fcp/testing",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "checkpoint_reader",
    srcs = ["checkpoint_reader.cc"],
    hdrs = ["checkpoint_reader.h"],
    copts = FCP_COPTS,
    deps = [
        ":converters",
        "//fcp/aggregation/core:tensor",
        "//fcp/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@org_tensorflow//tensorflow/c:checkpoint_reader",
        "@org_tensorflow//tensorflow/c:tf_status_headers",
        "@org_tensorflow//tensorflow/c:tf_status_helper",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "checkpoint_reader_test",
    srcs = ["checkpoint_reader_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":checkpoint_reader",
        "//fcp/aggregation/testing",
        "//fcp/base",
        "//fcp/testing",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "checkpoint_writer",
    srcs = ["checkpoint_writer.cc"],
    hdrs = ["checkpoint_writer.h"],
    copts = FCP_COPTS,
    deps = [
        "//fcp/aggregation/core:tensor",
        "//fcp/base",
        "//fcp/tensorflow:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@org_tensorflow//tensorflow/core:framework",
    ],
)

cc_test(
    name = "checkpoint_writer_test",
    srcs = ["checkpoint_writer_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":checkpoint_reader",
        ":checkpoint_writer",
        "//fcp/aggregation/core:tensor",
        "//fcp/aggregation/testing",
        "//fcp/aggregation/testing:test_data",
        "//fcp/testing",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tensorflow_checkpoint_builder_factory",
    srcs = ["tensorflow_checkpoint_builder_factory.cc"],
    hdrs = ["tensorflow_checkpoint_builder_factory.h"],
    copts = FCP_COPTS,
    deps = [
        ":checkpoint_writer",
        "//fcp/aggregation/core:tensor",
        "//fcp/aggregation/protocol/simple_aggregation",
        "//fcp/base",
        "//fcp/tensorflow:status",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@org_tensorflow//tensorflow/core/platform:env",
        "@org_tensorflow//tensorflow/core/platform:status",
    ],
)

cc_test(
    name = "tensorflow_checkpoint_builder_factory_test",
    srcs = ["tensorflow_checkpoint_builder_factory_test.cc"],
    deps = [
        ":tensorflow_checkpoint_builder_factory",
        "//fcp/aggregation/core:tensor",
        "//fcp/aggregation/testing",
        "//fcp/aggregation/testing:test_data",
        "//fcp/base",
        "//fcp/testing",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tensorflow_checkpoint_parser_factory",
    srcs = ["tensorflow_checkpoint_parser_factory.cc"],
    hdrs = ["tensorflow_checkpoint_parser_factory.h"],
    copts = FCP_COPTS,
    deps = [
        ":checkpoint_reader",
        "//fcp/aggregation/core:tensor",
        "//fcp/aggregation/protocol/simple_aggregation",
        "//fcp/base",
        "//fcp/tensorflow:status",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@org_tensorflow//tensorflow/core/platform:env",
    ],
)

cc_test(
    name = "tensorflow_checkpoint_parser_factory_test",
    srcs = ["tensorflow_checkpoint_parser_factory_test.cc"],
    deps = [
        ":tensorflow_checkpoint_parser_factory",
        "//fcp/aggregation/protocol/simple_aggregation",
        "//fcp/aggregation/testing",
        "//fcp/base",
        "//fcp/tensorflow:status",
        "//fcp/testing",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)
