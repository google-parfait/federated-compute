# Copyright 2023 Google LLC
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
package(
    default_visibility = ["//fcp:internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "client_payload",
    srcs = ["client_payload.cc"],
    hdrs = ["client_payload.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//fcp/protos/confidentialcompute:aggregation_client_payload_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "client_payload_test",
    srcs = ["client_payload_test.cc"],
    deps = [
        ":client_payload",
        "//fcp/testing",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "cose",
    srcs = ["cose.cc"],
    hdrs = ["cose.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//fcp/base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_protobuf//:protobuf",
        "@libcppbor",
    ],
)

cc_test(
    name = "cose_test",
    srcs = ["cose_test.cc"],
    deps = [
        ":cose",
        "//fcp/testing",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "crypto",
    srcs = ["crypto.cc"],
    hdrs = ["crypto.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cose",
        "//fcp/base",
        "//fcp/base:digest",
        "//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@boringssl//:crypto",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "crypto_test",
    srcs = ["crypto_test.cc"],
    deps = [
        ":cose",
        ":crypto",
        "//fcp/base",
        "//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "//fcp/testing",
        "@boringssl//:crypto",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "data_helper",
    srcs = ["data_helper.cc"],
    hdrs = ["data_helper.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//fcp/base",
        "//fcp/protos/confidentialcompute:file_info_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:cardinalities",
        "@org_tensorflow_federated//tensorflow_federated/proto/v0:computation_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/proto/v0:executor_cc_proto",
    ],
)

cc_test(
    name = "data_helper_test",
    srcs = ["data_helper_test.cc"],
    deps = [
        ":data_helper",
        "//fcp/protos/confidentialcompute:file_info_cc_proto",
        "//fcp/testing",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow_federated//tensorflow_federated/proto/v0:array_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/proto/v0:computation_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/proto/v0:executor_cc_proto",
    ],
)
