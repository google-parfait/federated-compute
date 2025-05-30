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

load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_visibility = ["//fcp:internal"],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "constants",
    srcs = ["constants.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "min_sep_data_source",
    srcs = ["min_sep_data_source.py"],
    deps = [
        ":constants",
        "//fcp/protos/confidentialcompute:file_info_py_pb2",
        "@com_google_protobuf//:protobuf_python",
        "@federated_language//federated_language",
        "@federated_language//federated_language/proto:computation_py_pb2",
        "@federated_language//federated_language/proto:data_type_py_pb2",
    ],
)

py_test(
    name = "min_sep_data_source_test",
    srcs = ["min_sep_data_source_test.py"],
    deps = [
        ":min_sep_data_source",
        "//fcp/protos/confidentialcompute:file_info_py_pb2",
        "@federated_language//federated_language/proto:computation_py_pb2",
        "@pypi_absl_py//:pkg",
    ],
)
