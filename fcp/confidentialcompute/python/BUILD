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
    default_applicable_licenses = ["//fcp:license"],
    licenses = ["notice"],
)

py_library(
    name = "constants",
    srcs = ["constants.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "dp_mf_aggregator",
    srcs = ["dp_mf_aggregator.py"],
    tags = ["manual"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/optax",
        "@federated_language//federated_language",
        "@org_tensorflow_federated//tensorflow_federated",
        "@pypi//jax",
        "@pypi//jax_privacy",
        "@pypi//tensorflow",
    ],
)

py_library(
    name = "external_service_handle",
    srcs = ["external_service_handle.py"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "dp_mf_aggregator_test",
    srcs = ["dp_mf_aggregator_test.py"],
    tags = ["manual"],
    deps = [
        ":dp_mf_aggregator",
        "//third_party/py/optax",
        "@federated_language//federated_language",
        "@pypi//absl_py",
        "@pypi//numpy",
        "@pypi//tensorflow",
    ],
)

py_library(
    name = "min_sep_data_source",
    srcs = ["min_sep_data_source.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":constants",
        ":program_input_provider",
        "//fcp/protos/confidentialcompute:file_info_py_pb2",
        "@com_google_protobuf//:protobuf_python",
        "@federated_language//federated_language",
        "@federated_language//federated_language/proto:array_py_pb2",
        "@federated_language//federated_language/proto:computation_py_pb2",
        "@federated_language//federated_language/proto:data_type_py_pb2",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor_py_pb2",
        "@pypi//absl_py",
    ],
)

py_test(
    name = "min_sep_data_source_test",
    srcs = ["min_sep_data_source_test.py"],
    deps = [
        ":min_sep_data_source",
        ":program_input_provider",
        "//fcp/protos/confidentialcompute:file_info_py_pb2",
        "@federated_language//federated_language/proto:array_py_pb2",
        "@federated_language//federated_language/proto:computation_py_pb2",
        "@federated_language//federated_language/proto:data_type_py_pb2",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor_py_pb2",
        "@pypi//absl_py",
    ],
)

py_library(
    name = "program_input_provider",
    srcs = ["program_input_provider.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor_py_pb2",
    ],
)

py_test(
    name = "program_input_provider_test",
    srcs = ["program_input_provider_test.py"],
    deps = [
        ":program_input_provider",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor_py_pb2",
        "@pypi//absl_py",
    ],
)

py_library(
    name = "accounting",
    srcs = ["accounting.py"],
    deps = ["@pypi//numpy"],
)

py_test(
    name = "accounting_test",
    srcs = ["accounting_test.py"],
    deps = [
        ":accounting",
        "@pypi//absl_py",
    ],
)
