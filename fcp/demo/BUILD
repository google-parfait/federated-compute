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

load("@rules_python//python:defs.bzl", "py_library", "py_test")

# The public interface for this package, providing the various Federated Program platform
# components needed to run a Federated Program using Federated Compute clients.
py_library(
    name = "demo",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":federated_computation",
        ":federated_context",
        ":federated_data_source",
    ],
)

py_library(
    name = "aggregations",
    srcs = ["aggregations.py"],
    data = ["@pybind11_abseil//pybind11_abseil:status.so"],
    srcs_version = "PY3",
    deps = [
        ":http_actions",
        ":media",
        "//fcp/aggregation/protocol:configuration_py_pb2",
        "//fcp/aggregation/protocol:py_pb2",
        "//fcp/aggregation/protocol/python:aggregation_protocol",
        "//fcp/aggregation/tensorflow/python:aggregation_protocols",
        "//fcp/protos:plan_py_pb2",
        "//fcp/protos/federatedcompute:federated_compute_py_pb2",
        "@com_google_googleapis//google/rpc:code_py_proto",
        "@googleapis_for_longrunning//google/longrunning:longrunning_py_proto",
    ],
)

py_test(
    name = "aggregations_test",
    srcs = ["aggregations_test.py"],
    data = ["@pybind11_abseil//pybind11_abseil:status.so"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":aggregations",
        ":http_actions",
        ":media",
        ":test_utils",
        "//fcp/aggregation/protocol:py_pb2",
        "//fcp/aggregation/protocol/python:aggregation_protocol",
        "//fcp/aggregation/tensorflow/python:aggregation_protocols",
        "//fcp/protos:plan_py_pb2",
        "//fcp/protos/federatedcompute:federated_compute_py_pb2",
    ],
)

py_library(
    name = "checkpoint_tensor_reference",
    srcs = ["checkpoint_tensor_reference.py"],
    srcs_version = "PY3",
)

py_test(
    name = "checkpoint_tensor_reference_test",
    srcs = ["checkpoint_tensor_reference_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":checkpoint_tensor_reference",
        ":test_utils",
    ],
)

py_library(
    name = "eligibility_eval_tasks",
    srcs = ["eligibility_eval_tasks.py"],
    srcs_version = "PY3",
    deps = [
        ":http_actions",
        "//fcp/protos/federatedcompute:federated_compute_py_pb2",
        "@com_google_googleapis//google/rpc:code_py_proto",
    ],
)

py_test(
    name = "eligibility_eval_tasks_test",
    srcs = ["eligibility_eval_tasks_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":eligibility_eval_tasks",
        ":http_actions",
        "//fcp/protos/federatedcompute:federated_compute_py_pb2",
        "@com_google_googleapis//google/rpc:code_py_proto",
    ],
)

py_library(
    name = "federated_computation",
    srcs = ["federated_computation.py"],
    srcs_version = "PY3",
)

py_test(
    name = "federated_computation_test",
    srcs = ["federated_computation_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":federated_computation"],
)

py_library(
    name = "federated_context",
    srcs = ["federated_context.py"],
    srcs_version = "PY3",
    deps = [
        ":checkpoint_tensor_reference",
        ":federated_computation",
        ":federated_data_source",
        ":server",
        "//fcp/artifact_building:artifact_constants",
        "//fcp/artifact_building:checkpoint_utils",
        "//fcp/artifact_building:data_spec",
        "//fcp/artifact_building:federated_compute_plan_builder",
        "//fcp/artifact_building:plan_utils",
        "//fcp/artifact_building:variable_helpers",
        "//fcp/protos:plan_py_pb2",
    ],
)

py_test(
    name = "federated_context_test",
    size = "medium",
    srcs = ["federated_context_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":federated_computation",
        ":federated_context",
        ":federated_data_source",
        ":server",
        ":test_utils",
        "//fcp/artifact_building:artifact_constants",
        "//fcp/artifact_building:federated_compute_plan_builder",
        "//fcp/artifact_building:plan_utils",
        "//fcp/artifact_building:variable_helpers",
        "//fcp/protos:plan_py_pb2",
    ],
)

py_library(
    name = "federated_data_source",
    srcs = ["federated_data_source.py"],
    srcs_version = "PY3",
    deps = [
        "//fcp/protos:plan_py_pb2",
        "//fcp/protos/federatedcompute:federated_compute_py_pb2",
    ],
)

py_test(
    name = "federated_data_source_test",
    srcs = ["federated_data_source_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":federated_data_source",
        "//fcp/protos:plan_py_pb2",
        "//fcp/protos/federatedcompute:federated_compute_py_pb2",
    ],
)

py_test(
    name = "federated_program_test",
    size = "medium",
    srcs = ["federated_program_test.py"],
    data = ["//fcp/client:client_runner_main"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":demo",
        "//fcp/client:client_runner_example_data_py_pb2",
        "//fcp/protos:plan_py_pb2",
    ],
)

py_library(
    name = "http_actions",
    srcs = ["http_actions.py"],
    srcs_version = "PY3",
    deps = ["@com_google_googleapis//google/api:annotations_py_proto"],
)

py_test(
    name = "http_actions_test",
    size = "medium",
    srcs = ["http_actions_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":http_actions",
        "//fcp/protos/federatedcompute:federated_compute_py_pb2",
    ],
)

py_library(
    name = "media",
    srcs = ["media.py"],
    srcs_version = "PY3",
    deps = [
        ":http_actions",
        "//fcp/protos/federatedcompute:federated_compute_py_pb2",
    ],
)

py_test(
    name = "media_test",
    srcs = ["media_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":http_actions",
        ":media",
        "//fcp/protos/federatedcompute:federated_compute_py_pb2",
    ],
)

py_library(
    name = "plan_utils",
    srcs = ["plan_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//fcp/protos:plan_py_pb2",
        "//fcp/tensorflow:serve_slices_py",
    ],
)

py_test(
    name = "plan_utils_test",
    size = "medium",
    srcs = ["plan_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":plan_utils",
        ":test_utils",
        "//fcp/protos:plan_py_pb2",
        "//fcp/tensorflow:serve_slices_py",
    ],
)

py_library(
    name = "server",
    srcs = ["server.py"],
    srcs_version = "PY3",
    deps = [
        ":aggregations",
        ":eligibility_eval_tasks",
        ":http_actions",
        ":media",
        ":plan_utils",
        ":task_assignments",
        "//fcp/protos:plan_py_pb2",
        "//fcp/protos/federatedcompute:federated_compute_py_pb2",
    ],
)

py_test(
    name = "server_test",
    size = "medium",
    srcs = ["server_test.py"],
    data = ["//fcp/client:client_runner_main"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":plan_utils",
        ":server",
        ":test_utils",
        "//fcp/protos:plan_py_pb2",
        "//fcp/protos/federatedcompute:federated_compute_py_pb2",
        "//fcp/tensorflow:external_dataset_py",
        "@googleapis_for_longrunning//google/longrunning:longrunning_py_proto",
    ],
)

py_library(
    name = "task_assignments",
    srcs = ["task_assignments.py"],
    srcs_version = "PY3",
    deps = [
        ":aggregations",
        ":http_actions",
        "//fcp/protos/federatedcompute:federated_compute_py_pb2",
        "@com_google_googleapis//google/rpc:code_py_proto",
        "@googleapis_for_longrunning//google/longrunning:longrunning_py_proto",
    ],
)

py_test(
    name = "task_assignments_test",
    srcs = ["task_assignments_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":aggregations",
        ":http_actions",
        ":task_assignments",
        "//fcp/protos/federatedcompute:federated_compute_py_pb2",
        "@com_google_googleapis//google/rpc:code_py_proto",
    ],
)

py_library(
    name = "test_utils",
    testonly = True,
    srcs = ["test_utils.py"],
)

py_test(
    name = "test_utils_test",
    size = "medium",
    srcs = ["test_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":test_utils"],
)
