load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

py_library(
    name = "aggregations",
    srcs = ["aggregations.py"],
    srcs_version = "PY3",
    deps = [
        ":http_actions",
        ":media",
        ":plan_utils",
        "//fcp/protos:plan_py_pb2",
        "//fcp/protos/federatedcompute:federated_compute_py_pb2",
        "@googleapis_for_longrunning//google/longrunning:longrunning_py_proto",
    ],
)

py_test(
    name = "aggregations_test",
    srcs = ["aggregations_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":aggregations",
        ":http_actions",
        ":media",
        ":plan_utils",
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
    name = "federated_data_source",
    srcs = ["federated_data_source.py"],
    srcs_version = "PY3",
    deps = ["//fcp/protos:plan_py_pb2"],
)

py_test(
    name = "federated_data_source_test",
    srcs = ["federated_data_source_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":federated_data_source",
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
    deps = ["//fcp/protos:plan_py_pb2"],
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
        ":server",
        ":test_utils",
        "//fcp/protos:plan_py_pb2",
        "//fcp/protos/federatedcompute:federated_compute_py_pb2",
        "//fcp/tensorflow:external_dataset_py",
        "@googleapis_for_longrunning//google/longrunning:longrunning_py_proto",
    ],
)

py_binary(
    name = "server_runner",
    srcs = ["server_runner.py"],
    deps = [
        ":server",
        "//fcp/protos:plan_py_pb2",
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
