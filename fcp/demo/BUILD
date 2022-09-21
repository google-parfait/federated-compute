load("@rules_python//python:defs.bzl", "py_library", "py_test")

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
        "@com_google_absl_py//absl/logging",
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
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "eligibility_eval_tasks",
    srcs = ["eligibility_eval_tasks.py"],
    srcs_version = "PY3",
    deps = [
        ":http_actions",
        "//fcp/protos/federatedcompute:federated_compute_py_pb2",
        "@com_google_absl_py//absl/logging",
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
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_googleapis//google/rpc:code_py_proto",
    ],
)

py_library(
    name = "http_actions",
    srcs = ["http_actions.py"],
    srcs_version = "PY3",
    deps = [
        "@com_google_absl_py//absl/logging",
        "@com_google_googleapis//google/api:annotations_py_proto",
        "@com_google_protobuf//:protobuf_python",
    ],
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
        "@com_google_absl_py//absl/testing:absltest",
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
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "plan_utils",
    srcs = ["plan_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//fcp/protos:plan_py_pb2",
        "@com_google_protobuf//:protobuf_python",
        "@pip//pypi__tensorflow",
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
        "//fcp/protos:plan_py_pb2",
        "@com_google_absl_py//absl/testing:absltest",
        "@pip//pypi__tensorflow",
    ],
)
