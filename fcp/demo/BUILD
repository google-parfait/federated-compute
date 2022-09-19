load("@rules_python//python:defs.bzl", "py_library", "py_test")

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
