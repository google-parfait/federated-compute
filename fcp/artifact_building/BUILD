load("@rules_python//python:defs.bzl", "py_library", "py_test")

default_visibility = ["//fcp:internal"]

py_library(
    name = "checkpoint_utils",
    srcs = ["checkpoint_utils.py"],
    srcs_version = "PY3",
    visibility = default_visibility + [
    ],
    deps = [
        ":tensor_utils",
        ":type_checks",
    ],
)

py_test(
    name = "checkpoint_utils_test",
    srcs = [
        "checkpoint_utils_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":checkpoint_utils",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_library(
    name = "data_spec",
    srcs = ["data_spec.py"],
    srcs_version = "PY3",
    visibility = default_visibility + [
    ],
    deps = [
        ":type_checks",
        "//fcp/protos:plan_py_pb2",
    ],
)

py_test(
    name = "data_spec_test",
    srcs = [
        "data_spec_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":data_spec",
        "//fcp/protos:plan_py_pb2",
    ],
)

py_library(
    name = "tensor_utils",
    srcs = ["tensor_utils.py"],
    srcs_version = "PY3",
    visibility = default_visibility + [
    ],
)

py_test(
    name = "tensor_utils_test",
    srcs = [
        "tensor_utils_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tensor_utils",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_library(
    name = "type_checks",
    srcs = ["type_checks.py"],
    srcs_version = "PY3",
    visibility = default_visibility + [
    ],
)

py_test(
    name = "type_checks_test",
    srcs = [
        "type_checks_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":type_checks"],
)
