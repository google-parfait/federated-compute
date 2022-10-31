load("@rules_python//python:defs.bzl", "py_library", "py_test")

default_visibility = ["//fcp:internal"]

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
