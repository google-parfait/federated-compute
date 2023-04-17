# pybind11 bindings for //fcp/tensorflow.

load("@rules_python//python:defs.bzl", "py_test")

package(
    default_visibility = ["//fcp:internal"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["serve_slices_registry.cc"])

py_test(
    name = "serve_slices_registry_test",
    srcs = ["serve_slices_registry_test.py"],
    deps = ["//fcp/tensorflow:serve_slices_py"],
)
