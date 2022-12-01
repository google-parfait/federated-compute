load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_test")

package(
    default_visibility = ["//fcp/aggregation:internal"],
    licenses = ["notice"],  # Apache 2.0
)

pybind_extension(
    name = "aggregation_protocols",
    srcs = ["aggregation_protocols.cc"],
    pytype_deps = [
        "//fcp/aggregation/protocol/python:aggregation_protocol",
    ],
    deps = [
        "//fcp/aggregation/protocol:aggregation_protocol",
        "//fcp/aggregation/protocol:configuration_cc_proto",
        "//fcp/aggregation/protocol/simple_aggregation",
        "//fcp/aggregation/tensorflow:tensorflow_checkpoint_builder_factory",
        "//fcp/aggregation/tensorflow:tensorflow_checkpoint_parser_factory",
        "@com_google_absl//absl/status:statusor",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)

py_test(
    name = "aggregation_protocols_test",
    srcs = ["aggregation_protocols_test.py"],
    data = ["@pybind11_abseil//pybind11_abseil:status.so"],
    deps = [
        ":aggregation_protocols",
        "//fcp/aggregation/protocol:configuration_py_pb2",
        "//fcp/aggregation/protocol:py_pb2",
        "//fcp/protos:plan_py_pb2",
    ],
)
