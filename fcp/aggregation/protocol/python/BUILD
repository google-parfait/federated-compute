load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension")

package(
    default_visibility = ["//fcp/aggregation:internal"],
    licenses = ["notice"],  # Apache 2.0
)

pybind_extension(
    name = "aggregation_protocol",
    srcs = ["aggregation_protocol.cc"],
    pytype_deps = [],
    deps = [
        "//fcp/aggregation/protocol:aggregation_protocol",
        "//fcp/aggregation/protocol:cc_proto",
        "//fcp/aggregation/protocol:configuration_cc_proto",
        "@com_google_absl//absl/status",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)
