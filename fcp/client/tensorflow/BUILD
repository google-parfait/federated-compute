load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = [
        "//fcp:internal",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "tensorflow_runner",
    hdrs = ["tensorflow_runner.h"],
    deps = [
        "//fcp/client:example_iterator_query_recorder",
        "//fcp/client:example_query_result_cc_proto",
        "//fcp/client:interfaces",
        "//fcp/client:interruptible_runner",
        "//fcp/client:runner_common",
        "//fcp/client/engine:common",
        "//fcp/client/engine:example_iterator_factory",
        "//fcp/client/opstats:opstats_logger",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "tensorflow_runner_factory",
    srcs = ["tensorflow_runner_factory.cc"],
    hdrs = ["tensorflow_runner_factory.h"],
    deps = [
        ":tensorflow_runner",
        "//fcp/base:function_registry",
    ],
)
