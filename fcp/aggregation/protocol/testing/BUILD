load("//fcp:config.bzl", "FCP_COPTS")

package(
    default_visibility = ["//fcp:internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "mock_callback",
    testonly = 1,
    hdrs = [
        "test_callback.h",
    ],
    copts = FCP_COPTS,
    deps = [
        "//fcp/aggregation/protocol:aggregation_protocol",
        "//fcp/aggregation/protocol:cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)
