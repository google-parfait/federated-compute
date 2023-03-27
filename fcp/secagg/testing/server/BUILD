load("//fcp:config.bzl", "FCP_COPTS")

package(
    default_visibility = ["//fcp:internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "server_mocks",
    testonly = 1,
    hdrs = [
        "mock_secagg_server_metrics_listener.h",
        "mock_send_to_clients_interface.h",
    ],
    copts = FCP_COPTS,
    deps = [
        "//fcp/secagg/server:secagg_server_metrics_listener",
        "//fcp/secagg/server:send_to_clients_interface",
        "//fcp/secagg/server:server_cc_proto",
        "//fcp/secagg/shared:cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "experiments",
    testonly = 1,
    hdrs = [
        "test_secagg_experiments.h",
    ],
    deps = [
        "//fcp/secagg/server:experiments_interface",
    ],
)

cc_library(
    name = "async_runner",
    testonly = 1,
    hdrs = [
        "test_async_runner.h",
    ],
    deps = [
        "//fcp/base:scheduler",
        "//fcp/secagg/server:secagg_scheduler",
    ],
)
