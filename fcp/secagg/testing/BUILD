# Description:
#   Mocks for SecAgg.

load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("//fcp:config.bzl", "FCP_COPTS")

package(
    default_visibility = ["//fcp:internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "common_mocks",
    testonly = 1,
    srcs = [
        "ecdh_pregenerated_test_keys.cc",
    ],
    hdrs = [
        "ecdh_pregenerated_test_keys.h",
        "fake_prng.h",
    ],
    deps = [
        "//fcp/secagg/shared",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "client_mocks",
    testonly = 1,
    hdrs = [
        "mock_send_to_server_interface.h",
        "mock_state_transition_listener.h",
    ],
    copts = FCP_COPTS,
    deps = [
        "//fcp/secagg/client",
        "//fcp/secagg/client:state_transition_listener",
        "//fcp/secagg/shared:cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "testing",
    testonly = 1,
    srcs = [
        "test_matchers.cc",
    ],
    hdrs = [
        "test_matchers.h",
    ],
    copts = FCP_COPTS,
    deps = [
        "//fcp/secagg/shared",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_googletest//:gtest_main",
    ],
)
