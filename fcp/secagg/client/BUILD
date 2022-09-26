# Description:
#   SecAgg client-specific components.

load("//fcp:config.bzl", "FCP_COPTS")

package(
    default_visibility = ["//fcp:internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "state_transition_listener",
    hdrs = ["state_transition_listener_interface.h"],
)

cc_library(
    name = "client",
    srcs = [
        "secagg_client.cc",
        "secagg_client_aborted_state.cc",
        "secagg_client_alive_base_state.cc",
        "secagg_client_completed_state.cc",
        "secagg_client_r0_advertise_keys_input_not_set_state.cc",
        "secagg_client_r0_advertise_keys_input_set_state.cc",
        "secagg_client_r1_share_keys_base_state.cc",
        "secagg_client_r1_share_keys_input_not_set_state.cc",
        "secagg_client_r1_share_keys_input_set_state.cc",
        "secagg_client_r2_masked_input_coll_base_state.cc",
        "secagg_client_r2_masked_input_coll_input_not_set_state.cc",
        "secagg_client_r2_masked_input_coll_input_set_state.cc",
        "secagg_client_r2_masked_input_coll_waiting_for_input_state.cc",
        "secagg_client_r3_unmasking_state.cc",
        "secagg_client_state.cc",
    ],
    hdrs = [
        "other_client_state.h",
        "secagg_client.h",
        "secagg_client_aborted_state.h",
        "secagg_client_alive_base_state.h",
        "secagg_client_completed_state.h",
        "secagg_client_r0_advertise_keys_input_not_set_state.h",
        "secagg_client_r0_advertise_keys_input_set_state.h",
        "secagg_client_r1_share_keys_base_state.h",
        "secagg_client_r1_share_keys_input_not_set_state.h",
        "secagg_client_r1_share_keys_input_set_state.h",
        "secagg_client_r2_masked_input_coll_base_state.h",
        "secagg_client_r2_masked_input_coll_input_not_set_state.h",
        "secagg_client_r2_masked_input_coll_input_set_state.h",
        "secagg_client_r2_masked_input_coll_waiting_for_input_state.h",
        "secagg_client_r3_unmasking_state.h",
        "secagg_client_state.h",
        "send_to_server_interface.h",
    ],
    copts = FCP_COPTS,
    deps = [
        ":state_transition_listener",
        "//fcp/base",
        "//fcp/secagg/shared",
        "//fcp/secagg/shared:cc_proto",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "client-test",
    size = "small",
    srcs = [
        "secagg_client_test.cc",
    ],
    copts = FCP_COPTS,
    deps = [
        ":client",
        ":state_transition_listener",
        "//fcp/base",
        "//fcp/secagg/shared",
        "//fcp/secagg/shared:cc_proto",
        "//fcp/secagg/testing:client_mocks",
        "//fcp/secagg/testing:common_mocks",
        "//fcp/testing",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "state-test",
    size = "small",
    srcs = [
        "secagg_client_aborted_state_test.cc",
        "secagg_client_completed_state_test.cc",
        "secagg_client_r0_advertise_keys_input_not_set_state_test.cc",
        "secagg_client_r0_advertise_keys_input_set_state_test.cc",
        "secagg_client_r1_share_keys_input_not_set_state_test.cc",
        "secagg_client_r1_share_keys_input_set_state_test.cc",
        "secagg_client_r2_masked_input_coll_input_not_set_state_test.cc",
        "secagg_client_r2_masked_input_coll_input_set_state_test.cc",
        "secagg_client_r2_masked_input_coll_waiting_for_input_state_test.cc",
        "secagg_client_r3_unmasking_state_test.cc",
    ],
    copts = FCP_COPTS,
    deps = [
        ":client",
        ":state_transition_listener",
        "//fcp/base",
        "//fcp/secagg/shared",
        "//fcp/secagg/shared:cc_proto",
        "//fcp/secagg/testing:client_mocks",
        "//fcp/secagg/testing:common_mocks",
        "//fcp/testing",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_googletest//:gtest_main",
    ],
)
