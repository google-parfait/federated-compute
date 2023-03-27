# Description:
#   AES SecAgg server protocol implementation.

load("//fcp:config.bzl", "FCP_COPTS")

package(
    default_visibility = ["//fcp:internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "aes",
    srcs = [
        "aes_secagg_server_protocol_impl.cc",
    ],
    hdrs = [
        "aes_secagg_server_protocol_impl.h",
    ],
    copts = FCP_COPTS,
    deps = [
        "//fcp/base",
        "//fcp/secagg/server:experiments_names",
        "//fcp/secagg/server:secagg_scheduler",
        "//fcp/secagg/server:secagg_server_protocol_impl",
        "//fcp/secagg/server:server_cc_proto",
        "//fcp/secagg/server:tracing_schema",
        "//fcp/secagg/shared",
        "//fcp/tracing",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/status",
    ],
)
