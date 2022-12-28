load("@bazel_skylib//rules:common_settings.bzl", "bool_setting")

package(default_visibility = ["//visibility:public"])

# Config setting to use in select()'s to distinguish building for the
# system-provided TensorFlow package.
config_setting(
    name = "system_provided_tf_build",
    flag_values = {":system_provided_tf_build_setting": "True"},
)

# Non-configurable build setting to indicate building using the system-provided
# TensorFlow package.
bool_setting(
    name = "system_provided_tf_build_setting",
    build_setting_default = False,
    visibility = ["//visibility:private"],
)

# Internal config setting to distinguish clang from other compilers. This target
# should not be used directly.
config_setting(
    name = "clang_compiler",
    flag_values = {"@bazel_tools//tools/cpp:compiler": "clang"},
)

cc_library(
    name = "tf_headers",
    hdrs = glob(["%{HEADERS_DIR}/**"]),
    includes = ["%{HEADERS_DIR}"]
)

cc_library(
    name = "libtensorflow_framework",
    srcs = ["%{LIBRARY_FILE}"],
)
