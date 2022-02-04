# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("//fcp:config.bzl", "FCP_COPTS")
load("//fcp/tracing:build_defs.bzl", "tracing_schema_cc_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_custom_op_library", "tf_gen_op_wrapper_py")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

default_visibility = ["//fcp:internal"]

package(
    default_visibility = default_visibility,
    licenses = ["notice"],  # Apache 2.0
)

# We want to build with -Werror. Here we disable some warnings arising from
# including TensorFlow headers. For simplicity, use this for all targets in
# this package.
FCP_TF_COPTS = FCP_COPTS + ["-Wno-error=thread-safety-precise"]

tf_cc_test(
    name = "tf_smoke_test",
    srcs = ["tf_smoke_test.cc"],
    extra_copts = FCP_TF_COPTS,
    deps = [
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/cc:client_session",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:tensorflow_opensource",
        "@org_tensorflow//tensorflow/core:testlib",
    ],
)

# Library for converting between the FCP and TensorFlow versions of a Status.
# Note that this library is intended to be usable in an op .so, thus it depends
# on TF headers but *not* an implementation (the final binary needs to link it
# in).
cc_library(
    name = "status",
    srcs = [
        "status.cc",
    ],
    hdrs = [
        "status.h",
    ],
    copts = FCP_TF_COPTS,
    deps = [
        "//fcp/base",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ],
)

cc_test(
    name = "status_test",
    srcs = [
        "status_test.cc",
    ],
    copts = FCP_TF_COPTS,
    deps = [
        ":status",
        "@com_google_googletest//:gtest_main",
        # See remarks on :status about the TF framework dependency
        "@org_tensorflow//tensorflow/core:framework",
    ],
)

cc_library(
    name = "host_object",
    srcs = [
        "host_object.cc",
    ],
    hdrs = [
        "host_object.h",
    ],
    copts = FCP_TF_COPTS,
    visibility = default_visibility + [
    ],
    deps = [
        "//fcp/base",
        "//fcp/base:random_token",
        "//fcp/base:unique_value",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "host_object_test",
    srcs = [
        "host_object_test.cc",
    ],
    copts = FCP_TF_COPTS,
    deps = [
        ":host_object",
        "@com_google_googletest//:gtest_main",
    ],
)

tracing_schema_cc_library(
    name = "tracing_schema",
    srcs = ["tracing_schema.fbs"],
)

cc_library(
    name = "tf_session",
    srcs = ["tf_session.cc"],
    hdrs = ["tf_session.h"],
    copts = FCP_TF_COPTS,
    deps = [
        ":status",
        ":tracing_schema",
        "//fcp/base",
        "//fcp/base:process_unique_id",
        "//fcp/base:result",
        "//fcp/protos:plan_cc_proto",
        "//fcp/tracing",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

cc_test(
    name = "tf_session_test",
    srcs = ["tf_session_test.cc"],
    copts = FCP_TF_COPTS,
    deps = [
        ":tf_session",
        ":tracing_schema",
        "//fcp/base:tracing_schema",
        "//fcp/protos:plan_cc_proto",
        "//fcp/tensorflow/testing:tf_helper",
        "//fcp/testing:result_matchers",
        "//fcp/tracing:test_tracing_recorder",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/cc:scope",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core:testlib",
    ],
)

# C++ interfaces for implementing an 'external dataset' (a kind of host object).
# Note this does *not* depend on TensorFlow.
cc_library(
    name = "external_dataset",
    srcs = [
    ],
    hdrs = [
        "external_dataset.h",
    ],
    copts = FCP_TF_COPTS,
    visibility = default_visibility + [
        # Needed by TFLite on-device training tests.
        "@org_tensorflow//tensorflow/lite/delegates/flex/training:__subpackages__",
    ],
    deps = [
        ":host_object",
        "//fcp/base:bounds",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

# The targets below produce a custom op, which involves a native library as
# well as Python wrappers. There is some significant complexity arising from
# the various ways that an op / kernel might be linked in, which we'll try to
# explain here:
#
#   - Ops / kernels are meant to be buildable as DSOs (dynamic shared objects,
#     i.e. .so files). Yet, all ops and kernels must agree on the same
#     'framework' for registration etc. When building a DSO, ops and kernels
#     can include TensorFlow framework *headers*, with implementations provided
#     by libtensorflow_framework.so at runtime.
#
#   - When using ops / kernels (and TensorFlow) from a standard Python
#     interpreter, they *must* be loaded as DSOs.
#
#   - When using ops / kernels (and TensorFlow) from C++, we have the option of
#     linking a monolithic binary, with Bazel's usual handling of deps. This is
#     in fact necessary to generate Python wrapper code (the generator links in
#     cc_library deps).
#
# Below, we generate *both* a DSO and cc_library variant of the ExternalDataset
# op and kernel:
#   cc_library: :external_dataset_op_lib
#   DSO: _external_dataset_op.so
#
# The ExternalDataset op is a peculiar case, since it is specifically intended
# to use objects provided by the program hosting TensorFlow (beyond the usual
# TensorFlow APIs). This is problematic, since separate host and DSO binaries
# each end up with their own definitions of symbols from common libraries (and
# likely export them!). Though this might appear to work sometimes, it must be
# avoided.
# See e.g. https://github.com/abseil/abseil-cpp/issues/125
#
#                    ---------------------------
#                    | _external_dataset_op.so |
#  -------------  -> |  absl                   |
#  | Host      | /   |  fcp/base               |
#  |  absl     |     ---------------------------
#  |  fcp/base | \           |
#  -------------  \          v
#                  \   ------------------------------
#                   -> | libtensorflow_framework.so |
#                      ------------------------------
#
# When using the cc_library version and Bazel's usual handling of the deps
# graph, this is of course not a problem.
#
# As such, the DSO version is specifically useful for *building graphs in
# Python*, whereas a (C++) host program should probably use the cc_library
# version.
#
# It's well-known that Bazel has a difficult time representing DSO dependencies. See
# https://docs.google.com/document/d/1-tv0_79zGyBoDmaP_pYWaBVUwHUteLpAs90_rUl-VY8/edit

EXTERNAL_DATASET_OP_SRCS = ["external_dataset_op.cc"]

EXTERNAL_DATASET_OP_DEPS = [
    ":external_dataset",
    ":status",
    "@com_google_absl//absl/strings:str_format",
    "//fcp/base:random_token",
]

# Public: TensorFlow op and op-kernel, that delegates to an ExternalDatasetStub
# host object. This is the cc_library version. See explanation above.
cc_library(
    name = "external_dataset_op_lib",
    srcs = EXTERNAL_DATASET_OP_SRCS,
    copts = FCP_TF_COPTS,
    visibility = ["//visibility:public"],
    deps = EXTERNAL_DATASET_OP_DEPS + [
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    # Uses TensorFlow's registration macros
    alwayslink = 1,
)

# DSO version of :external_dataset_op_lib, intended to be loaded by Python
# wrappers. See explanation above.
tf_custom_op_library(
    name = "_external_dataset_op.so",
    srcs = EXTERNAL_DATASET_OP_SRCS,
    copts = FCP_TF_COPTS,
    deps = EXTERNAL_DATASET_OP_DEPS,
)

# Generates the basic op wrapper for use in Python. As this is a dataset op,
# it's not useful directly; see :external_dataset_py.
tf_gen_op_wrapper_py(
    name = "gen_external_dataset_py",
    out = "gen_external_dataset_py.py",
    deps = [
        ":external_dataset_op_lib",
    ],
)

# Selector proto used in test dataset stubs and example selector fuser op.
proto_library(
    name = "test_selector_proto",
    testonly = True,
    srcs = [
        "test_selector.proto",
    ],
)

cc_proto_library(
    name = "test_selector_cc_proto",
    testonly = True,
    deps = [":test_selector_proto"],
)

# Custom op to compute the CRC32 checksum of a tensor.
cc_library(
    name = "crc32_op_lib",
    srcs = [
        "crc32_op.cc",
        "tensor_crc32.cc",
    ],
    hdrs = ["tensor_crc32.h"],
    copts = FCP_TF_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    # Uses TensorFlow's registration macros
    alwayslink = 1,
)

# Generates the basic op wrapper for use in Python.
tf_gen_op_wrapper_py(
    name = "gen_crc32_py",
    out = "gen_crc32_py.py",
    deps = [
        ":crc32_op_lib",
    ],
)

# Custom op to add resumption token to example selector.
cc_library(
    name = "example_selector_fuser_op_lib",
    srcs = [
        "example_selector_fuser_op.cc",
    ],
    copts = FCP_TF_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        "//fcp/protos:plan_cc_proto",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    # Uses TensorFlow's registration macros
    alwayslink = 1,
)

# Generates the basic op wrapper for use in Python.
tf_gen_op_wrapper_py(
    name = "gen_example_selector_fuser_op",
    out = "gen_example_selector_fuser_op.py",
    visibility = ["//visibility:public"],
    deps = [
        ":example_selector_fuser_op_lib",
    ],
)

# C++ library to set and access callbacks for slice serving requests.
# Used by the `ServeSlices` custom op below.
cc_library(
    name = "serve_slices_registry",
    hdrs = [
        "serve_slices_registry.h",
    ],
    copts = FCP_TF_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":host_object",
    ],
)

cc_test(
    name = "serve_slices_registry_test",
    srcs = ["serve_slices_registry_test.cc"],
    deps = [
        ":host_object",
        ":serve_slices_registry",
        "//fcp/base:random_token",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:framework",
    ],
)

SERVE_SLICES_OP_SRCS = ["serve_slices_op.cc"]

SERVE_SLICES_OP_DEPS = [
    ":serve_slices_registry",
    "@com_google_absl//absl/strings",
    "@com_google_absl//absl/strings:str_format",
]

# Custom op to register slices to serve for a `federated_select`.
cc_library(
    name = "serve_slices_op_lib",
    srcs = SERVE_SLICES_OP_SRCS,
    copts = FCP_TF_COPTS,
    visibility = ["//visibility:public"],
    deps = SERVE_SLICES_OP_DEPS + [
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    # Uses TensorFlow's registration macros
    alwayslink = 1,
)

# DSO version of `:serve_slices_op_lib`, intended to be loaded by Python
# wrappers. See explanation above starting with "The targets below...".
tf_custom_op_library(
    name = "_serve_slices_op.so",
    srcs = SERVE_SLICES_OP_SRCS,
    copts = FCP_TF_COPTS,
    deps = SERVE_SLICES_OP_DEPS,
)

# Generates the basic op wrapper for use in Python.
# Don't use this directly: use `:serve_slices_py` to ensure that the
# appropriate shared libraries are loaded.
tf_gen_op_wrapper_py(
    name = "gen_serve_slices_py",
    out = "gen_serve_slices_py.py",
    deps = [
        ":serve_slices_op_lib",
    ],
)

# Public: Python library for ServeSlices.

TENSOR_NAME_OP_SRCS = ["tensor_name_op.cc"]

TENSOR_NAME_OP_DEPS = [
    "@com_google_absl//absl/strings:str_format",
]

# Custom op to get the name of a tensor in the final graph at runtime.
cc_library(
    name = "tensor_name_op_lib",
    srcs = TENSOR_NAME_OP_SRCS,
    copts = FCP_TF_COPTS,
    visibility = ["//visibility:public"],
    deps = TENSOR_NAME_OP_DEPS + [
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    # Uses TensorFlow's registration macros
    alwayslink = 1,
)

# DSO version of `:tensor_name_op_lib`, intended to be loaded by Python
# wrappers. See explanation above starting with "The targets below...".
tf_custom_op_library(
    name = "_tensor_name_op.so",
    srcs = TENSOR_NAME_OP_SRCS,
    copts = FCP_TF_COPTS,
    deps = TENSOR_NAME_OP_DEPS,
)

# Generates the basic op wrapper for use in Python.
# Don't use this directly: use `:tensor_name_py` to ensure that the
# appropriate shared libraries are loaded.
tf_gen_op_wrapper_py(
    name = "gen_tensor_name_py",
    out = "gen_tensor_name_py.py",
    deps = [
        ":tensor_name_op_lib",
    ],
)

# Public: Python library for the `TensorName` op.

cc_library(
    name = "task_eligibility_info_ops_lib",
    srcs = ["task_eligibility_info_ops.cc"],
    copts = FCP_TF_COPTS,
    deps = [
        "//fcp/protos:federated_api_cc_proto",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    # Uses TensorFlow's registration macros
    alwayslink = 1,
)

# Generates the basic op wrapper for use in Python. We don't expose this wrapper
# directly, and rather we create a more user-friendly wrapper below, which uses
# this auto-generated one.
tf_gen_op_wrapper_py(
    name = "gen_task_eligibility_info_ops_py",
    out = "gen_task_eligibility_info_ops.py",
    visibility = ["//visibility:private"],
    deps = [
        ":task_eligibility_info_ops_lib",
    ],
)
