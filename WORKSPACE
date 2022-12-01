# Copyright 2021 Google LLC
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

# This file uses repository rules to fetch external dependencies.
#
# We require that all uses of repository rules are effectively deterministic -
# they may fail, but otherwise must always produce exactly the same output for
# given parameters. For the http_archive rule, this means that you must specify
# a sha256 parameter; consequently, make sure to use stable URLs, rather than
# e.g. one that names a git branch.
#
# This is so that all(*) inputs to the build are fully determined (i.e. a
# change to build inputs requires a change to our sources), which avoids
# confusing outcomes from caching. If it is ever productive to clear your Bazel
# cache, that's a bug.
#
# (*) A Bazel build depends on local compiler toolchains (and Bazel itself), so
# it can be useful to pick a particular container image too (like some version
# of http://l.gcr.io/bazel).
#
# The repository namespace
# ------------------------
#
# Bazel's handling of @repository_names// is very broken. There is a single,
# global namespace for repositories. Conflicts are silently ignored. This is
# problematic for common dependencies. As much as possible, we use TensorFlow's
# dependencies. Things become especially difficult if we try to use our own
# version of grpc or protobuf. Current overrides:
#
#  - @com_github_gflags_gflags: tf uses 2.2.1 and we need 2.2.2 apparently.
#  - @com_google_googletest: Need to patch in support for absl flags

workspace(name = "com_google_fcp")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Needed for user-defined configs
http_archive(
    name = "bazel_skylib",
    sha256 = "f7be3474d42aae265405a592bb7da8e171919d74c16f082a5457840f06054728",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.2.1/bazel-skylib-1.2.1.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.2.1/bazel-skylib-1.2.1.tar.gz",
    ],
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

# GoogleTest/GoogleMock framework. Used by most unit-tests.
http_archive(
    name = "com_google_googletest",
    patches = ["//fcp/patches:googletest.patch"],
    sha256 = "fcfac631041fce253eba4fc014c28fd620e33e3758f64f8ed5487cc3e1840e3d",
    strip_prefix = "googletest-5a509dbd2e5a6c694116e329c5a20dc190653724",
    urls = ["https://github.com/google/googletest/archive/5a509dbd2e5a6c694116e329c5a20dc190653724.zip"],
)

http_archive(
    name = "com_github_gflags_gflags",
    sha256 = "34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
    strip_prefix = "gflags-2.2.2",
    urls = ["https://github.com/gflags/gflags/archive/v2.2.2.tar.gz"],
)

http_archive(
    name = "com_github_google_glog",
    sha256 = "21bc744fb7f2fa701ee8db339ded7dce4f975d0d55837a97be7d46e8382dea5a",
    strip_prefix = "glog-0.5.0",
    urls = ["https://github.com/google/glog/archive/v0.5.0.zip"],
)

http_archive(
    name = "com_github_grpc_grpc",
    sha256 = "76900ab068da86378395a8e125b5cc43dfae671e09ff6462ddfef18676e2165a",
    strip_prefix = "grpc-1.50.0",
    urls = ["https://github.com/grpc/grpc/archive/refs/tags/v1.50.0.tar.gz"],
)

# The version provided by TensorFlow 2.10 is incompatible with recent
# pybind11_abseil versions. Remove when updating to TensorFlow 2.11.
http_archive(
    name = "pybind11",
    build_file = "@org_tensorflow//third_party:pybind11.BUILD",
    sha256 = "eacf582fa8f696227988d08cfc46121770823839fe9e301a20fbce67e7cd70ec",
    strip_prefix = "pybind11-2.10.0",
    urls = ["https://github.com/pybind/pybind11/archive/refs/tags/v2.10.0.tar.gz"],
)

# The version provided by TensorFlow 2.10 doesn't support absl::Cord.
http_archive(
    name = "pybind11_abseil",
    sha256 = "6481888831cd548858c09371ea892329b36c8d4d961f559876c64e009d0bc630",
    strip_prefix = "pybind11_abseil-3922b3861a2b27d4111e3ac971e6697ea030a36e",
    url = "https://github.com/pybind/pybind11_abseil/archive/3922b3861a2b27d4111e3ac971e6697ea030a36e.tar.gz",
)

http_archive(
    name = "pybind11_protobuf",
    sha256 = "fe2b8bf12a65997b853709a5e719f7561b2e86a4cdbb9d8b051e654dd0fd8d11",
    strip_prefix = "pybind11_protobuf-a50899c2eb604fc5f25deeb8901eff6231b8b3c0",
    url = "https://github.com/pybind/pybind11_protobuf/archive/a50899c2eb604fc5f25deeb8901eff6231b8b3c0.tar.gz",
)

# Define the @io_grpc_grpc_java repository, which is used by the
# @com_google_googleapis repository to define the Java protobuf targets such as
# @com_google_googleapis//google/rpc:rpc_java_proto). The pattern we use here is
# the same as @com_google_googleapis' WORKSPACE file uses.
#
# Note that the @com_google_googleapis repository is actually defined
# transitively by the @org_tensorflow workspace rules.
http_archive(
    name = "com_google_api_gax_java",
    sha256 = "ab63851160a00393b3f3c7b9fc0259a3fc91806a8db7f8c5edb7ff383b428a58",
    strip_prefix = "gax-java-2.11.0",
    urls = ["https://github.com/googleapis/gax-java/archive/v2.11.0.zip"],
)

load("@com_google_api_gax_java//:repository_rules.bzl", "com_google_api_gax_java_properties")

com_google_api_gax_java_properties(
    name = "com_google_api_gax_java_properties",
    file = "@com_google_api_gax_java//:dependencies.properties",
)

load("@com_google_api_gax_java//:repositories.bzl", "com_google_api_gax_java_repositories")

com_google_api_gax_java_repositories()

# Tensorflow v2.10.0
http_archive(
    name = "org_tensorflow",
    patch_tool = "patch",
    patches = [
        # This patch removes an import generated by the old version of
        # googletest that TensorFlow uses, and isn't generated by the newer
        # version of googletest that we use.
        "//fcp/patches:tensorflow_googletest.patch",
        # This patch enables googleapi Java and Python proto rules such as
        # @com_google_googleapis//google/rpc:rpc_java_proto.
        "//fcp/patches:tensorflow_googleapis_proto_rules.patch",
        # This patch works around failures in GitHub infrastructure to
        # download versions of LLVM pointed to by non-HEAD TensorFlow.
        # TODO(team): Remove this patch when resolved.
        "//fcp/patches:tensorflow_llvm_url.patch",
        # TensorFlow's custom pybind11 BUILD file is missing the osx config
        # setting expected by pybind11_bazel.
        "//fcp/patches:tensorflow_pybind11_osx.patch",
        # This patch removes tf_custom_op_py_library's dependency on the Bazel
        # version of TensorFlow since for all of our Python code, we rely on a
        # system-provided TensorFlow.
        "//fcp/patches:tensorflow_tf_custom_op_py_library.patch",
        # gRPC v1.48.0-pre1 and later include zconf.h in addition to zlib.h;
        # TensorFlow's build rule for zlib only exports the latter.
        "//fcp/patches:tensorflow_zlib.patch",
    ],
    sha256 = "b5a1bb04c84b6fe1538377e5a1f649bb5d5f0b2e3625a3c526ff3a8af88633e8",
    strip_prefix = "tensorflow-2.10.0",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/v2.10.0.tar.gz",
    ],
)

# The following is copied from TensorFlow's own WORKSPACE, see
# https://github.com/tensorflow/tensorflow/blob/v2.8.0/WORKSPACE#L9

load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()

load("//fcp/tensorflow/system_provided_tf:system_provided_tf.bzl", "system_provided_tf")

system_provided_tf(
    name = "system_provided_tf",
    # TensorFlow's pip package was built with libstdc++, not libc++.
    extra_cxxopts = ["-stdlib=libstdc++"],
    extra_linkopts = ["-stdlib=libstdc++"],
)

http_archive(
    name = "com_google_benchmark",
    sha256 = "23082937d1663a53b90cb5b61df4bcc312f6dee7018da78ba00dd6bd669dfef2",
    strip_prefix = "benchmark-1.5.1",
    urls = [
        "https://github.com/google/benchmark/archive/v1.5.1.tar.gz",
    ],
)

# Cpp ProtoDataStore
http_archive(
    name = "protodatastore_cpp",
    sha256 = "d2627231fce0c9944100812b42d33203f7e03e78215d10123e78041b491005c3",
    strip_prefix = "protodatastore-cpp-0cd8b124bc65bcac105bce4a706923218ae5d625",
    url = "https://github.com/google/protodatastore-cpp/archive/0cd8b124bc65bcac105bce4a706923218ae5d625.zip",
)

# libcurl is used in the http client.
http_archive(
    name = "libcurl",
    build_file = "//third_party:curl.BUILD.bzl",
    sha256 = "e9d74b8586e0d2e6b45dc948bbe77525a1fa7f7c004ad5192f12e72c365e376e",
    strip_prefix = "curl-curl-7_84_0",
    url = "https://github.com/curl/curl/archive/refs/tags/curl-7_84_0.zip",
)

# We use only the http test server.
http_archive(
    name = "tensorflow_serving",
    patch_tool = "patch",
    patches = [
        "//fcp/patches:tensorflow_serving.patch",
    ],
    sha256 = "6b428100be7ec4bb34fc7910b7f549b6556854016a3c4cbda5613b5c114797b3",
    strip_prefix = "serving-2.9.0",
    url = "https://github.com/tensorflow/serving/archive/refs/tags/2.9.0.zip",
)

load("@tensorflow_serving//tensorflow_serving:workspace.bzl", "tf_serving_workspace")

tf_serving_workspace()

# Java Maven-based repositories.
http_archive(
    name = "rules_jvm_external",
    sha256 = "cd1a77b7b02e8e008439ca76fd34f5b07aecb8c752961f9640dea15e9e5ba1ca",
    strip_prefix = "rules_jvm_external-4.2",
    url = "https://github.com/bazelbuild/rules_jvm_external/archive/4.2.zip",
)

load("@rules_jvm_external//:repositories.bzl", "rules_jvm_external_deps")

rules_jvm_external_deps()

load("@rules_jvm_external//:setup.bzl", "rules_jvm_external_setup")

rules_jvm_external_setup()

load("@rules_jvm_external//:defs.bzl", "maven_install")

maven_install(
    artifacts = [
        "com.google.code.findbugs:jsr305:3.0.2",
        "com.google.errorprone:error_prone_annotations:2.11.0",
        "com.google.guava:guava:31.0.1-jre",
        "com.google.truth:truth:1.1.3",
        "junit:junit:4.13",
        "org.mockito:mockito-core:4.3.1",
    ],
    repositories = [
        "https://maven.google.com",
        "https://repo1.maven.org/maven2",
    ],
)

# The version of googleapis imported by TensorFlow doesn't provide
# `py_proto_library` targets for //google/longrunning.
http_archive(
    name = "googleapis_for_longrunning",
    patches = [
        "//fcp/patches:googleapis_longrunning.patch",
    ],
    sha256 = "c1db0b022cdfc5b5ce5f05b0f00568e2d927c9890429ec9c35bda12f52d93065",
    strip_prefix = "googleapis-2d8030c4102f97bc6be4ddab74c7cbfe88d8c016",
    url = "https://github.com/googleapis/googleapis/archive/2d8030c4102f97bc6be4ddab74c7cbfe88d8c016.tar.gz",
)
