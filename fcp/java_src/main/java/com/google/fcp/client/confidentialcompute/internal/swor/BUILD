# Copyright 2026 Google LLC
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

# Sampling without replacement (SWOR) for the default fetcher and uploader. See SworTracker.

load("@com_google_protobuf//bazel:java_lite_proto_library.bzl", "java_lite_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@rules_android//rules:rules.bzl", "android_library")

package(
    default_applicable_licenses = ["//fcp:license"],
    default_visibility = [
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute:__subpackages__",
        "//fcp/java_src/test/java/com/google/fcp/client:__subpackages__",
    ],
    licenses = ["notice"],
)

proto_library(
    name = "swor_state_proto",
    srcs = ["swor_state.proto"],
)

java_lite_proto_library(
    name = "swor_state_java_proto_lite",
    deps = [":swor_state_proto"],
)

android_library(
    name = "swor",
    srcs = [
        "SworStore.java",
        "SworTracker.java",
    ],
    deps = [
        ":swor_state_java_proto_lite",
        "//fcp/java_src/main/java/com/google/fcp/client:time_source",
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute:api",
        "//fcp/java_src/oss_build_defs:flogger_android",
        "@com_google_protobuf//:protobuf_javalite",
        "@fcp_android_maven//:androidx_annotation_annotation_jvm",
        "@fcp_android_maven//:androidx_datastore_datastore_core_android",
        "@fcp_android_maven//:androidx_datastore_datastore_guava",
        "@fcp_android_maven//:com_google_guava_guava",
        "@fcp_android_maven//:org_jetbrains_kotlin_kotlin_stdlib",
        "@fcp_maven//:com_google_errorprone_error_prone_annotations",
    ],
)
