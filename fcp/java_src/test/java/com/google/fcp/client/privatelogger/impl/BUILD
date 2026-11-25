# Copyright 2025 Google LLC
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

load("@com_google_protobuf//bazel:java_lite_proto_library.bzl", "java_lite_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@rules_android//rules:rules.bzl", "android_local_test")

package(
    default_applicable_licenses = ["//fcp:license"],
    default_visibility = [
        "//visibility:private",
    ],
    licenses = ["notice"],
)

android_local_test(
    name = "PrivateLoggerImplTest",
    srcs = ["PrivateLoggerImplTest.java"],
    manifest_values = {
        "minSdkVersion": "30",
        "targetSdkVersion": "30",
    },
    deps = [
        ":test_java_proto_lite",
        "//fcp/client:selector_context_java_proto_lite",
        "//fcp/java_src/main/java/com/google/fcp/client:time_source",
        "//fcp/java_src/main/java/com/google/fcp/client/privatelogger",
        "//fcp/java_src/main/java/com/google/fcp/client/privatelogger/impl",
        "//fcp/java_src/main/java/com/google/fcp/client/stabilitymonitor",
        "//fcp/java_src/oss_build_defs:room_android",
        "//fcp/java_src/test/java/com/google/fcp/client:fake_time_source",
        "@com_google_protobuf//:protobuf_javalite",
        "@fcp_android_maven//:androidx_test_core",
        "@fcp_android_maven//:com_google_guava_guava",
        "@fcp_android_maven//:com_google_truth_truth",
        "@fcp_android_maven//:junit_junit",
        "@fcp_android_maven//:org_mockito_mockito_core",
        "@fcp_android_maven//:org_robolectric_annotations",
        "@fcp_android_maven//:org_robolectric_robolectric",
        "@fcp_android_maven//:org_robolectric_shadows_framework",
        "@robolectric//bazel:android-all",
    ],
)

android_local_test(
    name = "TimestampUtilTest",
    srcs = ["TimestampUtilTest.java"],
    manifest_values = {
        "minSdkVersion": "30",
        "targetSdkVersion": "30",
    },
    deps = [
        "//fcp/java_src/main/java/com/google/fcp/client/privatelogger/impl",
        "@fcp_android_maven//:com_google_truth_truth",
        "@fcp_android_maven//:junit_junit",
        "@fcp_android_maven//:org_robolectric_annotations",
        "@fcp_android_maven//:org_robolectric_robolectric",
        "@fcp_android_maven//:org_robolectric_shadows_framework",
        "@robolectric//bazel:android-all",
    ],
)

proto_library(
    name = "test_proto",
    srcs = ["test.proto"],
    deps = [
        "@com_google_protobuf//:timestamp_proto",
    ],
)

java_lite_proto_library(
    name = "test_java_proto_lite",
    deps = [":test_proto"],
)
