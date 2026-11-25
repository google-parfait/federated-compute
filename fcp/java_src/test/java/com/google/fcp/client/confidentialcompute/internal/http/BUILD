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

load("@rules_android//rules:rules.bzl", "android_library", "android_local_test")

package(
    default_applicable_licenses = ["//fcp:license"],
    default_visibility = [
        "//fcp:internal",
    ],
    licenses = ["notice"],
)

android_local_test(
    name = "UrlRedactionTest",
    srcs = ["UrlRedactionTest.java"],
    manifest_values = {
        "minSdkVersion": "30",
        "targetSdkVersion": "30",
    },
    deps = [
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute/internal/http",
        "@fcp_android_maven//:com_google_truth_truth",
        "@fcp_android_maven//:junit_junit",
        "@fcp_android_maven//:org_robolectric_annotations",
        "@fcp_android_maven//:org_robolectric_robolectric",
        "@fcp_android_maven//:org_robolectric_shadows_framework",
        "@robolectric//bazel:android-all",
    ],
)

android_local_test(
    name = "UriEncodingTest",
    srcs = ["UriEncodingTest.java"],
    manifest_values = {
        "minSdkVersion": "30",
        "targetSdkVersion": "30",
    },
    deps = [
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute/internal/http",
        "@fcp_android_maven//:com_google_truth_truth",
        "@fcp_android_maven//:junit_junit",
        "@fcp_android_maven//:org_robolectric_annotations",
        "@fcp_android_maven//:org_robolectric_robolectric",
        "@fcp_android_maven//:org_robolectric_shadows_framework",
        "@robolectric//bazel:android-all",
    ],
)

android_local_test(
    name = "FcpHttpExceptionTest",
    srcs = ["FcpHttpExceptionTest.java"],
    manifest_values = {
        "minSdkVersion": "30",
        "targetSdkVersion": "30",
    },
    deps = [
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute/internal/http",
        "//fcp/protos:code_java_proto_lite",
        "@fcp_android_maven//:com_google_truth_truth",
        "@fcp_android_maven//:junit_junit",
        "@fcp_android_maven//:org_robolectric_annotations",
        "@fcp_android_maven//:org_robolectric_robolectric",
        "@fcp_android_maven//:org_robolectric_shadows_framework",
        "@robolectric//bazel:android-all",
    ],
)

android_local_test(
    name = "FcpHttpClientTest",
    srcs = ["FcpHttpClientTest.java"],
    manifest_values = {
        "minSdkVersion": "30",
        "targetSdkVersion": "30",
    },
    deps = [
        ":fake_http_engine",
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute/http",
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute/internal/http",
        "@com_google_protobuf//:protobuf_javalite",
        "@fcp_android_maven//:com_google_guava_guava",
        "@fcp_android_maven//:com_google_truth_truth",
        "@fcp_android_maven//:junit_junit",
        "@fcp_android_maven//:org_robolectric_annotations",
        "@fcp_android_maven//:org_robolectric_robolectric",
        "@fcp_android_maven//:org_robolectric_shadows_framework",
        "@robolectric//bazel:android-all",
    ],
)

android_library(
    name = "fake_http_engine",
    testonly = 1,
    srcs = ["FakeHttpUrlConnectionEngine.java"],
    deps = [
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute/http",
        "@fcp_android_maven//:com_google_guava_guava",
    ],
)

android_local_test(
    name = "FakeHttpUrlConnectionEngineTest",
    srcs = ["FakeHttpUrlConnectionEngineTest.java"],
    manifest_values = {
        "minSdkVersion": "30",
        "targetSdkVersion": "30",
    },
    deps = [
        ":fake_http_engine",
        "@fcp_android_maven//:com_google_truth_truth",
        "@fcp_android_maven//:junit_junit",
        "@fcp_android_maven//:org_robolectric_annotations",
        "@fcp_android_maven//:org_robolectric_robolectric",
        "@fcp_android_maven//:org_robolectric_shadows_framework",
        "@robolectric//bazel:android-all",
    ],
)
