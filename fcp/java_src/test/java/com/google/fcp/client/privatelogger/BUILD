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

load("@rules_android//rules:rules.bzl", "android_local_test")

package(
    default_applicable_licenses = ["//fcp:license"],
    default_visibility = [
        "//visibility:private",
    ],
    licenses = ["notice"],
)

android_local_test(
    name = "FcpPrivateLoggerTest",
    srcs = ["FcpPrivateLoggerTest.java"],
    manifest_values = {
        "minSdkVersion": "30",
        "targetSdkVersion": "36",
    },
    deps = [
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute",
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute:api",
        "//fcp/java_src/main/java/com/google/fcp/client/privatelogger",
        "//fcp/java_src/main/java/com/google/fcp/client/privatelogger:fcp_privatelogger",
        "//fcp/java_src/test/java/com/google/fcp/client/confidentialcompute/internal/http:fake_http_engine",
        "//fcp/protos/confidentialcompute:access_policy_endorsement_options_java_proto_lite",
        "@com_google_protobuf//:protobuf_javalite",
        "@fcp_android_maven//:androidx_test_core",
        "@fcp_android_maven//:androidx_test_ext_junit",
        "@fcp_android_maven//:com_google_guava_guava",
        "@fcp_android_maven//:com_google_truth_truth",
        "@fcp_android_maven//:junit_junit",
        "@fcp_android_maven//:org_robolectric_annotations",
        "@fcp_android_maven//:org_robolectric_robolectric",
        "@fcp_android_maven//:org_robolectric_shadows_framework",
        "@robolectric//bazel:android-all",
    ],
)
