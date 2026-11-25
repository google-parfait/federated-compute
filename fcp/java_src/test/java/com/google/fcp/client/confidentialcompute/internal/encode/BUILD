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

android_library(
    name = "fake_crypto",
    testonly = True,
    srcs = ["FakeConfidentialComputeCrypto.java"],
    deps = [
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute:api",
        "//fcp/protos/confidentialcompute:access_policy_endorsement_options_java_proto_lite",
        "//fcp/protos/confidentialcompute:aggregation_client_payload_java_proto_lite",
        "//fcp/protos/confidentialcompute:key_java_proto_lite",
        "@com_google_protobuf//:protobuf_javalite",
    ],
)

android_local_test(
    name = "FederatedComputeCheckpointBuilderTest",
    srcs = ["FederatedComputeCheckpointBuilderTest.java"],
    manifest_values = {
        "minSdkVersion": "24",
        "targetSdkVersion": "36",
    },
    deps = [
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute:api",
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute/internal/encode",
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute/internal/encode:tensor_java_proto_lite",
        "@com_google_protobuf//:protobuf_javalite",
        "@fcp_android_maven//:com_google_guava_guava",
        "@fcp_android_maven//:junit_junit",
        "@fcp_android_maven//:org_robolectric_annotations",
        "@fcp_android_maven//:org_robolectric_robolectric",
        "@fcp_android_maven//:org_robolectric_shadows_framework",
        "@fcp_maven//:com_google_truth_truth",
        "@robolectric//bazel:android-all",
    ],
)

android_local_test(
    name = "ConfidentialComputeEncryptorTest",
    srcs = ["ConfidentialComputeEncryptorTest.java"],
    manifest_values = {
        "minSdkVersion": "24",
        "targetSdkVersion": "36",
    },
    deps = [
        ":fake_crypto",
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute/internal/encode",
        "//fcp/protos/confidentialcompute:aggregation_client_payload_java_proto_lite",
        "//fcp/protos/confidentialcompute:key_java_proto_lite",
        "@com_google_protobuf//:protobuf_javalite",
        "@fcp_android_maven//:junit_junit",
        "@fcp_android_maven//:org_robolectric_annotations",
        "@fcp_android_maven//:org_robolectric_robolectric",
        "@fcp_android_maven//:org_robolectric_shadows_framework",
        "@fcp_maven//:com_google_truth_truth",
        "@robolectric//bazel:android-all",
    ],
)

android_local_test(
    name = "ConfidentialPayloadEncoderTest",
    srcs = ["ConfidentialPayloadEncoderTest.java"],
    manifest_values = {
        "minSdkVersion": "24",
        "targetSdkVersion": "36",
    },
    deps = [
        ":fake_crypto",
        "//fcp/java_src/main/java/com/google/fcp/client:time_source",
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute:api",
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute/internal/encode",
        "//fcp/java_src/main/java/com/google/fcp/client/stabilitymonitor",
        "//fcp/java_src/test/java/com/google/fcp/client/stabilitymonitor:recording_stability_monitor",
        "//fcp/protos/confidentialcompute:access_policy_endorsement_options_java_proto_lite",
        "//fcp/protos/confidentialcompute:blob_header_java_proto_lite",
        "//fcp/protos/confidentialcompute:key_java_proto_lite",
        "@com_google_protobuf//:protobuf_javalite",
        "@fcp_android_maven//:androidx_test_ext_junit",
        "@fcp_android_maven//:com_google_guava_guava",
        "@fcp_android_maven//:junit_junit",
        "@fcp_android_maven//:org_robolectric_annotations",
        "@fcp_android_maven//:org_robolectric_robolectric",
        "@fcp_android_maven//:org_robolectric_shadows_framework",
        "@fcp_maven//:com_google_truth_truth",
        "@robolectric//bazel:android-all",
    ],
)

android_local_test(
    name = "AttestationRecordLoggerTest",
    srcs = ["AttestationRecordLoggerTest.java"],
    manifest_values = {
        "minSdkVersion": "30",
        "targetSdkVersion": "36",
    },
    deps = [
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute/internal/encode",
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
