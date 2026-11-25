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

# Confidential compute client: fetches workload sets, verifies their keys, encrypts data to them and
# uploads it. Callers build a ConfidentialCompute (:confidentialcompute) and call the stages through
# :api. internal/* holds the default stages and is visible only within this package tree. See
# README.md.

load("@rules_android//rules:rules.bzl", "android_library")

package(
    default_applicable_licenses = ["//fcp:license"],
    default_visibility = ["//fcp:internal"],
    licenses = ["notice"],
)

# Stage interfaces and value types. Separate from :confidentialcompute so internal/ can depend on it
# without a dependency cycle.
android_library(
    name = "api",
    srcs = [
        "ConfidentialComputeCrypto.java",
        "ConfidentialPayloadEncoder.java",
        "ConfidentialPayloadUploader.java",
        "DeviceAttestationProvider.java",
        "EncodedPayload.java",
        "LegacyFcpCompat.java",
        "PayloadEncodingException.java",
        "TimestampedInput.java",
        "UploadAbortedException.java",
        "UploadTokens.java",
        "WorkloadSet.java",
        "WorkloadSetFetcher.java",
        "WorkloadSetSelectorSpec.java",
    ],
    plugins = [
        "//fcp/java_src/oss_build_defs:auto_value_plugin",
    ],
    visibility = [
        "//fcp:internal",
        "//visibility:public",
    ],
    deps = [
        "//fcp/java_src/oss_build_defs:auto_value",
        "//fcp/protos/confidentialcompute:access_policy_endorsement_options_java_proto_lite",
        "//fcp/protos/confidentialcompute:key_java_proto_lite",
        "//fcp/protos/confidentialcompute:workload_set_state_java_proto_lite",
        "@com_google_protobuf//:protobuf_javalite",
        "@fcp_android_maven//:com_google_guava_guava",
        "@fcp_maven//:com_google_errorprone_error_prone_annotations",
    ],
)

# ConfidentialCompute, the entry point that builds the stages.
android_library(
    name = "confidentialcompute",
    srcs = ["ConfidentialCompute.java"],
    visibility = [
        "//fcp:internal",
        "//visibility:public",
    ],
    deps = [
        ":api",
        "//fcp/java_src/main/java/com/google/fcp/client:time_source",
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute/http",
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute/internal/crypto:tink_crypto",
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute/internal/encode",
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute/internal/fetch",
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute/internal/http",
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute/internal/swor",
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute/internal/upload",
        "//fcp/java_src/main/java/com/google/fcp/client/flags",
        "//fcp/java_src/main/java/com/google/fcp/client/stabilitymonitor",
        "@fcp_android_maven//:androidx_annotation_annotation_jvm",
        "@fcp_android_maven//:com_google_guava_guava",
        "@fcp_maven//:com_google_errorprone_error_prone_annotations",
    ],
)
