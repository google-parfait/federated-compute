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

load("@com_google_protobuf//bazel:java_lite_proto_library.bzl", "java_lite_proto_library")
load("@rules_android//rules:rules.bzl", "android_library")

package(
    default_applicable_licenses = ["//fcp:license"],
    default_visibility = [
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute:__subpackages__",
        "//fcp/java_src/test/java/com/google/fcp/client:__subpackages__",
    ],
    licenses = ["notice"],
)

java_lite_proto_library(
    name = "tensor_java_proto_lite",
    deps = ["@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor_proto"],
)

android_library(
    name = "encode",
    srcs = [
        "AttestationRecordLogger.java",
        "ConfidentialComputeEncryptor.java",
        "ConfidentialPayloadEncoderImpl.java",
        "FederatedComputeCheckpointBuilder.java",
    ],
    deps = [
        ":tensor_java_proto_lite",
        "//fcp/java_src/main/java/com/google/fcp/client:time_source",
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute:api",
        "//fcp/java_src/main/java/com/google/fcp/client/stabilitymonitor",
        "//fcp/java_src/oss_build_defs:flogger_android",
        "//fcp/protos/confidentialcompute:access_policy_endorsement_options_java_proto_lite",
        "//fcp/protos/confidentialcompute:aggregation_client_payload_java_proto_lite",
        "//fcp/protos/confidentialcompute:blob_header_java_proto_lite",
        "//fcp/protos/confidentialcompute:key_java_proto_lite",
        "@com_google_protobuf//:protobuf_javalite",
        "@fcp_android_maven//:com_google_guava_guava",
    ],
)
