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

load("@rules_android//rules:rules.bzl", "android_library")

package(
    default_applicable_licenses = ["//fcp:license"],
    default_visibility = [
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute:__subpackages__",
        "//fcp/java_src/test/java/com/google/fcp/client:__subpackages__",
    ],
    licenses = ["notice"],
)

# Uploader implementation transmitting encrypted payloads and reporting completion.
android_library(
    name = "upload",
    srcs = [
        "FcpConfidentialPayloadUploader.java",
        "FcpConfidentialPayloadUploaderDeps.java",
    ],
    plugins = [
        "//fcp/java_src/oss_build_defs:auto_value_plugin",
    ],
    deps = [
        "//fcp/client:diag_codes_java_proto_lite",
        "//fcp/java_src/main/java/com/google/fcp/client:time_source",
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute:api",
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute/http",
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute/internal/fetch",
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute/internal/http",
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute/internal/swor",
        "//fcp/java_src/main/java/com/google/fcp/client/flags",
        "//fcp/java_src/main/java/com/google/fcp/client/stabilitymonitor",
        "//fcp/java_src/oss_build_defs:auto_value",
        "//fcp/java_src/oss_build_defs:flogger_android",
        "//fcp/protos:code_java_proto_lite",
        "//fcp/protos/federatedcompute:federated_compute_java_proto_lite",
        "@fcp_android_maven//:com_google_guava_guava",
    ],
)
