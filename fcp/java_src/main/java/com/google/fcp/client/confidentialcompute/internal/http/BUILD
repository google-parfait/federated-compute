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

android_library(
    # Shared HTTP transport for the fetch and upload phases: FcpHttpClient and its helpers.
    name = "http",
    srcs = [
        "ElapsedRealtimeTicker.java",
        "FcpHttpClient.java",
        "FcpHttpException.java",
        "NetworkPhase.java",
        "UriEncoding.java",
        "UrlRedaction.java",
    ],
    deps = [
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute/http",
        "//fcp/java_src/main/java/com/google/fcp/client/stabilitymonitor",
        "//fcp/protos:code_java_proto_lite",
        "@com_google_protobuf//:protobuf_javalite",
        "@fcp_android_maven//:androidx_annotation_annotation_jvm",
        "@fcp_android_maven//:com_google_guava_guava",
    ],
)
