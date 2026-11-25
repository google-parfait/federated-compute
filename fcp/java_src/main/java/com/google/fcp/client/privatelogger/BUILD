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

load("@rules_android//rules:rules.bzl", "android_library")

package(
    default_applicable_licenses = ["//fcp:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

# Public API interfaces.
android_library(
    name = "privatelogger",
    srcs = [
        "LogEntry.java",
        "PrivateLogger.java",
        "PrivateLoggerConfig.java",
    ],
    deps = [
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute:api",
        "//fcp/java_src/oss_build_defs:auto_value",
        "@com_google_protobuf//:protobuf_javalite",
        "@fcp_android_maven//:com_google_guava_guava",
        "@fcp_maven//:com_google_errorprone_error_prone_annotations",
    ],
)

android_library(
    name = "privatelogger_exceptions",
    srcs = [
        "PrivateLoggerUnavailableException.java",
    ],
)

android_library(
    name = "fcp_privatelogger",
    srcs = [
        "FcpPrivateLogger.java",
    ],
    deps = [
        ":privatelogger",
        "//fcp/java_src/main/java/com/google/fcp/client:time_source",
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute",
        "//fcp/java_src/main/java/com/google/fcp/client/privatelogger/fcpconnector",
        "//fcp/java_src/main/java/com/google/fcp/client/privatelogger/impl",
        "//fcp/java_src/main/java/com/google/fcp/client/stabilitymonitor",
        "//fcp/protos/confidentialcompute:access_policy_endorsement_options_java_proto_lite",
        "@fcp_android_maven//:androidx_annotation_annotation_jvm",
        "@fcp_android_maven//:com_google_guava_guava",
        "@fcp_maven//:com_google_errorprone_error_prone_annotations",
    ],
)
