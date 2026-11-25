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
        "//fcp:internal",
    ],
    licenses = ["notice"],
)

android_library(
    name = "fcpconnector",
    srcs = [
        "FcpPrivateLoggerConnector.java",
        "FcpPrivateLoggerConnectorDeps.java",
    ],
    deps = [
        "//fcp/client:selector_context_java_proto_lite",
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute",
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute:api",
        "//fcp/java_src/main/java/com/google/fcp/client/privatelogger",
        "//fcp/java_src/main/java/com/google/fcp/client/privatelogger:privatelogger_exceptions",
        "//fcp/java_src/main/java/com/google/fcp/client/privatelogger/impl",
        "//fcp/java_src/main/java/com/google/fcp/client/stabilitymonitor",
        "//fcp/java_src/oss_build_defs:auto_value",
        "//fcp/java_src/oss_build_defs:flogger_android",
        "//fcp/protos/confidentialcompute:access_policy_endorsement_options_java_proto_lite",
        "@fcp_android_maven//:com_google_guava_guava",
    ],
)
