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
    default_visibility = ["//fcp:internal"],
    licenses = ["notice"],
)

# Crash and diagnostic event reporting for the client libraries. Its own target, like
# confidentialcompute/http, so that neither the privatelogger nor the confidentialcompute API owns
# it.
android_library(
    name = "stabilitymonitor",
    srcs = [
        "DiagnosticEvent.java",
        "StabilityMonitor.java",
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
        "@fcp_android_maven//:androidx_annotation_annotation_jvm",
        "@fcp_maven//:com_google_errorprone_error_prone_annotations",
    ],
)
