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
load("@rules_java//java:defs.bzl", "java_library", "java_plugin")

package(
    default_applicable_licenses = ["//fcp:license"],
    default_visibility = ["//fcp:internal"],
    licenses = ["notice"],
)

java_plugin(
    name = "auto_value_plugin",
    generates_api = True,
    processor_class = "com.google.auto.value.processor.AutoValueProcessor",
    deps = ["@fcp_maven//:com_google_auto_value_auto_value"],
)

java_library(
    name = "auto_value",
    exported_plugins = [":auto_value_plugin"],
    exports = [
        "@fcp_android_maven//:com_google_auto_value_auto_value_annotations",
        "@fcp_android_maven//:com_google_errorprone_error_prone_annotations",
    ],
)

java_library(
    name = "room_schema_resolver",
    srcs = ["NoWriteSchemaFileResolver.java"],
    resource_strip_prefix = "fcp/java_src/oss_build_defs/resources",
    resources = ["resources/META-INF/services/androidx.room.util.SchemaFileResolver"],
    deps = ["@fcp_maven//:androidx_room_room_compiler"],
)

java_plugin(
    name = "room_compiler_plugin",
    processor_class = "androidx.room.RoomProcessor",
    deps = [
        ":room_schema_resolver",
        "@fcp_maven//:androidx_room_room_compiler",
    ],
)

android_library(
    name = "room_android",
    exported_plugins = [":room_compiler_plugin"],
    exports = [
        "@fcp_android_maven//:androidx_annotation_annotation_jvm",
        "@fcp_android_maven//:androidx_room_room_runtime_android",
    ],
)

android_library(
    name = "flogger_android",
    exports = [
        "@fcp_android_maven//:com_google_flogger_flogger",
        "@fcp_android_maven//:com_google_flogger_flogger_system_backend",
        "@fcp_android_maven//:com_google_flogger_google_extensions",
    ],
)
