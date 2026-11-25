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
load("@rules_java//java:defs.bzl", "java_plugin")

package(
    default_applicable_licenses = ["//fcp:license"],
    default_visibility = [
        "//fcp:internal",
    ],
    licenses = ["notice"],
)

android_library(
    name = "impl",
    srcs = [
        "ContributedEntry.java",
        "ContributedEntryDao.java",
        "DataProvider.java",
        "DatabaseMetadata.java",
        "DatabaseMetadataDao.java",
        "DbLogEntry.java",
        "LegacyContributionMigration.java",
        "LogEntryDao.java",
        "PrivateLoggerConnector.java",
        "PrivateLoggerDatabase.java",
        "PrivateLoggerImpl.java",
        "PrivateLoggerImplDeps.java",
        "Task.java",
        "TaskDao.java",
        "TimestampUtil.java",
        "UploadOutcome.java",
    ],
    javacopts = ["-Aroom.schemaLocation=fcp/java_src/main/java/com/google/fcp/client/privatelogger/impl/schemas"],
    plugins = [
        ":schemas_plugin",
        "//fcp/java_src/oss_build_defs:auto_value_plugin",
    ],
    deps = [
        "//fcp/client:selector_context_java_proto_lite",
        "//fcp/java_src/main/java/com/google/fcp/client:time_source",
        "//fcp/java_src/main/java/com/google/fcp/client/privatelogger",
        "//fcp/java_src/main/java/com/google/fcp/client/stabilitymonitor",
        "//fcp/java_src/oss_build_defs:auto_value",
        "//fcp/java_src/oss_build_defs:room_android",
        "@com_google_protobuf//:protobuf_javalite",
        "@fcp_android_maven//:androidx_annotation_annotation_jvm",
        "@fcp_android_maven//:com_google_guava_guava",
        "@fcp_maven//:com_google_errorprone_error_prone_annotations",
    ],
)

filegroup(
    name = "schemas",
    srcs = glob(["schemas/**/*.json"]),
    visibility = ["//visibility:public"],
)

java_plugin(
    name = "schemas_plugin",
    data = [":schemas"],
    visibility = ["//visibility:public"],
)
