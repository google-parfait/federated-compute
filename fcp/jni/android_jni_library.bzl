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

"""Bazel macro for building Android JNI shared libraries."""

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

def android_jni_library(name, srcs = [], deps = [], testonly = False, visibility = None):
    """Builds lib<name>.so for JNI tests and exposes a CcInfo target named <name>."""
    cc_library(
        name = name + "_cc",
        srcs = srcs,
        deps = deps,
        alwayslink = True,
        testonly = testonly,
        visibility = ["//visibility:private"],
    )
    cc_binary(
        name = "lib" + name + ".so",
        deps = [":" + name + "_cc"],
        linkshared = True,
        testonly = testonly,
        visibility = visibility,
    )
    cc_import(
        name = name,
        shared_library = ":lib" + name + ".so",
        testonly = testonly,
        visibility = visibility,
    )
