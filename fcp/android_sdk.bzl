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

"""Hermetic Android SDK module extension for Federated Compute OSS builds.

Setting up a hermetic Android SDK toolchain with rules_android 0.8.1 is still
manual because rules_android's built-in android_sdk_repository_extension only
reads a preinstalled ANDROID_HOME directory from the host machine and does not
download SDK archives itself.

Instead, this dev-only extension:
  1. Downloads pinned Android SDK Platform (API 37), Build-Tools (37.0.0), and
     Platform-Tools archives directly from dl.google.com/android/repository/
     with verified SHA-256 digests.
  2. Symlinks each archive's single top-level directory into the standard
     Android SDK layout (platforms/android-37, build-tools/37.0.0,
     platform-tools).
  3. Writes the empty placeholder jar expected by rules_android's fallback SDK
     toolchain and generates a BUILD.bazel file invoking
     create_android_sdk_rules alongside the sdk_toolchain_type, has_androidsdk,
     adb, dexdump, and sdk_path targets that rules_android expects.

MODULE.bazel overrides android_sdk_repository_extension's androidsdk repo with
this repository (dev_dependency = True). This file can be removed once
rules_android provides a built-in hermetic SDK download extension.
"""

_PLATFORM_37_URL = "https://dl.google.com/android/repository/platform-37.2_r02.zip"
_PLATFORM_37_SHA256 = "17af98ec1427a1fed9ea278184aba71e53b87eca103655602f9fc64ef11ea4d3"

_BUILD_TOOLS_37_LINUX_URL = "https://dl.google.com/android/repository/build-tools_r37_linux.zip"
_BUILD_TOOLS_37_LINUX_SHA256 = "01af179347cbcd9c208b7f8171f7b21f6dd1d2f85bcd15e88caa51d5d7b86060"

_PLATFORM_TOOLS_LINUX_URL = "https://dl.google.com/android/repository/platform-tools_r37.0.3-linux.zip"
_PLATFORM_TOOLS_LINUX_SHA256 = "214548146cadb2ca3508143d8a1d4be4c59de5bb837253e3ede89ed74cf450d1"

_BUILD_FILE_TEMPLATE = """\
load(
    "@rules_android//rules/android_sdk_repository:helper.bzl",
    "create_android_sdk_rules",
    "create_system_images_filegroups",
)

package(default_visibility = ["//visibility:public"])

toolchain_type(name = "sdk_toolchain_type")

config_feature_flag(
    name = "true",
    allowed_values = ["true", "false"],
    default_value = "true",
)

config_setting(
    name = "always_true",
    flag_values = {{":true": "true"}},
)

config_setting(
    name = "always_false",
    flag_values = {{":true": "false"}},
)

alias(
    name = "has_androidsdk",
    actual = ":always_true",
)

create_android_sdk_rules(
    name = "{repo_name}",
    build_tools_version = "37.0.0",
    build_tools_directory = "37.0.0",
    api_levels = ["37"],
    default_api_level = "37",
)

exports_files(["platform-tools/adb"])

alias(
    name = "adb",
    actual = "platform-tools/adb",
)

alias(
    name = "dexdump",
    actual = "build-tools/37.0.0/dexdump",
)

filegroup(
    name = "sdk_path",
    srcs = ["."],
)

create_system_images_filegroups(
    system_image_dirs = [],
)
"""

def _download_single_root_zip(repo_ctx, url, sha256, dest_dir):
    tmp_dir = "_extract_" + dest_dir.replace("/", "_").replace(".", "_")
    repo_ctx.download_and_extract(url = url, output = tmp_dir, sha256 = sha256)
    entries = repo_ctx.path(tmp_dir).readdir()
    src = entries[0] if len(entries) == 1 and entries[0].is_dir else repo_ctx.path(tmp_dir)
    repo_ctx.symlink(src, dest_dir)

def _fcp_android_sdk_repo_impl(repo_ctx):
    _download_single_root_zip(
        repo_ctx,
        _PLATFORM_37_URL,
        _PLATFORM_37_SHA256,
        "platforms/android-37",
    )
    _download_single_root_zip(
        repo_ctx,
        _BUILD_TOOLS_37_LINUX_URL,
        _BUILD_TOOLS_37_LINUX_SHA256,
        "build-tools/37.0.0",
    )
    _download_single_root_zip(
        repo_ctx,
        _PLATFORM_TOOLS_LINUX_URL,
        _PLATFORM_TOOLS_LINUX_SHA256,
        "platform-tools",
    )

    # Filename required by rules_android's fallback SDK toolchain.
    repo_ctx.file("dummy.jar", "")  # RESPECTFUL_TERMS_EXCEPTION
    repo_ctx.file("BUILD.bazel", _BUILD_FILE_TEMPLATE.format(repo_name = repo_ctx.name))

_fcp_android_sdk_repo = repository_rule(
    implementation = _fcp_android_sdk_repo_impl,
)

def _fcp_android_sdk_ext_impl(module_ctx):
    _fcp_android_sdk_repo(name = "fcp_androidsdk")
    return module_ctx.extension_metadata(reproducible = True)

fcp_android_sdk = module_extension(
    implementation = _fcp_android_sdk_ext_impl,
)
