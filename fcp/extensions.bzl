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

"""Loads dependencies for Federated Compute."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _fcp_deps_impl(ctx):
    # If a dependency supports bzlmod, add it in MODULE.bazel instead.
    # go/keep-sorted start block=yes newline_separated=yes ignore_prefixes=git_repository,http_archive
    http_archive(
        name = "bazel_toolchains",
        sha256 = "294cdd859e57fcaf101d4301978c408c88683fbc46fbc1a3829da92afbea55fb",
        strip_prefix = "bazel-toolchains-8c717f8258cd5f6c7a45b97d974292755852b658",
        url = "https://github.com/bazelbuild/bazel-toolchains/archive/8c717f8258cd5f6c7a45b97d974292755852b658.tar.gz",
    )

    http_archive(
        name = "differential_privacy_cc",
        sha256 = "6e6e1cd7a819695caae408f4fa938129ab7a86e83fe2410137c85e50131abbe0",
        strip_prefix = "differential-privacy-3.0.0/cc",
        url = "https://github.com/google/differential-privacy/archive/refs/tags/v3.0.0.tar.gz",
    )

    http_archive(
        name = "differential_privacy_common",
        sha256 = "6e6e1cd7a819695caae408f4fa938129ab7a86e83fe2410137c85e50131abbe0",
        strip_prefix = "differential-privacy-3.0.0",
        url = "https://github.com/google/differential-privacy/archive/refs/tags/v3.0.0.tar.gz",
    )

    git_repository(
        name = "libcppbor",
        build_file = "//third_party:libcppbor.BUILD.bzl",
        commit = "20d2be8672d24bfb441d075f82cc317d17d601f8",
        patches = ["//fcp/patches:libcppbor.patch"],
        remote = "https://android.googlesource.com/platform/external/libcppbor",
    )

    http_archive(
        name = "libcurl",
        build_file = "//third_party:curl.BUILD.bzl",
        sha256 = "e9d74b8586e0d2e6b45dc948bbe77525a1fa7f7c004ad5192f12e72c365e376e",
        strip_prefix = "curl-curl-7_84_0",
        url = "https://github.com/curl/curl/archive/refs/tags/curl-7_84_0.zip",
    )

    http_archive(
        name = "org_tensorflow_federated",
        integrity = "sha256-hpA6dFWuPBhJH0/rzvU3n/Vcq2qds5aqDDiFIWgH6Xk=",
        strip_prefix = "tensorflow-federated-bfe964f86dcbf096bdc06e1113073622b6e972ee",
        url = "https://github.com/google-parfait/tensorflow-federated/archive/bfe964f86dcbf096bdc06e1113073622b6e972ee.tar.gz",
    )

    http_archive(
        name = "protodatastore-cpp",
        sha256 = "d2627231fce0c9944100812b42d33203f7e03e78215d10123e78041b491005c3",
        strip_prefix = "protodatastore-cpp-0cd8b124bc65bcac105bce4a706923218ae5d625",
        url = "https://github.com/google/protodatastore-cpp/archive/0cd8b124bc65bcac105bce4a706923218ae5d625.zip",
    )
    # go/keep-sorted end

    # The extension is reproducible because all downloads are pinned to
    # specific commit hashes.
    return ctx.extension_metadata(reproducible = True)

fcp_deps = module_extension(
    doc = """\
Non-bzlmod enabled, non-dev dependencies for Federated Compute.

These dependencies are loaded using a module extension instead of
`http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")`
in MODULE.bazel so that modules depending on FCP can reuse this extension to
avoid duplicate dependencies -- and possible linker errors.

The list of dependencies specified here is not guaranteed to be stable, with
the goal of all dependencies moving to MODULE.bazel when they support bzlmod.
""",
    implementation = _fcp_deps_impl,
)
