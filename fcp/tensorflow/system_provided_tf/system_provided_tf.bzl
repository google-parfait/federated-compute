# Copyright 2022 Google LLC
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
"""Sets up a repository for using the system-provided TensorFlow."""

def _process_compile_flags(repository_ctx, python3, headers_dir):
    """Processes compilation flags required by the system-provided TF package.

    The tf.sysconfig module provides the compilation flags that should be used
    for custom operators. These will include the directory containing the
    TensorFlow C++ headers ("-I/some/path") and possibly other flags (e.g.,
    "-DSOME_FLAG=2").

    A symlink is created from `headers_dir` to the directory containing the C++
    headers. The list of other flags is returned.
    """
    result = repository_ctx.execute([
        python3,
        "-c",
        ";".join([
            "import tensorflow as tf",
            "print('\\0'.join(tf.sysconfig.get_compile_flags()))",
        ]),
    ])
    if result.return_code != 0:
        fail("Failed to determine TensorFlow compile flags; is TensorFlow installed?")
    include_dir = None
    copts = []
    cxxopts = []
    for flag in result.stdout.strip().split("\0"):
        if flag.startswith("-I"):
            if include_dir != None:
                fail("Only one TensorFlow headers directory is supported.")
            include_dir = flag[2:]
        elif flag.startswith("--std=c++"):  # Don't add C++-only flags to copts.
            cxxopts.append(flag)
        else:
            copts.append(flag)

    if not include_dir:
        fail("Unable to find TensorFlow headers directory.")
    repository_ctx.symlink(include_dir, headers_dir)

    return copts, cxxopts

def _process_link_flags(repository_ctx, python3, library_file):
    """Processes linker flags required by the system-provided TF package.

    The tf.sysconfig module provides the linker flags that should be used
    for custom operators. These will include the directory containing
    libtensorflow_framework.so ("-L/some/path"), the library to link
    ("-l:libtensorflow_framework.so.2"), and possibly other flags.

    A symlink is created from `library_file` to libtensorflow_framework.so. The
    list of other flags is returned.
    """
    result = repository_ctx.execute([
        python3,
        "-c",
        ";".join([
            "import tensorflow as tf",
            "print('\\0'.join(tf.sysconfig.get_link_flags()))",
        ]),
    ])
    if result.return_code != 0:
        fail("Failed to determine TensorFlow link flags; is TensorFlow installed?")
    link_dir = None
    library = None
    linkopts = []
    for flag in result.stdout.strip().split("\0"):
        if flag.startswith("-L"):
            if link_dir != None:
                fail("Only one TensorFlow libraries directory is supported.")
            link_dir = flag[2:]
        elif flag.startswith("-l"):
            if library != None:
                fail("Only one TensorFlow library is supported.")

            # "-l" may be followed by ":" to force the linker to use exact
            # library name resolution.
            library = flag[2:].lstrip(":")
        else:
            linkopts.append(flag)

    if not link_dir or not library:
        fail("Unable to find TensorFlow library.")
    repository_ctx.symlink(link_dir + "/" + library, library_file)

    return linkopts

def _tf_custom_op_configure_impl(repository_ctx):
    """Defines a repository for using the system-provided TensorFlow package.

    This is a lot like new_local_repository except that (a) the files to
    include are dynamically determined using TensorFlow's `tf.sysconfig` Python
    module, and (b) it provides build rules to compile and link C++ code with
    the necessary options to be compatible with the system-provided TensorFlow
    package.
    """
    python3 = repository_ctx.os.environ.get("PYTHON_BIN_PATH", "python3")

    # Name of the sub-directory that will link to TensorFlow C++ headers.
    headers_dir = "headers"

    # Name of the file that will link to libtensorflow_framework.so.
    library_file = "libtensorflow_framework.so"

    copts, cxxopts = _process_compile_flags(repository_ctx, python3, headers_dir)
    linkopts = _process_link_flags(repository_ctx, python3, library_file)

    # Create a BUILD file providing targets for the TensorFlow C++ headers and
    # framework library.
    repository_ctx.template(
        "BUILD",
        Label("//fcp/tensorflow/system_provided_tf:templates/BUILD.tpl"),
        substitutions = {
            "%{HEADERS_DIR}": headers_dir,
            "%{LIBRARY_FILE}": library_file,
        },
        executable = False,
    )

    # Create a bzl file providing rules for compiling C++ code compatible with
    # the TensorFlow package.
    repository_ctx.template(
        "system_provided_tf.bzl",
        Label("//fcp/tensorflow/system_provided_tf:templates/system_provided_tf.bzl.tpl"),
        substitutions = {
            "%{COPTS}": str(copts),
            "%{CXXOPTS}": str(cxxopts),
            "%{LINKOPTS}": str(linkopts),
            "%{REPOSITORY_NAME}": repository_ctx.name,
        },
        executable = False,
    )

system_provided_tf = repository_rule(
    implementation = _tf_custom_op_configure_impl,
    configure = True,
    doc = """Creates a repository with targets for the system-provided TensorFlow.

This repository defines (a) //:tf_headers providing the C++ TensorFlow headers,
(b) //:libtensorflow_framework providing the TensorFlow framework shared
library, and (c) //:system_provided_tf.bzl for building custom op libraries
that are compatible with the system-provided TensorFlow package.
""",
    environ = [
        "PYTHON_BIN_PATH",
    ],
    local = True,
)
