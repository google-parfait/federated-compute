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
"""Provides rules for building custom TensorFlow ops compatible with pip."""

def _force_system_provided_tf_transition_impl(settings, attr):
    copts = list(settings["//command_line_option:copt"])
    cxxopts = list(settings["//command_line_option:cxxopt"])
    linkopts = list(settings["//command_line_option:linkopt"])
    copts += %{COPTS}
    cxxopts += %{CXXOPTS}
    linkopts += %{LINKOPTS}
    # TensorFlow's pip package was built with libstdc++; ensure the right
    # standard library is used if the compiler supports multiple.
    if attr.compiler_supports_stdlib:
        cxxopts += ["-stdlib=libstdc++"]
        linkopts += ["-stdlib=libstdc++"]
    return {
        "//:system_provided_tf_build_setting": True,
        "//command_line_option:copt": copts,
        "//command_line_option:cxxopt": cxxopts,
        "//command_line_option:linkopt": linkopts,
    }

_force_system_provided_tf_transition = transition(
    implementation = _force_system_provided_tf_transition_impl,
    inputs = [
        "//command_line_option:copt",
        "//command_line_option:cxxopt",
        "//command_line_option:linkopt",
    ],
    outputs = [
        "//:system_provided_tf_build_setting",
        "//command_line_option:copt",
        "//command_line_option:cxxopt",
        "//command_line_option:linkopt",
    ],
)

def _force_system_provided_tf_impl(ctx):
    cc_binary = ctx.attr.cc_binary[0]
    output_file = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.symlink(
        output = output_file,
        target_file = cc_binary.files.to_list()[0],
    )
    return DefaultInfo(
        files = depset([output_file]),
        data_runfiles = ctx.runfiles(transitive_files = depset([output_file])),
    )

_force_system_provided_tf = rule(
    doc = """Forces a shared library to be built in a way that's compatible
with the system-provided Python TensorFlow package.""",
    implementation = _force_system_provided_tf_impl,
    attrs = {
        "cc_binary": attr.label(
            cfg = _force_system_provided_tf_transition,
            mandatory = True,
            doc = "The cc_binary target to build with TensorFlow compatibility.",
        ),
        # Compiler information cannot be read by the transition directly because
        # the StarlarkAttributeTransitionProvider doesn't yet support providers
        # for dependency-typed attributes.
        "compiler_supports_stdlib": attr.bool(
            doc = "Whether the compiler supports the --stdlib flag.",
        ),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
)

def tf_custom_op_library(
        name,
        srcs = [],
        deps = [],
        tags = [],
        visibility = None,
        **kwargs):
    """Helper to build a dynamic library (.so) from the sources containing
    implementations of custom ops and kernels.

    This rule will force a transition to an environment that targets the
    system-provided TF library. This means that all deps of this target and the
    target's own sources will be compiled with the necessary compiler flags to
    correctly target a system-provided TF library.

    The `@system_provided_tf//:system_provided_tf_build` setting will also be
    true when those deps are built for this target.
    """

    native.cc_binary(
        name = name + "_lib",
        srcs = srcs,
        linkshared = 1,
        deps = deps + [
            "@%{REPOSITORY_NAME}//:libtensorflow_framework",
            "@%{REPOSITORY_NAME}//:tf_headers",
        ],
        tags = tags + ["manual"],
        visibility = ["//visibility:private"],
        **kwargs
    )

    _force_system_provided_tf(
        name = name,
        cc_binary = name + "_lib",
        compiler_supports_stdlib = select({
            "@%{REPOSITORY_NAME}//:clang_compiler": True,
            "//conditions:default": False,
        }),
        visibility = visibility,
        tags = tags,
    )
