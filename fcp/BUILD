# Description
# Federated Computation Platform (FCP) is a computational framework,
# which allows to orchestrate decentralized computations across
# large set of dynamically available nodes (e.g. mobile devices).

load("@rules_python//python:pip.bzl", "compile_pip_requirements")

package(
    default_visibility = [":internal"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files([
    "LICENSE",
])

package_group(
    name = "internal",
    includes = [
    ],
    packages = [
        "//fcp/...",
    ],
)

compile_pip_requirements(
    name = "requirements",
    timeout = "moderate",
    src = "requirements.in",
    extra_args = [
        "--allow-unsafe",
        "--resolver=backtracking",
    ],
    requirements_txt = "requirements.txt",
)
