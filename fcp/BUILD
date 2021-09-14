# Description
# Federated Computation Platform (FCP) is a computational framework,
# which allows to orchestrate decentralized computations across
# large set of dynamically available nodes (e.g. mobile devices).

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

package_group(
    name = "public_api_visibility",
    includes = [":internal"],
    packages = [
    ],
)
