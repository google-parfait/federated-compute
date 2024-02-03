"""
A custom build file for libcbor
"""

licenses(["notice"])

exports_files(["LICENSE.md"])

cc_library(
    name = "cbor",
    srcs = glob(["src/**/*.c", "src/**/*.h"]),
    hdrs = ["src/cbor.h"] + glob(["src/cbor/*.h"]),
    copts = ["-std=c99"],
    includes = ["src"],
    visibility = ["//visibility:public"],
)
