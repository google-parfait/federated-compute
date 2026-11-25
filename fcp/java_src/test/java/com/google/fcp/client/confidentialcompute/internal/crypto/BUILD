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

load("@rules_android//rules:rules.bzl", "android_library", "android_local_test")
load("//fcp/jni:android_jni_library.bzl", "android_jni_library")

package(
    default_applicable_licenses = ["//fcp:license"],
    default_visibility = ["//fcp:internal"],
    licenses = ["notice"],
)

android_library(
    name = "attestation_test_fixtures",
    testonly = True,
    srcs = ["AttestationTestFixtures.java"],
    deps = [
        "//fcp/protos/confidentialcompute:access_policy_endorsement_options_java_proto_lite",
        "//fcp/protos/confidentialcompute:key_java_proto_lite",
        "//fcp/protos/confidentialcompute:payload_transparency_java_proto_lite",
        "//fcp/protos/confidentialcompute:signed_endorsements_java_proto_lite",
        "//fcp/protos/federatedcompute:confidential_encryption_config_java_proto_lite",
        "@com_google_protobuf//:protobuf_javalite",
        "@fcp_android_maven//:com_google_guava_guava",
    ],
)

TINK_PRIMITIVES = "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute/internal/crypto:tink_primitives"

android_library(
    name = "aes_gcm_siv_rfc_test_vectors",
    testonly = True,
    srcs = ["AesGcmSivRfcTestVectors.java"],
    deps = [
        TINK_PRIMITIVES,
        "@fcp_android_maven//:com_google_guava_guava",
    ],
)

android_local_test(
    name = "AesGcmSivCallerProvidedNonceAeadTest",
    size = "medium",
    srcs = ["AesGcmSivCallerProvidedNonceAeadTest.java"],
    manifest_values = {
        "minSdkVersion": "24",
        "targetSdkVersion": "30",
    },
    deps = [
        ":aes_gcm_siv_rfc_test_vectors",
        TINK_PRIMITIVES,
        "@fcp_android_maven//:com_google_crypto_tink_tink_android",
        "@fcp_android_maven//:com_google_guava_guava",
        "@fcp_android_maven//:com_google_truth_truth",
        "@fcp_android_maven//:junit_junit",
        "@fcp_android_maven//:org_robolectric_annotations",
        "@fcp_android_maven//:org_robolectric_robolectric",
        "@fcp_android_maven//:org_robolectric_shadows_framework",
        "@fcp_maven//:junit_junit",
        "@robolectric//bazel:android-all",
    ],
)

android_local_test(
    name = "AesGcmSivCallerProvidedNonceAeadWithoutConscryptTest",
    size = "small",
    srcs = ["AesGcmSivCallerProvidedNonceAeadWithoutConscryptTest.java"],
    manifest_values = {
        "minSdkVersion": "24",
        "targetSdkVersion": "30",
    },
    deps = [
        ":aes_gcm_siv_rfc_test_vectors",
        TINK_PRIMITIVES,
        "@fcp_android_maven//:com_google_crypto_tink_tink_android",
        "@fcp_android_maven//:com_google_truth_truth",
        "@fcp_android_maven//:junit_junit",
        "@fcp_android_maven//:org_robolectric_annotations",
        "@fcp_android_maven//:org_robolectric_robolectric",
        "@fcp_android_maven//:org_robolectric_shadows_framework",
        "@fcp_maven//:junit_junit",
        "@robolectric//bazel:android-all",
    ],
)

android_local_test(
    name = "HpkeUtilTest",
    size = "small",
    srcs = ["HpkeUtilTest.java"],
    manifest_values = {
        "minSdkVersion": "30",
        "targetSdkVersion": "30",
    },
    deps = [
        TINK_PRIMITIVES,
        "@fcp_android_maven//:com_google_crypto_tink_tink_android",
        "@fcp_android_maven//:com_google_guava_guava",
        "@fcp_android_maven//:com_google_truth_truth",
        "@fcp_android_maven//:junit_junit",
        "@fcp_android_maven//:org_robolectric_annotations",
        "@fcp_android_maven//:org_robolectric_robolectric",
        "@fcp_android_maven//:org_robolectric_shadows_framework",
        "@fcp_maven//:junit_junit",
        "@robolectric//bazel:android-all",
    ],
)

android_library(
    name = "hpke_rfc_test_vectors",
    testonly = True,
    srcs = ["HpkeRfcTestVectors.java"],
    deps = ["@fcp_android_maven//:com_google_guava_guava"],
)

android_local_test(
    name = "HpkeContextTest",
    size = "small",
    srcs = ["HpkeContextTest.java"],
    manifest_values = {
        "minSdkVersion": "30",
        "targetSdkVersion": "30",
    },
    deps = [
        ":hpke_rfc_test_vectors",
        TINK_PRIMITIVES,
        "@fcp_android_maven//:com_google_crypto_tink_tink_android",
        "@fcp_android_maven//:com_google_guava_guava",
        "@fcp_android_maven//:com_google_truth_truth",
        "@fcp_android_maven//:junit_junit",
        "@fcp_android_maven//:org_robolectric_annotations",
        "@fcp_android_maven//:org_robolectric_robolectric",
        "@fcp_android_maven//:org_robolectric_shadows_framework",
        "@fcp_maven//:junit_junit",
        "@robolectric//bazel:android-all",
    ],
)

android_local_test(
    name = "X25519HpkeKemTest",
    size = "small",
    srcs = ["X25519HpkeKemTest.java"],
    manifest_values = {
        "minSdkVersion": "30",
        "targetSdkVersion": "30",
    },
    deps = [
        ":hpke_rfc_test_vectors",
        TINK_PRIMITIVES,
        "@fcp_android_maven//:com_google_crypto_tink_tink_android",
        "@fcp_android_maven//:com_google_truth_truth",
        "@fcp_android_maven//:junit_junit",
        "@fcp_android_maven//:org_robolectric_annotations",
        "@fcp_android_maven//:org_robolectric_robolectric",
        "@fcp_android_maven//:org_robolectric_shadows_framework",
        "@fcp_maven//:junit_junit",
        "@robolectric//bazel:android-all",
    ],
)

android_local_test(
    name = "HkdfHpkeKdfTest",
    size = "small",
    srcs = ["HkdfHpkeKdfTest.java"],
    manifest_values = {
        "minSdkVersion": "30",
        "targetSdkVersion": "30",
    },
    deps = [
        ":hpke_rfc_test_vectors",
        TINK_PRIMITIVES,
        "@fcp_android_maven//:com_google_crypto_tink_tink_android",
        "@fcp_android_maven//:com_google_guava_guava",
        "@fcp_android_maven//:com_google_truth_truth",
        "@fcp_android_maven//:junit_junit",
        "@fcp_android_maven//:org_robolectric_annotations",
        "@fcp_android_maven//:org_robolectric_robolectric",
        "@fcp_android_maven//:org_robolectric_shadows_framework",
        "@fcp_maven//:junit_junit",
        "@robolectric//bazel:android-all",
    ],
)

android_local_test(
    name = "AesGcmHpkeAeadTest",
    size = "small",
    srcs = ["AesGcmHpkeAeadTest.java"],
    manifest_values = {
        "minSdkVersion": "30",
        "targetSdkVersion": "30",
    },
    deps = [
        ":hpke_rfc_test_vectors",
        TINK_PRIMITIVES,
        "@fcp_android_maven//:com_google_truth_truth",
        "@fcp_android_maven//:junit_junit",
        "@fcp_android_maven//:org_robolectric_annotations",
        "@fcp_android_maven//:org_robolectric_robolectric",
        "@fcp_android_maven//:org_robolectric_shadows_framework",
        "@fcp_maven//:junit_junit",
        "@robolectric//bazel:android-all",
    ],
)

android_local_test(
    name = "HpkePrimitiveFactoryTest",
    size = "small",
    srcs = ["HpkePrimitiveFactoryTest.java"],
    manifest_values = {
        "minSdkVersion": "30",
        "targetSdkVersion": "30",
    },
    deps = [
        TINK_PRIMITIVES,
        "@fcp_android_maven//:com_google_crypto_tink_tink_android",
        "@fcp_android_maven//:com_google_truth_truth",
        "@fcp_android_maven//:junit_junit",
        "@fcp_android_maven//:org_robolectric_annotations",
        "@fcp_android_maven//:org_robolectric_robolectric",
        "@fcp_android_maven//:org_robolectric_shadows_framework",
        "@fcp_maven//:junit_junit",
        "@robolectric//bazel:android-all",
    ],
)

android_jni_library(
    name = "crypto_test_jni",
    testonly = True,
    srcs = ["crypto_test_jni.cc"],
    deps = [
        "//fcp/confidentialcompute:cose",
        "//fcp/confidentialcompute:crypto",
        "@bazel_tools//tools/jdk:jni",
        "@boringssl//:crypto",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

android_library(
    name = "crypto_test_jni_lib",
    testonly = True,
    srcs = ["CryptoTestJni.java"],
    deps = [":crypto_test_jni"],
)

android_local_test(
    name = "HpkeWithAssociatedDataTest",
    size = "small",
    srcs = ["HpkeWithAssociatedDataTest.java"],
    data = [":libcrypto_test_jni.so"],
    jvm_flags = ["-Djava.library.path=fcp/java_src/test/java/com/google/fcp/client/confidentialcompute/internal/crypto"],
    manifest_values = {
        "minSdkVersion": "30",
        "targetSdkVersion": "30",
    },
    deps = [
        ":crypto_test_jni_lib",
        ":hpke_rfc_test_vectors",
        TINK_PRIMITIVES,
        "@fcp_android_maven//:com_google_crypto_tink_tink_android",
        "@fcp_android_maven//:com_google_guava_guava",
        "@fcp_android_maven//:com_google_truth_truth",
        "@fcp_android_maven//:junit_junit",
        "@fcp_android_maven//:org_robolectric_annotations",
        "@fcp_android_maven//:org_robolectric_robolectric",
        "@fcp_android_maven//:org_robolectric_shadows_framework",
        "@fcp_maven//:junit_junit",
        "@robolectric//bazel:android-all",
    ],
)

android_local_test(
    name = "FcpTinkConfigurationTest",
    size = "small",
    srcs = ["FcpTinkConfigurationTest.java"],
    data = [":libcrypto_test_jni.so"],
    jvm_flags = ["-Djava.library.path=fcp/java_src/test/java/com/google/fcp/client/confidentialcompute/internal/crypto"],
    manifest_values = {
        "minSdkVersion": "30",
        "targetSdkVersion": "30",
    },
    deps = [
        ":crypto_test_jni_lib",
        TINK_PRIMITIVES,
        "@fcp_android_maven//:com_google_crypto_tink_tink_android",
        "@fcp_android_maven//:com_google_truth_truth",
        "@fcp_android_maven//:junit_junit",
        "@fcp_android_maven//:org_robolectric_annotations",
        "@fcp_android_maven//:org_robolectric_robolectric",
        "@fcp_android_maven//:org_robolectric_shadows_framework",
        "@fcp_maven//:junit_junit",
        "@robolectric//bazel:android-all",
    ],
)

android_local_test(
    name = "TinkConfidentialComputeCryptoTest",
    size = "small",
    srcs = ["TinkConfidentialComputeCryptoTest.java"],
    data = [":libcrypto_test_jni.so"],
    jvm_flags = ["-Djava.library.path=fcp/java_src/test/java/com/google/fcp/client/confidentialcompute/internal/crypto"],
    manifest_values = {
        "minSdkVersion": "30",
        "targetSdkVersion": "30",
    },
    deps = [
        ":attestation_test_fixtures",
        ":crypto_test_jni_lib",
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute:api",
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute/internal/crypto:tink_crypto",
        "//fcp/protos/confidentialcompute:access_policy_endorsement_options_java_proto_lite",
        "//fcp/protos/confidentialcompute:key_java_proto_lite",
        "//fcp/protos/confidentialcompute:payload_transparency_java_proto_lite",
        "//fcp/protos/confidentialcompute:signed_endorsements_java_proto_lite",
        "//fcp/protos/confidentialcompute:verification_record_java_proto_lite",
        "//fcp/protos/federatedcompute:confidential_encryption_config_java_proto_lite",
        "@com_google_protobuf//:protobuf_javalite",
        "@fcp_android_maven//:com_google_crypto_tink_tink_android",
        "@fcp_android_maven//:com_google_truth_truth",
        "@fcp_android_maven//:junit_junit",
        "@fcp_android_maven//:org_robolectric_annotations",
        "@fcp_android_maven//:org_robolectric_robolectric",
        "@fcp_android_maven//:org_robolectric_shadows_framework",
        "@robolectric//bazel:android-all",
    ],
)

android_local_test(
    name = "RekorLogEntryVerifierTest",
    size = "small",
    srcs = ["RekorLogEntryVerifierTest.java"],
    manifest_values = {
        "minSdkVersion": "30",
        "targetSdkVersion": "30",
    },
    deps = [
        ":attestation_test_fixtures",
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute/internal/crypto:tink_crypto",
        "//fcp/protos/confidentialcompute:key_java_proto_lite",
        "//fcp/protos/confidentialcompute:payload_transparency_java_proto_lite",
        "@com_google_protobuf//:protobuf_javalite",
        "@fcp_android_maven//:com_google_guava_guava",
        "@fcp_android_maven//:com_google_truth_truth",
        "@fcp_android_maven//:junit_junit",
        "@fcp_android_maven//:org_robolectric_annotations",
        "@fcp_android_maven//:org_robolectric_robolectric",
        "@fcp_android_maven//:org_robolectric_shadows_framework",
        "@robolectric//bazel:android-all",
    ],
)
