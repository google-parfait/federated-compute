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

load("@rules_android//rules:rules.bzl", "android_library")

package(
    default_applicable_licenses = ["//fcp:license"],
    default_visibility = [
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute:__subpackages__",
        "//fcp/java_src/test/java/com/google/fcp/client:__subpackages__",
    ],
    licenses = ["notice"],
)

# Custom Tink primitives of the confidential compute protocol (AES-GCM-SIV with a caller-provided
# nonce, HPKE with associated data), built on Tink's public key types and Configuration plumbing only.
android_library(
    name = "tink_primitives",
    srcs = [
        "AesGcmHpkeAead.java",
        "AesGcmSivCallerProvidedNonceAead.java",
        "CallerProvidedNonceAead.java",
        "FcpTinkConfiguration.java",
        "HkdfHpkeKdf.java",
        "HpkeAead.java",
        "HpkeContext.java",
        "HpkeEncryptWithAssociatedData.java",
        "HpkeKdf.java",
        "HpkeKem.java",
        "HpkeKemEncapOutput.java",
        "HpkePrimitiveFactory.java",
        "HpkeUtil.java",
        "HybridEncryptWithAssociatedData.java",
        "X25519HpkeKem.java",
    ],
    deps = [
        "@fcp_android_maven//:com_google_guava_guava",
        "@fcp_maven//:com_google_errorprone_error_prone_annotations",
        "@fcp_maven//:com_google_code_findbugs_jsr305",
        # Needed for AesGcmSivKey.getKeyBytes()/builder() and
        # HpkePublicKey.create()/getPublicKeyBytes() (allowlisted in
        # //third_party/tink/integration/access_lists/accesses_partial_key.bzl).
        "@fcp_android_maven//:com_google_crypto_tink_tink_android",
    ],
)

# ConfidentialComputeCrypto backed by Tink and FcpTinkConfiguration without JNI.
android_library(
    name = "tink_crypto",
    srcs = [
        "EcdsaSignatureVerifier.java",
        "RekorLogEntryVerifier.java",
        "SignedPayloadVerifier.java",
        "TinkConfidentialComputeCrypto.java",
    ],
    deps = [
        ":tink_primitives",
        "//fcp/java_src/main/java/com/google/fcp/client/confidentialcompute:api",
        "//fcp/protos/confidentialcompute:access_policy_endorsement_options_java_proto_lite",
        "//fcp/protos/confidentialcompute:key_java_proto_lite",
        "//fcp/protos/confidentialcompute:payload_transparency_java_proto_lite",
        "//fcp/protos/confidentialcompute:signed_endorsements_java_proto_lite",
        "//fcp/protos/confidentialcompute:verification_record_java_proto_lite",
        "//fcp/protos/federatedcompute:confidential_encryption_config_java_proto_lite",
        "@com_google_protobuf//:protobuf_javalite",
        "@fcp_android_maven//:com_google_crypto_tink_tink_android",
        "@fcp_android_maven//:com_google_guava_guava",
        "@fcp_maven//:com_google_errorprone_error_prone_annotations",
    ],
)
