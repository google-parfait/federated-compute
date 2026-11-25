/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Test-only JNI bindings wrapping fcp/confidentialcompute/crypto.cc decryption
// functions so Java encryption unit tests can verify compatibility against the
// native C++ recipient implementation.

#include <jni.h>

#include <cstdint>
#include <string>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/confidentialcompute/crypto.h"
#include "openssl/hpke.h"

namespace {

using ::fcp::confidential_compute::kCoseKeyOpDecrypt;
using ::fcp::confidential_compute::MessageDecryptor;
using ::fcp::confidential_compute::OkpKey;
using ::fcp::confidential_compute::crypto_internal::
    kHpkeBaseX25519Sha256Aes128Gcm;
using ::fcp::confidential_compute::crypto_internal::kX25519;
using ::fcp::confidential_compute::crypto_internal::UnwrapSymmetricKey;

std::string ByteArrayToString(JNIEnv* env, jbyteArray array) {
  jsize length = env->GetArrayLength(array);
  std::string out(length, '\0');
  env->GetByteArrayRegion(array, 0, length,
                          reinterpret_cast<jbyte*>(out.data()));
  return out;
}

jbyteArray StringToByteArray(JNIEnv* env, absl::string_view value) {
  jsize length = static_cast<jsize>(value.size());
  jbyteArray out = env->NewByteArray(length);
  if (out == nullptr) {
    return nullptr;
  }
  env->SetByteArrayRegion(out, 0, length,
                          reinterpret_cast<const jbyte*>(value.data()));
  return out;
}

void ThrowGeneralSecurityException(JNIEnv* env, const absl::Status& status) {
  jclass cls = env->FindClass("java/security/GeneralSecurityException");
  if (cls == nullptr) {
    return;
  }
  env->ThrowNew(cls, status.ToString().c_str());
  env->DeleteLocalRef(cls);
}

}  // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_com_google_fcp_client_confidentialcompute_internal_crypto_CryptoTestJni_nativeIsCborSupported(
    JNIEnv*, jclass) {
#ifdef FCP_SUPPORT_CBOR
  return JNI_TRUE;
#else
  return JNI_FALSE;
#endif
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_google_fcp_client_confidentialcompute_internal_crypto_CryptoTestJni_nativeUnwrapSymmetricKey(
    JNIEnv* env, jclass, jbyteArray recipient_private_key,
    jbyteArray encapped_key, jbyteArray encrypted_symmetric_key,
    jbyteArray associated_data) {
  std::string private_key_bytes = ByteArrayToString(env, recipient_private_key);
  bssl::ScopedEVP_HPKE_KEY hpke_key;
  ABSL_CHECK_EQ(EVP_HPKE_KEY_init(
                    hpke_key.get(), EVP_hpke_x25519_hkdf_sha256(),
                    reinterpret_cast<const uint8_t*>(private_key_bytes.data()),
                    private_key_bytes.size()),
                1);
  absl::StatusOr<std::string> plaintext = UnwrapSymmetricKey(
      hpke_key.get(), EVP_hpke_hkdf_sha256(), EVP_hpke_aes_128_gcm(),
      ByteArrayToString(env, encrypted_symmetric_key),
      ByteArrayToString(env, encapped_key),
      ByteArrayToString(env, associated_data));
  if (!plaintext.ok()) {
    ThrowGeneralSecurityException(env, plaintext.status());
    return nullptr;
  }
  return StringToByteArray(env, *plaintext);
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_google_fcp_client_confidentialcompute_internal_crypto_CryptoTestJni_nativeDecryptMessage(
    JNIEnv* env, jclass, jbyteArray recipient_private_key, jbyteArray key_id,
    jbyteArray ciphertext, jbyteArray ciphertext_associated_data,
    jbyteArray encrypted_symmetric_key,
    jbyteArray encrypted_symmetric_key_associated_data,
    jbyteArray encapped_key) {
  std::string key_id_str = ByteArrayToString(env, key_id);
  absl::StatusOr<std::string> cose_private_key =
      OkpKey{
          .key_id = key_id_str,
          .algorithm = kHpkeBaseX25519Sha256Aes128Gcm,
          .key_ops = {kCoseKeyOpDecrypt},
          .curve = kX25519,
          .d = ByteArrayToString(env, recipient_private_key),
      }
          .Encode();
  ABSL_CHECK_OK(cose_private_key);
  MessageDecryptor decryptor(std::vector<absl::string_view>{*cose_private_key});
  absl::StatusOr<std::string> plaintext = decryptor.Decrypt(
      ByteArrayToString(env, ciphertext),
      ByteArrayToString(env, ciphertext_associated_data),
      ByteArrayToString(env, encrypted_symmetric_key),
      ByteArrayToString(env, encrypted_symmetric_key_associated_data),
      ByteArrayToString(env, encapped_key), key_id_str);
  if (!plaintext.ok()) {
    ThrowGeneralSecurityException(env, plaintext.status());
    return nullptr;
  }
  return StringToByteArray(env, *plaintext);
}
