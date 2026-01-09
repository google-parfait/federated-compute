#include "fcp/confidentialcompute/crypto_test_util.h"

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/testing/testing.h"

namespace fcp::confidential_compute {
namespace {

using ::fcp::IsOk;

TEST(CryptoTestUtilTest, GenerateHpkeKeyPair) {
  auto [public_key, private_key] = GenerateHpkeKeyPair("key-id");

  absl::StatusOr<EncryptMessageResult> encrypt_result =
      MessageEncryptor().Encrypt("plaintext", public_key, "associated data");
  ASSERT_THAT(encrypt_result, IsOk());

  MessageDecryptor decryptor(std::vector<absl::string_view>{private_key});
  absl::StatusOr<std::string> decrypt_result = decryptor.Decrypt(
      encrypt_result->ciphertext, "associated data",
      encrypt_result->encrypted_symmetric_key, "associated data",
      encrypt_result->encapped_key, "key-id");
  ASSERT_THAT(decrypt_result, IsOk());
  EXPECT_EQ(*decrypt_result, "plaintext");
}

}  // namespace
}  // namespace fcp::confidential_compute
