#include "fcp/client/attestation/oak_rust_attestation_verifier.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/federatedcompute/confidential_aggregations.pb.h"
#include "fcp/testing/testing.h"
#include "third_party/oak/proto/attestation/endorsement.pb.h"
#include "third_party/oak/proto/attestation/evidence.pb.h"
#include "third_party/oak/proto/attestation/reference_value.pb.h"
#include "third_party/oak/proto/attestation/verification.pb.h"

namespace fcp::client::attestation {
namespace {
using ::fcp::confidential_compute::OkpCwt;
using ::google::internal::federatedcompute::v1::ConfidentialEncryptionConfig;
using ::oak::attestation::v1::ReferenceValues;
using ::testing::HasSubstr;

// Tests the case where default values are given to the verifier. Verification
// should not succeed in this case, since the Rust Oak Attestation Verification
// library will complain that required values are missing.
//
// This validates that we at least correctly call into the Rust Oak Attestation
// Verification library and get a result back, but it doesn't actually test
// the actual verification logic.
TEST(OakRustAttestationTest, DefaultValuesDoNotVerifySuccessfully) {
  // Generate a new public key, which we'll pass to the client in the
  // ConfidentialEncryptionConfig. We'll use the decryptor from which the public
  // key was generated to validate the encrypted payload at the end of the test.
  fcp::confidential_compute::MessageDecryptor decryptor;
  auto encoded_public_key = decryptor
                                .GetPublicKey([](absl::string_view payload) {
                                  return "fakesignature";
                                })
                                .value();
  absl::StatusOr<OkpCwt> parsed_public_key = OkpCwt::Decode(encoded_public_key);
  ASSERT_OK(parsed_public_key);
  ASSERT_TRUE(parsed_public_key->public_key.has_value());

  // Note: we don't specify any attestation evidence nor attestation
  // endorsements in the encryption config, since we can't generate valid
  // attestations in a test anyway.
  ConfidentialEncryptionConfig encryption_config;
  encryption_config.set_public_key(encoded_public_key);
  // Populate an empty Evidence proto.
  encryption_config.mutable_attestation_evidence();

  // Use an empty ReferenceValues input.
  ReferenceValues test_reference_values;
  OakRustAttestationVerifier verifier(test_reference_values, {});

  // The verification should fail, since neither reference values nor the
  // evidence are valid.
  auto result = verifier.Verify(absl::Cord(""), encryption_config);
  EXPECT_THAT(result.status(), IsCode(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(result.status().message(),
              HasSubstr("Attestation verification failed"));
}

// TODO: b/307312707 -  Add additional tests, incl. for the happy path where a
// valid policy & encryption config is provided.

}  // namespace

}  // namespace fcp::client::attestation
