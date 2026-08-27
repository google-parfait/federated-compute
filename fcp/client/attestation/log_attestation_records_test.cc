#include "fcp/client/attestation/log_attestation_records.h"

#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/escaping.h"
#include "absl/strings/string_view.h"
#include "fcp/base/compression.h"
#include "fcp/client/attestation/test_values.h"
#include "fcp/protos/confidentialcompute/payload_transparency.pb.h"
#include "fcp/protos/confidentialcompute/verification_record.pb.h"
#include "fcp/protos/federatedcompute/confidential_aggregations.pb.h"
#include "fcp/testing/testing.h"

namespace fcp::client::attestation {
namespace {
using ::fcp::client::attestation::test_values::GetKnownValidEncryptionConfig;
using ::testing::_;
using ::testing::Each;
using ::testing::FieldsAre;
using ::testing::Gt;
using ::testing::Le;
using ::testing::SizeIs;

// Verifies that the LogSerializedVerificationRecord correctly chunks up and
// encodes the serialized record data.
//
// Also see `extract_attestation_records_text.cc`, which similarly validates
// that serialized records can successfully be extracted from the generated log
// lines.
TEST(LogSerializedVerificationRecordTest,
     DecodingChunkedMessagesResultsInOriginalRecord) {
  // Create a verification record with a good amount of data in it.
  confidentialcompute::AttestationVerificationRecord record;
  auto encryption_config = GetKnownValidEncryptionConfig();
  *record.mutable_encryption_key() = encryption_config.encryption_key();
  std::string huge_string;
  for (int i = 0; i < 8000; ++i) {
    huge_string += std::to_string((i * 17) % 256);
    huge_string += "some extra padding";
  }
  record.mutable_pipeline_configuration()->set_payload(huge_string);

  // Call the LogSerializedVerificationRecord function (or rather, the internal
  // variant which allows us to inspect each of the chunks it would log), which
  // is expected to emit a number of chunks.
  std::vector<std::pair<std::string, bool>> encoded_record_data;
  internal::LogSerializedVerificationRecordWith(
      record, [&encoded_record_data](absl::string_view message_chunk,
                                     bool enclose_with_brackets) {
        encoded_record_data.push_back(
            std::make_pair(std::string(message_chunk), enclose_with_brackets));
      });

  // We expect to see at least 15 log message chunks.
  EXPECT_THAT(encoded_record_data, SizeIs(Gt(15)));
  // The first message chunk is expected to contain the following unenclosed
  // string.
  EXPECT_THAT(
      encoded_record_data.front(),
      FieldsAre(
          "This device is contributing data via the confidential aggregation "
          "protocol. The attestation verification record follows.",
          false));
  // The last chunk is expected to be an empty enclosed string, unambiguously
  // indicating the end of verification record stream.
  EXPECT_THAT(encoded_record_data.back(), FieldsAre("", true));
  encoded_record_data.erase(encoded_record_data.begin());
  encoded_record_data.erase(encoded_record_data.end() - 1);

  // The chunks in between are expected to contain enclosed data. Let's verify
  // that, and extract the inner data.
  std::string base64_record_data;
  EXPECT_THAT(encoded_record_data, Each(FieldsAre(_, true)));
  for (const auto& message_chunk : encoded_record_data) {
    base64_record_data += message_chunk.first;
    EXPECT_THAT(message_chunk.first, SizeIs(Le(200)));
  }

  // Now let's base64-decode that data, and verify that it can be parsed and
  // results in the record we started with at the top of the test.
  std::string decoded_record_data;
  ASSERT_TRUE(absl::Base64Unescape(base64_record_data, &decoded_record_data));
  absl::StatusOr<absl::Cord> uncompressed_record_data =
      UncompressWithGzip(decoded_record_data);
  ABSL_ASSERT_OK(uncompressed_record_data);
  confidentialcompute::AttestationVerificationRecord decoded_record;
  ASSERT_TRUE(decoded_record.ParseFromString(*uncompressed_record_data));
  EXPECT_THAT(decoded_record, EqualsProto(record));
}

}  // namespace
}  // namespace fcp::client::attestation
