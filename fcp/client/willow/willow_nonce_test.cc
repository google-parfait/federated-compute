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

#include "fcp/client/willow/willow_nonce.h"

#include <arpa/inet.h>

#include <cstdint>
#include <cstring>
#include <string>

#include "gtest/gtest.h"
#include "absl/time/time.h"

namespace fcp::client::willow {
namespace {

// Helper to extract the big-endian timestamp from the first 4 bytes of a nonce.
uint32_t ExtractTimestamp(const std::string& nonce) {
  uint32_t timestamp_be;
  std::memcpy(&timestamp_be, nonce.data(), sizeof(timestamp_be));
  return ntohl(timestamp_be);
}

TEST(WillowNonceEpochTest, ReturnsCorrectEpoch) {
  absl::Time epoch = WillowNonceEpoch();
  // 2026-01-01 00:00:00 UTC == Unix timestamp 1767225600.
  EXPECT_EQ(epoch, absl::FromUnixSeconds(1767225600));
}

TEST(GenerateTimestampPrefixedNonceTest, HasCorrectSize) {
  std::string nonce =
      GenerateTimestampPrefixedNonce(WillowNonceEpoch() + absl::Seconds(100));
  EXPECT_EQ(nonce.size(), kWillowNonceSizeInBytes);
}

TEST(GenerateTimestampPrefixedNonceTest, TimestampPrefixIsCorrect) {
  // Use a known time: 1000 seconds after the epoch.
  absl::Time now = WillowNonceEpoch() + absl::Seconds(1000);
  std::string nonce = GenerateTimestampPrefixedNonce(now);

  EXPECT_EQ(ExtractTimestamp(nonce), 1000);
}

TEST(GenerateTimestampPrefixedNonceTest, TimestampPrefixAtEpoch) {
  std::string nonce = GenerateTimestampPrefixedNonce(WillowNonceEpoch());

  // At the epoch, the timestamp prefix should be 0.
  EXPECT_EQ(ExtractTimestamp(nonce), 0);
}

TEST(GenerateTimestampPrefixedNonceTest, BeforeEpochUsesZeroTimestamp) {
  // Before the epoch, the nonce should use a timestamp of 0.
  absl::Time before_epoch = WillowNonceEpoch() - absl::Seconds(1);
  std::string nonce1 = GenerateTimestampPrefixedNonce(before_epoch);
  EXPECT_EQ(nonce1.size(), kWillowNonceSizeInBytes);
  EXPECT_EQ(ExtractTimestamp(nonce1), 0);

  // A second nonce generated before the epoch should also have timestamp 0,
  // but should differ in its random part.
  std::string nonce2 = GenerateTimestampPrefixedNonce(before_epoch);
  EXPECT_EQ(nonce2.size(), kWillowNonceSizeInBytes);
  EXPECT_EQ(ExtractTimestamp(nonce2), 0);
  EXPECT_NE(nonce1, nonce2);
}

TEST(GenerateTimestampPrefixedNonceTest, NoncesAreUnique) {
  absl::Time now = WillowNonceEpoch() + absl::Seconds(42);
  std::string nonce1 = GenerateTimestampPrefixedNonce(now);
  std::string nonce2 = GenerateTimestampPrefixedNonce(now);
  // Two nonces generated at the same time should differ (in the random suffix).
  EXPECT_NE(nonce1, nonce2);
}

TEST(GenerateTimestampPrefixedNonceTest, LexicographicOrderMatchesTemporal) {
  // Nonces generated at later times should be lexicographically greater.
  absl::Time t1 = WillowNonceEpoch() + absl::Seconds(100);
  absl::Time t2 = WillowNonceEpoch() + absl::Seconds(200);

  std::string nonce1 = GenerateTimestampPrefixedNonce(t1);
  std::string nonce2 = GenerateTimestampPrefixedNonce(t2);

  EXPECT_LT(nonce1, nonce2);
}

TEST(GenerateTimestampPrefixedNonceTest, LargeTimestamp) {
  // Test with a large timestamp (e.g., ~100 years after epoch, ~3.15 billion
  // seconds). This should still fit in a uint32_t (max ~4.29 billion).
  absl::Time far_future = WillowNonceEpoch() + absl::Seconds(3'150'000'000LL);
  std::string nonce = GenerateTimestampPrefixedNonce(far_future);
  EXPECT_EQ(nonce.size(), kWillowNonceSizeInBytes);

  EXPECT_EQ(ExtractTimestamp(nonce), 3'150'000'000U);
}

}  // namespace
}  // namespace fcp::client::willow
