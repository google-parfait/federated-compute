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

#include "absl/time/time.h"
#include "fcp/base/random_token.h"

namespace fcp::client::willow {

absl::Time WillowNonceEpoch() {
  // 2026-01-01 00:00:00 UTC.
  return absl::FromUnixSeconds(1767225600);
}

std::string GenerateTimestampPrefixedNonce(absl::Time now) {
  // Generate a full 128-bit random token.
  std::string nonce = fcp::RandomToken::Generate().ToString();

  // Compute the timestamp prefix as seconds since the custom epoch.
  absl::Duration since_epoch = now - WillowNonceEpoch();
  int64_t seconds_since_epoch = absl::ToInt64Seconds(since_epoch);

  // If the time is before the epoch (e.g. due to clock skew), clamp
  // timestamp to 0. This makes it easier to identify such nonces in downstream
  // pipelines than if we were to fall back to a fully random nonce.
  if (seconds_since_epoch < 0) {
    seconds_since_epoch = 0;
  }

  // Store the timestamp as a 32-bit big-endian value in the first 4 bytes.
  // Big-endian ensures that lexicographic ordering of nonces matches temporal
  // ordering, which is required for range-based partitioning.
  static_assert(kTimestampPrefixSizeInBytes == sizeof(uint32_t),
                "kTimestampPrefixSizeInBytes must equal sizeof(uint32_t)");
  uint32_t timestamp_be = htonl(static_cast<uint32_t>(seconds_since_epoch));
  std::memcpy(nonce.data(), &timestamp_be, sizeof(timestamp_be));

  return nonce;
}

}  // namespace fcp::client::willow
