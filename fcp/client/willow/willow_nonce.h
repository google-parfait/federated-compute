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

#ifndef FCP_CLIENT_WILLOW_WILLOW_NONCE_H_
#define FCP_CLIENT_WILLOW_WILLOW_NONCE_H_

#include <string>

#include "absl/time/time.h"

namespace fcp::client::willow {

// Size of a Willow nonce in bytes (128 bits = 16 bytes).
inline constexpr int kWillowNonceSizeInBytes = 16;

// Size of the timestamp prefix in bytes (32 bits = 4 bytes).
inline constexpr int kTimestampPrefixSizeInBytes = 4;

// The custom epoch used for the timestamp prefix: 2026-01-01 00:00:00 UTC.
// Using a custom epoch avoids the 2038 Unix timestamp overflow and provides
// coverage until 2162.
absl::Time WillowNonceEpoch();

// Generates a 128-bit nonce with a timestamp prefix for Willow ciphertext
// compaction.
//
// The nonce format is:
//   [32-bit timestamp (seconds since 2026-01-01 UTC)] || [96-bit random]
//
// The timestamp prefix enables time-based partitioning of nonces, allowing the
// pipeline to compact all contributions from a completed time window with
// confidence that no future contribution will have a nonce falling into that
// window.
//
// The 96-bit random suffix provides a collision probability of approximately
// n^2 / 2^97 within any given second, which is negligible for any practical
// deployment.
//
// Args:
//   now: The current time to use for the timestamp prefix. If `now` is before
//        WillowNonceEpoch(), the timestamp will be clamped to 0.
//
// Returns a 16-byte string containing the nonce, or the same string as
// RandomToken::Generate().ToString() would return if `now` is before the
// epoch (for backward compatibility during the transition period).
std::string GenerateTimestampPrefixedNonce(absl::Time now);

}  // namespace fcp::client::willow

#endif  // FCP_CLIENT_WILLOW_WILLOW_NONCE_H_
