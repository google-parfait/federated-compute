/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "fcp/client/rust/oak_attestation_verification_ffi.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "fcp/testing/testing.h"
#include "proto/attestation/endorsement.pb.h"
#include "proto/attestation/reference_value.pb.h"

namespace fcp::client::rust::oak_attestation_verification_ffi {
namespace {

TEST(OakAttestationVerificationFfiTest, VerifyEndorsementFailure) {
  // Error message is: "no endorsement in signed endorsement", as expected.
  EXPECT_THAT(VerifyEndorsement(absl::UniversalEpoch(), {}, {}),
              IsCode(absl::StatusCode::kFailedPrecondition));
}

}  // namespace
}  // namespace fcp::client::rust::oak_attestation_verification_ffi
