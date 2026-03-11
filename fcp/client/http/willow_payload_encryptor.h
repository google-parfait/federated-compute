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

#ifndef FCP_CLIENT_HTTP_WILLOW_PAYLOAD_ENCRYPTOR_H_
#define FCP_CLIENT_HTTP_WILLOW_PAYLOAD_ENCRYPTOR_H_

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "fcp/client/federated_protocol.h"

namespace fcp::client::http {

// Interface for encrypting the payload for Willow aggregation.
class WillowPayloadEncryptor {
 public:
  virtual ~WillowPayloadEncryptor() = default;

  // Encrypts the payload for Willow aggregation and serializes the result to
  // the wire format by adding the ClientPayloadHeader.
  //
  // Args:
  //  willow_agg_info: configuration for Willow aggregation.
  //  key: serialized `secure_aggregation.willow.Key` proto.
  //  inner_payload: serialized data (FCCheckpoint) to be encrypted.
  virtual absl::StatusOr<std::string> EncryptAndSerializePayload(
      const FederatedProtocol::WillowAggInfo& willow_agg_info,
      absl::string_view key, absl::string_view inner_payload) = 0;
};

// A default implementation of WillowPayloadEncryptor that always returns an
// error.
class AlwaysFailingWillowPayloadEncryptor : public WillowPayloadEncryptor {
 public:
  absl::StatusOr<std::string> EncryptAndSerializePayload(
      const FederatedProtocol::WillowAggInfo& willow_agg_info,
      absl::string_view key, absl::string_view inner_payload) override {
    return absl::UnimplementedError("Willow aggregation is not supported.");
  }
};

// A fake implementation of WillowPayloadEncryptor that just concatenates the
// inputs, for testing purposes.
class TestingFakeWillowPayloadEncryptor : public WillowPayloadEncryptor {
 public:
  absl::StatusOr<std::string> EncryptAndSerializePayload(
      const FederatedProtocol::WillowAggInfo& willow_agg_info,
      absl::string_view key, absl::string_view inner_payload) override {
    return absl::StrFormat("%v%v%v%v", willow_agg_info.input_spec,
                           willow_agg_info.max_number_of_clients, key,
                           inner_payload);
  }
};

}  // namespace fcp::client::http

#endif  // FCP_CLIENT_HTTP_WILLOW_PAYLOAD_ENCRYPTOR_H_
