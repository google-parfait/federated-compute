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

#ifndef FCP_CLIENT_WILLOW_WILLOW_PAYLOAD_ENCRYPTOR_IMPL_H_
#define FCP_CLIENT_WILLOW_WILLOW_PAYLOAD_ENCRYPTOR_IMPL_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "fcp/client/federated_protocol.h"
#include "fcp/client/willow/willow_payload_encryptor.h"

namespace fcp::client::willow {

// Implementation of WillowPayloadEncryptor that uses the Willow library.
class WillowPayloadEncryptorImpl : public WillowPayloadEncryptor {
 public:
  absl::StatusOr<std::string> EncryptAndSerializePayload(
      const FederatedProtocol::WillowAggInfo& willow_agg_info,
      absl::string_view key, absl::string_view inner_payload) override;
};

}  // namespace fcp::client::willow

#endif  // FCP_CLIENT_WILLOW_WILLOW_PAYLOAD_ENCRYPTOR_IMPL_H_
