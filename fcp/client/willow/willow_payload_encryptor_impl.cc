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

#include "fcp/client/willow/willow_payload_encryptor_impl.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "fcp/base/monitoring.h"
#include "fcp/base/random_token.h"
#include "fcp/client/federated_protocol.h"
#include "fcp/confidentialcompute/client_payload.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "third_party/secure_aggregation/willow/api/client.h"
#include "third_party/secure_aggregation/willow/input_encoding/codec.h"
#include "third_party/secure_aggregation/willow/input_encoding/codec_factory.h"
#include "third_party/secure_aggregation/willow/proto/shell/ciphertexts.pb.h"
#include "third_party/secure_aggregation/willow/proto/willow/aggregation_config.pb.h"
#include "third_party/secure_aggregation/willow/proto/willow/input_spec.pb.h"
#include "third_party/secure_aggregation/willow/proto/willow/key.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"

namespace fcp::client::willow {

using ::tensorflow_federated::aggregation::CheckpointParser;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;

const int64_t kFixedMaxDecryptors = 1;
const int64_t kFixedMaxDecryptorDropouts = 0;

absl::StatusOr<std::string>
WillowPayloadEncryptorImpl::EncryptAndSerializePayload(
    const FederatedProtocol::WillowAggInfo& willow_agg_info,
    absl::string_view key, absl::string_view inner_payload) {
  secure_aggregation::willow::InputSpec input_spec_proto;
  if (!input_spec_proto.ParseFromString(willow_agg_info.input_spec)) {
    return absl::InvalidArgumentError(
        "Could not parse InputSpec from input_spec");
  }

  // TODO: b/478268416 - [Willow] Update C++ client API to skip
  // serialization/deserialization.
  secure_aggregation::willow::Key key_proto;
  if (!key_proto.ParseFromString(key)) {
    return absl::InvalidArgumentError("Could not parse key proto");
  }
  secure_aggregation::willow::ShellAhePublicKey public_key;
  if (!public_key.ParseFromString(key_proto.key_material())) {
    return absl::InvalidArgumentError("Could not parse public key");
  }
  std::string key_id = key_proto.key_id();

  // Generate the nonce and use it as blob_id.
  std::string nonce = fcp::RandomToken::Generate().ToString();
  fcp::confidentialcompute::BlobHeader blob_header;
  blob_header.set_blob_id(nonce);
  blob_header.set_key_id(key_id);

  secure_aggregation::willow::AggregationConfigProto config_proto;
  config_proto.set_max_number_of_clients(willow_agg_info.max_number_of_clients);
  config_proto.set_max_number_of_decryptors(kFixedMaxDecryptors);
  config_proto.set_max_decryptor_dropouts(kFixedMaxDecryptorDropouts);
  config_proto.set_key_id(key_id);

  // Build vector_configs from input_spec.
  int64_t length = 1;
  for (const auto& spec : input_spec_proto.group_by_vector_specs()) {
    length *= spec.domain_spec().string_values().values_size();
  }
  for (const auto& spec : input_spec_proto.metric_vector_specs()) {
    auto& config = (*config_proto.mutable_vector_configs())[spec.vector_name()];
    config.set_length(length);
    if (spec.has_domain_spec() && spec.domain_spec().has_interval()) {
      config.set_bound(
          static_cast<int64_t>(spec.domain_spec().interval().max()));
    } else {
      config.set_bound(1LL << 30);
    }
  }

  // Willow encoded data to be populated from the inner_payload.
  absl::flat_hash_map<std::string, std::vector<std::string>> group_by_data;
  absl::flat_hash_map<std::string, std::vector<int64_t>> metric_data;

  // Temporary solution to integrate Willow with minimal changes in
  // the existing code:
  //    Copy to Cord and deserialize the inner_payload, which comes from a
  //    CopyCordToString of a serialized ExampleQueryResult in
  //    ReportViaOneShotAggregation.
  FederatedComputeCheckpointParserFactory parser_factory;
  absl::Cord inner_payload_cord(inner_payload);
  FCP_ASSIGN_OR_RETURN(std::unique_ptr<CheckpointParser> parser,
                       parser_factory.Create(inner_payload_cord));

  // Use the names in InputSpec to retrieve the tensors from the parser.
  for (const auto& spec : input_spec_proto.group_by_vector_specs()) {
    FCP_ASSIGN_OR_RETURN(auto tensor, parser->GetTensor(spec.vector_name()));
    group_by_data[spec.vector_name()] = tensor.ToStringVector();
  }
  for (const auto& spec : input_spec_proto.metric_vector_specs()) {
    FCP_ASSIGN_OR_RETURN(auto tensor, parser->GetTensor(spec.vector_name()));
    if (tensor.dtype() != tensorflow_federated::aggregation::DT_INT64) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported metric data type for vector: ", spec.vector_name()));
    }
    auto span = tensor.AsSpan<int64_t>();
    metric_data[spec.vector_name()].assign(span.begin(), span.end());
  }

  FCP_ASSIGN_OR_RETURN(
      std::unique_ptr<secure_aggregation::willow::Codec> encoder,
      secure_aggregation::willow::CodecFactory::CreateExplicitCodec(
          input_spec_proto, willow_agg_info.max_flattened_domain_size));
  FCP_ASSIGN_OR_RETURN(secure_aggregation::willow::EncodedData encoded_data,
                       encoder->Encode(group_by_data, metric_data));

  // Generate client contribution, encrypted towards public key with
  // client-generated nonce.
  FCP_ASSIGN_OR_RETURN(auto client_message,
                       secure_aggregation::GenerateClientContribution(
                           config_proto, encoded_data, public_key, nonce));

  // Wrap encrypted client message with ClientPayloadHeader.
  fcp::confidential_compute::ClientPayloadHeader payload_header;
  payload_header.is_gzip_compressed = false;
  payload_header.serialized_blob_header = blob_header.SerializeAsString();
  return fcp::confidential_compute::EncodeClientPayload(
      std::move(payload_header), client_message.SerializeAsString());
}

}  // namespace fcp::client::willow
