// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fcp/confidentialcompute/cose.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "fcp/base/monitoring.h"
#include "libcppbor/include/cppbor/cppbor.h"
#include "libcppbor/include/cppbor/cppbor_parse.h"

namespace fcp::confidential_compute {
namespace {

using ::cppbor::Array;
using ::cppbor::Bstr;
using ::cppbor::Map;

// CWT Claims; see https://www.iana.org/assignments/cwt/cwt.xhtml.
enum CwtClaim {
  kExp = 4,  // Expiration time.
  kIat = 6,  // Issued at.

  // Claims in the private space (-65537 and below).
  // See ../protos/confidentialcompute/cbor_ids.md for claims originating from
  // this project and
  // https://github.com/project-oak/oak/blob/main/oak_dice/src/cert.rs for Oak
  // claims.
  kPublicKey = -65537,         // Claim containing serialized public key.
  kConfigProperties = -65538,  // Claim containing configuration properties.
  kOakPublicKey = -4670552,    // Oak claim containing serialized public key.
};

// COSE Key parameters; see https://www.iana.org/assignments/cose/cose.xhtml.
enum CoseKeyParameter {
  // Common parameters.
  kKty = 1,
  kKid = 2,
  kAlg = 3,

  // OKP parameters.
  kOkpCrv = -1,
  kOkpX = -2,

  // Symmetric parameters.
  kSymmetricK = -1,
};

// COSE Key types; see https://www.iana.org/assignments/cose/cose.xhtml.
enum CoseKeyType {
  kOkp = 1,
  kSymmetric = 4,
};

// Builds the protected header for a CWT, which is simply an empty map encoded
// as a bstr.
absl::StatusOr<std::vector<uint8_t>> BuildCwtProtectedHeader(const OkpCwt&) {
  return Map().encode();
}

// Builds the payload for a CWT, which is a map of CWT claims encoded as a bstr.
// See RFC 8392 section 7.1.
absl::StatusOr<std::vector<uint8_t>> BuildCwtPayload(const OkpCwt& cwt) {
  Map map;
  if (cwt.expiration_time) {
    map.add(CwtClaim::kExp, absl::ToUnixSeconds(*cwt.expiration_time));
  }
  if (cwt.issued_at) {
    map.add(CwtClaim::kIat, absl::ToUnixSeconds(*cwt.issued_at));
  }
  if (cwt.public_key) {
    FCP_ASSIGN_OR_RETURN(std::string encoded_public_key,
                         cwt.public_key->Encode());
    map.add(CwtClaim::kPublicKey, Bstr(encoded_public_key));
  }
  if (!cwt.config_properties.fields().empty()) {
    map.add(CwtClaim::kConfigProperties,
            Bstr(cwt.config_properties.SerializeAsString()));
  }
  return map.encode();
}

}  // namespace

absl::StatusOr<OkpKey> OkpKey::Decode(absl::string_view encoded) {
  auto [item, end_pos, error] = cppbor::parse(
      reinterpret_cast<const uint8_t*>(encoded.data()), encoded.size());
  if (!error.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("failed to decode OkpKey: ", error));
  } else if (end_pos != reinterpret_cast<const uint8_t*>(encoded.data()) +
                            encoded.size()) {
    return absl::InvalidArgumentError(
        "failed to decode OkpKey: input contained extra data");
  } else if (item->type() != cppbor::MAP) {
    return absl::InvalidArgumentError("OkpKey is invalid");
  }

  // Process the parameters map.
  std::optional<uint64_t> kty;
  OkpKey okp_key;
  for (const auto& [key, value] : *item->asMap()) {
    if (key->asInt() == nullptr) continue;  // Ignore other key types.
    switch (key->asInt()->value()) {
      case CoseKeyParameter::kKty:
        if (value->asInt() == nullptr) {
          return absl::InvalidArgumentError(
              absl::StrCat("unsupported kty type ", value->type()));
        }
        kty = value->asInt()->value();
        break;

      case CoseKeyParameter::kKid:
        if (value->type() != cppbor::BSTR) {
          return absl::InvalidArgumentError(
              absl::StrCat("unsupported kid type ", value->type()));
        }
        okp_key.key_id.assign(value->asBstr()->value().begin(),
                              value->asBstr()->value().end());
        break;

      case CoseKeyParameter::kAlg:
        if (value->asInt() == nullptr) {
          return absl::InvalidArgumentError(
              absl::StrCat("unsupported alg type ", value->type()));
        }
        okp_key.algorithm = value->asInt()->value();
        break;

      case CoseKeyParameter::kOkpCrv:
        if (value->asInt() == nullptr) {
          return absl::InvalidArgumentError(
              absl::StrCat("unsupported curve type ", value->type()));
        }
        okp_key.curve = value->asInt()->value();
        break;

      case CoseKeyParameter::kOkpX:
        if (value->type() != cppbor::BSTR) {
          return absl::InvalidArgumentError(
              absl::StrCat("unsupported x type ", value->type()));
        }
        okp_key.x.assign(value->asBstr()->value().begin(),
                         value->asBstr()->value().end());
        break;

      default:
        break;
    }
  }

  if (!kty.has_value() || *kty != CoseKeyType::kOkp) {
    return absl::InvalidArgumentError("missing or wrong Cose_Key type");
  }
  return okp_key;
}

absl::StatusOr<std::string> OkpKey::Encode() const {
  // Generate a map containing the parameters that are set.
  Map map;
  map.add(CoseKeyParameter::kKty, CoseKeyType::kOkp);
  if (!key_id.empty()) {
    map.add(CoseKeyParameter::kKid, Bstr(key_id));
  }
  if (algorithm) {
    map.add(CoseKeyParameter::kAlg, *algorithm);
  }
  if (curve) {
    map.add(CoseKeyParameter::kOkpCrv, *curve);
  }
  if (!x.empty()) {
    map.add(CoseKeyParameter::kOkpX, Bstr(x));
  }
  return map.toString();
}

absl::StatusOr<SymmetricKey> SymmetricKey::Decode(absl::string_view encoded) {
  auto [item, end_pos, error] = cppbor::parse(
      reinterpret_cast<const uint8_t*>(encoded.data()), encoded.size());
  if (!error.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("failed to decode SymmetricKey: ", error));
  } else if (end_pos != reinterpret_cast<const uint8_t*>(encoded.data()) +
                            encoded.size()) {
    return absl::InvalidArgumentError(
        "failed to decode SymmetricKey: input contained extra data");
  } else if (item->type() != cppbor::MAP) {
    return absl::InvalidArgumentError("SymmetricKey is invalid");
  }

  // Process the parameters map.
  std::optional<uint64_t> kty;
  SymmetricKey symmetric_key;
  for (const auto& [key, value] : *item->asMap()) {
    if (key->asInt() == nullptr) continue;  // Ignore other key types.
    switch (key->asInt()->value()) {
      case CoseKeyParameter::kKty:
        if (value->asInt() == nullptr) {
          return absl::InvalidArgumentError(
              absl::StrCat("unsupported kty type ", value->type()));
        }
        kty = value->asInt()->value();
        break;

      case CoseKeyParameter::kAlg:
        if (value->asInt() == nullptr) {
          return absl::InvalidArgumentError(
              absl::StrCat("unsupported alg type ", value->type()));
        }
        symmetric_key.algorithm = value->asInt()->value();
        break;

      case CoseKeyParameter::kSymmetricK:
        if (value->type() != cppbor::BSTR) {
          return absl::InvalidArgumentError(
              absl::StrCat("unsupported k type ", value->type()));
        }
        symmetric_key.k.assign(value->asBstr()->value().begin(),
                               value->asBstr()->value().end());
        break;

      default:
        break;
    }
  }

  if (!kty.has_value() || *kty != CoseKeyType::kSymmetric) {
    return absl::InvalidArgumentError("missing or wrong Cose_Key type");
  }
  return symmetric_key;
}

absl::StatusOr<std::string> SymmetricKey::Encode() const {
  // Generate a map containing the parameters that are set.
  Map map;
  map.add(CoseKeyParameter::kKty, CoseKeyType::kSymmetric);
  if (algorithm) {
    map.add(CoseKeyParameter::kAlg, *algorithm);
  }
  if (!k.empty()) {
    map.add(CoseKeyParameter::kSymmetricK, Bstr(k));
  }
  return map.toString();
}

absl::StatusOr<std::string> OkpCwt::BuildSigStructure(
    absl::string_view aad) const {
  // See RFC 9052 section 4.4 for the contents of the signature structure.
  FCP_ASSIGN_OR_RETURN(std::vector<uint8_t> protected_header,
                       BuildCwtProtectedHeader(*this));
  FCP_ASSIGN_OR_RETURN(std::vector<uint8_t> payload, BuildCwtPayload(*this));
  return Array("Signature1", std::move(protected_header),
               Bstr(aad.begin(), aad.end()), std::move(payload))
      .toString();
}

absl::StatusOr<OkpCwt> OkpCwt::Decode(absl::string_view encoded) {
  auto [item, end_pos, error] = cppbor::parse(
      reinterpret_cast<const uint8_t*>(encoded.data()), encoded.size());
  if (!error.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("failed to decode CWT: ", error));
  } else if (end_pos != reinterpret_cast<const uint8_t*>(encoded.data()) +
                            encoded.size()) {
    return absl::InvalidArgumentError(
        "failed to decode CWT: input contained extra data");
  } else if (auto array = item->asArray();
             array == nullptr || array->size() != 4 ||
             array->get(0)->type() != cppbor::BSTR ||
             array->get(1)->type() != cppbor::MAP ||
             array->get(2)->type() != cppbor::BSTR ||
             array->get(3)->type() != cppbor::BSTR) {
    return absl::InvalidArgumentError("CWT is invalid");
  }

  // Extract the signature.
  OkpCwt cwt{
      .signature =
          std::string(item->asArray()->get(3)->asBstr()->value().begin(),
                      item->asArray()->get(3)->asBstr()->value().end()),
  };

  // Parse the payload, which is a map of CWT claims.
  const std::vector<uint8_t>& serialized_payload =
      item->asArray()->get(2)->asBstr()->value();
  std::unique_ptr<cppbor::Item> payload;
  std::tie(payload, end_pos, error) =
      cppbor::parse(serialized_payload.data(), serialized_payload.size());
  if (!error.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("failed to decode CWT payload: ", error));
  } else if (end_pos != serialized_payload.data() + serialized_payload.size()) {
    return absl::InvalidArgumentError(
        "failed to decode CWT payload: input contained extra data");
  } else if (payload->type() != cppbor::MAP) {
    return absl::InvalidArgumentError("CWT payload is invalid");
  }

  // Process the claims map.
  for (const auto& [key, value] : *payload->asMap()) {
    if (key->asInt() == nullptr) continue;  // Ignore other key types.
    switch (key->asInt()->value()) {
      case CwtClaim::kExp:
        if (value->asInt() == nullptr) {
          return absl::InvalidArgumentError(
              absl::StrCat("unsupported exp type ", value->type()));
        }
        cwt.expiration_time = absl::FromUnixSeconds(value->asInt()->value());
        break;

      case CwtClaim::kIat:
        if (value->asInt() == nullptr) {
          return absl::InvalidArgumentError(
              absl::StrCat("unsupported iat type ", value->type()));
        }
        cwt.issued_at = absl::FromUnixSeconds(value->asInt()->value());
        break;

      case CwtClaim::kOakPublicKey:
      case CwtClaim::kPublicKey: {
        if (value->type() != cppbor::BSTR) {
          return absl::InvalidArgumentError(
              absl::StrCat("unsupported public_key type ", value->type()));
        }
        FCP_ASSIGN_OR_RETURN(
            cwt.public_key,
            OkpKey::Decode(absl::string_view(
                reinterpret_cast<const char*>(value->asBstr()->value().data()),
                value->asBstr()->value().size())));
        break;
      }

      case CwtClaim::kConfigProperties:
        if (value->type() != cppbor::BSTR) {
          return absl::InvalidArgumentError(
              absl::StrCat("unsupported configuration type ", value->type()));
        }
        if (!cwt.config_properties.ParseFromArray(
                value->asBstr()->value().data(),
                static_cast<int>(value->asBstr()->value().size()))) {
          return absl::InvalidArgumentError("failed to parse configuration");
        }
        break;

      default:
        break;
    }
  }
  return cwt;
}

absl::StatusOr<std::string> OkpCwt::Encode() const {
  // See RFC 9052 section 4.2 for the contents of the COSE_Sign1 structure.
  FCP_ASSIGN_OR_RETURN(std::vector<uint8_t> protected_header,
                       BuildCwtProtectedHeader(*this));
  Map unprotected_header;
  FCP_ASSIGN_OR_RETURN(std::vector<uint8_t> payload, BuildCwtPayload(*this));
  return Array(std::move(protected_header), std::move(unprotected_header),
               std::move(payload), Bstr(signature))
      .toString();
}

}  // namespace fcp::confidential_compute
