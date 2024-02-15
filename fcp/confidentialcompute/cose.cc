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

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/cbor_utils.h"
#include "cbor.h"
#include "cbor/arrays.h"
#include "cbor/bytestrings.h"
#include "cbor/common.h"
#include "cbor/data.h"
#include "cbor/ints.h"
#include "cbor/maps.h"
#include "cbor/strings.h"

namespace fcp::confidential_compute {
namespace {

// CWT Claims; see https://www.iana.org/assignments/cwt/cwt.xhtml.
enum CwtClaim {
  kExp = 4,  // Expiration time.
  kIat = 6,  // Issued at.

  // Claims in the private space (-65537 and below).
  // See ../protos/confidentialcompute/cbor_ids.md for claims originating from
  // this project and
  // https://github.com/project-oak/oak/blob/main/oak_dice/src/cert.rs for Oak
  // claims.
  kPublicKey = -65537,       // Claim containing serialized public key.
  kOakPublicKey = -4670552,  // Oak claim containing serialized public key.
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
absl::StatusOr<CborRef> BuildCwtProtectedHeader(const OkpCwt&) {
  return SerializeCbor(*CborRef(cbor_new_definite_map(0)));
}

// Builds the payload for a CWT, which is a map of CWT claims encoded as a bstr.
// See RFC 8392 section 7.1.
absl::StatusOr<CborRef> BuildCwtPayload(const OkpCwt& cwt) {
  CborRef payload(cbor_new_definite_map(3));
  bool err = false;
  err |= cwt.expiration_time &&
         !cbor_map_add(
             payload.get(),
             {BuildCborInt(CwtClaim::kExp).get(),
              BuildCborInt(absl::ToUnixSeconds(*cwt.expiration_time)).get()});
  err |=
      cwt.issued_at &&
      !cbor_map_add(payload.get(),
                    {BuildCborInt(CwtClaim::kIat).get(),
                     BuildCborInt(absl::ToUnixSeconds(*cwt.issued_at)).get()});
  if (cwt.public_key) {
    FCP_ASSIGN_OR_RETURN(std::string encoded_public_key,
                         cwt.public_key->Encode());
    err |= !cbor_map_add(
        payload.get(),
        {BuildCborInt(CwtClaim::kPublicKey).get(),
         CborRef(cbor_build_bytestring(
                     reinterpret_cast<cbor_data>(encoded_public_key.data()),
                     encoded_public_key.size()))
             .get()});
  }
  if (err) {
    return absl::InvalidArgumentError("failed to encode CWT payload");
  }
  return SerializeCbor(*payload);
}

}  // namespace

absl::StatusOr<OkpKey> OkpKey::Decode(absl::string_view encoded) {
  cbor_load_result result;
  CborRef item(cbor_load(reinterpret_cast<cbor_data>(encoded.data()),
                         encoded.size(), &result));
  if (!item || result.read != encoded.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "failed to fully decode OkpKey (error ", result.error.code, ")"));
  }
  if (!cbor_isa_map(item.get())) {
    return absl::InvalidArgumentError("OkpKey is invalid");
  }

  // Process the parameters map.
  std::optional<uint64_t> kty;
  OkpKey okp_key;
  const cbor_pair* params = cbor_map_handle(item.get());
  const size_t params_count = cbor_map_size(item.get());
  for (int i = 0; i < params_count; ++i) {
    absl::StatusOr<int64_t> key = GetCborInt(*params[i].key);
    if (!key.ok()) continue;  // Ignore other key types.
    switch (*key) {
      case CoseKeyParameter::kKty:
        if (!cbor_isa_uint(params[i].value)) {
          return absl::InvalidArgumentError(absl::StrCat(
              "unsupported kty type ", cbor_typeof(params[i].value)));
        }
        kty = cbor_get_int(params[i].value);
        break;

      case CoseKeyParameter::kKid:
        if (!cbor_isa_bytestring(params[i].value)) {
          return absl::InvalidArgumentError(absl::StrCat(
              "unsupported kid type ", cbor_typeof(params[i].value)));
        }
        okp_key.key_id = std::string(
            reinterpret_cast<char*>(cbor_bytestring_handle(params[i].value)),
            cbor_bytestring_length(params[i].value));
        break;

      case CoseKeyParameter::kAlg: {
        if (!cbor_is_int(params[i].value)) {
          return absl::InvalidArgumentError(absl::StrCat(
              "unsupported alg type ", cbor_typeof(params[i].value)));
        }
        FCP_ASSIGN_OR_RETURN(okp_key.algorithm, GetCborInt(*params[i].value));
        break;
      }

      case CoseKeyParameter::kOkpCrv: {
        if (!cbor_is_int(params[i].value)) {
          return absl::InvalidArgumentError(absl::StrCat(
              "unsupported curve type ", cbor_typeof(params[i].value)));
        }
        FCP_ASSIGN_OR_RETURN(okp_key.curve, GetCborInt(*params[i].value));
        break;
      }

      case CoseKeyParameter::kOkpX:
        if (!cbor_isa_bytestring(params[i].value)) {
          return absl::InvalidArgumentError(absl::StrCat(
              "unsupported x type ", cbor_typeof(params[i].value)));
        }
        okp_key.x = std::string(
            reinterpret_cast<char*>(cbor_bytestring_handle(params[i].value)),
            cbor_bytestring_length(params[i].value));
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
  CborRef item(cbor_new_definite_map(5));
  if (!cbor_map_add(item.get(), {BuildCborInt(CoseKeyParameter::kKty).get(),
                                 BuildCborInt(CoseKeyType::kOkp).get()}) ||
      (!key_id.empty() &&
       !cbor_map_add(item.get(),
                     {BuildCborInt(CoseKeyParameter::kKid).get(),
                      CborRef(cbor_build_bytestring(
                                  reinterpret_cast<cbor_data>(key_id.data()),
                                  key_id.size()))
                          .get()})) ||
      (algorithm &&
       !cbor_map_add(item.get(), {BuildCborInt(CoseKeyParameter::kAlg).get(),
                                  BuildCborInt(*algorithm).get()})) ||
      (curve &&
       !cbor_map_add(item.get(), {BuildCborInt(CoseKeyParameter::kOkpCrv).get(),
                                  BuildCborInt(*curve).get()})) ||
      (!x.empty() &&
       !cbor_map_add(
           item.get(),
           {BuildCborInt(CoseKeyParameter::kOkpX).get(),
            CborRef(cbor_build_bytestring(reinterpret_cast<cbor_data>(x.data()),
                                          x.size()))
                .get()}))) {
    return absl::InvalidArgumentError("failed to encode symmetric key");
  }
  return SerializeCborToString(*item);
}

absl::StatusOr<SymmetricKey> SymmetricKey::Decode(absl::string_view encoded) {
  cbor_load_result result;
  CborRef item(cbor_load(reinterpret_cast<cbor_data>(encoded.data()),
                         encoded.size(), &result));
  if (!item || result.read != encoded.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "failed to fully decode SymmetricKey (error ", result.error.code, ")"));
  }
  if (!cbor_isa_map(item.get())) {
    return absl::InvalidArgumentError("SymmetricKey is invalid");
  }

  // Process the parameters map.
  std::optional<uint64_t> kty;
  SymmetricKey symmetric_key;
  const cbor_pair* params = cbor_map_handle(item.get());
  const size_t params_count = cbor_map_size(item.get());
  for (int i = 0; i < params_count; ++i) {
    absl::StatusOr<int64_t> key = GetCborInt(*params[i].key);
    if (!key.ok()) continue;  // Ignore other key types.
    switch (*key) {
      case CoseKeyParameter::kKty:
        if (!cbor_isa_uint(params[i].value)) {
          return absl::InvalidArgumentError(absl::StrCat(
              "unsupported kty type ", cbor_typeof(params[i].value)));
        }
        kty = cbor_get_int(params[i].value);
        break;

      case CoseKeyParameter::kAlg: {
        if (!cbor_is_int(params[i].value)) {
          return absl::InvalidArgumentError(absl::StrCat(
              "unsupported alg type ", cbor_typeof(params[i].value)));
        }
        FCP_ASSIGN_OR_RETURN(symmetric_key.algorithm,
                             GetCborInt(*params[i].value));
        break;
      }

      case CoseKeyParameter::kSymmetricK:
        if (!cbor_isa_bytestring(params[i].value)) {
          return absl::InvalidArgumentError(absl::StrCat(
              "unsupported k type ", cbor_typeof(params[i].value)));
        }
        symmetric_key.k = std::string(
            reinterpret_cast<char*>(cbor_bytestring_handle(params[i].value)),
            cbor_bytestring_length(params[i].value));
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
  CborRef item(cbor_new_definite_map(3));
  if (!cbor_map_add(item.get(),
                    {BuildCborInt(CoseKeyParameter::kKty).get(),
                     BuildCborInt(CoseKeyType::kSymmetric).get()}) ||
      (algorithm &&
       !cbor_map_add(item.get(), {BuildCborInt(CoseKeyParameter::kAlg).get(),
                                  BuildCborInt(*algorithm).get()})) ||
      (!k.empty() &&
       !cbor_map_add(
           item.get(),
           {BuildCborInt(CoseKeyParameter::kSymmetricK).get(),
            CborRef(cbor_build_bytestring(reinterpret_cast<cbor_data>(k.data()),
                                          k.size()))
                .get()}))) {
    return absl::InvalidArgumentError("failed to encode symmetric key");
  }
  return SerializeCborToString(*item);
}

absl::StatusOr<std::string> OkpCwt::BuildSigStructure(
    absl::string_view aad) const {
  // See RFC 9052 section 4.4 for the contents of the signature structure.
  CborRef signature_type(cbor_build_string("Signature1"));
  FCP_ASSIGN_OR_RETURN(CborRef protected_header,
                       BuildCwtProtectedHeader(*this));
  CborRef cbor_aad(cbor_build_bytestring(
      reinterpret_cast<cbor_data>(aad.data()), aad.size()));
  FCP_ASSIGN_OR_RETURN(CborRef payload, BuildCwtPayload(*this));

  CborRef item(cbor_new_definite_array(4));
  if (!cbor_array_push(item.get(), signature_type.get()) ||
      !cbor_array_push(item.get(), protected_header.get()) ||
      !cbor_array_push(item.get(), cbor_aad.get()) ||
      !cbor_array_push(item.get(), payload.get())) {
    return absl::InvalidArgumentError("failed to encode CWT sig structure");
  }
  return SerializeCborToString(*item);
}

absl::StatusOr<OkpCwt> OkpCwt::Decode(absl::string_view encoded) {
  cbor_load_result result;
  CborRef item(cbor_load(reinterpret_cast<cbor_data>(encoded.data()),
                         encoded.size(), &result));
  if (!item || result.read != encoded.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "failed to fully decode CWT (error ", result.error.code, ")"));
  }
  if (!cbor_isa_array(item.get()) || cbor_array_size(item.get()) != 4 ||
      !cbor_isa_bytestring(CborRef(cbor_array_get(item.get(), 0)).get()) ||
      !cbor_isa_map(CborRef(cbor_array_get(item.get(), 1)).get()) ||
      !cbor_isa_bytestring(CborRef(cbor_array_get(item.get(), 2)).get()) ||
      !cbor_isa_bytestring(CborRef(cbor_array_get(item.get(), 3)).get())) {
    return absl::InvalidArgumentError("CWT is invalid");
  }

  // Extract the signature.
  CborRef signature(cbor_array_get(item.get(), 3));
  OkpCwt cwt{
      .signature = std::string(
          reinterpret_cast<char*>(cbor_bytestring_handle(signature.get())),
          cbor_bytestring_length(signature.get())),
  };

  // Parse the payload, which is a map of CWT claims.
  CborRef serialized_payload(cbor_array_get(item.get(), 2));
  CborRef payload(cbor_load(cbor_bytestring_handle(serialized_payload.get()),
                            cbor_bytestring_length(serialized_payload.get()),
                            &result));
  if (!payload ||
      result.read != cbor_bytestring_length(serialized_payload.get())) {
    return absl::InvalidArgumentError(absl::StrCat(
        "failed to fully decode CWT payload (error ", result.error.code, ")"));
  } else if (!cbor_isa_map(payload.get())) {
    return absl::InvalidArgumentError("CWT payload is invalid");
  }

  // Process the claims map.
  const cbor_pair* claims = cbor_map_handle(payload.get());
  const size_t claim_count = cbor_map_size(payload.get());
  for (int i = 0; i < claim_count; ++i) {
    absl::StatusOr<int64_t> key = GetCborInt(*claims[i].key);
    if (!key.ok()) continue;  // Ignore other key types.
    switch (*key) {
      case CwtClaim::kExp:
        if (!cbor_isa_uint(claims[i].value)) {
          return absl::InvalidArgumentError(absl::StrCat(
              "unsupported exp type ", cbor_typeof(claims[i].value)));
        }
        cwt.expiration_time =
            absl::FromUnixSeconds(cbor_get_int(claims[i].value));
        break;

      case CwtClaim::kIat:
        if (!cbor_isa_uint(claims[i].value)) {
          return absl::InvalidArgumentError(absl::StrCat(
              "unsupported iat type ", cbor_typeof(claims[i].value)));
        }
        cwt.issued_at = absl::FromUnixSeconds(cbor_get_int(claims[i].value));
        break;

      case CwtClaim::kOakPublicKey:
      case CwtClaim::kPublicKey: {
        if (!cbor_isa_bytestring(claims[i].value)) {
          return absl::InvalidArgumentError(absl::StrCat(
              "unsupported public_key type ", cbor_typeof(claims[i].value)));
        }
        FCP_ASSIGN_OR_RETURN(cwt.public_key,
                             OkpKey::Decode(absl::string_view(
                                 reinterpret_cast<char*>(
                                     cbor_bytestring_handle(claims[i].value)),
                                 cbor_bytestring_length(claims[i].value))));
        break;
      }

      default:
        break;
    }
  }
  return cwt;
}

absl::StatusOr<std::string> OkpCwt::Encode() const {
  // See RFC 9052 section 4.2 for the contents of the COSE_Sign1 structure.
  FCP_ASSIGN_OR_RETURN(CborRef protected_header,
                       BuildCwtProtectedHeader(*this));
  CborRef unprotected_header(cbor_new_definite_map(0));
  FCP_ASSIGN_OR_RETURN(CborRef payload, BuildCwtPayload(*this));
  CborRef cbor_signature(cbor_build_bytestring(
      reinterpret_cast<cbor_data>(signature.data()), signature.size()));

  CborRef item(cbor_new_definite_array(4));
  if (!cbor_array_push(item.get(), protected_header.get()) ||
      !cbor_array_push(item.get(), unprotected_header.get()) ||
      !cbor_array_push(item.get(), payload.get()) ||
      !cbor_array_push(item.get(), cbor_signature.get())) {
    return absl::InvalidArgumentError("failed to encode CWT");
  }
  return SerializeCborToString(*item);
}

}  // namespace fcp::confidential_compute
