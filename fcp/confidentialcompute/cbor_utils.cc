// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fcp/confidentialcompute/cbor_utils.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "third_party/libcbor/src/cbor/bytestrings.h"
#include "third_party/libcbor/src/cbor/common.h"
#include "third_party/libcbor/src/cbor/data.h"
#include "third_party/libcbor/src/cbor/ints.h"
#include "third_party/libcbor/src/cbor/serialization.h"

namespace fcp::confidential_compute {

absl::StatusOr<int64_t> GetCborInt(const cbor_item_t& value) {
  if (!cbor_is_int(&value)) {
    return absl::InvalidArgumentError("value is not an integer");
  }
  uint64_t v = cbor_get_int(&value);
  if (v > std::numeric_limits<int64_t>::max()) {
    return absl::InvalidArgumentError("value is out of range");
  }
  // The logical value of a negint is `-v - 1`.
  return cbor_isa_negint(&value) ? -static_cast<int64_t>(v) - 1 : v;
}

absl::StatusOr<CborRef> SerializeCbor(const cbor_item_t& item) {
  size_t size = cbor_serialized_size(&item);
  CborRef encoded(cbor_new_definite_bytestring());
  cbor_bytestring_set_handle(
      encoded.get(), reinterpret_cast<cbor_mutable_data>(malloc(size)), size);
  if (cbor_serialize(&item, cbor_bytestring_handle(encoded.get()), size) !=
      size) {
    return absl::InvalidArgumentError("serialization failed");
  }
  return encoded;
}

absl::StatusOr<std::string> SerializeCborToString(const cbor_item_t& item) {
  size_t size = cbor_serialized_size(&item);
  std::string encoded(size, '\0');
  if (cbor_serialize(&item, reinterpret_cast<cbor_mutable_data>(encoded.data()),
                     size) != size) {
    return absl::InvalidArgumentError("serialization failed");
  }
  return encoded;
}

}  // namespace fcp::confidential_compute
