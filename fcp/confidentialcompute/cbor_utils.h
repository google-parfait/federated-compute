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

// This file contains helper functions for working with CBOR (RFC 8949)
// structures.

#ifndef FCP_CONFIDENTIALCOMPUTE_CBOR_UTILS_H_
#define FCP_CONFIDENTIALCOMPUTE_CBOR_UTILS_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "third_party/libcbor/src/cbor/common.h"
#include "third_party/libcbor/src/cbor/data.h"
#include "third_party/libcbor/src/cbor/ints.h"

namespace fcp::confidential_compute {
namespace cbor_utils_internal {
struct CborDecref;
}  // namespace cbor_utils_internal

// RAII type for a cbor_item_t that decrements the refcount when it goes out of
// scope.
using CborRef = std::unique_ptr<cbor_item_t, cbor_utils_internal::CborDecref>;

// Encodes an int into a minimally-sized cbor_item_t (e.g., a value in [0, 255]
// is converted to a uint8).
inline CborRef BuildCborInt(int64_t value);

// Converts an item to a signed integer, returning an error if the item isn't an
// integer or if it's outside the range of int64_t.
absl::StatusOr<int64_t> GetCborInt(const cbor_item_t& value);

// Serializes a cbor_item_t to a bytestring cbor_item_t.
absl::StatusOr<CborRef> SerializeCbor(const cbor_item_t& item);

// Serializes a cbor_item_t to a std::string.
absl::StatusOr<std::string> SerializeCborToString(const cbor_item_t& item);

// --------- Implementation follows ---------

struct cbor_utils_internal::CborDecref {
  void operator()(cbor_item_t* item) const {
    if (item) cbor_intermediate_decref(item);
  }
};

CborRef BuildCborInt(int64_t value) {
  if (value >= 0x100000000) return CborRef(cbor_build_uint64(value));
  if (value >= 0x10000) return CborRef(cbor_build_uint32(value));
  if (value >= 0x100) return CborRef(cbor_build_uint16(value));
  if (value >= 0) return CborRef(cbor_build_uint8(value));
  if (value >= -0x100) return CborRef(cbor_build_negint8(-value - 1));
  if (value >= -0x10000) return CborRef(cbor_build_negint16(-value - 1));
  if (value >= -0x100000000) return CborRef(cbor_build_negint32(-value - 1));
  return CborRef(cbor_build_negint64(-value - 1));
}

}  // namespace fcp::confidential_compute

#endif  // FCP_CONFIDENTIALCOMPUTE_CBOR_UTILS_H_
