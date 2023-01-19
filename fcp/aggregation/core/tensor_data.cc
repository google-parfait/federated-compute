/*
 * Copyright 2022 Google LLC
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

#include "fcp/aggregation/core/tensor_data.h"

#include "fcp/base/monitoring.h"

namespace fcp {
namespace aggregation {

bool TensorData::is_dense() const {
  if (num_slices() != 1) {
    return false;
  }
  auto slice = get_slice(0);
  return slice.byte_offset == 0 && slice.byte_size == byte_size();
}

Status TensorData::CheckValid(size_t value_size) const {
  FCP_CHECK(value_size > 0);
  if (byte_size() == 0) {
    return FCP_STATUS(FAILED_PRECONDITION)
           << "TensorData: non-empty size required";
  }

  if ((byte_size() % value_size) != 0) {
    return FCP_STATUS(FAILED_PRECONDITION)
           << "TensorData: byte_size() must be a multiple of value_size "
           << value_size;
  }

  if (num_slices() < 0) {
    return FCP_STATUS(FAILED_PRECONDITION)
           << "TensorData: negative number of slices";
  }

  size_t previous_slice_end = 0;

  for (int i = 0; i < num_slices(); i++) {
    const auto& slice = get_slice(i);

    if (slice.byte_offset < previous_slice_end) {
      return FCP_STATUS(FAILED_PRECONDITION)
             << "TensorData: slices must be ordered by byte_offset";
    }

    if ((slice.byte_offset % value_size) != 0) {
      return FCP_STATUS(FAILED_PRECONDITION)
             << "TensorData: Slice byte_offset is not aligned by value_size "
             << value_size;
    }

    if ((slice.byte_size % value_size) != 0) {
      return FCP_STATUS(FAILED_PRECONDITION)
             << "TensorData: Slice byte_size is not aligned by value_size "
             << value_size;
    }

    if ((reinterpret_cast<size_t>(slice.data) % value_size) != 0) {
      return FCP_STATUS(FAILED_PRECONDITION)
             << "TensorData: Slice data address is not aligned by value_size "
             << value_size;
    }

    // Check the byte size isn't zero and that byte_size isn't so large that
    // it causes an overflow.
    if (slice.byte_offset + slice.byte_size <= slice.byte_offset) {
      return FCP_STATUS(FAILED_PRECONDITION)
             << "TensorData: Invalid slice byte_size " << slice.byte_size;
    }

    if (slice.byte_offset + slice.byte_size > byte_size()) {
      return FCP_STATUS(FAILED_PRECONDITION)
             << "TensorData: Slice ends after the overall byte_size()";
    }

    previous_slice_end = slice.byte_offset + slice.byte_size;
  }

  return FCP_STATUS(OK);
}

}  // namespace aggregation
}  // namespace fcp
