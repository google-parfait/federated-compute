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

#ifndef FCP_AGGREGATION_CORE_TENSOR_DATA_H_
#define FCP_AGGREGATION_CORE_TENSOR_DATA_H_

#include <cstddef>

#include "fcp/base/monitoring.h"

namespace fcp::aggregation {

// Abstract representation of tensor data storage.
//
// The tensor data is represented as a collection of slices of binary data
// containing the spans of flattened raw tensor data with the gaps between
// the spans assumed to be "zero" values.
//
// If we consider the entire space of TensorData for any particular Tensor,
// then the Tensor values would be flattened the following way:
// - First iterating over the last dimension
// - Then incrementing the second from the last dimension and then iterating
//   over the last dimension
// - Then gradually moving towards the first dimension.
// For example, if we had a 3-dimensional {3 x 2 x 4} Tensor, the values
// in TensorData would be ordered in the following way, showing 3-dimensional
// indices of the tensor values:
//   (0,0,0), (0,0,1), (0,0,2), (0,0,3)
//   (0,1,0), (0,1,1), (0,1,2), (0,1,3)
//   (1,0,0), (1,0,1), (1,0,2), (1,0,3)
//   (1,1,0), (1,1,1), (1,1,2), (1,1,3)
//   (2,0,0), (2,0,1), (2,0,2), (2,0,3)
//   (2,1,0), (2,1,1), (2,1,2), (2,1,3)
//
// Tensor slices are expected to be non-overlapping and ordered by byte_offset.
//
// The tensor data isn't type specific at this level. It is up to the upstream
// class how to interpret the data. In particular a different number of bytes
// may be stored per element, depending on the tensor value type.
//
// The tensor data can be backed by different implementations depending on
// where the data comes from.
class TensorData {
 public:
  virtual ~TensorData() = default;

  // A single slice of tensor data.
  struct Slice {
    // Byte offset of the slice within the tensor data space.
    size_t byte_offset;
    // Byte size of the slice.
    size_t byte_size;
    // The slice data.
    const void* data;
  };

  // Gets the number of slices.
  virtual int num_slices() const = 0;

  // Gets the Nth slice - read-only access.
  virtual Slice get_slice(int n) const = 0;

  // The overall size of the flattened tensor data in bytes.
  // This value must match the product of sizes of all tensor dimensions
  // multiplied by the tensor value size. This defines a "virtual" space size
  // where all slices must fit into, meaning that for any given Slice,
  // its byte_offset + byte_size must not exceed this value.
  virtual size_t byte_size() const = 0;

  // Validates TensorData constraints given the specified value_size.
  // The value_size is the size of the native data type (e.g. 4 bytes for int32
  // or float, 8 bytes for int64). This is used to verify data alignment - that
  // all offsets and sizes are multiples of value_size that pointers are memory
  // aligned to the value_size.
  // TODO(team): Consider separate sizes for the pointer alignment and
  // the slices offsets/sizes. The latter may need to be more coarse.
  Status CheckValid(size_t value_size) const;
};

}  // namespace fcp::aggregation

#endif  // FCP_AGGREGATION_CORE_TENSOR_DATA_H_
