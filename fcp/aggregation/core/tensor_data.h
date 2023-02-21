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

namespace fcp {
namespace aggregation {

// Abstract representation of tensor data storage.
//
// Tensor data is flattened one-dimensional array of tensor of tensor values
// where each values takes sizeof(T) bytes.
//
// All tensor values are stored in a single blob regardless of whether the
// tensor is dense or sparse.
//
// If the tensor is dense, then the values are flattened into
// one-dimensional array the following way:
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
// If the tensor is sparse, then the order of values in the array is arbitrary
// and can be described by the tensor SparsityParameters which describes the
// mapping from the value indices in tensor data to indices in the dense tensor
// flattened the way described above.
//
// The tensor data can be backed by different implementations depending on
// where the data comes from.
class TensorData {
 public:
  virtual ~TensorData() = default;

  // Tensor data pointer.
  virtual const void* data() const = 0;

  // The overall size of the tensor data in bytes.
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

}  // namespace aggregation
}  // namespace fcp

#endif  // FCP_AGGREGATION_CORE_TENSOR_DATA_H_
