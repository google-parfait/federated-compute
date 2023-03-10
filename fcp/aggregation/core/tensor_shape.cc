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

#include "fcp/aggregation/core/tensor_shape.h"

#include <utility>

#include "fcp/base/monitoring.h"

#ifndef FCP_NANOLIBC
#include "fcp/aggregation/core/tensor.pb.h"
#endif

namespace fcp {
namespace aggregation {

size_t TensorShape::NumElements() const {
  size_t num_elements = 1;
  for (auto dim_size : dim_sizes_) {
    num_elements *= dim_size;
  }
  return num_elements;
}

#ifndef FCP_NANOLIBC

StatusOr<TensorShape> TensorShape::FromProto(
    const TensorShapeProto& shape_proto) {
  TensorShape::DimSizesVector dim_sizes;
  for (int64_t dim_size : shape_proto.dim_sizes()) {
    if (dim_size < 0) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "Negative dimension size isn't supported when converting from "
             << "shape_proto: " << shape_proto.ShortDebugString();
    }
    dim_sizes.push_back(dim_size);
  }
  return TensorShape(std::move(dim_sizes));
}

TensorShapeProto TensorShape::ToProto() const {
  TensorShapeProto shape_proto;
  for (auto dim_size : dim_sizes()) {
    shape_proto.add_dim_sizes(static_cast<int64_t>(dim_size));
  }
  return shape_proto;
}

#endif  // FCP_NANOLIBC

}  // namespace aggregation
}  // namespace fcp
