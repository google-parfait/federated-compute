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

#ifndef FCP_AGGREGATION_CORE_AGG_VECTOR_AGGREGATOR_H_
#define FCP_AGGREGATION_CORE_AGG_VECTOR_AGGREGATOR_H_

#include <memory>
#include <vector>

#include "fcp/aggregation/core/tensor_aggregator.h"
#include "fcp/aggregation/core/tensor_data.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/aggregation/core/vector_data.h"
#include "fcp/base/monitoring.h"

namespace fcp {
namespace aggregation {

// AggVectorAggregator class is a specialization of TensorAggregator which
// operates on AggVector<T> instances rather than tensors.
template <typename T>
class AggVectorAggregator : public TensorAggregator {
 public:
  AggVectorAggregator(DataType dtype, TensorShape shape)
      : AggVectorAggregator(dtype, shape,
                            new VectorData<T>(shape.NumElements())) {}

  // Provides mutable access to the aggregator data as a vector<T>
  inline std::vector<T>& data() { return data_vector_; }

 protected:
  // Implementation of the tensor aggregation.
  void AggregateTensor(const Tensor& tensor) override {
    FCP_CHECK(internal::TypeTraits<T>::kDataType == tensor.dtype())
        << "Incompatible tensor dtype()";
    // Delegate the actual aggregation to the specific aggregation
    // intrinsic implementation.
    AggregateVector(tensor.AsAggVector<T>());
  }

  // Delegates AggVector aggregation to a derived class.
  virtual void AggregateVector(const AggVector<T>& agg_vector) = 0;

 private:
  AggVectorAggregator(DataType dtype, TensorShape shape, VectorData<T>* data)
      : TensorAggregator(
            Tensor::Create(dtype, shape, std::unique_ptr<TensorData>(data))
                .value()),
        data_vector_(*data) {
    FCP_CHECK(internal::TypeTraits<T>::kDataType == dtype)
        << "Incompatible dtype";
  }

  std::vector<T>& data_vector_;
};

}  // namespace aggregation
}  // namespace fcp

#endif  // FCP_AGGREGATION_CORE_AGG_VECTOR_AGGREGATOR_H_
