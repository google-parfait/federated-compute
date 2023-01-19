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

#ifndef FCP_AGGREGATION_CORE_AGG_VECTOR_ITERATOR_H_
#define FCP_AGGREGATION_CORE_AGG_VECTOR_ITERATOR_H_

#include "fcp/aggregation/core/tensor_data.h"

namespace fcp {
namespace aggregation {

// Iterator for AggVector which allows to iterate over sparse values
// as a collection of {index, value} pairs.
//
// This allows a simple iteration loops like the following:
// for (auto [index, value] : agg_vector) {
//    ... aggregate the value at the given dense index
// }
template <typename T>
struct AggVectorIterator {
  struct IndexValuePair {
    size_t index;
    T value;

    friend bool operator==(const IndexValuePair& a, const IndexValuePair& b) {
      return a.index == b.index && a.value == b.value;
    }

    friend bool operator!=(const IndexValuePair& a, const IndexValuePair& b) {
      return a.index != b.index || a.value != b.value;
    }
  };

  using value_type = IndexValuePair;
  using pointer = value_type*;
  using reference = value_type&;

  explicit AggVectorIterator(const TensorData* data)
      : AggVectorIterator(data, 0, get_slice(data, 0)) {}

  // Current dense index corresponding to the current value.
  size_t index() const { return dense_index; }
  // Current value.
  T value() const { return *ptr; }
  // The current interator {index, value} pair value. This is used by
  // for loop iterators.
  IndexValuePair operator*() const { return {dense_index, *ptr}; }

  AggVectorIterator& operator++() {
    if (++ptr == slice_end_ptr) {
      TensorData::Slice slice = get_slice(data, ++slice_index);
      ptr = slice_begin(slice);
      slice_end_ptr = slice_end(slice);
      dense_index = slice_start_index(slice);
    } else {
      dense_index++;
    }
    return *this;
  }

  AggVectorIterator operator++(int) {
    AggVectorIterator tmp = *this;
    ++(*this);
    return tmp;
  }

  friend bool operator==(const AggVectorIterator& a,
                         const AggVectorIterator& b) {
    return a.ptr == b.ptr;
  }

  friend bool operator!=(const AggVectorIterator& a,
                         const AggVectorIterator& b) {
    return a.ptr != b.ptr;
  }

  static AggVectorIterator end() {
    return AggVectorIterator(nullptr, 0, TensorData::Slice{0, 0, nullptr});
  }

 private:
  AggVectorIterator(const TensorData* data, int slice_index,
                    TensorData::Slice slice)
      : ptr(slice_begin(slice)),
        slice_end_ptr(slice_end(slice)),
        dense_index(slice_start_index(slice)),
        data(data),
        slice_index(slice_index) {}

  static TensorData::Slice get_slice(const TensorData* data, int slice_index) {
    return slice_index < data->num_slices() ? data->get_slice(slice_index)
                                            : TensorData::Slice{0, 0, nullptr};
  }

  static const T* slice_begin(const TensorData::Slice& slice) {
    return static_cast<const T*>(slice.data);
  }

  static const T* slice_end(const TensorData::Slice& slice) {
    return slice_begin(slice) + slice.byte_size / sizeof(T);
  }

  static size_t slice_start_index(const TensorData::Slice& slice) {
    return slice.byte_offset / sizeof(T);
  }

  const T* ptr;
  const T* slice_end_ptr;
  size_t dense_index;
  const TensorData* data;
  int slice_index;
};

}  // namespace aggregation
}  // namespace fcp

#endif  // FCP_AGGREGATION_CORE_AGG_VECTOR_ITERATOR_H_
