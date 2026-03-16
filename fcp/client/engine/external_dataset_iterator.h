/*
 * Copyright 2019 Google LLC
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

#ifndef FCP_CLIENT_ENGINE_EXTERNAL_DATASET_ITERATOR_H_
#define FCP_CLIENT_ENGINE_EXTERNAL_DATASET_ITERATOR_H_

#include <string>

#include "absl/status/statusor.h"

namespace fcp {

/**
 * Interface for an iterator, created from a particular dataset. A single
 * dataset may be used to create multiple iterators.
 */
class ExternalDatasetIterator {
 public:
  virtual ~ExternalDatasetIterator() = default;

  /**
   * Returns the next element, if possible. Indicates end-of-stream with
   * OUT_OF_RANGE, even when repeatedly called. Corresponds to
   * tensorflow::data::IteratorBase::GetNext.
   *
   * Implementations must be thread-safe.
   */
  virtual absl::StatusOr<std::string> GetNext() = 0;
};

}  // namespace fcp

#endif  // FCP_CLIENT_ENGINE_EXTERNAL_DATASET_ITERATOR_H_
