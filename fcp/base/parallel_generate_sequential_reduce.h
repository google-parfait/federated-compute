/*
 * Copyright 2021 Google LLC
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

#ifndef FCP_BASE_PARALLEL_GENERATE_SEQUENTIAL_REDUCE_H_
#define FCP_BASE_PARALLEL_GENERATE_SEQUENTIAL_REDUCE_H_

#include <functional>
#include <memory>

#include "fcp/base/monitoring.h"

namespace fcp {

// Provides Cancellation mechanism for ParallelGenerateSequentialReduce.
class CancellationImpl {
 public:
  virtual ~CancellationImpl() = default;

  // Calling Cancel results in skipping the remaining, still pending
  // ParallelGenerateSequentialReduce. The call blocks waiting for any
  // currently active ongoing tasks to complete. Calling Cancel for the second
  // time has no additional effect.
  virtual void Cancel() = 0;
};

using CancellationToken = std::shared_ptr<CancellationImpl>;

// Interface for classes which provide parallel generation, followed by a
// sequential reduce, and finally a done callback.
template <typename T>
class ParallelGenerateSequentialReduce {
 public:
  virtual ~ParallelGenerateSequentialReduce() = default;

  // Starts executing multiple generator functions in parallel, then
  // call the accumulator function sequentially for each generator result
  // to aggregate all generator's results with the initial_value. Finally
  // on_complete is called when all results are aggregated.
  //
  // The returned CancellationToken may be used to abort an ongoing run,
  // potentially skipping some of the generator and accumulator calls.
  virtual CancellationToken Run(
      const std::vector<std::function<std::unique_ptr<T>()>>& generators,
      std::unique_ptr<T> initial_value,
      std::function<std::unique_ptr<T>(const T&, const T&)> accumulator,
      std::function<void(std::unique_ptr<T>)> on_complete) = 0;
};

}  // namespace fcp

#endif  // FCP_BASE_PARALLEL_GENERATE_SEQUENTIAL_REDUCE_H_
