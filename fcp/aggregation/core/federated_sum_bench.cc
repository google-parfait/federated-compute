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

#include <cstdint>
#include <utility>
#include <vector>

#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor_aggregator_factory.h"
#include "fcp/aggregation/core/tensor_aggregator_registry.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/aggregation/testing/test_data.h"

// Open-source version of benchmarking library
#include "benchmark//benchmark.h"

namespace fcp::aggregation {
namespace {

constexpr static int64_t kLength = 1000000;

static void BM_FederatedSumAccumulate(benchmark::State& state) {
  auto aggregator = (*GetAggregatorFactory("federated_sum"))
                        ->Create(DT_INT64, {kLength})
                        .value();
  auto test_data = CreateTestData<int64_t>(kLength);
  std::vector<int64_t> input(kLength);
  for (int64_t i = 0; i < kLength; ++i) {
    input[i] = i % 123;
  }
  test_data->AddSlice(0, input);
  auto tensor = Tensor::Create(DT_INT64, {kLength}, std::move(test_data));
  auto items_processed = 0;
  for (auto s : state) {
    benchmark::DoNotOptimize(aggregator->Accumulate(*tensor));
    items_processed += kLength;
  }
  state.SetItemsProcessed(items_processed);
}

BENCHMARK(BM_FederatedSumAccumulate);

}  // namespace
}  // namespace fcp::aggregation
