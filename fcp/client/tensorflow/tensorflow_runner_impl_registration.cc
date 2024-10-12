/*
 * Copyright 2024 Google LLC
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
#include <memory>

#include "fcp/base/function_registry.h"
#include "fcp/client/tensorflow/tensorflow_runner_factory.h"
#include "fcp/client/tensorflow/tensorflow_runner_impl.h"

namespace fcp::client {

// Register the TensorflowRunner implementation.
const auto kUnused = fcp::RegisterOrDie(
    GetGlobalTensorflowRunnerFactoryRegistry(),
    TensorflowRunnerImplementation::kTensorflowRunnerImpl,
    []() { return std::make_unique<TensorflowRunnerImpl>(); });

}  // namespace fcp::client
