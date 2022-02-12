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
#ifndef FCP_CLIENT_ENGINE_TFLITE_WRAPPER_H_
#define FCP_CLIENT_ENGINE_TFLITE_WRAPPER_H_

#include <functional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fcp/client/engine/caching_error_reporter.h"
#include "fcp/client/interruptible_runner.h"
#include "fcp/client/log_manager.h"
#include "fcp/client/simple_task_environment.h"
#include "tensorflow/lite/delegates/flex/delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"

namespace fcp {
namespace client {
namespace engine {

// A class to call into TFLite.
// All functions in this interface indicate errors as follows:
// - CANCELLED: interrupted execution
// - INVALID_ARGUMENT:
//    1. Invalid model.
//    2. Initialization failure for TFLite required classes such as Interpreter,
//    Delegate etc.
//    3. Missing required inputs.
//    4. TensorFlow error. The TensorFlow error messages are included in the
//    Status message.
// This class supports aborting ongoing calls, by polling the provided
// should_abort function.
class TfLiteWrapper {
 public:
  static absl::StatusOr<std::unique_ptr<TfLiteWrapper>> Create(
      const std::string& model, std::function<bool()> should_abort,
      const InterruptibleRunner::TimingConfig& timing_config,
      LogManager* log_manager,
      std::unique_ptr<absl::flat_hash_map<std::string, std::string>> inputs);

  // Wrapper around TfLite's Interpreter::Invoke method.
  // If the run succeeds, a vector of output tensors (empty if there's no
  // output tensors), or CANCELLED if the training run was cancelled or
  // INVALID_ARGUMENT for the rest of errors.
  absl::StatusOr<std::vector<tensorflow::Tensor>> Run();

 private:
  TfLiteWrapper(std::unique_ptr<tflite::FlatBufferModel> model,
                std::unique_ptr<CachingErrorReporter> error_reporter,
                tflite::TfLiteDelegateUniquePtr delegate,
                std::unique_ptr<tflite::Interpreter> interpreter,
                std::unique_ptr<InterruptibleRunner> interruptible_runner)
      : model_(std::move(model)),
        error_reporter_(std::move(error_reporter)),
        delegate_(std::move(delegate)),
        interpreter_(std::move(interpreter)),
        interruptible_runner_(std::move(interruptible_runner)) {}
  absl::Status ConvertTfLiteStatus(TfLiteStatus status);
  absl::StatusOr<std::vector<tensorflow::Tensor>> ConstructOutputs();

  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<CachingErrorReporter> error_reporter_;
  tflite::TfLiteDelegateUniquePtr delegate_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::unique_ptr<InterruptibleRunner> interruptible_runner_;
};

}  // namespace engine
}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_ENGINE_TFLITE_WRAPPER_H_
