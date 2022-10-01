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

struct OutputTensors {
  std::vector<std::string> output_tensor_names;
  std::vector<tensorflow::Tensor> output_tensors;
};

// Options for TFLite interpreter.
struct TfLiteInterpreterOptions {
  // When true, TFLite uses dynamic tensor allocation and release tensors that
  // are no longer needed.
  bool ensure_dynamic_tensors_are_released = false;
  // When the threshold is zero, dynamic allocation is not enabled for any
  // tensor.
  int32_t large_tensor_threshold_for_dynamic_allocation = 0;
  // Whether to disable the graph-reordering optimization that clusters delegate
  // ops together.
  bool disable_delegate_clustering = false;
};

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
// Parameters:
//    1. model: The serialized TFLite model.
//    2. should_abort: A function which will be polled periodically to determine
//    if the computation should be aborted.
//    3. timing_config: The TimingConfig for an InterruptibleRunner.
//    4. log_manager: A LogManager.
//    5. inputs: A hashmap which has input tensor name as key, tensor data as
//    value.
//    6. output_names: The names of the output tensors. The order for these
//    tensor names must be deterministic.
class TfLiteWrapper {
 public:
  static absl::StatusOr<std::unique_ptr<TfLiteWrapper>> Create(
      const std::string& model, std::function<bool()> should_abort,
      const InterruptibleRunner::TimingConfig& timing_config,
      LogManager* log_manager,
      std::unique_ptr<absl::flat_hash_map<std::string, std::string>> inputs,
      std::vector<std::string> output_names,
      const TfLiteInterpreterOptions& interpreter_options, int32_t num_threads);

  // Wrapper around TfLite's Interpreter::Invoke method.
  // If the run succeeds, a vector of output tensors (empty if there's no
  // output tensors), or CANCELLED if the training run was cancelled or
  // INVALID_ARGUMENT for the rest of errors.
  absl::StatusOr<OutputTensors> Run();

 private:
  TfLiteWrapper(std::unique_ptr<tflite::FlatBufferModel> model,
                std::unique_ptr<CachingErrorReporter> error_reporter,
                tflite::TfLiteDelegateUniquePtr delegate,
                std::unique_ptr<tflite::Interpreter> interpreter,
                std::unique_ptr<InterruptibleRunner> interruptible_runner,
                std::vector<std::string> output_names)
      : model_(std::move(model)),
        error_reporter_(std::move(error_reporter)),
        delegate_(std::move(delegate)),
        interpreter_(std::move(interpreter)),
        interruptible_runner_(std::move(interruptible_runner)),
        output_names_(std::move(output_names)) {}
  absl::Status ConvertTfLiteStatus(TfLiteStatus status);
  absl::StatusOr<OutputTensors> ConstructOutputs();

  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<CachingErrorReporter> error_reporter_;
  tflite::TfLiteDelegateUniquePtr delegate_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::unique_ptr<InterruptibleRunner> interruptible_runner_;
  const std::vector<std::string> output_names_;
};

}  // namespace engine
}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_ENGINE_TFLITE_WRAPPER_H_
