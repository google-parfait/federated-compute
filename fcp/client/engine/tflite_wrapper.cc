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
#include "fcp/client/engine/tflite_wrapper.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "google/protobuf/any.pb.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "fcp/base/monitoring.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/lite/delegates/flex/util.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/string_util.h"

namespace fcp {
namespace client {
namespace engine {

using ::tflite::ops::builtin::BuiltinOpResolver;

namespace {

absl::Status AssignStringInput(int index, const std::string& value,
                               tflite::Interpreter* interpreter) {
  TfLiteTensor* tensor = interpreter->tensor(index);
  if (tensor->type != kTfLiteString) {
    return absl::InvalidArgumentError("Input tensor is not a string tensor.");
  }

  tflite::DynamicBuffer buf;
  buf.AddString(value.data(), value.length());
  buf.WriteToTensor(tensor, nullptr);
  return absl::OkStatus();
}

}  // anonymous namespace

absl::StatusOr<std::unique_ptr<TfLiteWrapper>> TfLiteWrapper::Create(
    const std::string& model, std::function<bool()> should_abort,
    const InterruptibleRunner::TimingConfig& timing_config,
    LogManager* log_manager,
    std::unique_ptr<absl::flat_hash_map<std::string, std::string>> inputs,
    std::vector<std::string> output_names,
    const TfLiteInterpreterOptions& interpreter_options, int32_t num_threads) {
  std::unique_ptr<tflite::FlatBufferModel> flat_buffer_model =
      tflite::FlatBufferModel::BuildFromBuffer(model.c_str(), model.size());
  if (flat_buffer_model == nullptr) {
    return absl::InvalidArgumentError("Failed to build FlatBufferModel.");
  }
  // The training delegate needs to be created before the interpreter.
  auto delegate = tflite::FlexDelegate::Create();
  auto error_reporter = std::make_unique<CachingErrorReporter>();
  auto interpreter = std::make_unique<tflite::Interpreter>();

  if (tflite::InterpreterBuilder(
          flat_buffer_model->GetModel(), BuiltinOpResolver(),
          error_reporter.get())(&interpreter) != kTfLiteOk) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to initiate interpreter: ",
                     error_reporter->GetFirstErrorMessage()));
  }
  interpreter->SetNumThreads(num_threads);
  if (interpreter->ModifyGraphWithDelegate(delegate.get()) != kTfLiteOk) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to modify graph with TrainingFlexDelegate: ",
                     error_reporter->GetFirstErrorMessage()));
  }
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to allocate tensors: ",
                     error_reporter->GetFirstErrorMessage()));
  }
  interpreter->SetCancellationFunction(delegate->data_,
                                       tflite::FlexDelegate::HasCancelled);
  for (const auto& input : interpreter->inputs()) {
    std::string key = interpreter->GetInputName(input);
    if (inputs->find(key) == inputs->end()) {
      return absl::InvalidArgumentError("Unexpected input tensor.");
    }
    FCP_RETURN_IF_ERROR(
        AssignStringInput(input, inputs->at(key), interpreter.get()));
  }
  // Create an InterruptibleRunner to execute TF calls in a background thread,
  // allowing us to abort them if need be.
  auto runner = std::make_unique<InterruptibleRunner>(
      log_manager, should_abort, timing_config,
      InterruptibleRunner::DiagnosticsConfig{
          .interrupted =
              ProdDiagCode::BACKGROUND_TRAINING_INTERRUPT_TF_EXECUTION,
          .interrupt_timeout = ProdDiagCode::
              BACKGROUND_TRAINING_INTERRUPT_TF_EXECUTION_TIMED_OUT,
          .interrupted_extended = ProdDiagCode::
              BACKGROUND_TRAINING_INTERRUPT_TF_EXTENDED_EXECUTION_COMPLETED,
          .interrupt_timeout_extended = ProdDiagCode::
              BACKGROUND_TRAINING_INTERRUPT_TF_EXTENDED_EXECUTION_TIMED_OUT});
  return absl::WrapUnique(
      new TfLiteWrapper(std::move(flat_buffer_model), std::move(error_reporter),
                        std::move(delegate), std::move(interpreter),
                        std::move(runner), std::move(output_names)));
}

absl::StatusOr<OutputTensors> TfLiteWrapper::Run() {
  auto* interpreter_raw_pointer = interpreter_.get();
  auto tflite_runnable = [interpreter_raw_pointer, this]() {
    return ConvertTfLiteStatus(interpreter_raw_pointer->Invoke());
  };
  auto* delegate_raw_pointer =
      static_cast<tflite::FlexDelegate*>(delegate_->data_);
  auto abort_tflite = [delegate_raw_pointer]() {
    delegate_raw_pointer->Cancel();
  };
  FCP_RETURN_IF_ERROR(
      interruptible_runner_->Run(tflite_runnable, abort_tflite));
  // handles output tensors
  return ConstructOutputs();
}

absl::Status TfLiteWrapper::ConvertTfLiteStatus(TfLiteStatus status) {
  switch (status) {
    case kTfLiteOk:
      return absl::OkStatus();
    case kTfLiteError: {
      // TfLite doesn't differentiate the error type when the training is
      // cancelled or an error happened during training. It also doesn't
      // distinguish different error types thrown by Tensorflow. Therefore, we
      // need to check whether the training was cancelled, and record the error
      // message from the ErrorReporter.
      if (tflite::FlexDelegate::HasCancelled(delegate_->data_)) {
        return absl::CancelledError("Training is cancelled.");
      }
      std::string error = error_reporter_->GetFirstErrorMessage();
      if (error.empty()) {
        return absl::InvalidArgumentError("Empty error messages returned.");
      }
      // Use the first error we encountered.
      return absl::InvalidArgumentError(error);
    }
    case kTfLiteDelegateError:
      return absl::InvalidArgumentError("TfLite delegate error.");
    case kTfLiteApplicationError:
      return absl::InvalidArgumentError(
          "An error in applying a delegate due to incompatibility between "
          "runtime and delegate");
    case kTfLiteDelegateDataNotFound:
      return absl::InvalidArgumentError(
          "Serialized delegate data not being found");
    case kTfLiteDelegateDataWriteError:
      return absl::InvalidArgumentError(
          "Data-writing issues in delegate serialization");
    case kTfLiteDelegateDataReadError:
      return absl::InvalidArgumentError(
          "Data-reading issues in delegate serialization.");
    case kTfLiteUnresolvedOps:
      return absl::InvalidArgumentError(
          "The TF Lite model has ops that cannot be resolved at runtime.");
    default:
      return absl::InternalError("Unexpected TfLiteStatus.");
  }
}

absl::StatusOr<OutputTensors> TfLiteWrapper::ConstructOutputs() {
  if (interpreter_->outputs().size() != output_names_.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("The number of output tensors is wrong. Expected: %d, "
                        "Returned by TFLite interpreter: %d",
                        output_names_.size(), interpreter_->outputs().size()));
  }
  OutputTensors output_tensors;
  // The order of the output tensors should match the order of output tensor
  // names.
  for (int output_tensor_index : interpreter_->outputs()) {
    auto tensor = tflite::flex::CreateTfTensorFromTfLiteTensor(
        interpreter_->tensor(output_tensor_index));
    if (!tensor.ok()) {
#if TF_GRAPH_DEF_VERSION < 1467
      return absl::InvalidArgumentError(tensor.status().error_message());
#else
      return absl::InvalidArgumentError(tensor.status().message());
#endif
    }
    output_tensors.output_tensors.push_back(*tensor);
  }
  output_tensors.output_tensor_names = output_names_;
  return output_tensors;
}

}  // namespace engine
}  // namespace client
}  // namespace fcp
