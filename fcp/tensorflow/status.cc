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

#include "fcp/tensorflow/status.h"

namespace fcp {

#define MAP_TO_TENSORFLOW_STATUS(name) \
  case fcp::name:                      \
    return tensorflow::error::Code::name;

#define MAP_FROM_TENSORFLOW_STATUS(name) \
  case tensorflow::error::Code::name:    \
    return fcp::name;

tensorflow::error::Code ToTensorFlowStatusCode(StatusCode code) {
  switch (code) {
    MAP_TO_TENSORFLOW_STATUS(OK)
    MAP_TO_TENSORFLOW_STATUS(CANCELLED)
    MAP_TO_TENSORFLOW_STATUS(UNKNOWN)
    MAP_TO_TENSORFLOW_STATUS(INVALID_ARGUMENT)
    MAP_TO_TENSORFLOW_STATUS(DEADLINE_EXCEEDED)
    MAP_TO_TENSORFLOW_STATUS(NOT_FOUND)
    MAP_TO_TENSORFLOW_STATUS(ALREADY_EXISTS)
    MAP_TO_TENSORFLOW_STATUS(PERMISSION_DENIED)
    MAP_TO_TENSORFLOW_STATUS(UNAUTHENTICATED)
    MAP_TO_TENSORFLOW_STATUS(RESOURCE_EXHAUSTED)
    MAP_TO_TENSORFLOW_STATUS(FAILED_PRECONDITION)
    MAP_TO_TENSORFLOW_STATUS(ABORTED)
    MAP_TO_TENSORFLOW_STATUS(OUT_OF_RANGE)
    MAP_TO_TENSORFLOW_STATUS(UNIMPLEMENTED)
    MAP_TO_TENSORFLOW_STATUS(INTERNAL)
    MAP_TO_TENSORFLOW_STATUS(UNAVAILABLE)
    MAP_TO_TENSORFLOW_STATUS(DATA_LOSS)
    default:
      return tensorflow::error::Code::UNKNOWN;
  }
}

StatusCode FromTensorFlowStatusCode(tensorflow::error::Code code) {
  switch (code) {
    MAP_FROM_TENSORFLOW_STATUS(OK)
    MAP_FROM_TENSORFLOW_STATUS(CANCELLED)
    MAP_FROM_TENSORFLOW_STATUS(UNKNOWN)
    MAP_FROM_TENSORFLOW_STATUS(INVALID_ARGUMENT)
    MAP_FROM_TENSORFLOW_STATUS(DEADLINE_EXCEEDED)
    MAP_FROM_TENSORFLOW_STATUS(NOT_FOUND)
    MAP_FROM_TENSORFLOW_STATUS(ALREADY_EXISTS)
    MAP_FROM_TENSORFLOW_STATUS(PERMISSION_DENIED)
    MAP_FROM_TENSORFLOW_STATUS(UNAUTHENTICATED)
    MAP_FROM_TENSORFLOW_STATUS(RESOURCE_EXHAUSTED)
    MAP_FROM_TENSORFLOW_STATUS(FAILED_PRECONDITION)
    MAP_FROM_TENSORFLOW_STATUS(ABORTED)
    MAP_FROM_TENSORFLOW_STATUS(OUT_OF_RANGE)
    MAP_FROM_TENSORFLOW_STATUS(UNIMPLEMENTED)
    MAP_FROM_TENSORFLOW_STATUS(INTERNAL)
    MAP_FROM_TENSORFLOW_STATUS(UNAVAILABLE)
    MAP_FROM_TENSORFLOW_STATUS(DATA_LOSS)
    default:
      return StatusCode::kUnknown;
  }
}

tensorflow::Status ConvertToTensorFlowStatus(Status const& status) {
  tensorflow::error::Code code = ToTensorFlowStatusCode(status.code());
  if (code == tensorflow::error::Code::OK) {
    return tensorflow::Status();
  } else {
    // tensorflow::Status constructor asserts that code != OK if a message is
    // provided.
    return tensorflow::Status(code, status.message());
  }
}

Status ConvertFromTensorFlowStatus(tensorflow::Status const& tf_status) {
  return Status(FromTensorFlowStatusCode(tf_status.code()),
                tf_status.error_message());
}

}  // namespace fcp
