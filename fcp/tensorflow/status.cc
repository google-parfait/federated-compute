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

tensorflow::Status ConvertToTensorFlowStatus(Status const& status) {
  absl::StatusCode code = status.code();
  if (code == absl::StatusCode::kOk) {
    return tensorflow::Status();
  } else {
    // tensorflow::Status constructor asserts that code != OK if a message is
    // provided.
    // Remove the cast after TF 2.12 is released and used in FCP.
    return tensorflow::Status(static_cast<tsl::errors::Code>(code),
                              status.message());
  }
}

Status ConvertFromTensorFlowStatus(tensorflow::Status const& tf_status) {
  return Status(static_cast<absl::StatusCode>(tf_status.code()),
                tf_status.error_message());
}

}  // namespace fcp
