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

#include "fcp/base/result.h"

#include "fcp/base/tracing_schema.h"
#include "fcp/tracing/tracing_span.h"

namespace fcp {
Error ExpectBase::TraceExpectError(const char* expectation) const {
  return TraceError<ResultExpectError>(expectation, loc_.file_name(),
                                       loc_.line());
}

static TracingStatusCode ConvertToTracingStatus(absl::StatusCode code) {
  switch (code) {
    case absl::StatusCode::kOk:
      return TracingStatusCode_Ok;
    case absl::StatusCode::kCancelled:
      return TracingStatusCode_Cancelled;
    case absl::StatusCode::kInvalidArgument:
      return TracingStatusCode_InvalidArgument;
    case absl::StatusCode::kDeadlineExceeded:
      return TracingStatusCode_DeadlineExceeded;
    case absl::StatusCode::kNotFound:
      return TracingStatusCode_NotFound;
    case absl::StatusCode::kAlreadyExists:
      return TracingStatusCode_AlreadyExists;
    case absl::StatusCode::kPermissionDenied:
      return TracingStatusCode_PermissionDenied;
    case absl::StatusCode::kResourceExhausted:
      return TracingStatusCode_ResourceExhausted;
    case absl::StatusCode::kFailedPrecondition:
      return TracingStatusCode_FailedPrecondition;
    case absl::StatusCode::kAborted:
      return TracingStatusCode_Aborted;
    case absl::StatusCode::kOutOfRange:
      return TracingStatusCode_OutOfRange;
    case absl::StatusCode::kUnimplemented:
      return TracingStatusCode_Unimplemented;
    case absl::StatusCode::kInternal:
      return TracingStatusCode_Internal;
    case absl::StatusCode::kUnavailable:
      return TracingStatusCode_Unavailable;
    case absl::StatusCode::kDataLoss:
      return TracingStatusCode_DataLoss;
    case absl::StatusCode::kUnauthenticated:
      return TracingStatusCode_Unauthenticated;
    default:
      return TracingStatusCode_Unknown;
  }
}

Error ExpectBase::TraceUnexpectedStatus(absl::StatusCode expected_code,
                                        const absl::Status& actual) const {
  return TraceError<ResultExpectStatusError>(
      ConvertToTracingStatus(expected_code),
      ConvertToTracingStatus(actual.code()), actual.message(), loc_.file_name(),
      loc_.line());
}
}  // namespace fcp
