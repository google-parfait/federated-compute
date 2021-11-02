/*
 * Copyright 2020 Google LLC
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
#ifndef FCP_CLIENT_FLAGS_H_
#define FCP_CLIENT_FLAGS_H_

#include <string>

#include "absl/status/status.h"

namespace fcp {
namespace client {

// A class for changing runtime behavior with "flags" - typically, server
// provided values.
class Flags {
 public:
  virtual ~Flags() = default;

  // The period of time in milliseconds between device condition checks. This is
  // used during potentially long blocking calls such as TensorFlow or network
  // I/O, as well as for throttling regular condition checks during plan
  // execution (e.g. before fetching a new example).
  virtual int64_t condition_polling_period_millis() const = 0;

  // The period of time in milliseconds allowed for TensorFlow execution to
  // finish after it's been interrupted.
  virtual int64_t tf_execution_teardown_grace_period_millis() const = 0;

  // The period of time in milliseconds allowed for TensorFlow execution to
  // finish after the grace period. This allows us to decide if we want long
  // running native execution to be forcibly resolved or continue indefinitely.
  virtual int64_t tf_execution_teardown_extended_period_millis() const = 0;

  // The deadline in seconds for the gRPC channel used for communication
  // between the client and server.
  virtual int64_t grpc_channel_deadline_seconds() const = 0;

  // Whether to log the error message strings from TensorFlow exceptions.
  virtual bool log_tensorflow_error_messages() const = 0;

  // Whether to enable recording to and querying from the Operational Statistics
  // db.
  virtual bool enable_opstats() const { return true; }

  // If true, splits up TRAIN_STARTED, TRAIN_COMPLETED, TRAIN_ERROR and
  // TRAIN_INTERRUPTED into separate logs for eligibility, checkin, and
  // training.
  virtual bool per_phase_logs() const { return false; }

  // The number of days for data to live in the OpStatsDb without update.
  virtual int64_t opstats_ttl_days() const { return 30; }

  // The maximum size of the data stored by OpStatsDb.
  virtual int64_t opstats_db_size_limit_bytes() const { return 1 * 1024 * 1024; }

  // If true, the user won't be able to create two instances of OpStatsDb which
  // use the same underlying file.
  virtual bool opstats_enforce_singleton() const { return false; }

  // The retry delay to use when encountering a transient error during a
  // training run before having received a RetryWindow from the server.
  virtual int64_t federated_training_transient_errors_retry_delay_secs() const {
    // 15 minutes
    return 15 * 60;
  }

  // The amount of jitter to apply when using the
  // `federated_training_transient_errors_retry_delay_secs` flag. Must be a
  // value between 0 and 1. E.g. a value of 0.2 means that retry delays will
  // fall within [0.8 * target period, 1.2 * target period).
  virtual float federated_training_transient_errors_retry_delay_jitter_percent()
      const {
    return 0.2;
  }

  // The retry delay to use when encountering a permanent error during a
  // training run (regardless of whether the client already received a
  // RetryWindow from the server).
  virtual int64_t federated_training_permanent_errors_retry_delay_secs() const {
    // 4 hours
    return 4 * 60 * 60;
  }

  // The amount of jitter to apply when using the
  // `federated_training_permanent_errors_retry_delay_secs` flag. Must be a
  // value between 0 and 1. E.g. a value of 0.2 means that retry delays will
  // fall within [0.8 * target period, 1.2 * target period).
  virtual float federated_training_permanent_errors_retry_delay_jitter_percent()
      const {
    return 0.2;
  }

  // The list of error codes that should be considered 'permanent'.
  virtual std::vector<int32_t> federated_training_permanent_error_codes() const {
    return {
        // The server returns NOT_FOUND if the client checks in with an unknown
        // population name. While this can be resolved without any client
        // changes by creating the population server-side, it is nevertheless
        // wise to treat this as a 'permanent' error for which a longer
        // RetryPeriod is used, because such temporary mismatches in
        // client/server configuration are fairly common and otherwise cause
        // clients to check in unnecessarily frequently.
        static_cast<int32_t>(absl::StatusCode::kNotFound),
        // INVALID_ARGUMENT generally indicates a client-side issue (e.g. a bug
        // in the client's protocol implementation), which is unlikely to be
        // resolved by merely retrying the request.
        static_cast<int32_t>(absl::StatusCode::kInvalidArgument),
        // UNIMPLEMENTED similarly could indicate a client-side issue, or a
        // temporary server issue (e.g. a bug/missing feature implementation in
        // the server). Either way, it is also unlikely to be resolved by merely
        // retrying the request soon.
        static_cast<int32_t>(absl::StatusCode::kUnimplemented)};
  }

  // Whether to commit opstats to storage when the upload started event is
  // logged.
  virtual bool commit_opstats_on_upload_started() const { return false; }

  // Whether to set "earliest_trustworthy_time" field in OpStatsDb.
  virtual bool record_earliest_trustworthy_time_for_opstats() const {
    return false;
  }
};
}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_FLAGS_H_
