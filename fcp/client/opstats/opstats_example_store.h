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
#ifndef FCP_CLIENT_OPSTATS_OPSTATS_EXAMPLE_STORE_H_
#define FCP_CLIENT_OPSTATS_OPSTATS_EXAMPLE_STORE_H_

#include <memory>

#include "absl/status/statusor.h"
#include "fcp/client/engine/example_iterator_factory.h"
#include "fcp/client/log_manager.h"
#include "fcp/protos/plan.pb.h"

namespace fcp {
namespace client {
namespace opstats {
inline static constexpr char kOpStatsCollectionUri[] = "internal:/opstats";
inline static constexpr char kSupportsNeetContext[] = "supports_neet_context";

class OpStatsExampleIteratorFactory
    : public fcp::client::engine::ExampleIteratorFactory {
 public:
  explicit OpStatsExampleIteratorFactory(LogManager* log_manager)
      : log_manager_(log_manager) {}

  bool CanHandle(const google::internal::federated::plan::ExampleSelector&
                     example_selector) override;

  bool ShouldCollectStats() override { return true; }

  absl::StatusOr<std::unique_ptr<ExampleIterator>> CreateExampleIterator(
      const google::internal::federated::plan::ExampleSelector&
          example_selector) override;

 private:
  LogManager* log_manager_;
};

}  // namespace opstats
}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_OPSTATS_OPSTATS_EXAMPLE_STORE_H_
