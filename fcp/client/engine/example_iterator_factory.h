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
#ifndef FCP_CLIENT_ENGINE_EXAMPLE_ITERATOR_FACTORY_H_
#define FCP_CLIENT_ENGINE_EXAMPLE_ITERATOR_FACTORY_H_

#include <functional>
#include <memory>

#include "absl/status/statusor.h"
#include "fcp/client/simple_task_environment.h"
#include "fcp/protos/plan.pb.h"

namespace fcp {
namespace client {
namespace engine {

// An interface for engine-internal usage, describing a handle with which
// ExampleIterator instances can be created, based on a given ExampleSelector.
// Each handle indicates which type of ExampleSelector it is actually able to
// handle.
class ExampleIteratorFactory {
 public:
  // Whether this factory can create iterators that serve the given
  // `ExampleSelector`-described query.
  virtual bool CanHandle(
      const google::internal::federated::plan::ExampleSelector&
          example_selector) = 0;

  // Creates an iterator for the given `ExampleSelector`-described query.
  virtual absl::StatusOr<std::unique_ptr<ExampleIterator>>
  CreateExampleIterator(
      const google::internal::federated::plan::ExampleSelector&
          example_selector) = 0;

  // Whether stats should be generated and logged into the OpStats database for
  // iterators created by this factory.
  virtual bool ShouldCollectStats() = 0;

  virtual ~ExampleIteratorFactory() {}
};

// A utility ExampleIteratorFactory implementation that can use simple
// std::function objects and wrap them.
class FunctionalExampleIteratorFactory : public ExampleIteratorFactory {
 public:
  // Creates an `ExampleIteratorFactory` that can handle all queries and for
  // which stats are collected,  and delegates the creation of the
  // `ExampleIterator` to an std::function.
  explicit FunctionalExampleIteratorFactory(
      std::function<absl::StatusOr<std::unique_ptr<ExampleIterator>>(
          const google::internal::federated::plan::ExampleSelector&

          )>
          create_iterator_func)
      : can_handle_func_(
            [](const google::internal::federated::plan::ExampleSelector&) {
              return true;
            }),
        create_iterator_func_(create_iterator_func),
        should_collect_stats_(true) {}

  // Creates an `ExampleIteratorFactory` that delegates to an std::function to
  // determine if a given query can be handled, and delegates the creation of
  // the `ExampleIterator` to an std::function as well.
  FunctionalExampleIteratorFactory(
      std::function<
          bool(const google::internal::federated::plan::ExampleSelector&)>
          can_handle_func,
      std::function<absl::StatusOr<std::unique_ptr<ExampleIterator>>(
          const google::internal::federated::plan::ExampleSelector&

          )>
          create_iterator_func,
      bool should_collect_stats)
      : can_handle_func_(can_handle_func),
        create_iterator_func_(create_iterator_func),
        should_collect_stats_(should_collect_stats) {}

  bool CanHandle(const google::internal::federated::plan::ExampleSelector&
                     example_selector) override {
    return can_handle_func_(example_selector);
  }

  absl::StatusOr<std::unique_ptr<ExampleIterator>> CreateExampleIterator(
      const google::internal::federated::plan::ExampleSelector&
          example_selector) override {
    return create_iterator_func_(example_selector);
  }

  bool ShouldCollectStats() override { return should_collect_stats_; }

 private:
  std::function<bool(const google::internal::federated::plan::ExampleSelector&)>
      can_handle_func_;
  std::function<absl::StatusOr<std::unique_ptr<ExampleIterator>>(
      const google::internal::federated::plan::ExampleSelector&)>
      create_iterator_func_;
  bool should_collect_stats_;
};

}  // namespace engine
}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_ENGINE_EXAMPLE_ITERATOR_FACTORY_H_
