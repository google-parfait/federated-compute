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

#include <string>

#include "fcp/aggregation/core/tensor_aggregator_factory.h"

#ifdef FCP_BAREMETAL
#include <unordered_map>
#else
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#endif

namespace fcp {
namespace aggregation {

namespace internal {

class Registry final {
 public:
  void RegisterAggregatorFactory(const std::string& intrinsic_uri,
                                 const TensorAggregatorFactory* factory) {
    FCP_CHECK(factory != nullptr);

#ifndef FCP_BAREMETAL
    absl::MutexLock lock(&mutex_);
#endif
    FCP_CHECK(map_.find(intrinsic_uri) == map_.end())
        << "A factory for intrinsic_uri '" << intrinsic_uri
        << "' is already registered.";
    map_[intrinsic_uri] = factory;
    FCP_LOG(INFO) << "TensorAggregatorFactory for intrinsic_uri '"
                  << intrinsic_uri << "' is registered.";
  }

  StatusOr<const TensorAggregatorFactory*> GetAggregatorFactory(
      const std::string& intrinsic_uri) {
#ifndef FCP_BAREMETAL
    absl::MutexLock lock(&mutex_);
#endif
    auto it = map_.find(intrinsic_uri);
    if (it == map_.end()) {
      return FCP_STATUS(NOT_FOUND)
             << "Unknown factory for intrinsic_uri '" << intrinsic_uri << "'.";
    }
    return it->second;
  }

 private:
#ifdef FCP_BAREMETAL
  std::unordered_map<std::string, const TensorAggregatorFactory*> map_;
#else
  // Synchronization of potentially concurrent registry calls is done only in
  // the non-baremetal environment. In the baremetal environment, since there is
  // no OS, a single thread execution environment is expected and the
  // synchronization primitives aren't available.
  absl::Mutex mutex_;
  absl::flat_hash_map<std::string, const TensorAggregatorFactory*> map_
      ABSL_GUARDED_BY(mutex_);
#endif
};

#ifdef FCP_BAREMETAL
// TODO(team): Revise the registration mechanism below.
// In a baremetal build the static initialization mechanism isn't available
// which means that all the aggregation intrinsics need to be explicitly
// registered below.
extern "C" void RegisterFederatedSum();

void RegisterAll() { RegisterFederatedSum(); }
#endif  // FCP_BAREMETAL

Registry* GetRegistry() {
  static Registry* global_registry = new Registry();
#ifdef FCP_BAREMETAL
  // TODO(team): Revise the registration mechanism below.
  static bool registration_done = false;
  if (!registration_done) {
    registration_done = true;
    RegisterAll();
  }
#endif
  return global_registry;
}

}  // namespace internal

// Registers a factory instance for the given intrinsic type.
void RegisterAggregatorFactory(const std::string& intrinsic_uri,
                               const TensorAggregatorFactory* factory) {
  internal::GetRegistry()->RegisterAggregatorFactory(intrinsic_uri, factory);
}

// Looks up a factory instance for the given intrinsic type.
StatusOr<const TensorAggregatorFactory*> GetAggregatorFactory(
    const std::string& intrinsic_uri) {
  return internal::GetRegistry()->GetAggregatorFactory(intrinsic_uri);
}

}  // namespace aggregation
}  // namespace fcp
