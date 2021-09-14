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

#ifndef FCP_BASE_CONTAINER_H_
#define FCP_BASE_CONTAINER_H_

// Helpers for working with containers.

#include "absl/container/flat_hash_map.h"

namespace fcp {

/**
 * Lookup a value in a map, returning a mutable pointer to the value, or
 * null if not found. Note that pointer becomes invalid on map modifications.
 */
template <typename C, typename K>
auto TryFind(C* map, K key) -> typename C::mapped_type* {
  auto it = map->find(key);
  if (it == map->end()) {
    return nullptr;
  }
  return &it->second;
}

}  // namespace fcp

#endif  // FCP_BASE_CONTAINER_H_
