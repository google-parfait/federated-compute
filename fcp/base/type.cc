/*
 * Copyright 2018 Google LLC
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

#include "fcp/base/type.h"

#include <string>

#include "fcp/base/monitoring.h"

namespace fcp {

std::string TypeRep::ToString() const {
  char const* n = name();
  if (n == nullptr) {
    return "<unnamed type>";
  } else {
    return n;
  }
}

}  // namespace fcp
