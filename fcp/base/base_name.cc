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

#include "fcp/base/base_name.h"

#include <string>

namespace fcp {

#ifdef _WIN32
constexpr char kPathSeparator[] = "\\";
#else
constexpr char kPathSeparator[] = "/";
#endif

std::string BaseName(const std::string& path) {
  // Note that find_last_of returns npos if not found, and npos+1 is guaranteed
  // to be zero.
  return path.substr(path.find_last_of(kPathSeparator) + 1);
}

}  // namespace fcp
