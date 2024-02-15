/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FCP_BASE_STRING_STREAM_H_
#define FCP_BASE_STRING_STREAM_H_

#include <string>

#ifndef FCP_BAREMETAL
static_assert(false,
              "StringStream should be used only when building FCP in bare "
              "metal configuration.");
#endif

namespace fcp {
namespace internal {

// This is a class used for building diagnostic messages with
// FCP_LOG and FCP_STATUS macros.
// The class is designed to be a simplified replacement for std::ostringstream.
class StringStream final {
 public:
  StringStream() = default;
  explicit StringStream(const std::string& str) : str_(str) {}

  StringStream& operator<<(bool x);
  StringStream& operator<<(int32_t x);
  StringStream& operator<<(uint32_t x);
  StringStream& operator<<(int64_t x);
  StringStream& operator<<(uint64_t x);
  StringStream& operator<<(double x);
  StringStream& operator<<(const char* x);
  StringStream& operator<<(const std::string& x);

  std::string str() const { return str_; }

 private:
  std::string str_;
};

}  // namespace internal
}  // namespace fcp

#endif  // FCP_BASE_STRING_STREAM_H_
