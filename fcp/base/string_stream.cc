// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fcp/base/string_stream.h"

#include <stdio.h>

#include <string>

namespace fcp {
namespace internal {

StringStream& StringStream::operator<<(bool x) {
  return *this << (x ? "true" : "false");
}

template <typename T>
StringStream& AppendWithFormat(StringStream& string_buffer, const char* format,
                               T value) {
  char buf[64];
  // The buffer is large enough for any possible value.
  sprintf(buf, format, value);  // NOLINT
  return string_buffer << buf;
}

StringStream& StringStream::operator<<(int32_t x) {
  return AppendWithFormat(*this, "%d", x);
}

StringStream& StringStream::operator<<(uint32_t x) {
  return AppendWithFormat(*this, "%u", x);
}

StringStream& StringStream::operator<<(int64_t x) {
  return AppendWithFormat(*this, "%ld", x);
}

StringStream& StringStream::operator<<(uint64_t x) {
  return AppendWithFormat(*this, "%lu", x);
}

StringStream& StringStream::operator<<(double x) {
  return AppendWithFormat(*this, "%f", x);
}

StringStream& StringStream::operator<<(const char* x) {
  str_.append(x);
  return *this;
}

StringStream& StringStream::operator<<(const std::string& x) {
  str_.append(x);
  return *this;
}

}  // namespace internal
}  // namespace fcp
