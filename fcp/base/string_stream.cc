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
