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
