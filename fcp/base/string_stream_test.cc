#include "fcp/base/string_stream.h"

#include <string>

#include "gtest/gtest.h"

namespace fcp {
namespace internal {
namespace {

TEST(StringTest, Basic) {
  StringStream s;
  s << "A" << 1 << std::string("b") << 2U << "c" << -3L << "d" << 3.5f << "e"
    << -3.14 << "f" << 9UL << true << ":" << false;
  EXPECT_EQ(s.str(), "A1b2c-3d3.500000e-3.140000f9true:false");
}

}  // namespace
}  // namespace internal
}  // namespace fcp
