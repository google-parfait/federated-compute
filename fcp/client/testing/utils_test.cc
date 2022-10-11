#include "fcp/client/testing/utils.h"

#include <string>

#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "fcp/protos/plan.pb.h"

namespace fcp::client::testing {
namespace {

using google::internal::federated::plan::Dataset;

// The fixture for testing class Foo.
class TestExampleIteratorTest : public ::testing::Test {};

TEST_F(TestExampleIteratorTest, NextCallsForEmptyDatset) {
  Dataset::ClientDataset client_dataset;

  TestExampleIterator iterator(&client_dataset);

  absl::StatusOr<std::string> element;

  element = iterator.Next();
  EXPECT_FALSE(element.ok());

  element = iterator.Next();
  EXPECT_FALSE(element.ok());

  element = iterator.Next();
  EXPECT_FALSE(element.ok());
}

TEST_F(TestExampleIteratorTest, NextCallsForSingleElementDataset) {
  Dataset::ClientDataset client_dataset;
  client_dataset.add_example("abc");

  TestExampleIterator iterator(&client_dataset);

  absl::StatusOr<std::string> element;

  element = iterator.Next();
  ASSERT_TRUE(element.ok());
  EXPECT_EQ(element.value(), "abc");

  element = iterator.Next();
  EXPECT_FALSE(element.ok());

  element = iterator.Next();
  EXPECT_FALSE(element.ok());
}

// Tests that the Foo::Bar() method does Abc.
TEST_F(TestExampleIteratorTest, NextCallsForThreeElementDataset) {
  Dataset::ClientDataset client_dataset;
  client_dataset.add_example("a");
  client_dataset.add_example("b");
  client_dataset.add_example("c");

  TestExampleIterator iterator(&client_dataset);

  absl::StatusOr<std::string> element;

  element = iterator.Next();
  ASSERT_TRUE(element.ok());
  EXPECT_EQ(element.value(), "a");

  element = iterator.Next();
  ASSERT_TRUE(element.ok());
  EXPECT_EQ(element.value(), "b");

  element = iterator.Next();
  ASSERT_TRUE(element.ok());
  EXPECT_EQ(element.value(), "c");

  element = iterator.Next();
  EXPECT_FALSE(element.ok());

  element = iterator.Next();
  EXPECT_FALSE(element.ok());
}

}  // namespace
}  // namespace fcp::client::testing

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
