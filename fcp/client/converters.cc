#include "fcp/client/converters.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor.pb.h"
#include "fcp/aggregation/core/tensor_data.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "google/protobuf/repeated_ptr_field.h"

namespace fcp::client {

using ::absl::StatusOr;
using ::absl::string_view;
using ::fcp::aggregation::Tensor;
using ::fcp::aggregation::TensorData;
using ::fcp::aggregation::TensorShape;

// Similar to NumericTensorDataAdapter but performs additional conversion
// of the original value to string_view while keeping the reference to the
// original.
class StringTensorDataAdapter : public TensorData {
 public:
  explicit StringTensorDataAdapter(
      const ::google::protobuf::RepeatedPtrField<std::string>& value)
      : value_(value), string_views_(value_.begin(), value_.end()) {}

  size_t byte_size() const override {
    return string_views_.size() * sizeof(string_view);
  }
  const void* data() const override { return string_views_.data(); }

 private:
  const ::google::protobuf::RepeatedPtrField<std::string>& value_;
  std::vector<string_view> string_views_;
};

StatusOr<Tensor> ConvertStringTensor(
    TensorShape tensor_shape,
    const ::google::protobuf::RepeatedPtrField<std::string>& value) {
  return Tensor::Create(aggregation::DT_STRING, tensor_shape,
                        std::make_unique<StringTensorDataAdapter>(value));
}

}  // namespace fcp::client
