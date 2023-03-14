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

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <queue>
#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/io/table.h"
#include "tensorflow/core/lib/io/table_builder.h"
#include "tensorflow/core/lib/io/table_options.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/util/saved_tensor_slice.pb.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"
#include "tensorflow/core/util/tensor_slice_reader.h"
#include "tensorflow/core/util/tensor_slice_writer.h"

namespace fcp {
namespace {

using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;

}  // namespace

class DeleteFileOp : public OpKernel {
 public:
  explicit DeleteFileOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const tensorflow::Tensor& filename_t = context->input(0);
    {
      const int64_t size = filename_t.NumElements();
      OP_REQUIRES(
          context, size == 1,
          tensorflow::errors::InvalidArgument(
              "Input 0 (filename) must be a string scalar; got a tensor of ",
              size, "elements"));
    }
    const tensorflow::tstring& filename =
        filename_t.scalar<tensorflow::tstring>()();
    if (context->env()->FileExists(filename).ok()) {
      if (!context->env()->DeleteFile(filename).ok()) {
        // The file could be already deleted by another op. Let's not propagate
        // this error.
        LOG(WARNING) << "Failed to delete the file." << filename;
      }
    }
  }
};

REGISTER_OP("DeleteFile").Input("filename: string").SetIsStateful();
REGISTER_KERNEL_BUILDER(
    Name("DeleteFile").Device(tensorflow::DEVICE_CPU),
    DeleteFileOp);

}  // namespace fcp
