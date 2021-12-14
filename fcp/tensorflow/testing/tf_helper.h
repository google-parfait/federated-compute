/*
 * Copyright 2020 Google LLC
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

#ifndef FCP_TENSORFLOW_TESTING_TF_HELPER_H_
#define FCP_TENSORFLOW_TESTING_TF_HELPER_H_

#include <string>

#include "gtest/gtest.h"
#include "absl/strings/cord.h"
#include "fcp/base/result.h"
#include "fcp/tensorflow/tf_session.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/io_ops.h"
#include "tensorflow/cc/ops/state_ops.h"
#include "tensorflow/core/protobuf/saver.pb.h"

namespace fcp {

/**
 * Get a serialized graph with the operations defined on the provided scope.
 */
absl::Cord CreateGraph(tensorflow::Scope* root);

/**
 * Use a basic TfSession to create a checkpoint with a specified value from a
 * specified variable.
 */
template <typename T>
Result<absl::Cord> GetCheckpoint(const std::string& var_name,
                                 tensorflow::DataType dt,
                                 tensorflow::PartialTensorShape shape,
                                 const tensorflow::Input::Initializer& value) {
  // Construct a TensorFlow graph with all desired operations.
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  auto v = tensorflow::ops::Variable(root, shape, dt);
  // TODO(team): Consider breaking this function into templated and
  //  non-templated parts to reduce code size.
  auto init = tensorflow::ops::Assign(root.WithOpName("init"), v,
                                      tensorflow::ops::Const<T>(root, value));
  // Save the value of the variable in a serialized checkpoint.
  auto filename = tensorflow::ops::Placeholder(root.WithOpName("filename"),
                                               tensorflow::DT_STRING);
  auto save_a =
      tensorflow::ops::Save(root.WithOpName("save"), filename, {var_name},
                            std::initializer_list<tensorflow::Input>{v});

  // Run a session using the graph constructed above.
  TfSession session(::testing::TempDir(), CreateGraph(&root));
  FCP_TRY(session.RunOp("init"));
  // Create the checkpoint.
  google::internal::federated::plan::CheckpointOp save_checkpoint_op;
  save_checkpoint_op.mutable_saver_def()->set_save_tensor_name("save");
  save_checkpoint_op.mutable_saver_def()->set_filename_tensor_name("filename");
  return session.SaveState(save_checkpoint_op);
}

}  // namespace fcp

#endif  // FCP_TENSORFLOW_TESTING_TF_HELPER_H_
