/*
 * Copyright 2019 Google LLC
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

#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "fcp/client/grpc_bidi_stream.h"
#include "fcp/protos/federated_api.grpc.pb.h"
#include "fcp/testing/testing.h"

namespace fcp {
namespace client {
namespace test {
namespace {

using google::internal::federatedml::v2::ClientStreamMessage;
using google::internal::federatedml::v2::ServerStreamMessage;
using ::testing::Not;

}  // namespace
}  // namespace test
}  // namespace client
}  // namespace fcp
