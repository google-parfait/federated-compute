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

#ifndef FCP_AGGREGATION_CORE_DATATYPE_H_
#define FCP_AGGREGATION_CORE_DATATYPE_H_

namespace fcp::aggregation {

// A list of supported tensor value types.
enum DataType {
  DT_INVALID = 0,
  DT_FLOAT = 1,

  // TODO(team): Add other types.
  // This should be a small subset of tensorflow::DataType types and include
  // only simple numeric types and floating point types.
};

}  // namespace fcp::aggregation

#endif  // FCP_AGGREGATION_CORE_DATATYPE_H_
