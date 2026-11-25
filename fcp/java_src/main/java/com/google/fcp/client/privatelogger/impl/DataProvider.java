// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.fcp.client.privatelogger.impl;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.fcp.client.privatelogger.LogEntry;
import com.google.intelligence.fcp.client.SelectorContext;
import java.util.List;

/** A function that provides data for a private logging upload task. */
public interface DataProvider {
  ListenableFuture<List<LogEntry>> getData(SelectorContext selectorContext);
}
