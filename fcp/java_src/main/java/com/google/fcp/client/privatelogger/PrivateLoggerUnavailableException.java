/*
 * Copyright 2026 Google LLC
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
package com.google.fcp.client.privatelogger;

/**
 * Exception thrown when the PrivateLogger API is unavailable.
 *
 * <p>This can happen if the underlying implementation is not available (e.g. because the feature is
 * disabled via a flag), or if the connection to the implementation fails.
 */
public class PrivateLoggerUnavailableException extends Exception {
  public PrivateLoggerUnavailableException(String message, Throwable cause) {
    super(message, cause);
  }
}
