// Copyright 2026 Google LLC
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
package com.google.fcp.client.confidentialcompute;

import java.io.IOException;

/**
 * Signals that an upload failed and the payload is known not to have been stored, so the caller may
 * offer the same data again later.
 *
 * <p>Reported as the failure of the future returned by {@link ConfidentialPayloadUploader#upload}
 * when the server answers the blob upload with {@code ABORTED} (HTTP 409) or refuses authorization
 * (HTTP 403, such as an expired session), or when the upload was not started. Any other upload
 * failure surfaces the original exception instead, and whether the payload was stored is then
 * unknown.
 */
public final class UploadAbortedException extends IOException {

  /** Constructs an exception with the detail message and causal upload failure. */
  public UploadAbortedException(String message, Throwable cause) {
    super(message, cause);
  }
}
