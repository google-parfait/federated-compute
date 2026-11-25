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
package com.google.fcp.client.confidentialcompute.internal.http;

import com.google.fcp.client.confidentialcompute.internal.http.FcpHttpClient.InvalidRequestException;
import com.google.rpc.Code;
import java.io.IOException;
import java.net.SocketTimeoutException;

/**
 * Exception thrown when an HTTP request fails with a non-2xx status code.
 *
 * <p>Mirrors {@code ConvertHttpCodeToStatus} in {@code fcp/client/http/http_client_util.cc}.
 */
public final class FcpHttpException extends IOException {

  private final int responseCode;

  /** Creates an exception for an HTTP response with {@code responseCode} and {@code message}. */
  public FcpHttpException(int responseCode, String message) {
    super(message);
    this.responseCode = responseCode;
  }

  /** Returns the HTTP response status code. */
  public int responseCode() {
    return responseCode;
  }

  /**
   * Maps an {@link IOException} to a canonical {@link Code} value.
   *
   * <p>An {@link FcpHttpException} maps its HTTP response code as {@code
   * ConvertHttpCodeToStatusCode} does; a {@link SocketTimeoutException} maps to {@link
   * Code#DEADLINE_EXCEEDED_VALUE}, an {@link InvalidRequestException} (malformed or non-HTTPS URL,
   * cookie handler installed) to {@link Code#INVALID_ARGUMENT_VALUE}, and any other {@link
   * IOException} (connection failures, DNS, response too large) to {@link Code#UNAVAILABLE_VALUE},
   * matching {@code HttpRequestHandleImpl}.
   */
  public static int toRpcStatusCode(IOException error) {
    if (error instanceof FcpHttpException httpException) {
      return convertHttpCodeToRpcCode(httpException.responseCode());
    }
    if (error instanceof InvalidRequestException) {
      return Code.INVALID_ARGUMENT_VALUE;
    }
    // Same mapping as HttpRequestHandleImpl.getStatus for transport failures.
    if (error instanceof SocketTimeoutException) {
      return Code.DEADLINE_EXCEEDED_VALUE;
    }
    return Code.UNAVAILABLE_VALUE;
  }

  private static int convertHttpCodeToRpcCode(int code) {
    return switch (code) {
      case 400 -> Code.INVALID_ARGUMENT_VALUE;
      case 401 -> Code.UNAUTHENTICATED_VALUE;
      case 403 -> Code.PERMISSION_DENIED_VALUE;
      case 404 -> Code.NOT_FOUND_VALUE;
      case 409 -> Code.ABORTED_VALUE;
      case 429 -> Code.RESOURCE_EXHAUSTED_VALUE;
      case 499 -> Code.CANCELLED_VALUE;
      case 501 -> Code.UNIMPLEMENTED_VALUE;
      case 503 -> Code.UNAVAILABLE_VALUE;
      case 504 -> Code.DEADLINE_EXCEEDED_VALUE;
      default -> {
        if (code >= 400 && code < 500) {
          yield Code.FAILED_PRECONDITION_VALUE;
        }
        if (code >= 500 && code < 600) {
          yield Code.INTERNAL_VALUE;
        }
        yield Code.UNKNOWN_VALUE;
      }
    };
  }
}
