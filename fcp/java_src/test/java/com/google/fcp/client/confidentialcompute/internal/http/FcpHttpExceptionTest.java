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

import static com.google.common.truth.Truth.assertThat;

import com.google.fcp.client.confidentialcompute.internal.http.FcpHttpClient.InvalidRequestException;
import com.google.rpc.Code;
import java.io.IOException;
import java.net.SocketTimeoutException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class FcpHttpExceptionTest {

  @Test
  public void toRpcStatusCode_invalidRequest_mapsToInvalidArgument() {
    assertThat(FcpHttpException.toRpcStatusCode(new InvalidRequestException("invalid request")))
        .isEqualTo(Code.INVALID_ARGUMENT_VALUE);
  }

  @Test
  public void toRpcStatusCode_standardHttpCodes() {
    assertThat(FcpHttpException.toRpcStatusCode(new FcpHttpException(400, "err")))
        .isEqualTo(Code.INVALID_ARGUMENT_VALUE);
    assertThat(FcpHttpException.toRpcStatusCode(new FcpHttpException(401, "err")))
        .isEqualTo(Code.UNAUTHENTICATED_VALUE);
    assertThat(FcpHttpException.toRpcStatusCode(new FcpHttpException(403, "err")))
        .isEqualTo(Code.PERMISSION_DENIED_VALUE);
    assertThat(FcpHttpException.toRpcStatusCode(new FcpHttpException(404, "err")))
        .isEqualTo(Code.NOT_FOUND_VALUE);
    assertThat(FcpHttpException.toRpcStatusCode(new FcpHttpException(409, "err")))
        .isEqualTo(Code.ABORTED_VALUE);
    assertThat(FcpHttpException.toRpcStatusCode(new FcpHttpException(429, "err")))
        .isEqualTo(Code.RESOURCE_EXHAUSTED_VALUE);
    assertThat(FcpHttpException.toRpcStatusCode(new FcpHttpException(499, "err")))
        .isEqualTo(Code.CANCELLED_VALUE);
    assertThat(FcpHttpException.toRpcStatusCode(new FcpHttpException(501, "err")))
        .isEqualTo(Code.UNIMPLEMENTED_VALUE);
    assertThat(FcpHttpException.toRpcStatusCode(new FcpHttpException(503, "err")))
        .isEqualTo(Code.UNAVAILABLE_VALUE);
    assertThat(FcpHttpException.toRpcStatusCode(new FcpHttpException(504, "err")))
        .isEqualTo(Code.DEADLINE_EXCEEDED_VALUE);
  }

  @Test
  public void toRpcStatusCode_other4xx_mapsToFailedPrecondition() {
    assertThat(FcpHttpException.toRpcStatusCode(new FcpHttpException(412, "err")))
        .isEqualTo(Code.FAILED_PRECONDITION_VALUE);
    assertThat(FcpHttpException.toRpcStatusCode(new FcpHttpException(422, "err")))
        .isEqualTo(Code.FAILED_PRECONDITION_VALUE);
  }

  @Test
  public void toRpcStatusCode_other5xx_mapsToInternal() {
    assertThat(FcpHttpException.toRpcStatusCode(new FcpHttpException(500, "err")))
        .isEqualTo(Code.INTERNAL_VALUE);
    assertThat(FcpHttpException.toRpcStatusCode(new FcpHttpException(502, "err")))
        .isEqualTo(Code.INTERNAL_VALUE);
  }

  @Test
  public void toRpcStatusCode_otherHttpCodes_mapsToUnknown() {
    assertThat(FcpHttpException.toRpcStatusCode(new FcpHttpException(302, "err")))
        .isEqualTo(Code.UNKNOWN_VALUE);
  }

  @Test
  public void toRpcStatusCode_genericIOException_mapsToUnavailable() {
    assertThat(FcpHttpException.toRpcStatusCode(new IOException("generic error")))
        .isEqualTo(Code.UNAVAILABLE_VALUE);
  }

  @Test
  public void toRpcStatusCode_socketTimeout_mapsToDeadlineExceeded() {
    assertThat(FcpHttpException.toRpcStatusCode(new SocketTimeoutException("timeout")))
        .isEqualTo(Code.DEADLINE_EXCEEDED_VALUE);
  }
}
