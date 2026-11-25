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
package com.google.fcp.client.confidentialcompute.internal.encode;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.fcp.client.confidentialcompute.PayloadEncodingException;
import com.google.fcp.client.confidentialcompute.TimestampedInput;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.ExtensionRegistryLite;
import tensorflow_federated.aggregation.Tensor.DataType;
import tensorflow_federated.aggregation.Tensor.TensorProto;
import java.io.ByteArrayInputStream;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link FederatedComputeCheckpointBuilder}. */
@RunWith(JUnit4.class)
public final class FederatedComputeCheckpointBuilderTest {

  private static final byte[] expectedHeader = "FCv1".getBytes(UTF_8);

  @Test
  public void buildCheckpoint_emptyList_serializesHeaderAndFooter() throws Exception {
    byte[] checkpoint =
        FederatedComputeCheckpointBuilder.buildCheckpoint(ImmutableList.of(), ImmutableMap.of())
            .toByteArray();

    assertThat(checkpoint.length).isAtLeast(expectedHeader.length + 4);

    byte[] header = Arrays.copyOf(checkpoint, expectedHeader.length);
    assertThat(header).isEqualTo(expectedHeader);

    byte[] footer = Arrays.copyOfRange(checkpoint, checkpoint.length - 4, checkpoint.length);
    assertThat(footer).isEqualTo(new byte[4]);

    CodedInputStream cis =
        CodedInputStream.newInstance(
            new ByteArrayInputStream(
                checkpoint, expectedHeader.length, checkpoint.length - expectedHeader.length - 4));

    // 1. Check 'entry' Tensor
    long nameLen = cis.readUInt64();
    String name = new String(cis.readRawBytes((int) nameLen), UTF_8);
    assertThat(name).isEqualTo("entry");

    long tensorLen = cis.readUInt64();
    byte[] tensorBytes = cis.readRawBytes((int) tensorLen);
    TensorProto entriesTensor =
        TensorProto.parseFrom(tensorBytes, ExtensionRegistryLite.getEmptyRegistry());
    assertThat(entriesTensor.getDtype()).isEqualTo(DataType.DT_STRING);
    assertThat(entriesTensor.getShape().getDimSizes(0)).isEqualTo(0);

    // 2. Check 'confidential_compute_event_time' Tensor
    long timeNameLen = cis.readUInt64();
    String timeName = new String(cis.readRawBytes((int) timeNameLen), UTF_8);
    assertThat(timeName).isEqualTo("confidential_compute_event_time");

    long timeTensorLen = cis.readUInt64();
    byte[] timeTensorBytes = cis.readRawBytes((int) timeTensorLen);
    TensorProto eventTimesTensor =
        TensorProto.parseFrom(timeTensorBytes, ExtensionRegistryLite.getEmptyRegistry());
    assertThat(eventTimesTensor.getDtype()).isEqualTo(DataType.DT_STRING);
    assertThat(eventTimesTensor.getShape().getDimSizes(0)).isEqualTo(0);

    assertThat(cis.isAtEnd()).isTrue();
  }

  @Test
  public void buildCheckpoint_validEntries_serializesCorrectly() throws Exception {
    ImmutableList<TimestampedInput> entries =
        ImmutableList.of(
            TimestampedInput.of(ByteString.copyFromUtf8("cat"), "2026-05-26T12:00:00Z"),
            TimestampedInput.of(ByteString.copyFromUtf8("dog"), "2026-05-26T12:01:00Z"));

    byte[] checkpoint =
        FederatedComputeCheckpointBuilder.buildCheckpoint(entries, ImmutableMap.of()).toByteArray();

    CodedInputStream cis =
        CodedInputStream.newInstance(
            new ByteArrayInputStream(
                checkpoint, expectedHeader.length, checkpoint.length - expectedHeader.length - 4));

    // 1. Parse 'entry'
    long nameLen = cis.readUInt64();
    String name = new String(cis.readRawBytes((int) nameLen), UTF_8);
    assertThat(name).isEqualTo("entry");

    long tensorLen = cis.readUInt64();
    byte[] tensorBytes = cis.readRawBytes((int) tensorLen);
    TensorProto entriesTensor =
        TensorProto.parseFrom(tensorBytes, ExtensionRegistryLite.getEmptyRegistry());
    assertThat(entriesTensor.getDtype()).isEqualTo(DataType.DT_STRING);
    assertThat(entriesTensor.getShape().getDimSizes(0)).isEqualTo(2);

    CodedInputStream contentCis = entriesTensor.getContent().newCodedInput();
    long catSize = contentCis.readUInt64();
    long dogSize = contentCis.readUInt64();
    assertThat(catSize).isEqualTo(3);
    assertThat(dogSize).isEqualTo(3);

    byte[] catBytes = contentCis.readRawBytes(3);
    byte[] dogBytes = contentCis.readRawBytes(3);
    assertThat(new String(catBytes, UTF_8)).isEqualTo("cat");
    assertThat(new String(dogBytes, UTF_8)).isEqualTo("dog");

    // 2. Parse 'confidential_compute_event_time' as DT_STRING
    long timeNameLen = cis.readUInt64();
    String timeName = new String(cis.readRawBytes((int) timeNameLen), UTF_8);
    assertThat(timeName).isEqualTo("confidential_compute_event_time");

    long timeTensorLen = cis.readUInt64();
    byte[] timeTensorBytes = cis.readRawBytes((int) timeTensorLen);
    TensorProto eventTimesTensor =
        TensorProto.parseFrom(timeTensorBytes, ExtensionRegistryLite.getEmptyRegistry());
    assertThat(eventTimesTensor.getDtype()).isEqualTo(DataType.DT_STRING);
    assertThat(eventTimesTensor.getShape().getDimSizes(0)).isEqualTo(2);

    CodedInputStream timeContentCis = eventTimesTensor.getContent().newCodedInput();
    long t1Len = timeContentCis.readUInt64();
    long t2Len = timeContentCis.readUInt64();
    assertThat(t1Len).isEqualTo(20);
    assertThat(t2Len).isEqualTo(20);
    byte[] t1Bytes = timeContentCis.readRawBytes(20);
    byte[] t2Bytes = timeContentCis.readRawBytes(20);
    assertThat(new String(t1Bytes, UTF_8)).isEqualTo("2026-05-26T12:00:00Z");
    assertThat(new String(t2Bytes, UTF_8)).isEqualTo("2026-05-26T12:01:00Z");

    assertThat(cis.isAtEnd()).isTrue();
  }

  @Test
  public void buildCheckpoint_planOutputVectors_namesTensorsAfterOutputNames() throws Exception {
    ImmutableList<TimestampedInput> entries =
        ImmutableList.of(TimestampedInput.of(ByteString.copyFromUtf8("cat"), "2026-05-26T12:00Z"));
    // SQL data upload plans set both output name and vector name to "<query_name>/<column>".
    ImmutableMap<String, String> outputVectorNames =
        ImmutableMap.of(
            "upload_query/entry", "upload_query/entry",
            "upload_query/confidential_compute_event_time",
                "upload_query/confidential_compute_event_time");

    ImmutableMap<String, TensorProto> tensors =
        readTensors(FederatedComputeCheckpointBuilder.buildCheckpoint(entries, outputVectorNames));

    assertThat(tensors.keySet())
        .containsExactly("upload_query/entry", "upload_query/confidential_compute_event_time");
    ImmutableMap<String, TensorProto> bareTensors =
        readTensors(FederatedComputeCheckpointBuilder.buildCheckpoint(entries, ImmutableMap.of()));
    assertThat(tensors.get("upload_query/entry")).isEqualTo(bareTensors.get("entry"));
    assertThat(tensors.get("upload_query/confidential_compute_event_time"))
        .isEqualTo(bareTensors.get("confidential_compute_event_time"));
  }

  @Test
  public void buildCheckpoint_outputNameDiffersFromVectorName_usesOutputName() throws Exception {
    ImmutableList<TimestampedInput> entries =
        ImmutableList.of(TimestampedInput.of(ByteString.copyFromUtf8("cat"), "2026-05-26T12:00Z"));

    ImmutableMap<String, TensorProto> tensors =
        readTensors(
            FederatedComputeCheckpointBuilder.buildCheckpoint(
                entries,
                ImmutableMap.of(
                    "q/out_entry", "q/entry",
                    "q/out_time", "confidential_compute_event_time")));

    assertThat(tensors.keySet()).containsExactly("q/out_entry", "q/out_time");
  }

  @Test
  public void buildCheckpoint_columnNotInPlan_isDropped() throws Exception {
    ImmutableList<TimestampedInput> entries =
        ImmutableList.of(TimestampedInput.of(ByteString.copyFromUtf8("cat"), "2026-05-26T12:00Z"));

    ImmutableMap<String, TensorProto> tensors =
        readTensors(
            FederatedComputeCheckpointBuilder.buildCheckpoint(
                entries, ImmutableMap.of("upload_query/entry", "upload_query/entry")));

    assertThat(tensors.keySet()).containsExactly("upload_query/entry");
  }

  @Test
  public void buildCheckpoint_planOutputVectorIsNotAColumn_throws() {
    ImmutableList<TimestampedInput> entries =
        ImmutableList.of(TimestampedInput.of(ByteString.copyFromUtf8("cat"), "2026-05-26T12:00Z"));

    PayloadEncodingException e =
        assertThrows(
            PayloadEncodingException.class,
            () ->
                FederatedComputeCheckpointBuilder.buildCheckpoint(
                    entries,
                    ImmutableMap.of(
                        "upload_query/entry", "upload_query/entry",
                        "upload_query/other", "upload_query/other")));
    assertThat(e).hasMessageThat().contains("upload_query/other");
  }

  @Test
  public void buildCheckpoint_vectorNameOnlySharesSuffix_throws() {
    ImmutableList<TimestampedInput> entries =
        ImmutableList.of(TimestampedInput.of(ByteString.copyFromUtf8("cat"), "2026-05-26T12:00Z"));

    // "my_entry" ends in "entry" but not in "/entry", so it is not the entry column.
    assertThrows(
        PayloadEncodingException.class,
        () ->
            FederatedComputeCheckpointBuilder.buildCheckpoint(
                entries, ImmutableMap.of("q/my_entry", "q/my_entry")));
  }

  @Test
  public void buildCheckpoint_nullArguments_throwsNullPointerException() {
    assertThrows(
        NullPointerException.class,
        () -> FederatedComputeCheckpointBuilder.buildCheckpoint(null, ImmutableMap.of()));
    assertThrows(
        NullPointerException.class,
        () -> FederatedComputeCheckpointBuilder.buildCheckpoint(ImmutableList.of(), null));
    List<TimestampedInput> inputsWithNull = Collections.singletonList(null);
    assertThrows(
        NullPointerException.class,
        () -> FederatedComputeCheckpointBuilder.buildCheckpoint(inputsWithNull, ImmutableMap.of()));
  }

  /** Parses an FCv1 checkpoint into its tensors, keyed by name, in order. */
  private static ImmutableMap<String, TensorProto> readTensors(ByteString checkpoint)
      throws Exception {
    byte[] bytes = checkpoint.toByteArray();
    assertThat(Arrays.copyOf(bytes, expectedHeader.length)).isEqualTo(expectedHeader);
    CodedInputStream cis =
        CodedInputStream.newInstance(
            new ByteArrayInputStream(
                bytes, expectedHeader.length, bytes.length - expectedHeader.length - 4));
    ImmutableMap.Builder<String, TensorProto> tensors = ImmutableMap.builder();
    while (!cis.isAtEnd()) {
      String name = new String(cis.readRawBytes((int) cis.readUInt64()), UTF_8);
      tensors.put(
          name,
          TensorProto.parseFrom(
              cis.readRawBytes((int) cis.readUInt64()), ExtensionRegistryLite.getEmptyRegistry()));
    }
    return tensors.buildOrThrow();
  }
}
