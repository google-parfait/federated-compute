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

import com.google.common.collect.ImmutableMap;
import com.google.fcp.client.confidentialcompute.PayloadEncodingException;
import com.google.fcp.client.confidentialcompute.TimestampedInput;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedOutputStream;
import tensorflow_federated.aggregation.Tensor.DataType;
import tensorflow_federated.aggregation.Tensor.TensorProto;
import tensorflow_federated.aggregation.Tensor.TensorShapeProto;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Serializes {@link TimestampedInput} instances into the Federated Compute Checkpoint (FCv1)
 * format.
 *
 * <p>Mirrors {@code FederatedComputeCheckpointBuilder} in {@code
 * tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.cc}.
 */
final class FederatedComputeCheckpointBuilder {

  static final String ENTRIES_KEY = "entry";

  static final String EVENT_TIME_FEATURE_NAME = "confidential_compute_event_time";

  private static final ByteString HEADER = ByteString.copyFromUtf8("FCv1");
  private static final ByteString FOOTER = ByteString.copyFrom(new byte[4]);

  private FederatedComputeCheckpointBuilder() {}

  /**
   * Serializes {@code inputs} into an FCv1 checkpoint: the {@code "FCv1"} header, one {@code
   * DT_STRING} tensor record for the entries and one for the event times, and a 4-byte zero footer.
   *
   * <p>The tensors are named after the plan's output vectors, as {@code ExampleQueryPlanEngine}
   * does for PrivateLogger results in {@code fcp/client/engine/example_query_plan_engine.cc}: each
   * output vector whose vector name is {@link #ENTRIES_KEY} or {@link #EVENT_TIME_FEATURE_NAME}, or
   * ends in {@code /} followed by one of them, gets that column under its output name (for example
   * {@code upload_query/entry}). If {@code outputVectorNames} is empty the bare column names are
   * used.
   *
   * @param outputVectorNames the plan's output vectors, as output name to vector name; see {@link
   *     com.google.fcp.client.confidentialcompute.LegacyFcpCompat#outputVectorNames()}
   * @throws PayloadEncodingException if an output vector matches neither column
   */
  static ByteString buildCheckpoint(
      List<? extends TimestampedInput> inputs, Map<String, String> outputVectorNames)
      throws IOException, PayloadEncodingException {
    Objects.requireNonNull(inputs);
    Objects.requireNonNull(outputVectorNames);

    int count = inputs.size();
    TensorShapeProto shape = TensorShapeProto.newBuilder().addDimSizes(count).build();

    List<ByteString> entries = new ArrayList<>(count);
    List<ByteString> eventTimes = new ArrayList<>(count);
    for (TimestampedInput input : inputs) {
      Objects.requireNonNull(input);
      entries.add(Objects.requireNonNull(input.value()));
      eventTimes.add(ByteString.copyFromUtf8(Objects.requireNonNull(input.timestamp())));
    }
    ImmutableMap<String, TensorProto> columns =
        ImmutableMap.of(
            ENTRIES_KEY, createStringTensor(shape, entries),
            EVENT_TIME_FEATURE_NAME, createStringTensor(shape, eventTimes));

    ImmutableMap<String, TensorProto> namedTensors = nameTensors(columns, outputVectorNames);

    int totalSize = HEADER.size() + FOOTER.size();
    for (Map.Entry<String, TensorProto> tensor : namedTensors.entrySet()) {
      totalSize += CodedOutputStream.computeStringSizeNoTag(tensor.getKey());
      int tensorSize = tensor.getValue().getSerializedSize();
      totalSize += CodedOutputStream.computeUInt64SizeNoTag(tensorSize);
      totalSize += tensorSize;
    }

    ByteString.Output bos = ByteString.newOutput(totalSize);
    CodedOutputStream cos = CodedOutputStream.newInstance(bos);
    cos.writeRawBytes(HEADER);
    for (Map.Entry<String, TensorProto> tensor : namedTensors.entrySet()) {
      // varint64(name length) || name || varint64(tensor size) || tensor, matching
      // FederatedComputeCheckpointBuilder::Add in the C++ client. writeStringNoTag writes a
      // varint32 length, which encodes a non-negative length the same way as varint64.
      cos.writeStringNoTag(tensor.getKey());
      cos.writeUInt64NoTag(tensor.getValue().getSerializedSize());
      tensor.getValue().writeTo(cos);
    }
    cos.writeRawBytes(FOOTER);
    cos.flush();

    return bos.toByteString();
  }

  /**
   * Returns the tensors to write, keyed by tensor name. Without output vectors the columns keep
   * their own names. Otherwise each output vector takes the column its vector name matches, and a
   * column that no output vector asks for is dropped, as in C++.
   */
  private static ImmutableMap<String, TensorProto> nameTensors(
      ImmutableMap<String, TensorProto> columns, Map<String, String> outputVectorNames)
      throws PayloadEncodingException {
    if (outputVectorNames.isEmpty()) {
      return columns;
    }
    ImmutableMap.Builder<String, TensorProto> named = ImmutableMap.builder();
    for (Map.Entry<String, String> outputVector : outputVectorNames.entrySet()) {
      String vectorName = outputVector.getValue();
      TensorProto column = null;
      for (Map.Entry<String, TensorProto> candidate : columns.entrySet()) {
        if (vectorName.equals(candidate.getKey())
            || vectorName.endsWith("/" + candidate.getKey())) {
          column = candidate.getValue();
          break;
        }
      }
      if (column == null) {
        throw new PayloadEncodingException(
            "Plan output vector " + vectorName + " is not a PrivateLogger column");
      }
      named.put(outputVector.getKey(), column);
    }
    return named.buildOrThrow();
  }

  /**
   * Encodes a 1-D {@code DT_STRING} {@link TensorProto} in TFF aggregation format (all varint64
   * byte lengths followed by all raw string payloads, matching {@code StringDataAdapter} in {@code
   * tensorflow_federated/cc/core/impl/aggregation/core/tensor.cc}).
   */
  private static TensorProto createStringTensor(TensorShapeProto shape, List<ByteString> values)
      throws IOException {
    int contentSize = 0;
    for (ByteString value : values) {
      contentSize += CodedOutputStream.computeUInt64SizeNoTag(value.size());
      contentSize += value.size();
    }
    ByteString.Output bos = ByteString.newOutput(contentSize);
    CodedOutputStream cos = CodedOutputStream.newInstance(bos);
    for (ByteString value : values) {
      cos.writeUInt64NoTag(value.size());
    }
    for (ByteString value : values) {
      cos.writeRawBytes(value);
    }
    cos.flush();
    return TensorProto.newBuilder()
        .setDtype(DataType.DT_STRING)
        .setShape(shape)
        .setContent(bos.toByteString())
        .build();
  }
}
