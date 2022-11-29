# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Action handlers for the Aggregations service."""

import asyncio
import contextlib
import copy
import dataclasses
import enum
import functools
import http
import threading
from typing import Callable, Dict, Optional, Sequence, Set, Tuple
import uuid

from absl import logging

from google.longrunning import operations_pb2
from fcp.demo import http_actions
from fcp.demo import media
from fcp.demo import plan_utils
from fcp.protos import plan_pb2
from fcp.protos.federatedcompute import aggregations_pb2
from fcp.protos.federatedcompute import common_pb2


class AggregationStatus(enum.Enum):
  COMPLETED = 1
  PENDING = 2
  FAILED = 3
  ABORTED = 4


@dataclasses.dataclass
class SessionStatus:
  """The status of an aggregation session."""
  # The current state of the aggregation session.
  status: AggregationStatus = AggregationStatus.PENDING
  # Number of clients that successfully started and completed the aggregation
  # upload protocol.
  num_clients_completed: int = 0
  # Number of clients that started the aggregation upload protocol but failed
  # to complete (e.g dropped out in the middle of the protocol).
  num_clients_failed: int = 0
  # Number of clients that started the aggregation upload protocol but have not
  # yet finished (either successfully or not).
  num_clients_pending: int = 0
  # Number of clients that started the aggregation protocol but were aborted by
  # the server before they could complete (e.g., if progress on the session was
  # no longer needed).
  num_clients_aborted: int = 0
  # Number of inputs that were successfully aggregated and included in the
  # final aggregate. Note that even if a client successfully completes the
  # protocol (i.e., it is included in num_clients_completed), it is not
  # guaranteed that the uploaded report is included in the final aggregate yet.
  num_inputs_aggregated_and_included: int = 0
  # Number of inputs that were received by the server and are pending (i.e.,
  # the inputs have not been included in the final aggregate yet).
  num_inputs_aggregated_and_pending: int = 0
  # Number of inputs that were received by the server but discarded.
  num_inputs_discarded: int = 0


@dataclasses.dataclass(frozen=True)
class AggregationRequirements:
  # The minimum number of clients required before a result can be released
  # outside this service. Note that aggregation does not automatically stop if
  # minimum_clients_in_server_published_aggregate is met. It is up to callers
  # to stop aggregation when they want.
  minimum_clients_in_server_published_aggregate: int
  # The Plan to execute.
  plan: plan_pb2.Plan


@dataclasses.dataclass(eq=False)
class _WaitData:
  """Information about a pending wait operation."""
  # The condition under which the wait should complete.
  num_inputs_aggregated_and_pending: Optional[int]
  # The loop the caller is waiting on.
  loop: asyncio.AbstractEventLoop = dataclasses.field(
      default_factory=asyncio.get_running_loop)
  # The queue to which the SessionStatus will be written once the wait is over.
  queue: asyncio.Queue = dataclasses.field(default_factory=asyncio.Queue)


@dataclasses.dataclass(eq=False)
class _AggregationSessionState:
  """Internal state for an aggregation session."""
  # The session's aggregation requirements.
  requirements: AggregationRequirements
  # The TensorFlow aggregation session.
  agg_session: plan_utils.IntermediateAggregationSession
  # The current status of the session.
  status: SessionStatus = dataclasses.field(default_factory=SessionStatus)
  # Tokens for active clients. Once a client has contributed, its token is
  # removed.
  client_tokens: Set[str] = dataclasses.field(default_factory=set)
  # Media service handles for pending uploads, keyed by client token.
  pending_uploads: Dict[str, str] = dataclasses.field(default_factory=dict)
  # Information for in-progress wait calls on this session.
  pending_waits: Set[_WaitData] = dataclasses.field(default_factory=set)


class Service:
  """Implements the Aggregations service."""

  def __init__(self, forwarding_info: Callable[[], common_pb2.ForwardingInfo],
               media_service: media.Service):
    self._forwarding_info = forwarding_info
    self._media_service = media_service
    self._sessions: Dict[str, _AggregationSessionState] = {}
    self._sessions_lock = threading.Lock()

  def create_session(self,
                     aggregation_requirements: AggregationRequirements) -> str:
    """Creates a new aggregation session and returns its id."""
    session_id = str(uuid.uuid4())
    agg_session = plan_utils.IntermediateAggregationSession(
        aggregation_requirements.plan)

    with self._sessions_lock:
      self._sessions[session_id] = _AggregationSessionState(
          requirements=aggregation_requirements, agg_session=agg_session)
    return session_id

  def complete_session(
      self, session_id: str) -> Tuple[SessionStatus, Optional[bytes]]:
    """Completes the aggregation session and returns its results."""
    with self._sessions_lock:
      state = self._sessions.pop(session_id)

    try:
      if (state.status.num_inputs_aggregated_and_pending <
          state.requirements.minimum_clients_in_server_published_aggregate):
        raise ValueError(
            'minimum_clients_in_server_published_aggregate has not been met.')
      result = state.agg_session.finalize()
      state.status.status = AggregationStatus.COMPLETED
      state.status.num_inputs_aggregated_and_included += (
          state.status.num_inputs_aggregated_and_pending)
      state.status.num_inputs_aggregated_and_pending = 0
      return state.status, result
    except ValueError as e:
      logging.warning('Failed to finalize aggregation session: %s', e)
      state.status.status = AggregationStatus.FAILED
      return state.status, None
    finally:
      self._cleanup_session(state)

  def abort_session(self, session_id: str) -> SessionStatus:
    """Aborts/cancels an aggregation session."""
    with self._sessions_lock:
      state = self._sessions.pop(session_id)
    state.status.status = AggregationStatus.ABORTED
    # Cleaning up the session will mark all pending clients as aborted and all
    # pending inputs as discarded. Since no results will be returned, all
    # completed inputs should also be considered discarded.
    state.status.num_inputs_discarded += (
        state.status.num_inputs_aggregated_and_included)
    state.status.num_inputs_aggregated_and_included = 0
    self._cleanup_session(state)
    return state.status

  def get_session_status(self, session_id: str) -> SessionStatus:
    """Returns the status of an aggregation session."""
    with self._sessions_lock:
      return copy.deepcopy(self._sessions[session_id].status)

  async def wait(
      self,
      session_id: str,
      num_inputs_aggregated_and_pending: Optional[int] = None) -> SessionStatus:
    """Blocks until all conditions are satisfied or the aggregation fails."""
    with self._sessions_lock:
      state = self._sessions[session_id]
      # Check if any of the conditions are already satisfied.
      if (num_inputs_aggregated_and_pending is None or
          num_inputs_aggregated_and_pending <=
          state.status.num_inputs_aggregated_and_pending):
        return copy.deepcopy(state.status)

      wait_data = _WaitData(num_inputs_aggregated_and_pending)
      state.pending_waits.add(wait_data)
    return await wait_data.queue.get()

  def pre_authorize_clients(self, session_id: str,
                            num_tokens: int) -> Sequence[str]:
    """Generates tokens authorizing clients to contribute to the session."""
    tokens = [str(uuid.uuid4()) for _ in range(num_tokens)]
    with self._sessions_lock:
      self._sessions[session_id].client_tokens |= set(tokens)
    return tokens

  def _cleanup_session(self, state: _AggregationSessionState) -> None:
    """Cleans up the session and releases any resources.

    Any pending clients are considered failed.

    Args:
      state: The session state to clean up.
    """
    state.agg_session.close()
    state.status.num_clients_aborted += state.status.num_clients_pending
    state.status.num_clients_pending = 0
    state.status.num_inputs_discarded += (
        state.status.num_inputs_aggregated_and_pending)
    state.status.num_inputs_aggregated_and_pending = 0
    state.client_tokens.clear()
    for name in state.pending_uploads.values():
      self._media_service.finalize_upload(name)
    state.pending_uploads.clear()
    for data in state.pending_waits:
      data.loop.call_soon_threadsafe(
          functools.partial(data.queue.put_nowait, state.status))
    state.pending_waits.clear()

  @http_actions.proto_action(
      service='google.internal.federatedcompute.v1.Aggregations',
      method='StartAggregationDataUpload')
  def start_aggregation_data_upload(
      self, request: aggregations_pb2.StartAggregationDataUploadRequest
  ) -> operations_pb2.Operation:
    """Handles a StartAggregationDataUpload request."""
    with self._sessions_lock:
      try:
        state = self._sessions[request.aggregation_id]
      except KeyError as e:
        raise http_actions.HttpError(http.HTTPStatus.NOT_FOUND) from e
      if request.authorization_token not in state.client_tokens:
        raise http_actions.HttpError(http.HTTPStatus.UNAUTHORIZED)
      if request.authorization_token in state.pending_uploads:
        # The client should not have already called StartAggregationDataUpload.
        raise http_actions.HttpError(http.HTTPStatus.BAD_REQUEST)
      upload_name = self._media_service.register_upload()
      state.pending_uploads[request.authorization_token] = upload_name
      state.status.num_clients_pending += 1

    forwarding_info = self._forwarding_info()
    response = aggregations_pb2.StartAggregationDataUploadResponse(
        aggregation_protocol_forwarding_info=forwarding_info,
        resource=common_pb2.ByteStreamResource(
            data_upload_forwarding_info=forwarding_info,
            resource_name=upload_name))

    op = operations_pb2.Operation(name=f'operations/{uuid.uuid4()}', done=True)
    op.metadata.Pack(aggregations_pb2.StartAggregationDataUploadMetadata())
    op.response.Pack(response)
    return op

  @http_actions.proto_action(
      service='google.internal.federatedcompute.v1.Aggregations',
      method='SubmitAggregationResult')
  def submit_aggregation_result(
      self, request: aggregations_pb2.SubmitAggregationResultRequest
  ) -> aggregations_pb2.SubmitAggregationResultResponse:
    """Handles a SubmitAggregationResult request."""
    with self._sessions_lock:
      try:
        state = self._sessions[request.aggregation_id]
      except KeyError as e:
        raise http_actions.HttpError(http.HTTPStatus.NOT_FOUND) from e
      if request.client_token not in state.client_tokens:
        raise http_actions.HttpError(http.HTTPStatus.UNAUTHORIZED)
      # Ensure the client is using the resource name provided when they called
      # StartAggregationDataUpload.
      if (request.resource_name != state.pending_uploads.get(
          request.client_token)):
        raise http_actions.HttpError(http.HTTPStatus.BAD_REQUEST)

      def record_failure():
        state.status.num_clients_pending -= 1
        state.status.num_clients_failed += 1

      # Mark the client session as complete; the client will need to be
      # recorded as either completed or failed.
      state.client_tokens.remove(request.client_token)
      with contextlib.ExitStack() as stack:
        stack.callback(record_failure)
        try:
          update = self._media_service.finalize_upload(request.resource_name)
        except KeyError as e:
          raise http_actions.HttpError(http.HTTPStatus.BAD_REQUEST) from e
        finally:
          del state.pending_uploads[request.client_token]
        # Ensure data was actually uploaded.
        if update is None:
          raise http_actions.HttpError(http.HTTPStatus.BAD_REQUEST)

        try:
          state.agg_session.accumulate_client_update(update)
        except ValueError as e:
          raise http_actions.HttpError(http.HTTPStatus.BAD_REQUEST) from e

        # Record the successful aggregation.
        stack.pop_all()
        state.status.num_clients_pending -= 1
        state.status.num_clients_completed += 1
        state.status.num_inputs_aggregated_and_pending += 1

        # Check for any newly-satisfied pending wait operations.
        completed_waits = set()
        for data in state.pending_waits:
          if (data.num_inputs_aggregated_and_pending is not None and
              state.status.num_inputs_aggregated_and_pending >=
              data.num_inputs_aggregated_and_pending):
            data.loop.call_soon_threadsafe(
                functools.partial(data.queue.put_nowait, state.status))
            completed_waits.add(data)
        state.pending_waits -= completed_waits
    return aggregations_pb2.SubmitAggregationResultResponse()

  @http_actions.proto_action(
      service='google.internal.federatedcompute.v1.Aggregations',
      method='AbortAggregation')
  def abort_aggregation(
      self, request: aggregations_pb2.AbortAggregationRequest
  ) -> aggregations_pb2.AbortAggregationResponse:
    """Handles an AbortAggregation request."""
    with self._sessions_lock:
      try:
        state = self._sessions[request.aggregation_id]
      except KeyError as e:
        raise http_actions.HttpError(http.HTTPStatus.NOT_FOUND) from e
      try:
        state.client_tokens.remove(request.client_token)
      except KeyError as e:
        raise http_actions.HttpError(http.HTTPStatus.UNAUTHORIZED) from e
      try:
        del state.pending_uploads[request.client_token]
      except KeyError as e:
        raise http_actions.HttpError(http.HTTPStatus.BAD_REQUEST) from e

      state.status.num_clients_pending -= 1
      state.status.num_clients_failed += 1
    logging.debug('[%s] AbortAggregation: %s', request.aggregation_id,
                  request.status)
    return aggregations_pb2.AbortAggregationResponse()
