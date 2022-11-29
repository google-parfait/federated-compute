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
"""Tests for aggregations."""

import asyncio
import http
import unittest
from unittest import mock

from absl.testing import absltest

from fcp.demo import aggregations
from fcp.demo import http_actions
from fcp.demo import media
from fcp.demo import plan_utils
from fcp.protos import plan_pb2
from fcp.protos.federatedcompute import aggregations_pb2
from fcp.protos.federatedcompute import common_pb2

AGGREGATION_REQUIREMENTS = aggregations.AggregationRequirements(
    minimum_clients_in_server_published_aggregate=3, plan=plan_pb2.Plan())
FORWARDING_INFO = common_pb2.ForwardingInfo(
    target_uri_prefix='https://forwarding.example/')


class AggregationsTest(absltest.TestCase, unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.mock_media_service = self.enter_context(
        mock.patch.object(media, 'Service', autospec=True))
    self.mock_media_service.register_upload.return_value = 'upload-id'
    self.mock_media_service.finalize_upload.return_value = b'data'

    self.mock_session_ctor = self.enter_context(
        mock.patch.object(
            plan_utils, 'IntermediateAggregationSession', autospec=True))
    self.mock_session = self.mock_session_ctor.return_value

  def test_pre_authorize_clients(self):
    service = aggregations.Service(lambda: FORWARDING_INFO,
                                   self.mock_media_service)
    session_id = service.create_session(AGGREGATION_REQUIREMENTS)
    tokens = service.pre_authorize_clients(session_id, 3)
    self.assertLen(tokens, 3)
    # The tokens should all be unique.
    self.assertLen(set(tokens), 3)

  def test_pre_authorize_clients_with_missing_session_id(self):
    service = aggregations.Service(lambda: FORWARDING_INFO,
                                   self.mock_media_service)
    with self.assertRaises(KeyError):
      service.pre_authorize_clients('does-not-exist', 1)

  def test_pre_authorize_clients_with_bad_count(self):
    service = aggregations.Service(lambda: FORWARDING_INFO,
                                   self.mock_media_service)
    session_id = service.create_session(AGGREGATION_REQUIREMENTS)
    self.assertEmpty(service.pre_authorize_clients(session_id, 0))
    self.assertEmpty(service.pre_authorize_clients(session_id, -2))

  def test_create_session(self):
    service = aggregations.Service(lambda: FORWARDING_INFO,
                                   self.mock_media_service)
    session_id = service.create_session(AGGREGATION_REQUIREMENTS)
    self.mock_session_ctor.assert_called_once_with(
        AGGREGATION_REQUIREMENTS.plan)
    self.assertEqual(
        service.get_session_status(session_id),
        aggregations.SessionStatus(
            status=aggregations.AggregationStatus.PENDING,
            num_clients_completed=0,
            num_clients_failed=0,
            num_clients_pending=0,
            num_clients_aborted=0,
            num_inputs_aggregated_and_included=0,
            num_inputs_aggregated_and_pending=0,
            num_inputs_discarded=0))

  def test_complete_session(self):
    service = aggregations.Service(lambda: FORWARDING_INFO,
                                   self.mock_media_service)
    session_id = service.create_session(AGGREGATION_REQUIREMENTS)

    # Upload results from the client.
    num_clients = (
        AGGREGATION_REQUIREMENTS.minimum_clients_in_server_published_aggregate)
    for i in range(num_clients):
      tokens = service.pre_authorize_clients(session_id, 1)

      self.mock_media_service.register_upload.return_value = f'upload-{i}'
      operation = service.start_aggregation_data_upload(
          aggregations_pb2.StartAggregationDataUploadRequest(
              aggregation_id=session_id, authorization_token=tokens[0]))
      self.assertNotEmpty(operation.name)
      self.assertTrue(operation.done)

      metadata = aggregations_pb2.StartAggregationDataUploadMetadata()
      operation.metadata.Unpack(metadata)
      self.assertEqual(metadata,
                       aggregations_pb2.StartAggregationDataUploadMetadata())

      start_upload_response = (
          aggregations_pb2.StartAggregationDataUploadResponse())
      operation.response.Unpack(start_upload_response)
      self.assertEqual(
          start_upload_response,
          aggregations_pb2.StartAggregationDataUploadResponse(
              aggregation_protocol_forwarding_info=FORWARDING_INFO,
              resource=common_pb2.ByteStreamResource(
                  data_upload_forwarding_info=FORWARDING_INFO,
                  resource_name=(
                      self.mock_media_service.register_upload.return_value))))

      self.mock_media_service.finalize_upload.return_value = f'data{i}'.encode()
      submit_response = service.submit_aggregation_result(
          aggregations_pb2.SubmitAggregationResultRequest(
              aggregation_id=session_id,
              client_token=tokens[0],
              resource_name=start_upload_response.resource.resource_name))
      self.assertEqual(submit_response,
                       aggregations_pb2.SubmitAggregationResultResponse())
      self.mock_media_service.finalize_upload.assert_called_with(
          start_upload_response.resource.resource_name)
      self.mock_session.accumulate_client_update.assert_called_with(
          self.mock_media_service.finalize_upload.return_value)

    # Now that all clients have contributed, the aggregation session can be
    # completed.
    self.mock_session.finalize.return_value = b'aggregate-result'
    status, aggregate = service.complete_session(session_id)
    self.assertEqual(
        status,
        aggregations.SessionStatus(
            status=aggregations.AggregationStatus.COMPLETED,
            num_clients_completed=num_clients,
            num_clients_failed=0,
            num_clients_pending=0,
            num_clients_aborted=0,
            num_inputs_aggregated_and_included=num_clients,
            num_inputs_aggregated_and_pending=0,
            num_inputs_discarded=0))
    self.assertEqual(aggregate, self.mock_session.finalize.return_value)
    self.mock_session.finalize.assert_called_once()

    # get_session_status should no longer return results.
    with self.assertRaises(KeyError):
      service.get_session_status(session_id)

  def test_complete_session_without_enough_inputs(self):
    service = aggregations.Service(lambda: FORWARDING_INFO,
                                   self.mock_media_service)
    session_id = service.create_session(
        aggregations.AggregationRequirements(
            minimum_clients_in_server_published_aggregate=3,
            plan=AGGREGATION_REQUIREMENTS.plan))
    tokens = service.pre_authorize_clients(session_id, 2)

    # Upload results for one client.
    operation = service.start_aggregation_data_upload(
        aggregations_pb2.StartAggregationDataUploadRequest(
            aggregation_id=session_id, authorization_token=tokens[0]))
    self.assertTrue(operation.done)
    start_upload_response = (
        aggregations_pb2.StartAggregationDataUploadResponse())
    operation.response.Unpack(start_upload_response)
    service.submit_aggregation_result(
        aggregations_pb2.SubmitAggregationResultRequest(
            aggregation_id=session_id,
            client_token=tokens[0],
            resource_name=start_upload_response.resource.resource_name))

    # Complete the session before there are 2 completed clients.
    status, aggregate = service.complete_session(session_id)
    self.assertEqual(
        status,
        aggregations.SessionStatus(
            status=aggregations.AggregationStatus.FAILED,
            num_clients_completed=1,
            num_clients_failed=0,
            num_clients_pending=0,
            num_clients_aborted=0,
            num_inputs_aggregated_and_included=0,
            num_inputs_aggregated_and_pending=0,
            num_inputs_discarded=1))
    self.assertIsNone(aggregate)

  def test_complete_session_with_missing_session_id(self):
    service = aggregations.Service(lambda: FORWARDING_INFO,
                                   self.mock_media_service)
    with self.assertRaises(KeyError):
      service.complete_session('does-not-exist')

  def test_abort_session_with_no_uploads(self):
    service = aggregations.Service(lambda: FORWARDING_INFO,
                                   self.mock_media_service)
    session_id = service.create_session(AGGREGATION_REQUIREMENTS)
    self.assertEqual(
        service.abort_session(session_id),
        aggregations.SessionStatus(
            status=aggregations.AggregationStatus.ABORTED,
            num_clients_completed=0,
            num_clients_failed=0,
            num_clients_pending=0,
            num_clients_aborted=0,
            num_inputs_aggregated_and_included=0,
            num_inputs_aggregated_and_pending=0,
            num_inputs_discarded=0))

  def test_abort_session_with_uploads(self):
    service = aggregations.Service(lambda: FORWARDING_INFO,
                                   self.mock_media_service)
    session_id = service.create_session(AGGREGATION_REQUIREMENTS)
    tokens = service.pre_authorize_clients(session_id, 3)

    # Upload results for one client.
    self.mock_media_service.register_upload.return_value = 'upload1'
    service.start_aggregation_data_upload(
        aggregations_pb2.StartAggregationDataUploadRequest(
            aggregation_id=session_id, authorization_token=tokens[0]))
    service.submit_aggregation_result(
        aggregations_pb2.SubmitAggregationResultRequest(
            aggregation_id=session_id,
            client_token=tokens[0],
            resource_name='upload1'))

    # Start a partial upload from a second client.
    self.mock_media_service.register_upload.return_value = 'upload2'
    service.start_aggregation_data_upload(
        aggregations_pb2.StartAggregationDataUploadRequest(
            aggregation_id=session_id, authorization_token=tokens[1]))

    # Abort the session. The pending client should be treated as failed.
    self.assertEqual(
        service.abort_session(session_id),
        aggregations.SessionStatus(
            status=aggregations.AggregationStatus.ABORTED,
            num_clients_completed=1,
            num_clients_failed=0,
            num_clients_pending=0,
            num_clients_aborted=1,
            num_inputs_aggregated_and_included=0,
            num_inputs_aggregated_and_pending=0,
            num_inputs_discarded=1))
    # The registered upload for the second client should have been finalized.
    self.mock_media_service.finalize_upload.assert_called_with('upload2')
    # get_session_status should no longer return results.
    with self.assertRaises(KeyError):
      service.get_session_status(session_id)

  def test_abort_session_with_missing_session_id(self):
    service = aggregations.Service(lambda: FORWARDING_INFO,
                                   self.mock_media_service)
    with self.assertRaises(KeyError):
      service.abort_session('does-not-exist')

  def test_get_session_status_with_missing_session_id(self):
    service = aggregations.Service(lambda: FORWARDING_INFO,
                                   self.mock_media_service)
    with self.assertRaises(KeyError):
      service.get_session_status('does-not-exist')

  async def test_wait(self):
    service = aggregations.Service(lambda: FORWARDING_INFO,
                                   self.mock_media_service)
    session_id = service.create_session(AGGREGATION_REQUIREMENTS)
    task = asyncio.create_task(
        service.wait(session_id, num_inputs_aggregated_and_pending=1))
    # The awaitable should not be done yet.
    await asyncio.wait([task], timeout=0.1)
    self.assertFalse(task.done())

    # Upload results for one client.
    tokens = service.pre_authorize_clients(session_id, 1)
    self.mock_media_service.register_upload.return_value = 'upload'
    service.start_aggregation_data_upload(
        aggregations_pb2.StartAggregationDataUploadRequest(
            aggregation_id=session_id, authorization_token=tokens[0]))
    service.submit_aggregation_result(
        aggregations_pb2.SubmitAggregationResultRequest(
            aggregation_id=session_id,
            client_token=tokens[0],
            resource_name='upload'))

    # The awaitable should now return.
    await asyncio.wait([task], timeout=1)
    self.assertTrue(task.done())
    self.assertEqual(
        task.result(),
        aggregations.SessionStatus(
            status=aggregations.AggregationStatus.PENDING,
            num_clients_completed=1,
            num_clients_failed=0,
            num_clients_pending=0,
            num_clients_aborted=0,
            num_inputs_aggregated_and_included=0,
            num_inputs_aggregated_and_pending=1,
            num_inputs_discarded=0))

  async def test_wait_already_satisfied(self):
    service = aggregations.Service(lambda: FORWARDING_INFO,
                                   self.mock_media_service)
    session_id = service.create_session(AGGREGATION_REQUIREMENTS)

    # Upload results for one client.
    tokens = service.pre_authorize_clients(session_id, 1)
    self.mock_media_service.register_upload.return_value = 'upload'
    service.start_aggregation_data_upload(
        aggregations_pb2.StartAggregationDataUploadRequest(
            aggregation_id=session_id, authorization_token=tokens[0]))
    service.submit_aggregation_result(
        aggregations_pb2.SubmitAggregationResultRequest(
            aggregation_id=session_id,
            client_token=tokens[0],
            resource_name='upload'))

    # Since a client has already reported, the condition should already be
    # satisfied.
    task = asyncio.create_task(
        service.wait(session_id, num_inputs_aggregated_and_pending=1))
    await asyncio.wait([task], timeout=1)
    self.assertTrue(task.done())
    self.assertEqual(
        task.result(),
        aggregations.SessionStatus(
            status=aggregations.AggregationStatus.PENDING,
            num_clients_completed=1,
            num_clients_failed=0,
            num_clients_pending=0,
            num_inputs_aggregated_and_included=0,
            num_inputs_aggregated_and_pending=1,
            num_inputs_discarded=0))

  async def test_wait_with_abort(self):
    service = aggregations.Service(lambda: FORWARDING_INFO,
                                   self.mock_media_service)
    session_id = service.create_session(AGGREGATION_REQUIREMENTS)
    task = asyncio.create_task(
        service.wait(session_id, num_inputs_aggregated_and_pending=1))
    # The awaitable should not be done yet.
    await asyncio.wait([task], timeout=0.1)
    self.assertFalse(task.done())

    # The awaitable should return once the session is aborted.
    status = service.abort_session(session_id)
    await asyncio.wait([task], timeout=1)
    self.assertTrue(task.done())
    self.assertEqual(task.result(), status)

  async def test_wait_with_complete(self):
    service = aggregations.Service(lambda: FORWARDING_INFO,
                                   self.mock_media_service)
    session_id = service.create_session(
        aggregations.AggregationRequirements(
            minimum_clients_in_server_published_aggregate=0,
            plan=AGGREGATION_REQUIREMENTS.plan))
    task = asyncio.create_task(
        service.wait(session_id, num_inputs_aggregated_and_pending=1))
    # The awaitable should not be done yet.
    await asyncio.wait([task], timeout=0.1)
    self.assertFalse(task.done())

    # The awaitable should return once the session is completed.
    status, _ = service.complete_session(session_id)
    await asyncio.wait([task], timeout=1)
    self.assertTrue(task.done())
    self.assertEqual(task.result(), status)

  async def test_wait_without_condition(self):
    service = aggregations.Service(lambda: FORWARDING_INFO,
                                   self.mock_media_service)
    session_id = service.create_session(AGGREGATION_REQUIREMENTS)
    task = asyncio.create_task(service.wait(session_id))
    # If there are no conditions, the wait should be trivially satisfied.
    await asyncio.wait([task], timeout=1)
    self.assertTrue(task.done())
    self.assertEqual(
        task.result(),
        aggregations.SessionStatus(
            status=aggregations.AggregationStatus.PENDING,
            num_clients_completed=0,
            num_clients_failed=0,
            num_clients_pending=0,
            num_clients_aborted=0,
            num_inputs_aggregated_and_included=0,
            num_inputs_aggregated_and_pending=0,
            num_inputs_discarded=0))

  async def test_wait_with_missing_session_id(self):
    service = aggregations.Service(lambda: FORWARDING_INFO,
                                   self.mock_media_service)
    task = asyncio.create_task(service.wait('does-not-exist'))
    await asyncio.wait([task], timeout=1)
    self.assertTrue(task.done())
    self.assertIsInstance(task.exception(), KeyError)

  def test_start_aggregagation_data_upload_with_missing_session_id(self):
    service = aggregations.Service(lambda: FORWARDING_INFO,
                                   self.mock_media_service)
    session_id = service.create_session(AGGREGATION_REQUIREMENTS)
    tokens = service.pre_authorize_clients(session_id, 1)
    with self.assertRaises(http_actions.HttpError) as cm:
      service.start_aggregation_data_upload(
          aggregations_pb2.StartAggregationDataUploadRequest(
              aggregation_id='does-not-exist', authorization_token=tokens[0]))
    self.assertEqual(cm.exception.code, http.HTTPStatus.NOT_FOUND)
    self.assertEqual(
        service.get_session_status(session_id),
        aggregations.SessionStatus(
            status=aggregations.AggregationStatus.PENDING,
            num_clients_completed=0,
            num_clients_failed=0,
            num_clients_pending=0,
            num_clients_aborted=0,
            num_inputs_aggregated_and_included=0,
            num_inputs_aggregated_and_pending=0,
            num_inputs_discarded=0))

  def test_start_aggregagation_data_upload_with_invalid_token(self):
    service = aggregations.Service(lambda: FORWARDING_INFO,
                                   self.mock_media_service)
    session_id = service.create_session(AGGREGATION_REQUIREMENTS)
    with self.assertRaises(http_actions.HttpError) as cm:
      service.start_aggregation_data_upload(
          aggregations_pb2.StartAggregationDataUploadRequest(
              aggregation_id=session_id, authorization_token='does-not-exist'))
    self.assertEqual(cm.exception.code, http.HTTPStatus.UNAUTHORIZED)
    self.assertEqual(
        service.get_session_status(session_id),
        aggregations.SessionStatus(
            status=aggregations.AggregationStatus.PENDING,
            num_clients_completed=0,
            num_clients_failed=0,
            num_clients_pending=0,
            num_clients_aborted=0,
            num_inputs_aggregated_and_included=0,
            num_inputs_aggregated_and_pending=0,
            num_inputs_discarded=0))

  def test_start_aggregagation_data_upload_twice(self):
    service = aggregations.Service(lambda: FORWARDING_INFO,
                                   self.mock_media_service)
    session_id = service.create_session(AGGREGATION_REQUIREMENTS)
    tokens = service.pre_authorize_clients(session_id, 1)
    service.start_aggregation_data_upload(
        aggregations_pb2.StartAggregationDataUploadRequest(
            aggregation_id=session_id, authorization_token=tokens[0]))
    with self.assertRaises(http_actions.HttpError) as cm:
      service.start_aggregation_data_upload(
          aggregations_pb2.StartAggregationDataUploadRequest(
              aggregation_id=session_id, authorization_token=tokens[0]))
    self.assertEqual(cm.exception.code, http.HTTPStatus.BAD_REQUEST)
    self.assertEqual(
        service.get_session_status(session_id),
        aggregations.SessionStatus(
            status=aggregations.AggregationStatus.PENDING,
            num_clients_completed=0,
            num_clients_failed=0,
            num_clients_pending=1,
            num_clients_aborted=0,
            num_inputs_aggregated_and_included=0,
            num_inputs_aggregated_and_pending=0,
            num_inputs_discarded=0))

  def test_submit_aggregation_result_with_missing_session_id(self):
    service = aggregations.Service(lambda: FORWARDING_INFO,
                                   self.mock_media_service)
    session_id = service.create_session(AGGREGATION_REQUIREMENTS)
    tokens = service.pre_authorize_clients(session_id, 1)
    with self.assertRaises(http_actions.HttpError) as cm:
      service.submit_aggregation_result(
          aggregations_pb2.SubmitAggregationResultRequest(
              aggregation_id='does-not-exist',
              client_token=tokens[0],
              resource_name='upload-id'))
    self.assertEqual(cm.exception.code, http.HTTPStatus.NOT_FOUND)
    self.assertEqual(
        service.get_session_status(session_id),
        aggregations.SessionStatus(
            status=aggregations.AggregationStatus.PENDING,
            num_clients_completed=0,
            num_clients_failed=0,
            num_clients_pending=0,
            num_clients_aborted=0,
            num_inputs_aggregated_and_included=0,
            num_inputs_aggregated_and_pending=0,
            num_inputs_discarded=0))

  def test_submit_aggregation_result_with_invalid_token(self):
    service = aggregations.Service(lambda: FORWARDING_INFO,
                                   self.mock_media_service)
    session_id = service.create_session(AGGREGATION_REQUIREMENTS)
    with self.assertRaises(http_actions.HttpError) as cm:
      service.submit_aggregation_result(
          aggregations_pb2.SubmitAggregationResultRequest(
              aggregation_id=session_id,
              client_token='does-not-exist',
              resource_name='upload-id'))
    self.assertEqual(cm.exception.code, http.HTTPStatus.UNAUTHORIZED)
    self.assertEqual(
        service.get_session_status(session_id),
        aggregations.SessionStatus(
            status=aggregations.AggregationStatus.PENDING,
            num_clients_completed=0,
            num_clients_failed=0,
            num_clients_pending=0,
            num_clients_aborted=0,
            num_inputs_aggregated_and_included=0,
            num_inputs_aggregated_and_pending=0,
            num_inputs_discarded=0))

  def test_submit_aggregation_result_without_start_upload(self):
    service = aggregations.Service(lambda: FORWARDING_INFO,
                                   self.mock_media_service)
    session_id = service.create_session(AGGREGATION_REQUIREMENTS)
    tokens = service.pre_authorize_clients(session_id, 1)
    # The request should fail if the client never called
    # StartAggregationDataUpload.
    with self.assertRaises(http_actions.HttpError) as cm:
      service.submit_aggregation_result(
          aggregations_pb2.SubmitAggregationResultRequest(
              aggregation_id=session_id,
              client_token=tokens[0],
              resource_name='upload-id'))
    self.mock_media_service.finalize_upload.assert_not_called()
    self.assertEqual(cm.exception.code, http.HTTPStatus.BAD_REQUEST)
    self.assertEqual(
        service.get_session_status(session_id),
        aggregations.SessionStatus(
            status=aggregations.AggregationStatus.PENDING,
            num_clients_completed=0,
            num_clients_failed=0,
            num_clients_pending=0,
            num_clients_aborted=0,
            num_inputs_aggregated_and_included=0,
            num_inputs_aggregated_and_pending=0,
            num_inputs_discarded=0))

  def test_submit_aggregation_result_with_invalid_resource_name(self):
    service = aggregations.Service(lambda: FORWARDING_INFO,
                                   self.mock_media_service)
    session_id = service.create_session(AGGREGATION_REQUIREMENTS)
    tokens = service.pre_authorize_clients(session_id, 1)
    service.start_aggregation_data_upload(
        aggregations_pb2.StartAggregationDataUploadRequest(
            aggregation_id=session_id, authorization_token=tokens[0]))

    # If the resource_name doesn't correspond to a registered upload,
    # finalize_upload will raise a KeyError.
    self.mock_media_service.finalize_upload.side_effect = KeyError()
    with self.assertRaises(http_actions.HttpError) as cm:
      service.submit_aggregation_result(
          aggregations_pb2.SubmitAggregationResultRequest(
              aggregation_id=session_id,
              client_token=tokens[0],
              resource_name='upload-id'))
    self.assertEqual(cm.exception.code, http.HTTPStatus.BAD_REQUEST)
    self.assertEqual(
        service.get_session_status(session_id),
        aggregations.SessionStatus(
            status=aggregations.AggregationStatus.PENDING,
            num_clients_completed=0,
            num_clients_failed=1,
            num_clients_pending=0,
            num_clients_aborted=0,
            num_inputs_aggregated_and_included=0,
            num_inputs_aggregated_and_pending=0,
            num_inputs_discarded=0))

  def test_submit_aggregation_result_with_unuploaded_resource(self):
    service = aggregations.Service(lambda: FORWARDING_INFO,
                                   self.mock_media_service)
    session_id = service.create_session(AGGREGATION_REQUIREMENTS)
    tokens = service.pre_authorize_clients(session_id, 1)
    operation = service.start_aggregation_data_upload(
        aggregations_pb2.StartAggregationDataUploadRequest(
            aggregation_id=session_id, authorization_token=tokens[0]))
    self.assertTrue(operation.done)
    start_upload_response = (
        aggregations_pb2.StartAggregationDataUploadResponse())
    operation.response.Unpack(start_upload_response)

    # If the resource_name is valid but no resource was uploaded,
    # finalize_resource will return None.
    self.mock_media_service.finalize_upload.return_value = None
    with self.assertRaises(http_actions.HttpError) as cm:
      service.submit_aggregation_result(
          aggregations_pb2.SubmitAggregationResultRequest(
              aggregation_id=session_id,
              client_token=tokens[0],
              resource_name=start_upload_response.resource.resource_name))
    self.assertEqual(cm.exception.code, http.HTTPStatus.BAD_REQUEST)
    self.assertEqual(
        service.get_session_status(session_id),
        aggregations.SessionStatus(
            status=aggregations.AggregationStatus.PENDING,
            num_clients_completed=0,
            num_clients_failed=1,
            num_clients_pending=0,
            num_clients_aborted=0,
            num_inputs_aggregated_and_included=0,
            num_inputs_aggregated_and_pending=0,
            num_inputs_discarded=0))

  def test_abort_aggregation(self):
    service = aggregations.Service(lambda: FORWARDING_INFO,
                                   self.mock_media_service)
    session_id = service.create_session(AGGREGATION_REQUIREMENTS)
    tokens = service.pre_authorize_clients(session_id, 1)
    service.start_aggregation_data_upload(
        aggregations_pb2.StartAggregationDataUploadRequest(
            aggregation_id=session_id, authorization_token=tokens[0]))
    self.assertEqual(
        service.abort_aggregation(
            aggregations_pb2.AbortAggregationRequest(
                aggregation_id=session_id, client_token=tokens[0])),
        aggregations_pb2.AbortAggregationResponse())
    self.assertEqual(
        service.get_session_status(session_id),
        aggregations.SessionStatus(
            status=aggregations.AggregationStatus.PENDING,
            num_clients_completed=0,
            num_clients_failed=1,
            num_clients_pending=0,
            num_clients_aborted=0,
            num_inputs_aggregated_and_included=0,
            num_inputs_aggregated_and_pending=0,
            num_inputs_discarded=0))

  def test_abort_aggregation_with_missing_session_id(self):
    service = aggregations.Service(lambda: FORWARDING_INFO,
                                   self.mock_media_service)
    session_id = service.create_session(AGGREGATION_REQUIREMENTS)
    tokens = service.pre_authorize_clients(session_id, 1)
    service.start_aggregation_data_upload(
        aggregations_pb2.StartAggregationDataUploadRequest(
            aggregation_id=session_id, authorization_token=tokens[0]))
    with self.assertRaises(http_actions.HttpError) as cm:
      service.abort_aggregation(
          aggregations_pb2.AbortAggregationRequest(
              aggregation_id='does-not-exist', client_token=tokens[0]))
    self.assertEqual(cm.exception.code, http.HTTPStatus.NOT_FOUND)
    self.assertEqual(
        service.get_session_status(session_id),
        aggregations.SessionStatus(
            status=aggregations.AggregationStatus.PENDING,
            num_clients_completed=0,
            num_clients_failed=0,
            num_clients_pending=1,
            num_clients_aborted=0,
            num_inputs_aggregated_and_included=0,
            num_inputs_aggregated_and_pending=0,
            num_inputs_discarded=0))

  def test_abort_aggregation_with_invalid_token(self):
    service = aggregations.Service(lambda: FORWARDING_INFO,
                                   self.mock_media_service)
    session_id = service.create_session(AGGREGATION_REQUIREMENTS)
    with self.assertRaises(http_actions.HttpError) as cm:
      service.abort_aggregation(
          aggregations_pb2.AbortAggregationRequest(
              aggregation_id=session_id, client_token='does-not-exist'))
    self.assertEqual(cm.exception.code, http.HTTPStatus.UNAUTHORIZED)
    self.assertEqual(
        service.get_session_status(session_id),
        aggregations.SessionStatus(
            status=aggregations.AggregationStatus.PENDING,
            num_clients_completed=0,
            num_clients_failed=0,
            num_clients_pending=0,
            num_clients_aborted=0,
            num_inputs_aggregated_and_included=0,
            num_inputs_aggregated_and_pending=0,
            num_inputs_discarded=0))

  def test_abort_aggregation_without_start(self):
    service = aggregations.Service(lambda: FORWARDING_INFO,
                                   self.mock_media_service)
    session_id = service.create_session(AGGREGATION_REQUIREMENTS)
    tokens = service.pre_authorize_clients(session_id, 1)
    with self.assertRaises(http_actions.HttpError) as cm:
      service.abort_aggregation(
          aggregations_pb2.AbortAggregationRequest(
              aggregation_id=session_id, client_token=tokens[0]))
    self.assertEqual(cm.exception.code, http.HTTPStatus.BAD_REQUEST)
    self.assertEqual(
        service.get_session_status(session_id),
        aggregations.SessionStatus(
            status=aggregations.AggregationStatus.PENDING,
            num_clients_completed=0,
            num_clients_failed=0,
            num_clients_pending=0,
            num_clients_aborted=0,
            num_inputs_aggregated_and_included=0,
            num_inputs_aggregated_and_pending=0,
            num_inputs_discarded=0))


if __name__ == '__main__':
  absltest.main()
