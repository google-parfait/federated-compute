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
"""Tests for task_assignments."""

import http
from unittest import mock
import uuid

from absl.testing import absltest

from google.rpc import code_pb2
from fcp.demo import aggregations
from fcp.demo import http_actions
from fcp.demo import task_assignments
from fcp.protos.federatedcompute import common_pb2
from fcp.protos.federatedcompute import task_assignments_pb2

POPULATION_NAME = 'test/population'
FORWARDING_INFO = common_pb2.ForwardingInfo(
    target_uri_prefix='https://forwarding.example/')


class TaskAssignmentsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_aggregations_service = self.enter_context(
        mock.patch.object(aggregations, 'Service', autospec=True))
    self.mock_aggregations_service.pre_authorize_clients.return_value = ['']

  def test_start_task_assignment_with_wrong_population(self):
    service = task_assignments.Service(POPULATION_NAME, lambda: FORWARDING_INFO,
                                       self.mock_aggregations_service)
    request = task_assignments_pb2.StartTaskAssignmentRequest(
        population_name='other/population', session_id='session-id')
    with self.assertRaises(http_actions.HttpError) as cm:
      service.start_task_assignment(request)
    self.assertEqual(cm.exception.code, http.HTTPStatus.NOT_FOUND)

  @mock.patch.object(uuid, 'uuid4', return_value=uuid.uuid4(), autospec=True)
  def test_start_task_assignment_with_no_tasks(self, mock_uuid):
    service = task_assignments.Service(POPULATION_NAME, lambda: FORWARDING_INFO,
                                       self.mock_aggregations_service)
    request = task_assignments_pb2.StartTaskAssignmentRequest(
        population_name=POPULATION_NAME, session_id='session-id')
    operation = service.start_task_assignment(request)
    self.assertEqual(operation.name, f'operations/{mock_uuid.return_value}')
    self.assertTrue(operation.done)

    metadata = task_assignments_pb2.StartTaskAssignmentMetadata()
    operation.metadata.Unpack(metadata)
    self.assertEqual(metadata,
                     task_assignments_pb2.StartTaskAssignmentMetadata())

    response = task_assignments_pb2.StartTaskAssignmentResponse()
    operation.response.Unpack(response)
    self.assertEqual(
        response,
        task_assignments_pb2.StartTaskAssignmentResponse(
            rejection_info=common_pb2.RejectionInfo()))

  @mock.patch.object(uuid, 'uuid4', return_value=uuid.uuid4(), autospec=True)
  def test_start_task_assignment_with_tasks(self, mock_uuid):
    service = task_assignments.Service(POPULATION_NAME, lambda: FORWARDING_INFO,
                                       self.mock_aggregations_service)

    self.mock_aggregations_service.pre_authorize_clients.return_value = [
        'token'
    ]

    task_plan = common_pb2.Resource(uri='https://task.example/plan')
    task_checkpoint = common_pb2.Resource(uri='https://task.example/checkpoint')
    service.add_task('task', 'aggregation-session', task_plan, task_checkpoint)

    request = task_assignments_pb2.StartTaskAssignmentRequest(
        population_name=POPULATION_NAME, session_id='session-id')
    operation = service.start_task_assignment(request)
    self.assertEqual(operation.name, f'operations/{mock_uuid.return_value}')
    self.assertTrue(operation.done)

    metadata = task_assignments_pb2.StartTaskAssignmentMetadata()
    operation.metadata.Unpack(metadata)
    self.assertEqual(metadata,
                     task_assignments_pb2.StartTaskAssignmentMetadata())

    response = task_assignments_pb2.StartTaskAssignmentResponse()
    operation.response.Unpack(response)
    self.assertEqual(
        response,
        task_assignments_pb2.StartTaskAssignmentResponse(
            task_assignment=task_assignments_pb2.TaskAssignment(
                aggregation_data_forwarding_info=FORWARDING_INFO,
                aggregation_info=(
                    task_assignments_pb2.TaskAssignment.AggregationInfo()),
                session_id=request.session_id,
                aggregation_id='aggregation-session',
                client_token='token',
                task_name='task',
                plan=task_plan,
                init_checkpoint=task_checkpoint)))

    self.mock_aggregations_service.pre_authorize_clients.assert_called_once_with(
        'aggregation-session', num_tokens=1)

  def test_multiple_tasks(self):
    service = task_assignments.Service(POPULATION_NAME, lambda: FORWARDING_INFO,
                                       self.mock_aggregations_service)

    self.mock_aggregations_service.pre_authorize_clients.return_value = [
        'token'
    ]

    task1_plan = common_pb2.Resource(uri='https://task1.example/plan')
    task1_checkpoint = common_pb2.Resource(
        uri='https://task1.example/checkpoint')
    service.add_task('task1', 'aggregation-session1', task1_plan,
                     task1_checkpoint)
    task2_plan = common_pb2.Resource(uri='https://task2.example/plan')
    task2_checkpoint = common_pb2.Resource(
        uri='https://task2.example/checkpoint')
    service.add_task('task2', 'aggregation-session2', task2_plan,
                     task2_checkpoint)

    request = task_assignments_pb2.StartTaskAssignmentRequest(
        population_name=POPULATION_NAME, session_id='session-id')

    # Initially, task1 should be used.
    operation = service.start_task_assignment(request)
    response = task_assignments_pb2.StartTaskAssignmentResponse()
    operation.response.Unpack(response)
    self.assertEqual(
        response,
        task_assignments_pb2.StartTaskAssignmentResponse(
            task_assignment=task_assignments_pb2.TaskAssignment(
                aggregation_data_forwarding_info=FORWARDING_INFO,
                aggregation_info=(
                    task_assignments_pb2.TaskAssignment.AggregationInfo()),
                session_id=request.session_id,
                aggregation_id='aggregation-session1',
                client_token='token',
                task_name='task1',
                plan=task1_plan,
                init_checkpoint=task1_checkpoint)))
    self.mock_aggregations_service.pre_authorize_clients.assert_called_with(
        'aggregation-session1', num_tokens=1)

    # After task1 is removed, task2 should be used.
    service.remove_task('aggregation-session1')
    operation = service.start_task_assignment(request)
    response = task_assignments_pb2.StartTaskAssignmentResponse()
    operation.response.Unpack(response)
    self.assertEqual(
        response,
        task_assignments_pb2.StartTaskAssignmentResponse(
            task_assignment=task_assignments_pb2.TaskAssignment(
                aggregation_data_forwarding_info=FORWARDING_INFO,
                aggregation_info=(
                    task_assignments_pb2.TaskAssignment.AggregationInfo()),
                session_id=request.session_id,
                aggregation_id='aggregation-session2',
                client_token='token',
                task_name='task2',
                plan=task2_plan,
                init_checkpoint=task2_checkpoint)))
    self.mock_aggregations_service.pre_authorize_clients.assert_called_with(
        'aggregation-session2', num_tokens=1)

    # After task2 is removed, the client should be rejected.
    service.remove_task('aggregation-session2')
    operation = service.start_task_assignment(request)
    response = task_assignments_pb2.StartTaskAssignmentResponse()
    operation.response.Unpack(response)
    self.assertEqual(
        response,
        task_assignments_pb2.StartTaskAssignmentResponse(
            rejection_info=common_pb2.RejectionInfo()))

  def test_remove_missing_task(self):
    service = task_assignments.Service(POPULATION_NAME, lambda: FORWARDING_INFO,
                                       self.mock_aggregations_service)
    with self.assertRaises(KeyError):
      service.remove_task('does-not-exist')

  def test_report_task_result(self):
    service = task_assignments.Service(POPULATION_NAME, lambda: FORWARDING_INFO,
                                       self.mock_aggregations_service)
    request = task_assignments_pb2.ReportTaskResultRequest(
        population_name=POPULATION_NAME,
        session_id='session-id',
        aggregation_id='aggregation-id',
        computation_status_code=code_pb2.ABORTED)
    self.assertEqual(
        service.report_task_result(request),
        task_assignments_pb2.ReportTaskResultResponse())

  def test_report_task_result_with_wrong_population(self):
    service = task_assignments.Service(POPULATION_NAME, lambda: FORWARDING_INFO,
                                       self.mock_aggregations_service)
    request = task_assignments_pb2.ReportTaskResultRequest(
        population_name='other/population',
        session_id='session-id',
        aggregation_id='aggregation-id',
        computation_status_code=code_pb2.ABORTED)
    with self.assertRaises(http_actions.HttpError) as cm:
      service.report_task_result(request)
    self.assertEqual(cm.exception.code, http.HTTPStatus.NOT_FOUND)


if __name__ == '__main__':
  absltest.main()
