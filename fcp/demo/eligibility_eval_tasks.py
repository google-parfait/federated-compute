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
"""Action handlers for the EligibilityEvalTasks service."""

import datetime
import http
from typing import Callable
import uuid

from absl import logging

from google.rpc import code_pb2
from fcp.demo import http_actions
from fcp.protos.federatedcompute import common_pb2
from fcp.protos.federatedcompute import eligibility_eval_tasks_pb2


class Service:
  """Implements the EligibilityEvalTasks service."""

  def __init__(self, population_name: str,
               forwarding_info: Callable[[], common_pb2.ForwardingInfo]):
    self._population_name = population_name
    self._forwarding_info = forwarding_info

  @http_actions.proto_action(
      service='google.internal.federatedcompute.v1.EligibilityEvalTasks',
      method='RequestEligibilityEvalTask')
  def request_eligibility_eval_task(
      self, request: eligibility_eval_tasks_pb2.EligibilityEvalTaskRequest
  ) -> eligibility_eval_tasks_pb2.EligibilityEvalTaskResponse:
    """Handles a RequestEligibilityEvalTask request."""
    if request.population_name != self._population_name:
      raise http_actions.HttpError(http.HTTPStatus.NOT_FOUND)
    session_id = str(uuid.uuid4())
    logging.debug('[%s] RequestEligibilityEvalTask', session_id)

    # NOTE: A production implementation should vary the retry windows based on
    # the population size and other factors, as described in TFLaS §2.3.
    retry_window = common_pb2.RetryWindow()
    retry_window.delay_min.FromTimedelta(datetime.timedelta(seconds=10))
    retry_window.delay_max.FromTimedelta(datetime.timedelta(seconds=30))

    # This implementation does not support Eligibility Eval tasks, so we always
    # return NoEligibilityEvalConfigured.
    return eligibility_eval_tasks_pb2.EligibilityEvalTaskResponse(
        task_assignment_forwarding_info=self._forwarding_info(),
        session_id=str(uuid.uuid4()),
        no_eligibility_eval_configured=(
            eligibility_eval_tasks_pb2.NoEligibilityEvalConfigured()),
        retry_window_if_accepted=retry_window,
        retry_window_if_rejected=retry_window)

  @http_actions.proto_action(
      service='google.internal.federatedcompute.v1.EligibilityEvalTasks',
      method='ReportEligibilityEvalTaskResult')
  def report_eligibility_eval_task_result(
      self,
      request: eligibility_eval_tasks_pb2.ReportEligibilityEvalTaskResultRequest
  ) -> eligibility_eval_tasks_pb2.ReportEligibilityEvalTaskResultResponse:
    """Handles a ReportEligibilityEvalTaskResult request."""
    if request.population_name != self._population_name:
      raise http_actions.HttpError(http.HTTPStatus.NOT_FOUND)
    # NOTE: A production implementation should collect and report metrics. In
    # this implementation, we simply log the result.
    logging.log(
        logging.DEBUG if request.status_code == code_pb2.OK else logging.WARN,
        '[%s] ReportEligibilityEvalTaskResult: %s', request.session_id,
        code_pb2.Code.Name(request.status_code))
    return eligibility_eval_tasks_pb2.ReportEligibilityEvalTaskResultResponse()
