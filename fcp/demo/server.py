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
"""An in-process federated compute server."""

import contextlib
import gzip
import http.server
import socket
import socketserver
import ssl
from typing import Optional

from absl import logging

from fcp.demo import aggregations
from fcp.demo import eligibility_eval_tasks
from fcp.demo import http_actions
from fcp.demo import media
from fcp.demo import plan_utils
from fcp.demo import task_assignments
from fcp.protos import plan_pb2
from fcp.protos.federatedcompute import common_pb2


class InProcessServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
  """An in-process HTTP server implementing the Federated Compute protocol."""

  def __init__(self,
               *,
               population_name: str,
               host: str,
               port: int,
               address_family: Optional[socket.AddressFamily] = None):
    self._media_service = media.Service(self._get_forwarding_info)
    self._aggregations_service = aggregations.Service(self._get_forwarding_info,
                                                      self._media_service)
    self._task_assignments_service = task_assignments.Service(
        population_name, self._get_forwarding_info, self._aggregations_service)
    eligibility_eval_tasks_service = eligibility_eval_tasks.Service(
        population_name, self._get_forwarding_info)
    handler = http_actions.create_handler(self._media_service,
                                          self._aggregations_service,
                                          self._task_assignments_service,
                                          eligibility_eval_tasks_service)
    if address_family is not None:
      self.address_family = address_family
    http.server.HTTPServer.__init__(self, (host, port), handler)

  async def run_computation(self, task_name: str, plan: plan_pb2.Plan,
                            server_checkpoint: Optional[bytes],
                            number_of_clients: int) -> bytes:
    """Runs a computation, returning the resulting checkpoint.

    If there's already a computation in progress, the new computation will
    not start until the previous one has completed (either successfully or not).

    Args:
      task_name: The name of the task.
      plan: The Plan proto containing the client and server computations.
      server_checkpoint: The starting server checkpoint, if any.
      number_of_clients: The minimum number of clients to include.

    Returns:
      A TensorFlow checkpoint containing the aggregated results.
    """
    requirements = aggregations.AggregationRequirements(
        minimum_clients_in_server_published_aggregate=number_of_clients,
        plan=plan)
    session_id = self._aggregations_service.create_session(requirements)
    with contextlib.ExitStack() as stack:
      stack.callback(
          lambda: self._aggregations_service.abort_session(session_id))
      with plan_utils.Session(plan, server_checkpoint) as session:
        with self._media_service.register_download(
            gzip.compress(session.client_plan),
            content_type='application/x-protobuf+gzip'
        ) as plan_url, self._media_service.register_download(
            gzip.compress(session.client_checkpoint),
            content_type='application/octet-stream+gzip') as checkpoint_url:
          self._task_assignments_service.add_task(
              task_name, session_id, common_pb2.Resource(uri=plan_url),
              common_pb2.Resource(uri=checkpoint_url))
          try:
            status = await self._aggregations_service.wait(
                session_id,
                num_inputs_aggregated_and_included=number_of_clients)
            if status.status != aggregations.AggregationStatus.PENDING:
              raise ValueError('Aggregation failed.')
          finally:
            self._task_assignments_service.remove_task(session_id)

          stack.pop_all()
          status, intermedia_update = (
              self._aggregations_service.complete_session(session_id))
          if (status.status != aggregations.AggregationStatus.COMPLETED or
              intermedia_update is None):
            raise ValueError('Aggregation failed.')
        logging.debug('%s aggregation complete: %s', task_name, status)
        return session.finalize(intermedia_update)

  def _get_forwarding_info(self) -> common_pb2.ForwardingInfo:
    protocol = 'https' if isinstance(self.socket, ssl.SSLSocket) else 'http'
    return common_pb2.ForwardingInfo(
        target_uri_prefix=(
            f'{protocol}://{self.server_name}:{self.server_port}/'))
