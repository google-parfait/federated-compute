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
"""Temporary command-line interface for running the demo server."""

import asyncio
import ssl
import threading
from typing import Optional, Sequence

from absl import app
from absl import flags
from absl import logging

from fcp.demo import server
from fcp.protos import plan_pb2

_POPULATION_NAME = flags.DEFINE_string('population_name', 'test/population',
                                       'Population name')
_HOST = flags.DEFINE_string('host', 'localhost', 'Server hostname')
_PORT = flags.DEFINE_integer('port', 0, 'Server port')
_CERTFILE = flags.DEFINE_string('certfile', None,
                                'Path to the certificate to use for https')
_KEYFILE = flags.DEFINE_string(
    'keyfile', None, 'Path to the certificate\'s private key (if separate)')
_TASK_NAME = flags.DEFINE_string('task_name', 'example',
                                 'Name of the task to run')
_PLAN = flags.DEFINE_string('plan', None, 'Path to file containing Plan proto')
_CHECKPOINT = flags.DEFINE_string(
    'checkpoint', None, 'Path to file containing initial server checkpoint')
_NUM_ROUNDS = flags.DEFINE_integer('num_rounds', 1, 'Number of rounds to run')
_NUM_CLIENTS = flags.DEFINE_integer('num_clients', 1,
                                    'Number of clients per round')
_OUTPUT = flags.DEFINE_string('output', None,
                              'Path to file receiving the final checkpoint')


def _run_task(s: server.InProcessServer, task_name: str, plan: plan_pb2.Plan,
              checkpoint: Optional[bytes], num_clients: int,
              num_rounds: int) -> Optional[bytes]:
  """Helper function to run a task and wait for the result."""
  loop = asyncio.new_event_loop()
  successful_rounds = 0
  while successful_rounds < num_rounds:
    try:
      checkpoint = loop.run_until_complete(
          s.run_computation(task_name, plan, checkpoint, num_clients))
      logging.info('Round %d complete', successful_rounds)
      successful_rounds += 1
    except ValueError as e:
      logging.warn('Round %d failed: %s', successful_rounds, e)
  return checkpoint


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  with open(_PLAN.value, 'rb') as f:
    plan = plan_pb2.Plan.FromString(f.read())

  checkpoint = None
  if _CHECKPOINT.value:
    with open(_CHECKPOINT.value, 'rb') as f:
      checkpoint = f.read()

  with server.InProcessServer(
      population_name=_POPULATION_NAME.value,
      host=_HOST.value,
      port=_PORT.value) as s:
    if _CERTFILE.value:
      context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
      context.load_cert_chain(certfile=_CERTFILE.value, keyfile=_KEYFILE.value)
      s.socket = context.wrap_socket(s.socket, server_side=True)
    thread = threading.Thread(target=s.serve_forever, daemon=True)
    thread.start()
    logging.info('Running on %s:%d', s.server_name, s.server_port)

    try:
      checkpoint = _run_task(s, _TASK_NAME.value, plan, checkpoint,
                             _NUM_CLIENTS.value, _NUM_ROUNDS.value)
    finally:
      s.shutdown()
      thread.join()

  if checkpoint and _OUTPUT.value:
    with open(_OUTPUT.value, 'wb') as f:
      f.write(checkpoint)


if __name__ == '__main__':
  app.run(main)
