/*
 * Copyright 2022 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FCP_AGGREGATION_PROTOCOL_AGGREGATION_PROTOCOL_H_
#define FCP_AGGREGATION_PROTOCOL_AGGREGATION_PROTOCOL_H_

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "fcp/aggregation/protocol/aggregation_protocol_messages.pb.h"

namespace fcp::aggregation {

// Describes a abstract aggregation protocol interface between a networking
// layer (e.g. a service that handles receiving and sending messages with the
// client devices) and an implementation of an aggregation algorithm.
//
// The design of the AggregationProtocol follows a Bridge Pattern
// (https://en.wikipedia.org/wiki/Bridge_pattern) in that it is meant to
// decouple an abstraction of the layers above and below the AggregationProtocol
// from the implementation.
//
// In this interface the receiving and sending contributing inputs or
// messages is abstracted from the actual mechanism for sending and receiving
// data over the network and from the actual aggregation mechanism.
//
// Client identification: the real client identities are hidden from the
// protocol implementations. Instead each client is identified by a client_id
// number in a range [0, num_clients) where num_clients is the number of clients
// the protocol started with or the extended number of clients, which is the
// sum of the starting num_clients and num_clients passed to each subsequent
// AddClients call.
//
// Thread safety: for any given client identified by a unique client_id, the
// protocol methods are expected to be called sequentially. But there are no
// assumptions about concurrent calls made for different clients. Specific
// implementations of AggregationProtocol are expected to handle concurrent
// calls. The caller side of the protocol isn't expected to queue messages.
class AggregationProtocol {
 public:
  AggregationProtocol() = default;
  virtual ~AggregationProtocol() = default;

  // Instructs the protocol to start with the specified number of clients.
  //
  // Depending on the protocol implementation, the starting number of clients
  // may be zero.  This method is guaranteed to be the first method called on
  // the protocol.
  //
  // AcceptClients callback is expected in response to this method.
  virtual absl::Status Start(int64_t num_clients) = 0;

  // Adds an additional batch of clients to the protocol.
  //
  // Depending on the protocol implementation, adding clients may not be allowed
  // and this method might return an error Status.
  //
  // AcceptClients callback is expected in response to this method.
  virtual absl::Status AddClients(int64_t num_clients) = 0;

  // Handles a message from a given client.
  //
  // Depending on the specific protocol implementation there may be multiple
  // messages exchanged with each clients.
  //
  // This method should return an error status only if there is an unrecoverable
  // error which must result in aborting the protocol.  Any client specific
  // error, like an invalid message, should result in closing the protocol with
  // that specific client only, but this method should still return OK status.
  virtual absl::Status ReceiveClientMessage(int64_t client_id,
                                            const ClientMessage& message) = 0;

  // Notifies the protocol about a communication with a given client being
  // closed, either normally or abnormally.
  //
  // The client_status indicates whether the client connection was closed
  // normally.
  //
  // No further calls or callbacks specific to the given client are expected
  // after this method.
  virtual absl::Status CloseClient(int64_t client_id,
                                   absl::Status client_status) = 0;

  // Forces the protocol to complete.
  //
  // Once the protocol has completed successfully, the Complete callback will
  // be invoked and provide the aggregation result.  If the protocol cannot be
  // completed in its current state, this method should return an error status.
  // It is also possible for the completion to fail eventually due to finishing
  // some asynchronous work, in which case the Abort callback will be invoked.
  //
  // No further protocol method calls except Abort and GetStatus are expected
  // after this method.
  virtual absl::Status Complete() = 0;

  // Forces the protocol to Abort.
  //
  // No further protocol method calls except GetStatus are expected after this
  // method.
  virtual absl::Status Abort() = 0;

  // Called periodically to receive the protocol status.
  //
  // This method can still be called after the protocol has been completed or
  // aborted.
  virtual StatusMessage GetStatus() = 0;

  // Callback interface which methods are implemented by the protocol host.
  class Callback {
   public:
    Callback() = default;
    virtual ~Callback() = default;

    // Called in response to either StartProtocol or AddClients methods being
    // called and provides protocol parameters to be broadcasted to all newly
    // joined clients.
    virtual void OnAcceptClients(int64_t start_client_id, int64_t num_clients,
                                 const AcceptanceMessage& message) = 0;

    // Called by the protocol to deliver a message to a given client.
    //
    // Depending on the specific protocol implementation there may be multiple
    // messages exchanged with each clients, but not all protocols need to
    // send messages to clients.
    virtual void OnSendServerMessage(int64_t client_id,
                                     const ServerMessage& message) = 0;

    // Called by the protocol to force communication with a client to be closed,
    // for example due to a client specific error or due to the protocol getting
    // into a state where no further input for that client is needed.
    //
    // No further calls or callbacks specific to the given client are expected
    // after this method.
    virtual void OnCloseClient(int64_t client_id,
                               absl::Status diagnostic_status) = 0;

    // Indicates successful completion of the aggregation protocol, contains
    // the result of the aggregation.
    //
    // The format of the result blob is unspecified and can be different for
    // each specific aggregation protocol implementation.  Completing the
    // protocol should close communications with all remaining clients.
    virtual void OnComplete(absl::Cord result) = 0;

    // Called by the protocol to indicate that the protocol has been aborted
    // for internal reasons (e.g. the number of remaining clients dropping
    // too low).
    //
    // Aborting the protocol should close communications with all remaining
    // clients.
    virtual void OnAbort(absl::Status diagnostic_status) = 0;
  };
};

}  // namespace fcp::aggregation

#endif  // FCP_AGGREGATION_PROTOCOL_AGGREGATION_PROTOCOL_H_
