/*
 * Copyright 2019 Google LLC
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

#ifndef FCP_CLIENT_GRPC_BIDI_CHANNEL_H_
#define FCP_CLIENT_GRPC_BIDI_CHANNEL_H_

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "absl/base/attributes.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/strip.h"
#include "fcp/base/monitoring.h"
#include "grpcpp/channel.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"

namespace fcp {
namespace client {

class GrpcBidiChannel {
 public:
  static constexpr int kChannelMaxMessageSize = 20 * 1000 * 1000;
  static constexpr int kKeepAliveTimeSeconds = 60;
  static constexpr const absl::string_view kSecurePrefix = "https://";
  static constexpr const absl::string_view kSecureTestPrefix = "https+test://";

  /**
   * Create a channel to the remote endpoint.
   * @param target URI of the remote endpoint.
   * @param cert_path If the URI is https:// or https+test://, an optional
   *    path to a certificate root file or directory.
   * @return A shared pointer to the channel interface.
   */
  static std::shared_ptr<grpc::ChannelInterface> Create(
      const std::string& target, std::string cert_path) {
    bool secure = false;
    bool test = false;
    // This double check avoids a dependency on re2:
    if (absl::StartsWith(target, kSecureTestPrefix)) {
      secure = true;
      test = true;
    } else if (absl::StartsWith(target, kSecurePrefix)) {
      secure = true;
    }

    grpc::ChannelArguments channel_arguments;
    channel_arguments.SetMaxReceiveMessageSize(kChannelMaxMessageSize);
    channel_arguments.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS,
                             kKeepAliveTimeSeconds * 1000);

    if (!secure)
      return grpc::CreateCustomChannel(
          target, grpc::InsecureChannelCredentials(), channel_arguments);

    std::shared_ptr<grpc::ChannelCredentials> channel_creds;
    grpc::SslCredentialsOptions ssl_opts{};

    if (!cert_path.empty()) {
      std::ifstream cert_file(cert_path);
      FCP_LOG_IF(ERROR, cert_file.fail())
          << "Open for: " << cert_path << " failed: " << strerror(errno);
      std::stringstream string_stream;
      if (cert_file) {
        string_stream << cert_file.rdbuf();
      }
      FCP_LOG_IF(WARNING, string_stream.str().empty())
          << "Cert: " << cert_path << " is empty.";
      ssl_opts = {string_stream.str(), "", ""};
    }

    channel_creds = grpc::SslCredentials(ssl_opts);
    auto target_uri = absl::StrCat(
        "dns:///",
        absl::StripPrefix(target, test ? kSecureTestPrefix : kSecurePrefix));
    FCP_LOG(INFO) << "Creating channel to: " << target_uri;
    return CreateCustomChannel(target_uri, channel_creds, channel_arguments);
  }
};

}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_GRPC_BIDI_CHANNEL_H_
