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
#ifndef FCP_CLIENT_STATS_H_
#define FCP_CLIENT_STATS_H_

#include <cstdint>

#include "absl/time/time.h"

namespace fcp {
namespace client {

struct NetworkStats {
  // TODO(team): Remove the bytes_downloaded, bytes_uploaded, and
  // report_size_bytes fields once enable_per_phase_network_stats has rolled
  // out. And rename the chunking_layer_bytes_received/sent fields to simply
  // "bytes_downloaded" and "bytes_uploaded".

  // The estimated number of bytes downloaded from the network (after
  // decompression).
  int64_t bytes_downloaded = 0;
  // The estimated number of bytes uploaded to the network (after
  // decompression).
  int64_t bytes_uploaded = 0;
  // The estimated number of bytes downloaded from the network ("over the
  // wire").
  int64_t chunking_layer_bytes_received = 0;
  // The estimated number of bytes uploaded to the network ("over the
  // wire").
  int64_t chunking_layer_bytes_sent = 0;
  int64_t report_size_bytes = 0;

  // Returns the difference between two sets of network stats.
  NetworkStats operator-(const NetworkStats& other) const {
    return {
        .bytes_downloaded = bytes_downloaded - other.bytes_downloaded,
        .bytes_uploaded = bytes_uploaded - other.bytes_uploaded,
        .chunking_layer_bytes_received =
            chunking_layer_bytes_received - other.chunking_layer_bytes_received,
        .chunking_layer_bytes_sent =
            chunking_layer_bytes_sent - other.chunking_layer_bytes_sent,
        .report_size_bytes = report_size_bytes - other.report_size_bytes};
  }

  NetworkStats operator+(const NetworkStats& other) const {
    return {
        .bytes_downloaded = bytes_downloaded + other.bytes_downloaded,
        .bytes_uploaded = bytes_uploaded + other.bytes_uploaded,
        .chunking_layer_bytes_received =
            chunking_layer_bytes_received + other.chunking_layer_bytes_received,
        .chunking_layer_bytes_sent =
            chunking_layer_bytes_sent + other.chunking_layer_bytes_sent,
        .report_size_bytes = report_size_bytes + other.report_size_bytes};
  }
};

inline bool operator==(const NetworkStats& s1, const NetworkStats& s2) {
  return s1.bytes_downloaded == s2.bytes_downloaded &&
         s1.bytes_uploaded == s2.bytes_uploaded &&
         s1.chunking_layer_bytes_received == s2.chunking_layer_bytes_received &&
         s1.chunking_layer_bytes_sent == s2.chunking_layer_bytes_sent &&
         s1.report_size_bytes == s2.report_size_bytes;
}

struct ExampleStats {
  int example_count = 0;
  int64_t example_size_bytes = 0;
};

inline bool operator==(const ExampleStats& s1, const ExampleStats& s2) {
  return s1.example_count == s2.example_count &&
         s1.example_size_bytes == s2.example_size_bytes;
}

}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_STATS_H_
