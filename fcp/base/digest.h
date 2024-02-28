#ifndef FCP_BASE_DIGEST_H_
#define FCP_BASE_DIGEST_H_

#include <string>
#include <variant>

#include "absl/strings/cord.h"

namespace fcp {
// Returns the SHA256 hash for the given data. Note that the return value
// contains raw digest bytes, and not a human-readable hex-encoded string.
std::string ComputeSHA256(const absl::Cord& data);
// Returns the SHA256 hash for the given data. Note that the return value
// contains raw digest bytes, and not a human-readable hex-encoded string.
std::string ComputeSHA256(const std::string& data);
}  // namespace fcp

#endif  // FCP_BASE_DIGEST_H_
