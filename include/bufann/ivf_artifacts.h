#pragma once
#include <cstdint>
#include <iosfwd>
#include <string>

namespace diskann::ivf {
  struct ArtifactDigest {
    uint64_t bytes;
    uint64_t hash;
    bool     operator==(const ArtifactDigest&) const = default;
  };
  // Trusted immutable machine-local artifacts. FNV detects accidental changes,
  // not malicious replacement. No fsync/atomic multi-file publication
  // guarantee.
  void claim_artifact(const std::string& path);
#ifndef _WIN32
  // POSIX callers may preserve an existing creation-mode contract.
  void claim_artifact(const std::string& path, unsigned permissions);
#endif
  // Copy bytes only. Callers own opening/claiming, stream-state checks,
  // closing, and any separate integrity verification.
  void           copy_artifact_stream(std::istream& input, std::ostream& output);
  uint64_t       artifact_hash_file(const std::string& path);
  uint64_t       artifact_hash_prefix(const std::string& path, uint64_t bytes);
  ArtifactDigest artifact_digest(const std::string& path);
}  // namespace diskann::ivf
