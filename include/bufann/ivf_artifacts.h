#pragma once
#include <cstdint>
#include <string>

namespace diskann::ivf {
struct ArtifactDigest {
    uint64_t bytes;
    uint64_t hash;
    bool operator==(const ArtifactDigest&) const = default;
};
// Trusted immutable machine-local artifacts. FNV detects accidental changes,
// not malicious replacement. No fsync/atomic multi-file publication guarantee.
void claim_artifact(const std::string& path);
uint64_t artifact_hash_prefix(const std::string& path, uint64_t bytes);
ArtifactDigest artifact_digest(const std::string& path);
} // namespace diskann::ivf
