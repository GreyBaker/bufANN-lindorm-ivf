#include "bufann/ivf_artifacts.h"
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace diskann::ivf {
void claim_artifact(const std::string& path) {
    auto* file = std::fopen(path.c_str(), "wbx");
    if (!file) throw std::runtime_error("IVF artifact exists or cannot be created");
    if (std::fclose(file) != 0) throw std::runtime_error("IVF artifact close failed");
}
uint64_t artifact_hash_prefix(const std::string& path, uint64_t bytes) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read IVF artifact");
    uint64_t hash = 14695981039346656037ull;
    char buffer[8192];
    while (bytes) {
        const auto count = static_cast<std::streamsize>(std::min<uint64_t>(bytes, sizeof(buffer)));
        input.read(buffer, count);
        if (!input) throw std::runtime_error("Truncated IVF artifact hash input");
        for (std::streamsize i = 0; i < count; ++i) {
            hash ^= static_cast<unsigned char>(buffer[i]); hash *= 1099511628211ull;
        }
        bytes -= count;
    }
    return hash;
}
ArtifactDigest artifact_digest(const std::string& path) {
    const auto bytes = std::filesystem::file_size(path);
    return {bytes, artifact_hash_prefix(path, bytes)};
}
} // namespace diskann::ivf
