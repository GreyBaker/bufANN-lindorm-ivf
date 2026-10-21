#include "bufann/ivf_artifacts.h"
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

namespace diskann::ivf {
  namespace {
    void append_hash(uint64_t& hash, const char* data, std::streamsize size) {
      for (std::streamsize i = 0; i < size; ++i) {
        hash ^= static_cast<unsigned char>(data[i]);
        hash *= 1099511628211ull;
      }
    }
  }  // namespace
  void claim_artifact(const std::string& path) {
    auto* file = std::fopen(path.c_str(), "wbx");
    if (!file)
      throw std::runtime_error("IVF artifact exists or cannot be created");
    if (std::fclose(file) != 0)
      throw std::runtime_error("IVF artifact close failed");
  }
#ifndef _WIN32
  void claim_artifact(const std::string& path, unsigned permissions) {
    const int fd =
        ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, permissions);
    if (fd < 0)
      throw std::runtime_error("IVF artifact exists or cannot be created");
    if (::close(fd) != 0)
      throw std::runtime_error("IVF artifact close failed");
  }
#endif
  void copy_artifact_stream(std::istream& input, std::ostream& output) {
    char buffer[8192];
    while (input.read(buffer, sizeof(buffer)) || input.gcount())
      output.write(buffer, input.gcount());
  }
  uint64_t artifact_hash_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
      throw std::runtime_error("Cannot read IVF artifact");
    uint64_t hash = 14695981039346656037ull;
    char     buffer[8192];
    while (input.read(buffer, sizeof(buffer)) || input.gcount())
      append_hash(hash, buffer, input.gcount());
    if (!input.eof() || input.bad())
      throw std::runtime_error("IVF artifact read failed");
    return hash;
  }
  uint64_t artifact_hash_prefix(const std::string& path, uint64_t bytes) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
      throw std::runtime_error("Cannot read IVF artifact");
    uint64_t hash = 14695981039346656037ull;
    char     buffer[8192];
    while (bytes) {
      const auto count = static_cast<std::streamsize>(
          std::min<uint64_t>(bytes, sizeof(buffer)));
      input.read(buffer, count);
      if (!input)
        throw std::runtime_error("Truncated IVF artifact hash input");
      append_hash(hash, buffer, count);
      bytes -= count;
    }
    return hash;
  }
  ArtifactDigest artifact_digest(const std::string& path) {
    const auto bytes = std::filesystem::file_size(path);
    return {bytes, artifact_hash_prefix(path, bytes)};
  }
}  // namespace diskann::ivf
