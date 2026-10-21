#include "bufann/ivf_artifacts.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace diskann::ivf;
namespace {
  void require(bool condition, const char* message) {
    if (!condition)
      throw std::runtime_error(message);
  }
  template<class Function>
  void rejects(Function function, const char* expected = nullptr) {
    bool rejected = false;
    try {
      function();
    } catch (const std::runtime_error& error) {
      require(!expected || std::string(error.what()) == expected,
              "Unexpected artifact refusal message");
      rejected = true;
    }
    require(rejected, "Expected artifact refusal");
  }
  uint64_t reference_hash(const std::string& bytes) {
    uint64_t hash = 14695981039346656037ull;
    for (unsigned char byte : bytes) {
      hash ^= byte;
      hash *= 1099511628211ull;
    }
    return hash;
  }
  class FailingReadBuffer : public std::streambuf {
    std::streamsize xsgetn(char*, std::streamsize) override {
      throw std::runtime_error("Injected artifact read failure");
    }
  };
  class FailingWriteBuffer : public std::streambuf {
    std::streamsize xsputn(const char*, std::streamsize) override {
      return 0;
    }
  };
  void copy_fixtures(const std::filesystem::path& root,
                     const std::string& bytes) {
    for (size_t size : {0u, 1u, 8192u, 8193u}) {
      const auto source = (root / ("source_" + std::to_string(size))).string();
      const auto target = (root / ("copy_" + std::to_string(size))).string();
      claim_artifact(source);
      {
        std::ofstream output(source, std::ios::binary);
        output.write(bytes.data(), size);
        output.close();
        require(bool(output), "Copy source fixture write failed");
      }
      claim_artifact(target);
      {
        std::ifstream input(source, std::ios::binary);
        std::ofstream output(target, std::ios::binary);
        require(bool(input) && bool(output), "Copy fixture open failed");
        copy_artifact_stream(input, output);
        require(input.eof() && !input.bad(), "Copy did not retain clean EOF");
        output.close();
        require(bool(output), "Copy fixture write failed");
      }
      std::ifstream copied(target, std::ios::binary);
      require(std::string(std::istreambuf_iterator<char>(copied), {}) ==
                  bytes.substr(0, size),
              "Streaming copy bytes mismatch");
      const auto digest = artifact_digest(target);
      require(digest == artifact_digest(source), "Copy digest mismatch");
      rejects([&] { claim_artifact(target); },
              "IVF artifact exists or cannot be created");
      require(artifact_digest(target) == digest,
              "Copy target collision truncated bytes");
    }
    FailingReadBuffer   read_buffer;
    std::istream        unreadable(&read_buffer);
    std::ostringstream output;
    copy_artifact_stream(unreadable, output);
    require(unreadable.bad() && output.str().empty(),
            "Copy masked a read failure");
    FailingWriteBuffer write_buffer;
    std::istringstream input(bytes);
    std::ostream       unwritable(&write_buffer);
    copy_artifact_stream(input, unwritable);
    require(input.eof() && !input.bad() && unwritable.fail(),
            "Copy changed read-to-EOF/write-failure state contract");
  }
}  // namespace
int main(int argc, char** argv) {
  try {
    require(argc == 2, "Provide owned scratch directory");
    const auto root =
        std::filesystem::path(argv[1]) /
        std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
    std::filesystem::create_directories(root);
    const auto path = (root / "artifact.bin").string();
    claim_artifact(path);
    require(artifact_hash_file(path) == reference_hash(""),
            "Empty hash mismatch");
    std::string bytes(17001, '\0');
    for (size_t i = 0; i < bytes.size(); ++i)
      bytes[i] = static_cast<char>(i % 251);
    copy_fixtures(root, bytes);
    {
      std::ofstream output(path, std::ios::binary);
      output.write(bytes.data(), bytes.size());
    }
    require(artifact_hash_file(path) == reference_hash(bytes),
            "Full streaming hash mismatch");
    require(artifact_digest(path) ==
                ArtifactDigest{bytes.size(), reference_hash(bytes)},
            "Digest framing mismatch");
    for (uint64_t size : {0ull, 1ull, 8192ull, 17001ull})
      require(artifact_hash_prefix(path, size) ==
                  reference_hash(bytes.substr(0, size)),
              "Prefix hash mismatch");
    rejects([&] { artifact_hash_prefix(path, bytes.size() + 1); });
    rejects([&] { claim_artifact(path); },
            "IVF artifact exists or cannot be created");
    require(artifact_hash_file(path) == reference_hash(bytes),
            "Claim truncated an existing file");
    rejects([&] { artifact_hash_file((root / "missing").string()); });
    rejects([&] { artifact_hash_file(root.string()); });
#ifndef _WIN32
    const auto link = root / "dangling";
    std::filesystem::create_symlink(root / "absent", link);
    rejects([&] { claim_artifact(link.string()); });
    rejects([&] { claim_artifact(link.string(), 0644); });
    require(std::filesystem::is_symlink(link) &&
                !std::filesystem::exists(root / "absent"),
            "Claim followed dangling symlink");
    const auto mode_path = (root / "mode.bin").string();
    claim_artifact(mode_path, 0644);
    // Match the old syscall on this filesystem: DrvFS may ignore POSIX modes.
    const auto reference = (root / "reference_mode.bin").string();
    const int fd = ::open(reference.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    require(fd >= 0 && ::close(fd) == 0, "Reference claim failed");
    struct stat status {
    }, expected{};
    require(::stat(mode_path.c_str(), &status) == 0 &&
                ::stat(reference.c_str(), &expected) == 0 &&
                (status.st_mode & 0777) == (expected.st_mode & 0777),
            "Heap artifact creation mode changed");
    rejects([&] { claim_artifact(mode_path, 0644); },
            "IVF artifact exists or cannot be created");
#endif
    std::cout << "Artifact claim/copy/full/prefix hash fixtures passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
