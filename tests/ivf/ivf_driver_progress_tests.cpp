#include "../ivf_driver_progress.h"
#include <stdexcept>

namespace {
  struct FakeClock {
    using time_point = std::chrono::steady_clock::time_point;
    static inline time_point current{};
    static time_point        now() noexcept {
      return current;
    }
  };
  using Progress = ivf_cli::DriverProgress<FakeClock>;
  size_t lines(const std::string& text) {
    size_t count = 0;
    for (auto c : text)
      count += c == '\n';
    return count;
  }
  void require(bool ok) {
    if (!ok)
      throw std::runtime_error("driver progress contract failed");
  }
}  // namespace

int main() {
  std::ostringstream output;
  {
    Progress progress("insert", 10000, "published_inserts", {}, output);
    require(lines(output.str()) == 1);
    FakeClock::current += std::chrono::seconds(1);
    progress.observe(1023);
    require(lines(output.str()) == 1);
    progress.observe(1024);
    require(lines(output.str()) == 2);
    progress.observe(2048);
    require(lines(output.str()) == 2);  // Same time, no additional output.
    FakeClock::current += std::chrono::seconds(1);
    progress.observe(3072);
    require(lines(output.str()) == 3);
    progress.observe(10000);
    require(progress.finish() == 2);
    FakeClock::current += std::chrono::seconds(5);
    require(progress.finish() == 2);
    require(lines(output.str()) == 4);
  }
  require(output.str().find("event=running elapsed_s=1.000 completed=1024") !=
          std::string::npos);
  require(output.str().find("event=complete elapsed_s=2.000 completed=10000") !=
          std::string::npos);
  require(output.str().find("rate_per_s=5000.000") != std::string::npos);
  output.str("");
  try {
    Progress progress("insert", 10, "published_inserts", {}, output);
    progress.observe(3);
    throw std::runtime_error("original insertion failure");
  } catch (const std::runtime_error& error) {
    require(std::string(error.what()) == "original insertion failure");
  }
  require(
      output.str().find("event=failed elapsed_s=0.000 completed=3 total=10") !=
      std::string::npos);
  require(output.str().find("event=complete") == std::string::npos);
  output.str("");
  {
    Progress progress(
        "checkpoint", std::nullopt, "operations",
        [](std::ostream& out) { out << " heap_page_writes=5"; }, output);
    progress.finish();
  }
  require(output.str().find("heap_page_writes=5") != std::string::npos);
  require(output.str().find("completed=") == std::string::npos);
  output.str("");
  {
    Progress progress("validation", 5, "validated_rows", {}, output);
    progress.observe(2);
    progress.finish();
  }
  require(output.str().find("event=incomplete") != std::string::npos);
  output.str("");
  { Progress progress("abandoned", std::nullopt, "operations", {}, output); }
  require(output.str().find("event=abandoned") != std::string::npos);
  // A failed output sink must not throw or replace the workload exception.
  std::ostringstream broken;
  broken.exceptions(std::ios::badbit);
  try {
    broken.setstate(std::ios::badbit);
  } catch (...) {
  }
  {
    Progress progress("insert", 1, "published_inserts", {}, broken);
    progress.observe(1);
    progress.finish();
  }
}
