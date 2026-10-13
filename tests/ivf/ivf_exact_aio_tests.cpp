#include "bufann/exact_aio.h"

#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>

namespace {
  void require(bool ok, const char* message) {
    if (!ok)
      throw std::runtime_error(message);
  }
  struct FakeOps {
    std::array<iocb*, 256> pending{};
    size_t                 count = 0, submitted = 0, drained = 0;
    int                    limit = 256, collect_limit = 256;
    bool                   submit_interrupt = false, collect_interrupt = false;
    int                    submit_error = 0, collect_error = 0, abort_error = 0;
    long                   result = 4096, secondary = 0;
    bool                   foreign = false, duplicate = false, aborted = false;
    int                    submit(long n, iocb** blocks) {
      if (submit_interrupt) {
        submit_interrupt = false;
        return -EINTR;
      }
      if (submit_error)
        return submit_error;
      require(count == 0, "suffix submitted before accepted prefix drained");
      const int accepted = std::min(int(n), limit);
      for (int i = 0; i < accepted; ++i)
        pending[count++] = blocks[i];
      submitted += accepted;
      return accepted;
    }
    int collect(long, long maximum, io_event* events) {
      if (collect_interrupt) {
        collect_interrupt = false;
        return -EINTR;
      }
      if (collect_error)
        return collect_error;
      const int n = std::min({int(count), int(maximum), collect_limit});
      for (int i = 0; i < n; ++i) {
        // Deliberate reverse completion order; requests have different lengths.
        events[i] = {};
        events[i].obj = foreign ? nullptr : pending[count - 1];
        if (duplicate && i > 0)
          events[i].obj = events[0].obj;
        events[i].res =
            result == 4096 ? pending[count - 1]->u.c.nbytes : result;
        events[i].res2 = secondary;
        --count;
        ++drained;
      }
      return n;
    }
    int abort_pending() {
      aborted = true;
      if (!abort_error)
        count = 0;
      return abort_error;
    }
  };
  std::array<iocb, 3> blocks() {
    std::array<iocb, 3> result{};
    for (size_t i = 0; i < result.size(); ++i) {
      result[i].u.c.nbytes = 4096 * (i + 1);
      result[i].u.c.offset = 4096 * i;
    }
    return result;
  }
  std::string fails(FakeOps& ops) {
    auto requests = blocks();
    try {
      diskann::inplace::detail::exact_aio_batch(ops, requests);
    } catch (const std::runtime_error& error) {
      require(ops.count == 0, "exception escaped with pending kernel buffers");
      return error.what();
    }
    throw std::runtime_error("expected exact read failure");
  }
}  // namespace

int main() {
  for (int submit_limit : {1, 2, 3})
    for (int collect_limit : {1, 2, 3}) {
      FakeOps ops;
      ops.limit = submit_limit;
      ops.collect_limit = collect_limit;
      ops.submit_interrupt = ops.collect_interrupt = true;
      auto requests = blocks();
      diskann::inplace::detail::exact_aio_batch(ops, requests);
      require(ops.submitted == 3 && ops.drained == 3 && !ops.aborted,
              "partial/interrupted I/O duplicated or lost accepted work");
    }
  for (long result : {0L, 512L, -EIO * 1L}) {
    FakeOps ops;
    ops.result = result;
    ops.collect_limit = 1;
    const auto message = fails(ops);
    require(ops.drained == 3 && !ops.aborted,
            "completion failure was not fully drained");
    for (const char* key : {"operation=completion", "offset=", "requested=",
                            "actual=", "errno=", "res2="})
      require(message.find(key) != std::string::npos,
              "missing exact read diagnostic");
  }
  {
    FakeOps ops;
    ops.secondary = -EIO;
    fails(ops);
    require(ops.drained == 3, "secondary completion error bypassed draining");
  }
  for (bool duplicate : {false, true}) {
    FakeOps ops;
    ops.foreign = !duplicate;
    ops.duplicate = duplicate;
    require(fails(ops).find("completion_identity") != std::string::npos &&
                ops.aborted,
            "unknown/duplicate completion did not quiesce its context");
  }
  for (int error : {0, -EIO}) {
    FakeOps ops;
    ops.collect_error = error;
    ops.collect_limit = error == 0 ? 0 : 256;
    require(fails(ops).find("operation=collect") != std::string::npos &&
                ops.aborted,
            "collection failure did not synchronously abort pending I/O");
  }
  {
    FakeOps ops;
    ops.submit_error = -EAGAIN;
    fails(ops);
    require(!ops.aborted && !ops.submitted,
            "empty submission destroyed a healthy context");
  }
  // The only fail-stop fallback is inability to make in-flight buffers safe.
  const auto child = fork();
  require(child >= 0, "fork failed");
  if (child == 0) {
    std::set_terminate([] { std::_Exit(86); });
    FakeOps ops;
    ops.collect_error = -EIO;
    ops.abort_error = -EINVAL;
    fails(ops);
    std::_Exit(1);
  }
  int status = 0;
  require(waitpid(child, &status, 0) == child && WIFEXITED(status) &&
              WEXITSTATUS(status) == 86,
          "failed cancellation unwound unsafe buffers instead of failing-stop");
}
