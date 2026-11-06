#include "ivf_faiss_internal.h"
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/utils/distances.h>
#include <algorithm>
#include <bit>
#include <climits>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace diskann::ivf::faiss_detail {
  namespace {
    struct VisitedScratch {
      std::unique_ptr<faiss::VisitedTable> table;
      bool leased = false;
    };
    thread_local VisitedScratch visited_scratch;
    void check(bool ok, const char* why) {
      if (!ok)
        throw std::runtime_error(why);
    }
    template<class T>
    void write(std::ostream& out, const T& value) {
      out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }
    template<class T>
    T read(std::istream& in) {
      T value{};
      in.read(reinterpret_cast<char*>(&value), sizeof(T));
      check(bool(in), "Truncated FAISS graph artifact");
      return value;
    }
    struct CountedDistance final : faiss::DistanceComputer {
      std::unique_ptr<faiss::DistanceComputer> base;
      uint64_t calls = 0;
      explicit CountedDistance(std::unique_ptr<faiss::DistanceComputer> p)
          : base(std::move(p)) {
      }
      void set_query(const float* q) override {
        base->set_query(q);
      }
      float operator()(faiss::idx_t id) override {
        ++calls;
        return (*base)(id);
      }
      float symmetric_dis(faiss::idx_t a, faiss::idx_t b) override {
        ++calls;
        return base->symmetric_dis(a, b);
      }
    };
    struct ProjectedDistance final : faiss::DistanceComputer {
      std::span<const float> values;
      const float* query = nullptr;
      uint32_t dim;
      ProjectedDistance(std::span<const float> v, uint32_t d)
          : values(v), dim(d) {
      }
      void set_query(const float* q) override {
        query = q;
      }
      float operator()(faiss::idx_t id) override {
        return faiss::fvec_L2sqr(query, values.data() + size_t(id) * dim, dim);
      }
      float symmetric_dis(faiss::idx_t a, faiss::idx_t b) override {
        return faiss::fvec_L2sqr(values.data() + size_t(a) * dim,
                                  values.data() + size_t(b) * dim, dim);
      }
    };
    using RH = faiss::HeapBlockResultHandler<faiss::HNSW::C>;
    std::vector<Hit> hits(std::span<const float> d, std::span<const faiss::idx_t> ids) {
      std::vector<Hit> result;
      for (size_t i = 0; i < ids.size(); ++i)
        if (ids[i] >= 0)
          result.push_back({uint32_t(ids[i]), d[i]});
      std::sort(result.begin(), result.end(), [](auto a, auto b) {
        return a.distance != b.distance ? a.distance < b.distance : a.id < b.id;
      });
      return result;
    }
  }
  VisitedLease::VisitedLease(size_t count) {
    check(count && count <= INT32_MAX, "Invalid FAISS visited-table size");
    if (visited_scratch.leased) {
      temporary_ = std::make_unique<faiss::VisitedTable>(int(count));
      table_ = temporary_.get();
    } else {
      auto& cached = visited_scratch.table;
      if (!cached || cached->visited.size() != count) {
        // Release before allocation: do not retain the largest graph forever,
        // nor overlap two graph-sized caches when switching graph sizes.
        cached.reset();
        cached = std::make_unique<faiss::VisitedTable>(int(count));
      }
      visited_scratch.leased = true;
      table_ = cached.get();
    }
    epoch_ = table_->visno;
  }
  VisitedLease::~VisitedLease() {
    // HNSW::search advances on normal return; search_level_0 and
    // search_from_candidates do not. Also invalidate partial exception paths.
    if (table_->visno == epoch_)
      table_->advance();
    if (!temporary_)
      visited_scratch.leased = false;
  }
  size_t retained_visited_bytes() {
    return visited_scratch.table ? visited_scratch.table->visited.capacity() : 0;
  }
  float* coordinates(Index& index) {
    return dynamic_cast<faiss::IndexFlat&>(*index.storage).get_xb();
  }
  const float* coordinates(const Index& index) {
    return dynamic_cast<const faiss::IndexFlat&>(*index.storage).get_xb();
  }
  std::unique_ptr<faiss::DistanceComputer> distances(const Index& index) {
    return std::unique_ptr<faiss::DistanceComputer>(index.storage->get_distance_computer());
  }
  std::unique_ptr<Index> build(uint32_t dim, std::span<const float> values, PatchedGraphOptions o) {
    check(dim && dim <= INT32_MAX && !values.empty() &&
              values.size() % dim == 0 && values.size() / dim <= INT32_MAX &&
              o.degree >= 2 && o.degree <= 64 && o.breadth >= o.degree &&
              o.breadth <= INT32_MAX,
          "Invalid FAISS HNSW shape/options");
    const uint64_t count = values.size() / dim;
    const uint64_t per_node = uint64_t(dim) * 8 + uint64_t(o.degree) * 16 + 256;
    constexpr uint64_t fixed = 4ull * 1024 * 1024;
    check(count <= (UINT64_MAX - fixed) / per_node &&
              fixed + count * per_node <= o.memory_bytes,
          "FAISS graph exceeds memory allowance");
    for (float value : values)
      check(std::isfinite(value), "Nonfinite FAISS centroid");
    auto index = std::make_unique<Index>(dim, o.degree, faiss::METRIC_L2);
    index->hnsw.efConstruction = int(o.breadth);
    index->hnsw.rng = faiss::RandomGenerator(o.seed);
    // Upstream batch construction; one calling-task OpenMP ICV, restored even
    // after failure. Avoid add(1) loops (quadratic lock-array allocation).
    struct Threads {
      int saved = omp_get_max_threads();
      Threads() {
        omp_set_num_threads(1);
      }
      ~Threads() {
        omp_set_num_threads(saved);
      }
    } threads;
    index->add(values.size() / dim, values.data());
    return index;
  }
  std::vector<Hit> search(const Index& index, std::span<const float> query,
      uint32_t count, uint32_t breadth, std::span<const uint32_t> entries,
      std::span<const float> projected, uint32_t pdim, uint64_t* computations, bool seeded,
      double* kernel_us) {
    check(count && count <= index.ntotal && breadth && breadth <= INT32_MAX &&
              (projected.empty() ? query.size() == size_t(index.d)
                                : pdim && query.size() == pdim &&
                                      projected.size() == size_t(index.ntotal) * pdim),
          "Invalid FAISS search shape");
    for (auto id : entries)
      check(id < uint64_t(index.ntotal), "Invalid FAISS entry");
    std::unique_ptr<faiss::DistanceComputer> dc = projected.empty() ? distances(index) :
      std::make_unique<ProjectedDistance>(projected, pdim);
    if (computations)
      dc = std::make_unique<CountedDistance>(std::move(dc));
    dc->set_query(query.data());
    std::vector<float> d(count);
    std::vector<faiss::idx_t> ids(count);
    RH block(1, d.data(), ids.data(), count);
    RH::SingleResultHandler result(block);
    result.begin(0);
    VisitedLease lease(index.ntotal);
    auto& visited = lease.table();
    faiss::SearchParametersHNSW params;
    params.efSearch = int(std::min<uint64_t>(index.ntotal, std::max(count, breadth)));
    const auto start = kernel_us ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
    if (seeded && !entries.empty()) {
      std::vector<int32_t> seeds(entries.begin(), entries.end());
      std::vector<float> seed_distances;
      for (auto id : entries)
        seed_distances.push_back((*dc)(id));
      faiss::HNSWStats stats;
      index.hnsw.search_level_0(*dc, result, seeds.size(), seeds.data(),
                                 seed_distances.data(), 2, stats, visited, &params);
    } else {
      (void) index.hnsw.search(*dc, result, visited, &params);
    }
    if (kernel_us)
      *kernel_us = std::chrono::duration<double, std::micro>(
          std::chrono::steady_clock::now() - start).count();
    result.end();
    if (computations)
      *computations = static_cast<CountedDistance&>(*dc).calls;
    return hits(d, ids);
  }
  std::vector<Hit> search_layer(const Index& index, faiss::DistanceComputer& dc,
      uint32_t count, uint32_t breadth, int level, std::span<const uint32_t> entries, uint32_t exclude) {
    std::vector<float> d(count);
    std::vector<faiss::idx_t> ids(count);
    RH block(1, d.data(), ids.data(), count);
    RH::SingleResultHandler result(block);
    result.begin(0);
    // The existing node may be a bridge between its old neighbors. Traverse
    // through it while excluding it from results; pre-marking it visited would
    // sever that bridge during sequential reinsertion.
    VisitedLease lease(index.ntotal);
    auto& visited = lease.table();
    faiss::SearchParametersHNSW params;
    params.efSearch = int(breadth);
    faiss::IDSelectorRange own(exclude, uint64_t(exclude) + 1);
    faiss::IDSelectorNot other(&own);
    params.sel = &other;
    faiss::HNSW::MinimaxHeap candidates(std::max<size_t>(breadth, entries.size()));
    for (auto id : entries)
      if (id != exclude)
        candidates.push(id, dc(id));
    faiss::HNSWStats stats;
    faiss::search_from_candidates(index.hnsw, dc, result, candidates, visited,
                                    stats, level, 0, &params);
    result.end();
    return hits(d, ids);
  }
  void save(const Index& index, const std::string& path) {
    static_assert(sizeof(size_t) == 8 && std::endian::native == std::endian::little);
    std::ofstream out(path, std::ios::binary);
    out.write("IVFFAIS1", 8);
    for (uint32_t x : {uint32_t(index.d), uint32_t(index.ntotal), uint32_t(index.hnsw.nb_neighbors(1)),
                       uint32_t(index.hnsw.efConstruction), uint32_t(index.hnsw.max_level), uint32_t(index.hnsw.entry_point)})
      write(out, x);
    write(out, uint64_t(index.hnsw.neighbors.size()));
    for (int x : index.hnsw.levels)
      write(out, int32_t(x));
    for (auto x : index.hnsw.offsets)
      write(out, uint64_t(x));
    for (auto x : index.hnsw.neighbors)
      write(out, x);
    out.write(reinterpret_cast<const char*>(coordinates(index)), size_t(index.ntotal)*index.d*sizeof(float));
    out.close();
    check(bool(out), "FAISS graph write failed");
  }
  std::unique_ptr<Index> load(const std::string& path, uint32_t dim, std::span<const float> values, PatchedGraphOptions o) {
    check(dim && dim <= INT32_MAX && !values.empty() && values.size() % dim == 0 &&
              o.degree >= 2 && o.degree <= 64,
          "Invalid FAISS load shape/options");
    const uint64_t count = values.size() / dim;
    const auto size = std::filesystem::file_size(path);
    check(size <= o.memory_bytes && size >= 48, "FAISS artifact exceeds allowance/framing");
    std::ifstream in(path, std::ios::binary);
    char magic[8]{};
    in.read(magic, 8);
    check(std::memcmp(magic,"IVFFAIS1",8)==0 && read<uint32_t>(in)==dim &&
          read<uint32_t>(in)==count && read<uint32_t>(in)==o.degree && read<uint32_t>(in)==o.breadth,
          "Incompatible FAISS graph header");
    const auto level=read<uint32_t>(in), entry=read<uint32_t>(in);
    const auto neighbors=read<uint64_t>(in);
    check(count && count<=INT32_MAX && level<32 && entry<count &&
          neighbors <= count*(uint64_t(o.degree)*(level+2)) &&
          neighbors <= (UINT64_MAX-48-count*12-values.size()*4)/4 &&
          size == 48+count*12+neighbors*4+values.size()*4, "Invalid FAISS graph lengths");
    auto index = std::make_unique<Index>(dim,o.degree,faiss::METRIC_L2);
    auto& g = index->hnsw;
    g.efConstruction = o.breadth;
    g.max_level = level;
    g.entry_point = entry;
    g.levels.resize(count);
    g.offsets.resize(count + 1);
    int maximum = 0;
    for (auto& l : g.levels) {
      l = read<int32_t>(in);
      check(l > 0 && l <= int(level + 1) && l < int(g.cum_nneighbor_per_level.size()),
            "Invalid FAISS node level");
      maximum = std::max(maximum, l - 1);
    }
    uint64_t offset = 0;
    for (uint32_t id = 0; id <= count; ++id) {
      g.offsets[id] = read<uint64_t>(in);
      check(g.offsets[id] == offset && offset <= neighbors, "Invalid FAISS offsets");
      if (id < count)
        offset += g.cum_nb_neighbors(g.levels[id]);
    }
    check(offset==neighbors && maximum==int(level) && g.levels[entry]==int(level+1),"Invalid FAISS level/entry layout");
    g.neighbors.resize(neighbors);
    for (auto& id : g.neighbors)
      id = read<int32_t>(in);
    for (uint32_t id = 0; id < count; ++id) {
      for (int l = 0; l < g.levels[id]; ++l) {
        size_t begin, end;
        g.neighbor_range(id, l, &begin, &end);
        bool tail = false;
        std::vector<int32_t> seen;
        for (size_t j = begin; j < end; ++j) {
          const auto target = g.neighbors[j];
          if (target == -1) {
            tail = true;
            continue;
          }
          check(!tail && target >= 0 && uint64_t(target) < count && uint32_t(target) != id &&
                    g.levels[target] > l && std::find(seen.begin(), seen.end(), target) == seen.end(),
                "Invalid FAISS edge ID/level");
          seen.push_back(target);
        }
      }
    }
    // No untrusted read_index(): all allocation sizes and links were checked.
    index->storage->add(count, values.data());
    index->ntotal = count;
    std::vector<float> row(dim);
    for (uint32_t id = 0; id < count; ++id) {
      in.read(reinterpret_cast<char*>(row.data()), dim * 4);
      check(bool(in) && std::memcmp(row.data(), values.data() + size_t(id) * dim, dim * 4) == 0,
            "FAISS centroid/ID mismatch");
    }
    check(in.peek() == EOF, "Trailing FAISS artifact data");
    return index;
  }
}
