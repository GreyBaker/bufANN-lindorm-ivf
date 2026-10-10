#include "bufann/ivf_router.h"
#include "bufann/ivf_patched_graph.h"
#include "bufann/ivf_pca.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace {
  using namespace diskann::ivf;
  void require(bool ok, const char* why) {
    if (!ok)
      throw std::runtime_error(why);
  }
  template<class F>
  void rejects(F f, const char* expected = nullptr) {
    try {
      f();
    } catch (const std::exception& error) {
      require(!expected ||
                  std::string(error.what()).find(expected) != std::string::npos,
              "Unexpected refusal reason");
      return;
    }
    throw std::runtime_error("Expected refusal");
  }
  std::string bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), {}};
  }
  void pca_tests() {
    std::vector<float> rank_one;
    for (int r = -20; r <= 20; ++r)
      for (float axis : {1.f, 2.f, -3.f, 4.f})
        rank_one.push_back(r * axis + 9);
    auto p = CentroidPCA::fit(4, rank_one);
    require(p.retained_dim() == 1 && p.retained_variance() >= .95 &&
                !p.degenerate(),
            "Rank-one PCA failed");
    std::ostringstream out(std::ios::binary);
    p.write(out);
    const auto         saved = out.str();
    std::istringstream in(saved, std::ios::binary);
    auto               loaded =
        CentroidPCA::read(in, 4, rank_one, .95, 1ull << 30, 1000000000);
    require(
        loaded.project({rank_one.data(), 4}) == p.project({rank_one.data(), 4}),
        "PCA round trip changed projection");
    rejects([&] {
      std::istringstream truncated(saved.substr(0, saved.size() - 1),
                                   std::ios::binary);
      (void) CentroidPCA::read(truncated, 4, rank_one, .95, 1ull << 30,
                               1000000000);
    });
    rejects([&] { (void) CentroidPCA::fit(4, rank_one, .95, 1); });
    rejects([&] { (void) CentroidPCA::fit(4, rank_one, .95, 1ull << 30, 1); });
    std::vector<float> zero(12, 7);
    auto               z = CentroidPCA::fit(4, zero);
    require(z.degenerate() && z.retained_dim() == 4 &&
                z.project({zero.data(), 4}) == std::vector<float>(4, 0),
            "Zero-variance identity PCA failed");
    std::vector<float> repeated;
    for (uint32_t axis = 0; axis < 4; ++axis)
      for (float sign : {-1.f, 1.f})
        for (uint32_t d = 0; d < 4; ++d)
          repeated.push_back(d == axis ? sign : 0);
    auto a = CentroidPCA::fit(4, repeated, 1),
         b = CentroidPCA::fit(4, repeated, 1);
    std::ostringstream sa, sb;
    a.write(sa);
    b.write(sb);
    require(sa.str() == sb.str(), "Repeated eigenvalue PCA nondeterministic");
  }
  void router_tests(const std::filesystem::path& root) {
    constexpr uint32_t              dim = 13, n = 768, queries = 40;
    std::mt19937                    rng(291);
    std::normal_distribution<float> random(0, 1);
    std::vector<float>              v(n * dim), q(queries * dim);
    for (auto& x : v)
      x = random(rng);
    for (auto& x : q)
      x = random(rng);
    {
      std::ofstream  queries_file(root / "queries.bin", std::ios::binary);
      const uint32_t header[]{queries, dim};
      queries_file.write(reinterpret_cast<const char*>(header), sizeof(header));
      queries_file.write(reinterpret_cast<const char*>(q.data()),
                         q.size() * sizeof(float));
    }
    const std::array<RouterMode, 6> modes{
        RouterMode::Exact,   RouterMode::Hnswlib, RouterMode::PatchedFull,
        RouterMode::Refined, RouterMode::PCA,     RouterMode::PCARefined};
    RouterOptions options;
    options.native_modes = true;
    options.seed = 0;
    options.query_breadth = 128;
    options.insert_breadth = 128;
    auto       router = CentroidRouter::build(dim, v, options);
    const auto prefix = (root / "router_0").string();
    router->save(prefix);
    // Patch-off graph construction is byte-identical, not merely similar
    // recall.
    require(bytes(prefix + ".hnsw") == bytes(prefix + ".patched.hnsw"),
            "Patch-off construction differs from vanilla");
    auto loaded = CentroidRouter::load(prefix);
    for (uint32_t probe : {1u, 4u, 10u, 32u}) {
      for (auto mode : modes) {
        double   recall = 0, elapsed = 0;
        uint64_t distances = 0;
        for (uint32_t id = 0; id < queries; ++id) {
          auto         query = std::span<const float>(q).subspan(id * dim, dim);
          auto         truth = router->search(query, probe, true);
          RoutingStats stats;
          const auto   start = std::chrono::steady_clock::now();
          auto hits = router->search(query, probe, false, &stats, {mode, 3});
          elapsed += std::chrono::duration<double, std::micro>(
                         std::chrono::steady_clock::now() - start)
                         .count();
          require(
              hits == loaded->search(query, probe, false, nullptr, {mode, 3}),
              "Reload changed hits");
          if (mode == RouterMode::PatchedFull)
            require(hits == router->search(query, probe, false, nullptr,
                                           {RouterMode::Hnswlib}),
                    "Patch-off search differs from vanilla");
          for (auto hit : hits)
            for (auto gt : truth)
              if (hit.list_id == gt.list_id)
                recall += 1.0 / (queries * probe);
          require(
              stats.distance_count_available == (mode != RouterMode::Hnswlib),
              "Distance counter mislabeled");
          distances += stats.distance_evaluations;
        }
        std::cout << "{\"scope\":\"synthetic_s08_not_production\",\"seed\":0,"
                     "\"nprobe\":"
                  << probe << ",\"mode\":\"" << router_mode_name(mode)
                  << "\",\"recall\":" << recall
                  << ",\"mean_us\":" << elapsed / queries
                  << ",\"mean_distances\":" << double(distances) / queries
                  << "}\n";
      }
    }
    std::array<std::exception_ptr, 8> errors{};
    std::vector<std::jthread>         threads;
    for (size_t t = 0; t < errors.size(); ++t)
      threads.emplace_back([&, t] {
        try {
          for (uint32_t i = 0; i < queries; ++i)
            for (auto mode : modes) {
              auto query = std::span<const float>(q).subspan(i * dim, dim);
              require(router->search(query, 10, false, nullptr, {mode, 3}) ==
                          loaded->search(query, 10, false, nullptr, {mode, 3}),
                      "Concurrent query changed results");
            }
        } catch (...) {
          errors[t] = std::current_exception();
        }
      });
    threads.clear();
    for (auto e : errors)
      if (e)
        std::rethrow_exception(e);
    // Caller entries and alternate vector-space patches, including self
    // queries.
    auto graph = PatchedCentroidGraph::build(dim, v, {16, 200, 0});
    for (uint32_t id = 0; id < n; ++id) {
      const std::array<uint32_t, 1> entry{id};
      require(
          graph->search({v.data() + id * dim, dim}, 1, 128, entry)[0].id == id,
          "Caller entry point ignored");
    }
    auto pca = CentroidPCA::fit(dim, v);
    graph->configure_projection(pca.retained_dim());
    auto     projected = pca.project({q.data(), dim});
    uint64_t computations = 0;
    require(!graph->search(projected, 10, 128, {}, pca.projected(),
                           pca.retained_dim(), &computations)
                    .empty() &&
                computations,
            "Alternate traversal array unused");
    const std::array<uint32_t, 1> invalid{n};
    rejects([&] { graph->search({q.data(), dim}, 1, 128, invalid); });
    auto second = PatchedCentroidGraph::build(dim, v, {16, 200, 0});
    graph->refine(128);
    second->refine(128);
    graph->save((root / "refine_a").string());
    second->save((root / "refine_b").string());
    require(bytes((root / "refine_a").string()) ==
                bytes((root / "refine_b").string()),
            "Frozen refinement nondeterministic");
    if (!graph->diagnostics().reinserted_nodes) {
      const size_t upper_offset =
          96 + size_t(n) * (4 + 8 * options.degree + dim * 4 + 8);
      require(bytes(prefix + ".hnsw").substr(upper_offset) ==
                  bytes((root / "refine_a").string()).substr(upper_offset),
              "Refinement changed upper layers without isolated reinsertion");
    }
    std::vector<float> duplicates(600, 1);
    auto duplicate = PatchedCentroidGraph::build(3, duplicates, {2, 8, 0});
    duplicate->refine(16);
    require(duplicate->diagnostics().reinserted_nodes > 0 &&
                duplicate->diagnostics().reinserted_nodes <= 200,
            "Single isolated-node reinsertion pass not exercised");
    const std::array<float, 1> single{1};
    auto                       one = PatchedCentroidGraph::build(1, single);
    one->refine(1);
    require(one->search(single, 1, 1)[0].id == 0, "One-node refinement failed");
    RouterOptions duplicate_options;
    duplicate_options.native_modes = true;
    duplicate_options.degree = 2;
    duplicate_options.construction_breadth = 8;
    auto duplicate_router =
        CentroidRouter::build(3, duplicates, duplicate_options);
    for (uint32_t breadth : {8u, 32u, 200u})
      require(
          duplicate_router->search({duplicates.data(), 3}, 1, false, nullptr,
                                   {RouterMode::PatchedFull, 3, breadth}) ==
              duplicate_router->search({duplicates.data(), 3}, 1, false,
                                       nullptr,
                                       {RouterMode::Hnswlib, 3, breadth}),
          "Duplicate patch-off parity failed");
    auto plain = CentroidRouter::build(dim, v);
    rejects([&] {
      plain->search({q.data(), dim}, 1, false, nullptr,
                    {RouterMode::PatchedFull});
    });
    const auto plain_prefix = (root / "plain").string();
    plain->save(plain_prefix);
    require(!CentroidRouter::load(plain_prefix)->has_native_modes(),
            "Plain routing changed");
    // Both descriptor framings require stock1. Alter only test-owned metadata;
    // failed loads must neither rewrite it nor produce a relabeled copy.
    for (const auto& source : {plain_prefix, prefix}) {
      const auto descriptor = bytes(source + ".router");
      const auto backend_at = descriptor.find(router_backend_id);
      require(backend_at != std::string::npos,
              "Missing stock producer identity");
      for (const auto* backend :
           {"hnswlib-0.8.0-prefetch1", "hnswlib-0.8.0-nomv", "unknown"}) {
        auto rejected = descriptor;
        rejected.replace(backend_at, std::string(router_backend_id).size(),
                         backend);
        {
          std::ofstream file(source + ".router");
          file << rejected;
        }
        const auto copy = source + ".rejected_copy";
        rejects([&] { CentroidRouter::load(source)->save(copy); },
                "Unsupported IVF router producer");
        require(bytes(source + ".router") == rejected &&
                    !std::filesystem::exists(copy + ".router"),
                "Rejected producer was rewritten or copied");
      }
      {
        std::ofstream file(source + ".router");
        file << descriptor;
      }
      const auto copy = source + ".stock_copy";
      CentroidRouter::load(source)->save(copy);
      require(bytes(copy + ".router") == descriptor,
              "Stock producer roundtrip changed");
    }
    const auto corrupt = prefix + ".refined.hnsw";
    {
      std::fstream file(corrupt,
                        std::ios::binary | std::ios::in | std::ios::out);
      char         byte = 1;
      file.write(&byte, 1);
    }
    rejects([&] { CentroidRouter::load(prefix); });
    // Restore only the test-owned corruption so diagnostics can reuse the
    // family.
    graph->save(corrupt);
    require(bytes(corrupt) == bytes((root / "refine_a").string()),
            "Refinement save changed graph");
  }
}  // namespace
int main(int argc, char** argv) {
  try {
    require(argc == 2, "Provide owned scratch directory");
    std::filesystem::path root(argv[1]);
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    pca_tests();
    router_tests(root);
    std::cout << "Patched hnswlib/PCA fixtures passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
