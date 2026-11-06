#include "../../src/bufann/ivf_faiss_internal.h"
#include <faiss/utils/distances.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
#ifdef __linux__
#include <link.h>
#include <string_view>
#endif

namespace {
  using namespace diskann::ivf;
  void require(bool ok,const char* why) { if (!ok) throw std::runtime_error(why); }
  void visited_leases() {
    uint8_t* allocation = nullptr;
    for (size_t repeat = 0; repeat < 800; ++repeat) {
      faiss_detail::VisitedLease lease(193);
      auto& table = lease.table();
      if (!allocation) allocation = table.visited.data();
      require(table.visited.data() == allocation, "Visited scratch allocated per search");
      for (size_t id = 0; id < table.visited.size(); ++id)
        require(!table.get(id), "Visited epoch/rollover leaked an earlier search");
      table.set(repeat % 193);
      if (repeat % 2) table.advance();  // upstream normal-return convention
      {
        faiss_detail::VisitedLease nested(17);
        require(nested.table().visited.data() != allocation,
                "Reentrant search reused an active visited table");
        nested.table().set(3);
      }
      require(faiss_detail::retained_visited_bytes() == 193,
              "Nested search changed the calling thread's cached size");
    }
    try {
      faiss_detail::VisitedLease lease(193);
      lease.table().set(7);
      throw std::runtime_error("simulated distance-computer failure");
    } catch (const std::runtime_error&) {}
    {
      faiss_detail::VisitedLease lease(193);
      require(!lease.table().get(7), "Exception path leaked visited state");
    }
    for (size_t count : {17u, 1024u, 193u, 1u}) {
      faiss_detail::VisitedLease lease(count);
      require(lease.table().visited.size() == count &&
                  faiss_detail::retained_visited_bytes() == count,
              "Visited cache retained a different graph's allocation");
      for (size_t id = 0; id < count; ++id)
        require(!lease.table().get(id), "Graph-size switch reused stale epochs");
    }
  }
  void blas_abi() {
    constexpr size_t dim=13,nq=25,ny=35;
    std::vector<float> x(nq*dim), y(ny*dim), d(nq);
    for (size_t i=0;i<x.size();++i) x[i]=float(int(i*7%101)-50)*.03f;
    for (size_t i=0;i<y.size();++i) y[i]=float(int(i*13%107)-50)*.02f;
    std::vector<int64_t> ids(nq);
    require(faiss::distance_compute_blas_threshold<=int(nq),"Fixture did not exercise FAISS BLAS path");
    faiss::knn_L2sqr(x.data(),y.data(),dim,nq,ny,1,d.data(),ids.data());
#ifdef __linux__
    // ldd alone misses dispatchers that dlopen a second OpenMP runtime on the
    // first BLAS call. Check after exercising BLAS, including isolated builds.
    const auto conflicting_runtime = dl_iterate_phdr([](dl_phdr_info* info, size_t, void*) {
      const std::string_view name(info->dlpi_name);
      return int(name.find("libiomp5") != name.npos ||
                 name.find("libmkl_intel_thread") != name.npos ||
                 name.find("libmkl_rt") != name.npos);
    }, nullptr);
    require(!conflicting_runtime, "FAISS BLAS dynamically loaded MKL dispatcher/Intel OpenMP beside GNU OpenMP");
#endif
    for (size_t q=0;q<nq;++q) {
      double best=INFINITY; size_t winner=0;
      for (size_t id=0;id<ny;++id) {
        double sum=0;
        for (size_t j=0;j<dim;++j) { double delta=double(x[q*dim+j])-y[id*dim+j]; sum+=delta*delta; }
        if (sum<best) { best=sum; winner=id; }
      }
      require(ids[q]==int64_t(winner) && std::abs(d[q]-best)<1e-4,"FAISS BLAS integer ABI/distance mismatch");
    }
  }
  void vanilla_and_threads() {
    constexpr uint32_t dim=13,n=192;
    std::mt19937 rng(417); std::normal_distribution<float> normal;
    std::vector<float> values(n*dim), query(dim);
    for (auto& x:values) x=normal(rng);
    for (auto& x:query) x=normal(rng);
    auto graph=faiss_detail::build(dim,values,{8,64,17});
    std::array<std::vector<PatchedGraphHit>,4> expected;
    const std::array<uint32_t,4> counts{1,3,10,32}, breadths{8,16,48,96};
    for (size_t j=0;j<counts.size();++j) {
      expected[j]=faiss_detail::search(*graph,query,counts[j],breadths[j]);
      std::vector<float> distances(counts[j]); std::vector<faiss::idx_t> ids(counts[j]);
      faiss::SearchParametersHNSW params; params.efSearch=breadths[j];
      graph->search(1,query.data(),counts[j],distances.data(),ids.data(),&params);
      require(expected[j].size()==counts[j],"Vanilla candidate count differs from requested k");
      for (size_t k=0;k<counts[j];++k)
        require(expected[j][k].id==ids[k] && expected[j][k].distance==distances[k],
                "Per-call vanilla kernel differs from upstream IndexHNSW search");
    }
    const auto global=faiss::hnsw_stats;
    // Alternate normal and seeded upstream entry points across multiple epoch
    // rollovers, then repeat concurrently with independent breadths.
    const std::array<uint32_t, 2> seeds{0, 17};
    const auto seeded = faiss_detail::search(*graph, query, 10, 48, seeds, {}, 0, nullptr, true);
    for (size_t repeat = 0; repeat < 600; ++repeat) {
      const bool use_seed = repeat % 2;
      const auto actual = faiss_detail::search(*graph, query, 10, 48, seeds, {}, 0, nullptr, use_seed);
      const auto& wanted = use_seed ? seeded : expected[2];
      require(actual.size() == wanted.size(), "Mixed seeded search lost candidates");
      for (size_t k = 0; k < actual.size(); ++k)
        require(actual[k].id == wanted[k].id && actual[k].distance == wanted[k].distance,
                "Mixed seeded/unseeded epochs changed search results");
    }
    std::array<std::exception_ptr,8> errors{};
    std::vector<std::jthread> workers;
    for (size_t t=0;t<errors.size();++t) workers.emplace_back([&,t] {
      try {
        for (size_t repeat=0;repeat<600;++repeat) {
          const auto j=(t+repeat)%counts.size();
          uint64_t work=0;
          const auto hits=faiss_detail::search(*graph,query,counts[j],breadths[j],{}, {},0,t%2?&work:nullptr);
          require(hits.size()==expected[j].size(),"Concurrent FAISS candidate count changed");
          for (size_t k=0;k<hits.size();++k)
            require(hits[k].id==expected[j][k].id && hits[k].distance==expected[j][k].distance,
                    "Per-call FAISS breadth raced with another reader");
           if (t%2) require(work>0,"Local FAISS distance counter missing");
           const auto seeded_hits = faiss_detail::search(*graph, query, 10, 48, seeds, {}, 0, nullptr, true);
           require(seeded_hits.size() == seeded.size(), "Concurrent seeded candidate count changed");
           for (size_t k = 0; k < seeded.size(); ++k)
             require(seeded_hits[k].id == seeded[k].id && seeded_hits[k].distance == seeded[k].distance,
                     "Concurrent seeded search reused another reader's visited state");
        }
      } catch (...) { errors[t]=std::current_exception(); }
    });
    workers.clear();
    for (auto error:errors) if (error) std::rethrow_exception(error);
    require(global.n1==faiss::hnsw_stats.n1 && global.n2==faiss::hnsw_stats.n2 &&
            global.ndis==faiss::hnsw_stats.ndis && global.nhops==faiss::hnsw_stats.nhops,
            "Frozen adapter touched FAISS global statistics");
  }
}
int main() {
  try { visited_leases(); blas_abi(); vanilla_and_threads(); std::cout<<"FAISS visited scratch/rollover, BLAS ABI, upstream vanilla parity and per-call concurrent search passed\n"; }
  catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
