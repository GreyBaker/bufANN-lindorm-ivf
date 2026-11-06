#include "bufann/ivf_patched_graph.h"
#include "ivf_faiss_internal.h"
#include <faiss/impl/AuxIndexStructures.h>
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <stdexcept>

namespace diskann::ivf {
  namespace {
    void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
    using Index = faiss_detail::Index;
    using Graph = faiss::HNSW;
    std::vector<uint32_t> links(const Graph& g, uint32_t id, int level) {
      size_t begin,end; g.neighbor_range(id,level,&begin,&end);
      std::vector<uint32_t> result;
      for (size_t i=begin;i<end && g.neighbors[i]>=0;++i) result.push_back(g.neighbors[i]);
      return result;
    }
    void replace(Graph& g, uint32_t id, int level, std::span<const uint32_t> ids) {
      size_t begin,end; g.neighbor_range(id,level,&begin,&end);
      check(ids.size()<=end-begin,"FAISS neighbor capacity exceeded");
      std::fill(g.neighbors.begin()+begin,g.neighbors.begin()+end,-1);
      for (auto target : ids) g.neighbors[begin++]=target;
    }
    std::vector<uint32_t> prune(faiss::DistanceComputer& dc, uint32_t origin,
                                 std::span<const uint32_t> ids, uint32_t limit) {
      std::priority_queue<Graph::NodeDistFarther> input;
      auto unique=std::vector<uint32_t>(ids.begin(),ids.end());
      std::sort(unique.begin(),unique.end()); unique.erase(std::unique(unique.begin(),unique.end()),unique.end());
      for (auto id : unique) if (id!=origin) input.emplace(dc.symmetric_dis(origin,id),id);
      std::vector<Graph::NodeDistFarther> chosen;
      Graph::shrink_neighbor_list(dc,input,chosen,limit);
      std::vector<uint32_t> result;
      for (auto hit : chosen) result.push_back(hit.id);
      return result;
    }
    // Local quiescent extension: use FAISS traversal/pruning on FAISS adjacency.
    // Stable IDs/levels survive; old neighbors seed each layer. All coordinates
    // are replaced before any reinsertion; never rebuild a graph in PCA space.
    void reinsert(Index& index, uint32_t id, const std::vector<std::vector<uint32_t>>& old,
                    const std::vector<std::array<uint32_t,2>>& anchors, uint32_t breadth) {
      auto& g=index.hnsw;
      auto dc=faiss_detail::distances(index);
      dc->set_query(faiss_detail::coordinates(index)+size_t(id)*index.d);
      for (int level=g.levels[id]-1;level>=0;--level) {
        auto seeds=old[level];
        seeds.erase(std::remove(seeds.begin(),seeds.end(),id),seeds.end());
        if (!seeds.empty()) {
          const auto nearest=*std::min_element(seeds.begin(),seeds.end(),[&](uint32_t a,uint32_t b) {
            const auto da=(*dc)(a),db=(*dc)(b);
            return da!=db ? da<db : a<b;
          });
          // Already nonempty: retain one seed without an allocation path.
          seeds[0]=nearest;
          seeds.erase(seeds.begin()+1,seeds.end());
        }
        if (seeds.empty()) {
          const auto candidate=anchors[level][0]==id ? anchors[level][1] : anchors[level][0];
          if (candidate!=UINT32_MAX) seeds.push_back(candidate);
        }
        if (seeds.empty()) { replace(g,id,level,{}); continue; }
        const auto found=faiss_detail::search_layer(index,*dc,std::min<uint64_t>(breadth,index.ntotal),
                          std::min<uint64_t>(breadth,index.ntotal),level,seeds,id);
        std::vector<uint32_t> candidates;
        for (auto hit : found) candidates.push_back(hit.id);
        auto chosen=prune(*dc,id,candidates,g.nb_neighbors(level));
        replace(g,id,level,chosen);
        for (auto other : chosen) {
          auto back=links(g,other,level);
          if (std::find(back.begin(),back.end(),id)!=back.end()) continue;
          back.push_back(id);
          if (back.size()>size_t(g.nb_neighbors(level))) back=prune(*dc,other,back,g.nb_neighbors(level));
          replace(g,other,level,back);
        }
      }
    }
    std::vector<std::array<uint32_t,2>> anchors(const Graph& g) {
      std::vector<std::array<uint32_t,2>> result(g.max_level+1,{UINT32_MAX,UINT32_MAX});
      for (uint32_t id=0;id<g.levels.size();++id) for (int l=0;l<g.levels[id];++l) {
        auto& pair=result[l];
        if (pair[0]==UINT32_MAX) pair[0]=id; else if (pair[1]==UINT32_MAX) pair[1]=id;
      }
      return result;
    }
  }
  struct PatchedCentroidGraph::Impl {
    std::unique_ptr<Index> graph;
    uint32_t projection_dim=0;
  };
  PatchedCentroidGraph::PatchedCentroidGraph(uint32_t dim,uint32_t count,PatchedGraphOptions o)
      : dim_(dim),count_(count),options_(o),impl_(std::make_unique<Impl>()) {}
  PatchedCentroidGraph::~PatchedCentroidGraph() = default;
  void PatchedCentroidGraph::configure_projection(uint32_t dim) {
    check(dim && dim<=dim_,"Invalid frozen projection dimension"); impl_->projection_dim=dim;
  }
  std::unique_ptr<PatchedCentroidGraph> PatchedCentroidGraph::build(uint32_t dim,
      std::span<const float> v,PatchedGraphOptions o) {
    check(dim && !v.empty() && v.size()%dim==0,"Invalid optimized FAISS graph shape");
    auto result=std::unique_ptr<PatchedCentroidGraph>(new PatchedCentroidGraph(dim,uint32_t(v.size()/dim),o));
    result->impl_->graph=faiss_detail::build(dim,v,o);
    result->measure(); result->stats_.unreachable_before=result->stats_.unreachable_after;
    return result;
  }
  std::unique_ptr<PatchedCentroidGraph> PatchedCentroidGraph::load(const std::string& path,
      uint32_t dim,std::span<const float> v,PatchedGraphOptions o,PatchedGraphDiagnostics s) {
    check(dim && !v.empty() && v.size()%dim==0,"Invalid optimized FAISS load shape");
    auto result=std::unique_ptr<PatchedCentroidGraph>(new PatchedCentroidGraph(dim,uint32_t(v.size()/dim),o));
    result->impl_->graph=faiss_detail::load(path,dim,v,o);
    result->measure();
    check(result->stats_.unreachable_after==s.unreachable_after && result->stats_.zero_indegree==s.zero_indegree,
          "FAISS graph connectivity diagnostics mismatch"); result->stats_=s;
    return result;
  }
  void PatchedCentroidGraph::measure() {
    const auto& g=impl_->graph->hnsw;
    std::vector<uint32_t> indegree(count_);
    for (uint32_t id=0;id<count_;++id) for (auto target : links(g,id,0)) ++indegree[target];
    stats_.zero_indegree=std::count(indegree.begin(),indegree.end(),0);
    std::vector<uint8_t> visited(count_);
    std::vector<uint32_t> queue{uint32_t(g.entry_point)}; visited[queue[0]]=1;
    for (size_t at=0;at<queue.size();++at) for (auto id : links(g,queue[at],0))
      if (!visited[id]) { visited[id]=1; queue.push_back(id); }
    stats_.unreachable_after=count_-queue.size();
  }
  std::vector<PatchedGraphHit> PatchedCentroidGraph::search(std::span<const float> query,
      uint32_t count,uint32_t breadth,std::span<const uint32_t> entries,std::span<const float> projected,
      uint32_t pdim,uint64_t* work,bool seeded) const {
    check(projected.empty() || pdim==impl_->projection_dim,"Projection kernel not configured");
    return faiss_detail::search(*impl_->graph,query,count,breadth,entries,projected,pdim,work,seeded);
  }
  void PatchedCentroidGraph::refresh(std::span<const float> v) {
    check(v.size()==uint64_t(count_)*dim_,"Invalid training coordinate refresh");
    for (float x : v) check(std::isfinite(x),"Nonfinite training centroid");
    auto& g=impl_->graph->hnsw;
    const auto entry_points=anchors(g);
    std::vector<std::vector<std::vector<uint32_t>>> old(count_);
    uint64_t bytes=count_*sizeof(old[0]);
    const auto allowance=uint64_t(count_)*(uint64_t(options_.degree)*32+512);
    for (uint32_t id=0;id<count_;++id) {
      old[id].resize(g.levels[id]); bytes+=g.levels[id]*sizeof(std::vector<uint32_t>);
      for (int l=0;l<g.levels[id];++l) { old[id][l]=links(g,id,l); bytes+=old[id][l].size()*4; }
      check(bytes<=allowance,"Training FAISS neighbor snapshot exceeds allowance");
    }
    std::copy(v.begin(),v.end(),faiss_detail::coordinates(*impl_->graph));
    for (uint32_t id=0;id<count_;++id) reinsert(*impl_->graph,id,old[id],entry_points,options_.breadth);
    measure();
  }
  void PatchedCentroidGraph::refine(uint32_t breadth) {
    check(breadth && breadth<INT32_MAX,"Invalid refinement breadth");
    auto& index=*impl_->graph; auto& g=index.hnsw;
    auto dc=faiss_detail::distances(index);
    std::vector<std::vector<uint32_t>> proposed(count_);
    uint64_t work=0;
    for (uint32_t id=0;id<count_;++id) {
      uint64_t calls=0;
      auto found=search({faiss_detail::coordinates(index)+size_t(id)*dim_,dim_},std::min(count_,breadth+1),breadth,{}, {},0,&calls);
      check(calls<=options_.work_limit-work,"Refinement work allowance exceeded"); work+=calls;
      std::vector<uint32_t> candidates;
      for (auto hit : found) if (hit.id!=id) candidates.push_back(hit.id);
      // Pruning distance work is explicitly bounded too (not global FAISS stats).
      const uint64_t bound=uint64_t(candidates.size())*(g.nb_neighbors(0)+1);
      check(bound<=options_.work_limit-work,"Refinement pruning work allowance exceeded"); work+=bound;
      proposed[id]=prune(*dc,id,candidates,g.nb_neighbors(0));
    }
    for (uint32_t id=0;id<count_;++id) replace(g,id,0,proposed[id]);
    stats_.refined_nodes=count_; measure(); stats_.unreachable_before=stats_.unreachable_after;
    std::vector<uint32_t> indegree(count_);
    for (const auto& list : proposed) for (auto id : list) ++indegree[id];
    const auto entry_points=anchors(g);
    for (uint32_t id=0;id<count_;++id) if (!indegree[id] && count_>1) {
      std::vector<std::vector<uint32_t>> old(g.levels[id]);
      for (int l=0;l<g.levels[id];++l) old[l]=links(g,id,l);
      reinsert(index,id,old,entry_points,options_.breadth); ++stats_.reinserted_nodes;
    }
    measure();
  }
  void PatchedCentroidGraph::save(const std::string& path) const { faiss_detail::save(*impl_->graph,path); }
}
