# TODO: this whole process is somewhat dubious and likely ought be
# replaced with a more intentional and clean method

"""Apply the small v0.8.0 IVF patch series to build-tree header copies only.

Exact anchors deliberately fail on a different upstream version. Original
headers remain untouched. The generated vanilla copy has only patch 01.
"""
import pathlib
import shutil
import sys
import difflib


def once(text, old, new):
    assert text.count(old) == 1, ("upstream patch anchor", old[:100])
    return text.replace(old, new)


source, destination = map(pathlib.Path, sys.argv[1:])
(destination / "patches").mkdir(parents=True, exist_ok=True)
def record(name, before, after):
    (destination / "patches" / name).write_text(''.join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile="a/hnswlib/hnswalg.h", tofile="b/hnswlib/hnswalg.h")))

for family in ("ivf_vanilla", "ivf_patched"):
    directory = destination / family
    directory.mkdir(parents=True, exist_ok=True)
    for header in (source / "hnswlib").glob("*.h"):
        shutil.copyfile(header, directory / header.name)
    path = directory / "hnswalg.h"
    text = path.read_text()
    original = text
    # 01: bound prefetches, not SIMD distances. No speculative neighbor read
    # when the adjacency list is empty/full; no out-of-array +64 pointer.
    begin = text.index("#ifdef USE_SSE", text.index("searchBaseLayer("))
    end = text.index("    void getNeighborsByHeuristic2", begin)
    region = text[begin:end]
    region = region.replace("#ifdef USE_SSE\n            _mm_prefetch", "#ifdef USE_SSE\n            if (size) {\n            _mm_prefetch")
    region = region.replace("            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);\n", "")
    region = region.replace("            _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);", "            if (size > 1) _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);")
    region = region.replace("\n#endif\n\n            for", "\n            }\n#endif\n\n            for")
    region = region.replace("#ifdef USE_SSE\n                _mm_prefetch((char *) (visited_array + *(datal + j + 1))", "#ifdef USE_SSE\n                if (j + 1 < size) {\n                _mm_prefetch((char *) (visited_array + *(datal + j + 1))")
    region = region.replace("                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);\n#endif", "                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);\n                }\n#endif")
    region = region.replace("#ifdef USE_SSE\n                _mm_prefetch((char *) (visited_array + *(data + j + 1))", "#ifdef USE_SSE\n                if (j < size) {\n                _mm_prefetch((char *) (visited_array + *(data + j + 1))")
    region = region.replace("_MM_HINT_T0);  ////////////\n#endif", "_MM_HINT_T0);  ////////////\n                }\n#endif")
    text = text[:begin] + region + text[end:]
    text = text.replace("                    _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);", "                    if (size) _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);")
    text = text.replace("                        _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);", "                        if (i + 1 < size) _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);")
    text = once(text, "        memcpy(result.data(), ll, size * sizeof(tableint));", "        if (size) memcpy(result.data(), ll, size * sizeof(tableint));")
    record("01-prefetch-bounds.patch", original, text)
    if family == "ivf_patched":
        # 02: per-call traversal vectors/distance counters. No mutable query
        # parameters, global/TLS context or alternative graph algorithm.
        original = text
        text = once(text, "    static const tableint MAX_LABEL_OPERATION_LOCKS", """    struct IVFTraversal {
        const char* vectors = nullptr;
        size_t stride = 0;
        DISTFUNC<dist_t> distance = nullptr;
        void* parameter = nullptr;
        uint64_t* computations = nullptr;
    };
    dist_t ivfDistance(const void* query, tableint id, const IVFTraversal* traversal) const {
        if (traversal && traversal->computations) ++*traversal->computations;
        if (traversal && traversal->vectors)
            return traversal->distance(query, traversal->vectors + id * traversal->stride, traversal->parameter);
        return fstdistfunc_(query, getDataByInternalId(id), dist_func_param_);
    }
    static const tableint MAX_LABEL_OPERATION_LOCKS""")
        text = once(text, "BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {", "BaseSearchStopCondition<dist_t>* stop_condition = nullptr,\n        const IVFTraversal* traversal = nullptr) const {")
        text = once(text, "dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);", "dist_t dist = ivfDistance(data_point, ep_id, traversal);")
        text = once(text, "dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);", "dist_t dist = ivfDistance(data_point, candidate_id, traversal);")
        record("02-traversal-context.patch", original, text)
        original = text
        # 03: supplied bottom-layer entry points, global fallback. Default
        # searchKnn delegates to exactly the original traversal path.
        old = "searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {"
        new = """searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
        return searchKnnIVF(query_data, k, isIdAllowed);
    }
    std::priority_queue<std::pair<dist_t, labeltype>>
    searchKnnIVF(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr,
                const IVFTraversal* traversal = nullptr, const std::vector<tableint>& entries = {}) const {"""
        text = once(text, old, new)
        begin = text.index("    searchKnnIVF(")
        end = text.index("    searchStopConditionClosest(", begin)
        region = text[begin:end]
        region = region.replace("fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_)", "ivfDistance(query_data, enterpoint_node_, traversal)")
        region = region.replace("fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_)", "ivfDistance(query_data, cand, traversal)")
        region = region.replace("        std::priority_queue<std::pair<dist_t, tableint>,", """        for (tableint entry : entries) {
            if (entry >= cur_element_count) throw std::runtime_error("Invalid IVF entry point");
            dist_t distance = ivfDistance(query_data, entry, traversal);
            if (distance < curdist) { currObj = entry; curdist = distance; }
        }
        std::priority_queue<std::pair<dist_t, tableint>,""", 1)
        region = region.replace("currObj, query_data, std::max(ef_, k), isIdAllowed);", "currObj, query_data, std::max(ef_, k), isIdAllowed, nullptr, traversal);")
        text = text[:begin] + region + text[end:]
        record("03-caller-entries.patch", original, text)
        original = text
        # 04: quiescent bottom-layer replacement and one ordinary reinsertion.
        text = once(text, "    void updatePoint(const void *dataPoint", """    void replaceBottomIVF(tableint id, const std::vector<tableint>& neighbours) {
        if (id >= cur_element_count || neighbours.size() > maxM0_)
            throw std::runtime_error("Invalid IVF bottom-layer replacement");
        std::unordered_set<tableint> unique;
        for (tableint target : neighbours)
            if (target >= cur_element_count || target == id || !unique.insert(target).second)
                throw std::runtime_error("Invalid IVF bottom-layer neighbour");
        linklistsizeint* links = get_linklist0(id);
        setListCount(links, neighbours.size());
        if (!neighbours.empty()) memcpy(links + 1, neighbours.data(), neighbours.size() * sizeof(tableint));
    }
    void reinsertIVF(tableint id) {
        if (id >= cur_element_count) throw std::runtime_error("Invalid IVF reinsertion ID");
        std::vector<char> vector(data_size_);
        memcpy(vector.data(), getDataByInternalId(id), data_size_);
        updatePoint(vector.data(), id, 1.0f);
    }
    void updatePoint(const void *dataPoint""")
        record("04-quiescent-edits.patch", original, text)
        original = text
        # 05: S09 training only. True seeded searches bypass global upper descent;
        # default W10 queries retain patch 03's behavior. Reuse replaces all
        # coordinates first, then uses the nearest eligible old neighbor as each
        # layer's entry into ordinary update connection/pruning.
        text = once(text, "const IVFTraversal* traversal = nullptr, const std::vector<tableint>& entries = {}) const {",
                    "const IVFTraversal* traversal = nullptr, const std::vector<tableint>& entries = {}, bool seeded = false) const {")
        begin = text.index("    searchKnnIVF(")
        end = text.index("    searchStopConditionClosest(", begin)
        region = text[begin:end]
        region = once(region, "        tableint currObj = enterpoint_node_;", """        tableint currObj = enterpoint_node_;
        if (seeded && !entries.empty()) {
            if (entries.front() >= cur_element_count) throw std::runtime_error("Invalid IVF seed");
            currObj = entries.front();
        }""")
        region = region.replace("ivfDistance(query_data, enterpoint_node_, traversal)", "ivfDistance(query_data, currObj, traversal)")
        region = once(region, "for (int level = maxlevel_; level > 0; level--)",
                      "for (int level = (seeded && !entries.empty()) ? 0 : maxlevel_; level > 0; level--)")
        text = text[:begin] + region + text[end:]
        text = once(text, "        int maxLevel) {", "        int maxLevel, const std::vector<tableint>* oldEntries = nullptr) {")
        text = once(text, """        tableint currObj = entryPointInternalId;
        if (dataPointLevel < maxLevel) {""", """        tableint currObj = entryPointInternalId;
        bool suppliedEntry = false;
        if (oldEntries) {
            dist_t best = std::numeric_limits<dist_t>::max();
            for (tableint entry : *oldEntries) {
                if (entry >= cur_element_count || entry == dataPointInternalId)
                    throw std::runtime_error("Invalid IVF reinsertion entry");
                if (element_levels_[entry] < dataPointLevel) continue;
                dist_t distance = fstdistfunc_(dataPoint, getDataByInternalId(entry), dist_func_param_);
                if (!suppliedEntry || distance < best) { best = distance; currObj = entry; suppliedEntry = true; }
            }
        }
        if (dataPointLevel < maxLevel && !suppliedEntry) {""")
        anchor = """        for (int level = dataPointLevel; level >= 0; level--) {
            std::priority_queue"""
        text = once(text, anchor, """        for (int level = dataPointLevel; level >= 0; level--) {
            if (oldEntries) {
                dist_t best = std::numeric_limits<dist_t>::max();
                for (tableint entry : *oldEntries) {
                    if (entry >= cur_element_count || entry == dataPointInternalId)
                        throw std::runtime_error("Invalid IVF reinsertion entry");
                    if (element_levels_[entry] < level) continue;
                    dist_t distance = fstdistfunc_(dataPoint, getDataByInternalId(entry), dist_func_param_);
                    if (distance < best) { best = distance; currObj = entry; }
                }
            }
            std::priority_queue""")
        text = once(text, "    void updatePoint(const void *dataPoint", """    void replaceVectorIVF(tableint id, const void* vector) {
        if (id >= cur_element_count) throw std::runtime_error("Invalid IVF vector replacement");
        memcpy(getDataByInternalId(id), vector, data_size_);
    }
    void reinsertFromIVF(tableint id, const std::vector<tableint>& oldEntries) {
        if (id >= cur_element_count) throw std::runtime_error("Invalid IVF training reinsertion");
        if (cur_element_count == 1) return;
        repairConnectionsForUpdate(getDataByInternalId(id), enterpoint_node_, id,
            element_levels_[id], maxlevel_, &oldEntries);
    }
    void updatePoint(const void *dataPoint""")
        record("05-ann-training-reuse.patch", original, text)
    path.write_text(text)
    if family == "ivf_patched":
        for header in directory.glob("*.h"):
            header.write_text(header.read_text().replace("namespace hnswlib", "namespace ivf_hnswlib").replace("hnswlib::", "ivf_hnswlib::"))
