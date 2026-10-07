// S02 prerequisite: exercise the real BufANN heap without a vector graph/PQ.
#include "bufann/inplace_backend.h"

#include <array>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

template<typename T>
void exercise(const std::filesystem::path& prefix) {
    using namespace diskann::inplace;
    constexpr uint32_t dim = 13, rows = 128;
    const auto heap = prefix.string() + ".heap";
    const auto meta = prefix.string() + ".meta";
    auto verify = [](InPlaceGraphStore& store) {
        require(store.max_degree() == 0 && store.n_chunks() == 0,
                "graph/PQ storage unexpectedly enabled");
        require(store.aligned_dim() == 16, "unexpected aligned dimension");
        require(store.num_active() == rows, "active count mismatch");
        require(!store.is_active(rows), "unpublished allocation became visible");
        for (uint32_t id = 0; id < rows; ++id) {
            require(store.is_active(id), "published node missing");
            require(store.node_tag(id) == 1000 + id, "external tag changed");
            NodeRID rid;
            auto frame = store.pin_page_for_node(id, READ, &rid);
            require(frame.valid(), "coordinate page missing");
            auto guard = frame.read_guard();
            auto node = guard.read_node(rid);
            require(node.valid() && node.node_id() == id && node.degree() == 0,
                    "slot identity/degree mismatch");
            // Heap slots are packed, not SIMD-aligned. Copy under the guard;
            // never pass their coordinate pointers to aligned distance kernels.
            alignas(64) std::array<T, 16> coords{};
            std::memcpy(coords.data(), node.coords_bytes(), sizeof(coords));
            require(reinterpret_cast<uintptr_t>(coords.data()) % 64 == 0,
                    "scratch is not aligned");
            for (uint32_t d = 0; d < dim; ++d)
                require(coords[d] == static_cast<T>(int((id + d) % 31) - 15),
                        "coordinate round-trip mismatch");
            for (uint32_t d = dim; d < 16; ++d)
                require(coords[d] == T{}, "padding is not zero");
        }
    };
    {
        InPlaceGraphStore store;
        store.init(dim, 0, sizeof(T), 4096, 2, heap, 32, 100, true, rows + 1);
        store.set_page_data_locks_enabled(true);
        for (uint32_t id = 0; id < rows; ++id) {
            alignas(64) std::array<T, 16> coords{};
            for (uint32_t d = 0; d < dim; ++d)
                coords[d] = static_cast<T>(int((id + d) % 31) - 15);
            store.allocate_node(id);
            require(!store.is_active(id), "allocation published prematurely");
            {
                NodeRID rid;
                auto frame = store.pin_page_for_node(id, WRITE, &rid);
                require(frame.valid(), "allocated page missing");
                auto guard = frame.write_guard();
                auto node = guard.write_node(rid);
                require(node.valid(), "allocated slot missing");
                node.set_coords(coords.data());
                node.set_degree(0);
                guard.commit();
            }
            require(store.set_node_tag(id, 1000 + id), "tag assignment failed");
            store.publish_node(id);
        }
        verify(store);
        store.flush();
        store.save_snapshot(meta); // Quiescent checkpoint, not crash recovery.
    }
    {
        InPlaceGraphStore reopened;
        reopened.init(dim, 0, sizeof(T), 4096, 2, heap, 32, 100, false, rows + 1);
        reopened.set_page_data_locks_enabled(true);
        reopened.load_snapshot(meta);
        verify(reopened);
    }
    // An interrupted allocation must not become active. The existing backend
    // intentionally rejects its checkpoint on reopen; do not claim rollback
    // or clean persistence of failed insertions without an adapter policy.
    const auto failed_heap = prefix.string() + ".unpublished.heap";
    const auto failed_meta = prefix.string() + ".unpublished.meta";
    {
        InPlaceGraphStore failed;
        failed.init(dim, 0, sizeof(T), 4096, 2, failed_heap, 32, 100, true, 1);
        failed.allocate_node(0);
        require(!failed.is_active(0) && failed.num_active() == 0,
                "interrupted allocation became active");
        failed.flush();
        failed.save_snapshot(failed_meta);
    }
    {
        InPlaceGraphStore failed;
        failed.init(dim, 0, sizeof(T), 4096, 2, failed_heap, 32, 100, false, 1);
        bool rejected = false;
        try {
            failed.load_snapshot(failed_meta);
        } catch (const diskann::ANNException& error) {
            rejected = error.message().find("inactive/delete-pending") != std::string::npos;
        }
        require(rejected, "unpublished checkpoint was not explicitly rejected");
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2, "expected fixture directory");
        std::filesystem::create_directories(argv[1]);
        exercise<float>(std::filesystem::path(argv[1]) / "float");
        exercise<int8_t>(std::filesystem::path(argv[1]) / "int8");
        exercise<uint8_t>(std::filesystem::path(argv[1]) / "uint8");
        std::cout << "Graph-free heap publication/coordinates/reopen passed for all dtypes\n";
        return 0;
    } catch (const diskann::ANNException& error) {
        std::cerr << error.message() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
