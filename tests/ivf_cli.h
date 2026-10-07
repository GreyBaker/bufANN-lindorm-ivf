#pragma once

#include <charconv>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

#include "bufann/ann_bench_metrics.h"

namespace ivf_cli {

class Arguments {
  public:
    Arguments(int argc, char** argv, const std::set<std::string>& allowed) {
        for (int i = 1; i < argc; ++i) {
            const std::string key(argv[i]);
            if (key == "--help") {
                if (argc != 2) throw std::invalid_argument("--help must be used alone");
                help = true;
                return;
            }
            if (!allowed.count(key)) throw std::invalid_argument("unknown flag: " + key);
            if (i + 1 == argc) throw std::invalid_argument("missing value for " + key);
            if (!values_.emplace(key, argv[++i]).second) throw std::invalid_argument("duplicate flag: " + key);
        }
    }

    std::string get(const std::string& key, const std::string& fallback = "") const {
        const auto it = values_.find(key);
        return it == values_.end() ? fallback : it->second;
    }

    std::string require(const std::string& key) const {
        const std::string value = get(key);
        if (value.empty()) throw std::invalid_argument("required flag: " + key);
        return value;
    }

    uint64_t integer(const std::string& key, uint64_t fallback = 0) const {
        const auto it = values_.find(key);
        if (it == values_.end()) return fallback;
        uint64_t value = 0;
        const auto& text = it->second;
        const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
        if (parsed.ec != std::errc() || parsed.ptr != text.data() + text.size()) {
            throw std::invalid_argument("invalid unsigned integer for " + key);
        }
        return value;
    }

    uint32_t u32(const std::string& key, uint32_t fallback = 0) const {
        const uint64_t value = integer(key, fallback);
        if (value > std::numeric_limits<uint32_t>::max()) throw std::invalid_argument("uint32 overflow for " + key);
        return static_cast<uint32_t>(value);
    }

    double real(const std::string& key, double fallback) const {
        const auto it = values_.find(key);
        if (it == values_.end()) return fallback;
        size_t consumed = 0;
        const double value = std::stod(it->second, &consumed);
        if (consumed != it->second.size() || !std::isfinite(value)) {
            throw std::invalid_argument("invalid finite number for " + key);
        }
        return value;
    }

    void require_validation_mode() const {
        const uint32_t mode = u32("--validate_inputs");
        if (mode > 1) throw std::invalid_argument("validate_inputs must be 0 or 1");
        if (mode != 1) {
            throw std::runtime_error("IVF execution is not implemented yet; use --validate_inputs 1 for input checks");
        }
    }

    bool help = false;

  private:
    std::map<std::string, std::string> values_;
};

inline void validation_record(const char* operation, const char* dtype,
                              uint32_t dim, uint32_t rows, uint32_t chunks) {
    bool first = true;
    std::cout << "{";
    bufann_bench::json_kv(std::cout, first, "baseline", "IVF");
    bufann_bench::json_kv(std::cout, first, "phase", "input_validation");
    bufann_bench::json_kv(std::cout, first, "operation", operation);
    bufann_bench::json_kv(std::cout, first, "data_type", dtype);
    bufann_bench::json_kv(std::cout, first, "dim", dim);
    bufann_bench::json_kv(std::cout, first, "rows", rows);
    bufann_bench::json_kv(std::cout, first, "pq_chunks", chunks);
    std::cout << "}" << std::endl;
}

}  // namespace ivf_cli
