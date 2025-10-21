// Micro-benchmark for QuantizedTensor block & tile decode performance.
// Focus: relative throughput across formats & access patterns, LRU cache hit rates (approx).
// Usage: ctest -R QuantDecodeBench OR run binary directly for formatted output.

#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <sstream>
#include <algorithm>
#include "../src/tensors/TensorFactory.h"
#include "../src/QuantDequant.h"
#include "../llama.cpp/ggml/src/ggml-quants.h"

using namespace llaminar;

namespace
{
    struct BenchResult
    {
        std::string format;
        std::string pattern;
        size_t elements;       // number of decoded output elements
        double ms;             // wall clock ms
        double gbps;           // effective GB/s of decoded float bytes (approx)
        double blocks_decoded; // approximate block decodes (heuristic)
        uint64_t block_requests{0};
        uint64_t block_hits{0};
        uint64_t block_misses{0};
        uint64_t full_block_fastpath{0};
        double hit_rate{0.0};
    };

    template <typename F>
    double time_ms(F &&fn)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    std::vector<uint8_t> synthesize_block_q4_0()
    {
        // scale + 16 bytes
        std::vector<uint8_t> b(18);
        uint16_t scale = 0x3C00; // 1.0f
        std::memcpy(b.data(), &scale, 2);
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, 255);
        for (int i = 0; i < 16; ++i)
            b[2 + i] = (uint8_t)dist(rng);
        return b;
    }

    std::shared_ptr<QuantizedTensor> make_tensor(QuantFormat fmt, int rows, int cols)
    {
        // Build raw storage with repeated synthetic blocks appropriate to format.
        QuantBlockDescriptor desc{};
        switch (fmt)
        {
        case QuantFormat::Q4_0:
            desc.elements_per_block = 32;
            desc.bytes_per_block = 18;
            break;
        case QuantFormat::Q5_0:
            desc.elements_per_block = 32;
            desc.bytes_per_block = 20;
            break; // actual size used elsewhere, OK for bench
        case QuantFormat::Q8_0:
            desc.elements_per_block = 32;
            desc.bytes_per_block = 34;
            break;
        case QuantFormat::Q4_K:
            desc.elements_per_block = QK_K;
            desc.bytes_per_block = sizeof(block_q4_K);
            desc.is_k_quant = true;
            break;
        case QuantFormat::Q5_K:
            desc.elements_per_block = QK_K;
            desc.bytes_per_block = sizeof(block_q5_K);
            desc.is_k_quant = true;
            break;
        case QuantFormat::Q6_K:
            desc.elements_per_block = QK_K;
            desc.bytes_per_block = sizeof(block_q6_K);
            desc.is_k_quant = true;
            break;
        case QuantFormat::Q8_K:
            desc.elements_per_block = QK_K;
            desc.bytes_per_block = sizeof(block_q8_K);
            desc.is_k_quant = true;
            break;
        default:
            return nullptr;
        }
        size_t blocks_per_row = (size_t)cols / desc.elements_per_block;
        if ((size_t)cols % desc.elements_per_block)
            ++blocks_per_row;
        size_t total_blocks = (size_t)rows * blocks_per_row;
        std::vector<uint8_t> raw(total_blocks * (size_t)desc.bytes_per_block, 0);
        // Populate with simple deterministic pattern (scales=1, data random) to avoid branchy decode behavior variance.
        std::mt19937 rng(123);
        for (size_t b = 0; b < total_blocks; ++b)
        {
            uint8_t *ptr = raw.data() + b * desc.bytes_per_block;
            if (fmt == QuantFormat::Q4_0)
            {
                uint16_t h = 0x3C00;
                std::memcpy(ptr, &h, 2);
                for (int i = 0; i < 16; ++i)
                    ptr[2 + i] = (uint8_t)rng();
            }
            else if (fmt == QuantFormat::Q8_0)
            {
                uint16_t h = 0x3C00;
                std::memcpy(ptr, &h, 2);
                for (int i = 0; i < 32; ++i)
                    ptr[2 + i] = (uint8_t)rng();
            }
            else
            {
                // For K formats (and Q5_0), leave bytes pseudo-random; decoder will treat as structured.
                for (int i = 0; i < desc.bytes_per_block; ++i)
                    ptr[i] = (uint8_t)rng();
            }
        }
        QuantStorageLayout layout{fmt, {rows, cols}, total_blocks, desc};
        return std::make_shared<QuantizedTensor>(layout, raw);
    }

    BenchResult bench_tile(const std::shared_ptr<QuantizedTensor> &qt, int rows, int cols, int tile_h, int tile_w, const std::string &pattern, int iterations)
    {
        std::vector<_Float16> tile((size_t)tile_h * tile_w);
        auto &shape = qt->layout().original_shape;
        int max_r = shape[0] - tile_h;
        int max_c = shape[1] - tile_w;
        std::mt19937 rng(77);
        std::uniform_int_distribution<int> dist_r(0, std::max(0, max_r));
        std::uniform_int_distribution<int> dist_c(0, std::max(0, max_c));
        size_t blocks_decoded_samples = 0; // placeholder (kept for future refinement)

        QuantizedTensor::reset_cache_stats();
        QuantizedTensor::enable_cache_stats(true);

        double ms = time_ms([&]()
                            {
        for (int it = 0; it < iterations; ++it) {
            int r0, c0;
            if (pattern == "sequential") {
                r0 = (it * tile_h) % std::max(1, shape[0]-tile_h+1);
                c0 = 0;
            } else if (pattern == "row-scan") {
                r0 = 0;
                c0 = (it * tile_w) % std::max(1, shape[1]-tile_w+1);
            } else { // random
                r0 = dist_r(rng);
                c0 = dist_c(rng);
            }
            qt->decodeTileFP16(r0, tile_h, c0, tile_w, tile.data());
        } });

        // Approximation: treat each tile as touching ceil(tile_h * tile_w / elements_per_block) blocks.
        const int epb = qt->layout().block_desc.elements_per_block;
        double approx_blocks = (double)iterations * std::ceil((double)(tile_h * tile_w) / epb);
        double bytes = (double)iterations * tile_h * tile_w * sizeof(_Float16);
        double gbps = (bytes / (ms / 1000.0)) / 1e9;
        auto stats = QuantizedTensor::cache_stats();
        QuantizedTensor::enable_cache_stats(false);
        double hr = stats.block_requests ? (double)stats.block_hits / (double)stats.block_requests : 0.0;
        return {std::to_string((int)qt->layout().format), pattern, (size_t)iterations * (size_t)tile_h * tile_w, ms, gbps, approx_blocks,
                stats.block_requests, stats.block_hits, stats.block_misses, stats.full_block_fastpath, hr};
    }

    void print_header(bool stats)
    {
        std::cout << std::left << std::setw(6) << "FMT" << std::setw(12) << "PATTERN"
                  << std::setw(10) << "ELEMS" << std::setw(10) << "MS" << std::setw(10) << "GB/s" << std::setw(12) << "APPX_BLOCKS";
        if (stats)
        {
            std::cout << std::setw(14) << "REQ" << std::setw(14) << "HITS" << std::setw(14) << "MISSES" << std::setw(10) << "FASTPATH" << std::setw(10) << "HIT%";
        }
        std::cout << "\n";
    }

    void print_format_legend()
    {
        std::cout << "Format Legend: 2=Q4_0 3=Q5_0 4=Q8_0 5=Q4_K 6=Q5_K 7=Q6_K 8=Q8_K" << "\n";
    }

    void print_aggregate_summary(const std::vector<BenchResult> &rows, bool stats)
    {
        if (!stats || rows.empty())
            return;
        double total_requests = 0, total_hits = 0, total_misses = 0;
        uint64_t fastpath = 0;
        for (auto &r : rows)
        {
            total_requests += r.block_requests;
            total_hits += r.block_hits;
            total_misses += r.block_misses;
            fastpath += r.full_block_fastpath;
        }
        double hr = total_requests ? total_hits / total_requests : 0.0;
        std::cout << "\nAggregate: requests=" << (uint64_t)total_requests
                  << " hits=" << (uint64_t)total_hits
                  << " misses=" << (uint64_t)total_misses
                  << " hit%=" << std::fixed << std::setprecision(2) << (hr * 100.0)
                  << " fastpath_calls=" << fastpath << "\n";
    }

    void print_row(const BenchResult &r, bool stats)
    {
        std::cout << std::left << std::setw(6) << r.format
                  << std::setw(12) << r.pattern
                  << std::setw(10) << r.elements
                  << std::setw(10) << std::fixed << std::setprecision(2) << r.ms
                  << std::setw(10) << std::fixed << std::setprecision(2) << r.gbps
                  << std::setw(12) << (long long)r.blocks_decoded;
        if (stats)
        {
            std::cout << std::setw(14) << r.block_requests
                      << std::setw(14) << r.block_hits
                      << std::setw(14) << r.block_misses
                      << std::setw(10) << r.full_block_fastpath
                      << std::setw(10) << std::fixed << std::setprecision(1) << (r.hit_rate * 100.0);
        }
        std::cout << "\n";
    }

    void print_csv(const std::vector<BenchResult> &rows, bool stats)
    {
        std::cout << "format,pattern,elements,ms,gbps,approx_blocks";
        if (stats)
            std::cout << ",block_requests,block_hits,block_misses,full_block_fastpath,hit_rate";
        std::cout << "\n";
        for (auto &r : rows)
        {
            std::cout << r.format << ',' << r.pattern << ',' << r.elements << ','
                      << std::fixed << std::setprecision(4) << r.ms << ','
                      << std::fixed << std::setprecision(6) << r.gbps << ','
                      << r.blocks_decoded;
            if (stats)
            {
                std::cout << ',' << r.block_requests << ',' << r.block_hits
                          << ',' << r.block_misses << ',' << r.full_block_fastpath << ',' << std::setprecision(6) << r.hit_rate;
            }
            std::cout << "\n";
        }
    }

    struct CmdOpts
    {
        size_t elements = 0; // if non-zero override rows*cols for base cases (square assumption for simplicity)
        int iterations = 200;
        std::vector<int> formats; // numeric QuantFormat enums
        std::vector<std::string> patterns{"sequential", "row-scan", "random"};
        bool csv = false;
        bool stats = true;
    };

    CmdOpts parse(int argc, char **argv)
    {
        CmdOpts o;
        for (int i = 1; i < argc; ++i)
        {
            std::string a = argv[i];
            auto next = [&]()
            { return (i + 1 < argc) ? argv[++i] : (char *)""; };
            if (a == "--elements" || a == "-e")
            {
                o.elements = std::stoull(next());
            }
            else if (a == "--iterations" || a == "-n")
            {
                o.iterations = std::stoi(next());
            }
            else if (a == "--formats" || a == "-f")
            {
                std::stringstream ss(next());
                std::string tok;
                while (std::getline(ss, tok, ','))
                    o.formats.push_back(std::stoi(tok));
            }
            else if (a == "--patterns" || a == "-p")
            {
                o.patterns.clear();
                std::stringstream ss(next());
                std::string tok;
                while (std::getline(ss, tok, ','))
                    o.patterns.push_back(tok);
            }
            else if (a == "--no-stats")
            {
                o.stats = false;
            }
            else if (a == "--csv")
            {
                o.csv = true;
            }
            else if (a == "--help" || a == "-h")
            {
                std::cout << "Quant decode micro-benchmark options:\n"
                          << "  -e, --elements N       approximate square matrix with N total elements (rows=cols=sqrt(N))\n"
                          << "  -n, --iterations I     iterations per pattern (default 200)\n"
                          << "  -f, --formats list     comma list of format ids (2=Q4_0,4=Q5_0,5=Q8_0,6=Q4_K,7=Q5_K,8=Q6_K,9=Q8_K)\n"
                          << "  -p, --patterns list    comma list of access patterns (sequential,row-scan,random)\n"
                          << "      --no-stats         disable cache stats collection\n"
                          << "      --csv              CSV output\n"
                          << "  -h, --help             show this help\n";
                std::exit(0);
            }
        }
        return o;
    }

    std::string fmt_name(int id)
    {
        switch (id)
        {
        case 2:
            return "Q4_0";
        case 3:
            return "Q5_0";
        case 4:
            return "Q8_0"; // adjust if enum mapping changes
        case 5:
            return "Q4_K";
        case 6:
            return "Q5_K";
        case 7:
            return "Q6_K";
        case 8:
            return "Q8_K";
        default:
            return std::to_string(id);
        }
    }

    int run(int argc, char **argv)
    {
        CmdOpts opts = parse(argc, argv);
        // Default formats if none specified (subset representative)
        if (opts.formats.empty())
            opts.formats = {(int)QuantFormat::Q4_0, (int)QuantFormat::Q8_0, (int)QuantFormat::Q4_K, (int)QuantFormat::Q6_K};
        // Build cases: heuristic tile sizes based on format
        struct Case
        {
            int id;
            QuantFormat fmt;
            int rows;
            int cols;
            int th;
            int tw;
        };
        std::vector<Case> cases;
        for (int id : opts.formats)
        {
            QuantFormat fmt = static_cast<QuantFormat>(id);
            int base_rows = 512, base_cols = 512;
            if (fmt == QuantFormat::Q4_K || fmt == QuantFormat::Q5_K || fmt == QuantFormat::Q6_K || fmt == QuantFormat::Q8_K)
                base_cols = 1024;
            if (opts.elements)
            {
                size_t side = (size_t)std::sqrt((double)opts.elements);
                base_rows = (int)side;
                base_cols = (int)side;
            }
            int th = 16;
            int tw = (fmt == QuantFormat::Q4_K || fmt == QuantFormat::Q5_K || fmt == QuantFormat::Q6_K || fmt == QuantFormat::Q8_K) ? 64 : 32;
            cases.push_back({id, fmt, base_rows, base_cols, th, tw});
        }

        std::vector<BenchResult> results;
        for (auto &cs : cases)
        {
            auto qt = make_tensor(cs.fmt, cs.rows, cs.cols);
            if (!qt)
                continue;
            for (auto &pat : opts.patterns)
            {
                results.push_back(bench_tile(qt, cs.rows, cs.cols, cs.th, cs.tw, pat, opts.iterations));
                // Overwrite format string with human readable name
                results.back().format = fmt_name(cs.id);
            }
        }

        if (opts.csv)
        {
            print_csv(results, opts.stats);
        }
        else
        {
            print_format_legend();
            print_header(opts.stats);
            for (auto &r : results)
                print_row(r, opts.stats);
            print_aggregate_summary(results, opts.stats);
        }
        return 0;
    }
} // namespace

int main(int argc, char **argv) { return run(argc, argv); }
