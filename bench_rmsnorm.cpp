#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <cstring>
#include "src/kernels/common/rmsnorm_core.h"
#include "src/utils/debug_env.h"

using namespace llaminar::kernels;

struct BenchResult
{
    double ms;
    double gbytes;
    double calls;
};

static BenchResult bench_case(size_t rows, size_t cols, size_t iters, bool with_gamma)
{
    std::vector<float> src(rows * cols), dst(rows * cols), gamma(rows * cols, 1.0f);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto &v : src)
        v = dist(rng);
    if (with_gamma)
        for (size_t c = 0; c < cols; c++)
            gamma[c] = 0.5f + 0.5f * dist(rng);
    RMSNormExecOptions opts; // leave defaults (may be overridden by env flags inside fused impl)
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iters; i++)
    {
        rmsnorm_row_major_fused(src.data(), with_gamma ? gamma.data() : nullptr, dst.data(), rows, cols, 1e-5f, GammaMode::REPLICATED, 0, opts);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    double bytes = (double)(rows * cols * sizeof(float)) * 2 + (with_gamma ? cols * sizeof(float) : 0); // rough IO volume per call
    double gbytes = (bytes * iters) / 1e9;
    return {ms, gbytes, (double)iters};
}

int main(int argc, char **argv)
{
    bool all_variants = false;
    for (int i = 1; i < argc; i++)
    {
        if (std::string(argv[i]) == "--all" || std::string(argv[i]) == "--variants")
            all_variants = true;
    }
    if (const char *av = std::getenv("LLAMINAR_RMSNORM_BENCH_ALL"))
    {
        if (*av)
            all_variants = true;
    }
    const auto &env = llaminar::debugEnv();
    bool probe = env.rmsnorm.false_sharing_probe;
    struct Shape
    {
        size_t rows;
        size_t cols;
        size_t iters;
    };
    std::vector<Shape> shapes = {
        {1, 4096, 50000},
        {1, 8192, 25000},
        {4, 4096, 10000},
        {16, 4096, 4000},
        {64, 4096, 1000},
        {128, 4096, 400}};
    if (probe)
    {
        // Additional shapes to stress potential false sharing: many rows, small cols
        shapes.push_back({256, 256, 2000});
        shapes.push_back({512, 256, 1000});
    }
    std::cout << "# RMSNorm Bench (env controls: force_scalar=" << env.rmsnorm.force_scalar
              << " tls_scratch=" << (env.rmsnorm.disable_tls_scratch ? "off" : "on")
              << " prealloc_rows=" << env.rmsnorm.scratch_prealloc_rows
              << ")\n";
    std::cout << "# mode=" << (all_variants ? "all_variants" : "auto_only") << " (use --all or LLAMINAR_RMSNORM_BENCH_ALL=1 for full sweep)" << std::endl;
    std::cout << "rows,cols,iters,variant,ms,total_GB,GB/s,speedup_vs_auto" << std::endl;
    struct Variant
    {
        const char *name;
        const char *vec_impl;
        const char *fast_acc;
    };
    std::vector<Variant> variants;
    if (all_variants)
    {
        variants.reserve(7);
        variants.push_back({"auto", "0", "0"});
        variants.push_back({"scalar", "1", "0"});
        variants.push_back({"avx2", "2", "0"});
        variants.push_back({"avx512", "3", "0"});
        variants.push_back({"auto_fastacc", "0", "1"});
        variants.push_back({"avx2_fastacc", "2", "1"});
        variants.push_back({"avx512_fastacc", "3", "1"});
    }
    else
    {
        variants.reserve(1);
        variants.push_back({"auto", "0", "0"});
    }
    for (auto sh : shapes)
    {
        struct RowOut
        {
            const char *name;
            double ms;
            double gbytes;
            double gbps;
        };
        std::vector<RowOut> out_rows;
        out_rows.reserve(variants.size());
        for (const auto &v : variants)
        {
            setenv("LLAMINAR_RMSNORM_VEC_IMPL", v.vec_impl, 1);
            setenv("LLAMINAR_RMSNORM_FAST_ACC", v.fast_acc, 1);
            llaminar::debugEnvRefresh();
            auto r = bench_case(sh.rows, sh.cols, sh.iters, true);
            double gbps = r.gbytes / (r.ms / 1000.0);
            out_rows.push_back({v.name, r.ms, r.gbytes, gbps});
        }
        double base = 0.0;
        for (auto &ro : out_rows)
            if (std::string(ro.name) == "auto")
            {
                base = ro.gbps;
                break;
            }
        if (base <= 0.0)
            base = out_rows.front().gbps; // fallback
        for (auto &ro : out_rows)
        {
            double speedup = (base > 0.0) ? (ro.gbps / base) : 0.0;
            std::cout << sh.rows << ',' << sh.cols << ',' << sh.iters << ',' << ro.name << ',' << ro.ms << ',' << ro.gbytes << ',' << ro.gbps << ',' << speedup << std::endl;
        }
    }
    return 0;
}
