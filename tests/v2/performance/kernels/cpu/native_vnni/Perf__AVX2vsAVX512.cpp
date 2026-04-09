/**
 * @file Perf__AVX2vsAVX512.cpp
 * @brief Performance comparison of AVX2 emulated VNNI vs native AVX512-VNNI
 *        for GEMV (M=1) and GEMM (M>1) across all quantization formats.
 *
 * Measures wall-clock time for both ISA paths using the same packed weights
 * and activations, reporting slowdown ratio (AVX2 time / AVX512 time).
 *
 * Run with Release build:
 *   ctest --test-dir build_v2_release -R V2_Perf_AVX2vsAVX512 -V
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "kernels/cpu/native_vnni/CPUNativeVNNIGemv.h"
#include "kernels/cpu/native_vnni/CPUNativeVNNIWeightPacker.h"
#include "tensors/BlockStructures.h"
#include "tensors/SIMDHelpers.h"
#include "utils/CPUFeatures.h"
#include "fort.hpp"

#include "utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::cpu::native_vnni;
using namespace llaminar2::test;

namespace
{

// ============================================================================
// Configuration
// ============================================================================

static constexpr int WARMUP_ITERS = 100;
static constexpr int BENCH_ITERS = 500;

// Qwen model-relevant shapes: {N, K}
struct GemvShape
{
    int N, K;
    const char *label;
};

static const GemvShape GEMV_SHAPES[] = {
    {896, 896, "Qwen0.5B QKV"},
    {4864, 896, "Qwen0.5B FFN_Gate"},
    {896, 4864, "Qwen0.5B FFN_Down"},
    {3584, 3584, "Qwen7B QKV"},
    {18944, 3584, "Qwen7B FFN_Gate"},
    {3584, 18944, "Qwen7B FFN_Down"},
};

static const int GEMM_M_VALUES[] = {2, 4, 8, 16};

// Format descriptor for test instantiation
struct FormatDesc
{
    const char *name;
    const char *category; // "Nibble-LUT", "INT8", "K-quant", "IQ"
    // Factory function creates a random tensor of the given shape
    std::function<std::unique_ptr<TensorBase>(int N, int K, uint32_t seed)> create;
};

// All supported quantized formats
static const std::vector<FormatDesc> ALL_FORMATS = {
    // Nibble-LUT (4-bit, runtime vpshufb decode)
    {"Q4_0", "Nibble-LUT",
     [](int N, int K, uint32_t s)
     { return std::unique_ptr<TensorBase>(TestTensorFactory::createQ4_0Random({(size_t)N, (size_t)K}, s).release()); }},
    {"Q4_1", "Nibble-LUT",
     [](int N, int K, uint32_t s)
     { return std::unique_ptr<TensorBase>(TestTensorFactory::createQ4_1Random({(size_t)N, (size_t)K}, s).release()); }},
    {"IQ4_NL", "Nibble-LUT",
     [](int N, int K, uint32_t s)
     { return std::unique_ptr<TensorBase>(TestTensorFactory::createIQ4_NLRandom({(size_t)N, (size_t)K}, s).release()); }},
    {"IQ4_XS", "Nibble-LUT",
     [](int N, int K, uint32_t s)
     { return std::unique_ptr<TensorBase>(TestTensorFactory::createIQ4_XSRandom({(size_t)N, (size_t)K}, s).release()); }},

    // INT8 pre-decoded (per-block, 32-element)
    {"Q5_0", "INT8",
     [](int N, int K, uint32_t s)
     { return std::unique_ptr<TensorBase>(TestTensorFactory::createQ5_0Random({(size_t)N, (size_t)K}, s).release()); }},
    {"Q5_1", "INT8",
     [](int N, int K, uint32_t s)
     { return std::unique_ptr<TensorBase>(TestTensorFactory::createQ5_1Random({(size_t)N, (size_t)K}, s).release()); }},

    // K-quant (256-element superblocks, INT8 pre-decoded)
    {"Q6_K", "K-quant",
     [](int N, int K, uint32_t s)
     { return std::unique_ptr<TensorBase>(TestTensorFactory::createQ6_KRandom({(size_t)N, (size_t)K}, s).release()); }},
    {"Q5_K", "K-quant",
     [](int N, int K, uint32_t s)
     { return std::unique_ptr<TensorBase>(TestTensorFactory::createQ5_KRandom({(size_t)N, (size_t)K}, s).release()); }},
    {"Q4_K", "K-quant",
     [](int N, int K, uint32_t s)
     { return std::unique_ptr<TensorBase>(TestTensorFactory::createQ4_KRandom({(size_t)N, (size_t)K}, s).release()); }},
    {"Q3_K", "K-quant",
     [](int N, int K, uint32_t s)
     { return std::unique_ptr<TensorBase>(TestTensorFactory::createQ3_KRandom({(size_t)N, (size_t)K}, s).release()); }},
    {"Q2_K", "K-quant",
     [](int N, int K, uint32_t s)
     { return std::unique_ptr<TensorBase>(TestTensorFactory::createQ2_KRandom({(size_t)N, (size_t)K}, s).release()); }},

    // IQ formats (256-element superblocks, INT8 pre-decoded)
    {"IQ3_S", "IQ",
     [](int N, int K, uint32_t s)
     { return std::unique_ptr<TensorBase>(TestTensorFactory::createIQ3_SRandom({(size_t)N, (size_t)K}, s).release()); }},
    {"IQ3_XXS", "IQ",
     [](int N, int K, uint32_t s)
     { return std::unique_ptr<TensorBase>(TestTensorFactory::createIQ3_XXSRandom({(size_t)N, (size_t)K}, s).release()); }},
    {"IQ2_S", "IQ",
     [](int N, int K, uint32_t s)
     { return std::unique_ptr<TensorBase>(TestTensorFactory::createIQ2_SRandom({(size_t)N, (size_t)K}, s).release()); }},
    {"IQ2_XS", "IQ",
     [](int N, int K, uint32_t s)
     { return std::unique_ptr<TensorBase>(TestTensorFactory::createIQ2_XSRandom({(size_t)N, (size_t)K}, s).release()); }},
    {"IQ2_XXS", "IQ",
     [](int N, int K, uint32_t s)
     { return std::unique_ptr<TensorBase>(TestTensorFactory::createIQ2_XXSRandom({(size_t)N, (size_t)K}, s).release()); }},
    {"IQ1_S", "IQ",
     [](int N, int K, uint32_t s)
     { return std::unique_ptr<TensorBase>(TestTensorFactory::createIQ1_SRandom({(size_t)N, (size_t)K}, s).release()); }},
    {"IQ1_M", "IQ",
     [](int N, int K, uint32_t s)
     { return std::unique_ptr<TensorBase>(TestTensorFactory::createIQ1_MRandom({(size_t)N, (size_t)K}, s).release()); }},
};

// Create random Q8_1 activation blocks
std::vector<Q8_1Block> createRandomQ8_1(int K, int M = 1, uint32_t seed = 42)
{
    int K_blocks = (K + 31) / 32;
    std::vector<Q8_1Block> blocks(static_cast<size_t>(M) * K_blocks);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-127, 127);
    std::uniform_real_distribution<float> scale_dist(0.001f, 0.5f);

    for (auto &blk : blocks)
    {
        blk.d = simd::fp32_to_fp16(scale_dist(rng));
        int32_t sum = 0;
        for (int i = 0; i < 32; ++i)
        {
            blk.qs[i] = static_cast<int8_t>(dist(rng));
            sum += blk.qs[i];
        }
        blk.sum_qs = static_cast<int16_t>(std::clamp(sum, -32768, 32767));
    }
    return blocks;
}

// Benchmark a callable, return median time in microseconds
template <typename Fn>
double benchmarkMedianUs(Fn &&fn, int warmup, int iters)
{
    for (int i = 0; i < warmup; ++i)
    {
        fn();
        asm volatile("" ::: "memory");
    }

    std::vector<double> times(iters);
    for (int i = 0; i < iters; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        fn();
        asm volatile("" ::: "memory");
        auto end = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::micro>(end - start).count();
    }

    std::sort(times.begin(), times.end());
    return times[iters / 2]; // median
}

// Format a double with fixed precision into a string
std::string fmt(double v, int prec = 1)
{
    char buf[64];
    snprintf(buf, sizeof(buf), "%.*f", prec, v);
    return buf;
}

// ============================================================================
// Result collection
// ============================================================================

struct BenchResult
{
    std::string format;
    std::string category;
    std::string shape_label;
    int N, K, M;
    double avx512_us;
    double avx2_us;
    double ratio; // avx2_us / avx512_us
};

} // namespace

// ============================================================================
// Test fixture
// ============================================================================

class Perf__AVX2vsAVX512 : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (!cpu_supports_avx512_vnni())
            GTEST_SKIP() << "AVX512-VNNI not available";
    }
};

// ============================================================================
// GEMV benchmark (M=1) — all formats × representative shapes
// ============================================================================

TEST_F(Perf__AVX2vsAVX512, GEMV_AllFormats)
{
    // Use two representative shapes: small (Qwen0.5B FFN_Gate) and large (Qwen7B FFN_Gate)
    const GemvShape shapes[] = {
        {896, 896, "Small (896×896)"},
        {4864, 896, "Med (4864×896)"},
        {896, 4864, "Wide-K (896×4864)"},
        {18944, 3584, "Large (18944×3584)"},
    };

    std::vector<BenchResult> results;

    for (const auto &fmt : ALL_FORMATS)
    {
        for (const auto &shape : shapes)
        {
            auto weights = fmt.create(shape.N, shape.K, 42);
            CPUNativeVNNIPackedWeights packed;
            if (!packWeightsCPUNativeVNNI(weights.get(), packed))
                continue;

            auto A_q8 = createRandomQ8_1(shape.K, 1, 99);
            std::vector<float> out(shape.N);

            double t512 = benchmarkMedianUs(
                [&]()
                {
                    std::memset(out.data(), 0, out.size() * sizeof(float));
                    gemv_native_vnni_preq(packed, A_q8.data(), out.data(), ISAPath::AVX512);
                },
                WARMUP_ITERS, BENCH_ITERS);

            double t256 = benchmarkMedianUs(
                [&]()
                {
                    std::memset(out.data(), 0, out.size() * sizeof(float));
                    gemv_native_vnni_preq(packed, A_q8.data(), out.data(), ISAPath::AVX2);
                },
                WARMUP_ITERS, BENCH_ITERS);

            results.push_back({fmt.name, fmt.category, shape.label,
                               shape.N, shape.K, 1, t512, t256, t256 / t512});
        }
    }

    // Render results table
    fort::utf8_table table;
    table.set_border_style(FT_DOUBLE2_STYLE);
    table << fort::header
          << "Format" << "Category" << "Shape" << "AVX512 (µs)" << "AVX2 (µs)" << "Ratio"
          << fort::endr;

    table.column(0).set_cell_text_align(fort::text_align::left);
    table.column(1).set_cell_text_align(fort::text_align::left);
    table.column(2).set_cell_text_align(fort::text_align::left);
    for (int c = 3; c <= 5; ++c)
        table.column(c).set_cell_text_align(fort::text_align::right);

    std::string prev_cat;
    for (const auto &r : results)
    {
        if (!prev_cat.empty() && r.category != prev_cat)
            table << fort::separator;
        prev_cat = r.category;

        table << r.format << r.category << r.shape_label
              << fmt(r.avx512_us, 1) << fmt(r.avx2_us, 1)
              << (fmt(r.ratio, 2) + "×")
              << fort::endr;
    }

    // Summary row
    double sum_ratio = 0;
    for (const auto &r : results)
        sum_ratio += r.ratio;
    double avg_ratio = sum_ratio / results.size();

    table << fort::separator;
    table << "AVERAGE" << "" << "" << "" << "" << (fmt(avg_ratio, 2) + "×") << fort::endr;

    std::cout << "\n"
              << table.to_string() << std::endl;

    // Category averages
    std::map<std::string, std::pair<double, int>> cat_stats;
    for (const auto &r : results)
    {
        cat_stats[r.category].first += r.ratio;
        cat_stats[r.category].second++;
    }

    fort::utf8_table cat_table;
    cat_table.set_border_style(FT_DOUBLE2_STYLE);
    cat_table << fort::header << "Category" << "Avg Ratio" << "Tests" << fort::endr;
    cat_table.column(0).set_cell_text_align(fort::text_align::left);
    cat_table.column(1).set_cell_text_align(fort::text_align::right);
    cat_table.column(2).set_cell_text_align(fort::text_align::right);

    for (const auto &[cat, stats] : cat_stats)
    {
        double avg = stats.first / stats.second;
        cat_table << cat << (fmt(avg, 2) + "×") << std::to_string(stats.second) << fort::endr;
    }
    cat_table << fort::separator;
    cat_table << "OVERALL" << (fmt(avg_ratio, 2) + "×") << std::to_string((int)results.size()) << fort::endr;

    std::cout << cat_table.to_string() << std::endl;
}

// ============================================================================
// GEMM benchmark (M>1) — representative formats × M values
// ============================================================================

TEST_F(Perf__AVX2vsAVX512, GEMM_RepresentativeFormats)
{
    // Representative subset: one from each category
    const std::vector<std::string> fmt_names = {
        "Q4_0", "IQ4_NL", "Q5_0", "Q6_K", "Q3_K", "IQ3_S", "IQ2_S", "IQ1_S"};

    const GemvShape shape = {4864, 896, "Qwen0.5B FFN_Gate"};

    std::vector<BenchResult> results;

    for (const auto &fmt : ALL_FORMATS)
    {
        bool selected = false;
        for (const auto &name : fmt_names)
            if (name == fmt.name)
                selected = true;
        if (!selected)
            continue;

        auto weights = fmt.create(shape.N, shape.K, 42);
        CPUNativeVNNIPackedWeights packed;
        if (!packWeightsCPUNativeVNNI(weights.get(), packed))
            continue;

        for (int M : GEMM_M_VALUES)
        {
            auto A_q8 = createRandomQ8_1(shape.K, M, 99);
            std::vector<float> out(static_cast<size_t>(M) * shape.N);

            double t512 = benchmarkMedianUs(
                [&]()
                {
                    std::memset(out.data(), 0, out.size() * sizeof(float));
                    gemm_native_vnni_preq(packed, A_q8.data(), out.data(), M, shape.N, ISAPath::AVX512);
                },
                WARMUP_ITERS / 2, BENCH_ITERS / 2);

            double t256 = benchmarkMedianUs(
                [&]()
                {
                    std::memset(out.data(), 0, out.size() * sizeof(float));
                    gemm_native_vnni_preq(packed, A_q8.data(), out.data(), M, shape.N, ISAPath::AVX2);
                },
                WARMUP_ITERS / 2, BENCH_ITERS / 2);

            results.push_back({fmt.name, fmt.category, shape.label,
                               shape.N, shape.K, M, t512, t256, t256 / t512});
        }
    }

    // Render table
    fort::utf8_table table;
    table.set_border_style(FT_DOUBLE2_STYLE);
    table << fort::header
          << "Format" << "Category" << "M" << "AVX512 (µs)" << "AVX2 (µs)" << "Ratio"
          << fort::endr;

    table.column(0).set_cell_text_align(fort::text_align::left);
    table.column(1).set_cell_text_align(fort::text_align::left);
    for (int c = 2; c <= 5; ++c)
        table.column(c).set_cell_text_align(fort::text_align::right);

    std::string prev_fmt;
    for (const auto &r : results)
    {
        if (!prev_fmt.empty() && r.format != prev_fmt)
            table << fort::separator;
        prev_fmt = r.format;

        table << r.format << r.category << std::to_string(r.M)
              << fmt(r.avx512_us, 1) << fmt(r.avx2_us, 1)
              << (fmt(r.ratio, 2) + "×")
              << fort::endr;
    }

    double sum_ratio = 0;
    for (const auto &r : results)
        sum_ratio += r.ratio;
    double avg_ratio = sum_ratio / results.size();

    table << fort::separator;
    table << "AVERAGE" << "" << "" << "" << "" << (fmt(avg_ratio, 2) + "×") << fort::endr;

    std::cout << "\n"
              << table.to_string() << std::endl;

    // Per-M averages
    fort::utf8_table m_table;
    m_table.set_border_style(FT_DOUBLE2_STYLE);
    m_table << fort::header << "M" << "Avg Ratio" << "Tests" << fort::endr;
    m_table.column(0).set_cell_text_align(fort::text_align::right);
    m_table.column(1).set_cell_text_align(fort::text_align::right);
    m_table.column(2).set_cell_text_align(fort::text_align::right);

    for (int M : GEMM_M_VALUES)
    {
        double s = 0;
        int cnt = 0;
        for (const auto &r : results)
            if (r.M == M)
            {
                s += r.ratio;
                cnt++;
            }
        if (cnt > 0)
            m_table << std::to_string(M) << (fmt(s / cnt, 2) + "×") << std::to_string(cnt) << fort::endr;
    }

    std::cout << m_table.to_string() << std::endl;
}
