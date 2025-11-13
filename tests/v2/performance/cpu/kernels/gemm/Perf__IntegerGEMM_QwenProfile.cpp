/**
 * @file Perf__IntegerGEMM_QwenProfile.cpp
 * @brief Realistic Qwen 0.5B pipeline operation profiling for INT8 GEMM
 *
 * Purpose:
 * - Test representative operations from Qwen 0.5B inference pipeline
 * - Cover both single-token decode and large batch prefill scenarios
 * - Use appropriate kernel configurations per operation size
 * - Present comprehensive performance table across all operations
 *
 * Qwen 0.5B Architecture:
 * - d_model: 896
 * - n_heads: 14
 * - d_head: 64
 * - FFN intermediate: 4864
 * - Layers: 24
 *
 * Operations profiled:
 * 1. Single-token decode (batch=1, seq=1):
 *    - Q/K/V projections: [1, 896] × [896, 896] → [1, 896]
 *    - Attention output:  [1, 896] × [896, 896] → [1, 896]
 *    - FFN gate/up:       [1, 896] × [896, 4864] → [1, 4864]
 *    - FFN down:          [1, 4864] × [4864, 896] → [1, 896]
 *
 * 2. Large batch prefill (batch=128, seq=512):
 *    - Q/K/V projections: [512, 896] × [896, 896] → [512, 896]
 *    - FFN gate/up:       [512, 896] × [896, 4864] → [512, 4864]
 *    - FFN down:          [512, 4864] × [4864, 896] → [512, 896]
 *
 * Configuration strategy:
 * - Small ops (m≤16): MR=4, single-thread friendly
 * - Medium ops (16<m≤128): MR=8, static parallel
 * - Large ops (m>128): MR=16, full parallel
 * - KC tuned to operation size (256-1024)
 *
 * Performance Metrics:
 * - Measures INT8 GEMM throughput (GOPS = Giga Integer Operations Per Second)
 * - 1 operation = 1 INT8 multiply-accumulate (MAC)
 * - Total operations per GEMM: 2*M*N*K (M*N output elements, K MACs each)
 * - Theoretical peak based on AVX512VNNI: 28 cores × 2.2 GHz × 2 FMA × 64 INT8/cycle
 * - Efficiency = (achieved GOPS / theoretical peak) × 100%
 *
 * @author David Sanftenberg
 * @date November 11, 2025
 */

#include "kernels/cpu/gemm_v2/IntegerGemmKernelTemplateV2.h"
#include "kernels/cpu/gemm/GemmWeightCache.h"
#include "kernels/cpu/SimdTraits.h"
#include "tensors/Tensors.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cstring>
#include <algorithm>

using namespace llaminar2;
using namespace llaminar2::kernels::gemm;
using namespace llaminar2::kernels::simd;

// Simple Q8_0 block provider for testing
struct SimpleBlockProvider : public Q8_0BlockProvider
{
    const Q8_0Block *blocks;
    size_t k_blocks_;
    size_t num_rows_;

    SimpleBlockProvider(const Q8_0Block *b, size_t kb, size_t num_rows)
        : blocks(b), k_blocks_(kb), num_rows_(num_rows) {}

    const Q8_0Block *get_q8_block(size_t row_idx, size_t k_block_offset) override
    {
        return &blocks[row_idx * k_blocks_ + k_block_offset];
    }

    void warmup_cache(size_t, size_t, size_t, size_t) override {}
    size_t k_blocks() const override { return k_blocks_; }
    size_t num_rows() const override { return num_rows_; }
    bool is_zero_copy() const override { return true; }
};

// Operation descriptor
struct OpDescriptor
{
    std::string name;
    std::string category; // "decode" or "prefill"
    int m, n, k;
    int MR, NR;
    int UNROLL_K, PREFETCH_DIST;
    int MC, KC, NC;
};

// Benchmark result
struct BenchmarkResult
{
    std::string name;
    std::string category;
    int m, n, k;
    double gflops;
    double time_ms;
    double efficiency_pct;
    int scale_mismatches;
    int code_mismatches;
};

// Helper: Generate random Q8_0 blocks
void generate_random_q8_blocks(Q8_0Block *blocks, size_t num_blocks, std::mt19937 &rng)
{
    std::uniform_int_distribution<int> dist(-127, 127);
    std::uniform_real_distribution<float> scale_dist(0.001f, 0.1f);

    for (size_t i = 0; i < num_blocks; ++i)
    {
        blocks[i].d = fp32_to_fp16(scale_dist(rng));
        for (int j = 0; j < 32; ++j)
        {
            blocks[i].qs[j] = static_cast<int8_t>(dist(rng));
        }
    }
}

// Scalar reference implementation for correctness validation
void scalar_reference_gemm(
    const Q8_0Block *A, const Q8_0Block *B, Q8_0Block *C_ref,
    int m, int n, int k)
{
    const size_t k_blocks = (k + 31) / 32;
    const size_t n_blocks = (n + 31) / 32;

    // Temporary FP32 accumulator for each output element
    std::vector<float> fp_acc(m * n_blocks * 32, 0.0f); // Allocate full n_blocks*32 width

    for (int i = 0; i < m; ++i)
    {
        for (size_t nb = 0; nb < n_blocks; ++nb)
        {
            for (int j_in_block = 0; j_in_block < 32; ++j_in_block)
            {
                const int j = nb * 32 + j_in_block;
                if (j >= n)
                    continue; // Skip padding

                float sum = 0.0f;
                for (size_t kb = 0; kb < k_blocks; ++kb)
                {
                    const Q8_0Block &a_block = A[i * k_blocks + kb];
                    const Q8_0Block &b_block = B[j * k_blocks + kb];

                    float a_scale = fp16_to_fp32(a_block.d);
                    float b_scale = fp16_to_fp32(b_block.d);

                    int32_t dot = 0;
                    for (int kk = 0; kk < 32; ++kk)
                    {
                        dot += static_cast<int32_t>(a_block.qs[kk]) *
                               static_cast<int32_t>(b_block.qs[kk]);
                    }
                    sum += static_cast<float>(dot) * a_scale * b_scale;
                }
                fp_acc[i * n_blocks * 32 + nb * 32 + j_in_block] = sum;
            }
        }
    }

    // Quantize each row to Q8_0 blocks
    for (int i = 0; i < m; ++i)
    {
        for (size_t nb = 0; nb < n_blocks; ++nb)
        {
            const size_t row_offset = i * n_blocks * 32 + nb * 32;

            // Find max absolute value in this block
            float amax = 0.0f;
            for (int jj = 0; jj < 32; ++jj)
            {
                float val = std::fabs(fp_acc[row_offset + jj]);
                if (val > amax)
                    amax = val;
            }

            // Compute scale
            float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
            float inv_scale = 1.0f / scale;
            C_ref[i * n_blocks + nb].d = fp32_to_fp16(scale);

            // Quantize
            for (int jj = 0; jj < 32; ++jj)
            {
                float scaled = fp_acc[row_offset + jj] * inv_scale;
                int32_t q = static_cast<int32_t>(std::round(scaled));
                q = std::max(-127, std::min(127, q));
                C_ref[i * n_blocks + nb].qs[jj] = static_cast<int8_t>(q);
            }
        }
    }
}

// Compare two Q8_0 block arrays
void compare_q8_blocks(
    const Q8_0Block *C_test, const Q8_0Block *C_ref,
    int m, int n,
    int &scale_mismatches, int &code_mismatches)
{
    const size_t n_blocks = (n + 31) / 32;
    scale_mismatches = 0;
    code_mismatches = 0;

    for (int i = 0; i < m; ++i)
    {
        for (size_t nb = 0; nb < n_blocks; ++nb)
        {
            const Q8_0Block &test = C_test[i * n_blocks + nb];
            const Q8_0Block &ref = C_ref[i * n_blocks + nb];

            float test_scale = fp16_to_fp32(test.d);
            float ref_scale = fp16_to_fp32(ref.d);

            if (std::fabs(test_scale - ref_scale) > 1e-5f)
            {
                scale_mismatches++;
            }

            for (int j = 0; j < 32; ++j)
            {
                if (test.qs[j] != ref.qs[j])
                {
                    code_mismatches++;
                }
            }
        }
    }
}

// Benchmark a single operation
template <int MR, int NR, int K_BLOCKS_PER_ITER, int UNROLL_K, int PREFETCH_DIST, int MC, int KC, int NC>
BenchmarkResult benchmark_operation(
    const OpDescriptor &op,
    std::mt19937 &rng,
    bool use_simd,
    int warmup_iters = 3,
    int timed_iters = 10)
{
    using KernelType = IntegerGemmKernelV2<AVX512VNNITag, MR, NR, K_BLOCKS_PER_ITER, UNROLL_K, PREFETCH_DIST, MC, KC, NC>;
    // Note: V2 kernel doesn't have setUseSimd() - always uses best available (VNNI if available)

    const size_t m_blocks = (op.m + 31) / 32;
    const size_t k_blocks = (op.k + 31) / 32;
    const size_t n_blocks = (op.n + 31) / 32;

    // Allocate buffers
    // A is row-major: m rows × k_blocks blocks per row
    std::vector<Q8_0Block> A(op.m * k_blocks);
    std::vector<Q8_0Block> B(op.n * k_blocks); // B is stored row-major by N dimension
    // C is row-major: m rows × n_blocks blocks per row
    std::vector<Q8_0Block> C(op.m * n_blocks);
    std::vector<Q8_0Block> C_ref;

    // Always check correctness (scalar reference works for both scalar and SIMD kernels)
    bool check_correctness = true;
    if (check_correctness)
    {
        C_ref.resize(op.m * n_blocks);
    }

    generate_random_q8_blocks(A.data(), A.size(), rng);
    generate_random_q8_blocks(B.data(), B.size(), rng);

    SimpleBlockProvider B_provider(B.data(), k_blocks, op.n);

    // Warmup
    for (int i = 0; i < warmup_iters; ++i)
    {
        KernelType::multiply(A.data(), B_provider, C.data(), op.m, op.n, op.k);
    }

    // Timed run
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < timed_iters; ++i)
    {
        KernelType::multiply(A.data(), B_provider, C.data(), op.m, op.n, op.k);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double time_per_iter_ms = elapsed_ms / timed_iters;

    // Calculate GOPS (2*M*N*K INT8 operations: multiply + accumulate)
    // GEMM: 2*m*n*k operations (multiply + add per element)
    double ops = 2.0 * op.m * op.n * op.k;
    double gops = (ops / 1e9) / (time_per_iter_ms / 1000.0);

    // Theoretical peak performance (INT8 GEMM, 28 cores)
    // Calculation: 28 cores × 2.2 GHz × 2 FMA units × 64 INT8 ops/cycle (AVX512VNNI)
    //            = 28 × 2.2 × 2 × 64 = 7,884.8 GOPS
    // Note: This assumes base frequency (2.2 GHz). Actual may vary with turbo boost.
    double theoretical_gops = 7884.8; // INT8 operations per second (28 cores)
    double efficiency_pct = (gops / theoretical_gops) * 100.0;

    // Correctness check (only for scalar path)
    int scale_mismatches = 0;
    int code_mismatches = 0;

    if (check_correctness)
    {
        try
        {
            scalar_reference_gemm(A.data(), B.data(), C_ref.data(), op.m, op.n, op.k);
            compare_q8_blocks(C.data(), C_ref.data(), op.m, op.n, scale_mismatches, code_mismatches);
        }
        catch (const std::exception &e)
        {
            // Mark as failed if comparison crashes
            scale_mismatches = -1;
            code_mismatches = -1;
        }
    }

    BenchmarkResult result;
    result.name = op.name;
    result.category = op.category;
    result.m = op.m;
    result.n = op.n;
    result.k = op.k;
    result.gflops = gops; // Note: "gflops" field stores GOPS for backward compat
    result.time_ms = time_per_iter_ms;
    result.efficiency_pct = efficiency_pct;
    result.scale_mismatches = scale_mismatches;
    result.code_mismatches = code_mismatches;

    return result;
}

// Dispatch to appropriate kernel configuration
BenchmarkResult dispatch_benchmark(const OpDescriptor &op, std::mt19937 &rng, bool use_simd, int k_blocks_per_iter = 2)
{
    // Support testing different K-block processing modes
    if (k_blocks_per_iter == 8)
    {
        // 256-byte mode (ultra-wide, experimental)
        return benchmark_operation<16, 32, 8, 4, 2, 256, 512, 128>(op, rng, use_simd);
    }
    else if (k_blocks_per_iter == 4)
    {
        // 128-byte mode (experimental)
        return benchmark_operation<16, 32, 4, 4, 2, 256, 512, 128>(op, rng, use_simd);
    }
    else if (k_blocks_per_iter == 1)
    {
        // 32-byte mode (baseline)
        return benchmark_operation<16, 32, 1, 4, 2, 256, 512, 128>(op, rng, use_simd);
    }
    else
    {
        // Default: 64-byte mode (K_BLOCKS_PER_ITER=2, optimal)
        if (op.UNROLL_K == 8 && op.KC == 1024)
        {
            return benchmark_operation<16, 32, 2, 8, 3, 512, 1024, 256>(op, rng, use_simd);
        }
        else
        {
            return benchmark_operation<16, 32, 2, 4, 2, 256, 512, 128>(op, rng, use_simd);
        }
    }
}

int main(int argc, char **argv)
{
    bool use_simd = false;
    int k_blocks_per_iter = 2; // Default: 64-byte mode

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--simd")
        {
            use_simd = true;
        }
        else if (std::string(argv[i]).find("--k-blocks=") == 0)
        {
            k_blocks_per_iter = std::stoi(std::string(argv[i]).substr(11));
            if (k_blocks_per_iter != 1 && k_blocks_per_iter != 2 && k_blocks_per_iter != 4 && k_blocks_per_iter != 8)
            {
                std::cerr << "Error: --k-blocks must be 1, 2, 4, or 8\n";
                return 1;
            }
        }
    }

    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::vector<OpDescriptor> operations;

    // ========================================================================
    // SINGLE-TOKEN DECODE OPERATIONS (batch=1, seq=1)
    // ========================================================================

    // Q projection: [1, 896] × [896, 896] → [1, 896]
    operations.push_back({
        "Q_proj (decode)", "decode",
        1, 896, 896,
        16, 32,       // MR=16 (use single config)
        4, 2,         // UNROLL_K=4, PREFETCH_DIST=2
        256, 512, 128 // MC, KC, NC
    });

    // FFN gate: [1, 896] × [896, 4864] → [1, 4864]
    operations.push_back({"FFN_gate (decode)", "decode",
                          1, 4864, 896,
                          16, 32,
                          4, 2,
                          256, 512, 128});

    // FFN down: [1, 4864] × [4864, 896] → [1, 896]
    operations.push_back({"FFN_down (decode)", "decode",
                          1, 896, 4864,
                          16, 32,
                          4, 2,
                          256, 512, 128});

    // ========================================================================
    // SMALL BATCH PREFILL (batch=32, seq=32)
    // ========================================================================

    operations.push_back({"Q_proj (prefill-32)", "prefill",
                          32, 896, 896,
                          16, 32,
                          4, 2,
                          256, 512, 128});

    operations.push_back({"FFN_gate (prefill-32)", "prefill",
                          32, 4864, 896,
                          16, 32,
                          4, 2,
                          256, 512, 128});

    operations.push_back({"FFN_down (prefill-32)", "prefill",
                          32, 896, 4864,
                          16, 32,
                          4, 2,
                          256, 512, 128});

    // ========================================================================
    // MEDIUM BATCH PREFILL (batch=128)
    // ========================================================================

    operations.push_back({"Q_proj (prefill-128)", "prefill",
                          128, 896, 896,
                          16, 32,
                          4, 2,
                          256, 512, 128});

    operations.push_back({"FFN_gate (prefill-128)", "prefill",
                          128, 4864, 896,
                          16, 32,
                          4, 2,
                          256, 512, 128});

    operations.push_back({"FFN_down (prefill-128)", "prefill",
                          128, 896, 4864,
                          16, 32,
                          4, 2,
                          256, 512, 128});

    // ========================================================================
    // LARGE BATCH PREFILL (batch=512)
    // ========================================================================

    operations.push_back({"Q_proj (prefill-512)", "prefill",
                          512, 896, 896,
                          16, 32,
                          8, 3,
                          512, 1024, 256});

    operations.push_back({"FFN_gate (prefill-512)", "prefill",
                          512, 4864, 896,
                          16, 32,
                          8, 3,
                          512, 1024, 256});

    operations.push_back({"FFN_down (prefill-512)", "prefill",
                          512, 896, 4864,
                          16, 32,
                          8, 3,
                          512, 1024, 256});

    // ========================================================================
    // RUN BENCHMARKS
    // ========================================================================

    std::cout << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ QWEN 0.5B INT8 GEMM PIPELINE PROFILING                                     ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Architecture: d_model=896, n_heads=14, d_head=64, FFN=4864                ║\n";
    std::cout << "║ SIMD mode:    " << (use_simd ? "ENABLED (AVX512 VNNI)" : "DISABLED (scalar reference)")
              << std::string(42 - (use_simd ? 22 : 28), ' ') << "║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════╝\n\n";

    std::vector<BenchmarkResult> results;

    for (const auto &op : operations)
    {
        std::cout << "Running: " << std::left << std::setw(25) << op.name
                  << " [" << std::setw(4) << op.m << " × "
                  << std::setw(5) << op.n << " × "
                  << std::setw(5) << op.k << "]... " << std::flush;

        auto result = dispatch_benchmark(op, rng, use_simd, k_blocks_per_iter);
        results.push_back(result);

        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(8) << result.gflops << " GOPS ";

        if (result.scale_mismatches == 0 && result.code_mismatches == 0)
        {
            std::cout << "✓\n";
        }
        else
        {
            std::cout << "✗ (S:" << result.scale_mismatches
                      << " C:" << result.code_mismatches << ")\n";
        }
    }

    // ========================================================================
    // SUMMARY TABLE
    // ========================================================================

    std::cout << "\n╔════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ PERFORMANCE SUMMARY TABLE                                                                              ║\n";
    std::cout << "╠════════════════════╦═══════════════╦═══════════╦═══════════╦══════════╦═════════════╦═════════════════╣\n";
    std::cout << "║ Operation          ║ Dimensions    ║   GOPS    ║  Time(ms) ║   Eff%   ║ Correctness ║ Config (MR/K)   ║\n";
    std::cout << "╠════════════════════╬═══════════════╬═══════════╬═══════════╬══════════╬═════════════╬═════════════════╣\n";

    // Group by category
    std::vector<std::string> categories = {"decode", "prefill"};

    for (const auto &cat : categories)
    {
        bool first_in_category = true;

        for (const auto &r : results)
        {
            if (r.category != cat)
                continue;

            if (first_in_category)
            {
                std::cout << "║ " << std::left << std::setw(99)
                          << (cat == "decode" ? "SINGLE-TOKEN DECODE" : "BATCH PREFILL")
                          << "║\n";
                std::cout << "╠════════════════════╬═══════════════╬═══════════╬═══════════╬══════════╬═════════════╬═════════════════╣\n";
                first_in_category = false;
            }

            // Format dimension string
            std::ostringstream dim_str;
            dim_str << r.m << "×" << r.n << "×" << r.k;

            // Correctness status
            std::string corr_str;
            if (r.scale_mismatches == 0 && r.code_mismatches == 0)
            {
                corr_str = "✓ PASS";
            }
            else
            {
                std::ostringstream oss;
                oss << "✗ S:" << r.scale_mismatches << " C:" << r.code_mismatches;
                corr_str = oss.str();
            }

            // Extract MR and UNROLL_K from operations
            const auto *op_ptr = &operations[&r - &results[0]];
            std::ostringstream cfg_str;
            cfg_str << op_ptr->MR << "/" << op_ptr->UNROLL_K;

            std::cout << "║ " << std::left << std::setw(18) << r.name << " ║ "
                      << std::setw(13) << dim_str.str() << " ║ "
                      << std::right << std::setw(9) << std::fixed << std::setprecision(2) << r.gflops << " ║ "
                      << std::setw(9) << std::fixed << std::setprecision(3) << r.time_ms << " ║ "
                      << std::setw(7) << std::fixed << std::setprecision(1) << r.efficiency_pct << "% ║ "
                      << std::left << std::setw(11) << corr_str << " ║ "
                      << std::setw(15) << cfg_str.str() << " ║\n";
        }

        if (!first_in_category)
        {
            std::cout << "╠════════════════════╬═══════════════╬═══════════╬═══════════╬══════════╬═════════════╬═════════════════╣\n";
        }
    }

    // Overall statistics
    double total_gflops = 0.0;
    int total_pass = 0;
    int total_fail = 0;

    for (const auto &r : results)
    {
        total_gflops += r.gflops;
        if (r.scale_mismatches == 0 && r.code_mismatches == 0)
        {
            total_pass++;
        }
        else
        {
            total_fail++;
        }
    }

    double avg_gops = total_gflops / results.size();

    std::cout << "║ AVERAGE            ║               ║ "
              << std::right << std::setw(9) << std::fixed << std::setprecision(2) << avg_gops << " ║ "
              << std::setw(9) << " " << " ║ "
              << std::setw(7) << " " << "  ║ "
              << std::left << std::setw(11) << (total_pass > 0 ? std::to_string(total_pass) + " pass" : "") << " ║ "
              << std::setw(15) << (total_fail > 0 ? std::to_string(total_fail) + " fail" : "") << " ║\n";

    std::cout << "╚════════════════════╩═══════════════╩═══════════╩═══════════╩══════════╩═════════════╩═════════════════╝\n";

    // Final verdict
    if (total_fail == 0)
    {
        std::cout << "\n✓ ALL TESTS PASSED - Correctness validated across all operations\n";
        return 0;
    }
    else
    {
        std::cout << "\n✗ " << total_fail << " TESTS FAILED - Review correctness issues above\n";
        return 1;
    }
}
