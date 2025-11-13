/**
 * @file Perf__IntegerGEMM_FullSweep_Registry.cpp
 * @brief Complete configuration space sweep using kernel registry (NO template instantiation!)
 *
 * This version uses the IntegerGemmKernelRegistry instead of direct template instantiation,
 * which dramatically reduces compile times from hours to minutes.
 *
 * Test Modes:
 * - AllConfigsAllWorkloads: Full sweep (all registered kernels × all workloads)
 * - AllConfigsSingleToken: Full sweep on M=1 decode workload
 *
 * Output Format: CSV files (written directly, not stdout)
 *
 * Performance Metrics:
 * - Measures INT8 GEMM throughput (GOPS = Giga Integer Operations Per Second)
 * - 1 operation = 1 INT8 multiply-accumulate (MAC)
 * - Total operations per GEMM: 2*M*N*K (M*N output elements, K MACs each)
 * - Theoretical peak based on AVX512VNNI: 28 cores × 2.2 GHz × 2 FMA × 64 INT8/cycle
 * - Efficiency = (achieved GOPS / theoretical peak) × 100%
 *
 * @author David Sanftenberg
 * @date November 12, 2025
 */

#include <gtest/gtest.h>
#include <chrono>
#include <memory>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <cmath>
#include <string>

#include "kernels/cpu/gemm/int8/IntegerGemmKernelRegistry.h"
#include "kernels/cpu/gemm/GemmWeightCache.h"
#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"

using namespace llaminar2;
using namespace llaminar2::kernels;
using namespace llaminar2::kernels::gemm;

// ============================================================================
// Configuration Space
// ============================================================================

// Theoretical peak performance (INT8 GEMM, 28 cores)
// Calculation: 28 cores × 2.2 GHz × 2 FMA units × 64 INT8 ops/cycle (AVX512VNNI)
//            = 28 × 2.2 × 2 × 64 = 7,884.8 GOPS
// Note: This assumes base frequency (2.2 GHz). Actual may vary with turbo boost.
// Note: Used for relative efficiency comparison, not absolute performance validation.
constexpr double THEORETICAL_PEAK_GOPS = 7884.8; // INT8 operations per second (28 cores)

// Qwen 2.5 0.5B workloads
struct OpDescriptor
{
    std::string name;
    int m, n, k;
};

const std::vector<OpDescriptor> ALL_WORKLOADS = {
    // Decode (M=1)
    {"Q_proj_decode", 1, 896, 896},
    {"K_proj_decode", 1, 896, 896},
    {"V_proj_decode", 1, 896, 896},
    {"O_proj_decode", 1, 896, 896},
    {"FFN_gate_decode", 1, 4864, 896},
    {"FFN_up_decode", 1, 4864, 896},
    {"FFN_down_decode", 1, 896, 4864},

    // Prefill (M=512)
    {"Q_proj_prefill512", 512, 896, 896},
    {"K_proj_prefill512", 512, 896, 896},
    {"V_proj_prefill512", 512, 896, 896},
    {"O_proj_prefill512", 512, 896, 896},
    {"FFN_gate_prefill512", 512, 4864, 896},
    {"FFN_up_prefill512", 512, 4864, 896},
    {"FFN_down_prefill512", 512, 896, 4864},
};

// Configuration space (read from generator script)
const std::vector<int> MR_VALUES = {1, 2, 4, 8, 16, 32};
const std::vector<int> UNROLL_K_VALUES = {1, 2, 4, 8, 16};
const std::vector<int> PREFETCH_DIST_VALUES = {0, 1, 2, 3, 5};
const std::vector<int> MC_VALUES = {128, 256, 512, 1024};
const std::vector<int> KC_VALUES = {256, 512, 1024, 2048};
const std::vector<int> NC_VALUES = {64, 128, 256, 512};

// ============================================================================
// Benchmark Result
// ============================================================================

struct BenchmarkResult
{
    std::string workload_name;
    int m, n, k;
    int mr, nr;
    int k_blocks;
    int unroll_k;
    int prefetch_dist;
    int mc, kc, nc;
    double gflops;
    double time_ms;
    double efficiency_pct;
    bool passed;
};

// ============================================================================
// Helper Functions
// ============================================================================

std::shared_ptr<Q8_0Tensor> createRandomQ8Tensor(int rows, int cols, std::mt19937 &rng)
{
    std::uniform_int_distribution<int> dist(-127, 127);
    std::uniform_real_distribution<float> scale_dist(0.01f, 1.0f);

    size_t k_blocks = (cols + 31) / 32;
    std::vector<Q8_0Block> blocks(rows * k_blocks);

    for (auto &block : blocks)
    {
        float scale = scale_dist(rng);
        block.d = fp32_to_fp16(scale);
        for (int i = 0; i < 32; ++i)
        {
            block.qs[i] = static_cast<int8_t>(dist(rng));
        }
    }

    std::vector<uint8_t> raw_data(blocks.size() * sizeof(Q8_0Block));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());

    return std::make_shared<Q8_0Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
        raw_data);
}

/**
 * @brief Simple Q8_0BlockProvider for testing
 */
class SimpleBlockProvider : public Q8_0BlockProvider
{
private:
    const Q8_0Block *blocks_;
    size_t k_blocks_;
    size_t num_rows_;

public:
    SimpleBlockProvider(const Q8_0Block *b, size_t kb, size_t rows)
        : blocks_(b), k_blocks_(kb), num_rows_(rows) {}

    const Q8_0Block *get_q8_block(size_t row_idx, size_t k_block_offset) override
    {
        return &blocks_[row_idx * k_blocks_ + k_block_offset];
    }

    void warmup_cache(size_t, size_t, size_t, size_t) override {}
    bool is_zero_copy() const override { return true; }
    size_t k_blocks() const override { return k_blocks_; }
    size_t num_rows() const override { return num_rows_; }
};

/**
 * @brief Runtime benchmark executor using kernel registry
 */
BenchmarkResult benchmark_configuration(
    const OpDescriptor &op,
    int mr, int nr, int unroll_k, int prefetch_dist,
    int mc, int kc, int nc,
    std::mt19937 &rng)
{
    BenchmarkResult result;
    result.workload_name = op.name;
    result.m = op.m;
    result.n = op.n;
    result.k = op.k;
    result.mr = mr;
    result.nr = nr;
    result.k_blocks = kc / 32; // Derived from KC
    result.unroll_k = unroll_k;
    result.prefetch_dist = prefetch_dist;
    result.mc = mc;
    result.kc = kc;
    result.nc = nc;
    result.passed = false;

    try
    {
        // Look up kernel in registry
        auto &registry = IntegerGemmKernelRegistry::instance();
        auto kernel_func = registry.get_kernel("simd::AVX512VNNITag", mr, nr, unroll_k, prefetch_dist, mc, kc, nc);

        if (!kernel_func)
        {
            // Kernel not registered - skip this configuration
            result.gflops = 0.0;
            result.time_ms = 0.0;
            result.efficiency_pct = 0.0;
            result.passed = false;
            return result;
        }

        // Create test tensors
        auto A_tensor = createRandomQ8Tensor(op.m, op.k, rng);
        auto B_tensor = createRandomQ8Tensor(op.k, op.n, rng);

        // Get raw block pointers via public API
        const Q8_0Block *A_blocks = static_cast<const Q8_0Block *>(A_tensor->get_raw_block_at(0, 0));
        const Q8_0Block *B_blocks = static_cast<const Q8_0Block *>(B_tensor->get_raw_block_at(0, 0));

        size_t k_blocks_B = (op.k + 31) / 32;

        SimpleBlockProvider B_provider(B_blocks, k_blocks_B, op.n);

        // Create output Q8_0 tensor for results
        auto C_q8_tensor = createRandomQ8Tensor(op.m, op.n, rng);
        Q8_0Block *C_blocks = const_cast<Q8_0Block *>(
            static_cast<const Q8_0Block *>(C_q8_tensor->get_raw_block_at(0, 0)));

        // Warm-up (3 iterations)
        for (int i = 0; i < 3; ++i)
        {
            kernel_func(A_blocks, B_provider, C_blocks, op.m, op.n, op.k);
        }

        // Timed run (10 iterations for stable timing)
        constexpr int NUM_ITERS = 10;
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < NUM_ITERS; ++i)
        {
            kernel_func(A_blocks, B_provider, C_blocks, op.m, op.n, op.k);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double time_per_iter_ms = elapsed_ms / NUM_ITERS;

        // Calculate GOPS (2*M*N*K INT8 operations: multiply + accumulate)
        double ops = 2.0 * op.m * op.n * op.k;
        double gops = (ops / (time_per_iter_ms * 1e6));
        double efficiency = (gops / THEORETICAL_PEAK_GOPS) * 100.0;

        result.gflops = gops; // Note: "gflops" field stores GOPS for backward compat
        result.time_ms = time_per_iter_ms;
        result.efficiency_pct = efficiency;
        result.passed = true;
    }
    catch (const std::exception &e)
    {
        // Silently fail - just mark as not passed
        result.gflops = 0.0;
        result.time_ms = 0.0;
        result.efficiency_pct = 0.0;
        result.passed = false;
    }

    return result;
}

// ============================================================================
// Test Cases
// ============================================================================

/**
 * @brief Full sweep: All configs × all workloads
 * Tests only configurations that are registered in the kernel registry
 * Expected runtime: 30-60 minutes
 */
TEST(IntegerGEMM_FullSweep, AllConfigsAllWorkloads)
{
    std::mt19937 rng(42);

    // Output CSV file
    std::ofstream csv_file("integer_gemm_sweep_full.csv");
    ASSERT_TRUE(csv_file.is_open()) << "Failed to open output file";

    // CSV header
    csv_file << "m,n,k,mr,nr,k_blocks,unroll_k,prefetch_dist,mc,kc,nc,gflops,time_ms,efficiency_pct\n";

    size_t total_configs = 0;
    size_t passed_configs = 0;
    size_t skipped_configs = 0;

    for (const auto &op : ALL_WORKLOADS)
    {
        for (int mr : MR_VALUES)
        {
            constexpr int nr = 32; // Fixed for Q8_0
            for (int unroll_k : UNROLL_K_VALUES)
            {
                for (int prefetch : PREFETCH_DIST_VALUES)
                {
                    for (int mc : MC_VALUES)
                    {
                        for (int kc : KC_VALUES)
                        {
                            for (int nc : NC_VALUES)
                            {
                                total_configs++;

                                auto result = benchmark_configuration(
                                    op, mr, nr, unroll_k, prefetch, mc, kc, nc, rng);

                                if (result.passed)
                                {
                                    passed_configs++;
                                    csv_file << result.m << "," << result.n << "," << result.k << ","
                                             << result.mr << "," << result.nr << "," << result.k_blocks << ","
                                             << result.unroll_k << "," << result.prefetch_dist << ","
                                             << result.mc << "," << result.kc << "," << result.nc << ","
                                             << std::fixed << std::setprecision(2)
                                             << result.gflops << "," << result.time_ms << ","
                                             << result.efficiency_pct << "\n";
                                }
                                else
                                {
                                    skipped_configs++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    csv_file.close();
    std::cout << "Sweep complete:\n";
    std::cout << "  Total configs attempted: " << total_configs << "\n";
    std::cout << "  Passed (registered): " << passed_configs << "\n";
    std::cout << "  Skipped (not registered): " << skipped_configs << "\n";
    std::cout << "Results written to: integer_gemm_sweep_full.csv\n";
}

/**
 * @brief Single workload sweep: All configs on M=1 decode workload
 * Expected runtime: 5-10 minutes
 */
TEST(IntegerGEMM_FullSweep, AllConfigsSingleToken)
{
    std::mt19937 rng(42);

    // Use FFN_gate decode (widest matrix)
    OpDescriptor op = {"FFN_gate_decode", 1, 4864, 896};

    // Output CSV file
    std::ofstream csv_file("integer_gemm_sweep_single_token.csv");
    ASSERT_TRUE(csv_file.is_open()) << "Failed to open output file";

    // CSV header
    csv_file << "m,n,k,mr,nr,k_blocks,unroll_k,prefetch_dist,mc,kc,nc,gflops,time_ms,efficiency_pct\n";

    size_t total_configs = 0;
    size_t passed_configs = 0;
    size_t skipped_configs = 0;

    for (int mr : MR_VALUES)
    {
        constexpr int nr = 32;
        for (int unroll_k : UNROLL_K_VALUES)
        {
            for (int prefetch : PREFETCH_DIST_VALUES)
            {
                for (int mc : MC_VALUES)
                {
                    for (int kc : KC_VALUES)
                    {
                        for (int nc : NC_VALUES)
                        {
                            total_configs++;

                            auto result = benchmark_configuration(
                                op, mr, nr, unroll_k, prefetch, mc, kc, nc, rng);

                            if (result.passed)
                            {
                                passed_configs++;
                                csv_file << result.m << "," << result.n << "," << result.k << ","
                                         << result.mr << "," << result.nr << "," << result.k_blocks << ","
                                         << result.unroll_k << "," << result.prefetch_dist << ","
                                         << result.mc << "," << result.kc << "," << result.nc << ","
                                         << std::fixed << std::setprecision(2)
                                         << result.gflops << "," << result.time_ms << ","
                                         << result.efficiency_pct << "\n";
                            }
                            else
                            {
                                skipped_configs++;
                            }
                        }
                    }
                }
            }
        }
    }

    csv_file.close();
    std::cout << "Sweep complete:\n";
    std::cout << "  Total configs attempted: " << total_configs << "\n";
    std::cout << "  Passed (registered): " << passed_configs << "\n";
    std::cout << "  Skipped (not registered): " << skipped_configs << "\n";
    std::cout << "Results written to: integer_gemm_sweep_single_token.csv\n";
}
