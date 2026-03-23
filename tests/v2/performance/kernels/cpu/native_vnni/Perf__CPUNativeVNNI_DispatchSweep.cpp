/**
 * @file Perf__CPUNativeVNNI_DispatchSweep.cpp
 * @brief Parameter space sweep for CPU NativeVNNI tile dispatch heuristics.
 *
 * Sweeps (n_block_chunks, k_tile_blocks, m_unroll) across representative
 * Qwen2.5 shapes (0.5B, 3B, 7B, 14B, 32B) for GEMV (M=1, decode) and
 * GEMM (M>1, prefill).
 *
 * For each (shape, M, param-combo):
 *   1. Override tile config via LLAMINAR_CPU_VNNI_* env vars + DebugEnv reload
 *   2. Benchmark the NativeVNNI kernel
 *   3. Report latency, GFLOPS, effective bandwidth, roofline %
 *
 * Results are printed as CSV for downstream analysis and as a summary table
 * showing the best config vs the auto-heuristic default for each shape.
 *
 * Quant formats tested: Q8_0 (INT8 path) and Q4_0 (nibble-LUT path).
 *
 * @note Run with Release build: ctest -R V2_Perf_CPUNativeVNNI_DispatchSweep
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <iomanip>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

#include "kernels/cpu/native_vnni/CPUNativeVNNIGemmKernel.h"
#include "kernels/cpu/native_vnni/CPUNativeVNNITileConfig.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"
#include "utils/Logger.h"
#include "fort.hpp"

#include "utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::cpu::native_vnni;
using namespace llaminar2::test;

namespace
{

    // =========================================================================
    // MPI environment (shared with other perf tests)
    // =========================================================================
    void mpi_abort_signal_handler(int sig)
    {
        const char *msg = "\n[FATAL] Signal caught in dispatch sweep — MPI_Abort\n";
        [[maybe_unused]] auto _ = write(STDERR_FILENO, msg, strlen(msg));
        MPI_Abort(MPI_COMM_WORLD, sig);
        _exit(128 + sig);
    }

    class MPIEnvironment : public ::testing::Environment
    {
    public:
        void SetUp() override
        {
            int initialized = 0;
            MPI_Initialized(&initialized);
            if (!initialized)
                MPI_Init(nullptr, nullptr);
            std::signal(SIGSEGV, mpi_abort_signal_handler);
            std::signal(SIGABRT, mpi_abort_signal_handler);
            std::signal(SIGFPE, mpi_abort_signal_handler);
        }
        void TearDown() override
        {
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (!finalized)
                MPI_Finalize();
        }
    };

    static auto *g_mpi_env [[maybe_unused]] =
        ::testing::AddGlobalTestEnvironment(new MPIEnvironment);

    // Measured on 2x Xeon Gold 6238R (56 cores, DDR4-2933, 6ch/socket)
    static constexpr double MEASURED_READ_BW_GBS = 117.0;

    // =========================================================================
    // Shape definitions — Qwen2.5 family (0.5B through 32B)
    // =========================================================================

    struct GEMMShape
    {
        std::string name;
        std::string category; // "Attn" or "FFN"
        int N;
        int K;
    };

    /**
     * @brief Model architecture parameters for deriving GEMM shapes.
     *
     * | Size | d_model | n_heads | n_kv_heads | head_dim | d_ff   |
     * |------|---------|---------|------------|----------|--------|
     * | 0.5B |     896 |      14 |          2 |       64 |  4,864 |
     * | 3B   |   2,048 |      16 |          2 |      128 | 11,008 |
     * | 7B   |   3,584 |      28 |          4 |      128 | 18,944 |
     * | 14B  |   5,120 |      40 |          8 |      128 | 13,824 |
     * | 32B  |   5,120 |      40 |          8 |      128 | 27,648 |
     */
    struct ModelDims
    {
        std::string label;  // e.g. "0.5B", "7B"
        int d_model;
        int n_heads;
        int n_kv_heads;
        int head_dim;
        int d_ff;
    };

    static const std::vector<ModelDims> ALL_MODEL_DIMS = {
        {"0.5B",  896, 14, 2,  64,  4864},
        {"3B",   2048, 16, 2, 128, 11008},
        {"7B",   3584, 28, 4, 128, 18944},
        {"14B",  5120, 40, 8, 128, 13824},
        {"32B",  5120, 40, 8, 128, 27648},
    };

    static std::vector<GEMMShape> shapesForModel(const ModelDims &m)
    {
        int q_n = m.n_heads * m.head_dim;      // Q_proj output dim
        int kv_n = m.n_kv_heads * m.head_dim;  // K/V_proj output dim
        return {
            {m.label + "_Q_proj",    "Attn", q_n,      m.d_model},
            {m.label + "_K_proj",    "Attn", kv_n,     m.d_model},
            {m.label + "_V_proj",    "Attn", kv_n,     m.d_model},
            {m.label + "_Wo_proj",   "Attn", m.d_model, q_n},
            {m.label + "_FFN_Gate",  "FFN",  m.d_ff,    m.d_model},
            {m.label + "_FFN_Up",    "FFN",  m.d_ff,    m.d_model},
            {m.label + "_FFN_Down",  "FFN",  m.d_model, m.d_ff},
        };
    }

    /// Legacy alias — existing 7B-only tests use this.
    static std::vector<GEMMShape> qwen7BShapes()
    {
        // Qwen2.5-7B: d_model=3584, n_heads=28, n_kv_heads=4, head_dim=128, d_ff=18944
        return {
            {"Q_proj", "Attn", 3584, 3584},
            {"K_proj", "Attn", 512, 3584},
            {"V_proj", "Attn", 512, 3584},
            {"Wo_proj", "Attn", 3584, 3584},
            {"FFN_Gate", "FFN", 18944, 3584},
            {"FFN_Up", "FFN", 18944, 3584},
            {"FFN_Down", "FFN", 3584, 18944},
        };
    }

    // =========================================================================
    // Quant format helpers
    // =========================================================================

    struct FormatSpec
    {
        std::string name;
        bool is_nibble_lut;
    };

    static const std::vector<FormatSpec> SWEEP_FORMATS = {
        {"Q8_0", false}, // INT8 pre-decoded path (our current 7B model)
        {"Q4_0", true},  // Nibble-LUT path (common 4-bit format)
    };

    static std::unique_ptr<TensorBase> createWeights(const std::string &fmt, size_t N, size_t K)
    {
        if (fmt == "Q8_0")
            return TestTensorFactory::createQ8_0Random({N, K});
        if (fmt == "Q4_0")
            return TestTensorFactory::createQ4_0Random({N, K});
        return nullptr;
    }

    // =========================================================================
    // Benchmark infrastructure
    // =========================================================================

    static constexpr int WARMUP_ITERS = 30;
    static constexpr int BENCH_ITERS = 100;

    struct SweepResult
    {
        std::string format;
        std::string shape;
        std::string category;
        int M, N, K;
        int n_block_chunks;
        int k_tile_blocks;
        int m_unroll;
        int k_tiles;
        double latency_us;  // p10 latency
        double gflops;
        double bw_gbs;
        double roofline_pct;
        bool is_default; // true = auto-heuristic (no overrides)
    };

    /**
     * @brief Benchmark a single kernel invocation, return p10 latency (us).
     */
    static double benchKernel(ITensorGemm *kernel,
                              const float *A, float *C,
                              int M, int N, int K)
    {
        auto A_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)K});
        std::memcpy(A_tensor->mutable_data(), A, (size_t)M * K * sizeof(float));
        auto C_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)N});

        for (int i = 0; i < WARMUP_ITERS; ++i)
            kernel->multiply_tensor(A_tensor.get(), C_tensor.get(), M, N, K);

        std::vector<double> times(BENCH_ITERS);
        for (int i = 0; i < BENCH_ITERS; ++i)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            kernel->multiply_tensor(A_tensor.get(), C_tensor.get(), M, N, K);
            auto t1 = std::chrono::high_resolution_clock::now();
            times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
        }

        std::memcpy(C, C_tensor->data(), (size_t)M * N * sizeof(float));
        std::sort(times.begin(), times.end());
        int p10_idx = std::max(0, (int)(BENCH_ITERS * 0.1) - 1);
        return times[p10_idx];
    }

    // =========================================================================
    // Env-var helpers for sweep
    // =========================================================================

    static void clearOverrides()
    {
        unsetenv("LLAMINAR_CPU_VNNI_N_BLOCK_CHUNKS");
        unsetenv("LLAMINAR_CPU_VNNI_K_TILE_BLOCKS");
        unsetenv("LLAMINAR_CPU_VNNI_M_UNROLL");
        unsetenv("LLAMINAR_CPU_VNNI_K_TILES");
        unsetenv("LLAMINAR_CPU_VNNI_MIN_BPR_K_PARALLEL");
        mutableDebugEnv().cpu_vnni.reload();
    }

    static void setOverride(const char *name, int value)
    {
        if (value > 0)
            setenv(name, std::to_string(value).c_str(), 1);
        else
            unsetenv(name);
    }

    static void applyOverrides(int n_block_chunks, int k_tile_blocks, int m_unroll, int k_tiles)
    {
        setOverride("LLAMINAR_CPU_VNNI_N_BLOCK_CHUNKS", n_block_chunks);
        setOverride("LLAMINAR_CPU_VNNI_K_TILE_BLOCKS", k_tile_blocks);
        setOverride("LLAMINAR_CPU_VNNI_M_UNROLL", m_unroll);
        setOverride("LLAMINAR_CPU_VNNI_K_TILES", k_tiles);
        mutableDebugEnv().cpu_vnni.reload();
    }

    // =========================================================================
    // Parameter space definitions
    // =========================================================================

    // n_block_chunks: how many 64-col chunks per parallel task
    static const std::vector<int> N_BLOCK_CHUNKS_VALUES = {0, 1, 2, 4, 8, 16};
    // k_tile_blocks: K-blocks per tile (0 = full K, no tiling)
    static const std::vector<int> K_TILE_BLOCKS_VALUES = {0, 16, 32, 64, 128, 256};
    // m_unroll: M-loop unroll factor
    static const std::vector<int> M_UNROLL_VALUES = {0, 1, 2, 4};
    // Batch sizes for prefill
    static const std::vector<int> PREFILL_M_VALUES = {64, 256, 1024, 1788};

    // =========================================================================
    // Core sweep function
    // =========================================================================

    static std::vector<SweepResult> runDispatchSweep(
        const FormatSpec &fmt,
        const GEMMShape &shape,
        int M,
        bool csv_header_printed)
    {
        std::vector<SweepResult> results;

        auto weights = createWeights(fmt.name, shape.N, shape.K);
        EXPECT_NE(weights, nullptr) << "Failed to create " << fmt.name << " weights";
        if (!weights)
            return results;

        // Create the kernel once — weight packing is format-dependent, not config-dependent
        CPUNativeVNNIGemmKernel kernel(weights.get());
        EXPECT_TRUE(kernel.isValid()) << "Kernel invalid for " << fmt.name;
        if (!kernel.isValid())
            return results;

        // Random activation
        std::vector<float> A(static_cast<size_t>(M) * shape.K);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : A)
            v = dist(rng);
        std::vector<float> C(static_cast<size_t>(M) * shape.N, 0.0f);

        // Compute packed weight size for bandwidth calc
        const auto &packed = kernel.packedWeights();
        int N_chunks = (shape.N + 63) / 64;
        int bpr = packed.blocks_per_row;
        double weight_bytes = (double)N_chunks * bpr * packed.interleaved_block_stride +
                              (double)N_chunks * bpr * 64 * (4 + 4); // scales + comp
        if (packed.is_asymmetric)
            weight_bytes += (double)N_chunks * bpr * 64 * 4; // mins
        // For M>1, activation bytes matter too
        double activation_bytes = (double)M * shape.K * sizeof(float);
        double total_bytes = weight_bytes + activation_bytes;

        // Print CSV header
        if (!csv_header_printed)
        {
            std::cout << "format,shape,category,M,N,K,n_block_chunks,k_tile_blocks,"
                      << "m_unroll,k_tiles,latency_us,gflops,bw_gbs,roofline_pct,is_default"
                      << std::endl;
        }

        auto runOneBench = [&](int nbc, int ktb, int mu, int kt, bool is_default) -> SweepResult
        {
            if (is_default)
                clearOverrides();
            else
                applyOverrides(nbc, ktb, mu, kt);

            double lat_us = benchKernel(&kernel, A.data(), C.data(), M, shape.N, shape.K);
            double flops = 2.0 * M * shape.N * shape.K;
            double gflops = flops / (lat_us * 1e-6) / 1e9;
            double bw = total_bytes / (lat_us * 1e-6) / 1e9;
            double roof = bw / MEASURED_READ_BW_GBS * 100.0;

            // Query the actual tile config that was used
            NativeVNNITileConfig cfg = computeTileConfig(
                shape.N, shape.K, M, packed.payload_bytes, omp_get_max_threads());

            SweepResult r;
            r.format = fmt.name;
            r.shape = shape.name;
            r.category = shape.category;
            r.M = M;
            r.N = shape.N;
            r.K = shape.K;
            r.n_block_chunks = cfg.n_block_chunks;
            r.k_tile_blocks = cfg.k_tile_blocks;
            r.m_unroll = cfg.m_unroll;
            r.k_tiles = cfg.k_tiles;
            r.latency_us = lat_us;
            r.gflops = gflops;
            r.bw_gbs = bw;
            r.roofline_pct = roof;
            r.is_default = is_default;

            // CSV output
            std::cout << fmt.name << "," << shape.name << "," << shape.category << ","
                      << M << "," << shape.N << "," << shape.K << ","
                      << r.n_block_chunks << "," << r.k_tile_blocks << ","
                      << r.m_unroll << "," << r.k_tiles << ","
                      << std::fixed << std::setprecision(1) << lat_us << ","
                      << std::setprecision(2) << gflops << ","
                      << bw << "," << roof << ","
                      << (is_default ? "1" : "0") << std::endl;

            results.push_back(r);
            return r;
        };

        // 1. Default (auto-heuristic baseline)
        runOneBench(0, 0, 0, 0, /*is_default=*/true);

        // 2. Sweep n_block_chunks (main parallelism knob)
        for (int nbc : N_BLOCK_CHUNKS_VALUES)
        {
            if (nbc == 0)
                continue; // skip = same as default
            runOneBench(nbc, 0, 0, 0, false);
        }

        // 3. Sweep k_tile_blocks (cache tiling knob)
        int K_blocks = (shape.K + 31) / 32;
        for (int ktb : K_TILE_BLOCKS_VALUES)
        {
            if (ktb == 0)
                continue;
            if (ktb > K_blocks)
                continue; // can't tile larger than K
            runOneBench(0, ktb, 0, 0, false);
        }

        // 4. Sweep m_unroll (M>1 only)
        if (M > 1)
        {
            for (int mu : M_UNROLL_VALUES)
            {
                if (mu == 0)
                    continue;
                runOneBench(0, 0, mu, 0, false);
            }
        }

        // 5. Sweep k_tiles (K-parallel for GEMV)
        if (M == 1)
        {
            for (int kt : {2, 4, 7, 14, 28})
            {
                runOneBench(0, 0, 0, kt, false);
            }
        }

        // Reset overrides
        clearOverrides();
        return results;
    }

    // =========================================================================
    // Summary table renderer
    // =========================================================================

    static void renderSummaryTable(const std::string &title,
                                   const std::vector<SweepResult> &all_results)
    {
        // Group by (format, shape, M) and find best vs default
        struct ShapeKey
        {
            std::string format, shape;
            int M;
            bool operator<(const ShapeKey &o) const
            {
                if (format != o.format)
                    return format < o.format;
                if (shape != o.shape)
                    return shape < o.shape;
                return M < o.M;
            }
        };

        std::map<ShapeKey, std::pair<SweepResult, SweepResult>> best_vs_default;

        for (const auto &r : all_results)
        {
            ShapeKey key{r.format, r.shape, r.M};
            auto &entry = best_vs_default[key];

            if (r.is_default)
            {
                entry.first = r; // default
            }

            if (entry.second.latency_us == 0 || r.latency_us < entry.second.latency_us)
            {
                entry.second = r; // best
            }
        }

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);

        // Title
        std::cout << "\n";
        fort::utf8_table title_tbl;
        title_tbl.set_border_style(FT_DOUBLE2_STYLE);
        title_tbl << title << fort::endr;
        title_tbl[0][0].set_cell_text_align(fort::text_align::center);
        title_tbl.row(0).set_cell_row_type(fort::row_type::header);
        std::cout << title_tbl.to_string();

        table << fort::header
              << "Format" << "Shape" << "M" << "N" << "K"
              << "Default (us)" << "Best (us)" << "Speedup"
              << "Best nbc" << "Best ktb" << "Best mu" << "Best kt"
              << "GFLOPS" << "BW (GB/s)" << "Roof %"
              << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 14; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        for (const auto &[key, pair] : best_vs_default)
        {
            const auto &def = pair.first;
            const auto &best = pair.second;
            double speedup = (def.latency_us > 0) ? def.latency_us / best.latency_us : 0;

            std::ostringstream def_us, best_us, spd, gf, bw, roof;
            def_us << std::fixed << std::setprecision(1) << def.latency_us;
            best_us << std::fixed << std::setprecision(1) << best.latency_us;
            spd << std::fixed << std::setprecision(3) << speedup << "x";
            gf << std::fixed << std::setprecision(1) << best.gflops;
            bw << std::fixed << std::setprecision(1) << best.bw_gbs;
            roof << std::fixed << std::setprecision(1) << best.roofline_pct << "%";

            table << key.format << key.shape << key.M << best.N << best.K
                  << def_us.str() << best_us.str() << spd.str()
                  << best.n_block_chunks << best.k_tile_blocks
                  << best.m_unroll << best.k_tiles
                  << gf.str() << bw.str() << roof.str()
                  << fort::endr;
        }

        std::cout << table.to_string();
    }

    // =========================================================================
    // Test cases
    // =========================================================================

    class CPUNativeVNNIDispatchSweepTest : public ::testing::Test
    {
    protected:
        void SetUp() override {}
        void TearDown() override { clearOverrides(); }
    };

    // -----------------------------------------------------------------------
    // DECODE (M=1): Full sweep for Q8_0 across all Qwen 7B shapes
    // -----------------------------------------------------------------------
    TEST_F(CPUNativeVNNIDispatchSweepTest, Q8_0_Decode_7B)
    {
        auto shapes = qwen7BShapes();
        std::vector<SweepResult> all;
        bool header = false;

        for (const auto &shape : shapes)
        {
            auto results = runDispatchSweep({"Q8_0", false}, shape, 1, header);
            header = true;
            all.insert(all.end(), results.begin(), results.end());
        }

        renderSummaryTable("Q8_0 DECODE (M=1) DISPATCH SWEEP — Qwen 7B", all);
    }

    // -----------------------------------------------------------------------
    // DECODE (M=1): Full sweep for Q4_0 across all Qwen 7B shapes
    // -----------------------------------------------------------------------
    TEST_F(CPUNativeVNNIDispatchSweepTest, Q4_0_Decode_7B)
    {
        auto shapes = qwen7BShapes();
        std::vector<SweepResult> all;
        bool header = false;

        for (const auto &shape : shapes)
        {
            auto results = runDispatchSweep({"Q4_0", true}, shape, 1, header);
            header = true;
            all.insert(all.end(), results.begin(), results.end());
        }

        renderSummaryTable("Q4_0 DECODE (M=1) DISPATCH SWEEP — Qwen 7B", all);
    }

    // -----------------------------------------------------------------------
    // PREFILL: Q8_0 sweep at M=64, 256, 1024, 1788 for key FFN shapes
    // -----------------------------------------------------------------------
    TEST_F(CPUNativeVNNIDispatchSweepTest, Q8_0_Prefill_7B)
    {
        // Focus on the FFN shapes (70%+ of prefill time) + Wo
        std::vector<GEMMShape> prefill_shapes = {
            {"FFN_Gate", "FFN", 18944, 3584},
            {"FFN_Down", "FFN", 3584, 18944},
            {"Wo_proj", "Attn", 3584, 3584},
        };

        std::vector<SweepResult> all;
        bool header = false;

        for (int M : PREFILL_M_VALUES)
        {
            for (const auto &shape : prefill_shapes)
            {
                auto results = runDispatchSweep({"Q8_0", false}, shape, M, header);
                header = true;
                all.insert(all.end(), results.begin(), results.end());
            }
        }

        renderSummaryTable("Q8_0 PREFILL (M>1) DISPATCH SWEEP — Qwen 7B", all);
    }

    // -----------------------------------------------------------------------
    // PREFILL: Q4_0 sweep for key FFN shapes
    // -----------------------------------------------------------------------
    TEST_F(CPUNativeVNNIDispatchSweepTest, Q4_0_Prefill_7B)
    {
        std::vector<GEMMShape> prefill_shapes = {
            {"FFN_Gate", "FFN", 18944, 3584},
            {"FFN_Down", "FFN", 3584, 18944},
        };

        std::vector<SweepResult> all;
        bool header = false;

        for (int M : {256, 1788})
        {
            for (const auto &shape : prefill_shapes)
            {
                auto results = runDispatchSweep({"Q4_0", true}, shape, M, header);
                header = true;
                all.insert(all.end(), results.begin(), results.end());
            }
        }

        renderSummaryTable("Q4_0 PREFILL (M>1) DISPATCH SWEEP — Qwen 7B", all);
    }

    // -----------------------------------------------------------------------
    // DECODE (M=1): Multi-model sweep — Q8_0 across 0.5B, 3B, 7B, 14B, 32B
    //
    // Validates that dispatch heuristics scale across model sizes.
    // -----------------------------------------------------------------------
    TEST_F(CPUNativeVNNIDispatchSweepTest, Q8_0_Decode_MultiModel)
    {
        std::vector<SweepResult> all;
        bool header = false;

        for (const auto &model : ALL_MODEL_DIMS)
        {
            auto shapes = shapesForModel(model);
            for (const auto &shape : shapes)
            {
                auto results = runDispatchSweep({"Q8_0", false}, shape, 1, header);
                header = true;
                all.insert(all.end(), results.begin(), results.end());
            }
        }

        renderSummaryTable("Q8_0 DECODE (M=1) MULTI-MODEL SWEEP — Qwen 0.5B/3B/7B/14B/32B", all);
    }

    // -----------------------------------------------------------------------
    // PREFILL: Multi-model sweep — Q8_0 at M=256,1024 across all sizes
    //
    // Uses representative M values; FFN shapes dominate prefill time.
    // -----------------------------------------------------------------------
    TEST_F(CPUNativeVNNIDispatchSweepTest, Q8_0_Prefill_MultiModel)
    {
        std::vector<SweepResult> all;
        bool header = false;

        for (const auto &model : ALL_MODEL_DIMS)
        {
            // Focus on the 3 shapes that dominate prefill: FFN_Gate, FFN_Down, Wo
            int q_n = model.n_heads * model.head_dim;
            std::vector<GEMMShape> pfill_shapes = {
                {model.label + "_FFN_Gate", "FFN", model.d_ff,    model.d_model},
                {model.label + "_FFN_Down", "FFN", model.d_model, model.d_ff},
                {model.label + "_Wo_proj",  "Attn", model.d_model, q_n},
            };

            for (int M : {256, 1024})
            {
                for (const auto &shape : pfill_shapes)
                {
                    auto results = runDispatchSweep({"Q8_0", false}, shape, M, header);
                    header = true;
                    all.insert(all.end(), results.begin(), results.end());
                }
            }
        }

        renderSummaryTable("Q8_0 PREFILL (M>1) MULTI-MODEL SWEEP — Qwen 0.5B/3B/7B/14B/32B", all);
    }

    // -----------------------------------------------------------------------
    // DECODE: Focused n_block_chunks + k_tiles 2D sweep for FFN_Down
    //
    // FFN_Down (N=3584, K=18944) is the shape most likely to benefit from
    // K-parallel because K is very large (bpr=592) but N is moderate.
    // -----------------------------------------------------------------------
    TEST_F(CPUNativeVNNIDispatchSweepTest, Q8_0_Decode_FFN_Down_2D_Sweep)
    {
        GEMMShape shape = {"FFN_Down", "FFN", 3584, 18944};
        auto weights = createWeights("Q8_0", shape.N, shape.K);
        ASSERT_NE(weights, nullptr);

        CPUNativeVNNIGemmKernel kernel(weights.get());
        ASSERT_TRUE(kernel.isValid());

        const auto &packed = kernel.packedWeights();
        int N_chunks = (shape.N + 63) / 64;
        int bpr = packed.blocks_per_row;
        double weight_bytes = (double)N_chunks * bpr * packed.interleaved_block_stride +
                              (double)N_chunks * bpr * 64 * (4 + 4);
        double total_bytes = weight_bytes + (double)shape.K * sizeof(float);

        std::vector<float> A(shape.K);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : A)
            v = dist(rng);
        std::vector<float> C(shape.N, 0.0f);

        // CSV header
        std::cout << "nbc,k_tiles,latency_us,gflops,bw_gbs,roofline_pct" << std::endl;

        struct Point
        {
            int nbc, kt;
            double lat, gflops, bw, roof;
        };
        std::vector<Point> points;

        for (int nbc : {1, 2, 4, 8})
        {
            for (int kt : {0, 2, 4, 7, 14, 28})
            {
                applyOverrides(nbc, 0, 0, kt);

                double lat = benchKernel(&kernel, A.data(), C.data(), 1, shape.N, shape.K);
                double flops = 2.0 * shape.N * shape.K;
                double gf = flops / (lat * 1e-6) / 1e9;
                double bw = total_bytes / (lat * 1e-6) / 1e9;
                double roof = bw / MEASURED_READ_BW_GBS * 100.0;

                std::cout << nbc << "," << kt << ","
                          << std::fixed << std::setprecision(1) << lat << ","
                          << std::setprecision(2) << gf << ","
                          << bw << "," << roof << std::endl;

                points.push_back({nbc, kt, lat, gf, bw, roof});
            }
        }

        clearOverrides();

        // Summary table
        fort::utf8_table title;
        title.set_border_style(FT_DOUBLE2_STYLE);
        title << "Q8_0 FFN_Down (M=1, N=3584, K=18944) — nbc × k_tiles 2D Sweep" << fort::endr;
        title[0][0].set_cell_text_align(fort::text_align::center);
        title.row(0).set_cell_row_type(fort::row_type::header);
        std::cout << "\n" << title.to_string();

        fort::utf8_table tbl;
        tbl.set_border_style(FT_DOUBLE2_STYLE);
        tbl << fort::header << "nbc" << "k_tiles" << "Latency (us)" << "GFLOPS" << "BW (GB/s)" << "Roof %" << fort::endr;
        for (int c = 0; c <= 5; ++c)
            tbl.column(c).set_cell_text_align(fort::text_align::right);

        for (const auto &p : points)
        {
            std::ostringstream lat, gf, bw, rf;
            lat << std::fixed << std::setprecision(1) << p.lat;
            gf << std::fixed << std::setprecision(2) << p.gflops;
            bw << std::fixed << std::setprecision(1) << p.bw;
            rf << std::fixed << std::setprecision(1) << p.roof << "%";
            tbl << p.nbc << p.kt << lat.str() << gf.str() << bw.str() << rf.str() << fort::endr;
        }

        std::cout << tbl.to_string();

        // Find best
        auto best = std::min_element(points.begin(), points.end(),
                                     [](const Point &a, const Point &b)
                                     { return a.lat < b.lat; });
        std::cout << "\n  BEST: nbc=" << best->nbc << " k_tiles=" << best->kt
                  << " → " << std::fixed << std::setprecision(1) << best->lat << " us, "
                  << std::setprecision(2) << best->gflops << " GFLOPS, "
                  << std::setprecision(1) << best->bw << " GB/s ("
                  << best->roof << "% roofline)\n"
                  << std::endl;
    }

} // anonymous namespace
