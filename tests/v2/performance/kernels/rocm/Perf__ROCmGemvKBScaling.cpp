/**
 * @file Perf__ROCmGemvKBScaling.cpp
 * @brief KB (K-parallel split) sweep for INT8 VNNI GEMV decode at TP-sliced N sizes
 *
 * Measures decode GEMV (M=1) latency across forced KB values and TP-derived N sizes.
 * The `select_blockwise_qwo_outer_splits` heuristic selects KB to fill CUs, but at
 * TP=2/4 the halved N doubles KB, increasing atomicAdd contention.
 *
 * **Goal**: Find optimal KB for each (N, K) shape to tune the heuristic.
 *
 * **Tested Configurations**:
 * - Model: Qwen2.5-7B (hidden=3584, intermediate=18944, vocab=152064)
 * - Projection types: Q/Wo, K/V, FFN Gate/Up, FFN Down, LM Head
 * - TP degrees: 1, 2, 4 (each halves N)
 * - KB sweep: auto, 1, 2, 4, 7, 8, 14, 16, 28, 56
 *
 * **Metrics**:
 * - Latency (μs): median of timed runs
 * - Best KB vs auto KB: % difference
 * - Effective grid: grid_n × KB total blocks
 *
 * @date June 2026
 */

#include <gtest/gtest.h>

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <string>
#include <cstdio>
#include <cstdint>

#include "fort.hpp"

// Extern C wrappers from ROCmGemvKernel_INT8_VNNI.hip
extern "C"
{
    bool rocmQuantGemm_quantizeActivationsBlockwise(
        const float *d_A_fp32,
        int8_t *d_A_int8,
        float *d_scales_A_blockwise,
        int M, int K,
        int rocm_device_id, void *stream,
        int block_size);

    bool rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
        const int8_t *d_A_int8,
        const int8_t *d_B_int8_vnni,
        float *d_C_fp32,
        const float *d_scales_A_blockwise,
        const float *d_scale_B,
        int N, int K,
        float alpha,
        float beta,
        const float *d_C_existing,
        const float *d_bias,
        int device_id, void *stream);

    void rocmGemv_int8_vnni_set_tuning_overrides(int tn, int kb);
    void rocmGemv_int8_vnni_reset_tuning_overrides();
}

namespace
{

    // ============================================================================
    // Constants
    // ============================================================================

    constexpr int WARMUP_ITERS = 20;
    constexpr int BENCH_ITERS = 100;
    constexpr int ACT_BLOCK_K = 32;
    constexpr int TILE_N = 128; // GEMV_INT8_VNNI_GRID_KPAR_TILE_N
    constexpr int WK = 8;       // Default work-per-thread K-groups

    // ============================================================================
    // Model and projection definitions
    // ============================================================================

    struct ModelDims
    {
        const char *name;
        int hidden;
        int intermediate;
        int num_heads;
        int num_kv_heads;
        int head_dim;
        int vocab;
    };

    static constexpr ModelDims kQwen05B = {
        "Qwen2.5-0.5B", 896, 4864, 14, 2, 64, 151936};

    static constexpr ModelDims kQwen3B = {
        "Qwen2.5-3B", 2048, 11008, 16, 2, 128, 151936};

    static constexpr ModelDims kQwen7B = {
        "Qwen2.5-7B", 3584, 18944, 28, 4, 128, 152064};

    static constexpr ModelDims kQwen14B = {
        "Qwen2.5-14B", 5120, 13824, 40, 8, 128, 152064};

    static constexpr ModelDims kQwen32B = {
        "Qwen2.5-32B", 5120, 27648, 40, 8, 128, 152064};

    struct GemvShape
    {
        const char *name;
        int N;
        int K;
    };

    /// Get per-projection decode shapes for a model at a given TP degree
    static std::vector<GemvShape> getTPShapes(const ModelDims &m, int tp)
    {
        const int H = m.hidden;
        const int I = m.intermediate;
        const int kv_dim = m.num_kv_heads * m.head_dim;

        // At TP>1, N (output dim) is sharded. K (input dim) stays the same for
        // column-parallel projections. For row-parallel (Wo, FFN Down), K is sharded.
        // But for GEMV benchmarking purposes, what matters is the actual (N,K) seen
        // by each device.

        // Column-parallel: N/tp, K unchanged
        // Row-parallel: N unchanged, K/tp
        return {
            {"Q proj", H / tp, H},                      // column-parallel
            {"KV proj", std::max(kv_dim / tp, 128), H}, // column-parallel (floor at 128 for dispatch)
            {"Wo proj", H, H / tp},                     // row-parallel
            {"FFN Gate", I / tp, H},                    // column-parallel
            {"FFN Up", I / tp, H},                      // column-parallel
            {"FFN Down", H, I / tp},                    // row-parallel (input-parallel)
        };
    }

    // ============================================================================
    // Benchmark result
    // ============================================================================

    struct KBResult
    {
        int kb_forced; // 0 = auto (heuristic)
        double median_us;
        double min_us;
        double max_us;
        bool success;
    };

    // ============================================================================
    // VNNI weight packing (identical to Perf__ROCmGemvKernel.cpp)
    // ============================================================================

    static void packVnniWeights(
        const std::vector<int8_t> &B_int8, // [K × N] row-major
        int N, int K,
        std::vector<int8_t> &out_vnni)
    {
        out_vnni.clear();
        if ((K % 4) != 0)
            return;

        const size_t k_groups = static_cast<size_t>(K) / 4;
        out_vnni.resize(k_groups * static_cast<size_t>(N) * 4);
        for (int n = 0; n < N; ++n)
        {
            for (size_t kg = 0; kg < k_groups; ++kg)
            {
                const size_t src = (kg * 4) * static_cast<size_t>(N) + static_cast<size_t>(n);
                const size_t dst = (kg * static_cast<size_t>(N) + static_cast<size_t>(n)) * 4;
                out_vnni[dst + 0] = B_int8[src + static_cast<size_t>(0) * N];
                out_vnni[dst + 1] = B_int8[src + static_cast<size_t>(1) * N];
                out_vnni[dst + 2] = B_int8[src + static_cast<size_t>(2) * N];
                out_vnni[dst + 3] = B_int8[src + static_cast<size_t>(3) * N];
            }
        }
    }

    // ============================================================================
    // Test fixture
    // ============================================================================

    class ROCmGemvKBScalingPerf : public ::testing::Test
    {
    protected:
        int device_id_ = 0;
        int num_cus_ = 0;
        std::string device_name_;
        bool has_device_ = false;

        void SetUp() override
        {
#ifdef HAVE_ROCM
            int count = 0;
            hipError_t err = hipGetDeviceCount(&count);
            has_device_ = (err == hipSuccess && count > 0);
            if (has_device_)
            {
                (void)hipSetDevice(device_id_);
                hipDeviceProp_t props;
                (void)hipGetDeviceProperties(&props, device_id_);
                device_name_ = std::string(props.name) + " (" + props.gcnArchName + ")";
                num_cus_ = props.multiProcessorCount;
            }
#endif
        }

        // =========================================================================
        // Compute heuristic KB (mirrors select_blockwise_qwo_outer_splits)
        // =========================================================================
        int computeHeuristicKB(int N, int K) const
        {
            constexpr int NUM_CUS = 60;
            constexpr int SMALL_K_THRESHOLD = 16;
            constexpr int MIN_WAVES_PER_CU = 8;

            const int grid_n = (N + TILE_N - 1) / TILE_N;
            const int act_blocks = K / ACT_BLOCK_K;
            const int acts_per_wave = act_blocks / WK;

            int ideal_kb;
            if (acts_per_wave <= SMALL_K_THRESHOLD)
            {
                if (grid_n >= NUM_CUS * 2)
                    ideal_kb = 1;
                else
                    ideal_kb = (act_blocks + WK - 1) / WK; // ceiling division
            }
            else
            {
                const float blocks_per_cu_at_kb1 = static_cast<float>(grid_n) / NUM_CUS;
                if (blocks_per_cu_at_kb1 >= 3.0f)
                {
                    ideal_kb = std::max(1, acts_per_wave / 5);
                }
                else
                {
                    int conservative_kb = std::max(1, acts_per_wave / SMALL_K_THRESHOLD);
                    if (acts_per_wave > 64)
                        conservative_kb = std::min(conservative_kb, 4);
                    int conservative_blocks = grid_n * conservative_kb;
                    if (conservative_blocks >= NUM_CUS * 3 / 2)
                        ideal_kb = conservative_kb;
                    else
                        ideal_kb = (act_blocks * 2) / (WK * 5);
                }
            }

            // Occupancy floor only for small-K shapes
            if (acts_per_wave <= SMALL_K_THRESHOLD)
            {
                const int min_total_waves = MIN_WAVES_PER_CU * NUM_CUS;
                const int min_kb = (min_total_waves + grid_n * WK - 1) / (grid_n * WK);
                ideal_kb = std::max(ideal_kb, min_kb);
            }

            const int kb_max = std::max(1, (act_blocks + WK - 1) / WK);
            ideal_kb = std::min(ideal_kb, kb_max);

            return std::max(1, std::min(ideal_kb, act_blocks));
        }

        // =========================================================================
        // Benchmark one (N, K) at a specific forced KB
        // =========================================================================
        KBResult benchmarkGemvKB(int N, int K, int kb_forced)
        {
            KBResult result{};
            result.kb_forced = kb_forced;
#ifndef HAVE_ROCM
            return result;
#else
            // Generate test data
            std::mt19937 rng(42);
            std::uniform_real_distribution<float> dist_a(-1.0f, 1.0f);
            std::uniform_int_distribution<int> dist_b(-127, 127);
            std::uniform_real_distribution<float> dist_s(0.001f, 0.1f);

            std::vector<float> h_A(K);
            std::vector<int8_t> h_B(static_cast<size_t>(K) * N);
            std::vector<int8_t> h_B_vnni;
            std::vector<float> h_scale_B(N);

            for (auto &v : h_A)
                v = dist_a(rng);
            for (auto &v : h_B)
                v = static_cast<int8_t>(dist_b(rng));
            for (auto &v : h_scale_B)
                v = dist_s(rng);

            packVnniWeights(h_B, N, K, h_B_vnni);

            // Allocate device memory
            float *d_A = nullptr, *d_scale_B = nullptr, *d_C = nullptr;
            int8_t *d_B_vnni = nullptr, *d_A_int8 = nullptr;
            float *d_scale_A_bw = nullptr;

            const int blocks_per_row = K / ACT_BLOCK_K;

            (void)hipMalloc(&d_A, K * sizeof(float));
            (void)hipMalloc(&d_A_int8, K * sizeof(int8_t));
            (void)hipMalloc(&d_scale_A_bw, blocks_per_row * sizeof(float));
            (void)hipMalloc(&d_B_vnni, h_B_vnni.size() * sizeof(int8_t));
            (void)hipMalloc(&d_scale_B, N * sizeof(float));
            (void)hipMalloc(&d_C, N * sizeof(float));

            (void)hipMemcpy(d_A, h_A.data(), K * sizeof(float), hipMemcpyHostToDevice);
            (void)hipMemcpy(d_B_vnni, h_B_vnni.data(), h_B_vnni.size() * sizeof(int8_t), hipMemcpyHostToDevice);
            (void)hipMemcpy(d_scale_B, h_scale_B.data(), N * sizeof(float), hipMemcpyHostToDevice);
            (void)hipDeviceSynchronize();

            // Pre-quantize activations (done once, not timed)
            rocmQuantGemm_quantizeActivationsBlockwise(d_A, d_A_int8, d_scale_A_bw, 1, K, device_id_, nullptr, ACT_BLOCK_K);
            (void)hipDeviceSynchronize();

            // Set KB override (-1 for auto)
            rocmGemv_int8_vnni_set_tuning_overrides(-1, kb_forced == 0 ? -1 : kb_forced);

            // Warmup
            for (int i = 0; i < WARMUP_ITERS; ++i)
            {
                rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                    d_A_int8, d_B_vnni, d_C, d_scale_A_bw, d_scale_B,
                    N, K, 1.0f, 0.0f, nullptr, nullptr, device_id_, nullptr);
            }
            (void)hipDeviceSynchronize();

            // Timed runs
            hipEvent_t ev_start, ev_stop;
            (void)hipEventCreate(&ev_start);
            (void)hipEventCreate(&ev_stop);

            std::vector<double> times_us;
            times_us.reserve(BENCH_ITERS);

            for (int i = 0; i < BENCH_ITERS; ++i)
            {
                (void)hipDeviceSynchronize();
                (void)hipEventRecord(ev_start, nullptr);

                bool ok = rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                    d_A_int8, d_B_vnni, d_C, d_scale_A_bw, d_scale_B,
                    N, K, 1.0f, 0.0f, nullptr, nullptr, device_id_, nullptr);
                if (!ok)
                {
                    result.success = false;
                    goto cleanup;
                }

                (void)hipEventRecord(ev_stop, nullptr);
                (void)hipEventSynchronize(ev_stop);

                float ms = 0.0f;
                (void)hipEventElapsedTime(&ms, ev_start, ev_stop);
                times_us.push_back(static_cast<double>(ms) * 1000.0);
            }

            {
                std::sort(times_us.begin(), times_us.end());
                result.min_us = times_us.front();
                result.max_us = times_us.back();
                result.median_us = times_us[times_us.size() / 2];
                result.success = true;
            }

        cleanup:
            (void)hipEventDestroy(ev_start);
            (void)hipEventDestroy(ev_stop);
            rocmGemv_int8_vnni_reset_tuning_overrides();

            (void)hipFree(d_A);
            (void)hipFree(d_A_int8);
            (void)hipFree(d_scale_A_bw);
            (void)hipFree(d_B_vnni);
            (void)hipFree(d_scale_B);
            (void)hipFree(d_C);

            return result;
#endif
        }

        // =========================================================================
        // KB sweep for one projection shape
        // =========================================================================
        struct SweepRow
        {
            GemvShape shape;
            int tp;
            int heuristic_kb;
            std::vector<KBResult> results; // one per KB in sweep
            int best_kb;                   // KB with lowest median
            double best_us;
            double auto_us; // Latency at heuristic KB
        };

        SweepRow sweepKB(const GemvShape &shape, int tp, const std::vector<int> &kb_values)
        {
            SweepRow row{};
            row.shape = shape;
            row.tp = tp;
            row.heuristic_kb = computeHeuristicKB(shape.N, shape.K);
            row.best_us = 1e9;
            row.auto_us = 0.0;

            for (int kb : kb_values)
            {
                auto r = benchmarkGemvKB(shape.N, shape.K, kb);
                if (r.success && r.median_us < row.best_us)
                {
                    row.best_us = r.median_us;
                    row.best_kb = kb;
                }
                if (kb == 0 && r.success)
                {
                    row.auto_us = r.median_us;
                }
                row.results.push_back(r);
            }

            return row;
        }

        // =========================================================================
        // Full TP sweep for all projections
        // =========================================================================
        void runKBScalingSweep(
            const ModelDims &model,
            const std::vector<int> &tp_degrees,
            const std::vector<int> &kb_values)
        {
#ifndef HAVE_ROCM
            GTEST_SKIP() << "No ROCm support";
#else
            if (!has_device_)
                GTEST_SKIP() << "No ROCm device";

            // Collect all rows
            std::vector<SweepRow> rows;
            for (int tp : tp_degrees)
            {
                auto shapes = getTPShapes(model, tp);
                for (const auto &shape : shapes)
                {
                    // Skip if N < TILE_N (would use different kernel path)
                    if (shape.N < TILE_N)
                        continue;

                    auto row = sweepKB(shape, tp, kb_values);
                    rows.push_back(std::move(row));
                }
            }

            // =================================================================
            // Table 1: Full KB sweep — latency (μs) per KB value
            // =================================================================
            {
                fort::utf8_table table;
                table.set_border_style(FT_DOUBLE2_STYLE);

                // Header
                table << fort::header << "Projection" << "TP" << "N" << "K" << "grid_n" << "Heur KB";
                for (int kb : kb_values)
                {
                    char colhdr[16];
                    if (kb == 0)
                        snprintf(colhdr, sizeof(colhdr), "auto");
                    else
                        snprintf(colhdr, sizeof(colhdr), "KB=%d", kb);
                    table << colhdr;
                }
                table << "Best" << "Δ auto" << fort::endr;

                table.column(0).set_cell_text_align(fort::text_align::left);
                for (size_t i = 1; i < 6 + kb_values.size() + 2; ++i)
                    table.column(i).set_cell_text_align(fort::text_align::right);

                for (const auto &row : rows)
                {
                    const int grid_n = (row.shape.N + TILE_N - 1) / TILE_N;
                    table << row.shape.name;

                    char tp_str[8];
                    snprintf(tp_str, sizeof(tp_str), "%d", row.tp);
                    table << tp_str;

                    table << row.shape.N << row.shape.K << grid_n << row.heuristic_kb;

                    for (const auto &r : row.results)
                    {
                        char cell[16];
                        if (r.success)
                            snprintf(cell, sizeof(cell), "%.1f", r.median_us);
                        else
                            snprintf(cell, sizeof(cell), "FAIL");
                        table << cell;
                    }

                    // Best KB
                    char best_cell[16];
                    if (row.best_kb == 0)
                        snprintf(best_cell, sizeof(best_cell), "auto");
                    else
                        snprintf(best_cell, sizeof(best_cell), "KB=%d", row.best_kb);
                    table << best_cell;

                    // Delta vs auto
                    if (row.auto_us > 0.0 && row.best_us > 0.0)
                    {
                        double delta_pct = ((row.auto_us - row.best_us) / row.auto_us) * 100.0;
                        char delta_cell[16];
                        snprintf(delta_cell, sizeof(delta_cell), "%+.1f%%", -delta_pct);
                        if (std::abs(delta_pct) < 1.0)
                            snprintf(delta_cell, sizeof(delta_cell), "≈0%%");
                        table << delta_cell;
                    }
                    else
                    {
                        table << "N/A";
                    }

                    table << fort::endr;
                }

                fprintf(stderr,
                        "\n%s — GEMV KB Sweep (M=1 decode, LDS k-reduce, WK=%d, ACT_BK=%d)\n"
                        "Device: %s (%d CUs, TILE_N=%d)\n%s\n",
                        model.name, WK, ACT_BLOCK_K,
                        device_name_.c_str(), num_cus_, TILE_N,
                        table.to_string().c_str());
            }

            // =================================================================
            // Table 2: TP Scaling Summary — auto KB latency by TP degree
            // =================================================================
            {
                // Group by projection name, show TP=1 baseline vs TP=2/4
                struct ProjSummary
                {
                    const char *name;
                    double tp1_us = 0.0;
                    double tp2_us = 0.0;
                    double tp4_us = 0.0;
                    int tp1_kb = 0, tp2_kb = 0, tp4_kb = 0;
                };

                std::vector<ProjSummary> summaries;
                for (const auto &row : rows)
                {
                    // Find or create entry
                    ProjSummary *found = nullptr;
                    for (auto &s : summaries)
                    {
                        if (std::string(s.name) == row.shape.name)
                        {
                            found = &s;
                            break;
                        }
                    }
                    if (!found)
                    {
                        summaries.push_back({row.shape.name});
                        found = &summaries.back();
                    }

                    if (row.tp == 1)
                    {
                        found->tp1_us = row.auto_us;
                        found->tp1_kb = row.heuristic_kb;
                    }
                    else if (row.tp == 2)
                    {
                        found->tp2_us = row.auto_us;
                        found->tp2_kb = row.heuristic_kb;
                    }
                    else if (row.tp == 4)
                    {
                        found->tp4_us = row.auto_us;
                        found->tp4_kb = row.heuristic_kb;
                    }
                }

                fort::utf8_table table;
                table.set_border_style(FT_DOUBLE2_STYLE);

                table << fort::header << "Projection" << "TP=1 (μs)" << "KB"
                      << "TP=2 (μs)" << "KB" << "TP=2 Eff"
                      << "TP=4 (μs)" << "KB" << "TP=4 Eff" << fort::endr;

                table.column(0).set_cell_text_align(fort::text_align::left);
                for (int i = 1; i <= 8; ++i)
                    table.column(i).set_cell_text_align(fort::text_align::right);

                for (const auto &s : summaries)
                {
                    table << s.name;

                    char cell[16];
                    snprintf(cell, sizeof(cell), "%.1f", s.tp1_us);
                    table << cell << s.tp1_kb;

                    if (s.tp2_us > 0.0)
                    {
                        snprintf(cell, sizeof(cell), "%.1f", s.tp2_us);
                        table << cell << s.tp2_kb;

                        // TP=2 ideal is same latency (N halved = half the BW)
                        // But for row-parallel (K halved), ideal is also same latency
                        // Scaling efficiency: tp1_latency / tp2_latency (should be ~1.0 if no overhead)
                        // Actually for decode GEMV, halving N halves memory BW needed,
                        // so ideal TP=2 latency = tp1 / 2... but only if BW-bound.
                        // For small N, we're compute-bound, so ideally TP=2 latency ≈ tp1.
                        // Report as % of TP=1 to show overhead.
                        double pct = (s.tp2_us / s.tp1_us) * 100.0;
                        snprintf(cell, sizeof(cell), "%.0f%%", pct);
                        table << cell;
                    }
                    else
                    {
                        table << "N/A" << "" << "";
                    }

                    if (s.tp4_us > 0.0)
                    {
                        snprintf(cell, sizeof(cell), "%.1f", s.tp4_us);
                        table << cell << s.tp4_kb;

                        double pct = (s.tp4_us / s.tp1_us) * 100.0;
                        snprintf(cell, sizeof(cell), "%.0f%%", pct);
                        table << cell;
                    }
                    else
                    {
                        table << "N/A" << "" << "";
                    }

                    table << fort::endr;
                }

                fprintf(stderr,
                        "\n%s — GEMV TP Scaling Summary (auto KB, M=1 decode)\n"
                        "Efficiency = TP_N latency / TP=1 latency (lower is better for col-parallel)\n%s\n",
                        model.name, table.to_string().c_str());
            }

            // =================================================================
            // Table 3: Grid analysis — blocks, waves, atomicAdds per output
            // =================================================================
            {
                fort::utf8_table table;
                table.set_border_style(FT_DOUBLE2_STYLE);

                table << fort::header << "Projection" << "TP" << "N" << "K"
                      << "grid_n" << "KB (auto)" << "Total Blocks"
                      << "Waves/CU" << "AtomicAdds/elem" << "Best KB"
                      << "Best Blocks" << fort::endr;

                table.column(0).set_cell_text_align(fort::text_align::left);
                for (int i = 1; i <= 10; ++i)
                    table.column(i).set_cell_text_align(fort::text_align::right);

                for (const auto &row : rows)
                {
                    const int grid_n = (row.shape.N + TILE_N - 1) / TILE_N;
                    const int auto_blocks = grid_n * row.heuristic_kb;
                    const double waves_per_cu = static_cast<double>(auto_blocks) / num_cus_;
                    const int atomics_per_elem = std::max(0, row.heuristic_kb - 1);

                    table << row.shape.name << row.tp << row.shape.N << row.shape.K
                          << grid_n << row.heuristic_kb << auto_blocks;

                    char cell[16];
                    snprintf(cell, sizeof(cell), "%.1f", waves_per_cu);
                    table << cell << atomics_per_elem;

                    if (row.best_kb == 0)
                        table << "auto";
                    else
                    {
                        snprintf(cell, sizeof(cell), "%d", row.best_kb);
                        table << cell;
                    }

                    const int best_kb_actual = (row.best_kb == 0) ? row.heuristic_kb : row.best_kb;
                    table << (grid_n * best_kb_actual);

                    table << fort::endr;
                }

                fprintf(stderr,
                        "\n%s — GEMV Grid Analysis (%d CUs)\n%s\n",
                        model.name, num_cus_, table.to_string().c_str());
            }
#endif
        }
    };

    // ============================================================================
    // Test Cases
    // ============================================================================

    // ---------------------------------------------------------------------------
    // Full KB sweep: Qwen2.5-7B projections at TP=1/2/4
    // KB values chosen to cover: no split, small splits, heuristic range, high splits
    // ---------------------------------------------------------------------------
    TEST_F(ROCmGemvKBScalingPerf, Qwen7B_KBSweep)
    {
        runKBScalingSweep(
            kQwen7B,
            /*tp_degrees=*/{1, 2, 4},
            /*kb_values=*/{0, 1, 2, 4, 7, 8, 14, 16, 28, 56});
    }

    // ---------------------------------------------------------------------------
    // KB sweeps for all Qwen model sizes at TP=1/2/4
    // ---------------------------------------------------------------------------
    TEST_F(ROCmGemvKBScalingPerf, Qwen05B_KBSweep)
    {
        runKBScalingSweep(
            kQwen05B,
            /*tp_degrees=*/{1, 2, 4},
            /*kb_values=*/{0, 1, 2, 4, 7, 8, 14, 16, 28});
    }

    TEST_F(ROCmGemvKBScalingPerf, Qwen3B_KBSweep)
    {
        runKBScalingSweep(
            kQwen3B,
            /*tp_degrees=*/{1, 2, 4},
            /*kb_values=*/{0, 1, 2, 4, 7, 8, 14, 16, 28, 56});
    }

    TEST_F(ROCmGemvKBScalingPerf, Qwen14B_KBSweep)
    {
        runKBScalingSweep(
            kQwen14B,
            /*tp_degrees=*/{1, 2, 4},
            /*kb_values=*/{0, 1, 2, 4, 7, 8, 14, 16, 28, 56});
    }

    TEST_F(ROCmGemvKBScalingPerf, Qwen32B_KBSweep)
    {
        runKBScalingSweep(
            kQwen32B,
            /*tp_degrees=*/{1, 2, 4},
            /*kb_values=*/{0, 1, 2, 4, 7, 8, 14, 16, 28, 56});
    }

    // ---------------------------------------------------------------------------
    // Fine-grained KB sweep for Q/Wo at TP=2 (the most problematic case)
    // N=1792 at TP=2, act_blocks=112, heuristic picks KB=14
    // Sweep +/-3 around the heuristic to find exact optimum
    // ---------------------------------------------------------------------------
    TEST_F(ROCmGemvKBScalingPerf, Qwen7B_QWo_TP2_FineGrained)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device";

        // Q proj at TP=2: N=1792, K=3584
        const GemvShape shape{"Q/Wo proj (TP=2)", 1792, 3584};
        const int heur_kb = computeHeuristicKB(shape.N, shape.K);

        // Fine sweep: 1 through heuristic+4, plus 2× and 3× heuristic
        std::vector<int> kb_values = {0}; // auto first
        for (int kb = 1; kb <= std::min(heur_kb + 4, 56); ++kb)
            kb_values.push_back(kb);
        if (heur_kb * 2 <= 56)
            kb_values.push_back(heur_kb * 2);

        // Remove duplicates
        std::sort(kb_values.begin(), kb_values.end());
        kb_values.erase(std::unique(kb_values.begin(), kb_values.end()), kb_values.end());

        std::vector<KBResult> results;
        double best_us = 1e9;
        int best_kb = -1;

        for (int kb : kb_values)
        {
            auto r = benchmarkGemvKB(shape.N, shape.K, kb);
            ASSERT_TRUE(r.success) << "KB=" << kb;
            results.push_back(r);
            if (r.median_us < best_us)
            {
                best_us = r.median_us;
                best_kb = kb;
            }
        }

        // Render table
        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);

        table << fort::header << "KB" << "Median (μs)" << "Min (μs)" << "grid_n×KB"
              << "AtomicAdds" << "Δ best" << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::right);
        for (int i = 1; i <= 5; ++i)
            table.column(i).set_cell_text_align(fort::text_align::right);

        const int grid_n = (shape.N + TILE_N - 1) / TILE_N;

        for (size_t i = 0; i < results.size(); ++i)
        {
            const auto &r = results[i];
            int kb_actual = kb_values[i] == 0 ? heur_kb : kb_values[i];

            char kb_cell[16];
            if (kb_values[i] == 0)
                snprintf(kb_cell, sizeof(kb_cell), "auto(%d)", heur_kb);
            else
                snprintf(kb_cell, sizeof(kb_cell), "%d", kb_values[i]);
            table << kb_cell;

            char cell[16];
            snprintf(cell, sizeof(cell), "%.2f", r.median_us);
            table << cell;

            snprintf(cell, sizeof(cell), "%.2f", r.min_us);
            table << cell;

            table << (grid_n * kb_actual);
            table << std::max(0, kb_actual - 1);

            double delta = ((r.median_us - best_us) / best_us) * 100.0;
            snprintf(cell, sizeof(cell), "+%.1f%%", delta);
            if (delta < 0.5)
                snprintf(cell, sizeof(cell), "★ best");
            table << cell;

            table << fort::endr;
        }

        fprintf(stderr,
                "\n%s — Fine-Grained KB Sweep (N=%d, K=%d, grid_n=%d)\n"
                "Heuristic KB=%d, Best KB=%s\n"
                "Device: %s (%d CUs)\n%s\n",
                shape.name, shape.N, shape.K, grid_n,
                heur_kb, best_kb == 0 ? "auto" : std::to_string(best_kb).c_str(),
                device_name_.c_str(), num_cus_,
                table.to_string().c_str());
#endif
    }

    // ---------------------------------------------------------------------------
    // Fine-grained KB sweep for FFN Down at TP=2 (large K, row-parallel)
    // N=3584, K=9472 at TP=2 → large K means heuristic already picks low KB
    // but contention still possible with small grid_n
    // ---------------------------------------------------------------------------
    TEST_F(ROCmGemvKBScalingPerf, Qwen7B_FFNDown_TP2_FineGrained)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device";

        // FFN Down at TP=2: N=3584, K=9472 (input-parallel: K halved)
        const GemvShape shape{"FFN Down (TP=2)", 3584, 9472};
        const int heur_kb = computeHeuristicKB(shape.N, shape.K);

        std::vector<int> kb_values = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20};

        std::vector<KBResult> results;
        double best_us = 1e9;
        int best_kb = -1;

        for (int kb : kb_values)
        {
            auto r = benchmarkGemvKB(shape.N, shape.K, kb);
            ASSERT_TRUE(r.success) << "KB=" << kb;
            results.push_back(r);
            if (r.median_us < best_us)
            {
                best_us = r.median_us;
                best_kb = kb;
            }
        }

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);

        table << fort::header << "KB" << "Median (μs)" << "Min (μs)" << "grid_n×KB"
              << "AtomicAdds" << "Δ best" << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::right);
        for (int i = 1; i <= 5; ++i)
            table.column(i).set_cell_text_align(fort::text_align::right);

        const int grid_n = (shape.N + TILE_N - 1) / TILE_N;

        for (size_t i = 0; i < results.size(); ++i)
        {
            const auto &r = results[i];
            int kb_actual = kb_values[i] == 0 ? heur_kb : kb_values[i];

            char kb_cell[16];
            if (kb_values[i] == 0)
                snprintf(kb_cell, sizeof(kb_cell), "auto(%d)", heur_kb);
            else
                snprintf(kb_cell, sizeof(kb_cell), "%d", kb_values[i]);
            table << kb_cell;

            char cell[16];
            snprintf(cell, sizeof(cell), "%.2f", r.median_us);
            table << cell;

            snprintf(cell, sizeof(cell), "%.2f", r.min_us);
            table << cell;

            table << (grid_n * kb_actual);
            table << std::max(0, kb_actual - 1);

            double delta = ((r.median_us - best_us) / best_us) * 100.0;
            snprintf(cell, sizeof(cell), "+%.1f%%", delta);
            if (delta < 0.5)
                snprintf(cell, sizeof(cell), "★ best");
            table << cell;

            table << fort::endr;
        }

        fprintf(stderr,
                "\n%s — Fine-Grained KB Sweep (N=%d, K=%d, grid_n=%d)\n"
                "Heuristic KB=%d, Best KB=%s\n"
                "Device: %s (%d CUs)\n%s\n",
                shape.name, shape.N, shape.K, grid_n,
                heur_kb, best_kb == 0 ? "auto" : std::to_string(best_kb).c_str(),
                device_name_.c_str(), num_cus_,
                table.to_string().c_str());
#endif
    }

} // anonymous namespace
