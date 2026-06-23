/**
 * @file Perf__ROCmFlashAttentionDecode.cpp
 * @brief Performance benchmarks for ROCm Flash Decoding kernel with TP-sliced head counts
 *
 * Measures decode attention latency and CU occupancy across configurations that
 * mirror tensor-parallel slicing scenarios (TP=1, TP=2, TP=4).
 *
 * **Key insight**: At TP=2 with Qwen-7B (28 Q-heads → 14/device), only 14 of 60 CUs
 * are active for the flash decoding grid. This sweep measures whether the split-K
 * heuristic compensates by generating enough splits to fill extra CUs.
 *
 * **Tested Configurations**:
 * - Models: Qwen2.5-0.5B (14h/2kv, hd=64), Qwen2.5-7B (28h/4kv, hd=128)
 * - TP degrees: 1, 2, 4
 * - KV cache lengths: 128, 256, 512, 1024, 2048, 4096
 * - KV types: FP32  (FP16 / Q8_1 can be added later)
 *
 * **Metrics**:
 * - Latency (μs): median / min / max
 * - CU occupancy: n_heads × splits / total_CUs
 * - TP scaling efficiency: ideal_latency / actual_latency
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

#include "fort.hpp"

// Extern C wrappers from ROCmFlashAttentionKernels.hip
extern "C"
{
    int hipFlashAttn_decode_fp32(
        const float *Q, const float *K_cache, const float *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int batch_size, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int num_splits,
        const void *device_params,
        void *stream);
}

namespace
{

    // ============================================================================
    // Constants
    // ============================================================================

    constexpr int WARMUP_ITERS = 20;
    constexpr int BENCH_ITERS = 100;
    constexpr int MAX_SPLITS = 32; // Must match FD_MAX_SPLITS_SUPPORTED

    // ============================================================================
    // Model configurations
    // ============================================================================

    struct ModelConfig
    {
        const char *name;
        int n_heads;
        int n_kv_heads;
        int head_dim;
    };

    static constexpr ModelConfig kQwen05B = {"Qwen2.5-0.5B", 14, 2, 64};
    static constexpr ModelConfig kQwen7B = {"Qwen2.5-7B", 28, 4, 128};
    static constexpr ModelConfig kQwen14B = {"Qwen2.5-14B", 40, 8, 128};

    // ============================================================================
    // TP shape derivation
    // ============================================================================

    struct TPConfig
    {
        const char *label;
        int tp_degree;
        int n_heads;
        int n_kv_heads;
        int head_dim;
    };

    static std::vector<TPConfig> getTPConfigs(const ModelConfig &model, const std::vector<int> &tp_degrees)
    {
        std::vector<TPConfig> configs;
        for (int tp : tp_degrees)
        {
            int local_heads = model.n_heads / tp;
            int local_kv_heads = std::max(1, model.n_kv_heads / tp);
            if (local_heads < 1)
                continue;

            char label[64];
            snprintf(label, sizeof(label), "TP=%d (%dh/%dkv)", tp, local_heads, local_kv_heads);
            configs.push_back({strdup(label), tp, local_heads, local_kv_heads, model.head_dim});
        }
        return configs;
    }

    // ============================================================================
    // Benchmark result
    // ============================================================================

    struct DecodeResult
    {
        double median_us;
        double min_us;
        double max_us;
        int effective_splits; // From heuristic (auto)
        bool success;
    };

    // ============================================================================
    // Test fixture
    // ============================================================================

    class ROCmFlashAttentionDecodePerf : public ::testing::Test
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
        // Core benchmark: runs flash decoding with auto split-K selection
        // =========================================================================
        DecodeResult benchmarkDecode(
            int n_heads, int n_kv_heads, int head_dim, int kv_len,
            int forced_splits = 0) // 0 = auto
        {
            DecodeResult result{};
#ifndef HAVE_ROCM
            return result;
#else
            const int batch_size = 1;
            const size_t q_size = static_cast<size_t>(n_heads) * head_dim;
            const size_t kv_size = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;
            const size_t out_size = q_size;
            const size_t partial_size = static_cast<size_t>(n_heads) * MAX_SPLITS * head_dim;

            // Allocate device memory
            float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_O = nullptr;
            float *d_O_partial = nullptr, *d_m_partial = nullptr, *d_l_partial = nullptr;

            (void)hipMalloc(&d_Q, q_size * sizeof(float));
            (void)hipMalloc(&d_K, kv_size * sizeof(float));
            (void)hipMalloc(&d_V, kv_size * sizeof(float));
            (void)hipMalloc(&d_O, out_size * sizeof(float));
            (void)hipMalloc(&d_O_partial, partial_size * sizeof(float));
            (void)hipMalloc(&d_m_partial, static_cast<size_t>(n_heads) * MAX_SPLITS * sizeof(float));
            (void)hipMalloc(&d_l_partial, static_cast<size_t>(n_heads) * MAX_SPLITS * sizeof(float));

            // Initialize with random data
            {
                std::vector<float> h_Q(q_size), h_K(kv_size), h_V(kv_size);
                std::mt19937 rng(42);
                std::normal_distribution<float> dist(0.0f, 0.1f);
                for (auto &v : h_Q)
                    v = dist(rng);
                for (auto &v : h_K)
                    v = dist(rng);
                for (auto &v : h_V)
                    v = dist(rng);

                (void)hipMemcpy(d_Q, h_Q.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
                (void)hipMemcpy(d_K, h_K.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
                (void)hipMemcpy(d_V, h_V.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
            }
            (void)hipDeviceSynchronize();

            // device_params is a GPU-dereferenced pointer in the kernel.
            // Pass nullptr so the kernel uses the kv_len argument directly.

            // Warmup (also triggers autotuning)
            for (int i = 0; i < WARMUP_ITERS; ++i)
            {
                int rc = hipFlashAttn_decode_fp32(
                    d_Q, d_K, d_V, d_O,
                    d_O_partial, d_m_partial, d_l_partial,
                    batch_size, kv_len, n_heads, n_kv_heads, head_dim,
                    forced_splits, nullptr, nullptr);
                if (rc != 0)
                {
                    result.success = false;
                    goto cleanup;
                }
            }
            (void)hipDeviceSynchronize();

            // Benchmark with HIP events
            {
                hipEvent_t ev_start, ev_stop;
                (void)hipEventCreate(&ev_start);
                (void)hipEventCreate(&ev_stop);

                std::vector<double> times_us;
                times_us.reserve(BENCH_ITERS);

                for (int i = 0; i < BENCH_ITERS; ++i)
                {
                    (void)hipDeviceSynchronize();
                    (void)hipEventRecord(ev_start, nullptr);

                    int rc = hipFlashAttn_decode_fp32(
                        d_Q, d_K, d_V, d_O,
                        d_O_partial, d_m_partial, d_l_partial,
                        batch_size, kv_len, n_heads, n_kv_heads, head_dim,
                        forced_splits, nullptr, nullptr);
                    if (rc != 0)
                    {
                        result.success = false;
                        (void)hipEventDestroy(ev_start);
                        (void)hipEventDestroy(ev_stop);
                        goto cleanup;
                    }

                    (void)hipEventRecord(ev_stop, nullptr);
                    (void)hipEventSynchronize(ev_stop);

                    float ms = 0.0f;
                    (void)hipEventElapsedTime(&ms, ev_start, ev_stop);
                    times_us.push_back(static_cast<double>(ms) * 1000.0);
                }

                (void)hipEventDestroy(ev_start);
                (void)hipEventDestroy(ev_stop);

                std::sort(times_us.begin(), times_us.end());
                result.min_us = times_us.front();
                result.max_us = times_us.back();
                result.median_us = times_us[times_us.size() / 2];
                result.success = true;
            }

        cleanup:
            (void)hipFree(d_Q);
            (void)hipFree(d_K);
            (void)hipFree(d_V);
            (void)hipFree(d_O);
            (void)hipFree(d_O_partial);
            (void)hipFree(d_m_partial);
            (void)hipFree(d_l_partial);

            return result;
#endif
        }

        // =========================================================================
        // Run full TP scaling sweep for one model
        // =========================================================================
        void runTPScalingSweep(
            const ModelConfig &model,
            const std::vector<int> &kv_lengths,
            const std::vector<int> &tp_degrees)
        {
#ifndef HAVE_ROCM
            GTEST_SKIP() << "No ROCm support";
#else
            if (!has_device_)
                GTEST_SKIP() << "No ROCm device";

            auto tp_configs = getTPConfigs(model, tp_degrees);

            // Benchmark all (tp_config, kv_len) combinations
            struct Cell
            {
                DecodeResult result;
                int n_heads;
                int n_kv_heads;
            };

            // [tp_idx][kv_idx]
            std::vector<std::vector<Cell>> grid(tp_configs.size());
            for (size_t ti = 0; ti < tp_configs.size(); ++ti)
            {
                grid[ti].resize(kv_lengths.size());
                for (size_t ki = 0; ki < kv_lengths.size(); ++ki)
                {
                    const auto &tc = tp_configs[ti];
                    auto r = benchmarkDecode(tc.n_heads, tc.n_kv_heads, tc.head_dim, kv_lengths[ki]);
                    ASSERT_TRUE(r.success)
                        << model.name << " " << tc.label << " kv=" << kv_lengths[ki];
                    grid[ti][ki] = {r, tc.n_heads, tc.n_kv_heads};
                }
            }

            // =====================================================================
            // Table 1: Latency by TP degree × KV length
            // =====================================================================
            {
                fort::utf8_table table;
                table.set_border_style(FT_DOUBLE2_STYLE);

                // Header: TP Config | kv=128 | kv=256 | ...
                table << fort::header << "Config";
                for (int kv : kv_lengths)
                {
                    char colhdr[32];
                    snprintf(colhdr, sizeof(colhdr), "kv=%d", kv);
                    table << colhdr;
                }
                table << fort::endr;

                table.column(0).set_cell_text_align(fort::text_align::left);
                for (size_t i = 1; i <= kv_lengths.size(); ++i)
                    table.column(i).set_cell_text_align(fort::text_align::right);

                for (size_t ti = 0; ti < tp_configs.size(); ++ti)
                {
                    table << tp_configs[ti].label;
                    for (size_t ki = 0; ki < kv_lengths.size(); ++ki)
                    {
                        char cell[32];
                        snprintf(cell, sizeof(cell), "%.1f", grid[ti][ki].result.median_us);
                        table << cell;
                    }
                    table << fort::endr;
                }

                fprintf(stderr, "\n%s — Flash Decoding Latency (μs), head_dim=%d\nDevice: %s (%d CUs)\n%s\n",
                        model.name, model.head_dim, device_name_.c_str(), num_cus_,
                        table.to_string().c_str());
            }

            // =====================================================================
            // Table 2: TP Scaling Efficiency (TP>1 vs TP=1 baseline)
            // =====================================================================
            if (tp_configs.size() > 1)
            {
                fort::utf8_table table;
                table.set_border_style(FT_DOUBLE2_STYLE);

                table << fort::header << "Config" << "Heads";
                for (int kv : kv_lengths)
                {
                    char colhdr[32];
                    snprintf(colhdr, sizeof(colhdr), "kv=%d", kv);
                    table << colhdr;
                }
                table << fort::endr;

                table.column(0).set_cell_text_align(fort::text_align::left);
                table.column(1).set_cell_text_align(fort::text_align::right);
                for (size_t i = 2; i <= kv_lengths.size() + 1; ++i)
                    table.column(i).set_cell_text_align(fort::text_align::right);

                // TP=1 baseline row
                {
                    table << tp_configs[0].label << tp_configs[0].n_heads;
                    for (size_t ki = 0; ki < kv_lengths.size(); ++ki)
                        table << "baseline";
                    table << fort::endr;
                }

                // TP>1 rows with scaling efficiency
                for (size_t ti = 1; ti < tp_configs.size(); ++ti)
                {
                    table << tp_configs[ti].label << tp_configs[ti].n_heads;
                    for (size_t ki = 0; ki < kv_lengths.size(); ++ki)
                    {
                        double base_us = grid[0][ki].result.median_us;
                        double tp_us = grid[ti][ki].result.median_us;
                        double ideal_us = base_us / static_cast<double>(tp_configs[ti].tp_degree);
                        double efficiency = (ideal_us / tp_us) * 100.0;

                        char cell[32];
                        snprintf(cell, sizeof(cell), "%.0f%%", efficiency);
                        table << cell;
                    }
                    table << fort::endr;
                }

                fprintf(stderr, "\n%s — Flash Decoding TP Scaling Efficiency\n%s\n",
                        model.name, table.to_string().c_str());
            }

            // =====================================================================
            // Table 3: CU Occupancy Estimate
            // Occupancy = min(n_heads × splits, num_CUs) / num_CUs
            // This is a rough upper bound — actual occupancy depends on register
            // pressure, LDS usage, and wavefront scheduling.
            // =====================================================================
            if (num_cus_ > 0)
            {
                fort::utf8_table table;
                table.set_border_style(FT_DOUBLE2_STYLE);

                table << fort::header << "Config" << "Heads";
                for (int kv : kv_lengths)
                {
                    char colhdr[32];
                    snprintf(colhdr, sizeof(colhdr), "kv=%d", kv);
                    table << colhdr;
                }
                table << fort::endr;

                table.column(0).set_cell_text_align(fort::text_align::left);
                table.column(1).set_cell_text_align(fort::text_align::right);
                for (size_t i = 2; i <= kv_lengths.size() + 1; ++i)
                    table.column(i).set_cell_text_align(fort::text_align::right);

                for (size_t ti = 0; ti < tp_configs.size(); ++ti)
                {
                    table << tp_configs[ti].label << tp_configs[ti].n_heads;
                    for (size_t ki = 0; ki < kv_lengths.size(); ++ki)
                    {
                        // Estimate splits from the heuristic
                        int kv = kv_lengths[ki];
                        int hd = model.head_dim;
                        int64_t total_elems = static_cast<int64_t>(kv) * hd;
                        int base_splits = static_cast<int>((total_elems + 16383) / 16384);
                        base_splits = std::max(base_splits, 1);

                        // Aspect ratio adjustment (mirrors chooseDecodeSplitsAuto)
                        if (kv >= hd * 8)
                            base_splits = (base_splits * 3 + 1) / 2;
                        else if (kv <= hd * 2)
                            base_splits = std::max(1, (base_splits + 1) / 2);

                        // Round to power of 2, clamp to MAX_SPLITS
                        int splits = 1;
                        while (splits < base_splits && splits < MAX_SPLITS)
                            splits <<= 1;
                        splits = std::min(splits, MAX_SPLITS);

                        // CU-aware occupancy floor (mirrors chooseDecodeSplitsAuto)
                        int n_heads = tp_configs[ti].n_heads;
                        int target_blocks = (num_cus_ * 3) / 4; // 75% target
                        if (n_heads * splits < target_blocks)
                        {
                            int occ_splits = (target_blocks + n_heads - 1) / n_heads;
                            int occ_p2 = 1;
                            while (occ_p2 < occ_splits && occ_p2 < MAX_SPLITS)
                                occ_p2 <<= 1;
                            occ_p2 = std::min(occ_p2, MAX_SPLITS);
                            splits = std::max(splits, occ_p2);
                        }

                        int total_blocks = tp_configs[ti].n_heads * splits;
                        double occupancy = std::min(1.0, static_cast<double>(total_blocks) / num_cus_);

                        char cell[48];
                        snprintf(cell, sizeof(cell), "%d×%d=%d (%.0f%%)",
                                 tp_configs[ti].n_heads, splits, total_blocks,
                                 occupancy * 100.0);
                        table << cell;
                    }
                    table << fort::endr;
                }

                fprintf(stderr, "\n%s — CU Occupancy Estimate (heads × splits = blocks / %d CUs)\n%s\n",
                        model.name, num_cus_, table.to_string().c_str());
            }
#endif
        }

        // =========================================================================
        // Split-K sweep: force specific split counts and measure impact
        // =========================================================================
        void runSplitKSweep(
            const ModelConfig &model,
            int tp_degree,
            const std::vector<int> &kv_lengths,
            const std::vector<int> &split_counts)
        {
#ifndef HAVE_ROCM
            GTEST_SKIP() << "No ROCm support";
#else
            if (!has_device_)
                GTEST_SKIP() << "No ROCm device";

            int local_heads = model.n_heads / tp_degree;
            int local_kv_heads = std::max(1, model.n_kv_heads / tp_degree);

            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);

            // Header
            table << fort::header << "KV Len";
            for (int s : split_counts)
            {
                char colhdr[32];
                if (s == 0)
                    snprintf(colhdr, sizeof(colhdr), "auto");
                else
                    snprintf(colhdr, sizeof(colhdr), "s=%d", s);
                table << colhdr;
            }
            table << "Best" << fort::endr;

            table.column(0).set_cell_text_align(fort::text_align::right);
            for (size_t i = 1; i <= split_counts.size() + 1; ++i)
                table.column(i).set_cell_text_align(fort::text_align::right);

            for (int kv : kv_lengths)
            {
                table << kv;
                double best_us = 1e9;
                int best_splits = -1;

                for (int s : split_counts)
                {
                    auto r = benchmarkDecode(local_heads, local_kv_heads, model.head_dim, kv, s);
                    ASSERT_TRUE(r.success) << "splits=" << s << " kv=" << kv;

                    char cell[32];
                    snprintf(cell, sizeof(cell), "%.1f", r.median_us);
                    table << cell;

                    if (r.median_us < best_us)
                    {
                        best_us = r.median_us;
                        best_splits = s;
                    }
                }

                char best_cell[32];
                if (best_splits == 0)
                    snprintf(best_cell, sizeof(best_cell), "auto");
                else
                    snprintf(best_cell, sizeof(best_cell), "s=%d", best_splits);
                table << best_cell;

                table << fort::endr;
            }

            fprintf(stderr,
                    "\n%s TP=%d — Split-K Sweep (heads=%d, head_dim=%d)\n"
                    "Device: %s (%d CUs)\n%s\n",
                    model.name, tp_degree, local_heads, model.head_dim,
                    device_name_.c_str(), num_cus_,
                    table.to_string().c_str());
#endif
        }
    };

    // ============================================================================
    // Test Cases
    // ============================================================================

    // ---------------------------------------------------------------------------
    // Qwen2.5-7B: Full TP scaling sweep (primary target for TP optimization)
    // ---------------------------------------------------------------------------
    TEST_F(ROCmFlashAttentionDecodePerf, Qwen7B_TPScaling)
    {
        runTPScalingSweep(
            kQwen7B,
            /*kv_lengths=*/{128, 256, 512, 1024, 2048, 4096},
            /*tp_degrees=*/{1, 2, 4});
    }

    // ---------------------------------------------------------------------------
    // Qwen2.5-0.5B: Smaller model, head_dim=64 (different kernel characteristics)
    // ---------------------------------------------------------------------------
    TEST_F(ROCmFlashAttentionDecodePerf, Qwen05B_TPScaling)
    {
        runTPScalingSweep(
            kQwen05B,
            /*kv_lengths=*/{128, 256, 512, 1024, 2048, 4096},
            /*tp_degrees=*/{1, 2, 4});
    }

    // ---------------------------------------------------------------------------
    // Split-K sweep for Qwen-7B at TP=2 (14 heads): find optimal split count
    // that compensates for low CU occupancy
    // ---------------------------------------------------------------------------
    TEST_F(ROCmFlashAttentionDecodePerf, Qwen7B_TP2_SplitKSweep)
    {
        runSplitKSweep(
            kQwen7B,
            /*tp_degree=*/2,
            /*kv_lengths=*/{128, 256, 512, 1024, 2048, 4096},
            /*split_counts=*/{0, 1, 2, 4, 8, 16, 32});
    }

    // ---------------------------------------------------------------------------
    // Split-K sweep for Qwen-7B at TP=4 (7 heads): extreme low-head scenario
    // ---------------------------------------------------------------------------
    TEST_F(ROCmFlashAttentionDecodePerf, Qwen7B_TP4_SplitKSweep)
    {
        runSplitKSweep(
            kQwen7B,
            /*tp_degree=*/4,
            /*kv_lengths=*/{128, 256, 512, 1024, 2048, 4096},
            /*split_counts=*/{0, 1, 2, 4, 8, 16, 32});
    }

    // ---------------------------------------------------------------------------
    // Split-K sweep for Qwen-0.5B at TP=2 (7 heads, head_dim=64)
    // ---------------------------------------------------------------------------
    TEST_F(ROCmFlashAttentionDecodePerf, Qwen05B_TP2_SplitKSweep)
    {
        runSplitKSweep(
            kQwen05B,
            /*tp_degree=*/2,
            /*kv_lengths=*/{128, 256, 512, 1024, 2048, 4096},
            /*split_counts=*/{0, 1, 2, 4, 8, 16, 32});
    }

    // ---------------------------------------------------------------------------
    // Qwen2.5-14B: 40 Q-heads, 8 KV-heads, head_dim=128
    // At TP=2: 20h/4kv, TP=4: 10h/2kv — good CU saturation test for larger models
    // ---------------------------------------------------------------------------
    TEST_F(ROCmFlashAttentionDecodePerf, Qwen14B_TPScaling)
    {
        runTPScalingSweep(
            kQwen14B,
            /*kv_lengths=*/{128, 256, 512, 1024, 2048, 4096},
            /*tp_degrees=*/{1, 2, 4});
    }

    TEST_F(ROCmFlashAttentionDecodePerf, Qwen14B_TP2_SplitKSweep)
    {
        runSplitKSweep(
            kQwen14B,
            /*tp_degree=*/2,
            /*kv_lengths=*/{128, 256, 512, 1024, 2048, 4096},
            /*split_counts=*/{0, 1, 2, 4, 8, 16, 32});
    }

    TEST_F(ROCmFlashAttentionDecodePerf, Qwen14B_TP4_SplitKSweep)
    {
        runSplitKSweep(
            kQwen14B,
            /*tp_degree=*/4,
            /*kv_lengths=*/{128, 256, 512, 1024, 2048, 4096},
            /*split_counts=*/{0, 1, 2, 4, 8, 16, 32});
    }

} // anonymous namespace
