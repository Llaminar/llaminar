/**
 * @file Perf__CUDAStreamKAB.cpp
 * @brief A/B performance comparison: standard tiling vs stream-K persistent grid.
 *
 * For each (shape, M) combination, runs the GEMM kernel with stream-K forced OFF
 * and forced ON, printing a side-by-side comparison table. This enables empirical
 * identification of shapes where stream-K is profitable.
 *
 * Environment variables:
 *   LLAMINAR_STREAMK_AB_SHAPES    - Comma-separated shape names (default: 7B shapes)
 *   LLAMINAR_STREAMK_AB_PREFILL_M - Comma-separated M values (default: 64,128,256,512,596)
 *   LLAMINAR_STREAMK_AB_WARMUP    - Warmup runs (default: 5)
 *   LLAMINAR_STREAMK_AB_BENCH     - Benchmark runs (default: 20)
 *   LLAMINAR_STREAMK_AB_CSV       - CSV output path (default: stdout only)
 */

#include <gtest/gtest.h>

#ifdef HAVE_CUDA

#include <cuda_runtime.h>

#include "CUDANativeVNNIGemmPerfCommon.h"
#include "fort.hpp"

using namespace llaminar2::test::native_vnni_gemm_perf;

extern "C"
{
    void cudaNativeVNNIPrefill_setStreamKMode(int mode);
    int cudaNativeVNNIPrefill_getStreamKMode();
    void cudaNativeVNNIPrefill_freeStreamKFixup();
}

namespace
{
    struct ABConfig
    {
        int warmup_runs = 5;
        int bench_runs = 20;
        std::vector<int> prefill_m = {64, 128, 256, 512, 596};
        std::set<std::string> shape_filters;
        std::string csv_path;
    };

    ABConfig loadABConfig()
    {
        ABConfig cfg;

        if (const auto v = getEnvInt("LLAMINAR_STREAMK_AB_WARMUP"))
            cfg.warmup_runs = std::max(1, *v);
        if (const auto v = getEnvInt("LLAMINAR_STREAMK_AB_BENCH"))
            cfg.bench_runs = std::max(1, *v);

        const auto m_vals = getEnvCsvInts("LLAMINAR_STREAMK_AB_PREFILL_M");
        if (!m_vals.empty())
            cfg.prefill_m = m_vals;

        cfg.shape_filters = getEnvCsvSet("LLAMINAR_STREAMK_AB_SHAPES");

        const std::string csv = getEnvString("LLAMINAR_STREAMK_AB_CSV");
        if (!csv.empty())
            cfg.csv_path = csv;

        return cfg;
    }

    // 7B model shapes that matter for prefill throughput
    const std::vector<Shape> kStreamKShapes = {
        {"7B_QKV", 4608, 3584},       // attn Q+K+V projection
        {"7B_Wo", 3584, 3584},        // attn output projection
        {"7B_GateUp", 18944, 3584},   // FFN gate+up fused
        {"7B_Down", 3584, 18944},     // FFN down projection
        {"7B_LM_Head", 152064, 3584}, // LM head (vocab projection)
        // Also test TP=2 shapes
        {"7B_TP2_QKV", 2304, 3584},
        {"7B_TP2_Wo", 3584, 1792},
        {"7B_TP2_GateUp", 9472, 3584},
        {"7B_TP2_Down", 3584, 9472},
    };

    struct ABResult
    {
        std::string shape_name;
        int m, n, k;
        int tiles;      // standard tile count
        int sm_slots;   // nsm * max_blocks_per_sm
        float wave_eff; // tiles / ceil(tiles/slots) / slots
        double std_min_us;
        double std_mean_us;
        double sk_min_us;
        double sk_mean_us;
        double speedup; // std_min / sk_min (>1 = stream-K wins)
    };

    class CUDAStreamKABPerf : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            int device_count = 0;
            if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
                GTEST_SKIP() << "No CUDA devices available";
            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

            // Save original stream-K mode
            original_mode_ = cudaNativeVNNIPrefill_getStreamKMode();
        }

        void TearDown() override
        {
            // Restore original mode
            cudaNativeVNNIPrefill_setStreamKMode(original_mode_);
        }

        int original_mode_ = 0;
    };

    void computeTileStats(int m, int n, int k, int bm, int bn, int &tiles, int &sm_slots, float &wave_eff)
    {
        int nsm = 0;
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
        const int ntx = (n + bn - 1) / bn;
        const int nty = (m + bm - 1) / bm;
        tiles = ntx * nty;
        const int num_warps = (bm >= 128) ? 8 : 4;
        const int max_bps = (num_warps >= 8) ? 2 : 3;
        sm_slots = nsm * max_bps;
        const int full_waves = (tiles + sm_slots - 1) / sm_slots;
        wave_eff = static_cast<float>(tiles) / static_cast<float>(full_waves * sm_slots);
    }

    TEST_F(CUDAStreamKABPerf, StreamK_vs_Standard_AllShapes)
    {
        const ABConfig cfg = loadABConfig();

        // Create Q4_0 format factory
        auto create_q40 = [](size_t n, size_t k)
        { return llaminar2::test::TestTensorFactory::createQ4_0Random({n, k}); };

        std::vector<ABResult> results;

        for (const auto &shape : kStreamKShapes)
        {
            if (!cfg.shape_filters.empty() && !shouldRunName(cfg.shape_filters, shape.name))
                continue;

            for (int m : cfg.prefill_m)
            {
                ABResult r;
                r.shape_name = shape.name;
                r.m = m;
                r.n = shape.n;
                r.k = shape.k;

                // Use T128x128 for stats (dominant tile for large shapes)
                // For shapes that would use T64x128, this is approximate
                int bm = 128, bn = 128;
                if (m < 128)
                {
                    bm = 64;
                    bn = 128;
                }
                computeTileStats(m, shape.n, shape.k, bm, bn, r.tiles, r.sm_slots, r.wave_eff);

                std::fprintf(stderr, "[StreamK-AB] %s M=%d N=%d K=%d tiles=%d slots=%d wave_eff=%.2f\n",
                             shape.name.c_str(), m, shape.n, shape.k, r.tiles, r.sm_slots, r.wave_eff);

                // --- Standard tiling (stream-K OFF) ---
                cudaNativeVNNIPrefill_setStreamKMode(-1);
                {
                    auto weights = create_q40(static_cast<size_t>(shape.n), static_cast<size_t>(shape.k));
                    RunResult rr = runKernel(weights.get(), m, shape.n, shape.k,
                                             RunPath::NativeVNNITensorCore,
                                             cfg.warmup_runs, cfg.bench_runs, 0);
                    r.std_min_us = rr.min_us;
                    r.std_mean_us = rr.mean_us;
                }

                // --- Stream-K (stream-K ON) ---
                cudaNativeVNNIPrefill_setStreamKMode(1);
                {
                    auto weights = create_q40(static_cast<size_t>(shape.n), static_cast<size_t>(shape.k));
                    RunResult rr = runKernel(weights.get(), m, shape.n, shape.k,
                                             RunPath::NativeVNNITensorCore,
                                             cfg.warmup_runs, cfg.bench_runs, 0);
                    r.sk_min_us = rr.min_us;
                    r.sk_mean_us = rr.mean_us;
                }

                r.speedup = (r.sk_min_us > 0.0) ? (r.std_min_us / r.sk_min_us) : 0.0;
                results.push_back(r);
            }
        }

        // Restore auto mode
        cudaNativeVNNIPrefill_setStreamKMode(0);

        // --- Render results table ---
        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Shape" << "M" << "N" << "K" << "Tiles" << "Slots"
              << "Wave%" << "Std (us)" << "SK (us)" << "Speedup" << "Winner"
              << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 10; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        for (const auto &r : results)
        {
            char wave_pct[16], std_us[16], sk_us[16], spd[16];
            std::snprintf(wave_pct, sizeof(wave_pct), "%.1f%%", r.wave_eff * 100.0f);
            std::snprintf(std_us, sizeof(std_us), "%.1f", r.std_min_us);
            std::snprintf(sk_us, sizeof(sk_us), "%.1f", r.sk_min_us);
            std::snprintf(spd, sizeof(spd), "%.3fx", r.speedup);

            const char *winner = (r.speedup > 1.02) ? "SK" : (r.speedup < 0.98) ? "STD"
                                                                                : "TIE";

            table << r.shape_name << r.m << r.n << r.k << r.tiles << r.sm_slots
                  << wave_pct << std_us << sk_us << spd << winner << fort::endr;
        }

        std::fprintf(stderr, "\n%s\n", table.to_string().c_str());

        // --- CSV output ---
        if (!cfg.csv_path.empty())
        {
            FILE *fp = std::fopen(cfg.csv_path.c_str(), "w");
            if (fp)
            {
                std::fprintf(fp, "shape,m,n,k,tiles,slots,wave_eff,std_min_us,std_mean_us,sk_min_us,sk_mean_us,speedup,winner\n");
                for (const auto &r : results)
                {
                    const char *winner = (r.speedup > 1.02) ? "SK" : (r.speedup < 0.98) ? "STD"
                                                                                        : "TIE";
                    std::fprintf(fp, "%s,%d,%d,%d,%d,%d,%.4f,%.2f,%.2f,%.2f,%.2f,%.4f,%s\n",
                                 r.shape_name.c_str(), r.m, r.n, r.k, r.tiles, r.sm_slots,
                                 r.wave_eff, r.std_min_us, r.std_mean_us, r.sk_min_us, r.sk_mean_us,
                                 r.speedup, winner);
                }
                std::fclose(fp);
                std::fprintf(stderr, "\n[StreamK-AB] Results written to %s\n", cfg.csv_path.c_str());
            }
        }

        // Free stream-K fixup buffer
        cudaNativeVNNIPrefill_freeStreamKFixup();

        // Summary: count wins
        int sk_wins = 0, std_wins = 0, ties = 0;
        for (const auto &r : results)
        {
            if (r.speedup > 1.02)
                ++sk_wins;
            else if (r.speedup < 0.98)
                ++std_wins;
            else
                ++ties;
        }
        std::fprintf(stderr, "\n[StreamK-AB] Summary: Stream-K wins=%d, Standard wins=%d, Ties=%d\n",
                     sk_wins, std_wins, ties);
    }
}

#else
// No CUDA
TEST(CUDAStreamKABPerf, StreamK_vs_Standard_AllShapes)
{
    GTEST_SKIP() << "CUDA not available";
}
#endif
