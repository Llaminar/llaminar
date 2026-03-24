/**
 * @file Perf__CPUNativeVNNI_GEMV.cpp
 * @brief Comprehensive GEMV decode performance sweep across all Qwen model
 *        sizes (0.5B through 32B), TP degrees (1/2/4), and both quantized
 *        GEMV paths: packed VNNI and native Q8_0.
 *
 * For each (model, shape, TP degree):
 *   1. Report tile config (nbc, k_tiles, category)
 *   2. Benchmark packed VNNI GEMV (FP32→Q8_1 quantize + INT8 packed access)
 *   3. Benchmark native Q8_0 GEMV (direct block access, FP32 dequant + FMA)
 *   4. Report latency, effective BW, roofline %, and speedup
 *
 * NOTE: The VNNI column calls gemv_native_vnni() directly (which includes
 * activation quantization + packed INT8 VNNI compute). This gives a real
 * comparison vs native Q8_0, rather than going through multiply_tensor()
 * which short-circuits to the native Q8_0 path for codebook_id==19.
 *
 * Also decomposes per-GEMV overhead:
 *   - Activation quantization (FP32→Q8_1 per call)
 *   - Kernel compute time
 *   - Total including overhead
 *
 * System roofline: 2× Intel Xeon Gold 6238R, 6-channel DDR4-2933 per socket.
 * Single-socket (28 cores, numactl node 0): ~60 GB/s STREAM read
 * Dual-socket (56 cores): ~117 GB/s STREAM read
 *
 * @note Run with Release build:
 *   ctest -R V2_Perf_CPUNativeVNNI_GEMV --verbose
 *   Individual tests: --gtest_filter="*Q8_0_AllModels*"
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
#include "utils/Logger.h"
#include "fort.hpp"

#include "utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::cpu::native_vnni;
using namespace llaminar2::test;

namespace
{

    // =========================================================================
    // MPI environment
    // =========================================================================
    void mpi_abort_handler(int sig)
    {
        const char *msg = "\n[FATAL] Signal in GEMV perf test — MPI_Abort\n";
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
            std::signal(SIGSEGV, mpi_abort_handler);
            std::signal(SIGABRT, mpi_abort_handler);
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

    // =========================================================================
    // System roofline
    // =========================================================================
    // Single-socket STREAM read (28 cores, DDR4-2933, 6ch): ~60 GB/s
    // Dual-socket STREAM read (56 cores): ~117 GB/s
    static constexpr double SINGLE_SOCKET_BW_GBS = 60.0;
    static constexpr double DUAL_SOCKET_BW_GBS = 117.0;

    // Auto-detect: use thread count to pick roofline
    static double systemBandwidthGB()
    {
        int threads = omp_get_max_threads();
        return (threads > 32) ? DUAL_SOCKET_BW_GBS : SINGLE_SOCKET_BW_GBS;
    }

    // =========================================================================
    // Model definitions — Qwen2.5 family (0.5B through 32B)
    // =========================================================================
    struct ModelConfig
    {
        std::string name;
        int d_model;
        int n_heads;
        int n_kv_heads;
        int head_dim;
        int d_ff;
        int vocab; // vocabulary size for LM_Head
    };

    static const std::vector<ModelConfig> ALL_MODELS = {
        {"0.5B", 896, 14, 2, 64, 4864, 151936},
        {"1.5B", 1536, 12, 2, 128, 8960, 151936},
        {"3B", 2048, 16, 2, 128, 11008, 151936},
        {"7B", 3584, 28, 4, 128, 18944, 152064},
        {"14B", 5120, 40, 8, 128, 13824, 152064},
        {"32B", 5120, 40, 8, 128, 27648, 152064},
    };

    // =========================================================================
    // Shape definitions
    // =========================================================================
    struct GEMVShape
    {
        std::string name;
        std::string category; // "Attn", "FFN", "LM_Head"
        std::string shard;    // "ColPar", "RowPar", "Replicate"
        int N;
        int K;
    };

    static std::vector<GEMVShape> buildShapes(const ModelConfig &m, int tp = 1)
    {
        int n_q = m.n_heads * m.head_dim;
        int n_kv = m.n_kv_heads * m.head_dim;
        std::string suffix = (tp > 1) ? "_TP" + std::to_string(tp) : "";

        // Megatron-style TP sharding:
        //   Column-parallel (QKV, FFN Gate/Up): split N (output)
        //   Row-parallel (Wo, FFN Down):        split K (input)
        //   LM_Head: Column-parallel (split vocab)
        return {
            {m.name + "_Q_proj" + suffix, "Attn", "ColPar", n_q / tp, m.d_model},
            {m.name + "_K_proj" + suffix, "Attn", "ColPar", n_kv / tp, m.d_model},
            {m.name + "_V_proj" + suffix, "Attn", "ColPar", n_kv / tp, m.d_model},
            {m.name + "_Wo_proj" + suffix, "Attn", "RowPar", m.d_model, n_q / tp},
            {m.name + "_FFN_Gate" + suffix, "FFN", "ColPar", m.d_ff / tp, m.d_model},
            {m.name + "_FFN_Up" + suffix, "FFN", "ColPar", m.d_ff / tp, m.d_model},
            {m.name + "_FFN_Down" + suffix, "FFN", "RowPar", m.d_model, m.d_ff / tp},
            {m.name + "_LM_Head" + suffix, "LM_Head", "ColPar", m.vocab / tp, m.d_model},
        };
    }

    // =========================================================================
    // Benchmark infrastructure
    // =========================================================================
    static constexpr int WARMUP = 30;
    static constexpr int ITERS = 100;

    struct GEMVResult
    {
        std::string shape_name;
        std::string category;
        std::string shard;
        int N, K;
        // Tile config
        int nbc;              // n_block_chunks
        int k_tiles;          // k-parallel tiles (0=N-parallel only)
        std::string tile_cat; // ShapeCategory name
        // Packed VNNI GEMV (current production: quantize + packed access)
        double packed_us;
        double packed_bw_gbs;
        // Native Q8_0 GEMV (new: direct block access, FP32 compute)
        double native_us;
        double native_bw_gbs;
        // Roofline and comparison
        double roofline_bw;
        double packed_roof_pct;
        double native_roof_pct;
        double speedup; // packed_us / native_us
        // Weight data sizes
        double packed_bytes; // VNNI packed buffer
        double native_bytes; // Q8_0 native blocks
    };

    /**
     * @brief Benchmark packed VNNI GEMV directly, return p10 latency (us).
     *
     * Calls gemv_native_vnni() directly to measure the packed VNNI path
     * (FP32→Q8_1 quantization + INT8 packed VNNI compute). This bypasses
     * multiply_tensor's Q8_0 native shortcut, giving a real VNNI-vs-native
     * comparison for Q8_0 weights.
     */
    double benchPackedVNNI(const CPUNativeVNNIPackedWeights &packed,
                           const float *A, float *C, int N)
    {
        for (int i = 0; i < WARMUP; ++i)
            gemv_native_vnni(packed, A, C);

        std::vector<double> times(ITERS);
        for (int i = 0; i < ITERS; ++i)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            gemv_native_vnni(packed, A, C);
            auto t1 = std::chrono::high_resolution_clock::now();
            times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
        }
        std::sort(times.begin(), times.end());
        return times[std::max(0, (int)(ITERS * 0.1) - 1)];
    }

    /**
     * @brief Benchmark native Q8_0 GEMV via unified dispatch, return p10 latency (us).
     *
     * Calls gemv_native_vnni() with Q8_0 blocks pointer, which dispatches to
     * the native Q8_0 path (FP32 dequant + FMA) instead of the packed VNNI path.
     */
    double benchNativeQ8_0(const CPUNativeVNNIPackedWeights &packed,
                           const Q8_0Block *blocks, const float *A,
                           float *C, int bpr)
    {
        for (int i = 0; i < WARMUP; ++i)
            gemv_native_vnni(packed, A, C, blocks, bpr);

        std::vector<double> times(ITERS);
        for (int i = 0; i < ITERS; ++i)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            gemv_native_vnni(packed, A, C, blocks, bpr);
            auto t1 = std::chrono::high_resolution_clock::now();
            times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
        }
        std::sort(times.begin(), times.end());
        return times[std::max(0, (int)(ITERS * 0.1) - 1)];
    }

    static const char *categoryName(ShapeCategory cat)
    {
        switch (cat)
        {
        case ShapeCategory::ATTENTION:
            return "ATTN";
        case ShapeCategory::FFN:
            return "FFN";
        case ShapeCategory::LM_HEAD:
            return "LM";
        default:
            return "GEN";
        }
    }

    /**
     * @brief Benchmark one shape and return result.
     */
    GEMVResult benchShape(const GEMVShape &shape, double roofline_bw)
    {
        GEMVResult r{};
        r.shape_name = shape.name;
        r.category = shape.category;
        r.shard = shape.shard;
        r.N = shape.N;
        r.K = shape.K;
        r.roofline_bw = roofline_bw;

        // Create Q8_0 weights
        auto weights = TestTensorFactory::createQ8_0Random(
            {(size_t)shape.N, (size_t)shape.K});
        if (!weights)
            return r;

        auto *q8_tensor = dynamic_cast<Q8_0Tensor *>(weights.get());
        if (!q8_tensor)
            return r;

        // Random activations
        int bpr = (shape.K + 31) / 32;
        std::vector<float> A(shape.K);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : A)
            v = dist(rng);

        // --- Tile config ---
        int payload = 32; // Q8_0 payload bytes per block
        int threads = omp_get_max_threads();
        auto tile = computeTileConfig(shape.N, shape.K, 1, payload, threads);
        r.nbc = tile.n_block_chunks;
        r.k_tiles = tile.k_tiles;
        r.tile_cat = categoryName(tile.category);

        // --- Weight data sizes ---
        int N_chunks = (shape.N + 63) / 64;
        // Packed VNNI: interleaved blocks + scales + comps
        // INT8 pre-decoded: stride=2048 per chunk per K-block
        r.packed_bytes = (double)N_chunks * bpr * (2048 + 256 + 256); // interleaved + scales + comp
        // Native Q8_0: 34 bytes per block
        r.native_bytes = (double)shape.N * bpr * 34.0;

        // --- Packed VNNI GEMV (quantize FP32→Q8_1 + INT8 packed VNNI) ---
        CPUNativeVNNIGemmKernel packed_kernel(weights.get());
        if (packed_kernel.isValid())
        {
            std::vector<float> C_packed(shape.N, 0.0f);
            r.packed_us = benchPackedVNNI(
                packed_kernel.packedWeights(), A.data(), C_packed.data(), shape.N);
            r.packed_bw_gbs = r.packed_bytes / (r.packed_us * 1e-6) / 1e9;
            r.packed_roof_pct = r.packed_bw_gbs / roofline_bw * 100.0;
        }

        // --- Native Q8_0 GEMV (direct block access, FP32 dequant) ---
        {
            std::vector<float> C(shape.N, 0.0f);
            r.native_us = benchNativeQ8_0(
                packed_kernel.packedWeights(), q8_tensor->typed_data(),
                A.data(), C.data(), bpr);
            r.native_bw_gbs = r.native_bytes / (r.native_us * 1e-6) / 1e9;
            r.native_roof_pct = r.native_bw_gbs / roofline_bw * 100.0;
        }

        // Speedup of native over packed
        r.speedup = (r.native_us > 0) ? r.packed_us / r.native_us : 0;

        return r;
    }

    // =========================================================================
    // Table renderers
    // =========================================================================

    void renderGEMVTable(const std::string &title,
                         const std::vector<GEMVResult> &results)
    {
        double roofline = results.empty() ? 0 : results[0].roofline_bw;

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Shape" << "Cat" << "Shard" << "N" << "K"
              << "Tile" << "nbc" << "kt"
              << "VNNI \xc2\xb5s" << "VNNI BW" << "VNNI R%"
              << "Q8_0 \xc2\xb5s" << "Q8_0 BW" << "Q8_0 R%"
              << "Speedup"
              << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        table.column(1).set_cell_text_align(fort::text_align::left);
        table.column(2).set_cell_text_align(fort::text_align::left);
        table.column(5).set_cell_text_align(fort::text_align::left);
        for (int c : {3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14})
            table.column(c).set_cell_text_align(fort::text_align::right);

        double sum_packed = 0, sum_native = 0;
        double sum_packed_bytes = 0, sum_native_bytes = 0;
        std::string last_cat;

        for (const auto &r : results)
        {
            if (!last_cat.empty() && r.category != last_cat)
                table << fort::separator;
            last_cat = r.category;

            char pu[16], pbw[16], pr[16], nu[16], nbw[16], nr[16], sp[16];
            std::snprintf(pu, sizeof(pu), "%.0f", r.packed_us);
            std::snprintf(pbw, sizeof(pbw), "%.1f", r.packed_bw_gbs);
            std::snprintf(pr, sizeof(pr), "%.0f%%", r.packed_roof_pct);
            std::snprintf(nu, sizeof(nu), "%.0f", r.native_us);
            std::snprintf(nbw, sizeof(nbw), "%.1f", r.native_bw_gbs);
            std::snprintf(nr, sizeof(nr), "%.0f%%", r.native_roof_pct);
            std::snprintf(sp, sizeof(sp), "%.2fx", r.speedup);

            table << r.shape_name << r.category << r.shard
                  << r.N << r.K
                  << r.tile_cat << r.nbc << r.k_tiles
                  << pu << pbw << pr
                  << nu << nbw << nr
                  << sp
                  << fort::endr;

            sum_packed += r.packed_us;
            sum_native += r.native_us;
            sum_packed_bytes += r.packed_bytes;
            sum_native_bytes += r.native_bytes;
        }

        // Summary row
        table << fort::separator;
        double avg_packed_bw = sum_packed_bytes / (sum_packed * 1e-6) / 1e9;
        double avg_native_bw = sum_native_bytes / (sum_native * 1e-6) / 1e9;
        char tpu[16], tpbw[16], tpr[16], tnu[16], tnbw[16], tnr[16], tsp[16];
        std::snprintf(tpu, sizeof(tpu), "%.0f", sum_packed);
        std::snprintf(tpbw, sizeof(tpbw), "%.1f", avg_packed_bw);
        std::snprintf(tpr, sizeof(tpr), "%.0f%%", avg_packed_bw / roofline * 100);
        std::snprintf(tnu, sizeof(tnu), "%.0f", sum_native);
        std::snprintf(tnbw, sizeof(tnbw), "%.1f", avg_native_bw);
        std::snprintf(tnr, sizeof(tnr), "%.0f%%", avg_native_bw / roofline * 100);
        std::snprintf(tsp, sizeof(tsp), "%.2fx", sum_packed / sum_native);

        table << "TOTAL" << "" << "" << "" << "" << "" << "" << ""
              << tpu << tpbw << tpr
              << tnu << tnbw << tnr
              << tsp
              << fort::endr;

        std::cout << "\n"
                  << title << "\n"
                  << "Threads: " << omp_get_max_threads()
                  << "  |  Roofline: " << roofline << " GB/s"
                  << "  |  VNNI = packed INT8 path (FP32\xe2\x86\x92Q8_1 quant + VNNI compute)"
                  << "  |  Q8_0 = native blocks (FP32 dequant + FMA)\n\n"
                  << table.to_string() << std::endl;
    }

    /**
     * @brief Render per-model decode latency summary (simulated full-layer token time).
     */
    void renderDecodeProjection(const std::string &title,
                                const std::vector<std::pair<std::string, std::vector<GEMVResult>>> &model_results)
    {
        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Model" << "Layers"
              << "VNNI ms/tok" << "VNNI tok/s"
              << "Q8_0 ms/tok" << "Q8_0 tok/s"
              << "Speedup"
              << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 6; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        // Approximate layer counts
        auto layerCount = [](const std::string &name) -> int
        {
            if (name == "0.5B")
                return 24;
            if (name == "1.5B")
                return 28;
            if (name == "3B")
                return 36;
            if (name == "7B")
                return 28;
            if (name == "14B")
                return 48;
            if (name == "32B")
                return 64;
            return 28;
        };

        for (auto &[model_name, results] : model_results)
        {
            int layers = layerCount(model_name);
            // Sum per-layer GEMV time (all shapes except LM_Head)
            double layer_packed_us = 0, layer_native_us = 0;
            double lm_packed_us = 0, lm_native_us = 0;

            for (const auto &r : results)
            {
                if (r.category == "LM_Head")
                {
                    lm_packed_us += r.packed_us;
                    lm_native_us += r.native_us;
                }
                else
                {
                    layer_packed_us += r.packed_us;
                    layer_native_us += r.native_us;
                }
            }

            double packed_ms = (layer_packed_us * layers + lm_packed_us) / 1000.0;
            double native_ms = (layer_native_us * layers + lm_native_us) / 1000.0;

            char pm[16], pt[16], nm[16], nt[16], sp[16];
            std::snprintf(pm, sizeof(pm), "%.1f", packed_ms);
            std::snprintf(pt, sizeof(pt), "%.1f", 1000.0 / packed_ms);
            std::snprintf(nm, sizeof(nm), "%.1f", native_ms);
            std::snprintf(nt, sizeof(nt), "%.1f", 1000.0 / native_ms);
            std::snprintf(sp, sizeof(sp), "%.2fx", packed_ms / native_ms);

            table << model_name << layers
                  << pm << pt
                  << nm << nt
                  << sp
                  << fort::endr;
        }

        std::cout << "\n"
                  << title << "\n"
                  << "Projection: (per-layer GEMV total × layers) + LM_Head\n"
                  << "GEMV-only time (excludes attention, norms, sampling)\n\n"
                  << table.to_string() << std::endl;
    }

    /**
     * @brief Render TP scaling comparison table.
     */
    void renderTPScaling(const std::string &title,
                         const std::vector<GEMVResult> &tp1,
                         const std::vector<GEMVResult> &tp2,
                         const std::vector<GEMVResult> &tp4)
    {
        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Layer" << "Shard"
              << "TP1 N\xc3\x97K" << "Q8_0 \xc2\xb5s"
              << "TP2 N\xc3\x97K" << "Q8_0 \xc2\xb5s" << "TP Eff"
              << "TP4 N\xc3\x97K" << "Q8_0 \xc2\xb5s" << "TP Eff"
              << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        table.column(1).set_cell_text_align(fort::text_align::left);
        for (int c : {2, 3, 4, 5, 6, 7, 8, 9})
            table.column(c).set_cell_text_align(fort::text_align::right);

        size_t n = std::min({tp1.size(), tp2.size(), tp4.size()});
        for (size_t i = 0; i < n; ++i)
        {
            double eff2 = (tp1[i].native_us / 2.0) / tp2[i].native_us;
            double eff4 = (tp1[i].native_us / 4.0) / tp4[i].native_us;

            char nk1[24], u1[16], nk2[24], u2[16], e2[16], nk4[24], u4[16], e4[16];
            std::snprintf(nk1, sizeof(nk1), "%d\xc3\x97%d", tp1[i].N, tp1[i].K);
            std::snprintf(u1, sizeof(u1), "%.0f", tp1[i].native_us);
            std::snprintf(nk2, sizeof(nk2), "%d\xc3\x97%d", tp2[i].N, tp2[i].K);
            std::snprintf(u2, sizeof(u2), "%.0f", tp2[i].native_us);
            std::snprintf(e2, sizeof(e2), "%.0f%%", eff2 * 100);
            std::snprintf(nk4, sizeof(nk4), "%d\xc3\x97%d", tp4[i].N, tp4[i].K);
            std::snprintf(u4, sizeof(u4), "%.0f", tp4[i].native_us);
            std::snprintf(e4, sizeof(e4), "%.0f%%", eff4 * 100);

            // Extract short layer name from shape name (after model prefix)
            std::string layer = tp1[i].shape_name;
            auto pos = layer.find('_');
            if (pos != std::string::npos)
                layer = layer.substr(pos + 1);

            table << layer << tp1[i].shard
                  << nk1 << u1
                  << nk2 << u2 << e2
                  << nk4 << u4 << e4
                  << fort::endr;
        }

        std::cout << "\n"
                  << title << "\n"
                  << "TP Eff = (TP1_time / TP_degree) / actual_time\n"
                  << "Native Q8_0 GEMV only\n\n"
                  << table.to_string() << std::endl;
    }

    // =========================================================================
    // Test fixture
    // =========================================================================
    class CPUNativeVNNIGemvTest : public ::testing::Test
    {
    };

    // =========================================================================
    // TEST 1: Q8_0 All Models — Full shape sweep (TP=1)
    //
    // Comprehensive decode GEMV benchmark across all Qwen model sizes.
    // Compares packed VNNI (production) vs native Q8_0 for each shape.
    // =========================================================================
    TEST_F(CPUNativeVNNIGemvTest, Q8_0_AllModels)
    {
        double roofline = systemBandwidthGB();
        std::vector<std::pair<std::string, std::vector<GEMVResult>>> all_models;

        for (const auto &model : ALL_MODELS)
        {
            auto shapes = buildShapes(model);
            std::vector<GEMVResult> results;
            for (const auto &shape : shapes)
                results.push_back(benchShape(shape, roofline));

            renderGEMVTable(
                "=== Q8_0 GEMV Decode: Qwen " + model.name + " (TP=1) ===",
                results);
            all_models.push_back({model.name, results});
        }

        renderDecodeProjection(
            "=== Decode Latency Projection — GEMV Only (Q8_0) ===",
            all_models);
    }

    // =========================================================================
    // TEST 2: Q8_0 7B TP Scaling — TP=1/2/4
    //
    // Primary benchmark for the user's current model.
    // =========================================================================
    TEST_F(CPUNativeVNNIGemvTest, Q8_0_7B_TP_Scaling)
    {
        double roofline = systemBandwidthGB();
        const auto &model = ALL_MODELS[3]; // 7B

        auto shapes_tp1 = buildShapes(model, 1);
        auto shapes_tp2 = buildShapes(model, 2);
        auto shapes_tp4 = buildShapes(model, 4);

        std::vector<GEMVResult> res1, res2, res4;
        for (const auto &s : shapes_tp1)
            res1.push_back(benchShape(s, roofline));
        for (const auto &s : shapes_tp2)
            res2.push_back(benchShape(s, roofline));
        for (const auto &s : shapes_tp4)
            res4.push_back(benchShape(s, roofline));

        renderGEMVTable("=== Q8_0 GEMV Decode: Qwen 7B — TP=1 ===", res1);
        renderGEMVTable("=== Q8_0 GEMV Decode: Qwen 7B — TP=2 ===", res2);
        renderGEMVTable("=== Q8_0 GEMV Decode: Qwen 7B — TP=4 ===", res4);

        renderTPScaling(
            "=== Q8_0 GEMV TP Scaling: Qwen 7B (TP=1/2/4) ===",
            res1, res2, res4);
    }

    // =========================================================================
    // TEST 3: All Models TP Scaling — TP=1/2/4
    //
    // Comprehensive TP scaling across all model sizes.
    // =========================================================================
    TEST_F(CPUNativeVNNIGemvTest, Q8_0_AllModels_TP_Scaling)
    {
        double roofline = systemBandwidthGB();

        for (const auto &model : ALL_MODELS)
        {
            // Skip TP shapes that would produce N<32 or K<32
            // (0.5B K_proj has n_kv=2, head=64 → KV_N=128; TP=4 → 32)
            int min_kv = model.n_kv_heads * model.head_dim;
            bool skip_tp4 = (min_kv / 4 < 32);

            auto s1 = buildShapes(model, 1);
            auto s2 = buildShapes(model, 2);

            std::vector<GEMVResult> r1, r2;
            for (const auto &s : s1)
                r1.push_back(benchShape(s, roofline));
            for (const auto &s : s2)
                r2.push_back(benchShape(s, roofline));

            if (!skip_tp4)
            {
                auto s4 = buildShapes(model, 4);
                std::vector<GEMVResult> r4;
                for (const auto &s : s4)
                    r4.push_back(benchShape(s, roofline));

                renderTPScaling(
                    "=== Q8_0 GEMV TP Scaling: Qwen " + model.name + " ===",
                    r1, r2, r4);
            }
            else
            {
                // TP=2 only
                renderGEMVTable("=== Q8_0 GEMV: Qwen " + model.name + " TP=1 ===", r1);
                renderGEMVTable("=== Q8_0 GEMV: Qwen " + model.name + " TP=2 ===", r2);
            }
        }
    }

    // =========================================================================
    // TEST 4: Tile Config Inspection — All shapes, all models
    //
    // No benchmarking — just shows what tile config the heuristic picks
    // for each shape at each TP degree. Quick diagnostic test.
    // =========================================================================
    TEST_F(CPUNativeVNNIGemvTest, TileConfig_AllShapes)
    {
        int threads = omp_get_max_threads();
        int payload = 32; // Q8_0

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Model" << "Shape" << "TP" << "N" << "K"
              << "Category" << "nbc" << "k_tiles" << "N_chunks" << "Tasks"
              << "Wt MB"
              << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        table.column(1).set_cell_text_align(fort::text_align::left);
        table.column(5).set_cell_text_align(fort::text_align::left);
        for (int c : {2, 3, 4, 6, 7, 8, 9, 10})
            table.column(c).set_cell_text_align(fort::text_align::right);

        for (const auto &model : ALL_MODELS)
        {
            for (int tp : {1, 2, 4})
            {
                if (tp > 1)
                {
                    int min_kv = model.n_kv_heads * model.head_dim;
                    if (min_kv / tp < 32)
                        continue;
                }

                auto shapes = buildShapes(model, tp);
                for (const auto &s : shapes)
                {
                    auto cfg = computeTileConfig(s.N, s.K, 1, payload, threads);
                    int N_chunks = (s.N + 63) / 64;
                    int n_tasks = (N_chunks + cfg.n_block_chunks - 1) / cfg.n_block_chunks;
                    int k_tasks = (cfg.k_tiles > 0) ? cfg.k_tiles : 1;
                    int total_tasks = n_tasks * k_tasks;

                    // Weight size in MB (Q8_0 native: 34 bytes/block, bpr blocks/row)
                    int bpr = (s.K + 31) / 32;
                    double wt_mb = (double)s.N * bpr * 34.0 / 1e6;

                    char wt_buf[16];
                    std::snprintf(wt_buf, sizeof(wt_buf), "%.1f", wt_mb);

                    // Short shape name
                    std::string sname = s.name;
                    auto pos = sname.find('_');
                    if (pos != std::string::npos)
                        sname = sname.substr(pos + 1);

                    table << model.name << sname << tp
                          << s.N << s.K
                          << categoryName(cfg.category) << cfg.n_block_chunks
                          << cfg.k_tiles << N_chunks << total_tasks
                          << wt_buf
                          << fort::endr;
                }
                table << fort::separator;
            }
        }

        std::cout << "\n=== Tile Config Inspection (Q8_0, M=1) ===\n"
                  << "Threads: " << threads << "\n\n"
                  << table.to_string() << std::endl;
    }

    // =========================================================================
    // TEST 5: Quantization Overhead Isolation — 7B shapes
    //
    // Measures the FP32→Q8_1 activation quantization cost separately from
    // the GEMV compute. Helps quantify what % of packed GEMV time is
    // spent on quantization vs actual matrix-vector multiply.
    // =========================================================================
    TEST_F(CPUNativeVNNIGemvTest, QuantOverhead_7B)
    {
        const auto &model = ALL_MODELS[3]; // 7B
        auto shapes = buildShapes(model, 1);
        double roofline = systemBandwidthGB();

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Shape" << "K" << "K_blocks"
              << "Quant \xc2\xb5s" << "VNNI \xc2\xb5s" << "Q8_0 \xc2\xb5s"
              << "Quant %" << "Speedup"
              << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 7; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        for (const auto &shape : shapes)
        {
            int K = shape.K;
            int K_blocks = (K + 31) / 32;

            // Random activations
            std::vector<float> A(K);
            std::mt19937 rng(42);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (auto &v : A)
                v = dist(rng);

            // Benchmark quantization alone
            std::vector<Q8_1Block> q8_buf(K_blocks);
            for (int i = 0; i < WARMUP; ++i)
            {
                for (int kb = 0; kb < K_blocks; ++kb)
                {
                    int off = kb * 32;
                    int len = std::min(32, K - off);
                    simd::quantize_single_block(A.data() + off, q8_buf[kb], len);
                }
            }

            std::vector<double> quant_times(ITERS);
            for (int i = 0; i < ITERS; ++i)
            {
                auto t0 = std::chrono::high_resolution_clock::now();
                int kb = 0;
#if defined(__AVX512F__)
                bool aligned = (K % 32 == 0);
                if (aligned)
                {
                    for (; kb + 1 < K_blocks; kb += 2)
                        simd::quantize_two_blocks_avx512(
                            A.data() + kb * 32, q8_buf[kb], q8_buf[kb + 1]);
                }
#endif
                for (; kb < K_blocks; ++kb)
                {
                    int off = kb * 32;
                    int len = std::min(32, K - off);
                    simd::quantize_single_block(A.data() + off, q8_buf[kb], len);
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                quant_times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
            }
            std::sort(quant_times.begin(), quant_times.end());
            double quant_us = quant_times[std::max(0, (int)(ITERS * 0.1) - 1)];

            // Full packed GEMV and native GEMV
            auto r = benchShape(shape, roofline);

            double quant_pct = (r.packed_us > 0) ? (quant_us / r.packed_us) * 100.0 : 0;

            char qu[16], pu[16], nu[16], qp[16], sp[16];
            std::snprintf(qu, sizeof(qu), "%.1f", quant_us);
            std::snprintf(pu, sizeof(pu), "%.0f", r.packed_us);
            std::snprintf(nu, sizeof(nu), "%.0f", r.native_us);
            std::snprintf(qp, sizeof(qp), "%.1f%%", quant_pct);
            std::snprintf(sp, sizeof(sp), "%.2fx", r.speedup);

            std::string sname = shape.name;
            auto pos = sname.find('_');
            if (pos != std::string::npos)
                sname = sname.substr(pos + 1);

            table << sname << K << K_blocks
                  << qu << pu << nu
                  << qp << sp
                  << fort::endr;
        }

        std::cout << "\n=== Quantization Overhead Isolation: Qwen 7B ===\n"
                  << "Quant = FP32\xe2\x86\x92Q8_1 activation quantization only\n"
                  << "VNNI = packed INT8 path (FP32\xe2\x86\x92Q8_1 quant + VNNI compute)\n"
                  << "Q8_0 = native blocks (FP32 dequant + FMA)\n\n"
                  << table.to_string() << std::endl;
    }

} // anonymous namespace
