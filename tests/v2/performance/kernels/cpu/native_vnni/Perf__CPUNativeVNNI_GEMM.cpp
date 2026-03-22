/**
 * @file Perf__CPUNativeVNNI_GEMM.cpp
 * @brief Performance benchmark: NativeVNNI blocked GEMM vs oneDNN INT8 matmul.
 *
 * Compares the K-tiled 2-row microkernel GEMM against oneDNN's optimized
 * INT8 matmul across multiple M values, Qwen model shapes, and all 18
 * quantized formats.
 *
 * @note Requires libdnnl (apt install libdnnl-dev libdnnl3).
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

#include <oneapi/dnnl/dnnl.hpp>

#include "kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"
#include "kernels/cpu/native_vnni/CPUNativeVNNIGemmKernel.h"
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
    // MPI global environment: init once, finalize on exit, abort on crash
    // =========================================================================
    void mpi_abort_signal_handler(int sig)
    {
        const char *msg = "\n[FATAL] Signal caught in perf test — calling MPI_Abort\n";
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

    // =========================================================================
    // Constants
    // =========================================================================
    static constexpr int WARMUP_ITERS = 20;
    static constexpr int BENCH_ITERS = 100;

    // =========================================================================
    // Shape definitions
    // =========================================================================

    struct GEMMShape
    {
        std::string name;
        int M, N, K;
    };

    struct ModelConfig
    {
        std::string name;
        int d_model;
        int n_heads;
        int n_kv_heads;
        int head_dim;
        int d_ff;
    };

    static const ModelConfig kQwen05B = {"0.5B", 896, 14, 2, 64, 4864};
    static const ModelConfig kQwen3B = {"3B", 2048, 16, 2, 128, 11008};
    static const ModelConfig kQwen7B = {"7B", 3584, 28, 4, 128, 18944};

    static std::vector<GEMMShape> buildGemmShapes(const ModelConfig &m, int M)
    {
        int n_q = m.n_heads * m.head_dim;
        int n_kv = m.n_kv_heads * m.head_dim;
        return {
            {m.name + "_Q_proj", M, n_q, m.d_model},
            {m.name + "_K_proj", M, n_kv, m.d_model},
            {m.name + "_V_proj", M, n_kv, m.d_model},
            {m.name + "_Wo_proj", M, m.d_model, n_q},
            {m.name + "_FFN_Gate", M, m.d_ff, m.d_model},
            {m.name + "_FFN_Up", M, m.d_ff, m.d_model},
            {m.name + "_FFN_Down", M, m.d_model, m.d_ff},
        };
    }

    // =========================================================================
    // Tensor Parallel shape builders
    // =========================================================================
    // Megatron-style TP sharding:
    //   Column-parallel (QKV, FFN Gate/Up): split N (output) dimension
    //   Row-parallel (Wo, FFN Down):        split K (input) dimension

    static std::vector<GEMMShape> buildTPGemmShapes(const ModelConfig &m, int M, int tp)
    {
        int n_q = m.n_heads * m.head_dim;
        int n_kv = m.n_kv_heads * m.head_dim;
        std::string s = "_TP" + std::to_string(tp);
        return {
            {m.name + "_Q_proj" + s, M, n_q / tp, m.d_model},      // ColPar: split N
            {m.name + "_K_proj" + s, M, n_kv / tp, m.d_model},     // ColPar: split N
            {m.name + "_V_proj" + s, M, n_kv / tp, m.d_model},     // ColPar: split N
            {m.name + "_Wo_proj" + s, M, m.d_model, n_q / tp},     // RowPar: split K
            {m.name + "_FFN_Gate" + s, M, m.d_ff / tp, m.d_model}, // ColPar: split N
            {m.name + "_FFN_Up" + s, M, m.d_ff / tp, m.d_model},   // ColPar: split N
            {m.name + "_FFN_Down" + s, M, m.d_model, m.d_ff / tp}, // RowPar: split K
        };
    }

    static constexpr const char *TP_SHARD_MODES[] = {
        "ColPar", "ColPar", "ColPar", "RowPar",
        "ColPar", "ColPar", "RowPar"};

    static constexpr const char *TP_LAYER_NAMES[] = {
        "Q_proj", "K_proj", "V_proj", "Wo_proj",
        "FFN_Gate", "FFN_Up", "FFN_Down"};

    // =========================================================================
    // Format definitions
    // =========================================================================

    struct FormatSpec
    {
        std::string name;
        bool is_nibble_lut;
    };

    static const std::vector<FormatSpec> ALL_FORMATS = {
        {"Q4_0", true},
        {"Q4_1", true},
        {"IQ4_NL", true},
        {"IQ4_XS", true},
        {"Q5_0", false},
        {"Q5_1", false},
        {"Q6_K", false},
        {"Q3_K", false},
        {"Q2_K", false},
        {"IQ3_S", false},
        {"IQ3_XXS", false},
        {"IQ2_S", false},
        {"IQ2_XS", false},
        {"IQ2_XXS", false},
        {"IQ1_S", false},
        {"IQ1_M", false},
        {"Q8_0", false},
        {"Q8_1", false},
    };

    static const std::vector<FormatSpec> KEY_FORMATS = {
        {"Q4_0", true},
        {"IQ4_NL", true},
        {"Q4_1", true},
        {"Q5_0", false},
        {"Q6_K", false},
        {"Q8_0", false},
    };

    // =========================================================================
    // Weight factory
    // =========================================================================

    std::unique_ptr<TensorBase> createWeights(const std::string &fmt, size_t N, size_t K)
    {
        if (fmt == "Q4_0")
            return TestTensorFactory::createQ4_0Random({N, K});
        if (fmt == "IQ4_NL")
            return TestTensorFactory::createIQ4_NLRandom({N, K});
        if (fmt == "Q4_1")
            return TestTensorFactory::createQ4_1Random({N, K});
        if (fmt == "IQ4_XS")
            return TestTensorFactory::createIQ4_XSRandom({N, K});
        if (fmt == "Q5_0")
            return TestTensorFactory::createQ5_0Random({N, K});
        if (fmt == "Q5_1")
            return TestTensorFactory::createQ5_1Random({N, K});
        if (fmt == "Q6_K")
            return TestTensorFactory::createQ6_KRandom({N, K});
        if (fmt == "Q3_K")
            return TestTensorFactory::createQ3_KRandom({N, K});
        if (fmt == "Q2_K")
            return TestTensorFactory::createQ2_KRandom({N, K});
        if (fmt == "IQ3_S")
            return TestTensorFactory::createIQ3_SRandom({N, K});
        if (fmt == "IQ3_XXS")
            return TestTensorFactory::createIQ3_XXSRandom({N, K});
        if (fmt == "IQ2_S")
            return TestTensorFactory::createIQ2_SRandom({N, K});
        if (fmt == "IQ2_XS")
            return TestTensorFactory::createIQ2_XSRandom({N, K});
        if (fmt == "IQ2_XXS")
            return TestTensorFactory::createIQ2_XXSRandom({N, K});
        if (fmt == "IQ1_S")
            return TestTensorFactory::createIQ1_SRandom({N, K});
        if (fmt == "IQ1_M")
            return TestTensorFactory::createIQ1_MRandom({N, K});
        if (fmt == "Q8_0")
            return TestTensorFactory::createQ8_0Random({N, K});
        if (fmt == "Q8_1")
            return TestTensorFactory::createQ8_1Random({N, K});
        return nullptr;
    }

    // =========================================================================
    // oneDNN INT8 matmul engine
    // =========================================================================

    struct OneDNNMatmul
    {
        dnnl::engine eng_;
        dnnl::stream stream_;
        dnnl::matmul prim_;
        dnnl::memory src_mem_;
        dnnl::memory wei_mem_;
        dnnl::memory dst_mem_;
        int M_, N_, K_;

        OneDNNMatmul(int M, int N, int K, const int8_t *weight_data)
            : eng_(dnnl::engine::kind::cpu, 0),
              stream_(eng_),
              M_(M), N_(N), K_(K)
        {
            using namespace dnnl;
            auto src_md = memory::desc({M, K}, memory::data_type::s8, memory::format_tag::ab);
            auto wei_user_md = memory::desc({K, N}, memory::data_type::s8, memory::format_tag::ab);
            auto wei_any_md = memory::desc({K, N}, memory::data_type::s8, memory::format_tag::any);
            auto dst_md = memory::desc({M, N}, memory::data_type::f32, memory::format_tag::ab);

            auto pd = matmul::primitive_desc(eng_, src_md, wei_any_md, dst_md);
            prim_ = matmul(pd);

            src_mem_ = memory(src_md, eng_);
            dst_mem_ = memory(dst_md, eng_);

            // Repack weights
            auto wei_user_mem = memory(wei_user_md, eng_);
            std::memcpy(wei_user_mem.get_data_handle(), weight_data, (size_t)K * N);
            if (pd.weights_desc() != wei_user_md)
            {
                wei_mem_ = memory(pd.weights_desc(), eng_);
                dnnl::reorder(wei_user_mem, wei_mem_).execute(stream_, wei_user_mem, wei_mem_);
                stream_.wait();
            }
            else
            {
                wei_mem_ = wei_user_mem;
            }
        }

        void execute(const int8_t *src, float *dst)
        {
            std::memcpy(src_mem_.get_data_handle(), src, (size_t)M_ * K_);
            std::unordered_map<int, dnnl::memory> args;
            args[DNNL_ARG_SRC] = src_mem_;
            args[DNNL_ARG_WEIGHTS] = wei_mem_;
            args[DNNL_ARG_DST] = dst_mem_;
            prim_.execute(stream_, args);
            stream_.wait();
            std::memcpy(dst, dst_mem_.get_data_handle(), (size_t)M_ * N_ * sizeof(float));
        }
    };

    // =========================================================================
    // Benchmark helpers
    // =========================================================================

    double benchKernel(ITensorGemm *kernel, const float *A, float *C,
                       int M, int N, int K, int warmup, int iters)
    {
        for (int i = 0; i < warmup; ++i)
            kernel->multiply(A, C, M, N, K);

        std::vector<double> times(iters);
        for (int i = 0; i < iters; ++i)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            kernel->multiply(A, C, M, N, K);
            auto t1 = std::chrono::high_resolution_clock::now();
            times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
        }

        std::sort(times.begin(), times.end());
        return times[std::max(0, (int)(iters * 0.1) - 1)]; // p10
    }

    double benchOneDNN(OneDNNMatmul &mm, const int8_t *src, float *dst,
                       int warmup, int iters)
    {
        for (int i = 0; i < warmup; ++i)
            mm.execute(src, dst);

        std::vector<double> times(iters);
        for (int i = 0; i < iters; ++i)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            mm.execute(src, dst);
            auto t1 = std::chrono::high_resolution_clock::now();
            times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
        }

        std::sort(times.begin(), times.end());
        return times[std::max(0, (int)(iters * 0.1) - 1)]; // p10
    }

    // =========================================================================
    // Result struct
    // =========================================================================

    struct GEMMBenchResult
    {
        std::string shape_name;
        std::string format_name;
        int M, N, K;
        double onednn_us;
        double native_us;
        double baseline_us;     // CPUNativeVNNIGemmKernel
        double ratio_vs_dnnl;   // native / onednn (< 1 = we are faster)
        double speedup_vs_base; // baseline / native
        double gflops;          // 2*M*N*K / native_us
        bool is_nibble_lut;
    };

    // =========================================================================
    // Test fixture
    // =========================================================================

    class NativeVNNIGemmPerfTest : public ::testing::Test
    {
    protected:
        void SetUp() override {}

        void benchOneDNNShape(int M, int N, int K, double &out_us)
        {
            std::vector<int8_t> weights(static_cast<size_t>(K) * N);
            std::mt19937 rng(42);
            std::uniform_int_distribution<int> dist(-127, 127);
            for (auto &w : weights)
                w = static_cast<int8_t>(dist(rng));

            OneDNNMatmul mm(M, N, K, weights.data());

            std::vector<int8_t> src(static_cast<size_t>(M) * K);
            for (auto &v : src)
                v = static_cast<int8_t>(dist(rng));
            std::vector<float> dst(static_cast<size_t>(M) * N);

            out_us = benchOneDNN(mm, src.data(), dst.data(), WARMUP_ITERS, BENCH_ITERS);
        }

        bool benchTriple(const FormatSpec &fmt, const GEMMShape &shape,
                         double cached_onednn_us, GEMMBenchResult &result)
        {
            auto weights = createWeights(fmt.name, shape.N, shape.K);
            if (!weights)
                return false;

            CPUNativeVNNIGemmKernel native_kernel(weights.get());
            if (!native_kernel.isValid())
                return false;

            auto baseline_kernel = std::make_unique<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(
                weights.get());

            std::vector<float> A(static_cast<size_t>(shape.M) * shape.K);
            std::mt19937 rng(42);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (auto &v : A)
                v = dist(rng);

            std::vector<float> C_native(static_cast<size_t>(shape.M) * shape.N, 0.0f);
            std::vector<float> C_baseline(static_cast<size_t>(shape.M) * shape.N, 0.0f);

            double native_us = benchKernel(&native_kernel, A.data(), C_native.data(),
                                           shape.M, shape.N, shape.K, WARMUP_ITERS, BENCH_ITERS);
            double baseline_us = benchKernel(baseline_kernel.get(), A.data(), C_baseline.data(),
                                             shape.M, shape.N, shape.K, WARMUP_ITERS, BENCH_ITERS);

            result.shape_name = shape.name;
            result.format_name = fmt.name;
            result.M = shape.M;
            result.N = shape.N;
            result.K = shape.K;
            result.onednn_us = cached_onednn_us;
            result.native_us = native_us;
            result.baseline_us = baseline_us;
            result.ratio_vs_dnnl = native_us / cached_onednn_us;
            result.speedup_vs_base = baseline_us / native_us;
            result.gflops = (2.0 * shape.M * shape.N * shape.K) / (native_us * 1e3);
            result.is_nibble_lut = fmt.is_nibble_lut;

            return true;
        }

        // Rendering
        void renderResults(const std::string &title,
                           const std::vector<GEMMBenchResult> &results)
        {
            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);
            table << fort::header
                  << "Format" << "Path" << "Shape" << "M" << "N" << "K"
                  << "oneDNN us" << "Native us" << "Base us"
                  << "Nat/DNN" << "Spdup" << "GFLOPS"
                  << fort::endr;

            table.column(0).set_cell_text_align(fort::text_align::left);
            table.column(2).set_cell_text_align(fort::text_align::left);
            for (int c = 6; c <= 11; ++c)
                table.column(c).set_cell_text_align(fort::text_align::right);

            std::string last_fmt;
            for (const auto &r : results)
            {
                if (!last_fmt.empty() && r.format_name != last_fmt)
                    table << fort::separator;
                last_fmt = r.format_name;

                char dnn[16], nat[16], base[16], ratio[16], spd[16], gf[16];
                std::snprintf(dnn, sizeof(dnn), "%.1f", r.onednn_us);
                std::snprintf(nat, sizeof(nat), "%.1f", r.native_us);
                std::snprintf(base, sizeof(base), "%.1f", r.baseline_us);
                std::snprintf(ratio, sizeof(ratio), "%.2fx", r.ratio_vs_dnnl);
                std::snprintf(spd, sizeof(spd), "%.2fx", r.speedup_vs_base);
                std::snprintf(gf, sizeof(gf), "%.1f", r.gflops);

                table << r.format_name
                      << (r.is_nibble_lut ? "Nib" : "I8")
                      << r.shape_name << r.M << r.N << r.K
                      << dnn << nat << base
                      << ratio << spd << gf
                      << fort::endr;
            }

            std::cout << "\n"
                      << title << "\n"
                      << "oneDNN " << dnnl_version()->major << "."
                      << dnnl_version()->minor << "." << dnnl_version()->patch
                      << ", OMP threads: " << omp_get_max_threads() << "\n\n"
                      << table.to_string() << std::endl;
        }

        void renderFormatSummary(const std::vector<GEMMBenchResult> &results)
        {
            struct Summary
            {
                std::string name;
                std::string path;
                int count = 0;
                double total_native = 0, total_dnnl = 0, total_baseline = 0;
                double worst_ratio = 0;
                std::string worst_shape;
            };

            std::vector<Summary> summaries;
            std::string last_fmt;
            Summary *cur = nullptr;

            for (const auto &r : results)
            {
                if (r.format_name != last_fmt)
                {
                    summaries.push_back({});
                    cur = &summaries.back();
                    cur->name = r.format_name;
                    cur->path = r.is_nibble_lut ? "NibbleLUT" : "INT8";
                    last_fmt = r.format_name;
                }
                cur->count++;
                cur->total_native += r.native_us;
                cur->total_dnnl += r.onednn_us;
                cur->total_baseline += r.baseline_us;
                if (r.ratio_vs_dnnl > cur->worst_ratio)
                {
                    cur->worst_ratio = r.ratio_vs_dnnl;
                    cur->worst_shape = r.shape_name;
                }
            }

            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);
            table << fort::header
                  << "Format" << "Path" << "Shapes"
                  << "Avg Nat/DNN" << "Worst Nat/DNN" << "Worst Shape"
                  << "Avg Spdup vs Base"
                  << fort::endr;

            table.column(0).set_cell_text_align(fort::text_align::left);
            table.column(1).set_cell_text_align(fort::text_align::left);
            table.column(5).set_cell_text_align(fort::text_align::left);
            for (int c = 3; c <= 6; ++c)
                if (c != 5)
                    table.column(c).set_cell_text_align(fort::text_align::right);

            for (const auto &s : summaries)
            {
                double avg_ratio = s.total_native / s.total_dnnl;
                double avg_speedup = s.total_baseline / s.total_native;
                char avg_r[16], worst_r[16], spd[16];
                std::snprintf(avg_r, sizeof(avg_r), "%.2fx", avg_ratio);
                std::snprintf(worst_r, sizeof(worst_r), "%.2fx", s.worst_ratio);
                std::snprintf(spd, sizeof(spd), "%.2fx", avg_speedup);

                table << s.name << s.path << s.count
                      << avg_r << worst_r << s.worst_shape << spd
                      << fort::endr;
            }

            std::cout << "\n=== Format Summary vs oneDNN (GEMM) ===\n"
                      << "Nat/DNN = NativeVNNI latency / oneDNN latency\n"
                      << "  1.00x = parity with oneDNN\n"
                      << "  >1.0x = we are slower\n"
                      << "  <1.0x = we are faster\n"
                      << table.to_string() << std::endl;
        }

        // =====================================================================
        // TP Scaling renderers
        // =====================================================================

        void renderTPScaling(const std::string &title,
                             const std::vector<GEMMBenchResult> &full,
                             const std::vector<GEMMBenchResult> &tp2,
                             const std::vector<GEMMBenchResult> &tp4)
        {
            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);
            table << fort::header
                  << "Layer" << "Shard" << "TP" << "N" << "K"
                  << "DNN \xc2\xb5s" << "Nat \xc2\xb5s" << "Nat/DNN"
                  << "Ideal \xc2\xb5s" << "TP Eff"
                  << fort::endr;

            table.column(0).set_cell_text_align(fort::text_align::left);
            table.column(1).set_cell_text_align(fort::text_align::left);
            for (int c = 5; c <= 9; ++c)
                table.column(c).set_cell_text_align(fort::text_align::right);

            double sum_ideal_tp2 = 0, sum_actual_tp2 = 0;
            double sum_ideal_tp4 = 0, sum_actual_tp4 = 0;

            size_t n = std::min({full.size(), tp2.size(), tp4.size()});
            for (size_t i = 0; i < n; ++i)
            {
                const auto &f = full[i], &t2 = tp2[i], &t4 = tp4[i];

                // TP=1 row
                char d1[16], n1[16], r1[16];
                std::snprintf(d1, sizeof(d1), "%.1f", f.onednn_us);
                std::snprintf(n1, sizeof(n1), "%.1f", f.native_us);
                std::snprintf(r1, sizeof(r1), "%.2fx", f.ratio_vs_dnnl);
                table << TP_LAYER_NAMES[i] << TP_SHARD_MODES[i] << 1
                      << f.N << f.K << d1 << n1 << r1 << "-" << "-" << fort::endr;

                // TP=2 row
                double ideal2 = f.native_us / 2.0;
                double eff2 = ideal2 / t2.native_us;
                sum_ideal_tp2 += ideal2;
                sum_actual_tp2 += t2.native_us;
                char d2[16], n2[16], r2[16], i2[16], e2[16];
                std::snprintf(d2, sizeof(d2), "%.1f", t2.onednn_us);
                std::snprintf(n2, sizeof(n2), "%.1f", t2.native_us);
                std::snprintf(r2, sizeof(r2), "%.2fx", t2.ratio_vs_dnnl);
                std::snprintf(i2, sizeof(i2), "%.1f", ideal2);
                std::snprintf(e2, sizeof(e2), "%.0f%%", eff2 * 100);
                table << "" << "" << 2
                      << t2.N << t2.K << d2 << n2 << r2 << i2 << e2 << fort::endr;

                // TP=4 row
                double ideal4 = f.native_us / 4.0;
                double eff4 = ideal4 / t4.native_us;
                sum_ideal_tp4 += ideal4;
                sum_actual_tp4 += t4.native_us;
                char d4[16], n4[16], r4[16], i4[16], e4[16];
                std::snprintf(d4, sizeof(d4), "%.1f", t4.onednn_us);
                std::snprintf(n4, sizeof(n4), "%.1f", t4.native_us);
                std::snprintf(r4, sizeof(r4), "%.2fx", t4.ratio_vs_dnnl);
                std::snprintf(i4, sizeof(i4), "%.1f", ideal4);
                std::snprintf(e4, sizeof(e4), "%.0f%%", eff4 * 100);
                table << "" << "" << 4
                      << t4.N << t4.K << d4 << n4 << r4 << i4 << e4 << fort::endr;

                if (i + 1 < n)
                    table << fort::separator;
            }

            // Summary rows
            table << fort::separator;
            char se2[16], se4[16];
            std::snprintf(se2, sizeof(se2), "%.0f%%",
                          sum_actual_tp2 > 0 ? (sum_ideal_tp2 / sum_actual_tp2) * 100 : 0);
            std::snprintf(se4, sizeof(se4), "%.0f%%",
                          sum_actual_tp4 > 0 ? (sum_ideal_tp4 / sum_actual_tp4) * 100 : 0);
            table << "AVG" << "" << 2 << "" << "" << "" << "" << "" << "" << se2 << fort::endr;
            table << "" << "" << 4 << "" << "" << "" << "" << "" << "" << se4 << fort::endr;

            std::cout << "\n"
                      << title << "\n"
                      << "TP Eff = ideal_time / actual_time (100% = perfect linear scaling)\n"
                      << "Ideal \xc2\xb5s = full_time / tp_degree\n\n"
                      << table.to_string() << std::endl;
        }

        struct TPFormatSummary
        {
            std::string name;
            std::string path;
            double avg_eff_tp2 = 0, avg_eff_tp4 = 0;
            double worst_eff_tp2 = 1e9, worst_eff_tp4 = 1e9;
            std::string worst_layer_tp2, worst_layer_tp4;
            double avg_ratio_full = 0, avg_ratio_tp2 = 0, avg_ratio_tp4 = 0;
        };

        TPFormatSummary computeTPSummary(const std::string &fmt_name, bool is_nibble,
                                         const std::vector<GEMMBenchResult> &full,
                                         const std::vector<GEMMBenchResult> &tp2,
                                         const std::vector<GEMMBenchResult> &tp4)
        {
            TPFormatSummary s;
            s.name = fmt_name;
            s.path = is_nibble ? "NibbleLUT" : "INT8";
            size_t n = std::min({full.size(), tp2.size(), tp4.size()});
            double sum_eff2 = 0, sum_eff4 = 0;
            double sum_ratio_f = 0, sum_ratio_2 = 0, sum_ratio_4 = 0;
            for (size_t i = 0; i < n; ++i)
            {
                double eff2 = (full[i].native_us / 2.0) / tp2[i].native_us;
                double eff4 = (full[i].native_us / 4.0) / tp4[i].native_us;
                sum_eff2 += eff2;
                sum_eff4 += eff4;
                sum_ratio_f += full[i].ratio_vs_dnnl;
                sum_ratio_2 += tp2[i].ratio_vs_dnnl;
                sum_ratio_4 += tp4[i].ratio_vs_dnnl;
                if (eff2 < s.worst_eff_tp2)
                {
                    s.worst_eff_tp2 = eff2;
                    s.worst_layer_tp2 = TP_LAYER_NAMES[i];
                }
                if (eff4 < s.worst_eff_tp4)
                {
                    s.worst_eff_tp4 = eff4;
                    s.worst_layer_tp4 = TP_LAYER_NAMES[i];
                }
            }
            if (n > 0)
            {
                s.avg_eff_tp2 = sum_eff2 / n;
                s.avg_eff_tp4 = sum_eff4 / n;
                s.avg_ratio_full = sum_ratio_f / n;
                s.avg_ratio_tp2 = sum_ratio_2 / n;
                s.avg_ratio_tp4 = sum_ratio_4 / n;
            }
            return s;
        }

        void renderTPFormatSummary(const std::vector<TPFormatSummary> &summaries)
        {
            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);
            table << fort::header
                  << "Format" << "Path"
                  << "Nat/DNN TP1" << "Nat/DNN TP2" << "Nat/DNN TP4"
                  << "Avg Eff TP2" << "Worst TP2" << "Worst Layer"
                  << "Avg Eff TP4" << "Worst TP4" << "Worst Layer"
                  << fort::endr;

            table.column(0).set_cell_text_align(fort::text_align::left);
            table.column(1).set_cell_text_align(fort::text_align::left);
            table.column(7).set_cell_text_align(fort::text_align::left);
            table.column(10).set_cell_text_align(fort::text_align::left);
            for (int c : {2, 3, 4, 5, 6, 8, 9})
                table.column(c).set_cell_text_align(fort::text_align::right);

            for (const auto &s : summaries)
            {
                char rf[16], r2[16], r4[16], ae2[16], we2[16], ae4[16], we4[16];
                std::snprintf(rf, sizeof(rf), "%.2fx", s.avg_ratio_full);
                std::snprintf(r2, sizeof(r2), "%.2fx", s.avg_ratio_tp2);
                std::snprintf(r4, sizeof(r4), "%.2fx", s.avg_ratio_tp4);
                std::snprintf(ae2, sizeof(ae2), "%.0f%%", s.avg_eff_tp2 * 100);
                std::snprintf(we2, sizeof(we2), "%.0f%%", s.worst_eff_tp2 * 100);
                std::snprintf(ae4, sizeof(ae4), "%.0f%%", s.avg_eff_tp4 * 100);
                std::snprintf(we4, sizeof(we4), "%.0f%%", s.worst_eff_tp4 * 100);

                table << s.name << s.path
                      << rf << r2 << r4
                      << ae2 << we2 << s.worst_layer_tp2
                      << ae4 << we4 << s.worst_layer_tp4
                      << fort::endr;
            }

            std::cout << "\n=== TP Scaling Summary (GEMM) ===\n"
                      << "TP Eff = (full_time / tp_degree) / split_time\n"
                      << "  100% = perfect linear scaling\n"
                      << "  <100% = overhead from smaller shape dimensions\n"
                      << table.to_string() << std::endl;
        }
    };

    // =========================================================================
    // TEST 1: Key formats × M=8 × Qwen 0.5B (L3 resident)
    // =========================================================================

    TEST_F(NativeVNNIGemmPerfTest, KeyFormats_M8_05B)
    {
        const int M = 8;
        auto shapes = buildGemmShapes(kQwen05B, M);

        // Pre-compute oneDNN reference
        std::vector<double> dnnl_ref(shapes.size());
        for (size_t i = 0; i < shapes.size(); ++i)
            benchOneDNNShape(shapes[i].M, shapes[i].N, shapes[i].K, dnnl_ref[i]);

        std::vector<GEMMBenchResult> results;
        for (const auto &fmt : KEY_FORMATS)
        {
            for (size_t i = 0; i < shapes.size(); ++i)
            {
                GEMMBenchResult r;
                if (benchTriple(fmt, shapes[i], dnnl_ref[i], r))
                    results.push_back(r);
            }
        }

        renderResults(
            "==========================================================\n"
            "  NativeVNNI GEMM vs oneDNN: Key Formats × Qwen 0.5B (M=8)\n"
            "==========================================================",
            results);
        renderFormatSummary(results);
    }

    // =========================================================================
    // TEST 2: M sweep (M=2,4,8,16,32,64) × Q4_0 × Qwen 0.5B FFN
    //
    // Shows how GEMM scales with batch size
    // =========================================================================

    TEST_F(NativeVNNIGemmPerfTest, MSweep_Q4_0_05B_FFN)
    {
        static const std::vector<int> M_VALUES = {2, 4, 8, 16, 32, 64};
        const FormatSpec fmt = {"Q4_0", true};

        // FFN gate shape from Qwen 0.5B
        const int N = 4864, K = 896;

        std::vector<GEMMBenchResult> results;

        for (int M : M_VALUES)
        {
            GEMMShape shape = {"FFN_Gate_M" + std::to_string(M), M, N, K};

            double dnnl_us;
            benchOneDNNShape(M, N, K, dnnl_us);

            GEMMBenchResult r;
            if (benchTriple(fmt, shape, dnnl_us, r))
                results.push_back(r);
        }

        renderResults(
            "==========================================================\n"
            "  NativeVNNI GEMM M-Sweep: Q4_0 × FFN_Gate [4864×896]\n"
            "  Shows scaling with batch size\n"
            "==========================================================",
            results);
    }

    // =========================================================================
    // TEST 3: All formats × M=16 × Qwen 3B (DRAM-spilling)
    // =========================================================================

    TEST_F(NativeVNNIGemmPerfTest, AllFormats_M16_3B)
    {
        const int M = 16;
        auto shapes = buildGemmShapes(kQwen3B, M);

        std::vector<double> dnnl_ref(shapes.size());
        for (size_t i = 0; i < shapes.size(); ++i)
            benchOneDNNShape(shapes[i].M, shapes[i].N, shapes[i].K, dnnl_ref[i]);

        std::vector<GEMMBenchResult> results;
        for (const auto &fmt : ALL_FORMATS)
        {
            for (size_t i = 0; i < shapes.size(); ++i)
            {
                GEMMBenchResult r;
                if (benchTriple(fmt, shapes[i], dnnl_ref[i], r))
                    results.push_back(r);
            }
        }

        renderResults(
            "==========================================================\n"
            "  NativeVNNI GEMM vs oneDNN: All Formats × Qwen 3B (M=16)\n"
            "==========================================================",
            results);
        renderFormatSummary(results);
    }

    // =========================================================================
    // TEST 4: Key formats × M=32 × Qwen 7B (fully DRAM-bound)
    // =========================================================================

    TEST_F(NativeVNNIGemmPerfTest, KeyFormats_M32_7B)
    {
        const int M = 32;
        auto shapes = buildGemmShapes(kQwen7B, M);

        std::vector<double> dnnl_ref(shapes.size());
        for (size_t i = 0; i < shapes.size(); ++i)
            benchOneDNNShape(shapes[i].M, shapes[i].N, shapes[i].K, dnnl_ref[i]);

        std::vector<GEMMBenchResult> results;
        for (const auto &fmt : KEY_FORMATS)
        {
            for (size_t i = 0; i < shapes.size(); ++i)
            {
                GEMMBenchResult r;
                if (benchTriple(fmt, shapes[i], dnnl_ref[i], r))
                    results.push_back(r);
            }
        }

        renderResults(
            "==========================================================\n"
            "  NativeVNNI GEMM vs oneDNN: Key Formats × Qwen 7B (M=32)\n"
            "==========================================================",
            results);
        renderFormatSummary(results);
    }

    // =========================================================================
    // TEST 5: TP Scaling — Key Formats × M=8 × Qwen 7B
    //
    // Benchmarks full + TP=2 + TP=4 shapes to measure TP scaling efficiency.
    // 7B model keeps TP-split dimensions large enough to be representative.
    // =========================================================================

    TEST_F(NativeVNNIGemmPerfTest, TP_Scaling_KeyFormats_M8_7B)
    {
        const int M = 8;
        const auto &model = kQwen7B;
        auto shapes_full = buildGemmShapes(model, M);
        auto shapes_tp2 = buildTPGemmShapes(model, M, 2);
        auto shapes_tp4 = buildTPGemmShapes(model, M, 4);

        std::vector<TPFormatSummary> format_summaries;

        for (const auto &fmt : KEY_FORMATS)
        {
            // Pre-compute oneDNN references for all three TP degrees
            std::vector<double> dnnl_full(shapes_full.size());
            std::vector<double> dnnl_tp2(shapes_tp2.size());
            std::vector<double> dnnl_tp4(shapes_tp4.size());
            for (size_t i = 0; i < shapes_full.size(); ++i)
            {
                benchOneDNNShape(M, shapes_full[i].N, shapes_full[i].K, dnnl_full[i]);
                benchOneDNNShape(M, shapes_tp2[i].N, shapes_tp2[i].K, dnnl_tp2[i]);
                benchOneDNNShape(M, shapes_tp4[i].N, shapes_tp4[i].K, dnnl_tp4[i]);
            }

            std::vector<GEMMBenchResult> res_full, res_tp2, res_tp4;
            for (size_t i = 0; i < shapes_full.size(); ++i)
            {
                GEMMBenchResult r;
                if (benchTriple(fmt, shapes_full[i], dnnl_full[i], r))
                    res_full.push_back(r);
                if (benchTriple(fmt, shapes_tp2[i], dnnl_tp2[i], r))
                    res_tp2.push_back(r);
                if (benchTriple(fmt, shapes_tp4[i], dnnl_tp4[i], r))
                    res_tp4.push_back(r);
            }

            renderTPScaling(
                "=== TP Scaling: " + fmt.name + " \xc3\x97 Qwen 7B (M=8) ===",
                res_full, res_tp2, res_tp4);

            format_summaries.push_back(
                computeTPSummary(fmt.name, fmt.is_nibble_lut, res_full, res_tp2, res_tp4));
        }

        renderTPFormatSummary(format_summaries);
    }

    // =========================================================================
    // TEST 6: TP Scaling — Key Formats × M=8 × Qwen 0.5B
    //
    // Small model stresses TP scaling limits: K_proj TP=4 → N=32 (< 1 chunk!).
    // This is the worst case for kernel efficiency with TP-split shapes.
    // =========================================================================

    TEST_F(NativeVNNIGemmPerfTest, TP_Scaling_KeyFormats_M8_05B)
    {
        const int M = 8;
        const auto &model = kQwen05B;
        auto shapes_full = buildGemmShapes(model, M);
        auto shapes_tp2 = buildTPGemmShapes(model, M, 2);
        auto shapes_tp4 = buildTPGemmShapes(model, M, 4);

        std::vector<TPFormatSummary> format_summaries;

        for (const auto &fmt : KEY_FORMATS)
        {
            std::vector<double> dnnl_full(shapes_full.size());
            std::vector<double> dnnl_tp2(shapes_tp2.size());
            std::vector<double> dnnl_tp4(shapes_tp4.size());
            for (size_t i = 0; i < shapes_full.size(); ++i)
            {
                benchOneDNNShape(M, shapes_full[i].N, shapes_full[i].K, dnnl_full[i]);
                benchOneDNNShape(M, shapes_tp2[i].N, shapes_tp2[i].K, dnnl_tp2[i]);
                benchOneDNNShape(M, shapes_tp4[i].N, shapes_tp4[i].K, dnnl_tp4[i]);
            }

            std::vector<GEMMBenchResult> res_full, res_tp2, res_tp4;
            for (size_t i = 0; i < shapes_full.size(); ++i)
            {
                GEMMBenchResult r;
                if (benchTriple(fmt, shapes_full[i], dnnl_full[i], r))
                    res_full.push_back(r);
                if (benchTriple(fmt, shapes_tp2[i], dnnl_tp2[i], r))
                    res_tp2.push_back(r);
                if (benchTriple(fmt, shapes_tp4[i], dnnl_tp4[i], r))
                    res_tp4.push_back(r);
            }

            renderTPScaling(
                "=== TP Scaling: " + fmt.name + " \xc3\x97 Qwen 0.5B (M=8) ===",
                res_full, res_tp2, res_tp4);

            format_summaries.push_back(
                computeTPSummary(fmt.name, fmt.is_nibble_lut, res_full, res_tp2, res_tp4));
        }

        renderTPFormatSummary(format_summaries);
    }

} // anonymous namespace
