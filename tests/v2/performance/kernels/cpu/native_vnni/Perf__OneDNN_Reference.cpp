/**
 * @file Perf__OneDNN_Reference.cpp
 * @brief Side-by-side benchmark comparing NativeVNNI GEMV against oneDNN
 *        s8×s8→f32 matmul as a "known good" INT8 reference implementation.
 *
 * oneDNN (Intel) uses optimised AVX-512 VNNI paths internally and
 * represents an upper bound for how fast pure INT8 GEMV can run on this
 * hardware.  By comparing our NativeVNNI kernels against oneDNN we can
 * identify shapes / formats where we are leaving performance on the table.
 *
 * The benchmark uses oneDNN's `format_tag::any` for weights, which lets
 * oneDNN repack into its own optimal blocked layout (one-time cost, same
 * as our packing pass).
 *
 * System: 2× Xeon Gold 6238R, 56 physical cores, DDR4-2933, L3 38.5 MB/socket
 * Measured STREAM read BW: ~117 GB/s.
 *
 * @note Requires libdnnl (apt install libdnnl-dev libdnnl3).
 *       Compile with: -ldnnl
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <oneapi/dnnl/dnnl.hpp>

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
    // Constants
    // =========================================================================
    static constexpr double MEASURED_READ_BW_GBS = 117.0;
    static constexpr int WARMUP_ITERS = 50;
    static constexpr int BENCH_ITERS = 200;

    // =========================================================================
    // Model shapes (same definitions as the main sweep)
    // =========================================================================

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

    struct GEMVShape
    {
        std::string name;
        int N;
        int K;
    };

    static std::vector<GEMVShape> buildShapes(const ModelConfig &m)
    {
        int n_q = m.n_heads * m.head_dim;
        int n_kv = m.n_kv_heads * m.head_dim;
        return {
            {m.name + "_Q_proj", n_q, m.d_model},
            {m.name + "_K_proj", n_kv, m.d_model},
            {m.name + "_V_proj", n_kv, m.d_model},
            {m.name + "_Wo_proj", m.d_model, n_q},
            {m.name + "_FFN_Gate", m.d_ff, m.d_model},
            {m.name + "_FFN_Up", m.d_ff, m.d_model},
            {m.name + "_FFN_Down", m.d_model, m.d_ff},
        };
    }

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

    // Representative formats for the main comparison
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
    // oneDNN INT8 matmul engine (prepacked weights)
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

        /**
         * @brief Build a oneDNN s8×s8→f32 matmul with pre-packed weights.
         *
         * After construction, call execute() to run the GEMV.  The weight
         * repack to oneDNN's internal blocked format happens in the ctor
         * (one-time cost, excluded from benchmark).
         */
        OneDNNMatmul(int M, int N, int K, const int8_t *weight_data)
            : eng_(dnnl::engine::kind::cpu, 0),
              stream_(eng_),
              M_(M), N_(N), K_(K)
        {
            using namespace dnnl;
            auto src_md = memory::desc({M, K}, memory::data_type::s8, memory::format_tag::ab);
            auto wei_user_md = memory::desc({K, N}, memory::data_type::s8, memory::format_tag::ab);
            // Let oneDNN choose optimal internal layout
            auto wei_any_md = memory::desc({K, N}, memory::data_type::s8, memory::format_tag::any);
            auto dst_md = memory::desc({M, N}, memory::data_type::f32, memory::format_tag::ab);

            auto pd = matmul::primitive_desc(eng_, src_md, wei_any_md, dst_md);
            prim_ = matmul(pd);

            // Allocate runtime buffers
            src_mem_ = memory(src_md, eng_);
            dst_mem_ = memory(dst_md, eng_);

            // Repack weights (one-time cost)
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
            matmul_args_[DNNL_ARG_SRC] = src_mem_;
            matmul_args_[DNNL_ARG_WEIGHTS] = wei_mem_;
            matmul_args_[DNNL_ARG_DST] = dst_mem_;
            prim_.execute(stream_, matmul_args_);
            stream_.wait();
            std::memcpy(dst, dst_mem_.get_data_handle(), (size_t)M_ * N_ * sizeof(float));
        }

    private:
        std::unordered_map<int, dnnl::memory> matmul_args_;
    };

    // =========================================================================
    // Benchmark helpers
    // =========================================================================

    /**
     * @brief Benchmark a NativeVNNI or QuantisedGemm kernel, return p10 latency.
     */
    double benchKernel(ITensorGemm *kernel, const float *A, float *C,
                       int M, int N, int K, int warmup, int iters)
    {
        // Create tensor wrappers for multiply_tensor
        auto A_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)K});
        std::memcpy(A_tensor->mutable_data(), A, (size_t)M * K * sizeof(float));
        auto C_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)N});

        for (int i = 0; i < warmup; ++i)
            kernel->multiply_tensor(A_tensor.get(), C_tensor.get(), M, N, K);

        std::vector<double> times(iters);
        for (int i = 0; i < iters; ++i)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            kernel->multiply_tensor(A_tensor.get(), C_tensor.get(), M, N, K);
            auto t1 = std::chrono::high_resolution_clock::now();
            times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
        }

        // Copy result back to caller's C buffer
        std::memcpy(C, C_tensor->data(), (size_t)M * N * sizeof(float));

        std::sort(times.begin(), times.end());
        return times[std::max(0, (int)(iters * 0.1) - 1)];
    }

    /**
     * @brief Benchmark the oneDNN matmul, return p10 latency.
     */
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
        return times[std::max(0, (int)(iters * 0.1) - 1)];
    }

    // =========================================================================
    // Benchmark result struct
    // =========================================================================

    struct RefBenchResult
    {
        std::string shape_name;
        std::string format_name;
        int N, K;
        double onednn_us;       // oneDNN s8×s8→f32 latency
        double native_us;       // NativeVNNI latency
        double baseline_us;     // CPUNativeVNNIGemmKernel latency
        double onednn_bw_gbs;   // oneDNN BW (based on INT8 wt bytes)
        double native_bw_gbs;   // NativeVNNI BW (based on packed bytes)
        double ratio_vs_dnnl;   // native_us / onednn_us (>1 = we are slower)
        double speedup_vs_base; // baseline_us / native_us
        bool is_nibble_lut;
    };

    // =========================================================================
    // Test fixture
    // =========================================================================

    class OneDNNReferenceTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            int initialized = 0;
            MPI_Initialized(&initialized);
            if (!initialized)
                MPI_Init(nullptr, nullptr);
        }

        /**
         * @brief Run a shape against oneDNN only (no format-specific weights).
         *        Uses random INT8 weight data directly.
         */
        void benchOneDNNShape(const GEMVShape &shape,
                              double &out_onednn_us, double &out_bw_gbs)
        {
            const int M = 1, N = shape.N, K = shape.K;

            // Generate random INT8 weights
            std::vector<int8_t> weights(K * N);
            std::mt19937 rng(42);
            std::uniform_int_distribution<int> dist(-127, 127);
            for (auto &w : weights)
                w = static_cast<int8_t>(dist(rng));

            OneDNNMatmul mm(M, N, K, weights.data());

            // Random INT8 activations
            std::vector<int8_t> src(K);
            for (auto &v : src)
                v = static_cast<int8_t>(dist(rng));
            std::vector<float> dst(N);

            out_onednn_us = benchOneDNN(mm, src.data(), dst.data(),
                                        WARMUP_ITERS, BENCH_ITERS);
            double int8_bytes = (double)K * N;
            out_bw_gbs = int8_bytes / (out_onednn_us * 1e-6) / 1e9;
        }

        /**
         * @brief Full comparison: oneDNN vs NativeVNNI vs QuantisedGemm
         *        for a given format + shape.
         */
        bool benchTriple(const FormatSpec &fmt, const GEMVShape &shape,
                         double cached_onednn_us, double cached_onednn_bw,
                         RefBenchResult &result)
        {
            auto weights = createWeights(fmt.name, shape.N, shape.K);
            if (!weights)
                return false;

            // NativeVNNI
            CPUNativeVNNIGemmKernel native_kernel(weights.get());
            if (!native_kernel.isValid())
                return false;

            // Baseline CPUNativeVNNIGemmKernel
            auto baseline_kernel = std::make_unique<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(
                weights.get());

            // Random FP32 activations (same as main sweep)
            std::vector<float> A(shape.K);
            std::mt19937 rng(42);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (auto &v : A)
                v = dist(rng);

            std::vector<float> C_native(shape.N, 0.0f);
            std::vector<float> C_baseline(shape.N, 0.0f);

            double native_us = benchKernel(&native_kernel, A.data(), C_native.data(),
                                           1, shape.N, shape.K, WARMUP_ITERS, BENCH_ITERS);
            double baseline_us = benchKernel(baseline_kernel.get(), A.data(), C_baseline.data(),
                                             1, shape.N, shape.K, WARMUP_ITERS, BENCH_ITERS);

            // NativeVNNI packed bytes for BW calculation
            const auto &packed = native_kernel.packedWeights();
            int N_chunks = (shape.N + 63) / 64;
            int bpr = packed.blocks_per_row;
            double interleaved_bytes = (double)N_chunks * bpr * packed.interleaved_block_stride;
            double meta_bytes = (double)N_chunks * bpr * 64 * (4 + 4);
            double min_bytes = packed.is_asymmetric ? (double)N_chunks * bpr * 64 * 4 : 0;
            double total_packed = interleaved_bytes + meta_bytes + min_bytes;

            result.shape_name = shape.name;
            result.format_name = fmt.name;
            result.N = shape.N;
            result.K = shape.K;
            result.onednn_us = cached_onednn_us;
            result.native_us = native_us;
            result.baseline_us = baseline_us;
            result.onednn_bw_gbs = cached_onednn_bw;
            result.native_bw_gbs = total_packed / (native_us * 1e-6) / 1e9;
            result.ratio_vs_dnnl = native_us / cached_onednn_us;
            result.speedup_vs_base = baseline_us / native_us;
            result.is_nibble_lut = fmt.is_nibble_lut;

            return true;
        }

        // =====================================================================
        // Rendering
        // =====================================================================

        void renderOneDNNBaseline(const std::string &title,
                                  const std::vector<GEMVShape> &shapes,
                                  const std::vector<std::pair<double, double>> &onednn_results)
        {
            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);
            table << fort::header
                  << "Shape" << "N" << "K" << "INT8 MB"
                  << "oneDNN us" << "oneDNN GB/s"
                  << fort::endr;

            table.column(0).set_cell_text_align(fort::text_align::left);
            for (int c = 1; c <= 5; ++c)
                table.column(c).set_cell_text_align(fort::text_align::right);

            for (size_t i = 0; i < shapes.size(); ++i)
            {
                double int8_mb = (double)shapes[i].K * shapes[i].N / 1e6;
                char mb_buf[16], us_buf[16], bw_buf[16];
                std::snprintf(mb_buf, sizeof(mb_buf), "%.1f", int8_mb);
                std::snprintf(us_buf, sizeof(us_buf), "%.1f", onednn_results[i].first);
                std::snprintf(bw_buf, sizeof(bw_buf), "%.1f", onednn_results[i].second);

                table << shapes[i].name << shapes[i].N << shapes[i].K
                      << mb_buf << us_buf << bw_buf
                      << fort::endr;
            }

            std::cout << "\n"
                      << title << "\n"
                      << "oneDNN version: " << dnnl_version()->major << "."
                      << dnnl_version()->minor << "." << dnnl_version()->patch
                      << ", OMP threads: " << omp_get_max_threads() << "\n\n"
                      << table.to_string() << std::endl;
        }

        void renderComparison(const std::string &title,
                              const std::vector<RefBenchResult> &results)
        {
            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);
            table << fort::header
                  << "Format" << "Path" << "Shape" << "N" << "K"
                  << "oneDNN us" << "Native us" << "Base us"
                  << "Nat/DNN" << "Spdup"
                  << fort::endr;

            table.column(0).set_cell_text_align(fort::text_align::left);
            table.column(2).set_cell_text_align(fort::text_align::left);
            for (int c = 5; c <= 9; ++c)
                table.column(c).set_cell_text_align(fort::text_align::right);

            std::string last_fmt;
            for (const auto &r : results)
            {
                if (!last_fmt.empty() && r.format_name != last_fmt)
                    table << fort::separator;
                last_fmt = r.format_name;

                char dnn_buf[16], nat_buf[16], base_buf[16], ratio_buf[16], spd_buf[16];
                std::snprintf(dnn_buf, sizeof(dnn_buf), "%.1f", r.onednn_us);
                std::snprintf(nat_buf, sizeof(nat_buf), "%.1f", r.native_us);
                std::snprintf(base_buf, sizeof(base_buf), "%.1f", r.baseline_us);
                std::snprintf(ratio_buf, sizeof(ratio_buf), "%.2fx", r.ratio_vs_dnnl);
                std::snprintf(spd_buf, sizeof(spd_buf), "%.2fx", r.speedup_vs_base);

                table << r.format_name
                      << (r.is_nibble_lut ? "Nib" : "I8")
                      << r.shape_name << r.N << r.K
                      << dnn_buf << nat_buf << base_buf
                      << ratio_buf << spd_buf
                      << fort::endr;
            }

            std::cout << "\n"
                      << title << "\n"
                      << table.to_string() << std::endl;
        }

        void renderFormatSummary(const std::vector<RefBenchResult> &results)
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
                char avg_r[16], worst_r[16], spd_buf[16];
                std::snprintf(avg_r, sizeof(avg_r), "%.2fx", avg_ratio);
                std::snprintf(worst_r, sizeof(worst_r), "%.2fx", s.worst_ratio);
                std::snprintf(spd_buf, sizeof(spd_buf), "%.2fx", avg_speedup);

                table << s.name << s.path << s.count
                      << avg_r << worst_r << s.worst_shape << spd_buf
                      << fort::endr;
            }

            std::cout << "\n=== Format Summary vs oneDNN ===\n"
                      << "Nat/DNN = NativeVNNI latency / oneDNN latency\n"
                      << "  1.00x = parity with oneDNN\n"
                      << "  >1.0x = we are slower than oneDNN\n"
                      << "  <1.0x = we are faster than oneDNN\n"
                      << table.to_string() << std::endl;
        }
    };

    // =========================================================================
    // TEST 1: oneDNN baseline across all model shapes (no format comparison)
    //
    // Establishes the oneDNN reference latency for each shape.
    // =========================================================================

    TEST_F(OneDNNReferenceTest, OneDNN_Baseline_AllShapes)
    {
        printf("oneDNN version: %d.%d.%d, OMP threads: %d\n",
               dnnl_version()->major, dnnl_version()->minor, dnnl_version()->patch,
               omp_get_max_threads());

        for (const auto &model : {kQwen05B, kQwen3B, kQwen7B})
        {
            auto shapes = buildShapes(model);
            std::vector<std::pair<double, double>> onednn_results;
            for (const auto &shape : shapes)
            {
                double us, bw;
                benchOneDNNShape(shape, us, bw);
                onednn_results.push_back({us, bw});
            }

            char title[256];
            std::snprintf(title, sizeof(title),
                          "=== oneDNN s8xs8->f32 Baseline: Qwen %s (M=1 decode) ===",
                          model.name.c_str());
            renderOneDNNBaseline(title, shapes, onednn_results);
        }
    }

    // =========================================================================
    // TEST 2: Key formats vs oneDNN on 0.5B shapes
    //
    // Compares NativeVNNI (multiple formats) against oneDNN on small model
    // shapes.  All data fits in L3, isolating compute/packing efficiency.
    // =========================================================================

    TEST_F(OneDNNReferenceTest, KeyFormats_vs_OneDNN_05B)
    {
        auto shapes = buildShapes(kQwen05B);

        // Pre-compute oneDNN reference for each shape (shape-level, not format-level)
        std::vector<std::pair<double, double>> dnnl_ref(shapes.size());
        for (size_t i = 0; i < shapes.size(); ++i)
            benchOneDNNShape(shapes[i], dnnl_ref[i].first, dnnl_ref[i].second);

        // Benchmark NativeVNNI for each format × shape
        std::vector<RefBenchResult> results;
        for (const auto &fmt : KEY_FORMATS)
        {
            for (size_t i = 0; i < shapes.size(); ++i)
            {
                RefBenchResult r;
                if (benchTriple(fmt, shapes[i], dnnl_ref[i].first, dnnl_ref[i].second, r))
                    results.push_back(r);
            }
        }

        renderComparison(
            "==========================================================\n"
            "  NativeVNNI vs oneDNN: Key Formats x Qwen 0.5B (M=1)\n"
            "  Nat/DNN > 1.0 means we are slower than oneDNN\n"
            "==========================================================",
            results);
        renderFormatSummary(results);
    }

    // =========================================================================
    // TEST 3: All formats vs oneDNN on 7B shapes (DRAM-bound)
    //
    // Large FFN shapes (68 MB) spill to DRAM — tests true memory bandwidth
    // bottleneck.  This is the most production-relevant benchmark.
    // =========================================================================

    TEST_F(OneDNNReferenceTest, AllFormats_vs_OneDNN_7B)
    {
        auto shapes = buildShapes(kQwen7B);

        std::vector<std::pair<double, double>> dnnl_ref(shapes.size());
        for (size_t i = 0; i < shapes.size(); ++i)
            benchOneDNNShape(shapes[i], dnnl_ref[i].first, dnnl_ref[i].second);

        std::vector<RefBenchResult> results;
        for (const auto &fmt : ALL_FORMATS)
        {
            for (size_t i = 0; i < shapes.size(); ++i)
            {
                RefBenchResult r;
                if (benchTriple(fmt, shapes[i], dnnl_ref[i].first, dnnl_ref[i].second, r))
                    results.push_back(r);
            }
        }

        renderComparison(
            "==========================================================\n"
            "  NativeVNNI vs oneDNN: All Formats x Qwen 7B (M=1)\n"
            "  FFN weights 13-68 MB: DRAM-bound shapes\n"
            "  Nat/DNN > 1.0 means we are slower than oneDNN\n"
            "==========================================================",
            results);
        renderFormatSummary(results);
    }

    // =========================================================================
    // TEST 4: All formats vs oneDNN on 3B shapes (L3-resident)
    // =========================================================================

    TEST_F(OneDNNReferenceTest, AllFormats_vs_OneDNN_3B)
    {
        auto shapes = buildShapes(kQwen3B);

        std::vector<std::pair<double, double>> dnnl_ref(shapes.size());
        for (size_t i = 0; i < shapes.size(); ++i)
            benchOneDNNShape(shapes[i], dnnl_ref[i].first, dnnl_ref[i].second);

        std::vector<RefBenchResult> results;
        for (const auto &fmt : ALL_FORMATS)
        {
            for (size_t i = 0; i < shapes.size(); ++i)
            {
                RefBenchResult r;
                if (benchTriple(fmt, shapes[i], dnnl_ref[i].first, dnnl_ref[i].second, r))
                    results.push_back(r);
            }
        }

        renderComparison(
            "==========================================================\n"
            "  NativeVNNI vs oneDNN: All Formats x Qwen 3B (M=1)\n"
            "  Nat/DNN > 1.0 means we are slower than oneDNN\n"
            "==========================================================",
            results);
        renderFormatSummary(results);
    }

} // anonymous namespace
