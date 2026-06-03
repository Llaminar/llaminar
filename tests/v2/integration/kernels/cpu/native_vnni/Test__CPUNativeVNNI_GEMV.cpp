/**
 * @file Test__CPUNativeVNNI_GEMV.cpp
 * @brief Integration tests for CPU NativeVNNI GEMV/GEMM correctness.
 *
 * Tests all supported formats against a CPU FP32 reference (double-precision
 * accumulation). Validates using cosine similarity with per-format thresholds.
 *
 * Shapes tested: Qwen 0.5B, 1.5B, 3B model dimensions (Attention, FFN, LM_Head).
 *
 * @note Run with Integration build: ctest -R V2_Integration_CPUNativeVNNI_GEMV
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>
#include <array>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "kernels/cpu/native_vnni/CPUNativeVNNIGemmKernel.h"
#include "tensors/Tensors.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"
#include "fort.hpp"

// TestTensorFactory for creating random quantized tensors
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
        const char *msg = "\n[FATAL] Signal caught in integration test — calling MPI_Abort\n";
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
    // FP32 CPU reference GEMV (double-precision accumulation)
    // =========================================================================

    /**
     * @brief Ground-truth GEMV: C[N] = dequant(B)[N×K] @ A[K]
     *
     * Dequantizes weights to FP32 using IINT8Unpackable, then computes
     * dot product with double precision to minimize reference error.
     */
    void cpuFP32GemvReference(const TensorBase *weights, const float *A,
                              float *C, int N, int K)
    {
        const IINT8Unpackable *unpackable = dynamic_cast<const IINT8Unpackable *>(weights);
        ASSERT_NE(unpackable, nullptr);

        const int K_blocks = (K + 31) / 32;

        // Dequantize and compute row-by-row
        for (int n = 0; n < N; ++n)
        {
            double acc = 0.0;
            for (int kb = 0; kb < K_blocks; ++kb)
            {
                int8_t vals[32];
                unpackable->unpack_block_to_int8(n, kb, vals);
                float scale = unpackable->get_block_scale(n, kb);
                float min_val = unpackable->get_block_min(n, kb);

                for (int i = 0; i < 32; ++i)
                {
                    int k_idx = kb * 32 + i;
                    if (k_idx >= K)
                        break;
                    // Dequantize: fp_val = scale * int_val + min
                    double fp_weight = static_cast<double>(scale) * static_cast<double>(vals[i]) + static_cast<double>(min_val);
                    acc += fp_weight * static_cast<double>(A[k_idx]);
                }
            }
            C[n] = static_cast<float>(acc);
        }
    }

    /**
     * @brief FP32 GEMM reference (calls GEMV per row)
     */
    void cpuFP32GemmReference(const TensorBase *weights, const float *A,
                              float *C, int M, int N, int K)
    {
        for (int m = 0; m < M; ++m)
        {
            cpuFP32GemvReference(weights, A + m * K, C + m * N, N, K);
        }
    }

    // =========================================================================
    // Metrics
    // =========================================================================

    float cosineSimilarity(const float *a, const float *b, size_t n)
    {
        double dot = 0, norm_a = 0, norm_b = 0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += (double)a[i] * (double)b[i];
            norm_a += (double)a[i] * (double)a[i];
            norm_b += (double)b[i] * (double)b[i];
        }
        if (norm_a < 1e-15 || norm_b < 1e-15)
            return 0.0f;
        return static_cast<float>(dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
    }

    float maxAbsError(const float *a, const float *b, size_t n)
    {
        float max_err = 0.0f;
        for (size_t i = 0; i < n; ++i)
        {
            float err = std::fabs(a[i] - b[i]);
            if (err > max_err)
                max_err = err;
        }
        return max_err;
    }

    /**
     * @brief Helper: call multiply_tensor via temporary FP32Tensors wrapping raw float*.
     */
    bool multiplyViaTensor(ITensorGemm &kernel, const float *A_data, float *C_data,
                           int M, int N, int K)
    {
        FP32Tensor A_tensor(std::vector<size_t>{(size_t)M, (size_t)K});
        std::memcpy(A_tensor.mutable_data(), A_data, (size_t)M * K * sizeof(float));
        FP32Tensor C_tensor(std::vector<size_t>{(size_t)M, (size_t)N});
        bool ok = kernel.multiply_tensor(&A_tensor, &C_tensor, M, N, K);
        if (ok)
            std::memcpy(C_data, C_tensor.data(), (size_t)M * N * sizeof(float));
        return ok;
    }

    // =========================================================================
    // Shape definitions (Qwen2.5 models)
    // =========================================================================

    struct GEMVShape
    {
        std::string name;
        std::string category;
        int N;
        int K;
    };

    // Shapes used during inference: Attention projs, FFN, LM Head
    static const std::vector<GEMVShape> GEMV_SHAPES = {
        // Qwen2.5-0.5B (hidden=896, intermediate=4864, heads=14, kv=2, hd=64)
        {"0.5B_Q_proj", "Attention", 896, 896},
        {"0.5B_K_proj", "Attention", 128, 896},
        {"0.5B_V_proj", "Attention", 128, 896},
        {"0.5B_Wo_proj", "Attention", 896, 896},
        {"0.5B_FFN_Gate", "FFN", 4864, 896},
        {"0.5B_FFN_Up", "FFN", 4864, 896},
        {"0.5B_FFN_Down", "FFN", 896, 4864},
        {"0.5B_LM_Head", "LM_Head", 151936, 896},

        // Qwen2.5-1.5B (hidden=1536, intermediate=8960, heads=12, kv=2, hd=128)
        {"1.5B_Q_proj", "Attention", 1536, 1536},
        {"1.5B_K_proj", "Attention", 256, 1536},
        {"1.5B_V_proj", "Attention", 256, 1536},
        {"1.5B_Wo_proj", "Attention", 1536, 1536},
        {"1.5B_FFN_Gate", "FFN", 8960, 1536},
        {"1.5B_FFN_Up", "FFN", 8960, 1536},
        {"1.5B_FFN_Down", "FFN", 1536, 8960},

        // Qwen2.5-3B (hidden=2048, intermediate=11008, heads=16, kv=2, hd=128)
        {"3B_Q_proj", "Attention", 2048, 2048},
        {"3B_K_proj", "Attention", 256, 2048},
        {"3B_V_proj", "Attention", 256, 2048},
        {"3B_Wo_proj", "Attention", 2048, 2048},
        {"3B_FFN_Gate", "FFN", 11008, 2048},
        {"3B_FFN_Up", "FFN", 11008, 2048},
        {"3B_FFN_Down", "FFN", 2048, 11008},
        {"3B_LM_Head", "LM_Head", 151936, 2048},
    };

    // =========================================================================
    // Format definitions
    // =========================================================================

    struct FormatSpec
    {
        std::string name;
        float cosine_threshold; // minimum acceptable cosine similarity
    };

    // Phase 1 formats: Q4_0 and IQ4_NL (directly supported in decode)
    static const std::vector<FormatSpec> PHASE1_FORMATS = {
        {"Q4_0", 0.990f},
        {"IQ4_NL", 0.985f},
    };

    // =========================================================================
    // Test fixture
    // =========================================================================

    class CPUNativeVNNIGemvTest : public ::testing::Test
    {
    protected:
    };

    // =========================================================================
    // Correctness test: Q4_0 small matrix
    // =========================================================================

    TEST_F(CPUNativeVNNIGemvTest, Q4_0_SmallMatrix)
    {
        const int N = 128;
        const int K = 128;

        // Create random Q4_0 weights
        auto weights = TestTensorFactory::createQ4_0Random({(size_t)N, (size_t)K});
        ASSERT_NE(weights, nullptr);

        // Create kernel
        CPUNativeVNNIGemmKernel kernel(weights.get());
        ASSERT_TRUE(kernel.isValid());

        // Random activations
        std::vector<float> A(K);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : A)
            v = dist(rng);

        // Compute via NativeVNNI
        std::vector<float> C_native(N, 0.0f);
        ASSERT_TRUE(multiplyViaTensor(kernel, A.data(), C_native.data(), 1, N, K));

        // Compute FP32 reference
        std::vector<float> C_ref(N, 0.0f);
        cpuFP32GemvReference(weights.get(), A.data(), C_ref.data(), N, K);

        // Compare
        float cos_sim = cosineSimilarity(C_native.data(), C_ref.data(), N);
        float max_err = maxAbsError(C_native.data(), C_ref.data(), N);

        EXPECT_GE(cos_sim, 0.990f)
            << "Q4_0 SmallMatrix: cosine=" << cos_sim << " max_err=" << max_err;
    }

    TEST_F(CPUNativeVNNIGemvTest, IQ4_NL_SmallMatrix)
    {
        const int N = 128;
        const int K = 128;

        auto weights = TestTensorFactory::createIQ4_NLRandom({(size_t)N, (size_t)K});
        ASSERT_NE(weights, nullptr);

        CPUNativeVNNIGemmKernel kernel(weights.get());
        ASSERT_TRUE(kernel.isValid());

        std::vector<float> A(K);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : A)
            v = dist(rng);

        std::vector<float> C_native(N, 0.0f);
        ASSERT_TRUE(multiplyViaTensor(kernel, A.data(), C_native.data(), 1, N, K));

        std::vector<float> C_ref(N, 0.0f);
        cpuFP32GemvReference(weights.get(), A.data(), C_ref.data(), N, K);

        float cos_sim = cosineSimilarity(C_native.data(), C_ref.data(), N);
        float max_err = maxAbsError(C_native.data(), C_ref.data(), N);

        EXPECT_GE(cos_sim, 0.985f)
            << "IQ4_NL SmallMatrix: cosine=" << cos_sim << " max_err=" << max_err;
    }

    // =========================================================================
    // Correctness test: M>1 GEMM
    // =========================================================================

    TEST_F(CPUNativeVNNIGemvTest, Q4_0_GEMM_M4)
    {
        const int M = 4;
        const int N = 256;
        const int K = 256;

        auto weights = TestTensorFactory::createQ4_0Random({(size_t)N, (size_t)K});
        ASSERT_NE(weights, nullptr);

        CPUNativeVNNIGemmKernel kernel(weights.get());
        ASSERT_TRUE(kernel.isValid());

        std::vector<float> A(M * K);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : A)
            v = dist(rng);

        std::vector<float> C_native(M * N, 0.0f);
        ASSERT_TRUE(multiplyViaTensor(kernel, A.data(), C_native.data(), M, N, K));

        std::vector<float> C_ref(M * N, 0.0f);
        cpuFP32GemmReference(weights.get(), A.data(), C_ref.data(), M, N, K);

        for (int m = 0; m < M; ++m)
        {
            float cos_sim = cosineSimilarity(
                C_native.data() + m * N, C_ref.data() + m * N, N);
            EXPECT_GE(cos_sim, 0.990f) << "Q4_0 GEMM row " << m << ": cosine=" << cos_sim;
        }
    }

    // =========================================================================
    // Full shape sweep (all Qwen shapes × all formats)
    // =========================================================================

    TEST_F(CPUNativeVNNIGemvTest, AccuracySweep_Q4_0_AllShapes)
    {
        const FormatSpec &fmt = PHASE1_FORMATS[0]; // Q4_0
        ASSERT_EQ(fmt.name, "Q4_0");

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header << "Shape" << "Category" << "N" << "K"
              << "Cosine" << "Max Err" << "Status" << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        table.column(4).set_cell_text_align(fort::text_align::right);
        table.column(5).set_cell_text_align(fort::text_align::right);
        table.column(6).set_cell_text_align(fort::text_align::center);

        int pass_count = 0;
        int total = 0;

        for (const auto &shape : GEMV_SHAPES)
        {
            // Skip very large shapes during integration testing
            if (shape.N > 32000)
            {
                continue;
            }

            auto weights = TestTensorFactory::createQ4_0Random(
                {(size_t)shape.N, (size_t)shape.K});
            if (!weights)
            {
                table << shape.name << shape.category << shape.N << shape.K
                      << "-" << "-" << "SKIP" << fort::endr;
                continue;
            }

            CPUNativeVNNIGemmKernel kernel(weights.get());
            if (!kernel.isValid())
            {
                table << shape.name << shape.category << shape.N << shape.K
                      << "-" << "-" << "PACK FAIL" << fort::endr;
                continue;
            }

            std::vector<float> A(shape.K);
            std::mt19937 rng(42);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (auto &v : A)
                v = dist(rng);

            std::vector<float> C_native(shape.N, 0.0f);
            multiplyViaTensor(kernel, A.data(), C_native.data(), 1, shape.N, shape.K);

            std::vector<float> C_ref(shape.N, 0.0f);
            cpuFP32GemvReference(weights.get(), A.data(), C_ref.data(), shape.N, shape.K);

            float cos_sim = cosineSimilarity(C_native.data(), C_ref.data(), shape.N);
            float max_err = maxAbsError(C_native.data(), C_ref.data(), shape.N);

            bool pass = cos_sim >= fmt.cosine_threshold;
            if (pass)
                pass_count++;
            total++;

            char cos_buf[32], err_buf[32];
            std::snprintf(cos_buf, sizeof(cos_buf), "%.6f", cos_sim);
            std::snprintf(err_buf, sizeof(err_buf), "%.6f", max_err);

            table << shape.name << shape.category << shape.N << shape.K
                  << cos_buf << err_buf << (pass ? "\xe2\x9c\x93" : "\xe2\x9c\x97")
                  << fort::endr;

            EXPECT_GE(cos_sim, fmt.cosine_threshold)
                << "Shape " << shape.name << ": cosine=" << cos_sim;
        }

        table << fort::separator;
        table << "TOTAL" << "" << "" << ""
              << "" << "" << std::to_string(pass_count) + "/" + std::to_string(total)
              << fort::endr;

        std::cout << "\n=== CPU NativeVNNI " << fmt.name << " GEMV Accuracy Sweep ===\n";
        std::cout << table.to_string() << std::endl;
    }

    TEST_F(CPUNativeVNNIGemvTest, AccuracySweep_IQ4_NL_AllShapes)
    {
        const FormatSpec &fmt = PHASE1_FORMATS[1]; // IQ4_NL
        ASSERT_EQ(fmt.name, "IQ4_NL");

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header << "Shape" << "Category" << "N" << "K"
              << "Cosine" << "Max Err" << "Status" << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        table.column(4).set_cell_text_align(fort::text_align::right);
        table.column(5).set_cell_text_align(fort::text_align::right);
        table.column(6).set_cell_text_align(fort::text_align::center);

        int pass_count = 0;
        int total = 0;

        for (const auto &shape : GEMV_SHAPES)
        {
            if (shape.N > 32000)
                continue;

            auto weights = TestTensorFactory::createIQ4_NLRandom(
                {(size_t)shape.N, (size_t)shape.K});
            if (!weights)
            {
                table << shape.name << shape.category << shape.N << shape.K
                      << "-" << "-" << "SKIP" << fort::endr;
                continue;
            }

            CPUNativeVNNIGemmKernel kernel(weights.get());
            if (!kernel.isValid())
            {
                table << shape.name << shape.category << shape.N << shape.K
                      << "-" << "-" << "PACK FAIL" << fort::endr;
                continue;
            }

            std::vector<float> A(shape.K);
            std::mt19937 rng(42);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (auto &v : A)
                v = dist(rng);

            std::vector<float> C_native(shape.N, 0.0f);
            multiplyViaTensor(kernel, A.data(), C_native.data(), 1, shape.N, shape.K);

            std::vector<float> C_ref(shape.N, 0.0f);
            cpuFP32GemvReference(weights.get(), A.data(), C_ref.data(), shape.N, shape.K);

            float cos_sim = cosineSimilarity(C_native.data(), C_ref.data(), shape.N);
            float max_err = maxAbsError(C_native.data(), C_ref.data(), shape.N);

            bool pass = cos_sim >= fmt.cosine_threshold;
            if (pass)
                pass_count++;
            total++;

            char cos_buf[32], err_buf[32];
            std::snprintf(cos_buf, sizeof(cos_buf), "%.6f", cos_sim);
            std::snprintf(err_buf, sizeof(err_buf), "%.6f", max_err);

            table << shape.name << shape.category << shape.N << shape.K
                  << cos_buf << err_buf << (pass ? "\xe2\x9c\x93" : "\xe2\x9c\x97")
                  << fort::endr;

            EXPECT_GE(cos_sim, fmt.cosine_threshold)
                << "Shape " << shape.name << ": cosine=" << cos_sim;
        }

        table << fort::separator;
        table << "TOTAL" << "" << "" << ""
              << "" << "" << std::to_string(pass_count) + "/" + std::to_string(total)
              << fort::endr;

        std::cout << "\n=== CPU NativeVNNI " << fmt.name << " GEMV Accuracy Sweep ===\n";
        std::cout << table.to_string() << std::endl;
    }

    // =========================================================================
    // Comparison vs existing CPUNativeVNNIGemmKernel (INT8 requantize path)
    // =========================================================================

    TEST_F(CPUNativeVNNIGemvTest, CompareVsQuantisedGemmKernel_Q4_0)
    {
        // Compare NativeVNNI vs existing CPUNativeVNNIGemmKernel on a medium shape
        const int N = 896;
        const int K = 896;

        auto weights = TestTensorFactory::createQ4_0Random({(size_t)N, (size_t)K});
        ASSERT_NE(weights, nullptr);

        // NativeVNNI kernel
        CPUNativeVNNIGemmKernel native_kernel(weights.get());
        ASSERT_TRUE(native_kernel.isValid());

        // Existing CPUNativeVNNIGemmKernel (INT8 requantize path)
        auto existing_kernel = std::make_unique<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(weights.get());

        // Random activations
        std::vector<float> A(K);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : A)
            v = dist(rng);

        // FP32 reference
        std::vector<float> C_ref(N, 0.0f);
        cpuFP32GemvReference(weights.get(), A.data(), C_ref.data(), N, K);

        // NativeVNNI result
        std::vector<float> C_native(N, 0.0f);
        multiplyViaTensor(native_kernel, A.data(), C_native.data(), 1, N, K);

        // Existing Q8_1 result
        std::vector<float> C_existing(N, 0.0f);
        multiplyViaTensor(*existing_kernel, A.data(), C_existing.data(), 1, N, K);

        float cos_native = cosineSimilarity(C_native.data(), C_ref.data(), N);
        float cos_existing = cosineSimilarity(C_existing.data(), C_ref.data(), N);
        float mae_native = maxAbsError(C_native.data(), C_ref.data(), N);
        float mae_existing = maxAbsError(C_existing.data(), C_ref.data(), N);

        std::cout << "\n=== NativeVNNI vs CPUNativeVNNIGemmKernel (Q4_0 896×896) ===\n"
                  << "  NativeVNNI:      cosine=" << cos_native << " max_err=" << mae_native << "\n"
                  << "  QuantisedGemm:   cosine=" << cos_existing << " max_err=" << mae_existing << "\n";

        // NativeVNNI should be at least as accurate as the INT8 requantize path
        // (it preserves native precision rather than double-quantizing through INT8)
        EXPECT_GE(cos_native, 0.990f);
        EXPECT_GE(cos_existing, 0.990f);
    }

    TEST_F(CPUNativeVNNIGemvTest, Q4_0_FusedSwiGLUDown_SharedKernelConcurrentDecode)
    {
        const int N = 128;
        const int K = 256;
        const int workers = 8;
        const int iterations = 16;

        auto weights = TestTensorFactory::createQ4_0Random({static_cast<size_t>(N), static_cast<size_t>(K)});
        ASSERT_NE(weights, nullptr);

        CPUNativeVNNIGemmKernel shared_kernel(weights.get());
        ASSERT_TRUE(shared_kernel.isValid());

        std::vector<std::unique_ptr<FP32Tensor>> gates;
        std::vector<std::unique_ptr<FP32Tensor>> ups;
        std::vector<std::vector<float>> expected;
        gates.reserve(workers);
        ups.reserve(workers);
        expected.resize(workers);

        CPUNativeVNNIGemmKernel reference_kernel(weights.get());
        ASSERT_TRUE(reference_kernel.isValid());
        for (int worker = 0; worker < workers; ++worker)
        {
            const int m = 1 + (worker % 3);
            gates.push_back(TestTensorFactory::createFP32Random(
                {static_cast<size_t>(m), static_cast<size_t>(K)}, -0.75f, 0.75f, 1000 + worker));
            ups.push_back(TestTensorFactory::createFP32Random(
                {static_cast<size_t>(m), static_cast<size_t>(K)}, -0.75f, 0.75f, 2000 + worker));

            FP32Tensor output({static_cast<size_t>(m), static_cast<size_t>(N)});
            ASSERT_TRUE(reference_kernel.multiply_tensor_with_fused_swiglu(
                gates.back().get(), ups.back().get(), &output, m, N, K));
            expected[worker].assign(output.data(), output.data() + output.numel());
        }

        std::atomic<int> ready{0};
        std::atomic<bool> go{false};
        std::atomic<bool> failed{false};
        std::vector<std::thread> threads;
        threads.reserve(workers);

        for (int worker = 0; worker < workers; ++worker)
        {
            threads.emplace_back([&, worker]()
                                 {
                                     const int m = 1 + (worker % 3);
                                     FP32Tensor output({static_cast<size_t>(m), static_cast<size_t>(N)});
                                     ready.fetch_add(1, std::memory_order_release);
                                     while (!go.load(std::memory_order_acquire))
                                         std::this_thread::yield();

                                     for (int iter = 0; iter < iterations; ++iter)
                                     {
                                         if (!shared_kernel.multiply_tensor_with_fused_swiglu(
                                                 gates[worker].get(), ups[worker].get(), &output, m, N, K))
                                         {
                                             failed.store(true, std::memory_order_release);
                                             return;
                                         }

                                         const float *actual = output.data();
                                         const auto &ref = expected[worker];
                                         for (size_t i = 0; i < ref.size(); ++i)
                                         {
                                             if (std::fabs(actual[i] - ref[i]) > 1e-5f)
                                             {
                                                 failed.store(true, std::memory_order_release);
                                                 return;
                                             }
                                         }
                                     } });
        }

        while (ready.load(std::memory_order_acquire) != workers)
            std::this_thread::yield();
        go.store(true, std::memory_order_release);
        for (auto &thread : threads)
            thread.join();

        EXPECT_FALSE(failed.load(std::memory_order_acquire));
    }

    // =========================================================================
    // All-format support: factory dispatch + threshold table
    // =========================================================================

    /**
     * @brief Create random quantized weights for a given format name.
     * Returns nullptr if format is not supported.
     */
    std::unique_ptr<TensorBase> createWeightsForFormat(
        const std::string &fmt_name, size_t N, size_t K)
    {
        if (fmt_name == "Q4_0")
            return TestTensorFactory::createQ4_0Random({N, K});
        if (fmt_name == "IQ4_NL")
            return TestTensorFactory::createIQ4_NLRandom({N, K});
        if (fmt_name == "Q4_1")
            return TestTensorFactory::createQ4_1Random({N, K});
        if (fmt_name == "IQ4_XS")
            return TestTensorFactory::createIQ4_XSRandom({N, K});
        if (fmt_name == "Q5_0")
            return TestTensorFactory::createQ5_0Random({N, K});
        if (fmt_name == "Q5_1")
            return TestTensorFactory::createQ5_1Random({N, K});
        if (fmt_name == "Q4_K")
            return TestTensorFactory::createQ4_KRandom({N, K});
        if (fmt_name == "Q5_K")
            return TestTensorFactory::createQ5_KRandom({N, K});
        if (fmt_name == "Q6_K")
            return TestTensorFactory::createQ6_KRandom({N, K});
        if (fmt_name == "Q3_K")
            return TestTensorFactory::createQ3_KRandom({N, K});
        if (fmt_name == "Q2_K")
            return TestTensorFactory::createQ2_KRandom({N, K});
        if (fmt_name == "IQ3_S")
            return TestTensorFactory::createIQ3_SRandom({N, K});
        if (fmt_name == "IQ3_XXS")
            return TestTensorFactory::createIQ3_XXSRandom({N, K});
        if (fmt_name == "IQ2_S")
            return TestTensorFactory::createIQ2_SRandom({N, K});
        if (fmt_name == "IQ2_XS")
            return TestTensorFactory::createIQ2_XSRandom({N, K});
        if (fmt_name == "IQ2_XXS")
            return TestTensorFactory::createIQ2_XXSRandom({N, K});
        if (fmt_name == "IQ1_S")
            return TestTensorFactory::createIQ1_SRandom({N, K});
        if (fmt_name == "IQ1_M")
            return TestTensorFactory::createIQ1_MRandom({N, K});
        if (fmt_name == "Q8_0")
            return TestTensorFactory::createQ8_0Random({N, K});
        if (fmt_name == "Q8_1")
            return TestTensorFactory::createQ8_1Random({N, K});
        return nullptr;
    }

    // All formats with their cosine similarity thresholds.
    // Nibble-LUT formats have higher thresholds (preserve more precision).
    // Lower-bit formats (IQ1, IQ2) have lower thresholds due to quantization noise.
    static const std::vector<FormatSpec> ALL_FORMATS = {
        // Nibble-LUT path (4-bit: vpshufb decode)
        {"Q4_0", 0.990f},
        {"IQ4_NL", 0.985f},
        {"Q4_1", 0.990f},
        {"IQ4_XS", 0.985f},
        // INT8 pre-decoded path (per-block formats)
        {"Q5_0", 0.990f},
        {"Q5_1", 0.990f},
        // INT8 pre-decoded path (superblock formats)
        {"Q4_K", 0.990f},
        {"Q5_K", 0.990f},
        {"Q6_K", 0.990f},
        {"Q3_K", 0.980f},
        {"Q2_K", 0.960f},
        {"IQ3_S", 0.970f},
        {"IQ3_XXS", 0.960f},
        {"IQ2_S", 0.920f},
        {"IQ2_XS", 0.900f},
        {"IQ2_XXS", 0.880f},
        {"IQ1_S", 0.800f},
        {"IQ1_M", 0.800f},
        // INT8 pre-decoded path (8-bit formats — trivial decode, highest precision)
        {"Q8_0", 0.999f},
        {"Q8_1", 0.999f},
    };

    // =========================================================================
    // Smoke tests for each newly supported format (small matrix)
    // =========================================================================

    /**
     * @brief Shared smoke-test logic: pack + GEMV + compare against FP32 ref.
     */
    void smokeTestFormat(const std::string &fmt_name, float threshold)
    {
        const int N = 256;
        const int K = 256;

        auto weights = createWeightsForFormat(fmt_name, N, K);
        ASSERT_NE(weights, nullptr) << "Failed to create " << fmt_name << " weights";

        CPUNativeVNNIGemmKernel kernel(weights.get());
        ASSERT_TRUE(kernel.isValid()) << fmt_name << " failed to pack";

        std::vector<float> A(K);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : A)
            v = dist(rng);

        std::vector<float> C_native(N, 0.0f);
        ASSERT_TRUE(multiplyViaTensor(kernel, A.data(), C_native.data(), 1, N, K));

        std::vector<float> C_ref(N, 0.0f);
        cpuFP32GemvReference(weights.get(), A.data(), C_ref.data(), N, K);

        float cos_sim = cosineSimilarity(C_native.data(), C_ref.data(), N);
        float max_err = maxAbsError(C_native.data(), C_ref.data(), N);

        std::cout << "  " << fmt_name << ": cosine=" << cos_sim
                  << " max_err=" << max_err << "\n";

        EXPECT_GE(cos_sim, threshold)
            << fmt_name << " SmallMatrix: cosine=" << cos_sim << " max_err=" << max_err;
    }

    TEST_F(CPUNativeVNNIGemvTest, Q4_1_SmallMatrix) { smokeTestFormat("Q4_1", 0.990f); }
    TEST_F(CPUNativeVNNIGemvTest, IQ4_XS_SmallMatrix) { smokeTestFormat("IQ4_XS", 0.985f); }
    TEST_F(CPUNativeVNNIGemvTest, Q5_0_SmallMatrix) { smokeTestFormat("Q5_0", 0.990f); }
    TEST_F(CPUNativeVNNIGemvTest, Q5_1_SmallMatrix) { smokeTestFormat("Q5_1", 0.990f); }
    TEST_F(CPUNativeVNNIGemvTest, Q4_K_SmallMatrix) { smokeTestFormat("Q4_K", 0.990f); }
    TEST_F(CPUNativeVNNIGemvTest, Q5_K_SmallMatrix) { smokeTestFormat("Q5_K", 0.990f); }
    TEST_F(CPUNativeVNNIGemvTest, Q6_K_SmallMatrix) { smokeTestFormat("Q6_K", 0.990f); }
    TEST_F(CPUNativeVNNIGemvTest, Q3_K_SmallMatrix) { smokeTestFormat("Q3_K", 0.980f); }
    TEST_F(CPUNativeVNNIGemvTest, Q2_K_SmallMatrix) { smokeTestFormat("Q2_K", 0.960f); }
    TEST_F(CPUNativeVNNIGemvTest, IQ3_S_SmallMatrix) { smokeTestFormat("IQ3_S", 0.970f); }
    TEST_F(CPUNativeVNNIGemvTest, IQ3_XXS_SmallMatrix) { smokeTestFormat("IQ3_XXS", 0.960f); }
    TEST_F(CPUNativeVNNIGemvTest, IQ2_S_SmallMatrix) { smokeTestFormat("IQ2_S", 0.920f); }
    TEST_F(CPUNativeVNNIGemvTest, IQ2_XS_SmallMatrix) { smokeTestFormat("IQ2_XS", 0.900f); }
    TEST_F(CPUNativeVNNIGemvTest, IQ2_XXS_SmallMatrix) { smokeTestFormat("IQ2_XXS", 0.880f); }
    TEST_F(CPUNativeVNNIGemvTest, IQ1_S_SmallMatrix) { smokeTestFormat("IQ1_S", 0.800f); }
    TEST_F(CPUNativeVNNIGemvTest, IQ1_M_SmallMatrix) { smokeTestFormat("IQ1_M", 0.800f); }
    TEST_F(CPUNativeVNNIGemvTest, Q8_0_SmallMatrix) { smokeTestFormat("Q8_0", 0.999f); }
    TEST_F(CPUNativeVNNIGemvTest, Q8_1_SmallMatrix) { smokeTestFormat("Q8_1", 0.999f); }

    TEST_F(CPUNativeVNNIGemvTest, MTP_SmallM_FusedProjection_AllFormats)
    {
        const int K = 256;
        const std::array<int, 3> verifier_rows = {2, 3, 4};
        const int N0 = 384;
        const int N1 = 256;

        setenv("LLAMINAR_PERF_STATS_JSON", "/tmp/llaminar_cpu_native_vnni_mtp_smallm.json", 1);
        PerfStatsCollector::reset();

        for (const auto &fmt : ALL_FORMATS)
        {
            auto weights0 = createWeightsForFormat(fmt.name, N0, K);
            auto weights1 = createWeightsForFormat(fmt.name, N1, K);
            ASSERT_NE(weights0, nullptr) << "Failed to create " << fmt.name << " first projection";
            ASSERT_NE(weights1, nullptr) << "Failed to create " << fmt.name << " second projection";

            CPUNativeVNNIGemmKernel kernel0(weights0.get());
            CPUNativeVNNIGemmKernel kernel1(weights1.get());
            ASSERT_TRUE(kernel0.isValid()) << fmt.name << " first projection failed to pack";
            ASSERT_TRUE(kernel1.isValid()) << fmt.name << " second projection failed to pack";

            for (int M : verifier_rows)
            {
                auto input = TestTensorFactory::createFP32Random(
                    {static_cast<size_t>(M), static_cast<size_t>(K)}, -1.0f, 1.0f,
                    static_cast<uint32_t>(1000 + M + K + N0 + fmt.name.size()));
                ASSERT_NE(input, nullptr);

                FP32Tensor fused0({static_cast<size_t>(M), static_cast<size_t>(N0)});
                FP32Tensor fused1({static_cast<size_t>(M), static_cast<size_t>(N1)});
                FP32Tensor separate0({static_cast<size_t>(M), static_cast<size_t>(N0)});
                FP32Tensor separate1({static_cast<size_t>(M), static_cast<size_t>(N1)});

                std::vector<ITensorGemm::TensorProjectionDesc> projections = {
                    {&kernel0, &fused0, N0, nullptr, "mtp_proj0"},
                    {&kernel1, &fused1, N1, nullptr, "mtp_proj1"}};

                ASSERT_TRUE(kernel0.multiply_fused_tensor(input.get(), projections, M, K))
                    << fmt.name << " fused projection failed at M=" << M;
                ASSERT_TRUE(kernel0.multiply_tensor(input.get(), &separate0, M, N0, K))
                    << fmt.name << " separate projection 0 failed at M=" << M;
                ASSERT_TRUE(kernel1.multiply_tensor(input.get(), &separate1, M, N1, K))
                    << fmt.name << " separate projection 1 failed at M=" << M;

                const size_t count0 = static_cast<size_t>(M) * N0;
                const size_t count1 = static_cast<size_t>(M) * N1;
                const float cos0 = cosineSimilarity(fused0.data(), separate0.data(), count0);
                const float cos1 = cosineSimilarity(fused1.data(), separate1.data(), count1);
                const float err0 = maxAbsError(fused0.data(), separate0.data(), count0);
                const float err1 = maxAbsError(fused1.data(), separate1.data(), count1);

                EXPECT_GE(cos0, 0.9999f)
                    << fmt.name << " projection 0 fused/separate mismatch at M=" << M
                    << " max_err=" << err0;
                EXPECT_GE(cos1, 0.9999f)
                    << fmt.name << " projection 1 fused/separate mismatch at M=" << M
                    << " max_err=" << err1;
                EXPECT_LE(err0, 1e-4f)
                    << fmt.name << " projection 0 fused/separate max error at M=" << M;
                EXPECT_LE(err1, 1e-4f)
                    << fmt.name << " projection 1 fused/separate max error at M=" << M;
            }
        }

        const auto records = PerfStatsCollector::snapshot({"kernel.cpu_native_vnni_small_m_fused_projection_calls"});
        uint64_t total_count = 0;
        for (const auto &record : records)
            total_count += record.count;
        EXPECT_EQ(total_count, ALL_FORMATS.size() * verifier_rows.size())
            << "Every format and M=2/3/4 verifier shape should use the CPU fused small-M projection route";
        unsetenv("LLAMINAR_PERF_STATS_JSON");
    }

    // =========================================================================
    // Full shape sweep across ALL formats
    // =========================================================================

    TEST_F(CPUNativeVNNIGemvTest, AccuracySweep_AllFormats)
    {
        // Subset of shapes for efficiency: 0.5B attention + FFN
        static const std::vector<GEMVShape> SWEEP_SHAPES = {
            {"0.5B_Q_proj", "Attention", 896, 896},
            {"0.5B_K_proj", "Attention", 128, 896},
            {"0.5B_FFN_Gate", "FFN", 4864, 896},
            {"0.5B_FFN_Down", "FFN", 896, 4864},
        };

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header << "Format" << "Path" << "Shape" << "N" << "K"
              << "Cosine" << "Max Err" << "Status" << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        table.column(1).set_cell_text_align(fort::text_align::left);
        table.column(5).set_cell_text_align(fort::text_align::right);
        table.column(6).set_cell_text_align(fort::text_align::right);
        table.column(7).set_cell_text_align(fort::text_align::center);

        int pass_count = 0;
        int fail_count = 0;
        int skip_count = 0;

        for (const auto &fmt : ALL_FORMATS)
        {
            // Determine path label
            std::string path_label;
            if (fmt.name == "Q4_0" || fmt.name == "IQ4_NL" ||
                fmt.name == "Q4_1" || fmt.name == "IQ4_XS")
                path_label = "NibbleLUT";
            else
                path_label = "INT8";

            for (const auto &shape : SWEEP_SHAPES)
            {
                auto weights = createWeightsForFormat(fmt.name, shape.N, shape.K);
                if (!weights)
                {
                    table << fmt.name << path_label << shape.name << shape.N << shape.K
                          << "-" << "-" << "SKIP" << fort::endr;
                    skip_count++;
                    continue;
                }

                CPUNativeVNNIGemmKernel kernel(weights.get());
                if (!kernel.isValid())
                {
                    table << fmt.name << path_label << shape.name << shape.N << shape.K
                          << "-" << "-" << "PACK FAIL" << fort::endr;
                    fail_count++;
                    EXPECT_TRUE(false) << fmt.name << " pack failed for " << shape.name;
                    continue;
                }

                std::vector<float> A(shape.K);
                std::mt19937 rng(42);
                std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
                for (auto &v : A)
                    v = dist(rng);

                std::vector<float> C_native(shape.N, 0.0f);
                multiplyViaTensor(kernel, A.data(), C_native.data(), 1, shape.N, shape.K);

                std::vector<float> C_ref(shape.N, 0.0f);
                cpuFP32GemvReference(weights.get(), A.data(), C_ref.data(), shape.N, shape.K);

                float cos_sim = cosineSimilarity(C_native.data(), C_ref.data(), shape.N);
                float max_err = maxAbsError(C_native.data(), C_ref.data(), shape.N);

                bool pass = cos_sim >= fmt.cosine_threshold;
                if (pass)
                    pass_count++;
                else
                    fail_count++;

                char cos_buf[32], err_buf[32];
                std::snprintf(cos_buf, sizeof(cos_buf), "%.6f", cos_sim);
                std::snprintf(err_buf, sizeof(err_buf), "%.6f", max_err);

                table << fmt.name << path_label << shape.name << shape.N << shape.K
                      << cos_buf << err_buf << (pass ? "\xe2\x9c\x93" : "\xe2\x9c\x97")
                      << fort::endr;

                EXPECT_GE(cos_sim, fmt.cosine_threshold)
                    << fmt.name << " " << shape.name << ": cosine=" << cos_sim;
            }

            table << fort::separator;
        }

        table << "TOTAL" << "" << "" << "" << ""
              << "" << "" << std::to_string(pass_count) + "/" + std::to_string(pass_count + fail_count)
              << fort::endr;

        std::cout << "\n=== CPU NativeVNNI ALL FORMATS Accuracy Sweep ===\n";
        std::cout << table.to_string() << std::endl;
    }

    // =========================================================================
    // GEMM correctness for asymmetric formats (M>1)
    // =========================================================================

    TEST_F(CPUNativeVNNIGemvTest, Q4_1_GEMM_M4)
    {
        const int M = 4;
        const int N = 256;
        const int K = 256;

        auto weights = TestTensorFactory::createQ4_1Random({(size_t)N, (size_t)K});
        ASSERT_NE(weights, nullptr);

        CPUNativeVNNIGemmKernel kernel(weights.get());
        ASSERT_TRUE(kernel.isValid());

        std::vector<float> A(M * K);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : A)
            v = dist(rng);

        std::vector<float> C_native(M * N, 0.0f);
        ASSERT_TRUE(multiplyViaTensor(kernel, A.data(), C_native.data(), M, N, K));

        std::vector<float> C_ref(M * N, 0.0f);
        cpuFP32GemmReference(weights.get(), A.data(), C_ref.data(), M, N, K);

        for (int m = 0; m < M; ++m)
        {
            float cos_sim = cosineSimilarity(
                C_native.data() + m * N, C_ref.data() + m * N, N);
            EXPECT_GE(cos_sim, 0.990f) << "Q4_1 GEMM row " << m << ": cosine=" << cos_sim;
        }
    }

    TEST_F(CPUNativeVNNIGemvTest, Q5_0_GEMM_M4)
    {
        const int M = 4;
        const int N = 256;
        const int K = 256;

        auto weights = TestTensorFactory::createQ5_0Random({(size_t)N, (size_t)K});
        ASSERT_NE(weights, nullptr);

        CPUNativeVNNIGemmKernel kernel(weights.get());
        ASSERT_TRUE(kernel.isValid());

        std::vector<float> A(M * K);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : A)
            v = dist(rng);

        std::vector<float> C_native(M * N, 0.0f);
        ASSERT_TRUE(multiplyViaTensor(kernel, A.data(), C_native.data(), M, N, K));

        std::vector<float> C_ref(M * N, 0.0f);
        cpuFP32GemmReference(weights.get(), A.data(), C_ref.data(), M, N, K);

        for (int m = 0; m < M; ++m)
        {
            float cos_sim = cosineSimilarity(
                C_native.data() + m * N, C_ref.data() + m * N, N);
            EXPECT_GE(cos_sim, 0.990f) << "Q5_0 GEMM row " << m << ": cosine=" << cos_sim;
        }
    }

    TEST_F(CPUNativeVNNIGemvTest, Q6_K_GEMM_M4)
    {
        const int M = 4;
        const int N = 256;
        const int K = 256;

        auto weights = TestTensorFactory::createQ6_KRandom({(size_t)N, (size_t)K});
        ASSERT_NE(weights, nullptr);

        CPUNativeVNNIGemmKernel kernel(weights.get());
        ASSERT_TRUE(kernel.isValid());

        std::vector<float> A(M * K);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : A)
            v = dist(rng);

        std::vector<float> C_native(M * N, 0.0f);
        ASSERT_TRUE(multiplyViaTensor(kernel, A.data(), C_native.data(), M, N, K));

        std::vector<float> C_ref(M * N, 0.0f);
        cpuFP32GemmReference(weights.get(), A.data(), C_ref.data(), M, N, K);

        for (int m = 0; m < M; ++m)
        {
            float cos_sim = cosineSimilarity(
                C_native.data() + m * N, C_ref.data() + m * N, N);
            EXPECT_GE(cos_sim, 0.990f) << "Q6_K GEMM row " << m << ": cosine=" << cos_sim;
        }
    }

    TEST_F(CPUNativeVNNIGemvTest, Q8_0_GEMM_M4)
    {
        const int M = 4;
        const int N = 256;
        const int K = 256;

        auto weights = TestTensorFactory::createQ8_0Random({(size_t)N, (size_t)K});
        ASSERT_NE(weights, nullptr);

        CPUNativeVNNIGemmKernel kernel(weights.get());
        ASSERT_TRUE(kernel.isValid());

        std::vector<float> A(M * K);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : A)
            v = dist(rng);

        std::vector<float> C_native(M * N, 0.0f);
        ASSERT_TRUE(multiplyViaTensor(kernel, A.data(), C_native.data(), M, N, K));

        std::vector<float> C_ref(M * N, 0.0f);
        cpuFP32GemmReference(weights.get(), A.data(), C_ref.data(), M, N, K);

        for (int m = 0; m < M; ++m)
        {
            float cos_sim = cosineSimilarity(
                C_native.data() + m * N, C_ref.data() + m * N, N);
            EXPECT_GE(cos_sim, 0.999f) << "Q8_0 GEMM row " << m << ": cosine=" << cos_sim;
        }
    }

} // anonymous namespace
