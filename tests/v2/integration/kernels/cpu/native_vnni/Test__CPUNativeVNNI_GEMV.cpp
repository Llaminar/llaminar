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
#include <filesystem>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "kernels/cpu/rotation/ActivationRotation.h"
#ifdef HAVE_ONEDNN
#include "kernels/cpu/gemm/FloatingPointGemmKernel.h"
#endif
#include "kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"
#include "loaders/ModelLoader.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"
#include "utils/Logger.h"
#include "utils/OpenMPUtils.h"
#include "utils/PerfStatsCollector.h"
#include "fort.hpp"

#include "utils/QuantizedVerifierFormats.h"
#include "utils/VerifierRowTestInventory.h"

using namespace llaminar2;
using namespace llaminar2::cpu::native_vnni;
using namespace llaminar2::test;

namespace
{
    std::filesystem::path qwen36DenseModelPath()
    {
        if (const char *env = std::getenv("LLAMINAR_QWEN36_DENSE_MODEL"))
            return std::filesystem::path(env);
        return std::filesystem::path("/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf");
    }

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

    /**
     * @brief Temporarily override one environment variable and reload CPU VNNI knobs.
     *
     * The NativeVNNI tile selector reads `LLAMINAR_CPU_VNNI_*` through DebugEnv,
     * so tests that force a dispatch lane must update both the process
     * environment and the cached debug-env snapshot.  Restoring in the destructor
     * keeps later all-format sweeps on the production auto-selected policy.
     */
    class ScopedCPUVNNIEnv
    {
    public:
        ScopedCPUVNNIEnv(const char *name, const char *value)
            : name_(name),
              had_old_(std::getenv(name) != nullptr),
              old_value_(had_old_ ? std::getenv(name) : "")
        {
            setenv(name_.c_str(), value, 1);
            mutableDebugEnv().cpu_vnni.reload();
        }

        ~ScopedCPUVNNIEnv()
        {
            if (had_old_)
                setenv(name_.c_str(), old_value_.c_str(), 1);
            else
                unsetenv(name_.c_str());
            mutableDebugEnv().cpu_vnni.reload();
        }

        ScopedCPUVNNIEnv(const ScopedCPUVNNIEnv &) = delete;
        ScopedCPUVNNIEnv &operator=(const ScopedCPUVNNIEnv &) = delete;

    private:
        std::string name_;
        bool had_old_;
        std::string old_value_;
    };

    /**
     * @brief Physical CPU prefill route observed for one production launch.
     *
     * Correct values alone cannot prove that a regression exercised the
     * economical grouped kernel under review. This record mirrors the route
     * dimensions emitted by `gemm_native_vnni_preq()` so tests can assert the
     * exact candidate, K partition, ISA regime, and thread topology that ran.
     */
    struct ObservedCPUPrefillRoute
    {
        bool found = false;          ///< Whether a matching launch was emitted.
        std::string route;           ///< Physical grouped implementation name.
        int k_tiles = 0;             ///< Number of serial-equivalent K tiles.
        int k_tile_blocks = 0;       ///< Quantized K blocks in each tile.
        int n_block_chunks = 0;      ///< Consecutive 64-column chunks per task.
        uint64_t count = 0;          ///< Number of matching physical launches.
    };

    /**
     * @brief Find the production prefill route for one exact tensor geometry.
     *
     * @param M Runtime input-row count.
     * @param N Logical output-column count.
     * @param K Logical reduction dimension.
     * @param codebook Normalized NativeVNNI execution codebook.
     * @return Aggregated route evidence for the matching physical launch.
     */
    ObservedCPUPrefillRoute findObservedCPUPrefillRoute(
        int M,
        int N,
        int K,
        uint8_t codebook)
    {
        ObservedCPUPrefillRoute result;
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"kernel.cpu_native_vnni_prefill_gemm_launch"}))
        {
            if (record.domain != "kernel" ||
                record.name != "cpu_native_vnni_prefill_gemm_launch" ||
                record.kind != PerfStatRecord::Kind::Counter)
            {
                continue;
            }
            const auto tag = [&](const char *name) -> std::string
            {
                const auto iterator = record.tags.find(name);
                return iterator == record.tags.end()
                           ? std::string{}
                           : iterator->second;
            };
            if (tag("m") != std::to_string(M) ||
                tag("n") != std::to_string(N) ||
                tag("k") != std::to_string(K) ||
                tag("codebook") != std::to_string(codebook))
            {
                continue;
            }
            result.found = true;
            result.route = tag("route");
            result.k_tiles = std::stoi(tag("k_tiles"));
            result.k_tile_blocks = std::stoi(tag("k_tile_blocks"));
            result.n_block_chunks = std::stoi(tag("n_block_chunks"));
            result.count += record.count;
        }
        return result;
    }

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
     * @brief Assert byte-for-byte equality with a serial M=1 decode witness.
     *
     * Grouped verifier rows can be published into live MTP state, so the CPU
     * grouped path must reproduce the serial decode float stream exactly.  A
     * tolerance-based check is useful for model-level drift, but it can hide a
     * one-ulp reduction-order bug that later changes sampler decisions.
     */
    void expectBitwiseEqualFloatRows(
        const std::string &label,
        const float *actual,
        const float *expected,
        size_t count,
        size_t row_width)
    {
        ASSERT_NE(actual, nullptr) << label;
        ASSERT_NE(expected, nullptr) << label;
        ASSERT_GT(count, 0u) << label;
        if (std::memcmp(actual, expected, count * sizeof(float)) == 0)
            return;

        size_t first_mismatch = 0;
        std::uint32_t actual_bits = 0;
        std::uint32_t expected_bits = 0;
        for (; first_mismatch < count; ++first_mismatch)
        {
            std::memcpy(&actual_bits, actual + first_mismatch, sizeof(actual_bits));
            std::memcpy(&expected_bits, expected + first_mismatch, sizeof(expected_bits));
            if (actual_bits != expected_bits)
                break;
        }

        const size_t safe_row_width = row_width == 0 ? count : row_width;
        ADD_FAILURE()
            << label
            << " first_mismatch=" << first_mismatch
            << " row=" << (first_mismatch / safe_row_width)
            << " col=" << (first_mismatch % safe_row_width)
            << " actual=" << (first_mismatch < count ? actual[first_mismatch] : 0.0f)
            << " expected=" << (first_mismatch < count ? expected[first_mismatch] : 0.0f)
            << " actual_bits=0x" << std::hex << actual_bits
            << " expected_bits=0x" << expected_bits << std::dec
            << " max_abs=" << maxAbsError(actual, expected, count)
            << " cosine=" << cosineSimilarity(actual, expected, count);
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

    TEST_F(CPUNativeVNNIGemvTest, Q4_K_FusedExpertDownMatchesSerialDecodeChunks)
    {
        constexpr int experts = 3;
        const int N = 512;
        const int K = 512;

        std::array<std::unique_ptr<TensorBase>, experts> weights;
        std::array<std::unique_ptr<CPUNativeVNNIGemmKernel>, experts> kernels;
        std::array<std::unique_ptr<FP32Tensor>, experts> inputs;
        std::array<std::vector<float>, experts> fused_outputs;
        std::array<std::vector<float>, experts> serial_outputs;
        std::array<ITensorGemm::FusedExpertDownDesc, experts> descs;

        for (int expert = 0; expert < experts; ++expert)
        {
            weights[expert] = TestTensorFactory::createQ4_KRandom(
                {static_cast<size_t>(N), static_cast<size_t>(K)});
            ASSERT_NE(weights[expert], nullptr);

            kernels[expert] = std::make_unique<CPUNativeVNNIGemmKernel>(weights[expert].get());
            ASSERT_TRUE(kernels[expert]->isValid());

            inputs[expert] = TestTensorFactory::createFP32Random(
                {1, static_cast<size_t>(K)}, -0.75f, 0.75f,
                static_cast<uint32_t>(3300 + expert));
            ASSERT_NE(inputs[expert], nullptr);

            fused_outputs[expert].assign(N, 0.0f);
            serial_outputs[expert].assign(N, 0.0f);
            ASSERT_TRUE(multiplyViaTensor(
                *kernels[expert],
                inputs[expert]->data(),
                serial_outputs[expert].data(),
                1,
                N,
                K));

            descs[expert] = {
                kernels[expert].get(),
                inputs[expert]->data(),
                fused_outputs[expert].data(),
                N};
        }

        ASSERT_TRUE(kernels[0]->multiply_fused_expert_down(descs.data(), experts, 1, K));

        for (int expert = 0; expert < experts; ++expert)
        {
            expectBitwiseEqualFloatRows(
                "Q4_K fused expert-down expert " + std::to_string(expert),
                fused_outputs[expert].data(),
                serial_outputs[expert].data(),
                static_cast<size_t>(N),
                static_cast<size_t>(N));
        }
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
        for (const auto &format : quantizedVerifierFormats())
        {
            if (fmt_name == format.label)
                return format.create({N, K}, 42u);
        }
        return nullptr;
    }

    /** @brief Diagnostic FP32-reference threshold for one native tensor type. */
    float allFormatCosineThreshold(TensorType type)
    {
        switch (type)
        {
        case TensorType::IQ4_NL:
        case TensorType::IQ4_XS:
            return 0.985f;
        case TensorType::Q3_K:
            return 0.980f;
        case TensorType::Q2_K:
            return 0.960f;
        case TensorType::IQ3_S:
            return 0.970f;
        case TensorType::IQ3_XXS:
            return 0.960f;
        case TensorType::IQ2_S:
            return 0.920f;
        case TensorType::IQ2_XS:
            return 0.900f;
        case TensorType::IQ2_XXS:
            return 0.880f;
        case TensorType::IQ1_S:
        case TensorType::IQ1_M:
            return 0.800f;
        case TensorType::Q8_0:
        case TensorType::Q8_1:
        case TensorType::Q8_K:
            return 0.999f;
        default:
            return 0.990f;
        }
    }

    /**
     * @brief Canonical all-format matrix augmented with reference diagnostics.
     */
    const std::vector<FormatSpec> ALL_FORMATS = []
    {
        std::vector<FormatSpec> formats;
        formats.reserve(quantizedVerifierFormats().size());
        for (const auto &format : quantizedVerifierFormats())
            formats.push_back({format.label, allFormatCosineThreshold(format.tensor_type)});
        return formats;
    }();

    TEST_F(CPUNativeVNNIGemvTest, MTP_FusedExpertDown_AllFormatsMatchSerialDecodeRows)
    {
        constexpr int N = 256;
        constexpr int K = 256;

        setenv("LLAMINAR_PERF_STATS_JSON", "/tmp/llaminar_cpu_native_vnni_fused_expert_down.json", 1);
        PerfStatsCollector::reset();

        std::vector<std::unique_ptr<TensorBase>> weights;
        std::vector<std::unique_ptr<CPUNativeVNNIGemmKernel>> kernels;
        std::vector<std::unique_ptr<FP32Tensor>> inputs;
        std::vector<std::vector<float>> fused_outputs;
        std::vector<std::vector<float>> serial_outputs;
        std::vector<ITensorGemm::FusedExpertDownDesc> descs;
        weights.reserve(ALL_FORMATS.size());
        kernels.reserve(ALL_FORMATS.size());
        inputs.reserve(ALL_FORMATS.size());
        fused_outputs.reserve(ALL_FORMATS.size());
        serial_outputs.reserve(ALL_FORMATS.size());
        descs.reserve(ALL_FORMATS.size());

        for (size_t format_index = 0; format_index < ALL_FORMATS.size(); ++format_index)
        {
            const auto &fmt = ALL_FORMATS[format_index];
            weights.push_back(createWeightsForFormat(fmt.name, N, K));
            ASSERT_NE(weights.back(), nullptr) << fmt.name << " fused expert-down weights";

            kernels.push_back(std::make_unique<CPUNativeVNNIGemmKernel>(weights.back().get()));
            ASSERT_TRUE(kernels.back()->isValid()) << fmt.name << " fused expert-down pack";

            inputs.push_back(TestTensorFactory::createFP32Random(
                {1u, static_cast<size_t>(K)},
                -0.75f,
                0.75f,
                static_cast<uint32_t>(4300 + format_index)));
            ASSERT_NE(inputs.back(), nullptr) << fmt.name << " fused expert-down input";

            fused_outputs.emplace_back(static_cast<size_t>(N), 0.0f);
            serial_outputs.emplace_back(static_cast<size_t>(N), 0.0f);
            ASSERT_TRUE(multiplyViaTensor(
                *kernels.back(),
                inputs.back()->data(),
                serial_outputs.back().data(),
                1,
                N,
                K))
                << fmt.name << " serial expert-down decode";

            descs.push_back({
                kernels.back().get(),
                inputs.back()->data(),
                fused_outputs.back().data(),
                N});
        }

        ASSERT_TRUE(kernels.front()->multiply_fused_expert_down(
            descs.data(),
            static_cast<int>(descs.size()),
            1,
            K));

        for (size_t format_index = 0; format_index < ALL_FORMATS.size(); ++format_index)
        {
            expectBitwiseEqualFloatRows(
                "CPU fused expert-down " + ALL_FORMATS[format_index].name,
                fused_outputs[format_index].data(),
                serial_outputs[format_index].data(),
                static_cast<size_t>(N),
                static_cast<size_t>(N));
        }

        const auto records =
            PerfStatsCollector::snapshot({"kernel.cpu_native_vnni_fused_expert_down_calls"});
        uint64_t fused_expert_down_calls = 0;
        for (const auto &record : records)
        {
            if (record.domain != "kernel" ||
                record.name != "cpu_native_vnni_fused_expert_down_calls" ||
                record.kind != PerfStatRecord::Kind::Counter)
                continue;
            EXPECT_EQ(record.tags.at("m"), "1");
            EXPECT_EQ(record.tags.at("k"), std::to_string(K));
            EXPECT_EQ(record.tags.at("experts"), std::to_string(ALL_FORMATS.size()));
            fused_expert_down_calls += record.count;
        }
        EXPECT_EQ(fused_expert_down_calls, 1u)
            << "The CPU all-format expert-down sweep must use the fused multi-input path";

        PerfStatsCollector::reset();
        unsetenv("LLAMINAR_PERF_STATS_JSON");
    }

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

    TEST_F(CPUNativeVNNIGemvTest, MTP_SmallM_AllFormatsMatchSerialDecodeRows)
    {
        const int K = 256;
        const int N0 = 384;
        const int N1 = 320;
        const auto &verifier_rows = kGroupedVerifierRuntimeRows;

        setenv("LLAMINAR_PERF_STATS_JSON", "/tmp/llaminar_cpu_native_vnni_grouped_verifier.json", 1);
        PerfStatsCollector::reset();

        /*
         * A full format cross-product only repeats the descriptor scheduler:
         * each projection is independently type-erased after validation. Pair
         * every format with its successor instead. The cyclic pairing places
         * every codebook in both descriptor positions while leaving enough
         * budget to certify every runtime M contiguously.
         */
        for (size_t format_index = 0; format_index < ALL_FORMATS.size(); ++format_index)
        {
            const auto &fmt0 = ALL_FORMATS[format_index];
            const auto &fmt1 = ALL_FORMATS[(format_index + 1) % ALL_FORMATS.size()];
            SCOPED_TRACE(fmt0.name + std::string("_proj0/") + fmt1.name + "_proj1");

            auto weights0 = createWeightsForFormat(fmt0.name, N0, K);
            auto weights1 = createWeightsForFormat(fmt1.name, N1, K);
            ASSERT_NE(weights0, nullptr) << "Failed to create " << fmt0.name << " weights";
            ASSERT_NE(weights1, nullptr) << "Failed to create " << fmt1.name << " weights";

            CPUNativeVNNIGemmKernel kernel0(weights0.get());
            CPUNativeVNNIGemmKernel kernel1(weights1.get());
            ASSERT_TRUE(kernel0.isValid()) << fmt0.name << " failed to pack";
            ASSERT_TRUE(kernel1.isValid()) << fmt1.name << " failed to pack";

            for (int M : verifier_rows)
            {
                SCOPED_TRACE(std::string("M=") + std::to_string(M));
                auto input = TestTensorFactory::createFP32Random(
                    {static_cast<size_t>(M), static_cast<size_t>(K)}, -1.0f, 1.0f,
                    static_cast<uint32_t>(1700 + M + format_index * 257));
                ASSERT_NE(input, nullptr);

                FP32Tensor batched0({static_cast<size_t>(M), static_cast<size_t>(N0)});
                FP32Tensor batched1({static_cast<size_t>(M), static_cast<size_t>(N1)});
                std::vector<ITensorGemm::TensorProjectionDesc> projections = {
                    {&kernel0, &batched0, N0, nullptr, "mtp_verifier_projection0"},
                    {&kernel1, &batched1, N1, nullptr, "mtp_verifier_projection1"}};
                ASSERT_TRUE(kernel0.multiply_fused_verifier_rows_decode_equivalent(
                    input.get(), projections, M, K))
                    << fmt0.name << "/" << fmt1.name
                    << " grouped verifier GEMM hook failed at M=" << M;

                std::vector<float> serial0(
                    static_cast<size_t>(M) * static_cast<size_t>(N0),
                    0.0f);
                std::vector<float> serial1(
                    static_cast<size_t>(M) * static_cast<size_t>(N1),
                    0.0f);
                for (int row = 0; row < M; ++row)
                {
                    ASSERT_TRUE(multiplyViaTensor(
                        kernel0,
                        input->data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                        serial0.data() + static_cast<size_t>(row) * static_cast<size_t>(N0),
                        1,
                        N0,
                        K))
                        << fmt0.name << " serial decode GEMV failed at M=" << M
                        << " row=" << row;
                    ASSERT_TRUE(multiplyViaTensor(
                        kernel1,
                        input->data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                        serial1.data() + static_cast<size_t>(row) * static_cast<size_t>(N1),
                        1,
                        N1,
                        K))
                        << fmt1.name << " serial decode GEMV failed at M=" << M
                        << " row=" << row;
                }

                expectBitwiseEqualFloatRows(
                    fmt0.name + std::string("/") + fmt1.name +
                        " CPU grouped verifier projection0 M=" + std::to_string(M),
                    batched0.data(),
                    serial0.data(),
                    serial0.size(),
                    static_cast<size_t>(N0));
                expectBitwiseEqualFloatRows(
                    fmt0.name + std::string("/") + fmt1.name +
                        " CPU grouped verifier projection1 M=" + std::to_string(M),
                    batched1.data(),
                    serial1.data(),
                    serial1.size(),
                    static_cast<size_t>(N1));
            }
        }

        const auto records = PerfStatsCollector::snapshot({
            "kernel.cpu_native_vnni_fused_grouped_verifier_projection_calls",
            "kernel.cpu_native_vnni_grouped_verifier_projection_calls"});
        uint64_t fused_grouped_verifier_calls = 0;
        uint64_t per_projection_grouped_calls = 0;
        for (const auto &record : records)
        {
            if (record.domain == "kernel" &&
                record.name == "cpu_native_vnni_fused_grouped_verifier_projection_calls" &&
                record.kind == PerfStatRecord::Kind::Counter)
            {
                EXPECT_EQ(record.tags.at("k"), std::to_string(K));
                EXPECT_EQ(record.tags.at("projections"), "2");
                const int m_tag = std::stoi(record.tags.at("m"));
                EXPECT_TRUE((m_tag >= 2 && m_tag <= 16) || m_tag == 31);
                fused_grouped_verifier_calls += record.count;
            }
            if (record.domain == "kernel" &&
                record.name == "cpu_native_vnni_grouped_verifier_projection_calls" &&
                record.kind == PerfStatRecord::Kind::Counter)
            {
                per_projection_grouped_calls += record.count;
            }
        }
        EXPECT_EQ(
            fused_grouped_verifier_calls,
            ALL_FORMATS.size() * verifier_rows.size())
            << "Every CPU NativeVNNI mixed-format projection bundle and certified runtime-M "
               "shape must enter the fused grouped verifier path";
        EXPECT_EQ(per_projection_grouped_calls, 0u)
            << "Mixed-format CPU verifier bundles must not quietly drop to the "
               "per-projection grouped route";
        PerfStatsCollector::reset();
        unsetenv("LLAMINAR_PERF_STATS_JSON");
    }

    /**
     * @test Prove production full-K prefill candidates are batch invariant.
     *
     * The ordinary M>1 NativeVNNI GEMM is a separate launcher from the
     * dedicated verifier-row kernel. A learned large-M policy may vary N task
     * granularity, but it may not introduce a K-tile accumulation boundary
     * that changes FP32 parenthesization. This sweep therefore forces every
     * reviewed N-block candidate while fixing one full-K tile, executes the
     * real `multiply_tensor` path, and compares all result bytes with
     * independent production M1 decode rows for every codebook and certified
     * runtime M.
     */
    TEST_F(CPUNativeVNNIGemvTest,
           NativeVNNIPrefillFullKAllFormatsRuntimeMMatchesSerialDecode)
    {
        constexpr int N = 1024;
        constexpr int K = 256;
        constexpr std::array<int, 5> n_block_candidates = {1, 2, 4, 8, 16};
        const auto &runtime_rows = kGroupedVerifierRuntimeRows;
        const std::string full_k_blocks = std::to_string(K / 32);

        setenv(
            "LLAMINAR_PERF_STATS_JSON",
            "/tmp/llaminar_cpu_native_vnni_prefill_full_k.json",
            1);
        PerfStatsCollector::reset();

        for (size_t format_index = 0;
             format_index < ALL_FORMATS.size();
             ++format_index)
        {
            const auto &format = ALL_FORMATS[format_index];
            SCOPED_TRACE(format.name);
            auto weights = createWeightsForFormat(format.name, N, K);
            ASSERT_NE(weights, nullptr) << format.name;
            CPUNativeVNNIGemmKernel kernel(weights.get());
            ASSERT_TRUE(kernel.isValid()) << format.name;
            const auto &packed = kernel.packedWeights();

            for (int M : runtime_rows)
            {
                SCOPED_TRACE(std::string("M=") + std::to_string(M));
                auto input = TestTensorFactory::createFP32Random(
                    {static_cast<size_t>(M), static_cast<size_t>(K)},
                    -1.0f,
                    1.0f,
                    static_cast<uint32_t>(
                        0x6A00u + format_index * 977u + M * 131u));
                ASSERT_NE(input, nullptr);

                std::vector<float> serial(
                    static_cast<size_t>(M) * N,
                    0.0f);
                for (int row = 0; row < M; ++row)
                {
                    ASSERT_TRUE(multiplyViaTensor(
                        kernel,
                        input->data() + static_cast<size_t>(row) * K,
                        serial.data() + static_cast<size_t>(row) * N,
                        1,
                        N,
                        K))
                        << format.name << " serial M1 row " << row;
                }

                const int k_blocks = packed.blocks_per_row;
                std::vector<Q8_1Block> quantized_rows(
                    static_cast<size_t>(M) * k_blocks);
                quantize_activations_to_q8_1(
                    input->data(),
                    quantized_rows.data(),
                    M,
                    K,
                    k_blocks);

                /*
                 * Exercise the actual inference entry once before forcing any
                 * diagnostic candidate. Production Auto must remain total with
                 * no generated prefill policy installed; the explicit calls
                 * below prove the rest of the byte-exact candidate surface.
                 */
                std::vector<float> auto_grouped(
                    static_cast<size_t>(M) * N,
                    0.0f);
                ASSERT_TRUE(multiplyViaTensor(
                    kernel,
                    input->data(),
                    auto_grouped.data(),
                    M,
                    N,
                    K));
                expectBitwiseEqualFloatRows(
                    format.name + " heuristic production full-K prefill M=" +
                        std::to_string(M),
                    auto_grouped.data(),
                    serial.data(),
                    auto_grouped.size(),
                    static_cast<size_t>(N));

                const auto run_candidate = [&](
                                               int n_block_chunks,
                                               PrefillSchedulePolicy schedule,
                                               const char *schedule_name)
                {
                    SCOPED_TRACE(
                        std::string("n_block_chunks=") +
                        std::to_string(n_block_chunks));
                    SCOPED_TRACE(std::string("schedule=") + schedule_name);
                    const std::string n_block_text =
                        std::to_string(n_block_chunks);
                    ScopedCPUVNNIEnv force_n_blocks(
                        "LLAMINAR_CPU_VNNI_N_BLOCK_CHUNKS",
                        n_block_text.c_str());
                    ScopedCPUVNNIEnv force_full_k(
                        "LLAMINAR_CPU_VNNI_K_TILE_BLOCKS",
                        full_k_blocks.c_str());

                    std::vector<float> grouped(
                        static_cast<size_t>(M) * N,
                        0.0f);
                    gemm_native_vnni_preq(
                        packed,
                        quantized_rows.data(),
                        grouped.data(),
                        M,
                        N,
                        ISAPath::AUTO,
                        VerifierRowsPolicy::Pairwise,
                        schedule);
                    expectBitwiseEqualFloatRows(
                        format.name + " production full-K prefill M=" +
                            std::to_string(M) + " nbc=" +
                            std::to_string(n_block_chunks) + " schedule=" +
                            schedule_name,
                        grouped.data(),
                        serial.data(),
                        grouped.size(),
                        static_cast<size_t>(N));
                };
                run_candidate(
                    1,
                    PrefillSchedulePolicy::RowChunkGrid,
                    "row_chunk_grid");
                for (PrefillSchedulePolicy schedule : {
                         PrefillSchedulePolicy::TwoRowNMajor,
                         PrefillSchedulePolicy::TwoRowPairGrid})
                {
                    const char *schedule_name =
                        schedule == PrefillSchedulePolicy::TwoRowPairGrid
                            ? "two_row_pair_grid"
                            : "two_row_n_major";
                    for (int n_block_chunks : n_block_candidates)
                        run_candidate(
                            n_block_chunks, schedule, schedule_name);
                }
            }
        }

        const auto records = PerfStatsCollector::snapshot(
            {"kernel.cpu_native_vnni_prefill_gemm_launch"});
        uint64_t launch_count = 0;
        std::set<int> observed_n_block_chunks;
        std::set<std::string> observed_routes;
        for (const auto &record : records)
        {
            if (record.domain != "kernel" ||
                record.name != "cpu_native_vnni_prefill_gemm_launch" ||
                record.kind != PerfStatRecord::Kind::Counter)
            {
                continue;
            }
            EXPECT_EQ(record.tags.at("k"), std::to_string(K));
            EXPECT_EQ(record.tags.at("k_tile_blocks"), full_k_blocks);
            EXPECT_EQ(record.tags.at("k_tiles"), "1");
            EXPECT_TRUE(
                record.tags.at("route") == "row_chunk_grid" ||
                record.tags.at("route") == "two_row_n_major" ||
                record.tags.at("route") == "two_row_pair_grid");
            EXPECT_GT(std::stoi(record.tags.at("parallel_tasks")), 0);
            if (record.tags.at("route") == "two_row_pair_grid")
                EXPECT_EQ(record.tags.at("row_tile"), "2");
            observed_routes.insert(record.tags.at("route"));
            observed_n_block_chunks.insert(
                std::stoi(record.tags.at("n_block_chunks")));
            launch_count += record.count;
        }
        EXPECT_TRUE(std::ranges::includes(
            observed_n_block_chunks,
            std::set<int>(n_block_candidates.begin(), n_block_candidates.end())))
            << "The diagnostic override must still exercise every reviewed "
               "full-K N-block candidate";
        EXPECT_TRUE(std::ranges::includes(
            observed_routes,
            std::set<std::string>{
                "row_chunk_grid", "two_row_n_major", "two_row_pair_grid"}))
            << "The all-format sweep must execute every full-K task topology";
        EXPECT_EQ(
            launch_count,
            ALL_FORMATS.size() * runtime_rows.size() *
                (2u * n_block_candidates.size() + 2u))
            << "Every full-K candidate and heuristic Auto decision must publish "
               "one production M>1 route";

        const auto pair_grid_records = PerfStatsCollector::snapshot(
            {"kernel.cpu_native_vnni_pair_grid_execution"});
        uint64_t pair_grid_execution_count = 0;
        std::set<int> physically_executed_n_block_chunks;
        for (const auto &record : pair_grid_records)
        {
            if (record.domain != "kernel" ||
                record.name != "cpu_native_vnni_pair_grid_execution" ||
                record.kind != PerfStatRecord::Kind::Counter)
            {
                continue;
            }
            const int m = std::stoi(record.tags.at("m"));
            const int n = std::stoi(record.tags.at("n"));
            const int nbc = std::stoi(record.tags.at("n_block_chunks"));
            const int expected_tasks =
                ((m + 1) / 2) * (((n + 63) / 64 + nbc - 1) / nbc);
            EXPECT_EQ(
                std::stoi(record.tags.at("parallel_tasks")),
                expected_tasks);
            physically_executed_n_block_chunks.insert(nbc);
            pair_grid_execution_count += record.count;
        }
        EXPECT_TRUE(std::ranges::includes(
            physically_executed_n_block_chunks,
            std::set<int>(n_block_candidates.begin(), n_block_candidates.end())))
            << "The all-format sweep must prove the physical pair-grid task "
               "granularity, not only the outer requested-route label";
        EXPECT_GE(
            pair_grid_execution_count,
            ALL_FORMATS.size() * runtime_rows.size() *
                n_block_candidates.size())
            << "Every forced pair-grid candidate must publish its physical "
               "execution geometry";

        PerfStatsCollector::reset();
        unsetenv("LLAMINAR_PERF_STATS_JSON");
    }

    /**
     * @test Prove partial 64-column chunks remain inside each logical output row.
     *
     * The economical full-K prefill kernel shares one packed-weight chunk
     * across two M rows. Its physical AVX2/AVX512 microkernels always publish
     * 64 floats, while a model projection may end at any logical N. A direct
     * store for the final partial chunk corrupts the next row and eventually
     * writes past the output allocation. The fresh CPU prefill sealed-v2
     * geometry `N=608, K=704, M=64` exposed that defect first at flat output
     * index 608.
     *
     * This regression deliberately exercises:
     *
     * - every source format and normalized execution codebook;
     * - even M, odd M, and the exact long-M sealed failure;
     * - logical tail widths immediately below, at, and above 32 columns; and
     * - all five learned N-block schedules on a wide shape where none of them
     *   normalizes to the separate row-chunk-grid route.
     *
     * Serial M1 runs consume the exact same prequantized activation rows. Guard
     * regions around the logical output make both prefix underflow and suffix
     * overflow observable independently of allocator behavior.
     */
    TEST_F(CPUNativeVNNIGemvTest,
           NativeVNNIPrefillFullKAllFormatsPartialNChunksStayWithinRows)
    {
        constexpr int K = 704;
        constexpr float guard_value = -123456.75f;
        constexpr size_t guard_values = 64;
        const std::array<int, 5> narrow_tail_ns = {577, 607, 608, 609, 639};
        const std::array<int, 2> short_runtime_rows = {2, 3};
        const std::array<int, 1> narrow_candidates = {1};
        const std::array<int, 5> wide_candidates = {1, 2, 4, 8, 16};

        /*
         * The original sealed collector ran one 28-core socket per MPI rank.
         * Route normalization depends on thread count, so retain that topology
         * here; a whole-machine default could silently choose row_chunk_grid
         * for N=608 and fail to cover the repaired two-row implementation.
         */
        llaminar::v2::ThreadCountGuard socket_thread_regime(28);

        setenv(
            "LLAMINAR_PERF_STATS_JSON",
            "/tmp/llaminar_cpu_native_vnni_prefill_partial_n.json",
            1);

        const auto run_shape = [&](
                                   int N,
                                   const auto &runtime_rows,
                                   const auto &n_block_candidates)
        {
            for (size_t format_index = 0;
                 format_index < ALL_FORMATS.size();
                 ++format_index)
            {
                const auto &format = ALL_FORMATS[format_index];
                SCOPED_TRACE(format.name);
                SCOPED_TRACE(std::string("N=") + std::to_string(N));
                auto weights = createWeightsForFormat(format.name, N, K);
                ASSERT_NE(weights, nullptr) << format.name;
                CPUNativeVNNIGemmKernel kernel(weights.get());
                ASSERT_TRUE(kernel.isValid()) << format.name;
                const auto &packed = kernel.packedWeights();
                const int k_blocks = packed.blocks_per_row;
                const std::string full_k_blocks = std::to_string(k_blocks);

                for (int M : runtime_rows)
                {
                    SCOPED_TRACE(std::string("M=") + std::to_string(M));
                    std::mt19937 rng(static_cast<uint32_t>(
                        0xD17Au + format_index * 977u + M * 131u + N));
                    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
                    std::vector<float> input(static_cast<size_t>(M) * K);
                    for (float &value : input)
                        value = distribution(rng);

                    std::vector<Q8_1Block> quantized_rows(
                        static_cast<size_t>(M) * k_blocks);
                    quantize_activations_to_q8_1(
                        input.data(),
                        quantized_rows.data(),
                        M,
                        K,
                        k_blocks);

                    std::vector<float> serial(
                        static_cast<size_t>(M) * N,
                        0.0f);
                    for (int row = 0; row < M; ++row)
                    {
                        gemv_native_vnni_preq(
                            packed,
                            quantized_rows.data() +
                                static_cast<size_t>(row) * k_blocks,
                            serial.data() + static_cast<size_t>(row) * N,
                            ISAPath::AUTO);
                    }

                    for (PrefillSchedulePolicy schedule : {
                             PrefillSchedulePolicy::TwoRowNMajor,
                             PrefillSchedulePolicy::TwoRowPairGrid})
                    {
                        const char *expected_route =
                            schedule == PrefillSchedulePolicy::TwoRowPairGrid
                                ? "two_row_pair_grid"
                                : "two_row_n_major";
                        SCOPED_TRACE(
                            std::string("schedule=") + expected_route);
                        for (int n_block_chunks : n_block_candidates)
                        {
                            SCOPED_TRACE(
                                std::string("n_block_chunks=") +
                                std::to_string(n_block_chunks));
                            const std::string n_block_text =
                                std::to_string(n_block_chunks);
                            ScopedCPUVNNIEnv force_n_blocks(
                                "LLAMINAR_CPU_VNNI_N_BLOCK_CHUNKS",
                                n_block_text.c_str());
                            ScopedCPUVNNIEnv force_full_k(
                                "LLAMINAR_CPU_VNNI_K_TILE_BLOCKS",
                                full_k_blocks.c_str());

                            const size_t logical_values =
                                static_cast<size_t>(M) * N;
                            std::vector<float> guarded_output(
                                guard_values + logical_values + guard_values,
                                guard_value);
                            float *grouped =
                                guarded_output.data() + guard_values;

                            PerfStatsCollector::reset();
                            gemm_native_vnni_preq(
                                packed,
                                quantized_rows.data(),
                                grouped,
                                M,
                                N,
                                ISAPath::AUTO,
                                VerifierRowsPolicy::Pairwise,
                                schedule);

                            const ObservedCPUPrefillRoute route =
                                findObservedCPUPrefillRoute(
                                    M, N, K, packed.codebook_id);
                            ASSERT_TRUE(route.found);
                            EXPECT_EQ(route.route, expected_route);
                            EXPECT_EQ(route.n_block_chunks, n_block_chunks);
                            EXPECT_EQ(route.k_tiles, 1);
                            EXPECT_EQ(route.k_tile_blocks, k_blocks);
                            EXPECT_EQ(route.count, 1u);

                            expectBitwiseEqualFloatRows(
                                format.name + " partial-N full-K prefill M=" +
                                    std::to_string(M) + " N=" +
                                    std::to_string(N) + " nbc=" +
                                    std::to_string(n_block_chunks) +
                                    " schedule=" + expected_route,
                                grouped,
                                serial.data(),
                                logical_values,
                                static_cast<size_t>(N));

                            EXPECT_TRUE(std::all_of(
                                guarded_output.begin(),
                                guarded_output.begin() + guard_values,
                                [guard_value](float value)
                                { return value == guard_value; }))
                                << "Grouped prefill wrote before the logical output";
                            EXPECT_TRUE(std::all_of(
                                guarded_output.begin() + guard_values +
                                    logical_values,
                                guarded_output.end(),
                                [guard_value](float value)
                                { return value == guard_value; }))
                                << "Grouped prefill wrote past the logical output";
                        }
                    }
                }
            }
        };

        for (int N : narrow_tail_ns)
            run_shape(N, short_runtime_rows, narrow_candidates);

        /* Reproduce the exact fresh sealed-v2 failure for every format. */
        const std::array<int, 1> sealed_runtime_rows = {64};
        run_shape(608, sealed_runtime_rows, narrow_candidates);

        /*
         * N=7200 has 113 physical chunks. Even nbc=16 retains eight output
         * tasks on the 28-core blessed host, keeping every learned N-block
         * schedule on the two-row production route while ending at 32 columns.
         */
        run_shape(7200, short_runtime_rows, wide_candidates);

        PerfStatsCollector::reset();
        unsetenv("LLAMINAR_PERF_STATS_JSON");
    }

    /**
     * @test Prove grouped prefill inherits production serial K partitioning.
     *
     * Small-N, long-K projections such as Qwen2.5 3B FFN-down use K-parallel
     * serial decode. A grouped full-K accumulation is deterministic but cannot
     * be byte-identical because it changes FP32 parenthesization. This focused
     * all-format regression uses the real Qwen 1.5B FFN-down geometry, which is
     * one of the finite serial-K-part production shapes exercised by the
     * ordinary-prefill heuristic. The selected Pairwise or WideRows kernel
     * must compute independent tile partials and reduce them in exactly the
     * same order as M individual decode rows.
     */
    TEST_F(CPUNativeVNNIGemvTest,
           NativeVNNIPrefillKPartAllFormatsRuntimeMMatchesSerialDecode)
    {
        constexpr int N = 1536;
        constexpr int K = 8960;
        constexpr std::array<int, 4> runtime_rows = {2, 5, 15, 64};

        setenv(
            "LLAMINAR_PERF_STATS_JSON",
            "/tmp/llaminar_cpu_native_vnni_prefill_kpart.json",
            1);
        PerfStatsCollector::reset();

        for (size_t format_index = 0;
             format_index < ALL_FORMATS.size();
             ++format_index)
        {
            const auto &format = ALL_FORMATS[format_index];
            SCOPED_TRACE(format.name);
            auto weights = createWeightsForFormat(format.name, N, K);
            ASSERT_NE(weights, nullptr) << format.name;
            CPUNativeVNNIGemmKernel kernel(weights.get());
            ASSERT_TRUE(kernel.isValid()) << format.name;
            const auto &packed = kernel.packedWeights();
            const NativeVNNITileConfig serial_config = computeTileConfig(
                N,
                K,
                1,
                packed.payload_bytes,
                omp_get_max_threads());
            ASSERT_GT(serial_config.k_tiles, 1)
                << "regression geometry must exercise serial K partitioning";

            for (int M : runtime_rows)
            {
                SCOPED_TRACE(std::string("M=") + std::to_string(M));
                auto input = TestTensorFactory::createFP32Random(
                    {static_cast<size_t>(M), static_cast<size_t>(K)},
                    -1.0f,
                    1.0f,
                    static_cast<uint32_t>(
                        0x7B00u + format_index * 977u + M * 131u));
                ASSERT_NE(input, nullptr);

                const int k_blocks = packed.blocks_per_row;
                std::vector<Q8_1Block> quantized_rows(
                    static_cast<size_t>(M) * k_blocks);
                quantize_activations_to_q8_1(
                    input->data(),
                    quantized_rows.data(),
                    M,
                    K,
                    k_blocks);
                std::vector<float> serial(static_cast<size_t>(M) * N, 0.0f);
                for (int row = 0; row < M; ++row)
                {
                    gemv_native_vnni_preq(
                        packed,
                        quantized_rows.data() +
                            static_cast<size_t>(row) * k_blocks,
                        serial.data() + static_cast<size_t>(row) * N);
                }

                std::vector<float> grouped(static_cast<size_t>(M) * N, 0.0f);
                ASSERT_TRUE(multiplyViaTensor(
                    kernel,
                    input->data(),
                    grouped.data(),
                    M,
                    N,
                    K));
                expectBitwiseEqualFloatRows(
                    format.name + " production Auto serial-K-part prefill M=" +
                        std::to_string(M),
                    grouped.data(),
                    serial.data(),
                    grouped.size(),
                    static_cast<size_t>(N));

                std::fill(grouped.begin(), grouped.end(), 0.0f);
                gemm_native_vnni_preq(
                    packed,
                    quantized_rows.data(),
                    grouped.data(),
                    M,
                    N,
                    ISAPath::AUTO,
                    VerifierRowsPolicy::Pairwise);
                expectBitwiseEqualFloatRows(
                    format.name + " production serial-K-part prefill M=" +
                        std::to_string(M),
                    grouped.data(),
                    serial.data(),
                    grouped.size(),
                    static_cast<size_t>(N));
            }
        }

        const auto records = PerfStatsCollector::snapshot(
            {"kernel.cpu_native_vnni_prefill_gemm_launch"});
        uint64_t kpart_launches = 0;
        uint64_t pairwise_launches = 0;
        uint64_t wide_rows_launches = 0;
        for (const auto &record : records)
        {
            if (record.domain != "kernel" ||
                record.name != "cpu_native_vnni_prefill_gemm_launch" ||
                record.kind != PerfStatRecord::Kind::Counter)
            {
                continue;
            }
            EXPECT_EQ(
                record.tags.at("route"),
                "decode_equivalent_kpart_rows");
            EXPECT_GT(std::stoi(record.tags.at("k_tiles")), 1);
            EXPECT_TRUE(
                record.tags.at("effective_policy") == "Pairwise" ||
                record.tags.at("effective_policy") == "WideRows");
            if (record.tags.at("effective_policy") == "WideRows")
                wide_rows_launches += record.count;
            else
                pairwise_launches += record.count;
            kpart_launches += record.count;
        }
        const uint64_t cells = ALL_FORMATS.size() * runtime_rows.size();
#if LLAMINAR_COMPILED_WITH_AVX512
        if (activeISALevel() >= ISALevel::AVX512)
        {
            const uint64_t m2_cells = ALL_FORMATS.size();
            EXPECT_EQ(wide_rows_launches, cells - m2_cells)
                << "Every forceable native-AVX512 Auto cell is exhaustively "
                   "trained to the WideRows winner";
            EXPECT_EQ(pairwise_launches, cells + m2_cells)
                << "The diagnostic call exercises Pairwise for every cell, and "
                   "M2 explicitly selects its unique forceable Pairwise route";
        }
        else
#endif
        {
            EXPECT_EQ(wide_rows_launches, 0u);
            EXPECT_EQ(pairwise_launches, cells * 2u)
                << "Pairwise is the unique forceable AVX2 K-part schedule";
        }
        EXPECT_EQ(
            kpart_launches,
            cells * 2u);

        PerfStatsCollector::reset();
        unsetenv("LLAMINAR_PERF_STATS_JSON");
    }

    /**
     * @test Prove the bounded wide-row tile over every runtime M and format.
     *
     * The production selector learns a physical row tile, not a speculative
     * depth specialization. A WideRows decision must therefore compose four-row
     * tiles and 1/2/3-row tails for M=5 and above. This regression forces that
     * candidate through both the ordinary full-K launch and the K-part launch,
     * then compares every FP32 byte with independent production M=1 decode.
     * PerfStats route checks make a normalized Pairwise execution visible.
     */
    TEST_F(CPUNativeVNNIGemvTest,
           MTP_ForcedWideRows_AllFormatsRuntimeMMatchSerialDecodeRows)
    {
        constexpr int N = 130;
        const auto &verifier_rows = kGroupedVerifierRuntimeRows;

        auto run_k_tile_regime = [&](int K, int forced_k_tiles)
        {
            SCOPED_TRACE(
                std::string("K=") + std::to_string(K) +
                " k_tiles=" + std::to_string(forced_k_tiles));
            ScopedCPUVNNIEnv force_k_tiles(
                "LLAMINAR_CPU_VNNI_K_TILES",
                std::to_string(forced_k_tiles).c_str());
            setenv(
                "LLAMINAR_PERF_STATS_JSON",
                "/tmp/llaminar_cpu_native_vnni_forced_wide_runtime_m.json",
                1);
            PerfStatsCollector::reset();

            for (size_t format_index = 0; format_index < ALL_FORMATS.size(); ++format_index)
            {
                const auto &fmt = ALL_FORMATS[format_index];
                SCOPED_TRACE(fmt.name);
                auto weights = createWeightsForFormat(fmt.name, N, K);
                ASSERT_NE(weights, nullptr) << fmt.name;
                CPUNativeVNNIGemmKernel kernel(weights.get());
                ASSERT_TRUE(kernel.isValid()) << fmt.name;
                const auto &packed = kernel.packedWeights();

                for (int M : verifier_rows)
                {
                    SCOPED_TRACE(std::string("M=") + std::to_string(M));
                    auto input = TestTensorFactory::createFP32Random(
                        {static_cast<size_t>(M), static_cast<size_t>(K)},
                        -1.0f,
                        1.0f,
                        static_cast<uint32_t>(
                            0x5100u + M * 131u + K + format_index * 977u));
                    ASSERT_NE(input, nullptr);

                    std::vector<Q8_1Block> quantized_rows(
                        static_cast<size_t>(M) * packed.blocks_per_row);
                    quantize_activations_to_q8_1(
                        input->data(),
                        quantized_rows.data(),
                        M,
                        K,
                        packed.blocks_per_row);

                    std::vector<float> grouped(
                        static_cast<size_t>(M) * N,
                        0.0f);
                    std::vector<float> serial(
                        static_cast<size_t>(M) * N,
                        0.0f);
                    gemm_native_vnni_preq_decode_equivalent_rows(
                        packed,
                        quantized_rows.data(),
                        grouped.data(),
                        M,
                        N,
                        ISAPath::AUTO,
                        VerifierRowsPolicy::WideRows);
                    for (int row = 0; row < M; ++row)
                    {
                        gemv_native_vnni_preq(
                            packed,
                            quantized_rows.data() +
                                static_cast<size_t>(row) * packed.blocks_per_row,
                            serial.data() + static_cast<size_t>(row) * N,
                            ISAPath::AUTO);
                    }
                    expectBitwiseEqualFloatRows(
                        fmt.name + std::string(" forced WideRows M=") +
                            std::to_string(M) + " K=" + std::to_string(K),
                        grouped.data(),
                        serial.data(),
                        grouped.size(),
                        N);
                }
            }

            uint64_t observed_calls = 0;
            uint64_t effective_wide_calls = 0;
            for (const auto &record : PerfStatsCollector::snapshot(
                     {"kernel.cpu_native_vnni_verifier_rows_launch"}))
            {
                if (record.domain != "kernel" ||
                    record.name != "cpu_native_vnni_verifier_rows_launch" ||
                    record.kind != PerfStatRecord::Kind::Counter)
                {
                    continue;
                }
                EXPECT_EQ(record.tags.at("requested_policy"), "WideRows");
                EXPECT_EQ(
                    record.tags.at("k_tiles"),
                    std::to_string(forced_k_tiles));
                const int tagged_m = std::stoi(record.tags.at("m"));
                const bool should_be_wide =
                    activeISALevel() == ISALevel::AVX512 && tagged_m >= 3;
                EXPECT_EQ(
                    record.tags.at("effective_policy"),
                    should_be_wide ? "WideRows" : "Pairwise");
                observed_calls += record.count;
                if (record.tags.at("effective_policy") == "WideRows")
                    effective_wide_calls += record.count;
            }
            EXPECT_EQ(
                observed_calls,
                ALL_FORMATS.size() * verifier_rows.size());
            if (activeISALevel() == ISALevel::AVX512)
            {
                EXPECT_EQ(
                    effective_wide_calls,
                    ALL_FORMATS.size() * (verifier_rows.size() - 1));
            }

            PerfStatsCollector::reset();
            unsetenv("LLAMINAR_PERF_STATS_JSON");
        };

        run_k_tile_regime(/*K=*/256, /*forced_k_tiles=*/1);
        run_k_tile_regime(/*K=*/4096, /*forced_k_tiles=*/4);
    }

    TEST_F(CPUNativeVNNIGemvTest, MTP_FusedVerifierKParallel_AllFormatsMatchSerialDecodeRows)
    {
        constexpr int K = 4096;
        constexpr int N0 = 384;
        constexpr int N1 = 320;
        const auto &verifier_rows = kGroupedVerifierRuntimeRows;

        ScopedCPUVNNIEnv force_k_tiles("LLAMINAR_CPU_VNNI_K_TILES", "4");
        setenv("LLAMINAR_PERF_STATS_JSON", "/tmp/llaminar_cpu_native_vnni_kparallel_grouped_verifier.json", 1);
        PerfStatsCollector::reset();
        ASSERT_EQ(debugEnv().cpu_vnni.k_tiles, 4)
            << "This regression must force the CPU NativeVNNI K-parallel "
               "grouped verifier lane";

        for (const auto &fmt : ALL_FORMATS)
        {
            SCOPED_TRACE(fmt.name);

            auto weights0 = createWeightsForFormat(fmt.name, N0, K);
            auto weights1 = createWeightsForFormat(fmt.name, N1, K);
            ASSERT_NE(weights0, nullptr) << fmt.name << " projection0 weights";
            ASSERT_NE(weights1, nullptr) << fmt.name << " projection1 weights";

            CPUNativeVNNIGemmKernel kernel0(weights0.get());
            CPUNativeVNNIGemmKernel kernel1(weights1.get());
            ASSERT_TRUE(kernel0.isValid()) << fmt.name << " projection0 pack";
            ASSERT_TRUE(kernel1.isValid()) << fmt.name << " projection1 pack";

            for (int M : verifier_rows)
            {
                SCOPED_TRACE(std::string("M=") + std::to_string(M));

                auto input = TestTensorFactory::createFP32Random(
                    {static_cast<size_t>(M), static_cast<size_t>(K)},
                    -0.75f,
                    0.75f,
                    static_cast<uint32_t>(5100 + M * 17 + fmt.name.size() * 131));
                ASSERT_NE(input, nullptr);

                FP32Tensor grouped0({static_cast<size_t>(M), static_cast<size_t>(N0)});
                FP32Tensor grouped1({static_cast<size_t>(M), static_cast<size_t>(N1)});
                std::vector<ITensorGemm::TensorProjectionDesc> projections = {
                    {&kernel0, &grouped0, N0, nullptr, "mtp_verifier_kparallel_projection0"},
                    {&kernel1, &grouped1, N1, nullptr, "mtp_verifier_kparallel_projection1"}};

                ASSERT_TRUE(kernel0.multiply_fused_verifier_rows_decode_equivalent(
                    input.get(),
                    projections,
                    M,
                    K))
                    << fmt.name << " K-parallel grouped verifier projections failed";

                std::vector<float> serial0(
                    static_cast<size_t>(M) * static_cast<size_t>(N0),
                    0.0f);
                std::vector<float> serial1(
                    static_cast<size_t>(M) * static_cast<size_t>(N1),
                    0.0f);
                for (int row = 0; row < M; ++row)
                {
                    ASSERT_TRUE(multiplyViaTensor(
                        kernel0,
                        input->data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                        serial0.data() + static_cast<size_t>(row) * static_cast<size_t>(N0),
                        1,
                        N0,
                        K))
                        << fmt.name << " projection0 serial decode failed at M=" << M
                        << " row=" << row;
                    ASSERT_TRUE(multiplyViaTensor(
                        kernel1,
                        input->data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                        serial1.data() + static_cast<size_t>(row) * static_cast<size_t>(N1),
                        1,
                        N1,
                        K))
                        << fmt.name << " projection1 serial decode failed at M=" << M
                        << " row=" << row;
                }

                expectBitwiseEqualFloatRows(
                    fmt.name + std::string(" CPU K-parallel grouped verifier projection0 M=") +
                        std::to_string(M),
                    grouped0.data(),
                    serial0.data(),
                    serial0.size(),
                    static_cast<size_t>(N0));
                expectBitwiseEqualFloatRows(
                    fmt.name + std::string(" CPU K-parallel grouped verifier projection1 M=") +
                        std::to_string(M),
                    grouped1.data(),
                    serial1.data(),
                    serial1.size(),
                    static_cast<size_t>(N1));
            }
        }

        const auto records = PerfStatsCollector::snapshot({
            "kernel.cpu_native_vnni_fused_grouped_verifier_projection_calls",
            "kernel.cpu_native_vnni_fused_verifier_rows_projection_launch"});
        uint64_t fused_grouped_verifier_calls = 0;
        uint64_t grouped_k_parallel_projection_launches = 0;
        for (const auto &record : records)
        {
            if (record.domain == "kernel" &&
                record.name == "cpu_native_vnni_fused_grouped_verifier_projection_calls" &&
                record.kind == PerfStatRecord::Kind::Counter)
            {
                EXPECT_EQ(record.tags.at("k"), std::to_string(K));
                EXPECT_EQ(record.tags.at("projections"), "2");
                fused_grouped_verifier_calls += record.count;
            }
            if (record.domain == "kernel" &&
                record.name ==
                    "cpu_native_vnni_fused_verifier_rows_projection_launch" &&
                record.kind == PerfStatRecord::Kind::Counter)
            {
                EXPECT_EQ(record.tags.at("k"), std::to_string(K));
                EXPECT_EQ(record.tags.at("k_tiles"), "4");
                EXPECT_EQ(
                    record.tags.at("physical_row_tile"),
                    activeISALevel() == ISALevel::AVX512 ? "4" : "2");
                EXPECT_EQ(
                    record.tags.at("isa"),
                    activeISALevel() == ISALevel::AVX512 ? "AVX512" : "AVX2");
                EXPECT_EQ(
                    record.tags.at("route"),
                    "grouped_k_parallel_row_tiles");
                const int tagged_m = std::stoi(record.tags.at("m"));
                EXPECT_TRUE((tagged_m >= 2 && tagged_m <= 16) || tagged_m == 31);
                grouped_k_parallel_projection_launches += record.count;
            }
        }
        EXPECT_EQ(
            fused_grouped_verifier_calls,
            ALL_FORMATS.size() * verifier_rows.size())
            << "Every CPU NativeVNNI format and certified runtime-M shape must enter the "
               "fused grouped verifier route under forced K-parallel tiling";
        EXPECT_EQ(
            grouped_k_parallel_projection_launches,
            ALL_FORMATS.size() * verifier_rows.size() * 2)
            << "Each projection must use a shared physical row tile; a one-row "
               "K-part scheduler is not an economical grouped verifier path";

        PerfStatsCollector::reset();
        unsetenv("LLAMINAR_PERF_STATS_JSON");
    }

#ifdef HAVE_ONEDNN
    TEST_F(CPUNativeVNNIGemvTest, MTP_FP32FloatingVerifierRowsMatchSerialDecodeAndUseGroupedCounters)
    {
        const int K = 256;
        const int N0 = 192;
        const int N1 = 128;
        const int NDown = 160;
        const auto &verifier_rows = kGroupedVerifierRuntimeRows;

        setenv("LLAMINAR_PERF_STATS_JSON", "/tmp/llaminar_cpu_fp32_grouped_verifier.json", 1);
        PerfStatsCollector::reset();

        auto weights0 = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(N0), static_cast<size_t>(K)}, -0.2f, 0.2f, 8101);
        auto weights1 = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(N1), static_cast<size_t>(K)}, -0.2f, 0.2f, 8102);
        auto weights_down = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(NDown), static_cast<size_t>(K)}, -0.2f, 0.2f, 8103);
        ASSERT_NE(weights0, nullptr);
        ASSERT_NE(weights1, nullptr);
        ASSERT_NE(weights_down, nullptr);

        gemm::FloatingPointGemmKernel kernel0(weights0.get());
        gemm::FloatingPointGemmKernel kernel1(weights1.get());
        gemm::FloatingPointGemmKernel down_kernel(weights_down.get());

        for (int M : verifier_rows)
        {
            SCOPED_TRACE(std::string("FP32 M=") + std::to_string(M));
            auto input = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(M), static_cast<size_t>(K)}, -0.75f, 0.75f,
                static_cast<uint32_t>(8200 + M));
            ASSERT_NE(input, nullptr);

            FP32Tensor grouped0({static_cast<size_t>(M), static_cast<size_t>(N0)});
            FP32Tensor grouped1({static_cast<size_t>(M), static_cast<size_t>(N1)});
            std::vector<ITensorGemm::TensorProjectionDesc> grouped_projections = {
                {&kernel0, &grouped0, N0, nullptr, "fp32_proj0"},
                {&kernel1, &grouped1, N1, nullptr, "fp32_proj1"}};
            ASSERT_TRUE(kernel0.multiply_fused_verifier_rows_decode_equivalent(
                input.get(), grouped_projections, M, K))
                << "CPU FP32 grouped verifier projection failed";

            std::vector<float> serial0(static_cast<size_t>(M) * static_cast<size_t>(N0), 0.0f);
            std::vector<float> serial1(static_cast<size_t>(M) * static_cast<size_t>(N1), 0.0f);
            for (int row = 0; row < M; ++row)
            {
                FP32Tensor row_input({size_t{1}, static_cast<size_t>(K)});
                std::memcpy(
                    row_input.mutable_data(),
                    input->data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                    static_cast<size_t>(K) * sizeof(float));
                FP32Tensor row0({size_t{1}, static_cast<size_t>(N0)});
                FP32Tensor row1({size_t{1}, static_cast<size_t>(N1)});
                std::vector<ITensorGemm::TensorProjectionDesc> row_projections = {
                    {&kernel0, &row0, N0, nullptr, "fp32_proj0_serial"},
                    {&kernel1, &row1, N1, nullptr, "fp32_proj1_serial"}};

                ASSERT_TRUE(kernel0.multiply_fused_tensor(&row_input, row_projections, 1, K))
                    << "CPU FP32 serial decode projection failed for row " << row;
                std::memcpy(
                    serial0.data() + static_cast<size_t>(row) * static_cast<size_t>(N0),
                    row0.data(),
                    static_cast<size_t>(N0) * sizeof(float));
                std::memcpy(
                    serial1.data() + static_cast<size_t>(row) * static_cast<size_t>(N1),
                    row1.data(),
                    static_cast<size_t>(N1) * sizeof(float));
            }

            EXPECT_EQ(
                std::memcmp(grouped0.data(), serial0.data(), serial0.size() * sizeof(float)),
                0)
                << "CPU FP32 projection 0 grouped verifier rows must be bitwise serial-decode equivalent";
            EXPECT_EQ(
                std::memcmp(grouped1.data(), serial1.data(), serial1.size() * sizeof(float)),
                0)
                << "CPU FP32 projection 1 grouped verifier rows must be bitwise serial-decode equivalent";

            auto gate = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(M), static_cast<size_t>(K)}, -0.6f, 0.6f,
                static_cast<uint32_t>(8300 + M));
            auto up = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(M), static_cast<size_t>(K)}, -0.6f, 0.6f,
                static_cast<uint32_t>(8400 + M));
            ASSERT_NE(gate, nullptr);
            ASSERT_NE(up, nullptr);

            FP32Tensor grouped_down({static_cast<size_t>(M), static_cast<size_t>(NDown)});
            ASSERT_TRUE(down_kernel.multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
                gate.get(), up.get(), &grouped_down, M, NDown, K))
                << "CPU FP32 grouped verifier SwiGLU-down projection failed";

            std::vector<float> serial_down(static_cast<size_t>(M) * static_cast<size_t>(NDown), 0.0f);
            std::vector<float> swiglu_row(static_cast<size_t>(K), 0.0f);
            for (int row = 0; row < M; ++row)
            {
                primitives::compute_swiglu(
                    gate->data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                    up->data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                    swiglu_row.data(),
                    K);
                FP32Tensor row_swiglu({size_t{1}, static_cast<size_t>(K)});
                std::memcpy(row_swiglu.mutable_data(), swiglu_row.data(), static_cast<size_t>(K) * sizeof(float));
                FP32Tensor row_down({size_t{1}, static_cast<size_t>(NDown)});
                ASSERT_TRUE(down_kernel.multiply_tensor(&row_swiglu, &row_down, 1, NDown, K))
                    << "CPU FP32 serial SwiGLU-down projection failed for row " << row;
                std::memcpy(
                    serial_down.data() + static_cast<size_t>(row) * static_cast<size_t>(NDown),
                    row_down.data(),
                    static_cast<size_t>(NDown) * sizeof(float));
            }

            EXPECT_EQ(
                std::memcmp(grouped_down.data(), serial_down.data(), serial_down.size() * sizeof(float)),
                0)
                << "CPU FP32 grouped SwiGLU-down rows must be bitwise serial-decode equivalent";
        }

        uint64_t projection_calls = 0;
        for (const auto &record : PerfStatsCollector::snapshot({"kernel.cpu_fp32_grouped_verifier_projection_calls"}))
        {
            if (record.domain == "kernel" &&
                record.name == "cpu_fp32_grouped_verifier_projection_calls" &&
                record.kind == PerfStatRecord::Kind::Counter)
            {
                EXPECT_EQ(record.tags.at("k"), std::to_string(K));
                EXPECT_EQ(record.tags.at("projections"), "2");
                projection_calls += record.count;
            }
        }
        EXPECT_EQ(projection_calls, verifier_rows.size())
            << "CPU FP32 projection verifier must enter the grouped path for every runtime M";

        uint64_t swiglu_calls = 0;
        for (const auto &record : PerfStatsCollector::snapshot({"kernel.cpu_fp32_grouped_verifier_swiglu_down_calls"}))
        {
            if (record.domain == "kernel" &&
                record.name == "cpu_fp32_grouped_verifier_swiglu_down_calls" &&
                record.kind == PerfStatRecord::Kind::Counter)
            {
                EXPECT_EQ(record.tags.at("k"), std::to_string(K));
                EXPECT_EQ(record.tags.at("n"), std::to_string(NDown));
                swiglu_calls += record.count;
            }
        }
        EXPECT_EQ(swiglu_calls, verifier_rows.size())
            << "CPU FP32 SwiGLU-down verifier must enter the grouped path for every runtime M";

        PerfStatsCollector::reset();
        unsetenv("LLAMINAR_PERF_STATS_JSON");
    }

    TEST_F(CPUNativeVNNIGemvTest, MTP_FP16BF16FloatingVerifierRowsMatchSerialDecodeAndUseGroupedCounters)
    {
        /**
         * MTP publication can only trust grouped verifier rows when every row
         * is bitwise identical to the backend's own one-row decode contract.
         * This regression sweeps the two 16-bit floating tensor formats through
         * the CPU floating GEMM grouped verifier hook.  The hook must execute a
         * real grouped implementation and report its perfstats counter; a
         * hidden serial replay or unsupported false return would fail here.
         */
        const int K = 192;
        const int N0 = 96;
        const int N1 = 64;
        const int NDown = 80;
        const auto &verifier_rows = kGroupedVerifierRuntimeRows;

        setenv("LLAMINAR_PERF_STATS_JSON", "/tmp/llaminar_cpu_fp16_bf16_grouped_verifier.json", 1);
        PerfStatsCollector::reset();

        auto run_fp16_case = [&]()
        {
            auto weights0 = TestTensorFactory::createFP16Random(
                {static_cast<size_t>(N0), static_cast<size_t>(K)}, -0.25f, 0.25f, 9101);
            auto weights1 = TestTensorFactory::createFP16Random(
                {static_cast<size_t>(N1), static_cast<size_t>(K)}, -0.25f, 0.25f, 9102);
            auto weights_down = TestTensorFactory::createFP16Random(
                {static_cast<size_t>(NDown), static_cast<size_t>(K)}, -0.25f, 0.25f, 9103);
            ASSERT_NE(weights0, nullptr);
            ASSERT_NE(weights1, nullptr);
            ASSERT_NE(weights_down, nullptr);

            gemm::FloatingPointGemmKernel kernel0(weights0.get());
            gemm::FloatingPointGemmKernel kernel1(weights1.get());
            gemm::FloatingPointGemmKernel down_kernel(weights_down.get());

            for (int M : verifier_rows)
            {
                SCOPED_TRACE(std::string("FP16 M=") + std::to_string(M));
                auto input = TestTensorFactory::createFP16Random(
                    {static_cast<size_t>(M), static_cast<size_t>(K)}, -0.75f, 0.75f,
                    static_cast<uint32_t>(9200 + M));
                ASSERT_NE(input, nullptr);

                FP32Tensor grouped0({static_cast<size_t>(M), static_cast<size_t>(N0)});
                FP32Tensor grouped1({static_cast<size_t>(M), static_cast<size_t>(N1)});
                std::vector<ITensorGemm::TensorProjectionDesc> grouped_projections = {
                    {&kernel0, &grouped0, N0, nullptr, "fp16_proj0"},
                    {&kernel1, &grouped1, N1, nullptr, "fp16_proj1"}};
                ASSERT_TRUE(kernel0.multiply_fused_verifier_rows_decode_equivalent(
                    input.get(), grouped_projections, M, K));

                std::vector<float> serial0(static_cast<size_t>(M) * static_cast<size_t>(N0), 0.0f);
                std::vector<float> serial1(static_cast<size_t>(M) * static_cast<size_t>(N1), 0.0f);
                for (int row = 0; row < M; ++row)
                {
                    std::vector<uint16_t> row_bits(static_cast<size_t>(K));
                    std::memcpy(
                        row_bits.data(),
                        input->typed_data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                        static_cast<size_t>(K) * sizeof(uint16_t));
                    FP16Tensor row_input({size_t{1}, static_cast<size_t>(K)}, row_bits);
                    FP32Tensor row0({size_t{1}, static_cast<size_t>(N0)});
                    FP32Tensor row1({size_t{1}, static_cast<size_t>(N1)});

                    ASSERT_TRUE(kernel0.multiply_tensor(&row_input, &row0, 1, N0, K));
                    ASSERT_TRUE(kernel1.multiply_tensor(&row_input, &row1, 1, N1, K));
                    std::memcpy(
                        serial0.data() + static_cast<size_t>(row) * static_cast<size_t>(N0),
                        row0.data(),
                        static_cast<size_t>(N0) * sizeof(float));
                    std::memcpy(
                        serial1.data() + static_cast<size_t>(row) * static_cast<size_t>(N1),
                        row1.data(),
                        static_cast<size_t>(N1) * sizeof(float));
                }

                EXPECT_EQ(std::memcmp(grouped0.data(), serial0.data(), serial0.size() * sizeof(float)), 0);
                EXPECT_EQ(std::memcmp(grouped1.data(), serial1.data(), serial1.size() * sizeof(float)), 0);

                auto gate = TestTensorFactory::createFP32Random(
                    {static_cast<size_t>(M), static_cast<size_t>(K)}, -0.6f, 0.6f,
                    static_cast<uint32_t>(9500 + M));
                auto up = TestTensorFactory::createFP32Random(
                    {static_cast<size_t>(M), static_cast<size_t>(K)}, -0.6f, 0.6f,
                    static_cast<uint32_t>(9600 + M));
                ASSERT_NE(gate, nullptr);
                ASSERT_NE(up, nullptr);

                FP32Tensor grouped_down({static_cast<size_t>(M), static_cast<size_t>(NDown)});
                ASSERT_TRUE(down_kernel.multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
                    gate.get(), up.get(), &grouped_down, M, NDown, K))
                    << "CPU FP16 grouped verifier SwiGLU/down failed";

                std::vector<float> serial_down(static_cast<size_t>(M) * static_cast<size_t>(NDown), 0.0f);
                for (int row = 0; row < M; ++row)
                {
                    FP32Tensor gate_row({size_t{1}, static_cast<size_t>(K)});
                    FP32Tensor up_row({size_t{1}, static_cast<size_t>(K)});
                    FP32Tensor down_row({size_t{1}, static_cast<size_t>(NDown)});
                    std::memcpy(
                        gate_row.mutable_data(),
                        gate->data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                        static_cast<size_t>(K) * sizeof(float));
                    std::memcpy(
                        up_row.mutable_data(),
                        up->data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                        static_cast<size_t>(K) * sizeof(float));
                    ASSERT_TRUE(down_kernel.multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
                        &gate_row, &up_row, &down_row, 1, NDown, K))
                        << "CPU FP16 serial verifier SwiGLU/down failed for row " << row;
                    std::memcpy(
                        serial_down.data() + static_cast<size_t>(row) * static_cast<size_t>(NDown),
                        down_row.data(),
                        static_cast<size_t>(NDown) * sizeof(float));
                }

                EXPECT_EQ(
                    std::memcmp(grouped_down.data(), serial_down.data(), serial_down.size() * sizeof(float)),
                    0)
                    << "CPU FP16 grouped SwiGLU/down rows must be bitwise serial-decode equivalent";
            }
        };

        auto run_bf16_case = [&]()
        {
            auto weights0 = TestTensorFactory::createBF16Random(
                {static_cast<size_t>(N0), static_cast<size_t>(K)}, -0.25f, 0.25f, 9301);
            auto weights1 = TestTensorFactory::createBF16Random(
                {static_cast<size_t>(N1), static_cast<size_t>(K)}, -0.25f, 0.25f, 9302);
            auto weights_down = TestTensorFactory::createBF16Random(
                {static_cast<size_t>(NDown), static_cast<size_t>(K)}, -0.25f, 0.25f, 9303);
            ASSERT_NE(weights0, nullptr);
            ASSERT_NE(weights1, nullptr);
            ASSERT_NE(weights_down, nullptr);

            gemm::FloatingPointGemmKernel kernel0(weights0.get());
            gemm::FloatingPointGemmKernel kernel1(weights1.get());
            gemm::FloatingPointGemmKernel down_kernel(weights_down.get());

            for (int M : verifier_rows)
            {
                SCOPED_TRACE(std::string("BF16 M=") + std::to_string(M));
                auto input = TestTensorFactory::createBF16Random(
                    {static_cast<size_t>(M), static_cast<size_t>(K)}, -0.75f, 0.75f,
                    static_cast<uint32_t>(9400 + M));
                ASSERT_NE(input, nullptr);

                FP32Tensor grouped0({static_cast<size_t>(M), static_cast<size_t>(N0)});
                FP32Tensor grouped1({static_cast<size_t>(M), static_cast<size_t>(N1)});
                std::vector<ITensorGemm::TensorProjectionDesc> grouped_projections = {
                    {&kernel0, &grouped0, N0, nullptr, "bf16_proj0"},
                    {&kernel1, &grouped1, N1, nullptr, "bf16_proj1"}};
                ASSERT_TRUE(kernel0.multiply_fused_verifier_rows_decode_equivalent(
                    input.get(), grouped_projections, M, K));

                std::vector<float> serial0(static_cast<size_t>(M) * static_cast<size_t>(N0), 0.0f);
                std::vector<float> serial1(static_cast<size_t>(M) * static_cast<size_t>(N1), 0.0f);
                for (int row = 0; row < M; ++row)
                {
                    std::vector<uint16_t> row_bits(static_cast<size_t>(K));
                    std::memcpy(
                        row_bits.data(),
                        input->typed_data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                        static_cast<size_t>(K) * sizeof(uint16_t));
                    BF16Tensor row_input({size_t{1}, static_cast<size_t>(K)}, row_bits);
                    FP32Tensor row0({size_t{1}, static_cast<size_t>(N0)});
                    FP32Tensor row1({size_t{1}, static_cast<size_t>(N1)});

                    ASSERT_TRUE(kernel0.multiply_tensor(&row_input, &row0, 1, N0, K));
                    ASSERT_TRUE(kernel1.multiply_tensor(&row_input, &row1, 1, N1, K));
                    std::memcpy(
                        serial0.data() + static_cast<size_t>(row) * static_cast<size_t>(N0),
                        row0.data(),
                        static_cast<size_t>(N0) * sizeof(float));
                    std::memcpy(
                        serial1.data() + static_cast<size_t>(row) * static_cast<size_t>(N1),
                        row1.data(),
                        static_cast<size_t>(N1) * sizeof(float));
                }

                EXPECT_EQ(std::memcmp(grouped0.data(), serial0.data(), serial0.size() * sizeof(float)), 0);
                EXPECT_EQ(std::memcmp(grouped1.data(), serial1.data(), serial1.size() * sizeof(float)), 0);

                auto gate = TestTensorFactory::createFP32Random(
                    {static_cast<size_t>(M), static_cast<size_t>(K)}, -0.6f, 0.6f,
                    static_cast<uint32_t>(9700 + M));
                auto up = TestTensorFactory::createFP32Random(
                    {static_cast<size_t>(M), static_cast<size_t>(K)}, -0.6f, 0.6f,
                    static_cast<uint32_t>(9800 + M));
                ASSERT_NE(gate, nullptr);
                ASSERT_NE(up, nullptr);

                FP32Tensor grouped_down({static_cast<size_t>(M), static_cast<size_t>(NDown)});
                ASSERT_TRUE(down_kernel.multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
                    gate.get(), up.get(), &grouped_down, M, NDown, K))
                    << "CPU BF16 grouped verifier SwiGLU/down failed";

                std::vector<float> serial_down(static_cast<size_t>(M) * static_cast<size_t>(NDown), 0.0f);
                for (int row = 0; row < M; ++row)
                {
                    FP32Tensor gate_row({size_t{1}, static_cast<size_t>(K)});
                    FP32Tensor up_row({size_t{1}, static_cast<size_t>(K)});
                    FP32Tensor down_row({size_t{1}, static_cast<size_t>(NDown)});
                    std::memcpy(
                        gate_row.mutable_data(),
                        gate->data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                        static_cast<size_t>(K) * sizeof(float));
                    std::memcpy(
                        up_row.mutable_data(),
                        up->data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                        static_cast<size_t>(K) * sizeof(float));
                    ASSERT_TRUE(down_kernel.multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
                        &gate_row, &up_row, &down_row, 1, NDown, K))
                        << "CPU BF16 serial verifier SwiGLU/down failed for row " << row;
                    std::memcpy(
                        serial_down.data() + static_cast<size_t>(row) * static_cast<size_t>(NDown),
                        down_row.data(),
                        static_cast<size_t>(NDown) * sizeof(float));
                }

                EXPECT_EQ(
                    std::memcmp(grouped_down.data(), serial_down.data(), serial_down.size() * sizeof(float)),
                    0)
                    << "CPU BF16 grouped SwiGLU/down rows must be bitwise serial-decode equivalent";
            }
        };

        run_fp16_case();
        run_bf16_case();

        uint64_t fp16_calls = 0;
        uint64_t bf16_calls = 0;
        for (const auto &record : PerfStatsCollector::snapshot({"kernel.cpu_floating_grouped_verifier_projection_calls"}))
        {
            if (record.domain != "kernel" ||
                record.name != "cpu_floating_grouped_verifier_projection_calls" ||
                record.kind != PerfStatRecord::Kind::Counter)
            {
                continue;
            }
            EXPECT_EQ(record.tags.at("k"), std::to_string(K));
            EXPECT_EQ(record.tags.at("projections"), "2");
            if (record.tags.at("dtype") == "fp16")
                fp16_calls += record.count;
            else if (record.tags.at("dtype") == "bf16")
                bf16_calls += record.count;
        }
        EXPECT_EQ(fp16_calls, verifier_rows.size());
        EXPECT_EQ(bf16_calls, verifier_rows.size());

        uint64_t fp16_swiglu_calls = 0;
        uint64_t bf16_swiglu_calls = 0;
        for (const auto &record : PerfStatsCollector::snapshot({"kernel.cpu_floating_grouped_verifier_swiglu_down_calls"}))
        {
            if (record.domain != "kernel" ||
                record.name != "cpu_floating_grouped_verifier_swiglu_down_calls" ||
                record.kind != PerfStatRecord::Kind::Counter ||
                record.tags.at("n") != std::to_string(NDown) ||
                record.tags.at("k") != std::to_string(K))
            {
                continue;
            }
            const int m_tag = std::stoi(record.tags.at("m"));
            if (!((m_tag >= 2 && m_tag <= 16) || m_tag == 31))
                continue;
            if (record.tags.at("dtype") == "fp16")
                fp16_swiglu_calls += record.count;
            else if (record.tags.at("dtype") == "bf16")
                bf16_swiglu_calls += record.count;
        }
        EXPECT_EQ(fp16_swiglu_calls, verifier_rows.size());
        EXPECT_EQ(bf16_swiglu_calls, verifier_rows.size());

        PerfStatsCollector::reset();
        unsetenv("LLAMINAR_PERF_STATS_JSON");
    }
#endif

    TEST_F(CPUNativeVNNIGemvTest, MTP_VerifierPolicyIsTotalAcrossPositiveThreadRegimes)
    {
        /**
         * Exact overlays remain scoped to their measured OpenMP width, while
         * the learned generic tree evaluates wave geometry with the positive
         * runtime team size. Exercise non-power-of-two, measured, physical
         * core, SMT-sized, and oversized teams through the production resolver.
         */
        const int N = 128;
        const int K = 256;
        const int M = 3;

        auto weights = createWeightsForFormat("Q8_K", N, K);
        ASSERT_NE(weights, nullptr);

        CPUNativeVNNIGemmKernel kernel(weights.get());
        ASSERT_TRUE(kernel.isValid());
        const auto &packed = kernel.packedWeights();
        const int original_threads = omp_get_max_threads();
        for (const int threads : {1, 3, 7, 27, 28, 31, 56, 112})
        {
            SCOPED_TRACE(::testing::Message() << "threads=" << threads);
            omp_set_num_threads(threads);
            EXPECT_NO_THROW((void)selectVerifierRowsPolicy(packed, M, N, K));
        }
        omp_set_num_threads(original_threads);
    }

    TEST_F(CPUNativeVNNIGemvTest, MTP_VerifierRoutePublishesBuildAndRuntimeISA)
    {
        /**
         * The trainer treats build ISA and effective runtime ISA as policy-key
         * dimensions. This regression exercises the production grouped entry
         * point, checks byte equality against serial M=1 rows, and then proves
         * the route counter publishes the same dimensions consumed by the
         * strong-evidence adapter.
         */
        constexpr int M = 3;
        constexpr int N = 128;
        constexpr int K = 256;
        auto weights = createWeightsForFormat("Q8_K", N, K);
        ASSERT_NE(weights, nullptr);
        CPUNativeVNNIGemmKernel kernel(weights.get());
        ASSERT_TRUE(kernel.isValid());
        const auto &packed = kernel.packedWeights();

        std::mt19937 rng(0x15A5u);
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        std::vector<float> input(static_cast<size_t>(M) * K);
        for (float &value : input)
            value = distribution(rng);

        std::vector<Q8_1Block> quantized(
            static_cast<size_t>(M) * packed.blocks_per_row);
        quantize_activations_to_q8_1(
            input.data(),
            quantized.data(),
            M,
            K,
            packed.blocks_per_row);
        std::vector<float> grouped(static_cast<size_t>(M) * N, 0.0f);
        std::vector<float> serial(static_cast<size_t>(M) * N, 0.0f);

        setenv("LLAMINAR_PERF_STATS_JSON", "1", 1);
        PerfStatsCollector::reset();
        gemm_native_vnni_preq_decode_equivalent_rows(
            packed,
            quantized.data(),
            grouped.data(),
            M,
            N,
            ISAPath::AUTO,
            VerifierRowsPolicy::WideRows);
        for (int row = 0; row < M; ++row)
        {
            gemv_native_vnni_preq(
                packed,
                quantized.data() +
                    static_cast<size_t>(row) * packed.blocks_per_row,
                serial.data() + static_cast<size_t>(row) * N,
                ISAPath::AUTO);
        }
        expectBitwiseEqualFloatRows(
            "CPU ISA-aware verifier telemetry probe",
            grouped.data(),
            serial.data(),
            grouped.size(),
            N);

        const char *expected_runtime_isa =
            activeISALevel() == ISALevel::AVX512 ? "AVX512" : "AVX2";
        const char *expected_policy =
            activeISALevel() == ISALevel::AVX512 ? "WideRows" : "Pairwise";
        uint64_t route_count = 0;
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"kernel.cpu_native_vnni_verifier_rows_launch"}))
        {
            if (record.name != "cpu_native_vnni_verifier_rows_launch")
                continue;
            EXPECT_EQ(record.tags.at("build_isa"),
                      compiledNativeVNNIBuildISAName());
            EXPECT_EQ(record.tags.at("isa"), expected_runtime_isa);
            EXPECT_EQ(record.tags.at("requested_policy"), "WideRows");
            EXPECT_EQ(record.tags.at("effective_policy"), expected_policy);
            EXPECT_EQ(record.tags.at("threads"),
                      std::to_string(omp_get_max_threads()));
            route_count += record.count;
        }
        EXPECT_EQ(route_count, 1u);
        PerfStatsCollector::reset();
        unsetenv("LLAMINAR_PERF_STATS_JSON");
    }

    TEST_F(CPUNativeVNNIGemvTest, MTP_FusedProjectionWithActivationRotationMatchesSerialDecodeRows)
    {
        const int K = 128;
        const int M = 2;
        const int N0 = 384;
        const int N1 = 256;

        ActivationRotation rotation(K, 128);
        auto weights0 = createWeightsForFormat("Q4_0", N0, K);
        auto weights1 = createWeightsForFormat("Q4_0", N1, K);
        ASSERT_NE(weights0, nullptr);
        ASSERT_NE(weights1, nullptr);
        weights0->setActivationRotation(&rotation);
        weights1->setActivationRotation(&rotation);

        CPUNativeVNNIGemmKernel kernel0(weights0.get());
        CPUNativeVNNIGemmKernel kernel1(weights1.get());
        ASSERT_TRUE(kernel0.isValid());
        ASSERT_TRUE(kernel1.isValid());

        auto input = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(M), static_cast<size_t>(K)}, -1.0f, 1.0f, 3101);
        ASSERT_NE(input, nullptr);

        FP32Tensor grouped0({static_cast<size_t>(M), static_cast<size_t>(N0)});
        FP32Tensor grouped1({static_cast<size_t>(M), static_cast<size_t>(N1)});
        std::vector<ITensorGemm::TensorProjectionDesc> grouped_projections = {
            {&kernel0, &grouped0, N0, nullptr, "proj0"},
            {&kernel1, &grouped1, N1, nullptr, "proj1"}};

        ASSERT_TRUE(kernel0.multiply_fused_verifier_rows_decode_equivalent(
            input.get(), grouped_projections, M, K))
            << "Grouped verifier projection should support rotated CPU NativeVNNI weights";

        std::vector<float> serial0(static_cast<size_t>(M) * static_cast<size_t>(N0), 0.0f);
        std::vector<float> serial1(static_cast<size_t>(M) * static_cast<size_t>(N1), 0.0f);
        for (int row = 0; row < M; ++row)
        {
            FP32Tensor row_input({1, static_cast<size_t>(K)});
            std::memcpy(
                row_input.mutable_data(),
                input->data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                static_cast<size_t>(K) * sizeof(float));

            FP32Tensor row0({1, static_cast<size_t>(N0)});
            FP32Tensor row1({1, static_cast<size_t>(N1)});
            std::vector<ITensorGemm::TensorProjectionDesc> row_projections = {
                {&kernel0, &row0, N0, nullptr, "proj0"},
                {&kernel1, &row1, N1, nullptr, "proj1"}};

            ASSERT_TRUE(kernel0.multiply_fused_tensor(&row_input, row_projections, 1, K))
                << "Serial decode projection failed for row " << row;
            std::memcpy(
                serial0.data() + static_cast<size_t>(row) * static_cast<size_t>(N0),
                row0.data(),
                static_cast<size_t>(N0) * sizeof(float));
            std::memcpy(
                serial1.data() + static_cast<size_t>(row) * static_cast<size_t>(N1),
                row1.data(),
                static_cast<size_t>(N1) * sizeof(float));
        }

        expectBitwiseEqualFloatRows(
            "CPU rotated grouped verifier projection 0",
            grouped0.data(),
            serial0.data(),
            static_cast<size_t>(M) * static_cast<size_t>(N0),
            static_cast<size_t>(N0));
        expectBitwiseEqualFloatRows(
            "CPU rotated grouped verifier projection 1",
            grouped1.data(),
            serial1.data(),
            static_cast<size_t>(M) * static_cast<size_t>(N1),
            static_cast<size_t>(N1));
    }

    TEST_F(CPUNativeVNNIGemvTest, MTP_SmallM_Qwen36ShapesMatchSerialDecodeRows)
    {
        struct Shape
        {
            const char *name;
            const char *format;
            int N;
            int K;
        };

        static const std::array<Shape, 9> shapes = {{
            {"DenseGDNInner_Q4_K", "Q4_K", 10240, 5120},
            {"DenseGDNZ_Q4_K", "Q4_K", 6144, 5120},
            {"DenseGDNOut_Q4_K", "Q4_K", 5120, 6144},
            /*
             * Qwen3.6 Q4_K_S stores several output projections as Q5_K even
             * when adjacent in-projections are Q4_K.  The grouped verifier
             * path must prove the exact production codebook, not just a shape
             * alias, because Q5_K uses the asymmetric K-quant decode path.
             */
            {"DenseGDNOut_Q5_K", "Q5_K", 5120, 6144},
            {"DenseAttentionWo_Q4_K", "Q4_K", 5120, 6144},
            {"DenseFFNDown_Q5_K", "Q5_K", 5120, 17408},
            {"MoEGateUp_IQ2_S", "IQ2_S", 512, 256},
            {"MoEDown_IQ4_XS", "IQ4_XS", 256, 512},
            {"MoEBlock_IQ3_S", "IQ3_S", 7168, 5120},
        }};
        static const std::array<int, 3> verifier_rows = {2, 3, 4};

        for (const auto &shape : shapes)
        {
            SCOPED_TRACE(shape.name);
            auto weights = createWeightsForFormat(
                shape.format,
                static_cast<size_t>(shape.N),
                static_cast<size_t>(shape.K));
            ASSERT_NE(weights, nullptr) << "Failed to create " << shape.format
                                        << " weights for " << shape.name;

            CPUNativeVNNIGemmKernel kernel(weights.get());
            ASSERT_TRUE(kernel.isValid()) << shape.name << " failed to pack";

            for (const int M : verifier_rows)
            {
                SCOPED_TRACE(std::string("M=") + std::to_string(M));
                auto input = TestTensorFactory::createFP32Random(
                    {static_cast<size_t>(M), static_cast<size_t>(shape.K)}, -1.0f, 1.0f,
                    static_cast<uint32_t>(2200 + shape.N + shape.K + M));
                ASSERT_NE(input, nullptr);

                FP32Tensor batched({static_cast<size_t>(M), static_cast<size_t>(shape.N)});
                std::vector<ITensorGemm::TensorProjectionDesc> projections = {
                    {&kernel, &batched, shape.N, nullptr, "qwen36_verifier_projection"}};
                ASSERT_TRUE(kernel.multiply_fused_verifier_rows_decode_equivalent(
                    input.get(), projections, M, shape.K))
                    << shape.name << " grouped verifier GEMM hook failed";

                std::vector<float> serial(
                    static_cast<size_t>(M) * static_cast<size_t>(shape.N), 0.0f);
                for (int row = 0; row < M; ++row)
                {
                    ASSERT_TRUE(multiplyViaTensor(
                        kernel,
                        input->data() + static_cast<size_t>(row) * static_cast<size_t>(shape.K),
                        serial.data() + static_cast<size_t>(row) * static_cast<size_t>(shape.N),
                        1,
                        shape.N,
                        shape.K))
                        << shape.name << " serial decode GEMV failed at row=" << row;
                }

                expectBitwiseEqualFloatRows(
                    std::string(shape.name) + " CPU grouped verifier rows M=" +
                        std::to_string(M),
                    batched.data(),
                    serial.data(),
                    static_cast<size_t>(M) * static_cast<size_t>(shape.N),
                    static_cast<size_t>(shape.N));
            }
        }
    }

    TEST_F(CPUNativeVNNIGemvTest, MTP_RealQwen36GDNOutputWeightsMatchSerialDecodeRows)
    {
        const auto model_path = qwen36DenseModelPath();
        if (!std::filesystem::exists(model_path))
        {
            GTEST_SKIP() << "Qwen3.6 dense GGUF not found at " << model_path
                         << "; set LLAMINAR_QWEN36_DENSE_MODEL to run this real-weight regression";
        }

        ModelLoader loader;
        loader.setUseMmap(true);
        ASSERT_TRUE(loader.loadModel(model_path.string()))
            << "Failed to load Qwen3.6 dense model header from " << model_path;

        /*
         * The full CPU grouped-verifier parity regression first diverged at
         * these GDN output projections.  This focused test proves the exact
         * production weights and codebooks against the serial decode GEMV path
         * before we blame graph state wiring.
         */
        const std::array<const char *, 2> tensors = {
            "blk.6.ssm_out.weight",
            "blk.24.ssm_out.weight"};
        struct ActivationCase
        {
            const char *label;
            float min_value;
            float max_value;
            uint32_t seed_base;
        };
        const std::array<ActivationCase, 4> activation_cases = {{
            {"small_gdn_like", -0.25f, 0.25f, 3600u},
            {"medium_hidden", -1.0f, 1.0f, 3700u},
            {"wide_hidden", -3.0f, 3.0f, 3800u},
            {"positive_skew", -0.1f, 2.0f, 3900u},
        }};

        for (const char *tensor_name : tensors)
        {
            SCOPED_TRACE(tensor_name);
            auto weights = loader.loadTensor(tensor_name, DeviceId::cpu(), WeightPrecision::NATIVE);
            ASSERT_NE(weights, nullptr) << "Failed to load " << tensor_name;
            ASSERT_GE(weights->shape().size(), 2u);

            const int N = static_cast<int>(weights->shape()[0]);
            const int K = static_cast<int>(weights->shape()[1]);
            ASSERT_EQ(N, 5120);
            ASSERT_EQ(K, 6144);

            CPUNativeVNNIGemmKernel kernel(weights.get());
            ASSERT_TRUE(kernel.isValid()) << tensor_name << " failed NativeVNNI packing";

            for (int M : {2, 3, 4})
            {
                SCOPED_TRACE(std::string("M=") + std::to_string(M));
                for (const ActivationCase &activation_case : activation_cases)
                {
                    SCOPED_TRACE(activation_case.label);
                    auto input = TestTensorFactory::createFP32Random(
                        {static_cast<size_t>(M), static_cast<size_t>(K)},
                        activation_case.min_value,
                        activation_case.max_value,
                        activation_case.seed_base + static_cast<uint32_t>(M));
                    ASSERT_NE(input, nullptr);

                    FP32Tensor grouped({static_cast<size_t>(M), static_cast<size_t>(N)});
                    std::vector<ITensorGemm::TensorProjectionDesc> projections = {
                        {&kernel, &grouped, N, nullptr, "qwen36_gdn_out"}};
                    ASSERT_TRUE(kernel.multiply_fused_verifier_rows_decode_equivalent(
                        input.get(), projections, M, K))
                        << tensor_name << " grouped verifier GEMM hook failed";

                    std::vector<float> serial(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
                    for (int row = 0; row < M; ++row)
                    {
                        ASSERT_TRUE(multiplyViaTensor(
                            kernel,
                            input->data() + static_cast<size_t>(row) * static_cast<size_t>(K),
                            serial.data() + static_cast<size_t>(row) * static_cast<size_t>(N),
                            1,
                            N,
                            K))
                            << tensor_name << " serial decode GEMV failed at row=" << row;
                    }

                    expectBitwiseEqualFloatRows(
                        std::string(tensor_name) + " CPU real-weight grouped verifier rows M=" +
                            std::to_string(M) + " activation=" + activation_case.label,
                        grouped.data(),
                        serial.data(),
                        static_cast<size_t>(M) * static_cast<size_t>(N),
                        static_cast<size_t>(N));
                }
            }
        }
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
