/**
 * @file Test__CPUNativeVNNI_Epilogues.cpp
 * @brief Integration tests for bias add and SwiGLU epilogue support in CPUNativeVNNIGemmKernel.
 *
 * Tests:
 * - apply_bias_epilogue(): AVX-512 vectorized bias add
 * - multiply_tensor() with bias: GEMM + bias epilogue
 * - multiply_fused() with bias: quantize-once + bias epilogue per projection
 * - multiply_fused() with SwiGLU: gate * silu(up) activation fusion
 * - multiply_fused_tensor() with bias + SwiGLU
 * - supports_fused_projection(): returns true
 * - Pre-quantized GEMV/GEMM path correctness (quantize-once optimization)
 *
 * @note Run with Integration build: ctest -R V2_Integration_CPUNativeVNNI_Epilogues
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <csignal>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

#include "kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"
#include "kernels/cpu/primitives/SwiGLUPrimitives.h"
#include "tensors/Tensors.h"
#include "utils/Logger.h"

#include "utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::cpu::native_vnni;
using namespace llaminar2::test;

namespace
{

    // =========================================================================
    // MPI global environment
    // =========================================================================
    void mpi_abort_signal_handler(int sig)
    {
        const char *msg = "\n[FATAL] Signal caught — calling MPI_Abort\n";
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
    // FP32 reference GEMV (double-precision accumulation)
    // =========================================================================

    void cpuFP32GemvReference(const TensorBase *weights, const float *A,
                              float *C, int N, int K)
    {
        const IINT8Unpackable *unpackable = dynamic_cast<const IINT8Unpackable *>(weights);
        ASSERT_NE(unpackable, nullptr);
        const int K_blocks = (K + 31) / 32;

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
                    double fp_weight = static_cast<double>(scale) * static_cast<double>(vals[i]) + static_cast<double>(min_val);
                    acc += fp_weight * static_cast<double>(A[k_idx]);
                }
            }
            C[n] = static_cast<float>(acc);
        }
    }

    void cpuFP32GemmReference(const TensorBase *weights, const float *A,
                              float *C, int M, int N, int K)
    {
        for (int m = 0; m < M; ++m)
            cpuFP32GemvReference(weights, A + m * K, C + m * N, N, K);
    }

    // =========================================================================
    // Scalar SwiGLU reference
    // =========================================================================

    void swiglu_reference(const float *gate, const float *up, float *out, int size)
    {
        for (int i = 0; i < size; ++i)
        {
            float g = gate[i];
            float sigmoid_g = 1.0f / (1.0f + std::exp(-g));
            float silu_g = g * sigmoid_g;
            out[i] = silu_g * up[i];
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

    // Qwen2.5-0.5B model dimensions
    constexpr int QWEN_K = 896;      // hidden_size
    constexpr int QWEN_N_FFN = 4864; // intermediate_size

} // namespace

// =========================================================================
// Test: apply_bias_epilogue correctness
// =========================================================================

TEST(CPUNativeVNNIEpilogueTest, BiasEpilogue_M1)
{
    const int N = QWEN_N_FFN;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> C(N), bias(N), expected(N);
    for (int i = 0; i < N; ++i)
    {
        C[i] = dist(rng);
        bias[i] = dist(rng) * 0.01f; // Small bias values
        expected[i] = C[i] + bias[i];
    }

    apply_bias_epilogue(C.data(), bias.data(), 1, N, N);

    for (int i = 0; i < N; ++i)
        EXPECT_FLOAT_EQ(C[i], expected[i]) << "Mismatch at index " << i;
}

TEST(CPUNativeVNNIEpilogueTest, BiasEpilogue_M4)
{
    const int M = 4, N = QWEN_N_FFN;
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> C(M * N), bias(N), expected(M * N);
    for (int i = 0; i < M * N; ++i)
        C[i] = dist(rng);
    for (int i = 0; i < N; ++i)
        bias[i] = dist(rng) * 0.01f;
    for (int m = 0; m < M; ++m)
        for (int j = 0; j < N; ++j)
            expected[m * N + j] = C[m * N + j] + bias[j];

    apply_bias_epilogue(C.data(), bias.data(), M, N, N);

    for (int i = 0; i < M * N; ++i)
        EXPECT_FLOAT_EQ(C[i], expected[i]) << "Mismatch at index " << i;
}

TEST(CPUNativeVNNIEpilogueTest, BiasEpilogue_TailSize)
{
    // N not a multiple of 16 (AVX-512 width) — tests scalar tail
    const int M = 2, N = 137;
    std::mt19937 rng(999);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> C(M * N), bias(N), expected(M * N);
    for (int i = 0; i < M * N; ++i)
        C[i] = dist(rng);
    for (int i = 0; i < N; ++i)
        bias[i] = dist(rng) * 0.01f;
    for (int m = 0; m < M; ++m)
        for (int j = 0; j < N; ++j)
            expected[m * N + j] = C[m * N + j] + bias[j];

    apply_bias_epilogue(C.data(), bias.data(), M, N, N);

    for (int i = 0; i < M * N; ++i)
        EXPECT_FLOAT_EQ(C[i], expected[i]) << "Mismatch at index " << i;
}

// =========================================================================
// Test: supports_fused_projection returns true
// =========================================================================

TEST(CPUNativeVNNIEpilogueTest, SupportsFusedProjection)
{
    auto weights = TestTensorFactory::createQ4_0Random({64, 64});
    CPUNativeVNNIGemmKernel kernel(weights.get());
    ASSERT_TRUE(kernel.isValid());
    EXPECT_TRUE(kernel.supports_fused_projection());
}

// =========================================================================
// Test: multiply_tensor with bias
// =========================================================================

TEST(CPUNativeVNNIEpilogueTest, MultiplyTensor_WithBias_M1)
{
    const int N = QWEN_N_FFN, K = QWEN_K;
    auto weights = TestTensorFactory::createQ4_0Random({(size_t)N, (size_t)K});
    auto A = TestTensorFactory::createFP32Random({1, (size_t)K});
    auto C = TestTensorFactory::createFP32({1, (size_t)N});
    auto bias = TestTensorFactory::createFP32Random({1, (size_t)N});

    // Scale bias to be small (typical)
    float *bias_data = bias->mutable_data();
    for (int i = 0; i < N; ++i)
        bias_data[i] *= 0.01f;

    // Reference: GEMM + bias
    std::vector<float> ref(N);
    cpuFP32GemvReference(weights.get(), A->data(), ref.data(), N, K);
    for (int i = 0; i < N; ++i)
        ref[i] += bias_data[i];

    // Kernel: multiply_tensor with bias
    CPUNativeVNNIGemmKernel kernel(weights.get());
    ASSERT_TRUE(kernel.isValid());
    ASSERT_TRUE(kernel.multiply_tensor(A.get(), C.get(), 1, N, K,
                                       true, 1.0f, 0.0f, bias.get()));

    float cos = cosineSimilarity(C->data(), ref.data(), N);
    EXPECT_GT(cos, 0.990f) << "Cosine similarity too low for GEMV+bias";
}

TEST(CPUNativeVNNIEpilogueTest, MultiplyTensor_WithBias_M4)
{
    const int M = 4, N = QWEN_N_FFN, K = QWEN_K;
    auto weights = TestTensorFactory::createQ4_0Random({(size_t)N, (size_t)K});
    auto A = TestTensorFactory::createFP32Random({(size_t)M, (size_t)K});
    auto C = TestTensorFactory::createFP32({(size_t)M, (size_t)N});
    auto bias = TestTensorFactory::createFP32Random({1, (size_t)N});

    float *bias_data = bias->mutable_data();
    for (int i = 0; i < N; ++i)
        bias_data[i] *= 0.01f;

    // Reference
    std::vector<float> ref(M * N);
    cpuFP32GemmReference(weights.get(), A->data(), ref.data(), M, N, K);
    for (int m = 0; m < M; ++m)
        for (int j = 0; j < N; ++j)
            ref[m * N + j] += bias_data[j];

    CPUNativeVNNIGemmKernel kernel(weights.get());
    ASSERT_TRUE(kernel.isValid());
    ASSERT_TRUE(kernel.multiply_tensor(A.get(), C.get(), M, N, K,
                                       true, 1.0f, 0.0f, bias.get()));

    float cos = cosineSimilarity(C->data(), ref.data(), M * N);
    EXPECT_GT(cos, 0.990f) << "Cosine similarity too low for GEMM+bias";
}

TEST(CPUNativeVNNIEpilogueTest, MultiplyTensor_NoBias_Unchanged)
{
    // Verify that passing nullptr bias produces same result as before
    const int N = 128, K = 64;
    auto weights = TestTensorFactory::createQ4_0Random({(size_t)N, (size_t)K});
    auto A = TestTensorFactory::createFP32Random({1, (size_t)K});
    auto C1 = TestTensorFactory::createFP32({1, (size_t)N});
    auto C2 = TestTensorFactory::createFP32({1, (size_t)N});

    CPUNativeVNNIGemmKernel kernel(weights.get());
    ASSERT_TRUE(kernel.isValid());

    // Without bias
    ASSERT_TRUE(kernel.multiply(A->data(), C1->mutable_data(), 1, N, K));
    ASSERT_TRUE(kernel.multiply_tensor(A.get(), C2.get(), 1, N, K,
                                       true, 1.0f, 0.0f, nullptr));

    for (int i = 0; i < N; ++i)
        EXPECT_FLOAT_EQ(C1->data()[i], C2->data()[i]) << "Bias=nullptr should match multiply()";
}

// =========================================================================
// Test: multiply_fused with bias (quantize-once + bias epilogue)
// =========================================================================

TEST(CPUNativeVNNIEpilogueTest, MultiplyFused_TwoProjections_WithBias_M1)
{
    const int K = QWEN_K, N1 = QWEN_N_FFN, N2 = QWEN_N_FFN;
    auto w1 = TestTensorFactory::createQ4_0Random({(size_t)N1, (size_t)K});
    auto w2 = TestTensorFactory::createIQ4_NLRandom({(size_t)N2, (size_t)K});
    auto A = TestTensorFactory::createFP32Random({1, (size_t)K});
    auto bias1 = TestTensorFactory::createFP32Random({1, (size_t)N1});
    auto bias2 = TestTensorFactory::createFP32Random({1, (size_t)N2});

    // Scale biases
    for (int i = 0; i < N1; ++i)
        bias1->mutable_data()[i] *= 0.01f;
    for (int i = 0; i < N2; ++i)
        bias2->mutable_data()[i] *= 0.01f;

    // References
    std::vector<float> ref1(N1), ref2(N2);
    cpuFP32GemvReference(w1.get(), A->data(), ref1.data(), N1, K);
    cpuFP32GemvReference(w2.get(), A->data(), ref2.data(), N2, K);
    for (int i = 0; i < N1; ++i)
        ref1[i] += bias1->data()[i];
    for (int i = 0; i < N2; ++i)
        ref2[i] += bias2->data()[i];

    // Create kernels
    CPUNativeVNNIGemmKernel kernel1(w1.get());
    CPUNativeVNNIGemmKernel kernel2(w2.get());
    ASSERT_TRUE(kernel1.isValid());
    ASSERT_TRUE(kernel2.isValid());

    // Fused output buffers
    std::vector<float> out1(N1, 0.0f), out2(N2, 0.0f);

    std::vector<ITensorGemm::FusedProjectionDesc> projections = {
        {&kernel1, out1.data(), N1, bias1.get(), "proj1"},
        {&kernel2, out2.data(), N2, bias2.get(), "proj2"},
    };

    ASSERT_TRUE(kernel1.multiply_fused(A->data(), projections, 1, K));

    float cos1 = cosineSimilarity(out1.data(), ref1.data(), N1);
    float cos2 = cosineSimilarity(out2.data(), ref2.data(), N2);
    EXPECT_GT(cos1, 0.990f) << "Projection 1 cosine too low";
    EXPECT_GT(cos2, 0.985f) << "Projection 2 cosine too low";
}

TEST(CPUNativeVNNIEpilogueTest, MultiplyFused_TwoProjections_WithBias_M8)
{
    const int M = 8, K = QWEN_K, N1 = QWEN_N_FFN, N2 = QWEN_N_FFN;
    auto w1 = TestTensorFactory::createQ4_0Random({(size_t)N1, (size_t)K});
    auto w2 = TestTensorFactory::createQ4_0Random({(size_t)N2, (size_t)K});
    auto A = TestTensorFactory::createFP32Random({(size_t)M, (size_t)K});
    auto bias1 = TestTensorFactory::createFP32Random({1, (size_t)N1});
    auto bias2 = TestTensorFactory::createFP32Random({1, (size_t)N2});
    for (int i = 0; i < N1; ++i)
        bias1->mutable_data()[i] *= 0.01f;
    for (int i = 0; i < N2; ++i)
        bias2->mutable_data()[i] *= 0.01f;

    std::vector<float> ref1(M * N1), ref2(M * N2);
    cpuFP32GemmReference(w1.get(), A->data(), ref1.data(), M, N1, K);
    cpuFP32GemmReference(w2.get(), A->data(), ref2.data(), M, N2, K);
    for (int m = 0; m < M; ++m)
    {
        for (int j = 0; j < N1; ++j)
            ref1[m * N1 + j] += bias1->data()[j];
        for (int j = 0; j < N2; ++j)
            ref2[m * N2 + j] += bias2->data()[j];
    }

    CPUNativeVNNIGemmKernel kernel1(w1.get());
    CPUNativeVNNIGemmKernel kernel2(w2.get());
    ASSERT_TRUE(kernel1.isValid());
    ASSERT_TRUE(kernel2.isValid());

    std::vector<float> out1(M * N1, 0.0f), out2(M * N2, 0.0f);
    std::vector<ITensorGemm::FusedProjectionDesc> projections = {
        {&kernel1, out1.data(), N1, bias1.get(), "gate"},
        {&kernel2, out2.data(), N2, bias2.get(), "up"},
    };

    ASSERT_TRUE(kernel1.multiply_fused(A->data(), projections, M, K));

    float cos1 = cosineSimilarity(out1.data(), ref1.data(), M * N1);
    float cos2 = cosineSimilarity(out2.data(), ref2.data(), M * N2);
    EXPECT_GT(cos1, 0.990f) << "Gate projection cosine too low (M=8)";
    EXPECT_GT(cos2, 0.990f) << "Up projection cosine too low (M=8)";
}

// =========================================================================
// Test: multiply_fused with SwiGLU epilogue
// =========================================================================

TEST(CPUNativeVNNIEpilogueTest, MultiplyFused_SwiGLU_M1)
{
    const int K = QWEN_K, N = QWEN_N_FFN;
    auto w_gate = TestTensorFactory::createQ4_0Random({(size_t)N, (size_t)K});
    auto w_up = TestTensorFactory::createQ4_0Random({(size_t)N, (size_t)K});
    auto A = TestTensorFactory::createFP32Random({1, (size_t)K});

    // Reference: separate GEMV for gate and up, then SwiGLU
    std::vector<float> ref_gate(N), ref_up(N), ref_swiglu(N);
    cpuFP32GemvReference(w_gate.get(), A->data(), ref_gate.data(), N, K);
    cpuFP32GemvReference(w_up.get(), A->data(), ref_up.data(), N, K);
    swiglu_reference(ref_gate.data(), ref_up.data(), ref_swiglu.data(), N);

    CPUNativeVNNIGemmKernel kernel_gate(w_gate.get());
    CPUNativeVNNIGemmKernel kernel_up(w_up.get());
    ASSERT_TRUE(kernel_gate.isValid());
    ASSERT_TRUE(kernel_up.isValid());

    // Fused: gate proj + up proj with SwiGLU on up using gate as gate_input
    std::vector<float> out_gate(N, 0.0f), out_up(N, 0.0f);

    // First compute gate (no epilogue)
    std::vector<ITensorGemm::FusedProjectionDesc> projections = {
        {&kernel_gate, out_gate.data(), N, nullptr, "gate"},
        {&kernel_up, out_up.data(), N, nullptr, out_gate.data(), true, "up_swiglu"},
    };

    ASSERT_TRUE(kernel_gate.multiply_fused(A->data(), projections, 1, K));

    // out_up should now contain silu(gate) * up
    float cos = cosineSimilarity(out_up.data(), ref_swiglu.data(), N);

    // SwiGLU introduces additional quantization error from both gate and up projections
    EXPECT_GT(cos, 0.980f) << "SwiGLU fused cosine too low (M=1)";
}

TEST(CPUNativeVNNIEpilogueTest, MultiplyFused_SwiGLU_M4)
{
    const int M = 4, K = QWEN_K, N = QWEN_N_FFN;
    auto w_gate = TestTensorFactory::createQ4_0Random({(size_t)N, (size_t)K});
    auto w_up = TestTensorFactory::createQ4_0Random({(size_t)N, (size_t)K});
    auto A = TestTensorFactory::createFP32Random({(size_t)M, (size_t)K});

    std::vector<float> ref_gate(M * N), ref_up(M * N), ref_swiglu(M * N);
    cpuFP32GemmReference(w_gate.get(), A->data(), ref_gate.data(), M, N, K);
    cpuFP32GemmReference(w_up.get(), A->data(), ref_up.data(), M, N, K);
    swiglu_reference(ref_gate.data(), ref_up.data(), ref_swiglu.data(), M * N);

    CPUNativeVNNIGemmKernel kernel_gate(w_gate.get());
    CPUNativeVNNIGemmKernel kernel_up(w_up.get());
    ASSERT_TRUE(kernel_gate.isValid());
    ASSERT_TRUE(kernel_up.isValid());

    std::vector<float> out_gate(M * N, 0.0f), out_up(M * N, 0.0f);
    std::vector<ITensorGemm::FusedProjectionDesc> projections = {
        {&kernel_gate, out_gate.data(), N, nullptr, "gate"},
        {&kernel_up, out_up.data(), N, nullptr, out_gate.data(), true, "up_swiglu"},
    };

    ASSERT_TRUE(kernel_gate.multiply_fused(A->data(), projections, M, K));

    float cos = cosineSimilarity(out_up.data(), ref_swiglu.data(), M * N);
    EXPECT_GT(cos, 0.980f) << "SwiGLU fused cosine too low (M=4)";
}

// =========================================================================
// Test: multiply_fused with bias + SwiGLU combined
// =========================================================================

TEST(CPUNativeVNNIEpilogueTest, MultiplyFused_Bias_SwiGLU_Combined_M1)
{
    const int K = QWEN_K, N = QWEN_N_FFN;
    auto w_gate = TestTensorFactory::createQ4_0Random({(size_t)N, (size_t)K});
    auto w_up = TestTensorFactory::createQ4_0Random({(size_t)N, (size_t)K});
    auto A = TestTensorFactory::createFP32Random({1, (size_t)K});
    auto bias_gate = TestTensorFactory::createFP32Random({1, (size_t)N});
    auto bias_up = TestTensorFactory::createFP32Random({1, (size_t)N});
    for (int i = 0; i < N; ++i)
    {
        bias_gate->mutable_data()[i] *= 0.01f;
        bias_up->mutable_data()[i] *= 0.01f;
    }

    // Reference: GEMV + bias for each, then SwiGLU
    std::vector<float> ref_gate(N), ref_up(N), ref_swiglu(N);
    cpuFP32GemvReference(w_gate.get(), A->data(), ref_gate.data(), N, K);
    cpuFP32GemvReference(w_up.get(), A->data(), ref_up.data(), N, K);
    for (int i = 0; i < N; ++i)
    {
        ref_gate[i] += bias_gate->data()[i];
        ref_up[i] += bias_up->data()[i];
    }
    swiglu_reference(ref_gate.data(), ref_up.data(), ref_swiglu.data(), N);

    CPUNativeVNNIGemmKernel kernel_gate(w_gate.get());
    CPUNativeVNNIGemmKernel kernel_up(w_up.get());
    ASSERT_TRUE(kernel_gate.isValid());
    ASSERT_TRUE(kernel_up.isValid());

    std::vector<float> out_gate(N, 0.0f), out_up(N, 0.0f);
    std::vector<ITensorGemm::FusedProjectionDesc> projections = {
        {&kernel_gate, out_gate.data(), N, bias_gate.get(), "gate"},
        {&kernel_up, out_up.data(), N, bias_up.get(), out_gate.data(), true, "up_swiglu"},
    };

    ASSERT_TRUE(kernel_gate.multiply_fused(A->data(), projections, 1, K));

    float cos = cosineSimilarity(out_up.data(), ref_swiglu.data(), N);
    EXPECT_GT(cos, 0.980f) << "Bias+SwiGLU combined cosine too low";
}

// =========================================================================
// Test: multiply_fused_tensor
// =========================================================================

TEST(CPUNativeVNNIEpilogueTest, MultiplyFusedTensor_WithBias_M1)
{
    const int K = QWEN_K, N1 = QWEN_N_FFN, N2 = QWEN_N_FFN;
    auto w1 = TestTensorFactory::createQ4_0Random({(size_t)N1, (size_t)K});
    auto w2 = TestTensorFactory::createQ4_0Random({(size_t)N2, (size_t)K});
    auto A = TestTensorFactory::createFP32Random({1, (size_t)K});
    auto out1 = TestTensorFactory::createFP32({1, (size_t)N1});
    auto out2 = TestTensorFactory::createFP32({1, (size_t)N2});
    auto bias1 = TestTensorFactory::createFP32Random({1, (size_t)N1});
    auto bias2 = TestTensorFactory::createFP32Random({1, (size_t)N2});
    for (int i = 0; i < N1; ++i)
        bias1->mutable_data()[i] *= 0.01f;
    for (int i = 0; i < N2; ++i)
        bias2->mutable_data()[i] *= 0.01f;

    // Reference
    std::vector<float> ref1(N1), ref2(N2);
    cpuFP32GemvReference(w1.get(), A->data(), ref1.data(), N1, K);
    cpuFP32GemvReference(w2.get(), A->data(), ref2.data(), N2, K);
    for (int i = 0; i < N1; ++i)
        ref1[i] += bias1->data()[i];
    for (int i = 0; i < N2; ++i)
        ref2[i] += bias2->data()[i];

    CPUNativeVNNIGemmKernel kernel1(w1.get());
    CPUNativeVNNIGemmKernel kernel2(w2.get());
    ASSERT_TRUE(kernel1.isValid());
    ASSERT_TRUE(kernel2.isValid());

    std::vector<ITensorGemm::TensorProjectionDesc> projections = {
        {&kernel1, out1.get(), N1, bias1.get(), "proj1"},
        {&kernel2, out2.get(), N2, bias2.get(), "proj2"},
    };

    ASSERT_TRUE(kernel1.multiply_fused_tensor(A.get(), projections, 1, K));

    float cos1 = cosineSimilarity(out1->data(), ref1.data(), N1);
    float cos2 = cosineSimilarity(out2->data(), ref2.data(), N2);
    EXPECT_GT(cos1, 0.990f) << "Tensor projection 1 cosine too low";
    EXPECT_GT(cos2, 0.990f) << "Tensor projection 2 cosine too low";
}

TEST(CPUNativeVNNIEpilogueTest, MultiplyFusedTensor_SwiGLU_M1)
{
    const int K = QWEN_K, N = QWEN_N_FFN;
    auto w_gate = TestTensorFactory::createQ4_0Random({(size_t)N, (size_t)K});
    auto w_up = TestTensorFactory::createQ4_0Random({(size_t)N, (size_t)K});
    auto A = TestTensorFactory::createFP32Random({1, (size_t)K});
    auto out_gate = TestTensorFactory::createFP32({1, (size_t)N});
    auto out_up = TestTensorFactory::createFP32({1, (size_t)N});

    // Reference
    std::vector<float> ref_gate(N), ref_up(N), ref_swiglu(N);
    cpuFP32GemvReference(w_gate.get(), A->data(), ref_gate.data(), N, K);
    cpuFP32GemvReference(w_up.get(), A->data(), ref_up.data(), N, K);
    swiglu_reference(ref_gate.data(), ref_up.data(), ref_swiglu.data(), N);

    CPUNativeVNNIGemmKernel kernel_gate(w_gate.get());
    CPUNativeVNNIGemmKernel kernel_up(w_up.get());
    ASSERT_TRUE(kernel_gate.isValid());
    ASSERT_TRUE(kernel_up.isValid());

    std::vector<ITensorGemm::TensorProjectionDesc> projections = {
        {&kernel_gate, out_gate.get(), N, nullptr, "gate"},
        {&kernel_up, out_up.get(), N, nullptr, out_gate.get(), true, "up_swiglu"},
    };

    ASSERT_TRUE(kernel_gate.multiply_fused_tensor(A.get(), projections, 1, K));

    float cos = cosineSimilarity(out_up->data(), ref_swiglu.data(), N);
    EXPECT_GT(cos, 0.980f) << "Tensor SwiGLU fused cosine too low";
}

// =========================================================================
// Test: Pre-quantized path matches standard path (quantize-once correctness)
// =========================================================================

TEST(CPUNativeVNNIEpilogueTest, PreQuantized_GEMV_MatchesStandard)
{
    const int N = QWEN_N_FFN, K = QWEN_K;
    auto weights = TestTensorFactory::createQ4_0Random({(size_t)N, (size_t)K});
    auto A = TestTensorFactory::createFP32Random({1, (size_t)K});

    CPUNativeVNNIGemmKernel kernel(weights.get());
    ASSERT_TRUE(kernel.isValid());

    // Standard path: quantizes internally
    std::vector<float> C_standard(N, 0.0f);
    ASSERT_TRUE(kernel.multiply(A->data(), C_standard.data(), 1, N, K));

    // Fused path with single projection (uses pre-quantized path)
    std::vector<float> C_fused(N, 0.0f);
    std::vector<ITensorGemm::FusedProjectionDesc> projections = {
        {&kernel, C_fused.data(), N, nullptr, "test"},
    };
    ASSERT_TRUE(kernel.multiply_fused(A->data(), projections, 1, K));

    // Should be identical (same quantization, same compute)
    for (int i = 0; i < N; ++i)
        EXPECT_FLOAT_EQ(C_standard[i], C_fused[i]) << "Pre-quantized GEMV mismatch at " << i;
}

TEST(CPUNativeVNNIEpilogueTest, PreQuantized_GEMM_MatchesStandard)
{
    const int M = 4, N = QWEN_N_FFN, K = QWEN_K;
    auto weights = TestTensorFactory::createQ4_0Random({(size_t)N, (size_t)K});
    auto A = TestTensorFactory::createFP32Random({(size_t)M, (size_t)K});

    CPUNativeVNNIGemmKernel kernel(weights.get());
    ASSERT_TRUE(kernel.isValid());

    // Standard path
    std::vector<float> C_standard(M * N, 0.0f);
    ASSERT_TRUE(kernel.multiply(A->data(), C_standard.data(), M, N, K));

    // Fused path
    std::vector<float> C_fused(M * N, 0.0f);
    std::vector<ITensorGemm::FusedProjectionDesc> projections = {
        {&kernel, C_fused.data(), N, nullptr, "test"},
    };
    ASSERT_TRUE(kernel.multiply_fused(A->data(), projections, M, K));

    float cos = cosineSimilarity(C_standard.data(), C_fused.data(), M * N);
    // The pre-quantized path may differ slightly from the original because
    // quantization and compute are in separate OMP regions vs combined.
    // But actual values should be bitwise identical since same input.
    EXPECT_GT(cos, 0.99999f) << "Pre-quantized GEMM should match standard";
}

// =========================================================================
// Test: IQ4_NL format with epilogues
// =========================================================================

TEST(CPUNativeVNNIEpilogueTest, IQ4NL_WithBias_M1)
{
    const int N = QWEN_N_FFN, K = QWEN_K;
    auto weights = TestTensorFactory::createIQ4_NLRandom({(size_t)N, (size_t)K});
    auto A = TestTensorFactory::createFP32Random({1, (size_t)K});
    auto C = TestTensorFactory::createFP32({1, (size_t)N});
    auto bias = TestTensorFactory::createFP32Random({1, (size_t)N});
    for (int i = 0; i < N; ++i)
        bias->mutable_data()[i] *= 0.01f;

    std::vector<float> ref(N);
    cpuFP32GemvReference(weights.get(), A->data(), ref.data(), N, K);
    for (int i = 0; i < N; ++i)
        ref[i] += bias->data()[i];

    CPUNativeVNNIGemmKernel kernel(weights.get());
    ASSERT_TRUE(kernel.isValid());
    ASSERT_TRUE(kernel.multiply_tensor(A.get(), C.get(), 1, N, K,
                                       true, 1.0f, 0.0f, bias.get()));

    float cos = cosineSimilarity(C->data(), ref.data(), N);
    EXPECT_GT(cos, 0.985f) << "IQ4_NL + bias cosine too low";
}

// =========================================================================
// Test: multiply_fused with no epilogues (should still work)
// =========================================================================

TEST(CPUNativeVNNIEpilogueTest, MultiplyFused_NoEpilogues_M1)
{
    const int K = QWEN_K, N = QWEN_N_FFN;
    auto weights = TestTensorFactory::createQ4_0Random({(size_t)N, (size_t)K});
    auto A = TestTensorFactory::createFP32Random({1, (size_t)K});

    CPUNativeVNNIGemmKernel kernel(weights.get());
    ASSERT_TRUE(kernel.isValid());

    std::vector<float> ref(N);
    ASSERT_TRUE(kernel.multiply(A->data(), ref.data(), 1, N, K));

    std::vector<float> out(N, 0.0f);
    std::vector<ITensorGemm::FusedProjectionDesc> projections = {
        {&kernel, out.data(), N, nullptr, "plain"},
    };
    ASSERT_TRUE(kernel.multiply_fused(A->data(), projections, 1, K));

    for (int i = 0; i < N; ++i)
        EXPECT_FLOAT_EQ(ref[i], out[i]) << "Fused without epilogues should match standard";
}
