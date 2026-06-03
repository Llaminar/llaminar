/**
 * @file Test__ROCmFlashAttentionParity.cpp
 * @brief Parity tests for ROCm Flash Attention kernel vs CPU reference
 *
 * **Purpose**: Validate that ROCm Flash Attention kernels produce numerically
 * equivalent results to CPU attention kernels with high cosine similarity.
 *
 * **Tests**:
 * - Flash Attention 2 (prefill) vs CPU attention
 * - Flash Decoding (decode) vs CPU attention
 * - Various head dimensions (64, 128, 256)
 * - GQA configurations (n_heads != n_kv_heads)
 *
 * **Pass Criteria**:
 * - Cosine similarity >= 0.99 (attention is numerically sensitive)
 * - No NaN/Inf in outputs
 * - Relative error < 5% for FP32
 *
 * Target Hardware: AMD MI50 (gfx906 / Vega 20)
 *
 * @author Llaminar Team
 * @date January 2026
 */

#include <gtest/gtest.h>

#include "tensors/Tensors.h"
#include "execution/config/RuntimeConfig.h"
#include "execution/factory/InferenceRunnerFactory.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/coherence/GpuCoherence.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "loaders/ModelContext.h"
#include "transfer/TransferEngine.h"
#include "utils/MPIContext.h"
#include "utils/Logger.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#include "kernels/rocm/attention/ROCmFlashAttentionKernelT.h"
#include "kernels/cpu/attention/CPUFlashAttentionKernelT.h"
#include "kernels/cpu/CPURingKVCache.h"
#endif

#include <algorithm>
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>
#include <limits>

using namespace llaminar2;

namespace
{
    const std::string TEST_MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

    // ============================================================================
    // ROCm Availability Check
    // ============================================================================

    bool hasROCm()
    {
#ifdef HAVE_ROCM
        int count = 0;
        hipError_t err = hipGetDeviceCount(&count);
        return (err == hipSuccess && count > 0);
#else
        return false;
#endif
    }

    /**
     * @brief Check if Flash Attention 2 will fit in LDS for the given head_dim
     *
     * The FA2 kernel uses:
     * - Q_tile:    [64, head_dim + 8] in FP16
     * - K/V double buffer: 2 * 2 * [64, head_dim + 8] in FP16
     * - Scores:    [64, 68] in FP32
     *
     * MI50 has 65KB LDS per workgroup.
     *
     * @param head_dim The head dimension to check
     * @return true if the kernel will fit, false otherwise
     */
    bool flashAttn2FitsInLDS(int head_dim)
    {
#ifdef HAVE_ROCM
        constexpr int tile_q = 64;
        constexpr int tile_kv = 64;
        constexpr int lds_pad_fp16 = 8;
        constexpr int lds_pad_fp32 = 4;
        constexpr int num_stages = 2;
        constexpr int max_lds = 65536; // MI50 LDS size

        int q_stride = head_dim + lds_pad_fp16;
        int kv_stride = head_dim + lds_pad_fp16;
        int scores_stride = tile_kv + lds_pad_fp32;

        int q_tile_size = tile_q * q_stride * sizeof(uint16_t);    // FP16
        int kv_tile_size = tile_kv * kv_stride * sizeof(uint16_t); // FP16
        int scores_size = tile_q * scores_stride * sizeof(float);

        // Total: Q + 2*(K+V) for double buffer + scores
        int total_lds = q_tile_size + num_stages * 2 * kv_tile_size + scores_size;

        return total_lds <= max_lds;
#else
        return false;
#endif
    }

    // ============================================================================
    // Similarity Utilities
    // ============================================================================

    double cosineSimilarity(const float *a, const float *b, size_t count)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            dot += static_cast<double>(a[i]) * b[i];
            norm_a += static_cast<double>(a[i]) * a[i];
            norm_b += static_cast<double>(b[i]) * b[i];
        }
        double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
        if (denom < 1e-12)
            return 0.0;
        return dot / denom;
    }

    double relativeL2Error(const float *actual, const float *expected, size_t count)
    {
        double diff_norm = 0.0, expected_norm = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            double diff = actual[i] - expected[i];
            diff_norm += diff * diff;
            expected_norm += static_cast<double>(expected[i]) * expected[i];
        }
        if (expected_norm < 1e-12)
            return diff_norm > 1e-12 ? 1e9 : 0.0;
        return std::sqrt(diff_norm / expected_norm);
    }

    double maxAbsError(const float *actual, const float *expected, size_t count)
    {
        double max_err = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            double err = std::abs(static_cast<double>(actual[i]) - expected[i]);
            if (err > max_err)
                max_err = err;
        }
        return max_err;
    }

    bool hasNaNOrInf(const float *data, size_t count)
    {
        for (size_t i = 0; i < count; ++i)
        {
            if (std::isnan(data[i]) || std::isinf(data[i]))
            {
                return true;
            }
        }
        return false;
    }

    inline uint16_t floatToFP16(float f)
    {
        uint32_t bits;
        memcpy(&bits, &f, sizeof(float));
        uint32_t sign = (bits >> 16) & 0x8000;
        int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = (bits >> 13) & 0x3FF;

        if (exp <= 0)
            return static_cast<uint16_t>(sign);
        if (exp >= 31)
            return static_cast<uint16_t>(sign | 0x7C00);
        return static_cast<uint16_t>(sign | (exp << 10) | mant);
    }

    inline float fp16ToFloat(uint16_t fp16)
    {
        uint32_t sign = (fp16 & 0x8000) << 16;
        int32_t exp = (fp16 >> 10) & 0x1F;
        uint32_t mant = fp16 & 0x3FF;

        if (exp == 0)
        {
            if (mant == 0)
            {
                uint32_t bits = sign;
                float result;
                memcpy(&result, &bits, sizeof(float));
                return result;
            }
            exp = 1;
            while (!(mant & 0x400))
            {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
        }
        else if (exp == 31)
        {
            uint32_t bits = sign | 0x7F800000 | (mant << 13);
            float result;
            memcpy(&result, &bits, sizeof(float));
            return result;
        }

        uint32_t bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        float result;
        memcpy(&result, &bits, sizeof(float));
        return result;
    }

    void quantizeToFP16(const float *src, uint16_t *dst, size_t count)
    {
        for (size_t i = 0; i < count; ++i)
        {
            dst[i] = floatToFP16(src[i]);
        }
    }

    void dequantizeFP16(const uint16_t *src, float *dst, size_t count)
    {
        for (size_t i = 0; i < count; ++i)
        {
            dst[i] = fp16ToFloat(src[i]);
        }
    }

} // namespace

// ============================================================================
// Test Fixture
// ============================================================================

class Test__ROCmFlashAttentionParity : public ::testing::Test
{
protected:
    std::mt19937 rng_{42};
    std::uniform_real_distribution<float> dist_{-0.5f, 0.5f};
    MPIContext mpi_ctx_{0, 1, MPI_COMM_WORLD};

    // Workspace manager for FlashDecode tests (owned by fixture)
    std::unique_ptr<DeviceWorkspaceManager> workspace_;

    void SetUp() override
    {
#ifdef HAVE_ROCM
        ASSERT_EQ(hipSetDevice(0), hipSuccess) << "Failed to set active ROCm device to 0";
#endif
    }

    std::vector<float> randomFP32(size_t count)
    {
        std::vector<float> data(count);
        for (auto &val : data)
        {
            val = dist_(rng_);
        }
        return data;
    }

    std::vector<float> randomFP32Scaled(size_t count, float scale)
    {
        auto data = randomFP32(count);
        for (auto &val : data)
        {
            val *= scale;
        }
        return data;
    }

    void printComparisonStats(
        const char *test_name,
        double cosine, double l2_error, double max_error,
        size_t count)
    {
        std::cout << "  " << test_name << ": "
                  << "cosine=" << std::fixed << std::setprecision(6) << cosine
                  << ", L2_error=" << std::scientific << std::setprecision(3) << l2_error
                  << ", max_error=" << max_error
                  << ", count=" << count
                  << std::endl;
    }

#ifdef HAVE_ROCM
    /**
     * @brief Set up workspace for FlashDecode kernel
     *
     * FlashDecode requires workspace buffers for multi-block reduction:
     * - PARTIAL_OUTPUT: [batch × n_heads × num_splits × head_dim]
     * - PARTIAL_M: [batch × n_heads × num_splits]
     * - PARTIAL_L: [batch × n_heads × num_splits]
     */
    bool setupWorkspace(
        llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> &kernel,
        int batch_size, int n_heads, int head_dim)
    {
        auto reqs = kernel.getWorkspaceRequirements(batch_size, n_heads, head_dim);

        // Compute actual budget from requirements (with alignment padding) + 1MB margin
        const size_t required_bytes = reqs.total_bytes_with_alignment();
        const size_t budget = std::max(required_bytes + 1024 * 1024, static_cast<size_t>(16 * 1024 * 1024));
        workspace_ = std::make_unique<DeviceWorkspaceManager>(DeviceId::rocm(0), budget);
        if (!workspace_->allocate(reqs))
        {
            LOG_ERROR("Failed to allocate workspace for FlashDecode (required="
                      << (required_bytes / (1024 * 1024)) << "MB, budget="
                      << (budget / (1024 * 1024)) << "MB)");
            return false;
        }
        kernel.bindWorkspace(workspace_.get());
        return true;
    }

    void cleanupWorkspace(
        llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> &kernel)
    {
        if (workspace_)
        {
            kernel.unbindWorkspace();
            workspace_.reset();
        }
    }
#endif
};

#ifdef HAVE_ROCM

// ============================================================================
// Flash Attention 2 (Prefill) Parity Tests
// ============================================================================

TEST_F(Test__ROCmFlashAttentionParity, FlashAttn2_FP32_Small)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    // Small test case for basic correctness
    constexpr int seq_len = 8;
    constexpr int n_heads = 4;
    constexpr int n_kv_heads = 4; // MHA (not GQA)
    constexpr int head_dim = 32;
    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = seq_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    // Create test data
    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> rocm_output(out_size, 0.0f);

    // CPU reference
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true,    // causal
        -1,      // window_size
        nullptr, // workspace_scores
        nullptr, // workspace_buffer
        nullptr, // workspace_context
        nullptr, // workspace_mask
        false,   // use_bf16
        &mpi_ctx_,
        -1 // device_idx (CPU)
    );
    ASSERT_TRUE(cpu_success) << "CPU attention failed";

    // ROCm kernel
    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_output;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_output, out_size * sizeof(float));

    // Copy inputs to device
    hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, out_size * sizeof(float));

    // Execute ROCm kernel
    bool rocm_success = rocm_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true,    // causal
        -1,      // window_size
        nullptr, // workspace_scores
        nullptr, // workspace_buffer
        nullptr, // workspace_context
        nullptr, // workspace_mask
        false,   // use_bf16
        &mpi_ctx_,
        0 // device_idx
    );
    hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success) << "ROCm attention failed";

    // Copy output back
    hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    // Cleanup
    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    // Validate
    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), out_size)) << "ROCm output contains NaN/Inf";
    ASSERT_FALSE(hasNaNOrInf(cpu_output.data(), out_size)) << "CPU output contains NaN/Inf";

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(rocm_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 FP32 Small", cosine, l2_error, max_error, out_size);

    // Attention has some numerical sensitivity, so we use looser thresholds
    EXPECT_GE(cosine, 0.99) << "Cosine similarity too low";
    EXPECT_LE(l2_error, 0.05) << "L2 error too high";

    LOG_INFO("[FlashAttn2_FP32_Small] PASSED");
}

TEST_F(Test__ROCmFlashAttentionParity, FlashAttn2_FP32_Medium_GQA)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    // Medium test case with GQA (typical Qwen2 configuration)
    constexpr int seq_len = 64;
    constexpr int n_heads = 14;   // Qwen2-0.5B
    constexpr int n_kv_heads = 2; // GQA with ratio 7
    constexpr int head_dim = 64;
    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = seq_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> rocm_output(out_size, 0.0f);

    // CPU reference
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, -1);
    ASSERT_TRUE(cpu_success);

    // ROCm kernel
    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_output, out_size * sizeof(float));

    hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), out_size));
    ASSERT_FALSE(hasNaNOrInf(cpu_output.data(), out_size));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(rocm_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 FP32 Medium GQA", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99) << "Cosine similarity too low";
    EXPECT_LE(l2_error, 0.05) << "L2 error too high";

    LOG_INFO("[FlashAttn2_FP32_Medium_GQA] PASSED");
}

TEST_F(Test__ROCmFlashAttentionParity, FlashAttn2_FP32_LargeHeadDim)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    // Test with larger head_dim (Llama-3 style)
    // Kernel now automatically selects smaller tiles (32x32) to fit in LDS
    constexpr int seq_len = 32;
    constexpr int n_heads = 8;
    constexpr int n_kv_heads = 8; // MHA
    constexpr int head_dim = 128;

    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = seq_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> rocm_output(out_size, 0.0f);

    // CPU reference
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, -1);
    ASSERT_TRUE(cpu_success);

    // ROCm kernel
    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_output, out_size * sizeof(float));

    hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), out_size));
    ASSERT_FALSE(hasNaNOrInf(cpu_output.data(), out_size));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(rocm_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 FP32 Large HeadDim", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99) << "Cosine similarity too low";
    EXPECT_LE(l2_error, 0.05) << "L2 error too high";

    LOG_INFO("[FlashAttn2_FP32_LargeHeadDim] PASSED");
}

TEST_F(Test__ROCmFlashAttentionParity, FlashAttn2_FP32_LargeHeadDim_LargerSeq)
{
    // Tests adaptive tile sizing with larger sequence lengths
    // This ensures the 32x32 tile path works correctly across more Q tiles
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    constexpr int seq_len = 128; // More Q tiles with 32x32 tiling
    constexpr int n_heads = 8;
    constexpr int n_kv_heads = 8;
    constexpr int head_dim = 128; // Triggers adaptive tiling

    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = seq_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> rocm_output(out_size, 0.0f);

    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, -1);
    ASSERT_TRUE(cpu_success);

    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_output, out_size * sizeof(float));

    hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), out_size));
    ASSERT_FALSE(hasNaNOrInf(cpu_output.data(), out_size));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(rocm_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 FP32 LargeHeadDim LargerSeq", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99) << "Cosine similarity too low";
    EXPECT_LE(l2_error, 0.05) << "L2 error too high";

    LOG_INFO("[FlashAttn2_FP32_LargeHeadDim_LargerSeq] PASSED");
}

TEST_F(Test__ROCmFlashAttentionParity, FlashAttn2_FP32_LargeHeadDim_GQA)
{
    // Tests adaptive tile sizing with GQA (grouped query attention)
    // Llama-3 uses head_dim=128 with GQA
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    constexpr int seq_len = 64;
    constexpr int n_heads = 32;   // 32 query heads
    constexpr int n_kv_heads = 8; // 8 KV heads (GQA ratio = 4)
    constexpr int head_dim = 128; // Triggers adaptive tiling

    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = seq_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> rocm_output(out_size, 0.0f);

    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, -1);
    ASSERT_TRUE(cpu_success);

    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_output, out_size * sizeof(float));

    hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), out_size));
    ASSERT_FALSE(hasNaNOrInf(cpu_output.data(), out_size));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(rocm_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 FP32 LargeHeadDim GQA", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99) << "Cosine similarity too low";
    EXPECT_LE(l2_error, 0.05) << "L2 error too high";

    LOG_INFO("[FlashAttn2_FP32_LargeHeadDim_GQA] PASSED");
}

TEST_F(Test__ROCmFlashAttentionParity, RocblasPrefillHonorsExternalMask)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    constexpr int seq_len = 2;
    constexpr int n_heads = 1;
    constexpr int n_kv_heads = 1;
    constexpr int head_dim = 64; // Forces the rocBLAS prefill implementation.
    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = seq_len * n_kv_heads * head_dim;
    const size_t out_size = q_size;

    std::vector<float> Q_data(q_size, 0.0f);
    std::vector<float> K_data(kv_size, 0.0f);
    std::vector<float> V_data(kv_size, 0.0f);
    for (int d = 0; d < head_dim; ++d)
    {
        V_data[d] = 1.0f;
        V_data[head_dim + d] = 100.0f;
    }
    std::vector<float> rocm_output(out_size, 0.0f);

    FP32Tensor mask_tensor({static_cast<size_t>(seq_len), static_cast<size_t>(seq_len)});
    float *mask = mask_tensor.mutable_data();
    mask[0] = 0.0f;
    mask[1] = -std::numeric_limits<float>::infinity();
    mask[2] = 0.0f;
    mask[3] = 0.0f;
    auto upload = TransferEngine::instance().uploadFull(&mask_tensor, DeviceId::rocm(0));
    ASSERT_TRUE(upload.success) << upload.error;
    ASSERT_NE(mask_tensor.gpu_data_ptr(), nullptr);

    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    float *d_Q = nullptr;
    float *d_K = nullptr;
    float *d_V = nullptr;
    float *d_output = nullptr;
    ASSERT_EQ(hipMalloc(&d_Q, q_size * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_K, kv_size * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_V, kv_size * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_output, out_size * sizeof(float)), hipSuccess);

    ASSERT_EQ(hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemset(d_output, 0, out_size * sizeof(float)), hipSuccess);

    const bool rocm_success = rocm_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        false,   // external mask owns causality for continuation chunks
        -1,
        nullptr,
        nullptr,
        nullptr,
        &mask_tensor,
        false,
        &mpi_ctx_,
        0);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    ASSERT_TRUE(rocm_success);

    ASSERT_EQ(hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    for (int d = 0; d < head_dim; ++d)
    {
        EXPECT_NEAR(rocm_output[d], 1.0f, 1e-4f) << "row 0 dim " << d;
        EXPECT_NEAR(rocm_output[head_dim + d], 50.5f, 1e-4f) << "row 1 dim " << d;
    }
}

// ============================================================================
// Flash Decoding (Decode) Parity Tests
// ============================================================================

TEST_F(Test__ROCmFlashAttentionParity, FlashDecode_FP32_SmallKVLen)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    // Decode: single query attending to small KV cache
    constexpr int kv_len = 16;
    constexpr int n_heads = 4;
    constexpr int n_kv_heads = 4;
    constexpr int head_dim = 64;
    constexpr int seq_len = 1; // Decode = single query
    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = kv_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> rocm_output(out_size, 0.0f);

    // CPU reference - for decode with seq_len=1, use compute_decode
    // The CPU kernel expects Q[seq_len, n_heads, head_dim], K[kv_len, n_kv_heads, head_dim]
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute_decode(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        false); // Non-causal for decode (attend to full KV cache)
    ASSERT_TRUE(cpu_success);

    // ROCm kernel (decode path)
    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    // Setup workspace for FlashDecode (required for multi-block reduction)
    ASSERT_TRUE(setupWorkspace(rocm_kernel, seq_len, n_heads, head_dim));

    float *d_Q, *d_K, *d_V, *d_output;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_output, out_size * sizeof(float));

    hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, out_size * sizeof(float));

    // Use compute_decode for single-query path
    bool rocm_success = rocm_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        false, // Non-causal
        0);    // position_offset
    hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success) << "ROCm decode attention failed";

    hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    // Cleanup workspace before freeing GPU memory
    cleanupWorkspace(rocm_kernel);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), out_size));
    ASSERT_FALSE(hasNaNOrInf(cpu_output.data(), out_size));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(rocm_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashDecode FP32 Small KVLen", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99) << "Cosine similarity too low";
    EXPECT_LE(l2_error, 0.05) << "L2 error too high";

    LOG_INFO("[FlashDecode_FP32_SmallKVLen] PASSED");
}

TEST_F(Test__ROCmFlashAttentionParity, FlashDecode_FP32_LargeKVLen_GQA)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    // Decode with larger KV cache and GQA
    constexpr int kv_len = 512;
    constexpr int n_heads = 14;   // Qwen2
    constexpr int n_kv_heads = 2; // GQA
    constexpr int head_dim = 64;
    constexpr int seq_len = 1;
    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = kv_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> rocm_output(out_size, 0.0f);

    // CPU reference
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute_decode(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        false); // Non-causal
    ASSERT_TRUE(cpu_success);

    // ROCm kernel
    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    // Setup workspace for FlashDecode (required for multi-block reduction)
    ASSERT_TRUE(setupWorkspace(rocm_kernel, seq_len, n_heads, head_dim));

    float *d_Q, *d_K, *d_V, *d_output;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_output, out_size * sizeof(float));

    hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        false, // Non-causal
        0);    // position_offset
    hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    // Cleanup workspace before freeing GPU memory
    cleanupWorkspace(rocm_kernel);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), out_size));
    ASSERT_FALSE(hasNaNOrInf(cpu_output.data(), out_size));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(rocm_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashDecode FP32 Large KVLen GQA", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99) << "Cosine similarity too low";
    EXPECT_LE(l2_error, 0.05) << "L2 error too high";

    LOG_INFO("[FlashDecode_FP32_LargeKVLen_GQA] PASSED");
}

TEST_F(Test__ROCmFlashAttentionParity, FlashDecode_FP32_GraphReplayUsesUpdatedKVLenWithinBucket)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    constexpr int capture_kv_len = 65;
    constexpr int replay_kv_len = 66;
    constexpr int max_kv_len = 80;
    constexpr int n_heads = 4;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    constexpr int seq_len = 1;

    const DeviceId device = DeviceId::rocm(0);
    const size_t q_size = static_cast<size_t>(seq_len) * n_heads * head_dim;
    const size_t kv_size = static_cast<size_t>(max_kv_len) * n_kv_heads * head_dim;
    const size_t out_size = q_size;

    auto Q_data = randomFP32Scaled(q_size, 0.25f);
    auto K_data = randomFP32Scaled(kv_size, 0.25f);
    auto V_data = randomFP32Scaled(kv_size, 0.25f);

    FP32Tensor q_tensor({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * head_dim)});
    FP32Tensor k_tensor({static_cast<size_t>(max_kv_len), static_cast<size_t>(n_kv_heads * head_dim)});
    FP32Tensor v_tensor({static_cast<size_t>(max_kv_len), static_cast<size_t>(n_kv_heads * head_dim)});
    FP32Tensor captured_output({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * head_dim)});
    FP32Tensor direct_output({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * head_dim)});
    std::copy(Q_data.begin(), Q_data.end(), q_tensor.mutable_data());
    std::copy(K_data.begin(), K_data.end(), k_tensor.mutable_data());
    std::copy(V_data.begin(), V_data.end(), v_tensor.mutable_data());

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);

    auto &transfer = TransferEngine::instance();
    ASSERT_TRUE(transfer.uploadFull(&q_tensor, device, stream).success);
    ASSERT_TRUE(transfer.uploadFull(&k_tensor, device, stream).success);
    ASSERT_TRUE(transfer.uploadFull(&v_tensor, device, stream).success);
    ASSERT_TRUE(transfer.uploadFull(&captured_output, device, stream).success);
    ASSERT_TRUE(transfer.uploadFull(&direct_output, device, stream).success);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);
    rocm_kernel.setGPUStream(stream);
    ASSERT_TRUE(setupWorkspace(rocm_kernel, seq_len, n_heads, head_dim));

    // Warmup allocates pinned dynamic-param storage before capture.
    ASSERT_TRUE(rocm_kernel.compute_tensor(
        &q_tensor, &k_tensor, &v_tensor, &captured_output,
        1, seq_len, capture_kv_len, n_heads, n_kv_heads, head_dim,
        false, -1, nullptr, nullptr, &mpi_ctx_, 0));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    hipGraph_t graph = nullptr;
    hipGraphExec_t graph_exec = nullptr;
    {
        GraphCaptureGuard guard;
        ASSERT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), hipSuccess);
        ASSERT_TRUE(rocm_kernel.compute_tensor(
            &q_tensor, &k_tensor, &v_tensor, &captured_output,
            1, seq_len, capture_kv_len, n_heads, n_kv_heads, head_dim,
            false, -1, nullptr, nullptr, &mpi_ctx_, 0));
        ASSERT_EQ(hipStreamEndCapture(stream, &graph), hipSuccess);
    }
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), hipSuccess);

    rocm_kernel.setDynamicAttnParams(replay_kv_len, replay_kv_len - 1, 1);
    ASSERT_EQ(hipGraphLaunch(graph_exec, stream), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    std::vector<float> captured(out_size, 0.0f);
    ASSERT_EQ(hipMemcpyAsync(captured.data(), captured_output.gpu_data_ptr(),
                             out_size * sizeof(float), hipMemcpyDeviceToHost, stream),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    ASSERT_TRUE(rocm_kernel.compute_tensor(
        &q_tensor, &k_tensor, &v_tensor, &direct_output,
        1, seq_len, replay_kv_len, n_heads, n_kv_heads, head_dim,
        false, -1, nullptr, nullptr, &mpi_ctx_, 0));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    std::vector<float> direct(out_size, 0.0f);
    ASSERT_EQ(hipMemcpyAsync(direct.data(), direct_output.gpu_data_ptr(),
                             out_size * sizeof(float), hipMemcpyDeviceToHost, stream),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    ASSERT_FALSE(hasNaNOrInf(captured.data(), out_size));
    ASSERT_FALSE(hasNaNOrInf(direct.data(), out_size));

    const double cosine = cosineSimilarity(captured.data(), direct.data(), out_size);
    const double l2_error = relativeL2Error(captured.data(), direct.data(), out_size);
    const double max_error = maxAbsError(captured.data(), direct.data(), out_size);
    printComparisonStats("FlashDecode FP32 graph replay updated kv_len", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.9999);
    EXPECT_LE(l2_error, 1e-4);
    EXPECT_LE(max_error, 1e-4);

    hipGraphExecDestroy(graph_exec);
    hipGraphDestroy(graph);
    cleanupWorkspace(rocm_kernel);
    hipStreamDestroy(stream);
}

TEST_F(Test__ROCmFlashAttentionParity, FlashDecode_NativeFP16KV_Qwen35LongKVLen)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    constexpr int seq_len = 1;
    constexpr int kv_len = 1024;
    constexpr int n_heads = 16;
    constexpr int n_kv_heads = 4;
    constexpr int head_dim = 256;
    constexpr int gqa_ratio = n_heads / n_kv_heads;
    const size_t q_size = static_cast<size_t>(seq_len) * n_heads * head_dim;
    const size_t kv_size = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;
    const size_t out_size = q_size;

    // Long-context generation reads an FP16 KV cache with a single FP32 query.
    // This is the decode path used immediately after a long Qwen3.5 prefill.
    auto Q_data = randomFP32Scaled(q_size, 0.35f);
    auto K_data_fp32 = randomFP32Scaled(kv_size, 0.35f);
    auto V_data_fp32 = randomFP32Scaled(kv_size, 0.35f);
    std::vector<uint16_t> K_data_fp16(kv_size);
    std::vector<uint16_t> V_data_fp16(kv_size);
    quantizeToFP16(K_data_fp32.data(), K_data_fp16.data(), kv_size);
    quantizeToFP16(V_data_fp32.data(), V_data_fp16.data(), kv_size);

    std::vector<float> K_ref_fp32(kv_size);
    std::vector<float> V_ref_fp32(kv_size);
    dequantizeFP16(K_data_fp16.data(), K_ref_fp32.data(), kv_size);
    dequantizeFP16(V_data_fp16.data(), V_ref_fp32.data(), kv_size);

    std::vector<float> cpu_output(out_size, 0.0f);
    for (int head = 0; head < n_heads; ++head)
    {
        const int kv_head = head / gqa_ratio;
        const float *q = Q_data.data() + static_cast<size_t>(head) * head_dim;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        std::vector<float> scores(static_cast<size_t>(kv_len));
        float max_score = -std::numeric_limits<float>::infinity();

        for (int kv_pos = 0; kv_pos < kv_len; ++kv_pos)
        {
            const float *k = K_ref_fp32.data() +
                             (static_cast<size_t>(kv_pos) * n_kv_heads + kv_head) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d)
            {
                dot += q[d] * k[d];
            }
            const float score = dot * scale;
            scores[static_cast<size_t>(kv_pos)] = score;
            max_score = std::max(max_score, score);
        }

        double sum_exp = 0.0;
        float *out = cpu_output.data() + static_cast<size_t>(head) * head_dim;
        for (int kv_pos = 0; kv_pos < kv_len; ++kv_pos)
        {
            const float weight = std::exp(scores[static_cast<size_t>(kv_pos)] - max_score);
            const float *v = V_ref_fp32.data() +
                             (static_cast<size_t>(kv_pos) * n_kv_heads + kv_head) * head_dim;
            sum_exp += static_cast<double>(weight);
            for (int d = 0; d < head_dim; ++d)
            {
                out[d] += weight * v[d];
            }
        }

        const float inv_sum = static_cast<float>(1.0 / std::max(sum_exp, 1e-30));
        for (int d = 0; d < head_dim; ++d)
        {
            out[d] *= inv_sum;
        }
    }

    auto q_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * head_dim)});
    auto k_tensor = std::make_shared<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)},
        K_data_fp16);
    auto v_tensor = std::make_shared<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)},
        V_data_fp16);
    auto output_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * head_dim)});
    std::copy(Q_data.begin(), Q_data.end(), q_tensor->mutable_data());

    const DeviceId gpu_device = DeviceId::rocm(0);
    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);
    ASSERT_TRUE(setupWorkspace(rocm_kernel, seq_len, n_heads, head_dim));

    ASSERT_TRUE(with_gpu_coherence(
        gpu_device,
        {q_tensor.get(), k_tensor.get(), v_tensor.get()},
        {output_tensor.get()},
        [&]
        {
            return rocm_kernel.compute_tensor(
                q_tensor.get(), k_tensor.get(), v_tensor.get(), output_tensor.get(),
                1, seq_len, kv_len, n_heads, n_kv_heads, head_dim,
                false, -1, nullptr, nullptr, &mpi_ctx_, 0);
        }));

    const float *rocm_output = output_tensor->data();
    ASSERT_NE(rocm_output, nullptr);
    ASSERT_FALSE(hasNaNOrInf(rocm_output, out_size));

    const double cosine = cosineSimilarity(rocm_output, cpu_output.data(), out_size);
    const double l2_error = relativeL2Error(rocm_output, cpu_output.data(), out_size);
    const double max_error = maxAbsError(rocm_output, cpu_output.data(), out_size);

    printComparisonStats("FlashDecode native FP16 KV Qwen3.5 long KVLen", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.998);
    EXPECT_LE(l2_error, 0.02);
    EXPECT_LE(max_error, 0.03);

    cleanupWorkspace(rocm_kernel);
}

TEST_F(Test__ROCmFlashAttentionParity, FlashDecode_NativeFP16KV_Qwen35TwoKVHeads_512Decode)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    constexpr int seq_len = 1;
    constexpr int kv_len = 513; // 512-token prefill plus the current decode token
    constexpr int n_heads = 16;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 256;
    constexpr int gqa_ratio = n_heads / n_kv_heads;
    const size_t q_size = static_cast<size_t>(seq_len) * n_heads * head_dim;
    const size_t kv_size = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;
    const size_t out_size = q_size;

    // Use a wider score distribution than the smoke test above.  The long-model
    // failure appears only once the head_dim=256 split-decode path sees the
    // exact Qwen3.5 GQA shape and non-trivial score dynamic range.
    auto Q_data = randomFP32Scaled(q_size, 1.0f);
    auto K_data_fp32 = randomFP32Scaled(kv_size, 1.0f);
    auto V_data_fp32 = randomFP32Scaled(kv_size, 0.5f);
    std::vector<uint16_t> K_data_fp16(kv_size);
    std::vector<uint16_t> V_data_fp16(kv_size);
    quantizeToFP16(K_data_fp32.data(), K_data_fp16.data(), kv_size);
    quantizeToFP16(V_data_fp32.data(), V_data_fp16.data(), kv_size);

    std::vector<float> K_ref_fp32(kv_size);
    std::vector<float> V_ref_fp32(kv_size);
    dequantizeFP16(K_data_fp16.data(), K_ref_fp32.data(), kv_size);
    dequantizeFP16(V_data_fp16.data(), V_ref_fp32.data(), kv_size);

    std::vector<float> cpu_output(out_size, 0.0f);
    for (int head = 0; head < n_heads; ++head)
    {
        const int kv_head = head / gqa_ratio;
        const float *q = Q_data.data() + static_cast<size_t>(head) * head_dim;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        std::vector<float> scores(static_cast<size_t>(kv_len));
        float max_score = -std::numeric_limits<float>::infinity();

        for (int kv_pos = 0; kv_pos < kv_len; ++kv_pos)
        {
            const float *k = K_ref_fp32.data() +
                             (static_cast<size_t>(kv_pos) * n_kv_heads + kv_head) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d)
            {
                dot += q[d] * k[d];
            }
            const float score = dot * scale;
            scores[static_cast<size_t>(kv_pos)] = score;
            max_score = std::max(max_score, score);
        }

        double sum_exp = 0.0;
        float *out = cpu_output.data() + static_cast<size_t>(head) * head_dim;
        for (int kv_pos = 0; kv_pos < kv_len; ++kv_pos)
        {
            const float weight = std::exp(scores[static_cast<size_t>(kv_pos)] - max_score);
            const float *v = V_ref_fp32.data() +
                             (static_cast<size_t>(kv_pos) * n_kv_heads + kv_head) * head_dim;
            sum_exp += static_cast<double>(weight);
            for (int d = 0; d < head_dim; ++d)
            {
                out[d] += weight * v[d];
            }
        }

        const float inv_sum = static_cast<float>(1.0 / std::max(sum_exp, 1e-30));
        for (int d = 0; d < head_dim; ++d)
        {
            out[d] *= inv_sum;
        }
    }

    auto q_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * head_dim)});
    auto k_tensor = std::make_shared<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)},
        K_data_fp16);
    auto v_tensor = std::make_shared<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)},
        V_data_fp16);
    auto output_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * head_dim)});
    std::copy(Q_data.begin(), Q_data.end(), q_tensor->mutable_data());

    const DeviceId gpu_device = DeviceId::rocm(0);
    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);
    ASSERT_TRUE(setupWorkspace(rocm_kernel, seq_len, n_heads, head_dim));

    ASSERT_TRUE(with_gpu_coherence(
        gpu_device,
        {q_tensor.get(), k_tensor.get(), v_tensor.get()},
        {output_tensor.get()},
        [&]
        {
            return rocm_kernel.compute_tensor(
                q_tensor.get(), k_tensor.get(), v_tensor.get(), output_tensor.get(),
                1, seq_len, kv_len, n_heads, n_kv_heads, head_dim,
                false, -1, nullptr, nullptr, &mpi_ctx_, 0);
        }));

    const float *rocm_output = output_tensor->data();
    ASSERT_NE(rocm_output, nullptr);
    ASSERT_FALSE(hasNaNOrInf(rocm_output, out_size));

    const double cosine = cosineSimilarity(rocm_output, cpu_output.data(), out_size);
    const double l2_error = relativeL2Error(rocm_output, cpu_output.data(), out_size);
    const double max_error = maxAbsError(rocm_output, cpu_output.data(), out_size);

    printComparisonStats("FlashDecode native FP16 KV Qwen3.5 2KV 512-decode", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.998);
    EXPECT_LE(l2_error, 0.02);
    EXPECT_LE(max_error, 0.03);

    cleanupWorkspace(rocm_kernel);
}

TEST_F(Test__ROCmFlashAttentionParity, FlashDecode_Q81KVCacheConsumption_Parity)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    constexpr int kv_len = 128;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    constexpr int seq_len = 1;

    const size_t q_size = static_cast<size_t>(seq_len) * n_heads * head_dim;
    const size_t kv_size = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;
    const size_t out_size = static_cast<size_t>(seq_len) * n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data_fp32 = randomFP32(kv_size);
    auto V_data_fp32 = randomFP32(kv_size);

    std::vector<float> cpu_baseline_output(out_size, 0.0f);
    std::vector<float> cpu_q81_output(out_size, 0.0f);
    std::vector<float> rocm_q81_output(out_size, 0.0f);

    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;

    // Baseline decode with original FP32 K/V
    ASSERT_TRUE(cpu_kernel.compute_decode(
        Q_data.data(), K_data_fp32.data(), V_data_fp32.data(), cpu_baseline_output.data(),
        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        false));

    // Build Q8_1 cache and gather consumed K/V through IKVCache path
    MPIContext local_mpi_ctx(0, 1, MPI_COMM_WORLD);
    auto kv_cache = std::make_unique<CPURingKVCache<ActivationPrecision::Q8_1>>(
        local_mpi_ctx,
        1,      // layers
        1,      // batch_size
        kv_len, // max_seq_len
        n_kv_heads,
        head_dim,
        DeviceId::cpu());

    auto k_q81 = Q8_1Tensor::quantize_from_fp32(
        K_data_fp32.data(), {static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)});
    auto v_q81 = Q8_1Tensor::quantize_from_fp32(
        V_data_fp32.data(), {static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)});
    ASSERT_NE(k_q81, nullptr);
    ASSERT_NE(v_q81, nullptr);
    ASSERT_TRUE(kv_cache->append_kv(0, 0, k_q81.get(), v_q81.get(), kv_len));

    auto gathered_K_q81 = std::make_unique<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)});
    auto gathered_V_q81 = std::make_unique<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)});
    std::vector<int> kv_lens;
    int gathered_max = kv_cache->gather_kv_batched(0, 1, gathered_K_q81.get(), gathered_V_q81.get(), kv_lens);
    ASSERT_EQ(gathered_max, kv_len);
    ASSERT_EQ(kv_lens.size(), 1u);
    ASSERT_EQ(kv_lens[0], kv_len);

    const float *K_from_q81 = gathered_K_q81->fp32_data();
    const float *V_from_q81 = gathered_V_q81->fp32_data();
    ASSERT_NE(K_from_q81, nullptr);
    ASSERT_NE(V_from_q81, nullptr);

    // CPU decode consuming dequantized gathered K/V from Q8_1 cache
    ASSERT_TRUE(cpu_kernel.compute_decode(
        Q_data.data(), K_from_q81, V_from_q81, cpu_q81_output.data(),
        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        false));

    // ROCm decode consuming same dequantized gathered K/V
    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);
    ASSERT_TRUE(setupWorkspace(rocm_kernel, seq_len, n_heads, head_dim));

    float *d_Q = nullptr;
    float *d_K = nullptr;
    float *d_V = nullptr;
    float *d_out = nullptr;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_out, out_size * sizeof(float));

    hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K_from_q81, kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V_from_q81, kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_out, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute_decode(
        d_Q, d_K, d_V, d_out,
        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        false,
        0);
    hipDeviceSynchronize();
    ASSERT_TRUE(rocm_success);

    hipMemcpy(rocm_q81_output.data(), d_out, out_size * sizeof(float), hipMemcpyDeviceToHost);

    cleanupWorkspace(rocm_kernel);
    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_out);

    ASSERT_FALSE(hasNaNOrInf(cpu_q81_output.data(), out_size));
    ASSERT_FALSE(hasNaNOrInf(rocm_q81_output.data(), out_size));

    const double q81_rocm_cpu_cos = cosineSimilarity(rocm_q81_output.data(), cpu_q81_output.data(), out_size);
    const double q81_rocm_cpu_l2 = relativeL2Error(rocm_q81_output.data(), cpu_q81_output.data(), out_size);

    const double q81_vs_fp32_cos = cosineSimilarity(cpu_q81_output.data(), cpu_baseline_output.data(), out_size);
    const double q81_vs_fp32_l2 = relativeL2Error(cpu_q81_output.data(), cpu_baseline_output.data(), out_size);

    printComparisonStats("FlashDecode Q8_1-consumed ROCm vs CPU", q81_rocm_cpu_cos, q81_rocm_cpu_l2,
                         maxAbsError(rocm_q81_output.data(), cpu_q81_output.data(), out_size), out_size);
    printComparisonStats("FlashDecode Q8_1-consumed CPU vs FP32 baseline", q81_vs_fp32_cos, q81_vs_fp32_l2,
                         maxAbsError(cpu_q81_output.data(), cpu_baseline_output.data(), out_size), out_size);

    EXPECT_GE(q81_rocm_cpu_cos, 0.99) << "ROCm vs CPU parity too low for Q8_1-consumed path";
    EXPECT_LE(q81_rocm_cpu_l2, 0.05) << "ROCm vs CPU L2 too high for Q8_1-consumed path";

    EXPECT_GE(q81_vs_fp32_cos, 0.95) << "Q8_1-consumed drift vs FP32 baseline too high";
    EXPECT_LE(q81_vs_fp32_l2, 0.15) << "Q8_1-consumed L2 drift vs FP32 baseline too high";

    LOG_INFO("[FlashDecode_Q81KVCacheConsumption_Parity] PASSED");
}

TEST_F(Test__ROCmFlashAttentionParity, FlashDecode_Q81KVCacheConsumption_NonCausal_MHA)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    constexpr int kv_len = 96;
    constexpr int n_heads = 8;
    constexpr int n_kv_heads = 8; // MHA
    constexpr int head_dim = 64;
    constexpr int seq_len = 1;

    const size_t q_size = static_cast<size_t>(seq_len) * n_heads * head_dim;
    const size_t kv_size = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;
    const size_t out_size = static_cast<size_t>(seq_len) * n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data_fp32 = randomFP32(kv_size);
    auto V_data_fp32 = randomFP32(kv_size);

    std::vector<float> cpu_baseline_output(out_size, 0.0f);
    std::vector<float> cpu_q81_output(out_size, 0.0f);
    std::vector<float> rocm_q81_output(out_size, 0.0f);

    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;

    ASSERT_TRUE(cpu_kernel.compute_decode(
        Q_data.data(), K_data_fp32.data(), V_data_fp32.data(), cpu_baseline_output.data(),
        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        false));

    MPIContext local_mpi_ctx(0, 1, MPI_COMM_WORLD);
    auto kv_cache = std::make_unique<CPURingKVCache<ActivationPrecision::Q8_1>>(
        local_mpi_ctx, 1, 1, kv_len, n_kv_heads, head_dim, DeviceId::cpu());

    auto k_q81 = Q8_1Tensor::quantize_from_fp32(
        K_data_fp32.data(), {static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)});
    auto v_q81 = Q8_1Tensor::quantize_from_fp32(
        V_data_fp32.data(), {static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)});
    ASSERT_NE(k_q81, nullptr);
    ASSERT_NE(v_q81, nullptr);
    ASSERT_TRUE(kv_cache->append_kv(0, 0, k_q81.get(), v_q81.get(), kv_len));

    auto gathered_K_q81 = std::make_unique<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)});
    auto gathered_V_q81 = std::make_unique<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)});
    std::vector<int> kv_lens;
    int gathered_max = kv_cache->gather_kv_batched(0, 1, gathered_K_q81.get(), gathered_V_q81.get(), kv_lens);
    ASSERT_EQ(gathered_max, kv_len);

    const float *K_from_q81 = gathered_K_q81->fp32_data();
    const float *V_from_q81 = gathered_V_q81->fp32_data();
    ASSERT_NE(K_from_q81, nullptr);
    ASSERT_NE(V_from_q81, nullptr);

    ASSERT_TRUE(cpu_kernel.compute_decode(
        Q_data.data(), K_from_q81, V_from_q81, cpu_q81_output.data(),
        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        false));

    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);
    ASSERT_TRUE(setupWorkspace(rocm_kernel, seq_len, n_heads, head_dim));

    float *d_Q = nullptr;
    float *d_K = nullptr;
    float *d_V = nullptr;
    float *d_out = nullptr;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_out, out_size * sizeof(float));

    hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K_from_q81, kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V_from_q81, kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_out, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute_decode(
        d_Q, d_K, d_V, d_out,
        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        false,
        0);
    hipDeviceSynchronize();
    ASSERT_TRUE(rocm_success);

    hipMemcpy(rocm_q81_output.data(), d_out, out_size * sizeof(float), hipMemcpyDeviceToHost);

    cleanupWorkspace(rocm_kernel);
    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_out);

    const double q81_rocm_cpu_cos = cosineSimilarity(rocm_q81_output.data(), cpu_q81_output.data(), out_size);
    const double q81_rocm_cpu_l2 = relativeL2Error(rocm_q81_output.data(), cpu_q81_output.data(), out_size);
    const double q81_vs_fp32_cos = cosineSimilarity(cpu_q81_output.data(), cpu_baseline_output.data(), out_size);
    const double q81_vs_fp32_l2 = relativeL2Error(cpu_q81_output.data(), cpu_baseline_output.data(), out_size);

    printComparisonStats("FlashDecode Q8_1-consumed ROCm vs CPU (non-causal MHA)", q81_rocm_cpu_cos, q81_rocm_cpu_l2,
                         maxAbsError(rocm_q81_output.data(), cpu_q81_output.data(), out_size), out_size);

    EXPECT_GE(q81_rocm_cpu_cos, 0.99);
    EXPECT_LE(q81_rocm_cpu_l2, 0.05);
    EXPECT_GE(q81_vs_fp32_cos, 0.95);
    EXPECT_LE(q81_vs_fp32_l2, 0.15);

    LOG_INFO("[FlashDecode_Q81KVCacheConsumption_NonCausal_MHA] PASSED");
}

// ============================================================================
// Non-Causal Attention (for KV cache scenarios)
// ============================================================================

TEST_F(Test__ROCmFlashAttentionParity, FlashAttn2_FP32_NonCausal)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    // Non-causal attention (e.g., for cross-attention or encoder)
    constexpr int seq_len = 32;
    constexpr int n_heads = 8;
    constexpr int n_kv_heads = 8;
    constexpr int head_dim = 64;
    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = seq_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> rocm_output(out_size, 0.0f);

    // CPU reference
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        false, // NON-causal
        -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, -1);
    ASSERT_TRUE(cpu_success);

    // ROCm kernel
    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_output, out_size * sizeof(float));

    hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        false, // NON-causal
        -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), out_size));
    ASSERT_FALSE(hasNaNOrInf(cpu_output.data(), out_size));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(rocm_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 FP32 Non-Causal", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99) << "Cosine similarity too low";
    EXPECT_LE(l2_error, 0.05) << "L2 error too high";

    LOG_INFO("[FlashAttn2_FP32_NonCausal] PASSED");
}

// ============================================================================
// Additional Parity Tests (matching CUDA coverage)
// ============================================================================

TEST_F(Test__ROCmFlashAttentionParity, FlashAttn2_FP32_Large)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    // Large test case (longer sequence)
    constexpr int seq_len = 256;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = seq_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> rocm_output(out_size, 0.0f);

    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, -1);
    ASSERT_TRUE(cpu_success);

    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_output, out_size * sizeof(float));

    hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), out_size));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(rocm_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 FP32 Large", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99);
    EXPECT_LE(l2_error, 0.05);

    LOG_INFO("[FlashAttn2_FP32_Large] PASSED");
}

TEST_F(Test__ROCmFlashAttentionParity, FlashAttn2_FP32_Qwen35LongPrefillSampledRows)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    constexpr int seq_len = 1024;
    constexpr int n_heads = 16;
    constexpr int n_kv_heads = 4;
    constexpr int head_dim = 256;
    constexpr int gqa_ratio = n_heads / n_kv_heads;
    const size_t q_size = static_cast<size_t>(seq_len) * n_heads * head_dim;
    const size_t kv_size = static_cast<size_t>(seq_len) * n_kv_heads * head_dim;
    const size_t out_size = q_size;

    // This mirrors the Qwen3.5 35B FA buffer geometry while keeping the CPU
    // oracle cheap by checking only representative causal rows.
    auto Q_data = randomFP32Scaled(q_size, 0.35f);
    auto K_data = randomFP32Scaled(kv_size, 0.35f);
    auto V_data = randomFP32Scaled(kv_size, 0.35f);
    std::vector<float> rocm_output(out_size, 0.0f);

    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    float *d_Q = nullptr;
    float *d_K = nullptr;
    float *d_V = nullptr;
    float *d_output = nullptr;
    ASSERT_EQ(hipMalloc(&d_Q, q_size * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_K, kv_size * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_V, kv_size * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_output, out_size * sizeof(float)), hipSuccess);

    ASSERT_EQ(hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemset(d_output, 0, out_size * sizeof(float)), hipSuccess);

    const bool rocm_success = rocm_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    ASSERT_TRUE(rocm_success);
    ASSERT_EQ(hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), out_size));

    auto computeReferenceRow = [&](int query_row, int head) -> std::vector<float>
    {
        std::vector<float> scores(static_cast<size_t>(query_row + 1));
        const int kv_head = head / gqa_ratio;
        const float *q = Q_data.data() +
                         (static_cast<size_t>(query_row) * n_heads + head) * head_dim;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        float max_score = -std::numeric_limits<float>::infinity();

        // First pass: stable causal score maximum for this sampled row/head.
        for (int kv_pos = 0; kv_pos <= query_row; ++kv_pos)
        {
            const float *k = K_data.data() +
                             (static_cast<size_t>(kv_pos) * n_kv_heads + kv_head) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d)
            {
                dot += q[d] * k[d];
            }
            const float score = dot * scale;
            scores[static_cast<size_t>(kv_pos)] = score;
            max_score = std::max(max_score, score);
        }

        std::vector<float> ref(static_cast<size_t>(head_dim), 0.0f);
        double sum_exp = 0.0;

        // Second pass: normalize scores and accumulate the V-weighted row.
        for (int kv_pos = 0; kv_pos <= query_row; ++kv_pos)
        {
            const float weight = std::exp(scores[static_cast<size_t>(kv_pos)] - max_score);
            const float *v = V_data.data() +
                             (static_cast<size_t>(kv_pos) * n_kv_heads + kv_head) * head_dim;
            sum_exp += static_cast<double>(weight);
            for (int d = 0; d < head_dim; ++d)
            {
                ref[static_cast<size_t>(d)] += weight * v[d];
            }
        }

        const float inv_sum = static_cast<float>(1.0 / std::max(sum_exp, 1e-30));
        for (float &value : ref)
        {
            value *= inv_sum;
        }
        return ref;
    };

    const std::vector<int> sampled_rows = {0, 127, 511, 1023};
    double worst_cosine = 1.0;
    double worst_l2 = 0.0;
    double worst_max_error = 0.0;
    for (int row : sampled_rows)
    {
        for (int head = 0; head < n_heads; ++head)
        {
            const std::vector<float> ref = computeReferenceRow(row, head);
            const float *actual = rocm_output.data() +
                                  (static_cast<size_t>(row) * n_heads + head) * head_dim;
            const double cosine = cosineSimilarity(actual, ref.data(), ref.size());
            const double l2_error = relativeL2Error(actual, ref.data(), ref.size());
            const double max_error = maxAbsError(actual, ref.data(), ref.size());
            worst_cosine = std::min(worst_cosine, cosine);
            worst_l2 = std::max(worst_l2, l2_error);
            worst_max_error = std::max(worst_max_error, max_error);
        }
    }

    printComparisonStats("FlashAttn2 FP32 Qwen3.5 long prefill sampled rows",
                         worst_cosine, worst_l2, worst_max_error,
                         sampled_rows.size() * n_heads * head_dim);

    EXPECT_GE(worst_cosine, 0.999);
    EXPECT_LE(worst_l2, 0.01);
    EXPECT_LE(worst_max_error, 0.01);
}

TEST_F(Test__ROCmFlashAttentionParity, FlashAttn2_NativeFP16KV_Qwen35LongPrefillSampledRows)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    constexpr int seq_len = 1024;
    constexpr int n_heads = 16;
    constexpr int n_kv_heads = 4;
    constexpr int head_dim = 256;
    constexpr int gqa_ratio = n_heads / n_kv_heads;
    const size_t q_size = static_cast<size_t>(seq_len) * n_heads * head_dim;
    const size_t kv_size = static_cast<size_t>(seq_len) * n_kv_heads * head_dim;
    const size_t out_size = q_size;

    // Production ROCm prefill reads K/V back from the FP16 KV cache, so cover
    // the native FP16-KV dispatch separately from the all-FP32 path above.
    auto Q_data = randomFP32Scaled(q_size, 0.35f);
    auto K_data_fp32 = randomFP32Scaled(kv_size, 0.35f);
    auto V_data_fp32 = randomFP32Scaled(kv_size, 0.35f);
    std::vector<uint16_t> K_data_fp16(kv_size);
    std::vector<uint16_t> V_data_fp16(kv_size);
    quantizeToFP16(K_data_fp32.data(), K_data_fp16.data(), kv_size);
    quantizeToFP16(V_data_fp32.data(), V_data_fp16.data(), kv_size);

    std::vector<float> K_ref_fp32(kv_size);
    std::vector<float> V_ref_fp32(kv_size);
    dequantizeFP16(K_data_fp16.data(), K_ref_fp32.data(), kv_size);
    dequantizeFP16(V_data_fp16.data(), V_ref_fp32.data(), kv_size);

    auto q_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * head_dim)});
    auto k_tensor = std::make_shared<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)},
        K_data_fp16);
    auto v_tensor = std::make_shared<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)},
        V_data_fp16);
    auto output_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * head_dim)});
    std::copy(Q_data.begin(), Q_data.end(), q_tensor->mutable_data());

    const DeviceId gpu_device = DeviceId::rocm(0);
    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    ASSERT_TRUE(with_gpu_coherence(
        gpu_device,
        {q_tensor.get(), k_tensor.get(), v_tensor.get()},
        {output_tensor.get()},
        [&]
        {
            return rocm_kernel.compute_tensor(
                q_tensor.get(), k_tensor.get(), v_tensor.get(), output_tensor.get(),
                1, seq_len, seq_len, n_heads, n_kv_heads, head_dim,
                true, -1, nullptr, nullptr, &mpi_ctx_, 0);
        }));

    const float *rocm_output = output_tensor->data();
    ASSERT_NE(rocm_output, nullptr);
    ASSERT_FALSE(hasNaNOrInf(rocm_output, out_size));

    auto computeReferenceRow = [&](int query_row, int head) -> std::vector<float>
    {
        std::vector<float> scores(static_cast<size_t>(query_row + 1));
        const int kv_head = head / gqa_ratio;
        const float *q = Q_data.data() +
                         (static_cast<size_t>(query_row) * n_heads + head) * head_dim;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        float max_score = -std::numeric_limits<float>::infinity();

        // Match the runtime path: FP32 Q, FP16-cache K/V dequantized to FP32,
        // causal attention over all preceding prompt positions.
        for (int kv_pos = 0; kv_pos <= query_row; ++kv_pos)
        {
            const float *k = K_ref_fp32.data() +
                             (static_cast<size_t>(kv_pos) * n_kv_heads + kv_head) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d)
            {
                dot += q[d] * k[d];
            }
            const float score = dot * scale;
            scores[static_cast<size_t>(kv_pos)] = score;
            max_score = std::max(max_score, score);
        }

        std::vector<float> ref(static_cast<size_t>(head_dim), 0.0f);
        double sum_exp = 0.0;
        for (int kv_pos = 0; kv_pos <= query_row; ++kv_pos)
        {
            const float weight = std::exp(scores[static_cast<size_t>(kv_pos)] - max_score);
            const float *v = V_ref_fp32.data() +
                             (static_cast<size_t>(kv_pos) * n_kv_heads + kv_head) * head_dim;
            sum_exp += static_cast<double>(weight);
            for (int d = 0; d < head_dim; ++d)
            {
                ref[static_cast<size_t>(d)] += weight * v[d];
            }
        }

        const float inv_sum = static_cast<float>(1.0 / std::max(sum_exp, 1e-30));
        for (float &value : ref)
        {
            value *= inv_sum;
        }
        return ref;
    };

    const std::vector<int> sampled_rows = {0, 127, 511, 1023};
    double worst_cosine = 1.0;
    double worst_l2 = 0.0;
    double worst_max_error = 0.0;
    for (int row : sampled_rows)
    {
        for (int head = 0; head < n_heads; ++head)
        {
            const std::vector<float> ref = computeReferenceRow(row, head);
            const float *actual = rocm_output +
                                  (static_cast<size_t>(row) * n_heads + head) * head_dim;
            const double cosine = cosineSimilarity(actual, ref.data(), ref.size());
            const double l2_error = relativeL2Error(actual, ref.data(), ref.size());
            const double max_error = maxAbsError(actual, ref.data(), ref.size());
            worst_cosine = std::min(worst_cosine, cosine);
            worst_l2 = std::max(worst_l2, l2_error);
            worst_max_error = std::max(worst_max_error, max_error);
        }
    }

    printComparisonStats("FlashAttn2 native FP16 KV Qwen3.5 long prefill sampled rows",
                         worst_cosine, worst_l2, worst_max_error,
                         sampled_rows.size() * n_heads * head_dim);

    EXPECT_GE(worst_cosine, 0.998);
    EXPECT_LE(worst_l2, 0.02);
    EXPECT_LE(worst_max_error, 0.03);
}

TEST_F(Test__ROCmFlashAttentionParity, FlashAttn2_NativeFP16KV_Qwen2ScaleStress)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    constexpr int seq_len = 64;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    constexpr float scale = 4.0f;

    const size_t q_size = static_cast<size_t>(seq_len) * n_heads * head_dim;
    const size_t kv_size = static_cast<size_t>(seq_len) * n_kv_heads * head_dim;
    const size_t out_size = q_size;

    auto Q_data = randomFP32Scaled(q_size, scale);
    auto K_data_fp32 = randomFP32Scaled(kv_size, scale);
    auto V_data_fp32 = randomFP32Scaled(kv_size, scale);

    std::vector<uint16_t> K_data_fp16(kv_size);
    std::vector<uint16_t> V_data_fp16(kv_size);
    quantizeToFP16(K_data_fp32.data(), K_data_fp16.data(), kv_size);
    quantizeToFP16(V_data_fp32.data(), V_data_fp16.data(), kv_size);

    std::vector<float> K_ref_fp32(kv_size);
    std::vector<float> V_ref_fp32(kv_size);
    dequantizeFP16(K_data_fp16.data(), K_ref_fp32.data(), kv_size);
    dequantizeFP16(V_data_fp16.data(), V_ref_fp32.data(), kv_size);

    std::vector<float> cpu_output(out_size, 0.0f);
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    ASSERT_TRUE(cpu_kernel.compute(
        Q_data.data(), K_ref_fp32.data(), V_ref_fp32.data(), cpu_output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, -1));

    auto q_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * head_dim)});
    auto k_tensor = std::make_shared<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)},
        K_data_fp16);
    auto v_tensor = std::make_shared<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)},
        V_data_fp16);
    auto output_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * head_dim)});
    std::copy(Q_data.begin(), Q_data.end(), q_tensor->mutable_data());

    const DeviceId gpu_device = DeviceId::rocm(0);
    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    ASSERT_TRUE(with_gpu_coherence(
        gpu_device,
        {q_tensor.get(), k_tensor.get(), v_tensor.get()},
        {output_tensor.get()},
        [&]
        {
            return rocm_kernel.compute_tensor(
                q_tensor.get(), k_tensor.get(), v_tensor.get(), output_tensor.get(),
                1, seq_len, seq_len, n_heads, n_kv_heads, head_dim,
                true, -1, nullptr, nullptr, &mpi_ctx_, 0);
        }));

    const float *rocm_output = output_tensor->data();
    ASSERT_NE(rocm_output, nullptr);
    ASSERT_FALSE(hasNaNOrInf(rocm_output, out_size));

    const double cosine = cosineSimilarity(rocm_output, cpu_output.data(), out_size);
    const double l2_error = relativeL2Error(rocm_output, cpu_output.data(), out_size);
    const double max_error = maxAbsError(rocm_output, cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 native FP16 KV Qwen2 scale-stress", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99);
    EXPECT_LE(l2_error, 0.05);
}

TEST_F(Test__ROCmFlashAttentionParity, FlashAttn2_NativeQ81KV_Qwen2ScaleStress)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    constexpr int seq_len = 64;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    constexpr float scale = 4.0f;

    const size_t q_size = static_cast<size_t>(seq_len) * n_heads * head_dim;
    const size_t kv_size = static_cast<size_t>(seq_len) * n_kv_heads * head_dim;
    const size_t out_size = q_size;

    auto Q_data = randomFP32Scaled(q_size, scale);
    auto K_data_fp32 = randomFP32Scaled(kv_size, scale);
    auto V_data_fp32 = randomFP32Scaled(kv_size, scale);

    auto k_tensor = Q8_1Tensor::quantize_from_fp32(
        K_data_fp32.data(), {static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)});
    auto v_tensor = Q8_1Tensor::quantize_from_fp32(
        V_data_fp32.data(), {static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)});
    ASSERT_NE(k_tensor, nullptr);
    ASSERT_NE(v_tensor, nullptr);

    const float *K_ref_fp32 = k_tensor->fp32_data();
    const float *V_ref_fp32 = v_tensor->fp32_data();
    ASSERT_NE(K_ref_fp32, nullptr);
    ASSERT_NE(V_ref_fp32, nullptr);

    std::vector<float> cpu_output(out_size, 0.0f);
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    ASSERT_TRUE(cpu_kernel.compute(
        Q_data.data(), K_ref_fp32, V_ref_fp32, cpu_output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, -1));

    auto q_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * head_dim)});
    auto output_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * head_dim)});
    std::copy(Q_data.begin(), Q_data.end(), q_tensor->mutable_data());

    const DeviceId gpu_device = DeviceId::rocm(0);
    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    ASSERT_TRUE(with_gpu_coherence(
        gpu_device,
        {q_tensor.get(), k_tensor.get(), v_tensor.get()},
        {output_tensor.get()},
        [&]
        {
            return rocm_kernel.compute_tensor(
                q_tensor.get(), k_tensor.get(), v_tensor.get(), output_tensor.get(),
                1, seq_len, seq_len, n_heads, n_kv_heads, head_dim,
                true, -1, nullptr, nullptr, &mpi_ctx_, 0);
        }));

    const float *rocm_output = output_tensor->data();
    ASSERT_NE(rocm_output, nullptr);
    ASSERT_FALSE(hasNaNOrInf(rocm_output, out_size));

    const double cosine = cosineSimilarity(rocm_output, cpu_output.data(), out_size);
    const double l2_error = relativeL2Error(rocm_output, cpu_output.data(), out_size);
    const double max_error = maxAbsError(rocm_output, cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 native Q8_1 KV Qwen2 scale-stress", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99);
    EXPECT_LE(l2_error, 0.05);
}

TEST_F(Test__ROCmFlashAttentionParity, FlashAttn2_RealQwen2Layer3ContextParity)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    std::ifstream model_file(TEST_MODEL_PATH);
    if (!model_file.good())
    {
        GTEST_SKIP() << "Test model not found: " << TEST_MODEL_PATH;
    }

    auto prefix_ctx = ModelContext::createForPPStage(TEST_MODEL_PATH, 0, 4, true, false);
    ASSERT_NE(prefix_ctx, nullptr);

    FactoryPPStageConfig config;
    config.first_layer = 0;
    config.last_layer = 4;
    config.has_embedding = true;
    config.has_lm_head = false;
    ASSERT_TRUE(config.isValid());

    auto cpu_runner = createPPStageRunner(prefix_ctx, DeviceId::cpu(), config);
    ASSERT_NE(cpu_runner, nullptr);
    cpu_runner->enableSnapshotCapture();

    constexpr int seq_len = 64;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    const size_t q_size = static_cast<size_t>(seq_len) * n_heads * head_dim;
    const size_t kv_size = static_cast<size_t>(seq_len) * n_kv_heads * head_dim;

    std::vector<int> tokens(seq_len);
    for (int i = 0; i < seq_len; ++i)
    {
        tokens[i] = i % 1024;
    }
    ASSERT_TRUE(cpu_runner->forward(tokens.data(), seq_len));

    auto getSnapshotVector = [&](const std::string &key, size_t expected_size) -> std::vector<float>
    {
        size_t size = 0;
        const float *data = cpu_runner->getSnapshot(key, size);
        EXPECT_NE(data, nullptr) << "Missing snapshot: " << key;
        EXPECT_EQ(size, expected_size) << "Unexpected snapshot size for " << key;
        if (!data || size != expected_size)
        {
            return {};
        }
        return std::vector<float>(data, data + size);
    };

    const std::vector<float> q_data = getSnapshotVector("layer3_Q_ROPE", q_size);
    const std::vector<float> k_data = getSnapshotVector("layer3_K_ROPE", kv_size);
    const std::vector<float> v_data = getSnapshotVector("layer3_V_PROJECTION", kv_size);
    ASSERT_EQ(q_data.size(), q_size);
    ASSERT_EQ(k_data.size(), kv_size);
    ASSERT_EQ(v_data.size(), kv_size);

    std::vector<float> cpu_output(q_size, 0.0f);
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    ASSERT_TRUE(cpu_kernel.compute(
        q_data.data(), k_data.data(), v_data.data(), cpu_output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, -1));

    auto q_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * head_dim)});
    auto k_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)});
    auto v_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)});
    auto output_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * head_dim)});
    std::copy(q_data.begin(), q_data.end(), q_tensor->mutable_data());
    std::copy(k_data.begin(), k_data.end(), k_tensor->mutable_data());
    std::copy(v_data.begin(), v_data.end(), v_tensor->mutable_data());

    const DeviceId gpu_device = DeviceId::rocm(0);
    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    ASSERT_TRUE(with_gpu_coherence(
        gpu_device,
        {q_tensor.get(), k_tensor.get(), v_tensor.get()},
        {output_tensor.get()},
        [&]
        {
            return rocm_kernel.compute_tensor(
                q_tensor.get(), k_tensor.get(), v_tensor.get(), output_tensor.get(),
                1, seq_len, seq_len, n_heads, n_kv_heads, head_dim,
                true, -1, nullptr, nullptr, &mpi_ctx_, 0);
        }));

    const float *rocm_output = output_tensor->data();
    ASSERT_NE(rocm_output, nullptr);
    ASSERT_FALSE(hasNaNOrInf(rocm_output, q_size));

    const double cosine = cosineSimilarity(rocm_output, cpu_output.data(), q_size);
    const double l2_error = relativeL2Error(rocm_output, cpu_output.data(), q_size);
    const double max_error = maxAbsError(rocm_output, cpu_output.data(), q_size);

    printComparisonStats("FlashAttn2 real Qwen2 layer3 context", cosine, l2_error, max_error, q_size);

    EXPECT_GE(cosine, 0.99)
        << "ROCm FlashAttention diverges on real layer-3 Qwen2 attention inputs";
    EXPECT_LE(l2_error, 0.05)
        << "ROCm FlashAttention L2 error too high on real layer-3 Qwen2 attention inputs";
}

TEST_F(Test__ROCmFlashAttentionParity, FlashAttn2_RealQwen2Layer3InputSensitivity)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    std::ifstream model_file(TEST_MODEL_PATH);
    if (!model_file.good())
    {
        GTEST_SKIP() << "Test model not found: " << TEST_MODEL_PATH;
    }

    auto stage_ctx = ModelContext::createForPPStage(TEST_MODEL_PATH, 0, 4, true, false);
    ASSERT_NE(stage_ctx, nullptr);

    FactoryPPStageConfig config;
    config.first_layer = 0;
    config.last_layer = 4;
    config.has_embedding = true;
    config.has_lm_head = false;
    ASSERT_TRUE(config.isValid());

    auto cpu_runner = createPPStageRunner(stage_ctx, DeviceId::cpu(), config);
    auto rocm_runner = createPPStageRunner(stage_ctx, DeviceId::rocm(0), config);
    ASSERT_NE(cpu_runner, nullptr);
    ASSERT_NE(rocm_runner, nullptr);

    cpu_runner->enableSnapshotCapture();
    rocm_runner->enableSnapshotCapture();

    constexpr int seq_len = 64;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    const size_t q_size = static_cast<size_t>(seq_len) * n_heads * head_dim;
    const size_t kv_size = static_cast<size_t>(seq_len) * n_kv_heads * head_dim;

    std::vector<int> tokens(seq_len);
    for (int i = 0; i < seq_len; ++i)
    {
        tokens[i] = i % 1024;
    }
    ASSERT_TRUE(cpu_runner->forward(tokens.data(), seq_len));
    ASSERT_TRUE(rocm_runner->forward(tokens.data(), seq_len));

    auto getSnapshotVector = [&](IInferenceRunner *runner, const std::string &key, size_t expected_size) -> std::vector<float>
    {
        size_t size = 0;
        const float *data = runner->getSnapshot(key, size);
        EXPECT_NE(data, nullptr) << "Missing snapshot: " << key;
        EXPECT_EQ(size, expected_size) << "Unexpected snapshot size for " << key;
        if (!data || size != expected_size)
        {
            return {};
        }
        return std::vector<float>(data, data + size);
    };

    const std::vector<float> cpu_q = getSnapshotVector(cpu_runner.get(), "layer3_Q_ROPE", q_size);
    const std::vector<float> cpu_k = getSnapshotVector(cpu_runner.get(), "layer3_K_ROPE", kv_size);
    const std::vector<float> cpu_v = getSnapshotVector(cpu_runner.get(), "layer3_V_PROJECTION", kv_size);
    const std::vector<float> rocm_q = getSnapshotVector(rocm_runner.get(), "layer3_Q_ROPE", q_size);
    const std::vector<float> rocm_k = getSnapshotVector(rocm_runner.get(), "layer3_K_ROPE", kv_size);
    const std::vector<float> rocm_v = getSnapshotVector(rocm_runner.get(), "layer3_V_PROJECTION", kv_size);

    ASSERT_EQ(cpu_q.size(), q_size);
    ASSERT_EQ(cpu_k.size(), kv_size);
    ASSERT_EQ(cpu_v.size(), kv_size);
    ASSERT_EQ(rocm_q.size(), q_size);
    ASSERT_EQ(rocm_k.size(), kv_size);
    ASSERT_EQ(rocm_v.size(), kv_size);

    auto runCpuAttention = [&](const std::vector<float> &q,
                               const std::vector<float> &k,
                               const std::vector<float> &v) -> std::vector<float>
    {
        std::vector<float> output(q_size, 0.0f);
        CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
        EXPECT_TRUE(cpu_kernel.compute(
            q.data(), k.data(), v.data(), output.data(),
            seq_len, n_heads, n_kv_heads, head_dim,
            true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, -1));
        return output;
    };

    const std::vector<float> ref_output = runCpuAttention(cpu_q, cpu_k, cpu_v);
    const std::vector<float> rocm_input_output = runCpuAttention(rocm_q, rocm_k, rocm_v);
    const std::vector<float> q_only_output = runCpuAttention(rocm_q, cpu_k, cpu_v);
    const std::vector<float> k_only_output = runCpuAttention(cpu_q, rocm_k, cpu_v);
    const std::vector<float> v_only_output = runCpuAttention(cpu_q, cpu_k, rocm_v);

    const double all_inputs_cos = cosineSimilarity(rocm_input_output.data(), ref_output.data(), q_size);
    const double q_only_cos = cosineSimilarity(q_only_output.data(), ref_output.data(), q_size);
    const double k_only_cos = cosineSimilarity(k_only_output.data(), ref_output.data(), q_size);
    const double v_only_cos = cosineSimilarity(v_only_output.data(), ref_output.data(), q_size);

    std::cout << "  Layer3 input sensitivity: all_inputs_cos=" << std::fixed << std::setprecision(6) << all_inputs_cos
              << ", q_only_cos=" << q_only_cos
              << ", k_only_cos=" << k_only_cos
              << ", v_only_cos=" << v_only_cos << std::endl;

    // With rocBLAS prefill, GPU Q/K/V snapshots closely match CPU reference.
    // (The old cooperative kernel diverged here — cosine < 0.995.)
    EXPECT_GT(all_inputs_cos, 0.999)
        << "ROCm layer-3 Q/K/V snapshots should produce attention-context close to CPU reference";
}

TEST_F(Test__ROCmFlashAttentionParity, FlashDecode_FP32_VeryLong_Parity)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    // Very long KV cache - stress test for split-K with many splits
    constexpr int kv_len = 1024;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    constexpr int seq_len = 1;
    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = kv_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> rocm_output(out_size, 0.0f);

    // CPU reference using production CPUFlashAttentionKernelT::compute_decode()
    //
    // NOTE: For seq_len=1 decode, production (AttentionComputeStage.cpp) always
    // passes causal=false because the query is implicitly at position (kv_len-1)
    // and a causal mask would be all-zeros (sees everything). Passing causal=true
    // with position_offset=0 means "query at absolute position 0" — the CPU kernel
    // would mask out kv[1..kv_len-1] and only attend to kv[0], which is not what
    // this test intends to validate. The ROCm decode kernel does not honor a
    // causal flag at all (always behaves like causal=false), so causal=false on
    // the CPU side is the only setting that produces a meaningful parity check.
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute_decode(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, kv_len, n_heads, n_kv_heads, head_dim, false);
    ASSERT_TRUE(cpu_success) << "CPU compute_decode failed";

    // ROCm decode
    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    // Setup workspace for FlashDecode (required for multi-block reduction)
    ASSERT_TRUE(setupWorkspace(rocm_kernel, seq_len, n_heads, head_dim));

    float *d_Q, *d_K, *d_V, *d_output;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_output, out_size * sizeof(float));

    hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        false, // causal (see note above — production also uses false for seq_len=1)
        0);    // position_offset
    hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    // Cleanup workspace before freeing GPU memory
    cleanupWorkspace(rocm_kernel);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), out_size));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(rocm_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashDecode FP32 VeryLong Parity (kv=1024, split-K)", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99) << "Cosine similarity too low";
    EXPECT_LE(l2_error, 0.05) << "L2 error too high";

    LOG_INFO("[FlashDecode_FP32_VeryLong_Parity] PASSED");
}

TEST_F(Test__ROCmFlashAttentionParity, FlashDecode_FP32_MHA_Parity)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    // Multi-head attention (not GQA) - n_heads == n_kv_heads
    constexpr int kv_len = 256;
    constexpr int n_heads = 8;
    constexpr int n_kv_heads = 8; // MHA
    constexpr int head_dim = 64;
    constexpr int seq_len = 1;
    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = kv_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> rocm_output(out_size, 0.0f);

    // CPU reference using production CPUFlashAttentionKernelT::compute_decode()
    // (see VeryLong test above for note on why causal=false is correct here)
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute_decode(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, kv_len, n_heads, n_kv_heads, head_dim, false);
    ASSERT_TRUE(cpu_success) << "CPU compute_decode failed";

    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    // Setup workspace for FlashDecode (required for multi-block reduction)
    ASSERT_TRUE(setupWorkspace(rocm_kernel, seq_len, n_heads, head_dim));
    float *d_Q, *d_K, *d_V, *d_output;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_output, out_size * sizeof(float));

    hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        false, // causal (see VeryLong test for explanation)
        0);    // position_offset
    hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    // Cleanup workspace before freeing GPU memory
    cleanupWorkspace(rocm_kernel);
    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), out_size));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(rocm_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashDecode FP32 MHA Parity (n_heads=n_kv_heads)", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99);
    EXPECT_LE(l2_error, 0.05);

    LOG_INFO("[FlashDecode_FP32_MHA_Parity] PASSED");
}

TEST_F(Test__ROCmFlashAttentionParity, FlashDecode_FP32_HeadDim128_Parity)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    // Llama-style head_dim=128
    constexpr int kv_len = 256;
    constexpr int n_heads = 8;
    constexpr int n_kv_heads = 8;
    constexpr int head_dim = 128;
    constexpr int seq_len = 1;
    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = kv_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> rocm_output(out_size, 0.0f);

    // CPU reference using production CPUFlashAttentionKernelT::compute_decode()
    // (see VeryLong test above for note on why causal=false is correct here)
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute_decode(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, kv_len, n_heads, n_kv_heads, head_dim, false);
    ASSERT_TRUE(cpu_success) << "CPU compute_decode failed";

    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    // Setup workspace for FlashDecode (required for multi-block reduction)
    ASSERT_TRUE(setupWorkspace(rocm_kernel, seq_len, n_heads, head_dim));
    float *d_Q, *d_K, *d_V, *d_output;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_output, out_size * sizeof(float));

    hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        false, // causal (see VeryLong test for explanation)
        0);    // position_offset
    hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    // Cleanup workspace before freeing GPU memory
    cleanupWorkspace(rocm_kernel);
    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), out_size));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(rocm_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashDecode FP32 HeadDim128 Parity", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99);
    EXPECT_LE(l2_error, 0.05);

    LOG_INFO("[FlashDecode_FP32_HeadDim128_Parity] PASSED");
}

TEST_F(Test__ROCmFlashAttentionParity, FlashDecode_FP32_NonCausal_Parity)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    // Non-causal decode (bidirectional attention)
    constexpr int kv_len = 128;
    constexpr int n_heads = 8;
    constexpr int n_kv_heads = 8;
    constexpr int head_dim = 64;
    constexpr int seq_len = 1;
    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = kv_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> rocm_output(out_size, 0.0f);

    // CPU reference using production CPUFlashAttentionKernelT::compute_decode()
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute_decode(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, kv_len, n_heads, n_kv_heads, head_dim, false); // non-causal
    ASSERT_TRUE(cpu_success) << "CPU compute_decode failed";

    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    // Setup workspace for FlashDecode (required for multi-block reduction)
    ASSERT_TRUE(setupWorkspace(rocm_kernel, seq_len, n_heads, head_dim));
    float *d_Q, *d_K, *d_V, *d_output;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_output, out_size * sizeof(float));

    hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        false, // non-causal
        0);    // position_offset
    hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    // Cleanup workspace before freeing GPU memory
    cleanupWorkspace(rocm_kernel);
    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), out_size));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(rocm_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashDecode FP32 NonCausal Parity", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99);
    EXPECT_LE(l2_error, 0.05);

    LOG_INFO("[FlashDecode_FP32_NonCausal_Parity] PASSED");
}

TEST_F(Test__ROCmFlashAttentionParity, FlashDecode_BatchDecoding)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    // Test batch decoding: multiple independent sequences decoded in parallel
    // Each batch element has seq_len=1 (decode) with different KV cache lengths
    constexpr int batch_size = 4;
    constexpr int kv_len = 256; // Context length
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 2; // GQA
    constexpr int head_dim = 64;

    // For batch decoding, we process each batch sequentially using compute_decode
    // Q: [1, n_heads, head_dim] per batch
    // K/V: [kv_len, n_kv_heads, head_dim] per batch
    const size_t q_per_batch = 1 * n_heads * head_dim;
    const size_t kv_per_batch = kv_len * n_kv_heads * head_dim;
    const size_t out_per_batch = 1 * n_heads * head_dim;

    std::vector<float> Q_data = randomFP32(batch_size * q_per_batch);
    std::vector<float> K_data = randomFP32(batch_size * kv_per_batch);
    std::vector<float> V_data = randomFP32(batch_size * kv_per_batch);
    std::vector<float> rocm_output(batch_size * out_per_batch, 0.0f);

    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    // Setup workspace for FlashDecode (required for multi-block reduction)
    // Use batch_size=1 since we process one batch element at a time
    ASSERT_TRUE(setupWorkspace(rocm_kernel, 1, n_heads, head_dim));

    // Allocate device memory for largest batch element
    float *d_Q, *d_K, *d_V, *d_output;
    hipMalloc(&d_Q, q_per_batch * sizeof(float));
    hipMalloc(&d_K, kv_per_batch * sizeof(float));
    hipMalloc(&d_V, kv_per_batch * sizeof(float));
    hipMalloc(&d_output, out_per_batch * sizeof(float));

    bool all_success = true;

    // Process each batch element
    for (int b = 0; b < batch_size; b++)
    {
        hipMemcpy(d_Q, Q_data.data() + b * q_per_batch,
                  q_per_batch * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(d_K, K_data.data() + b * kv_per_batch,
                  kv_per_batch * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(d_V, V_data.data() + b * kv_per_batch,
                  kv_per_batch * sizeof(float), hipMemcpyHostToDevice);
        hipMemset(d_output, 0, out_per_batch * sizeof(float));

        // Use compute_decode for single-token decode
        bool success = rocm_kernel.compute_decode(
            d_Q, d_K, d_V, d_output,
            1,      // seq_len = 1 for decode
            kv_len, // kv_len from cache
            n_heads, n_kv_heads, head_dim,
            true, // causal
            0);   // position_offset
        hipDeviceSynchronize();

        if (!success)
        {
            std::cerr << "Batch " << b << " decode failed" << std::endl;
            all_success = false;
            continue;
        }

        hipMemcpy(rocm_output.data() + b * out_per_batch, d_output,
                  out_per_batch * sizeof(float), hipMemcpyDeviceToHost);
    }

    // Cleanup workspace before freeing GPU memory
    cleanupWorkspace(rocm_kernel);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    ASSERT_TRUE(all_success) << "All batch decode operations should succeed";
    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), batch_size * out_per_batch));

    // Verify each batch element has valid output
    bool all_batches_valid = true;
    for (int b = 0; b < batch_size; b++)
    {
        float batch_sum = 0.0f;
        float batch_max = -std::numeric_limits<float>::infinity();
        float batch_min = std::numeric_limits<float>::infinity();

        for (int h = 0; h < n_heads; h++)
        {
            for (int d = 0; d < head_dim; d++)
            {
                float val = rocm_output[b * out_per_batch + h * head_dim + d];
                batch_sum += val;
                batch_max = std::max(batch_max, val);
                batch_min = std::min(batch_min, val);
            }
        }

        // Each batch should have non-trivial output
        bool batch_valid = (batch_sum != 0.0f) &&
                           (batch_max != batch_min) &&
                           std::isfinite(batch_sum);

        if (!batch_valid)
        {
            std::cerr << "Batch " << b << " invalid: sum=" << batch_sum
                      << ", min=" << batch_min << ", max=" << batch_max << std::endl;
            all_batches_valid = false;
        }
    }

    EXPECT_TRUE(all_batches_valid) << "All batch elements should have valid, non-trivial output";

    // Verify batches are independent (different inputs should give different outputs)
    // Compare batch 0 and batch 1 outputs
    float diff_sum = 0.0f;
    for (size_t i = 0; i < out_per_batch; i++)
    {
        float diff = rocm_output[i] - rocm_output[out_per_batch + i];
        diff_sum += diff * diff;
    }
    EXPECT_GT(diff_sum, 0.0f) << "Different batch inputs should produce different outputs";

    std::cout << "  FlashDecode BatchDecoding: batch_size=" << batch_size
              << ", kv_len=" << kv_len << ", n_heads=" << n_heads
              << ", n_kv_heads=" << n_kv_heads << " - PASSED" << std::endl;

    LOG_INFO("[FlashDecode_BatchDecoding] PASSED");
}

TEST_F(Test__ROCmFlashAttentionParity, FlashAttn2_CausalMasking)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    // Test that causal masking is correctly applied:
    // Position i should only attend to positions j where j <= i
    constexpr int seq_len = 64;
    constexpr int n_heads = 8;
    constexpr int n_kv_heads = 8;
    constexpr int head_dim = 64;
    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = seq_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    // Use structured data to verify masking behavior
    // Q[i] = i+1 (so position 0 has Q=1, position 1 has Q=2, etc.)
    // K[j] = 1 for all j
    // V[j] = j+1 for all j
    // With causal masking, output[i] should be weighted average of V[0..i]
    std::vector<float> Q_data(q_size);
    std::vector<float> K_data(kv_size, 1.0f);
    std::vector<float> V_data(kv_size);

    for (int pos = 0; pos < seq_len; pos++)
    {
        for (int h = 0; h < n_heads; h++)
        {
            for (int d = 0; d < head_dim; d++)
            {
                Q_data[pos * n_heads * head_dim + h * head_dim + d] = static_cast<float>(pos + 1);
            }
        }
    }
    for (int pos = 0; pos < seq_len; pos++)
    {
        for (int h = 0; h < n_kv_heads; h++)
        {
            for (int d = 0; d < head_dim; d++)
            {
                V_data[pos * n_kv_heads * head_dim + h * head_dim + d] = static_cast<float>(pos + 1);
            }
        }
    }

    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> rocm_output(out_size, 0.0f);

    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true, // causal
        -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, -1);
    ASSERT_TRUE(cpu_success);

    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_output, out_size * sizeof(float));

    hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true, // causal
        -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), out_size));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(rocm_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 CausalMasking", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99);
    EXPECT_LE(l2_error, 0.05);

    // Additional verification: first position should only see V[0]
    // and last position should see weighted average of all V
    float first_pos_val = rocm_output[0];                                 // First element of first position
    float last_pos_val = rocm_output[(seq_len - 1) * n_heads * head_dim]; // First element of last position

    // First position with uniform K should output V[0] = 1.0
    EXPECT_NEAR(first_pos_val, 1.0f, 0.01f) << "First position should only attend to position 0";

    // Last position should have higher value (attending to all positions)
    EXPECT_GT(last_pos_val, first_pos_val) << "Last position should attend to more context";

    LOG_INFO("[FlashAttn2_CausalMasking] PASSED");
}

#endif // HAVE_ROCM

// ============================================================================
// No ROCm Fallback
// ============================================================================

TEST_F(Test__ROCmFlashAttentionParity, NoROCm_SkipsGracefully)
{
    if (!hasROCm())
    {
        SUCCEED() << "ROCm not available - tests skip gracefully";
    }
    else
    {
        SUCCEED() << "ROCm available - tests will run";
    }
}
