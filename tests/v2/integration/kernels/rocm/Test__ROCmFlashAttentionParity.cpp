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
 * - Cross-backend CPU-vs-ROCm attention parity keeps numerical gates because
 *   independent CPU and HIP reductions are not expected to produce identical
 *   FP32 bytes.
 * - MTP verifier grouped-row proofs are stricter: grouped rows must be
 *   byte-for-byte identical to running each row through the same ROCm serial
 *   M=1 decode route. Cosine, relative-L2, and max-absolute error are printed
 *   only to make the first drift easy to diagnose.
 *
 * Target Hardware: AMD MI50 (gfx906 / Vega 20)
 *
 * @author Llaminar Team
 * @date January 2026
 */

#include <gtest/gtest.h>

#include "tensors/Tensors.h"
#include "tensors/GpuTensorView.h"
#include "execution/config/RuntimeConfig.h"
#include "execution/compute_stages/stages/AttentionComputeStage.h"
#include "execution/compute_stages/stages/KVCacheAppendStage.h"
#include "execution/factory/InferenceRunnerFactory.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/coherence/GpuCoherence.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/KernelFactory.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "loaders/ModelContext.h"
#include "transfer/TransferEngine.h"
#include "utils/MPIContext.h"
#include "utils/Logger.h"
#include "../../../utils/ScopedGPUStream.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#include "kernels/rocm/attention/ROCmFlashAttentionKernelT.h"
#include "kernels/cpu/attention/CPUFlashAttentionKernelT.h"
#include "kernels/cpu/CPURingKVCache.h"
#endif

#include <algorithm>
#include <array>
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>
#include <limits>
#include <cstring>

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

    /**
     * @brief Assert that two same-backend FP32 rows have identical bytes.
     *
     * MTP verifier rows are publishable state, so the ROCm grouped verifier
     * path is not allowed to be "close" to serial decode.  This helper keeps
     * the pass/fail rule at raw byte equality while reporting ordinary numeric
     * breadcrumbs for the first differing element.  The diagnostics are useful
     * when a future kernel change alters accumulation order, KV row selection,
     * GQA head mapping, or graph-captured attention params.
     *
     * @param actual Grouped verifier row produced by the production path.
     * @param expected Serial M=1 decode row produced by the same backend.
     * @param count Number of FP32 elements to compare.
     * @param label Human-readable test context included in failures.
     * @return AssertionSuccess when every byte matches, otherwise AssertionFailure.
     */
    ::testing::AssertionResult expectBitwiseEqualFP32Rows(
        const float *actual,
        const float *expected,
        size_t count,
        const std::string &label)
    {
        if (!actual || !expected)
        {
            return ::testing::AssertionFailure()
                   << label << " received a null FP32 row pointer";
        }
        if (std::memcmp(actual, expected, count * sizeof(float)) == 0)
        {
            return ::testing::AssertionSuccess();
        }

        size_t first = 0;
        double max_abs = 0.0;
        size_t max_index = 0;
        for (; first < count; ++first)
        {
            if (std::memcmp(actual + first, expected + first, sizeof(float)) != 0)
                break;
        }
        for (size_t i = 0; i < count; ++i)
        {
            const double diff =
                std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
            if (diff > max_abs)
            {
                max_abs = diff;
                max_index = i;
            }
        }

        uint32_t actual_bits = 0;
        uint32_t expected_bits = 0;
        std::memcpy(&actual_bits, actual + first, sizeof(uint32_t));
        std::memcpy(&expected_bits, expected + first, sizeof(uint32_t));

        return ::testing::AssertionFailure()
               << label << " is not byte-identical to same-backend serial decode"
               << " first_diff_index=" << first
               << " actual=" << actual[first]
               << " expected=" << expected[first]
               << " actual_bits=0x" << std::hex << actual_bits
               << " expected_bits=0x" << expected_bits << std::dec
               << " max_abs_diff=" << max_abs
               << " max_abs_index=" << max_index
               << " cosine=" << cosineSimilarity(actual, expected, count)
               << " rel_l2=" << relativeL2Error(actual, expected, count);
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
#ifdef HAVE_ROCM
    hipStream_t test_stream_ = nullptr;
#endif

    void SetUp() override
    {
#ifdef HAVE_ROCM
        ASSERT_EQ(hipSetDevice(0), hipSuccess) << "Failed to set active ROCm device to 0";
        ASSERT_EQ(
            hipStreamCreateWithFlags(&test_stream_, hipStreamNonBlocking),
            hipSuccess)
            << "Failed to create the explicit ROCm fixture stream";
#endif
    }

    void TearDown() override
    {
#ifdef HAVE_ROCM
        workspace_.reset();
        if (test_stream_)
        {
            EXPECT_EQ(hipStreamSynchronize(test_stream_), hipSuccess);
            EXPECT_EQ(hipStreamDestroy(test_stream_), hipSuccess);
            test_stream_ = nullptr;
        }
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
     * @brief Bind an attention kernel to the exact stream that owns its inputs.
     *
     * Raw-pointer parity tests do not have an executor or compute stage to
     * perform the production stream binding.  The fixture therefore owns one
     * durable nonblocking stream and makes this helper the explicit binding
     * boundary before any direct kernel launch.  Tests with a richer
     * producer/consumer chain may supply that chain's stream instead.
     *
     * @param kernel Attention kernel that will consume the published inputs.
     * @param producer_stream Exact producer stream, or the fixture stream when
     *                        omitted.
     */
    void bindAttentionStream(
        llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> &kernel,
        hipStream_t producer_stream = nullptr)
    {
        hipStream_t exact_stream = producer_stream ? producer_stream : test_stream_;
        ASSERT_NE(exact_stream, nullptr)
            << "ROCm attention tests require an explicit non-null producer stream";
        kernel.setGPUStream(exact_stream);
    }

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
        int batch_size, int n_heads, int head_dim,
        hipStream_t stream = nullptr)
    {
        bindAttentionStream(kernel, stream);
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

    /**
     * @brief Bind cache-owned conversion and grouped-gather workspace.
     *
     * Production graph planning binds this storage before capture. Reference
     * caches in integration tests must honor the same allocation contract so
     * the oracle cannot rely on a dynamic scratch path that production forbids.
     */
    std::unique_ptr<DeviceWorkspaceManager> bindKVCacheWorkspace(
        IKVCache &cache,
        int max_tokens,
        int batch_size,
        int head_dim)
    {
        auto *consumer = dynamic_cast<IWorkspaceConsumer *>(&cache);
        if (!consumer)
        {
            ADD_FAILURE() << "ROCm KV cache does not implement IWorkspaceConsumer";
            return nullptr;
        }

        const WorkspaceRequirements requirements =
            consumer->getWorkspaceRequirements(max_tokens, batch_size, head_dim);
        auto workspace = std::make_unique<DeviceWorkspaceManager>(
            DeviceId::rocm(0),
            requirements.total_bytes_with_alignment() + 4096);
        if (!workspace->allocate(requirements))
        {
            ADD_FAILURE() << "Failed to allocate ROCm KV-cache workspace";
            return nullptr;
        }
        consumer->bindWorkspace(workspace.get());
        return workspace;
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

    void runAttentionStageFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
        MPIContext &mpi_ctx,
        int seq_len);

    void runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
        MPIContext &mpi_ctx,
        int seq_len,
        int history_len = -1,
        int n_heads = 16,
        int n_kv_heads = 2,
        int head_dim = 256,
        const char *label = nullptr,
        ActivationPrecision cache_precision = ActivationPrecision::FP16,
        int head_start = 0,
        int gqa_n_rep = 0);
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
    bindAttentionStream(rocm_kernel);

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_output;
    (void)hipMalloc(&d_Q, q_size * sizeof(float));
    (void)hipMalloc(&d_K, kv_size * sizeof(float));
    (void)hipMalloc(&d_V, kv_size * sizeof(float));
    (void)hipMalloc(&d_output, out_size * sizeof(float));

    // Copy inputs to device
    (void)hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemset(d_output, 0, out_size * sizeof(float));

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
    (void)hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success) << "ROCm attention failed";

    // Copy output back
    (void)hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    // Cleanup
    (void)hipFree(d_Q);
    (void)hipFree(d_K);
    (void)hipFree(d_V);
    (void)hipFree(d_output);

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
    bindAttentionStream(rocm_kernel);

    float *d_Q, *d_K, *d_V, *d_output;
    (void)hipMalloc(&d_Q, q_size * sizeof(float));
    (void)hipMalloc(&d_K, kv_size * sizeof(float));
    (void)hipMalloc(&d_V, kv_size * sizeof(float));
    (void)hipMalloc(&d_output, out_size * sizeof(float));

    (void)hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    (void)hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    (void)hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    (void)hipFree(d_Q);
    (void)hipFree(d_K);
    (void)hipFree(d_V);
    (void)hipFree(d_output);

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
    bindAttentionStream(rocm_kernel);

    float *d_Q, *d_K, *d_V, *d_output;
    (void)hipMalloc(&d_Q, q_size * sizeof(float));
    (void)hipMalloc(&d_K, kv_size * sizeof(float));
    (void)hipMalloc(&d_V, kv_size * sizeof(float));
    (void)hipMalloc(&d_output, out_size * sizeof(float));

    (void)hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    (void)hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    (void)hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    (void)hipFree(d_Q);
    (void)hipFree(d_K);
    (void)hipFree(d_V);
    (void)hipFree(d_output);

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
    bindAttentionStream(rocm_kernel);

    float *d_Q, *d_K, *d_V, *d_output;
    (void)hipMalloc(&d_Q, q_size * sizeof(float));
    (void)hipMalloc(&d_K, kv_size * sizeof(float));
    (void)hipMalloc(&d_V, kv_size * sizeof(float));
    (void)hipMalloc(&d_output, out_size * sizeof(float));

    (void)hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    (void)hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    (void)hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    (void)hipFree(d_Q);
    (void)hipFree(d_K);
    (void)hipFree(d_V);
    (void)hipFree(d_output);

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
    bindAttentionStream(rocm_kernel);

    float *d_Q, *d_K, *d_V, *d_output;
    (void)hipMalloc(&d_Q, q_size * sizeof(float));
    (void)hipMalloc(&d_K, kv_size * sizeof(float));
    (void)hipMalloc(&d_V, kv_size * sizeof(float));
    (void)hipMalloc(&d_output, out_size * sizeof(float));

    (void)hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    (void)hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    (void)hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    (void)hipFree(d_Q);
    (void)hipFree(d_K);
    (void)hipFree(d_V);
    (void)hipFree(d_output);

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

TEST_F(Test__ROCmFlashAttentionParity, DeviceResidentPrefillHonorsExternalMask)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    constexpr int seq_len = 2;
    constexpr int n_heads = 1;
    constexpr int n_kv_heads = 1;
    constexpr int head_dim = 64;
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
    bindAttentionStream(rocm_kernel);

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

    (void)hipFree(d_Q);
    (void)hipFree(d_K);
    (void)hipFree(d_V);
    (void)hipFree(d_output);

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
    (void)hipMalloc(&d_Q, q_size * sizeof(float));
    (void)hipMalloc(&d_K, kv_size * sizeof(float));
    (void)hipMalloc(&d_V, kv_size * sizeof(float));
    (void)hipMalloc(&d_output, out_size * sizeof(float));

    (void)hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemset(d_output, 0, out_size * sizeof(float));

    // Use compute_decode for single-query path
    bool rocm_success = rocm_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        false, // Non-causal
        0);    // position_offset
    (void)hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success) << "ROCm decode attention failed";

    (void)hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    // Cleanup workspace before freeing GPU memory
    cleanupWorkspace(rocm_kernel);

    (void)hipFree(d_Q);
    (void)hipFree(d_K);
    (void)hipFree(d_V);
    (void)hipFree(d_output);

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
    (void)hipMalloc(&d_Q, q_size * sizeof(float));
    (void)hipMalloc(&d_K, kv_size * sizeof(float));
    (void)hipMalloc(&d_V, kv_size * sizeof(float));
    (void)hipMalloc(&d_output, out_size * sizeof(float));

    (void)hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        false, // Non-causal
        0);    // position_offset
    (void)hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    (void)hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    // Cleanup workspace before freeing GPU memory
    cleanupWorkspace(rocm_kernel);

    (void)hipFree(d_Q);
    (void)hipFree(d_K);
    (void)hipFree(d_V);
    (void)hipFree(d_output);

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

/**
 * @brief Prove a captured HIP decode consumes a later device KV-length update.
 *
 * The graph owns a cache-capacity-sized physical split envelope. Replays cross
 * the former split-count and wavefront-count boundaries while device metadata
 * selects the active prefix. A direct same-backend decode must agree byte for
 * byte at every length, proving that mode shifts do not require recapture or
 * alter reduction order.
 */
TEST_F(Test__ROCmFlashAttentionParity, FlashDecode_FP32_GraphReplayUsesUpdatedDeviceKVLenByteExact)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    constexpr int capture_kv_len = 1;
    constexpr int max_kv_len = 16384;
    constexpr std::array<int, 24> replay_kv_lengths{
        1, 2, 3,
        7, 8, 9,
        127, 128,
        255, 256, 257,
        511, 512,
        1023, 1024, 1025,
        2047, 2048,
        4095, 4096,
        8191, 8192, 8193,
        16383};
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
    ASSERT_TRUE(setupWorkspace(rocm_kernel, seq_len, n_heads, head_dim, stream));

    // Warmup establishes allocations. The explicit capacity-bearing update
    // after it defines the immutable graph envelope used during capture.
    ASSERT_TRUE(rocm_kernel.compute_tensor(
        &q_tensor, &k_tensor, &v_tensor, &captured_output,
        1, seq_len, capture_kv_len, n_heads, n_kv_heads, head_dim,
        false, -1, nullptr, nullptr, &mpi_ctx_, 0));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_TRUE(rocm_kernel.prepareDynamicAttnParams(
        capture_kv_len,
        capture_kv_len - 1,
        1,
        stream,
        max_kv_len));

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

    for (const int replay_kv_len : replay_kv_lengths)
    {
        ASSERT_TRUE(rocm_kernel.prepareDynamicAttnParams(
            replay_kv_len,
            replay_kv_len - 1,
            1,
            stream,
            max_kv_len));
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

        EXPECT_TRUE(expectBitwiseEqualFP32Rows(
            captured.data(),
            direct.data(),
            out_size,
            "ROCm captured replay at KV length " +
                std::to_string(replay_kv_len)));
    }

    (void)hipGraphExecDestroy(graph_exec);
    (void)hipGraphDestroy(graph);
    cleanupWorkspace(rocm_kernel);
    (void)hipStreamDestroy(stream);
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
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
    rocm_kernel.setGPUStream(stream);
    ASSERT_TRUE(setupWorkspace(rocm_kernel, seq_len, n_heads, head_dim, stream));

    ASSERT_TRUE(with_gpu_coherence(
        gpu_device,
        {q_tensor.get(), k_tensor.get(), v_tensor.get()},
        {output_tensor.get()},
        static_cast<void *>(stream),
        [&]
        {
            return rocm_kernel.compute_tensor(
                q_tensor.get(), k_tensor.get(), v_tensor.get(), output_tensor.get(),
                1, seq_len, kv_len, n_heads, n_kv_heads, head_dim,
                false, -1, nullptr, nullptr, &mpi_ctx_, 0);
        }));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

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
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
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
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
    rocm_kernel.setGPUStream(stream);
    ASSERT_TRUE(setupWorkspace(rocm_kernel, seq_len, n_heads, head_dim, stream));

    ASSERT_TRUE(with_gpu_coherence(
        gpu_device,
        {q_tensor.get(), k_tensor.get(), v_tensor.get()},
        {output_tensor.get()},
        static_cast<void *>(stream),
        [&]
        {
            return rocm_kernel.compute_tensor(
                q_tensor.get(), k_tensor.get(), v_tensor.get(), output_tensor.get(),
                1, seq_len, kv_len, n_heads, n_kv_heads, head_dim,
                false, -1, nullptr, nullptr, &mpi_ctx_, 0);
        }));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

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
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST_F(Test__ROCmFlashAttentionParity, FlashDecode_NativeFP16KV_MultiRowContinuationMatchesSerialRows)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    /*
     * Regression for Qwen3.6 MTP fixed-depth-3 verification: the verifier runs
     * a short continuation (M=4) against an existing KV history.  Each row must
     * behave like a serial decode row at absolute position history + row; it
     * must not derive that position from a row-local truncated KV length.
     */
    constexpr int seq_len = 4;
    constexpr int history = 17;
    constexpr int kv_len = history + seq_len;
    constexpr int n_heads = 4;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    constexpr int gqa_ratio = n_heads / n_kv_heads;
    const size_t q_size = static_cast<size_t>(seq_len) * n_heads * head_dim;
    const size_t kv_size = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;
    const size_t out_size = q_size;

    auto Q_data = randomFP32Scaled(q_size, 0.25f);
    auto K_data_fp32 = randomFP32Scaled(kv_size, 0.25f);
    auto V_data_fp32 = randomFP32Scaled(kv_size, 0.25f);
    std::vector<uint16_t> K_data_fp16(kv_size);
    std::vector<uint16_t> V_data_fp16(kv_size);
    quantizeToFP16(K_data_fp32.data(), K_data_fp16.data(), kv_size);
    quantizeToFP16(V_data_fp32.data(), V_data_fp16.data(), kv_size);

    std::vector<float> K_ref_fp32(kv_size);
    std::vector<float> V_ref_fp32(kv_size);
    dequantizeFP16(K_data_fp16.data(), K_ref_fp32.data(), kv_size);
    dequantizeFP16(V_data_fp16.data(), V_ref_fp32.data(), kv_size);

    std::vector<float> cpu_output(out_size, 0.0f);
    for (int row = 0; row < seq_len; ++row)
    {
        const int q_abs = history + row;
        for (int head = 0; head < n_heads; ++head)
        {
            const int kv_head = head / gqa_ratio;
            const float *q = Q_data.data() +
                             (static_cast<size_t>(row) * n_heads + head) * head_dim;
            std::vector<float> scores(static_cast<size_t>(q_abs + 1));
            float max_score = -std::numeric_limits<float>::infinity();
            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

            for (int kv_pos = 0; kv_pos <= q_abs; ++kv_pos)
            {
                const float *k = K_ref_fp32.data() +
                                 (static_cast<size_t>(kv_pos) * n_kv_heads + kv_head) * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d)
                    dot += q[d] * k[d];
                const float score = dot * scale;
                scores[static_cast<size_t>(kv_pos)] = score;
                max_score = std::max(max_score, score);
            }

            double sum_exp = 0.0;
            float *out = cpu_output.data() +
                         (static_cast<size_t>(row) * n_heads + head) * head_dim;
            for (int kv_pos = 0; kv_pos <= q_abs; ++kv_pos)
            {
                const float weight =
                    std::exp(scores[static_cast<size_t>(kv_pos)] - max_score);
                const float *v = V_ref_fp32.data() +
                                 (static_cast<size_t>(kv_pos) * n_kv_heads + kv_head) * head_dim;
                sum_exp += static_cast<double>(weight);
                for (int d = 0; d < head_dim; ++d)
                    out[d] += weight * v[d];
            }

            const float inv_sum = static_cast<float>(1.0 / std::max(sum_exp, 1e-30));
            for (int d = 0; d < head_dim; ++d)
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
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
    rocm_kernel.setGPUStream(stream);
    ASSERT_TRUE(setupWorkspace(rocm_kernel, seq_len, n_heads, head_dim, stream));

    ASSERT_TRUE(with_gpu_coherence(
        gpu_device,
        {q_tensor.get(), k_tensor.get(), v_tensor.get()},
        {output_tensor.get()},
        static_cast<void *>(stream),
        [&]
        {
            return rocm_kernel.compute_tensor(
                q_tensor.get(), k_tensor.get(), v_tensor.get(), output_tensor.get(),
                1, seq_len, kv_len, n_heads, n_kv_heads, head_dim,
                true, -1, nullptr, nullptr, &mpi_ctx_, 0);
        }));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    const float *rocm_output = output_tensor->data();
    ASSERT_NE(rocm_output, nullptr);
    ASSERT_FALSE(hasNaNOrInf(rocm_output, out_size));

    const double cosine = cosineSimilarity(rocm_output, cpu_output.data(), out_size);
    const double l2_error = relativeL2Error(rocm_output, cpu_output.data(), out_size);
    const double max_error = maxAbsError(rocm_output, cpu_output.data(), out_size);

    printComparisonStats("FlashDecode native FP16 KV multi-row continuation",
                         cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.999);
    EXPECT_LE(l2_error, 0.02);
    EXPECT_LE(max_error, 0.05);

    cleanupWorkspace(rocm_kernel);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

void runQwen36DeviceDerivedRowsMatchSerialDecode(
    MPIContext &mpi_ctx,
    int seq_len)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    /*
     * Mirror the CUDA Qwen3.6 verifier regression for every production MTP
     * verifier depth.  The all-position verifier derives per-row attention
     * params from the device-side KV count after the append stage, then each
     * row must equal a serial one-token decode at its row-local KV length.
     */
    constexpr int kv_len = 599;
    constexpr int n_heads = 16;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 256;
    const size_t q_cols = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_cols = static_cast<size_t>(n_kv_heads) * head_dim;
    const size_t q_size = static_cast<size_t>(seq_len) * q_cols;
    const size_t kv_size = static_cast<size_t>(kv_len) * kv_cols;

    std::mt19937 rng(static_cast<unsigned>(42 + seq_len));
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    auto random_scaled = [&](size_t count) {
        std::vector<float> data(count);
        for (float &value : data)
            value = dist(rng) * 0.25f;
        return data;
    };

    auto Q_data = random_scaled(q_size);
    auto K_data_fp32 = random_scaled(kv_size);
    auto V_data_fp32 = random_scaled(kv_size);
    std::vector<uint16_t> K_data_fp16(kv_size);
    std::vector<uint16_t> V_data_fp16(kv_size);
    quantizeToFP16(K_data_fp32.data(), K_data_fp16.data(), kv_size);
    quantizeToFP16(V_data_fp32.data(), V_data_fp16.data(), kv_size);

    auto q_grouped_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), q_cols});
    auto k_tensor = std::make_shared<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(kv_len), kv_cols},
        K_data_fp16);
    auto v_tensor = std::make_shared<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(kv_len), kv_cols},
        V_data_fp16);
    auto out_grouped_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), q_cols});
    std::copy(Q_data.begin(), Q_data.end(), q_grouped_tensor->mutable_data());

    const DeviceId gpu_device = DeviceId::rocm(0);
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);

    auto &transfer = TransferEngine::instance();
    ASSERT_TRUE(transfer.uploadFull(q_grouped_tensor.get(), gpu_device, stream).success);
    ASSERT_TRUE(transfer.uploadFull(k_tensor.get(), gpu_device, stream).success);
    ASSERT_TRUE(transfer.uploadFull(v_tensor.get(), gpu_device, stream).success);
    ASSERT_TRUE(transfer.uploadFull(out_grouped_tensor.get(), gpu_device, stream).success);

    int *d_cached_tokens = nullptr;
    ASSERT_EQ(hipMalloc(&d_cached_tokens, sizeof(int)), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(d_cached_tokens, &kv_len, sizeof(int),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);

    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);
    rocm_kernel.setGPUStream(stream);
    auto reqs = rocm_kernel.getWorkspaceRequirements(seq_len, n_heads, head_dim);
    DeviceWorkspaceManager workspace(
        gpu_device,
        std::max(reqs.total_bytes_with_alignment() + 1024 * 1024,
                 static_cast<size_t>(16 * 1024 * 1024)));
    ASSERT_TRUE(workspace.allocate(reqs));
    rocm_kernel.bindWorkspace(&workspace);
    ASSERT_TRUE(rocm_kernel.prepareDynamicAttnParamsFromDeviceSequenceState(
        d_cached_tokens, seq_len, seq_len, stream, kv_len));
    ASSERT_TRUE(rocm_kernel.compute_tensor(
        q_grouped_tensor.get(), k_tensor.get(), v_tensor.get(), out_grouped_tensor.get(),
        1, seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, &mpi_ctx, 0,
        0, n_heads, n_kv_heads, n_heads / n_kv_heads));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    std::vector<float> grouped_output(q_size, 0.0f);
    ASSERT_EQ(hipMemcpyAsync(grouped_output.data(), out_grouped_tensor->gpu_data_ptr(),
                             q_size * sizeof(float), hipMemcpyDeviceToHost, stream),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    for (int row = 0; row < seq_len; ++row)
    {
        auto q_m1_tensor = std::make_shared<FP32Tensor>(
            std::vector<size_t>{size_t{1}, q_cols});
        auto out_m1_tensor = std::make_shared<FP32Tensor>(
            std::vector<size_t>{size_t{1}, q_cols});
        std::memcpy(q_m1_tensor->mutable_data(),
                    Q_data.data() + static_cast<size_t>(row) * q_cols,
                    q_cols * sizeof(float));

        ASSERT_TRUE(transfer.uploadFull(q_m1_tensor.get(), gpu_device, stream).success);
        ASSERT_TRUE(transfer.uploadFull(out_m1_tensor.get(), gpu_device, stream).success);

        const int row_kv_len = std::max(1, kv_len - (seq_len - 1 - row));
        ASSERT_TRUE(rocm_kernel.prepareDynamicAttnParams(
            row_kv_len, row_kv_len - 1, 1, stream));
        ASSERT_TRUE(rocm_kernel.compute_tensor(
            q_m1_tensor.get(), k_tensor.get(), v_tensor.get(), out_m1_tensor.get(),
            1, 1, row_kv_len,
            n_heads, n_kv_heads, head_dim,
            true, -1, nullptr, nullptr, &mpi_ctx, 0,
            0, n_heads, n_kv_heads, n_heads / n_kv_heads))
            << "single-row ROCm decode attention failed for row " << row;
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        std::vector<float> m1_output(q_cols, 0.0f);
        ASSERT_EQ(hipMemcpyAsync(m1_output.data(), out_m1_tensor->gpu_data_ptr(),
                                 q_cols * sizeof(float), hipMemcpyDeviceToHost, stream),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        const float *grouped_row = grouped_output.data() + static_cast<size_t>(row) * q_cols;
        const double cosine = cosineSimilarity(grouped_row, m1_output.data(), q_cols);
        const double l2_error = relativeL2Error(grouped_row, m1_output.data(), q_cols);
        const double max_error = maxAbsError(grouped_row, m1_output.data(), q_cols);
        const std::string label =
            "FlashDecode native FP16 KV Qwen3.6 M" + std::to_string(seq_len) +
            " device-derived vs M1";
        std::cout << "  " << label << ": "
                  << "cosine=" << std::fixed << std::setprecision(6) << cosine
                  << ", L2_error=" << std::scientific << std::setprecision(3) << l2_error
                  << ", max_error=" << max_error
                  << ", count=" << q_cols
                  << std::endl;
        EXPECT_TRUE(expectBitwiseEqualFP32Rows(grouped_row, m1_output.data(), q_cols,
                                               label + " row " + std::to_string(row)));
    }

    rocm_kernel.unbindWorkspace();
    (void)hipFree(d_cached_tokens);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST_F(Test__ROCmFlashAttentionParity, FlashDecode_NativeFP16KV_Qwen36M2DeviceDerivedRowsMatchSerialDecode)
{
    runQwen36DeviceDerivedRowsMatchSerialDecode(mpi_ctx_, 2);
}

TEST_F(Test__ROCmFlashAttentionParity, FlashDecode_NativeFP16KV_Qwen36M3DeviceDerivedRowsMatchSerialDecode)
{
    runQwen36DeviceDerivedRowsMatchSerialDecode(mpi_ctx_, 3);
}

TEST_F(Test__ROCmFlashAttentionParity, FlashDecode_NativeFP16KV_Qwen36M4DeviceDerivedRowsMatchSerialDecode)
{
    runQwen36DeviceDerivedRowsMatchSerialDecode(mpi_ctx_, 4);
}

void Test__ROCmFlashAttentionParity::runAttentionStageFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
    MPIContext &mpi_ctx,
    int seq_len)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    /*
     * Keep the production verifier boundary under test: the stage consumes
     * K/V from the ring cache, asks the cache for the RoPE-on-read FP16 shadow,
     * and derives one dynamic attention param row per verifier row from the
     * cache's device-side token count.  The result must equal serial one-row
     * decode at each row-local KV length.
     */
    constexpr int kv_len = 599;
    constexpr int n_heads = 16;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 256;
    constexpr float rope_theta = 10000000.0f;
    constexpr float partial_rotary_factor = 0.25f;
    const size_t q_cols = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_cols = static_cast<size_t>(n_kv_heads) * head_dim;
    const size_t q_size = static_cast<size_t>(seq_len) * q_cols;
    const size_t kv_size = static_cast<size_t>(kv_len) * kv_cols;

    auto Q_data = randomFP32Scaled(q_size, 0.25f);
    auto K_data_fp32 = randomFP32Scaled(kv_size, 0.25f);
    auto V_data_fp32 = randomFP32Scaled(kv_size, 0.25f);
    std::vector<uint16_t> K_data_fp16(kv_size);
    std::vector<uint16_t> V_data_fp16(kv_size);
    quantizeToFP16(K_data_fp32.data(), K_data_fp16.data(), kv_size);
    quantizeToFP16(V_data_fp32.data(), V_data_fp16.data(), kv_size);

    const DeviceId device = DeviceId::rocm(0);
    auto q_grouped_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), q_cols});
    auto k_tensor = std::make_shared<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(kv_len), kv_cols},
        K_data_fp16);
    auto v_tensor = std::make_shared<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(kv_len), kv_cols},
        V_data_fp16);
    auto out_grouped_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), q_cols});
    std::copy(Q_data.begin(), Q_data.end(), q_grouped_tensor->mutable_data());

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);

    auto &transfer = TransferEngine::instance();
    ASSERT_TRUE(transfer.uploadFull(q_grouped_tensor.get(), device, stream).success);
    ASSERT_TRUE(transfer.uploadFull(k_tensor.get(), device, stream).success);
    ASSERT_TRUE(transfer.uploadFull(v_tensor.get(), device, stream).success);
    ASSERT_TRUE(transfer.uploadFull(out_grouped_tensor.get(), device, stream).success);

    llaminar::v2::kernels::KVCacheConfig config;
    config.precision = ActivationPrecision::FP16;
    config.device = device;
    config.num_layers = 1;
    config.batch_size = 1;
    config.max_seq_len = kv_len + 8;
    config.n_kv_heads = n_kv_heads;
    config.head_dim = head_dim;
    auto kv_cache = llaminar::v2::kernels::KernelFactory::createKVCache(config);
    ASSERT_NE(kv_cache, nullptr);
    std::unique_ptr<DeviceWorkspaceManager> kv_workspace;
    if (auto *cache_workspace_consumer =
            dynamic_cast<IWorkspaceConsumer *>(kv_cache.get()))
    {
        /*
         * Production graph construction binds the cache workspace before any
         * forward or graph capture. The bind publishes immutable per-request
         * K/V pointer tables and reserves the fixed-stride FP16 conversion
         * output consumed by the device-owned grouped attention read. Keeping
         * this explicit prevents the regression from passing through a hidden
         * allocation or a legacy scalar cache shadow.
         */
        const WorkspaceRequirements kv_reqs =
            cache_workspace_consumer->getWorkspaceRequirements(
                kv_len,
                /*n=*/1,
                head_dim);
        kv_workspace = std::make_unique<DeviceWorkspaceManager>(
            device, kv_reqs.total_bytes_with_alignment() + 4096);
        ASSERT_TRUE(kv_workspace->allocate(kv_reqs));
        cache_workspace_consumer->bindWorkspace(kv_workspace.get());
    }
    ASSERT_TRUE(kv_cache->appendWithStream(0, 0, k_tensor.get(), v_tensor.get(), kv_len, stream));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(kv_cache->get_cached_tokens(0, 0), kv_len);
    ASSERT_NE(kv_cache->deviceCachedTokenCountPtr(0, 0), nullptr);

    AttentionComputeStage::Params params;
    params.device_id = device;
    params.Q = q_grouped_tensor.get();
    params.K = k_tensor.get();
    params.V = v_tensor.get();
    params.output = out_grouped_tensor.get();
    params.batch_size = 1;
    params.seq_len = seq_len;
    params.kv_len = kv_len;
    params.n_heads = n_heads;
    params.n_kv_heads = n_kv_heads;
    params.head_dim = head_dim;
    params.causal = true;
    params.auto_detect_mode = true;
    params.kv_cache = kv_cache.get();
    params.layer_idx = 0;
    params.read_kv_from_cache = true;
    params.apply_rope_to_k = true;
    params.rope_theta = rope_theta;
    params.partial_rotary_factor = partial_rotary_factor;
    params.mpi_ctx = &mpi_ctx;

    AttentionComputeStage stage(params);
    stage.setGPUStream(stream);
    const WorkspaceRequirements stage_reqs =
        stage.getWorkspaceRequirements(
            /*m=*/seq_len,
            /*n=*/n_heads,
            /*k=*/head_dim);
    DeviceWorkspaceManager stage_workspace(
        device, stage_reqs.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(stage_workspace.allocate(stage_reqs));
    stage.bindWorkspace(&stage_workspace);

    /*
     * The grouped verifier rows represent the final `seq_len` positions in the
     * already-populated cache. Publish the same pre/post append horizon that the
     * production KV-append stage supplies; the attention stage then derives
     * per-row lengths from canonical device cache state without reading a host
     * mirror.
     */
    stage.updateDynamicParams(
        /*pre_append_cached_tokens=*/kv_len - seq_len,
        /*logical_seq_len=*/seq_len);

    ASSERT_TRUE(stage.execute(nullptr));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    std::vector<float> grouped_output(q_size, 0.0f);
    ASSERT_EQ(hipMemcpyAsync(grouped_output.data(), out_grouped_tensor->gpu_data_ptr(),
                             q_size * sizeof(float), hipMemcpyDeviceToHost, stream),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    IKVCache::KVReadParams read_params;
    read_params.rope_theta = rope_theta;
    read_params.position_start = 0;
    read_params.n_kv_heads = n_kv_heads;
    read_params.head_dim = head_dim;
    read_params.rope_dim = static_cast<int>(partial_rotary_factor * head_dim);
    read_params.gpu_stream = stream;
    ITensor *cache_k = nullptr;
    ITensor *cache_v = nullptr;
    int cache_kv_len = 0;
    ASSERT_TRUE(kv_cache->get_kv_converted(
        0, 0, ActivationPrecision::FP16, &cache_k, &cache_v, &cache_kv_len, &read_params));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(cache_kv_len, kv_len);
    ASSERT_NE(cache_k, nullptr);
    ASSERT_NE(cache_v, nullptr);

    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> serial_kernel(0);
    serial_kernel.setGPUStream(stream);
    ASSERT_TRUE(setupWorkspace(serial_kernel, 1, n_heads, head_dim, stream));

    for (int row = 0; row < seq_len; ++row)
    {
        auto q_m1_tensor = std::make_shared<FP32Tensor>(
            std::vector<size_t>{size_t{1}, q_cols});
        auto out_m1_tensor = std::make_shared<FP32Tensor>(
            std::vector<size_t>{size_t{1}, q_cols});
        std::memcpy(q_m1_tensor->mutable_data(),
                    Q_data.data() + static_cast<size_t>(row) * q_cols,
                    q_cols * sizeof(float));

        ASSERT_TRUE(transfer.uploadFull(q_m1_tensor.get(), device, stream).success);
        ASSERT_TRUE(transfer.uploadFull(out_m1_tensor.get(), device, stream).success);

        const int row_kv_len = std::max(1, kv_len - (seq_len - 1 - row));
        ASSERT_TRUE(serial_kernel.prepareDynamicAttnParams(row_kv_len, row_kv_len - 1, 1, stream));
        ASSERT_TRUE(serial_kernel.compute_tensor(
            q_m1_tensor.get(), cache_k, cache_v, out_m1_tensor.get(),
            1, 1, row_kv_len,
            n_heads, n_kv_heads, head_dim,
            true, -1, nullptr, nullptr, &mpi_ctx, 0,
            0, n_heads, n_kv_heads, n_heads / n_kv_heads));
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        std::vector<float> m1_output(q_cols, 0.0f);
        ASSERT_EQ(hipMemcpyAsync(m1_output.data(), out_m1_tensor->gpu_data_ptr(),
                                 q_cols * sizeof(float), hipMemcpyDeviceToHost, stream),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        const float *grouped_row = grouped_output.data() + static_cast<size_t>(row) * q_cols;
        const std::string label = "AttentionStage FP16 cache Qwen3.6 M" +
                                  std::to_string(seq_len) +
                                  " RoPE-on-read vs M1";
        EXPECT_TRUE(expectBitwiseEqualFP32Rows(grouped_row, m1_output.data(), q_cols,
                                               label + " row " + std::to_string(row)));
    }

    cleanupWorkspace(serial_kernel);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST_F(Test__ROCmFlashAttentionParity, AttentionStage_FP16Cache_Qwen36M2RoPEOnReadRowsMatchSerialDecode)
{
    runAttentionStageFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(mpi_ctx_, 2);
}

TEST_F(Test__ROCmFlashAttentionParity, AttentionStage_FP16Cache_Qwen36M3RoPEOnReadRowsMatchSerialDecode)
{
    runAttentionStageFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(mpi_ctx_, 3);
}

TEST_F(Test__ROCmFlashAttentionParity, AttentionStage_FP16Cache_Qwen36M4RoPEOnReadRowsMatchSerialDecode)
{
    runAttentionStageFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(mpi_ctx_, 4);
}

void Test__ROCmFlashAttentionParity::runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
    MPIContext &mpi_ctx,
    int seq_len,
    int requested_history_len,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    const char *label,
    ActivationPrecision cache_precision,
    int head_start,
    int gqa_n_rep)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    /*
     * Regression for vLLM-style all-position verifier capture: the graph
     * records KV append for the current verifier rows and then consumes the
     * same cache through AttentionComputeStage.  Warming the RoPE-on-read
     * shadow at the history length exercises the incremental cache-shadow path
     * that real decode graphs use before adding the verifier rows.
     */
    const int history_len = requested_history_len > 0 ? requested_history_len : 599 - seq_len;
    const int kv_len = history_len + seq_len;
    const int effective_gqa_n_rep = gqa_n_rep > 0 ? gqa_n_rep : n_heads / n_kv_heads;
    constexpr float rope_theta = 10000000.0f;
    constexpr float partial_rotary_factor = 0.25f;
    const size_t q_cols = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_cols = static_cast<size_t>(n_kv_heads) * head_dim;
    const size_t q_size = static_cast<size_t>(seq_len) * q_cols;
    const size_t kv_size = static_cast<size_t>(kv_len) * kv_cols;
    const size_t history_size = static_cast<size_t>(history_len) * kv_cols;

    auto Q_data = randomFP32Scaled(q_size, 0.25f);
    auto K_data_fp32 = randomFP32Scaled(kv_size, 0.25f);
    auto V_data_fp32 = randomFP32Scaled(kv_size, 0.25f);
    std::vector<uint16_t> K_data_fp16(kv_size);
    std::vector<uint16_t> V_data_fp16(kv_size);
    quantizeToFP16(K_data_fp32.data(), K_data_fp16.data(), kv_size);
    quantizeToFP16(V_data_fp32.data(), V_data_fp16.data(), kv_size);

    const DeviceId device = DeviceId::rocm(0);
    const bool turboquant_cache =
        cache_precision == ActivationPrecision::TQ4 ||
        cache_precision == ActivationPrecision::TQ8;
    auto make_kv_source =
        [&](const std::vector<float> &fp32_source,
            const std::vector<uint16_t> &fp16_source,
            size_t offset,
            size_t count,
            const std::vector<size_t> &shape) -> std::shared_ptr<TensorBase>
    {
        if (turboquant_cache)
        {
            auto tensor = std::make_shared<FP32Tensor>(shape);
            std::copy(
                fp32_source.begin() + static_cast<std::ptrdiff_t>(offset),
                fp32_source.begin() + static_cast<std::ptrdiff_t>(offset + count),
                tensor->mutable_data());
            return tensor;
        }

        return std::make_shared<FP16Tensor>(
            shape,
            std::vector<uint16_t>(
                fp16_source.begin() + static_cast<std::ptrdiff_t>(offset),
                fp16_source.begin() + static_cast<std::ptrdiff_t>(offset + count)));
    };

    auto q_grouped_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), q_cols});
    auto history_k_tensor = make_kv_source(
        K_data_fp32, K_data_fp16, 0, history_size,
        std::vector<size_t>{static_cast<size_t>(history_len), kv_cols});
    auto history_v_tensor = make_kv_source(
        V_data_fp32, V_data_fp16, 0, history_size,
        std::vector<size_t>{static_cast<size_t>(history_len), kv_cols});
    auto current_k_tensor = make_kv_source(
        K_data_fp32, K_data_fp16, history_size,
        static_cast<size_t>(seq_len) * kv_cols,
        std::vector<size_t>{static_cast<size_t>(seq_len), kv_cols});
    auto current_v_tensor = make_kv_source(
        V_data_fp32, V_data_fp16, history_size,
        static_cast<size_t>(seq_len) * kv_cols,
        std::vector<size_t>{static_cast<size_t>(seq_len), kv_cols});
    auto full_k_tensor = make_kv_source(
        K_data_fp32, K_data_fp16, 0, kv_size,
        std::vector<size_t>{static_cast<size_t>(kv_len), kv_cols});
    auto full_v_tensor = make_kv_source(
        V_data_fp32, V_data_fp16, 0, kv_size,
        std::vector<size_t>{static_cast<size_t>(kv_len), kv_cols});
    auto out_grouped_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), q_cols});
    std::copy(Q_data.begin(), Q_data.end(), q_grouped_tensor->mutable_data());

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);

    auto &transfer = TransferEngine::instance();
    ASSERT_TRUE(transfer.uploadFull(q_grouped_tensor.get(), device, stream).success);
    ASSERT_TRUE(transfer.uploadFull(history_k_tensor.get(), device, stream).success);
    ASSERT_TRUE(transfer.uploadFull(history_v_tensor.get(), device, stream).success);
    ASSERT_TRUE(transfer.uploadFull(current_k_tensor.get(), device, stream).success);
    ASSERT_TRUE(transfer.uploadFull(current_v_tensor.get(), device, stream).success);
    ASSERT_TRUE(transfer.uploadFull(full_k_tensor.get(), device, stream).success);
    ASSERT_TRUE(transfer.uploadFull(full_v_tensor.get(), device, stream).success);
    ASSERT_TRUE(transfer.uploadFull(out_grouped_tensor.get(), device, stream).success);

    llaminar::v2::kernels::KVCacheConfig config;
    config.precision = cache_precision;
    config.device = device;
    config.num_layers = 1;
    config.batch_size = 1;
    config.max_seq_len = kv_len + 8;
    config.n_kv_heads = n_kv_heads;
    config.head_dim = head_dim;
    std::unique_ptr<TurboQuantContext> turboquant_context;
    if (turboquant_cache)
    {
        turboquant_context =
            std::make_unique<TurboQuantContext>(head_dim, 42, 42);
        config.turboquant_ctx = turboquant_context.get();
    }
    auto kv_cache = llaminar::v2::kernels::KernelFactory::createKVCache(config);
    ASSERT_NE(kv_cache, nullptr);
    std::unique_ptr<DeviceWorkspaceManager> kv_workspace;
    if (auto *cache_workspace_consumer = dynamic_cast<IWorkspaceConsumer *>(kv_cache.get()))
    {
        /*
         * Q8_1 verifier appends convert the FP16 projection rows into native
         * cache blocks before publishing them.  The conversion scratch must be
         * predeclared and bound before HIP graph capture; allocating it lazily
         * inside the captured verifier append would make the graph invalid.
         */
        const WorkspaceRequirements kv_reqs =
            cache_workspace_consumer->getWorkspaceRequirements(kv_len, 1, head_dim);
        kv_workspace = std::make_unique<DeviceWorkspaceManager>(
            device, kv_reqs.total_bytes_with_alignment() + 4096);
        ASSERT_TRUE(kv_workspace->allocate(kv_reqs));
        cache_workspace_consumer->bindWorkspace(kv_workspace.get());
    }
    ASSERT_TRUE(kv_cache->appendWithStream(0, 0, history_k_tensor.get(), history_v_tensor.get(), history_len, stream));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(kv_cache->get_cached_tokens(0, 0), history_len);

    IKVCache::KVReadParams warm_read_params;
    warm_read_params.rope_theta = rope_theta;
    warm_read_params.position_start = 0;
    warm_read_params.n_kv_heads = n_kv_heads;
    warm_read_params.head_dim = head_dim;
    warm_read_params.rope_dim = static_cast<int>(partial_rotary_factor * head_dim);
    warm_read_params.gpu_stream = stream;
    ITensor *warm_k = nullptr;
    ITensor *warm_v = nullptr;
    int warm_len = 0;
    ASSERT_TRUE(kv_cache->get_kv_converted(
        0, 0, ActivationPrecision::FP16, &warm_k, &warm_v, &warm_len, &warm_read_params));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(warm_len, history_len);

    AttentionComputeStage::Params attn_params;
    attn_params.device_id = device;
    attn_params.Q = q_grouped_tensor.get();
    attn_params.K = current_k_tensor.get();
    attn_params.V = current_v_tensor.get();
    attn_params.output = out_grouped_tensor.get();
    attn_params.batch_size = 1;
    attn_params.seq_len = seq_len;
    attn_params.kv_len = kv_len;
    attn_params.n_heads = n_heads;
    attn_params.n_kv_heads = n_kv_heads;
    attn_params.head_dim = head_dim;
    attn_params.causal = true;
    attn_params.auto_detect_mode = true;
    attn_params.kv_cache = kv_cache.get();
    attn_params.layer_idx = 0;
    attn_params.read_kv_from_cache = true;
    attn_params.apply_rope_to_k = true;
    attn_params.rope_theta = rope_theta;
    attn_params.partial_rotary_factor = partial_rotary_factor;
    attn_params.turboquant_ctx = turboquant_context.get();
    attn_params.mpi_ctx = &mpi_ctx;
    attn_params.head_start = head_start;
    attn_params.gqa_n_rep = effective_gqa_n_rep;

    KVCacheAppendStage::Params append_params;
    append_params.device_id = device;
    append_params.K = current_k_tensor.get();
    append_params.V = current_v_tensor.get();
    append_params.kv_cache = kv_cache.get();
    append_params.layer_idx = 0;
    append_params.seq_idx = 0;
    append_params.num_tokens = seq_len;
    append_params.batch_size = 1;
    append_params.seq_len = seq_len;
    append_params.head_dim = head_dim;
    append_params.turboquant_ctx = turboquant_context.get();

    KVCacheAppendStage append_stage(append_params);
    AttentionComputeStage attn_stage(attn_params);
    append_stage.setGPUStream(stream);
    attn_stage.setGPUStream(stream);

    const WorkspaceRequirements attn_reqs =
        attn_stage.getWorkspaceRequirements(/*m=*/seq_len, /*n=*/n_heads, /*k=*/head_dim);
    DeviceWorkspaceManager attn_workspace(device, attn_reqs.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(attn_workspace.allocate(attn_reqs));
    attn_stage.bindWorkspace(&attn_workspace);

    append_stage.updateDynamicParams(/*pos_offset=*/0, seq_len);
    attn_stage.updateDynamicParams(history_len, seq_len);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    hipGraph_t graph = nullptr;
    hipGraphExec_t graph_exec = nullptr;
    ASSERT_EQ(
        hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal),
        hipSuccess);
    bool capture_ok = false;
    std::string capture_exception;
    {
        GraphCaptureGuard guard;
        try
        {
            capture_ok =
                append_stage.execute(nullptr) &&
                attn_stage.execute(nullptr);
        }
        catch (const std::exception &error)
        {
            capture_exception = error.what();
        }
        catch (...)
        {
            capture_exception = "non-standard exception";
        }
    }
    const hipError_t capture_end_status =
        hipStreamEndCapture(stream, &graph);
    ASSERT_TRUE(capture_exception.empty())
        << "Captured append+attention threw: " << capture_exception;
    ASSERT_EQ(capture_end_status, hipSuccess)
        << "Captured append+attention did not close cleanly";
    ASSERT_TRUE(capture_ok);
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), hipSuccess);
    ASSERT_NE(graph_exec, nullptr);
    ASSERT_EQ(hipGraphLaunch(graph_exec, stream), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(kv_cache->get_cached_tokens(0, 0), kv_len);

    std::vector<float> grouped_output(q_size, 0.0f);
    ASSERT_EQ(hipMemcpyAsync(grouped_output.data(), out_grouped_tensor->gpu_data_ptr(),
                             q_size * sizeof(float), hipMemcpyDeviceToHost, stream),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    auto ref_cache = llaminar::v2::kernels::KernelFactory::createKVCache(config);
    ASSERT_NE(ref_cache, nullptr);
    auto ref_kv_workspace =
        bindKVCacheWorkspace(*ref_cache, kv_len, /*batch_size=*/1, head_dim);
    ASSERT_NE(ref_kv_workspace, nullptr);
    ASSERT_TRUE(ref_cache->appendWithStream(0, 0, full_k_tensor.get(), full_v_tensor.get(), kv_len, stream));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    ITensor *cache_k = nullptr;
    ITensor *cache_v = nullptr;
    int cache_kv_len = 0;
    ASSERT_TRUE(ref_cache->get_kv_converted(
        0, 0, ActivationPrecision::FP16, &cache_k, &cache_v, &cache_kv_len, &warm_read_params));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(cache_kv_len, kv_len);

    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> serial_kernel(0);
    serial_kernel.setGPUStream(stream);
    ASSERT_TRUE(setupWorkspace(serial_kernel, 1, n_heads, head_dim, stream));

    for (int row = 0; row < seq_len; ++row)
    {
        auto q_m1_tensor = std::make_shared<FP32Tensor>(
            std::vector<size_t>{size_t{1}, q_cols});
        auto out_m1_tensor = std::make_shared<FP32Tensor>(
            std::vector<size_t>{size_t{1}, q_cols});
        std::memcpy(q_m1_tensor->mutable_data(),
                    Q_data.data() + static_cast<size_t>(row) * q_cols,
                    q_cols * sizeof(float));
        ASSERT_TRUE(transfer.uploadFull(q_m1_tensor.get(), device, stream).success);
        ASSERT_TRUE(transfer.uploadFull(out_m1_tensor.get(), device, stream).success);

        const int row_kv_len = std::max(1, kv_len - (seq_len - 1 - row));
        ASSERT_TRUE(serial_kernel.prepareDynamicAttnParams(row_kv_len, row_kv_len - 1, 1, stream));
        ASSERT_TRUE(serial_kernel.compute_tensor(
            q_m1_tensor.get(), cache_k, cache_v, out_m1_tensor.get(),
            1, 1, row_kv_len,
            n_heads, n_kv_heads, head_dim,
            true, -1, nullptr, nullptr, &mpi_ctx, 0,
            head_start, n_heads, n_kv_heads, effective_gqa_n_rep));
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        std::vector<float> m1_output(q_cols, 0.0f);
        ASSERT_EQ(hipMemcpyAsync(m1_output.data(), out_m1_tensor->gpu_data_ptr(),
                                 q_cols * sizeof(float), hipMemcpyDeviceToHost, stream),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        const float *grouped_row = grouped_output.data() + static_cast<size_t>(row) * q_cols;
        const std::string generated_label =
            "Captured append+attention FP16 cache Qwen3.6 M" +
            std::to_string(seq_len) +
            " RoPE-on-read vs M1";
        EXPECT_TRUE(expectBitwiseEqualFP32Rows(
            grouped_row, m1_output.data(), q_cols,
            (label ? std::string(label) : generated_label) + " row " + std::to_string(row)));
    }

    cleanupWorkspace(serial_kernel);
    ASSERT_EQ(hipGraphExecDestroy(graph_exec), hipSuccess);
    ASSERT_EQ(hipGraphDestroy(graph), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST_F(Test__ROCmFlashAttentionParity, CapturedAppendThenAttention_FP16Cache_Qwen36M2RoPEOnReadRowsMatchSerialDecode)
{
    runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(mpi_ctx_, 2);
}

TEST_F(Test__ROCmFlashAttentionParity, CapturedAppendThenAttention_FP16Cache_Qwen36M3RoPEOnReadRowsMatchSerialDecode)
{
    runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(mpi_ctx_, 3);
}

TEST_F(Test__ROCmFlashAttentionParity, CapturedAppendThenAttention_FP16Cache_Qwen36M4RoPEOnReadRowsMatchSerialDecode)
{
    runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(mpi_ctx_, 4);
}

TEST_F(Test__ROCmFlashAttentionParity, CapturedAppendThenAttention_FP16Cache_Qwen36LocalTPM2RoPEOnReadRowsMatchSerialDecode)
{
    /*
     * LocalTP halves the Qwen3.6 full-attention head set to eight query heads
     * and one local KV head per device.  The grouped verifier graph still has
     * to publish rows that are identical to serial decode for that local shard.
     */
    runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
        mpi_ctx_,
        /*seq_len=*/2,
        /*history_len=*/597,
        /*n_heads=*/8,
        /*n_kv_heads=*/1,
        /*head_dim=*/256,
        "Captured append+attention FP16 cache Qwen3.6 ROCm LocalTP M2 RoPE-on-read vs M1");
}

TEST_F(Test__ROCmFlashAttentionParity, CapturedAppendThenAttention_FP16Cache_Qwen36LocalTPM3RoPEOnReadRowsMatchSerialDecode)
{
    /*
     * M3 is the first LocalTP verifier group with two speculative draft rows
     * plus the bonus row.  Byte equality here proves the ROCm captured append
     * and attention path does not accidentally use prefill-style visibility
     * across verifier rows.
     */
    runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
        mpi_ctx_,
        /*seq_len=*/3,
        /*history_len=*/596,
        /*n_heads=*/8,
        /*n_kv_heads=*/1,
        /*head_dim=*/256,
        "Captured append+attention FP16 cache Qwen3.6 ROCm LocalTP M3 RoPE-on-read vs M1");
}

TEST_F(Test__ROCmFlashAttentionParity, CapturedAppendThenAttention_FP16Cache_Qwen36LocalTPM4RoPEOnReadRowsMatchSerialDecode)
{
    /*
     * M4 covers the widest grouped verifier row count in the current MTP
     * matrix while preserving the long-context kv_len=599 regime used by the
     * production parity failures.
     */
    runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
        mpi_ctx_,
        /*seq_len=*/4,
        /*history_len=*/595,
        /*n_heads=*/8,
        /*n_kv_heads=*/1,
        /*head_dim=*/256,
        "Captured append+attention FP16 cache Qwen3.6 ROCm LocalTP M4 RoPE-on-read vs M1");
}

/**
 * @brief Prove every supported ROCm grouped-attention depth against serial decode.
 *
 * The grouped launcher is shared by MTP verifier rows and short prefix-cache
 * suffix prefills. Those suffixes can contain any row count through sixteen,
 * so testing only M=2..4 allowed a stale launcher cap to contradict the
 * device-parameter, workspace, and graph contracts. Sweep every legal depth
 * through the real captured append-plus-attention production sequence.
 *
 * Every production cache storage format is covered, including selectable
 * TQ8-K/TQ4-V and TQ8-K/TQ8-V. Quantized and reduced precision caches follow
 * the real converted FP16 RoPE-on-read path, and both LocalTP head offsets
 * verify that global GQA ownership remains correct on either shard. Every
 * grouped FP32 output row must match the corresponding ROCm M=1 decode row
 * byte-for-byte.
 */
TEST_F(
    Test__ROCmFlashAttentionParity,
    CapturedAppendThenAttention_AllNativeKVFormats_Qwen36LocalTPM2To16RoPEOnReadRowsMatchSerialDecode)
{
    constexpr int kFinalKVLength = 39;
    constexpr int kLocalHeads = 8;
    constexpr int kLocalKVHeads = 1;
    constexpr int kHeadDim = 256;
    constexpr int kGlobalGQARep = 16;
    constexpr std::array<ActivationPrecision, 6> kCacheFormats = {
        ActivationPrecision::FP32,
        ActivationPrecision::FP16,
        ActivationPrecision::BF16,
        ActivationPrecision::Q8_1,
        ActivationPrecision::TQ4,
        ActivationPrecision::TQ8,
    };

    for (const ActivationPrecision cache_precision : kCacheFormats)
    {
        for (const int head_start : {0, kLocalHeads})
        {
            for (int verifier_rows = 2;
                 verifier_rows <=
                     attention::kMaxGroupedVerifierAttentionRows;
                 ++verifier_rows)
            {
                SCOPED_TRACE(
                    "cache_precision=" +
                    std::to_string(static_cast<int>(cache_precision)) +
                    " head_start=" + std::to_string(head_start) +
                    " verifier_rows=" + std::to_string(verifier_rows));
                runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
                    mpi_ctx_,
                    verifier_rows,
                    kFinalKVLength - verifier_rows,
                    kLocalHeads,
                    kLocalKVHeads,
                    kHeadDim,
                    "Captured ROCm all-M grouped verifier attention vs M1",
                    cache_precision,
                    head_start,
                    kGlobalGQARep);
                if (::testing::Test::HasFatalFailure())
                    return;
            }
        }
    }
}

/**
 * @brief Cross every sequence-parallel split boundary through the real MTP graph.
 *
 * The exhaustive test above proves M-totality at one short-context geometry.
 * This companion matrix proves that cache format, LocalTP shard ownership, MTP
 * row count, and live sequence regime compose correctly. The selected M values
 * cover the first grouped case, a middle verifier depth, and the maximum
 * supported verifier pass. Final KV lengths bracket each production split
 * transition at 256, 1024 for LocalTP, and 8192.
 *
 * Each cell captures append plus grouped attention once and compares every
 * output row with an independent M=1 invocation on the same ROCm backend.
 * Equality is byte-for-byte; a numerically close result is still a failure.
 */
TEST_F(
    Test__ROCmFlashAttentionParity,
    CapturedAppendThenAttention_AllNativeKVFormats_Qwen36LocalTPSequenceParallelBoundariesMatchSerialByteExact)
{
    constexpr int kLocalHeads = 8;
    constexpr int kLocalKVHeads = 1;
    constexpr int kHeadDim = 256;
    constexpr int kGlobalGQARep = 16;
    constexpr std::array<ActivationPrecision, 6> kCacheFormats = {
        ActivationPrecision::FP32,
        ActivationPrecision::FP16,
        ActivationPrecision::BF16,
        ActivationPrecision::Q8_1,
        ActivationPrecision::TQ4,
        ActivationPrecision::TQ8,
    };
    constexpr std::array<int, 3> kVerifierRows = {2, 8, 16};
    constexpr std::array<int, 9> kFinalKVLengths = {
        255, 256, 257,
        1023, 1024, 1025,
        8191, 8192, 8193,
    };

    for (const ActivationPrecision cache_precision : kCacheFormats)
    {
        for (const int head_start : {0, kLocalHeads})
        {
            for (const int verifier_rows : kVerifierRows)
            {
                for (const int final_kv_len : kFinalKVLengths)
                {
                    ASSERT_GT(final_kv_len, verifier_rows);
                    SCOPED_TRACE(
                        "cache_precision=" +
                        std::to_string(static_cast<int>(cache_precision)) +
                        " head_start=" + std::to_string(head_start) +
                        " verifier_rows=" + std::to_string(verifier_rows) +
                        " final_kv_len=" + std::to_string(final_kv_len));
                    runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
                        mpi_ctx_,
                        verifier_rows,
                        final_kv_len - verifier_rows,
                        kLocalHeads,
                        kLocalKVHeads,
                        kHeadDim,
                        "Captured ROCm sequence-parallel boundary vs M1",
                        cache_precision,
                        head_start,
                        kGlobalGQARep);
                    if (::testing::Test::HasFatalFailure())
                        return;
                }
            }
        }
    }
}

TEST_F(Test__ROCmFlashAttentionParity, CapturedAppendThenAttention_Q81Cache_Qwen36LocalTPM2RoPEOnReadRowsMatchSerialDecode)
{
    /*
     * Production ROCm MTP decode normally stores KV in Q8_1 and consumes an
     * FP16 RoPE-on-read shadow.  This focused regression keeps the same
     * captured append+attention boundary as the model graph, without the full
     * parity harness cost.
     */
    runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
        mpi_ctx_,
        /*seq_len=*/2,
        /*history_len=*/597,
        /*n_heads=*/8,
        /*n_kv_heads=*/1,
        /*head_dim=*/256,
        "Captured append+attention Q8_1 cache Qwen3.6 ROCm LocalTP M2 RoPE-on-read vs M1",
        ActivationPrecision::Q8_1);
}

TEST_F(Test__ROCmFlashAttentionParity, CapturedAppendThenAttention_Q81Cache_Qwen36LocalTPM3RoPEOnReadRowsMatchSerialDecode)
{
    /*
     * Q8_1 M3 proves the converted-cache ROCm LocalTP path remains exactly
     * serial-row equivalent once more than one speculative draft row is present.
     */
    runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
        mpi_ctx_,
        /*seq_len=*/3,
        /*history_len=*/596,
        /*n_heads=*/8,
        /*n_kv_heads=*/1,
        /*head_dim=*/256,
        "Captured append+attention Q8_1 cache Qwen3.6 ROCm LocalTP M3 RoPE-on-read vs M1",
        ActivationPrecision::Q8_1);
}

TEST_F(Test__ROCmFlashAttentionParity, CapturedAppendThenAttention_Q81Cache_Qwen36LocalTPM4RoPEOnReadRowsMatchSerialDecode)
{
    /*
     * Q8_1 M4 is the widest converted-cache grouped verifier attention case in
     * the focused LocalTP sweep.  The byte-exact helper below is the production
     * publication threshold, not an approximate parity hint.
     */
    runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
        mpi_ctx_,
        /*seq_len=*/4,
        /*history_len=*/595,
        /*n_heads=*/8,
        /*n_kv_heads=*/1,
        /*head_dim=*/256,
        "Captured append+attention Q8_1 cache Qwen3.6 ROCm LocalTP M4 RoPE-on-read vs M1",
        ActivationPrecision::Q8_1);
}

TEST_F(Test__ROCmFlashAttentionParity, CapturedAppendThenAttention_Q81Cache_Qwen36LocalTPSmallHistoryM2RoPEOnReadRowsMatchSerialDecode)
{
    /*
     * The device-resident Qwen3.6 MoE publication parity fixture seeds the
     * verifier with a two-token prefix, then appends two MTP rows.  That tiny
     * KV span exercises the single-split ROCm flash-decode regime rather than
     * the long-context split-K regime covered above.  Keep this regression
     * byte-exact so a short-prefix LocalTP grouped verifier cannot drift from
     * serial decode while still looking fine in long-context attention tests.
     */
    runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
        mpi_ctx_,
        /*seq_len=*/2,
        /*history_len=*/2,
        /*n_heads=*/8,
        /*n_kv_heads=*/1,
        /*head_dim=*/256,
        "Captured append+attention Q8_1 cache Qwen3.6 ROCm LocalTP small-history M2 RoPE-on-read vs M1",
        ActivationPrecision::Q8_1);
}

TEST_F(Test__ROCmFlashAttentionParity, CapturedAppendThenAttention_FP16Cache_Qwen36LocalTPSecondShardM2RoPEOnReadRowsMatchSerialDecode)
{
    /*
     * The second LocalTP participant has a non-zero global Q-head offset.  Its
     * local KV head mapping must use the global GQA repetition count, or local
     * row groups can verify against the wrong KV head while still looking
     * plausible in broad cosine gates.
     */
    runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
        mpi_ctx_,
        /*seq_len=*/2,
        /*history_len=*/597,
        /*n_heads=*/8,
        /*n_kv_heads=*/1,
        /*head_dim=*/256,
        "Captured append+attention FP16 cache Qwen3.6 ROCm LocalTP shard1 M2 RoPE-on-read vs M1",
        ActivationPrecision::FP16,
        /*head_start=*/8,
        /*gqa_n_rep=*/16);
}

TEST_F(Test__ROCmFlashAttentionParity, CapturedAppendThenAttention_FP16Cache_Qwen36LocalTPSecondShardM3RoPEOnReadRowsMatchSerialDecode)
{
    /*
     * The second LocalTP shard exercises a non-zero global Q-head offset.  M3
     * makes that global GQA mapping prove itself for every publishable verifier
     * row, not only the M2 smoke shape.
     */
    runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
        mpi_ctx_,
        /*seq_len=*/3,
        /*history_len=*/596,
        /*n_heads=*/8,
        /*n_kv_heads=*/1,
        /*head_dim=*/256,
        "Captured append+attention FP16 cache Qwen3.6 ROCm LocalTP shard1 M3 RoPE-on-read vs M1",
        ActivationPrecision::FP16,
        /*head_start=*/8,
        /*gqa_n_rep=*/16);
}

TEST_F(Test__ROCmFlashAttentionParity, CapturedAppendThenAttention_FP16Cache_Qwen36LocalTPSecondShardM4RoPEOnReadRowsMatchSerialDecode)
{
    /*
     * Same second-shard global-GQA proof at the maximum verifier group size
     * used by the current long-context MTP matrix.
     */
    runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
        mpi_ctx_,
        /*seq_len=*/4,
        /*history_len=*/595,
        /*n_heads=*/8,
        /*n_kv_heads=*/1,
        /*head_dim=*/256,
        "Captured append+attention FP16 cache Qwen3.6 ROCm LocalTP shard1 M4 RoPE-on-read vs M1",
        ActivationPrecision::FP16,
        /*head_start=*/8,
        /*gqa_n_rep=*/16);
}

TEST_F(Test__ROCmFlashAttentionParity, CapturedAppendThenAttention_Q81Cache_Qwen36LocalTPSecondShardM2RoPEOnReadRowsMatchSerialDecode)
{
    /*
     * Same second-shard proof through Q8_1 cache storage.  This is the narrow
     * ROCm regression for grouped verifier rows that combine LocalTP,
     * graph-captured append, converted RoPE-on-read KV, and global GQA mapping.
     */
    runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
        mpi_ctx_,
        /*seq_len=*/2,
        /*history_len=*/597,
        /*n_heads=*/8,
        /*n_kv_heads=*/1,
        /*head_dim=*/256,
        "Captured append+attention Q8_1 cache Qwen3.6 ROCm LocalTP shard1 M2 RoPE-on-read vs M1",
        ActivationPrecision::Q8_1,
        /*head_start=*/8,
        /*gqa_n_rep=*/16);
}

TEST_F(Test__ROCmFlashAttentionParity, CapturedAppendThenAttention_Q81Cache_Qwen36LocalTPSecondShardM3RoPEOnReadRowsMatchSerialDecode)
{
    /*
     * Q8_1 second-shard M3 combines converted KV storage with global GQA head
     * offsets.  Keeping this byte-exact prevents LocalTP cache conversion from
     * hiding behind broad cosine gates.
     */
    runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
        mpi_ctx_,
        /*seq_len=*/3,
        /*history_len=*/596,
        /*n_heads=*/8,
        /*n_kv_heads=*/1,
        /*head_dim=*/256,
        "Captured append+attention Q8_1 cache Qwen3.6 ROCm LocalTP shard1 M3 RoPE-on-read vs M1",
        ActivationPrecision::Q8_1,
        /*head_start=*/8,
        /*gqa_n_rep=*/16);
}

TEST_F(Test__ROCmFlashAttentionParity, CapturedAppendThenAttention_Q81Cache_Qwen36LocalTPSecondShardM4RoPEOnReadRowsMatchSerialDecode)
{
    /*
     * Q8_1 second-shard M4 is the widest focused ROCm LocalTP attention
     * verifier case: converted cache, non-zero head offset, and four grouped
     * rows all have to line up with serial decode exactly.
     */
    runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
        mpi_ctx_,
        /*seq_len=*/4,
        /*history_len=*/595,
        /*n_heads=*/8,
        /*n_kv_heads=*/1,
        /*head_dim=*/256,
        "Captured append+attention Q8_1 cache Qwen3.6 ROCm LocalTP shard1 M4 RoPE-on-read vs M1",
        ActivationPrecision::Q8_1,
        /*head_start=*/8,
        /*gqa_n_rep=*/16);
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
    (void)hipMalloc(&d_Q, q_size * sizeof(float));
    (void)hipMalloc(&d_K, kv_size * sizeof(float));
    (void)hipMalloc(&d_V, kv_size * sizeof(float));
    (void)hipMalloc(&d_out, out_size * sizeof(float));

    (void)hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_K, K_from_q81, kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_V, V_from_q81, kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemset(d_out, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute_decode(
        d_Q, d_K, d_V, d_out,
        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        false,
        0);
    (void)hipDeviceSynchronize();
    ASSERT_TRUE(rocm_success);

    (void)hipMemcpy(rocm_q81_output.data(), d_out, out_size * sizeof(float), hipMemcpyDeviceToHost);

    cleanupWorkspace(rocm_kernel);
    (void)hipFree(d_Q);
    (void)hipFree(d_K);
    (void)hipFree(d_V);
    (void)hipFree(d_out);

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
    (void)hipMalloc(&d_Q, q_size * sizeof(float));
    (void)hipMalloc(&d_K, kv_size * sizeof(float));
    (void)hipMalloc(&d_V, kv_size * sizeof(float));
    (void)hipMalloc(&d_out, out_size * sizeof(float));

    (void)hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_K, K_from_q81, kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_V, V_from_q81, kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemset(d_out, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute_decode(
        d_Q, d_K, d_V, d_out,
        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        false,
        0);
    (void)hipDeviceSynchronize();
    ASSERT_TRUE(rocm_success);

    (void)hipMemcpy(rocm_q81_output.data(), d_out, out_size * sizeof(float), hipMemcpyDeviceToHost);

    cleanupWorkspace(rocm_kernel);
    (void)hipFree(d_Q);
    (void)hipFree(d_K);
    (void)hipFree(d_V);
    (void)hipFree(d_out);

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
    bindAttentionStream(rocm_kernel);

    float *d_Q, *d_K, *d_V, *d_output;
    (void)hipMalloc(&d_Q, q_size * sizeof(float));
    (void)hipMalloc(&d_K, kv_size * sizeof(float));
    (void)hipMalloc(&d_V, kv_size * sizeof(float));
    (void)hipMalloc(&d_output, out_size * sizeof(float));

    (void)hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        false, // NON-causal
        -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    (void)hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    (void)hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    (void)hipFree(d_Q);
    (void)hipFree(d_K);
    (void)hipFree(d_V);
    (void)hipFree(d_output);

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
    bindAttentionStream(rocm_kernel);

    float *d_Q, *d_K, *d_V, *d_output;
    (void)hipMalloc(&d_Q, q_size * sizeof(float));
    (void)hipMalloc(&d_K, kv_size * sizeof(float));
    (void)hipMalloc(&d_V, kv_size * sizeof(float));
    (void)hipMalloc(&d_output, out_size * sizeof(float));

    (void)hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    (void)hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    (void)hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    (void)hipFree(d_Q);
    (void)hipFree(d_K);
    (void)hipFree(d_V);
    (void)hipFree(d_output);

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
    bindAttentionStream(rocm_kernel);

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

    (void)hipFree(d_Q);
    (void)hipFree(d_K);
    (void)hipFree(d_V);
    (void)hipFree(d_output);

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
    llaminar2::test::ScopedGPUStream stream(gpu_device);
    rocm_kernel.setGPUStream(stream.get());

    ASSERT_TRUE(with_gpu_coherence(
        gpu_device,
        {q_tensor.get(), k_tensor.get(), v_tensor.get()},
        {output_tensor.get()},
        stream.get(),
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

/**
 * @brief Prove request-batched ROCm prefill is byte-identical to isolated requests.
 *
 * The device-resident prefill implementation consumes request-major Q/K/V
 * spans with a fixed physical stride. A one-request test cannot detect an
 * incorrect request base: request zero remains valid while request one can
 * read inactive capacity or another cache bank.
 *
 * This regression uses the exact dense Qwen 3.6 attention geometry that exposed
 * the defect and sweeps every production prefill storage route.  Each request
 * contains different values and is compared against the same ROCm backend run
 * with batch size one.  Raw FP32 byte equality is required because changing the
 * request batch must not change the per-request forward pass.
 */
TEST_F(
    Test__ROCmFlashAttentionParity,
    FlashAttn2_RequestBatchedPrefillAllKVFormatsMatchesIsolatedRequestsByteExact)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    constexpr int batch_size = 2;
    constexpr int seq_len = 16;
    constexpr int n_heads = 24;
    constexpr int n_kv_heads = 4;
    constexpr int head_dim = 256;
    constexpr int q_width = n_heads * head_dim;
    constexpr int kv_width = n_kv_heads * head_dim;
    constexpr int total_rows = batch_size * seq_len;
    constexpr std::array<int, batch_size> request_lengths{seq_len, 11};
    const size_t q_elements = static_cast<size_t>(total_rows) * q_width;
    const size_t kv_elements = static_cast<size_t>(total_rows) * kv_width;
    const size_t request_output_elements =
        static_cast<size_t>(seq_len) * q_width;

    std::vector<float> q_data = randomFP32Scaled(q_elements, 0.25f);
    std::vector<float> k_data = randomFP32Scaled(kv_elements, 0.25f);
    std::vector<float> v_data = randomFP32Scaled(kv_elements, 0.25f);
    for (int row = request_lengths[1]; row < seq_len; ++row)
    {
        const size_t q_row =
            static_cast<size_t>(seq_len + row) * q_width;
        const size_t kv_row =
            static_cast<size_t>(seq_len + row) * kv_width;
        std::fill_n(
            q_data.begin() + static_cast<std::ptrdiff_t>(q_row),
            q_width,
            700.0f + static_cast<float>(row));
        std::fill_n(
            k_data.begin() + static_cast<std::ptrdiff_t>(kv_row),
            kv_width,
            -800.0f - static_cast<float>(row));
        std::fill_n(
            v_data.begin() + static_cast<std::ptrdiff_t>(kv_row),
            kv_width,
            900.0f + static_cast<float>(row));
    }
    const DeviceId gpu_device = DeviceId::rocm(0);
    llaminar2::test::ScopedGPUStream stream(gpu_device);

    auto makeFP32 = [](const std::vector<float> &values, int rows, int width)
        -> std::shared_ptr<TensorBase>
    {
        auto tensor = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows),
                                static_cast<size_t>(width)});
        std::copy(values.begin(), values.end(), tensor->mutable_data());
        return tensor;
    };

    auto makeKV = [](
                      const std::vector<float> &values,
                      int rows,
                      int width,
                      ActivationPrecision precision)
        -> std::shared_ptr<TensorBase>
    {
        const std::vector<size_t> shape{
            static_cast<size_t>(rows), static_cast<size_t>(width)};
        switch (precision)
        {
        case ActivationPrecision::FP32:
        {
            auto tensor = std::make_shared<FP32Tensor>(shape);
            std::copy(values.begin(), values.end(), tensor->mutable_data());
            return tensor;
        }
        case ActivationPrecision::FP16:
        {
            std::vector<uint16_t> fp16(values.size());
            quantizeToFP16(values.data(), fp16.data(), values.size());
            return std::make_shared<FP16Tensor>(shape, std::move(fp16));
        }
        case ActivationPrecision::Q8_1:
            return Q8_1Tensor::quantize_from_fp32(values.data(), shape);
        default:
            return nullptr;
        }
    };

    const std::vector<ActivationPrecision> precisions{
        ActivationPrecision::FP32,
        ActivationPrecision::FP16,
        ActivationPrecision::Q8_1,
    };
    for (const ActivationPrecision precision : precisions)
    {
        const char *format_name =
            precision == ActivationPrecision::FP32
                ? "FP32"
                : (precision == ActivationPrecision::FP16 ? "FP16" : "Q8_1");
        SCOPED_TRACE(format_name);

        auto q_batch = makeFP32(q_data, total_rows, q_width);
        auto k_batch = makeKV(k_data, total_rows, kv_width, precision);
        auto v_batch = makeKV(v_data, total_rows, kv_width, precision);
        auto output_batch = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(total_rows),
                                static_cast<size_t>(q_width)});
        ASSERT_NE(k_batch, nullptr);
        ASSERT_NE(v_batch, nullptr);

        llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32>
            batch_kernel(0);
        batch_kernel.setGPUStream(stream.get());
        ASSERT_TRUE(with_gpu_coherence(
            gpu_device,
            {q_batch.get(), k_batch.get(), v_batch.get()},
            {output_batch.get()},
            stream.get(),
            [&]
            {
                return batch_kernel.compute_tensor(
                    q_batch.get(), k_batch.get(), v_batch.get(),
                    output_batch.get(),
                    batch_size, seq_len, seq_len,
                    n_heads, n_kv_heads, head_dim,
                    /*causal=*/true, /*window_size=*/-1,
                    /*workspace_scores=*/nullptr,
                    /*workspace_mask=*/nullptr,
                    &mpi_ctx_, /*device_idx=*/0);
            }));

        for (int request = 0; request < batch_size; ++request)
        {
            const int request_seq_len = request_lengths[request];
            const size_t q_begin =
                static_cast<size_t>(request) * request_output_elements;
            const size_t kv_begin =
                static_cast<size_t>(request) *
                static_cast<size_t>(seq_len) * kv_width;
            const std::vector<float> q_request(
                q_data.begin() + static_cast<std::ptrdiff_t>(q_begin),
                q_data.begin() + static_cast<std::ptrdiff_t>(
                                     q_begin +
                                     static_cast<size_t>(request_seq_len) *
                                         q_width));
            const std::vector<float> k_request(
                k_data.begin() + static_cast<std::ptrdiff_t>(kv_begin),
                k_data.begin() + static_cast<std::ptrdiff_t>(
                                     kv_begin +
                                     static_cast<size_t>(request_seq_len) *
                                         kv_width));
            const std::vector<float> v_request(
                v_data.begin() + static_cast<std::ptrdiff_t>(kv_begin),
                v_data.begin() + static_cast<std::ptrdiff_t>(
                                     kv_begin +
                                     static_cast<size_t>(request_seq_len) *
                                         kv_width));

            auto q_single = makeFP32(q_request, request_seq_len, q_width);
            auto k_single = makeKV(
                k_request, request_seq_len, kv_width, precision);
            auto v_single = makeKV(
                v_request, request_seq_len, kv_width, precision);
            auto output_single = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(request_seq_len),
                                    static_cast<size_t>(q_width)});
            ASSERT_NE(k_single, nullptr);
            ASSERT_NE(v_single, nullptr);

            llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32>
                single_kernel(0);
            single_kernel.setGPUStream(stream.get());
            ASSERT_TRUE(with_gpu_coherence(
                gpu_device,
                {q_single.get(), k_single.get(), v_single.get()},
                {output_single.get()},
                stream.get(),
                [&]
                {
                    return single_kernel.compute_tensor(
                        q_single.get(), k_single.get(), v_single.get(),
                        output_single.get(),
                        /*batch_size=*/1,
                        request_seq_len,
                        request_seq_len,
                        n_heads, n_kv_heads, head_dim,
                        /*causal=*/true, /*window_size=*/-1,
                        /*workspace_scores=*/nullptr,
                        /*workspace_mask=*/nullptr,
                        &mpi_ctx_, /*device_idx=*/0);
                }));

            ASSERT_TRUE(expectBitwiseEqualFP32Rows(
                output_batch->data() + q_begin,
                output_single->data(),
                static_cast<size_t>(request_seq_len) * q_width,
                std::string("ROCm request-batched prefill ") + format_name +
                    " request=" + std::to_string(request)));
        }
    }
}

/**
 * @brief Prove captured request-batch cache publication reads every ROCm ring.
 *
 * A direct batched-attention test cannot catch cache-topology defects because
 * it starts with an already-contiguous K/V tensor. Production prefill first
 * appends each request into an independent ring allocation, then asks the cache
 * to materialize a compact request-major view for attention. The original
 * broken implementation either published metadata only for request zero or
 * treated request zero's allocation as if all request rings followed it in
 * memory. In a two-cache worker, storing immutable entry pointers in shared
 * named workspace could also let the MTP cache overwrite the main cache's
 * topology before graph capture.
 *
 * This regression captures the actual `KVCacheAppendStage -> resident cache
 * gather -> AttentionComputeStage` chain. Request-specific K/V offsets make
 * sequence-zero substitution immediately visible. The graph is replayed
 * twice without recapture, growing every request beyond the live length seen
 * during capture. Every ordinary ROCm ring storage type is exercised, and
 * each output row must be byte-for-byte identical to a real batch-one M=1
 * decode over the same native cache prefix after both launches.  This is the
 * actual grouped-verifier contract; an isolated multirow prefill is not a
 * serial-decode oracle because it is free to use another reduction order.
 */
TEST_F(
    Test__ROCmFlashAttentionParity,
    CapturedGrowingRequestBatchAllStandardKVFormatsMatchesSerialDecodeRows)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    constexpr int batch_size = 2;
    constexpr int seq_len = 8;
    constexpr int n_heads = 4;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    const size_t q_cols = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_cols = static_cast<size_t>(n_kv_heads) * head_dim;
    const size_t request_q_elements = static_cast<size_t>(seq_len) * q_cols;
    const size_t request_kv_elements = static_cast<size_t>(seq_len) * kv_cols;
    const size_t q_elements = static_cast<size_t>(batch_size) * request_q_elements;
    const size_t kv_elements = static_cast<size_t>(batch_size) * request_kv_elements;
    const DeviceId device = DeviceId::rocm(0);

    std::vector<float> q_host = randomFP32Scaled(q_elements, 0.25f);
    std::vector<float> k_host = randomFP32Scaled(kv_elements, 0.25f);
    std::vector<float> v_host = randomFP32Scaled(kv_elements, 0.25f);
    for (int request = 0; request < batch_size; ++request)
    {
        const size_t begin = static_cast<size_t>(request) * request_kv_elements;
        for (size_t index = 0; index < request_kv_elements; ++index)
        {
            // Deliberately separate the two requests even if the seeded random
            // vectors happen to contain locally similar values.
            k_host[begin + index] += 0.25f * static_cast<float>(request);
            v_host[begin + index] -= 0.35f * static_cast<float>(request);
        }
    }

    auto makeFP32 = [](const std::vector<float> &values, size_t rows, size_t cols)
        -> std::shared_ptr<TensorBase>
    {
        auto tensor = std::make_shared<FP32Tensor>(
            std::vector<size_t>{rows, cols});
        std::copy(values.begin(), values.end(), tensor->mutable_data());
        return tensor;
    };
    auto makeNativeKV = [](
                            const std::vector<float> &values,
                            size_t rows,
                            size_t cols,
                            ActivationPrecision precision)
        -> std::shared_ptr<TensorBase>
    {
        const std::vector<size_t> shape{rows, cols};
        switch (precision)
        {
        case ActivationPrecision::FP32:
        {
            auto tensor = std::make_shared<FP32Tensor>(shape);
            std::copy(values.begin(), values.end(), tensor->mutable_data());
            return tensor;
        }
        case ActivationPrecision::FP16:
        {
            std::vector<uint16_t> fp16(values.size());
            quantizeToFP16(values.data(), fp16.data(), values.size());
            return std::make_shared<FP16Tensor>(shape, fp16);
        }
        case ActivationPrecision::BF16:
        {
            auto tensor = std::make_shared<BF16Tensor>(shape);
            tensor->from_fp32(values.data(), values.size());
            return tensor;
        }
        case ActivationPrecision::Q8_1:
            return Q8_1Tensor::quantize_from_fp32(values.data(), shape);
        default:
            return nullptr;
        }
    };

    struct FormatCase
    {
        ActivationPrecision precision;
        const char *name;
    };
    constexpr std::array<FormatCase, 4> formats{{
        {ActivationPrecision::FP32, "FP32"},
        {ActivationPrecision::FP16, "FP16"},
        {ActivationPrecision::BF16, "BF16"},
        {ActivationPrecision::Q8_1, "Q8_1"},
    }};

    auto &transfer = TransferEngine::instance();
    for (const FormatCase &format : formats)
    {
        SCOPED_TRACE(format.name);
        auto q_tensor = makeFP32(q_host, batch_size * seq_len, q_cols);
        auto k_tensor = makeNativeKV(
            k_host, batch_size * seq_len, kv_cols, format.precision);
        auto v_tensor = makeNativeKV(
            v_host, batch_size * seq_len, kv_cols, format.precision);
        auto batch_output = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(batch_size * seq_len), q_cols});
        ASSERT_NE(k_tensor, nullptr);
        ASSERT_NE(v_tensor, nullptr);

        hipStream_t stream = nullptr;
        ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
        ASSERT_TRUE(transfer.uploadFull(q_tensor.get(), device, stream).success);
        ASSERT_TRUE(transfer.uploadFull(k_tensor.get(), device, stream).success);
        ASSERT_TRUE(transfer.uploadFull(v_tensor.get(), device, stream).success);
        ASSERT_TRUE(transfer.uploadFull(batch_output.get(), device, stream).success);

        llaminar::v2::kernels::KVCacheConfig cache_config;
        cache_config.precision = format.precision;
        cache_config.device = device;
        cache_config.num_layers = 1;
        cache_config.batch_size = batch_size;
        cache_config.max_seq_len = seq_len + 8;
        cache_config.n_kv_heads = n_kv_heads;
        cache_config.head_dim = head_dim;
        auto kv_cache = llaminar::v2::kernels::KernelFactory::createKVCache(cache_config);
        ASSERT_NE(kv_cache, nullptr);

        auto *kv_workspace_consumer =
            dynamic_cast<IWorkspaceConsumer *>(kv_cache.get());
        ASSERT_NE(kv_workspace_consumer, nullptr);
        const WorkspaceRequirements kv_requirements =
            kv_workspace_consumer->getWorkspaceRequirements(
                /*m=*/seq_len,
                /*n=*/batch_size,
                /*k=*/head_dim);
        DeviceWorkspaceManager kv_workspace(
            device, kv_requirements.total_bytes_with_alignment() + 4096);
        ASSERT_TRUE(kv_workspace.allocate(kv_requirements));
        kv_workspace_consumer->bindWorkspace(&kv_workspace);

        KVCacheAppendStage append_stage({
            .device_id = device,
            .K = k_tensor.get(),
            .V = v_tensor.get(),
            .kv_cache = kv_cache.get(),
            .layer_idx = 0,
            .seq_idx = 0,
            .num_tokens = batch_size * seq_len,
            .batch_size = batch_size,
            .seq_len = seq_len,
            .head_dim = head_dim,
        });
        AttentionComputeStage attention_stage({
            .device_id = device,
            .mpi_ctx = &mpi_ctx_,
            .Q = q_tensor.get(),
            .K = k_tensor.get(),
            .V = v_tensor.get(),
            .output = batch_output.get(),
            .batch_size = batch_size,
            .seq_len = seq_len,
            .kv_len = seq_len,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .causal = true,
            .auto_detect_mode = true,
            .kv_cache = kv_cache.get(),
            .layer_idx = 0,
            .read_kv_from_cache = true,
        });
        append_stage.setGPUStream(stream);
        attention_stage.setGPUStream(stream);

        const WorkspaceRequirements attention_requirements =
            attention_stage.getWorkspaceRequirements(
                /*m=*/batch_size * seq_len,
                /*n=*/n_heads,
                /*k=*/head_dim);
        DeviceWorkspaceManager attention_workspace(
            device, attention_requirements.total_bytes_with_alignment() + 4096);
        ASSERT_TRUE(attention_workspace.allocate(attention_requirements));
        attention_stage.bindWorkspace(&attention_workspace);

        append_stage.updateDynamicParams(/*pos_offset=*/0, seq_len);
        attention_stage.updateDynamicParams(/*pos_offset=*/0, seq_len);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        hipGraph_t graph = nullptr;
        hipGraphExec_t graph_exec = nullptr;
        ASSERT_EQ(
            hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal),
            hipSuccess);
        bool capture_ok = false;
        {
            GraphCaptureGuard guard;
            capture_ok =
                append_stage.execute(nullptr) && attention_stage.execute(nullptr);
        }
        ASSERT_EQ(hipStreamEndCapture(stream, &graph), hipSuccess);
        ASSERT_TRUE(capture_ok);
        ASSERT_NE(graph, nullptr);
        ASSERT_EQ(
            hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
            hipSuccess);
        ASSERT_NE(graph_exec, nullptr);
        ASSERT_EQ(hipGraphLaunch(graph_exec, stream), hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        EXPECT_EQ(kv_cache->get_cached_tokens(0, 0), seq_len);
        EXPECT_EQ(kv_cache->get_cached_tokens(0, 1), seq_len);

        std::vector<float> actual(q_elements, 0.0f);
        ASSERT_EQ(
            hipMemcpyAsync(
                actual.data(), batch_output->gpu_data_ptr(),
                q_elements * sizeof(float), hipMemcpyDeviceToHost, stream),
            hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32>
            serial_kernel(0);
        serial_kernel.setGPUStream(stream);
        const WorkspaceRequirements serial_requirements =
            serial_kernel.getWorkspaceRequirements(
                /*m=*/1, /*n=*/n_heads, /*k=*/head_dim);
        DeviceWorkspaceManager serial_workspace(
            device, serial_requirements.total_bytes_with_alignment() + 4096);
        ASSERT_TRUE(serial_workspace.allocate(serial_requirements));
        serial_kernel.bindWorkspace(&serial_workspace);

        auto expectSerialDecodeRows =
            [&](const std::vector<float> &grouped_output,
                int completed_blocks,
                const std::string &phase)
        {
            for (int request = 0; request < batch_size; ++request)
            {
                const size_t q_request_begin =
                    static_cast<size_t>(request) * request_q_elements;
                const size_t kv_request_begin =
                    static_cast<size_t>(request) * request_kv_elements;
                const auto k_block_begin =
                    k_host.begin() +
                    static_cast<std::ptrdiff_t>(kv_request_begin);
                const auto k_block_end =
                    k_block_begin +
                    static_cast<std::ptrdiff_t>(request_kv_elements);
                const auto v_block_begin =
                    v_host.begin() +
                    static_cast<std::ptrdiff_t>(kv_request_begin);
                const auto v_block_end =
                    v_block_begin +
                    static_cast<std::ptrdiff_t>(request_kv_elements);

                for (int row = 0; row < seq_len; ++row)
                {
                    const size_t q_row_begin =
                        q_request_begin + static_cast<size_t>(row) * q_cols;
                    const std::vector<float> q_row(
                        q_host.begin() +
                            static_cast<std::ptrdiff_t>(q_row_begin),
                        q_host.begin() +
                            static_cast<std::ptrdiff_t>(q_row_begin + q_cols));

                    std::vector<float> k_prefix;
                    std::vector<float> v_prefix;
                    const int visible_rows =
                        completed_blocks * seq_len + row + 1;
                    k_prefix.reserve(
                        static_cast<size_t>(visible_rows) * kv_cols);
                    v_prefix.reserve(
                        static_cast<size_t>(visible_rows) * kv_cols);
                    for (int block = 0; block < completed_blocks; ++block)
                    {
                        k_prefix.insert(
                            k_prefix.end(), k_block_begin, k_block_end);
                        v_prefix.insert(
                            v_prefix.end(), v_block_begin, v_block_end);
                    }
                    k_prefix.insert(
                        k_prefix.end(),
                        k_block_begin,
                        k_block_begin +
                            static_cast<std::ptrdiff_t>(
                                static_cast<size_t>(row + 1) * kv_cols));
                    v_prefix.insert(
                        v_prefix.end(),
                        v_block_begin,
                        v_block_begin +
                            static_cast<std::ptrdiff_t>(
                                static_cast<size_t>(row + 1) * kv_cols));

                    auto q_single = makeFP32(q_row, /*rows=*/1, q_cols);
                    auto k_single = makeNativeKV(
                        k_prefix,
                        static_cast<size_t>(visible_rows),
                        kv_cols,
                        format.precision);
                    auto v_single = makeNativeKV(
                        v_prefix,
                        static_cast<size_t>(visible_rows),
                        kv_cols,
                        format.precision);
                    auto output_single = std::make_shared<FP32Tensor>(
                        std::vector<size_t>{1, q_cols});
                    ASSERT_NE(k_single, nullptr);
                    ASSERT_NE(v_single, nullptr);
                    ASSERT_TRUE(
                        transfer.uploadFull(
                            q_single.get(), device, stream)
                            .success);
                    ASSERT_TRUE(
                        transfer.uploadFull(
                            k_single.get(), device, stream)
                            .success);
                    ASSERT_TRUE(
                        transfer.uploadFull(
                            v_single.get(), device, stream)
                            .success);
                    ASSERT_TRUE(
                        transfer.uploadFull(
                            output_single.get(), device, stream)
                            .success);

                    ASSERT_TRUE(serial_kernel.compute_tensor(
                        q_single.get(),
                        k_single.get(),
                        v_single.get(),
                        output_single.get(),
                        /*batch_size=*/1,
                        /*seq_len=*/1,
                        visible_rows,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        /*causal=*/true,
                        /*window_size=*/-1,
                        /*workspace_scores=*/nullptr,
                        /*workspace_mask=*/nullptr,
                        &mpi_ctx_,
                        /*device_idx=*/0,
                        /*head_start=*/0,
                        /*local_n_heads=*/n_heads,
                        /*local_n_kv_heads=*/n_kv_heads,
                        /*gqa_n_rep=*/n_heads / n_kv_heads));

                    std::vector<float> expected(q_cols, 0.0f);
                    ASSERT_EQ(
                        hipMemcpyAsync(
                            expected.data(),
                            output_single->gpu_data_ptr(),
                            q_cols * sizeof(float),
                            hipMemcpyDeviceToHost,
                            stream),
                        hipSuccess);
                    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
                    EXPECT_TRUE(expectBitwiseEqualFP32Rows(
                        grouped_output.data() + q_row_begin,
                        expected.data(),
                        q_cols,
                        phase + " " + format.name +
                            " request=" + std::to_string(request) +
                            " row=" + std::to_string(row)));
                }
            }
        };

        expectSerialDecodeRows(
            actual,
            /*completed_blocks=*/0,
            "captured initial ROCm request batch");

        /*
         * Replay the identical graph without touching host-side attention
         * geometry.  The captured append advances each ring from eight to
         * sixteen live rows, the resident gather stripes to the new device
         * count, and attention must consume those rows with the original
         * fixed physical request stride.  This is the exact lifecycle that
         * previously left the second suffix block reading stale cache data.
         */
        ASSERT_EQ(hipGraphLaunch(graph_exec, stream), hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        EXPECT_EQ(kv_cache->get_cached_tokens(0, 0), 2 * seq_len);
        EXPECT_EQ(kv_cache->get_cached_tokens(0, 1), 2 * seq_len);

        /*
         * Prove the cache-to-attention handoff before comparing arithmetic.
         * A native-format append or fixed-stride gather bug can otherwise look
         * exactly like a batch-dependent attention reduction.  The expected
         * tensor is quantized once from the logical two-block history, while
         * the actual bytes are copied from the same resident gather view that
         * the production attention stage consumes.
         */
        std::vector<float> grown_k_host;
        std::vector<float> grown_v_host;
        grown_k_host.reserve(2 * kv_elements);
        grown_v_host.reserve(2 * kv_elements);
        for (int request = 0; request < batch_size; ++request)
        {
            const size_t kv_begin =
                static_cast<size_t>(request) * request_kv_elements;
            const auto k_begin =
                k_host.begin() + static_cast<std::ptrdiff_t>(kv_begin);
            const auto k_end =
                k_begin + static_cast<std::ptrdiff_t>(request_kv_elements);
            const auto v_begin =
                v_host.begin() + static_cast<std::ptrdiff_t>(kv_begin);
            const auto v_end =
                v_begin + static_cast<std::ptrdiff_t>(request_kv_elements);
            grown_k_host.insert(grown_k_host.end(), k_begin, k_end);
            grown_k_host.insert(grown_k_host.end(), k_begin, k_end);
            grown_v_host.insert(grown_v_host.end(), v_begin, v_end);
            grown_v_host.insert(grown_v_host.end(), v_begin, v_end);
        }

        auto expected_grown_k = makeNativeKV(
            grown_k_host,
            static_cast<size_t>(batch_size * 2 * seq_len),
            kv_cols,
            format.precision);
        auto expected_grown_v = makeNativeKV(
            grown_v_host,
            static_cast<size_t>(batch_size * 2 * seq_len),
            kv_cols,
            format.precision);
        ASSERT_NE(expected_grown_k, nullptr);
        ASSERT_NE(expected_grown_v, nullptr);

        ITensor *gathered_k = nullptr;
        ITensor *gathered_v = nullptr;
        ASSERT_TRUE(kv_cache->get_kv_batched_device_view(
            /*layer=*/0,
            /*first_seq_idx=*/0,
            batch_size,
            &gathered_k,
            &gathered_v,
            stream));
        ASSERT_NE(gathered_k, nullptr);
        ASSERT_NE(gathered_v, nullptr);
        ASSERT_EQ(gathered_k->size_bytes(), expected_grown_k->size_bytes());
        ASSERT_EQ(gathered_v->size_bytes(), expected_grown_v->size_bytes());

        std::vector<uint8_t> gathered_k_bytes(gathered_k->size_bytes());
        std::vector<uint8_t> gathered_v_bytes(gathered_v->size_bytes());
        ASSERT_EQ(
            hipMemcpyAsync(
                gathered_k_bytes.data(),
                gathered_k->gpu_data_ptr(),
                gathered_k_bytes.size(),
                hipMemcpyDeviceToHost,
                stream),
            hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(
                gathered_v_bytes.data(),
                gathered_v->gpu_data_ptr(),
                gathered_v_bytes.size(),
                hipMemcpyDeviceToHost,
                stream),
            hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        EXPECT_EQ(
            std::memcmp(
                gathered_k_bytes.data(),
                expected_grown_k->raw_data(),
                gathered_k_bytes.size()),
            0)
            << "Captured growing K gather changed native bytes for "
            << format.name;
        EXPECT_EQ(
            std::memcmp(
                gathered_v_bytes.data(),
                expected_grown_v->raw_data(),
                gathered_v_bytes.size()),
            0)
            << "Captured growing V gather changed native bytes for "
            << format.name;

        std::vector<float> grown_actual(q_elements, 0.0f);
        ASSERT_EQ(
            hipMemcpyAsync(
                grown_actual.data(), batch_output->gpu_data_ptr(),
                q_elements * sizeof(float), hipMemcpyDeviceToHost, stream),
            hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        expectSerialDecodeRows(
            grown_actual,
            /*completed_blocks=*/1,
            "captured grown ROCm request batch");

        ASSERT_EQ(hipGraphExecDestroy(graph_exec), hipSuccess);
        ASSERT_EQ(hipGraphDestroy(graph), hipSuccess);
        ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
    }
}

/**
 * @brief Prove captured unequal-history request decode across every ROCm cache format.
 *
 * Request-batched decode differs from ordinary attention batching in two ways:
 * each request owns an independent physical ring and each ring has its own live
 * post-append length.  The production gather therefore emits fixed-stride
 * request banks while the grouped decode kernel consumes cache-owned device
 * counts.  A host length, request-zero-only pointer, or one-row replay can all
 * pass an equal-length prefill test and still corrupt this path.
 *
 * Lengths 512 and 93 deliberately exercise different scalar launch policies on
 * gfx906: the long row selects a four-split, four-wavefront decode while the
 * short row selects a two-split, two-wavefront decode for this head geometry.
 * The grouped graph has one fixed physical launch but must reproduce both M=1
 * arithmetic orders byte-for-byte.  FP32, FP16, BF16, and Q8_1 cover every
 * standard ROCm ring format, including native grouped BF16 decode.
 */
TEST_F(
    Test__ROCmFlashAttentionParity,
    CapturedVariableLengthRequestBatchAllStandardKVFormatsDecodeMatchesSerialByteExact)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    constexpr int request_count = 2;
    constexpr int query_rows = 1;
    constexpr std::array<int, request_count> kv_lens{512, 93};
    constexpr int max_kv_len = kv_lens[0];
    constexpr int n_heads = 16;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 256;
    const size_t q_cols = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_cols = static_cast<size_t>(n_kv_heads) * head_dim;
    const size_t q_elements = static_cast<size_t>(request_count) * q_cols;
    const DeviceId device = DeviceId::rocm(0);

    const std::vector<float> q_host = randomFP32Scaled(q_elements, 0.25f);
    std::array<std::vector<float>, request_count> k_host;
    std::array<std::vector<float>, request_count> v_host;
    for (int request = 0; request < request_count; ++request)
    {
        const size_t elements =
            static_cast<size_t>(kv_lens[request]) * kv_cols;
        k_host[request] = randomFP32Scaled(elements, 0.25f);
        v_host[request] = randomFP32Scaled(elements, 0.25f);
        for (size_t index = 0; index < elements; ++index)
        {
            /* Request offsets expose accidental request-zero bank reuse. */
            k_host[request][index] +=
                static_cast<float>(request) * 0.375f;
            v_host[request][index] -=
                static_cast<float>(request) * 0.625f;
        }
    }

    auto makeFP32 = [](const std::vector<float> &values, size_t rows, size_t cols)
        -> std::shared_ptr<TensorBase>
    {
        auto tensor = std::make_shared<FP32Tensor>(
            std::vector<size_t>{rows, cols});
        std::copy(values.begin(), values.end(), tensor->mutable_data());
        return tensor;
    };
    auto makeNativeKV = [](
                            const std::vector<float> &values,
                            size_t rows,
                            size_t cols,
                            ActivationPrecision precision)
        -> std::shared_ptr<TensorBase>
    {
        const std::vector<size_t> shape{rows, cols};
        switch (precision)
        {
        case ActivationPrecision::FP32:
        {
            auto tensor = std::make_shared<FP32Tensor>(shape);
            std::copy(values.begin(), values.end(), tensor->mutable_data());
            return tensor;
        }
        case ActivationPrecision::FP16:
        {
            std::vector<uint16_t> fp16(values.size());
            quantizeToFP16(values.data(), fp16.data(), values.size());
            return std::make_shared<FP16Tensor>(shape, fp16);
        }
        case ActivationPrecision::BF16:
        {
            auto tensor = std::make_shared<BF16Tensor>(shape);
            tensor->from_fp32(values.data(), values.size());
            return tensor;
        }
        case ActivationPrecision::Q8_1:
            return Q8_1Tensor::quantize_from_fp32(values.data(), shape);
        default:
            return nullptr;
        }
    };

    struct FormatCase
    {
        ActivationPrecision precision;
        const char *name;
    };
    constexpr std::array<FormatCase, 4> formats{{
        {ActivationPrecision::FP32, "FP32"},
        {ActivationPrecision::FP16, "FP16"},
        {ActivationPrecision::BF16, "BF16"},
        {ActivationPrecision::Q8_1, "Q8_1"},
    }};

    auto &transfer = TransferEngine::instance();
    for (const FormatCase &format : formats)
    {
        SCOPED_TRACE(format.name);

        auto q_tensor = makeFP32(q_host, request_count, q_cols);
        auto grouped_output = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(request_count), q_cols});
        auto serial_output = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(request_count), q_cols});
        std::array<std::shared_ptr<TensorBase>, request_count> k_by_request;
        std::array<std::shared_ptr<TensorBase>, request_count> v_by_request;
        for (int request = 0; request < request_count; ++request)
        {
            k_by_request[request] = makeNativeKV(
                k_host[request], kv_lens[request], kv_cols, format.precision);
            v_by_request[request] = makeNativeKV(
                v_host[request], kv_lens[request], kv_cols, format.precision);
            ASSERT_NE(k_by_request[request], nullptr);
            ASSERT_NE(v_by_request[request], nullptr);
        }

        hipStream_t stream = nullptr;
        ASSERT_EQ(
            hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
            hipSuccess);
        ASSERT_TRUE(transfer.uploadFull(q_tensor.get(), device, stream).success);
        ASSERT_TRUE(
            transfer.uploadFull(grouped_output.get(), device, stream).success);
        ASSERT_TRUE(
            transfer.uploadFull(serial_output.get(), device, stream).success);
        for (int request = 0; request < request_count; ++request)
        {
            ASSERT_TRUE(transfer.uploadFull(
                k_by_request[request].get(), device, stream).success);
            ASSERT_TRUE(transfer.uploadFull(
                v_by_request[request].get(), device, stream).success);
        }

        llaminar::v2::kernels::KVCacheConfig cache_config;
        cache_config.precision = format.precision;
        cache_config.device = device;
        cache_config.num_layers = 1;
        cache_config.batch_size = request_count;
        cache_config.max_seq_len = max_kv_len + 8;
        cache_config.n_kv_heads = n_kv_heads;
        cache_config.head_dim = head_dim;
        auto kv_cache =
            llaminar::v2::kernels::KernelFactory::createKVCache(cache_config);
        ASSERT_NE(kv_cache, nullptr);

        auto *cache_workspace_consumer =
            dynamic_cast<IWorkspaceConsumer *>(kv_cache.get());
        ASSERT_NE(cache_workspace_consumer, nullptr);
        const WorkspaceRequirements cache_requirements =
            cache_workspace_consumer->getWorkspaceRequirements(
                max_kv_len, request_count, head_dim);
        DeviceWorkspaceManager cache_workspace(
            device,
            cache_requirements.total_bytes_with_alignment() + 4096);
        ASSERT_TRUE(cache_workspace.allocate(cache_requirements));
        cache_workspace_consumer->bindWorkspace(&cache_workspace);

        for (int request = 0; request < request_count; ++request)
        {
            ASSERT_TRUE(kv_cache->appendWithStream(
                /*layer=*/0,
                request,
                k_by_request[request].get(),
                v_by_request[request].get(),
                kv_lens[request],
                stream));
        }
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        for (int request = 0; request < request_count; ++request)
        {
            ASSERT_EQ(
                kv_cache->get_cached_tokens(/*layer=*/0, request),
                kv_lens[request]);
        }

        /*
         * K/V fields are declarative placeholders.  The production cache read
         * must replace them with the fixed-stride, request-major resident view.
         * Request zero's variable allocation is intentionally too short to be
         * mistaken for a contiguous two-request tensor.
         */
        AttentionComputeStage attention_stage({
            .device_id = device,
            .mpi_ctx = &mpi_ctx_,
            .Q = q_tensor.get(),
            .K = k_by_request[0].get(),
            .V = v_by_request[0].get(),
            .output = grouped_output.get(),
            .batch_size = request_count,
            .seq_len = query_rows,
            .kv_len = max_kv_len,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .causal = true,
            .auto_detect_mode = true,
            .kv_cache = kv_cache.get(),
            .layer_idx = 0,
            .read_kv_from_cache = true,
        });
        attention_stage.setGPUStream(stream);
        const WorkspaceRequirements attention_requirements =
            attention_stage.getWorkspaceRequirements(
                request_count * query_rows, n_heads, head_dim);
        DeviceWorkspaceManager attention_workspace(
            device,
            attention_requirements.total_bytes_with_alignment() + 4096);
        ASSERT_TRUE(attention_workspace.allocate(attention_requirements));
        attention_stage.bindWorkspace(&attention_workspace);

        /*
         * The host position is fixed launch geometry only.  The captured body
         * derives both unequal live lengths from cache-owned device state.
         */
        attention_stage.updateDynamicParams(
            max_kv_len - query_rows,
            query_rows);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        /* Materialize stable cache views before entering global capture. */
        ASSERT_TRUE(attention_stage.execute(nullptr));
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        hipGraph_t graph = nullptr;
        hipGraphExec_t graph_exec = nullptr;
        ASSERT_EQ(
            hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal),
            hipSuccess);
        bool capture_ok = false;
        {
            GraphCaptureGuard guard;
            capture_ok = attention_stage.execute(nullptr);
        }
        ASSERT_EQ(hipStreamEndCapture(stream, &graph), hipSuccess);
        ASSERT_TRUE(capture_ok);
        ASSERT_NE(graph, nullptr);
        ASSERT_EQ(
            hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
            hipSuccess);
        ASSERT_NE(graph_exec, nullptr);
        ASSERT_EQ(hipGraphLaunch(graph_exec, stream), hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32>
            serial_kernel(0);
        serial_kernel.setGPUStream(stream);
        const WorkspaceRequirements serial_requirements =
            serial_kernel.getWorkspaceRequirements(
                /*m=*/1, n_heads, head_dim);
        DeviceWorkspaceManager serial_workspace(
            device,
            serial_requirements.total_bytes_with_alignment() + 4096);
        ASSERT_TRUE(serial_workspace.allocate(serial_requirements));
        serial_kernel.bindWorkspace(&serial_workspace);

        auto *q_device = static_cast<float *>(q_tensor->gpu_data_ptr());
        auto *serial_device =
            static_cast<float *>(serial_output->gpu_data_ptr());
        ASSERT_NE(q_device, nullptr);
        ASSERT_NE(serial_device, nullptr);
        for (int request = 0; request < request_count; ++request)
        {
            GpuTensorView q_view(
                q_device + static_cast<size_t>(request) * q_cols,
                /*rows=*/1,
                q_cols,
                TensorType::FP32,
                /*device_id=*/0);
            GpuTensorView output_view(
                serial_device + static_cast<size_t>(request) * q_cols,
                /*rows=*/1,
                q_cols,
                TensorType::FP32,
                /*device_id=*/0);

            ASSERT_TRUE(serial_kernel.prepareDynamicAttnParams(
                kv_lens[request],
                kv_lens[request] - 1,
                /*query_rows=*/1,
                stream));
            ASSERT_TRUE(serial_kernel.compute_tensor(
                &q_view,
                k_by_request[request].get(),
                v_by_request[request].get(),
                &output_view,
                /*batch_size=*/1,
                /*seq_len=*/1,
                kv_lens[request],
                n_heads,
                n_kv_heads,
                head_dim,
                /*causal=*/true,
                /*window_size=*/-1,
                /*workspace_scores=*/nullptr,
                /*workspace_mask=*/nullptr,
                &mpi_ctx_,
                /*device_idx=*/0,
                /*head_start=*/0,
                /*local_n_heads=*/n_heads,
                /*local_n_kv_heads=*/n_kv_heads,
                /*gqa_n_rep=*/n_heads / n_kv_heads));
        }
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        std::vector<float> actual(q_elements, 0.0f);
        std::vector<float> expected(q_elements, 0.0f);
        ASSERT_EQ(
            hipMemcpyAsync(
                actual.data(),
                grouped_output->gpu_data_ptr(),
                q_elements * sizeof(float),
                hipMemcpyDeviceToHost,
                stream),
            hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(
                expected.data(),
                serial_output->gpu_data_ptr(),
                q_elements * sizeof(float),
                hipMemcpyDeviceToHost,
                stream),
            hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        for (int request = 0; request < request_count; ++request)
        {
            const size_t offset = static_cast<size_t>(request) * q_cols;
            EXPECT_TRUE(expectBitwiseEqualFP32Rows(
                actual.data() + offset,
                expected.data() + offset,
                q_cols,
                std::string("captured variable-length ROCm request decode ") +
                    format.name + " request=" + std::to_string(request)));
        }

        ASSERT_EQ(hipGraphExecDestroy(graph_exec), hipSuccess);
        ASSERT_EQ(hipGraphDestroy(graph), hipSuccess);
        ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
    }
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
    llaminar2::test::ScopedGPUStream stream(gpu_device);
    rocm_kernel.setGPUStream(stream.get());

    ASSERT_TRUE(with_gpu_coherence(
        gpu_device,
        {q_tensor.get(), k_tensor.get(), v_tensor.get()},
        {output_tensor.get()},
        stream.get(),
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
    llaminar2::test::ScopedGPUStream stream(gpu_device);
    rocm_kernel.setGPUStream(stream.get());

    ASSERT_TRUE(with_gpu_coherence(
        gpu_device,
        {q_tensor.get(), k_tensor.get(), v_tensor.get()},
        {output_tensor.get()},
        stream.get(),
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
    llaminar2::test::ScopedGPUStream stream(gpu_device);
    rocm_kernel.setGPUStream(stream.get());

    ASSERT_TRUE(with_gpu_coherence(
        gpu_device,
        {q_tensor.get(), k_tensor.get(), v_tensor.get()},
        {output_tensor.get()},
        stream.get(),
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

    /*
     * CPU PP fragments legally publish their terminal hidden state, so the CPU
     * snapshot source stops after layer four and does not introduce an LM-head
     * GEMV dependency. A standalone GPU forward has a stricter public boundary:
     * it must publish graph-declared logits. The ROCm graph therefore owns the
     * terminal norm and LM head. Both graphs execute identical embedding and
     * layer-0..4 prefixes, which is the only region sampled by this test.
     */
    auto cpu_stage_ctx =
        ModelContext::createForPPStage(
            TEST_MODEL_PATH,
            /*first_layer=*/0,
            /*last_layer=*/4,
            /*has_embedding=*/true,
            /*has_lm_head=*/false);
    auto rocm_stage_ctx =
        ModelContext::createForPPStage(
            TEST_MODEL_PATH,
            /*first_layer=*/0,
            /*last_layer=*/4,
            /*has_embedding=*/true,
            /*has_lm_head=*/true);
    ASSERT_NE(cpu_stage_ctx, nullptr);
    ASSERT_NE(rocm_stage_ctx, nullptr);

    FactoryPPStageConfig cpu_config;
    cpu_config.first_layer = 0;
    cpu_config.last_layer = 4;
    cpu_config.has_embedding = true;
    cpu_config.has_lm_head = false;
    ASSERT_TRUE(cpu_config.isValid());

    FactoryPPStageConfig rocm_config = cpu_config;
    rocm_config.has_lm_head = true;
    ASSERT_TRUE(rocm_config.isValid());

    auto cpu_runner =
        createPPStageRunner(cpu_stage_ctx, DeviceId::cpu(), cpu_config);
    auto rocm_runner =
        createPPStageRunner(rocm_stage_ctx, DeviceId::rocm(0), rocm_config);
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

    // The native tiled prefill path must keep real layer snapshots close to
    // the CPU reference in addition to preserving batch invariance.
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
    (void)hipMalloc(&d_Q, q_size * sizeof(float));
    (void)hipMalloc(&d_K, kv_size * sizeof(float));
    (void)hipMalloc(&d_V, kv_size * sizeof(float));
    (void)hipMalloc(&d_output, out_size * sizeof(float));

    (void)hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        false, // causal (see note above — production also uses false for seq_len=1)
        0);    // position_offset
    (void)hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    (void)hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    // Cleanup workspace before freeing GPU memory
    cleanupWorkspace(rocm_kernel);

    (void)hipFree(d_Q);
    (void)hipFree(d_K);
    (void)hipFree(d_V);
    (void)hipFree(d_output);

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
    (void)hipMalloc(&d_Q, q_size * sizeof(float));
    (void)hipMalloc(&d_K, kv_size * sizeof(float));
    (void)hipMalloc(&d_V, kv_size * sizeof(float));
    (void)hipMalloc(&d_output, out_size * sizeof(float));

    (void)hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        false, // causal (see VeryLong test for explanation)
        0);    // position_offset
    (void)hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    (void)hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    // Cleanup workspace before freeing GPU memory
    cleanupWorkspace(rocm_kernel);
    (void)hipFree(d_Q);
    (void)hipFree(d_K);
    (void)hipFree(d_V);
    (void)hipFree(d_output);

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
    (void)hipMalloc(&d_Q, q_size * sizeof(float));
    (void)hipMalloc(&d_K, kv_size * sizeof(float));
    (void)hipMalloc(&d_V, kv_size * sizeof(float));
    (void)hipMalloc(&d_output, out_size * sizeof(float));

    (void)hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        false, // causal (see VeryLong test for explanation)
        0);    // position_offset
    (void)hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    (void)hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    // Cleanup workspace before freeing GPU memory
    cleanupWorkspace(rocm_kernel);
    (void)hipFree(d_Q);
    (void)hipFree(d_K);
    (void)hipFree(d_V);
    (void)hipFree(d_output);

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
    (void)hipMalloc(&d_Q, q_size * sizeof(float));
    (void)hipMalloc(&d_K, kv_size * sizeof(float));
    (void)hipMalloc(&d_V, kv_size * sizeof(float));
    (void)hipMalloc(&d_output, out_size * sizeof(float));

    (void)hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        false, // non-causal
        0);    // position_offset
    (void)hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    (void)hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    // Cleanup workspace before freeing GPU memory
    cleanupWorkspace(rocm_kernel);
    (void)hipFree(d_Q);
    (void)hipFree(d_K);
    (void)hipFree(d_V);
    (void)hipFree(d_output);

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
    (void)hipMalloc(&d_Q, q_per_batch * sizeof(float));
    (void)hipMalloc(&d_K, kv_per_batch * sizeof(float));
    (void)hipMalloc(&d_V, kv_per_batch * sizeof(float));
    (void)hipMalloc(&d_output, out_per_batch * sizeof(float));

    bool all_success = true;

    // Process each batch element
    for (int b = 0; b < batch_size; b++)
    {
        (void)hipMemcpy(d_Q, Q_data.data() + b * q_per_batch,
                  q_per_batch * sizeof(float), hipMemcpyHostToDevice);
        (void)hipMemcpy(d_K, K_data.data() + b * kv_per_batch,
                  kv_per_batch * sizeof(float), hipMemcpyHostToDevice);
        (void)hipMemcpy(d_V, V_data.data() + b * kv_per_batch,
                  kv_per_batch * sizeof(float), hipMemcpyHostToDevice);
        (void)hipMemset(d_output, 0, out_per_batch * sizeof(float));

        // Use compute_decode for single-token decode
        bool success = rocm_kernel.compute_decode(
            d_Q, d_K, d_V, d_output,
            1,      // seq_len = 1 for decode
            kv_len, // kv_len from cache
            n_heads, n_kv_heads, head_dim,
            true, // causal
            0);   // position_offset
        (void)hipDeviceSynchronize();

        if (!success)
        {
            std::cerr << "Batch " << b << " decode failed" << std::endl;
            all_success = false;
            continue;
        }

        (void)hipMemcpy(rocm_output.data() + b * out_per_batch, d_output,
                  out_per_batch * sizeof(float), hipMemcpyDeviceToHost);
    }

    // Cleanup workspace before freeing GPU memory
    cleanupWorkspace(rocm_kernel);

    (void)hipFree(d_Q);
    (void)hipFree(d_K);
    (void)hipFree(d_V);
    (void)hipFree(d_output);

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
    bindAttentionStream(rocm_kernel);

    float *d_Q, *d_K, *d_V, *d_output;
    (void)hipMalloc(&d_Q, q_size * sizeof(float));
    (void)hipMalloc(&d_K, kv_size * sizeof(float));
    (void)hipMalloc(&d_V, kv_size * sizeof(float));
    (void)hipMalloc(&d_output, out_size * sizeof(float));

    (void)hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true, // causal
        -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    (void)hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    (void)hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    (void)hipFree(d_Q);
    (void)hipFree(d_K);
    (void)hipFree(d_V);
    (void)hipFree(d_output);

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
