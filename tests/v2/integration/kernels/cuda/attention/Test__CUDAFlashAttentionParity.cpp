/**
 * @file Test__CUDAFlashAttentionParity.cpp
 * @brief Parity tests for CUDA Flash Attention kernel vs CPU reference
 *
 * **Purpose**: Validate that CUDA Flash Attention kernels produce numerically
 * equivalent results to CPU attention kernels with high cosine similarity.
 *
 * **Tests**:
 * - Flash Attention 2 (prefill) vs CPU attention
 * - Flash Decoding (decode) vs CPU attention
 * - Various head dimensions (64, 128)
 * - GQA configurations (n_heads != n_kv_heads)
 *
 * **Pass Criteria**:
 * - Cosine similarity >= 0.99 (attention is numerically sensitive)
 * - No NaN/Inf in outputs
 * - Relative error < 5% for FP32
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>

// Include project headers BEFORE CUDATestUtils.h
#include "tensors/Tensors.h"
#include "execution/config/RuntimeConfig.h"
#include "execution/compute_stages/stages/AttentionComputeStage.h"
#include "execution/compute_stages/stages/KVCacheAppendStage.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/KernelFactory.h"
#include "transfer/TransferEngine.h"
#include "utils/MPIContext.h"
#include "kernels/cpu/CPURingKVCache.h"
#include "tensors/FP16Utils.h"

#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#include "kernels/cuda/attention/CUDAFlashAttentionKernelT.h"
#include "kernels/cpu/attention/CPUFlashAttentionKernelT.h"
#include <cuda_runtime.h>
#endif

// Now include test utils
#include "../../../../utils/CUDATestUtils.h"
#include "../../../../utils/TestTensorFactory.h"

#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>
#include <memory>
#include <filesystem>

#if LLAMINAR_CUDA_ATTENTION_PARITY_HAS_CNPY
#include <cnpy.h>
#endif

using namespace llaminar2;
using namespace llaminar2::test::cuda;
using namespace llaminar2::test;

namespace
{

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
     * @brief Load a float32 or float64 NumPy snapshot into a flat FP32 vector.
     *
     * The Qwen3.6 MoE classic parity suite writes stage snapshots as `.npy`
     * files. Loading them here lets the CUDA attention kernel replay the exact
     * Q/K/V tensors from the first full-attention layer, so a large-model math
     * failure has a small kernel-level reproducer.
     */
    std::vector<float> loadNpyFloatSnapshot(const std::filesystem::path &path)
    {
#if LLAMINAR_CUDA_ATTENTION_PARITY_HAS_CNPY
        const cnpy::NpyArray arr = cnpy::npy_load(path.string());
        std::vector<float> data;
        if (arr.word_size == sizeof(float))
        {
            const float *ptr = arr.data<float>();
            data.assign(ptr, ptr + arr.num_vals);
        }
        else if (arr.word_size == sizeof(double))
        {
            const double *ptr = arr.data<double>();
            data.resize(arr.num_vals);
            for (size_t i = 0; i < arr.num_vals; ++i)
            {
                data[i] = static_cast<float>(ptr[i]);
            }
        }
        return data;
#else
        (void)path;
        return {};
#endif
    }

#ifdef HAVE_CUDA
    /**
     * @brief Minimal cache facade with an intentionally suspect device count.
     *
     * The CUDA attention stage owns two possible sources of sequence length:
     * the host-visible cache count and, for graph-captured decode replay, a
     * device-resident count.  Regular prefill must not consume the device count
     * because no older cache state exists; doing so can make FA2 prefill attend
     * to stale or padded rows.  This fake lets the regression expose that bug
     * without requiring a full model graph.
     */
    class PrefillDeviceCountTrapKVCache final : public IKVCache
    {
    public:
        int host_cached_tokens = 0;
        const int *device_cached_tokens = nullptr;

        ActivationPrecision k_precision() const override { return ActivationPrecision::FP16; }
        ActivationPrecision v_precision() const override { return ActivationPrecision::FP16; }
        int get_cached_tokens(int, int = 0) const override { return host_cached_tokens; }
        int max_seq_len() const override { return 4096; }
        int n_layers() const override { return 1; }
        const int *deviceCachedTokenCountPtr(int, int = 0) const override
        {
            return device_cached_tokens;
        }

        bool get_kv(int, int, ITensor **out_k, ITensor **out_v,
                    int *out_kv_len = nullptr) override
        {
            if (out_k)
                *out_k = nullptr;
            if (out_v)
                *out_v = nullptr;
            if (out_kv_len)
                *out_kv_len = host_cached_tokens;
            return true;
        }

        bool get_kv(int, int, const ITensor **out_k, const ITensor **out_v,
                    int *out_kv_len = nullptr) const override
        {
            if (out_k)
                *out_k = nullptr;
            if (out_v)
                *out_v = nullptr;
            if (out_kv_len)
                *out_kv_len = host_cached_tokens;
            return true;
        }

        bool append(int, int, const ITensor *, const ITensor *, int) override
        {
            return false;
        }

        void clear() override { host_cached_tokens = 0; }
        void clear_sequence(int, int) override { host_cached_tokens = 0; }
        void clear_layer(int) override { host_cached_tokens = 0; }
    };
#endif

    // ============================================================================
    // CPU Reference for Decode Attention (single query attending to KV cache)
    // ============================================================================

    /**
     * @brief CPU reference implementation for decode attention
     *
     * Computes attention for a single query token (seq_len=1) attending to
     * a KV cache of length kv_len. This is the ground truth for Flash Decoding.
     *
     * Layout:
     *   Q: [n_heads, head_dim]
     *   K: [kv_len, n_kv_heads, head_dim]
     *   V: [kv_len, n_kv_heads, head_dim]
     *   O: [n_heads, head_dim]
     *
     * @param causal If true, Q at position (kv_len-1+position_offset) attends to K[0..kv_len-1].
     *               For standard decode, position_offset=0 means Q can see all of KV cache.
     */
    void cpuDecodeAttentionReference(
        const float *Q, // [n_heads, head_dim]
        const float *K, // [kv_len, n_kv_heads, head_dim]
        const float *V, // [kv_len, n_kv_heads, head_dim]
        float *O,       // [n_heads, head_dim]
        int kv_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        bool causal,
        int position_offset) // Q's logical position = kv_len - 1 + position_offset
    {
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        const int gqa_ratio = n_heads / n_kv_heads;

        // For each query head
        for (int h = 0; h < n_heads; h++)
        {
            const int kv_h = h / gqa_ratio; // GQA: which KV head to use

            const float *Q_head = Q + h * head_dim;
            float *O_head = O + h * head_dim;

            // Q's logical position for causal masking
            // In decode, Q is the "next" token, so it's at position (kv_len - 1 + position_offset)
            // If position_offset = 0, Q is at position kv_len-1 and can attend to all KV[0..kv_len-1]
            const int q_pos = kv_len - 1 + position_offset;

            // Step 1: Compute attention scores and find max for numerical stability
            std::vector<float> scores(kv_len);
            float max_score = -std::numeric_limits<float>::infinity();

            for (int kv_pos = 0; kv_pos < kv_len; kv_pos++)
            {
                // Causal mask: Q at q_pos can only attend to K at kv_pos if kv_pos <= q_pos
                bool masked = causal && (kv_pos > q_pos);

                if (masked)
                {
                    scores[kv_pos] = -std::numeric_limits<float>::infinity();
                }
                else
                {
                    const float *K_vec = K + kv_pos * n_kv_heads * head_dim + kv_h * head_dim;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; d++)
                    {
                        dot += Q_head[d] * K_vec[d];
                    }
                    scores[kv_pos] = dot * scale;
                    max_score = std::max(max_score, scores[kv_pos]);
                }
            }

            // Step 2: Compute softmax(scores) and weighted sum of V
            float sum_exp = 0.0f;
            for (int kv_pos = 0; kv_pos < kv_len; kv_pos++)
            {
                if (scores[kv_pos] > -std::numeric_limits<float>::infinity() / 2)
                {
                    scores[kv_pos] = std::exp(scores[kv_pos] - max_score);
                    sum_exp += scores[kv_pos];
                }
                else
                {
                    scores[kv_pos] = 0.0f;
                }
            }

            // Step 3: Compute output = sum(softmax_scores * V)
            for (int d = 0; d < head_dim; d++)
            {
                O_head[d] = 0.0f;
            }

            for (int kv_pos = 0; kv_pos < kv_len; kv_pos++)
            {
                if (scores[kv_pos] > 0.0f)
                {
                    float weight = scores[kv_pos] / sum_exp;
                    const float *V_vec = V + kv_pos * n_kv_heads * head_dim + kv_h * head_dim;
                    for (int d = 0; d < head_dim; d++)
                    {
                        O_head[d] += weight * V_vec[d];
                    }
                }
            }
        }
    }

    void cpuSmallCausalAttentionReference(
        const float *Q, // [seq_len, n_heads, head_dim]
        const float *K, // [kv_len, n_kv_heads, head_dim]
        const float *V, // [kv_len, n_kv_heads, head_dim]
        float *O,       // [seq_len, n_heads, head_dim]
        int seq_len,
        int kv_len,
        int n_heads,
        int n_kv_heads,
        int head_dim)
    {
        const size_t q_stride =
            static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim);
        for (int row = 0; row < seq_len; ++row)
        {
            const int row_kv_len = std::max(1, kv_len - (seq_len - 1 - row));
            cpuDecodeAttentionReference(
                Q + static_cast<size_t>(row) * q_stride,
                K,
                V,
                O + static_cast<size_t>(row) * q_stride,
                row_kv_len,
                n_heads,
                n_kv_heads,
                head_dim,
                false,
                0);
        }
    }

} // namespace

// ============================================================================
// Test Fixture
// ============================================================================

class Test__CUDAFlashAttentionParity : public CUDATestBase
{
protected:
    std::mt19937 rng_{42};
    std::uniform_real_distribution<float> dist_{-0.5f, 0.5f};
    MPIContext mpi_ctx_{0, 1, MPI_COMM_WORLD};

    std::vector<float> randomFP32(size_t count)
    {
        std::vector<float> data(count);
        for (auto &val : data)
        {
            val = dist_(rng_);
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

    std::unique_ptr<DeviceWorkspaceManager> bindAttentionWorkspace(
        llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> &kernel,
        int n_heads,
        int head_dim)
    {
        auto requirements = kernel.getWorkspaceRequirements(1, n_heads, head_dim);
        auto workspace = std::make_unique<DeviceWorkspaceManager>(
            gpu_device_, requirements.total_bytes_with_alignment() + 4096);
        if (!workspace->allocate(requirements))
        {
            ADD_FAILURE() << "Failed to allocate CUDA attention workspace";
            return nullptr;
        }
        kernel.bindWorkspace(workspace.get());
        return workspace;
    }
};

#ifdef HAVE_CUDA

TEST_F(Test__CUDAFlashAttentionParity, WorkspaceDeviceParamsSupportsSmallMVerifierRows)
{
    SKIP_IF_NO_CUDA();

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(cuda_ordinal_);
    const auto reqs = cuda_kernel.getWorkspaceRequirements(1, 14, 64);
    const auto *device_params =
        reqs.find(llaminar2::cuda::AttentionWorkspaceBuffers::DEVICE_PARAMS);
    ASSERT_NE(device_params, nullptr);
    EXPECT_GE(device_params->size_bytes,
              sizeof(llaminar2::attention::AttentionDeviceParams) * 4u)
        << "CUDA verifier attention needs one graph-replay param row per MTP verifier row";
}

TEST_F(Test__CUDAFlashAttentionParity, ComputeTensor_FP16KV_SmallM2_CausalGraphCapture)
{
    SKIP_IF_NO_CUDA();

    constexpr int seq_len = 2;
    constexpr int kv_len = 17;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    const size_t q_size =
        static_cast<size_t>(seq_len) * n_heads * head_dim;
    const size_t kv_size =
        static_cast<size_t>(kv_len) * n_kv_heads * head_dim;
    const size_t out_size = q_size;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);

    std::vector<uint16_t> K_fp16(kv_size);
    std::vector<uint16_t> V_fp16(kv_size);
    std::vector<float> K_ref(kv_size);
    std::vector<float> V_ref(kv_size);
    for (size_t i = 0; i < kv_size; ++i)
    {
        K_fp16[i] = fp32_to_fp16(K_data[i]);
        V_fp16[i] = fp32_to_fp16(V_data[i]);
        K_ref[i] = fp16_to_fp32(K_fp16[i]);
        V_ref[i] = fp16_to_fp32(V_fp16[i]);
    }

    std::vector<float> cpu_output(out_size, 0.0f);
    cpuSmallCausalAttentionReference(
        Q_data.data(), K_ref.data(), V_ref.data(), cpu_output.data(),
        seq_len, kv_len, n_heads, n_kv_heads, head_dim);

    auto q_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len),
                            static_cast<size_t>(n_heads * head_dim)},
        DeviceId::cpu());
    auto k_tensor = std::make_unique<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(kv_len),
                            static_cast<size_t>(n_kv_heads * head_dim)},
        K_fp16);
    auto v_tensor = std::make_unique<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(kv_len),
                            static_cast<size_t>(n_kv_heads * head_dim)},
        V_fp16);
    auto out_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len),
                            static_cast<size_t>(n_heads * head_dim)},
        DeviceId::cpu());

    std::copy(Q_data.begin(), Q_data.end(), q_tensor->mutable_data());

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    ASSERT_TRUE(q_tensor->ensureOnDevice(gpu_device_, stream));
    ASSERT_TRUE(k_tensor->ensureOnDevice(gpu_device_, stream));
    ASSERT_TRUE(v_tensor->ensureOnDevice(gpu_device_, stream));
    ASSERT_TRUE(out_tensor->ensureOnDevice(gpu_device_, stream));

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(cuda_ordinal_);
    cuda_kernel.setGPUStream(stream);
    DeviceWorkspaceManager workspace(gpu_device_, 64 * 1024 * 1024);
    ASSERT_TRUE(workspace.allocate(cuda_kernel.getWorkspaceRequirements(1, n_heads, head_dim)));
    cuda_kernel.bindWorkspace(&workspace);

    const int position_offset = kv_len - seq_len;
    ASSERT_TRUE(cuda_kernel.prepareDynamicAttnParams(kv_len, position_offset, seq_len, stream));
    ASSERT_TRUE(cuda_kernel.compute_tensor(
        q_tensor.get(), k_tensor.get(), v_tensor.get(), out_tensor.get(),
        1, seq_len, kv_len,
        n_heads, n_kv_heads, head_dim,
        true, -1,
        nullptr, nullptr,
        &mpi_ctx_, cuda_ordinal_,
        0, n_heads, n_kv_heads, n_heads / n_kv_heads));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<float> cuda_output(out_size, 0.0f);
    ASSERT_EQ(cudaMemcpyAsync(cuda_output.data(), out_tensor->gpu_data_ptr(),
                              out_size * sizeof(float), cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(cuda_output.data(), cpu_output.data(), out_size);
    printComparisonStats("ComputeTensor FP16KV Small-M2", cosine, l2_error, max_error, out_size);
    ASSERT_GE(cosine, 0.99);
    ASSERT_LE(l2_error, 0.05);

    ASSERT_EQ(cudaMemsetAsync(out_tensor->gpu_data_ptr(), 0, out_size * sizeof(float), stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_TRUE(cuda_kernel.prepareDynamicAttnParams(kv_len, position_offset, seq_len, stream));

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    {
        GraphCaptureGuard guard;
        ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), cudaSuccess);
        ASSERT_TRUE(cuda_kernel.compute_tensor(
            q_tensor.get(), k_tensor.get(), v_tensor.get(), out_tensor.get(),
            1, seq_len, kv_len,
            n_heads, n_kv_heads, head_dim,
            true, -1,
            nullptr, nullptr,
            &mpi_ctx_, cuda_ordinal_,
            0, n_heads, n_kv_heads, n_heads / n_kv_heads));
        ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
    }
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::fill(cuda_output.begin(), cuda_output.end(), 0.0f);
    ASSERT_EQ(cudaMemcpyAsync(cuda_output.data(), out_tensor->gpu_data_ptr(),
                              out_size * sizeof(float), cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), out_size);
    l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), out_size);
    max_error = maxAbsError(cuda_output.data(), cpu_output.data(), out_size);
    printComparisonStats("ComputeTensor FP16KV Small-M2 Captured", cosine, l2_error, max_error, out_size);
    EXPECT_GE(cosine, 0.99);
    EXPECT_LE(l2_error, 0.05);

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
}

TEST_F(Test__CUDAFlashAttentionParity, ComputeTensor_FP16KV_SmallM4MatchesSingleRowDecode)
{
    SKIP_IF_NO_CUDA();

    constexpr int seq_len = 4;
    constexpr int kv_len = 257;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    const size_t q_cols = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_cols = static_cast<size_t>(n_kv_heads) * head_dim;
    const size_t q_size = static_cast<size_t>(seq_len) * q_cols;
    const size_t kv_size = static_cast<size_t>(kv_len) * kv_cols;
    const size_t out_size = q_size;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);

    std::vector<uint16_t> K_fp16(kv_size);
    std::vector<uint16_t> V_fp16(kv_size);
    for (size_t i = 0; i < kv_size; ++i)
    {
        K_fp16[i] = fp32_to_fp16(K_data[i]);
        V_fp16[i] = fp32_to_fp16(V_data[i]);
    }

    auto q_m4_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), q_cols},
        DeviceId::cpu());
    auto k_tensor = std::make_unique<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(kv_len), kv_cols},
        K_fp16);
    auto v_tensor = std::make_unique<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(kv_len), kv_cols},
        V_fp16);
    auto out_m4_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), q_cols},
        DeviceId::cpu());

    std::copy(Q_data.begin(), Q_data.end(), q_m4_tensor->mutable_data());

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    ASSERT_TRUE(q_m4_tensor->ensureOnDevice(gpu_device_, stream));
    ASSERT_TRUE(k_tensor->ensureOnDevice(gpu_device_, stream));
    ASSERT_TRUE(v_tensor->ensureOnDevice(gpu_device_, stream));
    ASSERT_TRUE(out_m4_tensor->ensureOnDevice(gpu_device_, stream));

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(cuda_ordinal_);
    cuda_kernel.setGPUStream(stream);
    DeviceWorkspaceManager workspace(gpu_device_, 64 * 1024 * 1024);
    ASSERT_TRUE(workspace.allocate(cuda_kernel.getWorkspaceRequirements(1, n_heads, head_dim)));
    cuda_kernel.bindWorkspace(&workspace);

    const int position_offset = kv_len - seq_len;
    ASSERT_TRUE(cuda_kernel.prepareDynamicAttnParams(kv_len, position_offset, seq_len, stream));
    ASSERT_TRUE(cuda_kernel.compute_tensor(
        q_m4_tensor.get(), k_tensor.get(), v_tensor.get(), out_m4_tensor.get(),
        1, seq_len, kv_len,
        n_heads, n_kv_heads, head_dim,
        true, -1,
        nullptr, nullptr,
        &mpi_ctx_, cuda_ordinal_,
        0, n_heads, n_kv_heads, n_heads / n_kv_heads));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<float> m4_output(out_size, 0.0f);
    ASSERT_EQ(cudaMemcpyAsync(m4_output.data(), out_m4_tensor->gpu_data_ptr(),
                              out_size * sizeof(float), cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    for (int row = 0; row < seq_len; ++row)
    {
        auto q_m1_tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{size_t{1}, q_cols},
            DeviceId::cpu());
        auto out_m1_tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{size_t{1}, q_cols},
            DeviceId::cpu());
        std::memcpy(q_m1_tensor->mutable_data(),
                    Q_data.data() + static_cast<size_t>(row) * q_cols,
                    q_cols * sizeof(float));

        ASSERT_TRUE(q_m1_tensor->ensureOnDevice(gpu_device_, stream));
        ASSERT_TRUE(out_m1_tensor->ensureOnDevice(gpu_device_, stream));

        const int row_kv_len = std::max(1, kv_len - (seq_len - 1 - row));
        ASSERT_TRUE(cuda_kernel.prepareDynamicAttnParams(
            row_kv_len, row_kv_len - 1, 1, stream));
        ASSERT_TRUE(cuda_kernel.compute_tensor(
            q_m1_tensor.get(), k_tensor.get(), v_tensor.get(), out_m1_tensor.get(),
            1, 1, row_kv_len,
            n_heads, n_kv_heads, head_dim,
            true, -1,
            nullptr, nullptr,
            &mpi_ctx_, cuda_ordinal_,
            0, n_heads, n_kv_heads, n_heads / n_kv_heads))
            << "single-row decode attention failed for row " << row;
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        std::vector<float> m1_output(q_cols, 0.0f);
        ASSERT_EQ(cudaMemcpyAsync(m1_output.data(), out_m1_tensor->gpu_data_ptr(),
                                  q_cols * sizeof(float), cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        const float *m4_row = m4_output.data() + static_cast<size_t>(row) * q_cols;
        const double cosine = cosineSimilarity(m4_row, m1_output.data(), q_cols);
        const double l2_error = relativeL2Error(m4_row, m1_output.data(), q_cols);
        const double max_error = maxAbsError(m4_row, m1_output.data(), q_cols);
        printComparisonStats("ComputeTensor FP16KV Small-M4 vs M1", cosine, l2_error, max_error, q_cols);
        EXPECT_GE(cosine, 0.999999)
            << "M=4 verifier attention row " << row
            << " diverges from single-row decode attention";
        EXPECT_LE(l2_error, 1e-5)
            << "M=4 verifier attention row " << row
            << " relative L2 differs from single-row decode attention";
    }

    cudaStreamDestroy(stream);
}

TEST_F(Test__CUDAFlashAttentionParity, ComputeTensor_FP16KV_Qwen36M2FA2ContinuationRowsMatchSerialDecode)
{
    SKIP_IF_NO_CUDA();

    /*
     * The vLLM-style MTP verifier runs a two-row causal continuation after
     * both verifier K/V rows have already been appended to the cache.  That
     * path is intentionally different from the older row-local small-M decode
     * path: it prepares one dynamic AttentionDeviceParams record for the whole
     * verifier span, then FA2 masks future verifier rows by position.  Keep
     * this regression shaped like Qwen3.6 FA layers so the production path is
     * covered without loading a large model.
     */
    constexpr int seq_len = 2;
    constexpr int kv_len = 599;
    constexpr int n_heads = 16;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 256;
    const size_t q_cols = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_cols = static_cast<size_t>(n_kv_heads) * head_dim;
    const size_t q_size = static_cast<size_t>(seq_len) * q_cols;
    const size_t kv_size = static_cast<size_t>(kv_len) * kv_cols;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);

    std::vector<uint16_t> K_fp16(kv_size);
    std::vector<uint16_t> V_fp16(kv_size);
    for (size_t i = 0; i < kv_size; ++i)
    {
        K_fp16[i] = fp32_to_fp16(K_data[i]);
        V_fp16[i] = fp32_to_fp16(V_data[i]);
    }

    auto q_m2_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), q_cols},
        DeviceId::cpu());
    auto k_tensor = std::make_unique<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(kv_len), kv_cols},
        K_fp16);
    auto v_tensor = std::make_unique<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(kv_len), kv_cols},
        V_fp16);
    auto out_fa2_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), q_cols},
        DeviceId::cpu());
    std::copy(Q_data.begin(), Q_data.end(), q_m2_tensor->mutable_data());

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    auto &transfer = TransferEngine::instance();
    ASSERT_TRUE(transfer.uploadFull(q_m2_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(k_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(v_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(out_fa2_tensor.get(), gpu_device_, stream).success);

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> fa2_kernel(cuda_ordinal_);
    fa2_kernel.setGPUStream(stream);
    const auto fa2_reqs = fa2_kernel.getWorkspaceRequirements(1, n_heads, head_dim);
    DeviceWorkspaceManager fa2_workspace(gpu_device_,
                                         fa2_reqs.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(fa2_workspace.allocate(fa2_reqs));
    fa2_kernel.bindWorkspace(&fa2_workspace);

    const int position_offset = kv_len - seq_len;
    ASSERT_TRUE(fa2_kernel.prepareDynamicAttnParams(kv_len, position_offset, 1, stream));
    ASSERT_TRUE(fa2_kernel.compute_tensor(
        q_m2_tensor.get(), k_tensor.get(), v_tensor.get(), out_fa2_tensor.get(),
        1, seq_len, kv_len,
        n_heads, n_kv_heads, head_dim,
        true, -1,
        nullptr, nullptr,
        &mpi_ctx_, cuda_ordinal_,
        0, n_heads, n_kv_heads, n_heads / n_kv_heads));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<float> fa2_output(q_size, 0.0f);
    ASSERT_EQ(cudaMemcpyAsync(fa2_output.data(), out_fa2_tensor->gpu_data_ptr(),
                              q_size * sizeof(float), cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> serial_kernel(cuda_ordinal_);
    serial_kernel.setGPUStream(stream);
    const auto serial_reqs = serial_kernel.getWorkspaceRequirements(1, n_heads, head_dim);
    DeviceWorkspaceManager serial_workspace(gpu_device_,
                                            serial_reqs.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(serial_workspace.allocate(serial_reqs));
    serial_kernel.bindWorkspace(&serial_workspace);

    for (int row = 0; row < seq_len; ++row)
    {
        auto q_m1_tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{size_t{1}, q_cols},
            DeviceId::cpu());
        auto out_m1_tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{size_t{1}, q_cols},
            DeviceId::cpu());
        std::memcpy(q_m1_tensor->mutable_data(),
                    Q_data.data() + static_cast<size_t>(row) * q_cols,
                    q_cols * sizeof(float));
        ASSERT_TRUE(transfer.uploadFull(q_m1_tensor.get(), gpu_device_, stream).success);
        ASSERT_TRUE(transfer.uploadFull(out_m1_tensor.get(), gpu_device_, stream).success);

        const int row_kv_len = std::max(1, kv_len - (seq_len - 1 - row));
        ASSERT_TRUE(serial_kernel.prepareDynamicAttnParams(row_kv_len, row_kv_len - 1, 1, stream));
        ASSERT_TRUE(serial_kernel.compute_tensor(
            q_m1_tensor.get(), k_tensor.get(), v_tensor.get(), out_m1_tensor.get(),
            1, 1, row_kv_len,
            n_heads, n_kv_heads, head_dim,
            true, -1,
            nullptr, nullptr,
            &mpi_ctx_, cuda_ordinal_,
            0, n_heads, n_kv_heads, n_heads / n_kv_heads))
            << "serial decode attention failed for verifier row " << row;
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        std::vector<float> serial_output(q_cols, 0.0f);
        ASSERT_EQ(cudaMemcpyAsync(serial_output.data(), out_m1_tensor->gpu_data_ptr(),
                                  q_cols * sizeof(float), cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        const float *fa2_row = fa2_output.data() + static_cast<size_t>(row) * q_cols;
        const double cosine = cosineSimilarity(fa2_row, serial_output.data(), q_cols);
        const double l2_error = relativeL2Error(fa2_row, serial_output.data(), q_cols);
        const double max_error = maxAbsError(fa2_row, serial_output.data(), q_cols);
        printComparisonStats("ComputeTensor FP16KV Qwen3.6 M2 FA2 continuation vs M1",
                             cosine, l2_error, max_error, q_cols);
        EXPECT_GE(cosine, 0.999999)
            << "FA2 continuation row " << row
            << " diverges from serial decode attention";
        EXPECT_LE(l2_error, 1e-5)
            << "FA2 continuation row " << row
            << " relative L2 differs from serial decode attention";
    }

    cudaStreamDestroy(stream);
}

TEST_F(Test__CUDAFlashAttentionParity, ComputeTensor_FP16KV_Qwen36M2DeviceDerivedRowsMatchSerialDecode)
{
    SKIP_IF_NO_CUDA();

    /*
     * Qwen3.6 MTP verifier publication runs a tiny continuation against a
     * long live KV cache.  The production graph derives one attention param
     * row per verifier row from the device-side post-append cache count, then
     * each row must match the exact serial decode result at its row-local KV
     * length.  This catches bugs hidden by smaller head_dim=64 smoke tests.
     */
    constexpr int seq_len = 2;
    constexpr int kv_len = 599;
    constexpr int n_heads = 16;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 256;
    const size_t q_cols = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_cols = static_cast<size_t>(n_kv_heads) * head_dim;
    const size_t q_size = static_cast<size_t>(seq_len) * q_cols;
    const size_t kv_size = static_cast<size_t>(kv_len) * kv_cols;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);

    std::vector<uint16_t> K_fp16(kv_size);
    std::vector<uint16_t> V_fp16(kv_size);
    for (size_t i = 0; i < kv_size; ++i)
    {
        K_fp16[i] = fp32_to_fp16(K_data[i]);
        V_fp16[i] = fp32_to_fp16(V_data[i]);
    }

    auto q_m2_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), q_cols},
        DeviceId::cpu());
    auto k_tensor = std::make_unique<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(kv_len), kv_cols},
        K_fp16);
    auto v_tensor = std::make_unique<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(kv_len), kv_cols},
        V_fp16);
    auto out_m2_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), q_cols},
        DeviceId::cpu());
    std::copy(Q_data.begin(), Q_data.end(), q_m2_tensor->mutable_data());

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    auto &transfer = TransferEngine::instance();
    ASSERT_TRUE(transfer.uploadFull(q_m2_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(k_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(v_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(out_m2_tensor.get(), gpu_device_, stream).success);

    int *d_cached_tokens = nullptr;
    ASSERT_EQ(cudaMalloc(&d_cached_tokens, sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_cached_tokens, &kv_len, sizeof(int),
                              cudaMemcpyHostToDevice, stream),
              cudaSuccess);

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(cuda_ordinal_);
    cuda_kernel.setGPUStream(stream);
    DeviceWorkspaceManager workspace(gpu_device_,
                                     cuda_kernel.getWorkspaceRequirements(1, n_heads, head_dim)
                                             .total_bytes_with_alignment() +
                                         4096);
    ASSERT_TRUE(workspace.allocate(cuda_kernel.getWorkspaceRequirements(1, n_heads, head_dim)));
    cuda_kernel.bindWorkspace(&workspace);

    ASSERT_TRUE(cuda_kernel.prepareDynamicAttnParamsFromDeviceSequenceState(
        d_cached_tokens, seq_len, seq_len, stream));
    ASSERT_TRUE(cuda_kernel.compute_tensor(
        q_m2_tensor.get(), k_tensor.get(), v_tensor.get(), out_m2_tensor.get(),
        1, seq_len, kv_len,
        n_heads, n_kv_heads, head_dim,
        true, -1,
        nullptr, nullptr,
        &mpi_ctx_, cuda_ordinal_,
        0, n_heads, n_kv_heads, n_heads / n_kv_heads));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<float> m2_output(q_size, 0.0f);
    ASSERT_EQ(cudaMemcpyAsync(m2_output.data(), out_m2_tensor->gpu_data_ptr(),
                              q_size * sizeof(float), cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    for (int row = 0; row < seq_len; ++row)
    {
        auto q_m1_tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{size_t{1}, q_cols},
            DeviceId::cpu());
        auto out_m1_tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{size_t{1}, q_cols},
            DeviceId::cpu());
        std::memcpy(q_m1_tensor->mutable_data(),
                    Q_data.data() + static_cast<size_t>(row) * q_cols,
                    q_cols * sizeof(float));

        ASSERT_TRUE(transfer.uploadFull(q_m1_tensor.get(), gpu_device_, stream).success);
        ASSERT_TRUE(transfer.uploadFull(out_m1_tensor.get(), gpu_device_, stream).success);

        const int row_kv_len = std::max(1, kv_len - (seq_len - 1 - row));
        ASSERT_TRUE(cuda_kernel.prepareDynamicAttnParams(
            row_kv_len, row_kv_len - 1, 1, stream));
        ASSERT_TRUE(cuda_kernel.compute_tensor(
            q_m1_tensor.get(), k_tensor.get(), v_tensor.get(), out_m1_tensor.get(),
            1, 1, row_kv_len,
            n_heads, n_kv_heads, head_dim,
            true, -1,
            nullptr, nullptr,
            &mpi_ctx_, cuda_ordinal_,
            0, n_heads, n_kv_heads, n_heads / n_kv_heads))
            << "single-row decode attention failed for row " << row;
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        std::vector<float> m1_output(q_cols, 0.0f);
        ASSERT_EQ(cudaMemcpyAsync(m1_output.data(), out_m1_tensor->gpu_data_ptr(),
                                  q_cols * sizeof(float), cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        const float *m2_row = m2_output.data() + static_cast<size_t>(row) * q_cols;
        const double cosine = cosineSimilarity(m2_row, m1_output.data(), q_cols);
        const double l2_error = relativeL2Error(m2_row, m1_output.data(), q_cols);
        const double max_error = maxAbsError(m2_row, m1_output.data(), q_cols);
        printComparisonStats("ComputeTensor FP16KV Qwen3.6 M2 device-derived vs M1",
                             cosine, l2_error, max_error, q_cols);
        EXPECT_GE(cosine, 0.999999)
            << "Qwen3.6 M=2 verifier attention row " << row
            << " diverges from serial decode attention";
        EXPECT_LE(l2_error, 1e-5)
            << "Qwen3.6 M=2 verifier attention row " << row
            << " relative L2 differs from serial decode attention";
    }

    cudaFree(d_cached_tokens);
    cudaStreamDestroy(stream);
}

TEST_F(Test__CUDAFlashAttentionParity, AttentionStage_FP16Cache_Qwen36M2RoPEOnReadRowsMatchSerialDecode)
{
    SKIP_IF_NO_CUDA();

    /*
     * The Qwen3.6 MTP verifier uses AttentionComputeStage, not the bare
     * kernel.  This test keeps the production handoff in the loop: K/V are
     * appended to the ring cache, the stage reads the cache through the
     * RoPE-on-read FP16 shadow, and dynamic params are derived from the cache's
     * device-resident token count.  Each verifier row must still equal serial
     * one-row decode over the same cache view.
     */
    constexpr int seq_len = 2;
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

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);

    std::vector<uint16_t> K_fp16(kv_size);
    std::vector<uint16_t> V_fp16(kv_size);
    for (size_t i = 0; i < kv_size; ++i)
    {
        K_fp16[i] = fp32_to_fp16(K_data[i]);
        V_fp16[i] = fp32_to_fp16(V_data[i]);
    }

    const DeviceId device = gpu_device_;
    auto q_m2_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), q_cols},
        DeviceId::cpu());
    auto k_tensor = std::make_unique<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(kv_len), kv_cols},
        K_fp16);
    auto v_tensor = std::make_unique<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(kv_len), kv_cols},
        V_fp16);
    auto out_m2_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), q_cols},
        DeviceId::cpu());
    std::copy(Q_data.begin(), Q_data.end(), q_m2_tensor->mutable_data());

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    auto &transfer = TransferEngine::instance();
    ASSERT_TRUE(transfer.uploadFull(q_m2_tensor.get(), device, stream).success);
    ASSERT_TRUE(transfer.uploadFull(k_tensor.get(), device, stream).success);
    ASSERT_TRUE(transfer.uploadFull(v_tensor.get(), device, stream).success);
    ASSERT_TRUE(transfer.uploadFull(out_m2_tensor.get(), device, stream).success);

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
    ASSERT_TRUE(kv_cache->appendWithStream(0, 0, k_tensor.get(), v_tensor.get(), kv_len, stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(kv_cache->get_cached_tokens(0, 0), kv_len);
    ASSERT_NE(kv_cache->deviceCachedTokenCountPtr(0, 0), nullptr);

    AttentionComputeStage::Params params;
    params.device_id = device;
    params.Q = q_m2_tensor.get();
    params.K = k_tensor.get();
    params.V = v_tensor.get();
    params.output = out_m2_tensor.get();
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
    params.mpi_ctx = &mpi_ctx_;

    AttentionComputeStage stage(params);
    stage.setGPUStream(stream);
    const WorkspaceRequirements stage_reqs =
        stage.getWorkspaceRequirements(/*m=*/1, /*n=*/n_heads, /*k=*/head_dim);
    DeviceWorkspaceManager stage_workspace(
        device, stage_reqs.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(stage_workspace.allocate(stage_reqs));
    stage.bindWorkspace(&stage_workspace);

    ASSERT_TRUE(stage.execute(nullptr));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<float> m2_output(q_size, 0.0f);
    ASSERT_EQ(cudaMemcpyAsync(m2_output.data(), out_m2_tensor->gpu_data_ptr(),
                              q_size * sizeof(float), cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

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
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cache_kv_len, kv_len);
    ASSERT_NE(cache_k, nullptr);
    ASSERT_NE(cache_v, nullptr);

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> serial_kernel(cuda_ordinal_);
    serial_kernel.setGPUStream(stream);
    DeviceWorkspaceManager serial_workspace(
        device, serial_kernel.getWorkspaceRequirements(1, n_heads, head_dim)
                    .total_bytes_with_alignment() +
                    4096);
    ASSERT_TRUE(serial_workspace.allocate(serial_kernel.getWorkspaceRequirements(1, n_heads, head_dim)));
    serial_kernel.bindWorkspace(&serial_workspace);

    for (int row = 0; row < seq_len; ++row)
    {
        auto q_m1_tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{size_t{1}, q_cols},
            DeviceId::cpu());
        auto out_m1_tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{size_t{1}, q_cols},
            DeviceId::cpu());
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
            true, -1,
            nullptr, nullptr,
            &mpi_ctx_, cuda_ordinal_,
            0, n_heads, n_kv_heads, n_heads / n_kv_heads));
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        std::vector<float> m1_output(q_cols, 0.0f);
        ASSERT_EQ(cudaMemcpyAsync(m1_output.data(), out_m1_tensor->gpu_data_ptr(),
                                  q_cols * sizeof(float), cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        const float *m2_row = m2_output.data() + static_cast<size_t>(row) * q_cols;
        const double cosine = cosineSimilarity(m2_row, m1_output.data(), q_cols);
        const double l2_error = relativeL2Error(m2_row, m1_output.data(), q_cols);
        const double max_error = maxAbsError(m2_row, m1_output.data(), q_cols);
        printComparisonStats("AttentionStage FP16 cache Qwen3.6 M2 RoPE-on-read vs M1",
                             cosine, l2_error, max_error, q_cols);
        EXPECT_GE(cosine, 0.999999)
            << "stage row " << row << " must match serial decode";
        EXPECT_LE(l2_error, 1e-5)
            << "stage row " << row << " relative L2 differs from serial decode";
    }

    cudaStreamDestroy(stream);
}

TEST_F(Test__CUDAFlashAttentionParity, CapturedAppendThenAttention_FP16Cache_Qwen36M2RoPEOnReadRowsMatchSerialDecode)
{
    SKIP_IF_NO_CUDA();

    /*
     * Regression for vLLM-style all-position verifier capture: the verifier
     * graph records KV append for the current two rows and then immediately
     * consumes the same cache through AttentionComputeStage.  The cache's
     * RoPE-on-read FP16 shadow is warmed at the pre-append history length so
     * the captured path exercises the incremental shadow update used by real
     * decode graphs.
     */
    constexpr int seq_len = 2;
    constexpr int history_len = 597;
    constexpr int kv_len = history_len + seq_len;
    constexpr int n_heads = 16;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 256;
    constexpr float rope_theta = 10000000.0f;
    constexpr float partial_rotary_factor = 0.25f;
    const size_t q_cols = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_cols = static_cast<size_t>(n_kv_heads) * head_dim;
    const size_t q_size = static_cast<size_t>(seq_len) * q_cols;
    const size_t kv_size = static_cast<size_t>(kv_len) * kv_cols;
    const size_t history_size = static_cast<size_t>(history_len) * kv_cols;
    const size_t current_size = static_cast<size_t>(seq_len) * kv_cols;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);

    std::vector<uint16_t> K_fp16(kv_size);
    std::vector<uint16_t> V_fp16(kv_size);
    for (size_t i = 0; i < kv_size; ++i)
    {
        K_fp16[i] = fp32_to_fp16(K_data[i]);
        V_fp16[i] = fp32_to_fp16(V_data[i]);
    }

    const DeviceId device = gpu_device_;
    auto q_m2_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), q_cols},
        DeviceId::cpu());
    auto history_k_tensor = std::make_unique<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(history_len), kv_cols},
        std::vector<uint16_t>(K_fp16.begin(), K_fp16.begin() + static_cast<std::ptrdiff_t>(history_size)));
    auto history_v_tensor = std::make_unique<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(history_len), kv_cols},
        std::vector<uint16_t>(V_fp16.begin(), V_fp16.begin() + static_cast<std::ptrdiff_t>(history_size)));
    auto current_k_tensor = std::make_unique<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), kv_cols},
        std::vector<uint16_t>(K_fp16.begin() + static_cast<std::ptrdiff_t>(history_size), K_fp16.end()));
    auto current_v_tensor = std::make_unique<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), kv_cols},
        std::vector<uint16_t>(V_fp16.begin() + static_cast<std::ptrdiff_t>(history_size), V_fp16.end()));
    auto full_k_tensor = std::make_unique<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(kv_len), kv_cols},
        K_fp16);
    auto full_v_tensor = std::make_unique<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(kv_len), kv_cols},
        V_fp16);
    auto out_m2_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), q_cols},
        DeviceId::cpu());
    std::copy(Q_data.begin(), Q_data.end(), q_m2_tensor->mutable_data());

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    auto &transfer = TransferEngine::instance();
    ASSERT_TRUE(transfer.uploadFull(q_m2_tensor.get(), device, stream).success);
    ASSERT_TRUE(transfer.uploadFull(history_k_tensor.get(), device, stream).success);
    ASSERT_TRUE(transfer.uploadFull(history_v_tensor.get(), device, stream).success);
    ASSERT_TRUE(transfer.uploadFull(current_k_tensor.get(), device, stream).success);
    ASSERT_TRUE(transfer.uploadFull(current_v_tensor.get(), device, stream).success);
    ASSERT_TRUE(transfer.uploadFull(full_k_tensor.get(), device, stream).success);
    ASSERT_TRUE(transfer.uploadFull(full_v_tensor.get(), device, stream).success);
    ASSERT_TRUE(transfer.uploadFull(out_m2_tensor.get(), device, stream).success);

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
    ASSERT_TRUE(kv_cache->appendWithStream(0, 0, history_k_tensor.get(), history_v_tensor.get(), history_len, stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
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
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(warm_len, history_len);

    AttentionComputeStage::Params attn_params;
    attn_params.device_id = device;
    attn_params.Q = q_m2_tensor.get();
    attn_params.K = current_k_tensor.get();
    attn_params.V = current_v_tensor.get();
    attn_params.output = out_m2_tensor.get();
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
    attn_params.mpi_ctx = &mpi_ctx_;

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
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), cudaSuccess);
    bool capture_ok = false;
    {
        GraphCaptureGuard guard(/*host_bookkeeping=*/true);
        capture_ok = append_stage.execute(nullptr) && attn_stage.execute(nullptr);
    }
    ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
    ASSERT_TRUE(capture_ok);
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), cudaSuccess);
    ASSERT_NE(graph_exec, nullptr);
    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(kv_cache->get_cached_tokens(0, 0), kv_len);

    std::vector<float> m2_output(q_size, 0.0f);
    ASSERT_EQ(cudaMemcpyAsync(m2_output.data(), out_m2_tensor->gpu_data_ptr(),
                              q_size * sizeof(float), cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    auto ref_cache = llaminar::v2::kernels::KernelFactory::createKVCache(config);
    ASSERT_NE(ref_cache, nullptr);
    ASSERT_TRUE(ref_cache->appendWithStream(0, 0, full_k_tensor.get(), full_v_tensor.get(), kv_len, stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    ITensor *cache_k = nullptr;
    ITensor *cache_v = nullptr;
    int cache_kv_len = 0;
    ASSERT_TRUE(ref_cache->get_kv_converted(
        0, 0, ActivationPrecision::FP16, &cache_k, &cache_v, &cache_kv_len, &warm_read_params));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cache_kv_len, kv_len);

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> serial_kernel(cuda_ordinal_);
    serial_kernel.setGPUStream(stream);
    DeviceWorkspaceManager serial_workspace(
        device, serial_kernel.getWorkspaceRequirements(1, n_heads, head_dim)
                    .total_bytes_with_alignment() +
                    4096);
    ASSERT_TRUE(serial_workspace.allocate(serial_kernel.getWorkspaceRequirements(1, n_heads, head_dim)));
    serial_kernel.bindWorkspace(&serial_workspace);

    for (int row = 0; row < seq_len; ++row)
    {
        auto q_m1_tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{size_t{1}, q_cols},
            DeviceId::cpu());
        auto out_m1_tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{size_t{1}, q_cols},
            DeviceId::cpu());
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
            true, -1,
            nullptr, nullptr,
            &mpi_ctx_, cuda_ordinal_,
            0, n_heads, n_kv_heads, n_heads / n_kv_heads));
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        std::vector<float> m1_output(q_cols, 0.0f);
        ASSERT_EQ(cudaMemcpyAsync(m1_output.data(), out_m1_tensor->gpu_data_ptr(),
                                  q_cols * sizeof(float), cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        const float *m2_row = m2_output.data() + static_cast<size_t>(row) * q_cols;
        const double cosine = cosineSimilarity(m2_row, m1_output.data(), q_cols);
        const double l2_error = relativeL2Error(m2_row, m1_output.data(), q_cols);
        const double max_error = maxAbsError(m2_row, m1_output.data(), q_cols);
        printComparisonStats("Captured append+attention FP16 cache Qwen3.6 M2 RoPE-on-read vs M1",
                             cosine, l2_error, max_error, q_cols);
        EXPECT_GE(cosine, 0.999999)
            << "captured append+attention row " << row << " must match serial decode";
        EXPECT_LE(l2_error, 1e-5)
            << "captured append+attention row " << row << " relative L2 differs from serial decode";
    }

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
}

// ============================================================================
// Flash Attention 2 (Prefill) Parity Tests
// ============================================================================

TEST_F(Test__CUDAFlashAttentionParity, FlashAttn2_FP32_Small)
{
    SKIP_IF_NO_CUDA();

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
    std::vector<float> cuda_output(out_size, 0.0f);

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

    // CUDA kernel
    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(cuda_ordinal_);
    auto attention_workspace = bindAttentionWorkspace(cuda_kernel, n_heads, head_dim);
    ASSERT_NE(attention_workspace, nullptr);

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, q_size * sizeof(float));
    cudaMalloc(&d_K, kv_size * sizeof(float));
    cudaMalloc(&d_V, kv_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    // Copy inputs to device
    cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, out_size * sizeof(float));

    // Execute CUDA kernel
    bool cuda_success = cuda_kernel.compute(
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
    cudaDeviceSynchronize();

    ASSERT_TRUE(cuda_success) << "CUDA attention failed";

    // Copy output back
    cudaMemcpy(cuda_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    // Validate
    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), out_size)) << "CUDA output contains NaN/Inf";
    ASSERT_FALSE(hasNaNOrInf(cpu_output.data(), out_size)) << "CPU output contains NaN/Inf";

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(cuda_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 FP32 Small", cosine, l2_error, max_error, out_size);

    // Attention has some numerical sensitivity, so we use looser thresholds
    EXPECT_GE(cosine, 0.99) << "Cosine similarity too low";
    EXPECT_LE(l2_error, 0.05) << "L2 error too high";
}

TEST_F(Test__CUDAFlashAttentionParity, FlashAttn2_FP32_Medium)
{
    SKIP_IF_NO_CUDA();

    // Medium test case (typical prefill scenario)
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
    std::vector<float> cuda_output(out_size, 0.0f);

    // CPU reference
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, -1);
    ASSERT_TRUE(cpu_success);

    // CUDA kernel
    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(cuda_ordinal_);
    auto attention_workspace = bindAttentionWorkspace(cuda_kernel, n_heads, head_dim);
    ASSERT_NE(attention_workspace, nullptr);

    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, q_size * sizeof(float));
    cudaMalloc(&d_K, kv_size * sizeof(float));
    cudaMalloc(&d_V, kv_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, out_size * sizeof(float));

    bool cuda_success = cuda_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    cudaDeviceSynchronize();

    ASSERT_TRUE(cuda_success);

    cudaMemcpy(cuda_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), out_size));

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(cuda_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 FP32 Medium (GQA)", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99);
    EXPECT_LE(l2_error, 0.05);
}

TEST_F(Test__CUDAFlashAttentionParity, FlashAttn2_FP32_Large)
{
    SKIP_IF_NO_CUDA();

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
    std::vector<float> cuda_output(out_size, 0.0f);

    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, -1);
    ASSERT_TRUE(cpu_success);

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(cuda_ordinal_);
    auto attention_workspace = bindAttentionWorkspace(cuda_kernel, n_heads, head_dim);
    ASSERT_NE(attention_workspace, nullptr);

    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, q_size * sizeof(float));
    cudaMalloc(&d_K, kv_size * sizeof(float));
    cudaMalloc(&d_V, kv_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, out_size * sizeof(float));

    bool cuda_success = cuda_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    cudaDeviceSynchronize();

    ASSERT_TRUE(cuda_success);

    cudaMemcpy(cuda_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), out_size));

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(cuda_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 FP32 Large", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99);
    EXPECT_LE(l2_error, 0.05);
}

TEST_F(Test__CUDAFlashAttentionParity, FlashAttn2_FP32_Qwen36MoEShortFullAttentionShape)
{
    SKIP_IF_NO_CUDA();

    /*
     * Qwen3.6 MoE reaches its first full-attention layer at layer 3.  Classic
     * parity showed Q/K/V and RoPE were already close, then the CUDA attention
     * context diverged.  This keeps that failure shape small enough to debug:
     * 9 prompt rows, GQA 16:2, and the 256-wide attention heads used by the
     * Qwen3.6 dense/MoE family.
     */
    constexpr int seq_len = 9;
    constexpr int n_heads = 16;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 256;
    const size_t q_size = static_cast<size_t>(seq_len) * n_heads * head_dim;
    const size_t kv_size = static_cast<size_t>(seq_len) * n_kv_heads * head_dim;
    const size_t out_size = q_size;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    for (float &value : Q_data)
    {
        value *= 4.2f;
    }
    for (float &value : K_data)
    {
        value *= 4.8f;
    }
    for (float &value : V_data)
    {
        value *= 1.8f;
    }
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> cuda_output(out_size, 0.0f);

    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    ASSERT_TRUE(cpu_kernel.compute(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, -1));

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(cuda_ordinal_);
    auto attention_workspace = bindAttentionWorkspace(cuda_kernel, n_heads, head_dim);
    ASSERT_NE(attention_workspace, nullptr);

    float *d_Q = nullptr;
    float *d_K = nullptr;
    float *d_V = nullptr;
    float *d_output = nullptr;
    ASSERT_EQ(cudaMalloc(&d_Q, q_size * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_K, kv_size * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_V, kv_size * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_output, out_size * sizeof(float)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_output, 0, out_size * sizeof(float)), cudaSuccess);

    ASSERT_TRUE(cuda_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, cuda_ordinal_));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(cuda_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), out_size));

    const double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), out_size);
    const double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), out_size);
    const double max_error = maxAbsError(cuda_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 FP32 Qwen3.6 MoE short full-attention shape",
                         cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99);
    EXPECT_LE(l2_error, 0.05);
}

TEST_F(Test__CUDAFlashAttentionParity, AttentionStagePrefillIgnoresDeviceResidentCachedCount)
{
    SKIP_IF_NO_CUDA();

    /*
     * Regression for Qwen3.6 MoE classic CUDA parity: regular prefill has no
     * previous KV history, so AttentionComputeStage must not derive FA2 params
     * from the device cached-token mirror.  The mirror can be stale or already
     * advanced by graph-owned decode state.  A deliberately wrong mirror below
     * would make the stage attend to the padded tail if prefill consumed it.
     */
    constexpr int seq_len = 9;
    constexpr int padded_kv_len = 18;
    constexpr int n_heads = 16;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 256;
    const size_t q_cols = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_cols = static_cast<size_t>(n_kv_heads) * head_dim;
    const size_t q_size = static_cast<size_t>(seq_len) * q_cols;
    const size_t kv_size = static_cast<size_t>(padded_kv_len) * kv_cols;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    for (size_t i = static_cast<size_t>(seq_len) * kv_cols; i < kv_size; ++i)
    {
        // Large, structured tail values make accidental attention to padded
        // rows obvious while keeping the allocation in-bounds.
        K_data[i] = 3.5f + 0.01f * static_cast<float>(i % 17);
        V_data[i] = -2.0f + 0.02f * static_cast<float>(i % 19);
    }

    std::vector<uint16_t> K_fp16(kv_size);
    std::vector<uint16_t> V_fp16(kv_size);
    for (size_t i = 0; i < kv_size; ++i)
    {
        K_fp16[i] = fp32_to_fp16(K_data[i]);
        V_fp16[i] = fp32_to_fp16(V_data[i]);
    }

    auto q_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), q_cols},
        DeviceId::cpu());
    auto k_tensor = std::make_unique<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(padded_kv_len), kv_cols},
        K_fp16);
    auto v_tensor = std::make_unique<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(padded_kv_len), kv_cols},
        V_fp16);
    auto stage_out_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), q_cols},
        DeviceId::cpu());
    auto direct_out_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), q_cols},
        DeviceId::cpu());
    std::copy(Q_data.begin(), Q_data.end(), q_tensor->mutable_data());

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    auto &transfer = TransferEngine::instance();
    ASSERT_TRUE(transfer.uploadFull(q_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(k_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(v_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(stage_out_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(direct_out_tensor.get(), gpu_device_, stream).success);

    int *d_wrong_count = nullptr;
    const int wrong_count = padded_kv_len;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_wrong_count), sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_wrong_count, &wrong_count, sizeof(int),
                              cudaMemcpyHostToDevice, stream),
              cudaSuccess);

    PrefillDeviceCountTrapKVCache trap_cache;
    trap_cache.host_cached_tokens = seq_len;
    trap_cache.device_cached_tokens = d_wrong_count;

    AttentionComputeStage::Params params;
    params.device_id = gpu_device_;
    params.Q = q_tensor.get();
    params.K = k_tensor.get();
    params.V = v_tensor.get();
    params.output = stage_out_tensor.get();
    params.batch_size = 1;
    params.seq_len = seq_len;
    params.kv_len = seq_len;
    params.n_heads = n_heads;
    params.n_kv_heads = n_kv_heads;
    params.head_dim = head_dim;
    params.causal = true;
    params.auto_detect_mode = true;
    params.kv_cache = &trap_cache;
    params.layer_idx = 0;
    params.mpi_ctx = &mpi_ctx_;

    AttentionComputeStage stage(params);
    stage.setGPUStream(stream);
    const WorkspaceRequirements stage_reqs =
        stage.getWorkspaceRequirements(/*m=*/1, /*n=*/n_heads, /*k=*/head_dim);
    DeviceWorkspaceManager stage_workspace(
        gpu_device_, stage_reqs.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(stage_workspace.allocate(stage_reqs));
    stage.bindWorkspace(&stage_workspace);
    ASSERT_TRUE(stage.execute(nullptr));

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> direct_kernel(cuda_ordinal_);
    direct_kernel.setGPUStream(stream);
    const WorkspaceRequirements direct_reqs =
        direct_kernel.getWorkspaceRequirements(1, n_heads, head_dim);
    DeviceWorkspaceManager direct_workspace(
        gpu_device_, direct_reqs.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(direct_workspace.allocate(direct_reqs));
    direct_kernel.bindWorkspace(&direct_workspace);
    ASSERT_TRUE(direct_kernel.compute_tensor(
        q_tensor.get(), k_tensor.get(), v_tensor.get(), direct_out_tensor.get(),
        1, seq_len, seq_len,
        n_heads, n_kv_heads, head_dim,
        true, -1,
        nullptr, nullptr,
        &mpi_ctx_, cuda_ordinal_,
        0, n_heads, n_kv_heads, n_heads / n_kv_heads));

    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<float> stage_out(q_size, 0.0f);
    std::vector<float> direct_out(q_size, 0.0f);
    ASSERT_EQ(cudaMemcpyAsync(stage_out.data(), stage_out_tensor->gpu_data_ptr(),
                              q_size * sizeof(float), cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(direct_out.data(), direct_out_tensor->gpu_data_ptr(),
                              q_size * sizeof(float), cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    const double cosine = cosineSimilarity(stage_out.data(), direct_out.data(), q_size);
    const double l2_error = relativeL2Error(stage_out.data(), direct_out.data(), q_size);
    const double max_error = maxAbsError(stage_out.data(), direct_out.data(), q_size);
    printComparisonStats("AttentionStage prefill ignores device count",
                         cosine, l2_error, max_error, q_size);
    EXPECT_GE(cosine, 0.999999)
        << "prefill attention must not consume stale device cached-token mirrors";
    EXPECT_LE(l2_error, 1e-5)
        << "prefill attention differs from direct FA2 when device count is stale";

    cudaFree(d_wrong_count);
    cudaStreamDestroy(stream);
}

TEST_F(Test__CUDAFlashAttentionParity, FlashAttn2_FP32_Qwen36MoELayer3RealSnapshot)
{
    SKIP_IF_NO_CUDA();

#if !LLAMINAR_CUDA_ATTENTION_PARITY_HAS_CNPY
    GTEST_SKIP() << "cnpy unavailable; Qwen3.6 MoE real-snapshot replay disabled";
#else
    const std::filesystem::path snapshot_dir =
        "pytorch_qwen36_moe_singledevice_cuda_snapshots";
    const std::filesystem::path q_path = snapshot_dir / "layer3_Q_ROPE.npy";
    const std::filesystem::path k_path = snapshot_dir / "layer3_K_ROPE.npy";
    const std::filesystem::path v_path = snapshot_dir / "layer3_V_PROJECTION.npy";
    const std::filesystem::path expected_path = snapshot_dir / "layer3_ATTENTION_CONTEXT.npy";
    if (!std::filesystem::exists(q_path) ||
        !std::filesystem::exists(k_path) ||
        !std::filesystem::exists(v_path) ||
        !std::filesystem::exists(expected_path))
    {
        GTEST_SKIP() << "Qwen3.6 MoE CUDA PyTorch snapshots are not available";
    }

    constexpr int seq_len = 9;
    constexpr int n_heads = 16;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 256;
    const size_t q_size = static_cast<size_t>(seq_len) * n_heads * head_dim;
    const size_t kv_size = static_cast<size_t>(seq_len) * n_kv_heads * head_dim;
    const size_t out_size = q_size;

    const auto Q_data = loadNpyFloatSnapshot(q_path);
    const auto K_data = loadNpyFloatSnapshot(k_path);
    const auto V_data = loadNpyFloatSnapshot(v_path);
    const auto expected_output = loadNpyFloatSnapshot(expected_path);
    ASSERT_EQ(Q_data.size(), q_size);
    ASSERT_EQ(K_data.size(), kv_size);
    ASSERT_EQ(V_data.size(), kv_size);
    ASSERT_EQ(expected_output.size(), out_size);

    std::vector<float> cuda_output(out_size, 0.0f);

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(cuda_ordinal_);
    auto attention_workspace = bindAttentionWorkspace(cuda_kernel, n_heads, head_dim);
    ASSERT_NE(attention_workspace, nullptr);

    float *d_Q = nullptr;
    float *d_K = nullptr;
    float *d_V = nullptr;
    float *d_output = nullptr;
    ASSERT_EQ(cudaMalloc(&d_Q, q_size * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_K, kv_size * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_V, kv_size * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_output, out_size * sizeof(float)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_output, 0, out_size * sizeof(float)), cudaSuccess);

    ASSERT_TRUE(cuda_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, cuda_ordinal_));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(cuda_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), out_size));

    const double cosine = cosineSimilarity(cuda_output.data(), expected_output.data(), out_size);
    const double l2_error = relativeL2Error(cuda_output.data(), expected_output.data(), out_size);
    const double max_error = maxAbsError(cuda_output.data(), expected_output.data(), out_size);

    printComparisonStats("FlashAttn2 FP32 Qwen3.6 MoE layer3 real snapshot",
                         cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.999);
    EXPECT_LE(l2_error, 0.01);
#endif
}

// ============================================================================
// Flash Decoding Parity Tests (with CPU reference)
// ============================================================================

TEST_F(Test__CUDAFlashAttentionParity, FlashDecode_FP32_Short_Parity)
{
    SKIP_IF_NO_CUDA();

    // Short KV cache - tests the fallback path (may use prefill kernel)
    constexpr int kv_len = 32;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 2; // GQA
    constexpr int head_dim = 64;
    const size_t q_size = n_heads * head_dim; // [n_heads, head_dim]
    const size_t kv_size = kv_len * n_kv_heads * head_dim;
    const size_t out_size = n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> cuda_output(out_size, 0.0f);

    // CPU reference using CPUFlashAttentionKernelT::compute_decode() - apples-to-apples comparison
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute_decode(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        1, // seq_len = 1 (single query token)
        kv_len,
        n_heads, n_kv_heads, head_dim,
        true, kv_len - 1); // causal, position_offset for decode
    ASSERT_TRUE(cpu_success) << "CPU compute_decode failed";

    // CUDA decode
    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(cuda_ordinal_);
    auto attention_workspace = bindAttentionWorkspace(cuda_kernel, n_heads, head_dim);
    ASSERT_NE(attention_workspace, nullptr);

    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, q_size * sizeof(float));
    cudaMalloc(&d_K, kv_size * sizeof(float));
    cudaMalloc(&d_V, kv_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, out_size * sizeof(float));

    bool cuda_success = cuda_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        1, // seq_len = 1 (single query token)
        kv_len,
        n_heads, n_kv_heads, head_dim,
        true, // causal
        0);   // position_offset
    cudaDeviceSynchronize();

    ASSERT_TRUE(cuda_success) << "CUDA Flash Decoding failed";

    cudaMemcpy(cuda_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    // Validate
    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), out_size)) << "CUDA output contains NaN/Inf";
    ASSERT_FALSE(hasNaNOrInf(cpu_output.data(), out_size)) << "CPU output contains NaN/Inf";

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(cuda_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashDecode FP32 Short Parity", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99) << "Cosine similarity too low - decode kernel may be incorrect";
    EXPECT_LE(l2_error, 0.05) << "L2 error too high";
}

TEST_F(Test__CUDAFlashAttentionParity, FlashDecode_FP16KV_Qwen35FullAttentionShortDecodeShape)
{
    SKIP_IF_NO_CUDA();

    /*
     * Qwen3.5 4B classic decode parity first diverged in layer 3 after Q/K/V
     * projection and RoPE snapshots were already close to PyTorch.  Layer 3 is
     * the first full-attention layer, so this isolates the CUDA FP16-KV decode
     * kernel at the exact shape used there: 16 query heads, 4 KV heads, 256-wide
     * heads, and a short 9-token prompt plus the current decode token.
     */
    constexpr int kv_len = 10;
    constexpr int n_heads = 16;
    constexpr int n_kv_heads = 4;
    constexpr int head_dim = 256;
    const size_t q_size = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_size = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;
    const size_t out_size = q_size;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    for (float &value : Q_data)
        value *= 2.0f;
    for (float &value : K_data)
        value *= 2.0f;

    std::vector<uint16_t> K_fp16(kv_size);
    std::vector<uint16_t> V_fp16(kv_size);
    std::vector<float> K_rounded(kv_size);
    std::vector<float> V_rounded(kv_size);
    for (size_t i = 0; i < kv_size; ++i)
    {
        K_fp16[i] = fp32_to_fp16(K_data[i]);
        V_fp16[i] = fp32_to_fp16(V_data[i]);
        K_rounded[i] = fp16_to_fp32(K_fp16[i]);
        V_rounded[i] = fp16_to_fp32(V_fp16[i]);
    }

    std::vector<float> cpu_output(out_size, 0.0f);
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    ASSERT_TRUE(cpu_kernel.compute_decode(
        Q_data.data(), K_rounded.data(), V_rounded.data(), cpu_output.data(),
        1, kv_len,
        n_heads, n_kv_heads, head_dim,
        true, kv_len - 1));

    auto q_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{1}, q_size},
        DeviceId::cpu());
    auto k_tensor = std::make_unique<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(kv_len),
                            static_cast<size_t>(n_kv_heads * head_dim)},
        K_fp16);
    auto v_tensor = std::make_unique<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(kv_len),
                            static_cast<size_t>(n_kv_heads * head_dim)},
        V_fp16);
    auto out_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{1}, q_size},
        DeviceId::cpu());
    std::copy(Q_data.begin(), Q_data.end(), q_tensor->mutable_data());

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    auto &transfer = TransferEngine::instance();
    ASSERT_TRUE(transfer.uploadFull(q_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(k_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(v_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(out_tensor.get(), gpu_device_, stream).success);

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(cuda_ordinal_);
    cuda_kernel.setGPUStream(stream);
    auto attention_workspace = bindAttentionWorkspace(cuda_kernel, n_heads, head_dim);
    ASSERT_NE(attention_workspace, nullptr);

    ASSERT_TRUE(cuda_kernel.prepareDynamicAttnParams(kv_len, kv_len - 1, 1, stream));
    ASSERT_TRUE(cuda_kernel.compute_tensor(
        q_tensor.get(), k_tensor.get(), v_tensor.get(), out_tensor.get(),
        1, 1, kv_len,
        n_heads, n_kv_heads, head_dim,
        true, -1,
        nullptr, nullptr,
        &mpi_ctx_, cuda_ordinal_,
        0, n_heads, n_kv_heads, n_heads / n_kv_heads));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<float> cuda_output(out_size, 0.0f);
    ASSERT_EQ(cudaMemcpyAsync(cuda_output.data(), out_tensor->gpu_data_ptr(),
                              out_size * sizeof(float), cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    cudaStreamDestroy(stream);

    const double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), out_size);
    const double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), out_size);
    const double max_error = maxAbsError(cuda_output.data(), cpu_output.data(), out_size);
    printComparisonStats("FlashDecode FP16KV Qwen3.5 full-attention short decode",
                         cosine, l2_error, max_error, out_size);
    EXPECT_GE(cosine, 0.999999) << "CUDA FP16-KV decode must match rounded-FP16 CPU attention";
    EXPECT_LE(l2_error, 1e-5) << "CUDA FP16-KV decode relative L2 too high";
}

TEST_F(Test__CUDAFlashAttentionParity, AttentionStageAppendHandoff_FP16KV_Qwen35FullAttentionShortDecodeShape)
{
    SKIP_IF_NO_CUDA();

    /**
     * Qwen3.5/Qwen3.6 dense decode appends the current K/V row before
     * AttentionComputeStage consumes the live KV cache.  The attention stage
     * derives its KV length from the cache's device-owned sequence counter, so
     * this regression proves the append stage, device sequence metadata, and
     * attention stage agree without consulting a host-side fallback.
     */
    constexpr int history_len = 9;
    constexpr int seq_len = 1;
    constexpr int kv_len = history_len + seq_len;
    constexpr int n_heads = 16;
    constexpr int n_kv_heads = 4;
    constexpr int head_dim = 256;
    const size_t q_size = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_cols = static_cast<size_t>(n_kv_heads) * head_dim;
    const size_t history_size = static_cast<size_t>(history_len) * kv_cols;
    const size_t current_size = static_cast<size_t>(seq_len) * kv_cols;
    const size_t kv_size = static_cast<size_t>(kv_len) * kv_cols;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    for (float &value : Q_data)
        value *= 2.0f;
    for (float &value : K_data)
        value *= 2.0f;

    std::vector<uint16_t> K_fp16(kv_size);
    std::vector<uint16_t> V_fp16(kv_size);
    std::vector<float> K_rounded(kv_size);
    std::vector<float> V_rounded(kv_size);
    for (size_t i = 0; i < kv_size; ++i)
    {
        K_fp16[i] = fp32_to_fp16(K_data[i]);
        V_fp16[i] = fp32_to_fp16(V_data[i]);
        K_rounded[i] = fp16_to_fp32(K_fp16[i]);
        V_rounded[i] = fp16_to_fp32(V_fp16[i]);
    }

    std::vector<float> cpu_output(q_size, 0.0f);
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    ASSERT_TRUE(cpu_kernel.compute_decode(
        Q_data.data(), K_rounded.data(), V_rounded.data(), cpu_output.data(),
        1, kv_len,
        n_heads, n_kv_heads, head_dim,
        true, kv_len - 1));

    auto q_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{1}, q_size},
        DeviceId::cpu());
    auto history_k_tensor = std::make_unique<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(history_len), kv_cols},
        std::vector<uint16_t>(K_fp16.begin(),
                              K_fp16.begin() + static_cast<std::ptrdiff_t>(history_size)));
    auto history_v_tensor = std::make_unique<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(history_len), kv_cols},
        std::vector<uint16_t>(V_fp16.begin(),
                              V_fp16.begin() + static_cast<std::ptrdiff_t>(history_size)));
    auto current_k_tensor = std::make_unique<FP16Tensor>(
        std::vector<size_t>{size_t{1}, kv_cols},
        std::vector<uint16_t>(K_fp16.begin() + static_cast<std::ptrdiff_t>(history_size),
                              K_fp16.begin() + static_cast<std::ptrdiff_t>(history_size + current_size)));
    auto current_v_tensor = std::make_unique<FP16Tensor>(
        std::vector<size_t>{size_t{1}, kv_cols},
        std::vector<uint16_t>(V_fp16.begin() + static_cast<std::ptrdiff_t>(history_size),
                              V_fp16.begin() + static_cast<std::ptrdiff_t>(history_size + current_size)));
    auto out_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{1}, q_size},
        DeviceId::cpu());
    std::copy(Q_data.begin(), Q_data.end(), q_tensor->mutable_data());

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    auto &transfer = TransferEngine::instance();
    ASSERT_TRUE(transfer.uploadFull(q_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(history_k_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(history_v_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(current_k_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(current_v_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(out_tensor.get(), gpu_device_, stream).success);

    llaminar::v2::kernels::KVCacheConfig config;
    config.precision = ActivationPrecision::FP16;
    config.device = gpu_device_;
    config.num_layers = 1;
    config.batch_size = 1;
    config.max_seq_len = kv_len + 8;
    config.n_kv_heads = n_kv_heads;
    config.head_dim = head_dim;
    auto kv_cache = llaminar::v2::kernels::KernelFactory::createKVCache(config);
    ASSERT_NE(kv_cache, nullptr);
    ASSERT_TRUE(kv_cache->appendWithStream(
        0, 0, history_k_tensor.get(), history_v_tensor.get(), history_len, stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(kv_cache->get_cached_tokens(0, 0), history_len);
    ASSERT_NE(kv_cache->deviceCachedTokenCountPtr(0, 0), nullptr);

    KVCacheAppendStage::Params append_params;
    append_params.device_id = gpu_device_;
    append_params.K = current_k_tensor.get();
    append_params.V = current_v_tensor.get();
    append_params.kv_cache = kv_cache.get();
    append_params.layer_idx = 0;
    append_params.seq_idx = 0;
    append_params.num_tokens = seq_len;
    append_params.batch_size = 1;
    append_params.seq_len = seq_len;
    append_params.head_dim = head_dim;

    AttentionComputeStage::Params attn_params;
    attn_params.device_id = gpu_device_;
    attn_params.Q = q_tensor.get();
    attn_params.K = current_k_tensor.get();
    attn_params.V = current_v_tensor.get();
    attn_params.output = out_tensor.get();
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
    attn_params.apply_rope_to_k = false;
    attn_params.mpi_ctx = &mpi_ctx_;

    KVCacheAppendStage append_stage(append_params);
    AttentionComputeStage attn_stage(attn_params);
    append_stage.setGPUStream(stream);
    attn_stage.setGPUStream(stream);

    const WorkspaceRequirements attn_reqs =
        attn_stage.getWorkspaceRequirements(/*m=*/1, /*n=*/n_heads, /*k=*/head_dim);
    DeviceWorkspaceManager attn_workspace(
        gpu_device_, attn_reqs.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(attn_workspace.allocate(attn_reqs));
    attn_stage.bindWorkspace(&attn_workspace);

    append_stage.updateDynamicParams(/*pos_offset=*/history_len, seq_len);
    attn_stage.updateDynamicParams(/*pos_offset=*/history_len, seq_len);
    ASSERT_TRUE(append_stage.execute(nullptr));
    ASSERT_TRUE(attn_stage.execute(nullptr));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(kv_cache->get_cached_tokens(0, 0), kv_len);

    int device_count = 0;
    ASSERT_EQ(cudaMemcpyAsync(&device_count, kv_cache->deviceCachedTokenCountPtr(0, 0),
                              sizeof(int), cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(device_count, kv_len)
        << "device-owned KV sequence count must match the post-append cache length";

    std::vector<float> cuda_output(q_size, 0.0f);
    ASSERT_EQ(cudaMemcpyAsync(cuda_output.data(), out_tensor->gpu_data_ptr(),
                              q_size * sizeof(float), cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    cudaStreamDestroy(stream);

    const double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), q_size);
    const double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), q_size);
    const double max_error = maxAbsError(cuda_output.data(), cpu_output.data(), q_size);
    printComparisonStats("AttentionStage append handoff FP16KV Qwen3.5 full-attention short decode",
                         cosine, l2_error, max_error, q_size);
    EXPECT_GE(cosine, 0.999999)
        << "append-to-attention handoff must match rounded-FP16 CPU attention";
    EXPECT_LE(l2_error, 1e-5)
        << "append-to-attention handoff relative L2 too high";
}

TEST_F(Test__CUDAFlashAttentionParity, AttentionStageAppendHandoff_ConvertsGpuFP32IntoFP16Cache_Qwen35FullAttentionShortDecodeShape)
{
    SKIP_IF_NO_CUDA();

    /**
     * Production dense decode computes K/V projection rows as FP32 device
     * tensors, then appends them into the configured KV cache precision.  This
     * regression keeps the source tensors GPU-resident FP32 and verifies that
     * the CUDA append path performs device-side FP32->FP16 conversion before
     * AttentionComputeStage reads the cache.
     */
    constexpr int history_len = 9;
    constexpr int seq_len = 1;
    constexpr int kv_len = history_len + seq_len;
    constexpr int n_heads = 16;
    constexpr int n_kv_heads = 4;
    constexpr int head_dim = 256;
    const size_t q_size = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_cols = static_cast<size_t>(n_kv_heads) * head_dim;
    const size_t history_size = static_cast<size_t>(history_len) * kv_cols;
    const size_t current_size = static_cast<size_t>(seq_len) * kv_cols;
    const size_t kv_size = static_cast<size_t>(kv_len) * kv_cols;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    for (float &value : Q_data)
        value *= 2.0f;
    for (float &value : K_data)
        value *= 2.0f;

    std::vector<float> K_rounded(kv_size);
    std::vector<float> V_rounded(kv_size);
    for (size_t i = 0; i < kv_size; ++i)
    {
        K_rounded[i] = fp16_to_fp32(fp32_to_fp16(K_data[i]));
        V_rounded[i] = fp16_to_fp32(fp32_to_fp16(V_data[i]));
    }

    std::vector<float> cpu_output(q_size, 0.0f);
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    ASSERT_TRUE(cpu_kernel.compute_decode(
        Q_data.data(), K_rounded.data(), V_rounded.data(), cpu_output.data(),
        1, kv_len,
        n_heads, n_kv_heads, head_dim,
        true, kv_len - 1));

    auto q_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{1}, q_size},
        DeviceId::cpu());
    auto history_k_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(history_len), kv_cols},
        DeviceId::cpu());
    auto history_v_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(history_len), kv_cols},
        DeviceId::cpu());
    auto current_k_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{1}, kv_cols},
        DeviceId::cpu());
    auto current_v_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{1}, kv_cols},
        DeviceId::cpu());
    auto out_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{1}, q_size},
        DeviceId::cpu());

    std::copy(Q_data.begin(), Q_data.end(), q_tensor->mutable_data());
    std::copy(K_data.begin(), K_data.begin() + static_cast<std::ptrdiff_t>(history_size),
              history_k_tensor->mutable_data());
    std::copy(V_data.begin(), V_data.begin() + static_cast<std::ptrdiff_t>(history_size),
              history_v_tensor->mutable_data());
    std::copy(K_data.begin() + static_cast<std::ptrdiff_t>(history_size),
              K_data.begin() + static_cast<std::ptrdiff_t>(history_size + current_size),
              current_k_tensor->mutable_data());
    std::copy(V_data.begin() + static_cast<std::ptrdiff_t>(history_size),
              V_data.begin() + static_cast<std::ptrdiff_t>(history_size + current_size),
              current_v_tensor->mutable_data());

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    auto &transfer = TransferEngine::instance();
    ASSERT_TRUE(transfer.uploadFull(q_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(history_k_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(history_v_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(current_k_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(current_v_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(out_tensor.get(), gpu_device_, stream).success);

    llaminar::v2::kernels::KVCacheConfig config;
    config.precision = ActivationPrecision::FP16;
    config.device = gpu_device_;
    config.num_layers = 1;
    config.batch_size = 1;
    config.max_seq_len = kv_len + 8;
    config.n_kv_heads = n_kv_heads;
    config.head_dim = head_dim;
    auto kv_cache = llaminar::v2::kernels::KernelFactory::createKVCache(config);
    ASSERT_NE(kv_cache, nullptr);
    ASSERT_TRUE(kv_cache->appendWithStream(
        0, 0, history_k_tensor.get(), history_v_tensor.get(), history_len, stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(kv_cache->get_cached_tokens(0, 0), history_len);

    KVCacheAppendStage::Params append_params;
    append_params.device_id = gpu_device_;
    append_params.K = current_k_tensor.get();
    append_params.V = current_v_tensor.get();
    append_params.kv_cache = kv_cache.get();
    append_params.layer_idx = 0;
    append_params.seq_idx = 0;
    append_params.num_tokens = seq_len;
    append_params.batch_size = 1;
    append_params.seq_len = seq_len;
    append_params.head_dim = head_dim;

    AttentionComputeStage::Params attn_params;
    attn_params.device_id = gpu_device_;
    attn_params.Q = q_tensor.get();
    attn_params.K = current_k_tensor.get();
    attn_params.V = current_v_tensor.get();
    attn_params.output = out_tensor.get();
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
    attn_params.apply_rope_to_k = false;
    attn_params.mpi_ctx = &mpi_ctx_;

    KVCacheAppendStage append_stage(append_params);
    AttentionComputeStage attn_stage(attn_params);
    append_stage.setGPUStream(stream);
    attn_stage.setGPUStream(stream);

    const WorkspaceRequirements attn_reqs =
        attn_stage.getWorkspaceRequirements(/*m=*/1, /*n=*/n_heads, /*k=*/head_dim);
    DeviceWorkspaceManager attn_workspace(
        gpu_device_, attn_reqs.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(attn_workspace.allocate(attn_reqs));
    attn_stage.bindWorkspace(&attn_workspace);

    append_stage.updateDynamicParams(/*pos_offset=*/history_len, seq_len);
    attn_stage.updateDynamicParams(/*pos_offset=*/history_len, seq_len);
    ASSERT_TRUE(append_stage.execute(nullptr));
    ASSERT_TRUE(attn_stage.execute(nullptr));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(kv_cache->get_cached_tokens(0, 0), kv_len);

    std::vector<float> cuda_output(q_size, 0.0f);
    ASSERT_EQ(cudaMemcpyAsync(cuda_output.data(), out_tensor->gpu_data_ptr(),
                              q_size * sizeof(float), cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    cudaStreamDestroy(stream);

    const double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), q_size);
    const double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), q_size);
    const double max_error = maxAbsError(cuda_output.data(), cpu_output.data(), q_size);
    printComparisonStats("AttentionStage FP32 append into FP16 cache Qwen3.5 full-attention short decode",
                         cosine, l2_error, max_error, q_size);
    EXPECT_GE(cosine, 0.999999)
        << "GPU-resident FP32 K/V must be converted before entering the FP16 cache";
    EXPECT_LE(l2_error, 1e-5)
        << "FP32 append into FP16 cache relative L2 too high";
}

TEST_F(Test__CUDAFlashAttentionParity, AttentionStageAppendHandoff_RealQwen35Layer3FP32IntoFP16Cache)
{
    SKIP_IF_NO_CUDA();

#if !LLAMINAR_CUDA_ATTENTION_PARITY_HAS_CNPY
    GTEST_SKIP() << "cnpy unavailable; Qwen3.5 real-snapshot replay disabled";
#else
    /**
     * This is the kernel-sized reproducer for Qwen3.5-4B decode parity.  The
     * first production divergence appears at layer 3, the first full-attention
     * layer, even though Q/K/V projection and RoPE snapshots are already close
     * to PyTorch.  We replay the exact PyTorch Q/K/V tensors through the same
     * append-stage -> device-owned cache-count -> attention-stage handoff used
     * by decode, with FP32 source tensors and an FP16 KV cache.
     */
    const std::filesystem::path snapshot_dir = "pytorch_qwen35_4b_snapshots";
    const std::filesystem::path history_k_path = snapshot_dir / "layer3_K_ROPE.npy";
    const std::filesystem::path history_v_path = snapshot_dir / "layer3_V_PROJECTION.npy";
    const std::filesystem::path current_q_path = snapshot_dir / "decode_step0_layer3_Q_ROPE.npy";
    const std::filesystem::path current_k_path = snapshot_dir / "decode_step0_layer3_K_ROPE.npy";
    const std::filesystem::path current_v_path = snapshot_dir / "decode_step0_layer3_V_PROJECTION.npy";
    const std::filesystem::path expected_path = snapshot_dir / "decode_step0_layer3_ATTENTION_CONTEXT.npy";
    if (!std::filesystem::exists(history_k_path) ||
        !std::filesystem::exists(history_v_path) ||
        !std::filesystem::exists(current_q_path) ||
        !std::filesystem::exists(current_k_path) ||
        !std::filesystem::exists(current_v_path) ||
        !std::filesystem::exists(expected_path))
    {
        GTEST_SKIP() << "Qwen3.5 4B PyTorch snapshots are not available";
    }

    constexpr int history_len = 9;
    constexpr int seq_len = 1;
    constexpr int kv_len = history_len + seq_len;
    constexpr int n_heads = 16;
    constexpr int n_kv_heads = 4;
    constexpr int head_dim = 256;
    const size_t q_size = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_cols = static_cast<size_t>(n_kv_heads) * head_dim;
    const size_t history_size = static_cast<size_t>(history_len) * kv_cols;
    const size_t current_size = static_cast<size_t>(seq_len) * kv_cols;

    const auto Q_data = loadNpyFloatSnapshot(current_q_path);
    const auto history_K_data = loadNpyFloatSnapshot(history_k_path);
    const auto history_V_data = loadNpyFloatSnapshot(history_v_path);
    const auto current_K_data = loadNpyFloatSnapshot(current_k_path);
    const auto current_V_data = loadNpyFloatSnapshot(current_v_path);
    const auto expected_output = loadNpyFloatSnapshot(expected_path);
    ASSERT_EQ(Q_data.size(), q_size);
    ASSERT_EQ(history_K_data.size(), history_size);
    ASSERT_EQ(history_V_data.size(), history_size);
    ASSERT_EQ(current_K_data.size(), current_size);
    ASSERT_EQ(current_V_data.size(), current_size);
    ASSERT_EQ(expected_output.size(), q_size);

    auto q_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{1}, q_size},
        DeviceId::cpu());
    auto history_k_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(history_len), kv_cols},
        DeviceId::cpu());
    auto history_v_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(history_len), kv_cols},
        DeviceId::cpu());
    auto current_k_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{1}, kv_cols},
        DeviceId::cpu());
    auto current_v_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{1}, kv_cols},
        DeviceId::cpu());
    auto out_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{1}, q_size},
        DeviceId::cpu());

    std::copy(Q_data.begin(), Q_data.end(), q_tensor->mutable_data());
    std::copy(history_K_data.begin(), history_K_data.end(), history_k_tensor->mutable_data());
    std::copy(history_V_data.begin(), history_V_data.end(), history_v_tensor->mutable_data());
    std::copy(current_K_data.begin(), current_K_data.end(), current_k_tensor->mutable_data());
    std::copy(current_V_data.begin(), current_V_data.end(), current_v_tensor->mutable_data());

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    auto &transfer = TransferEngine::instance();
    ASSERT_TRUE(transfer.uploadFull(q_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(history_k_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(history_v_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(current_k_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(current_v_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(out_tensor.get(), gpu_device_, stream).success);

    llaminar::v2::kernels::KVCacheConfig config;
    config.precision = ActivationPrecision::FP16;
    config.device = gpu_device_;
    config.num_layers = 1;
    config.batch_size = 1;
    config.max_seq_len = kv_len + 8;
    config.n_kv_heads = n_kv_heads;
    config.head_dim = head_dim;
    auto kv_cache = llaminar::v2::kernels::KernelFactory::createKVCache(config);
    ASSERT_NE(kv_cache, nullptr);
    ASSERT_TRUE(kv_cache->appendWithStream(
        0, 0, history_k_tensor.get(), history_v_tensor.get(), history_len, stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(kv_cache->get_cached_tokens(0, 0), history_len);

    KVCacheAppendStage::Params append_params;
    append_params.device_id = gpu_device_;
    append_params.K = current_k_tensor.get();
    append_params.V = current_v_tensor.get();
    append_params.kv_cache = kv_cache.get();
    append_params.layer_idx = 0;
    append_params.seq_idx = 0;
    append_params.num_tokens = seq_len;
    append_params.batch_size = 1;
    append_params.seq_len = seq_len;
    append_params.head_dim = head_dim;

    AttentionComputeStage::Params attn_params;
    attn_params.device_id = gpu_device_;
    attn_params.Q = q_tensor.get();
    attn_params.K = current_k_tensor.get();
    attn_params.V = current_v_tensor.get();
    attn_params.output = out_tensor.get();
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
    attn_params.apply_rope_to_k = false;
    attn_params.mpi_ctx = &mpi_ctx_;

    KVCacheAppendStage append_stage(append_params);
    AttentionComputeStage attn_stage(attn_params);
    append_stage.setGPUStream(stream);
    attn_stage.setGPUStream(stream);

    const WorkspaceRequirements attn_reqs =
        attn_stage.getWorkspaceRequirements(/*m=*/1, /*n=*/n_heads, /*k=*/head_dim);
    DeviceWorkspaceManager attn_workspace(
        gpu_device_, attn_reqs.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(attn_workspace.allocate(attn_reqs));
    attn_stage.bindWorkspace(&attn_workspace);

    append_stage.updateDynamicParams(/*pos_offset=*/history_len, seq_len);
    attn_stage.updateDynamicParams(/*pos_offset=*/history_len, seq_len);
    ASSERT_TRUE(append_stage.execute(nullptr));
    ASSERT_TRUE(attn_stage.execute(nullptr));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(kv_cache->get_cached_tokens(0, 0), kv_len);

    std::vector<float> cuda_output(q_size, 0.0f);
    ASSERT_EQ(cudaMemcpyAsync(cuda_output.data(), out_tensor->gpu_data_ptr(),
                              q_size * sizeof(float), cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    cudaStreamDestroy(stream);

    const double cosine = cosineSimilarity(cuda_output.data(), expected_output.data(), q_size);
    const double l2_error = relativeL2Error(cuda_output.data(), expected_output.data(), q_size);
    const double max_error = maxAbsError(cuda_output.data(), expected_output.data(), q_size);
    printComparisonStats("AttentionStage real Qwen3.5 layer3 FP32 append into FP16 cache",
                         cosine, l2_error, max_error, q_size);
    EXPECT_GE(cosine, 0.999)
        << "real layer-3 decode attention should match PyTorch after FP16 KV rounding";
    EXPECT_LE(l2_error, 0.01)
        << "real layer-3 decode attention relative L2 too high";
#endif
}

TEST_F(Test__CUDAFlashAttentionParity, AttentionStageRoPEOnRead_RealQwen35Layer3FP32IntoFP16Cache)
{
    SKIP_IF_NO_CUDA();

#if !LLAMINAR_CUDA_ATTENTION_PARITY_HAS_CNPY
    GTEST_SKIP() << "cnpy unavailable; Qwen3.5 real-snapshot replay disabled";
#else
    /**
     * @brief Replays the production Qwen3.5 decode attention contract.
     *
     * The rotated-cache test above proves the CUDA decode kernel and the
     * append-to-attention handoff.  The real graph stores normalized but
     * unrotated K rows in the live cache, then AttentionComputeStage applies
     * RoPE while materializing the FP16 attention view.  That extra
     * RoPE-on-read step is a separate coherence boundary, so keep a dedicated
     * real-snapshot regression for it.
     */
    const std::filesystem::path snapshot_dir = "pytorch_qwen35_4b_snapshots";
    const std::filesystem::path history_k_path = snapshot_dir / "layer3_K_NORM.npy";
    const std::filesystem::path history_v_path = snapshot_dir / "layer3_V_PROJECTION.npy";
    const std::filesystem::path current_q_path = snapshot_dir / "decode_step0_layer3_Q_ROPE.npy";
    const std::filesystem::path current_k_path = snapshot_dir / "decode_step0_layer3_K_NORM.npy";
    const std::filesystem::path current_v_path = snapshot_dir / "decode_step0_layer3_V_PROJECTION.npy";
    const std::filesystem::path expected_path = snapshot_dir / "decode_step0_layer3_ATTENTION_CONTEXT.npy";
    if (!std::filesystem::exists(history_k_path) ||
        !std::filesystem::exists(history_v_path) ||
        !std::filesystem::exists(current_q_path) ||
        !std::filesystem::exists(current_k_path) ||
        !std::filesystem::exists(current_v_path) ||
        !std::filesystem::exists(expected_path))
    {
        GTEST_SKIP() << "Qwen3.5 4B PyTorch snapshots are not available";
    }

    constexpr int history_len = 9;
    constexpr int seq_len = 1;
    constexpr int kv_len = history_len + seq_len;
    constexpr int n_heads = 16;
    constexpr int n_kv_heads = 4;
    constexpr int head_dim = 256;
    constexpr float rope_theta = 10000000.0f;
    constexpr float partial_rotary_factor = 0.25f;
    const size_t q_size = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_cols = static_cast<size_t>(n_kv_heads) * head_dim;
    const size_t history_size = static_cast<size_t>(history_len) * kv_cols;
    const size_t current_size = static_cast<size_t>(seq_len) * kv_cols;

    const auto Q_data = loadNpyFloatSnapshot(current_q_path);
    const auto history_K_data = loadNpyFloatSnapshot(history_k_path);
    const auto history_V_data = loadNpyFloatSnapshot(history_v_path);
    const auto current_K_data = loadNpyFloatSnapshot(current_k_path);
    const auto current_V_data = loadNpyFloatSnapshot(current_v_path);
    const auto expected_output = loadNpyFloatSnapshot(expected_path);
    ASSERT_EQ(Q_data.size(), q_size);
    ASSERT_EQ(history_K_data.size(), history_size);
    ASSERT_EQ(history_V_data.size(), history_size);
    ASSERT_EQ(current_K_data.size(), current_size);
    ASSERT_EQ(current_V_data.size(), current_size);
    ASSERT_EQ(expected_output.size(), q_size);

    auto q_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{1}, q_size},
        DeviceId::cpu());
    auto history_k_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(history_len), kv_cols},
        DeviceId::cpu());
    auto history_v_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(history_len), kv_cols},
        DeviceId::cpu());
    auto current_k_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{1}, kv_cols},
        DeviceId::cpu());
    auto current_v_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{1}, kv_cols},
        DeviceId::cpu());
    auto out_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{size_t{1}, q_size},
        DeviceId::cpu());

    std::copy(Q_data.begin(), Q_data.end(), q_tensor->mutable_data());
    std::copy(history_K_data.begin(), history_K_data.end(), history_k_tensor->mutable_data());
    std::copy(history_V_data.begin(), history_V_data.end(), history_v_tensor->mutable_data());
    std::copy(current_K_data.begin(), current_K_data.end(), current_k_tensor->mutable_data());
    std::copy(current_V_data.begin(), current_V_data.end(), current_v_tensor->mutable_data());

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    auto &transfer = TransferEngine::instance();
    ASSERT_TRUE(transfer.uploadFull(q_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(history_k_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(history_v_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(current_k_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(current_v_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(out_tensor.get(), gpu_device_, stream).success);

    llaminar::v2::kernels::KVCacheConfig config;
    config.precision = ActivationPrecision::FP16;
    config.device = gpu_device_;
    config.num_layers = 1;
    config.batch_size = 1;
    config.max_seq_len = kv_len + 8;
    config.n_kv_heads = n_kv_heads;
    config.head_dim = head_dim;
    auto kv_cache = llaminar::v2::kernels::KernelFactory::createKVCache(config);
    ASSERT_NE(kv_cache, nullptr);
    ASSERT_TRUE(kv_cache->appendWithStream(
        0, 0, history_k_tensor.get(), history_v_tensor.get(), history_len, stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(kv_cache->get_cached_tokens(0, 0), history_len);

    KVCacheAppendStage::Params append_params;
    append_params.device_id = gpu_device_;
    append_params.K = current_k_tensor.get();
    append_params.V = current_v_tensor.get();
    append_params.kv_cache = kv_cache.get();
    append_params.layer_idx = 0;
    append_params.seq_idx = 0;
    append_params.num_tokens = seq_len;
    append_params.batch_size = 1;
    append_params.seq_len = seq_len;
    append_params.head_dim = head_dim;

    AttentionComputeStage::Params attn_params;
    attn_params.device_id = gpu_device_;
    attn_params.Q = q_tensor.get();
    attn_params.K = current_k_tensor.get();
    attn_params.V = current_v_tensor.get();
    attn_params.output = out_tensor.get();
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
    attn_params.mpi_ctx = &mpi_ctx_;

    KVCacheAppendStage append_stage(append_params);
    AttentionComputeStage attn_stage(attn_params);
    append_stage.setGPUStream(stream);
    attn_stage.setGPUStream(stream);

    const WorkspaceRequirements attn_reqs =
        attn_stage.getWorkspaceRequirements(/*m=*/1, /*n=*/n_heads, /*k=*/head_dim);
    DeviceWorkspaceManager attn_workspace(
        gpu_device_, attn_reqs.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(attn_workspace.allocate(attn_reqs));
    attn_stage.bindWorkspace(&attn_workspace);

    append_stage.updateDynamicParams(/*pos_offset=*/history_len, seq_len);
    attn_stage.updateDynamicParams(/*pos_offset=*/history_len, seq_len);
    ASSERT_TRUE(append_stage.execute(nullptr));
    ASSERT_TRUE(attn_stage.execute(nullptr));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(kv_cache->get_cached_tokens(0, 0), kv_len);

    std::vector<float> cuda_output(q_size, 0.0f);
    ASSERT_EQ(cudaMemcpyAsync(cuda_output.data(), out_tensor->gpu_data_ptr(),
                              q_size * sizeof(float), cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    cudaStreamDestroy(stream);

    const double cosine = cosineSimilarity(cuda_output.data(), expected_output.data(), q_size);
    const double l2_error = relativeL2Error(cuda_output.data(), expected_output.data(), q_size);
    const double max_error = maxAbsError(cuda_output.data(), expected_output.data(), q_size);
    printComparisonStats("AttentionStage real Qwen3.5 layer3 RoPE-on-read FP16 cache",
                         cosine, l2_error, max_error, q_size);
    EXPECT_GE(cosine, 0.999)
        << "real layer-3 decode attention should match PyTorch through RoPE-on-read";
    EXPECT_LE(l2_error, 0.01)
        << "real layer-3 RoPE-on-read attention relative L2 too high";
#endif
}

TEST_F(Test__CUDAFlashAttentionParity, FlashDecode_FP32_Long_Parity)
{
    SKIP_IF_NO_CUDA();

    // Longer KV cache - this exercises the split-K Flash Decoding path
    constexpr int kv_len = 512;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    const size_t q_size = n_heads * head_dim;
    const size_t kv_size = kv_len * n_kv_heads * head_dim;
    const size_t out_size = n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> cuda_output(out_size, 0.0f);

    // CPU reference using CPUFlashAttentionKernelT::compute_decode() - apples-to-apples comparison
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute_decode(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        1, kv_len, n_heads, n_kv_heads, head_dim,
        true, kv_len - 1); // causal, position_offset for decode
    ASSERT_TRUE(cpu_success) << "CPU compute_decode failed";

    // CUDA decode
    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(cuda_ordinal_);
    auto attention_workspace = bindAttentionWorkspace(cuda_kernel, n_heads, head_dim);
    ASSERT_NE(attention_workspace, nullptr);

    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, q_size * sizeof(float));
    cudaMalloc(&d_K, kv_size * sizeof(float));
    cudaMalloc(&d_V, kv_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, out_size * sizeof(float));

    bool cuda_success = cuda_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        1, kv_len, n_heads, n_kv_heads, head_dim,
        true); // causal
    cudaDeviceSynchronize();

    ASSERT_TRUE(cuda_success);

    cudaMemcpy(cuda_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), out_size));

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(cuda_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashDecode FP32 Long Parity (split-K) vs CPUFlashAttentionKernelT", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99) << "Cosine similarity too low - split-K reduction may be incorrect";
    EXPECT_LE(l2_error, 0.05) << "L2 error too high";
}

TEST_F(Test__CUDAFlashAttentionParity, FlashDecode_Q81KVCacheConsumption_Parity)
{
    SKIP_IF_NO_CUDA();

    constexpr int kv_len = 128;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;

    const size_t q_size = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_size = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;
    const size_t out_size = static_cast<size_t>(n_heads) * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data_fp32 = randomFP32(kv_size);
    auto V_data_fp32 = randomFP32(kv_size);

    std::vector<float> cpu_baseline_output(out_size, 0.0f);
    std::vector<float> cpu_q81_output(out_size, 0.0f);
    std::vector<float> cuda_q81_output(out_size, 0.0f);

    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;

    ASSERT_TRUE(cpu_kernel.compute_decode(
        Q_data.data(), K_data_fp32.data(), V_data_fp32.data(), cpu_baseline_output.data(),
        1, kv_len, n_heads, n_kv_heads, head_dim,
        true, kv_len - 1));

    MPIContext local_mpi_ctx(0, 1, MPI_COMM_WORLD);
    auto kv_cache = std::make_unique<CPURingKVCache<ActivationPrecision::Q8_1>>(
        local_mpi_ctx,
        1,
        1,
        kv_len,
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

    ASSERT_TRUE(cpu_kernel.compute_decode(
        Q_data.data(), K_from_q81, V_from_q81, cpu_q81_output.data(),
        1, kv_len, n_heads, n_kv_heads, head_dim,
        true, kv_len - 1));

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(cuda_ordinal_);
    auto attention_workspace = bindAttentionWorkspace(cuda_kernel, n_heads, head_dim);
    ASSERT_NE(attention_workspace, nullptr);

    float *d_Q = nullptr;
    float *d_K = nullptr;
    float *d_V = nullptr;
    float *d_output = nullptr;
    cudaMalloc(&d_Q, q_size * sizeof(float));
    cudaMalloc(&d_K, kv_size * sizeof(float));
    cudaMalloc(&d_V, kv_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_from_q81, kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_from_q81, kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, out_size * sizeof(float));

    bool cuda_success = cuda_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        1, kv_len, n_heads, n_kv_heads, head_dim,
        true,
        0);
    cudaDeviceSynchronize();
    ASSERT_TRUE(cuda_success);

    cudaMemcpy(cuda_q81_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(cpu_q81_output.data(), out_size));
    ASSERT_FALSE(hasNaNOrInf(cuda_q81_output.data(), out_size));

    const double q81_cuda_cpu_cos = cosineSimilarity(cuda_q81_output.data(), cpu_q81_output.data(), out_size);
    const double q81_cuda_cpu_l2 = relativeL2Error(cuda_q81_output.data(), cpu_q81_output.data(), out_size);

    const double q81_vs_fp32_cos = cosineSimilarity(cpu_q81_output.data(), cpu_baseline_output.data(), out_size);
    const double q81_vs_fp32_l2 = relativeL2Error(cpu_q81_output.data(), cpu_baseline_output.data(), out_size);

    printComparisonStats("FlashDecode Q8_1-consumed CUDA vs CPU", q81_cuda_cpu_cos, q81_cuda_cpu_l2,
                         maxAbsError(cuda_q81_output.data(), cpu_q81_output.data(), out_size), out_size);
    printComparisonStats("FlashDecode Q8_1-consumed CPU vs FP32 baseline", q81_vs_fp32_cos, q81_vs_fp32_l2,
                         maxAbsError(cpu_q81_output.data(), cpu_baseline_output.data(), out_size), out_size);

    EXPECT_GE(q81_cuda_cpu_cos, 0.99) << "CUDA vs CPU parity too low for Q8_1-consumed path";
    EXPECT_LE(q81_cuda_cpu_l2, 0.05) << "CUDA vs CPU L2 too high for Q8_1-consumed path";
    EXPECT_GE(q81_vs_fp32_cos, 0.95) << "Q8_1-consumed drift vs FP32 baseline too high";
    EXPECT_LE(q81_vs_fp32_l2, 0.15) << "Q8_1-consumed L2 drift vs FP32 baseline too high";
}

TEST_F(Test__CUDAFlashAttentionParity, FlashDecode_FP32_VeryLong_Parity)
{
    SKIP_IF_NO_CUDA();

    // Very long KV cache - stress test for split-K with many splits
    constexpr int kv_len = 2048;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    const size_t q_size = n_heads * head_dim;
    const size_t kv_size = kv_len * n_kv_heads * head_dim;
    const size_t out_size = n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> cuda_output(out_size, 0.0f);

    // CPU reference using production CPUFlashAttentionKernelT::compute_decode()
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute_decode(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        1, kv_len, n_heads, n_kv_heads, head_dim, true, kv_len - 1);
    ASSERT_TRUE(cpu_success) << "CPU compute_decode failed";

    // CUDA decode
    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(cuda_ordinal_);
    auto attention_workspace = bindAttentionWorkspace(cuda_kernel, n_heads, head_dim);
    ASSERT_NE(attention_workspace, nullptr);

    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, q_size * sizeof(float));
    cudaMalloc(&d_K, kv_size * sizeof(float));
    cudaMalloc(&d_V, kv_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, out_size * sizeof(float));

    bool cuda_success = cuda_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        1, kv_len, n_heads, n_kv_heads, head_dim,
        true, 0);
    cudaDeviceSynchronize();

    ASSERT_TRUE(cuda_success);

    cudaMemcpy(cuda_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), out_size));

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(cuda_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashDecode FP32 VeryLong Parity (kv=2048)", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99) << "Cosine similarity too low";
    EXPECT_LE(l2_error, 0.05) << "L2 error too high";
}

TEST_F(Test__CUDAFlashAttentionParity, FlashDecode_FP32_MHA_Parity)
{
    SKIP_IF_NO_CUDA();

    // Multi-head attention (not GQA) - n_heads == n_kv_heads
    constexpr int kv_len = 256;
    constexpr int n_heads = 8;
    constexpr int n_kv_heads = 8; // MHA
    constexpr int head_dim = 64;
    const size_t q_size = n_heads * head_dim;
    const size_t kv_size = kv_len * n_kv_heads * head_dim;
    const size_t out_size = n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> cuda_output(out_size, 0.0f);

    // CPU reference using production CPUFlashAttentionKernelT::compute_decode()
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute_decode(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        1, kv_len, n_heads, n_kv_heads, head_dim, true, kv_len - 1);
    ASSERT_TRUE(cpu_success) << "CPU compute_decode failed";

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(cuda_ordinal_);
    auto attention_workspace = bindAttentionWorkspace(cuda_kernel, n_heads, head_dim);
    ASSERT_NE(attention_workspace, nullptr);

    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, q_size * sizeof(float));
    cudaMalloc(&d_K, kv_size * sizeof(float));
    cudaMalloc(&d_V, kv_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, out_size * sizeof(float));

    bool cuda_success = cuda_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        1, kv_len, n_heads, n_kv_heads, head_dim,
        true, 0);
    cudaDeviceSynchronize();

    ASSERT_TRUE(cuda_success);

    cudaMemcpy(cuda_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), out_size));

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(cuda_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashDecode FP32 MHA Parity (CPUFlashAttentionKernelT vs CUDA)", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99);
    EXPECT_LE(l2_error, 0.05);
}

TEST_F(Test__CUDAFlashAttentionParity, FlashDecode_FP32_HeadDim128_Parity)
{
    SKIP_IF_NO_CUDA();

    // Llama-style head_dim=128
    constexpr int kv_len = 256;
    constexpr int n_heads = 8;
    constexpr int n_kv_heads = 8;
    constexpr int head_dim = 128;
    const size_t q_size = n_heads * head_dim;
    const size_t kv_size = kv_len * n_kv_heads * head_dim;
    const size_t out_size = n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> cuda_output(out_size, 0.0f);

    // CPU reference using production CPUFlashAttentionKernelT::compute_decode()
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute_decode(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        1, kv_len, n_heads, n_kv_heads, head_dim, true, kv_len - 1);
    ASSERT_TRUE(cpu_success) << "CPU compute_decode failed";

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(cuda_ordinal_);
    auto attention_workspace = bindAttentionWorkspace(cuda_kernel, n_heads, head_dim);
    ASSERT_NE(attention_workspace, nullptr);

    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, q_size * sizeof(float));
    cudaMalloc(&d_K, kv_size * sizeof(float));
    cudaMalloc(&d_V, kv_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, out_size * sizeof(float));

    bool cuda_success = cuda_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        1, kv_len, n_heads, n_kv_heads, head_dim,
        true, 0);
    cudaDeviceSynchronize();

    ASSERT_TRUE(cuda_success);

    cudaMemcpy(cuda_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), out_size));

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(cuda_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashDecode FP32 HeadDim128 Parity", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99);
    EXPECT_LE(l2_error, 0.05);
}

TEST_F(Test__CUDAFlashAttentionParity, FlashDecode_FP32_NonCausal_Parity)
{
    SKIP_IF_NO_CUDA();

    // Non-causal decode (bidirectional attention)
    constexpr int kv_len = 128;
    constexpr int n_heads = 8;
    constexpr int n_kv_heads = 8;
    constexpr int head_dim = 64;
    const size_t q_size = n_heads * head_dim;
    const size_t kv_size = kv_len * n_kv_heads * head_dim;
    const size_t out_size = n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> cuda_output(out_size, 0.0f);

    // CPU reference using production CPUFlashAttentionKernelT::compute_decode()
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute_decode(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        1, kv_len, n_heads, n_kv_heads, head_dim, false); // non-causal
    ASSERT_TRUE(cpu_success) << "CPU compute_decode failed";

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(cuda_ordinal_);
    auto attention_workspace = bindAttentionWorkspace(cuda_kernel, n_heads, head_dim);
    ASSERT_NE(attention_workspace, nullptr);

    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, q_size * sizeof(float));
    cudaMalloc(&d_K, kv_size * sizeof(float));
    cudaMalloc(&d_V, kv_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, out_size * sizeof(float));

    // Note: compute_decode may not support non-causal, but let's test it
    bool cuda_success = cuda_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        1, kv_len, n_heads, n_kv_heads, head_dim,
        false, // non-causal
        0);
    cudaDeviceSynchronize();

    ASSERT_TRUE(cuda_success);

    cudaMemcpy(cuda_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), out_size));

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(cuda_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashDecode FP32 NonCausal Parity", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99);
    EXPECT_LE(l2_error, 0.05);
}

// ============================================================================
// Head Dimension Tests
// ============================================================================

TEST_F(Test__CUDAFlashAttentionParity, FlashAttn2_HeadDim128)
{
    SKIP_IF_NO_CUDA();

    // Test with head_dim=128 (Llama-style)
    constexpr int seq_len = 32;
    constexpr int n_heads = 8;
    constexpr int n_kv_heads = 8;
    constexpr int head_dim = 128;
    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = seq_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> cuda_output(out_size, 0.0f);

    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, -1);
    ASSERT_TRUE(cpu_success);

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(cuda_ordinal_);
    auto attention_workspace = bindAttentionWorkspace(cuda_kernel, n_heads, head_dim);
    ASSERT_NE(attention_workspace, nullptr);

    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, q_size * sizeof(float));
    cudaMalloc(&d_K, kv_size * sizeof(float));
    cudaMalloc(&d_V, kv_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, out_size * sizeof(float));

    bool cuda_success = cuda_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    cudaDeviceSynchronize();

    ASSERT_TRUE(cuda_success);

    cudaMemcpy(cuda_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), out_size));

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(cuda_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 HeadDim128", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99);
    EXPECT_LE(l2_error, 0.05);
}

// ============================================================================
// Non-Causal Attention Test
// ============================================================================

TEST_F(Test__CUDAFlashAttentionParity, FlashAttn2_NonCausal)
{
    SKIP_IF_NO_CUDA();

    // Non-causal (bidirectional) attention
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
    std::vector<float> cuda_output(out_size, 0.0f);

    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        false, // non-causal
        -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, -1);
    ASSERT_TRUE(cpu_success);

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(cuda_ordinal_);
    auto attention_workspace = bindAttentionWorkspace(cuda_kernel, n_heads, head_dim);
    ASSERT_NE(attention_workspace, nullptr);

    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, q_size * sizeof(float));
    cudaMalloc(&d_K, kv_size * sizeof(float));
    cudaMalloc(&d_V, kv_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, out_size * sizeof(float));

    bool cuda_success = cuda_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        false, // non-causal
        -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    cudaDeviceSynchronize();

    ASSERT_TRUE(cuda_success);

    cudaMemcpy(cuda_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), out_size));

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(cuda_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 NonCausal", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99);
    EXPECT_LE(l2_error, 0.05);
}

// ============================================================================
// Causal Masking Verification Test
// ============================================================================

TEST_F(Test__CUDAFlashAttentionParity, FlashAttn2_CausalMasking)
{
    SKIP_IF_NO_CUDA();

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
    std::vector<float> cuda_output(out_size, 0.0f);

    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true, // causal
        -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, -1);
    ASSERT_TRUE(cpu_success);

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(cuda_ordinal_);
    auto attention_workspace = bindAttentionWorkspace(cuda_kernel, n_heads, head_dim);
    ASSERT_NE(attention_workspace, nullptr);

    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, q_size * sizeof(float));
    cudaMalloc(&d_K, kv_size * sizeof(float));
    cudaMalloc(&d_V, kv_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, out_size * sizeof(float));

    bool cuda_success = cuda_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true, // causal
        -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    cudaDeviceSynchronize();

    ASSERT_TRUE(cuda_success);

    cudaMemcpy(cuda_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), out_size));

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(cuda_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 CausalMasking", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99);
    EXPECT_LE(l2_error, 0.05);

    // Additional verification: first position should only see V[0]
    // and last position should see weighted average of all V
    float first_pos_val = cuda_output[0];                                 // First element of first position
    float last_pos_val = cuda_output[(seq_len - 1) * n_heads * head_dim]; // First element of last position

    // First position with uniform K should output V[0] = 1.0
    EXPECT_NEAR(first_pos_val, 1.0f, 0.01f) << "First position should only attend to position 0";

    // Last position should have higher value (attending to all positions)
    EXPECT_GT(last_pos_val, first_pos_val) << "Last position should attend to more context";
}

// ============================================================================
// Batch Decoding Test
// ============================================================================

TEST_F(Test__CUDAFlashAttentionParity, FlashDecode_BatchDecoding)
{
    SKIP_IF_NO_CUDA();

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
    std::vector<float> cuda_output(batch_size * out_per_batch, 0.0f);

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(cuda_ordinal_);
    auto attention_workspace = bindAttentionWorkspace(cuda_kernel, n_heads, head_dim);
    ASSERT_NE(attention_workspace, nullptr);

    // Allocate device memory for largest batch element
    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, q_per_batch * sizeof(float));
    cudaMalloc(&d_K, kv_per_batch * sizeof(float));
    cudaMalloc(&d_V, kv_per_batch * sizeof(float));
    cudaMalloc(&d_output, out_per_batch * sizeof(float));

    bool all_success = true;

    // Process each batch element
    for (int b = 0; b < batch_size; b++)
    {
        cudaMemcpy(d_Q, Q_data.data() + b * q_per_batch,
                   q_per_batch * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, K_data.data() + b * kv_per_batch,
                   kv_per_batch * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, V_data.data() + b * kv_per_batch,
                   kv_per_batch * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, out_per_batch * sizeof(float));

        // Use compute_decode for single-token decode
        bool success = cuda_kernel.compute_decode(
            d_Q, d_K, d_V, d_output,
            1,      // seq_len = 1 for decode
            kv_len, // kv_len from cache
            n_heads, n_kv_heads, head_dim,
            true, // causal
            0);   // position_offset
        cudaDeviceSynchronize();

        if (!success)
        {
            std::cerr << "Batch " << b << " decode failed" << std::endl;
            all_success = false;
            continue;
        }

        cudaMemcpy(cuda_output.data() + b * out_per_batch, d_output,
                   out_per_batch * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    ASSERT_TRUE(all_success) << "All batch decode operations should succeed";
    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), batch_size * out_per_batch));

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
                float val = cuda_output[b * out_per_batch + h * head_dim + d];
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
        float diff = cuda_output[i] - cuda_output[out_per_batch + i];
        diff_sum += diff * diff;
    }
    EXPECT_GT(diff_sum, 0.0f) << "Different batch inputs should produce different outputs";

    std::cout << "  FlashDecode BatchDecoding: batch_size=" << batch_size
              << ", kv_len=" << kv_len << ", n_heads=" << n_heads
              << ", n_kv_heads=" << n_kv_heads << " - PASSED" << std::endl;
}

// ============================================================================
// Fused Q8_1 Decode Parity Test
// Tests the new fused Q8_1 CUDA kernel (flash_decoding_q8kv_kernel) that reads
// Q8_1 blocks directly in the attention inner loop, eliminating the separate
// dequant-to-FP32-workspace step.
// ============================================================================

TEST_F(Test__CUDAFlashAttentionParity, FlashDecode_FusedQ81_Parity)
{
    SKIP_IF_NO_CUDA();

    constexpr int kv_len = 256;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;

    const size_t q_size = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_size = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;
    const size_t out_size = static_cast<size_t>(n_heads) * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data_fp32 = randomFP32(kv_size);
    auto V_data_fp32 = randomFP32(kv_size);

    // CPU FP32 reference output
    std::vector<float> cpu_fp32_output(out_size, 0.0f);
    cpuDecodeAttentionReference(
        Q_data.data(), K_data_fp32.data(), V_data_fp32.data(),
        cpu_fp32_output.data(),
        kv_len, n_heads, n_kv_heads, head_dim,
        true, 0);

    // Quantize K/V to Q8_1
    auto k_q81 = Q8_1Tensor::quantize_from_fp32(
        K_data_fp32.data(), {static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)});
    auto v_q81 = Q8_1Tensor::quantize_from_fp32(
        V_data_fp32.data(), {static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)});
    ASSERT_NE(k_q81, nullptr);
    ASSERT_NE(v_q81, nullptr);

    // CPU Q8_1 dequant reference (Q8_1→FP32 then FP32 attention)
    const float *K_deq = k_q81->fp32_data();
    const float *V_deq = v_q81->fp32_data();
    std::vector<float> cpu_q81_deq_output(out_size, 0.0f);
    cpuDecodeAttentionReference(
        Q_data.data(), K_deq, V_deq,
        cpu_q81_deq_output.data(),
        kv_len, n_heads, n_kv_heads, head_dim,
        true, 0);

    // Create FP32 Q tensor and output tensor for GPU
    auto Q_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(n_heads), static_cast<size_t>(head_dim)});
    memcpy(Q_tensor->mutable_data(), Q_data.data(), q_size * sizeof(float));

    auto output_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(n_heads), static_cast<size_t>(head_dim)});
    memset(output_tensor->mutable_data(), 0, out_size * sizeof(float));

    // Upload all tensors to GPU
    DeviceId gpu_dev = gpu_device_;
    ASSERT_TRUE(Q_tensor->ensureOnDevice(gpu_dev));
    ASSERT_TRUE(k_q81->ensureOnDevice(gpu_dev));
    ASSERT_TRUE(v_q81->ensureOnDevice(gpu_dev));
    ASSERT_TRUE(output_tensor->ensureOnDevice(gpu_dev));

    // Call compute_tensor with Q8_1 K/V — should trigger fused kernel
    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(cuda_ordinal_);
    auto attention_workspace = bindAttentionWorkspace(cuda_kernel, n_heads, head_dim);
    ASSERT_NE(attention_workspace, nullptr);
    bool success = cuda_kernel.compute_tensor(
        Q_tensor.get(), k_q81.get(), v_q81.get(), output_tensor.get(),
        1, // batch_size
        1, // seq_len (decode)
        kv_len,
        n_heads, n_kv_heads, head_dim,
        true,  // causal
        0,     // window_size
        nullptr, nullptr, nullptr, // workspace, mask, mpi
        cuda_ordinal_);    // device_idx
    ASSERT_TRUE(success) << "Fused Q8_1 CUDA decode kernel failed";

    // Sync GPU→host
    output_tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *cuda_output = output_tensor->data();
    ASSERT_NE(cuda_output, nullptr);

    // Verify no NaN/Inf
    ASSERT_FALSE(hasNaNOrInf(cuda_output, out_size)) << "CUDA fused Q8_1 output has NaN/Inf";

    // Compare CUDA fused Q8_1 vs CPU dequant Q8_1 (should be very close)
    const double fused_vs_deq_cos = cosineSimilarity(cuda_output, cpu_q81_deq_output.data(), out_size);
    const double fused_vs_deq_l2 = relativeL2Error(cuda_output, cpu_q81_deq_output.data(), out_size);

    // Compare CUDA fused Q8_1 vs CPU FP32 baseline
    const double fused_vs_fp32_cos = cosineSimilarity(cuda_output, cpu_fp32_output.data(), out_size);
    const double fused_vs_fp32_l2 = relativeL2Error(cuda_output, cpu_fp32_output.data(), out_size);

    printComparisonStats("FlashDecode Fused Q8_1 CUDA vs CPU dequant Q8_1",
                         fused_vs_deq_cos, fused_vs_deq_l2,
                         maxAbsError(cuda_output, cpu_q81_deq_output.data(), out_size), out_size);
    printComparisonStats("FlashDecode Fused Q8_1 CUDA vs FP32 baseline",
                         fused_vs_fp32_cos, fused_vs_fp32_l2,
                         maxAbsError(cuda_output, cpu_fp32_output.data(), out_size), out_size);

    // Fused kernel vs dequant should be very close (both operate on same Q8_1 data,
    // just different dequant paths — inline vs separate kernel)
    EXPECT_GE(fused_vs_deq_cos, 0.99) << "CUDA fused Q8_1 vs CPU dequant parity too low";
    EXPECT_LE(fused_vs_deq_l2, 0.05) << "CUDA fused Q8_1 vs CPU dequant L2 too high";

    // Q8_1 quantization error vs FP32 baseline (wider tolerance)
    EXPECT_GE(fused_vs_fp32_cos, 0.95) << "CUDA fused Q8_1 vs FP32 baseline drift too high";
    EXPECT_LE(fused_vs_fp32_l2, 0.15) << "CUDA fused Q8_1 vs FP32 baseline L2 too high";

    std::cout << "  FlashDecode Fused Q8_1: kv_len=" << kv_len
              << ", n_heads=" << n_heads << ", n_kv_heads=" << n_kv_heads
              << ", head_dim=" << head_dim << " - PASSED" << std::endl;
}

TEST_F(Test__CUDAFlashAttentionParity, FlashDecode_FusedQ81_HeadDim128_Parity)
{
    SKIP_IF_NO_CUDA();

    // Test with head_dim=128 (Llama-3 style) and longer KV
    constexpr int kv_len = 512;
    constexpr int n_heads = 8;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 128;

    const size_t q_size = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_size = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;
    const size_t out_size = static_cast<size_t>(n_heads) * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data_fp32 = randomFP32(kv_size);
    auto V_data_fp32 = randomFP32(kv_size);

    // CPU FP32 reference
    std::vector<float> cpu_fp32_output(out_size, 0.0f);
    cpuDecodeAttentionReference(
        Q_data.data(), K_data_fp32.data(), V_data_fp32.data(),
        cpu_fp32_output.data(),
        kv_len, n_heads, n_kv_heads, head_dim,
        true, 0);

    // Quantize to Q8_1
    auto k_q81 = Q8_1Tensor::quantize_from_fp32(
        K_data_fp32.data(), {static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)});
    auto v_q81 = Q8_1Tensor::quantize_from_fp32(
        V_data_fp32.data(), {static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)});
    ASSERT_NE(k_q81, nullptr);
    ASSERT_NE(v_q81, nullptr);

    // Create Q and output tensors
    auto Q_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(n_heads), static_cast<size_t>(head_dim)});
    memcpy(Q_tensor->mutable_data(), Q_data.data(), q_size * sizeof(float));

    auto output_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(n_heads), static_cast<size_t>(head_dim)});
    memset(output_tensor->mutable_data(), 0, out_size * sizeof(float));

    // Upload to GPU
    DeviceId gpu_dev = gpu_device_;
    ASSERT_TRUE(Q_tensor->ensureOnDevice(gpu_dev));
    ASSERT_TRUE(k_q81->ensureOnDevice(gpu_dev));
    ASSERT_TRUE(v_q81->ensureOnDevice(gpu_dev));
    ASSERT_TRUE(output_tensor->ensureOnDevice(gpu_dev));

    // Fused Q8_1 decode
    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(cuda_ordinal_);
    auto attention_workspace = bindAttentionWorkspace(cuda_kernel, n_heads, head_dim);
    ASSERT_NE(attention_workspace, nullptr);
    bool success = cuda_kernel.compute_tensor(
        Q_tensor.get(), k_q81.get(), v_q81.get(), output_tensor.get(),
        1, 1, kv_len,
        n_heads, n_kv_heads, head_dim,
        true, 0,
        nullptr, nullptr, nullptr, cuda_ordinal_);
    ASSERT_TRUE(success) << "Fused Q8_1 CUDA decode (hd=128) failed";

    output_tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *cuda_output = output_tensor->data();
    ASSERT_FALSE(hasNaNOrInf(cuda_output, out_size));

    const double cos = cosineSimilarity(cuda_output, cpu_fp32_output.data(), out_size);
    const double l2 = relativeL2Error(cuda_output, cpu_fp32_output.data(), out_size);

    printComparisonStats("FlashDecode Fused Q8_1 HD128 CUDA vs FP32",
                         cos, l2,
                         maxAbsError(cuda_output, cpu_fp32_output.data(), out_size), out_size);

    EXPECT_GE(cos, 0.95) << "CUDA fused Q8_1 (hd=128) vs FP32 parity too low";
    EXPECT_LE(l2, 0.15) << "CUDA fused Q8_1 (hd=128) L2 too high";

    std::cout << "  FlashDecode Fused Q8_1 HD128: kv_len=" << kv_len
              << ", n_heads=" << n_heads << " - PASSED" << std::endl;
}

#else // !HAVE_CUDA

TEST_F(Test__CUDAFlashAttentionParity, SkipWithoutCUDA)
{
    GTEST_SKIP() << "CUDA not available";
}

#endif
