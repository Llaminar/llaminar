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
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/KernelFactory.h"
#include "transfer/TransferEngine.h"
#include "utils/MPIContext.h"
#include "kernels/cpu/CPURingKVCache.h"
#include "tensors/FP16Utils.h"

#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#include "kernels/cuda/attention/CUDAFlashAttentionKernelT.h"
#include "kernels/cpu/attention/CPUFlashAttentionKernelT.h"
#include "tensors/GpuTensorView.h"
#include <cuda_runtime.h>
#endif

// Now include test utils
#include "../../../../utils/CUDATestUtils.h"
#include "../../../../utils/TestTensorFactory.h"

#include <cstdint>
#include <array>
#include <cstring>
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
     * @brief Assert that grouped verifier output is byte-identical to serial decode.
     *
     * MTP verifier rows are not ordinary approximate attention outputs: every
     * row is eligible to be published into live speculative state.  For those
     * rows, "close enough" would let grouped decode drift from the canonical
     * serial M=1 path and eventually poison KV publication, logits, and token
     * acceptance.  This helper therefore compares the raw FP32 bytes and only
     * uses cosine/L2/max-error numbers as diagnostics when equality fails.
     *
     * @param actual Grouped verifier row produced by the production CUDA path.
     * @param expected Serial M=1 decode row produced by the same CUDA kernel.
     * @param count Number of FP32 elements in the row.
     * @param label Human-readable context included in the assertion failure.
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
        for (; first < count; ++first)
        {
            if (std::memcmp(actual + first, expected + first, sizeof(float)) != 0)
                break;
        }

        double max_abs = 0.0;
        size_t max_index = 0;
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
               << " max_abs=" << max_abs
               << " max_index=" << max_index;
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
     * @brief Diagnostic facade that forbids legacy scalar cache reads on GPU.
     *
     * Production GPU attention must consume the cache's stable request-batched
     * device view and canonical device sequence count.  It must never recover
     * by querying a synchronous scalar count or by asking for a host-sized
     * tensor wrapper.  This facade delegates every graph-native operation to a
     * real CUDA ring cache while poisoning the retired scalar interfaces.  A
     * focused integration test can therefore prove that the real production
     * path remains fully device-owned without reaching into cache internals.
     */
    class DeviceOwnedCacheHostReadTrap final : public IKVCache
    {
    public:
        explicit DeviceOwnedCacheHostReadTrap(IKVCache &delegate)
            : delegate_(delegate)
        {
        }

        ActivationPrecision k_precision() const override
        {
            return delegate_.k_precision();
        }

        ActivationPrecision v_precision() const override
        {
            return delegate_.v_precision();
        }

        int get_cached_tokens(int, int = 0) const override
        {
            ++forbidden_scalar_read_count_;
            return 1;
        }

        int max_seq_len() const override
        {
            return delegate_.max_seq_len();
        }

        int n_layers() const override
        {
            return delegate_.n_layers();
        }

        int first_layer_index() const override
        {
            return delegate_.first_layer_index();
        }

        TensorLayout kv_layout() const override
        {
            return delegate_.kv_layout();
        }

        const int *deviceCachedTokenCountPtr(int layer, int seq_idx = 0) const override
        {
            return delegate_.deviceCachedTokenCountPtr(layer, seq_idx);
        }

        const int *deviceRingHeadPtr(int layer, int seq_idx = 0) const override
        {
            return delegate_.deviceRingHeadPtr(layer, seq_idx);
        }

        bool get_kv(int, int, ITensor **out_k, ITensor **out_v,
                    int *out_kv_len = nullptr) override
        {
            ++forbidden_scalar_read_count_;
            if (out_k)
                *out_k = nullptr;
            if (out_v)
                *out_v = nullptr;
            if (out_kv_len)
                *out_kv_len = 0;
            return false;
        }

        bool get_kv(int, int, const ITensor **out_k, const ITensor **out_v,
                    int *out_kv_len = nullptr) const override
        {
            ++forbidden_scalar_read_count_;
            if (out_k)
                *out_k = nullptr;
            if (out_v)
                *out_v = nullptr;
            if (out_kv_len)
                *out_kv_len = 0;
            return false;
        }

        bool append(int layer, int seq_idx,
                    const ITensor *K, const ITensor *V,
                    int num_tokens) override
        {
            return delegate_.append(layer, seq_idx, K, V, num_tokens);
        }

        bool appendWithStream(int layer, int seq_idx,
                              const ITensor *K, const ITensor *V,
                              int num_tokens, void *gpu_stream) override
        {
            return delegate_.appendWithStream(
                layer, seq_idx, K, V, num_tokens, gpu_stream);
        }

        bool isGraphCaptureReady() const override
        {
            return delegate_.isGraphCaptureReady();
        }

        bool bindGraphAppendCountSource(
            int layer,
            int seq_idx,
            const int32_t *append_tokens_device,
            int captured_max_tokens,
            void *gpu_stream) override
        {
            return delegate_.bindGraphAppendCountSource(
                layer,
                seq_idx,
                append_tokens_device,
                captured_max_tokens,
                gpu_stream);
        }

        bool get_kv_batched_device_view(
            int layer,
            int first_seq_idx,
            int request_count,
            ITensor **out_k,
            ITensor **out_v,
            void *gpu_stream) override
        {
            return delegate_.get_kv_batched_device_view(
                layer,
                first_seq_idx,
                request_count,
                out_k,
                out_v,
                gpu_stream);
        }

        bool get_kv_batched_converted_device_view(
            int layer,
            int first_seq_idx,
            int request_count,
            ActivationPrecision target,
            ITensor **out_k,
            ITensor **out_v,
            const KVReadParams &read) override
        {
            return delegate_.get_kv_batched_converted_device_view(
                layer,
                first_seq_idx,
                request_count,
                target,
                out_k,
                out_v,
                read);
        }

        bool resetRequestState(const StateResetContext &context) override
        {
            return delegate_.resetRequestState(context);
        }

        bool resetSequenceState(
            int seq_idx,
            const StateResetContext &context) override
        {
            return delegate_.resetSequenceState(seq_idx, context);
        }

        bool resetLayerSequenceState(
            int layer,
            int seq_idx,
            const StateResetContext &context) override
        {
            return delegate_.resetLayerSequenceState(layer, seq_idx, context);
        }

        bool resetLayerState(
            int layer,
            const StateResetContext &context) override
        {
            return delegate_.resetLayerState(layer, context);
        }

        bool is_sharded() const override
        {
            return delegate_.is_sharded();
        }

        int local_n_kv_heads() const override
        {
            return delegate_.local_n_kv_heads();
        }

        int n_kv_heads() const override
        {
            return delegate_.n_kv_heads();
        }

        int local_kv_dim() const override
        {
            return delegate_.local_kv_dim();
        }

        int kv_head_start() const override
        {
            return delegate_.kv_head_start();
        }

        int forbiddenScalarReadCount() const
        {
            return forbidden_scalar_read_count_;
        }

    private:
        IKVCache &delegate_;
        mutable int forbidden_scalar_read_count_ = 0;
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

    /**
     * @brief Bind graph-declared scratch for a production grouped KV read.
     *
     * Every cache format now enters the device-owned batched gather API. Native
     * FP16 therefore needs the same bound cache workspace as quantized formats,
     * especially when RoPE-on-read writes an isolated converted view. Binding
     * also publishes the immutable device entry-pointer tables used in capture.
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
            ADD_FAILURE() << "CUDA KV cache does not implement IWorkspaceConsumer";
            return nullptr;
        }

        const WorkspaceRequirements requirements =
            consumer->getWorkspaceRequirements(max_tokens, batch_size, head_dim);
        auto workspace = std::make_unique<DeviceWorkspaceManager>(
            gpu_device_, requirements.total_bytes_with_alignment() + 4096);
        if (!workspace->allocate(requirements))
        {
            ADD_FAILURE() << "Failed to allocate CUDA grouped KV-cache workspace";
            return nullptr;
        }
        consumer->bindWorkspace(workspace.get());
        return workspace;
    }

#ifdef HAVE_CUDA
    void runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
        int seq_len,
        int history_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        const char *label,
        ActivationPrecision cache_precision = ActivationPrecision::FP16,
        int head_start = 0,
        int gqa_n_rep = 0)
    {
        const int kv_len = history_len + seq_len;
        constexpr float rope_theta = 10000000.0f;
        constexpr float partial_rotary_factor = 0.25f;
        const size_t q_cols = static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim);
        const size_t kv_cols = static_cast<size_t>(n_kv_heads) * static_cast<size_t>(head_dim);
        const size_t q_size = static_cast<size_t>(seq_len) * q_cols;
        const size_t kv_size = static_cast<size_t>(kv_len) * kv_cols;
        const size_t history_size = static_cast<size_t>(history_len) * kv_cols;

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

        /**
         * @brief Build any production CUDA cache format from one logical source.
         *
         * The verifier regression is about row-local attention semantics, not
         * about quantizer error.  Both grouped and serial-reference caches are
         * filled from tensors produced by this one format-specific constructor.
         * A failure therefore means the captured verifier path is not
         * decode-equivalent; it cannot be explained by independently rounded
         * reference data.
         */
        auto make_kv_tensor =
            [&](const std::vector<float> &fp32_source,
                const std::vector<uint16_t> &fp16_source,
                size_t offset,
                size_t count,
                const std::vector<size_t> &shape) -> std::shared_ptr<TensorBase>
        {
            switch (cache_precision)
            {
            case ActivationPrecision::FP32:
            {
                std::vector<float> fp32_slice(
                    fp32_source.begin() + static_cast<std::ptrdiff_t>(offset),
                    fp32_source.begin() + static_cast<std::ptrdiff_t>(offset + count));
                auto tensor = std::make_shared<FP32Tensor>(shape);
                std::copy(
                    fp32_slice.begin(),
                    fp32_slice.end(),
                    tensor->mutable_data());
                return tensor;
            }
            case ActivationPrecision::FP16:
            {
                std::vector<uint16_t> fp16_slice(
                    fp16_source.begin() + static_cast<std::ptrdiff_t>(offset),
                    fp16_source.begin() + static_cast<std::ptrdiff_t>(offset + count));
                return std::make_shared<FP16Tensor>(shape, fp16_slice);
            }
            case ActivationPrecision::BF16:
            {
                std::vector<float> fp32_slice(
                    fp32_source.begin() + static_cast<std::ptrdiff_t>(offset),
                    fp32_source.begin() + static_cast<std::ptrdiff_t>(offset + count));
                auto tensor = std::make_shared<BF16Tensor>(shape);
                tensor->from_fp32(fp32_slice.data(), fp32_slice.size());
                return tensor;
            }
            case ActivationPrecision::Q8_1:
            {
                std::vector<float> fp32_slice(
                    fp32_source.begin() + static_cast<std::ptrdiff_t>(offset),
                    fp32_source.begin() + static_cast<std::ptrdiff_t>(offset + count));
                return Q8_1Tensor::quantize_from_fp32(fp32_slice.data(), shape);
            }
            default:
                return nullptr;
            }
        };

        auto q_grouped_tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), q_cols},
            DeviceId::cpu());
        auto history_k_tensor = make_kv_tensor(
            K_data,
            K_fp16,
            0,
            history_size,
            std::vector<size_t>{static_cast<size_t>(history_len), kv_cols});
        auto history_v_tensor = make_kv_tensor(
            V_data,
            V_fp16,
            0,
            history_size,
            std::vector<size_t>{static_cast<size_t>(history_len), kv_cols});
        auto current_k_tensor = make_kv_tensor(
            K_data,
            K_fp16,
            history_size,
            static_cast<size_t>(seq_len) * kv_cols,
            std::vector<size_t>{static_cast<size_t>(seq_len), kv_cols});
        auto current_v_tensor = make_kv_tensor(
            V_data,
            V_fp16,
            history_size,
            static_cast<size_t>(seq_len) * kv_cols,
            std::vector<size_t>{static_cast<size_t>(seq_len), kv_cols});
        auto full_k_tensor = make_kv_tensor(
            K_data,
            K_fp16,
            0,
            kv_size,
            std::vector<size_t>{static_cast<size_t>(kv_len), kv_cols});
        auto full_v_tensor = make_kv_tensor(
            V_data,
            V_fp16,
            0,
            kv_size,
            std::vector<size_t>{static_cast<size_t>(kv_len), kv_cols});
        auto out_grouped_tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), q_cols},
            DeviceId::cpu());
        std::copy(Q_data.begin(), Q_data.end(), q_grouped_tensor->mutable_data());

        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

        auto &transfer = TransferEngine::instance();
        ASSERT_TRUE(transfer.uploadFull(q_grouped_tensor.get(), gpu_device_, stream).success);
        ASSERT_TRUE(transfer.uploadFull(history_k_tensor.get(), gpu_device_, stream).success);
        ASSERT_TRUE(transfer.uploadFull(history_v_tensor.get(), gpu_device_, stream).success);
        ASSERT_TRUE(transfer.uploadFull(current_k_tensor.get(), gpu_device_, stream).success);
        ASSERT_TRUE(transfer.uploadFull(current_v_tensor.get(), gpu_device_, stream).success);
        ASSERT_TRUE(transfer.uploadFull(full_k_tensor.get(), gpu_device_, stream).success);
        ASSERT_TRUE(transfer.uploadFull(full_v_tensor.get(), gpu_device_, stream).success);
        ASSERT_TRUE(transfer.uploadFull(out_grouped_tensor.get(), gpu_device_, stream).success);

        llaminar::v2::kernels::KVCacheConfig config;
        config.precision = cache_precision;
        config.device = gpu_device_;
        config.num_layers = 1;
        config.batch_size = 1;
        config.max_seq_len = kv_len + 8;
        config.n_kv_heads = n_kv_heads;
        config.head_dim = head_dim;

        auto kv_cache = llaminar::v2::kernels::KernelFactory::createKVCache(config);
        ASSERT_NE(kv_cache, nullptr);
        auto kv_workspace =
            bindKVCacheWorkspace(*kv_cache, kv_len, /*batch_size=*/1, head_dim);
        ASSERT_NE(kv_workspace, nullptr);
        ASSERT_TRUE(kv_cache->appendWithStream(
            0, 0, history_k_tensor.get(), history_v_tensor.get(), history_len, stream));
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
        attn_params.device_id = gpu_device_;
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
        attn_params.head_start = head_start;
        attn_params.gqa_n_rep = gqa_n_rep;
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

        KVCacheAppendStage append_stage(append_params);
        AttentionComputeStage attn_stage(attn_params);
        append_stage.setGPUStream(stream);
        attn_stage.setGPUStream(stream);

        const WorkspaceRequirements attn_reqs =
            attn_stage.getWorkspaceRequirements(/*m=*/seq_len, /*n=*/n_heads, /*k=*/head_dim);
        DeviceWorkspaceManager attn_workspace(
            gpu_device_,
            attn_reqs.total_bytes_with_alignment() + 4096);
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
            GraphCaptureGuard guard;
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

        std::vector<float> grouped_output(q_size, 0.0f);
        ASSERT_EQ(cudaMemcpyAsync(grouped_output.data(), out_grouped_tensor->gpu_data_ptr(),
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
        const auto serial_reqs = serial_kernel.getWorkspaceRequirements(1, n_heads, head_dim);
        DeviceWorkspaceManager serial_workspace(
            gpu_device_, serial_reqs.total_bytes_with_alignment() + 4096);
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
                q_m1_tensor.get(), cache_k, cache_v, out_m1_tensor.get(),
                1, 1, row_kv_len,
                n_heads, n_kv_heads, head_dim,
                true, -1,
                nullptr, nullptr,
                &mpi_ctx_, cuda_ordinal_,
                head_start, n_heads, n_kv_heads,
                gqa_n_rep > 0 ? gqa_n_rep : n_heads / n_kv_heads));
            ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

            std::vector<float> serial_output(q_cols, 0.0f);
            ASSERT_EQ(cudaMemcpyAsync(serial_output.data(), out_m1_tensor->gpu_data_ptr(),
                                      q_cols * sizeof(float), cudaMemcpyDeviceToHost, stream),
                      cudaSuccess);
            ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

            const float *grouped_row = grouped_output.data() + static_cast<size_t>(row) * q_cols;
            EXPECT_TRUE(expectBitwiseEqualFP32Rows(
                grouped_row,
                serial_output.data(),
                q_cols,
                std::string(label) + " row " + std::to_string(row)));
        }

        cudaGraphExecDestroy(graph_exec);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
    }
#endif
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

/**
 * @brief Prove a captured CUDA decode consumes a later device-parameter update.
 *
 * The graph is captured for one KV length and then replayed after the same
 * kernel object enqueues a different explicit geometry block.  The captured
 * launch must match a direct same-backend decode at the replay length byte for
 * byte.  This specifically guards the stream-owned parameter writer: a stale
 * graph argument, host mirror, or missing writer dependency changes which KV
 * rows participate and fails the raw output comparison.
 */
TEST_F(Test__CUDAFlashAttentionParity, FlashDecode_FP32_GraphReplayUsesUpdatedDeviceKVLenByteExact)
{
    SKIP_IF_NO_CUDA();

    constexpr int capture_kv_len = 65;
    constexpr int replay_kv_len = 66;
    constexpr int max_kv_len = 80;
    constexpr int n_heads = 4;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    constexpr int seq_len = 1;

    const size_t q_size = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_size =
        static_cast<size_t>(max_kv_len) * n_kv_heads * head_dim;
    auto q_data = randomFP32(q_size);
    auto k_data = randomFP32(kv_size);
    auto v_data = randomFP32(kv_size);

    FP32Tensor q_tensor({size_t{1}, q_size}, DeviceId::cpu());
    FP32Tensor k_tensor(
        {static_cast<size_t>(max_kv_len),
         static_cast<size_t>(n_kv_heads * head_dim)},
        DeviceId::cpu());
    FP32Tensor v_tensor(
        {static_cast<size_t>(max_kv_len),
         static_cast<size_t>(n_kv_heads * head_dim)},
        DeviceId::cpu());
    FP32Tensor captured_output({size_t{1}, q_size}, DeviceId::cpu());
    FP32Tensor direct_output({size_t{1}, q_size}, DeviceId::cpu());
    std::copy(q_data.begin(), q_data.end(), q_tensor.mutable_data());
    std::copy(k_data.begin(), k_data.end(), k_tensor.mutable_data());
    std::copy(v_data.begin(), v_data.end(), v_tensor.mutable_data());

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    auto &transfer = TransferEngine::instance();
    ASSERT_TRUE(transfer.uploadFull(&q_tensor, gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(&k_tensor, gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(&v_tensor, gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(&captured_output, gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(&direct_output, gpu_device_, stream).success);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(
        cuda_ordinal_);
    cuda_kernel.setGPUStream(stream);
    auto workspace = bindAttentionWorkspace(cuda_kernel, n_heads, head_dim);
    ASSERT_NE(workspace, nullptr);

    ASSERT_TRUE(cuda_kernel.prepareDynamicAttnParams(
        capture_kv_len, 0, 1, stream));
    ASSERT_TRUE(cuda_kernel.compute_tensor(
        &q_tensor, &k_tensor, &v_tensor, &captured_output,
        1, seq_len, capture_kv_len,
        n_heads, n_kv_heads, head_dim,
        false, -1,
        nullptr, nullptr,
        &mpi_ctx_, cuda_ordinal_));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    {
        GraphCaptureGuard guard;
        ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), cudaSuccess);
        ASSERT_TRUE(cuda_kernel.compute_tensor(
            &q_tensor, &k_tensor, &v_tensor, &captured_output,
            1, seq_len, capture_kv_len,
            n_heads, n_kv_heads, head_dim,
            false, -1,
            nullptr, nullptr,
            &mpi_ctx_, cuda_ordinal_));
        ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
    }
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), cudaSuccess);

    ASSERT_TRUE(cuda_kernel.prepareDynamicAttnParams(
        replay_kv_len, 0, 1, stream));
    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<float> captured(q_size, 0.0f);
    ASSERT_EQ(cudaMemcpyAsync(
                  captured.data(),
                  captured_output.gpu_data_ptr(),
                  q_size * sizeof(float),
                  cudaMemcpyDeviceToHost,
                  stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    ASSERT_TRUE(cuda_kernel.compute_tensor(
        &q_tensor, &k_tensor, &v_tensor, &direct_output,
        1, seq_len, replay_kv_len,
        n_heads, n_kv_heads, head_dim,
        false, -1,
        nullptr, nullptr,
        &mpi_ctx_, cuda_ordinal_));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<float> direct(q_size, 0.0f);
    ASSERT_EQ(cudaMemcpyAsync(
                  direct.data(),
                  direct_output.gpu_data_ptr(),
                  q_size * sizeof(float),
                  cudaMemcpyDeviceToHost,
                  stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    EXPECT_TRUE(expectBitwiseEqualFP32Rows(
        captured.data(),
        direct.data(),
        q_size,
        "CUDA captured replay after device KV-length update"));

    ASSERT_EQ(cudaGraphExecDestroy(graph_exec), cudaSuccess);
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
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
    const WorkspaceRequirements grouped_requirements =
        cuda_kernel.getWorkspaceRequirements(seq_len, n_heads, head_dim);
    DeviceWorkspaceManager workspace(
        gpu_device_,
        grouped_requirements.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(workspace.allocate(grouped_requirements));
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

/**
 * @brief Prove long CUDA FA2 prefill is invariant to a restored-prefix split.
 *
 * Qwen3.6 phase-split serving can prefill 1024 rows in one request, or restore
 * the first 256 rows and evaluate the remaining 768 rows against the same
 * 1024-row FP16 KV history.  The latter is still a prefill operation even
 * though `query_rows < kv_rows`.  Every suffix output must be byte-identical
 * to the corresponding row from monolithic prefill; otherwise later GDN and KV
 * layers amplify a shape-dependent attention result into different tokens.
 *
 * This test enters the production FP32-Q/FP16-KV FA2 dispatch twice.  It does
 * not use a CPU oracle or deterministic debug mode: the monolithic CUDA launch
 * is the oracle for the suffix CUDA launch, and raw FP32 bytes are the gate.
 */
TEST_F(Test__CUDAFlashAttentionParity, FP16KV_LongSuffixPrefillMatchesMonolithicBytes)
{
    SKIP_IF_NO_CUDA();

    constexpr int full_seq_len = 1024;
    constexpr int prefix_len = 256;
    constexpr int suffix_seq_len = full_seq_len - prefix_len;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 256;
    constexpr int gqa_n_rep = n_heads / n_kv_heads;

    const size_t q_cols = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_cols = static_cast<size_t>(n_kv_heads) * head_dim;
    const size_t full_q_size = static_cast<size_t>(full_seq_len) * q_cols;
    const size_t suffix_q_size = static_cast<size_t>(suffix_seq_len) * q_cols;
    const size_t kv_size = static_cast<size_t>(full_seq_len) * kv_cols;

    const std::vector<float> q_host = randomFP32(full_q_size);
    const std::vector<float> k_source = randomFP32(kv_size);
    const std::vector<float> v_source = randomFP32(kv_size);
    std::vector<uint16_t> k_fp16(kv_size);
    std::vector<uint16_t> v_fp16(kv_size);
    for (size_t element = 0; element < kv_size; ++element)
    {
        k_fp16[element] = fp32_to_fp16(k_source[element]);
        v_fp16[element] = fp32_to_fp16(v_source[element]);
    }

    FP32Tensor q_full(
        {static_cast<size_t>(full_seq_len), q_cols},
        DeviceId::cpu());
    FP32Tensor q_suffix(
        {static_cast<size_t>(suffix_seq_len), q_cols},
        DeviceId::cpu());
    FP16Tensor k_tensor(
        {static_cast<size_t>(full_seq_len), kv_cols},
        k_fp16);
    FP16Tensor v_tensor(
        {static_cast<size_t>(full_seq_len), kv_cols},
        v_fp16);
    FP32Tensor out_full(
        {static_cast<size_t>(full_seq_len), q_cols},
        DeviceId::cpu());
    FP32Tensor out_suffix(
        {static_cast<size_t>(suffix_seq_len), q_cols},
        DeviceId::cpu());

    std::memcpy(
        q_full.mutable_data(),
        q_host.data(),
        full_q_size * sizeof(float));
    std::memcpy(
        q_suffix.mutable_data(),
        q_host.data() + static_cast<size_t>(prefix_len) * q_cols,
        suffix_q_size * sizeof(float));

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_TRUE(q_full.ensureOnDevice(gpu_device_, stream));
    ASSERT_TRUE(q_suffix.ensureOnDevice(gpu_device_, stream));
    ASSERT_TRUE(k_tensor.ensureOnDevice(gpu_device_, stream));
    ASSERT_TRUE(v_tensor.ensureOnDevice(gpu_device_, stream));
    ASSERT_TRUE(out_full.ensureOnDevice(gpu_device_, stream));
    ASSERT_TRUE(out_suffix.ensureOnDevice(gpu_device_, stream));

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> kernel(
        cuda_ordinal_);
    kernel.setGPUStream(stream);
    /*
     * The workspace API's M dimension is the number of simultaneous split-K
     * decode rows, not the FA2 prefill sequence length.  Ordinary prefill uses
     * one device-parameter row and does not allocate per-token split buffers.
     */
    const WorkspaceRequirements requirements =
        kernel.getWorkspaceRequirements(/*m=*/1, n_heads, head_dim);
    DeviceWorkspaceManager workspace(
        gpu_device_,
        requirements.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(workspace.allocate(requirements));
    kernel.bindWorkspace(&workspace);

    ASSERT_TRUE(kernel.prepareDynamicAttnParams(
        full_seq_len,
        /*position_offset=*/0,
        /*query_rows=*/1,
        stream));
    ASSERT_TRUE(kernel.compute_tensor(
        &q_full, &k_tensor, &v_tensor, &out_full,
        /*batch_size=*/1,
        full_seq_len,
        full_seq_len,
        n_heads,
        n_kv_heads,
        head_dim,
        /*causal=*/true,
        /*window_size=*/-1,
        nullptr,
        nullptr,
        &mpi_ctx_,
        cuda_ordinal_,
        /*head_start=*/0,
        n_heads,
        n_kv_heads,
        gqa_n_rep));

    ASSERT_TRUE(kernel.prepareDynamicAttnParams(
        full_seq_len,
        prefix_len,
        /*query_rows=*/1,
        stream));
    ASSERT_TRUE(kernel.compute_tensor(
        &q_suffix, &k_tensor, &v_tensor, &out_suffix,
        /*batch_size=*/1,
        suffix_seq_len,
        full_seq_len,
        n_heads,
        n_kv_heads,
        head_dim,
        /*causal=*/true,
        /*window_size=*/-1,
        nullptr,
        nullptr,
        &mpi_ctx_,
        cuda_ordinal_,
        /*head_start=*/0,
        n_heads,
        n_kv_heads,
        gqa_n_rep));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<float> full_output(full_q_size);
    std::vector<float> suffix_output(suffix_q_size);
    ASSERT_EQ(
        cudaMemcpyAsync(
            full_output.data(),
            out_full.gpu_data_ptr(),
            full_q_size * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream),
        cudaSuccess);
    ASSERT_EQ(
        cudaMemcpyAsync(
            suffix_output.data(),
            out_suffix.gpu_data_ptr(),
            suffix_q_size * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream),
        cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    EXPECT_TRUE(expectBitwiseEqualFP32Rows(
        suffix_output.data(),
        full_output.data() + static_cast<size_t>(prefix_len) * q_cols,
        suffix_q_size,
        "CUDA FP16-KV 256+768 suffix prefill"));

    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
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
    const WorkspaceRequirements grouped_requirements =
        cuda_kernel.getWorkspaceRequirements(seq_len, n_heads, head_dim);
    DeviceWorkspaceManager workspace(
        gpu_device_,
        grouped_requirements.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(workspace.allocate(grouped_requirements));
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

TEST_F(Test__CUDAFlashAttentionParity, GroupedVerifier_FP16KV_Qwen36M2RowsAreByteExactToSerialDecode)
{
    SKIP_IF_NO_CUDA();

    /*
     * The MTP verifier runs a two-row causal continuation after both verifier
     * K/V rows are present. Production must use the named grouped verifier
     * contract, which prepares one device parameter row per verifier row and
     * invokes the same decode kernel/reduction order as serial M=1. A generic
     * FA2 continuation is close numerically but not byte-identical, so it is
     * not a publishable grouped decode implementation.
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
    const auto fa2_reqs =
        fa2_kernel.getWorkspaceRequirements(seq_len, n_heads, head_dim);
    DeviceWorkspaceManager fa2_workspace(gpu_device_,
                                         fa2_reqs.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(fa2_workspace.allocate(fa2_reqs));
    fa2_kernel.bindWorkspace(&fa2_workspace);

    ASSERT_TRUE(fa2_kernel.compute_verifier_rows_decode_equivalent(
        q_m2_tensor.get(), k_tensor.get(), v_tensor.get(), out_fa2_tensor.get(),
        seq_len, kv_len,
        n_heads, n_kv_heads, head_dim,
        true, -1,
        &mpi_ctx_, cuda_ordinal_,
        0, n_heads / n_kv_heads));
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
        printComparisonStats("Grouped verifier FP16KV Qwen3.6 M2 vs M1",
                             cosine, l2_error, max_error, q_cols);
        EXPECT_EQ(
            std::memcmp(fa2_row, serial_output.data(), q_cols * sizeof(float)),
            0)
            << "grouped verifier attention row " << row
            << " must be byte-identical to serial decode";
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
                                     cuda_kernel.getWorkspaceRequirements(seq_len, n_heads, head_dim)
                                             .total_bytes_with_alignment() +
                                         4096);
    ASSERT_TRUE(workspace.allocate(cuda_kernel.getWorkspaceRequirements(seq_len, n_heads, head_dim)));
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
    auto kv_workspace =
        bindKVCacheWorkspace(*kv_cache, kv_len, /*batch_size=*/1, head_dim);
    ASSERT_NE(kv_workspace, nullptr);
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
        stage.getWorkspaceRequirements(/*m=*/seq_len, /*n=*/n_heads, /*k=*/head_dim);
    DeviceWorkspaceManager stage_workspace(
        device, stage_reqs.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(stage_workspace.allocate(stage_reqs));
    stage.bindWorkspace(&stage_workspace);

    /*
     * Dynamic graph metadata is prepared from the pre-append position.  This
     * fixture preloads the complete verifier span instead of running a sibling
     * append stage, so the equivalent pre-append count is `kv_len - seq_len`.
     * Without this update the stage legitimately executes a two-token prompt
     * at position zero, which is not the grouped continuation being compared.
     */
    stage.updateDynamicParams(kv_len - seq_len, seq_len);

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
    auto kv_workspace =
        bindKVCacheWorkspace(*kv_cache, kv_len, /*batch_size=*/1, head_dim);
    ASSERT_NE(kv_workspace, nullptr);
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
        GraphCaptureGuard guard;
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

TEST_F(Test__CUDAFlashAttentionParity, CapturedAppendThenAttention_FP16Cache_Qwen36LocalTPM2RoPEOnReadRowsMatchSerialDecode)
{
    SKIP_IF_NO_CUDA();

    /*
     * LocalTP shards Qwen3.6 attention to 8 local query heads and one local KV
     * head per GPU.  The MoE grouped verifier proof compares these compact
     * all-position rows against serial decode, so keep the same captured
     * append+attention boundary under a small synthetic test.
     */
    runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
        /*seq_len=*/2,
        /*history_len=*/597,
        /*n_heads=*/8,
        /*n_kv_heads=*/1,
        /*head_dim=*/256,
        "Captured append+attention FP16 cache Qwen3.6 LocalTP M2 RoPE-on-read vs M1");
}

TEST_F(Test__CUDAFlashAttentionParity, CapturedAppendThenAttention_FP16Cache_Qwen36LocalTPM3RoPEOnReadRowsMatchSerialDecode)
{
    SKIP_IF_NO_CUDA();

    /*
     * M3 is the first verifier depth where two speculative rows precede the
     * bonus row.  LocalTP must still evaluate each row with the exact serial
     * KV length and GQA mapping, otherwise accepted-state publication can
     * diverge even though the M2 smoke test remains green.
     */
    runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
        /*seq_len=*/3,
        /*history_len=*/596,
        /*n_heads=*/8,
        /*n_kv_heads=*/1,
        /*head_dim=*/256,
        "Captured append+attention FP16 cache Qwen3.6 LocalTP M3 RoPE-on-read vs M1");
}

TEST_F(Test__CUDAFlashAttentionParity, CapturedAppendThenAttention_FP16Cache_Qwen36LocalTPM4RoPEOnReadRowsMatchSerialDecode)
{
    SKIP_IF_NO_CUDA();

    /*
     * M4 covers the maximum grouped verifier row count currently used by the
     * focused MTP sweeps.  Keeping the history+rows total fixed at 599 mirrors
     * the long-context attention regime that exposed drift in the full model.
     */
    runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
        /*seq_len=*/4,
        /*history_len=*/595,
        /*n_heads=*/8,
        /*n_kv_heads=*/1,
        /*head_dim=*/256,
        "Captured append+attention FP16 cache Qwen3.6 LocalTP M4 RoPE-on-read vs M1");
}

/**
 * @brief Prove every supported CUDA grouped-attention depth against serial decode.
 *
 * Prefix-cache suffix prefills are not constrained to the historically common
 * MTP depths M=2..4. Any suffix with at most sixteen rows enters the same
 * production grouped verifier kernel, including the seven-row suffix that
 * exposed a stale launcher guard in the CUDA2 LLEP lifecycle test. Exercise
 * every legal row count so the public M=2..16 policy, captured graph geometry,
 * device parameter array, partial workspace, and kernel launcher cannot drift
 * apart again.
 *
 * FP32, FP16, BF16, and Q8_1 cache storage are all covered. Every source format
 * deliberately enters the production converted FP16 RoPE-on-read view consumed
 * by CUDA grouped decode, while the second-shard cases prove that a non-zero
 * global query-head offset preserves GQA ownership. Every grouped FP32 output
 * row must match the corresponding M=1 CUDA decode row byte-for-byte.
 */
TEST_F(
    Test__CUDAFlashAttentionParity,
    CapturedAppendThenAttention_AllNativeKVFormats_Qwen36LocalTPM2To16RoPEOnReadRowsMatchSerialDecode)
{
    SKIP_IF_NO_CUDA();

    constexpr int kFinalKVLength = 39;
    constexpr int kLocalHeads = 8;
    constexpr int kLocalKVHeads = 1;
    constexpr int kHeadDim = 256;
    constexpr int kGlobalGQARep = 16;
    constexpr std::array<ActivationPrecision, 4> kCacheFormats = {
        ActivationPrecision::FP32,
        ActivationPrecision::FP16,
        ActivationPrecision::BF16,
        ActivationPrecision::Q8_1,
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
                    verifier_rows,
                    kFinalKVLength - verifier_rows,
                    kLocalHeads,
                    kLocalKVHeads,
                    kHeadDim,
                    "Captured CUDA all-M grouped verifier attention vs M1",
                    cache_precision,
                    head_start,
                    kGlobalGQARep);
            }
        }
    }
}

TEST_F(Test__CUDAFlashAttentionParity, CapturedAppendThenAttention_Q81Cache_Qwen36LocalTPM2RoPEOnReadRowsMatchSerialDecode)
{
    SKIP_IF_NO_CUDA();

    /*
     * Regression for production LocalTP verifier graphs whose cache storage is
     * Q8_1 but whose RoPE-on-read attention view is FP16.  CUDA must prepare one
     * device-derived AttentionDeviceParams row per verifier row; otherwise row 0
     * can be evaluated with multi-row prefill semantics and observe row 1.
     */
    runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
        /*seq_len=*/2,
        /*history_len=*/597,
        /*n_heads=*/8,
        /*n_kv_heads=*/1,
        /*head_dim=*/256,
        "Captured append+attention Q8_1 cache Qwen3.6 LocalTP M2 RoPE-on-read vs M1",
        ActivationPrecision::Q8_1);
}

TEST_F(Test__CUDAFlashAttentionParity, CapturedAppendThenAttention_Q81Cache_Qwen36LocalTPM3RoPEOnReadRowsMatchSerialDecode)
{
    SKIP_IF_NO_CUDA();

    /*
     * Production CUDA MTP generally consumes Q8_1 KV storage through a converted
     * FP16 RoPE-on-read shadow.  This M3 case proves the converted-cache path is
     * still byte-equivalent row-by-row once more than one draft token is present.
     */
    runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
        /*seq_len=*/3,
        /*history_len=*/596,
        /*n_heads=*/8,
        /*n_kv_heads=*/1,
        /*head_dim=*/256,
        "Captured append+attention Q8_1 cache Qwen3.6 LocalTP M3 RoPE-on-read vs M1",
        ActivationPrecision::Q8_1);
}

TEST_F(Test__CUDAFlashAttentionParity, CapturedAppendThenAttention_Q81Cache_Qwen36LocalTPM4RoPEOnReadRowsMatchSerialDecode)
{
    SKIP_IF_NO_CUDA();

    /*
     * Q8_1 M4 combines the widest current verifier group with cache conversion.
     * It is intentionally byte-exact because approximate agreement is not a
     * sufficient proof for publishable grouped decode state.
     */
    runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
        /*seq_len=*/4,
        /*history_len=*/595,
        /*n_heads=*/8,
        /*n_kv_heads=*/1,
        /*head_dim=*/256,
        "Captured append+attention Q8_1 cache Qwen3.6 LocalTP M4 RoPE-on-read vs M1",
        ActivationPrecision::Q8_1);
}

TEST_F(Test__CUDAFlashAttentionParity, CapturedAppendThenAttention_FP16Cache_Qwen36LocalTPSecondShardM2RoPEOnReadRowsMatchSerialDecode)
{
    SKIP_IF_NO_CUDA();

    /*
     * The second LocalTP participant uses the same local tensor shapes as the
     * first participant but with a non-zero global Q-head offset.  Replicated
     * KV-head GQA must therefore use the global GQA ratio, not the local one,
     * or the grouped verifier can map local heads to the wrong KV head.
     */
    runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
        /*seq_len=*/2,
        /*history_len=*/597,
        /*n_heads=*/8,
        /*n_kv_heads=*/1,
        /*head_dim=*/256,
        "Captured append+attention FP16 cache Qwen3.6 LocalTP shard1 M2 RoPE-on-read vs M1",
        ActivationPrecision::FP16,
        /*head_start=*/8,
        /*gqa_n_rep=*/16);
}

TEST_F(Test__CUDAFlashAttentionParity, CapturedAppendThenAttention_FP16Cache_Qwen36LocalTPSecondShardM3RoPEOnReadRowsMatchSerialDecode)
{
    SKIP_IF_NO_CUDA();

    /*
     * The non-zero head offset is where local-vs-global GQA mistakes usually
     * hide.  Exercise M3 so the second TP shard proves every grouped row uses
     * the same KV head that serial decode would select.
     */
    runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
        /*seq_len=*/3,
        /*history_len=*/596,
        /*n_heads=*/8,
        /*n_kv_heads=*/1,
        /*head_dim=*/256,
        "Captured append+attention FP16 cache Qwen3.6 LocalTP shard1 M3 RoPE-on-read vs M1",
        ActivationPrecision::FP16,
        /*head_start=*/8,
        /*gqa_n_rep=*/16);
}

TEST_F(Test__CUDAFlashAttentionParity, CapturedAppendThenAttention_FP16Cache_Qwen36LocalTPSecondShardM4RoPEOnReadRowsMatchSerialDecode)
{
    SKIP_IF_NO_CUDA();

    /*
     * Same second-shard proof at M4, matching the maximum verifier group used
     * by the current long-context MTP integration cells.
     */
    runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
        /*seq_len=*/4,
        /*history_len=*/595,
        /*n_heads=*/8,
        /*n_kv_heads=*/1,
        /*head_dim=*/256,
        "Captured append+attention FP16 cache Qwen3.6 LocalTP shard1 M4 RoPE-on-read vs M1",
        ActivationPrecision::FP16,
        /*head_start=*/8,
        /*gqa_n_rep=*/16);
}

TEST_F(Test__CUDAFlashAttentionParity, CapturedAppendThenAttention_Q81Cache_Qwen36LocalTPSecondShardM2RoPEOnReadRowsMatchSerialDecode)
{
    SKIP_IF_NO_CUDA();

    /*
     * Same second-shard LocalTP proof, but through the production Q8_1 cache
     * storage path.  The cache read materializes a converted FP16 RoPE view
     * inside graph capture and the grouped attention rows must still match
     * serial decode row-for-row.
     */
    runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
        /*seq_len=*/2,
        /*history_len=*/597,
        /*n_heads=*/8,
        /*n_kv_heads=*/1,
        /*head_dim=*/256,
        "Captured append+attention Q8_1 cache Qwen3.6 LocalTP shard1 M2 RoPE-on-read vs M1",
        ActivationPrecision::Q8_1,
        /*head_start=*/8,
        /*gqa_n_rep=*/16);
}

TEST_F(Test__CUDAFlashAttentionParity, CapturedAppendThenAttention_Q81Cache_Qwen36LocalTPSecondShardM3RoPEOnReadRowsMatchSerialDecode)
{
    SKIP_IF_NO_CUDA();

    /*
     * Q8_1 second-shard M3 keeps the converted-cache and global-GQA dimensions
     * under the same strict decode-equivalence gate as the first shard.
     */
    runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
        /*seq_len=*/3,
        /*history_len=*/596,
        /*n_heads=*/8,
        /*n_kv_heads=*/1,
        /*head_dim=*/256,
        "Captured append+attention Q8_1 cache Qwen3.6 LocalTP shard1 M3 RoPE-on-read vs M1",
        ActivationPrecision::Q8_1,
        /*head_start=*/8,
        /*gqa_n_rep=*/16);
}

TEST_F(Test__CUDAFlashAttentionParity, CapturedAppendThenAttention_Q81Cache_Qwen36LocalTPSecondShardM4RoPEOnReadRowsMatchSerialDecode)
{
    SKIP_IF_NO_CUDA();

    /*
     * Q8_1 second-shard M4 is the highest-risk attention verifier shape in the
     * focused LocalTP matrix: global head offsets, converted KV, and four
     * publishable verifier rows all meet in one captured graph.
     */
    runCapturedAppendThenAttentionFP16CacheQwen36RoPEOnReadRowsMatchSerialDecode(
        /*seq_len=*/4,
        /*history_len=*/595,
        /*n_heads=*/8,
        /*n_kv_heads=*/1,
        /*head_dim=*/256,
        "Captured append+attention Q8_1 cache Qwen3.6 LocalTP shard1 M4 RoPE-on-read vs M1",
        ActivationPrecision::Q8_1,
        /*head_start=*/8,
        /*gqa_n_rep=*/16);
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

/**
 * @brief Require CUDA prefill to consume the real device-owned KV cache.
 *
 * The old CUDA prefill path could silently substitute projection K/V when the
 * cache path was unavailable.  That was both a coherence hazard and a false
 * success: graph replay could then use tensors from a different request phase
 * while tests still observed a numerically plausible attention result.
 *
 * This regression first proves that omitting the cache-read policy is a hard
 * failure.  It then executes the valid production path through a real CUDA
 * ring cache wrapped by @ref DeviceOwnedCacheHostReadTrap.  The declarative
 * projection tensors contain deliberately unrelated bytes, so byte equality
 * with direct attention over the appended tensors proves that no projection
 * fallback ran.  A zero trap count additionally proves that execution did not
 * consult any synchronous scalar cache state.
 */
TEST_F(Test__CUDAFlashAttentionParity, AttentionStagePrefillRequiresAndUsesDeviceOwnedCache)
{
    SKIP_IF_NO_CUDA();

    constexpr int seq_len = 9;
    constexpr int cache_capacity = 18;
    constexpr int n_heads = 16;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 256;
    const size_t q_cols = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_cols = static_cast<size_t>(n_kv_heads) * head_dim;
    const size_t q_size = static_cast<size_t>(seq_len) * q_cols;
    const size_t kv_size = static_cast<size_t>(seq_len) * kv_cols;

    auto Q_data = randomFP32(q_size);
    auto cache_K_data = randomFP32(kv_size);
    auto cache_V_data = randomFP32(kv_size);
    auto projection_K_data = randomFP32(kv_size);
    auto projection_V_data = randomFP32(kv_size);

    std::vector<uint16_t> cache_K_fp16(kv_size);
    std::vector<uint16_t> cache_V_fp16(kv_size);
    std::vector<uint16_t> projection_K_fp16(kv_size);
    std::vector<uint16_t> projection_V_fp16(kv_size);
    for (size_t i = 0; i < kv_size; ++i)
    {
        cache_K_fp16[i] = fp32_to_fp16(cache_K_data[i]);
        cache_V_fp16[i] = fp32_to_fp16(cache_V_data[i]);

        /*
         * Keep projection values finite but far from the cache payload.  If a
         * future change reintroduces projection substitution, the resulting
         * attention bytes cannot accidentally match the cache oracle.
         */
        projection_K_fp16[i] = fp32_to_fp16(
            projection_K_data[i] + 4.0f + 0.01f * static_cast<float>(i % 23));
        projection_V_fp16[i] = fp32_to_fp16(
            projection_V_data[i] - 3.0f - 0.02f * static_cast<float>(i % 19));
    }

    auto q_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), q_cols},
        DeviceId::cpu());
    auto cache_k_tensor = std::make_unique<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), kv_cols},
        cache_K_fp16);
    auto cache_v_tensor = std::make_unique<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), kv_cols},
        cache_V_fp16);
    auto projection_k_tensor = std::make_unique<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), kv_cols},
        projection_K_fp16);
    auto projection_v_tensor = std::make_unique<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), kv_cols},
        projection_V_fp16);
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
    ASSERT_TRUE(transfer.uploadFull(cache_k_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(cache_v_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(projection_k_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(projection_v_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(stage_out_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(direct_out_tensor.get(), gpu_device_, stream).success);

    llaminar::v2::kernels::KVCacheConfig cache_config;
    cache_config.precision = ActivationPrecision::FP16;
    cache_config.device = gpu_device_;
    cache_config.num_layers = 1;
    cache_config.batch_size = 1;
    cache_config.max_seq_len = cache_capacity;
    cache_config.n_kv_heads = n_kv_heads;
    cache_config.head_dim = head_dim;

    auto real_cache = llaminar::v2::kernels::KernelFactory::createKVCache(cache_config);
    ASSERT_NE(real_cache, nullptr);
    auto cache_workspace = bindKVCacheWorkspace(
        *real_cache, cache_capacity, /*batch_size=*/1, head_dim);
    ASSERT_NE(cache_workspace, nullptr);
    ASSERT_TRUE(real_cache->appendWithStream(
        /*layer=*/0,
        /*seq_idx=*/0,
        cache_k_tensor.get(),
        cache_v_tensor.get(),
        seq_len,
        stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(real_cache->get_cached_tokens(0, 0), seq_len);
    ASSERT_NE(real_cache->deviceCachedTokenCountPtr(0, 0), nullptr);

    DeviceOwnedCacheHostReadTrap cache_trap(*real_cache);

    AttentionComputeStage::Params params;
    params.device_id = gpu_device_;
    params.Q = q_tensor.get();
    params.K = projection_k_tensor.get();
    params.V = projection_v_tensor.get();
    params.output = stage_out_tensor.get();
    params.batch_size = 1;
    params.seq_len = seq_len;
    params.kv_len = seq_len;
    params.n_heads = n_heads;
    params.n_kv_heads = n_kv_heads;
    params.head_dim = head_dim;
    params.causal = true;
    params.auto_detect_mode = true;
    params.kv_cache = &cache_trap;
    params.layer_idx = 0;
    params.mpi_ctx = &mpi_ctx_;

    /*
     * The declarative projection inputs are valid and resident, so this would
     * have succeeded through the retired fallback.  The target architecture
     * instead rejects the configuration before launching any attention work.
     */
    AttentionComputeStage rejected_stage(params);
    rejected_stage.setGPUStream(stream);
    EXPECT_FALSE(rejected_stage.execute(nullptr));
    EXPECT_EQ(cache_trap.forbiddenScalarReadCount(), 0);

    params.read_kv_from_cache = true;
    AttentionComputeStage stage(params);
    stage.setGPUStream(stream);
    const WorkspaceRequirements stage_reqs =
        stage.getWorkspaceRequirements(/*m=*/1, /*n=*/n_heads, /*k=*/head_dim);
    DeviceWorkspaceManager stage_workspace(
        gpu_device_, stage_reqs.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(stage_workspace.allocate(stage_reqs));
    stage.bindWorkspace(&stage_workspace);
    stage.updateDynamicParams(/*pre_append_cached_tokens=*/0, seq_len);
    ASSERT_TRUE(stage.execute(nullptr));

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> direct_kernel(cuda_ordinal_);
    direct_kernel.setGPUStream(stream);
    const WorkspaceRequirements direct_reqs =
        direct_kernel.getWorkspaceRequirements(1, n_heads, head_dim);
    DeviceWorkspaceManager direct_workspace(
        gpu_device_, direct_reqs.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(direct_workspace.allocate(direct_reqs));
    direct_kernel.bindWorkspace(&direct_workspace);
    ASSERT_TRUE(direct_kernel.prepareDynamicAttnParams(
        seq_len,
        /*position_offset=*/0,
        /*query_rows=*/1,
        stream));
    ASSERT_TRUE(direct_kernel.compute_tensor(
        q_tensor.get(), cache_k_tensor.get(), cache_v_tensor.get(), direct_out_tensor.get(),
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

    EXPECT_EQ(cache_trap.forbiddenScalarReadCount(), 0)
        << "CUDA attention consulted retired host/scalar KV state";
    EXPECT_TRUE(expectBitwiseEqualFP32Rows(
        stage_out.data(),
        direct_out.data(),
        q_size,
        "CUDA prefill device-owned KV cache vs direct cache payload"));

    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
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
    auto kv_workspace =
        bindKVCacheWorkspace(*kv_cache, kv_len, /*batch_size=*/1, head_dim);
    ASSERT_NE(kv_workspace, nullptr);
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

/**
 * @brief Prove captured request-batched prefill reads every independent KV slot.
 *
 * This is the focused regression for a Qwen3.6 request-batch failure where the
 * second request's first full-attention context was entirely zero.  Two bugs
 * combined to create that symptom: graph replay stamped append metadata only
 * for sequence zero, and AttentionComputeStage treated sequence zero's cache
 * allocation as the base of a contiguous batch.  CUDA ring entries are separate
 * allocations, so that pointer arithmetic is never a valid batch contract.
 *
 * The test captures the real production append -> cache gather -> FA2 sequence
 * and replays it twice without recapture. Each request has deliberately
 * different K/V data. The grouped output must be byte-identical to running the
 * same CUDA FA2 kernel once per request at both the captured length and the
 * grown length, proving append publication, fixed-stride materialization, and
 * device-owned live-length consumption preserve canonical row arithmetic.
 */
TEST_F(Test__CUDAFlashAttentionParity, CapturedGrowingRequestBatchFP16CacheMatchesIsolatedPrefillByteExact)
{
    SKIP_IF_NO_CUDA();

    constexpr int batch_size = 2;
    constexpr int seq_len = 256;
    constexpr int n_heads = 8;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 256;
    constexpr int gqa_n_rep = 8;
    constexpr float rope_theta = 10000000.0f;
    constexpr float partial_rotary_factor = 0.25f;
    const size_t q_cols = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_cols = static_cast<size_t>(n_kv_heads) * head_dim;
    const size_t q_elements =
        static_cast<size_t>(batch_size) * seq_len * q_cols;
    const size_t kv_elements =
        static_cast<size_t>(batch_size) * seq_len * kv_cols;

    std::vector<float> q_host = randomFP32(q_elements);
    std::vector<float> k_source = randomFP32(kv_elements);
    std::vector<float> v_source = randomFP32(kv_elements);
    std::vector<float> q_host_second = randomFP32(q_elements);
    std::vector<float> k_source_second = randomFP32(kv_elements);
    std::vector<float> v_source_second = randomFP32(kv_elements);
    std::vector<uint16_t> k_fp16(kv_elements);
    std::vector<uint16_t> v_fp16(kv_elements);
    std::vector<uint16_t> k_fp16_second(kv_elements);
    std::vector<uint16_t> v_fp16_second(kv_elements);
    for (size_t i = 0; i < kv_elements; ++i)
    {
        /*
         * Request-specific offsets make accidental sequence-zero reuse fail
         * even if the random generator happens to produce similar rows.
         */
        const int request =
            static_cast<int>(i / (static_cast<size_t>(seq_len) * kv_cols));
        k_fp16[i] = fp32_to_fp16(k_source[i] + 0.25f * request);
        v_fp16[i] = fp32_to_fp16(v_source[i] - 0.35f * request);
        /*
         * The continuation block must not repeat the captured block. A
         * repeated source lets a broken resident gather substitute the newly
         * appended suffix for the historical prefix without making the
         * regression fail. Large request-specific offsets make both token
         * identity and request-bank identity visible in the byte oracle.
         */
        k_fp16_second[i] = fp32_to_fp16(
            k_source_second[i] + 1.5f + 0.45f * request);
        v_fp16_second[i] = fp32_to_fp16(
            v_source_second[i] - 1.75f - 0.55f * request);
    }

    auto q_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(batch_size * seq_len), q_cols},
        DeviceId::cpu());
    auto k_tensor = std::make_unique<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(batch_size * seq_len), kv_cols},
        k_fp16);
    auto v_tensor = std::make_unique<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(batch_size * seq_len), kv_cols},
        v_fp16);
    auto batch_output = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(batch_size * seq_len), q_cols},
        DeviceId::cpu());
    auto serial_output = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(batch_size * seq_len), q_cols},
        DeviceId::cpu());
    std::copy(q_host.begin(), q_host.end(), q_tensor->mutable_data());

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    auto &transfer = TransferEngine::instance();
    ASSERT_TRUE(transfer.uploadFull(q_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(k_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(v_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(batch_output.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(serial_output.get(), gpu_device_, stream).success);

    llaminar::v2::kernels::KVCacheConfig cache_config;
    cache_config.precision = ActivationPrecision::FP16;
    cache_config.device = gpu_device_;
    cache_config.num_layers = 1;
    cache_config.batch_size = batch_size;
    cache_config.max_seq_len = 2 * seq_len + 65;
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
        gpu_device_, kv_requirements.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(kv_workspace.allocate(kv_requirements));
    kv_workspace_consumer->bindWorkspace(&kv_workspace);

    KVCacheAppendStage append_stage({
        .device_id = gpu_device_,
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
        .device_id = gpu_device_,
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
        .head_start = 0,
        .gqa_n_rep = gqa_n_rep,
        .causal = true,
        .auto_detect_mode = true,
        .kv_cache = kv_cache.get(),
        .layer_idx = 0,
        .read_kv_from_cache = true,
        .apply_rope_to_k = true,
        .rope_theta = rope_theta,
        .partial_rotary_factor = partial_rotary_factor,
    });
    append_stage.setGPUStream(stream);
    attention_stage.setGPUStream(stream);

    const WorkspaceRequirements attention_requirements =
        attention_stage.getWorkspaceRequirements(
            /*m=*/batch_size * seq_len,
            /*n=*/n_heads,
            /*k=*/head_dim);
    DeviceWorkspaceManager attention_workspace(
        gpu_device_, attention_requirements.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(attention_workspace.allocate(attention_requirements));
    attention_stage.bindWorkspace(&attention_workspace);

    append_stage.updateDynamicParams(/*pos_offset=*/0, seq_len);
    attention_stage.updateDynamicParams(/*pos_offset=*/0, seq_len);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), cudaSuccess);
    bool capture_ok = false;
    {
        GraphCaptureGuard guard;
        capture_ok = append_stage.execute(nullptr) && attention_stage.execute(nullptr);
    }
    ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
    ASSERT_TRUE(capture_ok);
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), cudaSuccess);
    ASSERT_NE(graph_exec, nullptr);
    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    EXPECT_EQ(kv_cache->get_cached_tokens(0, 0), seq_len);
    EXPECT_EQ(kv_cache->get_cached_tokens(0, 1), seq_len);

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32>
        serial_kernel(cuda_ordinal_);
    serial_kernel.setGPUStream(stream);
    const WorkspaceRequirements serial_requirements =
        serial_kernel.getWorkspaceRequirements(
            /*m=*/1,
            /*n=*/n_heads,
            /*k=*/head_dim);
    DeviceWorkspaceManager serial_workspace(
        gpu_device_, serial_requirements.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(serial_workspace.allocate(serial_requirements));
    serial_kernel.bindWorkspace(&serial_workspace);

    auto *q_base = static_cast<float *>(q_tensor->gpu_data_ptr());
    auto *k_base = static_cast<uint16_t *>(k_tensor->gpu_data_ptr());
    auto *v_base = static_cast<uint16_t *>(v_tensor->gpu_data_ptr());
    auto *serial_base = static_cast<float *>(serial_output->gpu_data_ptr());
    ASSERT_NE(q_base, nullptr);
    ASSERT_NE(k_base, nullptr);
    ASSERT_NE(v_base, nullptr);
    ASSERT_NE(serial_base, nullptr);

    /**
     * @brief Replace graph inputs without changing their captured addresses.
     *
     * CUDA graph nodes retain the Q/K/V device pointers recorded during
     * capture. Updating the bytes at those stable addresses models the real
     * prefill graph, where each replay receives a different prompt block while
     * the graph topology and allocation identities remain unchanged.
     */
    auto upload_graph_block = [&](const std::vector<float> &q,
                                  const std::vector<uint16_t> &k,
                                  const std::vector<uint16_t> &v) -> bool
    {
        return cudaMemcpyAsync(
                   q_base,
                   q.data(),
                   q_elements * sizeof(float),
                   cudaMemcpyHostToDevice,
                   stream) == cudaSuccess &&
               cudaMemcpyAsync(
                   k_base,
                   k.data(),
                   kv_elements * sizeof(uint16_t),
                   cudaMemcpyHostToDevice,
                   stream) == cudaSuccess &&
               cudaMemcpyAsync(
                   v_base,
                   v.data(),
                   kv_elements * sizeof(uint16_t),
                   cudaMemcpyHostToDevice,
                   stream) == cudaSuccess;
    };

    /*
     * Build an eager cache oracle with the same pre-RoPE bytes.  Scalar cache
     * reads apply the identical CUDA RoPE transform but are not graph replayed,
     * so byte equality isolates the resident captured gather/RoPE/attention
     * chain without substituting a CPU implementation.
     */
    auto reference_cache =
        llaminar::v2::kernels::KernelFactory::createKVCache(cache_config);
    ASSERT_NE(reference_cache, nullptr);
    auto reference_kv_workspace = bindKVCacheWorkspace(
        *reference_cache, 2 * seq_len, batch_size, head_dim);
    ASSERT_NE(reference_kv_workspace, nullptr);
    for (int request = 0; request < batch_size; ++request)
    {
        const size_t kv_offset =
            static_cast<size_t>(request) * seq_len * kv_cols;
        GpuTensorView k_block(
            k_base + kv_offset, seq_len, kv_cols,
            TensorType::FP16, cuda_ordinal_);
        GpuTensorView v_block(
            v_base + kv_offset, seq_len, kv_cols,
            TensorType::FP16, cuda_ordinal_);
        ASSERT_TRUE(reference_cache->appendWithStream(
            0, request, &k_block, &v_block, seq_len, stream));
    }
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    IKVCache::KVReadParams reference_read;
    reference_read.rope_theta = rope_theta;
    reference_read.position_start = 0;
    reference_read.n_kv_heads = n_kv_heads;
    reference_read.head_dim = head_dim;
    reference_read.rope_dim =
        static_cast<int>(partial_rotary_factor * head_dim);
    reference_read.gpu_stream = stream;

    for (int request = 0; request < batch_size; ++request)
    {
        const size_t q_offset =
            static_cast<size_t>(request) * seq_len * q_cols;
        ITensor *reference_k = nullptr;
        ITensor *reference_v = nullptr;
        int reference_kv_len = 0;
        ASSERT_TRUE(reference_cache->get_kv_converted(
            0, request, ActivationPrecision::FP16,
            &reference_k, &reference_v, &reference_kv_len,
            &reference_read));
        ASSERT_EQ(reference_kv_len, seq_len);
        GpuTensorView q_view(
            q_base + q_offset, seq_len, q_cols, TensorType::FP32, cuda_ordinal_);
        GpuTensorView output_view(
            serial_base + q_offset, seq_len, q_cols, TensorType::FP32, cuda_ordinal_);
        ASSERT_TRUE(serial_kernel.compute_tensor(
            &q_view,
            reference_k,
            reference_v,
            &output_view,
            /*batch_size=*/1,
            seq_len,
            seq_len,
            n_heads,
            n_kv_heads,
            head_dim,
            /*causal=*/true,
            /*window_size=*/-1,
            /*workspace_scores=*/nullptr,
            /*workspace_mask=*/nullptr,
            &mpi_ctx_,
            cuda_ordinal_,
            /*head_start=*/0,
            /*local_n_heads=*/n_heads,
            /*local_n_kv_heads=*/n_kv_heads,
            /*gqa_n_rep=*/gqa_n_rep));
    }
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<float> actual(q_elements);
    std::vector<float> expected(q_elements);
    ASSERT_EQ(cudaMemcpyAsync(
                  actual.data(), batch_output->gpu_data_ptr(),
                  q_elements * sizeof(float), cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(
                  expected.data(), serial_output->gpu_data_ptr(),
                  q_elements * sizeof(float), cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    for (int request = 0; request < batch_size; ++request)
    {
        const size_t request_offset =
            static_cast<size_t>(request) * seq_len * q_cols;
        EXPECT_TRUE(expectBitwiseEqualFP32Rows(
            actual.data() + request_offset,
            expected.data() + request_offset,
            static_cast<size_t>(seq_len) * q_cols,
                "captured FP16 request-batch attention request=" +
                std::to_string(request)));
    }

    /*
     * The second launch grows each physical cache bank beyond the length that
     * existed when the graph was captured. No host metadata or launch shape is
     * refreshed here; resident gather and FA2 must observe the new device count
     * while preserving the max-sequence request stride.
     */
    ASSERT_TRUE(upload_graph_block(
        q_host_second,
        k_fp16_second,
        v_fp16_second));
    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(kv_cache->get_cached_tokens(0, 0), 2 * seq_len);
    EXPECT_EQ(kv_cache->get_cached_tokens(0, 1), 2 * seq_len);

    for (int request = 0; request < batch_size; ++request)
    {
        const size_t kv_offset =
            static_cast<size_t>(request) * seq_len * kv_cols;
        GpuTensorView k_block(
            k_base + kv_offset, seq_len, kv_cols,
            TensorType::FP16, cuda_ordinal_);
        GpuTensorView v_block(
            v_base + kv_offset, seq_len, kv_cols,
            TensorType::FP16, cuda_ordinal_);
        ASSERT_TRUE(reference_cache->appendWithStream(
            0, request, &k_block, &v_block, seq_len, stream));
    }
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    for (int request = 0; request < batch_size; ++request)
    {
        const size_t q_offset =
            static_cast<size_t>(request) * seq_len * q_cols;
        ITensor *reference_k = nullptr;
        ITensor *reference_v = nullptr;
        int reference_kv_len = 0;
        ASSERT_TRUE(reference_cache->get_kv_converted(
            0, request, ActivationPrecision::FP16,
            &reference_k, &reference_v, &reference_kv_len,
            &reference_read));
        ASSERT_EQ(reference_kv_len, 2 * seq_len);
        GpuTensorView q_view(
            q_base + q_offset, seq_len, q_cols,
            TensorType::FP32, cuda_ordinal_);
        GpuTensorView output_view(
            serial_base + q_offset, seq_len, q_cols,
            TensorType::FP32, cuda_ordinal_);
        ASSERT_TRUE(serial_kernel.compute_tensor(
            &q_view,
            reference_k,
            reference_v,
            &output_view,
            /*batch_size=*/1,
            seq_len,
            /*kv_len=*/2 * seq_len,
            n_heads,
            n_kv_heads,
            head_dim,
            /*causal=*/true,
            /*window_size=*/-1,
            /*workspace_scores=*/nullptr,
            /*workspace_mask=*/nullptr,
            &mpi_ctx_,
            cuda_ordinal_,
            /*head_start=*/0,
            /*local_n_heads=*/n_heads,
            /*local_n_kv_heads=*/n_kv_heads,
            /*gqa_n_rep=*/gqa_n_rep));
    }
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    ASSERT_EQ(cudaMemcpyAsync(
                  actual.data(), batch_output->gpu_data_ptr(),
                  q_elements * sizeof(float), cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(
                  expected.data(), serial_output->gpu_data_ptr(),
                  q_elements * sizeof(float), cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (int request = 0; request < batch_size; ++request)
    {
        const size_t request_offset =
            static_cast<size_t>(request) * seq_len * q_cols;
        EXPECT_TRUE(expectBitwiseEqualFP32Rows(
            actual.data() + request_offset,
            expected.data() + request_offset,
            static_cast<size_t>(seq_len) * q_cols,
            "captured growing FP16 request-batch attention request=" +
                std::to_string(request)));
    }

    /*
     * Production keeps the instantiated prefill graph across request resets.
     * Clear only logical cache state, then rebuild the same two distinct blocks
     * through the unchanged graph. This catches stale converted-shadow
     * metadata, graph nodes that retain a pre-reset sequence head, and gathers
     * that accidentally publish only the current append block.
     */
    ASSERT_TRUE(kv_cache->resetRequestState(
        IKVCache::StateResetContext::testReinitialization(stream)));
    EXPECT_EQ(kv_cache->get_cached_tokens(0, 0), 0);
    EXPECT_EQ(kv_cache->get_cached_tokens(0, 1), 0);
    ASSERT_TRUE(upload_graph_block(q_host, k_fp16, v_fp16));
    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    ASSERT_TRUE(upload_graph_block(
        q_host_second,
        k_fp16_second,
        v_fp16_second));
    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(kv_cache->get_cached_tokens(0, 0), 2 * seq_len);
    EXPECT_EQ(kv_cache->get_cached_tokens(0, 1), 2 * seq_len);

    ASSERT_EQ(cudaMemcpyAsync(
                  actual.data(), batch_output->gpu_data_ptr(),
                  q_elements * sizeof(float), cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (int request = 0; request < batch_size; ++request)
    {
        const size_t request_offset =
            static_cast<size_t>(request) * seq_len * q_cols;
        EXPECT_TRUE(expectBitwiseEqualFP32Rows(
            actual.data() + request_offset,
            expected.data() + request_offset,
            static_cast<size_t>(seq_len) * q_cols,
            "post-clear captured growing FP16 request-batch attention request=" +
                std::to_string(request)));
    }

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
}

/**
 * @brief Prove the maximum small CUDA request batch remains serial-row exact.
 *
 * The captured request-batch path has a distinct device-parameter publisher
 * from ordinary single-request grouped verification.  That publisher used to
 * reject more than four total rows even though the public attention contract,
 * graph workspace, and grouped kernel all support sixteen.  Two requests with
 * eight rows each exercise the exact upper boundary and force the implementation
 * to preserve one live device-owned cache count per independent request.
 *
 * The graph is captured once and replayed after a second, distinct K/V block
 * grows both request-local rings from eight rows to sixteen. FP32, FP16, BF16,
 * and Q8_1 source caches all enter the real converted FP16 RoPE-on-read path
 * consumed by CUDA grouped attention. Every output row is compared byte for
 * byte with the production CUDA M=1 decode kernel over the same converted
 * request-local prefix.
 */
TEST_F(
    Test__CUDAFlashAttentionParity,
    CapturedGrowingMaximumSmallRequestBatchAllStandardKVFormatsMatchesSerialDecodeRows)
{
    SKIP_IF_NO_CUDA();

    constexpr int request_count = 2;
    constexpr int query_rows = 8;
    constexpr int n_heads = 4;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    constexpr float rope_theta = 10000000.0f;
    constexpr float partial_rotary_factor = 0.25f;
    constexpr std::array<ActivationPrecision, 4> formats = {
        ActivationPrecision::FP32,
        ActivationPrecision::FP16,
        ActivationPrecision::BF16,
        ActivationPrecision::Q8_1,
    };
    static_assert(
        request_count * query_rows ==
            attention::kMaxGroupedVerifierAttentionRows,
        "The regression must exercise the grouped request-row capacity exactly");

    const size_t q_cols = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_cols = static_cast<size_t>(n_kv_heads) * head_dim;
    const size_t request_q_elements = static_cast<size_t>(query_rows) * q_cols;
    const size_t request_kv_elements = static_cast<size_t>(query_rows) * kv_cols;
    const size_t q_elements =
        static_cast<size_t>(request_count) * request_q_elements;
    const size_t kv_elements =
        static_cast<size_t>(request_count) * request_kv_elements;

    std::vector<float> q_first = randomFP32(q_elements);
    std::vector<float> k_first = randomFP32(kv_elements);
    std::vector<float> v_first = randomFP32(kv_elements);
    std::vector<float> q_second = randomFP32(q_elements);
    std::vector<float> k_second = randomFP32(kv_elements);
    std::vector<float> v_second = randomFP32(kv_elements);
    for (int request = 0; request < request_count; ++request)
    {
        const size_t begin =
            static_cast<size_t>(request) * request_kv_elements;
        for (size_t element = 0; element < request_kv_elements; ++element)
        {
            /*
             * Large request and replay offsets make request-zero substitution,
             * stale suffix reuse, and an unchanged graph input visible even
             * after a lossy cache quantizer.
             */
            k_first[begin + element] += 0.25f * request;
            v_first[begin + element] -= 0.35f * request;
            k_second[begin + element] += 1.5f + 0.45f * request;
            v_second[begin + element] -= 1.75f + 0.55f * request;
        }
    }

    auto makeFP32 =
        [](const std::vector<float> &values, size_t rows, size_t cols)
        -> std::shared_ptr<TensorBase>
    {
        auto tensor =
            std::make_shared<FP32Tensor>(std::vector<size_t>{rows, cols});
        std::copy(values.begin(), values.end(), tensor->mutable_data());
        return tensor;
    };
    auto makeNativeKV =
        [](const std::vector<float> &values,
           size_t rows,
           size_t cols,
           ActivationPrecision precision) -> std::shared_ptr<TensorBase>
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
            for (size_t index = 0; index < values.size(); ++index)
            {
                fp16[index] = fp32_to_fp16(values[index]);
            }
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

    auto &transfer = TransferEngine::instance();
    for (const ActivationPrecision format : formats)
    {
        SCOPED_TRACE(
            "cache_precision=" +
            std::to_string(static_cast<int>(format)));

        auto q_tensor = makeFP32(
            q_first, request_count * query_rows, q_cols);
        auto k_tensor = makeNativeKV(
            k_first, request_count * query_rows, kv_cols, format);
        auto v_tensor = makeNativeKV(
            v_first, request_count * query_rows, kv_cols, format);
        auto k_second_tensor = makeNativeKV(
            k_second, request_count * query_rows, kv_cols, format);
        auto v_second_tensor = makeNativeKV(
            v_second, request_count * query_rows, kv_cols, format);
        auto grouped_output = std::make_shared<FP32Tensor>(
            std::vector<size_t>{
                static_cast<size_t>(request_count * query_rows), q_cols});
        auto serial_output = std::make_shared<FP32Tensor>(
            std::vector<size_t>{1, q_cols});
        ASSERT_NE(k_tensor, nullptr);
        ASSERT_NE(v_tensor, nullptr);
        ASSERT_NE(k_second_tensor, nullptr);
        ASSERT_NE(v_second_tensor, nullptr);

        cudaStream_t stream = nullptr;
        ASSERT_EQ(
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            cudaSuccess);
        ASSERT_TRUE(
            transfer.uploadFull(q_tensor.get(), gpu_device_, stream).success);
        ASSERT_TRUE(
            transfer.uploadFull(k_tensor.get(), gpu_device_, stream).success);
        ASSERT_TRUE(
            transfer.uploadFull(v_tensor.get(), gpu_device_, stream).success);
        ASSERT_TRUE(
            transfer.uploadFull(
                k_second_tensor.get(), gpu_device_, stream)
                .success);
        ASSERT_TRUE(
            transfer.uploadFull(
                v_second_tensor.get(), gpu_device_, stream)
                .success);
        ASSERT_TRUE(
            transfer.uploadFull(
                grouped_output.get(), gpu_device_, stream)
                .success);
        ASSERT_TRUE(
            transfer.uploadFull(serial_output.get(), gpu_device_, stream)
                .success);

        llaminar::v2::kernels::KVCacheConfig cache_config;
        cache_config.precision = format;
        cache_config.device = gpu_device_;
        cache_config.num_layers = 1;
        cache_config.batch_size = request_count;
        cache_config.max_seq_len = 2 * query_rows + 8;
        cache_config.n_kv_heads = n_kv_heads;
        cache_config.head_dim = head_dim;

        auto production_cache =
            llaminar::v2::kernels::KernelFactory::createKVCache(cache_config);
        ASSERT_NE(production_cache, nullptr);
        auto production_cache_workspace = bindKVCacheWorkspace(
            *production_cache,
            2 * query_rows,
            request_count,
            head_dim);
        ASSERT_NE(production_cache_workspace, nullptr);

        KVCacheAppendStage append_stage({
            .device_id = gpu_device_,
            .K = k_tensor.get(),
            .V = v_tensor.get(),
            .kv_cache = production_cache.get(),
            .layer_idx = 0,
            .seq_idx = 0,
            .num_tokens = request_count * query_rows,
            .batch_size = request_count,
            .seq_len = query_rows,
            .head_dim = head_dim,
        });
        AttentionComputeStage attention_stage({
            .device_id = gpu_device_,
            .mpi_ctx = &mpi_ctx_,
            .Q = q_tensor.get(),
            .K = k_tensor.get(),
            .V = v_tensor.get(),
            .output = grouped_output.get(),
            .batch_size = request_count,
            .seq_len = query_rows,
            .kv_len = query_rows,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .head_start = 0,
            .gqa_n_rep = n_heads / n_kv_heads,
            .causal = true,
            .auto_detect_mode = true,
            .kv_cache = production_cache.get(),
            .layer_idx = 0,
            .read_kv_from_cache = true,
            .apply_rope_to_k = true,
            .rope_theta = rope_theta,
            .partial_rotary_factor = partial_rotary_factor,
        });
        append_stage.setGPUStream(stream);
        attention_stage.setGPUStream(stream);

        const WorkspaceRequirements attention_requirements =
            attention_stage.getWorkspaceRequirements(
                request_count * query_rows, n_heads, head_dim);
        DeviceWorkspaceManager attention_workspace(
            gpu_device_,
            attention_requirements.total_bytes_with_alignment() + 4096);
        ASSERT_TRUE(attention_workspace.allocate(attention_requirements));
        attention_stage.bindWorkspace(&attention_workspace);
        append_stage.updateDynamicParams(/*pos_offset=*/0, query_rows);
        attention_stage.updateDynamicParams(/*pos_offset=*/0, query_rows);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graph_exec = nullptr;
        ASSERT_EQ(
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
            cudaSuccess);
        bool capture_ok = false;
        {
            GraphCaptureGuard guard;
            capture_ok =
                append_stage.execute(nullptr) &&
                attention_stage.execute(nullptr);
        }
        ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
        ASSERT_TRUE(capture_ok);
        ASSERT_NE(graph, nullptr);
        ASSERT_EQ(
            cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
            cudaSuccess);
        ASSERT_NE(graph_exec, nullptr);
        ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_EQ(
            production_cache->get_cached_tokens(0, 0), query_rows);
        EXPECT_EQ(
            production_cache->get_cached_tokens(0, 1), query_rows);

        /*
         * The eager cache is an oracle only for cache conversion. Attention
         * arithmetic remains entirely same-backend: every expected row is
         * produced by the ordinary CUDA M=1 decode kernel.
         */
        auto reference_cache =
            llaminar::v2::kernels::KernelFactory::createKVCache(cache_config);
        ASSERT_NE(reference_cache, nullptr);
        auto reference_cache_workspace = bindKVCacheWorkspace(
            *reference_cache,
            2 * query_rows,
            request_count,
            head_dim);
        ASSERT_NE(reference_cache_workspace, nullptr);

        auto appendReferenceBlock =
            [&](const std::vector<float> &k_values,
                const std::vector<float> &v_values)
        {
            for (int request = 0; request < request_count; ++request)
            {
                const size_t begin =
                    static_cast<size_t>(request) * request_kv_elements;
                const auto k_begin =
                    k_values.begin() + static_cast<std::ptrdiff_t>(begin);
                const auto v_begin =
                    v_values.begin() + static_cast<std::ptrdiff_t>(begin);
                std::vector<float> request_k(
                    k_begin,
                    k_begin +
                        static_cast<std::ptrdiff_t>(request_kv_elements));
                std::vector<float> request_v(
                    v_begin,
                    v_begin +
                        static_cast<std::ptrdiff_t>(request_kv_elements));
                auto request_k_tensor = makeNativeKV(
                    request_k, query_rows, kv_cols, format);
                auto request_v_tensor = makeNativeKV(
                    request_v, query_rows, kv_cols, format);
                ASSERT_NE(request_k_tensor, nullptr);
                ASSERT_NE(request_v_tensor, nullptr);
                ASSERT_TRUE(
                    transfer.uploadFull(
                        request_k_tensor.get(), gpu_device_, stream)
                        .success);
                ASSERT_TRUE(
                    transfer.uploadFull(
                        request_v_tensor.get(), gpu_device_, stream)
                        .success);
                ASSERT_TRUE(reference_cache->appendWithStream(
                    0,
                    request,
                    request_k_tensor.get(),
                    request_v_tensor.get(),
                    query_rows,
                    stream));
            }
            ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        };
        appendReferenceBlock(k_first, v_first);

        llaminar2::cuda::CUDAFlashAttentionKernelT<
            ActivationPrecision::FP32>
            serial_kernel(cuda_ordinal_);
        serial_kernel.setGPUStream(stream);
        const WorkspaceRequirements serial_requirements =
            serial_kernel.getWorkspaceRequirements(
                /*m=*/1, n_heads, head_dim);
        DeviceWorkspaceManager serial_workspace(
            gpu_device_,
            serial_requirements.total_bytes_with_alignment() + 4096);
        ASSERT_TRUE(serial_workspace.allocate(serial_requirements));
        serial_kernel.bindWorkspace(&serial_workspace);

        IKVCache::KVReadParams reference_read;
        reference_read.rope_theta = rope_theta;
        reference_read.position_start = 0;
        reference_read.n_kv_heads = n_kv_heads;
        reference_read.head_dim = head_dim;
        reference_read.rope_dim =
            static_cast<int>(partial_rotary_factor * head_dim);
        reference_read.gpu_stream = stream;

        auto expectSerialRows =
            [&](const std::vector<float> &q_values,
                int completed_blocks,
                const std::string &phase)
        {
            std::vector<float> actual(q_elements, 0.0f);
            ASSERT_EQ(
                cudaMemcpyAsync(
                    actual.data(),
                    grouped_output->gpu_data_ptr(),
                    q_elements * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    stream),
                cudaSuccess);
            ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

            for (int request = 0; request < request_count; ++request)
            {
                ITensor *converted_k = nullptr;
                ITensor *converted_v = nullptr;
                int converted_rows = 0;
                ASSERT_TRUE(reference_cache->get_kv_converted(
                    0,
                    request,
                    ActivationPrecision::FP16,
                    &converted_k,
                    &converted_v,
                    &converted_rows,
                    &reference_read));
                ASSERT_EQ(
                    converted_rows,
                    (completed_blocks + 1) * query_rows);
                ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

                for (int row = 0; row < query_rows; ++row)
                {
                    const int visible_rows =
                        completed_blocks * query_rows + row + 1;
                    const size_t q_begin =
                        static_cast<size_t>(request) *
                            request_q_elements +
                        static_cast<size_t>(row) * q_cols;
                    auto q_single = makeFP32(
                        std::vector<float>(
                            q_values.begin() +
                                static_cast<std::ptrdiff_t>(q_begin),
                            q_values.begin() +
                                static_cast<std::ptrdiff_t>(
                                    q_begin + q_cols)),
                        /*rows=*/1,
                        q_cols);
                    ASSERT_TRUE(
                        transfer.uploadFull(
                            q_single.get(), gpu_device_, stream)
                            .success);

                    GpuTensorView k_prefix(
                        converted_k->gpu_data_ptr(),
                        visible_rows,
                        kv_cols,
                        TensorType::FP16,
                        cuda_ordinal_);
                    GpuTensorView v_prefix(
                        converted_v->gpu_data_ptr(),
                        visible_rows,
                        kv_cols,
                        TensorType::FP16,
                        cuda_ordinal_);
                    ASSERT_TRUE(serial_kernel.prepareDynamicAttnParams(
                        visible_rows,
                        visible_rows - 1,
                        /*query_rows=*/1,
                        stream));
                    ASSERT_TRUE(serial_kernel.compute_tensor(
                        q_single.get(),
                        &k_prefix,
                        &v_prefix,
                        serial_output.get(),
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
                        cuda_ordinal_,
                        /*head_start=*/0,
                        /*local_n_heads=*/n_heads,
                        /*local_n_kv_heads=*/n_kv_heads,
                        /*gqa_n_rep=*/n_heads / n_kv_heads));

                    std::vector<float> expected(q_cols, 0.0f);
                    ASSERT_EQ(
                        cudaMemcpyAsync(
                            expected.data(),
                            serial_output->gpu_data_ptr(),
                            q_cols * sizeof(float),
                            cudaMemcpyDeviceToHost,
                            stream),
                        cudaSuccess);
                    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
                    EXPECT_TRUE(expectBitwiseEqualFP32Rows(
                        actual.data() + q_begin,
                        expected.data(),
                        q_cols,
                        phase +
                            " format=" +
                            std::to_string(static_cast<int>(format)) +
                            " request=" + std::to_string(request) +
                            " row=" + std::to_string(row)));
                }
            }
        };

        expectSerialRows(
            q_first,
            /*completed_blocks=*/0,
            "captured initial CUDA request batch");

        /*
         * Graph nodes retain the original tensor addresses. Replace only the
         * bytes at those addresses, then replay without host-side attention or
         * cache metadata updates. This is the production device-owned lifetime.
         */
        ASSERT_EQ(
            cudaMemcpyAsync(
                q_tensor->gpu_data_ptr(),
                q_second.data(),
                q_elements * sizeof(float),
                cudaMemcpyHostToDevice,
                stream),
            cudaSuccess);
        ASSERT_EQ(
            cudaMemcpyAsync(
                k_tensor->gpu_data_ptr(),
                k_second_tensor->gpu_data_ptr(),
                k_tensor->size_bytes(),
                cudaMemcpyDeviceToDevice,
                stream),
            cudaSuccess);
        ASSERT_EQ(
            cudaMemcpyAsync(
                v_tensor->gpu_data_ptr(),
                v_second_tensor->gpu_data_ptr(),
                v_tensor->size_bytes(),
                cudaMemcpyDeviceToDevice,
                stream),
            cudaSuccess);
        ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_EQ(
            production_cache->get_cached_tokens(0, 0), 2 * query_rows);
        EXPECT_EQ(
            production_cache->get_cached_tokens(0, 1), 2 * query_rows);

        appendReferenceBlock(k_second, v_second);
        expectSerialRows(
            q_second,
            /*completed_blocks=*/1,
            "captured grown CUDA request batch");

        ASSERT_EQ(cudaGraphExecDestroy(graph_exec), cudaSuccess);
        ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }
}

/**
 * @brief Prove captured unequal-length request decode is byte-identical to M=1.
 *
 * Independent requests do not share a logical KV length.  The production cache
 * gather emits fixed-stride request banks, while the grouped flash-decode kernel
 * must read one canonical device count per bank.  Lengths 37 and 19 deliberately
 * select two and one scalar decode splits respectively, so byte equality proves
 * both request-bank addressing and row-local split/reduction policy.
 */
TEST_F(Test__CUDAFlashAttentionParity, CapturedVariableLengthRequestBatchFP16DecodeMatchesSerialByteExact)
{
    SKIP_IF_NO_CUDA();

    constexpr int request_count = 2;
    constexpr int query_rows = 1;
    constexpr std::array<int, request_count> kv_lens = {37, 19};
    constexpr int max_kv_len = kv_lens[0];
    constexpr int n_heads = 4;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    const size_t q_cols = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_cols = static_cast<size_t>(n_kv_heads) * head_dim;
    const size_t q_elements = static_cast<size_t>(request_count) * q_cols;

    auto q_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(request_count), q_cols},
        DeviceId::cpu());
    auto grouped_output = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(request_count), q_cols},
        DeviceId::cpu());
    auto serial_output = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(request_count), q_cols},
        DeviceId::cpu());
    const std::vector<float> q_host = randomFP32(q_elements);
    std::copy(q_host.begin(), q_host.end(), q_tensor->mutable_data());

    std::array<std::unique_ptr<FP16Tensor>, request_count> k_by_request;
    std::array<std::unique_ptr<FP16Tensor>, request_count> v_by_request;
    for (int request = 0; request < request_count; ++request)
    {
        const size_t elements =
            static_cast<size_t>(kv_lens[request]) * kv_cols;
        std::vector<float> k_source = randomFP32(elements);
        std::vector<float> v_source = randomFP32(elements);
        std::vector<uint16_t> k_fp16(elements);
        std::vector<uint16_t> v_fp16(elements);
        for (size_t i = 0; i < elements; ++i)
        {
            /*
             * A deterministic request offset makes accidental sequence-zero
             * reuse obvious even when the random streams contain close values.
             */
            k_fp16[i] = fp32_to_fp16(
                k_source[i] + static_cast<float>(request) * 0.375f);
            v_fp16[i] = fp32_to_fp16(
                v_source[i] - static_cast<float>(request) * 0.625f);
        }
        k_by_request[request] = std::make_unique<FP16Tensor>(
            std::vector<size_t>{static_cast<size_t>(kv_lens[request]), kv_cols},
            k_fp16);
        v_by_request[request] = std::make_unique<FP16Tensor>(
            std::vector<size_t>{static_cast<size_t>(kv_lens[request]), kv_cols},
            v_fp16);
    }

    cudaStream_t stream = nullptr;
    ASSERT_EQ(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        cudaSuccess);
    auto &transfer = TransferEngine::instance();
    ASSERT_TRUE(transfer.uploadFull(q_tensor.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(grouped_output.get(), gpu_device_, stream).success);
    ASSERT_TRUE(transfer.uploadFull(serial_output.get(), gpu_device_, stream).success);
    for (int request = 0; request < request_count; ++request)
    {
        ASSERT_TRUE(
            transfer.uploadFull(k_by_request[request].get(), gpu_device_, stream)
                .success);
        ASSERT_TRUE(
            transfer.uploadFull(v_by_request[request].get(), gpu_device_, stream)
                .success);
    }

    llaminar::v2::kernels::KVCacheConfig cache_config;
    cache_config.precision = ActivationPrecision::FP16;
    cache_config.device = gpu_device_;
    cache_config.num_layers = 1;
    cache_config.batch_size = request_count;
    cache_config.max_seq_len = max_kv_len + 8;
    cache_config.n_kv_heads = n_kv_heads;
    cache_config.head_dim = head_dim;
    auto kv_cache =
        llaminar::v2::kernels::KernelFactory::createKVCache(cache_config);
    ASSERT_NE(kv_cache, nullptr);
    auto cache_workspace = bindKVCacheWorkspace(
        *kv_cache,
        max_kv_len,
        request_count,
        head_dim);
    ASSERT_NE(cache_workspace, nullptr);

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
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (int request = 0; request < request_count; ++request)
    {
        ASSERT_EQ(kv_cache->get_cached_tokens(/*layer=*/0, request),
                  kv_lens[request]);
    }

    /*
     * K/V fields satisfy the stage's declarative inputs, but read_kv_from_cache
     * makes the captured production gather authoritative.  The supplied tensor
     * is intentionally only request zero's variable-length allocation: treating
     * it as a contiguous batch would therefore fail this test immediately.
     */
    AttentionComputeStage attention_stage({
        .device_id = gpu_device_,
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
            /*m=*/request_count * query_rows,
            /*n=*/n_heads,
            /*k=*/head_dim);
    DeviceWorkspaceManager attention_workspace(
        gpu_device_,
        attention_requirements.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(attention_workspace.allocate(attention_requirements));
    attention_stage.bindWorkspace(&attention_workspace);

    /*
     * Host position is fixed launch geometry only.  The captured grouped path
     * must derive the two live lengths from cache-owned device metadata.
     */
    attention_stage.updateDynamicParams(
        /*pos_offset=*/max_kv_len - query_rows,
        query_rows);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    ASSERT_EQ(
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
        cudaSuccess);
    bool capture_ok = false;
    {
        GraphCaptureGuard guard;
        capture_ok = attention_stage.execute(nullptr);
    }
    ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
    ASSERT_TRUE(capture_ok);
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(
        cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
        cudaSuccess);
    ASSERT_NE(graph_exec, nullptr);
    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32>
        serial_kernel(cuda_ordinal_);
    serial_kernel.setGPUStream(stream);
    const WorkspaceRequirements serial_requirements =
        serial_kernel.getWorkspaceRequirements(
            /*m=*/1,
            /*n=*/n_heads,
            /*k=*/head_dim);
    DeviceWorkspaceManager serial_workspace(
        gpu_device_,
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
            cuda_ordinal_);
        GpuTensorView output_view(
            serial_device + static_cast<size_t>(request) * q_cols,
            /*rows=*/1,
            q_cols,
            TensorType::FP32,
            cuda_ordinal_);

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
            cuda_ordinal_,
            /*head_start=*/0,
            /*local_n_heads=*/n_heads,
            /*local_n_kv_heads=*/n_kv_heads,
            /*gqa_n_rep=*/n_heads / n_kv_heads));
    }
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<float> actual(q_elements);
    std::vector<float> expected(q_elements);
    ASSERT_EQ(
        cudaMemcpyAsync(
            actual.data(),
            grouped_output->gpu_data_ptr(),
            q_elements * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream),
        cudaSuccess);
    ASSERT_EQ(
        cudaMemcpyAsync(
            expected.data(),
            serial_output->gpu_data_ptr(),
            q_elements * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream),
        cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    for (int request = 0; request < request_count; ++request)
    {
        const size_t offset = static_cast<size_t>(request) * q_cols;
        EXPECT_TRUE(expectBitwiseEqualFP32Rows(
            actual.data() + offset,
            expected.data() + offset,
            q_cols,
            "captured variable-length FP16 request decode request=" +
                std::to_string(request)));
    }

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
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
    auto kv_workspace =
        bindKVCacheWorkspace(*kv_cache, kv_len, /*batch_size=*/1, head_dim);
    ASSERT_NE(kv_workspace, nullptr);
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
    auto kv_workspace =
        bindKVCacheWorkspace(*kv_cache, kv_len, /*batch_size=*/1, head_dim);
    ASSERT_NE(kv_workspace, nullptr);
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
    auto kv_workspace =
        bindKVCacheWorkspace(*kv_cache, kv_len, /*batch_size=*/1, head_dim);
    ASSERT_NE(kv_workspace, nullptr);
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
    cudaStream_t stream = nullptr;
    ASSERT_EQ(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        cudaSuccess);
    ASSERT_TRUE(Q_tensor->ensureOnDevice(gpu_dev, stream));
    ASSERT_TRUE(k_q81->ensureOnDevice(gpu_dev, stream));
    ASSERT_TRUE(v_q81->ensureOnDevice(gpu_dev, stream));
    ASSERT_TRUE(output_tensor->ensureOnDevice(gpu_dev, stream));

    // Call compute_tensor with Q8_1 K/V — should trigger fused kernel
    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(cuda_ordinal_);
    cuda_kernel.setGPUStream(stream);
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
    TransferEngine::publishCurrentDeviceWrite(output_tensor, stream);
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
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
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
    cudaStream_t stream = nullptr;
    ASSERT_EQ(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        cudaSuccess);
    ASSERT_TRUE(Q_tensor->ensureOnDevice(gpu_dev, stream));
    ASSERT_TRUE(k_q81->ensureOnDevice(gpu_dev, stream));
    ASSERT_TRUE(v_q81->ensureOnDevice(gpu_dev, stream));
    ASSERT_TRUE(output_tensor->ensureOnDevice(gpu_dev, stream));

    // Fused Q8_1 decode
    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(cuda_ordinal_);
    cuda_kernel.setGPUStream(stream);
    auto attention_workspace = bindAttentionWorkspace(cuda_kernel, n_heads, head_dim);
    ASSERT_NE(attention_workspace, nullptr);
    bool success = cuda_kernel.compute_tensor(
        Q_tensor.get(), k_q81.get(), v_q81.get(), output_tensor.get(),
        1, 1, kv_len,
        n_heads, n_kv_heads, head_dim,
        true, 0,
        nullptr, nullptr, nullptr, cuda_ordinal_);
    ASSERT_TRUE(success) << "Fused Q8_1 CUDA decode (hd=128) failed";

    TransferEngine::publishCurrentDeviceWrite(output_tensor, stream);
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
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

#else // !HAVE_CUDA

TEST_F(Test__CUDAFlashAttentionParity, SkipWithoutCUDA)
{
    GTEST_SKIP() << "CUDA not available";
}

#endif
