/**
 * @file Test__ROCmAttentionPaddingParity.cpp
 * @brief ROCm attention parity tests for hostile padded prefill bucket tails.
 *
 * These tests call the ROCm FP32 attention kernel directly, independent of
 * model loading and graph execution. They prove that causal prefill real rows
 * are stable when future bucket padding rows contain values that would corrupt
 * attention output if visible.
 */

#include <gtest/gtest.h>

#include "backends/GPUDeviceContextPool.h"
#include "execution/compute_stages/stages/AttentionComputeStage.h"
#include "execution/compute_stages/stages/LMHeadStage.h"
#include "execution/local_execution/coherence/GpuCoherence.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "kernels/KernelFactory.h"
#include "utils/MPIContext.h"
#include "../../../utils/PreparedWeightTestHarness.h"

#ifdef HAVE_ROCM
#include "kernels/rocm/attention/ROCmFlashAttentionKernelT.h"
#include <hip/hip_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    /// @brief Host-side attention tensors with row-major [seq, heads, head_dim] layout.
    struct HostAttentionInputs
    {
        std::vector<float> query; ///< Query rows for all query heads.
        std::vector<float> key;   ///< Key rows for all KV heads.
        std::vector<float> value; ///< Value rows for all KV heads.
    };

    /// @brief Returns the element count for Q or output buffers.
    size_t queryElementCount(int seq_len, int n_heads, int head_dim)
    {
        return static_cast<size_t>(seq_len) * static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim);
    }

    /// @brief Returns the element count for K/V buffers.
    size_t keyValueElementCount(int seq_len, int n_kv_heads, int head_dim)
    {
        return static_cast<size_t>(seq_len) * static_cast<size_t>(n_kv_heads) * static_cast<size_t>(head_dim);
    }

    /**
     * @brief Fill real Q/K/V rows with deterministic low-magnitude values.
     *
     * The repeating patterns avoid degenerate softmax inputs without creating
     * large logits that would hide causal mask regressions behind saturation.
     */
    HostAttentionInputs makeRealInputs(int seq_len, int n_heads, int n_kv_heads, int head_dim)
    {
        const size_t q_cols = static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim);
        const size_t kv_cols = static_cast<size_t>(n_kv_heads) * static_cast<size_t>(head_dim);

        HostAttentionInputs inputs;
        inputs.query.resize(queryElementCount(seq_len, n_heads, head_dim));
        inputs.key.resize(keyValueElementCount(seq_len, n_kv_heads, head_dim));
        inputs.value.resize(keyValueElementCount(seq_len, n_kv_heads, head_dim));

        for (int row = 0; row < seq_len; ++row)
        {
            for (size_t col = 0; col < q_cols; ++col)
                inputs.query[static_cast<size_t>(row) * q_cols + col] = 0.01f * static_cast<float>((row + static_cast<int>(col)) % 17 + 1);

            for (size_t col = 0; col < kv_cols; ++col)
            {
                inputs.key[static_cast<size_t>(row) * kv_cols + col] = 0.02f * static_cast<float>((row * 3 + static_cast<int>(col)) % 13 + 1);
                inputs.value[static_cast<size_t>(row) * kv_cols + col] = 0.03f * static_cast<float>((row * 5 + static_cast<int>(col)) % 19 + 1);
            }
        }

        return inputs;
    }

    /**
     * @brief Copy real rows into a larger bucket and poison future pad rows.
     *
     * Large pad K/V values make the non-causal sanity test sensitive: if the
     * causal path accidentally sees these rows, the real prefix should move.
     */
    HostAttentionInputs makePaddedInputs(
        const HostAttentionInputs &real,
        int real_seq_len,
        int bucket_seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim)
    {
        const size_t q_cols = static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim);
        const size_t kv_cols = static_cast<size_t>(n_kv_heads) * static_cast<size_t>(head_dim);

        HostAttentionInputs padded;
        padded.query.assign(queryElementCount(bucket_seq_len, n_heads, head_dim), 0.0f);
        padded.key.assign(keyValueElementCount(bucket_seq_len, n_kv_heads, head_dim), 0.0f);
        padded.value.assign(keyValueElementCount(bucket_seq_len, n_kv_heads, head_dim), 0.0f);

        for (int row = 0; row < real_seq_len; ++row)
        {
            std::copy_n(real.query.begin() + static_cast<size_t>(row) * q_cols, q_cols,
                        padded.query.begin() + static_cast<size_t>(row) * q_cols);
            std::copy_n(real.key.begin() + static_cast<size_t>(row) * kv_cols, kv_cols,
                        padded.key.begin() + static_cast<size_t>(row) * kv_cols);
            std::copy_n(real.value.begin() + static_cast<size_t>(row) * kv_cols, kv_cols,
                        padded.value.begin() + static_cast<size_t>(row) * kv_cols);
        }

        for (int row = real_seq_len; row < bucket_seq_len; ++row)
        {
            for (size_t col = 0; col < q_cols; ++col)
                padded.query[static_cast<size_t>(row) * q_cols + col] = 7.0f;

            for (size_t col = 0; col < kv_cols; ++col)
            {
                padded.key[static_cast<size_t>(row) * kv_cols + col] = 16.0f;
                padded.value[static_cast<size_t>(row) * kv_cols + col] = (col % 2 == 0) ? 32.0f : -32.0f;
            }
        }

        return padded;
    }

    /// @brief Returns true if the output contains an invalid floating-point value.
    bool hasNaNOrInf(const std::vector<float> &values)
    {
        return std::any_of(values.begin(), values.end(), [](float value)
                           { return std::isnan(value) || std::isinf(value); });
    }

    /// @brief Compute maximum absolute difference over the real output prefix only.
    float maxPrefixDiff(const std::vector<float> &a, const std::vector<float> &b, size_t prefix_count)
    {
        float max_diff = 0.0f;
        for (size_t i = 0; i < prefix_count; ++i)
            max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
        return max_diff;
    }

    /// @brief Compute cosine similarity over the real output prefix only.
    double prefixCosine(const std::vector<float> &a, const std::vector<float> &b, size_t prefix_count)
    {
        double dot = 0.0;
        double norm_a = 0.0;
        double norm_b = 0.0;

        for (size_t i = 0; i < prefix_count; ++i)
        {
            dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }

        const double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
        return denom < 1e-12 ? 0.0 : dot / denom;
    }

    /**
     * @brief Creates hidden states with hostile padded rows for LM-head row-selection tests.
     */
    std::unique_ptr<FP32Tensor> createDeterministicHiddenStates(
        int bucket_seq_len,
        int d_model,
        int hostile_tail_start_row)
    {
        auto hidden_states = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(bucket_seq_len), static_cast<size_t>(d_model)},
            DeviceId::cpu());
        float *hidden_data = hidden_states->mutable_data();

        for (int row = 0; row < bucket_seq_len; ++row)
        {
            const bool hostile_tail = row >= hostile_tail_start_row;
            for (int feature = 0; feature < d_model; ++feature)
            {
                const size_t idx = static_cast<size_t>(row) * d_model + feature;
                hidden_data[idx] = hostile_tail
                                       ? 25.0f + 3.0f * static_cast<float>(row) + 0.125f * static_cast<float>(feature)
                                       : 0.03125f * static_cast<float>(row + 1) +
                                             0.001f * static_cast<float>((feature % 17) - 8);
            }
        }

        return hidden_states;
    }

    /**
     * @brief Creates deterministic FP32 LM-head weights in [vocab_size, d_model] layout.
     */
    std::unique_ptr<FP32Tensor> createDeterministicLmHeadWeights(int vocab_size, int d_model)
    {
        auto lm_head_weight = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(vocab_size), static_cast<size_t>(d_model)},
            DeviceId::cpu());
        float *weight_data = lm_head_weight->mutable_data();

        for (int vocab = 0; vocab < vocab_size; ++vocab)
        {
            for (int feature = 0; feature < d_model; ++feature)
            {
                const size_t idx = static_cast<size_t>(vocab) * d_model + feature;
                weight_data[idx] = 0.0025f * static_cast<float>((vocab % 13) - 6) +
                                   0.00075f * static_cast<float>((feature % 11) - 5);
            }
        }

        return lm_head_weight;
    }

    /// @brief CPU reference for one LM-head row projection.
    std::vector<float> computeLmHeadReference(
        const FP32Tensor &hidden_states,
        const FP32Tensor &lm_head_weight,
        int hidden_row,
        int vocab_size,
        int d_model)
    {
        const float *hidden_data = hidden_states.data();
        const float *weight_data = lm_head_weight.data();
        std::vector<float> reference(static_cast<size_t>(vocab_size), 0.0f);

        for (int vocab = 0; vocab < vocab_size; ++vocab)
        {
            float dot = 0.0f;
            for (int feature = 0; feature < d_model; ++feature)
                dot += hidden_data[static_cast<size_t>(hidden_row) * d_model + feature] *
                       weight_data[static_cast<size_t>(vocab) * d_model + feature];
            reference[static_cast<size_t>(vocab)] = dot;
        }

        return reference;
    }

    /// @brief Maximum absolute difference between logits row 0 and a reference vector.
    float maxLogitDifference(const FP32Tensor &logits, const std::vector<float> &reference)
    {
        const float *logit_data = logits.data();
        float max_difference = 0.0f;
        for (size_t i = 0; i < reference.size(); ++i)
            max_difference = std::max(max_difference, std::abs(logit_data[i] - reference[i]));
        return max_difference;
    }

    /// @brief Reference FP32 single-query GQA attention over a selected KV prefix.
    std::vector<float> referenceSingleQueryGQAAttention(
        const float *q_data,
        const float *k_data,
        const float *v_data,
        int kv_len,
        int n_heads,
        int n_kv_heads,
        int head_dim)
    {
        std::vector<float> output(static_cast<size_t>(n_heads * head_dim), 0.0f);
        std::vector<float> scores(static_cast<size_t>(kv_len), 0.0f);
        const int kv_row_stride = n_kv_heads * head_dim;
        const int gqa_group_size = n_heads / n_kv_heads;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        for (int head = 0; head < n_heads; ++head)
        {
            const int kv_head = head / gqa_group_size;
            const float *q_head = q_data + head * head_dim;
            float max_score = -std::numeric_limits<float>::infinity();

            for (int pos = 0; pos < kv_len; ++pos)
            {
                const float *k_head = k_data + pos * kv_row_stride + kv_head * head_dim;
                float dot = 0.0f;
                for (int dim = 0; dim < head_dim; ++dim)
                    dot += q_head[dim] * k_head[dim];
                scores[static_cast<size_t>(pos)] = dot * scale;
                max_score = std::max(max_score, scores[static_cast<size_t>(pos)]);
            }

            float denom = 0.0f;
            for (int pos = 0; pos < kv_len; ++pos)
            {
                const float weight = std::exp(scores[static_cast<size_t>(pos)] - max_score);
                scores[static_cast<size_t>(pos)] = weight;
                denom += weight;
            }

            for (int dim = 0; dim < head_dim; ++dim)
            {
                float value = 0.0f;
                for (int pos = 0; pos < kv_len; ++pos)
                {
                    const float *v_head = v_data + pos * kv_row_stride + kv_head * head_dim;
                    value += (scores[static_cast<size_t>(pos)] / denom) * v_head[dim];
                }
                output[static_cast<size_t>(head * head_dim + dim)] = value;
            }
        }

        return output;
    }

    /// @brief Largest absolute elementwise difference for a tensor row and reference vector.
    float maxTensorVectorDiff(const FP32Tensor &actual, const std::vector<float> &expected)
    {
        const float *actual_data = actual.data();
        float max_diff = 0.0f;
        for (size_t i = 0; i < expected.size(); ++i)
            max_diff = std::max(max_diff, std::abs(actual_data[i] - expected[i]));
        return max_diff;
    }

#ifdef HAVE_ROCM
    /// @brief Explicitly discard HIP cleanup status when already reporting an earlier failure.
    void discardHipCleanupStatus(hipError_t status)
    {
        (void)status;
    }

    /// @brief Returns true when a ROCm device is visible to HIP.
    bool hasROCm()
    {
        int count = 0;
        const hipError_t err = hipGetDeviceCount(&count);
        return err == hipSuccess && count > 0;
    }

    /// @brief Result bundle for a direct ROCm attention invocation.
    struct ROCmAttentionRunResult
    {
        bool ok = false;           ///< True when allocation, kernel launch, and copy-back succeeded.
        std::string error;         ///< Diagnostic text for the first failed HIP operation.
        std::vector<float> output; ///< Host copy of the output tensor.
    };

    /**
     * @brief Run ROCm FP32 attention on raw device buffers and copy output back.
     *
     * This keeps the test independent of tensors and graph capture so failures
     * point at the backend attention mask rather than orchestration code.
     */
    ROCmAttentionRunResult runROCmAttention(
        const HostAttentionInputs &inputs,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        bool causal,
        const MPIContext &mpi_ctx)
    {
        ROCmAttentionRunResult result;
        result.output.assign(queryElementCount(seq_len, n_heads, head_dim), 0.0f);

        float *device_query = nullptr;
        float *device_key = nullptr;
        float *device_value = nullptr;
        float *d_output = nullptr;

        auto fail = [&](const std::string &message)
        {
            if (device_query)
                discardHipCleanupStatus(hipFree(device_query));
            if (device_key)
                discardHipCleanupStatus(hipFree(device_key));
            if (device_value)
                discardHipCleanupStatus(hipFree(device_value));
            if (d_output)
                discardHipCleanupStatus(hipFree(d_output));
            result.error = message;
            result.ok = false;
            return result;
        };

        hipError_t err = hipSetDevice(0);
        if (err != hipSuccess)
            return fail(std::string("hipSetDevice failed: ") + hipGetErrorString(err));

        const size_t query_bytes = inputs.query.size() * sizeof(float);
        const size_t key_value_bytes = inputs.key.size() * sizeof(float);
        const size_t output_bytes = result.output.size() * sizeof(float);

        err = hipMalloc(reinterpret_cast<void **>(&device_query), query_bytes);
        if (err != hipSuccess)
            return fail(std::string("hipMalloc(Q) failed: ") + hipGetErrorString(err));
        err = hipMalloc(reinterpret_cast<void **>(&device_key), key_value_bytes);
        if (err != hipSuccess)
            return fail(std::string("hipMalloc(K) failed: ") + hipGetErrorString(err));
        err = hipMalloc(reinterpret_cast<void **>(&device_value), key_value_bytes);
        if (err != hipSuccess)
            return fail(std::string("hipMalloc(V) failed: ") + hipGetErrorString(err));
        err = hipMalloc(reinterpret_cast<void **>(&d_output), output_bytes);
        if (err != hipSuccess)
            return fail(std::string("hipMalloc(output) failed: ") + hipGetErrorString(err));

        err = hipMemcpy(device_query, inputs.query.data(), query_bytes, hipMemcpyHostToDevice);
        if (err != hipSuccess)
            return fail(std::string("hipMemcpy(Q) failed: ") + hipGetErrorString(err));
        err = hipMemcpy(device_key, inputs.key.data(), key_value_bytes, hipMemcpyHostToDevice);
        if (err != hipSuccess)
            return fail(std::string("hipMemcpy(K) failed: ") + hipGetErrorString(err));
        err = hipMemcpy(device_value, inputs.value.data(), key_value_bytes, hipMemcpyHostToDevice);
        if (err != hipSuccess)
            return fail(std::string("hipMemcpy(V) failed: ") + hipGetErrorString(err));
        err = hipMemset(d_output, 0, output_bytes);
        if (err != hipSuccess)
            return fail(std::string("hipMemset(output) failed: ") + hipGetErrorString(err));

        llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> kernel(0);
        const bool kernel_ok = kernel.compute(
            device_query, device_key, device_value, d_output,
            seq_len, n_heads, n_kv_heads, head_dim,
            causal, -1,
            nullptr, nullptr, nullptr, nullptr,
            false, &mpi_ctx, 0);
        err = hipDeviceSynchronize();
        if (!kernel_ok)
            return fail("ROCm attention kernel returned false");
        if (err != hipSuccess)
            return fail(std::string("hipDeviceSynchronize failed: ") + hipGetErrorString(err));

        err = hipMemcpy(result.output.data(), d_output, output_bytes, hipMemcpyDeviceToHost);
        if (err != hipSuccess)
            return fail(std::string("hipMemcpy(output) failed: ") + hipGetErrorString(err));

        discardHipCleanupStatus(hipFree(device_query));
        discardHipCleanupStatus(hipFree(device_key));
        discardHipCleanupStatus(hipFree(device_value));
        discardHipCleanupStatus(hipFree(d_output));

        result.ok = true;
        return result;
    }
#endif
} // namespace

class Test__ROCmAttentionPaddingParity : public ::testing::Test
{
protected:
    MPIContext mpi_ctx_{0, 1, MPI_COMM_WORLD};
};

#ifdef HAVE_ROCM

TEST_F(Test__ROCmAttentionPaddingParity, CausalMaskPreventsHostilePaddingFromChangingRealRows)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    constexpr int real_seq_len = 48;
    constexpr int bucket_seq_len = 80;
    constexpr int n_heads = 4;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 32;
    const size_t prefix_count = queryElementCount(real_seq_len, n_heads, head_dim);

    const auto real = makeRealInputs(real_seq_len, n_heads, n_kv_heads, head_dim);
    const auto padded = makePaddedInputs(real, real_seq_len, bucket_seq_len, n_heads, n_kv_heads, head_dim);

    const auto exact = runROCmAttention(real, real_seq_len, n_heads, n_kv_heads, head_dim, true, mpi_ctx_);
    ASSERT_TRUE(exact.ok) << exact.error;
    const auto bucket = runROCmAttention(padded, bucket_seq_len, n_heads, n_kv_heads, head_dim, true, mpi_ctx_);
    ASSERT_TRUE(bucket.ok) << bucket.error;

    ASSERT_FALSE(hasNaNOrInf(exact.output));
    ASSERT_FALSE(hasNaNOrInf(bucket.output));

    const float max_diff = maxPrefixDiff(exact.output, bucket.output, prefix_count);
    const double cosine = prefixCosine(exact.output, bucket.output, prefix_count);

    EXPECT_LE(max_diff, 1e-3f) << "Causal real-row prefix changed when bucket-tail padding was present";
    EXPECT_GE(cosine, 0.999f) << "Causal real-row prefix lost cosine similarity under padding";
}

TEST_F(Test__ROCmAttentionPaddingParity, NonCausalAttentionExposesHostilePaddingRows)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    constexpr int real_seq_len = 16;
    constexpr int bucket_seq_len = 24;
    constexpr int n_heads = 4;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 32;
    const size_t prefix_count = queryElementCount(real_seq_len, n_heads, head_dim);

    const auto real = makeRealInputs(real_seq_len, n_heads, n_kv_heads, head_dim);
    const auto padded = makePaddedInputs(real, real_seq_len, bucket_seq_len, n_heads, n_kv_heads, head_dim);

    const auto exact = runROCmAttention(real, real_seq_len, n_heads, n_kv_heads, head_dim, false, mpi_ctx_);
    ASSERT_TRUE(exact.ok) << exact.error;
    const auto bucket = runROCmAttention(padded, bucket_seq_len, n_heads, n_kv_heads, head_dim, false, mpi_ctx_);
    ASSERT_TRUE(bucket.ok) << bucket.error;

    ASSERT_FALSE(hasNaNOrInf(exact.output));
    ASSERT_FALSE(hasNaNOrInf(bucket.output));

    const float max_diff = maxPrefixDiff(exact.output, bucket.output, prefix_count);

    EXPECT_GT(max_diff, 1.0f) << "Hostile pad rows did not affect non-causal real-row prefix";
}

TEST_F(Test__ROCmAttentionPaddingParity, LMHeadPaddedBucketUsesLastRealRowOnGPU)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    constexpr int real_seq_len = 5;
    constexpr int bucket_seq_len = 8;
    constexpr int d_model = 64;
    constexpr int vocab_size = 96;
    const DeviceId device = DeviceId::rocm(0);

    auto hidden_states = createDeterministicHiddenStates(bucket_seq_len, d_model, real_seq_len);
    auto lm_head_weight = createDeterministicLmHeadWeights(vocab_size, d_model);
    auto logits = std::make_unique<FP32Tensor>(
        std::vector<size_t>{1, static_cast<size_t>(vocab_size)},
        DeviceId::cpu());
    ASSERT_TRUE(lm_head_weight->ensureOnDevice(device));
    auto prepared_lm_head = makePreparedGemmFixture(lm_head_weight.get(), device, "output.weight");

    LMHeadStage::Params params;
    params.device_id = device;
    params.hidden_states = hidden_states.get();
    params.lm_head_weight = lm_head_weight.get();
    params.logits = logits.get();
    params.seq_len = bucket_seq_len;
    params.d_model = d_model;
    params.vocab_size = vocab_size;
    params.prepared_ref = prepared_lm_head.ref;
    params.prepared_store = prepared_lm_head.store.get();

    LMHeadStage stage(params);
    stage.updatePrefillReplayParams(IComputeStage::PrefillReplayParams{
        real_seq_len,
        bucket_seq_len,
        /*token_offset=*/128});
    ASSERT_EQ(stage.activationRowOffsetForLogits(), real_seq_len - 1);

    ASSERT_TRUE(with_gpu_coherence(
        device,
        {hidden_states.get(), lm_head_weight.get()},
        {logits.get()},
        [&]()
        { return stage.execute(nullptr); }));
    ASSERT_FALSE(hasNaNOrInf(std::vector<float>(logits->data(), logits->data() + vocab_size)));

    const auto last_real_reference = computeLmHeadReference(
        *hidden_states, *lm_head_weight, real_seq_len - 1, vocab_size, d_model);
    const auto bucket_tail_reference = computeLmHeadReference(
        *hidden_states, *lm_head_weight, bucket_seq_len - 1, vocab_size, d_model);

    EXPECT_LT(maxLogitDifference(*logits, last_real_reference), 8e-2f)
        << "ROCm LM-head logits should come from the last real row.";
    EXPECT_GT(maxLogitDifference(*logits, bucket_tail_reference), 1.0f)
        << "Hostile bucket-tail row should be visibly different from selected logits.";
}

TEST_F(Test__ROCmAttentionPaddingParity, DecodeContinuationUsesRealKVLengthOnGPU)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    constexpr int real_kv_len = 5;
    constexpr int bucket_kv_len = 9;
    constexpr int n_heads = 4;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 32;
    constexpr int q_dim = n_heads * head_dim;
    constexpr int kv_dim = n_kv_heads * head_dim;
    const DeviceId device = DeviceId::rocm(0);

    auto &gpu_ctx = GPUDeviceContextPool::instance().getContext(device);
    void *stream = gpu_ctx.defaultStream();

    auto query = std::make_unique<FP32Tensor>(std::vector<size_t>{1, static_cast<size_t>(q_dim)}, DeviceId::cpu());
    auto key = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(bucket_kv_len), static_cast<size_t>(kv_dim)}, DeviceId::cpu());
    auto value = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(bucket_kv_len), static_cast<size_t>(kv_dim)}, DeviceId::cpu());
    auto output = std::make_unique<FP32Tensor>(std::vector<size_t>{1, static_cast<size_t>(q_dim)}, DeviceId::cpu());
    auto workspace_scores = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(n_heads * bucket_kv_len)}, DeviceId::cpu());
    auto workspace_context = std::make_unique<FP32Tensor>(std::vector<size_t>{1, static_cast<size_t>(q_dim)}, DeviceId::cpu());
    auto workspace_mask = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(bucket_kv_len)}, DeviceId::cpu());

    for (int head = 0; head < n_heads; ++head)
    {
        for (int dim = 0; dim < head_dim; ++dim)
            query->mutable_data()[head * head_dim + dim] = 0.02f * static_cast<float>((head + 1) * (dim % 7 + 1));
    }

    for (int pos = 0; pos < bucket_kv_len; ++pos)
    {
        for (int kv_head = 0; kv_head < n_kv_heads; ++kv_head)
        {
            for (int dim = 0; dim < head_dim; ++dim)
            {
                const int idx = pos * kv_dim + kv_head * head_dim + dim;
                if (pos < real_kv_len)
                {
                    const float sign = ((pos + kv_head + dim) % 2 == 0) ? 1.0f : -1.0f;
                    key->mutable_data()[idx] = sign * 0.015f * static_cast<float>(1 + ((pos * 11 + kv_head * 5 + dim) % 9));
                    value->mutable_data()[idx] = 0.025f * static_cast<float>((pos + 1) * (kv_head + 1)) +
                                                 0.001f * static_cast<float>(dim);
                }
                else
                {
                    key->mutable_data()[idx] = 4.0f;
                    value->mutable_data()[idx] = 25.0f + static_cast<float>(pos - real_kv_len) * 3.0f +
                                                 static_cast<float>(kv_head) + 0.1f * static_cast<float>(dim);
                }
            }
        }
    }

    llaminar::v2::kernels::KVCacheConfig config;
    config.precision = ActivationPrecision::FP32;
    config.device = device;
    config.num_layers = 1;
    config.batch_size = 1;
    config.max_seq_len = 32;
    config.n_kv_heads = n_kv_heads;
    config.head_dim = head_dim;
    auto kv_cache = llaminar::v2::kernels::KernelFactory::createKVCache(config);
    ASSERT_NE(kv_cache, nullptr);

    ASSERT_TRUE(kv_cache->appendWithStream(0, 0, key.get(), value.get(), bucket_kv_len, stream));
    gpu_ctx.synchronizeStream(stream);
    ASSERT_EQ(kv_cache->get_cached_tokens(0, 0), bucket_kv_len);
    kv_cache->clear_sequence(0, 0);
    kv_cache->advanceHead(0, 0, real_kv_len);
    ASSERT_EQ(kv_cache->get_cached_tokens(0, 0), real_kv_len);

    const auto expected_real_prefix = referenceSingleQueryGQAAttention(
        query->data(), key->data(), value->data(), real_kv_len, n_heads, n_kv_heads, head_dim);
    const auto expected_full_bucket = referenceSingleQueryGQAAttention(
        query->data(), key->data(), value->data(), bucket_kv_len, n_heads, n_kv_heads, head_dim);

    AttentionComputeStage::Params params;
    params.device_id = device;
    params.Q = query.get();
    params.K = key.get();
    params.V = value.get();
    params.output = output.get();
    params.batch_size = 1;
    params.seq_len = 1;
    params.kv_len = bucket_kv_len;
    params.n_heads = n_heads;
    params.n_kv_heads = n_kv_heads;
    params.head_dim = head_dim;
    params.causal = true;
    params.auto_detect_mode = true;
    params.workspace_scores = workspace_scores.get();
    params.workspace_context = workspace_context.get();
    params.workspace_mask = workspace_mask.get();
    params.kv_cache = kv_cache.get();
    params.layer_idx = 0;
    params.read_kv_from_cache = true;
    params.position_offset = real_kv_len - 1;
    params.mpi_ctx = &mpi_ctx_;

    AttentionComputeStage stage(params);
    stage.setGPUStream(stream);
    auto *workspace_consumer = stage.getKernelAsWorkspaceConsumer();
    std::unique_ptr<DeviceWorkspaceManager> workspace_manager;
    if (workspace_consumer)
    {
        workspace_manager = std::make_unique<DeviceWorkspaceManager>(device, 64 * 1024 * 1024);
        auto requirements = workspace_consumer->getWorkspaceRequirements(1, n_heads, head_dim);
        ASSERT_TRUE(workspace_manager->allocate(requirements));
        workspace_consumer->bindWorkspace(workspace_manager.get());
    }
    ASSERT_TRUE(with_gpu_coherence(
        device,
        {query.get()},
        {output.get(), workspace_scores.get(), workspace_context.get(), workspace_mask.get()},
        [&]()
        { return stage.execute(nullptr); }));
    gpu_ctx.synchronizeStream(stream);

    EXPECT_LT(maxTensorVectorDiff(*output, expected_real_prefix), 2e-2f)
        << "ROCm decode attention should use the real KV prefix length.";
    EXPECT_GT(maxTensorVectorDiff(*output, expected_full_bucket), 5.0f)
        << "Hostile pad rows should materially change a stale full-bucket reference.";
}

#endif
