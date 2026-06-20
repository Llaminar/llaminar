#include <gtest/gtest.h>

#include "backends/BackendManager.h"
#include "backends/IBackend.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/DeviceId.h"
#include "execution/mtp/MTPRejectionSampler.h"
#include "kernels/common/SamplingMath.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using namespace llaminar2;

namespace
{
    constexpr int kQwen36Vocab = 248320;
    constexpr int kPartialCapacity = 1024;

    /**
     * @brief Simple RAII wrapper for backend-owned device allocations.
     *
     * This perf harness deliberately goes through IBackend allocation APIs
     * because it is testing backend sampling primitives outside a full
     * DeviceGraphOrchestrator arena. Production graph stages still use
     * BufferArena/workspace declarations.
     */
    class DeviceAllocation
    {
    public:
        DeviceAllocation(IBackend *backend, int device_id, size_t bytes)
            : backend_(backend), device_id_(device_id)
        {
            if (backend_ && bytes > 0)
                ptr_ = backend_->allocate(bytes, device_id_);
        }

        ~DeviceAllocation()
        {
            if (backend_ && ptr_)
                backend_->free(ptr_, device_id_);
        }

        DeviceAllocation(const DeviceAllocation &) = delete;
        DeviceAllocation &operator=(const DeviceAllocation &) = delete;

        void *get() const { return ptr_; }
        explicit operator bool() const { return ptr_ != nullptr; }

    private:
        IBackend *backend_ = nullptr;
        int device_id_ = 0;
        void *ptr_ = nullptr;
    };

    int envInt(const char *name, int fallback)
    {
        const char *raw = std::getenv(name);
        if (!raw || !*raw)
            return fallback;
        char *end = nullptr;
        const long parsed = std::strtol(raw, &end, 10);
        if (end == raw || parsed <= 0)
            return fallback;
        return static_cast<int>(parsed);
    }

    int stochasticHotToken(int row, int rank, int cols)
    {
        return (151936 + row * 911 + rank * 577 + (rank * rank * 17)) % cols;
    }

    std::vector<float> makeVerifierLogits(int rows, int cols, const std::vector<int> &expected_tokens)
    {
        std::vector<float> logits(static_cast<size_t>(rows) * static_cast<size_t>(cols), -12.0f);
        for (int row = 0; row < rows; ++row)
        {
            const int token = expected_tokens[static_cast<size_t>(row)];
            const size_t base = static_cast<size_t>(row) * static_cast<size_t>(cols);
            logits[base + static_cast<size_t>(token)] = 20.0f + static_cast<float>(row);
            logits[base + static_cast<size_t>((token + 17) % cols)] = 19.0f;
        }
        return logits;
    }

    /**
     * @brief Create Qwen-sized stochastic logits with a stable compact top-k set.
     *
     * Each row has a broad cold tail and a deterministic hot-token ladder.  The
     * target and draft distributions can be made identical from this fixture,
     * which lets the stochastic verifier accept every sampled draft token while
     * still exercising the real top-k/top-p distribution kernels.
     */
    std::vector<float> makeStochasticLogits(int rows, int cols, int top_k)
    {
        std::vector<float> logits(static_cast<size_t>(rows) * static_cast<size_t>(cols), -18.0f);
        for (int row = 0; row < rows; ++row)
        {
            const size_t base = static_cast<size_t>(row) * static_cast<size_t>(cols);
            for (int i = 0; i < cols; ++i)
            {
                logits[base + static_cast<size_t>(i)] -=
                    0.00019f * static_cast<float>(((i + row * 131) * 47) % 997);
            }
            for (int rank = 0; rank < top_k; ++rank)
            {
                const int token = stochasticHotToken(row, rank, cols);
                logits[base + static_cast<size_t>(token)] =
                    9.0f - 0.061f * static_cast<float>(rank) +
                    0.017f * static_cast<float>(row);
            }
        }
        return logits;
    }

    /**
     * @brief Create target logits whose stochastic verifier accepts a fixed prefix.
     *
     * The draft logits keep the normal Qwen-style hot-token ladder.  For the
     * first row after the accepted prefix we remove every draft hot token from
     * the target top-k set and install a disjoint hot-token ladder.  That makes
     * the production speculative-verify kernel reject at a deterministic row
     * without adding a test-only verification path.
     */
    std::vector<float> makeStochasticTargetLogitsForAcceptedPrefix(
        int rows,
        int cols,
        int top_k,
        int accepted_prefix)
    {
        std::vector<float> logits = makeStochasticLogits(rows, cols, top_k);
        const int reject_row = accepted_prefix;
        if (reject_row < 0 || reject_row >= rows - 1)
            return logits;

        const size_t base = static_cast<size_t>(reject_row) * static_cast<size_t>(cols);
        for (int rank = 0; rank < top_k; ++rank)
        {
            const int draft_token = stochasticHotToken(reject_row, rank, cols);
            logits[base + static_cast<size_t>(draft_token)] = -22.0f;
        }
        for (int rank = 0; rank < top_k; ++rank)
        {
            const int target_token =
                (190000 + reject_row * 997 + rank * 313 + (rank * rank * 11)) % cols;
            logits[base + static_cast<size_t>(target_token)] =
                9.5f - 0.055f * static_cast<float>(rank);
        }
        return logits;
    }

    void expectGreedySummary(
        const std::array<int32_t, sampling_math::kSpeculativeBatchMaxOutputTokens> &tokens,
        const std::array<int, sampling_math::kSpeculativeBatchMetaCount> &meta,
        int first_token,
        const std::vector<int> &verifier_tokens,
        const std::vector<int> &draft_tokens,
        int compare_rows)
    {
        std::array<int32_t, sampling_math::kSpeculativeBatchMaxOutputTokens> expected_tokens{};
        std::array<int, sampling_math::kSpeculativeBatchMetaCount> expected_meta{};
        sampling_math::summarize_greedy_speculative_verify_batch(
            first_token,
            verifier_tokens.data(),
            draft_tokens.data(),
            compare_rows,
            /*stop_tokens=*/nullptr,
            /*stop_token_count=*/0,
            expected_tokens.data(),
            expected_meta.data());

        for (int i = 0; i < sampling_math::kSpeculativeBatchMetaCount; ++i)
            EXPECT_EQ(meta[static_cast<size_t>(i)], expected_meta[static_cast<size_t>(i)])
                << "meta[" << i << "]";
        for (int i = 0; i < sampling_math::kSpeculativeBatchMaxOutputTokens; ++i)
            EXPECT_EQ(tokens[static_cast<size_t>(i)], expected_tokens[static_cast<size_t>(i)])
                << "tokens[" << i << "]";
    }

    void runCase(const std::string &backend_name, IBackend *backend, DeviceId device, int rows)
    {
        if (!backend)
            GTEST_SKIP() << backend_name << " backend unavailable";
        ASSERT_TRUE(device.is_gpu());

        const int device_id = device.gpu_ordinal();
        auto &ctx = GPUDeviceContextPool::instance().getContext(device);
        void *stream = ctx.defaultStream();
        ASSERT_NE(stream, nullptr) << "GPU speculative summary perf requires an explicit stream";

        const int compare_rows = rows - 1;
        ASSERT_GE(compare_rows, 1);
        ASSERT_LE(rows, static_cast<int>(sampling_math::kSpeculativeBatchMaxRows));

        std::vector<int> draft_tokens(static_cast<size_t>(rows));
        draft_tokens[0] = 100;
        for (int i = 1; i < rows; ++i)
            draft_tokens[static_cast<size_t>(i)] = 100 + i;

        std::vector<int> verifier_tokens(static_cast<size_t>(rows));
        for (int row = 0; row < compare_rows; ++row)
            verifier_tokens[static_cast<size_t>(row)] = draft_tokens[static_cast<size_t>(row + 1)];
        verifier_tokens[static_cast<size_t>(compare_rows)] = 9000 + rows;

        std::vector<float> logits = makeVerifierLogits(rows, kQwen36Vocab, verifier_tokens);

        DeviceAllocation d_logits(
            backend,
            device_id,
            logits.size() * sizeof(float));
        DeviceAllocation d_draft_tokens(
            backend,
            device_id,
            draft_tokens.size() * sizeof(int));
        DeviceAllocation d_argmax_values(
            backend,
            device_id,
            static_cast<size_t>(rows) * sizeof(float));
        DeviceAllocation d_argmax_indices(
            backend,
            device_id,
            static_cast<size_t>(rows) * sizeof(int));
        DeviceAllocation d_partial_values(
            backend,
            device_id,
            kPartialCapacity * sizeof(float));
        DeviceAllocation d_partial_indices(
            backend,
            device_id,
            kPartialCapacity * sizeof(int));
        DeviceAllocation d_output_tokens(
            backend,
            device_id,
            sampling_math::kSpeculativeBatchMaxOutputTokens * sizeof(int));
        DeviceAllocation d_output_meta(
            backend,
            device_id,
            sampling_math::kSpeculativeBatchMetaCount * sizeof(int));

        ASSERT_TRUE(d_logits);
        ASSERT_TRUE(d_draft_tokens);
        ASSERT_TRUE(d_argmax_values);
        ASSERT_TRUE(d_argmax_indices);
        ASSERT_TRUE(d_partial_values);
        ASSERT_TRUE(d_partial_indices);
        ASSERT_TRUE(d_output_tokens);
        ASSERT_TRUE(d_output_meta);

        ASSERT_TRUE(backend->hostToDevice(
            d_logits.get(),
            logits.data(),
            logits.size() * sizeof(float),
            device_id,
            stream));
        ASSERT_TRUE(backend->synchronizeStream(stream, device_id));

        const auto run_once = [&]()
        {
            if (!backend->hostToDeviceOnStream(
                    d_draft_tokens.get(),
                    draft_tokens.data(),
                    draft_tokens.size() * sizeof(int),
                    device_id,
                    stream))
            {
                return false;
            }
            if (!backend->enqueueArgmaxF32BatchedRowsDevice(
                    d_logits.get(),
                    rows,
                    kQwen36Vocab,
                    device_id,
                    stream,
                    d_argmax_values.get(),
                    d_argmax_indices.get(),
                    d_partial_values.get(),
                    d_partial_indices.get(),
                    kPartialCapacity))
            {
                return false;
            }
            if (!backend->enqueueSummarizeGreedySpeculativeVerifyBatch(
                    d_argmax_indices.get(),
                    d_draft_tokens.get(),
                    compare_rows,
                    draft_tokens[0],
                    /*stop_tokens_host=*/nullptr,
                    /*stop_token_count=*/0,
                    device_id,
                    stream,
                    d_output_tokens.get(),
                    d_output_meta.get()))
            {
                return false;
            }
            return true;
        };

        ASSERT_TRUE(run_once());
        std::array<int32_t, sampling_math::kSpeculativeBatchMaxOutputTokens> output_tokens{};
        std::array<int, sampling_math::kSpeculativeBatchMetaCount> output_meta{};
        ASSERT_TRUE(backend->deviceToHostFast(
            output_tokens.data(),
            d_output_tokens.get(),
            output_tokens.size() * sizeof(int32_t),
            device_id,
            stream));
        ASSERT_TRUE(backend->deviceToHostFast(
            output_meta.data(),
            d_output_meta.get(),
            output_meta.size() * sizeof(int),
            device_id,
            stream));
        expectGreedySummary(
            output_tokens,
            output_meta,
            draft_tokens[0],
            verifier_tokens,
            draft_tokens,
            compare_rows);

        const int warmup = envInt("LLAMINAR_PERF_GPU_SPEC_SUMMARY_WARMUP", 10);
        const int iterations = envInt("LLAMINAR_PERF_GPU_SPEC_SUMMARY_ITERS", 100);
        for (int i = 0; i < warmup; ++i)
        {
            ASSERT_TRUE(run_once());
            ASSERT_TRUE(backend->synchronizeStream(stream, device_id));
        }

        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i)
        {
            ASSERT_TRUE(run_once());
            ASSERT_TRUE(backend->deviceToHostFast(
                output_tokens.data(),
                d_output_tokens.get(),
                output_tokens.size() * sizeof(int32_t),
                device_id,
                stream));
            ASSERT_TRUE(backend->deviceToHostFast(
                output_meta.data(),
                d_output_meta.get(),
                output_meta.size() * sizeof(int),
                device_id,
                stream));
        }
        const auto end = std::chrono::steady_clock::now();
        const double total_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        const double avg_us = (total_ms * 1000.0) / static_cast<double>(iterations);

        std::cout << "backend,case,rows,cols,iterations,total_ms,avg_us\n"
                  << backend_name << ",greedy_spec_summary,"
                  << rows << ',' << kQwen36Vocab << ','
                  << iterations << ','
                  << std::fixed << std::setprecision(3) << total_ms << ','
                  << std::fixed << std::setprecision(3) << avg_us << '\n';
    }

    struct StochasticComponentTimes
    {
        double target_build_ms = 0.0;
        double draft_build_ms = 0.0;
        double draft_sample_ms = 0.0;
        double verify_ms = 0.0;
        double bonus_sample_ms = 0.0;
        double summary_ms = 0.0;
        double d2h_ms = 0.0;
    };

    enum class StochasticDraftProposalMode
    {
        ProbabilityRows,
        DraftLogits,
        GreedyNoDraftProbabilities,
    };

    const char *stochasticDraftProposalModeName(StochasticDraftProposalMode mode)
    {
        switch (mode)
        {
        case StochasticDraftProposalMode::DraftLogits:
            return "stochastic_processed_target_draft_logits";
        case StochasticDraftProposalMode::GreedyNoDraftProbabilities:
            return "stochastic_processed_target_greedy_draft";
        case StochasticDraftProposalMode::ProbabilityRows:
        default:
            return "stochastic_processed_target_draft_probability";
        }
    }

    void runStochasticCase(const std::string &backend_name, IBackend *backend, DeviceId device, int row_count)
    {
        if (!backend)
            GTEST_SKIP() << backend_name << " backend unavailable";
        ASSERT_TRUE(device.is_gpu());

        constexpr int kOutStride = 64;
        constexpr float kTopP = 0.95f;
        constexpr float kTemperature = 0.6f;
        constexpr int kFirstToken = 100;
        constexpr int kScratchBlocksPerRow = 128;
        const int kTopK = envInt("LLAMINAR_PERF_GPU_SPEC_TOP_K", 40);
        ASSERT_GE(kTopK, 1);
        ASSERT_LE(kTopK, kOutStride)
            << "compact distribution perf output stride is fixed for small-k sweeps";

        const int device_id = device.gpu_ordinal();
        auto &ctx = GPUDeviceContextPool::instance().getContext(device);
        void *stream = ctx.defaultStream();
        ASSERT_NE(stream, nullptr) << "GPU stochastic spec perf requires an explicit stream";

        ASSERT_GE(row_count, 1);
        ASSERT_LE(row_count, static_cast<int>(sampling_math::kSpeculativeBatchMaxRows));
        const int target_rows = row_count + 1;
        const int scratch_capacity = target_rows * kScratchBlocksPerRow * kTopK;

        std::vector<float> target_logits =
            makeStochasticLogits(target_rows, kQwen36Vocab, kTopK);
        std::vector<float> draft_logits(
            target_logits.begin(),
            target_logits.begin() + static_cast<ptrdiff_t>(
                static_cast<size_t>(row_count) * kQwen36Vocab));

        DeviceAllocation d_target_logits(
            backend,
            device_id,
            target_logits.size() * sizeof(float));
        DeviceAllocation d_draft_logits(
            backend,
            device_id,
            draft_logits.size() * sizeof(float));
        DeviceAllocation d_target_ids(
            backend,
            device_id,
            static_cast<size_t>(target_rows) * kOutStride * sizeof(int));
        DeviceAllocation d_target_probs(
            backend,
            device_id,
            static_cast<size_t>(target_rows) * kOutStride * sizeof(float));
        DeviceAllocation d_draft_ids(
            backend,
            device_id,
            static_cast<size_t>(row_count) * kOutStride * sizeof(int));
        DeviceAllocation d_draft_probs(
            backend,
            device_id,
            static_cast<size_t>(row_count) * kOutStride * sizeof(float));
        DeviceAllocation d_scratch_values(
            backend,
            device_id,
            static_cast<size_t>(scratch_capacity) * sizeof(float));
        DeviceAllocation d_scratch_indices(
            backend,
            device_id,
            static_cast<size_t>(scratch_capacity) * sizeof(int));
        DeviceAllocation d_sampled_draft_tokens(
            backend,
            device_id,
            static_cast<size_t>(row_count) * sizeof(int));
        DeviceAllocation d_sampled_draft_probabilities(
            backend,
            device_id,
            static_cast<size_t>(row_count) * sizeof(float));
        DeviceAllocation d_verify_tokens(
            backend,
            device_id,
            static_cast<size_t>(target_rows) * sizeof(int));
        DeviceAllocation d_verify_accepted(
            backend,
            device_id,
            static_cast<size_t>(row_count) * sizeof(int));
        DeviceAllocation d_verify_accept_prob(
            backend,
            device_id,
            static_cast<size_t>(row_count) * sizeof(float));
        DeviceAllocation d_verify_accept_threshold(
            backend,
            device_id,
            static_cast<size_t>(row_count) * sizeof(float));
        DeviceAllocation d_output_tokens(
            backend,
            device_id,
            sampling_math::kSpeculativeBatchMaxOutputTokens * sizeof(int));
        DeviceAllocation d_output_meta(
            backend,
            device_id,
            sampling_math::kSpeculativeBatchMetaCount * sizeof(int));

        ASSERT_TRUE(d_target_logits);
        ASSERT_TRUE(d_draft_logits);
        ASSERT_TRUE(d_target_ids);
        ASSERT_TRUE(d_target_probs);
        ASSERT_TRUE(d_draft_ids);
        ASSERT_TRUE(d_draft_probs);
        ASSERT_TRUE(d_scratch_values);
        ASSERT_TRUE(d_scratch_indices);
        ASSERT_TRUE(d_sampled_draft_tokens);
        ASSERT_TRUE(d_sampled_draft_probabilities);
        ASSERT_TRUE(d_verify_tokens);
        ASSERT_TRUE(d_verify_accepted);
        ASSERT_TRUE(d_verify_accept_prob);
        ASSERT_TRUE(d_verify_accept_threshold);
        ASSERT_TRUE(d_output_tokens);
        ASSERT_TRUE(d_output_meta);

        ASSERT_TRUE(backend->hostToDevice(
            d_target_logits.get(),
            target_logits.data(),
            target_logits.size() * sizeof(float),
            device_id,
            stream));
        ASSERT_TRUE(backend->hostToDevice(
            d_draft_logits.get(),
            draft_logits.data(),
            draft_logits.size() * sizeof(float),
            device_id,
            stream));
        ASSERT_TRUE(backend->synchronizeStream(stream, device_id));

        std::array<float, sampling_math::kSpeculativeBatchMaxRows> accept_thresholds{};
        std::array<float, sampling_math::kSpeculativeBatchMaxRows> residual_thresholds{};
        std::fill(accept_thresholds.begin(), accept_thresholds.end(), 0.0f);
        std::fill(residual_thresholds.begin(), residual_thresholds.end(), 0.0f);

        std::array<int32_t, sampling_math::kSpeculativeBatchMaxOutputTokens> output_tokens{};
        std::array<int, sampling_math::kSpeculativeBatchMetaCount> output_meta{};

        auto sample_threshold = [](int row) -> float
        {
            return 0.11f + 0.17f * static_cast<float>(row % 5);
        };

        auto run_pipeline = [&]() -> bool
        {
            if (!backend->enqueueBuildTopKTopPDistributionsF32Device(
                    d_target_logits.get(),
                    target_rows,
                    kQwen36Vocab,
                    kQwen36Vocab,
                    kTopK,
                    kTopP,
                    kTemperature,
                    device_id,
                    stream,
                    d_target_ids.get(),
                    kOutStride,
                    d_target_probs.get(),
                    d_scratch_values.get(),
                    d_scratch_indices.get(),
                    scratch_capacity))
            {
                return false;
            }
            if (!backend->enqueueBuildTopKTopPDistributionsF32Device(
                    d_draft_logits.get(),
                    row_count,
                    kQwen36Vocab,
                    kQwen36Vocab,
                    kTopK,
                    kTopP,
                    kTemperature,
                    device_id,
                    stream,
                    d_draft_ids.get(),
                    kOutStride,
                    d_draft_probs.get(),
                    d_scratch_values.get(),
                    d_scratch_indices.get(),
                    scratch_capacity))
            {
                return false;
            }
            for (int row = 0; row < row_count; ++row)
            {
                if (!backend->enqueueSampleDistributionF32Device(
                        static_cast<int *>(d_draft_ids.get()) + row * kOutStride,
                        static_cast<float *>(d_draft_probs.get()) + row * kOutStride,
                        kTopK,
                        sample_threshold(row),
                        device_id,
                        stream,
                        static_cast<int *>(d_sampled_draft_tokens.get()) + row,
                        static_cast<float *>(d_sampled_draft_probabilities.get()) + row))
                {
                    return false;
                }
            }
            if (!backend->enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatchDeviceTokens(
                    d_target_ids.get(),
                    d_target_probs.get(),
                    d_draft_ids.get(),
                    d_draft_probs.get(),
                    kTopK,
                    kOutStride,
                    d_sampled_draft_tokens.get(),
                    accept_thresholds.data(),
                    residual_thresholds.data(),
                    row_count,
                    device_id,
                    stream,
                    d_verify_tokens.get(),
                    d_verify_accepted.get(),
                    d_verify_accept_prob.get(),
                    d_verify_accept_threshold.get(),
                    d_sampled_draft_probabilities.get()))
            {
                return false;
            }
            if (!backend->enqueueSampleDistributionF32Device(
                    static_cast<int *>(d_target_ids.get()) + row_count * kOutStride,
                    static_cast<float *>(d_target_probs.get()) + row_count * kOutStride,
                    kTopK,
                    0.42f,
                    device_id,
                    stream,
                    static_cast<int *>(d_verify_tokens.get()) + row_count))
            {
                return false;
            }
            if (!backend->enqueueSummarizeSpeculativeVerifyBatch(
                    d_verify_tokens.get(),
                    d_verify_accepted.get(),
                    row_count,
                    kFirstToken,
                    /*stop_tokens_host=*/nullptr,
                    /*stop_token_count=*/0,
                    static_cast<int *>(d_verify_tokens.get()) + row_count,
                    /*has_bonus_token=*/true,
                    device_id,
                    stream,
                    d_output_tokens.get(),
                    d_output_meta.get()))
            {
                return false;
            }
            return true;
        };

        ASSERT_TRUE(run_pipeline());
        ASSERT_TRUE(backend->deviceToHostFast(
            output_tokens.data(),
            d_output_tokens.get(),
            output_tokens.size() * sizeof(int32_t),
            device_id,
            stream));
        ASSERT_TRUE(backend->deviceToHostFast(
            output_meta.data(),
            d_output_meta.get(),
            output_meta.size() * sizeof(int),
            device_id,
            stream));
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaOk], 1);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaOutputCount], row_count + 1);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaAcceptedSpeculativePrefix], row_count);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaAllSpeculativeAccepted], 1);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaConsumedVerifierRows], row_count);
        EXPECT_EQ(output_tokens[0], kFirstToken);

        const int warmup = envInt("LLAMINAR_PERF_GPU_SPEC_STOCHASTIC_WARMUP", 5);
        const int iterations = envInt("LLAMINAR_PERF_GPU_SPEC_STOCHASTIC_ITERS", 50);
        for (int i = 0; i < warmup; ++i)
        {
            ASSERT_TRUE(run_pipeline());
            ASSERT_TRUE(backend->synchronizeStream(stream, device_id));
        }

        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i)
        {
            ASSERT_TRUE(run_pipeline());
            ASSERT_TRUE(backend->deviceToHostFast(
                output_tokens.data(),
                d_output_tokens.get(),
                output_tokens.size() * sizeof(int32_t),
                device_id,
                stream));
            ASSERT_TRUE(backend->deviceToHostFast(
                output_meta.data(),
                d_output_meta.get(),
                output_meta.size() * sizeof(int),
                device_id,
                stream));
        }
        const auto end = std::chrono::steady_clock::now();
        const double total_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        const double avg_us = (total_ms * 1000.0) / static_cast<double>(iterations);

        StochasticComponentTimes parts{};
        auto timed_sync = [&](auto &&fn) -> double
        {
            const auto t0 = std::chrono::steady_clock::now();
            EXPECT_TRUE(fn());
            EXPECT_TRUE(backend->synchronizeStream(stream, device_id));
            const auto t1 = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count();
        };

        parts.target_build_ms = timed_sync([&]()
        {
            return backend->enqueueBuildTopKTopPDistributionsF32Device(
                d_target_logits.get(), target_rows, kQwen36Vocab, kQwen36Vocab,
                kTopK, kTopP, kTemperature, device_id, stream,
                d_target_ids.get(), kOutStride, d_target_probs.get(),
                d_scratch_values.get(), d_scratch_indices.get(), scratch_capacity);
        });
        parts.draft_build_ms = timed_sync([&]()
        {
            return backend->enqueueBuildTopKTopPDistributionsF32Device(
                d_draft_logits.get(), row_count, kQwen36Vocab, kQwen36Vocab,
                kTopK, kTopP, kTemperature, device_id, stream,
                d_draft_ids.get(), kOutStride, d_draft_probs.get(),
                d_scratch_values.get(), d_scratch_indices.get(), scratch_capacity);
        });
        parts.draft_sample_ms = timed_sync([&]()
        {
            bool ok = true;
            for (int row = 0; row < row_count; ++row)
            {
                ok = ok && backend->enqueueSampleDistributionF32Device(
                    static_cast<int *>(d_draft_ids.get()) + row * kOutStride,
                    static_cast<float *>(d_draft_probs.get()) + row * kOutStride,
                    kTopK, sample_threshold(row), device_id, stream,
                    static_cast<int *>(d_sampled_draft_tokens.get()) + row,
                    static_cast<float *>(d_sampled_draft_probabilities.get()) + row);
            }
            return ok;
        });
        parts.verify_ms = timed_sync([&]()
        {
            return backend->enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatchDeviceTokens(
                d_target_ids.get(), d_target_probs.get(),
                d_draft_ids.get(), d_draft_probs.get(),
                kTopK, kOutStride, d_sampled_draft_tokens.get(),
                accept_thresholds.data(), residual_thresholds.data(),
                row_count, device_id, stream,
                d_verify_tokens.get(), d_verify_accepted.get(),
                d_verify_accept_prob.get(), d_verify_accept_threshold.get(),
                d_sampled_draft_probabilities.get());
        });
        parts.bonus_sample_ms = timed_sync([&]()
        {
            return backend->enqueueSampleDistributionF32Device(
                static_cast<int *>(d_target_ids.get()) + row_count * kOutStride,
                static_cast<float *>(d_target_probs.get()) + row_count * kOutStride,
                kTopK, 0.42f, device_id, stream,
                static_cast<int *>(d_verify_tokens.get()) + row_count);
        });
        parts.summary_ms = timed_sync([&]()
        {
            return backend->enqueueSummarizeSpeculativeVerifyBatch(
                d_verify_tokens.get(), d_verify_accepted.get(), row_count,
                kFirstToken, nullptr, 0,
                static_cast<int *>(d_verify_tokens.get()) + row_count,
                true, device_id, stream,
                d_output_tokens.get(), d_output_meta.get());
        });

        const auto d2h_start = std::chrono::steady_clock::now();
        ASSERT_TRUE(backend->deviceToHostFast(
            output_tokens.data(),
            d_output_tokens.get(),
            output_tokens.size() * sizeof(int32_t),
            device_id,
            stream));
        ASSERT_TRUE(backend->deviceToHostFast(
            output_meta.data(),
            d_output_meta.get(),
            output_meta.size() * sizeof(int),
            device_id,
            stream));
        const auto d2h_end = std::chrono::steady_clock::now();
        parts.d2h_ms =
            std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count();

        std::cout << "backend,case,rows,cols,top_k,iterations,total_ms,avg_us,"
                     "target_build_ms,draft_build_ms,draft_sample_ms,verify_ms,"
                     "bonus_sample_ms,summary_ms,d2h_ms\n"
                  << backend_name << ",stochastic_spec_batch,"
                  << row_count << ',' << kQwen36Vocab << ','
                  << kTopK << ','
                  << iterations << ','
                  << std::fixed << std::setprecision(3) << total_ms << ','
                  << std::fixed << std::setprecision(3) << avg_us << ','
                  << parts.target_build_ms << ','
                  << parts.draft_build_ms << ','
                  << parts.draft_sample_ms << ','
                  << parts.verify_ms << ','
                  << parts.bonus_sample_ms << ','
                  << parts.summary_ms << ','
                  << parts.d2h_ms << '\n';
    }

    void runStochasticProcessedLogitCase(
        const std::string &backend_name,
        IBackend *backend,
        DeviceId device,
        int row_count,
        int accepted_prefix)
    {
        if (!backend)
            GTEST_SKIP() << backend_name << " backend unavailable";
        ASSERT_TRUE(device.is_gpu());

        constexpr int kFirstToken = 100;
        constexpr float kTopP = 0.95f;
        constexpr float kTemperature = 0.6f;
        constexpr float kAcceptThreshold = 0.5f;
        constexpr float kResidualThreshold = 0.42f;
        constexpr float kBonusThreshold = 0.42f;
        constexpr int kScratchBlocksPerRow = 128;
        const int kTopK = envInt("LLAMINAR_PERF_GPU_SPEC_TOP_K", 40);
        ASSERT_GE(kTopK, 1);
        ASSERT_LE(kTopK, 64)
            << "processed-logit perf sweeps are intended for small-k MTP sampling";

        const int device_id = device.gpu_ordinal();
        auto &ctx = GPUDeviceContextPool::instance().getContext(device);
        void *stream = ctx.defaultStream();
        ASSERT_NE(stream, nullptr)
            << "GPU processed-logit stochastic perf requires an explicit stream";

        ASSERT_GE(row_count, 1);
        ASSERT_LE(row_count, static_cast<int>(sampling_math::kSpeculativeBatchMaxRows));
        ASSERT_GE(accepted_prefix, 0);
        ASSERT_LE(accepted_prefix, row_count);

        const int target_rows = row_count + 1;
        std::vector<float> target_logits =
            makeStochasticTargetLogitsForAcceptedPrefix(
                target_rows, kQwen36Vocab, kTopK, accepted_prefix);
        std::vector<float> draft_logits =
            makeStochasticLogits(row_count, kQwen36Vocab, kTopK);

        std::array<float, sampling_math::kSpeculativeBatchMaxRows> accept_thresholds{};
        std::array<float, sampling_math::kSpeculativeBatchMaxRows> residual_thresholds{};
        std::fill(accept_thresholds.begin(), accept_thresholds.end(), kAcceptThreshold);
        std::fill(residual_thresholds.begin(), residual_thresholds.end(), kResidualThreshold);

        const int scratch_capacity = target_rows * kScratchBlocksPerRow * kTopK;

        DeviceAllocation d_target_raw_logits(
            backend, device_id, target_logits.size() * sizeof(float));
        DeviceAllocation d_draft_raw_logits(
            backend, device_id, draft_logits.size() * sizeof(float));
        DeviceAllocation d_target_logits(
            backend, device_id, target_logits.size() * sizeof(float));
        DeviceAllocation d_draft_logits(
            backend, device_id, draft_logits.size() * sizeof(float));
        DeviceAllocation d_scratch_values(
            backend, device_id,
            static_cast<size_t>(scratch_capacity) * sizeof(float));
        DeviceAllocation d_scratch_indices(
            backend, device_id,
            static_cast<size_t>(scratch_capacity) * sizeof(int));
        DeviceAllocation d_sampled_draft_tokens(
            backend, device_id, static_cast<size_t>(row_count) * sizeof(int));
        DeviceAllocation d_sampled_draft_probabilities(
            backend, device_id, static_cast<size_t>(row_count) * sizeof(float));
        DeviceAllocation d_verify_tokens(
            backend, device_id, static_cast<size_t>(target_rows) * sizeof(int));
        DeviceAllocation d_verify_accepted(
            backend, device_id, static_cast<size_t>(row_count) * sizeof(int));
        DeviceAllocation d_verify_accept_prob(
            backend, device_id, static_cast<size_t>(row_count) * sizeof(float));
        DeviceAllocation d_verify_accept_threshold(
            backend, device_id, static_cast<size_t>(row_count) * sizeof(float));
        DeviceAllocation d_bonus_token(
            backend, device_id, sizeof(int));
        DeviceAllocation d_output_tokens(
            backend, device_id,
            sampling_math::kSpeculativeBatchMaxOutputTokens * sizeof(int));
        DeviceAllocation d_output_meta(
            backend, device_id,
            sampling_math::kSpeculativeBatchMetaCount * sizeof(int));

        ASSERT_TRUE(d_target_raw_logits);
        ASSERT_TRUE(d_draft_raw_logits);
        ASSERT_TRUE(d_target_logits);
        ASSERT_TRUE(d_draft_logits);
        ASSERT_TRUE(d_scratch_values);
        ASSERT_TRUE(d_scratch_indices);
        ASSERT_TRUE(d_sampled_draft_tokens);
        ASSERT_TRUE(d_sampled_draft_probabilities);
        ASSERT_TRUE(d_verify_tokens);
        ASSERT_TRUE(d_verify_accepted);
        ASSERT_TRUE(d_verify_accept_prob);
        ASSERT_TRUE(d_verify_accept_threshold);
        ASSERT_TRUE(d_bonus_token);
        ASSERT_TRUE(d_output_tokens);
        ASSERT_TRUE(d_output_meta);

        ASSERT_TRUE(backend->hostToDevice(
            d_target_raw_logits.get(),
            target_logits.data(),
            target_logits.size() * sizeof(float),
            device_id,
            stream));
        ASSERT_TRUE(backend->hostToDevice(
            d_draft_raw_logits.get(),
            draft_logits.data(),
            draft_logits.size() * sizeof(float),
            device_id,
            stream));
        ASSERT_TRUE(backend->synchronizeStream(stream, device_id));

        std::array<int32_t, sampling_math::kSpeculativeBatchMaxOutputTokens> output_tokens{};
        std::array<int, sampling_math::kSpeculativeBatchMetaCount> output_meta{};

        auto run_pipeline = [&]() -> bool
        {
            if (!backend->enqueueBuildTopKTopPProcessedLogitsF32Device(
                    d_target_raw_logits.get(),
                    target_rows,
                    kQwen36Vocab,
                    kQwen36Vocab,
                    kTopK,
                    kTopP,
                    kTemperature,
                    device_id,
                    stream,
                    d_target_logits.get(),
                    kQwen36Vocab,
                    d_scratch_values.get(),
                    d_scratch_indices.get(),
                    scratch_capacity))
            {
                return false;
            }
            if (!backend->enqueueBuildTopKTopPProcessedLogitsF32Device(
                    d_draft_raw_logits.get(),
                    row_count,
                    kQwen36Vocab,
                    kQwen36Vocab,
                    kTopK,
                    kTopP,
                    kTemperature,
                    device_id,
                    stream,
                    d_draft_logits.get(),
                    kQwen36Vocab,
                    d_scratch_values.get(),
                    d_scratch_indices.get(),
                    scratch_capacity))
            {
                return false;
            }
            for (int row = 0; row < row_count; ++row)
            {
                if (!backend->enqueueSampleProcessedLogitsF32Device(
                        static_cast<float *>(d_draft_logits.get()) +
                            static_cast<size_t>(row) * kQwen36Vocab,
                        kQwen36Vocab,
                        kQwen36Vocab,
                        0.11f + 0.17f * static_cast<float>(row % 5),
                        device_id,
                        stream,
                        static_cast<int *>(d_sampled_draft_tokens.get()) + row,
                        static_cast<float *>(d_sampled_draft_probabilities.get()) + row))
                {
                    return false;
                }
            }
            if (!backend->enqueueSpeculativeVerifyProcessedLogitsF32DeviceThresholdsBatchDeviceTokens(
                    d_target_logits.get(),
                    d_draft_logits.get(),
                    row_count,
                    kQwen36Vocab,
                    kQwen36Vocab,
                    kQwen36Vocab,
                    d_sampled_draft_tokens.get(),
                    accept_thresholds.data(),
                    residual_thresholds.data(),
                    device_id,
                    stream,
                    d_verify_tokens.get(),
                    d_verify_accepted.get(),
                    d_verify_accept_prob.get(),
                    d_verify_accept_threshold.get(),
                    d_sampled_draft_probabilities.get()))
            {
                return false;
            }

            if (accepted_prefix == row_count)
            {
                if (!backend->enqueueSampleProcessedLogitsF32Device(
                        static_cast<float *>(d_target_logits.get()) +
                            static_cast<size_t>(row_count) * kQwen36Vocab,
                        kQwen36Vocab,
                        kQwen36Vocab,
                        kBonusThreshold,
                        device_id,
                        stream,
                        d_bonus_token.get()))
                {
                    return false;
                }
            }

            return backend->enqueueSummarizeSpeculativeVerifyBatch(
                d_verify_tokens.get(),
                d_verify_accepted.get(),
                row_count,
                kFirstToken,
                nullptr,
                0,
                accepted_prefix == row_count
                    ? static_cast<int *>(d_bonus_token.get())
                    : nullptr,
                accepted_prefix == row_count,
                device_id,
                stream,
                d_output_tokens.get(),
                d_output_meta.get());
        };

        ASSERT_TRUE(run_pipeline());
        ASSERT_TRUE(backend->deviceToHostFast(
            output_tokens.data(),
            d_output_tokens.get(),
            output_tokens.size() * sizeof(int32_t),
            device_id,
            stream));
        ASSERT_TRUE(backend->deviceToHostFast(
            output_meta.data(),
            d_output_meta.get(),
            output_meta.size() * sizeof(int),
            device_id,
            stream));
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaOk], 1);
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaAcceptedSpeculativePrefix],
                  accepted_prefix);
        EXPECT_EQ(output_tokens[0], kFirstToken);

        const int warmup = envInt("LLAMINAR_PERF_GPU_SPEC_PROCESSED_WARMUP", 3);
        const int iterations = envInt("LLAMINAR_PERF_GPU_SPEC_PROCESSED_ITERS", 20);
        for (int i = 0; i < warmup; ++i)
        {
            ASSERT_TRUE(run_pipeline());
            ASSERT_TRUE(backend->synchronizeStream(stream, device_id));
        }

        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i)
        {
            ASSERT_TRUE(run_pipeline());
            ASSERT_TRUE(backend->deviceToHostFast(
                output_tokens.data(),
                d_output_tokens.get(),
                output_tokens.size() * sizeof(int32_t),
                device_id,
                stream));
            ASSERT_TRUE(backend->deviceToHostFast(
                output_meta.data(),
                d_output_meta.get(),
                output_meta.size() * sizeof(int),
                device_id,
                stream));
        }
        const auto end = std::chrono::steady_clock::now();
        const double total_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        const double avg_us = (total_ms * 1000.0) / static_cast<double>(iterations);

        std::cout << "backend,case,rows,accepted_prefix,cols,iterations,total_ms,avg_us\n"
                  << backend_name << ",stochastic_processed_warped_logits,"
                  << row_count << ','
                  << accepted_prefix << ','
                  << kQwen36Vocab << ','
                  << iterations << ','
                  << std::fixed << std::setprecision(3) << total_ms << ','
                  << std::fixed << std::setprecision(3) << avg_us << '\n';
    }

    void runStochasticProcessedTargetDraftProbabilityCase(
        const std::string &backend_name,
        IBackend *backend,
        DeviceId device,
        int row_count,
        int accepted_prefix,
        StochasticDraftProposalMode proposal_mode =
            StochasticDraftProposalMode::ProbabilityRows)
    {
        if (!backend)
            GTEST_SKIP() << backend_name << " backend unavailable";
        ASSERT_TRUE(device.is_gpu());

        constexpr int kFirstToken = 100;
        constexpr float kTopP = 0.95f;
        constexpr float kTargetTemperature = 0.6f;
        constexpr float kDraftTemperature = 0.6f;
        constexpr float kAcceptThreshold = 0.5f;
        constexpr float kBonusThreshold = 0.42f;
        constexpr int kScratchBlocksPerRow = 128;
        const int kTopK = envInt("LLAMINAR_PERF_GPU_SPEC_TOP_K", 40);
        ASSERT_GE(kTopK, 1);
        ASSERT_LE(kTopK, 64)
            << "production stochastic perf sweeps are intended for small-k MTP sampling";

        const int device_id = device.gpu_ordinal();
        auto &ctx = GPUDeviceContextPool::instance().getContext(device);
        void *stream = ctx.defaultStream();
        ASSERT_NE(stream, nullptr)
            << "GPU production stochastic perf requires an explicit stream";

        ASSERT_GE(row_count, 1);
        ASSERT_LE(row_count, static_cast<int>(sampling_math::kSpeculativeBatchMaxRows));
        ASSERT_GE(accepted_prefix, 0);
        ASSERT_LE(accepted_prefix, row_count);

        const int target_rows = row_count + 1;
        std::vector<float> target_logits =
            makeStochasticTargetLogitsForAcceptedPrefix(
                target_rows, kQwen36Vocab, kTopK, accepted_prefix);
        std::vector<float> draft_logits =
            makeStochasticLogits(row_count, kQwen36Vocab, kTopK);

        std::array<float, sampling_math::kSpeculativeBatchMaxRows> accept_thresholds{};
        std::fill(accept_thresholds.begin(), accept_thresholds.end(), kAcceptThreshold);

        const int scratch_capacity = target_rows * kScratchBlocksPerRow * kTopK;

        DeviceAllocation d_target_raw_logits(
            backend, device_id, target_logits.size() * sizeof(float));
        DeviceAllocation d_draft_raw_logits(
            backend, device_id, draft_logits.size() * sizeof(float));
        DeviceAllocation d_target_logits(
            backend, device_id, target_logits.size() * sizeof(float));
        DeviceAllocation d_draft_probabilities(
            backend, device_id, draft_logits.size() * sizeof(float));
        DeviceAllocation d_scratch_values(
            backend, device_id,
            static_cast<size_t>(scratch_capacity) * sizeof(float));
        DeviceAllocation d_scratch_indices(
            backend, device_id,
            static_cast<size_t>(scratch_capacity) * sizeof(int));
        DeviceAllocation d_sampled_draft_tokens(
            backend, device_id, static_cast<size_t>(row_count) * sizeof(int));
        DeviceAllocation d_sampled_draft_probabilities(
            backend, device_id, static_cast<size_t>(row_count) * sizeof(float));
        DeviceAllocation d_verify_tokens(
            backend, device_id, static_cast<size_t>(target_rows) * sizeof(int));
        DeviceAllocation d_verify_accepted(
            backend, device_id, static_cast<size_t>(row_count) * sizeof(int));
        DeviceAllocation d_verify_accept_prob(
            backend, device_id, static_cast<size_t>(row_count) * sizeof(float));
        DeviceAllocation d_verify_accept_threshold(
            backend, device_id, static_cast<size_t>(row_count) * sizeof(float));
        DeviceAllocation d_bonus_token(backend, device_id, sizeof(int));
        DeviceAllocation d_output_tokens(
            backend, device_id,
            sampling_math::kSpeculativeBatchMaxOutputTokens * sizeof(int));
        DeviceAllocation d_output_meta(
            backend, device_id,
            sampling_math::kSpeculativeBatchMetaCount * sizeof(int));

        ASSERT_TRUE(d_target_raw_logits);
        ASSERT_TRUE(d_draft_raw_logits);
        ASSERT_TRUE(d_target_logits);
        ASSERT_TRUE(d_draft_probabilities);
        ASSERT_TRUE(d_scratch_values);
        ASSERT_TRUE(d_scratch_indices);
        ASSERT_TRUE(d_sampled_draft_tokens);
        ASSERT_TRUE(d_sampled_draft_probabilities);
        ASSERT_TRUE(d_verify_tokens);
        ASSERT_TRUE(d_verify_accepted);
        ASSERT_TRUE(d_verify_accept_prob);
        ASSERT_TRUE(d_verify_accept_threshold);
        ASSERT_TRUE(d_bonus_token);
        ASSERT_TRUE(d_output_tokens);
        ASSERT_TRUE(d_output_meta);

        ASSERT_TRUE(backend->hostToDevice(
            d_target_raw_logits.get(),
            target_logits.data(),
            target_logits.size() * sizeof(float),
            device_id,
            stream));
        ASSERT_TRUE(backend->hostToDevice(
            d_draft_raw_logits.get(),
            draft_logits.data(),
            draft_logits.size() * sizeof(float),
            device_id,
            stream));
        ASSERT_TRUE(backend->synchronizeStream(stream, device_id));

        std::array<int32_t, sampling_math::kSpeculativeBatchMaxOutputTokens> output_tokens{};
        std::array<int, sampling_math::kSpeculativeBatchMetaCount> output_meta{};

        auto run_pipeline = [&]() -> bool
        {
            if (!backend->enqueueBuildTopKTopPProcessedLogitsF32Device(
                    d_target_raw_logits.get(),
                    target_rows,
                    kQwen36Vocab,
                    kQwen36Vocab,
                    kTopK,
                    kTopP,
                    kTargetTemperature,
                    device_id,
                    stream,
                    d_target_logits.get(),
                    kQwen36Vocab,
                    d_scratch_values.get(),
                    d_scratch_indices.get(),
                    scratch_capacity))
            {
                return false;
            }

            if (proposal_mode == StochasticDraftProposalMode::GreedyNoDraftProbabilities)
            {
                if (!backend->enqueueArgmaxF32BatchedRowsDevice(
                        d_draft_raw_logits.get(),
                        row_count,
                        kQwen36Vocab,
                        device_id,
                        stream,
                        d_sampled_draft_probabilities.get(),
                        d_sampled_draft_tokens.get(),
                        d_scratch_values.get(),
                        d_scratch_indices.get(),
                        scratch_capacity))
                {
                    return false;
                }
            }
            else
            {
                for (int row = 0; row < row_count; ++row)
                {
                    const bool proposal_ok =
                        proposal_mode == StochasticDraftProposalMode::DraftLogits
                            ? backend->enqueueScaleAndSampleTemperatureLogitsF32Device(
                        static_cast<float *>(d_draft_raw_logits.get()) +
                            static_cast<size_t>(row) * kQwen36Vocab,
                        kQwen36Vocab,
                        kQwen36Vocab,
                        kDraftTemperature,
                        0.11f + 0.17f * static_cast<float>(row % 5),
                        device_id,
                        stream,
                        static_cast<float *>(d_draft_probabilities.get()) +
                            static_cast<size_t>(row) * kQwen36Vocab,
                        kQwen36Vocab,
                        static_cast<int *>(d_sampled_draft_tokens.get()) + row,
                        static_cast<float *>(d_sampled_draft_probabilities.get()) + row)
                    : backend->enqueueSoftmaxAndSampleTemperatureLogitsF32Device(
                        static_cast<float *>(d_draft_raw_logits.get()) +
                            static_cast<size_t>(row) * kQwen36Vocab,
                        kQwen36Vocab,
                        kQwen36Vocab,
                        kDraftTemperature,
                        0.11f + 0.17f * static_cast<float>(row % 5),
                        device_id,
                        stream,
                        static_cast<float *>(d_draft_probabilities.get()) +
                            static_cast<size_t>(row) * kQwen36Vocab,
                        kQwen36Vocab,
                        static_cast<int *>(d_sampled_draft_tokens.get()) + row,
                        static_cast<float *>(d_sampled_draft_probabilities.get()) + row);
                    if (!proposal_ok)
                    {
                        return false;
                    }
                }
            }

            const bool verify_ok =
                proposal_mode == StochasticDraftProposalMode::DraftLogits
                ? backend->enqueueSpeculativeVerifyProcessedTargetDraftLogitsF32DeviceThresholdsBatchDeviceTokens(
                    d_target_logits.get(),
                    d_draft_probabilities.get(),
                    row_count,
                    kQwen36Vocab,
                    kQwen36Vocab,
                    kQwen36Vocab,
                    d_sampled_draft_tokens.get(),
                    accept_thresholds.data(),
                    /*inverse_sample_seed=*/123456789ull,
                    /*inverse_sample_first_logical_position=*/1000,
                    device_id,
                    stream,
                    d_verify_tokens.get(),
                    d_verify_accepted.get(),
                    d_verify_accept_prob.get(),
                    d_verify_accept_threshold.get(),
                    d_sampled_draft_probabilities.get())
                : proposal_mode == StochasticDraftProposalMode::GreedyNoDraftProbabilities
                ? backend->enqueueSpeculativeVerifyProcessedTargetDraftProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
                    d_target_logits.get(),
                    /*draft_probabilities_device=*/nullptr,
                    row_count,
                    kQwen36Vocab,
                    kQwen36Vocab,
                    /*draft_row_stride=*/0,
                    d_sampled_draft_tokens.get(),
                    accept_thresholds.data(),
                    /*inverse_sample_seed=*/123456789ull,
                    /*inverse_sample_first_logical_position=*/1000,
                    device_id,
                    stream,
                    d_verify_tokens.get(),
                    d_verify_accepted.get(),
                    d_verify_accept_prob.get(),
                    d_verify_accept_threshold.get(),
                    /*no_draft_probabilities=*/true)
                : backend->enqueueSpeculativeVerifyProcessedTargetDraftProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
                    d_target_logits.get(),
                    d_draft_probabilities.get(),
                    row_count,
                    kQwen36Vocab,
                    kQwen36Vocab,
                    kQwen36Vocab,
                    d_sampled_draft_tokens.get(),
                    accept_thresholds.data(),
                    /*inverse_sample_seed=*/123456789ull,
                    /*inverse_sample_first_logical_position=*/1000,
                    device_id,
                    stream,
                    d_verify_tokens.get(),
                    d_verify_accepted.get(),
                    d_verify_accept_prob.get(),
                    d_verify_accept_threshold.get());
            if (!verify_ok)
            {
                return false;
            }

            if (accepted_prefix == row_count)
            {
                if (!backend->enqueueSampleProcessedLogitsF32Device(
                        static_cast<float *>(d_target_logits.get()) +
                            static_cast<size_t>(row_count) * kQwen36Vocab,
                        kQwen36Vocab,
                        kQwen36Vocab,
                        kBonusThreshold,
                        device_id,
                        stream,
                        d_bonus_token.get()))
                {
                    return false;
                }
            }

            return backend->enqueueSummarizeSpeculativeVerifyBatch(
                d_verify_tokens.get(),
                d_verify_accepted.get(),
                row_count,
                kFirstToken,
                nullptr,
                0,
                accepted_prefix == row_count
                    ? static_cast<int *>(d_bonus_token.get())
                    : nullptr,
                accepted_prefix == row_count,
                device_id,
                stream,
                d_output_tokens.get(),
                d_output_meta.get());
        };

        ASSERT_TRUE(run_pipeline());
        ASSERT_TRUE(backend->deviceToHostFast(
            output_tokens.data(),
            d_output_tokens.get(),
            output_tokens.size() * sizeof(int32_t),
            device_id,
            stream));
        ASSERT_TRUE(backend->deviceToHostFast(
            output_meta.data(),
            d_output_meta.get(),
            output_meta.size() * sizeof(int),
            device_id,
            stream));
        EXPECT_EQ(output_meta[sampling_math::kSpecBatchMetaOk], 1);
        EXPECT_EQ(output_tokens[0], kFirstToken);

        const int warmup = envInt("LLAMINAR_PERF_GPU_SPEC_PRODUCTION_WARMUP", 3);
        const int iterations = envInt("LLAMINAR_PERF_GPU_SPEC_PRODUCTION_ITERS", 20);
        for (int i = 0; i < warmup; ++i)
        {
            ASSERT_TRUE(run_pipeline());
            ASSERT_TRUE(backend->synchronizeStream(stream, device_id));
        }

        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i)
        {
            ASSERT_TRUE(run_pipeline());
            ASSERT_TRUE(backend->deviceToHostFast(
                output_tokens.data(),
                d_output_tokens.get(),
                output_tokens.size() * sizeof(int32_t),
                device_id,
                stream));
            ASSERT_TRUE(backend->deviceToHostFast(
                output_meta.data(),
                d_output_meta.get(),
                output_meta.size() * sizeof(int),
                device_id,
                stream));
        }
        const auto end = std::chrono::steady_clock::now();
        const double total_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        const double avg_us = (total_ms * 1000.0) / static_cast<double>(iterations);

        StochasticComponentTimes parts{};
        auto timed_sync = [&](auto &&fn) -> double
        {
            const auto t0 = std::chrono::steady_clock::now();
            EXPECT_TRUE(fn());
            EXPECT_TRUE(backend->synchronizeStream(stream, device_id));
            const auto t1 = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count();
        };

        parts.target_build_ms = timed_sync([&]()
        {
            return backend->enqueueBuildTopKTopPProcessedLogitsF32Device(
                d_target_raw_logits.get(),
                target_rows,
                kQwen36Vocab,
                kQwen36Vocab,
                kTopK,
                kTopP,
                kTargetTemperature,
                device_id,
                stream,
                d_target_logits.get(),
                kQwen36Vocab,
                d_scratch_values.get(),
                d_scratch_indices.get(),
                scratch_capacity);
        });
        parts.draft_build_ms = timed_sync([&]()
        {
            if (proposal_mode == StochasticDraftProposalMode::GreedyNoDraftProbabilities)
            {
                return backend->enqueueArgmaxF32BatchedRowsDevice(
                    d_draft_raw_logits.get(),
                    row_count,
                    kQwen36Vocab,
                    device_id,
                    stream,
                    d_sampled_draft_probabilities.get(),
                    d_sampled_draft_tokens.get(),
                    d_scratch_values.get(),
                    d_scratch_indices.get(),
                    scratch_capacity);
            }
            bool ok = true;
            for (int row = 0; row < row_count; ++row)
            {
                ok = ok && (proposal_mode == StochasticDraftProposalMode::DraftLogits
                    ? backend->enqueueScaleAndSampleTemperatureLogitsF32Device(
                        static_cast<float *>(d_draft_raw_logits.get()) +
                            static_cast<size_t>(row) * kQwen36Vocab,
                        kQwen36Vocab,
                        kQwen36Vocab,
                        kDraftTemperature,
                        0.11f + 0.17f * static_cast<float>(row % 5),
                        device_id,
                        stream,
                        static_cast<float *>(d_draft_probabilities.get()) +
                            static_cast<size_t>(row) * kQwen36Vocab,
                        kQwen36Vocab,
                        static_cast<int *>(d_sampled_draft_tokens.get()) + row,
                        static_cast<float *>(d_sampled_draft_probabilities.get()) + row)
                    : backend->enqueueSoftmaxAndSampleTemperatureLogitsF32Device(
                        static_cast<float *>(d_draft_raw_logits.get()) +
                            static_cast<size_t>(row) * kQwen36Vocab,
                        kQwen36Vocab,
                        kQwen36Vocab,
                        kDraftTemperature,
                        0.11f + 0.17f * static_cast<float>(row % 5),
                        device_id,
                        stream,
                        static_cast<float *>(d_draft_probabilities.get()) +
                            static_cast<size_t>(row) * kQwen36Vocab,
                        kQwen36Vocab,
                        static_cast<int *>(d_sampled_draft_tokens.get()) + row,
                        static_cast<float *>(d_sampled_draft_probabilities.get()) + row));
            }
            return ok;
        });
        parts.verify_ms = timed_sync([&]()
        {
            return proposal_mode == StochasticDraftProposalMode::DraftLogits
                ? backend->enqueueSpeculativeVerifyProcessedTargetDraftLogitsF32DeviceThresholdsBatchDeviceTokens(
                    d_target_logits.get(),
                    d_draft_probabilities.get(),
                    row_count,
                    kQwen36Vocab,
                    kQwen36Vocab,
                    kQwen36Vocab,
                    d_sampled_draft_tokens.get(),
                    accept_thresholds.data(),
                    /*inverse_sample_seed=*/123456789ull,
                    /*inverse_sample_first_logical_position=*/1000,
                    device_id,
                    stream,
                    d_verify_tokens.get(),
                    d_verify_accepted.get(),
                    d_verify_accept_prob.get(),
                    d_verify_accept_threshold.get(),
                    d_sampled_draft_probabilities.get())
                : proposal_mode == StochasticDraftProposalMode::GreedyNoDraftProbabilities
                ? backend->enqueueSpeculativeVerifyProcessedTargetDraftProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
                    d_target_logits.get(),
                    /*draft_probabilities_device=*/nullptr,
                    row_count,
                    kQwen36Vocab,
                    kQwen36Vocab,
                    /*draft_row_stride=*/0,
                    d_sampled_draft_tokens.get(),
                    accept_thresholds.data(),
                    /*inverse_sample_seed=*/123456789ull,
                    /*inverse_sample_first_logical_position=*/1000,
                    device_id,
                    stream,
                    d_verify_tokens.get(),
                    d_verify_accepted.get(),
                    d_verify_accept_prob.get(),
                    d_verify_accept_threshold.get(),
                    /*no_draft_probabilities=*/true)
                : backend->enqueueSpeculativeVerifyProcessedTargetDraftProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
                d_target_logits.get(),
                d_draft_probabilities.get(),
                row_count,
                kQwen36Vocab,
                kQwen36Vocab,
                kQwen36Vocab,
                d_sampled_draft_tokens.get(),
                accept_thresholds.data(),
                /*inverse_sample_seed=*/123456789ull,
                /*inverse_sample_first_logical_position=*/1000,
                device_id,
                stream,
                d_verify_tokens.get(),
                d_verify_accepted.get(),
                d_verify_accept_prob.get(),
                d_verify_accept_threshold.get());
        });
        parts.bonus_sample_ms = timed_sync([&]()
        {
            return backend->enqueueSampleProcessedLogitsF32Device(
                static_cast<float *>(d_target_logits.get()) +
                    static_cast<size_t>(row_count) * kQwen36Vocab,
                kQwen36Vocab,
                kQwen36Vocab,
                kBonusThreshold,
                device_id,
                stream,
                d_bonus_token.get());
        });
        parts.summary_ms = timed_sync([&]()
        {
            return backend->enqueueSummarizeSpeculativeVerifyBatch(
                d_verify_tokens.get(),
                d_verify_accepted.get(),
                row_count,
                kFirstToken,
                nullptr,
                0,
                d_bonus_token.get(),
                true,
                device_id,
                stream,
                d_output_tokens.get(),
                d_output_meta.get());
        });
        const auto d2h_start = std::chrono::steady_clock::now();
        EXPECT_TRUE(backend->deviceToHostFast(
            output_tokens.data(),
            d_output_tokens.get(),
            output_tokens.size() * sizeof(int32_t),
            device_id,
            stream));
        EXPECT_TRUE(backend->deviceToHostFast(
            output_meta.data(),
            d_output_meta.get(),
            output_meta.size() * sizeof(int),
            device_id,
            stream));
        const auto d2h_end = std::chrono::steady_clock::now();
        parts.d2h_ms =
            std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count();

        std::cout << "backend,case,rows,accepted_prefix,cols,iterations,total_ms,avg_us,"
                     "target_processed_ms,draft_proposal_ms,verify_ms,bonus_sample_ms,"
                     "summary_ms,d2h_ms\n"
                  << backend_name << ','
                  << stochasticDraftProposalModeName(proposal_mode)
                  << ','
                  << row_count << ','
                  << accepted_prefix << ','
                  << kQwen36Vocab << ','
                  << iterations << ','
                  << std::fixed << std::setprecision(3) << total_ms << ','
                  << std::fixed << std::setprecision(3) << avg_us << ','
                  << parts.target_build_ms << ','
                  << parts.draft_build_ms << ','
                  << parts.verify_ms << ','
                  << parts.bonus_sample_ms << ','
                  << parts.summary_ms << ','
                  << parts.d2h_ms << '\n';
    }

    void runStochasticFullProbabilityVerifyCase(
        const std::string &backend_name,
        IBackend *backend,
        DeviceId device,
        int row_count)
    {
        if (!backend)
            GTEST_SKIP() << backend_name << " backend unavailable";
        ASSERT_TRUE(device.is_gpu());

        const int device_id = device.gpu_ordinal();
        auto &ctx = GPUDeviceContextPool::instance().getContext(device);
        void *stream = ctx.defaultStream();
        ASSERT_NE(stream, nullptr)
            << "GPU full-probability stochastic perf requires an explicit stream";

        ASSERT_GE(row_count, 1);
        ASSERT_LE(row_count, static_cast<int>(sampling_math::kSpeculativeBatchMaxRows));

        std::vector<float> target_probabilities(
            static_cast<size_t>(row_count) * kQwen36Vocab, 0.0f);
        std::vector<float> draft_probabilities(
            static_cast<size_t>(row_count) * kQwen36Vocab, 0.0f);
        std::vector<float> inverse_samples(
            static_cast<size_t>(row_count) * kQwen36Vocab, 1.0f);
        std::vector<int> draft_tokens(static_cast<size_t>(row_count), 0);
        std::array<float, sampling_math::kSpeculativeBatchMaxRows> accept_thresholds{};
        std::fill(accept_thresholds.begin(), accept_thresholds.end(), 0.90f);

        for (int row = 0; row < row_count; ++row)
        {
            const size_t base = static_cast<size_t>(row) * kQwen36Vocab;
            const int draft_token = stochasticHotToken(row, 0, kQwen36Vocab);
            const int recovered_token = stochasticHotToken(row, 3, kQwen36Vocab);
            const int tail_token = stochasticHotToken(row, 7, kQwen36Vocab);
            draft_tokens[static_cast<size_t>(row)] = draft_token;

            target_probabilities[base + static_cast<size_t>(draft_token)] = 0.10f;
            target_probabilities[base + static_cast<size_t>(recovered_token)] = 0.60f;
            target_probabilities[base + static_cast<size_t>(tail_token)] = 0.30f;

            draft_probabilities[base + static_cast<size_t>(draft_token)] = 0.40f;
            draft_probabilities[base + static_cast<size_t>(recovered_token)] = 0.10f;
            draft_probabilities[base + static_cast<size_t>(tail_token)] = 0.50f;

            inverse_samples[base + static_cast<size_t>(recovered_token)] = 2.0f;
        }

        DeviceAllocation d_target(
            backend, device_id, target_probabilities.size() * sizeof(float));
        DeviceAllocation d_draft(
            backend, device_id, draft_probabilities.size() * sizeof(float));
        DeviceAllocation d_inverse(
            backend, device_id, inverse_samples.size() * sizeof(float));
        DeviceAllocation d_draft_tokens(
            backend, device_id, draft_tokens.size() * sizeof(int));
        DeviceAllocation d_verify_tokens(
            backend, device_id, static_cast<size_t>(row_count) * sizeof(int));
        DeviceAllocation d_verify_accepted(
            backend, device_id, static_cast<size_t>(row_count) * sizeof(int));
        DeviceAllocation d_verify_accept_prob(
            backend, device_id, static_cast<size_t>(row_count) * sizeof(float));
        DeviceAllocation d_verify_accept_threshold(
            backend, device_id, static_cast<size_t>(row_count) * sizeof(float));

        ASSERT_TRUE(d_target);
        ASSERT_TRUE(d_draft);
        ASSERT_TRUE(d_inverse);
        ASSERT_TRUE(d_draft_tokens);
        ASSERT_TRUE(d_verify_tokens);
        ASSERT_TRUE(d_verify_accepted);
        ASSERT_TRUE(d_verify_accept_prob);
        ASSERT_TRUE(d_verify_accept_threshold);

        ASSERT_TRUE(backend->hostToDevice(
            d_target.get(), target_probabilities.data(),
            target_probabilities.size() * sizeof(float), device_id, stream));
        ASSERT_TRUE(backend->hostToDevice(
            d_draft.get(), draft_probabilities.data(),
            draft_probabilities.size() * sizeof(float), device_id, stream));
        ASSERT_TRUE(backend->hostToDevice(
            d_inverse.get(), inverse_samples.data(),
            inverse_samples.size() * sizeof(float), device_id, stream));
        ASSERT_TRUE(backend->hostToDevice(
            d_draft_tokens.get(), draft_tokens.data(),
            draft_tokens.size() * sizeof(int), device_id, stream));
        ASSERT_TRUE(backend->synchronizeStream(stream, device_id));

        auto run_verify = [&]() -> bool
        {
            return backend->enqueueSpeculativeVerifyProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
                d_target.get(),
                d_draft.get(),
                d_inverse.get(),
                row_count,
                kQwen36Vocab,
                kQwen36Vocab,
                kQwen36Vocab,
                kQwen36Vocab,
                d_draft_tokens.get(),
                accept_thresholds.data(),
                device_id,
                stream,
                d_verify_tokens.get(),
                d_verify_accepted.get(),
                d_verify_accept_prob.get(),
                d_verify_accept_threshold.get());
        };

        ASSERT_TRUE(run_verify());
        ASSERT_TRUE(backend->synchronizeStream(stream, device_id));

        const int warmup = envInt("LLAMINAR_PERF_GPU_SPEC_FULL_PROB_WARMUP", 5);
        const int iterations = envInt("LLAMINAR_PERF_GPU_SPEC_FULL_PROB_ITERS", 100);
        for (int i = 0; i < warmup; ++i)
        {
            ASSERT_TRUE(run_verify());
            ASSERT_TRUE(backend->synchronizeStream(stream, device_id));
        }

        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i)
        {
            ASSERT_TRUE(run_verify());
            ASSERT_TRUE(backend->synchronizeStream(stream, device_id));
        }
        const auto end = std::chrono::steady_clock::now();
        const double total_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        const double avg_us = (total_ms * 1000.0) / static_cast<double>(iterations);

        std::cout << "backend,case,rows,cols,iterations,total_ms,avg_us\n"
                  << backend_name << ",stochastic_full_probability_verify,"
                  << row_count << ','
                  << kQwen36Vocab << ','
                  << iterations << ','
                  << std::fixed << std::setprecision(3) << total_ms << ','
                  << std::fixed << std::setprecision(3) << avg_us << '\n';
    }

    void runProcessedLogitSoftmaxCase(
        const std::string &backend_name,
        IBackend *backend,
        DeviceId device,
        int row_count)
    {
        if (!backend)
            GTEST_SKIP() << backend_name << " backend unavailable";
        ASSERT_TRUE(device.is_gpu());

        const int kTopK = envInt("LLAMINAR_PERF_GPU_SPEC_TOP_K", 40);
        ASSERT_GE(kTopK, 1);
        ASSERT_LE(kTopK, 64)
            << "processed-logit softmax perf sweeps are intended for small-k MTP sampling";
        const int device_id = device.gpu_ordinal();
        auto &ctx = GPUDeviceContextPool::instance().getContext(device);
        void *stream = ctx.defaultStream();
        ASSERT_NE(stream, nullptr)
            << "GPU processed-logit softmax perf requires an explicit stream";

        std::vector<float> logits(
            static_cast<size_t>(row_count) * kQwen36Vocab,
            -std::numeric_limits<float>::infinity());
        for (int row = 0; row < row_count; ++row)
        {
            const size_t base = static_cast<size_t>(row) * kQwen36Vocab;
            for (int rank = 0; rank < kTopK; ++rank)
            {
                const int token = stochasticHotToken(row, rank, kQwen36Vocab);
                logits[base + static_cast<size_t>(token)] =
                    9.0f - 0.061f * static_cast<float>(rank);
            }
        }

        DeviceAllocation d_logits(
            backend, device_id, logits.size() * sizeof(float));
        DeviceAllocation d_probs(
            backend, device_id, logits.size() * sizeof(float));
        ASSERT_TRUE(d_logits);
        ASSERT_TRUE(d_probs);

        ASSERT_TRUE(backend->hostToDevice(
            d_logits.get(), logits.data(), logits.size() * sizeof(float),
            device_id, stream));
        ASSERT_TRUE(backend->synchronizeStream(stream, device_id));

        auto run_softmax = [&]() -> bool
        {
            return backend->enqueueSoftmaxProcessedLogitsF32Device(
                d_logits.get(),
                row_count,
                kQwen36Vocab,
                kQwen36Vocab,
                device_id,
                stream,
                d_probs.get(),
                kQwen36Vocab);
        };

        ASSERT_TRUE(run_softmax());
        ASSERT_TRUE(backend->synchronizeStream(stream, device_id));

        const int warmup = envInt("LLAMINAR_PERF_GPU_SPEC_SOFTMAX_WARMUP", 5);
        const int iterations = envInt("LLAMINAR_PERF_GPU_SPEC_SOFTMAX_ITERS", 100);
        for (int i = 0; i < warmup; ++i)
        {
            ASSERT_TRUE(run_softmax());
            ASSERT_TRUE(backend->synchronizeStream(stream, device_id));
        }

        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i)
        {
            ASSERT_TRUE(run_softmax());
            ASSERT_TRUE(backend->synchronizeStream(stream, device_id));
        }
        const auto end = std::chrono::steady_clock::now();
        const double total_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        const double avg_us = (total_ms * 1000.0) / static_cast<double>(iterations);

        std::cout << "backend,case,rows,cols,iterations,total_ms,avg_us\n"
                  << backend_name << ",processed_logit_softmax,"
                  << row_count << ','
                  << kQwen36Vocab << ','
                  << iterations << ','
                  << std::fixed << std::setprecision(3) << total_ms << ','
                  << std::fixed << std::setprecision(3) << avg_us << '\n';
    }
}

TEST(Perf__GPUSpeculativeSummary, GreedyQwen36Rows)
{
#ifdef HAVE_CUDA
    for (int rows : {2, 3, 4})
        runCase("CUDA", getCUDABackend(), DeviceId::cuda(0), rows);
#endif
#ifdef HAVE_ROCM
    for (int rows : {2, 3, 4})
        runCase("ROCm", getROCmBackend(), DeviceId::rocm(0), rows);
#endif
#if !defined(HAVE_CUDA) && !defined(HAVE_ROCM)
    GTEST_SKIP() << "No GPU backend enabled";
#endif
}

TEST(Perf__GPUSpeculativeSummary, StochasticQwen36Rows)
{
#ifdef HAVE_CUDA
    for (int rows : {1, 2, 3})
        runStochasticCase("CUDA", getCUDABackend(), DeviceId::cuda(0), rows);
#endif
#ifdef HAVE_ROCM
    for (int rows : {1, 2, 3})
        runStochasticCase("ROCm", getROCmBackend(), DeviceId::rocm(0), rows);
#endif
#if !defined(HAVE_CUDA) && !defined(HAVE_ROCM)
    GTEST_SKIP() << "No GPU backend enabled";
#endif
}

TEST(Perf__GPUSpeculativeSummary, StochasticProcessedLogitQwen36Rows)
{
#ifdef HAVE_CUDA
    for (int accepted_prefix : {0, 1, 3})
        runStochasticProcessedLogitCase("CUDA", getCUDABackend(), DeviceId::cuda(0), 3, accepted_prefix);
#endif
#ifdef HAVE_ROCM
    for (int accepted_prefix : {0, 1, 3})
        runStochasticProcessedLogitCase("ROCm", getROCmBackend(), DeviceId::rocm(0), 3, accepted_prefix);
#endif
#if !defined(HAVE_CUDA) && !defined(HAVE_ROCM)
    GTEST_SKIP() << "No GPU backend enabled";
#endif
}

TEST(Perf__GPUSpeculativeSummary, StochasticProcessedTargetDraftProbabilityQwen36Rows)
{
#ifdef HAVE_CUDA
    for (int accepted_prefix : {0, 1, 3})
        runStochasticProcessedTargetDraftProbabilityCase(
            "CUDA", getCUDABackend(), DeviceId::cuda(0), 3, accepted_prefix);
#endif
#ifdef HAVE_ROCM
    for (int accepted_prefix : {0, 1, 3})
        runStochasticProcessedTargetDraftProbabilityCase(
            "ROCm", getROCmBackend(), DeviceId::rocm(0), 3, accepted_prefix);
#endif
#if !defined(HAVE_CUDA) && !defined(HAVE_ROCM)
    GTEST_SKIP() << "No GPU backend enabled";
#endif
}

TEST(Perf__GPUSpeculativeSummary, StochasticProcessedTargetDraftLogitQwen36Rows)
{
#ifdef HAVE_CUDA
    for (int accepted_prefix : {0, 1, 3})
        runStochasticProcessedTargetDraftProbabilityCase(
            "CUDA", getCUDABackend(), DeviceId::cuda(0), 3, accepted_prefix,
            StochasticDraftProposalMode::DraftLogits);
#endif
#ifdef HAVE_ROCM
    for (int accepted_prefix : {0, 1, 3})
        runStochasticProcessedTargetDraftProbabilityCase(
            "ROCm", getROCmBackend(), DeviceId::rocm(0), 3, accepted_prefix,
            StochasticDraftProposalMode::DraftLogits);
#endif
#if !defined(HAVE_CUDA) && !defined(HAVE_ROCM)
    GTEST_SKIP() << "No GPU backend enabled";
#endif
}

TEST(Perf__GPUSpeculativeSummary, StochasticProcessedTargetGreedyDraftQwen36Rows)
{
#ifdef HAVE_CUDA
    for (int accepted_prefix : {0, 1, 3})
        runStochasticProcessedTargetDraftProbabilityCase(
            "CUDA", getCUDABackend(), DeviceId::cuda(0), 3, accepted_prefix,
            StochasticDraftProposalMode::GreedyNoDraftProbabilities);
#endif
#ifdef HAVE_ROCM
    for (int accepted_prefix : {0, 1, 3})
        runStochasticProcessedTargetDraftProbabilityCase(
            "ROCm", getROCmBackend(), DeviceId::rocm(0), 3, accepted_prefix,
            StochasticDraftProposalMode::GreedyNoDraftProbabilities);
#endif
#if !defined(HAVE_CUDA) && !defined(HAVE_ROCM)
    GTEST_SKIP() << "No GPU backend enabled";
#endif
}

TEST(Perf__GPUSpeculativeSummary, StochasticFullProbabilityQwen36Rows)
{
#ifdef HAVE_CUDA
    for (int rows : {1, 2, 3})
        runStochasticFullProbabilityVerifyCase("CUDA", getCUDABackend(), DeviceId::cuda(0), rows);
#endif
#ifdef HAVE_ROCM
    for (int rows : {1, 2, 3})
        runStochasticFullProbabilityVerifyCase("ROCm", getROCmBackend(), DeviceId::rocm(0), rows);
#endif
#if !defined(HAVE_CUDA) && !defined(HAVE_ROCM)
    GTEST_SKIP() << "No GPU backend enabled";
#endif
}

TEST(Perf__GPUSpeculativeSummary, ProcessedLogitSoftmaxQwen36Rows)
{
#ifdef HAVE_CUDA
    for (int rows : {1, 2, 3})
        runProcessedLogitSoftmaxCase("CUDA", getCUDABackend(), DeviceId::cuda(0), rows);
#endif
#ifdef HAVE_ROCM
    for (int rows : {1, 2, 3})
        runProcessedLogitSoftmaxCase("ROCm", getROCmBackend(), DeviceId::rocm(0), rows);
#endif
#if !defined(HAVE_CUDA) && !defined(HAVE_ROCM)
    GTEST_SKIP() << "No GPU backend enabled";
#endif
}
