/**
 * @file Test__MTPStochasticOutcome.cpp
 * @brief Issue #15: both stochastic laws own a complete captured outcome family.
 *
 * Run probability rejection first, without warming a seeded graph. Every depth
 * from one through fifteen captures the production stage once, then changes
 * only resident positions, seeds, distributions, drafts and controller budgets. A CPU
 * arithmetic oracle checks all output/metadata bytes, including untouched row
 * tails. Large logical positions expose captured-position bugs without loading
 * model weights; full-context allocation and HTTP proofs are separate gates.
 */
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "backends/IGPUGraphCapture.h"
#include "execution/compute_stages/stages/MTPStochasticOutcomeStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "kernels/common/SamplingMath.h"

#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <memory>
#include <string>

namespace
{
using namespace llaminar2;
using namespace llaminar2::sampling_math;
using Verification = MTPStochasticOutcomeStage::Verification;

/** @brief Run identical captured contracts on CUDA and HIP, without a model. */
class StochasticOutcome : public ::testing::TestWithParam<std::string> {};

/** @brief Check first-use capture, all depths, changing positions and replay tails. */
TEST_P(StochasticOutcome, UnseededFirstCaptureAndEveryDepthAtLongPositions)
{
    constexpr int rows = 16;
    constexpr int distribution_stride = 256;
    constexpr int vocabulary = 248320;
    constexpr uint64_t seed = 0xC001D00D1234ABCDull; // Resolved entropy, not an API seed.
    struct State
    {
        std::array<int32_t, rows * distribution_stride> ids;
        std::array<float, rows * distribution_stride> probabilities;
        std::array<int32_t, rows> inputs, sampled, accepted;
        std::array<int32_t, kSpeculativeBatchMaxStopTokens> stops;
        std::array<int, kDeviceGenerationControlCount> control;
        std::array<int32_t, kSpeculativeBatchMaxOutputTokens> output;
        std::array<int, kSpeculativeBatchMetaCount> meta;
        int32_t position;
        uint64_t seed;
    };
    IBackend *backend = GetParam() == "CUDA" ? getCUDABackend() : getROCmBackend();
    ASSERT_NE(backend, nullptr);
    const DeviceId device = GetParam() == "CUDA" ? DeviceId::cuda(0) : DeviceId::rocm(0);
    auto execution = IDeviceContext::create(device);
    ASSERT_NE(execution, nullptr);
    auto &worker = GetParam() == "CUDA"
        ? GPUDeviceContextPool::instance().getNvidiaContext(0)
        : GPUDeviceContextPool::instance().getAMDContext(0);
    worker.submitAndWait([&]
    {
        void *const stream = worker.defaultStream();
        ASSERT_NE(stream, nullptr);
        const auto release = [backend](State *ptr) { if (ptr) backend->free(ptr, 0); };
        std::unique_ptr<State, decltype(release)> allocation(
            static_cast<State *>(backend->allocate(sizeof(State), 0)), release);
        ASSERT_NE(allocation, nullptr);
        auto *d = allocation.get();

        // This order is essential: issue #15 failed only after the unseeded
        // path reached parent composition without a seeded outcome warm-up.
        for (const auto law : {Verification::OneHotProbabilityRejection,
                               Verification::SerialEquivalent})
        for (int top_k : {1, 2, 20, 40, 256})
        for (int depth = 1; depth <= 15; ++depth)
        {
            MTPStochasticOutcomeStage::Params params;
            params.device_id = device;
            params.backend = backend;
            params.verification = law;
            params.vocabulary_size = vocabulary;
            params.target_token_ids_device = d->ids.data();
            params.target_probs_device = d->probabilities.data();
            params.target_distribution_row_stride = distribution_stride;
            params.top_k = top_k;
            params.verifier_input_tokens_device = d->inputs.data();
            params.verifier_input_token_stride = rows;
            params.stop_tokens_device = d->stops.data();
            params.threshold_base_positions_device = &d->position;
            params.threshold_seeds_device = &d->seed;
            params.generation_control_device = d->control.data();
            params.generation_control_stride = kDeviceGenerationControlCount;
            params.verifier_row_capacity = depth + 1;
            params.sampled_target_tokens_device = d->sampled.data();
            params.sampled_target_token_stride = rows;
            params.accepted_rows_device = d->accepted.data();
            params.accepted_row_stride = rows;
            params.output_tokens_device = d->output.data();
            params.output_token_stride = kSpeculativeBatchMaxOutputTokens;
            params.output_meta_device = d->meta.data();
            params.output_meta_stride = kSpeculativeBatchMetaCount;
            params.request_count = 1;
            params.comparison_rows_per_request = depth;
            MTPStochasticOutcomeStage stage(params);
            stage.setGPUStream(stream);
            auto changed = params;
            changed.verification = law == Verification::SerialEquivalent
                ? Verification::OneHotProbabilityRejection : Verification::SerialEquivalent;
            EXPECT_FALSE(stage.hasSameCaptureIdentity(changed));
            changed = params;
            ++changed.vocabulary_size;
            EXPECT_FALSE(stage.hasSameCaptureIdentity(changed));
            changed = params;
            ++changed.threshold_seeds_device;
            EXPECT_FALSE(stage.hasSameCaptureIdentity(changed));

            auto graph = worker.createGraphCapture(stream);
            ASSERT_NE(graph, nullptr);
            ASSERT_TRUE(graph->beginCapture());
            const bool enqueued = stage.execute(execution.get());
            ASSERT_TRUE(graph->endCapture());
            ASSERT_TRUE(enqueued);
            ASSERT_TRUE(graph->instantiate());
            ASSERT_GT(graph->nodeCount(), 0u);

            for (int replay = 0; replay < 22; ++replay)
            {
                SCOPED_TRACE(::testing::Message() << GetParam() << " law=" << static_cast<int>(law)
                    << " top_k=" << top_k << " depth=" << depth << " replay=" << replay);
                State initial{};
                // Change both seed words under one retained graph. Replay 20
                // rejects an unadmitted bank; 21 proves recovery after reset.
                initial.seed = replay == 20 ? 0 : seed ^
                    (static_cast<uint64_t>(replay) << 32) ^ (0x9E3779B9ull * replay);
                initial.stops.fill(-1);
                initial.sampled.fill(-777);
                initial.accepted.fill(-777);
                initial.output.fill(-777);
                initial.meta.fill(-777);
                constexpr std::array<int, 5> positions{7, 8191, 81919, 199999, 262127};
                initial.position = positions[static_cast<size_t>(replay) % positions.size()];
                auto policy = DeviceGenerationPolicy::fixed(depth);
                ASSERT_TRUE(initialize_device_generation_control(64, 64, policy, initial.control.data()));
                initial.control[kDeviceGenerationControlTransactionCommitBudget] =
                    replay % 3 == 0 ? 1 : depth + 1;
                initial.control[kDeviceGenerationControlNextLeadingCommittedOutputCount] = replay % 2;
                initial.inputs[0] = 42;
                for (int row = 0; row <= depth; ++row)
                {
                    for (int column = 0; column < top_k; ++column)
                    {
                        const auto index = static_cast<size_t>(row * distribution_stride + column);
                        initial.ids[index] = 100 + row * top_k + column;
                        // Alternate exact one-hot acceptance with broad support
                        // and absent drafts, so both rejection and bonus execute.
                        initial.probabilities[index] = replay % 4 == 0
                            ? (column == 0 ? 1.0f : 0.0f) : 1.0f / top_k;
                    }
                    if (row < depth)
                        initial.inputs[row + 1] = replay % 4 == 1
                            ? 99999 : initial.ids[row * distribution_stride];
                }
                if (replay % 5 == 4)
                    initial.stops[0] = initial.inputs[0];
                State expected = initial;
                for (int row = 0; row <= depth; ++row)
                {
                    const int position = initial.position + 1 + row;
                    const int *ids = initial.ids.data() + row * distribution_stride;
                    const float *probs = initial.probabilities.data() + row * distribution_stride;
                    if (law == Verification::OneHotProbabilityRejection && row < depth)
                    {
                        speculative_verify_with_thresholds_one_hot_draft_vllm_recovered(
                            ids, probs, top_k, vocabulary, initial.inputs[row + 1],
                            mtp_spec_threshold_from_seed(initial.seed, position, 1), initial.seed, position,
                            &expected.sampled[row], &expected.accepted[row], nullptr, nullptr);
                    }
                    else
                        expected.sampled[row] = sample_distribution_with_threshold(
                            ids, probs, top_k, mtp_spec_threshold_from_seed(initial.seed, position, 0));
                }
                summarize_speculative_verify_batch_at_commit_boundary(
                    initial.inputs[0], expected.sampled.data(),
                    law == Verification::SerialEquivalent ? nullptr : expected.accepted.data(),
                    depth, initial.stops.data(), kSpeculativeBatchMaxStopTokens,
                    expected.sampled[depth], 1,
                    initial.control[kDeviceGenerationControlTransactionCommitBudget],
                    expected.output.data(), kSpeculativeBatchMaxOutputTokens, expected.meta.data(),
                    law == Verification::SerialEquivalent ? initial.inputs.data() : nullptr,
                    initial.control[kDeviceGenerationControlNextLeadingCommittedOutputCount]);

                ASSERT_TRUE(backend->hostToDevice(d, &initial, sizeof(State), 0, stream));
                ASSERT_TRUE(graph->launchOnStream(stream));
                State observed{};
                ASSERT_TRUE(backend->deviceToHost(&observed, d, sizeof(State), 0, stream));
                ASSERT_TRUE(backend->synchronizeStream(stream, 0)); // Test oracle boundary only.
                EXPECT_EQ(observed.seed, initial.seed);
                if (initial.seed == 0)
                {
                    EXPECT_EQ(observed.control[kDeviceGenerationControlOk], 0);
                    EXPECT_EQ(observed.control[kDeviceGenerationControlErrorCode],
                        static_cast<int>(DeviceGenerationError::InvalidRequestSeed));
                    EXPECT_EQ(observed.meta[kSpecBatchMetaOk], 0);
                    continue;
                }
                EXPECT_TRUE(stage.hasSameCaptureIdentity(params));
                EXPECT_EQ(observed.sampled, expected.sampled);
                EXPECT_EQ(observed.accepted, expected.accepted);
                EXPECT_EQ(observed.output, expected.output);
                EXPECT_EQ(observed.meta, expected.meta);
                EXPECT_EQ(observed.control, initial.control);
                EXPECT_EQ(observed.inputs, initial.inputs);
            }
        }
    });
}

INSTANTIATE_TEST_SUITE_P(Backends, StochasticOutcome, ::testing::Values("CUDA", "ROCm"),
    [](const auto &info) { return info.param; });
} // namespace
