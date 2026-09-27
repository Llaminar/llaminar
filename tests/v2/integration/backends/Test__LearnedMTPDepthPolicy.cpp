/**
 * @file Test__LearnedMTPDepthPolicy.cpp
 * @brief Captured CUDA/ROCm proof that live MTP consumes the learned policy.
 *
 * These model-free transactions enter the public backend initializer, budget
 * publisher and fused outcome commit. The complete resident row must equal
 * its CPU arithmetic oracle, and independent expected depths distinguish a
 * learned hold/promotion/demotion from the old generic-only controller.
 */
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "backends/IGPUGraphCapture.h"
#include "execution/mtp/MTPDeviceGenerationPolicy.h"
#include <gtest/gtest.h>
#include <array>
#include <memory>
#include <string>

namespace
{
    using namespace llaminar2;
    using namespace sampling_math;

    /** @brief Each vendor must execute both learned sampling specializations. */
    class LearnedMTPDepthPolicy : public ::testing::TestWithParam<std::string> {};

    /** @test Learned decisions are byte-exact across captured replay/reset. */
    TEST_P(LearnedMTPDepthPolicy, CapturedCommitUsesLearnedRules)
    {
        auto *backend = GetParam() == "CUDA" ? getCUDABackend() : getROCmBackend();
        ASSERT_NE(backend, nullptr);
        const auto policy_backend = GetParam() == "CUDA"
            ? MTPDepthPolicyBackend::CUDA : MTPDepthPolicyBackend::ROCm;

        // One test-owned allocation contains immutable input and distinct
        // response/controller/publication regions. No allocation enters capture.
        constexpr int response_capacity = 32;
        constexpr int token_capacity = 4;
        constexpr int meta_offset = token_capacity;
        constexpr int base_offset = meta_offset + kSpeculativeBatchMetaCount;
        constexpr int response_offset = base_offset + 1;
        constexpr int control_offset = response_offset + response_capacity;
        constexpr int publication_offset = control_offset + kDeviceGenerationControlCount;
        constexpr int storage_count = publication_offset + 6;
        auto release = [backend](void *memory) { if (memory) backend->free(memory, 0); };
        std::unique_ptr<void, decltype(release)> allocation(
            backend->allocate(storage_count * sizeof(int), 0), release);
        ASSERT_NE(allocation, nullptr);
        auto *memory = static_cast<int *>(allocation.get());

        auto run = [&](IWorkerGPUContext &context)
        {
            context.submitAndWait([&]()
            {
                auto *stream = context.defaultStream();
                ASSERT_NE(stream, nullptr);
                for (const auto verify : {MTPVerifyMode::Greedy, MTPVerifyMode::SpeculativeSampling})
                for (const int depth : {1, 2, 3})
                for (const bool full_accept : {false, true})
                {
                    SCOPED_TRACE(::testing::Message() << GetParam() << " depth=" << depth
                        << " verify=" << static_cast<int>(verify) << " accepted=" << full_accept);
                    MTPRuntimeConfig config;
                    config.enabled = true;
                    config.verify_mode = verify;
                    config.depth_policy.mode = MTPDepthPolicyMode::Dynamic;
                    config.depth_policy.backend = policy_backend;
                    config.depth_policy.model_class = MTPDepthPolicyModelClass::MoE;
                    config.depth_policy.initial_depth = depth;
                    config.depth_policy.window_size = config.depth_policy.min_samples = 1;
                    config.depth_policy.cooldown_steps = 0;
                    config.depth_policy.promote_consecutive_windows = 1;
                    const auto policy = resolveMTPDeviceGenerationDepthPolicy(config);
                    ASSERT_TRUE(policy.valid());

                    std::array<int, storage_count> input{};
                    for (int token = 0; token < token_capacity; ++token) input[token] = 10 + token;
                    int *meta = input.data() + meta_offset;
                    meta[kSpecBatchMetaOk] = 1;
                    meta[kSpecBatchMetaReadyToken] = -1;
                    meta[kSpecBatchMetaOutputCount] = full_accept ? depth + 1 : 2;
                    meta[kSpecBatchMetaAcceptedSpeculativePrefix] = full_accept ? depth : 0;
                    meta[kSpecBatchMetaTargetVerifierStateCommitCount] = full_accept ? depth : 1;
                    meta[kSpecBatchMetaConsumedVerifierRows] = full_accept ? depth : 1;
                    meta[kSpecBatchMetaAllSpeculativeAccepted] = full_accept ? 1 : 0;
                    meta[kSpecBatchMetaSampledTerminal] = full_accept ? 1 : 0;
                    input[base_offset] = 23;
                    auto expected = input;
                    int *expected_control = expected.data() + control_offset;
                    ASSERT_TRUE(initialize_device_generation_control(response_capacity,
                        response_capacity, policy, expected_control));
                    ASSERT_EQ(prepare_device_generation_transaction_budget(depth + 1,
                        depth + 1, expected_control), depth + 1);
                    ASSERT_TRUE(commit_device_generation_and_derive_speculative_publication_metadata(
                        expected.data(), token_capacity, expected.data() + meta_offset,
                        kSpeculativeBatchMetaCount, 0, depth + 1, expected[base_offset],
                        expected.data() + response_offset, response_capacity, expected_control,
                        expected.data() + publication_offset, expected.data() + publication_offset + 1,
                        expected.data() + publication_offset + 2, expected.data() + publication_offset + 3,
                        expected.data() + publication_offset + 4));
                    if (full_accept)
                    {
                        // The regenerated ROCm sampling policy agrees with the
                        // other MoE domains at full acceptance: promote to or
                        // hold three while preserving capacity through fifteen.
                        EXPECT_EQ(expected_control[kDeviceGenerationControlCurrentDraftDepth], 3);
                        EXPECT_EQ(expected_control[kDeviceGenerationControlLearnedDepthMatchedWindows], 1);
                    }
                    else if (policy_backend == MTPDepthPolicyBackend::ROCm &&
                             verify == MTPVerifyMode::SpeculativeSampling && depth == 3)
                    {
                        // Zero acceptance is outside the learned healthy hold.
                        // The generic demotion remains a real live transition.
                        EXPECT_EQ(expected_control[kDeviceGenerationControlCurrentDraftDepth], 2);
                        EXPECT_EQ(expected_control[kDeviceGenerationControlLearnedDepthMatchedWindows], 0);
                    }

                    ASSERT_TRUE(backend->hostToDevice(memory, input.data(), sizeof(input), 0, stream));
                    auto capture = context.createGraphCapture(stream);
                    ASSERT_NE(capture, nullptr);
                    ASSERT_TRUE(capture->beginCapture());
                    ASSERT_TRUE(backend->enqueueInitializeDeviceGeneration(1, response_capacity, policy,
                        DeviceGenerationLeadingRowDisposition::PendingResponse, response_capacity,
                        memory + response_offset, kDeviceGenerationControlCount, memory + control_offset, 0, stream));
                    ASSERT_TRUE(backend->enqueuePrepareDeviceGenerationTransactionBudget(memory + control_offset,
                        kDeviceGenerationControlCount, 1, depth + 1, nullptr, nullptr, nullptr, 0, stream));
                    ASSERT_TRUE(backend->enqueueCommitDeviceGenerationAndDeriveSpeculativePublicationMetadata(
                        memory, token_capacity, memory + meta_offset, kSpeculativeBatchMetaCount,
                        memory + base_offset, 1, depth + 1, memory + response_offset, response_capacity,
                        memory + control_offset, kDeviceGenerationControlCount, 0, stream,
                        memory + publication_offset, memory + publication_offset + 1,
                        memory + publication_offset + 2, memory + publication_offset + 3,
                        memory + publication_offset + 4, nullptr, nullptr, nullptr, nullptr,
                        memory + publication_offset + 5));
                    ASSERT_TRUE(capture->endCapture());
                    ASSERT_TRUE(capture->instantiate());
                    for (int replay = 0; replay < 3; ++replay)
                    {
                        ASSERT_TRUE(capture->launch());
                        // Terminal test observation only; there is no transfer
                        // or host decision inside the captured transaction.
                        std::array<int, storage_count> actual{};
                        ASSERT_TRUE(backend->deviceToHost(actual.data(), memory, sizeof(actual), 0, stream));
                        ASSERT_TRUE(backend->synchronizeStream(stream, 0));
                        for (int index = 0; index < kDeviceGenerationControlCount; ++index)
                            EXPECT_EQ(actual[control_offset + index], expected[control_offset + index]) << index;
                        EXPECT_EQ(actual[publication_offset + 3], 1);
                    }
                }
            });
        };
        if (GetParam() == "CUDA") run(GPUDeviceContextPool::instance().getNvidiaContext(0));
        else run(GPUDeviceContextPool::instance().getAMDContext(0));
    }

    INSTANTIATE_TEST_SUITE_P(Backends, LearnedMTPDepthPolicy, ::testing::Values("CUDA", "ROCm"),
        [](const ::testing::TestParamInfo<std::string> &info) { return info.param; });
}
