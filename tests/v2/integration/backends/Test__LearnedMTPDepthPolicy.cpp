/**
 * @file Test__LearnedMTPDepthPolicy.cpp
 * @brief Captured CUDA/ROCm proof for legacy and confirmed survival-cost policies.
 *
 * These model-free transactions enter the public backend initializer, budget
 * publisher and fused outcome commit. The complete resident row must equal
 * its CPU arithmetic oracle, and independent expected depths distinguish actual
 * survival-cost trials from legacy learned/generic transitions. Both
 * generic keys and a synthetic complete unmeasured scope traverse captured
 * initialization/commit/replay. Synthetic geometry is not runtime provenance.
 * Installed-scope scripts compare48 outcomes and eight persistent publication
 * words/outcome. These arithmetic checks never waive an offline economic gate.
 * Utility regressions use actual canonical carry/state progress, all-depth score
 * proposals and authenticated continuation on two equal device rows. Every
 * nonhold proposal is confirmed across disjoint windows before its real trial;
 * observed emitted Y is not the verified-state comparison reward. Scoped existing
 * words39/26 hold target/count; no controller or diagnostic storage is enlarged.
 * Synthetic ROCm-domain arithmetic on CUDA is not CUDA topology admission.
 * Observe regressions separately distinguish physical draft work from eligible
 * learning windows, including maintenance clipping and carried-row boundaries.
 */
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "backends/IGPUGraphCapture.h"
#include "execution/mtp/MTPDeviceGenerationPolicy.h"
#include "execution/mtp/DeviceGenerationContract.h"
#include "execution/mtp/DeviceGenerationExecutionPolicy.h"
#include "execution/mtp/MTPServingForwardCaptureGeometry.h"
#include <gtest/gtest.h>
#include <array>
#include <algorithm>
#include <memory>
#include <cstring>
#include <string>
#include <vector>

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
                for (const bool synthetic_unmeasured_scope : {false, true})
                for (const auto verify : {MTPVerifyMode::Greedy, MTPVerifyMode::SpeculativeSampling})
                for (const int depth : {1, 2, 3})
                for (const bool full_accept : {false, true})
                {
                    SCOPED_TRACE(::testing::Message() << GetParam() << " depth=" << depth
                        << " verify=" << static_cast<int>(verify) << " accepted=" << full_accept
                        << " synthetic_unmeasured_scope=" << synthetic_unmeasured_scope);
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
                    if (synthetic_unmeasured_scope)
                    {
                        // Deliberately non-model geometry cannot enter a future
                        // QFN measured overlay. Architecture1/0 is synthetic
                        // immutable test data, never a hardware measurement.
                        const auto device_type = policy_backend == MTPDepthPolicyBackend::CUDA
                            ? DeviceType::CUDA : DeviceType::ROCm;
                        const auto role = mtpVerifierExecutionRole(
                            declaredDeviceGenerationExecutionPolicy(device_type,
                                DeviceGenerationExecutionDomain::NativeDomain));
                        config.depth_policy.scope = {
                            .verifier_width_policy = resolveMTPVerifierPhysicalWidthPolicy(
                                role, MTPDepthPolicyMode::Dynamic),
                            .tp_width = 4, .pp_stages = 1, .retained_depth = 15,
                            .hidden_dim = 512, .main_layers = 7, .expert_count = 16,
                            .experts_per_token = 2, .compute_capability_major = 1,
                            .compute_capability_minor = 0};
                    }
                    const auto policy = resolveMTPDeviceGenerationDepthPolicy(config);
                    ASSERT_TRUE(policy.valid());

                    std::array<int, storage_count> input{};
                    std::array<int, token_capacity> target_tokens{}, acceptance{};
                    for (int token = 0; token < token_capacity; ++token)
                    {
                        target_tokens[token] = 20 + token;
                        acceptance[token] = full_accept ? 1 : 0;
                    }
                    int *meta = input.data() + meta_offset;
                    summarize_speculative_verify_batch_at_commit_boundary(10, target_tokens.data(),
                        acceptance.data(), depth, nullptr, 0, 77, 1, depth + 1,
                        input.data(), token_capacity, meta);
                    ASSERT_EQ(meta[kSpecBatchMetaOk], 1);
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
                        EXPECT_TRUE(device_generation_policy_matches(actual.data() + control_offset, policy));
                        EXPECT_EQ(actual[control_offset + kDeviceGenerationControlLearnedScopeHiddenDim],
                                  synthetic_unmeasured_scope ? 512 : 0);
                        EXPECT_EQ(actual[control_offset + kDeviceGenerationControlLearnedScopeComputeCapabilityMajor],
                                  synthetic_unmeasured_scope ? 1 : 0);
                        EXPECT_EQ(actual[publication_offset + 3], 1);
                    }
                }
            });
        };
        if (GetParam() == "CUDA") run(GPUDeviceContextPool::instance().getNvidiaContext(0));
        else run(GPUDeviceContextPool::instance().getAMDContext(0));
    }

    /**
     * @test Observe retains eligible windows while pricing all physical drafts.
     *
     * Six public transactions include full/zero/partial acceptance and both
     * uncarried and carried maintenance clipping. A 1024-sample window cannot
     * evaluate or reset during this sequence. The CPU complete-row oracle is
     * supplemented by independent integer counts so shared arithmetic cannot
     * silently substitute consumed comparisons for physical or eligible work.
     */
    TEST_P(LearnedMTPDepthPolicy, ObserveEligibleWindowExcludesClippedPhysicalWork)
    {
        auto *backend = GetParam() == "CUDA" ? getCUDABackend() : getROCmBackend();
        ASSERT_NE(backend, nullptr);
        constexpr int steps = 6;
        constexpr int token_stride = 16;
        constexpr int meta_in_step = token_stride;
        constexpr int base_in_step = meta_in_step + kSpeculativeBatchMetaCount;
        constexpr int maintenance_in_step = base_in_step + 1;
        constexpr int maintenance_due_in_step = maintenance_in_step + 1;
        constexpr int boundary_advanced_in_step = maintenance_due_in_step + 1;
        constexpr int step_stride = boundary_advanced_in_step + 1;
        constexpr int response_capacity = 128;
        constexpr int response_offset = steps * step_stride;
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
                for (const int depth : {1, 5, 15})
                {
                    SCOPED_TRACE(::testing::Message() << GetParam() << " depth=" << depth
                        << " verify=" << static_cast<int>(verify));
                    MTPRuntimeConfig config;
                    config.enabled = true;
                    config.verify_mode = verify;
                    config.draft_tokens = depth;
                    config.depth_policy.mode = MTPDepthPolicyMode::Observe;
                    config.depth_policy.backend = GetParam() == "CUDA"
                        ? MTPDepthPolicyBackend::CUDA : MTPDepthPolicyBackend::ROCm;
                    config.depth_policy.model_class = MTPDepthPolicyModelClass::MoE;
                    config.depth_policy.initial_depth = depth;
                    config.depth_policy.window_size = config.depth_policy.min_samples = 1024;
                    const auto policy = resolveMTPDeviceGenerationDepthPolicy(config);
                    ASSERT_TRUE(policy.valid());
                    ASSERT_EQ(policy.mode, DeviceGenerationPolicyMode::Observe);
                    ASSERT_FALSE(policy.learned.enabled);
                    ASSERT_EQ(policy.initial_depth, depth);

                    std::array<int, storage_count> input{};
                    // Poison mutable bytes before first admission and every
                    // retained replay; immutable transaction inputs stay fixed.
                    std::fill(input.begin() + response_offset, input.end(), 0x35353535);
                    auto expected = input;
                    auto *control = expected.data() + control_offset;
                    ASSERT_TRUE(initialize_device_generation_control(response_capacity,
                        response_capacity, policy, control));
                    constexpr std::array<int, steps> leading_rows{0, 0, 1, 1, 0, 0};
                    int first_token = 10;
                    int base = 23;
                    int eligible_runs = 0;
                    int eligible_accepted = 0;
                    for (int step = 0; step < steps; ++step)
                    {
                        SCOPED_TRACE(::testing::Message() << "transaction=" << step);
                        const bool clipped = step == 2 || step == 4;
                        const int maintenance = clipped ? 1 : depth + 1;
                        const int offset = step * step_stride;
                        auto *tokens = input.data() + offset;
                        auto *meta = tokens + meta_in_step;
                        std::array<int, token_stride> row_tokens{};
                        std::array<int, token_stride> accepted{};
                        accepted.fill(1);
                        for (int row = 0; row < token_stride; ++row)
                            row_tokens[row] = 100 + step * token_stride + row;
                        if (step == 1 || (step == 5 && depth == 1)) accepted[0] = 0;
                        if (step == 5 && depth > 1) accepted[1] = 0;
                        EXPECT_EQ(control[kDeviceGenerationControlNextLeadingCommittedOutputCount], leading_rows[step]);
                        summarize_speculative_verify_batch_at_commit_boundary(first_token,
                            row_tokens.data(), accepted.data(), depth, nullptr, 0,
                            1000 + step, 1, maintenance, tokens, token_stride, meta,
                            nullptr, leading_rows[step]);
                        ASSERT_EQ(meta[kSpecBatchMetaOk], 1);
                        input[offset + base_in_step] = base;
                        input[offset + maintenance_in_step] = maintenance;
                        std::copy_n(input.data() + offset, step_stride, expected.data() + offset);
                        ASSERT_EQ(prepare_device_generation_transaction_budget(depth + 1,
                            maintenance, control), maintenance);
                        const int previous_eligible_accepted = eligible_accepted;
                        if (!clipped)
                        {
                            ++eligible_runs;
                            eligible_accepted += step == 0 || step == 3 ? depth
                                : step == 5 && depth > 1 ? 1 : 0;
                        }
                        ASSERT_TRUE(commit_device_generation_and_derive_speculative_publication_metadata(
                            expected.data() + offset, token_stride, expected.data() + offset + meta_in_step,
                            kSpeculativeBatchMetaCount, 0, depth + 1, base,
                            expected.data() + response_offset, response_capacity, control,
                            expected.data() + publication_offset, expected.data() + publication_offset + 1,
                            expected.data() + publication_offset + 2, expected.data() + publication_offset + 3,
                            expected.data() + publication_offset + 4));
                        EXPECT_EQ(control[kDeviceGenerationControlCurrentDraftDepth], depth);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthWindowVerifierRuns], eligible_runs);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthWindowAttemptedTokens], eligible_runs * depth);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthWindowAcceptedTokens], eligible_accepted);
                        EXPECT_EQ(control[kDeviceGenerationControlAttemptedDraftTokenCount], (step + 1) * depth);
                        if (clipped)
                        {
                            EXPECT_EQ(eligible_accepted, previous_eligible_accepted);
                            // At depth1 a carried clipped transaction compares
                            // its sole draft and reports full acceptance, yet
                            // remains ineligible because its commit budget is1.
                            EXPECT_EQ(meta[kSpecBatchMetaAcceptedSpeculativePrefix], step == 2 ? 1 : 0);
                            EXPECT_EQ(meta[kSpecBatchMetaConsumedVerifierRows], step == 2 ? 1 : 0);
                            EXPECT_EQ(meta[kSpecBatchMetaCommitBoundaryClipped], step == 4 || depth > 1 ? 1 : 0);
                        }
                        first_token = expected[publication_offset + 4];
                        base = expected[publication_offset + 1];
                    }
                    const int partial = depth > 1 ? 1 : 0;
                    EXPECT_EQ(control[kDeviceGenerationControlDepthWindowVerifierRuns], 4);
                    EXPECT_EQ(control[kDeviceGenerationControlDepthWindowAttemptedTokens], 4 * depth);
                    EXPECT_EQ(control[kDeviceGenerationControlDepthWindowAcceptedTokens], 2 * depth + partial);
                    EXPECT_EQ(control[kDeviceGenerationControlDepthWindowRejectedTokens], 2 * depth - partial);
                    EXPECT_EQ(control[kDeviceGenerationControlDepthWindowFullAccepts], 2);
                    EXPECT_EQ(control[kDeviceGenerationControlDepthWindowZeroAccepts], depth == 1 ? 2 : 1);
                    EXPECT_EQ(control[kDeviceGenerationControlDepthWindowRollbacks], 2);
                    EXPECT_EQ(control[kDeviceGenerationControlDepthWindowAcceptedPrefixSum], 2 * depth + partial);
                    EXPECT_EQ(control[kDeviceGenerationControlAttemptedDraftTokenCount], 6 * depth);
                    EXPECT_EQ(control[kDeviceGenerationControlAcceptedSpeculativeTokenCount], 2 * depth + partial + 1);
                    EXPECT_EQ(control[kDeviceGenerationControlConsumedVerifierRowCount], 2 * depth + 2 + (depth == 1 ? 1 : 2));
                    EXPECT_EQ(control[kDeviceGenerationControlDepthEvaluatedWindows], 0);
                    EXPECT_EQ(control[kDeviceGenerationControlDepthUpdates], 0);
                    EXPECT_EQ(control[kDeviceGenerationControlLearnedDepthMatchedWindows], 0);
                    EXPECT_EQ(control[kDeviceGenerationControlTransactionCount], steps);
                    EXPECT_EQ(control[kDeviceGenerationControlOk], 1);
                    EXPECT_EQ(control[kDeviceGenerationControlRequestComplete], 0);

                    ASSERT_TRUE(backend->hostToDevice(memory, input.data(), sizeof(input), 0, stream));
                    auto capture = context.createGraphCapture(stream);
                    ASSERT_NE(capture, nullptr);
                    ASSERT_TRUE(capture->beginCapture());
                    ASSERT_TRUE(backend->enqueueInitializeDeviceGeneration(1, response_capacity, policy,
                        DeviceGenerationLeadingRowDisposition::PendingResponse, response_capacity,
                        memory + response_offset, kDeviceGenerationControlCount, memory + control_offset, 0, stream));
                    for (int step = 0; step < steps; ++step)
                    {
                        const int offset = step * step_stride;
                        ASSERT_TRUE(backend->enqueuePrepareDeviceGenerationTransactionBudget(memory + control_offset,
                            kDeviceGenerationControlCount, 1, depth + 1,
                            memory + offset + maintenance_in_step, memory + offset + maintenance_due_in_step,
                            memory + offset + boundary_advanced_in_step, 0, stream));
                        ASSERT_TRUE(backend->enqueueCommitDeviceGenerationAndDeriveSpeculativePublicationMetadata(
                            memory + offset, token_stride, memory + offset + meta_in_step, kSpeculativeBatchMetaCount,
                            memory + offset + base_in_step, 1, depth + 1, memory + response_offset, response_capacity,
                            memory + control_offset, kDeviceGenerationControlCount, 0, stream,
                            memory + publication_offset, memory + publication_offset + 1,
                            memory + publication_offset + 2, memory + publication_offset + 3,
                            memory + publication_offset + 4, nullptr, nullptr, nullptr, nullptr,
                            memory + publication_offset + 5));
                    }
                    ASSERT_TRUE(capture->endCapture());
                    ASSERT_TRUE(capture->instantiate());
                    for (int replay = 0; replay < 20; ++replay)
                    {
                        SCOPED_TRACE(::testing::Message() << "retained_replay=" << replay);
                        // Test-only terminal observation and explicit setup
                        // writes occur outside capture, never inside a branch.
                        ASSERT_TRUE(backend->hostToDevice(memory, input.data(), sizeof(input), 0, stream));
                        ASSERT_TRUE(capture->launch());
                        std::array<int, storage_count> actual{};
                        ASSERT_TRUE(backend->deviceToHost(actual.data(), memory, sizeof(actual), 0, stream));
                        ASSERT_TRUE(backend->synchronizeStream(stream, 0));
                        for (int index = 0; index < kDeviceGenerationControlCount; ++index)
                            EXPECT_EQ(actual[control_offset + index], expected[control_offset + index]) << index;
                        EXPECT_TRUE(device_generation_policy_matches(actual.data() + control_offset, policy));
                        for (int index = 0; index < control[kDeviceGenerationControlResponseTokenCount]; ++index)
                            EXPECT_EQ(actual[response_offset + index], expected[response_offset + index]) << index;
                        EXPECT_EQ(actual[publication_offset + 3], 1);
                    }
                }
            });
        };
        if (GetParam() == "CUDA") run(GPUDeviceContextPool::instance().getNvidiaContext(0));
        else run(GPUDeviceContextPool::instance().getAMDContext(0));
    }

    /**
     * @test Installed scoped controller words match CPU arithmetic at every depth.
     *
     * Immutable compact outcomes are scripted before capture; this is a model-
     * free controller arithmetic proof, not loaded-model or runtime scope
     * provenance. The public initializer, budget publisher and atomic outcome
     * commit execute on device. Per-step copies remain D2D until one terminal
     * test observation. No replacement learned table or test-only kernel exists.
     */
    TEST_P(LearnedMTPDepthPolicy, InstalledScopedAllDepthsPreserveCompleteControlOnReplay)
    {
        auto *backend = GetParam() == "CUDA" ? getCUDABackend() : getROCmBackend();
        ASSERT_NE(backend, nullptr);
        const auto policy_backend = GetParam() == "CUDA"
            ? MTPDepthPolicyBackend::CUDA : MTPDepthPolicyBackend::ROCm;
        enum class AcceptanceProfile { Zero, Partial, Full, StationaryTwo, StationaryThree, ScopedNoFull };
        constexpr int maximum_depth = 15;
        constexpr int token_capacity = maximum_depth + 1;
        constexpr int outcomes = 48;
        constexpr int response_capacity = 1024;
        constexpr int token_offset = 0;
        constexpr int meta_offset = token_offset + outcomes * token_capacity;
        constexpr int base_offset = meta_offset + outcomes * kSpeculativeBatchMetaCount;
        constexpr int response_offset = base_offset + outcomes;
        constexpr int control_offset = response_offset + response_capacity;
        constexpr int publication_offset = control_offset + kDeviceGenerationControlCount;
        constexpr int common_publication_words = 7;
        // Seven outputs share the CPU arithmetic oracle; the backend also owns
        // a distinct verifier-condition mailbox for each persistent outcome.
        constexpr int publication_words = common_publication_words + 1;
        constexpr int snapshot_offset = publication_offset + outcomes * publication_words;
        constexpr int journal_offset = snapshot_offset + (outcomes + 1) * kDeviceGenerationControlCount;
        constexpr int journal_words = sizeof(MTPDiagnosticRecordStorage) / sizeof(int);
        constexpr int storage_count = journal_offset + journal_words;
        static_assert(sizeof(MTPDiagnosticRecordStorage) == 3912);
        static_assert(kDeviceGenerationControlCount == 108);
        static_assert(journal_offset % 2 == 0); // Preserve the bank's native eight-byte alignment.
        static_assert(outcomes * token_capacity < response_capacity);
        static_assert(maximum_depth <= DeviceGenerationPolicy::kMaximumSupportedDraftDepth);
        auto release = [backend](void *memory) { if (memory) backend->free(memory, 0); };
        std::unique_ptr<void, decltype(release)> allocation(
            backend->allocate(storage_count * sizeof(int), 0), release);
        ASSERT_NE(allocation, nullptr);
        auto *memory = static_cast<int *>(allocation.get());

        // Backend-local scalar fixtures cover the actual installed ROCm scope.
        // CUDA exercises the same complete geometry with its declared native
        // width strategy and synthetic architecture1/0; it remains unmeasured.
        auto run = [&](IWorkerGPUContext &context)
        {
            context.submitAndWait([&]()
            {
                auto *stream = context.defaultStream();
                ASSERT_NE(stream, nullptr);
                for (const bool complete_scope : {false, true})
                for (const auto verify : {MTPVerifyMode::Greedy, MTPVerifyMode::SpeculativeSampling})
                for (int initial_depth = 0; initial_depth <= maximum_depth; ++initial_depth)
                for (const auto profile : {AcceptanceProfile::Zero, AcceptanceProfile::Partial, AcceptanceProfile::Full,
                                           AcceptanceProfile::StationaryTwo, AcceptanceProfile::StationaryThree, AcceptanceProfile::ScopedNoFull})
                {
                    if ((profile == AcceptanceProfile::StationaryTwo || profile == AcceptanceProfile::StationaryThree) &&
                        initial_depth != 5) continue; // Focus the two reproduced cycles without multiplying the full sweep.
                    const bool scoped_no_full = profile == AcceptanceProfile::ScopedNoFull;
                    if (scoped_no_full && (!complete_scope || verify != MTPVerifyMode::Greedy || initial_depth != 3))
                        continue;
                    // Both native compilers execute the REAL installed ROCm
                    // policy as immutable arithmetic input. On CUDA this is not
                    // a claim of ROCm hardware/model/topology admission.
                    const auto learned_backend = scoped_no_full ? MTPDepthPolicyBackend::ROCm : policy_backend;
                    SCOPED_TRACE(::testing::Message() << GetParam() << " initial=" << initial_depth
                        << " verify=" << static_cast<int>(verify) << " profile=" << static_cast<int>(profile)
                        << " complete_scope=" << complete_scope);
                    MTPRuntimeConfig config;
                    config.enabled = true;
                    config.draft_tokens = maximum_depth;
                    config.graph_capacity_draft_tokens = maximum_depth;
                    config.verify_mode = verify;
                    config.depth_policy.mode = MTPDepthPolicyMode::Dynamic;
                    config.depth_policy.backend = learned_backend;
                    config.depth_policy.model_class = MTPDepthPolicyModelClass::MoE;
                    config.depth_policy.min_depth = 1;
                    config.depth_policy.max_depth = maximum_depth;
                    config.depth_policy.initial_depth = initial_depth;
                    if (complete_scope)
                    {
                        const auto device_type = policy_backend == MTPDepthPolicyBackend::CUDA
                            ? DeviceType::CUDA : DeviceType::ROCm;
                        const auto role = mtpVerifierExecutionRole(
                            declaredDeviceGenerationExecutionPolicy(device_type,
                                DeviceGenerationExecutionDomain::NativeDomain));
                        config.depth_policy.scope = {
                            .verifier_width_policy = scoped_no_full ? MTPVerifierPhysicalWidthPolicy::ExactLogicalRows
                                : resolveMTPVerifierPhysicalWidthPolicy(role, MTPDepthPolicyMode::Dynamic),
                            .tp_width = 4, .pp_stages = 1, .retained_depth = maximum_depth,
                            .hidden_dim = 2560, .main_layers = 48, .expert_count = 512,
                            .experts_per_token = 10,
                            .compute_capability_major = learned_backend == MTPDepthPolicyBackend::ROCm ? 90 : 1,
                            .compute_capability_minor = learned_backend == MTPDepthPolicyBackend::ROCm ? 6 : 0};
                    }
                    const auto policy = resolveMTPDeviceGenerationDepthPolicy(config);
                    ASSERT_TRUE(policy.valid());
                    // Keep production defaults: partial windows, cooldown and
                    // hold timing are part of the equality, not bypassed here.
                    ASSERT_EQ(policy.window_size, 16);
                    ASSERT_EQ(policy.minimum_samples, 4);
                    ASSERT_EQ(policy.cooldown_steps, 8);
                    ASSERT_EQ(policy.promote_consecutive_windows, 3);
                    const bool measured = complete_scope && learned_backend == MTPDepthPolicyBackend::ROCm;
                    ASSERT_EQ(hasMTPMeasuredScope(policy.learned), measured)
                        << "Install the complete trained ROCm domain before this native gate";
                    if (measured && verify == MTPVerifyMode::Greedy && initial_depth == 0)
                        EXPECT_EQ(policy.initial_depth, 5); // Frozen training-only greedy startup.
                    if (measured && verify == MTPVerifyMode::SpeculativeSampling && initial_depth == 0)
                        EXPECT_EQ(policy.initial_depth, 5); // Independently fixed stochastic startup in the exact red diagnostic.

                    std::array<int, storage_count> input{};
                    auto expected = input;
                    // One preallocated bank is optional evidence, never controller state.
                    // Generic scripts prove null-disabled no-write behavior; scoped
                    // scripts cover holds/moves and censored/full/zero evidence.
                    const bool journal_enabled = complete_scope &&
                        (profile == AcceptanceProfile::Zero || profile == AcceptanceProfile::StationaryTwo ||
                         profile == AcceptanceProfile::StationaryThree || scoped_no_full);
                    const int generation_budget = journal_enabled ? 256 : response_capacity;
                    if (journal_enabled)
                        ASSERT_TRUE(validMTPDepthTransactionTraceAdmission(DeviceGenerationAdmissionKind::NewRequest,
                            1, generation_budget));
                    MTPDepthTransactionTrace expected_trace{};
                    int *control = expected.data() + control_offset;
                    ASSERT_TRUE(initialize_device_generation_control(generation_budget,
                        response_capacity, policy, control));
                    std::copy_n(control, kDeviceGenerationControlCount,
                        expected.data() + snapshot_offset);
                    std::array<int, outcomes> verifier_rows{};
                    int condition_token = 1000;
                    for (int step = 0; step < outcomes; ++step)
                    {
                        const int depth = control[kDeviceGenerationControlCurrentDraftDepth];
                        const int leading = control[kDeviceGenerationControlNextLeadingCommittedOutputCount];
                        const int accepted = profile == AcceptanceProfile::Full ? depth
                            : profile == AcceptanceProfile::Zero ? 0
                            : scoped_no_full ? depth - 1
                            : profile == AcceptanceProfile::StationaryTwo ? std::min(depth, 2)
                            : profile == AcceptanceProfile::StationaryThree ? std::min(depth, 3)
                            : depth == 1 ? step % 2 : depth / 2;
                        const bool full = accepted == depth;
                        verifier_rows[step] = depth + 1;
                        int *tokens = input.data() + token_offset + step * token_capacity;
                        std::array<int, token_capacity> target_tokens{}, acceptance{};
                        for (int token = 0; token < token_capacity; ++token)
                        {
                            target_tokens[token] = 1000 + step * token_capacity + token;
                            acceptance[token] = token < accepted ? 1 : 0;
                        }
                        int *meta = input.data() + meta_offset + step * kSpeculativeBatchMetaCount;
                        // Canonical publication is essential: a full batch commits
                        // d+1 states; a rejected batch carries its correction token.
                        summarize_speculative_verify_batch_at_commit_boundary(condition_token,
                            target_tokens.data(), acceptance.data(), depth, nullptr, 0,
                            20000 + step, 1, depth + 1, tokens, token_capacity, meta, nullptr, leading);
                        ASSERT_EQ(meta[kSpecBatchMetaOk], 1);
                        input[base_offset + step] = 23 + step * token_capacity;
                        std::copy_n(tokens, token_capacity,
                            expected.data() + token_offset + step * token_capacity);
                        std::copy_n(meta, kSpeculativeBatchMetaCount,
                            expected.data() + meta_offset + step * kSpeculativeBatchMetaCount);
                        expected[base_offset + step] = input[base_offset + step];
                        ASSERT_EQ(prepare_device_generation_transaction_budget(
                            verifier_rows[step], verifier_rows[step], control), verifier_rows[step]);
                        const int previous_full = control[kDeviceGenerationControlDepthWindowFullAccepts];
                        const int previous_promotions = control[kDeviceGenerationControlDepthPromotions];
                        std::array<int, kDeviceGenerationControlCount> before_window{};
                        std::copy_n(control, kDeviceGenerationControlCount, before_window.data());
                        int *publication = expected.data() + publication_offset + step * publication_words;
                        ASSERT_TRUE(commit_device_generation_and_derive_speculative_publication_metadata(
                            expected.data() + token_offset + step * token_capacity, token_capacity,
                            expected.data() + meta_offset + step * kSpeculativeBatchMetaCount,
                            kSpeculativeBatchMetaCount, 0, verifier_rows[step], expected[base_offset + step],
                            expected.data() + response_offset, response_capacity, control,
                            publication, publication + 1, publication + 2, publication + 3,
                            publication + 4, publication + 5, publication + 6,
                            journal_enabled ? &expected_trace : nullptr));
                        if (journal_enabled)
                        {
                            MTPDepthTransactionTraceValues entry;
                            ASSERT_TRUE(unpackMTPDepthTransactionTraceRecord(expected_trace.records[step], &entry));
                            EXPECT_EQ(entry.pre_depth, depth); EXPECT_EQ(entry.accepted_prefix, accepted);
                            EXPECT_EQ(entry.post_depth, control[kDeviceGenerationControlCurrentDraftDepth]);
                            EXPECT_EQ(entry.recommended_depth, control[kDeviceGenerationControlDepthLastRecommendedDepth]);
                            EXPECT_EQ(entry.rollback, !full); EXPECT_FALSE(entry.host_observation);
                            EXPECT_FALSE(entry.budget_limited);
                            EXPECT_EQ(entry.incoming_debt, leading != 0);
                            EXPECT_EQ(entry.outgoing_debt,
                                control[kDeviceGenerationControlNextLeadingCommittedOutputCount] != 0);
                            EXPECT_FALSE(entry.model_stopped);
                            EXPECT_EQ(entry.token_witness, control[kDeviceGenerationControlResponseTokenCount]);
                            EXPECT_EQ(entry.evaluated_windows, control[kDeviceGenerationControlDepthEvaluatedWindows]);
                            if (step == 0)
                                for (int word = 0; word < kDeviceGenerationControlCount; ++word)
                                    EXPECT_EQ(expected_trace.initial_control[word], before_window[word]) << word;
                        }
                        if (previous_full == 0 && !full)
                        {
                            const int next_depth = control[kDeviceGenerationControlCurrentDraftDepth];
                            if (next_depth > depth)
                            {
                                // A return restores a previously observed incumbent;
                                // it is not new evidence of a longer accepted prefix
                                // at the failed target. Prove
                                // the entire transition rather than relaxing the
                                // censored-prefix law or trusting one reporting word.
                                ASSERT_EQ(before_window[kDeviceGenerationControlDepthUtilityPhase],
                                    static_cast<int>(MTPReversibleProposalPhase::ProbePending)) << step;
                                EXPECT_EQ(next_depth, before_window[kDeviceGenerationControlDepthUtilityReferenceDepth]);
                                EXPECT_EQ(control[kDeviceGenerationControlDepthPromotions], previous_promotions + 1);
                                EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReturns],
                                    before_window[kDeviceGenerationControlDepthUtilityReturns] + 1);
                                EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityProbes],
                                    before_window[kDeviceGenerationControlDepthUtilityProbes]);
                                EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityKeeps],
                                    before_window[kDeviceGenerationControlDepthUtilityKeeps]);
                                EXPECT_EQ(control[kDeviceGenerationControlLearnedDepthMatchedWindows],
                                    before_window[kDeviceGenerationControlLearnedDepthMatchedWindows]);
                                EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityPhase],
                                    static_cast<int>(MTPReversibleProposalPhase::RejectedComparison));
                                EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReferenceDepth], depth);
                                EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReferenceRuns],
                                    before_window[kDeviceGenerationControlDepthWindowVerifierRuns] + 1);
                                EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReferenceVerifiedProgress],
                                    before_window[kDeviceGenerationControlDepthWindowVerifierRuns] + 1 +
                                    before_window[kDeviceGenerationControlDepthWindowAcceptedTokens] + accepted);
                                EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityWindowEmitted], 0);
                                EXPECT_EQ(control[kDeviceGenerationControlDepthWindowVerifierRuns], 0);
                            }
                            else
                                EXPECT_EQ(control[kDeviceGenerationControlDepthPromotions], previous_promotions)
                                    << "No full-prefix evidence or upward incumbent return at step=" << step;
                        }
                        condition_token = publication[4];
                        const int published = control[kDeviceGenerationControlPublishedStateCommitCount] -
                            before_window[kDeviceGenerationControlPublishedStateCommitCount];
                        EXPECT_EQ(published, accepted + 1);
                        EXPECT_EQ(control[kDeviceGenerationControlLastTransactionEmittedTokenCount] + leading -
                            control[kDeviceGenerationControlNextLeadingCommittedOutputCount], published);
                        if (measured)
                        {
                            EXPECT_EQ(control[kDeviceGenerationControlLearnedDepthMatchedWindows], 0);
                            // Explicit first decisions distinguish supported cost
                            // proposals from the retired generated4/5 classes.
                            if ((step == 3 || step == 7 || step == 11) && policy.initial_depth == 5)
                            {
                                const int target = profile == AcceptanceProfile::Zero ? 1
                                    : profile == AcceptanceProfile::Full ? 6
                                    : profile == AcceptanceProfile::StationaryTwo ? 2
                                    : profile == AcceptanceProfile::StationaryThree ? 4 : 0;
                                if (target != 0 && step < 11)
                                {
                                    EXPECT_EQ(control[kDeviceGenerationControlCurrentDraftDepth], 5);
                                    EXPECT_EQ(control[kDeviceGenerationControlDepthLastRecommendedDepth], target);
                                    EXPECT_EQ(control[kDeviceGenerationControlDepthPromotionStreak], (step + 1) / 4);
                                    EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityProbes], 0);
                                }
                                if (step == 11 && target != 0)
                                    EXPECT_EQ(control[kDeviceGenerationControlCurrentDraftDepth], target);
                            }
                            if (scoped_no_full && (step == 3 || step == 7 || step == 11))
                            {
                                EXPECT_EQ(control[kDeviceGenerationControlCurrentDraftDepth], step == 11 ? 2 : 3);
                                EXPECT_EQ(control[kDeviceGenerationControlDepthLastRecommendedDepth], 2);
                                EXPECT_EQ(control[kDeviceGenerationControlDepthPromotionStreak], step == 11 ? 0 : (step + 1) / 4);
                            }
                        }
                        std::copy_n(control, kDeviceGenerationControlCount,
                            expected.data() + snapshot_offset + (step + 1) * kDeviceGenerationControlCount);
                    }
                    ASSERT_EQ(control[kDeviceGenerationControlTransactionCount], outcomes);
                    ASSERT_EQ(control[kDeviceGenerationControlRequestComplete], 0);
                    if (measured) EXPECT_EQ(control[kDeviceGenerationControlLearnedDepthMatchedWindows], 0);

                    ASSERT_TRUE(backend->hostToDevice(memory, input.data(), sizeof(input), 0, stream));
                    auto capture = context.createGraphCapture(stream);
                    ASSERT_NE(capture, nullptr);
                    ASSERT_TRUE(capture->beginCapture());
                    ASSERT_TRUE(backend->enqueueInitializeDeviceGeneration(1, generation_budget, policy,
                        DeviceGenerationLeadingRowDisposition::PendingResponse, response_capacity,
                        memory + response_offset, kDeviceGenerationControlCount, memory + control_offset, 0, stream));
                    // Capture the copy on the exact stream. The blocking raw
                    // compatibility copy publishes an external event and is
                    // intentionally unavailable inside a retained graph.
                    ASSERT_TRUE(backend->deviceCopyAsync(memory + snapshot_offset, memory + control_offset,
                        kDeviceGenerationControlCount * sizeof(int), 0, stream));
                    for (int step = 0; step < outcomes; ++step)
                    {
                        ASSERT_TRUE(backend->enqueuePrepareDeviceGenerationTransactionBudget(memory + control_offset,
                            kDeviceGenerationControlCount, 1, verifier_rows[step], nullptr, nullptr, nullptr, 0, stream));
                        int *publication = memory + publication_offset + step * publication_words;
                        ASSERT_TRUE(backend->enqueueCommitDeviceGenerationAndDeriveSpeculativePublicationMetadata(
                            memory + token_offset + step * token_capacity, token_capacity,
                            memory + meta_offset + step * kSpeculativeBatchMetaCount, kSpeculativeBatchMetaCount,
                            memory + base_offset + step, 1, verifier_rows[step], memory + response_offset,
                            response_capacity, memory + control_offset, kDeviceGenerationControlCount, 0, stream,
                            publication, publication + 1, publication + 2, publication + 3,
                            publication + 4, publication + 5, publication + 6,
                            /*out_next_sidecar_condition_tokens_device=*/nullptr,
                            /*out_next_sidecar_position_ids_device=*/nullptr,
                            /*out_next_verifier_condition_tokens_device=*/publication + common_publication_words,
                            /*verifier_input_tokens_device=*/nullptr, /*verifier_input_token_stride=*/0,
                            /*out_committed_verifier_identity_device=*/nullptr,
                            journal_enabled ? reinterpret_cast<MTPDepthTransactionTrace *>(memory + journal_offset) : nullptr));
                        ASSERT_TRUE(backend->deviceCopyAsync(
                            memory + snapshot_offset + (step + 1) * kDeviceGenerationControlCount,
                            memory + control_offset, kDeviceGenerationControlCount * sizeof(int), 0, stream));
                    }
                    ASSERT_TRUE(capture->endCapture());
                    ASSERT_TRUE(capture->instantiate());
                    for (int replay = 0; replay < 3; ++replay)
                    {
                        // Dirty every mutable row, restore only immutable
                        // script inputs, and reuse this exact native executable.
                        auto dirty = input;
                        const int poison = -7000 - replay;
                        std::fill(dirty.begin() + response_offset, dirty.end(), poison);
                        ASSERT_TRUE(backend->hostToDevice(memory, dirty.data(), sizeof(dirty), 0, stream));
                        ASSERT_TRUE(capture->launch());
                        std::array<int, storage_count> actual{};
                        ASSERT_TRUE(backend->deviceToHost(actual.data(), memory, sizeof(actual), 0, stream));
                        for (int step = 0; step <= outcomes; ++step)
                        {
                            const int offset = snapshot_offset + step * kDeviceGenerationControlCount;
                            EXPECT_TRUE(device_generation_policy_matches(actual.data() + offset, policy));
                            for (int word = 0; word < kDeviceGenerationControlCount; ++word)
                                EXPECT_EQ(actual[offset + word], expected[offset + word])
                                    << "replay=" << replay << " step=" << step << " word=" << word;
                        }
                        for (int word = 0; word < kDeviceGenerationControlCount; ++word)
                            EXPECT_EQ(actual[control_offset + word], expected[control_offset + word]) << word;
                        for (int step = 0; step < outcomes; ++step)
                        {
                            const int offset = publication_offset + step * publication_words;
                            for (int word = 0; word < common_publication_words; ++word)
                                EXPECT_EQ(actual[offset + word], expected[offset + word])
                                    << "step=" << step << " word=" << word;
                            // CUDA and ROCm publish the committed next condition
                            // into this separate verifier mailbox after the common
                            // transaction. Keep this check distinct from CPU math.
                            EXPECT_EQ(actual[offset + common_publication_words], expected[offset + 4])
                                << "verifier condition step=" << step;
                        }
                        const int entries = journal_enabled
                            ? actual[control_offset + kDeviceGenerationControlTransactionCount] : 0;
                        const std::size_t populated_bytes = mtpDepthTransactionTraceBytes(entries);
                        if (journal_enabled) EXPECT_EQ(entries, outcomes);
                        if (populated_bytes != 0)
                            EXPECT_EQ(std::memcmp(actual.data() + journal_offset,
                                &expected_trace, populated_bytes), 0) << "replay=" << replay;
                        // Lazy reset overwrites only populated records. The tail
                        // remains poisoned and must never become copied evidence.
                        for (int word = static_cast<int>(populated_bytes / sizeof(int)); word < journal_words; ++word)
                            EXPECT_EQ(actual[journal_offset + word], poison)
                                << "replay=" << replay << " journal tail word=" << word;
                        const int emitted = control[kDeviceGenerationControlResponseTokenCount];
                        for (int token = 0; token < response_capacity; ++token)
                            EXPECT_EQ(actual[response_offset + token], token < emitted ? expected[response_offset + token] : poison) << token;
                    }
                }
            });
        };
        if (GetParam() == "CUDA") run(GPUDeviceContextPool::instance().getNvidiaContext(0));
        else run(GPUDeviceContextPool::instance().getAMDContext(0));
    }

    /** @brief Native script choices, with no alternative controller implementation. */
    enum class UtilityNativeScenario { Keep, Return, ContinuePending, Inactive, Unresolved, CacheAvoidance,
        StopExcluded, ClippedExcluded, ContinueConfirmation, StopConfirmation, DeferredReset, TargetChangeHold,
        CacheReopen, PoisonTarget, PoisonStreak, PoisonPending };

    /**
     * @brief Exercise actual captured commit/initialize owners on two equal rows.
     * @param vendor Native backend whose public kernels execute the script.
     * @param scenario Authored reward and lifecycle experiment.
     * @param mode Actual request mode; Observe/Fixed remain inactive utility lanes.
     * @param measured_scope Whether the immutable policy matches the measured domain.
     *
     * On CUDA the installed ROCm scope is explicit synthetic arithmetic input,
     * never a claim of CUDA model/hardware admission. Two independently owned
     * rows prove participant-equivalent selectors/state for identical inputs;
     * no TP communicator or transport economy is fabricated by this fixture.
     */
    void runCapturedUtilityProbeScript(const std::string &vendor,
        UtilityNativeScenario scenario, MTPDepthPolicyMode mode, bool measured_scope,
        MTPVerifyMode verify = MTPVerifyMode::Greedy, int confirmation_windows = 3)
    {
        auto *backend = vendor == "CUDA" ? getCUDABackend() : getROCmBackend();
        ASSERT_NE(backend, nullptr);
        constexpr int participants = 2;
        constexpr int transactions = 64;
        constexpr int snapshots = 80;
        constexpr int token_capacity = 16;
        constexpr int meta_in_row = token_capacity;
        constexpr int base_in_row = meta_in_row + kSpeculativeBatchMetaCount;
        constexpr int maintenance_in_row = base_in_row + 1;
        constexpr int due_in_row = maintenance_in_row + 1;
        constexpr int advanced_in_row = due_in_row + 1;
        constexpr int input_stride = advanced_in_row + 1;
        constexpr int response_capacity = 256;
        constexpr int poison_offset = transactions * input_stride;
        constexpr int response_offset = poison_offset + 4;
        constexpr int control_offset = response_offset + response_capacity;
        constexpr int publication_offset = control_offset + kDeviceGenerationControlCount;
        constexpr int publication_words = 8;
        constexpr int ticket_offset = publication_offset + transactions * publication_words;
        constexpr int ticket_words = sizeof(DeviceGenerationDispatchTicket) / sizeof(int);
        constexpr int snapshot_offset = ticket_offset + ticket_words;
        constexpr int snapshot_stride = kDeviceGenerationControlCount + response_capacity;
        constexpr int bank_words = snapshot_offset + snapshots * snapshot_stride;
        constexpr int poison = -912345;
        constexpr uint64_t epoch = 17, generation = 41;
        static_assert(kDeviceGenerationControlCount == 108);
        static_assert(sizeof(DeviceGenerationDispatchTicket) % sizeof(int) == 0);
        auto release = [backend](void *memory) { if (memory) backend->free(memory, 0); };
        std::unique_ptr<void, decltype(release)> allocation(
            backend->allocate(participants * bank_words * sizeof(int), 0), release);
        ASSERT_NE(allocation, nullptr);
        auto *memory = static_cast<int *>(allocation.get());

        auto run = [&](IWorkerGPUContext &context)
        {
            context.submitAndWait([&]()
            {
                auto *stream = context.defaultStream();
                ASSERT_NE(stream, nullptr);
                MTPRuntimeConfig config;
                config.enabled = true;
                config.draft_tokens = mode == MTPDepthPolicyMode::Fixed ? 5 : 15;
                config.graph_capacity_draft_tokens = 15;
                config.verify_mode = verify;
                config.depth_policy.mode = mode;
                config.depth_policy.backend = MTPDepthPolicyBackend::ROCm;
                config.depth_policy.model_class = MTPDepthPolicyModelClass::MoE;
                config.depth_policy.min_depth = scenario == UtilityNativeScenario::CacheReopen ? 5 : 1;
                config.depth_policy.max_depth = scenario == UtilityNativeScenario::TargetChangeHold ? 5
                    : scenario == UtilityNativeScenario::CacheReopen ? 7 : 15;
                config.depth_policy.promote_consecutive_windows = confirmation_windows;
                config.depth_policy.initial_depth = 5; // Authored incumbent; default confirmation3/cadence unchanged.
                config.depth_policy.scope = {
                    .verifier_width_policy = MTPVerifierPhysicalWidthPolicy::ExactLogicalRows,
                    .tp_width = measured_scope ? 4 : 2, .pp_stages = 1, .retained_depth = 15,
                    .hidden_dim = 2560, .main_layers = 48, .expert_count = 512,
                    .experts_per_token = 10, .compute_capability_major = 90,
                    .compute_capability_minor = 6};
                const auto policy = resolveMTPDeviceGenerationDepthPolicy(config);
                ASSERT_TRUE(policy.valid());
                ASSERT_EQ(policy.initial_depth, 5);
                ASSERT_EQ(policy.minimum_samples, 4);
                ASSERT_EQ(policy.window_size, 16);
                ASSERT_EQ(policy.cooldown_steps, 8);
                ASSERT_TRUE(confirmation_windows == 1 || confirmation_windows == 3);
                ASSERT_EQ(policy.promote_consecutive_windows, confirmation_windows);
                const bool active = measured_scope && mode == MTPDepthPolicyMode::Dynamic;
                ASSERT_EQ(active, scenario != UtilityNativeScenario::Inactive);

                /** @brief Immutable setup operation; capture never reads a live host selector. */
                struct Operation
                {
                    bool initialize = false;
                    bool continue_response = false;
                    int budget = 0;
                    DeviceGenerationLeadingRowDisposition leading = DeviceGenerationLeadingRowDisposition::PendingResponse;
                    int row = -1;
                    int verifier_rows = 0;
                    int snapshot = 0;
                    int poison_word = -1;
                    int poison_source = -1;
                };
                std::vector<Operation> operations;
                std::vector<int> input(bank_words, poison), expected(bank_words, poison);
                auto *control = expected.data() + control_offset;
                DeviceGenerationDispatchTicket ticket{};
                int row_count = 0, snapshot_count = 0, condition = 100, base = 23, poison_count = 0;
                auto snapshot = [&]()
                {
                    ASSERT_LT(snapshot_count, snapshots);
                    auto *destination = expected.data() + snapshot_offset + snapshot_count * snapshot_stride;
                    std::copy_n(control, kDeviceGenerationControlCount, destination);
                    std::copy_n(expected.data() + response_offset, response_capacity,
                        destination + kDeviceGenerationControlCount);
                    ++snapshot_count;
                };
                auto poison_word = [&](int word, int value)
                {
                    ASSERT_LT(poison_count, 4);
                    const int source = poison_offset + poison_count++;
                    input[source] = expected[source] = value;
                    control[word] = value; // Deliberate fixture corruption, never production repair.
                    operations.push_back({false, false, 0, DeviceGenerationLeadingRowDisposition::PendingResponse,
                        -1, 0, snapshot_count, word, source});
                    ASSERT_NO_FATAL_FAILURE(snapshot());
                };
                auto initialize = [&](int budget, bool continuing)
                {
                    const auto leading = continuing ? static_cast<DeviceGenerationLeadingRowDisposition>(
                        control[kDeviceGenerationControlNextLeadingCommittedOutputCount])
                        : DeviceGenerationLeadingRowDisposition::PendingResponse;
                    if (continuing)
                    {
                        ASSERT_TRUE(publish_device_generation_dispatch_ticket(control, nullptr, &ticket));
                        ASSERT_EQ(control[kDeviceGenerationControlRequestComplete], 1);
                        ASSERT_EQ(control[kDeviceGenerationControlRemainingTokenCount], 0);
                    }
                    ASSERT_TRUE(initialize_device_generation_control(budget, response_capacity, policy,
                        control, leading,
                        {.kind = continuing ? DeviceGenerationAdmissionKind::ContinueResponse
                                            : DeviceGenerationAdmissionKind::NewRequest,
                         .session_epoch = epoch, .workspace_generation = generation,
                         .prior_tickets = &ticket}));
                    ASSERT_TRUE(initialize_device_generation_dispatch_ticket(epoch, generation, &ticket));
                    ASSERT_TRUE(publish_device_generation_dispatch_ticket(control, nullptr, &ticket));
                    operations.push_back({true, continuing, budget, leading, -1, 0, snapshot_count});
                    ASSERT_NO_FATAL_FAILURE(snapshot());
                };
                auto commit = [&](int requested_accepted, bool clipped, bool stopped = false,
                                  bool expect_valid = true)
                {
                    ASSERT_LT(row_count, transactions);
                    const int depth = control[kDeviceGenerationControlCurrentDraftDepth];
                    const int leading = control[kDeviceGenerationControlNextLeadingCommittedOutputCount];
                    const int maintenance = clipped ? 1 : depth + 1;
                    const int accepted = std::min(requested_accepted, depth);
                    const int row = row_count++;
                    auto *tokens = input.data() + row * input_stride;
                    std::array<int, token_capacity> row_tokens{}, flags{};
                    for (int i = 0; i < token_capacity; ++i)
                    {
                        row_tokens[i] = 1000 + row * token_capacity + i;
                        flags[i] = i < accepted ? 1 : 0;
                    }
                    std::fill_n(tokens, input_stride, 0);
                    const int stop_token = row_tokens[0];
                    summarize_speculative_verify_batch_at_commit_boundary(condition, row_tokens.data(),
                        flags.data(), depth, stopped ? &stop_token : nullptr, stopped ? 1 : 0,
                        2000 + row, 1, maintenance,
                        tokens, token_capacity, tokens + meta_in_row, nullptr, leading);
                    ASSERT_EQ(tokens[meta_in_row + kSpecBatchMetaOk], 1);
                    ASSERT_EQ(tokens[meta_in_row + kSpecBatchMetaStoppedOnOutput], stopped ? 1 : 0);
                    tokens[base_in_row] = base;
                    tokens[maintenance_in_row] = maintenance;
                    std::copy_n(tokens, input_stride, expected.data() + row * input_stride);
                    ASSERT_EQ(prepare_device_generation_transaction_budget(depth + 1, maintenance, control), maintenance);
                    auto *publication = expected.data() + publication_offset + row * publication_words;
                    const int previous_published = control[kDeviceGenerationControlPublishedStateCommitCount];
                    std::array<int, kDeviceGenerationControlCount> before_commit{};
                    std::copy_n(control, kDeviceGenerationControlCount, before_commit.data());
                    const bool committed = commit_device_generation_and_derive_speculative_publication_metadata(
                        expected.data() + row * input_stride, token_capacity,
                        expected.data() + row * input_stride + meta_in_row, kSpeculativeBatchMetaCount,
                        0, depth + 1, base, expected.data() + response_offset, response_capacity,
                        control, publication, publication + 1, publication + 2, publication + 3,
                        publication + 4, publication + 5, publication + 6);
                    ASSERT_EQ(committed, expect_valid);
                    if (!expect_valid)
                    {
                        EXPECT_EQ(control[kDeviceGenerationControlOk], 0);
                        EXPECT_EQ(control[kDeviceGenerationControlErrorCode], static_cast<int>(DeviceGenerationError::InvalidDepthPolicy));
                        for (int word = kDeviceGenerationControlDepthStepsSinceChange;
                             word <= kDeviceGenerationControlDepthLastRecommendedDepth; ++word)
                            EXPECT_EQ(control[word], before_commit[word]) << "poison mutated adaptive word=" << word;
                        for (int word = kDeviceGenerationControlDepthUtilityPhase;
                             word <= kDeviceGenerationControlDepthUtilityReturns; ++word)
                            EXPECT_EQ(control[word], before_commit[word]) << "poison mutated gate word=" << word;
                    }
                    // The backend's eighth publication is checked independently
                    // against the actual common next-condition publication.
                    publication[7] = publication[4];
                    EXPECT_EQ(control[kDeviceGenerationControlOk], expect_valid ? 1 : 0);
                    if (expect_valid && !stopped)
                        EXPECT_EQ(control[kDeviceGenerationControlLastTransactionEmittedTokenCount] + leading -
                            control[kDeviceGenerationControlNextLeadingCommittedOutputCount],
                            control[kDeviceGenerationControlPublishedStateCommitCount] - previous_published);
                    if (expect_valid && !clipped && !stopped)
                        EXPECT_EQ(control[kDeviceGenerationControlPublishedStateCommitCount] - previous_published,
                            accepted + 1);
                    if (expect_valid)
                    {
                        condition = publication[4];
                        base = publication[1];
                    }
                    operations.push_back({false, false, 0, DeviceGenerationLeadingRowDisposition::PendingResponse,
                        row, depth + 1, snapshot_count});
                    ASSERT_NO_FATAL_FAILURE(snapshot());
                };

                const int incumbent_response = confirmation_windows == 1 ? 17 : 51;
                const int first_budget = scenario == UtilityNativeScenario::ContinuePending ? incumbent_response + 10
                    : scenario == UtilityNativeScenario::ContinueConfirmation ? 27 : 256;
                ASSERT_NO_FATAL_FAILURE(initialize(first_budget, false));
                for (int word = kDeviceGenerationControlDepthUtilityPhase;
                     word <= kDeviceGenerationControlDepthUtilityReturns; ++word)
                    EXPECT_EQ(control[word], 0) << "fresh utility word=" << word;
                if (!active)
                {
                    for (int step = 0; step < 4; ++step)
                        ASSERT_NO_FATAL_FAILURE(commit(3, false));
                    for (int word = kDeviceGenerationControlDepthUtilityPhase;
                         word <= kDeviceGenerationControlDepthUtilityReturns; ++word)
                        EXPECT_EQ(control[word], 0) << "inactive utility word=" << word;
                }
                else if (scenario == UtilityNativeScenario::PoisonTarget || scenario == UtilityNativeScenario::PoisonStreak)
                {
                    ASSERT_NO_FATAL_FAILURE(poison_word(scenario == UtilityNativeScenario::PoisonTarget
                        ? kDeviceGenerationControlDepthLastRecommendedDepth : kDeviceGenerationControlDepthPromotionStreak,
                        scenario == UtilityNativeScenario::PoisonTarget ? 16 : confirmation_windows));
                    ASSERT_NO_FATAL_FAILURE(commit(3, false, false, false));
                }
                else if (scenario == UtilityNativeScenario::DeferredReset ||
                         scenario == UtilityNativeScenario::ContinueConfirmation ||
                         scenario == UtilityNativeScenario::StopConfirmation ||
                         scenario == UtilityNativeScenario::TargetChangeHold)
                {
                    ASSERT_EQ(confirmation_windows, 3);
                    for (int step = 0; step < 4; ++step) ASSERT_NO_FATAL_FAILURE(commit(3, false));
                    EXPECT_EQ(control[kDeviceGenerationControlCurrentDraftDepth], 5);
                    EXPECT_EQ(control[kDeviceGenerationControlDepthLastRecommendedDepth], 4);
                    EXPECT_EQ(control[kDeviceGenerationControlDepthPromotionStreak], 1);
                    EXPECT_EQ(control[kDeviceGenerationControlDepthWindowVerifierRuns], 0);
                    EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityWindowEmitted], 0);
                    EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityProbes], 0);
                    EXPECT_EQ(control[kDeviceGenerationControlDepthEvaluatedWindows], 1);
                    if (scenario == UtilityNativeScenario::ContinueConfirmation)
                    {
                        // Keep at least the full six-state commit budget for
                        // BOTH eligible depth5 frames. Budget26 leaves only5
                        // before the second frame, so it cannot be eligible.
                        for (int step = 0; step < 2; ++step)
                        {
                            EXPECT_GE(control[kDeviceGenerationControlRemainingTokenCount], 6);
                            ASSERT_NO_FATAL_FAILURE(commit(3, false));
                        }
                        EXPECT_EQ(control[kDeviceGenerationControlResponseTokenCount], 25);
                        EXPECT_EQ(control[kDeviceGenerationControlRemainingTokenCount], 2);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthPromotionStreak], 1);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthWindowVerifierRuns], 2);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthWindowAcceptedTokens], 6);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityWindowEmitted], 8);
                        // Two ACTUAL one-token clipped publications close27;
                        // neither contributes another eligible confirmation.
                        for (int clipped = 0; clipped < 2; ++clipped)
                        {
                            ASSERT_NO_FATAL_FAILURE(commit(0, true));
                            EXPECT_EQ(control[kDeviceGenerationControlLastTransactionEmittedTokenCount], 1);
                            EXPECT_EQ(control[kDeviceGenerationControlRequestComplete], clipped == 1 ? 1 : 0);
                            EXPECT_EQ(control[kDeviceGenerationControlDepthWindowVerifierRuns], 2);
                            EXPECT_EQ(control[kDeviceGenerationControlDepthWindowAcceptedTokens], 6);
                            EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityWindowEmitted], 8);
                            EXPECT_EQ(control[kDeviceGenerationControlDepthLastRecommendedDepth], 4);
                            EXPECT_EQ(control[kDeviceGenerationControlDepthPromotionStreak], 1);
                            EXPECT_EQ(control[kDeviceGenerationControlDepthEvaluatedWindows], 1);
                            EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityProbes], 0);
                        }
                        ASSERT_EQ(control[kDeviceGenerationControlRequestComplete], 1);
                        EXPECT_EQ(control[kDeviceGenerationControlResponseTokenCount], 27);
                        EXPECT_EQ(control[kDeviceGenerationControlPublishedStateCommitCount], 26);
                        EXPECT_EQ(control[kDeviceGenerationControlNextLeadingCommittedOutputCount], 1);
                        EXPECT_EQ(control[kDeviceGenerationControlTransactionCount], 8);
                        const DeviceGenerationAdmissionRequest confirmed_admission{
                            .request_count = 1, .max_new_tokens = 27, .depth_policy = policy};
                        EXPECT_EQ(validateDeviceGenerationTerminal(
                            std::span<const int, kDeviceGenerationControlCount>(control, kDeviceGenerationControlCount),
                            confirmed_admission, response_capacity, 0, 0), DeviceGenerationTerminalError::None);
                        ASSERT_NO_FATAL_FAILURE(initialize(64, true));
                        EXPECT_EQ(control[kDeviceGenerationControlCurrentDraftDepth], 5);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthLastRecommendedDepth], 4);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthPromotionStreak], 1);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthWindowVerifierRuns], 2);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthWindowAcceptedTokens], 6);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityWindowEmitted], 8);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthEvaluatedWindows], 0);
                        for (int step = 0; step < 2; ++step) ASSERT_NO_FATAL_FAILURE(commit(3, false));
                        EXPECT_EQ(control[kDeviceGenerationControlCurrentDraftDepth], 5);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthPromotionStreak], 2);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthWindowVerifierRuns], 0);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthEvaluatedWindows], 1);
                        for (int step = 0; step < 4; ++step) ASSERT_NO_FATAL_FAILURE(commit(3, false));
                        EXPECT_EQ(control[kDeviceGenerationControlCurrentDraftDepth], 4);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthLastRecommendedDepth], 4);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthPromotionStreak], 0);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthEvaluatedWindows], 2);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityProbes], 1);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReferenceRuns], 4);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReferenceVerifiedProgress], 16);
                    }
                    else if (scenario == UtilityNativeScenario::StopConfirmation)
                    {
                        for (int step = 0; step < 2; ++step) ASSERT_NO_FATAL_FAILURE(commit(3, false));
                        ASSERT_NO_FATAL_FAILURE(commit(0, false, true));
                        EXPECT_EQ(control[kDeviceGenerationControlRequestComplete], 1);
                        EXPECT_EQ(control[kDeviceGenerationControlModelStopped], 1);
                        EXPECT_EQ(control[kDeviceGenerationControlCurrentDraftDepth], 5);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthLastRecommendedDepth], 4);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthPromotionStreak], 1);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthWindowVerifierRuns], 2);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthWindowAcceptedTokens], 6);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityWindowEmitted], 8);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthEvaluatedWindows], 1);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityProbes], 0);
                        const DeviceGenerationAdmissionRequest stopped_admission{
                            .request_count = 1, .max_new_tokens = 256, .depth_policy = policy};
                        EXPECT_EQ(validateDeviceGenerationTerminal(
                            std::span<const int, kDeviceGenerationControlCount>(control, kDeviceGenerationControlCount),
                            stopped_admission, response_capacity, 0, 0), DeviceGenerationTerminalError::None);
                    }
                    else if (scenario == UtilityNativeScenario::TargetChangeHold)
                    {
                        for (int step = 0; step < 4; ++step) ASSERT_NO_FATAL_FAILURE(commit(0, false));
                        EXPECT_EQ(control[kDeviceGenerationControlDepthLastRecommendedDepth], 1);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthPromotionStreak], 1);
                        for (int step = 0; step < 4; ++step) ASSERT_NO_FATAL_FAILURE(commit(3, false));
                        EXPECT_EQ(control[kDeviceGenerationControlDepthLastRecommendedDepth], 4);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthPromotionStreak], 1);
                        // Authored max5 prevents an upward alternative; a full
                        // Hold completes at16, clearing only then, not at4/5/6.
                        for (int step = 0; step < 16; ++step)
                        {
                            ASSERT_NO_FATAL_FAILURE(commit(15, false));
                            EXPECT_EQ(control[kDeviceGenerationControlDepthPromotionStreak], step == 15 ? 0 : 1);
                        }
                        EXPECT_EQ(control[kDeviceGenerationControlCurrentDraftDepth], 5);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthLastRecommendedDepth], 5);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthEvaluatedWindows], 4);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityProbes], 0);
                    }
                }
                else
                {
                    const bool upward = scenario == UtilityNativeScenario::Unresolved ||
                        scenario == UtilityNativeScenario::CacheAvoidance || scenario == UtilityNativeScenario::CacheReopen;
                    const int incumbent_steps = 4 * confirmation_windows;
                    for (int step = 0; step < incumbent_steps; ++step)
                    {
                        const int prefix = upward ? 15 : confirmation_windows > 1 && step == 7 ? 5 : 3;
                        ASSERT_NO_FATAL_FAILURE(commit(prefix, false));
                        if (step + 1 < incumbent_steps)
                        {
                            EXPECT_EQ(control[kDeviceGenerationControlCurrentDraftDepth], 5);
                            EXPECT_EQ(control[kDeviceGenerationControlDepthLastRecommendedDepth], step < 3 ? 5 : upward ? 6 : 4);
                            EXPECT_EQ(control[kDeviceGenerationControlDepthPromotionStreak], (step + 1) / 4);
                            EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityProbes], 0);
                            EXPECT_EQ(control[kDeviceGenerationControlDepthStepsSinceChange], policy.cooldown_steps + step + 1);
                        }
                        EXPECT_EQ(control[kDeviceGenerationControlDepthEvaluatedWindows], (step + 1) / 4);
                        if ((step + 1) % 4 == 0)
                        {
                            EXPECT_EQ(control[kDeviceGenerationControlDepthWindowVerifierRuns], 0);
                            EXPECT_EQ(control[kDeviceGenerationControlDepthWindowAcceptedTokens], 0);
                            EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityWindowEmitted], 0);
                        }
                    }
                    // Prefix3 gives exact V16 and a conservative4 score win.
                    // Full5 gives V24 and only censored optimism for6:an actual
                    // unresolved trial, never the old direct generated hold5.
                    EXPECT_EQ(control[kDeviceGenerationControlCurrentDraftDepth], upward ? 6 : 4);
                    EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityPhase], 1);
                    EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReferenceDepth], 5);
                    EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReferenceRuns], 4);
                    EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReferenceVerifiedProgress], upward ? 24 : 16);
                    EXPECT_EQ(control[kDeviceGenerationControlResponseTokenCount], upward ? 24 * confirmation_windows : incumbent_response);
                    EXPECT_EQ(control[kDeviceGenerationControlDepthLastRecommendedDepth], upward ? 6 : 4);
                    EXPECT_EQ(control[kDeviceGenerationControlDepthPromotionStreak], 0);
                    EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityProbes], 1);
                    EXPECT_EQ(control[kDeviceGenerationControlLearnedDepthMatchedWindows], 0);
                    EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityWindowEmitted], 0);

                    if (scenario == UtilityNativeScenario::PoisonPending)
                    {
                        ASSERT_NO_FATAL_FAILURE(poison_word(kDeviceGenerationControlDepthLastRecommendedDepth, 5));
                        ASSERT_NO_FATAL_FAILURE(poison_word(kDeviceGenerationControlDepthPromotionStreak, 1));
                        ASSERT_NO_FATAL_FAILURE(commit(3, false, false, false));
                    }
                    else if (scenario == UtilityNativeScenario::Keep)
                    {
                        // Exact state utility keeps4:15*C5 >16*C4. Emitted-only
                        // utility would RETURN because15*C5 <17*C4. This is an
                        // independent decision discriminator, not just a rename.
                        const auto costs = resolveMTPDepthCostProfile(policy.learned);
                        ASSERT_TRUE(costs.active());
                        EXPECT_GT(uint64_t{15} * costs.costAt(5), uint64_t{16} * costs.costAt(4));
                        EXPECT_LT(uint64_t{15} * costs.costAt(5), uint64_t{17} * costs.costAt(4));
                        for (const int accepted : {3, 3, 3, 2})
                            ASSERT_NO_FATAL_FAILURE(commit(accepted, false));
                        EXPECT_EQ(control[kDeviceGenerationControlCurrentDraftDepth], 4);
                        EXPECT_EQ(control[kDeviceGenerationControlResponseTokenCount], incumbent_response + 15);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityKeeps], 1);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReturns], 0);
                        for (int word = kDeviceGenerationControlDepthUtilityPhase;
                             word <= kDeviceGenerationControlDepthUtilityWindowEmitted; ++word)
                            EXPECT_EQ(control[word], 0);
                    }
                    else if (scenario == UtilityNativeScenario::ContinuePending ||
                             scenario == UtilityNativeScenario::ClippedExcluded)
                    {
                        ASSERT_NO_FATAL_FAILURE(commit(15, false));
                        EXPECT_EQ(control[kDeviceGenerationControlLastTransactionEmittedTokenCount], 4);
                        ASSERT_NO_FATAL_FAILURE(commit(15, false));
                        EXPECT_EQ(control[kDeviceGenerationControlLastTransactionEmittedTokenCount], 5);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthWindowVerifierRuns], 2);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthWindowAcceptedTokens], 8);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityWindowEmitted], 9);
                        const int steps_before = control[kDeviceGenerationControlDepthStepsSinceChange];
                        const int windows_before = control[kDeviceGenerationControlDepthEvaluatedWindows];
                        ASSERT_NO_FATAL_FAILURE(commit(0, true));
                        EXPECT_EQ(control[kDeviceGenerationControlLastTransactionEmittedTokenCount], 1);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthWindowVerifierRuns], 2);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthWindowAcceptedTokens], 8);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityWindowEmitted], 9);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthStepsSinceChange], steps_before);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthEvaluatedWindows], windows_before);
                        if (scenario == UtilityNativeScenario::ContinuePending)
                        {
                            ASSERT_EQ(control[kDeviceGenerationControlRequestComplete], 1); //Incumbent +4+5+clipped1.
                            const DeviceGenerationAdmissionRequest first_admission{
                                .request_count = 1, .max_new_tokens = first_budget, .depth_policy = policy};
                            EXPECT_EQ(validateDeviceGenerationTerminal(
                                std::span<const int, kDeviceGenerationControlCount>(control, kDeviceGenerationControlCount),
                                first_admission, response_capacity, 0, 0), DeviceGenerationTerminalError::None);
                            ASSERT_NO_FATAL_FAILURE(initialize(64, true));
                            EXPECT_EQ(control[kDeviceGenerationControlCurrentDraftDepth], 4);
                            EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReferenceDepth], 5);
                            EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReferenceRuns], 4);
                            EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReferenceVerifiedProgress], 16);
                            EXPECT_EQ(control[kDeviceGenerationControlDepthWindowVerifierRuns], 2);
                            EXPECT_EQ(control[kDeviceGenerationControlDepthWindowAcceptedTokens], 8);
                            EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityWindowEmitted], 9);
                            EXPECT_EQ(control[kDeviceGenerationControlDepthStepsSinceChange], steps_before);
                            for (const int word : {kDeviceGenerationControlTransactionCount,
                                kDeviceGenerationControlResponseTokenCount,
                                kDeviceGenerationControlAttemptedDraftTokenCount,
                                kDeviceGenerationControlDepthUpdates,
                                kDeviceGenerationControlLearnedDepthMatchedWindows,
                                kDeviceGenerationControlDepthUtilityProbes,
                                kDeviceGenerationControlDepthUtilityKeeps,
                                kDeviceGenerationControlDepthUtilityReturns})
                                EXPECT_EQ(control[word], 0) << word;
                        }
                        else EXPECT_EQ(control[kDeviceGenerationControlRequestComplete], 0);
                        ASSERT_NO_FATAL_FAILURE(commit(0, false));
                        EXPECT_EQ(control[kDeviceGenerationControlLastTransactionEmittedTokenCount], 2);
                        ASSERT_NO_FATAL_FAILURE(commit(0, false));
                        EXPECT_EQ(control[kDeviceGenerationControlLastTransactionEmittedTokenCount], 1);
                        EXPECT_EQ(control[kDeviceGenerationControlCurrentDraftDepth], 5);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReturns], 1);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityProbes],
                            scenario == UtilityNativeScenario::ContinuePending ? 0 : 1);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityPhase], 2);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReferenceDepth], 4);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReferenceRuns], 4);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReferenceVerifiedProgress], 12); //R4+A8.
                        // Complete the continued response with an actual stop;
                        // no fabricated completion word is used for validation.
                        ASSERT_NO_FATAL_FAILURE(commit(0, false, true));
                        ASSERT_EQ(control[kDeviceGenerationControlRequestComplete], 1);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReferenceVerifiedProgress], 12);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReturns], 1);
                    }
                    else if (scenario == UtilityNativeScenario::StopExcluded)
                    {
                        for (int step = 0; step < 2; ++step)
                            ASSERT_NO_FATAL_FAILURE(commit(3, false));
                        EXPECT_EQ(control[kDeviceGenerationControlDepthWindowVerifierRuns], 2);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthWindowAcceptedTokens], 6);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityWindowEmitted], 8);
                        const int windows_before = control[kDeviceGenerationControlDepthEvaluatedWindows];
                        const int steps_before = control[kDeviceGenerationControlDepthStepsSinceChange];
                        ASSERT_NO_FATAL_FAILURE(commit(0, false, true));
                        ASSERT_EQ(control[kDeviceGenerationControlRequestComplete], 1);
                        EXPECT_EQ(control[kDeviceGenerationControlModelStopped], 1);
                        EXPECT_EQ(control[kDeviceGenerationControlLastTransactionEmittedTokenCount], 1);
                        EXPECT_EQ(control[kDeviceGenerationControlNextLeadingCommittedOutputCount], 0);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthWindowVerifierRuns], 2);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthWindowAcceptedTokens], 6);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityWindowEmitted], 8);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityPhase], 1);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReferenceVerifiedProgress], 16);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityKeeps], 0);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReturns], 0);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthEvaluatedWindows], windows_before);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthStepsSinceChange], steps_before);
                    }
                    else
                    {
                        for (int step = 0; step < 4; ++step)
                            ASSERT_NO_FATAL_FAILURE(commit(scenario == UtilityNativeScenario::CacheReopen ? 3 : 0, false));
                        EXPECT_EQ(control[kDeviceGenerationControlCurrentDraftDepth], 5);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityPhase], 2);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReferenceDepth], upward ? 6 : 4);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReferenceRuns], 4);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReferenceVerifiedProgress],
                            scenario == UtilityNativeScenario::CacheReopen ? 16 : 4);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReturns], 1);
                        EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityKeeps], 0);
                        if (upward) EXPECT_EQ(control[kDeviceGenerationControlResponseTokenCount], 24 * confirmation_windows +
                            (scenario == UtilityNativeScenario::CacheReopen ? 17 : 5));
                        else EXPECT_EQ(control[kDeviceGenerationControlResponseTokenCount], incumbent_response + 4);
                        if (scenario == UtilityNativeScenario::CacheAvoidance || scenario == UtilityNativeScenario::CacheReopen)
                        {
                            const bool reopen = scenario == UtilityNativeScenario::CacheReopen;
                            const int target = reopen ? 6 : 7;
                            const int followup_steps = 8 + 4 * (confirmation_windows - 1);
                            constexpr std::array<int, 4> reopen_prefixes{0, 0, 1, 5};
                            for (int step = 0; step < followup_steps; ++step)
                            {
                                ASSERT_NO_FATAL_FAILURE(commit(reopen ? reopen_prefixes[step % 4] : 15, false));
                                if (step + 1 < followup_steps)
                                {
                                    EXPECT_EQ(control[kDeviceGenerationControlCurrentDraftDepth], 5);
                                    EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityPhase], 2);
                                    EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReferenceDepth], 6);
                                    EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReferenceVerifiedProgress], reopen ? 16 : 4);
                                    EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityProbes], 1);
                                }
                                if (step + 1 == 8 && confirmation_windows > 1)
                                {
                                    EXPECT_EQ(control[kDeviceGenerationControlDepthLastRecommendedDepth], target);
                                    EXPECT_EQ(control[kDeviceGenerationControlDepthPromotionStreak], 1);
                                }
                            }
                            // The cache is applied BEFORE each real completed
                            // confirmation:blocked6 permits7; a genuine better
                            // failed6 reference can reopen6, but still needs3.
                            EXPECT_EQ(control[kDeviceGenerationControlCurrentDraftDepth], target);
                            EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityPhase], 1);
                            EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReferenceDepth], 5);
                            EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReferenceRuns], confirmation_windows == 1 ? 8 : 4);
                            EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReferenceVerifiedProgress],
                                reopen ? (confirmation_windows == 1 ? 20 : 10) : (confirmation_windows == 1 ? 48 : 24));
                            EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityProbes], 2);
                            EXPECT_EQ(control[kDeviceGenerationControlDepthUtilityReturns], 1);
                            EXPECT_EQ(control[kDeviceGenerationControlDepthPromotionStreak], 0);
                        }
                    }
                    EXPECT_EQ(control[kDeviceGenerationControlLearnedDepthMatchedWindows], 0);
                    if (control[kDeviceGenerationControlRequestComplete] != 0 && control[kDeviceGenerationControlOk] != 0)
                    {
                        const bool continued = scenario == UtilityNativeScenario::ContinuePending;
                        const DeviceGenerationAdmissionRequest terminal_admission{
                            .request_count = 1, .max_new_tokens = continued ? 64 : 256,
                            .kind = continued ? DeviceGenerationAdmissionKind::ContinueResponse
                                              : DeviceGenerationAdmissionKind::NewRequest,
                            .depth_policy = policy};
                        EXPECT_EQ(validateDeviceGenerationTerminal(
                            std::span<const int, kDeviceGenerationControlCount>(control, kDeviceGenerationControlCount),
                            terminal_admission, response_capacity, 0, 0), DeviceGenerationTerminalError::None);
                    }
                }
                // A real captured NewRequest erases a pending/rejected/settled
                // old owner and all reporting words; it does not clear topology.
                ASSERT_NO_FATAL_FAILURE(initialize(64, false));
                EXPECT_EQ(control[kDeviceGenerationControlCurrentDraftDepth], config.depth_policy.initial_depth);
                EXPECT_EQ(control[kDeviceGenerationControlDepthLastRecommendedDepth], config.depth_policy.initial_depth);
                EXPECT_EQ(control[kDeviceGenerationControlDepthPromotionStreak], 0);
                for (int word = kDeviceGenerationControlDepthUtilityPhase;
                     word <= kDeviceGenerationControlDepthUtilityReturns; ++word)
                    EXPECT_EQ(control[word], 0) << "new request utility word=" << word;

                std::vector<int> initial(participants * bank_words, poison);
                for (int participant = 0; participant < participants; ++participant)
                    std::copy(input.begin(), input.end(), initial.begin() + participant * bank_words);
                ASSERT_TRUE(backend->hostToDevice(memory, initial.data(), initial.size() * sizeof(int), 0, stream));
                auto graph = context.createGraphCapture(stream);
                ASSERT_NE(graph, nullptr);
                ASSERT_TRUE(graph->beginCapture());
                for (int participant = 0; participant < participants; ++participant)
                {
                    auto *bank = memory + participant * bank_words;
                    auto *native_ticket = reinterpret_cast<DeviceGenerationDispatchTicket *>(bank + ticket_offset);
                    for (const auto &operation : operations)
                    {
                        if (operation.initialize)
                        {
                            if (operation.continue_response)
                                ASSERT_TRUE(backend->enqueuePublishDeviceGenerationDispatchTickets(
                                    bank + control_offset, kDeviceGenerationControlCount, 1,
                                    nullptr, native_ticket, 0, stream));
                            ASSERT_TRUE(backend->enqueueInitializeDeviceGeneration(1, operation.budget, policy,
                                operation.leading, response_capacity, bank + response_offset,
                                kDeviceGenerationControlCount, bank + control_offset, 0, stream,
                                {.kind = operation.continue_response ? DeviceGenerationAdmissionKind::ContinueResponse
                                                                    : DeviceGenerationAdmissionKind::NewRequest,
                                 .session_epoch = epoch, .workspace_generation = generation,
                                 .prior_tickets = native_ticket}));
                            ASSERT_TRUE(backend->enqueueInitializeDeviceGenerationDispatchTicket(epoch, generation,
                                bank + control_offset, kDeviceGenerationControlCount, 1, native_ticket, 0, stream));
                        }
                        else if (operation.poison_word >= 0)
                        {
                            ASSERT_TRUE(backend->deviceCopyAsync(bank + control_offset + operation.poison_word,
                                bank + operation.poison_source, sizeof(int), 0, stream));
                        }
                        else
                        {
                            auto *tokens = bank + operation.row * input_stride;
                            auto *publication = bank + publication_offset + operation.row * publication_words;
                            ASSERT_TRUE(backend->enqueuePrepareDeviceGenerationTransactionBudget(bank + control_offset,
                                kDeviceGenerationControlCount, 1, operation.verifier_rows,
                                tokens + maintenance_in_row, tokens + due_in_row, tokens + advanced_in_row, 0, stream));
                            ASSERT_TRUE(backend->enqueueCommitDeviceGenerationAndDeriveSpeculativePublicationMetadata(
                                tokens, token_capacity, tokens + meta_in_row, kSpeculativeBatchMetaCount,
                                tokens + base_in_row, 1, operation.verifier_rows, bank + response_offset, response_capacity,
                                bank + control_offset, kDeviceGenerationControlCount, 0, stream,
                                publication, publication + 1, publication + 2, publication + 3,
                                publication + 4, publication + 5, publication + 6,
                                nullptr, nullptr, publication + 7, nullptr, 0, nullptr, nullptr));
                        }
                        auto *destination = bank + snapshot_offset + operation.snapshot * snapshot_stride;
                        ASSERT_TRUE(backend->deviceCopyAsync(destination, bank + control_offset,
                            kDeviceGenerationControlCount * sizeof(int), 0, stream));
                        ASSERT_TRUE(backend->deviceCopyAsync(destination + kDeviceGenerationControlCount,
                            bank + response_offset, response_capacity * sizeof(int), 0, stream));
                    }
                }
                ASSERT_TRUE(graph->endCapture());
                ASSERT_TRUE(graph->instantiate());
                for (int replay = 0; replay < 3; ++replay)
                {
                    SCOPED_TRACE(::testing::Message() << vendor << " scenario=" << static_cast<int>(scenario)
                        << " mode=" << static_cast<int>(mode) << " measured=" << measured_scope << " replay=" << replay);
                    const int dirty_poison = poison - replay;
                    auto dirty = initial;
                    for (int participant = 0; participant < participants; ++participant)
                        std::fill(dirty.begin() + participant * bank_words + response_offset,
                            dirty.begin() + (participant + 1) * bank_words, dirty_poison);
                    ASSERT_TRUE(backend->hostToDevice(memory, dirty.data(), dirty.size() * sizeof(int), 0, stream));
                    ASSERT_TRUE(graph->launch());
                    std::vector<int> actual(participants * bank_words);
                    ASSERT_TRUE(backend->deviceToHost(actual.data(), memory, actual.size() * sizeof(int), 0, stream));
                    ASSERT_TRUE(backend->synchronizeStream(stream, 0));
                    for (int participant = 0; participant < participants; ++participant)
                    {
                        for (int snapshot_index = 0; snapshot_index < snapshot_count; ++snapshot_index)
                        {
                            const int offset = snapshot_offset + snapshot_index * snapshot_stride;
                            for (int word = 0; word < snapshot_stride; ++word)
                                EXPECT_EQ(actual[participant * bank_words + offset + word],
                                    expected[offset + word] == poison ? dirty_poison : expected[offset + word])
                                    << "participant=" << participant << " snapshot=" << snapshot_index << " word=" << word;
                            if (participant != 0)
                                EXPECT_TRUE(std::equal(actual.begin() + offset, actual.begin() + offset + snapshot_stride,
                                    actual.begin() + participant * bank_words + offset)) << "participant-equivalent snapshot=" << snapshot_index;
                        }
                        for (int row = 0; row < row_count; ++row)
                            for (int word = 0; word < publication_words; ++word)
                                EXPECT_EQ(actual[participant * bank_words + publication_offset + row * publication_words + word],
                                    expected[publication_offset + row * publication_words + word])
                                    << "publication row=" << row << " word=" << word;
                    }
                }
            });
        };
        if (vendor == "CUDA") run(GPUDeviceContextPool::instance().getNvidiaContext(0));
        else run(GPUDeviceContextPool::instance().getAMDContext(0));
    }

    /** @test Verified state progress, not emitted carry timing, owns strict keep/return. */
    TEST_P(LearnedMTPDepthPolicy, CapturedSurvivalTrialsUseVerifiedProgressAndActualCarry)
    {
        for (const auto verify : {MTPVerifyMode::Greedy, MTPVerifyMode::SpeculativeSampling})
            for (const auto scenario : {UtilityNativeScenario::Keep, UtilityNativeScenario::Return})
                ASSERT_NO_FATAL_FAILURE(runCapturedUtilityProbeScript(GetParam(), scenario,
                    MTPDepthPolicyMode::Dynamic, true, verify));
    }

    /** @test Actual clipped completion/continuation preserves partial V and resets reports. */
    TEST_P(LearnedMTPDepthPolicy, CapturedSurvivalContinuationPreservesVerifiedReference)
    {
        for (const auto verify : {MTPVerifyMode::Greedy, MTPVerifyMode::SpeculativeSampling})
            ASSERT_NO_FATAL_FAILURE(runCapturedUtilityProbeScript(GetParam(), UtilityNativeScenario::ContinuePending,
                MTPDepthPolicyMode::Dynamic, true, verify));
    }

    /** @test Unmeasured, Fixed and Observe preserve their declared actual legacy owner. */
    TEST_P(LearnedMTPDepthPolicy, CapturedUtilityInactiveDomainsLeaveAllGateWordsZero)
    {
        for (const auto verify : {MTPVerifyMode::Greedy, MTPVerifyMode::SpeculativeSampling})
        {
            ASSERT_NO_FATAL_FAILURE(runCapturedUtilityProbeScript(GetParam(), UtilityNativeScenario::Inactive,
                MTPDepthPolicyMode::Dynamic, false, verify));
            for (const auto mode : {MTPDepthPolicyMode::Fixed, MTPDepthPolicyMode::Observe})
                ASSERT_NO_FATAL_FAILURE(runCapturedUtilityProbeScript(GetParam(), UtilityNativeScenario::Inactive,
                    mode, true, verify));
        }
    }

    /** @test An optimistic higher score starts a real trial and can actually return. */
    TEST_P(LearnedMTPDepthPolicy, CapturedSurvivalUnresolvedUpwardProposalIsTrialed)
    {
        for (const auto verify : {MTPVerifyMode::Greedy, MTPVerifyMode::SpeculativeSampling})
            ASSERT_NO_FATAL_FAILURE(runCapturedUtilityProbeScript(GetParam(), UtilityNativeScenario::Unresolved,
                MTPDepthPolicyMode::Dynamic, true, verify));
    }

    /** @test A rejected nearest target6 cannot suppress the distinct next candidate7. */
    TEST_P(LearnedMTPDepthPolicy, CapturedSurvivalRejectedNearestDoesNotMonopolizeCandidates)
    {
        for (const auto verify : {MTPVerifyMode::Greedy, MTPVerifyMode::SpeculativeSampling})
            ASSERT_NO_FATAL_FAILURE(runCapturedUtilityProbeScript(GetParam(), UtilityNativeScenario::CacheAvoidance,
                MTPDepthPolicyMode::Dynamic, true, verify));
    }

    /** @test Both real stop and maintenance clipping remain physical but ineligible. */
    TEST_P(LearnedMTPDepthPolicy, CapturedSurvivalStopsAndClippingExcludeEvidence)
    {
        for (const auto verify : {MTPVerifyMode::Greedy, MTPVerifyMode::SpeculativeSampling})
            for (const auto scenario : {UtilityNativeScenario::StopExcluded, UtilityNativeScenario::ClippedExcluded})
                ASSERT_NO_FATAL_FAILURE(runCapturedUtilityProbeScript(GetParam(), scenario,
                    MTPDepthPolicyMode::Dynamic, true, verify));
    }

    /** @test Deferred completed windows and fresh reset use actual required3 history. */
    TEST_P(LearnedMTPDepthPolicy, CapturedConfirmationDeferralAndFreshResetOwnExistingWords)
    {
        for (const auto verify : {MTPVerifyMode::Greedy, MTPVerifyMode::SpeculativeSampling})
            ASSERT_NO_FATAL_FAILURE(runCapturedUtilityProbeScript(GetParam(), UtilityNativeScenario::DeferredReset,
                MTPDepthPolicyMode::Dynamic, true, verify));
    }

    /** @test Threshold1 preserves first-class immediate trials without changing defaults. */
    TEST_P(LearnedMTPDepthPolicy, CapturedConfirmationThresholdOneRetainsImmediateTrials)
    {
        for (const auto verify : {MTPVerifyMode::Greedy, MTPVerifyMode::SpeculativeSampling})
            for (const auto scenario : {UtilityNativeScenario::Keep, UtilityNativeScenario::Return})
                ASSERT_NO_FATAL_FAILURE(runCapturedUtilityProbeScript(GetParam(), scenario,
                    MTPDepthPolicyMode::Dynamic, true, verify, 1));
    }

    /** @test Authenticated continuation preserves target/streak AND the partial real bank. */
    TEST_P(LearnedMTPDepthPolicy, CapturedConfirmationContinuationRetainsDisjointWindows)
    {
        for (const auto verify : {MTPVerifyMode::Greedy, MTPVerifyMode::SpeculativeSampling})
            ASSERT_NO_FATAL_FAILURE(runCapturedUtilityProbeScript(GetParam(), UtilityNativeScenario::ContinueConfirmation,
                MTPDepthPolicyMode::Dynamic, true, verify));
    }

    /** @test Target replacement and completed Hold clear only their actual history. */
    TEST_P(LearnedMTPDepthPolicy, CapturedConfirmationTargetChangeAndHoldResetStreak)
    {
        for (const auto verify : {MTPVerifyMode::Greedy, MTPVerifyMode::SpeculativeSampling})
            ASSERT_NO_FATAL_FAILURE(runCapturedUtilityProbeScript(GetParam(), UtilityNativeScenario::TargetChangeHold,
                MTPDepthPolicyMode::Dynamic, true, verify));
    }

    /** @test A reopened target starts fresh confirmations while retaining the failed sample. */
    TEST_P(LearnedMTPDepthPolicy, CapturedConfirmationReopenedCacheStillRequiresThreeWindows)
    {
        for (const auto verify : {MTPVerifyMode::Greedy, MTPVerifyMode::SpeculativeSampling})
            ASSERT_NO_FATAL_FAILURE(runCapturedUtilityProbeScript(GetParam(), UtilityNativeScenario::CacheReopen,
                MTPDepthPolicyMode::Dynamic, true, verify));
    }

    /** @test Real stop leaves a deferred target and partial eligible bank untouched. */
    TEST_P(LearnedMTPDepthPolicy, CapturedConfirmationStoppedPartialWindowRemainsExcluded)
    {
        for (const auto verify : {MTPVerifyMode::Greedy, MTPVerifyMode::SpeculativeSampling})
            ASSERT_NO_FATAL_FAILURE(runCapturedUtilityProbeScript(GetParam(), UtilityNativeScenario::StopConfirmation,
                MTPDepthPolicyMode::Dynamic, true, verify));
    }

    /** @test Target/count/Pending poison is rejected before adaptive counter mutation. */
    TEST_P(LearnedMTPDepthPolicy, CapturedConfirmationPoisonCannotMutateLearnerBank)
    {
        for (const auto verify : {MTPVerifyMode::Greedy, MTPVerifyMode::SpeculativeSampling})
            for (const auto scenario : {UtilityNativeScenario::PoisonTarget, UtilityNativeScenario::PoisonStreak,
                                       UtilityNativeScenario::PoisonPending})
                ASSERT_NO_FATAL_FAILURE(runCapturedUtilityProbeScript(GetParam(), scenario,
                    MTPDepthPolicyMode::Dynamic, true, verify));
    }

    INSTANTIATE_TEST_SUITE_P(Backends, LearnedMTPDepthPolicy, ::testing::Values("CUDA", "ROCm"),
        [](const ::testing::TestParamInfo<std::string> &info) { return info.param; });
}
