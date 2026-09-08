/**
 * @file Test__OrdinaryGenerationController.cpp
 * @brief Device-free proof of ordinary generation's shared resident ledger.
 *
 * A sampled response and a consumed KV row are different events: the first
 * prefill sample commits no new state, and the final emitted token remains
 * pending. These tests exercise that distinction, continued requests, terminal
 * replay, malformed transitions and reuse without any model or accelerator.
 */
#include "kernels/common/SamplingMath.h"
#include "execution/mtp/DeviceGenerationContract.h"
#include <gtest/gtest.h>
#include <array>
#include <limits>

namespace
{
using namespace llaminar2::sampling_math;
using Control = std::array<int, kDeviceGenerationControlCount>;
using Source = OrdinaryGenerationSampleSource;
using llaminar2::DeviceGenerationAdmissionRequest;
using llaminar2::DeviceGenerationTerminalError;
using llaminar2::validateDeviceGenerationTerminal;

/** @test The shared transition owns both response and next-forward state, not just counters. */
TEST(OrdinaryGenerationController, PublicationAdvancesTheLiveFrontierExactlyOnce)
{
    for (int budget : {1, 2, 17, 256})
    for (auto leading : {DeviceGenerationLeadingRowDisposition::PendingResponse,
                         DeviceGenerationLeadingRowDisposition::AlreadyEmitted})
    {
        Control control{};
        std::array<int32_t, 256> response{};
        int32_t sample = 42, stop = 0, position = 7, next = 19, stopped = 0, ok = 1;
        OrdinaryGenerationPublication publication{
            .request_count = 1, .sampled_tokens = &sample, .stopped_flags = &stop,
            .source = leading == DeviceGenerationLeadingRowDisposition::PendingResponse
                ? Source::PrefillLogits : Source::DecodeLogits,
            .response_tokens = response.data(), .response_token_stride = response.size(),
            .control = control.data(), .control_stride = control.size(),
            .frontier = {.cached_tokens = &position, .next_condition_tokens = &next,
                .stopped_flags = &stopped, .publication_ok_flags = &ok,
                .request_capacity = 1, .context_capacity = 4096}};
        ASSERT_TRUE(initialize_device_generation_control(
            budget, response.size(), DeviceGenerationPolicy::ordinary(), control.data(), leading));
        for (int row = 0; row < budget; ++row) {
            sample = 42 + row;
            ASSERT_TRUE(publish_ordinary_generation_request(publication, 0));
            EXPECT_EQ(position, 7 + row +
                (leading == DeviceGenerationLeadingRowDisposition::AlreadyEmitted ? 1 : 0));
            EXPECT_EQ(next, sample);
            EXPECT_EQ(response[row], sample);
            EXPECT_EQ(ok, 1);
            publication.source = Source::DecodeLogits;
        }
        const auto terminal = control;
        const int terminal_position = position, terminal_next = next;
        sample = -1; stop = -1;
        for (int repeat = 0; repeat < 20; ++repeat)
            ASSERT_TRUE(publish_ordinary_generation_request(publication, 0));
        EXPECT_EQ(control, terminal);
        EXPECT_EQ(position, terminal_position);
        EXPECT_EQ(next, terminal_next);
    }
}

/** @test No invalid position, producer flag or sample partially commits the response. */
TEST(OrdinaryGenerationController, InvalidPublicationKeepsResponseAndFrontierUnadvanced)
{
    for (int defect = 0; defect < 6; ++defect) {
        Control control{};
        std::array<int32_t, 2> response{-77, -77};
        int32_t sample = 42, stop = 0, position = 7, next = 19, stopped = 0, ok = 1;
        OrdinaryGenerationPublication publication{
            .request_count = 1, .sampled_tokens = &sample, .stopped_flags = &stop,
            .source = Source::DecodeLogits, .response_tokens = response.data(), .response_token_stride = 2,
            .control = control.data(), .control_stride = control.size(),
            .frontier = {.cached_tokens = &position, .next_condition_tokens = &next,
                .stopped_flags = &stopped, .publication_ok_flags = &ok,
                .request_capacity = 1, .context_capacity = 8}};
        ASSERT_TRUE(initialize_device_generation_control(2, 2, DeviceGenerationPolicy::ordinary(),
            control.data(), DeviceGenerationLeadingRowDisposition::AlreadyEmitted));
        switch (defect) {
        case 0: position = -1; break;
        case 1: position = 8; break;
        case 2: position = std::numeric_limits<int32_t>::max(); break;
        case 3: ok = 0; break;
        case 4: stop = 2; break;
        case 5: sample = -1; break;
        }
        const int original_position = position;
        EXPECT_FALSE(publish_ordinary_generation_request(publication, 0));
        EXPECT_EQ(control[kDeviceGenerationControlOk], 0);
        EXPECT_EQ(control[kDeviceGenerationControlResponseTokenCount], 0);
        EXPECT_EQ(control[kDeviceGenerationControlPublishedStateCommitCount], 0);
        EXPECT_EQ(response, (std::array<int32_t, 2>{-77, -77}));
        EXPECT_EQ(position, original_position);
        EXPECT_EQ(next, 19);
        EXPECT_EQ(stopped, 0);
        EXPECT_EQ(ok, 0);
        const auto failed = control;
        EXPECT_FALSE(publish_ordinary_generation_request(publication, 0));
        EXPECT_EQ(control, failed);
    }
}

/** @test Forward-only consumes one condition, ignores sampler poison and clears the pending token. */
TEST(OrdinaryGenerationController, ForwardOnlyPublishesPositionWithoutSamplerState)
{
    Control control{};
    std::array<int32_t, 2> response{-77, -77};
    int32_t poison = -1, position = 7, next = 19, stopped = 0, ok = 1;
    OrdinaryGenerationPublication publication{
        .request_count = 1, .sampled_tokens = &poison, .stopped_flags = &poison,
        .source = Source::DecodeLogits, .response_tokens = response.data(), .response_token_stride = 2,
        .control = control.data(), .control_stride = control.size(),
        .frontier = {.cached_tokens = &position, .next_condition_tokens = &next,
            .stopped_flags = &stopped, .publication_ok_flags = &ok,
            .request_capacity = 1, .context_capacity = 8}};
    ASSERT_TRUE(initialize_device_generation_control(0, 2, DeviceGenerationPolicy::forwardOnly(),
        control.data(), DeviceGenerationLeadingRowDisposition::AlreadyEmitted));
    for (int repeat = 0; repeat < 20; ++repeat)
        ASSERT_TRUE(publish_ordinary_generation_request(publication, 0));
    EXPECT_EQ(position, 8);
    EXPECT_EQ(next, -1);
    EXPECT_EQ(stopped, 0);
    EXPECT_EQ(ok, 1);
    EXPECT_EQ(response, (std::array<int32_t, 2>{-77, -77}));
}

/** @test Sampling may directly own the destination rows, including EOS at full context. */
TEST(OrdinaryGenerationController, AliasedSamplerPublicationNeedsNoExtraStaging)
{
    Control control{};
    std::array<int32_t, 1> response{-77};
    int32_t position = std::numeric_limits<int32_t>::max(), next = 42, stopped = 1, ok = 1;
    OrdinaryGenerationPublication publication{
        .request_count = 1, .sampled_tokens = &next, .stopped_flags = &stopped,
        .source = Source::PrefillLogits, .response_tokens = response.data(), .response_token_stride = 1,
        .control = control.data(), .control_stride = control.size(),
        .frontier = {.cached_tokens = &position, .next_condition_tokens = &next,
            .stopped_flags = &stopped, .publication_ok_flags = &ok,
            .request_capacity = 1, .context_capacity = std::numeric_limits<int32_t>::max()}};
    ASSERT_TRUE(initialize_device_generation_control(1, 1, DeviceGenerationPolicy::ordinary(), control.data()));
    ASSERT_TRUE(publish_ordinary_generation_request(publication, 0));
    EXPECT_EQ(position, std::numeric_limits<int32_t>::max());
    EXPECT_EQ(response[0], 42);
    EXPECT_EQ(next, 42);
    EXPECT_EQ(stopped, 1);
    EXPECT_EQ(ok, 1);
    EXPECT_EQ(control[kDeviceGenerationControlModelStopped], 1);
    auto incomplete = publication;
    incomplete.frontier = {};
    EXPECT_FALSE(incomplete.valid());
    EXPECT_FALSE(publish_ordinary_generation_request(incomplete, 0));
    EXPECT_FALSE(publish_ordinary_generation_request(publication, 1));
}

TEST(OrdinaryGenerationController, PolicyIsExplicitAndKeepsTheSameStorageABI)
{
    static_assert(kDeviceGenerationControlCount == 46);
    static_assert(sizeof(DeviceGenerationPolicy) == 11 * sizeof(int));
    EXPECT_FALSE(DeviceGenerationPolicy::fixed(0).valid());
    EXPECT_TRUE(DeviceGenerationPolicy::ordinary().valid());
    EXPECT_TRUE(DeviceGenerationPolicy::ordinary().isOrdinary());
    for (int depth = 1; depth <= 15; ++depth)
    {
        EXPECT_TRUE(DeviceGenerationPolicy::fixed(depth).valid());
        EXPECT_FALSE(DeviceGenerationPolicy::fixed(depth).isOrdinary());
        auto invalid = DeviceGenerationPolicy::ordinary();
        invalid.initial_depth = depth;
        EXPECT_FALSE(invalid.valid());
    }
    auto invalid = DeviceGenerationPolicy::ordinary();
    invalid.window_size = 16;
    EXPECT_FALSE(invalid.valid());
}

/** @test A forward-only invocation commits one row, never samples or emits. */
TEST(OrdinaryGenerationController, ForwardOnlyIsAnExplicitZeroResponseOperation)
{
    const DeviceGenerationAdmissionRequest admission{
        .request_count = 1, .max_new_tokens = 0,
        .depth_policy = DeviceGenerationPolicy::forwardOnly(),
        .initial_leading_row_disposition = DeviceGenerationLeadingRowDisposition::AlreadyEmitted};
    ASSERT_TRUE(admission.valid());
    EXPECT_FALSE(DeviceGenerationAdmissionRequest({
        .request_count = 1, .max_new_tokens = 0,
        .depth_policy = DeviceGenerationPolicy::ordinary()}).valid());
    Control control{};
    std::array<int32_t, 4> response{-77, -77, -77, -77};
    for (int replay = 0; replay < 20; ++replay)
    {
        ASSERT_TRUE(initialize_device_generation_control(
            0, response.size(), admission.depth_policy, control.data(),
            admission.initial_leading_row_disposition));
        EXPECT_EQ(control[kDeviceGenerationControlRequestComplete], 0);
        ASSERT_TRUE(complete_device_generation_forward(control.data()));
        EXPECT_EQ(control[kDeviceGenerationControlPublishedStateCommitCount], 1);
        EXPECT_EQ(control[kDeviceGenerationControlResponseTokenCount], 0);
        EXPECT_EQ(control[kDeviceGenerationControlTransactionCount], 1);
        EXPECT_EQ(control[kDeviceGenerationControlNextLeadingCommittedOutputCount], 0);
        EXPECT_EQ(validateDeviceGenerationTerminal(control, admission, response.size(), 0, 0),
            DeviceGenerationTerminalError::None);
        const auto terminal = control;
        EXPECT_TRUE(complete_device_generation_forward(control.data()));
        EXPECT_EQ(control, terminal);
        for (int token : response) EXPECT_EQ(token, -77);
        // Forward-only must not accept a sampled token or a speculative result.
        EXPECT_FALSE(append_ordinary_sample_to_device_generation(
            42, false, Source::DecodeLogits, response.data(), response.size(), control.data()));
    }
    auto invalid = admission;
    invalid.max_new_tokens = 1;
    EXPECT_FALSE(invalid.valid());
    invalid = admission;
    invalid.initial_leading_row_disposition = DeviceGenerationLeadingRowDisposition::PendingResponse;
    EXPECT_FALSE(invalid.valid());
    ASSERT_TRUE(initialize_device_generation_control(
        1, response.size(), DeviceGenerationPolicy::ordinary(), control.data()));
    EXPECT_FALSE(complete_device_generation_forward(control.data()));
    EXPECT_EQ(control[kDeviceGenerationControlPublishedStateCommitCount], 0);
}

TEST(OrdinaryGenerationController, EveryBudgetAndStopPositionMatchesSerialState)
{
    for (int budget : {1, 2, 3, 15, 16, 31, 256})
    {
        // Include no EOS, EOS on the first sample, and every intermediate stop.
        for (int stop = 0; stop <= budget; ++stop)
        {
            Control control{};
            std::array<int32_t, 256> response;
            response.fill(-77);
            ASSERT_TRUE(initialize_device_generation_control(
                budget, response.size(), DeviceGenerationPolicy::ordinary(),
                control.data()));
            const int emitted = stop == 0 ? budget : stop;
            for (int row = 0; row < emitted; ++row)
            {
                ASSERT_TRUE(append_ordinary_sample_to_device_generation(
                    1000 + row, stop == row + 1,
                    row == 0 ? Source::PrefillLogits : Source::DecodeLogits,
                    response.data(), response.size(), control.data()));
                EXPECT_EQ(control[kDeviceGenerationControlPublishedStateCommitCount], row);
                EXPECT_EQ(control[kDeviceGenerationControlResponseTokenCount], row + 1);
            }
            EXPECT_EQ(control[kDeviceGenerationControlRequestComplete], 1);
            EXPECT_EQ(control[kDeviceGenerationControlModelStopped], stop != 0);
            EXPECT_EQ(control[kDeviceGenerationControlRemainingTokenCount], budget - emitted);
            EXPECT_EQ(control[kDeviceGenerationControlTransactionCount], emitted);
            EXPECT_EQ(control[kDeviceGenerationControlNextLeadingCommittedOutputCount], 1);
            const DeviceGenerationAdmissionRequest admission{
                .request_count = 1, .max_new_tokens = budget,
                .depth_policy = DeviceGenerationPolicy::ordinary()};
            EXPECT_EQ(validateDeviceGenerationTerminal(control, admission,
                response.size(), 0, 0), DeviceGenerationTerminalError::None);
            for (int row = 0; row < emitted; ++row)
                EXPECT_EQ(response[row], 1000 + row);
            for (size_t row = emitted; row < response.size(); ++row)
                EXPECT_EQ(response[row], -77);
            for (int index : {kDeviceGenerationControlAcceptedSpeculativeTokenCount,
                              kDeviceGenerationControlConsumedVerifierRowCount,
                              kDeviceGenerationControlAttemptedDraftTokenCount,
                              kDeviceGenerationControlVerifierTokenCount,
                              kDeviceGenerationControlDepthUpdates})
                EXPECT_EQ(control[index], 0);
            // Terminal replay is absorbing even if unused sampler scratch is
            // poisoned. Neither response bytes nor the terminal ledger change.
            const auto terminal = control;
            const auto tokens = response;
            ASSERT_TRUE(append_ordinary_sample_to_device_generation(
                -1, false, Source::DecodeLogits, response.data(), response.size(),
                control.data()));
            EXPECT_EQ(control, terminal);
            EXPECT_EQ(response, tokens);
        }
    }
}

TEST(OrdinaryGenerationController, ContinuationConsumesPreviouslyEmittedConditionOnce)
{
    Control control{};
    std::array<int32_t, 4> response{};
    for (int repeat = 0; repeat < 20; ++repeat)
    {
        ASSERT_TRUE(initialize_device_generation_control(
            4, 4, DeviceGenerationPolicy::ordinary(), control.data(),
            DeviceGenerationLeadingRowDisposition::AlreadyEmitted));
        for (int row = 0; row < 4; ++row)
        {
            ASSERT_TRUE(append_ordinary_sample_to_device_generation(
                row, false, Source::DecodeLogits, response.data(), 4, control.data()));
            EXPECT_EQ(control[kDeviceGenerationControlPublishedStateCommitCount], row + 1);
        }
        EXPECT_EQ(control[kDeviceGenerationControlTransactionCount], 4);
        const DeviceGenerationAdmissionRequest admission{
            .request_count = 1, .max_new_tokens = 4,
            .depth_policy = DeviceGenerationPolicy::ordinary(),
            .initial_leading_row_disposition = DeviceGenerationLeadingRowDisposition::AlreadyEmitted};
        EXPECT_EQ(validateDeviceGenerationTerminal(control, admission, 4, 0, 0),
            DeviceGenerationTerminalError::None);
        auto wrong_frontier = admission;
        wrong_frontier.initial_leading_row_disposition = DeviceGenerationLeadingRowDisposition::PendingResponse;
        EXPECT_EQ(validateDeviceGenerationTerminal(control, wrong_frontier, 4, 0, 0),
            DeviceGenerationTerminalError::InvalidAlgorithmAccounting);
    }
}

TEST(OrdinaryGenerationController, WrongFrontierAndCapacityFailWithoutPartialResponse)
{
    for (int invalid = 0; invalid < 8; ++invalid)
    {
        Control control{};
        std::array<int32_t, 2> response{-77, -77};
        ASSERT_TRUE(initialize_device_generation_control(
            2, 2, DeviceGenerationPolicy::ordinary(), control.data()));
        Source source = Source::PrefillLogits;
        int token = 9;
        int capacity = 2;
        switch (invalid)
        {
        case 0: source = Source::DecodeLogits; break;
        case 1: source = static_cast<Source>(7); break;
        case 2: token = -1; break;
        case 3: capacity = 1; break;
        case 4: control[kDeviceGenerationControlResponseTokenCount] = -1; break;
        case 5: control[kDeviceGenerationControlRemainingTokenCount] = 0; break;
        case 6: control[kDeviceGenerationControlPublishedStateCommitCount] = 1; break;
        case 7: control[kDeviceGenerationControlResponseTokenCount] =
                    control[kDeviceGenerationControlTransactionCount] =
                    control[kDeviceGenerationControlPublishedStateCommitCount] =
                        std::numeric_limits<int>::max();
                source = Source::DecodeLogits;
                control[kDeviceGenerationControlNextLeadingCommittedOutputCount] = 1;
                break;
        }
        EXPECT_FALSE(append_ordinary_sample_to_device_generation(
            token, false, source, response.data(), capacity, control.data()));
        EXPECT_EQ(control[kDeviceGenerationControlOk], 0);
        EXPECT_EQ(control[kDeviceGenerationControlRequestComplete], 1);
        EXPECT_EQ(response, (std::array<int32_t, 2>{-77, -77}));
        const auto failed = control;
        EXPECT_FALSE(append_ordinary_sample_to_device_generation(
            10, false, Source::DecodeLogits, response.data(), 2, control.data()));
        EXPECT_EQ(control, failed);
    }
}

TEST(OrdinaryGenerationController, PrefillCannotBeSampledTwiceOrConsumeSpeculativeOutcome)
{
    Control control{};
    std::array<int32_t, 2> response{};
    ASSERT_TRUE(initialize_device_generation_control(
        2, 2, DeviceGenerationPolicy::ordinary(), control.data()));
    ASSERT_TRUE(append_ordinary_sample_to_device_generation(
        10, false, Source::PrefillLogits, response.data(), 2, control.data()));
    EXPECT_FALSE(append_ordinary_sample_to_device_generation(
        11, false, Source::PrefillLogits, response.data(), 2, control.data()));
    EXPECT_EQ(control[kDeviceGenerationControlResponseTokenCount], 1);

    ASSERT_TRUE(initialize_device_generation_control(
        2, 2, DeviceGenerationPolicy::ordinary(), control.data()));
    std::array<int, kSpeculativeBatchMetaCount> meta{};
    EXPECT_FALSE(append_speculative_outcome_to_device_generation(
        response.data(), 2, meta.data(), meta.size(), response.data(), 2, control.data()));
    EXPECT_EQ(control[kDeviceGenerationControlResponseTokenCount], 0);
    EXPECT_EQ(control[kDeviceGenerationControlErrorCode],
              static_cast<int>(DeviceGenerationError::InvalidDepthPolicy));

    ASSERT_TRUE(initialize_device_generation_control(
        2, 2, DeviceGenerationPolicy::fixed(1), control.data()));
    EXPECT_FALSE(append_ordinary_sample_to_device_generation(
        10, false, Source::PrefillLogits, response.data(), 2, control.data()));
    EXPECT_EQ(control[kDeviceGenerationControlResponseTokenCount], 0);
}

/** @test Terminal observation rejects changed admission and poisoned counters. */
TEST(OrdinaryGenerationController, TerminalContractRejectsForgedPolicyBudgetAndStatistics)
{
    Control control{};
    std::array<int32_t, 3> response{};
    const DeviceGenerationAdmissionRequest admission{
        .request_count = 1, .max_new_tokens = 3,
        .depth_policy = DeviceGenerationPolicy::ordinary()};
    ASSERT_TRUE(initialize_device_generation_control(3, 3, admission.depth_policy, control.data()));
    for (int i = 0; i < 3; ++i)
        ASSERT_TRUE(append_ordinary_sample_to_device_generation(i, false,
            i == 0 ? Source::PrefillLogits : Source::DecodeLogits, response.data(), 3, control.data()));
    ASSERT_EQ(validateDeviceGenerationTerminal(control, admission, 3, 0, 0),
        DeviceGenerationTerminalError::None);
    // Every ABI word has a validity constraint in the ordinary terminal. The
    // one legal nonzero movement row is separately authenticated by its source.
    for (int index = 0; index < kDeviceGenerationControlCount; ++index)
    {
        SCOPED_TRACE(index);
        for (int poison : {-1, std::numeric_limits<int>::max()})
        {
            auto invalid = control;
            invalid[index] = poison;
            EXPECT_NE(validateDeviceGenerationTerminal(invalid, admission, 3, 0, 0),
                DeviceGenerationTerminalError::None);
        }
    }
    auto changed_budget = admission;
    changed_budget.max_new_tokens = 4;
    EXPECT_EQ(validateDeviceGenerationTerminal(control, changed_budget, 4, 0, 0),
        DeviceGenerationTerminalError::InvalidResponseAccounting);
    control[kDeviceGenerationControlCurrentBatchLLEPMovementLayerCount] = 2;
    EXPECT_EQ(validateDeviceGenerationTerminal(control, admission, 3, 0, 2),
        DeviceGenerationTerminalError::None);
    EXPECT_EQ(validateDeviceGenerationTerminal(control, admission, 3, 0, 1),
        DeviceGenerationTerminalError::InvalidMovementEvidence);
}

/** @test The shared terminal gate preserves fixed, observed and dynamic MTP. */
TEST(OrdinaryGenerationController, TerminalContractAuthenticatesEverySpeculativeDepth)
{
    for (auto mode : {DeviceGenerationPolicyMode::Fixed,
                     DeviceGenerationPolicyMode::Observe,
                     DeviceGenerationPolicyMode::Dynamic})
    for (int depth = 1; depth <= DeviceGenerationPolicy::kMaximumSupportedDraftDepth; ++depth)
    {
        SCOPED_TRACE(::testing::Message() << "mode=" << static_cast<int>(mode) << " depth=" << depth);
        auto policy = DeviceGenerationPolicy::fixed(depth);
        policy.mode = mode;
        const DeviceGenerationAdmissionRequest admission{
            .request_count = 1, .max_new_tokens = depth + 1, .depth_policy = policy};
        Control control{};
        std::array<int32_t, 16> response{};
        std::array<int32_t, 16> tokens{};
        std::array<int, kSpeculativeBatchMetaCount> meta{};
        for (int i = 0; i <= depth; ++i)
            tokens[i] = 100 + i;
        meta[kSpecBatchMetaOk] = 1;
        meta[kSpecBatchMetaOutputCount] = depth + 1;
        meta[kSpecBatchMetaAcceptedSpeculativePrefix] = depth;
        meta[kSpecBatchMetaTargetVerifierStateCommitCount] = depth;
        meta[kSpecBatchMetaAllSpeculativeAccepted] = 1;
        meta[kSpecBatchMetaConsumedVerifierRows] = depth;
        meta[kSpecBatchMetaSampledTerminal] = 1;
        ASSERT_TRUE(initialize_device_generation_control(depth + 1, response.size(), policy, control.data()));
        ASSERT_EQ(prepare_device_generation_transaction_budget(depth + 1, depth + 1, control.data()), depth + 1);
        ASSERT_TRUE(append_speculative_outcome_to_device_generation(tokens.data(), depth + 1,
            meta.data(), meta.size(), response.data(), response.size(), control.data()));
        EXPECT_EQ(validateDeviceGenerationTerminal(control, admission, response.size(), 0, 0),
            DeviceGenerationTerminalError::None);
        auto forged = control;
        ++forged[kDeviceGenerationControlDepthWindowSize];
        EXPECT_EQ(validateDeviceGenerationTerminal(forged, admission, response.size(), 0, 0),
            DeviceGenerationTerminalError::ChangedPolicy);
        forged = control;
        forged[kDeviceGenerationControlTransactionCount] = std::numeric_limits<int>::max();
        EXPECT_EQ(validateDeviceGenerationTerminal(forged, admission, response.size(), 0, 0),
            DeviceGenerationTerminalError::InvalidAlgorithmAccounting);
    }
}
}
