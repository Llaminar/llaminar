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
#include "backends/GenerationPenaltyHistory.h"
#include "execution/mtp/DeviceGenerationContract.h"
#include "execution/mtp/OrdinaryGenerationGraphPlan.h"
#include "kernels/common/GenerationLogicalState.h"
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

/** @test Admission cannot silently discard an ordinary request's sampling law. */
TEST(OrdinaryGenerationController, OrdinarySamplingAdmissionRejectsUnsupportedOrIncompleteLaw)
{
    llaminar2::SamplingParams law;
    law.top_k = 40;
    DeviceGenerationAdmissionRequest request{
        .request_count = 1, .max_new_tokens = 384,
        .depth_policy = DeviceGenerationPolicy::ordinary(),
        .sampling_seeds = llaminar2::GenerationRequestSeeds(std::vector<uint64_t>{19}),
        .ordinary_sampling = law};
    ASSERT_TRUE(request.valid());
    for (int defect = 0; defect < 9; ++defect)
    {
        auto broken = request;
        switch (defect)
        {
        case 0: broken.sampling_seeds.reset(); break;
        case 1: broken.ordinary_sampling->temperature = -1; break;
        case 2: broken.ordinary_sampling->top_k = 0; break;
        case 3: broken.ordinary_sampling->top_k = kMaxTopK + 1; break;
        case 4: broken.ordinary_sampling->top_p = 0; break;
        case 5: broken.ordinary_sampling->top_p = 1.01F; break;
        case 6: broken.ordinary_sampling->frequency_penalty = std::numeric_limits<float>::quiet_NaN(); break;
        case 7: broken.ordinary_sampling->dry_multiplier = 1; break;
        case 8: broken.depth_policy = DeviceGenerationPolicy::fixed(2); break;
        }
        EXPECT_FALSE(broken.valid()) << defect;
    }
    request.sampling_seeds.reset();
    request.ordinary_sampling->temperature = 0;
    EXPECT_TRUE(request.valid()) << "Greedy needs no unused random seed";
}

/** @test Shared speculative parent ownership is charged only once in the BOM. */
TEST(OrdinaryGenerationController, OrdinaryGraphOwnerInventoryMatchesNativeAndHostedPrograms)
{
    using llaminar2::DeviceId;
    using llaminar2::OrdinaryGenerationGraphPlan;
    EXPECT_EQ(OrdinaryGenerationGraphPlan::additionalExecutableCount(DeviceId::cuda(0), false), 2u);
    EXPECT_EQ(OrdinaryGenerationGraphPlan::additionalExecutableCount(DeviceId::cuda(0), true), 1u);
    EXPECT_EQ(OrdinaryGenerationGraphPlan::additionalExecutableCount(DeviceId::rocm(0), false), 4u);
    EXPECT_EQ(OrdinaryGenerationGraphPlan::additionalExecutableCount(DeviceId::rocm(0), true), 3u);
    EXPECT_THROW(OrdinaryGenerationGraphPlan::additionalExecutableCount(DeviceId::cpu(), false), std::invalid_argument);
}

/** @test Admission owns resolved randomness; wrong row geometry fails before launch. */
TEST(OrdinaryGenerationController, RequestSeedAdmissionOwnsExactNonzeroRows)
{
    using llaminar2::GenerationRequestSeeds;
    EXPECT_THROW(GenerationRequestSeeds(std::vector<uint64_t>{}), std::invalid_argument);
    EXPECT_THROW(GenerationRequestSeeds(std::vector<uint64_t>{19, 0}), std::invalid_argument);
    std::vector<uint64_t> caller_seeds{19, std::numeric_limits<uint64_t>::max(), 142};
    DeviceGenerationAdmissionRequest request{
        .request_count = 3, .max_new_tokens = 4,
        .depth_policy = DeviceGenerationPolicy::ordinary(),
        .sampling_seeds = GenerationRequestSeeds(caller_seeds)};
    caller_seeds.assign(3, 0);
    EXPECT_TRUE(request.valid());
    EXPECT_EQ(request.sampling_seeds->values()[0], 19u);
    EXPECT_EQ(request.sampling_seeds->values()[1], std::numeric_limits<uint64_t>::max());
    request.request_count = 2;
    EXPECT_FALSE(request.valid());
    request.request_count = 3;
    request.max_new_tokens = 0;
    request.depth_policy = DeviceGenerationPolicy::forwardOnly();
    request.initial_leading_row_disposition = DeviceGenerationLeadingRowDisposition::AlreadyEmitted;
    EXPECT_FALSE(request.valid()) << "A forward-only transaction cannot consume sampling seeds";
    request.sampling_seeds.reset();
    EXPECT_TRUE(request.valid());
}

/** @test Arena registration and PMA planning share the same packed seed geometry. */
TEST(OrdinaryGenerationController, RequestSeedGeometryIncludesRetainedInactiveRows)
{
    using llaminar2::GenerationRequestSeedGeometry;
    EXPECT_THROW(GenerationRequestSeedGeometry(0, 1), std::invalid_argument);
    EXPECT_THROW(GenerationRequestSeedGeometry(1, 0), std::invalid_argument);
    for (const int ordinary : {1, 3, 8})
        for (const int retained : {1, 3, 15})
        {
            const GenerationRequestSeedGeometry geometry(ordinary, retained);
            EXPECT_EQ(geometry.requests(), static_cast<size_t>(std::max(ordinary, retained)));
            EXPECT_EQ(geometry.bytes(), geometry.requests() * sizeof(uint64_t));
            EXPECT_EQ(geometry.requests() * geometry.words_per_request * sizeof(int32_t), geometry.bytes());
        }
}

/** @test Unsampled logits are a valid frontier, not a fabricated pending token. */
TEST(OrdinaryGenerationController, InitialFrontierHasExplicitUnsampledAndSampledStates)
{
    using llaminar2::GenerationInitialFrontier;
    using llaminar2::GenerationLogicalStateInitialization;
    std::array<std::array<int32_t, 4>, 7> rows{};
    std::array<int32_t, 4> positions{0, 17, 4096, -1};
    std::array<int32_t, 4> tokens{3, -1, 42, 7};
    GenerationLogicalStateInitialization initialization{
        .positions = positions.data(), .request_count = 4,
        .output = {.base_cached_tokens = rows[0].data(), .target_positions = rows[1].data(),
            .accepted_state_counts = rows[2].data(), .next_condition_tokens = rows[3].data(),
            .all_drafts_accepted_flags = rows[4].data(), .stopped_flags = rows[5].data(),
            .publication_ok_flags = rows[6].data()}};
    for (const auto source : {GenerationInitialFrontier::UnsampledLogits,
                             GenerationInitialFrontier::SampledCondition})
    {
        initialization.source = source;
        initialization.sampled_tokens = source == GenerationInitialFrontier::SampledCondition
            ? tokens.data() : nullptr;
        ASSERT_TRUE(initialization.valid());
        for (auto &row : rows) row.fill(-77);
        for (int request = 0; request < 4; ++request)
        {
            const bool healthy = positions[request] >= 0 &&
                (source == GenerationInitialFrontier::UnsampledLogits || tokens[request] >= 0);
            EXPECT_EQ(initialization.initializeRequest(request), healthy);
            EXPECT_EQ(rows[0][request], positions[request]);
            EXPECT_EQ(rows[1][request], positions[request]);
            EXPECT_EQ(rows[2][request], 0);
            EXPECT_EQ(rows[3][request], healthy && source == GenerationInitialFrontier::SampledCondition
                ? tokens[request] : -1);
            EXPECT_EQ(rows[4][request], 0);
            EXPECT_EQ(rows[5][request], 0);
            EXPECT_EQ(rows[6][request], healthy ? 1 : 0);
        }
        const auto before = rows;
        EXPECT_FALSE(initialization.initializeRequest(-1));
        EXPECT_FALSE(initialization.initializeRequest(4));
        EXPECT_EQ(rows, before);
    }
    initialization.sampled_tokens = nullptr;
    EXPECT_FALSE(initialization.valid());
    initialization.source = GenerationInitialFrontier::UnsampledLogits;
    initialization.sampled_tokens = tokens.data();
    EXPECT_FALSE(initialization.valid()) << "An unsampled declaration cannot smuggle in a condition token";
    initialization.source = static_cast<GenerationInitialFrontier>(255);
    EXPECT_FALSE(initialization.valid());
}

/** @test Shared and padded per-request stop rows have one checked interpretation. */
TEST(OrdinaryGenerationController, StopPolicyIsBoundedSharedOrRequestLocal)
{
    EXPECT_TRUE(OrdinaryGenerationStopTokens{}.valid());
    EXPECT_EQ(OrdinaryGenerationStopTokens{}.evaluate(0, 42), 0);
    std::array<int32_t, 2 * (kSpeculativeBatchMaxStopTokens + 2)> tokens;
    tokens.fill(-1);
    tokens[0] = 42;
    tokens[kSpeculativeBatchMaxStopTokens + 2] = 43;
    OrdinaryGenerationStopTokens policy{tokens.data(), kSpeculativeBatchMaxStopTokens,
        kSpeculativeBatchMaxStopTokens + 2};
    ASSERT_TRUE(policy.valid());
    EXPECT_EQ(policy.evaluate(0, 42), 1);
    EXPECT_EQ(policy.evaluate(1, 42), 0);
    EXPECT_EQ(policy.evaluate(1, 43), 1);
    policy.request_stride = 0;
    EXPECT_EQ(policy.evaluate(1, 42), 1);
    EXPECT_EQ(policy.evaluate(1, 43), 0);
    tokens[kSpeculativeBatchMaxStopTokens - 1] = -2;
    EXPECT_EQ(policy.evaluate(0, 42), -1) << "A match must not hide malformed tail entries";
    policy.request_stride = -1;
    EXPECT_FALSE(policy.valid());
    policy.request_stride = kSpeculativeBatchMaxStopTokens - 1;
    EXPECT_FALSE(policy.valid());
    policy.request_stride = 0;
    policy.count = kSpeculativeBatchMaxStopTokens + 1;
    EXPECT_FALSE(policy.valid());
    policy.count = -1;
    EXPECT_FALSE(policy.valid());
    policy.count = 1;
    policy.tokens = nullptr;
    EXPECT_FALSE(policy.valid());
}

/** @test The shared transition owns both response and next-forward state, not just counters. */
TEST(OrdinaryGenerationController, PublicationAdvancesTheLiveFrontierExactlyOnce)
{
    for (int budget : {1, 2, 17, 256})
    for (auto leading : {DeviceGenerationLeadingRowDisposition::PendingResponse,
                         DeviceGenerationLeadingRowDisposition::AlreadyEmitted})
    {
        Control control{};
        std::array<int32_t, 256> response{};
        std::array<int32_t, 512> history{};
        int32_t sample = 42, stop = -1, position = 7, next = 19, stopped = 0, ok = 1;
        OrdinaryGenerationPublication publication{
            .request_count = 1, .sampled_tokens = &sample, .stop_tokens = {&stop, 1, 0},
            .source = leading == DeviceGenerationLeadingRowDisposition::PendingResponse
                ? Source::PrefillLogits : Source::DecodeLogits,
            .response_tokens = response.data(), .response_token_stride = response.size(),
            .control = control.data(), .control_stride = control.size(),
            .frontier = {.cached_tokens = &position, .next_condition_tokens = &next,
                .stopped_flags = &stopped, .publication_ok_flags = &ok,
                .request_capacity = 1, .context_capacity = 4096},
            .history = {history.data(), 512, 512, 1}};
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
            EXPECT_EQ(history[sample], 1);
            publication.source = Source::DecodeLogits;
        }
        const auto terminal = control;
        const int terminal_position = position, terminal_next = next;
        sample = -1; stop = -2;
        for (int repeat = 0; repeat < 20; ++repeat)
            ASSERT_TRUE(publish_ordinary_generation_request(publication, 0));
        EXPECT_EQ(control, terminal);
        EXPECT_EQ(position, terminal_position);
        EXPECT_EQ(next, terminal_next);
        for (int token = 0; token < 512; ++token)
            EXPECT_EQ(history[token], token >= 42 && token < 42 + budget ? 1 : 0);
    }
}

/** @test No invalid position, stop policy or sample partially commits the response. */
TEST(OrdinaryGenerationController, InvalidPublicationKeepsResponseAndFrontierUnadvanced)
{
    for (int defect = 0; defect < 6; ++defect) {
        Control control{};
        std::array<int32_t, 2> response{-77, -77};
        std::array<int32_t, 64> history{};
        int32_t sample = 42, stop = -1, position = 7, next = 19, stopped = 0, ok = 1;
        OrdinaryGenerationPublication publication{
            .request_count = 1, .sampled_tokens = &sample, .stop_tokens = {&stop, 1, 0},
            .source = Source::DecodeLogits, .response_tokens = response.data(), .response_token_stride = 2,
            .control = control.data(), .control_stride = control.size(),
            .frontier = {.cached_tokens = &position, .next_condition_tokens = &next,
                .stopped_flags = &stopped, .publication_ok_flags = &ok,
                .request_capacity = 1, .context_capacity = 8},
            .history = {history.data(), 64, 64, 1}};
        ASSERT_TRUE(initialize_device_generation_control(2, 2, DeviceGenerationPolicy::ordinary(),
            control.data(), DeviceGenerationLeadingRowDisposition::AlreadyEmitted));
        switch (defect) {
        case 0: position = -1; break;
        case 1: position = 8; break;
        case 2: position = std::numeric_limits<int32_t>::max(); break;
        case 3: ok = 0; break;
        case 4: stop = -2; break;
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
        EXPECT_EQ(history, (std::array<int32_t, 64>{}));
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
    std::array<int32_t, 64> history;
    history.fill(-77); // Forward-only must not read or repair sampler history.
    int32_t poison = -1, position = 7, next = 19, stopped = 0, ok = 1;
    OrdinaryGenerationPublication publication{
        .request_count = 1, .sampled_tokens = &poison, .stop_tokens = {&poison, 1, 0},
        .source = Source::DecodeLogits, .response_tokens = response.data(), .response_token_stride = 2,
        .control = control.data(), .control_stride = control.size(),
        .frontier = {.cached_tokens = &position, .next_condition_tokens = &next,
            .stopped_flags = &stopped, .publication_ok_flags = &ok,
            .request_capacity = 1, .context_capacity = 8},
        .history = {history.data(), 64, 64, 1}};
    ASSERT_TRUE(initialize_device_generation_control(0, 2, DeviceGenerationPolicy::forwardOnly(),
        control.data(), DeviceGenerationLeadingRowDisposition::AlreadyEmitted));
    for (int repeat = 0; repeat < 20; ++repeat)
        ASSERT_TRUE(publish_ordinary_generation_request(publication, 0));
    EXPECT_EQ(position, 8);
    EXPECT_EQ(next, -1);
    EXPECT_EQ(stopped, 0);
    EXPECT_EQ(ok, 1);
    EXPECT_EQ(response, (std::array<int32_t, 2>{-77, -77}));
    for (int count : history) EXPECT_EQ(count, -77);
}

/** @test Sampling may directly own the destination rows, including EOS at full context. */
TEST(OrdinaryGenerationController, AliasedSamplerPublicationNeedsNoExtraStaging)
{
    Control control{};
    std::array<int32_t, 1> response{-77};
    std::array<int32_t, 64> history{};
    int32_t position = std::numeric_limits<int32_t>::max(), next = 42, stopped = 0, ok = 1;
    const int32_t stop_tokens[] = {-1, 42, -1};
    OrdinaryGenerationPublication publication{
        .request_count = 1, .sampled_tokens = &next, .stop_tokens = {stop_tokens, 3, 0},
        .source = Source::PrefillLogits, .response_tokens = response.data(), .response_token_stride = 1,
        .control = control.data(), .control_stride = control.size(),
        .frontier = {.cached_tokens = &position, .next_condition_tokens = &next,
            .stopped_flags = &stopped, .publication_ok_flags = &ok,
            .request_capacity = 1, .context_capacity = std::numeric_limits<int32_t>::max()},
        .history = {history.data(), 64, 64, 1}};
    ASSERT_TRUE(initialize_device_generation_control(1, 1, DeviceGenerationPolicy::ordinary(), control.data()));
    ASSERT_TRUE(publish_ordinary_generation_request(publication, 0));
    EXPECT_EQ(position, std::numeric_limits<int32_t>::max());
    EXPECT_EQ(response[0], 42);
    EXPECT_EQ(next, 42);
    EXPECT_EQ(stopped, 1);
    EXPECT_EQ(ok, 1);
    EXPECT_EQ(history[42], 1);
    EXPECT_EQ(control[kDeviceGenerationControlModelStopped], 1);
    auto incomplete = publication;
    incomplete.frontier = {};
    EXPECT_FALSE(incomplete.valid());
    EXPECT_FALSE(publish_ordinary_generation_request(incomplete, 0));
    EXPECT_FALSE(publish_ordinary_generation_request(publication, 1));
}

/** @test Ordinary history cannot masquerade as an incomplete speculative branch. */
TEST(OrdinaryGenerationController, PenaltyHistoryProvenanceIsExplicit)
{
    using llaminar2::GenerationPenaltyHistory;
    int counts = 0, policy = 0, tokens = 0, rows = 1;
    const auto committed = GenerationPenaltyHistory::committed(&counts, &policy);
    EXPECT_TRUE(committed.admitsRows(1));
    EXPECT_FALSE(committed.admitsRows(0));
    EXPECT_FALSE(committed.admitsRows(2));
    EXPECT_EQ(committed.counts(), &counts);
    EXPECT_EQ(committed.policy(), &policy);
    EXPECT_EQ(committed.branchTokens(), nullptr);
    EXPECT_EQ(committed.activeRows(), nullptr);
    const auto speculative = GenerationPenaltyHistory::speculative(&counts, &policy, &tokens, &rows);
    EXPECT_TRUE(speculative.admitsRows(1));
    EXPECT_TRUE(speculative.admitsRows(16));
    EXPECT_FALSE(speculative.admitsRows(-1));
    EXPECT_EQ(speculative.branchTokens(), &tokens);
    EXPECT_EQ(speculative.activeRows(), &rows);
    EXPECT_THROW(GenerationPenaltyHistory::committed(nullptr, &policy), std::invalid_argument);
    EXPECT_THROW(GenerationPenaltyHistory::committed(&counts, nullptr), std::invalid_argument);
    EXPECT_THROW(GenerationPenaltyHistory::speculative(&counts, &policy, nullptr, &rows), std::invalid_argument);
    EXPECT_THROW(GenerationPenaltyHistory::speculative(&counts, &policy, &tokens, nullptr), std::invalid_argument);
}

/** @test History bounds/overflow fail before any token or frontier becomes visible. */
TEST(OrdinaryGenerationController, HistoryAdmissionAndCommitAreExclusiveAndFailureAtomic)
{
    constexpr int vocab = 64, stride = 67;
    std::array<int32_t, 2 * stride> counts{};
    OrdinaryGenerationHistory history{counts.data(), vocab, stride, 2};
    EXPECT_TRUE(history.validFor(2));
    EXPECT_FALSE(history.validFor(3));
    auto invalid = history;
    invalid.request_stride = vocab - 1;
    EXPECT_FALSE(invalid.validFor(2));
    invalid = history;
    invalid.counts = nullptr;
    EXPECT_FALSE(invalid.validFor(1));

    for (int defect = 0; defect < 5; ++defect)
    {
        SCOPED_TRACE(defect);
        Control control{};
        std::array<int32_t, 2> response{-77, -77};
        int32_t sample = 42, position = 7, next = 19, stopped = 0, ok = 1;
        counts.fill(0);
        counts[42] = defect == 2 ? -1 : defect == 3 ? INT32_MAX : 5;
        counts[stride + 42] = 17; // Same token on another request is another owner.
        if (defect == 0) sample = -1;
        if (defect == 1) sample = vocab;
        const auto before = counts;
        OrdinaryGenerationPublication publication{
            .request_count = 1, .sampled_tokens = &sample,
            .stop_tokens = {&sample, 1, 0}, .source = Source::PrefillLogits,
            .response_tokens = response.data(), .response_token_stride = 2,
            .control = control.data(), .control_stride = control.size(),
            .frontier = {&position, &next, &stopped, &ok, 1, 8},
            .history = history};
        ASSERT_TRUE(initialize_device_generation_control(2, 2,
            DeviceGenerationPolicy::ordinary(), control.data()));
        EXPECT_EQ(publish_ordinary_generation_request(publication, 0), defect == 4);
        if (defect == 4)
        {
            EXPECT_EQ(response[0], 42);
            EXPECT_EQ(counts[42], 6);
            EXPECT_EQ(stopped, 1);
            for (int replay = 0; replay < 20; ++replay)
                EXPECT_TRUE(publish_ordinary_generation_request(publication, 0));
            EXPECT_EQ(counts[42], 6) << "EOS/terminal replay cannot add history twice";
            EXPECT_EQ(counts[stride + 42], 17);
        }
        else
        {
            EXPECT_EQ(counts, before);
            EXPECT_EQ(response, (std::array<int32_t, 2>{-77, -77}));
            EXPECT_EQ(next, 19);
            EXPECT_EQ(ok, 0);
            EXPECT_EQ(control[kDeviceGenerationControlErrorCode],
                static_cast<int>(DeviceGenerationError::InvalidOrdinarySample));
        }
        EXPECT_EQ(position, 7) << "Prefill publication must not consume model state";
    }
}

TEST(OrdinaryGenerationController, PolicyIsExplicitAndSharesTheResidentStorageABI)
{
    static_assert(kDeviceGenerationControlCount == 51);
    static_assert(sizeof(DeviceGenerationPolicy) == 15 * sizeof(int));
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
        for (const int key : {kDeviceGenerationControlLearnedDepthEnabled,
                              kDeviceGenerationControlLearnedDepthBackend,
                              kDeviceGenerationControlLearnedDepthModelClass,
                              kDeviceGenerationControlLearnedDepthVerifyMode})
        {
            forged = control;
            ++forged[key];
            EXPECT_EQ(validateDeviceGenerationTerminal(forged, admission, response.size(), 0, 0),
                DeviceGenerationTerminalError::ChangedPolicy);
        }
        forged = control;
        forged[kDeviceGenerationControlLearnedDepthMatchedWindows] = 1;
        EXPECT_EQ(validateDeviceGenerationTerminal(forged, admission, response.size(), 0, 0),
            DeviceGenerationTerminalError::InvalidDepthStatistics);
        forged = control;
        forged[kDeviceGenerationControlTransactionCount] = std::numeric_limits<int>::max();
        EXPECT_EQ(validateDeviceGenerationTerminal(forged, admission, response.size(), 0, 0),
            DeviceGenerationTerminalError::InvalidAlgorithmAccounting);
    }
}
}
