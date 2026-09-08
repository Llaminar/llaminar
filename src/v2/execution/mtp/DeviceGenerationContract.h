/**
 * @file DeviceGenerationContract.h
 * @brief Immutable admission and terminal validation for resident generation.
 *
 * The device ledger owns live counters. Host code retains only the admitted
 * request, then authenticates the terminal ledger against it after the final
 * event. Ordinary and speculative algorithms share that lifecycle but have
 * different state-commit and depth semantics. Keeping validation here makes
 * those differences explicit and testable without loading a backend or model.
 */
#pragma once

#include "kernels/common/SamplingMath.h"
#include <algorithm>
#include <cstdint>
#include <span>

namespace llaminar2
{
/**
 * @brief Complete immutable policy installed at one GPU request admission.
 *
 * An emitted correction must be consumed without returning it twice. Keeping
 * row ownership, budget and geometry together prevents rank/backend layers
 * from silently dropping part of that admission when surfacing the response.
 */
struct DeviceGenerationAdmissionRequest
{
    int request_count = 0; ///< Independent controller rows.
    int max_new_tokens = 0; ///< New response tokens permitted per row.
    sampling_math::DeviceGenerationPolicy depth_policy =
        sampling_math::DeviceGenerationPolicy::fixed(0); ///< Explicit algorithm and depth policy.
    sampling_math::DeviceGenerationLeadingRowDisposition initial_leading_row_disposition =
        sampling_math::DeviceGenerationLeadingRowDisposition::PendingResponse;

    /** @return Whether every immutable admission field is implemented and valid. */
    [[nodiscard]] bool valid() const noexcept
    {
        return request_count > 0 && sampling_math::valid_device_generation_admission(
            depth_policy, max_new_tokens, initial_leading_row_disposition);
    }
};

/** @brief First terminal invariant that failed; never a recoverable replay choice. */
enum class DeviceGenerationTerminalError
{
    None,
    InvalidAdmission,
    InvalidCompletion,
    InvalidResponseAccounting,
    ChangedPolicy,
    InvalidAlgorithmAccounting,
    InvalidDepthStatistics,
    InvalidMovementEvidence,
};

/**
 * @brief Authenticate one terminal controller row against its immutable request.
 * @param row Complete resident ABI copied at the terminal event, not live state.
 * @param admission Original request retained by the single admission owner.
 * @param response_capacity Physical token slots available in this response row.
 * @param request_index Row ordinal in the admitted request batch.
 * @param llep_layer_count Valid current-batch evidence source size, or zero.
 * @return First failed invariant, or None for a publishable terminal response.
 *
 * Widen counter arithmetic before sums/products so corrupt INT_MAX fields
 * cannot invoke host signed overflow while diagnosing a failed device request.
 */
[[nodiscard]] inline DeviceGenerationTerminalError validateDeviceGenerationTerminal(
    std::span<const int, sampling_math::kDeviceGenerationControlCount> row,
    const DeviceGenerationAdmissionRequest &admission,
    int response_capacity, int request_index, int llep_layer_count) noexcept
{
    using namespace sampling_math;
    using Error = DeviceGenerationTerminalError;
    if (!admission.valid() || response_capacity < admission.max_new_tokens ||
        request_index < 0 || request_index >= admission.request_count || llep_layer_count < 0)
        return Error::InvalidAdmission;
    if (row[kDeviceGenerationControlOk] != 1 ||
        row[kDeviceGenerationControlRequestComplete] != 1 ||
        row[kDeviceGenerationControlErrorCode] != static_cast<int>(DeviceGenerationError::None))
        return Error::InvalidCompletion;

    const int response = row[kDeviceGenerationControlResponseTokenCount];
    const int remaining = row[kDeviceGenerationControlRemainingTokenCount];
    const int stopped = row[kDeviceGenerationControlModelStopped];
    const int transactions = row[kDeviceGenerationControlTransactionCount];
    const int leading = row[kDeviceGenerationControlNextLeadingCommittedOutputCount];
    const int published = row[kDeviceGenerationControlPublishedStateCommitCount];
    if (response < 0 || response > response_capacity || remaining < 0 ||
        static_cast<int64_t>(response) + remaining != admission.max_new_tokens ||
        (stopped != 0 && stopped != 1) || (stopped == 0 && remaining != 0) ||
        transactions <= 0 || (leading != 0 && leading != 1) ||
        row[kDeviceGenerationControlTransactionCommitBudget] != 0)
        return Error::InvalidResponseAccounting;

    // Policy is request metadata, not a tunable device shadow. Dynamic mode
    // may change its current depth, but never rewrite its admitted boundaries.
    const auto &policy = admission.depth_policy;
    if (row[kDeviceGenerationControlDepthPolicyMode] != static_cast<int>(policy.mode) ||
        row[kDeviceGenerationControlMinimumDraftDepth] != policy.minimum_depth ||
        row[kDeviceGenerationControlMaximumDraftDepth] != policy.maximum_depth ||
        row[kDeviceGenerationControlDepthWindowSize] != policy.window_size ||
        row[kDeviceGenerationControlDepthMinimumSamples] != policy.minimum_samples ||
        row[kDeviceGenerationControlDepthCooldownSteps] != policy.cooldown_steps ||
        row[kDeviceGenerationControlDepthPromoteConsecutiveWindows] != policy.promote_consecutive_windows ||
        row[kDeviceGenerationControlDepthPromoteFullAcceptRatePPM] != policy.promote_full_accept_rate_ppm ||
        row[kDeviceGenerationControlDepthDemoteZeroAcceptRatePPM] != policy.demote_zero_accept_rate_ppm ||
        row[kDeviceGenerationControlDepthDemoteAcceptanceRatePPM] != policy.demote_acceptance_rate_ppm)
        return Error::ChangedPolicy;

    const int final_depth = row[kDeviceGenerationControlCurrentDraftDepth];
    const int accepted = row[kDeviceGenerationControlAcceptedSpeculativeTokenCount];
    const int rejected = row[kDeviceGenerationControlRejectedTransactionCount];
    const int consumed = row[kDeviceGenerationControlConsumedVerifierRowCount];
    const int attempted = row[kDeviceGenerationControlAttemptedDraftTokenCount];
    const int verifier = row[kDeviceGenerationControlVerifierTokenCount];
    const int last_depth = row[kDeviceGenerationControlLastTransactionDraftDepth];
    const int last_emitted = row[kDeviceGenerationControlLastTransactionEmittedTokenCount];
    if (policy.isOrdinary() || policy.isForwardOnly())
    {
        const int initial_leading = device_generation_leading_committed_output_count(
            admission.initial_leading_row_disposition);
        // The last emitted token is deliberately unconsumed, including EOS.
        // A pending prefill sample adds one response and no model-state row.
        const bool operation_accounting_valid = policy.isForwardOnly()
            ? response == 0 && transactions == 1 && leading == 0 && published == 1 &&
                stopped == 0 && last_emitted == 0
            : response > 0 && transactions == response && leading == 1 &&
                published == static_cast<int64_t>(response) + initial_leading - 1 && last_emitted == 1;
        if (!operation_accounting_valid ||
            accepted != 0 || rejected != 0 || consumed != 0 || attempted != 0 ||
            verifier != 0 || final_depth != 0 || last_depth != 0 ||
            row[kDeviceGenerationControlActiveVerifierRowCount] != 0)
            return Error::InvalidAlgorithmAccounting;
        for (int index = kDeviceGenerationControlDepthStepsSinceChange;
             index <= kDeviceGenerationControlDepthLastRecommendedDepth; ++index)
        {
            if (row[index] != 0)
                return Error::InvalidDepthStatistics;
        }
    }
    else
    {
        if ((stopped != 0 && leading != 0) || accepted < 0 || rejected < 0 ||
            rejected > transactions || consumed < accepted || published < 0 ||
            published > static_cast<int64_t>(response) + 1 ||
            final_depth < policy.minimum_depth || final_depth > policy.maximum_depth ||
            attempted < static_cast<int64_t>(transactions) * policy.minimum_depth ||
            attempted > static_cast<int64_t>(transactions) * policy.maximum_depth ||
            verifier != static_cast<int64_t>(attempted) + transactions ||
            last_depth < policy.minimum_depth || last_depth > policy.maximum_depth ||
            last_emitted <= 0 || last_emitted > response)
            return Error::InvalidAlgorithmAccounting;
        const int runs = row[kDeviceGenerationControlDepthWindowVerifierRuns];
        const int evaluated = row[kDeviceGenerationControlDepthEvaluatedWindows];
        const int updates = row[kDeviceGenerationControlDepthUpdates];
        const int promotions = row[kDeviceGenerationControlDepthPromotions];
        const int demotions = row[kDeviceGenerationControlDepthDemotions];
        if (runs < 0 || runs >= std::max(policy.window_size, policy.minimum_samples) ||
            row[kDeviceGenerationControlDepthWindowAttemptedTokens] < 0 ||
            evaluated < 0 || updates < 0 || updates > evaluated || promotions < 0 || demotions < 0 ||
            updates != static_cast<int64_t>(promotions) + demotions ||
            (policy.mode != DeviceGenerationPolicyMode::Dynamic && updates != 0))
            return Error::InvalidDepthStatistics;
    }

    const int movement = row[kDeviceGenerationControlCurrentBatchLLEPMovementLayerCount];
    const int non_owner = row[kDeviceGenerationControlCurrentBatchLLEPNonOwnerAssignmentLayerCount];
    if (movement < 0 || non_owner < 0 || movement > llep_layer_count || non_owner > llep_layer_count ||
        (request_index != 0 && (movement != 0 || non_owner != 0)))
        return Error::InvalidMovementEvidence;
    return Error::None;
}
}
