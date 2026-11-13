/**
 * @file MTPDepthTransactionTraceCPU.h
 * @brief Lossless observations of the actual fresh host MTP controller.
 *
 * No device policy is replayed by these producers. The actual host config,
 * observation and decision author this finite untimed evidence; the benchmark
 * collector owns one fresh scalar request and resets between cases. Missing
 * response progress is explicitly unavailable, never inferred from acceptance.
 */
#pragma once
#include "execution/mtp/MTPDepthController.h"
#include "execution/mtp/MTPDepthReplayBuildIdentity.h"
#include "kernels/common/MTPDepthTransactionTrace.h"
#include "MTPDepthTransactionTracePerfStats.h"
#include <nlohmann/json.hpp>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>
namespace llaminar2
{
/**
 * @brief Reject unsupported trace request modes before any model work.
 * @param mtp Actual startup or reset request policy.
 * @return True only for enabled scalar positive-depth supported MTP.
 *
 * Depth-zero CPU bypass has a separate observation entrypoint. It is excluded
 * explicitly instead of claiming that recordStep traces contain its inputs.
 * The canonical execution-depth resolver retains ownership of automatic caps.
 */
inline bool validMTPDepthTransactionTracePolicy(const MTPRuntimeConfig &mtp)
{
    const auto mode = mtp.depth_policy.mode;
    const bool known = mode == MTPDepthPolicyMode::Fixed || mode == MTPDepthPolicyMode::Observe ||
        mode == MTPDepthPolicyMode::Dynamic;
    return mtp.enabled && mtp.max_request_batch == 1 && known &&
        (mtp.verify_mode == MTPVerifyMode::Greedy || mtp.verify_mode == MTPVerifyMode::SpeculativeSampling) &&
        mtp.depth_policy.min_samples >= 1 && mtp.draft_tokens > 0 &&
        mtp.draft_tokens <= kMaximumSupportedMTPDraftDepth &&
        (mode == MTPDepthPolicyMode::Fixed || mtp.depth_policy.min_depth >= 1) &&
        resolveMTPMaximumExecutionDraftDepth(mtp) <= kMaximumSupportedMTPDraftDepth;
}
/**
 * @brief Encode every actual host observation, including fixed/clipped outcomes.
 * @param observation Exact arguments passed to the host controller.
 * @param decision Actual returned host decision, evaluated or not.
 * @param transactions Actual committed physical verifier count, starting at one.
 * @param evaluated_windows Actual host stats.windows after recordStep.
 * @return Compact lossless record; actual emitted delta is present; cumulative response progress is unavailable.
 * @throws std::invalid_argument On an unrepresentable observation/result.
 */
inline sampling_math::MTPDepthTransactionTraceRecord makeCPUDepthTransactionTraceRecord(
    const MTPDepthObservation &observation, const MTPDepthDecision &decision,
    std::uint64_t transactions, std::uint64_t evaluated_windows)
{
    if (!observation.publication || !observation.publication->valid() ||
        observation.effective_depth < 0 || observation.effective_depth > 15 ||
        transactions == 0 || transactions > sampling_math::kMTPDepthTransactionTraceCapacity ||
        evaluated_windows > transactions || !observation.newly_emitted_tokens ||
        *observation.newly_emitted_tokens == 0 || *observation.newly_emitted_tokens > 16 ||
        *observation.newly_emitted_tokens > static_cast<std::uint32_t>(decision.old_depth + 1) ||
        *observation.newly_emitted_tokens != observation.publication->newlyEmittedTokens() ||
        static_cast<std::uint64_t>(*observation.newly_emitted_tokens) +
            static_cast<unsigned>(observation.publication->incomingDebt()) >
            static_cast<std::uint64_t>(observation.effective_depth + 1))
        throw std::invalid_argument("MTP CPU transaction trace exceeds its finite committed extent");
    const sampling_math::MTPDepthTransactionTraceValues values{
        .pre_depth = decision.old_depth,
        .accepted_prefix = observation.accepted_speculative_prefix,
        .post_depth = decision.new_depth,
        .rollback = observation.rollback,
        .budget_limited = observation.budget_limited,
        .incoming_debt = observation.publication->incomingDebt(),
        .outgoing_debt = observation.publication->outgoingDebt(),
        .model_stopped = observation.publication->modelStopped(),
        .requested_depth = observation.requested_depth,
        .effective_depth = observation.effective_depth,
        .host_observation = true,
        .recommended_depth = decision.recommended_depth,
        .token_witness = static_cast<int>(*observation.newly_emitted_tokens),
        .evaluated_windows = static_cast<int>(evaluated_windows),
    };
    sampling_math::MTPDepthTransactionTraceRecord record{};
    if (!sampling_math::packMTPDepthTransactionTraceRecord(values, &record))
        throw std::invalid_argument("MTP CPU transaction trace cannot losslessly encode the real observation");
    return record;
}
/**
 * @brief Serialize the actual resolved fresh host policy without decimal loss.
 * @param controller Fresh host owner before its first physical observation.
 * @param configured_draft_tokens Actual request argument used to configure it.
 * @param publication Actual first transaction witness; its incoming debt is not inferred.
 * @return Complete replay policy and independently compiled source identity.
 * @throws std::invalid_argument If this is not a fresh supported host controller.
 *
 * nlohmann JSON emits doubles with round-trip precision. Six-decimal strings
 * or device PPM conversion would change the CPU comparison boundaries.
 */
inline nlohmann::json cpuDepthTransactionTraceInitialToJson(
    const MTPDepthController &controller, int configured_draft_tokens,
    const MTPPublicationWitness &publication)
{
    const auto &policy = controller.resolvedConfig();
    const auto &scope = policy.scope;
    if (!publication.valid() || controller.stats().windows != 0 || controller.currentDepth() != policy.initial_depth ||
        controller.lastDecision().reason != MTPDepthDecisionReason::WindowNotReady ||
        controller.lastDecision().window.verifier_runs != 0 ||
        policy.min_depth < 1 || policy.max_depth > kMaximumSupportedMTPDraftDepth ||
        !policy.demote_zero_accept_rate || configured_draft_tokens < 1 ||
        !std::isfinite(policy.promote_full_accept_rate) ||
        !std::isfinite(*policy.demote_zero_accept_rate) || !std::isfinite(policy.demote_acceptance_rate))
        throw std::invalid_argument("MTP CPU transaction trace requires a fresh positive-depth resolved policy");
    nlohmann::json hashes = nlohmann::json::object();
    for (const auto &[key, value] : mtp_depth_replay_identity::kSourceHashes)
        hashes[std::string(key)] = std::string(value);
    return {
        {"schema", "llaminar.mtp.cpu_depth_transaction_trace_initial.v3"},
        {"compiled_identity", {{"schema", std::string(mtp_depth_replay_identity::kSchema)},
            {"sources", std::move(hashes)}}},
        {"verify_mode", static_cast<int>(controller.verifyMode())},
        {"configured_draft_tokens", configured_draft_tokens},
        {"initial_depth", controller.currentDepth()},
        {"initial_incoming_debt", publication.incomingDebt()},
        {"policy", {
            {"mode", static_cast<int>(policy.mode)}, {"backend", static_cast<int>(policy.backend)},
            {"model_class", static_cast<int>(policy.model_class)},
            {"scope", {{"verifier_width_policy", static_cast<int>(scope.verifier_width_policy)},
                {"tp_width", scope.tp_width}, {"pp_stages", scope.pp_stages},
                {"retained_depth", scope.retained_depth}, {"hidden_dim", scope.hidden_dim},
                {"main_layers", scope.main_layers}, {"expert_count", scope.expert_count},
                {"experts_per_token", scope.experts_per_token},
                {"compute_capability_major", scope.compute_capability_major},
                {"compute_capability_minor", scope.compute_capability_minor}}},
            {"min_depth", policy.min_depth}, {"max_depth", policy.max_depth},
            {"initial_depth", policy.initial_depth}, {"window_size", policy.window_size},
            {"min_samples", policy.min_samples}, {"cooldown_steps", policy.cooldown_steps},
            {"promote_consecutive_windows", policy.promote_consecutive_windows},
            {"use_generated_policy", policy.use_generated_policy},
            {"promote_full_accept_rate", policy.promote_full_accept_rate},
            {"demote_zero_accept_rate", *policy.demote_zero_accept_rate},
            {"demote_acceptance_rate", policy.demote_acceptance_rate}}}};
}
/**
 * @brief Publish the actual CPU initial policy once per bounded measured request.
 * @param controller Actual fresh host controller.
 * @param configured_draft_tokens Original request's configure argument.
 * @param publication Actual first committed transaction's incoming debt authority.
 * @param device Actual continuation participant.
 * @param identity Request ownership within the per-case collector epoch.
 */
inline void publishCPUDepthTransactionTraceInitial(const MTPDepthController &controller,
    int configured_draft_tokens, const MTPPublicationWitness &publication,
    const std::string &device, const std::string &identity)
{
    PerfStatsCollector::addCounter("mtp", "depth_transaction_trace.initial", 1.0, "decode", device,
        {{"diagnostic", "untimed_depth_transaction_trace_v3"}, {"authority", "cpu_host"},
         {"observation_identity", identity},
         {"policy_json", cpuDepthTransactionTraceInitialToJson(controller, configured_draft_tokens, publication).dump()}});
}
} // namespace llaminar2
