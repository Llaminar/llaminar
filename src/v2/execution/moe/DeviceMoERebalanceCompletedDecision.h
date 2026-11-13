/**
 * @file DeviceMoERebalanceCompletedDecision.h
 * @brief Single-writer, request-scoped evidence of successfully finalized native decisions.
 *
 * Planner status is scratch, not completion. Only the successful durable
 * publication finalizer commits this bounded record, including a genuine
 * no-move decision. Terminal readers archive the count and latest receipt;
 * neither host submissions nor optional telemetry can author these values.
 */
#pragma once
#include "DeviceMoERebalanceABI.h"
#include "DeviceMoERebalanceLoadSpreadProof.h"
#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MOE_DECISION_HD __host__ __device__
#else
#define LLAMINAR_MOE_DECISION_HD
#endif
namespace llaminar2 {
/** @brief Availability of optional values in this bounded native receipt. */
enum class DeviceMoECompletedDecisionObservation : std::uint32_t { PlannerRejections = 1u << 0 };
/** Exact wire mask shared by publisher, validator and terminal projection. */
inline constexpr std::uint32_t kNativeMoEPlannerRejectionsObserved =
    static_cast<std::uint32_t>(DeviceMoECompletedDecisionObservation::PlannerRejections);
/** @brief Latest completed policy result; request reset clears the entire record. */
struct DeviceMoERebalanceCompletedDecision {
    std::uint64_t completed_windows = 0; ///< Successful finalizations, not planner launches.
    std::uint64_t candidate_epoch = 0; ///< Actual reservation; no-ops may reuse this epoch.
    std::uint64_t routed_observations = 0; ///< Native planner's sealed pre-wave load total.
    std::uint32_t command_epoch = 0; ///< Zero is legitimate for a no-move decision.
    std::uint32_t command_count = 0;
    std::uint32_t accepted_cycles = 0;
    std::uint32_t rejected_cycles = 0;
    std::uint32_t committed_decode_rounds = 0; ///< Actual device cadence at completion.
    /** Bit zero proves rejected_cycles was observed; retained load proofs omit that field. */
    std::uint32_t observation_flags = 0;
    /** @return Semantic identity, independent of padding or a host counter. */
    LLAMINAR_MOE_DECISION_HD bool operator==(const DeviceMoERebalanceCompletedDecision &b) const noexcept {
        return completed_windows == b.completed_windows && candidate_epoch == b.candidate_epoch &&
            routed_observations == b.routed_observations && command_epoch == b.command_epoch &&
            command_count == b.command_count && accepted_cycles == b.accepted_cycles &&
            rejected_cycles == b.rejected_cycles && committed_decode_rounds == b.committed_decode_rounds &&
            observation_flags == b.observation_flags;
    }
    /** @return A cleared record or a complete representable device receipt. */
    LLAMINAR_MOE_DECISION_HD bool valid() const noexcept {
        if (!completed_windows) return *this == DeviceMoERebalanceCompletedDecision{};
        return candidate_epoch != 0 && !(observation_flags & ~kNativeMoEPlannerRejectionsObserved) &&
            ((observation_flags & kNativeMoEPlannerRejectionsObserved) || !rejected_cycles) && ((command_count == 0) == (command_epoch == 0)) &&
            accepted_cycles <= command_count / 2u;
    }
};
static_assert(sizeof(DeviceMoERebalanceCompletedDecision) == moe_rebalance_abi::kCompletedDecisionBytes);
static_assert(std::is_trivially_copyable_v<DeviceMoERebalanceCompletedDecision>);
/** @brief Preparation never publishes evidence before the real finalization succeeds. */
enum class DeviceMoECompletedDecisionDisposition : std::uint32_t { Invalid, Inactive, Record };
/** @brief Small shared scratch retained only through one finalizer invocation. */
struct DeviceMoECompletedDecisionAppend {
    DeviceMoERebalanceCompletedDecision receipt;
    DeviceMoECompletedDecisionDisposition disposition = DeviceMoECompletedDecisionDisposition::Invalid;
};
/**
 * @brief Authenticate planner evidence without confusing a poll with a decision.
 * @param controller Original native owner, still healthy at finalization.
 * @param status Exact planner scratch from this same maintenance transaction.
 * @param candidate_epoch Actual reserved model-family generation.
 * @param command_epoch Prepared command identity, zero for no-move.
 * @param command_count Exact successful apply command extent, zero for no-move.
 * @return Prepared record; caller commits only after actual publication/abort success.
 *
 * This body is shared by native finalizers and device-free replay. Invalid or
 * incomplete work cannot advance the count. Load is kept in routed-row units;
 * no economic nanoseconds or movement bytes are inferred from it.
 */
template<class Controller, class Status>
LLAMINAR_MOE_DECISION_HD inline DeviceMoECompletedDecisionAppend prepareDeviceMoECompletedDecision(
    const Controller &controller, const Status &status, std::uint64_t candidate_epoch,
    std::uint32_t command_epoch, std::uint32_t command_count) noexcept {
    DeviceMoECompletedDecisionAppend out;
    if (controller.magic != moe_rebalance_abi::kMagic || controller.version != moe_rebalance_abi::kVersion ||
        controller.participant_count < 2 || controller.participant_id >= controller.participant_count ||
        controller.last_error_code || !controller.completed_decision.valid() ||
        status.magic != controller.magic || status.version != controller.version ||
        status.status_code > moe_rebalance_abi::kStatusWindowNotReady || status.invalid_runtime_layers || status.plan_overflow ||
        status.payload_bucket_overflow || !candidate_epoch || ((command_count == 0) != (command_epoch == 0)))
        return out;
    if (status.status_code == moe_rebalance_abi::kStatusWindowNotReady || !status.windows_observed || status.skipped_not_ready ||
        status.skipped_busy_wave) {
        // A retained moving wave must use its sealed command proof instead.
        // Scratch from a later skipped planner cannot authenticate that wave.
        if (!command_count) out.disposition = DeviceMoECompletedDecisionDisposition::Inactive;
        return out;
    }
    if (status.windows_observed != 1u || !status.window_required_slots ||
        status.window_ready_slots < status.window_required_slots ||
        (command_count && status.last_epoch != command_epoch) ||
        status.dynamic_ownership_swap_accepts > command_count / 2u ||
        controller.completed_decision.completed_windows == UINT64_MAX ||
        candidate_epoch < controller.completed_decision.candidate_epoch ||
        controller.decode_rounds_committed < controller.completed_decision.committed_decode_rounds)
        return out;
    out.receipt = {controller.completed_decision.completed_windows + 1u,
        candidate_epoch, status.pre_wave_load_total, command_epoch, command_count,
        status.dynamic_ownership_swap_accepts, status.dynamic_ownership_swap_rejections,
        controller.decode_rounds_committed, kNativeMoEPlannerRejectionsObserved};
    out.disposition = DeviceMoECompletedDecisionDisposition::Record;
    return out;
}
/**
 * @brief Prepare a completed moving decision from its retained immutable command proof.
 * @param controller Exact healthy controller retaining that PreparedForPublication wave.
 * @param proof Actual native ownership policy inputs, carried with the commands.
 * @param candidate_epoch Actual reserved family generation.
 * @param command_epoch Exact retained wave identity authenticated by the finalizer.
 * @param command_count Complete paired command extent, never allocation capacity.
 * @return Prepared receipt, never a mutation before the selector/journal commit.
 *
 * A busy reservation may postpone publication beyond its planner pass. The
 * current scratch can then describe WindowNotReady; it is not the old wave's
 * authority. Rejection counts are absent from the sealed load proof, so their
 * availability bit remains clear instead of reporting a fabricated zero.
 */
template<class Controller>
LLAMINAR_MOE_DECISION_HD inline DeviceMoECompletedDecisionAppend prepareDeviceMoECompletedOwnershipDecision(
    const Controller &controller, const DeviceMoERebalanceLoadSpreadProof &proof,
    std::uint64_t candidate_epoch, std::uint32_t command_epoch, std::uint32_t command_count) noexcept {
    DeviceMoECompletedDecisionAppend out;
    if (controller.magic != moe_rebalance_abi::kMagic || controller.version != moe_rebalance_abi::kVersion ||
        controller.participant_count < 2 || controller.participant_id >= controller.participant_count ||
        controller.last_error_code || !controller.completed_decision.valid() || !proof.valid() ||
        !candidate_epoch || !command_epoch || !command_count || command_count % 2u ||
        proof.ownership_swap_accepts != command_count / 2u ||
        controller.completed_decision.completed_windows == UINT64_MAX ||
        candidate_epoch < controller.completed_decision.candidate_epoch ||
        controller.decode_rounds_committed < controller.completed_decision.committed_decode_rounds)
        return out;
    out.receipt = {controller.completed_decision.completed_windows + 1u,
        candidate_epoch, proof.pre_wave_total, command_epoch, command_count,
        proof.ownership_swap_accepts, 0u, controller.decode_rounds_committed, 0u};
    out.disposition = DeviceMoECompletedDecisionDisposition::Record;
    return out;
}
/**
 * @brief Commit only the prepared next ordinal, after actual successful finalization.
 * @param state Original controller's embedded record, the sole writer authority.
 * @param append Prepared immutable scalar proof from this invocation.
 * @return False on stale/reused ordinal; inactive work leaves the record untouched.
 */
LLAMINAR_MOE_DECISION_HD inline bool commitDeviceMoECompletedDecision(
    DeviceMoERebalanceCompletedDecision &state, const DeviceMoECompletedDecisionAppend &append) noexcept {
    if (append.disposition == DeviceMoECompletedDecisionDisposition::Inactive) return true;
    if (append.disposition != DeviceMoECompletedDecisionDisposition::Record ||
        state.completed_windows == UINT64_MAX ||
        append.receipt.completed_windows != state.completed_windows + 1u || !append.receipt.valid()) return false;
    state = append.receipt;
    return true;
}
} // namespace llaminar2
#undef LLAMINAR_MOE_DECISION_HD
