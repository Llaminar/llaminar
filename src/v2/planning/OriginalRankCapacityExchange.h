/**
 * @file OriginalRankCapacityExchange.h
 * @brief One original-parent transport for certified capacity and final immutable-table metadata.
 *
 * CAP4 retains actual rank/device owner charges and observed capacity. Pipeline
 * framing adds only original stage identity so unequal local stage populations
 * exchange once, without duplicating a shared logical stage or remote bytes.
 */
#pragma once
#include "execution/moe/MoEOverlayCapacityAdmission.h"
#include "execution/global_pp/GlobalPPTopology.h"
#include "planning/ImmutableRowTableAdmissionWire.h"
#include "interfaces/IMPIContext.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
namespace llaminar2 {
/**
 * @brief Encode certified original-rank CAP4 records in little-endian order.
 * @param budgets Certified physical allocator contributions owned by the sender.
 * @return Exact encoded metadata bytes, or an empty vector for no contributions.
 * @throws std::overflow_error If the complete record extent overflows size_t.
 */
std::vector<std::uint8_t> serializeOverlayCapacityBudgets(
    const std::vector<MoEOverlayBoundPhysicalMemoryBudget>& budgets);

/**
 * @brief Append decoded certified CAP4 records, authenticating their original rank.
 * @param bytes Complete metadata message that contains the rank segment.
 * @param begin Inclusive byte offset of the first record.
 * @param end Exclusive byte offset after the last record.
 * @param contributing_rank Original sender expected on every record.
 * @param output Existing vector to append to. Discard this vector if decoding fails.
 * @throws std::invalid_argument If framing, allocator identity or certified charges are invalid.
 * @throws std::overflow_error If a decoded charge cannot be represented.
 */
void parseOverlayCapacityBudgetSegment(std::span<const std::uint8_t> bytes,
    std::size_t begin, std::size_t end, int contributing_rank,
    std::vector<MoEOverlayBoundPhysicalMemoryBudget>& output);

/**
 * @brief Gather ordinary overlay capacity records using the original setup context.
 * @param local_budgets Certified budgets for the local original rank.
 * @param mpi_context Original context, or null for process-local planning.
 * @return Certified contributions in original rank order.
 * @throws std::invalid_argument If a received segment fails CAP4 validation.
 * @throws std::overflow_error If the encoded metadata exceeds MPI's count range.
 * @throws std::runtime_error If the original context reports a transport failure.
 */
std::vector<MoEOverlayBoundPhysicalMemoryBudget> gatherOverlayCapacityBudgets(
    const std::vector<MoEOverlayBoundPhysicalMemoryBudget>& local_budgets,
    const std::shared_ptr<IMPIContext>& mpi_context);

/**
 * @brief Gather final immutable-table requests after the original readiness vote.
 * @param frame Local sender frame validated before readiness consensus.
 * @param world Original setup context, or null for process-local planning.
 * @param profile Model source metadata authenticating received requests.
 * @return Requests in original rank order for the final admission certificate.
 * @throws std::invalid_argument If a request is malformed or has foreign identity.
 * @throws std::overflow_error If the aggregate exceeds MPI's count range.
 * @throws std::runtime_error If the original context reports a transport failure.
 */
std::vector<ImmutableRowTableAdmissionRequest> gatherOverlayImmutableTableRequests(
    const ImmutableRowTableAdmissionSendFrame& frame, const std::shared_ptr<IMPIContext>& world,
    const ModelMemoryProfile& profile);

/**
 * @brief Authenticate a common row ladder before running per-candidate collectives.
 * @param local_candidates Positive rows in descending order, or the single uncaptured marker zero.
 * @param mpi_context Original setup context, or null for local validation only.
 * @throws std::invalid_argument If the ladder is invalid or differs between ranks.
 * @throws std::overflow_error If the ladder exceeds MPI's count or allocation range.
 * @throws std::runtime_error If the original context reports a transport failure.
 */
void requireIdenticalOverlayGraphRowCandidates(const std::vector<int>& local_candidates,
    const std::shared_ptr<IMPIContext>& mpi_context);

/**
 * @brief Run a candidate phase and agree whether it can proceed or try fewer rows.
 * @param original Original setup context on which every rank enters this phase.
 * @param rows Current retained row candidate, used as the consensus ordinal.
 * @param phase Stable phase name, identical on every original rank.
 * @param diagnostic Operator-facing description of the candidate operation.
 * @param work Local phase operation. Only PhysicalMemoryCapacityExhausted permits a smaller candidate.
 * @return Local or peer capacity diagnostic when fewer rows may be tried, otherwise nullopt.
 * @throws std::invalid_argument If any rank detects a structural failure.
 * @throws std::runtime_error If phase identity or consensus transport fails.
 */
std::optional<std::string> runOriginalPipelineCapacityTrial(const IMPIContext& original,
    int rows, std::string_view phase, std::string_view diagnostic, const std::function<void()>& work);

/** @brief One actual stage's local certified contribution; borrowed only while preparing its frame. */
struct OriginalPipelineFixedContribution {
    int stage_id;
    const std::vector<MoEOverlayBoundPhysicalMemoryBudget>& budgets;
};
/** @brief Certified fixed contributions for one original semantic stage, merged once across its actual ranks. */
struct OriginalPipelineStageBudgets {
    int stage_id;
    std::vector<MoEOverlayBoundPhysicalMemoryBudget> budgets;
};
/** @brief Prevalidated immutable sender bytes, prepared before original-parent readiness succeeds. */
class OriginalPipelineCapacitySendFrame final {
public:
    /**
     * @brief Seal one CAP4 contribution per actual local stage before the readiness vote.
     * @param topology Complete original pipeline topology.
     * @param sender World rank contributing all its local stage records.
     * @param local Certified fixed contributions for exactly the sender's original stages.
     * @return Validated frame owning its exact MPI metadata bytes.
     * @throws std::invalid_argument If a stage or allocator is missing, duplicated or foreign.
     * @throws std::overflow_error If an encoded record exceeds the wire extent.
     */
    static OriginalPipelineCapacitySendFrame prepare(const GlobalPPTopology& topology, int sender,
        std::span<const OriginalPipelineFixedContribution> local);
    /** @return Exact live metadata bytes, never vector capacity. */
    std::span<const std::uint8_t> bytes() const noexcept {return bytes_;}
    /** @return Exact original authenticated sender. */
    int senderRank() const noexcept {return sender_;}
private:
    /**
     * @brief Retain an already validated frame.
     * @param bytes Exact encoded metadata.
     * @param sender Original world rank authenticated during preparation.
     */
    OriginalPipelineCapacitySendFrame(std::vector<std::uint8_t> bytes,int sender):bytes_(std::move(bytes)),sender_(sender){}
    std::vector<std::uint8_t> bytes_;
    int sender_;
};
/**
 * @brief Decode one original sender's contribution in canonical stage order.
 * @param bytes Complete stage-framed CAP4 message, with no trailing bytes.
 * @param sender Original rank expected on the frame and every CAP4 record.
 * @param topology Complete original topology authenticating the sender's stage set.
 * @return One entry per original stage, with empty budgets for stages absent on the sender.
 * @throws std::invalid_argument If framing, membership, allocator identity or charges are invalid.
 * @throws std::overflow_error If a decoded charge cannot be represented.
 */
std::vector<OriginalPipelineStageBudgets> parseOriginalPipelineCapacityFrame(
    std::span<const std::uint8_t> bytes, int sender, const GlobalPPTopology& topology);

/**
 * @brief Exchange all actual stage contributions once over the original parent.
 * @param frame Validated local sender frame.
 * @param original Same original parent context used to authenticate readiness.
 * @param topology Complete original pipeline topology shared by the parent.
 * @return Budgets grouped by original stage, with each stage's budgets in original rank order.
 * @throws std::invalid_argument If the context, frame or contribution is inconsistent with the topology.
 * @throws std::overflow_error If metadata exceeds MPI's count range.
 * @throws std::runtime_error If the original context reports a transport failure.
 */
std::vector<OriginalPipelineStageBudgets> gatherOriginalPipelineCapacityBudgets(
    const OriginalPipelineCapacitySendFrame& frame, const std::shared_ptr<IMPIContext>& original,
    const GlobalPPTopology& topology);
}
