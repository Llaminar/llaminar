/**
 * @file MoECPUExpertShardRouteStage.h
 * @brief Participant-local CPU expert down-partial publication through original rooted TP.
 *
 * Each socket computes its own prepared fragment and contributes only live
 * unweighted records. The original root gathers once and combines fragments
 * in immutable source-cohort order. Original router weighting/return remains a
 * later graph edge; this stage never executes another participant's compute.
 */
#pragma once
#include "../IComputeStage.h"
#include "../../../collective/IGlobalTPContext.h"
#include "../../moe/CPUExpertShardRouteCompletion.h"
#include "../../moe/ExpertHistogramSource.h"
#include "../../moe/MoEOverlayParticipantResidency.h"
#include <optional>
#include <memory>
namespace llaminar2 {
class CPUExpertShardSourceSelection;
struct MoEOverlaySparseRows;
struct MoEOverlayReturnRows;
class MoEOverlayActivationEpochProtocol;
class MoEOverlayCanonicalRouteReturnTicketStorage;
/** @brief Explicit collective edge between prepared shard compute and original route return. */
class MoECPUExpertShardRouteStage final : public IComputeStage {
public:
    /** @brief Immutable bindings; all mutable spans borrow existing graph/PMA arena roots. */
    struct Params {
        std::shared_ptr<const CPUExpertShardSourceSelection> original_source;
        IGlobalTPContext *tp=nullptr;
        TensorBase *packed_records=nullptr;
        TensorBase *route_indices=nullptr;
        TensorBase *route_weights=nullptr;
        TensorBase *combined_routes=nullptr; ///< Root-only output; non-root needs no duplicate bank.
        std::span<RootedFloatRecordRange> source_ranges; ///< Root-only admitted provenance scratch.
        std::span<int32_t> participant_route_maps;
        std::span<float> required_route_weights;
        int rows=0,top_k=0,columns=0,root_participant=0;
        std::string stage_name;
        MoEOverlayReturnRows *original_return_rows=nullptr; ///< Original endpoint return descriptor, no new packet authority.
        float *original_preweighted_routes=nullptr; ///< Root-only original mapped canonical return bank.
        size_t original_return_route_capacity=0;
        const MoEOverlaySparseRows *original_sparse_routes=nullptr; ///< Original consumed sparse packet; excludes dense router tensors.
        std::shared_ptr<const MoEOverlayParticipantResidencyRegistry::CPUJointLayer> original_prepared_layer; ///< Closed original model source publication, mandatory for channel completion.
        std::shared_ptr<IGlobalTPContext> original_tp_owner; ///< Required retained native owner for channel-bound completion.
        std::shared_ptr<MoEOverlayCanonicalRouteReturnTicketStorage> original_canonical_ticket; ///< Sole original local ticket publication authority after the explicit fragment gather.
        DeviceId original_continuation_device=DeviceId::invalid(); ///< Exact enclosing GPU for the immutable local ticket, never inferred from the CPU member.
        /** Explicit mathematical phase for standalone dense-route execution; sparse execution uses its original authenticated key. */
        std::optional<ExpertHistogramSource> standalone_execution_phase;
    };
    /**
     * @brief Bind one original prepared fragment and its exact NODE_LOCAL TP domain.
     * @param params Source lease, original context and admitted per-participant buffers.
     * @throws std::invalid_argument For foreign membership, phase geometry or missing bindings.
     */
    explicit MoECPUExpertShardRouteStage(Params params);
    /** @param ctx Original CPU executor or exact ticket-bound GPU parent scheduling context.
     * @return True only after original live gather and complete root reduction; failure aborts TP. */
    bool execute(IDeviceContext *ctx) override;
    /**
     * @brief Execute the same collective under one exact original sparse activation packet.
     * @param ctx Original CPU execution context.
     * @param protocol Exact CPU follower endpoint whose dispatch was consumed.
     * @param identity Original active immutable invocation.
     * @param stage_ordinal Original operation in that channel's declared layer manifest.
     * @return Closed completed witness, absent on failure; no ordinary publish can fabricate it.
     */
    std::optional<CPUExpertShardRouteCompletion> executeOriginalJointPublication(
        IDeviceContext *ctx,MoEOverlayActivationEpochProtocol &protocol,
        const MoEOverlayActivationEpochIdentity &identity,std::uint32_t stage_ordinal);

    ComputeStageType type() const override {return ComputeStageType::ROOTED_COLLECTIVE;}
    std::string name() const override {return "moe_cpu_expert_shard_route_gather";}
    size_t estimatedFlops() const override {return 0;}
    bool supportsBackend(ComputeBackendType backend) const override {return backend==ComputeBackendType::CPU;}
    /**
     * @brief Match the scheduling identity of this participant-local CPU host service.
     * @param scheduling_device Existing executor context's device identity.
     * @return True for CPU scheduling, or the exact original ticket-bound continuation GPU.
     * @note This does not authenticate native readiness or grant CPU execution
     * authority. For ticket-bound service, execute() calls it only after the
     * original live native CPU borrower, source, ticket and sparse epoch have
     * been authenticated.
     */
    bool acceptsSchedulingContext(DeviceId scheduling_device) const noexcept;
    bool isGraphCapturable() const override {return false;}
    /**
     * @brief Admit padded prefill only as the original ticketed live-prefix host service.
     * @return True only for this exact original canonical ticket binding.
     * @note The parent retains the concurrent manual boundary. The CPU rooted
     * collective is not a native captured stage: it gathers only the original
     * sparse packet's live records and publishes that same compact prefix.
     */
    bool supportsPaddedPrefillGraphCapturePreflight() const override {
        return hasOriginalCanonicalTicket();
    }
    /**
     * @brief Preserve actual sparse-route work independently of bucket storage capacity.
     * @return True only when the same original ticket owns live-prefix publication.
     */
    bool supportsPaddedPrefillRealLengthContract() const override {
        return hasOriginalCanonicalTicket();
    }
    /** @return Whether this explicit collective is the declared local CPU-to-GPU ticket boundary. */
    bool isManualGraphBoundary() const override {return hasOriginalCanonicalTicket();}
    /** @return The same original retained-parent concurrent service policy as its producer. */
    ManualGraphBoundaryScheduling manualGraphBoundaryScheduling() const noexcept override {
        return hasOriginalCanonicalTicket() ? ManualGraphBoundaryScheduling::ConcurrentTicketService
            : ManualGraphBoundaryScheduling::BetweenExecutableLaunches;
    }
    /** @return Only this explicit collective owns the deferred local device-ingress terminal. */
    ConcurrentManualFailureRole concurrentManualFailureRole() const noexcept override {
        return hasOriginalCanonicalTicket() ? ConcurrentManualFailureRole::DeviceIngressPublisher
            : ConcurrentManualFailureRole::None;
    }
    /** @brief Publish the original ticket abort before cancelling the actual CPU domain. @return True if the consumer is released. */
    bool publishConcurrentManualFailure() noexcept override;
    /** @return Existing borrowed tensor requirements; no private allocation or capacity ledger. */
    StageBufferRequirements getBufferRequirements() const override;
private:
    /** @return Whether the original immutable ticket binding still names this exact layer, shape and GPU. */
    bool hasOriginalCanonicalTicket() const noexcept;
    /** @return Original immutable tensor/geometry diagnostics, without probing runtime state. */
    StageDumpInfo buildDumpInfoImpl() const override;
    Params params_;
};
} // namespace llaminar2
