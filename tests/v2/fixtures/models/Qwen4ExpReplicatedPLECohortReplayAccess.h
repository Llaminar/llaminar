/**
 * @file Qwen4ExpReplicatedPLECohortReplayAccess.h
 * @brief Closed checked metadata construction for actual cold rendezvous replay.
 *
 * These factories call the production validators. They grant no native source,
 * admitted PMA, Ready table, graph or runtime query capability. Test callbacks
 * observe only host cold ordering against the original mock TP context.
 */
#pragma once
#include "models/qwen4exp/Qwen4ExpReplicatedPLECohort.h"
#include "loaders/ResolvedImmutableRowTablePlacement.h"
namespace llaminar2::test {
/** @brief Device-free bridge to the same validated metadata/cohort constructors. */
struct Qwen4ExpReplicatedPLECohortReplayAccess final {
    /**
     * @brief Close an already checked shared layout without materializing any source.
     * @param layout Original typed physical owners, coordinator and compute readers.
     * @param policy Exact declared shared cache request.
     * @param rowElements Original source's required native row geometry.
     * @return Validated metadata declaration only, never allocation/runtime permission.
     * @throws std::exception From the same production constructor for inconsistent geometry/policy.
     */
    static std::shared_ptr<const ResolvedImmutableRowTablePlacement> declaration(
        ImmutableRowTableLayoutGeometry layout,AdaptiveVRAMRAMSSDRequest policy,std::uint64_t rowElements)
    {return std::shared_ptr<const ResolvedImmutableRowTablePlacement>(new ResolvedImmutableRowTablePlacement(std::move(layout),policy,rowElements));}
    /**
     * @brief Bind the actual cold-operation owner after original declaration validation.
     * @param context Original mock communicator; its existing contract owns rendezvous/abort.
     * @param model Original mock model-context identity retained throughout the replay.
     * @param placement Validated immutable declaration from the actual metadata constructor.
     * @param source Original checked source metadata, including all sixteen heads.
     * @return Actual cohort object, used only for host cold-phase method replay.
     * @throws std::exception From original declaration/communicator validation.
     */
    static std::unique_ptr<Qwen4ExpReplicatedPLECohort> cohort(ILocalTPContext& context,const IModelContext& model,
        std::shared_ptr<const ResolvedImmutableRowTablePlacement> placement,const Qwen4ExpConfig& source)
    {return std::unique_ptr<Qwen4ExpReplicatedPLECohort>(new Qwen4ExpReplicatedPLECohort(context,model,std::move(placement),source));}
};
} // namespace llaminar2::test
