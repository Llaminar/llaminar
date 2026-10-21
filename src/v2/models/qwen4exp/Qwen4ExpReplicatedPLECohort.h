/**
 * @file Qwen4ExpReplicatedPLECohort.h
 * @brief Rank-issued fixed-head source/communicator binding, without a live state owner.
 *
 * Rank already owns common request/reset/admission and deterministic mirrored
 * generation. This binding authenticates that exact producer composition and
 * source-head coordinate before explicit graph collectives are constructed.
 * It never stores token/count/history values, advances a request counter,
 * allocates payload storage, selects placement or repairs divergent outcomes.
 */
#pragma once
#include "Qwen4ExpReplicatedPLEQuery.h"
#include "Qwen4ExpFixedHeadShards.h"
#include "Qwen4ExpConfig.h"
#include "collective/ILocalTPContext.h"
#include "kernels/qwen4exp/Qwen4ExpKernels.h"
#include <memory>
namespace llaminar2 {
class RankOrchestrator;class IModelContext;class ResidentImmutableRowTable;struct DeviceGenerationAdmissionRequest;struct ForwardInput;class Qwen4ExpAuxiliaryState;
/** @brief One immutable source/cohort relation privately issued by its actual Rank owner. */
class Qwen4ExpReplicatedPLECohort final {
public:
    /** @brief Check original source/head coordinates against an actual collective declaration.
     * @param context Existing communicator, never constructed or modified here.
     * @param placement Sole resolved source/range metadata.
     * @param source Original checked model directory metadata.
     * @throws std::invalid_argument For unsupported membership, source or order.
     * @note Returns no cohort, query capability or runtime readiness. Only the
     * actual Rank constructor can issue a binding after this same validation.
     */
    static void requireDeclaration(ILocalTPContext &context,
        const ResolvedImmutableRowTablePlacement &placement,const Qwen4ExpConfig &source);
    /** @brief Compare complete immutable Rank-issued generation admissions without device reads.
     * @param original Existing canonical child admission.
     * @param participant Existing participant admission at the successful materialization boundary.
     * @throws std::invalid_argument For invalid, unequal or forward-only records.
     * @note Returns no generation/query authority. Forward-only graphs retain
     * their request producer and have no sampling generation parent to compare.
     */
    static void requireSameGenerationAdmission(const DeviceGenerationAdmissionRequest &original,
        const DeviceGenerationAdmissionRequest &participant);
    /** @return Existing communicator; its Rank owner retains it through graph retirement. */
    ILocalTPContext &context() const;
    /** @return Existing canonical resolved source/range declaration, never a copied range map. */
    const ResolvedImmutableRowTablePlacement &placement() const noexcept {return *placement_;}
    /** @return Complete original hash metadata; every participant hashes the same sixteen heads. */
    const qwen4exp::NgramHashPolicy &policy() const noexcept {return policy_;}
    /** @return Actual cohort degree2/4/8. */
    int degree() const noexcept {return static_cast<int>(members_.size());}
    /** @brief Authenticate exact source coordinate against the actual communicator coordinate.
     * @param device Original participant device.
     * @param participant Actual GraphConfig/collective index, never physical ordinal.
     * @return Original unchanged range/head assignment.
     * @throws std::exception Before graph mutation for foreign/reordered membership or source.
     */
    Qwen4ExpFixedHeadShardAssignment requireParticipant(DeviceId device,int participant) const;
    /** @brief Require the actual DGO query, canonical cache/history owner and complete original rows.
     * @param query DGO-issued immutable local projection.
     * @param input Complete actual graph construction invocation.
     * @param rows Original primitive row descriptor, including native counted M1.
     * @param history Cache-owned accepted-state authority; verifier prior still uses its existing lease.
     * @throws std::exception For foreign banks/cache, roles, counts or source metadata.
     */
    void requireQuery(const Qwen4ExpReplicatedPLEQuery &query,const ForwardInput &input,
        DeviceRowRange rows,const Qwen4ExpAuxiliaryState &history) const;
    /** @brief Require the exact Ready range owner published by the original shared model context.
     * @param table Original per-participant admitted owner, never an equal-shaped foreign source.
     * @throws std::invalid_argument If source owner identity differs before any graph mutation.
     */
    void requireNativeOwner(const std::shared_ptr<const ResidentImmutableRowTable> &table) const;
    /** @return Actual shared model context identity, without model payload inspection. */
    const IModelContext *modelOwner() const noexcept {return model_;}
private:
    friend class RankOrchestrator;
    /** @brief Bind only from the actual Rank's original model and owned native TP context. */
    Qwen4ExpReplicatedPLECohort(ILocalTPContext &context,const IModelContext &model,
        std::shared_ptr<const ResolvedImmutableRowTablePlacement> placement,const Qwen4ExpConfig &source);
    ILocalTPContext *context_;const IModelContext *model_;
    std::shared_ptr<const ResolvedImmutableRowTablePlacement> placement_;
    std::vector<GlobalDeviceAddress> members_;qwen4exp::NgramHashPolicy policy_;
};
} // namespace llaminar2
