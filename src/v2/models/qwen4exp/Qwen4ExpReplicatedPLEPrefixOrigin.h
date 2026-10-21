/**
 * @file Qwen4ExpReplicatedPLEPrefixOrigin.h
 * @brief Immutable original Rank rollback declaration for a fixed-head cohort.
 *
 * This process-local origin is issued only after actual child captures finish.
 * It retains their existing immutable handles and metadata, never checkpoint
 * payload copies or a live progress mirror. Portable prefix lookup/import keeps
 * its existing key/payload-plan authority and does not consume this declaration.
 */
#pragma once
#include "execution/prefix_cache/PrefixStateSnapshot.h"
#include <span>
namespace llaminar2 {
class RankOrchestrator;class IInferenceRunner;class Qwen4ExpReplicatedPLECohort;
/** @brief One immutable aggregate origin; no publication, allocation or byte accounting API. */
class Qwen4ExpReplicatedPLEPrefixOrigin final {
public:
    /** @brief Compare every child declaration before any child may mutate state.
     * @param producer Original immutable snapshot metadata and retained handles.
     * @param observed Supplied rollback snapshot.
     * @throws std::invalid_argument For any missing/replaced/reordered/changed child declaration.
     * @note This read-only validator returns no origin or query capability.
     */
    static void requireDeclaration(const PrefixStateSnapshot &producer,const PrefixStateSnapshot &observed);
    /** @brief Authenticate actual Rank/cohort/order/sequence and the complete aggregate.
     * @param rank Existing original Rank owner, never a portable identity.
     * @param cohort Original Rank-issued producer composition.
     * @param owners Existing original child runner order.
     * @param snapshot Supplied aggregate; all metadata is checked before restore begins.
     * @param sequence Original rollback sequence slot.
     * @throws std::invalid_argument For foreign or altered rollback provenance.
     */
    void require(const RankOrchestrator *rank,const Qwen4ExpReplicatedPLECohort &cohort,
        std::span<const IInferenceRunner *const> owners,const PrefixStateSnapshot &snapshot,int sequence) const;
private:
    friend class RankOrchestrator;
    /** @brief Retain only actual successful capture declarations, before assigning the outer origin field. */
    Qwen4ExpReplicatedPLEPrefixOrigin(const RankOrchestrator *rank,
        std::shared_ptr<const Qwen4ExpReplicatedPLECohort> cohort,
        std::vector<const IInferenceRunner *> owners,const PrefixStateSnapshot &snapshot,int sequence);
    const RankOrchestrator *rank_;
    std::shared_ptr<const Qwen4ExpReplicatedPLECohort> cohort_;
    const std::vector<const IInferenceRunner *> owners_;
    const PrefixStateSnapshot declaration_; ///< Handle/metadata copy only; no payload data or independent mutation authority.
    const int sequence_;
};
} // namespace llaminar2
