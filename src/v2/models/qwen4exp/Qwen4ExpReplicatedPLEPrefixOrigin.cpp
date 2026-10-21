/**
 * @file Qwen4ExpReplicatedPLEPrefixOrigin.cpp
 * @brief Validate complete process-local rollback provenance before mutation.
 *
 * Source handles retain their original pool/event/PMA lifetime. An immutable
 * metadata copy extends no allocation beyond this aggregate's own lifetime and
 * does not inspect bytes. No origin contains itself: Rank assigns the origin
 * only after this constructor captures an origin-free outer declaration.
 */
#include "Qwen4ExpReplicatedPLEPrefixOrigin.h"
#include "Qwen4ExpReplicatedPLECohort.h"
#include <algorithm>
#include <stdexcept>
namespace llaminar2 {
Qwen4ExpReplicatedPLEPrefixOrigin::Qwen4ExpReplicatedPLEPrefixOrigin(
    const RankOrchestrator *rank,std::shared_ptr<const Qwen4ExpReplicatedPLECohort> cohort,
    std::vector<const IInferenceRunner *> owners,const PrefixStateSnapshot &snapshot,int sequence)
    :rank_(rank),cohort_(std::move(cohort)),owners_(std::move(owners)),declaration_(snapshot),sequence_(sequence)
{
    if(!rank_ || !cohort_ || sequence!=0 || !snapshot.valid || snapshot.fixed_head_rollback_origin ||
        owners_.size()!=static_cast<std::size_t>(cohort_->degree()) || owners_.size()!=snapshot.participant_snapshots.size() ||
        std::ranges::any_of(owners_,[](const auto *p){return !p;}))
        throw std::invalid_argument("Fixed-head rollback requires its complete actual origin-free Rank capture");
    for(const auto &child:snapshot.participant_snapshots)
        if(!child.valid || !child.participant_snapshots.empty() || child.fixed_head_rollback_origin ||
            child.cached_tokens!=snapshot.cached_tokens)
            throw std::invalid_argument("Fixed-head rollback requires one actual DGO capture at the common frontier");
}
void Qwen4ExpReplicatedPLEPrefixOrigin::requireDeclaration(
    const PrefixStateSnapshot &producer,const PrefixStateSnapshot &observed)
{
    // These are metadata/handle copies only, not byte copies. Clearing only
    // the outer origin permits complete default equality to cover future
    // snapshot members without maintaining another field inventory. Children
    // keep their exact complete declaration and any nested origin identities.
    auto expected=producer;auto actual=observed;
    expected.fixed_head_rollback_origin.reset();actual.fixed_head_rollback_origin.reset();
    if(!expected.valid || !actual.valid || expected!=actual)
        throw std::invalid_argument("Fixed-head rollback snapshot changed its complete original child declaration");
}
void Qwen4ExpReplicatedPLEPrefixOrigin::require(const RankOrchestrator *rank,
    const Qwen4ExpReplicatedPLECohort &cohort,std::span<const IInferenceRunner *const> owners,
    const PrefixStateSnapshot &snapshot,int sequence) const
{
    if(rank!=rank_ || &cohort!=cohort_.get() || sequence!=sequence_ || snapshot.fixed_head_rollback_origin.get()!=this ||
        !std::ranges::equal(owners_,owners))
        throw std::invalid_argument("Fixed-head rollback has a foreign Rank/cohort/sequence/child order");
    requireDeclaration(declaration_,snapshot);
}
} // namespace llaminar2
