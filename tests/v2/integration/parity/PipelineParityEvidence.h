/**
 * @file PipelineParityEvidence.h
 * @brief Completed independent-comparator evidence for original split pipeline owners.
 *
 * These are diagnostic outputs, not inference state, tensor snapshots or a
 * communication scheduler. Original admitted stage metadata names ownership;
 * exact local comparator witnesses are combined only after forward completion.
 * Every decoder and global boundary needs exactly one authenticated owner.
 */
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <stdexcept>
#include <vector>
#include <type_traits>

namespace llaminar2::test::parity
{
/** @brief Original canonical forward phase, shared by local and distributed evidence. */
enum class ParityForwardPhase { Prefill, Decode };
/** @brief Passive original stage scope; last_layer remains exclusive. */
struct PipelineParityStageOwnership
{
    int stage=-1, rank=-1, first_layer=-1, last_layer=-1;
    bool embedding=false, logits=false;
};
/** @brief Fixed-size original claim and actual diagnostic record cardinality per decoder. */
struct PipelineParityLayerObservation
{
    std::uint32_t claims=0, records=0, valid_records=0;
};
/** @brief One actually claimed or observed decoder and its global index on the diagnostic wire. */
struct PipelineParityLayerRecord
{
    std::uint32_t layer=0;
    PipelineParityLayerObservation observation;
};
static_assert(std::is_trivially_copyable_v<PipelineParityLayerRecord> &&
    sizeof(PipelineParityLayerRecord)==4*sizeof(std::uint32_t));
/**
 * @brief Serialize only actual claims/observations, including invalid foreign evidence.
 * @param layers Original local global-indexed diagnostic projection.
 * @return Ascending indexed records; zero tuples for unowned/unobserved layers are absent.
 * Incorrect nonzero observations remain on the wire so the unchanged merger rejects them.
 */
inline std::vector<PipelineParityLayerRecord> compactPipelineParityLayers(
    std::span<const PipelineParityLayerObservation> layers)
{
    if(layers.size()>std::size_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Pipeline diagnostic layer index exceeds original geometry");
    std::vector<PipelineParityLayerRecord> result;
    for(std::size_t layer=0;layer<layers.size();++layer)
    {
        const auto& observed=layers[layer];
        if(observed.claims || observed.records || observed.valid_records)
            result.push_back({static_cast<std::uint32_t>(layer),observed});
    }
    return result;
}
/**
 * @brief Reconstruct rank-major diagnostic checks locally without communicating empty capacity.
 * @param records Concatenated original variable-length indexed rank payloads.
 * @param rank_counts Actual record counts in original communicator order, including empty ranks.
 * @param layer_count Authenticated global decoder count.
 * @return Local zero-initialized rank-major projection for the unchanged exact ownership merger.
 * @throws std::invalid_argument For duplicate/out-of-order/foreign indices, empty wire tuples,
 * overflow or mismatched recorded extents. Invalid nonzero evidence is retained for merger rejection.
 */
inline std::vector<PipelineParityLayerObservation> expandPipelineParityLayers(
    std::span<const PipelineParityLayerRecord> records,std::span<const std::size_t> rank_counts,
    int layer_count)
{
    if(layer_count<=0 || rank_counts.size()<2 ||
       rank_counts.size()>std::numeric_limits<std::size_t>::max()/std::size_t(layer_count))
        throw std::invalid_argument("Pipeline diagnostic compact geometry is invalid");
    std::vector<PipelineParityLayerObservation> result(rank_counts.size()*std::size_t(layer_count));
    std::size_t offset=0;
    for(std::size_t rank=0;rank<rank_counts.size();++rank)
    {
        const auto count=rank_counts[rank];
        if(count>std::size_t(layer_count) || count>records.size()-offset)
            throw std::invalid_argument("Pipeline diagnostic rank extent exceeds actual indexed records");
        std::uint32_t previous=0;
        for(std::size_t index=0;index<count;++index)
        {
            const auto& record=records[offset+index];
            const auto& observed=record.observation;
            if(record.layer>=std::uint32_t(layer_count) || (index && record.layer<=previous) ||
               !(observed.claims || observed.records || observed.valid_records))
                throw std::invalid_argument("Pipeline diagnostic record has foreign, duplicate or empty metadata");
            result[rank*std::size_t(layer_count)+record.layer]=observed;
            previous=record.layer;
        }
        offset+=count;
    }
    if(offset!=records.size())
        throw std::invalid_argument("Pipeline diagnostic payload has unclaimed trailing records");
    return result;
}
/** @brief One completed original rank's fixed-size comparison metadata; no model values. */
struct PipelineParityObservation
{
    int rank=-1, world=0, layers=0, phase=-1, step=-2;
    std::uint32_t scope_valid=0, local_raw_valid=0;
    std::uint32_t embedding_claims=0, logits_claims=0;
    std::uint32_t embedding_valid=0, has_logits=0, logits_valid=0;
    float embedding_cosine=0, cosine=0, kl=0, top1=0, top5=0;
    int embedding_passed=0, logit_passed=0, reference_in_top3=0;
    int actual_token=-1, reference_token=-1;
    int token_match=0, top3_match=0, top5_match=0;
};
/**
 * @brief Successful completed union; construction is closed to the actual merger.
 * No absence is a valid certificate. Metrics are copied from the unique
 * original head/tail, never averaged across owners or synthesized from counters.
 */
class PipelineParityEvidence final
{
public:
    /** @return Whether this certificate belongs to the exact completed phase/step/geometry. */
    bool matches(ParityForwardPhase phase,int step,int layers) const noexcept
    { return phase_==static_cast<int>(phase) && step_==step && layers_==layers && head_>=0 && tail_>=0; }
    /** @return Original embedding-owning rank, independent of MPI order. */
    int headRank() const noexcept { return head_; }
    /** @return Original LM-head-owning rank, independent of MPI order. */
    int tailRank() const noexcept { return tail_; }
    /** @return Actual original embedding comparison metadata. */
    const PipelineParityObservation& head() const noexcept { return head_observation_; }
    /** @return Actual original final-row comparison metadata. */
    const PipelineParityObservation& tail() const noexcept { return tail_observation_; }
private:
    friend PipelineParityEvidence mergePipelineParityEvidence(std::span<const PipelineParityObservation>,
        std::span<const PipelineParityLayerObservation>,int,ParityForwardPhase,int);
    PipelineParityEvidence(int phase,int step,int layers,int head,int tail,
        PipelineParityObservation head_observation,PipelineParityObservation tail_observation)
        :phase_(phase),step_(step),layers_(layers),head_(head),tail_(tail),
         head_observation_(head_observation),tail_observation_(tail_observation) {}
    int phase_,step_,layers_,head_,tail_;
    PipelineParityObservation head_observation_,tail_observation_;
};
/**
 * @brief Require exact original case/source/reference/request identities across participants.
 * @param identities Original literal declarations in actual communicator order.
 * @throws std::invalid_argument For absent or disagreeing original declarations.
 * No digest, downloaded snapshot or reconstructed model is involved.
 */
inline void requirePipelineParityIdentities(std::span<const std::string> identities)
{
    if(identities.size()<2 || identities[0].empty())
        throw std::invalid_argument("Pipeline comparison identity is incomplete");
    for(const auto& identity:identities)
        if(identity!=identities[0])throw std::invalid_argument("Pipeline original case/source/reference/request identity disagrees");
}
/**
 * @brief Validate the completed union against exact original claimed ownership.
 * @param observations One original header per actual rank, in communicator order.
 * @param layers Rank-major actual decoder claims/records, exactly world*layer_count.
 * @param layer_count Original global main-model decoder count.
 * @param phase Actual compared phase, never a rank-selected phase.
 * @param step Original decode index; prefill uses minus one.
 * @return Immutable complete certificate with untouched original head/tail metrics.
 * @throws std::invalid_argument For missing/foreign/duplicate ownership or evidence,
 * mismatched phase/world/step/geometry, nonfinite original metrics or invalid raw witnesses.
 */
inline PipelineParityEvidence mergePipelineParityEvidence(
    std::span<const PipelineParityObservation> observations,
    std::span<const PipelineParityLayerObservation> layers,int layer_count,
    ParityForwardPhase phase,int step)
{
    const auto world=observations.size();
    if((phase!=ParityForwardPhase::Prefill && phase!=ParityForwardPhase::Decode) ||
       (phase==ParityForwardPhase::Prefill ? step!=-1 : step<0) || world<2 || world>std::size_t(std::numeric_limits<int>::max()) || layer_count<=0 || step< -1 ||
       world>std::numeric_limits<std::size_t>::max()/std::size_t(layer_count) ||
       layers.size()!=world*std::size_t(layer_count))
        throw std::invalid_argument("Pipeline parity has incomplete original world/layer geometry");
    int head=-1,tail=-1;
    for(std::size_t rank=0;rank<world;++rank)
    {
        const auto& observed=observations[rank];
        if(observed.rank!=int(rank) || observed.world!=int(world) || observed.layers!=layer_count ||
           observed.phase!=static_cast<int>(phase) || observed.step!=step ||
           observed.scope_valid!=1 || observed.local_raw_valid!=1 ||
           observed.embedding_claims>1 || observed.logits_claims>1)
            throw std::invalid_argument("Pipeline parity source scope, raw evidence or phase identity is invalid");
        if(observed.embedding_claims)
        {
            if(head>=0)throw std::invalid_argument("Pipeline parity has duplicate original embedding ownership");
            head=int(rank);
        }
        if(observed.logits_claims)
        {
            if(tail>=0)throw std::invalid_argument("Pipeline parity has duplicate original LM-head ownership");
            tail=int(rank);
        }
        if((observed.embedding_valid && !observed.embedding_claims) ||
           ((observed.has_logits || observed.logits_valid) && !observed.logits_claims))
            throw std::invalid_argument("Pipeline parity contains a foreign global-boundary observation");
    }
    if(head<0 || tail<0 || (step<0 && observations[head].embedding_valid!=1) ||
       observations[tail].has_logits!=1 || observations[tail].logits_valid!=1 ||
       !std::isfinite(observations[tail].cosine) || !std::isfinite(observations[tail].kl) ||
       observations[tail].kl<0 || !std::isfinite(observations[tail].top1) ||
       !std::isfinite(observations[tail].top5) ||
       (step<0 && !std::isfinite(observations[head].embedding_cosine)))
        throw std::invalid_argument("Pipeline parity lacks finite original embedding/final-logit operands");
    if(layers[std::size_t(head)*layer_count].claims!=1 ||
       layers[std::size_t(tail)*layer_count+std::size_t(layer_count-1)].claims!=1)
        throw std::invalid_argument("Pipeline global boundary conflicts with original decoder ownership");
    for(int layer=0;layer<layer_count;++layer)
    {
        std::size_t claims=0,records=0;
        for(std::size_t rank=0;rank<world;++rank)
        {
            const auto& observed=layers[rank*std::size_t(layer_count)+std::size_t(layer)];
            if(observed.claims>1 || observed.records>1 || observed.valid_records>1 ||
               observed.records!=observed.claims || observed.valid_records!=observed.records)
                throw std::invalid_argument("Pipeline parity decoder has missing, foreign, duplicate or invalid evidence");
            claims+=observed.claims;
            records+=observed.records;
        }
        if(claims!=1 || records!=1)
            throw std::invalid_argument("Pipeline parity decoder does not have exactly one original owner and record");
    }
    return PipelineParityEvidence(static_cast<int>(phase),step,layer_count,head,tail,
        observations[head],observations[tail]);
}
}
