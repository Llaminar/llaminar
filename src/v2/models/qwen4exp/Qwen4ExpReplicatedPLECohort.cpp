/**
 * @file Qwen4ExpReplicatedPLECohort.cpp
 * @brief Bind source heads to actual Rank collective coordinates and canonical query/history producers.
 *
 * Validation reads immutable setup metadata only. The existing Rank dispatch,
 * cache reset/publication and deterministic samplers own all live progression.
 * No value equality protocol, duplicated count or implicit transfer is introduced.
 */
#include "Qwen4ExpReplicatedPLECohort.h"
#include "execution/local_execution/graph/IGraphBuilder.h"
#include "execution/qwen4exp/Qwen4ExpAuxiliaryState.h"
#include "interfaces/IModelContext.h"
#include "execution/mtp/DeviceGenerationContract.h"
#include "loaders/ResidentImmutableRowTable.h"
#include <algorithm>
#include <stdexcept>
namespace llaminar2 {
Qwen4ExpReplicatedPLECohort::Qwen4ExpReplicatedPLECohort(ILocalTPContext &context,
    const IModelContext &model,std::shared_ptr<const ResolvedImmutableRowTablePlacement> placement,
    const Qwen4ExpConfig &source)
    :context_(&context),model_(&model),placement_(std::move(placement)),members_(context.devices())
{
    if(!placement_)throw std::invalid_argument("Fixed-head PLE has no resolved declaration");
    requireDeclaration(context,*placement_,source);
    const auto &n=source.ngrams();
    std::copy(n.multipliers.begin(),n.multipliers.end(),policy_.multipliers.begin());
    std::copy(n.head_offsets.begin(),n.head_offsets.end(),policy_.offsets.begin());
    std::copy(n.head_vocab_sizes.begin(),n.head_vocab_sizes.end(),policy_.moduli.begin());
    policy_.eosToken=n.hash_reset_token;policy_.validate();
}
void Qwen4ExpReplicatedPLECohort::requireDeclaration(ILocalTPContext &context,
    const ResolvedImmutableRowTablePlacement &placement,const Qwen4ExpConfig &source)
{
    const auto &members=context.devices();
    if(!placement.isFixedHeadShards() || !context.isLocal() ||
        (members.size()!=2 && members.size()!=4 && members.size()!=8) ||
        context.degree()!=static_cast<int>(members.size()) || placement.consumers().size()!=members.size() ||
        !context.supportsRawAllgatherOnStreamGraphCapture())
        throw std::invalid_argument("Fixed-head PLE requires its exact Rank-owned native local TP cohort");
    const auto &n=source.ngrams();
    if(n.table.dimensions.size()!=2 || n.heads!=16 || n.head_offsets.size()!=16 ||
        n.head_vocab_sizes.size()!=16 || n.multipliers.size()!=3 ||
        placement.extent()!=ImmutableRowTableExtent::fromNative(n.table.type,n.table.dimensions[0],n.row_bytes))
        throw std::invalid_argument("Fixed-head PLE cohort source differs from original authenticated ngram metadata");
    qwen4exp::NgramHashPolicy policy;
    std::copy(n.multipliers.begin(),n.multipliers.end(),policy.multipliers.begin());
    std::copy(n.head_offsets.begin(),n.head_offsets.end(),policy.offsets.begin());
    std::copy(n.head_vocab_sizes.begin(),n.head_vocab_sizes.end(),policy.moduli.begin());
    policy.eosToken=n.hash_reset_token;policy.validate();
    const auto firstDevice=members.front().toLocalDeviceId();
    const auto backend=firstDevice.is_cuda()?CollectiveBackendType::NCCL:CollectiveBackendType::RCCL;
    if(!firstDevice.is_gpu() || context.backend()!=backend)
        throw std::invalid_argument("Fixed-head PLE needs an actual homogeneous native GPU communicator");
    for(std::size_t p=0;p<members.size();++p)
    {
        const auto device=members[p].toLocalDeviceId();
        if(device.type!=firstDevice.type)
            throw std::invalid_argument("Fixed-head PLE cannot reinterpret heterogeneous devices as one native cohort");
        const auto assignment=Qwen4ExpFixedHeadShards::assignmentFor(placement,{placement.worldRank(),device});
        const auto first=p?policy.offsets[p*16/members.size()]:0;
        const auto end=p+1==members.size()?placement.extent().rows():policy.offsets[(p+1)*16/members.size()];
        if(assignment.participant!=p || assignment.first_head!=p*16/members.size() ||
            assignment.head_count!=16/members.size() || assignment.source.firstRow()!=first ||
            assignment.source.rows()!=end-first)
            throw std::invalid_argument("Fixed-head PLE source/head coordinate disagrees with the actual collective rank");
    }
}

ILocalTPContext &Qwen4ExpReplicatedPLECohort::context() const
{
    if(!context_ || context_->devices()!=members_ || context_->degree()!=degree())
        throw std::logic_error("Fixed-head PLE communicator changed after Rank composition");
    return *context_;
}
Qwen4ExpFixedHeadShardAssignment Qwen4ExpReplicatedPLECohort::requireParticipant(DeviceId device,int participant) const
{
    const auto &ctx=context();
    const auto backend=device.is_cuda()?CollectiveBackendType::NCCL:CollectiveBackendType::RCCL;
    if(!device.is_gpu() || participant<0 || participant>=degree() || ctx.backend()!=backend ||
        members_[participant].toLocalDeviceId()!=device)
        throw std::invalid_argument("Fixed-head PLE has a foreign native participant/collective coordinate");
    for(const auto &member:members_)
        if(member.toLocalDeviceId().type!=device.type)
            throw std::invalid_argument("Fixed-head PLE cannot reinterpret heterogeneous devices as one native cohort");
    const auto assignment=Qwen4ExpFixedHeadShards::assignmentFor(*placement_,{placement_->worldRank(),device});
    if(assignment.participant!=static_cast<std::size_t>(participant) ||
        assignment.first_head!=static_cast<std::size_t>(participant)*16/members_.size() ||
        assignment.head_count!=16/members_.size())
        throw std::invalid_argument("Fixed-head PLE source/head coordinate disagrees with the actual collective rank");
    return assignment;
}
void Qwen4ExpReplicatedPLECohort::requireQuery(const Qwen4ExpReplicatedPLEQuery &q,
    const ForwardInput &input,DeviceRowRange rows,const Qwen4ExpAuxiliaryState &history) const
{
    (void)requireParticipant(q.device_,q.participant_);
    if(q.isWorkspaceDeclaration()!=(input.graph_submission_intent==ForwardGraphSubmissionIntent::WorkspaceDeclaration))
        throw std::invalid_argument("Fixed-head executable query and workspace declaration authority cannot be interchanged");
    if(q.cohort_!=this || !q.owner_ || q.device_!=input.device || input.batch_size!=1 ||
        q.tokens_!=input.token_ids_device || !q.tokens_ || q.cache_!=input.getKVCacheForDevice(input.device) ||
        q.cache_!=&history.cache() || history.device()!=input.device ||
        q.role_!=input.execution_role || q.phase_!=input.execution_phase ||
        q.rows_.physicalRows()!=input.seq_len || q.rows_.captureIdentity()!=rows.captureIdentity())
        throw std::invalid_argument("Fixed-head PLE query lost its actual DGO token/count/cache producer");
    // Resolving the canonical lexical endpoint proves this participant owns
    // the two accepted tokens. Verifier readers use the existing authenticated
    // immutable prior capability; no second history state is retained here.
    (void)history.ple(0);
}
void Qwen4ExpReplicatedPLECohort::requireNativeOwner(
    const std::shared_ptr<const ResidentImmutableRowTable> &table) const
{
    if(!table || !model_ || model_->residentImmutableRowTable(table->device())!=table)
        throw std::invalid_argument("Fixed-head PLE requires its original shared model source owner");
}
void Qwen4ExpReplicatedPLECohort::requireSameGenerationAdmission(
    const DeviceGenerationAdmissionRequest &original,const DeviceGenerationAdmissionRequest &participant)
{
    if(!original.valid() || !participant.valid() || original.depth_policy.isForwardOnly() ||
        participant.depth_policy.isForwardOnly() || original!=participant)
        throw std::invalid_argument("Fixed-head PLE requires the complete original mirrored generation admission");
}
} // namespace llaminar2
