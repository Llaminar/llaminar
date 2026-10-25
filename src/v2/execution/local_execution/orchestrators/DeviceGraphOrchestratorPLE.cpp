/**
 * @file DeviceGraphOrchestratorPLE.cpp
 * @brief Project fixed-head PLE query producers from existing DGO native owners.
 *
 * This graph-construction boundary reads only immutable addresses and roles.
 * Request banks, chunk progress, ordinary frontier and verifier counts stay in
 * their original arena/cache/controller. No request cursor, live-value read,
 * payload allocation or corrective participant transfer is introduced.
 */
#include "DeviceGraphOrchestrator.h"
#include "models/qwen4exp/Qwen4ExpReplicatedPLECohort.h"
#include "execution/compute_stages/stages/LMHeadStage.h"
#include "execution/compute_stages/stages/NativeVocabularyAllGatherStage.h"
#include "loaders/PreparedWeightStore.h"
#include <algorithm>
#include <tuple>
#include <unordered_set>
#include <stdexcept>
namespace llaminar2 {
Qwen4ExpReplicatedPLEQuery DeviceGraphOrchestrator::fixedHeadPLEQuery(const ForwardInput &input) const
{
    const auto &cfg=graph_builder_->config();const auto &cohort=cfg.ngram_replicated_ple_cohort;
    if(!cohort || !cfg.ngram_table_placement || !cfg.ngram_table_placement->isFixedHeadShards() ||
        cohort->placement()!=*cfg.ngram_table_placement || &cohort->context()!=cfg.tp_ctx ||
        cohort->modelOwner()!=injected_model_ctx_.get() || !state_.device_id.is_gpu() ||
        input.device!=state_.device_id || input.batch_size!=1 || input.seq_len<=0 ||
        !input.token_ids_device || input.getKVCacheForDevice(input.device)!=state_.kv_cache.get())
        throw std::invalid_argument("Fixed-head PLE query requires its actual Rank cohort, native DGO bank and canonical main cache");
    (void)cohort->requireParticipant(input.device,cfg.tp_device_idx);
    const auto *tokens=static_cast<const std::int32_t *>(input.token_ids_device);
    const std::int32_t *count=nullptr;Qwen4ExpPLEQuerySource source;
    if(input.graph_submission_intent==ForwardGraphSubmissionIntent::WorkspaceDeclaration)
    {
        // Only the actual DGO workspace family can issue this source arm.
        // Permanent native addresses describe capacity; no current query/history
        // contents are asserted ready and the resulting graph cannot execute.
        const int activation=state_.activation_seq_len>0?state_.activation_seq_len:state_.max_seq_len;
        bool original=false;
        if(input.seq_len<=activation && !input.device_prefill_chunk && !input.device_request_query)
        {
            if(input.execution_role==ForwardExecutionRole::MainInference)
            {
                original=tokens==request_token_ids_dev_ && input.position_ids_device==request_position_ids_dev_ &&
                    input.seq_len<=request_input_row_capacity_ &&
                    (!input.sequence_lengths_device || input.sequence_lengths_device==request_sequence_lengths_dev_);
                if(input.execution_phase==ForwardExecutionPhase::Prefill)count=input.sequence_lengths_device;
                else original=original && input.execution_phase==ForwardExecutionPhase::Decode && input.seq_len==1;
            }
            else if(input.execution_role==ForwardExecutionRole::GroupedMTPVerifier)
            {
                original=cfg.grouped_mtp_verifier && input.execution_phase==ForwardExecutionPhase::Decode &&
                    tokens==mtp_verifier_input_tokens_dev_ && input.position_ids_device==mtp_verifier_position_ids_dev_ &&
                    input.sequence_lengths_device==mtp_verifier_request_lengths_dev_ && input.seq_len<=mtp_max_verifier_rows_;
                count=input.sequence_lengths_device;
            }
            else if(input.execution_role==ForwardExecutionRole::MTPCondition)
            {
                const auto &logical=device_resident_logical_sequence_state_storage_;
                original=cfg.live_mtp_request_batch_condition && input.seq_len==1 && input.execution_phase==ForwardExecutionPhase::Decode &&
                    tokens==logical.next_condition_tokens_device && input.position_ids_device==logical.target_cached_tokens_device &&
                    input.sequence_lengths_device==logical.target_cached_tokens_device;
            }
        }
        if(!arena_ || !original)
            throw std::invalid_argument("Fixed-head workspace declaration changed its actual cohort/arena input geometry");
        return Qwen4ExpReplicatedPLEQuery(cohort.get(),this,input.device,cfg.tp_device_idx,tokens,
            count?DeviceRowRange::deviceCounted(input.seq_len,count):DeviceRowRange::fullyActive(input.seq_len),
            state_.kv_cache.get(),input.execution_role,input.execution_phase,Qwen4ExpPLEQuerySource::WorkspaceDeclaration);
    }
    if(!executableForwardSubmissionIntent(input.graph_submission_intent))
        throw std::invalid_argument("Fixed-head query has an unknown submission intent");
    if(input.execution_role==ForwardExecutionRole::MainInference && input.execution_phase==ForwardExecutionPhase::Prefill)
    {
        if(input.device_prefill_chunk)
        {
            const auto &chunk=*input.device_prefill_chunk;
            if(!chunk.valid() || chunk.request_token_ids_device!=request_token_ids_dev_ ||
                chunk.request_position_ids_device!=request_position_ids_dev_ ||
                chunk.request_total_rows_device!=request_sequence_lengths_dev_ ||
                chunk.cached_tokens_device!=state_.kv_cache->deviceSequenceCachedTokenCountPtr(0) ||
                chunk.chunk_token_ids_device!=prefill_chunk_token_ids_dev_ ||
                chunk.chunk_position_ids_device!=prefill_chunk_position_ids_dev_ ||
                chunk.chunk_real_rows_device!=prefill_chunk_real_rows_dev_ ||
                chunk.chunk_row_stride_device!=prefill_chunk_row_stride_dev_ ||
                chunk.bucket_seq_len!=input.seq_len || tokens!=prefill_chunk_token_ids_dev_ ||
                input.sequence_lengths_device!=prefill_chunk_real_rows_dev_)
                throw std::invalid_argument("Fixed-head PLE prefill chunk is not the original DGO request/progress producer");
            count=prefill_chunk_real_rows_dev_;source=Qwen4ExpPLEQuerySource::PrefillChunk;
        }
        else
        {
            if(tokens!=request_token_ids_dev_ || input.seq_len>request_input_row_capacity_ ||
                input.sequence_lengths_device!=request_sequence_lengths_dev_)
                throw std::invalid_argument("Fixed-head PLE prefill lost its original common request admission bank");
            requireDeviceRequestQueryOwnsInput(input);
            count=static_cast<const std::int32_t *>(request_sequence_lengths_dev_);source=Qwen4ExpPLEQuerySource::Request;
        }
    }
    else if(input.execution_role==ForwardExecutionRole::GroupedMTPVerifier && input.execution_phase==ForwardExecutionPhase::Decode)
    {
        if(!cfg.grouped_mtp_verifier || !cfg.usesMTPGroupedDecodeEquivalentRows() ||
            tokens!=mtp_verifier_input_tokens_dev_ || input.sequence_lengths_device!=mtp_verifier_request_lengths_dev_ ||
            input.seq_len>mtp_max_verifier_rows_)
            throw std::invalid_argument("Fixed-head PLE verifier requires its actual token-plan bank and original request count");
        count=static_cast<const std::int32_t *>(mtp_verifier_request_lengths_dev_);source=Qwen4ExpPLEQuerySource::Verifier;
    }
    else if(input.seq_len==1 && input.execution_phase==ForwardExecutionPhase::Decode &&
        input.execution_role==ForwardExecutionRole::MainInference)
    {
        if(input.sequence_lengths_device ||
            (tokens!=device_resident_logical_sequence_state_storage_.next_condition_tokens_device && tokens!=request_token_ids_dev_))
            throw std::invalid_argument("Fixed-head ordinary PLE query requires its original logical frontier or common scalar ingress");
        source=tokens==request_token_ids_dev_?Qwen4ExpPLEQuerySource::Request:Qwen4ExpPLEQuerySource::OrdinaryFrontier;
    }
    else if(input.seq_len==1 && input.execution_phase==ForwardExecutionPhase::Decode &&
        input.execution_role==ForwardExecutionRole::MTPCondition)
    {
        const auto &logical=device_resident_logical_sequence_state_storage_;
        if(!cfg.live_mtp_request_batch_condition || !logical.validFor(1) || tokens!=logical.next_condition_tokens_device ||
            input.sequence_lengths_device!=logical.target_cached_tokens_device)
            throw std::invalid_argument("Fixed-head condition PLE query requires its exact live logical mailbox");
        // This count is sequence length/position, not a token-query row count.
        // The existing scalar condition invocation owns exactly one query row.
        source=Qwen4ExpPLEQuerySource::ConditionFrontier;
    }
    else throw std::invalid_argument("Fixed-head PLE has no replicated producer for this invocation role/phase");
    return Qwen4ExpReplicatedPLEQuery(cohort.get(),this,input.device,cfg.tp_device_idx,tokens,
        count?DeviceRowRange::deviceCounted(input.seq_len,count):DeviceRowRange::fullyActive(input.seq_len),
        state_.kv_cache.get(),input.execution_role,input.execution_phase,source);
}
/**
 * @brief Authenticate the actual prepared mirrored terminal programs before Rank publishes readiness.
 *
 * Rank already distributes one immutable admission and orders every request,
 * reset and prefix operation across these children. This setup-only projection
 * checks that their successful compilers retained the same source/arithmetic
 * and sampling law, and that each publisher writes its own canonical banks.
 * It neither compares live outcomes nor stores a second publication state.
 */
void DeviceGraphOrchestrator::requireReplicatedPLEGenerationProducer(
    const DeviceGraphOrchestrator &canonical) const
{
    const auto fail=[](){throw std::invalid_argument("Fixed-head PLE requires the actual common Rank admission and complete deterministic mirrored terminal producers");};
    if(!graph_builder_ || !canonical.graph_builder_ || !forward_engine_ || !arena_ ||
        !active_device_generation_admission_ || !canonical.active_device_generation_admission_)fail();
    const auto &cfg=graph_builder_->config();const auto &other=canonical.graph_builder_->config();
    const auto &cohort=cfg.ngram_replicated_ple_cohort;
    const auto &loop=deviceGenerationLoopGraph();
    const auto &reference=canonical.deviceGenerationLoopGraph();
    if (!cohort || cohort != other.ngram_replicated_ple_cohort ||
        cohort->modelOwner() != injected_model_ctx_.get() ||
        injected_model_ctx_ != canonical.injected_model_ctx_ || cfg.tp_ctx != &cohort->context() ||
        other.tp_ctx != cfg.tp_ctx || state_.device_id.type != canonical.state_.device_id.type ||
        !state_.device_id.is_gpu() || !loop.valid || !loop.capture || !loop.stream ||
        loop.launched || loop.workspace_generation != workspaceGeneration(state_.device_id) ||
        !reference.valid || !reference.capture || reference.launched || loop.request_count != 1 ||
        loop.request_count != reference.request_count ||
        sparseQueryGraphProgram() != canonical.sparseQueryGraphProgram() ||
        loop.execution_kind != reference.execution_kind ||
        loop.sampling_mode != reference.sampling_mode ||
        loop.draft_depth != reference.draft_depth ||
        loop.verifier_rows_per_request != reference.verifier_rows_per_request ||
        loop.minimum_draft_depth != reference.minimum_draft_depth ||
        loop.maximum_draft_depth != reference.maximum_draft_depth ||
        loop.depth_policy_mode != reference.depth_policy_mode ||
        cfg.activation_precision != other.activation_precision ||
        cfg.kv_cache_precision != other.kv_cache_precision ||
        cfg.attention_arithmetic != other.attention_arithmetic ||
        cfg.attention_phase_schedule != other.attention_phase_schedule ||
        cfg.qfn_hc_up_activation != other.qfn_hc_up_activation ||
        cfg.qfn_gdn_output_activation != other.qfn_gdn_output_activation ||
        cfg.qfn_fa_output_activation != other.qfn_fa_output_activation ||
        cfg.qfn_gdn_qkv_activation != other.qfn_gdn_qkv_activation ||
        cfg.getAllreducePrecision() != other.getAllreducePrecision() ||
        mtp_request_penalty_policy_ != canonical.mtp_request_penalty_policy_ ||
        mtp_request_stop_token_count_ != canonical.mtp_request_stop_token_count_ ||
        mtp_request_stop_tokens_ != canonical.mtp_request_stop_tokens_)
        fail();
    (void)cohort->requireParticipant(state_.device_id,cfg.tp_device_idx);
    const auto &a=*active_device_generation_admission_;const auto &b=*canonical.active_device_generation_admission_;
    Qwen4ExpReplicatedPLECohort::requireSameGenerationAdmission(b,a);

    // Source fragments are existing compiler-issued executable identities.
    // Lookup validates that a candidate is actually embedded, not merely an
    // old sidecar/preparation cache left over from another request shape.
    const auto embedded=[&](const IGPUGraphCapture *capture,const ComputeGraph *graph){
        return (capture && std::ranges::any_of(loop.source_fragments,[&](const auto &f){return f.capture==capture;})) ||
            std::ranges::any_of(loop.hosted_fragments,[&](const auto &f){return (capture && f.capture==capture) || (graph && f.semantic_graph==graph);});
    };
    using Head=std::tuple<std::string,std::uint64_t,std::size_t,std::size_t,int,int,int,bool,bool,int>;
    std::vector<Head> heads;
    const auto inspect=[&](const ComputeGraph &graph,const DeviceGraphOrchestrator &owner,std::vector<Head> &out,const ITensor *selectedLogits){
        const auto &ownerCfg=owner.graph_builder_->config();
        const auto precedes=[&](const std::string &producer,const std::string &consumer){
            std::vector<std::string> pending{consumer};std::unordered_set<std::string> visited;
            while(!pending.empty()) {
                auto name=std::move(pending.back());pending.pop_back();
                if(name==producer)return true;
                if(!visited.insert(name).second)continue;
                const auto *node=graph.getNode(name);if(!node)fail();
                pending.insert(pending.end(),node->dependencies.begin(),node->dependencies.end());
            }
            return false;
        };
        for(const auto &name:graph.getExecutionOrder())
        {
            const auto *node=graph.getNode(name);if(!node || !node->stage)fail();
            if(const auto *terminal=dynamic_cast<const MTPVerifierOutcomeStage *>(node->stage.get())) {
                const auto &t=terminal->getParams();
                if(t.device_id!=owner.state_.device_id || t.mode!=MTPVerifierOutcomeGraphMode::Greedy ||
                    t.ownership_policy!=MTPVerifierOutcomeOwnershipPolicy::ParticipantLocal ||
                    !t.participant_full_vocabulary || t.logits!=selectedLogits ||
                    t.vocab_size!=owner.state_.vocab_size ||
                    t.binding.verifier_input_tokens_device!=owner.mtp_verifier_input_tokens_dev_ ||
                    t.binding.active_verifier_row_count_device!=owner.mtp_verifier_request_lengths_dev_ ||
                    t.binding.generated_token_counts_device!=owner.mtp_generated_token_counts_dev_)fail();
            }
            const auto *head=dynamic_cast<const LMHeadStage *>(node->stage.get());
            if(!head)continue;
            const auto &p=head->getParams();
            if(p.device_id!=owner.state_.device_id || !p.prepared_ref || !p.prepared_store || !p.logits ||
                p.prepared_ref->device!=owner.state_.device_id || p.d_model!=owner.state_.d_model)fail();
            const auto binding=p.prepared_store->binding(*p.prepared_ref);
            if(!binding || !binding->immutable || binding->identity.role!=WeightRole::LMHead ||
                binding->identity.canonical_name.empty())fail();
            bool full=p.vocab_size==owner.state_.vocab_size && selectedLogits && p.logits==selectedLogits;
            if(!full)
                for(const auto &gname:graph.getExecutionOrder())
                {
                    const auto *gn=graph.getNode(gname);
                    const auto *g=gn && gn->stage?dynamic_cast<const NativeVocabularyAllGatherStage *>(gn->stage.get()):nullptr;
                    if(!g)continue;
                    const auto &gp=g->params();
                    if(gp.local_logits==p.logits && gp.tp_ctx==ownerCfg.tp_ctx && gp.participant==ownerCfg.tp_device_idx &&
                        gp.local_vocabulary==p.vocab_size && gp.vocabulary==owner.state_.vocab_size &&
                        selectedLogits && gp.full_logits==selectedLogits && precedes(name,gname))
                        full=true;
                }
            if(!full)fail();
            // Every participant refers to the same original source. Local
            // vocabulary slices may differ; their existing typed gather above
            // authenticates ownership. Device addresses are checked locally,
            // never compared as if distinct replicas shared one allocation.
            out.emplace_back(binding->identity.canonical_name,binding->identity.model_id.value,
                binding->slice.source_rows,binding->slice.source_cols,p.d_model,owner.state_.vocab_size,
                p.serial_equivalent_partition_width,p.compute_all_positions,
                p.force_decode_equivalent_verifier_prefill,p.seq_len);
        }
    };
    const auto forward=[&](const ForwardGraphSignature &signature){
        if(signature.sparse_query_execution_program!=sparseQueryGraphProgram())fail();
        const auto retained=forward_engine_->retainedDecodeGraph(signature);
        const auto native=forward_engine_->deviceLoopGraphTemplate(signature);
        if(!retained || !retained->graph_owner || retained->device!=state_.device_id ||
            retained->workspace_generation!=loop.workspace_generation ||
            !embedded(native?native->capture:nullptr,retained->graph_owner))fail();
        inspect(*retained->graph_owner,*this,heads,
            signature.all_position_logits || signature.live_mtp_request_batch_condition
                ? state_.all_position_logits.get():state_.logits.get());
    };
    if(a.depth_policy.isOrdinary())
    {
        if(!loop.ordinary_sampling_identity || !loop.ordinary_forward_identity)fail();
        const auto &p=*loop.ordinary_sampling_identity;const auto &pub=p.publication;
        const auto &logical=device_resident_logical_sequence_state_storage_;
        if(p.device_id!=state_.device_id || !p.logits || !state_.logits || p.logits!=state_.logits->gpu_data_ptr() ||
            p.vocab_size!=state_.vocab_size || p.logits_row_stride!=state_.vocab_size || !pub.valid() ||
            pub.control!=device_generation_storage_.control_device || pub.response_tokens!=device_generation_storage_.response_tokens_device ||
            pub.frontier.next_condition_tokens!=logical.next_condition_tokens_device)fail();
        if(const auto *random=std::get_if<OrdinaryGenerationSamplingStage::Stochastic>(&p.policy))
        {
            const auto seeds=arena_->getSharedTensor(BufferId::SAMPLING_REQUEST_SEEDS);
            if(!a.ordinary_sampling || !a.sampling_seeds || !seeds || random->seeds!=seeds->gpu_data_ptr() ||
                random->top_k!=a.ordinary_sampling->top_k || random->top_p!=a.ordinary_sampling->top_p ||
                random->temperature!=a.ordinary_sampling->temperature)fail();
        }
        else if(!a.ordinary_sampling || !a.ordinary_sampling->is_greedy())fail();
        forward(*loop.ordinary_forward_identity);
    }
    else
    {
        const auto &publication=mtpStatePublicationCache();
        if(!publication.valid || !publication.stage || publication.workspace_generation!=loop.workspace_generation)fail();
        const auto &p=publication.stage->getParams();const auto &logical=device_resident_logical_sequence_state_storage_;
        if(p.authority!=MTPSpeculativeStatePublicationStage::Authority::BoundedGeneration || p.request_count!=1 ||
            p.generation_control_device!=device_generation_storage_.control_device ||
            p.verifier_input_tokens_device!=mtp_verifier_input_tokens_dev_ ||
            p.next_condition_tokens_device!=logical.next_condition_tokens_device ||
            p.target_cached_tokens_device!=logical.target_cached_tokens_device ||
            p.main_kv_bindings.size()!=1 || p.main_kv_bindings.front().cache!=state_.kv_cache.get())fail();
        const auto pc=publication.segment_cache.deviceLoopGraphTemplate(*publication.graph);
        if(!pc || !embedded(pc->capture,publication.graph.get()))fail();
        bool verifier=false;
        for(const auto &preparation:mtp_verifier_preparation_graphs_)
        {
            if(!preparation || !preparation->valid)continue;
            // The verifier preparation is shared, but each complete parent
            // consumes its own authenticated immutable main-forward program.
            const auto &paired=pairedVerifierForwardSignature(*preparation);
            if(!paired)continue;
            const auto native=forward_engine_->deviceLoopGraphTemplate(*paired);
            const auto retained=forward_engine_->retainedDecodeGraph(*paired);
            if(!retained || !embedded(native?native->capture:nullptr,retained->graph_owner))continue;
            const auto &signature=*paired;
            if(signature.execution_role!=ForwardExecutionRole::GroupedMTPVerifier ||
                signature.device_token_ids!=mtp_verifier_input_tokens_dev_ ||
                signature.device_sequence_lengths!=mtp_verifier_request_lengths_dev_)fail();
            forward(signature);verifier=true;
        }
        if(!verifier)fail();
    }
    if(heads.empty())fail();
    std::vector<Head> referenceHeads;
    const auto canonicalEmbedded=[&](const IGPUGraphCapture *capture,const ComputeGraph *graph){
        return (capture && std::ranges::any_of(reference.source_fragments,[&](const auto &f){return f.capture==capture;})) ||
            std::ranges::any_of(reference.hosted_fragments,[&](const auto &f){return (capture && f.capture==capture) || (graph && f.semantic_graph==graph);});
    };
    const auto referenceForward=[&](const ForwardGraphSignature &signature){
        if(signature.sparse_query_execution_program!=canonical.sparseQueryGraphProgram())fail();
        const auto retained=canonical.forward_engine_->retainedDecodeGraph(signature);
        const auto native=canonical.forward_engine_->deviceLoopGraphTemplate(signature);
        if(!retained || !retained->graph_owner || !canonicalEmbedded(native?native->capture:nullptr,retained->graph_owner))fail();
        inspect(*retained->graph_owner,canonical,referenceHeads,
            signature.all_position_logits || signature.live_mtp_request_batch_condition
                ? canonical.state_.all_position_logits.get():canonical.state_.logits.get());
    };
    if(a.depth_policy.isOrdinary())
    {
        if(!reference.ordinary_forward_identity)fail();
        referenceForward(*reference.ordinary_forward_identity);
    }
    else
        for(const auto &preparation:canonical.mtp_verifier_preparation_graphs_)
        {
            if(!preparation || !preparation->valid)continue;
            const auto &paired=canonical.pairedVerifierForwardSignature(*preparation);
            if(!paired)continue;
            const auto native=canonical.forward_engine_->deviceLoopGraphTemplate(*paired);
            const auto retained=canonical.forward_engine_->retainedDecodeGraph(*paired);
            if(retained && canonicalEmbedded(native?native->capture:nullptr,retained->graph_owner))
                referenceForward(*paired);
        }
    const auto sidecars=[&](const DeviceGraphOrchestrator &owner,std::vector<Head> &out){
        const auto &parent=owner.deviceGenerationLoopGraph();
        const auto visit=[&](const auto &cache){
            if(!cache.valid || !cache.graph || cache.workspace_generation!=parent.workspace_generation)return;
            const auto native=cache.segment_cache.deviceLoopGraphTemplate(*cache.graph);
            const bool selected=(native && std::ranges::any_of(parent.source_fragments,[&](const auto &f){return f.capture==native->capture;})) ||
                std::ranges::any_of(parent.hosted_fragments,[&](const auto &f){return f.semantic_graph==cache.graph.get();});
            if(selected) {
                const auto found=owner.state_.extension_buffers.find(owner.mtpSamplingLogitsBufferId());
                inspect(*cache.graph,owner,out,found==owner.state_.extension_buffers.end()?nullptr:found->second.get());
            }
        };
        visit(owner.mtp_sidecar_depth0_cache_);visit(owner.mtp_sidecar_depth0_device_token_cache_);
        visit(owner.mtp_sidecar_depth0_chained_cache_);visit(owner.mtp_sidecar_depth0_chained_device_token_cache_);
        visit(owner.mtp_sidecar_depth0_kv_only_cache_);visit(owner.mtp_sidecar_depth0_kv_only_device_token_cache_);
        for(const auto &cache:owner.mtp_sidecar_depth0_kv_only_batch_caches_)if(cache)visit(*cache);
    };
    if(!a.depth_policy.isOrdinary())
    {
        sidecars(*this,heads);sidecars(canonical,referenceHeads);
        if(loop.sampling_mode==DeviceGenerationSamplingMode::Stochastic)
        {
            const auto &target=mtp_stochastic_target_distribution_graph_;
            const auto &outcome=mtp_stochastic_outcome_graph_;
            const auto &peerTarget=canonical.mtp_stochastic_target_distribution_graph_;
            const auto &peerOutcome=canonical.mtp_stochastic_outcome_graph_;
            if(!target.valid || !target.stage || !target.graph || !outcome.valid || !outcome.stage || !outcome.graph ||
                !peerTarget.valid || !peerTarget.stage || !peerOutcome.valid || !peerOutcome.stage ||
                target.workspace_generation!=loop.workspace_generation || outcome.workspace_generation!=loop.workspace_generation)fail();
            const auto tc=target.segment_cache.deviceLoopGraphTemplate(*target.graph);
            const auto oc=outcome.segment_cache.deviceLoopGraphTemplate(*outcome.graph);
            if(!tc || !oc || !embedded(tc->capture,target.graph.get()) || !embedded(oc->capture,outcome.graph.get()))fail();
            const auto &t=target.stage->getParams();const auto &pt=peerTarget.stage->getParams();
            const auto &o=outcome.stage->getParams();const auto &po=peerOutcome.stage->getParams();
            const auto seeds=arena_->getSharedTensor(BufferId::SAMPLING_REQUEST_SEEDS);
            if(t.device_id!=state_.device_id || t.verifier_input_tokens_device!=mtp_verifier_input_tokens_dev_ ||
                t.active_rows_device!=mtp_verifier_request_lengths_dev_ || t.vocab_size!=state_.vocab_size ||
                !state_.all_position_logits || t.logits_device!=state_.all_position_logits->gpu_data_ptr() ||
                t.generated_token_counts_device!=mtp_generated_token_counts_dev_ ||
                std::tuple(t.top_k,t.top_p,t.temperature,t.apply_penalties,t.row_count,t.logits_row_stride)!=
                    std::tuple(pt.top_k,pt.top_p,pt.temperature,pt.apply_penalties,pt.row_count,pt.logits_row_stride) ||
                o.device_id!=state_.device_id || o.verifier_input_tokens_device!=mtp_verifier_input_tokens_dev_ ||
                o.generation_control_device!=device_generation_storage_.control_device || !seeds ||
                o.threshold_seeds_device!=seeds->gpu_data_ptr() || !a.sampling_seeds ||
                std::tuple(o.verification,o.top_k,o.request_count,o.comparison_rows_per_request,o.verifier_row_capacity,
                    o.threshold_position_offset,o.target_distribution_row_stride)!=
                    std::tuple(po.verification,po.top_k,po.request_count,po.comparison_rows_per_request,po.verifier_row_capacity,
                        po.threshold_position_offset,po.target_distribution_row_stride))fail();
        }
    }
    std::ranges::sort(heads);std::ranges::sort(referenceHeads);
    if(heads!=referenceHeads || state_.vocab_size!=canonical.state_.vocab_size || state_.d_model!=canonical.state_.d_model)fail();

}
} // namespace llaminar2
