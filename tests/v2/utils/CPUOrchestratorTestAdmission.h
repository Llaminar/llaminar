/**
 * @file CPUOrchestratorTestAdmission.h
 * @brief Source-authored admission for actual CPU graph/orchestrator fixtures.
 *
 * Schema roots are priced through GraphArenaStorageLayout, including retained
 * continuation mailboxes. Inference peers consume canonical MemoryPlanner
 * owner contributions projected from the fixture's source graph configuration.
 * No live allocation count or anonymous owner allowance enters this helper.
 */
#pragma once
#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "memory/GraphArenaStorageLayout.h"
#include "planning/MemoryPlanner.h"
#include "tensors/TensorClasses.h"
#include <initializer_list>
#include <utility>
#include "../mocks/MockModelContext.h"
#include <omp.h>
namespace llaminar2::test
{
/** @brief Project actual source tensor formats and axes into the canonical planning inventory.
 * @param weights Borrowed typed fixture weights, with source owners retained by the test.
 * @param main_layers Main-layer interval in the fixture's graph.
 * @return Native metadata only; no weight values are copied or converted.
 */
inline std::vector<TensorSizeInfo> cpuFixtureTensorInventory(const ModelWeights &weights,int main_layers)
{
    std::vector<TensorSizeInfo> result;
    const auto add=[&](const std::string &name,TensorBase *tensor,int layer)
    {
        if(!tensor) return;
        const auto &shape=tensor->shape();
        const char *format=tensorTypeName(tensor->native_type());
        if(tensor->native_type()==TensorType::FP32) format="F32";
        if(tensor->native_type()==TensorType::FP16) format="F16";
        result.push_back({.name=name,.native_bytes=tensor->size_bytes(),.quant_type=format,
            .elements=tensor->numel(),.K=shape.size()<2?0:shape.size()==3?shape.front():shape.back(),.layer_index=layer});
    };
    const auto add_layer=[&](const LayerWeights &layer,int index)
    {
        const std::string prefix="blk."+std::to_string(index)+".";
        for(const auto &[name,tensor]:std::initializer_list<std::pair<const char*,TensorBase*>>{
            {"attn_q.weight",layer.wq},{"attn_k.weight",layer.wk},{"attn_v.weight",layer.wv},
            {"attn_output.weight",layer.wo},{"attn_norm.weight",layer.attn_norm},
            {"attn_q_norm.weight",layer.q_norm},{"attn_k_norm.weight",layer.k_norm},
            {"attn_qkv.weight",layer.attn_qkv},{"attn_gate.weight",layer.attn_gate},
            {"ssm_alpha.weight",layer.ssm_alpha},{"ssm_beta.weight",layer.ssm_beta},
            {"ssm_conv1d.weight",layer.ssm_conv1d},{"ssm_dt.bias",layer.ssm_dt_bias},
            {"ssm_a",layer.ssm_a},{"ssm_norm.weight",layer.ssm_norm},{"ssm_out.weight",layer.ssm_out},
            {"ffn_gate.weight",layer.gate_proj},{"ffn_up.weight",layer.up_proj},
            {"ffn_down.weight",layer.down_proj},{"post_attention_norm.weight",layer.ffn_norm},
            {"ffn_gate_inp.weight",layer.moe_gate},{"ffn_gate_exps.weight",layer.moe_gate_exps},
            {"ffn_up_exps.weight",layer.moe_up_exps},{"ffn_down_exps.weight",layer.moe_down_exps},
            {"ffn_gate_shexp.weight",layer.shared_expert_gate},{"ffn_up_shexp.weight",layer.shared_expert_up},
            {"ffn_down_shexp.weight",layer.shared_expert_down},{"ffn_gate_inp_shexp.weight",layer.shared_expert_gate_inp}})
            add(prefix+name,tensor,index);
    };
    add("token_embd.weight",weights.embedding_table,-1);
    add("output_norm.weight",weights.final_norm,-1);add("output.weight",weights.lm_head,-1);
    if(weights.get_layer_weights)
        for(int layer=0;layer<main_layers;++layer) add_layer(weights.get_layer_weights(layer),layer);
    for(size_t depth=0;depth<weights.mtp.depths.size();++depth)
    {
        // Typed MTP source bindings may use a deliberately remote/global test
        // layer id. Planning consumes their physical predictor slot, retaining
        // each original tensor's true native format and dimensions.
        const int layer=main_layers+static_cast<int>(depth);
        add_layer(weights.mtp.depths[depth].fa_block,layer);
        add("blk."+std::to_string(layer)+".nextn.eh_proj.weight",weights.mtp.depths[depth].fc,layer);
    }
    return result;
}
/** @brief Adapt frozen bindings through their existing typed model-facing projection. */
inline std::vector<TensorSizeInfo> cpuFixtureTensorInventory(const FrozenModelWeightSet &weights,int main_layers)
{
    return cpuFixtureTensorInventory(toLegacyModelWeights(makeModelWeightBindings(weights)),main_layers);
}
/** @return Actual registered graph roots for the fixture's authored maximum shape. */
inline GraphArenaStorageLayout cpuFixtureGraphRoots(const IGraphBuilder &graph,int batch,int rows)
{
    const auto &config=graph.config();
    const bool retained=retainsMTPGraphCapacity(config.mtp);
    return GraphArenaStorageLayout::resolve(graph.getSchema(),graph.getResolverConfig(rows),{
        .full_prefill_kv_handoff=config.dense_tp_enabled&&config.dense_tp_decode_replicated&&config.qkv_column_parallel,
        .terminal_hidden_rows=retained?size_t(resolveMTPTerminalHiddenRowCapacity(batch,config.mtp)):0,
        .terminal_hidden_columns=retained?size_t(config.residualWidth()):0});
}
/** @return Empty test dependency bundle with explicit graph and rank identity. */
inline DeviceGraphOrchestrator::Dependencies cpuFixtureDependencies(
    std::shared_ptr<IGraphBuilder> graph,std::shared_ptr<IMPIContext> mpi)
{
    DeviceGraphOrchestrator::Dependencies deps;
    deps.model_ctx=MockModelContext::createMinimal();deps.graph_builder=std::move(graph);deps.mpi_ctx=std::move(mpi);
    return deps;
}
/** @brief Admit only the arena owner for fixtures that call initializeBuffers.
 * @param graph Actual schema/resolver owner.
 * @param mpi Fixture rank identity.
 * @param rows Maximum initialization capacity authored by this test.
 * @return Dependencies for createForTest; no storage is materialized here.
 */
inline DeviceGraphOrchestrator::Dependencies cpuGraphBuffersFixtureDependencies(
    std::shared_ptr<IGraphBuilder> graph,std::shared_ptr<IMPIContext> mpi,int rows)
{
    auto deps=cpuFixtureDependencies(graph,mpi);
    const auto bytes=cpuFixtureGraphRoots(*graph,1,rows).cpuPhysicalBytes();
    const int rank=mpi?mpi->rank():0;
    PhysicalMemoryPlanBuilder plan;
    plan.add({rank,graph->config().default_device,bytes,bytes},PhysicalMemoryOwner::ActivationArena,bytes);
    deps.physical_memory_authority=std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),rank);
    return deps;
}
/** @brief Admit graph-state initialization from its schema and canonical state geometry.
 *
 * Weight-free fixtures initialize buffers/caches without preparing model weights.
 * An empty source inventory therefore declares no prepared weight payload;
 * inference fixtures below still require their actual native source inventory.
 * @param graph Actual model graph, including exact sharding and retained MTP policy.
 * @param mpi Fixture rank identity.
 * @param batch Maximum request slots initialized by the fixture.
 * @param context Exact cache horizon.
 * @param rows Maximum activation rows initialized by the fixture.
 * @param native_sources Optional actual weight formats for fixtures that also prepare weights.
 * @return Dependencies retaining one complete typed CPU fixture authority.
 */
inline DeviceGraphOrchestrator::Dependencies cpuGraphStateFixtureDependencies(
    std::shared_ptr<IGraphBuilder> graph,std::shared_ptr<IMPIContext> mpi,int batch,int context,int rows,
    std::vector<TensorSizeInfo> native_sources = {})
{
    auto deps=cpuFixtureDependencies(graph,mpi);
    const auto &c=graph->config();
    ModelMemoryProfile profile;
    profile.architecture=graph->architectureName();
    profile.n_layers=c.total_n_layers>0?c.total_n_layers:c.n_layers;
    profile.d_model=c.d_model;profile.d_ff=c.d_ff;profile.n_heads=c.n_heads;
    profile.n_kv_heads=c.n_kv_heads;profile.head_dim=c.head_dim;profile.vocab_size=c.vocab_size;
    profile.max_seq_len=context;profile.mtp_layer_count=retainsMTPGraphCapacity(c.mtp)?1:0;
    profile.n_layers+=profile.mtp_layer_count;
    profile.gdn_group_count=c.gdn.group_count;profile.gdn_time_step_rank=c.gdn.time_step_rank;
    profile.gdn_state_size=c.gdn.state_size;profile.gdn_inner_size=c.gdn.inner_size;
    profile.gdn_conv_kernel_size=c.gdn.conv_kernel_size;profile.full_attention_interval=c.gdn.full_attention_interval;
    profile.expert_count=c.moe.num_experts;profile.expert_used_count=c.moe.top_k;
    profile.expert_feed_forward_length=c.moe.intermediate_size;
    profile.expert_shared_feed_forward_length=c.moe.shared_intermediate_size;
    profile.tensors=std::move(native_sources);
    DevicePlanConfig config;
    config.device=graph->getResolverConfig(rows).default_device;config.world_rank=mpi?mpi->rank():0;
    config.device_total_bytes=config.device_free_bytes=size_t{1}<<40;
    config.cpu_execution=CPUExecutionGeometry::local();config.device_compute_units=omp_get_max_threads();
    config.batch_size=batch;config.max_seq_len=context;config.activation_seq_len=rows;
    config.first_layer=c.pp_layer_offset;config.last_layer=c.pp_layer_offset+c.n_layers-1;
    // Sampler/fingerprint TP metadata alone does not create a sharded cache.
    // Match DGO's actual collective-domain gate before selecting local heads.
    const bool has_cache_domain=(c.tp_ctx&&c.tp_ctx->degree()>1)||(mpi&&mpi->world_size()>1);
    config.local_kv_heads=has_cache_domain&&c.local_n_kv_heads>0&&
        !(c.dense_tp_enabled&&c.dense_tp_decode_replicated)?c.local_n_kv_heads:c.n_kv_heads;
    // AUTO is a source policy, not a physical storage token. Resolve it at
    // the same CPU storage boundary as the actual cache constructor.
    config.kv_precision=c.kv_cache_precision==KVCachePrecision::AUTO
        ? activationPrecisionToString(resolveKVCacheStoragePrecision(c.kv_cache_precision,true))
        : kvCachePrecisionToString(c.kv_cache_precision);
    config.prefix_cache=c.prefix_cache;
    config.mtp_enabled=retainsMTPGraphCapacity(c.mtp);
    config.mtp_target_query_rows=config.mtp_enabled?resolveMTPRetainedTargetQueryRows(c.mtp):1;
    config.routed_compute_policy=c.moe.routed_compute_policy;
    config.mtp_terminal_logits_layout=resolveMTPTerminalLogitsLayout(c.lm_head_column_parallel,c.mtp.terminal_head_policy);
    config.generation_request_capacity=std::max(batch,c.mtp.max_request_batch);
    config.captured_serving_graphs.mtp_graph_owners=MTPGraphOwnerPlan(c.mtp,
        MTPVerifierGraphFamily::BoundedBuckets,MTPGenerationParentRole::Controller);
    if(c.dense_tp_enabled&&c.tp_ctx&&c.tp_ctx->degree()>1)
    {
        config.total_shards=c.tp_ctx->degree();config.shard_index=c.tp_device_idx;
    }
    // Sampler/logit-gather TP metadata alone does not shard the CPU cache.
    // DGO also requires graph-declared local cache heads before applying an
    // assignment, even when MPI supplies a real vocabulary-gather collective.
    if(c.tp_config&&c.tp_config->worldSize()>1&&c.local_n_kv_heads>0&&
       ((c.tp_ctx&&c.tp_ctx->degree()>1)||(mpi&&mpi->world_size()>1)))
    {
        config.total_shards=c.tp_config->worldSize();config.shard_index=c.tp_device_idx;
        config.bindTensorParallelAssignment(c.tp_config->forRank(c.tp_device_idx));
    }
    config.mtp_shifted_kv_head_layout=resolveMTPShiftedKVHeadLayout(c.mtp,c.dense_tp_enabled,config.total_shards);
    const auto estimated=MemoryPlanner::plan(profile,{config});
    const auto &source=estimated.physicalPlan().resources().front();
    const size_t arena=cpuFixtureGraphRoots(*graph,batch,rows).cpuPhysicalBytes();
    PhysicalMemoryPlanBuilder admission;
    for(size_t i=0;i<static_cast<size_t>(PhysicalMemoryOwner::Count);++i)
    {
        const auto owner=static_cast<PhysicalMemoryOwner>(i);
        admission.add(source.resource(),owner,owner==PhysicalMemoryOwner::ActivationArena?arena:source.bytes(owner));
    }
    deps.physical_memory_authority=std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(admission.build()),config.world_rank);
    return deps;
}
/**
 * @brief Admit inference with the fixture's actual immutable weight inventory.
 * @param graph Actual model graph and exact retained policy.
 * @param mpi Fixture rank identity.
 * @param batch Maximum initialized request slots.
 * @param context Exact cache horizon.
 * @param rows Maximum initialized activation rows.
 * @param native_sources Mandatory actual source formats and K/N geometry.
 * @return Complete source-authored fixture dependencies.
 * @throws std::invalid_argument For an absent source inventory.
 */
inline DeviceGraphOrchestrator::Dependencies cpuInferenceFixtureDependencies(
    std::shared_ptr<IGraphBuilder> graph, std::shared_ptr<IMPIContext> mpi,
    int batch, int context, int rows, std::vector<TensorSizeInfo> native_sources)
{
    if (native_sources.empty())
        throw std::invalid_argument("CPU inference fixture requires its actual native weight inventory");
    return cpuGraphStateFixtureDependencies(std::move(graph), std::move(mpi),
        batch, context, rows, std::move(native_sources));
}

}
