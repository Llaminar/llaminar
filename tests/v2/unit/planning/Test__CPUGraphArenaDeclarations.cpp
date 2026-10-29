/**
 * @file Test__CPUGraphArenaDeclarations.cpp
 * @brief Device-free production-schema replay of CPU graph arena admission.
 *
 * Actual model graph schemas/resolvers are compared root by root with metadata
 * admission. Local TP degree comes from a real configured mock participant
 * domain; retained MTP policy remains independent of whether execution is on.
 * Small native arenas prove complete declaration/materialization equality.
 * No-capacity MTP replay uses production planner semantics, including fixed depth.
 */
#include "planning/ActivationMemoryEstimator.h"
#include "config/TensorParallelConfig.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "memory/GraphArenaStorageLayout.h"
#include "memory/CPUTensorStorageDeclaration.h"
#include "memory/BufferArena.h"
#include "models/qwen35moe/Qwen35MoEGraph.h"
#include "models/qwen4exp/Qwen4ExpGraph.h"
#include "models/qwen4exp/Qwen4ExpGraphConfigBuilder.h"
#include "execution/local_execution/graph/GraphResolver.h"
#include "../../mocks/MockLocalTPContext.h"
#include "../models/qwen4exp/Qwen4ExpMetadataFixture.h"
#include "../../utils/Qwen4ExpPlacementFixture.h"
#include <gtest/gtest.h>
#include <map>
using namespace llaminar2;
namespace
{
const DeviceId cpu1{DeviceType::CPU,1};
/** @brief Install exact local domain degree without changing runtime graph semantics. */
void configureTP(GraphConfig &config,test::MockLocalTPContext &tp,int degree,int first_cpu_ordinal=1)
{
    std::vector<GlobalDeviceAddress> participants;
    std::vector<DeviceId> allocators;
    for(int i=0;i<degree;++i)
    {
        participants.push_back(GlobalDeviceAddress::cpu(i+first_cpu_ordinal));
        allocators.push_back(DeviceId{DeviceType::CPU,i+first_cpu_ordinal});
    }
    tp.setDevices(std::move(participants));tp.setBackend(CollectiveBackendType::HOST);
    config.tp_ctx=&tp;config.tp_device_idx=0;config.default_device=DeviceId{DeviceType::CPU,first_cpu_ordinal};
    config.dense_tp_enabled=true;config.qkv_column_parallel=degree>1;
    config.ffn_column_parallel=degree>1;config.lm_head_column_parallel=degree>1;
    // A LocalTP context supplies the collective degree, while the resolved
    // TensorParallelConfig authenticates dense and replicated MTP ownership.
    // Build both from one assignment and consume its actual intervals: the
    // splitter also owns FFN alignment and the selected KV replication policy.
    config.tp_config=std::make_shared<TensorParallelConfig>(TensorParallelConfig::equalSplit(
        degree,config.n_heads,config.n_kv_heads,config.d_ff,config.vocab_size,allocators));
    config.local_rank=0;
    const auto &local=config.tp_config->forRank(config.local_rank);
    config.head_start=local.head_start;config.local_n_heads=local.head_count;
    config.local_n_kv_heads=local.kv_head_count;
    config.d_ff_local=local.d_ff_count;config.vocab_local=local.vocab_count;
    config.max_seq_len=16;config.max_activation_rows=3;config.max_request_count=1;
}
/** @brief Distinguish no capacity, retained-disabled and enabled execution. */
void configureMTP(GraphConfig &config,int mode)
{
    config.mtp.enabled=mode==2;
    config.mtp.draft_tokens=3;
    config.mtp.graph_capacity_draft_tokens=mode?3:0;
}
/** @return Small but complete source metadata for a real hybrid MoE graph. */
GraphConfig tinyQwen35()
{
    GraphConfig config;
    config.n_layers=4;config.total_n_layers=4;config.d_model=16;
    config.n_heads=16;config.n_kv_heads=8;config.head_dim=2;
    // The production splitter requires one 32-column FFN tile per participant.
    // TP8 is therefore declared with 256 source columns, still a tiny arena.
    config.d_ff=256;config.vocab_size=32;
    config.gdn.group_count=8;config.gdn.time_step_rank=16;
    config.gdn.state_size=2;config.gdn.inner_size=32;config.gdn.conv_kernel_size=4;
    config.gdn.full_attention_interval=2;
    config.moe.num_experts=2;config.moe.top_k=1;config.moe.intermediate_size=8;
    config.moe.has_shared_expert=true;config.moe.shared_intermediate_size=8;
    return config;
}
/** @return Source facts only; no root names, shapes or byte formulas are copied. */
ModelMemoryProfile sourceProfile(const GraphConfig &config)
{
    ModelMemoryProfile profile;
    profile.architecture="qwen35moe";profile.n_layers=config.total_n_layers;
    profile.d_model=config.d_model;profile.d_ff=config.d_ff;
    profile.n_heads=config.n_heads;profile.n_kv_heads=config.n_kv_heads;
    profile.head_dim=config.head_dim;profile.vocab_size=config.vocab_size;
    profile.max_seq_len=config.max_seq_len;
    profile.gdn_group_count=config.gdn.group_count;
    profile.gdn_time_step_rank=config.gdn.time_step_rank;
    profile.gdn_state_size=config.gdn.state_size;profile.gdn_inner_size=config.gdn.inner_size;
    profile.gdn_conv_kernel_size=config.gdn.conv_kernel_size;
    profile.full_attention_interval=config.gdn.full_attention_interval;
    profile.expert_count=config.moe.num_experts;profile.expert_used_count=config.moe.top_k;
    profile.expert_feed_forward_length=config.moe.intermediate_size;
    profile.expert_shared_feed_forward_length=config.moe.shared_intermediate_size;
    profile.tensors.push_back({.name="blk.0.ssm_out.weight",.layer_index=0});
    return profile;
}
/** @return Exact admitted runtime geometry, including retained-disabled capacity. */
ActivationGraphMemoryGeometry runtimeGeometry(const GraphConfig &config,int degree)
{
    if(!config.tp_config || !config.tp_ctx || config.tp_config->worldSize()!=degree || config.tp_ctx->degree()!=degree)
        throw std::invalid_argument("CPU graph fixture requires one consistent resolved TP assignment and local domain");
    ActivationGraphMemoryGeometry geometry;
    geometry.batch_size=config.max_request_count;geometry.resident_graph_rows=config.max_activation_rows;
    geometry.context_tokens=config.max_seq_len;
    geometry.local_d_ff=config.dense_tp_decode_replicated?config.d_ff:config.d_ff_local;
    geometry.local_n_heads=config.dense_tp_decode_replicated?config.n_heads:config.local_n_heads;
    geometry.local_n_kv_heads=config.dense_tp_decode_replicated?config.n_kv_heads:config.local_n_kv_heads;
    geometry.local_vocab=config.dense_tp_decode_replicated?config.vocab_size:config.vocab_local;
    geometry.replicated_dense_decode=config.dense_tp_decode_replicated;
    geometry.first_layer=config.pp_layer_offset;
    geometry.last_layer=config.pp_layer_offset+config.n_layers-1;
    geometry.total_shards=degree;geometry.routed_compute_policy=config.moe.routed_compute_policy;
    // Match RankMemoryPlanInputs' production projection, rather than copying
    // the runtime resolver's choice and accidentally authenticating its drift.
    geometry.mtp_target_query_rows=retainsMTPGraphCapacity(config.mtp)
        ?resolveMTPRetainedTargetQueryRows(config.mtp):1;
    geometry.retained_mtp_candidate_rows=retainsMTPGraphCapacity(config.mtp)?resolveMTPRetainedTargetQueryRows(config.mtp):0;
    geometry.mtp_terminal_logits_layout=config.mtpTerminalLogitsLayout();
    geometry.generation_request_capacity=std::max(1,config.mtp.max_request_batch);
    geometry.attention_arithmetic=config.attention_arithmetic;
    geometry.attention_phase_schedule=config.attention_phase_schedule;
    geometry.mtp_shifted_kv_head_layout=resolveMTPShiftedKVHeadLayout(config.mtp,config.dense_tp_enabled,config.tp_config->worldSize());
    if(config.ngram_table_placement)
        geometry.ngram_lookup=config.ngram_table_placement->lookupForConsumer(
            {config.ngram_table_placement->worldRank(),config.default_device});
    return geometry;
}
/** @return Actual graph schema inventory with the same explicit orchestration roots. */
template<typename Graph>
GraphArenaStorageLayout actualLayout(const Graph &graph,const GraphConfig &config)
{
    const bool retained=retainsMTPGraphCapacity(config.mtp);
    return GraphArenaStorageLayout::resolve(graph.getSchema(),graph.getResolverConfig(config.max_activation_rows),{
        .full_prefill_kv_handoff=config.dense_tp_decode_replicated,
        .terminal_hidden_rows=retained?size_t(std::max(1,resolveMTPMaxTargetQueryRows(config.mtp))):0,
        .terminal_hidden_columns=retained?size_t(config.residualWidth()):0});
}
/** @brief Compare full root identities and physical extents, not just an aggregate sum. */
void compare(const GraphArenaStorageLayout &actual,const GraphArenaStorageLayout &metadata)
{
    std::map<BufferId,const BufferDescriptor *> expected,observed;
    for(const auto &entry:actual.entries()) ASSERT_TRUE(expected.emplace(entry.id,&entry.descriptor).second);
    for(const auto &entry:metadata.entries()) ASSERT_TRUE(observed.emplace(entry.id,&entry.descriptor).second);
    EXPECT_EQ(observed.size(),expected.size());
    for(const auto &[id,descriptor]:expected)
    {
        SCOPED_TRACE(std::string(bufferIdName(id))+" native root "+descriptor->name);
        const auto found=observed.find(id);ASSERT_NE(found,observed.end());
        const auto &candidate=*found->second;
        EXPECT_EQ(candidate.name,descriptor->name);
        EXPECT_EQ(candidate.device,descriptor->device);
        EXPECT_EQ(candidate.tensor_type,descriptor->tensor_type);
        EXPECT_EQ(candidate.shape,descriptor->shape)<<"root="<<descriptor->name;
        const auto wanted=CPUTensorStorageDeclaration::fromDescriptor(*descriptor,PhysicalMemoryOwner::ActivationArena);
        const auto got=CPUTensorStorageDeclaration::fromDescriptor(candidate,PhysicalMemoryOwner::ActivationArena);
        EXPECT_EQ(got,wanted);EXPECT_EQ(got.physicalBytes(),wanted.physicalBytes());
    }
    EXPECT_EQ(metadata.cpuPhysicalBytes(),actual.cpuPhysicalBytes())<<"complete native arena root inventory";
}
/** @return Exact bounded owner admission for the native materialization replay. */
std::shared_ptr<PhysicalMemoryAuthority> admitted(size_t bytes)
{
    PhysicalMemoryPlanBuilder builder;
    builder.add(PhysicalMemoryResource{.world_rank=0,.device=cpu1,
        .total_bytes=bytes,.admission_available_bytes=bytes},PhysicalMemoryOwner::ActivationArena,bytes);
    return std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()),0);
}
}
TEST(CPUGraphArenaDeclarations,Qwen35ActualSchemaMatchesMetadataAtEveryLocalDegreeAndMTPPolicy)
{
    for(int degree:{1,2,4,8})for(int mode:{0,1,2})
    {
        SCOPED_TRACE(::testing::Message()<<"degree="<<degree<<" mode="<<mode);
        test::MockLocalTPContext tp;auto config=tinyQwen35();
        configureTP(config,tp,degree);configureMTP(config,mode);
        ASSERT_EQ(config.tp_config->worldSize(),tp.degree());
        EXPECT_EQ(config.mtpUsesReplicatedDenseSidecarBinding(),mode!=0 && degree>1);
        Qwen35MoEGraph graph(config);
        const auto metadata=ActivationMemoryEstimator::declareCPUStorage(sourceProfile(config),runtimeGeometry(config,degree),cpu1);
        compare(actualLayout(graph,config),metadata);
    }
}
TEST(CPUGraphArenaDeclarations,QwenFlashNextActualSchemasMatchMetadataAcrossTPMTPAndPPPlacements)
{
    const auto directory=test::qwen4MetadataFixture(true);
    const auto profile=ModelMemoryProfile::fromGGUF(directory);
    // PLE+GDN only, sparse only and GDN only retain distinct extension inventories.
    for(int first:{-1,0,1,3})for(int degree:{1,2,4,8})for(int mode:{0,1,2})
    {
        SCOPED_TRACE(::testing::Message()<<"first="<<first<<" degree="<<degree<<" mode="<<mode);
        GraphConfig config;config.pp_layer_offset=std::max(0,first);config.n_layers=first<0?0:1;
        config.default_device=cpu1;
        Qwen4ExpGraphConfigBuilder::populateFromDirectory(directory,config);
        test::MockLocalTPContext tp;configureTP(config,tp,degree);configureMTP(config,mode);
        ASSERT_EQ(config.tp_config->worldSize(),tp.degree());
        EXPECT_EQ(config.mtpUsesReplicatedDenseSidecarBinding(),mode!=0 && degree>1);
        test::admitQwen4ExpTestTable(config);
        Qwen4ExpGraph graph(config);
        const auto metadata=ActivationMemoryEstimator::declareCPUStorage(profile,runtimeGeometry(config,degree),cpu1);
        compare(actualLayout(graph,config),metadata);
    }
}
TEST(CPUGraphArenaDeclarations,SmallActualSchemaMaterializesExactlyAndRejectsShortOwnerBudget)
{
    test::MockLocalTPContext tp;auto config=tinyQwen35();configureTP(config,tp,2);configureMTP(config,1);
    Qwen35MoEGraph graph(config);const auto layout=actualLayout(graph,config);
    const auto bytes=layout.cpuPhysicalBytes();ASSERT_GT(bytes,0u);ASSERT_LT(bytes,size_t{4}*1024*1024);
    auto memory=admitted(bytes);
    {
        BufferArena arena({.physical_memory_authority=memory});
        for(const auto &entry:layout.entries()) ASSERT_TRUE(arena.registerBuffer(entry.id,entry.descriptor));
        ASSERT_TRUE(arena.allocate());
        EXPECT_EQ(memory->claimedBytes(cpu1,PhysicalMemoryOwner::ActivationArena,
            PhysicalMemoryMaterializationKind::NewAllocation),bytes);
    }
    EXPECT_EQ(memory->claimedBytes(cpu1,PhysicalMemoryOwner::ActivationArena,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
    auto shortMemory=admitted(bytes-1);
    {
        BufferArena arena({.physical_memory_authority=shortMemory});
        for(const auto &entry:layout.entries()) ASSERT_TRUE(arena.registerBuffer(entry.id,entry.descriptor));
        EXPECT_THROW(arena.allocate(),std::exception);
    }
    EXPECT_EQ(shortMemory->claimedBytes(cpu1,PhysicalMemoryOwner::ActivationArena,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
}
TEST(CPUGraphArenaDeclarations,UnknownAndDuplicatePhysicalIdentitiesFailBeforeMaterialization)
{
    GraphResolverConfig resolver;resolver.default_device=cpu1;
    GraphSchema unknown;
    unknown.layer_buffers.push_back({"unknown_physical_root",{"1","1"},"fp32",BufferSemantic::Scratch});
    EXPECT_THROW(GraphArenaStorageLayout::resolve(unknown,resolver),std::invalid_argument);
    GraphSchema duplicate;
    resolver.buffer_name_to_id["registered_duplicate_root"]=BufferId::HIDDEN_STATE;
    duplicate.layer_buffers.push_back({"registered_duplicate_root",{"1","1"},"fp32",BufferSemantic::Scratch});
    duplicate.model_buffers=duplicate.layer_buffers;
    EXPECT_THROW(GraphArenaStorageLayout::resolve(duplicate,resolver),std::invalid_argument);
}

/**
 * @test Native-sized CPU0 schema roots use admitted retained MTP capacity for every depth policy.
 *
 * The published-geometry directory contains metadata only. Comparing 4096-row
 * declarations retains the failing 48-layer/12-FA context geometry without
 * allocating model payloads or the multi-gigabyte activation arena.
 */
TEST(CPUGraphArenaDeclarations,NativeSizedDisabledMTPUsesAdmittedStorageRowsAcrossDepthPolicies)
{
    const auto directory=test::qwen4MetadataFixture(true);
    const auto profile=ModelMemoryProfile::fromGGUF(directory);
    for(const auto policy:{MTPDepthPolicyMode::Fixed,MTPDepthPolicyMode::Dynamic})
    for(const int degree:{1,2,4,8})
    for(const int mode:{0,1,2})
    {
        SCOPED_TRACE(::testing::Message()<<"policy="<<static_cast<int>(policy)<<" degree="<<degree<<" mode="<<mode);
        GraphConfig config;config.default_device=DeviceId::cpu();
        Qwen4ExpGraphConfigBuilder::populateFromDirectory(directory,config);
        test::MockLocalTPContext tp;configureTP(config,tp,degree,0);
        config.max_seq_len=4096;config.max_activation_rows=4096;
        configureMTP(config,mode);
        config.mtp.draft_tokens=1;
        config.mtp.depth_policy.mode=policy;
        config.mtp.depth_policy.max_depth=3;
        test::admitQwen4ExpTestTable(config);
        ASSERT_EQ(config.n_layers,48);
        Qwen4ExpGraph graph(config);
        const auto geometry=runtimeGeometry(config,degree);
        const auto resolver=graph.getResolverConfig(config.max_activation_rows);
        const size_t expected_rows=mode?4:1;
        EXPECT_EQ(geometry.mtp_target_query_rows,expected_rows);
        EXPECT_EQ(resolver.custom_formulas.at("mtp_target_query_rows"),expected_rows);
        EXPECT_EQ(resolver.custom_formulas.at("mtp_kv_prefill_rows"),4096u);
        EXPECT_EQ(geometry.retained_mtp_candidate_rows,mode?4:0);
        const auto actual=actualLayout(graph,config);
        const auto metadata=ActivationMemoryEstimator::declareCPUStorage(profile,geometry,DeviceId::cpu());
        compare(actual,metadata);
        if(degree==1 && mode==0)
            EXPECT_EQ(actual.cpuPhysicalBytes(),4491052224ULL)
                <<"exact admitted native CPU0 declaration from the focused source-admission replay";
    }
}

/**
 * @test Ordinary request-batched compact logits retain one input row per
 * request even when MTP owns only its one-row disabled placeholder.
 */
TEST(CPUGraphArenaDeclarations,OrdinaryTerminalRowsStayDistinctFromSpeculativeScratch)
{
    for(const int requests:{1,2,4})for(const int mode:{0,1})
    {
        SCOPED_TRACE(::testing::Message()<<"requests="<<requests<<" retained="<<mode);
        test::MockLocalTPContext tp;auto config=tinyQwen35();
        configureTP(config,tp,1);configureMTP(config,mode);
        config.max_request_count=requests;
        config.mtp.max_request_batch=1; // MTP-owned request capacity stays distinct.
        ASSERT_FALSE(config.mtp.enabled);
        Qwen35MoEGraph graph(config);
        auto geometry=runtimeGeometry(config,1);geometry.batch_size=requests;
        geometry.generation_request_capacity=1;
        const int flattened=requests*config.max_activation_rows; // seq>1 requires compact row gathering.
        const auto resolver=graph.getResolverConfig(flattened);
        const size_t speculative=mode?4u:1u;
        const size_t terminal=std::max(size_t(requests),speculative);
        EXPECT_EQ(resolver.custom_formulas.at("mtp_target_query_rows"),speculative);
        EXPECT_EQ(resolver.custom_formulas.at("terminal_logit_row_capacity"),terminal);
        const auto actual=GraphArenaStorageLayout::resolve(graph.getSchema(),resolver,{
            .full_prefill_kv_handoff=false,
            .terminal_hidden_rows=mode?terminal:0,
            .terminal_hidden_columns=mode?size_t(config.residualWidth()):0});
        const auto metadata=ActivationMemoryEstimator::declareCPUStorage(sourceProfile(config),geometry,cpu1);
        compare(actual,metadata);
        bool found_terminal=false,found_mtp=false;
        for(const auto &entry:actual.entries())
        {
            if(entry.id==BufferId::LM_HEAD_INPUT_ROWS)
            {found_terminal=true;EXPECT_EQ(entry.descriptor.shape.front(),terminal);}
            if(entry.id==BufferId::MTP_HIDDEN)
            {found_mtp=true;EXPECT_EQ(entry.descriptor.shape.front(),speculative);}
        }
        EXPECT_TRUE(found_terminal);EXPECT_TRUE(found_mtp);
    }
}
/** @test Flash Next's real schema inherits the same compact terminal-row contract for batched main graphs. */
TEST(CPUGraphArenaDeclarations,FlashNextOrdinaryTerminalRowsMatchActualBatchedSchema)
{
    const auto directory=test::qwen4MetadataFixture(true);
    const auto profile=ModelMemoryProfile::fromGGUF(directory);
    for(const int requests:{1,2,4})for(const int mode:{0,1})
    {
        SCOPED_TRACE(::testing::Message()<<"requests="<<requests<<" retained="<<mode);
        GraphConfig config;config.default_device=cpu1;
        Qwen4ExpGraphConfigBuilder::populateFromDirectory(directory,config);
        test::MockLocalTPContext tp;configureTP(config,tp,1);configureMTP(config,mode);
        config.max_request_count=requests;config.mtp.max_request_batch=1;
        test::admitQwen4ExpTestTable(config);Qwen4ExpGraph graph(config);
        auto geometry=runtimeGeometry(config,1);geometry.batch_size=requests;
        // Match production planner input: MTP request/seed capacity stays one,
        // independently of the ordinary graph's sparse request-bank capacity.
        geometry.generation_request_capacity=1;
        const auto resolver=graph.getResolverConfig(requests*config.max_activation_rows);
        const size_t speculative=mode?4u:1u;
        const size_t terminal=std::max(size_t(requests),speculative);
        EXPECT_EQ(resolver.custom_formulas.at("mtp_target_query_rows"),speculative);
        EXPECT_EQ(resolver.custom_formulas.at("terminal_logit_row_capacity"),terminal);
        const auto actual=GraphArenaStorageLayout::resolve(graph.getSchema(),resolver,{
            .full_prefill_kv_handoff=false,.terminal_hidden_rows=mode?terminal:0,
            .terminal_hidden_columns=mode?size_t(config.residualWidth()):0});
        compare(actual,ActivationMemoryEstimator::declareCPUStorage(profile,geometry,cpu1));
    }
}

/**
 * @test QFN sparse request banks use ordinary batch capacity independently of
 * retained predictor/seed request capacity across the actual CPU TP schemas.
 *
 * Production planner inputs keep generation_request_capacity tied to the MTP
 * policy. Comparing actual roots under independent capacities prevents a test
 * from masking underadmission by replacing that field with ordinary batch size.
 */
TEST(CPUGraphArenaDeclarations,FlashNextSparseRequestBanksKeepOrdinaryAndMTPCapacitiesIndependent)
{
    const auto directory=test::qwen4MetadataFixture(true);
    const auto profile=ModelMemoryProfile::fromGGUF(directory);
    for(const int degree:{1,2,4,8})
    for(const int requests:{1,2,4})
    for(const int mtp_requests:{1,3})
    for(const int mode:{0,1,2})
    {
        SCOPED_TRACE(::testing::Message()<<"degree="<<degree<<" requests="<<requests
            <<" mtp_requests="<<mtp_requests<<" mode="<<mode);
        GraphConfig config;config.default_device=cpu1;
        Qwen4ExpGraphConfigBuilder::populateFromDirectory(directory,config);
        test::MockLocalTPContext tp;configureTP(config,tp,degree);configureMTP(config,mode);
        config.max_request_count=requests;
        config.mtp.max_request_batch=mtp_requests;
        test::admitQwen4ExpTestTable(config);
        Qwen4ExpGraph graph(config);
        const auto geometry=runtimeGeometry(config,degree);
        EXPECT_EQ(geometry.batch_size,requests);
        EXPECT_EQ(geometry.generation_request_capacity,mtp_requests);
        const auto resolver=graph.getResolverConfig(requests*config.max_activation_rows);
        const auto terminal=std::max(requests,resolveMTPRetainedTargetQueryRows(config.mtp));
        const auto actual=GraphArenaStorageLayout::resolve(graph.getSchema(),resolver,{
            .full_prefill_kv_handoff=false,
            .terminal_hidden_rows=mode?size_t(terminal):0,
            .terminal_hidden_columns=mode?size_t(config.residualWidth()):0});
        compare(actual,ActivationMemoryEstimator::declareCPUStorage(profile,geometry,cpu1));
    }
}
