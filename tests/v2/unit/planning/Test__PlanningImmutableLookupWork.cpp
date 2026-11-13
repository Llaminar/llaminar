/**
 * @file Test__PlanningImmutableLookupWork.cpp
 * @brief Device-free replay of live immutable traffic and physical source ownership.
 *
 * Metadata-only placement factories retain native byte widths and actual PP/TP
 * ownership. Dirty capacities and source gaps cannot inflate query traffic;
 * no model, GPU, source payload or measurement service is opened by this suite.
 */
#include "planning/PlanningImmutableLookupWork.h"
#include "planning/MemoryPlanner.h"
#include "planning/ImmutableRowTablePlacementPlanner.h"
#include "models/qwen4exp/Qwen4ExpFixedHeadShards.h"
#include "planning/ImmutableRowTableExecutionProvision.h"
#include "planning/PlanningModelMetadata.h"
#include "planning/OrchestrationCandidateAdmission.h"
#include "fixtures/planning/ImmutableRowTableAdmissionWireReplayAccess.h"
#include "fixtures/planning/RecordedNgramOrdering.h"
#include "loaders/ImmutableRowTable.h"
#include "../../utils/Qwen4ExpPlanningGGUFFixture.h"
#include "../../utils/PlanningHostServiceFixture.h"
#include "../../utils/CPUExecutionTestGeometry.h"
#include <gtest/gtest.h>
#include <array>
#include <algorithm>
#include <limits>

namespace llaminar2::test {
namespace {
/** @return General source fixture, deliberately distinct from the production model dimensions. */
ModelMemoryProfile profile(std::size_t heads=3,std::size_t elements=4,std::size_t row_bytes=16) {
    ModelMemoryProfile result;result.architecture="qwen4exp";result.n_layers=4;
    Qwen4ExpMemoryGeometry g;g.ple_heads=heads;g.ple_row_elements=elements;g.ple_decoder_layer=1;
    g.ple_table_rows=heads*10;g.ple_table_row_bytes=row_bytes;g.ple_table_native_bytes=g.ple_table_rows*row_bytes;
    for(std::size_t i=0;i<heads;++i){g.ple_head_offsets.push_back(i*10);g.ple_head_vocab_sizes.push_back(10);}
    result.flash_next=g;
    result.tensors.push_back({.name="per_layer_token_embd.weight",.native_bytes=g.ple_table_native_bytes,
        .quant_type="F32",.elements=g.ple_table_rows*elements,.K=elements});
    return result;
}
/** @return Original rank-owned resource; available bytes are only placement admission inputs. */
PhysicalMemoryResource resource(DeviceId device) { return {0,device,1u<<24,1u<<24}; }
/** @return Actual whole-tier placement from the canonical PMA selector. */
std::shared_ptr<const ResolvedImmutableRowTablePlacement> whole(const ModelMemoryProfile &p,
    ImmutableRowStoragePreference mode,DeviceId consumer) {
    const auto &g=*p.flash_next;
    const auto extent=ImmutableRowTableExtent::fromNative(static_cast<GGUFTensorType>(0),g.ple_table_rows,g.ple_table_row_bytes);
    const std::array consumers{resource(consumer)};
    const auto input=ImmutableRowTablePlacementInput::forConsumers(extent,resource(DeviceId::cpu()),consumers);
    PhysicalMemoryPlanBuilder base;base.add(consumers[0],PhysicalMemoryOwner::PrimaryModelWeights,1);
    return ImmutableRowTablePlacementPlanner::select({mode,65536,1},input,base.build(),{.vram=PhysicalMemoryPlan{}}).placement;
}
/** @return One actual consumer with an authenticated final source; no graph row capacity becomes work. */
DevicePlanConfig device(DeviceId id,std::shared_ptr<const ResolvedImmutableRowTablePlacement> placement) {
    DevicePlanConfig result;result.world_rank=0;result.device=id;result.first_layer=0;result.last_layer=3;
    result.batch_size=1;result.activation_seq_len=512;result.ngram_table_placement=std::move(placement);
    return result;
}
/** @return Exact source-bound metadata consumer sharing the existing recorded ordering/service fixture. */
DevicePlanConfig sourceConsumer(DeviceType backend,int ordinal,const ImmutableRowTablePolicy &policy) {
    DevicePlanConfig c;c.world_rank=0;c.device={backend,ordinal};c.device_compute_units=64;
    c.device_total_bytes=512ull<<30;c.device_free_bytes=500ull<<30;
    c.associated_host_memory=PhysicalMemoryResource{0,DeviceId::cpu(),512ull<<30,500ull<<30};
    c.max_seq_len=64;c.activation_seq_len=32;c.first_layer=0;c.last_layer=47;
    c.ngram_execution_policy=policy;c.native_ordering_events=recordedNgramOrdering(c.device);
    declareRecordedNgramHostServices(c);return c;
}
/** @return Production immutable request; no replacement geometry or source authority is minted. */
ImmutableRowTableAdmissionRequest sourceRequest(const ModelMemoryProfile &profile,
    std::vector<DevicePlanConfig> &configs,const ImmutableRowTablePolicy &policy) {
    std::vector<DeviceId> endpoints;for(const auto &c:configs)endpoints.push_back(c.device);
    const auto ordering=NativeOrderingEventGroupReplayAccess::projection(endpoints);
    for(auto &c:configs)c.native_ordering_events=ordering;
    shareRecordedNgramHostServices(configs);
    const std::vector<std::vector<DevicePlanConfig>> groups{configs};
    return MemoryPlanner::immutableRowTableAdmissionRequest(profile,policy,groups,{},
        ImmutableRowTableHostPlacement::taskPolicy()).value();
}

}
/** @test Original IQ4_NL160 row width90 remains independent of whole-table row count. */
TEST(PlanningImmutableLookupWork, LiveVolumeReplayRowsFormatsPartialAndEmpty) {
    for(const std::size_t rows:{0u,1u,2u,7u,64u,512u})
        for(const std::size_t heads:{1u,3u,16u}) {
            const auto volume=PlanningImmutableLookupVolume::forTokenHeads(rows,2,heads,90,160);
            EXPECT_EQ(volume.queries(),rows*2*heads);EXPECT_EQ(volume.nativeBytes(),rows*2*heads*90);
            EXPECT_EQ(volume.decodedBytes(),rows*2*heads*160*sizeof(float));
        }
    EXPECT_THROW(PlanningImmutableLookupVolume::forTokenHeads(1,0,1,90,160),std::invalid_argument);
    EXPECT_THROW(PlanningImmutableLookupVolume::forTokenHeads(std::numeric_limits<std::size_t>::max(),2,1,90,160),std::overflow_error);
}
/** @test CPU/CUDA/ROCm source and actual discovery ownership survive large-to-small replay. */
TEST(PlanningImmutableLookupWork, WholeRAMAndVRAMUseLiveQueriesOnActualEndpoints) {
    const std::array discovery{7};
    for(const auto id:{DeviceId::cpu(),DeviceId::cuda(2),DeviceId::rocm(9)}) {
        const auto p=profile();
        const std::array devices{device(id,whole(p,ImmutableRowStoragePreference::RAM,id))};
        for(const std::size_t rows:{512u,1u,0u,7u}) {
            const auto work=compilePlanningImmutableLookupWork(p,devices,discovery,rows);
            EXPECT_EQ(work.live_queries,rows*3);EXPECT_FALSE(work.requires_storage_read_service);
            if(!rows){EXPECT_TRUE(work.memory.empty());EXPECT_TRUE(work.channels.empty());continue;}
            ASSERT_EQ(work.memory.size(),2u);EXPECT_EQ(work.memory[1].read_bytes,rows*3*16);
            EXPECT_EQ(work.memory[1].endpoint.discovery_rank,7);EXPECT_TRUE(work.memory[1].endpoint.device.is_cpu());
            EXPECT_EQ(work.channels.size(),id.is_cpu()?0u:1u);
        }
        if(id.is_gpu()) {
            const std::array resident{device(id,whole(p,ImmutableRowStoragePreference::VRAM,id))};
            const auto work=compilePlanningImmutableLookupWork(p,resident,discovery,2);
            EXPECT_TRUE(work.channels.empty());EXPECT_EQ(work.memory[1].read_bytes,2u*3*16);
        }
    }
}
/** @test Native source gaps and unequal stored sizes never become lookup communication capacity. */
TEST(PlanningImmutableLookupWork, FixedHeadRangesKeepGapsOutOfLiveReads) {
    auto p=profile(16);auto &g=*p.flash_next;g.ple_head_offsets.clear();g.ple_head_vocab_sizes.assign(16,5);
    for(std::size_t i=0;i<16;++i)g.ple_head_offsets.push_back(3+8*i);
    g.ple_table_rows=130;g.ple_table_native_bytes=130*g.ple_table_row_bytes;
    const auto extent=ImmutableRowTableExtent::fromNative(static_cast<GGUFTensorType>(0),g.ple_table_rows,g.ple_table_row_bytes);
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm}) {
        const std::array owners{PhysicalMemoryAllocatorIdentity{0,{backend,1}},PhysicalMemoryAllocatorIdentity{0,{backend,7}}};
        auto placement=Qwen4ExpFixedHeadShards::declare(extent,g.ple_head_offsets,g.ple_head_vocab_sizes,owners);
        const std::array devices{device(owners[0].device,placement),device(owners[1].device,placement)};
        const std::array discovery{4};const auto work=compilePlanningImmutableLookupWork(p,devices,discovery,3);
        std::size_t native_reads=0;for(const auto &traffic:work.memory)native_reads+=traffic.read_bytes;
        EXPECT_EQ(native_reads,3u*16*g.ple_table_row_bytes);ASSERT_EQ(work.channels.size(),1u);
        EXPECT_EQ(work.channels[0].response_bytes,3u*16*g.ple_table_row_bytes);
        EXPECT_EQ(work.channels[0].native_group,(std::vector<DeviceId>{owners[0].device,owners[1].device}));
        EXPECT_EQ(work.channels[0].owner_live_bytes,(std::vector<size_t>{3u*8*g.ple_table_row_bytes,3u*8*g.ple_table_row_bytes}));
        EXPECT_EQ(work.channels[0].request_bytes,0u);
        EXPECT_EQ(work.channels[0].header_bytes_per_owner,8u);
        EXPECT_EQ(work.channels[0].gather_protocol,PlanningImmutableGatherProtocol::EqualDeviceRows);
    }
}
/** @test Empty work stays empty and repeated partial gathers retain exact native owner fragments. */
TEST(PlanningImmutableLookupWork, NativeGatherFragmentsReplayPartialAndEmptyRows) {
    const auto p=profile(16);const auto &g=*p.flash_next;
    const auto extent=ImmutableRowTableExtent::fromNative(static_cast<GGUFTensorType>(0),
        g.ple_table_rows,g.ple_table_row_bytes);
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm}) {
        const std::array owners{PhysicalMemoryAllocatorIdentity{0,{backend,2}},PhysicalMemoryAllocatorIdentity{0,{backend,9}}};
        const auto placement=Qwen4ExpFixedHeadShards::declare(extent,g.ple_head_offsets,g.ple_head_vocab_sizes,owners);
        auto first=device(owners[0].device,placement),second=device(owners[1].device,placement);
        first.batch_size=second.batch_size=2;
        const std::array configs{first,second};const std::array discovery{7};
        for(const size_t rows:{9u,0u,1u,9u,1u}) {
            const auto work=compilePlanningImmutableLookupWork(p,configs,discovery,rows);
            if(!rows)EXPECT_TRUE(work.memory.empty());
            ASSERT_EQ(work.channels.size(),1u);const auto &gather=work.channels.front();
            EXPECT_EQ(gather.header_bytes_per_owner,8u);
            EXPECT_EQ(gather.gather_protocol,PlanningImmutableGatherProtocol::EqualDeviceRows);
            EXPECT_EQ(gather.kind,PlanningImmutableChannelKind::NativeGather);
            EXPECT_EQ(gather.owner_live_bytes,(std::vector<size_t>{rows*2*8*g.ple_table_row_bytes,rows*2*8*g.ple_table_row_bytes}));
            EXPECT_EQ(gather.response_bytes,rows*2*16*g.ple_table_row_bytes);
        }
    }
}
/** @test Unsupported cold SSD latency is explicit, and missing final source cannot disappear. */
TEST(PlanningImmutableLookupWork, ColdSSDAndMissingPlacementRemainExplicit) {
    const auto p=profile();const std::array discovery{0};
    const std::array mapped{device(DeviceId::cpu(),whole(p,ImmutableRowStoragePreference::MemoryMapped,DeviceId::cpu()))};
    EXPECT_TRUE(compilePlanningImmutableLookupWork(p,mapped,discovery,1).requires_storage_read_service);
    const std::array missing{device(DeviceId::cpu(),{})};
    EXPECT_THROW(compilePlanningImmutableLookupWork(p,missing,discovery,1),std::invalid_argument);
    EXPECT_TRUE(compilePlanningImmutableLookupWork(p,missing,discovery,0).memory.empty());
}
/** @test Every supported native decoder format uses its actual loader-computed row width. */
TEST(PlanningImmutableLookupWork, EveryNativeFormatUsesOriginalRowWidth) {
    constexpr std::array formats{GGUFTensorType::F32,GGUFTensorType::F16,GGUFTensorType::BF16,
        GGUFTensorType::Q4_0,GGUFTensorType::Q4_1,GGUFTensorType::Q5_0,GGUFTensorType::Q5_1,
        GGUFTensorType::Q8_0,GGUFTensorType::Q2_K,GGUFTensorType::Q3_K,GGUFTensorType::Q4_K,
        GGUFTensorType::Q5_K,GGUFTensorType::Q6_K,GGUFTensorType::Q8_K,GGUFTensorType::IQ1_S,
        GGUFTensorType::IQ1_M,GGUFTensorType::IQ2_XXS,GGUFTensorType::IQ2_XS,GGUFTensorType::IQ2_S,
        GGUFTensorType::IQ3_XXS,GGUFTensorType::IQ3_S,GGUFTensorType::IQ4_NL,GGUFTensorType::IQ4_XS};
    for(const auto format:formats) {
        SCOPED_TRACE(static_cast<int>(format));
        const ImmutableRowTableSource source{.rows=97,.row_elements=256,.format=format};
        const auto volume=PlanningImmutableLookupVolume::forTokenHeads(7,1,3,source.rowBytes(),source.row_elements);
        EXPECT_EQ(volume.nativeBytes(),21*source.rowBytes());EXPECT_EQ(volume.decodedBytes(),21u*256*sizeof(float));
    }
}
/** @test Actual ConsumerLocal RAM/cache placement pays every cold reader source miss with no capacity credit. */
TEST(PlanningImmutableLookupWork, ConsumerLocalRAMUsesAdmittedSourceAndEveryReader) {
    Qwen4ExpPlanningGGUFFixture file;PlanningModelSource source(file.path());const auto &p=source.metadata().memoryProfile();
    const std::array discovery{9};
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm}) {
        ImmutableRowTablePolicy policy{.storage=ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD};
        policy.gpu_adaptive.source_backing=AdaptiveImmutableRowBacking::RAM;
        policy.gpu_adaptive.maximum_gpu_cache_rows=64;policy.gpu_adaptive.gpu_cache_physical_byte_limit=1u<<20;
        std::vector<DevicePlanConfig> configs{sourceConsumer(backend,1,policy),sourceConsumer(backend,3,policy)};
        const std::vector<DeviceId> endpoints{configs[0].device,configs[1].device};
        const auto tp=TensorParallelConfig::equalSplit(2,p.n_heads,p.n_kv_heads,p.d_ff,p.vocab_size,endpoints,KVHeadCoveragePolicy::QueryInterval);
        for(std::size_t i=0;i<configs.size();++i) {
            auto &c=configs[i];c.total_shards=2;c.shard_index=i;c.bindTensorParallelAssignment(tp.forRank(i));
            c.ngram_tp_topology=ImmutableRowTableTPTopology::HomogeneousFullMainRankLocal;
            c.local_tp_backend=backend==DeviceType::CUDA?CollectiveBackendType::NCCL:CollectiveBackendType::RCCL;
            declareRecordedNgramHostServices(c);
        }
        const auto request=sourceRequest(p,configs,policy);
        const auto base=MemoryPlanner::plan(p,configs).physicalPlan();
        const auto selected=ImmutableRowTablePlacementPlanner::select(request,base);
        ASSERT_TRUE(selected.placement->isGPUAdaptive());
        EXPECT_EQ(selected.placement->gpuAdaptiveLayout().adaptiveGPUDomain(),ImmutableRowTableAdaptiveGPUDomain::ConsumerLocal);
        for(auto &c:configs)c.ngram_table_placement=selected.placement;
        for(const std::size_t rows:{7u,1u}) {
            const auto work=compilePlanningImmutableLookupWork(p,configs,discovery,rows);
            EXPECT_FALSE(work.requires_storage_read_service);EXPECT_EQ(work.channels.size(),2u);
            EXPECT_EQ(work.live_queries,rows*p.flash_next->ple_heads*2);
            std::size_t reads=0;for(const auto &traffic:work.memory)if(traffic.endpoint.device.is_cpu())reads+=traffic.read_bytes;
            EXPECT_EQ(reads,rows*p.flash_next->ple_heads*2*p.flash_next->ple_table_row_bytes);
        }
    }
}
/** @test A later PP stage owns live source reads/channels without becoming another PLE consumer. */
TEST(PlanningImmutableLookupWork, FixedPipelinePricesStorageOnlyStageAndUnequalDomains) {
    Qwen4ExpPlanningGGUFFixture file;PlanningModelSource source(file.path());const auto &p=source.metadata().memoryProfile();
    const std::array discovery{6};
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})
    for(const std::size_t second_width:{1u,2u}) {
        std::vector<std::vector<PhysicalMemoryAllocatorIdentity>> roster{{{0,{backend,1}},{0,{backend,4}}},{}};
        for(std::size_t i=0;i<second_width;++i)roster[1].push_back({0,{backend,static_cast<int>(i+8)}});
        const auto projection=std::make_shared<const ImmutableFixedHeadPipelineProjection>(
            ImmutableFixedHeadPipelineProjection::checked(0,0,p.flash_next->ple_heads,32,roster));
        ImmutableRowTablePolicy policy{.storage=ImmutableRowStoragePreference::VRAM};policy.distribution=ImmutableRowTableDistribution::FixedHeadSharded;
        std::vector<DevicePlanConfig> configs;
        for(std::size_t stage=0;stage<roster.size();++stage) {
            std::vector<DeviceId> endpoints;for(const auto &member:roster[stage])endpoints.push_back(member.device);
            const auto tp=TensorParallelConfig::equalSplit(endpoints.size(),p.n_heads,p.n_kv_heads,p.d_ff,p.vocab_size,endpoints,KVHeadCoveragePolicy::QueryInterval);
            for(std::size_t i=0;i<roster[stage].size();++i) {
                auto c=sourceConsumer(backend,roster[stage][i].device.ordinal,policy);
                c.first_layer=24*stage;c.last_layer=c.first_layer+23;c.total_shards=endpoints.size();c.shard_index=i;
                if(endpoints.size()>1)c.bindTensorParallelAssignment(tp.forRank(i));
                c.local_tp_backend=backend==DeviceType::CUDA?CollectiveBackendType::NCCL:CollectiveBackendType::RCCL;
                c.local_pipeline_backend=c.local_tp_backend;c.ngram_fixed_head_pipeline=projection;
                if(stage==0)c.ngram_tp_topology=ImmutableRowTableTPTopology::HomogeneousPipelineRankLocal;
                else c.ngram_execution_policy.reset();
                configs.push_back(std::move(c));
            }
        }
        const auto request=sourceRequest(p,configs,policy);
        const auto placement=request.execution_provision->fixedHeadPlacement(request.input);
        for(auto &c:configs)c.ngram_table_placement=placement;
        for(const std::size_t rows:{9u,0u,1u}) {
            const auto work=compilePlanningImmutableLookupWork(p,configs,discovery,rows);
            std::size_t reads=0,writes=0,later_reads=0;
            for(const auto &traffic:work.memory) {
                reads+=traffic.read_bytes;writes+=traffic.write_bytes;
                if(traffic.endpoint.device.ordinal>=8)later_reads+=traffic.read_bytes;
            }
            EXPECT_EQ(reads,rows*p.flash_next->ple_heads*p.flash_next->ple_table_row_bytes);
            if(rows)EXPECT_GT(later_reads,0u);
            else EXPECT_EQ(later_reads,0u);
            EXPECT_EQ(writes,2*rows*p.flash_next->ple_heads*p.flash_next->ple_row_elements*sizeof(float));
            EXPECT_EQ(work.channels.size(),rows?second_width+1:1u); // remote source edges plus one actual reader gather.
            const auto gather=std::find_if(work.channels.begin(),work.channels.end(),[](const auto &channel) {
                return channel.kind==PlanningImmutableChannelKind::NativeGather;
            });
            ASSERT_NE(gather,work.channels.end());
            EXPECT_EQ(gather->header_bytes_per_owner,8u);
            EXPECT_EQ(gather->gather_protocol,second_width==1
                ?PlanningImmutableGatherProtocol::CountedDevicePrefix:PlanningImmutableGatherProtocol::EqualDeviceRows);
            std::vector<size_t> expected_bytes(2,0);
            // The sealed projection owns the actual local/remote head assignment;
            // storage owners feeding the same reader must accumulate there.
            for(const auto &owner:projection->owners())
                expected_bytes[owner.consumer_index]+=rows*owner.head_count*p.flash_next->ple_table_row_bytes;
            EXPECT_EQ(gather->owner_live_bytes,expected_bytes);
            EXPECT_EQ(gather->native_group,(std::vector<DeviceId>{roster[0][0].device,roster[0][1].device}));
            EXPECT_EQ(gather->response_bytes,expected_bytes[0]+expected_bytes[1]);
            if(second_width==1 && rows)EXPECT_NE(expected_bytes[0],expected_bytes[1]);
        }
    }
}

}

namespace llaminar2::test {
/** @test Real automatic PP admission publishes its one final source into every escaping stage device plan. */
TEST(PlanningImmutableLookupWork, AutomaticPipelinePublishesFinalFixedHeadPlacementToStageDevices) {
    Qwen4ExpPlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    const auto &profile=source.metadata().memoryProfile();
    ASSERT_TRUE(profile.flash_next);
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm}) {
        SCOPED_TRACE(deviceTypeToString(backend));
        ClusterInventory inventory;inventory.world_size=2;
        std::vector<DeviceId> devices;
        for(int ordinal=1;ordinal<=8;++ordinal)devices.emplace_back(backend,ordinal);
        for(int original=0;original<2;++original) {
            RankInventory rank;rank.rank=original;rank.node_id=0;rank.local_rank=original;rank.hostname="pp-source-node";
            rank.cpu.numa_node=original;rank.cpu.memory_bytes=rank.cpu.free_memory_bytes=512ull<<30;
            rank.cpu_memory_bytes=rank.cpu.memory_bytes;rank.cpu_cores=rank.cpu.compute_units=rank.cpu_worker_threads=2;
            rank.cpu_execution=kSyntheticCPUExecutionGeometry;rank.numa_nodes=1;
            CPUSocketInfo socket;socket.socket_id=original;socket.numa_node=original;socket.memory_bytes=rank.cpu.memory_bytes;
            socket.physical_cores={2*original,2*original+1};rank.cpu_socket_info.push_back(std::move(socket));
            for(const auto device:devices)rank.gpus.push_back({.type=backend,.local_device_id=device.ordinal,
                .memory_bytes=512ull<<30,.free_memory_bytes=500ull<<30,.compute_units=64,
                .uuid="pp-source-gpu-"+std::to_string(device.ordinal),.numa_node=1});
            auto &count=backend==DeviceType::CUDA?rank.p2p_cuda_count:rank.p2p_rocm_count;
            auto &matrix=backend==DeviceType::CUDA?rank.p2p_cuda:rank.p2p_rocm;
            count=8;matrix.assign(64,true);inventory.ranks.push_back(std::move(rank));
        }
        inventory.buildNodeAggregations();
        OrchestrationConfig request;request.model_path=source.path();request.max_seq_len=64;
        request.mtp.enabled=false;request.mtp_activation_policy=MTPActivationPolicy::Disabled;
        request.prefix_cache.enabled=false;request.moe_rebalance.mode=MoERebalanceRuntimeMode::Off;
        request.routed_expert_compute_policy=RoutedExpertComputePolicy::GateUpOwnedDownColumns;
        request.routed_expert_owner_order=RoutedExpertOwnerOrder::Ordinal;
        request.ngram_table_policy.storage=ImmutableRowStoragePreference::VRAM;
        request.ngram_table_policy.distribution=ImmutableRowTableDistribution::FixedHeadSharded;
        request.automatic_planning.only_backends=std::vector{backend};
        request.automatic_planning.only_strategies=std::vector{OrchestrationStrategy::PipelineParallel};
        request.automatic_planning.device_counts=std::vector<AutomaticBackendDeviceCount>{{backend,8}};
        request.automatic_planning.pipeline_stage_widths=std::vector<int>{4,4};
        std::optional<AdmittedOrchestrationCandidate> admitted;
        visitAutomaticOrchestrationCandidates(request,source.metadata(),inventory,[&](auto proposal) {
            if(admitted || proposal.membership.discoveryRanks()!=std::vector<int>{1} ||
               proposal.config.pp_stage_definitions.size()!=2 || proposal.config.pp_stage_definitions[0].last_layer!=23)return;
            OrchestrationCandidateMemoryPolicy policy{.prefill_bucket_rows={32,64},.minimum_prefill_sequence_rows=1,
                .maximum_cached_prefill_buckets=8};
            policy.native_ordering_events=NativeOrderingEventGroupReplayAccess::projection(devices);
            policy.retained_host_services=planningHostServiceBudgets(inventory.world_size);
            admitted.emplace(AdmittedOrchestrationCandidate::admit(std::move(proposal),source,policy));
        });
        ASSERT_TRUE(admitted);ASSERT_TRUE(admitted->pipelineMemory());
        ASSERT_EQ(admitted->rankPlans().size(),1u);
        ASSERT_EQ(admitted->pipelineMemory()->stages().size(),2u);
        ASSERT_EQ(admitted->devicePlans().size(),8u);
        const auto placement=admitted->rankPlans().front().runtime.ngram_table_placement;
        ASSERT_TRUE(placement);ASSERT_NE(placement->fixedHeadPipeline(),nullptr);
        EXPECT_EQ(placement->worldRank(),0); // Original discovery rank1 is execution rank0.
        EXPECT_EQ(admitted->membership().discoveryRanks(),(std::vector<int>{1}));
        for(const auto &stage:admitted->pipelineMemory()->stages()) {
            EXPECT_EQ(stage.rankPlan().runtime.ngram_table_placement,placement);
            for(const auto &device:stage.devicePlans())EXPECT_EQ(device.ngram_table_placement,placement);
        }
        for(const auto &device:admitted->devicePlans())EXPECT_EQ(device.ngram_table_placement,placement);
        for(const size_t rows:{1u,9u}) {
            const auto work=compilePlanningImmutableLookupWork(profile,admitted->devicePlans(),admitted->membership().discoveryRanks(),rows);
            size_t reads=0,later_reads=0;
            for(const auto &traffic:work.memory) {
                reads+=traffic.read_bytes;
                if(traffic.endpoint.device.ordinal>=5)later_reads+=traffic.read_bytes;
            }
            EXPECT_EQ(reads,rows*profile.flash_next->ple_heads*profile.flash_next->ple_table_row_bytes);
            EXPECT_GT(later_reads,0u);EXPECT_GT(work.live_queries,0u);
        }
        auto missing=admitted->devicePlans();bool cleared=false;
        for(auto &device:missing)if(device.first_layer<=static_cast<int>(profile.flash_next->ple_decoder_layer) &&
            device.last_layer>=static_cast<int>(profile.flash_next->ple_decoder_layer)) {
            device.ngram_table_placement.reset();cleared=true;break;
        }
        ASSERT_TRUE(cleared);
        EXPECT_THROW(compilePlanningImmutableLookupWork(profile,missing,admitted->membership().discoveryRanks(),1),std::invalid_argument);
    }
}
} // namespace llaminar2::test
