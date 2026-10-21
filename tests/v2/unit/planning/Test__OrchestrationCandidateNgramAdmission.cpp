/**
 * @file Test__OrchestrationCandidateNgramAdmission.cpp
 * @brief Real metadata-source candidate finalization after immutable row/quota decisions.
 *
 * Sparse original GGUF extents are parsed by the real loader and every learned
 * FA/GDN/MoE projection has complete source geometry. No payload, device context,
 * inference graph or synthetic allocation authority is materialized here.
 */
#include "planning/OrchestrationCandidateAdmission.h"
#include "planning/RankMemoryPlanInputs.h"
#include "loaders/ImmutableRowTable.h"
#include "loaders/CpuAdaptiveImmutableRowCache.h"
#include "config/OrchestrationConfigDocument.h"
#include "../../utils/Qwen4ExpPlanningGGUFFixture.h"
#include "../../utils/CPUExecutionTestGeometry.h"
#include <gtest/gtest.h>
using namespace llaminar2;
namespace {
/** @return Explicit observed resources; these are not runtime capacity balances. */
ClusterInventory candidateInventory(DeviceType backend,int ranks=1,bool gpu_every_rank=false)
{
    ClusterInventory result;result.world_size=ranks;
    for(int rank=0;rank<ranks;++rank) {
        RankInventory r;r.rank=rank;r.node_id=rank;r.local_rank=0;r.hostname="candidate-"+std::to_string(rank);
        r.cpu.numa_node=0;r.cpu.memory_bytes=512ull<<30;r.cpu.free_memory_bytes=500ull<<30;
        r.cpu.compute_units=r.cpu_cores=r.cpu_worker_threads=8;r.numa_nodes=1;
        r.cpu_execution=test::kSyntheticCPUExecutionGeometry;
        if(backend!=DeviceType::CPU&&(rank==0||gpu_every_rank))
            r.gpus.push_back({.type=backend,.local_device_id=2,.memory_bytes=512ull<<30,
                .free_memory_bytes=500ull<<30,.compute_units=64,.uuid="candidate-gpu-"+std::to_string(rank),.numa_node=3});
        result.ranks.push_back(std::move(r));
    }
    result.buildNodeAggregations();return result;
}
/** @return One authentic expander-produced explicit candidate for the required rank set. */
AutomaticOrchestrationCandidate ngramCandidate(const PlanningModelSource &source,DeviceType backend,
    OrchestrationStrategy strategy=OrchestrationStrategy::SingleDevice,int ranks=1,bool gpu_every_rank=false)
{
    OrchestrationConfig request;request.model_path=source.path();request.max_seq_len=64;
    request.mtp.enabled=false;request.prefix_cache.enabled=false;request.moe_rebalance.mode=MoERebalanceRuntimeMode::Off;
    request.automatic_planning.only_backends=strategy==OrchestrationStrategy::ExpertOverlay
        ?std::vector{backend,DeviceType::CPU}:std::vector{backend};
    request.automatic_planning.only_strategies={strategy};
    std::optional<AutomaticOrchestrationCandidate> selected;
    visitAutomaticOrchestrationCandidates(request,source.metadata(),candidateInventory(backend,ranks,gpu_every_rank),[&](auto c) {
        if(c.membership.discoveryRanks().size()==static_cast<std::size_t>(ranks))selected=std::move(c);
    });
    if(!selected)throw std::logic_error("Complete ngram fixture emitted no requested candidate");
    return std::move(*selected);
}
/**
 * @brief Author exact cross-rank continuation or expert-only membership.
 * @param source Original metadata source, unchanged by the topology declaration.
 * @param backend CUDA or ROCm family shared by both explicit GPU participants.
 * @param scope NODE_LOCAL declares supported distributed dense TP; GLOBAL
 *        preserves owner-only dense continuation and a remote expert endpoint.
 * @return Explicit candidate; no automatic pool search or inferred PLE owner.
 *
 * Distributed dense continuation is supported by the NODE_LOCAL TP contract.
 * Its two ranks share one physical host and own distinct GPU ordinals. A GLOBAL
 * expert domain alone does not create another dense continuation consumer.
 */
AutomaticOrchestrationCandidate multirankNgramCandidate(const PlanningModelSource &source,DeviceType backend,
    ExecutionDomainScope scope=ExecutionDomainScope::NODE_LOCAL)
{
    auto inventory=candidateInventory(backend,2,true);
    if(scope==ExecutionDomainScope::NODE_LOCAL) {
        for(auto &rank:inventory.ranks) {
            rank.node_id=0;rank.hostname="candidate-node";
            rank.gpus.front().local_device_id=2+rank.rank;
        }
        inventory.buildNodeAggregations();
    }
    ExecutionRankMembership membership(inventory,{0,1});
    OrchestrationConfig config;config.model_path=source.path();config.max_seq_len=64;
    config.planning_mode=OrchestrationPlanningMode::Apply;
    config.mtp.enabled=false;config.prefix_cache.enabled=false;
    config.moe_rebalance.mode=MoERebalanceRuntimeMode::Off;
    ExecutionDomainDefinition domain;domain.name="global-compute";
    domain.scope=scope;domain.ranks={0,1};domain.owner_rank=0;
    for(const auto &rank:inventory.ranks) {
        const auto ordinal=rank.gpus.front().local_device_id;
        domain.participants.push_back(backend==DeviceType::CUDA
            ?GlobalDeviceAddress::cuda(ordinal,3,rank.hostname):GlobalDeviceAddress::rocm(ordinal,3,rank.hostname));
    }
    domain.routed_compute_policy=config.routed_expert_compute_policy;
    auto overlay=std::make_shared<MoERoutedExpertPlacementPlan>();overlay->enabled=true;
    overlay->owner_order=config.routed_expert_owner_order;
    overlay->residency_policy=defaultRoutedExpertResidencyPolicy(config.moe_rebalance.mode);
    overlay->continuation_domain=domain.name;overlay->base_model_domain=domain.name;
    overlay->shared_expert_domain=domain.name;
    overlay->continuation_domain_spec.domain=domain.name;
    overlay->continuation_domain_spec.setDensePolicy(DenseParallelPolicy::TensorParallel);
    overlay->continuation_dense_policy_intent=MoEContinuationDensePolicyIntent::Explicit;
    overlay->domains={RoutedExpertDomain::fromExecutionDomainDefinition(domain)};
    overlay->routed_tiers={{.name="global-experts",.domain=domain.name,.priority=0}};
    config.moe_routed_expert_plan=std::move(overlay);
    const auto errors=normalizeMoERoutedExpertPlacementDomains(config);
    if(!errors.empty())throw std::logic_error("Explicit global TP fixture normalization failed: "+errors.front());
    return {OrchestrationStrategy::TensorParallel,std::move(membership),std::move(config)};
}
/** @return Frozen small capture ladder and original bounded upload policy. */
OrchestrationCandidateMemoryPolicy ngramPolicy(std::size_t slot=1u<<20)
{return {.prefill_bucket_rows={32,64},.minimum_prefill_sequence_rows=1,.maximum_cached_prefill_buckets=8,
    .weight_load={.staging_stream_count=1,.staging_budget_bytes=slot}};}
/** @brief Authenticate pointer publication and exact canonical table owner on every admitted consumer. */
void expectCompleteTable(const AdmittedOrchestrationCandidate &admitted,ImmutableRowStorage storage)
{
    std::size_t placements=0;
    for(const auto &rank:admitted.rankPlans()) {
        const auto &placement=rank.runtime.ngram_table_placement;if(!placement)continue;++placements;
        EXPECT_EQ(placement->storage(),storage);EXPECT_EQ(placement->worldRank(),rank.rank);
        for(const auto &consumer:placement->consumers()) {
            const auto device=std::find_if(admitted.devicePlans().begin(),admitted.devicePlans().end(),[&](const auto &d) {
                return d.world_rank==consumer.world_rank&&d.device==consumer.device;
            });
            ASSERT_NE(device,admitted.devicePlans().end());EXPECT_EQ(device->ngram_table_placement,placement);
        }
        for(const auto &owner:placement->allocators()) {
            const auto *bom=admitted.physicalAdmission().plan().find(owner);ASSERT_NE(bom,nullptr);
            EXPECT_EQ(bom->bytes(PhysicalMemoryOwner::ImmutableTableStorage),
                ImmutableRowTable::physicalBytesForNativeExtent(placement->extent().nativeBytes(),placement->storagePolicy()));
        }
    }
    EXPECT_GT(placements,0u);
}
}
/** @test Ordinary candidates seal concrete original table owners and publish exact consumer bindings. */
TEST(OrchestrationCandidateNgramAdmission, OrdinaryFinalCertificateIncludesRequestedTierAndAutomaticOrder)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource source(file.path());
    for(const auto backend:{DeviceType::CPU,DeviceType::CUDA,DeviceType::ROCm}) {
        for(const auto requested:{ImmutableRowStoragePreference::RAM,ImmutableRowStoragePreference::MemoryMapped,
            ImmutableRowStoragePreference::Automatic,ImmutableRowStoragePreference::VRAM}) {
            SCOPED_TRACE(::testing::Message()<<int(backend)<<":"<<int(requested));
            auto proposal=ngramCandidate(source,backend);proposal.config.ngram_table_policy={requested,65536,3};
            if(backend==DeviceType::CPU&&requested==ImmutableRowStoragePreference::VRAM) {
                EXPECT_THROW((void)AdmittedOrchestrationCandidate::admit(proposal,source,ngramPolicy()),std::invalid_argument);continue;
            }
            const auto result=AdmittedOrchestrationCandidate::admit(proposal,source,ngramPolicy());
            const auto expected=requested==ImmutableRowStoragePreference::MemoryMapped?ImmutableRowStorage::MemoryMapped:
                requested==ImmutableRowStoragePreference::RAM||backend==DeviceType::CPU?ImmutableRowStorage::RAM:ImmutableRowStorage::VRAM;
            expectCompleteTable(result,expected);
        }
        if(backend==DeviceType::CPU)continue;
        auto hosted=ngramCandidate(source,backend);hosted.config.ngram_table_policy={ImmutableRowStoragePreference::Automatic,65536,3};
        const auto resident=AdmittedOrchestrationCandidate::admit(hosted,source,ngramPolicy());
        const auto placement=resident.rankPlans().front().runtime.ngram_table_placement;ASSERT_TRUE(placement);
        ASSERT_EQ(placement->storage(),ImmutableRowStorage::VRAM);
        const auto *gpu=resident.physicalAdmission().plan().find(placement->allocators().front());ASSERT_NE(gpu,nullptr);
        // The actual complete VRAM candidate misses by one byte, while its
        // frozen higher-priority base still fits without the table bank.
        auto limited=hosted.membership.inventory();limited.ranks.front().gpus.front().free_memory_bytes=gpu->totalBytes()-1;
        limited.buildNodeAggregations();hosted.membership=ExecutionRankMembership(limited,hosted.membership.discoveryRanks());
        expectCompleteTable(AdmittedOrchestrationCandidate::admit(hosted,source,ngramPolicy()),ImmutableRowStorage::RAM);
        limited.ranks.front().cpu.free_memory_bytes=2ull<<30;limited.buildNodeAggregations();
        hosted.membership=ExecutionRankMembership(limited,hosted.membership.discoveryRanks());
        expectCompleteTable(AdmittedOrchestrationCandidate::admit(hosted,source,ngramPolicy()),ImmutableRowStorage::MemoryMapped);
    }
}
/** @test Distinct ranks extend the same certificate once each without same-rank table collisions. */
TEST(OrchestrationCandidateNgramAdmission, MultirankConsumersRetainDistinctOriginalTableOwners)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource source(file.path());
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm}) {
        auto proposal=multirankNgramCandidate(source,backend);
        proposal.config.ngram_table_policy.storage=ImmutableRowStoragePreference::VRAM;
        const auto result=AdmittedOrchestrationCandidate::admit(proposal,source,ngramPolicy());
        ASSERT_EQ(result.rankPlans().size(),2u);expectCompleteTable(result,ImmutableRowStorage::VRAM);
        const auto first=result.rankPlans()[0].runtime.ngram_table_placement;
        const auto second=result.rankPlans()[1].runtime.ngram_table_placement;
        ASSERT_TRUE(first);ASSERT_TRUE(second);EXPECT_NE(first,second);EXPECT_NE(first->worldRank(),second->worldRank());
        ASSERT_EQ(first->consumers().size(),1u);ASSERT_EQ(second->consumers().size(),1u);
        EXPECT_EQ(first->consumers().front().device,DeviceId(backend,2));
        EXPECT_EQ(second->consumers().front().device,DeviceId(backend,3));
        EXPECT_EQ(first->extent().nativeBytes(),second->extent().nativeBytes());
        for(const auto &rank:result.rankPlans()) {
            EXPECT_TRUE(rank.usesGlobalTP());
            EXPECT_EQ(rank.weight_shard.total_shards,2);
            const auto consumer=std::find_if(result.devicePlans().begin(),result.devicePlans().end(),[&](const auto &device) {
                return device.world_rank==rank.rank&&device.device==DeviceId(backend,2+rank.rank);
            });
            ASSERT_NE(consumer,result.devicePlans().end());
            EXPECT_EQ(consumer->execution_role,DeviceExecutionMemoryRole::ContinuationGraph);
            EXPECT_EQ(consumer->first_layer,0);
            EXPECT_GE(consumer->last_layer,static_cast<int>(source.metadata().memoryProfile().flash_next->ple_decoder_layer));
        }
        ASSERT_TRUE(result.overlayCapacity());
        EXPECT_EQ(result.overlayCapacity()->physical_memory_admission.get(),&result.physicalAdmission());
        ASSERT_EQ(result.overlayCapacity()->ngram_table_placements.size(),2u);
        EXPECT_EQ(result.overlayCapacity()->ngram_table_placements[0],first);
        EXPECT_EQ(result.overlayCapacity()->ngram_table_placements[1],second);
    }
}
/** @test A remote expert endpoint does not become another PLE or immutable-table owner. */
TEST(OrchestrationCandidateNgramAdmission, GlobalExpertOnlyRankOwnsNoImmutableTable)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource source(file.path());
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm}) {
        auto proposal=multirankNgramCandidate(source,backend,ExecutionDomainScope::GLOBAL);
        proposal.config.ngram_table_policy.storage=ImmutableRowStoragePreference::VRAM;
        const auto result=AdmittedOrchestrationCandidate::admit(proposal,source,ngramPolicy());
        ASSERT_EQ(result.rankPlans().size(),2u);
        ASSERT_TRUE(result.rankPlans()[0].runtime.ngram_table_placement);
        EXPECT_FALSE(result.rankPlans()[1].runtime.ngram_table_placement);
        const auto endpoint=std::find_if(result.devicePlans().begin(),result.devicePlans().end(),[&](const auto &device) {
            return device.world_rank==1&&device.device==DeviceId(backend,2);
        });
        ASSERT_NE(endpoint,result.devicePlans().end());
        EXPECT_EQ(endpoint->execution_role,DeviceExecutionMemoryRole::RoutedExpertParticipant);
        EXPECT_FALSE(endpoint->ngram_table_placement);
        for(const auto &bom:result.physicalAdmission().plan().resources())
            if(bom.resource().world_rank==1)
                EXPECT_EQ(bom.bytes(PhysicalMemoryOwner::ImmutableTableStorage),0u);
        ASSERT_TRUE(result.overlayCapacity());
        ASSERT_EQ(result.overlayCapacity()->ngram_table_placements.size(),1u);
        EXPECT_EQ(result.overlayCapacity()->ngram_table_placements.front()->worldRank(),0);
    }
}
/** @test A final table failure cannot select smaller fitting rows or publish an incomplete ordinary/overlay certificate. */
TEST(OrchestrationCandidateNgramAdmission, FinalTableRejectionNeverReopensRowsOrExpertQuotaSearch)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource source(file.path());
    for(const auto strategy:{OrchestrationStrategy::SingleDevice,OrchestrationStrategy::ExpertOverlay})
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm}) {
        SCOPED_TRACE(::testing::Message()<<int(strategy)<<":"<<int(backend));
        auto proposal=ngramCandidate(source,backend,strategy,strategy==OrchestrationStrategy::ExpertOverlay?2:1);
        proposal.config.ngram_table_policy={ImmutableRowStoragePreference::RAM,65536,3};
        const auto high=AdmittedOrchestrationCandidate::admit(proposal,source,ngramPolicy());
        ASSERT_EQ(high.rankPlans().front().runtime.resident_graph_rows,64);
        const auto placement=high.rankPlans().front().runtime.ngram_table_placement;ASSERT_TRUE(placement);
        const auto *host=high.physicalAdmission().plan().find(placement->allocators().front());ASSERT_NE(host,nullptr);
        auto limited=proposal.membership.inventory();limited.ranks.front().cpu.free_memory_bytes=host->totalBytes()-1;
        limited.buildNodeAggregations();proposal.membership=ExecutionRankMembership(limited,proposal.membership.discoveryRanks());
        const auto original=serializeOrchestrationConfig(proposal.config);
        try {(void)AdmittedOrchestrationCandidate::admit(proposal,source,ngramPolicy());FAIL()<<"Final table exceeded its frozen owner";}
        catch(const PhysicalMemoryCapacityExhausted &e) {EXPECT_NE(std::string(e.what()).find("Explicit ngram storage cannot be admitted"),std::string::npos)<<e.what();}
        EXPECT_EQ(serializeOrchestrationConfig(proposal.config),original);
        auto small=ngramPolicy();small.prefill_bucket_rows={32};
        const auto lower=AdmittedOrchestrationCandidate::admit(proposal,source,small);
        ASSERT_EQ(lower.rankPlans().front().runtime.resident_graph_rows,32);expectCompleteTable(lower,ImmutableRowStorage::RAM);
        if(high.overlayCapacity()) {
            ASSERT_TRUE(lower.overlayCapacity());
            EXPECT_EQ(high.overlayCapacity()->tiers.size(),lower.overlayCapacity()->tiers.size());
            for(std::size_t tier=0;tier<high.overlayCapacity()->tiers.size();++tier)
                EXPECT_EQ(high.overlayCapacity()->tiers[tier].live_experts_per_layer,
                    lower.overlayCapacity()->tiers[tier].live_experts_per_layer);
            ASSERT_EQ(lower.overlayCapacity()->ngram_table_placements.size(),1u);
            EXPECT_EQ(lower.overlayCapacity()->ngram_table_placements.front(),lower.rankPlans().front().runtime.ngram_table_placement);
        }
    }
}

/** @test Mandatory hosted execution affects the first quota decision, never a final quota retry. */
TEST(OrchestrationCandidateNgramAdmission, HostedExecutionIsInFixedBaseBeforeExpertQuotasFreeze)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource source(file.path());
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})
    for(const auto preference:{ImmutableRowStoragePreference::RAM,ImmutableRowStoragePreference::MemoryMapped})
    {
        SCOPED_TRACE(::testing::Message()<<int(backend)<<":"<<int(preference));
        auto proposal=ngramCandidate(source,backend,OrchestrationStrategy::ExpertOverlay,2);
        proposal.config.ngram_table_policy={preference,65536,3};
        const auto high=AdmittedOrchestrationCandidate::admit(proposal,source,ngramPolicy());
        ASSERT_TRUE(high.overlayCapacity());
        auto groups=std::vector<std::vector<DevicePlanConfig>>{high.devicePlans()};
        for(auto &cfg:groups.front())cfg.ngram_table_placement.reset();
        const auto uploads=MemoryPlanner::immutableRowTableUploadBindings(source.metadata().memoryProfile(),groups);
        const auto q=MemoryPlanner::immutableRowTableAdmissionRequest(source.metadata().memoryProfile(),
            proposal.config.ngram_table_policy,groups,uploads,
            ImmutableRowTableHostPlacement::fromObservedNUMANode(high.rankPlans().front().numa_node));
        ASSERT_TRUE(q);ASSERT_TRUE(q->execution_provision);
        const auto device=DeviceId(backend,2);
        const auto *execution=q->execution_provision->executionPlan().find({0,device});ASSERT_NE(execution,nullptr);
        const auto mandatory=execution->totalBytes();ASSERT_GT(mandatory,1u);
        const auto *full=high.physicalAdmission().plan().find({0,device});ASSERT_NE(full,nullptr);
        ASSERT_GT(full->bytes(PhysicalMemoryOwner::RoutedExpertWeights),mandatory);
        // The old incomplete fixed base allowed all of these expert slots, then
        // failed when final hosted lookup execution was appended. This budget
        // admits that exact old footprint but not the real complete footprint.
        auto inventory=proposal.membership.inventory();
        inventory.ranks.front().gpus.front().free_memory_bytes=full->totalBytes()-mandatory+1;
        inventory.buildNodeAggregations();proposal.membership=ExecutionRankMembership(inventory,proposal.membership.discoveryRanks());
        const auto admitted=AdmittedOrchestrationCandidate::admit(proposal,source,ngramPolicy());
        ASSERT_TRUE(admitted.overlayCapacity());EXPECT_TRUE(admitted.physicalAdmission().plan().fits());
        EXPECT_EQ(admitted.rankPlans().front().runtime.resident_graph_rows,high.rankPlans().front().runtime.resident_graph_rows);
        const auto *limited=admitted.physicalAdmission().plan().find({0,device});ASSERT_NE(limited,nullptr);
        EXPECT_LT(limited->bytes(PhysicalMemoryOwner::RoutedExpertWeights),full->bytes(PhysicalMemoryOwner::RoutedExpertWeights));
        expectCompleteTable(admitted,preference==ImmutableRowStoragePreference::RAM?ImmutableRowStorage::RAM:ImmutableRowStorage::MemoryMapped);
    }
}

/** @test Explicit adaptive ceilings cannot reopen frozen graph rows or expert grants after mandatory execution is admitted. */
TEST(OrchestrationCandidateNgramAdmission, HostAdaptiveRamSsdOptionalSlotsNeverRepriceHigherPriorityOwners)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource source(file.path());
    for(const auto backend:{DeviceType::CPU,DeviceType::CUDA,DeviceType::ROCm})
    for(const auto strategy:{OrchestrationStrategy::SingleDevice,OrchestrationStrategy::ExpertOverlay})
    {
        if(backend==DeviceType::CPU && strategy==OrchestrationStrategy::ExpertOverlay)continue;
        auto proposal=ngramCandidate(source,backend,strategy,strategy==OrchestrationStrategy::ExpertOverlay?2:1);
        proposal.config.ngram_table_policy={ImmutableRowStoragePreference::HostAdaptiveRAMSSD,65536,3};
        const auto zero=AdmittedOrchestrationCandidate::admit(proposal,source,ngramPolicy());
        proposal.config.ngram_table_policy.host_adaptive={17,1u<<20,{}};
        const auto optional=AdmittedOrchestrationCandidate::admit(proposal,source,ngramPolicy());
        ASSERT_EQ(zero.rankPlans().size(),optional.rankPlans().size());std::size_t providers=0;
        for(std::size_t rank=0;rank<zero.rankPlans().size();++rank)
        {
            const auto &a=zero.rankPlans()[rank],&b=optional.rankPlans()[rank];
            EXPECT_EQ(a.runtime.resident_graph_rows,b.runtime.resident_graph_rows);
            if(!a.runtime.ngram_table_placement){EXPECT_FALSE(b.runtime.ngram_table_placement);continue;}
            ++providers;ASSERT_TRUE(b.runtime.ngram_table_placement);
            const auto &pa=*a.runtime.ngram_table_placement,&pb=*b.runtime.ngram_table_placement;
            EXPECT_TRUE(pa.isHostAdaptive());EXPECT_TRUE(pb.isHostAdaptive());
            EXPECT_EQ(pa.hostAdaptiveLayout().owners().front().cache_rows,0u);
            EXPECT_EQ(pb.hostAdaptiveLayout().owners().front().cache_rows,17u);
            EXPECT_EQ(pa.extent(),pb.extent());EXPECT_EQ(pa.hostAdaptiveMaximumLiveRows(),pb.hostAdaptiveMaximumLiveRows());
        }
        EXPECT_GT(providers,0u);
        for(const auto &line:zero.physicalAdmission().plan().resources())
        {
            const auto *other=optional.physicalAdmission().plan().find({line.resource().world_rank,line.resource().device});ASSERT_NE(other,nullptr);
            for(std::size_t i=0;i<PhysicalMemoryBOM::ownerCount();++i)
            {
                const auto owner=PhysicalMemoryBOM::ownerAt(i);if(owner==PhysicalMemoryOwner::ImmutableTableStorage)continue;
                EXPECT_EQ(line.bytes(owner),other->bytes(owner));
            }
        }
        if(zero.overlayCapacity())
        {
            ASSERT_TRUE(optional.overlayCapacity());ASSERT_EQ(zero.overlayCapacity()->tiers.size(),optional.overlayCapacity()->tiers.size());
            for(std::size_t t=0;t<zero.overlayCapacity()->tiers.size();++t)
                EXPECT_EQ(zero.overlayCapacity()->tiers[t].live_experts_per_layer,optional.overlayCapacity()->tiers[t].live_experts_per_layer);
        }
    }
}
