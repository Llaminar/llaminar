/**
 * @file Test__DeclaredGPUCachePlacement.cpp
 * @brief Device-free original-resource and role-preservation replay for explicit cache cohorts.
 *
 * These checks exercise production request parsing, the public configuration
 * codec, and the actual checked layout producer. No worker is created and no
 * geometry declaration is mistaken for remote captured inference qualification.
 */
#include "planning/ImmutableRowTablePlacementPlanner.h"
#include "loaders/ImmutableRowAllocatorInventory.h"
#include "fixtures/planning/RecordedNgramOrdering.h"
#include "planning/ImmutableRowTableAdmissionWire.h"
#include "execution/runner/PreparedWeightAuthorityIdentity.h"
#include "loaders/AdaptiveImmutableLookupGeometry.h"
#include "config/OrchestrationConfigDocument.h"
#include "config/OrchestrationConfigParser.h"
#include "integration/parity/qwen4exp/Qwen4ExpGPUAdaptiveParityCases.h"
#include "integration/parity/qwen4exp/Qwen4ExpSourceAdmissionCI.h"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <array>
namespace llaminar2::test {
namespace {
/** @return One exact original observation, without a runtime probe or PMA allocation. */
PhysicalMemoryResource observed(DeviceId device){return {0,device,1ull<<30,1ull<<29};}
/** @return Original explicit bounded request, independently naming coordinator and ordered owners. */
ImmutableRowTablePolicy declared(){
    ImmutableRowTablePolicy p{.storage=ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD};
    p.gpu_adaptive.placement=parseAdaptiveGPUCachePlacement("0:cuda:0/64/1048576,0:rocm:0/32/524288@0:cuda:0");
    p.validate();return p;
}
/** @return Actual single PLE consumer; source owners are separately authenticated. */
ImmutableRowTablePlacementInput input(){
    const std::array consumer{observed(DeviceId::cuda(0))};
    return ImmutableRowTablePlacementInput::forConsumers(
        ImmutableRowTableExtent::fromNative(static_cast<GGUFTensorType>(0),128,64),observed(DeviceId::cpu()),consumer);
}
}
/** @test New source siblings retain canonical TP/expert/model policies and all four original setup modes. */
TEST(DeclaredGPUCachePlacement, ConsumerLocalTP2SourceSiblingsPreserveOriginalPoliciesAndSeparateQualification)
{
    using namespace parity;
    using namespace parity::qwen4exp;
    const auto& seeds=qwen4ExpConsumerLocalTP2AdaptiveSourceAdmissionSeeds();
    ASSERT_EQ(seeds.size(),4u);
    const auto expanded=qwen4ExpExpandSourceAdmissionMTP(seeds);
    ASSERT_EQ(expanded.size(),16u);
    std::array<std::size_t,4> policies{};
    for(const auto& cell:expanded) {
        ASSERT_TRUE(cell.ngram_table_policy);
        EXPECT_EQ(cell.ngram_table_policy->gpu_adaptive.placement.mode,AdaptiveGPUCachePlacementMode::ConsumerLocal);
        ASSERT_TRUE(cell.topology.expert_overlay_plan);
        const auto& plan=*cell.topology.expert_overlay_plan;
        ASSERT_EQ(plan.domains.front().participants.size(),2u);
        EXPECT_EQ(plan.routed_tiers.front().max_experts_per_layer,8);
        EXPECT_EQ(cell.expert_overlay,std::optional{ModelParityExpertOverlayPolicy{}});
        switch(cell.mtp) {
        case ModelParityMTP::Off: ++policies[0];EXPECT_EQ(cell.retained_mtp_draft_capacity,0);break;
        case ModelParityMTP::Depth1: ++policies[1];EXPECT_EQ(cell.retained_mtp_draft_capacity,15);break;
        case ModelParityMTP::Depth3: ++policies[2];EXPECT_EQ(cell.retained_mtp_draft_capacity,15);break;
        case ModelParityMTP::DynamicDepth: ++policies[3];EXPECT_EQ(cell.retained_mtp_draft_capacity,15);break;
        default: FAIL()<<"ConsumerLocal source sibling introduced another MTP mode";
        }
    }
    for(const auto count:policies)EXPECT_EQ(count,4u);
}

/** @test Concurrent channels never alias publication words or payload; endpoint identity remains authored. */
TEST(DeclaredGPUCachePlacement, ConsumerLocalServiceFramesAreDisjointAndPricedByCanonicalMappedAllocation)
{
    const auto layout = qwen4exp::ImmutableLookupLayout::createSubordinate(32, 64);
    const std::array devices{DeviceId::cuda(1), DeviceId::cuda(0)};
    const auto geometry = qwen4exp::ImmutableLookupServiceGeometry::checked(layout, devices);
    ASSERT_EQ(geometry.channelCount(), 2u);
    EXPECT_EQ(geometry.device(0), devices[0]);
    EXPECT_EQ(geometry.device(1), devices[1]);
    EXPECT_EQ(geometry.offset(0), 0u);
    EXPECT_EQ(geometry.offset(1) % 64, 0u);
    EXPECT_GE(geometry.offset(1), geometry.channelLayout(0).bytes());
    EXPECT_LE(geometry.offset(1) + geometry.channelLayout(1).bytes(), geometry.bytes());
    EXPECT_EQ(geometry.hostBytes(), TransferEngine::mappedHostRegionAllocationBytes(geometry.bytes()));
    const auto single = qwen4exp::ImmutableLookupServiceGeometry::checked(layout, std::span{devices}.first(1));
    EXPECT_EQ(single.hostBytes(), qwen4exp::ImmutableLookupService::hostBytes(layout));
    EXPECT_THROW(qwen4exp::ImmutableLookupServiceGeometry::checked(layout, std::span<const DeviceId>{}), std::invalid_argument);
    const std::array duplicate{devices[0], devices[0]};
    EXPECT_THROW(qwen4exp::ImmutableLookupServiceGeometry::checked(layout, duplicate), std::invalid_argument);
    const std::array cpu{DeviceId::cpu()};
    EXPECT_THROW(qwen4exp::ImmutableLookupServiceGeometry::checked(layout, cpu), std::invalid_argument);
    EXPECT_THROW(geometry.offset(2), std::out_of_range);
}

/** @test ConsumerLocal keeps both original requesters and projects independent local directories by key. */
TEST(DeclaredGPUCachePlacement, ConsumerLocalPluralLayoutProjectsOriginalRequesterWithoutSharedCoordinator)
{
    ImmutableRowTablePolicy policy{.storage = ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD};
    policy.gpu_adaptive.maximum_gpu_cache_rows = 64;
    policy.gpu_adaptive.gpu_cache_physical_byte_limit = 1u << 20;
    policy.validate();
    const std::array consumers{observed(DeviceId::cuda(1)), observed(DeviceId::cuda(0))};
    const auto original = ImmutableRowTablePlacementInput::forConsumers(
        ImmutableRowTableExtent::fromNative(static_cast<GGUFTensorType>(0), 128, 64),
        observed(DeviceId::cpu()), consumers);
    const std::array<std::size_t, 2> slots{0, 32};
    const auto layout = original.gpuAdaptiveLayout(policy, slots, 0, 32);
    ASSERT_EQ(layout.consumers().size(), 2u);
    ASSERT_EQ(layout.owners().size(), 3u);
    EXPECT_EQ(layout.adaptiveGPUDomain(), ImmutableRowTableAdaptiveGPUDomain::ConsumerLocal);
    EXPECT_FALSE(layout.gpuCoordinator());
    const PhysicalMemoryAllocatorIdentity first{0, DeviceId::cuda(0)}, second{0, DeviceId::cuda(1)};
    EXPECT_FALSE(layout.gpuCoordinatorFor(first));
    EXPECT_EQ(layout.gpuCoordinatorFor(second), second);
    const ImmutableRowTableSource source{.rows = 128, .row_elements = 16, .format = static_cast<GGUFTensorType>(0)};
    const auto firstGeometry = AdaptiveImmutableLookupGeometry::forRequester(layout, source, policy.gpu_adaptive, first);
    const auto secondGeometry = AdaptiveImmutableLookupGeometry::forRequester(layout, source, policy.gpu_adaptive, second);
    ASSERT_EQ(firstGeometry.physical().owners().size(), 1u);
    ASSERT_EQ(secondGeometry.physical().owners().size(), 1u);
    EXPECT_EQ(firstGeometry.physical().owners().front().allocator, first);
    EXPECT_EQ(secondGeometry.physical().owners().front().allocator, second);
    EXPECT_EQ(firstGeometry.physical().owners().front().slots, 0u);
    EXPECT_EQ(secondGeometry.physical().owners().front().slots, 32u);
    EXPECT_EQ(firstGeometry.host().sourceBytes(), secondGeometry.host().sourceBytes());
    EXPECT_THROW(AdaptiveImmutableLookupGeometry::fromLayout(layout, source, policy.gpu_adaptive), std::invalid_argument);
    EXPECT_THROW(AdaptiveImmutableLookupGeometry::forRequester(layout, source, declared().gpu_adaptive, first), std::invalid_argument);
    EXPECT_THROW(AdaptiveImmutableLookupGeometry::forRequester(layout, source, policy.gpu_adaptive,
        (PhysicalMemoryAllocatorIdentity{0, DeviceId::cuda(2)})), std::invalid_argument);
    const std::array<std::size_t, 2> zero{};
    const auto empty = original.gpuAdaptiveLayout(policy, zero, 0, 32);
    EXPECT_EQ(empty.consumers().size(), 2u);
    EXPECT_EQ(empty.owners().size(), 3u);
    EXPECT_FALSE(empty.gpuCoordinatorFor(first));
    EXPECT_FALSE(empty.gpuCoordinatorFor(second));
    const auto singleton = input().gpuAdaptiveLayout(policy, 32, 0, 32);
    EXPECT_EQ(singleton.adaptiveGPUDomain(), ImmutableRowTableAdaptiveGPUDomain::SharedCoordinator);
    EXPECT_EQ(singleton.gpuCoordinator(), first);
}

/** @test Old singleton is explicit; declared C0 participants cannot vanish or become compute domains. */
TEST(DeclaredGPUCachePlacement, PublicGrammarKeepsBothFirstClassArmsAndExactOrder){
    const auto local=parseAdaptiveGPUCachePlacement("consumer-local");
    EXPECT_EQ(local.mode,AdaptiveGPUCachePlacementMode::ConsumerLocal);EXPECT_TRUE(local.declaredParticipants().empty());
    const auto p=declared();const auto participants=p.gpu_adaptive.placement.declaredParticipants();
    ASSERT_EQ(participants.size(),2u);EXPECT_EQ(participants[0].device(),DeviceId::cuda(0));EXPECT_EQ(participants[1].device(),DeviceId::rocm(0));
    EXPECT_EQ(p.gpu_adaptive.maximum_gpu_cache_rows,0u);
    auto zero=p;for(auto &owner:zero.gpu_adaptive.placement.owners){owner.maximum_cache_rows=0;owner.cache_physical_byte_limit=0;}
    EXPECT_NO_THROW(zero.validate());EXPECT_EQ(zero.gpu_adaptive.placement.declaredParticipants(),participants);
}
/** @test Invalid requests never silently select local cache, another rank, or native same-vendor P2P. */
TEST(DeclaredGPUCachePlacement, RejectsMixedMalformedDuplicateForeignAndUnimplementedRequests){
    for(const auto *text:{"", "auto", "0:cuda:0/1/1", "0:cuda:0/1/1,@0:cuda:0", "0:cpu:0/0/0@0:cpu:0",
        "0:cuda:0/1/1,0:cuda:0/1/1@0:cuda:0", "1:rocm:0/1/1,0:cuda:0/1/1@0:cuda:0",
        "0:cuda:0/1/1,0:cuda:1/1/1@0:cuda:0", "0:rocm:0/1/1@0:cuda:0", "0:cuda:0/-1/1@0:cuda:0",
        "0:cuda:0/2147483648/1@0:cuda:0", "0:cuda:0/1/1@0:cuda:0@0:cuda:0"})
        EXPECT_THROW(parseAdaptiveGPUCachePlacement(text),std::invalid_argument)<<text;
    auto mixed=declared();mixed.gpu_adaptive.maximum_gpu_cache_rows=1;EXPECT_THROW(mixed.validate(),std::invalid_argument);
    mixed=declared();mixed.storage=ImmutableRowStoragePreference::RAM;EXPECT_THROW(mixed.validate(),std::invalid_argument);
}
/** @test Only original storage observations enter geometry; PLE inventory remains exactly one consumer. */
TEST(DeclaredGPUCachePlacement, CheckedLayoutAuthenticatesOriginalResourcesWithoutInventingComputeRoles){
    const auto p=declared();const std::array devices{observed(DeviceId::cuda(0)),observed(DeviceId::rocm(0))};
    EXPECT_THROW(input().gpuAdaptiveLayout(p,0,0,16),std::invalid_argument);
    EXPECT_THROW(input().withGPUCacheResources(p,std::span{devices}.first(1)),std::invalid_argument);
    auto conflicting=devices;conflicting[0].admission_available_bytes--;
    EXPECT_THROW(input().withGPUCacheResources(p,conflicting),std::invalid_argument);
    const auto bound=input().withGPUCacheResources(p,devices);ASSERT_EQ(bound.consumers().size(),1u);ASSERT_EQ(bound.gpuResources().size(),2u);
    const std::array<std::size_t,2> slots{16,8};const auto layout=bound.gpuAdaptiveLayout(p,slots,0,16);
    ASSERT_EQ(layout.owners().size(),3u);ASSERT_EQ(layout.consumers().size(),1u);EXPECT_EQ(layout.consumers()[0].device,DeviceId::cuda(0));
    const auto remote=std::find_if(layout.owners().begin(),layout.owners().end(),[](const auto& o){return o.key.allocator.device==DeviceId::rocm(0);});
    ASSERT_NE(remote,layout.owners().end());EXPECT_EQ(remote->cache_rows,8u);
    const std::array<std::size_t,2> zero{};const auto reset=bound.gpuAdaptiveLayout(p,zero,0,16);
    ASSERT_EQ(reset.owners().size(),3u);EXPECT_EQ(reset.owners()[1].key,layout.owners()[1].key);EXPECT_EQ(reset.consumers()[0],layout.consumers()[0]);
    auto stale=p;stale.gpu_adaptive.placement.owners[1].participant.ordinal=2;
    EXPECT_THROW(bound.gpuAdaptiveLayout(stale,slots,0,16),std::invalid_argument);
    const std::array<std::size_t,2> excessive{65,0};EXPECT_THROW(bound.gpuAdaptiveLayout(p,excessive,0,16),std::invalid_argument);
    EXPECT_THROW(bound.gpuAdaptiveLayout(p,std::span{slots}.first(1),0,16),std::invalid_argument);
}
/** @test Setup wire preserves owner order and original observations without adding graph consumers. */
TEST(DeclaredGPUCachePlacement, OriginalParticipantWireRejectsTruncationAndPreviousSingletonSchema){
    const auto p=declared();const std::array devices{observed(DeviceId::cuda(0)),observed(DeviceId::rocm(0))};
    const auto bound=input().withGPUCacheResources(p,devices);
    const std::array requests{ImmutableRowTableAdmissionRequest{p,bound,{}}};
    const auto bytes=serializeImmutableRowTableAdmissionRequests(requests);
    const auto parsed=parseImmutableRowTableAdmissionRequests(bytes,0);ASSERT_EQ(parsed.size(),1u);
    EXPECT_EQ(parsed[0].requested,p);ASSERT_EQ(parsed[0].input.consumers().size(),1u);
    ASSERT_EQ(parsed[0].input.gpuResources().size(),2u);
    EXPECT_EQ(parsed[0].input.gpuResources()[1].device,DeviceId::rocm(0));
    EXPECT_EQ(parsed[0].input.gpuResources()[1].admission_available_bytes,devices[1].admission_available_bytes);
    for(std::size_t n=0;n<bytes.size();++n)
        EXPECT_THROW((void)parseImmutableRowTableAdmissionRequests(std::span{bytes}.first(n),0),std::invalid_argument)<<n;
    auto previous=bytes;previous[4]=9;EXPECT_THROW((void)parseImmutableRowTableAdmissionRequests(previous,0),std::invalid_argument);
    EXPECT_THROW((void)parseImmutableRowTableAdmissionRequests(bytes,1),std::invalid_argument);
}
/** @test Each existing geometry BOM charges only its real owner; C0 keeps mandatory roots and consumers. */
TEST(DeclaredGPUCachePlacement, PerOwnerBOMsRemainIndependentAndPreserveMandatoryC0Roots){
    const auto p=declared();const std::array devices{observed(DeviceId::cuda(0)),observed(DeviceId::rocm(0))};
    const auto bound=input().withGPUCacheResources(p,devices);const std::array<std::size_t,2> slots{16,8},empty{};
    const ImmutableRowTableSource source{.rows=128,.row_elements=16,.format=static_cast<GGUFTensorType>(0)};
    const auto actual=AdaptiveImmutableLookupGeometry::fromLayout(bound.gpuAdaptiveLayout(p,slots,0,16),source,p.gpu_adaptive);
    const auto zero=AdaptiveImmutableLookupGeometry::fromLayout(bound.gpuAdaptiveLayout(p,empty,0,16),source,p.gpu_adaptive);
    const auto floor=recordedNgramMandatory(zero,bound.gpuResources(),bound.host());
    const auto grown=recordedNgramMandatory(actual,bound.gpuResources(),bound.host());
    const auto optional=recordedNgramOptional(actual,bound.gpuResources(),bound.host());
    PhysicalMemoryPlanBuilder reconstructed(floor),full(grown);
    for(const auto& bom:optional.resources())reconstructed.add(bom);
    const auto cache=actual.cacheBOM(bound.gpuResources(),bound.host());
    for(const auto& bom:cache.resources())full.add(bom);
    EXPECT_FALSE(reconstructed.build().requiredFootprintMismatch(full.build()));
    EXPECT_FALSE(full.build().requiredFootprintMismatch(reconstructed.build()));
    for(const auto& device:devices){
        const PhysicalMemoryAllocatorIdentity id{device.world_rank,device.device};
        ASSERT_NE(floor.find(id),nullptr);EXPECT_GT(floor.find(id)->bytes(PhysicalMemoryOwner::ImmutableLookupStaging),0u);
        ASSERT_NE(optional.find(id),nullptr);EXPECT_EQ(optional.find(id)->bytes(PhysicalMemoryOwner::ImmutableTableStorage),actual.gpuCacheBytes(id));
    }
    EXPECT_THROW((void)recordedNgramOptional(actual,std::span{devices}.first(1),bound.host()),std::invalid_argument);
    EXPECT_THROW((void)actual.gpuCacheBytes(),std::logic_error);
    auto reversed=p;std::swap(reversed.gpu_adaptive.placement.owners[0],reversed.gpu_adaptive.placement.owners[1]);
    const auto bound_reverse=input().withGPUCacheResources(reversed,devices);const std::array<std::size_t,2> reverse_slots{8,16};
    const auto reverse_geometry=AdaptiveImmutableLookupGeometry::fromLayout(bound_reverse.gpuAdaptiveLayout(reversed,reverse_slots,0,16),source,reversed.gpu_adaptive);
    EXPECT_EQ(reverse_geometry.physical().owners()[0].allocator.device,DeviceId::rocm(0));
    EXPECT_EQ(reverse_geometry.physical().owners()[1].allocator.device,DeviceId::cuda(0));
    EXPECT_EQ(reverse_geometry.physical().owners()[0].slots,8u);EXPECT_EQ(reverse_geometry.physical().owners()[1].slots,16u);
}
/** @test Public CLI/YAML use the same typed request and reject order-dependent arm mixing. */
TEST(DeclaredGPUCachePlacement, PublicCLIAndYAMLProjectExactSameStorageRequest){
    const std::string recipe="0:cuda:0/64/1048576,0:rocm:0/32/524288@0:cuda:0";
    const auto parse=[](std::vector<std::string> args){std::vector<char*> argv;for(auto& arg:args)argv.push_back(arg.data());
        return OrchestrationConfigParser{}.parseArgs(static_cast<int>(argv.size()),argv.data());};
    const auto cli=parse({"llaminar2","--ngram-storage","gpu-adaptive-vram-ram-ssd","--ngram-gpu-cache-placement",recipe});
    const auto yaml=OrchestrationConfigParser{}.parseYamlString("ngram:\n  storage: gpu-adaptive-vram-ram-ssd\n  gpu_cache_placement: "+recipe+"\n");
    EXPECT_EQ(cli.ngram_table_policy,declared());EXPECT_EQ(yaml.ngram_table_policy,cli.ngram_table_policy);
    EXPECT_THROW((void)parse({"llaminar2","--ngram-storage","gpu-adaptive-vram-ram-ssd","--ngram-gpu-cache-placement",recipe,
        "--ngram-gpu-cache-max-rows","1"}),std::invalid_argument);
    EXPECT_THROW((void)parse({"llaminar2","--ngram-storage","gpu-adaptive-vram-ram-ssd","--ngram-gpu-cache-max-rows","1",
        "--ngram-gpu-cache-placement",recipe}),std::invalid_argument);
}
/** @test The exact existing policy codec retains ordered owners and scalar-arm separation. */
TEST(DeclaredGPUCachePlacement, PublicDocumentPreservesEveryRequestedOwnerAndCoordinator){
    OrchestrationConfig c;c.ngram_table_policy=declared();
    const auto encoded=serializeOrchestrationConfig(c);const auto restored=deserializeOrchestrationConfig(encoded);
    EXPECT_EQ(restored.ngram_table_policy,c.ngram_table_policy);
    auto bad=nlohmann::json::parse(encoded);bad["configuration"]["ngram_table_policy"]["gpu_adaptive"]["placement"]["mode"]="Automatic";
    EXPECT_THROW((void)deserializeOrchestrationConfig(bad.dump()),std::invalid_argument);
}
/** @test Typed source discovery keeps storage hardware requirements out of compute topology. */
TEST(DeclaredGPUCachePlacement, TypedDiscoverySeparatesStorageFromComputeParticipants){
    using namespace parity;using namespace parity::qwen4exp;
    for(const auto mode:{Qwen4ExpParityTopology::CUDACPU,Qwen4ExpParityTopology::ROCmCPU}){
        for(const auto& cell:expandModelParityDefinition(qwen4ExpDeclaredGPUCacheSourceDefinition(mode))){
            std::ostringstream out;PrintTo(Qwen4ExpSourceAdmissionCase(cell),&out);
            const auto record=nlohmann::json::parse(out.str());
            ASSERT_EQ(record["participants"].size(),2u);ASSERT_EQ(record["storage_participants"].size(),2u);
            EXPECT_EQ(record["storage_coordinator"]["backend"],mode==Qwen4ExpParityTopology::CUDACPU?"cuda":"rocm");
            EXPECT_EQ(record["storage_participants"][1]["backend"],mode==Qwen4ExpParityTopology::CUDACPU?"rocm":"cuda");
            EXPECT_LT(cell.testName().size(),256u);
        }
    }
}
/** @test Every owner/coordinator/limit change separates original retained capacity authority. */
TEST(DeclaredGPUCachePlacement, RetainedAuthorityAuthenticatesAuthoredOwnerOrderAndCeilings){
    using namespace parity;using namespace parity::qwen4exp;
    auto cell=expandModelParityDefinition(qwen4ExpDeclaredGPUCacheSourceDefinition(Qwen4ExpParityTopology::CUDACPU)).front();
    auto config=cell.makeOrchestrationConfig("/tmp/original.gguf",0);
    const auto plan=config.moe_routed_expert_plan;ASSERT_TRUE(plan);
    const auto original=routedWeightAuthorityRequestIdentity(plan,config);ASSERT_FALSE(original.empty());
    for(int mutation=0;mutation<5;++mutation){
        auto changed=config;auto& p=changed.ngram_table_policy.gpu_adaptive.placement;
        if(mutation==0)std::swap(p.owners[0],p.owners[1]);
        if(mutation==1)++p.owners[1].maximum_cache_rows;
        if(mutation==2)++p.owners[1].cache_physical_byte_limit;
        if(mutation==3)++p.owners[1].participant.ordinal;
        if(mutation==4)p.coordinator=p.owners[1].participant;
        EXPECT_NE(routedWeightAuthorityRequestIdentity(plan,changed),original)<<mutation;
    }
    EXPECT_EQ(routedWeightAuthorityRequestIdentity(plan,config),original);
}
/** @test Cache owners remain distinct from actual full-main continuation and original expert policy domains. */
TEST(DeclaredGPUCachePlacement, CanonicalCandidatePreservesComputeExpertAndMTPPolicies){
    using namespace parity;using namespace parity::qwen4exp;
    for(const auto mode:{Qwen4ExpParityTopology::CUDACPU,Qwen4ExpParityTopology::ROCmCPU}){
        const auto ordinary=qwen4ExpExpandSourceAdmissionMTP(expandModelParityDefinition(qwen4ExpGPUAdaptiveParityDefinition(mode)));
        const auto explicit_owners=qwen4ExpExpandSourceAdmissionMTP(expandModelParityDefinition(qwen4ExpDeclaredGPUCacheSourceDefinition(mode)));
        ASSERT_EQ(explicit_owners.size(),ordinary.size());
        for(std::size_t i=0;i<ordinary.size();++i){
            const auto a=ordinary[i].makeOrchestrationConfig("/tmp/original.gguf",0),b=explicit_owners[i].makeOrchestrationConfig("/tmp/original.gguf",0);
            auto normalized=b;normalized.ngram_table_policy=a.ngram_table_policy;
            EXPECT_EQ(serializeOrchestrationConfig(a),serializeOrchestrationConfig(normalized));
            EXPECT_EQ(b.ngram_table_policy.gpu_adaptive.placement.declaredParticipants().size(),2u);
            EXPECT_EQ(explicit_owners[i].topology.participants.size(),ordinary[i].topology.participants.size());
            EXPECT_NE(explicit_owners[i].testName(),ordinary[i].testName());
            for(const auto ch:explicit_owners[i].testName())EXPECT_TRUE(std::isalnum(static_cast<unsigned char>(ch)) || ch=='_');
        }
    }
}
}

namespace llaminar2::test {
/** @test Physical inventory borrows all checked owners while keeping consumers separate, including C0. */
TEST(DeclaredGPUCachePlacement, PhysicalInventoryBorrowsStorageOwnersWithoutDuplicatingConsumers){
    const auto p=declared();const std::array devices{observed(DeviceId::cuda(0)),observed(DeviceId::rocm(0))};
    const auto bound=input().withGPUCacheResources(p,devices);
    for(const auto slots:{std::array<std::size_t,2>{0,0},std::array<std::size_t,2>{16,8}}){
        const auto layout=bound.gpuAdaptiveLayout(p,slots,0,16);
        const ImmutableRowAllocatorInventory owners(std::span<const ImmutableRowTableLayoutOwner>{layout.owners()});
        ASSERT_EQ(owners.size(),3u);ASSERT_EQ(layout.consumers().size(),1u);
        std::size_t i=0;for(auto it=owners.begin();it!=owners.end();++it,++i){
            EXPECT_EQ(&*it,&layout.owners()[i].key.allocator);
            EXPECT_EQ(it.operator->(),&layout.owners()[i].key.allocator);
        }
        for(const auto device:{DeviceId::cpu(),DeviceId::cuda(0),DeviceId::rocm(0)})
            EXPECT_EQ(std::count(owners.begin(),owners.end(),PhysicalMemoryAllocatorIdentity{0,device}),1);
        const auto copied=owners;EXPECT_EQ(copied.begin(),owners.begin());EXPECT_EQ(copied.end(),owners.end());
    }
}

}
