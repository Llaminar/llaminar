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
#include <set>
#include <array>
#include "collective/LocalTPCollectiveInventory.h"
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
/** @test Same-vendor intent needs exact original None evidence; syntax alone grants no transport. */
TEST(DeclaredGPUCachePlacement, ExplicitNoPeerUsesOriginalInventoryAndSurvivesWireWithoutComputeRoles)
{
    for (const auto type : {DeviceType::CUDA, DeviceType::ROCm}) {
        const std::string backend = type == DeviceType::CUDA ? "cuda" : "rocm";
        for (const unsigned count : {2u, 4u, 8u}) {
            RankInventory inventory;
            inventory.rank = 0;
            std::string text;
            std::vector<PhysicalMemoryResource> resources;
            for (unsigned i = 0; i < count; ++i) {
                inventory.gpus.push_back(DeviceInfo{.type = type, .local_device_id = static_cast<int>(i)});
                if (i) text += ",";
                text += "0:" + backend + ":" + std::to_string(i) + "/64/1048576";
                resources.push_back(observed({type, static_cast<int>(i)}));
            }
            text += "@0:" + backend + ":0";
            auto& matrix = type == DeviceType::CUDA ? inventory.p2p_cuda : inventory.p2p_rocm;
            auto& matrixCount = type == DeviceType::CUDA ? inventory.p2p_cuda_count : inventory.p2p_rocm_count;
            matrix.assign(count * count, false);
            // Only coordinator-owner edges carry packets. Unused owner-owner
            // connectivity must not manufacture a rejection of the star.
            for (unsigned from = 1; from < count; ++from)
                for (unsigned to = 1; to < count; ++to)
                    if (from != to) matrix[from * count + to] = true;
            matrixCount = count;
            ImmutableRowTablePolicy policy{.storage = ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD};
            policy.gpu_adaptive.placement = parseAdaptiveGPUCachePlacement(text);
            EXPECT_EQ(ImmutableRowTableDirectedTransportPolicy::observe(policy.gpu_adaptive.placement, inventory), PeerAccessCoverage::None);
            auto originalInventory = std::make_shared<ClusterInventory>();
            originalInventory->world_size = 1;
            originalInventory->ranks.push_back(inventory);
            std::vector<PhysicalMemoryAllocatorIdentity> endpoints;
            for (const auto& resource : resources) endpoints.push_back({0,resource.device});
            EXPECT_NO_THROW(ImmutableRowTableDirectedTransportPolicy::requireOriginal(endpoints.front(),endpoints,
                originalInventory,PeerAccessCoverage::None));
            EXPECT_THROW(ImmutableRowTableDirectedTransportPolicy::requireOriginal(endpoints.front(),endpoints,
                {},PeerAccessCoverage::None),std::invalid_argument);
            auto wrongInventory = std::make_shared<ClusterInventory>(*originalInventory);
            wrongInventory->ranks.front().rank = 1;
            EXPECT_THROW(ImmutableRowTableDirectedTransportPolicy::requireOriginal(endpoints.front(),endpoints,
                wrongInventory,PeerAccessCoverage::None),std::invalid_argument);
            auto changedInventory = std::make_shared<ClusterInventory>(*originalInventory);
            auto& changedMatrix = type == DeviceType::CUDA ? changedInventory->ranks.front().p2p_cuda : changedInventory->ranks.front().p2p_rocm;
            changedMatrix[1] = true;
            EXPECT_THROW(ImmutableRowTableDirectedTransportPolicy::requireOriginal(endpoints.front(),endpoints,
                changedInventory,PeerAccessCoverage::None),std::invalid_argument);
            auto duplicateEndpoints = endpoints;
            duplicateEndpoints.push_back(endpoints.front());
            EXPECT_THROW(ImmutableRowTableDirectedTransportPolicy::requireOriginal(endpoints.front(),duplicateEndpoints,
                originalInventory,PeerAccessCoverage::None),std::invalid_argument);
            const std::array consumers{resources.front()};
            const auto original = ImmutableRowTablePlacementInput::forConsumers(
                ImmutableRowTableExtent::fromNative(GGUFTensorType::F32, 128, 64), observed(DeviceId::cpu()), consumers)
                .withGPUCacheResources(policy, resources).withGPUCachePeerAccess(PeerAccessCoverage::None);
            const std::array requests{ImmutableRowTableAdmissionRequest{policy, original, {}}};
            const auto wire = serializeImmutableRowTableAdmissionRequests(requests);
            const auto parsed = parseImmutableRowTableAdmissionRequests(wire, 0);
            ASSERT_EQ(parsed.size(), 1u);
            EXPECT_EQ(parsed.front().input.gpuCachePeerAccess(), PeerAccessCoverage::None);
            EXPECT_EQ(parsed.front().input.gpuResources().size(), count);
            EXPECT_EQ(parsed.front().input.consumers().size(), 1u);
            const std::vector<std::size_t> slots(count, 0);
            const auto layout = parsed.front().input.gpuAdaptiveLayout(policy, slots, 0, 16);
            EXPECT_EQ(layout.peerAccess(), PeerAccessCoverage::None);
            EXPECT_EQ(layout.consumers().size(), 1u);
            EXPECT_NO_THROW(original.withGPUCachePeerAccess(PeerAccessCoverage::Complete));
            EXPECT_THROW(original.withGPUCachePeerAccess(PeerAccessCoverage::Partial), std::invalid_argument);
            matrix[1] = true;
            EXPECT_THROW(ImmutableRowTableDirectedTransportPolicy::observe(policy.gpu_adaptive.placement, inventory), std::invalid_argument);
            matrix.assign(count * count, true);
            EXPECT_EQ(ImmutableRowTableDirectedTransportPolicy::observe(policy.gpu_adaptive.placement, inventory), PeerAccessCoverage::Complete);
            matrix.clear();
            EXPECT_THROW(ImmutableRowTableDirectedTransportPolicy::observe(policy.gpu_adaptive.placement, inventory), std::invalid_argument);
            inventory.rank = 1;
            EXPECT_THROW(ImmutableRowTableDirectedTransportPolicy::observe(policy.gpu_adaptive.placement, inventory), std::invalid_argument);
        }
    }
    const PhysicalMemoryAllocatorIdentity source{0, DeviceId::cuda(0)}, remote{0, DeviceId::cuda(1)};
    EXPECT_THROW(ImmutableRowTableDirectedTransportPolicy::require(source, remote, std::nullopt), std::invalid_argument);
    EXPECT_THROW(ImmutableRowTableDirectedTransportPolicy::require(source, {1, DeviceId::cuda(1)}, PeerAccessCoverage::None), std::invalid_argument);
    EXPECT_THROW(ImmutableRowTableDirectedTransportPolicy::require(source, {0, DeviceId::cpu()}, PeerAccessCoverage::None), std::invalid_argument);
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
        "0:rocm:0/1/1@1:cuda:0", "0:cuda:0/-1/1@0:cuda:0",
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

namespace llaminar2::test {
/** @test A non-first authored coordinator owns one source; a foreign reader owns only result/ACK roots. */
TEST(DeclaredGPUCachePlacement, SharedCoordinatorReadersPreserveOneSourceAndExactResultACKBOM)
{
    ImmutableRowTablePolicy policy{.storage=ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD};
    policy.gpu_adaptive.placement=parseAdaptiveGPUCachePlacement("0:rocm:2/64/1048576@0:rocm:1");
    policy.gpu_adaptive.host.maximum_cache_rows=16;
    policy.gpu_adaptive.host.cache_physical_byte_limit=1048576;
    policy.validate();
    const std::array devices{observed(DeviceId::rocm(0)),observed(DeviceId::rocm(1)),observed(DeviceId::rocm(2))};
    const std::array consumers{devices[1],devices[0]}; // Actual TP order is deliberately nonordinal.
    const auto host=observed(DeviceId::cpu());
    const std::array readerEdges{ImmutableRowTableReaderPeerAccess{
        {0,DeviceId::rocm(1)}, {0,DeviceId::rocm(0)}, PeerAccessCoverage::None}};
    const auto input=ImmutableRowTablePlacementInput::forConsumers(
        ImmutableRowTableExtent::fromNative(GGUFTensorType::F32,128,64),host,consumers)
        .withGPUCacheResources(policy,devices).withGPUCachePeerAccess(PeerAccessCoverage::None)
        .withReaderPeerAccess(readerEdges);
    const ImmutableRowTableSource source{.rows=128,.row_elements=16,.format=GGUFTensorType::F32};
    for (const auto storageCoverage : {PeerAccessCoverage::None, PeerAccessCoverage::Complete})
    for (const auto readerCoverage : {PeerAccessCoverage::None, PeerAccessCoverage::Complete})
    for(const size_t slots:{0u,64u}){
        const std::array owned{slots};
        auto selectedEdges = readerEdges;
        selectedEdges.front().coverage = readerCoverage;
        const auto selectedInput = input.withGPUCachePeerAccess(storageCoverage).withReaderPeerAccess(selectedEdges);
        const auto layout=selectedInput.gpuAdaptiveLayout(policy,owned,16,32);
        const auto geometry=AdaptiveImmutableLookupGeometry::fromLayout(layout,source,policy.gpu_adaptive);
        ASSERT_EQ(layout.consumers().size(),2u);
        ASSERT_TRUE(geometry.readerGeometry());
        const PhysicalMemoryAllocatorIdentity coordinator{0,DeviceId::rocm(1)},reader{0,DeviceId::rocm(0)};
        EXPECT_EQ(geometry.physical().coordinator(),coordinator);
        EXPECT_EQ(geometry.computeReaderCount(),2u);
        EXPECT_EQ(geometry.maximumIds(),32u); // Replicated sixteen-head original; never a two-consumer union.
        EXPECT_EQ(geometry.readerGeometry()->maximumOriginal(),32u);
        EXPECT_EQ(geometry.readerGeometry()->resultMaximum(),32u*16u*sizeof(float));
        EXPECT_EQ(geometry.gpuCacheBytes(reader),0u);
        EXPECT_EQ(geometry.mandatoryDeviceBytes(reader),0u);
        EXPECT_EQ(geometry.gpuSlotDependentBytes(reader),0u);
        EXPECT_GT(geometry.mandatoryDeviceBytes(coordinator),0u);
        const auto projected=AdaptiveImmutableLookupGeometry::sourceLayout(layout,policy.gpu_adaptive);
        ASSERT_EQ(projected.consumers().size(),1u);EXPECT_EQ(projected.consumers().front(),coordinator);
        EXPECT_EQ(projected.owners().size(),layout.owners().size());
        const auto single=AdaptiveImmutableLookupGeometry::fromLayout(projected,source,policy.gpu_adaptive);
        EXPECT_FALSE(single.readerGeometry());EXPECT_EQ(single.computeReaderCount(),1u);
        const std::array source_devices{devices[1],devices[2]};
        const auto baseline=recordedNgramMandatory(single,source_devices,host);
        const auto actual=recordedNgramMandatory(geometry,devices,host);
        const auto reader_bom=geometry.readerGeometry()->completeBOM(host,devices);
        PhysicalMemoryPlanBuilder expected(baseline);
        for(const auto& line:reader_bom.resources())expected.add(line);
        EXPECT_FALSE(actual.requiredFootprintMismatch(expected.build()));
        EXPECT_FALSE(expected.build().requiredFootprintMismatch(actual));
        EXPECT_EQ(actual.find({0,DeviceId::cpu()})->bytes(PhysicalMemoryOwner::ImmutableTableStorage),
            baseline.find({0,DeviceId::cpu()})->bytes(PhysicalMemoryOwner::ImmutableTableStorage));
        EXPECT_EQ(actual.find(reader)->bytes(PhysicalMemoryOwner::ImmutableLookupStaging),0u);
        EXPECT_EQ(actual.find(reader)->bytes(PhysicalMemoryOwner::ActivationTransportStaging),
            sizeof(qwen4exp::replicated_reader::Control)+geometry.readerGeometry()->resultMemory(reader).cursor_bytes_per_device+
            geometry.readerGeometry()->acknowledgementMemory(reader).cursor_bytes_per_device+
            geometry.readerGeometry()->acknowledgementMemory(reader).producer_payload_bytes);
        EXPECT_EQ(policy.lookupIOConsumer(reader.device,2),coordinator.device);
        const std::array requests{ImmutableRowTableAdmissionRequest{policy,selectedInput,{}}};
        const auto restored=parseImmutableRowTableAdmissionRequests(serializeImmutableRowTableAdmissionRequests(requests),0);
        ASSERT_EQ(restored.size(),1u);
        EXPECT_EQ(restored.front().input.consumers().size(),2u);
        EXPECT_EQ(restored.front().input.gpuResources().size(),3u);
        EXPECT_EQ(restored.front().requested,policy);
        EXPECT_EQ(restored.front().input.gpuAdaptiveLayout(policy,owned,16,32), layout);
        ASSERT_EQ(restored.front().input.readerPeerAccess().size(), 1u);
        EXPECT_EQ(restored.front().input.readerPeerAccess().front(), selectedEdges.front());
        const auto wire = serializeImmutableRowTableAdmissionRequests(requests);
        auto oldSchema = wire;
        oldSchema[4] = 14;
        EXPECT_THROW(parseImmutableRowTableAdmissionRequests(oldSchema,0), std::invalid_argument);
        std::vector<std::uint8_t> edgeWords;
        const auto word = [&](std::uint32_t value) {
            for (unsigned byte = 0; byte < 4; ++byte)
                edgeWords.push_back(static_cast<std::uint8_t>(value >> (byte * 8)));
        };
        for (const auto endpoint : {coordinator, reader}) {
            word(endpoint.world_rank);
            word(static_cast<std::uint32_t>(endpoint.device.type));
            word(endpoint.device.ordinal);
        }
        word(static_cast<std::uint32_t>(readerCoverage));
        const auto encoded = std::search(wire.begin(), wire.end(), edgeWords.begin(), edgeWords.end());
        ASSERT_NE(encoded, wire.end());
        ASSERT_EQ(std::search(encoded + edgeWords.size(), wire.end(), edgeWords.begin(), edgeWords.end()), wire.end());
        const auto offset = static_cast<std::size_t>(encoded - wire.begin());
        auto malformed = wire;
        malformed[offset + 24] = 255;
        EXPECT_THROW(parseImmutableRowTableAdmissionRequests(malformed,0), std::invalid_argument);
        malformed = wire;
        malformed[offset + 8] = 2; // Real storage endpoint is not the authored query/source coordinator.
        EXPECT_THROW(parseImmutableRowTableAdmissionRequests(malformed,0), std::invalid_argument);
    }
    auto foreign=policy;foreign.gpu_adaptive.placement.coordinator=AdaptiveGPUCacheParticipant{0,DeviceType::ROCm,3};
    EXPECT_THROW(input.withGPUCacheResources(foreign,devices),std::invalid_argument);
    EXPECT_THROW(input.withGPUCacheResources(policy,std::span{devices}.first(2)),std::invalid_argument);
}
}

namespace llaminar2::test {
/** @test Shared storage changes only the explicit placement arm of original ROCm TP2 source policies. */
TEST(DeclaredGPUCachePlacement, SharedCoordinatorTP2SourceSiblingsPreserveOriginalPolicies)
{
    using namespace parity;using namespace parity::qwen4exp;
    const auto& seeds=qwen4ExpSharedCoordinatorTP2AdaptiveSourceAdmissionSeeds();
    ASSERT_EQ(seeds.size(),2u);
    const auto expanded=qwen4ExpExpandSourceAdmissionMTP(seeds);
    ASSERT_EQ(expanded.size(),8u);
    std::array<size_t,4> modes{};
    for(const auto& cell:expanded){
        ASSERT_TRUE(cell.ngram_table_policy);
        const auto& placement=cell.ngram_table_policy->gpu_adaptive.placement;
        ASSERT_TRUE(placement.coordinator);ASSERT_EQ(placement.owners.size(),1u);
        EXPECT_EQ(placement.mode,AdaptiveGPUCachePlacementMode::DeclaredRankLocal);
        EXPECT_EQ(placement.coordinator->device(),DeviceId::rocm(0));
        EXPECT_EQ(placement.owners.front().participant.device(),DeviceId::rocm(2));
        ASSERT_TRUE(cell.topology.expert_overlay_plan);
        const auto& plan=*cell.topology.expert_overlay_plan;
        ASSERT_EQ(plan.domains.front().participants.size(),2u);
        EXPECT_EQ(plan.domains.front().participants[0].toLocalDeviceId(),DeviceId::rocm(0));
        EXPECT_EQ(plan.domains.front().participants[1].toLocalDeviceId(),DeviceId::rocm(1));
        EXPECT_EQ(plan.routed_tiers.front().max_experts_per_layer,8);
        EXPECT_EQ(cell.expert_overlay,std::optional{ModelParityExpertOverlayPolicy{}});
        switch(cell.mtp){
        case ModelParityMTP::Off:++modes[0];EXPECT_EQ(cell.retained_mtp_draft_capacity,0);break;
        case ModelParityMTP::Depth1:++modes[1];EXPECT_EQ(cell.retained_mtp_draft_capacity,15);break;
        case ModelParityMTP::Depth3:++modes[2];EXPECT_EQ(cell.retained_mtp_draft_capacity,15);break;
        case ModelParityMTP::DynamicDepth:++modes[3];EXPECT_EQ(cell.retained_mtp_draft_capacity,15);break;
        default:FAIL()<<"Shared source introduced an unauthored MTP mode";
        }
    }
    for(auto count:modes)EXPECT_EQ(count,2u);
}
}

namespace llaminar2::test {
/** @test The distinct CUDA source family preserves original model/math/policies and storage hardware roles. */
TEST(DeclaredGPUCachePlacement, SharedCoordinatorCUDATP2SourceSiblingsPreserveOriginalPoliciesAndStorageRole)
{
    using namespace parity;
    using namespace parity::qwen4exp;
    const auto rocmBefore=qwen4ExpExpandSourceAdmissionMTP(qwen4ExpSharedCoordinatorTP2AdaptiveSourceAdmissionSeeds());
    std::vector<std::string> rocmNames;
    for(const auto& cell:rocmBefore)rocmNames.push_back(cell.testName());
    const auto& seeds=qwen4ExpSharedCoordinatorCUDATP2AdaptiveSourceAdmissionSeeds();
    ASSERT_EQ(seeds.size(),2u);
    const auto expanded=qwen4ExpExpandSourceAdmissionMTP(seeds);
    ASSERT_EQ(expanded.size(),8u);
    std::array<std::size_t,4> modes{};
    std::set<std::string> names;
    for(const auto& cell:expanded){
        EXPECT_TRUE(names.insert(cell.testName()).second);
        EXPECT_EQ(cell.model.model_path,rocmBefore.front().model.model_path);
        EXPECT_EQ(cell.model.transformer_layers,rocmBefore.front().model.transformer_layers);
        ASSERT_TRUE(cell.ngram_table_policy);
        const auto reference=std::find_if(rocmBefore.begin(),rocmBefore.end(),[&](const auto& original){
            return original.mtp==cell.mtp && original.ngram_table_policy->gpu_adaptive.placement.owners.front().maximum_cache_rows==
                cell.ngram_table_policy->gpu_adaptive.placement.owners.front().maximum_cache_rows;
        });
        ASSERT_NE(reference,rocmBefore.end());
        EXPECT_EQ(cell.model.shared_predictor,reference->model.shared_predictor);
        EXPECT_EQ(cell.activation_precision,rocmBefore.front().activation_precision);
        EXPECT_EQ(cell.kv_cache_precision,rocmBefore.front().kv_cache_precision);
        EXPECT_EQ(cell.model.token_ids,rocmBefore.front().model.token_ids);
        ASSERT_TRUE(cell.ngram_table_policy);
        const auto& placement=cell.ngram_table_policy->gpu_adaptive.placement;
        ASSERT_TRUE(placement.coordinator);
        ASSERT_EQ(placement.owners.size(),1u);
        EXPECT_EQ(placement.coordinator->device(),DeviceId::cuda(0));
        EXPECT_EQ(placement.owners.front().participant.device(),DeviceId::rocm(0));
        EXPECT_TRUE(placement.owners.front().maximum_cache_rows==0 || placement.owners.front().maximum_cache_rows==64);
        ASSERT_TRUE(cell.topology.expert_overlay_plan);
        const auto& plan=*cell.topology.expert_overlay_plan;
        EXPECT_EQ(plan.domains.front().participants,(std::vector{GlobalDeviceAddress::cuda(0),GlobalDeviceAddress::cuda(1)}));
        EXPECT_EQ(plan.domains.back().participants,(std::vector{GlobalDeviceAddress::cpu(0)}));
        EXPECT_EQ(plan.routed_tiers.front().max_experts_per_layer,8);
        const auto config=cell.toTestConfig();
        ASSERT_TRUE(config.local_hardware_requirements);
        EXPECT_EQ(config.local_hardware_requirements->cuda_devices,2);
        EXPECT_EQ(config.local_hardware_requirements->rocm_devices,1);
        EXPECT_EQ(cell.expert_overlay,std::optional{ModelParityExpertOverlayPolicy{}});
        switch(cell.mtp){
        case ModelParityMTP::Off:++modes[0];EXPECT_EQ(cell.retained_mtp_draft_capacity,0);break;
        case ModelParityMTP::Depth1:++modes[1];EXPECT_EQ(cell.retained_mtp_draft_capacity,15);break;
        case ModelParityMTP::Depth3:++modes[2];EXPECT_EQ(cell.retained_mtp_draft_capacity,15);break;
        case ModelParityMTP::DynamicDepth:++modes[3];EXPECT_EQ(cell.retained_mtp_draft_capacity,15);break;
        default:FAIL()<<"CUDA shared source introduced an unauthored MTP mode";
        }
    }
    for(auto count:modes)EXPECT_EQ(count,2u);
    const auto rocmAfter=qwen4ExpExpandSourceAdmissionMTP(qwen4ExpSharedCoordinatorTP2AdaptiveSourceAdmissionSeeds());
    ASSERT_EQ(rocmAfter.size(),rocmNames.size());
    for(std::size_t i=0;i<rocmAfter.size();++i)EXPECT_EQ(rocmAfter[i].testName(),rocmNames[i]);
}
}
