/**
 * @file Test__MoEOverlayNodeLocalRouteTransport.cpp
 * @brief Device-free peer-topology and ExpertOverlay transport policy tests.
 *
 * These tests inject driver-shaped matrices and never enumerate or occupy a
 * physical accelerator.  They lock down the production rule that native
 * NCCL/RCCL remains authoritative whenever any peer edge exists, while a
 * proven no-P2P homogeneous cell uses mapped sparse routing for both phases.
 * Explicit diagnostics preserve that topology and decode/MTP default while
 * independently validating the ordinary-prefill transport support contract.
 */

#include "backends/ComputeBackend.h"
#include "execution/moe/MoEOverlayNodeLocalRouteExchange.h"
#include "fixtures/moe/NodeLocalRouteExchangeReplayAccess.h"
#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "planning/ObservedPeerGrouping.h"
#include "transfer/TransferEngine.h"
#include <array>
#include <type_traits>
#include "execution/moe/MoEOverlayNodeLocalRouteTransport.h"

#include <gtest/gtest.h>

#include <optional>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        /** Build one well-formed matrix with caller-controlled directed edges. */
        P2PMatrix matrix(
            ComputeBackendType backend,
            std::vector<int> device_ids,
            std::vector<std::vector<bool>> edges)
        {
            return P2PMatrix{
                .backend = backend,
                .device_ids = std::move(device_ids),
                .can_access = std::move(edges),
            };
        }
    } // namespace

    TEST(Test__MoEOverlayNodeLocalRouteTransport,
         ClassifiesOrdinalAddressedPeerCoverageExactly)
    {
        const auto complete = matrix(
            ComputeBackendType::GPU_CUDA,
            {4, 9},
            {{true, true}, {true, true}});
        EXPECT_EQ(
            complete.coverageForDevices({9, 4}),
            PeerAccessCoverage::Complete)
            << "Device ordinals, not matrix positions, define the domain";

        const auto none = matrix(
            ComputeBackendType::GPU_ROCM,
            {2, 5, 8},
            {{true, false, false},
             {false, true, false},
             {false, false, true}});
        EXPECT_EQ(
            none.coverageForDevices({2, 5, 8}),
            PeerAccessCoverage::None);

        const auto partial = matrix(
            ComputeBackendType::GPU_ROCM,
            {2, 5, 8},
            {{true, true, false},
             {true, true, false},
             {false, false, true}});
        EXPECT_EQ(
            partial.coverageForDevices({8, 2, 5}),
            PeerAccessCoverage::Partial);

        const auto asymmetric = matrix(
            ComputeBackendType::GPU_CUDA,
            {0, 1},
            {{true, true}, {false, true}});
        EXPECT_EQ(
            asymmetric.coverageForDevices({0, 1}),
            PeerAccessCoverage::Partial)
            << "One driver edge is still a native-library topology opportunity";
    }

    TEST(Test__MoEOverlayNodeLocalRouteTransport,
         RejectsIncompleteOrAmbiguousPeerMatrices)
    {
        const auto valid = matrix(
            ComputeBackendType::GPU_CUDA,
            {0, 1},
            {{true, false}, {false, true}});
        EXPECT_FALSE(valid.coverageForDevices({0}).has_value());
        EXPECT_FALSE(valid.coverageForDevices({0, 0}).has_value());
        EXPECT_FALSE(valid.coverageForDevices({0, 2}).has_value());

        const auto malformed = matrix(
            ComputeBackendType::GPU_CUDA,
            {0, 1},
            {{true}, {false, true}});
        EXPECT_FALSE(malformed.coverageForDevices({0, 1}).has_value());
    }

    TEST(Test__MoEOverlayNodeLocalRouteTransport,
         HomogeneousPolicyUsesMappedSparseOnlyWithoutP2P)
    {
        const std::vector<DeviceId> cuda_devices{
            DeviceId::cuda(0), DeviceId::cuda(1)};
        EXPECT_EQ(
            selectMoEOverlayNodeLocalRouteTransportPolicy(
                cuda_devices, PeerAccessCoverage::Complete),
            (MoEOverlayNodeLocalRouteTransportPolicy{
                .ordinary_prefill =
                    MoEOverlayNodeLocalRouteTransport::NativeCollective,
                .decode =
                    MoEOverlayNodeLocalRouteTransport::NativeCollective,
            }));
        EXPECT_EQ(
            selectMoEOverlayNodeLocalRouteTransportPolicy(
                cuda_devices, PeerAccessCoverage::Partial),
            (MoEOverlayNodeLocalRouteTransportPolicy{
                .ordinary_prefill =
                    MoEOverlayNodeLocalRouteTransport::NativeCollective,
                .decode =
                    MoEOverlayNodeLocalRouteTransport::NativeCollective,
            }));
        const auto no_p2p = selectMoEOverlayNodeLocalRouteTransportPolicy(
            cuda_devices, PeerAccessCoverage::None);
        EXPECT_EQ(
            no_p2p.ordinary_prefill,
            MoEOverlayNodeLocalRouteTransport::MappedSparse);
        EXPECT_EQ(
            no_p2p.decode,
            MoEOverlayNodeLocalRouteTransport::MappedSparse);
        EXPECT_EQ(
            no_p2p.forPhase(
                MoEOverlayNodeLocalRoutePhase::OrdinaryPrefill),
            MoEOverlayNodeLocalRouteTransport::MappedSparse);
        EXPECT_EQ(
            no_p2p.forPhase(MoEOverlayNodeLocalRoutePhase::Decode),
            MoEOverlayNodeLocalRouteTransport::MappedSparse);
        EXPECT_TRUE(no_p2p.resolved());
        EXPECT_TRUE(no_p2p.requiresMappedExchange());
    }

    TEST(Test__MoEOverlayNodeLocalRouteTransport,
         HeterogeneousCellsUseMappedSparseWithoutPretendingToHaveOneMatrix)
    {
        const std::vector<DeviceId> mixed_devices{
            DeviceId::cuda(3), DeviceId::rocm(1)};
        const auto mixed_policy =
            selectMoEOverlayNodeLocalRouteTransportPolicy(
                mixed_devices, std::nullopt);
        EXPECT_EQ(
            mixed_policy.ordinary_prefill,
            MoEOverlayNodeLocalRouteTransport::MappedSparse);
        EXPECT_EQ(
            mixed_policy.decode,
            MoEOverlayNodeLocalRouteTransport::MappedSparse);
        EXPECT_TRUE(mixed_policy.requiresMappedExchange());
        EXPECT_THROW(
            (void)selectMoEOverlayNodeLocalRouteTransportPolicy(
                mixed_devices, PeerAccessCoverage::None),
            std::invalid_argument);
    }

    TEST(Test__MoEOverlayNodeLocalRouteTransport,
         MissingOrInvalidTopologyCannotSilentlyBecomeAHostBounce)
    {
        const std::vector<DeviceId> cuda_devices{
            DeviceId::cuda(0), DeviceId::cuda(1)};
        EXPECT_THROW(
            (void)selectMoEOverlayNodeLocalRouteTransportPolicy(
                cuda_devices, std::nullopt),
            std::invalid_argument);
        EXPECT_THROW(
            (void)selectMoEOverlayNodeLocalRouteTransportPolicy(
                std::vector<DeviceId>{DeviceId::cuda(0)},
                PeerAccessCoverage::None),
            std::invalid_argument);
        EXPECT_THROW(
            (void)selectMoEOverlayNodeLocalRouteTransportPolicy(
                std::vector<DeviceId>{DeviceId::cpu(), DeviceId::cuda(0)},
                std::nullopt),
            std::invalid_argument);
        EXPECT_THROW(
            (void)selectMoEOverlayNodeLocalRouteTransportPolicy(
                std::vector<DeviceId>{DeviceId::cuda(0), DeviceId::cuda(0)},
                PeerAccessCoverage::Complete),
            std::invalid_argument);
    }

    TEST(Test__MoEOverlayNodeLocalRouteTransport,
         DiagnosticParsesOnlyExplicitCanonicalPrefillSelections)
    {
        EXPECT_FALSE(parseMoEOverlayDiagnosticPrefillTransport(nullptr));
        EXPECT_EQ(parseMoEOverlayDiagnosticPrefillTransport("native_collective"),
            MoEOverlayNodeLocalRouteTransport::NativeCollective);
        EXPECT_EQ(parseMoEOverlayDiagnosticPrefillTransport("mapped_sparse"),
            MoEOverlayNodeLocalRouteTransport::MappedSparse);
        for (const auto *invalid : {"", "auto", "mapped", "MAPPED_SPARSE", " mapped_sparse"})
            EXPECT_THROW((void)parseMoEOverlayDiagnosticPrefillTransport(invalid),
                std::invalid_argument);
    }

    TEST(Test__MoEOverlayNodeLocalRouteTransport,
         DiagnosticParsedStartupValueDoesNotBorrowEnvironmentStorage)
    {
        char original[] = "mapped_sparse";
        const auto frozen = parseMoEOverlayDiagnosticPrefillTransport(original);
        original[0] = 'x';
        EXPECT_EQ(frozen, MoEOverlayNodeLocalRouteTransport::MappedSparse);
        const std::vector<DeviceId> devices{DeviceId::rocm(2), DeviceId::rocm(5)};
        const auto policy = selectMoEOverlayDiagnosticPrefillTransportPolicy(
            devices, PeerAccessCoverage::Complete, frozen);
        EXPECT_EQ(policy.ordinary_prefill, MoEOverlayNodeLocalRouteTransport::MappedSparse);
        EXPECT_EQ(policy.decode, MoEOverlayNodeLocalRouteTransport::NativeCollective);
    }

    TEST(Test__MoEOverlayNodeLocalRouteTransport,
         DiagnosticMappedPrefillPreservesNativeDecodeAndActualPeerCoverage)
    {
        for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
        {
            const std::vector<DeviceId> devices = backend == DeviceType::CUDA
                ? std::vector<DeviceId>{DeviceId::cuda(4), DeviceId::cuda(9)}
                : std::vector<DeviceId>{DeviceId::rocm(2), DeviceId::rocm(5)};
            for (const auto coverage : {PeerAccessCoverage::Partial, PeerAccessCoverage::Complete})
            {
                const auto default_policy = selectMoEOverlayNodeLocalRouteTransportPolicy(devices, coverage);
                EXPECT_EQ(selectMoEOverlayDiagnosticPrefillTransportPolicy(devices, coverage, std::nullopt),
                    default_policy);
                const auto selected = selectMoEOverlayDiagnosticPrefillTransportPolicy(devices, coverage,
                    parseMoEOverlayDiagnosticPrefillTransport("mapped_sparse"));
                EXPECT_EQ(selected.ordinary_prefill, MoEOverlayNodeLocalRouteTransport::MappedSparse);
                EXPECT_EQ(selected.decode, default_policy.decode);
                EXPECT_EQ(selected.forPhase(MoEOverlayNodeLocalRoutePhase::Decode),
                    MoEOverlayNodeLocalRouteTransport::NativeCollective);
                EXPECT_TRUE(selected.requiresMappedExchange());
            }
        }
    }

    TEST(Test__MoEOverlayNodeLocalRouteTransport,
         DiagnosticNativeSupportDoesNotPretendNoPeerCoverageIsUnsupported)
    {
        const std::vector<DeviceId> devices{DeviceId::rocm(2), DeviceId::rocm(5)};
        const auto selected = selectMoEOverlayDiagnosticPrefillTransportPolicy(devices,
            PeerAccessCoverage::None, parseMoEOverlayDiagnosticPrefillTransport("native_collective"));
        EXPECT_EQ(selected.ordinary_prefill, MoEOverlayNodeLocalRouteTransport::NativeCollective);
        EXPECT_EQ(selected.decode, MoEOverlayNodeLocalRouteTransport::MappedSparse);
        EXPECT_TRUE(selected.requiresMappedExchange());
    }

    TEST(Test__MoEOverlayNodeLocalRouteTransport,
         DiagnosticAuxiliaryCPUAndWholeExpertRanksDoNotRequireLocalTP)
    {
        const std::vector<DeviceId> continuation{DeviceId::rocm(1), DeviceId::rocm(2)};
        // The CPU2 capacity rank and another whole-expert rank carry the same
        // original plan; neither owns the continuation's execution layout.
        const std::vector<DeviceId> cpu_auxiliary{DeviceId::cpu()};
        EXPECT_FALSE(validateMoEOverlayDiagnosticPrefillTransportOwner(
            cpu_auxiliary, continuation, /*rank_local_tp=*/false));
        const std::vector<DeviceId> gpu_auxiliary{DeviceId::rocm(3)};
        EXPECT_FALSE(validateMoEOverlayDiagnosticPrefillTransportOwner(
            gpu_auxiliary, continuation, /*rank_local_tp=*/false));
        const std::vector<DeviceId> reordered{DeviceId::rocm(2), DeviceId::rocm(1)};
        EXPECT_TRUE(validateMoEOverlayDiagnosticPrefillTransportOwner(
            reordered, continuation, /*rank_local_tp=*/true));
    }

    TEST(Test__MoEOverlayNodeLocalRouteTransport,
         DiagnosticActualContinuationOwnerRejectsPartialOrUnsupportedLayout)
    {
        const std::vector<DeviceId> continuation{DeviceId::rocm(1), DeviceId::rocm(2)};
        const std::vector<DeviceId> partial{DeviceId::rocm(1)};
        EXPECT_THROW((void)validateMoEOverlayDiagnosticPrefillTransportOwner(
            partial, continuation, /*rank_local_tp=*/true), std::invalid_argument);
        EXPECT_THROW((void)validateMoEOverlayDiagnosticPrefillTransportOwner(
            continuation, continuation, /*rank_local_tp=*/false), std::invalid_argument);
        const std::vector<DeviceId> mixed_owner{DeviceId::rocm(1), DeviceId::cpu()};
        EXPECT_THROW((void)validateMoEOverlayDiagnosticPrefillTransportOwner(
            mixed_owner, continuation, /*rank_local_tp=*/true), std::invalid_argument);
    }

    TEST(Test__MoEOverlayNodeLocalRouteTransport,
         DiagnosticRejectsNativeMixedVendorAndIncompleteTopology)
    {
        const std::vector<DeviceId> mixed{DeviceId::cuda(3), DeviceId::rocm(1)};
        EXPECT_THROW((void)selectMoEOverlayDiagnosticPrefillTransportPolicy(mixed, std::nullopt,
            parseMoEOverlayDiagnosticPrefillTransport("native_collective")), std::invalid_argument);
        EXPECT_NO_THROW((void)selectMoEOverlayDiagnosticPrefillTransportPolicy(mixed, std::nullopt,
            parseMoEOverlayDiagnosticPrefillTransport("mapped_sparse")));
        const std::vector<DeviceId> single{DeviceId::rocm(1)};
        EXPECT_THROW((void)selectMoEOverlayDiagnosticPrefillTransportPolicy(single, PeerAccessCoverage::Complete,
            parseMoEOverlayDiagnosticPrefillTransport("mapped_sparse")), std::invalid_argument);
        const std::vector<DeviceId> homogeneous{DeviceId::rocm(1), DeviceId::rocm(2)};
        EXPECT_THROW((void)selectMoEOverlayDiagnosticPrefillTransportPolicy(homogeneous, std::nullopt,
            parseMoEOverlayDiagnosticPrefillTransport("mapped_sparse")), std::invalid_argument);
    }

    TEST(Test__MoEOverlayNodeLocalRouteTransport,
         DensePublicationRequiresNoP2PAndEconomicalPayload)
    {
        constexpr std::size_t threshold = 4096u;

        static_assert(
            kMoEOverlayMappedDensePublicationMinimumBytes == 0u);
        EXPECT_FALSE(shouldUseMoEOverlayMappedDensePublication(
            MoEOverlayNodeLocalRouteTransport::MappedSparse,
            /*payload_bytes=*/64u * 1024u * 1024u));

        EXPECT_FALSE(shouldUseMoEOverlayMappedDensePublication(
            MoEOverlayNodeLocalRouteTransport::NativeCollective,
            threshold * 4u));
        EXPECT_FALSE(shouldUseMoEOverlayMappedDensePublication(
            MoEOverlayNodeLocalRouteTransport::MappedSparse,
            threshold - 1u));
        EXPECT_TRUE(shouldUseMoEOverlayMappedDensePublication(
            MoEOverlayNodeLocalRouteTransport::MappedSparse,
            threshold,
            threshold));
        EXPECT_TRUE(shouldUseMoEOverlayMappedDensePublication(
            MoEOverlayNodeLocalRouteTransport::MappedSparse,
            /*payload_bytes=*/4096u,
            /*minimum_payload_bytes=*/4096u));
        EXPECT_FALSE(shouldUseMoEOverlayMappedDensePublication(
            MoEOverlayNodeLocalRouteTransport::MappedSparse,
            /*payload_bytes=*/4096u,
            /*minimum_payload_bytes=*/0u));
    }
} // namespace llaminar2::test

namespace llaminar2::test {
namespace {
/** @brief Recorded complete publication; sparse ordinals never become matrix positions. */
std::shared_ptr<ClusterInventory> routeOriginalInventory(std::size_t count) {
    auto result=std::make_shared<ClusterInventory>();result->world_size=2;result->ranks.resize(2);
    for(int rank=0;rank<2;++rank){auto& original=result->ranks[rank];original.rank=rank;
        original.cpu.numa_node=rank?23:11;original.hostname="recorded-route-host";original.node_id=7;}
    auto& original=result->ranks[1];
    for(std::size_t index=0;index<count;++index){DeviceInfo gpu;gpu.type=DeviceType::ROCm;
        gpu.local_device_id=2+3*index;gpu.numa_node=23;gpu.uuid="route-original-"+std::to_string(index);
        original.gpus.push_back(gpu);}
    original.p2p_rocm_count=count;original.p2p_rocm.assign(count*count,false);
    return result;
}
/** @brief Actual recorded original endpoint order, independent of physical ordinal adjacency. */
MoEOverlayNodeLocalRouteExchange::Config routeConfig(const ClusterInventory& inventory) {
    MoEOverlayNodeLocalRouteExchange::Config config;
    for(const auto& gpu:inventory.ranks[1].gpus)config.devices.push_back({gpu.type,gpu.local_device_id});
    config.root_device=config.devices.back();config.identity="recorded-admitted-route";return config;
}
/** @brief Sole rank-bound declaration issuer; constructing it creates no device resource. */
std::shared_ptr<PhysicalMemoryAuthority> routeMemory(const MoEOverlayNodeLocalRouteExchange::StorageDeclaration& declaration,
    DeviceId root,int rank=1) {
    constexpr std::size_t capacity=std::size_t{1}<<40;
    auto bom=declaration.bom({rank,DeviceId::cpu(),capacity,capacity},{rank,root,capacity,capacity});
    return std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(std::move(bom)),rank);
}
}
static_assert(!std::is_constructible_v<MoEOverlayNodeLocalRouteExchange,MoEOverlayNodeLocalRouteExchange::Config>,
    "Production must not construct raw unadmitted route storage");

TEST(Test__MoEOverlayNodeLocalRouteTransport, AdmittedStorageHasExactRootAndSharedMappedGeometryWithoutCountCap) {
    for(const std::size_t participants:{2u,3u,5u,9u}) {
        const auto inventory=routeOriginalInventory(participants);const auto config=routeConfig(*inventory);
        for(const std::uint32_t rows:{1u,3u,65u}) {
            const auto declaration=MoEOverlayNodeLocalRouteExchange::declareStorage(config,rows,3,17);
            EXPECT_EQ(declaration.rootBytes(),(participants-1)*rows*3*17*sizeof(float));
            const auto bom=declaration.bom({1,DeviceId::cpu(),1ull<<40,1ull<<40},{1,config.root_device,1ull<<40,1ull<<40});
            ASSERT_EQ(bom.resources().size(),2u);
            const std::array requests{MappedHostTransferArena::AllocationRequest{declaration.mappingBytes(),4096}};
            for(const auto& resource:bom.resources()) {
                if(resource.resource().device.is_cpu())EXPECT_EQ(resource.bytes(PhysicalMemoryOwner::ActivationTransportStaging),
                    MappedHostTransferArena::backingBytes(requests));
                else {EXPECT_GE(resource.bytes(PhysicalMemoryOwner::ActivationTransportStaging),declaration.rootBytes());
                    EXPECT_LT(resource.bytes(PhysicalMemoryOwner::ActivationTransportStaging)-declaration.rootBytes(),256u);}
            }
            // This original constructor binds identity only; native creators are deferred to materialize.
            auto memory=routeMemory(declaration,config.root_device);
            EXPECT_NO_THROW((MoEOverlayNodeLocalRouteExchange(config,declaration,memory,inventory)));
        }
    }
}
TEST(Test__MoEOverlayNodeLocalRouteTransport, AdmittedStorageRejectsOriginalRankNUMAUUIDAndGeometryBeforeNativeCreation) {
    auto inventory=routeOriginalInventory(3);auto config=routeConfig(*inventory);
    const auto declaration=MoEOverlayNodeLocalRouteExchange::declareStorage(config,4,3,17);
    auto memory=routeMemory(declaration,config.root_device);
    EXPECT_THROW((MoEOverlayNodeLocalRouteExchange(config,declaration,nullptr,inventory)),std::invalid_argument);
    EXPECT_THROW((MoEOverlayNodeLocalRouteExchange(config,declaration,memory,nullptr)),std::invalid_argument);
    auto wrong_root=config;wrong_root.root_device=config.devices.front();
    EXPECT_THROW((MoEOverlayNodeLocalRouteExchange(wrong_root,declaration,memory,inventory)),std::invalid_argument);
    auto wrong_rank=routeMemory(declaration,config.root_device,0);
    EXPECT_THROW((MoEOverlayNodeLocalRouteExchange(config,declaration,wrong_rank,inventory)),std::invalid_argument);
    inventory->ranks[1].gpus[0].numa_node=11;
    EXPECT_THROW((MoEOverlayNodeLocalRouteExchange(config,declaration,memory,inventory)),std::invalid_argument);
    inventory->ranks[1].gpus[0].numa_node=23;inventory->ranks[1].gpus[0].uuid=inventory->ranks[1].gpus[1].uuid;
    EXPECT_THROW((MoEOverlayNodeLocalRouteExchange(config,declaration,memory,inventory)),std::invalid_argument);
    EXPECT_THROW((MoEOverlayNodeLocalRouteExchange::declareStorage(config,0,3,17)),std::invalid_argument);
    EXPECT_THROW((MoEOverlayNodeLocalRouteExchange::declareStorage(config,UINT32_MAX,2,17)),std::overflow_error);
    config.devices[1]=config.devices[0];
    EXPECT_THROW((MoEOverlayNodeLocalRouteExchange::declareStorage(config,4,3,17)),std::invalid_argument);
}
TEST(Test__MoEOverlayNodeLocalRouteTransport, AdmittedStorageReuseRequiresExactOriginalSourceAndCaptureIdentity) {
    auto inventory=routeOriginalInventory(3);const auto config=routeConfig(*inventory);
    const auto declaration=MoEOverlayNodeLocalRouteExchange::declareStorage(config,4,3,17);
    auto memory=routeMemory(declaration,config.root_device);
    MoEOverlayNodeLocalRouteExchange exchange(config,declaration,memory,inventory);
    EXPECT_NO_THROW(exchange.requireOriginalAdmission(declaration,memory,inventory));
    auto raw=NodeLocalRouteExchangeReplayAccess::createUnique(config);
    EXPECT_THROW(raw->requireOriginalAdmission(declaration,memory,inventory),std::logic_error);
    const auto changed=MoEOverlayNodeLocalRouteExchange::declareStorage(config,3,3,17);
    EXPECT_THROW(exchange.requireOriginalAdmission(changed,memory,inventory),std::logic_error);
    auto same_values_new_inventory=std::make_shared<ClusterInventory>(*inventory);
    EXPECT_THROW(exchange.requireOriginalAdmission(declaration,memory,same_values_new_inventory),std::logic_error);
    auto independent=routeMemory(declaration,config.root_device);
    EXPECT_THROW(exchange.requireOriginalAdmission(declaration,independent,inventory),std::logic_error);
    auto domain=config;domain.identity+="-other";
    const auto other_domain=MoEOverlayNodeLocalRouteExchange::declareStorage(domain,4,3,17);
    EXPECT_THROW(exchange.requireOriginalAdmission(other_domain,memory,inventory),std::logic_error);
}
TEST(Test__MoEOverlayNodeLocalRouteTransport, OriginalDirectedPeerCoveragePreservesNativeAuthorityAndRejectsMissingFacts) {
    auto inventory=routeOriginalInventory(5);const auto config=routeConfig(*inventory);
    auto& original=inventory->ranks[1];
    EXPECT_EQ(ObservedPeerGrouping(original).coverage(config.devices),PeerAccessCoverage::None);
    original.p2p_rocm[1]=true;
    EXPECT_EQ(ObservedPeerGrouping(original).coverage(config.devices),PeerAccessCoverage::Partial);
    EXPECT_EQ(selectMoEOverlayNodeLocalRouteTransportPolicy(config.devices,ObservedPeerGrouping(original).coverage(config.devices)).decode,
        MoEOverlayNodeLocalRouteTransport::NativeCollective);
    std::fill(original.p2p_rocm.begin(),original.p2p_rocm.end(),true);
    EXPECT_EQ(ObservedPeerGrouping(original).coverage(config.devices),PeerAccessCoverage::Complete);
    original.p2p_rocm.pop_back();EXPECT_THROW((ObservedPeerGrouping(original).coverage(config.devices)),std::invalid_argument);
    original.p2p_rocm.clear();EXPECT_THROW((ObservedPeerGrouping(original).coverage(config.devices)),std::invalid_argument);
}
TEST(Test__MoEOverlayNodeLocalRouteTransport, OriginalBoundPolicyIsOneAuthorityForQuoteAndRuntime) {
    auto inventory=routeOriginalInventory(3);const auto config=routeConfig(*inventory);
    MoERoutedExpertPlacementPlan requested;requested.enabled=true;requested.continuation_domain="main";
    requested.base_model_domain="main";requested.shared_expert_domain="main";requested.continuation_domain_spec.domain="main";
    RoutedExpertDomain domain;domain.name="main";domain.scope=ExecutionDomainScope::RANK_LOCAL;domain.owner_rank=1;
    for(const auto device:config.devices){domain.participants.push_back(GlobalDeviceAddress::rocm(device.ordinal,23,"recorded-route-host"));
        domain.world_ranks.push_back(1);}
    requested.domains={domain};requested.routed_tiers={{.name="main",.domain="main"}};
    const auto none=bindMoEExpertOverlayPlanToClusterInventory(requested,*inventory);
    ASSERT_TRUE(none->observed_route_transport);EXPECT_TRUE(none->observed_route_transport->requiresMappedExchange());
    inventory->ranks[1].p2p_rocm[1]=true;
    EXPECT_THROW((bindMoEExpertOverlayPlanToClusterInventory(*none,*inventory)),std::invalid_argument);
    const auto partial=bindMoEExpertOverlayPlanToClusterInventory(requested,*inventory);
    ASSERT_TRUE(partial->observed_route_transport);EXPECT_FALSE(partial->observed_route_transport->requiresMappedExchange());
    const auto diagnostic=bindMoEExpertOverlayPlanToClusterInventory(requested,*inventory,MoEOverlayNodeLocalRouteTransport::MappedSparse);
    ASSERT_TRUE(diagnostic->observed_route_transport);
    EXPECT_EQ(diagnostic->observed_route_transport->ordinary_prefill,MoEOverlayNodeLocalRouteTransport::MappedSparse);
    EXPECT_EQ(diagnostic->observed_route_transport->decode,MoEOverlayNodeLocalRouteTransport::NativeCollective);
    EXPECT_EQ(diagnostic->observed_route_diagnostic,MoEOverlayNodeLocalRouteTransport::MappedSparse);
}
}
