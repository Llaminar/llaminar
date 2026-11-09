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
