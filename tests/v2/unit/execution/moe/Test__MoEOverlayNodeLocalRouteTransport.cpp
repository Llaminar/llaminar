/**
 * @file Test__MoEOverlayNodeLocalRouteTransport.cpp
 * @brief Device-free peer-topology and ExpertOverlay transport policy tests.
 *
 * These tests inject driver-shaped matrices and never enumerate or occupy a
 * physical accelerator.  They lock down the production rule that native
 * NCCL/RCCL remains authoritative whenever any peer edge exists, while the
 * mapped sparse protocol is reserved for proven no-P2P or heterogeneous cells.
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
         NativeCollectivesWinWheneverHomogeneousP2PExists)
    {
        const std::vector<DeviceId> cuda_devices{
            DeviceId::cuda(0), DeviceId::cuda(1)};
        EXPECT_EQ(
            selectMoEOverlayNodeLocalRouteTransport(
                cuda_devices, PeerAccessCoverage::Complete),
            MoEOverlayNodeLocalRouteTransport::NativeCollective);
        EXPECT_EQ(
            selectMoEOverlayNodeLocalRouteTransport(
                cuda_devices, PeerAccessCoverage::Partial),
            MoEOverlayNodeLocalRouteTransport::NativeCollective);
        EXPECT_EQ(
            selectMoEOverlayNodeLocalRouteTransport(
                cuda_devices, PeerAccessCoverage::None),
            MoEOverlayNodeLocalRouteTransport::MappedSparse);
    }

    TEST(Test__MoEOverlayNodeLocalRouteTransport,
         HeterogeneousCellsUseMappedSparseWithoutPretendingToHaveOneMatrix)
    {
        const std::vector<DeviceId> mixed_devices{
            DeviceId::cuda(3), DeviceId::rocm(1)};
        EXPECT_EQ(
            selectMoEOverlayNodeLocalRouteTransport(
                mixed_devices, std::nullopt),
            MoEOverlayNodeLocalRouteTransport::MappedSparse);
        EXPECT_THROW(
            (void)selectMoEOverlayNodeLocalRouteTransport(
                mixed_devices, PeerAccessCoverage::None),
            std::invalid_argument);
    }

    TEST(Test__MoEOverlayNodeLocalRouteTransport,
         MissingOrInvalidTopologyCannotSilentlyBecomeAHostBounce)
    {
        const std::vector<DeviceId> cuda_devices{
            DeviceId::cuda(0), DeviceId::cuda(1)};
        EXPECT_THROW(
            (void)selectMoEOverlayNodeLocalRouteTransport(
                cuda_devices, std::nullopt),
            std::invalid_argument);
        EXPECT_THROW(
            (void)selectMoEOverlayNodeLocalRouteTransport(
                std::vector<DeviceId>{DeviceId::cuda(0)},
                PeerAccessCoverage::None),
            std::invalid_argument);
        EXPECT_THROW(
            (void)selectMoEOverlayNodeLocalRouteTransport(
                std::vector<DeviceId>{DeviceId::cpu(), DeviceId::cuda(0)},
                std::nullopt),
            std::invalid_argument);
        EXPECT_THROW(
            (void)selectMoEOverlayNodeLocalRouteTransport(
                std::vector<DeviceId>{DeviceId::cuda(0), DeviceId::cuda(0)},
                PeerAccessCoverage::Complete),
            std::invalid_argument);
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
