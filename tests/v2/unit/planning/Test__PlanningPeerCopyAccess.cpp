/**
 * @file Test__PlanningPeerCopyAccess.cpp
 * @brief Device-free directed copy eligibility from original physical ownership and matrix positions.
 *
 * These synthetic RankInventory records exercise a pure projection contract;
 * they are not observations of available hardware. No MPI, driver, worker or
 * accelerator is initialized. Sparse ordinals and nonmatching rank/NUMA IDs
 * prevent index arithmetic from impersonating discovery or execution ownership.
 */
#include "planning/PlanningPeerCopyAccess.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <string>

using namespace llaminar2;
namespace
{
    /** @return Documented synthetic physical facts, with discovery order independent of ordinal order. */
    RankInventory recordedRank(DeviceType backend, size_t count)
    {
        RankInventory rank;
        rank.rank = 17;
        rank.node_id = 4;
        rank.local_rank = 2;
        rank.hostname = "synthetic-peer-access-owner";
        rank.cpu.type = DeviceType::CPU;
        rank.cpu.numa_node = 11;
        for (size_t index = 0; index < count; ++index)
        {
            DeviceInfo gpu;
            gpu.type = backend;
            gpu.local_device_id = static_cast<int>(5 + 7 * (count - index));
            gpu.numa_node = 11;
            gpu.uuid = "synthetic-physical-gpu-" + std::to_string(index);
            rank.gpus.push_back(gpu);
        }
        auto &matrix = backend == DeviceType::CUDA ? rank.p2p_cuda : rank.p2p_rocm;
        auto &matrix_count = backend == DeviceType::CUDA ? rank.p2p_cuda_count : rank.p2p_rocm_count;
        matrix_count = static_cast<int>(count);
        matrix.assign(count * count, false);
        return rank;
    }

    /** @return Original discovery order, without renumbering sparse physical ordinals. */
    std::vector<DeviceId> devicesOf(const RankInventory &rank)
    {
        std::vector<DeviceId> result;
        for (const auto &gpu : rank.gpus) result.emplace_back(gpu.type, gpu.local_device_id);
        return result;
    }
}

/** @test Source-to-destination movement requires destination-to-source read access, not its transpose. */
TEST(PlanningPeerCopyAccess, DirectedCopyUsesDestinationAccessToSource)
{
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        auto rank = recordedRank(backend, 3);
        auto &matrix = backend == DeviceType::CUDA ? rank.p2p_cuda : rank.p2p_rocm;
        matrix[1 * 3 + 0] = true; // Discovery endpoint1 can read endpoint0; reverse is absent.
        const auto original = devicesOf(rank);
        const std::array selected{original[1], original[0]};
        const auto eligibility = planningDirectedPeerCopyAccess(rank, selected);
        EXPECT_EQ(eligibility, (std::vector<std::uint8_t>{0, 0, 1, 0}));
        EXPECT_EQ(rank.rank, 17); // Rank identity does not imply NUMA17 or ordinal17.
        EXPECT_EQ(rank.cpu.numa_node, 11);
        EXPECT_TRUE(matrix[3]);
        EXPECT_FALSE(matrix[1]);
        // A repeated pure projection preserves every original observation.
        EXPECT_EQ(planningDirectedPeerCopyAccess(rank, selected), eligibility);
    }
}

/** @test Count follows the actual group, including four/eight and odd widths, without a count allowlist. */
TEST(PlanningPeerCopyAccess, SparseReorderedFourEightAndOddGroupsRetainDiscoveryCoordinates)
{
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
        for (const size_t count : {3u, 4u, 5u, 8u, 9u})
        {
            auto rank = recordedRank(backend, count);
            auto &matrix = backend == DeviceType::CUDA ? rank.p2p_cuda : rank.p2p_rocm;
            for (size_t row = 0; row < count; ++row)
                for (size_t column = 0; column < count; ++column)
                    matrix[row * count + column] = (row + 2 * column) % 3 == 0;
            auto selected = devicesOf(rank);
            std::reverse(selected.begin(), selected.end());
            const auto result = planningDirectedPeerCopyAccess(rank, selected);
            ASSERT_EQ(result.size(), count * count);
            for (size_t source = 0; source < count; ++source)
                for (size_t destination = 0; destination < count; ++destination)
                {
                    const size_t discovery_source = count - source - 1;
                    const size_t discovery_destination = count - destination - 1;
                    EXPECT_EQ(result[source * count + destination], source != destination &&
                        matrix[discovery_destination * count + discovery_source]);
                }
        }
}

/** @test Matrix positions are backend-specific even when another vendor is interleaved in discovery. */
TEST(PlanningPeerCopyAccess, OtherVendorVisibilityNeverShiftsNativeMatrixPositions)
{
    auto rank = recordedRank(DeviceType::ROCm, 3);
    auto selected = devicesOf(rank);
    DeviceInfo foreign;
    foreign.type = DeviceType::CUDA;
    foreign.local_device_id = 101;
    foreign.numa_node = 11;
    foreign.uuid = "synthetic-other-vendor";
    rank.gpus.insert(rank.gpus.begin() + 1, foreign);
    rank.p2p_rocm[2 * 3 + 0] = true;
    const auto result = planningDirectedPeerCopyAccess(rank, selected);
    EXPECT_EQ(result[0 * 3 + 2], 1u);
    EXPECT_EQ(result[2 * 3 + 0], 0u);
    selected[1] = DeviceId::cuda(101);
    EXPECT_THROW(planningDirectedPeerCopyAccess(rank, selected), std::invalid_argument);
}

/** @test Missing physical identity, malformed directed matrices and refused ownership cannot authorize copying. */
TEST(PlanningPeerCopyAccess, RejectsMissingMalformedAndForeignOriginalFacts)
{
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
        for (int defect = 0; defect < 13; ++defect)
        {
            auto rank = recordedRank(backend, 4);
            auto selected = devicesOf(rank);
            auto &matrix = backend == DeviceType::CUDA ? rank.p2p_cuda : rank.p2p_rocm;
            auto &count = backend == DeviceType::CUDA ? rank.p2p_cuda_count : rank.p2p_rocm_count;
            switch (defect)
            {
            case 0: matrix.clear(); break;
            case 1: matrix.pop_back(); break;
            case 2: count = 3; break;
            case 3: count = -1; break;
            case 4: rank.gpus.front().uuid.clear(); break;
            case 5: rank.gpus.front().numa_node = 23; break;
            case 6: rank.gpus.front().numa_node = -1; break;
            case 7: rank.cpu.numa_node = -1; break;
            case 8: rank.cpu.numa_node = 17; break; // Matching rank ID grants no NUMA11 ownership.
            case 9: selected.front() = DeviceId(backend, 999); break;
            case 10: selected.back() = selected.front(); break;
            case 11: rank.gpus.push_back(rank.gpus.front()); break;
            case 12: selected.clear(); break;
            }
            SCOPED_TRACE(defect);
            EXPECT_THROW(planningDirectedPeerCopyAccess(rank, selected), std::invalid_argument);
        }
    auto rank = recordedRank(DeviceType::CUDA, 4);
    const std::array singleton{DeviceId::cuda(rank.gpus.front().local_device_id)};
    EXPECT_THROW(planningDirectedPeerCopyAccess(rank, singleton), std::invalid_argument);
}
