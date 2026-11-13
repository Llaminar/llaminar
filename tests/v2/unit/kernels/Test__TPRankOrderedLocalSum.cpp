/**
 * @file Test__TPRankOrderedLocalSum.cpp
 * @brief Device-free admission for local-aware fixed-rank FP32 fold operands.
 *
 * These tests call the same checked immutable descriptor and exact-stream guard
 * used by the CUDA/HIP launchers. They do not dereference synthetic addresses,
 * initialize a backend, mirror device state, or assert native arithmetic. Actual
 * captured native-versus-local fold byte gates remain explicit GPU integrations.
 */
#include "kernels/common/TPRankOrderedReductionKernels.h"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <limits>

using namespace llaminar2;

/** @test Every communicator index retains the original explicit operand identity. */
TEST(TPRankOrderedLocalSum, OriginalRangesAndRankIndexAreImmutable)
{
    std::array<float, 33> local{}, output{};
    std::array<float, 264> peers{};
    for (const int degree : {2, 3, 4, 8})
        for (int rank = 0; rank < degree; ++rank)
            for (const std::size_t elements : {1u, 7u, 32u, 33u})
                for (const bool in_place : {false, true})
                {
                    auto *destination = in_place ? local.data() : output.data();
                    const TPRankOrderedLocalSumFP32 launch({local.data(), local.size(), rank},
                        peers.data(), peers.size(), 33, destination, 33, elements, degree);
                    EXPECT_EQ(launch.local().data, local.data());
                    EXPECT_EQ(launch.local().participant_index, rank);
                    EXPECT_EQ(launch.local().capacity_elements, local.size());
                    EXPECT_EQ(launch.peerBanks(), peers.data());
                    EXPECT_EQ(launch.bankStride(), 33u);
                    EXPECT_EQ(launch.output(), destination);
                    EXPECT_EQ(launch.elements(), elements);
                    EXPECT_EQ(launch.rankCount(), degree);
                }
}

/** @test No incomplete geometry or aliases can reach native device selection. */
TEST(TPRankOrderedLocalSum, RejectsGeometryBoundsAndAllPartialAliases)
{
    std::array<float, 256> storage{};
    auto *local = storage.data(), *peers = local + 64, *out = local + 224;
    const auto make = [&](const float *source, std::size_t source_capacity, int rank,
        const float *banks, std::size_t banks_capacity, std::size_t stride,
        float *output, std::size_t output_capacity, std::size_t elements, int degree) {
        return TPRankOrderedLocalSumFP32({source, source_capacity, rank}, banks,
            banks_capacity, stride, output, output_capacity, elements, degree);
    };
    EXPECT_NO_THROW((void)make(local, 32, 0, peers, 128, 32, out, 32, 32, 4));
    EXPECT_NO_THROW((void)make(local, 32, 3, peers, 128, 32, local, 32, 32, 4));
    EXPECT_THROW((void)make(nullptr, 32, 0, peers, 128, 32, out, 32, 32, 4), std::invalid_argument);
    EXPECT_THROW((void)make(local, 32, 0, nullptr, 128, 32, out, 32, 32, 4), std::invalid_argument);
    EXPECT_THROW((void)make(local, 32, 0, peers, 128, 32, nullptr, 32, 32, 4), std::invalid_argument);
    EXPECT_THROW((void)make(local, 32, -1, peers, 128, 32, out, 32, 32, 4), std::invalid_argument);
    EXPECT_THROW((void)make(local, 32, 4, peers, 128, 32, out, 32, 32, 4), std::invalid_argument);
    EXPECT_THROW((void)make(local, 32, 0, peers, 128, 32, out, 32, 32, 1), std::invalid_argument);
    EXPECT_THROW((void)make(local, 32, 0, peers, 128, 32, out, 32, 0, 4), std::invalid_argument);
    EXPECT_THROW((void)make(local, 31, 0, peers, 128, 32, out, 32, 32, 4), std::invalid_argument);
    EXPECT_THROW((void)make(local, 32, 0, peers, 128, 32, out, 31, 32, 4), std::invalid_argument);
    EXPECT_THROW((void)make(local, 32, 0, peers, 127, 32, out, 32, 32, 4), std::invalid_argument);
    EXPECT_THROW((void)make(local, 32, 0, peers, 128, 31, out, 32, 32, 4), std::invalid_argument);
    EXPECT_THROW((void)make(local + 60, 32, 0, peers, 128, 32, out, 32, 32, 4), std::invalid_argument);
    EXPECT_THROW((void)make(local, 32, 0, peers, 128, 32, peers + 127, 32, 32, 4), std::invalid_argument);
    EXPECT_THROW((void)make(local, 32, 0, peers, 128, 32, local + 1, 32, 32, 4), std::invalid_argument);
    EXPECT_THROW((void)make(local + 1, 32, 0, peers, 128, 32, local, 32, 32, 4), std::invalid_argument);
    // Equal pointers with unequal declared ranges are not exact alias identity.
    EXPECT_THROW((void)make(local, 32, 0, peers, 128, 32, local, 33, 32, 4), std::invalid_argument);
    const auto *misaligned = reinterpret_cast<const float *>(reinterpret_cast<std::uintptr_t>(local) + 1);
    EXPECT_THROW((void)make(misaligned, 32, 0, peers, 128, 32, out, 32, 32, 4), std::invalid_argument);
    EXPECT_THROW((void)make(local, 32, 0, misaligned, 128, 32, out, 32, 32, 4), std::invalid_argument);
    EXPECT_THROW((void)make(local, 32, 0, peers, 128, 32, const_cast<float *>(misaligned), 32, 32, 4), std::invalid_argument);
}

/** @test Every byte-span and multiplication overflow rejects without reading storage. */
TEST(TPRankOrderedLocalSum, RejectsUnrepresentableWholeAllocationRanges)
{
    std::array<float, 8> local{}, out{}, peers{};
    const auto maximum = std::numeric_limits<std::size_t>::max();
    EXPECT_THROW((void)TPRankOrderedLocalSumFP32({local.data(), 8, 0}, peers.data(), maximum,
        maximum / 2 + 1, out.data(), 8, 1, 2), std::overflow_error);
    EXPECT_THROW((void)TPRankOrderedLocalSumFP32({local.data(), maximum, 0}, peers.data(), 8,
        4, out.data(), 8, 1, 2), std::overflow_error);
    EXPECT_THROW((void)TPRankOrderedLocalSumFP32({local.data(), 8, 0}, peers.data(), maximum,
        4, out.data(), 8, 1, 2), std::overflow_error);
    EXPECT_THROW((void)TPRankOrderedLocalSumFP32({local.data(), 8, 0}, peers.data(), 8,
        4, out.data(), maximum, 1, 2), std::overflow_error);
    const auto end = std::numeric_limits<std::uintptr_t>::max() & ~std::uintptr_t{3};
    EXPECT_THROW((void)TPRankOrderedLocalSumFP32({reinterpret_cast<const float *>(end), 2, 0},
        peers.data(), 8, 4, out.data(), 8, 1, 2), std::overflow_error);
}

/** @test Exact-stream guard rejects both vendors' default sentinels before all native APIs. */
TEST(TPRankOrderedLocalSum, RejectsDefaultStreamsBeforeNativeDeviceSelection)
{
    for (const std::uintptr_t handle : {0u, 1u, 2u})
        for (const int ordinal : {-1, 0, 17})
            EXPECT_FALSE(tpRankOrderedLocalLaunchValid(ordinal, reinterpret_cast<void *>(handle)));
    EXPECT_FALSE(tpRankOrderedLocalLaunchValid(-1, reinterpret_cast<void *>(0x1000)));
    EXPECT_TRUE(tpRankOrderedLocalLaunchValid(0, reinterpret_cast<void *>(0x1000)));
    EXPECT_TRUE(tpRankOrderedLocalLaunchValid(17, reinterpret_cast<void *>(0x1000)));
}
