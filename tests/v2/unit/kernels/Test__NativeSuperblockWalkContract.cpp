/**
 * @file Test__NativeSuperblockWalkContract.cpp
 * @brief Device-free admission and progress checks for physical superblock walks.
 */
#include <gtest/gtest.h>
#include "kernels/cpu/gemm/CPUNativeSuperblockWalkContract.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedPreparedFootprint.h"
using namespace llaminar2::cpu::native_quantised;
TEST(NativeSuperblockWalkContract, BoundedPartitionsProgress) {
    for (int blocks : {0, 1, 7, 8, 9, 31, INT_MAX})
        for (int phase = 0; phase < 8; ++phase)
            for (int partitions : {1, 3, 17, 37}) {
                const auto width = (std::int64_t(blocks) + partitions - 1) / partitions;
                for (int p = 0; p < partitions; ++p) {
                    const int first = std::min<std::int64_t>(p * width, blocks);
                    const int last = std::min<std::int64_t>((p + 1) * width, blocks);
                    requireSuperblockWalk(first, last, blocks, phase);
                    CPUNativeQuantisedBlockGeometry geometry{.blocks_per_row = std::size_t(blocks), .phase = std::size_t(phase)};
                    // Sample both ends, including the ragged final superblock without walking INT_MAX blocks.
                    for (int b : {first, last - 1}) if (b >= first && b < last)
                        EXPECT_GT(std::min<std::size_t>(last, geometry.superblockEnd(b)), std::size_t(b));
                }
            }
}
TEST(NativeSuperblockWalkContract, RejectsOriginalStuckRange) {
    EXPECT_DEATH(requireSuperblockWalk(8, 9, 8), "Invalid CPU superblock walk range");
    EXPECT_DEATH(requireSuperblockWalk(-1, 1, 8), "Invalid CPU superblock walk range");
    EXPECT_DEATH(requireSuperblockWalk(2, 1, 8), "Invalid CPU superblock walk range");
    EXPECT_DEATH(requireSuperblockWalk(0, 1, std::size_t(INT_MAX) + 1), "Invalid CPU superblock walk range");
    EXPECT_DEATH(requireSuperblockWalk(0, 1, 8, 8), "Invalid CPU superblock walk range");
}
