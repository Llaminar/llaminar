/**
 * @file Test__GPUGraphKernelMetadata.cpp
 * @brief Device-free validation of complete captured kernel resource records.
 *
 * Empty HIP kernels can report zero VGPRs. The actual native ACK cancellation
 * replay exercises that compiler result; these interface tests prove accepting
 * zero registers does not accept missing function, geometry or launch limits.
 * No accelerator runtime, model or native context is needed here.
 */
#include "backends/IGPUGraphCapture.h"
#include <gtest/gtest.h>
#include <array>

namespace llaminar2::test
{
namespace
{
/** @return A synthetic complete launch record with normalized function identity. */
GPUGraphKernelNodeInfo completeRecord()
{
    return {
        .name = "unresolved_kernel",
        .graph_path = "root/node[0]",
        .function_identity = 1,
        .grid_x = 1,
        .grid_y = 1,
        .grid_z = 1,
        .block_x = 1,
        .block_y = 1,
        .block_z = 1,
        .registers_per_thread = 0,
        .max_threads_per_block = 256,
        .max_active_blocks_per_sm = 32,
    };
}
}

/** @test Zero compiler-assigned registers retain a complete launchable record. */
TEST(GPUGraphKernelMetadata, ZeroRegisterKernelRetainsCompleteLaunchContract)
{
    auto record = completeRecord();
    for (const auto registers : {0u, 1u, 16u, 40u, 0u})
    {
        record.registers_per_thread = registers;
        EXPECT_TRUE(record.valid());
    }
    EXPECT_FALSE(record.name_resolved);
    EXPECT_EQ(record.local_memory_bytes_per_thread, 0u);
}

/** @test Missing geometry/resources remain invalid across repeated record reset. */
TEST(GPUGraphKernelMetadata, IncompleteRecordsRejectAfterRepeatedReset)
{
    constexpr std::array required{
        &GPUGraphKernelNodeInfo::grid_x,
        &GPUGraphKernelNodeInfo::grid_y,
        &GPUGraphKernelNodeInfo::grid_z,
        &GPUGraphKernelNodeInfo::block_x,
        &GPUGraphKernelNodeInfo::block_y,
        &GPUGraphKernelNodeInfo::block_z,
        &GPUGraphKernelNodeInfo::max_threads_per_block,
        &GPUGraphKernelNodeInfo::max_active_blocks_per_sm,
    };
    for (int replay = 0; replay < 20; ++replay)
    {
        auto record = completeRecord();
        ASSERT_TRUE(record.valid());
        record.function_identity = 0;
        EXPECT_FALSE(record.valid());
        for (const auto field : required)
        {
            record = completeRecord();
            record.*field = 0;
            EXPECT_FALSE(record.valid());
        }
        record = {};
        EXPECT_FALSE(record.valid());
        record = completeRecord();
        EXPECT_TRUE(record.valid());
    }
}
}
