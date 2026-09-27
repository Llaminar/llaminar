/**
 * @file Test__ROCmMoEKernelResources.cpp
 * @brief Prevent compiler scratch from returning to small MoE route grouping.
 *
 * The kernel always launches one 256-thread workgroup. An implicit 1024-thread
 * compiler bound previously forced 14 VGPR spills and 64 private bytes per
 * thread. Inspect all four actual loaded specializations. Symbol-enabled HIP
 * compilation may retain a printf ABI frame, so compare with a minimal compiled
 * diagnostic rather than calling that frame a spill or allowing magic bytes.
 * This is a structural invariant, not a timing threshold or a benchmark.
 */
#include "backends/GPUDeviceContextPool.h"
#include "kernels/rocm/moe/ROCmMoEKernelResources.h"

#include <gtest/gtest.h>

/** @brief Inspect the matching HIP compiler's diagnostic frame; no launch. */
bool rocmMoEDiagnosticFrameBytes(std::size_t &bytes);

namespace llaminar2::test
{
    /** @test Every small runtime grouping specialization remains scratch-free. */
    TEST(ROCmMoEKernelResources, RuntimeGroupingUsesRegistersWithoutPrivateScratch)
    {
        GPUDeviceContextPool::instance().getAMDContext(0).submitAndWait([]
        {
            std::size_t diagnostic_frame = 0;
            ASSERT_TRUE(rocmMoEDiagnosticFrameBytes(diagnostic_frame));
            RecordProperty("diagnostic_frame_bytes", static_cast<int>(diagnostic_frame));
            for (const auto variant : {
                     ROCmRuntimeGroupVariant::RouterRoutes,
                     ROCmRuntimeGroupVariant::AssignedRoutes,
                     ROCmRuntimeGroupVariant::RouterCompletePlan,
                     ROCmRuntimeGroupVariant::AssignedCompletePlan})
            {
                SCOPED_TRACE(static_cast<int>(variant));
                const auto resources = queryROCmRuntimeGroupResources(variant);
                ASSERT_TRUE(resources.has_value()) << "HIP resource query failed";
                EXPECT_EQ(resources->private_bytes_per_thread, diagnostic_frame);
                EXPECT_EQ(resources->launch_threads, 256);
                EXPECT_EQ(resources->maximum_threads_per_block, resources->launch_threads);
                EXPECT_GT(resources->registers_per_thread, 0);
                EXPECT_GT(resources->resident_blocks_per_multiprocessor, 0);
                RecordProperty("registers_" + std::to_string(static_cast<int>(variant)),
                               resources->registers_per_thread);
                RecordProperty("resident_blocks_" + std::to_string(static_cast<int>(variant)),
                               resources->resident_blocks_per_multiprocessor);
            }
            EXPECT_FALSE(queryROCmRuntimeGroupResources(
                static_cast<ROCmRuntimeGroupVariant>(-1)).has_value());
        });
    }
}
