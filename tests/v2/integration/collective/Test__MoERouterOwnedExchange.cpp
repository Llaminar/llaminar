/**
 * @file Test__MoERouterOwnedExchange.cpp
 * @brief Captured real-GPU router rows, byte extents and replay lifetime regression.
 *
 * Local all-format arithmetic coverage lives in MoERouterOwnedRows. This gate
 * adds actual peer transfers, disjoint original row ownership and complete
 * downstream publication, including counts that leave some peers empty.
 */
#include "utils/MoERouterOwnedExchangeFixture.h"
#include <gtest/gtest.h>

namespace
{
    /** @brief Exercise every small count and production-size bucket transitions. */
    void prove(llaminar2::DeviceId first, int participants)
    {
        std::vector<int> counts{65, 0, 1, 64, 2};
        for (int rows = 0; rows <= 65; ++rows) counts.push_back(rows);
        counts.insert(counts.end(), {65, 1, 0, 33, 65});
        llaminar2::test::proveMoERouterOwnedExchange(first, participants, 65, 96, 17, counts);
        llaminar2::test::proveMoERouterOwnedExchange(first, participants, 512, 2048, 256,
            {512, 448, 447, 449, 1, 0, 512, 64, 65, 2, 0, 448});
    }
}
#ifdef HAVE_CUDA
/** @test Native CUDA arithmetic with exact no-P2P pair transport. */
TEST(MoERouterOwnedExchange, CUDA2) { prove(llaminar2::DeviceId::cuda(0), 2); }
#endif
#ifdef HAVE_ROCM
/** @test Native HIP arithmetic and full hidden-Q8 side product on two GPUs. */
TEST(MoERouterOwnedExchange, ROCm2) { prove(llaminar2::DeviceId::rocm(0), 2); }
/** @test More empty/unequal sources and a larger directed exchange fabric. */
TEST(MoERouterOwnedExchange, ROCm4) { prove(llaminar2::DeviceId::rocm(0), 4); }
#endif

#ifdef HAVE_CUDA
/** @test Original local NCCL composition with arbitrary counts, exact payload bytes and retained reset/replay. */
TEST(NativeCountedAllGather, CUDA2){llaminar2::test::proveNativeCountedAllGather(llaminar2::DeviceId::cuda(0),2);}
#endif
#ifdef HAVE_ROCM
/** @test Actual original local RCCL composition, including all-empty and maximally skewed counts. */
TEST(NativeCountedAllGather, ROCm2){llaminar2::test::proveNativeCountedAllGather(llaminar2::DeviceId::rocm(0),2);}
/** @test Four original logical participants without pair-only indexing. */
TEST(NativeCountedAllGather, ROCm4){llaminar2::test::proveNativeCountedAllGather(llaminar2::DeviceId::rocm(0),4);}
/** @test Eight original logical participants without a compile-time participant cap. */
TEST(NativeCountedAllGather, ROCm8){llaminar2::test::proveNativeCountedAllGather(llaminar2::DeviceId::rocm(0),8);}
#endif
