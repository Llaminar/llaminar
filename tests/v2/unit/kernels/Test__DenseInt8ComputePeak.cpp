/**
 * @file Test__DenseInt8ComputePeak.cpp
 * @brief Device-free replay of dense INT8 throughput denominators.
 *
 * Published device configurations independently witness each admitted rate.
 * The local 3090 observation exposes the former 2304-ops sm_86 overcount;
 * A100/H100 controls prevent confusing FP16 capacity with INT8 capacity.
 */
#include "performance/kernels/native_quantised_dispatch/DenseInt8ComputePeak.h"
#include <gtest/gtest.h>

namespace
{
using namespace llaminar2::test::compute_peak;

TEST(DenseInt8ComputePeak, PublishedDenseConfigurationsAndNativeClockReplay)
{
    // Published whitepaper configurations, with sparse throughput excluded.
    EXPECT_DOUBLE_EQ(denseInt8Tops(72, 1770000, cudaDenseInt8OpsPerSmClock(75)), 260.99712);
    EXPECT_DOUBLE_EQ(denseInt8Tops(108, 1410000, cudaDenseInt8OpsPerSmClock(80)), 623.73888);
    EXPECT_DOUBLE_EQ(denseInt8Tops(82, 1695000, cudaDenseInt8OpsPerSmClock(86)), 284.65152);
    EXPECT_DOUBLE_EQ(denseInt8Tops(128, 2520000, cudaDenseInt8OpsPerSmClock(89)), 660.60288);
    EXPECT_DOUBLE_EQ(denseInt8Tops(132, 1830000, cudaDenseInt8OpsPerSmClock(90)), 1978.85952);
    // Replay the actual native clock attribute; NVML's maximum bin is separate.
    EXPECT_DOUBLE_EQ(denseInt8Tops(82, 1860000, cudaDenseInt8OpsPerSmClock(86)), 312.36096);
    EXPECT_DOUBLE_EQ(denseInt8Tops(96, 2482000, 512), 121.995264);
    // gfx1100 HIP reports WGPs; each contains two physical CUs. Preserve units.
    EXPECT_DOUBLE_EQ(denseInt8Tops(48, 2482000, 1024), denseInt8Tops(96, 2482000, 512));
}

TEST(DenseInt8ComputePeak, IncompleteOrUnqualifiedInputsCannotInventPeak)
{
    for (int capability : {0, 74, 79, 81, 87, 91, 100})
        EXPECT_EQ(cudaDenseInt8OpsPerSmClock(capability), 0u);
    EXPECT_EQ(denseInt8Tops(0, 1860000, 2048), 0.0);
    EXPECT_EQ(denseInt8Tops(82, 0, 2048), 0.0);
    EXPECT_EQ(denseInt8Tops(-1, 1860000, 2048), 0.0);
    EXPECT_EQ(denseInt8Tops(82, -1, 2048), 0.0);
    EXPECT_EQ(denseInt8Tops(82, 1860000, 0), 0.0);
}
}
