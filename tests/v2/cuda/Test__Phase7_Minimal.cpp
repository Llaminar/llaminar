#include <gtest/gtest.h>
#include "kernels/cuda/CudaGemmKernelPhase7_CUTLASS.h"
#include <vector>
#include <iostream>

using namespace llaminar::v2;

TEST(Phase7Minimal, JustCheckCompiles) {
    std::cout << "Test is running\n";
    EXPECT_TRUE(true);
}
