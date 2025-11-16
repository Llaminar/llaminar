/**
 * @file Perf__VNNIGemmRegistry.cpp
 * @brief Test VNNI GEMM kernel registry initialization
 * @author David Sanftenberg
 */

#include "kernels/cpu/gemm_v3/VNNIGemmKernelRegistry.h"
#include <iostream>

using namespace llaminar2;

// Force-link all instantiation shards (defined in VNNIGemmKernelInit.cpp)
namespace llaminar2
{
    void forceLink_VNNIGemmKernelRegistry();
}

int main()
{
    std::cout << "VNNI GEMM Registry Test\n";
    std::cout << "=======================\n\n";

    // Force instantiation linking
    llaminar2::forceLink_VNNIGemmKernelRegistry();

    // Get registry instance
    auto &registry = VNNIGemmKernelRegistry::instance();

    // Check registry size
    size_t num_kernels = registry.size();
    std::cout << "Number of registered VNNI GEMM kernels: " << num_kernels << "\n";

    if (num_kernels == 432)
    {
        std::cout << "✅ SUCCESS: Registry contains all 432 expected kernel configurations!\n";
        return 0;
    }
    else
    {
        std::cout << "❌ FAILURE: Expected 432 kernels, got " << num_kernels << "\n";
        return 1;
    }
}
