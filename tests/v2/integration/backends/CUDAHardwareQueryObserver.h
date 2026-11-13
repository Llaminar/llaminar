/**
 * @file CUDAHardwareQueryObserver.h
 * @brief Shared host observation interface for CUDA discovery-call authentication.
 *
 * This header contains no runtime declarations or GPU code. Host ELF observers
 * live in a C++ translation unit, while the native replay kernel remains CUDA.
 * Both static and shared core calls reach the executable's exported observers.
 */
#pragma once
#include <array>

namespace llaminar2::test::cuda_metadata_native
{
    /**
     * @brief Read actual forwarded count/property/PCI query totals, in that order.
     * @return Atomic observation only; never invokes or initializes a CUDA runtime.
     */
    std::array<unsigned, 3> queryCounts() noexcept;
}
