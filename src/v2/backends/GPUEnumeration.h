/**
 * @file GPUEnumeration.h
 * @brief Declarations for GPU device enumeration (separate compilation units)
 *
 * This header declares functions implemented in separate CUDA/ROCm compilation units
 * to avoid header conflicts when both backends are enabled.
 *
 * @author David Sanftenberg
 */

#pragma once

#include <vector>
#include "ComputeBackend.h"

namespace llaminar2
{

// CUDA enumeration (implemented in CUDAEnumeration.cu)
#ifdef HAVE_CUDA
    namespace cuda_enumeration
    {
        std::vector<ComputeDevice> enumerate_cuda_devices();
        int get_cuda_device_numa_node(int device_id);
    }
#endif

// ROCm enumeration (implemented in ROCmEnumeration.cpp)
#ifdef HAVE_ROCM
    namespace rocm_enumeration
    {
        std::vector<ComputeDevice> enumerate_rocm_devices();
        int get_rocm_device_numa_node(int device_id);
    }
#endif

} // namespace llaminar2
