/**
 * @file HybridKVCacheConfig.cpp
 * @brief Out-of-line methods for HybridGDNLayerState that require full kernel definitions
 */

#include "HybridKVCacheConfig.h"
#include "../tensors/TensorKernels.h"

namespace llaminar2
{

    void HybridGDNLayerState::resetGPUKernelState()
    {
        if (conv_kernel)
            conv_kernel->resetGPUState();
        if (rec_kernel)
            rec_kernel->resetGPUState();
    }

} // namespace llaminar2
