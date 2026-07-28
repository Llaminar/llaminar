/**
 * @file HybridKVCacheConfig.cpp
 * @brief Out-of-line methods for HybridGDNLayerState that require full kernel definitions
 */

#include "HybridKVCacheConfig.h"
#include "../tensors/TensorKernels.h"

namespace llaminar2
{

    bool HybridGDNLayerState::resetGPUKernelState(void *stream)
    {
        const bool conv_ok = !conv_kernel || conv_kernel->resetGPUState(stream);
        const bool recurrence_ok = !rec_kernel || rec_kernel->resetGPUState(stream);
        return conv_ok && recurrence_ok;
    }

} // namespace llaminar2
