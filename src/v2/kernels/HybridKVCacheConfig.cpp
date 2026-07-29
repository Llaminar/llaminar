/**
 * @file HybridKVCacheConfig.cpp
 * @brief Out-of-line methods for HybridGDNLayerState that require full kernel definitions
 */

#include "HybridKVCacheConfig.h"
#include "../tensors/TensorKernels.h"
#include "../utils/Logger.h"

#include <cstdio>
#include <cstring>
#include <unistd.h>

namespace llaminar2
{

    bool HybridGDNLayerState::resetGPUKernelState(void *stream)
    {
        const bool conv_ok = !conv_kernel || conv_kernel->resetGPUState(stream);
        const bool recurrence_ok = !rec_kernel || rec_kernel->resetGPUState(stream);
        if (!conv_ok || !recurrence_ok)
        {
            LOG_ERROR("[HybridGDNLayerState] Device-state reset rejected persistent ownership"
                      << " short_conv=" << (conv_ok ? "ready" : "rejected")
                      << " recurrence=" << (recurrence_ok ? "ready" : "rejected")
                      << " stream=" << stream);
            const char *message =
                !conv_ok && !recurrence_ok
                    ? "[FATAL] Hybrid GDN reset failed: "
                      "short_conv=rejected recurrence=rejected\n"
                    : (!conv_ok
                           ? "[FATAL] Hybrid GDN reset failed: "
                             "short_conv=rejected recurrence=ready\n"
                           : "[FATAL] Hybrid GDN reset failed: "
                             "short_conv=ready recurrence=rejected\n");
            (void)::write(STDERR_FILENO, message, std::strlen(message));
        }
        return conv_ok && recurrence_ok;
    }

} // namespace llaminar2
