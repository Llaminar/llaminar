/**
 * @file CUDARingKVCacheTensorAdapter.cpp
 * @brief ITensor adapter for CUDA KV Cache
 *
 * This file is compiled by the regular C++ compiler (not nvcc) so it can
 * include heavy headers like CPUTensors.h without MPI header issues.
 *
 * Provides the ICUDARingKVCache::append(ITensor*) implementation that
 * extracts GPU pointers and delegates to the void* append method.
 */

#include "CUDARingKVCache.h"
#include "../../utils/Logger.h"

namespace llaminar2
{

    // =========================================================================
    // ICUDARingKVCache::append(ITensor*) implementation
    // =========================================================================
    // NOTE: This is in a separate .cpp file (not .cu) because nvcc has issues
    // with some C++ headers. The ITensor interface is lightweight and doesn't
    // require heavy includes.

    bool ICUDARingKVCache::append(int layer, int seq_idx,
                                  const ITensor *K, const ITensor *V,
                                  int num_tokens)
    {
        if (!K || !V)
        {
            LOG_DEBUG("[ICUDARingKVCache::append(ITensor)] Null K or V tensor");
            return false;
        }

        // Get GPU data pointers from tensors via ITensor interface
        const void *d_k = K->gpu_data_ptr();
        const void *d_v = V->gpu_data_ptr();

        if (!d_k || !d_v)
        {
            // Tensors don't have GPU data - caller should have called ensureOnDevice()
            LOG_ERROR("[ICUDARingKVCache::append(ITensor)] K or V tensor lacks GPU data. "
                      << "Call ensureOnDevice() before append.");
            return false;
        }

        // Delegate to the device pointer version (with default stream 0)
        return append(layer, seq_idx, d_k, d_v, num_tokens, 0);
    }

} // namespace llaminar2
