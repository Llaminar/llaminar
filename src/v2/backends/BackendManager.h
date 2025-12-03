/**
 * @file BackendManager.h
 * @brief Global GPU backend accessor (Phase 1 GPU Device-Aware Slicing)
 *
 * **Purpose**: Provides a singleton-style accessor for the GPU backend (CUDA/ROCm).
 * Avoids passing IBackend* through every layer of the call stack.
 *
 * **Thread Safety**: Initialization is thread-safe via call_once.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "IBackend.h"

namespace llaminar2
{

    /**
     * @brief Get the global GPU backend instance
     *
     * Returns:
     * - CUDABackend* if HAVE_CUDA is defined
     * - ROCmBackend* if HAVE_ROCM is defined
     * - nullptr if CPU-only build
     *
     * @return IBackend* or nullptr
     *
     * @note First call initializes the backend (lazy init)
     * @note Thread-safe via std::call_once
     */
    IBackend *getGPUBackend();

    /**
     * @brief Check if GPU backend is available
     * @return true if getGPUBackend() != nullptr
     */
    bool hasGPUBackend();

} // namespace llaminar2
