/**
 * @file GPUDeviceContextPool.h
 * @brief Singleton pool managing all GPU device contexts
 *
 * Provides lazy initialization and thread-safe access to GPU device contexts.
 * Contexts are created on first access and persist for the lifetime of the process.
 *
 * This class is DECOUPLED from CUDA/HIP dependencies - it only knows about
 * IWorkerGPUContext (the abstract interface). Concrete context creation is
 * delegated to factory functions registered by cuda_backend and rocm_backend.
 *
 * Usage:
 *   // Get NVIDIA context for device 0
 *   auto& ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);
 *
 *   // Get context by device type string
 *   auto& ctx = GPUDeviceContextPool::instance().getContext("cuda", 0);
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "IWorkerGPUContext.h"
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <functional>

namespace llaminar2
{

    /**
     * @brief Factory function type for creating GPU device contexts
     *
     * Factory functions are registered by cuda_backend and rocm_backend libraries
     * to create concrete NvidiaDeviceContext / AMDDeviceContext instances.
     *
     * @param device_ordinal GPU device index (0, 1, 2, ...)
     * @return Unique pointer to the created context (as IWorkerGPUContext)
     */
    using GPUContextFactory = std::function<std::unique_ptr<IWorkerGPUContext>(int device_ordinal)>;

    /**
     * @brief Singleton pool managing all GPU device contexts
     *
     * Provides lazy initialization and thread-safe access to device contexts.
     * Contexts are created on first access and persist for the lifetime of the process.
     *
     * ## Backend Decoupling
     *
     * This class does NOT directly depend on CUDA or HIP headers. Instead, it uses
     * factory functions registered by the cuda_backend and rocm_backend libraries
     * during their static initialization. This allows GPUDeviceContextPool to live
     * in llaminar2_core without pulling in GPU-specific headers.
     *
     * ## Thread Safety
     *
     * All public methods are thread-safe. Context creation is protected by a mutex
     * to ensure only one context per device is created even under concurrent access.
     *
     * ## Lazy Initialization
     *
     * Contexts are not created until first requested via getNvidiaContext(),
     * getAMDContext(), or getContext(). This avoids unnecessary GPU initialization
     * overhead for unused devices.
     *
     * ## Lifetime
     *
     * The pool is a process-global singleton. Contexts persist until shutdown()
     * is called or the process exits. The destructor calls shutdown() automatically.
     */
    class GPUDeviceContextPool
    {
    public:
        /**
         * @brief Get the singleton instance
         * @return Reference to the global pool instance
         * @thread_safety Thread-safe (uses Meyers singleton with static local)
         */
        static GPUDeviceContextPool &instance();

        // Non-copyable, non-movable
        GPUDeviceContextPool(const GPUDeviceContextPool &) = delete;
        GPUDeviceContextPool &operator=(const GPUDeviceContextPool &) = delete;
        GPUDeviceContextPool(GPUDeviceContextPool &&) = delete;
        GPUDeviceContextPool &operator=(GPUDeviceContextPool &&) = delete;

        // =========================================================================
        // Factory Registration (called by cuda_backend / rocm_backend)
        // =========================================================================

        /**
         * @brief Register factory for NVIDIA (CUDA) context creation
         *
         * Called by cuda_backend during static initialization to register the
         * NvidiaDeviceContext factory function.
         *
         * @param factory Factory function that creates NvidiaDeviceContext instances
         * @param device_count Number of available NVIDIA devices
         * @thread_safety Thread-safe
         */
        void registerNvidiaFactory(GPUContextFactory factory, int device_count);

        /**
         * @brief Register factory for AMD (ROCm) context creation
         *
         * Called by rocm_backend during static initialization to register the
         * AMDDeviceContext factory function.
         *
         * @param factory Factory function that creates AMDDeviceContext instances
         * @param device_count Number of available AMD devices
         * @thread_safety Thread-safe
         */
        void registerAMDFactory(GPUContextFactory factory, int device_count);

        // =========================================================================
        // Context Access (thread-safe, lazy initialization)
        // =========================================================================

        /**
         * @brief Get context for NVIDIA device by ordinal
         *
         * Creates context lazily if not exists. The context is initialized on
         * first access, which includes creating the CUDA primary context,
         * worker thread, and library handles (cuBLAS, etc.).
         *
         * @param device_ordinal CUDA device index (0, 1, 2, ...)
         * @return Reference to the device context
         * @throws std::runtime_error if CUDA support is not available
         * @throws std::runtime_error if device_ordinal is invalid
         * @thread_safety Thread-safe, concurrent calls for same device return same context
         */
        IWorkerGPUContext &getNvidiaContext(int device_ordinal);

        /**
         * @brief Get context for AMD device by ordinal
         *
         * Creates context lazily if not exists. The context is initialized on
         * first access, which includes creating the HIP context, worker thread,
         * and library handles (hipBLAS, etc.).
         *
         * @param device_ordinal HIP device index (0, 1, 2, ...)
         * @return Reference to the device context
         * @throws std::runtime_error if ROCm support is not available
         * @throws std::runtime_error if device_ordinal is invalid
         * @thread_safety Thread-safe, concurrent calls for same device return same context
         */
        IWorkerGPUContext &getAMDContext(int device_ordinal);

        /**
         * @brief Get context by device type string (auto-detects vendor)
         *
         * Convenience method that parses the device type string and delegates
         * to the appropriate vendor-specific getter.
         *
         * @param device_type Device type: "cuda", "CUDA", "rocm", "ROCm", "hip", "HIP"
         * @param device_ordinal Device index (0, 1, 2, ...)
         * @return Reference to the device context
         * @throws std::invalid_argument if device_type is not recognized
         * @throws std::runtime_error if the requested backend is not available
         * @thread_safety Thread-safe
         */
        IWorkerGPUContext &getContext(const std::string &device_type, int device_ordinal);

        // =========================================================================
        // Availability Queries (thread-safe)
        // =========================================================================

        /**
         * @brief Check if NVIDIA (CUDA) contexts are available
         *
         * Returns true if the NVIDIA factory has been registered (by cuda_backend)
         * and at least one NVIDIA GPU is available.
         *
         * @return true if NVIDIA contexts can be created
         * @thread_safety Thread-safe
         */
        bool hasNvidiaSupport() const;

        /**
         * @brief Check if AMD (ROCm) contexts are available
         *
         * Returns true if the AMD factory has been registered (by rocm_backend)
         * and at least one AMD GPU is available.
         *
         * @return true if AMD contexts can be created
         * @thread_safety Thread-safe
         */
        bool hasAMDSupport() const;

        /**
         * @brief Get the number of available NVIDIA devices
         * @return Number of CUDA devices, or 0 if CUDA not available
         * @thread_safety Thread-safe
         */
        int nvidiaDeviceCount() const;

        /**
         * @brief Get the number of available AMD devices
         * @return Number of ROCm devices, or 0 if ROCm not available
         * @thread_safety Thread-safe
         */
        int amdDeviceCount() const;

        // =========================================================================
        // Lifecycle Management
        // =========================================================================

        /**
         * @brief Shutdown all contexts and release resources
         *
         * Stops all worker threads and destroys all contexts. After calling this,
         * the pool can be used again (contexts will be re-created lazily).
         *
         * This is primarily useful for testing or explicit cleanup before
         * process exit.
         *
         * @thread_safety Thread-safe, but callers must ensure no other threads
         *                are using contexts during shutdown
         */
        void shutdown();

    private:
        GPUDeviceContextPool() = default;
        ~GPUDeviceContextPool();

        // Mutex protecting all mutable state
        mutable std::mutex mutex_;

        // Device contexts - keyed by device ordinal
        // Using IWorkerGPUContext for full type erasure (no CUDA/HIP deps in header)
        std::unordered_map<int, std::unique_ptr<IWorkerGPUContext>> nvidia_contexts_;
        std::unordered_map<int, std::unique_ptr<IWorkerGPUContext>> amd_contexts_;

        // Factory functions registered by cuda_backend / rocm_backend
        GPUContextFactory nvidia_factory_;
        GPUContextFactory amd_factory_;

        // Backend availability (set when factories are registered)
        int cuda_device_count_ = 0;
        int rocm_device_count_ = 0;
    };

} // namespace llaminar2
