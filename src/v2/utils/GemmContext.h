/**
 * @file GemmContext.h
 * @brief Thread-local GEMM context for profiling granularity
 *
 * Provides a mechanism for compute stages to tag GEMM kernel invocations
 * with their functional purpose (attention, FFN, LM head). The GPU kernel
 * profilers (ROCmKernelProfiler, CUDAKernelProfiler) read this context
 * in their record() method to attribute GEMM time to the correct sub-type.
 *
 * Usage in stages:
 *   {
 *       ScopedGemmContext ctx(GemmContext::ATTN);
 *       gemm->multiply_tensor(...);  // Attributed to GEMM_ATTN
 *   }
 */
#pragma once

#include <cstdint>

namespace llaminar2
{

    /**
     * @brief Functional context for GEMM profiling attribution
     *
     * Set by compute stages before invoking GEMM kernels. The GPU profilers
     * remap the generic GEMM kernel type to a context-specific sub-type
     * (e.g., GEMM_ATTN, GEMM_FFN) based on this value.
     */
    enum class GemmContext : uint8_t
    {
        NONE = 0, ///< No context set (records as generic GEMM)
        ATTN,     ///< Attention projections (QKV, Wo)
        FFN,      ///< FFN projections (gate, up, down)
        LM_HEAD   ///< Language model head projection
    };

    /**
     * @brief Get/set the current thread-local GEMM context
     */
    inline GemmContext &currentGemmContext()
    {
        thread_local GemmContext ctx = GemmContext::NONE;
        return ctx;
    }

    /**
     * @brief RAII guard for setting GEMM context around kernel calls
     *
     * Sets the context on construction, clears it on destruction.
     * Zero overhead when profiling is disabled (the context is just
     * a thread-local enum that the profiler checks only when recording).
     */
    class ScopedGemmContext
    {
    public:
        explicit ScopedGemmContext(GemmContext ctx)
        {
            currentGemmContext() = ctx;
        }

        ~ScopedGemmContext()
        {
            currentGemmContext() = GemmContext::NONE;
        }

        ScopedGemmContext(const ScopedGemmContext &) = delete;
        ScopedGemmContext &operator=(const ScopedGemmContext &) = delete;
    };

} // namespace llaminar2
