#pragma once

/**
 * @file KernelRegistry.h
 * @brief Public-facing alias for the slimmed KernelFactory (Phase 8).
 *
 * After Phase 8, KernelFactory is reduced to a **KernelRegistry** role:
 * - Device-scoped kernel selection (RoPE, RMSNorm, SwiGLU, attention, etc.)
 * - Non-weight kernel caches (device-local, not model-weight-lifetime)
 * - GEMM engine creation helpers
 *
 * Model-weight-lifetime ownership (prepared GEMM handles, embedding handles,
 * fused caches, sliced caches) now lives in PreparedWeightStore, owned by
 * the ModelContext.
 *
 * New code should use `KernelRegistry` for device-scoped kernel access.
 * `KernelFactory` remains available for backward compatibility.
 */

#include "KernelFactory.h"

namespace llaminar::v2::kernels
{
    /// Phase 8 alias: KernelFactory with model-weight-lifetime stripped out.
    /// Use this for device-scoped kernel selection (RoPE, RMSNorm, SwiGLU, etc.)
    using KernelRegistry = KernelFactory;
} // namespace llaminar::v2::kernels

namespace llaminar2
{
    /// Convenience alias at top-level namespace
    using KernelRegistry = llaminar::v2::kernels::KernelRegistry;
} // namespace llaminar2
