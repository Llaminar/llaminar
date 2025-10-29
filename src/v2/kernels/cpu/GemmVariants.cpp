/**
 * @file GemmVariants.cpp
 * @brief Registration wrapper for microkernel-based GEMM variants
 *
 * This file now uses the MicroKernelRegistry system with 1,225 pre-compiled
 * template instantiations across 64 translation units. This replaces the
 * old runtime template generation approach.
 *
 * @author David Sanftenberg
 * @date October 2025
 */

#include "GemmVariants.h"
#include "GemmMicroKernelAdapter.h"
#include "SmartGemmSearch.h"

namespace llaminar2
{
    using llaminar::v2::kernels::IBlockDecoder;
    using llaminar::v2::kernels::IQuantizedGemmVariant;

    /**
     * @brief Register all GEMM variants for auto-tuning
     *
     * This now uses the MicroKernelRegistry which provides:
     * - 1,225 pre-compiled template instantiations
     * - ISA: AVX512Tag, AVX2Tag
     * - MR (tile rows): {1, 2, 4, 8, 16}
     * - NR (tile cols): {1, 2, 4, 6, 8, 16, 32, 64}
     * - UNROLL_K: {1, 2, 4, 8}
     * - PREFETCH_DIST: {0, 1, 2}
     * - MC/KC/NC: Default blocking parameters
     *
     * The auto-tuner will benchmark a subset and cache the optimal
     * configuration for each matrix shape.
     */
    std::vector<std::unique_ptr<IQuantizedGemmVariant>> registerAllGemmVariants(
        const IBlockDecoder *decoder)
    {
        // Use the new microkernel registry system
        return kernels::gemm::registerMicroKernelVariants(decoder);
    }

} // namespace llaminar2
