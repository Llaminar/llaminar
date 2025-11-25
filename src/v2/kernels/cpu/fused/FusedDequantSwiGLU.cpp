/**
 * @file FusedDequantSwiGLU.cpp
 * @brief Implementation of SwiGLU activation for FP32 inputs
 * @author David Sanftenberg
 * @date 2025-11-23
 * @updated 2025-11-24 - Reworked for FP32 residual architecture (no dequant needed)
 */

#include "FusedDequantSwiGLU.h"
#include "../../../utils/Logger.h"
#include "../primitives/SwiGLUPrimitives.h"

namespace llaminar2
{
    bool FusedSwiGLU::execute(
        const float *gate,
        const float *up,
        float *output,
        int m, int n)
    {
        if (!gate || !up || !output)
        {
            LOG_ERROR("[FusedSwiGLU] Null pointer in execute()");
            return false;
        }

        if (m <= 0 || n <= 0)
        {
            LOG_ERROR("[FusedSwiGLU] Invalid dimensions: m=" << m << " n=" << n);
            return false;
        }

        // Apply SwiGLU using well-tested primitives
        // SwiGLU: output = gate * silu(up)
        primitives::compute_swiglu(gate, up, output, static_cast<size_t>(m) * static_cast<size_t>(n));

        return true;
    }

} // namespace llaminar2
