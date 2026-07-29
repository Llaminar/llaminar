/**
 * @file TurboQuantKVParallelPolicy.h
 * @brief Shared CPU work-partition policy for TurboQuant KV hot loops.
 *
 * KV publication and materialization operate on independent `(row, head)`
 * vectors. This header centralizes the measured crossover at which those
 * vectors provide enough work to amortize one static-scheduled OpenMP region.
 * Keeping the policy independent of either implementation prevents quantize
 * and dequantize helpers from acquiring a circular or accidental dependency.
 */

#pragma once

namespace llaminar2
{
    /**
     * @brief Minimum independent K/V head-pair count for OpenMP execution.
     *
     * Below this boundary, production helpers execute ordinary direct loops
     * and never enter the OpenMP runtime. The value is validated by the
     * ProductionFusedKVMatrix performance suite under both AVX2 and AVX-512.
     */
    inline constexpr int kTurboQuantKVParallelWorkItems = 32;

} // namespace llaminar2
