/**
 * @file SharedExpertColumnGateKernels.h
 * @brief Backend-neutral shared-gate epilogue for disjoint output-column owners.
 *
 * Each participant holds complete normalized input and the same FP32 gate
 * vector, but only its reduced shared/routed output columns. The gate uses
 * the unchanged backend full-row reduction, then a separately rounded multiply
 * and add as in the existing TP shared-gate/residual stages. Narrowing ownership
 * never changes the dot product, intermediate rounding or padding contract.
 */
#pragma once
#include "backends/DeviceId.h"

namespace llaminar2
{
    /** @brief Immutable launch geometry; all output rows are contiguous local slices. */
    struct SharedExpertColumnGate
    {
        const float *input = nullptr; ///< Full [rows,model_columns] normalized input.
        const float *gate = nullptr; ///< Prepared full model-width FP32 vector.
        float *shared = nullptr; ///< Reduced local columns; overwritten by the gated contribution.
        const float *routed = nullptr; ///< Ordered routed sum for the same columns.
        float *combined = nullptr; ///< Disjoint destination for routed + gated shared.
        int rows = 0;
        int model_columns = 0;
        int local_columns = 0;
        const int *active_rows = nullptr; ///< Optional device-owned prefix length; tail becomes zero.
    };

    /**
     * @brief Enqueue the backend's canonical shared gate over a local column shard.
     * @param device Exact GPU participant; CPU projection ownership is not admitted.
     * @param binding Complete resident operand geometry; no implicit allocation/coherence.
     * @param stream Exact non-null producer stream.
     * @return False for invalid ranges, unsupported backend or an enqueue failure.
     *
     * Every written region must be disjoint from the other operands. The stage
     * publishes the two outputs after enqueue; this kernel interface owns no
     * tensor lifecycle, stream, memory ledger or synchronization.
     */
    bool gateSharedExpertColumnsFP32(DeviceId device, const SharedExpertColumnGate &binding, void *stream);
}
