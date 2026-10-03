/**
 * @file MoEGroupedFloatingPrefillKernels.h
 * @brief Shared host/device ABI for exact floating MoE projection phases.
 *
 * The source width is an arithmetic identity; a down slice's physical width
 * is not another model. Gate/up produces existing FP32 SwiGLU workspace, and
 * down consumes those exact bits with the fixed serial reduction tree. Each
 * phase borrows persistent device storage and the caller's exact non-null
 * stream. Transport, allocation, grouping and lifecycle publication belong to
 * their explicit graph owners, never these launch bridges.
 */
#pragma once

#include "DeviceMoEFloatingMatrixDesc.h"
#include "MoEPrefillProjectionExecution.h"

extern "C"
{
#ifdef HAVE_CUDA
    /**
     * @brief Enqueue the declared CUDA phase without touching absent operands.
     *
     * Hidden/gate/up are required only for gate/up. Down descriptors, weights
     * and an output are required only for down. Both phases require the same
     * route map, format and FP32 intermediate. A down descriptor contains only
     * execution.columnCount() rows, at the declared original source interval.
     * @return False for incomplete bindings, invalid geometry or launch failure.
     */
    bool cudaMoE_grouped_floating_prefill_pipeline(
        const float *hidden,
        const llaminar2::DeviceMoEFloatingMatrixDesc *gate_descs,
        const llaminar2::DeviceMoEFloatingMatrixDesc *up_descs,
        const llaminar2::DeviceMoEFloatingMatrixDesc *down_descs,
        const int *original_to_grouped, const int *original_expert_ids,
        const float *grouped_weights, float *grouped_swiglu,
        float *output, float *canonical_routes,
        int seq_len, int total_slots, int top_k, int d_model,
        int intermediate, int num_experts,
        llaminar2::DeviceMoEWeightFormat format, int device_idx, void *stream,
        const llaminar2::MoEPrefillProjectionExecution &execution);
#endif
#ifdef HAVE_ROCM
    /**
     * @brief Enqueue the declared HIP phase with the identical floating contract.
     *
     * FP16, BF16 and FP32 identify weight storage, not intermediate precision.
     * The FP32 intermediate and fixed K reduction remain identical across
     * complete, separated and column-sharded execution.
     * @return False for incomplete bindings, invalid geometry or launch failure.
     */
    bool rocmMoE_grouped_floating_prefill_pipeline(
        const float *hidden,
        const llaminar2::DeviceMoEFloatingMatrixDesc *gate_descs,
        const llaminar2::DeviceMoEFloatingMatrixDesc *up_descs,
        const llaminar2::DeviceMoEFloatingMatrixDesc *down_descs,
        const int *original_to_grouped, const int *original_expert_ids,
        const float *grouped_weights, float *grouped_swiglu,
        float *output, float *canonical_routes,
        int seq_len, int total_slots, int top_k, int d_model,
        int intermediate, int num_experts,
        llaminar2::DeviceMoEWeightFormat format, int device_idx, void *stream,
        const llaminar2::MoEPrefillProjectionExecution &execution);
#endif
}
