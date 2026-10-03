/**
 * @file ROCmMoEGroupedPrefillKernels.h
 * @brief Canonical host/HIP launch boundary for exact grouped expert projections.
 *
 * This interface borrows already-prepared descriptors, stable device grouping
 * and arena-owned scratch. The explicit execution policy defines which buffers
 * are produced and whether down output is a physical column slice. Projection
 * phases share the installed production kernels; no test-only arithmetic is
 * introduced for their correctness or economy probes.
 */
#pragma once

#include "../../common/MoEPrefillProjectionExecution.h"
#include <cstdint>

/**
 * @brief Capture a complete or explicitly split quantized expert transaction.
 *
 * Gate/up requires hidden rows, gate/up descriptors, group token indices and
 * input/gate/up scratch. Down requires down descriptors, the inverse route map,
 * grouped route weights, projection scratch, and an output target. Both require
 * group counts/offsets, work-directory storage and grouped SwiGLU INT8/scales.
 * An optional prequantized hidden publication must provide both rows and scales.
 * Format masks are immutable capture identity for the complete source pipeline,
 * even when only one projection phase runs.
 *
 * Buffers retain the existing complete-pipeline capacities. A down slice's
 * descriptor N and output row stride are execution.columnCount(); its K and
 * serial-M1 policy retain the original unsharded dimensions. No caller may
 * change these pointers, dimensions or policy after graph recording.
 *
 * @param execution Validated phase and immutable source/output geometry.
 * @return True if the declared transaction was enqueued on the non-null stream.
 */
extern "C"
bool rocmMoE_grouped_prefill_pipeline(
    const float *d_hidden,
    const int8_t *d_prequantized_hidden,
    const float *d_prequantized_hidden_scales,
    const void *d_gate_desc_table,
    const void *d_up_desc_table,
    const void *d_down_desc_table,
    const int *d_group_counts,
    const int *d_group_offsets,
    const int *d_group_token_indices,
    const int *d_original_to_grouped,
    const int *d_original_expert_ids,
    const float *d_group_weights,
    int *d_group_work_directory,
    int8_t *d_scratch_A_int8,
    float *d_scratch_scales,
    float *d_scratch_gate,
    float *d_scratch_up,
    int8_t *d_scratch_swiglu_int8,
    float *d_scratch_swiglu_scales,
    float *d_output,
    float *d_canonical_route_contributions,
    int num_experts,
    int d_model,
    int intermediate,
    int total_slots,
    int top_k,
    int grouped_indices_are_route_slots,
    uint32_t gateup_codebook_mask,
    uint32_t down_codebook_mask,
    uint32_t gateup_policy_codebook_mask,
    uint32_t down_policy_codebook_mask,
    int device_id,
    void *stream,
    const llaminar2::MoEPrefillProjectionExecution &execution);

/**
 * @brief Fold already-weighted route outputs in the original router order.
 *
 * Independent down dots may execute concurrently, but every token/output
 * column has one writer accumulating routes in ascending order. The input
 * is [seq_len, top_k, d_model]; the output is [seq_len, d_model]. A column
 * shard supplies its physical width without changing the route arithmetic.
 * Both buffers must remain bound through completion on the explicit stream.
 *
 * @return True if the exact ordered fold was enqueued successfully.
 */
extern "C"
bool rocmMoE_reduce_canonical_route_contributions(
    const float *d_route_contributions,
    float *d_output,
    int seq_len,
    int top_k,
    int d_model,
    int device_id,
    void *stream);
