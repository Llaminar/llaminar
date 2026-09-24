/**
 * @file ROCmGroupedVerifierLaunch.h
 * @brief Typed launch boundary for ordered, device-counted NativeVNNI rows.
 *
 * The physical matrix width determines grid and scratch strides. An optional
 * row descriptor borrows the device controller's live count; kernels skip
 * inactive tiles before reading weights and never publish their scratch rows.
 * This declaration is shared by production adapters and captured regressions
 * so adding a launch argument cannot leave a stale hand-written C ABI behind.
 */
#pragma once

#include "kernels/common/DeviceRowRange.h"
#include <cstdint>

extern "C"
{
    /**
     * @brief Execute grouped verifier rows using serial-M1 ordered arithmetic.
     * @param d_A_int8 Quantized row-major activations [M,K].
     * @param d_payload Prepared native-format device weight bytes.
     * @param d_block_scales Prepared per-block primary scale metadata.
     * @param d_block_mins Optional format-owned secondary scale/minimum metadata.
     * @param d_block_emins Optional format-owned extended minimum metadata.
     * @param d_C_fp32 Persistent row-major output [M,N].
     * @param d_scale_A_blockwise Activation scale per row and K block.
     * @param d_sum_A_blockwise Activation sum per block, where required.
     * @param d_partial_fp32 Declared split-K workspace; its row stride is physical.
     * @param M Captured physical rows, at least two.
     * @param N Output columns.
     * @param K Reduction width, a multiple of 32.
     * @param codebook_id Physical packed weight decoder.
     * @param arithmetic_policy_codebook_id Source-format serial reduction policy.
     * @param device_id ROCm device owning all operands.
     * @param stream Exact non-null execution stream.
     * @param row_range Optional host descriptor copied into launch parameters;
     *        its device count must be published before this stream consumes it.
     * @return False for invalid arguments or failed launch, never another route.
     */
    bool rocmGemv_native_vnni_small_m_fp32_with_sums_policy(
        const int8_t *d_A_int8, const uint8_t *d_payload,
        const void *d_block_scales, const void *d_block_mins,
        const void *d_block_emins, float *d_C_fp32,
        const float *d_scale_A_blockwise, const int32_t *d_sum_A_blockwise,
        float *d_partial_fp32, int M, int N, int K,
        uint8_t codebook_id, uint8_t arithmetic_policy_codebook_id,
        int device_id, void *stream,
        const llaminar2::DeviceRowRange *row_range = nullptr);

    /**
     * @brief Fused same-format projections sharing quantization and row admission.
     *
     * Operand planes follow the single-projection contract above. Pointer/N
     * arrays contain num_projections entries and are copied into kernel args;
     * every output/partial owns a distinct slice. policy_kb and
     * policy_target_waves must reproduce the generated serial-M1 schedule.
     * row_range is shared by the entire bundle, never read on the host.
     * d_sum_A_blockwise is required: the shared quantizer is the activation
     * sum producer for every fused decoder, including symmetric-format bundles.
     * @return False on invalid bindings or a failed launch, without replay.
     */
    bool rocmGemv_native_vnni_small_m_batched_fp32_with_sums_policy(
        const int8_t *d_A_int8,
        const uint8_t *const *d_payloads,
        const uint16_t *const *d_block_scales,
        const uint16_t *const *d_block_mins,
        const uint32_t *const *d_block_emins,
        const float *const *d_biases,
        float *const *d_outputs,
        const float *d_scale_A_blockwise, // [M × blocks_per_row]
        const int32_t *d_sum_A_blockwise, // Required [M × blocks_per_row]
        float *const *d_partials,         // per-projection [KB_MAX × M × N]
        const int *Ns,
        int num_projections,
        int M, int K,
        uint8_t codebook_id,
        int device_id, void *stream,
        int policy_kb,
        int policy_target_waves,
        const llaminar2::DeviceRowRange *row_range = nullptr);

    /**
     * @brief Mixed-format fused bundle with one physical/live row authority.
     *
     * The contract is identical to the same-format bundle except codebook_ids
     * selects the physical decoder per projection. The adapter groups only
     * compatible serial-M1 partition schedules into one launch.
     * @return False on an invalid format, binding, geometry, or launch.
     */
    bool rocmGemv_native_vnni_small_m_batched_mixed_fp32_with_sums_policy(
        const int8_t *d_A_int8,
        const uint8_t *const *d_payloads,
        const uint16_t *const *d_block_scales,
        const uint16_t *const *d_block_mins,
        const uint32_t *const *d_block_emins,
        const float *const *d_biases,
        float *const *d_outputs,
        const float *d_scale_A_blockwise, // [M × blocks_per_row]
        const int32_t *d_sum_A_blockwise, // Required [M × blocks_per_row]
        float *const *d_partials,         // per-projection [KB_MAX × M × N]
        const int *Ns,
        const uint8_t *codebook_ids,
        int num_projections,
        int M, int K,
        int device_id, void *stream,
        int policy_kb,
        int policy_target_waves,
        const llaminar2::DeviceRowRange *row_range = nullptr);
}
