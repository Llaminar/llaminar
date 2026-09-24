/**
 * @file CUDAGroupedVerifierLaunch.h
 * @brief Canonical CUDA grouped-verifier launch ABI with device-owned rows.
 *
 * Graph recording fixes all pointers, grid dimensions, and scratch strides.
 * Only the borrowed controller count changes between replays. All physical
 * codebook shards consume the same descriptor and preserve serial-M1 order.
 */
#pragma once

#include "CUDADeviceWorkspace.h"
#include "kernels/common/DeviceRowRange.h"
#include <cstdint>

extern "C"
{
    /**
     * @brief Launch captured grouped GEMV under the source-format M1 policy.
     * @param d_A_int8 Row-major quantized activations [M,K].
     * @param d_payload Prepared native-format weight bytes.
     * @param d_scales Primary packed weight scale plane.
     * @param d_mins Optional secondary scale/minimum plane.
     * @param d_emins Optional extended minimum plane.
     * @param d_C_fp32 Persistent physical output matrix [M,N].
     * @param d_scales_A_block Activation scales per row and 32-value block.
     * @param M Fixed physical rows (at least two), not the current live count.
     * @param N Output width.
     * @param K Reduction width, a positive multiple of 32.
     * @param alpha Product scale.
     * @param beta Existing-output scale.
     * @param d_C_existing Optional existing output for nonzero beta.
     * @param d_bias Optional output-column bias.
     * @param codebook_id Physical packed decoder identity.
     * @param arithmetic_policy_codebook_id Source-format serial reduction policy.
     * @param cuda_device_id Device owning all operands and workspace.
     * @param stream Exact non-null stream ordered after count publication.
     * @param gemv_ctx Pre-bound persistent split-K workspace owner.
     * @param rm_slot Optional pre-prepared row-major weights for that policy.
     * @param row_range Optional host launch descriptor copied by value; its
     *        borrowed count is read only on device, never during graph recording.
     * @return False for invalid geometry, missing resources, or launch failure.
     */
    bool cudaNativeVNNIGemvTuned_small_m_fp32_withPolicy(
        const int8_t *d_A_int8, const uint8_t *d_payload,
        const uint16_t *d_scales, const uint16_t *d_mins,
        const uint32_t *d_emins, float *d_C_fp32,
        const float *d_scales_A_block, int M, int N, int K,
        float alpha, float beta, const float *d_C_existing,
        const float *d_bias, uint8_t codebook_id,
        uint8_t arithmetic_policy_codebook_id, int cuda_device_id,
        void *stream, CUDAGemvContext *gemv_ctx,
        CUDARowMajorWeights **rm_slot,
        const llaminar2::DeviceRowRange *row_range = nullptr);
}
