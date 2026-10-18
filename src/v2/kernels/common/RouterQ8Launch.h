/**
 * @file RouterQ8Launch.h
 * @brief Typed basis contract for captured Q8 router preparation and execution.
 *
 * Gate bytes and temporary hidden bytes must use the same immutable basis.
 * Each logits bridge returns the hidden bank in ORIGINAL coordinates for
 * expert consumers, after its last router reader on the exact supplied stream.
 * No bank allocation, host copy or synchronization occurs in these bridges.
 */
#pragma once
#include "config/MoERouterArithmeticPolicy.h"
#include <cstdint>

extern "C"
{
    /** @brief Quantize original floating gate rows once in the selected basis.
     * @param weights Immutable FP32 source [experts,width].
     * @param codes Retained Q8 destination. @param scales Retained block32 scales.
     * @param width Positive multiple of 32. @param experts Positive row count.
     * @param device Owning ordinal. @param stream Exact non-null producer stream.
     * @param basis Immutable paired weight/activation basis. @return Launch success. */
    bool cudaMoE_quantize_router_gate_q8(const float *weights, std::int8_t *codes,
        float *scales, int width, int experts, int device, void *stream,
        llaminar2::RouterQ8Basis basis = llaminar2::RouterQ8Basis::Original);
    /** @copydoc cudaMoE_quantize_router_gate_q8 */
    bool hipMoE_quantize_router_gate_q8(const float *weights, std::int8_t *codes,
        float *scales, int width, int experts, int device, void *stream,
        llaminar2::RouterQ8Basis basis = llaminar2::RouterQ8Basis::Original);

    /** @brief Compute one row, then publish original-basis hidden Q8 for experts.
     * @param hidden Immutable original FP32 input.
     * @param hidden_codes Retained Q8 bank; original basis at completion.
     * @param hidden_scales Retained original-basis block32 scales at completion.
     * @param gate_codes Prepared router codes in basis. @param gate_scales Their scales.
     * @param logits FP32 expert outputs. @param width Positive multiple of 32.
     * @param experts Positive output width. @param device Owning ordinal.
     * @param stream Exact non-null producer stream. @param basis Prepared gate basis.
     * @return Success for every ordered launch, including expert-bank publication. */
    bool cudaMoE_gate_logits_single_token_q8_weights(const float *hidden,
        std::int8_t *hidden_codes, float *hidden_scales, const std::int8_t *gate_codes,
        const float *gate_scales, float *logits, int width, int experts, int device,
        void *stream, llaminar2::RouterQ8Basis basis = llaminar2::RouterQ8Basis::Original);
    /** @copydoc cudaMoE_gate_logits_single_token_q8_weights */
    bool hipMoE_gate_logits_single_token_q8_weights(const float *hidden,
        std::int8_t *hidden_codes, float *hidden_scales, const std::int8_t *gate_codes,
        const float *gate_scales, float *logits, int width, int experts, int device,
        void *stream, llaminar2::RouterQ8Basis basis = llaminar2::RouterQ8Basis::Original);

    /** @brief Group rows with the serial arithmetic and original expert-bank publication.
     * @param hidden Original FP32 rows. @param hidden_codes Retained original-basis output bank.
     * @param hidden_scales Original-basis output scales. @param gate_codes Prepared router codes.
     * @param gate_scales Prepared router scales. @param logits FP32 row-major output.
     * @param rows Captured row capacity. @param width Positive multiple of 32.
     * @param experts Output width. @param device Owning ordinal.
     * @param stream Exact non-null producer stream. @param basis Prepared gate basis.
     * @return Success for every ordered launch. */
    bool cudaMoE_gate_logits_q8_weights_decode_equivalent_rows(const float *hidden,
        std::int8_t *hidden_codes, float *hidden_scales, const std::int8_t *gate_codes,
        const float *gate_scales, float *logits, int rows, int width, int experts,
        int device, void *stream,
        llaminar2::RouterQ8Basis basis = llaminar2::RouterQ8Basis::Original);
    /** @brief HIP grouped contract with device-owned live rows; inactive rows stay untouched.
     * @param hidden Original FP32 rows. @param hidden_codes Original-basis output bank.
     * @param hidden_scales Original-basis output scales. @param gate_codes Prepared router codes.
     * @param gate_scales Prepared router scales. @param logits FP32 row-major output.
     * @param rows Captured capacity. @param width Positive multiple of 32.
     * @param experts Output width. @param device Owning ordinal. @param stream Exact non-null stream.
     * @param live_rows Optional device-owned active extent. @param basis Prepared gate basis.
     * @return Success for every ordered launch. */
    bool hipMoE_gate_logits_q8_weights_decode_equivalent_rows(const float *hidden,
        std::int8_t *hidden_codes, float *hidden_scales, const std::int8_t *gate_codes,
        const float *gate_scales, float *logits, int rows, int width, int experts,
        int device, void *stream, const int *live_rows = nullptr,
        llaminar2::RouterQ8Basis basis = llaminar2::RouterQ8Basis::Original);
}
