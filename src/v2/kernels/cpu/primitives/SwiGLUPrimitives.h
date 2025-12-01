/**
 * @file SwiGLUPrimitives.h
 * @brief SwiGLU primitives (scalar and vectorized)
 * @author GitHub Copilot
 */

#pragma once

#include <cstdint>

namespace llaminar2::primitives
{
    /**
     * @brief Compute SwiGLU: output = silu(gate) * up
     *
     * This implements the HuggingFace FFN formula: act_fn(gate_proj) * up_proj
     * where act_fn is SiLU (Sigmoid Linear Unit): silu(x) = x * sigmoid(x)
     *
     * @param gate Pointer to gate tensor data (activation applied here)
     * @param up Pointer to up tensor data (linear multiplier)
     * @param output Pointer to output tensor data
     * @param size Number of elements
     */
    void compute_swiglu(const float *gate, const float *up, float *output, int size);

    void compute_swiglu_bf16(const uint16_t *gate, const uint16_t *up, uint16_t *output, int size);
    void compute_swiglu_fp16(const uint16_t *gate, const uint16_t *up, uint16_t *output, int size);
    void compute_swiglu_q8_1(const void *gate, const void *up, void *output, int size);

} // namespace llaminar2::primitives
