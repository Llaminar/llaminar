/**
 * @file SwiGLUPrimitives.h
 * @brief CPU SwiGLU over caller-owned original floating or typed block storage.
 *
 * Callers retain both inputs and the output until serial return or the retained
 * workshare completion edge. Each output bank must be disjoint from or exactly
 * identical to each input bank; partial aliases and malformed live geometry are
 * rejected before mutation. Empty workshare entrypoints accept null storage and
 * still join the retained team; serial variants remain noncollective. No API
 * allocates an expanded tensor or publishes coherence.
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
     * @param size Nonnegative element count, including bounded vector tails.
     * @throws std::invalid_argument On malformed live buffers or partial aliases.
     */
    void compute_swiglu(const float *gate, const float *up, float *output, int size);

    /**
     * @brief Compute ordinary SwiGLU without an OpenMP workshare or team.
     * @param gate Original gate values.
     * @param up Original multiplier values.
     * @param output Destination, disjoint from or identical to each source.
     * @param size Nonnegative element count, with the same native tail program.
     * @throws std::invalid_argument On malformed live buffers or partial aliases.
     */
    void compute_swiglu_serial(const float *gate, const float *up, float *output, int size);

    /**
     * @brief Compute GPU-aligned expert SwiGLU with the CPU/CUDA/ROCm byte contract.
     *
     * This workshare-aware entry point is used for grouped CPU expert rows. It
     * preserves the exact polynomial and rounding sequence used by device
     * experts rather than selecting the ordinary CPU fast-exp approximation.
     *
     * @param gate Gate projection values.
     * @param up Up projection values.
     * @param output Placement-invariant FP32 results.
     * @param size Nonnegative element count.
     * @throws std::invalid_argument On malformed live buffers or partial aliases.
     */
    void compute_swiglu_gpu_aligned_expert(
        const float *gate,
        const float *up,
        float *output,
        int size);

    /**
     * @brief Serial variant of compute_swiglu_gpu_aligned_expert().
     * @param gate Gate projection values.
     * @param up Up projection values.
     * @param output Placement-invariant FP32 results.
     * @param size Nonnegative element count.
     * @throws std::invalid_argument On malformed live buffers or partial aliases.
     */
    void compute_swiglu_gpu_aligned_expert_serial(
        const float *gate,
        const float *up,
        float *output,
        int size);

    /**
     * @brief Workshare ordinary SwiGLU over BF16 storage without a tensor mirror.
     * @param gate Original BF16 gate words.
     * @param up Original BF16 multiplier words.
     * @param output BF16 results using the existing ties-even storage program.
     * @param size Nonnegative element count, including bounded storage tails.
     * @throws std::invalid_argument On malformed live buffers or partial aliases.
     */
    void compute_swiglu_bf16(const uint16_t *gate, const uint16_t *up, uint16_t *output, int size);

    /**
     * @brief Workshare ordinary SwiGLU over portable binary16 storage.
     * @param gate Original binary16 gate words.
     * @param up Original binary16 multiplier words.
     * @param output Results retaining normal ties-even/subnormal truncation.
     * @param size Nonnegative element count, including bounded storage tails.
     * @throws std::invalid_argument On malformed live buffers or partial aliases.
     */
    void compute_swiglu_fp16(const uint16_t *gate, const uint16_t *up, uint16_t *output, int size);

    /**
     * @brief Workshare complete ordinary Q8_1 SwiGLU blocks.
     * @param gate Original Q8_1 gate blocks.
     * @param up Original Q8_1 multiplier blocks.
     * @param output Complete result blocks with exact signed sums.
     * @param size Nonnegative multiple of 32, never a silently truncated extent.
     * @throws std::invalid_argument Before mutation on malformed geometry or partial aliases.
     */
    void compute_swiglu_q8_1(const void *gate, const void *up, void *output, int size);

} // namespace llaminar2::primitives
