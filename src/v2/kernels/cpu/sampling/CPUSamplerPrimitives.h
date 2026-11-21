#pragma once

#include <cstddef>
#if LLAMINAR_COMPILED_WITH_NEON
#include <arm_neon.h>
#endif

namespace llaminar2::cpu_sampling
{

    /**
     * Select the top-k logits in descending order.
     *
     * The scalar, AVX2, and AVX512 entry points are intentionally public so
     * unit tests can prove the ISA-specific paths remain equivalent. The
     * runtime-dispatched select_topk() follows CPUFeatures.h.
     *
     * @return Number of entries written to out_logits/out_ids.
     */
    int select_topk_scalar(
        const float *logits,
        int vocab_size,
        int top_k,
        float *out_logits,
        int *out_ids);

    int select_topk_avx2(
        const float *logits,
        int vocab_size,
        int top_k,
        float *out_logits,
        int *out_ids);

    int select_topk_avx512(
        const float *logits,
        int vocab_size,
        int top_k,
        float *out_logits,
        int *out_ids);

#if LLAMINAR_COMPILED_WITH_NEON
    /**
     * Select the top-k logits using the native four-lane threshold filter.
     *
     * The selection result is identical to select_topk_scalar: the same
     * insert_candidate order and tie handling in increasing index order; NEON
     * only accelerates the threshold comparison (vcgtq_f32 lane mask) that
     * skips candidates already below the current k-th logit. Compile-time
     * NEON builds expose this named entry so unit tests can prove equivalence.
     */
    int select_topk_neon(
        const float *logits,
        int vocab_size,
        int top_k,
        float *out_logits,
        int *out_ids);
#endif

    int select_topk(
        const float *logits,
        int vocab_size,
        int top_k,
        float *out_logits,
        int *out_ids);

} // namespace llaminar2::cpu_sampling
