#pragma once

#include <cstddef>

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

    int select_topk(
        const float *logits,
        int vocab_size,
        int top_k,
        float *out_logits,
        int *out_ids);

} // namespace llaminar2::cpu_sampling
