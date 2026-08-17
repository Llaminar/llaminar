/**
 * @file ParityGDNHeadPermutation.h
 * @brief Pure comparison-time layout mapping for unequal-head Qwen GDN tensors.
 *
 * Qwen MoE GGUF tensors retain a ratio-grouped value-head order while the
 * Hugging Face reference loader publishes the equivalent heads interleaved.
 * Production arithmetic must retain its model-native layout; parity remaps a
 * copied diagnostic tensor at the comparison boundary only. Keeping the pure
 * mapping here makes every supported stage and row geometry unit-testable.
 */

#pragma once

#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

namespace llaminar2::test::parity
{
    /**
     * @brief Model-derived head geometry required by the diagnostic mapping.
     */
    struct ParityGDNHeadConfig
    {
        int n_k_heads = 0;
        int n_v_heads = 0;
        int d_state = 0;
        bool is_moe = false;

        /** @brief Return whether this model uses the unequal-head MoE layout. */
        bool needsPermutation() const
        {
            return is_moe && n_k_heads > 0 && n_v_heads > 0 && d_state > 0 &&
                   n_k_heads != n_v_heads && n_v_heads % n_k_heads == 0;
        }

        /** @brief Return value heads stored in each GGUF ratio group. */
        int headsPerGroup() const
        {
            return n_k_heads > 0 ? n_v_heads / n_k_heads : 1;
        }
    };

    /**
     * @brief Map a copied production checkpoint into Hugging Face head order.
     *
     * Full value-head stages are interpreted as
     * `[row, value_head, d_state]`. Alpha and beta are scalar per-head stages
     * interpreted as `[row, value_head]`. Packed QKV/conv stages preserve Q and
     * K and map only the value-head suffix. The production tensor is never
     * mutated.
     *
     * @param production_data Model-native diagnostic tensor.
     * @param size Number of float elements in the tensor.
     * @param stage Canonical parity checkpoint name.
     * @param gdn Model-derived unequal-head geometry.
     * @return Mapped copy, or an empty vector when no mapping applies or the
     *         tensor shape is incompatible with the stage geometry.
     */
    inline std::vector<float> applyParityGDNHeadPermutation(
        const float *production_data,
        size_t size,
        const std::string &stage,
        const ParityGDNHeadConfig &gdn)
    {
        if (!production_data || !gdn.needsPermutation())
            return {};

        const int n_k = gdn.n_k_heads;
        const int n_v = gdn.n_v_heads;
        const int d = gdn.d_state;
        const int heads_per_group = gdn.headsPerGroup();

        std::vector<int> inverse_permutation(static_cast<size_t>(n_v));
        for (int reference_head = 0; reference_head < n_v; ++reference_head)
        {
            const int ratio = reference_head % heads_per_group;
            const int group = reference_head / heads_per_group;
            inverse_permutation[static_cast<size_t>(reference_head)] =
                ratio * n_k + group;
        }

        const auto permute_all_value_heads =
            [&](const float *source,
                size_t element_count,
                size_t head_width) -> std::vector<float>
        {
            const size_t head_count = static_cast<size_t>(n_v);
            const size_t row_width = head_count * head_width;
            if (head_width == 0 || row_width == 0 ||
                element_count % row_width != 0)
            {
                return {};
            }

            const size_t rows = element_count / row_width;
            std::vector<float> output(element_count);
            for (size_t row = 0; row < rows; ++row)
            {
                for (size_t reference_head = 0;
                     reference_head < head_count;
                     ++reference_head)
                {
                    const size_t production_head = static_cast<size_t>(
                        inverse_permutation[reference_head]);
                    std::memcpy(
                        output.data() +
                            (row * head_count + reference_head) * head_width,
                        source +
                            (row * head_count + production_head) * head_width,
                        head_width * sizeof(float));
                }
            }
            return output;
        };

        if (stage == "GDN_ALPHA" || stage == "GDN_BETA")
        {
            return permute_all_value_heads(
                production_data,
                size,
                /*head_width=*/1);
        }

        if (stage == "GDN_Z_PROJECTION" ||
            stage == "GDN_DELTA_RULE_OUTPUT" ||
            stage == "GDN_NORM_GATE_OUTPUT")
        {
            return permute_all_value_heads(
                production_data,
                size,
                static_cast<size_t>(d));
        }

        if (stage == "QKV_PROJECTION" || stage == "GDN_CONV1D_OUTPUT")
        {
            const size_t q_width = static_cast<size_t>(n_k * d);
            const size_t k_width = static_cast<size_t>(n_k * d);
            const size_t v_width = static_cast<size_t>(n_v * d);
            const size_t row_width = q_width + k_width + v_width;
            if (row_width == 0 || size % row_width != 0)
                return {};

            const size_t rows = size / row_width;
            std::vector<float> output(production_data, production_data + size);
            for (size_t row = 0; row < rows; ++row)
            {
                const float *v_source =
                    production_data + row * row_width + q_width + k_width;
                float *v_output =
                    output.data() + row * row_width + q_width + k_width;
                for (size_t reference_head = 0;
                     reference_head < static_cast<size_t>(n_v);
                     ++reference_head)
                {
                    const size_t production_head = static_cast<size_t>(
                        inverse_permutation[reference_head]);
                    std::memcpy(
                        v_output + reference_head * static_cast<size_t>(d),
                        v_source + production_head * static_cast<size_t>(d),
                        static_cast<size_t>(d) * sizeof(float));
                }
            }
            return output;
        }

        return {};
    }
} // namespace llaminar2::test::parity
