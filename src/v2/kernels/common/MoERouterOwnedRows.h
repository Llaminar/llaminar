/**
 * @file MoERouterOwnedRows.h
 * @brief Participant-local routing of complete token rows without new arithmetic.
 *
 * These launch bridges share the installed router bodies. They produce compact
 * local logits and selections; a separate explicit collective must publish
 * complete selections before experts or routing histograms consume them.
 * They neither perform communication nor install a model-graph routing policy.
 */
#pragma once
#include "MoERouterRowPacket.h"
#include "config/MoERouterArithmeticPolicy.h"

namespace llaminar2
{
    /** @brief Compact destination owned by a row router, before collective publication.
     * Counts belong to the domain fabric; the packet belongs to the admitted arena.
     * Neither this binding nor the router owns another transport epoch. */
    struct MoERouterOwnedRowOutputs
    {
        DeviceRowPartition partition;
        MoERouterSelectedRoute *selected = nullptr;
        std::uint64_t *selected_bytes = nullptr;
    };

    /**
     * @brief sqrt(softplus) score/hash routing applied to the compact local logits (DeepSeek-V4).
     * @details Selection is the top-k of s + bias (or the hash table row of
     * the token); weights are the unbiased s of the chosen experts normalized
     * to one and multiplied by @c route_scale. Disabled means ordinary softmax.
     */
    struct MoERouterScoredRouting
    {
        bool enabled = false;
        const float *bias = nullptr;            ///< [experts] selection bias (score routing).
        const std::int32_t *hash_table = nullptr; ///< [vocab, top_k] (hash routing).
        int hash_vocab = 0;
        const std::int32_t *token_ids = nullptr;  ///< [capacity] original-coordinate token ids.
        float route_scale = 1.0f;
        const float *image_bias = nullptr;        ///< exp_probs_b_vl for image sentinels (vision), or null.
        std::int32_t image_sentinel_base = 0;
        std::int32_t image_sentinel_count = 0;

        /** @return Exactly one selection authority, with its inputs. */
        [[nodiscard]] bool valid() const noexcept
        {
            return !enabled || ((hash_table ? (token_ids && hash_vocab > 0 && !bias) : bias != nullptr) &&
                                (!image_bias || (token_ids && image_sentinel_count > 0)));
        }
    };

    /** @brief Already prepared router representation, independent of expert codebook. */
    enum class MoERouterPreparedFormat { FP32, FP16, BF16, BlockQ8 };

    /** @brief Complete immutable launch bindings; every pointer is participant-local. */
    struct MoERouterOwnedRowsLaunch
    {
        DeviceRowPartition partition;
        int capacity = 0; ///< Prefill matrix capacity (at least two), not local live rows.
        int width = 0;
        int experts = 0;
        int top_k = 0;
        bool normalize = true;
        const std::int32_t *live_rows = nullptr; ///< Existing canonical device authority.
        const float *hidden = nullptr; ///< All original rows retained for local consumers.
        MoERouterPreparedFormat format = MoERouterPreparedFormat::FP32;
        RouterQ8Basis q8_basis = RouterQ8Basis::Original; ///< Only BlockQ8 consumes a rotated basis.
        const void *gate = nullptr;
        const float *gate_scales = nullptr; ///< Required only by prepared BlockQ8.
        std::int8_t *hidden_q8 = nullptr; ///< Full-row side product, never a partial publication.
        float *hidden_scales = nullptr;
        float *logits = nullptr; ///< Compact local rows, same expert order and FP32 bits.
        MoERouterSelectedRoute *selected = nullptr; ///< Compact pairs in original top-k order.
        std::uint64_t *selected_bytes = nullptr; ///< Canonical local collective extent, written even when empty.
        MoERouterScoredRouting scored; ///< Disabled: softmax routing.

        /** @return Whether shared geometry and mandatory addresses are complete.
         * Backend-native format limits are checked by each launch bridge. */
        [[nodiscard]] bool valid() const noexcept
        {
            // CUDA scalar decode has a different established K tree. This
            // prefill primitive must not silently substitute that contract.
            // A retained prefill may still replay with zero or one LIVE row.
            return scored.valid() &&
                capacity >= 2 && width > 0 && experts > 0 && top_k > 0 && top_k <= experts &&
                live_rows && hidden && gate && logits && selected && selected_bytes &&
                partition.participants() > 1 &&
                (q8_basis == RouterQ8Basis::Original ||
                    (q8_basis == RouterQ8Basis::SignedHadamard32 && format == MoERouterPreparedFormat::BlockQ8)) &&
                (format != MoERouterPreparedFormat::BlockQ8 ||
                    (width % 32 == 0 && gate_scales && hidden_q8 && hidden_scales));
        }
    };
    namespace cuda
    {
        struct RouterQ8PrefillHardware;
        /** @brief Enqueue supported native prefill routing on the exact CUDA stream.
         * @param launch Frozen ownership, prepared weights and persistent local banks.
         * @param stream Non-null explicit execution stream on the calling device.
         * @param hardware Immutable native limits prepared before Q8 capture.
         * @return False for invalid bindings, unsupported format or native launch error. */
        bool routeOwnedRows(const MoERouterOwnedRowsLaunch &launch, void *stream,
                            const RouterQ8PrefillHardware &hardware);
    }
    namespace rocm
    {
        /** @brief Enqueue native routing and full hidden-Q8 publication when required.
         * @param launch Frozen ownership, prepared weights and persistent local banks.
         * @param stream Non-null explicit execution stream on the calling device.
         * @return False for invalid bindings, unsupported format or native launch error. */
        bool routeOwnedRows(const MoERouterOwnedRowsLaunch &launch, void *stream);
    }
}
