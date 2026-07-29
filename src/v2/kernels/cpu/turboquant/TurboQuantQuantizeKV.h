/**
 * @file TurboQuantQuantizeKV.h
 * @brief Fused CPU quantization for selectable TurboQuant KV cache modes.
 *
 * Production CPU KV publication stores keys as TQ8 and selects either TQ4 or
 * TQ8 for values. This file owns the economical grouped implementation shared
 * by decode, MTP verifier groups, and prefill:
 *
 * 1. Derived per-head rotation contexts are resolved under one lock.
 * 2. One OpenMP workshare distributes independent `(row, head)` items.
 * 3. K and V for an item are quantized together, retaining the rotation matrix
 *    in cache and avoiding a second OpenMP launch.
 * 4. Every vector calls the same ISA-dispatched scalar mathematical primitive
 *    used by one-row decode, preserving byte identity for every M.
 *
 * The caller owns all output storage. The hot loop performs no allocation,
 * locking, atomics, transfer, or synchronization beyond its one workshare.
 */

#pragma once

#include "TurboQuantContext.h"
#include "TurboQuantKVParallelPolicy.h"
#include "TurboQuantQuantizeTQ4.h"
#include "TurboQuantQuantizeTQ8.h"
#include "utils/OpenMPUtils.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace llaminar2
{
    /// Generous fixed bound that keeps context resolution allocation-free.
    inline constexpr int kMaxTurboQuantKVHeads = 128;

    /**
     * @brief Quantize one compile-time TurboQuant K/V geometry.
     *
     * @tparam D Per-head vector width.
     * @tparam VUsesTQ8 Whether V uses TQ8 (`true`) or TQ4 (`false`).
     */
    template <int D, bool VUsesTQ8>
    inline void turboquant_quantize_kv_rows_impl(
        const float *k_fp32,
        const float *v_fp32,
        uint8_t *k_blocks,
        uint8_t *v_blocks,
        int num_rows,
        int n_kv_heads,
        size_t k_row_bytes,
        size_t v_row_bytes,
        size_t k_block_bytes,
        size_t v_block_bytes,
        const TurboQuantContext *const *head_contexts)
    {
        const int kv_dim = n_kv_heads * D;
        const int work_items = num_rows * n_kv_heads;
        const auto quantize_one = [&](int row, int head)
        {
            const size_t input_offset =
                static_cast<size_t>(row) * kv_dim +
                static_cast<size_t>(head) * D;
            const size_t k_output_offset =
                static_cast<size_t>(row) * k_row_bytes +
                static_cast<size_t>(head) * k_block_bytes;
            const size_t v_output_offset =
                static_cast<size_t>(row) * v_row_bytes +
                static_cast<size_t>(head) * v_block_bytes;
            const TurboQuantContext &head_ctx = *head_contexts[head];

            alignas(64) float scratch0[D];
            alignas(64) float scratch1[D];
            auto *k_block = reinterpret_cast<TQ8Block<D> *>(
                k_blocks + k_output_offset);
            turboquant_quantize_tq8<D>(
                k_fp32 + input_offset,
                head_ctx,
                *k_block,
                scratch0,
                scratch1);

            if constexpr (VUsesTQ8)
            {
                auto *v_block = reinterpret_cast<TQ8Block<D> *>(
                    v_blocks + v_output_offset);
                turboquant_quantize_tq8<D>(
                    v_fp32 + input_offset,
                    head_ctx,
                    *v_block,
                    scratch0,
                    scratch1);
            }
            else
            {
                auto *v_block = reinterpret_cast<TQ4Block<D> *>(
                    v_blocks + v_output_offset);
                turboquant_quantize_tq4<D>(
                    v_fp32 + input_offset,
                    head_ctx,
                    *v_block,
                    scratch0,
                    scratch1);
            }
        };

        /*
         * Keep latency-sensitive decode groups entirely outside the OpenMP
         * runtime. OMP_WORKSHARE_COLLAPSE2_IF historically emulated serial
         * execution by creating a one-thread parallel region; measurements
         * showed that setup as a visible M=1..4 latency tax.
         */
        if (work_items < kTurboQuantKVParallelWorkItems)
        {
            for (int row = 0; row < num_rows; ++row)
            {
                for (int head = 0; head < n_kv_heads; ++head)
                {
                    quantize_one(row, head);
                }
            }
            return;
        }

        auto parallel_work = [&]()
        {
#pragma omp for collapse(2) schedule(static)
            for (int row = 0; row < num_rows; ++row)
            {
                for (int head = 0; head < n_kv_heads; ++head)
                {
                    quantize_one(row, head);
                }
            }
        };
        OMP_WORKSHARE_COLLAPSE2_IF(parallel_work, true);
    }

    /**
     * @brief Quantize FP32 K/V rows into TQ8-K/TQ4-V or TQ8-K/TQ8-V blocks.
     *
     * @param value_uses_tq8 Selects symmetric TQ8 value storage when true.
     * @throws std::invalid_argument for unsupported dimensions or null storage.
     */
    inline void turboquant_quantize_kv_rows(
        const float *k_fp32,
        const float *v_fp32,
        uint8_t *k_blocks,
        uint8_t *v_blocks,
        int num_rows,
        int head_dim,
        int n_kv_heads,
        size_t k_row_bytes,
        size_t v_row_bytes,
        size_t k_block_bytes,
        size_t v_block_bytes,
        const TurboQuantContext &context,
        bool value_uses_tq8)
    {
        if (!k_fp32 || !v_fp32 || !k_blocks || !v_blocks ||
            num_rows <= 0 || n_kv_heads <= 0 ||
            n_kv_heads > kMaxTurboQuantKVHeads)
        {
            throw std::invalid_argument(
                "turboquant_quantize_kv_rows: invalid storage or geometry");
        }

        std::array<const TurboQuantContext *, kMaxTurboQuantKVHeads>
            head_contexts{};
        context.resolve_layer_contexts(
            n_kv_heads, head_contexts.data(), head_contexts.size());

        const auto launch = [&]<int D>()
        {
            if (value_uses_tq8)
            {
                turboquant_quantize_kv_rows_impl<D, true>(
                    k_fp32, v_fp32, k_blocks, v_blocks,
                    num_rows, n_kv_heads,
                    k_row_bytes, v_row_bytes,
                    k_block_bytes, v_block_bytes,
                    head_contexts.data());
            }
            else
            {
                turboquant_quantize_kv_rows_impl<D, false>(
                    k_fp32, v_fp32, k_blocks, v_blocks,
                    num_rows, n_kv_heads,
                    k_row_bytes, v_row_bytes,
                    k_block_bytes, v_block_bytes,
                    head_contexts.data());
            }
        };

        switch (head_dim)
        {
        case 64:
            launch.template operator()<64>();
            break;
        case 128:
            launch.template operator()<128>();
            break;
        case 256:
            launch.template operator()<256>();
            break;
        default:
            throw std::invalid_argument(
                "turboquant_quantize_kv_rows: head_dim must be 64, 128, or 256");
        }
    }

} // namespace llaminar2
