/**
 * @file CUDARouterQ8Prefill.h
 * @brief Shape- and device-aware captured INT8 router prefill with exact Q8 arithmetic.
 *
 * Tile selection uses output-grid occupancy, matrix tails and reduction depth.
 * Device limits are observed before capture and remain immutable launch metadata.
 * A sparse output grid shares the canonical K tree between cooperating warps;
 * a dense grid groups output tiles. Neither choice changes quantization or the
 * reduction order. The enclosing publisher restores the original expert basis.
 */
#pragma once
#include "kernels/common/DeviceRowPartition.h"
#include "config/MoERouterArithmeticPolicy.h"
#include <cstdint>

namespace llaminar2::cuda
{
    /** @brief Immutable setup observation; no device query occurs during capture. */
    struct RouterQ8PrefillHardware
    {
        int device = -1;
        int multiprocessors = 0;
        int block_slots_per_sm = 0;

        /** @return Whether the complete native observation is usable. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            return device >= 0 && multiprocessors > 0 &&
                block_slots_per_sm > 0 && block_slots_per_sm <= 1024;
        }
    };

    /** @brief Observe the explicit stream's device before any capture begins.
     * @param device Exact CUDA ordinal. @param stream Non-null producer stream.
     * @param result Receives the complete observation only on success.
     * @return False for a missing/mismatched stream, capture, or native query error. */
    bool prepareRouterQ8PrefillHardware(int device, void *stream,
                                       RouterQ8PrefillHardware &result);

    /** @brief Borrowed operands and immutable physical geometry for one transaction. */
    struct RouterQ8PrefillView
    {
        const std::int8_t *hidden_codes = nullptr;
        const float *hidden_scales = nullptr;
        const std::int8_t *gate_codes = nullptr;
        const float *gate_scales = nullptr;
        float *logits = nullptr;
        int capacity = 0;
        int width = 0;
        int experts = 0;
        const int *live_rows = nullptr;
        DeviceRowPartition partition;
        RouterQ8PrefillHardware hardware;

        /** @return Whether every bank, logical extent and hardware limit is declared. */
        [[nodiscard]] bool valid() const noexcept
        {
            return hidden_codes && hidden_scales && gate_codes && gate_scales && logits &&
                capacity > 0 && width > 0 && width % 32 == 0 && experts > 0 &&
                hardware.valid() && (partition.participants() == 1 || live_rows);
        }
    };

    /** @brief Complete tile geometry; K partitions retain a fixed arithmetic tree. */
    struct RouterQ8PrefillGeometry
    {
        int rows = 0;
        int experts = 0;
        int warps = 0;
        int k_warps = 0;
        int constant_width = 0;
    };

    /**
     * @brief Choose a tile from matrix geometry and observed GPU block capacity.
     * @param local_capacity Admitted participant-local rows, possibly zero.
     * @param width Block32 hidden width. @param experts Complete router output width.
     * @param hardware Immutable limits prepared on the launch stream's device.
     * @return Complete launch geometry, or a zero geometry for malformed inputs.
     *
     * Sparse grids target five eighths of one-warp block capacity with K
     * cooperation, capped at sixteen warps and one block per K worker. This
     * exposes parallelism without assigning workers only empty leaves. Dense
     * grids group two or four output
     * tiles once a one-warp grid exceeds device block capacity. Among equal-size
     * groups, minimize inactive warp tiles, then distinct operand rows per CTA.
     * These are grid/resource thresholds, never special cases for particular M.
     */
    [[nodiscard]] constexpr RouterQ8PrefillGeometry routerQ8PrefillGeometry(
        int local_capacity, int width, int experts,
        const RouterQ8PrefillHardware &hardware) noexcept
    {
        if (local_capacity < 0 || width <= 0 || width % 32 || experts <= 0 || !hardware.valid())
            return {};
        const std::int64_t row_tiles = local_capacity ? (std::int64_t(local_capacity) + 15) / 16 : 1;
        const std::int64_t expert_tiles = (std::int64_t(experts) + 7) / 8;
        const std::int64_t tiles = row_tiles * expert_tiles;
        const std::int64_t block_capacity = std::int64_t(hardware.multiprocessors) * hardware.block_slots_per_sm;
        const int constant_width = width <= 8192 && (width & (width - 1)) == 0 ? width : 0;
        int k_warps = 1;
        while (k_warps < 16 && tiles * k_warps * 8 < block_capacity * 5 && width / 32 >= 2 * k_warps)
            k_warps *= 2;
        if (k_warps > 1) return {16, 8, k_warps, k_warps, constant_width};

        const int warps = tiles > block_capacity * 2 ? 4 : tiles > block_capacity ? 2 : 1;
        constexpr int shapes[][2] = {{1,1}, {1,2}, {2,1}, {1,4}, {2,2}, {4,1}};
        RouterQ8PrefillGeometry best;
        std::int64_t best_padding = 0;
        int best_operands = 0;
        for (const auto &shape : shapes)
        {
            const int r = shape[0], c = shape[1];
            if (r * c != warps) continue;
            const auto padded_warps = ((row_tiles + r - 1) / r) * ((expert_tiles + c - 1) / c) * warps;
            const int operands = 16 * r + 8 * c;
            if (!best.warps || padded_warps < best_padding ||
                (padded_warps == best_padding && operands < best_operands))
            {
                best = {16 * r, 8 * c, warps, 1, constant_width};
                best_padding = padded_warps;
                best_operands = operands;
            }
        }
        return best;
    }

    /** @brief Enqueue one tensor-core logits kernel without allocation or queries.
     * @param view Frozen operands, geometry, hardware and device-owned live count.
     * @param stream Exact non-null producer stream. @return Validity and launch success. */
    bool routerQ8Prefill(const RouterQ8PrefillView &view, void *stream);
}

extern "C"
{
    /** @brief Quantize, compute tensor-core logits, then publish original expert rows.
     * @param hidden Original FP32 input. @param hidden_codes Admitted Q8 bank.
     * @param hidden_scales Admitted block32 scales. @param gate_codes Immutable gate.
     * @param gate_scales Its block32 scales. @param logits FP32 router output.
     * @param rows Captured capacity. @param width Block32 width. @param experts Output width.
     * @param device Owning ordinal. @param stream Exact non-null stream.
     * @param live_rows Optional device live extent. @param basis Paired gate/input basis.
     * @param hardware Immutable matching device observation prepared before capture.
     * @return Whether every ordered launch succeeded. */
    bool cudaMoE_gate_logits_q8_weights_prefill_rows(const float *hidden,
        std::int8_t *hidden_codes, float *hidden_scales, const std::int8_t *gate_codes,
        const float *gate_scales, float *logits, int rows, int width, int experts,
        int device, void *stream, const int *live_rows, llaminar2::RouterQ8Basis basis,
        const llaminar2::cuda::RouterQ8PrefillHardware &hardware);
}
