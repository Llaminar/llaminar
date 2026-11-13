/**
 * @file RoPEAngleRows.h
 * @brief Borrowed absolute-position angle planes for one CPU RoPE invocation.
 *
 * CPU RoPE never owns angle storage. The stage binds one admitted
 * `cpu_rope_angles` workspace buffer, sized from the graph's maximum rows and
 * the physical head width, and every invocation carves its live rows from the
 * start of that buffer. Rows are independent functions of their absolute
 * position, so a shared buffer filled by any partition of one team holds the
 * same bytes as a serial fill. Nothing here allocates, resizes or caches.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>

namespace llaminar2::primitives
{
    /**
     * @brief Live angle planes for `rows` positions of `values_per_row` pairs.
     *
     * The Q15 planes serve only the Q8_1 integer format, whose own angle
     * authority truncates the same FP32 values. `active` marks rows whose
     * position is nonnegative; padding rows keep their storage untouched.
     */
    struct RoPEAngleRows
    {
        std::span<float> cos;         ///< rows x values_per_row cosines.
        std::span<float> sin;         ///< rows x values_per_row sines.
        std::span<int16_t> cos_q15;   ///< rows x values_per_row truncated Q15 cosines.
        std::span<int16_t> sin_q15;   ///< rows x values_per_row truncated Q15 sines.
        std::span<uint8_t> active;    ///< One flag per row.
        int rows = 0;                 ///< Live rows carved for this invocation.
        int values_per_row = 0;       ///< Rotary pairs per row (rotary width / 2).

        /** @return First cosine of one row. */
        float *cos_row(int row) const noexcept { return cos.data() + std::size_t(row) * values_per_row; }
        /** @return First sine of one row. */
        float *sin_row(int row) const noexcept { return sin.data() + std::size_t(row) * values_per_row; }
        /** @return First Q15 cosine of one row. */
        int16_t *cos_q15_row(int row) const noexcept { return cos_q15.data() + std::size_t(row) * values_per_row; }
        /** @return First Q15 sine of one row. */
        int16_t *sin_q15_row(int row) const noexcept { return sin_q15.data() + std::size_t(row) * values_per_row; }
    };

    /** @brief Byte offsets of each plane inside one carved angle buffer. */
    struct RoPEAngleLayout
    {
        std::size_t cos = 0;      ///< Offset of the FP32 cosine plane.
        std::size_t sin = 0;      ///< Offset of the FP32 sine plane.
        std::size_t cos_q15 = 0;  ///< Offset of the Q15 cosine plane.
        std::size_t sin_q15 = 0;  ///< Offset of the Q15 sine plane.
        std::size_t active = 0;   ///< Offset of the per-row active flags.
        std::size_t bytes = 0;    ///< Complete extent, including plane alignment.
    };

    /** @brief Alignment of every plane; one cache line keeps planes independent. */
    inline constexpr std::size_t kRoPEAnglePlaneAlignment = 64;

    /**
     * @brief Compute the checked plane layout for one angle geometry.
     * @param rows Positive live rows.
     * @param values_per_row Positive rotary pairs per row.
     * @return Offsets and total extent; monotone in both arguments.
     * @throws std::invalid_argument On nonpositive or overflowing geometry.
     */
    inline RoPEAngleLayout rope_angle_layout(int rows, int values_per_row)
    {
        if (rows <= 0 || values_per_row <= 0)
            throw std::invalid_argument("CPU RoPE angle rows require positive rows and rotary pairs");
        const std::size_t count = std::size_t(rows) * std::size_t(values_per_row);
        if (count / std::size_t(rows) != std::size_t(values_per_row) ||
            count > std::numeric_limits<std::size_t>::max() / 16)
            throw std::invalid_argument("CPU RoPE angle extent overflows");
        const auto align = [](std::size_t offset)
        { return (offset + kRoPEAnglePlaneAlignment - 1) / kRoPEAnglePlaneAlignment * kRoPEAnglePlaneAlignment; };
        RoPEAngleLayout layout;
        layout.cos = 0;
        layout.sin = align(layout.cos + count * sizeof(float));
        layout.cos_q15 = align(layout.sin + count * sizeof(float));
        layout.sin_q15 = align(layout.cos_q15 + count * sizeof(int16_t));
        layout.active = align(layout.sin_q15 + count * sizeof(int16_t));
        layout.bytes = align(layout.active + std::size_t(rows));
        return layout;
    }

    /**
     * @brief Carve live angle planes from the start of one admitted buffer.
     * @param storage Borrowed workspace bytes; never allocated or resized here.
     * @param rows Positive live rows of this invocation.
     * @param values_per_row Positive rotary pairs per row.
     * @return Non-owning planes valid while the workspace binding lives.
     * @throws std::invalid_argument When the buffer is absent, misaligned or too small.
     */
    inline RoPEAngleRows carve_rope_angle_rows(std::span<std::byte> storage, int rows, int values_per_row)
    {
        const RoPEAngleLayout layout = rope_angle_layout(rows, values_per_row);
        if (storage.empty())
            throw std::invalid_argument("CPU RoPE requires its admitted cpu_rope_angles workspace binding");
        if (reinterpret_cast<std::uintptr_t>(storage.data()) % kRoPEAnglePlaneAlignment != 0)
            throw std::invalid_argument("CPU RoPE angle workspace is not plane aligned");
        if (storage.size() < layout.bytes)
            throw std::invalid_argument(
                "CPU RoPE angle workspace holds " + std::to_string(storage.size()) + " bytes but " +
                std::to_string(rows) + " rows of " + std::to_string(values_per_row) + " pairs need " +
                std::to_string(layout.bytes));
        const std::size_t count = std::size_t(rows) * std::size_t(values_per_row);
        std::byte *base = storage.data();
        return RoPEAngleRows{
            .cos = {reinterpret_cast<float *>(base + layout.cos), count},
            .sin = {reinterpret_cast<float *>(base + layout.sin), count},
            .cos_q15 = {reinterpret_cast<int16_t *>(base + layout.cos_q15), count},
            .sin_q15 = {reinterpret_cast<int16_t *>(base + layout.sin_q15), count},
            .active = {reinterpret_cast<uint8_t *>(base + layout.active), std::size_t(rows)},
            .rows = rows,
            .values_per_row = values_per_row,
        };
    }

    /**
     * @brief Reject planes that cannot hold the requested geometry.
     * @param angles Carved planes supplied by the caller.
     * @param rows Rows the primitive will fill.
     * @param values_per_row Rotary pairs the primitive will fill.
     * @throws std::invalid_argument Before any store when the planes are too small.
     */
    inline void require_rope_angle_rows(const RoPEAngleRows &angles, int rows, int values_per_row)
    {
        if (rows <= 0 || values_per_row <= 0 || angles.rows < rows || angles.values_per_row != values_per_row ||
            angles.cos.size() < std::size_t(rows) * values_per_row ||
            angles.sin.size() < std::size_t(rows) * values_per_row ||
            angles.cos_q15.size() < std::size_t(rows) * values_per_row ||
            angles.sin_q15.size() < std::size_t(rows) * values_per_row ||
            angles.active.size() < std::size_t(rows))
            throw std::invalid_argument("CPU RoPE angle planes do not cover the requested rows and rotary pairs");
    }
}
