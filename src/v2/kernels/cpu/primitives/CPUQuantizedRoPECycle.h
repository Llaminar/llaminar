/**
 * @file CPUQuantizedRoPECycle.h
 * @brief Bounded-fragment ownership schedule for in-place quantized rotary heads.
 *
 * A physical block's new scale depends on both rotated halves. Even block counts
 * form independent two-block cycles; odd counts form one cycle through all
 * blocks. The schedule retains one anchor half, one carried half and one output
 * block, then retires a block only after both original halves were consumed.
 * This supports single/odd/even geometries without a decoded-head mirror,
 * allocation, source-scale shadow bank or overwritten input reuse. Native math
 * providers own the explicit architecture arithmetic; this file only wires the
 * original half-block coordinates and byte ownership.
 */
#pragma once
#include "tensors/BlockStructures.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace llaminar2::primitives
{
    /** @brief Retained quantized block families with complete native rotary programs. */
    template <typename Block>
    concept QuantizedRoPEBlock = std::is_same_v<Block, Q8_1Block> || std::is_same_v<Block, Q16_1Block> ||
        std::is_same_v<Block, Q16_1Block_64> || std::is_same_v<Block, Q16_1Block_128>;

    /**
     * @brief Rotate an original quantized head with at most two blocks of private FP32 fragments.
     * @tparam Block Original retained block encoding, including its complete physical size.
     * @tparam Math Native provider of rotateHalf() and requantize(); no scalar implementation is selected here.
     * @param head Original sole-authority in-place block bank.
     * @param blocks Complete nonnegative physical block count; zero is an empty leaf.
     * @param cosine Original Q15 cosine words for half the logical head.
     * @param sine Matching original Q15 sine words.
     * @throws std::invalid_argument Before mutation on malformed geometry or coefficient overlap.
     * @details A later provider failure is fatal, not rollback of already retired blocks.
     */
    template <QuantizedRoPEBlock Block, typename Math>
    inline void rotate_quantized_head_in_place(Block *head, int blocks,
        const std::int16_t *cosine, const std::int16_t *sine)
    {
        constexpr int width = static_cast<int>(Block::BLOCK_SIZE);
        constexpr int half = width / 2;
        static_assert(width == 32 || width == 64 || width == 128);
        if (blocks < 0 || blocks > std::numeric_limits<int>::max() / width ||
            (blocks && (!head || !cosine || !sine)))
            throw std::invalid_argument("CPU quantized RoPE requires complete representable head/coefficient geometry");
        if (!blocks) return;
        const auto bytes = static_cast<std::size_t>(blocks) * sizeof(Block);
        const auto coefficient_bytes = static_cast<std::size_t>(blocks) * half * sizeof(std::int16_t);
        for (const auto *table : {cosine, sine})
        {
            const auto x = reinterpret_cast<std::uintptr_t>(head), y = reinterpret_cast<std::uintptr_t>(table);
            if (x <= y ? y - x < bytes : x - y < coefficient_bytes)
                throw std::invalid_argument("CPU quantized RoPE writable blocks alias retained coefficients");
        }
        alignas(64) std::array<float, half> anchor{}, carry{};
        alignas(64) std::array<float, width> output{};
        const int cycles = blocks % 2 ? 1 : blocks / 2;
        const auto pair = [&](int first_unit, int second_unit, float *first_output, float *second_output) {
            Math::rotateHalf(head[first_unit / 2], (first_unit % 2) * half,
                head[second_unit / 2], (second_unit % 2) * half,
                cosine + static_cast<std::size_t>(first_unit) * half,
                sine + static_cast<std::size_t>(first_unit) * half,
                first_output, second_output, half);
        };
        for (int cycle = 0; cycle < cycles; ++cycle)
        {
            const int anchor_unit = cycle * 2;
            int carried_unit = anchor_unit + blocks;
            pair(anchor_unit, carried_unit, anchor.data(), carry.data());
            for (;;)
            {
                const int block = carried_unit / 2;
                const int carried_half = carried_unit % 2;
                const int other_unit = carried_unit ^ 1;
                std::memcpy(output.data() + carried_half * half, carry.data(), half * sizeof(float));
                float *other_output = output.data() + (1 - carried_half) * half;
                if (other_unit == anchor_unit)
                {
                    std::memcpy(other_output, anchor.data(), half * sizeof(float));
                    Math::requantize(output.data(), head[block]);
                    break;
                }
                const int partner = other_unit < blocks ? other_unit + blocks : other_unit - blocks;
                if (other_unit < blocks) pair(other_unit, partner, other_output, carry.data());
                else pair(partner, other_unit, carry.data(), other_output);
                // This block's two original halves have now both been consumed.
                // Its scale may be replaced without changing any future input.
                Math::requantize(output.data(), head[block]);
                carried_unit = partner;
            }
        }
    }
} // namespace llaminar2::primitives
