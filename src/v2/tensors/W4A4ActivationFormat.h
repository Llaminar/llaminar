/**
 * @file W4A4ActivationFormat.h
 * @brief Packed group-64 A4 operands, distinct from immutable W4A4 source blocks.
 *
 * Activations carry one whole-token FP32 unit, four asymmetric zero points and
 * four integer group multipliers per block. They intentionally retain the
 * pinned native 144-byte ABI so recorded operand bytes can be checked directly.
 * Their metadata has different meanings from weight metadata: using a distinct
 * type prevents a weight descriptor from becoming an activation scale policy.
 */
#pragma once

#include "W4A4Format.h"

namespace llaminar2::w4a4
{
    /** @brief Group-64 subdivision tag in the native activation descriptor. */
    inline constexpr std::uint32_t kActivationGroupTag = 2u << 16;

    /** @brief One prepared A4 block; the invocation's admitted arena owns its bytes. */
    struct ActivationBlock
    {
        float unit;                      ///< Whole-token unit, repeated exactly in every block.
        std::uint32_t zero_points;       ///< Group subdivision tag and four 4-bit zero points.
        std::uint32_t multipliers;       ///< Four integer group multipliers in [1,255].
        std::int32_t rotation_length;    ///< Original row/transform width retained for replay.
        std::uint8_t codes[128];         ///< Adjacent signed nibbles representing q minus eight.
    };
    static_assert(sizeof(ActivationBlock) == kSourceBlockBytes);
    static_assert(offsetof(ActivationBlock, codes) == 16);
    static_assert(std::is_standard_layout_v<ActivationBlock> && std::is_trivially_copyable_v<ActivationBlock>);
} // namespace llaminar2::w4a4
