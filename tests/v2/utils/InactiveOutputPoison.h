/**
 * @file InactiveOutputPoison.h
 * @brief Allocation-free byte validation of retained FP32 output poison spans.
 *
 * The capture fixture fills output capacity with 0xff before every replay. All
 * inactive bytes must remain exactly 0xff, including NaN payload bits. Compare
 * against one cache-resident constant in bounded chunks so libc can vectorize
 * the successful scan; only a failed chunk needs scalar first-byte diagnosis.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>

namespace llaminar2::test
{
    /** @brief Exact first changed byte in the original output element coordinates. */
    struct InactiveOutputPoisonMismatch
    {
        size_t element_index; ///< Absolute element index, including the live prefix.
        size_t byte_in_element; ///< First changed byte within that element.
        uint32_t actual_bits; ///< Complete observed representation, without FP evaluation.
    };

    /**
     * @brief Check every inactive output byte without allocating or evaluating NaNs.
     * @param output Complete copied-back output capacity, including its live prefix.
     * @param live_elements Number of live elements excluded from this poison check.
     * @return Empty only when every inactive byte is 0xff; otherwise the exact
     * first changed element, byte and observed element bits.
     * @throws std::invalid_argument When the supplied live extent exceeds capacity.
     */
    inline std::optional<InactiveOutputPoisonMismatch> findInactiveOutputPoisonMismatch(
        std::span<const float> output, size_t live_elements)
    {
        static_assert(sizeof(float) == sizeof(uint32_t));
        if (live_elements > output.size())
            throw std::invalid_argument("Inactive poison live extent exceeds output capacity");

        // This immutable reference is shared by all calls. Neither clean replay
        // nor mismatch diagnosis creates a temporary capacity-sized allocation.
        static constexpr auto reference = [] {
            std::array<std::byte, 4096> bytes{};
            bytes.fill(std::byte{0xff});
            return bytes;
        }();
        const auto inactive = std::as_bytes(output.subspan(live_elements));
        for (size_t offset = 0; offset < inactive.size();)
        {
            const size_t count = std::min(reference.size(), inactive.size() - offset);
            if (std::memcmp(inactive.data() + offset, reference.data(), count) != 0)
            {
                const auto chunk = inactive.subspan(offset, count);
                const auto changed = std::find_if(chunk.begin(), chunk.end(),
                    [](std::byte value) { return value != std::byte{0xff}; });
                const size_t byte = offset + size_t(changed - chunk.begin());
                const size_t element = live_elements + byte / sizeof(float);
                uint32_t actual = 0;
                std::memcpy(&actual, output.data() + element, sizeof(actual));
                return InactiveOutputPoisonMismatch{
                    element, byte % sizeof(float), actual};
            }
            offset += count;
        }
        return std::nullopt;
    }
}
