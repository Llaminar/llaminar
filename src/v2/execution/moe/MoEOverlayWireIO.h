/**
 * @file MoEOverlayWireIO.h
 * @brief Bounded, allocation-free little-endian scalars for overlay control packets.
 *
 * Packet owners supply storage and a cursor. These helpers neither allocate nor
 * retain a view, and a malformed extent never advances the cursor past storage.
 * Native struct padding and process-local pointer values are never serialized.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace llaminar2::moe_overlay_wire
{
    /** @brief Write one integral scalar after checking the complete destination extent. */
    template <typename Value>
    void writeLittleEndian(std::span<std::uint8_t> destination, std::size_t &offset, Value value)
    {
        static_assert(std::is_integral_v<Value> && !std::is_same_v<Value, bool>);
        if (offset > destination.size() || sizeof(Value) > destination.size() - offset)
            throw std::out_of_range("Overlay wire scalar exceeds destination storage");
        using Unsigned = std::make_unsigned_t<Value>;
        Unsigned bits = static_cast<Unsigned>(value);
        for (std::size_t byte = 0; byte < sizeof(Value); ++byte)
        {
            destination[offset++] = static_cast<std::uint8_t>(bits & 0xffu);
            bits >>= 8u;
        }
    }

    /** @return One integral scalar, rejecting a truncated packet before reading any byte. */
    template <typename Value>
    Value readLittleEndian(std::span<const std::uint8_t> packet, std::size_t &offset)
    {
        static_assert(std::is_integral_v<Value> && !std::is_same_v<Value, bool>);
        if (offset > packet.size() || sizeof(Value) > packet.size() - offset)
            throw std::out_of_range("Overlay wire scalar exceeds packet storage");
        using Unsigned = std::make_unsigned_t<Value>;
        Unsigned bits = 0;
        for (std::size_t byte = 0; byte < sizeof(Value); ++byte)
            bits |= static_cast<Unsigned>(packet[offset++]) << (byte * 8u);
        return static_cast<Value>(bits);
    }
}
