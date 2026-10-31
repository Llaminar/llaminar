/**
 * @file Test__InactiveOutputPoison.cpp
 * @brief Device-free replay of the actual retained GDN inactive-byte validator.
 *
 * Exercise every byte around chunk and element boundaries, exact NaN payloads,
 * live-prefix exclusion, first-mismatch ordering and large-to-small reset. The
 * same allocation-free helper is called by the native GPU capture fixture.
 */
#include "../../utils/InactiveOutputPoison.h"
#include <gtest/gtest.h>
#include <bit>
#include <vector>

namespace llaminar2::test
{
    /** @brief Restore exactly the byte poison used by the production capture fixture. */
    static void resetPoison(std::span<float> output)
    {
        std::memset(output.data(), 0xff, output.size_bytes());
    }

    TEST(InactiveOutputPoison, EmptyFullAndPartialSpans)
    {
        EXPECT_FALSE(findInactiveOutputPoisonMismatch({}, 0));
        std::vector<float> output(2059);
        resetPoison(output);
        for (size_t live : {size_t{0}, size_t{1}, size_t{3}, size_t{2058}, output.size()})
            EXPECT_FALSE(findInactiveOutputPoisonMismatch(output, live));
        std::memset(output.data(), 0, output.size() * sizeof(float));
        EXPECT_FALSE(findInactiveOutputPoisonMismatch(output, output.size()));
        EXPECT_THROW(findInactiveOutputPoisonMismatch(output, output.size() + 1),
            std::invalid_argument);
        EXPECT_THROW(findInactiveOutputPoisonMismatch({}, 1), std::invalid_argument);
    }

    TEST(InactiveOutputPoison, EveryInactiveByteReportsExactFirstElementAndByte)
    {
        std::vector<float> output(2059);
        for (size_t live : {size_t{0}, size_t{3}})
        {
            resetPoison(output);
            auto bytes = std::as_writable_bytes(std::span(output));
            for (size_t byte = live * sizeof(float); byte < bytes.size(); ++byte)
            {
                bytes[byte] = std::byte{0xfe};
                const auto mismatch = findInactiveOutputPoisonMismatch(output, live);
                ASSERT_TRUE(mismatch.has_value()) << "undetected byte " << byte;
                EXPECT_EQ(mismatch->element_index, byte / sizeof(float));
                EXPECT_EQ(mismatch->byte_in_element, byte % sizeof(float));
                uint32_t actual = 0;
                std::memcpy(&actual, output.data() + byte / sizeof(float), sizeof(actual));
                EXPECT_EQ(mismatch->actual_bits, actual);
                bytes[byte] = std::byte{0xff};
            }
        }
    }

    TEST(InactiveOutputPoison, LivePrefixIsExcludedAndEarliestDirtyByteWins)
    {
        std::vector<float> output(2059);
        resetPoison(output);
        auto bytes = std::as_writable_bytes(std::span(output));
        bytes[0] = std::byte{0};
        bytes[11] = std::byte{0};
        EXPECT_FALSE(findInactiveOutputPoisonMismatch(output, 3));
        // Descending corruptions cross both 4KiB chunk boundaries and a partial
        // final chunk. Every new first byte must replace the later diagnosis.
        for (size_t relative : {size_t{8223}, size_t{8192}, size_t{4096}, size_t{4095}, size_t{0}})
        {
            const size_t byte = 3 * sizeof(float) + relative;
            bytes[byte] = std::byte{0};
            const auto mismatch = findInactiveOutputPoisonMismatch(output, 3);
            ASSERT_TRUE(mismatch.has_value());
            EXPECT_EQ(mismatch->element_index, byte / sizeof(float));
            EXPECT_EQ(mismatch->byte_in_element, byte % sizeof(float));
        }
    }

    TEST(InactiveOutputPoison, NaNPayloadAndLastDirtyByteRemainByteExact)
    {
        std::vector<float> output(1027);
        for (uint32_t bits : {0x7fc00000u, 0xffc00000u, 0x7f800001u,
            0xff800001u, 0x7f800000u, 0xff800000u, 0u, 0x80000000u})
        {
            resetPoison(output);
            std::memcpy(&output.back(), &bits, sizeof(bits));
            const auto mismatch = findInactiveOutputPoisonMismatch(output, 1);
            ASSERT_TRUE(mismatch.has_value());
            EXPECT_EQ(mismatch->element_index, output.size() - 1);
            EXPECT_EQ(mismatch->actual_bits, bits);
        }
        resetPoison(output);
        auto bytes = std::as_writable_bytes(std::span(output));
        bytes.back() = std::byte{0x7f};
        const auto mismatch = findInactiveOutputPoisonMismatch(output, 1);
        ASSERT_TRUE(mismatch.has_value());
        EXPECT_EQ(mismatch->element_index, output.size() - 1);
        EXPECT_EQ(mismatch->byte_in_element, sizeof(float) - 1);
    }

    TEST(InactiveOutputPoison, ReplayLargeToSmallResetAndRepeatedCapacity)
    {
        std::vector<float> output(2081);
        for (int replay = 0; replay < 20; ++replay)
        {
            resetPoison(output);
            std::memset(output.data(), 0, 1000 * sizeof(float));
            EXPECT_FALSE(findInactiveOutputPoisonMismatch(output, 1000));
            const auto stale = findInactiveOutputPoisonMismatch(output, 3);
            ASSERT_TRUE(stale.has_value());
            EXPECT_EQ(stale->element_index, 3u);
            EXPECT_EQ(stale->byte_in_element, 0u);
            resetPoison(output);
            std::memset(output.data(), 0, 3 * sizeof(float));
            EXPECT_FALSE(findInactiveOutputPoisonMismatch(output, 3));
            resetPoison(output);
            EXPECT_FALSE(findInactiveOutputPoisonMismatch(output, 0));
        }
    }
}
