/**
 * @file Test__CPUNativeBlockDecode.cpp
 * @brief Independent byte oracles for source-order CPU NativeVNNI decode and encoding policy.
 *
 * Complete nibble/high-bit domains and coordinate-sensitive fixtures exercise
 * the production native decoder, without tensor/model allocation or accelerator
 * state. Source format and prepared encoding remain distinct: the shared
 * registry supplies all source identities, including source-scale aliases.
 */
#include "kernels/cpu/gemm/CPUNativeVNNIDecode.h"
#include "kernels/cpu/gemm/CPUNativeVNNIEncoding.h"

#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <set>
#include <string>

namespace llaminar2::test
{
using namespace cpu::native_vnni;

/** @test All low/high nibble pairs preserve signed linear and nonlinear values. */
TEST(CPUNativeBlockDecode, EveryNibblePairMatchesIndependentIntegerOracle)
{
    for (int raw = 0; raw < 256; ++raw)
    {
        std::array<std::uint8_t, 16> payload;
        payload.fill(static_cast<std::uint8_t>(raw));
        for (const int codebook : {0, 4, 5, int(kNativeVnniMXFP4Codebook)})
        {
            std::array<std::int8_t, 32> actual;
            decode_native_block(static_cast<std::uint8_t>(codebook), payload.data(), actual.data());
            for (int lane = 0; lane < 32; ++lane)
            {
                const int index = lane < 16 ? raw & 15 : raw >> 4;
                // MXFP4: sign bit 3, doubled OCP E2M1 magnitude in bits 0..2.
                static constexpr int kE2M1Doubled[8] = {0, 1, 2, 3, 4, 6, 8, 12};
                const int expected = codebook == kNativeVnniMXFP4Codebook
                    ? ((index & 8) ? -kE2M1Doubled[index & 7] : kE2M1Doubled[index & 7])
                    : codebook == 4 ? static_cast<int>(kvalues_iq4nl[index])
                    : codebook == 0 ? index - 8 : index;
                ASSERT_EQ(int(actual[lane]), expected)
                    << "codebook=" << codebook << " raw=" << raw << " lane=" << lane;
            }
        }
        std::array<std::int8_t, 32> xs, nl;
        decode_iq4_xs_block(payload.data(), xs.data());
        decode_iq4_nl_block(payload.data(), nl.data());
        ASSERT_EQ(xs, nl);
    }
}

/** @test Every high-bit byte/nibble pair retains signedness and original lane ordering. */
TEST(CPUNativeBlockDecode, FiveBitExhaustiveBytesMatchIndependentIntegerOracle)
{
    for (int raw = 0; raw < 256; ++raw)
        for (int high = 0; high < 256; ++high)
        {
            std::array<std::uint8_t, 20> payload;
            for (int lane = 0; lane < 16; ++lane)
                payload[lane] = static_cast<std::uint8_t>(raw + lane * 17);
            payload[16] = static_cast<std::uint8_t>(high);
            payload[17] = static_cast<std::uint8_t>(high ^ 255);
            payload[18] = static_cast<std::uint8_t>((high << 1) | (high >> 7));
            payload[19] = static_cast<std::uint8_t>(127 - high);
            for (const auto codebook : {6, 7})
            {
                std::array<std::int8_t, 32> actual;
                decode_native_block(static_cast<std::uint8_t>(codebook), payload.data(), actual.data());
                for (int lane = 0; lane < 32; ++lane)
                {
                    const int nibble = lane < 16 ? payload[lane] & 15 : payload[lane - 16] >> 4;
                    const int high_bit = (payload[16 + lane / 8] >> (lane % 8)) & 1;
                    const int expected = nibble + 16 * high_bit - (codebook == 6 ? 16 : 0);
                    ASSERT_EQ(int(actual[lane]), expected)
                        << "codebook=" << codebook << " raw=" << raw
                        << " high=" << high << " lane=" << lane;
                }
            }
        }
}

/** @test The canonical registry remains total over six physical encodings and source tuples. */
TEST(CPUNativeBlockDecode, AllSourceFormatsRetainTheirPreparedEncodingIdentity)
{
    std::set<CPUNativeVNNIEncoding> encodings;
    for (const auto &source : native_vnni_formats::kAllSourceFormats)
    {
        SCOPED_TRACE(std::string(source.quant_type));
        const auto &format = *source.metadata;
        const auto encoding = preparedEncodingForSource(format);
        EXPECT_TRUE(isKnownPreparedEncoding(encoding));
        EXPECT_GT(preparedDataStride(encoding), 0u);
        EXPECT_GE(preparedInterleavedBlockStride(encoding, format.is_asymmetric),
                  preparedDataStride(encoding) + 256u);
        EXPECT_EQ(native_vnni_formats::forSourceIdentity(format.codebook_id, format.is_superblock),
                  source.metadata);
        EXPECT_EQ(preparedEncodingForSource(format, true), CPUNativeVNNIEncoding::ExpandedInt8);
        encodings.insert(encoding);
    }
    EXPECT_EQ(encodings.size(), 6u);
    EXPECT_EQ(preparedEncodingForSource(native_vnni_formats::IQ4_NL), CPUNativeVNNIEncoding::NibbleLUT);
    EXPECT_EQ(preparedEncodingForSource(native_vnni_formats::IQ4_XS), CPUNativeVNNIEncoding::NibbleLUTSourceScale);
    EXPECT_EQ(preparedEncodingForSource(native_vnni_formats::IQ3_S), CPUNativeVNNIEncoding::IQ3SExpandedSourceScale);
}
} // namespace llaminar2::test
