/**
 * @file Test__NativeQuantisedSuperblockMetadata.cpp
 * @brief Exhaustive proof that superblock-coded GPU metadata reproduces the legacy staged words.
 *
 * The legacy GPU representation stores per-32 FP16 words computed by the VNNI
 * repack kernels. The superblock-coded representation stores the source scale
 * code per block and `d`/`dmin` per superblock, and reconstructs the words at
 * load time. Here every code value is combined with every binary16 `d` (and,
 * for Q2_K, `dmin`) bit pattern and the reconstruction is compared bit for bit
 * with the literal repack expression. The reference widens binary16 through
 * the compiler's `_Float16`, independent of the codec's integer widening.
 * Address helpers and plane extents are checked for both matrix layouts,
 * nonzero superblock phases and ragged K.
 */
#include "tensors/NativeQuantisedSuperblockMetadata.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <set>

namespace llaminar2::native_quantised_superblock::test
{
    namespace
    {
        /** @brief Hardware/compiler binary16 widening (the reference for __half2float). */
        float refWiden(std::uint16_t bits) { return static_cast<float>(std::bit_cast<_Float16>(bits)); }

        std::uint16_t cph(float v) { return canonicalPreparedHalfBits(v); }

        /**
         * @brief Equality of half words where a NaN's sign is unspecified.
         *
         * `-dmin * 0` with an infinite `dmin` is a NaN whose sign depends on
         * whether the compiler negates the operand or the product (both legal);
         * the repack kernels have the same freedom. Every non-NaN word,
         * including signed zeros and infinities, must match exactly.
         */
        bool sameHalf(std::uint16_t a, std::uint16_t b)
        {
            const auto nan = [](std::uint16_t h) { return (h & 0x7C00u) == 0x7C00u && (h & 0x03FFu) != 0; };
            return nan(a) && nan(b) ? (a & 0x7FFFu) == (b & 0x7FFFu) : a == b;
        }

        /** @brief Every binary16 bit pattern. */
        template <class F>
        void forEachHalf(F &&f)
        {
            for (std::uint32_t bits = 0; bits <= 0xFFFFu; ++bits)
                f(static_cast<std::uint16_t>(bits));
        }
    } // namespace

    TEST(NativeQuantisedSuperblockMetadata, WideningMatchesHardwareForEveryHalf)
    {
        forEachHalf([](std::uint16_t h) {
            const float a = halfBitsToFloat(h), b = refWiden(h);
            if (b != b)
            {
                // NaN: payload equality up to the quiet bit (arithmetic quiets both).
                ASSERT_TRUE(a != a) << std::hex << h;
                ASSERT_EQ(std::bit_cast<std::uint32_t>(a) | 0x00400000u, std::bit_cast<std::uint32_t>(b) | 0x00400000u)
                    << std::hex << h;
            }
            else
                ASSERT_EQ(std::bit_cast<std::uint32_t>(a), std::bit_cast<std::uint32_t>(b)) << std::hex << h;
        });
    }

    TEST(NativeQuantisedSuperblockMetadata, CodeAndFieldWidthsCarryExactlyTheSourceBits)
    {
        // Codebooks 5/7 are shared with Q4_1/Q5_1 and 4 is IQ4_NL: never coded.
        for (const std::uint8_t cb : {0, 4, 5, 6, 7, 19, 20, 21, 23, 25})
            EXPECT_FALSE(isSuperblockCoded(cb)) << int(cb);
        for (const std::uint8_t cb : {11, 12, 13, 14, 15, 16, 17, 24})
            EXPECT_EQ(superblockCodeBytes(cb), 1u) << int(cb);
        for (const std::uint8_t cb : {8, 9, 10})
            EXPECT_EQ(superblockCodeBytes(cb), 2u) << int(cb);
        EXPECT_EQ(superblockFieldBytes(10), 4u);
        for (const std::uint8_t cb : {8, 9, 11, 12, 13, 14, 15, 16, 17, 24})
            EXPECT_EQ(superblockFieldBytes(cb), 2u) << int(cb);
    }

    /** IQ2_XXS (15): primary = half(d (0.5 + s) / 4); s = aux32_1 >> 28. */
    TEST(NativeQuantisedSuperblockMetadata, IQ2XXSReproducesTheRepackScale)
    {
        forEachHalf([](std::uint16_t d) {
            const float df = refWiden(d);
            for (std::uint32_t s = 0; s < 16; ++s)
            {
                const auto m = decodeSuperblockMetadata<15>(s, packSuperblockWord(d));
                ASSERT_EQ(m.primary, cph(df * (0.5f + static_cast<float>(s)) * 0.25f)) << d << ' ' << s;
                ASSERT_EQ(m.secondary, 0u);
                ASSERT_EQ(m.emin, 0u);
            }
        });
    }

    /** IQ2_XS (14): two half scales from the low/high nibble of the source scale byte. */
    TEST(NativeQuantisedSuperblockMetadata, IQ2XSReproducesBothRepackScales)
    {
        forEachHalf([](std::uint16_t d) {
            const float df = refWiden(d);
            for (std::uint32_t sc = 0; sc < 256; ++sc)
            {
                const auto m = decodeSuperblockMetadata<14>(sc, packSuperblockWord(d));
                ASSERT_EQ(m.primary, cph(df * (0.5f + static_cast<float>(sc & 0xF)) * 0.25f)) << d << ' ' << sc;
                ASSERT_EQ(m.secondary, cph(df * (0.5f + static_cast<float>(sc >> 4)) * 0.25f)) << d << ' ' << sc;
            }
        });
    }

    /** IQ3_XXS (12): primary = half(d (0.5 + s) / 2). */
    TEST(NativeQuantisedSuperblockMetadata, IQ3XXSReproducesTheRepackScale)
    {
        forEachHalf([](std::uint16_t d) {
            const float df = refWiden(d);
            for (std::uint32_t s = 0; s < 16; ++s)
                ASSERT_EQ(decodeSuperblockMetadata<12>(s, packSuperblockWord(d)).primary,
                          cph(df * (0.5f + static_cast<float>(s)) * 0.5f))
                    << d << ' ' << s;
        });
    }

    /** IQ1_S (16): dl = d (2 sel + 1); secondary = half(dl * +-1/8); code = qh >> 12. */
    TEST(NativeQuantisedSuperblockMetadata, IQ1SReproducesScaleAndDelta)
    {
        forEachHalf([](std::uint16_t d) {
            const float df = refWiden(d);
            for (std::uint32_t qh_high = 0; qh_high < 16; ++qh_high)
            {
                const std::uint16_t qh = static_cast<std::uint16_t>(qh_high << 12);
                const float dl = df * (2.0f * static_cast<float>((qh >> 12) & 7) + 1.0f);
                const float delta = (qh & 0x8000) ? -0.125f : 0.125f;
                const auto m = decodeSuperblockMetadata<16>(qh_high, packSuperblockWord(d));
                ASSERT_EQ(m.primary, cph(dl)) << d << ' ' << qh_high;
                ASSERT_EQ(m.secondary, cph(dl * delta)) << d << ' ' << qh_high;
            }
        });
    }

    /** IQ1_M (17): two odd multipliers of the reconstructed d; code = lo | hi << 3. */
    TEST(NativeQuantisedSuperblockMetadata, IQ1MReproducesBothScales)
    {
        forEachHalf([](std::uint16_t d) {
            const float df = refWiden(d);
            for (std::uint32_t code = 0; code < 64; ++code)
            {
                const auto m = decodeSuperblockMetadata<17>(code, packSuperblockWord(d));
                ASSERT_EQ(m.primary, cph(df * (2.0f * static_cast<float>(code & 7) + 1.0f))) << d << ' ' << code;
                ASSERT_EQ(m.secondary, cph(df * (2.0f * static_cast<float>(code >> 3) + 1.0f))) << d << ' ' << code;
            }
        });
    }

    /** Q3_K (9): half(d (s - 32)) for the two unpacked 6-bit scales. */
    TEST(NativeQuantisedSuperblockMetadata, Q3KReproducesBothScales)
    {
        forEachHalf([](std::uint16_t d) {
            const float df = refWiden(d);
            for (std::uint32_t lo = 0; lo < 64; ++lo)
            {
                const std::uint32_t hi = (lo * 37u + 11u) & 63u; // every hi value is also covered
                const auto m = decodeSuperblockMetadata<9>(lo | (hi << 6), packSuperblockWord(d));
                ASSERT_EQ(m.primary, cph(df * static_cast<float>(static_cast<int>(lo) - 32))) << d << ' ' << lo;
                ASSERT_EQ(m.secondary, cph(df * static_cast<float>(static_cast<int>(hi) - 32))) << d << ' ' << hi;
            }
        });
    }

    /** Q2_K (10): scales from d, packed effective minima from dmin. */
    TEST(NativeQuantisedSuperblockMetadata, Q2KReproducesScalesAndEffectiveMinima)
    {
        // Scales depend on (d, low nibbles); minima on (dmin, high nibbles).
        // Sweep each against every half with the other field fixed.
        forEachHalf([](std::uint16_t d) {
            const float df = refWiden(d);
            for (std::uint32_t nibbles = 0; nibbles < 256; ++nibbles)
            {
                const std::uint32_t lo = (nibbles & 0xF) | 0x50, hi = (nibbles >> 4) | 0xA0;
                const auto m = decodeSuperblockMetadata<10>(lo | (hi << 8), packSuperblockWord(d, 0x3C00));
                ASSERT_EQ(m.primary, cph(df * static_cast<float>(lo & 0xF))) << d << ' ' << nibbles;
                ASSERT_EQ(m.secondary, cph(df * static_cast<float>(hi & 0xF))) << d << ' ' << nibbles;
            }
        });
        forEachHalf([](std::uint16_t dmin) {
            const float mf = refWiden(dmin);
            for (std::uint32_t nibbles = 0; nibbles < 256; ++nibbles)
            {
                const std::uint32_t lo = ((nibbles & 0xF) << 4) | 0x3, hi = (nibbles & 0xF0) | 0xC;
                const auto m = decodeSuperblockMetadata<10>(lo | (hi << 8), packSuperblockWord(0x3C00, dmin));
                const std::uint32_t expected = static_cast<std::uint32_t>(cph(-mf * static_cast<float>(lo >> 4))) |
                                               (static_cast<std::uint32_t>(cph(-mf * static_cast<float>(hi >> 4))) << 16);
                ASSERT_TRUE(sameHalf(static_cast<std::uint16_t>(m.emin), static_cast<std::uint16_t>(expected)) &&
                            sameHalf(static_cast<std::uint16_t>(m.emin >> 16), static_cast<std::uint16_t>(expected >> 16)))
                    << dmin << ' ' << nibbles << ' ' << m.emin << ' ' << expected;
            }
        });
    }

    /** Source-scale formats keep d verbatim and the coefficient zero-extended. */
    TEST(NativeQuantisedSuperblockMetadata, SourceScaleFormatsKeepOriginalWords)
    {
        for (std::uint32_t d = 0; d <= 0xFFFFu; d += 257)
        {
            for (std::uint32_t c = 0; c < 256; ++c)
            {
                const auto iq2s = decodeSuperblockMetadata<13>(c, d);
                EXPECT_EQ(iq2s.primary, d);
                EXPECT_EQ(iq2s.secondary, native_quantised_iq2s_scale::pack(static_cast<std::uint8_t>(c)));
            }
            for (std::uint32_t c = 0; c < 16; ++c)
                EXPECT_EQ(decodeSuperblockMetadata<11>(c, d).secondary,
                          native_quantised_iq3s_scale::encodeSourceCoefficient(static_cast<std::uint8_t>(c)));
            for (std::uint32_t c = 0; c < 64; ++c)
                EXPECT_EQ(decodeSuperblockMetadata<24>(c, d).secondary,
                          native_quantised_iq4xs_scale::encodeSourceCoefficient(static_cast<std::uint8_t>(c)));
            for (std::uint32_t c = 0; c < 0x10000u; c += 97)
            {
                const auto q6 = decodeSuperblockMetadata<8>(c, d);
                EXPECT_EQ(q6.primary, d);
                EXPECT_EQ(q6.secondary, native_quantised_q6_scale::pack(static_cast<std::int8_t>(c & 0xFF),
                                                                        static_cast<std::int8_t>(c >> 8)));
            }
        }
    }

    /** The runtime-codebook dispatcher equals the templated decoder. */
    TEST(NativeQuantisedSuperblockMetadata, RuntimeDispatchMatchesTemplates)
    {
        for (const std::uint8_t cb : {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 24})
            for (std::uint32_t code = 0; code < 1024; code += 7)
            {
                const auto a = decodeSuperblockMetadata(cb, code, 0x2E66BC00u);
                StagedBlockMetadata b;
                switch (cb)
                {
                case 8: b = decodeSuperblockMetadata<8>(code, 0x2E66BC00u); break;
                case 9: b = decodeSuperblockMetadata<9>(code, 0x2E66BC00u); break;
                case 10: b = decodeSuperblockMetadata<10>(code, 0x2E66BC00u); break;
                case 11: b = decodeSuperblockMetadata<11>(code, 0x2E66BC00u); break;
                case 12: b = decodeSuperblockMetadata<12>(code, 0x2E66BC00u); break;
                case 13: b = decodeSuperblockMetadata<13>(code, 0x2E66BC00u); break;
                case 14: b = decodeSuperblockMetadata<14>(code, 0x2E66BC00u); break;
                case 15: b = decodeSuperblockMetadata<15>(code, 0x2E66BC00u); break;
                case 16: b = decodeSuperblockMetadata<16>(code, 0x2E66BC00u); break;
                case 17: b = decodeSuperblockMetadata<17>(code, 0x2E66BC00u); break;
                default: b = decodeSuperblockMetadata<24>(code, 0x2E66BC00u); break;
                }
                EXPECT_EQ(a.primary, b.primary);
                EXPECT_EQ(a.secondary, b.secondary);
                EXPECT_EQ(a.emin, b.emin);
            }
        EXPECT_EQ(decodeSuperblockMetadata(4, 3, 0x3C00).primary, 0u) << "IQ4_NL is not superblock-coded";
    }

    /**
     * Address helpers: every block of every row maps to the superblock that
     * holds its source `d`, for both layouts, every phase and ragged K, and the
     * superblock index space is dense and exactly planeBytes() wide.
     */
    TEST(NativeQuantisedSuperblockMetadata, AddressesCoverDenseSuperblockPlanesForEveryPhase)
    {
        for (std::uint32_t phase = 0; phase < kBlocksPerSuperblock; ++phase)
            for (const std::uint32_t kb : {1u, 3u, 8u, 9u, 15u, 16u, 21u, 64u})
                for (const std::uint32_t rows : {1u, 5u, 64u})
                {
                    const std::uint32_t spr = superblocksPerRow(kb, phase);
                    std::set<std::size_t> bm, rm;
                    for (std::uint32_t r = 0; r < rows; ++r)
                        for (std::uint32_t b = 0; b < kb; ++b)
                        {
                            const auto a = blockMajorAddress(b, r, rows, phase);
                            const auto c = rowMajorAddress(b, r, kb, phase);
                            ASSERT_EQ(a.linear, static_cast<std::size_t>(b) * rows + r);
                            ASSERT_EQ(c.linear, static_cast<std::size_t>(r) * kb + b);
                            // Same source superblock <=> same superblock index.
                            const std::uint32_t source_sb = (b + phase) / kBlocksPerSuperblock;
                            ASSERT_EQ(a.superblock, static_cast<std::size_t>(source_sb) * rows + r);
                            ASSERT_EQ(c.superblock, static_cast<std::size_t>(r) * spr + source_sb);
                            bm.insert(a.superblock);
                            rm.insert(c.superblock);
                        }
                    const std::size_t entries = static_cast<std::size_t>(rows) * spr;
                    ASSERT_EQ(bm.size(), entries);
                    ASSERT_EQ(rm.size(), entries);
                    ASSERT_EQ(*bm.rbegin(), entries - 1);
                    ASSERT_EQ(*rm.rbegin(), entries - 1);
                    const auto planes = planeBytes(10, rows, kb, phase);
                    EXPECT_EQ(planes.code, static_cast<std::size_t>(rows) * kb * 2u);
                    EXPECT_EQ(planes.superblock, entries * 4u);
                }
    }

    /** loadBlockMetadata composes the plane loads with the decoder. */
    TEST(NativeQuantisedSuperblockMetadata, PlaneLoadsComposeWithTheDecoder)
    {
        const std::uint8_t codes8[4] = {0x00, 0x1F, 0xF1, 0x7A};
        const std::uint16_t codes16[2] = {0x1234, 0xABCD};
        const std::uint16_t d16[2] = {0x3C00, 0x2C00};
        const std::uint32_t d32[1] = {0x40003800u};
        for (std::size_t i = 0; i < 4; ++i)
        {
            const auto got = loadBlockMetadata<13>(codes8, d16, BlockAddress{i, i / 2});
            const auto want = decodeSuperblockMetadata<13>(codes8[i], d16[i / 2]);
            EXPECT_EQ(got.primary, want.primary);
            EXPECT_EQ(got.secondary, want.secondary);
        }
        const auto q2 = loadBlockMetadata<10>(codes16, d32, BlockAddress{1, 0});
        const auto q2_want = decodeSuperblockMetadata<10>(0xABCD, 0x40003800u);
        EXPECT_EQ(q2.primary, q2_want.primary);
        EXPECT_EQ(q2.emin, q2_want.emin);
    }

    /**
     * Code-in-payload formats: parity restoration is ksigns_iq2xs for every
     * seven-bit selector and idempotent on per-block masks; embedding then
     * extracting returns every code while restoring parity (and IQ2_XS's spare
     * qh nibble) recovers the original payload byte for byte.
     */
    TEST(NativeQuantisedSuperblockMetadata, CodeInPayloadRoundTripsAndRestoresSigns)
    {
        for (std::uint32_t index = 0; index < 128; ++index)
        {
            const std::uint32_t parity = static_cast<std::uint32_t>(std::popcount(index)) & 1u;
            const auto mask = static_cast<std::uint8_t>(index | (parity << 7));
            EXPECT_EQ(restoreSignParity(mask), mask);
            EXPECT_EQ(restoreSignParity(static_cast<std::uint8_t>(mask ^ 0x80u)), mask);
        }
        EXPECT_EQ(superblockCodePlaneBytes(12), 0u);
        EXPECT_EQ(superblockCodePlaneBytes(13), 1u);
        EXPECT_EQ(superblockCodePlaneBytes(10), 2u);
        const auto check = [&]<std::uint8_t Codebook>(std::uint32_t codes) {
            constexpr std::uint32_t bytes = superblockCodePayloadBytes(Codebook);
            for (std::uint32_t seed = 0; seed < 64; ++seed)
                for (std::uint32_t code = 0; code < codes; ++code)
                {
                    std::uint8_t legacy[bytes];
                    for (std::uint32_t i = 0; i < bytes; ++i)
                        legacy[i] = static_cast<std::uint8_t>(seed * 131u + i * 29u + 7u);
                    // Legacy invariants: parity-correct sign masks, IQ2_XS qh upper nibble zero,
                    // IQ1_S qh bits 12..15 hold the code.
                    const int signs = Codebook == 12 ? 8 : Codebook == 15 ? 4 : Codebook == 14 ? 5 : -1;
                    if (signs >= 0)
                        for (int i = 0; i < 4; ++i) legacy[signs + i] = restoreSignParity(legacy[signs + i]);
                    if constexpr (Codebook == 14) legacy[4] &= 0x0Fu;
                    if constexpr (Codebook == 16)
                        legacy[5] = static_cast<std::uint8_t>((legacy[5] & 0x0Fu) | (code << 4));
                    std::uint8_t coded[bytes];
                    std::memcpy(coded, legacy, bytes);
                    embedCodeInPayload<Codebook>(coded, code);
                    ASSERT_EQ(codeFromPayload<Codebook>(coded), code);
                    ASSERT_EQ(loadCode<Codebook>(coded, 0), code);
                    for (std::uint32_t i = 0; i < bytes; ++i)
                    {
                        std::uint8_t value = coded[i];
                        if (signs >= 0 && static_cast<int>(i) >= signs && static_cast<int>(i) < signs + 4)
                            value = restoreSignParity(value);
                        if (Codebook == 14 && i == 4) value &= 0x0Fu;
                        ASSERT_EQ(value, legacy[i]) << int(Codebook) << " byte " << i;
                    }
                }
        };
        check.template operator()<12>(16);
        check.template operator()<15>(16);
        check.template operator()<14>(256);
        check.template operator()<16>(16);
    }
} // namespace llaminar2::native_quantised_superblock::test
