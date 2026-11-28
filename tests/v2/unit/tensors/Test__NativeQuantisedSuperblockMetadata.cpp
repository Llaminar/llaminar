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
 * nonzero superblock phases and ragged K. The compile-time read forms used
 * by the CUDA prefill hot loops (loadBlockMetadataAs) are checked against the
 * runtime representation loader for both representations and every shift.
 */
#include "tensors/NativeQuantisedSuperblockMetadata.h"
#include "kernels/common/NativeQuantisedScaleLoad.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

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
        // Codebook 4 is IQ4_NL (never superblock); 0/6/19.. have no source superblock codes.
        for (const std::uint8_t cb : {0, 4, 6, 19, 20, 21, 23, 25})
            EXPECT_FALSE(isSuperblockCoded(cb)) << int(cb);
        // Per 32 values: code bits + superblock bits / 8 + payload bits equal the GGUF bits.
        struct Expect { std::uint8_t cb; std::uint32_t code_bits, field_bytes, payload_bytes, file_bits_per_256; };
        const Expect table[] = {
            {5, 12, 4, 16, 144 * 8},  {7, 12, 4, 20, 176 * 8},  {8, 16, 2, 24, 210 * 8},
            {9, 12, 2, 12, 110 * 8},  {10, 16, 4, 8, 84 * 8},   {11, 4, 2, 13, 110 * 8},
            {12, 4, 2, 12, 98 * 8},   {13, 8, 2, 9, 82 * 8},    {14, 8, 2, 9, 74 * 8},
            {15, 4, 2, 8, 66 * 8},    {16, 4, 2, 6, 50 * 8},    {17, 6, 2, 6, 56 * 8},
            {24, 6, 2, 16, 136 * 8},
        };
        for (const auto &e : table)
        {
            SCOPED_TRACE(int(e.cb));
            EXPECT_EQ(superblockCodeBits(e.cb), e.code_bits);
            EXPECT_EQ(superblockFieldBytes(e.cb), e.field_bytes);
            // Code-in-payload formats spend no separate plane bits.
            const std::uint32_t plane_bits = superblockCodePlaneBits(e.cb);
            const std::uint32_t device_bits = 8u * (e.payload_bytes * 8u) + 8u * plane_bits + 8u * e.field_bytes;
            const bool in_payload = superblockCodeInPayload(e.cb);
            // In-payload codes reuse payload bits (sign parity / spare qh / existing qh bits).
            EXPECT_EQ(device_bits, e.file_bits_per_256) << (in_payload ? "code in payload" : "separate plane");
        }
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
                // Keep the floating-expression oracle independent of production's integer multiplier.
                ASSERT_EQ(cph(dl), cph(df * static_cast<float>(2u * (qh_high & 7u) + 1u)));
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
                // Exhaust all scale bits and codes against the original floating expression.
                ASSERT_EQ(cph(df * (2.0f * static_cast<float>(code & 7u) + 1.0f)),
                    cph(df * static_cast<float>(2u * (code & 7u) + 1u)));
                ASSERT_EQ(cph(df * (2.0f * static_cast<float>(code >> 3) + 1.0f)),
                    cph(df * static_cast<float>(2u * (code >> 3) + 1u)));
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
                    EXPECT_EQ(planes.code, static_cast<std::size_t>(rows) * kb * 2u + kSuperblockCodePlanePadBytes);
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
        EXPECT_EQ(superblockCodePlaneBits(12), 0u);
        EXPECT_EQ(superblockCodePlaneBits(13), 8u);
        EXPECT_EQ(superblockCodePlaneBits(10), 16u);
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

    /** Q4_K/Q5_K (5/7): primary = half(d*sc), secondary = half(-dmin*m); code = sc | m << 6. */
    TEST(NativeQuantisedSuperblockMetadata, Q4KQ5KReproduceScaleAndMinimum)
    {
        forEachHalf([](std::uint16_t h) {
            const float v = refWiden(h);
            for (std::uint32_t six = 0; six < 64; ++six)
            {
                const std::uint32_t other = (six * 37u + 11u) & 63u;
                const auto q4 = decodeSuperblockMetadata<5>(six | (other << 6), packSuperblockWord(h, 0x3C00));
                ASSERT_EQ(q4.primary, cph(v * static_cast<float>(six))) << h << ' ' << six;
                const auto q5 = decodeSuperblockMetadata<7>(other | (six << 6), packSuperblockWord(0x3C00, h));
                ASSERT_TRUE(sameHalf(q5.secondary, cph(-v * static_cast<float>(six)))) << h << ' ' << six;
                ASSERT_EQ(q5.primary, cph(1.0f * static_cast<float>(other)));
            }
        });
    }

    /** Bit-packed code planes: every width, every bit shift, random codes round-trip exactly. */
    TEST(NativeQuantisedSuperblockMetadata, BitPackedCodesRoundTripAtEveryShift)
    {
        for (const std::uint8_t cb : {5, 9, 10, 11, 13, 17, 24})
        {
            const std::uint32_t bits = superblockCodeBits(cb);
            for (std::uint32_t shift = 0; shift < 8; ++shift)
            {
                constexpr std::size_t count = 97;
                std::vector<std::uint8_t> plane((shift + count * bits + 7) / 8 + kSuperblockCodePlanePadBytes, 0xA5);
                std::vector<std::uint32_t> codes(count);
                for (std::size_t i = 0; i < count; ++i)
                {
                    codes[i] = static_cast<std::uint32_t>((i * 2654435761u) >> 7) & ((1u << bits) - 1u);
                    for (std::uint32_t b = 0; b < bits; ++b)
                    {
                        const std::size_t bit = shift + i * bits + b;
                        plane[bit / 8] = static_cast<std::uint8_t>((plane[bit / 8] & ~(1u << (bit % 8))) |
                                                                   (((codes[i] >> b) & 1u) << (bit % 8)));
                    }
                }
                for (std::size_t i = 0; i < count; ++i)
                    ASSERT_EQ(loadCode(cb, plane.data(), i, shift), codes[i]) << int(cb) << ' ' << shift << ' ' << i;
            }
        }
    }

    /** decodeCodedMatrix reproduces per-block words and restores code-in-payload bytes for test oracles. */
    TEST(NativeQuantisedSuperblockMetadata, HostMatrixDecodeMatchesPerBlockWordsAndRestoresPayload)
    {
        constexpr std::size_t rows = 5;
        constexpr std::uint32_t blocks = 11, phase = 3, shift = 5;
        for (const std::uint8_t cb : {5, 10, 12, 13, 14, 15, 16, 17})
        {
            SCOPED_TRACE(int(cb));
            const std::uint32_t pb = superblockCodePayloadBytes(cb) ? superblockCodePayloadBytes(cb) : 16u;
            const std::uint32_t bits = superblockCodeBits(cb);
            const std::size_t count = rows * blocks;
            const std::size_t superblocks = rows * superblocksPerRow(blocks, phase);
            std::vector<std::uint8_t> legacy(count * pb), payload, plane((shift + count * bits + 7) / 8 + 4, 0);
            for (std::size_t i = 0; i < legacy.size(); ++i)
                legacy[i] = static_cast<std::uint8_t>((i * 131u + 7u) & 0xFFu);
            // Legacy per-block payload: even-parity sign selectors, clear spare nibble.
            for (std::size_t b = 0; b < count; ++b)
                restoreCodedPayload(cb, legacy.data() + b * pb);
            payload = legacy;
            std::vector<std::uint32_t> codes(count);
            std::vector<std::uint32_t> words(superblocks);
            for (std::size_t i = 0; i < superblocks; ++i)
                words[i] = static_cast<std::uint32_t>(0x3800u + (i * 37u) % 1024u) |
                           (superblockFieldBytes(cb) == 4 ? (0x3400u + (i * 11u) % 512u) << 16 : 0u);
            for (std::size_t i = 0; i < count; ++i)
            {
                codes[i] = static_cast<std::uint32_t>((i * 2654435761u) >> 9) & ((1u << bits) - 1u);
                std::uint8_t *block = payload.data() + i * pb;
                switch (cb)
                {
                case 12: embedCodeInPayload<12>(block, codes[i]); break;
                case 14: embedCodeInPayload<14>(block, codes[i]); break;
                case 15: embedCodeInPayload<15>(block, codes[i]); break;
                case 16: codes[i] = codeFromPayload<16>(block); break;
                default:
                    for (std::uint32_t b = 0; b < bits; ++b)
                    {
                        const std::size_t bit = shift + i * bits + b;
                        plane[bit / 8] = static_cast<std::uint8_t>(plane[bit / 8] | (((codes[i] >> b) & 1u) << (bit % 8)));
                    }
                }
            }
            std::vector<std::uint16_t> primary(count), secondary(count);
            std::vector<std::uint32_t> emin(count);
            std::vector<std::uint16_t> half_words(superblocks);
            for (std::size_t i = 0; i < superblocks; ++i) half_words[i] = static_cast<std::uint16_t>(words[i]);
            const void *sb_plane = superblockFieldBytes(cb) == 4 ? static_cast<const void *>(words.data())
                                                                 : static_cast<const void *>(half_words.data());
            decodeCodedMatrix(cb, rows, blocks, phase, shift, pb, payload.data(), plane.data(), sb_plane,
                              primary.data(), secondary.data(), emin.data());
            EXPECT_EQ(payload, legacy) << "per-block payload not restored";
            for (std::uint32_t blk = 0; blk < blocks; ++blk)
                for (std::size_t row = 0; row < rows; ++row)
                {
                    const auto a = blockMajorAddress(blk, static_cast<std::uint32_t>(row), rows, phase);
                    const std::uint32_t w = superblockFieldBytes(cb) == 4 ? words[a.superblock] : half_words[a.superblock];
                    const auto want = decodeSuperblockMetadata(cb, codes[a.linear], w);
                    ASSERT_EQ(primary[a.linear], want.primary);
                    ASSERT_EQ(secondary[a.linear], want.secondary);
                    ASSERT_EQ(emin[a.linear], want.emin);
                }
        }
    }
    namespace
    {
        namespace nqs = ::llaminar2::native_quantised_scale;
        using Form = nqs::NativeQuantisedMetadataReadForm;

        /** @return The read form withMetadataRepresentation() hands its body for @p planes. */
        template <std::uint8_t CB>
        Form selectedForm(const nqs::NativeQuantisedMetadataPlanes &planes)
        {
            return nqs::withMetadataRepresentation<CB>(planes, [](auto form) { return decltype(form)::value; });
        }

        bool sameWords(const StagedBlockMetadata &a, const StagedBlockMetadata &b)
        {
            return a.primary == b.primary && a.secondary == b.secondary && a.emin == b.emin;
        }

        /**
         * @brief Every compile-time read form equals the runtime loader on random planes.
         *
         * Coded planes are filled with random code bytes and superblock words
         * for every code bit shift 0..7 and superblock phase 0..7; per-block
         * planes with random primary/secondary/emin words. The coded form
         * must equal loadBlockMetadata() at every block, the byte-aligned form
         * (8/16-bit code planes) must equal it whenever the shift is zero,
         * and withMetadataRepresentation() must select exactly that form.
         */
        template <std::uint8_t CB>
        void expectReadFormsMatchRuntimeLoader()
        {
            SCOPED_TRACE("codebook " + std::to_string(CB));
            constexpr std::uint32_t rows = 9, blocks = 21;
            std::mt19937 rng(77u + CB);
            const auto random_bytes = [&](std::size_t n) {
                std::vector<std::uint8_t> v(n);
                for (auto &b : v) b = static_cast<std::uint8_t>(rng());
                return v;
            };
            for (std::uint32_t shift = 0; shift < 8; ++shift)
                for (std::uint32_t phase = 0; phase < 8; ++phase)
                {
                    const auto bytes = planeBytes(CB, rows, blocks, phase);
                    // Shifted matrices start `shift` bits into their first byte.
                    auto codes = random_bytes(bytes.code + 1);
                    auto supers = random_bytes(bytes.superblock + 4);
                    std::vector<std::uint32_t> super_words((supers.size() + 3) / 4);
                    std::memcpy(super_words.data(), supers.data(), supers.size());
                    std::vector<std::uint16_t> code_words((codes.size() + 1) / 2 + 1);
                    std::memcpy(code_words.data(), codes.data(), codes.size());
                    const nqs::NativeQuantisedMetadataPlanes coded{
                        code_words.data(), super_words.data(), nullptr, phase,
                        NativeQuantisedMetadataRepresentation::SuperblockCoded, shift};
                    const Form expected_form = nqs::nativeQuantisedByteAlignedCodePlane(CB) && shift == 0
                                                   ? Form::CodedByteAligned
                                                   : Form::Coded;
                    ASSERT_EQ(selectedForm<CB>(coded), expected_form) << "shift " << shift;
                    for (std::uint32_t block = 0; block < blocks; ++block)
                        for (std::uint32_t row = 0; row < rows; ++row)
                        {
                            const auto address = nqs::blockMajorAddress(coded, block, row, rows);
                            const auto want = nqs::loadBlockMetadata<CB>(coded, address);
                            ASSERT_TRUE(sameWords(nqs::loadBlockMetadataAs<CB, Form::Coded>(coded, address), want))
                                << "shift " << shift << " phase " << phase << " block " << block << " row " << row;
                            if constexpr (nqs::nativeQuantisedByteAlignedCodePlane(CB))
                                if (shift == 0)
                                    ASSERT_TRUE(sameWords(
                                        nqs::loadBlockMetadataAs<CB, Form::CodedByteAligned>(coded, address), want))
                                        << "phase " << phase << " block " << block << " row " << row;
                        }
                }
            const std::size_t count = std::size_t{rows} * blocks;
            std::vector<std::uint16_t> primary(count), secondary(count);
            std::vector<std::uint32_t> emins(count);
            for (auto &w : primary) w = static_cast<std::uint16_t>(rng());
            for (auto &w : secondary) w = static_cast<std::uint16_t>(rng());
            for (auto &w : emins) w = rng();
            for (const bool with_secondary : {false, true})
            {
                const nqs::NativeQuantisedMetadataPlanes per_block{
                    primary.data(), with_secondary ? secondary.data() : nullptr, CB == 10 ? emins.data() : nullptr, 0,
                    NativeQuantisedMetadataRepresentation::PerBlockPlanes, 0};
                ASSERT_EQ(selectedForm<CB>(per_block), Form::PerBlock);
                for (std::uint32_t block = 0; block < blocks; ++block)
                    for (std::uint32_t row = 0; row < rows; ++row)
                    {
                        const auto address = nqs::blockMajorAddress(per_block, block, row, rows);
                        ASSERT_TRUE(sameWords(nqs::loadBlockMetadataAs<CB, Form::PerBlock>(per_block, address),
                                              nqs::loadBlockMetadata<CB>(per_block, address)));
                    }
            }
        }
    } // namespace

    /**
     * The CUDA prefill hot loops branch once on the matrix representation and
     * read metadata through loadBlockMetadataAs(); that must return exactly
     * the runtime loader's words for every separate-code-plane codebook,
     * every code bit shift and every superblock phase.
     */
    TEST(NativeQuantisedSuperblockMetadata, CompileTimeReadFormsMatchTheRuntimeLoader)
    {
        expectReadFormsMatchRuntimeLoader<5>();
        expectReadFormsMatchRuntimeLoader<7>();
        expectReadFormsMatchRuntimeLoader<8>();
        expectReadFormsMatchRuntimeLoader<9>();
        expectReadFormsMatchRuntimeLoader<10>();
        expectReadFormsMatchRuntimeLoader<11>();
        expectReadFormsMatchRuntimeLoader<13>();
        expectReadFormsMatchRuntimeLoader<17>();
        expectReadFormsMatchRuntimeLoader<24>();
    }
} // namespace llaminar2::native_quantised_superblock::test
