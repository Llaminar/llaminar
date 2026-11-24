/**
 * @file Test__VnniCodedPacking.cpp
 * @brief Host superblock-coded packing reproduces the per-block packer through the codec.
 *
 * Every superblock-codable source format is packed twice by its host
 * IINT8Unpackable::packVnniBlock(): once into the per-block scale/minimum
 * planes and once into the superblock-coded destination (one code per block,
 * one d/dmin word per source superblock). For every row and K block the coded
 * payload, restored with restoreCodedPayload(), must equal the per-block
 * payload byte for byte, and decodeSuperblockMetadata(code, word) must equal
 * the per-block staged words. Per-block words are compared after the GPU
 * publication contract's signed-zero canonicalization (canonicalPreparedHalfBits
 * maps a computed -0 to +0; the host converter keeps the sign), so this gate
 * proves the codes and words, not a host/device rounding difference.
 *
 * Windows that start inside a superblock (origin not a multiple of 8) prove the
 * window-local superblock index and the first-block publication rule. The
 * tests are device-free and use the shared random tensor fixtures.
 */
#include <gtest/gtest.h>

#include "../../utils/TestTensorFactory.h"
#include "tensors/NativeQuantisedSuperblockMetadata.h"
#include "tensors/TensorClasses.h"
#include "tensors/VnniPackContext.h"

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        namespace sb = native_quantised_superblock;
        using Factory = TestTensorFactory;

        /** @brief One codable source format and its random-tensor factory. */
        struct CodableSource
        {
            std::string name;
            std::uint8_t codebook = 0;
            int payload_bytes = 0;
            std::function<std::unique_ptr<TensorBase>(const std::vector<size_t> &, uint32_t)> make;
        };

        /** @return Every superblock-codable source format with its physical codebook. */
        std::vector<CodableSource> codableSources()
        {
            return {
                {"Q4_K", 5, 16, [](auto s, auto r) { return Factory::createQ4_KRandom(s, r); }},
                {"Q5_K", 7, 20, [](auto s, auto r) { return Factory::createQ5_KRandom(s, r); }},
                {"Q6_K", 8, 24, [](auto s, auto r) { return Factory::createQ6_KRandom(s, r); }},
                {"Q3_K", 9, 12, [](auto s, auto r) { return Factory::createQ3_KRandom(s, r); }},
                {"Q2_K", 10, 8, [](auto s, auto r) { return Factory::createQ2_KRandom(s, r); }},
                {"IQ3_S", 11, 13, [](auto s, auto r) { return Factory::createIQ3_SRandom(s, r); }},
                {"IQ3_XXS", 12, 12, [](auto s, auto r) { return Factory::createIQ3_XXSRandom(s, r); }},
                {"IQ2_S", 13, 9, [](auto s, auto r) { return Factory::createIQ2_SRandom(s, r); }},
                {"IQ2_XS", 14, 9, [](auto s, auto r) { return Factory::createIQ2_XSRandom(s, r); }},
                {"IQ2_XXS", 15, 8, [](auto s, auto r) { return Factory::createIQ2_XXSRandom(s, r); }},
                {"IQ1_S", 16, 6, [](auto s, auto r) { return Factory::createIQ1_SRandom(s, r); }},
                {"IQ1_M", 17, 6, [](auto s, auto r) { return Factory::createIQ1_MRandom(s, r); }},
                {"IQ4_XS", 24, 16, [](auto s, auto r) { return Factory::createIQ4_XSRandom(s, r); }},
            };
        }

        /** @return The code a code-in-payload block carries. */
        std::uint32_t payloadCode(std::uint8_t codebook, const std::uint8_t *payload)
        {
            switch (codebook)
            {
            case 12: return sb::codeFromPayload<12>(payload);
            case 14: return sb::codeFromPayload<14>(payload);
            case 15: return sb::codeFromPayload<15>(payload);
            case 16: return sb::codeFromPayload<16>(payload);
            default: return 0xFFFFFFFFu;
            }
        }

        /** @return @p word with the publication contract's -0 -> +0 canonicalization. */
        std::uint16_t canonicalZero(std::uint16_t word) { return word == 0x8000u ? 0u : word; }

        /**
         * @brief Pack rows x [origin, origin+count) both ways and compare every block.
         * @param source Codable source format.
         * @param rows Source rows.
         * @param k Source K (multiple of 256 plus an optional ragged tail).
         * @param origin First packed source K block (window origin).
         * @param count Packed K blocks.
         */
        void expectCodedEqualsPerBlock(const CodableSource &source, int rows, int k, int origin, int count)
        {
            SCOPED_TRACE(source.name + " origin=" + std::to_string(origin) + " count=" + std::to_string(count));
            const auto tensor = source.make({static_cast<size_t>(rows), static_cast<size_t>(k)}, 1234u + source.codebook);
            const auto *unpackable = dynamic_cast<const IINT8Unpackable *>(tensor.get());
            ASSERT_NE(unpackable, nullptr);

            const size_t blocks = static_cast<size_t>(rows) * count;
            const int window_superblocks = (origin + count - 1) / 8 - origin / 8 + 1;
            std::vector<uint8_t> legacy_payload(blocks * source.payload_bytes), coded_payload(blocks * source.payload_bytes);
            std::vector<uint16_t> scales(blocks), mins(blocks);
            std::vector<uint32_t> emins(blocks), codes(blocks, 0xFFFFFFFFu);
            std::vector<uint32_t> superblocks(static_cast<size_t>(window_superblocks) * rows, 0xFFFFFFFFu);

            VnniPackContext legacy{};
            legacy.N = rows;
            legacy.K = k;
            legacy.blocks_per_row = (k + 31) / 32;
            legacy.payload_bytes = source.payload_bytes;
            legacy.payload_array = legacy_payload.data();
            legacy.scales_array = scales.data();
            legacy.mins_array = mins.data();
            legacy.emins_array = source.codebook == 10 ? emins.data() : nullptr;
            legacy.destination_block_origin = origin;
            legacy.destination_block_count = count;
            VnniPackContext coded = legacy;
            coded.payload_array = coded_payload.data();
            coded.scales_array = nullptr;
            coded.mins_array = nullptr;
            coded.emins_array = nullptr;
            coded.codes_array = codes.data();
            coded.superblocks_array = superblocks.data();

            for (int n = 0; n < rows; ++n)
                for (int b = origin; b < origin + count; ++b)
                {
                    unpackable->packVnniBlock(legacy, n, n, b);
                    unpackable->packVnniBlock(coded, n, n, b);
                }

            for (int n = 0; n < rows; ++n)
                for (int b = origin; b < origin + count; ++b)
                {
                    SCOPED_TRACE("row=" + std::to_string(n) + " block=" + std::to_string(b));
                    const size_t linear = vnniLinearIdx(legacy, n, b);
                    std::vector<uint8_t> restored(coded_payload.begin() + linear * source.payload_bytes,
                                                  coded_payload.begin() + (linear + 1) * source.payload_bytes);
                    if (sb::superblockCodeInPayload(source.codebook))
                        EXPECT_EQ(payloadCode(source.codebook, restored.data()), codes[linear]);
                    sb::restoreCodedPayload(source.codebook, restored.data());
                    EXPECT_EQ(std::memcmp(restored.data(), legacy_payload.data() + linear * source.payload_bytes,
                                          source.payload_bytes), 0);
                    ASSERT_LT(codes[linear], 1u << sb::superblockCodeBits(source.codebook));
                    const size_t sb_index = static_cast<size_t>(b / 8 - origin / 8) * rows + n;
                    ASSERT_NE(superblocks[sb_index], 0xFFFFFFFFu) << "superblock word never published";
                    const auto words = sb::decodeSuperblockMetadata(source.codebook, codes[linear], superblocks[sb_index]);
                    EXPECT_EQ(words.primary, canonicalZero(scales[linear]));
                    EXPECT_EQ(words.secondary, canonicalZero(mins[linear]));
                    if (source.codebook == 10)
                        EXPECT_EQ(words.emin, (uint32_t{canonicalZero(static_cast<uint16_t>(emins[linear]))}) |
                                                  (uint32_t{canonicalZero(static_cast<uint16_t>(emins[linear] >> 16))} << 16));
                }
        }
    } // namespace

    /** Whole superblocks: every block, every codable format. */
    TEST(VnniCodedPacking, CodedHostPackingMatchesPerBlockPackerForEveryFormat)
    {
        for (const auto &source : codableSources())
            expectCodedEqualsPerBlock(source, 5, 512, 0, 16);
    }

    /** Windows starting inside a superblock publish its word from their first block. */
    TEST(VnniCodedPacking, WindowsStartingInsideASuperblockPublishEveryWord)
    {
        for (const auto &source : codableSources())
        {
            expectCodedEqualsPerBlock(source, 3, 768, 3, 13);
            expectCodedEqualsPerBlock(source, 3, 768, 9, 1);
        }
    }

    /** A coded destination never needs (or touches) the per-block planes. */
    TEST(VnniCodedPacking, CodedDestinationRequiresBothCodedPlanes)
    {
        const auto tensor = Factory::createQ3_KRandom({1, 256}, 7u);
        const auto *unpackable = dynamic_cast<const IINT8Unpackable *>(tensor.get());
        ASSERT_NE(unpackable, nullptr);
        std::vector<uint8_t> payload(12);
        std::vector<uint32_t> codes(1);
        VnniPackContext context{};
        context.N = 1;
        context.K = 256;
        context.blocks_per_row = 8;
        context.payload_bytes = 12;
        context.payload_array = payload.data();
        context.codes_array = codes.data();
        EXPECT_THROW(unpackable->packVnniBlock(context, 0, 0, 0), std::invalid_argument);
    }
} // namespace llaminar2::test
