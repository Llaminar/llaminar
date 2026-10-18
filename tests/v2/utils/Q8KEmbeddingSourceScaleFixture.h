/**
 * @file Q8KEmbeddingSourceScaleFixture.h
 * @brief Original Q8_K binary32 scale edge cases for GPU embedding regression.
 *
 * These source words cannot survive an intermediate binary16 publication.
 * The fixture constructs native Q8_K blocks directly; production preparation
 * must preserve them and the untouched Tensor decoder supplies the oracle.
 */
#pragma once

#include "EmbeddingVerifierFormats.h"

#include <array>
#include <cstring>
#include <stdexcept>

namespace llaminar2::test
{
    /**
     * @brief Construct Q8_K rows with signed normal and subnormal source scales.
     * @param shape Native two-dimensional table; width must be 256-aligned.
     * @return Owned source tensor with all six exact binary32 scale words.
     * @throws std::invalid_argument If the table cannot hold the complete fixture.
     */
    inline std::unique_ptr<TensorBase> createQ8KEmbeddingSourceScaleEdges(
        const std::vector<size_t> &shape)
    {
        constexpr std::array<uint32_t, 6> scale_bits = {
            0x3f800001u, 0xbf800001u, // Adjacent to +/-1; not binary16 representable.
            0x00800001u, 0x80800001u, // Tiny normal binary32, nonzero source metadata.
            0x00012345u, 0x80012345u  // Signed subnormal binary32 source metadata.
        };
        constexpr std::array<int8_t, 5> codes = {-127, -1, 0, 1, 127};
        if (shape.size() != 2 || shape[0] < scale_bits.size() ||
            shape[1] == 0 || shape[1] % Q8_KBlock::BLOCK_SIZE != 0)
            throw std::invalid_argument("Q8_K scale fixture requires six rows and aligned width");
        const size_t blocks_per_row = shape[1] / Q8_KBlock::BLOCK_SIZE;
        std::vector<uint8_t> bytes(shape[0] * blocks_per_row * sizeof(Q8_KBlock));
        auto *blocks = reinterpret_cast<Q8_KBlock *>(bytes.data());
        for (size_t row = 0; row < shape[0]; ++row)
            for (size_t column_block = 0; column_block < blocks_per_row; ++column_block)
            {
                auto &block = blocks[row * blocks_per_row + column_block];
                const uint32_t bits = scale_bits[row % scale_bits.size()];
                std::memcpy(&block.d, &bits, sizeof(bits));
                for (size_t partial = 0; partial < 16; ++partial)
                {
                    int sum = 0;
                    for (size_t lane = 0; lane < 16; ++lane)
                    {
                        const size_t element = partial * 16 + lane;
                        block.qs[element] = codes[element % codes.size()];
                        sum += block.qs[element];
                    }
                    block.bsums[partial] = static_cast<int16_t>(sum);
                }
            }
        return std::make_unique<Q8_KTensor>(shape, bytes);
    }

    /**
     * @brief Describe the edge fixture using the production native-source route.
     * @return Fixture accepted by the shared captured/grouped/serial test helper.
     */
    inline EmbeddingVerifierFormatCase q8KEmbeddingSourceScaleEdges()
    {
        return {"Q8_K", true, "native_quantized", "prepared_device_native_quantized",
                [](const std::vector<size_t> &shape, uint32_t) -> std::unique_ptr<TensorBase>
                { return createQ8KEmbeddingSourceScaleEdges(shape); }};
    }
} // namespace llaminar2::test
