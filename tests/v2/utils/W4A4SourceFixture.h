/**
 * @file W4A4SourceFixture.h
 * @brief Small valid native source blocks for ownership/publication contracts.
 *
 * Values expose original row/block coordinates without any floating weight
 * conversion. These synthetic bytes test storage contracts, not arithmetic
 * replay; immutable upstream recordings remain the numerical oracle.
 */
#pragma once
#include "tensors/W4A4MatrixContract.h"
#include "tensors/TensorType.h"
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace llaminar2::test
{
    /**
     * @brief Construct complete native rows with visible original coordinates.
     * @param type Explicit W4A4 or W4A4H source identity.
     * @param rows Positive logical N.
     * @param columns Complete block-aligned original K.
     * @return Valid native payload, with identity or length128 Sylvester policy.
     * @throws std::invalid_argument For unsupported type or source geometry.
     */
    inline std::vector<w4a4::SourceBlock> w4a4SourceBlocks(TensorType type, size_t rows, size_t columns)
    {
        const auto extent = w4a4::MatrixContract::sourceExtent(rows, columns);
        if (!isW4A4Format(type) || !extent)
            throw std::invalid_argument("Native source fixture requires complete W4A4/H rows");
        std::vector<w4a4::SourceBlock> blocks(*extent / sizeof(w4a4::SourceBlock));
        const auto blocks_per_row = columns / w4a4::kBlockElements;
        for (size_t i = 0; i < blocks.size(); ++i)
        {
            blocks[i].unit = float(1 + (i / blocks_per_row) % 8) * 0.125f;
            blocks[i].descriptor = 0x60004321;
            blocks[i].rotation_length = type == TensorType::W4A4 ? static_cast<int>(columns) : 128;
            std::fill_n(blocks[i].codes, 128, static_cast<uint8_t>(i));
        }
        return blocks;
    }
}
