/**
 * @file PreparedHalfWeights.h
 * @brief Original-source half-RNE materialization without a decoded matrix shadow.
 *
 * The returned tensor uses the ordinary tensor/PMA allocation authority. The
 * original decoder owns source offsets and factors; only one bounded source
 * tile is widened on each setup worker. Runtime kernels never call this helper.
 */
#pragma once
#include "tensors/TensorClasses.h"
#include "tensors/TensorSlice.h"
#include "kernels/common/DeviceHalfMetadataContract.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <memory>
#include <stdexcept>
namespace llaminar2
{
    /**
     * @brief Round original decoded projection weights once into model-owned half words.
     * @param source Host-resident original matrix, including a materialized slice.
     * @return Half-RNE tensor retaining exactly the source logical dimensions.
     * @throws std::invalid_argument For absent/unsupported source tiles, nonfinite
     *         decoded values, or half overflow; no clamp or fallback is applied.
     *
     * Generic FP16Tensor::from_fp32 is deliberately avoided: its scalar/tail
     * conversion has a different subnormal rounding contract. No intermediate
     * full FP32 matrix, alternate packed scales or second ownership cache exists.
     */
    inline std::shared_ptr<FP16Tensor> materializePreparedHalfWeights(const TensorBase &source)
    {
        if (source.shape().size() != 2 || source.rows() == 0 || source.cols() == 0 ||
            source.is_raw_data_released() || source.raw_data() == nullptr)
            throw std::invalid_argument("Prepared half weights requires retained original host matrix bytes");
        const TensorBase *native = &source;
        while (const auto *slice = dynamic_cast<const TensorSlice *>(native)) native = slice->inner();
        const auto *decoder = dynamic_cast<const ITensorGemmTileDataProvider *>(native);
        const bool bf16 = source.native_type() == TensorType::BF16;
        if (!bf16 && (!decoder || decoder->block_size() == 0 || decoder->block_size() > 256 ||
            decoder->decoder_rows() != source.rows() || decoder->decoder_cols() != source.cols()))
            throw std::invalid_argument("Prepared half weights requires an original source decoder with tiles <=256");
        auto half = std::make_shared<FP16Tensor>(source.shape());
        auto *out = half->mutable_typed_data();
        const auto rows = source.rows(), columns = source.cols();
        // Setup-only source rows are independent. Every worker owns its tile;
        // the canonical half conversion and source factors do not depend on ISA.
        unsigned invalid = 0;
#pragma omp parallel for schedule(static) reduction(|:invalid)
        for (size_t row = 0; row < rows; ++row)
        {
            if (bf16)
            {
                const auto *words = static_cast<const uint16_t *>(source.raw_data()) + row * columns;
                for (size_t col = 0; col < columns; ++col)
                {
                    const auto value = std::bit_cast<float>(static_cast<uint32_t>(words[col]) << 16);
                    const auto bits = canonicalPreparedHalfBits(value);
                    invalid |= !std::isfinite(value) || (bits & 0x7c00u) == 0x7c00u;
                    out[row * columns + col] = bits;
                }
                continue;
            }
            alignas(64) float tile[256];
            const auto width = decoder->block_size();
            for (size_t first = 0, block = 0; first < columns; first += width, ++block)
            {
                decoder->decode_block_at(row, block, tile);
                for (size_t col = 0; col < std::min(width, columns - first); ++col)
                {
                    const auto bits = canonicalPreparedHalfBits(tile[col]);
                    invalid |= !std::isfinite(tile[col]) || (bits & 0x7c00u) == 0x7c00u;
                    out[row * columns + first + col] = bits;
                }
            }
        }
        if (invalid)
            throw std::invalid_argument("Prepared half weights decoded a nonfinite or half-overflowing source value");
        return half;
    }
}
