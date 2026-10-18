/**
 * @file NativeEmbeddingLayout.h
 * @brief Original quantized embedding storage shared by preparation and lookup.
 *
 * This metadata describes immutable source blocks, never a derived codebook.
 * Checked host factories own the exact physical geometry used by allocation,
 * admission and CUDA/HIP lookup. Device code only consumes these plain fields;
 * tensor classes, SIMD headers and backend ownership are intentionally absent.
 */
#pragma once

#include "tensors/BlockStructures.h"
#include "tensors/TensorType.h"

#include <cstddef>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace llaminar2
{
    /** @brief One source-native embedding allocation's immutable physical geometry. */
    struct NativeEmbeddingLayout
    {
        TensorType source_type = TensorType::FP32; ///< Original source codebook; floating is invalid here.
        size_t block_elements = 0; ///< Logical values represented by one source block.
        size_t block_bytes = 0; ///< Complete original block including original scale/minimum metadata.
        size_t blocks_per_row = 0; ///< Complete physical source blocks per vocabulary row.
        size_t row_stride_bytes = 0; ///< Byte distance between consecutive native source rows.
        size_t byte_size = 0; ///< Exact native backing extent; not unused allocation capacity.

        /**
         * @brief Report whether the existing model loader supports this quantized source.
         * @param source Original tensor scalar/codebook type.
         * @return True for the canonical quantized model-weight inventory only.
         */
        [[nodiscard]] static constexpr bool supports(TensorType source) noexcept
        {
            switch (source)
            {
            case TensorType::Q4_0: case TensorType::Q4_1:
            case TensorType::Q5_0: case TensorType::Q5_1:
            case TensorType::Q8_0: case TensorType::Q8_1:
            case TensorType::IQ4_NL: case TensorType::IQ4_XS:
            case TensorType::Q2_K: case TensorType::Q3_K:
            case TensorType::Q4_K: case TensorType::Q5_K:
            case TensorType::Q6_K: case TensorType::Q8_K:
            case TensorType::IQ1_S: case TensorType::IQ1_M:
            case TensorType::IQ2_XXS: case TensorType::IQ2_XS:
            case TensorType::IQ2_S: case TensorType::IQ3_XXS:
            case TensorType::IQ3_S:
                return true;
            default:
                return false;
            }
        }

        /**
         * @brief Resolve checked bytes without converting any source values.
         * @param source Original quantized tensor type.
         * @param vocab_size Number of physically owned vocabulary rows.
         * @param d_model Logical values in each row.
         * @return Exact native block geometry and backing extent.
         * @throws std::invalid_argument For unsupported sources or non-positive width.
         * @throws std::overflow_error If row or allocation bytes overflow size_t.
         */
        [[nodiscard]] static NativeEmbeddingLayout forSource(
            TensorType source, size_t vocab_size, int d_model)
        {
            if (d_model <= 0)
                throw std::invalid_argument("Native embedding requires a positive row width");
            NativeEmbeddingLayout layout;
            layout.source_type = source;
#define LLAMINAR_NATIVE_EMBEDDING_BLOCK(TYPE) \
            case TensorType::TYPE: \
                layout.block_elements = TYPE##Block::BLOCK_SIZE; \
                layout.block_bytes = sizeof(TYPE##Block); \
                break
            switch (source)
            {
                LLAMINAR_NATIVE_EMBEDDING_BLOCK(Q4_0);
                LLAMINAR_NATIVE_EMBEDDING_BLOCK(Q4_1);
                LLAMINAR_NATIVE_EMBEDDING_BLOCK(Q5_0);
                LLAMINAR_NATIVE_EMBEDDING_BLOCK(Q5_1);
                LLAMINAR_NATIVE_EMBEDDING_BLOCK(Q8_0);
                LLAMINAR_NATIVE_EMBEDDING_BLOCK(Q8_1);
                LLAMINAR_NATIVE_EMBEDDING_BLOCK(IQ4_NL);
                LLAMINAR_NATIVE_EMBEDDING_BLOCK(IQ4_XS);
                LLAMINAR_NATIVE_EMBEDDING_BLOCK(Q2_K);
                LLAMINAR_NATIVE_EMBEDDING_BLOCK(Q3_K);
                LLAMINAR_NATIVE_EMBEDDING_BLOCK(Q4_K);
                LLAMINAR_NATIVE_EMBEDDING_BLOCK(Q5_K);
                LLAMINAR_NATIVE_EMBEDDING_BLOCK(Q6_K);
                LLAMINAR_NATIVE_EMBEDDING_BLOCK(Q8_K);
                LLAMINAR_NATIVE_EMBEDDING_BLOCK(IQ1_S);
                LLAMINAR_NATIVE_EMBEDDING_BLOCK(IQ1_M);
                LLAMINAR_NATIVE_EMBEDDING_BLOCK(IQ2_XXS);
                LLAMINAR_NATIVE_EMBEDDING_BLOCK(IQ2_XS);
                LLAMINAR_NATIVE_EMBEDDING_BLOCK(IQ2_S);
                LLAMINAR_NATIVE_EMBEDDING_BLOCK(IQ3_XXS);
                LLAMINAR_NATIVE_EMBEDDING_BLOCK(IQ3_S);
            default:
                throw std::invalid_argument("Native embedding requires a supported quantized source");
            }
#undef LLAMINAR_NATIVE_EMBEDDING_BLOCK
            const size_t width = static_cast<size_t>(d_model);
            layout.blocks_per_row = width / layout.block_elements +
                                    (width % layout.block_elements != 0u);
            if (layout.blocks_per_row > std::numeric_limits<size_t>::max() / layout.block_bytes)
                throw std::overflow_error("Native embedding row bytes overflow size_t");
            layout.row_stride_bytes = layout.blocks_per_row * layout.block_bytes;
            if (vocab_size > std::numeric_limits<size_t>::max() / layout.row_stride_bytes)
                throw std::overflow_error("Native embedding backing bytes overflow size_t");
            layout.byte_size = vocab_size * layout.row_stride_bytes;
            return layout;
        }

        /**
         * @brief Resolve the planning profile through the same source-byte authority.
         * @param source Canonical GGUF format label or existing K-quant policy alias.
         * @param vocab_size Number of physically owned vocabulary rows.
         * @param d_model Logical values in each row.
         * @return The same layout returned for the corresponding TensorType.
         * @throws std::invalid_argument For floating, unsupported or malformed formats.
         * @throws std::overflow_error If backing extent overflows size_t.
         */
        [[nodiscard]] static NativeEmbeddingLayout forFormat(
            std::string_view source, size_t vocab_size, int d_model)
        {
            if (source == "Q3_K_S" || source == "Q3_K_M" || source == "Q3_K_L") source = "Q3_K";
            if (source == "Q4_K_S" || source == "Q4_K_M") source = "Q4_K";
            if (source == "Q5_K_S" || source == "Q5_K_M") source = "Q5_K";
            for (const auto type : {TensorType::Q4_0, TensorType::Q4_1, TensorType::Q5_0,
                     TensorType::Q5_1, TensorType::Q8_0, TensorType::Q8_1, TensorType::IQ4_NL,
                     TensorType::IQ4_XS, TensorType::Q2_K, TensorType::Q3_K, TensorType::Q4_K,
                     TensorType::Q5_K, TensorType::Q6_K, TensorType::Q8_K, TensorType::IQ1_S,
                     TensorType::IQ1_M, TensorType::IQ2_XXS, TensorType::IQ2_XS, TensorType::IQ2_S,
                     TensorType::IQ3_XXS, TensorType::IQ3_S})
            {
                if (source == tensorTypeName(type))
                    return forSource(type, vocab_size, d_model);
            }
            throw std::invalid_argument("Native embedding profile has an unsupported source format");
        }

        /**
         * @brief Compare the complete representation identity, including backing extent.
         * @param other Layout whose source format, stride and extent must match.
         * @return True when both layouts describe the same physical representation.
         */
        [[nodiscard]] bool operator==(const NativeEmbeddingLayout &other) const = default;
    };
} // namespace llaminar2
