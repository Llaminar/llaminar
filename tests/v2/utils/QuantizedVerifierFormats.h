/**
 * @file QuantizedVerifierFormats.h
 * @brief Canonical model-weight format registry for grouped verifier sweeps.
 *
 * Every CPU, CUDA, and ROCm grouped-verifier test must derive its quantized
 * matrix from this registry.  Backend tests may replace a creator with a
 * stronger nonzero fixture, but they must preserve the complete entry set.
 * Keeping source and device codebook metadata beside the creator also catches
 * preparation bugs where a source layout is accepted but published under an
 * incompatible execution descriptor.
 */

#pragma once

#include "TestTensorFactory.h"

#include "loaders/gpu_pipeline/RepackFormat.h"

#include <cstring>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace llaminar2::test
{
    /**
     * @brief Signature shared by deterministic quantized test-weight creators.
     */
    using QuantizedVerifierWeightCreator = std::function<std::unique_ptr<TensorBase>(
        const std::vector<size_t> &shape,
        uint32_t seed)>;

    /**
     * @brief One loadable quantized model-weight format and its packing contract.
     */
    struct QuantizedVerifierFormatCase
    {
        const char *label;                         ///< Stable tensor-format name.
        TensorType tensor_type;                    ///< Native source tensor type.
        uint8_t source_codebook_id;                ///< Preparation/source-layout id.
        bool source_is_superblock;                 ///< True for 256-value source blocks.
        uint8_t device_execution_codebook_id;      ///< Published device GEMM/MoE id.
        QuantizedVerifierWeightCreator create;     ///< Deterministic native tensor factory.
    };

    /**
     * @brief Return every quantized model-weight format accepted by V2 loaders.
     *
     * Q8_1 is primarily an activation format, but WeightManager and GEMM accept
     * it as a prepared tensor.  It remains in this exhaustive matrix so the raw
     * INT8 packing family is proved as a whole alongside model-loadable Q8_0 and
     * Q8_K.
     */
    inline const std::vector<QuantizedVerifierFormatCase> &quantizedVerifierFormats()
    {
        static const std::vector<QuantizedVerifierFormatCase> formats = {
            {"Q4_0", TensorType::Q4_0, 0, false, 0, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createQ4_0Random(shape, seed); }},
            {"IQ4_NL", TensorType::IQ4_NL, 4, false, 4, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createIQ4_NLRandom(shape, seed); }},
            {"IQ4_XS", TensorType::IQ4_XS, 4, true, 4, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createIQ4_XSRandom(shape, seed); }},
            {"Q4_1", TensorType::Q4_1, 5, false, 5, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createQ4_1Random(shape, seed); }},
            {"Q4_K", TensorType::Q4_K, 5, true, 5, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createQ4_KRandom(shape, seed); }},
            {"Q5_0", TensorType::Q5_0, 6, false, 6, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createQ5_0Random(shape, seed); }},
            {"Q5_1", TensorType::Q5_1, 7, false, 7, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createQ5_1Random(shape, seed); }},
            {"Q5_K", TensorType::Q5_K, 7, true, 7, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createQ5_KRandom(shape, seed); }},
            {"Q6_K", TensorType::Q6_K, 8, true, 8, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createQ6_KRandom(shape, seed); }},
            {"Q3_K", TensorType::Q3_K, 9, true, 9, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createQ3_KRandom(shape, seed); }},
            {"Q2_K", TensorType::Q2_K, 10, true, 10, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createQ2_KRandom(shape, seed); }},
            {"IQ3_S", TensorType::IQ3_S, 11, true, 11, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createIQ3_SRandom(shape, seed); }},
            {"IQ3_XXS", TensorType::IQ3_XXS, 12, true, 12, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createIQ3_XXSRandom(shape, seed); }},
            {"IQ2_S", TensorType::IQ2_S, 13, true, 13, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createIQ2_SRandom(shape, seed); }},
            {"IQ2_XS", TensorType::IQ2_XS, 14, true, 14, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createIQ2_XSRandom(shape, seed); }},
            {"IQ2_XXS", TensorType::IQ2_XXS, 15, true, 15, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createIQ2_XXSRandom(shape, seed); }},
            {"IQ1_S", TensorType::IQ1_S, 16, true, 16, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createIQ1_SRandom(shape, seed); }},
            {"IQ1_M", TensorType::IQ1_M, 17, true, 17, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createIQ1_MRandom(shape, seed); }},
            {"Q8_0", TensorType::Q8_0, 19, false, 19, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createQ8_0Random(shape, seed); }},
            {"Q8_1", TensorType::Q8_1, 20, false, 19, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createQ8_1Random(shape, -1.0f, 1.0f, seed); }},
            {"Q8_K", TensorType::Q8_K, 21, true, 19, [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createQ8_KRandom(shape, seed); }},
        };
        return formats;
    }

    /**
     * @brief Create bounded IQ4_XS weights for non-degenerate MoE witnesses.
     *
     * IQ4_XS encodes each sub-block scale as `d * (ls - 32)`.  The generic
     * random factory is useful for tensor-decoder coverage, but its synthetic
     * scales can make a gate/up/down composition overflow or collapse to zero.
     * Keeping `ls` close to 32 produces finite, nonzero routed-expert rows while
     * preserving the real IQ4_XS source layout and codebook-4 preparation path.
     */
    inline std::unique_ptr<TensorBase> createBoundedMoEIQ4XS(
        const std::vector<size_t> &shape,
        uint32_t seed)
    {
        constexpr size_t block_size = IQ4_XSBlock::BLOCK_SIZE;
        const size_t rows = shape.at(0);
        const size_t cols = shape.at(1);
        const size_t blocks_per_row = (cols + block_size - 1) / block_size;
        const size_t total_blocks = rows * blocks_per_row;

        std::vector<uint8_t> raw_data(total_blocks * sizeof(IQ4_XSBlock));
        auto *blocks = reinterpret_cast<IQ4_XSBlock *>(raw_data.data());
        for (size_t block_idx = 0; block_idx < total_blocks; ++block_idx)
        {
            auto &block = blocks[block_idx];
            block.d = 0x2400; // FP16 0.015625.
            block.scales_h = 0;
            std::memset(block.scales_l, 0, sizeof(block.scales_l));

            for (int sub = 0; sub < 8; ++sub)
            {
                const uint16_t ls = static_cast<uint16_t>(
                    33u + ((seed + block_idx + static_cast<size_t>(sub)) & 0x3u));
                block.scales_l[sub / 2] |= static_cast<uint8_t>(
                    (ls & 0x0fu) << (4 * (sub & 1)));
                block.scales_h |= static_cast<uint16_t>(
                    ((ls >> 4) & 0x3u) << (2 * sub));
            }

            for (size_t value = 0; value < std::size(block.qs); ++value)
            {
                const uint8_t lo = static_cast<uint8_t>(
                    (seed + block_idx * 19u + value * 5u) & 0x0fu);
                const uint8_t hi = static_cast<uint8_t>(
                    ((seed >> 4) + block_idx * 23u + value * 7u) & 0x0fu);
                block.qs[value] = static_cast<uint8_t>(lo | (hi << 4));
            }
        }

        return std::make_unique<IQ4_XSTensor>(shape, raw_data);
    }

    /**
     * @brief Create signed, nonzero IQ3_S weights for routed-expert sweeps.
     *
     * The generic random helper intentionally initializes only a minimal IQ3_S
     * representation.  That is too weak for an FFN regression because both a
     * broken and a correct grouped path can produce an all-zero row.  This
     * creator fills every payload, high-bit, sign, and scale plane so a byte-
     * equality assertion is backed by a non-degenerate production descriptor.
     */
    inline std::unique_ptr<TensorBase> createNonzeroMoEIQ3S(
        const std::vector<size_t> &shape,
        uint32_t seed)
    {
        constexpr size_t block_size = IQ3_SBlock::BLOCK_SIZE;
        const size_t rows = shape.at(0);
        const size_t cols = shape.at(1);
        const size_t blocks_per_row = (cols + block_size - 1) / block_size;
        const size_t total_blocks = rows * blocks_per_row;

        std::vector<uint8_t> raw_data(total_blocks * sizeof(IQ3_SBlock));
        auto *blocks = reinterpret_cast<IQ3_SBlock *>(raw_data.data());
        for (size_t block_idx = 0; block_idx < total_blocks; ++block_idx)
        {
            auto &block = blocks[block_idx];
            block.d = 0x3000; // FP16 0.125.

            for (size_t value = 0; value < std::size(block.qs); ++value)
            {
                block.qs[value] = static_cast<uint8_t>(
                    (seed + block_idx * 37u + value * 13u) & 0xffu);
            }
            for (size_t value = 0; value < std::size(block.qh); ++value)
            {
                block.qh[value] = static_cast<uint8_t>(
                    ((seed >> 3) + block_idx * 11u + value * 29u) & 0xffu);
            }
            for (size_t value = 0; value < std::size(block.signs); ++value)
            {
                block.signs[value] = static_cast<uint8_t>(
                    (0x5au ^ seed ^ (block_idx * 17u + value * 7u)) & 0xffu);
            }
            for (size_t value = 0; value < std::size(block.scales); ++value)
            {
                const uint8_t lo = static_cast<uint8_t>(
                    (1u + seed + block_idx + value) & 0x3u);
                const uint8_t hi = static_cast<uint8_t>(
                    (2u + (seed >> 2) + block_idx + value) & 0x3u);
                block.scales[value] = static_cast<uint8_t>(lo | (hi << 4));
            }
        }

        return std::make_unique<IQ3_STensor>(shape, raw_data);
    }

    /**
     * @brief Return the canonical format-complete MoE verifier registry.
     *
     * Entry identity, source metadata, and device codebook IDs remain exactly
     * those of @ref quantizedVerifierFormats.  Only synthetic creators known to
     * produce degenerate composed FFN witnesses are strengthened.  Every CPU,
     * CUDA, and ROCm routed-expert sweep should consume this registry rather
     * than maintaining backend-private format exceptions.
     */
    inline const std::vector<QuantizedVerifierFormatCase> &quantizedMoEVerifierFormats()
    {
        static const std::vector<QuantizedVerifierFormatCase> formats = []
        {
            auto result = quantizedVerifierFormats();
            for (auto &format : result)
            {
                if (format.tensor_type == TensorType::IQ4_XS)
                    format.create = createBoundedMoEIQ4XS;
                else if (format.tensor_type == TensorType::IQ3_S)
                    format.create = createNonzeroMoEIQ3S;
            }
            return result;
        }();
        return formats;
    }

    /**
     * @brief Resolve a canonical format entry by its stable label.
     * @throws std::invalid_argument when a test names a non-canonical format.
     */
    inline const QuantizedVerifierFormatCase &quantizedVerifierFormat(std::string_view label)
    {
        for (const auto &format : quantizedVerifierFormats())
        {
            if (label == format.label)
                return format;
        }
        throw std::invalid_argument("Unknown canonical verifier format: " + std::string(label));
    }
} // namespace llaminar2::test
