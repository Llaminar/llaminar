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
#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <random>
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
     * Independent PRNG draws for each payload and scale avoid short row periods:
     * correlated two-row patterns previously saturated every gate negative and
     * underflowed the composed SwiGLU Q8 scale, making replica parity vacuous.
     *
     * @param shape Native matrix dimensions [rows, cols].
     * @param seed Stable seed for independent packed payload and scale draws.
     * @return Bounded native IQ4_XS tensor with varied rows and expert slices.
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
        std::mt19937 rng(seed);
        for (size_t block_idx = 0; block_idx < total_blocks; ++block_idx)
        {
            auto &block = blocks[block_idx];
            block.d = 0x2400; // FP16 0.015625.
            block.scales_h = 0;
            std::memset(block.scales_l, 0, sizeof(block.scales_l));

            for (int sub = 0; sub < 8; ++sub)
            {
                const uint16_t ls = static_cast<uint16_t>(
                    33u + (rng() & 0x3u));
                block.scales_l[sub / 2] |= static_cast<uint8_t>(
                    (ls & 0x0fu) << (4 * (sub & 1)));
                block.scales_h |= static_cast<uint16_t>(
                    ((ls >> 4) & 0x3u) << (2 * sub));
            }

            for (size_t value = 0; value < std::size(block.qs); ++value)
            {
                const uint8_t lo = static_cast<uint8_t>(rng() & 0x0fu);
                const uint8_t hi = static_cast<uint8_t>(rng() & 0x0fu);
                block.qs[value] = static_cast<uint8_t>(lo | (hi << 4));
            }
        }

        return std::make_unique<IQ4_XSTensor>(shape, raw_data);
    }

    /**
     * @brief Bound a composed verifier input to the native Q8 metadata domain.
     * @param gate Native gate weights before preparation retires source storage.
     * @param up Native up weights with the same reduction dimension.
     * @param maximum_hidden_value Largest absolute value across all input patterns.
     * @return Exact power-of-two input multiplier, at most one.
     * @throws std::invalid_argument for invalid dimensions or nonfinite fixtures.
     *
     * Generic decoder fixtures deliberately use very different weight scales.
     * An unnormalized Q8_K matrix can produce billion-scale SwiGLU values from
     * the wide hidden pattern, outside binary16 Q8 scale storage. A row L1 bound
     * limits both projections to 512, hence their product to 262144. Scaling the
     * input by a power of two retains its pattern and relative amplitude sweep
     * without re-encoding any native weight, codebook or correction plane.
     */
    inline float composedMoEVerifierInputMultiplier(
        const TensorBase &gate, const TensorBase &up, float maximum_hidden_value)
    {
        if (gate.shape().size() != 2 || up.shape().size() != 2 ||
            gate.cols() != up.cols() || maximum_hidden_value <= 0 ||
            !std::isfinite(maximum_hidden_value))
            throw std::invalid_argument("Composed MoE fixture has invalid input geometry");
        double maximum_projection = 0;
        std::vector<float> row(static_cast<size_t>(gate.cols()));
        for (const TensorBase *weights : {&gate, &up})
            for (size_t r = 0; r < static_cast<size_t>(weights->rows()); ++r)
            {
                weights->to_fp32_row(r, row.data());
                double norm = 0;
                for (const float value : row)
                {
                    if (!std::isfinite(value))
                        throw std::invalid_argument("Composed MoE fixture has nonfinite weights");
                    norm += std::abs(static_cast<double>(value));
                }
                maximum_projection = std::max(maximum_projection,
                    norm * static_cast<double>(maximum_hidden_value));
            }
        float multiplier = 1.f;
        while (maximum_projection * multiplier > 512.) multiplier *= .5f;
        if (multiplier == 0.f)
            throw std::invalid_argument("Composed MoE fixture input bound underflowed");
        return multiplier;
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

    /**
     * @brief Resolve a non-degenerate MoE fixture by its stable source label.
     *
     * Production-derived mixed-format sweeps must use the same strengthened
     * IQ3_S/IQ4_XS fixtures as the homogeneous all-format verifier sweep. This
     * helper prevents a caller from accidentally resolving the generic tensor
     * creator and certifying an all-zero composed FFN witness.
     *
     * @throws std::invalid_argument when @p label is not a canonical format.
     */
    inline const QuantizedVerifierFormatCase &quantizedMoEVerifierFormat(
        std::string_view label)
    {
        for (const auto &format : quantizedMoEVerifierFormats())
        {
            if (label == format.label)
                return format;
        }
        throw std::invalid_argument(
            "Unknown canonical MoE verifier format: " + std::string(label));
    }
} // namespace llaminar2::test
