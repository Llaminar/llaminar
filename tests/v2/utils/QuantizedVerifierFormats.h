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
