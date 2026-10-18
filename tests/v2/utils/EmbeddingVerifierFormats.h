/**
 * @file EmbeddingVerifierFormats.h
 * @brief Canonical embedding weight-format matrix for grouped verifier tests.
 *
 * CPU, CUDA, and ROCm must prove the same set of model embedding formats. A
 * single matrix prevents backend-local tests from quietly omitting a codebook
 * when formats are added or renamed. Each creator produces a native GGUF-style
 * tensor whose rows can be consumed by the production embedding preparation
 * and lookup APIs.
 */

#pragma once

#include "QuantizedVerifierFormats.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace llaminar2::test
{
    /**
     * @brief One embedding-table format exercised by every backend sweep.
     */
    struct EmbeddingVerifierFormatCase
    {
        const char *label;             ///< Stable diagnostic and perf-counter label.
        bool prepared_native_quantized; ///< True when GPU execution consumes original quantized source blocks.
        const char *cpu_weight_route;  ///< Expected CPU perfstats route.
        const char *gpu_weight_route;  ///< Expected CUDA/ROCm perfstats route.
        std::function<std::unique_ptr<TensorBase>(const std::vector<size_t> &, uint32_t)> create;
    };

    /**
     * @brief Return every floating or quantized embedding format accepted by V2.
     *
     * Keep this list in lockstep with the model loader's supported GGUF
     * codebooks. GPU tests prepare each quantized source through
     * KernelFactory::prepareEmbeddingHandleLocal(), while FP32 remains directly
     * resident on the target device.
     * @return Immutable canonical floating and quantized source-format inventory.
     */
    inline const std::vector<EmbeddingVerifierFormatCase> &embeddingVerifierFormats()
    {
        static const std::vector<EmbeddingVerifierFormatCase> formats = []
        {
            std::vector<EmbeddingVerifierFormatCase> result = {
                {"FP32", false, "direct_fp32", "resident_fp32", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
                 { return TestTensorFactory::createFP32Random(shape, -1.0f, 1.0f, seed); }},
                {"FP16", false, "native_fp16", "resident_fp16", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
                 { return TestTensorFactory::createFP16Random(shape, -1.0f, 1.0f, seed); }},
                {"BF16", false, "native_bf16", "resident_bf16", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
                 { return TestTensorFactory::createBF16Random(shape, -1.0f, 1.0f, seed); }},
            };
            result.reserve(result.size() + quantizedVerifierFormats().size());
            for (const auto &format : quantizedVerifierFormats())
            {
                result.push_back({
                    format.label,
                    true,
                    "native_quantized",
                    "prepared_device_native_quantized",
                    format.create});
            }
            return result;
        }();
        return formats;
    }
} // namespace llaminar2::test
