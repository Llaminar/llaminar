/**
 * @file InferenceRunnerFactory.h
 * @brief Factory for creating IInferenceRunner implementations
 * @author David Sanftenberg
 * @date December 2025
 *
 * This file provides a factory function for creating inference runners.
 * The factory handles:
 * - Building Qwen2GraphConfig from GGUF model metadata
 * - Configuring tensor parallelism (head/FFN/vocab sharding) based on MPI world size
 * - Loading and wiring up model weights
 * - Initializing GraphOrchestrator with proper state
 *
 * The actual interface (IInferenceRunner) is defined in IInferenceRunner.h.
 *
 * @code
 * // Create runner
 * auto runner = createInferenceRunner(model_ctx, mpi_ctx, device_idx, config);
 *
 * // Inference
 * runner->forward(tokens, seq_len);
 * const float* logits = runner->logits();
 * @endcode
 */

#pragma once

#include "IInferenceRunner.h"
#include "../loaders/ModelContext.h"
#include "RuntimeConfig.h"
#include "../utils/MPIContext.h"
#include <memory>

namespace llaminar2
{
    // Forward declarations to avoid pulling in full headers
    class GraphOrchestrator;

    /**
     * @brief Configuration for inference runner creation
     */
    struct InferenceRunnerConfig
    {
        int max_seq_len = 4096;
        int batch_size = 1;

        // Explicit graph path selection (only graph path is supported)
        bool force_graph = false;

        // Activation precision
        ActivationPrecision activation_precision = ActivationPrecision::FP32;

        // Fused attention backend selection
        // - JIT: AVX-512 VNNI optimized (default)
        // - REFERENCE: Pure C++ implementation for testing
        // - TILED: Cache-blocked implementation
        // - Q16_INTEGER: Pure Q16 integer-domain kernel (experimental)
        FusedAttentionBackend fused_attention_backend = FusedAttentionBackend::JIT;

        // Fixed scale for Q16_1 KV cache quantization (FP32 range: ±kv_cache_scale)
        // See RuntimeConfig.h for detailed documentation on scale selection.
        // Default 64.0f covers Q8_1 activation ranges (max ±35) with ~2× headroom.
        float kv_cache_scale = 256.0f; ///< Fixed Q16 scale. Must cover Q projection max (~130 for Qwen2)
    };

    /**
     * @brief Factory function to create GraphOrchestrator inference runner
     *
     * @param model_ctx Model context with weights
     * @param mpi_ctx MPI context (nullptr for single-rank)
     * @param device_idx Target device
     * @param config Runner configuration
     * @return Unique pointer to IInferenceRunner, or nullptr on failure
     */
    std::unique_ptr<IInferenceRunner> createInferenceRunner(
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<MPIContext> mpi_ctx,
        int device_idx,
        const InferenceRunnerConfig &config = {});

} // namespace llaminar2
