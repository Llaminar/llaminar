/**
 * @file IModelContext.h
 * @brief Interface for complete model context (loader + weights + metadata)
 *
 * Abstracts the complete model setup to enable:
 * 1. Testing inference without GGUF files
 * 2. Mock model configurations
 * 3. Testing different model architectures
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../loaders/IModelLoader.h"
#include "../loaders/IWeightManager.h"
#include <memory>
#include <string>

namespace llaminar2
{

    /**
     * @brief Interface for complete model context (loader + weights + metadata)
     *
     * Facade that combines model loading, weight management, and metadata access.
     * Abstracts the complete model setup to enable:
     * 1. Testing inference without GGUF files
     * 2. Mock model configurations
     * 3. Testing different model architectures
     *
     * Implementations:
     * - ModelContext: Real implementation with GGUF file backing
     * - MockModelContext: Test implementation with in-memory model configuration
     *
     * Usage:
     * @code
     * // Production code works with interface
     * void initializePipeline(IModelContext& ctx) {
     *     auto embd = ctx.getWeightForDevice("token_embd.weight");
     *     int num_layers = ctx.blockCount();
     *     auto& loader = ctx.loader();
     *     // ...
     * }
     *
     * // Test code uses mock
     * auto mock = MockModelContextBuilder()
     *     .usePreset(ModelPreset::QWEN2_05B)
     *     .addWeight("token_embd.weight", createFP32Tensor({151936, 896}))
     *     .build();
     * initializePipeline(*mock);
     * @endcode
     */
    class IModelContext
    {
    public:
        virtual ~IModelContext() = default;

        // =========================================================================
        // Path and Architecture Access
        // =========================================================================

        /**
         * @brief Get model file path
         * @return Path to the model file (may be dummy path for mocks)
         */
        virtual const std::string &path() const = 0;

        /**
         * @brief Get architecture string
         * @return Architecture identifier (e.g., "qwen2", "llama", "gpt2")
         */
        virtual const std::string &architecture() const = 0;

        // =========================================================================
        // Component Access
        // =========================================================================

        /**
         * @brief Get the model loader interface
         *
         * For production: Returns real ModelLoader
         * For testing: Returns MockModelLoader
         *
         * @return Shared pointer to loader interface
         */
        virtual std::shared_ptr<IModelLoader> loader() = 0;

        /**
         * @brief Get the weight manager interface
         *
         * For production: Returns real WeightManager
         * For testing: Returns MockWeightManager
         *
         * @return Shared pointer to weight manager interface
         */
        virtual std::shared_ptr<IWeightManager> weightManager() = 0;

        // =========================================================================
        // Convenience Hyperparameter Accessors
        // =========================================================================

        /**
         * @brief Get number of transformer layers
         */
        virtual int blockCount() const = 0;

        /**
         * @brief Get hidden dimension (d_model)
         */
        virtual int embeddingLength() const = 0;

        /**
         * @brief Get number of attention heads
         */
        virtual int headCount() const = 0;

        /**
         * @brief Get number of key-value heads (for GQA)
         */
        virtual int headCountKV() const = 0;

        /**
         * @brief Get vocabulary size
         */
        virtual int vocabSize() const = 0;

        /**
         * @brief Get maximum context length
         */
        virtual int contextLength() const = 0;

        /**
         * @brief Get feed-forward hidden dimension (intermediate_size)
         *
         * For Qwen2/Llama models, this is the FFN intermediate dimension used in
         * gate/up/down projections. Typically ~5.5x d_model for SwiGLU architectures.
         *
         * Example: Qwen2-0.5B has d_model=896, feedForwardLength=4864
         *
         * @return FFN intermediate dimension, or 0 if unknown
         */
        virtual int feedForwardLength() const = 0;

        /**
         * @brief Get attention head dimension (key_length from GGUF)
         *
         * Returns 0 if not explicitly set; caller should fall back to d_model / n_heads.
         */
        virtual int keyLength() const = 0;

        // =========================================================================
        // Weight Access (Delegates to WeightManager)
        // =========================================================================

        /**
         * @brief Get weight tensor for a specific device (device-isolated instance)
         *
         * For multi-device scenarios (LOCAL TP), each device needs its own tensor
         * instance to track coherence state independently.
         *
         * Convenience method that delegates to weightManager()->getWeightForDevice().
         *
         * @param name GGUF tensor name (e.g., "token_embd.weight", "blk.0.attn_q.weight")
         * @param device Target device for this tensor instance (default: CPU)
         * @return Device-specific tensor instance, or nullptr on error
         */
        virtual std::shared_ptr<TensorBase> getWeightForDevice(
            const std::string &name,
            DeviceId device = DeviceId::cpu()) = 0;

        // =========================================================================
        // Tensor Existence Check (Delegates to Loader)
        // =========================================================================

        /**
         * @brief Check if a tensor exists in the model
         *
         * Convenience method that delegates to loader()->hasTensor().
         *
         * @param name GGUF tensor name to check
         * @return true if tensor exists, false otherwise
         */
        virtual bool hasTensor(const std::string &name) const = 0;
    };

} // namespace llaminar2
