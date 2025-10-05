/**
 * @file abstract_pipeline.h
 * @brief Architecture-agnostic inference pipeline base (multi-architecture scaffolding).
 * @author David Sanftenberg
 */
#pragma once

#include "transformer_config.h"
#include "tensors/tensor_base.h"
#include <memory>
#include <vector>
#include <string>

namespace llaminar
{
    /**
     * @brief InferenceStage enumerates coarse pipeline phases for backend selection and diagnostics.
     */
    enum class InferenceStage : uint8_t
    {
        Prefill = 0,
        Decode = 1
    };

    /**
     * @brief Simple stage context passed through execute paths.
     */
    struct StageContext
    {
        InferenceStage stage{InferenceStage::Prefill};
        int seq_len = 0;     // current prompt / token window length
        int generated = 0;   // number of tokens already generated (decode context)
        int kv_capacity = 0; // total allocated KV slots (per layer) if available
        int kv_used = 0;     // populated KV token rows
    };

    /**
     * @brief Shared KV cache tracking state (architecture agnostic).
     */
    struct KVCacheState
    {
        int capacity_tokens = 0; ///< Allocated token rows per layer
        int used_tokens = 0;     ///< Rows populated with valid K/V entries
        int growth_events = 0;   ///< Number of reallocations/growth operations
    };

    /**
     * @brief Interface for model weights (architecture independent façade).
     */
    struct IModelWeights
    {
        virtual ~IModelWeights() = default;
    };

    /**
     * @brief Abstract multi-architecture pipeline.
     * Derivations implement prefill/decode; legacy MPITransformerPipeline
     * will be adapted behind a feature flag.
     */
    class AbstractPipeline
    {
    public:
        virtual ~AbstractPipeline() = default;

        /**
         * @brief Get the model configuration.
         *
         * Returns the full ModelConfig including architecture and feature flags.
         * For backward compatibility, the layer config can be accessed via config().getLayerConfig().
         */
        virtual const ModelConfig &config() const = 0;

        virtual bool prefill(const std::vector<int> &tokens,
                             const IModelWeights &weights,
                             StageContext &ctx) = 0;
        virtual bool decode(int next_token,
                            const IModelWeights &weights,
                            StageContext &ctx) = 0;
        virtual bool logits(std::shared_ptr<TensorBase> &out_logits) = 0; // obtain last computed logits
        virtual std::string name() const = 0;

        /**
         * @brief Load model weights from file path.
         *
         * @param path Path to the model file (e.g., GGUF file)
         * @return Loaded weights wrapped in architecture-specific IModelWeights implementation
         *
         * This replaces the legacy free loadModelWeights() functions with a virtual
         * method that allows each architecture adapter to handle its own loading logic.
         */
        virtual std::unique_ptr<IModelWeights> loadWeights(const std::string &path) = 0;

        // KV cache management (optional – implementations without KV caching may return nullptr / false)
        virtual const KVCacheState *kvCacheState() const { return nullptr; }
        virtual bool ensureKVCapacity(int /*required_tokens*/) { return true; }
    };

    /**
     * @brief Factory for creating architecture-specific pipelines.
     */
    class PipelineFactory
    {
    public:
        using CreateFn = std::unique_ptr<AbstractPipeline> (*)(const ModelConfig &);
        static PipelineFactory &instance();
        void registerCreator(const std::string &arch, CreateFn fn);
        std::unique_ptr<AbstractPipeline> create(const ModelConfig &cfg) const;

    private:
        PipelineFactory() = default;
        std::vector<std::pair<std::string, CreateFn>> creators_; // small list, linear scan fine
    };

} // namespace llaminar
