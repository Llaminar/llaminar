/**
 * @file AbstractPipeline.h
 * @brief Architecture-agnostic inference pipeline base (multi-architecture scaffolding).
 * @author David Sanftenberg
 */
#pragma once

#include "TransformerConfig.h"
#include "tensors/TensorBase.h"
#include "PipelineStages.h"
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
        int batch_size = 1;  // number of sequences in batch (1 for single sequence)
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

        // === Single-Sequence Methods (Original Interface) ===

        virtual bool prefill(const std::vector<int> &tokens,
                             const IModelWeights &weights,
                             StageContext &ctx) = 0;
        virtual bool decode(int next_token,
                            const IModelWeights &weights,
                            StageContext &ctx) = 0;
        virtual bool logits(std::shared_ptr<TensorBase> &out_logits) = 0; // obtain last computed logits
        virtual std::string name() const = 0;

        // === Batch Processing Methods (Parallel Batching) ===

        /**
         * @brief Process multiple sequences in parallel (prefill phase)
         *
         * @param token_batches Vector of token sequences, one per sequence in batch
         * @param weights Model weights
         * @param ctx Stage context (will be updated with aggregate stats)
         * @param out_logits Output logits tensor [batch, max_seq_len, vocab_size]
         * @return true if successful, false otherwise
         *
         * Default implementation: not supported (returns false)
         */
        virtual bool prefillBatch(
            const std::vector<std::vector<int>> &token_batches,
            const IModelWeights &weights,
            StageContext &ctx,
            std::shared_ptr<TensorBase> &out_logits)
        {
            (void)token_batches;
            (void)weights;
            (void)ctx;
            (void)out_logits;
            return false; // Not implemented by default
        }

        /**
         * @brief Generate next token for multiple sequences in parallel (decode phase)
         *
         * @param next_tokens Next token for each sequence [batch_size]
         * @param weights Model weights
         * @param ctx Stage context
         * @param out_logits Output logits tensor [batch, vocab_size]
         * @return true if successful, false otherwise
         *
         * Default implementation: not supported (returns false)
         */
        virtual bool decodeBatch(
            const std::vector<int> &next_tokens,
            const IModelWeights &weights,
            StageContext &ctx,
            std::shared_ptr<TensorBase> &out_logits)
        {
            (void)next_tokens;
            (void)weights;
            (void)ctx;
            (void)out_logits;
            return false; // Not implemented by default
        }

        /**
         * @brief Reset pipeline for batch processing
         *
         * @param batch_size Number of sequences in batch
         *
         * Allocates/initializes state for processing batch_size sequences simultaneously.
         * Default implementation: no-op (single-sequence pipelines ignore this)
         */
        virtual void resetBatch(size_t batch_size)
        {
            (void)batch_size;
            // No-op by default
        }

        /**
         * @brief Get current batch size
         *
         * @return Current batch size (1 for single-sequence pipelines)
         */
        virtual size_t currentBatchSize() const
        {
            return 1; // Single-sequence by default
        }

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

        // === Parity Testing and Instrumentation ===

        /**
         * @brief Capture a snapshot of intermediate pipeline state for parity testing
         *
         * This is an optional hook that pipelines can implement to enable snapshot capture
         * for comparison with reference implementations (e.g., PyTorch). The default
         * implementation delegates to PipelineSnapshotManager, which is:
         * - Zero overhead in release builds (compiled out)
         * - Controlled by LLAMINAR_PARITY_CAPTURE=1 in debug builds
         * - Thread-safe and MPI-aware (typically only rank 0 captures)
         *
         * @param stage Pipeline stage identifier
         * @param layer_index Layer index (-1 for non-layer stages like embedding, final norm)
         * @param data Pointer to tensor data (float array)
         * @param seq_len Sequence length dimension
         * @param feature_dim Feature dimension (hidden size, vocab size, etc.)
         *
         * @note Override only if you need custom snapshot behavior
         * @note The default implementation uses PipelineSnapshotManager::instance()
         */
        virtual void captureStageSnapshot(
            PipelineStage stage,
            int layer_index,
            const float *data,
            int seq_len,
            int feature_dim,
            const std::string &source = "llaminar");

        /**
         * @brief Check if parity testing/snapshot capture is enabled
         *
         * @return true if snapshots should be captured, false otherwise
         *
         * Default implementation checks PipelineSnapshotManager, which reads
         * LLAMINAR_PARITY_CAPTURE environment variable in debug builds.
         * Always returns false in release builds.
         */
        virtual bool isParityEnabled() const;
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
