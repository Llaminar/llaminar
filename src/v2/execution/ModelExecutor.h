/**
 * @file ModelExecutor.h
 * @brief Base implementation of model-level execution orchestration
 * @author David Sanftenberg
 * @date December 2025
 *
 * ModelExecutor provides the core implementation of IModelExecutor,
 * handling compute graph execution and delegation to LayerExecutor.
 * Architecture-specific details are delegated to derived classes
 * (e.g., Qwen2ModelExecutor).
 *
 * Key Responsibilities:
 * - Graph execution orchestration
 * - Device context management
 * - Statistics collection
 * - Snapshot callback propagation
 *
 * Architecture-specific responsibilities (in derived classes):
 * - Graph building (buildFullForwardGraph, buildLayerGraph, etc.)
 * - Weight access patterns
 * - Buffer allocation strategies
 */

#pragma once

#include "IModelExecutor.h"
#include "LayerExecutor.h"
#include "DeviceContext.h"
#include "../backends/DeviceId.h"
#include "../loaders/ModelContext.h"
#include "../utils/MPIContext.h"
#include "../tensors/TensorFactory.h"
#include <memory>
#include <unordered_map>
#include <chrono>

namespace llaminar2
{

    /**
     * @brief Base implementation of model-level execution
     *
     * ModelExecutor handles the common logic for executing compute graphs
     * across the full forward pass. Derived classes implement architecture-
     * specific graph building.
     */
    class ModelExecutor : public IModelExecutor
    {
    public:
        /**
         * @brief Construct ModelExecutor
         *
         * @param model_ctx Model context with GGUF metadata and weights
         * @param mpi_ctx MPI context (nullptr for single-rank)
         * @param config Executor configuration
         */
        ModelExecutor(std::shared_ptr<ModelContext> model_ctx,
                      std::shared_ptr<MPIContext> mpi_ctx,
                      const ModelExecutorConfig &config = ModelExecutorConfig{});

        ~ModelExecutor() override = default;

        // Non-copyable, movable
        ModelExecutor(const ModelExecutor &) = delete;
        ModelExecutor &operator=(const ModelExecutor &) = delete;
        ModelExecutor(ModelExecutor &&) = default;
        ModelExecutor &operator=(ModelExecutor &&) = default;

        // =========================================================================
        // Configuration (IModelExecutor interface)
        // =========================================================================

        const ModelExecutorConfig &config() const override { return config_; }

        void setSnapshotCallback(StageSnapshotCallback callback) override
        {
            snapshot_callback_ = std::move(callback);
            layer_executor_.setSnapshotCallback(snapshot_callback_);
        }

        // =========================================================================
        // Graph Building (Must be implemented by derived classes)
        // =========================================================================

        ComputeGraph buildFullForwardGraph(
            const ForwardInput &input,
            ForwardOutput &output) override = 0;

        ComputeGraph buildEmbeddingGraph(
            const ForwardInput &input,
            TensorBase *output_hidden) override = 0;

        ComputeGraph buildTransformerLayersGraph(
            TensorBase *input_hidden,
            ICPUKVCache *kv_cache,
            const int *position_ids,
            DeviceId device) override = 0;

        ComputeGraph buildLayerGraph(
            int layer_idx,
            TensorBase *input_hidden,
            ICPUKVCache *kv_cache,
            const int *position_ids,
            DeviceId device) override = 0;

        ComputeGraph buildLMHeadGraph(
            TensorBase *hidden_states,
            TensorBase *output_logits,
            int total_tokens,
            DeviceId device,
            TensorBase *logits_local = nullptr) override = 0;

        // =========================================================================
        // Execution (IModelExecutor interface)
        // =========================================================================

        bool executeForward(
            const ForwardInput &input,
            ForwardOutput &output) override;

        bool execute(ComputeGraph &graph, IDeviceContext *ctx) override;

        // =========================================================================
        // Statistics (IModelExecutor interface)
        // =========================================================================

        const ModelExecutorStats &stats() const override { return stats_; }

        void resetStats() override { stats_.reset(); }

        // =========================================================================
        // State Management (IModelExecutor interface)
        // =========================================================================

        void clearCache() override;

    protected:
        // Context
        std::shared_ptr<ModelContext> model_ctx_;
        std::shared_ptr<MPIContext> mpi_ctx_;
        ModelExecutorConfig config_;

        // Layer executor for per-layer orchestration
        LayerExecutor layer_executor_;

        // Device contexts (created lazily)
        std::unordered_map<DeviceId, std::unique_ptr<IDeviceContext>> device_contexts_;

        // Statistics
        ModelExecutorStats stats_;

        // Snapshot callback
        StageSnapshotCallback snapshot_callback_;

        // =========================================================================
        // Helper Methods
        // =========================================================================

        /**
         * @brief Get or create device context for given device
         */
        IDeviceContext *getDeviceContext(DeviceId device);

        /**
         * @brief Execute embedding phase with timing
         */
        bool executeEmbedding(const ForwardInput &input, TensorBase *output_hidden);

        /**
         * @brief Execute transformer layers with timing
         */
        bool executeTransformerLayers(
            TensorBase *hidden,
            ICPUKVCache *kv_cache,
            const int *position_ids,
            DeviceId device);

        /**
         * @brief Execute LM head with timing
         * @param hidden Hidden states tensor
         * @param logits Output logits tensor (full vocab size)
         * @param total_tokens Number of tokens (batch_size * seq_len)
         * @param device Target device
         * @param logits_local Optional local logits for column-parallel LM head
         */
        bool executeLMHead(TensorBase *hidden, TensorBase *logits, int total_tokens, DeviceId device,
                           TensorBase *logits_local = nullptr);

        /**
         * @brief Record timing for a phase
         */
        void recordPhaseTime(const std::string &phase, double ms);
    };

} // namespace llaminar2
