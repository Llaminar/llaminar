/**
 * @file QwenGraphBase.h
 * @brief Common base class for all Qwen-family compute graph builders
 * @author David Sanftenberg
 * @date January 2026
 *
 * QwenGraphBase extracts the shared graph-building infrastructure from QwenStandardGraph.
 * All Qwen-family models (Qwen2, Qwen3, Qwen3.5) inherit from this base class
 * rather than from each other, promoting a clean separation of model-specific
 * attention implementations from shared transformer infrastructure.
 *
 * The hierarchy is:
 *   IGraphBuilder (pure interface)
 *   └── QwenGraphBase (shared Qwen infrastructure)
 *       ├── QwenStandardGraph (standard multi-head attention)
 *       └── Qwen35Graph (hybrid GDN + full attention)
 *
 * Shared infrastructure includes:
 * - Full, partial, and unified forward graph construction
 * - Embedding, transformer layers, and LM head graph building
 * - FFN (SwiGLU) graph building
 * - Arena/buffer management and weight resolution
 * - TP allreduce stage creation and domain routing
 * - Schema-based graph resolution support
 *
 * Model-specific (pure virtual):
 * - architectureName() — model identifier string
 * - getSchema() — declarative GraphSchema
 * - buildAttentionGraph() — attention block (QKV→RoPE→attn→Wo vs GDN)
 */

#pragma once

#include "../GraphTypes.h"
#include "../../execution/local_execution/graph/DeviceGraphExecutor.h"
#include "../../execution/compute_stages/ComputeStages.h"
#include "../../execution/local_execution/device/DeviceContext.h"
#include "../../execution/config/ExecutionPolicy.h"
#include "../../memory/BufferArena.h"
#include "../../execution/local_execution/graph/IGraphBuilder.h"
#include "../../execution/config/RuntimeConfig.h"
#include "../../execution/local_execution/graph/GraphResolver.h"
#include "../../backends/DeviceId.h"
#include "../../tensors/Tensors.h"
#include "../../tensors/TensorFactory.h"
#include "../../kernels/cpu/CPUKVCache.h"
#include "../../loaders/ModelContext.h"
#include "../../utils/MPIContext.h"
#include "../../config/TensorParallelConfig.h"
#include "../../config/TPDomain.h"
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <unordered_map>

namespace llaminar2
{

    // Forward declarations
    class ITensorGemm;
    class Qwen2Pipeline;

    /**
     * @brief Common base class for Qwen-family compute graph builders
     *
     * Provides shared graph-building infrastructure for all Qwen-family models.
     * Subclasses implement model-specific attention patterns and schemas.
     */
    class QwenGraphBase : public IGraphBuilder
    {
    public:
        /**
         * @brief Construct with full model context
         *
         * @param model_ctx Model context with GGUF metadata
         * @param mpi_ctx MPI context (nullptr for single-rank)
         * @param config Graph configuration
         */
        QwenGraphBase(std::shared_ptr<ModelContext> model_ctx,
                      std::shared_ptr<IMPIContext> mpi_ctx,
                      const GraphConfig &config);

        /**
         * @brief Construct for layer-level operations only
         *
         * Used when only layer-level graph building is needed,
         * without model-level operations like embedding or LM head.
         *
         * @param config Graph configuration
         * @param mpi_ctx MPI context (nullptr for single-rank)
         */
        QwenGraphBase(const GraphConfig &config,
                      std::shared_ptr<IMPIContext> mpi_ctx = nullptr);

        ~QwenGraphBase() override = default;

        // Non-copyable
        QwenGraphBase(const QwenGraphBase &) = delete;
        QwenGraphBase &operator=(const QwenGraphBase &) = delete;

        // =====================================================================
        // Pure Virtual (model-specific)
        // =====================================================================

        std::string architectureName() const override = 0;
        GraphSchema getSchema() const override = 0;

        /**
         * @brief Build attention block graph (model-specific)
         *
         * Each Qwen variant implements its own attention pattern:
         * - Qwen2/3: Standard QKV → RoPE → attention → Wo
         * - Qwen3.5: Dispatches between GDN and FA per layer
         */
        ComputeGraph buildAttentionGraph(
            const LayerWeights &layer,
            ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            int batch_size,
            IKVCache *kv_cache,
            const int *position_ids,
            DeviceId device,
            const std::vector<int> *sequence_lengths = nullptr) override = 0;

        // =====================================================================
        // Configuration
        // =====================================================================

        const GraphConfig &config() const override { return config_; }

        void setPipelineConfig(std::shared_ptr<PipelineConfig> pipeline_config) override
        {
            config_.pipeline_config = std::move(pipeline_config);
        }

        void setPPContext(int from_stage, int to_stage, ILocalPPContext *pp_ctx) override
        {
            config_.pp_contexts[{from_stage, to_stage}] = pp_ctx;
        }

        void setTPContext(const std::string &domain_name, ITPContext *tp_ctx) override
        {
            config_.domain_tp_contexts[domain_name] = tp_ctx;
        }

        void setWeights(const ModelWeights &weights) override { weights_ = weights; }
        void setWeightBindings(const ModelWeightBindings &bindings) override { weight_bindings_ = bindings; }
        void setBuffers(const ModelBuffers &buffers) override { buffers_ = buffers; }

        /**
         * @brief Set the BufferArena for arena-managed buffer resolution
         *
         * Populates buffers_ from the arena, allowing all graph-building
         * methods to use arena-allocated tensors via the existing buffers_ paths.
         */
        void setArena(BufferArena *arena) override;

        void setPreparedWeightStore(PreparedWeightStore *store) override
        {
            prepared_weight_store_ = store;
        }

        bool setComputeAllPositionLogits(bool enabled) override
        {
            config_.compute_all_position_logits = enabled;
            return true;
        }

        bool setComputeRowIndexedAllPositionLogits(bool enabled, int row_count) override
        {
            if (enabled && (row_count <= 0 || row_count > 4))
                return false;
            config_.compute_row_indexed_logits = enabled;
            config_.row_indexed_logits_row_count = enabled ? row_count : 0;
            return true;
        }

        void setModelContext(std::shared_ptr<IModelContext> model_ctx) override;

        BufferArena *arena() const { return arena_; }
        const ModelBuffers &buffers() const override;

        void setTensorFactory(TensorFactory *factory) { tensor_factory_ = factory; }

        /**
         * @brief Get resolver config for buffer allocation
         *
         * Creates a GraphResolverConfig populated with model dimensions,
         * including tensor-parallel local dimensions. Virtual so models
         * with extra formulas (e.g., GDN) can extend.
         */
        GraphResolverConfig getResolverConfig(int seq_len) const override;

        void setSnapshotCallback(StageSnapshotCallback callback) override
        {
            snapshot_callback_ = std::move(callback);
        }

        const StageSnapshotCallback &getSnapshotCallback() const { return snapshot_callback_; }

        // =====================================================================
        // IGraphBuilder Interface Implementation
        // =====================================================================

        ComputeGraph buildForwardGraph(
            const ForwardInput &input,
            ForwardOutput &output) override;

        ComputeGraph buildLayerGraph(const LayerContext &ctx) override;

        int numLayers() const override { return config_.n_layers; }
        int hiddenDim() const override { return config_.d_model; }

        bool isInitialized() const override
        {
            return weight_bindings_.get_layer_weights != nullptr || weights_.get_layer_weights != nullptr;
        }

        // =====================================================================
        // Model-Level Graph Building
        // =====================================================================

        ComputeGraph buildFullForwardGraph(
            const ForwardInput &input,
            ForwardOutput &output) override;

        ComputeGraph buildPartialForwardGraph(
            const ForwardInput &input,
            ForwardOutput &output,
            int first_layer,
            int last_layer,
            bool has_embedding,
            bool has_lm_head) override;

        ComputeGraph buildUnifiedPipelineGraph(
            const ForwardInput &input,
            ForwardOutput &output) override;

        ComputeGraph buildForwardGraphFromSchema(
            const ForwardInput &input,
            ForwardOutput &output);

        ComputeGraph buildEmbeddingGraph(
            const ForwardInput &input,
            TensorBase *output_hidden);

        ComputeGraph buildTransformerLayersGraph(
            TensorBase *input_hidden,
            IKVCache *kv_cache,
            const int *position_ids,
            DeviceId device);

        ComputeGraph buildLayerGraph(
            int layer_idx,
            TensorBase *input_hidden,
            IKVCache *kv_cache,
            const int *position_ids,
            DeviceId device);

        ComputeGraph buildLMHeadGraph(
            TensorBase *hidden_states,
            TensorBase *output_logits,
            int total_tokens,
            DeviceId device,
            TensorBase *logits_local = nullptr);

        // =====================================================================
        // Shared Layer-Level Graph Building
        // =====================================================================

        ComputeGraph buildFFNGraph(
            const LayerWeights &layer,
            ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            int batch_size,
            DeviceId device) override;

    protected:
        // =====================================================================
        // Configuration (protected for subclass access)
        // =====================================================================
        GraphConfig config_;
        std::shared_ptr<ModelContext> model_ctx_;
        std::shared_ptr<IMPIContext> mpi_ctx_;
        TensorFactory *tensor_factory_ = nullptr;
        BufferArena *arena_ = nullptr;
        PreparedWeightStore *prepared_weight_store_ = nullptr;
        ModelWeights weights_;
        ModelWeightBindings weight_bindings_;
        ModelBuffers buffers_;
        StageSnapshotCallback snapshot_callback_;

        // =====================================================================
        // Helpers
        // =====================================================================

        TensorContext buildTensorContext() const;
        bool needsTPAllreduce() const;
        bool hasActiveExpertMask(const std::vector<bool> &expert_mask) const;
        bool hasLayerWeightSource() const;
        LayerWeightBindings layerWeightBindingsForGraph(int layer_idx) const;
        LayerWeights layerWeightsForGraph(int layer_idx) const;
        TensorBase *modelEmbeddingTable() const;
        TensorBase *modelFinalNorm() const;
        TensorBase *modelLMHead() const;
        const WeightBinding *modelEmbeddingBinding() const;
        const WeightBinding *modelFinalNormBinding() const;
        const WeightBinding *modelLMHeadBinding() const;
        std::optional<PreparedWeightRef> preparedRefForGraphWeight(
            const WeightBinding *binding,
            DeviceId device) const;

        std::string describeMissingExpertGemmEngine(
            int num_experts,
            const std::vector<bool> &expert_mask,
            const std::vector<ITensorGemm *> &gate_gemm,
            const std::vector<ITensorGemm *> &up_gemm,
            const std::vector<ITensorGemm *> &down_gemm) const;

        /**
         * @brief Insert bucketed-prefill LM-head row selection when needed.
         *
         * Bucketed prefill graph replay must not bake a real-length-dependent
         * LM-head activation offset into the captured GEMM. When the input is a
         * fixed bucket, this helper adds a row-select stage from final norm into
         * the one-row LM-head scratch buffer and returns that scratch tensor.
         * Non-bucket graphs return final_norm_output unchanged.
         *
         * @param graph Graph receiving the optional row-select node.
         * @param dependency_node Node name that produces final_norm_output.
         * @param final_norm_output Full [seq_len, d_model] final norm output.
         * @param total_tokens Fixed graph token count for the bucket.
         * @param real_seq_len Real token count for initial execution (0 = total_tokens).
         * @param bucket_seq_len Bucket length marker (0 = non-bucket path).
         * @param device Device assigned to the row-select stage.
         * @param dependency_out Receives the node name LM head should depend on.
         * @param input_buffer_id BufferId for final_norm_output.
         * @return Tensor that LM head should read.
         */
        TensorBase *maybeAddLMHeadRowSelect(
            ComputeGraph &graph,
            const std::string &dependency_node,
            TensorBase *final_norm_output,
            int total_tokens,
            int real_seq_len,
            int bucket_seq_len,
            DeviceId device,
            std::string &dependency_out,
            BufferId input_buffer_id = BufferId::NORMALIZED) const;

        [[noreturn]] void failMissingGpuExpertGemmEngines(
            DeviceId device,
            int layer_idx,
            const std::string &reason) const;

        std::unique_ptr<IComputeStage> createTPAllreduceStage(
            TensorBase *buffer,
            size_t count,
            DeviceId device,
            int layer_idx,
            bool is_attention,
            const std::string &stage_name = "",
            std::optional<BufferId> tensor_buffer_id = std::nullopt) const;

        // =====================================================================
        // Shared Attention Building Blocks
        // =====================================================================

        /** Resolve TP-aware local head counts. */
        std::pair<int, int> resolveLocalHeadCounts() const;

        /**
         * Add pre-attention RMSNorm (fused residual+norm or standalone).
         * @param check_hybrid_q16 If true, skip fusion in HybridQ16 mode
         * @return Node name (prefix + "attn_norm")
         */
        std::string addPreAttentionNorm(
            ComputeGraph &graph,
            const std::string &prefix,
            ActivationBuffers &buffers,
            TensorBase *norm_gamma,
            int total_tokens,
            int layer_idx,
            DeviceId device,
            bool check_hybrid_q16 = true);

        /**
         * Add per-head QK RMSNorm stages if norms are present.
         * @return true if norms were added
         */
        bool addQKNorms(
            ComputeGraph &graph,
            const std::string &prefix,
            ActivationBuffers &buffers,
            const LayerWeights &layer,
            int local_n_heads,
            int local_n_kv_heads,
            int total_tokens,
            DeviceId device,
            const std::string &q_dependency,
            const std::string &k_dependency);

        /**
         * Add RoPE stage on Q and K.
         * @return Node name (prefix + "rope")
         */
        std::string addRoPE(
            ComputeGraph &graph,
            const std::string &prefix,
            ActivationBuffers &buffers,
            int local_n_heads,
            int local_n_kv_heads,
            int total_tokens,
            const int *position_ids,
            DeviceId device);

        /**
         * Add KV cache append stage.
         * @return KV append node name, or rope_dependency when no KV cache is present.
         */
        std::string addKVCacheAppend(
            ComputeGraph &graph,
            const std::string &prefix,
            ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            int batch_size,
            IKVCache *kv_cache,
            DeviceId device,
            const std::string &rope_dependency);

        /**
         * Add KV cache append, attention compute, and optional gather stages.
         * @return Terminal attention node name
         */
        std::string addKVCacheAndAttention(
            ComputeGraph &graph,
            const std::string &prefix,
            ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            int batch_size,
            int local_n_heads,
            int local_n_kv_heads,
            IKVCache *kv_cache,
            const int *position_ids,
            DeviceId device,
            bool has_qkv_proj,
            const std::string &rope_dependency);

        /**
         * Add Wo GEMM projection + optional TP allreduce.
         * @return Terminal node name
         */
        std::string addWoProjectionAndAllreduce(
            ComputeGraph &graph,
            const std::string &prefix,
            ActivationBuffers &buffers,
            TensorBase *wo_weight,
            const WeightBinding *wo_binding,
            int total_tokens,
            int layer_idx,
            DeviceId device,
            const std::string &dependency,
            const std::string &wo_node_suffix = "wo_proj",
            const std::string &allreduce_node_suffix = "wo_allreduce");

    public:
        static std::vector<int> buildPositionIds(int seq_len, int batch_size, int offset);

        void addFinalNormToGraph(
            ComputeGraph &graph,
            TensorBase *hidden,
            TensorBase *normalized_out,
            const std::string &prev_node,
            int seq_len,
            DeviceId device);

        const TPDomain *getDomainForLayer(int layer_idx, bool is_attention) const;
    };

} // namespace llaminar2
