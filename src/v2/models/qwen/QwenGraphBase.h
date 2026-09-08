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
#include "../../execution/compute_stages/stages/TPAllreduceStage.h"
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
#include "../../kernels/attention/AttentionExecutionPolicy.h"
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
            const std::vector<int> *sequence_lengths = nullptr,
            const void *position_ids_device = nullptr,
            const int32_t *sequence_lengths_device = nullptr) override = 0;

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
        void setDecodeReplicatedDenseWeightBindings(const ModelWeightBindings &bindings) override
        {
            decode_replicated_dense_weight_bindings_ = bindings;
        }
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

        void setCPUCurrentBatchLLEPPhysicalExecutor(
            ICPUCurrentBatchLLEPPhysicalExecutor *executor) override
        {
            cpu_current_batch_llep_executor_ = executor;
        }

        bool setComputeAllPositionLogits(bool enabled) override
        {
            config_.compute_all_position_logits = enabled;
            return true;
        }

        bool setGroupedMTPVerifier(bool enabled) override
        {
            config_.grouped_mtp_verifier = enabled;
            return true;
        }

        bool setMTPVerifierOutcomeGraphMode(
            MTPVerifierOutcomeGraphMode mode) override
        {
            config_.mtp_verifier_outcome_graph_mode = mode;
            return true;
        }

        bool setMTPVerifierOutcomeGraphBinding(
            const MTPVerifierOutcomeGraphBinding &binding) override
        {
            config_.mtp_verifier_outcome_graph_binding = binding;
            return true;
        }

        bool setLiveMTPRequestBatchCondition(bool enabled) override
        {
            config_.live_mtp_request_batch_condition = enabled;
            return true;
        }

        bool setComputeRowIndexedAllPositionLogits(bool enabled, int row_count) override
        {
            const int max_rows = resolveMTPMaxTargetQueryRows(config_.mtp);
            if (enabled && (row_count <= 0 || row_count > max_rows))
                return false;
            if (enabled &&
                !config_.row_indexed_logits_selected_rows.empty() &&
                static_cast<int>(config_.row_indexed_logits_selected_rows.size()) != row_count)
            {
                return false;
            }
            config_.compute_row_indexed_logits = enabled;
            config_.row_indexed_logits_row_count = enabled ? row_count : 0;
            if (!enabled)
                config_.row_indexed_logits_selected_rows.clear();
            return true;
        }

        /**
         * @brief Install explicit compact verifier source rows for the next graph.
         *
         * Empty keeps the legacy leading-row behavior. A non-empty row plan is
         * accepted before or after row-indexed mode is enabled; when enabled,
         * the size must match the fixed compact LM-head row count so graph
         * capture cannot race against a changing output shape.
         */
        bool setRowIndexedAllPositionLogitRows(const std::vector<int> &selected_rows) override
        {
            if (!selected_rows.empty() &&
                config_.compute_row_indexed_logits &&
                static_cast<int>(selected_rows.size()) != config_.row_indexed_logits_row_count)
            {
                return false;
            }
            config_.row_indexed_logits_selected_rows = selected_rows;
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
            const void *position_ids_device,
            DeviceId device);

        ComputeGraph buildLayerGraph(
            int layer_idx,
            TensorBase *input_hidden,
            IKVCache *kv_cache,
            const int *position_ids,
            const void *position_ids_device,
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
            DeviceId device,
            void *device_state_publication_stream,
            const int32_t *sequence_lengths_device = nullptr,
            const int32_t *absolute_position_ids_device = nullptr) override;

    protected:
        /**
         * @brief Borrow the device count for a contiguous single-request verifier.
         * @param device Participant owning the activations and count.
         * @param seq_len Physical rows per request in the retained graph.
         * @param batch_size Number of independently padded requests.
         * @param lengths Stable device request lengths, already in capture identity.
         * @return Counted geometry for one GPU request; otherwise the existing
         *         full physical extent. Independently padded multi-request
         *         matrices are not one contiguous live prefix.
         */
        std::optional<DeviceRowRange> projectionVerifierRows(
            DeviceId device, int seq_len, int batch_size,
            const int32_t *lengths) const;

        /**
         * @brief Resolve the semantic node prefix for one FFN subgraph.
         *
         * Ordinary transformer layers use their model-layer identity. Model
         * families that reuse the FFN builder inside an independently
         * executable graph, such as a recursive MTP sidecar, override this
         * hook so graph identity, snapshots, and diagnostics all describe the
         * actual execution owner rather than the source weight layer.
         *
         * @param layer_idx Source transformer layer supplying the weights.
         * @return Stable node prefix including its trailing underscore.
         */
        [[nodiscard]] virtual std::string ffnGraphStagePrefix(
            int layer_idx) const;

        // =====================================================================
        // Configuration (protected for subclass access)
        // =====================================================================
        GraphConfig config_;
        std::shared_ptr<ModelContext> model_ctx_;
        std::shared_ptr<IMPIContext> mpi_ctx_;
        TensorFactory *tensor_factory_ = nullptr;
        BufferArena *arena_ = nullptr;
        PreparedWeightStore *prepared_weight_store_ = nullptr;
        ICPUCurrentBatchLLEPPhysicalExecutor *
            cpu_current_batch_llep_executor_ = nullptr;
        ModelWeights weights_;
        ModelWeightBindings weight_bindings_;
        ModelWeightBindings decode_replicated_dense_weight_bindings_;
        ModelBuffers buffers_;
        StageSnapshotCallback snapshot_callback_;
        bool decode_replicated_dense_graph_active_ = false;
        bool mtp_replicated_dense_graph_active_ = false;
        bool replicated_attention_state_graph_active_ = false;
        bool decode_mirrored_embedding_graph_active_ = false;
        bool mtp_mirrored_lm_head_graph_active_ = false;
        bool mtp_kv_cache_only_graph_active_ = false;
        bool prefix_runtime_rehydration_graph_active_ = false;
        std::optional<ForwardExecutionPhase> forward_execution_phase_;

        // =====================================================================
        // Helpers
        // =====================================================================

        TensorContext buildTensorContext() const;
        /**
         * @brief Append graph-integrated shifted MTP KV prefill when declared.
         *
         * The main graph owns this transaction after its terminal logits are
         * complete. One device stage packs shifted hidden/token/position rows
         * and archives request terminals; a bucket-wide depth-zero MTP KV-only
         * subgraph then consumes that payload. The returned node is therefore
         * the terminal of the complete prefill transaction.
         *
         * @param graph Main forward graph receiving the transaction.
         * @param input Typed forward input and immutable shifted-MTP binding.
         * @param dependency_node Existing main-forward terminal node.
         * @param device Participant device for every appended stage.
         * @return New graph terminal, or @p dependency_node when no transaction
         *         was declared.
         * @throws std::runtime_error when a declared GPU transaction is partial.
         */
        std::string maybeAddShiftedMTPPrefillTransaction(
            ComputeGraph &graph,
            const ForwardInput &input,
            const std::string &dependency_node,
            DeviceId device);

        /**
         * @brief Append the typed MTP main-decode terminal-hidden publication.
         *
         * The publication copies the main model's one decoded hidden row per
         * request into the persistent MTP terminal archive inside the same
         * graph that produced the logits.  It is absent from prefill, whose
         * shifted-MTP transaction already owns terminal archival, and from
         * grouped verification, whose accepted-state publisher owns the
         * canonical row transition.
         *
         * @param graph Main-model graph receiving the publication stage.
         * @param input Complete typed forward input.
         * @param dependency_node Existing graph terminal after logits/outcome.
         * @param device Participant executing the graph.
         * @return New terminal node, or dependency_node when no binding exists.
         */
        std::string maybeAddMTPMainTerminalHiddenPublication(
            ComputeGraph &graph,
            const ForwardInput &input,
            const std::string &dependency_node,
            DeviceId device);
        bool needsTPAllreduce() const;
        bool denseTPAllreduceEnabledForCurrentGraph() const;
        bool hasActiveExpertMask(const std::vector<bool> &expert_mask) const;
        bool hasLayerWeightSource() const;
        /**
         * @brief Return whether replicated dense decode has layer bindings.
         *
         * Stage-owned global bindings are validated at the orchestrator setup
         * boundary and again by modelEmbeddingTable()/modelLMHead() when those
         * stages are actually built.  This predicate gates dense layer execution
         * only, which keeps PP stages that do not own embedding or LM head from
         * requiring globals they will never consume.
         */
        bool hasDecodeReplicatedDenseWeightSource() const;

        /**
         * @brief Render the replicated dense decode binding state for hard failures.
         *
         * @return Compact list of present/missing required replicated decode bindings.
         */
        std::string describeDecodeReplicatedDenseBindingState() const;
        bool useDecodeReplicatedDenseWeights() const;
        /** @return Whether the active MTP graph owns complete dense/shared bindings. */
        bool useReplicatedMTPSidecarDenseWeights() const;
        bool hasDecodeMirroredEmbeddingWeightSource() const;
        bool useDecodeMirroredEmbeddingWeights() const;
        bool hasMirroredMTPHeadWeightSource() const;
        /**
         * @brief Return whether MTP owns a full terminal head per TP participant.
         *
         * The result depends only on the typed MTP/head-layout policy. TP scope
         * must not silently reinterpret mirrored ownership as sharded ownership.
         */
        bool mirroredMTPHeadConfigured() const;
        /**
         * @brief Decide whether a terminal projection uses the mirrored MTP head.
         *
         * Mirrored MTP owns one replicated full-vocabulary terminal head per
         * participant at every TP scope. That head is required not only by the speculative
         * verifier and NextN sidecar, but also by the grouped main-model
         * condition forward that samples the first target token of a request
         * batch.  Keeping the decision in one projected-row policy prevents
         * that live condition transaction from silently rebuilding the
         * column-parallel head and reintroducing a tiny logits collective.
         *
         * @param total_tokens Number of activation rows entering the graph.
         *        Ordinary forward graphs project only their terminal row;
         *        all-position and compact row-indexed graphs project their
         *        corresponding declared output geometry.
         * @return true when the graph must bind the replicated full-vocabulary
         *         final norm and LM-head weights.
         */
        bool mirroredMTPHeadActiveForProjectedRows(int total_tokens) const;
        bool useMirroredMTPHeadWeights() const;
        bool useFullVocabEmbeddingForCurrentGraph() const;
        bool usesVocabParallelEmbeddingForCurrentGraph() const;
        int embeddingVocabOffsetForCurrentGraph(DeviceId device) const;
        bool useReplicatedAttentionStateWeights() const;

        /**
         * @brief Whether the active typed forward may use decode-only layout.
         *
         * MTP sidecar builders do not consume ForwardInput and therefore leave
         * the phase unset; their dedicated sidecar scope retains the existing
         * compact-row policy. Every ordinary/main-model forward installs an
         * explicit phase and may select replicated topology only for Decode.
         */
        bool forwardPhaseAllowsDecodeTopology() const noexcept;

        /**
         * @brief Immutable arithmetic decision for one TP reduction node.
         *
         * Policy and transport precision form one numerical contract.  They
         * must be resolved together: a canonical FP32 rank fold transported
         * through a lower-precision collective would no longer be the same
         * arithmetic operation.
         */
        struct TPAllreducePlan
        {
            TPAllreduceArithmeticPolicy arithmetic_policy =
                TPAllreduceArithmeticPolicy::NativeCollective;
            std::string transport_precision;
        };

        /**
         * @brief Resolve the sole arithmetic authority for a TP sum.
         *
         * The typed forward phase, rather than MTP enablement or row-count
         * heuristics, decides whether serial-row byte equivalence is required.
         * Decode on LocalTP GPU domains wider than two participants uses a
         * native allgather followed by an ascending-rank device fold.  Prefill
         * and mathematically unambiguous two-participant sums retain the native
         * throughput collective.
         *
         * @param buffer In-place FP32 tensor being reduced.
         * @param count Exact number of elements participating in the sum.
         * @param device Participant that owns @p buffer.
         * @param layer_idx Layer used to resolve native transport precision.
         * @param precision_override Optional caller-selected native precision.
         * @return Complete immutable policy/precision plan for the stage.
         * @throws std::logic_error when a decode graph requiring canonical
         *         arithmetic has malformed geometry or an unsupported LocalTP
         *         backend.
         */
        [[nodiscard]] TPAllreducePlan resolveTPAllreducePlan(
            const TensorBase *buffer,
            size_t count,
            DeviceId device,
            int layer_idx,
            const std::optional<std::string> &precision_override) const;

        /**
         * @brief Declarative source for the norm that feeds a final projection.
         *
         * The ordinary model LM head consumes `output_norm.weight`, while MTP
         * sidecars consume their own `mtp.norm.weight` or
         * `nextn.shared_head_norm.weight`.  Keeping this as a typed policy
         * prevents graph code from accidentally pairing a full-vocabulary LM
         * head with the wrong normalizer.
         */
        enum class FinalNormSource
        {
            ModelOutputNorm,
            MTPSidecarNorm,
        };

        /**
         * @brief Declarative LM-head weight/layout selected for a graph.
         */
        enum class FinalHeadPolicy
        {
            PrimaryColumnParallel,
            PrimaryFullVocabulary,
            DecodeReplicatedFullVocabulary,
            MirroredMTPFullVocabulary,
        };

        /**
         * @brief Request object for resolving a graph final-projection policy.
         */
        struct FinalProjectionPolicyRequest
        {
            FinalNormSource norm_source = FinalNormSource::ModelOutputNorm;
            TensorBase *mtp_norm = nullptr;
            TensorBase *full_vocab_output = nullptr;
            TensorBase *column_parallel_output = nullptr;
            int total_tokens = 1;
            bool force_full_vocabulary_head = false;
            bool compute_all_positions = false;
        };

        /**
         * @brief Resolved tensors and metadata for a final projection stage.
         *
         * Model-specific graph files should ask for one of these policies and
         * wire the returned tensors into the base graph-building blocks instead
         * of duplicating sharded/replicated LM-head decisions inline.
         */
        struct FinalProjectionPolicy
        {
            FinalNormSource norm_source = FinalNormSource::ModelOutputNorm;
            FinalHeadPolicy head_policy = FinalHeadPolicy::PrimaryFullVocabulary;
            TensorBase *norm_gamma = nullptr;
            TensorBase *lm_head_weight = nullptr;
            const WeightBinding *lm_head_binding = nullptr;
            TensorBase *lm_head_output = nullptr;
            int lm_head_vocab_size = 0;
            int serial_equivalent_partition_width = 0;
            bool column_parallel = false;
            bool needs_allgather = false;
        };

        FinalProjectionPolicy resolveFinalProjectionPolicy(
            const FinalProjectionPolicyRequest &request) const;
        /**
         * @brief Decide whether the current LM-head stage writes a local vocab shard.
         *
         * Ordinary TP prefill/decode uses the primary column-parallel path only
         * while dense TP allreduce is active for the graph.  Phase-split
         * decode-replicated verifier rows deliberately use the replicated
         * full-vocab LM head so their logits are produced by the same binding
         * and prepared GEMM descriptor as rowwise serial decode.
         *
         * @param logits_local Candidate output tensor for local-vocab logits.
         * @return true when LMHeadStage must bind the primary sharded LM head.
         */
        bool useColumnParallelLMHeadForGraph(TensorBase *logits_local) const;

        /**
         * @brief Resolve the canonical serial arithmetic width for a mirrored head.
         *
         * The largest typed vocabulary assignment defines the regular serial
         * partition. Smaller assignments are legal remainder shards. Mirrored
         * participants all use this same width for generated-policy selection,
         * while still launching the complete physical vocabulary projection.
         *
         * @param column_parallel True when the graph owns only its local shard.
         * @return Zero when no equivalence scope is required, otherwise the
         *         positive canonical serial vocabulary width.
         * @throws std::logic_error When TP accounting is incomplete or disagrees
         *         with the graph's local vocabulary width.
         */
        int serialEquivalentLMHeadPartitionWidth(bool column_parallel) const;

        /**
         * @brief Decide whether a column-parallel LM head needs an MPI gather.
         *
         * A single MPI rank is not a distributed vocabulary domain. LocalTP
         * runners publish their shard-local tensor to RankOrchestrator, which
         * coordinates sampling across local devices. Emitting a one-rank MPI
         * allgather would instead route GPU logits through the legacy host path
         * and falsely advertise the dormant full-vocabulary tensor as produced.
         *
         * @param column_parallel Whether LMHeadStage writes the local-vocab tensor.
         * @return true only for a real multi-rank MPI vocabulary gather.
         */
        bool needsDistributedLMHeadAllGather(bool column_parallel) const;

        /**
         * @brief Resolve the tensor actually produced at the graph boundary.
         *
         * This function keeps terminal-stage wiring and ForwardOutput metadata
         * identical. A non-gathered column-parallel graph publishes LOGITS_LOCAL;
         * a replicated head or completed distributed allgather publishes LOGITS.
         *
         * @param column_parallel Whether LMHeadStage writes LOGITS_LOCAL.
         * @return Exact tensor downstream publication and consumers must use.
         */
        TensorBase *graphLMHeadOutput(bool column_parallel) const;

        bool denseDecodeReplicatedActiveForTokens(int total_tokens) const;
        /**
         * @brief Whether the graph under construction owns prefix rehydration.
         *
         * Derived MoE builders use this immutable build scope to attach a
         * one-shot transfer transaction to every relevant layer. It must never
         * be inferred from mutable host runtime-table contents.
         */
        bool prefixRuntimeRehydrationGraphActive() const noexcept
        {
            return prefix_runtime_rehydration_graph_active_;
        }
        bool denseDecodeMirroredEmbeddingActiveForTokens(int total_tokens) const;
        bool replicatedAttentionStateActiveForTokens(int total_tokens) const;
        bool attentionTPAllreduceEnabledForCurrentGraph() const;
        bool needsPhaseSplitPrefillKVCacheHandoff(
            int total_tokens,
            IKVCache *kv_cache,
            DeviceId device) const;

        class DecodeReplicatedDenseScope
        {
        public:
            /**
             * @brief Select the complete participant-local dense binding view.
             * @param owner Graph builder whose immutable build state is scoped.
             * @param total_tokens Physical row count of the graph being built.
             * @param force_replicated_dense True only for a typed graph family,
             *        such as a replicated MTP predictor, that owns complete
             *        dense/shared bindings independently of ordinary decode.
             */
            DecodeReplicatedDenseScope(
                QwenGraphBase &owner,
                int total_tokens,
                bool force_replicated_dense = false);
            ~DecodeReplicatedDenseScope();

            DecodeReplicatedDenseScope(const DecodeReplicatedDenseScope &) = delete;
            DecodeReplicatedDenseScope &operator=(const DecodeReplicatedDenseScope &) = delete;

        private:
            QwenGraphBase &owner_;
            bool previous_;
            bool previous_mtp_dense_;
            bool previous_attention_;
            bool previous_embedding_;
            bool previous_mtp_head_;
        };

        /**
         * @brief Publish the caller's typed forward phase to nested builders.
         *
         * Qwen graph construction delegates through full-graph, layer,
         * attention, and FFN builders. Those builders historically inferred
         * decode-equivalent policy from M independently, allowing a short
         * prompt to select replicated decode weights. This scope gives every
         * nested policy query one immutable phase for the complete build.
         */
        class ForwardExecutionPhaseScope
        {
        public:
            ForwardExecutionPhaseScope(
                QwenGraphBase &owner,
                ForwardExecutionPhase phase);
            ~ForwardExecutionPhaseScope();

            ForwardExecutionPhaseScope(
                const ForwardExecutionPhaseScope &) = delete;
            ForwardExecutionPhaseScope &operator=(
                const ForwardExecutionPhaseScope &) = delete;

        private:
            QwenGraphBase &owner_;
            std::optional<ForwardExecutionPhase> previous_;
        };

        /**
         * @brief Temporarily select replicated full-vocab LM-head bindings for MTP.
         *
         * Dedicated MTP sidecar graphs do not necessarily set
         * compute_all_position_logits, so they cannot rely on
         * DecodeReplicatedDenseScope's verifier-row detection.  This scope lets
         * sidecar builders opt into the mirrored terminal head only for the
         * rows whose logits are consumed by the MTP sampler/verifier.
         */
        class MirroredMTPHeadScope
        {
        public:
            MirroredMTPHeadScope(QwenGraphBase &owner, bool active);
            ~MirroredMTPHeadScope();

            MirroredMTPHeadScope(const MirroredMTPHeadScope &) = delete;
            MirroredMTPHeadScope &operator=(const MirroredMTPHeadScope &) = delete;

        private:
            QwenGraphBase &owner_;
            bool previous_;
        };

        /**
         * @brief Mark construction of an MTP shifted-KV-only sidecar graph.
         *
         * These sidecar graphs run inside the same decode-replicated binding
         * scope as verifier rows so prepared weight references stay aligned,
         * but their Q/K/V projections can still be local TP shards.  The
         * phase-split full-prefill KV handoff must therefore remain eligible
         * even while decode-replicated graph state is active.
         */
        class MTPKVCacheOnlyScope
        {
        public:
            MTPKVCacheOnlyScope(QwenGraphBase &owner, bool active);
            ~MTPKVCacheOnlyScope();

            MTPKVCacheOnlyScope(const MTPKVCacheOnlyScope &) = delete;
            MTPKVCacheOnlyScope &operator=(const MTPKVCacheOnlyScope &) = delete;

        private:
            QwenGraphBase &owner_;
            bool previous_;
        };

        LayerWeightBindings layerWeightBindingsForGraph(int layer_idx) const;
        LayerWeights layerWeightsForGraph(int layer_idx) const;
        TensorBase *modelEmbeddingTable() const;
        TensorBase *modelFinalNorm() const;
        TensorBase *modelLMHead() const;
        /**
         * @brief Resolve the LM-head tensor for a specific output layout.
         *
         * Column-parallel logits use the primary sharded LM-head binding.
         * Full-vocab logits use modelLMHead(), which selects the replicated
         * dense decode view when DecodeReplicatedDenseScope is active.
         */
        TensorBase *modelLMHeadForGraph(bool column_parallel) const;
        const WeightBinding *modelEmbeddingBinding() const;
        const WeightBinding *modelFinalNormBinding() const;
        const WeightBinding *modelLMHeadBinding() const;
        /**
         * @brief Resolve the prepared-weight binding paired with modelLMHeadForGraph().
         *
         * Keeping tensor and PreparedWeightRef selection in one place prevents a
         * verifier graph from pairing a primary sharded tensor with a replicated
         * prepared GEMM descriptor, or vice versa.
         */
        const WeightBinding *modelLMHeadBindingForGraph(bool column_parallel) const;
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
         * @brief Resolve the device owner of compact LM-head source-row indices.
         *
         * Request-batched prefill has no explicit verifier row plan, so one
         * terminal row per request is derived from resident request lengths.
         * Grouped MTP verification installs explicit query-row indices while the
         * same request-length arena remains live; those explicit rows must take
         * precedence or the graph will reinterpret verifier rows as requests.
         *
         * @param has_request_sequence_lengths Whether the active graph has a
         *        device-resident request-length row.
         * @return RequestTerminalLengths for compact prefill, otherwise
         *         WorkspaceBoundDeviceIndices for explicit verifier metadata.
         */
        HiddenStateRowsSelectStage::DeviceRowIndexSource
        resolveLMHeadDeviceRowIndexSource(
            bool has_request_sequence_lengths) const;

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
         * @param request_sequence_lengths_device Optional device-owned request
         *        lengths used to derive one terminal row per padded request.
         * @param request_row_stride Padded source-row stride for each request.
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
            const int32_t *request_sequence_lengths_device = nullptr,
            int request_row_stride = 0,
            BufferId input_buffer_id = BufferId::NORMALIZED) const;

        /**
         * @brief Append the typed graph-owned MTP verifier transaction.
         *
         * @param graph Forward graph whose LM-head output is complete.
         * @param dependency_node Exact logits-producing terminal node.
         * @param logits Full-vocabulary verifier logits tensor.
         * @param verifier_row_count Number of compact verifier rows.
         * @param device Participant device.
         * @return New terminal node, or @p dependency_node when the policy is
         *         Disabled.
         * @throws std::runtime_error when an enabled policy is malformed or
         *         cannot be represented by the current topology.
         */
        std::string addMTPVerifierOutcomeToGraph(
            ComputeGraph &graph,
            const std::string &dependency_node,
            TensorBase *logits,
            int verifier_row_count,
            DeviceId device) const;

        /**
         * @brief Seal the terminal unit of one complete inference transaction.
         *
         * Main forward graphs and retained MTP sidecars use the same terminal
         * lifecycle. Ordinary and device-owned graphs only record their logical
         * terminal. A heterogeneous ticket graph additionally names that final
         * captured unit with a cross-participant identity, allowing authority
         * and follower plans to prove the same captured/manual/captured shape.
         * Keeping this transition here prevents individual graph builders from
         * forgetting the terminal contract when they append a new suffix.
         *
         * @param graph Complete participant-local graph being sealed.
         * @param terminal_node Existing node that completes the transaction.
         * @throws std::out_of_range when @p terminal_node is absent.
         */
        void sealInferenceTransactionGraph(
            ComputeGraph &graph,
            const std::string &terminal_node) const;

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
            std::optional<BufferId> tensor_buffer_id = std::nullopt,
            std::vector<TPAllreduceSidebandWorkspaceBinding> sideband_workspace_bindings = {},
            std::optional<std::string> precision_override = std::nullopt) const;

        // =====================================================================
        // Shared Attention Building Blocks
        // =====================================================================

        /**
         * @brief Resolve one complete attention policy for a graph participant.
         *
         * Every backend receives geometry-selected physical ownership. CPU,
         * CUDA, and ROCm each resolve that declaration through their own typed
         * launch policy; model graphs never pin a backend to a legacy physical
         * schedule. Explicit query/context requests remain isolated kernel
         * tournament controls rather than production graph policy.
         *
         * CPU and native floating GPU caches publish post-RoPE K so attention
         * can consume persistent ring bytes without a conversion shadow.
         * CUDA and ROCm honor the model's @c rope_on_read preference for
         * quantized caches whose established arithmetic transforms after
         * dequantization. A cacheless graph always rotates projected K before
         * attention.
         *
         * @param device Participant that will execute the attention node.
         * @param cache_backed True when the graph publishes and consumes a KV cache.
         * @return Immutable policy shared by every attention-related graph node.
         * @throws std::invalid_argument for an unsupported or invalid device.
         */
        [[nodiscard]] attention::AttentionExecutionPolicy
        resolveAttentionExecutionPolicy(
            DeviceId device,
            bool cache_backed) const;

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
         * @brief Make a projected K tensor ready for exact KV-cache publication.
         *
         * The helper owns the graph-wide K publication policy: it applies the
         * model's optional per-head K norm and applies K-only RoPE only when the
         * cache stores post-RoPE keys. With RoPE-on-read, projected normalized K
         * remains pre-RoPE and no rotary node is emitted.
         *
         * @param graph Graph receiving the cache-publication transforms.
         * @param prefix Stable node-name prefix.
         * @param buffers Activation buffers containing K and its binding map.
         * @param layer Layer weights containing the optional K norm.
         * @param local_n_kv_heads Number of K heads represented by @p buffers.
         * @param total_tokens Number of physical K rows.
         * @param position_ids Optional host position row for CPU execution.
         * @param position_ids_device Optional device-owned GPU position row.
         * @param device Stage device.
         * @param execution_policy Resolved publication/consumption contract.
         * @param projection_dependency Node that produces K.
         * @return The final node whose completion makes K cache-ready.
         */
        std::string addKeyCachePublicationTransforms(
            ComputeGraph &graph,
            const std::string &prefix,
            ActivationBuffers &buffers,
            const LayerWeights &layer,
            int local_n_kv_heads,
            int total_tokens,
            const int *position_ids,
            const void *position_ids_device,
            DeviceId device,
            const attention::AttentionExecutionPolicy &execution_policy,
            const std::string &projection_dependency);

        /**
         * @brief Add the declarative RoPE stage for Q and, when configured, K.
         *
         * The resolved execution policy is the sole owner of whether K is
         * rotated here or while reading a GPU KV cache. Callers must pass the
         * same policy to append and attention construction, because divergent
         * producer/consumer policies give full and chunked prefill different
         * numerical paths.
         *
         * @return Node name (prefix + "rope").
         */
        std::string addRoPE(
            ComputeGraph &graph,
            const std::string &prefix,
            ActivationBuffers &buffers,
            int local_n_heads,
            int local_n_kv_heads,
            int total_tokens,
            const int *position_ids,
            const void *position_ids_device,
            DeviceId device,
            const attention::AttentionExecutionPolicy &execution_policy);

        /**
         * @brief Add a KV cache append stage.
         *
         * Normal transformer graphs pass a global model layer id and the helper
         * translates it to the PP-stage-local KV cache layer. Sidecar graphs,
         * such as MTP, own a separate cache whose layer ids are already local.
         *
         * @param layer_idx Global model layer id by default, or cache-local id
         *                  when @p layer_idx_is_cache_local is true.
         * @param layer_idx_is_cache_local Treat @p layer_idx as an already-local
         *                                 KV cache layer id.
         * @param request_sequence_lengths_device Device-owned request lengths
         *        consumed by captured GPU append publication.
         * @param execution_policy Resolved publication/consumption contract.
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
            const int32_t *request_sequence_lengths_device,
            DeviceId device,
            const attention::AttentionExecutionPolicy &execution_policy,
            const std::string &rope_dependency,
            const std::vector<std::string> &cache_source_dependencies = {},
            bool layer_idx_is_cache_local = false,
            int first_seq_idx = 0);

        /**
         * @brief Add KV cache append, attention compute, and optional gather stages.
         *
         * @param layer_idx Global model layer id by default, or cache-local id
         *                  when @p layer_idx_is_cache_local is true.
         * @param layer_idx_is_cache_local Treat @p layer_idx as an already-local
         *                                 KV cache layer id.
         * @param request_sequence_lengths_device Device-owned request lengths
         *        consumed by captured GPU append publication.
         * @param execution_policy Resolved publication/consumption contract.
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
            const void *position_ids_device,
            const int32_t *request_sequence_lengths_device,
            DeviceId device,
            bool has_qkv_proj,
            const attention::AttentionExecutionPolicy &execution_policy,
            const std::string &rope_dependency,
            const std::vector<std::string> &cache_source_dependencies = {},
            bool layer_idx_is_cache_local = false);

        /**
         * @brief Add the participant-local attention output projection.
         *
         * This transition publishes only the local row-parallel partial.  A
         * caller that requires tensor-parallel reconstruction must pass the
         * returned node through addWoAllreduce().  Keeping the two transitions
         * explicit allows architecture-specific diagnostics to observe the
         * local partial without racing the collective that consumes it.
         *
         * @param graph Participant-local graph being assembled.
         * @param prefix Stable layer node prefix.
         * @param buffers Activation tensors and their arena identities.
         * @param wo_weight Local or replicated output-projection weight.
         * @param wo_binding Prepared-weight binding for @p wo_weight.
         * @param total_tokens Physical activation rows in this graph.
         * @param device Participant that owns the projection.
         * @param dependency Node that publishes @c buffers.attn_output.
         * @param wo_node_suffix Stable suffix for the projection node.
         * @param verifier_rows Immutable physical/live verifier row geometry.
         * @return Projection node, or @p dependency when no weight exists.
         */
        std::string addWoProjection(
            ComputeGraph &graph,
            const std::string &prefix,
            ActivationBuffers &buffers,
            TensorBase *wo_weight,
            const WeightBinding *wo_binding,
            int total_tokens,
            DeviceId device,
            const std::string &dependency,
            const std::string &wo_node_suffix = "wo_proj",
            std::optional<DeviceRowRange> verifier_rows = std::nullopt);

        /**
         * @brief Reconstruct a row-parallel attention projection with TP sum.
         *
         * The collective consumes the already-published local partial from
         * @c buffers.attn_proj.  Replicated weights and non-TP graphs retain
         * @p dependency unchanged, making the returned node the exact published
         * attention-output boundary in every topology.
         *
         * @param graph Participant-local graph being assembled.
         * @param prefix Stable layer node prefix.
         * @param buffers Activation tensors and their arena identities.
         * @param wo_weight Weight whose sharding declares whether reduction is required.
         * @param total_tokens Physical activation rows in this graph.
         * @param layer_idx Global layer index used by precision policy.
         * @param device Participant that owns the collective.
         * @param dependency Local projection or an ordered observation of it.
         * @param allreduce_node_suffix Stable suffix for the collective node.
         * @return Collective node when required, otherwise @p dependency.
         */
        std::string addWoAllreduce(
            ComputeGraph &graph,
            const std::string &prefix,
            ActivationBuffers &buffers,
            TensorBase *wo_weight,
            int total_tokens,
            int layer_idx,
            DeviceId device,
            const std::string &dependency,
            const std::string &allreduce_node_suffix = "wo_allreduce");

        /**
         * @brief Optionally retain exact embedding rows inside the forward DAG.
         *
         * The base implementation is a topology-preserving no-op. A derived
         * architecture may use this hook for opt-in device diagnostics, but it
         * must chain every checkpoint into the returned dependency so graph
         * completion proves that the observation belongs to this invocation.
         *
         * @param graph Forward graph under construction.
         * @param source Complete embedding output matrix.
         * @param source_buffer_id Arena slot that owns @p source.
         * @param dependency Embedding or embedding-collective producer node.
         * @param total_tokens Exact physical row count in @p source.
         * @param device Device that owns source and checkpoint destinations.
         * @return @p dependency or the final diagnostic checkpoint node.
         */
        virtual std::string maybeAddEmbeddingDiagnosticCheckpoints(
            ComputeGraph &graph,
            TensorBase *source,
            BufferId source_buffer_id,
            const std::string &dependency,
            int total_tokens,
            DeviceId device)
        {
            (void)graph;
            (void)source;
            (void)source_buffer_id;
            (void)total_tokens;
            (void)device;
            return dependency;
        }

        /**
         * @brief Optionally insert a graph-owned checkpoint around final norm.
         *
         * The base implementation is a no-op. Architectures with an opt-in
         * device diagnostic may override it to retain one terminal row before
         * and after final norm. Returning the new terminal dependency keeps the
         * checkpoint in the forward DAG's required ordering chain.
         *
         * @param graph Forward graph under construction.
         * @param boundary Stable semantic boundary name.
         * @param source Tensor whose terminal row should be retained.
         * @param source_buffer_id Arena slot that owns @p source.
         * @param dependency Producer that must complete before the checkpoint.
         * @param total_tokens Number of logical rows in @p source.
         * @param device Device that owns @p source.
         * @param sequence_lengths_device Optional device-owned real row counts.
         *        Padded GPU diagnostics consume this pointer inside their
         *        captured row-copy kernel instead of adopting a host row.
         * @return @p dependency for the no-op base implementation, or the
         *         derived checkpoint node name.
         */
        virtual std::string maybeAddFinalNormDiagnosticCheckpoint(
            ComputeGraph &graph,
            const std::string &boundary,
            TensorBase *source,
            BufferId source_buffer_id,
            const std::string &dependency,
            int total_tokens,
            DeviceId device,
            const int32_t *sequence_lengths_device)
        {
            (void)graph;
            (void)boundary;
            (void)source;
            (void)source_buffer_id;
            (void)total_tokens;
            (void)device;
            (void)sequence_lengths_device;
            return dependency;
        }

    public:
        static std::vector<int> buildPositionIds(int seq_len, int batch_size, int offset);

        /**
         * @brief Add final RMSNorm and return its ordered terminal node.
         *
         * The returned node may be a derived diagnostic checkpoint chained
         * after `final_norm`; callers must use it as the LM-head dependency so
         * graph completion also owns checkpoint completion.
         *
         * @param sequence_lengths_device Optional stable device owner for real
         *        request row counts. It is forwarded only to diagnostic
         *        checkpoints; RMSNorm itself remains shape-static.
         */
        std::string addFinalNormToGraph(
            ComputeGraph &graph,
            TensorBase *hidden,
            TensorBase *normalized_out,
            const std::string &prev_node,
            int seq_len,
            DeviceId device,
            BufferId input_buffer_id = BufferId::HIDDEN_STATE,
            const int32_t *sequence_lengths_device = nullptr);

        const TPDomain *getDomainForLayer(int layer_idx, bool is_attention) const;
    };

} // namespace llaminar2
