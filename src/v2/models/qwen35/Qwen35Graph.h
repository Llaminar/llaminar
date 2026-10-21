/**
 * @file Qwen35Graph.h
 * @brief Qwen 3.5 compute graph builder for hybrid GDN + full attention architecture
 *
 * Inherits shared Qwen infrastructure from QwenGraphBase:
 *   IGraphBuilder → QwenGraphBase → Qwen35Graph
 *
 * Supports the Qwen 3.5 "Dense" architecture with heterogeneous transformer layers:
 *   - GDN (Gated Delta Net) layers: linear attention with delta rule recurrence
 *   - Full Attention (FA) layers: standard multi-head attention with RoPE
 *
 * The layer type pattern is determined by full_attention_interval from GGUF metadata.
 * For example, with interval=4: layers 3,7,11,...,31 are FA; all others are GDN.
 *
 * Both layer types share:
 *   - Attention output gate: sigmoid(gate) * attn_output
 *   - FFN: SwiGLU (gate_up_proj → swiglu → down_proj)
 *   - Residual connections
 *
 * GDN layers use:
 *   - Fused QKV + Z + A + B projections (4 separate GEMMs)
 *   - Short causal conv1d + SiLU on QKV
 *   - Delta rule recurrence (chunk-parallel prefill, single-step decode)
 *   - Gated RMSNorm: RMSNorm(output) * SiLU(Z)
 *   - Output projection (Wo GEMM)
 *
 * FA layers use:
 *   - Standard Q/K/V projections (separate weights)
 *   - QK normalization (pre-RoPE RMSNorm)
 *   - Partial RoPE (rope.dimension_count / head_dim < 1.0)
 *   - KV cache + standard attention
 *   - Output projection (Wo GEMM)
 */

#pragma once

#include "../qwen/QwenGraphBase.h"
#include <memory>

namespace llaminar2
{

    /**
     * @brief Qwen 3.5 graph builder with GDN + FA hybrid attention
     *
     * Inherits shared transformer infrastructure from QwenGraphBase.
     * Implements hybrid attention dispatch per layer type.
     */
    class Qwen35Graph : public QwenGraphBase
    {
    public:
        /// Construct with full model context
        Qwen35Graph(std::shared_ptr<ModelContext> model_ctx,
                    std::shared_ptr<IMPIContext> mpi_ctx,
                    const GraphConfig &config);

        /// Construct for layer-level operations only
        Qwen35Graph(const GraphConfig &config,
                    std::shared_ptr<IMPIContext> mpi_ctx = nullptr);

        ~Qwen35Graph() = default;

        // =====================================================================
        // IGraphBuilder overrides
        // =====================================================================

        std::string architectureName() const override { return "qwen35"; }

        GraphSchema getSchema() const override;

        /// Reset GDN conv/recurrence state between sessions (no-op: state is in hybrid cache)
        void resetState(void *execution_stream = nullptr) override
        {
            (void)execution_stream;
        }

        /// Wire GDN-specific arena buffers after base wiring
        void setArena(BufferArena *arena) override;

        /// Extends base resolver config with GDN-specific formulas
        /// and buffer name→id mappings, keeping core infrastructure agnostic.
        GraphResolverConfig getResolverConfig(int seq_len) const override;

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
            const int32_t *sequence_lengths_device = nullptr) override;

        /**
         * @brief Build one MTP graph from an explicitly materialized weight view.
         *
         * This legacy tensor overload assumes @p weights already represent the
         * graph's declared sidecar layout. Production frozen-weight callers use
         * the typed binding overload below so replicated predictors cannot be
         * handed a primary-model TP shard.
         */
        ComputeGraph buildMTPGraph(
            int depth_idx,
            const MTPDepthWeights &weights,
            const MTPForwardInput &input,
            MTPForwardOutput &output);

        /**
         * @brief Build one MTP graph from the canonical typed sidecar binding.
         *
         * Tensor-parallel predictors consume @p bindings directly. A declared
         * replicated-per-participant predictor instead resolves the matching
         * depth from the auxiliary full-width binding set installed on the
         * builder. This single selection boundary serves retained live
         * sidecars and graph-integrated shifted-prefill cache publication.
         *
         * @param depth_idx Logical MTP predictor depth being built.
         * @param bindings Caller-visible primary weight binding; used directly
         *        only when the sidecar policy is tensor parallel.
         * @param input Typed sidecar input and shifted-state ownership.
         * @param output Persistent graph output bindings.
         * @return Complete participant-local MTP transaction graph.
         * @throws std::runtime_error when a replicated policy has no unique
         *         full-width binding for @p depth_idx.
         */
        ComputeGraph buildMTPGraph(
            int depth_idx,
            const MTPDepthWeightBindings &bindings,
            const MTPForwardInput &input,
            MTPForwardOutput &output) override;

        /**
         * @brief Declare dense Qwen3.5/3.6 MTP sidecar state ownership.
         *
         * The sidecar writes only MTP-prefixed activation buffers and its
         * shifted KV cache. Its first shifted row is the same row that dense
         * accepted-state publication would append, so the row may be retained.
         */
        [[nodiscard]] MTPSidecarStateContract
        mtpSidecarStateContract() const noexcept override
        {
            return {
                .main_state = MTPSidecarMainStatePolicy::Preserved,
                .shifted_row =
                    MTPShiftedRowPublicationPolicy::ReuseSidecarRow,
            };
        }

    protected:
        /**
         * @brief Return whether a recursive MTP sidecar is being assembled.
         *
         * Derived dense/MoE FFN builders use this typed construction context to
         * select sidecar-owned buffers, registries, and semantic node names.
         * It is never runtime inference state and is restored by RAII before
         * graph construction returns.
         */
        [[nodiscard]] bool mtpGraphContextActive() const noexcept
        {
            return mtp_graph_context_active_;
        }

        /**
         * @brief Return the depth owned by the active MTP graph context.
         * @return Non-negative MTP depth while @ref mtpGraphContextActive is
         *         true, otherwise -1.
         */
        [[nodiscard]] int mtpGraphDepthIndex() const noexcept
        {
            return mtp_graph_depth_idx_;
        }

        /**
         * @brief Name dense FFN nodes after their executable sidecar owner.
         *
         * The trailing Qwen3.6 NextN block borrows weights from a model layer,
         * but it is not that main-graph layer. Keeping the depth namespace
         * here aligns dense and MoE graphs and makes every production
         * checkpoint addressable as `MTP<n>_*`.
         */
        [[nodiscard]] std::string ffnGraphStagePrefix(
            int layer_idx) const override;

        /**
         * @brief Optionally insert one terminal-row checkpoint into a GDN graph.
         *
         * The dense graph has no diagnostic storage and returns @p dependency
         * unchanged. Derived graph families can override this hook to make
         * projection, short-convolution, recurrence, and output-projection
         * boundaries observable without teaching the GDN graph builder about
         * a particular logging or tensor-retention policy.
         *
         * Implementations that add a node must return that node's name. The
         * caller makes the next mutating GDN stage depend on it, so an in-place
         * short-convolution cannot overwrite the projection row before the
         * checkpoint has captured it.
         *
         * @param graph GDN subgraph being assembled.
         * @param boundary Stable backend-neutral boundary name.
         * @param source Tensor whose final real row is retained.
         * @param source_buffer_id Arena slot that owns @p source.
         * @param dependency Producer that must complete before row selection.
         * @param layer_idx Global transformer layer index.
         * @param total_tokens Flattened captured graph row count.
         * @param feature_dim Number of FP32 values in one source row.
         * @param device Device that owns both source and checkpoint.
         * @param sequence_lengths_device Resident real-length owner for padded
         *        prefill, or null for exact-row graph regimes.
         * @return @p dependency when disabled, otherwise the inserted node.
         */
        virtual std::string maybeAddGDNDiagnosticCheckpoint(
            ComputeGraph &graph,
            const std::string &boundary,
            const ITensor *source,
            BufferId source_buffer_id,
            const std::string &dependency,
            int layer_idx,
            int total_tokens,
            int feature_dim,
            DeviceId device,
            const int32_t *sequence_lengths_device);

        /**
         * @brief Resolve the independently executable GDN graph-role namespace.
         *
         * Mutable long-context scratch is shared by all serialized GDN layers
         * in one graph role, but it must never alias another role that may
         * replay on a different stream. Keeping this policy in the declarative
         * graph builder makes the ownership distinction explicit and prevents
         * individual stages from inventing ad hoc workspace identities.
         *
         * @return Empty for main inference, otherwise a stable role name.
         */
        std::string gdnWorkspaceRoleNamespace() const;

    private:
    protected:
        /**
         * @brief Select the exact frozen sidecar depth using the existing explicit replication policy.
         * @param depth_idx Learned predictor depth identity.
         * @param bindings Primary prepared source; retained when replication is not selected.
         * @return Primary or existing admitted full binding, never a tensor-shaped substitute.
         * @throws std::runtime_error For an absent or duplicate selected replica.
         */
        const MTPDepthWeightBindings &mtpBindingsForPolicy(int depth_idx, const MTPDepthWeightBindings &bindings) const;

    private:
        /**
         * @brief RAII owner for one recursive MTP graph-construction context.
         *
         * Nested calls are supported because binding-based builders delegate
         * to legacy tensor views. Restoring the previous context prevents an
         * unsuccessful sidecar build from leaking its depth into the next
         * ordinary forward graph.
         * The sidecar owns different query lengths, so this same scope suspends
         * the enclosing main-prefill collective prefix rather than letting a
         * nested graph accidentally consume another graph family's authority.
         */
        class ScopedMTPGraphContext
        {
        public:
            /** @brief Enter one sidecar role and suspend the enclosing main-prefill row binding. */
            ScopedMTPGraphContext(Qwen35Graph &graph, int depth_idx) noexcept;
            /** @brief Restore all enclosing graph-role operands, including on exceptions. */
            ~ScopedMTPGraphContext();

            ScopedMTPGraphContext(const ScopedMTPGraphContext &) = delete;
            ScopedMTPGraphContext &operator=(
                const ScopedMTPGraphContext &) = delete;

        private:
            Qwen35Graph &graph_;
            bool previous_active_ = false;
            int previous_depth_idx_ = -1;
            std::optional<DeviceRowRange> previous_prefill_rows_;
        };

        // =====================================================================
        // FA (Full Attention) Sub-Graph Building
        // =====================================================================

        /**
         * @brief Build FA attention sub-graph with Q gate split + sigmoid output gate
         *
         * Qwen3.5 FA layers differ from standard Qwen2 attention:
         *   1. Q projection outputs 2× (query + sigmoid gate interleaved per head)
         *   2. Partial RoPE (only first 64/256 dims rotated)
         *   3. sigmoid(gate) applied to attention output before Wo
         *
         * Stages: norm → fused_qkv (Q→fa_q_raw, K, V) → q_gate_split → qk_norm
         *         → rope → kv_append → attention → output_gate → wo_proj
         */
        ComputeGraph buildFAAttentionGraph(
            const LayerWeights &layer,
            ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            int batch_size,
            IKVCache *kv_cache,
            const int *position_ids,
            const void *position_ids_device,
            DeviceId device,
            const std::vector<int> *sequence_lengths,
            const int32_t *sequence_lengths_device,
            const std::string &stage_prefix_override = {},
            bool layer_idx_is_cache_local = false);

    protected:
        /**
         * @brief Reuse the canonical FA body with an explicit immutable source/cache/reader.
         * @param identity Sealed real source, native cache index and dense/selected policy.
         * @param predictor Explicit QFN predictor residual construction, or nullptr for unchanged main/inline wiring.
         * @param stage_prefix_override Existing snapshot prefix, empty selects the sealed identity prefix.
         * @return The same participant-local FA chain with caller-admitted residual destinations.
         * @throws std::invalid_argument For contradictory predictor source, rows, participant or roles.
         */
        ComputeGraph buildFAAttentionGraphForLayer(
            const LayerWeights &layer, ActivationBuffers &buffers,
            const FullAttentionLayerIdentity &identity, int seq_len, int batch_size,
            IKVCache *kv_cache, const int *position_ids, const void *position_ids_device,
            DeviceId device, const std::vector<int> *sequence_lengths,
            const int32_t *sequence_lengths_device, const std::string &stage_prefix_override,
            const Qwen4ExpPredictorResidualBinding *predictor);

        /**
         * @brief Reuse the KV-only body with explicit source48/cache0 and counted HC input.
         * @param identity Checked dense predictor source/cache policy.
         * @param predictor Explicit frozen predictor HC weights and original token rows.
         * @param first_seq_idx Exact existing request cache slot; initial Dense0 supports slot0 only.
         * @return Native K/V append producer, without Q, attention, FFN, head or auxiliary state.
         * @throws std::invalid_argument For contradictory source/count/cache geometry.
         */
        ComputeGraph buildMTPFAKVCacheAppendGraphForLayer(
            const LayerWeights &layer, ActivationBuffers &buffers,
            const FullAttentionLayerIdentity &identity, int seq_len, int batch_size,
            IKVCache *kv_cache, const int *position_ids, const void *position_ids_device,
            const int32_t *sequence_lengths_device, DeviceId device,
            const std::string &stage_prefix_override, int first_seq_idx,
            const Qwen4ExpPredictorResidualBinding *predictor);

        // =====================================================================
        // GDN Attention Sub-Graph Building
        // =====================================================================

    protected:
        /**
         * @brief Compose the real GDN Stage6 output projection from frozen source and query ownership.
         * @param graph Participant-local graph receiving the projection node.
         * @param prefix Original layer prefix, independent of arithmetic selection.
         * @param buffers Existing GDN output/input and projection output arena bindings.
         * @param weight Original immutable output matrix (or existing baseline optional operand).
         * @param binding Source-authenticated prepared-weight binding for that exact matrix.
         * @param seq_len Physical query rows per request in this graph.
         * @param batch_size Number of independent request prefixes.
         * @param device Exact projection/prepared-weight/count owner.
         * @param lengths Original grouped verifier count; never an ordinary query substitute.
         * @param dependency Gated norm producer of the existing GDN output tensor.
         * @return The actual GEMMStage node name, retaining original source and query pointers.
         * @throws std::exception Before node insertion for unsupported explicit TwoTerm source or query ownership.
         */
        std::string addGDNOutputProjection(ComputeGraph &graph, const std::string &prefix,
            ActivationBuffers &buffers, TensorBase *weight, const WeightBinding *binding,
            int seq_len, int batch_size, DeviceId device, const int32_t *lengths,
            const std::string &dependency);

        /**
         * @brief Compose actual main FA Stage5 from frozen original source/query authority.
         * @param graph Participant graph receiving the projection after its real output gate.
         * @param prefix Existing original layer node namespace.
         * @param buffers Canonical original attention output and projection result owners.
         * @param weight Original immutable Q8_0 matrix for explicit correction.
         * @param binding Exact prepared source binding; no source copy or new preparation.
         * @param source_layer Original global main layer, independent of PP-local cache indices.
         * @param seq_len Physical token-query rows per request.
         * @param batch_size Independent request count, one for explicit correction.
         * @param device Actual projection/source/count participant.
         * @param lengths Original formal verifier count, checked against complete input.
         * @param dependency Actual sigmoid-output-gate producer.
         * @return Actual existing GEMMStage node name and arithmetic/capture identity.
         * @throws std::exception Before insertion for unsupported source/query geometry.
         */
        std::string addFAOutputProjection(ComputeGraph &graph, const std::string &prefix,
            ActivationBuffers &buffers, TensorBase *weight, const WeightBinding *binding,
            int source_layer, int seq_len, int batch_size, DeviceId device,
            const int32_t *lengths, const std::string &dependency);

        /**
         * @brief Authenticate only the selected original main GDN-QKV source and original query rows.
         * @param weight Original unchanged native Q8_0 source matrix.
         * @param binding Actual full-source immutable prepared binding.
         * @param source_layer Original model-global main layer coordinate.
         * @param seq_len Physical query rows per request.
         * @param batch_size Independent request count; correction supports R1.
         * @param device Exact prepared-source/query participant.
         * @param lengths Original formal native query-count argument.
         * @return No QKV-only override for SingleQ8, otherwise canonical complete source-owned query geometry.
         * @throws std::exception Before node insertion for foreign/unprepared/partial source or invalid query authority.
         */
        std::optional<DeviceRowRange> gdnQKVProjectionRows(TensorBase *weight,
            const WeightBinding *binding, int source_layer, int seq_len, int batch_size,
            DeviceId device, const int32_t *lengths) const;

    private:
        /**
         * @brief Resolve only the explicit GDN correction's complete input and query contract.
         * @param weight Original output matrix; selected operations require unchanged native Q8_0.
         * @param binding Actual immutable full-source binding in the prepared store.
         * @param seq_len Physical query rows per request.
         * @param batch_size Independent request count.
         * @param device Exact source, query and graph participant.
         * @param lengths Grouped verifier's original count owner.
         * @return Existing baseline rows or exact selected prefill/verifier/compact query geometry.
         * @throws std::exception For missing/unprepared/foreign source, padded query or unsupported request geometry.
         */
        std::optional<DeviceRowRange> gdnOutputProjectionRows(TensorBase *weight,
            const WeightBinding *binding, int seq_len, int batch_size,
            DeviceId device, const int32_t *lengths) const;

        /**
         * @brief Authenticate main-only FA output source, prepared interval and original query.
         * @param weight Original output matrix; no mirror or converted format is accepted.
         * @param binding Canonical immutable source and existing prepared store identity.
         * @param source_layer Original model-global main coordinate, not a stage/cache index.
         * @param seq_len Physical token-query rows.
         * @param batch_size Independent request count.
         * @param device Actual participant for source and query authority.
         * @param lengths Original formal verifier count, never inferred from KV length.
         * @return Baseline descriptor or the canonical selected single-request query range.
         * @throws std::exception Before node insertion for unsupported explicit invocation.
         */
        std::optional<DeviceRowRange> faOutputProjectionRows(TensorBase *weight,
            const WeightBinding *binding, int source_layer, int seq_len, int batch_size,
            DeviceId device, const int32_t *lengths) const;

        bool gdnLiveStateAllGatherAvailable(int total_tokens, DeviceId device) const;

        /**
         * @brief Build GDN attention sub-graph for a single layer
         *
         * Stages: norm → gdn_proj → short_conv → gdn_recurrence → gated_norm
         *         → gdn_out_proj → attn_output_gate → residual_add
         */
        ComputeGraph buildGDNAttentionGraph(
            const LayerWeights &layer,
            ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            int batch_size,
            IKVCache *kv_cache,
            DeviceId device,
            const std::vector<int> *sequence_lengths,
            const int32_t *sequence_lengths_device);

        /**
         * @brief Check if a layer uses GDN (vs full attention)
         */
        bool isGDNLayer(int layer_idx) const;

        /** Graph-construction-only owner for depth-specific sidecar policy. */
        bool mtp_graph_context_active_ = false;
        /** Recursive depth paired with @ref mtp_graph_context_active_. */
        int mtp_graph_depth_idx_ = -1;
    };

} // namespace llaminar2
