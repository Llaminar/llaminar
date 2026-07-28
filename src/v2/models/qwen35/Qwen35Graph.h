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
        void resetState() override {};

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

        ComputeGraph buildMTPGraph(
            int depth_idx,
            const MTPDepthWeights &weights,
            const MTPForwardInput &input,
            MTPForwardOutput &output);

        ComputeGraph buildMTPGraph(
            int depth_idx,
            const MTPDepthWeightBindings &bindings,
            const MTPForwardInput &input,
            MTPForwardOutput &output) override;

            /**
             * @brief Resolve the global GDN value-head offset for a local TP shard.
             *
             * GDN value heads can have a different count from FA/Q attention heads,
             * so recurrence state indexing must follow the actual value-projection
             * shard rather than GraphConfig::head_start when slice metadata exists.
             */
            static int resolveGDNGlobalVHeadOffset(
                const WeightBinding *value_projection_binding,
                int d_v,
                int n_v_heads,
                int n_v_heads_full,
                const GraphConfig &config,
                const IMPIContext *mpi_ctx);

    protected:
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

        ComputeGraph buildFAKVCacheAppendGraph(
            const LayerWeights &layer,
            ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            int batch_size,
            IKVCache *kv_cache,
            const int *position_ids,
            const void *position_ids_device,
            DeviceId device,
            const std::string &stage_prefix_override = {},
            bool layer_idx_is_cache_local = false,
            int first_seq_idx = 0);

        // =====================================================================
        // GDN Attention Sub-Graph Building
        // =====================================================================

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
    };

} // namespace llaminar2
