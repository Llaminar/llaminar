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
            const std::vector<int> *sequence_lengths = nullptr) override;

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
            DeviceId device,
            const std::vector<int> *sequence_lengths);

        // =====================================================================
        // GDN Attention Sub-Graph Building
        // =====================================================================

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
            DeviceId device);

        /**
         * @brief Check if a layer uses GDN (vs full attention)
         */
        bool isGDNLayer(int layer_idx) const;
    };

} // namespace llaminar2
