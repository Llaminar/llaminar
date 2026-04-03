/**
 * @file Qwen35Graph.h
 * @brief Qwen 3.5 compute graph builder for hybrid GDN + full attention architecture
 *
 * Extends Qwen2Graph to support the Qwen 3.5 "Dense" architecture which uses
 * heterogeneous transformer layers:
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

#include "../qwen/Qwen2Graph.h"
#include <memory>

namespace llaminar2
{
    class ITensorShortConvolution;
    class ITensorGatedDeltaNet;
} // namespace llaminar2

namespace llaminar2
{

    /**
     * @brief Qwen 3.5 graph builder extending Qwen2Graph with GDN layer support
     *
     * Overrides:
     *   - architectureName()     → "qwen35"
     *   - getSchema()            → Qwen35SchemaFactory with named templates
     *   - buildFullForwardGraph() → Dispatches per-layer between GDN and FA paths
     *   - buildLayerGraph()      → Single-layer dispatch
     *   - buildAttentionGraph()  → Per-layer dispatch for attention sub-graph
     */
    class Qwen35Graph : public Qwen2Graph
    {
    public:
        /// Construct with full model context
        Qwen35Graph(std::shared_ptr<ModelContext> model_ctx,
                    std::shared_ptr<MPIContext> mpi_ctx,
                    const GraphConfig &config);

        /// Construct for layer-level operations only
        Qwen35Graph(const GraphConfig &config,
                    std::shared_ptr<MPIContext> mpi_ctx = nullptr);

        ~Qwen35Graph() = default;

        // =====================================================================
        // IGraphBuilder overrides
        // =====================================================================

        std::string architectureName() const override { return "qwen35"; }

        GraphSchema getSchema() const override;

        /// Layer dispatch delegates to GDN or FA based on layer type.
        /// buildFullForwardGraph() inherited from Qwen2Graph calls
        /// buildAttentionGraph() virtually, so GDN dispatch is automatic.

        ComputeGraph buildLayerGraph(const LayerContext &ctx) override;

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

    private:
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
            DeviceId device);

        /**
         * @brief Check if a layer uses GDN (vs full attention)
         */
        bool isGDNLayer(int layer_idx) const;

        // =====================================================================
        // GDN State Management
        // =====================================================================

        /// Per-layer conv state buffers [channels, kernel_size - 1]
        /// Persistent across decode steps, initialized to zero on first use
        std::vector<std::vector<float>> conv_states_;

        /// Per-layer recurrence state buffers [n_heads, d_k, d_v]
        /// Persistent across decode steps, initialized to zero on first use
        std::vector<std::vector<float>> recurrence_states_;

        /// Per-layer GDN kernel instances (kept alive for stages that hold raw ptrs)
        std::vector<std::shared_ptr<ITensorShortConvolution>> conv_kernels_;
        std::vector<std::shared_ptr<ITensorGatedDeltaNet>> rec_kernels_;

        /// Initialize GDN state buffers if not already allocated
        void ensureGDNStates();
    };

} // namespace llaminar2
