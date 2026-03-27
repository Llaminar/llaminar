/**
 * @file GraphTypes.h
 * @brief Model-agnostic types for graph configuration, weights, and activation buffers
 *
 * These types are used by all model architectures (Qwen2, Qwen3, Llama, etc.)
 * and contain no model-specific fields. They were originally defined as Qwen2*
 * types but are generic to any transformer model.
 *
 * Includes:
 * - GraphConfig: Architecture + execution configuration
 * - LayerWeights: Per-layer weight pointers for attention + FFN
 * - ModelWeights: Model-level weight pointers + per-layer accessor
 * - ActivationBuffers: Per-layer activation/scratch buffers
 * - ModelBuffers: Model-level buffers (hidden state, logits, layer buffers)
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../execution/local_execution/graph/DeviceGraphExecutor.h"
#include "../execution/config/ExecutionPolicy.h"
#include "../execution/config/RuntimeConfig.h"
#include "../backends/DeviceId.h"
#include "../config/TensorParallelConfig.h"
#include "../config/TPDomain.h"
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace llaminar2
{

    // Forward declarations
    class TensorBase;
    class ILocalTPContext;
    class ILocalPPContext;
    class TurboQuantContext;
    struct PipelineConfig;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Model-agnostic graph configuration
     *
     * Combines architecture parameters with execution settings.
     * Used by all transformer model architectures.
     */
    struct GraphConfig
    {
        // Model architecture
        int n_layers = 0;       ///< Number of transformer layers (local count for PP stages)
        int total_n_layers = 0; ///< Total model layers (for absolute-index validation in PP)
        int d_model = 0;        ///< Model hidden dimension
        int n_heads = 0;        ///< Number of attention heads
        int n_kv_heads = 0;     ///< Number of KV heads (GQA)
        int head_dim = 0;       ///< Dimension per head
        int d_ff = 0;           ///< FFN intermediate dimension
        int vocab_size = 0;     ///< Vocabulary size

        /// Pipeline Parallelism layer offset for KV cache indexing.
        /// When building graphs for PP stage [first_layer, last_layer), this offset
        /// is subtracted from the global layer index to get the local KV cache index.
        /// E.g., for PP stage 1 with layers [12, 24), pp_layer_offset=12, so layer 12
        /// maps to KV cache layer 0.
        int pp_layer_offset = 0;

        // FFN sharding (for tensor parallelism)
        int d_ff_local = 0; ///< Local FFN dim per rank
        bool ffn_column_parallel = false;

        // =================================================================
        // LM Head TP Parameters (Phase 5: Column-Parallel LM Head)
        // =================================================================
        /// Local vocab size per rank (vocab_size / world_size)
        int vocab_local = 0;

        /// Enable column-parallel LM head projection (weights sharded by vocab)
        bool lm_head_column_parallel = false;

        // =================================================================
        // Attention TP Parameters (Phase 3: Column-Parallel QKV)
        // =================================================================
        /// First query head for this rank (0-indexed, default 0 = no sharding)
        int head_start = 0;

        /// Number of query heads for this rank (default -1 = use full n_heads)
        int local_n_heads = -1;

        /// Number of KV heads for this rank (default -1 = use full n_kv_heads)
        /// For GQA: may equal local_n_heads / gqa_ratio
        int local_n_kv_heads = -1;

        /// Enable column-parallel QKV projection (weights sharded by head)
        bool qkv_column_parallel = false;

        // Precision and execution
        float rms_norm_eps = 1e-6f;
        float rope_theta = 10000.0f;
        ActivationPrecision activation_precision = ActivationPrecision::FP32;

        // =================================================================
        // Q16 KV Cache VNNI Safety (Phase 5.4)
        // =================================================================
        /// Fixed scale for Q16 KV cache quantization (FP32 range: ±kv_cache_scale).
        /// All K/V values are quantized with this scale, producing INT16 values in
        /// range [-32767 * kv_cache_scale, +32767 * kv_cache_scale]. The value must
        /// match the expected activation range to avoid clipping.
        /// See VNNISafetyConstants.h for VNNI overflow limits.
        float kv_cache_scale = 256.0f; ///< Fixed Q16 scale. Must cover Q projection max (~130)

        /// Explicit KV cache precision mode (AUTO preserves legacy behavior).
        KVCachePrecision kv_cache_precision = KVCachePrecision::AUTO;

        /// TurboQuant context for TQ4 KV cache quantization.
        /// Holds rotation matrix. Not owned by GraphConfig.
        const TurboQuantContext *turboquant_ctx = nullptr;

        /// RoPE-on-read mode: store pre-RoPE K in the KV cache and apply
        /// position embeddings lazily during attention (fused with TQ4 dequant).
        /// Benefits: (1) fused dequant+RoPE is nearly free (O(D) vs O(D²) dequant),
        /// (2) position-free cache enables speculative decoding,
        /// (3) eliminates separate RoPE computation for K.
        /// Currently supported for TQ4 (fused) and FP32 (in-place) KV precision.
        bool rope_on_read = false;

        // Execution settings
        DeviceId default_device = DeviceId::cpu(); ///< Default device for execution
        bool enable_profiling = false;
        bool enable_validation = false;

        /// Use graph-managed buffer allocation with aliasing optimization.
        /// When true, the graph builder uses DeviceGraphBufferManager to allocate
        /// activation buffers with automatic aliasing of non-overlapping SCRATCH buffers.
        bool use_graph_buffer_management = true;

        /// Maximum sequence length for buffer allocation (when use_graph_buffer_management=true)
        int max_seq_len = 2048;

        /// Execution policy controlling which operations run
        ExecutionPolicy execution_policy = ExecutionPolicy::allEnabled();

        /// Base DeviceGraphExecutor configuration
        GraphExecutorConfig executor_config = GraphExecutorConfig{};

        /// Fused attention backend selection
        /// - JIT: AVX-512 VNNI optimized (default)
        /// - REFERENCE: Pure C++ implementation for testing
        /// - TILED: Cache-blocked implementation
        /// - Q16_INTEGER: Pure Q16 integer-domain kernel (experimental, requires HybridQ16)
        FusedAttentionBackend fused_attention_backend = FusedAttentionBackend::JIT;

        // =================================================================
        // Heterogeneous Tensor Parallelism (Phase 1c: Proportional TP)
        // =================================================================
        /// Optional TensorParallelConfig for proportional head/FFN/vocab assignment.
        /// When set, overrides equal-split head/KV/FFN computation.
        /// Use for heterogeneous GPU setups (e.g., NVIDIA 73% + AMD 27%).
        std::shared_ptr<TensorParallelConfig> tp_config = nullptr;

        /// Local rank index (used with tp_config to look up assignment)
        int local_rank = 0;

        // =================================================================
        // Multi-Domain Tensor Parallelism (Phase 4.3: Heterogeneous TP)
        // =================================================================
        /// Optional MultiDomainTPConfig for domain-based collective routing.
        /// When set, collective operations (AllreduceStage, AllGatherStage) will
        /// use the domain communicator instead of the global MPI communicator.
        /// This enables heterogeneous TP with separate GPU and CPU domains.
        MultiDomainTPConfig *multi_domain_tp_config = nullptr;

        // =================================================================
        // LOCAL Tensor Parallelism (Intra-Rank Multi-Device)
        // =================================================================
        /// Optional ILocalTPContext for LOCAL tensor parallelism.
        /// When set, collective operations use TPAllreduceStage with the local
        /// TP context. LOCAL TP runs on a single MPI rank with multiple devices,
        /// using high-bandwidth backends like NCCL, RCCL, or PCIeBAR for collectives.
        ///
        /// Distinction from GLOBAL TP:
        /// - GLOBAL TP: Multiple MPI ranks (world_size > 1), MPI collectives
        /// - LOCAL TP: Single MPI rank (world_size = 1), ILocalTPContext collectives
        ///
        /// Note: Either mpi_ctx (for GLOBAL TP) OR local_tp_ctx (for LOCAL TP) should
        /// be active, not both. If both are set, LOCAL TP takes precedence for collectives.
        ILocalTPContext *local_tp_ctx = nullptr;

        /// Device index within the LOCAL TP context (0 to degree-1).
        /// Each device runs a separate graph instance with sharded weights.
        int local_tp_device_idx = 0;

        // =================================================================
        // Unified Pipeline Parallel Configuration (Phase 2)
        // =================================================================
        /// Pipeline configuration for PP + TP composition.
        /// When set, graph building uses multi-stage PP-aware logic.
        std::shared_ptr<PipelineConfig> pipeline_config = nullptr;

        /// PP contexts for inter-stage activation transfers.
        /// Key: {from_stage_id, to_stage_id}
        /// Created by the orchestrator and passed to the graph builder.
        std::map<std::pair<int, int>, ILocalPPContext *> pp_contexts;

        /// TP contexts for each domain (one per domain name).
        /// Each domain may have internal tensor parallelism.
        /// Created by the orchestrator and passed to the graph builder.
        std::map<std::string, ILocalTPContext *> domain_tp_contexts;

        /// Helper: Check if unified PP mode is enabled
        bool hasUnifiedPP() const;

        /// Helper: Get the device for a specific layer in unified PP mode
        DeviceId getDeviceForLayer(int layer_idx) const;

        /**
         * @brief Helper to get DeviceShardingAssignment for current rank
         * @return Pointer to assignment, or nullptr if tp_config is not set
         */
        const DeviceShardingAssignment *getAssignment() const
        {
            if (!tp_config)
                return nullptr;
            return &tp_config->forRank(local_rank);
        }

        // =================================================================
        // Per-Layer TP Allreduce Precision
        // =================================================================

        /// Per-layer allreduce precision map: layer_idx -> precision string.
        /// Populated from the GraphSchema precision policy (first N layers FP32,
        /// rest use schema default). Layers not in the map fall back to the
        /// global DebugEnv::allreduce_precision ("fp32" by default).
        std::unordered_map<int, std::string> tp_allreduce_precision;

        /**
         * @brief Get allreduce precision for a specific layer
         *
         * Resolution order:
         * 1. Per-layer override from tp_allreduce_precision map
         * 2. Global fallback from debugEnv().allreduce_precision
         *
         * @param layer_idx Transformer layer index (0-based)
         * @return Precision string ("fp32", "fp16", "bf16")
         */
        std::string getAllreducePrecisionForLayer(int layer_idx) const
        {
            auto it = tp_allreduce_precision.find(layer_idx);
            if (it != tp_allreduce_precision.end())
                return it->second;
            return ""; // Empty = defer to global DebugEnv default
        }

        /**
         * @brief Populate the precision map from a GraphSchema precision policy
         *
         * Applies the schema's fp32_layer_count + default_precision to build
         * the per-layer map for this model's n_layers.
         *
         * @param schema_default Default precision for layers beyond fp32 count
         * @param fp32_count Number of initial layers forced to FP32
         */
        void populateAllreducePrecision(const std::string &schema_default, int fp32_count)
        {
            tp_allreduce_precision.clear();
            for (int i = 0; i < n_layers; ++i)
            {
                tp_allreduce_precision[i] = (i < fp32_count) ? "fp32" : schema_default;
            }
        }
    };

    // =========================================================================
    // Weight Structures
    // =========================================================================

    /**
     * @brief Layer weights for attention and FFN blocks
     *
     * Raw pointers since the graph builder does NOT own these weights.
     * Supports all standard transformer architectures:
     * - Qwen2: uses q_bias, k_bias, v_bias
     * - Qwen3: uses q_norm, k_norm (QK normalization)
     * - Llama: uses subset (no biases, no QK norms)
     */
    struct LayerWeights
    {
        // Attention weights
        TensorBase *wq = nullptr;        ///< Query projection
        TensorBase *wk = nullptr;        ///< Key projection
        TensorBase *wv = nullptr;        ///< Value projection
        TensorBase *wo = nullptr;        ///< Output projection
        TensorBase *attn_norm = nullptr; ///< Pre-attention norm gamma

        // Attention biases (optional, e.g., Qwen2 uses Q/K/V biases)
        TensorBase *q_bias = nullptr; ///< Query bias [d_model]
        TensorBase *k_bias = nullptr; ///< Key bias [n_kv_heads * head_dim]
        TensorBase *v_bias = nullptr; ///< Value bias [n_kv_heads * head_dim]

        // QK norm weights (optional, e.g., Qwen3 per-head RMSNorm before RoPE)
        TensorBase *q_norm = nullptr; ///< Q norm gamma [head_dim]
        TensorBase *k_norm = nullptr; ///< K norm gamma [head_dim]

        // FFN weights
        TensorBase *gate_proj = nullptr; ///< FFN gate projection
        TensorBase *up_proj = nullptr;   ///< FFN up projection
        TensorBase *down_proj = nullptr; ///< FFN down projection
        TensorBase *ffn_norm = nullptr;  ///< Pre-FFN norm gamma
    };

    /**
     * @brief Model-level weights
     *
     * Provides access to global weights and per-layer accessor.
     */
    struct ModelWeights
    {
        TensorBase *embedding_table = nullptr; ///< [vocab_size, d_model]
        TensorBase *final_norm = nullptr;      ///< [d_model]
        TensorBase *lm_head = nullptr;         ///< [vocab_size, d_model]

        /// Accessor for per-layer weights
        std::function<LayerWeights(int layer_idx)> get_layer_weights;
    };

    // =========================================================================
    // Activation Buffers
    // =========================================================================

    /**
     * @brief Activation buffers for layer execution
     *
     * ## Buffer Lifecycle Semantics
     *
     * **INOUT (Input modified in-place)**:
     *   - `residual`: Accumulates attention/FFN outputs via +=
     *   - `normalized`: Receives norm output, may be reused
     *
     * **SCRATCH (Temporary workspace)**:
     *   - `Q`, `K`, `V`: Projection outputs, consumed by attention
     *   - `attn_output`: Pre-Wo output, consumed by projection
     *   - `gate`, `up`: FFN projections, consumed by SwiGLU
     *   - `ffn_output`: SwiGLU output, consumed by down projection
     *   - `workspace_scores`, `workspace_context`, `workspace_mask`: Attention scratch
     *
     * **OUTPUT (Write-only)**:
     *   - `current_hidden`: Final hidden state output
     *   - `attn_proj`: After Wo projection, before residual add
     */
    struct ActivationBuffers
    {
        // === INOUT Buffers ===
        TensorBase *residual = nullptr;
        TensorBase *normalized = nullptr;

        // === SCRATCH Buffers ===
        TensorBase *Q = nullptr;
        TensorBase *K = nullptr;
        TensorBase *V = nullptr;
        TensorBase *attn_output = nullptr;
        TensorBase *gate = nullptr;
        TensorBase *up = nullptr;
        TensorBase *ffn_output = nullptr;
        TensorBase *workspace_scores = nullptr;
        TensorBase *workspace_context = nullptr;
        TensorBase *workspace_mask = nullptr;

        // === Hybrid Mode Buffers ===
        /// FP32 Q after RoPE (Hybrid mode only - avoids requantization)
        /// When set, RoPE outputs to this buffer instead of modifying Q in-place
        TensorBase *Q_rope = nullptr;
        /// FP32 K after RoPE (Hybrid mode only - avoids requantization)
        /// When set, RoPE outputs to this buffer instead of modifying K in-place
        TensorBase *K_rope = nullptr;
        /// FP32 V dequantized (Hybrid mode only - for KV cache when V is Q8_1)
        /// V doesn't go through RoPE but needs to match KV cache precision
        TensorBase *V_dequant = nullptr;

        // === Batched Decode Buffers (for gather from multiple cache slots) ===
        TensorBase *gathered_K = nullptr; ///< [batch_size * max_kv_len, kv_dim]
        TensorBase *gathered_V = nullptr; ///< [batch_size * max_kv_len, kv_dim]

        // === HybridQ16 K Precision Fix: Per-head K scales ===
        /// Per-head dynamic scales from RoPE Q16->Q16 path
        /// Shape: [seq_len * n_kv_heads] for prefill, stored in KV cache for decode
        /// Only used when GEMM outputs K as Q16_1 (K precision fix mode)
        float *K_head_scales = nullptr;
        /// Capacity of K_head_scales buffer (in floats)
        size_t K_head_scales_capacity = 0;

        // === OUTPUT Buffers ===
        TensorBase *attn_proj = nullptr;
        TensorBase *current_hidden = nullptr;

        // === SNAPSHOT Buffers (for debugging, enabled with ENABLE_PIPELINE_SNAPSHOTS) ===
        /// Optional buffer to capture attention context before Wo projection
        /// Shape: [batch_size * seq_len, n_heads * head_dim]
        TensorBase *context_snapshot = nullptr;

        /// Optional buffer to capture attention output (Wo projection result, before residual add)
        /// Shape: [batch_size * seq_len, d_model] - corresponds to ATTENTION_OUTPUT
        TensorBase *attention_output_snapshot = nullptr;

        /// Optional buffer to capture attention residual (after residual add)
        /// Shape: [batch_size * seq_len, d_model] - corresponds to ATTENTION_RESIDUAL
        TensorBase *attention_residual_snapshot = nullptr;
    };

    /**
     * @brief Model-level buffers for full forward pass
     */
    struct ModelBuffers
    {
        TensorBase *current_hidden = nullptr; ///< [batch_size * seq_len, d_model]
        TensorBase *logits = nullptr;         ///< [batch_size * seq_len, vocab_size]

        /// Local logits for column-parallel LM head [batch_size * seq_len, vocab_local]
        /// Only used when lm_head_column_parallel is enabled
        TensorBase *logits_local = nullptr;

        /// Per-layer activation buffers
        ActivationBuffers layer_buffers;
    };

    // =========================================================================
    // Backward-compatible type aliases
    // =========================================================================
    using Qwen2GraphConfig = GraphConfig;
    using Qwen2LayerWeights = LayerWeights;
    using Qwen2ModelWeights = ModelWeights;
    using Qwen2ActivationBuffers = ActivationBuffers;
    using Qwen2ModelBuffers = ModelBuffers;

} // namespace llaminar2
