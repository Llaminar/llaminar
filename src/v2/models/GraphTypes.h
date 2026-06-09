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
#include "../memory/BufferId.h"
#include "../config/TensorParallelConfig.h"
#include "../config/TPDomain.h"
#include "../loaders/WeightPlan.h"
#include "../utils/ToolCallTypes.h"
#include <algorithm>
#include <functional>
#include <exception>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace llaminar2
{

    // Forward declarations
    class TensorBase;
    class IKVCache;
    class IMPIContext;
    class ITPContext;
    class ILocalTPContext;
    class ILocalPPContext;
    class TurboQuantContext;
    class ActivationRotation;
    class DecodeExpertHistogram;
    struct MoEExpertParallelPlan;
    struct MoEExpertOverlayExecutionPlan;
    class MoEExpertOverlayRuntimePlan;
    struct PipelineConfig;
    enum class MoERebalanceMode;

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
        /// Fixed scale for Q16 KV cache quantization.
        /// K and V use separate scales because their activation ranges differ:
        /// K has large outliers from RoPE (especially low-frequency dims), while
        /// V values are typically much smaller. Separate scales avoid wasting
        /// INT16 precision range on V.
        /// See VNNISafetyConstants.h for VNNI overflow limits.
        float kv_cache_scale_k = 256.0f; ///< K scale (FP32 range ±scale_k). Covers post-RoPE K max
        float kv_cache_scale_v = 32.0f;  ///< V scale (FP32 range ±scale_v). V activations are small

        /// Explicit KV cache precision mode (AUTO preserves legacy behavior).
        KVCachePrecision kv_cache_precision = KVCachePrecision::AUTO;

        /// Prefix-state cache feature gates and storage limits.
        PrefixCacheRuntimeConfig prefix_cache;

        /// Multi-token prediction feature gates and verification mode.
        MTPRuntimeConfig mtp;

        /// Runtime-only decode verifier mode: compute LM-head logits for every
        /// input row instead of the selected final row.
        bool compute_all_position_logits = false;

        /// Runtime-only verifier optimization: pack selected hidden rows into a
        /// compact scratch before LM head instead of projecting every row in
        /// the verifier activation tensor. Valid only when
        /// compute_all_position_logits is enabled.
        bool compute_row_indexed_logits = false;
        int row_indexed_logits_row_count = 0;

        /// TurboQuant context for TQ4 KV cache quantization.
        /// Holds rotation matrix. Not owned by GraphConfig.
        const TurboQuantContext *turboquant_ctx = nullptr;

        /// Block-diagonal orthogonal rotation applied to K/V before Q16_1
        /// quantization to reduce activation kurtosis. When non-null, Q is also
        /// rotated before the attention dot product and the output is inverse-
        /// rotated after the weighted-V accumulation. Not owned by GraphConfig.
        const ActivationRotation *kv_rotation = nullptr;

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

        // =================================================================
        // Tool Calling Configuration
        // =================================================================
        /// Tool call output format for this model. Determines how raw model
        /// output text is parsed into structured tool_calls objects.
        /// Default: HERMES_2_PRO (covers Qwen 2.5, Qwen 3, Hermes, etc.)
        /// Set to NONE to disable tool call detection.
        ToolCallFormat tool_call_format = ToolCallFormat::HERMES_2_PRO;

        /// Use graph-managed buffer allocation with aliasing optimization.
        /// When true, the graph builder uses DeviceGraphBufferManager to allocate
        /// activation buffers with automatic aliasing of non-overlapping SCRATCH buffers.
        bool use_graph_buffer_management = true;

        /// Maximum sequence length for buffer allocation (when use_graph_buffer_management=true)
        int max_seq_len = 4096;

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
        // Tensor Parallelism Context (Polymorphic — LOCAL or GLOBAL)
        // =================================================================
        /// Polymorphic ITPContext for tensor parallelism collective operations.
        /// Set to either an ILocalTPContext* (intra-rank multi-device: NCCL/RCCL/HOST)
        /// or an IGlobalTPContext* (cross-rank: MPI/UPI).
        ///
        /// Graph builders use this via TPAllreduceStage for all TP modes.
        /// Code needing LOCAL TP-specific features (device lists)
        /// should check tp_ctx->isLocal() and static_cast<ILocalTPContext*>.
        ///
        /// In a nested PP+TP topology like:
        ///   PipelineParallel(LocalTP(cuda:0..3), GlobalTP(cpu:0, cpu:1))
        /// each PP stage's GraphConfig has its own tp_ctx for that stage's TP domain.
        ITPContext *tp_ctx = nullptr;

        /// Device index within the TP context (0 to degree-1).
        /// For LOCAL TP: which device within the local TP group.
        /// For GLOBAL TP: which MPI rank within the TP domain.
        /// Each device runs a separate graph instance with sharded weights.
        int tp_device_idx = 0;

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
        std::map<std::string, ITPContext *> domain_tp_contexts;

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

        /**
         * @brief Populate the precision map with layer-type awareness
         *
         * For hybrid architectures (e.g., Qwen3.5 GDN+FA), certain layer types
         * may require FP32 allreduce regardless of their position. This overload
         * forces FP32 for layers in fp32_forced_layers, and applies the standard
         * count-based policy to all other layers.
         *
         * @param schema_default Default precision for non-forced layers beyond fp32 count
         * @param fp32_count Number of initial non-forced layers that get FP32
         * @param fp32_forced_layers Layer indices always forced to FP32 (e.g., FA layers)
         */
        void populateAllreducePrecision(const std::string &schema_default, int fp32_count,
                                        const std::set<int> &fp32_forced_layers)
        {
            tp_allreduce_precision.clear();
            int non_forced_idx = 0; // Count of non-forced layers seen so far
            for (int i = 0; i < n_layers; ++i)
            {
                if (fp32_forced_layers.count(i))
                {
                    tp_allreduce_precision[i] = "fp32";
                }
                else
                {
                    tp_allreduce_precision[i] = (non_forced_idx < fp32_count) ? "fp32" : schema_default;
                    ++non_forced_idx;
                }
            }
        }

        /// Returns true when activation precision is HybridQ16.
        /// Replaces repeated InferenceMode(activation_precision).isHybridQ16() calls.
        bool isHybridQ16() const
        {
            return activation_precision == ActivationPrecision::HybridQ16;
        }

        // =================================================================
        // MoE Configuration
        // =================================================================

        struct MoEConfig
        {
            int num_experts = 0;              ///< Total expert count (e.g. 256 for Qwen3.5)
            int top_k = 0;                    ///< Experts activated per token (e.g. 8)
            int intermediate_size = 0;        ///< Per-expert FFN intermediate dim
            bool norm_topk_prob = false;      ///< Normalize top-k routing weights
            bool has_shared_expert = false;   ///< Has always-active shared expert
            int shared_intermediate_size = 0; ///< Shared expert FFN intermediate dim
            bool shared_expert_gate = false;  ///< Has sigmoid gating on shared expert

            /// Routed expert execution mode for the standard graph path.
            MoEExpertMode expert_mode = MoEExpertMode::ExpertParallel;

            /// Static contiguous expert-id range owned by this TP participant.
            /// count < 0 means the routed expert output is full/replicated.
            int local_expert_start = 0;
            int local_expert_count = -1;

            /// Bounded remote hot-expert cache configuration for dynamic EP.
            MoEHotExpertCacheConfig hot_expert_cache;

            /// Runtime rebalance config carried for diagnostics and controller setup.
            MoERebalanceRuntimeConfig rebalance_config;

            /// Optional histogram for decode expert tracking.
            /// Set by the orchestrator when MoE rebalancing is enabled.
            /// Lifetime managed by MoERebalanceController. Not owned.
            DecodeExpertHistogram *decode_histogram = nullptr;

            /// MoE rebalancing mode (OFF / OBSERVE / DYNAMIC).
            /// Set by InferenceRunnerFactory from MoERebalanceController.
            MoERebalanceMode rebalance_mode{}; // default-initialized to OFF (value 0)

            /// Optional same-layer expert-parallel overlay plan.
            /// Phase 1 stores the validated value only; graph execution remains unchanged.
            std::shared_ptr<MoEExpertParallelPlan> expert_parallel_plan = nullptr;

            /// Runtime-resolved overlay descriptor for domain devices, ranks, and MVP lowering.
            std::shared_ptr<MoEExpertOverlayRuntimePlan> expert_overlay_runtime_plan = nullptr;

            /// Optional rank-role execution plan used by overlay preparation and diagnostics.
            std::shared_ptr<const MoEExpertOverlayExecutionPlan> expert_overlay_execution_plan = nullptr;

            /// Optional MPI context dedicated to overlay domain-worker commands.
            /// It is intentionally separate from the graph-builder MPI context so
            /// continuation-root graphs can avoid unrelated world collectives.
            std::shared_ptr<IMPIContext> overlay_mpi_ctx = nullptr;

            /// Returns true if MoE is enabled
            bool enabled() const { return num_experts > 0 && top_k > 0; }
        } moe;

        // =================================================================
        // Heterogeneous Layer Configuration (Phase B)
        // =================================================================

        /// Per-layer type names (e.g., "full_attention", "gdn").
        /// When non-empty, size must equal n_layers.
        /// Used by schema factories to assign named templates.
        std::vector<std::string> layer_types;

        /// Per-layer FFN intermediate dimensions (for variable-width FFN).
        /// When non-empty, size must equal n_layers.
        /// Empty = all layers use `d_ff`.
        std::vector<int> layer_d_ff;

        /// Per-layer KV head counts (for variable GQA ratios).
        /// When non-empty, size must equal n_layers.
        /// Empty = all layers use `n_kv_heads`.
        std::vector<int> layer_n_kv_heads;

        /// Per-layer query head counts (for variable head configs).
        /// When non-empty, size must equal n_layers.
        /// Empty = all layers use `n_heads`.
        std::vector<int> layer_n_heads;

        /// Partial RoPE factor — fraction of head_dim that receives rotation.
        /// 1.0 = full RoPE (default, standard transformers).
        /// <1.0 = partial RoPE (e.g., 0.5 means first half of head_dim gets RoPE,
        ///         second half is pass-through). Used by Qwen 3.5 GDN layers.
        float partial_rotary_factor = 1.0f;

        // ── GDN (Gated Delta Network) Configuration ─────────────────────

        struct GDNConfig
        {
            int conv_kernel_size = 0; ///< Short convolution kernel width (e.g., 4 for Qwen 3.5)
            int state_size = 0;       ///< Recurrence state dimension (== head_dim)
            int inner_size = 0;       ///< Inner projection size (ssm.inner_size in GGUF)
            int group_count = 0;      ///< Group count (ssm.group_count in GGUF)
            int time_step_rank = 0;   ///< Time-step rank (ssm.time_step_rank in GGUF)

            /// Full attention layer interval (e.g. 4 = every 4th layer is FA)
            /// 0 means all layers use the default template (no hybrid)
            int full_attention_interval = 0;

            /// Returns true if GDN is configured
            bool enabled() const { return conv_kernel_size > 0 && state_size > 0; }
        } gdn;

        /// Per-layer head dimensions (for variable head_dim across layer types).
        /// When non-empty, size must equal n_layers.
        /// Empty = all layers use `head_dim`.
        std::vector<int> layer_head_dim;

        /// Whether attention output gating (sigmoid) is used
        bool has_attention_output_gate = false;

        /// Whether RMSNorm uses subtract-one weight transform:
        /// gamma_effective = 1.0 + gamma_stored (DeepSeek/Qwen3.5 convention)
        bool rms_norm_subtract_one = false;

        /// Get FFN dimension for a specific layer (falls back to d_ff)
        int getLayerDFF(int layer_idx) const
        {
            if (!layer_d_ff.empty() && layer_idx < static_cast<int>(layer_d_ff.size()))
                return layer_d_ff[layer_idx];
            return d_ff;
        }

        /// Get KV head count for a specific layer (falls back to n_kv_heads)
        int getLayerNKVHeads(int layer_idx) const
        {
            if (!layer_n_kv_heads.empty() && layer_idx < static_cast<int>(layer_n_kv_heads.size()))
                return layer_n_kv_heads[layer_idx];
            return n_kv_heads;
        }

        /// Get query head count for a specific layer (falls back to n_heads)
        int getLayerNHeads(int layer_idx) const
        {
            if (!layer_n_heads.empty() && layer_idx < static_cast<int>(layer_n_heads.size()))
                return layer_n_heads[layer_idx];
            return n_heads;
        }

        /// Get head dimension for a specific layer (falls back to head_dim)
        int getLayerHeadDim(int layer_idx) const
        {
            if (!layer_head_dim.empty() && layer_idx < static_cast<int>(layer_head_dim.size()))
                return layer_head_dim[layer_idx];
            return head_dim;
        }

        /// Returns true if layer at idx is a full-attention layer (vs GDN)
        bool isFullAttentionLayer(int layer_idx) const
        {
            if (!layer_types.empty() && layer_idx < static_cast<int>(layer_types.size()))
                return layer_types[layer_idx] == "full_attention";
            return true; // no heterogeneous config = all FA
        }

        /// Returns true if this model has GDN (gated delta network) layers
        bool hasGDN() const { return gdn.enabled(); }

        /// Returns true if this model uses MoE
        bool isMoE() const { return moe.enabled(); }

        /// Returns true if this model has heterogeneous layer types
        bool hasHeterogeneousLayers() const { return !layer_types.empty(); }
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
     * - Qwen3.5: uses GDN weights (attn_qkv, attn_gate, ssm_*, gdn_norm)
     * - Llama: uses subset (no biases, no QK norms)
     */
    struct LayerWeights
    {
        // Attention weights (Full Attention layers)
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

        // GDN (Gated Delta Network) weights — used by Qwen3.5 GDN layers
        TensorBase *attn_qkv = nullptr;    ///< Fused QKV projection [d_model, 2*inner_size]
        TensorBase *attn_gate = nullptr;   ///< Output gate Z [d_model, inner_size]
        TensorBase *ssm_alpha = nullptr;   ///< Alpha projection (decay) [d_model, time_step_rank]
        TensorBase *ssm_beta = nullptr;    ///< Beta projection (input gate) [d_model, time_step_rank]
        TensorBase *ssm_conv1d = nullptr;  ///< Short conv1d weights [kernel, inner_size*2]
        TensorBase *ssm_dt_bias = nullptr; ///< Time-step bias [time_step_rank]
        TensorBase *ssm_a = nullptr;       ///< Decay parameter A [time_step_rank]
        TensorBase *ssm_norm = nullptr;    ///< GDN output norm gamma [state_size]
        TensorBase *ssm_out = nullptr;     ///< GDN output projection [inner_size, d_model]

        // FFN weights
        TensorBase *gate_proj = nullptr; ///< FFN gate projection
        TensorBase *up_proj = nullptr;   ///< FFN up projection
        TensorBase *down_proj = nullptr; ///< FFN down projection
        TensorBase *ffn_norm = nullptr;  ///< Pre-FFN norm gamma

        // MoE (Mixture of Experts) weights — used by Qwen3.5 MoE layers
        TensorBase *moe_gate = nullptr;               ///< Router gate [num_experts, d_model]
        TensorBase *moe_gate_exps = nullptr;          ///< Expert gate projections 3D [num_experts, intermediate, d_model]
        TensorBase *moe_up_exps = nullptr;            ///< Expert up projections 3D [num_experts, intermediate, d_model]
        TensorBase *moe_down_exps = nullptr;          ///< Expert down projections 3D [num_experts, d_model, intermediate]
        TensorBase *shared_expert_gate = nullptr;     ///< Shared expert gate proj [intermediate, d_model]
        TensorBase *shared_expert_up = nullptr;       ///< Shared expert up proj [intermediate, d_model]
        TensorBase *shared_expert_down = nullptr;     ///< Shared expert down proj [d_model, intermediate]
        TensorBase *shared_expert_gate_inp = nullptr; ///< Shared expert sigmoid gate [d_model]

        // Optional graph-facing bindings for prepared-weight lookup. Legacy
        // tests and hand-built graphs can leave these null; frozen model paths
        // populate them through toLegacyLayerWeights().
        const WeightBinding *wq_binding = nullptr;
        const WeightBinding *wk_binding = nullptr;
        const WeightBinding *wv_binding = nullptr;
        const WeightBinding *wo_binding = nullptr;
        const WeightBinding *gate_proj_binding = nullptr;
        const WeightBinding *up_proj_binding = nullptr;
        const WeightBinding *down_proj_binding = nullptr;
    };

    /**
     * @brief Graph-facing layer weight bindings.
     *
     * This is the Phase 6 bridge from frozen model-weight bindings to the
     * legacy TensorBase* stage params. Graph builders can inspect identity,
     * residency, and prepared refs while existing stages continue to consume
     * raw tensor pointers through toLegacyLayerWeights().
     */
    struct LayerWeightBindings
    {
        // Attention weights (Full Attention layers)
        const WeightBinding *wq = nullptr;
        const WeightBinding *wk = nullptr;
        const WeightBinding *wv = nullptr;
        const WeightBinding *wo = nullptr;
        const WeightBinding *attn_norm = nullptr;

        // Attention biases
        const WeightBinding *q_bias = nullptr;
        const WeightBinding *k_bias = nullptr;
        const WeightBinding *v_bias = nullptr;

        // QK norm weights
        const WeightBinding *q_norm = nullptr;
        const WeightBinding *k_norm = nullptr;

        // GDN weights
        const WeightBinding *attn_qkv = nullptr;
        const WeightBinding *attn_gate = nullptr;
        const WeightBinding *ssm_alpha = nullptr;
        const WeightBinding *ssm_beta = nullptr;
        const WeightBinding *ssm_conv1d = nullptr;
        const WeightBinding *ssm_dt_bias = nullptr;
        const WeightBinding *ssm_a = nullptr;
        const WeightBinding *ssm_norm = nullptr;
        const WeightBinding *ssm_out = nullptr;

        // FFN weights
        const WeightBinding *gate_proj = nullptr;
        const WeightBinding *up_proj = nullptr;
        const WeightBinding *down_proj = nullptr;
        const WeightBinding *ffn_norm = nullptr;

        // MoE weights
        const WeightBinding *moe_gate = nullptr;
        const WeightBinding *moe_gate_exps = nullptr;
        const WeightBinding *moe_up_exps = nullptr;
        const WeightBinding *moe_down_exps = nullptr;
        const WeightBinding *shared_expert_gate = nullptr;
        const WeightBinding *shared_expert_up = nullptr;
        const WeightBinding *shared_expert_down = nullptr;
        const WeightBinding *shared_expert_gate_inp = nullptr;
    };

    /**
     * @brief Model-facing weights for one MTP/next-token-prediction depth.
     *
     * MTP sidecars are not part of the main transformer layer loop. They still
     * contain a normal full-attention block plus projection/norm weights that
     * combine the draft-token embedding with the terminal hidden row.
     */
    struct MTPDepthWeights
    {
        int depth_index = 0;
        int source_layer_index = -1;
        bool nextn_block_layout = false;

        TensorBase *fc = nullptr;
        TensorBase *pre_fc_norm_hidden = nullptr;
        TensorBase *pre_fc_norm_embedding = nullptr;
        TensorBase *final_norm = nullptr;
        const WeightBinding *fc_binding = nullptr;
        LayerWeights fa_block;
    };

    struct MTPWeights
    {
        int depth = 0;
        bool use_dedicated_embeddings = false;
        std::vector<MTPDepthWeights> depths;

        bool empty() const { return depth <= 0 || depths.empty(); }
    };

    struct MTPForwardInput
    {
        const int *draft_token_ids = nullptr;
        TensorBase *terminal_hidden = nullptr;
        IKVCache *kv_cache = nullptr;
        const int *position_ids = nullptr;
        const std::vector<int> *sequence_lengths = nullptr;
        int batch_size = 1;
        int seq_len = 1;
        DeviceId device = DeviceId::cpu();
        BufferId terminal_hidden_buffer_id = BufferId::PREFIX_TERMINAL_HIDDEN;
        bool kv_cache_only = false;
    };

    struct MTPForwardOutput
    {
        TensorBase *logits = nullptr;
        TensorBase *hidden = nullptr;

        TensorBase *embedding = nullptr;
        TensorBase *norm_hidden = nullptr;
        TensorBase *norm_embedding = nullptr;
        TensorBase *concat = nullptr;
        TensorBase *projected = nullptr;

        TensorBase *q = nullptr;
        TensorBase *k = nullptr;
        TensorBase *v = nullptr;
        TensorBase *q_raw = nullptr;
        TensorBase *q_gate = nullptr;
        TensorBase *attn_output = nullptr;
        TensorBase *attn_proj = nullptr;
        TensorBase *gate = nullptr;
        TensorBase *up = nullptr;
        TensorBase *ffn_output = nullptr;

        TensorBase *moe_expert_indices = nullptr;
        TensorBase *moe_expert_weights = nullptr;
        TensorBase *moe_combined_output = nullptr;
        TensorBase *moe_shared_expert_output = nullptr;
        TensorBase *moe_gate_scratch = nullptr;
        TensorBase *moe_up_scratch = nullptr;
    };

    struct MTPDepthWeightBindings
    {
        int depth_index = 0;
        int source_layer_index = -1;
        bool nextn_block_layout = false;

        const WeightBinding *fc = nullptr;
        const WeightBinding *pre_fc_norm_hidden = nullptr;
        const WeightBinding *pre_fc_norm_embedding = nullptr;
        const WeightBinding *final_norm = nullptr;
        LayerWeightBindings fa_block;
    };

    struct MTPWeightBindings
    {
        int depth = 0;
        bool use_dedicated_embeddings = false;
        std::vector<MTPDepthWeightBindings> depths;

        bool empty() const { return depth <= 0 || depths.empty(); }
    };

    inline TensorBase *legacyTensor(const WeightBinding *binding)
    {
        return binding ? binding->tensor : nullptr;
    }

    inline LayerWeights toLegacyLayerWeights(const LayerWeightBindings &bindings)
    {
        LayerWeights weights;
        weights.wq = legacyTensor(bindings.wq);
        weights.wk = legacyTensor(bindings.wk);
        weights.wv = legacyTensor(bindings.wv);
        weights.wo = legacyTensor(bindings.wo);
        weights.attn_norm = legacyTensor(bindings.attn_norm);
        weights.q_bias = legacyTensor(bindings.q_bias);
        weights.k_bias = legacyTensor(bindings.k_bias);
        weights.v_bias = legacyTensor(bindings.v_bias);
        weights.q_norm = legacyTensor(bindings.q_norm);
        weights.k_norm = legacyTensor(bindings.k_norm);
        weights.attn_qkv = legacyTensor(bindings.attn_qkv);
        weights.attn_gate = legacyTensor(bindings.attn_gate);
        weights.ssm_alpha = legacyTensor(bindings.ssm_alpha);
        weights.ssm_beta = legacyTensor(bindings.ssm_beta);
        weights.ssm_conv1d = legacyTensor(bindings.ssm_conv1d);
        weights.ssm_dt_bias = legacyTensor(bindings.ssm_dt_bias);
        weights.ssm_a = legacyTensor(bindings.ssm_a);
        weights.ssm_norm = legacyTensor(bindings.ssm_norm);
        weights.ssm_out = legacyTensor(bindings.ssm_out);
        weights.gate_proj = legacyTensor(bindings.gate_proj);
        weights.up_proj = legacyTensor(bindings.up_proj);
        weights.down_proj = legacyTensor(bindings.down_proj);
        weights.ffn_norm = legacyTensor(bindings.ffn_norm);
        weights.moe_gate = legacyTensor(bindings.moe_gate);
        weights.moe_gate_exps = legacyTensor(bindings.moe_gate_exps);
        weights.moe_up_exps = legacyTensor(bindings.moe_up_exps);
        weights.moe_down_exps = legacyTensor(bindings.moe_down_exps);
        weights.shared_expert_gate = legacyTensor(bindings.shared_expert_gate);
        weights.shared_expert_up = legacyTensor(bindings.shared_expert_up);
        weights.shared_expert_down = legacyTensor(bindings.shared_expert_down);
        weights.shared_expert_gate_inp = legacyTensor(bindings.shared_expert_gate_inp);
        weights.wq_binding = bindings.wq;
        weights.wk_binding = bindings.wk;
        weights.wv_binding = bindings.wv;
        weights.wo_binding = bindings.wo;
        weights.gate_proj_binding = bindings.gate_proj;
        weights.up_proj_binding = bindings.up_proj;
        weights.down_proj_binding = bindings.down_proj;
        return weights;
    }

    inline MTPDepthWeights toLegacyMTPDepthWeights(const MTPDepthWeightBindings &bindings)
    {
        MTPDepthWeights weights;
        weights.depth_index = bindings.depth_index;
        weights.source_layer_index = bindings.source_layer_index;
        weights.nextn_block_layout = bindings.nextn_block_layout;
        weights.fc = legacyTensor(bindings.fc);
        weights.pre_fc_norm_hidden = legacyTensor(bindings.pre_fc_norm_hidden);
        weights.pre_fc_norm_embedding = legacyTensor(bindings.pre_fc_norm_embedding);
        weights.final_norm = legacyTensor(bindings.final_norm);
        weights.fc_binding = bindings.fc;
        weights.fa_block = toLegacyLayerWeights(bindings.fa_block);
        return weights;
    }

    inline MTPWeights toLegacyMTPWeights(const MTPWeightBindings &bindings)
    {
        MTPWeights weights;
        weights.depth = bindings.depth;
        weights.use_dedicated_embeddings = bindings.use_dedicated_embeddings;
        weights.depths.reserve(bindings.depths.size());
        for (const auto &depth : bindings.depths)
            weights.depths.push_back(toLegacyMTPDepthWeights(depth));
        return weights;
    }

    /**
     * @brief Model-level frozen weight bindings.
     */
    struct ModelWeightBindings
    {
        const WeightBinding *embedding_table = nullptr;
        const WeightBinding *final_norm = nullptr;
        const WeightBinding *lm_head = nullptr;
        MTPWeightBindings mtp;

        std::function<LayerWeightBindings(int layer_idx)> get_layer_weights;
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
        MTPWeights mtp;

        /// Accessor for per-layer weights
        std::function<LayerWeights(int layer_idx)> get_layer_weights;
    };

    inline ModelWeights toLegacyModelWeights(const ModelWeightBindings &bindings)
    {
        ModelWeights weights;
        weights.embedding_table = legacyTensor(bindings.embedding_table);
        weights.final_norm = legacyTensor(bindings.final_norm);
        weights.lm_head = legacyTensor(bindings.lm_head);
        weights.mtp = toLegacyMTPWeights(bindings.mtp);
        if (bindings.get_layer_weights)
        {
            weights.get_layer_weights = [get_layer_weights = bindings.get_layer_weights](int layer_idx)
            {
                return toLegacyLayerWeights(get_layer_weights(layer_idx));
            };
        }
        return weights;
    }

    inline const WeightBinding *optionalGlobalBinding(
        const FrozenModelWeightSet &weight_set,
        const std::string &canonical_name)
    {
        try
        {
            return &weight_set.global(canonical_name);
        }
        catch (const std::exception &)
        {
            return nullptr;
        }
    }

    inline std::vector<int> discoverNextNMTPLayers(const FrozenModelWeightSet &weight_set)
    {
        std::vector<int> layers;
        for (const auto &binding : weight_set.bindings())
        {
            const std::string &name = binding.identity.canonical_name;
            if (name.find(".nextn.eh_proj.weight") == std::string::npos)
                continue;
            const int layer = inferWeightLayer(name);
            if (layer >= 0)
                layers.push_back(layer);
        }

        std::sort(layers.begin(), layers.end());
        layers.erase(std::unique(layers.begin(), layers.end()), layers.end());
        return layers;
    }

    inline MTPDepthWeightBindings makeNextNDepthWeightBindings(
        const FrozenModelWeightSet &weight_set,
        int depth_index,
        int source_layer_index)
    {
        auto get = [&weight_set, source_layer_index](const std::string &suffix)
        {
            return weight_set.optionalLayer(source_layer_index, suffix);
        };

        MTPDepthWeightBindings depth;
        depth.depth_index = depth_index;
        depth.source_layer_index = source_layer_index;
        depth.nextn_block_layout = true;
        depth.fc = get("nextn.eh_proj.weight");
        depth.pre_fc_norm_hidden = get("nextn.hnorm.weight");
        depth.pre_fc_norm_embedding = get("nextn.enorm.weight");
        depth.final_norm = get("nextn.shared_head_norm.weight");

        depth.fa_block.wq = get("attn_q.weight");
        depth.fa_block.wk = get("attn_k.weight");
        depth.fa_block.wv = get("attn_v.weight");
        depth.fa_block.wo = get("attn_output.weight");
        depth.fa_block.attn_norm = get("attn_norm.weight");
        depth.fa_block.q_norm = get("attn_q_norm.weight");
        depth.fa_block.k_norm = get("attn_k_norm.weight");
        depth.fa_block.ffn_norm = get("post_attention_norm.weight");
        depth.fa_block.gate_proj = get("ffn_gate.weight");
        depth.fa_block.up_proj = get("ffn_up.weight");
        depth.fa_block.down_proj = get("ffn_down.weight");

        depth.fa_block.moe_gate = get("ffn_gate_inp.weight");
        depth.fa_block.moe_gate_exps = get("ffn_gate_exps.weight");
        depth.fa_block.moe_up_exps = get("ffn_up_exps.weight");
        depth.fa_block.moe_down_exps = get("ffn_down_exps.weight");
        depth.fa_block.shared_expert_gate = get("ffn_gate_shexp.weight");
        depth.fa_block.shared_expert_up = get("ffn_up_shexp.weight");
        depth.fa_block.shared_expert_down = get("ffn_down_shexp.weight");
        depth.fa_block.shared_expert_gate_inp = get("ffn_gate_inp_shexp.weight");
        return depth;
    }

    inline MTPWeightBindings makeMTPWeightBindings(const FrozenModelWeightSet &weight_set)
    {
        MTPWeightBindings bindings;

        auto nextn_layers = discoverNextNMTPLayers(weight_set);
        if (!nextn_layers.empty())
        {
            bindings.depth = static_cast<int>(nextn_layers.size());
            bindings.depths.reserve(nextn_layers.size());
            for (size_t i = 0; i < nextn_layers.size(); ++i)
            {
                bindings.depths.push_back(makeNextNDepthWeightBindings(
                    weight_set,
                    static_cast<int>(i),
                    nextn_layers[i]));
            }
            return bindings;
        }

        if (!optionalGlobalBinding(weight_set, "mtp.fc.weight"))
            return bindings;

        MTPDepthWeightBindings depth;
        depth.depth_index = 0;
        depth.fc = optionalGlobalBinding(weight_set, "mtp.fc.weight");
        depth.pre_fc_norm_hidden = optionalGlobalBinding(weight_set, "mtp.pre_fc_norm_hidden.weight");
        depth.pre_fc_norm_embedding = optionalGlobalBinding(weight_set, "mtp.pre_fc_norm_embedding.weight");
        depth.final_norm = optionalGlobalBinding(weight_set, "mtp.norm.weight");

        const std::string prefix = "mtp.layers.0.";
        depth.fa_block.attn_norm = optionalGlobalBinding(weight_set, prefix + "input_layernorm.weight");
        depth.fa_block.wq = optionalGlobalBinding(weight_set, prefix + "self_attn.q_proj.weight");
        depth.fa_block.wk = optionalGlobalBinding(weight_set, prefix + "self_attn.k_proj.weight");
        depth.fa_block.wv = optionalGlobalBinding(weight_set, prefix + "self_attn.v_proj.weight");
        depth.fa_block.wo = optionalGlobalBinding(weight_set, prefix + "self_attn.o_proj.weight");
        depth.fa_block.q_norm = optionalGlobalBinding(weight_set, prefix + "self_attn.q_norm.weight");
        depth.fa_block.k_norm = optionalGlobalBinding(weight_set, prefix + "self_attn.k_norm.weight");
        depth.fa_block.ffn_norm = optionalGlobalBinding(weight_set, prefix + "post_attention_layernorm.weight");
        depth.fa_block.gate_proj = optionalGlobalBinding(weight_set, prefix + "mlp.gate_proj.weight");
        depth.fa_block.up_proj = optionalGlobalBinding(weight_set, prefix + "mlp.up_proj.weight");
        depth.fa_block.down_proj = optionalGlobalBinding(weight_set, prefix + "mlp.down_proj.weight");

        bindings.depth = 1;
        bindings.depths.push_back(depth);
        return bindings;
    }

    inline ModelWeightBindings makeModelWeightBindings(const FrozenModelWeightSet &weight_set)
    {
        ModelWeightBindings bindings;
        bindings.embedding_table = optionalGlobalBinding(weight_set, "token_embd.weight");
        bindings.final_norm = optionalGlobalBinding(weight_set, "output_norm.weight");
        bindings.lm_head = optionalGlobalBinding(weight_set, "output.weight");
        bindings.mtp = makeMTPWeightBindings(weight_set);
        bindings.get_layer_weights = [&weight_set](int layer_idx)
        {
            LayerWeightBindings layer;
            auto get = [&weight_set, layer_idx](const std::string &suffix)
            {
                return weight_set.optionalLayer(layer_idx, suffix);
            };

            layer.wq = get("attn_q.weight");
            layer.wk = get("attn_k.weight");
            layer.wv = get("attn_v.weight");
            layer.wo = get("attn_output.weight");
            layer.attn_norm = get("attn_norm.weight");
            layer.q_bias = get("attn_q.bias");
            layer.k_bias = get("attn_k.bias");
            layer.v_bias = get("attn_v.bias");
            layer.q_norm = get("attn_q_norm.weight");
            layer.k_norm = get("attn_k_norm.weight");
            layer.attn_qkv = get("attn_qkv.weight");
            layer.attn_gate = get("attn_gate.weight");
            layer.ssm_alpha = get("ssm_alpha.weight");
            layer.ssm_beta = get("ssm_beta.weight");
            layer.ssm_conv1d = get("ssm_conv1d.weight");
            layer.ssm_dt_bias = get("ssm_dt.bias");
            layer.ssm_a = get("ssm_a");
            layer.ssm_norm = get("ssm_norm.weight");
            layer.ssm_out = get("ssm_out.weight");
            layer.gate_proj = get("ffn_gate.weight");
            layer.up_proj = get("ffn_up.weight");
            layer.down_proj = get("ffn_down.weight");
            layer.ffn_norm = get("ffn_norm.weight");
            if (!layer.ffn_norm)
                layer.ffn_norm = get("post_attention_norm.weight");
            layer.moe_gate = get("ffn_gate_inp.weight");
            layer.moe_gate_exps = get("ffn_gate_exps.weight");
            layer.moe_up_exps = get("ffn_up_exps.weight");
            layer.moe_down_exps = get("ffn_down_exps.weight");
            layer.shared_expert_gate = get("ffn_gate_shexp.weight");
            layer.shared_expert_up = get("ffn_up_shexp.weight");
            layer.shared_expert_down = get("ffn_down_shexp.weight");
            layer.shared_expert_gate_inp = get("ffn_gate_inp_shexp.weight");
            return layer;
        };
        return bindings;
    }

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

        // === Dynamic Extension Buffers ==========================================
        /// Model-specific buffers keyed by BufferId (GDN, MoE, etc.)
        /// Populated automatically from BufferArena — new models add
        /// BufferId entries and schema buffer specs; no infrastructure changes.
        std::unordered_map<BufferId, TensorBase *> extensions;

        /// Optional remap from the canonical transformer buffer id used by
        /// shared graph builders to the concrete buffer id for this graph
        /// fragment. MTP sidecars use this to keep attention/FFN contracts on
        /// MTP scratch buffers instead of the main graph scratch buffers.
        std::unordered_map<BufferId, BufferId> binding_ids;

        /// Look up a buffer by BufferId in the extensions map.
        /// Returns nullptr if the BufferId is not present.
        TensorBase *get(BufferId id) const
        {
            auto it = extensions.find(id);
            return it != extensions.end() ? it->second : nullptr;
        }

        /// Resolve the arena BufferId for a canonical activation role.
        BufferId idFor(BufferId id) const
        {
            auto it = binding_ids.find(id);
            return it != binding_ids.end() ? it->second : id;
        }
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
    using QwenStandardGraphConfig = GraphConfig;
    using Qwen2LayerWeights = LayerWeights;
    using Qwen2ModelWeights = ModelWeights;
    using Qwen2ActivationBuffers = ActivationBuffers;
    using Qwen2ModelBuffers = ModelBuffers;

} // namespace llaminar2
