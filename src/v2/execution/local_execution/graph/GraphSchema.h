/**
 * @file GraphSchema.h
 * @brief Declarative graph schema definitions for model architectures
 * @author David Sanftenberg
 * @date December 2025
 *
 * This file defines the declarative schema structures used to specify
 * model architectures without imperative logic. The schema describes
 * WHAT stages exist and HOW they connect, but not WHEN they execute
 * (that's determined by GraphResolver based on runtime config).
 *
 * Design Philosophy:
 * - Schemas are purely declarative (could be expressed as YAML)
 * - No runtime conditionals in schema definitions
 * - Annotations (tp_mode, requires_kv_cache) guide the resolver
 * - One schema per model architecture (Qwen2, DeepSeek, Llama3, etc.)
 *
 * Usage:
 *   GraphSchema schema = Qwen2Schema::create(model_config);
 *   ResolvedGraphSpec resolved = resolver.resolve(schema, runtime_config);
 *   ComputeGraph graph = builder.build(resolved);
 */

#pragma once

#include "../../../backends/DeviceId.h"
#include "../../StageShardingMode.h"
#include "../../../utils/Sampler.h"
#include "../../../utils/ToolCallTypes.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <optional>
#include <functional>

namespace llaminar2
{

    // Forward declarations
    class TensorBase;

    // =========================================================================
    // Enumerations
    // =========================================================================

    /**
     * @brief Stage types corresponding to ComputeStage implementations
     */
    enum class StageType
    {
        // Normalization
        RMSNorm,
        LayerNorm,

        // Projections
        GEMM,
        FusedQKVGEMM,
        FusedGateUpGEMM,

        // Attention
        RoPE,
        KVCacheAppend,
        KVCacheGather,
        AttentionCompute,

        // Activations
        SwiGLU,
        GELU,

        // Residuals
        ResidualAdd,

        // Embedding / Output
        Embedding,
        LMHead,

        // MPI Collectives
        Allreduce,
        Allgather,

        // MoE
        MoERouter,       ///< Router: hidden × gate → softmax top-k
        MoEFFN,          ///< Full MoE FFN: route + expert SwiGLU + combine
        MoECombine,      ///< Weighted combination of expert outputs
        MoESharedExpert, ///< Always-active shared expert SwiGLU FFN
        MoESharedGate,   ///< Sigmoid gating on shared expert output

        // Quantization
        Quantize,
        Dequantize,

        // Per-head normalization (Qwen3)
        QKNorm,

        // GDN (Gated Delta Network) stages
        AttentionOutputGate, ///< Sigmoid gate on attention output
        GatedRMSNorm,        ///< RMSNorm with learned multiplicative gate
        GDNProjection,       ///< 4 separate GEMMs: in_proj_qkv, in_proj_z, in_proj_a, in_proj_b
        ShortConv1d,         ///< Causal depthwise conv1d (kernel=4) + SiLU
        GDNRecurrence,       ///< Delta rule recurrence (chunk prefill, single-step decode)
    };

    /**
     * @brief Tensor parallelism modes for stages
     *
     * These annotations tell GraphResolver how to handle MPI distribution:
     * - None: No parallelism, full tensor on each rank
     * - ColumnParallel: Output dimension sharded (QKV, gate/up, LM head)
     * - RowParallel: Input dimension sharded (Wo, down proj)
     * - ExpertParallel: MoE expert distribution (future)
     */
    enum class TPMode
    {
        None,           ///< No tensor parallelism
        ColumnParallel, ///< Shard output dimension, allgather after
        RowParallel,    ///< Shard input dimension, allreduce after
        ExpertParallel, ///< MoE expert parallelism (future)
    };

    /**
     * @brief Buffer lifecycle semantics
     */
    enum class BufferSemantic
    {
        Input,   ///< Read-only input
        Output,  ///< Write-only output
        InOut,   ///< Modified in-place (e.g., residual)
        Scratch, ///< Temporary workspace
    };

    // =========================================================================
    // Schema Structures (Declarative - could be YAML)
    // =========================================================================

    /**
     * @brief Reference to a buffer or weight by name
     *
     * Examples:
     * - "hidden" → activation buffer
     * - "weights.wq" → layer weight
     * - "weights.embedding_table" → model weight
     * - "kv_cache" → KV cache reference
     *
     * For weight references (names starting with "weights."), the is_optional
     * flag indicates whether the weight MUST exist in the model file:
     * - is_optional=false (default): Missing weight is an error
     * - is_optional=true: Missing weight is acceptable (e.g., QKV biases)
     */
    struct TensorRef
    {
        std::string name;                                ///< Reference name
        BufferSemantic semantic = BufferSemantic::Input; ///< How the tensor is used
        bool is_optional = false;                        ///< True if weight may not exist in model

        TensorRef() = default;
        TensorRef(std::string n, BufferSemantic s = BufferSemantic::Input, bool optional = false)
            : name(std::move(n)), semantic(s), is_optional(optional) {}

        // Convenience constructor from string (required by default)
        TensorRef(const char *n) : name(n), semantic(BufferSemantic::Input), is_optional(false) {}

        /// Create an optional tensor reference
        static TensorRef optional(std::string n, BufferSemantic s = BufferSemantic::Input)
        {
            return TensorRef(std::move(n), s, true);
        }
    };

    /**
     * @brief Declarative specification of a compute stage
     *
     * This describes WHAT a stage does, not HOW to build it at runtime.
     * The GraphResolver uses annotations to decide whether to emit the stage
     * and whether to insert MPI collectives.
     */
    struct StageSpec
    {
        std::string name; ///< Unique name within graph (e.g., "layer0_attn_norm")
        StageType type;   ///< Stage type (RMSNorm, GEMM, etc.)

        std::vector<TensorRef> inputs;         ///< Input tensor references
        std::vector<TensorRef> outputs;        ///< Output tensor references
        std::vector<std::string> dependencies; ///< Explicit dependencies (usually inferred)

        // =================================================================
        // Annotations (guide resolver, not runtime conditionals)
        // =================================================================

        /// Tensor parallelism mode - resolver adds allreduce/allgather as needed
        TPMode tp_mode = TPMode::None;

        /// Stage requires KV cache to be present (e.g., KVCacheAppend)
        bool requires_kv_cache = false;

        /// Stage can be disabled by execution policy (debugEnv toggles)
        /// If true, resolver checks exec_policy before emitting
        bool is_optional = false;

        /// Execution policy key (e.g., "exec_gemm", "exec_rope")
        /// Only used if is_optional = true
        std::string exec_policy_key;

        /// Device override (invalid = use runtime default_device)
        DeviceId device; // Default-constructed = invalid, inherits from runtime config

        // =================================================================
        // Stage-specific parameters (declarative, not pointers)
        // =================================================================

        /// For RMSNorm: epsilon value
        std::optional<float> rms_norm_eps;

        /// For RMSNorm: subtract-one weight mode (gamma_effective = 1.0 + gamma_stored)
        std::optional<bool> subtract_one;

        /// For RoPE: theta base
        std::optional<float> rope_theta;

        /// For RoPE: partial rotary factor (fraction of head_dim that gets rotation).
        /// Overrides runtime default. 1.0 = full RoPE, 0.5 = half rotated.
        std::optional<float> partial_rotary_factor;

        /// For Attention: causal mask
        std::optional<bool> causal;

        /// For Attention: window size (-1 = full)
        std::optional<int> window_size;
    };

    /**
     * @brief Template for a single transformer layer
     *
     * Describes the sequence of stages in attention and FFN blocks.
     * The resolver instantiates this template for each layer.
     */
    struct LayerTemplate
    {
        std::vector<StageSpec> attention_stages; ///< Pre-norm, QKV, RoPE, attn, Wo, residual
        std::vector<StageSpec> ffn_stages;       ///< Pre-norm, gate/up, activation, down, residual
    };

    /**
     * @brief Buffer dimension specification
     *
     * Dimensions can reference config values via string formulas.
     * Examples: "seq_len", "d_model", "n_heads * head_dim"
     *
     * Supports precision-conditional dtype resolution via `dtype_overrides`:
     * when the resolver's activation_precision matches a key in the map,
     * that dtype is used instead of the default. Builder methods provide
     * a fluent API for specifying overrides and conditions.
     *
     * Supports conditional buffer inclusion via `conditions`: if non-empty,
     * the buffer is only resolved when activation_precision matches one of
     * the listed values. Empty conditions = always included.
     */
    struct BufferSpec
    {
        std::string name;               ///< Buffer name (e.g., "hidden", "Q")
        std::vector<std::string> shape; ///< Shape formulas ["seq_len", "d_model"]
        std::string dtype = "fp32";     ///< Data type ("fp32", "bf16", "q8_1")
        BufferSemantic semantic;        ///< Lifecycle semantic (Input/Output/Scratch/InOut)

        // =================================================================
        // Aliasing Configuration (for memory optimization)
        // =================================================================

        /// Alias group name - buffers in same group can share memory
        /// Empty string means no aliasing. Examples: "attn_scratch", "ffn_scratch"
        std::string alias_group;

        /// Priority within alias group (higher = allocated first, others alias to it)
        /// Used to ensure largest buffer in group gets physical allocation
        int alias_priority = 0;

        /// Human-readable description (for documentation/debugging)
        std::string description;

        // =================================================================
        // Precision-Conditional Type Resolution
        // =================================================================

        /// Map: activation precision name → dtype string.
        /// When the resolver's activation_precision matches a key here,
        /// the overridden dtype is used instead of the default.
        /// Example: {{"HybridQ16", "q16_1"}} means use Q16_1 in HybridQ16 mode.
        std::unordered_map<std::string, std::string> dtype_overrides;

        /// Conditions for buffer inclusion (activation precision names).
        /// If non-empty, this buffer is only resolved when the activation
        /// precision matches one of the listed values.
        /// Empty = always included.
        std::vector<std::string> conditions;

        // Convenience constructors
        BufferSpec() = default;
        BufferSpec(std::string n, std::vector<std::string> s, std::string dt, BufferSemantic sem)
            : name(std::move(n)), shape(std::move(s)), dtype(std::move(dt)), semantic(sem) {}
        BufferSpec(std::string n, std::vector<std::string> s, std::string dt, BufferSemantic sem,
                   std::string alias, int priority = 0, std::string desc = "")
            : name(std::move(n)), shape(std::move(s)), dtype(std::move(dt)), semantic(sem),
              alias_group(std::move(alias)), alias_priority(priority), description(std::move(desc)) {}

        // =================================================================
        // Builder Methods (fluent API for precision-conditional specs)
        // =================================================================

        /// Add a dtype override for a specific activation precision mode.
        BufferSpec &withDtypeOverride(const std::string &precision, const std::string &override_dtype)
        {
            dtype_overrides[precision] = override_dtype;
            return *this;
        }

        /// Add a condition — buffer only included when precision matches.
        BufferSpec &whenPrecision(const std::string &precision)
        {
            conditions.push_back(precision);
            return *this;
        }
    };

    /**
     * @brief Alias group specification for memory optimization
     *
     * Defines a group of buffers that can share physical memory because
     * their lifetimes don't overlap. DeviceGraphBufferManager uses this to
     * reduce activation memory footprint.
     *
     * Example: Q, K, V are consumed before gate, up are written,
     * so they can share memory in Qwen2.
     */
    struct AliasGroupSpec
    {
        std::string name;                      ///< Group identifier (e.g., "attn_scratch")
        std::vector<std::string> buffer_names; ///< Buffers in this group
        std::string description;               ///< Human-readable description

        /// Estimated memory savings as percentage (informational)
        float estimated_savings_percent = 0.0f;
    };

    // =========================================================================
    // Weight Sharding Configuration
    // =========================================================================

    /**
     * @brief Sharding mode for weight tensors in tensor parallelism
     *
     * This enum is architecture-independent - the pattern→mode mapping
     * is defined per-model in the schema.
     */
    enum class WeightShardingMode
    {
        Replicate,       ///< Full copy on each rank (norms, biases, embeddings)
        ColumnParallel,  ///< Split output dimension (rows of weight) - QKV, Gate/Up, LM head
        RowParallel,     ///< Split output dimension + allreduce - Wo projection
        InputParallel,   ///< Split input dimension (columns of weight) + allreduce - Down proj
        ExpertParallel   ///< Split expert dimension of 3D MoE tensors across ranks
    };

    /**
     * @brief Dimension type that a weight operates on for TP sharding
     *
     * This tells the weight loader WHICH dimension from TensorParallelConfig
     * to use when computing slice boundaries. The sharding mode tells HOW
     * to slice (rows vs columns), but this tells WHAT to slice by.
     */
    enum class WeightDimensionType
    {
        None,             ///< Not sharded (replicated weights)
        Heads,            ///< Attention heads (Q projection) - uses head_start/head_count
        KVHeads,          ///< KV attention heads (K/V projections) - uses kv_head_start/kv_head_count
        FFNHidden,        ///< FFN hidden dimension (Gate/Up/Down) - uses d_ff_start/d_ff_count
        Vocab,            ///< Vocabulary dimension (LM head) - uses vocab_start/vocab_count
        Bias1D,           ///< 1D bias that follows its weight's dimension type
        FusedQKVHeads,    ///< Fused QKV: 3 equal sub-blocks [Q|K|V] each split by heads
        ProportionalHeads ///< Proportional slice using head_start/head_count ratio against totalHeads.
                          ///< For weights whose output dim != n_heads (e.g. GDN ssm_alpha/beta with
                          ///< n_v_heads != n_heads). Computes: start = total_size * head_start / totalHeads,
                          ///< count = total_size * head_count / totalHeads.
    };

    /**
     * @brief Pattern-based weight sharding rule
     *
     * Used to map GGUF tensor names to sharding modes.
     * Patterns are matched using string contains (find != npos).
     */
    struct WeightShardingPattern
    {
        std::string pattern;           ///< Substring pattern to match (e.g., "attn_q.weight")
        WeightShardingMode mode;       ///< Sharding mode for matched weights
        WeightDimensionType dimension; ///< Which dimension to use for slicing
        std::string description;       ///< Human-readable explanation

        // Constructor for backward compatibility
        WeightShardingPattern(std::string p, WeightShardingMode m, std::string d)
            : pattern(std::move(p)), mode(m), dimension(WeightDimensionType::None), description(std::move(d)) {}

        // Full constructor with dimension type
        WeightShardingPattern(std::string p, WeightShardingMode m, WeightDimensionType dim, std::string d)
            : pattern(std::move(p)), mode(m), dimension(dim), description(std::move(d)) {}
    };

    /**
     * @brief Weight sharding configuration for a model architecture
     *
     * Defines how GGUF weight tensors should be distributed across
     * MPI ranks for tensor parallelism. Patterns are evaluated in order;
     * first match wins. Unmatched weights default to Replicate.
     */
    struct WeightShardingConfig
    {
        /// Patterns evaluated in order (first match wins)
        std::vector<WeightShardingPattern> patterns;

        /// Exact-match overrides for mode (checked before patterns)
        std::unordered_map<std::string, WeightShardingMode> exact_matches;

        /// Exact-match overrides for dimension type (checked before patterns)
        std::unordered_map<std::string, WeightDimensionType> exact_dimension_matches;

        /// Default mode for unmatched weights
        WeightShardingMode default_mode = WeightShardingMode::Replicate;

        /**
         * @brief Determine sharding mode for a weight tensor
         * @param name GGUF tensor name (e.g., "blk.0.attn_q.weight")
         * @return Sharding mode for this weight
         */
        WeightShardingMode getMode(const std::string &name) const
        {
            // Check exact matches first
            auto it = exact_matches.find(name);
            if (it != exact_matches.end())
            {
                return it->second;
            }

            // Check patterns in order
            for (const auto &rule : patterns)
            {
                if (name.find(rule.pattern) != std::string::npos)
                {
                    return rule.mode;
                }
            }

            return default_mode;
        }

        /**
         * @brief Determine dimension type for a weight tensor
         * @param name GGUF tensor name (e.g., "blk.0.attn_q.weight")
         * @return Dimension type for slicing this weight
         */
        WeightDimensionType getDimensionType(const std::string &name) const
        {
            // Check exact matches first
            auto it = exact_dimension_matches.find(name);
            if (it != exact_dimension_matches.end())
            {
                return it->second;
            }

            // Check patterns in order
            for (const auto &rule : patterns)
            {
                if (name.find(rule.pattern) != std::string::npos)
                {
                    return rule.dimension;
                }
            }

            return WeightDimensionType::None;
        }

        /**
         * @brief Get both mode and dimension for a weight in one lookup
         * @param name GGUF tensor name
         * @return Pair of (mode, dimension_type)
         */
        std::pair<WeightShardingMode, WeightDimensionType> getModeAndDimension(const std::string &name) const
        {
            // Check exact matches first
            auto mode_it = exact_matches.find(name);
            auto dim_it = exact_dimension_matches.find(name);
            if (mode_it != exact_matches.end())
            {
                WeightDimensionType dim = (dim_it != exact_dimension_matches.end())
                                              ? dim_it->second
                                              : WeightDimensionType::None;
                return {mode_it->second, dim};
            }

            // Check patterns in order
            for (const auto &rule : patterns)
            {
                if (name.find(rule.pattern) != std::string::npos)
                {
                    return {rule.mode, rule.dimension};
                }
            }

            return {default_mode, WeightDimensionType::None};
        }

        /**
         * @brief Check if a weight should be excluded from GEMM packing
         *
         * Non-GEMM weights (norms, biases, embeddings) need raw data retained.
         * GEMM weights can have raw data released after packing.
         *
         * @param name GGUF tensor name
         * @return true if this is NOT a GEMM weight
         */
        bool isNonGemmWeight(const std::string &name) const
        {
            // Norms are 1D.  Some sidecar layouts use compact names such as
            // hnorm/enorm instead of *_norm; they still contain norm.weight.
            if (name.find("norm.weight") != std::string::npos)
                return true;
            // Biases are 1D
            if (name.find(".bias") != std::string::npos)
                return true;
            // Embeddings are used directly
            if (name.find("token_embd") != std::string::npos)
                return true;
            // SSM/GDN convolution kernels are not GEMM weights
            if (name.find("ssm_conv1d") != std::string::npos)
                return true;
            // SSM/GDN per-head scalar parameters (e.g. ssm_a, ssm_dt) without .weight suffix
            if (name.find(".ssm_a") != std::string::npos && name.find(".weight") == std::string::npos)
                return true;
            // MoE 3D expert weight tensors (gate_exps, up_exps, down_exps)
            if (name.find("_exps.weight") != std::string::npos)
                return true;
            // MoE router gate and shared expert sigmoid gate
            if (name.find("ffn_gate_inp") != std::string::npos)
                return true;
            return false;
        }
    };

    /**
     * @brief Complete schema for a model architecture
     *
     * This is the top-level declarative specification.
     * One schema instance per architecture (Qwen2, DeepSeek, etc.)
     */
    struct GraphSchema
    {
        std::string name;    ///< Architecture name (e.g., "qwen2")
        std::string version; ///< Schema version

        // =================================================================
        // Model structure
        // =================================================================

        StageSpec embedding;                   ///< Embedding lookup stage
        LayerTemplate layer_template;          ///< Template for transformer layers (default/fallback)
        std::vector<StageSpec> lm_head_stages; ///< Final norm + LM head

        // =================================================================
        // Heterogeneous Layer Support (Phase B)
        // =================================================================

        /// Named layer templates for heterogeneous architectures.
        /// Models with mixed layer types (e.g., full attention + GDN layers) define
        /// multiple templates here and assign them to layers via `layer_template_names`.
        /// When empty, all layers use `layer_template` (backward compatible).
        std::unordered_map<std::string, LayerTemplate> named_templates;

        /// Per-layer template name assignment.
        /// Maps layer index → template name in `named_templates`.
        /// Size must equal n_layers when non-empty.
        /// When empty, all layers use the default `layer_template`.
        std::vector<std::string> layer_template_names;

        /// Get the LayerTemplate for a specific layer index.
        /// Falls back to `layer_template` when named_templates is empty or
        /// layer_template_names is not set.
        const LayerTemplate &getTemplateForLayer(int layer_idx) const
        {
            if (!layer_template_names.empty() && !named_templates.empty() &&
                layer_idx >= 0 &&
                layer_idx < static_cast<int>(layer_template_names.size()))
            {
                const auto &name = layer_template_names[layer_idx];
                auto it = named_templates.find(name);
                if (it != named_templates.end())
                {
                    return it->second;
                }
            }
            return layer_template;
        }

        // =================================================================
        // Buffer specifications
        // =================================================================

        std::vector<BufferSpec> layer_buffers; ///< Per-layer activation buffers
        std::vector<BufferSpec> model_buffers; ///< Model-level buffers (hidden, logits)

        // =================================================================
        // Aliasing specifications (for memory optimization)
        // =================================================================

        /// Alias groups for buffer memory sharing
        /// Buffers in the same group with non-overlapping lifetimes can share memory
        std::vector<AliasGroupSpec> alias_groups;

        // =================================================================
        // Config parameter declarations
        // =================================================================

        /// Parameter names that must be provided at resolution time
        std::vector<std::string> required_params; // ["n_layers", "d_model", "n_heads", ...]

        // =================================================================
        // Quantization Configuration
        // =================================================================

        /// KV cache scale for Q16_1 quantized attention (default: ±8.0 range)
        ///
        /// This is a "Conservative Fixed Scale" approach to avoid the "growing scale"
        /// problem where new tokens with larger activations would require re-quantizing
        /// the entire KV cache.
        ///
        /// The scale value represents the maximum absolute value that can be represented
        /// without clipping: [-kv_cache_scale, +kv_cache_scale].
        ///
        /// Default of 8.0 covers typical K/V activation ranges with headroom:
        /// - Post-RMSNorm activations typically fall in [-3, 3]
        /// - K/V projections can amplify slightly
        /// - 8.0 provides ~2× headroom over typical maximum values
        ///
        /// Architecture-specific overrides:
        /// - Models with known activation ranges can use tighter scales for precision
        /// - Use KV activation profiling tool (python/tools/profile_kv_activations.py)
        ///   to determine optimal scales for specific models
        float kv_cache_scale_k = 256.0f; ///< Fixed K Q16 scale (FP32 range ±scale_k)
        float kv_cache_scale_v = 32.0f;  ///< Fixed V Q16 scale (FP32 range ±scale_v)

        // =================================================================
        // TP Allreduce Precision Policy
        // =================================================================

        /// Default allreduce precision for layers NOT covered by fp32_layer_count.
        /// Valid values: "fp32", "fp16", "bf16"
        /// The global DebugEnv default ("fp32") serves as the ultimate fallback.
        std::string tp_allreduce_default_precision = "fp32";

        /// Number of initial transformer layers forced to FP32 allreduce.
        /// These early layers are the most sensitive to precision loss since
        /// errors compound through subsequent layers. Layers beyond this count
        /// use tp_allreduce_default_precision.
        int tp_allreduce_fp32_layer_count = 0;

        /// Layer indices that are ALWAYS forced to FP32 allreduce regardless
        /// of their position relative to fp32_layer_count. Used for hybrid
        /// architectures (e.g., Qwen3.5) where full-attention layers are more
        /// sensitive to allreduce precision loss than GDN layers.
        std::vector<int> tp_allreduce_fp32_forced_layers;
    };

    // =========================================================================
    // Resolved Structures (Output of GraphResolver)
    // =========================================================================

    /**
     * @brief Resolved stage with concrete tensor pointers
     *
     * After resolution, all conditionals are evaluated and tensor
     * references are resolved to actual pointers.
     */
    struct ResolvedStage
    {
        std::string name;
        StageType type;

        std::vector<TensorBase *> inputs;  ///< Resolved input tensors
        std::vector<TensorBase *> outputs; ///< Resolved output tensors
        std::vector<std::string> dependencies;

        DeviceId device = DeviceId::cpu();

        // =================================================================
        // Stage-specific resolved parameters
        // =================================================================

        /// Generic parameter storage (type-erased)
        /// Resolver populates these based on stage type
        std::unordered_map<std::string, float> float_params;
        std::unordered_map<std::string, int> int_params;
        std::unordered_map<std::string, bool> bool_params;
        std::unordered_map<std::string, std::string> string_params;
        std::unordered_map<std::string, TensorBase *> tensor_params;
        std::unordered_map<std::string, void *> opaque_params; // IMPIContext*, ICPUKVCache*, etc.
    };

    /**
     * @brief Fully resolved graph specification
     *
     * This is the output of GraphResolver - a flat list of stages
     * ready to be built into a ComputeGraph with no further conditionals.
     */
    struct ResolvedGraphSpec
    {
        std::string name;                  ///< Graph name (for debugging)
        std::vector<ResolvedStage> stages; ///< Ordered list of stages

        /// Statistics from resolution
        struct Stats
        {
            int stages_emitted = 0;     ///< Stages included in graph
            int stages_skipped = 0;     ///< Stages skipped (disabled)
            int allreduce_inserted = 0; ///< Allreduce stages added
            int allgather_inserted = 0; ///< Allgather stages added
        } stats;
    };

    // =========================================================================
    // Schema Factory Interface
    // =========================================================================

    /**
     * @brief Interface for architecture-specific schema factories
     *
     * Each model architecture (Qwen2, DeepSeek, Llama3) provides a factory
     * that creates its schema. This allows schema creation to use model
     * config values while keeping the schema itself declarative.
     */
    class ISchemaFactory
    {
    public:
        virtual ~ISchemaFactory() = default;

        /// Create schema for this architecture
        virtual GraphSchema createSchema() const = 0;

        /// Architecture name (e.g., "qwen2")
        virtual std::string architectureName() const = 0;

        /// Get weight sharding configuration for tensor parallelism
        virtual WeightShardingConfig getWeightShardingConfig() const = 0;

        /**
         * @brief Get stage output sharding configuration for tensor parallelism
         *
         * Returns a map from stage type strings (e.g., "Q_PROJECTION", "FFN_DOWN",
         * "LM_HEAD") to their SnapshotShardingMode. This drives snapshot reassembly
         * in RankOrchestrator and parity testing.
         *
         * Each model architecture declares how its stage outputs are distributed
         * across TP devices, replacing the hardcoded if-chain in getStageShardingMode().
         *
         * @return StageShardingConfig mapping stage type → SnapshotShardingMode
         */
        virtual StageShardingConfig getStageShardingConfig() const = 0;

        /**
         * @brief Check if a weight tensor is optional in this architecture
         *
         * This allows callers to distinguish between:
         * - "Weight doesn't exist because it's optional for this architecture"
         * - "Weight should exist but failed to load (error condition)"
         *
         * @param gguf_weight_name The GGUF tensor name (e.g., "blk.0.attn_q.bias")
         * @return true if the weight is optional, false if it's required
         */
        virtual bool isWeightOptional(const std::string &gguf_weight_name) const = 0;

        /**
         * @brief Get per-layer weight suffixes for this architecture
         *
         * Returns the list of weight name suffixes (without the "blk.N." prefix)
         * that should be checked for each transformer layer. The validator
         * constructs full names as "blk.<layer_idx>.<suffix>" and uses
         * isWeightOptional() to classify each as required or optional.
         *
         * @return Vector of weight suffixes (e.g., "attn_q.weight", "attn_q.bias")
         */
        virtual std::vector<std::string> layerWeightSuffixes() const = 0;

        /**
         * @brief Get recommended sampling parameters for this model architecture
         *
         * Returns model-specific recommended defaults for sampling parameters.
         * These serve as fallback values when the user doesn't specify parameters.
         * For example, Qwen3.5 thinking models need presence_penalty=1.5.
         *
         * Default implementation returns standard defaults (no penalties).
         * Override in model-specific factories to provide architecture-tuned values.
         *
         * @return SamplingParams with recommended defaults for this architecture
         */
        virtual SamplingParams getRecommendedSamplingParams() const
        {
            return SamplingParams{}; // Standard defaults
        }

        /**
         * @brief Get the stop-thinking prompt for thinking budget enforcement
         *
         * When the thinking token budget is exhausted, this string is tokenized
         * and injected token-by-token into the generation stream to gracefully
         * force the model out of thinking mode.
         *
         * Default: empty string (no stop-thinking support).
         * Override in model-specific factories for models that support thinking.
         *
         * @return Stop-thinking prompt string, or empty if not supported
         */
        virtual std::string getStopThinkingPrompt() const
        {
            return "";
        }

        /**
         * @brief Get the tool call output format for this model architecture
         *
         * Returns the format the model uses to emit tool calls in its raw text
         * output. Used by ChatCompletionHandler to parse structured tool_calls
         * from model output.
         *
         * Default: HERMES_2_PRO (covers Qwen 2.5, Qwen 3, Hermes, most models).
         * Override in model-specific factories to use a different format.
         *
         * @return ToolCallFormat enum value
         */
        virtual ToolCallFormat getToolCallFormat() const
        {
            return ToolCallFormat::HERMES_2_PRO;
        }
    };

} // namespace llaminar2
