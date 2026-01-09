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

#include "../backends/DeviceId.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <optional>
#include <functional>

namespace llaminar2
{

    // Forward declarations
    class CPUTensorBase;
    using TensorBase = CPUTensorBase; // Backward compatibility alias

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

        // MoE (future)
        MoERouter,
        MoEFFN,

        // Quantization
        Quantize,
        Dequantize,
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
     */
    struct TensorRef
    {
        std::string name;                                ///< Reference name
        BufferSemantic semantic = BufferSemantic::Input; ///< How the tensor is used

        TensorRef() = default;
        TensorRef(std::string n, BufferSemantic s = BufferSemantic::Input)
            : name(std::move(n)), semantic(s) {}

        // Convenience constructor from string
        TensorRef(const char *n) : name(n), semantic(BufferSemantic::Input) {}
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

        /// Device override (CPU = use default)
        DeviceId device = DeviceId::cpu();

        // =================================================================
        // Stage-specific parameters (declarative, not pointers)
        // =================================================================

        /// For RMSNorm: epsilon value
        std::optional<float> rms_norm_eps;

        /// For RoPE: theta base
        std::optional<float> rope_theta;

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

        // Convenience constructors
        BufferSpec() = default;
        BufferSpec(std::string n, std::vector<std::string> s, std::string dt, BufferSemantic sem)
            : name(std::move(n)), shape(std::move(s)), dtype(std::move(dt)), semantic(sem) {}
        BufferSpec(std::string n, std::vector<std::string> s, std::string dt, BufferSemantic sem,
                   std::string alias, int priority = 0, std::string desc = "")
            : name(std::move(n)), shape(std::move(s)), dtype(std::move(dt)), semantic(sem),
              alias_group(std::move(alias)), alias_priority(priority), description(std::move(desc)) {}
    };

    /**
     * @brief Alias group specification for memory optimization
     *
     * Defines a group of buffers that can share physical memory because
     * their lifetimes don't overlap. GraphBufferManager uses this to
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
        Replicate,      ///< Full copy on each rank (norms, biases, embeddings)
        ColumnParallel, ///< Split output dimension (rows of weight) - QKV, Gate/Up, LM head
        RowParallel,    ///< Split output dimension + allreduce - Wo projection
        InputParallel   ///< Split input dimension (columns of weight) + allreduce - Down proj
    };

    /**
     * @brief Pattern-based weight sharding rule
     *
     * Used to map GGUF tensor names to sharding modes.
     * Patterns are matched using string contains (find != npos).
     */
    struct WeightShardingPattern
    {
        std::string pattern;     ///< Substring pattern to match (e.g., "attn_q.weight")
        WeightShardingMode mode; ///< Sharding mode for matched weights
        std::string description; ///< Human-readable explanation
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

        /// Exact-match overrides (checked before patterns)
        std::unordered_map<std::string, WeightShardingMode> exact_matches;

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
            // Norms are 1D
            if (name.find("_norm.weight") != std::string::npos)
                return true;
            // Biases are 1D
            if (name.find(".bias") != std::string::npos)
                return true;
            // Embeddings are used directly
            if (name.find("token_embd") != std::string::npos)
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
        LayerTemplate layer_template;          ///< Template for transformer layers
        std::vector<StageSpec> lm_head_stages; ///< Final norm + LM head

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
        float kv_cache_scale = 256.0f; ///< Fixed Q16 scale. Must handle Q projection max_abs (~130 for Qwen2)
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
        std::unordered_map<std::string, void *> opaque_params; // MPIContext*, ICPUKVCache*, etc.
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
    };

} // namespace llaminar2
