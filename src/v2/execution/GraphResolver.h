/**
 * @file GraphResolver.h
 * @brief Resolves declarative graph schemas into concrete graph specifications
 * @author David Sanftenberg
 * @date December 2025
 *
 * GraphResolver is the "brain" that evaluates all runtime conditionals
 * and transforms a declarative GraphSchema into a ResolvedGraphSpec.
 *
 * KEY DESIGN PRINCIPLE:
 *   All imperative logic (MPI checks, debugEnv toggles, attention mode
 *   detection) lives HERE, not in individual graph builders. This means:
 *   - New models only need to define schemas (declarative)
 *   - All models share the same TP/MPI logic (no copy-paste)
 *   - Logic can be unit-tested in isolation
 *
 * Resolution Process:
 *   1. Evaluate is_optional stages against ExecutionPolicy
 *   2. Resolve tensor references to actual TensorBase pointers
 *   3. Evaluate tp_mode annotations and insert MPI collectives
 *   4. Detect attention mode based on runtime state
 *   5. Expand layer templates for n_layers
 *   6. Return flat list of resolved stages
 *
 * Usage:
 *   GraphResolver resolver;
 *   GraphResolverConfig config{.world_size = 2, .exec_policy = debugEnv().execution};
 *   ResolvedGraphSpec spec = resolver.resolve(schema, config);
 *   ComputeGraph graph = GraphBuilder::build(spec);
 */

#pragma once

#include "GraphSchema.h"
#include "BufferRole.h"
#include "../backends/DeviceId.h"
#include "../utils/MPIContext.h"
#include <memory>
#include <unordered_map>
#include <functional>

namespace llaminar2
{

    // Forward declarations
    class TensorBase;
    class IKVCache;
    class ComputeGraph;

    // =========================================================================
    // Runtime Configuration (Input to Resolver)
    // =========================================================================

    /**
     * @brief Execution policy flags (mirrors debugEnv().execution)
     *
     * These flags allow selective disabling of stages for debugging.
     */
    struct ExecutionPolicyFlags
    {
        bool exec_embedding = true;
        bool exec_rmsnorm = true;
        bool exec_gemm = true;
        bool exec_rope = true;
        bool exec_attention = true;
        bool exec_swiglu = true;
        bool exec_residual = true;
        bool exec_lm_head = true;

        /// Check if a stage type should be executed
        bool shouldExecute(const std::string &policy_key) const;

        /// Create from debugEnv
        static ExecutionPolicyFlags fromDebugEnv();
    };

    /**
     * @brief Information about a weight tensor's sharding
     */
    struct ShardingInfo
    {
        bool is_sharded = false;    ///< Whether weight is sharded
        TPMode mode = TPMode::None; ///< How it's sharded
        int shard_dim = -1;         ///< Which dimension is sharded
        int local_size = 0;         ///< Size on this rank
        int global_size = 0;        ///< Total size across ranks
    };

    /**
     * @brief Runtime configuration for graph resolution
     *
     * This captures all runtime state needed to resolve a schema.
     * Collected once at the start of forward pass.
     */
    struct GraphResolverConfig
    {
        // =================================================================
        // MPI / Tensor Parallelism
        // =================================================================

        int world_size = 1;                  ///< Number of MPI ranks
        int rank = 0;                        ///< This rank's index
        const MPIContext *mpi_ctx = nullptr; ///< MPI context (required for TP)

        /// Weight sharding information, keyed by weight name
        std::unordered_map<std::string, ShardingInfo> weight_sharding;

        // =================================================================
        // Execution Policy
        // =================================================================

        ExecutionPolicyFlags exec_policy; ///< Stage enable/disable flags

        // =================================================================
        // Sequence / Batch Configuration
        // =================================================================

        int batch_size = 1;      ///< Number of sequences in batch
        int seq_len = 0;         ///< Sequence length (tokens)
        int position_offset = 0; ///< KV cache position offset

        // =================================================================
        // KV Cache State (for attention mode detection)
        // =================================================================

        bool has_kv_cache = false; ///< KV cache present
        int cached_tokens = 0;     ///< Tokens already in cache (layer 0)

        // =================================================================
        // Device Configuration
        // =================================================================

        DeviceId default_device = DeviceId::cpu(); ///< Default device

        // =================================================================
        // Model Configuration (from schema params)
        // =================================================================

        int n_layers = 0;
        int d_model = 0;
        int n_heads = 0;
        int n_kv_heads = 0;
        int head_dim = 0;
        int d_ff = 0;
        int vocab_size = 0;
        float rms_norm_eps = 1e-6f;
        float rope_theta = 10000.0f;

        // =================================================================
        // Q16 KV Cache VNNI Safety (Phase 5.4)
        // =================================================================

        /// Fixed scale for Q16 KV cache quantization (FP32 range: ±kv_cache_scale).
        /// Used by KVCacheAppendStage for VNNI-safe quantization.
        float kv_cache_scale = 256.0f; ///< Fixed Q16 scale. Must cover Q projection max (~130 for Qwen2)

        // TP-adjusted dimensions
        int local_n_heads = 0;    ///< Heads on this rank
        int local_n_kv_heads = 0; ///< KV heads on this rank
        int local_d_ff = 0;       ///< FFN dim on this rank
        int local_vocab = 0;      ///< Vocab size on this rank
    };

    /**
     * @brief Tensor lookup context for resolution
     *
     * Maps string references to actual TensorBase pointers.
     */
    struct TensorContext
    {
        /// Model weights by name (e.g., "embedding_table", "final_norm")
        std::unordered_map<std::string, TensorBase *> model_weights;

        /// Per-layer weights accessor
        /// layer_weights[layer_idx]["wq"], ["wk"], etc.
        std::function<TensorBase *(int layer_idx, const std::string &name)> get_layer_weight;

        /// Activation buffers by name (e.g., "hidden", "Q", "K")
        std::unordered_map<std::string, TensorBase *> buffers;

        /// KV cache (optional)
        IKVCache *kv_cache = nullptr;

        /// Position IDs pointer
        const int *position_ids = nullptr;

        /// Token IDs pointer
        const int *token_ids = nullptr;

        /// Resolve a tensor reference to pointer
        TensorBase *resolve(const TensorRef &ref, int layer_idx = -1) const;
    };

    // =========================================================================
    // GraphResolver
    // =========================================================================

    /**
     * @brief Resolves declarative schemas into executable graph specs
     *
     * This class encapsulates ALL imperative logic for graph building:
     * - MPI collective insertion
     * - Execution policy filtering
     * - Attention mode detection
     * - Layer template expansion
     */
    class GraphResolver
    {
    public:
        GraphResolver() = default;
        ~GraphResolver() = default;

        // Non-copyable (stateless, but prevent accidental copies)
        GraphResolver(const GraphResolver &) = delete;
        GraphResolver &operator=(const GraphResolver &) = delete;

        // =================================================================
        // Main Resolution API
        // =================================================================

        /**
         * @brief Resolve a complete graph schema
         *
         * @param schema Declarative graph schema
         * @param runtime Runtime configuration (MPI, batch, exec policy)
         * @param tensors Tensor lookup context (weights, buffers)
         * @return Resolved graph specification ready for building
         */
        ResolvedGraphSpec resolve(
            const GraphSchema &schema,
            const GraphResolverConfig &runtime,
            const TensorContext &tensors) const;

        /**
         * @brief Resolve a single layer's stages
         *
         * Useful for layer-level graph building.
         *
         * @param layer_template Layer stage template
         * @param layer_idx Layer index
         * @param runtime Runtime configuration
         * @param tensors Tensor lookup context
         * @return Resolved stages for this layer
         */
        std::vector<ResolvedStage> resolveLayer(
            const LayerTemplate &layer_template,
            int layer_idx,
            const GraphResolverConfig &runtime,
            const TensorContext &tensors) const;

        /**
         * @brief Resolve a single stage specification
         *
         * @param spec Stage specification
         * @param layer_idx Layer index (-1 for non-layer stages)
         * @param runtime Runtime configuration
         * @param tensors Tensor lookup context
         * @return Resolved stage (or nullopt if stage should be skipped)
         */
        std::optional<ResolvedStage> resolveStage(
            const StageSpec &spec,
            int layer_idx,
            const GraphResolverConfig &runtime,
            const TensorContext &tensors) const;

    private:
        // =================================================================
        // Resolution Helpers
        // =================================================================

        /**
         * @brief Check if a stage should be emitted based on conditions
         *
         * Evaluates:
         * - is_optional + exec_policy_key
         * - requires_kv_cache
         */
        bool shouldEmitStage(
            const StageSpec &spec,
            const GraphResolverConfig &runtime) const;

        /**
         * @brief Resolve TP annotations and determine if MPI collective needed
         *
         * Returns the collective stage to insert (if any).
         */
        std::optional<ResolvedStage> resolveTPCollective(
            const StageSpec &spec,
            const ResolvedStage &resolved,
            const GraphResolverConfig &runtime) const;

        /**
         * @brief Detect attention mode from runtime state
         */
        int detectAttentionMode(const GraphResolverConfig &runtime) const;

        /**
         * @brief Expand a stage name with layer prefix
         *
         * "attn_norm" + layer 5 → "layer5_attn_norm"
         */
        std::string expandStageName(
            const std::string &base_name,
            int layer_idx) const;

        /**
         * @brief Resolve tensor reference to pointer
         */
        TensorBase *resolveTensorRef(
            const TensorRef &ref,
            int layer_idx,
            const TensorContext &tensors) const;

        /**
         * @brief Populate stage-specific parameters
         */
        void populateStageParams(
            ResolvedStage &resolved,
            const StageSpec &spec,
            const GraphResolverConfig &runtime,
            const TensorContext &tensors,
            int layer_idx) const;
    };

    // =========================================================================
    // BufferAllocator (Generic buffer shape resolution)
    // =========================================================================

    /**
     * @brief Resolved buffer specification with concrete dimensions
     *
     * This is the output of BufferAllocator - a BufferSpec with
     * string shape formulas resolved to actual size_t dimensions.
     */
    struct ResolvedBufferSpec
    {
        std::string name;          ///< Buffer name
        std::vector<size_t> shape; ///< Resolved shape (concrete dimensions)
        std::string dtype;         ///< Data type ("fp32", "bf16", etc.)
        BufferSemantic semantic;   ///< Lifecycle semantic
        std::string alias_group;   ///< Alias group (from schema)
        int alias_priority;        ///< Alias priority (from schema)
        std::string description;   ///< Description (from schema)
        DeviceId device;           ///< Target device

        /// Calculate total bytes for this buffer
        size_t totalBytes() const;
    };

    /**
     * @brief Generic buffer allocator that resolves schema BufferSpecs
     *
     * This class evaluates string shape formulas from the schema using
     * the runtime configuration, producing concrete buffer dimensions.
     *
     * Shape formula examples:
     * - "d_model" → config.d_model
     * - "seq_len" → config.seq_len
     * - "local_qkv_dim" → config.local_n_heads * config.head_dim
     * - "local_kv_dim" → config.local_n_kv_heads * config.head_dim
     * - "local_d_ff" → config.local_d_ff
     * - "vocab_size" → config.vocab_size
     * - "local_vocab" → config.local_vocab
     *
     * This is GENERIC - works for any model schema, not just Qwen2.
     * All model-specific shape formulas are defined in the schema.
     */
    class BufferAllocator
    {
    public:
        /**
         * @brief Resolve a single BufferSpec to concrete dimensions
         *
         * @param spec Buffer specification from schema
         * @param config Runtime configuration with dimension values
         * @return Resolved buffer spec with concrete shape
         */
        static ResolvedBufferSpec resolve(
            const BufferSpec &spec,
            const GraphResolverConfig &config);

        /**
         * @brief Resolve all buffers from a schema
         *
         * @param schema Graph schema with buffer definitions
         * @param config Runtime configuration
         * @return Vector of resolved buffer specs
         */
        static std::vector<ResolvedBufferSpec> resolveAll(
            const GraphSchema &schema,
            const GraphResolverConfig &config);

        /**
         * @brief Evaluate a single shape dimension formula
         *
         * @param formula Shape formula string (e.g., "d_model", "local_qkv_dim")
         * @param config Runtime configuration
         * @return Concrete dimension value
         */
        static size_t evaluateFormula(
            const std::string &formula,
            const GraphResolverConfig &config);

        /**
         * @brief Get aliasing groups with resolved buffer sizes
         *
         * Returns alias groups from schema with buffer sizes calculated,
         * useful for memory optimization planning.
         *
         * @param schema Graph schema
         * @param config Runtime configuration
         * @return Alias groups with resolved buffer information
         */
        static std::vector<AliasGroupSpec> resolveAliasGroups(
            const GraphSchema &schema,
            const GraphResolverConfig &config);

        /**
         * @brief Estimate memory savings from aliasing
         *
         * @param schema Graph schema
         * @param config Runtime configuration
         * @return Pair of (original_bytes, optimized_bytes)
         */
        static std::pair<size_t, size_t> estimateMemorySavings(
            const GraphSchema &schema,
            const GraphResolverConfig &config);

        /**
         * @brief Convert resolved buffers to StageBufferRequirements
         *
         * This bridges the declarative schema world with the existing
         * buffer allocation infrastructure.
         *
         * @param resolved Resolved buffer specifications
         * @return StageBufferRequirements suitable for GraphBufferManager
         */
        static StageBufferRequirements toBufferRequirements(
            const std::vector<ResolvedBufferSpec> &resolved);

        /**
         * @brief Resolve layer buffers from schema and convert to requirements
         *
         * Convenience method combining resolveAll + filter for layer buffers.
         *
         * @param schema Graph schema
         * @param config Runtime configuration
         * @return StageBufferRequirements for layer buffers
         */
        static StageBufferRequirements resolveLayerBuffers(
            const GraphSchema &schema,
            const GraphResolverConfig &config);

        /**
         * @brief Resolve model buffers from schema and convert to requirements
         *
         * Convenience method combining resolveAll + filter for model buffers.
         *
         * @param schema Graph schema
         * @param config Runtime configuration
         * @return StageBufferRequirements for model buffers
         */
        static StageBufferRequirements resolveModelBuffers(
            const GraphSchema &schema,
            const GraphResolverConfig &config);
    };

    // =========================================================================
    // GraphBuilder (Trivial - no conditionals)
    // =========================================================================

    // Forward declarations
    class ComputeGraph;
    class IComputeStage;

    /**
     * @brief Builds ComputeGraph from resolved specification
     *
     * This is intentionally trivial - just loops and calls addNode().
     * All the interesting logic is in GraphResolver.
     */
    class GraphBuilder
    {
    public:
        /**
         * @brief Build ComputeGraph from resolved spec
         *
         * @param spec Resolved graph specification
         * @return Constructed ComputeGraph ready for execution
         */
        static ComputeGraph build(const ResolvedGraphSpec &spec);

    private:
        /**
         * @brief Create a compute stage from resolved stage spec
         *
         * Uses ComputeStageFactory to create the appropriate stage type.
         */
        static std::unique_ptr<IComputeStage> createStage(const ResolvedStage &stage);
    };

} // namespace llaminar2
