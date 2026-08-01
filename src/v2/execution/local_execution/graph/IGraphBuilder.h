/**
 * @file IGraphBuilder.h
 * @brief Interface for declarative compute graph builders
 * @author David Sanftenberg
 * @date December 19, 2025
 *
 * This interface defines the contract for building compute graphs in a
 * declarative, stateless manner. Model-specific implementations (QwenStandardGraph,
 * Qwen3Graph, LlamaGraph, etc.) derive from this interface.
 *
 * Design Principles:
 * - Stateless: Graph builders should not hold mutable state
 * - Declarative: Methods return ComputeGraph objects, not execute them
 * - Testable: Interface enables mock implementations for unit testing
 */

#pragma once

#include "DeviceGraphExecutor.h"
#include "GraphSchema.h"
#include "GraphResolver.h"
#include "../../../models/GraphTypes.h"
#include "../../../backends/DeviceId.h"

#include <memory>
#include <vector>

namespace llaminar2
{
    // Forward declarations
    class TensorBase;
    class ICPUKVCache;
    struct LayerWeights;
    struct ActivationBuffers;

    // Forward declarations for IGraphBuilder interface
    struct PipelineConfig;
    class ILocalPPContext;
    class ILocalTPContext;
    class ITPContext;
    class PreparedWeightStore;
    class IModelContext;
    struct PrefixFingerprintMaterial;

    // =========================================================================
    // Generic Input/Output Structures
    // =========================================================================

    /**
     * @brief Declares how a forward invocation represents logical positions.
     *
     * Position geometry is part of graph policy, not something graph builders
     * may infer from whichever pointers happen to be non-null. ExplicitRows is
     * required for request-batched or otherwise non-contiguous positions and
     * must provide host or device rows. ContiguousOffset represents the single
     * sequence `[position_offset, position_offset + seq_len)` directly; GPU
     * stages publish that scalar on their execution stream and must not create
     * or upload a host row array.
     */
    enum class ForwardPositionPolicy : uint8_t
    {
        ExplicitRows,
        ContiguousOffset,
    };

    /**
     * @brief Semantic owner of one concrete forward-graph invocation.
     *
     * Shape and output selection do not identify graph purpose. In particular,
     * request-batched prefill and a grouped MTP verifier may both request
     * all-position logits. This role travels through the execution engine and
     * returns in @ref ForwardExecutionProvenance so consumers can validate the
     * producer they actually require.
     */
    enum class ForwardExecutionRole : uint8_t
    {
        MainInference,      ///< User-visible prefill or serial decode.
        GroupedMTPVerifier, ///< Main-model verification of speculative rows.
        MTPCondition,       ///< One live main-model condition row per request.
    };

    /**
     * @brief Mathematical phase selected for one concrete forward graph.
     *
     * Row count is not a phase discriminator. A short user prompt can have the
     * same M as a grouped MTP verifier, while request-batched decode can have
     * M greater than one. Carrying the phase beside @ref ForwardExecutionRole
     * makes weight layout, collective, recurrent-state, and workspace policy
     * declarative instead of inferring them from shape.
     */
    enum class ForwardExecutionPhase : uint8_t
    {
        Prefill, ///< Prompt ingestion using the configured prefill topology.
        Decode,  ///< Serial or grouped decode-equivalent execution.
    };

    /**
     * @brief Generic forward pass input
     *
     * Contains all fields needed for forward pass execution including
     * pipeline parallelism and variable-length batching support.
     */
    struct ForwardInput
    {
        const int *token_ids = nullptr;    ///< Token IDs [batch_size * seq_len]
        /**
         * @brief Optional device-resident INT32 token IDs.
         *
         * GPU verifier and sidecar graphs can use this to keep token IDs in
         * arena/workspace memory across graph capture/replay. When this pointer
         * is set, embedding stages treat it as the execution source of truth;
         * `token_ids` may still be populated as a stable host shadow for logs,
         * cache bookkeeping, and CPU-only helpers.
         */
        const void *token_ids_device = nullptr;
        const int *position_ids = nullptr; ///< Host position IDs [batch_size * seq_len]
        /**
         * @brief Optional device-resident INT32 position IDs.
         *
         * Phase 10 resident MTP continuation can publish the next logical
         * positions directly in device memory.  GPU RoPE stages should consume
         * this pointer without recording a host-to-device copy.  The host
         * `position_ids` pointer may still be set for CPU execution, logging,
         * and graph-key bookkeeping, but device execution must treat this
         * pointer as the source of truth when it is present.
         */
        const void *position_ids_device = nullptr;
        /**
         * @brief Materialize scalar serial-decode position from live GPU KV state.
         *
         * Ordinary GPU decode binds a stable arena position row into the
         * captured graph. Immediately before replay, the orchestrator snapshots
         * the canonical device KV count into that row on the graph stream. This
         * keeps RoPE, attention, and replicated-expert tie breaking on one
         * immutable device-owned logical position even when attention advances
         * the live cache count later in the same graph.
         */
        bool materialize_serial_decode_position_from_device_kv = false;
        /**
         * @brief Semantic representation of this invocation's positions.
         *
         * The default preserves the long-standing public contract that callers
         * provide explicit rows. Internal single-request prefill planning opts
         * into ContiguousOffset deliberately. A contiguous input must have
         * batch_size == 1 and leave both position pointers null so there is one
         * unambiguous source of truth.
         */
        ForwardPositionPolicy position_policy = ForwardPositionPolicy::ExplicitRows;
        ForwardExecutionRole execution_role =
            ForwardExecutionRole::MainInference; ///< Semantic owner of this invocation.
        ForwardExecutionPhase execution_phase =
            ForwardExecutionPhase::Prefill; ///< Typed math/topology phase; never inferred from M.
        int batch_size = 1;                ///< Number of sequences
        int seq_len = 0;                   ///< Sequence length per batch
        /**
         * @brief Absolute logical position for decode and legacy prefill callers.
         *
         * Decode paths use this as the current KV/RoPE position.  Prefill
         * callers should prefer @ref token_offset for the owned request range;
         * graph helpers may still treat this value as a compatibility fallback
         * when @ref token_offset is left at its default.
         */
        int position_offset = 0;
        int real_seq_len = 0;              ///< Real tokens in a bucketed prefill chunk (0 = seq_len)
        int bucket_seq_len = 0;            ///< Fixed bucket length for graph shape (0 = seq_len)
        /**
         * @brief Absolute token offset of the first real token in this prefill range.
         *
         * This is the first-class owner for prefill chunk/request boundaries.
         * Restored-prefix suffix prefill, padded graph-bucket replay, KV append
         * metadata, and position-id generation must all observe this same
         * value.  A value of zero is both the default and the valid offset for
         * the beginning of a request; callers with a non-zero logical cursor
         * must set it explicitly.
         */
        int token_offset = 0;
        int prefill_chunk_index = 0;       ///< Stable chunk ordinal for chunked graph-captured prefill.
        /**
         * @brief Select the one-shot graph that reconstructs prefix-owned GPU state.
         *
         * This is immutable graph policy for one invocation, not mutable host
         * placement data. When true, model stages prepend their captured
         * device-resident payload reconstruction transaction before any route,
         * attention, or decode consumer observes the restored state.
         */
        bool rehydrate_prefix_runtime_on_device = false;
        /**
         * @brief Exact producer stream for model-owned GPU state publication.
         *
         * Graph construction may perform one-time publication of immutable
         * descriptor tables whose addresses are captured by later graph nodes.
         * GPU callers must provide the explicit stream that owns those writes;
         * builders must never substitute the legacy null/default stream.
         * CPU graph construction leaves this null.
         */
        void *device_state_publication_stream = nullptr;
        DeviceId device = DeviceId::cpu(); ///< Target device
        IKVCache *kv_cache = nullptr;      ///< KV cache (optional)

        // ----- Pipeline Parallelism -----

        /// Per-device KV caches for Pipeline Parallelism (PP).
        /// When set (non-empty), each PP stage uses the KV cache for its device.
        /// Takes precedence over kv_cache when device is found in this map.
        const std::unordered_map<DeviceId, IKVCache *> *pp_kv_caches = nullptr;

        /// External hidden state input (for PP middle stages that don't have embedding).
        /// When set, embedding is skipped and this tensor is used as initial hidden state.
        TensorBase *external_hidden_state = nullptr;

        // ----- Variable-Length Batching -----

        /// Sequence lengths for variable-length batching (nullptr = all equal to seq_len).
        /// When set, enables proper batch-separating attention masks that
        /// prevent cross-sequence attention in batched execution.
        const std::vector<int> *sequence_lengths = nullptr;

        /**
         * @brief Device owner for per-request valid row counts in a padded batch.
         *
         * The serving boundary uploads `sequence_lengths` once before GPU graph
         * execution. Stateful graph stages such as short convolution and GDN
         * recurrence consume this stable arena pointer directly, so graph
         * capture never records an H2D copy and padded rows cannot advance live
         * request state. CPU graphs leave this pointer null and read the host
         * vector above.
         */
        const int32_t *sequence_lengths_device = nullptr;

        /// Batched input (alternative to token_ids)
        struct Batch
        {
            const int *tokens;
            int len;
            int offset;
        };
        const Batch *batches = nullptr;
        int num_batches = 0;

        // ----- Helpers -----

        /// Get the KV cache for a specific device (PP) or the default (non-PP)
        IKVCache *getKVCacheForDevice(const DeviceId &dev) const
        {
            if (pp_kv_caches && !pp_kv_caches->empty())
            {
                auto it = pp_kv_caches->find(dev);
                if (it != pp_kv_caches->end())
                    return it->second;
            }
            return kv_cache;
        }

        virtual ~ForwardInput() = default;
    };

    /**
     * @brief Device-ordering provenance for one concrete forward execution.
     *
     * A tensor pointer identifies storage, but it does not identify the GPU
     * stream that most recently wrote that storage. Downstream device-only
     * consumers therefore need this execution-scoped record in addition to the
     * output tensor itself. Carrying the record in ForwardOutput makes the
     * producer-to-consumer handoff explicit and prevents a later graph launch
     * from overwriting an engine-global "last execution" lookup before the
     * original consumer has queued its dependency.
     *
     * CPU execution is synchronous and publishes a valid record with a null
     * stream. A successful GPU execution must publish a valid record with a
     * non-null, Llaminar-owned stream.
     */
    struct ForwardExecutionProvenance
    {
        bool valid = false;                    ///< True only after successful graph execution.
        DeviceId device = DeviceId::invalid(); ///< Device that owns the produced output.
        void *stream = nullptr;                ///< Exact GPU producer stream; null for CPU.
        ForwardExecutionRole execution_role =
            ForwardExecutionRole::MainInference; ///< Typed purpose copied from ForwardInput.
        bool is_decode = false;                ///< Whether decode semantics selected this graph.
        bool all_position_logits = false;      ///< Whether this invocation produced all-position logits.
        int graph_seq_len = 0;                 ///< Captured graph rows per request, including prefill bucketing.
        int graph_batch_size = 0;              ///< Captured graph request count.
    };

    /**
     * @brief Generic forward pass output and its concrete execution provenance.
     */
    struct ForwardOutput
    {
        TensorBase *logits = nullptr; ///< Output logits [batch_size * seq_len, vocab_size]
        TensorBase *hidden = nullptr; ///< Optional: final hidden states
        ForwardExecutionProvenance execution; ///< Ordering owner for this invocation's outputs.

        virtual ~ForwardOutput() = default;
    };

    /**
     * @brief Context for layer-level graph building
     */
    struct LayerContext
    {
        int layer_idx = 0;                 ///< Layer index
        int seq_len = 0;                   ///< Sequence length
        int batch_size = 1;                ///< Batch size (number of sequences)
        DeviceId device = DeviceId::cpu(); ///< Target device
        /**
         * @brief Exact producer stream for model-owned GPU state publication.
         *
         * GPU layer-graph callers must provide the stream that owns any
         * graph-build descriptor writes. CPU callers explicitly leave this
         * null because their graph construction is synchronous.
         */
        void *device_state_publication_stream = nullptr;
        const int *position_ids = nullptr; ///< Host position IDs for RoPE
        const void *position_ids_device = nullptr; ///< Device INT32 position IDs for GPU RoPE
        IKVCache *kv_cache = nullptr;      ///< KV cache
        /// Sequence lengths for variable-length batching (nullptr = all equal)
        const std::vector<int> *sequence_lengths = nullptr;
        /// Device-owned counterpart used by GPU request-batched recurrent stages.
        const int32_t *sequence_lengths_device = nullptr;
    };

    // =========================================================================
    // IGraphBuilder Interface
    // =========================================================================

    /**
     * @brief Interface for declarative compute graph builders
     *
     * This interface defines the contract that all model graph builders must
     * implement. It enables:
     * - Polymorphic graph building across different model architectures
     * - Mock implementations for unit testing
     * - Clear separation between graph building and execution
     *
     * Example usage:
     * @code
     * std::unique_ptr<IGraphBuilder> builder = std::make_unique<QwenStandardGraph>(...);
     * ForwardInput input{...};
     * ForwardOutput output{...};
     * ComputeGraph graph = builder->buildForwardGraph(input, output);
     * executor.execute(graph, ctx);
     * @endcode
     */
    class IGraphBuilder
    {
    public:
        virtual ~IGraphBuilder() = default;

        // =====================================================================
        // Core Graph Building Methods
        // =====================================================================

        /**
         * @brief Build complete forward graph
         *
         * Constructs a ComputeGraph representing the full forward pass:
         * embedding → transformer layers → output projection (LM head).
         *
         * @param input Forward pass input parameters
         * @param output Forward pass output tensors (logits, optional hidden)
         * @return Complete forward compute graph
         */
        virtual ComputeGraph buildForwardGraph(
            const ForwardInput &input,
            ForwardOutput &output) = 0;

        /**
         * @brief Build single transformer layer graph
         *
         * Constructs a ComputeGraph for one transformer layer (attention + FFN).
         *
         * @param ctx Layer context with index, seq_len, device, etc.
         * @return Single layer compute graph
         */
        virtual ComputeGraph buildLayerGraph(const LayerContext &ctx) = 0;

        virtual ComputeGraph buildMTPGraph(
            int depth_idx,
            const MTPDepthWeightBindings &bindings,
            const MTPForwardInput &input,
            MTPForwardOutput &output)
        {
            (void)depth_idx;
            (void)bindings;
            (void)input;
            (void)output;
            return {};
        }

        // =====================================================================
        // Optional Methods (with default implementations)
        // =====================================================================

        /**
         * @brief Get the number of transformer layers
         *
         * @return Number of layers in the model
         */
        virtual int numLayers() const { return 0; }

        /**
         * @brief Get model hidden dimension
         *
         * @return Hidden dimension (d_model)
         */
        virtual int hiddenDim() const { return 0; }

        /**
         * @brief Check if the builder is properly initialized
         *
         * @return true if weights and buffers are set
         */
        virtual bool isInitialized() const { return false; }

        // =====================================================================
        // Configuration Access
        // =====================================================================

        /// Get model configuration (dimensions, layer count, etc.)
        virtual const GraphConfig &config() const = 0;

        /// Get the architecture name (e.g. "qwen2", "qwen3", "llama")
        virtual std::string architectureName() const { return "unknown"; }

        /// Append model-owned prefix-cache fingerprint material, such as MoE placement.
        virtual void appendPrefixCacheFingerprintMaterial(PrefixFingerprintMaterial &material) const
        {
            (void)material;
        }

        /**
         * @brief Capture model-owned runtime state needed to continue from a prefix block.
         *
         * Prefix payloads already store KV, hybrid recurrence, MTP state, and terminal
         * logits/hidden rows. Dynamic model features that affect subsequent execution
         * but are not part of those tensors, such as graph-facing MoE placement banks,
         * can serialize same-runner runtime state here.
         */
        virtual bool capturePrefixCacheRuntimeState(std::vector<uint8_t> &state, void *stream)
        {
            (void)stream;
            state.clear();
            return true;
        }

        /// Restore state captured by capturePrefixCacheRuntimeState().
        virtual bool restorePrefixCacheRuntimeState(const std::vector<uint8_t> &state, void *stream)
        {
            (void)stream;
            return state.empty();
        }

        /**
         * @brief Whether restored logical state still needs GPU payload reconstruction.
         *
         * A true result means restorePrefixCacheRuntimeState() imported a
         * pointer-free placement record whose rolling payload replicas must be
         * recreated by the next dedicated captured forward graph. The query is
         * deliberately separate from restore success: success means the archive
         * was valid and its device plan was published, not that payload transport
         * has already executed.
         */
        virtual bool prefixCacheRuntimeStateRequiresDeviceRehydration() const
        {
            return false;
        }

        /**
         * @brief Retire the one-shot device rehydration contract after execution.
         *
         * Called only after the dedicated graph has completed successfully. A
         * model that advertised pending work must treat an unexpected call or
         * incomplete device plan as fatal rather than silently retaining stale
         * placement.
         */
        virtual void completePrefixCacheRuntimeStateDeviceRehydration() {}

        // =====================================================================
        // Weight / Buffer Management
        // =====================================================================

        /// Set model weights
        virtual void setWeights(const ModelWeights &weights) = 0;

        /// Set frozen model-weight bindings for graph-build validation and diagnostics.
        virtual void setWeightBindings(const ModelWeightBindings &bindings) { (void)bindings; }

        /// Set optional full dense bindings used by replicated-dense decode graphs.
        virtual void setDecodeReplicatedDenseWeightBindings(const ModelWeightBindings &bindings) { (void)bindings; }

        /// Set activation buffers (for manual buffer management)
        virtual void setBuffers(const ModelBuffers &buffers) = 0;

        /// Set buffer arena for arena-managed allocation
        virtual void setArena(BufferArena *arena) { (void)arena; }

        /// Set prepared weight store for kernel lifecycle management (Phase 10)
        virtual void setPreparedWeightStore(PreparedWeightStore *store) { (void)store; }

        /// Enable or disable all-position LM-head logits for speculative verification.
        virtual bool setComputeAllPositionLogits(bool enabled)
        {
            (void)enabled;
            return false;
        }

        /**
         * @brief Select the grouped speculative-verifier graph policy.
         *
         * This is deliberately independent of all-position logits because MTP
         * prompt prefill also requests logits for multiple rows while still
         * producing the terminal hidden state consumed by the first draft
         * sidecar.
         */
        virtual bool setGroupedMTPVerifier(bool enabled)
        {
            (void)enabled;
            return false;
        }

        /**
         * @brief Select the terminal device-owned verifier outcome transaction.
         *
         * This policy changes graph topology and therefore participates in the
         * forward graph cache signature.  Implementations must reject a mode
         * they cannot build; treating an unsupported mode as Disabled would
         * silently return to post-graph host orchestration.
         */
        virtual bool setMTPVerifierOutcomeGraphMode(
            MTPVerifierOutcomeGraphMode mode)
        {
            return mode == MTPVerifierOutcomeGraphMode::Disabled;
        }

        /**
         * @brief Install persistent arena bindings for terminal MTP reduction.
         *
         * Bindings are lifetime resources, not request metadata.  They may be
         * replaced only while graph caches are inactive, normally during arena
         * initialization.
         */
        virtual bool setMTPVerifierOutcomeGraphBinding(
            const MTPVerifierOutcomeGraphBinding &binding)
        {
            (void)binding;
            return false;
        }

        /**
         * @brief Select the grouped live MTP condition graph policy.
         *
         * Unlike all-position verification, this mode advances live recurrent
         * state and requests one terminal-logit row for each request. Builders
         * must not implement it by enabling speculative verifier state capture.
         */
        virtual bool setLiveMTPRequestBatchCondition(bool enabled)
        {
            (void)enabled;
            return false;
        }

        /// Enable compact row-indexed all-position verifier logits.
        virtual bool setComputeRowIndexedAllPositionLogits(bool enabled, int row_count)
        {
            (void)enabled;
            (void)row_count;
            return false;
        }

        /**
         * @brief Set the source rows used by compact row-indexed verifier logits.
         *
         * Empty means "use the default leading rows". A non-empty vector must
         * contain exactly the row count configured by
         * setComputeRowIndexedAllPositionLogits(). Device runners install this
         * from the MTP verifier metadata plan before building the verifier graph,
         * so CPU and GPU graph paths consume the same logical row plan even when
         * GPU stages later read the indices from workspace memory during replay.
         */
        virtual bool setRowIndexedAllPositionLogitRows(const std::vector<int> &selected_rows)
        {
            return selected_rows.empty();
        }

        /// Set model context for registry-created builders in tests/dependency injection.
        virtual void setModelContext(std::shared_ptr<IModelContext> model_ctx) { (void)model_ctx; }

        /// Get current activation buffers
        virtual const ModelBuffers &buffers() const = 0;

        // =====================================================================
        // Schema / Resolver
        // =====================================================================

        /// Get the declarative schema for this architecture
        virtual GraphSchema getSchema() const { return {}; }

        /// Get resolver config for buffer allocation
        virtual GraphResolverConfig getResolverConfig(int seq_len) const
        {
            (void)seq_len;
            return {};
        }

        // =====================================================================
        // Pipeline / Parallelism Configuration
        // =====================================================================

        /// Set pipeline configuration for PP graph building
        virtual void setPipelineConfig(std::shared_ptr<PipelineConfig> config)
        {
            (void)config;
        }

        /// Register a PP context for inter-stage transfers
        virtual void setPPContext(int from_stage, int to_stage, ILocalPPContext *pp_ctx)
        {
            (void)from_stage;
            (void)to_stage;
            (void)pp_ctx;
        }

        /// Register a TP context for a named domain
        virtual void setTPContext(const std::string &domain_name, ITPContext *tp_ctx)
        {
            (void)domain_name;
            (void)tp_ctx;
        }

        // =====================================================================
        // Snapshot
        // =====================================================================

        /// Set snapshot callback for debugging
        virtual void setSnapshotCallback(StageSnapshotCallback callback)
        {
            (void)callback;
        }

        // =====================================================================
        // PP Graph Building Variants
        // =====================================================================

        /// Build forward graph for a PP subset of layers
        virtual ComputeGraph buildPartialForwardGraph(
            const ForwardInput &input,
            ForwardOutput &output,
            int first_layer,
            int last_layer,
            bool has_embedding,
            bool has_lm_head)
        {
            (void)input;
            (void)output;
            (void)first_layer;
            (void)last_layer;
            (void)has_embedding;
            (void)has_lm_head;
            return {};
        }

        /// Build unified PP+TP pipeline graph
        virtual ComputeGraph buildUnifiedPipelineGraph(
            const ForwardInput &input,
            ForwardOutput &output)
        {
            (void)input;
            (void)output;
            return {};
        }

        // =====================================================================
        // Full Forward & Layer-Level Graph Building
        // =====================================================================

        /// Build the complete forward graph (embedding → all layers → LM head)
        virtual ComputeGraph buildFullForwardGraph(
            const ForwardInput &input,
            ForwardOutput &output)
        {
            (void)input;
            (void)output;
            return {};
        }

        /// Build attention block sub-graph for a single layer
        virtual ComputeGraph buildAttentionGraph(
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
            const int32_t *sequence_lengths_device = nullptr)
        {
            (void)layer;
            (void)buffers;
            (void)layer_idx;
            (void)seq_len;
            (void)batch_size;
            (void)kv_cache;
            (void)position_ids;
            (void)position_ids_device;
            (void)device;
            (void)sequence_lengths;
            (void)sequence_lengths_device;
            return {};
        }

        /**
         * @brief Build the FFN subgraph for one transformer layer.
         *
         * GPU graph construction may publish model-owned side state, such as
         * the active MoE runtime-table bank. The caller must therefore supply
         * the exact stream that owns those writes. Implementations must reject
         * a null stream for GPU devices before allocating, copying, or
         * publishing device state. CPU construction is synchronous and names
         * that fact explicitly by passing `nullptr`.
         *
         * @param layer Layer weights consumed by the FFN stages.
         * @param buffers Activation buffers bound into the generated graph.
         * @param layer_idx Model layer index.
         * @param seq_len Sequence length represented by this graph.
         * @param batch_size Number of request rows represented by this graph.
         * @param device Device on which the graph will execute.
         * @param device_state_publication_stream Exact producer stream for GPU
         *        graph-build publications; `nullptr` is legal only for CPU.
         * @param sequence_lengths_device Optional device-resident real-length
         *        array used by padded grouped execution.
         * @param absolute_position_ids_device Optional device-resident INT32
         *        absolute position row shared with RoPE. Grouped resident
         *        assignment policies that promise M-invariance must require it.
         */
        virtual ComputeGraph buildFFNGraph(
            const LayerWeights &layer,
            ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            int batch_size,
            DeviceId device,
            void *device_state_publication_stream,
            const int32_t *sequence_lengths_device = nullptr,
            const int32_t *absolute_position_ids_device = nullptr)
        {
            (void)layer;
            (void)buffers;
            (void)layer_idx;
            (void)seq_len;
            (void)batch_size;
            (void)device;
            (void)device_state_publication_stream;
            (void)sequence_lengths_device;
            (void)absolute_position_ids_device;
            return {};
        }

        /**
         * @brief Build one atomic device-side MoE maintenance transaction.
         *
         * The returned graph owns histogram collection, controller planning,
         * command publication, immutable expert-payload packing, NCCL/RCCL
         * transport, and runtime-table apply. Those phases must remain in one
         * captured transaction so no host readback or graph selection can sit
         * between planning and publication. Ordinary decode graphs contain
         * only their cheap ready-wave consumers. Non-MoE builders and
         * unsupported placements return an empty graph.
         *
         * @param device Participant device whose symmetric maintenance graph
         *        should be constructed.
         * @return A complete per-device maintenance graph, or an empty graph
         *         when the model has no device-side MoE maintenance contract.
         */
        virtual ComputeGraph buildDeviceMoERebalanceMaintenanceGraph(
            DeviceId device)
        {
            (void)device;
            return {};
        }

        // =====================================================================
        // State Management
        // =====================================================================

        /**
         * @brief Reset model-internal request state at an ordinary request boundary.
         *
         * Called by typed inference-state reset when a completed request gives
         * ownership of model-local runtime state back to the graph builder.
         * Implementations may preserve graph-replay-compatible baseline state
         * here, such as descriptor tables captured by warm decode graphs. GPU
         * implementations must enqueue every device mutation on
         * @p execution_stream; the request-reset transaction publishes its
         * readiness event on that same stream only after this method returns.
         * A GPU builder must reject a null stream instead of creating an
         * private stream or synchronizing the device.
         *
         * @param execution_stream Explicit request-reset stream for GPU
         *        mutation, or nullptr for a CPU-only builder.
         * Prefix restore uses resetPrefixCacheRuntimeStateWithoutSnapshot()
         * instead when the cache block has no model-runtime payload.
         */
        virtual void resetState(void *execution_stream = nullptr)
        {
            (void)execution_stream;
        }

        /**
         * @brief Reset model runtime state for prefix restore without a payload.
         *
         * A prefix-cache restore is a replacement of live request state, not an
         * ordinary request rollover.  When the matched prefix block does not
         * carry model-owned runtime bytes, the graph builder must install an
         * explicit "no prefix-owned model runtime" state before suffix prefill.
         * This is intentionally distinct from resetState(): resetState() may
         * keep graph-replay-compatible baseline state, while this boundary must
         * not resurrect dynamic MoE placement, LLEP transfer slots, histograms,
         * or other request-local state from a previous request. GPU
         * implementations obey the same explicit-stream contract as
         * resetState().
         *
         * @param execution_stream Explicit request-reset stream for GPU
         *        mutation, or nullptr for a CPU-only builder.
         */
        virtual void resetPrefixCacheRuntimeStateWithoutSnapshot(
            void *execution_stream = nullptr)
        {
            resetState(execution_stream);
        }

        // =====================================================================
        // Utility Methods
        // =====================================================================

        /**
         * @brief Build position IDs array for RoPE
         *
         * Static utility function that can be used by any graph builder.
         *
         * @param seq_len Sequence length
         * @param batch_size Number of sequences
         * @param offset Position offset (for KV cache continuation)
         * @return Vector of position IDs [batch_size * seq_len]
         */
        static std::vector<int> buildPositionIds(int seq_len, int batch_size, int offset)
        {
            std::vector<int> pos_ids(batch_size * seq_len);
            for (int b = 0; b < batch_size; ++b)
            {
                for (int s = 0; s < seq_len; ++s)
                {
                    pos_ids[b * seq_len + s] = offset + s;
                }
            }
            return pos_ids;
        }
    };

    // =========================================================================
    // MockGraphBuilder for Testing
    // =========================================================================

    /**
     * @brief Mock graph builder for unit testing
     *
     * Provides a controllable mock implementation of IGraphBuilder that:
     * - Records method calls for verification
     * - Returns configurable mock graphs
     * - Enables testing of DeviceGraphOrchestrator without real model weights
     *
     * Example usage:
     * @code
     * auto mock = std::make_shared<MockGraphBuilder>();
     * mock->setMockForwardGraph(some_graph);
     * mock->setNumLayers(24);
     *
     * DeviceGraphOrchestrator orchestrator(mock);
     * orchestrator.executeForward(input, output);
     *
     * EXPECT_EQ(mock->buildForwardGraphCallCount(), 1);
     * @endcode
     */
    class MockGraphBuilder : public IGraphBuilder
    {
    public:
        MockGraphBuilder() = default;
        ~MockGraphBuilder() override = default;

        // =====================================================================
        // IGraphBuilder Implementation
        // =====================================================================

        ComputeGraph buildForwardGraph(
            const ForwardInput &input,
            ForwardOutput &output) override
        {
            ++build_forward_calls_;
            last_forward_input_ = &input;
            last_forward_output_ = &output;

            if (forward_graph_factory_)
            {
                return forward_graph_factory_(input, output);
            }
            // Return empty graph by default (ComputeGraph is move-only)
            return ComputeGraph{};
        }

        ComputeGraph buildLayerGraph(const LayerContext &ctx) override
        {
            ++build_layer_calls_;
            last_layer_ctx_ = ctx;

            if (layer_graph_factory_)
            {
                return layer_graph_factory_(ctx);
            }

            // Check for layer-specific factory
            if (ctx.layer_idx < static_cast<int>(layer_graph_factories_.size()) &&
                layer_graph_factories_[ctx.layer_idx])
            {
                return layer_graph_factories_[ctx.layer_idx](ctx);
            }

            // Return empty graph by default
            return ComputeGraph{};
        }

        int numLayers() const override { return num_layers_; }
        int hiddenDim() const override { return hidden_dim_; }
        bool isInitialized() const override { return initialized_; }

        // New IGraphBuilder overrides
        const GraphConfig &config() const override { return config_; }
        void setWeights(const ModelWeights &weights) override { weights_ = weights; }
        void setBuffers(const ModelBuffers &buffers) override { buffers_ = buffers; }
        const ModelBuffers &buffers() const override { return buffers_; }

        // =====================================================================
        // Mock Configuration
        // =====================================================================

        /**
         * @brief Set factory function for forward graph
         *
         * The factory will be called each time buildForwardGraph is invoked,
         * allowing dynamic graph creation based on input parameters.
         *
         * @param factory Function that creates a ComputeGraph from ForwardInput
         */
        using ForwardGraphFactory = std::function<ComputeGraph(const ForwardInput &, ForwardOutput &)>;
        void setForwardGraphFactory(ForwardGraphFactory factory)
        {
            forward_graph_factory_ = std::move(factory);
        }

        /**
         * @brief Set factory function for layer graph (default for all layers)
         *
         * @param factory Function that creates a ComputeGraph from LayerContext
         */
        using LayerGraphFactory = std::function<ComputeGraph(const LayerContext &)>;
        void setLayerGraphFactory(LayerGraphFactory factory)
        {
            layer_graph_factory_ = std::move(factory);
        }

        /**
         * @brief Set factory function for a specific layer
         *
         * @param layer_idx Layer index
         * @param factory Function that creates a ComputeGraph for this layer
         */
        void setLayerGraphFactory(int layer_idx, LayerGraphFactory factory)
        {
            if (layer_idx >= static_cast<int>(layer_graph_factories_.size()))
            {
                layer_graph_factories_.resize(layer_idx + 1);
            }
            layer_graph_factories_[layer_idx] = std::move(factory);
        }

        /// Configure mock model properties
        void setNumLayers(int n)
        {
            num_layers_ = n;
            config_.n_layers = n;
        }
        void setHiddenDim(int d)
        {
            hidden_dim_ = d;
            config_.d_model = d;
        }
        void setInitialized(bool init) { initialized_ = init; }
        void setConfig(const GraphConfig &cfg)
        {
            config_ = cfg;
            num_layers_ = cfg.n_layers;
            hidden_dim_ = cfg.d_model;
        }

        /// Access stored weights (for test assertions after setWeights())
        const ModelWeights &storedWeights() const { return weights_; }

        // =====================================================================
        // Call Tracking (for test assertions)
        // =====================================================================

        /// Get number of buildForwardGraph calls
        int buildForwardGraphCallCount() const { return build_forward_calls_; }

        /// Get number of buildLayerGraph calls
        int buildLayerGraphCallCount() const { return build_layer_calls_; }

        /// Get last forward input (for inspection)
        const ForwardInput *lastForwardInput() const { return last_forward_input_; }

        /// Get last forward output (for inspection)
        const ForwardOutput *lastForwardOutput() const { return last_forward_output_; }

        /// Get last layer context (for inspection)
        const LayerContext &lastLayerContext() const { return last_layer_ctx_; }

        /// Reset all call counters
        void resetCallCounts()
        {
            build_forward_calls_ = 0;
            build_layer_calls_ = 0;
            last_forward_input_ = nullptr;
            last_forward_output_ = nullptr;
            last_layer_ctx_ = {};
        }

        // =====================================================================
        // Schema / Resolver Support (for buffer management testing)
        // =====================================================================

        /// Override getSchema() to return a custom schema
        GraphSchema getSchema() const override
        {
            if (schema_)
                return *schema_;
            return {};
        }

        /// Override getResolverConfig() to return a custom resolver config
        GraphResolverConfig getResolverConfig(int seq_len) const override
        {
            if (resolver_config_factory_)
                return resolver_config_factory_(seq_len);
            return {};
        }

        /// Set the schema returned by getSchema()
        void setSchema(GraphSchema schema)
        {
            schema_ = std::make_unique<GraphSchema>(std::move(schema));
        }

        /// Set a factory for resolver config (receives seq_len)
        using ResolverConfigFactory = std::function<GraphResolverConfig(int)>;
        void setResolverConfigFactory(ResolverConfigFactory factory)
        {
            resolver_config_factory_ = std::move(factory);
        }

    private:
        // Factory functions for dynamic graph creation
        ForwardGraphFactory forward_graph_factory_;
        LayerGraphFactory layer_graph_factory_;
        std::vector<LayerGraphFactory> layer_graph_factories_;

        // Schema / Resolver support
        std::unique_ptr<GraphSchema> schema_;
        ResolverConfigFactory resolver_config_factory_;

        // Model properties
        GraphConfig config_{};
        ModelWeights weights_{};
        ModelBuffers buffers_{};
        int num_layers_ = 24;
        int hidden_dim_ = 896;
        bool initialized_ = true;

        // Call tracking
        int build_forward_calls_ = 0;
        int build_layer_calls_ = 0;
        const ForwardInput *last_forward_input_ = nullptr;
        const ForwardOutput *last_forward_output_ = nullptr;
        LayerContext last_layer_ctx_;
    };

} // namespace llaminar2
