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
#include "../../../execution/moe/MoEOverlayDeviceControllerRuntimeBinding.h"
#include "../../../execution/moe/MoEOverlayNodeLocalDeviceControllerFabric.h"

#include <cstdint>
#include <memory>
#include <optional>
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
    class ICPUCurrentBatchLLEPPhysicalExecutor;
    class IModelContext;
    class IBackend;
    class DeviceMoEOverlayEpochArena;
    struct PrefixFingerprintMaterial;
    struct DeviceMoELayerRuntime;

    /**
     * @brief Mutation policy for one MTP sidecar forward transaction.
     *
     * The model graph that owns the actual sidecar wiring declares this
     * policy. Runtime orchestration must not infer the contract from a model
     * name, backend, or topology: none of those prove that every sidecar stage
     * is bound to MTP-owned scratch and shifted KV state.
     */
    enum class MTPSidecarMainStatePolicy
    {
        Undeclared,
        Preserved,
    };

    /**
     * @brief Authoritative source for the first accepted shifted-MTP row.
     *
     * Dense sidecars can retain their first shifted row when it is exactly the
     * accepted target row. MoE sidecars intentionally defer that publication
     * to the target verifier so routed expert state and shifted KV are committed
     * from one accepted row boundary.
     */
    enum class MTPShiftedRowPublicationPolicy
    {
        Undeclared,
        ReuseSidecarRow,
        TargetVerifierAuthoritative,
    };

    /**
     * @brief Declarative ownership contract for a model's MTP sidecar graph.
     *
     * A graph may advertise this contract only when its sidecar activations,
     * cache, recurrent state, router metadata, and bookkeeping have explicit
     * owners. The orchestrator uses it to decide whether a verifier-base
     * restore is required; an undeclared contract cannot enable the shortcut.
     */
    struct MTPSidecarStateContract
    {
        MTPSidecarMainStatePolicy main_state =
            MTPSidecarMainStatePolicy::Undeclared;
        MTPShiftedRowPublicationPolicy shifted_row =
            MTPShiftedRowPublicationPolicy::Undeclared;

        /** @return Whether sidecar execution leaves every main-model state surface unchanged. */
        [[nodiscard]] constexpr bool preservesMainState() const noexcept
        {
            return main_state == MTPSidecarMainStatePolicy::Preserved;
        }

        /** @return Whether the first accepted shifted row may remain from sidecar execution. */
        [[nodiscard]] constexpr bool reusesSidecarShiftedRow() const noexcept
        {
            return shifted_row ==
                   MTPShiftedRowPublicationPolicy::ReuseSidecarRow;
        }
    };

    /**
     * @brief Device-owned source for request-final current-batch LLEP evidence.
     *
     * A model graph that performs current-batch LLEP keeps one sticky movement
     * and non-owner-assignment marker in every layer runtime.  The orchestrator
     * borrows this contiguous table only to enqueue a backend reduction on an
     * explicitly ordered stream.  It must never inspect or mirror the table on
     * the host during production execution.  The graph also publishes the
     * immutable byte extent of one complete expert payload so the terminal
     * reduction can turn each apply-only movement marker into a conservative
     * physical-byte lower bound without consulting a rolling maintenance wave.
     */
    struct DeviceMoECurrentBatchLLEPEvidenceSource
    {
        const DeviceMoELayerRuntime *runtime_layers_device = nullptr;
        int layer_count = 0;
        uint64_t expert_payload_slot_bytes = 0;

        /** @return true when the complete contiguous runtime table is bound. */
        constexpr bool valid() const noexcept
        {
            return runtime_layers_device != nullptr && layer_count > 0 &&
                   expert_payload_slot_bytes > 0;
        }
    };

    /**
     * @brief Stable device-side admission state for one ExpertOverlay request family.
     *
     * A complete durable main/MTP transaction must select one placement epoch before
     * its first graph and retain that reader through its final state
     * publication.  The graph builder owns the model-lifetime arena because it
     * also owns every durable runtime table that embeds the ticket address;
     * transient current-batch LLEP tables are outside this publication domain. The
     * orchestrator owns admission and release because only it can see the
     * complete transaction boundary.
     *
     * The request slot is immutable graph topology.  A runner that admits more
     * concurrent transactions than the arena declares must fail admission
     * rather than alias one live ticket between requests.
     */
    struct DeviceMoEOverlayEpochExecutionBinding
    {
        std::shared_ptr<DeviceMoEOverlayEpochArena> arena; ///< Stable model-lifetime storage.
        std::uint32_t request_slot = 0u;                   ///< Captured ticket/status slot.
        /** Device-authenticated grace-period receipt lane, when device-owned. */
        std::optional<MoEOverlayDeviceControllerParticipantBinding>
            retirement_readiness_controller;

        /** @return Whether a model supplied an epoch arena for this participant. */
        [[nodiscard]] explicit operator bool() const noexcept
        {
            return arena != nullptr;
        }
    };

    /**
     * @brief Immutable-address contract for graph-integrated shifted MTP prefill.
     *
     * A normal prefill transaction produces both target logits and the hidden
     * rows needed to seed the depth-zero MTP KV cache. GPU execution lowers
     * those two results into one captured graph: a row-preparation stage derives
     * shifted rows from canonical device KV counters, then one bucket-wide
     * KV-only MTP subgraph consumes the prepared rows. Every pointer in this
     * structure is model-lifetime storage. Request-varying values remain in
     * those device allocations and are never reconstructed by the host.
     */
    struct ShiftedMTPPrefillGraphBinding
    {
        /**
         * @brief Lifecycle in which this immutable pointer set may be used.
         *
         * RuntimeExecution is the ordinary captured production transaction and
         * carries the finalized workspace generation in `capture_identity`.
         * WorkspaceFamilyDeclaration exists only while setup materializes the
         * exact graph topology used to size the first workspace generation.
         * ForwardExecutionEngine rejects the latter, making a declaration graph
         * structurally incapable of becoming an inference fallback.
         */
        enum class Purpose
        {
            RuntimeExecution,
            WorkspaceFamilyDeclaration,
        };

        IKVCache *kv_cache = nullptr; ///< Depth-zero shifted MTP KV cache.
        TensorBase *terminal_hidden_archive = nullptr; ///< One persistent terminal hidden row per request.
        const int32_t *request_token_ids_device = nullptr; ///< Stable admitted request-token bank.
        const int32_t *request_position_ids_device = nullptr; ///< Stable admitted absolute-position bank.
        const int32_t *request_segment_lengths_device = nullptr; ///< Real rows in this shifted-MTP segment, distinct from main attention history.
        int32_t *shifted_token_ids_device = nullptr; ///< Packed shifted condition-token rows.
        int32_t *shifted_position_ids_device = nullptr; ///< Packed positions paired with shifted tokens.
        int32_t *append_lengths_device = nullptr; ///< Real append width for each padded request row.
        const int32_t *request_row_stride_device = nullptr; ///< Resident padded request width.
        std::vector<const int32_t *> main_cached_tokens_device; ///< Canonical main-KV count per request.
        std::vector<const int32_t *> shifted_cached_tokens_device; ///< Canonical shifted-KV count per request.
        MTPForwardOutput output; ///< Arena-owned depth-zero KV-only scratch tensors.
        uint64_t capture_identity = 0; ///< Complete immutable pointer/generation identity used by graph caching.
        Purpose purpose = Purpose::RuntimeExecution; ///< Permitted lifecycle for this binding.

        /**
         * @brief Validate the immutable portion of the graph binding.
         *
         * Request lengths, token values, positions, and cache counts are
         * deliberately excluded: they are live device data consumed during
         * replay. The vectors themselves must contain one exact counter address
         * per request before graph construction begins.
         */
        [[nodiscard]] bool validForRequestCount(int request_count) const noexcept
        {
            return request_count > 0 && kv_cache && terminal_hidden_archive &&
                   request_token_ids_device && request_position_ids_device &&
                   request_segment_lengths_device &&
                   shifted_token_ids_device && shifted_position_ids_device &&
                   append_lengths_device && request_row_stride_device &&
                   capture_identity != 0 &&
                   main_cached_tokens_device.size() ==
                       static_cast<size_t>(request_count) &&
                   shifted_cached_tokens_device.size() ==
                       static_cast<size_t>(request_count);
        }

        /**
         * @brief Return whether this binding may enter an inference executor.
         *
         * The common validity check intentionally accepts declaration-only
         * bindings so model builders can expose their exact stage topology to
         * workspace planning. This stricter predicate is the execution gate.
         *
         * @param request_count Number of logical requests in the graph.
         * @return true only for a complete runtime binding.
         */
        [[nodiscard]] bool executableForRequestCount(
            int request_count) const noexcept
        {
            return purpose == Purpose::RuntimeExecution &&
                   validForRequestCount(request_count);
        }
    };

    /**
     * @brief Immutable-address contract for main-decode terminal-hidden publication.
     *
     * Every MTP main-model decode produces the hidden row that conditions the
     * next depth-zero sidecar.  The row is part of the live speculative state,
     * so GPU execution publishes it inside the same captured graph as the main
     * logits instead of launching a later row-select graph or retaining a
     * host-side freshness guess.  Decode contributes exactly one physical row
     * per request; the graph therefore copies the fixed contiguous request row
     * range into the persistent terminal archive.
     *
     * The archive pointer and workspace generation are graph identity.  Live
     * token, position, and cache values remain device-owned inputs elsewhere in
     * @ref ForwardInput and are deliberately absent from this binding.
     */
    struct MTPMainTerminalHiddenGraphBinding
    {
        /** @brief Lifecycle in which the immutable pointer set may be used. */
        enum class Purpose
        {
            RuntimeExecution, ///< Finalized workspace generation; executable.
            WorkspaceFamilyDeclaration, ///< Topology sizing only; never executable.
        };

        TensorBase *terminal_hidden_archive = nullptr; ///< One FP32 terminal row per request.
        int request_count = 0; ///< Fixed number of rows published by this graph.
        int source_row_start = 0; ///< First immutable decoded row copied to the archive.
        uint64_t capture_identity = 0; ///< Complete pointer/generation identity.
        Purpose purpose = Purpose::RuntimeExecution; ///< Permitted lifecycle.

        /**
         * @brief Validate the complete graph-execution contract.
         *
         * @param expected_request_count Request count carried by ForwardInput.
         * @return true when this binding can enter graph construction/execution.
         */
        [[nodiscard]] bool validForRequestCount(
            int expected_request_count) const noexcept
        {
            return terminal_hidden_archive != nullptr && request_count > 0 &&
                   request_count == expected_request_count &&
                   source_row_start >= 0 && capture_identity != 0;
        }

        /**
         * @brief Return whether this complete binding may enter inference.
         *
         * @param expected_request_count Request count carried by ForwardInput.
         * @return true only for a complete runtime binding.
         */
        [[nodiscard]] bool executableForRequestCount(
            int expected_request_count) const noexcept
        {
            return purpose == Purpose::RuntimeExecution &&
                   validForRequestCount(expected_request_count);
        }
    };

    /**
     * @brief Immutable-address contract for a captured long-prefill chunk view.
     *
     * Request admission owns one complete token/position bank. A bucketed GPU
     * graph must not ask the host to slice that bank or upload a new row for
     * every replay. Instead, the graph begins with one materialization stage
     * that derives its relative source offset from the canonical device KV
     * count, copies the current bucket into stable arena storage, pads its tail,
     * and publishes the real row count plus physical stride.
     *
     * The cached-token count is deliberately the progression authority. There
     * is no second chunk cursor that could diverge from KV state after prefix
     * restore, graph replay, or an interrupted request. Every pointer and scalar
     * below is immutable graph identity; values behind the pointers are live
     * stream-ordered device state.
     */
    struct DevicePrefillChunkGraphBinding
    {
        IBackend *backend = nullptr; ///< Backend that owns the graph-capturable preparation primitive.
        const int32_t *request_token_ids_device = nullptr; ///< Complete admitted request-token bank.
        const int32_t *request_position_ids_device = nullptr; ///< Complete admitted absolute-position bank.
        const int32_t *request_total_rows_device = nullptr; ///< Admitted logical request length.
        const int32_t *cached_tokens_device = nullptr; ///< Canonical live main-KV progress counter.
        int32_t *chunk_token_ids_device = nullptr; ///< Stable physical bucket consumed by embedding.
        int32_t *chunk_position_ids_device = nullptr; ///< Stable physical bucket consumed by RoPE.
        int32_t *chunk_real_rows_device = nullptr; ///< Current logical row count consumed by stateful stages.
        int32_t *chunk_row_stride_device = nullptr; ///< Current immutable physical bucket stride.
        int request_row_capacity = 0; ///< Flattened rows available in the admitted request bank.
        int bucket_seq_len = 0; ///< Physical graph width materialized on every replay.
        int pad_token_id = 0; ///< Deterministic token used for inactive bucket rows.
        uint64_t capture_identity = 0; ///< Complete pointer/geometry identity used by graph caching.

        /** @return true when every captured address and geometry field is complete. */
        [[nodiscard]] bool valid() const noexcept
        {
            return backend && request_token_ids_device &&
                   request_position_ids_device && request_total_rows_device &&
                   cached_tokens_device && chunk_token_ids_device &&
                   chunk_position_ids_device && chunk_real_rows_device &&
                   chunk_row_stride_device && request_row_capacity > 0 &&
                   bucket_seq_len > 0 &&
                   bucket_seq_len <= request_row_capacity &&
                   capture_identity != 0;
        }
    };

    /**
     * @brief Semantic placement effect of importing model-owned prefix state.
     *
     * Runtime counters, histograms, and epochs may be restored without changing
     * where model payloads reside or execute. This enum prevents callers from
     * treating every successful archive import as a placement mutation.
     */
    enum class PrefixCacheRuntimePlacementEffect : uint8_t
    {
        Unchanged,
        Changed,
    };

    /**
     * @brief Atomic result of one model-owned prefix-runtime restore.
     *
     * `restored` validates the archive transaction. `placement_effect` reports
     * only semantic ownership/residency changes. `device_rehydration_required`
     * declares persistent device work that the next captured main graph must
     * complete before restored routes may execute.
     */
    struct PrefixCacheRuntimeRestoreResult
    {
        bool restored = false;
        PrefixCacheRuntimePlacementEffect placement_effect =
            PrefixCacheRuntimePlacementEffect::Unchanged;
        bool device_rehydration_required = false;

        explicit constexpr operator bool() const noexcept { return restored; }
    };

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
     * @brief Physical tensor surface written by one terminal LM-head graph.
     *
     * Logical graph purpose and physical storage are deliberately independent.
     * A scalar MTP condition is a main-model continuation, but it reuses the
     * preplanned all-position allocation so its captured address remains stable
     * across scalar and request-batched condition graphs. Consumers must use the
     * publication descriptor rather than assuming that a logical main result is
     * always stored in `LOGITS`.
     */
    enum class ForwardLogitsStorageSurface : uint8_t
    {
        Unknown,          ///< A non-null output did not match an owned surface.
        None,             ///< This graph participant has no terminal LM head.
        CanonicalFull,    ///< Full-vocabulary `LOGITS` storage.
        CanonicalLocal,   ///< Column-parallel `LOGITS_LOCAL` storage.
        AllPositionFull,  ///< Full-vocabulary all-position/preplanned storage.
        AllPositionLocal, ///< Column-parallel all-position/preplanned storage.
    };

    /**
     * @brief Typed identity of the logits bytes produced by one forward.
     *
     * `logical_all_position_logits` describes graph semantics: grouped verifier
     * rows are logically all-position output, while a scalar MTP condition is
     * not. `storage_surface` describes only the allocation that received those
     * bytes. Keeping both dimensions prevents a sampler from selecting storage
     * by role name and accidentally reading a stale row.
     */
    struct ForwardLogitsPublicationDescriptor
    {
        ForwardExecutionRole execution_role =
            ForwardExecutionRole::MainInference;
        bool logical_all_position_logits = false;
        ForwardLogitsStorageSurface storage_surface =
            ForwardLogitsStorageSurface::Unknown;

        /** @return true when the producer is a main-model continuation. */
        [[nodiscard]] constexpr bool isMainModelOutput() const noexcept
        {
            return execution_role == ForwardExecutionRole::MainInference ||
                   execution_role == ForwardExecutionRole::MTPCondition;
        }

        /** @return true when a scalar semantic-main consumer may read it. */
        [[nodiscard]] constexpr bool supportsScalarMainConsumer() const noexcept
        {
            return isMainModelOutput() &&
                   !logical_all_position_logits &&
                   storage_surface != ForwardLogitsStorageSurface::Unknown &&
                   storage_surface != ForwardLogitsStorageSurface::None;
        }

        /** @return true when a grouped main-condition consumer may read it. */
        [[nodiscard]] constexpr bool supportsMainRequestBatchConsumer() const noexcept
        {
            return isMainModelOutput() &&
                   (storage_surface ==
                        ForwardLogitsStorageSurface::AllPositionFull ||
                    storage_surface ==
                        ForwardLogitsStorageSurface::AllPositionLocal);
        }

        /** @return true when token columns are sharded across participants. */
        [[nodiscard]] constexpr bool isColumnParallelStorage() const noexcept
        {
            return storage_surface ==
                       ForwardLogitsStorageSurface::CanonicalLocal ||
                   storage_surface ==
                       ForwardLogitsStorageSurface::AllPositionLocal;
        }
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
     * @brief Persistent-state transaction owned by one main forward graph.
     *
     * Most main forwards update only the state naturally owned by their
     * mathematical phase. A one-token suffix after an MTP prefix restore is a
     * deliberate exception: its main-model arithmetic must be serial-decode
     * equivalent, while the same captured transaction must bridge the restored
     * terminal hidden row into depth-zero shifted MTP KV. Naming that lifecycle
     * here prevents a caller from reconstructing it with a prefill-shaped graph
     * or a post-forward sidecar.
     */
    enum class ForwardStateTransaction : uint8_t
    {
        Ordinary = 0, ///< No additional restored-prefix state transition.
        RestoredPrefixMTPDecodeBridge, ///< Decode plus shifted-MTP KV/archive bridge.
    };

    /**
     * @brief Submission lifecycle requested for one forward graph.
     *
     * Production inference normally builds and launches the graph atomically.
     * A setup authority may instead capture and instantiate the exact executable
     * while leaving all model arithmetic, manual boundaries, output publication,
     * and transaction numbering untouched. The first ordinary invocation then
     * owns transaction zero. This policy is deliberately carried beside the
     * input rather than inferred from missing request data.
     */
    enum class ForwardGraphSubmissionIntent : uint8_t
    {
        Execute = 0, ///< Submit one production forward transaction.
        MaterializeExecutableWithoutLaunch, ///< Setup-only capture/instantiation.
    };

    /**
     * @brief Complete asynchronous work represented by a forward completion event.
     *
     * A main-model prefill can either end after the ordinary model graph or own
     * the shifted depth-zero MTP KV population inside that same captured graph.
     * Consumers must not infer this distinction from global MTP configuration:
     * doing so made the benchmark wait for a retired sidecar event after shifted
     * prefill became graph-integrated. The scope travels with invocation
     * provenance so one durable event has an exact, typed meaning.
     */
    enum class ForwardCompletionScope : uint8_t
    {
        ModelForwardOnly, ///< Ordinary model forward outputs and live-state writes.
        GraphIntegratedMTPTerminalHidden, ///< Main decode plus terminal-hidden archive publication.
        GraphIntegratedShiftedMTPPrefill, ///< Model forward plus shifted MTP KV/archive writes.
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
         * arena/workspace memory across graph capture/replay. GPU callers must
         * leave `token_ids` null when this pointer is set: carrying two live
         * representations would make token ownership ambiguous and permit a
         * stale host shadow to re-enter graph construction.
         */
        const void *token_ids_device = nullptr;
        const int *position_ids = nullptr; ///< Host position IDs [batch_size * seq_len]
        /**
         * @brief Optional device-resident INT32 position IDs.
         *
         * Phase 10 resident MTP continuation can publish the next logical
         * positions directly in device memory.  GPU RoPE stages should consume
         * this pointer without recording a host-to-device copy. GPU callers
         * must leave `position_ids` null when it is present; diagnostics may
         * observe device state explicitly but may not retain a competing host
         * owner in the production input contract.
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
        ForwardStateTransaction state_transaction =
            ForwardStateTransaction::Ordinary; ///< Typed persistent-state transition embedded by this graph.
        ForwardGraphSubmissionIntent graph_submission_intent =
            ForwardGraphSubmissionIntent::Execute; ///< Whether this call executes or only seals native graph units.
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
         * @brief Monotonic request generation for graph-native MoE sparse collectives.
         *
         * This scalar is published by @c OrchestrationRunner to every overlay
         * rank at the request boundary. It makes wire keys independent of
         * graph capture, cache residency, and stage-object construction order.
         * A value of zero means this invocation does not use the explicit
         * distributed overlay protocol.
         */
        uint64_t moe_overlay_collective_generation_id = 0;
        /**
         * @brief Absolute logical operation offset for graph-native MoE wire keys.
         *
         * Chunked prefill sets this to the chunk's real-token start; decode and
         * MTP lanes set it to their current logical position. The generation
         * and this value together identify one logical collective operation.
         */
        uint64_t moe_overlay_collective_step_id = 0;
        /**
         * @brief Immutable rank-local execution-sequence identity.
         *
         * The ExpertOverlay coordinator assigns one identity to either one
         * serial graph or the complete MTP draft-plus-verifier sequence.  The
         * device orchestrator uses it only to retain and validate the residency
         * lease across graph submissions; it is orchestration metadata and is
         * deliberately excluded from native graph-cache identity.
         */
        uint64_t moe_overlay_sequence_id = 0;
        /** @brief Placement epoch pinned for the complete execution sequence. */
        uint64_t moe_overlay_sequence_placement_epoch = 0;
        /** @brief Zero-based graph ordinal within @ref moe_overlay_sequence_id. */
        int moe_overlay_sequence_graph_ordinal = -1;
        /** @brief Exact admitted graph count for @ref moe_overlay_sequence_id. */
        int moe_overlay_sequence_graph_count = 0;
        /**
         * @brief Exact pre-launch arm for a heterogeneous ExpertOverlay graph.
         *
         * The continuation coordinator reserves the transaction identity before
         * graph construction so sparse stages can capture immutable wire keys,
         * but it must not publish the remote follower ticket until native capture
         * and instantiation are complete. The executor invokes this callback on
         * its exact non-null execution stream immediately before every initial or
         * replay launch. It is request-scoped orchestration state and deliberately
         * does not participate in graph-cache identity.
         */
        DeviceGraphExecutor::GraphLaunchDependencyHook
            moe_overlay_graph_launch_dependency;
        /**
         * @brief Explicit MTP namespace depth for an overlay transaction.
         *
         * Main-model decode/prefill leave this at -1. Grouped verification
         * publishes the controller-admitted maximum draft depth, independent
         * of the fixed physical verifier bucket.
         */
        int moe_overlay_mtp_depth = -1;
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

        /**
         * @brief Optional graph-integrated shifted MTP prefill transaction.
         *
         * This binding is present only for a GPU main-prefill graph whose MTP
         * depth-zero cache must advance alongside the main cache. Its presence
         * changes graph topology and therefore participates in capture identity.
         * CPU execution retains its host-owned implementation and leaves this
         * field empty.
         */
        std::optional<ShiftedMTPPrefillGraphBinding> shifted_mtp_prefill;

        /**
         * @brief Optional graph-integrated MTP main-decode terminal publication.
         *
         * This binding is present for an MTP-enabled main-model decode and is
         * mutually exclusive with @ref shifted_mtp_prefill.  Its presence
         * guarantees that completion of the forward graph also publishes the
         * exact hidden row needed by prefix checkpoints and the next sidecar.
         */
        std::optional<MTPMainTerminalHiddenGraphBinding>
            mtp_main_terminal_hidden;

        /**
         * @brief Optional captured materialization of one admitted prefill chunk.
         *
         * When present, `token_ids_device`, `position_ids_device`, and
         * `sequence_lengths_device` name the stable chunk outputs in this
         * binding. The preparation stage is inserted by generic graph machinery
         * ahead of every model root, keeping model graph definitions declarative.
         */
        std::optional<DevicePrefillChunkGraphBinding> device_prefill_chunk;

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
     * @brief Resolve the exact completion scope declared by one forward input.
     *
     * Presence of the immutable shifted-prefill binding changes graph topology;
     * it therefore also changes what completion of that graph proves. Binding
     * validity is checked by graph construction, while this helper deliberately
     * performs only the total, side-effect-free policy classification needed by
     * provenance publication and unit tests.
     *
     * @param input Concrete forward invocation whose graph policy is fixed.
     * @return Typed work scope completed by the invocation's producer stream.
     */
    [[nodiscard]] inline constexpr ForwardCompletionScope
    forwardCompletionScopeForInput(const ForwardInput &input) noexcept
    {
        if (input.shifted_mtp_prefill.has_value())
            return ForwardCompletionScope::GraphIntegratedShiftedMTPPrefill;
        if (input.mtp_main_terminal_hidden.has_value())
            return ForwardCompletionScope::GraphIntegratedMTPTerminalHidden;
        return ForwardCompletionScope::ModelForwardOnly;
    }

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
        ForwardCompletionScope completion_scope =
            ForwardCompletionScope::ModelForwardOnly; ///< Complete work transitively covered by this producer.
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

        /**
         * @brief Return the graph-declared mutation and publication contract for MTP sidecars.
         *
         * The default is deliberately undeclared. A concrete model graph must
         * opt in only after its real sidecar has focused state-preservation
         * coverage; merely implementing buildMTPGraph() is not sufficient.
         */
        [[nodiscard]] virtual MTPSidecarStateContract
        mtpSidecarStateContract() const noexcept
        {
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

        /**
         * @brief Restore state captured by capturePrefixCacheRuntimeState().
         *
         * The typed result is the sole publication of restore validity,
         * placement effect, and pending device work. A successful import with
         * Unchanged placement must not invalidate graph identity or advance an
         * MoE movement epoch.
         */
        virtual PrefixCacheRuntimeRestoreResult restorePrefixCacheRuntimeState(
            const std::vector<uint8_t> &state,
            void *stream)
        {
            (void)stream;
            return PrefixCacheRuntimeRestoreResult{
                .restored = state.empty(),
            };
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

        /**
         * @brief Bind the CPU current-batch LLEP physical transfer executor.
         *
         * The ExpertOverlay residency controller remains the placement
         * authority and pins the parent epoch. Model graphs declare the child
         * begin/restore stages; the device orchestrator only executes their
         * packed expert movement and graph-local publication.
         */
        virtual void setCPUCurrentBatchLLEPPhysicalExecutor(
            ICPUCurrentBatchLLEPPhysicalExecutor *executor)
        {
            (void)executor;
        }

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

        /**
         * @brief Return the stable ExpertOverlay epoch binding for one device.
         *
         * Non-overlay builders return an empty binding.  Overlay builders must
         * return the same arena and slot every time, including while main and
         * MTP graph variants are being materialized.  This method is allowed to
         * allocate during model setup; callers must resolve it before capture
         * or request admission.
         *
         * @param device Exact participant whose runtime tables consume the ticket.
         * @return Stable arena/slot binding, or empty for an explicit non-overlay graph.
         */
        virtual DeviceMoEOverlayEpochExecutionBinding
        deviceMoEOverlayEpochExecutionBinding(DeviceId device)
        {
            (void)device;
            return {};
        }

        /**
         * @brief Return the canonical current-batch LLEP evidence table.
         *
         * Non-MoE graphs and MoE policies without current-batch LLEP return an
         * empty source. Implementations must reject ambiguous sources rather
         * than selecting one by container iteration order.
         *
         * @param device Participant whose device-owned runtime table is needed.
         * @return Stable table pointer and complete layer count, or an empty source.
         */
        virtual DeviceMoECurrentBatchLLEPEvidenceSource
        deviceMoECurrentBatchLLEPEvidenceSource(DeviceId device) const
        {
            (void)device;
            return {};
        }

        /**
         * @brief Return the canonical durable runtime table used by device policy.
         *
         * The returned pointer remains model-lifetime stable and is consumed
         * only by a captured backend pack kernel. Non-overlay builders return
         * an empty binding. Implementations must reject ambiguous main tables.
         */
        virtual MoEOverlayDeviceControllerRuntimeBinding
        deviceMoEOverlayControllerRuntimeBinding(DeviceId device) const
        {
            (void)device;
            return {};
        }

        /**
         * @brief Complete the retained-layer runtime image before controller use.
         *
         * The caller owns @p publication_stream and has already ordered it
         * after every graph-build producer. Implementations publish only
         * missing model-lifetime placement layers and must be idempotent.
         * Non-overlay graph builders have no work and return true.
         *
         * @param device Exact local GPU participant.
         * @param publication_stream Exact non-null publication stream.
         * @return True when the complete retained family is queued for use.
         */
        virtual bool finalizeMoEOverlayDeviceControllerRuntime(
            DeviceId device,
            void *publication_stream)
        {
            (void)device;
            (void)publication_stream;
            return true;
        }

        /**
         * @brief Retire model-owned references to borrowed execution streams.
         *
         * A device orchestrator calls this terminal lifecycle edge after all
         * published inference work is complete and before it destroys native
         * graph caches or the device contexts that own their streams. Concrete
         * builders must join any internal producer DAGs while those exact
         * streams remain valid, then erase the borrowed identities. Builders
         * without such references have no work.
         *
         * This hook is terminal for @p device. It must not synchronize a whole
         * device or create a substitute stream; a model-internal maintenance
         * authority may take one exact terminal stream fence after inference
         * admission has stopped.
         *
         * @param device Exact participant whose stream owners are retiring.
         */
        virtual void retireBorrowedExecutionStreams(DeviceId device)
        {
            (void)device;
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
