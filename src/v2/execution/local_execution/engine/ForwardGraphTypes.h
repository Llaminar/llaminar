/**
 * @file ForwardGraphTypes.h
 * @brief Data types for forward graph caching (extracted from DeviceGraphOrchestrator)
 *
 * Contains ForwardGraphSignature, ForwardGraphSignatureHash, and ForwardGraphCache —
 * the key data structures for caching compiled forward graphs between decode steps.
 */

#pragma once

#include "../../../backends/DeviceId.h"
#include "../../../backends/IGPUGraphCapture.h"
#include "../../../backends/IWorkerGPUContext.h"
#include "../graph/DeviceGraphExecutor.h"
#include "../../compute_stages/IComputeStage.h"
#include "../graph/IGraphBuilder.h" // For ForwardOutput
#include "PrefillGraphCache.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Result of a graph build operation
     *
     * Holds either a successfully built ComputeGraph + ForwardOutput, or
     * an error string. Extracted from DeviceGraphOrchestrator for use by the
     * ForwardExecutionEngine.
     */
    class GraphBuildResult
    {
    public:
        GraphBuildResult() = default;
        GraphBuildResult(ComputeGraph graph, ForwardOutput output)
            : graph_(std::move(graph)), output_(output), success_(true) {}
        explicit GraphBuildResult(std::string error)
            : error_(std::move(error)), success_(false) {}

        [[nodiscard]] bool success() const { return success_; }
        [[nodiscard]] bool failed() const { return !success_; }
        [[nodiscard]] const std::string &error() const { return error_; }
        [[nodiscard]] ComputeGraph &graph() { return graph_; }
        [[nodiscard]] const ComputeGraph &graph() const { return graph_; }
        [[nodiscard]] const ForwardOutput &output() const { return output_; }
        [[nodiscard]] ComputeGraph takeGraph() { return std::move(graph_); }
        explicit operator bool() const { return success_; }

    private:
        ComputeGraph graph_;
        ForwardOutput output_{};
        std::string error_;
        bool success_ = false;
    };

    /**
     * @brief Configuration for graph caching behaviour
     */
    struct GraphCacheConfig
    {
        bool enabled = true;         ///< Enable graph caching (Phase 10)
        int decode_seq_len = 4;      ///< Ordinary-continuation decode heuristic; typed MTP roles are not bounded by it.
        bool cache_attention = true; ///< Cache attention graphs
        bool cache_ffn = true;       ///< Cache FFN graphs
    };

    /**
     * @brief Immutable inputs to forward-phase policy resolution.
     *
     * Execution role is the authoritative discriminator for internal MTP
     * graphs. Shape-based continuation detection remains only for the legacy
     * public MainInference API, whose caller does not yet carry a typed phase.
     * Keeping those two policies in one value object prevents a future graph
     * entry point from accidentally applying the ordinary-continuation M
     * heuristic to grouped verification again.
     */
    /**
     * @brief Typed public-to-graph invocation policy.
     *
     * This single discriminator replaces the former pair of force-prefill and
     * force-decode booleans, whose fourth combination was invalid. The restored
     * prefix bridge is a decode invocation with an additional graph-owned state
     * transition; keeping it distinct here makes that topology impossible to
     * request accidentally through ordinary decode.
     */
    enum class ForwardInvocationKind : uint8_t
    {
        Automatic = 0, ///< Apply the ordinary public MainInference heuristic.
        ExplicitPrefill, ///< Require prefill mathematical topology.
        ExplicitDecode, ///< Require decode mathematical topology.
        RestoredPrefixMTPDecodeBridge, ///< Decode plus restored-prefix shifted-MTP state bridge.
    };

    struct ForwardExecutionPhaseRequest
    {
        ForwardExecutionRole role = ForwardExecutionRole::MainInference;
        ForwardInvocationKind invocation =
            ForwardInvocationKind::Automatic;
        int seq_len = 0;
        int batch_size = 0;
        int decode_max_seq_len = 1;
        int logical_position = 0;
    };

    /**
     * @brief Resolve the mathematical and topology phase for one forward.
     *
     * Grouped MTP verification and device-resident MTP condition graphs are
     * decode-equivalent by contract for every positive row count. Their M may
     * be 1, the current dynamic-depth range 2..16, or a larger future
     * speculative batch; none may cross into prefill topology merely because
     * it exceeds an ordinary-continuation cache heuristic.
     *
     * MainInference retains the existing compatibility rule: a scalar row is
     * decode, and a small continuation with established history is decode.
     * Explicit phase requests take precedence for that public role.
     *
     * @param request Complete typed role, explicit overrides, and legacy
     *                MainInference shape/history inputs.
     * @return Prefill or decode topology selected without consulting mutable
     *         graph-builder state.
     */
    [[nodiscard]] inline ForwardExecutionPhase resolveForwardExecutionPhase(
        const ForwardExecutionPhaseRequest &request) noexcept
    {
        if (request.role != ForwardExecutionRole::MainInference)
            return ForwardExecutionPhase::Decode;

        switch (request.invocation)
        {
        case ForwardInvocationKind::ExplicitPrefill:
            return ForwardExecutionPhase::Prefill;
        case ForwardInvocationKind::ExplicitDecode:
        case ForwardInvocationKind::RestoredPrefixMTPDecodeBridge:
            return ForwardExecutionPhase::Decode;
        case ForwardInvocationKind::Automatic:
            break;
        }

        const bool scalar_decode =
            request.seq_len == 1 && request.batch_size <= 1;
        const bool short_continuation_decode =
            request.batch_size <= 1 &&
            request.seq_len > 1 &&
            request.seq_len <= std::max(1, request.decode_max_seq_len) &&
            request.logical_position > 0;
        return scalar_decode || short_continuation_decode
                   ? ForwardExecutionPhase::Decode
                   : ForwardExecutionPhase::Prefill;
    }

    /**
     * @brief Whether one forward owns a graph-integrated shifted-MTP archive.
     *
     * Prompt prefill and the one-row restored-prefix bridge have different
     * mathematical phases, but both append depth-zero shifted KV and archive
     * the terminal main-model hidden row inside the same captured transaction.
     * Keeping this classification in one typed predicate prevents callers from
     * publishing only one of those two legal lifecycle edges.  In particular,
     * prefix harvest must not run a sidecar refresh after either transaction:
     * that refresh can overwrite the just-produced main logits before they are
     * archived.
     *
     * @param role Model-graph ownership of the forward.
     * @param phase Mathematical execution phase.
     * @param transaction Persistent-state transaction topology.
     * @param batch_size Logical request count.
     * @param seq_len Physical rows per request.
     * @return true only for a complete shifted-MTP producer topology.
     */
    [[nodiscard]] constexpr bool isGraphIntegratedShiftedMTPTransaction(
        ForwardExecutionRole role,
        ForwardExecutionPhase phase,
        ForwardStateTransaction transaction,
        int batch_size,
        int seq_len) noexcept
    {
        const bool main_prefill =
            role == ForwardExecutionRole::MainInference &&
            phase == ForwardExecutionPhase::Prefill &&
            transaction == ForwardStateTransaction::Ordinary &&
            batch_size > 0 && seq_len > 0;
        const bool restored_prefix_bridge =
            role == ForwardExecutionRole::MTPCondition &&
            phase == ForwardExecutionPhase::Decode &&
            transaction ==
                ForwardStateTransaction::RestoredPrefixMTPDecodeBridge &&
            batch_size == 1 && seq_len == 1;
        return main_prefill || restored_prefix_bridge;
    }

    /**
     * @brief Signature for caching full forward graphs.
     *
     * Captures the execution shape so that graphs built for identical shapes
     * can be reused across decode steps without rebuilding stages/kernels.
     */
    struct ForwardGraphSignature
    {
        int seq_len = 0;
        int batch_size = 0;
        DeviceId device = DeviceId::cpu();
        ForwardExecutionRole execution_role =
            ForwardExecutionRole::MainInference; ///< Mathematical owner embedded by this graph.
        ForwardStateTransaction state_transaction =
            ForwardStateTransaction::Ordinary; ///< Persistent-state topology embedded by this graph.
        bool decode = false;
        bool decode_has_history = false; ///< True for decode calls that already have KV/GDN history.
        bool all_position_logits = false;
        bool live_mtp_request_batch_condition = false; ///< True for one live main-model row per MTP request.
        int all_position_logit_rows = 0; ///< Compact verifier logits row count when all-position logits are row-indexed.
        MTPVerifierOutcomeGraphMode mtp_verifier_outcome_graph_mode =
            MTPVerifierOutcomeGraphMode::Disabled; ///< Terminal compact outcome topology captured by this graph.
        bool uses_device_token_ids = false; ///< True when embedding reads token IDs from a stable device buffer.
        bool uses_device_position_ids = false; ///< True when RoPE reads position IDs from a stable device buffer.
        ForwardPositionPolicy position_policy = ForwardPositionPolicy::ExplicitRows; ///< Position geometry captured by this graph.
        const int32_t *device_sequence_lengths = nullptr; ///< Exact borrowed row-count owner embedded by captured stages.
        uint64_t device_prefill_chunk_capture_identity = 0; ///< Non-zero when a captured device chunk materializer precedes model roots.
        uint64_t shifted_mtp_prefill_capture_identity = 0; ///< Non-zero only when this capture embeds shifted MTP KV prefill.
        uint64_t mtp_main_terminal_hidden_capture_identity = 0; ///< Non-zero when main decode publishes its MTP terminal row in-graph.
        bool standard_path = true;
        bool pp_stage_enabled = false;
        int pp_first_layer = -1;
        int pp_last_layer = -1;
        bool pp_has_embedding = false;
        bool pp_has_lm_head = false;
        bool is_bucketed_prefill = false;
        int bucket_seq_len = 0;
        bool rehydrate_prefix_runtime_on_device = false;
        uint64_t moe_placement_epoch = 0;
        /** Semantic diagnostic-node topology embedded in the native graph. */
        uint64_t snapshot_configuration_identity = 1;

        /** @return Whether this graph embeds a device-owned request-length source. */
        [[nodiscard]] bool usesDeviceSequenceLengths() const noexcept
        {
            return device_sequence_lengths != nullptr;
        }

        bool operator==(const ForwardGraphSignature &other) const
        {
            return seq_len == other.seq_len &&
                   batch_size == other.batch_size &&
                   device == other.device &&
                   execution_role == other.execution_role &&
                   state_transaction == other.state_transaction &&
                   decode == other.decode &&
                   decode_has_history == other.decode_has_history &&
                   all_position_logits == other.all_position_logits &&
                   live_mtp_request_batch_condition ==
                       other.live_mtp_request_batch_condition &&
                   all_position_logit_rows == other.all_position_logit_rows &&
                   mtp_verifier_outcome_graph_mode ==
                       other.mtp_verifier_outcome_graph_mode &&
                   uses_device_token_ids == other.uses_device_token_ids &&
                   uses_device_position_ids == other.uses_device_position_ids &&
                   position_policy == other.position_policy &&
                   device_sequence_lengths == other.device_sequence_lengths &&
                   device_prefill_chunk_capture_identity ==
                       other.device_prefill_chunk_capture_identity &&
                   shifted_mtp_prefill_capture_identity ==
                       other.shifted_mtp_prefill_capture_identity &&
                   mtp_main_terminal_hidden_capture_identity ==
                       other.mtp_main_terminal_hidden_capture_identity &&
                   standard_path == other.standard_path &&
                   pp_stage_enabled == other.pp_stage_enabled &&
                   pp_first_layer == other.pp_first_layer &&
                   pp_last_layer == other.pp_last_layer &&
                   pp_has_embedding == other.pp_has_embedding &&
                   pp_has_lm_head == other.pp_has_lm_head &&
                   is_bucketed_prefill == other.is_bucketed_prefill &&
                   bucket_seq_len == other.bucket_seq_len &&
                   rehydrate_prefix_runtime_on_device ==
                       other.rehydrate_prefix_runtime_on_device &&
                   moe_placement_epoch == other.moe_placement_epoch &&
                   snapshot_configuration_identity ==
                       other.snapshot_configuration_identity;
        }
    };

    /**
     * @brief Derive diagnostic-arena sharing solely from captured graph identity.
     *
     * The policy deliberately depends on typed execution roles rather than
     * stage names or allocation sizes. Prefill buckets and grouped-verifier
     * outcomes share the wide checkpoint lane because prefill publication is
     * complete before device generation begins. Live/restored-prefix MTP
     * conditions use another lane because a retained MTP parent can execute a
     * condition and verifier in the same transaction. Every unproven graph
     * stays dedicated.
     *
     * @param signature Complete forward-cache identity.
     * @return Reuse class and diagnostic configuration namespace.
     */
    inline DeviceGraphExecutor::GraphSnapshotArenaReusePolicy
    graphSnapshotArenaReusePolicyForSignature(
        const ForwardGraphSignature &signature) noexcept
    {
        using ReuseClass =
            DeviceGraphExecutor::GraphSnapshotArenaReuseClass;
        using ReusePolicy =
            DeviceGraphExecutor::GraphSnapshotArenaReusePolicy;

        if (!signature.device.is_gpu())
            return ReusePolicy{};

        ReuseClass reuse_class = ReuseClass::Dedicated;
        if (signature.execution_role ==
            ForwardExecutionRole::GroupedMTPVerifier)
        {
            reuse_class =
                ReuseClass::PrefillOrMTPVerifierAlternative;
        }
        else if (signature.execution_role ==
                 ForwardExecutionRole::MTPCondition)
        {
            reuse_class = ReuseClass::MTPConditionAlternative;
        }
        else if (!signature.decode &&
                 signature.execution_role ==
                     ForwardExecutionRole::MainInference)
        {
            reuse_class =
                ReuseClass::PrefillOrMTPVerifierAlternative;
        }

        if (reuse_class == ReuseClass::Dedicated)
            return ReusePolicy{};
        return ReusePolicy{
            .reuse_class = reuse_class,
            .configuration_identity =
                signature.snapshot_configuration_identity,
        };
    }

    struct ForwardGraphSignatureHash
    {
        size_t operator()(const ForwardGraphSignature &sig) const
        {
            size_t h = std::hash<int>{}(sig.seq_len);
            h ^= (std::hash<int>{}(sig.batch_size) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<DeviceId>{}(sig.device) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<uint8_t>{}(
                      static_cast<uint8_t>(sig.execution_role)) +
                  0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<uint8_t>{}(
                      static_cast<uint8_t>(sig.state_transaction)) +
                  0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<bool>{}(sig.decode) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<bool>{}(sig.decode_has_history) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<bool>{}(sig.all_position_logits) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<bool>{}(sig.live_mtp_request_batch_condition) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<int>{}(sig.all_position_logit_rows) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<uint8_t>{}(
                      static_cast<uint8_t>(
                          sig.mtp_verifier_outcome_graph_mode)) +
                  0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<bool>{}(sig.uses_device_token_ids) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<bool>{}(sig.uses_device_position_ids) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<uint8_t>{}(static_cast<uint8_t>(sig.position_policy)) +
                  0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<const int32_t *>{}(sig.device_sequence_lengths) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<uint64_t>{}(
                      sig.device_prefill_chunk_capture_identity) +
                  0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<uint64_t>{}(
                      sig.shifted_mtp_prefill_capture_identity) +
                  0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<uint64_t>{}(
                      sig.mtp_main_terminal_hidden_capture_identity) +
                  0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<bool>{}(sig.standard_path) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<bool>{}(sig.pp_stage_enabled) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<int>{}(sig.pp_first_layer) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<int>{}(sig.pp_last_layer) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<bool>{}(sig.pp_has_embedding) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<bool>{}(sig.pp_has_lm_head) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<bool>{}(sig.is_bucketed_prefill) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<int>{}(sig.bucket_seq_len) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<bool>{}(sig.rehydrate_prefix_runtime_on_device) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<uint64_t>{}(sig.moe_placement_epoch) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<uint64_t>{}(
                      sig.snapshot_configuration_identity) +
                  0x9e3779b9 + (h << 6) + (h >> 2));
            return h;
        }
    };

    /**
     * @brief Convert a forward-cache identity into immutable replay telemetry.
     *
     * The graph signature is the authoritative source of execution geometry:
     * cache lookup, graph construction, and replay all use that same value.
     * Copying its scalar fields into the segment cache prevents asynchronous
     * GPU-event reclamation from consulting whichever request happens to be
     * current later.  Invalid or overflowing dimensions produce an invalid
     * descriptor, causing PerfStats to omit geometry instead of publishing a
     * plausible but false M value.
     */
    inline DeviceGraphExecutor::GraphSegmentCache::ReplayWorkloadGeometry
    replayWorkloadGeometryForSignature(
        const ForwardGraphSignature &signature) noexcept
    {
        using Geometry =
            DeviceGraphExecutor::GraphSegmentCache::ReplayWorkloadGeometry;

        Geometry geometry{
            .seq_len = signature.seq_len,
            .batch_size = signature.batch_size,
            .m = 0,
            .all_position_rows = signature.all_position_logit_rows,
            .verifier_outcome_mode = static_cast<uint8_t>(
                signature.mtp_verifier_outcome_graph_mode),
            .position_policy = static_cast<uint8_t>(signature.position_policy),
            .moe_placement_epoch = signature.moe_placement_epoch,
            .decode = signature.decode,
            .all_position_logits = signature.all_position_logits,
            .live_mtp_request_batch_condition =
                signature.live_mtp_request_batch_condition,
        };

        if (signature.seq_len > 0 && signature.batch_size > 0 &&
            signature.batch_size <=
                std::numeric_limits<int>::max() / signature.seq_len)
        {
            geometry.m = signature.seq_len * signature.batch_size;
        }
        return geometry;
    }

    enum class ForwardReplayStateCacheClass
    {
        Other,
        ExactPrefill,
        BucketedPrefill,
        OrdinaryDecode,
        SingleTokenOrdinaryDecode,
        AllPositionVerifier,
    };

    enum class ForwardReplayStateMutationKind
    {
        GeneralLiveStateMutation,
        MTPCorrectionReplayBoundary,
        PrefixCheckpointRestore,
        RequestBoundaryStateReset,
    };

    enum class ForwardReplayStateAction
    {
        ResetReplayState,
        PreserveReplayStateAndRebindStreams,
    };

    inline ForwardReplayStateCacheClass classifyForwardReplayStateCache(
        const ForwardGraphSignature &signature)
    {
        if (!signature.decode)
        {
            if (signature.is_bucketed_prefill)
                return ForwardReplayStateCacheClass::BucketedPrefill;
            return ForwardReplayStateCacheClass::ExactPrefill;
        }
        if (signature.all_position_logits)
            return ForwardReplayStateCacheClass::AllPositionVerifier;
        if (signature.seq_len == 1 && signature.batch_size <= 1)
            return ForwardReplayStateCacheClass::SingleTokenOrdinaryDecode;
        return ForwardReplayStateCacheClass::OrdinaryDecode;
    }

    /**
     * @brief True when a segmented decode capture is tied to a specific live
     *        KV/recurrent-state epoch rather than only to stable buffer names.
     *
     * Multi-token ordinary decode encodes live KV/recurrent-state progression
     * directly in the replayed graph.  After MTP publishes an accepted row,
     * that capture must be refreshed before it can read the newly published
     * live state.
     *
     * All-position verifier graphs are different: verifier-row GDN/short-conv
     * state is published through stage-owned capture slots, and verifier row
     * metadata is refreshed before every launch.  Keeping that graph warm is
     * the vLLM-style fast path; stream/event handoff and per-launch metadata
     * updates provide the freshness boundary instead of live-epoch recapture.
     * Single-token decode is also version-safe because token/position metadata
     * is updated before replay and it reads stable live-state buffer addresses.
     */
    inline bool isLiveStateVersionedReplayCache(
        const ForwardGraphSignature &signature)
    {
        if (!signature.decode)
            return false;
        if (signature.all_position_logits)
            return false;
        return !(signature.seq_len == 1 && signature.batch_size <= 1);
    }

    inline ForwardReplayStateAction chooseForwardReplayStateAction(
        ForwardReplayStateMutationKind mutation,
        ForwardReplayStateCacheClass cache_class)
    {
        if ((mutation ==
                 ForwardReplayStateMutationKind::MTPCorrectionReplayBoundary ||
             mutation ==
                 ForwardReplayStateMutationKind::PrefixCheckpointRestore) &&
            cache_class != ForwardReplayStateCacheClass::OrdinaryDecode)
        {
            return ForwardReplayStateAction::PreserveReplayStateAndRebindStreams;
        }
        if (mutation == ForwardReplayStateMutationKind::RequestBoundaryStateReset &&
            (cache_class == ForwardReplayStateCacheClass::SingleTokenOrdinaryDecode ||
             cache_class == ForwardReplayStateCacheClass::AllPositionVerifier ||
             cache_class == ForwardReplayStateCacheClass::ExactPrefill ||
             cache_class == ForwardReplayStateCacheClass::BucketedPrefill))
        {
            return ForwardReplayStateAction::PreserveReplayStateAndRebindStreams;
        }
        return ForwardReplayStateAction::ResetReplayState;
    }

    inline ForwardReplayStateAction chooseForwardReplayStateAction(
        ForwardReplayStateMutationKind mutation,
        const ForwardGraphSignature &signature)
    {
        if ((mutation ==
                 ForwardReplayStateMutationKind::MTPCorrectionReplayBoundary ||
             mutation ==
                 ForwardReplayStateMutationKind::PrefixCheckpointRestore ||
             mutation == ForwardReplayStateMutationKind::RequestBoundaryStateReset) &&
            isLiveStateVersionedReplayCache(signature))
        {
            return ForwardReplayStateAction::ResetReplayState;
        }
        if (mutation == ForwardReplayStateMutationKind::RequestBoundaryStateReset &&
            classifyForwardReplayStateCache(signature) == ForwardReplayStateCacheClass::Other)
        {
            return ForwardReplayStateAction::ResetReplayState;
        }
        return chooseForwardReplayStateAction(
            mutation,
            classifyForwardReplayStateCache(signature));
    }

    /**
     * @brief Latest runtime metadata observed for a prefill graph-cache entry.
     *
     * The cache key remains bucket-shaped so one captured graph can safely serve
     * multiple real-token lengths in the same bucket. This observation records
     * the latest real chunk that flowed through that stable shape for diagnostics,
     * perf export, and Phase 12 graph-capture acceptance.
     */
    struct PrefillGraphExecutionObservation
    {
        bool valid = false;
        int chunk_index = 0;
        int bucket_seq_len = 0;
        int real_token_start = 0;
        int real_token_count = 0;
        int real_token_end = 0;
        std::string domain_id = "single";
        int participant_id = 0;
        uint64_t placement_epoch = 0;
        uint64_t topology_signature = 0;
        std::string capture_phase = "cold";
        std::string recapture_reason = "none";
        std::string reject_stage_name;
        std::string reject_stage_type;
    };

    /**
     * @brief RAII owner for an explicit GPU stream used by cached graph capture.
     *
     * The stream is created through the same worker GPU context that owns the
     * graph-capture backend. It is move-only so ForwardGraphCache entries remain
     * movable inside unordered_map storage without leaking backend stream handles.
     */
    struct CachedGraphStream
    {
        void *stream = nullptr;           ///< Backend stream handle (hipStream_t/cudaStream_t as void*)
        IWorkerGPUContext *ctx = nullptr; ///< Context that created the stream (not owned)

        CachedGraphStream() = default;
        ~CachedGraphStream() { reset(); }

        CachedGraphStream(const CachedGraphStream &) = delete;
        CachedGraphStream &operator=(const CachedGraphStream &) = delete;

        CachedGraphStream(CachedGraphStream &&other) noexcept
            : stream(other.stream), ctx(other.ctx)
        {
            other.stream = nullptr;
            other.ctx = nullptr;
        }

        CachedGraphStream &operator=(CachedGraphStream &&other) noexcept
        {
            if (this != &other)
            {
                reset();
                stream = other.stream;
                ctx = other.ctx;
                other.stream = nullptr;
                other.ctx = nullptr;
            }
            return *this;
        }

        /// @brief Ensure a stream exists for the provided GPU context.
        bool ensure(IWorkerGPUContext *new_ctx)
        {
            if (!new_ctx)
                return false;
            if (stream && ctx == new_ctx)
                return true;

            reset();
            stream = new_ctx->createStream();
            if (!stream)
            {
                ctx = nullptr;
                return false;
            }
            ctx = new_ctx;
            return true;
        }

        /// @brief Destroy the owned stream, if any.
        void reset()
        {
            if (stream && ctx)
            {
                ctx->synchronizeStream(stream);
                ctx->destroyStream(stream);
            }
            stream = nullptr;
            ctx = nullptr;
        }
    };

    /**
     * @brief Cached full forward topology for exact-shape forward execution.
     *
     * Decode and exact CPU prefill both have stable topology for a matching
     * signature; only request-owned token, position, and runtime state changes.
     * Instead of rebuilding hundreds of stage objects and their persistent
     * kernel scratch every forward call, this object owns the graph after first
     * materialization. GPU prefill additionally attaches its native captured
     * executable through @ref prefill_graph_cache.
     *
     * Stable buffers (token_ids, position_ids) are owned here so that
     * cached stages' pointers remain valid across calls.
     */
    struct ForwardGraphCache
    {
        std::unique_ptr<ComputeGraph> graph; ///< Cached compute graph
        ForwardOutput output;                ///< Cached output (logits pointer)
        bool valid = false;                  ///< Whether cache is usable

        /// Workspace generation recorded after the graph's stages were bound.
        uint64_t workspace_generation = 0;

        // Stable buffers — stages point to these, contents updated each step
        std::vector<int> token_ids;    ///< Persistent decode token storage
        std::vector<int> position_ids; ///< Persistent decode position IDs

        // PP hidden state copy — for non-embedding PP stages, the external
        // hidden state must be copied to the working buffer on every forward.
        // During graph build (cache MISS) this copy happens inline in
        // QwenStandardGraph::buildPartialForwardGraph(). On cache HIT we must redo
        // the copy here because the graph build code is not re-executed.
        TensorBase *pp_external_hidden_state = nullptr; ///< Source (stage N-1 output)
        TensorBase *pp_working_buffer = nullptr;        ///< Destination (local residual/hidden)
        size_t pp_copy_bytes = 0;
        DeviceId pp_device;
        bool pp_needs_copy = false;

        // Pre-computed collective stage names for fast decode intercept
        std::unordered_set<std::string> collective_nodes;

        // Pre-cached pointers to stages that override updateDynamicParams().
        // Only ~4 stages (RoPE, Attention, FusedAttention, KVCacheAppend) need
        // updating — avoids iterating all ~339 stages with hash lookups each step.
        std::vector<IComputeStage *> dynamic_param_stages;
        bool dynamic_param_stages_cached = false;

        // Pre-cached pointers to stages that consume fixed-bucket prefill replay
        // metadata. These are updated before prefill capture/replay so device
        // kernels consume real token counts instead of padded rows.
        std::vector<IComputeStage *> prefill_replay_param_stages;
        bool prefill_replay_param_stages_cached = false;

        /**
         * Sparse MoE manual boundaries that consume root-authoritative wire
         * transaction identity. These pointers are cached separately from
         * ordinary dynamic parameters because capture/replay lifecycle must
         * never be allowed to alter distributed key ordering.
         */
        std::vector<IComputeStage *> moe_overlay_collective_runtime_stages;
        bool moe_overlay_collective_runtime_stages_cached = false;

        // Tracks whether setGPUStream has been applied to all stages.
        // Decode graph replay reapplies the capture stream before dynamic
        // params because manual/collective segments may temporarily bind a
        // different stream during execution.
        bool gpu_stream_applied = false;

        // The stream pointer that was last applied to all stages. Replay-state
        // resets preserve the capture stream, but workspace rebinds and full
        // invalidation can still force a new stream binding epoch. Track the
        // pointer so cached stages are rebound whenever that epoch changes.
        void *applied_stream = nullptr;

        // Tracks whether Phase 3 graph replay is active (no markCompleted calls),
        // allowing us to skip the 339-node graph.reset() since flags are already clear.
        bool phase3_active = false;

        /// Live replay-state epoch that the current cached graph capture is safe for.
        /// Multi-token ordinary decode and multi-row all-position verifier graphs
        /// are invalidated when speculative publication advances live state to a
        /// newer epoch. Single-token decode, including MTP condition decode, is
        /// version-safe: it updates token/position metadata before every launch
        /// and reads stable live-state buffer addresses.
        uint64_t graph_replay_live_state_epoch = 0;

        bool requiresLiveStateEpochRecapture(bool live_state_versioned_context,
                                             bool graph_replay_allowed,
                                             uint64_t live_state_epoch) const
        {
            return live_state_versioned_context &&
                   graph_replay_allowed &&
                   segment_cache.initialized &&
                   !segment_cache.needs_capture &&
                   graph_replay_live_state_epoch != 0 &&
                   graph_replay_live_state_epoch != live_state_epoch;
        }

        /// GPU graph capture/replay for eliminating per-kernel launch overhead
        std::unique_ptr<IGPUGraphCapture> gpu_graph;

        /**
         * GPU snapshot descriptors and stable D2D destinations owned by this
         * exact graph geometry. Stage names repeat across cache entries, so this
         * state must never live in a process-wide executor map.
         */
        DeviceGraphExecutor::GraphSnapshotManifest snapshot_manifest;
        uint64_t snapshot_configuration_epoch = 0;

        /// Cached GPU graph replay plan. A fully capturable graph is replayed as
        /// one unit; non-capturable stages and manual boundaries split it.
        DeviceGraphExecutor::GraphSegmentCache segment_cache;

        /// GPU stream (from IWorkerGPUContext::defaultStream()) for kernel dispatch
        /// Set when gpu_graph is created; used by stages to dispatch on correct stream
        void *gpu_stream = nullptr;

        /// GPU context for creating new graph captures (not owned)
        IWorkerGPUContext *gpu_ctx = nullptr;

        /// Prefill graph capture/replay cache (keyed by seq_len)
        std::unique_ptr<PrefillGraphCache> prefill_graph_cache;

        /// Latest diagnostic metadata for bucketed/chunked prefill graph execution.
        PrefillGraphExecutionObservation last_prefill_graph_observation;

        /// Monotonic engine-level LRU tick for all reusable prefill topology entries.
        uint64_t prefill_last_access_tick = 0;

        /// Explicit stream for prefill warmup/capture/replay.
        CachedGraphStream prefill_capture_stream;

        /** @brief Typed decode launch path used to resolve producer ownership. */
        enum class DecodeLaunchPath : uint8_t
        {
            Direct,          ///< Warmup or ordinary non-replay graph execution.
            CapturedReplay,  ///< Launch of a retained native graph executable.
        };

        /**
         * @brief Resolve the exact stream that produced one cached decode.
         *
         * A cached graph owns several stream-shaped fields with different
         * meanings. `applied_stream` only says which stream was most recently
         * installed into stage bindings; replay-state maintenance may update
         * that field without producing the graph output. When a decode
         * invocation actually replayed its captured graph, the segment cache's
         * capture stream is therefore the only valid producer provenance.
         *
         * Prefill deliberately has no matching resolver here because
         * heterogeneous segmented and monolithic prefill use different graph
         * caches. Its state machine returns the concrete launch stream.
         *
         * @param launch_path Whether execution was direct or a retained replay.
         * @return Exact producer stream, or nullptr when the required typed
         *         stream was not published.
         */
        void *decodeOutputProducerStream(
            DecodeLaunchPath launch_path) const noexcept
        {
            if (launch_path == DecodeLaunchPath::CapturedReplay)
                return segment_cache.capture_stream;
            if (applied_stream)
                return applied_stream;
            if (segment_cache.capture_stream)
                return segment_cache.capture_stream;
            return gpu_stream;
        }

        /// Number of consecutive graph update failures (fallback heuristic)
        int gpu_graph_update_failures = 0;

        /// Maximum consecutive update failures before disabling graph capture
        static constexpr int kMaxGraphUpdateFailures = 4;

        /**
         * @brief Reset GPU graph replay/capture state while keeping the cached ComputeGraph.
         *
         * Hard resets preserve graph topology but discard graph executables.
         * Use this after topology/workspace/live-state mutations whose capture
         * safety is not proven. Request-boundary resets that want served-style
         * capture reuse should use resetSessionStatePreservingGraphReplay().
         *
         * The capture stream itself is retained because cached stages store that
         * stream pointer internally. Destroying it here would leave dynamic-param
         * updates (for example token-id preloads) with a dangling HIP stream before
         * the next warmup has a chance to rebind every stage.
         */
        void resetReplayState()
        {
            if (gpu_graph)
            {
                gpu_graph->reset();
                gpu_graph.reset();
            }
            segment_cache.reset(DeviceGraphExecutor::GraphSegmentCache::StreamResetPolicy::Preserve);
            gpu_graph_update_failures = 0;
            phase3_active = false;
            graph_replay_live_state_epoch = 0;
        }

        /**
         * @brief Drop captured replay state after workspace buffers are rebound.
         *
         * The cached ComputeGraph remains valid, but HIP/CUDA graph captures
         * encode raw workspace addresses. Rebinding stages to a new workspace
         * manager therefore requires throwing away any captured decode segments
         * and monolithic prefill graph entries before the next launch.
         */
        void resetReplayStateAfterWorkspaceRebind()
        {
            resetReplayState();
            if (prefill_graph_cache)
                prefill_graph_cache->invalidateAll();
            gpu_stream_applied = false;
            applied_stream = nullptr;
        }

        /**
         * @brief Force cached stages to rebind their explicit GPU stream.
         *
         * KernelFactory::resetAllDynamicState() clears cached kernel stream
         * bindings. Some correction-replay paths deliberately preserve
         * verifier graph executables, but those stages still need to receive
         * the capture stream again before updateDynamicParams() or replay
         * callbacks can safely touch backend objects.
         */
        void markGPUStreamBindingsDirty()
        {
            gpu_stream_applied = false;
            applied_stream = nullptr;
        }

        /**
         * @brief Stamp a preserved replay capture for the current live state.
         *
         * This is intentionally reserved for replay classes whose state-safety
         * has been proven separately. In particular, single-token condition
         * decode and all-position verifier replay may be preserved across MTP
         * accepted-state publication; multi-token ordinary decode remains
         * epoch-versioned and recaptures after publication.
         */
        void markReplayStateSafeForLiveEpoch(uint64_t live_state_epoch)
        {
            graph_replay_live_state_epoch = live_state_epoch;
        }

        /**
         * @brief Reset request-scoped stage state while preserving safe graph replay.
         *
         * Request boundaries clear KV/GDN/short-conv live state, token metadata,
         * and backend stream bindings, but single-token decode and all-position
         * verifier captures are designed to read stable device buffers whose
         * contents are refreshed before every launch.  Keeping those graph
         * executables hot is the served-inference path we want: warmup captures
         * once, later requests replay after device-state reset.
         *
         * Prefill executables follow the same stable-address contract. Their
         * kernels read persistent KV, recurrent, routing, and request-input
         * storage; request reset clears contents but does not replace those
         * allocations. Ready executables therefore survive, while mutable
         * request metadata is republished on their exact stream before replay.
         * Warmup and Initialized entries retain only lazy initialization because
         * they do not yet own a complete executable.
         */
        void resetSessionStatePreservingGraphReplay()
        {
            if (gpu_graph)
            {
                gpu_graph->reset();
                gpu_graph.reset();
            }
            PrefillGraphRequestResetSummary prefill_reset;
            if (prefill_graph_cache)
            {
                prefill_reset =
                    prefill_graph_cache->prepareEntriesForRequestReset(
                        /*preserve_ready_executables=*/true);
            }
            const bool prefill_request_state_was_reset =
                prefill_reset.ready_demoted > 0 ||
                prefill_reset.initialized > 0 ||
                prefill_reset.dropped > 0;
            const bool captured_replay_preserved =
                (segment_cache.initialized && !segment_cache.needs_capture) ||
                prefill_reset.ready_preserved > 0;
            if (!captured_replay_preserved)
            {
                last_prefill_graph_observation = {};
            }
            /*
             * The observation also carries the complete durable cache-key
             * identity (domain, participant, placement epoch, and topology
             * signature) used by backend-neutral readiness probes.  When a
             * Ready executable survives this reset, clearing that identity
             * would make diagnostics query a synthetic default key and report
             * the live graph as Cold.  Request-shaped token offsets remain a
             * historical last-execution observation until the next replay;
             * current readiness still comes exclusively from PrefillGraphCache.
             */

            if (graph)
            {
                const bool lazy_prefill_only =
                    !captured_replay_preserved &&
                    (prefill_reset.ready_demoted > 0 || prefill_reset.initialized > 0);

                graph->reset();
                for (const auto &node_name : graph->getExecutionOrder())
                {
                    ComputeNode *node = graph->getNode(node_name);
                    if (node && node->stage)
                    {
                        if (lazy_prefill_only)
                            node->stage->resetSessionStatePreservingLazyInitialization();
                        else
                            node->stage->resetSessionStatePreservingCapturedReplay();
                    }
                }
            }

            (void)prefill_request_state_was_reset;
            markGPUStreamBindingsDirty();
            gpu_graph_update_failures = 0;
            phase3_active = segment_cache.initialized && !segment_cache.needs_capture;
            graph_replay_live_state_epoch = 0;
        }

        /**
         * @brief Reset request-scoped state while preserving reusable graph objects.
         *
         * This is the request-boundary counterpart to invalidate(): it keeps the
         * cached ComputeGraph, stable token buffers, and workspace bindings, but
         * clears stage/kernels' dynamic metadata and decode replay captures so
         * the next prompt starts from cleared KV/GDN model state.
         *
         * Hard request resets use this path for cache classes whose replay is
         * not proven safe. It keeps the cached ComputeGraph, but discards
         * prefill graph cache executables and the owned prefill capture stream
         * while preserving workspace bindings and prepared weights.
         */
        void resetSessionState()
        {
            resetReplayState();
            if (prefill_graph_cache)
                prefill_graph_cache->invalidateAll(PrefillGraphRejectReason::RequestStateReset);
            last_prefill_graph_observation = {};

            if (graph)
            {
                for (const auto &node_name : graph->getExecutionOrder())
                {
                    ComputeNode *node = graph->getNode(node_name);
                    if (node && node->stage)
                        node->stage->resetSessionState();
                }
            }

            prefill_capture_stream.reset();
            gpu_stream_applied = false;
            applied_stream = nullptr;
            gpu_stream = nullptr;
            gpu_ctx = nullptr;
            graph_replay_live_state_epoch = 0;
        }

        void invalidate()
        {
            resetReplayState();
            segment_cache.reset(DeviceGraphExecutor::GraphSegmentCache::StreamResetPolicy::Destroy);
            if (prefill_graph_cache)
                prefill_graph_cache->invalidateAll();
            last_prefill_graph_observation = {};
            prefill_capture_stream.reset();
            prefill_last_access_tick = 0;
            graph.reset();
            snapshot_manifest.clear();
            snapshot_configuration_epoch = 0;
            valid = false;
            workspace_generation = 0;
            token_ids.clear();
            position_ids.clear();
            collective_nodes.clear();
            dynamic_param_stages.clear();
            dynamic_param_stages_cached = false;
            prefill_replay_param_stages.clear();
            prefill_replay_param_stages_cached = false;
            moe_overlay_collective_runtime_stages.clear();
            moe_overlay_collective_runtime_stages_cached = false;
            gpu_stream_applied = false;
            applied_stream = nullptr;
            gpu_stream = nullptr;
            gpu_ctx = nullptr;
            phase3_active = false;
            graph_replay_live_state_epoch = 0;
            pp_external_hidden_state = nullptr;
            pp_working_buffer = nullptr;
            pp_copy_bytes = 0;
            pp_needs_copy = false;
        }
    };

    /**
     * @brief Return the flattened row count owned by explicit position IDs.
     *
     * ForwardInput::seq_len is the padded row width of one request, while host
     * and device position arrays are laid out as `[batch_size * seq_len]`.
     * Dynamic RoPE updates must consume the flattened count or later requests
     * remain unrotated.  Returning zero for malformed or overflowing geometry
     * gives cache-miss and replay callers one shared validation contract.
     *
     * @param input Current forward invocation and its request geometry.
     * @return Positive flattened position-row count, or zero when invalid.
     */
    inline int forwardPositionRowCount(const ForwardInput &input)
    {
        if (input.batch_size <= 0 || input.seq_len <= 0 ||
            input.seq_len >
                std::numeric_limits<int>::max() / input.batch_size)
        {
            return 0;
        }
        return input.batch_size * input.seq_len;
    }

    /**
     * @brief Select the host position rows that should refresh a cached forward replay.
     *
     * Cached forward graphs keep `position_ids` as stable graph-build storage, but that
     * storage is not the owner of replay-time absolute positions.  A replay input can
     * reuse the same bucket shape for a different chunk of the request, so the current
     * `ForwardInput` must win whenever it provides fresh host rows.  The cache-owned
     * rows are only a compatibility fallback for callers that have no explicit replay
     * rows.  When device-resident rows are present, this helper returns null so callers
     * keep the device pointer as the single source of truth.
     *
     * @param forward_cache Cache entry that owns graph-build fallback rows.
     * @param input Current replay input for this forward invocation.
     * @return Host position row pointer for dynamic replay, or null when none applies.
     */
    inline const int *selectForwardReplayHostPositionIds(
        const ForwardGraphCache &forward_cache,
        const ForwardInput &input)
    {
        if (input.position_ids_device)
            return nullptr;
        if (input.position_ids)
            return input.position_ids;
        if (!forward_cache.position_ids.empty())
            return forward_cache.position_ids.data();
        return nullptr;
    }

} // namespace llaminar2
