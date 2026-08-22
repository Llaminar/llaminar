/**
 * @file MoEOverlayActivationEpochABI.h
 * @brief Shared CUDA/ROCm activation-epoch ABI for node-local ExpertOverlay.
 *
 * Heterogeneous ExpertOverlay execution uses one retained transaction on every
 * participating device.  The devices exchange compact activation packets via
 * node-local shared pages and order those packets with capture-stable leased
 * timeline values. This header defines the fixed-width records shared by the
 * CPU reference protocol, CUDA, and ROCm.  It deliberately contains no owning
 * pointers, containers, virtual methods, or backend types.
 *
 * Each mutable field has exactly one writer.  The scheduler arms
 * @ref MoEOverlayActivationEpochIdentity once; the continuation device owns the
 * dispatch descriptors/signals; the follower device owns the return
 * descriptors/signals; and each device owns only its endpoint status. Timeline
 * words carry bank-local visit ordinals and are cleared only after both endpoint
 * terminal events retire the channel lease. The independently monotonic
 * authenticated generation in every identity/descriptor rejects stale packets.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MOE_ACTIVATION_HD __host__ __device__
#else
#define LLAMINAR_MOE_ACTIVATION_HD
#endif

namespace llaminar2
{
    /**
     * @brief Already-built graph family selected by an inference ticket.
     *
     * This enum lives in the device-friendly activation ABI because an
     * authenticated mapped epoch carries the role from the scheduler ticket
     * into an endpoint-private GPU grant.  Device-side consumers such as
     * service telemetry may inspect that immutable grant without consulting a
     * mutable host shadow or capturing one executable per semantic phase.
     */
    enum class MoEOverlayInferenceGraphRole : std::uint32_t
    {
        None = 0, ///< Required for Complete and Abort terminal tickets.
        MainPrefill = 1, ///< Main-model prompt or bounded prefill segment.
        MainDecode = 2, ///< Main-model one-row decode per logical request.
        MTPDraft = 3, ///< One NextN/MTP sidecar depth per logical request.
        MTPGroupedVerifier = 4, ///< Main-model grouped verifier transaction.
    };

    /** Number of alternating payload banks in one activation transaction slot. */
    inline constexpr std::uint32_t kMoEOverlayActivationBufferCount = 2u;

    /** Low timeline bits reserved for a bank-local visit ordinal. */
    inline constexpr std::uint32_t kMoEOverlayActivationVisitBits = 16u;

    /** Largest authenticated generation; zero and the all-ones sentinel are reserved. */
    inline constexpr std::uint64_t kMoEOverlayActivationMaxGeneration =
        ~std::uint64_t{1};

    /** Largest generation representable by the packed diagnostic timeline. */
    inline constexpr std::uint64_t
        kMoEOverlayActivationMaxPackedTimelineGeneration =
            (std::uint64_t{1}
             << (64u - kMoEOverlayActivationVisitBits)) -
            2u;

    /** Terminal value written by a failed endpoint to release every peer wait. */
    inline constexpr std::uint64_t kMoEOverlayActivationAbortTimeline =
        ~std::uint64_t{0};

    /** Capture-stable scheduler publication that admits one retained replay. */
    inline constexpr std::uint64_t kMoEOverlayActivationAdmissionTimeline = 1u;

    /** Fixed activation-channel header magic (`MOEA`). */
    inline constexpr std::uint32_t kMoEOverlayActivationMagic = 0x41454f4du;

    /** Current binary layout and semantic-contract version. */
    inline constexpr std::uint32_t kMoEOverlayActivationABIVersion = 4u;

    /** Device endpoint participating in one sparse activation round trip. */
    enum class MoEOverlayActivationEndpoint : std::uint32_t
    {
        Continuation = 1u, ///< Owns dispatch publication and return consumption.
        Follower = 2u, ///< Owns dispatch consumption and return publication.
    };

    /** Scheduler-owned lifecycle for one fixed activation transaction slot. */
    enum class MoEOverlayActivationAdmissionState : std::uint32_t
    {
        Idle = 0u, ///< No transaction identity is currently published.
        Armed = 1u, ///< Immutable identity is visible to both retained graphs.
        Failed = 2u, ///< Watchdog or scheduler detected a terminal protocol fault.
    };

    /** Device-owned lifecycle for one transaction endpoint. */
    enum class MoEOverlayActivationEndpointState : std::uint32_t
    {
        Idle = 0u, ///< Slot is quiescent and carries no live endpoint work.
        Armed = 1u, ///< Scheduler published identity; device has not validated it.
        Active = 2u, ///< Device validated identity and may advance epochs.
        Complete = 3u, ///< Device completed every stage in the exact transaction.
        Aborted = 4u, ///< Terminal fault; this slot cannot be reset or reused.
    };

    /** Operation most recently attempted by one endpoint. */
    enum class MoEOverlayActivationOperation : std::uint32_t
    {
        None = 0u,
        Activate = 1u,
        PublishDispatch = 2u,
        ConsumeDispatch = 3u,
        PublishReturn = 4u,
        ConsumeReturn = 5u,
        Complete = 6u,
        Abort = 7u,
        Reset = 8u,
        Watchdog = 9u,
    };

    /** Semantic result recorded independently of a kernel/stream launch result. */
    enum class MoEOverlayActivationStatusCode : std::uint32_t
    {
        Idle = 0u,
        Success = 1u,
        NotReady = 2u, ///< Expected peer timeline has not been published yet.
        BufferBusy = 3u, ///< Alternating bank still contains an unconsumed return.
        InvalidControl = 4u,
        InvalidIdentity = 5u,
        StaleGeneration = 6u,
        OutOfOrder = 7u,
        PayloadMismatch = 8u,
        PeerAborted = 9u,
        ExplicitAbort = 10u,
        TimedOut = 11u,
        GenerationOverflow = 12u,
    };

    /** Two independent digest lanes derived from ticket, topology, and channel. */
    struct MoEOverlayActivationDigest
    {
        std::uint64_t low = 0u;
        std::uint64_t high = 0u;

        /** @return Whether both independent lanes are populated. */
        [[nodiscard]] LLAMINAR_MOE_ACTIVATION_HD constexpr bool valid()
            const noexcept
        {
            return low != 0u && high != 0u;
        }

        bool operator==(const MoEOverlayActivationDigest &) const = default;
    };

    /**
     * @brief Immutable setup identity for one participant-to-participant lane.
     *
     * The stage-manifest digest covers the ordered model-layer list embedded by
     * both retained graph families.  A topology, graph, capacity, or layer-list
     * change therefore requires a new channel instead of accepting a vaguely
     * compatible transaction.
     */
    struct alignas(64) MoEOverlayActivationChannelHeader
    {
        std::uint32_t magic = kMoEOverlayActivationMagic;
        std::uint32_t abi_version = kMoEOverlayActivationABIVersion;
        std::uint64_t channel_nonce = 0u;
        std::uint64_t topology_fingerprint_low = 0u;
        std::uint64_t topology_fingerprint_high = 0u;
        std::uint64_t workspace_generation = 0u;
        std::uint64_t stage_manifest_digest = 0u;
        std::int32_t source_world_rank = -1;
        std::int32_t target_world_rank = -1;
        std::int32_t source_participant_id = -1;
        std::int32_t target_participant_id = -1;
        std::int32_t source_tier_priority = 0;
        std::int32_t target_tier_priority = 0;
        std::int32_t source_domain_ordinal = -1;
        std::int32_t target_domain_ordinal = -1;
        std::uint32_t stage_count = 0u;
        std::uint32_t buffer_count = kMoEOverlayActivationBufferCount;
        std::uint32_t lane_ordinal = 0u;
        /** Bit N admits @ref MoEOverlayInferenceGraphRole value N. */
        std::uint32_t graph_role_mask = 0u;
        std::uint64_t reserved[4] = {};
    };

    /**
     * @brief Scheduler-owned admission line for the currently armed identity.
     *
     * `last_generation` survives a successful reset and new transactions must
     * increase it strictly. Timeline signals remain capture-stable across graph
     * replays, so generation and digest fields provide replay authentication.
     * `ready_signal` is written last with release ordering. Both endpoint
     * graphs wait on that exact word before reading any scheduler-owned bytes,
     * which makes admission visibility explicit even when distinct device APIs
     * register different virtual mappings of the same node-local pages. A
     * watchdog failure is terminal and intentionally prevents reset.
     */
    struct alignas(64) MoEOverlayActivationAdmissionControl
    {
        std::uint64_t last_generation = 0u;
        std::uint64_t deadline_ns = 0u;
        std::uint64_t observed_timeout_ns = 0u;
        MoEOverlayActivationDigest digest{};
        std::uint32_t state = static_cast<std::uint32_t>(
            MoEOverlayActivationAdmissionState::Idle);
        std::uint32_t code = static_cast<std::uint32_t>(
            MoEOverlayActivationStatusCode::Idle);
        std::uint64_t ready_signal = 0u;
        std::uint64_t reserved = 0u;
    };

    /**
     * @brief Plain immutable transaction identity consumed by both GPU graphs.
     *
     * These fields are copied from the already-authenticated scheduler ticket.
     * The two digest lanes also include the channel nonce, activation
     * generation, and ordered stage-manifest digest.  Tokens, KV state, logits,
     * routing payloads, and sampler state are deliberately absent.
     */
    struct alignas(64) MoEOverlayActivationEpochIdentity
    {
        std::uint64_t request_generation = 0u;
        std::uint64_t command_id = 0u;
        std::uint64_t transaction_ordinal = 0u;
        std::uint64_t logical_step_id = 0u;
        std::uint64_t workspace_generation = 0u;
        /**
         * Oldest placement epoch this host-scheduled transaction may admit.
         *
         * Device-resident ExpertOverlay maintenance can publish a newer epoch
         * after the immutable scheduler ticket is sent.  Each endpoint must
         * therefore bind its exact device-acquired placement ticket at stage
         * zero and prove that it is no older than this floor.  This field is
         * authentication evidence, never a host mirror of live placement.
         */
        std::uint64_t placement_epoch_floor = 0u;
        std::uint64_t topology_fingerprint_low = 0u;
        std::uint64_t topology_fingerprint_high = 0u;
        std::uint64_t channel_nonce = 0u;
        std::uint64_t epoch_generation = 0u;
        std::uint64_t stage_manifest_digest = 0u;
        MoEOverlayActivationDigest digest{};
        std::int32_t source_world_rank = -1;
        std::int32_t target_world_rank = -1;
        std::int32_t source_participant_id = -1;
        std::int32_t target_participant_id = -1;
        std::int32_t source_tier_priority = 0;
        std::int32_t target_tier_priority = 0;
        std::int32_t source_domain_ordinal = -1;
        std::int32_t target_domain_ordinal = -1;
        std::uint32_t graph_role = 0u;
        std::uint32_t request_count = 0u;
        std::uint32_t logical_rows_per_request = 0u;
        std::uint32_t physical_rows_per_request = 0u;
        std::int32_t draft_depth = -1;
        std::int32_t sidecar_depth = -1;
        std::uint32_t stage_count = 0u;
        std::uint32_t lane_ordinal = 0u;
        std::uint32_t reserved[5] = {};

        bool operator==(
            const MoEOverlayActivationEpochIdentity &) const = default;
    };

    /** @brief Feed one canonical byte through the activation identity digest. */
    LLAMINAR_MOE_ACTIVATION_HD constexpr void
    mixMoEOverlayActivationDigestByte(
        MoEOverlayActivationDigest &digest,
        std::uint8_t byte) noexcept
    {
        digest.low ^= static_cast<std::uint64_t>(byte);
        digest.low *= 1099511628211ull;
        digest.high ^=
            static_cast<std::uint64_t>(byte) + 0x9e3779b97f4a7c15ull;
        digest.high *= 14029467366897019727ull;
        digest.high ^= digest.high >> 29u;
    }

    /** @brief Mix one fixed-width scalar in canonical little-endian order. */
    template <typename Value>
    LLAMINAR_MOE_ACTIVATION_HD constexpr void
    mixMoEOverlayActivationDigestScalar(
        MoEOverlayActivationDigest &digest,
        Value value) noexcept
    {
        using Unsigned = std::make_unsigned_t<Value>;
        Unsigned bits = static_cast<Unsigned>(value);
        for (std::size_t byte = 0; byte < sizeof(Unsigned); ++byte)
        {
            mixMoEOverlayActivationDigestByte(
                digest,
                static_cast<std::uint8_t>(bits & Unsigned{0xffu}));
            bits >>= 8u;
        }
    }

    /**
     * @brief Derive the exact two-lane digest validated by CPU, CUDA, and ROCm.
     *
     * The digest is an integrity/authentication witness for trusted node-local
     * participants, not a network cryptographic MAC.  Its channel nonce and
     * independently derived topology fingerprints prevent a valid ticket from
     * being replayed into another rank pair or retained graph family.
     */
    [[nodiscard]] LLAMINAR_MOE_ACTIVATION_HD constexpr
    MoEOverlayActivationDigest moeOverlayActivationIdentityDigest(
        const MoEOverlayActivationEpochIdentity &identity) noexcept
    {
        MoEOverlayActivationDigest digest{
            .low = 14695981039346656037ull,
            .high = 7809847782465536322ull,
        };
#define LLAMINAR_MOE_ACTIVATION_MIX(field) \
        mixMoEOverlayActivationDigestScalar(digest, identity.field)
        LLAMINAR_MOE_ACTIVATION_MIX(request_generation);
        LLAMINAR_MOE_ACTIVATION_MIX(command_id);
        LLAMINAR_MOE_ACTIVATION_MIX(transaction_ordinal);
        LLAMINAR_MOE_ACTIVATION_MIX(logical_step_id);
        LLAMINAR_MOE_ACTIVATION_MIX(workspace_generation);
        LLAMINAR_MOE_ACTIVATION_MIX(placement_epoch_floor);
        LLAMINAR_MOE_ACTIVATION_MIX(topology_fingerprint_low);
        LLAMINAR_MOE_ACTIVATION_MIX(topology_fingerprint_high);
        LLAMINAR_MOE_ACTIVATION_MIX(channel_nonce);
        LLAMINAR_MOE_ACTIVATION_MIX(epoch_generation);
        LLAMINAR_MOE_ACTIVATION_MIX(stage_manifest_digest);
        LLAMINAR_MOE_ACTIVATION_MIX(source_world_rank);
        LLAMINAR_MOE_ACTIVATION_MIX(target_world_rank);
        LLAMINAR_MOE_ACTIVATION_MIX(source_participant_id);
        LLAMINAR_MOE_ACTIVATION_MIX(target_participant_id);
        LLAMINAR_MOE_ACTIVATION_MIX(source_tier_priority);
        LLAMINAR_MOE_ACTIVATION_MIX(target_tier_priority);
        LLAMINAR_MOE_ACTIVATION_MIX(source_domain_ordinal);
        LLAMINAR_MOE_ACTIVATION_MIX(target_domain_ordinal);
        LLAMINAR_MOE_ACTIVATION_MIX(graph_role);
        LLAMINAR_MOE_ACTIVATION_MIX(request_count);
        LLAMINAR_MOE_ACTIVATION_MIX(logical_rows_per_request);
        LLAMINAR_MOE_ACTIVATION_MIX(physical_rows_per_request);
        LLAMINAR_MOE_ACTIVATION_MIX(draft_depth);
        LLAMINAR_MOE_ACTIVATION_MIX(sidecar_depth);
        LLAMINAR_MOE_ACTIVATION_MIX(stage_count);
        LLAMINAR_MOE_ACTIVATION_MIX(lane_ordinal);
#undef LLAMINAR_MOE_ACTIVATION_MIX
        if (digest.low == 0u)
            digest.low = 0x9e3779b97f4a7c15ull;
        if (digest.high == 0u)
            digest.high = 0xd6e8feb86659fd93ull;
        return digest;
    }

    /**
     * @brief One cache-line packet descriptor published before its timeline.
     *
     * Payload bytes themselves live in setup-owned storage adjacent to, or
     * referenced by, this control ABI.  A producer writes the descriptor and
     * packet first, then publishes `timeline` with a release/fence operation.
     * The consumer waits on the matching signal and validates every descriptor
     * field before reading packet bytes.
     */
    struct alignas(64) MoEOverlayActivationPayloadDescriptor
    {
        MoEOverlayActivationDigest digest{};
        std::uint64_t timeline = 0u;
        std::uint64_t placement_epoch = 0u;
        std::uint64_t live_rows = 0u;
        std::uint64_t live_entries = 0u;
        std::uint64_t payload_bytes = 0u;
        std::uint32_t stage_ordinal = 0u;
        std::int32_t model_layer_index = -1;

        bool operator==(
            const MoEOverlayActivationPayloadDescriptor &) const = default;
    };

    /**
     * @brief Isolated 64-bit device timeline word.
     *
     * Isolation prevents payload metadata and the opposite direction from
     * sharing a CPU cache line or GPU system-memory transaction.  Backends use
     * exact-stream 64-bit wait/write operations on `value`.
     */
    struct alignas(64) MoEOverlayActivationTimelineSignal
    {
        std::uint64_t value = 0u;
        std::uint64_t reserved[7] = {};
    };

    /** One alternating payload bank with independent single-writer directions. */
    struct alignas(64) MoEOverlayActivationBuffer
    {
        MoEOverlayActivationPayloadDescriptor dispatch_descriptor{};
        MoEOverlayActivationTimelineSignal dispatch_signal{};
        MoEOverlayActivationPayloadDescriptor return_descriptor{};
        MoEOverlayActivationTimelineSignal return_signal{};
    };

    /** Device-owned semantic status for one endpoint. */
    struct alignas(64) MoEOverlayActivationEndpointStatus
    {
        MoEOverlayActivationDigest digest{};
        std::uint64_t generation = 0u;
        std::uint64_t observed_timeline = 0u;
        std::uint32_t state = static_cast<std::uint32_t>(
            MoEOverlayActivationEndpointState::Idle);
        std::uint32_t code = static_cast<std::uint32_t>(
            MoEOverlayActivationStatusCode::Idle);
        std::uint32_t operation = static_cast<std::uint32_t>(
            MoEOverlayActivationOperation::None);
        std::int32_t last_published_stage = -1;
        std::int32_t last_consumed_stage = -1;
        std::int32_t last_model_layer = -1;
        /**
         * Predicate bits captured at the first terminal device-side failure.
         *
         * Packet kernels populate this word only on failure.  Each operation
         * defines its own documented bit layout; zero remains the ordinary
         * value for success and for failures that need no additional detail.
         * Keeping the witness in endpoint-owned mapped status lets a watchdog
         * explain a retained-graph rejection without a diagnostic D2H copy.
         */
        std::uint32_t failure_diagnostic = 0u;

        /**
         * Operation-specific compact values accompanying @ref failure_diagnostic.
         * Packet admission currently stores the low 16 bits of the private
         * grant generation and placement epoch here.
         */
        std::uint32_t failure_auxiliary = 0u;

        /** Bytes published by this endpoint during the active epoch. */
        std::uint64_t published_payload_bytes = 0u;

        /** Sum of compact live rows published across all transaction stages. */
        std::uint64_t published_live_rows = 0u;

        /** Sum of compact route entries published across all stages. */
        std::uint64_t published_live_entries = 0u;

        /** Number of authenticated payload descriptors this endpoint published. */
        std::uint64_t published_stage_count = 0u;

        /*
         * These totals are deliberately part of the endpoint-owned ABI rather
         * than host-side mirrors.  The continuation accumulates dispatch
         * traffic and the follower accumulates return traffic on their exact
         * graph streams.  A terminal observer acquires Complete before reading
         * them, so production tests can prove useful sparse work without a D2H
         * copy or a second host authority over routing state.
         */

        /** @return Typed endpoint lifecycle last published by its owner. */
        [[nodiscard]] LLAMINAR_MOE_ACTIVATION_HD constexpr
        MoEOverlayActivationEndpointState typedState() const noexcept
        {
            return static_cast<MoEOverlayActivationEndpointState>(state);
        }

        /** @return Typed semantic result last published by its owner. */
        [[nodiscard]] LLAMINAR_MOE_ACTIVATION_HD constexpr
        MoEOverlayActivationStatusCode typedCode() const noexcept
        {
            return static_cast<MoEOverlayActivationStatusCode>(code);
        }
    };

    /** Complete control plane for one fixed node-local transaction slot. */
    struct alignas(64) MoEOverlayActivationEpochControl
    {
        MoEOverlayActivationChannelHeader channel{};
        MoEOverlayActivationAdmissionControl admission{};
        MoEOverlayActivationEpochIdentity identity{};
        MoEOverlayActivationEndpointStatus continuation_status{};
        MoEOverlayActivationEndpointStatus follower_status{};
        MoEOverlayActivationBuffer buffers[kMoEOverlayActivationBufferCount]{};
    };

    /** @return Alternating payload bank for one zero-based stage ordinal. */
    [[nodiscard]] LLAMINAR_MOE_ACTIVATION_HD constexpr std::uint32_t
    moeOverlayActivationBufferIndex(std::uint32_t stage_ordinal) noexcept
    {
        return stage_ordinal & (kMoEOverlayActivationBufferCount - 1u);
    }

    /** @return One-based visit count of @p stage_ordinal to its selected bank. */
    [[nodiscard]] LLAMINAR_MOE_ACTIVATION_HD constexpr std::uint64_t
    moeOverlayActivationBufferVisit(std::uint32_t stage_ordinal) noexcept
    {
        return static_cast<std::uint64_t>(stage_ordinal / 2u) + 1u;
    }

    /**
     * @brief Return the capture-stable value for one leased bank visit.
     *
     * Retained CUDA/HIP graph memory-operation nodes embed their wait/write
     * values. They therefore use only this graph-family stage ordinal. The
     * scheduler clears both banks after both endpoint terminal events and
     * before the next authenticated generation is armed.
     *
     * @return Zero when @p visit cannot be represented safely.
     */
    [[nodiscard]] LLAMINAR_MOE_ACTIVATION_HD constexpr std::uint64_t
    moeOverlayActivationLeasedTimelineValue(std::uint64_t visit) noexcept
    {
        constexpr std::uint64_t visit_mask =
            (std::uint64_t{1} << kMoEOverlayActivationVisitBits) - 1u;
        return visit == 0u || visit > visit_mask ? 0u : visit;
    }

    /**
     * @brief Pack generation/visit for uncaptured monotonic transport stress.
     *
     * This helper is not the retained activation-channel protocol. Direct
     * TransferEngine integration tests use it to enqueue many generations
     * without a scheduler-owned channel lease/reset between them.
     *
     * @return Zero when either field cannot be represented safely.
     */
    [[nodiscard]] LLAMINAR_MOE_ACTIVATION_HD constexpr std::uint64_t
    moeOverlayActivationPackedTimelineValue(
        std::uint64_t generation,
        std::uint64_t visit) noexcept
    {
        if (generation == 0u ||
            generation >
                kMoEOverlayActivationMaxPackedTimelineGeneration)
        {
            return 0u;
        }
        const std::uint64_t leased =
            moeOverlayActivationLeasedTimelineValue(visit);
        return leased == 0u
                   ? 0u
                   : (generation << kMoEOverlayActivationVisitBits) | leased;
    }

    static_assert(kMoEOverlayActivationBufferCount == 2u);
    static_assert(std::is_trivially_copyable_v<MoEOverlayActivationDigest>);
    static_assert(std::is_trivially_copyable_v<MoEOverlayActivationChannelHeader>);
    static_assert(std::is_trivially_copyable_v<MoEOverlayActivationEpochIdentity>);
    static_assert(std::is_trivially_copyable_v<MoEOverlayActivationPayloadDescriptor>);
    static_assert(std::is_trivially_copyable_v<MoEOverlayActivationEpochControl>);
    static_assert(sizeof(MoEOverlayActivationChannelHeader) == 128u);
    static_assert(sizeof(MoEOverlayActivationAdmissionControl) == 64u);
    static_assert(sizeof(MoEOverlayActivationEpochIdentity) == 192u);
    static_assert(sizeof(MoEOverlayActivationPayloadDescriptor) == 64u);
    static_assert(sizeof(MoEOverlayActivationTimelineSignal) == 64u);
    static_assert(sizeof(MoEOverlayActivationBuffer) == 256u);
    static_assert(sizeof(MoEOverlayActivationEndpointStatus) == 128u);
    static_assert(sizeof(MoEOverlayActivationEpochControl) == 1152u);
    static_assert(alignof(MoEOverlayActivationEpochControl) == 64u);
} // namespace llaminar2

#undef LLAMINAR_MOE_ACTIVATION_HD
