/**
 * @file MoEOverlayDistributedResidencyProtocol.h
 * @brief Typed rank-consensus protocol for distributed ExpertOverlay epochs.
 *
 * A cross-rank residency wave is safe to publish only after every rank has
 * staged its physical arrivals and installed a complete inactive participant
 * bank.  This file owns the device-free wire identity and rank-local state
 * machine for those two votes.  MPI progress, network payloads, and backend
 * events are deliberately supplied by a separate transport: this protocol
 * decides when their results form one globally publishable epoch.
 */

#pragma once

#include "MoEOverlayResidencyAuthority.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace llaminar2
{
    /** @brief Stable 128-bit digest of one complete residency transaction. */
    struct MoEOverlayResidencyTransactionFingerprint
    {
        std::uint64_t low = 0;
        std::uint64_t high = 0;

        /** @return Whether both independently mixed digest lanes are present. */
        [[nodiscard]] bool valid() const noexcept
        {
            return low != 0 && high != 0;
        }

        bool operator==(
            const MoEOverlayResidencyTransactionFingerprint &) const = default;
    };

    /**
     * @brief Fingerprint a transaction without using process-local pointers.
     *
     * The digest covers old and candidate ownership, endpoint topology, tier
     * policy, the frozen histogram, migrations, cycles, and shadow-slot BOM.
     * Independently constructed rank-local transactions therefore agree while
     * a stale epoch, divergent histogram, or topology mismatch cannot vote in
     * the same wave.
     *
     * @param transaction Structurally valid residency transaction.
     * @return Stable fingerprint suitable for fixed-layout rank votes.
     * @throws std::invalid_argument When @p transaction is not valid.
     */
    [[nodiscard]] MoEOverlayResidencyTransactionFingerprint
    fingerprintMoEOverlayResidencyTransaction(
        const MoEOverlayResidencyTransaction &transaction);

    /** @brief Fixed header preceding one coordinator-published histogram bank. */
    struct MoEOverlayDistributedHistogramHeader
    {
        static constexpr std::uint32_t kMagic = 0x484F4F4Du; // "MOOH"
        static constexpr std::uint32_t kABIVersion = 2u;
        static constexpr std::size_t kWireBytes = 64u;

        std::uint32_t magic = kMagic;
        std::uint32_t abi_version = kABIVersion;
        std::uint64_t generation = 0;
        std::uint64_t token_count = 0;
        std::int32_t num_layers = 0;
        std::int32_t num_experts = 0;
        std::uint32_t production_source_count =
            static_cast<std::uint32_t>(
                kExpertHistogramProductionSourceCount);
        std::uint32_t reserved = 0;
        std::uint64_t expert_count_entries = 0;
        std::uint64_t source_expert_count_entries = 0;
        std::uint64_t counts_fingerprint = 0;

        /** @return Whether geometry, count, and digest form a valid envelope. */
        [[nodiscard]] bool valid() const noexcept;
    };

    static_assert(
        std::is_trivially_copyable_v<
            MoEOverlayDistributedHistogramHeader>);
    static_assert(
        sizeof(MoEOverlayDistributedHistogramHeader) ==
        MoEOverlayDistributedHistogramHeader::kWireBytes);

    /**
     * @brief Return exact fixed-packet bytes for one model histogram geometry.
     * @throws std::invalid_argument For non-positive geometry.
     * @throws std::overflow_error When flattened counts exceed size_t.
     */
    [[nodiscard]] std::size_t moeOverlayDistributedHistogramWireBytes(
        int num_layers,
        int num_experts);

    /**
     * @brief Stable digest of generation, geometry, and every expert count.
     * @param window Valid immutable routing-evidence generation.
     * @return Non-zero endian-independent digest.
     * @throws std::invalid_argument For an invalid window.
     */
    [[nodiscard]] std::uint64_t fingerprintDecodeExpertHistogramWindow(
        const DecodeExpertHistogramWindow &window);

    /**
     * @brief Encode one frozen histogram into an exact little-endian packet.
     * @param window Valid immutable routing-evidence generation.
     * @param destination Exact-size caller-owned packet storage.
     * @param error Optional validation diagnostic.
     * @return True only when every destination byte was written.
     */
    bool encodeMoEOverlayDistributedHistogramWindow(
        const DecodeExpertHistogramWindow &window,
        std::span<std::uint8_t> destination,
        std::string *error = nullptr);

    /**
     * @brief Decode and authenticate one fixed histogram packet.
     * @param packet Complete little-endian packet from the coordinator.
     * @param expected_layers Model-owned layer geometry.
     * @param expected_experts Model-owned expert geometry.
     * @param window Receives the authenticated generation and counts.
     * @param error Optional malformed-packet diagnostic.
     * @return True only for exact size, geometry, and digest agreement.
     *
     * Callers may pre-size aggregate and source count vectors to reuse
     * persistent storage; resize to the same counts performs no allocation.
     */
    bool decodeMoEOverlayDistributedHistogramWindow(
        std::span<const std::uint8_t> packet,
        int expected_layers,
        int expected_experts,
        DecodeExpertHistogramWindow *window,
        std::string *error = nullptr);

    /** @brief Fixed-layout identity shared by every rank in one migration wave. */
    struct MoEOverlayDistributedResidencyWaveIdentity
    {
        static constexpr std::uint32_t kMagic = 0x52574F4Du; // "MOWR"
        static constexpr std::uint32_t kABIVersion = 1u;

        std::uint32_t magic = kMagic;
        std::uint32_t abi_version = kABIVersion;
        std::uint64_t expected_epoch = 0;
        std::uint64_t candidate_epoch = 0;
        std::uint64_t histogram_generation = 0;
        std::uint64_t migration_count = 0;
        std::uint64_t cycle_count = 0;
        MoEOverlayResidencyTransactionFingerprint transaction_fingerprint;

        /** @return Whether ABI, epoch progression, counts, and digest are valid. */
        [[nodiscard]] bool valid() const noexcept;

        bool operator==(
            const MoEOverlayDistributedResidencyWaveIdentity &) const = default;
    };

    static_assert(
        std::is_trivially_copyable_v<
            MoEOverlayDistributedResidencyWaveIdentity>);

    /**
     * @brief Build the fixed-layout consensus identity for a transaction.
     * @param transaction Valid, non-empty residency transaction.
     * @return Identity naming the exact candidate epoch and movement set.
     * @throws std::invalid_argument For an invalid or empty transaction.
     */
    [[nodiscard]] MoEOverlayDistributedResidencyWaveIdentity
    makeMoEOverlayDistributedResidencyWaveIdentity(
        const MoEOverlayResidencyTransaction &transaction);

    /** @brief Global barrier represented by one rank vote. */
    enum class MoEOverlayDistributedResidencyVotePhase : std::uint32_t
    {
        Reserved = 1, ///< All local slots, pins, and lane admission are owned.
        Staged = 2,   ///< Physical arrivals are complete and authenticated.
        InactivePrepared = 3, ///< Candidate banks are locally installed and ready.
        RuntimePublished = 4, ///< Every local device selector names the candidate.
        LeaseDrained = 5, ///< Every rank released all old-epoch dispatches.
    };

    /** @brief Rank-local outcome contributed to one global barrier. */
    enum class MoEOverlayDistributedResidencyVoteDecision : std::uint32_t
    {
        Ready = 1,  ///< This rank completed the named phase.
        Failed = 2, ///< This rank rejects the unpublished candidate.
        Deferred = 3, ///< This rank lacks transient stage capacity; all retry.
    };

    /**
     * @brief Fixed-layout rank vote exchanged by an asynchronous communicator.
     *
     * The full diagnostic remains in the failing rank's logs.  Peers receive a
     * positive typed code and stable detail digest, which is sufficient to
     * identify the first failing rank without putting variable strings in the
     * MPI wire ABI.
     */
    struct MoEOverlayDistributedResidencyVote
    {
        static constexpr std::uint32_t kMagic = 0x56574F4Du; // "MOWV"
        static constexpr std::uint32_t kABIVersion = 4u;

        std::uint32_t magic = kMagic;
        std::uint32_t abi_version = kABIVersion;
        MoEOverlayDistributedResidencyWaveIdentity identity;
        MoEOverlayDistributedResidencyVotePhase phase =
            MoEOverlayDistributedResidencyVotePhase::Staged;
        MoEOverlayDistributedResidencyVoteDecision decision =
            MoEOverlayDistributedResidencyVoteDecision::Ready;
        std::int32_t world_rank = -1;
        std::int32_t error_code = 0;
        std::uint64_t detail_fingerprint = 0;

        /**
         * @brief Validate ABI, identity, rank range, and decision diagnostics.
         * @param world_size Positive communicator size.
         * @return Whether this is one well-formed fixed-layout vote.
         */
        [[nodiscard]] bool valid(int world_size) const noexcept;

        bool operator==(
            const MoEOverlayDistributedResidencyVote &) const = default;
    };

    static_assert(
        std::is_trivially_copyable_v<MoEOverlayDistributedResidencyVote>);

    /**
     * @brief Asynchronous all-rank exchange for one fixed-layout residency vote.
     *
     * The protocol intentionally depends on this small device-free boundary
     * instead of MPI itself. Production binds it to a private MPI communicator;
     * unit tests bind it to an adversarial in-process exchange. Exactly one
     * exchange may be active per lane and all methods are non-blocking.
     */
    class IMoEOverlayResidencyConsensusLane
    {
    public:
        virtual ~IMoEOverlayResidencyConsensusLane() = default;

        /**
         * @brief Start exchanging one valid vote from this lane's local rank.
         * @param vote Fixed-layout local vote for the currently due phase.
         * @param error Receives an exact validation or transport diagnostic.
         * @return True only after the lane owns an asynchronous exchange.
         */
        virtual bool begin(
            const MoEOverlayDistributedResidencyVote &vote,
            std::string *error = nullptr) = 0;

        /**
         * @brief Progress the active exchange once without waiting.
         * @param votes Receives one rank-ordered vote per participant on Ready.
         * @param error Receives an exact transport diagnostic on Failed.
         * @return Pending, Ready, or Failed for the active exchange.
         */
        virtual MoEOverlayResidencyWaveProgress poll(
            std::vector<MoEOverlayDistributedResidencyVote> *votes,
            std::string *error = nullptr) = 0;

        /** @return Whether this lane owns no in-flight exchange. */
        [[nodiscard]] virtual bool idle() const noexcept = 0;

        /** @return World rank contributed by this lane. */
        [[nodiscard]] virtual int worldRank() const noexcept = 0;

        /** @return Number of ranks participating in every exchange. */
        [[nodiscard]] virtual int worldSize() const noexcept = 0;
    };

    /** @brief Rank-local lifecycle enforced around asynchronous vote exchange. */
    enum class MoEOverlayDistributedResidencyProtocolState
    {
        AwaitingLocalReservation,
        AwaitingReservationConsensus,
        AwaitingLocalStage,
        AwaitingStageConsensus,
        AwaitingLocalPrepare,
        AwaitingPrepareConsensus,
        AwaitingLocalPublication,
        AwaitingPublicationConsensus,
        ReadyForAuthorityPublication,
        Deferred,
        Failed,
        Published,
        AwaitingRetirementConsensus,
        ReadyToRetire,
        Retired,
        Aborted,
    };

    /** @brief Deterministic identity of the first rank that rejected a phase. */
    struct MoEOverlayDistributedResidencyFailure
    {
        MoEOverlayDistributedResidencyVotePhase phase =
            MoEOverlayDistributedResidencyVotePhase::Staged;
        int world_rank = -1;
        int error_code = 0;
        std::uint64_t detail_fingerprint = 0;
    };

    /**
     * @brief Device-free rank-local publication and retirement state machine.
     *
     * The eventual MPI transport calls @ref makeLocalVote after its local
     * event DAG reaches a terminal result, exchanges exactly one vote per rank,
     * and passes the gathered fixed-layout records to @ref acceptConsensus.
     * No method performs I/O, waits, allocates backend memory, or publishes an
     * authority epoch.
     */
    class MoEOverlayDistributedResidencyProtocol final
    {
    public:
        /** @brief Immutable communicator membership and wave identity. */
        struct Config
        {
            MoEOverlayDistributedResidencyWaveIdentity identity;
            int local_world_rank = -1;
            int world_size = 0;
        };

        /**
         * @brief Construct a rank-local protocol at the staging boundary.
         * @param config Valid identity and exact communicator membership.
         * @throws std::invalid_argument For invalid identity or rank geometry.
         */
        explicit MoEOverlayDistributedResidencyProtocol(Config config);

        /**
         * @brief Record this rank's terminal result for the currently due phase.
         * @param decision Ready or Failed.
         * @param error_code Zero for Ready; positive for Failed.
         * @param diagnostic Empty for Ready; precise local failure for Failed.
         * @return Fixed-layout vote to exchange exactly once.
         * @throws std::logic_error When the local phase is not awaiting a vote.
         * @throws std::invalid_argument For inconsistent diagnostics.
         */
        [[nodiscard]] MoEOverlayDistributedResidencyVote makeLocalVote(
            MoEOverlayDistributedResidencyVoteDecision decision,
            int error_code = 0,
            const std::string &diagnostic = {});

        /**
         * @brief Validate and apply one complete all-rank vote generation.
         * @param votes Exactly one vote for every rank in the communicator.
         * @param error Receives a precise mismatch or remote-failure diagnostic.
         * @return True only for unanimous readiness of the expected phase.
         *
         * Unanimous reservation permits physical staging, unanimous staging
         * permits inactive-bank preparation, unanimous preparation permits
         * selector publication, unanimous selector publication becomes
         * ReadyForAuthorityPublication, and the later lease-drain vote makes
         * the previous epoch ReadyToRetire. Any malformed or failed generation
         * is terminal.
         */
        bool acceptConsensus(
            const std::vector<MoEOverlayDistributedResidencyVote> &votes,
            std::string *error = nullptr);

        /**
         * @brief Mark the globally selector-published candidate authoritative.
         * @throws std::logic_error Unless state is ReadyForAuthorityPublication.
         */
        void markAuthorityPublished();

        /**
         * @brief Mark the globally lease-drained old epoch physically retired.
         * @throws std::logic_error Unless retirement consensus is complete.
         */
        void markRetired();

        /**
         * @brief Mark an unpublished protocol explicitly aborted.
         * @return False when the epoch was already published.
         */
        bool abort() noexcept;

        /** @return Current rank-local lifecycle state. */
        [[nodiscard]] MoEOverlayDistributedResidencyProtocolState state()
            const noexcept
        {
            return state_;
        }

        /** @return Immutable consensus identity. */
        [[nodiscard]] const MoEOverlayDistributedResidencyWaveIdentity &identity()
            const noexcept
        {
            return config_.identity;
        }

        /** @return First deterministic failed vote, when terminally failed. */
        [[nodiscard]] const std::optional<
            MoEOverlayDistributedResidencyFailure> &failure() const noexcept
        {
            return failure_;
        }

    private:
        /** @brief Expected vote phase implied by the current lifecycle state. */
        [[nodiscard]] MoEOverlayDistributedResidencyVotePhase expectedPhase()
            const;

        /** @brief Enter terminal failure and retain its deterministic identity. */
        void fail(
            MoEOverlayDistributedResidencyFailure failure,
            std::string message,
            std::string *error);

        Config config_;
        MoEOverlayDistributedResidencyProtocolState state_ =
            MoEOverlayDistributedResidencyProtocolState::
                AwaitingLocalReservation;
        std::optional<MoEOverlayDistributedResidencyVote> local_vote_;
        std::optional<MoEOverlayDistributedResidencyFailure> failure_;
        std::string failure_message_;
    };
} // namespace llaminar2
