/**
 * @file MoEOverlayActivationEpochProtocol.h
 * @brief CPU reference state machine for node-local device-owned MoE epochs.
 *
 * The production CUDA and ROCm implementations advance the ABI in
 * @ref MoEOverlayActivationEpochABI.h from exact non-null streams.  This class
 * is the device-free executable specification used by unit tests and by setup
 * validation.  It never serves as a host shadow for a GPU-owned live control
 * block: once device execution starts, production observes only terminal status
 * through the explicit watchdog boundary.
 */

#pragma once

#include "MoEOverlayActivationEpochABI.h"
#include "MoEOverlayInferenceTransaction.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace llaminar2
{
    /**
     * @brief Backend-agnostic endpoint selected by the topology planner.
     *
     * No CUDA, ROCm, or CPU tag belongs here. The participant inventory maps
     * this logical identity to a physical execution backend after planning, so
     * either GPU family (or CPU) can occupy either protocol role.
     */
    struct MoEOverlayActivationLaneEndpoint
    {
        std::int32_t world_rank = -1;
        std::int32_t participant_id = -1;
        std::int32_t tier_priority = 0;
        std::int32_t domain_ordinal = -1;

        /** @return Whether rank, participant, and domain identities are present. */
        [[nodiscard]] bool valid() const noexcept
        {
            return world_rank >= 0 && participant_id >= 0 &&
                   domain_ordinal >= 0;
        }

        bool operator==(const MoEOverlayActivationLaneEndpoint &) const = default;
    };

    /** @brief Immutable setup contract shared independently by both ranks. */
    struct MoEOverlayActivationEpochConfig
    {
        std::uint64_t channel_nonce = 0u;
        std::uint64_t topology_fingerprint_low = 0u;
        std::uint64_t topology_fingerprint_high = 0u;
        /** Exact retained graph/workspace generation authenticated by tickets. */
        std::uint64_t workspace_generation = 0u;
        MoEOverlayActivationLaneEndpoint source;
        MoEOverlayActivationLaneEndpoint target;
        std::uint32_t lane_ordinal = 0u;
        /** Bit N admits @ref MoEOverlayInferenceGraphRole value N. */
        std::uint32_t graph_role_mask = 0u;
        /** Strictly increasing model-layer indices embedded in the transaction. */
        std::vector<std::int32_t> model_layer_indices;

        /** @return Whether the rank pair, topology, and stage manifest are usable. */
        [[nodiscard]] bool valid() const noexcept;

        /** @return Whether every immutable scheduler field is identical. */
        bool operator==(const MoEOverlayActivationEpochConfig &) const =
            default;
    };

    /**
     * @brief Immutable traffic totals published by one completed device epoch.
     *
     * Dispatch values belong to the continuation endpoint; return values
     * belong to the follower endpoint.  They are acquired only after both
     * endpoint Complete publications, so the record is evidence produced by
     * the retained graphs themselves rather than inferred host-side traffic.
     */
    struct MoEOverlayActivationEpochTraffic
    {
        std::uint64_t dispatch_payload_bytes = 0u;
        std::uint64_t return_payload_bytes = 0u;
        std::uint64_t dispatch_live_rows = 0u;
        std::uint64_t return_live_rows = 0u;
        std::uint64_t dispatch_live_entries = 0u;
        std::uint64_t return_live_entries = 0u;
        std::uint64_t dispatch_stage_count = 0u;
        std::uint64_t return_stage_count = 0u;
    };

    /**
     * @brief Lock-free, single-writer reference implementation of activation epochs.
     *
     * The scheduler calls @ref arm before submitting either retained graph and
     * @ref reset only after both endpoint terminal events and Complete states.
     * Reset clears capture-stable bank signals before the next lease. Continuation
     * and follower methods may execute concurrently.  They communicate only via
     * release/acquire timeline words and never wait internally; a missing peer
     * publication returns a typed `NotReady` diagnostic so tests can drive any
     * adversarial interleaving without introducing a hidden blocking primitive.
     */
    class MoEOverlayActivationEpochProtocol final
    {
    public:
        /**
         * @brief Initialize one pristine shared control block during setup.
         * @throws std::invalid_argument for an invalid channel configuration.
         * @throws std::logic_error when @p control is not pristine.
         */
        static void initialize(
            MoEOverlayActivationEpochControl &control,
            const MoEOverlayActivationEpochConfig &config);

        /**
         * @brief Bind the reference endpoint to an initialized exact channel.
         * @throws std::invalid_argument for an invalid expected configuration.
         * @throws std::logic_error when the mapped header disagrees with it.
         */
        MoEOverlayActivationEpochProtocol(
            MoEOverlayActivationEpochControl &control,
            MoEOverlayActivationEpochConfig config);

        /**
         * @brief Derive the stable ordered-layer digest stored in the channel ABI.
         * @param model_layer_indices Exact graph stage order; indices must be non-negative.
         * @return Non-zero stable digest, or zero for an invalid/empty manifest.
         */
        [[nodiscard]] static std::uint64_t stageManifestDigest(
            std::span<const std::int32_t> model_layer_indices) noexcept;

        /**
         * @brief Arm a quiescent slot with one immutable scheduler ticket.
         * @param ticket Already-validated Execute ticket for this rank pair.
         * @param epoch_generation Strictly increasing timeline generation.
         * @param deadline_ns Positive absolute watchdog deadline.
         * @param error Optional stable diagnostic.
         * @return Complete device identity, or empty on rejection.
         */
        [[nodiscard]] std::optional<MoEOverlayActivationEpochIdentity> arm(
            const MoEOverlayInferenceTransactionTicket &ticket,
            std::uint64_t epoch_generation,
            std::uint64_t deadline_ns,
            std::string *error = nullptr);

        /**
         * @brief Validate the armed identity and make one endpoint device-active.
         * @return False and terminally aborts that endpoint on any mismatch.
         */
        bool activate(
            MoEOverlayActivationEndpoint endpoint,
            const MoEOverlayActivationEpochIdentity &identity,
            std::string *error = nullptr);

        /**
         * @brief Continuation publication of one compact dispatch packet.
         *
         * Two adjacent stages may be outstanding. Reusing a bank for stage N+2
         * is backpressured until the continuation consumed stage N's return.
         */
        bool publishDispatch(
            const MoEOverlayActivationEpochIdentity &identity,
            std::uint32_t stage_ordinal,
            std::uint64_t live_rows,
            std::uint64_t live_entries,
            std::uint64_t payload_bytes,
            std::string *error = nullptr);

        /**
         * @brief Follower validation and consumption of one dispatch descriptor.
         * @return Descriptor on success; empty while not ready or after rejection.
         */
        [[nodiscard]] std::optional<MoEOverlayActivationPayloadDescriptor>
        consumeDispatch(
            const MoEOverlayActivationEpochIdentity &identity,
            std::uint32_t stage_ordinal,
            std::string *error = nullptr);

        /**
         * @brief Follower publication of the exact stage's compact return packet.
         * @param payload_bytes Compact dense-row bytes; zero only for an empty
         *        dispatch. Device-owned canonical-route publishers write their
         *        entry geometry directly into the shared ABI.
         */
        bool publishReturn(
            const MoEOverlayActivationEpochIdentity &identity,
            std::uint32_t stage_ordinal,
            std::uint64_t payload_bytes,
            std::string *error = nullptr);

        /**
         * @brief Continuation validation and consumption of one return descriptor.
         * @return Descriptor on success; empty while not ready or after rejection.
         */
        [[nodiscard]] std::optional<MoEOverlayActivationPayloadDescriptor>
        consumeReturn(
            const MoEOverlayActivationEpochIdentity &identity,
            std::uint32_t stage_ordinal,
            std::string *error = nullptr);

        /**
         * @brief Publish endpoint completion after every exact stage round trip.
         * @return False for premature or mismatched completion.
         */
        bool complete(
            MoEOverlayActivationEndpoint endpoint,
            const MoEOverlayActivationEpochIdentity &identity,
            std::string *error = nullptr);

        /**
         * @brief Publish a terminal endpoint fault and release peer stream waits.
         *
         * The endpoint writes the abort sentinel only to timeline directions it
         * owns. The peer's next wait passes, then descriptor validation reports
         * `PeerAborted`. Aborted slots are process-terminal and cannot reset.
         */
        bool abort(
            MoEOverlayActivationEndpoint endpoint,
            const MoEOverlayActivationEpochIdentity &identity,
            MoEOverlayActivationStatusCode code,
            std::string *error = nullptr);

        /**
         * @brief Record that one endpoint observed its peer's terminal abort.
         * @return False unless the peer already published Aborted.
         */
        bool acknowledgePeerAbort(
            MoEOverlayActivationEndpoint endpoint,
            const MoEOverlayActivationEpochIdentity &identity,
            std::string *error = nullptr);

        /**
         * @brief Terminal watchdog observation at the explicit host boundary.
         * @param epoch_generation Exact currently armed generation.
         * @param observed_ns Monotonic timestamp to compare with the armed deadline.
         * @return True only when this call transitions Armed to Failed/TimedOut.
         */
        bool markTimedOut(
            std::uint64_t epoch_generation,
            std::uint64_t observed_ns,
            std::string *error = nullptr);

        /**
         * @brief Retire a successful lease and clear its capture-stable signals.
         * @return False unless both endpoint terminal events completed this identity.
         * @note The caller proves terminal-event completion; semantic Complete
         *       states alone do not prove every kernel in a GPU graph retired.
         */
        bool reset(
            const MoEOverlayActivationEpochIdentity &identity,
            std::string *error = nullptr);

        /** @return Copy of the scheduler-owned active identity. */
        [[nodiscard]] MoEOverlayActivationEpochIdentity activeIdentity()
            const noexcept;

        /** @return Current typed scheduler admission state. */
        [[nodiscard]] MoEOverlayActivationAdmissionState admissionState()
            const noexcept;

        /** @return Current typed lifecycle of @p endpoint. */
        [[nodiscard]] MoEOverlayActivationEndpointState endpointState(
            MoEOverlayActivationEndpoint endpoint) const noexcept;

        /** @return Snapshot of @p endpoint's device-owned semantic status. */
        [[nodiscard]] MoEOverlayActivationEndpointStatus endpointStatus(
            MoEOverlayActivationEndpoint endpoint) const noexcept;

        /**
         * @brief Acquire graph-owned traffic totals for one successful epoch.
         *
         * The method is valid only at the explicit terminal scheduler boundary
         * after both devices have published Complete and before @ref reset.
         * It validates digest, generation, stage totality, and dispatch/return
         * row symmetry before exposing the immutable evidence.
         *
         * @param identity Exact still-armed epoch identity.
         * @param error Optional stable rejection diagnostic.
         * @return Completed traffic, or empty for stale/incomplete evidence.
         */
        [[nodiscard]] std::optional<MoEOverlayActivationEpochTraffic>
        completedTraffic(
            const MoEOverlayActivationEpochIdentity &identity,
            std::string *error = nullptr) const;

        /** @return Acquire-load of a selected dispatch timeline word. */
        [[nodiscard]] std::uint64_t dispatchTimeline(
            std::uint32_t buffer_index) const noexcept;

        /** @return Acquire-load of a selected return timeline word. */
        [[nodiscard]] std::uint64_t returnTimeline(
            std::uint32_t buffer_index) const noexcept;

        /** @return Exact immutable channel header. */
        [[nodiscard]] const MoEOverlayActivationChannelHeader &channelHeader()
            const noexcept
        {
            return control_->channel;
        }

    private:
        /** @return Whether identity exactly matches the active authenticated value. */
        [[nodiscard]] bool matchesActiveIdentity(
            const MoEOverlayActivationEpochIdentity &identity) const noexcept;

        /** @return Model layer embedded for one valid stage, or -1. */
        [[nodiscard]] std::int32_t modelLayer(
            std::uint32_t stage_ordinal) const noexcept;

        /** @return Exact endpoint-owned status record. */
        [[nodiscard]] MoEOverlayActivationEndpointStatus &status(
            MoEOverlayActivationEndpoint endpoint) noexcept;
        [[nodiscard]] const MoEOverlayActivationEndpointStatus &status(
            MoEOverlayActivationEndpoint endpoint) const noexcept;

        /** @brief Publish a successful/transient endpoint diagnostic. */
        void publishStatus(
            MoEOverlayActivationEndpoint endpoint,
            MoEOverlayActivationOperation operation,
            MoEOverlayActivationStatusCode code,
            std::int32_t stage_ordinal,
            std::int32_t model_layer,
            std::uint64_t timeline) noexcept;

        /** @brief Abort an endpoint for an invariant violation and report false. */
        bool reject(
            MoEOverlayActivationEndpoint endpoint,
            MoEOverlayActivationOperation operation,
            MoEOverlayActivationStatusCode code,
            std::string message,
            std::string *error);

        /** @return Whether endpoint is active under the exact identity. */
        bool validateActiveEndpoint(
            MoEOverlayActivationEndpoint endpoint,
            const MoEOverlayActivationEpochIdentity &identity,
            MoEOverlayActivationOperation operation,
            std::string *error);

        /** @return Whether one published descriptor matches exact stage identity. */
        [[nodiscard]] bool validDescriptor(
            const MoEOverlayActivationPayloadDescriptor &descriptor,
            const MoEOverlayActivationEpochIdentity &identity,
            std::uint32_t stage_ordinal,
            std::uint64_t timeline) const noexcept;

        /** @brief Clear a status during successful scheduler-owned reset. */
        static void clearStatus(
            MoEOverlayActivationEndpointStatus &status) noexcept;

        MoEOverlayActivationEpochControl *control_ = nullptr;
        MoEOverlayActivationEpochConfig config_;
    };
} // namespace llaminar2
