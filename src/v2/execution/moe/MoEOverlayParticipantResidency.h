/**
 * @file MoEOverlayParticipantResidency.h
 * @brief Epoch-indexed prepared-expert banks for one overlay participant.
 *
 * Sparse ExpertOverlay packets may remain in flight while maintenance prepares
 * and publishes a new owner map.  A graph stage therefore cannot resolve an
 * expert through a mutable "current" mask.  It must resolve the exact epoch
 * carried by the packet and retain every prepared GEMM lifetime used by that
 * epoch until the residency authority retires it.  This file supplies that
 * participant-local RCU storage without owning routing or global publication.
 */

#pragma once

#include "../../backends/DeviceId.h"
#include "DecodeExpertHistogram.h"
#include "MoEExpertOwnerMap.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{
    class ExpertGemmRegistry;
    class ITensorGemm;

    /**
     * @brief Coherent raw service totals for one participant/layer coordinate.
     *
     * Durations cover the exact prepared gate/up/down execution boundary. The
     * setup certifier normalizes `total_nanoseconds` by `activation_count`
     * only after this participant has supplied every production phase. Raw
     * integer totals avoid rounding each short GPU event independently.
     */
    struct MoEOverlayParticipantLayerServiceTotals
    {
        int participant_id = -1;
        int layer = -1;
        std::array<uint64_t, kExpertHistogramProductionSourceCount>
            total_nanoseconds{};
        std::array<uint64_t, kExpertHistogramProductionSourceCount>
            activation_count{};
        std::array<uint64_t, kExpertHistogramProductionSourceCount>
            sample_count{};
        std::array<bool, kExpertHistogramProductionSourceCount> overflowed{};

        /** @return Whether the coordinate and accumulated arithmetic are coherent. */
        [[nodiscard]] bool valid() const noexcept;
    };

    /** @brief Nonblocking result of publishing one exact service observation. */
    enum class MoEOverlayServiceMeasurementRecordStatus
    {
        Recorded,  ///< The complete duration/activation pair was accumulated.
        Disabled,  ///< Static residency deliberately has no economy profiler.
        Contended, ///< Another writer owns this cell; inference did not wait.
        Invalid,   ///< Phase, layer, duration, or activation geometry was invalid.
        Overflow,  ///< The cell saturated and cannot be used for certification.
    };

    /**
     * @brief Complete prepared gate/up/down engines for one resident expert.
     *
     * Shared ownership is intentional.  Registry entries and cached graph
     * stages may come and go independently, while this triplet is the lifetime
     * authority for the exact residency epoch that names the engines.
     */
    struct MoEOverlayPreparedExpertTriplet
    {
        std::shared_ptr<ITensorGemm> gate;
        std::shared_ptr<ITensorGemm> up;
        std::shared_ptr<ITensorGemm> down;

        /** @return Whether all three projections are present. */
        [[nodiscard]] bool complete() const noexcept;

        /** @return Whether no projection lifetime is present. */
        [[nodiscard]] bool empty() const noexcept;

        /** @return Whether two triplets name the same three engine objects. */
        [[nodiscard]] bool sameIdentity(
            const MoEOverlayPreparedExpertTriplet &other) const noexcept;
    };

    /**
     * @brief Resolve exact shared participant-scoped engine lifetimes for one layer.
     *
     * Raw graph-stage pointers are insufficient for RCU residency because the
     * mutable preparation registry may replace them during migration. This
     * helper converts the canonical participant mask into complete shared
     * triplets suitable for an immutable epoch bank.
     *
     * @param registry Model-owned prepared engine registry.
     * @param participant Canonical logical endpoint descriptor.
     * @param layer_idx Transformer layer index.
     * @param num_experts Exact global expert count.
     * @param resident_mask Canonical residents for this endpoint and layer.
     * @param output Receives one triplet per global expert id.
     * @param error Optional failure diagnostic.
     * @return True only when every resident has an exact owned triplet and
     *         every non-resident output entry is empty.
     */
    bool resolveMoEOverlayPreparedExpertTriplets(
        const ExpertGemmRegistry &registry,
        const MoEExpertOwnerParticipant &participant,
        int layer_idx,
        int num_experts,
        const std::vector<bool> &resident_mask,
        std::vector<MoEOverlayPreparedExpertTriplet> &output,
        std::string *error = nullptr);

    /** @brief Immutable-ready engine table and local mask for one model layer. */
    struct MoEOverlayParticipantLayerBank
    {
        std::vector<bool> resident_mask;
        std::vector<MoEOverlayPreparedExpertTriplet> experts;

        /**
         * @brief Install one complete resident expert into this candidate bank.
         * @throws std::out_of_range When @p expert_id is outside the bank.
         * @throws std::invalid_argument When @p engines is incomplete.
         */
        void setResidentExpert(
            int expert_id,
            MoEOverlayPreparedExpertTriplet engines);

        /**
         * @brief Remove one expert from this candidate without touching old banks.
         * @throws std::out_of_range When @p expert_id is outside the bank.
         */
        void clearExpert(int expert_id);

        /** @return Whether geometry and resident/engine totality are exact. */
        [[nodiscard]] bool valid(int num_experts) const noexcept;

        /** @return Whether masks and engine identities are identical. */
        [[nodiscard]] bool sameIdentity(
            const MoEOverlayParticipantLayerBank &other) const noexcept;
    };

    /**
     * @brief One complete participant-local prepared bank for a residency epoch.
     *
     * Candidate construction clones the previous bank, applies every arrival
     * and departure, validates the complete layer geometry, and only then hands
     * the value to @ref MoEOverlayParticipantResidency for publication-ready
     * retention.  After installation callers must treat their source value as
     * immutable; the owner stores an independent copy to enforce that rule.
     */
    struct MoEOverlayParticipantResidencyBank
    {
        uint64_t epoch = 0;
        int participant_id = -1;
        DeviceId device = DeviceId::invalid();
        std::vector<MoEOverlayParticipantLayerBank> layers;

        /** @return Whether identity, geometry, masks, and engine triplets are exact. */
        [[nodiscard]] bool valid(
            int expected_participant_id,
            DeviceId expected_device,
            int num_layers,
            int num_experts) const noexcept;

        /** @return Whether two banks are the same immutable publication. */
        [[nodiscard]] bool sameIdentity(
            const MoEOverlayParticipantResidencyBank &other) const noexcept;
    };

    /** @brief Opaque retained-bank node owned only by the residency authority. */
    struct MoEOverlayParticipantPublishedBank;

    /** @brief Opaque lock-free reader and maintenance-side slot state. */
    class MoEOverlayParticipantResidencyState;

    /**
     * @brief Fully validated immutable bank prepared away from publication.
     *
     * Construction performs the allocation and complete geometry validation.
     * The maintenance transaction can therefore prepare every endpoint before
     * publishing its first one.  Installing this move-only value transfers one
     * already-built node into a fixed slot; it never copies an expert table or
     * allocates storage at the inference-visible publication boundary.
     */
    class MoEOverlayPreparedParticipantBank final
    {
    public:
        /** @brief Construct an empty, non-publishable value. */
        MoEOverlayPreparedParticipantBank() noexcept;

        /** @brief Destroy an unpublished prepared node, if one remains. */
        ~MoEOverlayPreparedParticipantBank();

        /** @brief Transfer the sole unpublished-node ownership. */
        MoEOverlayPreparedParticipantBank(
            MoEOverlayPreparedParticipantBank &&other) noexcept;

        /** @brief Replace this value with another unpublished node. */
        MoEOverlayPreparedParticipantBank &operator=(
            MoEOverlayPreparedParticipantBank &&other) noexcept;

        MoEOverlayPreparedParticipantBank(
            const MoEOverlayPreparedParticipantBank &) = delete;
        MoEOverlayPreparedParticipantBank &operator=(
            const MoEOverlayPreparedParticipantBank &) = delete;

        /** @return Whether this value still owns a publishable node. */
        [[nodiscard]] bool valid() const noexcept;

        /** @return Prepared residency epoch, or zero for an empty value. */
        [[nodiscard]] uint64_t epoch() const noexcept;

    private:
        friend class MoEOverlayParticipantResidency;

        /** @brief Adopt one validated, fully allocated unpublished node. */
        explicit MoEOverlayPreparedParticipantBank(
            std::unique_ptr<MoEOverlayParticipantPublishedBank> node) noexcept;

        std::unique_ptr<MoEOverlayParticipantPublishedBank> node_;
    };

    /**
     * @brief Allocation-free inference lease over one immutable epoch bank.
     *
     * Acquisition first enters a short lock-free hazard window, then increments
     * the selected bank's reader count. Retirement removes the raw publication
     * pointer immediately but reclaims the node only after both the hazard
     * window and every returned lease have drained. Copies retain the same node
     * with one atomic increment; no maintenance mutex participates.
     */
    class MoEOverlayParticipantBankLease final
    {
    public:
        /** @brief Construct an empty lease. */
        MoEOverlayParticipantBankLease() noexcept;

        /** @brief Release one reader from the retained immutable bank. */
        ~MoEOverlayParticipantBankLease();

        /** @brief Retain the same immutable bank for another local consumer. */
        MoEOverlayParticipantBankLease(
            const MoEOverlayParticipantBankLease &other) noexcept;

        /** @brief Release the old bank and retain @p other. */
        MoEOverlayParticipantBankLease &operator=(
            const MoEOverlayParticipantBankLease &other) noexcept;

        /** @brief Transfer one retained reader without changing its count. */
        MoEOverlayParticipantBankLease(
            MoEOverlayParticipantBankLease &&other) noexcept;

        /** @brief Release the old bank and transfer @p other. */
        MoEOverlayParticipantBankLease &operator=(
            MoEOverlayParticipantBankLease &&other) noexcept;

        /** @return Immutable bank pointer, or null for an empty lease. */
        [[nodiscard]] const MoEOverlayParticipantResidencyBank *get()
            const noexcept;

        /** @return Immutable bank referenced by this non-empty lease. */
        [[nodiscard]] const MoEOverlayParticipantResidencyBank &operator*()
            const noexcept;

        /** @return Immutable bank pointer for member access. */
        [[nodiscard]] const MoEOverlayParticipantResidencyBank *operator->()
            const noexcept;

        /** @return Whether this lease names a retained bank. */
        [[nodiscard]] explicit operator bool() const noexcept;

        /** @brief Release the retained reader and become empty. */
        void reset() noexcept;

        /** @return Whether @p lease is empty. */
        friend bool operator==(
            const MoEOverlayParticipantBankLease &lease,
            std::nullptr_t) noexcept
        {
            return lease.get() == nullptr;
        }

        /** @return Whether @p lease is empty. */
        friend bool operator==(
            std::nullptr_t,
            const MoEOverlayParticipantBankLease &lease) noexcept
        {
            return lease.get() == nullptr;
        }

    private:
        friend class MoEOverlayParticipantResidency;

        /** @brief Adopt one reader already counted by lock-free acquisition. */
        MoEOverlayParticipantBankLease(
            std::shared_ptr<MoEOverlayParticipantResidencyState> state,
            MoEOverlayParticipantPublishedBank *node) noexcept;

        /** @brief Increment the current node reader count for a copy. */
        void retain() noexcept;

        std::shared_ptr<MoEOverlayParticipantResidencyState> state_;
        MoEOverlayParticipantPublishedBank *node_ = nullptr;
    };

    /** @brief Typed result of installing one publication-ready epoch bank. */
    enum class MoEOverlayParticipantBankInstallStatus
    {
        Installed,
        AlreadyInstalled,
        CapacityUnavailable,
        Invalid,
        EpochConflict,
    };

    /**
     * @brief Participant-local multi-version prepared-engine authority.
     *
     * The global @ref MoEOverlayResidencyAuthority decides which epoch new
     * tickets acquire.  This object owns the corresponding physical prepared
     * banks for one endpoint.  Candidate banks may be installed before global
     * publication because no packet can name that epoch yet.  Old banks remain
     * addressable until the global authority invokes retirement after its last
     * ticket lease drains.
     */
    class MoEOverlayParticipantResidency final
    {
    public:
        /** @brief Immutable endpoint geometry and bounded shadow capacity. */
        struct Config
        {
            int participant_id = -1;
            DeviceId device = DeviceId::invalid();
            int num_layers = 0;
            int num_experts = 0;
            /**
             * Maximum simultaneously retained ready epochs.
             * Two represents one live bank plus one candidate/retiring bank.
             */
            std::size_t retained_epoch_capacity = 2;
            /**
             * Collect exact prepared-engine service observations for certification.
             *
             * Dynamic production overlays enable this once at endpoint
             * construction. Static overlays leave it false, proving their
             * no-movement path pays no timing-event or accounting cost.
             */
            bool collect_economy_service_measurements = false;
        };

        /**
         * @brief Construct one empty endpoint authority.
         * @throws std::invalid_argument For invalid identity, geometry, or capacity.
         */
        explicit MoEOverlayParticipantResidency(Config config);

        MoEOverlayParticipantResidency(
            const MoEOverlayParticipantResidency &) = delete;
        MoEOverlayParticipantResidency &operator=(
            const MoEOverlayParticipantResidency &) = delete;

        /** @return Stable logical participant id. */
        [[nodiscard]] int participantId() const noexcept
        {
            return config_.participant_id;
        }

        /** @return Exact execution device for this endpoint. */
        [[nodiscard]] DeviceId device() const noexcept
        {
            return config_.device;
        }

        /** @return Fixed transformer-layer count. */
        [[nodiscard]] int numLayers() const noexcept
        {
            return config_.num_layers;
        }

        /** @return Fixed routed-expert count per layer. */
        [[nodiscard]] int numExperts() const noexcept
        {
            return config_.num_experts;
        }

        /** @return Whether this endpoint accepts service timing observations. */
        [[nodiscard]] bool collectsEconomyServiceMeasurements() const noexcept
        {
            return config_.collect_economy_service_measurements;
        }

        /**
         * @brief Accumulate one exact prepared-expert execution without waiting.
         *
         * The method attempts one ownership CAS for the addressed phase/layer
         * cell. A collision drops this calibration sample and returns
         * `Contended`; it never spins, allocates, or delays inference. Duration
         * and activation totals are updated together while the cell is owned,
         * so maintenance cannot observe half of one sample.
         *
         * @param layer Transformer layer measured by the local expert stage.
         * @param source Decode, prefill, or grouped-verifier semantic phase.
         * @param elapsed_nanoseconds Complete packet-to-return-row wall time.
         * @param activations Number of routed expert activations in that duration.
         */
        [[nodiscard]] MoEOverlayServiceMeasurementRecordStatus
        recordServiceMeasurement(
            int layer,
            ExpertHistogramSource source,
            uint64_t elapsed_nanoseconds,
            uint64_t activations) noexcept;

        /**
         * @brief Install one complete device-produced cumulative snapshot.
         *
         * The maintenance thread uses this once after a participant graph has
         * release-published device-local timing totals into its mapped page.
         * Every cell is try-owned before any value changes. Existing identical
         * totals make the operation idempotent; different nonempty totals are
         * rejected so host and device evidence can never be silently mixed.
         *
         * @param rows Exact layer-ordered rows for this participant.
         * @param error Optional rejection diagnostic.
         * @return True when the full snapshot is installed or already present.
         */
        [[nodiscard]] bool importServiceMeasurements(
            const std::vector<MoEOverlayParticipantLayerServiceTotals> &rows,
            std::string *error = nullptr);

        /**
         * @brief Try to copy a coherent maintenance-side service snapshot.
         *
         * Each phase/layer cell is acquired at most once. If inference owns any
         * cell, the method clears @p output and returns false immediately; the
         * maintenance worker retries later instead of waiting on inference.
         * Returned rows are ordered by layer.
         *
         * @param output Destination replaced with one row per layer on success.
         * @return True when every cell was copied coherently.
         */
        [[nodiscard]] bool trySnapshotServiceMeasurements(
            std::vector<MoEOverlayParticipantLayerServiceTotals> *output) const;

        /** @return Exact samples dropped rather than blocking inference. */
        [[nodiscard]] uint64_t droppedServiceMeasurementCount() const noexcept
        {
            return dropped_service_measurements_.load(
                std::memory_order_relaxed);
        }

        /**
         * @brief Build an independently mutable candidate cloned from @p epoch.
         * @param previous_epoch Installed source epoch.
         * @param candidate_epoch Strictly newer target epoch.
         * @throws std::out_of_range When the source epoch is not retained.
         * @throws std::invalid_argument When the target epoch is not newer.
         */
        [[nodiscard]] MoEOverlayParticipantResidencyBank cloneCandidate(
            uint64_t previous_epoch,
            uint64_t candidate_epoch) const;

        /**
         * @brief Validate and allocate an immutable bank on maintenance work.
         *
         * This is the deliberately heavy half of publication. It validates all
         * layers and triplets, moves the supplied value into an independently
         * owned node, and returns that node without changing inference-visible
         * state. A transaction prepares every local endpoint before installing
         * any endpoint, so allocation can never lengthen a partial publication.
         *
         * @param bank Mutable candidate value consumed into immutable storage.
         * @param error Optional exact validation/allocation diagnostic.
         * @return Prepared move-only node, or nullopt on invalid input/failure.
         */
        [[nodiscard]] std::optional<MoEOverlayPreparedParticipantBank>
        prepareReadyBank(
            MoEOverlayParticipantResidencyBank bank,
            std::string *error = nullptr) const noexcept;

        /**
         * @brief Atomically install one prebuilt bank without changing routing.
         *
         * The endpoint maintenance mutex serializes only writer lifecycle. The
         * operation itself scans fixed slots, transfers one unique pointer, and
         * release-publishes one raw pointer. It performs no allocation, expert
         * table copy, device work, or inference-reader synchronization.
         * CapacityUnavailable is backpressure: defer the complete closed cycle;
         * never evict a live bank. An identical epoch is idempotent and a
         * different identity under the same epoch is an EpochConflict.
         */
        [[nodiscard]] MoEOverlayParticipantBankInstallStatus installReadyBank(
            MoEOverlayPreparedParticipantBank &&bank,
            std::string *error = nullptr);

        /**
         * @brief Acquire one exact immutable bank for packet execution.
         *
         * This hot-path operation takes no mutex, allocates no storage, and
         * cannot wait for maintenance. The returned lease keeps the exact node
         * alive even when maintenance concurrently removes its publication.
         *
         * @return Reader lease, or an empty lease when the epoch is absent.
         */
        [[nodiscard]] MoEOverlayParticipantBankLease acquire(
            uint64_t epoch) const noexcept;

        /**
         * @brief Remove an unpublished candidate during asynchronous abort.
         * @return Whether the named bank existed and was removed.
         *
         * The global transport calls this only for a candidate epoch that was
         * never exposed through ticket publication.  Ready packet execution
         * retains any previously acquired shared pointer until it returns.
         */
        bool abortUnpublished(uint64_t epoch) noexcept;

        /**
         * @brief Retire one old epoch after its global ticket leases drain.
         * @return Whether the named retained bank was removed.
         */
        bool retire(uint64_t epoch) noexcept;

        /** @return Number of ready epochs currently retained by the endpoint. */
        [[nodiscard]] std::size_t retainedEpochCount() const noexcept;

        /** @return Whether another complete candidate bank can be installed. */
        [[nodiscard]] bool hasCandidateCapacity() const noexcept;

    private:
        /** @brief One allocation-free, try-owned phase/layer accumulation cell. */
        struct ServiceMeasurementCell
        {
            mutable std::atomic_flag owned = ATOMIC_FLAG_INIT;
            uint64_t total_nanoseconds = 0;
            uint64_t activation_count = 0;
            uint64_t sample_count = 0;
            bool overflowed = false;
        };

        /** @brief Flatten one validated layer/phase coordinate. */
        [[nodiscard]] std::size_t serviceMeasurementOffset(
            int layer,
            ExpertHistogramSource source) const noexcept;

        Config config_;
        std::shared_ptr<MoEOverlayParticipantResidencyState> state_;
        std::unique_ptr<ServiceMeasurementCell[]> service_measurements_;
        std::atomic<uint64_t> dropped_service_measurements_{0};
        /** True after device snapshots become this endpoint's sole evidence. */
        std::atomic<bool> device_service_measurements_imported_{false};
    };

    /**
     * @brief Process-local registry of residency endpoints owned by one rank.
     *
     * The global owner map assigns stable logical participant ids independently
     * of rank numbering.  This registry materializes only the participant ids
     * physically hosted by the current process and assembles their epoch-one
     * banks layer by layer as model graph construction resolves prepared engine
     * lifetimes.  Subsequent cached graph variants verify the same identities
     * idempotently instead of creating another residency authority.
     */
    class MoEOverlayParticipantResidencyRegistry final
    {
    public:
        /**
         * @brief One layer selection read from an installed initial bank.
         *
         * This value is setup-only evidence of physical prepared-weight
         * publication. It is deliberately derived from the immutable bank,
         * not reconstructed from the desired owner map, so parity campaigns
         * can distinguish a correct plan from a correctly materialized plan.
         */
        struct InitialBankExpertSelection
        {
            int participant_id = -1;
            DeviceId device = DeviceId::invalid();
            int layer_idx = -1;
            std::vector<int> expert_ids;
        };

        /**
         * @brief Exact process-local initial-bank publication deficit.
         *
         * Missing layers identify graph construction that never supplied a
         * canonical prepared engine set.  An empty @ref missing_layers with
         * @ref publication_pending set means every layer was registered but
         * the endpoint still has no immutable bank for the initial epoch; that
         * state is a publication/lifecycle defect rather than a graph omission.
         */
        struct IncompleteInitialBank
        {
            int participant_id = -1;
            DeviceId device;
            std::vector<int> missing_layers;
            bool publication_pending = false;
        };

        /** @brief Immutable topology and initial publication identity. */
        struct Config
        {
            MoEExpertOwnerMap owner_map;
            std::vector<int> local_participant_ids;
            int num_layers = 0;
            int num_experts = 0;
            uint64_t initial_epoch = 0;
            std::size_t retained_epoch_capacity = 2;
            /** Enable exact service evidence on every process-local endpoint. */
            bool collect_economy_service_measurements = false;
        };

        /**
         * @brief Create every process-local endpoint and its expected initial mask.
         * Relay-only ranks may supply no local participant ids; their registry
         * is vacuously ready while still preserving the process-wide typed
         * topology contract.
         *
         * @throws std::invalid_argument For incomplete geometry or invalid ids.
         */
        explicit MoEOverlayParticipantResidencyRegistry(Config config);

        MoEOverlayParticipantResidencyRegistry(
            const MoEOverlayParticipantResidencyRegistry &) = delete;
        MoEOverlayParticipantResidencyRegistry &operator=(
            const MoEOverlayParticipantResidencyRegistry &) = delete;

        /**
         * @brief Register one graph-resolved initial layer for an endpoint.
         *
         * The supplied mask must equal the canonical owner map. Resident
         * experts require complete shared triplets and non-residents require
         * empty triplets. Once every layer with resident experts is registered,
         * the complete initial bank is installed atomically in that endpoint.
         * Repeated cached-graph registration is accepted only for identical
         * engine identities.
         *
         * @return true on first or identical idempotent registration.
         */
        bool registerInitialLayer(
            int participant_id,
            int layer_idx,
            const std::vector<bool> &resident_mask,
            const std::vector<MoEOverlayPreparedExpertTriplet> &experts,
            std::string *error = nullptr);

        /** @return Process-local endpoint for @p participant_id, or null. */
        [[nodiscard]] std::shared_ptr<MoEOverlayParticipantResidency>
        endpoint(int participant_id) const noexcept;

        /** @return Sorted process-local participant ids. */
        [[nodiscard]] std::vector<int> localParticipantIds() const;

        /**
         * @return Whether every process-local initial bank completed its
         *         one-way publication transition.
         *
         * Completion remains true after maintenance retires the initial epoch;
         * this is setup certification, not a claim that epoch is still live.
         */
        [[nodiscard]] bool allInitialBanksReady() const noexcept;

        /**
         * @brief Snapshot expert selections from every installed initial bank.
         *
         * Results are ordered by participant and then layer. The method is a
         * setup/diagnostic boundary and may allocate; inference never calls it.
         * Each returned selection comes from a retained immutable bank lease,
         * proving that graph construction published complete prepared engines
         * for the corresponding resident mask.
         *
         * @return Process-local physical selections in deterministic order.
         * @throws std::logic_error If any initial bank is not yet installed or
         *         an installed bank no longer has the configured geometry.
         */
        [[nodiscard]] std::vector<InitialBankExpertSelection>
        initialBankExpertSelections() const;

        /**
         * @brief Describe every process-local initial bank that is not ready.
         *
         * The result is sorted by participant id so startup failures remain
         * deterministic across unordered-map implementations.  This method is
         * diagnostic-only and does not publish, repair, or otherwise mutate a
         * partially assembled bank.
         *
         * @return Structured participant and layer deficits.
         */
        [[nodiscard]] std::vector<IncompleteInitialBank>
        incompleteInitialBanks() const;

        /**
         * @brief Install one local participant's immutable device snapshot.
         * @param participant_id Exact process-local participant identity.
         * @param rows Complete layer-ordered cumulative service totals.
         * @param error Optional rejection diagnostic.
         * @return True when the endpoint accepted the complete snapshot.
         */
        [[nodiscard]] bool importDeviceServiceMeasurements(
            int participant_id,
            const std::vector<MoEOverlayParticipantLayerServiceTotals> &rows,
            std::string *error = nullptr);

        /**
         * @brief Try to gather coherent service totals from all local endpoints.
         *
         * The result is sorted by participant then layer. A busy endpoint
         * causes an immediate false result with an empty output; maintenance
         * retries instead of contending with inference.
         */
        [[nodiscard]] bool trySnapshotServiceMeasurements(
            std::vector<MoEOverlayParticipantLayerServiceTotals> *output) const;

        /** @return Canonical initial residency epoch. */
        [[nodiscard]] uint64_t initialEpoch() const noexcept
        {
            return config_.initial_epoch;
        }

    private:
        /**
         * @brief One-way bootstrap publication lifecycle for a local endpoint.
         *
         * `Completed` records that the canonical initial bank was published at
         * least once.  It deliberately remains completed after that epoch is
         * retired: later graph-cache materialization may validate the frozen
         * engine identities, but must never resurrect the bootstrap epoch.
         */
        enum class InitialPublicationState
        {
            Assembling,
            Completed,
        };

        struct EndpointAssembly
        {
            std::shared_ptr<MoEOverlayParticipantResidency> endpoint;
            MoEOverlayParticipantResidencyBank initial_bank;
            std::vector<bool> registered_layers;
            InitialPublicationState publication_state =
                InitialPublicationState::Assembling;
        };

        Config config_;
        mutable std::shared_mutex mutex_;
        std::unordered_map<int, EndpointAssembly> endpoints_;
    };
} // namespace llaminar2
