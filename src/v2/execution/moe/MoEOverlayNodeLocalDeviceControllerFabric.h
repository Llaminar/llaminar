/**
 * @file MoEOverlayNodeLocalDeviceControllerFabric.h
 * @brief Node-local mapped control fabric for an all-GPU ExpertOverlay.
 *
 * The fabric materializes one topology-wide controller authority across any
 * mixture of local CUDA and ROCm ranks. It owns setup, NUMA first-touch, driver
 * registration, and immutable device aliases only. Policy, histogram reads,
 * command publication, admission, and epoch mutation remain GPU-owned.
 */

#pragma once

#include "DeviceMoERebalancePolicyShared.h"
#include "MoEOverlayDeviceControllerKernels.h"
#include "MoEOverlayDeviceControllerFabricABI.h"
#include "MoEOverlayDeviceControllerTopology.h"
#include "backends/DeviceId.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{
    class IMPIContext;
    class MappedHostTransferRegion;
    struct MoEOverlayCertifiedEconomyProfiles;
    struct MoEOverlayParticipantLayerServiceTotals;

    /** Host-computed, device-consumed immutable mapping geometry. */
    struct MoEOverlayDeviceControllerFabricLayout
    {
        std::size_t page_bytes = 0u;
        std::size_t mapping_bytes = 0u;
        std::size_t setup_header_offset = 0u;
        std::size_t leader_owned_begin = 0u;
        std::size_t leader_owned_end = 0u;
        MoEOverlayDeviceControllerFabricLayoutHeader header;
        std::vector<MoEOverlayDeviceControllerFabricGroupLayout> groups;

        /** @return Whether every region is aligned, bounded, and non-overlapping. */
        [[nodiscard]] bool valid() const noexcept;
    };

    /**
     * @brief Exact captured pointer family for one local GPU participant.
     *
     * The opaque lifetime keeps both the POSIX mapping and every CUDA/HIP host
     * registration alive. Callers receive no host pointer and cannot use this
     * binding to create a host policy shadow.
     */
    struct MoEOverlayDeviceControllerParticipantBinding
    {
        DeviceId device = DeviceId::invalid();
        int participant_id = -1;
        int group_id = -1;
        bool authority_leader = false;
        bool group_root = false;
        /** True only for members of the symmetric continuation epoch barrier. */
        bool inference_epoch_member = false;
        std::byte *mapped_base_device = nullptr;
        std::size_t mapped_bytes = 0u;
        const MoEOverlayDeviceControllerFabricLayoutHeader *layout = nullptr;
        const MoEOverlayDeviceControllerParticipantMetadata *participants = nullptr;
        const MoEOverlayDeviceControllerFabricGroupLayout *groups = nullptr;
        MoEOverlayDeviceControllerSharedHeader *controller = nullptr;
        MoEOverlayDeviceControllerInferenceEpochRecord *inference_epoch_record =
            nullptr;
        MoEOverlayDeviceControllerCommandHeader *command = nullptr;
        MoEOverlayDeviceMovementCommand *command_entries = nullptr;
        const std::uint64_t *payload_bytes_per_layer = nullptr;
        /** Leader-owned phase-separated demand retained across transactions. */
        std::uint64_t *demand_history = nullptr;
        /** Immutable measured economy published by the host evidence owner. */
        const MoEOverlayDeviceControllerEconomyHeader *economy = nullptr;
        const std::uint64_t *economy_service_costs = nullptr;
        const MoEOverlayDeviceControllerMigrationCost *economy_migration_costs =
            nullptr;
        /** Device leader writes committed movement generations only. */
        std::uint64_t *economy_last_moved = nullptr;
        MoEOverlayDeviceControllerGroupRecord *local_group = nullptr;
        /** Physical readiness written by the group-local transport worker. */
        MoEOverlayDeviceControllerTransportRecord *local_transport = nullptr;
        MoEOverlayDeviceControllerParticipantRecord *group_participant_records =
            nullptr;
        MoEOverlayDeviceControllerParticipantRecord *local_participant_record =
            nullptr;
        std::uint64_t *group_collected_state = nullptr;
        std::uint64_t *participant_collected_state = nullptr;
        /** Participant-owned mapped destination for finite service snapshots. */
        MoEOverlayDeviceServiceTelemetryPublicationHeader
            *service_telemetry_publication = nullptr;
        std::shared_ptr<const void> lifetime;

        /** @return Whether all role-independent pointers and identities are bound. */
        [[nodiscard]] bool valid() const noexcept;

        /**
         * @return Trivially copyable aliases suitable for one captured kernel.
         * @throws std::logic_error when this setup binding is incomplete.
         */
        [[nodiscard]] MoEOverlayDeviceControllerDeviceBinding deviceBinding()
            const;
    };

    /**
     * @brief Narrow host view used only to progress one physical transfer wave.
     *
     * Controller, command header, and command entries are const: the worker
     * cannot author policy or publish an epoch. Its sole mutable address is the
     * group-local transport completion record consumed by the group-root GPU.
     * The participant records are a const lifecycle-only view used to delay
     * physical publication/reclamation until every device has crossed the
     * matching RCU edge. Keeping the view typed makes accidental access to
     * histogram pages or participant owner state unrepresentable.
     */
    struct MoEOverlayDeviceControllerTransportBinding
    {
        int group_id = -1;
        int root_world_rank = -1;
        std::uint64_t topology_fingerprint = 0u;
        std::uint32_t command_capacity = 0u;
        const MoEOverlayDeviceControllerFabricLayoutHeader *layout = nullptr;
        const MoEOverlayDeviceControllerSharedHeader *controller = nullptr;
        const MoEOverlayDeviceControllerCommandHeader *command = nullptr;
        const MoEOverlayDeviceMovementCommand *command_entries = nullptr;
        /**
         * Topology-wide lifecycle acknowledgements. These const pointers expose
         * no policy, histogram, or placement payload; the background scheduler
         * uses them only to decide when a bounded retained phase may be
         * submitted without leaving a cross-device wait kernel resident.
         */
        std::array<const MoEOverlayDeviceControllerGroupRecord *,
                   kMoEOverlayDeviceControllerFabricMaxParticipants>
            group_records = {};
        /**
         * Topology-wide participant lifecycle lanes, grouped by controller
         * cell. They let the scheduler prove a complete reader grace period
         * before submitting a bounded retirement graph. No runtime descriptor,
         * histogram, or placement payload is reachable through these aliases.
         */
        std::array<const MoEOverlayDeviceControllerParticipantRecord *,
                   kMoEOverlayDeviceControllerFabricMaxParticipants>
            topology_participant_records = {};
        std::array<std::uint32_t,
                   kMoEOverlayDeviceControllerFabricMaxParticipants>
            topology_participant_record_counts = {};
        std::uint32_t group_record_count = 0u;
        const MoEOverlayDeviceControllerParticipantRecord *
            participant_records = nullptr;
        std::uint32_t participant_record_count = 0u;
        MoEOverlayDeviceControllerTransportRecord *transport = nullptr;
        std::shared_ptr<const void> lifetime;

        /** @return Whether this binding exposes exactly the transport boundary. */
        [[nodiscard]] bool valid() const noexcept;
    };

    /**
     * @brief Model-lifetime mapped controller pages shared across local ranks.
     *
     * Construction is a setup-only collective among ranks that own at least one
     * routed GPU participant. Every owner first-touches its page-isolated write
     * lanes before any process registers the complete mapping with CUDA or HIP.
     * No method performs inference-time polling, copying, or synchronization.
     */
    class MoEOverlayNodeLocalDeviceControllerFabric final
    {
    public:
        /** Immutable topology and maximum command/state geometry. */
        struct Config
        {
            std::shared_ptr<IMPIContext> mpi_ctx;
            std::shared_ptr<const MoEOverlayDeviceControllerTopology> topology;
            std::uint32_t num_layers = 0u;
            std::uint32_t num_experts = 0u;
            /** Model router fan-out used by exact token-horizon economics. */
            std::uint32_t routed_experts_per_token = 1u;
            std::uint32_t command_capacity = 0u;
            std::uint64_t initial_durable_epoch = 1u;
            /** Exact complete packed expert bytes; required by Dynamic policy. */
            std::vector<std::uint64_t> payload_bytes_per_layer;
            /** Observation floor before an epoch is economically meaningful. */
            std::uint64_t minimum_window_activations = 1u;
            /** Tunable complete-cycle budget for one asynchronous epoch wave. */
            std::uint32_t maximum_cycles_per_wave = 1u;
            /** Same-priority skew ratio required before local rebalance. */
            std::uint32_t dynamic_imbalance_threshold_per_mille =
                moe_rebalance_policy::
                    kDefaultDynamicImbalanceThresholdPerMille;
            /** Minimum fractional objective reduction for each cycle. */
            std::uint32_t dynamic_minimum_improvement_per_mille =
                moe_rebalance_policy::
                    kDefaultDynamicMinImprovementPerMille;
            /** Maximum accepted cycles contributed by one layer. */
            std::uint32_t dynamic_maximum_cycles_per_layer =
                moe_rebalance_policy::kDefaultDynamicMaxSwapsPerLayer;
            /** Maximum movement commands in one device-authored wave. */
            std::uint32_t dynamic_maximum_commands_per_wave =
                moe_rebalance_policy::
                    kDefaultDynamicMaxPlanEntriesPerWave;
        };

        /**
         * @brief Compute page ownership and all device-relative offsets.
         * @throws std::invalid_argument for unsupported or overflowing geometry.
         */
        [[nodiscard]] static MoEOverlayDeviceControllerFabricLayout planLayout(
            const MoEOverlayDeviceControllerTopology &topology,
            std::uint32_t num_layers,
            std::uint32_t num_experts,
            std::uint32_t command_capacity,
            std::size_t page_bytes = 4096u,
            const std::vector<std::uint64_t> &payload_bytes_per_layer = {},
            std::uint64_t minimum_window_activations = 1u,
            std::uint32_t maximum_cycles_per_wave = 1u,
            std::uint32_t dynamic_imbalance_threshold_per_mille =
                moe_rebalance_policy::
                    kDefaultDynamicImbalanceThresholdPerMille,
            std::uint32_t dynamic_minimum_improvement_per_mille =
                moe_rebalance_policy::
                    kDefaultDynamicMinImprovementPerMille,
            std::uint32_t dynamic_maximum_cycles_per_layer =
                moe_rebalance_policy::kDefaultDynamicMaxSwapsPerLayer,
            std::uint32_t dynamic_maximum_commands_per_wave =
                moe_rebalance_policy::
                    kDefaultDynamicMaxPlanEntriesPerWave,
            std::uint32_t routed_experts_per_token = 1u);

        /**
         * @brief Create/attach, first-touch, initialize, and register the fabric.
         * @throws std::invalid_argument for malformed topology or local identity.
         * @throws std::runtime_error for mapping, timeout, or registration failure.
         */
        explicit MoEOverlayNodeLocalDeviceControllerFabric(Config config);

        /** Unregister driver pages before unmapping and unlinking the channel. */
        ~MoEOverlayNodeLocalDeviceControllerFabric();

        MoEOverlayNodeLocalDeviceControllerFabric(
            const MoEOverlayNodeLocalDeviceControllerFabric &) = delete;
        MoEOverlayNodeLocalDeviceControllerFabric &operator=(
            const MoEOverlayNodeLocalDeviceControllerFabric &) = delete;

        /** @return Exact immutable page geometry embedded by all local graphs. */
        [[nodiscard]] const MoEOverlayDeviceControllerFabricLayout &layout()
            const noexcept
        {
            return layout_;
        }

        /** @return Local participant ids in canonical dense order. */
        [[nodiscard]] const std::vector<int> &localParticipantIds() const noexcept
        {
            return local_participant_ids_;
        }

        /**
         * @brief Resolve one local participant's exact process-local GPU alias.
         * @throws std::out_of_range when the participant belongs to another rank.
         */
        [[nodiscard]] MoEOverlayDeviceControllerParticipantBinding
        participantBinding(int participant_id) const;

        /**
         * @brief Resolve one group-root host transport completion lane.
         * @param group_id Exact dense group whose root belongs to this rank.
         * @return Const command view plus the sole mutable completion record.
         * @throws std::out_of_range when this rank does not own the group root.
         */
        [[nodiscard]] MoEOverlayDeviceControllerTransportBinding
        transportBinding(int group_id) const;

        /**
         * @brief Observe the number of durably published movement epochs.
         *
         * This diagnostic reads the system-coherent controller word through
         * its mapped host alias and subtracts the immutable initial epoch. It
         * does not wait, copy device state, or participate in placement. The
         * value is suitable for PerfStats and benchmark evidence only; device
         * kernels remain the sole epoch authority.
         *
         * @return Monotonic count of completed Dynamic publications, or zero
         *         before the first publication and for an unavailable mapping.
         */
        [[nodiscard]] std::uint64_t completedDurableMovementEpochs() const
            noexcept;

        /** @return Whether this rank owns the sole host-evidence publication. */
        [[nodiscard]] bool ownsEconomyPublication() const noexcept;

        /**
         * @brief Release-publish one complete measured profile to device policy.
         *
         * Only the rank containing the frozen authority-leader participant may
         * call this method. It writes immutable service/migration arrays into
         * their pre-materialized mapped pages and publishes one release word;
         * it cannot express placement, movement commands, or an epoch.
         *
         * @throws std::logic_error for a non-publisher rank or repeated publish.
         * @throws std::invalid_argument for incomplete/profile-mismatched data.
         */
        void publishCertifiedEconomyProfiles(
            const MoEOverlayCertifiedEconomyProfiles &profiles);

        /** @return Whether a complete economy profile is acquire-visible. */
        [[nodiscard]] bool economyProfilesPublished() const noexcept;

        /**
         * @brief Publish this rank's completed calibration-demand rebase.
         *
         * The caller may invoke this only after every process-local GPU has
         * crossed its exact inference boundary and advanced both phase
         * histogram baselines on device. The mapped publication carries no
         * histogram values, placement, or policy; it is only a topology-wide
         * lifecycle acknowledgement.
         *
         * @throws std::logic_error when the mapping/rank identity is invalid,
         *         the economy profile is absent, or this rank publishes twice.
         */
        void publishLocalDemandHistogramRebase();

        /**
         * @return Whether every participating rank release-published its
         *         completed calibration-demand rebase.
         */
        [[nodiscard]] bool demandHistogramsRebased() const noexcept;

        /**
         * @brief Copy one coherent local participant service publication.
         *
         * The method performs only acquire loads and host memcpy from the
         * already-mapped node-local page. It never launches a GPU operation,
         * waits on a stream, or reads placement state. A generation change
         * clears @p output and returns false so maintenance can retry.
         *
         * @param participant_id Exact process-local participant id.
         * @param output Replaced with one cumulative row per model layer.
         * @param generation Optional acquired publication generation.
         * @return True for one complete nonzero coherent generation.
         */
        [[nodiscard]] bool trySnapshotServiceTelemetry(
            int participant_id,
            std::vector<MoEOverlayParticipantLayerServiceTotals> *output,
            std::uint64_t *generation = nullptr) const noexcept;

        /**
         * @brief Describe the device-owned continuation admission barrier.
         *
         * This failure-only diagnostic reads the system-coherent mapped record
         * through its host alias. It must never be used to plan, admit, or wait
         * for inference: CUDA/HIP remain the sole live controller authority.
         * The snapshot is useful after an existing fatal timeout because it
         * distinguishes an unmatched participant arrival from a later graph
         * edge without adding a D2H transfer or stream operation.
         *
         * @return Compact controller/barrier lifecycle evidence.
         */
        [[nodiscard]] std::string describeInferenceEpochBarrier() const;

        /** @return Run-scoped POSIX name for setup diagnostics only. */
        [[nodiscard]] const std::string &channelName() const noexcept
        {
            return channel_name_;
        }

    private:
        struct MappingLifetime;

        /** Create or attach the run-scoped POSIX mapping. */
        void mapOrAttach();
        /** First-touch only ranges owned by this world rank. */
        void placeOwnedPages();
        /** Publish immutable records after every owner has placed its pages. */
        void initializeControllerRecords();
        /** Register the mapping with every process-local participant GPU. */
        void registerLocalDevices();
        /** Wait until every participating rank publishes one setup generation. */
        void waitForRankPublications(
            const std::uint64_t *publications,
            std::uint64_t expected,
            const char *description) const;
        /** Latch a setup failure for peers before propagating an exception. */
        void publishSetupError(
            MoEOverlayDeviceControllerFabricSetupError error) noexcept;

        Config config_;
        MoEOverlayDeviceControllerFabricLayout layout_;
        std::vector<int> participating_world_ranks_;
        std::vector<int> local_participant_ids_;
        std::vector<DeviceId> local_devices_;
        std::string channel_name_;
        bool creator_ = false;
        std::shared_ptr<MappingLifetime> mapping_lifetime_;
        std::shared_ptr<MappedHostTransferRegion> mapped_region_;
    };
} // namespace llaminar2
