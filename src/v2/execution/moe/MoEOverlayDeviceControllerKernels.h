/**
 * @file MoEOverlayDeviceControllerKernels.h
 * @brief Backend-neutral launch ABI for the mapped GPU controller lifecycle.
 *
 * The values in this file are embedded into retained CUDA/HIP graphs. They
 * contain only immutable mapped aliases and device-owned policy inputs. No
 * host pointer, callback, rank role inference, or mutable policy mirror is
 * representable through this interface.
 */

#pragma once

#include "DeviceMoEOverlayEpochABI.h"
#include "MoEOverlayDeviceControllerFabricABI.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__)
#define LLAMINAR_MOE_CONTROLLER_HD __host__ __device__ __forceinline__
#elif defined(__HIPCC__)
#define LLAMINAR_MOE_CONTROLLER_HD \
    __host__ __device__ inline __attribute__((always_inline))
#else
#define LLAMINAR_MOE_CONTROLLER_HD inline
#endif

namespace llaminar2
{
    struct DeviceMoEExpertDescriptor;
    struct DeviceMoELayerRuntime;

    /** Binary identity of one device-authored policy result (`MOEP`). */
    inline constexpr std::uint32_t kMoEOverlayDevicePolicyResultMagic =
        0x50454f4du;

    /** Version of @ref MoEOverlayDeviceControllerPolicyResult. */
    inline constexpr std::uint32_t kMoEOverlayDevicePolicyResultVersion = 3u;

    /** One graph-capturable transition in the sole controller state machine. */
    enum class MoEOverlayDeviceControllerAction : std::uint32_t
    {
        Invalid = 0u,
        BeginTransaction = 1u,
        PublishGroupSnapshot = 2u,
        PublishCommand = 3u,
        AcknowledgePrepared = 4u,
        BeginCommit = 5u,
        AcknowledgePublished = 6u,
        PublishAdmission = 7u,
        BeginDynamicRetirement = 8u,
        AcknowledgeRetired = 9u,
        CompleteDynamicRetirement = 10u,
        BeginLLEPRestore = 11u,
        AcknowledgeLLEPRestored = 12u,
        CompleteLLEPRestore = 13u,
        /** Device-author the zero-movement policy used by Static/Observe. */
        AuthorStaticPolicy = 14u,
        /** Acquire the topology-wide Complete publication before graph exit. */
        AwaitTransactionComplete = 15u,
        /** Device-author one bounded two-axis durable Dynamic epoch. */
        AuthorDynamicPolicy = 16u,
        /** Publish completion of this participant's runtime-table pack. */
        PublishParticipantSnapshot = 17u,
        /** Clone/apply one authenticated durable command into the local peer bank. */
        ApplyRuntimeCandidate = 18u,
        /** Publish a prepared local bank and acknowledge its new RCU selector. */
        PublishRuntimeCandidate = 19u,
        /** Acknowledge reader-drained reclamation of the prior local bank. */
        PublishRuntimeRetirement = 20u,
        /** Hold this participant stream until topology-wide commit opens. */
        AwaitRuntimeCommit = 21u,
        /** Hold this participant stream until durable retirement opens. */
        AwaitRuntimeRetirement = 22u,
        /** Publish a device-authenticated prior-bank grace-period receipt. */
        PublishRuntimeRetirementReadiness = 23u,
        /** Complete a device-authored Dynamic decision with no movement. */
        CompleteEmptyDynamicDecision = 24u,
        /** Author cycles back to the immutable loader-prepared owner table. */
        AuthorPreparedContextRestore = 25u,
    };

    /** Exact arithmetic result of adding one phase-pure observation window. */
    struct MoEOverlayDeviceDemandHistoryUpdate
    {
        /** New cumulative count for the phase that supplied this window. */
        std::uint64_t updated_phase_count = 0u;
        /** Sum of the updated phase and the retained opposite phase. */
        std::uint64_t combined_count = 0u;
    };

    /**
     * @brief Convert a typed inference phase to its persistent-history plane.
     * @return Plane zero for prefill, plane one for decode, or the plane count
     *         for an invalid phase.
     */
    LLAMINAR_MOE_CONTROLLER_HD std::uint32_t
    moeOverlayDeviceDemandPhaseIndex(
        MoEOverlayDeviceDemandPhase phase) noexcept
    {
        return phase == MoEOverlayDeviceDemandPhase::Prefill
            ? 0u
            : phase == MoEOverlayDeviceDemandPhase::Decode
            ? 1u
            : kMoEOverlayDeviceControllerDemandPhaseCount;
    }

    /**
     * @brief Locate one `[phase][layer][expert]` history word.
     *
     * Callers validate the phase and geometry once before entering the hot
     * loop. Keeping this arithmetic shared prevents CUDA, HIP, and CPU tests
     * from silently adopting different plane ordering.
     */
    LLAMINAR_MOE_CONTROLLER_HD std::uint64_t
    moeOverlayDeviceDemandHistoryOffset(
        std::uint32_t phase_index,
        std::uint32_t layer,
        std::uint32_t expert,
        std::uint32_t layer_count,
        std::uint32_t expert_count) noexcept
    {
        return static_cast<std::uint64_t>(phase_index) * layer_count *
                   expert_count +
               static_cast<std::uint64_t>(layer) * expert_count + expert;
    }

    /**
     * @brief Add one phase delta without erasing demand learned in the other.
     *
     * Both additions saturate. A stalled maintenance worker can therefore
     * never wrap a very long-lived model's demand history and invert hotness.
     * Publication ordering remains the caller's responsibility because mapped
     * controller pages require backend-specific system-scope stores.
     */
    LLAMINAR_MOE_CONTROLLER_HD MoEOverlayDeviceDemandHistoryUpdate
    moeOverlayAccumulateDeviceDemandHistory(
        std::uint64_t selected_phase_count,
        std::uint64_t retained_other_phase_count,
        std::uint64_t phase_delta) noexcept
    {
        constexpr std::uint64_t maximum = ~std::uint64_t{0};
        const std::uint64_t updated =
            phase_delta > maximum - selected_phase_count
                ? maximum
                : selected_phase_count + phase_delta;
        return {
            .updated_phase_count = updated,
            .combined_count =
                retained_other_phase_count > maximum - updated
                    ? maximum
                    : updated + retained_other_phase_count,
        };
    }

    /** Semantic result of one participant-local command application. */
    enum class MoEOverlayDeviceRuntimeApplyCode : std::uint32_t
    {
        Idle = 0u,
        Success = 1u,
        InvalidBinding = 2u,
        InvalidCommand = 3u,
        InvalidReservation = 4u,
        InvalidRuntime = 5u,
        MissingArrival = 6u,
    };

    /**
     * @brief Device-owned evidence for one complete local inactive-bank build.
     *
     * The result code is written last after a device fence. Later controller
     * actions consume it in stream order; the host may copy it only as terminal
     * diagnostics. No pointer or mutable owner map appears in this record.
     */
    struct alignas(64) MoEOverlayDeviceRuntimeApplyStatus
    {
        std::uint64_t transaction_id = 0u;
        std::uint64_t base_epoch = 0u;
        std::uint64_t candidate_epoch = 0u;
        std::uint32_t code = static_cast<std::uint32_t>(
            MoEOverlayDeviceRuntimeApplyCode::Idle);
        std::uint32_t candidate_bank = kDeviceMoEOverlayInvalidBank;
        std::uint32_t commands_observed = 0u;
        std::uint32_t commands_applied = 0u;
        std::uint32_t changed_layers = 0u;
        std::uint32_t missing_arrivals = 0u;
        std::uint32_t invalid_runtime_layers = 0u;
        std::uint32_t reserved0 = 0u;
        std::uint64_t reserved[1] = {};

        /** @return Whether the complete local candidate was prepared. */
        LLAMINAR_MOE_CONTROLLER_HD bool succeeded() const noexcept
        {
            return code == static_cast<std::uint32_t>(
                               MoEOverlayDeviceRuntimeApplyCode::Success);
        }
    };

    /**
     * @brief Immutable-address participant-local publication resources.
     *
     * `prepared_arrivals[ordinal]` is meaningful only on the command's exact
     * destination participant. The group transport record is the release edge
     * proving those descriptor bytes were uploaded on their transfer streams.
     * Every other placement field is derived by the GPU from the authenticated
     * command and frozen participant metadata.
     */
    struct MoEOverlayDeviceRuntimePublicationBinding
    {
        DeviceMoELayerRuntime *runtime_layers = nullptr;
        DeviceMoEOverlayEpochControl *epoch_control = nullptr;
        /** Controller-owned semantic result for reserve/publish/retire. */
        DeviceMoEOverlayEpochStatus *epoch_status = nullptr;
        const DeviceMoEExpertDescriptor *prepared_arrivals = nullptr;
        MoEOverlayDeviceRuntimeApplyStatus *apply_status = nullptr;
        std::uint32_t arrival_capacity = 0u;
        std::uint32_t layer_count = 0u;
        std::uint32_t expert_count = 0u;
        std::uint32_t domain_participant_id = 0u;
        std::uint32_t domain_participant_count = 0u;

        /** @return Whether every persistent local publication address is usable. */
        LLAMINAR_MOE_CONTROLLER_HD bool valid() const noexcept
        {
            return runtime_layers != nullptr && epoch_control != nullptr &&
                   epoch_status != nullptr &&
                   prepared_arrivals != nullptr && apply_status != nullptr &&
                   arrival_capacity > 0u && layer_count > 0u &&
                   expert_count > 0u &&
                   expert_count <=
                       kMoEOverlayDeviceControllerFabricMaxExperts &&
                   domain_participant_count > 0u &&
                   domain_participant_count <=
                       kMoEOverlayDeviceControllerFabricMaxParticipants &&
                   domain_participant_id < domain_participant_count;
        }
    };

    /** Minimal capture-stable state needed to publish reader-drain readiness. */
    struct MoEOverlayDeviceRetirementReadinessBinding
    {
        DeviceMoEOverlayEpochControl *epoch_control = nullptr;

        /** @return Whether the participant-local reader authority is bound. */
        LLAMINAR_MOE_CONTROLLER_HD bool valid() const noexcept
        {
            return epoch_control != nullptr;
        }
    };

    /**
     * @brief Exact mapped aliases and role of one participant GPU.
     *
     * Every pointer is process-local but addresses the same physical POSIX
     * pages. Offsets in @ref layout and @ref groups remain authoritative, so
     * CUDA and HIP processes may map the channel at unrelated virtual
     * addresses without changing graph semantics.
     */
    struct MoEOverlayDeviceControllerDeviceBinding
    {
        std::byte *mapped_base = nullptr;
        std::uint64_t mapped_bytes = 0u;
        const MoEOverlayDeviceControllerFabricLayoutHeader *layout = nullptr;
        const MoEOverlayDeviceControllerParticipantMetadata *participants =
            nullptr;
        const MoEOverlayDeviceControllerFabricGroupLayout *groups = nullptr;
        MoEOverlayDeviceControllerSharedHeader *controller = nullptr;
        /** Shared exact-epoch barrier used only by continuation-group members. */
        MoEOverlayDeviceControllerInferenceEpochRecord *inference_epoch_record =
            nullptr;
        MoEOverlayDeviceControllerCommandHeader *command = nullptr;
        MoEOverlayDeviceMovementCommand *command_entries = nullptr;
        const std::uint64_t *payload_bytes_per_layer = nullptr;
        /** Immutable setup-time target for reusable-context restoration. */
        const std::uint32_t *initial_owner_participants = nullptr;
        /** Persistent leader-authored history; no host policy can address it. */
        std::uint64_t *demand_history = nullptr;
        /** Host-measured immutable costs acquired before device policy reads. */
        const MoEOverlayDeviceControllerEconomyHeader *economy = nullptr;
        const std::uint64_t *economy_service_costs = nullptr;
        const MoEOverlayDeviceControllerMigrationCost *economy_migration_costs =
            nullptr;
        /** Sole device authority's committed-movement hysteresis history. */
        std::uint64_t *economy_last_moved = nullptr;
        MoEOverlayDeviceControllerGroupRecord *local_group = nullptr;
        MoEOverlayDeviceControllerTransportRecord *local_transport = nullptr;
        MoEOverlayDeviceControllerParticipantRecord *group_participant_records =
            nullptr;
        MoEOverlayDeviceControllerParticipantRecord *local_participant_record =
            nullptr;
        std::uint64_t *group_collected_state = nullptr;
        std::uint64_t *participant_collected_state = nullptr;
        std::uint64_t topology_fingerprint = 0u;
        std::uint32_t participant_id = 0u;
        std::uint32_t group_id = 0u;
        std::uint32_t role_flags = 0u;
        std::uint32_t reserved = 0u;

        /** @return Whether every fixed pointer and identity is present. */
        LLAMINAR_MOE_CONTROLLER_HD bool valid() const noexcept
        {
            return mapped_base != nullptr && mapped_bytes != 0u && layout &&
                   participants && groups && controller &&
                   inference_epoch_record && command &&
                   command_entries && payload_bytes_per_layer &&
                   initial_owner_participants &&
                   demand_history && economy && economy_service_costs &&
                   economy_migration_costs && economy_last_moved &&
                   local_group &&
                   local_transport &&
                   group_participant_records && local_participant_record &&
                   group_collected_state &&
                   participant_collected_state &&
                   topology_fingerprint != 0u;
        }

        /** @return Whether this participant is the sole policy leader. */
        LLAMINAR_MOE_CONTROLLER_HD bool authorityLeader() const noexcept
        {
            return (role_flags & static_cast<std::uint32_t>(
                                     MoEOverlayDeviceControllerParticipantFlags::
                                         AuthorityLeader)) != 0u;
        }

        /** @return Whether this participant owns its group publication lane. */
        LLAMINAR_MOE_CONTROLLER_HD bool groupRoot() const noexcept
        {
            return (role_flags & static_cast<std::uint32_t>(
                                     MoEOverlayDeviceControllerParticipantFlags::
                                         GroupRoot)) != 0u;
        }
    };

    /**
     * @brief Device-authored result consumed by the leader publication action.
     *
     * A Static result must carry zero commands and bytes. Dynamic requires both
     * to be positive. LLEP requires an assignment command but may move no bytes
     * when the selected expert is already resident. The policy kernel writes
     * this record and any command entries before `PublishCommand` executes on
     * the same stream or through an explicit event edge.
     */
    struct alignas(64) MoEOverlayDeviceControllerPolicyResult
    {
        std::uint32_t magic = kMoEOverlayDevicePolicyResultMagic;
        std::uint32_t version = kMoEOverlayDevicePolicyResultVersion;
        std::uint32_t kind = static_cast<std::uint32_t>(
            MoEOverlayDeviceControllerTransactionKind::Invalid);
        std::uint32_t command_count = 0u;
        std::uint64_t command_digest = 0u;
        std::uint64_t packed_weight_bytes = 0u;
        std::uint64_t snapshot_observations = 0u;
        std::uint64_t priority_cost_before = 0u;
        std::uint64_t priority_cost_after = 0u;
        std::uint64_t same_priority_makespan_before = 0u;
        std::uint64_t same_priority_makespan_after = 0u;
        std::uint32_t accepted_cycles = 0u;
        std::uint32_t rejected_cycles = 0u;
        std::uint32_t promotions = 0u;
        std::uint32_t demotions = 0u;
        std::uint32_t same_priority_moves = 0u;
        std::uint32_t changed_layers = 0u;
        /** First layer examined by this fair cyclic policy scan. */
        std::uint32_t layer_scan_start = 0u;
        /** Persistent device cursor published for the next policy wave. */
        std::uint32_t layer_scan_next = 0u;
        std::uint64_t projected_service_gain_ns = 0u;
        std::uint64_t projected_transfer_and_repack_ns = 0u;
        std::uint64_t projected_inference_interference_ns = 0u;
        std::uint64_t projected_net_benefit_ns = 0u;
        std::uint32_t payoff_rejected_cycles = 0u;
        std::uint32_t residency_rejected_cycles = 0u;
        std::uint64_t reserved[1] = {};

        /** @return Whether the producer published a supported exact result. */
        LLAMINAR_MOE_CONTROLLER_HD bool valid() const noexcept
        {
            return magic == kMoEOverlayDevicePolicyResultMagic &&
                   version == kMoEOverlayDevicePolicyResultVersion &&
                   command_digest != 0u &&
                   kind >= static_cast<std::uint32_t>(
                               MoEOverlayDeviceControllerTransactionKind::
                                   StaticCheck) &&
                   kind <= static_cast<std::uint32_t>(
                               MoEOverlayDeviceControllerTransactionKind::
                                   PreparedContextRestore);
        }
    };

    /** @return Seed binding an entry digest to its exact command cardinality. */
    LLAMINAR_MOE_CONTROLLER_HD std::uint64_t
    moeOverlayCommandDigestSeed(std::uint32_t command_count) noexcept
    {
        std::uint64_t value =
            static_cast<std::uint64_t>(command_count) ^
            0x6a09e667f3bcc909ULL;
        value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27u)) * 0x94d049bb133111ebULL;
        value ^= value >> 31u;
        return value == 0u ? 1u : value;
    }

    /** @return Indexed digest contribution for one immutable command word. */
    LLAMINAR_MOE_CONTROLLER_HD std::uint64_t
    moeOverlayCommandDigestWord(
        std::uint64_t word,
        std::uint64_t word_index) noexcept
    {
        std::uint64_t value =
            word ^ (word_index + 0x9e3779b97f4a7c15ULL);
        value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27u)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31u);
    }

    /** @return Whether one movement entry matches its immutable transaction. */
    LLAMINAR_MOE_CONTROLLER_HD bool moeOverlayMovementCommandValid(
        const MoEOverlayDeviceMovementCommand &entry,
        std::uint32_t ordinal,
        MoEOverlayDeviceControllerTransactionKind kind,
        std::uint32_t participant_count,
        std::uint32_t num_layers,
        std::uint32_t num_experts,
        std::uint64_t base_epoch,
        std::uint64_t candidate_epoch) noexcept
    {
        const auto movement_axis =
            static_cast<MoEOverlayDeviceMovementAxis>(entry.flags);
        const bool movement_axis_valid =
            movement_axis == MoEOverlayDeviceMovementAxis::TierResidency ||
            movement_axis ==
                MoEOverlayDeviceMovementAxis::ParticipantPlacement ||
            movement_axis == MoEOverlayDeviceMovementAxis::Combined;
        if (entry.magic != kMoEOverlayDeviceMovementCommandMagic ||
            entry.version != kMoEOverlayDeviceMovementCommandVersion ||
            entry.ordinal != ordinal || entry.layer >= num_layers ||
            entry.expert >= num_experts ||
            entry.source_participant >= participant_count ||
            entry.destination_participant >= participant_count ||
            !movement_axis_valid || entry.source_epoch != base_epoch ||
            entry.candidate_epoch != candidate_epoch)
        {
            return false;
        }

        const auto op = static_cast<MoEOverlayDeviceMovementOp>(entry.op);
        const bool durable_placement =
            kind == MoEOverlayDeviceControllerTransactionKind::
                        DynamicPlacement ||
            kind == MoEOverlayDeviceControllerTransactionKind::
                        PreparedContextRestore;
        const bool physical_movement =
            op == MoEOverlayDeviceMovementOp::DurableMove ||
            op == MoEOverlayDeviceMovementOp::TransientArrival;
        if (physical_movement)
        {
            const bool correct_kind =
                (durable_placement &&
                 op == MoEOverlayDeviceMovementOp::DurableMove) ||
                (kind == MoEOverlayDeviceControllerTransactionKind::
                             CurrentBatchLLEP &&
                 op == MoEOverlayDeviceMovementOp::TransientArrival);
            const bool correct_axis =
                durable_placement ||
                movement_axis ==
                    MoEOverlayDeviceMovementAxis::ParticipantPlacement;
            return correct_kind && correct_axis &&
                   entry.source_participant !=
                       entry.destination_participant &&
                   entry.payload_slot == ordinal &&
                   entry.payload_bytes != 0u && entry.source_epoch != 0u &&
                   ((op == MoEOverlayDeviceMovementOp::DurableMove &&
                     entry.source_epoch != UINT64_MAX &&
                     entry.candidate_epoch == entry.source_epoch + 1u) ||
                    (op == MoEOverlayDeviceMovementOp::TransientArrival &&
                     entry.candidate_epoch == entry.source_epoch));
        }

        return kind ==
                   MoEOverlayDeviceControllerTransactionKind::CurrentBatchLLEP &&
               op == MoEOverlayDeviceMovementOp::TransientAssignment &&
               movement_axis ==
                   MoEOverlayDeviceMovementAxis::ParticipantPlacement &&
               entry.source_participant == entry.destination_participant &&
               entry.payload_slot == kMoEOverlayDeviceInvalidSlot &&
               entry.payload_bytes == 0u && entry.source_epoch != 0u &&
               entry.candidate_epoch == entry.source_epoch;
    }

    /** @return Whether two adjacent entries retain canonical independent order. */
    LLAMINAR_MOE_CONTROLLER_HD bool
    moeOverlayIndependentCanonicalDestination(
        const MoEOverlayDeviceMovementCommand &previous,
        const MoEOverlayDeviceMovementCommand &current) noexcept
    {
        if (previous.destination_participant !=
            current.destination_participant)
        {
            return previous.destination_participant <
                   current.destination_participant;
        }
        if (previous.layer != current.layer)
            return previous.layer < current.layer;

        const bool previous_assignment =
            previous.op == static_cast<std::uint32_t>(
                               MoEOverlayDeviceMovementOp::
                                   TransientAssignment);
        const bool current_assignment =
            current.op == static_cast<std::uint32_t>(
                              MoEOverlayDeviceMovementOp::
                                  TransientAssignment);
        if (previous_assignment != current_assignment)
            return !previous_assignment;
        return previous.expert < current.expert;
    }

    /** Immutable launch value for one controller action kernel. */
    struct MoEOverlayDeviceControllerActionLaunch
    {
        MoEOverlayDeviceControllerDeviceBinding binding;
        MoEOverlayDeviceControllerAction action =
            MoEOverlayDeviceControllerAction::Invalid;
        MoEOverlayDeviceControllerTransactionKind transaction_kind =
            MoEOverlayDeviceControllerTransactionKind::Invalid;
        /** Required only by Dynamic authoring; captured as immutable identity. */
        MoEOverlayDeviceDemandPhase demand_phase =
            MoEOverlayDeviceDemandPhase::Invalid;
        MoEOverlayDeviceControllerPolicyResult *policy_result = nullptr;
        MoEOverlayDeviceRuntimePublicationBinding runtime_publication;
        MoEOverlayDeviceRetirementReadinessBinding retirement_readiness;

        /** @return Whether the action has every statically required binding. */
        LLAMINAR_MOE_CONTROLLER_HD bool valid() const noexcept
        {
            if (!binding.valid() ||
                action == MoEOverlayDeviceControllerAction::Invalid)
            {
                return false;
            }
            if (action ==
                MoEOverlayDeviceControllerAction::BeginTransaction)
            {
                const bool dynamic_phase_valid =
                    transaction_kind !=
                        MoEOverlayDeviceControllerTransactionKind::
                            DynamicPlacement ||
                    demand_phase == MoEOverlayDeviceDemandPhase::Prefill ||
                    demand_phase == MoEOverlayDeviceDemandPhase::Decode;
                return binding.authorityLeader() && dynamic_phase_valid &&
                       transaction_kind >=
                           MoEOverlayDeviceControllerTransactionKind::
                               StaticCheck &&
                       transaction_kind <=
                           MoEOverlayDeviceControllerTransactionKind::
                               PreparedContextRestore;
            }
            if (action == MoEOverlayDeviceControllerAction::PublishCommand ||
                action ==
                    MoEOverlayDeviceControllerAction::AuthorStaticPolicy)
                return binding.authorityLeader() && policy_result != nullptr;
            if (action ==
                MoEOverlayDeviceControllerAction::AuthorDynamicPolicy)
            {
                return binding.authorityLeader() && policy_result != nullptr &&
                       (demand_phase == MoEOverlayDeviceDemandPhase::Prefill ||
                        demand_phase == MoEOverlayDeviceDemandPhase::Decode);
            }
            if (action == MoEOverlayDeviceControllerAction::
                              AuthorPreparedContextRestore)
            {
                return binding.authorityLeader() && policy_result != nullptr &&
                       demand_phase == MoEOverlayDeviceDemandPhase::Invalid;
            }
            if (action ==
                    MoEOverlayDeviceControllerAction::ApplyRuntimeCandidate ||
                action ==
                    MoEOverlayDeviceControllerAction::PublishRuntimeCandidate ||
                action == MoEOverlayDeviceControllerAction::
                              PublishRuntimeRetirement)
            {
                return runtime_publication.valid();
            }
            if (action == MoEOverlayDeviceControllerAction::
                              PublishRuntimeRetirementReadiness)
                return retirement_readiness.valid();
            return true;
        }
    };

    static_assert(std::is_trivially_copyable_v<
                  MoEOverlayDeviceControllerDeviceBinding>);
    static_assert(std::is_trivially_copyable_v<
                  MoEOverlayDeviceControllerPolicyResult>);
    static_assert(std::is_trivially_copyable_v<
                  MoEOverlayDeviceRuntimeApplyStatus>);
    static_assert(std::is_trivially_copyable_v<
                  MoEOverlayDeviceRuntimePublicationBinding>);
    static_assert(std::is_trivially_copyable_v<
                  MoEOverlayDeviceRetirementReadinessBinding>);
    static_assert(sizeof(MoEOverlayDeviceRuntimeApplyStatus) == 64u);
    static_assert(sizeof(MoEOverlayDeviceControllerPolicyResult) == 192u);
    static_assert(std::is_trivially_copyable_v<
                  MoEOverlayDeviceControllerActionLaunch>);
    static_assert(
        sizeof(MoEOverlayDeviceControllerPolicyResult) == 192u,
        "Policy result and measured-economy evidence must occupy exactly three cache lines");
} // namespace llaminar2

#undef LLAMINAR_MOE_CONTROLLER_HD
