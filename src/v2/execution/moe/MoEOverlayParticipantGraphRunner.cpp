/**
 * @file MoEOverlayParticipantGraphRunner.cpp
 * @brief Real-weight expert-only graph execution for remote overlay ranks.
 *
 * The dense continuation rank and every auxiliary ExpertOverlay rank enter
 * one ordered sparse-collective protocol.  This file owns the auxiliary side:
 * it prepares only that rank's real GGUF expert weights, constructs one
 * fixed-capacity participant-local dispatch/compute/return graph, and retains the
 * stable resources those graphs reference.  In particular, compact routed-row
 * tensors are allocated once per logical participant at planner-admitted
 * capacity.  They are shared only by graph cache entries that this
 * single-threaded @ref IInferenceRunner contract proves are serial; two
 * participants, even two CPU/NUMA endpoints with the same backend name, never
 * alias their compact packets.
 *
 * The surrounding orchestration layer is responsible for segmenting a request
 * before it exceeds @c max_graph_activation_rows.  This runner refuses to
 * resize a captured/manual-boundary graph's storage at execution time, keeping
 * graph identity and memory admission authoritative.
 */

#include "MoEOverlayParticipantGraphRunner.h"
#include "MoEOverlayParticipantResidency.h"

#include "backends/GPUDeviceContextPool.h"
#include "backends/BackendManager.h"
#include "collective/CollectiveTimeoutPolicy.h"
#include "execution/compute_stages/ComputeStageFactory.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/compute_stages/stages/MoELocalExpertStage.h"
#include "execution/compute_stages/stages/MoEOverlayActivationPacketStages.h"
#include "execution/compute_stages/stages/MoEOverlayEpochBoundaryStage.h"
#include "execution/compute_stages/stages/MoERankBatchSparseStages.h"
#include "execution/compute_stages/stages/MoESparseDispatchStage.h"
#include "execution/compute_stages/stages/MoESparseReturnReduceStage.h"
#include "execution/local_execution/device/WorkspaceAllocator.h"
#include "execution/local_execution/engine/PrefillBucketUtils.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/DeviceExecutionTimeline.h"
#include "execution/moe/DeviceMoEExpertDescriptorBuilder.h"
#include "execution/moe/DeviceMoEOverlayEpochArena.h"
#include "execution/moe/MoEOverlayNodeLocalDeviceControllerFabric.h"
#include "execution/moe/MoEOverlayNodeLocalRankBatchTransport.h"
#include "execution/moe/MoEOverlayInferenceInterferenceProbe.h"
#include "execution/moe/MoEOverlayActivationRendezvousDeadline.h"
#include "execution/moe/MoEOverlayRankBatchTransport.h"
#include "execution/moe/MoEOverlaySparseCollective.h"
#include "execution/moe/MoEOverlayResidencyAuthority.h"
#include "execution/moe/MoERuntimeTable.h"
#include "loaders/ModelContext.h"
#include "loaders/PreparedWeightStore.h"
#include "loaders/WeightManager.h"
#include "memory/BufferArena.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "tensors/Tensors.h"
#include "transfer/MappedTransferProgressEpoch.h"
#include "transfer/TransferEngine.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"
#include "utils/WeightLoadingProfiler.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace llaminar2
{
    namespace
    {
        /**
         * @brief Describe one routed-expert parent tensor and its semantic role.
         */
        struct ExpertParentDescriptor
        {
            const char *suffix = nullptr;
            WeightRole role = WeightRole::Other;
        };

        constexpr std::array<ExpertParentDescriptor, 3>
            kExpertParents{{
                {"ffn_gate_exps.weight", WeightRole::MoEExpertGate},
                {"ffn_up_exps.weight", WeightRole::MoEExpertUp},
                {"ffn_down_exps.weight", WeightRole::MoEExpertDown},
            }};

        /** @brief Build the canonical GGUF tensor name for one expert parent. */
        std::string parentName(int layer, const char *suffix)
        {
            return "blk." + std::to_string(layer) + "." + suffix;
        }

        /** @brief Return true when at least one expert bit is owned. */
        bool hasActiveMask(const std::vector<bool> &mask)
        {
            return std::any_of(
                mask.begin(), mask.end(), [](bool active) { return active; });
        }

        /**
         * @brief Multiply two arena geometry terms while rejecting overflow.
         *
         * Persistent graph buffers are part of capacity admission and capture
         * identity, so wrapping their geometry would bind a smaller allocation
         * than the captured kernels address.  Reject that configuration before
         * allocating or materializing any participant graph.
         *
         * @param lhs Left-hand geometry term.
         * @param rhs Right-hand geometry term.
         * @param what Human-readable allocation component for the diagnostic.
         * @return The exact product.
         * @throws std::overflow_error When the product cannot fit in `size_t`.
         */
        size_t checkedGeometryProduct(
            size_t lhs,
            size_t rhs,
            const char *what)
        {
            if (lhs != 0u && rhs > std::numeric_limits<size_t>::max() / lhs)
            {
                throw std::overflow_error(
                    std::string("MoE overlay participant geometry overflows: ") +
                    what);
            }
            return lhs * rhs;
        }

        /**
         * @brief Resolve the worker stream used by the normal graph executor.
         *
         * Preparing persistent sparse buffers on this same stream creates an
         * explicit producer edge and prevents preparation from guessing a
         * backend default stream.
         */
        IWorkerGPUContext *participantWorkerContext(DeviceId device)
        {
            if (!device.is_gpu())
                return nullptr;
            auto &worker =
                GPUDeviceContextPool::instance().getContext(device);
            if (!worker.defaultStream())
            {
                throw std::runtime_error(
                    "MoE overlay participant GPU worker has no exact stream for " +
                    device.to_string());
            }
            return &worker;
        }

        /** @brief Return the non-default producer stream owned by the worker context. */
        void *participantWorkerStream(DeviceId device)
        {
            IWorkerGPUContext *const worker =
                participantWorkerContext(device);
            return worker ? worker->defaultStream() : nullptr;
        }

        /** @brief Convert a local rank role to the weight residency audit role. */
        WeightResidencyCategory participantResidencyCategory(
            const OverlayRankPlan &rank,
            DeviceId device)
        {
            if (device.is_gpu())
                return WeightResidencyCategory::AcceleratorRoutedExpert;
            if (rank.loads_worker_fallback_experts)
                return WeightResidencyCategory::WorkerFallbackExpert;
            return WeightResidencyCategory::CpuFallbackExpert;
        }

        /** @brief Render a compact comma-separated participant id list. */
        std::string participantIdList(
            const std::vector<const MoEExpertOwnerParticipant *> &participants)
        {
            std::string result;
            for (const auto *participant : participants)
            {
                if (!participant)
                    continue;
                if (!result.empty())
                    result += ",";
                result += std::to_string(participant->participant_id);
            }
            return result.empty() ? "<relay>" : result;
        }

        /** @brief Return unique rank-local devices in participant order. */
        std::vector<DeviceId> participantDevices(
            const std::vector<const MoEExpertOwnerParticipant *> &participants)
        {
            std::vector<DeviceId> devices;
            for (const auto *participant : participants)
            {
                if (!participant ||
                    std::find(
                        devices.begin(), devices.end(),
                        participant->device) != devices.end())
                {
                    continue;
                }
                devices.push_back(participant->device);
            }
            return devices;
        }

        /** @brief Render all rank-local device ids for diagnostics. */
        std::string participantDeviceList(
            const std::vector<const MoEExpertOwnerParticipant *> &participants)
        {
            std::string result;
            for (const DeviceId device : participantDevices(participants))
            {
                if (!result.empty())
                    result += ",";
                result += device.to_string();
            }
            return result.empty() ? "CPU" : result;
        }
    } // namespace

    struct MoEOverlayParticipantGraphRunner::CachedParticipantGraph
    {
        /**
         * @brief Scheduler authority shared by every physical-row specialization.
         *
         * One mapped control belongs to `(participant, graph family)`, not to
         * a captured row geometry. Keeping the monotonic generation here makes
         * switching between decode, verifier, and prefill executables safe.
         */
        struct MappedLaneAuthority
        {
            std::shared_ptr<IMoEOverlayRankBatchTransport> transport_lifetime;
            std::unique_ptr<MoEOverlayActivationEpochProtocol> protocol;
            int participant_id = -1;
            int tier_index = -1;
            int tier_priority = 0;
            int routed_domain_ordinal = -1;
            DeviceId device = DeviceId::invalid();
            size_t graph_family_ordinal = 0u;
            uint64_t next_epoch_generation = 1u;
        };

        /** @brief One complete retained follower parent on one physical GPU. */
        struct MappedGPUFollowerEndpoint
        {
            int participant_id = -1;
            DeviceId device = DeviceId::invalid();
            int physical_rows = 0;
            std::unique_ptr<ComputeGraph> graph;
            std::unique_ptr<BufferArena> arena;
            /**
             * @brief Device-family workspace shared by mutually exclusive shapes.
             *
             * Every physical-row graph for one device is submitted on the
             * same participant stream, so their scratch lifetimes never
             * overlap. Keeping this owner shared makes the runtime allocation
             * match the serial-family BOM instead of reserving one complete
             * MoE workspace for every prefill bucket.
             */
            std::shared_ptr<WorkspaceAllocator> workspace_allocator;
            std::unique_ptr<DeviceGraphExecutor> executor;
            std::shared_ptr<MappedLaneAuthority> lane_authority;
            /** Exact capture-stable lane, retained for terminal diagnostics. */
            MoEOverlayMappedActivationDeviceLane lane;
            std::shared_ptr<INT32Tensor> active_rows;
            std::vector<std::shared_ptr<TensorBase>> tensor_lifetimes;
            DeviceGraphExecutor::GraphSegmentCache cache;
        };

        /**
         * @brief One CPU layer endpoint bound directly to mapped packet pages.
         *
         * CPU is an intentional heterogeneous boundary and therefore cannot be
         * embedded in a native CUDA/HIP parent. The layer nevertheless shares
         * the same authenticated epoch and packet storage as its GPU siblings;
         * it never stages, schedules, or otherwise mediates their traffic.
         */
        struct MappedCPUFollowerLayer
        {
            int model_layer_index = -1;
            std::uint32_t stage_ordinal = 0u;
            std::unique_ptr<MoELocalExpertStage> local_expert;
        };

        /** @brief One serial host follower for a planner-declared CPU endpoint. */
        struct MappedCPUFollowerEndpoint
        {
            int participant_id = -1;
            DeviceId device = DeviceId::invalid();
            int tier_index = -1;
            int routed_domain_ordinal = -1;
            std::shared_ptr<MappedLaneAuthority> lane_authority;
            /** Exact CPU alias set used to bind geometry-selected payload views. */
            MoEOverlayMappedActivationDeviceLane lane;
            std::shared_ptr<MoEOverlaySparseRows> input_rows;
            std::shared_ptr<MoEOverlayReturnRows> output_rows;
            std::vector<MappedCPUFollowerLayer> layers;
        };

        /** @brief Concurrent endpoint set for one exact ticket row geometry. */
        struct MappedGPUFollowerShape
        {
            int physical_rows = 0;
            std::vector<std::unique_ptr<MappedGPUFollowerEndpoint>> endpoints;
        };

        std::unique_ptr<ComputeGraph> graph;
        DeviceGraphExecutor::RetainedMultiDeviceExecutionPlan
            execution_plan; ///< Setup-resolved immutable heterogeneous host schedule.
        /**
         * Runner-owned workspace shared by all mutually exclusive graph
         * families.  Keeping a shared lifetime here ensures cached stages lose
         * their embedded pointers before the final runner owner releases the
         * physical block.
         */
        std::shared_ptr<WorkspaceAllocator> workspace_allocator;
        std::vector<std::shared_ptr<TensorBase>> tensor_lifetimes;
        ParticipantGraphFamilyRole role = ParticipantGraphFamilyRole::Main;
        int mtp_graph_depth = -1;
        int row_capacity = 0;
        std::vector<int> source_layers;
        std::vector<std::shared_ptr<MappedLaneAuthority>> mapped_lane_authorities;
        std::vector<MappedGPUFollowerShape> mapped_gpu_shapes;
        std::vector<std::unique_ptr<MappedCPUFollowerEndpoint>>
            mapped_cpu_endpoints;

        /** @return Whether this family uses authenticated mapped activation epochs. */
        [[nodiscard]] bool usesMappedActivationEpochs() const noexcept
        {
            return !mapped_lane_authorities.empty();
        }

        /** @return Exact captured row specialization, or null if not planned. */
        [[nodiscard]] MappedGPUFollowerShape *mappedShapeForRows(
            int physical_rows) noexcept
        {
            const auto found = std::find_if(
                mapped_gpu_shapes.begin(),
                mapped_gpu_shapes.end(),
                [physical_rows](const auto &shape)
                {
                    return shape.physical_rows == physical_rows;
                });
            return found == mapped_gpu_shapes.end() ? nullptr : &*found;
        }
    };

    /**
     * @brief Durable execution state and inference boundary for one follower GPU.
     *
     * Every retained row shape and graph family for one participant shares the
     * same runtime table, epoch ticket, route scratch, and exact worker stream.
     * The placement-policy authority is orthogonal: a heterogeneous host
     * controller publishes this device state through its retained publisher,
     * while an all-GPU controller additionally supplies a mapped fabric
     * binding. A mapped follower has already observed its exact terminal event
     * before its MPI command can complete, so it publishes an immutable
     * generation receipt instead of recording a late event on a mutable shared
     * inference stream.
     */
    struct MoEOverlayParticipantGraphRunner::ParticipantGpuRuntime final
        : public IMoEOverlayDeviceInferenceBoundary,
          public IMoEOverlayDeviceInitialRuntimePublisher
    {
        int participant_id = -1;
        DeviceId device = DeviceId::invalid();
        std::uint32_t domain_participant_id = 0u;
        std::uint32_t domain_participant_count = 0u;
        /** Frozen policy-authority locus; execution state is always on device. */
        MoEOverlayAuthorityExecutionKind authority_execution =
            MoEOverlayAuthorityExecutionKind::Unresolved;
        /** Device-policy receipt lane; absent for host-resident authority. */
        std::optional<MoEOverlayDeviceControllerParticipantBinding>
            controller_binding;
        IWorkerGPUContext *worker = nullptr;
        std::shared_ptr<DeviceMoESerialRouteScratchArena> route_scratch;
        std::shared_ptr<DeviceMoEOverlayEpochArena> epoch_arena;
        std::unique_ptr<DeviceMoERuntimeTable> runtime_table;
        std::vector<std::uint8_t> initialized_layers;
        /** Exact stream on which graph construction published initial banks. */
        void *runtime_publication_stream = nullptr;
        /** Preallocated join from graph-build publication to controller work. */
        std::shared_ptr<void> initial_runtime_source_ready_event;
        /** Durable completion edge for an idempotent replacement controller. */
        std::shared_ptr<void> initial_runtime_published_event;
        void *initial_runtime_finalization_stream = nullptr;
        bool initial_runtime_published = false;
        /** Shared finite relay graph launched ahead of follower inference. */
        std::shared_ptr<MappedTransferProgressEpoch>
            transfer_progress_epoch;

        /** Exact serial submission/completion receipt for this follower. */
        MoEOverlayInferenceBoundaryReceipt inference_boundary_receipt;

        /** @copydoc IMoEOverlayDeviceInitialRuntimePublisher::publishMoEOverlayDeviceInitialRuntime */
        [[nodiscard]] bool publishMoEOverlayDeviceInitialRuntime(
            void *controller_stream) override
        {
            if (!device.is_gpu() || !worker || !runtime_table ||
                !runtime_publication_stream || !controller_stream ||
                !initial_runtime_source_ready_event ||
                !initial_runtime_published_event ||
                initialized_layers.size() !=
                    static_cast<std::size_t>(runtime_table->layerCount()) ||
                std::any_of(
                    initialized_layers.begin(),
                    initialized_layers.end(),
                    [](std::uint8_t initialized)
                    {
                        return initialized == 0u;
                    }))
            {
                LOG_ERROR(
                    "[MoEOverlayParticipantGraphRunner] Follower initial "
                    "runtime publication is incomplete for participant "
                    << participant_id);
                return false;
            }

            IBackend *const backend = getBackendFor(device);
            if (!backend)
                return false;
            const auto edge = DeviceEventEdge::at(
                                  DeviceTimelinePoint::
                                      MoEOverlayInitialRuntimeReady)
                                  .from(
                                      DeviceTimelineRole::
                                          MoEOverlayRuntimePublication);
            if (initial_runtime_published)
            {
                return initial_runtime_finalization_stream &&
                    edge.to(DeviceTimelineRole::MoERebalanceMaintenance)
                        .enqueueWait(
                            *backend,
                            device,
                            initial_runtime_published_event.get(),
                            initial_runtime_finalization_stream,
                            controller_stream);
            }

            if (!edge.publish(
                    *backend,
                    device,
                    initial_runtime_source_ready_event.get(),
                    runtime_publication_stream) ||
                !edge.to(DeviceTimelineRole::MoEOverlayRuntimePublication)
                     .enqueueWait(
                         *backend,
                         device,
                         initial_runtime_source_ready_event.get(),
                         runtime_publication_stream,
                         controller_stream) ||
                !edge.publish(
                    *backend,
                    device,
                    initial_runtime_published_event.get(),
                    controller_stream))
            {
                LOG_ERROR(
                    "[MoEOverlayParticipantGraphRunner] Follower could not "
                    "order its complete initial runtime onto the controller stream for participant "
                    << participant_id);
                return false;
            }

            initial_runtime_finalization_stream = controller_stream;
            initial_runtime_published = true;
            PerfStatsCollector::addCounter(
                "moe_overlay_controller",
                "initial_runtime_publications",
                1.0,
                "model_setup",
                device.toString(),
                {{"participant", std::to_string(participant_id)},
                 {"source", "mapped_follower_graph_family"},
                 {"ordering", "event_wait"},
                 {"blocking", "false"}});
            return true;
        }

        /** @copydoc IMoEOverlayDeviceInferenceBoundary::enqueueMoEOverlayDeviceInferenceBoundary */
        MoEOverlayInferenceBoundaryStatus
        enqueueMoEOverlayDeviceInferenceBoundary(
            void *maintenance_stream,
            MoEOverlayInferenceBoundaryRequest request) override
        {
            if (!device.is_gpu() || !worker || !maintenance_stream)
            {
                LOG_ERROR(
                    "[MoEOverlayParticipantGraphRunner] Follower controller "
                    "boundary has no exact GPU worker or maintenance stream for participant "
                    << participant_id);
                return MoEOverlayInferenceBoundaryStatus::Failed;
            }

            const auto receipt =
                inference_boundary_receipt.consumeLatestCompleted();
            if (receipt.newerSubmissionInFlight())
            {
                /*
                 * The participant-local RCU bank can overlap physical
                 * preparation with a reader, but the topology-wide decision
                 * graph cannot begin from a completed generation on one GPU
                 * while another GPU is still executing a newer sparse
                 * transaction. The leader's mapped fan-in waits can otherwise
                 * enter a cycle with that transaction's dispatch/return waits.
                 * Defer only maintenance; the inference producer remains
                 * completely untouched and publishes the exact receipt when
                 * its retained terminal is observed.
                 */
                PerfStatsCollector::addCounter(
                    "moe_overlay_controller",
                    "follower_inference_boundary_deferrals",
                    1.0,
                    "maintenance",
                    device.toString(),
                    {{"participant", std::to_string(participant_id)},
                     {"boundary", request.name()},
                     {"submitted_generation",
                      std::to_string(receipt.submitted_generation)},
                     {"completed_generation",
                      std::to_string(receipt.completed_generation)},
                     {"reason", "latest_sparse_transaction_in_flight"},
                     {"blocking_inference", "false"}});
                return MoEOverlayInferenceBoundaryStatus::Deferred;
            }
            if (!receipt.hasCompletedBoundary())
            {
                /* No request has acquired this participant's reader yet. */
                PerfStatsCollector::addCounter(
                    "moe_overlay_controller",
                    "follower_inference_boundaries_without_prior_reader",
                    1.0,
                    "maintenance",
                    device.toString(),
                    {{"participant", std::to_string(participant_id)},
                     {"blocking", "false"}});
                return MoEOverlayInferenceBoundaryStatus::Submitted;
            }

            /*
             * The completed receipt is stronger than another stream event: the
             * fixed transaction terminal was already queried successfully and
             * its captured release kernel has finished.  The generation check
             * above also proves that no newer follower transaction is live.
             * Never append a fresh event to the shared follower stream here,
             * because that event could retarget the completed boundary to work
             * submitted after this maintenance epoch was admitted.
             */

            PerfStatsCollector::addCounter(
                "moe_overlay_controller",
                "follower_inference_boundary_receipts_consumed",
                1.0,
                "maintenance",
                device.toString(),
                {{"participant", std::to_string(participant_id)},
                 {"boundary", request.name()},
                 {"completed_generation",
                  std::to_string(receipt.completed_generation)},
                 {"fresh_completion",
                  receipt.fresh_completion ? "true" : "false"},
                 {"newer_submission_in_flight",
                  "false"},
                 {"stream_edge", "completed_transaction_receipt"},
                 {"blocking", "false"},
                 {"host_epoch_mirror", "false"}});
            return MoEOverlayInferenceBoundaryStatus::Submitted;
        }

        /** @copydoc IMoEOverlayDeviceInferenceBoundary::installMoEOverlayTransferProgressEpoch */
        bool installMoEOverlayTransferProgressEpoch(
            std::shared_ptr<MappedTransferProgressEpoch> epoch) override
        {
            if (!epoch || !device.is_gpu() || epoch->device() != device)
            {
                LOG_ERROR(
                    "[MoEOverlayParticipantGraphRunner] Rejected a retained transfer epoch with mismatched follower ownership for participant "
                    << participant_id);
                return false;
            }
            if (transfer_progress_epoch && transfer_progress_epoch != epoch)
            {
                LOG_ERROR(
                    "[MoEOverlayParticipantGraphRunner] Rejected replacement of an installed retained transfer epoch for participant "
                    << participant_id);
                return false;
            }
            transfer_progress_epoch = std::move(epoch);
            return true;
        }
    };

    WeightPlan buildMoEOverlayParticipantWeightPlan(
        const ModelContext &model_context,
        const MoEExpertOverlayExecutionPlan &execution_plan,
        const MoEExpertOwnerMap &owner_map,
        const MoEExpertOwnerParticipant &participant)
    {
        InferenceStrategy strategy;
        strategy.mode = WeightInferenceMode::SingleDevice;
        strategy.model_id = ModelContextId{
            reinterpret_cast<uint64_t>(&model_context)};
        strategy.devices = {participant.device};

        WeightPlan plan(
            strategy,
            PhysicalMemoryOwner::RoutedExpertWeights);
        const auto &loader = model_context.concreteLoader();
        const auto &rank = execution_plan.currentRankPlan();
        const auto residency =
            participantResidencyCategory(rank, participant.device);

        for (int layer = 0; layer < model_context.totalBlockCount(); ++layer)
        {
            std::vector<int> experts = owner_map.expertsForParticipant(
                layer, participant.participant_id);
            std::sort(experts.begin(), experts.end());
            experts.erase(
                std::unique(experts.begin(), experts.end()), experts.end());
            if (experts.empty())
                continue;

            for (const auto &parent : kExpertParents)
            {
                const std::string name = parentName(layer, parent.suffix);
                const auto shape = loader.getTensorShape(name);
                if (!shape || shape->size() != 3u)
                {
                    throw std::runtime_error(
                        "MoE overlay participant weight plan requires a 3-D "
                        "expert parent: " +
                        name);
                }
                if (std::any_of(
                        experts.begin(), experts.end(),
                        [expert_count = (*shape)[2]](int expert)
                        {
                            return expert < 0 ||
                                   static_cast<size_t>(expert) >= expert_count;
                        }))
                {
                    throw std::runtime_error(
                        "MoE overlay participant owner map names an expert "
                        "outside tensor geometry for " +
                        name);
                }

                WeightRequirement requirement;
                requirement.canonical_name = name;
                requirement.required = true;
                requirement.role = parent.role;
                requirement.derivation = WeightDerivationKind::ExpertSlice;
                requirement.layer = layer;
                requirement.target_device = participant.device;
                requirement.lookup_device = DeviceId::cpu();
                requirement.bypass_tensor_parallel = true;
                requirement.residency_category = residency;
                requirement.overlay_domain = participant.domain_name;
                requirement.overlay_participant_index =
                    participant.domain_participant_index;
                requirement.overlay_participant_world_rank =
                    participant.world_rank_known
                        ? participant.world_rank
                        : -1;
                requirement.host_policy = participant.device.is_gpu()
                                              ? WeightHostPolicy::
                                                    RequiredUntilPreparedOrTransferred
                                              : WeightHostPolicy::
                                                    RequiredForCPUExecution;
                requirement.expected_prepared_kind =
                    PreparedWeightKind::MoeExpertSlab;
                requirement.slice.source_rows = (*shape)[0];
                requirement.slice.source_cols = (*shape)[1];
                requirement.slice.row_count = (*shape)[0];
                requirement.slice.col_count = (*shape)[1];
                requirement.slice.expert_start =
                    static_cast<size_t>(experts.front());
                requirement.slice.expert_count = experts.size();
                requirement.slice.expert_ids = experts;
                requirement.slice.inner_is_presliced = true;
                plan.add(std::move(requirement));
            }
        }

        /*
         * Current residency is data, not graph topology. An automatically
         * filled higher-priority tier may leave this declared endpoint empty
         * at epoch one, while a later economical promotion/demotion can still
         * publish experts into its preallocated shadow bank. Return a valid
         * empty plan so the caller can retain the endpoint graph without
         * materializing nonexistent initial payloads.
         */
        return plan;
    }

    MoEOverlayParticipantGraphRunner::MoEOverlayParticipantGraphRunner(
        MoEOverlayParticipantGraphRunnerConfig config)
        : config_(std::move(config))
    {
        if (!config_.model_context || !config_.mpi_context ||
            !config_.placement_plan)
        {
            throw std::invalid_argument(
                "MoE overlay participant runner requires model, MPI, and "
                "placement authorities");
        }
        if (!config_.placement_plan->usesExpertOverlayAuthority())
        {
            throw std::invalid_argument(
                "MoE overlay participant runner requires a tiered overlay plan");
        }
        if (config_.max_seq_len <= 0)
        {
            throw std::invalid_argument(
                "MoE overlay participant runner requires a positive maximum "
                "sequence length");
        }
        if (config_.max_graph_activation_rows <= 0)
        {
            throw std::invalid_argument(
                "MoE overlay participant runner requires a positive planner-admitted "
                "graph activation capacity");
        }
        const auto normalized_prefill_shapes =
            normalizePrefillGraphBuckets(config_.prefill_graph_row_shapes);
        if (normalized_prefill_shapes != config_.prefill_graph_row_shapes ||
            normalized_prefill_shapes.empty() ||
            normalized_prefill_shapes.back() >
                config_.max_graph_activation_rows)
        {
            throw std::invalid_argument(
                "MoE overlay participant runner requires the exact sorted "
                "prefill-row manifest admitted by the distributed schedule");
        }
        if (config_.max_decode_activation_rows <= 0 ||
            config_.max_decode_activation_rows >
                config_.max_graph_activation_rows)
        {
            throw std::invalid_argument(
                "MoE overlay participant runner decode/MTP row capacity must "
                "be positive and fit the planner-admitted graph capacity");
        }
        const bool retains_mtp_graph_family =
            config_.mtp_graph_family_policy ==
            MoEOverlayMTPGraphFamilyPolicy::RetainModelSidecars;
        if (retains_mtp_graph_family &&
            config_.max_mtp_draft_depth <= 0)
        {
            throw std::invalid_argument(
                "MTP-capable participant runner requires a positive admitted draft depth");
        }
        if (!retains_mtp_graph_family &&
            config_.max_mtp_draft_depth != 0)
        {
            throw std::invalid_argument(
                "Main-only participant runner cannot advertise an MTP draft capacity");
        }
        if (config_.max_request_count <= 0)
        {
            throw std::invalid_argument(
                "Participant runner requires a positive transaction request capacity");
        }

        architecture_ = config_.model_context->architecture();
        residency_authority_ = config_.residency_authority;
        decode_histogram_ = config_.decode_histogram;
        if (!residency_authority_ || !config_.participant_residency)
        {
            throw std::invalid_argument(
                "Production MoE overlay participant runner requires the shared "
                "residency authority and process-local prepared-bank registry");
        }
        {
            ScopedWeightLoadDetailTimer timer(
                "overlay.graph.resolve_topology");
            resolveTopology();
        }
        {
            ScopedWeightLoadDetailTimer timer(
                "overlay.graph.resolve_families");
            resolveGraphFamilies();
        }
        {
            /*
             * Every native follower graph on one GPU draws from the same
             * opaque CUDA/HIP pool. Commit the complete admitted family before
             * creating contexts or graph owners; an instantiation-time memory
             * delta is evidence about this pool and cannot become a second
             * per-executable accounting authority.
             */
            const auto weight_manager =
                config_.model_context->concreteWeightManager();
            const auto memory_authority = weight_manager
                                              ? weight_manager
                                                    ->physicalMemoryAuthority()
                                              : nullptr;
            if (!memory_authority)
            {
                throw std::logic_error(
                    "Participant graph has no canonical physical-memory authority for native graph-pool admission");
            }
            for (const DeviceId device :
                 participantDevices(local_participants_))
            {
                if (!device.is_gpu())
                    continue;
                const std::size_t admitted_graph_bytes =
                    memory_authority->plannedBytes(
                        device,
                        PhysicalMemoryOwner::NativeGraphExecutable);
                if (admitted_graph_bytes == 0u)
                {
                    throw std::logic_error(
                        "Participant GPU graph family has zero native graph-pool admission for " +
                        device.toString());
                }
                native_graph_memory_reservations_.emplace(
                    device,
                    std::make_shared<PhysicalMemoryOwnerReservation>(
                        memory_authority->reserveNewAllocations(
                            device,
                            PhysicalMemoryOwner::NativeGraphExecutable,
                            admitted_graph_bytes)));
            }
            serial_graph_family_workspace_allocator_ =
                std::make_shared<WorkspaceAllocator>(memory_authority);
        }
        {
            ScopedWeightLoadDetailTimer timer(
                "overlay.graph.create_compact_arenas");
            createSerialCompactBufferArenas();
        }
        {
            ScopedWeightLoadDetailTimer timer(
                "overlay.graph.prepare_participant_weights");
            prepareParticipantWeights();
        }
        {
            ScopedWeightLoadDetailTimer timer(
                "overlay.graph.create_device_contexts");
            createDeviceContexts();
        }
        {
            ScopedWeightLoadDetailTimer timer(
                "overlay.graph.create_participant_gpu_runtimes");
            createParticipantGpuRuntimes();
        }
        /*
         * The residency-maintenance service starts immediately after every
         * rank finishes runner construction. Materialize the complete bounded
         * graph now so each local stage resolves its exact prepared engines
         * and atomically publishes the initial participant bank before that
         * service can inspect or clone an epoch. The sparse packet supplies
         * the live row count later; graph addresses and capacity never change.
         */
        ParticipantGraphBuildSpec main_spec;
        main_spec.role = ParticipantGraphFamilyRole::Main;
        main_spec.row_capacity = config_.max_graph_activation_rows;
        main_spec.source_layers.reserve(
            static_cast<size_t>(main_layer_count_));
        for (int layer = 0; layer < main_layer_count_; ++layer)
            main_spec.source_layers.push_back(layer);
        {
            ScopedWeightLoadDetailTimer timer(
                "overlay.graph.build_main_family");
            main_graph_ = buildGraph(main_spec);
        }
        if (!main_graph_)
        {
            throw std::runtime_error(
                "MoE overlay participant main graph construction returned null");
        }

        {
            ScopedWeightLoadDetailTimer timer(
                "overlay.graph.build_mtp_families");
            mtp_sidecar_graphs_.reserve(mtp_source_layers_.size());
            for (size_t depth = 0; depth < mtp_source_layers_.size(); ++depth)
            {
                ParticipantGraphBuildSpec mtp_spec;
                mtp_spec.role = ParticipantGraphFamilyRole::MTPDraft;
                mtp_spec.source_layers = {mtp_source_layers_[depth]};
                mtp_spec.mtp_graph_depth = static_cast<int>(depth);
                mtp_spec.row_capacity = config_.max_decode_activation_rows;
                auto graph = buildGraph(mtp_spec);
                if (!graph)
                {
                    throw std::runtime_error(
                        "MoE overlay participant MTP sidecar graph construction returned null for graph depth " +
                        std::to_string(depth));
                }
                mtp_sidecar_graphs_.push_back(std::move(graph));
            }
        }
        for (const auto &[participant_id, runtime] :
             participant_gpu_runtimes_)
        {
            if (!runtime || !runtime->runtime_table ||
                runtime->initialized_layers.size() !=
                    static_cast<std::size_t>(
                        runtime->runtime_table->layerCount()) ||
                std::any_of(
                    runtime->initialized_layers.begin(),
                    runtime->initialized_layers.end(),
                    [](std::uint8_t initialized)
                    {
                        return initialized == 0u;
                    }))
            {
                throw std::runtime_error(
                    "Participant retained graph families did not publish every topology-wide device runtime layer for participant " +
                    std::to_string(participant_id));
            }
        }
        if (!config_.participant_residency->allInitialBanksReady())
        {
            throw std::runtime_error(
                "Participant retained graph families completed construction before every local initial prepared residency bank became ready");
        }
        releasePreparedSourceBytes();

        PerfStatsCollector::addCounter(
            "moe_overlay_participant_graph",
            "prepared_ranks",
            1.0,
            "setup",
            participantDeviceList(local_participants_),
            {
                {"participants", participantIdList(local_participants_)},
                {"world_rank", std::to_string(config_.mpi_context->rank())},
                {"weight_bindings",
                 std::to_string(
                     frozen_weights_ ? frozen_weights_->bindings().size() : 0u)},
                {"main_layers", std::to_string(main_layer_count_)},
                {"mtp_sidecar_graphs",
                 std::to_string(mtp_sidecar_graphs_.size())},
            });
    }

    MoEOverlayParticipantGraphRunner::~MoEOverlayParticipantGraphRunner() =
        default;

    ServingGraphPreparationKind
    MoEOverlayParticipantGraphRunner::servingGraphPreparationKind()
        const noexcept
    {
        if (local_participants_.empty())
            return ServingGraphPreparationKind::Unresolved;

        bool owns_gpu_endpoint = false;
        for (const auto *participant : local_participants_)
        {
            if (!participant || !participant->device.is_valid())
                return ServingGraphPreparationKind::Unresolved;
            owns_gpu_endpoint =
                owns_gpu_endpoint || participant->device.is_gpu();
        }

        /* CPU endpoints are already materialized in their retained eager
         * graph. A mixed follower still reports the stronger native-device
         * transition because every GPU endpoint must be sealed before the
         * rank can accept its first authenticated transaction. */
        return owns_gpu_endpoint
                   ? ServingGraphPreparationKind::
                         NativeDeviceExecutableFamily
                   : ServingGraphPreparationKind::EagerHostGraph;
    }

    bool MoEOverlayParticipantGraphRunner::
        installMoEOverlayTransferProgressEpoch(
            std::shared_ptr<MappedTransferProgressEpoch> epoch)
    {
        if (!epoch || !epoch->device().is_gpu() ||
            serving_graph_family_lifecycle_ ==
                ServingGraphFamilyLifecycle::Sealed)
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Transfer-progress epoch installation requires one local GPU before follower graph sealing");
            return false;
        }
        const DeviceId device = epoch->device();
        const bool owns_device = std::any_of(
            local_participants_.begin(),
            local_participants_.end(),
            [device](const auto *participant)
            {
                return participant && participant->device == device;
            });
        if (!owns_device)
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Transfer-progress epoch names a non-local follower device "
                << device.toString());
            return false;
        }

        const auto found = transfer_progress_epochs_.find(device);
        if (found != transfer_progress_epochs_.end() &&
            found->second != epoch)
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Refused to replace the transfer-progress authority on "
                << device.toString());
            return false;
        }
        transfer_progress_epochs_[device] = epoch;

        /* Every follower GPU owns one execution-state runtime. Transfer
         * progress is independent of whether host or device policy authors the
         * next placement, so bind the retained relay epoch to the same exact
         * inference-boundary owner in both regimes. */
        for (auto &[participant_id, runtime] : participant_gpu_runtimes_)
        {
            (void)participant_id;
            if (runtime && runtime->device == device &&
                !runtime->installMoEOverlayTransferProgressEpoch(epoch))
            {
                LOG_ERROR(
                    "[MoEOverlayParticipantGraphRunner] Could not bind transfer progress to a device-controller runtime on "
                    << device.toString());
                return false;
            }
        }
        return true;
    }

    bool MoEOverlayParticipantGraphRunner::
        materializeServingGraphFamilyWithoutLaunch(
            const ServingGraphFamilyMaterializationPlan &plan)
    {
        ScopedWeightLoadDetailTimer materialization_timer(
            "overlay.graph.materialize_serving_family");
        const auto normalized_buckets =
            normalizePrefillGraphBuckets(plan.prefill_bucket_rows);
        if (!plan.valid() || normalized_buckets != plan.prefill_bucket_rows ||
            normalized_buckets != config_.prefill_graph_row_shapes ||
            normalized_buckets.back() > config_.max_graph_activation_rows)
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Follower serving graph setup disagrees with the frozen distributed bucket inventory");
            return false;
        }

        const auto branch_factory_for = [](DeviceId)
        {
            return GraphCaptureAuxiliaryBranchFactory{};
        };
        const auto endpoint_matches_final_identity =
            [&](const CachedParticipantGraph::MappedGPUFollowerEndpoint &endpoint)
        {
            const auto factory = branch_factory_for(endpoint.device);
            const bool owns_branch =
                static_cast<bool>(endpoint.cache.auxiliary_branch);
            return endpoint.cache.initialized &&
                   endpoint.cache.retained_full_graph_replay.valid() &&
                   endpoint.cache.executable_submission_state ==
                       DeviceGraphExecutor::GraphSegmentCache::
                           ExecutableSubmissionState::MaterializedUnlaunched &&
                   owns_branch == factory.valid() &&
                   (!factory.valid() ||
                    endpoint.cache.auxiliary_branch_authority ==
                        factory.authority_identity);
        };

        std::vector<CachedParticipantGraph *> families;
        if (main_graph_)
            families.push_back(main_graph_.get());
        for (auto &sidecar : mtp_sidecar_graphs_)
        {
            if (sidecar)
                families.push_back(sidecar.get());
        }
        if (families.empty())
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Follower serving setup has no declared graph family");
            return false;
        }

        if (serving_graph_family_lifecycle_ ==
            ServingGraphFamilyLifecycle::Sealed)
        {
            for (const CachedParticipantGraph *family : families)
            {
                if (!family)
                    return false;
                for (const auto &shape : family->mapped_gpu_shapes)
                {
                    for (const auto &endpoint : shape.endpoints)
                    {
                        if (!endpoint ||
                            !endpoint_matches_final_identity(*endpoint))
                        {
                            LOG_ERROR(
                                "[MoEOverlayParticipantGraphRunner] Repeated follower serving setup found stale graph-branch identity");
                            return false;
                        }
                    }
                }
            }
            return true;
        }

        std::size_t total_native_transactions = 0u;
        std::size_t total_branch_transactions = 0u;
        try
        {
            for (CachedParticipantGraph *family : families)
            {
                if (!family)
                    throw std::logic_error(
                        "Follower serving family retained a null graph owner");
                if (!family->usesMappedActivationEpochs())
                    continue;

                const std::size_t graph_family_ordinal =
                    family->role == ParticipantGraphFamilyRole::Main
                        ? 0u
                        : static_cast<std::size_t>(
                              family->mtp_graph_depth + 1);
                std::size_t family_native_transactions = 0u;
                std::size_t family_branch_transactions = 0u;

                for (auto &shape : family->mapped_gpu_shapes)
                {
                    for (auto &endpoint : shape.endpoints)
                    {
                        if (!endpoint || !endpoint->graph ||
                            !endpoint->executor ||
                            !endpoint->cache.capture_stream ||
                            endpoint->cache.initialized)
                        {
                            throw std::runtime_error(
                                "Mapped follower endpoint is not pristine at final serving-graph materialization");
                        }
                        const auto context_it =
                            execution_contexts_.find(endpoint->device);
                        IWorkerGPUContext *const worker =
                            participantWorkerContext(endpoint->device);
                        if (context_it == execution_contexts_.end() ||
                            !context_it->second || !worker ||
                            endpoint->graph->nativeCaptureEnvelope() !=
                                GraphNativeCaptureEnvelope::
                                    DeviceOwnedTimelineTransaction)
                        {
                            throw std::runtime_error(
                                "Mapped follower could not resolve its final native capture owner");
                        }

                        const auto branch_factory =
                            branch_factory_for(endpoint->device);
                        bool materialized = false;
                        worker->submitAndWait(
                            [&]()
                            {
                                endpoint->graph->reset();
                                materialized = endpoint->executor
                                                   ->executeWithCachedGraphReplay(
                                                       *endpoint->graph,
                                                       context_it->second,
                                                       endpoint->cache,
                                                       endpoint->cache.capture_stream,
                                                       worker,
                                                       /*collective_nodes=*/nullptr,
                                                       /*collectives_graph_capturable=*/false,
                                                       /*force_recapture=*/false,
                                                       /*defer_final_sync=*/true,
                                                       {},
                                                       DeviceGraphExecutor::
                                                           GraphReplayPlanPolicy::
                                                               RequireFullGraph,
                                                       {},
                                                       {},
                                                       {},
                                                       DeviceGraphExecutor::
                                                           GraphInitialSubmissionPolicy::
                                                               MaterializeWithoutLaunch,
                                                       branch_factory) &&
                                               endpoint->cache
                                                   .prepareCaptureStreamTerminal(
                                                       worker);
                            });
                        if (!materialized ||
                            !endpoint_matches_final_identity(*endpoint))
                        {
                            throw std::runtime_error(
                                "Mapped follower could not seal its graph-owned transfer branch for participant " +
                                std::to_string(endpoint->participant_id) +
                                " rows=" +
                                std::to_string(shape.physical_rows));
                        }
                        ++family_native_transactions;
                        if (branch_factory.valid())
                            ++family_branch_transactions;
                    }
                }

                /* Every row shape for one participant borrows the same stream.
                 * One exact event fence per device proves setup left no work in
                 * front of transaction zero; it is never an inference wait. */
                std::vector<DeviceId> quiescent_devices;
                for (auto &shape : family->mapped_gpu_shapes)
                {
                    for (auto &endpoint : shape.endpoints)
                    {
                        if (!endpoint ||
                            std::find(
                                quiescent_devices.begin(),
                                quiescent_devices.end(),
                                endpoint->device) != quiescent_devices.end())
                        {
                            continue;
                        }
                        IWorkerGPUContext *const worker =
                            participantWorkerContext(endpoint->device);
                        if (!worker)
                        {
                            throw std::runtime_error(
                                "Mapped follower setup lost its participant worker context");
                        }
                        worker->submitAndWait(
                            [&]()
                            {
                                endpoint->cache.waitForCaptureStreamFence(
                                    DeviceGraphExecutor::GraphSegmentCache::
                                        HostFenceWaitPolicy::ActiveProgress,
                                    [device = endpoint->device,
                                     family_ordinal = graph_family_ordinal,
                                     rows = endpoint->physical_rows]()
                                    {
                                        return
                                            "final mapped follower graph materialization left queued work on device=" +
                                            device.toString() +
                                            " graph_family=" +
                                            std::to_string(family_ordinal) +
                                            " rows=" +
                                            std::to_string(rows);
                                    });
                            });
                        quiescent_devices.push_back(endpoint->device);
                    }
                }

                PerfStatsCollector::addCounter(
                    "moe_overlay_participant_graph",
                    "mapped_follower_setup_stream_quiescence_proofs",
                    static_cast<double>(quiescent_devices.size()),
                    "model_setup",
                    participantDeviceList(local_participants_),
                    {{"graph_family", std::to_string(graph_family_ordinal)},
                     {"ordering", "exact_stream_event"},
                     {"host_blocking", "setup_only"}});
                PerfStatsCollector::addCounter(
                    "moe_overlay_participant_graph",
                    "materialized_mapped_follower_families",
                    1.0,
                    "model_setup",
                    participantDeviceList(local_participants_),
                    {{"graph_family", std::to_string(graph_family_ordinal)},
                     {"row_capacity", std::to_string(family->row_capacity)},
                     {"setup_materialized_gpu_transactions",
                      std::to_string(family_native_transactions)},
                     {"setup_materialized_cpu_endpoints",
                      std::to_string(
                          family->mapped_cpu_endpoints.size())},
                     {"captured_transfer_branches",
                      std::to_string(family_branch_transactions)},
                     {"standalone_progress_launch", "false"}});
                total_native_transactions += family_native_transactions;
                total_branch_transactions += family_branch_transactions;
            }
        }
        catch (const std::exception &error)
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Final follower graph materialization failed: "
                << error.what());
            return false;
        }
        catch (...)
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Final follower graph materialization threw a non-standard exception");
            return false;
        }

        serving_graph_family_lifecycle_ =
            ServingGraphFamilyLifecycle::Sealed;
        PerfStatsCollector::addCounter(
            "moe_overlay_participant_graph",
            "mapped_follower_serving_graph_family_completions",
            1.0,
            "model_setup",
            participantDeviceList(local_participants_),
            {{"native_transactions",
              std::to_string(total_native_transactions)},
             {"captured_transfer_branches",
              std::to_string(total_branch_transactions)},
             {"authority_installation", "before_native_capture"}});
        return true;
    }

    /**
     * @brief Resolve immutable rank-local endpoint ownership from the topology.
     *
     * The runtime and execution plans are both derived from the supplied
     * placement authority and MPI rank.  The resulting raw participant
     * pointers refer to @ref owner_map_, whose lifetime exceeds every cached
     * graph, so graph construction never re-resolves topology from a mutable
     * request path.
     */
    void MoEOverlayParticipantGraphRunner::resolveTopology()
    {
        runtime_plan_ = resolveMoEExpertOverlayRuntimePlan(
            config_.placement_plan,
            MoEExpertOverlayRuntimeResolverOptions{
                .current_world_rank = config_.mpi_context->rank(),
                .validate_mvp_root_reachability = false,
            });
        if (!runtime_plan_)
            throw std::runtime_error("Failed to resolve MoE overlay runtime plan");

        execution_plan_ =
            std::make_shared<MoEExpertOverlayExecutionPlan>(
                buildMoEExpertOverlayExecutionPlan(
                    *runtime_plan_, config_.mpi_context->world_size()));
        if (!execution_plan_->currentRankPlan()
                 .usesExpertTransactionFollower())
        {
            throw std::invalid_argument(
                "Only an expert-only transaction follower may use the "
                "retained participant graph runner; resolved execution kind=" +
                std::string(toString(
                    execution_plan_->currentRankPlan().execution_kind)));
        }

        const auto snapshot = residency_authority_->snapshot();
        if (!snapshot || !snapshot->valid() || !snapshot->placement_plan)
        {
            throw std::invalid_argument(
                "MoE overlay participant runner received an invalid residency snapshot");
        }
        const MoEExpertOwnerMap configured_owner_map =
            MoEExpertOwnerMap::build(*config_.placement_plan);
        const auto &published_participants =
            snapshot->owner_map.participants();
        const auto &configured_participants =
            configured_owner_map.participants();
        const bool same_topology =
            published_participants.size() == configured_participants.size() &&
            std::equal(
                published_participants.begin(),
                published_participants.end(),
                configured_participants.begin(),
                [](const auto &published, const auto &configured)
                {
                    return published.participant_id ==
                               configured.participant_id &&
                           published.tier_idx == configured.tier_idx &&
                           published.tier_name == configured.tier_name &&
                           published.domain_name == configured.domain_name &&
                           published.domain_participant_index ==
                               configured.domain_participant_index &&
                           published.address == configured.address &&
                           published.world_rank == configured.world_rank &&
                           published.world_rank_known ==
                               configured.world_rank_known;
                });
        if (!same_topology)
        {
            throw std::invalid_argument(
                "MoE overlay participant runner authority topology differs from its graph plan");
        }
        owner_map_ = std::make_unique<MoEExpertOwnerMap>(snapshot->owner_map);
        for (const auto &participant : owner_map_->participants())
        {
            if (participant.world_rank_known &&
                participant.world_rank == config_.mpi_context->rank())
            {
                local_participants_.push_back(&participant);
            }
        }

        std::sort(
            local_participants_.begin(),
            local_participants_.end(),
            [](const auto *left, const auto *right)
            {
                return left->participant_id < right->participant_id;
            });
        const auto duplicate = std::adjacent_find(
            local_participants_.begin(),
            local_participants_.end(),
            [](const auto *left, const auto *right)
            {
                return left->participant_id == right->participant_id;
            });
        if (duplicate != local_participants_.end())
        {
            throw std::invalid_argument(
                "MoE overlay participant rank resolves duplicate global "
                "participant id " +
                std::to_string((*duplicate)->participant_id));
        }

        std::vector<int> resolved_local_ids;
        resolved_local_ids.reserve(local_participants_.size());
        for (const auto *participant : local_participants_)
            resolved_local_ids.push_back(participant->participant_id);
        if (config_.participant_residency->initialEpoch() != snapshot->epoch ||
            config_.participant_residency->localParticipantIds() !=
                resolved_local_ids)
        {
            throw std::invalid_argument(
                "MoE overlay participant runner prepared-bank registry differs "
                "from the published rank topology or initial epoch");
        }

        /*
         * A RelayOnly rank intentionally owns no endpoint but must enter every
         * matched MPI boundary once. All other auxiliary roles must resolve at
         * least one concrete participant from the canonical owner map.
         */
        if (local_participants_.empty() &&
            execution_plan_->currentRankPlan().role !=
                OverlayRankRole::RelayOnly)
        {
            throw std::invalid_argument(
                "MoE overlay auxiliary rank " +
                std::to_string(config_.mpi_context->rank()) +
                " owns no routed participant endpoints");
        }
    }

    /**
     * @brief Partition raw GGUF blocks into main and learned NextN graph families.
     *
     * The raw Qwen block count includes the trailing NextN block, while the
     * main model graph must stop before it. The trailing block keeps its raw
     * layer number everywhere that matters—prepared-weight lookup, residency
     * bank indexing, and sparse wire keys. This method therefore records it as
     * a separate graph source instead of aliasing it onto an ordinary layer.
     */
    void MoEOverlayParticipantGraphRunner::resolveGraphFamilies()
    {
        const auto &loader = config_.model_context->concreteLoader();
        const int raw_layer_count =
            config_.model_context->totalBlockCount();
        transaction_graph_family_ =
            resolveMoEOverlayInferenceGraphFamilyIdentity(
                loader,
                architecture_,
                raw_layer_count,
                config_.mtp_graph_family_policy,
                /*graph_family_generation=*/1,
                config_.max_graph_activation_rows,
                config_.max_decode_activation_rows,
                config_.max_request_count,
                config_.max_mtp_draft_depth);
        main_layer_count_ = transaction_graph_family_.main_layer_count;
        mtp_source_layers_ = transaction_graph_family_.mtp_source_layers;
        routed_layer_capacity_ =
            transaction_graph_family_.routedLayerCapacity();

        /* The model-frozen placement plan is the topology-wide storage
         * authority. The independently resolved graph family describes the
         * exact graphs this follower will construct. Requiring equality here
         * prevents a raw GGUF sidecar block from silently enlarging only one
         * rank's runtime table, and catches a missing MTP bank before capture. */
        const int planned_layer_capacity =
            config_.placement_plan->placementLayerCapacity();
        if (routed_layer_capacity_ <= 0 ||
            planned_layer_capacity != routed_layer_capacity_)
        {
            throw std::invalid_argument(
                "MoE overlay participant graph-family layer capacity " +
                std::to_string(routed_layer_capacity_) +
                " disagrees with frozen placement capacity " +
                std::to_string(planned_layer_capacity));
        }

        const int num_experts = loader.getInt(
            architecture_ + ".expert_count", 0);
        if (num_experts <= 0)
        {
            throw std::runtime_error(
                "MoE participant graph has no routed-expert geometry");
        }
        for (const int source_layer : mtp_source_layers_)
        {
            for (int expert = 0; expert < num_experts; ++expert)
            {
                if (owner_map_->ownerCountForExpert(
                        source_layer, expert) != 1u)
                {
                    throw std::runtime_error(
                        "MTP participant source layer " +
                        std::to_string(source_layer) +
                        " lacks an exact placement owner for expert " +
                        std::to_string(expert));
                }
            }
        }

        transaction_topology_identity_ =
            makeMoEOverlayInferenceTopologyIdentity(
                *owner_map_,
                transaction_graph_family_,
                execution_plan_->continuation_root_rank,
                config_.mpi_context->rank());
    }

    /**
     * @brief Establish one planner-bounded compact tensor family per local endpoint.
     *
     * A routed local-expert stage packs each input row once and retains its
     * participant-local routes on the tensor's top-k axis. Allocate a complete
     * row-family manifest published by the distributed schedule before graph
     * construction. Small power-of-two decode/MTP shapes are added to that
     * exact prefill ladder. Every layer keeps stable addresses, while a live
     * decode/verifier packet transfers one hidden and one aggregate output row
     * rather than repeating both payloads for every selected expert.
     *
     * Relay-only ranks intentionally produce no arena. Every declared local
     * endpoint produces exactly one arena even when its epoch-one resident
     * mask is empty. Residency migration can make that endpoint live later,
     * and captured topology may not be allocated or rebound at that point.
     * Graph construction treats a missing entry as a fatal ownership
     * violation rather than allocating a stage-private substitute.
     */
    void MoEOverlayParticipantGraphRunner::createSerialCompactBufferArenas()
    {
        const int d_model = config_.model_context->embeddingLength();
        const auto &loader = config_.model_context->concreteLoader();
        const std::string &architecture = config_.model_context->architecture();
        const int num_experts = loader.getInt(architecture + ".expert_count", 0);
        const int routing_top_k =
            loader.getInt(architecture + ".expert_used_count", 0);
        int expert_intermediate =
            loader.getInt(
                architecture + ".expert_feed_forward_length", 0);
        if (expert_intermediate <= 0)
            expert_intermediate = config_.model_context->feedForwardLength();
        if (d_model <= 0 || num_experts <= 0 || routing_top_k <= 0 ||
            expert_intermediate <= 0)
        {
            throw std::runtime_error(
                "Participant serial compact arena could not resolve model routing geometry");
        }

        const size_t row_capacity =
            static_cast<size_t>(config_.max_graph_activation_rows);
        if (row_capacity == 0)
        {
            throw std::runtime_error(
                "Participant serial compact arena resolved zero row capacity");
        }

        for (const auto *participant : local_participants_)
        {
            if (!participant)
                throw std::logic_error(
                    "Participant serial compact arena received a null endpoint");

            MoELocalExpertSerialBufferArena::Config arena_config;
            arena_config.device_id = participant->device;
            arena_config.row_capacity = row_capacity;
            arena_config.row_capacity_buckets =
                MoELocalExpertSerialBufferArena::powerOfTwoRowBucketsThrough(
                    static_cast<size_t>(
                        config_.max_decode_activation_rows));
            /*
             * The continuation ticket carries one of these exact physical
             * shapes. Copy the schedule-owned manifest verbatim so follower
             * capture and admission cannot drift when the production ladder
             * contains non-power-of-two buckets such as 600 rows.
             */
            arena_config.row_capacity_buckets.reserve(
                arena_config.row_capacity_buckets.size() +
                config_.prefill_graph_row_shapes.size());
            for (const int shape : config_.prefill_graph_row_shapes)
            {
                arena_config.row_capacity_buckets.push_back(
                    static_cast<size_t>(shape));
            }
            arena_config.d_model = d_model;
            arena_config.routing_top_k = routing_top_k;
            if (participant->device.is_cpu())
            {
                arena_config.cpu_canonical_route_storage =
                    MoELocalExpertSerialBufferArena::
                        CPUCanonicalRouteStoragePolicy::RetainSerialMaximum;
                arena_config.cpu_grouped_scratch_storage =
                    MoELocalExpertSerialBufferArena::
                        CPUGroupedScratchStoragePolicy::RetainSerialMaximum;
                arena_config.num_experts = num_experts;
                arena_config.expert_intermediate = expert_intermediate;
            }
            arena_config.logical_participant_id = participant->participant_id;
            arena_config.debug_name =
                "moe_overlay_participant_serial_compact.p" +
                std::to_string(participant->participant_id) + "@" +
                participant->device.to_string();

            auto arena = std::make_shared<MoELocalExpertSerialBufferArena>(
                std::move(arena_config));
            const bool inserted = serial_compact_buffer_arenas_.emplace(
                participant->participant_id, std::move(arena)).second;
            if (!inserted)
            {
                throw std::logic_error(
                    "Participant serial compact arena has duplicate logical endpoint p" +
                    std::to_string(participant->participant_id));
            }

        }
    }

    /**
     * @brief Return the only compact tensor owner legal for @p participant.
     *
     * This extra validation is intentionally at graph materialization rather
     * than hidden in a map lookup.  It catches a stale owner map, a malformed
     * participant id, or an accidental device reassignment before either graph
     * can capture the wrong tensor address.
     */
    std::shared_ptr<MoELocalExpertSerialBufferArena>
    MoEOverlayParticipantGraphRunner::serialCompactBufferArenaForParticipant(
        const MoEExpertOwnerParticipant &participant) const
    {
        const auto found = serial_compact_buffer_arenas_.find(
            participant.participant_id);
        if (found == serial_compact_buffer_arenas_.end() || !found->second)
        {
            throw std::logic_error(
                "Participant graph is missing its immutable serial compact arena for p" +
                std::to_string(participant.participant_id));
        }
        const auto &arena = found->second;
        if (arena->deviceId() != participant.device ||
            !arena->logicalParticipantId() ||
            *arena->logicalParticipantId() != participant.participant_id)
        {
            throw std::logic_error(
                "Participant graph compact arena ownership does not match p" +
                std::to_string(participant.participant_id));
        }
        return arena;
    }

    /**
     * @brief Materialize and register exact real-weight GEMM engines for local experts.
     *
     * Weight preparation happens before graph construction and uses the
     * rank-specific map above.  A later local stage is registry-only; it cannot
     * fall back to a raw model parent or prepare a new kernel on the hot path.
     */
    void MoEOverlayParticipantGraphRunner::prepareParticipantWeights()
    {
        auto weight_manager = config_.model_context->concreteWeightManager();
        if (!weight_manager)
            throw std::runtime_error("Participant model context has no WeightManager");

        InferenceStrategy strategy;
        strategy.mode = WeightInferenceMode::ExpertOverlayRank;
        strategy.model_id = ModelContextId{
            reinterpret_cast<uint64_t>(config_.model_context.get())};
        strategy.devices = participantDevices(local_participants_);
        WeightPlan plan(std::move(strategy));
        if (config_.prepared_weight_admission ==
            PreparedWeightAdmission::AllocateCompleteSet)
        {
            ScopedWeightLoadDetailTimer timer(
                "overlay.weights.build_plan");
            for (const auto *participant : local_participants_)
            {
                if (!participant)
                    continue;
                const WeightPlan participant_plan =
                    buildMoEOverlayParticipantWeightPlan(
                        *config_.model_context,
                        *execution_plan_,
                        *owner_map_,
                        *participant);
                for (const auto &requirement :
                     participant_plan.requirements())
                {
                    plan.add(requirement);
                }
            }
        }
        const ModelContextId model_id = plan.strategy().model_id;

        {
            ScopedWeightLoadDetailTimer timer(
                "overlay.weights.bind_prepared_store");
            prepared_store_ = weight_manager->preparedWeightStoreIfInitialized();
            if (!prepared_store_)
            {
                if (config_.prepared_weight_admission ==
                    PreparedWeightAdmission::ReuseCertifiedCompleteSet)
                {
                    throw std::runtime_error(
                        "Certified participant-weight reuse lost the exact "
                        "model-owned PreparedWeightStore");
                }
                prepared_store_ =
                    std::make_shared<PreparedWeightStore>(model_id);
                weight_manager->setPreparedWeightStore(prepared_store_);
            }
            if (!prepared_store_->bindModelIdIfUnset(model_id))
            {
                throw std::runtime_error(
                    "Participant PreparedWeightStore belongs to another model");
            }
        }

        if (plan.empty() &&
            config_.prepared_weight_admission ==
                PreparedWeightAdmission::AllocateCompleteSet)
        {
            /* Relay ranks own protocol state but no expert weight authority. */
            return;
        }

        if (!plan.empty())
        {
            ScopedWeightLoadDetailTimer timer(
                "overlay.weights.materialize_plan");
            frozen_weights_ = std::make_unique<FrozenModelWeightSet>(
                weight_manager->materialize(plan));
        }
        for (const DeviceId device : participantDevices(local_participants_))
        {
            if (!weight_manager->prepareMoEExpertOverlayWeights(
                    *runtime_plan_,
                    device,
                    frozen_weights_.get(),
                    execution_plan_.get(),
                    config_.prepared_weight_admission))
            {
                throw std::runtime_error(
                    "Failed to prepare exact MoE overlay participant weights on " +
                    device.to_string());
            }
        }
    }

    /**
     * @brief Create the exact contexts that execute locally owned graph nodes.
     *
     * CPU transport stages always need a host context.  GPU contexts are
     * created once here, before graph materialization, so local-expert stages
     * can bind their exact worker stream and persistent arena storage without
     * consulting a backend default stream during inference.
     */
    void MoEOverlayParticipantGraphRunner::createDeviceContexts()
    {
        std::vector<DeviceId> devices = participantDevices(local_participants_);
        if (std::find(devices.begin(), devices.end(), DeviceId::cpu()) ==
            devices.end())
        {
            /* Every graph has CPU transport boundary stages. */
            devices.push_back(DeviceId::cpu());
        }

        for (const DeviceId device : devices)
        {
            auto context = IDeviceContext::create(device);
            if (!context)
            {
                throw std::runtime_error(
                    "Failed to create participant context for " +
                    device.to_string());
            }
            auto *raw_context = context.get();
            auto [it, inserted] = owned_device_contexts_.emplace(
                device, std::move(context));
            if (!inserted)
            {
                throw std::runtime_error(
                    "Duplicate participant device context for " +
                    device.to_string());
            }
            execution_contexts_.emplace(device, raw_context);
        }
    }

    /**
     * @brief Materialize one shared placement/runtime family per follower GPU.
     *
     * Policy ownership and GPU execution-state ownership are different axes.
     * Every GPU receives a real mirrored runtime table and epoch ticket. Under
     * host-resident heterogeneous authority, background publication writes its
     * inactive bank through the table's retained host recipe. Under all-GPU
     * device authority, the same arena additionally consumes the mapped global
     * admission word. Row-shape and MTP graphs borrow these addresses; none may
     * create a private placement generation.
     */
    void MoEOverlayParticipantGraphRunner::createParticipantGpuRuntimes()
    {
        const auto &loader = config_.model_context->concreteLoader();
        const std::string &arch = config_.model_context->architecture();
        /* resolveGraphFamilies() already proved this value against the frozen
         * placement plan. Never re-derive it from totalBlockCount(): Qwen GGUFs
         * include an inactive trailing NextN block when MTP is disabled. */
        const int runtime_layer_count = routed_layer_capacity_;
        const int num_experts = loader.getInt(arch + ".expert_count", 0);
        const int top_k = loader.getInt(arch + ".expert_used_count", 0);
        const auto authority_execution =
            config_.placement_plan->authority_execution;
        if (runtime_layer_count <= 0 || num_experts <= 0 || top_k <= 0 ||
            authority_execution ==
                MoEOverlayAuthorityExecutionKind::Unresolved)
        {
            throw std::invalid_argument(
                "Mapped follower GPU runtime has unresolved model geometry or policy authority");
        }

        const bool device_resident_authority =
            authority_execution ==
            MoEOverlayAuthorityExecutionKind::DeviceResident;
        if (device_resident_authority !=
            static_cast<bool>(config_.device_controller_fabric))
        {
            throw std::invalid_argument(
                device_resident_authority
                    ? "Device-resident follower GPU runtime has no topology-wide controller fabric"
                    : "Host-resident follower GPU runtime unexpectedly received a device-controller fabric");
        }

        std::vector<int> expected_gpu_participants;
        for (const auto *participant : local_participants_)
        {
            if (participant && participant->device.is_gpu())
                expected_gpu_participants.push_back(participant->participant_id);
        }
        std::sort(
            expected_gpu_participants.begin(),
            expected_gpu_participants.end());
        if (config_.device_controller_fabric)
        {
            const auto &layout = config_.device_controller_fabric->layout();
            const auto &header = layout.header;
            if (!layout.valid() ||
                header.num_layers !=
                    static_cast<std::uint32_t>(runtime_layer_count) ||
                header.num_experts !=
                    static_cast<std::uint32_t>(num_experts))
            {
                throw std::invalid_argument(
                    "Mapped follower GPU runtime geometry disagrees with the topology-wide controller fabric");
            }
            auto fabric_participants =
                config_.device_controller_fabric->localParticipantIds();
            std::sort(
                fabric_participants.begin(), fabric_participants.end());
            if (expected_gpu_participants != fabric_participants)
            {
                throw std::invalid_argument(
                    "Mapped follower GPU participants differ from the local controller-fabric membership");
            }
        }

        for (const auto *participant : local_participants_)
        {
            if (!participant || !participant->device.is_gpu())
                continue;

            std::optional<MoEOverlayDeviceControllerParticipantBinding>
                controller_binding;
            if (config_.device_controller_fabric)
            {
                controller_binding =
                    config_.device_controller_fabric->participantBinding(
                        participant->participant_id);
                if (!controller_binding->valid() ||
                    controller_binding->participant_id !=
                        participant->participant_id ||
                    controller_binding->device != participant->device ||
                    !controller_binding->controller ||
                    !controller_binding->lifetime)
                {
                    throw std::logic_error(
                        "Mapped follower received an incomplete controller-fabric participant binding");
                }
            }

            auto domain_participants =
                owner_map_->participantIdsForTier(participant->tier_idx);
            if (domain_participants.empty() ||
                domain_participants.size() > kDeviceMoEMaxParticipants ||
                participant->domain_participant_index < 0 ||
                static_cast<std::size_t>(
                    participant->domain_participant_index) >=
                    domain_participants.size())
            {
                throw std::invalid_argument(
                    "Mapped follower domain is outside the device runtime participant ABI");
            }
            std::vector<std::uint8_t> dense_domain_ids(
                domain_participants.size(), 0u);
            for (const int global_participant_id : domain_participants)
            {
                const auto *const member =
                    owner_map_->participantForId(global_participant_id);
                if (!member || member->tier_idx != participant->tier_idx ||
                    member->domain_name != participant->domain_name ||
                    member->domain_participant_index < 0 ||
                    static_cast<std::size_t>(
                        member->domain_participant_index) >=
                        dense_domain_ids.size() ||
                    dense_domain_ids[static_cast<std::size_t>(
                        member->domain_participant_index)] != 0u)
                {
                    throw std::invalid_argument(
                        "Mapped follower tier does not provide a dense unique domain participant namespace");
                }
                dense_domain_ids[static_cast<std::size_t>(
                    member->domain_participant_index)] = 1u;
            }

            const bool collect_dynamic_service_telemetry =
                config_.durable_maintenance_policy ==
                MoEOverlayDurableMaintenancePolicy::DynamicPlacement;
            const auto service_telemetry_catalog =
                collect_dynamic_service_telemetry &&
                        config_.residency_authority
                    ? config_.residency_authority->economyLayerCatalog()
                    : nullptr;
            if (collect_dynamic_service_telemetry &&
                !service_telemetry_catalog)
            {
                throw std::logic_error(
                    "Dynamic mapped follower has no canonical economy layer catalog");
            }

            auto runtime = std::make_unique<ParticipantGpuRuntime>();
            runtime->participant_id = participant->participant_id;
            runtime->device = participant->device;
            runtime->domain_participant_id = static_cast<std::uint32_t>(
                participant->domain_participant_index);
            runtime->domain_participant_count = static_cast<std::uint32_t>(
                domain_participants.size());
            runtime->authority_execution = authority_execution;
            runtime->controller_binding = controller_binding;
            runtime->worker = participantWorkerContext(participant->device);
            if (!runtime->worker)
            {
                throw std::runtime_error(
                    "Mapped follower runtime could not resolve its exact GPU worker");
            }

            runtime->worker->submitAndWait(
                [&]
                {
                    runtime->runtime_publication_stream =
                        runtime->worker->defaultStream();
                    IBackend *const backend = getBackendFor(
                        participant->device);
                    if (!backend || !runtime->runtime_publication_stream)
                    {
                        throw std::runtime_error(
                            "Mapped follower runtime could not resolve its "
                            "initial-publication backend or exact stream");
                    }
                    const int ordinal = participant->device.gpu_ordinal();
                    const auto make_event =
                        [backend, ordinal]() -> std::shared_ptr<void>
                    {
                        void *const event = backend->createEvent(ordinal);
                        if (!event)
                            return {};
                        return std::shared_ptr<void>(
                            event,
                            [backend, ordinal](void *owned)
                            {
                                if (owned)
                                    backend->destroyEvent(owned, ordinal);
                            });
                    };
                    runtime->initial_runtime_source_ready_event =
                        make_event();
                    runtime->initial_runtime_published_event = make_event();
                    runtime->route_scratch = std::make_shared<
                        DeviceMoESerialRouteScratchArena>(
                        DeviceMoESerialRouteScratchArena::Config{
                            .device_id = participant->device,
                            .num_experts = num_experts,
                            .top_k = top_k,
                            .token_capacity =
                                config_.max_graph_activation_rows,
                        });
                    DeviceMoEOverlayEpochArena::Config epoch_config{
                        .device_id = participant->device,
                        .initial_epoch = 1u,
                        .initial_bank = 1u,
                        .request_slot_capacity = 1u,
                    };
                    if (controller_binding)
                    {
                        epoch_config.external_admission_epoch =
                            &controller_binding->controller->admission_epoch;
                        epoch_config.external_admission_lifetime =
                            controller_binding->lifetime;
                    }
                    runtime->epoch_arena = std::make_shared<
                        DeviceMoEOverlayEpochArena>(
                        std::move(epoch_config));
                    runtime->runtime_table = std::make_unique<
                        DeviceMoERuntimeTable>(
                        DeviceMoERuntimeTable::Config{
                            .device_id = participant->device,
                            .num_layers = runtime_layer_count,
                            .num_experts = num_experts,
                            .top_k = top_k,
                            .mirror_to_device = true,
                            .overlay_service_telemetry_coverage =
                                collect_dynamic_service_telemetry
                                    ? MoEOverlayServiceTelemetryCoverage::
                                          CatalogStratifiedSample
                                    : MoEOverlayServiceTelemetryCoverage::
                                          Disabled,
                            .overlay_service_telemetry_catalog =
                                service_telemetry_catalog,
                            .prefill_token_capacity =
                                config_.max_graph_activation_rows,
                            .deferred_verifier_token_capacity =
                                config_.max_decode_activation_rows,
                            .serial_route_scratch_arena =
                                runtime->route_scratch,
                            .overlay_epoch_arena = runtime->epoch_arena,
                            .overlay_epoch_ticket_slot = 0u,
                        });
                });
            if (!runtime->route_scratch || !runtime->epoch_arena ||
                !runtime->runtime_table ||
                !runtime->runtime_publication_stream ||
                !runtime->initial_runtime_source_ready_event ||
                !runtime->initial_runtime_published_event)
            {
                throw std::runtime_error(
                    "Mapped follower runtime allocation returned incomplete model-lifetime state");
            }
            runtime->initialized_layers.assign(
                static_cast<std::size_t>(runtime_layer_count), 0u);

            const auto [_, inserted] =
                participant_gpu_runtimes_.emplace(
                    participant->participant_id, std::move(runtime));
            if (!inserted)
            {
                throw std::logic_error(
                    "Mapped follower constructed duplicate controller runtime authority");
            }

            PerfStatsCollector::addCounter(
                "moe_overlay_controller",
                "follower_runtime_tables_materialized",
                1.0,
                "model_setup",
                participant->device.toString(),
                {{"participant",
                  std::to_string(participant->participant_id)},
                 {"domain_participant",
                  std::to_string(participant->domain_participant_index)},
                 {"domain_participants",
                  std::to_string(domain_participants.size())},
                 {"layers", std::to_string(runtime_layer_count)},
                 {"blocking_hot_path", "false"},
                 {"authority_execution",
                  std::string(toString(authority_execution))},
                 {"mapped_device_controller",
                  controller_binding ? "true" : "false"}});
        }
    }

    /** @brief Resolve the exact model-lifetime runtime for one global participant. */
    MoEOverlayParticipantGraphRunner::ParticipantGpuRuntime &
    MoEOverlayParticipantGraphRunner::participantGpuRuntimeForParticipant(
        int participant_id) const
    {
        const auto found = participant_gpu_runtimes_.find(participant_id);
        if (found == participant_gpu_runtimes_.end() || !found->second)
        {
            throw std::out_of_range(
                "Mapped follower has no GPU placement runtime for participant " +
                std::to_string(participant_id));
        }
        return *found->second;
    }

    /**
     * @brief Release mapped raw GGUF bytes once prepared engines own the selected experts.
     *
     * The frozen bindings retain metadata and packed-engine lifetimes while the
     * raw host pages are no longer an execution authority.  A failure to
     * release one mapping is diagnostic-only because it cannot change the
     * prepared graph's numerical result or ownership.
     */
    void MoEOverlayParticipantGraphRunner::releasePreparedSourceBytes()
    {
        if (!frozen_weights_)
            return;

        size_t released_bytes = 0;
        for (const auto &binding : frozen_weights_->bindings())
        {
            auto tensor = binding.tensor_owner;
            if (!tensor || tensor->is_view() ||
                tensor->is_raw_data_released())
            {
                continue;
            }
            const size_t bytes = tensor->size_bytes();
            try
            {
                tensor->release_host_weight_data();
                released_bytes += bytes;
            }
            catch (const std::exception &error)
            {
                LOG_WARN(
                    "[MoEOverlayParticipantGraphRunner] Could not release "
                    << binding.identity.canonical_name << ": "
                    << error.what());
            }
        }

        PerfStatsCollector::addCounter(
            "moe_overlay_participant_graph",
            "released_source_bytes",
            static_cast<double>(released_bytes),
            "setup",
            participantDeviceList(local_participants_),
            {{"participants", participantIdList(local_participants_)}});
    }

    /**
     * @brief Validate one live packet shape and return the setup-owned graph.
     *
     * Every possible row count at or below the planner admission shares this
     * immutable graph. The per-transaction sparse views contain the live row
     * count, so no tail-shaped graph construction is legal in the hot path.
     */
    MoEOverlayParticipantGraphRunner::CachedParticipantGraph &
    MoEOverlayParticipantGraphRunner::graphForRows(int logical_rows)
    {
        if (logical_rows <= 0 ||
            logical_rows > config_.max_graph_activation_rows)
        {
            throw std::invalid_argument(
                "Participant live rows " + std::to_string(logical_rows) +
                " exceed immutable planner-admitted graph capacity " +
                std::to_string(config_.max_graph_activation_rows));
        }
        if (!main_graph_ || !main_graph_->graph)
        {
            throw std::logic_error(
                "Participant main graph was not materialized during setup");
        }
        return *main_graph_;
    }

    /**
     * @brief Build one ordered sparse dispatch/local-expert/return graph variant.
     *
     * This graph contains every globally ordered remote MPI boundary so it
     * remains protocol-symmetric with the dense continuation root graph.
     * Participants in a multi-device continuation domain execute in their own
     * captured LocalTP graphs and therefore never enter the rank-level MPI
     * protocol. Only remote target participants owned by this rank receive a
     * local compute stage. Those stages bind their pre-existing arena; graph
     * construction is forbidden from allocating a layer-private compact
     * packet.
     */
    std::unique_ptr<MoEOverlayParticipantGraphRunner::CachedParticipantGraph>
    MoEOverlayParticipantGraphRunner::buildGraph(
        const ParticipantGraphBuildSpec &spec)
    {
        const bool valid_main =
            spec.role == ParticipantGraphFamilyRole::Main &&
            spec.mtp_graph_depth == -1 &&
            spec.row_capacity == config_.max_graph_activation_rows;
        const bool valid_mtp =
            spec.role == ParticipantGraphFamilyRole::MTPDraft &&
            spec.mtp_graph_depth >= 0 &&
            spec.row_capacity == config_.max_decode_activation_rows;
        if ((!valid_main && !valid_mtp) || spec.source_layers.empty())
        {
            throw std::invalid_argument(
                "Participant retained graph spec has an invalid role, depth, layer family, or admitted capacity");
        }
        const int row_capacity = spec.row_capacity;

        auto result = std::make_unique<CachedParticipantGraph>();
        result->graph = std::make_unique<ComputeGraph>();
        result->role = spec.role;
        result->mtp_graph_depth = spec.mtp_graph_depth;
        result->row_capacity = row_capacity;
        result->source_layers = spec.source_layers;
        const int d_model = config_.model_context->embeddingLength();
        const auto &loader = config_.model_context->concreteLoader();
        const std::string &arch = config_.model_context->architecture();
        const int num_experts = loader.getInt(arch + ".expert_count", 0);
        const int top_k = loader.getInt(arch + ".expert_used_count", 0);
        int expert_intermediate =
            loader.getInt(arch + ".expert_feed_forward_length", 0);
        if (expert_intermediate <= 0)
            expert_intermediate = config_.model_context->feedForwardLength();
        if (d_model <= 0 || num_experts <= 0 || top_k <= 0 ||
            expert_intermediate <= 0)
        {
            throw std::runtime_error(
                "Participant graph could not resolve Qwen MoE geometry");
        }

        const std::vector<int> local_participant_ids = participantIds();
        const auto weight_manager =
            config_.model_context->concreteWeightManager();
        if (!weight_manager)
        {
            throw std::logic_error(
                "Participant graph has no model-owned WeightManager");
        }
        if (!weight_manager->physicalMemoryAuthority() ||
            !serial_graph_family_workspace_allocator_)
        {
            throw std::logic_error(
                "Participant graph has no runner-owned serial workspace authority");
        }
        auto &registry = weight_manager->expertGemmRegistry();
        const int continuation_root =
            config_.placement_plan->continuation_domain_spec
                .logical_root_participant;
        const auto *const declared_continuation_root =
            owner_map_->participantForId(continuation_root);
        if (continuation_root < 0 || !declared_continuation_root ||
            declared_continuation_root->domain_name !=
                config_.placement_plan->continuation_domain ||
            !declared_continuation_root->world_rank_known ||
            declared_continuation_root->world_rank !=
                execution_plan_->continuation_root_rank)
        {
            throw std::runtime_error(
                "Participant graph could not resolve the planner-declared continuation root participant");
        }
        const int local_world_rank = config_.mpi_context->rank();
        const int source_world_rank =
            execution_plan_->continuation_root_rank;
        if (local_world_rank == source_world_rank)
        {
            throw std::logic_error(
                "Remote ExpertOverlay participant graph cannot run on the continuation authority rank");
        }

        const size_t row_capacity_size =
            static_cast<size_t>(row_capacity);
        const size_t top_k_size = static_cast<size_t>(top_k);
        if (row_capacity_size >
            std::numeric_limits<size_t>::max() / top_k_size)
        {
            throw std::overflow_error(
                "Participant graph sparse entry capacity overflows size_t");
        }
        const size_t entry_capacity =
            row_capacity_size * top_k_size;

        /*
         * Rank transport identity belongs to the complete serial graph family,
         * not to the currently selected local compute specialization. The
         * continuation graph owns one maximum-capacity channel reused by main,
         * decode, and every MTP depth. A draft endpoint may execute a four-row
         * retained graph inside that channel, but deriving its POSIX mapping
         * from four rows would make it wait on a different object from the
         * continuation's prefill-sized mapping. Keep local graph admission at
         * `row_capacity_size` while both ranks derive transport storage from the
         * planner-published family maximum.
         */
        const size_t transport_row_capacity_size =
            static_cast<size_t>(config_.max_graph_activation_rows);
        if (transport_row_capacity_size == 0 ||
            transport_row_capacity_size < row_capacity_size ||
            transport_row_capacity_size >
                std::numeric_limits<size_t>::max() / top_k_size)
        {
            throw std::overflow_error(
                "Participant graph rank-batch family capacity is invalid");
        }
        const size_t transport_entry_capacity =
            transport_row_capacity_size * top_k_size;

        /*
         * A batch receives all local endpoint packets at once. Each endpoint
         * therefore owns one independent packet family, while layers reuse that
         * endpoint family serially. This is the smallest setup-only allocation
         * compatible with concurrent participant device streams.
         */
        std::unordered_map<
            int,
            std::shared_ptr<MoEOverlayCollectiveWorkspace>>
            protocol_workspaces;
        const auto workspace_for_participant =
            [&](int participant)
                -> std::shared_ptr<MoEOverlayCollectiveWorkspace>
        {
            const auto existing =
                protocol_workspaces.find(participant);
            if (existing != protocol_workspaces.end())
                return existing->second;
            auto workspace =
                std::make_shared<MoEOverlayCollectiveWorkspace>(
                    MoEOverlayCollectiveWorkspace::FixedCapacityConfig{
                        .max_rows = row_capacity_size,
                        .max_entries = entry_capacity,
                        .d_model = d_model,
                        .top_k = top_k,
                        .device = DeviceId::cpu(),
                        .reuse_policy =
                            MoEOverlayCollectiveWorkspace::
                                StorageReusePolicy::SerialGraphFamily,
                    });
            protocol_workspaces.emplace(participant, workspace);
            return workspace;
        };

        const auto domain_ordinal =
            [&](const std::string &domain_name)
        {
            for (size_t index = 0;
                 index < config_.placement_plan->domains.size();
                 ++index)
            {
                if (config_.placement_plan->domains[index].name ==
                    domain_name)
                {
                    return static_cast<int>(index);
                }
            }
            throw std::logic_error(
                "Participant rank batch has no stable routed-domain ordinal for " +
                domain_name);
        };

        const auto *const continuation_descriptor =
            owner_map_->participantForId(continuation_root);
        if (!continuation_descriptor ||
            !continuation_descriptor->world_rank_known ||
            continuation_descriptor->world_rank != source_world_rank ||
            continuation_descriptor->tier_idx < 0 ||
            static_cast<size_t>(continuation_descriptor->tier_idx) >=
                config_.placement_plan->routed_tiers.size())
        {
            throw std::logic_error(
                "Participant activation channel cannot resolve its continuation endpoint topology");
        }
        const MoEOverlayActivationLaneEndpoint continuation_endpoint{
            .world_rank = source_world_rank,
            .participant_id = continuation_root,
            .tier_priority =
                config_.placement_plan
                    ->routed_tiers[static_cast<size_t>(
                        continuation_descriptor->tier_idx)]
                    .priority,
            .domain_ordinal =
                domain_ordinal(continuation_descriptor->domain_name),
        };
        const auto activation_graph_families =
            makeMoEOverlayActivationGraphFamilyManifests(
                transaction_graph_family_);

        /*
         * One transport owns one immutable rank/tier participant group and is
         * shared by every transformer layer. Its fixed replay ledger keys each
         * layer separately; its in-flight guard makes accidental graph overlap
         * a fatal ownership error.
         */
        const auto transport_for_group =
            [&](int tier_index,
                int routed_domain_ordinal,
                const std::vector<int> &participants)
                -> std::shared_ptr<IMoEOverlayRankBatchTransport>
        {
            if (participants.empty())
            {
                throw std::invalid_argument(
                    "Participant rank batch cannot bind an empty group");
            }
            const std::string key =
                makeMoEOverlayRankBatchChannelIdentity(
                    tier_index,
                    routed_domain_ordinal,
                    source_world_rank,
                    local_world_rank,
                    participants);
            const auto existing =
                rank_batch_transports_.find(key);
            if (existing != rank_batch_transports_.end())
                return existing->second;

            if (resolveMoEOverlayRankBatchTransportKind(
                    *config_.mpi_context,
                    source_world_rank,
                    local_world_rank) ==
                MoEOverlayRankBatchTransportKind::NodeLocalSharedRows)
            {
                if (!config_.rank_batch_transport_registry)
                {
                    throw std::logic_error(
                        "Participant graph has no preflight node-local activation-channel registry for " +
                        key);
                }
                auto transport =
                    config_.rank_batch_transport_registry->require(
                        key,
                        source_world_rank,
                        local_world_rank,
                        participants);
                rank_batch_transports_.emplace(key, transport);
                return transport;
            }

            const size_t row_multiplier =
                std::min(top_k_size, participants.size());
            if (transport_row_capacity_size >
                std::numeric_limits<size_t>::max() /
                    row_multiplier)
            {
                throw std::overflow_error(
                    "Participant rank-batch row capacity overflows size_t");
            }
            auto wire_workspace =
                std::make_shared<MoEOverlayRankBatchWireWorkspace>(
                    MoEOverlayRankBatchWireWorkspace::Config{
                        .participant_ids = participants,
                        .max_total_rows =
                            transport_row_capacity_size * row_multiplier,
                        .max_total_entries = transport_entry_capacity,
                        .d_model = d_model,
                        .top_k = top_k,
                    });

            std::vector<MoEOverlayActivationLocalLaneBinding> local_lanes;
            for (const int participant_id : participants)
            {
                const auto *const participant =
                    owner_map_->participantForId(participant_id);
                if (!participant || !participant->world_rank_known ||
                    participant->world_rank != local_world_rank ||
                    participant->tier_idx != tier_index ||
                    domain_ordinal(participant->domain_name) !=
                        routed_domain_ordinal ||
                    !participant->device.is_valid())
                {
                    throw std::logic_error(
                        "Participant activation channel group disagrees with the immutable owner map");
                }
                local_lanes.push_back({
                    .participant_id = participant_id,
                    .device = participant->device,
                });
            }
            auto transport = createMoEOverlayRankBatchTransport(
                    MoEOverlayRankBatchTransportConfig{
                        .mpi_ctx = config_.mpi_context,
                        .source_world_rank = source_world_rank,
                        .target_world_rank = local_world_rank,
                        .workspace = std::move(wire_workspace),
                        .max_rows_per_participant =
                            transport_row_capacity_size,
                        .max_entries_per_participant =
                            transport_entry_capacity,
                        .d_model = d_model,
                        .top_k = top_k,
                        .tier_index = tier_index,
                        .domain_ordinal = routed_domain_ordinal,
                        .channel_identity = key,
                        .transaction_slot_count = 4096,
                        .transaction_topology =
                            transaction_topology_identity_,
                        .source_endpoint = continuation_endpoint,
                        .target_tier_priority =
                            config_.placement_plan
                                ->routed_tiers[static_cast<size_t>(tier_index)]
                                .priority,
                        .activation_graph_families =
                            activation_graph_families,
                        .local_lanes = std::move(local_lanes),
                    });
            rank_batch_transports_.emplace(key, transport);
            return transport;
        };

        /*
         * A same-node rank pair does not need a host-scheduled sparse graph.
         * Materialize one complete retained follower parent per physical GPU
         * and per admitted row geometry. Every endpoint graph owns the exact
         * dispatch-consume -> expert-compute -> return-pack sequence for all
         * source layers. The transaction executor launches the whole endpoint
         * set before observing any terminal event, so independent CUDA/ROCm
         * streams overlap instead of degenerating into submit/wait pairs.
         */
        const bool node_local_activation_epoch =
            resolveMoEOverlayRankBatchTransportKind(
                *config_.mpi_context,
                source_world_rank,
                local_world_rank) ==
            MoEOverlayRankBatchTransportKind::NodeLocalSharedRows;
        if (node_local_activation_epoch)
        {
            struct MappedParticipantBinding
            {
                const MoEExpertOwnerParticipant *participant = nullptr;
                const RoutedExpertTier *tier = nullptr;
                int tier_index = -1;
                int routed_domain_ordinal = -1;
                std::shared_ptr<IMoEOverlayRankBatchTransport> transport;
                IMoEOverlayMappedActivationTransport *mapped = nullptr;
                std::shared_ptr<CachedParticipantGraph::MappedLaneAuthority>
                    authority;
            };

            const size_t graph_family_ordinal =
                spec.role == ParticipantGraphFamilyRole::Main
                    ? 0u
                    : static_cast<size_t>(spec.mtp_graph_depth + 1);
            std::vector<MappedParticipantBinding> bindings;
            bindings.reserve(local_participants_.size());

            for (size_t tier_index = 0;
                 tier_index < config_.placement_plan->routed_tiers.size();
                 ++tier_index)
            {
                const auto &tier =
                    config_.placement_plan->routed_tiers[tier_index];
                const int routed_domain_ordinal =
                    domain_ordinal(tier.domain);
                std::vector<int> participants;
                for (const auto *participant : local_participants_)
                {
                    if (!participant ||
                        participant->tier_idx !=
                            static_cast<int>(tier_index) ||
                        participant->domain_name != tier.domain ||
                        participant->world_rank != local_world_rank)
                    {
                        continue;
                    }
                    participants.push_back(participant->participant_id);
                }
                if (participants.empty())
                    continue;
                std::sort(participants.begin(), participants.end());

                auto transport = transport_for_group(
                    static_cast<int>(tier_index),
                    routed_domain_ordinal,
                    participants);
                auto *const mapped = dynamic_cast<
                    IMoEOverlayMappedActivationTransport *>(transport.get());
                if (!mapped || graph_family_ordinal >=
                                   mapped->activationGraphFamilyCount())
                {
                    throw std::runtime_error(
                        "Node-local ExpertOverlay participant transport did not expose its required mapped activation family");
                }

                for (const int participant_id : participants)
                {
                    const auto *const participant =
                        owner_map_->participantForId(participant_id);
                    if (!participant || !participant->device.is_valid())
                    {
                        throw std::runtime_error(
                            "Node-local mapped ExpertOverlay resolved an invalid participant endpoint");
                    }
                    auto authority = std::make_shared<
                        CachedParticipantGraph::MappedLaneAuthority>();
                    authority->transport_lifetime = transport;
                    authority->participant_id = participant_id;
                    authority->tier_index = static_cast<int>(tier_index);
                    authority->tier_priority = tier.priority;
                    authority->routed_domain_ordinal =
                        routed_domain_ordinal;
                    authority->device = participant->device;
                    authority->graph_family_ordinal = graph_family_ordinal;
                    authority->protocol =
                        std::make_unique<MoEOverlayActivationEpochProtocol>(
                            mapped->activationEpochControl(
                                participant_id, graph_family_ordinal),
                            mapped->activationEpochConfig(
                                participant_id, graph_family_ordinal));
                    result->mapped_lane_authorities.push_back(authority);
                    bindings.push_back(MappedParticipantBinding{
                        .participant = participant,
                        .tier = &tier,
                        .tier_index = static_cast<int>(tier_index),
                        .routed_domain_ordinal = routed_domain_ordinal,
                        .transport = transport,
                        .mapped = mapped,
                        .authority = std::move(authority),
                    });
                }
            }
            if (bindings.empty())
            {
                throw std::runtime_error(
                    "Node-local ExpertOverlay participant graph resolved no mapped endpoints");
            }

            const auto first_arena =
                serialCompactBufferArenaForParticipant(
                    *bindings.front().participant);
            std::vector<int> row_shapes;
            for (const auto &family : first_arena->families())
            {
                if (family.row_capacity == 0u ||
                    family.row_capacity >
                        static_cast<size_t>(row_capacity))
                {
                    continue;
                }
                if (family.row_capacity >
                    static_cast<size_t>(
                        std::numeric_limits<int>::max()))
                {
                    throw std::overflow_error(
                        "Mapped ExpertOverlay follower row geometry exceeds int");
                }
                row_shapes.push_back(
                    static_cast<int>(family.row_capacity));
            }
            if (row_shapes.empty() || row_shapes.back() != row_capacity)
            {
                throw std::logic_error(
                    "Mapped ExpertOverlay follower arena does not contain the admitted exact row geometry");
            }

            /*
             * CPU endpoints consume the same shared-page lanes as GPU parents,
             * but remain explicit host boundaries because no native GPU graph
             * can own CPU compute. Build them once at the admitted maximum and
             * reuse their stable packet views for every exact ticket shape. GPU
             * siblings are still submitted before this host loop starts.
             */
            for (const auto &binding : bindings)
            {
                if (!binding.participant->device.is_cpu())
                    continue;

                auto endpoint = std::make_unique<
                    CachedParticipantGraph::MappedCPUFollowerEndpoint>();
                endpoint->participant_id =
                    binding.participant->participant_id;
                endpoint->device = binding.participant->device;
                endpoint->tier_index = binding.tier_index;
                endpoint->routed_domain_ordinal =
                    binding.routed_domain_ordinal;
                endpoint->lane_authority = binding.authority;
                endpoint->lane = binding.mapped->activationDeviceLane(
                    endpoint->participant_id,
                    graph_family_ordinal,
                    endpoint->device);
                endpoint->input_rows =
                    std::make_shared<MoEOverlaySparseRows>(
                        endpoint->lane.hostDispatchPayload(row_capacity));
                endpoint->output_rows =
                    std::make_shared<MoEOverlayReturnRows>(
                        binding.transport->sharedReturnRows(
                            endpoint->participant_id));
                endpoint->layers.reserve(spec.source_layers.size());

                const auto participant_residency =
                    config_.participant_residency->endpoint(
                        endpoint->participant_id);
                if (!participant_residency)
                {
                    throw std::runtime_error(
                        "Mapped ExpertOverlay CPU endpoint is missing its prepared residency authority");
                }
                const auto compact_arena =
                    serialCompactBufferArenaForParticipant(
                        *binding.participant);

                for (const int layer : spec.source_layers)
                {
                    const auto expert_mask =
                        owner_map_->expertMaskForParticipant(
                            layer,
                            endpoint->participant_id,
                            num_experts);
                    std::vector<MoEOverlayPreparedExpertTriplet>
                        prepared_triplets;
                    std::string residency_error;
                    if (!resolveMoEOverlayPreparedExpertTriplets(
                            registry,
                            *binding.participant,
                            layer,
                            num_experts,
                            expert_mask,
                            prepared_triplets,
                            &residency_error) ||
                        !config_.participant_residency
                             ->registerInitialLayer(
                                 endpoint->participant_id,
                                 layer,
                                 expert_mask,
                                 prepared_triplets,
                                 &residency_error))
                    {
                        throw std::runtime_error(
                            "Mapped ExpertOverlay CPU follower could not publish its initial residency bank for layer " +
                            std::to_string(layer) + " participant " +
                            std::to_string(endpoint->participant_id) +
                            ": " + residency_error);
                    }

                    MoELocalExpertStage::Params local_params;
                    local_params.device_id = endpoint->device;
                    local_params.input_rows_lifetime = endpoint->input_rows;
                    local_params.output_rows_lifetime = endpoint->output_rows;
                    local_params.cpu_canonical_route_return =
                        MoELocalExpertStage::CPUCanonicalRouteReturnBinding{
                            .original_route_slots =
                                endpoint->lane.dispatch.original_route_slots,
                            .compact_route_slots =
                                endpoint->lane.dispatch.compact_route_slots,
                            .preweighted_route_contributions =
                                endpoint->lane.returned
                                    .canonical_route_contributions_fp32,
                            .route_slot_capacity =
                                endpoint->lane.returned.route_slot_capacity,
                        };
                    local_params.serial_compact_buffer_arena = compact_arena;
                    local_params.graph_row_capacity =
                        static_cast<size_t>(row_capacity);
                    local_params.completion_policy =
                        MoELocalExpertStage::CompletionPolicy::Inline;
                    local_params.num_experts = num_experts;
                    local_params.top_k = top_k;
                    local_params.d_model = d_model;
                    local_params.expert_intermediate = expert_intermediate;
                    local_params.layer_idx = layer;
                    local_params.expert_mask = expert_mask;
                    local_params.overlay_participant_residency =
                        participant_residency;
                    local_params.prepared_store = prepared_store_.get();
                    local_params.expert_registry = &registry;
                    local_params.runtime_participant_index =
                        endpoint->participant_id;
                    local_params.expert_weight_resolution_policy =
                        MoELocalExpertStage::
                            ExpertWeightResolutionPolicy::RegistryOnly;

                    (void)registry.populateExpertEnginesForParticipant(
                        binding.tier->domain,
                        endpoint->device,
                        binding.participant->world_rank,
                        binding.participant->domain_participant_index,
                        layer,
                        num_experts,
                        local_params.prepared_gate_gemm,
                        local_params.prepared_up_gemm,
                        local_params.prepared_down_gemm);
                    /*
                     * An initially empty tier deliberately has no engines yet.
                     * The epoch-indexed residency bank can install arrivals later,
                     * and the stage resolves that exact bank on every invocation.
                     */
                    if (hasActiveMask(expert_mask) &&
                        !MoELocalExpertStage::prepareExpertGemmEngines(
                            local_params))
                    {
                        throw std::runtime_error(
                            "Mapped ExpertOverlay CPU follower has incomplete prepared expert engines for layer " +
                            std::to_string(layer) + " participant " +
                            std::to_string(endpoint->participant_id));
                    }

                    endpoint->layers.push_back(
                        CachedParticipantGraph::MappedCPUFollowerLayer{
                            .model_layer_index = layer,
                            .stage_ordinal =
                                binding.mapped->activationStageOrdinal(
                                    graph_family_ordinal, layer),
                            .local_expert =
                                std::make_unique<MoELocalExpertStage>(
                                    std::move(local_params)),
                        });
                }
                result->mapped_cpu_endpoints.push_back(std::move(endpoint));
            }

            /*
             * Materialize the largest geometry first. The runner-owned
             * WorkspaceAllocator spans main, verifier, and MTP families as
             * well as every physical-row specialization. It therefore binds
             * every smaller retained graph to the already-published maximum
             * buffers and rejects any shape the setup BOM omitted. Iterating
             * in ascending order would ask a live serial allocator to grow
             * after its first graph captured addresses, which is intentionally
             * forbidden.
             */
            for (auto shape_it = row_shapes.rbegin();
                 shape_it != row_shapes.rend();
                 ++shape_it)
            {
                const int physical_rows = *shape_it;
                CachedParticipantGraph::MappedGPUFollowerShape shape;
                shape.physical_rows = physical_rows;
                shape.endpoints.reserve(bindings.size());

                for (const auto &binding : bindings)
                {
                    if (!binding.participant->device.is_gpu())
                        continue;
                    const DeviceId target_device =
                        binding.participant->device;
                    auto endpoint = std::make_unique<
                        CachedParticipantGraph::MappedGPUFollowerEndpoint>();
                    endpoint->participant_id =
                        binding.participant->participant_id;
                    endpoint->device = target_device;
                    endpoint->physical_rows = physical_rows;
                    endpoint->lane_authority = binding.authority;
                    endpoint->graph = std::make_unique<ComputeGraph>();
                    endpoint->graph->setNativeCaptureEnvelope(
                        GraphNativeCaptureEnvelope::
                            DeviceOwnedTimelineTransaction);
                    endpoint->arena = std::make_unique<BufferArena>();
                    endpoint->active_rows =
                        std::make_shared<INT32Tensor>(
                            std::vector<size_t>{1u});

                    const auto compact_arena =
                        serialCompactBufferArenaForParticipant(
                            *binding.participant);
                    const auto *const tensor_family =
                        compact_arena->smallestFamilySupporting(
                            static_cast<size_t>(physical_rows));
                    if (!tensor_family ||
                        tensor_family->row_capacity !=
                            static_cast<size_t>(physical_rows) ||
                        !tensor_family->hidden ||
                        !tensor_family->routing_indices ||
                        !tensor_family->routing_weights ||
                        !tensor_family->output)
                    {
                        throw std::logic_error(
                            "Mapped ExpertOverlay follower could not bind an exact compact tensor family");
                    }
                    auto canonical_it =
                        participant_canonical_route_buffers_.find(
                            endpoint->participant_id);
                    if (canonical_it ==
                        participant_canonical_route_buffers_.end())
                    {
                        /* Only the node-local mapped GPU path consumes this
                         * bank. Allocate it on first graph-family materialization
                         * at the shared transport maximum, then reuse that stable
                         * address for main, verifier, and every MTP family whose
                         * executions are scheduler-serialized. */
                        auto canonical_routes =
                            std::make_shared<FP32Tensor>(
                                std::vector<size_t>{
                                    transport_entry_capacity,
                                    static_cast<size_t>(d_model)});
                        canonical_it =
                            participant_canonical_route_buffers_
                                .emplace(
                                    endpoint->participant_id,
                                    std::move(canonical_routes))
                                .first;
                    }
                    if (canonical_it ==
                            participant_canonical_route_buffers_.end() ||
                        !canonical_it->second ||
                        canonical_it->second->numel() !=
                            checkedGeometryProduct(
                                transport_entry_capacity,
                                static_cast<size_t>(d_model),
                                "participant canonical route elements"))
                    {
                        throw std::logic_error(
                            "Mapped ExpertOverlay follower has no participant-owned canonical route bank");
                    }
                    const auto &canonical_routes = canonical_it->second;

                    TransferEngine::allocateDeviceStorage(
                        tensor_family->hidden.get(), target_device);
                    TransferEngine::allocateDeviceStorage(
                        tensor_family->routing_indices.get(), target_device);
                    TransferEngine::allocateDeviceStorage(
                        tensor_family->routing_weights.get(), target_device);
                    TransferEngine::allocateDeviceStorage(
                        tensor_family->output.get(), target_device);
                    TransferEngine::allocateDeviceStorage(
                        canonical_routes.get(), target_device);
                    TransferEngine::allocateDeviceStorage(
                        endpoint->active_rows.get(), target_device);
                    if (!endpoint->active_rows->gpu_data_ptr())
                    {
                        throw std::runtime_error(
                            "Mapped ExpertOverlay follower active-row scalar has no device storage");
                    }

                    if (!endpoint->arena->registerExternalBuffer(
                            BufferId::NORMALIZED,
                            tensor_family->hidden.get()) ||
                        !endpoint->arena->registerExternalBuffer(
                            BufferId::MOE_EXPERT_INDICES,
                            tensor_family->routing_indices.get()) ||
                        !endpoint->arena->registerExternalBuffer(
                            BufferId::MOE_EXPERT_WEIGHTS,
                            tensor_family->routing_weights.get()) ||
                        !endpoint->arena->registerExternalBuffer(
                            BufferId::MOE_COMBINED_OUTPUT,
                            tensor_family->output.get()) ||
                        !endpoint->arena->registerExternalBuffer(
                            BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS,
                            canonical_routes.get()))
                    {
                        throw std::runtime_error(
                            "Mapped ExpertOverlay follower could not register its complete arena frontier");
                    }

                    endpoint->tensor_lifetimes = {
                        tensor_family->hidden,
                        tensor_family->routing_indices,
                        tensor_family->routing_weights,
                        tensor_family->output,
                        canonical_routes,
                    };

                    auto lane = binding.mapped->activationDeviceLane(
                        endpoint->participant_id,
                        graph_family_ordinal,
                        target_device);
                    if (!lane.valid())
                    {
                        throw std::runtime_error(
                            "Mapped ExpertOverlay follower received an invalid device lane");
                    }
                    endpoint->lane = lane;

                    ParticipantGpuRuntime *gpu_runtime =
                        &participantGpuRuntimeForParticipant(
                            endpoint->participant_id);
                    std::string epoch_acquire_name;
                    if (gpu_runtime->device != target_device ||
                        !gpu_runtime->runtime_table ||
                        !gpu_runtime->epoch_arena)
                    {
                        throw std::logic_error(
                            "Mapped follower endpoint disagrees with its durable GPU runtime authority");
                    }

                    epoch_acquire_name =
                        "mapped_follower_p" +
                        std::to_string(endpoint->participant_id) +
                        "_epoch_acquire";
                    MoEOverlayEpochBoundaryStage::Params acquire_params;
                    acquire_params.device_id = target_device;
                    acquire_params.arena = gpu_runtime->epoch_arena;
                    acquire_params.request_slot = 0u;
                    acquire_params.operation =
                        MoEOverlayEpochBoundaryStage::Operation::Acquire;
                    if (spec.source_layers.empty())
                    {
                        throw std::logic_error(
                            "Mapped follower epoch transaction has no routed layer");
                    }
                    const std::uint32_t first_stage_ordinal =
                        binding.mapped->activationStageOrdinal(
                            graph_family_ordinal,
                            spec.source_layers.front());
                    if (first_stage_ordinal != 0u)
                    {
                        throw std::logic_error(
                            "Mapped follower epoch transaction does not begin at stage zero");
                    }
                    acquire_params.peer_epoch_source =
                        MoEOverlayEpochBoundaryStage::PeerEpochSource{
                            .mapped_region = lane.mapped_region,
                            .binding = {
                                .control = lane.control_device,
                                .grant = lane.grant_device,
                                .stage_ordinal = first_stage_ordinal,
                            },
                        };
                    acquire_params.stage_name = epoch_acquire_name;
                    endpoint->graph->addNode(
                        epoch_acquire_name,
                        std::make_unique<MoEOverlayEpochBoundaryStage>(
                            std::move(acquire_params)),
                        target_device);
                    std::string previous_layer_return;
                    for (const int layer : spec.source_layers)
                    {
                        const std::uint32_t stage_ordinal =
                            binding.mapped->activationStageOrdinal(
                                graph_family_ordinal, layer);
                        const std::string stem =
                            "mapped_follower_p" +
                            std::to_string(endpoint->participant_id) +
                            "_layer" + std::to_string(layer);
                        const std::string consume_name =
                            stem + "_dispatch_consume";

                        MoEOverlayActivationDispatchConsumeStage::Params
                            consume_params;
                        consume_params.device_id = target_device;
                        consume_params.lane = lane;
                        consume_params.placement =
                            gpu_runtime->runtime_table
                                ->overlayRoutePlacementBinding(layer);
                        consume_params.hidden = tensor_family->hidden.get();
                        consume_params.routing_indices =
                            tensor_family->routing_indices.get();
                        consume_params.routing_weights =
                            tensor_family->routing_weights.get();
                        consume_params.active_row_count_device =
                            static_cast<std::int32_t *>(
                                endpoint->active_rows->gpu_data_ptr());
                        consume_params.physical_rows = physical_rows;
                        consume_params.stage_ordinal = stage_ordinal;
                        consume_params.model_layer_index = layer;
                        endpoint->graph->addNode(
                            consume_name,
                            ComputeStageFactory::
                                createMoEOverlayActivationDispatchConsume(
                                    consume_params),
                            target_device);
                        if (!previous_layer_return.empty())
                        {
                            endpoint->graph->addDependency(
                                consume_name, previous_layer_return);
                        }
                        else if (!epoch_acquire_name.empty())
                        {
                            endpoint->graph->addDependency(
                                consume_name, epoch_acquire_name);
                        }

                        const auto expert_mask =
                            owner_map_->expertMaskForParticipant(
                                layer,
                                endpoint->participant_id,
                                num_experts);
                        std::vector<MoEOverlayPreparedExpertTriplet>
                            prepared_triplets;
                        std::string residency_error;
                        if (!resolveMoEOverlayPreparedExpertTriplets(
                                registry,
                                *binding.participant,
                                layer,
                                num_experts,
                                expert_mask,
                                prepared_triplets,
                                &residency_error) ||
                            !config_.participant_residency
                                 ->registerInitialLayer(
                                     endpoint->participant_id,
                                     layer,
                                     expert_mask,
                                     prepared_triplets,
                                     &residency_error))
                        {
                            throw std::runtime_error(
                                "Mapped ExpertOverlay follower could not publish its initial residency bank for layer " +
                                std::to_string(layer) + " participant " +
                                std::to_string(endpoint->participant_id) +
                                ": " + residency_error);
                        }
                        std::string compute_dependency = consume_name;
                        std::vector<ITensorGemm *> prepared_gate;
                        std::vector<ITensorGemm *> prepared_up;
                        std::vector<ITensorGemm *> prepared_down;
                        (void)registry.populateExpertEnginesForParticipant(
                            binding.tier->domain,
                            target_device,
                            binding.participant->world_rank,
                            binding.participant->domain_participant_index,
                            layer,
                            num_experts,
                            prepared_gate,
                            prepared_up,
                            prepared_down);
                        if (hasActiveMask(expert_mask))
                        {
                            MoELocalExpertStage::Params validation_params;
                            validation_params.device_id = target_device;
                            validation_params.num_experts = num_experts;
                            validation_params.top_k = top_k;
                            validation_params.d_model = d_model;
                            validation_params.expert_intermediate =
                                expert_intermediate;
                            validation_params.layer_idx = layer;
                            validation_params.expert_mask = expert_mask;
                            validation_params.prepared_gate_gemm =
                                prepared_gate;
                            validation_params.prepared_up_gemm = prepared_up;
                            validation_params.prepared_down_gemm =
                                prepared_down;
                            validation_params.expert_weight_resolution_policy =
                                MoELocalExpertStage::
                                    ExpertWeightResolutionPolicy::RegistryOnly;
                            if (!MoELocalExpertStage::
                                     prepareExpertGemmEngines(
                                         validation_params))
                            {
                                throw std::runtime_error(
                                    "Mapped ExpertOverlay follower has incomplete prepared expert engines for layer " +
                                    std::to_string(layer) + " participant " +
                                    std::to_string(endpoint->participant_id));
                            }
                        }

                        if (layer < 0 ||
                            static_cast<std::size_t>(layer) >=
                                gpu_runtime->initialized_layers.size())
                        {
                            throw std::out_of_range(
                                "Mapped follower source layer is outside its topology-wide runtime table");
                        }
                        auto &initialized =
                            gpu_runtime->initialized_layers[
                                static_cast<std::size_t>(layer)];
                        if (initialized == 0u)
                        {
                                    MoEPlacementUpdate update;
                                    update.epoch = 1u;
                                    update.expert_count =
                                        static_cast<std::uint32_t>(
                                            num_experts);
                                    update.participant_id =
                                        gpu_runtime->domain_participant_id;
                                    update.participant_count =
                                        gpu_runtime->domain_participant_count;
                                    update.experts.resize(
                                        static_cast<std::size_t>(
                                            num_experts));
                                    update.local_compute_mask.assign(
                                        static_cast<std::size_t>(
                                            num_experts),
                                        0u);
                                    update.replica_role.assign(
                                        static_cast<std::size_t>(
                                            num_experts),
                                        static_cast<std::uint8_t>(
                                            DeviceMoEReplicaRole::None));
                                    update.resident_participant_mask.assign(
                                        static_cast<std::size_t>(
                                            num_experts),
                                        0u);
                                    update.overlay_route_participant.assign(
                                        static_cast<std::size_t>(
                                            num_experts),
                                        -1);

                                    for (int expert = 0;
                                         expert < num_experts;
                                         ++expert)
                                    {
                                        const auto *const owner =
                                            owner_map_->ownerFor(
                                                layer, expert);
                                        if (!owner ||
                                            owner->owner_participant < 0)
                                        {
                                            throw std::logic_error(
                                                "Mapped follower runtime initialization found an unowned expert");
                                        }

                                        auto &descriptor =
                                            update.experts[
                                                static_cast<std::size_t>(
                                                    expert)];
                                        descriptor.logical_expert_id = expert;
                                        update.overlay_route_participant[
                                            static_cast<std::size_t>(expert)] =
                                            owner->owner_participant;
                                        if (owner->tier_idx ==
                                            binding.participant->tier_idx)
                                        {
                                            if (owner
                                                        ->domain_participant_index <
                                                    0 ||
                                                static_cast<std::uint32_t>(
                                                    owner
                                                        ->domain_participant_index) >=
                                                    update.participant_count)
                                            {
                                                throw std::logic_error(
                                                    "Mapped follower owner has no valid domain-local runtime identity");
                                            }
                                            descriptor.owner_participant =
                                                owner
                                                    ->domain_participant_index;
                                            update
                                                .resident_participant_mask[
                                                    static_cast<std::size_t>(
                                                        expert)] =
                                                1u << static_cast<std::uint32_t>(
                                                    owner
                                                        ->domain_participant_index);
                                        }
                                        else
                                        {
                                            descriptor.owner_participant = -1;
                                        }

                                        if (owner->owner_participant !=
                                            endpoint->participant_id)
                                        {
                                            continue;
                                        }
                                        if (!expert_mask[
                                                static_cast<std::size_t>(
                                                    expert)] ||
                                            !exportDeviceMoEExpertWeightDescriptors(
                                                prepared_gate[
                                                    static_cast<std::size_t>(
                                                        expert)],
                                                prepared_up[
                                                    static_cast<std::size_t>(
                                                        expert)],
                                                prepared_down[
                                                    static_cast<std::size_t>(
                                                        expert)],
                                                d_model,
                                                expert_intermediate,
                                                descriptor))
                                        {
                                            throw std::runtime_error(
                                                "Mapped follower could not export the authoritative prepared expert descriptor");
                                        }
                                        descriptor.logical_expert_id = expert;
                                        descriptor.owner_participant =
                                            static_cast<std::int32_t>(
                                                update.participant_id);
                                        descriptor.local_slot = expert;
                                        descriptor.flags = toMoEExpertFlags(
                                            DeviceMoEExpertFlags::Valid |
                                            DeviceMoEExpertFlags::Resident |
                                            DeviceMoEExpertFlags::LocalCompute |
                                            DeviceMoEExpertFlags::PreferredOwner);
                                        update.local_compute_mask[
                                            static_cast<std::size_t>(expert)] =
                                            1u;
                                        update.replica_role[
                                            static_cast<std::size_t>(expert)] =
                                            static_cast<std::uint8_t>(
                                                DeviceMoEReplicaRole::Primary);
                                        update
                                            .resident_participant_mask[
                                                static_cast<std::size_t>(
                                                    expert)] |=
                                            1u << update.participant_id;
                                    }

                                    void *const publication_stream =
                                        participantWorkerStream(
                                            target_device);
                                    gpu_runtime->worker->submitAndWait(
                                        [&]
                                        {
                                            gpu_runtime
                                                ->runtime_table
                                                ->prepareInactiveBank(
                                                    layer, update);
                                            gpu_runtime
                                                ->runtime_table
                                                ->flipActiveBank(
                                                    layer,
                                                    update.epoch,
                                                    publication_stream);
                                        });
                            initialized = 1u;
                        }

                        if (hasActiveMask(expert_mask))
                        {
                            MoEExpertComputeStage::Params compute_params;
                            compute_params.device_id = target_device;
                            compute_params.input = tensor_family->hidden.get();
                            compute_params.seq_len = physical_rows;
                            compute_params.d_model = d_model;
                            compute_params.num_experts = num_experts;
                            compute_params.top_k = top_k;
                            compute_params.expert_intermediate =
                                expert_intermediate;
                            compute_params.layer_idx = layer;
                            compute_params.expert_mask = expert_mask;
                            compute_params.routing_indices =
                                tensor_family->routing_indices.get();
                            compute_params.routing_weights =
                                tensor_family->routing_weights.get();
                            compute_params.output =
                                tensor_family->output.get();
                            compute_params.canonical_route_contributions =
                                canonical_routes.get();
                            compute_params.canonical_route_arithmetic =
                                MoECanonicalRouteArithmeticPolicy::
                                    PreweightedContributionThenOrderedAdd;
                            compute_params.canonical_route_layout =
                                MoECanonicalRoutePublicationLayout::
                                    DenseOriginalRouteSlots;
                            compute_params.active_row_count_device =
                                static_cast<const std::int32_t *>(
                                    endpoint->active_rows->gpu_data_ptr());
                            /*
                             * A one-row follower consumes the packet's
                             * device-owned router tensors through the fused
                             * explicit-routing decode path. It is ordinary
                             * decode, not an MTP grouped-prefill verifier, and
                             * must retain the economical M=1 kernel route.
                             */
                            compute_params
                                .require_device_routing_tensor_decode = true;
                            compute_params.output_registered_in_arena = true;
                            compute_params.prepared_gate_gemm =
                                std::move(prepared_gate);
                            compute_params.prepared_up_gemm =
                                std::move(prepared_up);
                            compute_params.prepared_down_gemm =
                                std::move(prepared_down);
                            compute_params.prepared_store =
                                prepared_store_.get();
                            compute_params.expert_registry = &registry;
                            compute_params.moe_runtime_table =
                                gpu_runtime->runtime_table.get();
                            compute_params
                                .runtime_service_graph_role_device =
                                &lane.grant_device->graph_role;
                            compute_params.use_runtime_row_grouping = true;
                            compute_params.weight_descriptor_source =
                                MoEDecodeDescriptorSource::
                                    RuntimePlacementTable;
                            compute_params
                                .runtime_decode_has_explicit_owner_metadata =
                                true;
                            compute_params.my_socket_id =
                                static_cast<int>(
                                    gpu_runtime->domain_participant_id);
                            compute_params.participant_count =
                                static_cast<int>(
                                    gpu_runtime->domain_participant_count);
                            compute_params.routed_assignment_policy =
                                RoutedExpertAssignmentPolicy::StaticOwner;
                            compute_params.routed_row_execution_policy =
                                RoutedExpertRowExecutionPolicy::
                                    ParticipantAssigned;

                            auto compute_stage =
                                std::make_unique<MoEExpertComputeStage>(
                                    std::move(compute_params));
                            compute_stage->releaseRawExpertWeights();
                            const std::string compute_name =
                                stem + "_expert_compute";
                            endpoint->graph->addNode(
                                compute_name,
                                std::move(compute_stage),
                                target_device);
                            endpoint->graph->addDependency(
                                compute_name, consume_name);
                            compute_dependency = compute_name;
                        }

                        MoEOverlayActivationReturnPackStage::Params
                            return_params;
                        return_params.device_id = target_device;
                        return_params.lane = lane;
                        return_params.local_canonical_route_contributions =
                            canonical_routes.get();
                        return_params.physical_rows = physical_rows;
                        return_params.stage_ordinal = stage_ordinal;
                        return_params.model_layer_index = layer;
                        const std::string return_name =
                            stem + "_return_pack";
                        endpoint->graph->addNode(
                            return_name,
                            ComputeStageFactory::
                                createMoEOverlayActivationReturnPack(
                                    return_params),
                            target_device);
                        endpoint->graph->addDependency(
                            return_name, compute_dependency);
                        previous_layer_return = return_name;
                    }

                    if (previous_layer_return.empty())
                    {
                        throw std::logic_error(
                            "Mapped follower epoch transaction contains no routed layer to release");
                    }
                    const std::string release_name =
                        "mapped_follower_p" +
                        std::to_string(endpoint->participant_id) +
                        "_epoch_release";
                    MoEOverlayEpochBoundaryStage::Params release_params;
                    release_params.device_id = target_device;
                    release_params.arena = gpu_runtime->epoch_arena;
                    release_params.request_slot = 0u;
                    release_params.operation =
                        MoEOverlayEpochBoundaryStage::Operation::Release;
                    if (config_.durable_maintenance_policy ==
                            MoEOverlayDurableMaintenancePolicy::
                                DynamicPlacement &&
                        gpu_runtime->controller_binding)
                    {
                        release_params.retirement_readiness_controller =
                            *gpu_runtime->controller_binding;
                    }
                    release_params.stage_name = release_name;
                    endpoint->graph->addNode(
                        release_name,
                        std::make_unique<MoEOverlayEpochBoundaryStage>(
                            std::move(release_params)),
                        target_device);
                    endpoint->graph->addDependency(
                        release_name, previous_layer_return);

                    endpoint->workspace_allocator =
                        serial_graph_family_workspace_allocator_;
                    if (!endpoint->workspace_allocator)
                    {
                        throw std::runtime_error(
                            "Mapped ExpertOverlay follower could not create its serial device workspace owner");
                    }
                    WorkspaceSizingHints mapped_hints;
                    mapped_hints.max_seq_len = physical_rows;
                    mapped_hints.serial_family_max_rows = physical_rows;
                    mapped_hints.d_model = d_model;
                    mapped_hints.batch_size = 1;
                    mapped_hints.graph_family_policy =
                        WorkspaceGraphFamilyPolicy::
                            SerialDeviceFamilyExactParticipant;
                    if (!endpoint->workspace_allocator->allocateForGraph(
                            *endpoint->graph, mapped_hints))
                    {
                        throw std::runtime_error(
                            "Mapped ExpertOverlay follower could not allocate graph workspace");
                    }

                    IWorkerGPUContext *const worker =
                        participantWorkerContext(target_device);
                    GraphExecutorConfig executor_config;
                    executor_config.default_device = target_device;
                    executor_config.worker_gpu_context_resolver =
                        [worker, target_device](DeviceId requested)
                            -> IWorkerGPUContext *
                        {
                            return requested == target_device
                                       ? worker
                                       : nullptr;
                        };
                    executor_config.worker_gpu_context_uses_process_pool =
                        false;
                    endpoint->executor =
                        std::make_unique<DeviceGraphExecutor>(
                            executor_config);
                    endpoint->executor->setArena(endpoint->arena.get());
                    endpoint->cache.perf_context =
                        "moe_overlay_mapped_follower_p" +
                        std::to_string(endpoint->participant_id) +
                        "_family" +
                        std::to_string(graph_family_ordinal) +
                        "_rows" + std::to_string(physical_rows);
                    endpoint->cache.steady_replay_host_policy =
                        DeviceGraphExecutor::GraphSegmentCache::
                            SteadyReplayHostPolicy::RetainedFullGraph;
                    if (!endpoint->cache.bindBorrowedCaptureStream(
                            worker,
                            participantWorkerStream(target_device),
                            target_device,
                            /*context_from_process_pool=*/false))
                    {
                        throw std::runtime_error(
                            "Mapped ExpertOverlay follower could not bind its exact participant stream");
                    }
                    worker->submitAndWait(
                        [&]
                        {
                            if (!endpoint->cache.ensureCaptureOutputEvent(worker))
                            {
                                throw std::runtime_error(
                                    "Mapped ExpertOverlay follower could not allocate its setup-owned terminal fence event");
                            }
                        });
                    shape.endpoints.push_back(std::move(endpoint));
                }
                result->mapped_gpu_shapes.push_back(std::move(shape));
            }

            /* Native capture waits for physical-fabric epoch installation.
             * Graphs, buffers, prepared engines, and borrowed streams are all
             * durable now; only the final graph-owned transfer branch identity
             * is intentionally unresolved until orchestration finishes the
             * topology-accounted lane BOM. */
            PerfStatsCollector::addCounter(
                "moe_overlay_participant_graph",
                "declared_mapped_follower_families",
                1.0,
                "model_setup",
                participantDeviceList(local_participants_),
                {{"participants", participantIdList(local_participants_)},
                 {"row_shapes", std::to_string(row_shapes.size())},
                 {"gpu_timeline_executables",
                  std::to_string(
                      result->mapped_gpu_shapes.size() *
                      static_cast<size_t>(std::count_if(
                          bindings.begin(),
                          bindings.end(),
                          [](const auto &binding)
                          {
                              return binding.participant->device.is_gpu();
                          })))},
                 {"cpu_endpoints",
                  std::to_string(result->mapped_cpu_endpoints.size())},
                 {"graph_family", std::to_string(graph_family_ordinal)},
                 {"source_layers", std::to_string(spec.source_layers.size())},
                 {"native_capture", "deferred_until_physical_fabric"},
                 {"host_sparse_stages", "0"}});
            return result;
        }

        std::string previous_protocol_node;
        for (const int layer : spec.source_layers)
        {
            if (layer < 0 ||
                layer >= config_.model_context->totalBlockCount())
            {
                throw std::invalid_argument(
                    "Participant retained graph names an invalid source layer " +
                    std::to_string(layer));
            }
            for (size_t tier_index = 0;
                 tier_index < config_.placement_plan->routed_tiers.size();
                 ++tier_index)
            {
                const auto &tier =
                    config_.placement_plan->routed_tiers[tier_index];
                std::vector<int> targets =
                    owner_map_->participantIdsForTier(
                        static_cast<int>(tier_index));
                std::sort(targets.begin(), targets.end());

                std::vector<int> participants;
                for (const int target : targets)
                {
                    const auto target_mask =
                        owner_map_->expertMaskForParticipant(
                            layer, target, num_experts);
                    if (!hasActiveMask(target_mask))
                        continue;
                    const auto *descriptor =
                        owner_map_->participantForId(target);
                    if (!descriptor ||
                        !descriptor->world_rank_known)
                    {
                        throw std::runtime_error(
                            "Participant rank batch target has no authenticated MPI owner");
                    }
                    if (descriptor->domain_name ==
                            config_.placement_plan->continuation_domain &&
                        descriptor->world_rank ==
                            source_world_rank)
                    {
                        continue;
                    }
                    if (descriptor->world_rank != local_world_rank)
                        continue;
                    if (std::find(
                            local_participant_ids.begin(),
                            local_participant_ids.end(),
                            target) == local_participant_ids.end())
                    {
                        throw std::logic_error(
                            "Participant rank batch addresses an endpoint absent from the rank-local topology");
                    }
                    participants.push_back(target);
                }
                if (participants.empty())
                    continue;
                std::sort(participants.begin(), participants.end());

                const int routed_domain_ordinal =
                    domain_ordinal(tier.domain);
                auto transport = transport_for_group(
                    static_cast<int>(tier_index),
                    routed_domain_ordinal,
                    participants);

                std::vector<
                    std::shared_ptr<MoEOverlayCollectiveWorkspace>>
                    group_workspaces;
                std::vector<std::shared_ptr<MoEOverlaySparseRows>>
                    dispatch_inbound;
                std::vector<std::shared_ptr<MoEOverlayReturnRows>>
                    local_outputs;
                group_workspaces.reserve(participants.size());
                dispatch_inbound.reserve(participants.size());
                local_outputs.reserve(participants.size());
                for (const int participant : participants)
                {
                    auto workspace =
                        workspace_for_participant(participant);
                    group_workspaces.push_back(workspace);
                    dispatch_inbound.push_back(
                        std::make_shared<MoEOverlaySparseRows>(
                            transport->hasSharedRowStorage()
                                ? transport->sharedDispatchRows(participant)
                                : workspace->dispatchReceive(
                                      layer,
                                      static_cast<int>(tier_index))));
                    local_outputs.push_back(
                        std::make_shared<MoEOverlayReturnRows>(
                            transport->hasSharedRowStorage()
                                ? transport->sharedReturnRows(participant)
                                : workspace->localExpertOutput(
                                      layer,
                                      static_cast<int>(tier_index))));
                }

                const auto make_rank_batch_key =
                    [&](MoEOverlayCollectiveDirection direction)
                {
                    if (spec.role ==
                        ParticipantGraphFamilyRole::MTPDraft)
                    {
                        return makeMTPMoEOverlayRankBatchKey(
                            1,
                            0,
                            spec.mtp_graph_depth,
                            layer,
                            static_cast<int>(tier_index),
                            routed_domain_ordinal,
                            source_world_rank,
                            local_world_rank,
                            direction);
                    }
                    return makeMoEOverlayRankBatchKey(
                        1,
                        0,
                        ExpertHistogramSource::DecodeToken,
                        layer,
                        static_cast<int>(tier_index),
                        routed_domain_ordinal,
                        source_world_rank,
                        local_world_rank,
                        direction);
                };

                MoERankBatchDispatchStage::Params dispatch_params;
                dispatch_params.device_id = DeviceId::cpu();
                dispatch_params.endpoint_role =
                    MoERankBatchEndpointRole::RemoteTarget;
                dispatch_params.transport = transport;
                dispatch_params.key = make_rank_batch_key(
                    MoEOverlayCollectiveDirection::Dispatch);
                dispatch_params.source_participant =
                    continuation_root;
                dispatch_params.participant_ids = participants;
                dispatch_params.inbound_rows =
                    dispatch_inbound;
                dispatch_params.seq_len = row_capacity;
                dispatch_params.top_k = top_k;
                dispatch_params.d_model = d_model;
                dispatch_params.tier_index =
                    static_cast<int>(tier_index);

                const std::string stem =
                    "participant_layer" +
                    std::to_string(layer) +
                    "_tier" + std::to_string(tier_index) +
                    "_rank" + std::to_string(local_world_rank);
                const std::string dispatch_name =
                    stem + "_batch_dispatch";
                result->graph->addNode(
                    dispatch_name,
                    ComputeStageFactory::createMoERankBatchDispatch(
                        dispatch_params),
                    DeviceId::cpu());
                if (!previous_protocol_node.empty())
                {
                    result->graph->addDependency(
                        dispatch_name,
                        previous_protocol_node);
                }

                std::vector<std::string> local_names;
                local_names.reserve(participants.size());
                struct DeferredLocalCompletion
                {
                    MoELocalExpertStage *producer = nullptr;
                    DeviceId device = DeviceId::invalid();
                    int participant = -1;
                };
                std::vector<DeferredLocalCompletion>
                    deferred_local_completions;
                deferred_local_completions.reserve(participants.size());
                for (size_t participant_index = 0;
                     participant_index < participants.size();
                     ++participant_index)
                {
                    const int target =
                        participants[participant_index];
                    const auto *target_descriptor =
                        owner_map_->participantForId(target);
                    const auto local_participant_it =
                        std::find_if(
                            local_participants_.begin(),
                            local_participants_.end(),
                            [target](const auto *participant)
                            {
                                return participant &&
                                       participant->participant_id ==
                                           target;
                            });
                    if (!target_descriptor ||
                        local_participant_it ==
                            local_participants_.end() ||
                        (*local_participant_it)->device !=
                            target_descriptor->device)
                    {
                        throw std::runtime_error(
                            "Participant rank batch local endpoint topology changed during graph construction");
                    }

                    const DeviceId target_device =
                        target_descriptor->device;
                    MoELocalExpertStage::Params local_params;
                    local_params.device_id = target_device;
                    local_params.input_rows_lifetime =
                        dispatch_inbound[participant_index];
                    local_params.output_rows_lifetime =
                        local_outputs[participant_index];
                    local_params.workspace_lifetime =
                        group_workspaces[participant_index];
                    local_params.num_experts = num_experts;
                    local_params.top_k = top_k;
                    local_params.d_model = d_model;
                    local_params.expert_intermediate =
                        expert_intermediate;
                    local_params.layer_idx = layer;
                    local_params.expert_mask =
                        owner_map_->expertMaskForParticipant(
                            layer, target, num_experts);
                    local_params.prepared_store =
                        prepared_store_.get();
                    local_params.expert_registry = &registry;
                    local_params.runtime_participant_index =
                        target;
                    local_params.graph_row_capacity =
                        static_cast<size_t>(row_capacity);
                    local_params.completion_policy =
                        target_device.is_gpu()
                            ? MoELocalExpertStage::CompletionPolicy::
                                  DeferredExplicitStage
                            : MoELocalExpertStage::CompletionPolicy::Inline;
                    local_params.worker_gpu_context =
                        participantWorkerContext(target_device);
                    local_params.serial_compact_buffer_arena =
                        serialCompactBufferArenaForParticipant(
                            **local_participant_it);
                    local_params.expert_weight_resolution_policy =
                        MoELocalExpertStage::
                            ExpertWeightResolutionPolicy::
                                RegistryOnly;

                    (void)registry.populateExpertEnginesForParticipant(
                        tier.domain,
                        target_device,
                        target_descriptor->world_rank,
                        target_descriptor->domain_participant_index,
                        layer,
                        num_experts,
                        local_params.prepared_gate_gemm,
                        local_params.prepared_up_gemm,
                        local_params.prepared_down_gemm);
                    if (!MoELocalExpertStage::
                             prepareExpertGemmEngines(local_params))
                    {
                        throw std::runtime_error(
                            "Participant rank batch has incomplete prepared expert engines for layer " +
                            std::to_string(layer) + " participant " +
                            std::to_string(target));
                    }

                    auto endpoint =
                        config_.participant_residency->endpoint(
                            target);
                    if (!endpoint)
                    {
                        throw std::runtime_error(
                            "Participant rank batch is missing prepared-bank endpoint p" +
                            std::to_string(target));
                    }
                    std::vector<MoEOverlayPreparedExpertTriplet>
                        prepared_triplets;
                    std::string residency_error;
                    if (!resolveMoEOverlayPreparedExpertTriplets(
                            registry,
                            *target_descriptor,
                            layer,
                            num_experts,
                            local_params.expert_mask,
                            prepared_triplets,
                            &residency_error) ||
                        !config_.participant_residency
                             ->registerInitialLayer(
                                 target,
                                 layer,
                                 local_params.expert_mask,
                                 prepared_triplets,
                                 &residency_error))
                    {
                        throw std::runtime_error(
                            "Participant rank batch could not install initial prepared residency bank for layer " +
                            std::to_string(layer) + " participant " +
                            std::to_string(target) + ": " +
                            residency_error);
                    }
                    local_params.overlay_participant_residency =
                        std::move(endpoint);

                    auto local_stage =
                        ComputeStageFactory::createMoELocalExpert(
                            local_params);
                    auto *const local_stage_ptr =
                        dynamic_cast<MoELocalExpertStage *>(
                            local_stage.get());
                    if (!local_stage_ptr)
                    {
                        throw std::runtime_error(
                            "Participant rank batch factory returned a non-local-expert stage");
                    }
                    if (target_device.is_gpu())
                    {
                        local_stage->setGPUStream(
                            participantWorkerStream(
                                target_device));
                        deferred_local_completions.push_back(
                            DeferredLocalCompletion{
                                .producer = local_stage_ptr,
                                .device = target_device,
                                .participant = target,
                            });
                    }
                    const std::string local_name =
                        stem + "_p" + std::to_string(target) +
                        "_local_expert";
                    result->graph->addNode(
                        local_name,
                        std::move(local_stage),
                        target_device);
                    result->graph->addDependency(
                        local_name,
                        dispatch_name);
                    local_names.push_back(local_name);
                }

                std::vector<std::string> completion_names;
                completion_names.reserve(
                    deferred_local_completions.size());
                for (const auto &deferred :
                     deferred_local_completions)
                {
                    MoELocalExpertCompletionStage::Params
                        completion_params;
                    completion_params.device_id = deferred.device;
                    completion_params.producer = deferred.producer;
                    auto completion_stage =
                        ComputeStageFactory::
                            createMoELocalExpertCompletion(
                                completion_params);
                    completion_stage->setGPUStream(
                        participantWorkerStream(
                            deferred.device));

                    const std::string completion_name =
                        stem + "_p" +
                        std::to_string(deferred.participant) +
                        "_local_expert_completion";
                    result->graph->addNode(
                        completion_name,
                        std::move(completion_stage),
                        deferred.device);

                    /*
                     * Every participant submission must be queued before the
                     * first host-visible output wait. This all-submit barrier
                     * is what turns distinct per-device streams into actual
                     * overlap rather than a series of submit/wait pairs.
                     */
                    for (const auto &local_name : local_names)
                    {
                        result->graph->addDependency(
                            completion_name,
                            local_name);
                    }
                    completion_names.push_back(completion_name);
                }

                std::vector<
                    std::shared_ptr<const MoEOverlayReturnRows>>
                    outbound_rows;
                outbound_rows.reserve(local_outputs.size());
                for (const auto &rows : local_outputs)
                    outbound_rows.push_back(rows);

                MoERankBatchReturnReduceStage::Params return_params;
                return_params.device_id = DeviceId::cpu();
                return_params.endpoint_role =
                    MoERankBatchEndpointRole::RemoteTarget;
                return_params.transport = transport;
                return_params.key = make_rank_batch_key(
                    MoEOverlayCollectiveDirection::ReturnReduce);
                return_params.continuation_participant =
                    continuation_root;
                return_params.participant_ids = participants;
                return_params.outbound_rows =
                    std::move(outbound_rows);
                return_params.seq_len = row_capacity;
                return_params.d_model = d_model;

                const std::string return_name =
                    stem + "_batch_return";
                result->graph->addNode(
                    return_name,
                    ComputeStageFactory::
                        createMoERankBatchReturnReduce(
                            return_params),
                    DeviceId::cpu());
                const auto &return_dependencies =
                    completion_names.empty()
                        ? local_names
                        : completion_names;
                for (const auto &dependency : return_dependencies)
                {
                    result->graph->addDependency(
                        return_name,
                        dependency);
                }
                previous_protocol_node = return_name;
            }
        }

        if (result->graph->size() == 0)
        {
            throw std::runtime_error(
                "Participant graph contains no sparse protocol stages");
        }
        result->workspace_allocator =
            serial_graph_family_workspace_allocator_;
        WorkspaceSizingHints hints;
        hints.max_seq_len = row_capacity;
        hints.serial_family_max_rows = row_capacity;
        hints.d_model = d_model;
        hints.batch_size = 1;
        hints.graph_family_policy =
            WorkspaceGraphFamilyPolicy::
                SerialDeviceFamilyExactParticipant;
        if (!result->workspace_allocator->allocateForGraph(
                *result->graph, hints))
        {
            throw std::runtime_error(
                "Failed to allocate participant graph workspace");
        }

        if (std::any_of(
                local_participants_.begin(),
                local_participants_.end(),
                [](const auto *participant)
                {
                    return participant && participant->device.is_gpu();
                }))
        {
            for (const auto &node_name :
                 result->graph->getExecutionOrder())
            {
                auto *node = result->graph->getNode(node_name);
                if (!node || !node->stage)
                    continue;
                auto *local_stage =
                    dynamic_cast<MoELocalExpertStage *>(node->stage.get());
                if (!local_stage)
                    continue;
                if (!local_stage->preparePersistentBuffers())
                {
                    throw std::runtime_error(
                        "Failed to prepare participant GPU sparse buffers");
                }
            }
        }

        std::string retained_plan_error;
        if (!executor_.prepareRetainedMultiDeviceExecutionPlan(
                *result->graph,
                execution_contexts_,
                result->execution_plan,
                &retained_plan_error))
        {
            throw std::runtime_error(
                "Failed to seal participant multi-device execution plan: " +
                retained_plan_error);
        }

        PerfStatsCollector::addCounter(
            "moe_overlay_participant_graph",
            "materialized_graphs",
            1.0,
            "model_setup",
            participantDeviceList(local_participants_),
            {
                {"participants", participantIdList(local_participants_)},
                {"row_capacity", std::to_string(row_capacity)},
                {"live_rows_source", "sparse_packet"},
                {"allocation_policy", "setup_only"},
                {"immutable", "true"},
                {"stages", std::to_string(result->graph->size())},
                {"dependency_waves",
                 std::to_string(result->execution_plan.waves.size())},
                {"parallel_gpu_waves",
                 std::to_string(
                     result->execution_plan.concurrent_gpu_wave_count)},
                {"max_wave_width",
                 std::to_string(result->execution_plan.max_wave_width)},
                {"graph_role",
                 spec.role == ParticipantGraphFamilyRole::Main
                     ? "main"
                     : "mtp_draft"},
                {"mtp_graph_depth",
                 std::to_string(spec.mtp_graph_depth)},
                {"source_layer_count",
                 std::to_string(spec.source_layers.size())},
            });
        return result;
    }

    /**
     * @brief Execute one exact live-row sparse protocol transaction.
     *
     * Tokens are intentionally not read: the continuation rank alone owns
     * token embedding and publishes the routed hidden rows through the first
     * sparse dispatch boundary.  The supplied count is still authoritative for
     * graph-cache identity and for advancing this participant's request cursor.
     */
    bool MoEOverlayParticipantGraphRunner::forward(
        const int *tokens,
        int seq_len)
    {
        /*
         * The unsegmented public runner API has no phase parameter. Preserve
         * its established serial-row attribution here; the explicit chunked
         * prefill API below passes Prefill for every member of a request,
         * including its one-row tail.
         */
        return executeAtLogicalStep(
            tokens,
            seq_len,
            position_,
            seq_len == 1
                ? SparseTransactionPhase::Decode
                : SparseTransactionPhase::Prefill);
    }

    bool MoEOverlayParticipantGraphRunner::forwardPrefill(
        const int *tokens,
        int seq_len)
    {
        return executeAtLogicalStep(
            tokens,
            seq_len,
            position_,
            SparseTransactionPhase::Prefill);
    }

    /**
     * @brief Execute one expert-only graph using an explicit protocol cursor.
     *
     * The caller supplies the continuation runner's logical start row.  We
     * require it to match the local cursor before executing, which catches a
     * missing prefix/reset publication before it can become a sparse MPI
     * mismatch. The already-built fixed-capacity graph remains independent of
     * the continuation graph's capture history because its manual boundaries
     * receive the shared identity directly.
     */
    bool MoEOverlayParticipantGraphRunner::executeAtLogicalStep(
        const int *tokens,
        int seq_len,
        int logical_step,
        SparseTransactionPhase phase,
        int physical_rows)
    {
        (void)tokens;
        if (physical_rows < 0)
            physical_rows = seq_len;
        const ExpertHistogramSource histogram_source =
            phase == SparseTransactionPhase::Decode
                ? ExpertHistogramSource::DecodeToken
                : (phase == SparseTransactionPhase::Prefill
                       ? ExpertHistogramSource::PrefillChunk
                       : ExpertHistogramSource::SyntheticTest);
        MoEOverlayInferenceInterferenceScope interference_scope(
            interference_probe_.get(),
            makeMoEOverlayInferenceWorkloadIdentity(
                histogram_source,
                seq_len,
                physical_rows,
                /*transaction_count=*/1,
                /*speculative_depth=*/0));
        try
        {
            if (seq_len <= 0 || physical_rows < seq_len || logical_step < 0 ||
                logical_step != position_)
            {
                interference_scope.discard();
                LOG_ERROR(
                    "[MoEOverlayParticipantGraphRunner] Invalid sparse "
                    "collective request cursor logical_step="
                    << logical_step << " local_position=" << position_
                    << " rows=" << seq_len);
                return false;
            }
            if (overlay_collective_request_generation_ == 0)
            {
                interference_scope.discard();
                LOG_ERROR(
                    "[MoEOverlayParticipantGraphRunner] Refusing sparse "
                    "graph execution before the orchestration layer published "
                    "a request generation");
                return false;
            }
            auto &cached = graphForRows(seq_len);
            if (!stampMoEOverlayCollectiveRuntime(
                    cached,
                    overlay_collective_request_generation_,
                    static_cast<uint64_t>(logical_step),
                    phase))
            {
                interference_scope.discard();
                return false;
            }
            std::string execution_error;
            const bool ok = executor_.executeRetainedMultiDevice(
                cached.execution_plan, &execution_error);
            if (!ok && !execution_error.empty())
            {
                LOG_ERROR(
                    "[MoEOverlayParticipantGraphRunner] Retained participant schedule failed: "
                    << execution_error);
            }
            if (!ok)
                interference_scope.discard();
            if (ok)
            {
                position_ += seq_len;
                PerfStatsCollector::addCounter(
                    "moe_overlay_participant_graph",
                    "fixed_capacity_graph_reuses",
                    1.0,
                    phase == SparseTransactionPhase::Decode
                        ? "decode"
                        : "prefill",
                    participantDeviceList(local_participants_),
                    {{"participants", participantIdList(local_participants_)},
                     {"logical_rows", std::to_string(seq_len)},
                     {"row_capacity", std::to_string(
                          config_.max_graph_activation_rows)},
                     {"parallel_gpu_waves",
                      std::to_string(
                          cached.execution_plan.concurrent_gpu_wave_count)},
                     {"max_wave_width",
                      std::to_string(
                          cached.execution_plan.max_wave_width)},
                     {"allocation_policy", "setup_only"}});
                const char *const evidence_phase =
                    phase == SparseTransactionPhase::Decode
                        ? "decode"
                        : "prefill";
                /*
                 * Fold identity into a bounded sequence witness; never retain
                 * request generations or logical steps as aggregation keys.
                 */
                PerfStatsCollector::recordOrderedSequenceStep(
                    "forward_graph",
                    "moe_overlay_collective_transaction_sequence",
                    {overlay_collective_request_generation_,
                     static_cast<uint64_t>(logical_step),
                     static_cast<uint64_t>(seq_len),
                     static_cast<uint64_t>(seq_len)},
                    evidence_phase,
                    participantDeviceList(local_participants_),
                    {{"role", "expert_participant_graph"},
                     {"identity_source", "orchestration_request_and_chunk"},
                     {"logical_step_semantics", "monotonic_transaction"}});
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "moe_overlay_collective_transaction",
                    1.0,
                    evidence_phase,
                    participantDeviceList(local_participants_),
                    {{"role", "expert_participant_graph"},
                     {"identity_source", "orchestration_request_and_chunk"},
                     {"logical_step_semantics", "monotonic_transaction"},
                     {"logical_rows", std::to_string(seq_len)}});
            }
            PerfStatsCollector::addCounter(
                "moe_overlay_participant_graph",
                ok ? "successful_steps" : "failed_steps",
                1.0,
                seq_len == 1 ? "decode" : "prefill",
                participantDeviceList(local_participants_),
                    {
                        {"participants", participantIdList(local_participants_)},
                        {"logical_rows", std::to_string(seq_len)},
                    });
            return ok;
        }
        catch (const std::exception &error)
        {
            interference_scope.discard();
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Forward failed on rank "
                << config_.mpi_context->rank() << ": " << error.what());
            return false;
        }
    }

    /**
     * @brief Stamp all sparse manual boundaries with this rank's shared operation identity.
     *
     * The root and remote runners derive @p logical_step from the same
     * request cursor and frozen chunk schedule. The stage objects themselves
     * may have unrelated capture/replay histories, so this loop is the one
     * place that connects their wire packets to the root-authoritative
     * request generation.
     */
    bool MoEOverlayParticipantGraphRunner::stampMoEOverlayCollectiveRuntime(
        CachedParticipantGraph &cached,
        uint64_t generation_id,
        uint64_t logical_step,
        SparseTransactionPhase phase,
        int mtp_graph_depth,
        uint64_t placement_epoch)
    {
        if (!cached.graph || generation_id == 0)
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Invalid sparse "
                "collective runtime stamp");
            return false;
        }

        using Semantics = IComputeStage::MoEOverlayCollectiveRuntimeParams::
            ExecutionSemantics;
        Semantics execution_semantics = Semantics::Unspecified;
        switch (phase)
        {
        case SparseTransactionPhase::Prefill:
            execution_semantics = Semantics::Prefill;
            break;
        case SparseTransactionPhase::Decode:
            execution_semantics = Semantics::Decode;
            break;
        case SparseTransactionPhase::MTPDraft:
            execution_semantics = Semantics::MTPDraft;
            break;
        case SparseTransactionPhase::GroupedVerifier:
            execution_semantics = Semantics::GroupedVerifier;
            break;
        }
        const bool main_semantics =
            phase == SparseTransactionPhase::Prefill ||
            phase == SparseTransactionPhase::Decode;
        if ((main_semantics && mtp_graph_depth != -1) ||
            (!main_semantics && mtp_graph_depth < 0) ||
            (phase == SparseTransactionPhase::MTPDraft &&
             (cached.role != ParticipantGraphFamilyRole::MTPDraft ||
              cached.mtp_graph_depth != mtp_graph_depth)) ||
            (phase != SparseTransactionPhase::MTPDraft &&
             cached.role != ParticipantGraphFamilyRole::Main))
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Sparse graph role and MTP namespace depth disagree");
            return false;
        }

        const IComputeStage::MoEOverlayCollectiveRuntimeParams params{
            .generation_id = generation_id,
            .step_id = logical_step,
            .execution_semantics = execution_semantics,
            .mtp_depth = mtp_graph_depth,
            .placement_epoch = placement_epoch,
        };
        size_t stamped_stages = 0;
        for (const auto &node_name : cached.graph->getExecutionOrder())
        {
            auto *node = cached.graph->getNode(node_name);
            if (!node || !node->stage ||
                !node->stage->hasMoEOverlayCollectiveRuntimeParams())
            {
                continue;
            }
            node->stage->updateMoEOverlayCollectiveRuntimeParams(params);
            ++stamped_stages;
        }
        if (stamped_stages == 0)
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Graph-native overlay "
                "participant graph has no sparse stages to stamp");
            return false;
        }
        return true;
    }

    /**
     * @brief Accept a root-published request generation for future sparse operations.
     */
    bool MoEOverlayParticipantGraphRunner::setMoEOverlayCollectiveRequestGeneration(
        uint64_t generation_id)
    {
        if (generation_id == 0)
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Sparse collective "
                "request generation must be non-zero");
            return false;
        }
        overlay_collective_request_generation_ = generation_id;
        return true;
    }

    bool MoEOverlayParticipantGraphRunner::
        setMoEOverlayInferenceInterferenceProbe(
            std::shared_ptr<MoEOverlayInferenceInterferenceProbe> probe)
    {
        if (!probe || (interference_probe_ && interference_probe_ != probe))
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Requires one stable "
                "non-null ExpertOverlay interference probe");
            return false;
        }
        interference_probe_ = std::move(probe);
        return true;
    }

    /**
     * @brief Submit all mapped follower parents and retire their shared lease.
     *
     * Ticket reception is the only host-visible scheduler boundary. Every GPU
     * lane is device-owned after admission. A configured CPU lane is then
     * serviced directly from its shared pages at the explicit heterogeneous
     * boundary, while the already-submitted GPU parents continue concurrently.
     * The CPU loop never observes or forwards a GPU sibling's packet.
     */
    bool MoEOverlayParticipantGraphRunner::
        executeMappedFollowerTransaction(
            CachedParticipantGraph &cached,
            const MoEOverlayInferenceTransactionTicket &ticket,
            std::string *error)
    {
        using Clock = std::chrono::steady_clock;
        const bool collect_timeline =
            PerfStatsCollector::isDomainEnabled(
                "moe_overlay_participant_graph");
        std::uint64_t cpu_dispatch_wait_ns = 0u;
        std::uint64_t cpu_packet_prepare_ns = 0u;
        std::uint64_t cpu_expert_compute_ns = 0u;
        std::uint64_t cpu_return_publication_ns = 0u;
        std::uint64_t endpoint_completion_wait_ns = 0u;
        /**
         * @brief One deferred CPU-layer timing observation.
         *
         * Samples are accumulated in the ticket-service hot path and published
         * only after the complete heterogeneous transaction has retired. This
         * keeps PerfStats locking/string work out of the GPU/CPU rendezvous
         * whose latency the diagnostic is intended to measure.
         */
        struct CPULayerTimelineSample
        {
            int participant_id = -1;
            int model_layer_index = -1;
            std::uint32_t stage_ordinal = 0u;
            std::size_t live_rows = 0u;
            std::size_t live_entries = 0u;
            std::uint64_t dispatch_wait_ns = 0u;
            std::uint64_t packet_prepare_ns = 0u;
            std::uint64_t expert_compute_ns = 0u;
            std::uint64_t return_publication_ns = 0u;
        };
        std::vector<CPULayerTimelineSample> cpu_layer_timeline;
        const auto elapsed_nanoseconds = [](
                                             Clock::time_point begin,
                                             Clock::time_point end)
            -> std::uint64_t
        {
            const auto elapsed =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    end - begin)
                    .count();
            return elapsed > 0
                       ? static_cast<std::uint64_t>(elapsed)
                       : 0u;
        };
        const auto accumulate_elapsed = [&elapsed_nanoseconds](
                                            std::uint64_t &total,
                                            Clock::time_point begin,
                                            Clock::time_point end)
            -> std::uint64_t
        {
            const std::uint64_t elapsed = elapsed_nanoseconds(begin, end);
            if (elapsed != 0u &&
                total <= std::numeric_limits<std::uint64_t>::max() -
                             elapsed)
            {
                total += elapsed;
            }
            return elapsed;
        };
        if (error)
            error->clear();
        const auto reject = [error](std::string message)
        {
            if (error)
                *error = std::move(message);
            return false;
        };
        if (serving_graph_family_lifecycle_ !=
            ServingGraphFamilyLifecycle::Sealed)
        {
            return reject(
                "Mapped ExpertOverlay follower received a ticket before its physical-fabric graph branches were sealed");
        }
        const uint64_t physical_rows_u64 =
            static_cast<uint64_t>(ticket.request_count) *
            static_cast<uint64_t>(ticket.physical_rows_per_request);
        if (physical_rows_u64 == 0u ||
            physical_rows_u64 >
                static_cast<uint64_t>(std::numeric_limits<int>::max()))
        {
            return reject(
                "Mapped ExpertOverlay follower ticket has an invalid physical row geometry");
        }
        auto *const shape = cached.mappedShapeForRows(
            static_cast<int>(physical_rows_u64));
        if (!shape ||
            shape->endpoints.size() + cached.mapped_cpu_endpoints.size() !=
                cached.mapped_lane_authorities.size())
        {
            return reject(
                "Mapped ExpertOverlay follower has no setup-owned executable for the ticket row geometry");
        }
        if (collect_timeline)
        {
            std::size_t layer_capacity = 0u;
            for (const auto &endpoint : cached.mapped_cpu_endpoints)
            {
                if (endpoint)
                    layer_capacity += endpoint->layers.size();
            }
            cpu_layer_timeline.reserve(layer_capacity);
        }

        LOG_DEBUG(
            "[MoEOverlayParticipantGraphRunner] Admitted mapped follower "
            "transaction role="
            << static_cast<std::uint32_t>(ticket.graph_role)
            << " command=" << ticket.command_id
            << " ordinal=" << ticket.transaction_ordinal
            << " logical_step=" << ticket.logical_step_id
            << " physical_rows=" << physical_rows_u64
            << " gpu_endpoints=" << shape->endpoints.size()
            << " cpu_endpoints=" << cached.mapped_cpu_endpoints.size());

        struct ArmedLane
        {
            std::shared_ptr<CachedParticipantGraph::MappedLaneAuthority>
                authority;
            MoEOverlayActivationEpochIdentity identity;
        };
        std::vector<ArmedLane> armed;
        armed.reserve(cached.mapped_lane_authorities.size());

        const auto timeout_eligibility =
            MoEOverlayActivationRendezvousDeadline::begin(
                MoEOverlayActivationRendezvousKind::EndpointCompletion,
                std::chrono::milliseconds(
                    collective_timeout_policy::
                        kDefaultCollectiveTimeoutMs));
        const auto timeout_not_before_ns =
            timeout_eligibility.deadlineNanoseconds();
        if (!timeout_not_before_ns)
        {
            return reject(
                "Mapped ExpertOverlay follower could not derive a watchdog timeout lower bound");
        }

        const auto abort_armed = [&]() noexcept
        {
            for (auto &lane : armed)
            {
                if (!lane.authority || !lane.authority->protocol)
                    continue;
                std::string ignored;
                (void)lane.authority->protocol->abort(
                    MoEOverlayActivationEndpoint::Follower,
                    lane.identity,
                    MoEOverlayActivationStatusCode::ExplicitAbort,
                    &ignored);
            }
        };

        for (const auto &authority : cached.mapped_lane_authorities)
        {
            if (!authority || !authority->protocol ||
                authority->next_epoch_generation == 0u ||
                authority->next_epoch_generation >
                    kMoEOverlayActivationMaxGeneration)
            {
                abort_armed();
                return reject(
                    "Mapped ExpertOverlay follower exhausted or lost a lane generation authority");
            }
            std::string arm_error;
            const uint64_t generation =
                authority->next_epoch_generation;
            auto identity = authority->protocol->arm(
                ticket,
                generation,
                *timeout_not_before_ns,
                &arm_error);
            if (!identity)
            {
                abort_armed();
                return reject(
                    "Mapped ExpertOverlay follower could not arm participant " +
                        std::to_string(authority->participant_id) + ": " +
                        arm_error);
            }
            ++authority->next_epoch_generation;
            armed.push_back(ArmedLane{
                .authority = authority,
                .identity = *identity,
            });
        }

        const auto armed_lane_for = [&](const auto &authority)
            -> const ArmedLane *
        {
            const auto found = std::find_if(
                armed.begin(),
                armed.end(),
                [&](const ArmedLane &lane)
                {
                    return lane.authority == authority;
                });
            return found == armed.end() ? nullptr : &*found;
        };

        std::size_t captured_gpu_transactions = 0u;
        std::size_t replayed_gpu_transactions = 0u;
        struct SubmittedInferenceBoundary
        {
            ParticipantGpuRuntime *runtime = nullptr;
            std::uint64_t generation = 0u;
            bool graph_submitted = false;
        };
        struct PreparedGpuSubmission
        {
            CachedParticipantGraph::MappedGPUFollowerEndpoint *endpoint =
                nullptr;
            IDeviceContext *context = nullptr;
            IWorkerGPUContext *worker = nullptr;
            bool had_native_transaction = false;
        };
        struct SubmittedGpuTerminal
        {
            CachedParticipantGraph::MappedGPUFollowerEndpoint *endpoint =
                nullptr;
            DeviceGraphExecutor::GraphSegmentCache::
                CaptureStreamTerminalTicket ticket;
        };
        std::vector<SubmittedInferenceBoundary> submitted_boundaries;
        submitted_boundaries.reserve(shape->endpoints.size());
        std::vector<PreparedGpuSubmission> prepared_gpu_submissions;
        prepared_gpu_submissions.reserve(shape->endpoints.size());
        std::vector<SubmittedGpuTerminal> submitted_gpu_terminals;
        submitted_gpu_terminals.reserve(shape->endpoints.size());
        std::uint64_t mapped_dispatch_bytes = 0u;
        std::uint64_t mapped_return_bytes = 0u;
        std::uint64_t mapped_dispatch_rows = 0u;
        std::uint64_t mapped_return_rows = 0u;
        std::uint64_t mapped_dispatch_entries = 0u;
        const char *const transaction_phase =
            ticket.graph_role ==
                    MoEOverlayInferenceGraphRole::MainPrefill
                ? "prefill"
                : "decode";

        /*
         * Resolve every immutable launch dependency before publishing any
         * inference admission.  This keeps setup errors exactly reversible:
         * after the second pass begins, every receipt already protects a real
         * executable that is ready to enter its backend worker queue.
         */
        for (const auto &endpoint : shape->endpoints)
        {
            if (!endpoint || !endpoint->graph || !endpoint->executor ||
                !endpoint->lane_authority ||
                !endpoint->cache.capture_stream)
            {
                abort_armed();
                return reject(
                    "Mapped ExpertOverlay follower endpoint lost its retained graph identity");
            }
            const auto context_it = execution_contexts_.find(endpoint->device);
            IWorkerGPUContext *const worker =
                participantWorkerContext(endpoint->device);
            if (context_it == execution_contexts_.end() ||
                !context_it->second || !worker ||
                endpoint->graph->nativeCaptureEnvelope() !=
                    GraphNativeCaptureEnvelope::
                        DeviceOwnedTimelineTransaction)
            {
                abort_armed();
                return reject(
                    "Mapped ExpertOverlay follower endpoint is missing its device context or native timeline envelope");
            }

            prepared_gpu_submissions.push_back({
                .endpoint = endpoint.get(),
                .context = context_it->second,
                .worker = worker,
                .had_native_transaction =
                    endpoint->cache.retained_full_graph_replay.valid(),
            });
        }

        const auto cancel_unsubmitted_boundaries = [&]() noexcept
        {
            bool cancelled_every_boundary = true;
            for (auto boundary = submitted_boundaries.rbegin();
                 boundary != submitted_boundaries.rend();
                 ++boundary)
            {
                if (boundary->graph_submitted)
                    continue;
                cancelled_every_boundary =
                    boundary->runtime &&
                    boundary->runtime->inference_boundary_receipt
                        .cancelUnsubmitted(boundary->generation) &&
                    cancelled_every_boundary;
            }
            return cancelled_every_boundary;
        };

        /*
         * Publish every local follower admission before launching the first
         * graph.  The controller's topology preflight can therefore observe
         * either the wholly quiescent prior transaction or an in-flight member
         * of this transaction; it can never certify a graph that has already
         * entered a backend queue but is still absent from its receipt.
         */
        if (!prepared_gpu_submissions.empty())
        {
            for (const auto &prepared : prepared_gpu_submissions)
            {
                auto &runtime = participantGpuRuntimeForParticipant(
                    prepared.endpoint->participant_id);
                if (runtime.device != prepared.endpoint->device)
                {
                    (void)cancel_unsubmitted_boundaries();
                    abort_armed();
                    return reject(
                        "Mapped ExpertOverlay follower prepared a cache on the wrong controller-runtime device");
                }
                const std::uint64_t boundary_generation =
                    runtime.inference_boundary_receipt.beginSubmission();
                if (boundary_generation == 0u)
                {
                    (void)cancel_unsubmitted_boundaries();
                    abort_armed();
                    return reject(
                        "Mapped ExpertOverlay follower exhausted its inference-boundary generation namespace for participant " +
                        std::to_string(prepared.endpoint->participant_id));
                }
                submitted_boundaries.push_back({
                    .runtime = &runtime,
                    .generation = boundary_generation,
                    .graph_submitted = false,
                });
            }
        }

        /* Queue every complete native GPU transaction before CPU work begins. */
        for (std::size_t submission_index = 0u;
             submission_index < prepared_gpu_submissions.size();
             ++submission_index)
        {
            const auto &prepared = prepared_gpu_submissions[submission_index];
            auto *const endpoint = prepared.endpoint;
            /* Transfer progress is a maintenance-owned retained replay. A
             * follower inference graph must never join it or share its stream. */
            const GraphCaptureAuxiliaryBranchFactory branch_factory{};
            bool submitted = false;
            DeviceGraphExecutor::GraphSegmentCache::
                CaptureStreamTerminalTicket terminal_ticket;
            try
            {
                /* The worker queue is the sole submission authority for this
                 * device. Recording the dedicated terminal in the same closure
                 * makes it impossible for controller maintenance or another
                 * graph to interleave stream work between launch and receipt. */
                prepared.worker->submitAndWait(
                    [&]()
                    {
                        submitted = endpoint->executor
                                        ->executeWithCachedGraphReplay(
                                            *endpoint->graph,
                                            prepared.context,
                                            endpoint->cache,
                                            endpoint->cache.capture_stream,
                                            prepared.worker,
                                            /*collective_nodes=*/nullptr,
                                            /*collectives_graph_capturable=*/false,
                                            /*force_recapture=*/false,
                                            /*defer_final_sync=*/true,
                                            {},
                                            DeviceGraphExecutor::
                                                GraphReplayPlanPolicy::
                                                    RequireFullGraph,
                                            {},
                                            {},
                                            {},
                                            DeviceGraphExecutor::
                                                GraphInitialSubmissionPolicy::
                                                    CaptureInstantiateAndLaunch,
                                            branch_factory);
                        if (submitted)
                        {
                            terminal_ticket = endpoint->cache
                                                  .publishCaptureStreamTerminal();
                        }
                    });
            }
            catch (const std::exception &exception)
            {
                if (!submitted && !cancel_unsubmitted_boundaries())
                {
                    abort_armed();
                    return reject(
                        "Mapped ExpertOverlay follower could not roll back an unsubmitted inference boundary after graph submission threw");
                }
                abort_armed();
                return reject(
                    "Mapped ExpertOverlay follower graph submission threw for participant " +
                    std::to_string(endpoint->participant_id) + ": " +
                    exception.what());
            }
            if (!submitted || !terminal_ticket.valid())
            {
                if (!submitted && !cancel_unsubmitted_boundaries())
                {
                    abort_armed();
                    return reject(
                        "Mapped ExpertOverlay follower could not roll back an unsubmitted inference boundary after native timeline rejection");
                }
                abort_armed();
                return reject(
                    "Mapped ExpertOverlay follower native timeline submission failed for participant " +
                        std::to_string(endpoint->participant_id));
            }
            if (!endpoint->cache.retained_full_graph_replay.valid())
            {
                abort_armed();
                return reject(
                    "Mapped ExpertOverlay follower submission returned without a sealed native timeline replay identity for participant " +
                        std::to_string(endpoint->participant_id));
            }
            submitted_gpu_terminals.push_back({
                .endpoint = endpoint,
                .ticket = terminal_ticket,
            });
            submitted_boundaries[submission_index].graph_submitted = true;
            if (prepared.had_native_transaction)
                ++replayed_gpu_transactions;
            else
                ++captured_gpu_transactions;
        }

        LOG_DEBUG(
            "[MoEOverlayParticipantGraphRunner] Submitted every mapped "
            "follower GPU parent command="
            << ticket.command_id
            << " ordinal=" << ticket.transaction_ordinal
            << " captured=" << captured_gpu_transactions
            << " replayed=" << replayed_gpu_transactions);

        std::size_t cpu_layer_dispatches = 0u;
        for (const auto &endpoint : cached.mapped_cpu_endpoints)
        {
            const ArmedLane *const lane = endpoint
                                               ? armed_lane_for(
                                                     endpoint->lane_authority)
                                               : nullptr;
            const auto context_it = endpoint
                                        ? execution_contexts_.find(
                                              endpoint->device)
                                        : execution_contexts_.end();
                if (!endpoint || !endpoint->device.is_cpu() || !lane ||
                !endpoint->lane_authority ||
                !endpoint->lane_authority->protocol ||
                !endpoint->lane.valid() ||
                !endpoint->input_rows || !endpoint->output_rows ||
                context_it == execution_contexts_.end() ||
                !context_it->second)
            {
                abort_armed();
                return reject(
                    "Mapped ExpertOverlay CPU endpoint lost its exact protocol, packet, or execution-context authority");
            }

            auto &protocol = *endpoint->lane_authority->protocol;
            std::string protocol_error;
            if (!protocol.activate(
                    MoEOverlayActivationEndpoint::Follower,
                    lane->identity,
                    &protocol_error))
            {
                abort_armed();
                return reject(
                    "Mapped ExpertOverlay CPU follower could not activate participant " +
                    std::to_string(endpoint->participant_id) + ": " +
                    protocol_error);
            }

            /*
             * Bind the exact hidden matrix before the first descriptor is
             * acquired. The object address retained by every CPU local-expert
             * stage stays fixed; only this typed, ticket-derived view changes.
             * Metadata aliases are identical across shapes, while the hidden
             * pointer/layout pair is indivisible and may not be overridden by
             * a layer or inferred from live descriptor counts.
             */
            const auto admitted_payload =
                endpoint->lane.hostDispatchPayload(
                    static_cast<std::int32_t>(physical_rows_u64));
            if (!admitted_payload.hidden_rows_fp32 ||
                admitted_payload.hidden_row_capacity !=
                    static_cast<size_t>(physical_rows_u64) ||
                !isValidMoEOverlayActivationHiddenPayloadLayout(
                    admitted_payload.hidden_payload_layout))
            {
                abort_armed();
                return reject(
                    "Mapped ExpertOverlay CPU follower could not bind the exact ticket-selected activation payload view");
            }
            *endpoint->input_rows = admitted_payload;

            for (const auto &layer : endpoint->layers)
            {
                if (!layer.local_expert || layer.model_layer_index < 0)
                {
                    abort_armed();
                    return reject(
                        "Mapped ExpertOverlay CPU follower lost a prepared layer endpoint");
                }
                CPULayerTimelineSample layer_timing{
                    .participant_id = endpoint->participant_id,
                    .model_layer_index = layer.model_layer_index,
                    .stage_ordinal = layer.stage_ordinal,
                };
                const std::uint32_t bank =
                    moeOverlayActivationBufferIndex(layer.stage_ordinal);
                const std::uint64_t expected_timeline =
                    moeOverlayActivationLeasedTimelineValue(
                        moeOverlayActivationBufferVisit(
                            layer.stage_ordinal));
                /*
                 * Each routed layer is one collective rendezvous. Starting a
                 * fresh deadline here prevents a healthy 48-layer transaction
                 * from inheriting a wall-clock budget created before layer 0.
                 */
                const auto dispatch_rendezvous =
                    MoEOverlayActivationRendezvousDeadline::begin(
                        MoEOverlayActivationRendezvousKind::
                            DispatchPublication,
                        std::chrono::milliseconds(
                            collective_timeout_policy::
                                kDefaultCollectiveTimeoutMs));
                const auto dispatch_wait_begin = collect_timeline
                                                     ? Clock::now()
                                                     : Clock::time_point{};
                while (dispatch_rendezvous.waitingAllowed() &&
                       protocol.dispatchTimeline(bank) < expected_timeline)
                {
                    if (protocol.endpointState(
                            MoEOverlayActivationEndpoint::Continuation) ==
                        MoEOverlayActivationEndpointState::Aborted)
                    {
                        break;
                    }
                    std::this_thread::yield();
                }

                auto descriptor = protocol.consumeDispatch(
                    lane->identity,
                    layer.stage_ordinal,
                    &protocol_error);
                if (collect_timeline)
                {
                    layer_timing.dispatch_wait_ns = accumulate_elapsed(
                        cpu_dispatch_wait_ns,
                        dispatch_wait_begin,
                        Clock::now());
                }
                if (!descriptor)
                {
                    /*
                     * This is a terminal-only snapshot of the shared protocol.
                     * It adds no steady-path polling, but distinguishes an
                     * expired watchdog from an illegal timeline regression or
                     * a continuation graph that never opened its generation.
                     */
                    const auto now = std::chrono::steady_clock::now();
                    const auto remaining_us =
                        dispatch_rendezvous.remainingMicroseconds(now);
                    const auto active_identity = protocol.activeIdentity();
                    const auto continuation = protocol.endpointStatus(
                        MoEOverlayActivationEndpoint::Continuation);
                    const auto follower = protocol.endpointStatus(
                        MoEOverlayActivationEndpoint::Follower);
                    std::ostringstream diagnostic;
                    diagnostic
                        << "Mapped ExpertOverlay CPU follower could not consume layer "
                        << layer.model_layer_index
                        << " for participant " << endpoint->participant_id
                        << ": " << protocol_error
                        << "; terminal_snapshot={timeline="
                        << protocol.dispatchTimeline(bank)
                        << ",expected=" << expected_timeline
                        << ",deadline_remaining_us=" << remaining_us
                        << ",admission="
                        << static_cast<std::uint32_t>(
                               protocol.admissionState())
                        << ",active_generation="
                        << active_identity.epoch_generation
                        << ",active_placement_floor="
                        << active_identity.placement_epoch_floor
                        << ",continuation_state=" << continuation.state
                        << ",continuation_code=" << continuation.code
                        << ",continuation_operation="
                        << continuation.operation
                        << ",continuation_observed="
                        << continuation.observed_timeline
                        << ",continuation_diagnostic="
                        << continuation.failure_diagnostic
                        << ",continuation_auxiliary="
                        << continuation.failure_auxiliary
                        << ",continuation_generation="
                        << continuation.generation
                        << ",continuation_published="
                        << continuation.last_published_stage
                        << ",follower_state=" << follower.state
                        << ",follower_code=" << follower.code
                        << ",follower_operation=" << follower.operation
                        << ",follower_generation=" << follower.generation
                        << ",follower_consumed="
                        << follower.last_consumed_stage << "}";
                    if (config_.device_controller_fabric)
                    {
                        /*
                         * A missing dispatch can be downstream of the
                         * continuation group's captured epoch-acquire barrier.
                         * The barrier lives in mapped host pages, so this
                         * terminal-only diagnostic can expose every participant
                         * sequence without submitting GPU work or perturbing
                         * the inference stream whose failure we are reporting.
                         */
                        diagnostic
                            << ' '
                            << config_.device_controller_fabric
                                   ->describeInferenceEpochBarrier();
                    }
                    abort_armed();
                    return reject(diagnostic.str());
                }

                auto &input = *endpoint->input_rows;
                auto &output = *endpoint->output_rows;
                const auto packet_prepare_begin = collect_timeline
                                                      ? Clock::now()
                                                      : Clock::time_point{};
                if (descriptor->live_rows > physical_rows_u64 ||
                    descriptor->live_rows > input.row_capacity ||
                    descriptor->live_entries > input.entry_capacity ||
                    descriptor->live_entries >
                        descriptor->live_rows *
                            static_cast<std::uint64_t>(input.top_k))
                {
                    abort_armed();
                    return reject(
                        "Mapped ExpertOverlay CPU dispatch descriptor exceeds its exact ticket or shared-page capacity");
                }

                switch (ticket.graph_role)
                {
                case MoEOverlayInferenceGraphRole::MainPrefill:
                case MoEOverlayInferenceGraphRole::MainDecode:
                    input.key = makeMoEOverlayCollectiveKey(
                        ticket.request_generation,
                        ticket.logical_step_id,
                        layer.model_layer_index,
                        endpoint->tier_index,
                        endpoint->routed_domain_ordinal,
                        endpoint->participant_id,
                        MoEOverlayCollectiveDirection::Dispatch);
                    input.key.histogram_source =
                        ticket.graph_role ==
                                MoEOverlayInferenceGraphRole::MainPrefill
                            ? ExpertHistogramSource::PrefillChunk
                            : ExpertHistogramSource::DecodeToken;
                    break;
                case MoEOverlayInferenceGraphRole::MTPDraft:
                case MoEOverlayInferenceGraphRole::MTPGroupedVerifier:
                    input.key = makeMTPMoEOverlayCollectiveKey(
                        ticket.request_generation,
                        ticket.logical_step_id,
                        ticket.graph_role ==
                                MoEOverlayInferenceGraphRole::MTPDraft
                            ? cached.mtp_graph_depth
                            : ticket.draft_depth,
                        layer.model_layer_index,
                        endpoint->tier_index,
                        endpoint->routed_domain_ordinal,
                        endpoint->participant_id,
                        MoEOverlayCollectiveDirection::Dispatch);
                    input.key.histogram_source =
                        ExpertHistogramSource::GroupedVerifier;
                    break;
                case MoEOverlayInferenceGraphRole::None:
                    abort_armed();
                    return reject(
                        "Mapped ExpertOverlay CPU follower received a terminal graph role");
                }
                input.residency_epoch = descriptor->placement_epoch;
                input.source_participant =
                    lane->identity.source_participant_id;
                input.target_participant = endpoint->participant_id;
                input.live_row_count =
                    static_cast<size_t>(descriptor->live_rows);
                input.live_entry_count =
                    static_cast<size_t>(descriptor->live_entries);
                layer_timing.live_rows = input.live_row_count;
                layer_timing.live_entries = input.live_entry_count;

                /*
                 * Match the device consume kernel's structural validation before
                 * CPU GEMM sees any shared bytes. Release publication of the
                 * descriptor is the acquire edge for every payload read below.
                 */
                const size_t expected_payload_bytes =
                    compactMoEOverlayDispatchBytes(input);
                bool valid_packet =
                    descriptor->payload_bytes == expected_payload_bytes &&
                    input.entry_offsets_host[0] == 0;
                std::int32_t previous_row = -1;
                std::int32_t previous_offset = 0;
                for (size_t row = 0;
                     valid_packet && row < input.live_row_count;
                     ++row)
                {
                    const std::int32_t row_id = input.row_ids_host[row];
                    const std::int32_t begin =
                        input.entry_offsets_host[row];
                    const std::int32_t end =
                        input.entry_offsets_host[row + 1u];
                    valid_packet =
                        row_id > previous_row && row_id >= 0 &&
                        static_cast<std::uint64_t>(row_id) <
                            physical_rows_u64 &&
                        begin == previous_offset && end > begin &&
                        end - begin <= input.top_k &&
                        end <= static_cast<std::int32_t>(
                                   input.live_entry_count);
                    previous_row = row_id;
                    previous_offset = end;
                }
                valid_packet = valid_packet &&
                               previous_offset ==
                                   static_cast<std::int32_t>(
                                       input.live_entry_count);
                if (!valid_packet)
                {
                    std::ostringstream diagnostic;
                    diagnostic
                        << "Mapped ExpertOverlay CPU follower rejected "
                           "malformed shared dispatch rows"
                        << " layer=" << layer.model_layer_index
                        << " participant=" << endpoint->participant_id
                        << " live_rows=" << input.live_row_count
                        << " live_entries=" << input.live_entry_count
                        << " payload_bytes=" << descriptor->payload_bytes
                        << " expected_payload_bytes="
                        << expected_payload_bytes
                        << " initial_offset="
                        << input.entry_offsets_host[0]
                        << " final_offset=" << previous_offset;
                    if (input.live_row_count > 0u)
                    {
                        diagnostic
                            << " first_row_id=" << input.row_ids_host[0]
                            << " first_row_end="
                            << input.entry_offsets_host[1u];
                    }
                    abort_armed();
                    return reject(diagnostic.str());
                }
                if (collect_timeline)
                {
                    layer_timing.packet_prepare_ns = accumulate_elapsed(
                        cpu_packet_prepare_ns,
                        packet_prepare_begin,
                        Clock::now());
                }

                output.live_row_count = 0u;
                const auto expert_compute_begin = collect_timeline
                                                      ? Clock::now()
                                                      : Clock::time_point{};
                const bool expert_compute_ok =
                    layer.local_expert->execute(context_it->second);
                if (collect_timeline)
                {
                    layer_timing.expert_compute_ns = accumulate_elapsed(
                        cpu_expert_compute_ns,
                        expert_compute_begin,
                        Clock::now());
                }
                if (!expert_compute_ok ||
                    output.live_row_count != input.live_row_count ||
                    output.residency_epoch != input.residency_epoch ||
                    output.source_participant != endpoint->participant_id ||
                    output.target_participant != input.source_participant)
                {
                    abort_armed();
                    return reject(
                        "Mapped ExpertOverlay CPU local expert failed or returned an incomplete packet for layer " +
                        std::to_string(layer.model_layer_index) +
                        " participant " +
                        std::to_string(endpoint->participant_id));
                }
                for (size_t row = 0;
                     row < output.live_row_count;
                     ++row)
                {
                    if (output.row_ids_host[row] !=
                        input.row_ids_host[row])
                    {
                        abort_armed();
                        return reject(
                            "Mapped ExpertOverlay CPU return row identity diverged from dispatch order");
                    }
                }

                const auto return_publication_begin = collect_timeline
                                                          ? Clock::now()
                                                          : Clock::time_point{};
                const std::uint64_t return_bytes =
                    moeOverlayReturnPayloadBytes(
                        input.live_entry_count,
                        static_cast<std::uint32_t>(input.d_model));
                if (!protocol.publishReturn(
                        lane->identity,
                        layer.stage_ordinal,
                        return_bytes,
                        &protocol_error))
                {
                    abort_armed();
                    return reject(
                        "Mapped ExpertOverlay CPU follower could not publish layer " +
                        std::to_string(layer.model_layer_index) + ": " +
                        protocol_error);
                }
                if (collect_timeline)
                {
                    layer_timing.return_publication_ns = accumulate_elapsed(
                        cpu_return_publication_ns,
                        return_publication_begin,
                        Clock::now());
                    cpu_layer_timeline.push_back(layer_timing);
                }
                ++cpu_layer_dispatches;
            }

            if (!protocol.complete(
                    MoEOverlayActivationEndpoint::Follower,
                    lane->identity,
                    &protocol_error))
            {
                abort_armed();
                return reject(
                    "Mapped ExpertOverlay CPU follower could not publish terminal completion: " +
                    protocol_error);
            }
        }

        /*
         * A source parent fans one sparse stage out to every follower before it
         * can advance.  Waiting endpoint events one at a time means the first
         * incomplete event is not necessarily the lane withholding a return:
         * it may be parked behind another participant at the source fan-in.
         * Capture one topology-wide snapshot for every timeout so the fatal
         * report identifies the actual protocol frontier without adding any
         * steady-path polling, transfer, or synchronization.
         */
        const auto topology_timeout_diagnostic =
            [this, shape, &ticket](
                const CachedParticipantGraph::MappedGPUFollowerEndpoint *
                    observed_endpoint)
        {
            /*
             * A timed-out resident wait can occupy the backend scheduler in a
             * way that also prevents a newly submitted diagnostic D2H copy from
             * making progress. Reading device protocol structs here therefore
             * risks hanging the fatal path after its bounded terminal timeout.
             * The activation epoch and controller fabrics are host-mapped by
             * design, so report their acquire-loaded state directly without
             * submitting GPU work or introducing a second synchronization
             * protocol solely for diagnostics.
             */
            std::ostringstream out;
            out << "ticket={" << ticket.toString() << "}"
                << " physical_rows=" << shape->physical_rows
                << " gpu_endpoint_count=" << shape->endpoints.size();
            if (config_.device_controller_fabric)
            {
                /*
                 * Continuation participants freeze one exact placement epoch
                 * through a node-local mapped barrier before they can arm the
                 * remote sparse lanes.  A follower terminal timeout is often
                 * the first host-visible symptom of a missing continuation
                 * arrival, so include the acquire-loaded barrier lanes once
                 * per topology snapshot.  This reads only the fabric's mapped
                 * diagnostic alias; it neither submits GPU work nor changes
                 * the inference/maintenance ordering under investigation.
                 */
                out << ' '
                    << config_.device_controller_fabric
                           ->describeInferenceEpochBarrier();
            }
            for (const auto &owned_endpoint : shape->endpoints)
            {
                const auto *const endpoint = owned_endpoint.get();
                out << " endpoint={";
                if (!endpoint)
                {
                    out << "null}";
                    continue;
                }
                out << "observed="
                    << (endpoint == observed_endpoint ? 1 : 0)
                    << ",participant=" << endpoint->participant_id
                    << ",device=" << endpoint->device.toString()
                    << ",cache_initialized="
                    << (endpoint->cache.initialized ? 1 : 0)
                    << ",submission_state="
                    << static_cast<std::uint32_t>(
                           endpoint->cache.executable_submission_state)
                    << ",replay_ready="
                    << (endpoint->cache.retained_full_graph_replay.valid()
                            ? 1
                            : 0);
                if (!endpoint->lane_authority ||
                    !endpoint->lane_authority->protocol)
                {
                    out << ",protocol=missing}";
                    continue;
                }

                const auto &authority = *endpoint->lane_authority;
                const auto &protocol = *authority.protocol;
                const auto identity = protocol.activeIdentity();
                const auto continuation = protocol.endpointStatus(
                    MoEOverlayActivationEndpoint::Continuation);
                const auto follower = protocol.endpointStatus(
                    MoEOverlayActivationEndpoint::Follower);
                out << ",next_generation="
                    << authority.next_epoch_generation
                    << ",active_generation=" << identity.epoch_generation
                    << ",active_step=" << identity.logical_step_id
                    << ",active_role=" << identity.graph_role
                    << ",active_rows="
                    << identity.physical_rows_per_request
                    << ",admission="
                    << static_cast<std::uint32_t>(
                           protocol.admissionState())
                    << ",continuation={state=" << continuation.state
                    << ",code=" << continuation.code
                    << ",operation=" << continuation.operation
                    << ",evidence=" << continuation.observed_timeline
                    << ",diagnostic="
                    << continuation.failure_diagnostic
                    << ",auxiliary="
                    << continuation.failure_auxiliary
                    << ",consumed="
                    << continuation.last_consumed_stage
                    << ",published="
                    << continuation.last_published_stage << "}"
                    << ",follower={state=" << follower.state
                    << ",code=" << follower.code
                    << ",operation=" << follower.operation
                    << ",evidence=" << follower.observed_timeline
                    << ",diagnostic=" << follower.failure_diagnostic
                    << ",auxiliary=" << follower.failure_auxiliary
                    << ",consumed=" << follower.last_consumed_stage
                    << ",published=" << follower.last_published_stage
                    << "}";
                for (std::uint32_t bank = 0u;
                     bank < kMoEOverlayActivationBufferCount;
                     ++bank)
                {
                    out << ",bank" << bank << "={dispatch="
                        << protocol.dispatchTimeline(bank)
                        << ",return="
                        << protocol.returnTimeline(bank) << "}";
                }

                const auto runtime_it =
                    participant_gpu_runtimes_.find(
                        endpoint->participant_id);
                if (runtime_it != participant_gpu_runtimes_.end() &&
                    runtime_it->second)
                {
                    const auto &receipt =
                        runtime_it->second->inference_boundary_receipt;
                    out << ",boundary_receipt={submitted="
                        << receipt.submittedGeneration()
                        << ",completed="
                        << receipt.completedGeneration()
                        << ",consumed="
                        << receipt.consumedGeneration() << "}";
                }
                const auto progress_it =
                    transfer_progress_epochs_.find(endpoint->device);
                if (progress_it != transfer_progress_epochs_.end() &&
                    progress_it->second)
                {
                    const auto progress = progress_it->second->stats();
                    out << ",transfer_progress={slots="
                        << progress.slots_reserved
                        << ",published=" << progress.commands_published
                        << ",completed=" << progress.commands_completed
                        << ",bytes=" << progress.bytes_completed
                        << ",dma_submissions="
                        << progress.dma_submissions
                        << ",idle_skips="
                        << progress.idle_submission_skips
                        << ",in_flight_observations="
                        << progress.in_flight_observations
                        << ",failures=" << progress.command_failures
                        << "}";
                }
                if (config_.device_controller_fabric)
                {
                    try
                    {
                        const auto binding =
                            config_.device_controller_fabric
                                ->participantBinding(
                                    endpoint->participant_id);
                        const auto load32 = [](std::uint32_t &value)
                        {
                            return std::atomic_ref<std::uint32_t>(value)
                                .load(std::memory_order_acquire);
                        };
                        const auto load64 = [](std::uint64_t &value)
                        {
                            return std::atomic_ref<std::uint64_t>(value)
                                .load(std::memory_order_acquire);
                        };
                        if (binding.controller && binding.command)
                        {
                            auto &controller = *binding.controller;
                            auto &command = *binding.command;
                            out << ",controller={state="
                                << load32(controller.state)
                                << ",kind="
                                << load32(controller.transaction_kind)
                                << ",error="
                                << load32(controller.error_code)
                                << ",transaction="
                                << load64(controller.transaction_id)
                                << ",base_epoch="
                                << load64(controller.base_epoch)
                                << ",candidate_epoch="
                                << load64(controller.candidate_epoch)
                                << ",durable_epoch="
                                << load64(controller.current_durable_epoch)
                                << ",admission_transaction="
                                << load64(controller.admission_transaction)
                                << ",completed_transaction="
                                << load64(controller.completed_transaction)
                                << ",placement_layer_cursor="
                                << load32(controller.placement_layer_cursor)
                                << ",admission_epoch="
                                << load64(controller.admission_epoch)
                                << ",command_transaction="
                                << load64(controller.command_transaction)
                                << ",commit_transaction="
                                << load64(controller.commit_transaction)
                                << ",commands="
                                << load32(command.command_count) << "}";
                        }
                        if (binding.local_participant_record)
                        {
                            auto &record =
                                *binding.local_participant_record;
                            out << ",controller_participant={status="
                                << load32(record.status_code)
                                << ",snapshot="
                                << load64(record.snapshot_transaction)
                                << ",prepared="
                                << load64(record.prepared_transaction)
                                << ",published="
                                << load64(record.published_transaction)
                                << ",retirement_ready="
                                << load64(record.retirement_ready_epoch)
                                << ",retired="
                                << load64(record.retired_epoch) << "}";
                        }
                        if (binding.local_group)
                        {
                            auto &group = *binding.local_group;
                            out << ",controller_group={status="
                                << load32(group.status_code)
                                << ",snapshot="
                                << load64(group.snapshot_transaction)
                                << ",prepared="
                                << load64(group.prepared_transaction)
                                << ",published="
                                << load64(group.published_transaction)
                                << ",retired="
                        << load64(group.retired_epoch) << "}";
                        }
                    }
                    catch (const std::exception &diagnostic_error)
                    {
                        out << ",controller=<unavailable:"
                            << diagnostic_error.what() << ">";
                    }
                }
                out << "}";
            }
            return out.str();
        };

        try
        {
            for (const auto &terminal : submitted_gpu_terminals)
            {
                if (!terminal.endpoint || !terminal.ticket.valid())
                {
                    throw std::logic_error(
                        "Mapped follower lost an exact GPU terminal ticket");
                }
                terminal.endpoint->cache
                    .waitForPublishedCaptureStreamTerminal(
                    terminal.ticket,
                    DeviceGraphExecutor::GraphSegmentCache::
                        HostFenceWaitPolicy::ActiveProgress,
                    [&, observed_endpoint = terminal.endpoint]()
                    {
                        return topology_timeout_diagnostic(
                            observed_endpoint);
                    });
            }
        }
        catch (const std::exception &exception)
        {
            abort_armed();
            return reject(
                std::string(
                    "Mapped ExpertOverlay follower terminal event failed: ") +
                    exception.what());
        }

        /* Endpoint completion is a final independent rendezvous after all
         * exact local work and GPU terminal events have retired. */
        const auto completion_rendezvous =
            MoEOverlayActivationRendezvousDeadline::begin(
                MoEOverlayActivationRendezvousKind::EndpointCompletion,
                std::chrono::milliseconds(
                    collective_timeout_policy::
                        kDefaultCollectiveTimeoutMs));
        const auto endpoint_completion_begin = collect_timeline
                                                   ? Clock::now()
                                                   : Clock::time_point{};
        while (completion_rendezvous.waitingAllowed())
        {
            bool complete = true;
            for (const auto &lane : armed)
            {
                const auto follower_state =
                    lane.authority->protocol->endpointState(
                        MoEOverlayActivationEndpoint::Follower);
                const auto continuation_state =
                    lane.authority->protocol->endpointState(
                        MoEOverlayActivationEndpoint::Continuation);
                if (follower_state ==
                        MoEOverlayActivationEndpointState::Aborted ||
                    continuation_state ==
                        MoEOverlayActivationEndpointState::Aborted)
                {
                    const auto follower_status =
                        lane.authority->protocol->endpointStatus(
                            MoEOverlayActivationEndpoint::Follower);
                    const auto continuation_status =
                        lane.authority->protocol->endpointStatus(
                            MoEOverlayActivationEndpoint::Continuation);
                    std::ostringstream diagnostic;
                    diagnostic
                        << "Mapped ExpertOverlay follower observed an aborted "
                           "device endpoint for participant "
                        << lane.authority->participant_id
                        << "; continuation={state="
                        << continuation_status.state << ",code="
                        << continuation_status.code << ",operation="
                        << continuation_status.operation << ",timeline="
                        << continuation_status.observed_timeline
                        << ",published_stage="
                        << continuation_status.last_published_stage
                        << ",consumed_stage="
                        << continuation_status.last_consumed_stage
                        << ",model_layer="
                        << continuation_status.last_model_layer
                        << "}; follower={state=" << follower_status.state
                        << ",code=" << follower_status.code
                        << ",operation=" << follower_status.operation
                        << ",timeline=" << follower_status.observed_timeline
                        << ",published_stage="
                        << follower_status.last_published_stage
                        << ",consumed_stage="
                        << follower_status.last_consumed_stage
                        << ",model_layer="
                        << follower_status.last_model_layer << "}";
                    try
                    {
                        auto *const mapped = dynamic_cast<
                            IMoEOverlayMappedActivationTransport *>(
                            lane.authority->transport_lifetime.get());
                        if (mapped && lane.authority->transport_lifetime &&
                            lane.authority->transport_lifetime
                                ->hasSharedRowStorage())
                        {
                            const auto &control =
                                mapped->activationEpochControl(
                                    lane.authority->participant_id,
                                    lane.authority->graph_family_ordinal);
                            const auto rows =
                                lane.authority->transport_lifetime
                                    ->sharedDispatchRows(
                                        lane.authority->participant_id);
                            const auto &buffer = control.buffers[0];
                            diagnostic
                                << "; shared_stage0={admission="
                                << control.admission.ready_signal << "/"
                                << control.admission.state
                                << ",dispatch_signal="
                                << buffer.dispatch_signal.value
                                << ",return_signal="
                                << buffer.return_signal.value
                                << ",descriptor_timeline="
                                << buffer.dispatch_descriptor.timeline
                                << ",descriptor_rows="
                                << buffer.dispatch_descriptor.live_rows
                                << ",descriptor_entries="
                                << buffer.dispatch_descriptor.live_entries
                                << ",descriptor_bytes="
                                << buffer.dispatch_descriptor.payload_bytes
                                << ",descriptor_stage="
                                << buffer.dispatch_descriptor.stage_ordinal
                                << ",descriptor_layer="
                                << buffer.dispatch_descriptor.model_layer_index;
                            if (rows.row_ids_host &&
                                rows.entry_offsets_host)
                            {
                                diagnostic
                                    << ",row0=" << rows.row_ids_host[0]
                                    << ",offset0="
                                    << rows.entry_offsets_host[0]
                                    << ",offset1="
                                    << rows.entry_offsets_host[1];
                            }
                            diagnostic << "}";
                        }
                    }
                    catch (const std::exception &exception)
                    {
                        diagnostic << "; shared_stage0=<unavailable:"
                                   << exception.what() << ">";
                    }
                    abort_armed();
                    return reject(diagnostic.str());
                }
                complete = complete &&
                           follower_state ==
                               MoEOverlayActivationEndpointState::Complete &&
                           continuation_state ==
                               MoEOverlayActivationEndpointState::Complete;
            }
            if (complete)
                break;
            std::this_thread::yield();
        }
        if (collect_timeline)
        {
            accumulate_elapsed(
                endpoint_completion_wait_ns,
                endpoint_completion_begin,
                Clock::now());
        }

        for (const auto &lane : armed)
        {
            if (lane.authority->protocol->endpointState(
                    MoEOverlayActivationEndpoint::Follower) !=
                    MoEOverlayActivationEndpointState::Complete ||
                lane.authority->protocol->endpointState(
                    MoEOverlayActivationEndpoint::Continuation) !=
                    MoEOverlayActivationEndpointState::Complete)
            {
                std::string timeout_error;
                const auto timeout_observation =
                    completion_rendezvous.timeoutObservationNanoseconds();
                if (!timeout_observation)
                {
                    return reject(
                        "Mapped ExpertOverlay follower reached an incomplete endpoint outside a valid timeout observation");
                }
                (void)lane.authority->protocol->markTimedOut(
                    lane.identity.epoch_generation,
                    *timeout_observation,
                    &timeout_error);
                return reject(
                    "Mapped ExpertOverlay follower timed out waiting for both endpoint-complete publications: " +
                        timeout_error);
            }
        }

        /*
         * Both endpoint Complete stores are now acquired, making their
         * graph-owned traffic totals immutable.  Materialize PerfStats before
         * reset clears the lease; this is the only host observation and it
         * occurs once per heterogeneous transaction, never once per layer.
         */
        const auto add_checked = [&](std::uint64_t &total,
                                     std::uint64_t value,
                                     const char *label) -> bool
        {
            if (total > std::numeric_limits<std::uint64_t>::max() - value)
            {
                if (error)
                    *error = std::string("Mapped ExpertOverlay ") + label +
                             " evidence overflowed uint64_t";
                return false;
            }
            total += value;
            return true;
        };
        const char *const service_source = [&ticket]() noexcept
        {
            switch (ticket.graph_role)
            {
            case MoEOverlayInferenceGraphRole::MainPrefill:
                return "prefill";
            case MoEOverlayInferenceGraphRole::MainDecode:
                return "decode";
            case MoEOverlayInferenceGraphRole::MTPDraft:
            case MoEOverlayInferenceGraphRole::MTPGroupedVerifier:
                return "grouped_verifier";
            case MoEOverlayInferenceGraphRole::None:
                return "invalid";
            }
            return "invalid";
        }();
        for (const auto &lane : armed)
        {
            std::string traffic_error;
            const auto traffic = lane.authority->protocol->completedTraffic(
                lane.identity, &traffic_error);
            if (!traffic)
            {
                return reject(
                    "Mapped ExpertOverlay follower could not acquire completed traffic for participant " +
                    std::to_string(lane.authority->participant_id) + ": " +
                    traffic_error);
            }
            if (!add_checked(
                    mapped_dispatch_bytes,
                    traffic->dispatch_payload_bytes,
                    "dispatch-byte") ||
                !add_checked(
                    mapped_return_bytes,
                    traffic->return_payload_bytes,
                    "return-byte") ||
                !add_checked(
                    mapped_dispatch_rows,
                    traffic->dispatch_live_rows,
                    "dispatch-row") ||
                !add_checked(
                    mapped_return_rows,
                    traffic->return_live_rows,
                    "return-row") ||
                !add_checked(
                    mapped_dispatch_entries,
                    traffic->dispatch_live_entries,
                    "dispatch-entry"))
            {
                return false;
            }

            const auto &authority = *lane.authority;
            const char *const device_kind =
                authority.device.is_cpu()
                    ? "CPU"
                    : (authority.device.is_cuda() ? "CUDA" : "ROCm");
            const PerfStatsCollector::Tags traffic_tags{
                {"completion", "device_owned_epoch_complete"},
                {"device_kind", device_kind},
                {"identity_source", "device_owned_activation_epoch"},
                {"participant",
                 std::to_string(authority.participant_id)},
                {"stage_count",
                 std::to_string(traffic->dispatch_stage_count)},
                {"tier", std::to_string(authority.tier_index)},
                {"tier_priority",
                 std::to_string(authority.tier_priority)},
                {"transport", "compact"},
            };
            if (traffic->dispatch_payload_bytes > 0u)
            {
                PerfStatsCollector::addCounter(
                    "moe_overlay",
                    "compact_dispatch_bytes",
                    static_cast<double>(traffic->dispatch_payload_bytes),
                    "gn_sparse_dispatch",
                    authority.device.to_string(),
                    traffic_tags);
            }
            if (traffic->return_payload_bytes > 0u)
            {
                PerfStatsCollector::addCounter(
                    "moe_overlay",
                    "compact_return_bytes",
                    static_cast<double>(traffic->return_payload_bytes),
                    "gn_return_reduce",
                    authority.device.to_string(),
                    traffic_tags);
            }
            if (traffic->return_live_rows > 0u)
            {
                PerfStatsCollector::addCounter(
                    "moe_overlay",
                    authority.device.is_cpu()
                        ? "device_epoch_cpu_rows"
                        : "device_epoch_gpu_rows",
                    static_cast<double>(traffic->return_live_rows),
                    "gn_local_expert",
                    authority.device.to_string(),
                    {{"completion", "device_owned_epoch_complete"},
                     {"domain_kind",
                      authority.device.is_cpu() ? "CPU" : "GPU"},
                     {"participant",
                      std::to_string(authority.participant_id)},
                     {"tier", std::to_string(authority.tier_index)},
                     {"transport", "local"}});
            }
            if (traffic->dispatch_live_entries > 0u)
            {
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "moe_overlay_local_expert_active_routes",
                    static_cast<double>(traffic->dispatch_live_entries),
                    "moe_overlay",
                    authority.device.to_string(),
                    {{"completion", "device_owned_epoch_complete"},
                     {"device_kind", device_kind},
                     {"identity_source",
                      "device_owned_activation_epoch"},
                     {"input_rows",
                      std::to_string(traffic->dispatch_live_rows)},
                     {"output_rows",
                      std::to_string(traffic->return_live_rows)},
                     {"participant",
                      std::to_string(authority.participant_id)},
                     {"service_source", service_source},
                     {"tier", std::to_string(authority.tier_index)}});
            }
        }

        if (captured_gpu_transactions > 0u)
        {
            PerfStatsCollector::addCounter(
                "forward_graph",
                "device_timeline_transaction_captures",
                static_cast<double>(captured_gpu_transactions),
                transaction_phase,
                participantDeviceList(local_participants_),
                {{"gpu_host_layer_dispatches", "0"},
                 {"physical_rows", std::to_string(physical_rows_u64)},
                 {"role", "expert_transaction_follower"}});
        }
        if (replayed_gpu_transactions > 0u)
        {
            PerfStatsCollector::addCounter(
                "forward_graph",
                "device_timeline_transaction_replays",
                static_cast<double>(replayed_gpu_transactions),
                transaction_phase,
                participantDeviceList(local_participants_),
                {{"gpu_host_layer_dispatches", "0"},
                 {"physical_rows", std::to_string(physical_rows_u64)},
                 {"role", "expert_transaction_follower"}});
        }

        for (const auto &lane : armed)
        {
            std::string reset_error;
            if (!lane.authority->protocol->reset(
                    lane.identity, &reset_error))
            {
                return reject(
                    "Mapped ExpertOverlay follower could not retire participant " +
                        std::to_string(
                            lane.authority->participant_id) + ": " +
                        reset_error);
            }
        }

        /*
         * Protocol reset is possible only after every exact GPU terminal event
         * and both endpoint Complete publications were observed.  Publish each
         * participant receipt last so a maintenance notification can never
         * mistake a merely submitted graph for a committed inference boundary.
         */
        for (const auto &boundary : submitted_boundaries)
        {
            if (!boundary.runtime ||
                !boundary.runtime->inference_boundary_receipt
                     .completeSubmission(boundary.generation))
            {
                return reject(
                    "Mapped ExpertOverlay follower could not publish its serial inference-boundary receipt");
            }
        }

        PerfStatsCollector::addCounter(
            "moe_overlay_participant_graph",
            "mapped_device_owned_epochs",
            1.0,
            transaction_phase,
            participantDeviceList(local_participants_),
            {{"captured_gpu_transactions",
              std::to_string(captured_gpu_transactions)},
             {"dispatch_bytes", std::to_string(mapped_dispatch_bytes)},
             {"dispatch_entries",
              std::to_string(mapped_dispatch_entries)},
             {"dispatch_rows", std::to_string(mapped_dispatch_rows)},
             {"gpu_endpoints", std::to_string(shape->endpoints.size())},
             {"cpu_endpoints",
              std::to_string(cached.mapped_cpu_endpoints.size())},
             {"physical_rows", std::to_string(shape->physical_rows)},
             {"gpu_host_layer_dispatches", "0"},
             {"cpu_host_layer_dispatches",
              std::to_string(cpu_layer_dispatches)},
             {"replayed_gpu_transactions",
              std::to_string(replayed_gpu_transactions)},
             {"return_bytes", std::to_string(mapped_return_bytes)},
             {"return_rows", std::to_string(mapped_return_rows)},
             {"terminal_event_fences",
              std::to_string(shape->endpoints.size())}});
        if (collect_timeline && !cached.mapped_cpu_endpoints.empty())
        {
            const PerfStatsCollector::Tags timeline_tags{
                {"cpu_endpoints",
                 std::to_string(cached.mapped_cpu_endpoints.size())},
                {"cpu_layer_dispatches",
                 std::to_string(cpu_layer_dispatches)},
                {"graph_role",
                 std::to_string(
                     static_cast<std::uint32_t>(ticket.graph_role))},
                {"physical_rows", std::to_string(physical_rows_u64)},
                {"timing_semantics", "aggregate_nonoverlapping_cpu_phases"},
            };
            const auto record_timeline = [&](const char *name,
                                             std::uint64_t ns)
            {
                PerfStatsCollector::recordTimingNs(
                    "moe_overlay_participant_graph",
                    name,
                    std::max<std::uint64_t>(1u, ns),
                    transaction_phase,
                    participantDeviceList(local_participants_),
                    timeline_tags);
            };
            record_timeline(
                "mapped_cpu_dispatch_wait", cpu_dispatch_wait_ns);
            record_timeline(
                "mapped_cpu_packet_prepare", cpu_packet_prepare_ns);
            record_timeline(
                "mapped_cpu_expert_compute", cpu_expert_compute_ns);
            record_timeline(
                "mapped_cpu_return_publication",
                cpu_return_publication_ns);
            record_timeline(
                "mapped_endpoint_completion_wait",
                endpoint_completion_wait_ns);

            /* Emit fine-grained observations after every live endpoint and
             * terminal event has completed. The recorded values therefore
             * describe the unperturbed rendezvous rather than PerfStats work
             * interleaved with the inference transaction. */
            for (const auto &sample : cpu_layer_timeline)
            {
                const PerfStatsCollector::Tags layer_tags{
                    {"graph_role",
                     std::to_string(
                         static_cast<std::uint32_t>(ticket.graph_role))},
                    {"live_entries", std::to_string(sample.live_entries)},
                    {"live_rows", std::to_string(sample.live_rows)},
                    {"model_layer",
                     std::to_string(sample.model_layer_index)},
                    {"participant",
                     std::to_string(sample.participant_id)},
                    {"physical_rows", std::to_string(physical_rows_u64)},
                    {"stage_ordinal",
                     std::to_string(sample.stage_ordinal)},
                    {"timing_semantics",
                     "deferred_per_layer_nonoverlapping"},
                };
                const auto record_layer_timeline =
                    [&](const char *name, std::uint64_t ns)
                {
                    PerfStatsCollector::recordTimingNs(
                        "moe_overlay_participant_graph",
                        name,
                        std::max<std::uint64_t>(1u, ns),
                        transaction_phase,
                        "CPU",
                        layer_tags);
                };
                record_layer_timeline(
                    "mapped_cpu_layer_dispatch_wait",
                    sample.dispatch_wait_ns);
                record_layer_timeline(
                    "mapped_cpu_layer_packet_prepare",
                    sample.packet_prepare_ns);
                record_layer_timeline(
                    "mapped_cpu_layer_expert_compute",
                    sample.expert_compute_ns);
                record_layer_timeline(
                    "mapped_cpu_layer_return_publication",
                    sample.return_publication_ns);
            }
        }
        return true;
    }

    /**
     * @brief Select and execute one retained graph from an authenticated ticket.
     *
     * Qwen's one learned NextN block is recurrent across speculative positions:
     * ticket `sidecar_depth` names that position, not another set of weights.
     * The retained graph therefore stays at manifest graph depth zero while
     * the transaction sequence carries positions 0..draft_depth-1. A future
     * model with several learned sidecar blocks must extend the ticket ABI with
     * a distinct graph-depth field instead of guessing from that ordinal.
     */
    bool MoEOverlayParticipantGraphRunner::
        executeMoEOverlayInferenceTransaction(
            const MoEOverlayInferenceTransactionTicket &ticket,
            std::string *error)
    {
        using Clock = std::chrono::steady_clock;
        const bool timing_enabled =
            PerfStatsCollector::isDomainEnabled(
                "moe_overlay_participant_graph");
        const auto selection_begin =
            timing_enabled ? Clock::now() : Clock::time_point{};
        if (error)
            error->clear();
        std::optional<MoEOverlayInferenceInterferenceScope>
            interference_scope;
        const auto fail = [&](std::string diagnostic)
        {
            if (interference_scope)
                interference_scope->discard();
            if (error)
                *error = diagnostic;
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Transaction execution rejected: "
                << diagnostic << " (" << ticket.toString() << ")");
            return false;
        };

        if (!ticket.valid() ||
            ticket.action != MoEOverlayInferenceTransactionAction::Execute)
        {
            return fail(
                "executor requires one valid retained-graph execution ticket");
        }
        if (ticket.target_world_rank != config_.mpi_context->rank() ||
            ticket.source_world_rank !=
                execution_plan_->continuation_root_rank)
        {
            return fail(
                "ticket source/target ranks do not match this participant graph");
        }

        const uint64_t request_count =
            static_cast<uint64_t>(ticket.request_count);
        const uint64_t logical_rows =
            request_count *
            static_cast<uint64_t>(ticket.logical_rows_per_request);
        const uint64_t physical_rows =
            request_count *
            static_cast<uint64_t>(ticket.physical_rows_per_request);
        if (logical_rows == 0 || logical_rows > physical_rows ||
            physical_rows > static_cast<uint64_t>(
                                config_.max_graph_activation_rows))
        {
            return fail(
                "ticket row geometry exceeds the setup-owned participant family");
        }

        CachedParticipantGraph *cached = nullptr;
        SparseTransactionPhase phase = SparseTransactionPhase::Decode;
        int mtp_graph_depth = -1;
        const char *role_name = "invalid";
        switch (ticket.graph_role)
        {
        case MoEOverlayInferenceGraphRole::MainPrefill:
            cached = main_graph_.get();
            phase = SparseTransactionPhase::Prefill;
            role_name = "main_prefill";
            break;
        case MoEOverlayInferenceGraphRole::MainDecode:
            cached = main_graph_.get();
            phase = SparseTransactionPhase::Decode;
            role_name = "main_decode";
            break;
        case MoEOverlayInferenceGraphRole::MTPGroupedVerifier:
            cached = main_graph_.get();
            phase = SparseTransactionPhase::GroupedVerifier;
            mtp_graph_depth = ticket.draft_depth;
            role_name = "mtp_grouped_verifier";
            break;
        case MoEOverlayInferenceGraphRole::MTPDraft:
            if (mtp_sidecar_graphs_.size() != 1u)
            {
                return fail(
                    "recurrent MTP tickets require exactly one learned retained sidecar graph");
            }
            cached = mtp_sidecar_graphs_.front().get();
            phase = SparseTransactionPhase::MTPDraft;
            mtp_graph_depth = cached ? cached->mtp_graph_depth : -1;
            role_name = "mtp_draft";
            break;
        case MoEOverlayInferenceGraphRole::None:
            return fail("terminal graph role reached the execution authority");
        }

        ExpertHistogramSource calibration_source =
            ExpertHistogramSource::SyntheticTest;
        int calibration_depth = 0;
        if (ticket.graph_role == MoEOverlayInferenceGraphRole::MainPrefill)
            calibration_source = ExpertHistogramSource::PrefillChunk;
        else if (ticket.graph_role == MoEOverlayInferenceGraphRole::MainDecode)
            calibration_source = ExpertHistogramSource::DecodeToken;
        else if (ticket.graph_role ==
                 MoEOverlayInferenceGraphRole::MTPGroupedVerifier)
        {
            calibration_source = ExpertHistogramSource::GroupedVerifier;
            calibration_depth = ticket.draft_depth;
        }
        if (calibration_source != ExpertHistogramSource::SyntheticTest)
        {
            interference_scope.emplace(
                interference_probe_.get(),
                makeMoEOverlayInferenceWorkloadIdentity(
                    calibration_source,
                    static_cast<int>(logical_rows),
                    static_cast<int>(physical_rows),
                    /*transaction_count=*/1,
                    calibration_depth));
            if (interference_scope->active())
            {
                LOG_DEBUG(
                    "[ExpertOverlay][Calibration] Follower rank "
                    << config_.mpi_context->rank()
                    << " claimed baseline/concurrent sample logical_step="
                    << ticket.logical_step_id << " rows=" << logical_rows
                    << '/' << physical_rows);
            }
        }

        if (!cached || cached->row_capacity <= 0 ||
            (!cached->graph && !cached->usesMappedActivationEpochs()) ||
            physical_rows > static_cast<uint64_t>(cached->row_capacity))
        {
            return fail(
                "selected retained graph is absent or smaller than ticket geometry");
        }
        const auto selection_end =
            timing_enabled ? Clock::now() : Clock::time_point{};
        const bool mapped_followers = cached->usesMappedActivationEpochs();
        if (!mapped_followers &&
            !stampMoEOverlayCollectiveRuntime(
                *cached,
                ticket.request_generation,
                ticket.logical_step_id,
                phase,
                mtp_graph_depth,
                ticket.placement_epoch))
        {
            return fail(
                "selected retained graph rejected its runtime wire namespace");
        }
        const auto stamp_end =
            timing_enabled ? Clock::now() : Clock::time_point{};
        std::string execution_error;
        const bool executed = mapped_followers
                                  ? executeMappedFollowerTransaction(
                                        *cached,
                                        ticket,
                                        &execution_error)
                                  : executor_.executeRetainedMultiDevice(
                                        cached->execution_plan,
                                        &execution_error);
        if (!executed)
        {
            std::string diagnostic =
                "selected retained participant graph execution failed: " +
                execution_error;
            if (config_.device_controller_fabric)
            {
                /*
                 * The continuation rank owns the mapped LocalTP epoch barrier.
                 * If its retained parent stalls before publishing layer zero,
                 * expose each device arrival lane at this terminal boundary;
                 * the remote follower cannot inspect this rank-local fabric.
                 */
                diagnostic += ' ';
                diagnostic += config_.device_controller_fabric
                                  ->describeInferenceEpochBarrier();
            }
            return fail(std::move(diagnostic));
        }
        if (interference_scope && interference_scope->active())
        {
            LOG_DEBUG(
                "[ExpertOverlay][Calibration] Follower rank "
                << config_.mpi_context->rank()
                << " reached exact retained-graph terminal logical_step="
                << ticket.logical_step_id << " rows=" << logical_rows
                << '/' << physical_rows);
        }
        const auto execution_end =
            timing_enabled ? Clock::now() : Clock::time_point{};

        if (timing_enabled)
        {
            const PerfStatsCollector::Tags tags{
                {"logical_rows", std::to_string(logical_rows)},
                {"physical_rows", std::to_string(physical_rows)},
                {"draft_depth", std::to_string(ticket.draft_depth)},
                {"sidecar_ordinal", std::to_string(ticket.sidecar_depth)},
                {"parallel_gpu_waves",
                 mapped_followers
                     ? "device_owned_epoch"
                     : std::to_string(
                           cached->execution_plan.concurrent_gpu_wave_count)},
                {"max_wave_width",
                 mapped_followers
                     ? std::to_string(
                           cached->mappedShapeForRows(
                               static_cast<int>(physical_rows))
                               ->endpoints.size())
                     : std::to_string(
                           cached->execution_plan.max_wave_width)},
                {"mapped_gpu_endpoints",
                 mapped_followers
                     ? std::to_string(
                           cached->mappedShapeForRows(
                               static_cast<int>(physical_rows))
                               ->endpoints.size())
                     : "0"},
                {"mapped_cpu_endpoints",
                 mapped_followers
                     ? std::to_string(
                           cached->mapped_cpu_endpoints.size())
                     : "0"},
                {"gpu_host_layer_dispatches", "0"},
            };
            const auto record = [&](const char* name, auto begin, auto end)
            {
                const auto elapsed =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        end - begin)
                        .count();
                PerfStatsCollector::recordTimingNs(
                    "moe_overlay_participant_graph",
                    name,
                    static_cast<std::uint64_t>(
                        std::max<std::int64_t>(1, elapsed)),
                    role_name,
                    participantDeviceList(local_participants_),
                    tags);
            };
            record(
                "ticket_selection_and_validation",
                selection_begin,
                selection_end);
            record(
                mapped_followers
                    ? "mapped_epoch_selection"
                    : "runtime_sparse_stage_stamping",
                selection_end,
                stamp_end);
            record(
                mapped_followers
                    ? "mapped_device_owned_epoch_execution"
                    : "multi_device_graph_execution",
                stamp_end,
                execution_end);
        }

        const char *const evidence_phase =
            phase == SparseTransactionPhase::Prefill
                ? "prefill"
                : (phase == SparseTransactionPhase::Decode
                       ? "decode"
                       : role_name);
        PerfStatsCollector::recordOrderedSequenceStep(
            "forward_graph",
            "moe_overlay_collective_transaction_sequence",
            {ticket.request_generation,
             ticket.logical_step_id,
             logical_rows,
             physical_rows},
            evidence_phase,
            participantDeviceList(local_participants_),
            {{"role", "expert_participant_graph"},
             {"identity_source", "orchestration_request_and_chunk"},
             {"logical_step_semantics", "monotonic_transaction"}});
        PerfStatsCollector::addCounter(
            "moe_overlay_participant_graph",
            "ticket_selected_graphs",
            1.0,
            role_name,
            participantDeviceList(local_participants_),
            {{"logical_step_semantics", "monotonic_transaction"},
             {"logical_rows", std::to_string(logical_rows)},
             {"physical_rows", std::to_string(physical_rows)},
             {"draft_depth", std::to_string(ticket.draft_depth)},
             {"sidecar_ordinal", std::to_string(ticket.sidecar_depth)},
             {"mtp_graph_depth", std::to_string(mtp_graph_depth)},
             {"parallel_gpu_waves",
              mapped_followers
                  ? "device_owned_epoch"
                  : std::to_string(
                        cached->execution_plan.concurrent_gpu_wave_count)},
             {"max_wave_width",
              mapped_followers
                  ? std::to_string(
                        cached->mappedShapeForRows(
                            static_cast<int>(physical_rows))
                            ->endpoints.size())
                  : std::to_string(
                        cached->execution_plan.max_wave_width)},
             {"transport_path",
              mapped_followers
                  ? (cached->mapped_cpu_endpoints.empty()
                         ? "node_local_mapped_device_epoch"
                         : "node_local_mapped_gpu_epoch_cpu_boundary")
                  : "host_scheduled_rank_batch"},
             {"mapped_gpu_endpoints",
              mapped_followers
                  ? std::to_string(
                        cached->mappedShapeForRows(
                            static_cast<int>(physical_rows))
                            ->endpoints.size())
                  : "0"},
             {"mapped_cpu_endpoints",
              mapped_followers
                  ? std::to_string(cached->mapped_cpu_endpoints.size())
                  : "0"},
             {"gpu_host_layer_dispatches", "0"},
             {"position_mutated", "false"}});
        return true;
    }

    /**
     * @brief Validate that a bounded shared schedule can fit this remote graph.
     *
     * The root passes the exact schedule into @ref forwardPrefillChunkSchedule.
     * This capability check rejects requests outside the full request context
     * or an unplanned compact-buffer family before an MPI collective can begin.
     */
    bool MoEOverlayParticipantGraphRunner::supportsPrefillChunkSchedule(
        int seq_len) const
    {
        return seq_len > 1 && config_.max_seq_len > 0 &&
               seq_len <= config_.max_seq_len &&
               config_.max_graph_activation_rows > 0;
    }

    /**
     * @brief Execute the continuation root's schedule with live sparse shapes.
     *
     * Padded continuation buckets are a GPU capture concern.  Sparse dispatch
     * packets carry only real rows, so this endpoint must build/replay its
     * graph at @c chunk.real_count rather than padding expert traffic or
     * changing the collective record order.  The root-published contract has
     * already limited bucket capacity to every endpoint's planner admission;
     * checks here keep a corrupted or stale caller from violating that promise.
     */
    bool MoEOverlayParticipantGraphRunner::forwardPrefillChunkSchedule(
        const int *tokens,
        int seq_len,
        const PrefillChunkSchedulerPolicy &policy,
        int pad_token_id,
        bool allow_padded_execution)
    {
        (void)pad_token_id;
        (void)allow_padded_execution;
        if (!tokens || !supportsPrefillChunkSchedule(seq_len) ||
            policy.real_token_count != seq_len)
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Invalid shared prefill "
                "chunk schedule input");
            return false;
        }

        const PrefillChunkSchedule schedule = planPrefillChunkSchedule(policy);
        if (!schedule || schedule.chunks.empty())
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Could not plan shared "
                "prefill chunks: " << schedule.error);
            return false;
        }
        if (position_ != policy.real_token_start)
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Shared prefill starts "
                "at a cursor not owned by this participant runner "
                "schedule_start=" << policy.real_token_start
                << " local_position=" << position_);
            return false;
        }

        uint64_t executed_real_rows = 0;
        for (const PrefillChunkPlan &chunk : schedule.chunks)
        {
            const int relative_offset =
                chunk.token_offset - policy.real_token_start;
            if (chunk.real_count <= 0 || relative_offset < 0 ||
                relative_offset > seq_len ||
                chunk.real_count > seq_len - relative_offset ||
                chunk.bucket_seq_len <= 0 ||
                chunk.bucket_seq_len > config_.max_graph_activation_rows ||
                chunk.real_count > config_.max_graph_activation_rows)
            {
                LOG_ERROR(
                    "[MoEOverlayParticipantGraphRunner] Shared prefill chunk "
                    "exceeds this participant's immutable graph capacity"
                    << " chunk=" << chunk.chunk_index
                    << " real_rows=" << chunk.real_count
                    << " bucket_rows=" << chunk.bucket_seq_len
                    << " capacity=" << config_.max_graph_activation_rows);
                return false;
            }

            if (!executeAtLogicalStep(
                    tokens + relative_offset,
                    chunk.real_count,
                    chunk.token_offset,
                    SparseTransactionPhase::Prefill,
                    chunk.bucket_seq_len))
            {
                LOG_ERROR(
                    "[MoEOverlayParticipantGraphRunner] Shared prefill chunk "
                    << chunk.chunk_index << " failed");
                return false;
            }
            executed_real_rows += static_cast<uint64_t>(chunk.real_count);
        }

        if (executed_real_rows != static_cast<uint64_t>(seq_len))
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Shared prefill chunks did "
                "not cover the requested real-token range");
            return false;
        }

        PerfStatsCollector::addCounter(
            "forward_graph",
            "moe_overlay_participant_prefill_chunk_schedules",
            static_cast<double>(schedule.chunks.size()),
            "prefill",
            participantDeviceList(local_participants_),
            {
                {"participants", participantIdList(local_participants_)},
                {"logical_rows", std::to_string(seq_len)},
                {"bucket_rows", std::to_string(schedule.chunks.front().bucket_seq_len)},
                {"capacity", std::to_string(config_.max_graph_activation_rows)},
                {"live_rows_only", "true"},
            });
        return true;
    }

    const float *MoEOverlayParticipantGraphRunner::logits() const
    {
        return nullptr;
    }

    int MoEOverlayParticipantGraphRunner::vocab_size() const
    {
        return 0;
    }

    void MoEOverlayParticipantGraphRunner::clear_cache()
    {
        position_ = 0;
    }

    bool MoEOverlayParticipantGraphRunner::purgePrefixCache()
    {
        return true;
    }

    int MoEOverlayParticipantGraphRunner::get_position() const
    {
        return position_;
    }

    ExecutionPath MoEOverlayParticipantGraphRunner::executionPath() const
    {
        return ExecutionPath::GRAPH;
    }

    const char *MoEOverlayParticipantGraphRunner::architecture() const
    {
        return architecture_.c_str();
    }

    std::vector<int>
    MoEOverlayParticipantGraphRunner::participantIds() const
    {
        std::vector<int> ids;
        ids.reserve(local_participants_.size());
        for (const auto *participant : local_participants_)
        {
            if (participant)
                ids.push_back(participant->participant_id);
        }
        return ids;
    }

    std::vector<DeviceId>
    MoEOverlayParticipantGraphRunner::localDevices() const
    {
        return participantDevices(local_participants_);
    }

    size_t MoEOverlayParticipantGraphRunner::cachedGraphCount() const noexcept
    {
        return (main_graph_ ? 1u : 0u) + mtp_sidecar_graphs_.size();
    }

    std::vector<MoEOverlayDeviceControllerRuntimeBinding>
    MoEOverlayParticipantGraphRunner::
        moeOverlayDeviceControllerRuntimeBindings() const
    {
        std::vector<int> participant_ids;
        participant_ids.reserve(participant_gpu_runtimes_.size());
        for (const auto &[participant_id, runtime] :
             participant_gpu_runtimes_)
        {
            if (!runtime)
            {
                throw std::logic_error(
                    "Mapped follower retained a null device-controller runtime");
            }
            participant_ids.push_back(participant_id);
        }
        std::sort(participant_ids.begin(), participant_ids.end());

        std::vector<MoEOverlayDeviceControllerRuntimeBinding> bindings;
        bindings.reserve(participant_ids.size());
        for (const int participant_id : participant_ids)
        {
            const auto &runtime =
                participantGpuRuntimeForParticipant(participant_id);
            if (!runtime.runtime_table || !runtime.epoch_arena ||
                runtime.runtime_table->layerCount() <= 0 ||
                runtime.initialized_layers.size() !=
                    static_cast<std::size_t>(
                        runtime.runtime_table->layerCount()) ||
                std::any_of(
                    runtime.initialized_layers.begin(),
                    runtime.initialized_layers.end(),
                    [](std::uint8_t initialized)
                    {
                        return initialized == 0u;
                    }))
            {
                throw std::logic_error(
                    "Mapped follower controller binding was requested before every runtime layer became durable");
            }

            MoEOverlayDeviceControllerRuntimeBinding binding{
                .device = runtime.device,
                .runtime_layers_device =
                    runtime.runtime_table->deviceLayerState(0),
                .runtime_table_host = runtime.runtime_table.get(),
                .service_telemetry_device =
                    runtime.runtime_table
                        ->deviceOverlayServiceTelemetry(),
                .service_samples_device =
                    runtime.runtime_table
                        ->deviceOverlayServiceTelemetrySample(0),
                .overlay_participant_id = runtime.participant_id,
                .domain_participant_id =
                    runtime.domain_participant_id,
                .domain_participant_count =
                    runtime.domain_participant_count,
                .layer_count = static_cast<std::uint32_t>(
                    runtime.runtime_table->layerCount()),
                .expert_count = static_cast<std::uint32_t>(
                    runtime.runtime_table->expertCount()),
                .top_k = static_cast<std::uint32_t>(
                    runtime.runtime_table->topK()),
                .epoch_control =
                    runtime.epoch_arena->deviceControlAddress(),
                .maintenance_epoch =
                    runtime.epoch_arena
                        ->deviceMaintenanceEpochAddress(),
                .maintenance_status =
                    runtime.epoch_arena
                        ->deviceMaintenanceStatusAddress(),
                .inference_boundary = const_cast<
                    ParticipantGpuRuntime *>(&runtime),
                .initial_runtime_publisher = const_cast<
                    ParticipantGpuRuntime *>(&runtime),
            };
            if (!binding.backgroundPublicationValid() ||
                !binding.initialRuntimePublicationValid())
            {
                throw std::logic_error(
                    "Mapped follower produced an incomplete background-publication binding");
            }
            bindings.push_back(binding);
        }
        return bindings;
    }

    MoEOverlayInferenceTransactionProtocol::Config
    MoEOverlayParticipantGraphRunner::
        inferenceTransactionProtocolConfig() const
    {
        if (!transaction_topology_identity_.valid())
        {
            throw std::logic_error(
                "Participant retained graph family has no transaction topology identity");
        }
        return {
            .topology = transaction_topology_identity_,
            .slot_count = moeOverlayInferenceTransactionSlotCount(
                config_.max_mtp_draft_depth),
            .max_request_count = config_.max_request_count,
            .max_rows_per_request =
                config_.max_graph_activation_rows,
            .max_mtp_draft_depth =
                config_.max_mtp_draft_depth,
        };
    }

    std::unique_ptr<IInferenceRunner>
    createMoEOverlayParticipantGraphRunner(
        MoEOverlayParticipantGraphRunnerConfig config)
    {
        try
        {
            return std::make_unique<MoEOverlayParticipantGraphRunner>(
                std::move(config));
        }
        catch (const std::exception &error)
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Construction failed: "
                << error.what());
            return nullptr;
        }
    }

} // namespace llaminar2
