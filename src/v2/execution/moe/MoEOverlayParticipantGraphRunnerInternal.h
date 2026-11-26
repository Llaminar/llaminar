/**
 * @file MoEOverlayParticipantGraphRunnerInternal.h
 * @brief Shared internal runtime structs and participant helpers for MoEOverlayParticipantGraphRunner shards.
 */
#pragma once
/**
 * @file MoEOverlayParticipantGraphRunner.cpp
 * @brief Real-weight retained experts under original rank or auxiliary ownership.
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
 * CPU serial storage declarations survive admission and runtime projection.
 * Per-layer CPU packet roots retain their own exact backing allocation claims.
 */
#include "execution/local_execution/graph/PreparedFollowerCaptureTerminals.h"

#include "MoEOverlayParticipantGraphRunner.h"
#include "MoEOverlayFinitePlacementGraphs.h"
#include "MoEOverlayPreparedGraphBank.h"
#include <mutex>
#include "execution/qwen4exp/PooledImmutableRetainedParticipants.h"
#include "MoEOverlayMTPSourceContract.h"
#include "MoEOverlayFollowerRowGeometry.h"
#include "MoECPUExpertWorkspaceSource.h"
#include "execution/compute_stages/stages/MoECPUExpertShardRouteStage.h"
#include "execution/moe/ExpertShardSourceSelection.h"
#include "MoEOverlayHostServiceContract.h"
#include "MoEOverlayParticipantResidency.h"
#include "MoEExpertOverlayPreparationPlan.h"

#include "backends/GPUDeviceContextPool.h"
#include "backends/BackendManager.h"
#include "collective/CollectiveTimeoutPolicy.h"
#include "execution/compute_stages/ComputeStageFactory.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/compute_stages/stages/MoELocalExpertStage.h"
#include "execution/compute_stages/stages/MoEOverlayActivationPacketStages.h"
#include "execution/compute_stages/stages/MoEOverlayROCmMappedFollowerStage.h"
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
#include "loaders/PreparedOverlaySourceRetirement.h"
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

#include "MoEOverlayOriginalNativeInvocationRetirement.h"
#include <algorithm>
#include <future>
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
#include <omp.h>

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

        /**
         * @brief Declare CPU arrival scratch from the original main/predictor loader.
         * @param loader Source-aware loader already admitted by the executing rank.
         * @param layer Layer whose complete experts this participant executes.
         * @param d_model Unsharded expert input/output width.
         * @param intermediate Unsharded gate/up output width.
         * @param experts Logical count used to validate all three parent shapes.
         * @return Source identities and this executing rank's CPU policy facts.
         * @throws std::logic_error For missing or inconsistent parent metadata.
         *
         * No weight is loaded or prepared here. Empty tiers declare the same
         * eventual scratch as populated tiers, before any captured peer starts.
         */
        MoELocalExpertStage::CPUExpertWorkspaceSource cpuExpertWorkspaceSource(
            const ModelLoader &loader, int layer, int d_model, int intermediate, int experts)
        {
            const auto formats = originalCPUExpertWorkspaceSourceFormats(
                loader, layer, d_model, intermediate, experts);
            return {.gate = formats[0], .up = formats[1], .down = formats[2],
                    .workers = omp_in_parallel() ? omp_get_num_threads() : omp_get_max_threads(),
                    .execution = CPUExecutionGeometry::local()};
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
             * Each physical-row graph retains its own exact capture stream.
             * The runtime joins the original previous terminal before launching
             * another family/shape, so their scratch lifetimes never overlap. Keeping this owner shared makes the runtime allocation
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
            std::unique_ptr<MoECPUExpertShardRouteStage> joint_route_collective; ///< Explicit original-domain edge before root weighting/return.
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
            std::vector<std::shared_ptr<MappedGPUFollowerEndpoint>> endpoints;
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
        /** True only when the admitted expert packet manifest is provably empty. */
        bool admitted_empty = false;

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

        /** Original declared endpoint keeps its source executor/arena alive for finite helpers. */
        std::shared_ptr<CachedParticipantGraph::MappedGPUFollowerEndpoint> finite_boundary_endpoint;
        /** Original participant context; borrowed only while this runner is inference-ready. */
        IDeviceContext *finite_boundary_context=nullptr;
        /** Bounded helper captures retire before the retained endpoint/source. */
        MoEOverlayFinitePlacementGraphs finite_placement_graphs;

        /** @copydoc IMoEOverlayDeviceInferenceBoundary::prepareMoEOverlayFinitePlacementGraphs */
        bool prepareMoEOverlayFinitePlacementGraphs() override
        {
            if(!worker || !epoch_arena || !controller_binding ||
               !finite_boundary_endpoint || !finite_boundary_endpoint->executor ||
               !finite_boundary_endpoint->workspace_allocator || !finite_boundary_context)return false;
            worker->submitAndWait([&] {
                finite_placement_graphs.prepare(*finite_boundary_endpoint->executor,
                    *finite_boundary_context,*worker,epoch_arena,*controller_binding,
                    finite_boundary_endpoint->workspace_allocator->deviceGeneration(device),false);
            });
            return true;
        }

        /** @copydoc IMoEOverlayDeviceInferenceBoundary::runMoEOverlayFinitePlacementBoundary */
        bool runMoEOverlayFinitePlacementBoundary(DeviceMoEOverlayEpochParentAction action) override
        {
            finite_placement_graphs.run(action,[&](void *stream) {
                return enqueueMoEOverlayDeviceInferenceBoundary(stream,
                    {.purpose=MoEOverlayInferenceBoundaryPurpose::PlacementDecisionSnapshot})==
                    MoEOverlayInferenceBoundaryStatus::Submitted;
            });
            return true;
        }

        /**
         * @brief Exact previous native producer protecting this participant's shared epoch slot/workspace.
         * Worker FIFO orders SDK enqueues; different cache streams require this
         * original event edge before another family or shape can reuse storage.
         */
        struct InferencePredecessor
        {
            std::shared_ptr<CachedParticipantGraph::MappedGPUFollowerEndpoint> endpoint;
            DeviceGraphExecutor::GraphSegmentCache::CaptureStreamTerminalTicket terminal;
        };
        /** Serializes only host receipt metadata; never held while observing native completion. */
        std::mutex inference_predecessor_mutex;
        std::optional<InferencePredecessor> inference_predecessor;

        /**
         * @brief Enqueue the previous exact producer on the next original capture stream.
         * @param next Actual retained endpoint for this same participant/device.
         * @param original_worker Same original worker which published the predecessor.
         * This consumes no host completion and allocates/records no event.
         */
        void enqueueOriginalInferencePredecessor(
            const CachedParticipantGraph::MappedGPUFollowerEndpoint &next,
            IWorkerGPUContext *original_worker)
        {
            if (next.participant_id != participant_id || next.device != device ||
                !next.cache.capture_stream || original_worker != worker)
                throw std::logic_error("Follower predecessor consumer changed its original participant/device/worker");
            std::lock_guard lock(inference_predecessor_mutex);
            if (!inference_predecessor) return;
            const auto &prior=*inference_predecessor;
            if (!prior.endpoint || prior.endpoint->participant_id != participant_id ||
                prior.endpoint->device != device || !prior.terminal.valid())
                throw std::logic_error("Follower predecessor lost its original retained producer");
            prior.endpoint->cache.enqueuePublishedCaptureStreamTerminalWait(
                prior.terminal, next.cache.capture_stream, original_worker);
        }

        /**
         * @brief Retain the actual successful terminal publication as the next exact producer.
         * @param endpoint Same original endpoint owner; keeps its cache/event alive.
         * @param terminal Receipt returned by that endpoint's actual native publication.
         */
        void publishOriginalInferencePredecessor(
            std::shared_ptr<CachedParticipantGraph::MappedGPUFollowerEndpoint> endpoint,
            DeviceGraphExecutor::GraphSegmentCache::CaptureStreamTerminalTicket terminal)
        {
            if (!endpoint || endpoint->participant_id != participant_id || endpoint->device != device ||
                !terminal.valid())
                throw std::logic_error("Follower predecessor publication has a foreign or missing original terminal");
            std::lock_guard lock(inference_predecessor_mutex);
            inference_predecessor=InferencePredecessor{std::move(endpoint),terminal};
        }

        /**
         * @brief Withdraw only the latest link AFTER its actual original terminal was observed.
         * @param endpoint Producer whose host receipt observation just completed.
         * @param terminal Exact original observed publication, never an enqueue future.
         * Older aggregate observations leave a newer outstanding producer intact.
         */
        void retireOriginalInferencePredecessor(
            const CachedParticipantGraph::MappedGPUFollowerEndpoint *endpoint,
            DeviceGraphExecutor::GraphSegmentCache::CaptureStreamTerminalTicket terminal)
        {
            std::lock_guard lock(inference_predecessor_mutex);
            if (inference_predecessor && inference_predecessor->endpoint.get()==endpoint &&
                inference_predecessor->terminal.generation==terminal.generation)
                inference_predecessor.reset();
        }

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

} // namespace llaminar2
