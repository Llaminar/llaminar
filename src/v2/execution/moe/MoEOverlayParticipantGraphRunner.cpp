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
#include "collective/CollectiveTimeoutPolicy.h"
#include "execution/compute_stages/ComputeStageFactory.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/compute_stages/stages/MoELocalExpertStage.h"
#include "execution/compute_stages/stages/MoEOverlayActivationPacketStages.h"
#include "execution/compute_stages/stages/MoERankBatchSparseStages.h"
#include "execution/compute_stages/stages/MoESparseDispatchStage.h"
#include "execution/compute_stages/stages/MoESparseReturnReduceStage.h"
#include "execution/local_execution/device/WorkspaceAllocator.h"
#include "execution/local_execution/engine/PrefillBucketUtils.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/moe/MoEOverlayNodeLocalRankBatchTransport.h"
#include "execution/moe/MoEOverlayRankBatchTransport.h"
#include "execution/moe/MoEOverlaySparseCollective.h"
#include "execution/moe/MoEOverlayResidencyAuthority.h"
#include "loaders/ModelContext.h"
#include "loaders/PreparedWeightStore.h"
#include "loaders/WeightManager.h"
#include "memory/BufferArena.h"
#include "tensors/Tensors.h"
#include "transfer/TransferEngine.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <array>
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
        std::unique_ptr<WorkspaceAllocator> workspace_allocator;
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

        WeightPlan plan(strategy);
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

        if (plan.empty())
        {
            throw std::runtime_error(
                "MoE overlay participant owns no routed-expert weight "
                "requirements");
        }
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
        if (config_.mtp_enabled && config_.max_mtp_draft_depth <= 0)
        {
            throw std::invalid_argument(
                "MTP-enabled participant runner requires a positive admitted draft depth");
        }
        if (!config_.mtp_enabled && config_.max_mtp_draft_depth != 0)
        {
            throw std::invalid_argument(
                "MTP-disabled participant runner cannot advertise a draft-depth capacity");
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
        resolveTopology();
        resolveGraphFamilies();
        createSerialCompactBufferArenas();
        prepareParticipantWeights();
        createDeviceContexts();
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
        main_graph_ = buildGraph(main_spec);
        if (!main_graph_)
        {
            throw std::runtime_error(
                "MoE overlay participant main graph construction returned null");
        }

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
        if (execution_plan_->buildsRootGraph())
        {
            throw std::invalid_argument(
                "Continuation root must use the dense model runner, not the "
                "expert-only participant runner");
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
                config_.mtp_enabled,
                /*graph_family_generation=*/1,
                config_.max_graph_activation_rows,
                config_.max_decode_activation_rows,
                config_.max_request_count,
                config_.max_mtp_draft_depth);
        main_layer_count_ = transaction_graph_family_.main_layer_count;
        mtp_source_layers_ = transaction_graph_family_.mtp_source_layers;

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
     * Relay-only ranks intentionally produce no arena.  Any endpoint that
     * owns at least one expert in the authenticated map must produce exactly
     * one arena; later graph construction treats a missing entry as a fatal
     * ownership violation rather than allocating a stage-private substitute.
     */
    void MoEOverlayParticipantGraphRunner::createSerialCompactBufferArenas()
    {
        const int d_model = config_.model_context->embeddingLength();
        const auto &loader = config_.model_context->concreteLoader();
        const std::string &architecture = config_.model_context->architecture();
        const int num_experts = loader.getInt(architecture + ".expert_count", 0);
        const int routing_top_k =
            loader.getInt(architecture + ".expert_used_count", 0);
        if (d_model <= 0 || num_experts <= 0 || routing_top_k <= 0)
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

            bool owns_selected_expert = false;
            for (int layer = 0;
                 layer < config_.model_context->totalBlockCount();
                 ++layer)
            {
                if (!owner_map_->expertsForParticipant(
                        layer, participant->participant_id)
                         .empty())
                {
                    owns_selected_expert = true;
                    break;
                }
            }
            if (!owns_selected_expert)
                continue;

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
        const ModelContextId model_id = plan.strategy().model_id;

        prepared_store_ = weight_manager->preparedWeightStoreIfInitialized();
        if (!prepared_store_)
        {
            prepared_store_ =
                std::make_shared<PreparedWeightStore>(model_id);
            weight_manager->setPreparedWeightStore(prepared_store_);
        }
        if (!prepared_store_->bindModelIdIfUnset(model_id))
        {
            throw std::runtime_error(
                "Participant PreparedWeightStore belongs to another model");
        }

        if (plan.empty())
        {
            /* Relay ranks own protocol state but no expert weight authority. */
            return;
        }

        frozen_weights_ = std::make_unique<FrozenModelWeightSet>(
            weight_manager->materialize(plan));
        for (const DeviceId device : participantDevices(local_participants_))
        {
            if (!weight_manager->prepareMoEExpertOverlayWeights(
                    *runtime_plan_,
                    device,
                    frozen_weights_.get(),
                    execution_plan_.get()))
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
        auto &registry =
            config_.model_context->concreteWeightManager()
                ->expertGemmRegistry();
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
            std::ostringstream key_stream;
            key_stream << "tier" << tier_index
                       << "#domain" << routed_domain_ordinal
                       << "#rank" << source_world_rank << "to"
                       << local_world_rank << "#p";
            for (const int participant : participants)
                key_stream << participant << ',';
            const std::string key = key_stream.str();
            const auto existing =
                rank_batch_transports_.find(key);
            if (existing != rank_batch_transports_.end())
                return existing->second;

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

            std::vector<DeviceId> local_devices;
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
                if (std::find(
                        local_devices.begin(),
                        local_devices.end(),
                        participant->device) == local_devices.end())
                {
                    local_devices.push_back(participant->device);
                }
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
                        .local_devices = std::move(local_devices),
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
                endpoint->input_rows =
                    std::make_shared<MoEOverlaySparseRows>(
                        binding.transport->sharedDispatchRows(
                            endpoint->participant_id));
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

            std::unordered_map<
                DeviceId,
                std::shared_ptr<WorkspaceAllocator>>
                mapped_workspace_allocators;

            /*
             * Materialize the largest geometry first. WorkspaceAllocator's
             * serial-family contract then binds every smaller retained graph
             * to the already-published maximum buffers and rejects any shape
             * the setup BOM omitted. Iterating in ascending order would ask a
             * live serial allocator to grow after its first graph captured
             * addresses, which is intentionally forbidden.
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

                    TransferEngine::allocateDeviceStorage(
                        tensor_family->hidden.get(), target_device);
                    TransferEngine::allocateDeviceStorage(
                        tensor_family->routing_indices.get(), target_device);
                    TransferEngine::allocateDeviceStorage(
                        tensor_family->routing_weights.get(), target_device);
                    TransferEngine::allocateDeviceStorage(
                        tensor_family->output.get(), target_device);
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
                            tensor_family->output.get()))
                    {
                        throw std::runtime_error(
                            "Mapped ExpertOverlay follower could not register its complete arena frontier");
                    }

                    endpoint->tensor_lifetimes = {
                        tensor_family->hidden,
                        tensor_family->routing_indices,
                        tensor_family->routing_weights,
                        tensor_family->output,
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
                        if (hasActiveMask(expert_mask))
                        {
                            std::vector<ITensorGemm *> prepared_gate;
                            std::vector<ITensorGemm *> prepared_up;
                            std::vector<ITensorGemm *> prepared_down;
                            (void)registry.populateExpertEnginesForParticipant(
                                binding.tier->domain,
                                target_device,
                                binding.participant->world_rank,
                                binding.participant
                                    ->domain_participant_index,
                                layer,
                                num_experts,
                                prepared_gate,
                                prepared_up,
                                prepared_down);

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
                        return_params.local_output =
                            tensor_family->output.get();
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

                    auto [workspace_it, workspace_inserted] =
                        mapped_workspace_allocators.try_emplace(
                            target_device, nullptr);
                    if (workspace_inserted)
                    {
                        workspace_it->second =
                            std::make_shared<WorkspaceAllocator>();
                    }
                    endpoint->workspace_allocator = workspace_it->second;
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
                    shape.endpoints.push_back(std::move(endpoint));
                }
                result->mapped_gpu_shapes.push_back(std::move(shape));
            }

            std::size_t setup_materialized_gpu_transactions = 0u;
            for (auto &shape : result->mapped_gpu_shapes)
            {
                for (auto &endpoint : shape.endpoints)
                {
                    if (!endpoint || !endpoint->graph || !endpoint->executor ||
                        !endpoint->cache.capture_stream)
                    {
                        throw std::runtime_error(
                            "Mapped ExpertOverlay follower lost a graph, executor, or exact stream before setup materialization");
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
                            "Mapped ExpertOverlay follower could not resolve its setup-time device context or native timeline envelope");
                    }

                    /*
                     * Record and instantiate every admitted endpoint graph before
                     * request admission. The dispatch-consume kernels are only
                     * recorded here: no activation epoch exists, no graph is
                     * launched, and no arena write is published. Consequently an
                     * authenticated inference ticket can submit a replay
                     * immediately instead of performing first-use HIP/CUDA graph
                     * construction while the continuation device is already
                     * waiting for its remote expert rows.
                     */
                    endpoint->graph->reset();
                    if (!endpoint->executor->executeWithCachedGraphReplay(
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
                            DeviceGraphExecutor::GraphReplayPlanPolicy::
                                RequireFullGraph,
                            {},
                            {},
                            {},
                            DeviceGraphExecutor::GraphInitialSubmissionPolicy::
                                MaterializeWithoutLaunch) ||
                        !endpoint->cache.retained_full_graph_replay.valid() ||
                        endpoint->cache.executable_submission_state !=
                            DeviceGraphExecutor::GraphSegmentCache::
                                ExecutableSubmissionState::
                                    MaterializedUnlaunched)
                    {
                        throw std::runtime_error(
                            "Mapped ExpertOverlay follower could not seal its setup-owned native timeline transaction without launching it for participant " +
                            std::to_string(endpoint->participant_id) +
                            " rows=" + std::to_string(shape.physical_rows));
                    }
                    ++setup_materialized_gpu_transactions;
                }
            }

            PerfStatsCollector::addCounter(
                "moe_overlay_participant_graph",
                "materialized_mapped_follower_families",
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
                 {"setup_materialized_gpu_transactions",
                  std::to_string(setup_materialized_gpu_transactions)},
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
            std::make_unique<WorkspaceAllocator>();
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
        SparseTransactionPhase phase)
    {
        (void)tokens;
        try
        {
            if (seq_len <= 0 || logical_step < 0 || logical_step != position_)
            {
                LOG_ERROR(
                    "[MoEOverlayParticipantGraphRunner] Invalid sparse "
                    "collective request cursor logical_step="
                    << logical_step << " local_position=" << position_
                    << " rows=" << seq_len);
                return false;
            }
            if (overlay_collective_request_generation_ == 0)
            {
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
                return false;
            std::string execution_error;
            const bool ok = executor_.executeRetainedMultiDevice(
                cached.execution_plan, &execution_error);
            if (!ok && !execution_error.empty())
            {
                LOG_ERROR(
                    "[MoEOverlayParticipantGraphRunner] Retained participant schedule failed: "
                    << execution_error);
            }
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
                /*
                 * One record represents the whole remote graph transaction,
                 * not each layer's manual boundary.  Its key tags must match
                 * the continuation-graph record emitted by
                 * ForwardExecutionEngine for this same request chunk.
                 */
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "moe_overlay_collective_transaction",
                    1.0,
                    phase == SparseTransactionPhase::Decode
                        ? "decode"
                        : "prefill",
                    participantDeviceList(local_participants_),
                    {{"role", "expert_participant_graph"},
                     {"identity_source", "orchestration_request_and_chunk"},
                     {"generation", std::to_string(overlay_collective_request_generation_)},
                     {"logical_step", std::to_string(logical_step)}});
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
                    {"logical_step", std::to_string(logical_step)},
                });
            return ok;
        }
        catch (const std::exception &error)
        {
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
        int mtp_graph_depth)
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
        if (error)
            error->clear();
        const auto reject = [error](std::string message)
        {
            if (error)
                *error = std::move(message);
            return false;
        };
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

        struct ArmedLane
        {
            std::shared_ptr<CachedParticipantGraph::MappedLaneAuthority>
                authority;
            MoEOverlayActivationEpochIdentity identity;
        };
        std::vector<ArmedLane> armed;
        armed.reserve(cached.mapped_lane_authorities.size());

        const auto deadline =
            std::chrono::steady_clock::now() +
            std::chrono::milliseconds(
                collective_timeout_policy::kDefaultCollectiveTimeoutMs);
        const auto deadline_ns_signed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                deadline.time_since_epoch())
                .count();
        if (deadline_ns_signed <= 0)
        {
            return reject(
                "Mapped ExpertOverlay follower could not derive a watchdog deadline");
        }
        const uint64_t deadline_ns =
            static_cast<uint64_t>(deadline_ns_signed);

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
                ticket, generation, deadline_ns, &arm_error);
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

        /* Queue every complete native GPU transaction before CPU work begins. */
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
                abort_armed();
                return reject(
                    "Mapped ExpertOverlay follower endpoint is missing its device context or native timeline envelope");
            }
            const bool had_native_transaction =
                endpoint->cache.retained_full_graph_replay.valid();
            if (!endpoint->executor->executeWithCachedGraphReplay(
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
                    DeviceGraphExecutor::GraphReplayPlanPolicy::
                        RequireFullGraph,
                    {},
                    {},
                    {}))
            {
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
            if (had_native_transaction)
                ++replayed_gpu_transactions;
            else
                ++captured_gpu_transactions;
        }

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

            for (const auto &layer : endpoint->layers)
            {
                if (!layer.local_expert || layer.model_layer_index < 0)
                {
                    abort_armed();
                    return reject(
                        "Mapped ExpertOverlay CPU follower lost a prepared layer endpoint");
                }
                const std::uint32_t bank =
                    moeOverlayActivationBufferIndex(layer.stage_ordinal);
                const std::uint64_t expected_timeline =
                    moeOverlayActivationLeasedTimelineValue(
                        moeOverlayActivationBufferVisit(
                            layer.stage_ordinal));
                while (std::chrono::steady_clock::now() < deadline &&
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
                if (!descriptor)
                {
                    abort_armed();
                    return reject(
                        "Mapped ExpertOverlay CPU follower could not consume layer " +
                        std::to_string(layer.model_layer_index) +
                        " for participant " +
                        std::to_string(endpoint->participant_id) + ": " +
                        protocol_error);
                }

                auto &input = *endpoint->input_rows;
                auto &output = *endpoint->output_rows;
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

                /*
                 * Match the device consume kernel's structural validation before
                 * CPU GEMM sees any shared bytes. Release publication of the
                 * descriptor is the acquire edge for every payload read below.
                 */
                bool valid_packet =
                    descriptor->payload_bytes ==
                        compactMoEOverlayDispatchBytes(input) &&
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
                    abort_armed();
                    return reject(
                        "Mapped ExpertOverlay CPU follower rejected malformed shared dispatch rows");
                }

                output.live_row_count = 0u;
                if (!layer.local_expert->execute(context_it->second) ||
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

                const std::uint64_t return_bytes =
                    static_cast<std::uint64_t>(
                        compactMoEOverlayReturnBytes(output));
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

        try
        {
            for (const auto &endpoint : shape->endpoints)
            {
                endpoint->cache.waitForCaptureStreamFence(
                    DeviceGraphExecutor::GraphSegmentCache::
                        HostFenceWaitPolicy::ActiveProgress);
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

        while (std::chrono::steady_clock::now() < deadline)
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
                (void)lane.authority->protocol->markTimedOut(
                    lane.identity.epoch_generation,
                    deadline_ns,
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
                {"epoch_generation",
                 std::to_string(lane.identity.epoch_generation)},
                {"generation",
                 std::to_string(lane.identity.request_generation)},
                {"identity_source", "device_owned_activation_epoch"},
                {"logical_step",
                 std::to_string(lane.identity.logical_step_id)},
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
                    authority.device.is_cpu() ? "cpu_rows" : "gpu_rows",
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
                     {"generation",
                      std::to_string(lane.identity.request_generation)},
                     {"identity_source",
                      "device_owned_activation_epoch"},
                     {"input_rows",
                      std::to_string(traffic->dispatch_live_rows)},
                     {"logical_step",
                      std::to_string(lane.identity.logical_step_id)},
                     {"output_rows",
                      std::to_string(traffic->return_live_rows)},
                     {"participant",
                      std::to_string(authority.participant_id)},
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
        const auto fail = [&](std::string diagnostic)
        {
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
                mtp_graph_depth))
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
            return fail(
                "selected retained participant graph execution failed: " +
                execution_error);
        }
        const auto execution_end =
            timing_enabled ? Clock::now() : Clock::time_point{};

        if (timing_enabled)
        {
            const PerfStatsCollector::Tags tags{
                {"command_id", std::to_string(ticket.command_id)},
                {"transaction_ordinal",
                 std::to_string(ticket.transaction_ordinal)},
                {"logical_step", std::to_string(ticket.logical_step_id)},
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

        PerfStatsCollector::addCounter(
            "moe_overlay_participant_graph",
            "ticket_selected_graphs",
            1.0,
            role_name,
            participantDeviceList(local_participants_),
            {{"request_generation",
              std::to_string(ticket.request_generation)},
             {"command_id", std::to_string(ticket.command_id)},
             {"transaction_ordinal",
              std::to_string(ticket.transaction_ordinal)},
             {"logical_step", std::to_string(ticket.logical_step_id)},
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
                    SparseTransactionPhase::Prefill))
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
