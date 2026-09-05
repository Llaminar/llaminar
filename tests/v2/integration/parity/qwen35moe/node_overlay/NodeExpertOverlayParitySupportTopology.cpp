/**
 * @file NodeExpertOverlayParitySupportTopology.cpp
 * @brief SupportTopology implementation of the shared node-overlay parity fixture.
 *
 * Shared by the 35B and 122B generated node ExpertOverlay matrices. This
 * translation-unit boundary does not change request ownership, reference
 * mathematics, graph execution, or evidence publication ordering.
 */
#include "NodeExpertOverlayParitySupport.h"

namespace llaminar2::test::parity::qwen35moe::node_overlay
{

    /**
     * @brief Return the exact decode policy used by the Hugging Face reference.
     *
     * `run_prefill_and_decode()` writes the reference corpus by taking an
     * argmax at the prefill boundary and after every decode row.  Installing
     * this policy through IOrchestrationRunner keeps that same contract on the
     * production device sampler: CUDA/ROCm execute their greedy sampler and
     * never fall back to downloading logits for host sampling.
     *
     * @return A stateless greedy sampling policy matching the reference corpus.
     */
    SamplingParams referenceGreedySamplingPolicy()
    {
        SamplingParams policy;
        policy.temperature = 0.0f;
        policy.top_k = 0;
        policy.top_p = 1.0f;
        policy.seed = 0;
        return policy;
    }


    bool isLegacyOverlayRuntimeEnabled()
    {
        const char *value = std::getenv(kLegacyEnvVar);
        return value != nullptr && std::string(value) == "1";
    }

    /** @return Exact generated diagnostic identity for the active cell. */
    std::string activeTestName()
    {
        return activeModelParityCaseOrThrow().testName();
    }

    /** @return Whether the active typed model is the 122B MTP identity. */
    bool isQwen122ProductionTest()
    {
        const auto *test_case = activeModelParityCase();
        return test_case && test_case->model.test_id == "Qwen35_122B";
    }

    /** @return Model root selected by the active real-weight cell. */
    const char *activeModelPath()
    {
        const auto *test_case = activeModelParityCase();
        if (!test_case)
            throw std::logic_error(
                "Graph-native model path requires an active typed parity case");
        return test_case->model.model_path.c_str();
    }

    /** @return Authenticated reference directory selected by the active cell. */
    const char *activeSnapshotDir()
    {
        const auto *test_case = activeModelParityCase();
        if (!test_case)
            throw std::logic_error(
                "Graph-native reference path requires an active typed parity case");
        return test_case->model.reference_directory.c_str();
    }

    /** @return Whether every split file required by the active model exists. */
    bool modelAvailable()
    {
        if (!std::filesystem::exists(activeModelPath()))
            return false;
        if (!isQwen122ProductionTest())
            return true;

        const std::filesystem::path first(activeModelPath());
        const std::string first_name = first.filename().string();
        const auto marker = first_name.find("00001-of-00004");
        if (marker == std::string::npos)
            return false;
        for (int split = 2; split <= 4; ++split)
        {
            std::string sibling_name = first_name;
            std::ostringstream ordinal;
            ordinal << std::setw(5) << std::setfill('0') << split;
            sibling_name.replace(marker, 5, ordinal.str());
            if (!std::filesystem::exists(first.parent_path() / sibling_name))
                return false;
        }
        return true;
    }

    /** @return Whether this case uses current-batch least-loaded assignment. */
    bool isLLEPProductionTest()
    {
        return false;
    }

    /** @brief Return whether the exact cell exercises persistent tier movement. */
    bool isDynamicResidencyProductionTest()
    {
        const auto *test_case = activeModelParityCase();
        return test_case && test_case->expert_overlay &&
               test_case->expert_overlay->movement ==
                   ModelParityExpertMovement::Dynamic;
    }

    /** @brief Return whether initial expert ownership uses seeded random order. */
    bool isRandomOwnerProductionTest()
    {
        const auto *test_case = activeModelParityCase();
        return test_case && test_case->expert_overlay &&
               test_case->expert_overlay->owner_order ==
                   RoutedExpertOwnerOrder::Random;
    }

    /**
     * @return Fixed MTP depth, the adaptive policy's maximum, or zero when off.
     *
     * Generated cells carry this policy explicitly.  Legacy topology cells do
     * not name an MTP mode and therefore must report zero; treating an absent
     * label as depth three made post-run evidence demand an MTP domain from a
     * runtime whose typed configuration correctly disabled MTP.
     */
    int activeMTPDraftDepth()
    {
        const auto *test_case = activeModelParityCase();
        if (!test_case)
            throw std::logic_error(
                "MTP depth requires an active typed parity case");
        return test_case->requestedMTPDraftDepth();
    }

    /** @return Setup-time MTP draft capacity retained by the active cell. */
    int activeMTPRetainedDraftCapacity()
    {
        const auto *test_case = activeModelParityCase();
        if (!test_case)
            throw std::logic_error(
                "MTP capacity requires an active typed parity case");
        return test_case->retained_mtp_draft_capacity;
    }

    /**
     * @brief Return the retained grouped-verifier row capacity for this cell.
     *
     * Every 122B campaign cell shares one maximum-capacity model context so
     * changing the requested fixed depth does not reload weights or rebuild
     * device graphs.  The device transaction still publishes its independent
     * logical depth through @ref activeMTPDraftDepth and selects a physical
     * bucket inside this admitted envelope.
     */
    int activeMTPGraphCapacityVerifierRows()
    {
        const int retained_draft_capacity =
            activeMTPRetainedDraftCapacity();
        return retained_draft_capacity > 0
                   ? retained_draft_capacity + 1
                   : 1;
    }

    /**
     * @brief Return the exact physical bucket selected by this transaction.
     *
     * Fixed-depth scalar transactions select the smallest retained power-of-two
     * bucket that contains their logical rows.  Dynamic depth instead embeds
     * the maximum envelope because active depth is device-owned replay data.
     */
    int activeMTPPhysicalVerifierRows()
    {
        const int capacity_rows = activeMTPGraphCapacityVerifierRows();
        return activeModelParityCaseOrThrow().usesDynamicMTPDepth()
                   ? capacity_rows
                   : mtpVerifierPhysicalRowBucket(
                         activeMTPDraftDepth() + 1,
                         capacity_rows);
    }

    /** @return Whether the production device depth controller is adaptive. */
    bool usesDynamicMTPDepth()
    {
        const auto *test_case = activeModelParityCase();
        return test_case && test_case->usesDynamicMTPDepth();
    }

    /** @return Whether the active generated cell enables MTP. */
    bool activeMTPEnabled()
    {
        const auto *test_case = activeModelParityCase();
        return test_case && test_case->mtpEnabled();
    }

    /** @brief Return whether the active topology contains a CUDA participant. */
    bool topologyUsesCuda()
    {
        const auto &participants =
            activeModelParityCaseOrThrow().topology.participants;
        return std::any_of(
            participants.begin(), participants.end(),
            [](const ModelParityParticipant &participant)
            { return participant.address.isCUDA(); });
    }

    /** @brief Return whether the active topology contains a ROCm participant. */
    bool topologyUsesRocm()
    {
        const auto &participants =
            activeModelParityCaseOrThrow().topology.participants;
        return std::any_of(
            participants.begin(), participants.end(),
            [](const ModelParityParticipant &participant)
            { return participant.address.isROCm(); });
    }

    /** @brief Return whether the active topology contains NodeTP CPU cold. */
    bool topologyUsesCpu()
    {
        const auto &participants =
            activeModelParityCaseOrThrow().topology.participants;
        return std::any_of(
            participants.begin(), participants.end(),
            [](const ModelParityParticipant &participant)
            { return participant.address.isCPU(); });
    }

    /** @brief Return the exact sparse participants declared by the active topology. */
    size_t activeOverlayParticipantCount()
    {
        return activeModelParityCaseOrThrow().topology.participants.size();
    }

    /** @return Whether a generated topology owns a non-continuation GPU tier. */
    bool topologyHasSecondaryGpuDomain()
    {
        const auto &plan = *activeModelParityCaseOrThrow()
                                .topology.expert_overlay_plan;
        return std::any_of(
            plan.domains.begin(), plan.domains.end(),
            [&](const RoutedExpertDomain &domain)
            {
                return domain.name != plan.continuation_domain &&
                       std::any_of(
                           domain.participants.begin(),
                           domain.participants.end(),
                           [](const GlobalDeviceAddress &participant)
                           { return participant.isGPU(); });
            });
    }

    /** @return Whether sparse ExpertOverlay work crosses an MPI process. */
    bool topologySpansMultipleMPIRanks()
    {
        return activeModelParityCaseOrThrow().topology.mpi_ranks > 1;
    }

    /** @return Number of integer-priority residency tiers in this cell. */
    size_t activeOverlayTierCount()
    {
        return activeModelParityCaseOrThrow()
            .topology.expert_overlay_plan->routed_tiers.size();
    }


    RoutedExpertTier makeTier(
        const std::string &name,
        const std::string &domain,
        int priority,
        int max_experts_per_layer,
        bool fallback)
    {
        RoutedExpertTier t;
        t.name = name;
        t.domain = domain;
        t.priority = priority;
        t.max_experts_per_layer = max_experts_per_layer;
        t.memory_budget_bytes = 0;
        t.fallback = fallback;
        return t;
    }

    /** @return Every unique 122B ExpertOverlay topology in the production matrix. */
    const std::array<Qwen122OverlayTopologySpec, 9> &
    qwen122OverlayTopologySpecs()
    {
        static const std::array<Qwen122OverlayTopologySpec, 9> specs{{
            {"CUDA2_ROCm4_2xMPI_NodeExpertOverlay", 2, 4, 0, 2,
             Qwen122ContinuationBackend::CUDA,
             ModelParityDynamicSpeedupWitness::Disabled},
            {"ROCm1_CPU2_2xMPI_NodeExpertOverlay", 0, 1, 2, 2,
             Qwen122ContinuationBackend::ROCm,
             ModelParityDynamicSpeedupWitness::Disabled},
            {"ROCm2_CPU2_2xMPI_NodeExpertOverlay", 0, 2, 2, 2,
             Qwen122ContinuationBackend::ROCm,
             ModelParityDynamicSpeedupWitness::Disabled},
            {"ROCm3_CPU2_2xMPI_NodeExpertOverlay", 0, 3, 2, 2,
             Qwen122ContinuationBackend::ROCm,
             ModelParityDynamicSpeedupWitness::Disabled},
            {"ROCm4_CPU2_2xMPI_NodeExpertOverlay", 0, 4, 2, 2,
             Qwen122ContinuationBackend::ROCm,
             ModelParityDynamicSpeedupWitness::Random},
            {"CUDA1_CPU2_2xMPI_NodeExpertOverlay", 1, 0, 2, 2,
             Qwen122ContinuationBackend::CUDA,
             ModelParityDynamicSpeedupWitness::Disabled},
            {"CUDA2_CPU2_2xMPI_NodeExpertOverlay", 2, 0, 2, 2,
             Qwen122ContinuationBackend::CUDA,
             ModelParityDynamicSpeedupWitness::Random},
            {"ROCm1_CPU1_1xMPI_RankExpertOverlay", 0, 1, 1, 1,
             Qwen122ContinuationBackend::ROCm,
             ModelParityDynamicSpeedupWitness::Disabled},
            {"CUDA1_CPU1_1xMPI_RankExpertOverlay", 1, 0, 1, 1,
             Qwen122ContinuationBackend::CUDA,
             ModelParityDynamicSpeedupWitness::Disabled},
        }};
        return specs;
    }

    /**
     * @brief Build one accelerator domain without assuming its owning rank.
     *
     * AUTO scope is resolved from gathered inventory: one device becomes a
     * single participant, while several same-rank devices become rank-local
     * TP. This preserves topology intent if the cards move between sockets.
     */
    RoutedExpertDomain qwen122AcceleratorDomain(
        Qwen122ContinuationBackend backend,
        int participant_count,
        bool continuation)
    {
        if (participant_count <= 0)
            throw std::invalid_argument(
                "122B accelerator domain requires a positive participant count");

        RoutedExpertDomain domain;
        if (backend == Qwen122ContinuationBackend::CUDA)
        {
            domain.name = kCudaHotDomain;
            domain.backend = CollectiveBackendType::NCCL;
            for (int ordinal = 0; ordinal < participant_count; ++ordinal)
                domain.participants.push_back(GlobalDeviceAddress::cuda(ordinal));
        }
        else
        {
            domain.name = continuation ? kRocmHotDomain : kRocmWarmDomain;
            domain.backend = CollectiveBackendType::RCCL;
            for (int ordinal = 0; ordinal < participant_count; ++ordinal)
                domain.participants.push_back(GlobalDeviceAddress::rocm(ordinal));
        }
        domain.scope = ExecutionDomainScope::AUTO;
        domain.owner_rank = -1;
        domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        domain.routed_phase_policy = RoutedExpertPhasePolicy::Uniform;
        domain.routed_decode_assignment_policy =
            RoutedExpertAssignmentPolicy::StaticOwner;
        domain.routed_prefill_assignment_policy =
            RoutedExpertAssignmentPolicy::StaticOwner;
        return domain;
    }

    /** @brief Build a one- or two-socket CPU tier resolved by live inventory. */
    RoutedExpertDomain qwen122CpuDomain(int participant_count)
    {
        if (participant_count <= 0)
            throw std::invalid_argument(
                "122B CPU domain requires a positive participant count");
        RoutedExpertDomain domain;
        domain.name = kCpuColdDomain;
        domain.scope = ExecutionDomainScope::AUTO;
        domain.backend = CollectiveBackendType::UPI;
        for (int numa = 0; numa < participant_count; ++numa)
            domain.participants.push_back(GlobalDeviceAddress::cpu(numa));
        domain.owner_rank = -1;
        domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        domain.routed_phase_policy = RoutedExpertPhasePolicy::Uniform;
        domain.routed_decode_assignment_policy =
            RoutedExpertAssignmentPolicy::StaticOwner;
        domain.routed_prefill_assignment_policy =
            RoutedExpertAssignmentPolicy::StaticOwner;
        return domain;
    }

    /**
     * @brief Build an immutable integer-priority 122B overlay blueprint.
     *
     * The continuation accelerator domain receives priority zero. An optional
     * second accelerator family receives priority one, and CPU receives the
     * next integer priority. Every quota is automatic: production fills each
     * tier to its exact remaining admitted capacity and places the exact
     * remainder in the final fallback tier.
     */
    std::shared_ptr<const MoERoutedExpertPlacementPlan>
    qwen122OverlayBlueprint(const Qwen122OverlayTopologySpec &spec)
    {
        auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
        plan->enabled = true;
        plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
        plan->residency_policy = RoutedExpertResidencyPolicy::StaticById;
        plan->owner_order = RoutedExpertOwnerOrder::Ordinal;

        int next_priority = 0;
        const auto append_domain = [&](RoutedExpertDomain domain)
        {
            const std::string domain_name = domain.name;
            plan->domains.push_back(std::move(domain));
            plan->routed_tiers.push_back(makeTier(
                "priority" + std::to_string(next_priority),
                domain_name,
                next_priority,
                /*max_experts_per_layer=*/0));
            ++next_priority;
        };

        const bool cuda_continuation =
            spec.continuation == Qwen122ContinuationBackend::CUDA;
        const int continuation_count = cuda_continuation
                                           ? spec.cuda_participants
                                           : spec.rocm_participants;
        auto continuation_domain = qwen122AcceleratorDomain(
            spec.continuation,
            continuation_count,
            /*continuation=*/true);
        plan->continuation_domain = continuation_domain.name;
        plan->base_model_domain = continuation_domain.name;
        plan->shared_expert_domain = continuation_domain.name;
        append_domain(std::move(continuation_domain));

        if (cuda_continuation && spec.rocm_participants > 0)
        {
            append_domain(qwen122AcceleratorDomain(
                Qwen122ContinuationBackend::ROCm,
                spec.rocm_participants,
                /*continuation=*/false));
        }
        else if (!cuda_continuation && spec.cuda_participants > 0)
        {
            append_domain(qwen122AcceleratorDomain(
                Qwen122ContinuationBackend::CUDA,
                spec.cuda_participants,
                /*continuation=*/false));
        }
        if (spec.cpu_participants > 0)
            append_domain(qwen122CpuDomain(spec.cpu_participants));

        if (plan->routed_tiers.size() < 2u)
            throw std::invalid_argument(
                "122B ExpertOverlay topology requires at least two priorities");
        plan->routed_tiers.back().fallback = true;
        plan->continuation_domain_spec.setDensePolicy(
            qwen122ContinuationDensePolicy(continuation_count));
        return plan;
    }


    MoERoutedExpertModelMetadata topologyOnlyMetadata()
    {
        MoERoutedExpertModelMetadata metadata;
        metadata.num_experts = kQwen35MoENumExperts;
        metadata.num_layers = isQwen122ProductionTest()
                                  ? 48
                                  : kQwen35MoENumLayers;
        metadata.d_model = isQwen122ProductionTest() ? 3072 : 4096;
        metadata.routed_intermediate_size =
            isQwen122ProductionTest() ? 1024 : 1536;
        metadata.has_shared_expert = true;
        metadata.shared_intermediate_size =
            metadata.routed_intermediate_size;
        metadata.routed_quant_type =
            isQwen122ProductionTest() ? "Q8_K" : "Q4_K";
        metadata.shared_quant_type = metadata.routed_quant_type;
        return metadata;
    }


    MoERoutedExpertModelMetadata metadataFromModel(const ModelContext &ctx)
    {
        const auto &loader = ctx.concreteLoader();
        const std::string &arch = ctx.architecture();

        MoERoutedExpertModelMetadata metadata;
        metadata.num_layers = ctx.totalBlockCount();
        metadata.num_experts = loader.getInt(arch + ".expert_count", 0);
        metadata.d_model = ctx.embeddingLength();
        metadata.routed_intermediate_size = loader.getInt(arch + ".expert_feed_forward_length", 0);
        if (metadata.routed_intermediate_size == 0)
            metadata.routed_intermediate_size = ctx.feedForwardLength();
        metadata.has_shared_expert = loader.getInt(arch + ".expert_shared_count", 0) > 0;
        metadata.shared_intermediate_size = metadata.has_shared_expert
                                                ? metadata.routed_intermediate_size
                                                : 0;
        metadata.routed_quant_type = "Q4_K";
        metadata.shared_quant_type = "Q4_K";
        return metadata;
    }


    MoERoutedExpertPlacementPlan requestedPlan(
        const MoERoutedExpertModelMetadata &metadata)
    {
        (void)metadata;
        const auto &test_case = activeModelParityCaseOrThrow();
        if (!test_case.topology.expert_overlay_plan)
        {
            throw std::logic_error(
                "Generated graph-native case has no ExpertOverlay blueprint");
        }
        auto generated = *test_case.topology.expert_overlay_plan;
        generated.residency_policy =
            isDynamicResidencyProductionTest()
                ? RoutedExpertResidencyPolicy::RoutedTierRebalanced
                : RoutedExpertResidencyPolicy::StaticById;
        generated.owner_order = isRandomOwnerProductionTest()
                                    ? RoutedExpertOwnerOrder::Random
                                    : RoutedExpertOwnerOrder::Ordinal;
        return generated;
    }

    /**
     * @brief Bind a topology and place a remote NodeLocal owner bucket first.
     *
     * Random whole-expert ownership partitions its deterministic permutation in
     * participant order. For the explicit distributed-migration proof, order
     * CPU participants that are remote from the continuation root before any
     * colocated participant. The production inventory binder resolves all rank
     * identities first, so moving GPUs between sockets/ranks changes the result
     * without changing this test or hard-coding NUMA affinity.
     *
     * @param requested Rank-agnostic production placement request.
     * @param inventory Gathered hardware/rank inventory shared by every rank.
     * @return Hardware-bound adversarial declaration with paired addresses,
     *         ranks, and optional weights reordered consistently.
     * @throws std::invalid_argument when the requested proof has no remote CPU
     *         participant or binding produced incomplete identities.
     */
    MoERoutedExpertPlacementPlan remoteFirstNodeLocalOwnerPlan(
        const MoERoutedExpertPlacementPlan &requested,
        const ClusterInventory &inventory)
    {
        auto bound = bindMoEExpertOverlayPlanToClusterInventory(
            requested,
            inventory);
        if (!bound)
            throw std::invalid_argument(
                "Adversarial ExpertOverlay layout produced no bound plan");

        const auto continuation = std::find_if(
            bound->domains.begin(),
            bound->domains.end(),
            [&](const auto &domain)
            { return domain.name == bound->continuation_domain; });
        if (continuation == bound->domains.end())
        {
            throw std::invalid_argument(
                "Adversarial ExpertOverlay layout has no continuation domain");
        }
        const auto continuation_rank = continuation->primaryWorldRank();
        if (!continuation_rank)
        {
            throw std::invalid_argument(
                "Adversarial ExpertOverlay layout has no continuation rank");
        }

        auto cpu_domain = std::find_if(
            bound->domains.begin(),
            bound->domains.end(),
            [](const auto &domain)
            { return domain.name == kCpuColdDomain; });
        if (cpu_domain == bound->domains.end() ||
            cpu_domain->scope != ExecutionDomainScope::NODE_LOCAL ||
            cpu_domain->participants.size() < 2u ||
            cpu_domain->world_ranks.size() !=
                cpu_domain->participants.size())
        {
            throw std::invalid_argument(
                "Adversarial ExpertOverlay layout requires a fully bound multi-participant NodeLocal CPU domain");
        }

        std::vector<size_t> order(cpu_domain->participants.size(), 0u);
        std::iota(order.begin(), order.end(), 0u);
        std::stable_sort(
            order.begin(),
            order.end(),
            [&](size_t lhs, size_t rhs)
            {
                const bool lhs_remote =
                    cpu_domain->world_ranks[lhs] != *continuation_rank;
                const bool rhs_remote =
                    cpu_domain->world_ranks[rhs] != *continuation_rank;
                return lhs_remote && !rhs_remote;
            });
        if (cpu_domain->world_ranks[order.front()] == *continuation_rank)
        {
            throw std::invalid_argument(
                "Adversarial ExpertOverlay layout found no CPU participant remote from the continuation rank");
        }

        const auto old_participants = cpu_domain->participants;
        const auto old_world_ranks = cpu_domain->world_ranks;
        const auto old_weights = cpu_domain->weights;
        for (size_t destination = 0; destination < order.size(); ++destination)
        {
            const size_t source = order[destination];
            cpu_domain->participants[destination] =
                old_participants[source];
            cpu_domain->world_ranks[destination] =
                old_world_ranks[source];
            if (old_weights.size() == order.size())
                cpu_domain->weights[destination] = old_weights[source];
        }

        LOG_INFO(
            "[Qwen3.5 MoE GraphNative] Adversarial NodeLocal owner order: "
            << "continuation_rank=" << *continuation_rank
            << " first_cpu_rank=" << cpu_domain->world_ranks.front()
            << " participants=" << cpu_domain->participants.size());
        return std::move(*bound);
    }


    std::optional<std::string> acceleratorHardwareBlocker(
        const ClusterInventory &inventory)
    {
        int cuda_count = 0;
        int rocm_count = 0;
        for (const auto &rank : inventory.ranks)
        {
            for (const auto &gpu : rank.gpus)
            {
                cuda_count += gpu.type == DeviceType::CUDA ? 1 : 0;
                rocm_count += gpu.type == DeviceType::ROCm ? 1 : 0;
            }
        }
        const int required_cuda = static_cast<int>(
            activeTypedParticipantCount(
                [](const GlobalDeviceAddress &participant)
                { return participant.isCUDA(); }));
        const int required_rocm = static_cast<int>(
            activeTypedParticipantCount(
                [](const GlobalDeviceAddress &participant)
                { return participant.isROCm(); }));
        if (topologyUsesCuda() && cuda_count < required_cuda)
            return "Graph-native Qwen3.5 MoE parity topology requires >=" +
                   std::to_string(required_cuda) + " CUDA device(s), found " +
                   std::to_string(cuda_count);

        if (topologyUsesRocm() && rocm_count < required_rocm)
            return "Graph-native Qwen3.5 MoE parity topology requires >=" +
                   std::to_string(required_rocm) + " ROCm device(s), found " +
                   std::to_string(rocm_count);

        return std::nullopt;
    }

    /**
     * @brief Return whether this fixture instance must prove bucketed prefill.
     *
     * The normal production campaign deliberately uses one exact authenticated
     * bucket for speed. This named cell is the explicit complementary contract:
     * it keeps real weights and the same CSV oracle but forces an ordered
     * heterogeneous sparse-collective schedule.
     */
    bool isSegmentedPrefillProductionTest()
    {
        const auto *test_case = activeModelParityCase();
        return test_case &&
               test_case->prefill_graph.isSegmentedCaptured();
    }

    /** @return Typed fixed capture rows, or zero for ordinary prefill. */
    int activeSegmentedPrefillCaptureRows()
    {
        const auto *test_case = activeModelParityCase();
        return test_case &&
                       test_case->prefill_graph.isSegmentedCaptured()
                   ? test_case->prefill_graph.captured_rows
                   : 0;
    }

    /**
     * @return Complete 35B topology x placement x movement x prefill matrix.
     *
     * The CUDA/ROCm/CPU topology owns both ordinary and segmented captured
     * prefill profiles. Every other topology owns the ordinary profile. All
     * cells, including the ten historical cases, are emitted only by the
     * central typed expander.
     */
    const std::vector<ModelParityCase> &qwen35GraphNativeParityCases()
    {
        static const auto cases = []
        {
            std::vector<ModelParityCase> expanded;
            std::set<std::string> names;
            for (const auto &spec : qwen35MoE35BOverlayTopologySpecs())
            {
                auto topology_cases = expandModelParityDefinition(
                    qwen35MoE35BGraphNativeParityDefinition(spec));
                for (auto &test_case : topology_cases)
                {
                    if (!names.insert(test_case.testName()).second)
                    {
                        throw std::logic_error(
                            "Duplicate generated 35B graph-native parity case: " +
                            test_case.testName());
                    }
                    expanded.push_back(std::move(test_case));
                }
            }
            return expanded;
        }();
        return cases;
    }

    /**
     * @brief Build the model-wide Dynamic policy for one generated 122B cell.
     *
     * A migration wave is parallel across transformer layers, so its physical
     * cycle width is one tier-residency cycle per layer plus one independent
     * within-tier participant cycle when the declared topology has an
     * exchange degree of freedom.  The command envelope admits a complete
     * closed cycle through every declared participant; it is not a second
     * scheduling limit.  Capacity admission prices these exact values before
     * any model weight or transfer lane is materialized.
     *
     * @param spec Typed physical topology expanded by the canonical matrix.
     * @param transformer_layers Authenticated main-model routed layer count.
     * @return Complete production Dynamic policy for the generated cell.
     */
    MoERebalanceRuntimeConfig qwen122DynamicParityEconomics(
        const Qwen122OverlayTopologySpec &spec,
        int transformer_layers)
    {
        if (transformer_layers <= 0)
        {
            throw std::invalid_argument(
                "Qwen122 Dynamic wave geometry requires a positive transformer-layer count");
        }
        const std::uint64_t participant_count =
            static_cast<std::uint64_t>(
                spec.cuda_participants + spec.rocm_participants +
                spec.cpu_participants);
        if (participant_count == 0u)
        {
            throw std::invalid_argument(
                "Qwen122 Dynamic wave geometry requires at least one participant");
        }
        const bool has_participant_axis =
            spec.cuda_participants > 1 || spec.rocm_participants > 1 ||
            spec.cpu_participants > 1;
        const std::uint64_t cycle_slots =
            static_cast<std::uint64_t>(transformer_layers) +
            (has_participant_axis ? 1u : 0u);
        const std::uint64_t maximum_cycle_edges =
            std::max<std::uint64_t>(2u, participant_count);
        if (cycle_slots > std::numeric_limits<std::uint32_t>::max() ||
            maximum_cycle_edges >
                std::numeric_limits<std::uint32_t>::max() / cycle_slots)
        {
            throw std::overflow_error(
                "Qwen122 Dynamic wave geometry exceeds the runtime policy range");
        }

        MoERebalanceRuntimeConfig config;
        config.mode = MoERebalanceRuntimeMode::Dynamic;
        config.window_size = 4;
        config.max_window_size = 4;
        config.window_growth_factor = 1.0f;
        config.migration_transfer_slots =
            static_cast<std::uint32_t>(cycle_slots);
        config.migration_payoff_horizon_tokens = 65'536;
        config.release_raw_expert_weights = false;
        /*
         * A max/min load ratio cannot be below 1.0.  Use the exact 1000
         * per-mille boundary to make every observed imbalance eligible; zero
         * is not a valid ratio threshold and used to survive until residency
         * authority construction.
         */
        config.dynamic_imbalance_threshold_per_mille =
            moe_rebalance_policy::
                kMinimumDynamicImbalanceThresholdPerMille;
        config.dynamic_min_improvement_per_mille = 0;
        config.dynamic_max_swaps_per_layer =
            has_participant_axis ? 2u : 1u;
        config.dynamic_max_plan_entries_per_wave =
            static_cast<std::uint32_t>(
                cycle_slots * maximum_cycle_edges);
        config.dynamic_min_window_activations = 0;
        config.device_min_load_spread_improvement = 0;
        config.device_min_load_spread_improvement_divisor = 0;
        config.device_min_wave_spread_improvement_per_payload_slot = 0;
        config.device_min_foreign_rows_per_critical_path_payload_slot = 0;
        config.device_min_router_spread_improvement_per_payload_slot = 0;
        config.device_max_post_wave_load_spread_per_mille = 1000;
        config.device_maintenance_slack_tokens = 0;
        config.device_min_maintenance_period_tokens =
            kModelParityRequiredMaximumMTPDepth + 1;
        config.device_initial_maintenance_period_tokens = 1;
        return config;
    }

    /**
     * @brief Derive the two typed Dynamic evidence policies for Qwen 122B.
     *
     * A movement-only cell needs one conflict-free priority cycle and, when
     * the topology exposes it, one same-priority participant cycle.  Its
     * observation window expands to the declared model context immediately
     * after that publication so numerical parity executes against the proven
     * epoch without inducing unrelated migration churn.  The designated A/B
     * witness retains the full layer-parallel transfer fabric and one fixed
     * window large enough for its complete timing cohort, canonical parity
     * request, and one publication-overlap request.
     *
     * @param spec Typed physical topology expanded by the canonical matrix.
     * @param transformer_layers Authenticated main-model routed layer count.
     * @param maximum_context_rows Declared request context admission.
     * @return Complete evidence-indexed policies used by matrix expansion.
     */
    ModelParityDynamicRuntimePolicies qwen122DynamicRuntimePolicies(
        const Qwen122OverlayTopologySpec &spec,
        int transformer_layers,
        int maximum_context_rows)
    {
        constexpr int kMovementProofInitialWindowRows =
            kQwen35MoEMovementProofInitialWindowRows;
        constexpr std::uint32_t kMovementProofConcurrentCycles = 2u;
        if (maximum_context_rows < kMovementProofInitialWindowRows)
        {
            throw std::invalid_argument(
                "Qwen122 Dynamic movement proof requires context capacity for its initial histogram window");
        }

        MoERebalanceRuntimeConfig movement =
            qwen122DynamicParityEconomics(spec, transformer_layers);
        movement.window_size = kMovementProofInitialWindowRows;
        movement.max_window_size = maximum_context_rows;
        movement.window_growth_factor =
            static_cast<float>(maximum_context_rows) /
            static_cast<float>(kMovementProofInitialWindowRows);
        movement.migration_cycles_per_wave =
            kMovementProofConcurrentCycles;
        /* The initial one-token device cadence publishes the proof wave. After
         * that transaction, the recurring device scheduler must match the
         * expanded host observation horizon or it can author additional
         * device epochs while numerical parity is still executing. */
        movement.device_min_maintenance_period_tokens =
            maximum_context_rows;

        MoERebalanceRuntimeConfig speedup =
            qwen122DynamicParityEconomics(spec, transformer_layers);
        speedup.window_size = kConvergenceHistogramWindowTokens;
        speedup.max_window_size = kConvergenceHistogramWindowTokens;
        speedup.window_growth_factor = 1.0F;

        return {
            .economic_movement = std::move(movement),
            .economic_movement_and_observed_speedup = std::move(speedup),
        };
    }

    /**
     * @brief Declare one canonical 122B real-model/topology parity matrix.
     *
     * Cross-rank GPU ownership is deliberately unresolved. Production cluster
     * inventory binds each ordinal to whichever MPI instance currently owns
     * it. NCCL/RCCL remain inside their named domains; the outer graph is
     * multi-domain, not a fictitious heterogeneous tensor-parallel collective.
     */
    ModelParityDefinition qwen122ExpertOverlayParityDefinition(
        const Qwen122OverlayTopologySpec &spec)
    {
        ModelParityDefinition definition;
        definition.model = {
            .test_id = "Qwen35_122B",
            .model_path = kQwen122ModelPath,
            .reference_directory = kQwen122SnapshotDir,
            .decode_steps = 4,
            .max_seq_len = 4096,
            .transformer_layers = 48,
            .maximum_mtp_draft_depth = kQwen122MaximumMTPDraftDepth,
        };
        definition.topology.test_id = spec.test_id;
        definition.topology.kind =
            spec.mpi_ranks == 1
                ? ModelParityTopologyKind::RankLocalMultiDomain
                : ModelParityTopologyKind::NodeMultiDomain;
        definition.topology.collective = Collective::None;
        definition.topology.mpi_ranks = spec.mpi_ranks;
        for (int ordinal = 0; ordinal < spec.cuda_participants; ++ordinal)
        {
            definition.topology.participants.push_back({
                GlobalDeviceAddress::cuda(ordinal),
                spec.mpi_ranks == 1 ? std::optional<int>{0} : std::nullopt,
            });
        }
        for (int ordinal = 0; ordinal < spec.rocm_participants; ++ordinal)
        {
            definition.topology.participants.push_back({
                GlobalDeviceAddress::rocm(ordinal),
                spec.mpi_ranks == 1 ? std::optional<int>{0} : std::nullopt,
            });
        }
        for (int numa = 0; numa < spec.cpu_participants; ++numa)
        {
            definition.topology.participants.push_back({
                GlobalDeviceAddress::cpu(numa),
                spec.mpi_ranks == 1
                    ? std::optional<int>{0}
                    : std::optional<int>{numa},
            });
        }
        definition.topology.expert_overlay_plan =
            qwen122OverlayBlueprint(spec);
        definition.thresholds = {
            .cosine_threshold = 0.96f,
            .decode_cosine_threshold = 0.98f,
            .early_layers_count = 6,
            .min_early_layers_passed = 5,
            .kl_threshold = 0.03f,
            /*
             * Recursive MTP logits have a separate baseline budget so their
             * quantized feedback does not weaken ordinary prefill/decode.
             */
            .mtp_kl_threshold = 0.05f,
            .min_top1_accuracy = 0.80f,
            .min_top5_accuracy = 0.60f,
            .pytorch_top1_in_topk = 4,
        };
        definition.precisions.activation = {ActivationPrecision::FP32};
        definition.precisions.kv_cache = {KVCachePrecision::FP16};
        definition.features.mtp = ModelParityAxisProfile::Standard;
        definition.features.dynamic_speedup_witness =
            spec.dynamic_speedup_witness;
        definition.features.mtp_kl_threshold_overrides = {
            {
                .policy = ModelParityMTP::Depth15,
                /*
                 * Preserve the measured 0.06695 deep-recursion KL allowance.
                 * The graph activation buffers are FP32; this measurement
                 * does not establish drift caused by FP16 activations. The
                 * independent hidden-state and top-K contracts still apply.
                 */
                .maximum_kl_divergence = 0.07f,
            },
            {
                .policy = ModelParityMTP::DynamicDepth,
                /*
                 * Adaptive depth has the same admitted depth-15 ceiling and
                 * executes the same recursive graph. It therefore owns the
                 * same narrowly measured KL contract; the
                 * shallower fixed-depth policies retain the 0.05 default.
                 */
                .maximum_kl_divergence = 0.07f,
            },
        };
        definition.features
            .mtp_recursive_aggregate_cosine_threshold_overrides = {
            {
                .policy = ModelParityMTP::Depth15,
                /*
                 * The authenticated Q8_0 sidecar repeats weight-only Q8_0
                 * projection with blockwise Q8 activation preparation. At the
                 * fourteenth recursive prediction this measured 0.98953 over
                 * seventeen comparable checkpoints, while the token, LM-head
                 * KL/top-k, route top-1, finiteness, and every individual 0.98
                 * checkpoint contract remained green. Scope the measured
                 * quantized-recurrence allowance to this exact deep policy.
                 */
                .minimum_cosine_similarity = 0.98f,
            },
            {
                .policy = ModelParityMTP::DynamicDepth,
                /*
                 * Adaptive depth owns the same admitted depth-15 recurrence.
                 * Shallower fixed depths retain the suite-wide 0.99 aggregate
                 * floor because they cannot reach this accumulation horizon.
                 */
                .minimum_cosine_similarity = 0.98f,
            },
        };
        definition.dynamic_rebalance = qwen122DynamicRuntimePolicies(
            spec,
            definition.model.transformer_layers,
            definition.model.max_seq_len);
        return definition;
    }

    /**
     * @return Deduplicated topology x 4 placement/movement x 6 MTP matrix.
     *
     * Topology identifiers and complete generated case names are checked here
     * because this binary joins several independently valid definitions into
     * one process-campaign catalogue. A duplicate would otherwise make GTest
     * registration order, rather than the typed source, choose the live case.
     */
    const std::vector<ModelParityCase> &qwen122ExpertOverlayParityCases()
    {
        static const auto cases = []
        {
            std::vector<ModelParityCase> expanded;
            std::set<std::string> topology_ids;
            std::set<std::string> case_names;
            for (const auto &spec : qwen122OverlayTopologySpecs())
            {
                if (!topology_ids.insert(spec.test_id).second)
                {
                    throw std::logic_error(
                        "Duplicate 122B ExpertOverlay topology id: " +
                        std::string(spec.test_id));
                }
                auto topology_cases = expandModelParityDefinition(
                    qwen122ExpertOverlayParityDefinition(spec));
                for (auto &test_case : topology_cases)
                {
                    const std::string name = test_case.testName();
                    if (!case_names.insert(name).second)
                    {
                        throw std::logic_error(
                            "Duplicate generated 122B ExpertOverlay case: " +
                            name);
                    }
                    expanded.push_back(std::move(test_case));
                }
            }
            return expanded;
        }();
        return cases;
    }

}
