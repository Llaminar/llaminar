/**
 * @file Test__Qwen35MoE_GraphNative_CudaHotRocmWarmCpuCold_Parity.cpp
 * @brief Production-path PyTorch parity gates for rank-agnostic Qwen3.5 MoE
 *        graph-native expert-overlay topologies.
 *
 * This fixture drives real Qwen3.5-35B-A3B weights through two production
 * OrchestrationRunner instances. It validates the same sparse-collective graph
 * shape used by serving for CUDA/ROCm/CPU three-tier and two-tier cells against
 * the Python/Hugging Face checkpoint corpus. The fixture owns only test
 * admission and evidence collection. Dynamic CPU-tier cells use the production
 * inventory binder plus authenticated reference routing to declare a
 * rank-relative adversarial initial layout. Production still validates that
 * layout and owns graph construction, live histogram generation, placement
 * planning, transfer ordering, migration economics, and token sampling.
 *
 * @author David Sanftenberg
 * @date August 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>

#include "Qwen35MoEParityTestBase.h"
#include "backends/ComputeBackend.h"
#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"
#include "config/OrchestrationConfig.h"
#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "execution/moe/MoEExpertOwnerMap.h"
#include "execution/moe/MoEOverlayResidencyAuthority.h"
#include "execution/moe/MoERoutedExpertPlacementPlanner.h"
#include "execution/moe/RoutedExpertOwnerAssignment.h"
#include "execution/prefix_cache/PrefixCacheStateProbe.h"
#include "execution/runner/OrchestrationRunner.h"
#include "planning/ClusterInventoryGatherer.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"
#include "utils/Sampler.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen35moe;

namespace
{
    constexpr const char *kModelPath = "/opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf";
    constexpr const char *kSnapshotDir = "pytorch_qwen35_moe_snapshots";
    constexpr const char *kCudaHotDomain = "cuda_hot";
    constexpr const char *kRocmHotDomain = "rocm_hot";
    constexpr const char *kRocmWarmDomain = "rocm_warm";
    constexpr const char *kCpuColdDomain = "cpu_cold";
    constexpr const char *kLegacyEnvVar = "LLAMINAR_MOE_LEGACY_OVERLAY_DOMAIN_RUNTIME";
    constexpr const char *kDynamicCudaRocmTest =
        "ProductionParity_DynamicRandom_CUDA_ROCm";
    constexpr const char *kDynamicCudaCpuTest =
        "ProductionParity_DynamicRandom_CUDA_CPU";
    constexpr const char *kDynamicRocmCpuTest =
        "ProductionParity_DynamicRandom_ROCm_CPU";
    constexpr const char *kDynamicCudaRocmCpuTest =
        "ProductionParity_DynamicRandom_CUDA_ROCm_CPU";
    constexpr const char *kStaticRandomCudaRocmTest =
        "ProductionParity_StaticRandom_CUDA_ROCm";

    // Production owner-map IDs are dense in declarative domain order. The
    // participant counts include both independent NodeTP CPU sockets.
    constexpr size_t kOverlayParticipantCount = 4;
    constexpr size_t kCudaCpuOverlayParticipantCount = 3;
    constexpr size_t kRocmCpuOverlayParticipantCount = 3;
    constexpr size_t kCudaRocmOverlayParticipantCount = 2;

    /**
     * @brief Production topology selected by the exact parity matrix cell.
     *
     * The enum keeps backend intent out of loosely coupled boolean flags. Each
     * value maps to one complete domain/tier declaration that production binds
     * to the gathered cluster inventory.
     */
    enum class OverlayTopology
    {
        CudaRocmCpu,
        CudaCpu,
        RocmCpu,
        CudaRocm,
    };

    // The authenticated Hugging Face corpus currently contains nine prefill
    // tokens. A four-row fixed bucket therefore executes the real production
    // schedule [4, 4, 1] and exercises both replay and padded final segments.
    constexpr int kSegmentedPrefillCaptureRows = 4;

    // Approximate Qwen3.5-35B-A3B metadata for topology-only, model-free planning.
    constexpr int kQwen35MoENumExperts = 256;
    constexpr int kQwen35MoENumLayers = 94;

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

    bool modelAvailable()
    {
        return std::filesystem::exists(kModelPath);
    }

    /** @brief Return whether the exact cell exercises persistent tier movement. */
    bool isDynamicResidencyProductionTest()
    {
        const auto *info =
            ::testing::UnitTest::GetInstance()->current_test_info();
        if (!info)
            return false;
        const std::string name = info->name();
        return name == kDynamicCudaRocmTest ||
               name == kDynamicCudaCpuTest ||
               name == kDynamicRocmCpuTest ||
               name == kDynamicCudaRocmCpuTest;
    }

    /** @brief Return whether initial expert ownership uses seeded random order. */
    bool isRandomOwnerProductionTest()
    {
        const auto *info =
            ::testing::UnitTest::GetInstance()->current_test_info();
        if (!info)
            return false;
        const std::string name = info->name();
        return isDynamicResidencyProductionTest() ||
               name == kStaticRandomCudaRocmTest;
    }

    /**
     * @brief Return the topology declared by the active exact GTest cell.
     *
     * Test selection changes only the request supplied to OrchestrationRunner.
     * Exact-name matching prevents a missing backend from silently selecting a
     * smaller topology and keeps campaign resource identity auditable.
     */
    OverlayTopology activeTopology()
    {
        const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
        if (!info)
            return OverlayTopology::CudaRocmCpu;

        const std::string name = info->name();
        if (name == "ProductionParity_CUDA_CPU" ||
            name == kDynamicCudaCpuTest)
        {
            return OverlayTopology::CudaCpu;
        }
        if (name == "ProductionParity_ROCm_CPU" ||
            name == kDynamicRocmCpuTest)
            return OverlayTopology::RocmCpu;
        if (name == "ProductionParity_CUDA_ROCm" ||
            name == kDynamicCudaRocmTest ||
            name == kStaticRandomCudaRocmTest)
            return OverlayTopology::CudaRocm;
        return OverlayTopology::CudaRocmCpu;
    }

    /** @brief Return whether the active topology contains a CUDA participant. */
    bool topologyUsesCuda()
    {
        const auto topology = activeTopology();
        return topology == OverlayTopology::CudaRocmCpu ||
               topology == OverlayTopology::CudaCpu ||
               topology == OverlayTopology::CudaRocm;
    }

    /** @brief Return whether the active topology contains a ROCm participant. */
    bool topologyUsesRocm()
    {
        const auto topology = activeTopology();
        return topology == OverlayTopology::CudaRocmCpu ||
               topology == OverlayTopology::RocmCpu ||
               topology == OverlayTopology::CudaRocm;
    }

    /** @brief Return whether the active topology contains NodeTP CPU cold. */
    bool topologyUsesCpu()
    {
        return activeTopology() != OverlayTopology::CudaRocm;
    }

    /** @brief Return the exact sparse participants declared by the active topology. */
    size_t activeOverlayParticipantCount()
    {
        switch (activeTopology())
        {
        case OverlayTopology::CudaRocmCpu:
            return kOverlayParticipantCount;
        case OverlayTopology::CudaCpu:
            return kCudaCpuOverlayParticipantCount;
        case OverlayTopology::RocmCpu:
            return kRocmCpuOverlayParticipantCount;
        case OverlayTopology::CudaRocm:
            return kCudaRocmOverlayParticipantCount;
        }
        throw std::logic_error("Unhandled Qwen3.5 MoE overlay topology");
    }

    RoutedExpertDomain cudaHotDomain()
    {
        RoutedExpertDomain domain;
        domain.name = kCudaHotDomain;
        domain.scope = ExecutionDomainScope::SINGLE;
        domain.backend = CollectiveBackendType::NCCL;
        domain.participants = {GlobalDeviceAddress::cuda(0)};
        domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        return domain;
    }

    RoutedExpertDomain rocmWarmDomain()
    {
        RoutedExpertDomain domain;
        domain.name = kRocmWarmDomain;
        domain.scope = ExecutionDomainScope::SINGLE;
        domain.backend = CollectiveBackendType::RCCL;
        domain.participants = {GlobalDeviceAddress::rocm(0)};
        domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        return domain;
    }

    /** @brief Declare the ROCm continuation domain for the ROCm/CPU cell. */
    RoutedExpertDomain rocmHotDomain()
    {
        auto domain = rocmWarmDomain();
        domain.name = kRocmHotDomain;
        return domain;
    }

    RoutedExpertDomain cpuColdDomain()
    {
        RoutedExpertDomain domain;
        domain.name = kCpuColdDomain;
        domain.scope = ExecutionDomainScope::NODE_LOCAL;
        domain.backend = CollectiveBackendType::UPI;
        domain.participants = {
            GlobalDeviceAddress::cpu(0),
            GlobalDeviceAddress::cpu(1),
        };
        domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        return domain;
    }

    RoutedExpertTier makeTier(
        const std::string &name,
        const std::string &domain,
        int priority,
        int max_experts_per_layer,
        bool fallback = false)
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

    MoERoutedExpertModelMetadata topologyOnlyMetadata()
    {
        MoERoutedExpertModelMetadata metadata;
        metadata.num_experts = kQwen35MoENumExperts;
        metadata.num_layers = kQwen35MoENumLayers;
        metadata.d_model = 4096;
        metadata.routed_intermediate_size = 1536;
        metadata.has_shared_expert = true;
        metadata.shared_intermediate_size = 1536;
        metadata.routed_quant_type = "Q4_K";
        metadata.shared_quant_type = "Q4_K";
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

    MoERoutedExpertPlacementPlan requestedPlan(const MoERoutedExpertModelMetadata &metadata)
    {
        const int cuda_capacity = std::max(1, metadata.num_experts / 2);
        const int half_capacity = std::max(1, metadata.num_experts / 2);

        MoERoutedExpertPlacementPlan plan;
        plan.enabled = true;
        plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
        plan.residency_policy = isDynamicResidencyProductionTest()
                                    ? RoutedExpertResidencyPolicy::RoutedTierRebalanced
                                    : RoutedExpertResidencyPolicy::StaticById;
        plan.owner_order = isRandomOwnerProductionTest()
                               ? RoutedExpertOwnerOrder::Random
                               : RoutedExpertOwnerOrder::Ordinal;

        switch (activeTopology())
        {
        case OverlayTopology::CudaCpu:
            plan.continuation_domain = kCudaHotDomain;
            plan.shared_expert_domain = kCudaHotDomain;
            plan.domains = {
                cudaHotDomain(),
                cpuColdDomain(),
            };
            plan.routed_tiers = {
                makeTier(
                    "cuda_priority",
                    kCudaHotDomain,
                    isDynamicResidencyProductionTest() ? -20 : 0,
                    half_capacity),
                makeTier(
                    "cpu_priority",
                    kCpuColdDomain,
                    isDynamicResidencyProductionTest() ? 17 : 1,
                    0,
                    /*fallback=*/true),
            };
            return plan;

        case OverlayTopology::RocmCpu:
            plan.continuation_domain = kRocmHotDomain;
            plan.shared_expert_domain = kRocmHotDomain;
            plan.domains = {
                rocmHotDomain(),
                cpuColdDomain(),
            };
            plan.routed_tiers = {
                makeTier(
                    "rocm_priority",
                    kRocmHotDomain,
                    isDynamicResidencyProductionTest() ? -20 : 0,
                    half_capacity),
                makeTier(
                    "cpu_priority",
                    kCpuColdDomain,
                    isDynamicResidencyProductionTest() ? 17 : 1,
                    0,
                    /*fallback=*/true),
            };
            return plan;

        case OverlayTopology::CudaRocm:
            plan.continuation_domain = kCudaHotDomain;
            plan.shared_expert_domain = kCudaHotDomain;
            plan.domains = {
                cudaHotDomain(),
                rocmWarmDomain(),
            };
            plan.routed_tiers = {
                makeTier(
                    "hot",
                    kCudaHotDomain,
                    isDynamicResidencyProductionTest() ? -20 : 0,
                    half_capacity),
                makeTier(
                    "warm",
                    kRocmWarmDomain,
                    isDynamicResidencyProductionTest() ? 17 : 1,
                    std::max(1, metadata.num_experts - half_capacity),
                    /*fallback=*/true),
            };
            return plan;

        case OverlayTopology::CudaRocmCpu:
            break;
        }

        int rocm_capacity = std::max(1, metadata.num_experts / 4);
        if (metadata.num_experts >= 3 && cuda_capacity + rocm_capacity >= metadata.num_experts)
            rocm_capacity = std::max(1, metadata.num_experts - cuda_capacity - 1);

        plan.continuation_domain = kCudaHotDomain;
        plan.shared_expert_domain = kCudaHotDomain;
        plan.domains = {
            cudaHotDomain(),
            rocmWarmDomain(),
            cpuColdDomain(),
        };
        plan.routed_tiers = {
            makeTier(
                "cuda_priority",
                kCudaHotDomain,
                isDynamicResidencyProductionTest() ? -20 : 0,
                cuda_capacity),
            makeTier(
                "rocm_priority",
                kRocmWarmDomain,
                isDynamicResidencyProductionTest() ? 7 : 1,
                rocm_capacity),
            makeTier(
                "cpu_priority",
                kCpuColdDomain,
                isDynamicResidencyProductionTest() ? 41 : 2,
                0,
                /*fallback=*/true),
        };
        return plan;
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
        const int continuation_rank =
            !continuation->world_ranks.empty()
                ? continuation->world_ranks.front()
                : continuation->owner_rank;
        if (continuation_rank < 0)
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
                    cpu_domain->world_ranks[lhs] != continuation_rank;
                const bool rhs_remote =
                    cpu_domain->world_ranks[rhs] != continuation_rank;
                return lhs_remote && !rhs_remote;
            });
        if (cpu_domain->world_ranks[order.front()] == continuation_rank)
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
            << "continuation_rank=" << continuation_rank
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
        if (topologyUsesCuda() && cuda_count < 1)
            return "Graph-native Qwen3.5 MoE parity topology requires >=1 CUDA device, found " +
                   std::to_string(cuda_count);

        if (topologyUsesRocm() && rocm_count < 1)
            return "Graph-native Qwen3.5 MoE parity topology requires >=1 ROCm device, found " +
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
        const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
        return info &&
               std::string(info->name()) ==
                   "ProductionParity_SegmentedPrefill_CUDA_ROCm_CPU";
    }

    /** @brief Build parity metadata whose device list matches one topology. */
    TestConfig makeGraphNativeTestConfig(OverlayTopology topology)
    {
        TestConfig config;
        switch (topology)
        {
        case OverlayTopology::CudaRocmCpu:
            config.name = isDynamicResidencyProductionTest()
                              ? "GraphNative_DynamicRandom_Cuda_Rocm_Cpu"
                              : "GraphNative_CudaHot_RocmWarm_CpuCold";
            config.devices = {
                ParityDeviceType::CUDA,
                ParityDeviceType::ROCm,
                ParityDeviceType::CPU,
            };
            break;
        case OverlayTopology::CudaCpu:
            config.name = isDynamicResidencyProductionTest()
                              ? "GraphNative_DynamicRandom_Cuda_Cpu"
                              : "GraphNative_CudaHot_CpuCold";
            config.devices = {
                ParityDeviceType::CUDA,
                ParityDeviceType::CPU,
            };
            break;
        case OverlayTopology::RocmCpu:
            config.name = isDynamicResidencyProductionTest()
                              ? "GraphNative_DynamicRandom_Rocm_Cpu"
                              : "GraphNative_RocmHot_CpuCold";
            config.devices = {
                ParityDeviceType::ROCm,
                ParityDeviceType::CPU,
            };
            break;
        case OverlayTopology::CudaRocm:
            config.name = isDynamicResidencyProductionTest()
                              ? "GraphNative_DynamicRandom_Cuda_Rocm"
                              : (isRandomOwnerProductionTest()
                                     ? "GraphNative_StaticRandom_Cuda_Rocm"
                                     : "GraphNative_CudaHot_RocmWarm");
            config.devices = {
                ParityDeviceType::CUDA,
                ParityDeviceType::ROCm,
            };
            break;
        }
        config.parallelism = Parallelism::None;
        config.collective = Collective::None;
        config.thresholds = {
            .cosine_threshold = 0.90f,
            .decode_cosine_threshold = 0.80f,
            .early_layers_count = 6,
            .min_early_layers_passed = 5,
            .kl_threshold = 0.05f,
            .min_top1_accuracy = 0.80f,
            .min_top5_accuracy = 0.60f,
            .pytorch_top1_in_topk = 4,
        };
        config.mpi_ranks = 2;
        config.model_path = kModelPath;
        config.snapshot_dir = kSnapshotDir;
        config.activation_precision = ActivationPrecision::FP32;
        config.kv_cache_precision = KVCachePrecision::FP16;
        return config;
    }

} // namespace

class Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold
    : public Qwen35MoEConfigDrivenParityTest<Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold>
{
public:
    static const TestConfig &staticConfig()
    {
        static TestConfig kConfig =
            makeGraphNativeTestConfig(OverlayTopology::CudaRocmCpu);
        return kConfig;
    }

    static const TestConfig &cudaHotCpuColdConfig()
    {
        static TestConfig kConfig =
            makeGraphNativeTestConfig(OverlayTopology::CudaCpu);
        return kConfig;
    }

    /** @brief Return the ROCm-hot/NodeTP-CPU-cold test contract. */
    static const TestConfig &rocmHotCpuColdConfig()
    {
        static TestConfig kConfig =
            makeGraphNativeTestConfig(OverlayTopology::RocmCpu);
        return kConfig;
    }

    /** @brief Return the CUDA-hot/ROCm-warm all-GPU test contract. */
    static const TestConfig &cudaHotRocmWarmConfig()
    {
        static TestConfig kConfig =
            makeGraphNativeTestConfig(OverlayTopology::CudaRocm);
        return kConfig;
    }

    /**
     * @brief Return the configuration whose declared backends match this exact cell.
     *
     * The production campaign discovery derives its resource signature from
     * the test name; this return value keeps the test fixture's result labels,
     * reference evidence, and runner configuration aligned with that same
     * declarative topology.
     */
    const TestConfig &getTestConfig() const
    {
        switch (activeTopology())
        {
        case OverlayTopology::CudaRocmCpu:
            return staticConfig();
        case OverlayTopology::CudaCpu:
            return cudaHotCpuColdConfig();
        case OverlayTopology::RocmCpu:
            return rocmHotCpuColdConfig();
        case OverlayTopology::CudaRocm:
            return cudaHotRocmWarmConfig();
        }
        throw std::logic_error("Unhandled Qwen3.5 MoE parity topology config");
    }

protected:
    using Base = Qwen35MoEConfigDrivenParityTest<Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold>;

    /**
     * @brief Allocation-owning samples for the observed convergence gate.
     *
     * Baseline values are ordinary production intervals claimed by the
     * economy controller in its `Baseline` state, before any residency epoch
     * can publish. Converged values use the identical authenticated request
     * after at least two profitable epochs. A sample whose surrounding
     * committed-wave count changes is discarded instead of being attributed
     * to either layout.
     */
    struct ResidencyConvergenceTimings
    {
        std::vector<std::uint64_t> baseline_prefill_ns;
        std::vector<std::uint64_t> baseline_decode_ns;
        std::vector<std::uint64_t> converged_prefill_ns;
        std::vector<std::uint64_t> converged_decode_ns;
        std::vector<std::uint64_t> converged_prefill_epochs;
        std::vector<std::uint64_t> converged_decode_epochs;
    };

    /** @return Monotonic elapsed nanoseconds, clamped away from zero. */
    static std::uint64_t elapsedNanoseconds(
        std::chrono::steady_clock::time_point start) noexcept
    {
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start)
                .count();
        return static_cast<std::uint64_t>(std::max<std::int64_t>(elapsed, 1));
    }

    /** @return Median of a non-empty timing corpus without changing it. */
    static std::uint64_t medianNanoseconds(
        const std::vector<std::uint64_t> &samples)
    {
        if (samples.empty())
            throw std::invalid_argument("Cannot take the median of no inference samples");
        auto ordered = samples;
        const std::size_t midpoint = ordered.size() / 2u;
        std::nth_element(
            ordered.begin(),
            ordered.begin() + static_cast<std::ptrdiff_t>(midpoint),
            ordered.end());
        if ((ordered.size() & 1u) != 0u)
            return ordered[midpoint];
        const auto lower = *std::max_element(
            ordered.begin(),
            ordered.begin() + static_cast<std::ptrdiff_t>(midpoint));
        return lower + (ordered[midpoint] - lower) / 2u;
    }

    /** @return Whether this cell owns an authenticated adversarial CPU tier. */
    bool requiresObservedConvergenceSpeedup() const noexcept
    {
        if (!isDynamicResidencyProductionTest())
            return false;
        const auto topology = activeTopology();
        return topology == OverlayTopology::CudaCpu ||
               topology == OverlayTopology::RocmCpu ||
               topology == OverlayTopology::CudaRocmCpu;
    }

    void SetUp() override
    {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized)
        {
            if (productionParityCampaignEnabled())
                FAIL() << "Production GraphNative CudaHot/RocmWarm/CpuCold parity requires MPI initialization";
            GTEST_SKIP() << "GraphNative CudaHot/RocmWarm/CpuCold parity requires MPI initialization";
        }

        int rank = 0;
        int world_size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        if (world_size < cfg().mpi_ranks)
        {
            if (productionParityCampaignEnabled())
            {
                FAIL() << "Production GraphNative CudaHot/RocmWarm/CpuCold parity requires "
                       << cfg().mpi_ranks << " MPI ranks (got "
                       << world_size << ")";
            }
            GTEST_SKIP() << "GraphNative CudaHot/RocmWarm/CpuCold parity requires "
                         << cfg().mpi_ranks << " MPI ranks (got " << world_size << ")";
        }

        mpi_ctx_ = std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD);
        try
        {
            cluster_inventory_ = gatherClusterInventory(mpi_ctx_);
        }
        catch (const std::exception &e)
        {
            FAIL() << "Failed to gather cluster inventory for topology binding: "
                   << e.what();
        }

        Base::SetUp();
        if (this->HasFatalFailure() || !isSegmentedPrefillProductionTest() ||
            !modelAvailable())
        {
            return;
        }

        ASSERT_GT(
            config_.token_ids.size(),
            static_cast<size_t>(kSegmentedPrefillCaptureRows))
            << "Segmented graph-native parity requires an authenticated prompt "
               "longer than its captured bucket";
        configureSegmentedProductionParityPrefillGraphBucket(
            kSegmentedPrefillCaptureRows);
        ASSERT_EQ(
            debugEnv().execution.prefill_graph_bucket_sizes,
            std::vector<int>{kSegmentedPrefillCaptureRows})
            << "Segmented parity must override the exact-bucket default with "
               "one explicit fixed capture bucket";
    }

    void applyModelOverrides() override
    {
        if (config_.prompt.empty())
            config_.prompt = "The quick brown fox jumps over the lazy dog";
        if (config_.token_ids.empty())
            config_.token_ids = {785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562};

        if (!cfg().model_path.empty())
            config_.model_path = cfg().model_path;
        if (!cfg().snapshot_dir.empty())
            config_.snapshot_dir = cfg().snapshot_dir;

        if (!modelAvailable())
            return;

        const auto metadata_path = std::filesystem::path(config_.snapshot_dir) / "metadata.txt";
        const bool metadata_missing = !std::filesystem::exists(metadata_path);
        const bool metadata_stale = !metadata_missing &&
                                    readSnapshotVersion(metadata_path) < kRequiredSnapshotVersion;
        const int local_needs_regen = (metadata_missing || metadata_stale) ? 1 : 0;
        int global_needs_regen = 0;
        MPI_Allreduce(&local_needs_regen, &global_needs_regen, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        int local_regen_failed = 0;
        if (global_needs_regen && isRank0())
        {
            LOG_INFO("[Qwen3.5 MoE GraphNative] Regenerating PyTorch snapshots on rank 0");
            local_regen_failed = regeneratePyTorchSnapshots() ? 0 : 1;
        }
        int global_regen_failed = 0;
        MPI_Allreduce(&local_regen_failed, &global_regen_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (global_regen_failed)
        {
            ADD_FAILURE() << "Qwen3.5 MoE snapshot regeneration failed";
            return;
        }

        auto prefill_tokens = readPrefillTokensFromMetadata();
        if (!prefill_tokens.empty())
        {
            config_.token_ids = std::move(prefill_tokens);
            LOG_INFO("[Qwen3.5 MoE GraphNative] Loaded " << config_.token_ids.size()
                                                         << " prefill token IDs from metadata");
        }
    }

    bool broadcastRootFlag(bool root_value) const
    {
        const int root_rank = parityArtifactAuthorityRank();
        int flag = isRootParityRank() && root_value ? 1 : 0;
        MPI_Bcast(&flag, 1, MPI_INT, root_rank, MPI_COMM_WORLD);
        return flag != 0;
    }

    bool synchronizeRanksOk(bool local_ok) const
    {
        int ok = local_ok ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        return ok == 1;
    }

    /**
     * @brief Prove each expert-only endpoint owns one immutable compact arena.
     *
     * Every process retains its own PerfStats records, while the graph-native
     * overlay spreads its accelerator tier and the two CPU-NUMA endpoints
     * across MPI instances. The dense continuation uses its model graph's
     * activation arena and must not duplicate the follower runner's storage.
     * Aggregate setup evidence only after the worker loop closes; this proves
     * the real follower graph did not allocate one packet per layer or alias
     * independent participants.
     */
    void assertParticipantCompactBufferArenaEvidence() const
    {
        constexpr size_t kAllocationValue = 0;
        constexpr size_t kAllocationRecordCount = 1;
        constexpr size_t kAllocationBytes = 2;
        constexpr size_t kByteRecordCount = 3;
        constexpr size_t kMalformedRecords = 4;
        constexpr size_t kParticipantStart = 5;
        const size_t participant_count = activeOverlayParticipantCount();
        ASSERT_GT(participant_count, 1u);
        const size_t follower_participant_count = participant_count - 1u;
        const size_t evidence_count = kParticipantStart + participant_count;

        const int local_memory_domain_enabled =
            PerfStatsCollector::isDomainEnabled("memory") ? 1 : 0;
        int all_memory_domains_enabled = 0;
        MPI_Allreduce(
            &local_memory_domain_enabled,
            &all_memory_domains_enabled,
            1,
            MPI_INT,
            MPI_MIN,
            MPI_COMM_WORLD);
        if (all_memory_domains_enabled == 0)
        {
            if (isRootParityRank())
            {
                ADD_FAILURE()
                    << "Production graph-native parity must retain the PerfStats "
                       "memory domain to prove serial compact-arena ownership";
            }
            return;
        }

        std::vector<uint64_t> local(evidence_count, 0u);
        for (const auto &record : PerfStatsCollector::snapshot({"memory"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "memory")
            {
                continue;
            }

            const bool is_allocation =
                record.name == "moe_serial_local_expert_buffer_arena_allocations";
            const bool is_bytes =
                record.name == "moe_serial_local_expert_buffer_arena_bytes";
            if (!is_allocation && !is_bytes)
                continue;

            const auto ownership = record.tags.find("ownership");
            const auto immutable = record.tags.find("immutable");
            const auto participant = record.tags.find("participant");
            const bool tag_contract_ok =
                record.phase == "model_setup" &&
                ownership != record.tags.end() &&
                ownership->second == "per_device_participant_serial_graph_family" &&
                immutable != record.tags.end() && immutable->second == "true" &&
                participant != record.tags.end();
            if (!tag_contract_ok || record.count != 1u || record.value <= 0.0)
            {
                ++local[kMalformedRecords];
                continue;
            }

            int participant_id = -1;
            for (int candidate = 0;
                 candidate < static_cast<int>(participant_count);
                 ++candidate)
            {
                if (participant->second == std::to_string(candidate))
                {
                    participant_id = candidate;
                    break;
                }
            }
            if (participant_id < 0)
            {
                ++local[kMalformedRecords];
                continue;
            }

            if (is_allocation)
            {
                if (record.value != 1.0)
                {
                    ++local[kMalformedRecords];
                    continue;
                }
                ++local[kAllocationValue];
                ++local[kAllocationRecordCount];
                ++local[kParticipantStart + static_cast<size_t>(participant_id)];
                continue;
            }

            local[kAllocationBytes] += static_cast<uint64_t>(record.value);
            ++local[kByteRecordCount];
        }

        std::vector<uint64_t> global(evidence_count, 0u);
        MPI_Allreduce(
            local.data(),
            global.data(),
            static_cast<int>(global.size()),
            MPI_UINT64_T,
            MPI_SUM,
            MPI_COMM_WORLD);

        if (!isRootParityRank())
            return;

        EXPECT_EQ(global[kMalformedRecords], 0u)
            << "Serial compact-arena PerfStats tags or aggregation were malformed";
        EXPECT_EQ(global[kAllocationValue], follower_participant_count)
            << "Each expert-only follower must allocate one compact graph-family arena";
        EXPECT_EQ(global[kAllocationRecordCount], follower_participant_count)
            << "Each expert-only follower must publish one unaggregated arena allocation record";
        EXPECT_EQ(global[kByteRecordCount], follower_participant_count)
            << "Each expert-only follower must publish one arena byte-accounting record";
        EXPECT_GT(global[kAllocationBytes], 0u)
            << "Follower compact arenas must account for their fixed graph-family storage";
        EXPECT_EQ(global[kParticipantStart], 0u)
            << "The dense continuation participant must not allocate the expert-only runner's duplicate compact arena";
        for (size_t participant = 1;
             participant < participant_count;
             ++participant)
        {
            EXPECT_EQ(global[kParticipantStart + participant], 1u)
                << "Expected exactly one follower compact arena for participant p"
                << participant;
        }
    }

    /**
     * @brief Prove tickets selected setup-owned mapped graph specializations.
     *
     * Device-owned epochs replaced the old host-scheduled fixed-capacity
     * participant runner. Setup materializes a bounded row-shape family once;
     * each authenticated ticket selects one member without mutating token or
     * position state on the follower. Native parent capture/replay is asserted
     * independently by the shared production-path evidence gate.
     */
    void assertMappedParticipantGraphEvidence() const
    {
        constexpr size_t kMalformedRecords = 0;
        constexpr size_t kMainFamilyMaterializations = 1;
        constexpr size_t kSelectedGraphs = 2;
        constexpr size_t kLargestPrefillSelections = 3;
        constexpr size_t kOneRowPrefillSelections = 4;
        constexpr size_t kOneRowDecodeSelections = 5;
        constexpr size_t kRetiredHostRunnerRecords = 6;
        constexpr size_t kEvidenceCount = 7;

        int expected_largest_prefill_rows = 0;
        if (isRootParityRank())
        {
            expected_largest_prefill_rows =
                isSegmentedPrefillProductionTest()
                    ? kSegmentedPrefillCaptureRows
                    : static_cast<int>(config_.token_ids.size());
        }
        MPI_Bcast(
            &expected_largest_prefill_rows,
            1,
            MPI_INT,
            parityArtifactAuthorityRank(),
            MPI_COMM_WORLD);
        ASSERT_GT(expected_largest_prefill_rows, 0);

        const int local_domain_enabled =
            PerfStatsCollector::isDomainEnabled(
                "moe_overlay_participant_graph")
                ? 1
                : 0;
        int all_domains_enabled = 0;
        MPI_Allreduce(
            &local_domain_enabled,
            &all_domains_enabled,
            1,
            MPI_INT,
            MPI_MIN,
            MPI_COMM_WORLD);
        if (all_domains_enabled == 0)
        {
            if (isRootParityRank())
            {
                ADD_FAILURE()
                    << "Production graph-native parity must retain the "
                       "moe_overlay_participant_graph PerfStats domain";
            }
            return;
        }

        std::array<uint64_t, kEvidenceCount> local{};
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"moe_overlay_participant_graph"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "moe_overlay_participant_graph")
            {
                continue;
            }
            if (record.name == "materialized_graphs" ||
                record.name == "fixed_capacity_graph_reuses")
            {
                local[kRetiredHostRunnerRecords] += record.count;
                continue;
            }

            const auto tag = [&record](const char *name) -> const std::string *
            {
                const auto found = record.tags.find(name);
                return found == record.tags.end() ? nullptr : &found->second;
            };
            const auto parse_positive = [&](const char *name) -> int
            {
                const auto value = tag(name);
                if (!value)
                    return 0;
                int parsed = 0;
                const char *const begin = value->data();
                const char *const end = begin + value->size();
                const auto result = std::from_chars(begin, end, parsed);
                return result.ec == std::errc{} && result.ptr == end &&
                               parsed > 0
                           ? parsed
                           : 0;
            };

            if (record.name == "materialized_mapped_follower_families")
            {
                const auto host_sparse_stages = tag("host_sparse_stages");
                const auto graph_family = tag("graph_family");
                const bool valid =
                    record.phase == "model_setup" && record.value == 1.0 &&
                    record.count == 1u && parse_positive("row_shapes") > 0 &&
                    parse_positive("source_layers") > 0 &&
                    host_sparse_stages && *host_sparse_stages == "0" &&
                    graph_family;
                if (!valid)
                {
                    ++local[kMalformedRecords];
                    continue;
                }
                if (*graph_family == "0")
                    ++local[kMainFamilyMaterializations];
                continue;
            }
            if (record.name != "ticket_selected_graphs")
                continue;

            const int logical_rows = parse_positive("logical_rows");
            const int physical_rows = parse_positive("physical_rows");
            const auto transport_path = tag("transport_path");
            const auto position_mutated = tag("position_mutated");
            const auto gpu_host_dispatches = tag("gpu_host_layer_dispatches");
            const bool phase_valid =
                record.phase == "main_prefill" ||
                record.phase == "main_decode" ||
                record.phase == "mtp_grouped_verifier" ||
                record.phase == "mtp_draft";
            const bool valid =
                record.value > 0.0 && record.count > 0u && phase_valid &&
                logical_rows > 0 && physical_rows >= logical_rows &&
                transport_path &&
                transport_path->rfind("node_local_mapped_", 0) == 0 &&
                position_mutated && *position_mutated == "false" &&
                gpu_host_dispatches && *gpu_host_dispatches == "0";
            if (!valid)
            {
                ++local[kMalformedRecords];
                continue;
            }

            local[kSelectedGraphs] += record.count;
            if (record.phase == "main_prefill" &&
                logical_rows == expected_largest_prefill_rows)
            {
                local[kLargestPrefillSelections] += record.count;
            }
            if (record.phase == "main_prefill" && logical_rows == 1)
                local[kOneRowPrefillSelections] += record.count;
            if (record.phase == "main_decode" && logical_rows == 1)
                local[kOneRowDecodeSelections] += record.count;
        }

        std::array<uint64_t, kEvidenceCount> global{};
        MPI_Allreduce(
            local.data(),
            global.data(),
            static_cast<int>(global.size()),
            MPI_UINT64_T,
            MPI_SUM,
            MPI_COMM_WORLD);
        if (!isRootParityRank())
            return;

        EXPECT_EQ(global[kMalformedRecords], 0u)
            << "Mapped participant graph evidence was malformed";
        EXPECT_EQ(global[kMainFamilyMaterializations], 1u)
            << "The one auxiliary MPI runner must materialize its main mapped graph family exactly once";
        EXPECT_GT(global[kSelectedGraphs], 0u)
            << "No authenticated ticket selected a mapped participant graph";
        EXPECT_GT(global[kLargestPrefillSelections], 0u)
            << "The mapped family never served the largest root-published live prefill chunk";
        EXPECT_GT(global[kOneRowDecodeSelections], 0u)
            << "Decode never selected its setup-owned one-row retained parent";
        EXPECT_EQ(global[kRetiredHostRunnerRecords], 0u)
            << "The retired host-scheduled participant graph path executed";
        if (isSegmentedPrefillProductionTest())
        {
            EXPECT_GT(global[kOneRowPrefillSelections], 0u)
                << "The short prefill tail never selected its bounded mapped graph";
        }
    }

    /**
     * @brief Attribute live parity router checkpoints to the published epoch.
     *
     * Snapshot values are exact integer expert ids produced by the real router.
     * Resolve them through the immutable residency snapshot while both remain
     * live, before the parity harness clears request diagnostics. This avoids a
     * host shadow: the test reads one already-published epoch and one existing
     * numerical checkpoint, then retains only aggregate evidence for the final
     * cross-rank gate.
     */
    void cachePublishedEpochRouteEvidence()
    {
        if (!isRootParityRank())
            return;

        auto *const concrete =
            dynamic_cast<OrchestrationRunner *>(orch_runner_.get());
        const auto snapshot = concrete
                                  ? concrete->expertOverlayResidencySnapshotForDiagnostics()
                                  : nullptr;
        if (!snapshot || !snapshot->valid())
        {
            ADD_FAILURE()
                << "Production parity could not inspect its immutable live ExpertOverlay residency epoch";
            return;
        }

        const size_t participant_count = activeOverlayParticipantCount();
        if (snapshot->owner_map.participants().size() != participant_count)
        {
            ADD_FAILURE()
                << "Published ExpertOverlay participant cardinality changed before route attribution";
            return;
        }

        std::vector<uint64_t> route_counts(participant_count, 0u);
        std::vector<bool> requires_remote_completion(
            participant_count, false);
        for (const auto &participant : snapshot->owner_map.participants())
        {
            if (participant.participant_id < 0 ||
                static_cast<size_t>(participant.participant_id) >=
                    participant_count ||
                !participant.world_rank_known)
            {
                ADD_FAILURE()
                    << "Published ExpertOverlay participant identity is incomplete";
                return;
            }
            requires_remote_completion[
                static_cast<size_t>(participant.participant_id)] =
                participant.world_rank != parityArtifactAuthorityRank();
        }

        size_t checkpoint_layers = 0u;
        for (const auto &placement :
             snapshot->placement_plan->placements)
        {
            if (placement.layer < 0 ||
                placement.routed_expert_tier.empty())
            {
                ADD_FAILURE()
                    << "Published ExpertOverlay placement contains invalid layer geometry";
                return;
            }
            size_t route_elements = 0u;
            const std::string key =
                "layer" + std::to_string(placement.layer) +
                "_MOE_ROUTING_INDICES";
            const float *const routes = activeSnapshot(key, route_elements);
            if (!routes)
                continue;
            ++checkpoint_layers;
            for (size_t index = 0u; index < route_elements; ++index)
            {
                const float routed = routes[index];
                if (!std::isfinite(routed) || routed < 0.0f ||
                    routed > static_cast<float>(std::numeric_limits<int>::max()))
                {
                    ADD_FAILURE()
                        << "Router checkpoint " << key
                        << " contains a non-integral or out-of-range expert ID "
                        << routed << " at element " << index;
                    return;
                }

                // Validate the floating snapshot before converting it: a cast
                // of NaN or an out-of-range float to int is undefined behavior.
                const int expert = static_cast<int>(routed);
                if (routed != static_cast<float>(expert) ||
                    static_cast<size_t>(expert) >=
                        placement.routed_expert_tier.size())
                {
                    ADD_FAILURE()
                        << "Router checkpoint " << key
                        << " contains a non-integral or out-of-range expert id";
                    return;
                }
                const auto *const owner = snapshot->owner_map.ownerFor(
                    placement.layer, expert);
                if (!owner || owner->owner_participant < 0 ||
                    static_cast<size_t>(owner->owner_participant) >=
                        participant_count)
                {
                    ADD_FAILURE()
                        << "Published residency map cannot resolve " << key
                        << " expert " << expert;
                    return;
                }
                ++route_counts[
                    static_cast<size_t>(owner->owner_participant)];
            }
        }
        if (checkpoint_layers == 0u)
        {
            ADD_FAILURE()
                << "Production parity retained no MoE routing checkpoint for live-epoch attribution";
            return;
        }
        parity_route_counts_by_participant_ = std::move(route_counts);
        parity_route_requires_remote_completion_ =
            std::move(requires_remote_completion);
    }

    /**
     * @brief Prove every declared participant was selected and remote work completed.
     *
     * The routed checkpoint proves selection under the exact published owner
     * map. Cross-rank participants additionally require endpoint-owned traffic
     * totals acquired only after both retained graphs publish Complete. Local
     * participant arithmetic is covered by the same per-layer/LM-head parity
     * comparison that supplied the routing checkpoint.
     */
    void assertActiveTierRouteEvidence() const
    {
        struct ExpectedRoute
        {
            int participant;
            int tier;
            const char *device_kind;
            const char *label;
        };

        std::vector<ExpectedRoute> expected;
        switch (activeTopology())
        {
        case OverlayTopology::CudaRocmCpu:
            expected = {
                {0, 0, "CUDA", "CUDA hot"},
                {1, 1, "ROCm", "ROCm warm"},
                {2, 2, "CPU", "CPU cold NUMA0"},
                {3, 2, "CPU", "CPU cold NUMA1"},
            };
            break;
        case OverlayTopology::CudaCpu:
            expected = {
                {0, 0, "CUDA", "CUDA hot"},
                {1, 1, "CPU", "CPU cold NUMA0"},
                {2, 1, "CPU", "CPU cold NUMA1"},
            };
            break;
        case OverlayTopology::RocmCpu:
            expected = {
                {0, 0, "ROCm", "ROCm hot"},
                {1, 1, "CPU", "CPU cold NUMA0"},
                {2, 1, "CPU", "CPU cold NUMA1"},
            };
            break;
        case OverlayTopology::CudaRocm:
            expected = {
                {0, 0, "CUDA", "CUDA hot"},
                {1, 1, "ROCm", "ROCm warm"},
            };
            break;
        }

        constexpr size_t kMalformedRecords = 0;
        constexpr size_t kSelectedStart = 1;
        const size_t completed_start = kSelectedStart + expected.size();
        std::vector<uint64_t> local(
            completed_start + expected.size(), 0u);
        if (isRootParityRank())
        {
            if (parity_route_counts_by_participant_.size() !=
                    expected.size() ||
                parity_route_requires_remote_completion_.size() !=
                    expected.size())
            {
                ++local[kMalformedRecords];
            }
            else
            {
                std::copy(
                    parity_route_counts_by_participant_.begin(),
                    parity_route_counts_by_participant_.end(),
                    local.begin() + kSelectedStart);
            }
        }
        for (const auto &record : PerfStatsCollector::snapshot({"forward_graph"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "forward_graph" ||
                record.name != "moe_overlay_local_expert_active_routes")
            {
                continue;
            }

            const auto tag = [&record](const char *name) -> const std::string *
            {
                const auto it = record.tags.find(name);
                return it == record.tags.end() ? nullptr : &it->second;
            };
            const auto completion = tag("completion");
            const auto device_kind = tag("device_kind");
            const auto generation = tag("generation");
            const auto identity_source = tag("identity_source");
            const auto participant = tag("participant");
            const auto tier = tag("tier");
            // PerfStats coalesces repeated calls with the same full tag set.
            // A completed-route record can therefore represent several live
            // sparse packets (for example one packet per MoE layer), rather
            // than one call.  `value` is their summed route count and `count`
            // is the number of completed packets, so requiring count==1 here
            // would reject the strongest possible production evidence.
            const bool legacy_completion =
                completion &&
                *completion == "local_expert_packet_complete" &&
                identity_source &&
                *identity_source == "sparse_collective_key";
            const bool device_epoch_completion =
                completion &&
                *completion == "device_owned_epoch_complete" &&
                identity_source &&
                *identity_source == "device_owned_activation_epoch";
            const bool record_contract_ok =
                record.phase == "moe_overlay" &&
                record.count > 0u &&
                record.value > 0.0 &&
                generation && *generation != "0" &&
                (legacy_completion || device_epoch_completion) &&
                participant && tier && device_kind;
            if (!record_contract_ok)
            {
                ++local[kMalformedRecords];
                continue;
            }

            auto route = std::find_if(
                expected.begin(),
                expected.end(),
                [&](const ExpectedRoute &candidate)
                {
                    return *participant == std::to_string(candidate.participant) &&
                           *tier == std::to_string(candidate.tier) &&
                           *device_kind == candidate.device_kind;
                });
            if (route == expected.end())
            {
                ++local[kMalformedRecords];
                continue;
            }

            const size_t destination = completed_start +
                static_cast<size_t>(std::distance(expected.begin(), route));
            local[destination] += static_cast<uint64_t>(record.value);
        }

        std::vector<uint64_t> global(local.size(), 0u);
        MPI_Allreduce(
            local.data(),
            global.data(),
            static_cast<int>(global.size()),
            MPI_UINT64_T,
            MPI_SUM,
            MPI_COMM_WORLD);

        if (!isRootParityRank())
            return;

        EXPECT_EQ(global[kMalformedRecords], 0u)
            << "Graph-native local-expert route evidence was malformed";
        for (size_t index = 0; index < expected.size(); ++index)
        {
            EXPECT_GT(global[kSelectedStart + index], 0u)
                << expected[index].label
                << " participant p" << expected[index].participant
                << " was never selected by the real parity router under the published residency epoch";
            if (parity_route_requires_remote_completion_.size() ==
                    expected.size() &&
                parity_route_requires_remote_completion_[index])
            {
                EXPECT_GT(global[completed_start + index], 0u)
                    << expected[index].label
                    << " participant p" << expected[index].participant
                    << " published no completed device-owned sparse traffic";
            }
        }
    }

    /**
     * @brief Prove the real sparse transport moved compact packets between tiers.
     *
     * Local-route completion proves that every participant ran an expert, but
     * it does not independently prove that the production sparse collective
     * carried compact request and result packets. The graph-native transport
     * publishes those byte counts through PerfStats. Folding them across both
     * MPI instances makes a missing dispatch, missing return, or silently
     * bypassed CPU cold tier a fatal parity failure without paying for a second
     * model setup in a profiler-only smoke test.
     */
    void assertSparseTransportPerfStatsEvidence() const
    {
        constexpr size_t kMalformedRecords = 0;
        constexpr size_t kCompactDispatchBytes = 1;
        constexpr size_t kCompactReturnBytes = 2;
        constexpr size_t kCpuRows = 3;
        constexpr size_t kGpuRows = 4;
        constexpr size_t kEvidenceCount = 5;

        const int local_domain_enabled =
            PerfStatsCollector::isDomainEnabled("moe_overlay") ? 1 : 0;
        int all_domains_enabled = 0;
        MPI_Allreduce(
            &local_domain_enabled,
            &all_domains_enabled,
            1,
            MPI_INT,
            MPI_MIN,
            MPI_COMM_WORLD);
        if (all_domains_enabled == 0)
        {
            if (isRootParityRank())
            {
                ADD_FAILURE()
                    << "Production graph-native parity must retain the "
                       "moe_overlay PerfStats domain";
            }
            return;
        }

        std::array<uint64_t, kEvidenceCount> local{};
        for (const auto &record : PerfStatsCollector::snapshot({"moe_overlay"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "moe_overlay")
            {
                continue;
            }

            const auto tag = [&record](const char *name) -> const std::string *
            {
                const auto found = record.tags.find(name);
                return found == record.tags.end() ? nullptr : &found->second;
            };
            const auto transport = tag("transport");
            const auto domain_kind = tag("domain_kind");
            const auto participant = tag("participant");
            const bool positive = record.count > 0u && record.value > 0.0;

            if (record.name == "compact_dispatch_bytes")
            {
                if (record.phase != "gn_sparse_dispatch" || !positive ||
                    !transport || *transport != "compact")
                {
                    ++local[kMalformedRecords];
                    continue;
                }
                local[kCompactDispatchBytes] +=
                    static_cast<uint64_t>(record.value);
                continue;
            }
            if (record.name == "compact_return_bytes")
            {
                if (record.phase != "gn_return_reduce" || !positive ||
                    !transport || *transport != "compact")
                {
                    ++local[kMalformedRecords];
                    continue;
                }
                local[kCompactReturnBytes] +=
                    static_cast<uint64_t>(record.value);
                continue;
            }
            if (record.name != "cpu_rows" && record.name != "gpu_rows")
                continue;

            int participant_id = -1;
            try
            {
                participant_id = participant ? std::stoi(*participant) : -1;
            }
            catch (const std::exception &)
            {
                participant_id = -1;
            }
            const bool cpu_record = record.name == "cpu_rows";
            const char *expected_kind = cpu_record ? "CPU" : "GPU";
            if (record.phase != "gn_local_expert" || !positive ||
                !transport || *transport != "local" ||
                !domain_kind || *domain_kind != expected_kind ||
                participant_id < 0)
            {
                ++local[kMalformedRecords];
                continue;
            }
            local[cpu_record ? kCpuRows : kGpuRows] +=
                static_cast<uint64_t>(record.value);
        }

        std::array<uint64_t, kEvidenceCount> global{};
        MPI_Allreduce(
            local.data(),
            global.data(),
            static_cast<int>(global.size()),
            MPI_UINT64_T,
            MPI_SUM,
            MPI_COMM_WORLD);
        if (!isRootParityRank())
            return;

        EXPECT_EQ(global[kMalformedRecords], 0u)
            << "Graph-native sparse transport PerfStats evidence was malformed";
        EXPECT_GT(global[kCompactDispatchBytes], 0u)
            << "Production sparse collective emitted no compact dispatch bytes";
        EXPECT_GT(global[kCompactReturnBytes], 0u)
            << "Production sparse collective emitted no compact return bytes";
        EXPECT_GT(global[kGpuRows], 0u)
            << "No GPU tier published completed graph-native expert rows";
        if (topologyUsesCpu())
        {
            EXPECT_GT(global[kCpuRows], 0u)
                << "The configured CPU cold tier published no completed expert rows";
        }
        else
        {
            EXPECT_EQ(global[kCpuRows], 0u)
                << "The CUDA/ROCm-only topology unexpectedly executed CPU expert rows";
        }
    }

    /**
     * @brief Prove every real heterogeneous prefill request used one shared [4,4,1] schedule.
     *
     * PrefixRuntimeStateSnapshot proves the public OrchestrationRunner admitted
     * a chunked request instead of treating the test as three unrelated
     * forwards. Per-rank PerfStats then prove the distributed schedule contract
     * was published, an expert-only participant consumed the same live-row
     * chunks, and the CUDA continuation retained complete prompt-wide
     * checkpoints for the existing Hugging Face/CSV comparator. This fixture
     * deliberately pre-fills twice: once for prompt-wide checkpoint comparison
     * and once after reset to establish the independent decode comparison.
     * Evidence must therefore prove both complete production transactions.
     */
    void assertSegmentedPrefillEvidence() const
    {
        constexpr uint64_t kExpectedChunks = 3;
        constexpr uint64_t kExpectedRealTokens = 9;
        constexpr uint64_t kExpectedPaddedTokens = 3;
        constexpr uint64_t kExpectedPrefillTransactions = 2;
        constexpr uint64_t kExpectedObservedChunks =
            kExpectedChunks * kExpectedPrefillTransactions;
        constexpr uint64_t kExpectedObservedRealTokens =
            kExpectedRealTokens * kExpectedPrefillTransactions;
        constexpr uint64_t kExpectedObservedPaddedTokens =
            kExpectedPaddedTokens * kExpectedPrefillTransactions;
        constexpr size_t kContractRecords = 0;
        constexpr size_t kRemoteScheduleRecords = 1;
        constexpr size_t kSnapshotAggregationRecords = 2;
        constexpr size_t kContinuationTransactionRecords = 3;
        constexpr size_t kParticipantTransactionRecords = 4;
        constexpr size_t kContinuationTransactionSteps = 5;
        constexpr size_t kParticipantTransactionSteps = 6;
        constexpr size_t kMalformedRecords = 7;
        constexpr size_t kEvidenceCount = 8;
        constexpr uint64_t kExpectedTransactionStepMask = 0x7u; // offsets 0, 4, 8

        std::array<uint64_t, kEvidenceCount> local{};
        for (const auto &record : PerfStatsCollector::snapshot({"forward_graph"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "forward_graph")
            {
                continue;
            }

            const auto tagEquals = [&record](
                                       const char *name,
                                       const std::string &expected)
            {
                const auto it = record.tags.find(name);
                return it != record.tags.end() && it->second == expected;
            };

            if (record.name == "moe_overlay_prefill_schedule_contract_rows")
            {
                if (record.phase != "model_setup" ||
                    record.value != static_cast<double>(kSegmentedPrefillCaptureRows) ||
                    record.count != 1u ||
                    !tagEquals("authority", "continuation_root") ||
                    !tagEquals("bucket_count", "1") ||
                    !tagEquals("immutable", "true"))
                {
                    ++local[kMalformedRecords];
                }
                else
                {
                    ++local[kContractRecords];
                }
                continue;
            }

            if (record.name == "moe_overlay_participant_prefill_chunk_schedules")
            {
                if (record.phase != "prefill" ||
                    record.value !=
                        static_cast<double>(kExpectedObservedChunks) ||
                    record.count != kExpectedPrefillTransactions ||
                    !tagEquals("logical_rows", std::to_string(kExpectedRealTokens)) ||
                    !tagEquals("bucket_rows", std::to_string(kSegmentedPrefillCaptureRows)) ||
                    !tagEquals("capacity", std::to_string(kSegmentedPrefillCaptureRows)) ||
                    !tagEquals("live_rows_only", "true"))
                {
                    ++local[kMalformedRecords];
                }
                else
                {
                    ++local[kRemoteScheduleRecords];
                }
                continue;
            }

            if (record.name == "prefill_chunk_snapshot_sequence_keys")
            {
                if (record.phase != "prefill" || record.value <= 0.0 ||
                    record.count != kExpectedPrefillTransactions ||
                    !tagEquals("chunks", std::to_string(kExpectedChunks)) ||
                    !tagEquals("diagnostic_only", "true"))
                {
                    ++local[kMalformedRecords];
                }
                else
                {
                    ++local[kSnapshotAggregationRecords];
                }
                continue;
            }

            if (record.name == "moe_overlay_collective_transaction")
            {
                /*
                 * Decode parity performs its own prefill initialization, then
                 * emits serial decode transactions. Those records are covered
                 * by the decode comparator; they are not malformed prefill
                 * evidence merely because they share the sparse metric name.
                 */
                if (record.phase != "prefill")
                    continue;

                const auto role = record.tags.find("role");
                const auto generation = record.tags.find("generation");
                const auto logical_step = record.tags.find("logical_step");
                uint64_t step_mask = 0;
                if (logical_step != record.tags.end())
                {
                    if (logical_step->second == "0")
                        step_mask = 0x1u;
                    else if (logical_step->second == "4")
                        step_mask = 0x2u;
                    else if (logical_step->second == "8")
                        step_mask = 0x4u;
                }

                const bool transaction_contract_ok =
                    record.value == 1.0 &&
                    record.count == 1u &&
                    tagEquals("identity_source", "orchestration_request_and_chunk") &&
                    generation != record.tags.end() &&
                    generation->second != "0" &&
                    step_mask != 0;
                if (!transaction_contract_ok || role == record.tags.end())
                {
                    ++local[kMalformedRecords];
                    continue;
                }

                if (role->second == "continuation_graph")
                {
                    ++local[kContinuationTransactionRecords];
                    local[kContinuationTransactionSteps] |= step_mask;
                }
                else if (role->second == "expert_participant_graph")
                {
                    ++local[kParticipantTransactionRecords];
                    local[kParticipantTransactionSteps] |= step_mask;
                }
                else
                {
                    ++local[kMalformedRecords];
                }
            }
        }

        std::array<uint64_t, kEvidenceCount> global{};
        MPI_Allreduce(
            local.data(),
            global.data(),
            static_cast<int>(global.size()),
            MPI_UINT64_T,
            MPI_SUM,
            MPI_COMM_WORLD);

        if (!isRootParityRank())
            return;

        const PrefixRuntimeStateSnapshot probe = activePrefixStateProbe();
        EXPECT_EQ(
            probe.prefill_chunk_schedules,
            kExpectedPrefillTransactions)
            << "Prompt and decode setup must each register one scheduler transaction";
        EXPECT_EQ(
            probe.prefill_chunk_successful_schedules,
            kExpectedPrefillTransactions)
            << "Every production prefill transaction must complete its shared scheduler";
        EXPECT_EQ(probe.prefill_chunks, kExpectedObservedChunks)
            << "Each authenticated prefill must run the full [4,4,1] schedule";
        EXPECT_EQ(probe.prefill_chunk_real_tokens, kExpectedObservedRealTokens)
            << "Padding must never be counted as routed expert work";
        EXPECT_EQ(
            probe.prefill_chunk_padded_tokens,
            kExpectedObservedPaddedTokens)
            << "Only each final one-real-row bucket may contribute padding";
        EXPECT_EQ(probe.prefill_chunk_failures, 0u)
            << "A graph-native heterogeneous schedule may not recover through "
               "an eager or uncaptured fallback";

        EXPECT_EQ(global[kMalformedRecords], 0u)
            << "Segmented prefill PerfStats tags or geometry were malformed";
        EXPECT_EQ(
            global[kContractRecords],
            static_cast<uint64_t>(mpiWorldSize()))
            << "Every MPI participant must install the immutable shared bucket contract";
        EXPECT_EQ(global[kRemoteScheduleRecords], 1u)
            << "The one remote ExpertOverlay graph runner must account for both "
               "root-authoritative live-row schedules";
        EXPECT_EQ(global[kSnapshotAggregationRecords], 1u)
            << "The continuation graph must aggregate both prefills into one evidence record";
        EXPECT_EQ(
            global[kContinuationTransactionRecords],
            kExpectedObservedChunks)
            << "The continuation graph must stamp every [0,4,8] prefill transaction";
        EXPECT_EQ(
            global[kParticipantTransactionRecords],
            kExpectedObservedChunks)
            << "The remote expert graph must stamp every [0,4,8] prefill transaction";
        EXPECT_EQ(global[kContinuationTransactionSteps], kExpectedTransactionStepMask)
            << "The continuation graph must use the absolute chunk offsets [0,4,8]";
        EXPECT_EQ(global[kParticipantTransactionSteps], kExpectedTransactionStepMask)
            << "The remote graph must use the exact root-published chunk offsets [0,4,8]";
    }

    /**
     * @brief Verify every chunk-scoped Hugging Face checkpoint became full-prompt data.
     *
     * SnapshotCapture retains context-qualified copies such as
     * `PREFILL_CHUNK_0_layer0_...` for diagnosis and rewrites the bare semantic
     * key to the ordered aggregate. This check prevents a short final chunk
     * from passing merely because a comparison used the shorter tensor length.
     */
    void assertSegmentedPrefillCheckpointCoverage()
    {
        constexpr const char *kFirstChunkPrefix = "PREFILL_CHUNK_0_";
        const auto keys = activeSnapshotKeys();
        size_t checked = 0;
        for (const std::string &scoped_key : keys)
        {
            if (scoped_key.rfind(kFirstChunkPrefix, 0) != 0)
                continue;

            const std::string semantic_key =
                scoped_key.substr(std::char_traits<char>::length(kFirstChunkPrefix));
            if (semantic_key.empty() || semantic_key == "LM_HEAD")
                continue;

            const std::vector<float> reference = loadPyTorchSnapshot(semantic_key);
            if (reference.empty())
                continue;

            size_t aggregate_elements = 0;
            const float *aggregate = activeSnapshot(
                semantic_key,
                aggregate_elements);
            ASSERT_NE(aggregate, nullptr)
                << "Segmented checkpoint '" << semantic_key
                << "' lost its bare parity key after aggregation";
            EXPECT_EQ(aggregate_elements, reference.size())
                << "Segmented checkpoint '" << semantic_key
                << "' must contain every real prompt row";
            ++checked;
        }

        EXPECT_GT(checked, 0u)
            << "Segmented production prefill published no Hugging Face-backed "
               "chunk checkpoints";
    }

    bool collectivelyCheckHardwareAndModel() const
    {
        bool available = false;
        if (isRootParityRank())
            available =
                !acceleratorHardwareBlocker(cluster_inventory_).has_value() &&
                modelAvailable();
        return broadcastRootFlag(available);
    }

    /**
     * @brief Lexicographic objective for one adversarial CPU-tier subsequence.
     *
     * The first component maximizes authenticated routes assigned to the CPU
     * participant remote from the continuation rank. Only after that total is
     * fixed does the second component minimize traffic owned by the colocated
     * CPU participant. This is an initial-layout construction objective, never
     * a runtime histogram or placement decision.
     */
    struct AdversarialCpuPlacementScore
    {
        std::uint64_t remote_routes = 0;
        std::uint64_t local_routes = 0;
        bool reachable = false;
    };

    /** @return Whether @p candidate strictly improves the layout objective. */
    static bool improvesAdversarialCpuPlacement(
        const AdversarialCpuPlacementScore &candidate,
        const AdversarialCpuPlacementScore &incumbent) noexcept
    {
        if (!candidate.reachable)
            return false;
        if (!incumbent.reachable)
            return true;
        if (candidate.remote_routes != incumbent.remote_routes)
            return candidate.remote_routes > incumbent.remote_routes;
        return candidate.local_routes < incumbent.local_routes;
    }

    /**
     * @brief Select exact lower-tier membership under random owner partitioning.
     *
     * Whole-expert ownership first sorts the selected expert IDs, applies the
     * production ordinal/random permutation to those positions, and gives one
     * balanced span to each participant. Selecting a different set therefore
     * changes the sorted-rank occupied by every later expert. A small dynamic
     * program solves that subsequence problem exactly instead of assuming an
     * expert ID maps directly to a participant.
     *
     * @param routes Authenticated prefill route count for every expert.
     * @param layer Model layer used by the production random permutation.
     * @param tier_index Lower-priority CPU tier index used by that permutation.
     * @param selected_count Exact tier quota for this layer.
     * @param participant_count Number of balanced NodeTP CPU owners.
     * @return Boolean expert mask with exactly @p selected_count entries.
     */
    static std::vector<bool> selectAdversarialCpuExperts(
        const std::vector<std::uint64_t> &routes,
        int layer,
        int tier_index,
        int selected_count,
        int participant_count,
        RoutedExpertOwnerOrder owner_order)
    {
        if (routes.empty() || layer < 0 || tier_index < 0 ||
            selected_count <= 0 ||
            selected_count > static_cast<int>(routes.size()) ||
            participant_count < 2 || selected_count < participant_count)
        {
            throw std::invalid_argument(
                "Adversarial CPU owner selection has invalid geometry");
        }

        std::vector<int> selected_ranks(
            static_cast<std::size_t>(selected_count));
        std::iota(selected_ranks.begin(), selected_ranks.end(), 0);
        routed_expert_ownership::applyOwnerOrder(
            selected_ranks, owner_order, layer, tier_index);

        const int base = selected_count / participant_count;
        const int remainder = selected_count % participant_count;
        const int first_participant_count = base + (remainder > 0 ? 1 : 0);
        std::vector<bool> remote_selected_rank(
            static_cast<std::size_t>(selected_count), false);
        for (int offset = 0; offset < first_participant_count; ++offset)
        {
            remote_selected_rank.at(static_cast<std::size_t>(
                selected_ranks.at(static_cast<std::size_t>(offset)))) = true;
        }

        const std::size_t expert_count = routes.size();
        const std::size_t columns =
            static_cast<std::size_t>(selected_count) + 1u;
        std::vector<AdversarialCpuPlacementScore> scores(
            (expert_count + 1u) * columns);
        std::vector<std::uint8_t> took_expert(scores.size(), 0u);
        const auto index = [columns](std::size_t experts_seen,
                                     std::size_t experts_selected)
        {
            return experts_seen * columns + experts_selected;
        };
        scores[index(0u, 0u)].reachable = true;

        for (std::size_t experts_seen = 1u;
             experts_seen <= expert_count;
             ++experts_seen)
        {
            const std::size_t maximum_selected = std::min(
                experts_seen, static_cast<std::size_t>(selected_count));
            for (std::size_t experts_selected = 0u;
                 experts_selected <= maximum_selected;
                 ++experts_selected)
            {
                auto best = scores[index(
                    experts_seen - 1u, experts_selected)];
                bool take = false;
                if (experts_selected > 0u)
                {
                    auto candidate = scores[index(
                        experts_seen - 1u, experts_selected - 1u)];
                    if (candidate.reachable)
                    {
                        const std::uint64_t demand =
                            routes[experts_seen - 1u];
                        auto &total = remote_selected_rank[
                                          experts_selected - 1u]
                                          ? candidate.remote_routes
                                          : candidate.local_routes;
                        if (demand >
                            std::numeric_limits<std::uint64_t>::max() - total)
                        {
                            throw std::overflow_error(
                                "Adversarial CPU routing objective overflowed");
                        }
                        total += demand;
                        if (improvesAdversarialCpuPlacement(candidate, best))
                        {
                            best = candidate;
                            take = true;
                        }
                    }
                }
                scores[index(experts_seen, experts_selected)] = best;
                took_expert[index(experts_seen, experts_selected)] =
                    take ? 1u : 0u;
            }
        }

        if (!scores[index(expert_count,
                          static_cast<std::size_t>(selected_count))]
                 .reachable)
        {
            throw std::logic_error(
                "Adversarial CPU owner selection found no exact tier quota");
        }

        std::vector<bool> selected(expert_count, false);
        std::size_t experts_selected =
            static_cast<std::size_t>(selected_count);
        for (std::size_t experts_seen = expert_count;
             experts_seen > 0u;
             --experts_seen)
        {
            if (took_expert[index(experts_seen, experts_selected)] == 0u)
                continue;
            selected[experts_seen - 1u] = true;
            --experts_selected;
        }
        if (experts_selected != 0u ||
            static_cast<int>(std::count(
                selected.begin(), selected.end(), true)) != selected_count)
        {
            throw std::logic_error(
                "Adversarial CPU owner selection reconstruction changed its quota");
        }
        return selected;
    }

    /**
     * @brief Install an authenticated workload-adversarial initial tier layout.
     *
     * The Hugging Face pack is already the mathematical oracle for this parity
     * cell. Its routing IDs define only the starting condition: high-demand
     * experts are deliberately left on lower-priority participants, with the
     * strongest CPU candidates owned by the rank remote from continuation.
     * Runtime movement remains driven exclusively by histograms emitted from
     * the real Llaminar sparse-collective graphs.
     *
     * @param requested Inventory-bound dynamic production request.
     * @return Same request with complete explicit per-layer initial placement.
     * @throws std::exception For missing/malformed reference evidence or a
     *         layout that cannot make remote CPU demand strictly dominant.
     */
    MoERoutedExpertPlacementPlan installReferenceAdversarialPlacements(
        MoERoutedExpertPlacementPlan requested)
    {
        const auto metadata_path =
            std::filesystem::path(config_.snapshot_dir) / "metadata.txt";
        const auto layer_text =
            readSnapshotMetadataValue(metadata_path, "n_layers");
        int layer_count = 0;
        if (!layer_text)
        {
            throw std::invalid_argument(
                "Adversarial ExpertOverlay placement requires reference n_layers metadata");
        }
        const char *const layer_begin = layer_text->data();
        const char *const layer_end = layer_begin + layer_text->size();
        const auto parsed_layers =
            std::from_chars(layer_begin, layer_end, layer_count);
        if (parsed_layers.ec != std::errc{} ||
            parsed_layers.ptr != layer_end || layer_count <= 0)
        {
            throw std::invalid_argument(
                "Adversarial ExpertOverlay placement has invalid reference n_layers metadata");
        }

        auto cpu_tier = std::find_if(
            requested.routed_tiers.begin(),
            requested.routed_tiers.end(),
            [](const auto &tier)
            { return tier.domain == kCpuColdDomain; });
        if (cpu_tier == requested.routed_tiers.end())
        {
            throw std::invalid_argument(
                "Adversarial ExpertOverlay placement has no NodeTP CPU tier");
        }
        const int cpu_tier_index = static_cast<int>(std::distance(
            requested.routed_tiers.begin(), cpu_tier));
        const auto cpu_domain = std::find_if(
            requested.domains.begin(),
            requested.domains.end(),
            [](const auto &domain)
            { return domain.name == kCpuColdDomain; });
        const auto continuation = std::find_if(
            requested.domains.begin(),
            requested.domains.end(),
            [&](const auto &domain)
            { return domain.name == requested.continuation_domain; });
        if (cpu_domain == requested.domains.end() ||
            continuation == requested.domains.end() ||
            cpu_domain->participants.size() != 2u ||
            cpu_domain->world_ranks.size() != 2u ||
            continuation->world_ranks.empty() ||
            cpu_domain->world_ranks.front() ==
                continuation->world_ranks.front())
        {
            throw std::invalid_argument(
                "Adversarial ExpertOverlay placement requires remote-first two-participant CPU ownership");
        }
        const int continuation_rank = continuation->world_ranks.front();

        MoERoutedExpertModelMetadata metadata = topologyOnlyMetadata();
        metadata.num_layers = layer_count;
        const int expert_count = metadata.num_experts;
        std::vector<std::vector<std::uint64_t>> reference_routes(
            static_cast<std::size_t>(layer_count),
            std::vector<std::uint64_t>(
                static_cast<std::size_t>(expert_count), 0u));
        for (int layer = 0; layer < layer_count; ++layer)
        {
            const auto routing = loadPyTorchSnapshot(
                "layer" + std::to_string(layer) +
                "_MOE_ROUTING_INDICES");
            if (routing.empty())
            {
                throw std::invalid_argument(
                    "Adversarial ExpertOverlay placement lacks reference routing for layer " +
                    std::to_string(layer));
            }
            for (const float routed_expert : routing)
            {
                const int expert = static_cast<int>(routed_expert);
                if (!std::isfinite(routed_expert) ||
                    routed_expert != static_cast<float>(expert) ||
                    expert < 0 || expert >= expert_count)
                {
                    throw std::invalid_argument(
                        "Adversarial ExpertOverlay placement found a malformed reference expert ID");
                }
                ++reference_routes[static_cast<std::size_t>(layer)]
                                  [static_cast<std::size_t>(expert)];
            }
        }

        /*
         * Ask the production planner for exact tier quotas, then replace only
         * membership. This keeps capacity/fallback semantics identical to a
         * normal by-ID start and avoids duplicating quota arithmetic here.
         */
        auto quota_request = requested;
        quota_request.residency_policy =
            RoutedExpertResidencyPolicy::StaticById;
        quota_request.placements.clear();
        const auto quota_plan = MoERoutedExpertPlacementPlanner::plan(
                                    quota_request, metadata)
                                    .planned_plan;

        requested.placements.clear();
        requested.placements.reserve(static_cast<std::size_t>(layer_count));
        for (int layer = 0; layer < layer_count; ++layer)
        {
            const auto &quota_seed = quota_plan.placements.at(
                static_cast<std::size_t>(layer));
            std::vector<int> tier_quotas(requested.routed_tiers.size(), 0);
            for (const int tier : quota_seed.routed_expert_tier)
                ++tier_quotas.at(static_cast<std::size_t>(tier));
            const int cpu_quota =
                tier_quotas.at(static_cast<std::size_t>(cpu_tier_index));
            const auto cpu_experts = selectAdversarialCpuExperts(
                reference_routes.at(static_cast<std::size_t>(layer)),
                layer,
                cpu_tier_index,
                cpu_quota,
                static_cast<int>(cpu_domain->participants.size()),
                requested.owner_order);

            RoutedExpertLayerPlacement placement;
            placement.layer = layer;
            placement.routed_expert_tier.assign(
                static_cast<std::size_t>(expert_count), -1);
            std::vector<int> remaining;
            remaining.reserve(
                static_cast<std::size_t>(expert_count - cpu_quota));
            for (int expert = 0; expert < expert_count; ++expert)
            {
                if (cpu_experts.at(static_cast<std::size_t>(expert)))
                {
                    placement.routed_expert_tier[
                        static_cast<std::size_t>(expert)] = cpu_tier_index;
                }
                else
                {
                    remaining.push_back(expert);
                }
            }

            /*
             * Among the remaining tiers, place the next-hottest experts on
             * progressively lower-priority tiers. This gives the three-tier
             * cell real multi-domain improvement work without interpreting
             * any tier label as a thermal role.
             */
            std::stable_sort(
                remaining.begin(), remaining.end(),
                [&](int lhs, int rhs)
                {
                    const auto lhs_routes = reference_routes[
                        static_cast<std::size_t>(layer)]
                        [static_cast<std::size_t>(lhs)];
                    const auto rhs_routes = reference_routes[
                        static_cast<std::size_t>(layer)]
                        [static_cast<std::size_t>(rhs)];
                    if (lhs_routes != rhs_routes)
                        return lhs_routes > rhs_routes;
                    return lhs < rhs;
                });
            std::vector<int> non_cpu_tiers;
            for (int tier = 0;
                 tier < static_cast<int>(requested.routed_tiers.size());
                 ++tier)
            {
                if (tier != cpu_tier_index)
                    non_cpu_tiers.push_back(tier);
            }
            std::stable_sort(
                non_cpu_tiers.begin(), non_cpu_tiers.end(),
                [&](int lhs, int rhs)
                {
                    const int lhs_priority = requested.routed_tiers[
                        static_cast<std::size_t>(lhs)]
                                                 .priority;
                    const int rhs_priority = requested.routed_tiers[
                        static_cast<std::size_t>(rhs)]
                                                 .priority;
                    if (lhs_priority != rhs_priority)
                        return lhs_priority > rhs_priority;
                    return lhs > rhs;
                });

            std::size_t cursor = 0u;
            for (const int tier : non_cpu_tiers)
            {
                const int quota =
                    tier_quotas.at(static_cast<std::size_t>(tier));
                for (int offset = 0; offset < quota; ++offset)
                {
                    if (cursor >= remaining.size())
                        throw std::logic_error(
                            "Adversarial ExpertOverlay placement exhausted non-CPU experts");
                    placement.routed_expert_tier[
                        static_cast<std::size_t>(remaining[cursor++])] = tier;
                }
            }
            if (cursor != remaining.size() ||
                std::find(
                    placement.routed_expert_tier.begin(),
                    placement.routed_expert_tier.end(), -1) !=
                    placement.routed_expert_tier.end())
            {
                throw std::logic_error(
                    "Adversarial ExpertOverlay placement did not preserve exact tier quotas");
            }
            requested.placements.push_back(std::move(placement));
        }

        const auto owner_map = MoEExpertOwnerMap::build(requested);
        std::uint64_t remote_routes = 0u;
        std::uint64_t local_routes = 0u;
        std::uint64_t strongest_remote = 0u;
        std::uint64_t strongest_local = 0u;
        std::int64_t best_layer_margin =
            std::numeric_limits<std::int64_t>::min();
        for (int layer = 0; layer < layer_count; ++layer)
        {
            std::uint64_t layer_remote_peak = 0u;
            std::uint64_t layer_local_peak = 0u;
            for (int expert = 0; expert < expert_count; ++expert)
            {
                const auto *owner = owner_map.ownerFor(layer, expert);
                if (!owner || owner->tier_idx != cpu_tier_index)
                    continue;
                const auto demand = reference_routes[
                    static_cast<std::size_t>(layer)]
                    [static_cast<std::size_t>(expert)];
                if (owner->owner_world_rank != continuation_rank)
                {
                    remote_routes += demand;
                    layer_remote_peak = std::max(layer_remote_peak, demand);
                }
                else
                {
                    local_routes += demand;
                    layer_local_peak = std::max(layer_local_peak, demand);
                }
            }
            strongest_remote = std::max(strongest_remote, layer_remote_peak);
            strongest_local = std::max(strongest_local, layer_local_peak);
            best_layer_margin = std::max(
                best_layer_margin,
                static_cast<std::int64_t>(layer_remote_peak) -
                    static_cast<std::int64_t>(layer_local_peak));
        }
        if (remote_routes == 0u || best_layer_margin <= 0)
        {
            throw std::logic_error(
                "Authenticated adversarial layout did not make a remote CPU expert hotter than its colocated peers");
        }

        LOG_INFO(
            "[Qwen3.5 MoE GraphNative] Authenticated adversarial layout: "
            << "layers=" << layer_count
            << " remote_cpu_routes=" << remote_routes
            << " local_cpu_routes=" << local_routes
            << " strongest_remote_expert=" << strongest_remote
            << " strongest_local_expert=" << strongest_local
            << " best_layer_margin=" << best_layer_margin
            << " owner_order="
            << routedExpertOwnerOrderToString(requested.owner_order));
        return requested;
    }

    bool setupPipeline()
    {
        try
        {
            /*
             * This is the same immutable declarative plan supplied to the
             * production application. OrchestrationRunner validates it against
             * the real GGUF metadata during initialization; the parity fixture
             * does not build, patch, or execute a graph itself.
             */
            /*
             * Supply the same model-independent request a user supplies at the
             * production boundary. OrchestrationRunner freezes exact per-layer
             * placements only after it has authenticated the real GGUF
             * metadata. Pre-planning from approximate test constants would make
             * the fixture, rather than production, the placement authority and
             * can silently give a 40-layer model a 94-layer ownership map.
             */
            auto requested = requestedPlan(topologyOnlyMetadata());
            if (isDynamicResidencyProductionTest() && topologyUsesCpu())
            {
                requested = remoteFirstNodeLocalOwnerPlan(
                    requested,
                    cluster_inventory_);
                requested = installReferenceAdversarialPlacements(
                    std::move(requested));
            }
            overlay_plan_ = std::make_shared<MoERoutedExpertPlacementPlan>(
                std::move(requested));
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[Qwen3.5 MoE GraphNative] Overlay plan construction failed: " << e.what());
            return false;
        }

        OrchestrationConfig orchestration = OrchestrationConfig::defaults();
        orchestration.model_path = config_.model_path;
        orchestration.max_seq_len = 4096;
        orchestration.batch_size = 1;
        orchestration.activation_precision = "fp32";
        orchestration.kv_cache_precision = "fp16";
        /*
         * The named MoE domains are the sole placement authority. During
         * normalization they become the production execution-domain inventory,
         * so supplying a second --device-map authority would be both ambiguous
         * and correctly rejected by ConfigValidator. The overlay execution-plan
         * resolver derives each rank's continuation/participant role directly
         * from the domains' world_ranks mappings.
         */
        orchestration.device_mode = DeviceAssignmentMode::AUTO;
        orchestration.tp_degree = 1;
        orchestration.pp_degree = 1;
        orchestration.deterministic = true;
        orchestration.moe_routed_expert_plan = overlay_plan_;
        /*
         * The residency policy and its runtime writer are two explicit axes.
         * StaticById is this cell's durable placement contract, so do not
         * inherit the production Dynamic default and accidentally construct a
         * migration authority behind a test that later proves immobility.
         * Dynamic cells opt back in below and exercise the complete physical
         * publication path.
         */
        orchestration.moe_rebalance.mode =
            MoERebalanceRuntimeMode::Off;
        if (isDynamicResidencyProductionTest())
        {
            /*
             * Eight committed decode rows form one real histogram epoch. This
             * is deliberately workload policy, not a test-side histogram: the
             * production local-expert stages remain the only count producers.
             */
            orchestration.moe_rebalance.mode =
                MoERebalanceRuntimeMode::Dynamic;
            orchestration.moe_rebalance.window_size = 8;
            orchestration.moe_rebalance.max_window_size = 8;
            orchestration.moe_rebalance.window_growth_factor = 1.0f;
            /*
             * Select two independent closed cycles per wave in this scenario.
             * The value is deliberately supplied through the same public
             * runtime knob as production: it is neither an architectural
             * constant nor tied to the two-epoch convergence assertion below.
             * Config parser coverage separately exercises other positive
             * values, while this real-model cell proves concurrent staging.
             */
            orchestration.moe_rebalance.migration_max_cycles_per_wave = 2;
            /*
             * Residency persists across requests. The complete directed-lane
             * calibration on the two-rank GPU/CPU cell puts its remote-lane
             * break-even above the colocated path (about 18.1k routed tokens
             * on the reference host). A 64k model-server lifetime lets thermal
             * benefit dominate participant locality in this adversarial proof
             * while the measured payoff gate still
             * rejects any move whose projected net benefit is non-positive.
             */
            orchestration.moe_rebalance.migration_payoff_horizon_tokens =
                65'536;
            orchestration.moe_rebalance.release_raw_expert_weights = false;
        }

        const auto snapshot_setup_mode =
            isDynamicResidencyProductionTest()
                ? ParitySnapshotSetupMode::Disabled
                : ParitySnapshotSetupMode::Enabled;
        if (!setupOrchestrationRunner(
                orchestration,
                nullptr,
                snapshot_setup_mode))
            return false;

        if (!orch_runner_)
        {
            LOG_ERROR("[Qwen3.5 MoE GraphNative] Production runner disappeared after setup");
            return false;
        }

        try
        {
            /*
             * Configure both rank-local runners before enabling the command
             * loop.  Once rank zero enters coordinated mode, this public API
             * deliberately broadcasts SET_SAMPLING to workers; doing that
             * before the workers are listening would invert the production
             * command protocol.  The pre-loop setting is therefore the normal
             * request-admission configuration boundary for this test topology.
             */
            orch_runner_->setSamplingParams(referenceGreedySamplingPolicy());
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[Qwen3.5 MoE GraphNative] Failed to install Hugging Face "
                      "greedy sampling policy: " << e.what());
            return false;
        }

        return true;
    }

    /** @brief Sum one process-local ExpertOverlay residency counter. */
    static double residencyCounter(
        const std::vector<PerfStatRecord> &records,
        const std::string &name)
    {
        double total = 0.0;
        for (const auto &record : records)
        {
            if (record.kind == PerfStatRecord::Kind::Counter &&
                record.domain == "moe_overlay_residency" &&
                record.name == name)
            {
                total += record.value;
            }
        }
        return total;
    }

    /** @brief Snapshot and sum one process-local residency counter. */
    double localResidencyCounter(const std::string &name) const
    {
        return residencyCounter(
            PerfStatsCollector::snapshot({"moe_overlay_residency"}),
            name);
    }

    /** @return Exact local committed-wave count, or throw on malformed data. */
    std::uint64_t localCommittedWaveCount() const
    {
        const double value = localResidencyCounter("committed_waves");
        if (value < 0.0 || !std::isfinite(value) ||
            value > static_cast<double>(
                        std::numeric_limits<std::uint64_t>::max()))
        {
            throw std::runtime_error(
                "ExpertOverlay committed-wave PerfStats value is invalid");
        }
        const auto count = static_cast<std::uint64_t>(value);
        if (static_cast<double>(count) != value)
        {
            throw std::runtime_error(
                "ExpertOverlay committed-wave PerfStats value is not integral");
        }
        return count;
    }

    /**
     * @brief Measure stable post-publication requests on the live production runner.
     *
     * Two untimed requests absorb cache/capture and just-published retirement
     * effects. Each measured interval brackets the production committed-wave
     * counter. A publication inside the interval makes its epoch ambiguous, so
     * that observation is discarded rather than assigned to either layout.
     * Maintenance is still notified after every token exactly as in serving;
     * the measurement never pauses or joins the background worker.
     */
    bool collectConvergedInferenceTimings()
    {
        if (!requiresObservedConvergenceSpeedup())
            return true;

        constexpr int kWarmupRequests = 2;
        constexpr int kMeasuredPrefillRequests = 6;
        constexpr int kDecodeForwardsPerRequest = 5;
        constexpr int kMaximumRequests = 16;
        constexpr std::size_t kRequiredDecodeSamples =
            static_cast<std::size_t>(kMeasuredPrefillRequests) *
            kDecodeForwardsPerRequest;

        if (localCommittedWaveCount() < 2u)
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Convergence timing began before two residency epochs published");
            return false;
        }

        const std::vector<int32_t> prompt(
            config_.token_ids.begin(), config_.token_ids.end());
        for (int request = 0; request < kMaximumRequests; ++request)
        {
            activeClearCache();

            const std::uint64_t prefill_waves_before =
                localCommittedWaveCount();
            const auto prefill_start = std::chrono::steady_clock::now();
            if (!orch_runner_->prefill(prompt))
            {
                LOG_ERROR(
                    "[Qwen3.5 MoE GraphNative] Converged timing prefill failed: "
                    << orch_runner_->lastError());
                return false;
            }
            const std::uint64_t prefill_ns =
                elapsedNanoseconds(prefill_start);
            const std::uint64_t prefill_waves_after =
                localCommittedWaveCount();
            if (!orch_runner_->maybeApplyMoERebalance())
            {
                LOG_ERROR(
                    "[Qwen3.5 MoE GraphNative] Converged timing prefill maintenance notification failed: "
                    << orch_runner_->lastError());
                return false;
            }

            if (request >= kWarmupRequests &&
                prefill_waves_before == prefill_waves_after &&
                prefill_waves_before >= 2u &&
                convergence_timings_.converged_prefill_ns.size() <
                    static_cast<std::size_t>(kMeasuredPrefillRequests))
            {
                convergence_timings_.converged_prefill_ns.push_back(
                    prefill_ns);
                convergence_timings_.converged_prefill_epochs.push_back(
                    prefill_waves_before + 1u);
            }

            /* The first call consumes already-produced prefill logits. */
            GenerationResult boundary_sample = orch_runner_->decodeStep();
            if (!boundary_sample.success() || boundary_sample.tokens.empty())
            {
                LOG_ERROR(
                    "[Qwen3.5 MoE GraphNative] Converged timing prefill-boundary sample failed: "
                    << boundary_sample.error);
                return false;
            }
            if (!orch_runner_->maybeApplyMoERebalance())
                return false;

            for (int step = 0; step < kDecodeForwardsPerRequest; ++step)
            {
                const std::uint64_t decode_waves_before =
                    localCommittedWaveCount();
                const auto decode_start = std::chrono::steady_clock::now();
                GenerationResult decoded = orch_runner_->decodeStep();
                const std::uint64_t decode_ns =
                    elapsedNanoseconds(decode_start);
                const std::uint64_t decode_waves_after =
                    localCommittedWaveCount();
                if (!decoded.success() || decoded.tokens.empty())
                {
                    LOG_ERROR(
                        "[Qwen3.5 MoE GraphNative] Converged timing decode failed: "
                        << decoded.error);
                    return false;
                }
                if (!orch_runner_->maybeApplyMoERebalance())
                    return false;

                if (request >= kWarmupRequests &&
                    decode_waves_before == decode_waves_after &&
                    decode_waves_before >= 2u &&
                    convergence_timings_.converged_decode_ns.size() <
                        kRequiredDecodeSamples)
                {
                    convergence_timings_.converged_decode_ns.push_back(
                        decode_ns);
                    convergence_timings_.converged_decode_epochs.push_back(
                        decode_waves_before + 1u);
                }
            }

            if (convergence_timings_.converged_prefill_ns.size() >=
                    static_cast<std::size_t>(kMeasuredPrefillRequests) &&
                convergence_timings_.converged_decode_ns.size() >=
                    kRequiredDecodeSamples)
            {
                return true;
            }
        }

        LOG_ERROR(
            "[Qwen3.5 MoE GraphNative] Could not collect enough epoch-stable converged inference timings: prefill="
            << convergence_timings_.converged_prefill_ns.size()
            << " decode="
            << convergence_timings_.converged_decode_ns.size());
        return false;
    }

    /** @return Stable diagnostic spelling for the active priority topology. */
    std::string convergenceTopologyName() const
    {
        switch (activeTopology())
        {
        case OverlayTopology::CudaRocmCpu:
            return "CUDA+ROCm+CPU";
        case OverlayTopology::CudaCpu:
            return "CUDA+CPU";
        case OverlayTopology::RocmCpu:
            return "ROCm+CPU";
        case OverlayTopology::CudaRocm:
            return "CUDA+ROCm";
        }
        throw std::logic_error("Unhandled convergence timing topology");
    }

    /**
     * @brief Assert and serialize the real before/after convergence evidence.
     *
     * A two-percent floor is deliberately larger than timer quantization and
     * ordinary run-to-run jitter on this host. The median makes the gate robust
     * to background OS activity while retaining a directional performance
     * requirement for both time-to-first-token prefill and steady decode.
     */
    void assertAndWriteObservedConvergenceSpeedup()
    {
        if (!requiresObservedConvergenceSpeedup() || !isRootParityRank())
            return;

        ASSERT_GE(convergence_timings_.baseline_prefill_ns.size(), 2u);
        ASSERT_GE(convergence_timings_.baseline_decode_ns.size(), 2u);
        ASSERT_GE(convergence_timings_.converged_prefill_ns.size(), 6u);
        ASSERT_GE(convergence_timings_.converged_decode_ns.size(), 30u);

        const std::uint64_t baseline_prefill = medianNanoseconds(
            convergence_timings_.baseline_prefill_ns);
        const std::uint64_t baseline_decode = medianNanoseconds(
            convergence_timings_.baseline_decode_ns);
        const std::uint64_t converged_prefill = medianNanoseconds(
            convergence_timings_.converged_prefill_ns);
        const std::uint64_t converged_decode = medianNanoseconds(
            convergence_timings_.converged_decode_ns);
        constexpr long double kMaximumConvergedRatio = 0.98L;
        const bool prefill_passed =
            static_cast<long double>(converged_prefill) <=
            static_cast<long double>(baseline_prefill) *
                kMaximumConvergedRatio;
        const bool decode_passed =
            static_cast<long double>(converged_decode) <=
            static_cast<long double>(baseline_decode) *
                kMaximumConvergedRatio;
        const auto improvement = [](std::uint64_t baseline,
                                    std::uint64_t converged)
        {
            return 100.0L *
                   (static_cast<long double>(baseline) -
                    static_cast<long double>(converged)) /
                   static_cast<long double>(baseline);
        };
        const long double prefill_improvement =
            improvement(baseline_prefill, converged_prefill);
        const long double decode_improvement =
            improvement(baseline_decode, converged_decode);

        const auto path =
            ensureResultsDir() / "expert_overlay_convergence.csv";
        std::ofstream csv(path, std::ios::trunc);
        ASSERT_TRUE(csv.is_open())
            << "Cannot create observed ExpertOverlay convergence CSV at "
            << path;
        csv << "backend,topology,phase,cohort,sample,residency_epoch,latency_ns,baseline_median_ns,converged_median_ns,improvement_percent,passed\n";
        csv << std::fixed << std::setprecision(4);
        const auto write_samples = [&](const char *phase,
                                       const char *cohort,
                                       const std::vector<std::uint64_t> &samples,
                                       const std::vector<std::uint64_t> &epochs,
                                       std::uint64_t baseline_median,
                                       std::uint64_t converged_median,
                                       long double improvement_percent,
                                       bool passed)
        {
            for (std::size_t index = 0; index < samples.size(); ++index)
            {
                const std::uint64_t epoch =
                    epochs.empty() ? 1u : epochs.at(index);
                csv << getBackendName() << ','
                    << convergenceTopologyName() << ','
                    << phase << ','
                    << cohort << ','
                    << index << ','
                    << epoch << ','
                    << samples[index] << ','
                    << baseline_median << ','
                    << converged_median << ','
                    << static_cast<double>(improvement_percent) << ','
                    << (passed ? "true" : "false") << '\n';
            }
        };
        write_samples(
            "prefill",
            "initial_epoch",
            convergence_timings_.baseline_prefill_ns,
            {},
            baseline_prefill,
            converged_prefill,
            prefill_improvement,
            prefill_passed);
        write_samples(
            "prefill",
            "converged_epoch",
            convergence_timings_.converged_prefill_ns,
            convergence_timings_.converged_prefill_epochs,
            baseline_prefill,
            converged_prefill,
            prefill_improvement,
            prefill_passed);
        write_samples(
            "decode",
            "initial_epoch",
            convergence_timings_.baseline_decode_ns,
            {},
            baseline_decode,
            converged_decode,
            decode_improvement,
            decode_passed);
        write_samples(
            "decode",
            "converged_epoch",
            convergence_timings_.converged_decode_ns,
            convergence_timings_.converged_decode_epochs,
            baseline_decode,
            converged_decode,
            decode_improvement,
            decode_passed);
        csv.flush();
        ASSERT_TRUE(csv.good())
            << "Failed to write observed ExpertOverlay convergence CSV at "
            << path;

        LOG_INFO(
            "[Qwen3.5 MoE GraphNative] Observed convergence performance: topology="
            << convergenceTopologyName()
            << " prefill_initial_ns=" << baseline_prefill
            << " prefill_converged_ns=" << converged_prefill
            << " prefill_improvement_percent="
            << static_cast<double>(prefill_improvement)
            << " decode_initial_ns=" << baseline_decode
            << " decode_converged_ns=" << converged_decode
            << " decode_improvement_percent="
            << static_cast<double>(decode_improvement));
        EXPECT_TRUE(prefill_passed)
            << "Priority convergence did not improve observed prefill latency by at least 2%";
        EXPECT_TRUE(decode_passed)
            << "Priority convergence did not improve observed decode latency by at least 2%";
    }

    /** @brief Complete identity of one production calibration probe request. */
    using CalibrationProbeIdentity = std::array<std::string, 6>;

    /** @brief Test admission demand reconstructed from production PerfStats. */
    struct CalibrationProbeDemand
    {
        CalibrationProbeIdentity identity; ///< Arm/sample correlation key.
        std::uint64_t sequence = 0;         ///< Monotonic controller attempt.
        ExpertHistogramSource source =
            ExpertHistogramSource::SyntheticTest; ///< Required live phase.
        bool concurrent = false; ///< Whether movement must overlap the phase.
    };

    /**
     * @brief Extract the immutable correlation key from a probe event.
     *
     * @param record One arm or consumed-sample PerfStats record.
     * @return Canonical identity, or no value when required tags are missing.
     */
    static std::optional<CalibrationProbeIdentity>
    calibrationProbeIdentity(const PerfStatRecord &record)
    {
        constexpr std::array<const char *, 6> kIdentityTags{
            "calibration_sequence",
            "mode",
            "source",
            "layer",
            "source_participant",
            "destination_participant",
        };
        CalibrationProbeIdentity identity;
        for (std::size_t index = 0; index < kIdentityTags.size(); ++index)
        {
            const auto tag = record.tags.find(kIdentityTags[index]);
            if (tag == record.tags.end())
                return std::nullopt;
            identity[index] = tag->second;
        }
        return identity;
    }

    /**
     * @brief Return the newest armed probe without a matching consumed sample.
     *
     * Older canceled attempts may intentionally lack samples. Selecting the
     * highest sequence means a retry supersedes them, while the baseline and
     * concurrent mode tag disambiguates the two requests within one attempt.
     * The result controls only which ordinary production request is admitted;
     * the maintenance worker remains the sole calibration-state authority.
     */
    std::optional<CalibrationProbeDemand>
    outstandingCalibrationProbe(
        const std::vector<PerfStatRecord> &records) const
    {
        std::set<CalibrationProbeIdentity> sampled;
        for (const auto &record : records)
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "moe_overlay_residency" ||
                record.name != "economy_calibration_probe_samples")
            {
                continue;
            }
            if (const auto identity = calibrationProbeIdentity(record))
                sampled.insert(*identity);
        }

        std::optional<CalibrationProbeDemand> newest;
        for (const auto &record : records)
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "moe_overlay_residency" ||
                record.name != "economy_calibration_probe_arms")
            {
                continue;
            }

            const auto identity = calibrationProbeIdentity(record);
            if (!identity || sampled.contains(*identity))
                continue;

            const auto sequence_tag =
                record.tags.find("calibration_sequence");
            const auto source_tag = record.tags.find("source");
            const auto mode_tag = record.tags.find("mode");
            if (sequence_tag == record.tags.end() ||
                source_tag == record.tags.end() ||
                mode_tag == record.tags.end())
            {
                continue;
            }

            uint64_t sequence = 0;
            const char *const begin = sequence_tag->second.data();
            const char *const end = begin + sequence_tag->second.size();
            const auto parsed = std::from_chars(begin, end, sequence);
            if (parsed.ec != std::errc{} || parsed.ptr != end)
            {
                continue;
            }

            std::optional<ExpertHistogramSource> parsed_source;
            if (source_tag->second == "prefill")
                parsed_source = ExpertHistogramSource::PrefillChunk;
            else if (source_tag->second == "decode")
                parsed_source = ExpertHistogramSource::DecodeToken;
            else if (source_tag->second == "grouped_verifier")
                parsed_source = ExpertHistogramSource::GroupedVerifier;
            if (!parsed_source)
                continue;

            const bool concurrent =
                mode_tag->second == "concurrent_movement";
            if (!concurrent && mode_tag->second != "baseline")
                continue;
            if (newest &&
                (sequence < newest->sequence ||
                 (sequence == newest->sequence &&
                  !concurrent && newest->concurrent)))
            {
                continue;
            }
            newest = CalibrationProbeDemand{
                .identity = *identity,
                .sequence = sequence,
                .source = *parsed_source,
                .concurrent = concurrent,
            };
        }
        return newest;
    }

    /**
     * @brief Feed bounded real requests until distributed residency actually moves.
     *
     * The maintenance controller alternates baseline and concurrent samples for
     * decode and prefill. PerfStats identifies the phase of its newest armed
     * probe so this workload driver admits matching real requests rather than
     * replaying unrelated rows. The background worker still owns timing,
     * staging, transfer, overlap validation, certification, and publication;
     * no histogram, timing, placement, or completion value is injected here.
     *
     * @return True after this rank observes certification, two committed
     *         improving epochs, and a cross-rank edge; false after bounded
     *         traffic is exhausted.
     */
    bool driveDynamicResidencyToDistributedMigration()
    {
        if (!isDynamicResidencyProductionTest())
            return true;

        constexpr int kMaximumCertifiedRequests = 24;
        constexpr int kDecodeStepsPerCertifiedRequest = 9;
        constexpr double kMinimumCommittedWaves = 2.0;
        constexpr double kMaximumRejectedProposals = 8.0;
        constexpr auto kProbeProgressTimeout = std::chrono::seconds(30);
        constexpr auto kProbePollPeriod = std::chrono::milliseconds(10);

        const double expected_pairs_value =
            localResidencyCounter("economy_calibration_expected_pairs");
        const auto expected_pairs =
            static_cast<std::uint64_t>(expected_pairs_value);
        if (expected_pairs == 0 ||
            static_cast<double>(expected_pairs) != expected_pairs_value ||
            expected_pairs >
                (std::numeric_limits<std::uint64_t>::max() - 32u) / 8u)
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Dynamic calibration published an "
                "invalid expected-pair plan: " << expected_pairs_value);
            return false;
        }
        /*
         * Every accepted pair needs a baseline and concurrent interval. The
         * remaining factor covers decode admission prefills and exact-overlap
         * retries. The bound scales with the production-authored corpus instead
         * of encoding a topology-specific magic request count.
         */
        const std::uint64_t maximum_calibration_forwards =
            expected_pairs * 8u + 32u;

        const std::vector<int32_t> prompt(
            config_.token_ids.begin(), config_.token_ids.end());
        const int vocabulary_size = orch_runner_->vocabSize();
        if (vocabulary_size <= 4'096)
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Dynamic routing corpus resolved an "
                "invalid vocabulary size: " << vocabulary_size);
            return false;
        }
        const auto routingCorpusPrompt = [&](int request_index)
        {
            if (request_index == 0)
                return prompt;

            /*
             * Exercise a broad, deterministic set of real embedding rows. The
             * first request retains the authenticated parity prompt; subsequent
             * entries are stable valid token sequences, with each sequence
             * repeated twice so smoothing sees a sustained distribution. This
             * is ordinary model traffic: only local-expert stages produce the
             * histogram consumed by the residency authority.
             */
            const std::uint64_t corpus_index =
                1u + static_cast<std::uint64_t>(request_index - 1) / 2u;
            std::uint64_t state =
                0x9e3779b97f4a7c15ULL ^
                (corpus_index * 0xbf58476d1ce4e5b9ULL);
            const std::uint64_t usable_vocabulary =
                static_cast<std::uint64_t>(vocabulary_size - 2'048);
            std::vector<int32_t> result(prompt.size(), 0);
            for (size_t token_offset = 0;
                 token_offset < result.size(); ++token_offset)
            {
                state += 0x9e3779b97f4a7c15ULL;
                std::uint64_t mixed = state;
                mixed = (mixed ^ (mixed >> 30u)) *
                        0xbf58476d1ce4e5b9ULL;
                mixed = (mixed ^ (mixed >> 27u)) *
                        0x94d049bb133111ebULL;
                mixed ^= mixed >> 31u;
                result[token_offset] = static_cast<int32_t>(
                    256u + mixed % usable_vocabulary);
            }
            return result;
        };
        bool request_active = false;
        std::uint64_t calibration_forwards = 0;
        int certified_requests = 0;
        int service_coverage_request = 1;
        int service_coverage_decode_steps =
            kDecodeStepsPerCertifiedRequest;
        std::optional<CalibrationProbeIdentity> last_served_probe;
        std::optional<CalibrationProbeIdentity> last_observed_probe;
        double last_accepted_pairs =
            localResidencyCounter("economy_calibration_pairs_accepted");
        auto last_progress = std::chrono::steady_clock::now();

        const auto wakeMaintenance = [&](const char *phase)
        {
            if (orch_runner_->maybeApplyMoERebalance())
                return true;
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Dynamic maintenance wake after "
                << phase << " failed: " << orch_runner_->lastError());
            return false;
        };
        const auto runPrefill = [&](const std::vector<int32_t> &tokens,
                                    std::uint64_t *elapsed_ns = nullptr)
        {
            activeClearSnapshots();
            activeClearCache();
            const auto start = std::chrono::steady_clock::now();
            if (!orch_runner_->prefill(tokens))
            {
                LOG_ERROR(
                    "[Qwen3.5 MoE GraphNative] Dynamic calibration prefill failed: "
                    << orch_runner_->lastError());
                return false;
            }
            if (elapsed_ns)
                *elapsed_ns = elapsedNanoseconds(start);
            request_active = true;
            return wakeMaintenance("prefill");
        };
        const auto runDecode = [&](std::uint64_t *elapsed_ns = nullptr)
        {
            const auto start = std::chrono::steady_clock::now();
            const GenerationResult generated = orch_runner_->decodeStep();
            if (!generated.success() || generated.tokens.empty())
            {
                LOG_ERROR(
                    "[Qwen3.5 MoE GraphNative] Dynamic calibration decode failed: "
                    << generated.error);
                return std::optional<bool>{};
            }
            if (elapsed_ns)
                *elapsed_ns = elapsedNanoseconds(start);
            if (!wakeMaintenance("decode"))
                return std::optional<bool>{};
            request_active = !generated.is_complete;
            return std::optional<bool>{generated.is_complete};
        };

        /*
         * Admit the first ordinary request before inspecting probe demand.
         * This mirrors a newly started server: calibration consumes live
         * traffic, so there is deliberately no synthetic setup sample. It also
         * gives the just-created background worker a full inference interval
         * in which to publish its first arm before the test begins taking
         * comparatively expensive structured PerfStats snapshots.
         */
        if (!runPrefill(prompt))
            return false;
        ++calibration_forwards;
        last_progress = std::chrono::steady_clock::now();

        while (calibration_forwards < maximum_calibration_forwards)
        {
            /*
             * One immutable snapshot supplies the entire scheduling decision.
             * Repeatedly copying the structured corpus for each scalar can
             * otherwise contend with the background producer being measured.
             */
            const auto residency_records =
                PerfStatsCollector::snapshot({"moe_overlay_residency"});
            if (residencyCounter(
                    residency_records,
                    "economy_certification_complete") > 0.0)
            {
                break;
            }
            const auto requested_probe =
                outstandingCalibrationProbe(residency_records);
            const double accepted_pairs =
                residencyCounter(
                    residency_records,
                    "economy_calibration_pairs_accepted");
            const bool movement_calibration_complete =
                residencyCounter(
                    residency_records,
                    "economy_calibration_complete") > 0.0;
            const bool probe_changed =
                requested_probe.has_value() != last_observed_probe.has_value() ||
                (requested_probe &&
                 requested_probe->identity != *last_observed_probe);
            if (probe_changed || accepted_pairs != last_accepted_pairs)
            {
                last_observed_probe = requested_probe
                                          ? std::optional<CalibrationProbeIdentity>{
                                                requested_probe->identity}
                                          : std::nullopt;
                last_accepted_pairs = accepted_pairs;
                last_progress = std::chrono::steady_clock::now();
            }

            if (!requested_probe && movement_calibration_complete)
            {
                /*
                 * Movement calibration is complete, but live sparse routing
                 * may not yet have sampled every participant and exact layer
                 * class in both decode and prefill. Continue with broad,
                 * ordinary model traffic while the all-rank readiness vote is
                 * false. No timing or placement value is injected: the same
                 * prepared expert stages publish the missing measurements.
                 */
                if (!request_active ||
                    service_coverage_decode_steps >=
                        kDecodeStepsPerCertifiedRequest)
                {
                    if (!runPrefill(
                            routingCorpusPrompt(
                                service_coverage_request++)))
                    {
                        return false;
                    }
                    service_coverage_decode_steps = 0;
                }
                else
                {
                    const auto complete = runDecode();
                    if (!complete)
                        return false;
                    ++service_coverage_decode_steps;
                    if (*complete)
                    {
                        service_coverage_decode_steps =
                            kDecodeStepsPerCertifiedRequest;
                    }
                }
                ++calibration_forwards;
                last_progress = std::chrono::steady_clock::now();
                continue;
            }

            if (!requested_probe ||
                (last_served_probe &&
                 requested_probe->identity == *last_served_probe))
            {
                if (!wakeMaintenance("calibration probe progress"))
                    return false;
                if (std::chrono::steady_clock::now() - last_progress >
                    kProbeProgressTimeout)
                {
                    const double arm_events = residencyCounter(
                        residency_records,
                        "economy_calibration_probe_arms");
                    const double sample_events = residencyCounter(
                        residency_records,
                        "economy_calibration_probe_samples");
                    LOG_ERROR(
                        "[Qwen3.5 MoE GraphNative] Dynamic calibration made no "
                        "probe or accepted-pair progress for 30 seconds: "
                        "forwards=" << calibration_forwards
                        << " arm_events=" << arm_events
                        << " sample_events=" << sample_events
                        << " parsed_outstanding="
                        << (requested_probe ? "true" : "false")
                        << " last_served="
                        << (last_served_probe ? "true" : "false")
                        << (requested_probe
                                ? " requested_sequence=" +
                                      std::to_string(
                                          requested_probe->sequence) +
                                      " requested_source=" +
                                      std::string(
                                          requested_probe->source ==
                                                  ExpertHistogramSource::DecodeToken
                                              ? "decode"
                                              : (requested_probe->source ==
                                                         ExpertHistogramSource::PrefillChunk
                                                     ? "prefill"
                                                     : "grouped_verifier"))
                                : std::string{})
                        << "\n"
                        << PerfStatsCollector::summaryString(
                               {"moe_overlay_residency"}));
                    return false;
                }
                std::this_thread::sleep_for(kProbePollPeriod);
                continue;
            }

            /*
             * A decode probe needs an admitted request first. Its setup prefill
             * is still a real production call and may satisfy a newly armed
             * prefill probe by the time the background worker observes it.
             */
            if (requested_probe->source ==
                ExpertHistogramSource::PrefillChunk)
            {
                std::uint64_t elapsed_ns = 0;
                if (!runPrefill(
                        prompt,
                        requested_probe->concurrent
                            ? nullptr
                            : &elapsed_ns))
                    return false;
                if (!requested_probe->concurrent && elapsed_ns > 0)
                {
                    convergence_timings_.baseline_prefill_ns.push_back(
                        elapsed_ns);
                }
                last_served_probe = requested_probe->identity;
                ++calibration_forwards;
                continue;
            }

            if (requested_probe->source ==
                ExpertHistogramSource::GroupedVerifier)
            {
                LOG_ERROR(
                    "[Qwen3.5 MoE GraphNative] Non-MTP calibration armed an "
                    "unreachable grouped-verifier probe");
                return false;
            }
            if (!request_active)
            {
                /* Admission does not satisfy the still-outstanding decode arm. */
                if (!runPrefill(prompt))
                    return false;
                ++calibration_forwards;
                continue;
            }
            /*
             * The first decodeStep after prefill samples already-produced
             * prefill logits; it does not execute a DecodeToken model phase.
             * Keep the probe outstanding across that sampling-only boundary so
             * the following ordinary decode forward can claim it. Treating the
             * returned token as a probe sample was the causal scheduler bug:
             * the controller correctly remained in AwaitBaseline forever.
             */
            const bool executes_decode_phase =
                !activePrefixStateProbe().prefill_logits_ready;
            std::uint64_t elapsed_ns = 0;
            if (!runDecode(
                    !requested_probe->concurrent && executes_decode_phase
                        ? &elapsed_ns
                        : nullptr))
                return false;
            if (executes_decode_phase)
            {
                last_served_probe = requested_probe->identity;
                if (!requested_probe->concurrent && elapsed_ns > 0)
                {
                    convergence_timings_.baseline_decode_ns.push_back(
                        elapsed_ns);
                }
            }
            ++calibration_forwards;
        }

        if (localResidencyCounter("economy_certification_complete") == 0.0)
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Dynamic economy calibration did not "
                "complete after " << calibration_forwards
                << " phase-directed production forwards\n"
                << PerfStatsCollector::summaryString(
                       {"moe_overlay_residency"}));
            return false;
        }

        double last_committed_migrations =
            localResidencyCounter("committed_expert_migrations");
        double proposals_at_last_migration =
            localResidencyCounter("economy_proposals");

        for (; certified_requests < kMaximumCertifiedRequests;
             ++certified_requests)
        {
            /*
             * Calibration uses broad traffic only to measure every declared
             * service coordinate. Once certified, residency must optimize the
             * same stationary workload used by the before/after latency gate.
             * Changing the prompt every epoch trains for a different routing
             * distribution and cannot prove convergence for any one tenant.
             */
            if (!runPrefill(prompt))
                return false;
            for (int step = 0;
                 step < kDecodeStepsPerCertifiedRequest;
                 ++step)
            {
                const auto complete = runDecode();
                if (!complete)
                    return false;
                if (*complete)
                    break;
            }

            const auto movement_records =
                PerfStatsCollector::snapshot({"moe_overlay_residency"});
            const double committed_migrations = residencyCounter(
                movement_records,
                "committed_expert_migrations");
            const double committed_waves = residencyCounter(
                movement_records,
                "committed_waves");
            const double cross_rank_migrations = residencyCounter(
                movement_records,
                "cross_rank_migrations");
            const double proposals = residencyCounter(
                movement_records,
                "economy_proposals");

            if (committed_migrations > last_committed_migrations)
            {
                /*
                 * A newly published epoch resets the rejected-proposal budget.
                 * Later windows may select another physical participant even
                 * though the deterministic token workload is unchanged.
                 */
                last_committed_migrations = committed_migrations;
                proposals_at_last_migration = proposals;
            }

            /*
             * Every rank derives and publishes the same transaction, so the
             * coordinator's local counters prove distributed completion. A
             * NodeTP CPU tier can legally make the first GPU/CPU swap with
             * its colocated rank; keep driving production windows until an edge
             * also crosses MPI ranks instead of weakening the final assertion.
             */
            if (committed_waves >= kMinimumCommittedWaves &&
                committed_migrations > 0.0 &&
                cross_rank_migrations > 0.0)
            {
                return true;
            }
            if (proposals - proposals_at_last_migration >=
                kMaximumRejectedProposals)
            {
                /*
                 * The request is deterministic, so repeated certified windows
                 * converge toward the same smoothed distribution. Once eight
                 * proposals after the latest commit add no movement, further
                 * replay cannot expose another physical participant and only
                 * obscures the causal evidence with redundant rows.
                 */
                break;
            }
        }

        LOG_ERROR(
            "[Qwen3.5 MoE GraphNative] Dynamic residency did not produce two profitable publication epochs and the required cross-rank migration after "
            << calibration_forwards << " calibration forwards and "
            << certified_requests << " certified histogram requests\n"
            << PerfStatsCollector::summaryString({"moe_overlay_residency"}));
        return false;
    }

    /**
     * @brief Fold and validate static immobility or dynamic movement evidence.
     *
     * Dynamic movement must be a capacity-preserving promotion/demotion cycle
     * across the two distinct domains, MPI ranks, and GPU backends. Static cells
     * must publish their typed immobility check and no positive movement edge.
     */
    void assertResidencyMovementEvidence() const
    {
        enum Evidence : size_t
        {
            DomainEnabled,
            StaticChecks,
            CalibrationComplete,
            CertificationComplete,
            AuthorityCertifications,
            MaintenanceCertifications,
            CommittedWaves,
            CommittedMigrations,
            Promotions,
            Demotions,
            CrossDomain,
            CrossRank,
            CrossBackend,
            EstimatedBytes,
            BackgroundNotifications,
            StageFailures,
            CommitFailures,
            FatalFailures,
            BlockingInferenceViolations,
            ImprovingEpochs,
            ServiceGainEvidenceViolations,
            NetBenefitEvidenceViolations,
            PriorityDirectionViolations,
            ConfiguredCycleCapRecords,
            CycleCapViolations,
            FullCycleCapWaves,
            AdoptedInitialSlots,
            BootstrapSlotsRecycled,
            PhysicalWavesPrepared,
            PhysicalEvidenceViolations,
            EvidenceCount,
        };

        std::array<uint64_t, EvidenceCount> local{};
        std::set<uint64_t> committed_epochs;
        std::set<uint64_t> positive_service_gain_epochs;
        std::set<uint64_t> positive_net_benefit_epochs;
        local[DomainEnabled] =
            PerfStatsCollector::isDomainEnabled("moe_overlay_residency")
                ? 1u
                : 0u;
        const auto addValue = [&local](Evidence index, double value)
        {
            if (value > 0.0)
                local[index] += static_cast<uint64_t>(value);
        };
        const auto parse_u64 = [](const std::string &text,
                                  uint64_t &value)
        {
            const char *const begin = text.data();
            const char *const end = begin + text.size();
            const auto parsed = std::from_chars(begin, end, value);
            return parsed.ec == std::errc{} && parsed.ptr == end;
        };
        const auto parse_int = [](const std::string &text, int &value)
        {
            const char *const begin = text.data();
            const char *const end = begin + text.size();
            const auto parsed = std::from_chars(begin, end, value);
            return parsed.ec == std::errc{} && parsed.ptr == end;
        };
        for (const auto &record :
             PerfStatsCollector::snapshot({"moe_overlay_residency"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "moe_overlay_residency")
            {
                continue;
            }

            if (record.name == "static_no_movement_checks")
                addValue(StaticChecks, record.value);
            else if (record.name == "production_maintenance_composed")
            {
                addValue(ConfiguredCycleCapRecords, record.value);
                const auto cap_tag =
                    record.tags.find("max_concurrent_cycles");
                uint64_t cap = 0;
                const uint64_t expected_cap =
                    isDynamicResidencyProductionTest() ? 2u : 1u;
                if (cap_tag == record.tags.end() ||
                    !parse_u64(cap_tag->second, cap) ||
                    cap != expected_cap)
                {
                    ++local[CycleCapViolations];
                }
            }
            else if (record.name == "physical_fabrics_materialized")
            {
                const auto adopted_tag =
                    record.tags.find("adopted_initial_slots");
                uint64_t adopted = 0;
                if (adopted_tag == record.tags.end() ||
                    !parse_u64(adopted_tag->second, adopted))
                {
                    ++local[PhysicalEvidenceViolations];
                }
                else
                {
                    local[AdoptedInitialSlots] += adopted;
                }
            }
            else if (record.name == "economy_calibration_complete")
                addValue(CalibrationComplete, record.value);
            else if (record.name == "economy_certification_complete")
                addValue(CertificationComplete, record.value);
            else if (record.name == "economy_certifications")
                addValue(AuthorityCertifications, record.value);
            else if (record.name == "maintenance_economy_certifications")
                addValue(MaintenanceCertifications, record.value);
            else if (record.name == "committed_waves")
            {
                addValue(CommittedWaves, record.value);
                const auto epoch_tag = record.tags.find("epoch");
                uint64_t epoch = 0;
                if (epoch_tag == record.tags.end() ||
                    !parse_u64(epoch_tag->second, epoch) || epoch <= 1u)
                {
                    ++local[ServiceGainEvidenceViolations];
                    ++local[NetBenefitEvidenceViolations];
                }
                else
                {
                    committed_epochs.insert(epoch);
                }
            }
            else if (record.name == "committed_expert_migrations")
                addValue(CommittedMigrations, record.value);
            else if (record.name == "committed_migration_cycles")
            {
                if (record.value == 2.0)
                    ++local[FullCycleCapWaves];
                else if (record.value > 2.0)
                    ++local[CycleCapViolations];
            }
            else if (record.name == "bootstrap_live_slots_recycled")
                addValue(BootstrapSlotsRecycled, record.value);
            else if (record.name == "physical_waves_prepared")
                addValue(PhysicalWavesPrepared, record.value);
            else if (record.name == "promotions")
                addValue(Promotions, record.value);
            else if (record.name == "demotions")
                addValue(Demotions, record.value);
            else if (record.name == "cross_domain_migrations")
                addValue(CrossDomain, record.value);
            else if (record.name == "cross_rank_migrations")
                addValue(CrossRank, record.value);
            else if (record.name == "cross_backend_migrations")
                addValue(CrossBackend, record.value);
            else if (record.name == "estimated_weight_bytes")
                addValue(EstimatedBytes, record.value);
            else if (record.name == "decode_boundary_background_notifications")
                addValue(BackgroundNotifications, record.value);
            else if (record.name == "migration_stage_failures")
                addValue(StageFailures, record.value);
            else if (record.name == "migration_commit_failures")
                addValue(CommitFailures, record.value);
            else if (record.name == "maintenance_fatal_failures")
                addValue(FatalFailures, record.value);
            else if (record.name ==
                     "committed_projected_service_gain_ns")
            {
                const auto epoch_tag = record.tags.find("epoch");
                uint64_t epoch = 0;
                if (record.value <= 0.0 ||
                    epoch_tag == record.tags.end() ||
                    !parse_u64(epoch_tag->second, epoch) || epoch <= 1u)
                {
                    ++local[ServiceGainEvidenceViolations];
                }
                else
                {
                    positive_service_gain_epochs.insert(epoch);
                }
            }
            else if (record.name ==
                     "committed_projected_net_benefit_ns")
            {
                const auto epoch_tag = record.tags.find("epoch");
                uint64_t epoch = 0;
                if (record.value <= 0.0 ||
                    epoch_tag == record.tags.end() ||
                    !parse_u64(epoch_tag->second, epoch) || epoch <= 1u)
                {
                    ++local[NetBenefitEvidenceViolations];
                }
                else
                {
                    positive_net_benefit_epochs.insert(epoch);
                }
            }
            else if (record.name == "expert_migration_edges")
            {
                const auto direction = record.tags.find("direction");
                const auto source = record.tags.find("source_priority");
                const auto destination =
                    record.tags.find("destination_priority");
                int source_priority = 0;
                int destination_priority = 0;
                bool direction_valid =
                    direction != record.tags.end() &&
                    source != record.tags.end() &&
                    destination != record.tags.end() &&
                    parse_int(source->second, source_priority) &&
                    parse_int(destination->second, destination_priority);
                if (direction_valid)
                {
                    direction_valid =
                        (direction->second == "promotion" &&
                         destination_priority < source_priority) ||
                        (direction->second == "demotion" &&
                         destination_priority > source_priority) ||
                        (direction->second == "same_priority" &&
                         destination_priority == source_priority);
                }
                if (!direction_valid)
                    ++local[PriorityDirectionViolations];
            }

            const auto blocking = record.tags.find("blocking");
            const auto blocking_inference =
                record.tags.find("blocking_inference");
            const auto inference_path = record.tags.find("inference_path");
            const bool setup_only_blocking =
                blocking != record.tags.end() && blocking->second == "true" &&
                record.phase == "model_setup" &&
                inference_path != record.tags.end() &&
                inference_path->second == "false";
            if ((blocking != record.tags.end() && blocking->second == "true" &&
                 !setup_only_blocking) ||
                (blocking_inference != record.tags.end() &&
                 blocking_inference->second != "false"))
            {
                ++local[BlockingInferenceViolations];
            }
        }

        /*
         * A positive projected service gain is the exact phase-weighted
         * placement objective delta computed from the live routing window and
         * measured tier/layer service profile. Requiring one for every
         * published epoch proves monotonically improving placement, while the
         * positive net benefit additionally proves transfer/repack and measured
         * inference interference did not erase that gain.
         */
        for (const uint64_t epoch : committed_epochs)
        {
            if (!positive_service_gain_epochs.contains(epoch))
                ++local[ServiceGainEvidenceViolations];
            if (!positive_net_benefit_epochs.contains(epoch))
                ++local[NetBenefitEvidenceViolations];
        }
        local[ImprovingEpochs] = committed_epochs.size();

        std::array<uint64_t, EvidenceCount> global{};
        MPI_Allreduce(
            local.data(),
            global.data(),
            static_cast<int>(global.size()),
            MPI_UINT64_T,
            MPI_SUM,
            MPI_COMM_WORLD);
        if (!isRootParityRank())
            return;

        ASSERT_EQ(global[DomainEnabled], static_cast<uint64_t>(mpiWorldSize()))
            << "Every rank must retain ExpertOverlay residency PerfStats";
        EXPECT_EQ(global[StageFailures], 0u);
        EXPECT_EQ(global[CommitFailures], 0u);
        EXPECT_EQ(global[FatalFailures], 0u);
        EXPECT_EQ(global[BlockingInferenceViolations], 0u)
            << "Expert movement exposed a blocking inference-path tag";
        EXPECT_EQ(global[PriorityDirectionViolations], 0u)
            << "Migration direction did not match numeric tier priority";

        if (!isDynamicResidencyProductionTest())
        {
            EXPECT_EQ(global[StaticChecks], static_cast<uint64_t>(mpiWorldSize()))
                << "Every static authority must prove immobility exactly once";
            EXPECT_EQ(global[CommittedWaves], 0u);
            EXPECT_EQ(global[CommittedMigrations], 0u)
                << "Static ExpertOverlay placement moved an expert";
            EXPECT_EQ(global[Promotions], 0u);
            EXPECT_EQ(global[Demotions], 0u);
            EXPECT_EQ(global[ImprovingEpochs], 0u);
            EXPECT_EQ(global[BootstrapSlotsRecycled], 0u)
                << "Static ExpertOverlay recycled no bootstrap assignment";
            return;
        }

        const uint64_t ranks = static_cast<uint64_t>(mpiWorldSize());
        EXPECT_GE(global[ConfiguredCycleCapRecords], ranks)
            << "Every rank must publish its production cycle cap";
        EXPECT_EQ(global[CycleCapViolations], 0u)
            << "The public two-cycle policy was not preserved through production composition";
        EXPECT_EQ(global[PhysicalEvidenceViolations], 0u)
            << "The physical fabric published malformed slot-adoption evidence";
        if (activeTopology() == OverlayTopology::CudaRocmCpu)
        {
            EXPECT_GE(global[FullCycleCapWaves], ranks)
                << "The three-tier real-model campaign never admitted a full "
                   "two-cycle wave";
        }
        EXPECT_GT(global[AdoptedInitialSlots], 0u)
            << "The physical fabric did not adopt loader-owned expert slots";
        EXPECT_GT(global[BootstrapSlotsRecycled], 0u)
            << "Movement never recycled a retired loader-owned expert slot";
        EXPECT_GT(global[PhysicalWavesPrepared], 0u)
            << "No physical migration wave reached background preparation";
        EXPECT_GE(global[CalibrationComplete], ranks);
        EXPECT_GE(global[CertificationComplete], ranks);
        EXPECT_GE(global[AuthorityCertifications], ranks);
        EXPECT_GE(global[MaintenanceCertifications], ranks);
        EXPECT_GE(global[ImprovingEpochs], 2u * ranks)
            << "Each rank must publish at least two distinct improving epochs";
        EXPECT_EQ(global[ImprovingEpochs], global[CommittedWaves])
            << "Every committed wave must name one distinct publication epoch";
        EXPECT_EQ(global[ServiceGainEvidenceViolations], 0u)
            << "Every committed epoch must improve measured service cost";
        EXPECT_EQ(global[NetBenefitEvidenceViolations], 0u)
            << "Every committed epoch must remain profitable after movement cost and interference";
        EXPECT_GT(global[CommittedMigrations], 0u);
        EXPECT_GT(global[Promotions], 0u);
        EXPECT_EQ(global[Promotions], global[Demotions])
            << "Capacity-preserving tier movement requires paired promotion/demotion";
        EXPECT_EQ(global[CrossDomain], global[CommittedMigrations]);
        EXPECT_EQ(global[CrossBackend], global[CommittedMigrations]);
        if (activeTopology() == OverlayTopology::CudaRocm)
        {
            EXPECT_EQ(global[CrossRank], global[CommittedMigrations])
                << "The two single-participant GPU domains live on distinct MPI ranks";
        }
        else
        {
            /*
             * NodeTP deliberately supplies one CPU participant per rank.
             * A GPU/CPU edge can therefore be rank-local or rank-remote based
             * on inventory binding and selected expert. Require the wave to
             * exercise the distributed path without inventing a socket-to-GPU
             * affinity requirement.
             */
            EXPECT_GT(global[CrossRank], 0u);
            EXPECT_LE(global[CrossRank], global[CommittedMigrations]);
        }
        EXPECT_GT(global[EstimatedBytes], 0u);
        EXPECT_GT(global[BackgroundNotifications], 0u)
            << "Inference never exercised the wake-only maintenance boundary";
    }

    /**
     * @brief Prove the post-publication parity request routes a promoted expert.
     *
     * A committed migration edge proves that the background transfer and
     * publication protocols completed, but by itself it does not prove that a
     * later inference consumed the changed residency map.  Match positive-
     * traffic promotion records to the full-prompt routing checkpoints while
     * those snapshots are still live.  Numerical parity then covers the output
     * produced after that migrated expert was selected by the real router.
     */
    void assertParityPrefillExercisesPromotedExpert() const
    {
        if (!isDynamicResidencyProductionTest() || !isRootParityRank())
            return;

        struct PromotedExpert
        {
            int layer = -1;
            int expert = -1;
        };

        std::vector<PromotedExpert> positive_promotions;
        size_t malformed_edges = 0;
        for (const auto &record :
             PerfStatsCollector::snapshot({"moe_overlay_residency"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "moe_overlay_residency" ||
                record.name != "expert_migration_edges")
            {
                continue;
            }

            const auto direction = record.tags.find("direction");
            if (direction == record.tags.end() ||
                direction->second != "promotion")
            {
                continue;
            }

            const auto layer_tag = record.tags.find("layer");
            const auto expert_tag = record.tags.find("expert");
            const auto activation_tag = record.tags.find("activation_count");
            if (layer_tag == record.tags.end() ||
                expert_tag == record.tags.end() ||
                activation_tag == record.tags.end())
            {
                ++malformed_edges;
                continue;
            }

            int layer = -1;
            int expert = -1;
            uint64_t activations = 0;
            const auto parse_int = [](const std::string &text, int &value)
            {
                const char *const begin = text.data();
                const char *const end = begin + text.size();
                const auto parsed = std::from_chars(begin, end, value);
                return parsed.ec == std::errc{} && parsed.ptr == end;
            };
            const auto parse_u64 = [](const std::string &text, uint64_t &value)
            {
                const char *const begin = text.data();
                const char *const end = begin + text.size();
                const auto parsed = std::from_chars(begin, end, value);
                return parsed.ec == std::errc{} && parsed.ptr == end;
            };
            if (!parse_int(layer_tag->second, layer) ||
                !parse_int(expert_tag->second, expert) ||
                !parse_u64(activation_tag->second, activations) ||
                layer < 0 || expert < 0)
            {
                ++malformed_edges;
                continue;
            }
            if (activations > 0)
                positive_promotions.push_back({layer, expert});
        }

        ASSERT_EQ(malformed_edges, 0u)
            << "Committed promotion PerfStats contained malformed identity tags";
        ASSERT_FALSE(positive_promotions.empty())
            << "Dynamic residency committed no positive-traffic promotion";

        bool exercised = false;
        std::ostringstream candidates;
        for (const auto &promotion : positive_promotions)
        {
            if (candidates.tellp() > 0)
                candidates << ", ";
            candidates << "layer" << promotion.layer << ":expert"
                       << promotion.expert;

            size_t route_elements = 0;
            const std::string snapshot_key =
                "layer" + std::to_string(promotion.layer) +
                "_MOE_ROUTING_INDICES";
            const float *const routes =
                activeSnapshot(snapshot_key, route_elements);
            if (!routes)
                continue;

            // Router IDs are stored exactly as FP32 integers in parity dumps.
            exercised = std::any_of(
                routes,
                routes + route_elements,
                [&promotion](float routed_expert)
                {
                    return routed_expert ==
                           static_cast<float>(promotion.expert);
                });
            if (exercised)
                break;
        }

        EXPECT_TRUE(exercised)
            << "The post-publication parity prefill did not route any promoted "
               "positive-traffic expert; candidates: "
            << candidates.str();
    }

    /**
     * @brief Return the inventory-resolved dense continuation authority.
     *
     * Before runner setup, rank zero retains reference-pack preparation. Once
     * production has bound the topology, comparisons and CSV output move to
     * the same rank that owns logits and stage snapshots.
     */
    int parityArtifactAuthorityRank() const override
    {
        return orch_runner_ ? orch_runner_->coordinatedRootRank() : 0;
    }

    /** @return Whether this process owns the inventory-resolved continuation. */
    bool isRootParityRank() const
    {
        return mpi_ctx_
                   ? mpi_ctx_->rank() == parityArtifactAuthorityRank()
                   : isRank0();
    }

    bool synchronizedDecodeWorkAvailable()
    {
        bool available = true;
        if (isRootParityRank())
        {
            available = !loadPyTorchSnapshot("decode_step0_LM_HEAD").empty() &&
                        !readDecodeTokensFromMetadata().empty();
        }
        return broadcastRootFlag(available);
    }

    bool producedPrefillSummary(const ParityTestSummary &summary) const
    {
        return summary.embedding_passed ||
               !summary.layer_stats.empty() ||
               summary.lm_head_passed ||
               summary.lm_head_cosine != 0.0f ||
               summary.total_layers_passed > 0;
    }

    bool producedDecodeSummary(const DecodeParitySummary &summary) const
    {
        return !summary.step_stats.empty() ||
               summary.steps_total > 0 ||
               summary.top1_matches > 0;
    }

    [[noreturn]] void abortGraphNativeWorld(const std::string &reason) const
    {
        const std::string message =
            "[Qwen3.5 MoE GraphNative] " + reason +
            "; aborting MPI world to avoid stranding rocm_warm/cpu_cold participants";
        LOG_ERROR(message);
        std::cerr << message << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 2);
        std::abort();
    }

    [[noreturn]] void abortAfterRootThrow(const char *phase, const std::string &what) const
    {
        abortGraphNativeWorld(std::string("root rank threw during ") + phase + ": " + what);
    }

    /**
     * @brief Fold the identical post-loop evidence sequence on every MPI rank.
     *
     * Worker ranks enter this sequence immediately after receiving the typed
     * SHUTDOWN command. The continuation authority must call it in the same
     * order after closing the loop so no test-only collective can race the
     * production command communicator.
     */
    void assertEvidenceAfterWorkerShutdown()
    {
        assertParticipantCompactBufferArenaEvidence();
        assertMappedParticipantGraphEvidence();
        assertActiveTierRouteEvidence();
        assertSparseTransportPerfStatsEvidence();
        assertResidencyMovementEvidence();
        if (isSegmentedPrefillProductionTest())
            assertSegmentedPrefillEvidence();
        finishProductionParityEvidence();
    }

    /**
     * @brief Run the complete three-tier graph-native parity contract once.
     */
    void runGraphNativeProductionParityBody()
    {
        beginProductionParityEvidence();
        if (isLegacyOverlayRuntimeEnabled())
        {
            FAIL() << kLegacyEnvVar
                   << " is set in the environment. This test requires graph-native overlay lowering.";
        }

        const bool hardware_and_model_ok = collectivelyCheckHardwareAndModel();
        if (!hardware_and_model_ok)
        {
            if (isRootParityRank())
            {
                const auto blocker = acceleratorHardwareBlocker(cluster_inventory_);
                if (blocker)
                    ADD_FAILURE() << "Production GraphNative prerequisite failed: " << *blocker;
                else
                    ADD_FAILURE() << "Production parity model not found at " << kModelPath;
            }
            return;
        }

        const bool setup_ok = setupPipeline();
        ASSERT_TRUE(synchronizeRanksOk(setup_ok)) << "Pipeline setup failed on one or more ranks";

        const bool decode_available = synchronizedDecodeWorkAvailable();
        ASSERT_TRUE(decode_available)
            << "Production parity requires incremental-decode snapshots and metadata";

        /*
         * Exercise the same coordinated application boundary used by an
         * interactive/server deployment. Rank zero issues typed inference
         * commands; every other MPI instance remains a real
         * OrchestrationRunner and executes whichever accelerator and CPU-NUMA
         * participants runtime inventory binding assigned to it. This is
         * deliberately not a parity-fixture MPI loop.
         */
        orch_runner_->setMPICoordinatedMode(true);
        MPI_Barrier(MPI_COMM_WORLD);
        if (!isRootParityRank())
        {
            orch_runner_->runMPIWorkerLoop();
            assertEvidenceAfterWorkerShutdown();
            return;
        }

        struct WorkerShutdownGuard
        {
            IOrchestrationRunner *runner = nullptr;
            ~WorkerShutdownGuard()
            {
                if (runner)
                    runner->shutdownMPIWorkers();
            }
        } worker_shutdown{orch_runner_.get()};

        if (!driveDynamicResidencyToDistributedMigration())
        {
            /*
             * No command is active here: every prefill/decode boundary already
             * completed successfully. Close the production worker protocol so
             * both ranks can fold PerfStats and retain the rejected-payoff
             * evidence. MPI_Abort remains reserved for exceptions that leave a
             * published command's collective order indeterminate.
             */
            orch_runner_->shutdownMPIWorkers();
            worker_shutdown.runner = nullptr;
            assertEvidenceAfterWorkerShutdown();
            ADD_FAILURE()
                << "Real dynamic residency workload did not prove a cross-rank expert migration";
            return;
        }

        if (isDynamicResidencyProductionTest())
        {
            const bool convergence_timings_ready =
                collectConvergedInferenceTimings();
            EXPECT_TRUE(convergence_timings_ready)
                << "Could not collect epoch-stable observed convergence timings";
            if (convergence_timings_ready)
                assertAndWriteObservedConvergenceSpeedup();

            /*
             * Only the dense continuation authority owns parity artifacts.
             * Re-enabling its diagnostic nodes invalidates that local graph
             * topology once; runPrefillParity then performs the normal
             * warmup/capture retry while sparse participants keep their lean
             * production graphs and unchanged collective schedule.
             */
            orch_runner_->enableSnapshotCapture();
            /*
             * Calibration is an ordinary production workload and therefore
             * leaves KV, short-convolution, GDN recurrence, logical position,
             * and snapshot state belonging to that request.  The Hugging Face
             * reference pack starts from an empty request.  Cross the same
             * typed request boundary used by serving before collecting parity
             * checkpoints; residency is model-lifetime state and deliberately
             * survives this reset, so the following forward still exercises
             * the migrated epoch proved above.
             */
            activeClearSnapshots();
            activeClearCache();
        }

        ParityTestSummary prefill;
        try
        {
            prefill = runPrefillParity();
        }
        catch (const std::exception &e)
        {
            abortAfterRootThrow("prefill parity", e.what());
        }
        catch (...)
        {
            abortAfterRootThrow("prefill parity", "unknown exception");
        }

        if (isRootParityRank() && !producedPrefillSummary(prefill))
        {
            abortGraphNativeWorld(
                "root produced no prefill parity summary - forward likely failed before "
                "all overlay tiers completed");
        }

        if (isRootParityRank())
        {
            assertParity(prefill);
            assertParityPrefillExercisesPromotedExpert();
            if (isSegmentedPrefillProductionTest())
                assertSegmentedPrefillCheckpointCoverage();
            assertProductionParitySnapshotInfrastructure();
            cachePublishedEpochRouteEvidence();
        }
        activeClearSnapshots();
        activeClearCache();

        DecodeParitySummary decode;
        try
        {
            decode = runDecodeParity();
        }
        catch (const std::exception &e)
        {
            abortAfterRootThrow("decode parity", e.what());
        }
        catch (...)
        {
            abortAfterRootThrow("decode parity", "unknown exception");
        }

        if (isRootParityRank() && !producedDecodeSummary(decode))
        {
            abortGraphNativeWorld(
                "root produced no decode parity summary - forward likely failed before "
                "all overlay tiers completed");
        }

        if (isRootParityRank())
            assertDecodeParity(decode);

        /*
         * End the production worker protocol before entering the evidence
         * allreduce.  This keeps the command loop and the test-only collective
         * from competing for the same MPI messages while retaining the real
         * production setup and forward path above.
         */
        orch_runner_->shutdownMPIWorkers();
        worker_shutdown.runner = nullptr;
        assertEvidenceAfterWorkerShutdown();
    }

    std::shared_ptr<MoERoutedExpertPlacementPlan> overlay_plan_;
    ClusterInventory cluster_inventory_;
    ResidencyConvergenceTimings convergence_timings_;
    std::vector<uint64_t>
        parity_route_counts_by_participant_; ///< Live checkpoint routes under the published epoch.
    std::vector<bool>
        parity_route_requires_remote_completion_; ///< Participants requiring a follower Complete proof.
};

TEST_F(Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold, ProductionParity_CUDA_ROCm_CPU)
{
    runGraphNativeProductionParityBody();
}

/**
 * @brief Run real-weight CUDA-hot/CPU-cold graph-native parity through production MPI runners.
 *
 * The fixture keeps both CPU sockets in the cold NodeTP tier, requires
 * captured CUDA execution, compares every Hugging Face checkpoint, writes the
 * normal CSV artifacts, and asserts completed local-route evidence for all
 * three sparse participants after the coordinated worker protocol exits.
 */
TEST_F(Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold, ProductionParity_CUDA_CPU)
{
    runGraphNativeProductionParityBody();
}

/**
 * @brief Run real-weight ROCm-hot/NodeTP-CPU-cold production parity.
 *
 * Runtime inventory binding may place the ROCm continuation on either MPI
 * instance. Both CPU sockets participate in the cold tier, and PerfStats must
 * prove that all three sparse endpoints completed routed expert work.
 */
TEST_F(Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold, ProductionParity_ROCm_CPU)
{
    runGraphNativeProductionParityBody();
}

/**
 * @brief Run real-weight CUDA-hot/ROCm-warm all-GPU production parity.
 *
 * The two GPU backends share the packed expert representation, while the
 * production sparse collective and transaction lifecycle remain identical to
 * the larger overlay cells.
 */
TEST_F(Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold, ProductionParity_CUDA_ROCm)
{
    runGraphNativeProductionParityBody();
}

/**
 * @brief Isolate seeded random ownership without permitting runtime movement.
 *
 * This cell is the control for the dynamic migration proof: it exercises the
 * same real sparse-collective graph and CSV oracle with random initial expert
 * placement while PerfStats must certify that static policy moved nothing.
 */
TEST_F(
    Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold,
    ProductionParity_StaticRandom_CUDA_ROCm)
{
    runGraphNativeProductionParityBody();
}

/**
 * @brief Prove random initial ownership converges through real CUDA/ROCm movement.
 *
 * Calibration, histogram production, asynchronous blob transfer, epoch
 * publication, and the subsequent checkpoint comparison all use the live
 * two-rank production graph. Integer priorities -20 and 17 are deliberately
 * non-semantic and non-consecutive; tier names never determine direction.
 */
TEST_F(
    Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold,
    ProductionParity_DynamicRandom_CUDA_ROCm)
{
    runGraphNativeProductionParityBody();
}

/**
 * @brief Prove CUDA/NodeTP-CPU movement with streaming GPU-side repack.
 *
 * Promotions convert CPU NativeVNNI bytes into the CUDA packed layout as the
 * bytes arrive; demotions perform the inverse conversion on CUDA before D2H
 * publication. Both socket participants remain eligible independently of the
 * rank on which inventory binding locates the GPU.
 */
TEST_F(
    Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold,
    ProductionParity_DynamicRandom_CUDA_CPU)
{
    runGraphNativeProductionParityBody();
}

/**
 * @brief Prove the symmetric ROCm/NodeTP-CPU streaming-repack path.
 */
TEST_F(
    Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold,
    ProductionParity_DynamicRandom_ROCm_CPU)
{
    runGraphNativeProductionParityBody();
}

/**
 * @brief Prove arbitrary-priority movement across CUDA, ROCm, and CPU tiers.
 *
 * Priorities -20, 7, and 41 intentionally have no semantic names or fixed
 * spacing. The production controller must improve random ownership across both
 * the shared GPU packed format and the GPU/CPU streaming conversion boundary.
 */
TEST_F(
    Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold,
    ProductionParity_DynamicRandom_CUDA_ROCm_CPU)
{
    runGraphNativeProductionParityBody();
}

TEST_F(
    Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold,
    ProductionParity_SegmentedPrefill_CUDA_ROCm_CPU)
{
    runGraphNativeProductionParityBody();
}

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "[Rank " << rank << "] Qwen3.5 MoE GraphNative CudaHot/RocmWarm/CpuCold parity test\n";

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();

    MPI_Finalize();

    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
