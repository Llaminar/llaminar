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
#include "execution/mtp/MTPVerifierPolicy.h"
#include "execution/prefix_cache/PrefixCacheStateProbe.h"
#include "execution/runner/OrchestrationRunner.h"
#include "planning/ClusterInventoryGatherer.h"
#include "utils/MTPParitySnapshotContext.h"
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
#include <map>
#include <memory>
#include <mutex>
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
    constexpr const char *kQwen122ModelPath =
        "/opt/llaminar-models/Qwen3.5-122B-A10B-UD-Q8_K_XL-00001-of-00004.gguf";
    constexpr const char *kQwen122SnapshotDir =
        "pytorch_qwen35_122b_moe_mtp_snapshots";
    /** Maximum recurrent predictor depth certified by the 122B campaign. */
    constexpr int kQwen122MaximumMTPDraftDepth = 15;
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
    constexpr size_t kCuda2Rocm4OverlayParticipantCount = 6;

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
        Cuda2Rocm4,
    };

    /**
     * @brief Movement axes that the resolved participant catalogue can express.
     *
     * Promotion/demotion needs two integer priorities. Same-priority skew
     * balancing additionally needs at least two physical participants carrying
     * one priority. Keeping that distinction typed prevents a singleton tier
     * from being asked to manufacture an impossible movement while preserving
     * the stronger two-axis gate for NodeTP CPU and multi-GPU tiers.
     */
    enum class DynamicMovementAxisContract : std::uint8_t
    {
        PriorityMigrationOnly,
        PriorityMigrationAndParticipantBalance,
    };

    /**
     * @brief Derive the physically expressible Dynamic movement contract.
     *
     * Tier names and accelerator vendors are deliberately ignored. The same
     * integer priority may be represented by one or more declarative domains;
     * their participant counts are therefore accumulated before deciding
     * whether a same-priority exchange can exist.
     *
     * @param plan Inventory-bound or rank-agnostic ExpertOverlay plan.
     * @return Exact movement-axis contract implied by its participant catalogue.
     * @throws std::invalid_argument when a tier references no declared domain.
     */
    DynamicMovementAxisContract dynamicMovementAxisContract(
        const MoERoutedExpertPlacementPlan &plan)
    {
        std::map<int, std::size_t> participants_by_priority;
        for (const auto &tier : plan.routed_tiers)
        {
            const auto domain = std::find_if(
                plan.domains.begin(),
                plan.domains.end(),
                [&](const RoutedExpertDomain &candidate)
                { return candidate.name == tier.domain; });
            if (domain == plan.domains.end())
            {
                throw std::invalid_argument(
                    "Dynamic movement-axis resolution cannot find routed domain '" +
                    tier.domain + "'");
            }
            participants_by_priority[tier.priority] +=
                domain->participants.size();
        }

        const bool has_same_priority_peers = std::any_of(
            participants_by_priority.begin(),
            participants_by_priority.end(),
            [](const auto &entry)
            { return entry.second > 1u; });
        return has_same_priority_peers
                   ? DynamicMovementAxisContract::
                         PriorityMigrationAndParticipantBalance
                   : DynamicMovementAxisContract::PriorityMigrationOnly;
    }

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

    /** @return Exact GTest name, or an empty string outside a running cell. */
    std::string activeTestName()
    {
        const auto *info =
            ::testing::UnitTest::GetInstance()->current_test_info();
        return info ? std::string(info->name()) : std::string{};
    }

    /** @return Whether this is the production 122B six-GPU campaign. */
    bool isQwen122ProductionTest()
    {
        return activeTestName().find("Qwen35_122B_CUDA2_ROCm4") !=
               std::string::npos;
    }

    /** @return Model root selected by the active real-weight cell. */
    const char *activeModelPath()
    {
        return isQwen122ProductionTest() ? kQwen122ModelPath : kModelPath;
    }

    /** @return Authenticated reference directory selected by the active cell. */
    const char *activeSnapshotDir()
    {
        return isQwen122ProductionTest() ? kQwen122SnapshotDir : kSnapshotDir;
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
        return activeTestName().find("_LLEP_") != std::string::npos;
    }

    /** @brief Return whether the exact cell exercises persistent tier movement. */
    bool isDynamicResidencyProductionTest()
    {
        const std::string name = activeTestName();
        return name == kDynamicCudaRocmTest ||
               name == kDynamicCudaCpuTest ||
               name == kDynamicRocmCpuTest ||
               name == kDynamicCudaRocmCpuTest ||
               (isQwen122ProductionTest() &&
                (name.find("_Dynamic_") != std::string::npos ||
                 isLLEPProductionTest()));
    }

    /** @brief Return whether initial expert ownership uses seeded random order. */
    bool isRandomOwnerProductionTest()
    {
        const std::string name = activeTestName();
        return name == kDynamicCudaRocmTest ||
               name == kDynamicCudaCpuTest ||
               name == kDynamicRocmCpuTest ||
               name == kDynamicCudaRocmCpuTest ||
               name == kStaticRandomCudaRocmTest ||
               name.find("_Random_") != std::string::npos;
    }

    /** @return Fixed MTP depth, or the adaptive policy's maximum depth. */
    int activeMTPDraftDepth()
    {
        const std::string name = activeTestName();
        // Test the longest spelling first: `MTPDepth15` contains `MTPDepth1`.
        if (name.find("MTPDepth15") != std::string::npos)
            return kQwen122MaximumMTPDraftDepth;
        if (name.find("MTPDepth1") != std::string::npos)
            return 1;
        if (name.find("MTPDepth2") != std::string::npos)
            return 2;
        if (name.find("MTPDynamicDepth") != std::string::npos)
            return kQwen122MaximumMTPDraftDepth;
        return 3;
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
        return isQwen122ProductionTest()
                   ? kQwen122MaximumMTPDraftDepth + 1
                   : activeMTPDraftDepth() + 1;
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
        const bool dynamic_depth =
            activeTestName().find("MTPDynamicDepth") != std::string::npos;
        return dynamic_depth
                   ? capacity_rows
                   : mtpVerifierPhysicalRowBucket(
                         activeMTPDraftDepth() + 1,
                         capacity_rows);
    }

    /** @return Whether the production device depth controller is adaptive. */
    bool usesDynamicMTPDepth()
    {
        return activeTestName().find("MTPDynamicDepth") !=
               std::string::npos;
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
        if (name.find("Qwen35_122B_CUDA2_ROCm4") != std::string::npos)
            return OverlayTopology::Cuda2Rocm4;
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
               topology == OverlayTopology::CudaRocm ||
               topology == OverlayTopology::Cuda2Rocm4;
    }

    /** @brief Return whether the active topology contains a ROCm participant. */
    bool topologyUsesRocm()
    {
        const auto topology = activeTopology();
        return topology == OverlayTopology::CudaRocmCpu ||
               topology == OverlayTopology::RocmCpu ||
               topology == OverlayTopology::CudaRocm ||
               topology == OverlayTopology::Cuda2Rocm4;
    }

    /** @brief Return whether the active topology contains NodeTP CPU cold. */
    bool topologyUsesCpu()
    {
        const auto topology = activeTopology();
        return topology != OverlayTopology::CudaRocm &&
               topology != OverlayTopology::Cuda2Rocm4;
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
        case OverlayTopology::Cuda2Rocm4:
            return kCuda2Rocm4OverlayParticipantCount;
        }
        throw std::logic_error("Unhandled Qwen3.5 MoE overlay topology");
    }

    RoutedExpertDomain cudaHotDomain()
    {
        RoutedExpertDomain domain;
        domain.name = kCudaHotDomain;
        domain.scope = isQwen122ProductionTest()
                           ? ExecutionDomainScope::RANK_LOCAL
                           : ExecutionDomainScope::SINGLE;
        domain.backend = CollectiveBackendType::NCCL;
        domain.participants = isQwen122ProductionTest()
                                  ? std::vector<GlobalDeviceAddress>{
                                        GlobalDeviceAddress::cuda(0),
                                        GlobalDeviceAddress::cuda(1)}
                                  : std::vector<GlobalDeviceAddress>{
                                        GlobalDeviceAddress::cuda(0)};
        domain.owner_rank = -1;
        domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        domain.routed_phase_policy = RoutedExpertPhasePolicy::Uniform;
        domain.routed_decode_assignment_policy =
            RoutedExpertAssignmentPolicy::StaticOwner;
        domain.routed_prefill_assignment_policy =
            isLLEPProductionTest()
                ? RoutedExpertAssignmentPolicy::LeastLoadedResident
                : RoutedExpertAssignmentPolicy::StaticOwner;
        return domain;
    }

    RoutedExpertDomain rocmWarmDomain()
    {
        RoutedExpertDomain domain;
        domain.name = kRocmWarmDomain;
        domain.scope = isQwen122ProductionTest()
                           ? ExecutionDomainScope::RANK_LOCAL
                           : ExecutionDomainScope::SINGLE;
        domain.backend = CollectiveBackendType::RCCL;
        domain.participants = isQwen122ProductionTest()
                                  ? std::vector<GlobalDeviceAddress>{
                                        GlobalDeviceAddress::rocm(0),
                                        GlobalDeviceAddress::rocm(1),
                                        GlobalDeviceAddress::rocm(2),
                                        GlobalDeviceAddress::rocm(3)}
                                  : std::vector<GlobalDeviceAddress>{
                                        GlobalDeviceAddress::rocm(0)};
        domain.owner_rank = -1;
        domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        domain.routed_phase_policy = RoutedExpertPhasePolicy::Uniform;
        domain.routed_decode_assignment_policy =
            RoutedExpertAssignmentPolicy::StaticOwner;
        domain.routed_prefill_assignment_policy =
            isLLEPProductionTest()
                ? RoutedExpertAssignmentPolicy::LeastLoadedResident
                : RoutedExpertAssignmentPolicy::StaticOwner;
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
        case OverlayTopology::Cuda2Rocm4:
            /*
             * Automatic model-aware admission is the capacity authority.  A
             * zero limit is not a magic quota: it asks the resolver to fill
             * every participant to its measured safety margin, then places
             * the exact remainder in the lower-priority fallback tier.
             */
            plan.continuation_domain = kCudaHotDomain;
            plan.base_model_domain = kCudaHotDomain;
            plan.shared_expert_domain = kCudaHotDomain;
            plan.continuation_domain_spec.setDensePolicy(
                DenseParallelPolicy::TensorParallelDecodeMirroredEmbedding);
            plan.domains = {
                cudaHotDomain(),
                rocmWarmDomain(),
            };
            plan.routed_tiers = {
                makeTier(
                    "priority0",
                    kCudaHotDomain,
                    /*priority=*/0,
                    /*max_experts_per_layer=*/0),
                makeTier(
                    "priority1",
                    kRocmWarmDomain,
                    /*priority=*/1,
                    /*max_experts_per_layer=*/0,
                    /*fallback=*/true),
            };
            return plan;

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
        const int required_cuda =
            activeTopology() == OverlayTopology::Cuda2Rocm4 ? 2 : 1;
        const int required_rocm =
            activeTopology() == OverlayTopology::Cuda2Rocm4 ? 4 : 1;
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
        case OverlayTopology::Cuda2Rocm4:
            config.name = activeTestName();
            config.devices = {
                ParityDeviceType::CUDA,
                ParityDeviceType::CUDA,
                ParityDeviceType::ROCm,
                ParityDeviceType::ROCm,
                ParityDeviceType::ROCm,
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
        config.model_path = activeModelPath();
        config.snapshot_dir = activeSnapshotDir();
        config.activation_precision =
            topology == OverlayTopology::Cuda2Rocm4
                ? ActivationPrecision::FP16
                : ActivationPrecision::FP32;
        config.kv_cache_precision = KVCachePrecision::FP16;
        if (topology == OverlayTopology::Cuda2Rocm4)
        {
            config.decode_steps = 4;
            config.thresholds.cosine_threshold = 0.96f;
            config.thresholds.decode_cosine_threshold = 0.98f;
            config.thresholds.kl_threshold = 0.03f;
        }
        return config;
    }

    /**
     * @brief One bounded, physical-identity-indexed 122B prepared authority.
     *
     * Residency policy and owner order affect the automatic capacity solution
     * and therefore the exact expert subset packed into model-owned storage.
     * Every MTP cell admits the same depth-fifteen graph envelope, so requested
     * execution depth is intentionally absent from physical identity. The
     * identity space has four possible values, but retaining four complete
     * 122B authorities would itself
     * consume the VRAM needed to admit the next one. Consecutive compatible
     * cells reuse this single slot; an identity transition retires the old
     * ModelContext before the next physical plan is solved. Every MPI process
     * owns its own cache and therefore retains only its rank-local authority.
     */
    struct Qwen122OverlayModelContextCampaignCache
    {
        std::mutex mutex;
        std::string model_path;
        std::optional<std::size_t> physical_identity_slot;
        std::optional<ModelContextReuseContract> contract;
    };

    /** @return The sole bounded 122B prepared-weight cache in this MPI process. */
    Qwen122OverlayModelContextCampaignCache &
    qwen122OverlayModelContextCampaignCache()
    {
        static Qwen122OverlayModelContextCampaignCache cache;
        return cache;
    }

    /** @brief One live recursive checkpoint retained until its HF branch exists. */
    struct DeferredMTPCheckpoint
    {
        std::string stage;
        std::string production_key;
        std::vector<float> actual;
    };

    /**
     * @brief Complete immutable evidence for one missing recursive HF context.
     *
     * Only branch-dependent sidecar tensors are copied. Main-model, canonical
     * sidecar, token, movement, and graph-path evidence remain compared in the
     * originating production cell. The copy breaks the runner lifetime cleanly:
     * no graph, arena, model context, or device allocation survives into the
     * CPU reference phase.
     */
    struct DeferredMTPBranchContext
    {
        std::string test_name;
        std::string model_path;
        std::string prompt;
        std::string snapshot_dir;
        std::filesystem::path snapshot_csv_path;
        int decode_steps = 0;
        int call = 0;
        int reference_step = -1;
        int reference_depth = 0;
        int vocab_size = 0;
        int top_k = 0;
        int num_experts = 0;
        float cosine_threshold = 0.0f;
        float decode_cosine_threshold = 0.0f;
        float kl_threshold = 0.0f;
        std::vector<int32_t> condition_tokens;
        std::vector<DeferredMTPCheckpoint> checkpoints;
    };

    /** @brief Process-resident queue resolved after all production cells retire. */
    struct DeferredMTPBranchCampaign
    {
        std::mutex mutex;
        std::vector<DeferredMTPBranchContext> contexts;
    };

    /** @return The one bounded deferred-reference queue in this test process. */
    DeferredMTPBranchCampaign &deferredMTPBranchCampaign()
    {
        static DeferredMTPBranchCampaign campaign;
        return campaign;
    }

    /** @brief Release the final immutable 122B authority before Python loads it. */
    void releaseQwen122OverlayModelContextCampaignCache()
    {
        auto &cache = qwen122OverlayModelContextCampaignCache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        cache.contract.reset();
        cache.physical_identity_slot.reset();
        cache.model_path.clear();
    }

    /** @return A shell-safe single argument preserving every input byte. */
    std::string quoteDeferredReferenceArgument(const std::string &value)
    {
        std::string quoted = "'";
        for (const char character : value)
        {
            if (character == '\'')
                quoted += "'\"'\"'";
            else
                quoted += character;
        }
        quoted += '\'';
        return quoted;
    }

    /**
     * @brief Load one FP32/FP64 NPY tensor without constructing a parity fixture.
     * @param path Exact authenticated reference tensor path.
     * @param error Receives a precise load/type failure.
     * @return FP32 values, or an empty optional on failure.
     */
    std::optional<std::vector<float>> loadDeferredReferenceTensor(
        const std::filesystem::path &path,
        std::string *error)
    {
        try
        {
            const cnpy::NpyArray array = cnpy::npy_load(path.string());
            std::vector<float> values(array.num_vals);
            if (array.word_size == sizeof(float))
            {
                const float *const data = array.data<float>();
                std::copy(data, data + array.num_vals, values.begin());
            }
            else if (array.word_size == sizeof(double))
            {
                const double *const data = array.data<double>();
                std::transform(
                    data,
                    data + array.num_vals,
                    values.begin(),
                    [](double value) { return static_cast<float>(value); });
            }
            else
            {
                if (error)
                {
                    *error = "unsupported NPY word size " +
                             std::to_string(array.word_size) + " at " +
                             path.string();
                }
                return std::nullopt;
            }
            return values;
        }
        catch (const std::exception &exception)
        {
            if (error)
                *error = path.string() + ": " + exception.what();
            return std::nullopt;
        }
    }

    /**
     * @brief Compare sparse routing weights after aligning them by expert ID.
     * @return Mean sparse-vector cosine and maximum absolute expert-mass error.
     */
    std::pair<float, float> compareDeferredRoutingWeights(
        const std::vector<float> &actual_weights,
        const std::vector<float> &expected_weights,
        const std::vector<float> &actual_indices,
        const std::vector<float> &expected_indices,
        int top_k,
        int num_experts)
    {
        if (top_k <= 0 || num_experts <= 0 || actual_weights.empty() ||
            actual_weights.size() != expected_weights.size() ||
            actual_weights.size() != actual_indices.size() ||
            actual_weights.size() != expected_indices.size() ||
            actual_weights.size() % static_cast<size_t>(top_k) != 0u)
        {
            return {0.0f, std::numeric_limits<float>::infinity()};
        }

        const size_t rows =
            actual_weights.size() / static_cast<size_t>(top_k);
        double total_cosine = 0.0;
        float maximum_error = 0.0f;
        for (size_t row = 0; row < rows; ++row)
        {
            std::vector<float> actual_sparse(
                static_cast<size_t>(num_experts), 0.0f);
            std::vector<float> expected_sparse(
                static_cast<size_t>(num_experts), 0.0f);
            for (int index = 0; index < top_k; ++index)
            {
                const size_t offset =
                    row * static_cast<size_t>(top_k) +
                    static_cast<size_t>(index);
                const int actual_expert =
                    static_cast<int>(actual_indices[offset]);
                const int expected_expert =
                    static_cast<int>(expected_indices[offset]);
                if (actual_expert >= 0 && actual_expert < num_experts)
                {
                    actual_sparse[static_cast<size_t>(actual_expert)] =
                        actual_weights[offset];
                }
                if (expected_expert >= 0 && expected_expert < num_experts)
                {
                    expected_sparse[static_cast<size_t>(expected_expert)] =
                        expected_weights[offset];
                }
            }
            total_cosine += computeCosineSimilarity(
                actual_sparse.data(),
                expected_sparse.data(),
                actual_sparse.size());
            for (size_t expert = 0; expert < actual_sparse.size(); ++expert)
            {
                maximum_error = std::max(
                    maximum_error,
                    std::abs(actual_sparse[expert] - expected_sparse[expert]));
            }
        }
        return {
            static_cast<float>(total_cosine / static_cast<double>(rows)),
            maximum_error};
    }

    /**
     * @brief Complete one deferred context with the same route-aware math gate.
     * @param context Immutable live evidence copied before runner teardown.
     * @param error Receives every failed invariant for campaign diagnostics.
     * @return True only when all tensors and the aggregate semantic gate pass.
     */
    bool compareDeferredMTPBranchContext(
        const DeferredMTPBranchContext &context,
        std::string *error)
    {
        std::string reference_prefix =
            "decode_step" + std::to_string(context.reference_step) +
            "_BRANCH";
        for (const int32_t token : context.condition_tokens)
            reference_prefix += "_" + std::to_string(token);
        reference_prefix +=
            "_MTP" + std::to_string(context.reference_depth) + "_";

        std::map<std::string, std::vector<float>> references;
        for (const auto &checkpoint : context.checkpoints)
        {
            const auto path = std::filesystem::path(context.snapshot_dir) /
                              (reference_prefix + checkpoint.stage + ".npy");
            auto loaded = loadDeferredReferenceTensor(path, error);
            if (!loaded || loaded->empty())
                return false;
            if (loaded->size() > checkpoint.actual.size() &&
                !checkpoint.actual.empty() &&
                loaded->size() % checkpoint.actual.size() == 0u)
            {
                loaded->erase(
                    loaded->begin(),
                    loaded->end() -
                        static_cast<ptrdiff_t>(checkpoint.actual.size()));
            }
            if (loaded->size() != checkpoint.actual.size())
            {
                if (error)
                {
                    *error = context.test_name + " " + checkpoint.stage +
                             " element mismatch actual=" +
                             std::to_string(checkpoint.actual.size()) +
                             " reference=" +
                             std::to_string(loaded->size());
                }
                return false;
            }
            references.emplace(checkpoint.stage, std::move(*loaded));
        }

        std::ofstream csv(context.snapshot_csv_path, std::ios::app);
        if (!csv.is_open())
        {
            if (error)
                *error = "could not append " + context.snapshot_csv_path.string();
            return false;
        }

        bool context_finite = true;
        bool routing_indices_exact = true;
        bool routing_top1_match = true;
        float routing_overlap = 1.0f;
        bool routing_weights_equivalent = false;
        bool routed_expert_output_equivalent = false;
        bool lm_head_passed = false;
        double numerical_cosine_sum = 0.0;
        size_t numerical_stage_count = 0u;
        size_t compared_stages = 0u;

        const auto actual_indices_it = std::find_if(
            context.checkpoints.begin(),
            context.checkpoints.end(),
            [](const DeferredMTPCheckpoint &checkpoint)
            { return checkpoint.stage == "MOE_ROUTING_INDICES"; });
        const auto reference_indices_it = references.find(
            "MOE_ROUTING_INDICES");

        for (const auto &checkpoint : context.checkpoints)
        {
            const auto reference_it = references.find(checkpoint.stage);
            if (reference_it == references.end())
                return false;
            const auto &actual = checkpoint.actual;
            const auto &expected = reference_it->second;
            bool finite = true;
            bool exact_indices = true;
            double maximum_absolute_error = 0.0;
            for (size_t index = 0; index < actual.size(); ++index)
            {
                finite = finite && std::isfinite(actual[index]) &&
                         std::isfinite(expected[index]);
                maximum_absolute_error = std::max(
                    maximum_absolute_error,
                    std::abs(
                        static_cast<double>(actual[index]) -
                        static_cast<double>(expected[index])));
                if (checkpoint.stage == "MOE_ROUTING_INDICES")
                    exact_indices = exact_indices &&
                                    actual[index] == expected[index];
            }

            float cosine = computeCosineSimilarity(
                actual.data(), expected.data(), actual.size());
            float stage_routing_overlap = 1.0f;
            bool stage_routing_top1_match = true;
            float kl = 0.0f;
            bool passed = finite;
            if (checkpoint.stage == "MOE_ROUTING_INDICES")
            {
                if (context.top_k <= 0 ||
                    actual.size() % static_cast<size_t>(context.top_k) != 0u)
                {
                    passed = false;
                }
                else
                {
                    const size_t rows =
                        actual.size() / static_cast<size_t>(context.top_k);
                    double overlap_sum = 0.0;
                    size_t top1_matches = 0u;
                    for (size_t row = 0; row < rows; ++row)
                    {
                        std::set<int> actual_experts;
                        std::set<int> expected_experts;
                        for (int slot = 0; slot < context.top_k; ++slot)
                        {
                            const size_t offset =
                                row * static_cast<size_t>(context.top_k) +
                                static_cast<size_t>(slot);
                            actual_experts.insert(
                                static_cast<int>(actual[offset]));
                            expected_experts.insert(
                                static_cast<int>(expected[offset]));
                        }
                        size_t intersection = 0u;
                        for (const int expert : actual_experts)
                        {
                            if (expected_experts.contains(expert))
                                ++intersection;
                        }
                        overlap_sum +=
                            static_cast<double>(intersection) /
                            static_cast<double>(context.top_k);
                        top1_matches +=
                            actual[row * static_cast<size_t>(context.top_k)] ==
                            expected[row * static_cast<size_t>(context.top_k)];
                    }
                    stage_routing_overlap = static_cast<float>(
                        overlap_sum / static_cast<double>(rows));
                    stage_routing_top1_match = top1_matches == rows;
                    cosine = stage_routing_overlap;
                    maximum_absolute_error = 1.0 - stage_routing_overlap;
                    const float minimum_overlap =
                        1.0f - 1.0f / static_cast<float>(context.top_k);
                    passed = finite && stage_routing_top1_match &&
                             stage_routing_overlap >= minimum_overlap;
                }
                routing_indices_exact = exact_indices;
                routing_top1_match =
                    routing_top1_match && stage_routing_top1_match;
                routing_overlap = std::min(
                    routing_overlap, stage_routing_overlap);
            }
            else if (checkpoint.stage == "MOE_ROUTING_WEIGHTS")
            {
                if (actual_indices_it == context.checkpoints.end() ||
                    reference_indices_it == references.end())
                {
                    passed = false;
                }
                else
                {
                    const auto [sparse_cosine, sparse_max_error] =
                        compareDeferredRoutingWeights(
                            actual,
                            expected,
                            actual_indices_it->actual,
                            reference_indices_it->second,
                            context.top_k,
                            context.num_experts);
                    cosine = sparse_cosine;
                    maximum_absolute_error = sparse_max_error;
                    stage_routing_overlap = sparse_cosine;
                    passed = finite &&
                             sparse_cosine >= context.cosine_threshold;
                }
                routing_weights_equivalent = passed;
            }
            else
            {
                const float threshold =
                    checkpoint.stage == "MOE_EXPERT_OUTPUT" &&
                            !routing_indices_exact
                        ? context.cosine_threshold
                        : context.decode_cosine_threshold;
                passed = finite && cosine >= threshold;
                if (checkpoint.stage == "MOE_EXPERT_OUTPUT")
                    routed_expert_output_equivalent = passed;
            }

            if (checkpoint.stage == "LM_HEAD")
            {
                if (context.vocab_size <= 0 ||
                    actual.size() %
                            static_cast<size_t>(context.vocab_size) !=
                        0u)
                {
                    passed = false;
                }
                else
                {
                    kl = computeKLDivergence(
                        actual.data(),
                        expected.data(),
                        actual.size(),
                        static_cast<size_t>(context.vocab_size));
                    passed = passed && kl < context.kl_threshold &&
                             pytorchTop1InLlaminarTopK(
                                 actual.data(),
                                 expected.data(),
                                 actual.size(),
                                 static_cast<size_t>(context.vocab_size),
                                 3) >= 1.0f &&
                             pytorchTop1InLlaminarTopK(
                                 expected.data(),
                                 actual.data(),
                                 actual.size(),
                                 static_cast<size_t>(context.vocab_size),
                                 3) >= 1.0f;
                }
                lm_head_passed = passed;
            }

            context_finite = context_finite && finite;
            if (parityStageContributesToLayerCosine(
                    checkpoint.stage, routing_indices_exact))
            {
                numerical_cosine_sum += cosine;
                ++numerical_stage_count;
            }

            csv << context.call << ',' << context.reference_step << ','
                << context.reference_depth << ','
                << checkpoint.production_key << ','
                << reference_prefix << checkpoint.stage << ','
                << actual.size() << ',' << cosine << ','
                << maximum_absolute_error << ',' << kl << ','
                << (exact_indices ? 1 : 0) << ','
                << stage_routing_overlap << ','
                << (stage_routing_top1_match ? 1 : 0) << ','
                << (finite ? 1 : 0) << ',' << (passed ? 1 : 0) << '\n';
            ++compared_stages;
        }
        csv.flush();
        if (!csv.good())
        {
            if (error)
                *error = "failed while appending " + context.snapshot_csv_path.string();
            return false;
        }

        const float minimum_overlap =
            context.top_k > 0
                ? 1.0f - 1.0f / static_cast<float>(context.top_k)
                : 1.0f;
        const bool routed_contribution_equivalent =
            routing_weights_equivalent ||
            (routing_top1_match && routing_overlap >= minimum_overlap &&
             routed_expert_output_equivalent);
        const double numerical_cosine =
            numerical_stage_count > 0u
                ? numerical_cosine_sum /
                      static_cast<double>(numerical_stage_count)
                : 0.0;
        const bool passed = compared_stages > 0u && context_finite &&
                            routing_top1_match &&
                            routed_contribution_equivalent &&
                            numerical_cosine >=
                                context.decode_cosine_threshold &&
                            lm_head_passed;
        if (!passed && error)
        {
            std::ostringstream detail;
            detail << context.test_name
                   << " deferred recursive MTP parity failed"
                   << " step=" << context.reference_step
                   << " depth=" << context.reference_depth
                   << " compared_stages=" << compared_stages
                   << " finite=" << context_finite
                   << " routing_top1=" << routing_top1_match
                   << " routing_overlap=" << routing_overlap
                   << " routed_value=" << routed_contribution_equivalent
                   << " numerical_cosine=" << numerical_cosine
                   << " lm_head=" << lm_head_passed
                   << " csv=" << context.snapshot_csv_path;
            *error = detail.str();
        }
        return passed;
    }

    /**
     * @brief Generate all missing branches with one loaded HF model and compare.
     * @param contexts Immutable queue removed from the live campaign.
     * @param error Receives the generator output or first numerical failure.
     */
    bool resolveDeferredMTPBranchCampaign(
        const std::vector<DeferredMTPBranchContext> &contexts,
        std::string *error)
    {
        if (contexts.empty())
            return true;

        const auto &identity = contexts.front();
        int maximum_depth = 1;
        int decode_steps = identity.decode_steps;
        std::set<std::pair<int, std::vector<int32_t>>> unique_branches;
        for (const auto &context : contexts)
        {
            if (context.model_path != identity.model_path ||
                context.prompt != identity.prompt ||
                context.snapshot_dir != identity.snapshot_dir)
            {
                if (error)
                    *error = "deferred MTP campaign mixed reference identities";
                return false;
            }
            maximum_depth = std::max(
                maximum_depth,
                static_cast<int>(context.condition_tokens.size()) + 1);
            decode_steps = std::max(decode_steps, context.decode_steps);
            unique_branches.emplace(
                context.reference_step, context.condition_tokens);
        }

        const std::filesystem::path request_path =
            contexts.front().snapshot_csv_path.parent_path() /
            "mtp_hf_branch_campaign_requests.json";
        std::ofstream request(request_path, std::ios::trunc);
        if (!request.is_open())
        {
            if (error)
                *error = "could not write " + request_path.string();
            return false;
        }
        request << "[\n";
        size_t branch_index = 0u;
        for (const auto &[step, tokens] : unique_branches)
        {
            if (branch_index++ != 0u)
                request << ",\n";
            request << "  {\"" << step << "\": [";
            for (size_t token_index = 0; token_index < tokens.size(); ++token_index)
            {
                if (token_index != 0u)
                    request << ", ";
                request << tokens[token_index];
            }
            request << "]}";
        }
        request << "\n]\n";
        request.flush();
        if (!request.good())
        {
            if (error)
                *error = "failed while writing " + request_path.string();
            return false;
        }

        std::ostringstream script;
        script
            << "unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS "
               "OMP_PROC_BIND OMP_PLACES KMP_AFFINITY; "
            << "if [ -f /workspaces/llaminar/.venv/bin/activate ]; then "
               "source /workspaces/llaminar/.venv/bin/activate; fi; "
            << "python3 python/reference/"
               "generate_qwen35_moe_pipeline_snapshots.py"
            << " --model "
            << quoteDeferredReferenceArgument(identity.model_path)
            << " --prompt "
            << quoteDeferredReferenceArgument(identity.prompt)
            << " --output "
            << quoteDeferredReferenceArgument(identity.snapshot_dir)
            << " --decode-steps " << decode_steps
            << " --mtp-sidecar-snapshots --mtp-max-draft-depth "
            << maximum_depth << " --mtp-branch-overrides "
            << quoteDeferredReferenceArgument(request_path.string());
        const std::string command =
            "bash -c " + quoteDeferredReferenceArgument(script.str()) +
            " 2>&1";

        LOG_INFO(
            "[Qwen3.5 MoE MTP Parity] Resolving "
            << unique_branches.size()
            << " deferred branch trajectories with one HF model load after "
               "production residency retirement");
        FILE *pipe = popen(command.c_str(), "r");
        if (!pipe)
        {
            if (error)
                *error = "could not start deferred HF branch generator";
            return false;
        }
        std::string output;
        std::array<char, 512> buffer{};
        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr)
            output += buffer.data();
        const int exit_code = pclose(pipe);
        if (exit_code != 0)
        {
            if (error)
                *error = "deferred HF branch generation failed:\n" + output;
            return false;
        }

        for (const auto &context : contexts)
        {
            if (!compareDeferredMTPBranchContext(context, error))
                return false;
        }
        return true;
    }

} // namespace

class Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold
    : public Qwen35MoEConfigDrivenParityTest<Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold>
{
public:
    /**
     * @brief Resolve forced-branch HF evidence after production residency ends.
     *
     * GoogleTest calls this only after every fixture TearDown has destroyed its
     * runner. Releasing the external prepared-weight cache on every MPI process
     * then creates an explicit memory phase boundary: the sole artifact owner
     * loads Hugging Face once for the union of observed branches, while every
     * other rank waits without retaining a model authority. The final broadcast
     * makes a reference or numerical failure visible to the complete test world.
     */
    static void TearDownTestSuite()
    {
        releaseQwen122OverlayModelContextCampaignCache();
        MPI_Barrier(MPI_COMM_WORLD);

        auto &campaign = deferredMTPBranchCampaign();
        std::vector<DeferredMTPBranchContext> local_contexts;
        {
            std::lock_guard<std::mutex> lock(campaign.mutex);
            local_contexts = std::move(campaign.contexts);
            campaign.contexts.clear();
        }

        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        const int local_owner = local_contexts.empty() ? 0 : rank + 1;
        const int local_owner_count = local_contexts.empty() ? 0 : 1;
        int owner = 0;
        int owner_count = 0;
        MPI_Allreduce(
            &local_owner,
            &owner,
            1,
            MPI_INT,
            MPI_MAX,
            MPI_COMM_WORLD);
        MPI_Allreduce(
            &local_owner_count,
            &owner_count,
            1,
            MPI_INT,
            MPI_SUM,
            MPI_COMM_WORLD);
        if (owner_count == 0)
            return;

        EXPECT_EQ(owner_count, 1)
            << "Deferred HF branch evidence had more than one artifact authority";
        if (owner_count != 1 || owner <= 0)
            return;
        --owner;

        int success = 1;
        std::string error;
        if (rank == owner)
        {
            success = resolveDeferredMTPBranchCampaign(
                          local_contexts, &error)
                          ? 1
                          : 0;
            if (success == 0)
            {
                LOG_ERROR(
                    "[Qwen3.5 MoE MTP Parity] Deferred branch campaign failed: "
                    << error);
            }
        }
        MPI_Bcast(&success, 1, MPI_INT, owner, MPI_COMM_WORLD);
        EXPECT_EQ(success, 1)
            << (rank == owner
                    ? error
                    : "Deferred HF branch campaign failed on artifact authority rank " +
                          std::to_string(owner));
        MPI_Barrier(MPI_COMM_WORLD);
    }

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

    /** @brief Return one immutable config per exact 122B matrix cell. */
    static const TestConfig &qwen122Cuda2Rocm4Config()
    {
        static std::map<std::string, TestConfig> configs;
        const std::string name = activeTestName();
        const auto [it, inserted] = configs.try_emplace(
            name,
            makeGraphNativeTestConfig(OverlayTopology::Cuda2Rocm4));
        (void)inserted;
        return it->second;
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
        case OverlayTopology::Cuda2Rocm4:
            return qwen122Cuda2Rocm4Config();
        }
        throw std::logic_error("Unhandled Qwen3.5 MoE parity topology config");
    }

protected:
    using Base = Qwen35MoEConfigDrivenParityTest<Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold>;

    /** @brief The 122B matrix mathematically compares recursive MTP sidecars. */
    bool requiresMTPSidecarReferenceSnapshots() const override
    {
        return isQwen122ProductionTest();
    }

    /** @return Maximum recurrent sidecar depth admitted by the 122B matrix. */
    int requiredMTPSidecarReferenceDraftDepth() const override
    {
        return isQwen122ProductionTest()
                   ? kQwen122MaximumMTPDraftDepth
                   : 0;
    }

    /** @return Whether this process intentionally runs consecutive 122B cells. */
    bool mayReuseQwen122OverlayModelContext() const
    {
        return isQwen122ProductionTest() &&
               DebugEnv::isTruthyEnv(
                   "LLAMINAR_PRODUCTION_PARITY_PROCESS_CAMPAIGN");
    }

    /**
     * @return Cache slot selected by every physical-capacity identity axis.
     *
     * The two bits encode residency policy and whole-expert owner order. Every
     * cell reserves the same maximum retained graph geometry, matching the
     * production reuse identity without coupling physical placement to the
     * fixed depth selected for one request.
     */
    std::size_t qwen122ModelContextCacheSlot() const noexcept
    {
        const std::size_t policy =
            isDynamicResidencyProductionTest() ? 4u : 0u;
        const std::size_t owner_order =
            isRandomOwnerProductionTest() ? 2u : 0u;
        return policy + owner_order;
    }

    /**
     * @brief Find the prior runner's rank-local prepared-weight certificate.
     *
     * A miss does not synthesize ModelContext configuration from test metadata;
     * only a previously initialized production runner may populate the slot.
     */
    std::optional<ModelContextReuseContract>
    findQwen122OverlayModelContext(
        bool *cache_hit,
        std::string *error) const
    {
        if (cache_hit)
            *cache_hit = false;
        if (!mayReuseQwen122OverlayModelContext())
            return std::nullopt;

        auto &cache = qwen122OverlayModelContextCampaignCache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (!cache.model_path.empty() &&
            cache.model_path != config_.model_path)
        {
            if (error)
                *error = "122B campaign cache belongs to a different model path";
            return std::nullopt;
        }

        const std::size_t requested_slot = qwen122ModelContextCacheSlot();
        if (!cache.contract || !cache.physical_identity_slot)
            return std::nullopt;
        if (*cache.physical_identity_slot != requested_slot)
        {
            /*
             * The prior runner has already torn down all mutable state. Drop
             * the cache's final model-owned reference now so GPU allocations
             * are returned before automatic capacity observes free memory for
             * the incompatible physical plan.
             */
            cache.contract.reset();
            cache.physical_identity_slot.reset();
            PerfStatsCollector::addCounter(
                "weight_loading",
                "parity_campaign_model_context_cache_evictions",
                1.0,
                "setup",
                {},
                {{"next_physical_identity_slot",
                  std::to_string(requested_slot)}});
            return std::nullopt;
        }
        if (cache_hit)
            *cache_hit = true;
        return cache.contract;
    }

    /**
     * @brief Publish one initialized runner's immutable rank-local authority.
     *
     * The cache never replaces a live slot: doing so could conceal a teardown
     * or identity defect. The contract itself remains the production source of
     * truth for participant topology and prepared-weight compatibility.
     */
    bool publishQwen122OverlayModelContext(
        const ModelContextReuseContract &contract,
        std::string *error) const
    {
        if (!mayReuseQwen122OverlayModelContext() || !contract.context ||
            contract.routed_weight_authority_identity.empty())
        {
            if (error)
            {
                *error = "cannot publish an ineligible, null, or uncertified "
                         "122B ExpertOverlay model authority";
            }
            return false;
        }
        if (contract.context->path() != config_.model_path)
        {
            if (error)
                *error = "production runner returned a different model authority";
            return false;
        }

        auto &cache = qwen122OverlayModelContextCampaignCache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (!cache.model_path.empty() &&
            cache.model_path != config_.model_path)
        {
            if (error)
                *error = "122B campaign attempted to replace a live model path";
            return false;
        }

        const std::size_t requested_slot = qwen122ModelContextCacheSlot();
        if (cache.contract)
        {
            if (cache.physical_identity_slot == requested_slot &&
                cache.contract->context == contract.context &&
                cache.contract->routed_weight_authority_identity ==
                    contract.routed_weight_authority_identity)
            {
                return true;
            }
            if (error)
                *error = "122B campaign attempted to replace a live physical-capacity authority";
            return false;
        }

        cache.model_path = config_.model_path;
        cache.physical_identity_slot = requested_slot;
        cache.contract = contract;
        return true;
    }

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
        // Preserve the complete declarative TestConfig contract, including
        // decode depth. Reimplementing only model/prompt fields here left the
        // 122B MTP corpus at ParityConfig's historical five-step default.
        // The shared parity base is also the sole reference-pack lifecycle
        // authority: it validates model identity and sidecar schema, performs
        // one rank-zero regeneration, then publishes readiness to every rank.
        Base::applyModelOverrides();
    }

    bool broadcastRootFlag(bool root_value) const
    {
        const int root_rank = parityArtifactAuthorityRank();
        int flag = isRootParityRank() && root_value ? 1 : 0;
        MPI_Bcast(&flag, 1, MPI_INT, root_rank, MPI_COMM_WORLD);
        return flag != 0;
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
        /*
         * A continuation domain may itself be tensor-parallel. Every one of
         * those participants reuses its dense model arena; only participants
         * outside that domain own the compact follower arena. Derive the exact
         * set from the published owner-map topology instead of assuming one
         * continuation participant at global id zero.
         */
        std::vector<uint64_t> expected_compact_participant(
            participant_count, 0u);
        int topology_valid = 1;
        if (isRootParityRank())
        {
            const auto *const concrete =
                dynamic_cast<const OrchestrationRunner *>(orch_runner_.get());
            const auto snapshot = concrete
                                      ? concrete
                                            ->expertOverlayResidencySnapshotForDiagnostics()
                                      : nullptr;
            if (!snapshot || !snapshot->valid() || !overlay_plan_)
            {
                topology_valid = 0;
            }
            else
            {
                for (const auto &participant :
                     snapshot->owner_map.participants())
                {
                    if (participant.participant_id < 0 ||
                        static_cast<size_t>(participant.participant_id) >=
                            participant_count)
                    {
                        topology_valid = 0;
                        break;
                    }
                    if (participant.domain_name !=
                        overlay_plan_->continuation_domain)
                    {
                        expected_compact_participant[
                            static_cast<size_t>(participant.participant_id)] =
                            1u;
                    }
                }
            }
        }
        MPI_Bcast(
            &topology_valid,
            1,
            MPI_INT,
            parityArtifactAuthorityRank(),
            MPI_COMM_WORLD);
        MPI_Bcast(
            expected_compact_participant.data(),
            static_cast<int>(expected_compact_participant.size()),
            MPI_UINT64_T,
            parityArtifactAuthorityRank(),
            MPI_COMM_WORLD);
        ASSERT_EQ(topology_valid, 1)
            << "Published ExpertOverlay topology could not classify compact followers";
        const size_t follower_participant_count =
            static_cast<size_t>(std::accumulate(
                expected_compact_participant.begin(),
                expected_compact_participant.end(),
                uint64_t{0}));
        ASSERT_GT(follower_participant_count, 0u);
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
        for (size_t participant = 0;
             participant < participant_count;
             ++participant)
        {
            const uint64_t expected =
                expected_compact_participant[participant];
            EXPECT_EQ(global[kParticipantStart + participant], expected)
                << (expected != 0u
                        ? "Expected exactly one follower compact arena for participant p"
                        : "Dense continuation participant must not duplicate a compact arena for p")
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
        constexpr size_t kFamilyMaterializations = 1;
        constexpr size_t kMainFamilyMaterializations = 2;
        constexpr size_t kMTPFamilyCapacity = 3;
        constexpr size_t kSelectedGraphs = 4;
        constexpr size_t kLargestPrefillSelections = 5;
        constexpr size_t kOneRowPrefillSelections = 6;
        constexpr size_t kOneRowDecodeSelections = 7;
        constexpr size_t kSelectedMTPPhysicalRows = 8;
        constexpr size_t kRetiredHostRunnerRecords = 9;
        constexpr size_t kMaterializedGpuTransactions = 10;
        constexpr size_t kMaterializedCpuEndpoints = 11;
        constexpr size_t kFollowerGpuRuntimeTables = 12;
        constexpr size_t kHostAuthorityGpuRuntimes = 13;
        constexpr size_t kDeviceAuthorityGpuRuntimes = 14;
        constexpr size_t kMappedControllerGpuRuntimes = 15;
        constexpr size_t kEvidenceCount = 16;

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
            const auto parse_nonnegative = [&](const char *name) -> int
            {
                const auto value = tag(name);
                if (!value)
                    return -1;
                int parsed = -1;
                const char *const begin = value->data();
                const char *const end = begin + value->size();
                const auto result = std::from_chars(begin, end, parsed);
                return result.ec == std::errc{} && result.ptr == end &&
                               parsed >= 0
                           ? parsed
                           : -1;
            };

            if (record.name == "materialized_mapped_follower_families")
            {
                const auto graph_family = tag("graph_family");
                const auto standalone_progress =
                    tag("standalone_progress_launch");
                const int graph_family_ordinal =
                    parse_nonnegative("graph_family");
                const int row_capacity = parse_positive("row_capacity");
                const int gpu_transactions = parse_nonnegative(
                    "setup_materialized_gpu_transactions");
                const int cpu_endpoints = parse_nonnegative(
                    "setup_materialized_cpu_endpoints");
                const bool valid =
                    record.phase == "model_setup" && record.value == 1.0 &&
                    record.count == 1u && graph_family &&
                    graph_family_ordinal >= 0 &&
                    row_capacity > 0 && gpu_transactions >= 0 &&
                    cpu_endpoints >= 0 &&
                    (gpu_transactions > 0 || cpu_endpoints > 0) &&
                    parse_nonnegative("captured_transfer_branches") >= 0 &&
                    standalone_progress &&
                    *standalone_progress == "false";
                if (!valid)
                {
                    ++local[kMalformedRecords];
                    continue;
                }
                ++local[kFamilyMaterializations];
                local[kMaterializedGpuTransactions] +=
                    static_cast<uint64_t>(gpu_transactions);
                local[kMaterializedCpuEndpoints] +=
                    static_cast<uint64_t>(cpu_endpoints);
                if (graph_family_ordinal == 0)
                    ++local[kMainFamilyMaterializations];
                if (graph_family_ordinal == 1 &&
                    row_capacity == activeMTPGraphCapacityVerifierRows())
                {
                    ++local[kMTPFamilyCapacity];
                }
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
            if (record.phase == "mtp_grouped_verifier" &&
                physical_rows == activeMTPPhysicalVerifierRows())
            {
                local[kSelectedMTPPhysicalRows] += record.count;
            }
        }

        /*
         * Policy location and follower execution state are orthogonal. Every
         * remote GPU endpoint needs a device runtime table even when a CPU
         * participant makes the sole policy authority host-resident. This is
         * the focused production-path regression for the former manual
         * dispatch-consume segment in an otherwise device-owned envelope.
         */
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"moe_overlay_controller"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "moe_overlay_controller" ||
                record.name != "follower_runtime_tables_materialized")
            {
                continue;
            }
            const auto tag = [&record](const char *name)
                -> const std::string *
            {
                const auto found = record.tags.find(name);
                return found == record.tags.end() ? nullptr : &found->second;
            };
            const auto authority = tag("authority_execution");
            const auto mapped = tag("mapped_device_controller");
            const auto blocking = tag("blocking_hot_path");
            const auto layers = tag("layers");
            int parsed_layers = 0;
            if (layers)
            {
                const char *const begin = layers->data();
                const char *const end = begin + layers->size();
                const auto parsed = std::from_chars(
                    begin, end, parsed_layers);
                if (parsed.ec != std::errc{} || parsed.ptr != end)
                    parsed_layers = 0;
            }
            const bool valid =
                record.phase == "model_setup" && record.value == 1.0 &&
                record.count == 1u && authority && mapped && blocking &&
                *blocking == "false" && parsed_layers > 0 &&
                (*authority == "host-resident" ||
                 *authority == "device-resident") &&
                (*mapped == "true" || *mapped == "false") &&
                ((*authority == "host-resident" && *mapped == "false") ||
                 (*authority == "device-resident" && *mapped == "true"));
            if (!valid)
            {
                ++local[kMalformedRecords];
                continue;
            }
            ++local[kFollowerGpuRuntimeTables];
            if (*authority == "host-resident")
                ++local[kHostAuthorityGpuRuntimes];
            else
                ++local[kDeviceAuthorityGpuRuntimes];
            if (*mapped == "true")
                ++local[kMappedControllerGpuRuntimes];
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
        /*
         * Qwen3.5 owns one set of MTP sidecar weights and recursively replays
         * that same retained graph for every speculative slot. Draft depth is
         * therefore transaction geometry, not graph-family identity. A
         * non-MTP cell materializes only the main family; an MTP cell adds
         * exactly one recursive sidecar. Deriving this count from the actual
         * production request prevents the evidence gate from inventing MTP
         * work in the 35B correctness campaign.
         */
        const uint64_t expected_family_materializations =
            isQwen122ProductionTest() ? 2u : 1u;
        EXPECT_EQ(
            global[kFamilyMaterializations],
            expected_family_materializations)
            << "The auxiliary MPI runner materialized a graph family outside the production request";
        EXPECT_EQ(global[kMainFamilyMaterializations], 1u)
            << "The one auxiliary MPI runner must materialize its main mapped graph family exactly once";
        const bool has_remote_gpu_endpoint =
            activeTopology() == OverlayTopology::CudaRocmCpu ||
            activeTopology() == OverlayTopology::CudaRocm ||
            activeTopology() == OverlayTopology::Cuda2Rocm4;
        if (has_remote_gpu_endpoint)
        {
            EXPECT_GT(global[kMaterializedGpuTransactions], 0u)
                << "The mapped follower family materialized no native GPU transaction";
            EXPECT_GT(global[kFollowerGpuRuntimeTables], 0u)
                << "A remote GPU follower executed without a device-resident placement table";
            if (topologyUsesCpu())
            {
                EXPECT_EQ(
                    global[kHostAuthorityGpuRuntimes],
                    global[kFollowerGpuRuntimeTables])
                    << "CPU-participating topology must retain one host policy authority while every GPU keeps local execution state";
                EXPECT_EQ(global[kMappedControllerGpuRuntimes], 0u)
                    << "Host-authority GPU followers must not acquire a competing mapped policy controller";
            }
            else
            {
                EXPECT_EQ(
                    global[kDeviceAuthorityGpuRuntimes],
                    global[kFollowerGpuRuntimeTables]);
                EXPECT_EQ(
                    global[kMappedControllerGpuRuntimes],
                    global[kFollowerGpuRuntimeTables])
                    << "All-GPU followers must bind the sole mapped device policy controller";
            }
        }
        if (topologyUsesCpu())
        {
            EXPECT_GT(global[kMaterializedCpuEndpoints], 0u)
                << "The mapped follower family retained no typed CPU boundary endpoint";
        }
        if (isQwen122ProductionTest())
        {
            EXPECT_EQ(global[kMTPFamilyCapacity], 1u)
                << "The recursive MTP follower family did not retain the shared "
                << activeMTPGraphCapacityVerifierRows()
                << "-row campaign capacity";
        }
        else
        {
            EXPECT_EQ(global[kMTPFamilyCapacity], 0u)
                << "An MTP-disabled campaign materialized a recursive sidecar family";
        }
        EXPECT_GT(global[kSelectedGraphs], 0u)
            << "No authenticated ticket selected a mapped participant graph";
        EXPECT_GT(global[kLargestPrefillSelections], 0u)
            << "The mapped family never served the largest root-published live prefill chunk";
        EXPECT_GT(global[kOneRowDecodeSelections], 0u)
            << "Decode never selected its setup-owned one-row retained parent";
        if (isQwen122ProductionTest())
        {
            EXPECT_GT(global[kSelectedMTPPhysicalRows], 0u)
                << "No mapped grouped-verifier graph selected the admitted "
                << activeMTPPhysicalVerifierRows()
                << "-row physical transaction bucket";
        }
        EXPECT_EQ(global[kRetiredHostRunnerRecords], 0u)
            << "The retired host-scheduled participant graph path executed";
        if (isSegmentedPrefillProductionTest())
        {
            EXPECT_GT(global[kOneRowPrefillSelections], 0u)
                << "The short prefill tail never selected its bounded mapped graph";
        }
    }

    /**
     * @brief Exact device snapshots for one routed layer under one request.
     *
     * `overlay_participants` is the overlay-wide `(expert -> global
     * participant)` bank selected by the request's acquired epoch status.
     * `domain_participants` is the invocation-local `(route slot -> domain
     * participant)` schedule; `-1` means the selected expert belongs to another
     * overlay domain and is completed by the heterogeneous return transaction.
     * Both pointers remain owned by SnapshotCapture.
     */
    struct PinnedDeviceRouteEvidence
    {
        const float *overlay_participants = nullptr;
        const float *domain_participants = nullptr;
        const float *runtime_weights = nullptr;
        size_t expert_count = 0u;
        size_t route_count = 0u;
        int selected_bank = -1;
    };

    /**
     * @brief Resolve both typed route projections from live device checkpoints.
     *
     * @param layer Exact transformer layer.
     * @param expected_route_count Router slots in the current checkpoint.
     * @param expected_expert_count Logical routed experts in the model layer.
     * @return Complete evidence, or no value after recording a test failure.
     */
    std::optional<PinnedDeviceRouteEvidence>
    pinnedDeviceRouteEvidence(
        int layer,
        size_t expected_route_count,
        size_t expected_expert_count) const
    {
        const std::string prefix = "layer" + std::to_string(layer) + '_';
        size_t domain_elements = 0u;
        const float *const domain_participants = activeSnapshot(
            prefix + "MOE_DOMAIN_ROUTE_PARTICIPANT_IDS",
            domain_elements);
        size_t runtime_weight_elements = 0u;
        const float *const runtime_weights = activeSnapshot(
            prefix + "MOE_RUNTIME_ROUTE_WEIGHTS",
            runtime_weight_elements);
        size_t bank0_elements = 0u;
        const float *const bank0 = activeSnapshot(
            prefix + "MOE_OVERLAY_ROUTE_PARTICIPANTS_BANK0",
            bank0_elements);
        size_t bank1_elements = 0u;
        const float *const bank1 = activeSnapshot(
            prefix + "MOE_OVERLAY_ROUTE_PARTICIPANTS_BANK1",
            bank1_elements);
        size_t selected_bank_elements = 0u;
        const float *const selected_bank_value = activeSnapshot(
            prefix + "MOE_OVERLAY_ROUTE_SELECTED_BANK",
            selected_bank_elements);
        if (!domain_participants || !runtime_weights || !bank0 || !bank1 ||
            !selected_bank_value)
        {
            ADD_FAILURE()
                << prefix
                << "mapped reducer did not publish both pinned route projections";
            return std::nullopt;
        }
        if (domain_elements != expected_route_count ||
            runtime_weight_elements != expected_route_count ||
            bank0_elements != expected_expert_count ||
            bank1_elements != expected_expert_count ||
            selected_bank_elements != 1u)
        {
            ADD_FAILURE()
                << prefix << "route evidence geometry mismatch: domain="
                << domain_elements << " expected_routes="
                << expected_route_count << " bank0=" << bank0_elements
                << " runtime_weights=" << runtime_weight_elements
                << " bank1=" << bank1_elements << " expected_experts="
                << expected_expert_count << " selected_bank_elements="
                << selected_bank_elements;
            return std::nullopt;
        }
        const float selected = selected_bank_value[0];
        if (!std::isfinite(selected) ||
            (selected != 0.0f && selected != 1.0f))
        {
            ADD_FAILURE()
                << prefix << "acquired route epoch selected invalid bank "
                << selected;
            return std::nullopt;
        }
        const int selected_bank = static_cast<int>(selected);
        return PinnedDeviceRouteEvidence{
            .overlay_participants = selected_bank == 0 ? bank0 : bank1,
            .domain_participants = domain_participants,
            .runtime_weights = runtime_weights,
            .expert_count = expected_expert_count,
            .route_count = expected_route_count,
            .selected_bank = selected_bank,
        };
    }

    /**
     * @brief Attribute live parity router checkpoints to the published epoch.
     *
     * Snapshot values are exact integer expert ids produced by the real router.
     * Pair them with the request-selected global placement bank and the final
     * domain-local schedule while all checkpoints remain live, before the
     * parity harness clears diagnostics. The setup-time residency snapshot
     * supplies only stable endpoint topology; it is deliberately not used to
     * reconstruct live placement after Dynamic movement.
     */
    void cacheDeviceRouteAssignmentEvidence()
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
            const auto route_evidence = pinnedDeviceRouteEvidence(
                placement.layer,
                route_elements,
                placement.routed_expert_tier.size());
            if (!route_evidence)
                return;
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
                const float global_assigned =
                    route_evidence->overlay_participants[expert];
                if (!std::isfinite(global_assigned) ||
                    global_assigned < 0.0f || global_assigned >
                        static_cast<float>(
                            std::numeric_limits<int>::max()))
                {
                    ADD_FAILURE()
                        << "Pinned overlay placement for " << key
                        << " contains an invalid global participant for expert "
                        << expert;
                    return;
                }
                const int participant = static_cast<int>(global_assigned);
                const auto *const global_endpoint =
                    snapshot->owner_map.participantForId(participant);
                if (global_assigned != static_cast<float>(participant) ||
                    participant < 0 ||
                    static_cast<size_t>(participant) >= participant_count ||
                    !global_endpoint)
                {
                    ADD_FAILURE()
                        << "Pinned overlay placement for " << key
                        << " names unknown global participant "
                        << global_assigned;
                    return;
                }

                const float domain_assigned =
                    route_evidence->domain_participants[index];
                if (!std::isfinite(domain_assigned) ||
                    domain_assigned < -1.0f || domain_assigned >
                        static_cast<float>(
                            std::numeric_limits<int>::max()))
                {
                    ADD_FAILURE()
                        << "Domain route schedule for " << key
                        << " contains invalid participant " << domain_assigned
                        << " at element " << index;
                    return;
                }
                const int domain_participant =
                    static_cast<int>(domain_assigned);
                if (domain_assigned !=
                    static_cast<float>(domain_participant))
                {
                    ADD_FAILURE()
                        << "Domain route schedule for " << key
                        << " contains non-integral participant "
                        << domain_assigned;
                    return;
                }
                const bool belongs_to_continuation =
                    global_endpoint->domain_name ==
                    overlay_plan_->continuation_domain;
                if (!belongs_to_continuation && domain_participant != -1)
                {
                    ADD_FAILURE()
                        << "Remote-domain expert " << expert << " in " << key
                        << " must retain the -1 domain-route sentinel, observed "
                        << domain_participant;
                    return;
                }
                if (belongs_to_continuation)
                {
                    const auto local_endpoint = std::find_if(
                        snapshot->owner_map.participants().begin(),
                        snapshot->owner_map.participants().end(),
                        [&](const MoEExpertOwnerParticipant &candidate)
                        {
                            return candidate.domain_name ==
                                       overlay_plan_->continuation_domain &&
                                   candidate.domain_participant_index ==
                                       domain_participant;
                        });
                    if (domain_participant < 0 ||
                        local_endpoint ==
                            snapshot->owner_map.participants().end())
                    {
                        ADD_FAILURE()
                            << "Continuation-domain expert " << expert
                            << " in " << key
                            << " names unknown domain participant "
                            << domain_participant;
                        return;
                    }
                }
                ++route_counts[static_cast<size_t>(participant)];
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
     * @brief Persist immutable setup ownership and participant topology.
     *
     * The host residency snapshot is the cold-start topology authority. In an
     * all-GPU Dynamic cell it intentionally does not shadow later device-owned
     * placement epochs. Keep the established `expert_owner_map.csv` artifact
     * as the setup baseline and endpoint dictionary; exact live assignments
     * are recorded per route in `prefill_routed_expert_routes.csv` from the
     * reducer's device ledger.
     */
    void writeExpertOwnerTopologyBaselineCsv() const
    {
        if (!isRootParityRank())
            return;

        const auto *const concrete =
            dynamic_cast<const OrchestrationRunner *>(orch_runner_.get());
        const auto snapshot = concrete
                                  ? concrete->expertOverlayResidencySnapshotForDiagnostics()
                                  : nullptr;
        ASSERT_NE(snapshot, nullptr);
        ASSERT_TRUE(snapshot->valid());

        const auto path = ensureResultsDir() / "expert_owner_map.csv";
        std::ofstream output(path, std::ios::trunc);
        ASSERT_TRUE(output.is_open()) << path;
        output
            << "epoch,layer,expert,tier_index,tier_name,domain_name,"
               "participant,domain_participant,world_rank,device,resident\n";
        for (const auto &owner : snapshot->owner_map.owners())
        {
            output
                << snapshot->epoch << ',' << owner.layer_idx << ','
                << owner.expert_id << ',' << owner.tier_idx << ','
                << owner.tier_name << ',' << owner.domain_name << ','
                << owner.owner_participant << ','
                << owner.domain_participant_index << ','
                << owner.owner_world_rank << ',' << owner.device.toString()
                << ',' << (owner.resident ? 1 : 0) << '\n';
        }
        output.flush();
        EXPECT_TRUE(output.good()) << path;
    }

    /**
     * @brief Persist value-level routed-expert evidence for baseline and moves.
     *
     * Aggregate cosine metrics cannot distinguish a missing participant from
     * a correct route computed with the wrong weight slice.  The baseline
     * layer and every layer containing a committed promotion are therefore
     * recorded element by element against Hugging Face.  Per-route norms also
     * expose a zero, duplicated, or explosive migrated contribution without
     * requiring another instrumented inference run.  This diagnostic executes
     * only after the captured production forward and before snapshot teardown.
     */
    void writePrefillRoutedExpertDiagnosticCsv()
    {
        if (!isRootParityRank())
            return;

        std::vector<int> diagnostic_layers{0};
        for (const auto &promotion : promoted_experts_)
        {
            if (promotion.layer >= 0 &&
                std::find(
                    diagnostic_layers.begin(),
                    diagnostic_layers.end(),
                    promotion.layer) == diagnostic_layers.end())
            {
                diagnostic_layers.push_back(promotion.layer);
            }
        }
        std::sort(diagnostic_layers.begin(), diagnostic_layers.end());

        const auto values_path =
            ensureResultsDir() / "prefill_routed_expert_values.csv";
        std::ofstream values(values_path, std::ios::trunc);
        ASSERT_TRUE(values.is_open()) << values_path;
        values << std::setprecision(9);
        values
            << "layer,row,column,llaminar,pytorch,difference,"
               "continuation_local_sum\n";

        const auto *const concrete =
            dynamic_cast<const OrchestrationRunner *>(orch_runner_.get());
        const auto snapshot = concrete
                                  ? concrete->expertOverlayResidencySnapshotForDiagnostics()
                                  : nullptr;
        ASSERT_NE(snapshot, nullptr);
        ASSERT_TRUE(snapshot->valid());

        const auto routes_path =
            ensureResultsDir() / "prefill_routed_expert_routes.csv";
        std::ofstream route_output(routes_path, std::ios::trunc);
        ASSERT_TRUE(route_output.is_open()) << routes_path;
        route_output
            << "layer,row,slot,expert,weight,runtime_weight,participant,domain_participant,"
               "selected_placement_bank,tier_index,"
               "domain_name,world_rank,device,canonical_l2,"
               "canonical_max_abs,canonical_nonzero,canonical_first\n";
        route_output << std::setprecision(9);

        const auto *const model = activeModelContextForDiagnostics();
        ASSERT_NE(model, nullptr);
        const size_t width = static_cast<size_t>(
            model->model().embedding_length);
        ASSERT_GT(width, 0u);

        for (const int diagnostic_layer : diagnostic_layers)
        {
            const std::string prefix =
                "layer" + std::to_string(diagnostic_layer) + '_';
            size_t actual_elements = 0u;
            const float *const actual = activeSnapshot(
                prefix + "MOE_EXPERT_OUTPUT", actual_elements);
            const auto reference = loadPyTorchSnapshot(
                prefix + "MOE_EXPERT_OUTPUT");
            ASSERT_NE(actual, nullptr) << prefix;
            ASSERT_EQ(actual_elements, reference.size()) << prefix;
            ASSERT_EQ(actual_elements % width, 0u) << prefix;
            const size_t rows = actual_elements / width;

            size_t route_elements = 0u;
            size_t weight_elements = 0u;
            const float *const routes = activeSnapshot(
                prefix + "MOE_ROUTING_INDICES", route_elements);
            const float *const weights = activeSnapshot(
                prefix + "MOE_ROUTING_WEIGHTS", weight_elements);
            ASSERT_NE(routes, nullptr) << prefix;
            ASSERT_NE(weights, nullptr) << prefix;
            ASSERT_EQ(route_elements, weight_elements) << prefix;
            ASSERT_GT(rows, 0u) << prefix;
            ASSERT_EQ(route_elements % rows, 0u) << prefix;
            const size_t top_k = route_elements / rows;
            const auto placement = std::find_if(
                snapshot->placement_plan->placements.begin(),
                snapshot->placement_plan->placements.end(),
                [diagnostic_layer](const RoutedExpertLayerPlacement &entry)
                { return entry.layer == diagnostic_layer; });
            ASSERT_NE(
                placement,
                snapshot->placement_plan->placements.end())
                << prefix << " has no declared routed-expert placement";
            const auto route_evidence = pinnedDeviceRouteEvidence(
                diagnostic_layer,
                route_elements,
                placement->routed_expert_tier.size());
            ASSERT_TRUE(route_evidence.has_value()) << prefix;

            size_t canonical_elements = 0u;
            const float *const canonical = activeSnapshot(
                prefix + "MOE_CANONICAL_ROUTE_CONTRIBUTIONS",
                canonical_elements);
            ASSERT_NE(canonical, nullptr)
                << prefix
                << " continuation did not retain canonical route-slot evidence";
            ASSERT_EQ(canonical_elements, route_elements * width) << prefix;

            std::vector<float> continuation_local_sum(
                actual_elements, 0.0f);
            for (size_t row = 0u; row < rows; ++row)
            {
                for (size_t slot = 0u; slot < top_k; ++slot)
                {
                    const size_t index = row * top_k + slot;
                    const size_t route_offset = index * width;
                    const size_t output_offset = row * width;
                    double squared_norm = 0.0;
                    float max_abs = 0.0f;
                    size_t nonzero = 0u;
                    for (size_t column = 0u; column < width; ++column)
                    {
                        const float contribution =
                            canonical[route_offset + column];
                        continuation_local_sum[output_offset + column] +=
                            contribution;
                        squared_norm += static_cast<double>(contribution) *
                                        static_cast<double>(contribution);
                        max_abs = std::max(max_abs, std::abs(contribution));
                        nonzero += contribution != 0.0f ? 1u : 0u;
                    }

                    ASSERT_TRUE(std::isfinite(routes[index]));
                    const int expert = static_cast<int>(routes[index]);
                    ASSERT_EQ(routes[index], static_cast<float>(expert));
                    ASSERT_GE(expert, 0);
                    ASSERT_LT(
                        static_cast<size_t>(expert),
                        route_evidence->expert_count);
                    const float global_assigned =
                        route_evidence->overlay_participants[expert];
                    ASSERT_TRUE(std::isfinite(global_assigned));
                    const int participant =
                        static_cast<int>(global_assigned);
                    ASSERT_EQ(
                        global_assigned,
                        static_cast<float>(participant))
                        << "Non-integral global route participant at layer "
                        << diagnostic_layer << " route slot " << index;
                    const float domain_assigned =
                        route_evidence->domain_participants[index];
                    ASSERT_TRUE(std::isfinite(domain_assigned));
                    const int domain_participant =
                        static_cast<int>(domain_assigned);
                    ASSERT_EQ(
                        domain_assigned,
                        static_cast<float>(domain_participant))
                        << "Non-integral domain route participant at layer "
                        << diagnostic_layer << " route slot " << index;
                    const float runtime_weight =
                        route_evidence->runtime_weights[index];
                    ASSERT_TRUE(std::isfinite(runtime_weight))
                        << "Non-finite runtime route weight at layer "
                        << diagnostic_layer << " route slot " << index;
                    const auto *const assigned_endpoint =
                        snapshot->owner_map.participantForId(participant);
                    ASSERT_NE(assigned_endpoint, nullptr)
                        << "Missing endpoint metadata for layer "
                        << diagnostic_layer << " expert " << expert
                        << " assigned participant " << participant;
                    route_output
                        << diagnostic_layer << ',' << row << ',' << slot
                        << ',' << expert << ',' << weights[index] << ','
                        << runtime_weight << ','
                        << participant << ',' << domain_participant << ','
                        << route_evidence->selected_bank << ','
                        << assigned_endpoint->tier_idx
                        << ',' << assigned_endpoint->domain_name << ','
                        << assigned_endpoint->world_rank << ','
                        << assigned_endpoint->device.toString() << ','
                        << std::sqrt(squared_norm) << ',' << max_abs << ','
                        << nonzero << ',' << canonical[route_offset] << '\n';
                }
            }

            for (size_t index = 0u; index < actual_elements; ++index)
            {
                values
                    << diagnostic_layer << ',' << index / width << ','
                    << index % width << ',' << actual[index] << ','
                    << reference[index] << ','
                    << (actual[index] - reference[index]) << ','
                    << continuation_local_sum[index] << '\n';
            }
        }
        values.flush();
        EXPECT_TRUE(values.good()) << values_path;
        route_output.flush();
        EXPECT_TRUE(route_output.good()) << routes_path;
    }

    /**
     * @brief Prove every declared participant was selected and remote work completed.
     *
     * The routed checkpoint and device assignment ledger prove selection under
     * the exact execution-time route decision. Cross-rank participants
     * additionally require endpoint-owned traffic totals acquired only after
     * both retained graphs publish Complete. Local participant arithmetic is
     * covered by the same per-layer/LM-head parity comparison.
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
        case OverlayTopology::Cuda2Rocm4:
            expected = {
                {0, 0, "CUDA", "CUDA priority-0 participant 0"},
                {1, 0, "CUDA", "CUDA priority-0 participant 1"},
                {2, 1, "ROCm", "ROCm priority-1 participant 0"},
                {3, 1, "ROCm", "ROCm priority-1 participant 1"},
                {4, 1, "ROCm", "ROCm priority-1 participant 2"},
                {5, 1, "ROCm", "ROCm priority-1 participant 3"},
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
            if (record.name != "device_epoch_cpu_rows" &&
                record.name != "device_epoch_gpu_rows")
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
            const bool cpu_record =
                record.name == "device_epoch_cpu_rows";
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
        const bool has_remote_gpu_endpoint =
            activeTopology() == OverlayTopology::CudaRocmCpu ||
            activeTopology() == OverlayTopology::CudaRocm ||
            activeTopology() == OverlayTopology::Cuda2Rocm4;
        if (has_remote_gpu_endpoint)
        {
            EXPECT_GT(global[kGpuRows], 0u)
                << "No remote GPU tier published completed device-owned expert rows";
        }
        else
        {
            EXPECT_EQ(global[kGpuRows], 0u)
                << "A topology without a remote GPU endpoint published remote GPU rows";
        }
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
        constexpr size_t kSnapshotAggregationRecords = 1;
        constexpr size_t kContinuationTransactionRecords = 2;
        constexpr size_t kParticipantTransactionRecords = 3;
        constexpr size_t kContinuationCompleteSchedules = 4;
        constexpr size_t kParticipantCompleteSchedules = 5;
        constexpr size_t kMalformedRecords = 6;
        constexpr size_t kEvidenceCount = 7;
        constexpr uint64_t kExpectedChunkMask = 0x7u;

        struct ContinuationScheduleEvidence
        {
            uint64_t transactions = 0;
            uint64_t chunk_mask = 0;
            std::set<uint64_t> transaction_steps;
            bool valid = true;
        };
        struct ParticipantScheduleEvidence
        {
            uint64_t transactions = 0;
            uint64_t logical_rows = 0;
            uint64_t physical_rows = 0;
            uint64_t four_row_transactions = 0;
            uint64_t one_row_transactions = 0;
            uint64_t transaction_ordinal_mask = 0;
            std::set<uint64_t> transaction_steps;
            bool valid = true;
        };

        std::array<uint64_t, kEvidenceCount> local{};
        std::map<uint64_t, ContinuationScheduleEvidence>
            continuation_schedules;
        std::map<uint64_t, ParticipantScheduleEvidence>
            participant_schedules;
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"forward_graph", "moe_overlay_participant_graph"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter)
                continue;

            const auto tagEquals = [&record](
                                       const char *name,
                                       const std::string &expected)
            {
                const auto it = record.tags.find(name);
                return it != record.tags.end() && it->second == expected;
            };
            const auto unsignedTag = [&record](const char *name)
                -> std::optional<uint64_t>
            {
                const auto it = record.tags.find(name);
                if (it == record.tags.end() || it->second.empty())
                    return std::nullopt;
                uint64_t value = 0;
                const char *const begin = it->second.data();
                const char *const end = begin + it->second.size();
                const auto parsed = std::from_chars(begin, end, value);
                if (parsed.ec != std::errc{} || parsed.ptr != end)
                    return std::nullopt;
                return value;
            };

            if (record.domain == "forward_graph" &&
                record.name == "moe_overlay_prefill_schedule_contract_rows")
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

            if (record.domain == "forward_graph" &&
                record.name == "prefill_chunk_snapshot_sequence_keys")
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

            if (record.domain == "forward_graph" &&
                record.name == "moe_overlay_collective_transaction")
            {
                /*
                 * Decode parity performs its own prefill initialization, then
                 * emits serial decode transactions. Those records are covered
                 * by the decode comparator; they are not malformed prefill
                 * evidence merely because they share the sparse metric name.
                 */
                if (record.phase != "prefill")
                    continue;

                const auto generation = unsignedTag("generation");
                const auto transaction_step = unsignedTag("logical_step");
                const auto chunk_index = unsignedTag("prefill_chunk_index");
                const auto token_offset = unsignedTag("token_offset");
                const auto logical_rows = unsignedTag("logical_rows");
                const auto physical_rows = unsignedTag("physical_rows");
                const bool chunk_geometry_ok =
                    chunk_index && token_offset && logical_rows &&
                    physical_rows && *chunk_index < kExpectedChunks &&
                    *physical_rows == kSegmentedPrefillCaptureRows &&
                    ((*chunk_index == 0u && *token_offset == 0u &&
                      *logical_rows == 4u) ||
                     (*chunk_index == 1u && *token_offset == 4u &&
                      *logical_rows == 4u) ||
                     (*chunk_index == 2u && *token_offset == 8u &&
                      *logical_rows == 1u));
                const bool transaction_contract_ok =
                    record.value == 1.0 &&
                    record.count == 1u &&
                    tagEquals("role", "continuation_graph") &&
                    tagEquals("identity_source", "orchestration_request_and_chunk") &&
                    tagEquals("logical_step_semantics", "monotonic_transaction") &&
                    generation && *generation != 0u &&
                    transaction_step && *transaction_step != 0u &&
                    chunk_geometry_ok;
                if (!transaction_contract_ok)
                {
                    ++local[kMalformedRecords];
                    continue;
                }

                auto &schedule = continuation_schedules[*generation];
                ++schedule.transactions;
                schedule.chunk_mask |= 1u << *chunk_index;
                schedule.valid =
                    schedule.transaction_steps.insert(*transaction_step).second &&
                    schedule.valid;
                ++local[kContinuationTransactionRecords];
                continue;
            }

            if (record.domain == "moe_overlay_participant_graph" &&
                record.name == "ticket_selected_graphs" &&
                record.phase == "main_prefill")
            {
                const auto generation = unsignedTag("request_generation");
                const auto transaction_step = unsignedTag("logical_step");
                const auto transaction_ordinal =
                    unsignedTag("transaction_ordinal");
                const auto logical_rows = unsignedTag("logical_rows");
                const auto physical_rows = unsignedTag("physical_rows");
                const auto schedule_real_rows =
                    unsignedTag("prefill_schedule_real_rows");
                const auto schedule_execution_rows =
                    unsignedTag("prefill_schedule_execution_rows");
                const auto schedule_transactions =
                    unsignedTag("prefill_schedule_transaction_count");
                const auto schedule_fingerprint =
                    unsignedTag("prefill_schedule_fingerprint");
                /*
                 * This cell is Static, so it has no movement-interference
                 * sample to authenticate. The request generation plus the
                 * coordinator-owned ordinal is the complete segmented
                 * schedule identity. Dynamic cells deliberately carry the
                 * additional aggregate workload in these same ticket fields.
                 */
                const bool no_dynamic_interference_schedule =
                    schedule_real_rows && *schedule_real_rows == 0u &&
                    schedule_execution_rows &&
                    *schedule_execution_rows == 0u &&
                    schedule_transactions && *schedule_transactions == 0u &&
                    schedule_fingerprint && *schedule_fingerprint == 0u;
                const bool transaction_geometry_ok =
                    transaction_ordinal && *transaction_ordinal >= 1u &&
                    *transaction_ordinal <= kExpectedChunks && logical_rows &&
                    ((*transaction_ordinal == 1u && *logical_rows == 4u) ||
                     (*transaction_ordinal == 2u && *logical_rows == 4u) ||
                     (*transaction_ordinal == 3u && *logical_rows == 1u));
                const bool valid =
                    record.value == 1.0 && record.count == 1u &&
                    tagEquals("logical_step_semantics", "monotonic_transaction") &&
                    generation && *generation != 0u &&
                    transaction_step && *transaction_step != 0u &&
                    transaction_geometry_ok &&
                    physical_rows &&
                    *physical_rows == kSegmentedPrefillCaptureRows &&
                    no_dynamic_interference_schedule;
                if (!valid)
                {
                    std::ostringstream detail;
                    detail << "Malformed segmented-prefill follower ticket: "
                           << "phase=" << record.phase
                           << ",value=" << record.value
                           << ",count=" << record.count;
                    for (const auto &[name, value] : record.tags)
                        detail << ',' << name << '=' << value;
                    ADD_FAILURE() << detail.str();
                    ++local[kMalformedRecords];
                    continue;
                }

                auto &schedule = participant_schedules[*generation];
                ++schedule.transactions;
                schedule.logical_rows += *logical_rows;
                schedule.physical_rows += *physical_rows;
                schedule.four_row_transactions += *logical_rows == 4u ? 1u : 0u;
                schedule.one_row_transactions += *logical_rows == 1u ? 1u : 0u;
                schedule.transaction_ordinal_mask |=
                    1u << (*transaction_ordinal - 1u);
                schedule.valid =
                    schedule.transaction_steps.insert(*transaction_step).second &&
                    schedule.valid;
                ++local[kParticipantTransactionRecords];
            }
        }

        for (const auto &[generation, schedule] : continuation_schedules)
        {
            (void)generation;
            if (schedule.valid &&
                schedule.transactions == kExpectedChunks &&
                schedule.chunk_mask == kExpectedChunkMask &&
                schedule.transaction_steps.size() == kExpectedChunks)
            {
                ++local[kContinuationCompleteSchedules];
            }
            else
            {
                ++local[kMalformedRecords];
            }
        }
        for (const auto &[generation, schedule] : participant_schedules)
        {
            (void)generation;
            if (schedule.valid &&
                schedule.transactions == kExpectedChunks &&
                schedule.logical_rows == kExpectedRealTokens &&
                schedule.physical_rows ==
                    kExpectedChunks * kSegmentedPrefillCaptureRows &&
                schedule.four_row_transactions == 2u &&
                schedule.one_row_transactions == 1u &&
                schedule.transaction_ordinal_mask == kExpectedChunkMask &&
                schedule.transaction_steps.size() == kExpectedChunks)
            {
                ++local[kParticipantCompleteSchedules];
            }
            else
            {
                ++local[kMalformedRecords];
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
        EXPECT_EQ(global[kSnapshotAggregationRecords], 1u)
            << "The continuation graph must aggregate both prefills into one evidence record";
        EXPECT_EQ(
            global[kContinuationTransactionRecords],
            kExpectedObservedChunks)
            << "The continuation graph must stamp every [0,4,8] prefill transaction";
        EXPECT_EQ(
            global[kParticipantTransactionRecords],
            kExpectedObservedChunks)
            << "The remote expert graph must consume every authenticated prefill ticket";
        EXPECT_EQ(
            global[kContinuationCompleteSchedules],
            kExpectedPrefillTransactions)
            << "Each continuation request must publish the complete [0,4,8] live-row range";
        EXPECT_EQ(
            global[kParticipantCompleteSchedules],
            kExpectedPrefillTransactions)
            << "The remote graph must consume two complete authenticated [4,4,1] schedules";
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
        orchestration.activation_precision =
            isQwen122ProductionTest() ? "fp16" : "fp32";
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
        orchestration.deterministic = !isQwen122ProductionTest();
        orchestration.mtp.enabled = isQwen122ProductionTest();
        orchestration.mtp.draft_tokens = activeMTPDraftDepth();
        orchestration.mtp.graph_capacity_draft_tokens =
            isQwen122ProductionTest()
                ? kQwen122MaximumMTPDraftDepth
                : 0;
        orchestration.mtp.verify_mode = MTPVerifyMode::Greedy;
        if (usesDynamicMTPDepth())
        {
            orchestration.mtp.depth_policy.mode =
                MTPDepthPolicyMode::Dynamic;
            orchestration.mtp.depth_policy.min_depth = 1;
            orchestration.mtp.depth_policy.max_depth =
                kQwen122MaximumMTPDraftDepth;
            orchestration.mtp.depth_policy.initial_depth =
                kQwen122MaximumMTPDraftDepth;
            orchestration.mtp.depth_policy.window_size = 1;
            orchestration.mtp.depth_policy.min_samples = 1;
            orchestration.mtp.depth_policy.cooldown_steps = 0;
            orchestration.mtp.depth_policy.promote_consecutive_windows = 1;
        }
        orchestration.moe_routed_expert_plan = overlay_plan_;
        if (isLLEPProductionTest())
        {
            orchestration.moe_routed_prefill =
                RoutedExpertPrefillRuntimeConfig{
                    .assignment_window_tokens = 0,
                    .least_loaded_min_routed_rows = 0,
                    .llep_alpha_numerator = 9,
                    .llep_alpha_denominator = 10,
                    .llep_lambda_numerator = 13,
                    .llep_lambda_denominator = 10,
                    .llep_enable_balanced_skip = false,
                };
        }
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
            if (isQwen122ProductionTest())
            {
                /*
                 * A short authenticated campaign should expose real movement,
                 * not spend its wall time waiting for a serving-scale
                 * histogram. Economy certification remains authoritative and
                 * may still reject a non-profitable transaction.
                 */
                orchestration.moe_rebalance.window_size = 4;
                orchestration.moe_rebalance.max_window_size = 4;
                orchestration.moe_rebalance.dynamic_imbalance_threshold_per_mille = 0;
                orchestration.moe_rebalance.dynamic_min_improvement_per_mille = 0;
                orchestration.moe_rebalance.dynamic_max_swaps_per_layer = 4;
                orchestration.moe_rebalance.dynamic_max_plan_entries_per_wave = 32;
                orchestration.moe_rebalance.dynamic_min_window_activations = 0;
                orchestration.moe_rebalance.device_min_load_spread_improvement = 0;
                orchestration.moe_rebalance.device_min_load_spread_improvement_divisor = 0;
                orchestration.moe_rebalance.device_min_wave_spread_improvement_per_payload_slot = 0;
                orchestration.moe_rebalance.device_min_foreign_rows_per_critical_path_payload_slot = 0;
                orchestration.moe_rebalance.device_min_router_spread_improvement_per_payload_slot = 0;
                orchestration.moe_rebalance.device_max_post_wave_load_spread_per_mille = 1000;
                orchestration.moe_rebalance.device_maintenance_slack_tokens = 0;
                orchestration.moe_rebalance.device_min_maintenance_period_tokens = 1;
                orchestration.moe_rebalance.device_initial_maintenance_period_tokens = 1;
            }
        }

        const auto snapshot_setup_mode =
            isDynamicResidencyProductionTest()
                ? ParitySnapshotSetupMode::Disabled
                : ParitySnapshotSetupMode::Enabled;
        bool model_context_cache_hit = false;
        std::string model_context_cache_error;
        auto reuse_contract = findQwen122OverlayModelContext(
            &model_context_cache_hit,
            &model_context_cache_error);
        if (!model_context_cache_error.empty())
        {
            LOG_ERROR("[Qwen3.5 MoE GraphNative] "
                      << model_context_cache_error);
            return false;
        }

        const bool setup_ok = reuse_contract
                                  ? setupOrchestrationRunner(
                                        orchestration,
                                        *reuse_contract,
                                        snapshot_setup_mode)
                                  : setupOrchestrationRunner(
                                        orchestration,
                                        nullptr,
                                        snapshot_setup_mode);
        if (!setup_ok)
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

        /*
         * Use the same readiness surface as the HTTP server and benchmark.
         * Dynamic ExpertOverlay may need a bounded set of physical topology
         * measurements before ordinary requests are admissible; that work is
         * owned entirely below this interface and never leaks calibration
         * planning into the parity driver.
         */
        if (!orch_runner_->prepareForInference())
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Production inference preparation failed: "
                << orch_runner_->lastError());
            return false;
        }

        if (mayReuseQwen122OverlayModelContext())
        {
            if (model_context_cache_hit)
            {
                double prepared_store_reuses = 0.0;
                for (const auto &record :
                     PerfStatsCollector::snapshot({"weight_loading"}))
                {
                    if (record.kind == PerfStatRecord::Kind::Counter &&
                        record.domain == "weight_loading" &&
                        record.name ==
                            "preloaded_prepared_weight_store_reuses")
                    {
                        prepared_store_reuses += record.value;
                    }
                }
                if (prepared_store_reuses <= 0.0)
                {
                    LOG_ERROR(
                        "[Qwen3.5 MoE GraphNative] Campaign cache hit did not "
                        "reuse a production-certified PreparedWeightStore");
                    return false;
                }
            }
            else
            {
                const auto published_contract =
                    orch_runner_->modelContextReuseContract();
                if (!published_contract)
                {
                    LOG_ERROR(
                        "[Qwen3.5 MoE GraphNative] Initialized runner did not "
                        "publish its exact rank-local ExpertOverlay weight contract");
                    return false;
                }
                if (!publishQwen122OverlayModelContext(
                        *published_contract,
                        &model_context_cache_error))
                {
                    LOG_ERROR("[Qwen3.5 MoE GraphNative] "
                              << model_context_cache_error);
                    return false;
                }
            }

            PerfStatsCollector::addCounter(
                "weight_loading",
                model_context_cache_hit
                    ? "parity_campaign_model_context_cache_hits"
                    : "parity_campaign_model_context_cache_misses",
                1.0,
                "setup",
                orch_runner_->primaryDeviceId().toString(),
                {{"owner_order",
                  isRandomOwnerProductionTest() ? "random" : "ordinal"},
                 {"mtp_depth", std::to_string(activeMTPDraftDepth())}});
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

    /** @brief Sum one process-local device-controller counter. */
    static double controllerCounter(
        const std::vector<PerfStatRecord> &records,
        const std::string &name)
    {
        double total = 0.0;
        for (const auto &record : records)
        {
            if (record.kind == PerfStatRecord::Kind::Counter &&
                record.domain == "moe_overlay_controller" &&
                record.name == name)
            {
                total += record.value;
            }
        }
        return total;
    }

    /** @brief Count committed movement edges with one typed direction. */
    static double movementDirectionCount(
        const std::vector<PerfStatRecord> &records,
        const std::string &domain,
        const std::string &name,
        const std::string &direction)
    {
        double total = 0.0;
        for (const auto &record : records)
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != domain || record.name != name)
            {
                continue;
            }
            const auto found = record.tags.find("direction");
            if (found != record.tags.end() && found->second == direction)
                total += record.value;
        }
        return total;
    }

    /** @return Exact local committed-wave count, or throw on malformed data. */
    std::uint64_t localCommittedWaveCount() const
    {
        const double value = topologyUsesCpu()
                                 ? localResidencyCounter("committed_waves")
                                 : static_cast<double>(
                                       orch_runner_
                                           ? orch_runner_
                                                 ->moeRuntimeMovementEpoch()
                                           : 0u);
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
            if (!orch_runner_->maybeApplyMoERebalance(
                    boundary_sample.tokens.size()))
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
                if (!orch_runner_->maybeApplyMoERebalance(
                        decoded.tokens.size()))
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
        case OverlayTopology::Cuda2Rocm4:
            return "CUDA2+ROCm4";
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

    /**
     * @brief Feed bounded real requests until distributed residency actually moves.
     *
     * Physical topology preparation is completed through the serving readiness
     * API before this method begins. This driver then supplies only ordinary
     * prefill/decode/MTP requests from a deterministic real-token corpus. The
     * background worker owns interval selection, timing, staging, transfer,
     * overlap validation, certification, and publication; no calibration arm,
     * histogram, timing, placement, or completion value is injected here.
     *
     * The parity-artifact rank is the sole traffic-control authority. Remote
     * ranks are already inside `MPIWorkerLoop` and execute authenticated
     * transaction-follower commands; they are not peer test drivers. Calling a
     * test-owned MPI collective here would create a second command protocol
     * and collide with the follower's next typed command receive.
     *
     * @return True on the coordinated root after it observes certification,
     *         the required improving epochs, a capacity-preserving
     *         promotion/demotion pair, and within-tier skew movement (plus a
     *         cross-rank edge where the resolved topology requires one); false
     *         after bounded traffic is exhausted.
     */
    bool driveDynamicResidencyToDistributedMigration()
    {
        if (!isDynamicResidencyProductionTest())
            return true;
        if (!isRootParityRank())
        {
            throw std::logic_error(
                "Only the coordinated parity root may drive Dynamic residency traffic");
        }

        const int maximum_certified_requests = 24;
        const int decode_steps_per_certified_request =
            isQwen122ProductionTest() ? 8 : 9;
        const double minimum_committed_waves =
            isQwen122ProductionTest() ? 1.0 : 2.0;
        constexpr double kMaximumRejectedProposals = 8.0;
        const int maximum_service_profile_requests =
            maximum_certified_requests * 2;

        const std::vector<int32_t> prompt(
            config_.token_ids.begin(), config_.token_ids.end());
        const int vocabulary_size = orch_runner_->vocabSize();
        if (prompt.empty() || vocabulary_size <= 4'096)
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Economy service coverage requires a non-empty authenticated prompt and a valid vocabulary");
            return false;
        }
        const auto serviceCoveragePrompt =
            [&prompt, vocabulary_size](int request_index)
        {
            if (request_index == 0)
                return prompt;

            /*
             * Service certification prices every possible owner, whereas one
             * deterministic request follows one fixed sparse route set. Use a
             * bounded deterministic corpus of valid embedding rows to make
             * ordinary production routing visit otherwise-idle owners. No
             * route, timing, histogram, placement, or completion value is
             * injected; the real model graph remains the sole source of every
             * service observation. The host-authority lifecycle rebases its
             * demand window when certification becomes actionable, so these
             * topology-coverage requests cannot train the first movement.
             */
            std::uint64_t state =
                0x9e3779b97f4a7c15ULL ^
                (static_cast<std::uint64_t>(request_index) *
                 0xbf58476d1ce4e5b9ULL);
            const auto usable_vocabulary =
                static_cast<std::uint64_t>(vocabulary_size - 2'048);
            std::vector<int32_t> varied(prompt.size(), 0);
            for (auto &token : varied)
            {
                state += 0x9e3779b97f4a7c15ULL;
                std::uint64_t mixed = state;
                mixed = (mixed ^ (mixed >> 30u)) *
                        0xbf58476d1ce4e5b9ULL;
                mixed = (mixed ^ (mixed >> 27u)) *
                        0x94d049bb133111ebULL;
                mixed ^= mixed >> 31u;
                token = static_cast<int32_t>(
                    256u + mixed % usable_vocabulary);
            }
            return varied;
        };

        std::uint64_t service_profile_forwards = 0;
        int certified_requests = 0;
        const DynamicMovementAxisContract movement_axis_contract =
            dynamicMovementAxisContract(*overlay_plan_);
        const std::uint64_t initial_runtime_movement_epoch =
            orch_runner_->moeRuntimeMovementEpoch();

        const auto wakeMaintenance = [&](const char *phase,
                                         uint64_t committed_tokens)
        {
            if (orch_runner_->maybeApplyMoERebalance(committed_tokens))
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
            return true;
        };
        const auto runDecode = [&](int response_token_budget,
                                   std::uint64_t *elapsed_ns = nullptr)
        {
            /*
             * `decodeStep()` intentionally treats a zero token budget as an
             * unbounded serving response. The first generated token consumes
             * prefill logits and does not execute a DecodeToken graph, so a
             * serial-decode coverage request must admit at least two tokens.
             * Four-token requests additionally exercise multiple grouped-MTP
             * transactions without consuming the 4K KV capacity. This gives
             * every active production phase natural routing opportunities
             * under an adversarial initial expert layout.
             */
            if (response_token_budget <= 0)
                return std::optional<bool>{};
            orch_runner_->setDecodeStepTokenBudget(response_token_budget);
            const auto start = std::chrono::steady_clock::now();
            const GenerationResult generated = orch_runner_->decodeStep();
            orch_runner_->setDecodeStepTokenBudget(0);
            if (!generated.success() || generated.tokens.empty())
            {
                LOG_ERROR(
                    "[Qwen3.5 MoE GraphNative] Dynamic calibration decode failed: "
                    << generated.error);
                return std::optional<bool>{};
            }
            if (elapsed_ns)
                *elapsed_ns = elapsedNanoseconds(start);
            if (!wakeMaintenance("decode", generated.tokens.size()))
                return std::optional<bool>{};
            return std::optional<bool>{generated.is_complete};
        };

        /*
         * Physical topology preparation completed through
         * prepareForInference(). The remaining economy profile is learned from
         * ordinary traffic. Service cost is a topology-wide certificate, so a
         * stationary prompt is insufficient: sparse routing can leave a valid
         * owner forever idle. The bounded valid-token corpus above supplies
         * natural production coverage only. Certification publication rebases
         * demand before proposal admission; every later histogram epoch and
         * the Hugging Face comparison therefore describe the stationary
         * authenticated parity request. Production alone decides which
         * intervals become baseline or concurrent samples, and the test never
         * reconstructs private calibration arms from PerfStats.
         */
        for (int request_index = 0;
             request_index < maximum_service_profile_requests &&
             localResidencyCounter("economy_certification_complete") == 0.0;
             ++request_index)
        {
            std::uint64_t prefill_ns = 0;
            if (!runPrefill(
                    serviceCoveragePrompt(request_index),
                    requiresObservedConvergenceSpeedup() ? &prefill_ns
                                                         : nullptr))
            {
                return false;
            }
            ++service_profile_forwards;
            if (requiresObservedConvergenceSpeedup() && prefill_ns > 0 &&
                convergence_timings_.baseline_prefill_ns.size() < 2u)
            {
                convergence_timings_.baseline_prefill_ns.push_back(prefill_ns);
            }

            for (int step = 0;
                 step < decode_steps_per_certified_request;
                 ++step)
            {
                std::uint64_t decode_ns = 0;
                const int response_token_budget =
                    isQwen122ProductionTest() &&
                            step < decode_steps_per_certified_request / 2
                        ? 1
                        : (isQwen122ProductionTest() ? 4 : 2);
                const auto complete = runDecode(
                    response_token_budget,
                    requiresObservedConvergenceSpeedup() ? &decode_ns
                                                         : nullptr);
                if (!complete)
                    return false;
                ++service_profile_forwards;
                if (requiresObservedConvergenceSpeedup() && decode_ns > 0 &&
                    convergence_timings_.baseline_decode_ns.size() < 2u)
                {
                    convergence_timings_.baseline_decode_ns.push_back(
                        decode_ns);
                }
                if (*complete)
                    break;
            }
        }

        if (localResidencyCounter("economy_certification_complete") == 0.0)
        {
            struct ServiceTrafficTotals
            {
                double active_routes = 0.0;
                std::uint64_t completed_packets = 0u;
            };
            std::map<std::pair<std::string, std::string>,
                     ServiceTrafficTotals>
                service_traffic_by_participant_phase;
            for (const auto &record :
                 PerfStatsCollector::snapshot({"forward_graph"}))
            {
                if (record.kind != PerfStatRecord::Kind::Counter ||
                    record.domain != "forward_graph")
                {
                    continue;
                }
                const auto participant =
                    record.tags.find("participant");
                const auto source =
                    record.tags.find("service_source");
                if (participant == record.tags.end() ||
                    source == record.tags.end())
                {
                    continue;
                }
                auto &totals = service_traffic_by_participant_phase[
                    {participant->second, source->second}];
                if (record.name ==
                    "moe_overlay_local_expert_active_routes")
                {
                    totals.active_routes += record.value;
                }
                else if (record.name ==
                         "moe_overlay_local_expert_completions")
                {
                    totals.completed_packets += record.count;
                }
            }
            std::ostringstream service_traffic;
            service_traffic
                << "rank=" << (mpi_ctx_ ? mpi_ctx_->rank() : 0);
            for (const auto &[coordinate, totals] :
                 service_traffic_by_participant_phase)
            {
                service_traffic
                    << " p" << coordinate.first << '/'
                    << coordinate.second << "{routes="
                    << static_cast<std::uint64_t>(totals.active_routes)
                    << ",packets=" << totals.completed_packets << '}';
            }
            if (isRootParityRank())
            {
                LOG_ERROR(
                    "[Qwen3.5 MoE GraphNative] Dynamic economy certification did "
                    "not complete after " << service_profile_forwards
                    << " ordinary production forwards; service_traffic="
                    << service_traffic.str() << "\n"
                    << PerfStatsCollector::summaryString(
                           {"moe_overlay_residency"}));
            }
            return false;
        }

        double last_committed_migrations =
            localResidencyCounter("committed_expert_migrations");
        double proposals_at_last_migration =
            localResidencyCounter("economy_proposals");

        for (; certified_requests < maximum_certified_requests;
             ++certified_requests)
        {
            /*
             * Broad traffic exists only to certify every measured service
             * coordinate. Production rebases those calibration histograms at
             * the EconomyCertified transition; every placement epoch after
             * that boundary must optimize the exact authenticated workload
             * whose Hugging Face checkpoints certify the published layout.
             * A second mixed movement corpus would create two workload
             * authorities and make post-publication use a sampling accident.
             */
            if (!runPrefill(prompt))
                return false;
            for (int step = 0;
                 step < decode_steps_per_certified_request;
                 ++step)
            {
                const auto complete = runDecode(
                    isQwen122ProductionTest() ? 4 : 2);
                if (!complete)
                    return false;
                if (*complete)
                    break;
            }

            const auto movement_records = PerfStatsCollector::snapshot(
                {"moe_overlay_residency", "moe_overlay_controller"});
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
            const double promotions = topologyUsesCpu()
                                          ? movementDirectionCount(
                                                movement_records,
                                                "moe_overlay_residency",
                                                "expert_migration_edges",
                                                "promotion")
                                          : controllerCounter(
                                                movement_records,
                                                "dynamic_promotions");
            const double demotions = topologyUsesCpu()
                                         ? movementDirectionCount(
                                               movement_records,
                                               "moe_overlay_residency",
                                               "expert_migration_edges",
                                               "demotion")
                                         : controllerCounter(
                                               movement_records,
                                               "dynamic_demotions");
            const double same_priority_moves = movementDirectionCount(
                movement_records,
                topologyUsesCpu() ? "moe_overlay_residency"
                                  : "moe_overlay_controller",
                topologyUsesCpu() ? "expert_migration_edges"
                                  : "dynamic_migration_edges",
                "same_priority");
            const std::uint64_t runtime_movement_epoch =
                orch_runner_->moeRuntimeMovementEpoch();
            const std::uint64_t completed_device_epochs =
                runtime_movement_epoch >= initial_runtime_movement_epoch
                    ? runtime_movement_epoch -
                          initial_runtime_movement_epoch
                    : 0u;

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
            const bool cross_rank_requirement_met =
                isQwen122ProductionTest() || cross_rank_migrations > 0.0;
            const bool same_priority_requirement_met =
                movement_axis_contract ==
                    DynamicMovementAxisContract::PriorityMigrationOnly
                    ? same_priority_moves == 0.0
                    : same_priority_moves > 0.0;
            const bool device_authority_moved =
                !topologyUsesCpu() &&
                completed_device_epochs >= static_cast<std::uint64_t>(
                    minimum_committed_waves) &&
                promotions > 0.0 && demotions > 0.0 &&
                same_priority_requirement_met;
            const bool host_authority_moved =
                topologyUsesCpu() &&
                committed_waves >= minimum_committed_waves &&
                committed_migrations > 0.0 &&
                cross_rank_requirement_met && promotions > 0.0 &&
                demotions > 0.0 && same_priority_requirement_met;
            if (device_authority_moved || host_authority_moved)
            {
                return true;
            }
            const bool rejected_proposal_budget_exhausted =
                topologyUsesCpu() &&
                proposals - proposals_at_last_migration >=
                    kMaximumRejectedProposals;
            if (rejected_proposal_budget_exhausted)
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

        if (isRootParityRank())
        {
            /*
             * Preserve every tagged planner/admission decision before the
             * coordinated shutdown starts. A teardown failure must not erase
             * the numerical and economy evidence for the original gate.
             */
            writeResidencyFailureDiagnosticsCsv();
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Dynamic residency did not produce the required profitable publication epoch(s) and topology-valid movement after "
                << service_profile_forwards << " service-profile forwards and "
                << certified_requests << " certified histogram requests\n"
                << PerfStatsCollector::summaryString(
                       {"moe_overlay_residency", "moe_overlay_controller"}));
        }
        return false;
    }

    /**
     * @brief Prove all-GPU placement with the sole device-resident authority.
     *
     * The heterogeneous host authority publishes `committed_waves` and its
     * migration ledger under `moe_overlay_residency`.  An all-GPU topology has
     * no such second authority: the authenticated device-controller command is
     * the placement decision and its completed physical transaction is the
     * evidence. This fold checks numeric-priority promotion/demotion in every
     * Dynamic topology and additionally requires same-priority skew movement
     * whenever one declared priority owns two or more physical participants.
     */
    void assertDeviceResidentMovementEvidence() const
    {
        enum Evidence : size_t
        {
            DomainEnabled,
            StaticChecks,
            EconomyReady,
            CertificationComplete,
            RuntimeEpochParticipants,
            Transactions,
            Commands,
            PhysicalBytes,
            Promotions,
            Demotions,
            SamePriorityMoves,
            CrossDomainMoves,
            CrossRankMoves,
            CrossBackendMoves,
            MigrationEdges,
            BackgroundNotifications,
            PhysicalOperations,
            ConfiguredCycleCapRecords,
            AdoptedInitialSlots,
            BootstrapSlotsRecycled,
            UniqueImprovingEpochs,
            CapacityConservationCertifications,
            TaggedTransactions,
            TaggedCommands,
            TaggedPhysicalBytes,
            TaggedPromotions,
            TaggedDemotions,
            TaggedSamePriorityMoves,
            TaggedCrossDomainMoves,
            TaggedCrossRankMoves,
            TaggedCrossBackendMoves,
            EvidenceViolations,
            EvidenceCount,
        };

        std::array<uint64_t, EvidenceCount> local{};
        local[DomainEnabled] =
            PerfStatsCollector::isDomainEnabled("moe_overlay_controller")
                ? 1u
                : 0u;
        local[RuntimeEpochParticipants] =
            orch_runner_ && orch_runner_->moeRuntimeMovementEpoch() > 0u
                ? 1u
                : 0u;
        std::set<uint64_t> improving_epochs;
        const auto add = [&local](Evidence evidence, double value)
        {
            if (value > 0.0 && std::isfinite(value) &&
                value <= static_cast<double>(
                    std::numeric_limits<uint64_t>::max()))
            {
                local[evidence] += static_cast<uint64_t>(value);
            }
        };
        const auto parse_u64 = [](const std::string &text,
                                  uint64_t &value)
        {
            const char *const begin = text.data();
            const char *const end = begin + text.size();
            const auto parsed = std::from_chars(begin, end, value);
            return parsed.ec == std::errc{} && parsed.ptr == end;
        };
        const auto parse_i64 = [](const std::string &text,
                                  std::int64_t &value)
        {
            const char *const begin = text.data();
            const char *const end = begin + text.size();
            const auto parsed = std::from_chars(begin, end, value);
            return parsed.ec == std::errc{} && parsed.ptr == end;
        };
        const auto tag_u64 = [&](const PerfStatRecord &record,
                                 const char *name,
                                 uint64_t &value)
        {
            const auto found = record.tags.find(name);
            return found != record.tags.end() &&
                   parse_u64(found->second, value);
        };

        for (const auto &record : PerfStatsCollector::snapshot(
                 {"moe_overlay_controller", "moe_overlay_residency"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter)
                continue;

            if (record.domain == "moe_overlay_residency")
            {
                if (record.name == "economy_certification_complete")
                    add(CertificationComplete, record.value);
                else if (record.name == "physical_fabrics_materialized")
                {
                    add(ConfiguredCycleCapRecords, record.value);
                    uint64_t cycle_cap = 0u;
                    uint64_t adopted_slots = 0u;
                    if (!tag_u64(
                            record,
                            "maximum_concurrent_cycles",
                            cycle_cap) ||
                        cycle_cap != 2u ||
                        !tag_u64(
                            record,
                            "adopted_initial_slots",
                            adopted_slots))
                    {
                        ++local[EvidenceViolations];
                    }
                    else
                    {
                        local[AdoptedInitialSlots] += adopted_slots;
                    }
                }
                else if (record.name == "bootstrap_live_slots_recycled")
                {
                    add(BootstrapSlotsRecycled, record.value);
                }
                continue;
            }
            if (record.domain != "moe_overlay_controller")
                continue;

            if (record.name == "static_no_movement_transactions")
            {
                uint64_t commands = 1u;
                uint64_t bytes = 1u;
                const auto waits = record.tags.find("inference_stream_waits");
                const bool valid =
                    record.phase == "model_setup" &&
                    record.value == 1.0 && record.count == 1u &&
                    tag_u64(record, "movement_commands", commands) &&
                    commands == 0u &&
                    tag_u64(record, "packed_weight_bytes", bytes) &&
                    bytes == 0u && waits != record.tags.end() &&
                    waits->second == "0";
                if (valid)
                    add(StaticChecks, record.value);
                else
                    ++local[EvidenceViolations];
            }
            else if (record.name == "device_economy_ready")
            {
                add(EconomyReady, record.value);
            }
            else if (record.name == "dynamic_movement_transactions")
            {
                uint64_t transaction = 0u;
                uint64_t base_epoch = 0u;
                uint64_t candidate_epoch = 0u;
                uint64_t commands = 0u;
                uint64_t bytes = 0u;
                uint64_t promotions = 0u;
                uint64_t demotions = 0u;
                uint64_t same_priority = 0u;
                uint64_t cross_domain = 0u;
                uint64_t cross_rank = 0u;
                uint64_t cross_backend = 0u;
                uint64_t accepted_cycles = 0u;
                uint64_t service_gain = 0u;
                uint64_t net_benefit = 0u;
                const auto policy_owner = record.tags.find("policy_owner");
                const auto blocking = record.tags.find("blocking_inference");
                const bool valid =
                    record.phase == "maintenance" && record.value > 0.0 &&
                    record.count > 0u &&
                    tag_u64(record, "transaction", transaction) &&
                    transaction > 0u &&
                    tag_u64(record, "base_epoch", base_epoch) &&
                    tag_u64(record, "candidate_epoch", candidate_epoch) &&
                    candidate_epoch == base_epoch + 1u &&
                    tag_u64(record, "movement_commands", commands) &&
                    commands > 0u &&
                    tag_u64(record, "physical_bytes", bytes) && bytes > 0u &&
                    tag_u64(record, "promotions", promotions) &&
                    tag_u64(record, "demotions", demotions) &&
                    tag_u64(record, "same_priority_moves", same_priority) &&
                    tag_u64(record, "cross_domain_moves", cross_domain) &&
                    tag_u64(record, "cross_rank_moves", cross_rank) &&
                    tag_u64(record, "cross_backend_moves", cross_backend) &&
                    tag_u64(record, "accepted_cycles", accepted_cycles) &&
                    accepted_cycles > 0u && accepted_cycles <= 2u &&
                    tag_u64(
                        record,
                        "projected_service_gain_ns",
                        service_gain) &&
                    service_gain > 0u &&
                    tag_u64(
                        record,
                        "projected_net_benefit_ns",
                        net_benefit) &&
                    net_benefit > 0u &&
                    policy_owner != record.tags.end() &&
                    policy_owner->second == "device" &&
                    blocking != record.tags.end() &&
                    blocking->second == "false";
                if (!valid)
                {
                    ++local[EvidenceViolations];
                    continue;
                }
                add(Transactions, record.value);
                add(TaggedTransactions, record.value);
                local[TaggedCommands] += commands;
                local[TaggedPhysicalBytes] += bytes;
                local[TaggedPromotions] += promotions;
                local[TaggedDemotions] += demotions;
                local[TaggedSamePriorityMoves] += same_priority;
                local[TaggedCrossDomainMoves] += cross_domain;
                local[TaggedCrossRankMoves] += cross_rank;
                local[TaggedCrossBackendMoves] += cross_backend;
                improving_epochs.insert(candidate_epoch);
            }
            else if (record.name == "dynamic_movement_commands")
                add(Commands, record.value);
            else if (record.name == "dynamic_physical_bytes")
                add(PhysicalBytes, record.value);
            else if (record.name == "dynamic_promotions")
                add(Promotions, record.value);
            else if (record.name == "dynamic_demotions")
                add(Demotions, record.value);
            else if (record.name == "dynamic_same_priority_moves")
                add(SamePriorityMoves, record.value);
            else if (record.name == "dynamic_cross_domain_moves")
                add(CrossDomainMoves, record.value);
            else if (record.name == "dynamic_cross_rank_moves")
                add(CrossRankMoves, record.value);
            else if (record.name == "dynamic_cross_backend_moves")
                add(CrossBackendMoves, record.value);
            else if (record.name ==
                     "dynamic_capacity_conservation_certifications")
            {
                uint64_t edges = 0u;
                uint64_t participant_coordinates = 0u;
                uint64_t tier_coordinates = 0u;
                uint64_t malformed = 0u;
                uint64_t participant_violations = 0u;
                uint64_t tier_violations = 0u;
                const auto direction_proxy = record.tags.find(
                    "direction_counts_are_capacity_proof");
                const bool valid =
                    record.value == 1.0 && record.count == 1u &&
                    tag_u64(record, "edges_checked", edges) && edges > 0u &&
                    tag_u64(
                        record,
                        "participant_coordinates_checked",
                        participant_coordinates) &&
                    participant_coordinates > 0u &&
                    tag_u64(
                        record,
                        "tier_coordinates_checked",
                        tier_coordinates) &&
                    tier_coordinates > 0u &&
                    tag_u64(record, "malformed_edges", malformed) &&
                    malformed == 0u &&
                    tag_u64(
                        record,
                        "participant_flow_violations",
                        participant_violations) &&
                    participant_violations == 0u &&
                    tag_u64(
                        record,
                        "tier_flow_violations",
                        tier_violations) &&
                    tier_violations == 0u &&
                    direction_proxy != record.tags.end() &&
                    direction_proxy->second == "false";
                if (valid)
                    add(
                        CapacityConservationCertifications,
                        record.value);
                else
                    ++local[EvidenceViolations];
            }
            else if (record.name == "background_notification_batches")
                add(BackgroundNotifications, record.value);
            else if (record.name ==
                     "physical_wave_parallel_operations_started")
                add(PhysicalOperations, record.value);
            else if (record.name == "dynamic_migration_edges")
            {
                std::int64_t source_priority = 0;
                std::int64_t destination_priority = 0;
                uint64_t estimated_bytes = 0u;
                const auto direction = record.tags.find("direction");
                const auto source = record.tags.find("source_priority");
                const auto destination =
                    record.tags.find("destination_priority");
                const auto blocking = record.tags.find("blocking_inference");
                const bool parsed =
                    direction != record.tags.end() &&
                    source != record.tags.end() &&
                    destination != record.tags.end() &&
                    parse_i64(source->second, source_priority) &&
                    parse_i64(destination->second, destination_priority) &&
                    tag_u64(
                        record,
                        "estimated_weight_bytes",
                        estimated_bytes) &&
                    estimated_bytes > 0u;
                const bool direction_valid = parsed &&
                    ((direction->second == "promotion" &&
                      destination_priority < source_priority) ||
                     (direction->second == "demotion" &&
                      destination_priority > source_priority) ||
                     (direction->second == "same_priority" &&
                      destination_priority == source_priority));
                if (!direction_valid || blocking == record.tags.end() ||
                    blocking->second != "false")
                {
                    ++local[EvidenceViolations];
                }
                else
                {
                    add(MigrationEdges, record.value);
                }
            }
        }
        local[UniqueImprovingEpochs] = improving_epochs.size();

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

        const uint64_t ranks = static_cast<uint64_t>(mpiWorldSize());
        ASSERT_EQ(global[DomainEnabled], ranks)
            << "Every all-GPU participant rank must retain device-controller evidence";
        EXPECT_EQ(global[EvidenceViolations], 0u);
        if (!isDynamicResidencyProductionTest())
        {
            EXPECT_EQ(global[StaticChecks], ranks)
                << "Every Static device follower must certify zero movement";
            EXPECT_EQ(global[Transactions], 0u);
            EXPECT_EQ(global[Commands], 0u);
            EXPECT_EQ(global[PhysicalBytes], 0u);
            EXPECT_EQ(global[Promotions], 0u);
            EXPECT_EQ(global[Demotions], 0u);
            EXPECT_EQ(global[SamePriorityMoves], 0u);
            EXPECT_EQ(global[MigrationEdges], 0u);
            return;
        }

        EXPECT_GE(global[EconomyReady], ranks)
            << "Every device follower must acquire the certified economy profile";
        EXPECT_GE(global[CertificationComplete], ranks);
        EXPECT_EQ(global[RuntimeEpochParticipants], ranks)
            << "Every rank must observe the completed durable movement epoch";
        EXPECT_GE(global[ConfiguredCycleCapRecords], ranks);
        EXPECT_GT(global[AdoptedInitialSlots], 0u);
        EXPECT_GT(global[BootstrapSlotsRecycled], 0u);
        EXPECT_GE(global[Transactions], ranks);
        EXPECT_GE(global[UniqueImprovingEpochs], ranks);
        EXPECT_GT(global[Commands], 0u);
        EXPECT_GT(global[PhysicalBytes], 0u);
        EXPECT_GT(global[BackgroundNotifications], 0u);
        EXPECT_GT(global[PhysicalOperations], 0u);
        EXPECT_GT(global[Promotions], 0u)
            << "Dynamic device policy never promoted a histogram-hot expert";
        EXPECT_GT(global[Demotions], 0u)
            << "Dynamic device policy never demoted an expert to release capacity";
        EXPECT_EQ(
            global[CapacityConservationCertifications],
            global[Transactions])
            << "Every device-owned transaction must explicitly conserve participant and tier slots";
        if (dynamicMovementAxisContract(*overlay_plan_) ==
            DynamicMovementAxisContract::
                PriorityMigrationAndParticipantBalance)
        {
            EXPECT_GT(global[SamePriorityMoves], 0u)
                << "Dynamic device policy never reduced within-tier participant skew";
        }
        else
        {
            EXPECT_EQ(global[SamePriorityMoves], 0u)
                << "A singleton-priority topology reported an impossible same-priority move";
        }
        EXPECT_EQ(
            global[CrossDomainMoves],
            global[Promotions] + global[Demotions]);
        EXPECT_EQ(
            global[CrossBackendMoves],
            global[Promotions] + global[Demotions]);
        EXPECT_LE(global[CrossRankMoves], global[Commands]);
        EXPECT_EQ(global[MigrationEdges], global[Commands]);

        EXPECT_EQ(global[TaggedTransactions], global[Transactions]);
        EXPECT_EQ(global[TaggedCommands], global[Commands]);
        EXPECT_EQ(global[TaggedPhysicalBytes], global[PhysicalBytes]);
        EXPECT_EQ(global[TaggedPromotions], global[Promotions]);
        EXPECT_EQ(global[TaggedDemotions], global[Demotions]);
        EXPECT_EQ(
            global[TaggedSamePriorityMoves],
            global[SamePriorityMoves]);
        EXPECT_EQ(global[TaggedCrossDomainMoves], global[CrossDomainMoves]);
        EXPECT_EQ(global[TaggedCrossRankMoves], global[CrossRankMoves]);
        EXPECT_EQ(global[TaggedCrossBackendMoves], global[CrossBackendMoves]);
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
        if (!topologyUsesCpu())
        {
            assertDeviceResidentMovementEvidence();
            return;
        }

        enum Evidence : size_t
        {
            DomainEnabled,
            StaticChecks,
            TransportProfileComplete,
            CertificationComplete,
            AuthorityCertifications,
            MaintenanceCertifications,
            CommittedWaves,
            CommittedMigrations,
            Promotions,
            Demotions,
            SamePriority,
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
            CapacityConservationCertifications,
            CapacityConservationViolations,
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
        const auto tag_u64 = [&parse_u64](
                                 const PerfStatRecord &record,
                                 const char *name,
                                 uint64_t &value)
        {
            const auto found = record.tags.find(name);
            return found != record.tags.end() &&
                   parse_u64(found->second, value);
        };
        for (const auto &record :
             PerfStatsCollector::snapshot(
                 {"moe_overlay_residency", "moe_overlay_controller"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter)
            {
                continue;
            }

            if (record.domain == "moe_overlay_controller" &&
                record.name == "static_no_movement_transactions")
            {
                const auto movement_commands =
                    record.tags.find("movement_commands");
                const auto packed_weight_bytes =
                    record.tags.find("packed_weight_bytes");
                const auto inference_stream_waits =
                    record.tags.find("inference_stream_waits");
                const bool valid =
                    record.phase == "model_setup" &&
                    record.value == 1.0 && record.count == 1u &&
                    movement_commands != record.tags.end() &&
                    movement_commands->second == "0" &&
                    packed_weight_bytes != record.tags.end() &&
                    packed_weight_bytes->second == "0" &&
                    inference_stream_waits != record.tags.end() &&
                    inference_stream_waits->second == "0";
                if (valid)
                    addValue(StaticChecks, record.value);
                else
                    ++local[PhysicalEvidenceViolations];
                continue;
            }
            if (record.domain != "moe_overlay_residency")
                continue;

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
            else if (record.name == "economy_transport_profile_complete")
            {
                const auto synthetic =
                    record.tags.find("synthetic_inference");
                const auto publishes_residency =
                    record.tags.find("publish_residency");
                const auto waves_tag = record.tags.find("waves");
                const auto elapsed_tag =
                    record.tags.find("elapsed_nanoseconds");
                uint64_t waves = 0u;
                uint64_t elapsed = 0u;
                const bool valid =
                    record.phase == "model_setup" &&
                    record.value == 1.0 && record.count == 1u &&
                    synthetic != record.tags.end() &&
                    synthetic->second == "false" &&
                    publishes_residency != record.tags.end() &&
                    publishes_residency->second == "false" &&
                    waves_tag != record.tags.end() &&
                    parse_u64(waves_tag->second, waves) && waves > 0u &&
                    elapsed_tag != record.tags.end() &&
                    parse_u64(elapsed_tag->second, elapsed) && elapsed > 0u;
                if (valid)
                    addValue(TransportProfileComplete, record.value);
                else
                    ++local[PhysicalEvidenceViolations];
            }
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
            else if (record.name ==
                     "capacity_conservation_certifications")
            {
                uint64_t edges = 0u;
                uint64_t cycles = 0u;
                uint64_t participant_coordinates = 0u;
                uint64_t tier_coordinates = 0u;
                uint64_t malformed = 0u;
                uint64_t participant_violations = 0u;
                uint64_t tier_violations = 0u;
                const auto direction_proxy = record.tags.find(
                    "direction_counts_are_capacity_proof");
                const bool valid =
                    record.value == 1.0 && record.count == 1u &&
                    tag_u64(record, "edges_checked", edges) && edges > 0u &&
                    tag_u64(record, "closed_cycles", cycles) && cycles > 0u &&
                    tag_u64(
                        record,
                        "participant_coordinates_checked",
                        participant_coordinates) &&
                    participant_coordinates > 0u &&
                    tag_u64(
                        record,
                        "tier_coordinates_checked",
                        tier_coordinates) &&
                    tier_coordinates > 0u &&
                    tag_u64(record, "malformed_edges", malformed) &&
                    tag_u64(
                        record,
                        "participant_flow_violations",
                        participant_violations) &&
                    tag_u64(
                        record,
                        "tier_flow_violations",
                        tier_violations) &&
                    direction_proxy != record.tags.end() &&
                    direction_proxy->second == "false";
                if (valid)
                    addValue(
                        CapacityConservationCertifications,
                        record.value);
                if (!valid || malformed != 0u ||
                    participant_violations != 0u || tier_violations != 0u)
                {
                    ++local[CapacityConservationViolations];
                }
            }
            else if (record.name == "promotions")
                addValue(Promotions, record.value);
            else if (record.name == "demotions")
                addValue(Demotions, record.value);
            else if (record.name == "same_priority_moves")
                addValue(SamePriority, record.value);
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
                uint64_t cycle_index = 0u;
                uint64_t cycle_size = 0u;
                bool direction_valid =
                    direction != record.tags.end() &&
                    source != record.tags.end() &&
                    destination != record.tags.end() &&
                    parse_int(source->second, source_priority) &&
                    parse_int(destination->second, destination_priority) &&
                    tag_u64(record, "cycle_index", cycle_index) &&
                    tag_u64(record, "cycle_size", cycle_size) &&
                    cycle_size > 0u;
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
            EXPECT_EQ(global[SamePriority], 0u);
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
        EXPECT_GE(global[TransportProfileComplete], ranks)
            << "Every dynamic authority must finish bounded transport profiling without synthetic inference";
        EXPECT_GE(global[CertificationComplete], ranks);
        EXPECT_GE(global[AuthorityCertifications], ranks);
        EXPECT_GE(global[MaintenanceCertifications], ranks);
        const uint64_t minimum_epochs_per_rank =
            isQwen122ProductionTest() ? 1u : 2u;
        EXPECT_GE(
            global[ImprovingEpochs],
            minimum_epochs_per_rank * ranks)
            << "Each rank must publish the required distinct improving epoch count";
        EXPECT_EQ(global[ImprovingEpochs], global[CommittedWaves])
            << "Every committed wave must name one distinct publication epoch";
        EXPECT_EQ(global[ServiceGainEvidenceViolations], 0u)
            << "Every committed epoch must improve measured service cost";
        EXPECT_EQ(global[NetBenefitEvidenceViolations], 0u)
            << "Every committed epoch must remain profitable after movement cost and interference";
        EXPECT_GT(global[CommittedMigrations], 0u);
        EXPECT_GT(global[Promotions], 0u);
        EXPECT_GT(global[Demotions], 0u);
        EXPECT_EQ(global[CapacityConservationViolations], 0u)
            << "A committed wave violated typed per-tier or per-participant slot flow";
        EXPECT_EQ(
            global[CapacityConservationCertifications],
            global[CommittedWaves])
            << "Every committed wave must carry one explicit flow-conservation proof";
        if (dynamicMovementAxisContract(*overlay_plan_) ==
            DynamicMovementAxisContract::
                PriorityMigrationAndParticipantBalance)
        {
            EXPECT_GT(global[SamePriority], 0u)
                << "Dynamic residency never reduced participant skew within a tier";
        }
        else
        {
            EXPECT_EQ(global[SamePriority], 0u)
                << "A singleton-priority topology reported an impossible same-priority move";
        }
        EXPECT_EQ(
            global[CommittedMigrations],
            global[Promotions] + global[Demotions] +
                global[SamePriority]);
        EXPECT_EQ(
            global[CrossDomain],
            global[Promotions] + global[Demotions]);
        EXPECT_EQ(
            global[CrossBackend],
            global[Promotions] + global[Demotions]);
        if (activeTopology() == OverlayTopology::CudaRocm)
        {
            EXPECT_EQ(global[CrossRank], global[CommittedMigrations])
                << "The two single-participant GPU domains live on distinct MPI ranks";
        }
        else if (activeTopology() == OverlayTopology::Cuda2Rocm4)
        {
            /*
             * Domain binding follows live NUMA placement. Moving either GPU
             * family between sockets may co-locate the two rank-local domains;
             * cross-domain and cross-backend evidence remains mandatory while
             * a cross-rank edge is required only when the resolved owners differ.
             */
            EXPECT_LE(global[CrossRank], global[CommittedMigrations]);
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
     * @brief Prove request-local LLEP movement or static immobility.
     *
     * Durable tier migration is certified by moe_overlay_residency above.
     * LLEP additionally owns a request-scoped assignment/copy/apply graph, so
     * its proof must come from the independent moe_rebalance counters. Static
     * cells check the same counters at zero; setup-time initial placement is
     * intentionally outside this request-movement surface.
     */
    void assertRequestMovementPolicyEvidence() const
    {
        enum Counter : size_t
        {
            LLEPAssignments,
            LLEPMovementLayers,
            CopiedArrivals,
            AppliedArrivals,
            UsefulPayloadBytes,
            CounterCount,
        };
        std::array<uint64_t, CounterCount> local{};
        const auto add = [&local](Counter counter, double value)
        {
            if (value > 0.0)
                local[counter] += static_cast<uint64_t>(value);
        };
        for (const auto &record :
             PerfStatsCollector::snapshot({"moe_rebalance"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "moe_rebalance")
            {
                continue;
            }
            if (record.name ==
                "device_rebalance_llep_resident_assignment_calls")
                add(LLEPAssignments, record.value);
            else if (record.name ==
                     "device_rebalance_prefill_current_batch_movement_layers")
                add(LLEPMovementLayers, record.value);
            else if (record.name ==
                         "device_rebalance_copy_copied_arrivals" ||
                     record.name ==
                         "device_rebalance_transfer_current_copied_arrivals" ||
                     record.name ==
                         "device_rebalance_wave_copied_arrivals_total")
                add(CopiedArrivals, record.value);
            else if (record.name ==
                         "device_rebalance_apply_applied_arrivals" ||
                     record.name ==
                         "device_rebalance_transfer_current_applied_arrivals" ||
                     record.name ==
                         "device_rebalance_wave_applied_arrivals_total")
                add(AppliedArrivals, record.value);
            else if (record.name ==
                         "device_rebalance_transfer_useful_payload_bytes" ||
                     record.name ==
                         "device_rebalance_request_useful_payload_bytes_lower_bound")
                add(UsefulPayloadBytes, record.value);
        }

        std::array<uint64_t, CounterCount> global{};
        MPI_Allreduce(
            local.data(),
            global.data(),
            static_cast<int>(global.size()),
            MPI_UINT64_T,
            MPI_SUM,
            MPI_COMM_WORLD);
        if (!isRootParityRank() || !isQwen122ProductionTest())
            return;

        if (isLLEPProductionTest())
        {
            EXPECT_GT(global[LLEPAssignments], 0u)
                << "LLEP ran no least-loaded resident assignment";
            EXPECT_GT(global[LLEPMovementLayers], 0u)
                << "LLEP moved no expert payload for the current prefill";
            EXPECT_GT(global[CopiedArrivals], 0u);
            EXPECT_GT(global[AppliedArrivals], 0u);
            EXPECT_GT(global[UsefulPayloadBytes], 0u);
            return;
        }

        if (!isDynamicResidencyProductionTest())
        {
            EXPECT_EQ(global[LLEPAssignments], 0u);
            EXPECT_EQ(global[LLEPMovementLayers], 0u);
            EXPECT_EQ(global[CopiedArrivals], 0u)
                << "Static placement copied a request-time expert payload";
            EXPECT_EQ(global[AppliedArrivals], 0u)
                << "Static placement applied a request-time expert payload";
            EXPECT_EQ(global[UsefulPayloadBytes], 0u)
                << "Static placement transferred request-time expert bytes";
        }
    }

    /** @brief Stable identity of one histogram-driven promotion edge. */
    struct PromotedExpert
    {
        int layer = -1;
        int expert = -1;
        int destination_participant = -1;
        uint64_t candidate_epoch = 0u;

        /** @return Whether two records name the same routed expert. */
        bool operator==(const PromotedExpert &) const = default;
    };

    /** @brief Exact post-publication route and numerical checkpoint witness. */
    struct PromotedExpertExecutionWitness
    {
        ParityForwardPhase phase = ParityForwardPhase::Prefill;
        int step = -1;
        PromotedExpert promotion;
        int selected_placement_bank = -1;
        float expert_output_cosine = 0.0f;
        bool numerically_passed = false;
    };

    /**
     * @brief Retain promotion identities before a parity collector reset.
     *
     * The parity harness resets live PerfStats between campaign phases so the
     * CSV for each numerical comparison has an unambiguous interval. Movement
     * is model-lifetime state and deliberately survives that reset. Preserve
     * only the immutable layer/expert identities from the production movement
     * ledger; routing values and numerical outputs are still read from the
     * later live graph checkpoints.
     */
    void cacheCommittedPromotionEvidence()
    {
        if (!isDynamicResidencyProductionTest() || !isRootParityRank())
            return;

        const bool host_authoritative = topologyUsesCpu();
        const std::string evidence_domain =
            host_authoritative
                ? "moe_overlay_residency"
                : "moe_overlay_controller";
        const std::string evidence_name =
            host_authoritative
                ? "expert_migration_edges"
                : "dynamic_migration_edges";
        size_t malformed_edges = 0;
        for (const auto &record : PerfStatsCollector::snapshot(
                 {evidence_domain}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != evidence_domain ||
                record.name != evidence_name)
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
            const auto destination_tag =
                record.tags.find("destination_participant");
            const auto epoch_tag = record.tags.find("candidate_epoch");
            const auto activation_tag = record.tags.find("activation_count");
            if (layer_tag == record.tags.end() ||
                expert_tag == record.tags.end() ||
                destination_tag == record.tags.end() ||
                epoch_tag == record.tags.end() ||
                (host_authoritative &&
                 activation_tag == record.tags.end()))
            {
                ++malformed_edges;
                continue;
            }

            int layer = -1;
            int expert = -1;
            int destination_participant = -1;
            uint64_t candidate_epoch = 0u;
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
            const bool activation_ok =
                !host_authoritative ||
                parse_u64(activation_tag->second, activations);
            if (!parse_int(layer_tag->second, layer) ||
                !parse_int(expert_tag->second, expert) ||
                !parse_int(
                    destination_tag->second,
                    destination_participant) ||
                !parse_u64(epoch_tag->second, candidate_epoch) ||
                !activation_ok || layer < 0 || expert < 0 ||
                destination_participant < 0 || candidate_epoch == 0u)
            {
                ++malformed_edges;
                continue;
            }
            if (host_authoritative && activations == 0)
                continue;

            const PromotedExpert promotion{
                .layer = layer,
                .expert = expert,
                .destination_participant = destination_participant,
                .candidate_epoch = candidate_epoch,
            };
            if (std::find(
                    promoted_experts_.begin(),
                    promoted_experts_.end(),
                    promotion) == promoted_experts_.end())
            {
                promoted_experts_.push_back(promotion);
            }
        }

        EXPECT_EQ(malformed_edges, 0u)
            << "Committed promotion PerfStats contained malformed identity tags";
    }

    /**
     * @brief Persist the authenticated physical movement ledger used by parity.
     *
     * Numerical CSVs name the layer and routed expert that diverged, but that is
     * not enough to diagnose a moved-weight defect: the production transaction
     * may have crossed a rank, backend, or numeric-priority boundary.  Export the
     * already-published PerfStats edge records before the parity harness resets
     * its measurement interval.  This remains observational test evidence; it
     * does not reconstruct placement or influence the controller.
     */
    void writeCommittedMovementEvidenceCsv() const
    {
        if (!isDynamicResidencyProductionTest() || !isRootParityRank())
            return;

        const bool host_authoritative = topologyUsesCpu();
        const std::string evidence_domain =
            host_authoritative
                ? "moe_overlay_residency"
                : "moe_overlay_controller";
        const std::string evidence_name =
            host_authoritative
                ? "expert_migration_edges"
                : "dynamic_migration_edges";
        const auto records =
            PerfStatsCollector::snapshot({evidence_domain});
        const auto path = ensureResultsDir() / "expert_movement.csv";
        std::ofstream output(path, std::ios::trunc);
        ASSERT_TRUE(output.is_open()) << path;
        output
            << "domain,name,count,value,transaction,candidate_epoch,layer,expert,"
               "cycle_index,cycle_size,direction,source_participant,"
               "destination_participant,"
               "source_priority,destination_priority,source_device,"
               "destination_device,source_world_rank,destination_world_rank,"
               "estimated_weight_bytes,activation_count,blocking_inference,"
               "policy_owner\n";

        const auto tag = [](const PerfStatRecord &record, const char *name)
            -> std::string
        {
            const auto found = record.tags.find(name);
            return found == record.tags.end() ? std::string{} : found->second;
        };
        size_t edge_count = 0;
        for (const auto &record : records)
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != evidence_domain ||
                record.name != evidence_name)
            {
                continue;
            }
            ++edge_count;
            output
                << record.domain << ',' << record.name << ',' << record.count
                << ',' << std::setprecision(17) << record.value << ','
                << tag(record, "transaction") << ','
                << tag(record, "candidate_epoch") << ','
                << tag(record, "layer") << ',' << tag(record, "expert") << ','
                << tag(record, "cycle_index") << ','
                << tag(record, "cycle_size") << ','
                << tag(record, "direction") << ','
                << tag(record, "source_participant") << ','
                << tag(record, "destination_participant") << ','
                << tag(record, "source_priority") << ','
                << tag(record, "destination_priority") << ','
                << tag(record, "source_device") << ','
                << tag(record, "destination_device") << ','
                << tag(record, "source_world_rank") << ','
                << tag(record, "destination_world_rank") << ','
                << tag(record, "estimated_weight_bytes") << ','
                << tag(record, "activation_count") << ','
                << tag(record, "blocking_inference") << ','
                << tag(record, "policy_owner") << '\n';
        }
        output.flush();
        EXPECT_TRUE(output.good()) << path;
        EXPECT_GT(edge_count, 0u)
            << "Dynamic parity produced no authenticated movement-ledger row";
    }

    /**
     * @brief Preserve the complete residency decision trail on a failed gate.
     *
     * A failed movement assertion commonly terminates the coordinated worker
     * lifecycle before `OrchestrationRunner` performs its process-exit
     * PerfStats flush. The compact `expert_movement.csv` intentionally
     * contains committed edges only, so it cannot distinguish a participant
     * rebalance that was never proposed from one rejected by hysteresis,
     * economy, capacity, or the per-wave scheduler. Export the complete
     * residency/controller domains while every tagged decision is still live.
     * This is observational evidence only and cannot affect placement.
     */
    void writeResidencyFailureDiagnosticsCsv() const noexcept
    {
        if (!isDynamicResidencyProductionTest() || !isRootParityRank())
            return;

        try
        {
            const auto path =
                ensureResultsDir() / "expert_residency_diagnostics.csv";
            if (!PerfStatsCollector::writeCsv(
                    path.string(),
                    {"moe_overlay_residency", "moe_overlay_controller"}))
            {
                LOG_ERROR(
                    "[Qwen3.5 MoE GraphNative] Could not write failed-residency diagnostics to "
                    << path);
            }
        }
        catch (const std::exception &error)
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Failed-residency diagnostics raised: "
                << error.what());
        }
        catch (...)
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Failed-residency diagnostics raised a non-standard exception");
        }
    }

    /**
     * @brief Retain exact promoted-expert use from prefill or serial decode.
     *
     * Histogram movement is trained by the complete authenticated request, so
     * a profitable promotion may be hot only during decode.  Restricting the
     * witness to the prompt checkpoint made valid movement pass or fail based
     * on which phase selected the expert.  Observe both numerically compared
     * phases while their immutable route bank is live and retain only compact
     * identity/metric evidence.
     */
    void observeComparedParityCheckpoint(
        ParityForwardPhase phase,
        int step,
        const std::vector<LayerStats> &layers) override
    {
        if (!isDynamicResidencyProductionTest() || !isRootParityRank())
            return;
        const auto *const concrete =
            dynamic_cast<const OrchestrationRunner *>(orch_runner_.get());
        const auto residency = concrete
                                   ? concrete
                                         ->expertOverlayResidencySnapshotForDiagnostics()
                                   : nullptr;
        ASSERT_NE(residency, nullptr);
        ASSERT_TRUE(residency->valid());

        for (const auto &promotion : promoted_experts_)
        {
            size_t route_elements = 0;
            const std::string snapshot_key =
                "layer" + std::to_string(promotion.layer) +
                "_MOE_ROUTING_INDICES";
            const float *const routes =
                activeSnapshot(snapshot_key, route_elements);
            if (!routes)
                continue;
            const auto placement = std::find_if(
                residency->placement_plan->placements.begin(),
                residency->placement_plan->placements.end(),
                [&](const RoutedExpertLayerPlacement &entry)
                { return entry.layer == promotion.layer; });
            ASSERT_NE(
                placement,
                residency->placement_plan->placements.end());
            const auto route_evidence = pinnedDeviceRouteEvidence(
                promotion.layer,
                route_elements,
                placement->routed_expert_tier.size());
            ASSERT_TRUE(route_evidence.has_value());
            ASSERT_GE(promotion.expert, 0);
            ASSERT_LT(
                static_cast<size_t>(promotion.expert),
                route_evidence->expert_count);
            const float placed_participant =
                route_evidence->overlay_participants[promotion.expert];
            if (placed_participant != static_cast<float>(
                                          promotion
                                              .destination_participant))
            {
                // A later committed epoch may have moved this expert again.
                continue;
            }
            const auto *const destination =
                residency->owner_map.participantForId(
                    promotion.destination_participant);
            ASSERT_NE(destination, nullptr);
            const int expected_domain_participant =
                destination->domain_name == overlay_plan_->continuation_domain
                    ? destination->domain_participant_index
                    : -1;

            // Both IDs are stored exactly as FP32 integers in parity dumps.
            bool routed_to_destination = false;
            for (size_t index = 0u; index < route_elements; ++index)
            {
                if (routes[index] !=
                    static_cast<float>(promotion.expert))
                {
                    continue;
                }
                if (route_evidence->domain_participants[index] ==
                    static_cast<float>(expected_domain_participant))
                {
                    routed_to_destination = true;
                    break;
                }
            }
            if (!routed_to_destination)
                continue;

            /*
             * An aggregate layer score can hide one omitted routed
             * contribution.  MOE_EXPERT_OUTPUT is the first checkpoint below
             * both the acquired placement bank and domain schedule, so retain
             * its exact numerical verdict with the route identity.
             */
            const auto layer_it = std::find_if(
                layers.begin(),
                layers.end(),
                [&](const LayerStats &stats)
                { return stats.layer_idx == promotion.layer; });
            ASSERT_NE(layer_it, layers.end())
                << "No parity summary was produced for promoted-expert layer "
                << promotion.layer;

            const auto expert_output_it = std::find_if(
                layer_it->stage_results.begin(),
                layer_it->stage_results.end(),
                [](const StageComparisonResult &result)
                { return result.stage_name == "MOE_EXPERT_OUTPUT"; });
            ASSERT_NE(expert_output_it, layer_it->stage_results.end())
                << "Promoted-expert layer " << promotion.layer
                << " omitted the MOE_EXPERT_OUTPUT checkpoint";

            const auto duplicate = std::find_if(
                promoted_expert_execution_witnesses_.begin(),
                promoted_expert_execution_witnesses_.end(),
                [&](const PromotedExpertExecutionWitness &witness)
                {
                    return witness.phase == phase && witness.step == step &&
                           witness.promotion == promotion;
                });
            if (duplicate == promoted_expert_execution_witnesses_.end())
            {
                promoted_expert_execution_witnesses_.push_back(
                    PromotedExpertExecutionWitness{
                        .phase = phase,
                        .step = step,
                        .promotion = promotion,
                        .selected_placement_bank =
                            route_evidence->selected_bank,
                        .expert_output_cosine =
                            expert_output_it->cosine_similarity,
                        .numerically_passed = expert_output_it->passed,
                    });
            }
            EXPECT_TRUE(expert_output_it->passed)
                << "Promoted-expert layer " << promotion.layer
                << " failed at the first routed numerical boundary: cosine="
                << expert_output_it->cosine_similarity
                << " threshold=" << config_.cosine_threshold;
        }
    }

    /**
     * @brief Assert and export a post-publication promoted-expert witness.
     *
     * Movement, route selection, and numerical comparison remain three
     * independently produced authorities.  This epilogue only joins their
     * immutable evidence; it neither chooses a route nor causes maintenance.
     */
    void assertParityExecutionExercisesPromotedExpert() const
    {
        if (!isDynamicResidencyProductionTest() || !isRootParityRank())
            return;

        ASSERT_FALSE(promoted_experts_.empty())
            << "Dynamic residency committed no promotion edge";

        const auto csv_path =
            ensureResultsDir() / "promoted_expert_execution.csv";
        std::ofstream csv(csv_path, std::ios::trunc);
        ASSERT_TRUE(csv.is_open()) << csv_path;
        csv << "phase,step,layer,expert,destination_participant,"
               "candidate_epoch,selected_placement_bank,"
               "moe_expert_output_cosine,numerically_passed\n";
        for (const auto &witness : promoted_expert_execution_witnesses_)
        {
            csv << parityForwardPhaseName(witness.phase) << ','
                << witness.step << ','
                << witness.promotion.layer << ','
                << witness.promotion.expert << ','
                << witness.promotion.destination_participant << ','
                << witness.promotion.candidate_epoch << ','
                << witness.selected_placement_bank << ','
                << witness.expert_output_cosine << ','
                << (witness.numerically_passed ? "true" : "false")
                << '\n';
        }
        csv.flush();
        ASSERT_TRUE(csv.good()) << csv_path;

        std::ostringstream candidates;
        for (const auto &promotion : promoted_experts_)
        {
            if (candidates.tellp() > 0)
                candidates << ", ";
            candidates << "layer" << promotion.layer << ":expert"
                       << promotion.expert << "->participant"
                       << promotion.destination_participant << "@epoch"
                       << promotion.candidate_epoch;
        }
        ASSERT_FALSE(promoted_expert_execution_witnesses_.empty())
            << "No numerically compared post-publication prefill or decode "
               "checkpoint executed a promoted expert on its acquired "
               "destination; candidates: "
            << candidates.str();
        EXPECT_TRUE(std::all_of(
            promoted_expert_execution_witnesses_.begin(),
            promoted_expert_execution_witnesses_.end(),
            [](const PromotedExpertExecutionWitness &witness)
            { return witness.numerically_passed; }));
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

    /**
     * @brief Test whether one exact noncanonical HF branch is already complete.
     *
     * Missing branches are deliberately not generated here. This method runs
     * while the production graph and prepared 122B weights are resident; loading
     * the Python model at this boundary previously overlapped roughly 500 GB of
     * live state and was killed by the host OOM policy. The immutable production
     * checkpoints are queued below and the suite resolves them after all model
     * authorities retire.
     *
     * @param reference_step Main decode step that owns the sidecar transaction.
     * @param condition_tokens Recursive condition tokens consumed by MTP1..N.
     * @return True only when the deepest branch checkpoint is complete on disk.
     */
    bool hasHuggingFaceMTPBranchReference(
        int reference_step,
        const std::vector<int32_t> &condition_tokens) const
    {
        if (reference_step < 0 || condition_tokens.empty() ||
            condition_tokens.size() >=
                static_cast<size_t>(kQwen122MaximumMTPDraftDepth) ||
            std::any_of(
                condition_tokens.begin(),
                condition_tokens.end(),
                [](int32_t token) { return token < 0; }))
        {
            return false;
        }

        std::ostringstream qualifier;
        for (const int32_t token : condition_tokens)
            qualifier << '_' << token;
        const std::string branch_stem =
            "decode_step" + std::to_string(reference_step) + "_BRANCH" +
            qualifier.str() + "_MTP" +
            std::to_string(condition_tokens.size());
        const std::filesystem::path deepest_lm_head =
            std::filesystem::path(config_.snapshot_dir) /
            (branch_stem + "_LM_HEAD.npy");
        const std::filesystem::path deepest_embedding =
            std::filesystem::path(config_.snapshot_dir) /
            (branch_stem + "_EMBEDDING.npy");
        return std::filesystem::is_regular_file(deepest_lm_head) &&
               std::filesystem::is_regular_file(deepest_embedding);
    }

    /**
     * @brief Copy one missing recursive context into the post-residency queue.
     * @param call Public grouped-decode call index used by the CSV.
     * @param reference_step Canonical main-model decode position.
     * @param reference_depth Number of recursive condition tokens consumed.
     * @param condition_tokens Exact device-owned condition-token trajectory.
     * @param production_prefix Snapshot namespace for the live recursive row.
     * @param required_stages Complete sidecar checkpoint contract.
     * @param snapshot_csv_path Existing per-cell diagnostic CSV to append later.
     * @return True only when every live checkpoint was copied successfully.
     */
    bool deferHuggingFaceMTPBranchReference(
        int call,
        int reference_step,
        int reference_depth,
        const std::vector<int32_t> &condition_tokens,
        const std::string &production_prefix,
        const std::array<std::string_view, 20> &required_stages,
        const std::filesystem::path &snapshot_csv_path)
    {
        DeferredMTPBranchContext context{
            .test_name = activeTestName(),
            .model_path = config_.model_path,
            .prompt = config_.prompt,
            .snapshot_dir = config_.snapshot_dir,
            .snapshot_csv_path = snapshot_csv_path,
            .decode_steps = config_.decode_steps,
            .call = call,
            .reference_step = reference_step,
            .reference_depth = reference_depth,
            .vocab_size = orch_runner_ ? orch_runner_->vocabSize() : 0,
            .cosine_threshold = config_.cosine_threshold,
            .decode_cosine_threshold = config_.decode_cosine_threshold,
            .kl_threshold = config_.kl_threshold,
            .condition_tokens = condition_tokens,
        };
        const auto moe = getMoEConfig();
        context.top_k = moe.top_k;
        context.num_experts = moe.num_experts;
        context.checkpoints.reserve(required_stages.size());
        for (const std::string_view stage : required_stages)
        {
            const std::string production_key =
                production_prefix + "MTP0_" + std::string(stage);
            size_t elements = 0u;
            const float *const actual =
                activeSnapshot(production_key, elements);
            if (!actual || elements == 0u)
            {
                ADD_FAILURE()
                    << "Live sidecar omitted deferred checkpoint "
                    << production_key;
                return false;
            }
            context.checkpoints.push_back({
                .stage = std::string(stage),
                .production_key = production_key,
                .actual = std::vector<float>(actual, actual + elements),
            });
        }

        auto &campaign = deferredMTPBranchCampaign();
        std::lock_guard<std::mutex> lock(campaign.mutex);
        campaign.contexts.push_back(std::move(context));
        return true;
    }

    /**
     * @brief Compare the live grouped-MTP graph with recursive HF checkpoints.
     *
     * The classic decode parity loop is deliberately teacher forced so every
     * main-model row follows the exact Hugging Face trajectory. That loop does
     * not execute speculative sidecars. This check first records a serial
     * production trajectory by constraining the public decodeStep boundary to
     * one token. When both requests hold the same residency epoch, grouped MTP
     * must reproduce that trajectory exactly. Dynamic and LLEP requests may
     * legitimately publish a new placement between those independent requests;
     * such rows are instead compared directly with their Hugging Face main-model
     * checkpoints and the epoch mismatch is retained in the diagnostic CSV.
     *
     * Sidecar tensors remain compared directly with Hugging Face whenever the
     * serial production prefix still names the same main-model row. Recursive
     * predictors select branch-qualified reference tensors using the proposal
     * tokens observed from the device authority. This keeps every checkpoint
     * mathematically comparable even when a narrow quantized-logit tie sends
     * production down a different draft branch from canonical HF argmax. The
     * two diagnostic CSVs complement—never replace—the six canonical
     * prefill/decode artifacts.
     */
    void runMTPHuggingFaceCheckpointParity()
    {
        if (!isQwen122ProductionTest() || !isRootParityRank())
            return;

        ASSERT_NE(orch_runner_, nullptr);
        const std::vector<int> expected_tokens =
            readDecodeTokensFromMetadata();
        ASSERT_GE(expected_tokens.size(), 2u)
            << "MTP parity needs a sampled prefill token and one sidecar transaction";

        const auto result_dir = ensureResultsDir();
        const auto token_csv_path =
            result_dir / "mtp_sidecar_token_trace.csv";
        const auto snapshot_csv_path =
            result_dir / "mtp_sidecar_snapshot_breakdown.csv";
        std::ofstream token_csv(token_csv_path, std::ios::trunc);
        std::ofstream snapshot_csv(snapshot_csv_path, std::ios::trunc);
        ASSERT_TRUE(token_csv.is_open()) << token_csv_path;
        ASSERT_TRUE(snapshot_csv.is_open()) << snapshot_csv_path;
        token_csv
            << "call,reference_step,selected_depth,emitted_tokens,"
               "serial_expected_tokens,hf_expected_tokens,hf_branch_compatible,"
               "serial_epoch_compatible,serial_movement_epoch,"
               "grouped_movement_epoch_begin,grouped_movement_epoch_end,"
               "production_mtp0_top1,hf_mtp0_top1,recursive_branch_compatible,"
               "verifier_identity_transaction_count,verifier_identity_depth,"
               "production_verifier_draft_tokens,"
               "draft_steps,verifier_runs,accepted,rejected,commits,"
               "rollbacks,validation_failures,current_position\n";
        snapshot_csv
            << "call,reference_step,reference_depth,production_key,reference_key,"
               "elements,cosine,max_abs_diff,kl,exact_indices,routing_overlap,"
               "routing_top1_match,finite,passed\n";

        const std::array<std::string_view, 20> required_stages = {
            "EMBEDDING",
            "NORM_HIDDEN",
            "CONCAT",
            "FC",
            "ATTENTION_NORM",
            "Q_PROJECTION",
            "ATTENTION_CONTEXT",
            "ATTENTION_OUTPUT",
            "FFN_NORM",
            "MOE_ROUTER_OUTPUT",
            "MOE_ROUTING_INDICES",
            "MOE_ROUTING_WEIGHTS",
            "MOE_EXPERT_OUTPUT",
            "MOE_SHARED_EXPERT_OUTPUT",
            "MOE_SHARED_GATE_OUTPUT",
            "MOE_COMBINED_OUTPUT",
            "FFN_RESIDUAL",
            "FINAL_NORM",
            "LM_HEAD",
            "V_PROJECTION",
        };

        const auto join_tokens = [](const std::vector<int32_t> &tokens)
        {
            std::ostringstream out;
            for (size_t index = 0; index < tokens.size(); ++index)
            {
                if (index != 0)
                    out << ';';
                out << tokens[index];
            }
            return out.str();
        };
        const std::array<std::string_view, 17> main_verifier_stage_suffixes = {
            "ATTENTION_NORM",
            "Q_PROJECTION",
            "ATTENTION_CONTEXT",
            "ATTENTION_OUTPUT",
            "FFN_NORM",
            "MOE_ROUTER_OUTPUT",
            "MOE_ROUTING_INDICES",
            "MOE_ROUTING_WEIGHTS",
            "MOE_EXPERT_OUTPUT",
            "MOE_SHARED_EXPERT_OUTPUT",
            "MOE_SHARED_GATE_OUTPUT",
            "MOE_COMBINED_OUTPUT",
            "FFN_RESIDUAL",
            "GDN_CONV1D_OUTPUT",
            "GDN_DELTA_RULE_OUTPUT",
            "GDN_NORM_GATE_OUTPUT",
            "GDN_OUTPUT",
        };
        const auto is_main_verifier_diagnostic_key =
            [&](const std::string &key)
        {
            if (key == "EMBEDDING" || key == "FINAL_NORM" ||
                key == "LM_HEAD" || key == "LM_HEAD_ROWS_SELECT")
            {
                return true;
            }
            if (key.rfind("layer", 0) != 0)
                return false;
            return std::any_of(
                main_verifier_stage_suffixes.begin(),
                main_verifier_stage_suffixes.end(),
                [&](std::string_view suffix)
                {
                    return std::string_view(key).ends_with(suffix);
                });
        };
        const auto capture_main_verifier_diagnostics = [&]
        {
            std::map<std::string, std::vector<float>> snapshots;
            for (const auto &key : activeSnapshotKeys())
            {
                if (!is_main_verifier_diagnostic_key(key))
                    continue;
                size_t elements = 0;
                const float *const data = activeSnapshot(key, elements);
                if (data && elements > 0)
                    snapshots.emplace(key, std::vector<float>(data, data + elements));
            }
            return snapshots;
        };
        /*
         * Record the serial production oracle through the same public surface
         * used by the HTTP server. A one-token response budget cannot enter a
         * speculative transaction, but it still advances the captured main
         * graph and its shifted-MTP state exactly as a normal request does.
         *
         * Residency movement was already proved before numerical parity. Do
         * not submit an additional test-driven maintenance epoch here. Normal
         * Dynamic/LLEP requests can still publish histogram work between these
         * independent serving calls, so each captured row records its actual
         * movement epoch. Only rows with matching epochs form a valid strict
         * serial-equivalence experiment; all grouped rows retain the direct HF
         * oracle below.
         */
        activeClearSnapshots();
        activeClearCache();
        ASSERT_TRUE(orch_runner_->prefill(config_.token_ids))
            << orch_runner_->lastError();
        const uint64_t mtp_certification_movement_epoch =
            orch_runner_->moeRuntimeMovementEpoch();
        /** @brief One serial row plus the residency epoch that executed it. */
        struct SerialVerifierOracle
        {
            std::map<std::string, std::vector<float>> snapshots;
            uint64_t movement_epoch_begin = 0;
            uint64_t movement_epoch_end = 0;

            /** @return Whether this row executed wholly within @p epoch. */
            bool executedInEpoch(uint64_t epoch) const
            {
                return movement_epoch_begin == epoch &&
                       movement_epoch_end == epoch;
            }
        };
        std::vector<int32_t> serial_tokens;
        const size_t serial_oracle_token_count = std::max(
            expected_tokens.size(),
            usesDynamicMTPDepth()
                ? static_cast<size_t>(activeMTPDraftDepth() + 1)
                : expected_tokens.size());
        std::vector<SerialVerifierOracle>
            serial_oracles_by_output_count(serial_oracle_token_count + 1u);
        while (serial_tokens.size() < serial_oracle_token_count)
        {
            activeClearSnapshots();
            orch_runner_->setDecodeStepTokenBudget(1);
            const uint64_t movement_epoch_begin =
                orch_runner_->moeRuntimeMovementEpoch();
            const GenerationResult serial_step = orch_runner_->decodeStep();
            const uint64_t movement_epoch_end =
                orch_runner_->moeRuntimeMovementEpoch();
            ASSERT_TRUE(serial_step.success()) << serial_step.error;
            ASSERT_EQ(serial_step.tokens.size(), 1u)
                << "A one-token serial oracle boundary returned a grouped response";
            serial_tokens.push_back(serial_step.tokens.front());
            serial_oracles_by_output_count[serial_tokens.size()] = {
                .snapshots = capture_main_verifier_diagnostics(),
                .movement_epoch_begin = movement_epoch_begin,
                .movement_epoch_end = movement_epoch_end,
            };
        }
        orch_runner_->setDecodeStepTokenBudget(0);

        ASSERT_EQ(serial_tokens.size(), serial_oracle_token_count);
        ASSERT_GE(serial_tokens.size(), expected_tokens.size());
        ASSERT_EQ(serial_tokens.front(), expected_tokens.front())
            << "The prefill boundary must agree exactly before MTP branch comparison";

        /* Start a fresh request so the first grouped sidecar is decode_step0. */
        activeClearSnapshots();
        activeClearCache();
        ASSERT_TRUE(orch_runner_->prefill(config_.token_ids))
            << orch_runner_->lastError();

        std::vector<int32_t> emitted;
        const auto initial_state = activePrefixStateProbe();
        int speculative_calls = 0;
        int compared_stages = 0;
        int compared_recursive_stages = 0;
        int deferred_recursive_contexts = 0;
        int compared_main_stages = 0;
        int compared_main_lm_heads = 0;
        int failed_main_lm_heads = 0;
        double grouped_main_cosine_sum = 0.0;
        size_t grouped_main_cosine_count = 0u;
        int serial_epoch_compatible_calls = 0;
        int call = 0;
        while (emitted.size() < expected_tokens.size())
        {
            activeClearSnapshots();
            const auto before = activePrefixStateProbe();
            const int remaining = static_cast<int>(
                expected_tokens.size() - emitted.size());
            int selected_depth_before = usesDynamicMTPDepth()
                                            ? before.mtp_current_depth
                                            : activeMTPDraftDepth();
            if (selected_depth_before <= 0)
                selected_depth_before = activeMTPDraftDepth();
            ASSERT_GE(selected_depth_before, 1);
            ASSERT_LE(selected_depth_before, activeMTPDraftDepth());
            /*
             * A two-token response boundary is the smallest public serving
             * request that admits speculative execution. A rejection can let
             * one public call retire more than one device transaction, so the
             * counter fold below validates every retired transaction. The
             * row-zero snapshot comparison is enabled only when the counters
             * prove that this call has one unambiguous verifier identity.
             */
            orch_runner_->setDecodeStepTokenBudget(
                std::min(remaining, 2));
            const uint64_t grouped_movement_epoch_begin =
                orch_runner_->moeRuntimeMovementEpoch();
            const GenerationResult step = orch_runner_->decodeStep();
            const uint64_t grouped_movement_epoch_end =
                orch_runner_->moeRuntimeMovementEpoch();
            ASSERT_TRUE(step.success()) << step.error;
            ASSERT_FALSE(step.tokens.empty());
            ASSERT_LE(step.tokens.size(), static_cast<size_t>(remaining));
            const auto after = activePrefixStateProbe();
            const size_t output_begin = emitted.size();

            bool serial_epoch_compatible =
                grouped_movement_epoch_begin == grouped_movement_epoch_end;
            uint64_t serial_movement_epoch =
                grouped_movement_epoch_begin;
            for (size_t offset = 0; offset < step.tokens.size(); ++offset)
            {
                const size_t oracle_index = output_begin + offset + 1u;
                ASSERT_LT(oracle_index, serial_oracles_by_output_count.size());
                const auto &oracle =
                    serial_oracles_by_output_count[oracle_index];
                serial_epoch_compatible =
                    serial_epoch_compatible &&
                    oracle.executedInEpoch(grouped_movement_epoch_begin);
                if (offset == 0)
                    serial_movement_epoch = oracle.movement_epoch_begin;
            }
            if (serial_epoch_compatible)
                ++serial_epoch_compatible_calls;

            for (size_t offset = 0; offset < step.tokens.size(); ++offset)
            {
                const size_t output_index = emitted.size() + offset;
                ASSERT_LT(output_index, expected_tokens.size());
                if (serial_epoch_compatible)
                {
                    EXPECT_EQ(step.tokens[offset], serial_tokens[output_index])
                        << "Grouped MTP diverged from the serial production token "
                           "trajectory at output "
                        << output_index;
                }
            }

            const MTPParityTransactionCounters before_counters{
                .draft_steps = before.mtp_draft_steps,
                .verifier_runs = before.mtp_verifier_runs,
            };
            const MTPParityTransactionCounters after_counters{
                .draft_steps = after.mtp_draft_steps,
                .verifier_runs = after.mtp_verifier_runs,
            };
            const auto transaction_activity =
                classifyMTPParityTransactionActivity(
                    before_counters,
                    after_counters);
            EXPECT_NE(
                transaction_activity,
                MTPParityTransactionActivity::Inconsistent)
                << "MTP draft/verifier counters advanced inconsistently: before "
                << "drafts=" << before.mtp_draft_steps
                << " verifiers=" << before.mtp_verifier_runs
                << " after drafts=" << after.mtp_draft_steps
                << " verifiers=" << after.mtp_verifier_runs;
            const bool speculative =
                transaction_activity ==
                MTPParityTransactionActivity::Speculative;
            const uint64_t executed_transaction_count =
                mtpParityExecutedTransactionCount(
                    before_counters,
                    after_counters);
            const uint64_t attempted_draft_tokens =
                mtpParityAttemptedDraftTokenCount(
                    before_counters,
                    after_counters);

            int reference_step = -1;
            int selected_depth = 0;
            int production_mtp0_top1 = -1;
            int hf_mtp0_top1 = -1;
            bool recursive_branch_compatible = false;
            bool hf_branch_compatible = false;
            if (speculative)
            {
                ++speculative_calls;
                reference_step =
                    mtpParityReferenceStepForConditionPosition(
                        before.mtp_next_condition_position,
                        static_cast<int>(config_.token_ids.size()));
                ASSERT_GE(reference_step, 0)
                    << "Device MTP controller omitted its live condition position";
                selected_depth = after.mtp_last_transaction_draft_depth;
                ASSERT_GE(selected_depth, 1);
                ASSERT_LE(selected_depth, activeMTPDraftDepth());
                EXPECT_EQ(selected_depth, selected_depth_before)
                    << "The device-owned transaction selected a different "
                       "depth than its pre-submission controller state";
                const size_t reference_prefix_length =
                    static_cast<size_t>(reference_step + 1);
                std::vector<int32_t> grouped_visible_prefix = emitted;
                grouped_visible_prefix.insert(
                    grouped_visible_prefix.end(),
                    step.tokens.begin(),
                    step.tokens.end());
                hf_branch_compatible =
                    reference_prefix_length <= expected_tokens.size() &&
                    reference_prefix_length <= grouped_visible_prefix.size() &&
                    std::equal(
                        grouped_visible_prefix.begin(),
                        grouped_visible_prefix.begin() + reference_prefix_length,
                        expected_tokens.begin());
                ASSERT_GE(executed_transaction_count, 1u);
                ASSERT_GE(attempted_draft_tokens, executed_transaction_count)
                    << "Every retired verifier transaction must attempt a draft";
                ASSERT_LE(
                    attempted_draft_tokens,
                    executed_transaction_count *
                        static_cast<uint64_t>(activeMTPDraftDepth()))
                    << "The device controller attempted more drafts than its "
                       "captured transaction capacity";
                ASSERT_EQ(
                    after.mtp_observed_verifier_transaction_count,
                    static_cast<int>(executed_transaction_count))
                    << "The durable verifier identity must be committed by "
                       "the same device transaction(s) reported by production";
                ASSERT_EQ(
                    after.mtp_observed_verifier_draft_depth,
                    selected_depth)
                    << "The durable verifier identity names a different "
                       "dynamic-depth branch than the committed controller";
                ASSERT_EQ(
                    after.mtp_observed_verifier_draft_tokens.size(),
                    static_cast<size_t>(selected_depth))
                    << "Every committed draft token must have an exact, "
                       "response-visible verifier identity";

                const std::string first_sidecar_prefix = std::string(
                    mtpParityCheckpointContextPrefix(
                        before.mtp_verifier_runs == 0
                            ? MTPParityCheckpointContext::
                                  DeviceTargetTokenLivePosition
                            : MTPParityCheckpointContext::
                                  DeviceResidentLogicalState));
                size_t production_lm_head_size = 0;
                const float *production_lm_head = activeSnapshot(
                    first_sidecar_prefix + "MTP0_LM_HEAD",
                    production_lm_head_size);
                const std::vector<float> hf_mtp0_lm_head =
                    loadPyTorchSnapshot(
                        "decode_step" + std::to_string(reference_step) +
                        "_MTP0_LM_HEAD");
                ASSERT_NE(production_lm_head, nullptr);
                ASSERT_EQ(production_lm_head_size, hf_mtp0_lm_head.size());
                production_mtp0_top1 = static_cast<int>(std::distance(
                    production_lm_head,
                    std::max_element(
                        production_lm_head,
                        production_lm_head + production_lm_head_size)));
                hf_mtp0_top1 = static_cast<int>(std::distance(
                    hf_mtp0_lm_head.begin(),
                    std::max_element(
                        hf_mtp0_lm_head.begin(), hf_mtp0_lm_head.end())));
                recursive_branch_compatible =
                    production_mtp0_top1 == hf_mtp0_top1;

                struct ActiveContext
                {
                    std::string prefix;
                    int reference_depth = 0;
                };
                std::vector<ActiveContext> contexts;
                contexts.push_back({
                    .prefix = std::string(
                        mtpParityCheckpointContextPrefix(
                            before.mtp_verifier_runs == 0
                                ? MTPParityCheckpointContext::DeviceTargetTokenLivePosition
                                : MTPParityCheckpointContext::DeviceResidentLogicalState)),
                    .reference_depth = 0,
                });
                const size_t recursive_condition_tokens =
                    selected_depth > 1
                        ? static_cast<size_t>(selected_depth - 1)
                        : 0u;
                const bool has_recursive_proposal_identity =
                    recursive_condition_tokens > 0u &&
                    after.mtp_observed_verifier_draft_tokens.size() >=
                        recursive_condition_tokens &&
                    std::all_of(
                        after.mtp_observed_verifier_draft_tokens.begin(),
                        after.mtp_observed_verifier_draft_tokens.begin() +
                            static_cast<ptrdiff_t>(recursive_condition_tokens),
                        [](int32_t token) { return token >= 0; });
                bool recursive_reference_deferred = false;
                std::vector<int32_t> recursive_reference_condition_tokens;

                /**
                 * @brief Decide whether an observed proposal follows the
                 *        canonical Hugging Face recursive branch.
                 *
                 * Canonical recursive snapshots have no `_BRANCH_...`
                 * qualifier. Additive snapshots use that qualifier only after
                 * production selects a condition token different from the
                 * corresponding canonical HF predictor. Compare the complete
                 * preceding predictor chain so a depth-two checkpoint is
                 * canonical only when both condition tokens agree.
                 *
                 * @param reference_depth Number of recursive condition tokens
                 *        consumed by the requested checkpoint.
                 * @return True when the unqualified HF checkpoint is the exact
                 *         oracle for the observed production branch.
                 */
                const auto follows_canonical_hf_branch =
                    [&](int reference_depth) -> bool
                {
                    if (reference_depth < 0 ||
                        after.mtp_observed_verifier_draft_tokens.size() <
                            static_cast<size_t>(reference_depth))
                    {
                        ADD_FAILURE()
                            << "Cannot resolve recursive HF branch depth "
                            << reference_depth << " from "
                            << after.mtp_observed_verifier_draft_tokens.size()
                            << " observed proposal tokens";
                        return false;
                    }
                    for (int proposal_depth = 0;
                         proposal_depth < reference_depth;
                         ++proposal_depth)
                    {
                        const std::vector<float> canonical_logits =
                            loadPyTorchSnapshot(
                                "decode_step" +
                                std::to_string(reference_step) + "_MTP" +
                                std::to_string(proposal_depth) +
                                "_LM_HEAD");
                        if (canonical_logits.empty())
                        {
                            ADD_FAILURE()
                                << "Hugging Face pack omitted the canonical MTP"
                                << proposal_depth
                                << " LM-head oracle at decode step "
                                << reference_step;
                            return false;
                        }
                        const int32_t canonical_token = static_cast<int32_t>(
                            std::distance(
                                canonical_logits.begin(),
                                std::max_element(
                                    canonical_logits.begin(),
                                    canonical_logits.end())));
                        if (after.mtp_observed_verifier_draft_tokens[
                                static_cast<size_t>(proposal_depth)] !=
                            canonical_token)
                        {
                            return false;
                        }
                    }
                    return true;
                };
                if (has_recursive_proposal_identity)
                {
                    const int recursive_reference_depth = selected_depth - 1;
                    recursive_branch_compatible =
                        follows_canonical_hf_branch(
                            recursive_reference_depth);
                    if (!recursive_branch_compatible)
                    {
                        recursive_reference_condition_tokens.assign(
                            after.mtp_observed_verifier_draft_tokens.begin(),
                            after.mtp_observed_verifier_draft_tokens.begin() +
                                recursive_reference_depth);
                        recursive_reference_deferred =
                            hf_branch_compatible &&
                            !hasHuggingFaceMTPBranchReference(
                                reference_step,
                                recursive_reference_condition_tokens);
                    }
                    contexts.push_back({
                        .prefix = std::string(
                            mtpParityCheckpointContextPrefix(
                                MTPParityCheckpointContext::DeviceChainedTokenLivePosition)),
                        .reference_depth = recursive_reference_depth,
                    });
                }

                for (const auto &context : contexts)
                {
                    if (context.reference_depth > 0 &&
                        recursive_reference_deferred)
                    {
                        ASSERT_TRUE(deferHuggingFaceMTPBranchReference(
                            call,
                            reference_step,
                            context.reference_depth,
                            recursive_reference_condition_tokens,
                            context.prefix,
                            required_stages,
                            snapshot_csv_path))
                            << "Could not preserve the live recursive MTP "
                               "checkpoint set before runner retirement";
                        ++deferred_recursive_contexts;
                        continue;
                    }
                    /*
                     * A quantized router may exchange only the lowest-weight
                     * top-k boundary expert while retaining the top-1 expert
                     * and k-1 set overlap. The routing checkpoint below proves
                     * that bounded discrete difference explicitly. Remember
                     * whether it occurred so the immediately dependent raw
                     * routed sum uses the established intermediate-tensor
                     * threshold; exact routes retain the stricter decode floor.
                     * Downstream combined output and LM-head checks are never
                     * relaxed by this state.
                     */
                    bool routing_indices_exact = true;
                    bool context_finite = true;
                    bool context_routing_top1_match = true;
                    float context_routing_overlap = 1.0f;
                    bool context_routing_weights_equivalent = false;
                    bool context_routed_expert_output_equivalent = false;
                    bool context_lm_head_passed = false;
                    double context_numerical_cosine_sum = 0.0;
                    size_t context_numerical_stage_count = 0u;
                    const auto sidecar_moe = getMoEConfig();
                    for (const std::string_view stage : required_stages)
                    {
                        const std::string production_key =
                            context.prefix + "MTP0_" + std::string(stage);
                        std::string reference_prefix =
                            "decode_step" + std::to_string(reference_step);
                        if (context.reference_depth > 0)
                        {
                            ASSERT_GE(
                                after.mtp_observed_verifier_draft_tokens.size(),
                                static_cast<size_t>(context.reference_depth))
                                << "The proposal authority omitted tokens "
                                   "needed to identify recursive HF depth "
                                << context.reference_depth;
                            if (!follows_canonical_hf_branch(
                                    context.reference_depth))
                            {
                                reference_prefix += "_BRANCH";
                                for (int branch_depth = 0;
                                     branch_depth < context.reference_depth;
                                     ++branch_depth)
                                {
                                    reference_prefix += "_" + std::to_string(
                                        after.mtp_observed_verifier_draft_tokens[
                                            static_cast<size_t>(branch_depth)]);
                                }
                            }
                        }
                        const std::string reference_key =
                            reference_prefix + "_MTP" +
                            std::to_string(context.reference_depth) + "_" +
                            std::string(stage);
                        if (!hf_branch_compatible)
                            continue;

                        size_t actual_size = 0;
                        const float *actual =
                            activeSnapshot(production_key, actual_size);
                        const std::vector<float> reference =
                            loadPyTorchSnapshot(reference_key);
                        ASSERT_NE(actual, nullptr)
                            << "Live sidecar omitted " << production_key;
                        ASSERT_GT(actual_size, 0u)
                            << "Live sidecar published an empty " << production_key;
                        ASSERT_FALSE(reference.empty())
                            << "Hugging Face pack omitted " << reference_key;

                        const float *expected = reference.data();
                        size_t expected_size = reference.size();
                        if (expected_size > actual_size && actual_size > 0 &&
                            expected_size % actual_size == 0)
                        {
                            expected += expected_size - actual_size;
                            expected_size = actual_size;
                        }
                        ASSERT_EQ(actual_size, expected_size)
                            << production_key << " versus " << reference_key;

                        bool finite = true;
                        bool exact_indices = true;
                        float routing_overlap = 1.0f;
                        bool routing_top1_match = true;
                        double max_abs_diff = 0.0;
                        for (size_t index = 0; index < actual_size; ++index)
                        {
                            finite = finite && std::isfinite(actual[index]) &&
                                     std::isfinite(expected[index]);
                            max_abs_diff = std::max(
                                max_abs_diff,
                                std::abs(
                                    static_cast<double>(actual[index]) -
                                    static_cast<double>(expected[index])));
                            if (stage == "MOE_ROUTING_INDICES")
                            {
                                exact_indices = exact_indices &&
                                                actual[index] == expected[index];
                            }
                        }
                        float cosine = computeCosineSimilarity(
                            actual, expected, actual_size);
                        float kl = 0.0f;
                        bool passed = finite;
                        if (stage == "MOE_ROUTING_INDICES")
                        {
                            /*
                             * Quantized router logits may swap one expert at the
                             * low-weight top-k boundary even when the routing
                             * distribution and routed value remain numerically
                             * equivalent. Preserve the highest-weight expert and
                             * require at least k-1 set overlap, matching the
                             * production parity suite's discrete-routing model.
                             */
                            std::set<int> actual_experts;
                            std::set<int> reference_experts;
                            for (size_t index = 0; index < actual_size; ++index)
                            {
                                actual_experts.insert(
                                    static_cast<int>(actual[index]));
                                reference_experts.insert(
                                    static_cast<int>(expected[index]));
                            }
                            size_t intersection = 0;
                            for (const int expert : actual_experts)
                            {
                                if (reference_experts.contains(expert))
                                    ++intersection;
                            }
                            routing_overlap = actual_size > 0
                                                  ? static_cast<float>(intersection) /
                                                        static_cast<float>(actual_size)
                                                  : 0.0f;
                            routing_top1_match =
                                actual_size > 0 && actual[0] == expected[0];
                            routing_indices_exact = exact_indices;
                            cosine = routing_overlap;
                            max_abs_diff = 1.0 - routing_overlap;
                            const float minimum_boundary_overlap =
                                sidecar_moe.top_k > 0
                                    ? 1.0f -
                                          1.0f /
                                              static_cast<float>(
                                                  sidecar_moe.top_k)
                                    : 1.0f;
                            passed = passed && routing_top1_match &&
                                     routing_overlap >=
                                         minimum_boundary_overlap;
                            context_routing_top1_match =
                                context_routing_top1_match &&
                                routing_top1_match;
                            context_routing_overlap = std::min(
                                context_routing_overlap,
                                routing_overlap);
                        }
                        else if (stage == "MOE_ROUTING_WEIGHTS")
                        {
                            /*
                             * Ordered weight vectors can look identical even
                             * when their low-weight entries name different
                             * experts. Reconstruct sparse expert-ID vectors,
                             * exactly as the canonical decode campaign does,
                             * so the routing proof measures contribution mass
                             * rather than array position.
                             */
                            size_t actual_indices_size = 0u;
                            const float *const actual_indices = activeSnapshot(
                                context.prefix +
                                    "MTP0_MOE_ROUTING_INDICES",
                                actual_indices_size);
                            const std::vector<float> reference_indices =
                                loadPyTorchSnapshot(
                                    reference_prefix + "_MTP" +
                                    std::to_string(context.reference_depth) +
                                    "_MOE_ROUTING_INDICES");
                            ASSERT_NE(actual_indices, nullptr);
                            ASSERT_EQ(actual_indices_size, actual_size);
                            ASSERT_EQ(reference_indices.size(), actual_size);
                            const StageComparisonResult routing_result =
                                compareRoutingWeights(
                                    actual,
                                    reference,
                                    actual_indices,
                                    reference_indices,
                                    actual_size,
                                    sidecar_moe.top_k,
                                    sidecar_moe.num_experts,
                                    std::string(stage));
                            cosine = routing_result.cosine_similarity;
                            max_abs_diff = routing_result.max_abs_diff;
                            routing_overlap =
                                routing_result.routing_overlap;
                            passed = finite && routing_result.passed;
                            context_routing_weights_equivalent = passed;
                        }
                        else
                        {
                            const float numerical_threshold =
                                stage == "MOE_EXPERT_OUTPUT" &&
                                        !routing_indices_exact
                                    ? config_.cosine_threshold
                                    : config_.decode_cosine_threshold;
                            passed = passed &&
                                     cosine >= numerical_threshold;
                            if (stage == "MOE_EXPERT_OUTPUT")
                            {
                                context_routed_expert_output_equivalent =
                                    passed;
                            }
                        }
                        if (stage == "LM_HEAD")
                        {
                            kl = computeKLDivergence(
                                actual,
                                expected,
                                actual_size,
                                static_cast<size_t>(orch_runner_->vocabSize()));
                            passed = passed && kl < config_.kl_threshold;
                            const float reference_top1_in_production_top3 =
                                pytorchTop1InLlaminarTopK(
                                    actual,
                                    expected,
                                    actual_size,
                                    actual_size,
                                    3);
                            const float production_top1_in_reference_top3 =
                                pytorchTop1InLlaminarTopK(
                                    expected,
                                    actual,
                                    actual_size,
                                    actual_size,
                                    3);
                            passed = passed &&
                                     reference_top1_in_production_top3 >= 1.0f &&
                                     production_top1_in_reference_top3 >= 1.0f;
                            context_lm_head_passed = passed;
                        }

                        context_finite = context_finite && finite;
                        if (parityStageContributesToLayerCosine(
                                stage,
                                routing_indices_exact))
                        {
                            context_numerical_cosine_sum += cosine;
                            ++context_numerical_stage_count;
                        }

                        snapshot_csv
                            << call << ',' << reference_step << ','
                            << context.reference_depth << ','
                            << production_key << ',' << reference_key << ','
                            << actual_size << ',' << cosine << ','
                            << max_abs_diff << ',' << kl << ','
                            << (exact_indices ? 1 : 0) << ','
                            << routing_overlap << ','
                            << (routing_top1_match ? 1 : 0) << ','
                            << (finite ? 1 : 0) << ','
                            << (passed ? 1 : 0) << '\n';
                        ++compared_stages;
                        if (context.reference_depth > 0)
                            ++compared_recursive_stages;
                    }

                    if (!hf_branch_compatible)
                    {
                        /*
                         * A residency epoch can move the independent Dynamic
                         * request onto a different quantized-logit branch. The
                         * token CSV retains that fact, but no HF tensor from
                         * the canonical branch is an oracle for this context.
                         */
                        continue;
                    }

                    /*
                     * Apply the same route-aware numerical contract as
                     * runDecodeParity(): every checkpoint remains in the CSV,
                     * routing uses dedicated discrete/sparse metrics, raw
                     * expert sums from different legal boundary routes do not
                     * enter an elementwise cosine, and the remaining semantic
                     * tensors form one sidecar-layer aggregate. The LM-head
                     * still independently proves cosine, KL, and symmetric
                     * top-3 containment, so aggregation cannot hide a wrong
                     * token distribution.
                     */
                    ASSERT_GT(context_numerical_stage_count, 0u);
                    const double context_numerical_cosine =
                        context_numerical_cosine_sum /
                        static_cast<double>(
                            context_numerical_stage_count);
                    EXPECT_TRUE(context_finite)
                        << "Recursive MTP context published a non-finite "
                           "checkpoint at reference depth "
                        << context.reference_depth << "\nCSV: "
                        << snapshot_csv_path;
                    EXPECT_TRUE(context_routing_top1_match)
                        << "Recursive MTP context changed its highest-weight "
                           "expert at reference depth "
                        << context.reference_depth << "\nCSV: "
                        << snapshot_csv_path;
                    const float minimum_boundary_overlap =
                        sidecar_moe.top_k > 0
                            ? 1.0f -
                                  1.0f /
                                      static_cast<float>(sidecar_moe.top_k)
                            : 1.0f;
                    const bool routed_contribution_equivalent =
                        context_routing_weights_equivalent ||
                        (context_routing_top1_match &&
                         context_routing_overlap >=
                             minimum_boundary_overlap &&
                         context_routed_expert_output_equivalent);
                    EXPECT_TRUE(routed_contribution_equivalent)
                        << "Recursive MTP context changed both sparse routed "
                           "mass and the resulting expert value at reference depth "
                        << context.reference_depth << "\nCSV: "
                        << snapshot_csv_path;
                    EXPECT_GE(
                        context_numerical_cosine,
                        static_cast<double>(
                            config_.decode_cosine_threshold))
                        << "Recursive MTP context numerical aggregate failed "
                           "at reference depth "
                        << context.reference_depth << "\nCSV: "
                        << snapshot_csv_path;
                    EXPECT_TRUE(context_lm_head_passed)
                        << "Recursive MTP context LM-head/KL/top-k proof "
                           "failed at reference depth "
                        << context.reference_depth << "\nCSV: "
                        << snapshot_csv_path;
                }

                /*
                 * Compare grouped verifier row zero directly with the canonical
                 * HF main-model row while the production-visible prefix names
                 * that reference branch. This is the authoritative numerical
                 * oracle when a Dynamic/LLEP publication makes the separately
                 * executed serial request a different residency experiment.
                 */
                if (hf_branch_compatible && executed_transaction_count == 1u)
                {
                    const size_t verifier_rows = static_cast<size_t>(
                        activeMTPPhysicalVerifierRows());
                    const auto gdn = getGDNHeadConfig();
                    const auto moe = getMoEConfig();
                    for (const auto &key : activeSnapshotKeys())
                    {
                        if (!is_main_verifier_diagnostic_key(key) ||
                            key == "LM_HEAD_ROWS_SELECT")
                        {
                            continue;
                        }

                        size_t grouped_elements = 0;
                        const float *const grouped =
                            activeSnapshot(key, grouped_elements);
                        if (!grouped || grouped_elements == 0u ||
                            grouped_elements % verifier_rows != 0u)
                        {
                            continue;
                        }
                        const size_t row_elements =
                            grouped_elements / verifier_rows;
                        const std::string reference_key =
                            "decode_step" + std::to_string(reference_step) +
                            "_" + key;
                        std::vector<float> reference =
                            loadPyTorchSnapshot(reference_key);
                        if (reference.empty() ||
                            reference.size() < row_elements ||
                            reference.size() % row_elements != 0u)
                        {
                            continue;
                        }
                        if (reference.size() > row_elements)
                        {
                            reference.erase(
                                reference.begin(),
                                reference.end() -
                                    static_cast<ptrdiff_t>(row_elements));
                        }

                        std::string stage = key;
                        if (key.rfind("layer", 0) == 0)
                        {
                            const size_t delimiter = key.find('_');
                            if (delimiter != std::string::npos)
                                stage = key.substr(delimiter + 1u);
                        }
                        const std::vector<float> permuted =
                            applyGDNHeadPermutation(
                                grouped, row_elements, stage, gdn);
                        const float *const actual = permuted.empty()
                                                        ? grouped
                                                        : permuted.data();

                        StageComparisonResult result;
                        if (stage == "MOE_ROUTING_INDICES")
                        {
                            result = compareRoutingIndices(
                                actual,
                                reference,
                                row_elements,
                                moe.top_k,
                                stage);
                        }
                        else if (stage == "MOE_ROUTING_WEIGHTS")
                        {
                            const std::string index_key =
                                key.substr(
                                    0,
                                    key.size() -
                                        std::string("MOE_ROUTING_WEIGHTS").size()) +
                                "MOE_ROUTING_INDICES";
                            size_t grouped_index_elements = 0;
                            const float *const grouped_indices =
                                activeSnapshot(
                                    index_key,
                                    grouped_index_elements);
                            std::vector<float> reference_indices =
                                loadPyTorchSnapshot(
                                    "decode_step" +
                                    std::to_string(reference_step) + "_" +
                                    index_key);
                            if (reference_indices.size() > row_elements &&
                                reference_indices.size() % row_elements == 0u)
                            {
                                reference_indices.erase(
                                    reference_indices.begin(),
                                    reference_indices.end() -
                                        static_cast<ptrdiff_t>(row_elements));
                            }
                            if (grouped_indices &&
                                grouped_index_elements ==
                                    verifier_rows * row_elements &&
                                reference_indices.size() == row_elements)
                            {
                                result = compareRoutingWeights(
                                    actual,
                                    reference,
                                    grouped_indices,
                                    reference_indices,
                                    row_elements,
                                    moe.top_k,
                                    moe.num_experts,
                                    stage);
                            }
                            else
                            {
                                result = compareTensors(
                                    actual, reference, row_elements, stage);
                            }
                        }
                        else
                        {
                            result = compareTensors(
                                actual, reference, row_elements, stage);
                        }

                        bool finite = true;
                        bool exact_indices = true;
                        for (size_t index = 0; index < row_elements; ++index)
                        {
                            finite = finite && std::isfinite(actual[index]) &&
                                     std::isfinite(reference[index]);
                            exact_indices = exact_indices &&
                                            actual[index] == reference[index];
                        }
                        float kl = 0.0f;
                        bool passed = finite && result.passed;
                        if (stage == "MOE_ROUTING_INDICES")
                        {
                            const float minimum_overlap =
                                row_elements > 0u
                                    ? 1.0f -
                                          1.0f /
                                              static_cast<float>(row_elements)
                                    : 1.0f;
                            passed = finite &&
                                     result.routing_top1_match >= 1.0f &&
                                     result.routing_overlap >= minimum_overlap;
                        }
                        if (key == "LM_HEAD")
                        {
                            ++compared_main_lm_heads;
                            kl = computeKLDivergence(
                                actual,
                                reference.data(),
                                row_elements,
                                static_cast<size_t>(orch_runner_->vocabSize()));
                            const float reference_top1_in_production_top3 =
                                pytorchTop1InLlaminarTopK(
                                    actual,
                                    reference.data(),
                                    row_elements,
                                    row_elements,
                                    3);
                            const float production_top1_in_reference_top3 =
                                pytorchTop1InLlaminarTopK(
                                    reference.data(),
                                    actual,
                                    row_elements,
                                    row_elements,
                                    3);
                            passed = finite &&
                                     result.cosine_similarity >=
                                         config_.decode_cosine_threshold &&
                                     kl < config_.kl_threshold &&
                                     reference_top1_in_production_top3 >= 1.0f &&
                                     production_top1_in_reference_top3 >= 1.0f;
                            if (!passed)
                                ++failed_main_lm_heads;
                        }

                        /*
                         * Match runDecodeParity's established aggregation:
                         * routing has its own set/sparse-vector metrics, while
                         * numerical tensors contribute to the mean-cosine
                         * gate. Individual low-energy intermediates remain
                         * visible through their CSV `passed` field without
                         * letting an ill-conditioned cosine replace the
                         * end-to-end logit/KL/top-k proof.
                         */
                        if (!result.is_routing_stage)
                        {
                            grouped_main_cosine_sum +=
                                result.cosine_similarity;
                            ++grouped_main_cosine_count;
                        }

                        snapshot_csv
                            << call << ',' << reference_step << ",-2,"
                            << key << ',' << reference_key << ','
                            << row_elements << ','
                            << result.cosine_similarity << ','
                            << result.max_abs_diff << ',' << kl << ','
                            << (exact_indices ? 1 : 0) << ','
                            << (result.is_routing_stage
                                    ? result.routing_overlap
                                    : 1.0f)
                            << ','
                            << (result.is_routing_stage
                                    ? result.routing_top1_match
                                    : 1.0f)
                            << ',' << (finite ? 1 : 0) << ','
                            << (passed ? 1 : 0) << '\n';
                        ++compared_main_stages;
                    }
                }

                /*
                 * The first transaction of a two-token public boundary has an
                 * unambiguous grouped-verifier row-zero oracle: the serial
                 * request after consuming the same first visible token. Copy
                 * every compact M-row checkpoint against that serial M=1 row.
                 * This turns a wrong correction token into an earliest-stage
                 * CSV diagnosis instead of leaving only the final token ID.
                 */
                /*
                 * The verifier row is identified by the device-owned live
                 * condition position, not by the number of tokens returned by
                 * this public call. An accepted draft can make output_begin
                 * advance farther than the next verifier condition; mapping by
                 * response count then compares different autoregressive rows.
                 */
                const size_t serial_snapshot_index =
                    static_cast<size_t>(reference_step + 1);
                if (serial_epoch_compatible &&
                    executed_transaction_count == 1u &&
                    serial_snapshot_index <
                        serial_oracles_by_output_count.size())
                {
                    const auto &serial_snapshots =
                        serial_oracles_by_output_count[serial_snapshot_index]
                            .snapshots;
                    const size_t verifier_rows = static_cast<size_t>(
                        activeMTPPhysicalVerifierRows());
                    for (const auto &[key, serial_values] : serial_snapshots)
                    {
                        size_t grouped_elements = 0;
                        const float *const grouped =
                            activeSnapshot(key, grouped_elements);
                        if (!grouped || serial_values.empty() ||
                            grouped_elements !=
                                verifier_rows * serial_values.size())
                        {
                            continue;
                        }

                        bool finite = true;
                        bool exact_indices = true;
                        double max_abs_diff = 0.0;
                        for (size_t index = 0;
                             index < serial_values.size();
                             ++index)
                        {
                            finite = finite &&
                                     std::isfinite(grouped[index]) &&
                                     std::isfinite(serial_values[index]);
                            max_abs_diff = std::max(
                                max_abs_diff,
                                std::abs(
                                    static_cast<double>(grouped[index]) -
                                    static_cast<double>(serial_values[index])));
                            exact_indices = exact_indices &&
                                            grouped[index] ==
                                                serial_values[index];
                        }
                        const float cosine = computeCosineSimilarity(
                            grouped,
                            serial_values.data(),
                            serial_values.size());
                        const bool routing_indices =
                            std::string_view(key).ends_with(
                                "MOE_ROUTING_INDICES");
                        float kl = 0.0f;
                        bool passed = finite &&
                                      (routing_indices
                                           ? exact_indices
                                           : cosine >=
                                                 config_.decode_cosine_threshold);
                        if (key == "LM_HEAD")
                        {
                            kl = computeKLDivergence(
                                grouped,
                                serial_values.data(),
                                serial_values.size(),
                                static_cast<size_t>(orch_runner_->vocabSize()));
                            passed = passed && kl < config_.kl_threshold;
                        }
                        snapshot_csv
                            << call << ',' << reference_step << ",-1,"
                            << key << ",serial_output" << serial_snapshot_index
                            << '_' << key << ',' << serial_values.size() << ','
                            << cosine << ',' << max_abs_diff << ',' << kl << ','
                            << (exact_indices ? 1 : 0) << ",1,1,"
                            << (finite ? 1 : 0) << ','
                            << (passed ? 1 : 0) << '\n';
                        EXPECT_TRUE(passed)
                            << "Grouped verifier row zero diverged from serial "
                               "decode at "
                            << key << " cosine=" << cosine
                            << " max_abs_diff=" << max_abs_diff
                            << " kl=" << kl << "\nCSV: "
                            << snapshot_csv_path;
                    }
                }
            }

            const size_t output_end = output_begin + step.tokens.size();
            const std::vector<int32_t> serial_expected(
                serial_tokens.begin() + output_begin,
                serial_tokens.begin() + output_end);
            const std::vector<int32_t> hf_expected(
                expected_tokens.begin() + output_begin,
                expected_tokens.begin() + output_end);
            token_csv
                << call << ',' << reference_step << ',' << selected_depth
                << ',' << join_tokens(step.tokens)
                << ',' << join_tokens(serial_expected)
                << ',' << join_tokens(hf_expected)
                << ',' << (hf_branch_compatible ? 1 : 0) << ','
                << (serial_epoch_compatible ? 1 : 0) << ','
                << serial_movement_epoch << ','
                << grouped_movement_epoch_begin << ','
                << grouped_movement_epoch_end << ','
                << production_mtp0_top1 << ',' << hf_mtp0_top1 << ','
                << (recursive_branch_compatible ? 1 : 0) << ','
                << after.mtp_observed_verifier_transaction_count << ','
                << after.mtp_observed_verifier_draft_depth << ','
                << join_tokens(after.mtp_observed_verifier_draft_tokens)
                << ','
                << after.mtp_draft_steps << ',' << after.mtp_verifier_runs
                << ',' << after.mtp_accepted_tokens << ','
                << after.mtp_rejected_tokens << ','
                << after.mtp_transaction_commits << ','
                << after.mtp_transaction_rollbacks << ','
                << after.mtp_transaction_validation_failures << ','
                << after.current_position << '\n';
            emitted.insert(emitted.end(), step.tokens.begin(), step.tokens.end());
            ++call;
        }
        orch_runner_->setDecodeStepTokenBudget(0);

        if (usesDynamicMTPDepth())
        {
            /*
             * The two-token boundaries above deliberately isolate one
             * verifier identity for checkpoint diagnosis. At the configured
             * maximum depth of fifteen, those requests are budget-limited and
             * must not contaminate the adaptive controller's economy window.
             * Submit one ordinary full-width serving request so the
             * device-owned controller sees a complete real-weight
             * transaction and can make an evidence-backed depth decision.
             */
            activeClearSnapshots();
            activeClearCache();
            ASSERT_TRUE(orch_runner_->prefill(config_.token_ids))
                << orch_runner_->lastError();
            const auto policy_before = activePrefixStateProbe();
            const uint64_t policy_movement_epoch_begin =
                orch_runner_->moeRuntimeMovementEpoch();
            const int policy_response_budget = activeMTPDraftDepth() + 1;
            orch_runner_->setDecodeStepTokenBudget(policy_response_budget);
            const GenerationResult policy_step = orch_runner_->decodeStep();
            orch_runner_->setDecodeStepTokenBudget(0);
            const uint64_t policy_movement_epoch_end =
                orch_runner_->moeRuntimeMovementEpoch();
            ASSERT_TRUE(policy_step.success()) << policy_step.error;
            ASSERT_EQ(
                policy_step.tokens.size(),
                static_cast<size_t>(policy_response_budget))
                << "The dynamic-depth proof did not retire its admitted "
                   "full-width response budget";
            const auto policy_after = activePrefixStateProbe();
            EXPECT_GT(
                policy_after.mtp_depth_policy_windows,
                policy_before.mtp_depth_policy_windows)
                << "A non-budget-limited real-model verifier transaction did "
                   "not evaluate the device-owned depth policy";
            const auto promotions =
                policy_after.mtp_depth_policy_promotions -
                policy_before.mtp_depth_policy_promotions;
            const auto demotions =
                policy_after.mtp_depth_policy_demotions -
                policy_before.mtp_depth_policy_demotions;
            const auto updates =
                policy_after.mtp_depth_policy_updates -
                policy_before.mtp_depth_policy_updates;
            EXPECT_EQ(updates, promotions + demotions)
                << "The device-owned depth policy did not account for its "
                   "evaluated decision exactly once";
            if (demotions > 0u)
            {
                EXPECT_LT(
                    policy_after.mtp_current_depth,
                    policy_before.mtp_current_depth)
                    << "A recorded depth demotion did not reduce the selected width";
            }
            else if (promotions > 0u)
            {
                EXPECT_GT(
                    policy_after.mtp_current_depth,
                    policy_before.mtp_current_depth)
                    << "A recorded depth promotion did not increase the selected width";
            }
            else
            {
                EXPECT_EQ(
                    policy_after.mtp_current_depth,
                    policy_before.mtp_current_depth)
                    << "An evidence-backed hold changed depth without recording an update";
            }

            if (!isDynamicResidencyProductionTest() &&
                policy_movement_epoch_begin == policy_movement_epoch_end)
            {
                ASSERT_GE(serial_tokens.size(), policy_step.tokens.size());
                for (size_t index = 0; index < policy_step.tokens.size(); ++index)
                {
                    EXPECT_EQ(policy_step.tokens[index], serial_tokens[index])
                        << "Dynamic-depth policy proof diverged from serial "
                           "decode at output "
                        << index;
                }
            }
        }

        const uint64_t final_movement_epoch =
            orch_runner_->moeRuntimeMovementEpoch();
        if (!isDynamicResidencyProductionTest())
        {
            EXPECT_EQ(final_movement_epoch, mtp_certification_movement_epoch)
                << "Static MTP certification crossed a residency epoch";
            EXPECT_GT(serial_epoch_compatible_calls, 0)
                << "Static MTP produced no same-epoch serial equivalence proof";
        }

        const auto final_state = activePrefixStateProbe();
        EXPECT_GT(speculative_calls, 0)
            << "No grouped MTP transaction executed";
        EXPECT_GT(compared_stages, 0)
            << "No live MTP checkpoint was compared";
        if (activeMTPDraftDepth() > 1)
        {
            EXPECT_GT(
                compared_recursive_stages + deferred_recursive_contexts,
                0)
                << "No recursive MTP checkpoint had an observed proposal "
                   "identity or a deferred exact HF branch proof";
        }
        EXPECT_GT(compared_main_stages, 0)
            << "No grouped main-model checkpoint was compared with Hugging Face";
        ASSERT_GT(grouped_main_cosine_count, 0u)
            << "Grouped main-model comparison produced no numerical rows";
        EXPECT_GE(
            grouped_main_cosine_sum /
                static_cast<double>(grouped_main_cosine_count),
            static_cast<double>(config_.decode_cosine_threshold))
            << "Grouped main-model aggregate cosine failed against Hugging Face; CSV: "
            << snapshot_csv_path;
        EXPECT_GT(compared_main_lm_heads, 0)
            << "Grouped main-model comparison omitted LM_HEAD";
        EXPECT_EQ(failed_main_lm_heads, 0)
            << "Grouped main-model LM_HEAD failed cosine/KL/mutual-top3 parity; CSV: "
            << snapshot_csv_path;
        EXPECT_GT(
            final_state.mtp_accepted_tokens,
            initial_state.mtp_accepted_tokens)
            << "The real 122B MTP request accepted no draft tokens";
        EXPECT_GT(
            final_state.mtp_transaction_commits,
            initial_state.mtp_transaction_commits);
        EXPECT_EQ(final_state.mtp_transaction_rollbacks, 0u);
        EXPECT_EQ(final_state.mtp_transaction_validation_failures, 0u);
        EXPECT_EQ(final_state.mtp_max_depth, activeMTPDraftDepth())
            << "MTP runtime capacity did not match the cell's admitted maximum";
        if (usesDynamicMTPDepth())
        {
            EXPECT_GT(final_state.mtp_depth_policy_windows, 0u)
                << "Dynamic-depth policy observed no completed verifier window";
            EXPECT_EQ(final_state.mtp_min_depth, 1);
        }
        token_csv.flush();
        snapshot_csv.flush();
        EXPECT_TRUE(token_csv.good());
        EXPECT_TRUE(snapshot_csv.good());
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
     * @brief Persist participant-local sparse endpoint evidence beside parity CSVs.
     *
     * PerfStats is process-local by design, so a root-only export cannot show
     * which CUDA/ROCm follower accepted each routed-expert packet.  Every MPI
     * instance writes a rank-qualified file after the production worker loop
     * has closed.  The records retain layer, logical step, expert ids, route
     * count, participant, tier, and service timing, which makes a missing or
     * corrupt participant contribution diagnosable without changing the live
     * graph or downloading any additional device data.
     */
    void writeSparseEndpointEvidenceCsv() const
    {
        if (!PerfStatsCollector::isDomainEnabled("moe_overlay_endpoint"))
            return;

        const int rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
        const auto path = ensureResultsDir() /
                          ("sparse_endpoint_rank_" +
                           std::to_string(rank) + ".csv");
        std::ofstream output(path, std::ios::trunc);
        ASSERT_TRUE(output.is_open()) << path;
        output << PerfStatsCollector::csvString({"moe_overlay_endpoint"});
        output.flush();
        EXPECT_TRUE(output.good()) << path;
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
        writeSparseEndpointEvidenceCsv();
        assertParticipantCompactBufferArenaEvidence();
        assertMappedParticipantGraphEvidence();
        assertActiveTierRouteEvidence();
        assertSparseTransportPerfStatsEvidence();
        assertResidencyMovementEvidence();
        assertRequestMovementPolicyEvidence();
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
                    ADD_FAILURE() << "Production parity model not found at "
                                  << activeModelPath();
            }
            return;
        }

        /*
         * OrchestrationRunner initialization is already a rank-consensual
         * lifecycle: every setup phase publishes one result through the
         * production MPI context before either rank may advance.  A second
         * test-owned all-reduce here is not additional validation.  If one
         * graph builder rejects a phase, that reduction can match the peer's
         * final production-phase reduction and leave the peer waiting in this
         * later call while the failing rank enters TearDown.  Trust the typed
         * production result and keep the test protocol at exactly one
         * rendezvous per lifecycle transition.
         */
        const bool setup_ok = setupPipeline();
        ASSERT_TRUE(setup_ok)
            << "Production pipeline setup failed on this rank or a peer rank";

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
            orch_runner_->setMPICoordinatedMode(false);
            assertEvidenceAfterWorkerShutdown();
            return;
        }

        struct WorkerShutdownGuard
        {
            IOrchestrationRunner *runner = nullptr;
            ~WorkerShutdownGuard()
            {
                if (runner)
                {
                    runner->shutdownMPIWorkers();
                    runner->setMPICoordinatedMode(false);
                }
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
            orch_runner_->setMPICoordinatedMode(false);
            worker_shutdown.runner = nullptr;
            assertEvidenceAfterWorkerShutdown();
            ADD_FAILURE()
                << "Real dynamic residency workload did not prove a cross-rank expert migration";
            return;
        }
        cacheCommittedPromotionEvidence();

        if (isDynamicResidencyProductionTest())
        {
            const bool convergence_timings_ready =
                collectConvergedInferenceTimings();
            EXPECT_TRUE(convergence_timings_ready)
                << "Could not collect epoch-stable observed convergence timings";
            if (convergence_timings_ready)
                assertAndWriteObservedConvergenceSpeedup();
            cacheCommittedPromotionEvidence();
            writeCommittedMovementEvidenceCsv();

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
            if (isSegmentedPrefillProductionTest())
                assertSegmentedPrefillCheckpointCoverage();
            assertProductionParitySnapshotInfrastructure();
            cacheDeviceRouteAssignmentEvidence();
            writeExpertOwnerTopologyBaselineCsv();
            writePrefillRoutedExpertDiagnosticCsv();
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
        {
            assertDecodeParity(decode);
            assertParityExecutionExercisesPromotedExpert();
        }

        try
        {
            runMTPHuggingFaceCheckpointParity();
        }
        catch (const std::exception &e)
        {
            abortAfterRootThrow("MTP Hugging Face checkpoint parity", e.what());
        }
        catch (...)
        {
            abortAfterRootThrow(
                "MTP Hugging Face checkpoint parity",
                "unknown exception");
        }

        /*
         * End the production worker protocol before entering the evidence
         * allreduce.  This keeps the command loop and the test-only collective
         * from competing for the same MPI messages while retaining the real
         * production setup and forward path above.
         */
        orch_runner_->shutdownMPIWorkers();
        orch_runner_->setMPICoordinatedMode(false);
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
    std::vector<PromotedExpert>
        promoted_experts_; ///< Promotion identities retained across parity collector resets.
    std::vector<PromotedExpertExecutionWitness>
        promoted_expert_execution_witnesses_; ///< Exact parity routes through promoted destinations.
};

#ifndef LLAMINAR_QWEN122_MATRIX_ONLY

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

#else

/**
 * @brief Register one exact production 122B policy/owner/depth matrix cell.
 *
 * Every expansion owns a fresh request runner but shares one process-resident
 * authenticated GGUF/reference corpus. The runner itself still constructs the
 * same two-rank, six-participant graph used by serving.
 */
#define LLAMINAR_QWEN122_PARITY_CELL(POLICY, OWNER, DEPTH)                  \
    TEST_F(                                                               \
        Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold,                       \
        ProductionParity_Qwen35_122B_CUDA2_ROCm4_##POLICY##_##OWNER##_##DEPTH) \
    {                                                                     \
        runGraphNativeProductionParityBody();                             \
    }

#define LLAMINAR_QWEN122_OWNER_DEPTHS(POLICY, OWNER) \
    LLAMINAR_QWEN122_PARITY_CELL(POLICY, OWNER, MTPDepth1) \
    LLAMINAR_QWEN122_PARITY_CELL(POLICY, OWNER, MTPDepth2) \
    LLAMINAR_QWEN122_PARITY_CELL(POLICY, OWNER, MTPDepth3) \
    LLAMINAR_QWEN122_PARITY_CELL(POLICY, OWNER, MTPDynamicDepth) \
    LLAMINAR_QWEN122_PARITY_CELL(POLICY, OWNER, MTPDepth15)

LLAMINAR_QWEN122_OWNER_DEPTHS(Static, Ordinal)
LLAMINAR_QWEN122_OWNER_DEPTHS(Static, Random)
LLAMINAR_QWEN122_OWNER_DEPTHS(Dynamic, Ordinal)
LLAMINAR_QWEN122_OWNER_DEPTHS(Dynamic, Random)

/* Current-batch LLEP is intentionally absent from this production campaign.
 * Its device-owned implementation is incomplete; registering a skipped or
 * compatibility cell would misrepresent support. Restore the two owner rows
 * only when the production controller and its economy proof are complete. */

#undef LLAMINAR_QWEN122_OWNER_DEPTHS
#undef LLAMINAR_QWEN122_PARITY_CELL

#endif

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
