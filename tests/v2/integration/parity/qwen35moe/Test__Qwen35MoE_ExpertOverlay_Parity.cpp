/**
 * @file Test__Qwen35MoE_ExpertOverlay_Parity.cpp
 * @brief Qwen3.5 MoE parity tests for same-layer expert overlay plans.
 *
 * This is a V2 parity-harness test: it loads the real Qwen3.5 MoE GGUF,
 * regenerates/loads PyTorch snapshots through Qwen35MoEParityTestBase, injects
 * a planned MoEExpertParallelPlan into the production graph config, and compares
 * Llaminar prefill/decode snapshots against the PyTorch reference.
 *
 * Bridge Phase 5A audit contract: OverlayPlanTopology_* tests only prove that the
 * planned same-layer tier assignments are non-empty and stable. PrefillParity_*
 * and DecodeParity_* are the real V2 inference parity bodies; while production
 * overlay gaps remain, they must report a clear GTest skip from
 * overlayRuntimeBlockers() instead of passing as topology-only coverage.
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>

#include "Qwen35MoEParityTestBase.h"
#include "backends/ComputeBackend.h"
#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"
#include "execution/compute_stages/stages/MoEExpertOverlayCPUFallbackStage.h"
#include "execution/factory/InferenceRunnerFactory.h"
#include "execution/moe/MoEExpertParallelPlanner.h"
#include "loaders/ExpertGemmRegistry.h"
#include "mocks/MockComputeStage.h"
#include "tensors/Tensors.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen35moe;

namespace
{
    constexpr const char *kModelPath = "/opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf";
    constexpr const char *kSnapshotDir = "pytorch_qwen35_moe_snapshots";
    constexpr const char *kRocmSharedHotDomain = "rocm_shared_hot";
    constexpr const char *kCudaSharedHotDomain = "cuda_shared_hot";
    constexpr const char *kRocmHotDomain = "rocm_hot";
    constexpr const char *kCpuColdDomain = "cpu_cold";

    enum class OverlayTopologyKind
    {
        RocmSharedHotCpuCold,
        CudaSharedHotRocmHotCpuCold,
    };

    struct OverlayParityCase
    {
        std::string name;
        TestConfig config;
        OverlayTopologyKind topology;
    };

    size_t gib(size_t value)
    {
        return value * 1024ULL * 1024ULL * 1024ULL;
    }

    ExpertComputeDomain rocmLocalTPDomain(const std::string &name)
    {
        ExpertComputeDomain domain;
        domain.name = name;
        domain.kind = ExpertDomainKind::LocalTP;
        domain.backend = CollectiveBackendType::RCCL;
        domain.participants = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)};
        domain.owner_rank = 0;
        domain.compute_kind = ExpertDomainComputeKind::TensorParallelExperts;
        return domain;
    }

    ExpertComputeDomain cudaSharedHotDomain()
    {
        ExpertComputeDomain domain;
        domain.name = kCudaSharedHotDomain;
        domain.kind = ExpertDomainKind::SingleDevice;
        domain.backend = CollectiveBackendType::NCCL;
        domain.participants = {GlobalDeviceAddress::cuda(0)};
        domain.world_ranks = {0};
        domain.owner_rank = 0;
        domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
        return domain;
    }

    ExpertComputeDomain cpuNodeLocalTPDomain()
    {
        ExpertComputeDomain domain;
        domain.name = kCpuColdDomain;
        domain.kind = ExpertDomainKind::NodeLocalTP;
        domain.backend = CollectiveBackendType::UPI;
        domain.participants = {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)};
        domain.world_ranks = {0, 1};
        domain.owner_rank = 0;
        domain.compute_kind = ExpertDomainComputeKind::TensorParallelExperts;
        return domain;
    }

    ExpertRoutedTier tier(
        const std::string &name,
        const std::string &domain,
        int priority,
        int max_experts_per_layer,
        size_t memory_budget_bytes,
        bool fallback = false)
    {
        ExpertRoutedTier result;
        result.name = name;
        result.domain = domain;
        result.priority = priority;
        result.max_experts_per_layer = max_experts_per_layer;
        result.memory_budget_bytes = memory_budget_bytes;
        result.fallback = fallback;
        return result;
    }

    MoEExpertModelMetadata metadataFromModel(const ModelContext &ctx)
    {
        const auto &loader = ctx.concreteLoader();
        const std::string &arch = ctx.architecture();

        MoEExpertModelMetadata metadata;
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

    std::string validationErrors(const MoEExpertParallelValidationResult &validation)
    {
        std::ostringstream message;
        for (const auto &error : validation.errors)
            message << "\n - " << error;
        return message.str();
    }

    MoEExpertParallelPlan requestedPlan(OverlayTopologyKind topology, const MoEExpertModelMetadata &metadata)
    {
        const int small_hot_capacity = std::max(1, metadata.num_experts / 8);
        const int medium_hot_capacity = std::max(1, metadata.num_experts / 4);

        MoEExpertParallelPlan plan;
        plan.enabled = true;
        plan.execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
        plan.residency_policy = ExpertResidencyPolicy::StaticById;

        switch (topology)
        {
        case OverlayTopologyKind::RocmSharedHotCpuCold:
            plan.continuation_domain = kRocmSharedHotDomain;
            plan.shared_expert_domain = kRocmSharedHotDomain;
            plan.domains = {
                rocmLocalTPDomain(kRocmSharedHotDomain),
                cpuNodeLocalTPDomain(),
            };
            plan.routed_tiers = {
                tier("shared_hot", kRocmSharedHotDomain, 0, medium_hot_capacity, gib(4)),
                tier("cold", kCpuColdDomain, 1, 0, 0, true),
            };
            break;
        case OverlayTopologyKind::CudaSharedHotRocmHotCpuCold:
            plan.continuation_domain = kCudaSharedHotDomain;
            plan.shared_expert_domain = kCudaSharedHotDomain;
            plan.domains = {
                cudaSharedHotDomain(),
                rocmLocalTPDomain(kRocmHotDomain),
                cpuNodeLocalTPDomain(),
            };
            plan.routed_tiers = {
                tier("shared_hottest", kCudaSharedHotDomain, 0, small_hot_capacity, gib(2)),
                tier("hot", kRocmHotDomain, 1, medium_hot_capacity, gib(4)),
                tier("cold", kCpuColdDomain, 2, 0, 0, true),
            };
            break;
        }

        return plan;
    }

    std::shared_ptr<MoEExpertParallelPlan> makeOverlayPlan(
        OverlayTopologyKind topology,
        const ModelContext &ctx)
    {
        const auto metadata = metadataFromModel(ctx);
        auto planned = MoEExpertParallelPlanner::plan(
                           requestedPlan(topology, metadata),
                           metadata)
                           .planned_plan;

        MoEExpertParallelValidationOptions options;
        options.layer_count = metadata.num_layers;
        options.routed_expert_count = metadata.num_experts;
        auto validation = validateMoEExpertParallelPlan(planned, options);
        if (!validation.ok())
        {
            throw std::invalid_argument("Invalid planned MoE expert overlay:" + validationErrors(validation));
        }

        return std::make_shared<MoEExpertParallelPlan>(std::move(planned));
    }

    TestConfig makeBaseConfig(const std::string &name)
    {
        TestConfig config;
        config.name = name;
        config.devices = {ParityDeviceType::CPU};
        config.parallelism = Parallelism::None;
        config.collective = Collective::None;
        config.thresholds = {
            .cosine_threshold = 0.90f,
            .decode_cosine_threshold = 0.80f,
            .early_layers_count = 6,
            .min_early_layers_passed = 5,
            .kl_threshold = 0.05f,
            .min_top1_accuracy = 0.80f,
            .min_top5_accuracy = 0.80f,
            .pytorch_top1_in_topk = 4,
        };
        config.mpi_ranks = 2;
        config.model_path = kModelPath;
        config.snapshot_dir = kSnapshotDir;
        config.activation_precision = ActivationPrecision::FP32;
        config.kv_cache_precision = KVCachePrecision::FP16;
        return config;
    }

    const std::vector<OverlayParityCase> kOverlayParityCases = {
        {
            .name = "ROCm2TP_SharedHot_CPU2NodeLocalTP_Cold",
            .config = makeBaseConfig("ExpertOverlay_ROCm2TP_SharedHot_CPU2NodeLocalTP_Cold"),
            .topology = OverlayTopologyKind::RocmSharedHotCpuCold,
        },
        {
            .name = "CUDA1_SharedHot_ROCm2TP_Hot_CPU2NodeLocalTP_Cold",
            .config = makeBaseConfig("ExpertOverlay_CUDA1_SharedHot_ROCm2TP_Hot_CPU2NodeLocalTP_Cold"),
            .topology = OverlayTopologyKind::CudaSharedHotRocmHotCpuCold,
        },
    };

    const OverlayParityCase &caseForCurrentTest()
    {
        const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        if (!test_info)
            return kOverlayParityCases.front();

        const std::string test_name = test_info->name();
        for (const auto &test_case : kOverlayParityCases)
        {
            if (test_name.find(test_case.name) != std::string::npos)
                return test_case;
        }

        return kOverlayParityCases.front();
    }

    std::vector<size_t> tierExpertCounts(const MoEExpertParallelPlan &plan)
    {
        std::vector<size_t> counts(plan.routed_tiers.size(), 0);
        for (const auto &placement : plan.placements)
        {
            for (int tier_index : placement.routed_expert_tier)
            {
                if (tier_index >= 0 && tier_index < static_cast<int>(counts.size()))
                    ++counts[static_cast<size_t>(tier_index)];
            }
        }
        return counts;
    }

    const ExpertComputeDomain *findDomain(
        const MoEExpertParallelPlan &plan,
        const std::string &domain_name)
    {
        auto it = std::find_if(plan.domains.begin(), plan.domains.end(),
                               [&domain_name](const ExpertComputeDomain &domain)
                               {
                                   return domain.name == domain_name;
                               });
        return it == plan.domains.end() ? nullptr : &(*it);
    }

    bool tierOwnsAnyExpert(
        const ExpertLayerPlacement &placement,
        int tier_index)
    {
        return std::any_of(placement.routed_expert_tier.begin(),
                           placement.routed_expert_tier.end(),
                           [tier_index](int mapped_tier)
                           {
                               return mapped_tier == tier_index;
                           });
    }

    const ExpertLayerPlacement *findLayerPlacement(
        const MoEExpertParallelPlan &plan,
        int layer_idx)
    {
        auto it = std::find_if(plan.placements.begin(), plan.placements.end(),
                               [layer_idx](const ExpertLayerPlacement &placement)
                               {
                                   return placement.layer == layer_idx;
                               });
        return it == plan.placements.end() ? nullptr : &(*it);
    }

    bool anyExpertEnabled(const std::vector<bool> &mask)
    {
        return std::any_of(mask.begin(), mask.end(), [](bool enabled)
                           { return enabled; });
    }

    std::string joinBlockers(const std::vector<std::string> &blockers)
    {
        std::ostringstream message;
        for (size_t i = 0; i < blockers.size(); ++i)
        {
            if (i != 0)
                message << "; ";
            message << blockers[i];
        }
        return message.str();
    }

    std::string overlayPlanOnlyRuntimeBlockers(const MoEExpertParallelPlan &plan)
    {
        (void)plan;
        return {};
    }

    // Bridge Phase 5A skip audit: every blocker reported here is an intentionally
    // remaining production capability gap. Keep these messages specific so a
    // CTest-pass/GTest-skip result cannot be mistaken for completed inference.
    std::string overlayRuntimeBlockers(
        const MoEExpertParallelPlan &plan,
        ModelContext &ctx)
    {
        std::vector<std::string> blockers;
        std::set<std::string> reported_missing_registry;
        auto weight_mgr = ctx.concreteWeightManager();
        if (!weight_mgr)
        {
            return "Bridge Phase 5B accelerator residency verification requires a concrete WeightManager to verify prepared expert residency";
        }

        const auto &registry = weight_mgr->expertGemmRegistry();
        for (const auto &placement : plan.placements)
        {
            for (size_t tier_index = 0; tier_index < plan.routed_tiers.size(); ++tier_index)
            {
                if (!tierOwnsAnyExpert(placement, static_cast<int>(tier_index)))
                    continue;

                const auto &tier = plan.routed_tiers[tier_index];
                const ExpertComputeDomain *domain = findDomain(plan, tier.domain);
                if (!domain || domain->participants.empty())
                {
                    blockers.push_back("Expert overlay tier '" + tier.name +
                                       "' references an unavailable compute domain");
                    continue;
                }

                struct ParticipantDevice
                {
                    DeviceId device = DeviceId::invalid();
                    int participant_index = -1;
                };

                std::vector<ParticipantDevice> participant_devices;
                for (size_t participant_index = 0; participant_index < domain->participants.size(); ++participant_index)
                {
                    const DeviceId participant_device = domain->participants[participant_index].toLocalDeviceId();
                    if (participant_device.is_gpu())
                    {
                        participant_devices.push_back(ParticipantDevice{
                            .device = participant_device,
                            .participant_index = static_cast<int>(participant_index),
                        });
                    }
                }
                if (participant_devices.empty())
                    continue;

                // Do not require direct ROCm P2P here. RCCL owns transport
                // selection and may legally stage through host memory when the
                // topology lacks peer access; P2P is a performance signal, not
                // a correctness precondition for Phase 8A parity.

                // Phase 5E expected skip: real inference can run only when
                // every active accelerator participant has domain-scoped
                // prepared gate/up/down engines for its planned routed experts.
                for (const auto &participant : participant_devices)
                {
                    bool missing_gpu_engine = false;
                    for (int expert = 0; expert < static_cast<int>(placement.routed_expert_tier.size()); ++expert)
                    {
                        if (placement.routed_expert_tier[static_cast<size_t>(expert)] != static_cast<int>(tier_index))
                            continue;

                        using Role = ExpertGemmRegistry::WeightRole;
                        if (!registry.getEngineForDomain(tier.domain, participant.device, placement.layer, expert, Role::GATE) ||
                            !registry.getEngineForDomain(tier.domain, participant.device, placement.layer, expert, Role::UP) ||
                            !registry.getEngineForDomain(tier.domain, participant.device, placement.layer, expert, Role::DOWN))
                        {
                            missing_gpu_engine = true;
                            break;
                        }
                    }

                    const std::string registry_key = tier.domain + ":" + participant.device.to_string() +
                                                     ":" + std::to_string(participant.participant_index);
                    if (missing_gpu_engine && reported_missing_registry.insert(registry_key).second)
                    {
                        blockers.push_back(
                            "Bridge Phase 5E accelerator prepared-engine blocker: expert overlay tier '" + tier.name +
                            "' lowers to domain '" + tier.domain + "' participant " +
                            std::to_string(participant.participant_index) + " on " + participant.device.to_string() +
                            " but the ExpertGemmRegistry does not contain all active gate/up/down expert engines for that tier");
                    }
                }
            }
        }

        const auto plan_only_blockers = overlayPlanOnlyRuntimeBlockers(plan);
        if (!plan_only_blockers.empty())
        {
            if (!blockers.empty())
                blockers.push_back(plan_only_blockers);
            else
                return plan_only_blockers;
        }

        return joinBlockers(blockers);
    }
} // namespace

class Qwen35MoEExpertOverlay
    : public Qwen35MoEConfigDrivenParityTest<Qwen35MoEExpertOverlay>
{
public:
    const TestConfig &getTestConfig() const { return caseForCurrentTest().config; }

protected:
    using Base = Qwen35MoEConfigDrivenParityTest<Qwen35MoEExpertOverlay>;

    void SetUp() override
    {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized)
        {
            // Harness precondition: these tests are discovered as MPI CTests, and
            // running outside MPI would not exercise the overlay audit surface.
            GTEST_SKIP() << "ExpertOverlay parity requires MPI initialization";
        }

        int rank = 0;
        int world_size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        if (world_size < cfg().mpi_ranks)
        {
            // Harness precondition: the CPU fallback tier is expressed as a
            // two-rank NodeLocalTP domain in this Phase 0 audit fixture.
            GTEST_SKIP() << "ExpertOverlay parity requires " << cfg().mpi_ranks
                         << " MPI ranks (got " << world_size << ")";
        }

        mpi_ctx_ = std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD);
        Base::SetUp();
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
        if (cfg().decode_steps > 0)
            config_.decode_steps = cfg().decode_steps;

        const auto metadata_path = std::filesystem::path(config_.snapshot_dir) / "metadata.txt";
        const bool metadata_missing = !std::filesystem::exists(metadata_path);
        const bool metadata_stale = !metadata_missing &&
                                    readSnapshotVersion(metadata_path) < kRequiredSnapshotVersion;
        const bool model_available = std::filesystem::exists(config_.model_path);
        const int local_needs_regen = (metadata_missing || metadata_stale) && model_available ? 1 : 0;
        int global_needs_regen = 0;
        MPI_Allreduce(&local_needs_regen, &global_needs_regen, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        int local_regen_failed = 0;
        if (global_needs_regen && isRank0())
        {
            LOG_INFO("[Qwen3.5 MoE ExpertOverlay] Regenerating snapshots once on rank 0");
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
            LOG_INFO("[Qwen3.5 MoE ExpertOverlay] Loaded " << config_.token_ids.size()
                                                           << " prefill token IDs from metadata");
        }
    }

    bool setupPipeline()
    {
        if (isRank0())
            DeviceManager::instance().initialize(-1);

        model_ctx_ = ModelContext::create(
            config_.model_path,
            nullptr,
            nullptr,
            nullptr,
            WeightDistributionStrategy::REPLICATED);
        if (!model_ctx_)
        {
            LOG_ERROR("[Qwen3.5 MoE ExpertOverlay] Failed to load model");
            return false;
        }

        configureModel(model_ctx_);

        try
        {
            overlay_plan_ = makeOverlayPlan(caseForCurrentTest().topology, *model_ctx_);
            setup_runtime_blocker_ = overlayPlanOnlyRuntimeBlockers(*overlay_plan_);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[Qwen3.5 MoE ExpertOverlay] " << e.what());
            return false;
        }

        if (!setup_runtime_blocker_.empty())
        {
            if (!isRank0())
                cpu_ctx_ = std::make_unique<llaminar2::testing::MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);
            return true;
        }

        if (!isRank0())
        {
            cpu_ctx_ = std::make_unique<llaminar2::testing::MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);
            return true;
        }

        InferenceRunnerConfig inf_config;
        inf_config.max_seq_len = 4096;
        inf_config.batch_size = 1;
        inf_config.force_graph = true;
        inf_config.activation_precision = cfg().activation_precision;
        inf_config.kv_cache_precision = cfg().kv_cache_precision;
        inf_config.moe_expert_parallel_plan = overlay_plan_;

        runner_ = createInferenceRunner(model_ctx_, nullptr, DeviceId::cpu(), inf_config);
        if (!runner_)
        {
            LOG_ERROR("[Qwen3.5 MoE ExpertOverlay] Failed to create inference runner");
            return false;
        }

        runner_->enableSnapshotCapture();
        return true;
    }

    bool isRootParityRank() const
    {
        return isRank0();
    }

    bool synchronizeRanksOk(bool local_ok) const
    {
        int ok = local_ok ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        return ok == 1;
    }

    std::string broadcastRootString(const std::string &root_value) const
    {
        std::string value = isRootParityRank() ? root_value : std::string();
        int length = static_cast<int>(value.size());
        MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (length < 0)
            return {};
        value.resize(static_cast<size_t>(length));
        if (length > 0)
            MPI_Bcast(value.data(), length, MPI_CHAR, 0, MPI_COMM_WORLD);
        return value;
    }

    bool broadcastRootFlag(bool root_value) const
    {
        int flag = isRootParityRank() && root_value ? 1 : 0;
        MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
        return flag != 0;
    }

    std::string synchronizedRuntimeBlocker()
    {
        std::string blocker;
        if (isRootParityRank())
        {
            if (!setup_runtime_blocker_.empty())
                blocker = setup_runtime_blocker_;
            else if (!overlay_plan_ || !model_ctx_)
                blocker = "ExpertOverlay parity setup did not produce a model context and overlay plan";
            else
                blocker = overlayRuntimeBlockers(*overlay_plan_, *model_ctx_);
        }
        return broadcastRootString(blocker);
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

    int continuationRootWorldRank() const
    {
        if (!overlay_plan_)
            return 0;

        const ExpertComputeDomain *continuation = findDomain(*overlay_plan_, overlay_plan_->continuation_domain);
        if (!continuation)
            return 0;
        if (!continuation->world_ranks.empty())
            return continuation->world_ranks.front();
        return std::max(0, continuation->owner_rank);
    }

    int cpuFallbackTierIndex() const
    {
        if (!overlay_plan_)
            return -1;

        for (size_t tier_index = 0; tier_index < overlay_plan_->routed_tiers.size(); ++tier_index)
        {
            const auto &tier = overlay_plan_->routed_tiers[tier_index];
            if (tier.domain == kCpuColdDomain && tier.fallback)
                return static_cast<int>(tier_index);
        }
        return -1;
    }

    std::vector<bool> fallbackExpertMaskForLayer(int layer_idx, int tier_index) const
    {
        if (!overlay_plan_)
            return {};

        const ExpertLayerPlacement *placement = findLayerPlacement(*overlay_plan_, layer_idx);
        if (!placement)
            return {};

        std::vector<bool> mask;
        mask.reserve(placement->routed_expert_tier.size());
        for (int mapped_tier : placement->routed_expert_tier)
            mask.push_back(mapped_tier == tier_index);
        return mask;
    }

    std::shared_ptr<TensorBase> loadFallbackWeight(int layer_idx, const char *suffix)
    {
        auto weight_mgr = model_ctx_ ? model_ctx_->concreteWeightManager() : nullptr;
        if (!weight_mgr)
            return nullptr;

        const std::string name = "blk." + std::to_string(layer_idx) + "." + suffix;
        auto weight = weight_mgr->getWeightForDevice(name, DeviceId::cpu(), layer_idx);
        if (!weight)
            LOG_ERROR("[Qwen3.5 MoE ExpertOverlay] Missing CPU fallback weight " << name);
        return weight;
    }

    void clearFallbackWeightCache()
    {
        auto weight_mgr = model_ctx_ ? model_ctx_->concreteWeightManager() : nullptr;
        if (weight_mgr)
            weight_mgr->clearCache();
    }

    bool executeFallbackParticipantPass(int seq_len)
    {
        // Temporary Phase 0 orchestration scaffolding. Non-root ranks manually
        // run CPU fallback participant work until the composite overlay runner
        // described in docs/v2/MOE_EXPERT_OVERLAY_ORCHESTRATION_REFACTOR_PLAN.md
        // replaces this parity-only helper.
        if (seq_len <= 0)
            return true;
        if (!overlay_plan_ || !model_ctx_)
        {
            LOG_ERROR("[Qwen3.5 MoE ExpertOverlay] Fallback participant missing overlay plan/model context");
            return false;
        }

        const ExpertComputeDomain *cpu_domain = findDomain(*overlay_plan_, kCpuColdDomain);
        const int tier_index = cpuFallbackTierIndex();
        if (!cpu_domain || tier_index < 0)
        {
            LOG_ERROR("[Qwen3.5 MoE ExpertOverlay] CPU fallback domain/tier not found");
            return false;
        }

        const auto &loader = model_ctx_->concreteLoader();
        const std::string &arch = model_ctx_->architecture();
        const int d_model = model_ctx_->embeddingLength();
        const int num_experts = loader.getInt(arch + ".expert_count", 0);
        const int top_k = loader.getInt(arch + ".expert_used_count", 0);
        int expert_intermediate = loader.getInt(arch + ".expert_feed_forward_length", 0);
        if (expert_intermediate == 0)
            expert_intermediate = model_ctx_->feedForwardLength();

        if (d_model <= 0 || num_experts <= 0 || top_k <= 0 || expert_intermediate <= 0)
        {
            LOG_ERROR("[Qwen3.5 MoE ExpertOverlay] Invalid CPU fallback metadata: d_model=" << d_model
                                                                                            << " num_experts=" << num_experts
                                                                                            << " top_k=" << top_k
                                                                                            << " expert_intermediate=" << expert_intermediate);
            return false;
        }

        if (!cpu_ctx_)
            cpu_ctx_ = std::make_unique<llaminar2::testing::MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);

        for (int layer_idx = 0; layer_idx < model_ctx_->totalBlockCount(); ++layer_idx)
        {
            auto fallback_mask = fallbackExpertMaskForLayer(layer_idx, tier_index);
            if (!anyExpertEnabled(fallback_mask))
                continue;

            auto gate = loadFallbackWeight(layer_idx, "ffn_gate_exps.weight");
            auto up = loadFallbackWeight(layer_idx, "ffn_up_exps.weight");
            auto down = loadFallbackWeight(layer_idx, "ffn_down_exps.weight");
            if (!gate || !up || !down)
            {
                clearFallbackWeightCache();
                return false;
            }

            auto input = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
            auto routing_indices = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});
            auto routing_weights = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});
            auto output = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});

            std::fill_n(input->mutable_data(), input->numel(), 0.0f);
            std::fill_n(routing_indices->mutable_data(), routing_indices->numel(), -1.0f);
            std::fill_n(routing_weights->mutable_data(), routing_weights->numel(), 0.0f);
            std::fill_n(output->mutable_data(), output->numel(), 0.0f);

            MoEExpertOverlayCPUFallbackStage::Params params;
            params.device_id = DeviceId::cpu();
            params.mpi_ctx = mpi_ctx_.get();
            params.domain = *cpu_domain;
            params.root_world_rank = continuationRootWorldRank();
            params.domain_id = MoEExpertOverlayCPUFallback::stableDomainId(cpu_domain->name);
            params.input = input.get();
            params.routing_indices = routing_indices.get();
            params.routing_weights = routing_weights.get();
            params.gate_exps = gate.get();
            params.up_exps = up.get();
            params.down_exps = down.get();
            params.output = output.get();
            params.seq_len = seq_len;
            params.d_model = d_model;
            params.num_experts = num_experts;
            params.top_k = top_k;
            params.expert_intermediate = expert_intermediate;
            params.layer_idx = layer_idx;
            params.expert_mask = std::move(fallback_mask);
            params.transfer_mode = MoEExpertTransferMode::Auto;

            MoEExpertOverlayCPUFallbackStage stage(std::move(params));
            const bool executed = stage.execute(cpu_ctx_.get());
            clearFallbackWeightCache();
            if (!executed)
                return false;
        }

        return true;
    }

    bool runFallbackParticipantPrefill()
    {
        // See executeFallbackParticipantPass(): this wrapper is temporary
        // scaffolding for the Phase 0 orchestration gap, not production runner
        // behavior. Reference:
        // docs/v2/MOE_EXPERT_OVERLAY_ORCHESTRATION_REFACTOR_PLAN.md.
        return executeFallbackParticipantPass(static_cast<int>(config_.token_ids.size()));
    }

    bool runFallbackParticipantDecode()
    {
        // See executeFallbackParticipantPass(): this wrapper is temporary
        // scaffolding for the Phase 0 orchestration gap, not production runner
        // behavior. Reference:
        // docs/v2/MOE_EXPERT_OVERLAY_ORCHESTRATION_REFACTOR_PLAN.md.
        if (!runFallbackParticipantPrefill())
            return false;

        const auto decode_tokens = readDecodeTokensFromMetadata();
        const size_t steps = std::min(decode_tokens.size(), static_cast<size_t>(config_.decode_steps));
        for (size_t step = 0; step < steps; ++step)
        {
            if (!executeFallbackParticipantPass(1))
                return false;
        }
        return true;
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
               summary.top1_matches > 0 ||
               summary.top3_matches > 0 ||
               summary.top5_matches > 0;
    }

    [[noreturn]] void abortOverlayWorld(const std::string &reason) const
    {
        const std::string message = "[Qwen3.5 MoE ExpertOverlay] " + reason +
                                    "; aborting MPI world to avoid stranding CPU NodeLocalTP participants";
        LOG_ERROR(message);
        std::cerr << message << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 2);
        std::abort();
    }

    void runOverlayPrefillParityBody()
    {
        const bool setup_ok = setupPipeline();
        ASSERT_TRUE(synchronizeRanksOk(setup_ok)) << "Pipeline setup failed";

        if (const auto blocker = synchronizedRuntimeBlocker(); !blocker.empty())
            GTEST_SKIP() << blocker;

        if (!isRootParityRank())
        {
            ASSERT_TRUE(runFallbackParticipantPrefill())
                << "Rank " << mpiRank() << " failed CPU fallback participant prefill";
            return;
        }

        auto summary = runPrefillParity();
        if (!producedPrefillSummary(summary))
        {
            abortOverlayWorld("root rank produced no prefill parity summary, likely because forward failed before all overlay tiers completed");
        }
        assertParity(summary);
    }

    void runOverlayDecodeParityBody()
    {
        const bool setup_ok = setupPipeline();
        ASSERT_TRUE(synchronizeRanksOk(setup_ok)) << "Pipeline setup failed";

        if (const auto blocker = synchronizedRuntimeBlocker(); !blocker.empty())
            GTEST_SKIP() << blocker;

        if (!synchronizedDecodeWorkAvailable())
            GTEST_SKIP() << "Decode snapshots or decode token metadata are unavailable";

        if (!isRootParityRank())
        {
            ASSERT_TRUE(runFallbackParticipantDecode())
                << "Rank " << mpiRank() << " failed CPU fallback participant decode";
            return;
        }

        auto summary = runDecodeParity();
        if (!producedDecodeSummary(summary))
        {
            abortOverlayWorld("root rank produced no decode parity summary, likely because forward failed before all overlay tiers completed");
        }
        assertDecodeParity(summary);
    }

    std::shared_ptr<MoEExpertParallelPlan> overlay_plan_;
    std::string setup_runtime_blocker_;
    std::unique_ptr<llaminar2::testing::MockDeviceContext> cpu_ctx_;
};

TEST_F(Qwen35MoEExpertOverlay, OverlayPlanTopology_ROCm2TP_SharedHot_CPU2NodeLocalTP_Cold)
{
    if (!isRootParityRank())
        GTEST_SKIP() << "Rank " << mpiRank() << " participates in MPI setup/barriers only; rank 0 owns the topology assertion";

    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    ASSERT_NE(overlay_plan_, nullptr);
    EXPECT_TRUE(overlay_plan_->isTieredOverlay());
    EXPECT_EQ(overlay_plan_->placements.size(), static_cast<size_t>(model_ctx_->totalBlockCount()));
    EXPECT_EQ(overlay_plan_->continuation_domain, kRocmSharedHotDomain);
    EXPECT_EQ(overlay_plan_->shared_expert_domain, kRocmSharedHotDomain);
    ASSERT_EQ(overlay_plan_->routed_tiers.size(), 2u);
    EXPECT_EQ(overlay_plan_->routed_tiers[0].memory_budget_bytes, gib(4));
    EXPECT_TRUE(overlay_plan_->routed_tiers.back().fallback);

    const auto counts = tierExpertCounts(*overlay_plan_);
    ASSERT_EQ(counts.size(), 2u);
    EXPECT_GT(counts[0], 0u);
    EXPECT_GT(counts[1], 0u);
}

TEST_F(Qwen35MoEExpertOverlay, OverlayPlanTopology_CUDA1_SharedHot_ROCm2TP_Hot_CPU2NodeLocalTP_Cold)
{
    if (!isRootParityRank())
        GTEST_SKIP() << "Rank " << mpiRank() << " participates in MPI setup/barriers only; rank 0 owns the topology assertion";

    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    ASSERT_NE(overlay_plan_, nullptr);
    EXPECT_TRUE(overlay_plan_->isTieredOverlay());
    EXPECT_EQ(overlay_plan_->placements.size(), static_cast<size_t>(model_ctx_->totalBlockCount()));
    EXPECT_EQ(overlay_plan_->continuation_domain, kCudaSharedHotDomain);
    EXPECT_EQ(overlay_plan_->shared_expert_domain, kCudaSharedHotDomain);
    ASSERT_EQ(overlay_plan_->routed_tiers.size(), 3u);
    EXPECT_EQ(overlay_plan_->routed_tiers[0].memory_budget_bytes, gib(2));
    EXPECT_EQ(overlay_plan_->routed_tiers[1].memory_budget_bytes, gib(4));
    EXPECT_TRUE(overlay_plan_->routed_tiers.back().fallback);

    const auto counts = tierExpertCounts(*overlay_plan_);
    ASSERT_EQ(counts.size(), 3u);
    EXPECT_GT(counts[0], 0u);
    EXPECT_GT(counts[1], 0u);
    EXPECT_GT(counts[2], 0u);
}

TEST_F(Qwen35MoEExpertOverlay, PrefillParity_ROCm2TP_SharedHot_CPU2NodeLocalTP_Cold)
{
    runOverlayPrefillParityBody();
}

TEST_F(Qwen35MoEExpertOverlay, DecodeParity_ROCm2TP_SharedHot_CPU2NodeLocalTP_Cold)
{
    runOverlayDecodeParityBody();
}

TEST_F(Qwen35MoEExpertOverlay, PrefillParity_CUDA1_SharedHot_ROCm2TP_Hot_CPU2NodeLocalTP_Cold)
{
    runOverlayPrefillParityBody();
}

TEST_F(Qwen35MoEExpertOverlay, DecodeParity_CUDA1_SharedHot_ROCm2TP_Hot_CPU2NodeLocalTP_Cold)
{
    runOverlayDecodeParityBody();
}

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (rank == 0)
    {
        std::cout << "Qwen3.5 MoE ExpertOverlay V2 parity suite: MPI world size="
                  << world_size << ", thread support="
                  << (provided >= MPI_THREAD_MULTIPLE ? "MPI_THREAD_MULTIPLE" : "limited")
                  << std::endl;
    }

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    int global_result = 0;
    MPI_Allreduce(&result, &global_result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();

    MPI_Finalize();

    std::cout.flush();
    std::cerr.flush();
    _exit(global_result);
}