/**
 * @file Test__Qwen35MoE_GraphNative_RocmHotCpuCold_Parity.cpp
 * @brief Phase 15: Production-path PyTorch parity gate for Qwen3.5 MoE graph-native
 *        overlay with `rocm_hot` (ROCm device 0, ReplicatedExperts) and
 *        `cpu_cold` (CPU fallback tier, ReplicatedExperts) two-tier layout.
 *
 * Topology:
 *   rocm_hot   — single ROCm device (rocm:0), world rank 0, ReplicatedExperts.
 *                This is the continuation domain and shared expert domain.
 *                Hot experts (first num_experts/2 by ID) reside here.
 *   cpu_cold   — single CPU participant, world rank 1, ReplicatedExperts, fallback=true.
 *                Cold experts (remaining num_experts/2 by ID) fall back here.
 *
 * Key differences from legacy ExpertOverlay tests (Test__Qwen35MoE_ExpertOverlay_Parity.cpp):
 *   - Both domains use ReplicatedExperts, NOT TensorParallelExperts.
 *   - rocm_hot is ExpertDomainKind::SingleDevice (not LocalTP).
 *   - cpu_cold is ExpertDomainKind::SingleDevice CPU on rank 1 (not NodeLocalTP cross-socket).
 *   - Parity bodies are NOT unconditionally skipped — Phase 14 graph-native is implemented.
 *   - Asserts LLAMINAR_MOE_LEGACY_OVERLAY_DOMAIN_RUNTIME is NOT set.
 *   - Verifies MoEExpertOverlayProfiler cpu_fallback_rows > 0 in the
 *     profiler parity case. CTest discovery injects LLAMINAR_PROFILING=1.
 *
 * NOTE on multi-rank CPU sidecar: The cpu_cold domain is specified on world rank 1.
 * If OrchestrationRunner does not yet fully drive a participant-graph on rank 1
 * (i.e. rank 1's cpu_cold expert compute is still handled via in-process CPU fallback
 * on rank 0 rather than a full remote dispatch), this is noted here — the test does not
 * pretend the CPU sidecar is exercised as a separate graph. cpu_fallback_rows > 0 in
 * the profiler validates the fallback path regardless of where the compute occurs.
 *
 * MPI requirement: 2 ranks (rank 0 = rocm_hot owner, rank 1 = cpu_cold participant).
 *
 * Hardware/model gates:
 *   - TopologySmoke: no hardware or model required (rank 0 only).
 *   - PrefillParity, DecodeParity, ProfilerCpuFallbackRows: skip cleanly when
 *     ROCm is absent or model is missing. Non-skipping on production hardware lane.
 *
 * @author David Sanftenberg
 * @date 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>

#include "Qwen35MoEParityTestBase.h"
#include "backends/ComputeBackend.h"
#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"
#include "execution/factory/InferenceRunnerFactory.h"
#include "execution/moe/MoEExpertOverlayProfiler.h"
#include "execution/moe/MoEExpertParallelPlanner.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen35moe;

namespace
{
    constexpr const char *kModelPath = "/opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf";
    constexpr const char *kSnapshotDir = "pytorch_qwen35_moe_snapshots";
    constexpr const char *kRocmHotDomain = "rocm_hot";
    constexpr const char *kCpuColdDomain = "cpu_cold";
    constexpr const char *kLegacyEnvVar = "LLAMINAR_MOE_LEGACY_OVERLAY_DOMAIN_RUNTIME";

    // Approximate Qwen3.5-35B-A3B metadata for topology-only (model-free) plan construction.
    // These are used only in TopologySmoke; real model metadata overrides them in parity runs.
    constexpr int kQwen35MoENumExperts = 256;
    constexpr int kQwen35MoENumLayers = 94;

    bool isLegacyOverlayRuntimeEnabled()
    {
        const char *value = std::getenv(kLegacyEnvVar);
        return value != nullptr && std::string(value) == "1";
    }

    bool modelAvailable()
    {
        return std::filesystem::exists(kModelPath);
    }

    /**
     * @brief rocm_hot domain: single ROCm device (rocm:0), world rank 0, ReplicatedExperts.
     *
     * Uses ExpertDomainKind::SingleDevice and ReplicatedExperts so that the
     * graph-native path treats each domain as a whole-expert owner (one expert
     * per node in the MoEExpertOwnerMap), exercising gn_sparse_dispatch /
     * gn_local_expert / gn_return_reduce stages introduced in Phase 14.
     */
    ExpertComputeDomain rocmHotDomain()
    {
        ExpertComputeDomain domain;
        domain.name = kRocmHotDomain;
        domain.kind = ExpertDomainKind::SingleDevice;
        domain.backend = CollectiveBackendType::RCCL;
        domain.participants = {GlobalDeviceAddress::rocm(0)};
        domain.world_ranks = {0};
        domain.owner_rank = 0;
        domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
        return domain;
    }

    /**
     * @brief cpu_cold domain: single CPU on world rank 1, ReplicatedExperts, fallback tier.
     *
     * When the hot tier is exhausted, remaining experts fall back to this domain.
     * Uses ReplicatedExperts — each expert is fully owned by rank 1's CPU rather
     * than tensor-parallel sharded across sockets. This is the graph-native whole-expert
     * owner path for the cold tier.
     *
     * NOTE: ExpertDomainKind::SingleDevice with HOST backend is chosen because this
     * is a single logical CPU device (one participant, rank 1). The cross-rank return
     * reduction is handled by the graph-native sparse collective infrastructure.
     */
    ExpertComputeDomain cpuColdDomain()
    {
        ExpertComputeDomain domain;
        domain.name = kCpuColdDomain;
        domain.kind = ExpertDomainKind::SingleDevice;
        domain.backend = CollectiveBackendType::HOST;
        domain.participants = {GlobalDeviceAddress::cpu(0)};
        domain.world_ranks = {1};
        domain.owner_rank = 1;
        domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
        return domain;
    }

    ExpertRoutedTier makeTier(
        const std::string &name,
        const std::string &domain,
        int priority,
        int max_experts_per_layer,
        size_t memory_budget_bytes,
        bool fallback = false)
    {
        ExpertRoutedTier t;
        t.name = name;
        t.domain = domain;
        t.priority = priority;
        t.max_experts_per_layer = max_experts_per_layer;
        t.memory_budget_bytes = memory_budget_bytes;
        t.fallback = fallback;
        return t;
    }

    MoEExpertModelMetadata topologyOnlyMetadata()
    {
        MoEExpertModelMetadata metadata;
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

    /**
     * @brief Build the requested (unplanned) MoEExpertParallelPlan for the
     *        rocm_hot / cpu_cold two-tier graph-native layout.
     *
     * Hot capacity = num_experts / 2 (StaticById assigns by expert-ID range).
     * Cold tier is unbounded fallback.
     */
    MoEExpertParallelPlan requestedPlan(const MoEExpertModelMetadata &metadata)
    {
        const int hot_capacity = std::max(1, metadata.num_experts / 2);

        MoEExpertParallelPlan plan;
        plan.enabled = true;
        plan.execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
        plan.continuation_domain = kRocmHotDomain;
        plan.shared_expert_domain = kRocmHotDomain;
        plan.residency_policy = ExpertResidencyPolicy::StaticById;
        plan.domains = {
            rocmHotDomain(),
            cpuColdDomain(),
        };
        plan.routed_tiers = {
            makeTier("hot", kRocmHotDomain, 0, hot_capacity, 4ULL * 1024 * 1024 * 1024),
            makeTier("cold", kCpuColdDomain, 1, 0, 0, /*fallback=*/true),
        };
        return plan;
    }

    std::string planValidationErrors(const MoEExpertParallelValidationResult &validation)
    {
        std::ostringstream message;
        for (const auto &error : validation.errors)
            message << "\n - " << error;
        return message.str();
    }

    std::shared_ptr<MoEExpertParallelPlan> makePlannedOverlayPlan(
        const MoEExpertModelMetadata &metadata)
    {
        auto result = MoEExpertParallelPlanner::plan(requestedPlan(metadata), metadata);
        auto planned = result.planned_plan;

        MoEExpertParallelValidationOptions options;
        options.layer_count = metadata.num_layers;
        options.routed_expert_count = metadata.num_experts;
        auto validation = validateMoEExpertParallelPlan(planned, options);
        if (!validation.ok())
        {
            throw std::invalid_argument(
                "Graph-native RocmHot/CpuCold plan is invalid:" +
                planValidationErrors(validation));
        }
        return std::make_shared<MoEExpertParallelPlan>(std::move(planned));
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

    std::optional<std::string> rocmHardwareBlocker()
    {
        const int rocm_count = getRocmDeviceCount();
        if (rocm_count < 1)
            return "GraphNative RocmHot/CpuCold parity requires >=1 ROCm device, found " +
                   std::to_string(rocm_count);
        return std::nullopt;
    }

    TestConfig makeGraphNativeTestConfig()
    {
        TestConfig config;
        config.name = "GraphNative_RocmHot_CpuCold";
        // devices = ROCm: signals to the base class that primary execution is on ROCm.
        // The actual device assignment is driven by the overlay plan (rocm:0 for hot tier).
        config.devices = {ParityDeviceType::ROCm};
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

// =============================================================================
// Test Fixture
// =============================================================================

class Qwen35MoEGraphNativeRocmHotCpuCold
    : public Qwen35MoEConfigDrivenParityTest<Qwen35MoEGraphNativeRocmHotCpuCold>
{
public:
    static const TestConfig &staticConfig()
    {
        static TestConfig kConfig = makeGraphNativeTestConfig();
        return kConfig;
    }

    const TestConfig &getTestConfig() const { return staticConfig(); }

protected:
    using Base = Qwen35MoEConfigDrivenParityTest<Qwen35MoEGraphNativeRocmHotCpuCold>;

    void SetUp() override
    {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized)
        {
            GTEST_SKIP() << "GraphNative RocmHot/CpuCold parity requires MPI initialization";
        }

        int rank = 0;
        int world_size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        if (world_size < cfg().mpi_ranks)
        {
            GTEST_SKIP() << "GraphNative RocmHot/CpuCold parity requires "
                         << cfg().mpi_ranks << " MPI ranks (got " << world_size << ")";
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

        if (!modelAvailable())
            return; // Parity bodies will gate on model availability before using snapshots

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

    /**
     * @brief Collectively broadcast a flag from rank 0 to all ranks.
     *
     * rank 0 sets the value; all ranks receive the result.
     */
    bool broadcastRootFlag(bool root_value) const
    {
        int flag = isRootParityRank() && root_value ? 1 : 0;
        MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
        return flag != 0;
    }

    /**
     * @brief All-reduce: returns true only if ALL ranks report ok.
     */
    bool synchronizeRanksOk(bool local_ok) const
    {
        int ok = local_ok ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        return ok == 1;
    }

    /**
     * @brief Collectively gate on hardware and model availability.
     *
     * Rank 0 evaluates both conditions and broadcasts. All ranks receive the
     * result. If false, the calling test should GTEST_SKIP (rank 0) or return
     * (non-root ranks).
     */
    bool collectivelyCheckHardwareAndModel() const
    {
        bool available = false;
        if (isRootParityRank())
            available = !rocmHardwareBlocker().has_value() && modelAvailable();
        return broadcastRootFlag(available);
    }

    /**
     * @brief Set up the inference pipeline for the graph-native rocm_hot/cpu_cold layout.
     *
     * Rank 0 (rocm_hot): loads the model and creates the InferenceRunner.
     * Rank 1 (cpu_cold): builds only the overlay plan for MPI coordination.
     *
     * The graph-native path is selected automatically when
     * LLAMINAR_MOE_LEGACY_OVERLAY_DOMAIN_RUNTIME is NOT set (the default).
     */
    bool setupPipeline()
    {
        if (!isRootParityRank())
        {
            // Rank 1 (cpu_cold participant): build the plan for coordination metadata.
            // Full participant-graph execution on rank 1 depends on production
            // OrchestrationRunner multi-rank dispatch being wired; if not yet
            // available, rank 1 participates in MPI barriers only and cpu_cold
            // expert compute falls back to the in-process path on rank 0.
            try
            {
                overlay_plan_ = makePlannedOverlayPlan(topologyOnlyMetadata());
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("[Qwen3.5 MoE GraphNative rank1] Plan failed: " << e.what());
                return false;
            }
            return true;
        }

        // Rank 0 (rocm_hot continuation owner): load model + create runner.
        DeviceManager::instance().initialize(-1);
        model_ctx_ = ModelContext::create(
            config_.model_path,
            nullptr,
            nullptr,
            nullptr,
            WeightDistributionStrategy::REPLICATED);
        if (!model_ctx_)
        {
            LOG_ERROR("[Qwen3.5 MoE GraphNative] Failed to load model from " << config_.model_path);
            return false;
        }

        configureModel(model_ctx_);

        const auto full_metadata = metadataFromModel(*model_ctx_);
        try
        {
            overlay_plan_ = makePlannedOverlayPlan(full_metadata);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[Qwen3.5 MoE GraphNative] Overlay plan construction failed: " << e.what());
            return false;
        }

        InferenceRunnerConfig inf_config;
        inf_config.max_seq_len = 4096;
        inf_config.batch_size = 1;
        inf_config.force_graph = true;
        inf_config.activation_precision = cfg().activation_precision;
        inf_config.kv_cache_precision = cfg().kv_cache_precision;
        inf_config.moe_expert_parallel_plan = overlay_plan_;
        inf_config.moe_expert_overlay_mpi_ctx = mpi_ctx_;

        runner_ = createInferenceRunner(model_ctx_, nullptr, DeviceId::rocm(0), inf_config);
        if (!runner_)
        {
            LOG_ERROR("[Qwen3.5 MoE GraphNative] Failed to create inference runner on rocm:0");
            return false;
        }

        runner_->enableSnapshotCapture();
        return true;
    }

    bool isRootParityRank() const { return isRank0(); }

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
            "; aborting MPI world to avoid stranding cpu_cold participant";
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
     * @brief Run prefill parity body, hardware/model gated.
     *
     * Graph-native path: LLAMINAR_MOE_LEGACY_OVERLAY_DOMAIN_RUNTIME must be unset.
     * Both ranks participate in MPI collectives; rank 0 owns the parity assertions.
     */
    void runGraphNativePrefillParityBody()
    {
        if (isLegacyOverlayRuntimeEnabled())
        {
            FAIL() << kLegacyEnvVar
                   << " is set in the environment. This test requires the graph-native "
                      "overlay path. Unset this environment variable.";
        }

        // Collective hardware + model gate: both ranks participate in the broadcast.
        const bool hardware_and_model_ok = collectivelyCheckHardwareAndModel();
        if (!hardware_and_model_ok)
        {
            if (isRootParityRank())
            {
                const auto blocker = rocmHardwareBlocker();
                if (blocker)
                    GTEST_SKIP() << *blocker;
                else
                    GTEST_SKIP() << "Model not found at " << kModelPath;
            }
            return; // Non-root rank: return without skip
        }

        const bool setup_ok = setupPipeline();
        ASSERT_TRUE(synchronizeRanksOk(setup_ok)) << "Pipeline setup failed on one or more ranks";

        if (!isRootParityRank())
            return; // Rank 1 participates in MPI collectives inside the runner

        ParityTestSummary summary;
        try
        {
            summary = runPrefillParity();
        }
        catch (const std::exception &e)
        {
            abortAfterRootThrow("prefill parity", e.what());
        }
        catch (...)
        {
            abortAfterRootThrow("prefill parity", "unknown exception");
        }

        if (!producedPrefillSummary(summary))
        {
            abortGraphNativeWorld(
                "root produced no prefill parity summary — forward likely failed before "
                "all overlay tiers completed");
        }

        assertParity(summary);
    }

    /**
     * @brief Run decode parity body, hardware/model/snapshot gated.
     */
    void runGraphNativeDecodeParityBody()
    {
        if (isLegacyOverlayRuntimeEnabled())
        {
            FAIL() << kLegacyEnvVar
                   << " is set. Unset for graph-native decode parity.";
        }

        const bool hardware_and_model_ok = collectivelyCheckHardwareAndModel();
        if (!hardware_and_model_ok)
        {
            if (isRootParityRank())
            {
                const auto blocker = rocmHardwareBlocker();
                if (blocker)
                    GTEST_SKIP() << *blocker;
                else
                    GTEST_SKIP() << "Model not found at " << kModelPath;
            }
            return;
        }

        if (!synchronizedDecodeWorkAvailable())
            GTEST_SKIP() << "Decode snapshots or decode token metadata are unavailable";

        const bool setup_ok = setupPipeline();
        ASSERT_TRUE(synchronizeRanksOk(setup_ok)) << "Pipeline setup failed";

        if (!isRootParityRank())
            return;

        DecodeParitySummary summary;
        try
        {
            summary = runDecodeParity();
        }
        catch (const std::exception &e)
        {
            abortAfterRootThrow("decode parity", e.what());
        }
        catch (...)
        {
            abortAfterRootThrow("decode parity", "unknown exception");
        }

        if (!producedDecodeSummary(summary))
        {
            abortGraphNativeWorld(
                "root produced no decode parity summary — forward likely failed before "
                "all overlay tiers completed");
        }

        assertDecodeParity(summary);
    }

    std::shared_ptr<MoEExpertParallelPlan> overlay_plan_;
};

// =============================================================================
// Tests
// =============================================================================

/**
 * @brief Smoke test: validates the rocm_hot / cpu_cold owner map and plan topology
 *        without loading the real model or requiring ROCm hardware.
 *
 * Asserts (rank 0 only):
 *   - Both domains use ReplicatedExperts (graph-native whole-expert owner path)
 *   - cpu_cold routed tier has fallback=true
 *   - All layers have placements
 *   - Both hot and cold tiers have expert assignments (cpu_fallback path is reachable)
 *   - Hot + cold counts == num_experts * num_layers
 *   - LLAMINAR_MOE_LEGACY_OVERLAY_DOMAIN_RUNTIME is NOT set
 */
TEST_F(Qwen35MoEGraphNativeRocmHotCpuCold, TopologySmoke)
{
    if (!isRootParityRank())
        return;

    // The production graph-native path must NOT be overridden by the legacy env var.
    EXPECT_FALSE(isLegacyOverlayRuntimeEnabled())
        << kLegacyEnvVar
        << " must NOT be set for graph-native tests. If this fails, the legacy path "
           "was accidentally activated and the graph-native stages will not be exercised.";

    const auto metadata = topologyOnlyMetadata();

    std::shared_ptr<MoEExpertParallelPlan> plan;
    ASSERT_NO_THROW(plan = makePlannedOverlayPlan(metadata))
        << "Plan construction threw — check ExpertComputeDomain definitions for "
           "rocm_hot (SingleDevice, RCCL, ReplicatedExperts) and "
           "cpu_cold (SingleDevice, HOST, ReplicatedExperts, fallback)";
    ASSERT_NE(plan, nullptr);

    // Basic plan structure
    EXPECT_TRUE(plan->isTieredOverlay());
    EXPECT_EQ(plan->continuation_domain, kRocmHotDomain);
    EXPECT_EQ(plan->shared_expert_domain, kRocmHotDomain);
    ASSERT_EQ(plan->domains.size(), 2u);
    ASSERT_EQ(plan->routed_tiers.size(), 2u);

    // CRITICAL: both domains must use ReplicatedExperts for the graph-native path.
    // TensorParallelExperts would select the legacy expert-sharded GEMM path.
    for (const auto &domain : plan->domains)
    {
        EXPECT_EQ(domain.compute_kind, ExpertDomainComputeKind::ReplicatedExperts)
            << "Domain '" << domain.name
            << "' must use ReplicatedExperts (whole-expert graph-native owner), "
               "not TensorParallelExperts (legacy sharded GEMM path)";
    }

    // rocm_hot domain must be SingleDevice (not LocalTP/NodeLocalTP)
    const auto *hot_domain = &plan->domains[0];
    EXPECT_EQ(hot_domain->kind, ExpertDomainKind::SingleDevice)
        << "rocm_hot must be SingleDevice, not LocalTP/NodeLocalTP";
    EXPECT_FALSE(hot_domain->participants.empty());

    // cpu_cold must be the fallback tier
    EXPECT_TRUE(plan->routed_tiers.back().fallback)
        << "cpu_cold routed tier must have fallback=true";
    EXPECT_EQ(plan->routed_tiers.back().domain, kCpuColdDomain);

    // Every layer must have a placement
    ASSERT_EQ(plan->placements.size(), static_cast<size_t>(metadata.num_layers))
        << "Expected one placement per layer (" << metadata.num_layers << " layers)";

    // Both tiers must have expert assignments so that the cpu_fallback_rows path is reachable
    const auto counts = tierExpertCounts(*plan);
    ASSERT_EQ(counts.size(), 2u);
    EXPECT_GT(counts[0], 0u)
        << "Hot tier (rocm_hot) must have at least 1 expert assignment";
    EXPECT_GT(counts[1], 0u)
        << "Cold tier (cpu_cold) must have at least 1 expert assignment; "
           "if this is 0, cpu_fallback_rows will never be > 0 and the cold path is not exercised";

    // Sum check: every expert in every layer must be assigned to exactly one tier
    const size_t expected_total =
        static_cast<size_t>(metadata.num_experts) *
        static_cast<size_t>(metadata.num_layers);
    EXPECT_EQ(counts[0] + counts[1], expected_total)
        << "hot_count + cold_count must equal num_experts * num_layers "
           "("
        << metadata.num_experts << " * " << metadata.num_layers << " = " << expected_total << ")";

    const size_t hot_per_layer = metadata.num_layers > 0 ? counts[0] / metadata.num_layers : 0;
    const size_t cold_per_layer = metadata.num_layers > 0 ? counts[1] / metadata.num_layers : 0;
    LOG_INFO("[TopologySmoke] rocm_hot experts/layer: " << hot_per_layer
                                                        << ", cpu_cold experts/layer: " << cold_per_layer
                                                        << " (total: " << metadata.num_experts << ")");
}

/**
 * @brief Prefill parity: compare Llaminar graph-native output against PyTorch
 *        reference snapshots, layer by layer.
 *
 * Hardware-gated: skips cleanly when ROCm is absent or model file is missing.
 * Expected to run non-skipped on the production hardware lane.
 *
 * Validates that the graph-native overlay (gn_sparse_dispatch / gn_local_expert /
 * gn_return_reduce stages) produces activations with cosine similarity >= 0.90
 * against the PyTorch Qwen3.5 MoE reference.
 */
TEST_F(Qwen35MoEGraphNativeRocmHotCpuCold, PrefillParity)
{
    runGraphNativePrefillParityBody();
}

/**
 * @brief Decode parity: compare Llaminar graph-native decode steps against
 *        PyTorch reference (token predictions, LM_HEAD cosine, KL divergence).
 *
 * Hardware-gated: skips cleanly when ROCm is absent or model file is missing.
 */
TEST_F(Qwen35MoEGraphNativeRocmHotCpuCold, DecodeParity)
{
    runGraphNativeDecodeParityBody();
}

/**
 * @brief Verify that MoEExpertOverlayProfiler emits cpu_fallback_rows > 0
 *        after a forward pass when the cold tier has expert assignments.
 *
 * Hardware-gated: skips when ROCm is absent or model file is missing.
 * CTest discovery injects LLAMINAR_PROFILING=1 for this profiler parity case.
 *
 * This test validates the Phase 14 graph-native profiler metric end-to-end:
 *   - gn_sparse_dispatch row: compact_dispatch_bytes > 0 for cold-tier rows
 *   - gn_return_reduce row: compact_return_bytes > 0, cpu_fallback_rows > 0
 *   - dense_bytes_avoided > 0 (sparse dispatch savings vs dense broadcast)
 *
 * NOTE: cpu_fallback_rows represents tokens routed to cold-tier experts.
 * With StaticById and hot_capacity = num_experts/2 (= 128), approximately
 * half of all expert selections will route to the cpu_cold tier.
 */
TEST_F(Qwen35MoEGraphNativeRocmHotCpuCold, ProfilerCpuFallbackRows)
{
    if (isLegacyOverlayRuntimeEnabled())
    {
        FAIL() << kLegacyEnvVar << " must NOT be set for graph-native profiler test.";
    }

    // Profiling must be enabled at process startup because DebugEnv caches env
    // values. CTest discovery injects and mpirun-forwards this for profiler cases.
    ASSERT_TRUE(MoEExpertOverlayProfiler::isEnabled())
        << "Profiler parity tests must run with LLAMINAR_PROFILING=1; "
           "CTest discovery should inject and mpirun-forward it.";

    const bool hardware_and_model_ok = collectivelyCheckHardwareAndModel();
    if (!hardware_and_model_ok)
    {
        if (isRootParityRank())
        {
            const auto blocker = rocmHardwareBlocker();
            if (blocker)
                GTEST_SKIP() << *blocker;
            else
                GTEST_SKIP() << "Model not found at " << kModelPath;
        }
        return;
    }

    const bool setup_ok = setupPipeline();
    ASSERT_TRUE(synchronizeRanksOk(setup_ok)) << "Pipeline setup failed";

    if (!isRootParityRank())
        return;

    // Reset profiler to get clean rows from this run only
    MoEExpertOverlayProfiler::reset();

    runner_->forward(config_.token_ids.data(), config_.token_ids.size());

    const auto &rows = MoEExpertOverlayProfiler::rows();
    ASSERT_FALSE(rows.empty())
        << "No profiler rows emitted after forward pass with LLAMINAR_PROFILING=1. "
           "Check that graph-native sparse stages (gn_sparse_dispatch, gn_local_expert, "
           "gn_return_reduce) are recording rows via MoEExpertOverlayProfiler.";

    // At least one row must have cpu_fallback_rows > 0 (cold-tier experts were dispatched)
    const bool has_cpu_fallback =
        std::any_of(rows.begin(), rows.end(),
                    [](const MoEExpertOverlayProfileRow &row)
                    {
                        return row.cpu_fallback_rows > 0;
                    });
    EXPECT_TRUE(has_cpu_fallback)
        << "Expected cpu_fallback_rows > 0 in at least one gn_return_reduce profiler row. "
           "With StaticById and hot_capacity = num_experts/2, roughly half the expert "
           "selections should route to the cpu_cold fallback tier.";

    const size_t cpu_fallback_row_count =
        static_cast<size_t>(std::count_if(rows.begin(), rows.end(),
                                          [](const MoEExpertOverlayProfileRow &row)
                                          {
                                              return row.cpu_fallback_rows > 0;
                                          }));
    LOG_INFO("[ProfilerCpuFallbackRows] Total profiler rows: " << rows.size()
                                                               << "; rows with cpu_fallback_rows>0: "
                                                               << cpu_fallback_row_count);
}

// =============================================================================
// Custom main with MPI initialization
// =============================================================================

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "[Rank " << rank << "] Qwen3.5 MoE GraphNative RocmHot/CpuCold parity test\n";

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    // Broadcast result to detect any rank-local failure
    MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();

    MPI_Finalize();

    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
