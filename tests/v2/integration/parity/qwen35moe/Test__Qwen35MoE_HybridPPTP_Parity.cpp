/**
 * @file Test__Qwen35MoE_HybridPPTP_Parity.cpp
 * @brief Hybrid PP+TP parity coverage for Qwen3.5 MoE expert GPU-cache migration.
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <numeric>
#include <optional>
#include <sstream>
#include "Qwen35MoEParityTestBase.h"
#include "backends/ComputeBackend.h"
#include "collective/BackendRouter.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/global/DomainCommunicatorRegistry.h"
#include "execution/global/GlobalOrchestrator.h"
#include "execution/global/StageRunnerFactory.h"
#include "execution/global_pp/GlobalPPRankPlanBuilder.h"
#include "execution/mpi_orchestration/ExecutionPlanBuilder.h"
#include "execution/moe/ExpertWeightTransfer.h"
#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "execution/local_execution/orchestrators/RankOrchestrator.h"
#include "loaders/ModelContextConfig.h"
#include "planning/ClusterInventoryGatherer.h"
#include "utils/DebugEnv.h"

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen35moe;

namespace
{
    constexpr const char *kRocmDomainRank0 = "rocm_socket0";
    constexpr const char *kRocmDomainRank1 = "rocm_socket1";
    constexpr const char *kCpuDomain = "cpu_sockets";
    constexpr int kExpectedWorldSize = 2;
    constexpr const char *kHybridPPTPGpuCacheSkipReason =
        "Qwen35 MoE HybridPPTP GPU-cache parity is skipped: the current sequential PP topology "
        "splits layers between ROCm and CPU domains, so static hot/cold expert masks cannot "
        "reconstruct same-layer routed expert outputs. Re-enable after the expert-residency "
        "architecture supports same-layer cold-expert fallback or a parallel expert domain split.";

    using Clock = std::chrono::steady_clock;

    double elapsedMs(Clock::time_point start)
    {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }

    void logPhaseTiming(const std::string &name, Clock::time_point start, int rank)
    {
        std::ostringstream duration_stream;
        duration_stream << std::fixed << std::setprecision(1) << elapsedMs(start);
        LOG_INFO("[HybridPPTP][rank " << rank << "] " << name << " took "
                 << duration_stream.str() << " ms");
    }

    std::string activationPrecisionConfigValue(ActivationPrecision precision)
    {
        switch (precision)
        {
        case ActivationPrecision::BF16: return "bf16";
        case ActivationPrecision::FP16: return "fp16";
        case ActivationPrecision::Q8_1: return "q8_1";
        case ActivationPrecision::Q16_1: return "q16_1";
        case ActivationPrecision::Hybrid: return "hybrid";
        case ActivationPrecision::HybridQ16: return "hybridq16";
        case ActivationPrecision::FP32:
        default: return "fp32";
        }
    }

    std::string kvCachePrecisionConfigValue(KVCachePrecision precision)
    {
        switch (precision)
        {
        case KVCachePrecision::FP32: return "fp32";
        case KVCachePrecision::FP16: return "fp16";
        case KVCachePrecision::Q8_1: return "q8_1";
        case KVCachePrecision::Q16_1: return "q16_1";
        case KVCachePrecision::TQ4: return "tq4";
        case KVCachePrecision::TQ: return "tq";
        case KVCachePrecision::AUTO:
        default: return "auto";
        }
    }

    const GlobalPPStageSpec *stageByDomain(const GlobalPPTopology &topology,
                                           const std::string &domain_name)
    {
        for (const auto &stage : topology.stages)
        {
            if (stage.domain_name == domain_name)
            {
                return &stage;
            }
        }
        return nullptr;
    }

    const GlobalPPStageSpec *stageById(const GlobalPPTopology &topology, int stage_id)
    {
        for (const auto &stage : topology.stages)
        {
            if (stage.stage_id == stage_id)
            {
                return &stage;
            }
        }
        return nullptr;
    }

    bool vectorEquals(const std::vector<int> &actual, std::initializer_list<int> expected)
    {
        return actual == std::vector<int>(expected);
    }

    ClusterInventory makeTopologySmokeInventory()
    {
        ClusterInventory inventory;
        inventory.world_size = kExpectedWorldSize;

        RankInventory rank0;
        rank0.rank = 0;
        rank0.node_id = 0;
        rank0.local_rank = 0;
        rank0.hostname = "localhost";
        rank0.numa_nodes = 2;
        rank0.cpu.type = DeviceType::CPU;
        rank0.cpu.local_device_id = 0;
        rank0.cpu.numa_node = 0;
        rank0.cpu_cores = 32;
        rank0.cpu_memory_bytes = 256ULL * 1024ULL * 1024ULL * 1024ULL;
        for (int ordinal = 0; ordinal < 2; ++ordinal)
        {
            DeviceInfo gpu;
            gpu.type = DeviceType::ROCm;
            gpu.local_device_id = ordinal;
            gpu.numa_node = 0;
            gpu.memory_bytes = 48ULL * 1024ULL * 1024ULL * 1024ULL;
            gpu.free_memory_bytes = gpu.memory_bytes;
            gpu.name = "ROCm test GPU";
            rank0.gpus.push_back(gpu);
        }

        RankInventory rank1;
        rank1.rank = 1;
        rank1.node_id = 0;
        rank1.local_rank = 1;
        rank1.hostname = "localhost";
        rank1.numa_nodes = 2;
        rank1.cpu.type = DeviceType::CPU;
        rank1.cpu.local_device_id = 0;
        rank1.cpu.numa_node = 1;
        rank1.cpu_cores = 32;
        rank1.cpu_memory_bytes = 256ULL * 1024ULL * 1024ULL * 1024ULL;

        inventory.ranks = {std::move(rank0), std::move(rank1)};
        inventory.buildNodeAggregations();
        return inventory;
    }

    ModelConfig makeSmokeModelConfig()
    {
        ModelConfig model;
        model.name = "Qwen3.5-MoE-35B";
        model.n_layers = 40;
        model.n_heads = 64;
        model.n_kv_heads = 8;
        model.hidden_size = 8192;
        model.intermediate_size = 29568;
        model.vocab_size = 248320;
        model.head_dim = 128;
        return model;
    }

    ModelConfig makeModelConfigFromContext(const ModelContext &ctx)
    {
        ModelConfig model;
        model.name = ctx.architecture();
        model.n_layers = ctx.totalBlockCount();
        model.n_heads = ctx.headCount();
        model.n_kv_heads = ctx.headCountKV();
        model.hidden_size = ctx.embeddingLength();
        model.intermediate_size = ctx.feedForwardLength();
        model.vocab_size = ctx.vocabSize();
        model.head_dim = ctx.keyLength() > 0
                             ? ctx.keyLength()
                             : (ctx.headCount() > 0 ? ctx.embeddingLength() / ctx.headCount() : 0);
        return model;
    }

    OrchestrationConfig makeNamedDomainConfig(const std::string &model_path,
                                              int n_layers,
                                              ActivationPrecision activation_precision,
                                              KVCachePrecision kv_cache_precision,
                                              bool mirrored_gpu_owner)
    {
        const int split = std::max(1, n_layers / 2);
        const std::string rocm_domain = mirrored_gpu_owner ? kRocmDomainRank1 : kRocmDomainRank0;
        const int rocm_owner = mirrored_gpu_owner ? 1 : 0;

        OrchestrationConfig config = OrchestrationConfig::defaults();
        config.model_path = model_path;
        config.max_seq_len = 4096;
        config.batch_size = 1;
        config.activation_precision = activationPrecisionConfigValue(activation_precision);
        config.kv_cache_precision = kvCachePrecisionConfigValue(kv_cache_precision);
        config.pp_degree = 2;
        config.domain_definitions = {
            DomainDefinition::parse(
                rocm_domain + "=0:rocm:0,0:rocm:1;scope=local;backend=rccl;owner=" +
                std::to_string(rocm_owner)),
            DomainDefinition::parse(
                std::string(kCpuDomain) + "=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;ranks=0,1"),
        };
        config.pp_stage_definitions = {
            PPStageDefinition::parse(
                "0=" + rocm_domain + ":0-" + std::to_string(split - 1)),
            PPStageDefinition::parse(
                std::string("1=") + kCpuDomain + ":" + std::to_string(split) + "-" +
                std::to_string(n_layers - 1)),
        };
        return config;
    }

    bool assertNamedDomainTopologyShape(const GlobalPPTopology &topology,
                                        bool mirrored_gpu_owner)
    {
        const std::string rocm_domain = mirrored_gpu_owner ? kRocmDomainRank1 : kRocmDomainRank0;
        const int rocm_owner = mirrored_gpu_owner ? 1 : 0;
        bool ok = true;

        const auto *rocm_stage = stageByDomain(topology, rocm_domain);
        const auto *cpu_stage = stageByDomain(topology, kCpuDomain);
        EXPECT_NE(rocm_stage, nullptr);
        EXPECT_NE(cpu_stage, nullptr);
        if (!rocm_stage || !cpu_stage)
        {
            return false;
        }

        EXPECT_EQ(topology.numStages(), 2);
        EXPECT_EQ(rocm_stage->stage_id, 0);
        EXPECT_EQ(rocm_stage->owning_rank, rocm_owner);
        EXPECT_FALSE(rocm_stage->is_global_tp);
        EXPECT_EQ(rocm_stage->inner_mode, InnerParallelism::LOCAL_TP);
        EXPECT_EQ(rocm_stage->backend, CollectiveBackendType::RCCL);
        EXPECT_EQ(rocm_stage->devices.size(), 2u);
        ok &= rocm_stage->devices.size() == 2u;
        for (const auto &device : rocm_stage->devices)
        {
            EXPECT_TRUE(device.isROCm()) << device.toString();
            ok &= device.isROCm();
        }

        EXPECT_EQ(cpu_stage->stage_id, 1);
        EXPECT_TRUE(cpu_stage->is_global_tp);
        EXPECT_EQ(cpu_stage->backend, CollectiveBackendType::UPI);
        EXPECT_TRUE(vectorEquals(cpu_stage->participating_ranks, {0, 1}));
        EXPECT_EQ(cpu_stage->per_rank_devices.size(), 2u);
        ok &= cpu_stage->is_global_tp;
        ok &= cpu_stage->backend == CollectiveBackendType::UPI;
        ok &= vectorEquals(cpu_stage->participating_ranks, {0, 1});
        for (const auto &device : cpu_stage->per_rank_devices)
        {
            EXPECT_TRUE(device.isCPU()) << device.toString();
            ok &= device.isCPU();
        }

        if (!mirrored_gpu_owner)
        {
            EXPECT_EQ(topology.stagesForRank(0).size(), 2u);
            EXPECT_EQ(topology.stagesForRank(1).size(), 1u);
            ok &= topology.stagesForRank(0).size() == 2u;
            ok &= topology.stagesForRank(1).size() == 1u;
        }
        else
        {
            EXPECT_EQ(topology.stagesForRank(0).size(), 1u);
            EXPECT_EQ(topology.stagesForRank(1).size(), 2u);
            ok &= topology.stagesForRank(0).size() == 1u;
            ok &= topology.stagesForRank(1).size() == 2u;
        }

        return ok;
    }

    const std::vector<std::string> kHybridPPTPMoEExcludedStages = {
        "Q_PROJECTION",
        "K_PROJECTION",
        "V_PROJECTION",
        "Q_NORM",
        "K_NORM",
        "Q_ROPE",
        "K_ROPE",
        "ATTENTION_CONTEXT",
        "FFN_GATE",
        "FFN_UP",
        "FFN_SWIGLU",
        "QKV_PROJECTION",
        "GDN_Z_PROJECTION",
        "GDN_DELTA_RULE_OUTPUT",
        "GDN_NORM_GATE_OUTPUT",
    };

    const std::vector<std::string> kHybridPPTPMoEAllreduceStages = {
        "MOE_EXPERT_OUTPUT",
        "MOE_SHARED_EXPERT_OUTPUT",
        "MOE_SHARED_GATE_OUTPUT",
        "MOE_COMBINED_OUTPUT",
    };

    const std::vector<TestConfig> kHybridPPTPMoEConfigs = {
        {
            .name = "NamedDomainPP_LocalTP_ROCm_NodeLocalTP_CPU_35B_MoE",
            .devices = {ParityDeviceType::ROCm, ParityDeviceType::ROCm, ParityDeviceType::CPU, ParityDeviceType::CPU},
            .parallelism = Parallelism::NodeLocalPP,
            .thresholds = {
                .cosine_threshold = 0.88f,
                .decode_cosine_threshold = 0.76f,
                .early_layers_count = 6,
                .min_early_layers_passed = 4,
                .kl_threshold = 0.06f,
                .excluded_stages = kHybridPPTPMoEExcludedStages,
                .allreduce_stages = kHybridPPTPMoEAllreduceStages,
                .min_top1_accuracy = 0.60f,
                .min_top5_accuracy = 0.80f,
                .pytorch_top1_in_topk = 5,
            },
            .mpi_ranks = kExpectedWorldSize,
            .model_path = "/opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf",
            .snapshot_dir = "pytorch_qwen35_moe_snapshots",
            .activation_precision = ActivationPrecision::FP32,
            .kv_cache_precision = KVCachePrecision::FP16,
            .decode_steps = 2,
        },
    };
}

TEST(Qwen35MoEHybridPPTPNamedDomainTopology, Rank0OwnsRocmLocalTPStage)
{
    auto config = makeNamedDomainConfig("/tmp/qwen35-moe.gguf", 40,
                                        ActivationPrecision::FP32,
                                        KVCachePrecision::FP16,
                                        /*mirrored_gpu_owner=*/false);
    auto inventory = makeTopologySmokeInventory();
    auto model = makeSmokeModelConfig();

    ExecutionPlanBuilder builder;
    auto topology = builder.buildGlobalPPTopology(config, model, inventory);

    auto errors = topology.validate();
    ASSERT_TRUE(errors.empty()) << (errors.empty() ? "" : errors.front());
    ASSERT_TRUE(assertNamedDomainTopologyShape(topology, /*mirrored_gpu_owner=*/false));
    LOG_INFO("[HybridPPTP Smoke] rank0-owned topology:\n"
             << renderMultiDomainTopologyInfo(topology, kExpectedWorldSize));
}

TEST(Qwen35MoEHybridPPTPNamedDomainTopology, Rank1OwnsMirroredRocmLocalTPStage)
{
    auto config = makeNamedDomainConfig("/tmp/qwen35-moe.gguf", 40,
                                        ActivationPrecision::FP32,
                                        KVCachePrecision::FP16,
                                        /*mirrored_gpu_owner=*/true);
    auto inventory = makeTopologySmokeInventory();
    auto model = makeSmokeModelConfig();

    ExecutionPlanBuilder builder;
    auto topology = builder.buildGlobalPPTopology(config, model, inventory);

    auto errors = topology.validate();
    ASSERT_TRUE(errors.empty()) << (errors.empty() ? "" : errors.front());
    ASSERT_TRUE(assertNamedDomainTopologyShape(topology, /*mirrored_gpu_owner=*/true));
    LOG_INFO("[HybridPPTP Smoke] rank1-owned mirrored topology:\n"
             << renderMultiDomainTopologyInfo(topology, kExpectedWorldSize));
}

class Qwen35MoEHybridPPTPParityTest
    : public Qwen35MoEConfigDrivenParityTest<Qwen35MoEHybridPPTPParityTest>,
      public ::testing::WithParamInterface<TestConfig>
{
public:
    const TestConfig &getTestConfig() const { return GetParam(); }

protected:
    using Base = Qwen35MoEConfigDrivenParityTest<Qwen35MoEHybridPPTPParityTest>;

    void SetUp() override
    {
        int world_size = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        if (world_size != kExpectedWorldSize)
        {
            GTEST_SKIP() << "HybridPPTP named-domain parity requires exactly "
                         << kExpectedWorldSize << " MPI ranks (got " << world_size << ")";
        }

        setenv("LLAMINAR_MOE_REBALANCE", "dynamic", 1);
        setenv("LLAMINAR_MOE_REBALANCE_WINDOW", "2", 1);
        setenv("LLAMINAR_MOE_GPU_EXPERT_CACHE", "8", 1);
            setenv("LLAMINAR_FAIL_ON_ZERO", "0", 1);
        mutableDebugEnv().reload();

        Base::SetUp();
    }

    void TearDown() override
    {
        Base::TearDown();
        unsetenv("LLAMINAR_MOE_GPU_EXPERT_CACHE");
        unsetenv("LLAMINAR_MOE_REBALANCE_WINDOW");
        unsetenv("LLAMINAR_MOE_REBALANCE");
            unsetenv("LLAMINAR_FAIL_ON_ZERO");
        mutableDebugEnv().reload();
    }

    void configureModel(std::shared_ptr<ModelContext> model_ctx) override
    {
        Qwen35MoESchemaFactory schema_factory;
        model_ctx->weightManager()->setWeightShardingConfig(
            schema_factory.getWeightShardingConfig());
    }

    bool setupPipeline()
    {
        const auto setup_start = Clock::now();

        DeviceManager::instance().initialize(-1);
        GlobalBackendRouter::initForTests();

        ModelContextConfig model_config = ModelContextConfig::defaults();
        model_config.mpi_ctx = this->mpi_ctx_;
        model_config.strategy = WeightDistributionStrategy::REPLICATED;
        model_config.use_mmap = true;

        this->model_ctx_ = ModelContext::create(this->config_.model_path, model_config);
        if (!this->model_ctx_)
        {
            ADD_FAILURE() << "Failed to load model context for HybridPPTP named-domain parity";
            return false;
        }
        configureModel(this->model_ctx_);

        const ModelConfig plan_model = makeModelConfigFromContext(*this->model_ctx_);
        auto orchestration_config = makeNamedDomainConfig(this->config_.model_path,
                                                          plan_model.n_layers,
                                                          this->cfg().activation_precision,
                                                          this->cfg().kv_cache_precision,
                                                          /*mirrored_gpu_owner=*/false);

        auto cluster_inventory = gatherClusterInventory(this->mpi_ctx_, {}, "");

        ExecutionPlanBuilder builder;
        GlobalPPTopology topology = builder.buildGlobalPPTopology(
            orchestration_config, plan_model, cluster_inventory);

        const auto topology_errors = topology.validate();
        if (!topology_errors.empty())
        {
            for (const auto &error : topology_errors)
            {
                ADD_FAILURE() << "Named-domain topology error: " << error;
            }
            return false;
        }

        if (this->isRank0())
        {
            LOG_INFO("[HybridPPTP] Named-domain topology:\n"
                     << renderMultiDomainTopologyInfo(topology, this->mpiWorldSize()));
        }
        if (!assertNamedDomainTopologyShape(topology, /*mirrored_gpu_owner=*/false))
        {
            return false;
        }

        DomainCommunicatorRegistry domain_registry;
        bool has_global_tp = false;
        for (const auto &stage : topology.stages)
        {
            if (stage.is_global_tp)
            {
                has_global_tp = true;
                break;
            }
        }
        if (has_global_tp)
        {
            domain_registry.initialize(topology, this->mpi_ctx_->communicator(), this->mpiRank());
        }

        GlobalPPRankPlan rank_plan = GlobalPPRankPlanBuilder::build(topology, this->mpiRank());

        InferenceRunnerConfig runner_config;
        runner_config.max_seq_len = 4096;
        runner_config.batch_size = 1;
        runner_config.force_graph = true;
        runner_config.activation_precision = this->cfg().activation_precision;
        runner_config.kv_cache_precision = this->cfg().kv_cache_precision;
        runner_config.use_mapped_memory = true;

        StageBuildContext build_context;
        build_context.model_ctx = this->model_ctx_;
        build_context.mpi_ctx = this->mpi_ctx_;
        build_context.runner_config = runner_config;
        build_context.domain_registry = &domain_registry;

        std::vector<StageRunnerEntry> stage_runners;
        for (const auto &step : rank_plan.steps)
        {
            if (step.type != GlobalPPRankPlan::Step::Type::EXECUTE_STAGE ||
                step.stage_action.role != RankStageAction::Role::EXECUTE)
            {
                continue;
            }

            const auto *stage = stageById(topology, step.stage_action.stage_id);
            if (!stage)
            {
                ADD_FAILURE() << "No stage spec for action stage " << step.stage_action.stage_id;
                return false;
            }

            try
            {
                stage_runners.push_back(StageRunnerFactory::create(*stage, step.stage_action, build_context));
            }
            catch (const std::exception &e)
            {
                ADD_FAILURE() << "StageRunnerFactory failed for stage "
                              << step.stage_action.stage_id << ": " << e.what();
                return false;
            }
        }

        GlobalOrchestrator::Config global_config;
        global_config.topology = topology;
        global_config.rank = this->mpiRank();
        global_config.world_size = this->mpiWorldSize();
        global_config.mpi_ctx = this->mpi_ctx_.get();
        global_config.stage_runners = std::move(stage_runners);
        global_config.vocab_size = this->model_ctx_->vocabSize();
        global_config.d_model = this->model_ctx_->embeddingLength();
        global_config.architecture_name = this->model_ctx_->architecture();

        auto global = std::make_unique<GlobalOrchestrator>(std::move(global_config));
        this->global_orchestrator_ptr_ = global.get();
        if (!verifyLocalStageRunners(*global))
        {
            return false;
        }

        global->enableSnapshotCapture("");
        this->runner_ = std::move(global);
        logPhaseTiming("setup", setup_start, this->mpiRank());
        return true;
    }

    bool verifyLocalStageRunners(GlobalOrchestrator &global)
    {
        const int rank = this->mpiRank();
        const size_t expected_count = rank == 0 ? 2u : 1u;
        EXPECT_EQ(global.stageRunnerCount(), expected_count);
        LOG_INFO("[HybridPPTP][rank " << rank << "] owns "
                 << global.stageRunnerCount() << " stage runner(s)");
        if (global.stageRunnerCount() != expected_count)
        {
            return false;
        }

        auto *cpu_entry = global.stageRunnerEntryForDomain(kCpuDomain);
        EXPECT_NE(cpu_entry, nullptr);
        if (!cpu_entry)
        {
            return false;
        }
        EXPECT_TRUE(cpu_entry->action.is_global_tp);
        EXPECT_EQ(cpu_entry->action.backend, CollectiveBackendType::UPI);
        EXPECT_EQ(cpu_entry->action.tp_domain_size, kExpectedWorldSize);
        EXPECT_TRUE(cpu_entry->action.device.isCPU()) << cpu_entry->action.device.toString();
        EXPECT_NE(cpu_entry->global_tp_ctx, nullptr);
        if (cpu_entry->global_tp_ctx)
        {
            EXPECT_EQ(cpu_entry->global_tp_ctx->degree(), kExpectedWorldSize);
            EXPECT_EQ(cpu_entry->global_tp_ctx->backend(), CollectiveBackendType::UPI);
            EXPECT_TRUE(cpu_entry->global_tp_ctx->isNodeLocal() ||
                        cpu_entry->global_tp_ctx->isGlobal());
            EXPECT_TRUE(vectorEquals(cpu_entry->global_tp_ctx->worldRanks(), {0, 1}));
            LOG_INFO("[HybridPPTP][rank " << rank << "] CPU domain " << kCpuDomain
                     << " uses " << collectiveBackendTypeToString(cpu_entry->global_tp_ctx->backend())
                     << " TP degree=" << cpu_entry->global_tp_ctx->degree()
                     << " ranks=[0,1]");
        }

        if (rank == 0)
        {
            auto *rocm_entry = global.stageRunnerEntryForDomain(kRocmDomainRank0);
            EXPECT_NE(rocm_entry, nullptr);
            if (!rocm_entry)
            {
                return false;
            }
            EXPECT_FALSE(rocm_entry->action.is_global_tp);
            EXPECT_EQ(rocm_entry->action.inner_mode, InnerParallelism::LOCAL_TP);
            EXPECT_EQ(rocm_entry->action.backend, CollectiveBackendType::RCCL);
            EXPECT_NE(rocm_entry->local_tp_ctx, nullptr);
            EXPECT_EQ(rocm_entry->action.devices.size(), 2u);
            for (const auto &device : rocm_entry->action.devices)
            {
                EXPECT_TRUE(device.isROCm()) << device.toString();
            }
            if (rocm_entry->local_tp_ctx)
            {
                EXPECT_EQ(rocm_entry->local_tp_ctx->degree(), 2);
                EXPECT_EQ(rocm_entry->local_tp_ctx->backend(), CollectiveBackendType::RCCL);
                LOG_INFO("[HybridPPTP][rank 0] ROCm domain " << kRocmDomainRank0
                         << " uses " << collectiveBackendTypeToString(rocm_entry->local_tp_ctx->backend())
                         << " local TP over rocm:0,rocm:1");
            }
        }
        else
        {
            EXPECT_EQ(global.stageRunnerEntryForDomain(kRocmDomainRank0), nullptr);
            EXPECT_EQ(global.stageRunnerForDomain(kRocmDomainRank0), nullptr);
        }

        return !::testing::Test::HasFailure();
    }

    bool applyMasksToDomain(GlobalOrchestrator &global,
                            const std::string &domain_name,
                            const std::vector<std::vector<bool>> &masks)
    {
        auto *runner = global.stageRunnerForDomain(domain_name);
        if (!runner)
        {
            ADD_FAILURE() << "No local runner for domain " << domain_name;
            return false;
        }

        if (auto *rank = dynamic_cast<RankOrchestrator *>(runner))
        {
            std::vector<std::vector<std::vector<bool>>> masks_by_device(
                static_cast<size_t>(rank->device_count()), masks);
            rank->applyMoEExpertMasksForAllDevices(masks_by_device);
            return true;
        }

        if (auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(runner))
        {
            dgo->applyExpertMasks(masks, ReceivedWeightsMap{});
            return true;
        }

        ADD_FAILURE() << "Domain " << domain_name
                      << " runner does not expose MoE expert-mask application";
        return false;
    }

    bool warmGraphAndApplyStaticGpuExpertCache()
    {
        auto *global = dynamic_cast<GlobalOrchestrator *>(this->runner_.get());
        if (!global)
        {
            ADD_FAILURE() << "HybridPPTP test expected a GlobalOrchestrator root";
            return false;
        }

        const auto warmup_start = Clock::now();
        if (!this->runner_->forward(this->config_.token_ids.data(), this->config_.token_ids.size()))
        {
            ADD_FAILURE() << "HybridPPTP graph warmup forward failed";
            return false;
        }
        logPhaseTiming("warmup forward", warmup_start, this->mpiRank());

        const auto mask_start = Clock::now();
        const int num_layers = this->model_ctx_ ? this->model_ctx_->totalBlockCount() : 40;
        const auto moe_config = this->getMoEConfig();
        const int num_experts = moe_config.num_experts > 0 ? moe_config.num_experts : 256;
        const int cache_experts = std::clamp(
            debugEnv().moe_rebalance.gpu_cache_experts_per_layer, 0, num_experts);

        std::vector<std::vector<bool>> gpu_hot_masks(
            num_layers, std::vector<bool>(num_experts, false));
        std::vector<std::vector<bool>> cpu_cold_masks(
            num_layers, std::vector<bool>(num_experts, false));

        for (int layer = 0; layer < num_layers; ++layer)
        {
            for (int expert = 0; expert < num_experts; ++expert)
            {
                const bool cached_on_gpu = expert < cache_experts;
                gpu_hot_masks[layer][expert] = cached_on_gpu;
                cpu_cold_masks[layer][expert] = !cached_on_gpu;
            }
        }

        if (this->mpiRank() == 0)
        {
            if (!applyMasksToDomain(*global, kRocmDomainRank0, gpu_hot_masks))
            {
                return false;
            }
        }
        if (!applyMasksToDomain(*global, kCpuDomain, cpu_cold_masks))
        {
            return false;
        }
        this->mpiBarrier();
        logPhaseTiming("mask application", mask_start, this->mpiRank());

        const auto cleanup_start = Clock::now();
        this->runner_->clear_cache();
        this->runner_->clearSnapshots();
        logPhaseTiming("post-mask cache reset", cleanup_start, this->mpiRank());
        return true;
    }
};

TEST_P(Qwen35MoEHybridPPTPParityTest, PrefillParityWithGpuExpertCache)
{
    GTEST_SKIP() << kHybridPPTPGpuCacheSkipReason;

    const auto setup_start = Clock::now();
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    logPhaseTiming("test setup wrapper", setup_start, mpiRank());
    ASSERT_TRUE(warmGraphAndApplyStaticGpuExpertCache()) << "GPU expert cache mask application failed";

    const auto parity_start = Clock::now();
    auto summary = runPrefillParity();
    logPhaseTiming("prefill parity execution", parity_start, mpiRank());
    assertParity(summary);
}

TEST_P(Qwen35MoEHybridPPTPParityTest, DecodeParityWithGpuExpertCache)
{
    GTEST_SKIP() << kHybridPPTPGpuCacheSkipReason;

    const auto setup_start = Clock::now();
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    logPhaseTiming("test setup wrapper", setup_start, mpiRank());
    ASSERT_TRUE(warmGraphAndApplyStaticGpuExpertCache()) << "GPU expert cache mask application failed";

    const auto parity_start = Clock::now();
    auto summary = runDecodeParity();
    logPhaseTiming("decode parity execution", parity_start, mpiRank());
    assertDecodeParity(summary);
}

TEST_P(Qwen35MoEHybridPPTPParityTest, SnapshotInfrastructureWithGpuExpertCache)
{
    GTEST_SKIP() << kHybridPPTPGpuCacheSkipReason;

    const auto setup_start = Clock::now();
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    logPhaseTiming("test setup wrapper", setup_start, mpiRank());
    ASSERT_TRUE(warmGraphAndApplyStaticGpuExpertCache()) << "GPU expert cache mask application failed";

    const auto snapshot_start = Clock::now();
    auto embedding = loadPyTorchSnapshot("EMBEDDING");
    ASSERT_FALSE(embedding.empty()) << "Failed to load EMBEDDING snapshot";

    ASSERT_TRUE(runner_ != nullptr);
    runner_->forward(config_.token_ids.data(), config_.token_ids.size());

    auto keys = runner_->getSnapshotKeys();
    EXPECT_GT(keys.size(), 0) << "No snapshots captured";
    EXPECT_NE(std::find(keys.begin(), keys.end(), "EMBEDDING"), keys.end())
        << "Missing EMBEDDING snapshot";
    EXPECT_NE(std::find(keys.begin(), keys.end(), "LM_HEAD"), keys.end())
        << "Missing LM_HEAD snapshot";
    logPhaseTiming("snapshot infrastructure execution", snapshot_start, mpiRank());
}

INSTANTIATE_TEST_SUITE_P(
    Qwen35MoEHybridPPTP,
    Qwen35MoEHybridPPTPParityTest,
    ::testing::ValuesIn(kHybridPPTPMoEConfigs),
    [](const ::testing::TestParamInfo<TestConfig> &info)
    {
        return info.param.name;
    });

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();

    MPI_Finalize();

    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
