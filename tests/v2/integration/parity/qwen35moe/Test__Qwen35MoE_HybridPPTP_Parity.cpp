/**
 * @file Test__Qwen35MoE_HybridPPTP_Parity.cpp
 * @brief Hybrid PP+TP topology coverage for Qwen3.5 MoE expert GPU-cache migration.
 *
 * Static hot/cold expert-cache parity used to live here, but that topology
 * cannot express same-layer routed expert ownership. The graph-native
 * ExpertOverlay parity suite now covers real hot/cold expert inference. This
 * target stays as named-domain HybridPPTP topology coverage so every
 * registered test runs.
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>

#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"
#include "config/OrchestrationConfig.h"
#include "execution/mpi_orchestration/ExecutionPlanBuilder.h"
#include "planning/ClusterInventoryGatherer.h"
#include "utils/Logger.h"

#include <algorithm>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

using namespace llaminar2;

namespace
{
    constexpr const char *kRocmDomainRank0 = "rocm_socket0";
    constexpr const char *kRocmDomainRank1 = "rocm_socket1";
    constexpr const char *kCpuDomain = "cpu_sockets";
    constexpr int kExpectedWorldSize = 2;

    std::string activationPrecisionConfigValue(ActivationPrecision precision)
    {
        switch (precision)
        {
        case ActivationPrecision::BF16:
            return "bf16";
        case ActivationPrecision::FP16:
            return "fp16";
        case ActivationPrecision::Q8_1:
            return "q8_1";
        case ActivationPrecision::Q16_1:
            return "q16_1";
        case ActivationPrecision::Hybrid:
            return "hybrid";
        case ActivationPrecision::HybridQ16:
            return "hybridq16";
        case ActivationPrecision::FP32:
        default:
            return "fp32";
        }
    }

    std::string kvCachePrecisionConfigValue(KVCachePrecision precision)
    {
        switch (precision)
        {
        case KVCachePrecision::FP32:
            return "fp32";
        case KVCachePrecision::FP16:
            return "fp16";
        case KVCachePrecision::Q8_1:
            return "q8_1";
        case KVCachePrecision::Q16_1:
            return "q16_1";
        case KVCachePrecision::TQ4:
            return "tq4";
        case KVCachePrecision::TQ:
            return "tq";
        case KVCachePrecision::AUTO:
        default:
            return "auto";
        }
    }

    const GlobalPPStageSpec *stageByDomain(
        const GlobalPPTopology &topology,
        const std::string &domain_name)
    {
        for (const auto &stage : topology.stages)
        {
            if (stage.domain_name == domain_name)
                return &stage;
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
            rank0.gpus.push_back(std::move(gpu));
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

    OrchestrationConfig makeNamedDomainConfig(
        const std::string &model_path,
        int n_layers,
        ActivationPrecision activation_precision,
        KVCachePrecision kv_cache_precision,
        bool mirrored_gpu_owner)
    {
        const int split = std::max(1, n_layers / 2);
        const std::string rocm_domain =
            mirrored_gpu_owner ? kRocmDomainRank1 : kRocmDomainRank0;
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
                std::string(kCpuDomain) +
                "=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;ranks=0,1"),
        };
        config.pp_stage_definitions = {
            PPStageDefinition::parse(
                "0=" + rocm_domain + ":0-" + std::to_string(split - 1)),
            PPStageDefinition::parse(
                std::string("1=") + kCpuDomain + ":" + std::to_string(split) +
                "-" + std::to_string(n_layers - 1)),
        };
        return config;
    }

    bool assertNamedDomainTopologyShape(
        const GlobalPPTopology &topology,
        bool mirrored_gpu_owner)
    {
        const std::string rocm_domain =
            mirrored_gpu_owner ? kRocmDomainRank1 : kRocmDomainRank0;
        const int rocm_owner = mirrored_gpu_owner ? 1 : 0;
        bool ok = true;

        const auto *rocm_stage = stageByDomain(topology, rocm_domain);
        const auto *cpu_stage = stageByDomain(topology, kCpuDomain);
        EXPECT_NE(rocm_stage, nullptr);
        EXPECT_NE(cpu_stage, nullptr);
        if (!rocm_stage || !cpu_stage)
            return false;

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
}

TEST(Qwen35MoEHybridPPTPNamedDomainTopology, Rank0OwnsRocmLocalTPStage)
{
    auto config = makeNamedDomainConfig(
        "/tmp/qwen35-moe.gguf",
        40,
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
    auto config = makeNamedDomainConfig(
        "/tmp/qwen35-moe.gguf",
        40,
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
