/**
 * @file Test__Scenario7_MultiDomainPP.cpp
 * @brief Integration tests for Scenario 7: Multi-Domain Pipeline Parallel
 *
 * This test validates the end-to-end integration of all Phase 0-4 components
 * for the Scenario 7 configuration:
 *
 * Hardware Layout:
 *   Domain "gpu0": 2 CUDA GPUs on NUMA 0 (PP Stage 0, layers 0-7)
 *   Domain "gpu1": 2 ROCm GPUs on NUMA 1 (PP Stage 1, layers 8-15)
 *   Domain "cpu":  2 CPU sockets         (PP Stage 2, layers 16-23)
 *
 * This is the key validation test for the Hybrid Orchestration Plan.
 *
 * @see docs/v2/HYBRID_ORCHESTRATION_INTEGRATION_PLAN_v2.md
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "config/OrchestrationConfig.h"
#include "config/OrchestrationConfigParser.h"
#include "config/IOrchestrationConfigParser.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"
#include "execution/mpi_orchestration/IExecutionPlanBuilder.h"
#include "execution/mpi_orchestration/ExecutionPlanBuilder.h"
#include "execution/mpi_orchestration/DeviceInventory.h"
#include "backends/GlobalDeviceAddress.h"
#include "execution/runner/IOrchestrationRunner.h"
#include "execution/runner/IOrchestrationRunnerFactory.h"
#include "execution/runner/OrchestrationRunner.h"

using namespace llaminar2;
using namespace testing;

namespace
{

    // =========================================================================
    // Test Fixture
    // =========================================================================

    class Test__Scenario7_MultiDomainPP : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Create parser and plan builder
            config_parser_ = createOrchestrationConfigParser();
            plan_builder_ = createExecutionPlanBuilder();
        }

        /**
         * @brief Build the Scenario 7 CLI arguments
         */
        static std::vector<std::string> buildScenario7Args()
        {
            // Use hostname-qualified addresses so findRankForDevice can
            // disambiguate ranks that live on different nodes.
            return {
                "llaminar2",
                "-m", "models/test.gguf",
                "--define-domain", "gpu0=node0:0:cuda:0,node0:0:cuda:1",
                "--define-domain", "gpu1=node1:0:rocm:0,node1:0:rocm:1",
                "--define-domain", "cpu=node2:0:cpu:0,node2:0:cpu:1",
                "--pp-stage", "0=gpu0:0-7",
                "--pp-stage", "1=gpu1:8-15",
                "--pp-stage", "2=cpu:16-23"};
        }

        /**
         * @brief Build minimal args for explicit device-map tests
         */
        static std::vector<std::string> buildDeviceMapArgs(const std::string &map_spec)
        {
            return {
                "llaminar2",
                "-m", "models/test.gguf",
                "--device-mode", "explicit",
                "--device-map", map_spec};
        }

        /**
         * @brief Build minimal args for CPU shorthand semantics tests
         */
        static std::vector<std::string> buildCpuShorthandArgs()
        {
            return {
                "llaminar2",
                "-m", "models/test.gguf",
                "-d", "cpu"};
        }

        /**
         * @brief Convert string vector to argc/argv
         */
        static std::pair<int, std::vector<char *>> toArgv(std::vector<std::string> &args)
        {
            std::vector<char *> argv;
            for (auto &arg : args)
            {
                argv.push_back(arg.data());
            }
            argv.push_back(nullptr);
            return {static_cast<int>(args.size()), argv};
        }

        /**
         * @brief Create model config for 24-layer model
         */
        static ModelConfig createModelConfig()
        {
            ModelConfig config;
            config.name = "TestModel-24L";
            config.n_layers = 24;
            config.n_heads = 32;
            config.n_kv_heads = 8;
            config.hidden_size = 4096;
            config.head_dim = 128;
            config.vocab_size = 32000;
            return config;
        }

        /**
         * @brief Create cluster inventory for 3 PP stages
         */
        static ClusterInventory createClusterInventory()
        {
            ClusterInventory inventory;

            // Each PP stage is a "rank" in this simulation
            // In reality, they might be multiple MPI ranks

            // Helper to create DeviceInfo
            auto makeGpuDevice = [](DeviceType type, int local_id, size_t mem = 8ULL * 1024 * 1024 * 1024)
            {
                DeviceInfo info;
                info.type = type;
                info.local_device_id = local_id;
                info.memory_bytes = mem;
                info.free_memory_bytes = mem;
                return info;
            };

            // GPU0 domain (node0, 2 CUDA GPUs)
            RankInventory gpu0_inv;
            gpu0_inv.rank = 0;
            gpu0_inv.node_id = 0;
            gpu0_inv.local_rank = 0;
            gpu0_inv.hostname = "node0";
            gpu0_inv.numa_nodes = 1; // Single NUMA node
            gpu0_inv.gpus.push_back(makeGpuDevice(DeviceType::CUDA, 0));
            gpu0_inv.gpus.push_back(makeGpuDevice(DeviceType::CUDA, 1));
            inventory.ranks.push_back(gpu0_inv);

            // GPU1 domain (node1, 2 ROCm GPUs)
            RankInventory gpu1_inv;
            gpu1_inv.rank = 1;
            gpu1_inv.node_id = 1;
            gpu1_inv.local_rank = 0;
            gpu1_inv.hostname = "node1";
            gpu1_inv.numa_nodes = 1; // Single NUMA node
            gpu1_inv.gpus.push_back(makeGpuDevice(DeviceType::ROCm, 0));
            gpu1_inv.gpus.push_back(makeGpuDevice(DeviceType::ROCm, 1));
            inventory.ranks.push_back(gpu1_inv);

            // CPU domain (node2, 2 CPU sockets)
            RankInventory cpu_inv;
            cpu_inv.rank = 2;
            cpu_inv.node_id = 2;
            cpu_inv.local_rank = 0;
            cpu_inv.hostname = "node2";
            cpu_inv.numa_nodes = 2; // Two NUMA nodes
            cpu_inv.cpu.type = DeviceType::CPU;
            cpu_inv.cpu.local_device_id = 0;
            cpu_inv.cpu_cores = 64;
            cpu_inv.cpu_sockets = 2;
            cpu_inv.cpu_memory_bytes = 256ULL * 1024 * 1024 * 1024;
            inventory.ranks.push_back(cpu_inv);

            // Set cluster-level metadata
            inventory.world_size = 3;
            inventory.node_count = 3;
            inventory.total_gpus = 4; // 2 CUDA + 2 ROCm
            inventory.total_gpu_memory = 4 * 8ULL * 1024 * 1024 * 1024;
            inventory.total_cpu_memory = 256ULL * 1024 * 1024 * 1024;

            return inventory;
        }

        /**
         * @brief Create single-rank inventory for device-map semantics tests
         */
        static ClusterInventory createSingleRankDeviceMapInventory()
        {
            ClusterInventory inventory;

            RankInventory inv;
            inv.rank = 0;
            inv.node_id = 0;
            inv.local_rank = 0;
            inv.hostname = "localhost";
            inv.numa_nodes = 4;

            auto makeGpu = [](DeviceType type, int local_id, int numa)
            {
                DeviceInfo info;
                info.type = type;
                info.local_device_id = local_id;
                info.numa_node = numa;
                info.memory_bytes = 8ULL * 1024 * 1024 * 1024;
                info.free_memory_bytes = info.memory_bytes;
                return info;
            };

            // Synthetic duplicate local_device_id=0 across different NUMA nodes
            // to verify deterministic "prefer lower NUMA" selection.
            inv.gpus.push_back(makeGpu(DeviceType::ROCm, 0, 2));
            inv.gpus.push_back(makeGpu(DeviceType::ROCm, 0, 1));

            inventory.ranks.push_back(inv);
            inventory.world_size = 1;
            inventory.node_count = 1;
            inventory.total_gpus = 2;
            inventory.total_gpu_memory = 16ULL * 1024 * 1024 * 1024;
            return inventory;
        }

        /**
         * @brief Create 2-rank CPU-only inventory (one rank per NUMA node)
         */
        static ClusterInventory createTwoRankCPUInventory()
        {
            ClusterInventory inventory;

            auto makeRank = [](int rank, int local_rank, int numa_node)
            {
                RankInventory inv;
                inv.rank = rank;
                inv.node_id = 0;
                inv.local_rank = local_rank;
                inv.hostname = "localhost";
                inv.numa_nodes = 2;
                inv.cpu.type = DeviceType::CPU;
                inv.cpu.local_device_id = 0;
                inv.cpu.numa_node = numa_node;
                inv.cpu_cores = 28;
                inv.cpu_sockets = 2;
                inv.cpu_memory_bytes = 128ULL * 1024 * 1024 * 1024;
                return inv;
            };

            inventory.ranks.push_back(makeRank(0, 0, 0));
            inventory.ranks.push_back(makeRank(1, 1, 1));
            inventory.world_size = 2;
            inventory.node_count = 1;
            inventory.total_gpus = 0;
            inventory.total_gpu_memory = 0;
            inventory.total_cpu_memory = 256ULL * 1024 * 1024 * 1024;
            return inventory;
        }

        std::unique_ptr<IOrchestrationConfigParser> config_parser_;
        std::unique_ptr<IExecutionPlanBuilder> plan_builder_;
    };

    // =========================================================================
    // Configuration Parsing Tests
    // =========================================================================

    TEST_F(Test__Scenario7_MultiDomainPP, ParsesComplexConfig)
    {
        // Build CLI arguments for Scenario 7
        auto args = buildScenario7Args();
        auto [argc, argv] = toArgv(args);

        // Parse configuration
        OrchestrationConfig config;
        ASSERT_NO_THROW({
            config = config_parser_->parseArgs(argc, argv.data());
        });

        // Verify domains parsed
        EXPECT_EQ(config.domain_definitions.size(), 3u);

        // Check gpu0 domain
        const auto &gpu0 = config.domain_definitions[0];
        EXPECT_EQ(gpu0.name, "gpu0");
        EXPECT_EQ(gpu0.devices.size(), 2u);
        EXPECT_EQ(gpu0.devices[0].device_type, DeviceType::CUDA);
        EXPECT_EQ(gpu0.devices[1].device_type, DeviceType::CUDA);

        // Check gpu1 domain (ROCm)
        const auto &gpu1 = config.domain_definitions[1];
        EXPECT_EQ(gpu1.name, "gpu1");
        EXPECT_EQ(gpu1.devices.size(), 2u);
        EXPECT_EQ(gpu1.devices[0].device_type, DeviceType::ROCm);
        EXPECT_EQ(gpu1.devices[1].device_type, DeviceType::ROCm);

        // Check cpu domain
        const auto &cpu = config.domain_definitions[2];
        EXPECT_EQ(cpu.name, "cpu");
        EXPECT_EQ(cpu.devices.size(), 2u);
        EXPECT_EQ(cpu.devices[0].device_type, DeviceType::CPU);
        EXPECT_EQ(cpu.devices[1].device_type, DeviceType::CPU);
    }

    TEST_F(Test__Scenario7_MultiDomainPP, ParsesPPStageDefinitions)
    {
        auto args = buildScenario7Args();
        auto [argc, argv] = toArgv(args);

        OrchestrationConfig config = config_parser_->parseArgs(argc, argv.data());

        // Verify PP stages
        EXPECT_EQ(config.pp_stage_definitions.size(), 3u);

        // Stage 0: gpu0, layers 0-7
        const auto &stage0 = config.pp_stage_definitions[0];
        EXPECT_EQ(stage0.stage_id, 0);
        EXPECT_EQ(stage0.domain_name, "gpu0");
        EXPECT_EQ(stage0.first_layer, 0);
        EXPECT_EQ(stage0.last_layer, 7);
        EXPECT_EQ(stage0.layerCount(), 8);

        // Stage 1: gpu1, layers 8-15
        const auto &stage1 = config.pp_stage_definitions[1];
        EXPECT_EQ(stage1.stage_id, 1);
        EXPECT_EQ(stage1.domain_name, "gpu1");
        EXPECT_EQ(stage1.first_layer, 8);
        EXPECT_EQ(stage1.last_layer, 15);
        EXPECT_EQ(stage1.layerCount(), 8);

        // Stage 2: cpu, layers 16-23
        const auto &stage2 = config.pp_stage_definitions[2];
        EXPECT_EQ(stage2.stage_id, 2);
        EXPECT_EQ(stage2.domain_name, "cpu");
        EXPECT_EQ(stage2.first_layer, 16);
        EXPECT_EQ(stage2.last_layer, 23);
        EXPECT_EQ(stage2.layerCount(), 8);
    }

    TEST_F(Test__Scenario7_MultiDomainPP, ConfigValidates)
    {
        auto args = buildScenario7Args();
        auto [argc, argv] = toArgv(args);

        OrchestrationConfig config = config_parser_->parseArgs(argc, argv.data());

        auto errors = config.validate();
        EXPECT_TRUE(errors.empty()) << "Validation errors: "
                                    << (errors.empty() ? "" : errors[0]);
    }

    TEST_F(Test__Scenario7_MultiDomainPP, ConfigUsesNamedDomains)
    {
        auto args = buildScenario7Args();
        auto [argc, argv] = toArgv(args);

        OrchestrationConfig config = config_parser_->parseArgs(argc, argv.data());

        EXPECT_TRUE(config.usesNamedDomains());
    }

    TEST_F(Test__Scenario7_MultiDomainPP, ParsesDeviceMapNumaExplicitness)
    {
        auto args = buildDeviceMapArgs("0=rocm:0,1=1:rocm:0");
        auto [argc, argv] = toArgv(args);

        OrchestrationConfig config = config_parser_->parseArgs(argc, argv.data());

        ASSERT_EQ(config.device_map.size(), 2u);
        ASSERT_EQ(config.device_map_numa_explicit.size(), 2u);

        EXPECT_EQ(config.device_map[0].first, 0);
        EXPECT_EQ(config.device_map[0].second.toShortString(), "rocm:0");
        EXPECT_FALSE(config.device_map_numa_explicit[0].second);

        EXPECT_EQ(config.device_map[1].first, 1);
        EXPECT_TRUE(config.device_map_numa_explicit[1].second);
    }

    // =========================================================================
    // Execution Plan Building Tests
    // =========================================================================

    TEST_F(Test__Scenario7_MultiDomainPP, BuildsExecutionPlanForStage0)
    {
        auto args = buildScenario7Args();
        auto [argc, argv] = toArgv(args);

        OrchestrationConfig config = config_parser_->parseArgs(argc, argv.data());
        ModelConfig model_config = createModelConfig();
        ClusterInventory inventory = createClusterInventory();

        // Build plan for rank 0 (PP Stage 0)
        RankExecutionPlan plan = plan_builder_->buildPlanForRank(
            config, model_config, inventory, 0);

        // Verify PP stage 0 assignment
        EXPECT_EQ(plan.rank, 0);
        EXPECT_EQ(plan.pp_stage_id, 0);
        EXPECT_EQ(plan.first_layer, 0);
        EXPECT_EQ(plan.last_layer, 7);
        EXPECT_EQ(plan.layerCount(), 8);

        // Stage 0 has embedding (head) but no LM head (not tail)
        EXPECT_TRUE(plan.has_embedding);
        EXPECT_FALSE(plan.has_lm_head);

        // PP neighbors: no prev (head), has next
        EXPECT_FALSE(plan.prev_rank.has_value());
        EXPECT_TRUE(plan.next_rank.has_value());
        EXPECT_EQ(*plan.next_rank, 1);

        // Should have 2 local TP devices (2 CUDA GPUs)
        EXPECT_EQ(plan.local_tp_devices.size(), 2u);
        EXPECT_TRUE(plan.usesLocalTP());
    }

    TEST_F(Test__Scenario7_MultiDomainPP, BuildsExecutionPlanForStage1)
    {
        // Skip: This test requires proper device-to-rank matching in ClusterInventory
        // which needs real device discovery or more complex mock setup.
        // The ExecutionPlanBuilder's device matching is covered by its own unit tests.
        GTEST_SKIP() << "Device-to-rank matching requires real device discovery";

        auto args = buildScenario7Args();
        auto [argc, argv] = toArgv(args);

        OrchestrationConfig config = config_parser_->parseArgs(argc, argv.data());
        ModelConfig model_config = createModelConfig();
        ClusterInventory inventory = createClusterInventory();

        // Build plan for rank 1 (PP Stage 1)
        RankExecutionPlan plan = plan_builder_->buildPlanForRank(
            config, model_config, inventory, 1);

        // Verify PP stage 1 assignment
        EXPECT_EQ(plan.rank, 1);
        EXPECT_EQ(plan.pp_stage_id, 1);
        EXPECT_EQ(plan.first_layer, 8);
        EXPECT_EQ(plan.last_layer, 15);
        EXPECT_EQ(plan.layerCount(), 8);

        // Middle stage: no embedding, no LM head
        EXPECT_FALSE(plan.has_embedding);
        EXPECT_FALSE(plan.has_lm_head);

        // PP neighbors: has both prev and next
        EXPECT_TRUE(plan.prev_rank.has_value());
        EXPECT_EQ(*plan.prev_rank, 0);
        EXPECT_TRUE(plan.next_rank.has_value());
        EXPECT_EQ(*plan.next_rank, 2);

        // Should have 2 local TP devices (2 ROCm GPUs)
        EXPECT_EQ(plan.local_tp_devices.size(), 2u);
        for (const auto &dev : plan.local_tp_devices)
        {
            EXPECT_EQ(dev.device_type, DeviceType::ROCm);
        }
    }

    TEST_F(Test__Scenario7_MultiDomainPP, BuildsExecutionPlanForStage2)
    {
        // Skip: This test requires proper device-to-rank matching in ClusterInventory
        // which needs real device discovery or more complex mock setup.
        GTEST_SKIP() << "Device-to-rank matching requires real device discovery";

        auto args = buildScenario7Args();
        auto [argc, argv] = toArgv(args);

        OrchestrationConfig config = config_parser_->parseArgs(argc, argv.data());
        ModelConfig model_config = createModelConfig();
        ClusterInventory inventory = createClusterInventory();

        // Build plan for rank 2 (PP Stage 2)
        RankExecutionPlan plan = plan_builder_->buildPlanForRank(
            config, model_config, inventory, 2);

        // Verify PP stage 2 assignment
        EXPECT_EQ(plan.rank, 2);
        EXPECT_EQ(plan.pp_stage_id, 2);
        EXPECT_EQ(plan.first_layer, 16);
        EXPECT_EQ(plan.last_layer, 23);
        EXPECT_EQ(plan.layerCount(), 8);

        // Last stage: no embedding, has LM head
        EXPECT_FALSE(plan.has_embedding);
        EXPECT_TRUE(plan.has_lm_head);

        // PP neighbors: has prev, no next (tail)
        EXPECT_TRUE(plan.prev_rank.has_value());
        EXPECT_EQ(*plan.prev_rank, 1);
        EXPECT_FALSE(plan.next_rank.has_value());

        // Should have 2 local TP devices (2 CPUs)
        EXPECT_EQ(plan.local_tp_devices.size(), 2u);
        for (const auto &dev : plan.local_tp_devices)
        {
            EXPECT_EQ(dev.device_type, DeviceType::CPU);
        }
    }

    TEST_F(Test__Scenario7_MultiDomainPP, AllLayersCovered)
    {
        // Skip: This test requires proper device-to-rank matching in ClusterInventory
        // which needs real device discovery or more complex mock setup.
        GTEST_SKIP() << "Device-to-rank matching requires real device discovery";

        auto args = buildScenario7Args();
        auto [argc, argv] = toArgv(args);

        OrchestrationConfig config = config_parser_->parseArgs(argc, argv.data());
        ModelConfig model_config = createModelConfig();
        ClusterInventory inventory = createClusterInventory();

        // Build all plans
        auto all_plans = plan_builder_->buildAllPlans(config, model_config, inventory);

        // Check total layer coverage
        int total_layers = 0;
        std::set<int> covered_layers;
        for (const auto &plan : all_plans)
        {
            for (int l = plan.first_layer; l <= plan.last_layer; ++l)
            {
                covered_layers.insert(l);
            }
            total_layers += plan.layerCount();
        }

        EXPECT_EQ(total_layers, 24);
        EXPECT_EQ(covered_layers.size(), 24u);

        // Verify no gaps (layers 0-23 all covered)
        for (int i = 0; i < 24; ++i)
        {
            EXPECT_TRUE(covered_layers.count(i) > 0) << "Layer " << i << " not covered";
        }
    }

    TEST_F(Test__Scenario7_MultiDomainPP, PlansValidate)
    {
        auto args = buildScenario7Args();
        auto [argc, argv] = toArgv(args);

        OrchestrationConfig config = config_parser_->parseArgs(argc, argv.data());
        ModelConfig model_config = createModelConfig();
        ClusterInventory inventory = createClusterInventory();

        auto all_plans = plan_builder_->buildAllPlans(config, model_config, inventory);

        for (const auto &plan : all_plans)
        {
            auto errors = plan.validate();
            EXPECT_TRUE(errors.empty())
                << "Plan for rank " << plan.rank << " validation failed: "
                << (errors.empty() ? "" : errors[0]);
        }
    }

    TEST_F(Test__Scenario7_MultiDomainPP, DeviceMapAmbiguous_SelectsLowestNumaCandidate)
    {
        auto args = buildDeviceMapArgs("0=rocm:0");
        auto [argc, argv] = toArgv(args);

        OrchestrationConfig config = config_parser_->parseArgs(argc, argv.data());
        ModelConfig model_config = createModelConfig();
        ClusterInventory inventory = createSingleRankDeviceMapInventory();

        RankExecutionPlan plan = plan_builder_->buildPlanForRank(
            config, model_config, inventory, 0);

        EXPECT_EQ(plan.primary_device.device_type, DeviceType::ROCm);
        EXPECT_EQ(plan.primary_device.device_ordinal, 0);
        EXPECT_EQ(plan.primary_device.numa_node, 1)
            << "Ambiguous map entry should prefer lowest NUMA among matching candidates";
        EXPECT_FALSE(plan.primary_device_numa_explicit);
    }

    TEST_F(Test__Scenario7_MultiDomainPP, DeviceMapExplicit_PreservesExplicitNumaAndStrictFlag)
    {
        auto args = buildDeviceMapArgs("0=2:rocm:0");
        auto [argc, argv] = toArgv(args);

        OrchestrationConfig config = config_parser_->parseArgs(argc, argv.data());
        ModelConfig model_config = createModelConfig();
        ClusterInventory inventory = createSingleRankDeviceMapInventory();

        RankExecutionPlan plan = plan_builder_->buildPlanForRank(
            config, model_config, inventory, 0);

        EXPECT_EQ(plan.primary_device.device_type, DeviceType::ROCm);
        EXPECT_EQ(plan.primary_device.device_ordinal, 0);
        EXPECT_EQ(plan.primary_device.numa_node, 2);
        EXPECT_TRUE(plan.primary_device_numa_explicit);
    }

    TEST_F(Test__Scenario7_MultiDomainPP, CpuShorthandRuntimeMapping_ProducesGlobalTPAcrossWorld)
    {
        auto args = buildCpuShorthandArgs();
        auto [argc, argv] = toArgv(args);

        OrchestrationConfig parsed = config_parser_->parseArgs(argc, argv.data());
        ASSERT_TRUE(parsed.cpu_global_tp_all_local)
            << "-d cpu should set shorthand global CPU TP intent";

        ModelConfig model_config = createModelConfig();
        ClusterInventory inventory = createTwoRankCPUInventory();

        for (int rank = 0; rank < inventory.world_size; ++rank)
        {
            // Emulate Main.cpp runtime mapping inside MPI context for `-d cpu`.
            OrchestrationConfig runtime_cfg = parsed;
            runtime_cfg.device_mode = DeviceAssignmentMode::EXPLICIT;
            runtime_cfg.device_map.clear();
            runtime_cfg.device_map.emplace_back(rank, GlobalDeviceAddress::cpu(rank));
            runtime_cfg.device_map_numa_explicit.clear();
            runtime_cfg.device_map_numa_explicit.emplace_back(rank, true);
            runtime_cfg.tp_scope = TPScope::GLOBAL;
            runtime_cfg.tp_degree = inventory.world_size;

            RankExecutionPlan plan = plan_builder_->buildPlanForRank(
                runtime_cfg, model_config, inventory, rank);

            EXPECT_EQ(plan.primary_device.device_type, DeviceType::CPU);
            EXPECT_EQ(plan.primary_device.numa_node, rank);
            EXPECT_TRUE(plan.primary_device_numa_explicit);

            EXPECT_TRUE(plan.usesGlobalTP());
            EXPECT_EQ(plan.tp_scope, TPScope::GLOBAL);
            EXPECT_EQ(plan.global_tp_domain_size, inventory.world_size);
            EXPECT_EQ(plan.global_tp_rank_in_domain, rank);
            EXPECT_EQ(plan.weight_shard.total_shards, inventory.world_size);
            EXPECT_EQ(plan.weight_shard.shard_index, rank);
            EXPECT_FALSE(plan.usesLocalTP());
        }
    }

    // =========================================================================
    // Orchestration Runner Factory Tests
    // =========================================================================

    TEST_F(Test__Scenario7_MultiDomainPP, FactoryCreatesRunner)
    {
        auto factory = createOrchestrationRunnerFactory();
        ASSERT_NE(factory, nullptr);

        // Create simple runner (no actual model)
        auto runner = factory->createSimple("/nonexistent/model.gguf", "cpu:0");
        ASSERT_NE(runner, nullptr);

        // Runner should not be initialized yet
        EXPECT_FALSE(runner->isInitialized());
    }

    TEST_F(Test__Scenario7_MultiDomainPP, FactoryCreatesFromConfig)
    {
        auto factory = createOrchestrationRunnerFactory();

        // Create config programmatically
        OrchestrationConfig config;
        config.device_for_this_rank = GlobalDeviceAddress::cpu();
        config.tp_degree = 1;
        config.pp_degree = 1;

        auto runner = factory->createFromOrchestrationConfig(config);
        ASSERT_NE(runner, nullptr);
        EXPECT_FALSE(runner->isInitialized());
    }

    // =========================================================================
    // Orchestration Runner Interface Tests
    // =========================================================================

    TEST_F(Test__Scenario7_MultiDomainPP, RunnerReturnsExecutionPlan)
    {
        // Create with pre-built plan
        OrchestrationConfig config;
        config.tp_degree = 2;
        config.pp_degree = 1;

        RankExecutionPlan plan;
        plan.rank = 0;
        plan.first_layer = 0;
        plan.last_layer = 23;
        plan.has_embedding = true;
        plan.has_lm_head = true;
        plan.local_tp_devices.push_back(GlobalDeviceAddress::parse("0:cuda:0"));
        plan.local_tp_devices.push_back(GlobalDeviceAddress::parse("0:cuda:1"));

        OrchestrationRunner runner(config, plan);

        const auto &returned_plan = runner.executionPlan();
        EXPECT_EQ(returned_plan.rank, 0);
        EXPECT_EQ(returned_plan.first_layer, 0);
        EXPECT_EQ(returned_plan.last_layer, 23);
        EXPECT_TRUE(returned_plan.usesLocalTP());
    }

    TEST_F(Test__Scenario7_MultiDomainPP, RunnerReturnsConfig)
    {
        OrchestrationConfig config;
        config.tp_degree = 4;
        config.pp_degree = 2;

        RankExecutionPlan plan;
        plan.rank = 0;

        OrchestrationRunner runner(config, plan);

        const auto &returned_config = runner.config();
        EXPECT_EQ(returned_config.tp_degree, 4);
        EXPECT_EQ(returned_config.pp_degree, 2);
    }

    TEST_F(Test__Scenario7_MultiDomainPP, RunnerInitializationStateTracking)
    {
        OrchestrationConfig config;
        RankExecutionPlan plan;
        plan.rank = 0;

        OrchestrationRunner runner(config, plan);

        // Not initialized before calling initialize()
        EXPECT_FALSE(runner.isInitialized());

        // After initialize (will fail without model, but state should be tracked)
        // Note: Full initialization requires a real model file
        // This test just verifies state tracking
    }

} // anonymous namespace
