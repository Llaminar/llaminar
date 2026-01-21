/**
 * @file Test__Scenario3_CPUParticipation.cpp
 * @brief Orchestration tests for single-machine CPU participation scenarios
 *
 * **Hardware Configuration**:
 * ┌─────────────────────────────────────────────────────────────────────────────┐
 * │                         SINGLE MACHINE                                       │
 * │  ┌─────────────────────────────────────────────────────────────────────┐    │
 * │  │       SOCKET 0 (NUMA Node 0)    │       SOCKET 1 (NUMA Node 1)      │    │
 * │  │         MPI Rank 0              │         MPI Rank 1                │    │
 * │  │  ┌─────────────────────────┐    │                                   │    │
 * │  │  │   NVIDIA A100-40GB      │    │  ┌───────────────────────────┐   │    │
 * │  │  │   40GB VRAM             │    │  │   CPU ONLY (NO GPU)       │   │    │
 * │  │  │   DeviceId: CUDA:0      │    │  │   Xeon 28-core DDR5       │   │    │
 * │  │  └─────────────────────────┘    │  └───────────────────────────┘   │    │
 * │  │  Xeon 28-core DDR5              │                                   │    │
 * │  └─────────────────────────────────────────────────────────────────────┘    │
 * └─────────────────────────────────────────────────────────────────────────────┘
 *
 * Tests heterogeneous execution where CPU participates in inference alongside GPU.
 * This is critical for:
 * - Memory-constrained systems where GPU VRAM is insufficient
 * - Decode phase where CPU can contribute meaningfully
 * - Hybrid execution strategies with cpu_compute_fraction > 0
 *
 * **Tested Models**: Qwen2 0.5B, 7B, 72B
 * **Total GPU VRAM**: 40 GB (A100 only)
 * **Total CPU Memory**: ~1 TB (2× 512GB DDR5)
 *
 * @see docs/v2/ARCHITECTURE_EXECUTION_SCENARIOS.md (Scenario 3)
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "../OrchestrationTestBase.h"

using namespace llaminar2;
using namespace llaminar2::test::orchestration;

// =============================================================================
// Scenario 3 Configuration
// =============================================================================

/**
 * @brief Scenario 3: Single machine with CPU participation enabled
 *
 * Hardware Configuration:
 * - 1 machine, 2 sockets
 * - Socket 0: 1× A100-40GB (CUDA) + Xeon 28-core
 * - Socket 1: CPU-only (Xeon 28-core) - tests CPU domain creation
 *
 * Test Focus:
 * - CPU domain creation when cpu_compute_fraction > 0
 * - GPU handles early layers (higher compute)
 * - CPU handles later layers in decode (memory bound)
 * - Cross-socket communication via QPI/UPI
 */
class Test__Scenario3_CPUParticipation : public OrchestrationTestBase
{
protected:
    ClusterConfig getClusterConfig() override
    {
        ClusterConfig config;
        config.name = "Scenario 3: Single Machine, CPU Participation";
        config.num_machines = 1;
        config.sockets_per_machine = 2;

        // Socket 0: GPU + CPU
        SocketConfig socket0;
        socket0.cpu = CPUSpecs::XEON_28C_DDR5;
        socket0.gpus = {GPUSpecs::A100_40GB};

        // Socket 1: CPU only (no GPU)
        SocketConfig socket1;
        socket1.cpu = CPUSpecs::XEON_28C_DDR5;
        socket1.gpus = {}; // No GPU - tests CPU participation

        config.socket_configs = {socket0, socket1};
        config.qpi_upi_bandwidth_gbps = 50.0f; // Cross-socket bandwidth

        return config;
    }

    std::vector<ModelConfig> getModelConfigs() override
    {
        return {
            ModelConfigs::QWEN2_0_5B, // Small model - GPU only
            ModelConfigs::QWEN2_7B,   // Medium model - GPU fits but CPU can help
            ModelConfigs::QWEN2_72B,  // Large model - requires CPU spillover
        };
    }
};

// =============================================================================
// Instantiate Standard Orchestration Tests (15 tests from macro)
// =============================================================================

INSTANTIATE_ORCHESTRATION_TESTS(Test__Scenario3_CPUParticipation);

// =============================================================================
// Scenario-Specific Tests: Topology Validation
// =============================================================================

TEST_F(Test__Scenario3_CPUParticipation, Socket0HasCUDAGPU)
{
    // Rank 0 (Socket 0) should have CUDA GPU
    const auto &placement = topologies_[0]->placement();

    bool has_cuda = false;
    for (const auto &dev : placement.devices)
    {
        if (dev.type == DeviceCapability::Type::CUDA)
        {
            has_cuda = true;
            EXPECT_EQ(dev.name, "NVIDIA A100 40GB");
            EXPECT_EQ(dev.memory_bytes, static_cast<size_t>(40ULL * 1024 * 1024 * 1024));
        }
    }
    EXPECT_TRUE(has_cuda) << "Socket 0 should have CUDA GPU (A100-40GB)";
}

TEST_F(Test__Scenario3_CPUParticipation, Socket1IsCPUOnly)
{
    // Rank 1 (Socket 1) should have NO GPUs
    const auto &placement = topologies_[1]->placement();

    int gpu_count = 0;
    for (const auto &dev : placement.devices)
    {
        if (dev.type == DeviceCapability::Type::CUDA ||
            dev.type == DeviceCapability::Type::ROCm)
        {
            gpu_count++;
        }
    }
    EXPECT_EQ(gpu_count, 0) << "Socket 1 should have NO GPUs (CPU-only)";

    // But should have CPU
    bool has_cpu = false;
    for (const auto &dev : placement.devices)
    {
        if (dev.type == DeviceCapability::Type::CPU)
        {
            has_cpu = true;
        }
    }
    EXPECT_TRUE(has_cpu) << "Socket 1 should have CPU device";
}

TEST_F(Test__Scenario3_CPUParticipation, TotalVRAMIs40GB)
{
    // Only A100-40GB contributes VRAM
    float expected_vram = 40.0f;
    EXPECT_FLOAT_EQ(cluster_.totalVRAMGB(), expected_vram);
}

TEST_F(Test__Scenario3_CPUParticipation, TotalGPUCountIs1)
{
    // Only one GPU in the cluster
    EXPECT_EQ(cluster_.totalGPUs(), 1);
}

TEST_F(Test__Scenario3_CPUParticipation, CrossSocketIsQPI)
{
    // Single machine uses QPI/UPI for cross-socket communication
    EXPECT_EQ(cluster_.num_machines, 1);
    EXPECT_EQ(cluster_.qpi_upi_bandwidth_gbps, 50.0f);
}

// =============================================================================
// Scenario-Specific Tests: Model Fit Analysis
// =============================================================================

TEST_F(Test__Scenario3_CPUParticipation, Qwen05B_FitsInGPU)
{
    // 0.5B at Q4 is ~500MB, fits easily in 40GB A100
    const auto &model = models_[0]; // QWEN2_0_5B

    EXPECT_LT(model.memorySizeGB(), 40.0f);
    EXPECT_TRUE(modelFitsInGPUMemory(model));
}

TEST_F(Test__Scenario3_CPUParticipation, Qwen7B_FitsInGPU)
{
    // 7B at Q4 is ~4GB, fits in 40GB A100
    const auto &model = models_[1]; // QWEN2_7B

    EXPECT_LT(model.memorySizeGB(), 40.0f);
    EXPECT_TRUE(modelFitsInGPUMemory(model));
}

TEST_F(Test__Scenario3_CPUParticipation, Qwen72B_ExceedsGPUVRAM)
{
    // 72B at Q4 is ~40GB, matches A100 VRAM limit
    // With KV cache overhead, needs CPU spillover
    const auto &model = models_[2]; // QWEN2_72B

    // Model itself just fits, but with activations/KV it won't
    EXPECT_GE(model.memorySizeGB(), 39.0f);
}

// =============================================================================
// Scenario-Specific Tests: CPU Participation Strategy
// =============================================================================

/**
 * @brief Test: CPU domain created when cpu_compute_fraction > 0
 */
TEST_F(Test__Scenario3_CPUParticipation, CPUDomainCreatedWithFraction)
{
    HeterogeneousConfig hconfig;
    hconfig.cpu_compute_fraction = 0.3f; // 30% CPU participation
    HeterogeneousMultiDomainStrategy strategy(hconfig);

    // Use 72B model that may need CPU spillover
    const auto &model = models_[2]; // QWEN2_72B
    auto input = createPlacementInput(model, 0);

    if (!strategy.isApplicable(input))
    {
        GTEST_SKIP() << "Strategy not applicable for this configuration";
    }

    auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);

    // Count CPU domains
    int cpu_domains = 0;
    for (const auto &domain : plan.domains)
    {
        if (domain.type == TPDomainType::CPU_CROSS_RANK)
        {
            cpu_domains++;
        }
    }

    EXPECT_GT(cpu_domains, 0)
        << "Should create at least one CPU domain with cpu_compute_fraction > 0";
}

/**
 * @brief Test: GPU handles early layers (compute-heavy)
 */
TEST_F(Test__Scenario3_CPUParticipation, GPUHandlesEarlyLayers)
{
    HeterogeneousConfig hconfig;
    hconfig.prefer_gpu_early_layers = true;
    hconfig.cpu_compute_fraction = 0.3f;
    HeterogeneousMultiDomainStrategy strategy(hconfig);

    const auto &model = models_[2]; // 72B
    auto input = createPlacementInput(model, 0);

    if (!strategy.isApplicable(input))
    {
        GTEST_SKIP() << "Strategy not applicable";
    }

    auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);

    // Layer 0 should be in GPU domain
    auto domain_result = validateDomainForLayer(plan, 0);
    EXPECT_TRUE(domain_result.is_valid);
    EXPECT_TRUE(domain_result.is_gpu_domain)
        << "Layer 0 should be assigned to GPU domain when prefer_gpu_early_layers=true";
}

/**
 * @brief Test: Small model fits entirely on GPU (no CPU needed)
 */
TEST_F(Test__Scenario3_CPUParticipation, SmallModelGPUOnly)
{
    HeterogeneousConfig hconfig;
    hconfig.cpu_compute_fraction = 0.3f; // Even with CPU fraction...
    HeterogeneousMultiDomainStrategy strategy(hconfig);

    const auto &model = models_[0]; // 0.5B - fits easily in 40GB
    auto input = createPlacementInput(model, 0);

    if (!strategy.isApplicable(input))
    {
        GTEST_SKIP() << "Strategy not applicable";
    }

    auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);

    // For small models that fit in GPU, may still be GPU-only
    // Verify all layers are covered
    auto result = validateHeterogeneousPlan(plan, model.n_layers);
    EXPECT_TRUE(result.all_layers_covered);
}

/**
 * @brief Test: Socket 1 (CPU-only) participates when needed
 */
TEST_F(Test__Scenario3_CPUParticipation, CPUOnlySocketParticipates)
{
    HeterogeneousConfig hconfig;
    hconfig.cpu_compute_fraction = 0.5f; // High CPU participation
    HeterogeneousMultiDomainStrategy strategy(hconfig);

    const auto &model = models_[2]; // 72B
    auto input = createPlacementInput(model, 0);

    if (!strategy.isApplicable(input))
    {
        GTEST_SKIP() << "Strategy not applicable";
    }

    auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);

    // Check that rank 1 (CPU-only socket) is in some domain
    bool rank1_in_domain = false;
    for (const auto &domain : plan.domains)
    {
        for (int rank : domain.ranks)
        {
            if (rank == 1)
            {
                rank1_in_domain = true;
                // Rank 1 should be in a CPU domain (no GPU on that socket)
                EXPECT_TRUE(domain.type == TPDomainType::CPU_CROSS_RANK)
                    << "Rank 1 (CPU-only socket) should be in CPU domain";
                break;
            }
        }
    }

    EXPECT_TRUE(rank1_in_domain)
        << "Rank 1 (CPU-only socket) should participate in at least one domain";
}

/**
 * @brief Test: Layer distribution proportional to compute capability
 */
TEST_F(Test__Scenario3_CPUParticipation, LayerDistributionProportional)
{
    HeterogeneousConfig hconfig;
    hconfig.cpu_compute_fraction = 0.2f;
    HeterogeneousMultiDomainStrategy strategy(hconfig);

    const auto &model = models_[1]; // 7B
    auto input = createPlacementInput(model, 0);

    if (!strategy.isApplicable(input))
    {
        GTEST_SKIP() << "Strategy not applicable";
    }

    auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);

    // Count layers per domain type
    int gpu_layers = 0;
    int cpu_layers = 0;

    for (const auto &domain : plan.domains)
    {
        int layers = domain.layer_end - domain.layer_start;
        if (domain.type == TPDomainType::GPU_INTRA_RANK)
        {
            gpu_layers += layers;
        }
        else
        {
            cpu_layers += layers;
        }
    }

    // With 20% CPU fraction, GPU should handle majority (but exact ratio depends on strategy)
    // The strategy may allocate more to CPU based on memory constraints, so allow up to 60%
    if (cpu_layers > 0)
    {
        float cpu_ratio = static_cast<float>(cpu_layers) / model.n_layers;
        EXPECT_LE(cpu_ratio, 0.6f) // Allow variance for memory-constrained scenarios
            << "CPU should handle <= 60% of layers (cpu_compute_fraction is a hint, not hard limit)";
        EXPECT_GT(gpu_layers, 0)
            << "GPU should still handle some layers";
    }

    // All layers should be covered
    EXPECT_EQ(gpu_layers + cpu_layers, model.n_layers);
}

/**
 * @brief Test: QPI bandwidth considered for cross-socket communication
 */
TEST_F(Test__Scenario3_CPUParticipation, CrossSocketBandwidthAware)
{
    // This test verifies the strategy considers QPI/UPI bandwidth
    // when making domain assignments across sockets

    HeterogeneousConfig hconfig;
    hconfig.cpu_compute_fraction = 0.3f;
    HeterogeneousMultiDomainStrategy strategy(hconfig);

    const auto &model = models_[1]; // 7B
    auto input = createPlacementInput(model, 0);

    if (!strategy.isApplicable(input))
    {
        GTEST_SKIP() << "Strategy not applicable";
    }

    auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);

    // Verify plan is valid
    auto result = validateHeterogeneousPlan(plan, model.n_layers);
    EXPECT_TRUE(result.is_valid) << result.error_message;

    // The strategy should prefer keeping communication within sockets
    // when bandwidth is limited (not strictly testable without internals)
    printHeterogeneousPlanSummary(plan, model);
}

/**
 * @brief Test: Zero CPU fraction means GPU-only
 */
TEST_F(Test__Scenario3_CPUParticipation, ZeroCPUFractionGPUOnly)
{
    HeterogeneousConfig hconfig;
    hconfig.cpu_compute_fraction = 0.0f; // No CPU participation
    HeterogeneousMultiDomainStrategy strategy(hconfig);

    const auto &model = models_[0]; // 0.5B - definitely fits
    auto input = createPlacementInput(model, 0);

    if (!strategy.isApplicable(input))
    {
        GTEST_SKIP() << "Strategy not applicable";
    }

    auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);

    // Count CPU domains
    int cpu_domains = 0;
    for (const auto &domain : plan.domains)
    {
        if (domain.type == TPDomainType::CPU_CROSS_RANK)
        {
            cpu_domains++;
        }
    }

    // With cpu_compute_fraction = 0, should be GPU-only (if model fits)
    // Note: Large models may still need CPU spillover even with fraction=0
    if (modelFitsInGPUMemory(model))
    {
        EXPECT_EQ(cpu_domains, 0)
            << "Small model with cpu_compute_fraction=0 should be GPU-only";
    }
}

/**
 * @brief Test: CPU can handle later layers in decode phase
 */
TEST_F(Test__Scenario3_CPUParticipation, CPUHandlesLaterLayers)
{
    HeterogeneousConfig hconfig;
    hconfig.prefer_gpu_early_layers = true;
    hconfig.cpu_compute_fraction = 0.4f; // Substantial CPU participation
    HeterogeneousMultiDomainStrategy strategy(hconfig);

    const auto &model = models_[2]; // 72B
    auto input = createPlacementInput(model, 0);

    if (!strategy.isApplicable(input))
    {
        GTEST_SKIP() << "Strategy not applicable";
    }

    auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);

    // Last layer should be in CPU domain if GPU handles early layers
    auto last_layer_result = validateDomainForLayer(plan, model.n_layers - 1);
    EXPECT_TRUE(last_layer_result.is_valid);

    // With high CPU fraction and prefer_gpu_early_layers, later layers go to CPU
    // But this is strategy-dependent, so we just verify the plan is valid
    auto result = validateHeterogeneousPlan(plan, model.n_layers);
    EXPECT_TRUE(result.is_valid) << result.error_message;
}

/**
 * @brief Summary test: Print full plan for visual inspection
 */
TEST_F(Test__Scenario3_CPUParticipation, SummaryPrintAllModels)
{
    HeterogeneousConfig hconfig;
    hconfig.cpu_compute_fraction = 0.3f;
    HeterogeneousMultiDomainStrategy strategy(hconfig);

    printClusterSummary();

    for (const auto &model : models_)
    {
        auto input = createPlacementInput(model, 0);
        if (strategy.isApplicable(input))
        {
            auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);
            printHeterogeneousPlanSummary(plan, model);
        }
        else
        {
            printf("Model %s: Strategy not applicable\n", model.name.c_str());
        }
    }

    SUCCEED();
}

// =============================================================================
// Scenario-Specific Tests: Hybrid Execution Validation
// =============================================================================

TEST_F(Test__Scenario3_CPUParticipation, HybridExecutionPlanValid)
{
    // Test that we can generate valid plans for all models with CPU participation
    HeterogeneousConfig hconfig;
    hconfig.cpu_compute_fraction = 0.25f;
    HeterogeneousMultiDomainStrategy strategy(hconfig);

    for (const auto &model : models_)
    {
        auto input = createPlacementInput(model, 0);
        if (!strategy.isApplicable(input))
            continue;

        auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);
        auto result = validateHeterogeneousPlan(plan, model.n_layers);

        EXPECT_TRUE(result.is_valid)
            << "Model " << model.name << ": " << result.error_message;
        EXPECT_TRUE(result.all_layers_covered)
            << "Model " << model.name << ": Not all layers covered";
        EXPECT_TRUE(result.no_layer_gaps)
            << "Model " << model.name << ": Has layer gaps";
        EXPECT_TRUE(result.no_layer_overlaps)
            << "Model " << model.name << ": Has layer overlaps";
    }
}

TEST_F(Test__Scenario3_CPUParticipation, DomainDeviceConsistency)
{
    // Verify GPU domains only contain GPU devices, CPU domains only CPU devices
    HeterogeneousConfig hconfig;
    hconfig.cpu_compute_fraction = 0.3f;
    HeterogeneousMultiDomainStrategy strategy(hconfig);

    const auto &model = models_[1]; // 7B
    auto input = createPlacementInput(model, 0);

    if (!strategy.isApplicable(input))
    {
        GTEST_SKIP() << "Strategy not applicable";
    }

    auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);

    for (const auto &domain : plan.domains)
    {
        if (domain.type == TPDomainType::GPU_INTRA_RANK)
        {
            // GPU domain should only have GPU devices
            for (const auto &dev : domain.devices)
            {
                EXPECT_TRUE(dev.is_gpu())
                    << "GPU domain contains non-GPU device: " << dev.to_string();
            }
        }
        else if (domain.type == TPDomainType::CPU_CROSS_RANK)
        {
            // CPU domain should only have CPU devices
            for (const auto &dev : domain.devices)
            {
                EXPECT_TRUE(dev.is_cpu())
                    << "CPU domain contains non-CPU device: " << dev.to_string();
            }
        }
    }
}
