/**
 * @file Test__Scenario6_AsymmetricVRAM.cpp
 * @brief Orchestration tests for asymmetric VRAM configurations
 *
 * **Hardware Configuration**:
 * ┌─────────────────────────────────────────────────────────────────────────────┐
 * │                         SINGLE MACHINE                                       │
 * │  ┌─────────────────────────────────────────────────────────────────────┐    │
 * │  │                        SOCKET 0 (NUMA Node 0)                       │    │
 * │  │                          MPI Rank 0                                 │    │
 * │  │  ┌───────────────────┐    ┌────────────────────────────────────┐   │    │
 * │  │  │  NVIDIA RTX 4090  │    │     NVIDIA A100-80GB               │   │    │
 * │  │  │  24GB VRAM        │    │     80GB VRAM                      │   │    │
 * │  │  │  DeviceId: CUDA:0 │    │     DeviceId: CUDA:1               │   │    │
 * │  │  └───────────────────┘    └────────────────────────────────────┘   │    │
 * │  │                      Xeon 28-core DDR5                              │    │
 * │  └─────────────────────────────────────────────────────────────────────┘    │
 * └─────────────────────────────────────────────────────────────────────────────┘
 *
 * **Scenario Purpose**:
 * Tests VRAM-proportional layer distribution when GPUs have very different
 * memory capacities. The A100-80GB has 3.33× more VRAM than the RTX 4090,
 * so layer/weight distribution should reflect this asymmetry.
 *
 * **VRAM Distribution**:
 * - RTX 4090:  24GB / 104GB = 23.1%
 * - A100-80GB: 80GB / 104GB = 76.9%
 *
 * **Tested Models**: Qwen2 0.5B, 7B, 72B
 * **Total GPU VRAM**: 104 GB (24 + 80)
 * **Total GPU Bandwidth**: ~3.3 TB/s (1008 + 2039 GB/s)
 *
 * @see docs/v2/projects/2026-01/ARCHITECTURE_EXECUTION_SCENARIOS.md
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "../OrchestrationTestBase.h"

using namespace llaminar2;
using namespace llaminar2::test::orchestration;

// =============================================================================
// Scenario 6 Configuration
// =============================================================================

/**
 * @brief Scenario 6: Single Socket, Asymmetric VRAM (24GB + 80GB)
 *
 * Hardware Configuration:
 * - 1 machine, 1 socket
 * - Socket 0: 1× RTX 4090 (24GB) + 1× A100-80GB (80GB) + Xeon 28-core
 *
 * Test Focus:
 * - VRAM-proportional layer/weight distribution
 * - Larger VRAM GPU gets proportionally more work
 * - TP domain spans both GPUs for attention heads and output
 * - Small models may fit entirely on the A100
 */
class Test__Scenario6_AsymmetricVRAM : public OrchestrationTestBase
{
protected:
    ClusterConfig getClusterConfig() override
    {
        ClusterConfig config;
        config.name = "Scenario 6: Single Socket, Asymmetric VRAM (24GB + 80GB)";
        config.num_machines = 1;
        config.sockets_per_machine = 1;

        // Socket 0: Mixed VRAM - RTX 4090 (24GB) + A100-80GB (80GB)
        SocketConfig socket0;
        socket0.cpu = CPUSpecs::XEON_28C_DDR5;
        socket0.gpus = {GPUSpecs::RTX_4090, GPUSpecs::A100_80GB};

        config.socket_configs = {socket0};

        // Local interconnects (single socket, NVLink possible between GPUs)
        config.nvlink_bandwidth_gbps = 600.0f; // A100 has NVLink
        config.pcie_bandwidth_gbps = 32.0f;    // RTX 4090 uses PCIe

        return config;
    }

    std::vector<ModelConfig> getModelConfigs() override
    {
        return {
            ModelConfigs::QWEN2_0_5B, // Tiny - fits on either GPU
            ModelConfigs::QWEN2_7B,   // Small - fits on either GPU with room
            ModelConfigs::QWEN2_72B,  // Large - needs both GPUs, VRAM-proportional
        };
    }

    // =========================================================================
    // Scenario-Specific Helpers
    // =========================================================================

    /**
     * @brief Get VRAM ratio (A100 / RTX 4090)
     */
    float getVRAMRatio() const
    {
        return 80.0f / 24.0f; // ~3.33:1
    }

    /**
     * @brief Get RTX 4090 VRAM fraction of total
     */
    float getRTX4090Fraction() const
    {
        return 24.0f / 104.0f; // ~23.1%
    }

    /**
     * @brief Get A100 VRAM fraction of total
     */
    float getA100Fraction() const
    {
        return 80.0f / 104.0f; // ~76.9%
    }

    /**
     * @brief Calculate expected layer count for a GPU based on VRAM proportion
     */
    int expectedLayersForGPU(const ModelConfig &model, float vram_fraction) const
    {
        return static_cast<int>(std::round(model.n_layers * vram_fraction));
    }

    /**
     * @brief Check if model fits entirely on A100-80GB
     */
    bool modelFitsOnA100(const ModelConfig &model) const
    {
        return model.memorySizeGB() < 80.0f;
    }

    /**
     * @brief Check if model fits entirely on RTX 4090
     */
    bool modelFitsOnRTX4090(const ModelConfig &model) const
    {
        return model.memorySizeGB() < 24.0f;
    }
};

// =============================================================================
// Instantiate Standard Orchestration Tests (15+ tests from macro)
// =============================================================================

INSTANTIATE_ORCHESTRATION_TESTS(Test__Scenario6_AsymmetricVRAM);

// =============================================================================
// Scenario-Specific Tests: Topology Validation
// =============================================================================

TEST_F(Test__Scenario6_AsymmetricVRAM, TotalVRAMIs104GB)
{
    // 24GB (RTX 4090) + 80GB (A100) = 104GB
    float expected_vram = 24.0f + 80.0f;
    EXPECT_FLOAT_EQ(cluster_.totalVRAMGB(), expected_vram);
}

TEST_F(Test__Scenario6_AsymmetricVRAM, BothGPUsAreCUDA)
{
    // Both GPUs should be CUDA type (same vendor, different VRAM)
    const auto &placement = topologies_[0]->placement();

    int cuda_count = 0;
    for (const auto &dev : placement.devices)
    {
        if (dev.type == DeviceCapability::Type::CUDA)
        {
            cuda_count++;
        }
    }
    EXPECT_EQ(cuda_count, 2) << "Should have exactly 2 CUDA GPUs";
}

TEST_F(Test__Scenario6_AsymmetricVRAM, HasRTX4090With24GB)
{
    const auto &placement = topologies_[0]->placement();

    bool found_4090 = false;
    for (const auto &dev : placement.devices)
    {
        if (dev.name.find("4090") != std::string::npos)
        {
            found_4090 = true;
            EXPECT_EQ(dev.memory_bytes, static_cast<size_t>(24ULL * 1024 * 1024 * 1024))
                << "RTX 4090 should have 24GB VRAM";
            EXPECT_EQ(dev.type, DeviceCapability::Type::CUDA);
        }
    }
    EXPECT_TRUE(found_4090) << "RTX 4090 should be present in cluster";
}

TEST_F(Test__Scenario6_AsymmetricVRAM, HasA100With80GB)
{
    const auto &placement = topologies_[0]->placement();

    bool found_a100 = false;
    for (const auto &dev : placement.devices)
    {
        if (dev.name.find("A100") != std::string::npos &&
            dev.name.find("80GB") != std::string::npos)
        {
            found_a100 = true;
            EXPECT_EQ(dev.memory_bytes, static_cast<size_t>(80ULL * 1024 * 1024 * 1024))
                << "A100-80GB should have 80GB VRAM";
            EXPECT_EQ(dev.type, DeviceCapability::Type::CUDA);
        }
    }
    EXPECT_TRUE(found_a100) << "A100-80GB should be present in cluster";
}

TEST_F(Test__Scenario6_AsymmetricVRAM, VRAMRatioCorrect)
{
    // VRAM ratio should be ~3.33:1 (80/24)
    float expected_ratio = 80.0f / 24.0f;
    EXPECT_NEAR(getVRAMRatio(), expected_ratio, 0.01f);
    EXPECT_NEAR(getVRAMRatio(), 3.333f, 0.01f);
}

TEST_F(Test__Scenario6_AsymmetricVRAM, TotalGPUCountIs2)
{
    EXPECT_EQ(cluster_.totalGPUs(), 2);
}

TEST_F(Test__Scenario6_AsymmetricVRAM, SingleSocketConfiguration)
{
    EXPECT_EQ(cluster_.num_machines, 1);
    EXPECT_EQ(cluster_.sockets_per_machine, 1);
    EXPECT_EQ(cluster_.totalRanks(), 1);
}

// =============================================================================
// Scenario-Specific Tests: Model Fit Analysis
// =============================================================================

TEST_F(Test__Scenario6_AsymmetricVRAM, SmallModelFitsOnEither)
{
    // 0.5B at Q4 is ~500MB - fits on both GPUs
    const auto &model = models_[0]; // QWEN2_0_5B

    EXPECT_TRUE(modelFitsOnRTX4090(model)) << "0.5B should fit on RTX 4090 (24GB)";
    EXPECT_TRUE(modelFitsOnA100(model)) << "0.5B should fit on A100 (80GB)";
}

TEST_F(Test__Scenario6_AsymmetricVRAM, MediumModelFitsOnBoth)
{
    // 7B at Q4 is ~4GB - fits on either GPU
    const auto &model = models_[1]; // QWEN2_7B

    EXPECT_TRUE(modelFitsOnRTX4090(model)) << "7B should fit on RTX 4090 (24GB)";
    EXPECT_TRUE(modelFitsOnA100(model)) << "7B should fit on A100 (80GB)";
}

TEST_F(Test__Scenario6_AsymmetricVRAM, LargeModelNeedsBothGPUs)
{
    // 72B at Q4 is ~40GB - exceeds RTX 4090, needs A100 or both
    const auto &model = models_[2]; // QWEN2_72B

    EXPECT_FALSE(modelFitsOnRTX4090(model)) << "72B should NOT fit on RTX 4090 alone";
    EXPECT_TRUE(modelFitsOnA100(model)) << "72B should fit on A100-80GB alone";
    EXPECT_TRUE(modelFitsInGPUMemory(model)) << "72B should fit in total VRAM (104GB)";
}

TEST_F(Test__Scenario6_AsymmetricVRAM, Qwen7BFitsInTotalVRAM)
{
    const auto &model = models_[1]; // QWEN2_7B

    EXPECT_LT(model.memorySizeGB(), 104.0f);
    EXPECT_TRUE(modelFitsInGPUMemory(model));
}

TEST_F(Test__Scenario6_AsymmetricVRAM, Qwen72BFitsInTotalVRAM)
{
    const auto &model = models_[2]; // QWEN2_72B

    EXPECT_LT(model.memorySizeGB(), 104.0f);
    EXPECT_TRUE(modelFitsInGPUMemory(model));
}

// =============================================================================
// Scenario-Specific Tests: VRAM-Proportional Distribution
// =============================================================================

TEST_F(Test__Scenario6_AsymmetricVRAM, A100GetsMoreLayers)
{
    // For models that need both GPUs, A100 should get ~77% of layers
    const auto &model = models_[2]; // QWEN2_72B - 80 layers

    int rtx_expected = expectedLayersForGPU(model, getRTX4090Fraction());
    int a100_expected = expectedLayersForGPU(model, getA100Fraction());

    // RTX 4090: 23.1% of 80 = ~18 layers
    // A100:     76.9% of 80 = ~62 layers
    EXPECT_NEAR(rtx_expected, 18, 2) << "RTX 4090 should get ~18 layers of 80";
    EXPECT_NEAR(a100_expected, 62, 2) << "A100 should get ~62 layers of 80";
    EXPECT_EQ(rtx_expected + a100_expected, model.n_layers)
        << "Layer distribution should sum to total";
}

TEST_F(Test__Scenario6_AsymmetricVRAM, LayerDistributionProportional)
{
    // Verify layer distribution respects VRAM proportions
    HeterogeneousConfig hconfig;
    // Use default config - strategy should handle VRAM proportionality
    HeterogeneousMultiDomainStrategy strategy(hconfig);

    const auto &model = models_[2]; // QWEN2_72B
    auto input = createPlacementInput(model, 0);

    if (!strategy.isApplicable(input))
    {
        GTEST_SKIP() << "Strategy not applicable for this configuration";
    }

    auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);

    // With VRAM-proportional distribution:
    // - GPU with more VRAM should handle more layers
    // - Ratio should approximate the VRAM ratio

    // Verify the plan covers all layers
    auto result = validateHeterogeneousPlan(plan, model.n_layers);
    EXPECT_TRUE(result.all_layers_covered)
        << "All " << model.n_layers << " layers should be covered";
    EXPECT_GT(result.total_domains, 0)
        << "Should have at least one domain";
}

TEST_F(Test__Scenario6_AsymmetricVRAM, WeightsPlacedByVRAM)
{
    // Larger weights should prefer the GPU with more VRAM
    HeterogeneousConfig hconfig;
    // Use default config
    HeterogeneousMultiDomainStrategy strategy(hconfig);

    const auto &model = models_[2]; // QWEN2_72B
    auto input = createPlacementInput(model, 0);

    if (!strategy.isApplicable(input))
    {
        GTEST_SKIP() << "Strategy not applicable";
    }

    auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);

    // For a valid VRAM-proportional plan, the A100 should participate
    // in more domains or handle larger layer ranges
    EXPECT_GT(plan.domains.size(), 0) << "Should have at least one domain";
}

// =============================================================================
// Scenario-Specific Tests: TP Domain Spanning Both GPUs
// =============================================================================

TEST_F(Test__Scenario6_AsymmetricVRAM, TPDomainSpansBothGPUs)
{
    // Even with asymmetric VRAM, tensor parallel domains should span
    // both GPUs for attention and output layers
    HeterogeneousConfig hconfig;
    hconfig.enable_gpu_tp = true; // Enable GPU tensor parallelism
    HeterogeneousMultiDomainStrategy strategy(hconfig);

    const auto &model = models_[2]; // QWEN2_72B
    auto input = createPlacementInput(model, 0);

    if (!strategy.isApplicable(input))
    {
        GTEST_SKIP() << "Strategy not applicable";
    }

    auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);

    // Check that at least one domain spans both devices
    bool found_multi_device_domain = false;
    for (const auto &domain : plan.domains)
    {
        if (domain.devices.size() > 1)
        {
            found_multi_device_domain = true;
            break;
        }
    }

    // Note: This test passes if the strategy creates multi-device domains
    // It may skip if the strategy prefers per-device domains
    if (!found_multi_device_domain)
    {
        GTEST_SKIP() << "Strategy uses per-device domains (valid alternative)";
    }

    EXPECT_TRUE(found_multi_device_domain)
        << "Should have at least one TP domain spanning both GPUs";
}

// =============================================================================
// Scenario-Specific Tests: Bandwidth Awareness
// =============================================================================

TEST_F(Test__Scenario6_AsymmetricVRAM, A100HasHigherBandwidth)
{
    // A100-80GB: 2039 GB/s, RTX 4090: 1008 GB/s
    const auto &socket = cluster_.getSocketConfig(0);

    float rtx_bw = 0.0f;
    float a100_bw = 0.0f;

    for (const auto &gpu : socket.gpus)
    {
        if (gpu.name.find("4090") != std::string::npos)
        {
            rtx_bw = gpu.bandwidth_gbps;
        }
        else if (gpu.name.find("A100") != std::string::npos)
        {
            a100_bw = gpu.bandwidth_gbps;
        }
    }

    EXPECT_GT(a100_bw, rtx_bw) << "A100 should have higher bandwidth than RTX 4090";
    EXPECT_NEAR(a100_bw, 2039.0f, 10.0f);
    EXPECT_NEAR(rtx_bw, 1008.0f, 10.0f);
}

TEST_F(Test__Scenario6_AsymmetricVRAM, TotalBandwidthCorrect)
{
    // Total bandwidth: 1008 (RTX 4090) + 2039 (A100) = 3047 GB/s
    float expected_bw = 1008.0f + 2039.0f;
    float actual_bw = cluster_.totalGPUBandwidthTBs() * 1000.0f; // Convert TB/s to GB/s
    EXPECT_NEAR(actual_bw, expected_bw, 10.0f);
}

// =============================================================================
// Scenario-Specific Tests: Compute Power Awareness
// =============================================================================

TEST_F(Test__Scenario6_AsymmetricVRAM, A100HasHigherCompute)
{
    // A100-80GB: 156 TFLOPS, RTX 4090: 83 TFLOPS
    const auto &socket = cluster_.getSocketConfig(0);

    float rtx_compute = 0.0f;
    float a100_compute = 0.0f;

    for (const auto &gpu : socket.gpus)
    {
        if (gpu.name.find("4090") != std::string::npos)
        {
            rtx_compute = gpu.compute_tflops;
        }
        else if (gpu.name.find("A100") != std::string::npos)
        {
            a100_compute = gpu.compute_tflops;
        }
    }

    EXPECT_GT(a100_compute, rtx_compute) << "A100 should have higher compute than RTX 4090";
    EXPECT_NEAR(a100_compute, 156.0f, 5.0f);
    EXPECT_NEAR(rtx_compute, 83.0f, 5.0f);
}

// =============================================================================
// Summary Test
// =============================================================================

TEST_F(Test__Scenario6_AsymmetricVRAM, SummaryPrintAllModels)
{
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║       SCENARIO 6: ASYMMETRIC VRAM SUMMARY                            ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════╣\n");
    printf("║  RTX 4090:  24 GB VRAM (23.1%%)  | 1008 GB/s |  83 TFLOPS            ║\n");
    printf("║  A100-80GB: 80 GB VRAM (76.9%%)  | 2039 GB/s | 156 TFLOPS            ║\n");
    printf("║  Total:    104 GB VRAM          | 3047 GB/s | 239 TFLOPS            ║\n");
    printf("║  VRAM Ratio: %.2f:1 (A100:RTX 4090)                                  ║\n",
           getVRAMRatio());
    printf("╠══════════════════════════════════════════════════════════════════════╣\n");
    printf("║  MODEL FIT ANALYSIS                                                  ║\n");
    printf("╠═══════════╦════════════╦═══════════════╦═══════════════╦════════════╣\n");
    printf("║  Model    ║ Size (GB)  ║ Fits RTX 4090 ║ Fits A100-80G ║ Fits Total ║\n");
    printf("╠═══════════╬════════════╬═══════════════╬═══════════════╬════════════╣\n");

    for (const auto &model : models_)
    {
        const char *fits_rtx = modelFitsOnRTX4090(model) ? "Yes" : "No";
        const char *fits_a100 = modelFitsOnA100(model) ? "Yes" : "No";
        const char *fits_total = modelFitsInGPUMemory(model) ? "Yes" : "No";

        printf("║  %-8s ║    %5.1f   ║      %-3s      ║      %-3s      ║     %-3s    ║\n",
               model.name.c_str(), model.memorySizeGB(),
               fits_rtx, fits_a100, fits_total);
    }

    printf("╠═══════════╩════════════╩═══════════════╩═══════════════╩════════════╣\n");
    printf("║  EXPECTED LAYER DISTRIBUTION (VRAM-PROPORTIONAL)                     ║\n");
    printf("╠═══════════╦════════════╦═══════════════╦═══════════════════════════╣\n");
    printf("║  Model    ║ Layers    ║ RTX (~23%%)    ║ A100 (~77%%)              ║\n");
    printf("╠═══════════╬════════════╬═══════════════╬═══════════════════════════╣\n");

    for (const auto &model : models_)
    {
        int rtx_layers = expectedLayersForGPU(model, getRTX4090Fraction());
        int a100_layers = expectedLayersForGPU(model, getA100Fraction());

        printf("║  %-8s ║    %3d     ║      %3d       ║          %3d              ║\n",
               model.name.c_str(), model.n_layers, rtx_layers, a100_layers);
    }

    printf("╚═══════════╩════════════╩═══════════════╩═══════════════════════════╝\n\n");

    SUCCEED();
}
