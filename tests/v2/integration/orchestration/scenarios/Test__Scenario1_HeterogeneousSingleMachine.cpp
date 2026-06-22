/**
 * @file Test__Scenario1_HeterogeneousSingleMachine.cpp
 * @brief Orchestration test for Scenario 1: Single Machine, 2 Ranks, Heterogeneous
 *
 * **Hardware Configuration** (from ARCHITECTURE_EXECUTION_SCENARIOS.md):
 * ┌─────────────────────────────────────────────────────────────────────────────┐
 * │                         SINGLE MACHINE                                       │
 * │  ┌─────────────────────────────────────────────────────────────────────┐    │
 * │  │       SOCKET 0 (NUMA Node 0)    │       SOCKET 1 (NUMA Node 1)      │    │
 * │  │         MPI Rank 0              │         MPI Rank 1                │    │
 * │  │  ┌─────────────────────────┐    │  ┌───────────┐ ┌───────────┐     │    │
 * │  │  │   NVIDIA RTX 3090       │    │  │ AMD Mi50  │ │ AMD Mi50  │     │    │
 * │  │  │   24GB VRAM             │    │  │ 16GB VRAM │ │ 16GB VRAM │     │    │
 * │  │  │   DeviceId: CUDA:0      │    │  │ ROCm:0    │ │ ROCm:1    │     │    │
 * │  │  └─────────────────────────┘    │  └───────────┘ └───────────┘     │    │
 * │  └─────────────────────────────────────────────────────────────────────┘    │
 * └─────────────────────────────────────────────────────────────────────────────┘
 *
 * **Tested Models**: Qwen2 7B, 72B, 235B (fictional)
 * **Total GPU VRAM**: 56 GB (24 + 16 + 16)
 * **Total GPU Bandwidth**: ~2.4 TB/s
 *
 * @see docs/v2/projects/2026-01/ARCHITECTURE_EXECUTION_SCENARIOS.md (Scenario 1)
 */

#include <gtest/gtest.h>
#include "../OrchestrationTestBase.h"

using namespace llaminar2;
using namespace llaminar2::test::orchestration;

// =============================================================================
// Scenario 1 Configuration
// =============================================================================

class Test__Scenario1_HeterogeneousSingleMachine : public OrchestrationTestBase {
protected:
    ClusterConfig getClusterConfig() override {
        ClusterConfig config;
        config.name = "Scenario 1: Single Machine, 2 Ranks, Heterogeneous (CUDA + ROCm)";
        config.num_machines = 1;
        config.sockets_per_machine = 2;
        
        // Socket 0: 1× RTX 3090 (CUDA)
        SocketConfig socket0;
        socket0.cpu = CPUSpecs::XEON_28C_DDR4;
        socket0.gpus = {GPUSpecs::RTX_3090};
        
        // Socket 1: 2× Mi50 (ROCm)
        SocketConfig socket1;
        socket1.cpu = CPUSpecs::XEON_28C_DDR4;
        socket1.gpus = {GPUSpecs::MI50, GPUSpecs::MI50};
        
        config.socket_configs = {socket0, socket1};
        
        // Local interconnects (single machine, no Infiniband)
        config.qpi_upi_bandwidth_gbps = 50.0f;  // Cross-socket
        config.pcie_bandwidth_gbps = 32.0f;
        
        return config;
    }
    
    std::vector<ModelConfig> getModelConfigs() override {
        return {
            ModelConfigs::QWEN2_7B,    // Fits in single GPU
            ModelConfigs::QWEN2_72B,   // Requires multi-GPU
            ModelConfigs::QWEN2_235B,  // Requires CPU spillover
        };
    }
};

// =============================================================================
// Instantiate Standard Orchestration Tests
// =============================================================================

INSTANTIATE_ORCHESTRATION_TESTS(Test__Scenario1_HeterogeneousSingleMachine);

// =============================================================================
// Scenario-Specific Tests
// =============================================================================

TEST_F(Test__Scenario1_HeterogeneousSingleMachine, Socket0HasCUDA)
{
    // Rank 0 (Socket 0) should have CUDA GPU
    const auto& placement = topologies_[0]->placement();
    
    bool has_cuda = false;
    for (const auto& dev : placement.devices) {
        if (dev.type == DeviceCapability::Type::CUDA) {
            has_cuda = true;
            EXPECT_EQ(dev.name, "NVIDIA RTX 3090");
            EXPECT_EQ(dev.memory_bytes, static_cast<size_t>(24ULL * 1024 * 1024 * 1024));
        }
    }
    EXPECT_TRUE(has_cuda) << "Socket 0 should have CUDA GPU";
}

TEST_F(Test__Scenario1_HeterogeneousSingleMachine, Socket1HasROCm)
{
    // Rank 1 (Socket 1) should have ROCm GPUs
    const auto& placement = topologies_[1]->placement();
    
    int rocm_count = 0;
    for (const auto& dev : placement.devices) {
        if (dev.type == DeviceCapability::Type::ROCm) {
            rocm_count++;
            EXPECT_EQ(dev.name, "AMD Instinct Mi50");
        }
    }
    EXPECT_EQ(rocm_count, 2) << "Socket 1 should have 2 ROCm GPUs";
}

TEST_F(Test__Scenario1_HeterogeneousSingleMachine, TotalVRAMIs56GB)
{
    // 24 (3090) + 16 (Mi50) + 16 (Mi50) = 56 GB
    float expected_vram = 24.0f + 16.0f + 16.0f;
    EXPECT_FLOAT_EQ(cluster_.totalVRAMGB(), expected_vram);
}

TEST_F(Test__Scenario1_HeterogeneousSingleMachine, Qwen7B_FitsInSingle3090)
{
    // 7B at Q4 is ~4GB, fits in RTX 3090's 24GB
    const auto& model = models_[0];  // QWEN2_7B
    
    EXPECT_LT(model.memorySizeGB(), 24.0f);
    EXPECT_TRUE(modelFitsInGPUMemory(model));
}

TEST_F(Test__Scenario1_HeterogeneousSingleMachine, Qwen72B_RequiresAllGPUs)
{
    // 72B at Q4 is ~40GB, needs all GPUs (56GB total)
    const auto& model = models_[1];  // QWEN2_72B
    
    EXPECT_LT(model.memorySizeGB(), cluster_.totalVRAMGB());
    EXPECT_GT(model.memorySizeGB(), 24.0f);  // Doesn't fit in just 3090
    EXPECT_TRUE(modelFitsInGPUMemory(model));
}

TEST_F(Test__Scenario1_HeterogeneousSingleMachine, Qwen235B_RequiresCPUSpillover)
{
    // 235B at Q4 is ~130GB, exceeds 56GB GPU VRAM
    const auto& model = models_[2];  // QWEN2_235B
    
    EXPECT_FALSE(modelFitsInGPUMemory(model));
    EXPECT_GT(model.memorySizeGB(), cluster_.totalVRAMGB());
}

TEST_F(Test__Scenario1_HeterogeneousSingleMachine, CrossSocketIsQPI_NotInfiniband)
{
    // Single machine uses QPI/UPI, not Infiniband
    EXPECT_EQ(cluster_.num_machines, 1);
    EXPECT_GT(cluster_.qpi_upi_bandwidth_gbps, cluster_.infiniband_bandwidth_gbps)
        << "QPI/UPI should be faster than Infiniband for cross-socket";
}
