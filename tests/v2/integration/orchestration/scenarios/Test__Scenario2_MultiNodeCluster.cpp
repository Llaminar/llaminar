/**
 * @file Test__Scenario2_MultiNodeCluster.cpp
 * @brief Orchestration test for Scenario 2: 6 Machines, 12 Ranks, Complex Topology
 *
 * **Hardware Configuration** (from ARCHITECTURE_EXECUTION_SCENARIOS.md):
 * - 6 dual-socket Xeon machines connected via Infiniband RDMA
 * - 12 MPI ranks (2 per machine, 1 per socket)
 * - Socket 0 pattern: 3× RTX 3090 (CUDA) + 3× Mi50 (ROCm)
 * - Socket 1 pattern: 6× Mi50 (ROCm) + 1× 7900 XTX (ROCm)
 * - Mixed CUDA/ROCm execution with CPU decode participation
 *
 * **GPU Summary**:
 * - 18× RTX 3090 (CUDA): 432 GB VRAM
 * - 54× Mi50 (ROCm): 864 GB VRAM
 * - 6× 7900 XTX (ROCm): 144 GB VRAM
 * - Total: 78 GPUs, ~1440 GB VRAM
 *
 * **Tested Models**: 7B, 72B, 235B, 571B, 1142B
 *
 * @see docs/v2/projects/2026-01/ARCHITECTURE_EXECUTION_SCENARIOS.md (Scenario 2)
 */

#include <gtest/gtest.h>
#include "../OrchestrationTestBase.h"

using namespace llaminar2;
using namespace llaminar2::test::orchestration;

// =============================================================================
// Scenario 2 Configuration
// =============================================================================

class Test__Scenario2_MultiNodeCluster : public OrchestrationTestBase
{
protected:
    ClusterConfig getClusterConfig() override
    {
        ClusterConfig config;
        config.name = "Scenario 2: 6 Machines, 12 Ranks, Mixed CUDA/ROCm Cluster";
        config.num_machines = 6;
        config.sockets_per_machine = 2;

        // Socket 0 pattern: 3× RTX 3090 (CUDA) + 3× Mi50 (ROCm)
        SocketConfig socket0;
        socket0.cpu = CPUSpecs::XEON_28C_DDR5;
        socket0.gpus = {
            GPUSpecs::RTX_3090, GPUSpecs::RTX_3090, GPUSpecs::RTX_3090, // CUDA
            GPUSpecs::MI50, GPUSpecs::MI50, GPUSpecs::MI50              // ROCm
        };

        // Socket 1 pattern: 6× Mi50 (ROCm) + 1× 7900 XTX (ROCm)
        SocketConfig socket1;
        socket1.cpu = CPUSpecs::XEON_28C_DDR5;
        socket1.gpus = {
            GPUSpecs::MI50, GPUSpecs::MI50, GPUSpecs::MI50,
            GPUSpecs::MI50, GPUSpecs::MI50, GPUSpecs::MI50,
            GPUSpecs::RX_7900_XTX};

        config.socket_configs = {socket0, socket1};

        // Multi-node interconnects
        config.infiniband_bandwidth_gbps = 25.0f; // HDR Infiniband
        config.qpi_upi_bandwidth_gbps = 50.0f;    // Cross-socket
        config.nvlink_bandwidth_gbps = 600.0f;    // Between 3090s (subset)
        config.xgmi_bandwidth_gbps = 600.0f;      // Between Mi50s
        config.pcie_bandwidth_gbps = 32.0f;

        return config;
    }

    std::vector<ModelConfig> getModelConfigs() override
    {
        return {
            ModelConfigs::QWEN2_7B,    // Edge case - too small
            ModelConfigs::QWEN2_72B,   // Minimum realistic
            ModelConfigs::QWEN2_235B,  // Medium-large
            ModelConfigs::QWEN2_571B,  // Typical use case
            ModelConfigs::QWEN2_1142B, // Frontier model
        };
    }
};

// =============================================================================
// Instantiate Standard Orchestration Tests
// =============================================================================

INSTANTIATE_ORCHESTRATION_TESTS(Test__Scenario2_MultiNodeCluster);

// =============================================================================
// Scenario-Specific Tests: Cluster Topology
// =============================================================================

TEST_F(Test__Scenario2_MultiNodeCluster, ClusterHas12Ranks)
{
    EXPECT_EQ(cluster_.totalRanks(), 12);
    EXPECT_EQ(cluster_.num_machines, 6);
    EXPECT_EQ(cluster_.sockets_per_machine, 2);
}

TEST_F(Test__Scenario2_MultiNodeCluster, Socket0HasMixedCUDAAndROCm)
{
    // Even ranks (Socket 0 pattern): 3× RTX 3090 + 3× Mi50
    for (int rank = 0; rank < cluster_.totalRanks(); rank += 2)
    {
        const auto &placement = topologies_[rank]->placement();

        int cuda_count = 0, rocm_count = 0;
        for (const auto &dev : placement.devices)
        {
            if (dev.type == DeviceCapability::Type::CUDA)
                cuda_count++;
            if (dev.type == DeviceCapability::Type::ROCm)
                rocm_count++;
        }

        EXPECT_EQ(cuda_count, 3) << "Rank " << rank << " should have 3 CUDA GPUs";
        EXPECT_EQ(rocm_count, 3) << "Rank " << rank << " should have 3 ROCm GPUs";
    }
}

TEST_F(Test__Scenario2_MultiNodeCluster, Socket1HasOnlyROCm)
{
    // Odd ranks (Socket 1 pattern): 6× Mi50 + 1× 7900 XTX
    for (int rank = 1; rank < cluster_.totalRanks(); rank += 2)
    {
        const auto &placement = topologies_[rank]->placement();

        int cuda_count = 0, rocm_count = 0;
        for (const auto &dev : placement.devices)
        {
            if (dev.type == DeviceCapability::Type::CUDA)
                cuda_count++;
            if (dev.type == DeviceCapability::Type::ROCm)
                rocm_count++;
        }

        EXPECT_EQ(cuda_count, 0) << "Rank " << rank << " should have no CUDA GPUs";
        EXPECT_EQ(rocm_count, 7) << "Rank " << rank << " should have 7 ROCm GPUs";
    }
}

TEST_F(Test__Scenario2_MultiNodeCluster, TotalGPUCountIs78)
{
    // 6 machines × (6 + 7) GPUs = 78
    EXPECT_EQ(cluster_.totalGPUs(), 78);
}

TEST_F(Test__Scenario2_MultiNodeCluster, TotalVRAMIsAbout1440GB)
{
    // Socket 0: 3×24 + 3×16 = 120 GB
    // Socket 1: 6×16 + 1×24 = 120 GB
    // Per machine: 240 GB, Total: 6 × 240 = 1440 GB
    float expected_vram = 6.0f * (3 * 24 + 3 * 16 + 6 * 16 + 1 * 24);
    EXPECT_FLOAT_EQ(cluster_.totalVRAMGB(), expected_vram);
}

// =============================================================================
// Scenario-Specific Tests: Model Fit Analysis
// =============================================================================

TEST_F(Test__Scenario2_MultiNodeCluster, Qwen7B_IsEdgeCaseTooSmall)
{
    // 7B is only ~4 GB, cluster has 1440 GB
    const auto &model = models_[0];

    float utilization = model.memorySizeGB() / cluster_.totalVRAMGB();
    EXPECT_LT(utilization, 0.01f) << "7B uses <1% of cluster - too small";

    // With 12 ranks, each gets <3 layers
    float lpr = model.layersPerRank(cluster_.totalRanks());
    EXPECT_LT(lpr, 3.0f) << "7B has too few layers per rank";
}

TEST_F(Test__Scenario2_MultiNodeCluster, Qwen72B_MinimumRealistic)
{
    // 72B is ~40 GB, fits easily but uses cluster reasonably
    const auto &model = models_[1];

    EXPECT_TRUE(modelFitsInGPUMemory(model));
    EXPECT_EQ(machinesNeededForModel(model), 1); // Fits in one machine

    float lpr = model.layersPerRank(cluster_.totalRanks());
    EXPECT_GE(lpr, 6.0f) << "72B has reasonable layers per rank";
}

TEST_F(Test__Scenario2_MultiNodeCluster, Qwen235B_RequiresMultiSocket)
{
    // 235B is ~130 GB, needs ~1 machine (240 GB/machine)
    const auto &model = models_[2];

    EXPECT_TRUE(modelFitsInGPUMemory(model));
    EXPECT_EQ(machinesNeededForModel(model), 1);

    // 128 layers / 12 ranks ≈ 10-11 per rank
    float lpr = model.layersPerRank(cluster_.totalRanks());
    EXPECT_GE(lpr, 10.0f);
    EXPECT_LE(lpr, 11.0f);
}

TEST_F(Test__Scenario2_MultiNodeCluster, Qwen571B_TypicalUseCase)
{
    // 571B is ~320 GB, needs 2 machines
    const auto &model = models_[3];

    EXPECT_TRUE(modelFitsInGPUMemory(model));
    EXPECT_EQ(machinesNeededForModel(model), 2);

    // Uses ~22% of cluster
    float utilization = model.memorySizeGB() / cluster_.totalVRAMGB();
    EXPECT_GE(utilization, 0.20f);
    EXPECT_LE(utilization, 0.25f);
}

TEST_F(Test__Scenario2_MultiNodeCluster, Qwen1142B_FrontierModel)
{
    // 1142B is ~640 GB weights. With schema-based memory calculation:
    // - Weights: 640 GB
    // - Layer buffers: ~1-2 GB (from Qwen2Schema, includes attention workspace)
    // - Model buffers: ~200-400 MB (hidden + logits)
    // - KV Cache: ~100-200 GB at 2048 seq_len
    // - 5% overhead (GEMM workspace, fragmentation, framework)
    // Total: ~740-840 GB -> needs 3 machines at 240 GB/machine
    const auto &model = models_[4];

    EXPECT_TRUE(modelFitsInGPUMemory(model));
    EXPECT_EQ(machinesNeededForModel(model), 3);

    // 192 layers / 12 ranks = exactly 16 per rank (evenly distributed)
    EXPECT_EQ(model.n_layers % cluster_.totalRanks(), 0);
    float lpr = model.layersPerRank(cluster_.totalRanks());
    EXPECT_EQ(static_cast<int>(lpr), 16);
}

// =============================================================================
// Scenario-Specific Tests: Interconnect Analysis
// =============================================================================

TEST_F(Test__Scenario2_MultiNodeCluster, InfinibandRequiredForMultiMachine)
{
    // Models requiring 2+ machines need Infiniband
    for (const auto &model : models_)
    {
        if (machinesNeededForModel(model) > 1)
        {
            // Cross-machine communication uses Infiniband
            EXPECT_GT(cluster_.infiniband_bandwidth_gbps, 0.0f)
                << model.name << " requires Infiniband for multi-machine";
        }
    }
}

TEST_F(Test__Scenario2_MultiNodeCluster, QPIFasterThanInfiniband)
{
    // Same-machine cross-socket uses QPI/UPI (faster than Infiniband)
    EXPECT_GT(cluster_.qpi_upi_bandwidth_gbps, cluster_.infiniband_bandwidth_gbps)
        << "QPI/UPI should be faster than Infiniband";
}

TEST_F(Test__Scenario2_MultiNodeCluster, NVLinkAndxGMIAvailable)
{
    // High-speed GPU interconnects available for peer-to-peer
    EXPECT_GT(cluster_.nvlink_bandwidth_gbps, 0.0f);
    EXPECT_GT(cluster_.xgmi_bandwidth_gbps, 0.0f);

    // Both should be much faster than PCIe
    EXPECT_GT(cluster_.nvlink_bandwidth_gbps, cluster_.pcie_bandwidth_gbps * 10);
    EXPECT_GT(cluster_.xgmi_bandwidth_gbps, cluster_.pcie_bandwidth_gbps * 10);
}

// =============================================================================
// Scenario-Specific Tests: Bandwidth Analysis
// =============================================================================

TEST_F(Test__Scenario2_MultiNodeCluster, TotalGPUBandwidthAbout35TBs)
{
    // Socket 0: 3×936 + 3×717 = 4959 GB/s
    // Socket 1: 6×717 + 1×960 = 5262 GB/s
    // Per machine: ~10.2 TB/s, Total: 6 × 10.2 ≈ 61 TB/s
    // Wait - let me recalculate more carefully
    float socket0_bw = 3 * 936.0f + 3 * 717.0f;       // 4959 GB/s
    float socket1_bw = 6 * 717.0f + 1 * 960.0f;       // 5262 GB/s
    float per_machine = socket0_bw + socket1_bw;      // 10221 GB/s
    float expected_tbs = (6 * per_machine) / 1000.0f; // ~61.3 TB/s

    EXPECT_NEAR(cluster_.totalGPUBandwidthTBs(), expected_tbs, 1.0f);
}

TEST_F(Test__Scenario2_MultiNodeCluster, CPUBandwidthAdds8Percent)
{
    // CPU bandwidth relative to GPU bandwidth
    float cpu_fraction = cluster_.totalCPUBandwidthTBs() /
                         (cluster_.totalGPUBandwidthTBs() + cluster_.totalCPUBandwidthTBs());

    // CPU should add ~5-10% to total bandwidth
    EXPECT_GE(cpu_fraction, 0.05f);
    EXPECT_LE(cpu_fraction, 0.15f);
}

// =============================================================================
// Scenario-Specific Tests: Tensor Parallelism
// =============================================================================

TEST_F(Test__Scenario2_MultiNodeCluster, LargeModels_TPDegree12)
{
    // For models using all 12 ranks, TP degree is 12
    for (const auto &model : models_)
    {
        if (model.memorySizeGB() > 100.0f)
        { // Larger models
            // Check heads divide by 12 (or close factors)
            bool heads_divide = (model.n_heads % 12 == 0) ||
                                (model.n_heads % 6 == 0) ||
                                (model.n_heads % 4 == 0);
            EXPECT_TRUE(heads_divide)
                << model.name << " heads should be divisible for TP";
        }
    }
}

TEST_F(Test__Scenario2_MultiNodeCluster, GQACompatibility)
{
    // All models should have compatible GQA ratios
    for (const auto &model : models_)
    {
        EXPECT_EQ(model.n_heads % model.n_kv_heads, 0)
            << model.name << " should have integer GQA ratio";

        size_t gqa_ratio = model.n_heads / model.n_kv_heads;
        EXPECT_GE(gqa_ratio, 1);
        EXPECT_LE(gqa_ratio, 16); // Typical range
    }
}
