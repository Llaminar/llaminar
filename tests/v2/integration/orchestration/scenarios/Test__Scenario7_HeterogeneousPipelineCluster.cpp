/**
 * @file Test__Scenario7_HeterogeneousPipelineCluster.cpp
 * @brief Orchestration test for Scenario 7: 2-Node Heterogeneous Pipeline Cluster
 *
 * **Hardware Configuration** (from conversation):
 * - 2 physical nodes connected via InfiniBand (400 Gbps)
 * - 4 MPI ranks (2 per node, 1 per socket)
 * - Each rank has 1× RTX 3090 (CUDA) + 1× MI50 (ROCm) = cross-vendor TP via HETEROGENEOUS backend
 * - 2 CPU sockets per node, connected via UPI for CPU TP
 *
 * **GPU Summary**:
 * - 4× RTX 3090 (CUDA): 96 GB VRAM total
 * - 4× Mi50 (ROCm): 64 GB VRAM total
 * - Total: 8 GPUs, 160 GB VRAM
 *
 * **Domain Structure (per node)**:
 * - GPU_TP_0: 3090_0 + MI50_0 on Rank 0 (HETEROGENEOUS, NUMA 0)
 * - GPU_TP_1: 3090_1 + MI50_1 on Rank 1 (HETEROGENEOUS, NUMA 1)
 * - CPU_TP: CPU0 + CPU1 across Ranks 0+1 (MPI/UPI, NUMA 0+1)
 *
 * **Intra-Node Pipeline** (3 stages):
 * - Stage 0: GPU_TP_0 (Rank 0) → layers 0-4
 * - Stage 1: GPU_TP_1 (Rank 1) → layers 5-9
 * - Stage 2: CPU_TP (Ranks 0+1) → layers 10-13
 *
 * **Inter-Node Pipeline** (PP via InfiniBand):
 * - Node 1 (PP Stage 0) → Node 2 (PP Stage 1)
 *
 * **Tested Models**: 7B, 72B, 120B
 *
 * @see docs/v2/ARCHITECTURE_EXECUTION_SCENARIOS.md
 */

#include <gtest/gtest.h>
#include "../OrchestrationTestBase.h"

using namespace llaminar2;
using namespace llaminar2::test::orchestration;

// =============================================================================
// Scenario 7 Configuration
// =============================================================================

class Test__Scenario7_HeterogeneousPipelineCluster : public OrchestrationTestBase
{
protected:
    ClusterConfig getClusterConfig() override
    {
        ClusterConfig config;
        config.name = "Scenario 7: 2-Node Heterogeneous Pipeline (Cross-Vendor TP)";
        config.num_machines = 2;
        config.sockets_per_machine = 2;

        // Each socket has 1× RTX 3090 + 1× MI50 (cross-vendor TP)
        SocketConfig socket;
        socket.cpu = CPUSpecs::XEON_28C_DDR5;
        socket.gpus = {
            GPUSpecs::RTX_3090, // CUDA
            GPUSpecs::MI50      // ROCm
        };

        config.socket_configs = {socket}; // Same config for all sockets

        // Interconnects
        config.pcie_bandwidth_gbps = 32.0f;        // PCIe 4.0 x16
        config.qpi_upi_bandwidth_gbps = 83.2f;     // UPI 3.0
        config.infiniband_bandwidth_gbps = 400.0f; // HDR400 InfiniBand

        return config;
    }

    std::vector<ModelConfig> getModelConfigs() override
    {
        return {
            ModelConfigs::QWEN2_7B,  // Small - fits in one GPU domain
            ModelConfigs::QWEN2_72B, // Medium - needs full cluster
            // Create a 120B fictional model for testing CPU participation
            {"Qwen2-120B (fictional)", 96, 12288, 43008, 151936, 96, 12,
             70ULL * 1024 * 1024 * 1024, "Q4_0"},
        };
    }

    /**
     * @brief Declarative expectations for what the orchestrator should deduce
     *
     * Given the hardware topology (2 nodes × 2 sockets × (3090 + MI50)), the
     * orchestrator should conclude:
     *
     * - PP = 2 (one stage per physical node, connected via InfiniBand)
     * - GPU TP = 2 (both GPUs per rank: 3090 + MI50 via HETEROGENEOUS backend)
     * - CPU TP = 2 (both ranks per node: CPU0 + CPU1 via UPI)
     * - 4 GPU domains (1 per rank × 4 ranks)
     * - 2 CPU domains (1 per node × 2 nodes)
     * - 3 domains per PP stage (2 GPU + 1 CPU)
     * - Total: 8 GPUs, 4 CPU ranks
     */
    ExpectedParallelismConfig getExpectedParallelism() const override
    {
        ExpectedParallelismConfig expected;

        // Pipeline parallelism: 2 stages (1 per node)
        expected.pp_degree = 2;
        expected.pp_stages_contiguous = true;

        // Tensor parallelism within domains
        expected.gpu_tp_degree = 2; // 2 GPUs per GPU domain (3090 + MI50)
        expected.cpu_tp_degree = 2; // 2 ranks per CPU domain (both sockets)

        // Domain structure
        expected.total_domains = 6;        // 4 GPU + 2 CPU
        expected.gpu_domains_total = 4;    // 1 per rank × 4 ranks
        expected.cpu_domains_total = 2;    // 1 per node × 2 nodes
        expected.domains_per_pp_stage = 3; // 2 GPU + 1 CPU per node

        // Domain types
        expected.gpu_domains_are_intra_rank = true; // HETEROGENEOUS (same rank)
        expected.cpu_domains_are_cross_rank = true; // MPI over UPI (2 ranks)

        // Layer distribution
        expected.gpu_layers_precede_cpu = true;
        expected.min_gpu_layer_fraction = 0.6f; // GPUs get majority
        expected.max_cpu_layer_fraction = 0.4f;

        // Total parallelism
        expected.total_gpu_devices = 8; // 4 domains × 2 GPUs
        expected.total_cpu_ranks = 4;   // 2 domains × 2 ranks

        return expected;
    }
};

// =============================================================================
// Instantiate Standard Orchestration Tests (includes declarative validation)
// =============================================================================

INSTANTIATE_ORCHESTRATION_TESTS(Test__Scenario7_HeterogeneousPipelineCluster);

// =============================================================================
// Scenario-Specific Tests: Cluster Topology
// =============================================================================

TEST_F(Test__Scenario7_HeterogeneousPipelineCluster, ClusterHas4Ranks)
{
    EXPECT_EQ(cluster_.totalRanks(), 4);
    EXPECT_EQ(cluster_.num_machines, 2);
    EXPECT_EQ(cluster_.sockets_per_machine, 2);
}

TEST_F(Test__Scenario7_HeterogeneousPipelineCluster, EachRankHas2GPUs)
{
    for (int rank = 0; rank < cluster_.totalRanks(); ++rank)
    {
        const auto &socket = cluster_.getSocketConfig(rank);
        EXPECT_EQ(socket.gpus.size(), 2u) << "Rank " << rank;
    }
}

TEST_F(Test__Scenario7_HeterogeneousPipelineCluster, GPUsAreMixedVendor)
{
    // Each socket should have exactly 1 NVIDIA and 1 AMD GPU
    const auto &socket_cfg = cluster_.socket_configs[0];

    int cuda_count = 0, rocm_count = 0;
    for (const auto &gpu : socket_cfg.gpus)
    {
        if (gpu.type == DeviceCapability::Type::CUDA)
            cuda_count++;
        if (gpu.type == DeviceCapability::Type::ROCm)
            rocm_count++;
    }

    EXPECT_EQ(cuda_count, 1) << "Socket should have 1 CUDA GPU";
    EXPECT_EQ(rocm_count, 1) << "Socket should have 1 ROCm GPU";
}

TEST_F(Test__Scenario7_HeterogeneousPipelineCluster, TotalVRAMIs160GB)
{
    // 4× RTX 3090 (24 GB) + 4× Mi50 (16 GB) = 96 + 64 = 160 GB
    float expected_vram = 4.0f * 24.0f + 4.0f * 16.0f; // 160 GB
    EXPECT_NEAR(cluster_.totalVRAMGB(), expected_vram, 1.0f);
}

TEST_F(Test__Scenario7_HeterogeneousPipelineCluster, HasPCIeInterconnect)
{
    // Cross-vendor GPU communication uses PCIe
    EXPECT_NEAR(cluster_.pcie_bandwidth_gbps, 32.0f, 1.0f);
}

TEST_F(Test__Scenario7_HeterogeneousPipelineCluster, HasUPIInterconnect)
{
    // Inter-socket (CPU to CPU within node): UPI
    EXPECT_NEAR(cluster_.qpi_upi_bandwidth_gbps, 83.2f, 5.0f);
}

TEST_F(Test__Scenario7_HeterogeneousPipelineCluster, HasInfinibandInterconnect)
{
    // Inter-machine: InfiniBand
    EXPECT_NEAR(cluster_.infiniband_bandwidth_gbps, 400.0f, 10.0f);
}

// =============================================================================
// Scenario-Specific Tests: Cross-Vendor GPU Configuration
// =============================================================================

TEST_F(Test__Scenario7_HeterogeneousPipelineCluster, CrossVendorGPUsDetected)
{
    // Verify the test fixture correctly configures mixed-vendor GPUs
    EXPECT_TRUE(clusterHasHeterogeneousGPUs())
        << "Test fixture should configure mixed NVIDIA (3090) + AMD (MI50) GPUs";
}

// =============================================================================
// Documentation: Expected Orchestrator Conclusions (Human-Readable Summary)
// =============================================================================

TEST_F(Test__Scenario7_HeterogeneousPipelineCluster, Doc_ExpectedOrchestratorConclusions)
{
    // This test documents what the orchestrator SHOULD deduce from the hardware.
    // The actual validation is done declaratively via getExpectedParallelism()
    // and the auto-generated Parallelism_* tests from INSTANTIATE_ORCHESTRATION_TESTS.
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║           EXPECTED ORCHESTRATOR DEDUCTIONS (Scenario 7)              ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════╣\n");
    printf("║ Hardware Presented:                                                  ║\n");
    printf("║   • 2 physical nodes connected via InfiniBand                        ║\n");
    printf("║   • 2 sockets/node × 28-core Xeon CPUs (UPI-connected)               ║\n");
    printf("║   • 2 GPUs/rank: 1× RTX 3090 (24GB) + 1× MI50 (16GB)                 ║\n");
    printf("║   • 256GB DRAM per socket                                            ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════╣\n");
    printf("║ Orchestrator MUST conclude (validated via getExpectedParallelism()): ║\n");
    printf("║                                                                      ║\n");
    printf("║   1. DOMAIN STRUCTURE (6 total):                                     ║\n");
    printf("║      • 4 GPU_INTRA_RANK domains (1 per rank × 4 ranks)               ║\n");
    printf("║      • 2 CPU_CROSS_RANK domains (1 per node × 2 nodes)               ║\n");
    printf("║                                                                      ║\n");
    printf("║   2. TENSOR PARALLELISM:                                             ║\n");
    printf("║      • GPU TP = 2 (3090 + MI50 via PCIe P2P per rank)                ║\n");
    printf("║      • CPU TP = 2 (both sockets via UPI per node)                    ║\n");
    printf("║                                                                      ║\n");
    printf("║   3. PIPELINE PARALLELISM:                                           ║\n");
    printf("║      • PP = 2 (1 stage per physical node via InfiniBand)             ║\n");
    printf("║      • 3 domains per PP stage (2 GPU + 1 CPU)                        ║\n");
    printf("║                                                                      ║\n");
    printf("║   4. LAYER DISTRIBUTION:                                             ║\n");
    printf("║      • GPU domains: early layers (≥60%% of total)                     ║\n");
    printf("║      • CPU domains: later layers (≤40%% of total)                     ║\n");
    printf("║                                                                      ║\n");
    printf("║   5. TOTAL DEVICES:                                                  ║\n");
    printf("║      • 8 GPUs (4 domains × 2 GPUs)                                   ║\n");
    printf("║      • 4 CPU ranks (2 domains × 2 ranks)                             ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════╝\n");
    EXPECT_TRUE(true);
}
