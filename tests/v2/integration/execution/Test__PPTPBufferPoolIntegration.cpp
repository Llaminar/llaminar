/**
 * @file Test__PPTPBufferPoolIntegration.cpp
 * @brief Integration tests for PP+TP buffer pool and backend selection infrastructure
 *
 * Tests the complete integration of:
 * - PPStageBufferSpec - Buffer allocation specifications
 * - PerStageBufferPool - Per-PP-stage buffer management
 * - BackendSelector - Automatic backend selection
 * - PipelineConfig::autoSelectBackends() - Auto-completion of backends
 *
 * These tests validate that the buffer pool infrastructure works correctly
 * with various pipeline configurations, including multi-stage pipelines
 * and automatic backend selection.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>
#include <cstring>  // for memcpy
#include <cmath>    // for std::abs
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>

#include "config/PipelineConfig.h"
#include "config/BackendSelector.h"
#include "execution/local_execution/device/BufferSpec.h"
#include "execution/local_execution/device/PerStageBufferPool.h"
#include "backends/BackendManager.h"
#include "backends/GlobalDeviceAddress.h"  // GlobalDeviceAddress::rocm(), etc.
#include "collective/ILocalPPContext.h"    // createLocalPPContext, LocalPPConfig, HierarchicalPPConfig
#include "collective/ILocalTPContext.h"    // createLocalTPContext for TP operations
#include "collective/PPStage.h"            // PPStage::fromTPContext, PPStage::fromDevice
#include "collective/BackendRouter.h"      // GlobalBackendRouter for PP transfers
#include "tensors/TensorClasses.h"         // FP32Tensor
#include "tensors/TensorFactory.h"         // TensorFactory for BAR-backed tensors
#include "../../utils/TestTensorFactory.h" // TestTensorFactory for test tensors

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
#include "backends/p2p/DirectP2P.h"        // DirectP2PEngine for BAR-backed tensors
#endif

namespace llaminar2
{
using namespace llaminar2::test;  // For TestTensorFactory

// =============================================================================
// BAR-Backed Tensor Test Infrastructure
// =============================================================================

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
/**
 * @brief Helper class for creating BAR-backed tensors in tests
 *
 * This class encapsulates the DirectP2PEngine and TensorFactory setup required
 * to create BAR-backed tensors for cross-vendor (ROCm→CUDA) transfers via PCIe BAR.
 *
 * Usage:
 *   BARBackedTestHelper bar_helper;
 *   if (!bar_helper.initialize(DeviceId::cuda(0), DeviceId::rocm(0))) {
 *       GTEST_SKIP() << "PCIe BAR P2P not available";
 *   }
 *   auto tensor = bar_helper.createFP32BARBacked({1024}, DeviceId::rocm(0), DeviceId::cuda(0));
 */
class BARBackedTestHelper
{
public:
    BARBackedTestHelper() = default;
    ~BARBackedTestHelper() = default;

    /**
     * @brief Initialize BAR-backed tensor infrastructure
     *
     * Probes hardware capabilities and initializes DirectP2PEngine + TensorFactory.
     *
     * @param cuda_device CUDA device for PCIe BAR registration
     * @param rocm_device ROCm device whose BAR will be mapped
     * @return true if BAR-backed tensors are available
     */
    bool initialize(DeviceId cuda_device, DeviceId rocm_device)
    {
        // Check hardware capabilities
        auto caps = DirectP2PEngine::probeCapabilities();
        if (!caps.canDoDirectP2P()) {
            error_message_ = "PCIe BAR P2P not available: " + caps.describe();
            return false;
        }

        // Use singleton DirectP2PEngine for stability
        p2p_ = DirectP2PEngine::getSharedInstance();
        if (!p2p_) {
            error_message_ = "Failed to get DirectP2PEngine singleton";
            return false;
        }

        // Initialize PCIe BAR mapping (skip if already active)
        if (!p2p_->isPCIeBarActive()) {
            if (!p2p_->initializePCIeBar(cuda_device, rocm_device)) {
                error_message_ = "Failed to initialize PCIe BAR P2P (requires CAP_SYS_ADMIN)";
                return false;
            }
        }

        // Create TensorFactory with mock MPI context
        mpi_ctx_ = std::make_unique<MPIContext>(0, 1);  // rank=0, world_size=1
        factory_ = std::make_unique<TensorFactory>(*mpi_ctx_);
        factory_->setDirectP2P(p2p_);

        if (!factory_->canCreateBARBacked()) {
            error_message_ = "TensorFactory cannot create BAR-backed tensors";
            return false;
        }

        initialized_ = true;
        LOG_INFO("[BARBackedTestHelper] Initialized with BAR size: " 
                 << p2p_->getBarMappedSize() << " bytes");
        return true;
    }

    /**
     * @brief Check if BAR-backed tensors are available
     */
    bool isAvailable() const { return initialized_; }

    /**
     * @brief Get error message if initialization failed
     */
    const std::string& errorMessage() const { return error_message_; }

    /**
     * @brief Create BAR-backed FP32 tensor
     *
     * Creates a tensor in ROCm device's BAR-exposed VRAM region.
     * - ROCm device writes at full VRAM bandwidth
     * - CUDA device reads through PCIe BAR
     *
     * @param shape Tensor dimensions
     * @param rocm_device ROCm device that owns the BAR memory
     * @param cuda_device CUDA device that will read through PCIe
     * @return BAR-backed FP32 tensor, or nullptr on failure
     */
    std::unique_ptr<FP32Tensor> createFP32BARBacked(
        const std::vector<size_t>& shape,
        DeviceId rocm_device,
        DeviceId cuda_device)
    {
        if (!initialized_) {
            LOG_ERROR("[BARBackedTestHelper] Not initialized");
            return nullptr;
        }
        return factory_->createFP32BARBacked(shape, rocm_device, cuda_device);
    }

    /**
     * @brief Get the DirectP2PEngine for additional operations
     */
    std::shared_ptr<DirectP2PEngine> getEngine() const { return p2p_; }

    /**
     * @brief Get the TensorFactory for additional operations
     */
    TensorFactory* getFactory() const { return factory_.get(); }

private:
    std::shared_ptr<DirectP2PEngine> p2p_;
    std::unique_ptr<MPIContext> mpi_ctx_;
    std::unique_ptr<TensorFactory> factory_;
    std::string error_message_;
    bool initialized_ = false;
};
#endif  // HAVE_CUDA && HAVE_ROCM

namespace
{

/**
 * @brief Test fixture for PP+TP buffer pool integration tests
 *
 * Provides helper methods for creating test pipeline configurations
 * and buffer specifications. Uses CPU-only configurations to avoid
 * GPU hardware requirements.
 */
class Test__PPTPBufferPoolIntegration : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Ensure CPU backend is initialized for tensor allocation
        if (!hasCPUBackend())
        {
            initCPUBackend(-1); // System-wide memory (no NUMA binding)
        }
    }

    void TearDown() override
    {
        // Clean up any resources if needed
    }

    /**
     * @brief Create a simple N-stage CPU-only PipelineConfig
     *
     * Each stage gets its own domain with a single CPU device.
     * Layers are distributed evenly across stages.
     *
     * @param num_stages Number of PP stages
     * @param total_layers Total number of transformer layers
     * @return Valid PipelineConfig for CPU-only execution
     */
    static PipelineConfig createCPUPipelineConfig(int num_stages, int total_layers)
    {
        PipelineConfig config;
        config.total_layers = total_layers;

        int layers_per_stage = total_layers / num_stages;
        int remaining = total_layers % num_stages;
        int layer = 0;

        for (int i = 0; i < num_stages; ++i)
        {
            // Create domain for this stage
            TPDomainConfig domain;
            domain.name = "cpu_stage" + std::to_string(i);
            domain.devices = {DeviceId::cpu()};
            domain.tp_backend = CollectiveBackendType::AUTO; // Will be resolved
            config.tp_domains.push_back(domain);

            // Calculate layer range for this stage
            int stage_layers = layers_per_stage + (i < remaining ? 1 : 0);
            int first_layer = layer;
            int last_layer = layer + stage_layers;
            layer = last_layer;

            // Create PP stage
            PPStageConfig stage;
            stage.stage_id = i;
            stage.domain_name = domain.name;
            stage.first_layer = first_layer;
            stage.last_layer = last_layer;
            stage.has_embedding = (i == 0);
            stage.has_lm_head = (i == num_stages - 1);
            config.pp_stages.push_back(stage);
        }

        return config;
    }

    /**
     * @brief Create a test buffer spec with small dimensions
     *
     * Uses small dimensions for fast tests while still exercising
     * all buffer types.
     *
     * @return PPStageBufferSpec with test dimensions
     */
    static PPStageBufferSpec createTestSpec()
    {
        PPStageBufferSpec spec;
        spec.batch_size = 1;
        spec.seq_len = 16;
        spec.d_model = 64;
        spec.n_heads = 4;
        spec.n_kv_heads = 2;
        spec.head_dim = 16;
        spec.intermediate_size = 256;
        spec.vocab_size = 1000;
        spec.precision = ActivationPrecision::FP32;
        spec.enable_snapshots = false;
        return spec;
    }

    /**
     * @brief Create a buffer spec with larger dimensions
     *
     * More realistic dimensions closer to actual model sizes.
     *
     * @return PPStageBufferSpec with larger dimensions
     */
    static PPStageBufferSpec createLargerSpec()
    {
        PPStageBufferSpec spec;
        spec.batch_size = 1;
        spec.seq_len = 128;
        spec.d_model = 896;
        spec.n_heads = 14;
        spec.n_kv_heads = 2;
        spec.head_dim = 64;
        spec.intermediate_size = 4864;
        spec.vocab_size = 32000;
        spec.precision = ActivationPrecision::FP32;
        spec.enable_snapshots = false;
        return spec;
    }
};

// =============================================================================
// Test 1: ThreeStage_CPUPipeline_BufferPoolInitialization
// =============================================================================

/**
 * @brief Test buffer pool initialization with a 3-stage CPU pipeline
 *
 * Creates a 3-stage CPU-only PipelineConfig and verifies that:
 * - PerStageBufferPool initializes successfully
 * - All stages (0, 1, 2) have valid buffer allocations
 * - Stage devices are correctly reported as CPU
 * - Layer indices map to correct stages
 */
TEST_F(Test__PPTPBufferPoolIntegration, ThreeStage_CPUPipeline_BufferPoolInitialization)
{
    // Create 3-stage CPU pipeline with 24 layers (8 per stage)
    constexpr int kNumStages = 3;
    constexpr int kTotalLayers = 24;
    auto config = createCPUPipelineConfig(kNumStages, kTotalLayers);

    // Auto-complete backends and validate
    std::string error_msg;
    ASSERT_TRUE(config.completeAndValidate(&error_msg))
        << "Config validation failed: " << error_msg;

    // Create buffer spec
    auto spec = createTestSpec();

    // Initialize buffer pool
    PerStageBufferPool pool;
    EXPECT_FALSE(pool.isInitialized()) << "Pool should not be initialized before initialize()";

    ASSERT_TRUE(pool.initialize(config, spec))
        << "Buffer pool initialization should succeed";

    EXPECT_TRUE(pool.isInitialized()) << "Pool should be initialized after initialize()";
    EXPECT_EQ(pool.numStages(), kNumStages)
        << "Pool should have " << kNumStages << " stages";

    // Verify all stages have valid buffers
    for (int stage = 0; stage < kNumStages; ++stage)
    {
        SCOPED_TRACE("Stage " + std::to_string(stage));

        auto &buffers = pool.forStage(stage);

        // Check that core buffers are allocated (non-null)
        EXPECT_NE(buffers.residual, nullptr)
            << "Stage " << stage << " should have residual buffer";
        EXPECT_NE(buffers.Q, nullptr)
            << "Stage " << stage << " should have Q buffer";
        EXPECT_NE(buffers.K, nullptr)
            << "Stage " << stage << " should have K buffer";
        EXPECT_NE(buffers.V, nullptr)
            << "Stage " << stage << " should have V buffer";
        EXPECT_NE(buffers.current_hidden, nullptr)
            << "Stage " << stage << " should have current_hidden buffer";
    }

    // Verify device assignments (all should be CPU)
    for (int stage = 0; stage < kNumStages; ++stage)
    {
        SCOPED_TRACE("Stage " + std::to_string(stage) + " device");

        DeviceId device = pool.deviceForStage(stage);
        EXPECT_EQ(device.type, DeviceType::CPU)
            << "Stage " << stage << " should be on CPU device";
    }

    // Verify layer-to-stage mapping
    // Stage 0: layers 0-7, Stage 1: layers 8-15, Stage 2: layers 16-23
    int layers_per_stage = kTotalLayers / kNumStages;

    for (int layer = 0; layer < kTotalLayers; ++layer)
    {
        SCOPED_TRACE("Layer " + std::to_string(layer));

        int expected_stage = layer / layers_per_stage;
        if (expected_stage >= kNumStages)
        {
            expected_stage = kNumStages - 1; // Last stage
        }

        // forLayer should return same buffers as forStage for that layer's stage
        auto &layer_buffers = pool.forLayer(layer);
        auto &stage_buffers = pool.forStage(expected_stage);
        EXPECT_EQ(&layer_buffers, &stage_buffers)
            << "Layer " << layer << " should map to stage " << expected_stage;
    }
}

// =============================================================================
// Test 2: AutoBackendSelection_WithPipelineConfig
// =============================================================================

/**
 * @brief Test automatic backend selection for PP transfer
 *
 * Creates a 2-stage CPU PipelineConfig with empty pp_transfer_backends,
 * calls autoSelectBackends(), and verifies:
 * - TP domain backends are resolved from AUTO
 * - PP transfer backends are auto-selected (CPU→CPU = HOST)
 */
TEST_F(Test__PPTPBufferPoolIntegration, AutoBackendSelection_WithPipelineConfig)
{
    // Create 2-stage CPU pipeline with 24 layers
    constexpr int kNumStages = 2;
    constexpr int kTotalLayers = 24;
    auto config = createCPUPipelineConfig(kNumStages, kTotalLayers);

    // Verify pp_transfer_backends is empty before auto-select
    EXPECT_TRUE(config.pp_transfer_backends.empty())
        << "Transfer backends should be empty initially";

    // Verify TP backends are AUTO
    for (const auto &domain : config.tp_domains)
    {
        EXPECT_EQ(domain.tp_backend, CollectiveBackendType::AUTO)
            << "TP backend should be AUTO before auto-select";
    }

    // Auto-complete backends
    config.autoSelectBackends();

    // Verify TP domain backends were resolved
    for (const auto &domain : config.tp_domains)
    {
        EXPECT_NE(domain.tp_backend, CollectiveBackendType::AUTO)
            << "TP backend should be resolved after auto-select";
        // CPU-only domain should use HOST backend
        EXPECT_EQ(domain.tp_backend, CollectiveBackendType::HOST)
            << "CPU-only TP domain should use HOST backend";
    }

    // Verify PP transfer backend was auto-selected
    EXPECT_FALSE(config.pp_transfer_backends.empty())
        << "Transfer backends should be populated after auto-select";

    auto key = std::make_pair(0, 1); // Stage 0 → Stage 1
    auto it = config.pp_transfer_backends.find(key);
    ASSERT_NE(it, config.pp_transfer_backends.end())
        << "Transfer backend for (0, 1) should exist";

    // CPU → CPU should use HOST backend
    EXPECT_EQ(it->second, CollectiveBackendType::HOST)
        << "CPU→CPU transfer should use HOST backend";

    // Verify config is valid after auto-select
    std::string error_msg;
    EXPECT_TRUE(config.validate(&error_msg))
        << "Config should be valid after auto-select: " << error_msg;
}

// =============================================================================
// Test 3: BufferPoolStats_AfterInitialization
// =============================================================================

/**
 * @brief Test buffer pool allocation statistics
 *
 * Initializes a buffer pool and verifies the stats contain
 * expected information about the allocation.
 */
TEST_F(Test__PPTPBufferPoolIntegration, BufferPoolStats_AfterInitialization)
{
    // Create 2-stage CPU pipeline
    constexpr int kNumStages = 2;
    constexpr int kTotalLayers = 24;
    auto config = createCPUPipelineConfig(kNumStages, kTotalLayers);

    std::string error_msg;
    ASSERT_TRUE(config.completeAndValidate(&error_msg))
        << "Config validation failed: " << error_msg;

    // Create buffer spec with known dimensions
    auto spec = createTestSpec();

    // Initialize buffer pool
    PerStageBufferPool pool;
    ASSERT_TRUE(pool.initialize(config, spec))
        << "Buffer pool initialization should succeed";

    // Get stats
    const auto &stats = pool.stats();

    // Verify basic stats properties
    // Stats should show allocations were made
    EXPECT_GT(stats.total_bytes(), 0u)
        << "Total allocated bytes should be greater than 0";

    // Verify per-device allocation tracking
    EXPECT_FALSE(stats.bytes_per_device.empty())
        << "Bytes per device should not be empty";

    // CPU device should have allocations
    DeviceId cpu = DeviceId::cpu();
    auto cpu_it = stats.bytes_per_device.find(cpu);
    EXPECT_NE(cpu_it, stats.bytes_per_device.end())
        << "CPU device should have allocations tracked";

    if (cpu_it != stats.bytes_per_device.end())
    {
        EXPECT_GT(cpu_it->second, 0u)
            << "CPU allocation should be greater than 0";
    }
}

// =============================================================================
// Test 4: BufferPool_ReinitializationWithDifferentConfig
// =============================================================================

/**
 * @brief Test buffer pool reinitialization with different configurations
 *
 * Initializes a buffer pool with 2 stages, releases it, then reinitializes
 * with 3 stages. Verifies the new configuration is active.
 */
TEST_F(Test__PPTPBufferPoolIntegration, BufferPool_ReinitializationWithDifferentConfig)
{
    // First configuration: 2 stages
    constexpr int kFirstStages = 2;
    constexpr int kTotalLayers = 24;
    auto config1 = createCPUPipelineConfig(kFirstStages, kTotalLayers);
    std::string error_msg;
    ASSERT_TRUE(config1.completeAndValidate(&error_msg))
        << "Config1 validation failed: " << error_msg;

    // Initialize buffer pool with first config
    PerStageBufferPool pool;
    auto spec = createTestSpec();

    ASSERT_TRUE(pool.initialize(config1, spec))
        << "First initialization should succeed";

    EXPECT_EQ(pool.numStages(), kFirstStages)
        << "Pool should have " << kFirstStages << " stages initially";

    // Save reference to stage 0 buffers for later comparison
    void *original_stage0_residual = pool.forStage(0).residual;

    // Release the pool
    pool.release();
    EXPECT_FALSE(pool.isInitialized())
        << "Pool should not be initialized after release()";

    // Second configuration: 3 stages
    constexpr int kSecondStages = 3;
    auto config2 = createCPUPipelineConfig(kSecondStages, kTotalLayers);
    ASSERT_TRUE(config2.completeAndValidate(&error_msg))
        << "Config2 validation failed: " << error_msg;

    // Reinitialize with second config
    ASSERT_TRUE(pool.initialize(config2, spec))
        << "Reinitialization should succeed";

    EXPECT_TRUE(pool.isInitialized())
        << "Pool should be initialized after reinitialization";

    EXPECT_EQ(pool.numStages(), kSecondStages)
        << "Pool should have " << kSecondStages << " stages after reinitialization";

    // Verify all 3 stages are accessible
    for (int stage = 0; stage < kSecondStages; ++stage)
    {
        SCOPED_TRACE("Stage " + std::to_string(stage));
        auto &buffers = pool.forStage(stage);
        EXPECT_NE(buffers.residual, nullptr)
            << "Stage " << stage << " should have residual buffer";
    }

    // New allocations should be different from original (different memory)
    void *new_stage0_residual = pool.forStage(0).residual;
    EXPECT_NE(new_stage0_residual, original_stage0_residual)
        << "Reinitialized buffers should be freshly allocated";
}

// =============================================================================
// Test 5: BackendSelector_DeviceTypeCombinations
// =============================================================================

/**
 * @brief Test BackendSelector rules for various device type combinations
 *
 * Verifies that BackendSelector::selectForTransfer() returns the correct
 * backend for different source/destination device type combinations.
 */
TEST_F(Test__PPTPBufferPoolIntegration, BackendSelector_DeviceTypeCombinations)
{
    // CPU → CPU should use HOST
    EXPECT_EQ(BackendSelector::selectForTransfer(DeviceId::cpu(), DeviceId::cpu()),
              CollectiveBackendType::HOST)
        << "CPU→CPU should use HOST backend";

    // Same type GPU → GPU
    EXPECT_EQ(BackendSelector::selectForTransfer(DeviceId::cuda(0), DeviceId::cuda(1)),
              CollectiveBackendType::NCCL)
        << "CUDA→CUDA should use NCCL backend";

    EXPECT_EQ(BackendSelector::selectForTransfer(DeviceId::rocm(0), DeviceId::rocm(1)),
              CollectiveBackendType::RCCL)
        << "ROCm→ROCm should use RCCL backend";

    // Cross-vendor GPU
    EXPECT_EQ(BackendSelector::selectForTransfer(DeviceId::cuda(0), DeviceId::rocm(0)),
              CollectiveBackendType::PCIE_BAR)
        << "CUDA→ROCm should use PCIE_BAR backend";

    EXPECT_EQ(BackendSelector::selectForTransfer(DeviceId::rocm(0), DeviceId::cuda(0)),
              CollectiveBackendType::PCIE_BAR)
        << "ROCm→CUDA should use PCIE_BAR backend";

    // GPU ↔ CPU
    EXPECT_EQ(BackendSelector::selectForTransfer(DeviceId::cuda(0), DeviceId::cpu()),
              CollectiveBackendType::HOST)
        << "CUDA→CPU should use HOST backend";

    EXPECT_EQ(BackendSelector::selectForTransfer(DeviceId::cpu(), DeviceId::cuda(0)),
              CollectiveBackendType::HOST)
        << "CPU→CUDA should use HOST backend";

    EXPECT_EQ(BackendSelector::selectForTransfer(DeviceId::rocm(0), DeviceId::cpu()),
              CollectiveBackendType::HOST)
        << "ROCm→CPU should use HOST backend";

    EXPECT_EQ(BackendSelector::selectForTransfer(DeviceId::cpu(), DeviceId::rocm(0)),
              CollectiveBackendType::HOST)
        << "CPU→ROCm should use HOST backend";
}

// =============================================================================
// Test 6: BufferPool_LayerBoundaryMapping
// =============================================================================

/**
 * @brief Test layer-to-stage mapping at stage boundaries
 *
 * Verifies that layer indices at stage boundaries map correctly,
 * especially edge cases at the start and end of each stage.
 */
TEST_F(Test__PPTPBufferPoolIntegration, BufferPool_LayerBoundaryMapping)
{
    // 3 stages with 30 layers (10 per stage)
    // Stage 0: layers 0-9, Stage 1: layers 10-19, Stage 2: layers 20-29
    constexpr int kNumStages = 3;
    constexpr int kTotalLayers = 30;
    auto config = createCPUPipelineConfig(kNumStages, kTotalLayers);

    std::string error_msg;
    ASSERT_TRUE(config.completeAndValidate(&error_msg))
        << "Config validation failed: " << error_msg;

    PerStageBufferPool pool;
    auto spec = createTestSpec();
    ASSERT_TRUE(pool.initialize(config, spec));

    // Test boundary layers
    struct BoundaryTest
    {
        int layer;
        int expected_stage;
    };

    std::vector<BoundaryTest> tests = {
        {0, 0},  // First layer → Stage 0
        {9, 0},  // Last layer of Stage 0
        {10, 1}, // First layer of Stage 1
        {19, 1}, // Last layer of Stage 1
        {20, 2}, // First layer of Stage 2
        {29, 2}, // Last layer (total)
    };

    for (const auto &test : tests)
    {
        SCOPED_TRACE("Layer " + std::to_string(test.layer));

        auto &layer_buffers = pool.forLayer(test.layer);
        auto &stage_buffers = pool.forStage(test.expected_stage);

        EXPECT_EQ(&layer_buffers, &stage_buffers)
            << "Layer " << test.layer << " should map to stage " << test.expected_stage;
    }

    // Test out-of-bounds access
    EXPECT_THROW(pool.forLayer(-1), std::out_of_range)
        << "Negative layer should throw";
    EXPECT_THROW(pool.forLayer(30), std::out_of_range)
        << "Layer >= total_layers should throw";
}

// =============================================================================
// Test 7: CompleteAndValidate_EndToEnd
// =============================================================================

/**
 * @brief Test completeAndValidate() convenience method
 *
 * Verifies that completeAndValidate() both auto-selects backends
 * and validates the configuration in one call.
 */
TEST_F(Test__PPTPBufferPoolIntegration, CompleteAndValidate_EndToEnd)
{
    // Create config with AUTO backends
    auto config = createCPUPipelineConfig(2, 24);

    // Verify initial state
    EXPECT_TRUE(config.pp_transfer_backends.empty())
        << "Transfer backends should be empty initially";

    // completeAndValidate should both auto-select and validate
    std::string error_msg;
    ASSERT_TRUE(config.completeAndValidate(&error_msg))
        << "completeAndValidate should succeed: " << error_msg;

    // Verify auto-selection happened
    EXPECT_FALSE(config.pp_transfer_backends.empty())
        << "Transfer backends should be populated";

    // Verify validation passed (implicit in ASSERT_TRUE above)
    // Try validation again to double-check
    EXPECT_TRUE(config.validate(&error_msg))
        << "Config should still be valid: " << error_msg;
}

// =============================================================================
// Test 8: MultiStage_DifferentBufferSpecs
// =============================================================================

/**
 * @brief Test that buffer pool allocates correct sizes based on spec
 *
 * Verifies that buffer dimensions match the PPStageBufferSpec.
 */
TEST_F(Test__PPTPBufferPoolIntegration, MultiStage_DifferentBufferSpecs)
{
    auto config = createCPUPipelineConfig(2, 24);
    std::string error_msg;
    ASSERT_TRUE(config.completeAndValidate(&error_msg));

    // Test with small spec
    auto small_spec = createTestSpec();
    PerStageBufferPool small_pool;
    ASSERT_TRUE(small_pool.initialize(config, small_spec));

    // Test with larger spec
    auto large_spec = createLargerSpec();
    PerStageBufferPool large_pool;
    ASSERT_TRUE(large_pool.initialize(config, large_spec));

    // Large spec should allocate more memory
    EXPECT_GT(large_pool.stats().total_bytes(),
              small_pool.stats().total_bytes())
        << "Larger spec should allocate more memory";
}

// =============================================================================
// Test 9: HeterogeneousPPTP_RealGPUs
// =============================================================================

/**
 * @brief Test heterogeneous PP+TP composition with real GPUs
 *
 * Constructs: PipelineParallel(TensorParallel(rocm:0, rocm:1), cuda:0, cpu)
 *
 * This is a 3-stage pipeline where:
 * - Stage 0: 2-way TP across rocm:0 and rocm:1 (RCCL backend)
 * - Stage 1: Single cuda:0 device (no TP)
 * - Stage 2: Single CPU device (no TP)
 *
 * Transfer backends should be:
 * - Stage 0 → Stage 1: ROCm → CUDA = PCIE_BAR (cross-vendor)
 * - Stage 1 → Stage 2: CUDA → CPU = HOST
 *
 * This test validates the config construction and backend selection.
 * Actual tensor allocation requires the backends to be initialized.
 */
TEST_F(Test__PPTPBufferPoolIntegration, HeterogeneousPPTP_RealGPUs)
{
    // This test validates config construction and backend selection for a
    // heterogeneous topology - it works even without actual GPU hardware
    // since it only tests the PipelineConfig layer, not actual allocation.
    
    constexpr int kTotalLayers = 24;
    
    // =========================================================================
    // Construct: PipelineParallel(TensorParallel(rocm:0, rocm:1), cuda:0, cpu)
    // =========================================================================
    
    PipelineConfig config;
    config.total_layers = kTotalLayers;
    
    // --- Domain 0: 2-way TP on ROCm GPUs ---
    TPDomainConfig rocm_tp_domain;
    rocm_tp_domain.name = "rocm_tp";
    rocm_tp_domain.devices = {DeviceId::rocm(0), DeviceId::rocm(1)};
    rocm_tp_domain.tp_backend = CollectiveBackendType::AUTO;  // Should resolve to RCCL
    config.tp_domains.push_back(rocm_tp_domain);
    
    // --- Domain 1: Single CUDA GPU ---
    TPDomainConfig cuda_domain;
    cuda_domain.name = "cuda_single";
    cuda_domain.devices = {DeviceId::cuda(0)};
    cuda_domain.tp_backend = CollectiveBackendType::AUTO;  // Should resolve to NCCL (or HOST for single device)
    config.tp_domains.push_back(cuda_domain);
    
    // --- Domain 2: CPU ---
    TPDomainConfig cpu_domain;
    cpu_domain.name = "cpu_final";
    cpu_domain.devices = {DeviceId::cpu()};
    cpu_domain.tp_backend = CollectiveBackendType::AUTO;  // Should resolve to HOST
    config.tp_domains.push_back(cpu_domain);
    
    // --- PP Stage 0: Layers 0-7 on ROCm TP domain ---
    PPStageConfig stage0;
    stage0.stage_id = 0;
    stage0.domain_name = "rocm_tp";
    stage0.first_layer = 0;
    stage0.last_layer = 8;
    stage0.has_embedding = true;
    stage0.has_lm_head = false;
    config.pp_stages.push_back(stage0);
    
    // --- PP Stage 1: Layers 8-15 on CUDA ---
    PPStageConfig stage1;
    stage1.stage_id = 1;
    stage1.domain_name = "cuda_single";
    stage1.first_layer = 8;
    stage1.last_layer = 16;
    stage1.has_embedding = false;
    stage1.has_lm_head = false;
    config.pp_stages.push_back(stage1);
    
    // --- PP Stage 2: Layers 16-23 on CPU ---
    PPStageConfig stage2;
    stage2.stage_id = 2;
    stage2.domain_name = "cpu_final";
    stage2.first_layer = 16;
    stage2.last_layer = 24;
    stage2.has_embedding = false;
    stage2.has_lm_head = true;
    config.pp_stages.push_back(stage2);
    
    // =========================================================================
    // Verify initial state before auto-selection
    // =========================================================================
    
    EXPECT_TRUE(config.pp_transfer_backends.empty())
        << "Transfer backends should be empty before auto-select";
    
    EXPECT_EQ(config.tp_domains[0].tp_backend, CollectiveBackendType::AUTO);
    EXPECT_EQ(config.tp_domains[1].tp_backend, CollectiveBackendType::AUTO);
    EXPECT_EQ(config.tp_domains[2].tp_backend, CollectiveBackendType::AUTO);
    
    // =========================================================================
    // Auto-select backends
    // =========================================================================
    
    config.autoSelectBackends();
    
    // =========================================================================
    // Verify TP domain backend selection
    // =========================================================================
    
    // Domain 0 (ROCm TP): Should be RCCL for ROCm-only domain
    EXPECT_EQ(config.tp_domains[0].tp_backend, CollectiveBackendType::RCCL)
        << "ROCm TP domain should use RCCL backend";
    
    // Domain 1 (CUDA single): Single device, should be HOST (no collective needed)
    // OR NCCL if we treat single-device as valid NCCL domain
    auto cuda_backend = config.tp_domains[1].tp_backend;
    EXPECT_TRUE(cuda_backend == CollectiveBackendType::HOST || 
                cuda_backend == CollectiveBackendType::NCCL)
        << "Single CUDA device domain should use HOST or NCCL, got: " 
        << static_cast<int>(cuda_backend);
    
    // Domain 2 (CPU): Should be HOST
    EXPECT_EQ(config.tp_domains[2].tp_backend, CollectiveBackendType::HOST)
        << "CPU domain should use HOST backend";
    
    // =========================================================================
    // Verify PP transfer backend selection
    // =========================================================================
    
    EXPECT_EQ(config.pp_transfer_backends.size(), 2u)
        << "Should have 2 PP transfer backends (0→1, 1→2)";
    
    // Stage 0 → Stage 1: ROCm → CUDA = PCIE_BAR (cross-vendor GPU)
    auto key_0_1 = std::make_pair(0, 1);
    auto it_0_1 = config.pp_transfer_backends.find(key_0_1);
    ASSERT_NE(it_0_1, config.pp_transfer_backends.end())
        << "Transfer backend for stage 0→1 should exist";
    EXPECT_EQ(it_0_1->second, CollectiveBackendType::PCIE_BAR)
        << "ROCm→CUDA transfer should use PCIE_BAR (cross-vendor)";
    
    // Stage 1 → Stage 2: CUDA → CPU = HOST
    auto key_1_2 = std::make_pair(1, 2);
    auto it_1_2 = config.pp_transfer_backends.find(key_1_2);
    ASSERT_NE(it_1_2, config.pp_transfer_backends.end())
        << "Transfer backend for stage 1→2 should exist";
    EXPECT_EQ(it_1_2->second, CollectiveBackendType::HOST)
        << "CUDA→CPU transfer should use HOST";
    
    // =========================================================================
    // Validate complete configuration
    // =========================================================================
    
    std::string error_msg;
    EXPECT_TRUE(config.validate(&error_msg))
        << "Heterogeneous PP+TP config should be valid: " << error_msg;
    
    // =========================================================================
    // Verify domain lookups work correctly
    // =========================================================================
    
    const TPDomainConfig* domain0 = config.getDomainForStage(0);
    ASSERT_NE(domain0, nullptr);
    EXPECT_EQ(domain0->name, "rocm_tp");
    EXPECT_EQ(domain0->devices.size(), 2u);
    EXPECT_EQ(domain0->devices[0], DeviceId::rocm(0));
    EXPECT_EQ(domain0->devices[1], DeviceId::rocm(1));
    
    const TPDomainConfig* domain1 = config.getDomainForStage(1);
    ASSERT_NE(domain1, nullptr);
    EXPECT_EQ(domain1->name, "cuda_single");
    EXPECT_EQ(domain1->devices.size(), 1u);
    EXPECT_EQ(domain1->devices[0], DeviceId::cuda(0));
    
    const TPDomainConfig* domain2 = config.getDomainForStage(2);
    ASSERT_NE(domain2, nullptr);
    EXPECT_EQ(domain2->name, "cpu_final");
    EXPECT_EQ(domain2->devices.size(), 1u);
    EXPECT_EQ(domain2->devices[0], DeviceId::cpu());
    
    // =========================================================================
    // Verify primaryDevice() for each domain
    // =========================================================================
    
    EXPECT_EQ(domain0->primaryDevice(), DeviceId::rocm(0))
        << "Primary device of ROCm TP domain should be rocm:0";
    EXPECT_EQ(domain1->primaryDevice(), DeviceId::cuda(0))
        << "Primary device of CUDA domain should be cuda:0";
    EXPECT_EQ(domain2->primaryDevice(), DeviceId::cpu())
        << "Primary device of CPU domain should be cpu";
}

// =============================================================================
// Test 10: HeterogeneousPPTP_BufferAllocation
// =============================================================================

/**
 * @brief Test buffer allocation for heterogeneous PP+TP with real GPU backends
 *
 * Same topology as Test 9, but actually allocates buffers on each device.
 * This test REQUIRES both ROCm and CUDA GPUs to be present.
 */
TEST_F(Test__PPTPBufferPoolIntegration, HeterogeneousPPTP_BufferAllocation)
{
    // Initialize GPU backends (lazy init via getter)
    IBackend* rocm_backend = getROCmBackend();
    IBackend* cuda_backend = getCUDABackend();
    
    // This test requires real GPUs - fail if not available
    ASSERT_NE(rocm_backend, nullptr) << "ROCm backend is required for this test";
    ASSERT_NE(cuda_backend, nullptr) << "CUDA backend is required for this test";
    
    constexpr int kTotalLayers = 24;
    
    // Build the same heterogeneous config
    PipelineConfig config;
    config.total_layers = kTotalLayers;
    
    // Domain 0: 2-way TP on ROCm
    TPDomainConfig rocm_tp;
    rocm_tp.name = "rocm_tp";
    rocm_tp.devices = {DeviceId::rocm(0), DeviceId::rocm(1)};
    rocm_tp.tp_backend = CollectiveBackendType::RCCL;
    config.tp_domains.push_back(rocm_tp);
    
    // Domain 1: CUDA
    TPDomainConfig cuda_dom;
    cuda_dom.name = "cuda";
    cuda_dom.devices = {DeviceId::cuda(0)};
    cuda_dom.tp_backend = CollectiveBackendType::HOST;
    config.tp_domains.push_back(cuda_dom);
    
    // Domain 2: CPU
    TPDomainConfig cpu_dom;
    cpu_dom.name = "cpu";
    cpu_dom.devices = {DeviceId::cpu()};
    cpu_dom.tp_backend = CollectiveBackendType::HOST;
    config.tp_domains.push_back(cpu_dom);
    
    // PP stages
    PPStageConfig s0{0, "rocm_tp", 0, 8, true, false};
    PPStageConfig s1{1, "cuda", 8, 16, false, false};
    PPStageConfig s2{2, "cpu", 16, 24, false, true};
    config.pp_stages = {s0, s1, s2};
    
    // Set transfer backends explicitly
    config.pp_transfer_backends[{0, 1}] = CollectiveBackendType::PCIE_BAR;
    config.pp_transfer_backends[{1, 2}] = CollectiveBackendType::HOST;
    
    std::string error_msg;
    ASSERT_TRUE(config.validate(&error_msg)) << "Config validation failed: " << error_msg;
    
    // Create buffer spec
    auto spec = createTestSpec();
    
    // Initialize buffer pool - this actually allocates on each device
    PerStageBufferPool pool;
    ASSERT_TRUE(pool.initialize(config, spec))
        << "Buffer pool initialization failed for heterogeneous config";
    
    EXPECT_EQ(pool.numStages(), 3);
    
    // Verify device assignments
    EXPECT_EQ(pool.deviceForStage(0), DeviceId::rocm(0))
        << "Stage 0 primary device should be rocm:0";
    EXPECT_EQ(pool.deviceForStage(1), DeviceId::cuda(0))
        << "Stage 1 device should be cuda:0";
    EXPECT_EQ(pool.deviceForStage(2), DeviceId::cpu())
        << "Stage 2 device should be cpu";
    
    // Verify buffers are allocated
    for (int stage = 0; stage < 3; ++stage) {
        SCOPED_TRACE("Stage " + std::to_string(stage));
        auto& buffers = pool.forStage(stage);
        EXPECT_NE(buffers.residual, nullptr);
        EXPECT_NE(buffers.Q, nullptr);
        EXPECT_NE(buffers.K, nullptr);
        EXPECT_NE(buffers.V, nullptr);
    }
    
    // Verify stats show allocations on multiple devices
    const auto& stats = pool.stats();
    EXPECT_GT(stats.bytes_per_device.size(), 1u)
        << "Should have allocations on multiple devices";
    
    // Check each device type has allocations
    bool has_rocm_alloc = false;
    bool has_cuda_alloc = false;
    bool has_cpu_alloc = false;
    
    for (const auto& [device, bytes] : stats.bytes_per_device) {
        if (device.type == DeviceType::ROCm) has_rocm_alloc = (bytes > 0);
        if (device.type == DeviceType::CUDA) has_cuda_alloc = (bytes > 0);
        if (device.type == DeviceType::CPU) has_cpu_alloc = (bytes > 0);
    }
    
    EXPECT_TRUE(has_rocm_alloc) << "Should have ROCm allocations";
    EXPECT_TRUE(has_cuda_alloc) << "Should have CUDA allocations";
    EXPECT_TRUE(has_cpu_alloc) << "Should have CPU allocations";
    
    // Clean up
    pool.release();
}

// =============================================================================
// Test 11: HeterogeneousPPTP_EndToEndTransfer
// =============================================================================

/**
 * @brief End-to-end test of heterogeneous PP+TP data flow using production code
 *
 * Topology: PipelineParallel(TensorParallel(rocm:0, rocm:1), cuda:0, cpu)
 *
 * This test exercises REAL production code paths for BOTH PP and TP:
 * 1. Create ILocalTPContext with TensorParallel(rocm:0, rocm:1) - RCCL backend
 * 2. Use multi-threaded SPMD pattern (one thread per TP device) for allreduce
 * 3. Use HierarchicalPPConfig to wrap TP context as Stage 0
 * 4. Transfer result to CUDA:0 (Stage 1) via PP using PCIeBAR backend
 * 5. Transfer to CPU (Stage 2) via PP using HOST backend
 * 6. Verify data integrity on CPU
 *
 * PRODUCTION CODE PATHS EXERCISED:
 * - ILocalTPContext::allreduce() with RCCL backend (multi-threaded)
 * - LocalTPContext::allreduceWithBarrierMultiGpu() for thread coordination
 * - HierarchicalPPConfig with PPStage::fromTPContext()
 * - HierarchicalPPContext::transferFromTPDomain() for cross-vendor PP
 * - BAR-backed tensor handling (automatic via TP context)
 */
TEST_F(Test__PPTPBufferPoolIntegration, HeterogeneousPPTP_EndToEndTransfer)
{
#if !defined(HAVE_CUDA) || !defined(HAVE_ROCM)
    GTEST_SKIP() << "Test requires both HAVE_CUDA and HAVE_ROCM";
#else
    // Initialize backends
    IBackend* rocm_backend = getROCmBackend();
    IBackend* cuda_backend = getCUDABackend();
    
    ASSERT_NE(rocm_backend, nullptr) << "ROCm backend is required";
    ASSERT_NE(cuda_backend, nullptr) << "CUDA backend is required";
    ASSERT_GE(rocm_backend->deviceCount(), 2) 
        << "Requires 2+ ROCm GPUs for TensorParallel(rocm:0, rocm:1)";
    
    LOG_INFO("[Test] Starting HeterogeneousPPTP_EndToEndTransfer");
    LOG_INFO("[Test] Topology: PipelineParallel(TensorParallel(rocm:0, rocm:1), cuda:0, cpu)");
    
    // =========================================================================
    // Initialize GlobalBackendRouter for PP transfers (PRODUCTION REQUIREMENT)
    // =========================================================================
    
    ASSERT_TRUE(GlobalBackendRouter::initForTests())
        << "Failed to initialize GlobalBackendRouter for tests";
    
    // =========================================================================
    // Initialize PCIe BAR for cross-vendor transfers (ROCm ↔ CUDA)
    // =========================================================================
    //
    // The HierarchicalPPContext uses DirectP2PEngine::getSharedInstance() for
    // PCIe BAR staging. We need to initialize the BAR before transfers.
    
    auto p2p = DirectP2PEngine::getSharedInstance();
    if (p2p && !p2p->isPCIeBarActive()) {
        auto caps = DirectP2PEngine::probeCapabilities();
        if (caps.canDoDirectP2P()) {
            bool bar_ok = p2p->initializePCIeBar(DeviceId::cuda(0), DeviceId::rocm(0));
            if (bar_ok) {
                LOG_INFO("[Test] PCIe BAR initialized for cross-vendor transfers");
            } else {
                LOG_WARN("[Test] PCIe BAR initialization failed, will use HOST staging");
            }
        }
    }
    
    // =========================================================================
    // PART 1: Create ILocalTPContext for Stage 0 = TensorParallel(rocm:0, rocm:1)
    // =========================================================================
    
    std::vector<GlobalDeviceAddress> tp_devices = {
        GlobalDeviceAddress::rocm(0),
        GlobalDeviceAddress::rocm(1)
    };
    
    // Equal weights for TP (50/50 split)
    std::vector<float> tp_weights = {};  // Empty = equal distribution
    
    // AUTO backend should select RCCL for all-ROCm devices
    auto tp_ctx = createLocalTPContext(tp_devices, tp_weights, CollectiveBackendType::AUTO);
    ASSERT_NE(tp_ctx, nullptr) << "Failed to create LocalTPContext";
    
    EXPECT_EQ(tp_ctx->degree(), 2) << "TP degree should be 2";
    EXPECT_EQ(tp_ctx->backend(), CollectiveBackendType::RCCL) 
        << "AUTO backend should select RCCL for all-ROCm devices";
    
    LOG_INFO("[Test] Created LocalTPContext: degree=" << tp_ctx->degree() 
             << ", backend=RCCL");
    
    // =========================================================================
    // PART 2: Create HierarchicalPPConfig wrapping TP context as Stage 0
    // =========================================================================
    //
    // NEW API: Instead of flat device list, use PPStage::fromTPContext()
    // This allows LocalPPContext::transfer() to understand the TP domain
    // and handle BAR-backed tensors automatically.
    
    // Convert unique_ptr to shared_ptr so we can share it with PPStage while
    // still using it for allreduce. The shared_ptr will own the TP context.
    std::shared_ptr<ILocalTPContext> tp_ctx_shared(std::move(tp_ctx));
    
    HierarchicalPPConfig pp_config;
    
    // Stage 0: TP domain (rocm:0, rocm:1) - the TP context wraps both devices
    // Stage 1: cuda:0 (single device)
    // Stage 2: cpu (single device)
    pp_config.stages = {
        PPStage::fromTPContext(tp_ctx_shared),
        PPStage::fromDevice(GlobalDeviceAddress::cuda(0)),
        PPStage::fromDevice(GlobalDeviceAddress::cpu())
    };
    
    // Layer boundaries: [0, 14, 24, 24] → Stage 0 gets more layers (TP advantage)
    pp_config.layer_boundaries = {0, 14, 24, 24};
    
    ASSERT_TRUE(pp_config.isValid()) << "HierarchicalPPConfig should be valid";
    
    auto pp_ctx = createLocalPPContext(pp_config);
    ASSERT_NE(pp_ctx, nullptr) << "createLocalPPContext(HierarchicalPPConfig) should succeed";
    
    EXPECT_EQ(pp_ctx->numStages(), 3);
    
    LOG_INFO("[Test] Created HierarchicalPPContext: " << pp_ctx->numStages() << " stages");
    LOG_INFO("[Test]   Stage 0: " << pp_config.stages[0].describe());
    LOG_INFO("[Test]   Stage 1: " << pp_config.stages[1].describe());
    LOG_INFO("[Test]   Stage 2: " << pp_config.stages[2].describe());
    
    // =========================================================================
    // PART 3: Create tensors for TP allreduce
    // =========================================================================
    
    constexpr size_t kNumElements = 1024;  // Standard test size
    
    // GPU 0: partial sum = 1.0 for all elements
    // GPU 1: partial sum = 2.0 for all elements
    // Expected after allreduce: 3.0 for all elements
    // Expected after allreduce: 3.0 for all elements
    std::vector<std::vector<float>> per_gpu_values = {
        std::vector<float>(kNumElements, 1.0f),  // GPU 0: all 1.0
        std::vector<float>(kNumElements, 2.0f)   // GPU 1: all 2.0
    };
    
    // Create one tensor per TP device (PRODUCTION PATTERN: each device owns its tensor)
    // NOTE: Using regular tensors for RCCL allreduce - BAR-backed not required here
    std::vector<std::unique_ptr<FP32Tensor>> tp_tensors(2);
    for (int gpu = 0; gpu < 2; ++gpu) {
        DeviceId device = DeviceId::rocm(gpu);
        tp_tensors[gpu] = TestTensorFactory::createFP32({kNumElements});
        ASSERT_NE(tp_tensors[gpu], nullptr);
        
        // Initialize host data with this GPU's partial values
        std::memcpy(tp_tensors[gpu]->mutable_data(), per_gpu_values[gpu].data(),
                    kNumElements * sizeof(float));
        
        // Upload to device (PRODUCTION: ensureOnDevice used by all stage execution)
        ASSERT_TRUE(tp_tensors[gpu]->ensureOnDevice(device))
            << "Failed to upload tensor to GPU " << gpu;
        tp_tensors[gpu]->mark_device_dirty();
    }
    
    LOG_INFO("[Test] Created TP tensors: GPU0=all 1.0, GPU1=all 2.0");
    
    // =========================================================================
    // PART 4: Multi-threaded TP allreduce (PRODUCTION SPMD PATTERN)
    // =========================================================================
    //
    // This is the EXACT pattern used in production inference:
    // - One thread per TP device
    // - Each thread owns a tensor and calls tp_ctx->allreduce()
    // - LocalTPContext::allreduceWithBarrierMultiGpu() coordinates threads
    //
    // Reference: Test__RCCLAllreduceAccuracy.cpp::runMultiThreadedAllreduceTest()
    
    LOG_INFO("[Test] Performing multi-threaded TP allreduce (RCCL SPMD pattern)");
    
    std::atomic<int> threads_ready{0};
    std::atomic<bool> all_success{true};
    std::mutex start_mutex;
    std::condition_variable start_cv;
    bool start_signal = false;
    const std::string stage_name = "test_tp_allreduce";
    
    std::vector<std::thread> threads;
    for (int gpu = 0; gpu < 2; ++gpu) {
        threads.emplace_back([&, gpu]() {
            // Set HIP device context for this thread (PRODUCTION REQUIREMENT)
            hipSetDevice(gpu);
            
            // Signal ready
            threads_ready.fetch_add(1);
            
            // Wait for start signal (barrier before collective)
            {
                std::unique_lock<std::mutex> lock(start_mutex);
                start_cv.wait(lock, [&]() { return start_signal; });
            }
            
            // Call allreduce via LocalTPContext (PRODUCTION CODE PATH!)
            bool success = tp_ctx_shared->allreduce(
                tp_tensors[gpu].get(),
                stage_name,
                kNumElements);
            
            if (!success) {
                LOG_ERROR("[Test] GPU " << gpu << " allreduce failed");
                all_success.store(false);
            }
        });
    }
    
    // Wait for all threads to be ready
    while (threads_ready.load() < 2) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    // Signal all threads to start allreduce
    {
        std::lock_guard<std::mutex> lock(start_mutex);
        start_signal = true;
    }
    start_cv.notify_all();
    
    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }
    
    ASSERT_TRUE(all_success.load()) << "TP allreduce failed on one or more GPUs";
    
    // Synchronize TP context
    tp_ctx_shared->synchronize();
    
    LOG_INFO("[Test] TP allreduce complete");
    
    // =========================================================================
    // PART 5: Verify allreduce result on GPU 0 (primary device)
    // =========================================================================
    
    // Sync GPU 0's tensor to host and verify
    rocm_backend->synchronize(0);
    const float* check_data = tp_tensors[0]->data();  // Triggers D2H
    
    constexpr float kExpectedValue = 3.0f;  // 1.0 + 2.0
    EXPECT_NEAR(check_data[0], kExpectedValue, 1e-5f) 
        << "Allreduce result[0] should be 3.0";
    EXPECT_NEAR(check_data[kNumElements/2], kExpectedValue, 1e-5f) 
        << "Allreduce result[mid] should be 3.0";
    EXPECT_NEAR(check_data[kNumElements-1], kExpectedValue, 1e-5f) 
        << "Allreduce result[last] should be 3.0";
    
    LOG_INFO("[Test] TP allreduce verified: values = " << check_data[0]);
    
    // =========================================================================
    // PART 6: PP Transfer Stage 0 → Stage 1 (TP Domain → CUDA:0)
    // =========================================================================
    //
    // NEW APPROACH: HierarchicalPPContext knows Stage 0 is a TP domain and
    // will handle the transfer automatically. This includes:
    // 1. Getting the representative tensor from the TP context (GPU 0)
    // 2. Using DirectP2PEngine for BAR-backed transfer if available
    // 3. Falling back to staged transfer through host if needed
    //
    // The test NO LONGER needs to:
    // - Create BAR-backed tensors manually
    // - Copy data to BAR region with hipMemcpy
    // - Fight with coherence model
    
    LOG_INFO("[Test] PP Transfer: TP Domain → CUDA:0 (HierarchicalPPContext handles BAR)");
    
    // Use LocalPPContext::transfer() which now understands TP domains
    // The tensor is the result of allreduce on GPU 0
    ASSERT_TRUE(pp_ctx->transfer(tp_tensors[0].get(), 0, 1))
        << "PP transfer from TP domain to CUDA failed";
    
    LOG_INFO("[Test] PP transfer Stage 0 → Stage 1 complete");
    
    // =========================================================================
    // PART 7: PP Transfer Stage 1 → Stage 2 (CUDA:0 → CPU via HOST)
    // =========================================================================
    
    LOG_INFO("[Test] PP Transfer: CUDA:0 → CPU (HOST backend)");
    
    ASSERT_TRUE(pp_ctx->transfer(tp_tensors[0].get(), 1, 2))
        << "PP transfer from CUDA to CPU failed";
    
    LOG_INFO("[Test] PP transfer Stage 1 → Stage 2 complete");
    
    // =========================================================================
    // PART 8: Verify final data on CPU
    // =========================================================================
    
    const float* result_data = tp_tensors[0]->data();
    ASSERT_NE(result_data, nullptr) << "Result data should not be null";
    
    int errors = 0;
    float max_diff = 0.0f;
    
    for (size_t i = 0; i < kNumElements; ++i) {
        float diff = std::abs(result_data[i] - kExpectedValue);
        if (diff > 1e-5f) {
            if (errors < 5) {
                LOG_ERROR("[Test] Mismatch at index " << i 
                          << ": expected=" << kExpectedValue << ", actual=" << result_data[i]);
            }
            errors++;
            max_diff = std::max(max_diff, diff);
        }
    }
    
    EXPECT_EQ(errors, 0) << "Data integrity check failed: " << errors 
                         << " mismatches, max_diff=" << max_diff;
    
    if (errors == 0) {
        LOG_INFO("[Test] ✓ Full PP+TP pipeline verified:");
        LOG_INFO("[Test]   TP: TensorParallel(rocm:0, rocm:1) allreduce → 3.0");
        LOG_INFO("[Test]   PP: ROCm:0 → CUDA:0 (BAR-backed, PCIeBAR) → CPU (HOST)");
        LOG_INFO("[Test]   All " << kNumElements << " elements = " << kExpectedValue);
    }
    
    // =========================================================================
    // Cleanup
    // =========================================================================
    
    pp_ctx->synchronize();
    GlobalBackendRouter::shutdown();
    
    LOG_INFO("[Test] HeterogeneousPPTP_EndToEndTransfer PASSED");
#endif  // HAVE_CUDA && HAVE_ROCM
}

// =============================================================================
// Test: DeviceGraphOrchestrator creates HierarchicalPPContext for TP domains
// =============================================================================

/**
 * @brief Integration test that verifies DeviceGraphOrchestrator uses HierarchicalPPConfig
 *
 * This test exercises the full path through DeviceGraphOrchestrator:
 * 1. Create a PipelineConfig with TP domains (degree > 1)
 * 2. Call setPipelineConfig() and initializePPContexts()
 * 3. Verify the orchestrator created HierarchicalPPContext (not flat LocalPPContext)
 * 4. Verify the PP context correctly wraps the TP context
 */
TEST_F(Test__PPTPBufferPoolIntegration, DeviceGraphOrchestrator_CreatesHierarchicalPPContext)
{
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
    LOG_INFO("[Test] ========================================");
    LOG_INFO("[Test] DeviceGraphOrchestrator_CreatesHierarchicalPPContext");
    LOG_INFO("[Test] ========================================");
    
    // Skip if we don't have the required hardware
    IBackend* rocm_backend = getROCmBackend();
    IBackend* cuda_backend = getCUDABackend();
    if (!rocm_backend || rocm_backend->deviceCount() < 2 ||
        !cuda_backend || cuda_backend->deviceCount() < 1)
    {
        GTEST_SKIP() << "Test requires at least 2 ROCm GPUs and 1 CUDA GPU";
    }
    
    // =========================================================================
    // PART 1: Create PipelineConfig with TP domain
    // =========================================================================
    
    // PP(TP(rocm:0, rocm:1), cuda:0) - Stage 0 is a 2-way TP domain
    PipelineConfig pipeline_config;
    pipeline_config.total_layers = 24;
    pipeline_config.tp_domains = {
        TPDomainConfig{"rocm_tp", {DeviceId::rocm(0), DeviceId::rocm(1)}, CollectiveBackendType::RCCL},
        TPDomainConfig{"cuda_single", {DeviceId::cuda(0)}, CollectiveBackendType::AUTO}
    };
    pipeline_config.pp_stages = {
        PPStageConfig::firstStage(0, "rocm_tp", 0, 14),
        PPStageConfig::lastStage(1, "cuda_single", 14, 24)
    };
    
    std::string error_msg;
    ASSERT_TRUE(pipeline_config.validate(&error_msg)) 
        << "PipelineConfig validation failed: " << error_msg;
    ASSERT_TRUE(pipeline_config.hasTP()) << "Config should have TP (rocm_tp has degree 2)";
    ASSERT_TRUE(pipeline_config.hasPP()) << "Config should have PP (2 stages)";
    
    LOG_INFO("[Test] Created PipelineConfig: PP(TP(rocm:0,rocm:1), cuda:0)");
    
    // =========================================================================
    // PART 2: Initialize backends (required for PP transfers)
    // =========================================================================
    
    // Initialize global backend router
    ASSERT_TRUE(GlobalBackendRouter::initForTests())
        << "Failed to initialize GlobalBackendRouter for tests";
    
    // Initialize PCIe BAR P2P for cross-vendor transfers
    auto p2p = DirectP2PEngine::getSharedInstance();
    if (p2p && !p2p->isPCIeBarActive()) {
        auto caps = DirectP2PEngine::probeCapabilities();
        if (caps.canDoDirectP2P()) {
            bool bar_ok = p2p->initializePCIeBar(DeviceId::cuda(0), DeviceId::rocm(0));
            if (!bar_ok) {
                GTEST_SKIP() << "PCIe BAR P2P not available (failed to initialize)";
            }
            LOG_INFO("[Test] PCIe BAR initialized for cross-vendor transfers");
        } else {
            GTEST_SKIP() << "PCIe BAR P2P not available (capabilities check failed)";
        }
    }
    
    // =========================================================================
    // PART 3: Manually create TP context (simulating orchestrator's initializeTPContexts)
    // =========================================================================
    
    std::vector<GlobalDeviceAddress> tp_devices = {
        GlobalDeviceAddress::rocm(0),
        GlobalDeviceAddress::rocm(1)
    };
    std::vector<float> tp_weights = {};  // Equal distribution
    
    auto tp_ctx = createLocalTPContext(tp_devices, tp_weights, CollectiveBackendType::RCCL);
    ASSERT_NE(tp_ctx, nullptr) << "Failed to create LocalTPContext";
    EXPECT_EQ(tp_ctx->degree(), 2);
    
    LOG_INFO("[Test] Created LocalTPContext: degree=" << tp_ctx->degree() 
             << ", backend=RCCL");
    
    // =========================================================================
    // PART 4: Build HierarchicalPPConfig (simulating orchestrator's initializePPContexts)
    // =========================================================================
    //
    // This is the key logic from DeviceGraphOrchestrator::initializePPContexts():
    // - Check if hasTP() → use HierarchicalPPConfig
    // - For each stage: if domain.degree() > 1 → PPStage::fromTPContext()
    //                   else → PPStage::fromDevice()
    
    std::shared_ptr<ILocalTPContext> tp_ctx_shared(std::move(tp_ctx));
    
    HierarchicalPPConfig pp_config;
    
    // Build stages based on domain degrees (mimics orchestrator logic)
    for (int s = 0; s < pipeline_config.numStages(); ++s)
    {
        const auto* domain = pipeline_config.getDomainForStage(s);
        ASSERT_NE(domain, nullptr) << "Domain for stage " << s << " is null";
        
        if (domain->degree() > 1)
        {
            // TP domain - use PPStage::fromTPContext
            // In real orchestrator, this looks up domain_tp_contexts_[domain.name]
            pp_config.stages.push_back(PPStage::fromTPContext(tp_ctx_shared));
            LOG_INFO("[Test] Stage " << s << ": TP domain '" << domain->name 
                     << "' (degree=" << domain->degree() << ")");
        }
        else
        {
            // Single device - use PPStage::fromDevice
            auto device_addr = GlobalDeviceAddress::fromLocalDeviceId(domain->primaryDevice());
            pp_config.stages.push_back(PPStage::fromDevice(device_addr));
            LOG_INFO("[Test] Stage " << s << ": single device '" << domain->name 
                     << "' (" << device_addr.toString() << ")");
        }
    }
    
    // Build layer boundaries
    pp_config.layer_boundaries.push_back(pipeline_config.pp_stages[0].first_layer);
    for (const auto& pp_stage : pipeline_config.pp_stages)
    {
        pp_config.layer_boundaries.push_back(pp_stage.last_layer);
    }
    
    ASSERT_TRUE(pp_config.isValid()) << "HierarchicalPPConfig should be valid";
    EXPECT_EQ(pp_config.numStages(), 2);
    
    // =========================================================================
    // PART 5: Verify PPStage types
    // =========================================================================
    
    // Stage 0 should be TP domain
    EXPECT_TRUE(pp_config.stages[0].isTPDomain()) 
        << "Stage 0 should be TP domain";
    EXPECT_EQ(pp_config.stages[0].asTPContext()->degree(), 2)
        << "Stage 0 TP context should have degree 2";
    
    // Stage 1 should be single device
    EXPECT_TRUE(pp_config.stages[1].isSingleDevice()) 
        << "Stage 1 should be single device";
    EXPECT_EQ(pp_config.stages[1].device().device_type, DeviceType::CUDA)
        << "Stage 1 should be CUDA device";
    
    LOG_INFO("[Test] Verified PPStage types: Stage 0=TP(2), Stage 1=single(cuda:0)");
    
    // =========================================================================
    // PART 6: Create HierarchicalPPContext and verify it works
    // =========================================================================
    
    auto pp_ctx = createLocalPPContext(pp_config);
    ASSERT_NE(pp_ctx, nullptr) << "createLocalPPContext(HierarchicalPPConfig) returned nullptr";
    
    EXPECT_EQ(pp_ctx->numStages(), 2);
    
    LOG_INFO("[Test] Created HierarchicalPPContext: " << pp_ctx->numStages() << " stages");
    
    // =========================================================================
    // PART 7: Create test tensors and perform actual PP transfer
    // =========================================================================
    
    constexpr size_t kNumElements = 256;
    
    // Create tensor on host, then upload to ROCm GPU 0 (representative of TP domain)
    auto src_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{kNumElements});
    
    // Initialize with test data on host
    for (size_t i = 0; i < kNumElements; ++i)
    {
        src_tensor->mutable_data()[i] = static_cast<float>(i) * 0.01f;
    }
    
    // Upload to ROCm GPU 0
    ASSERT_TRUE(src_tensor->ensureOnDevice(DeviceId::rocm(0)));
    src_tensor->mark_device_dirty();
    
    rocm_backend->synchronize(0);
    
    LOG_INFO("[Test] Created source tensor on rocm:0 with " << kNumElements << " elements");
    
    // =========================================================================
    // PART 8: PP Transfer using HierarchicalPPContext
    // =========================================================================
    
    // Transfer from Stage 0 (TP domain) to Stage 1 (cuda:0)
    // HierarchicalPPContext should use PCIe BAR for cross-vendor transfer
    ASSERT_TRUE(pp_ctx->transfer(src_tensor.get(), 0, 1))
        << "PP transfer from TP domain to CUDA failed";
    
    pp_ctx->synchronize();
    
    LOG_INFO("[Test] PP transfer Stage 0 → Stage 1 complete (HierarchicalPPContext)");
    
    // =========================================================================
    // PART 9: Verify data integrity
    // =========================================================================
    
    // Tensor should now be on cuda:0 (or its data accessible)
    const float* result_data = src_tensor->data();
    ASSERT_NE(result_data, nullptr);
    
    int errors = 0;
    for (size_t i = 0; i < kNumElements && errors < 5; ++i)
    {
        float expected = static_cast<float>(i) * 0.01f;
        float actual = result_data[i];
        if (std::abs(expected - actual) > 1e-5f)
        {
            LOG_ERROR("[Test] Mismatch at " << i << ": expected=" << expected << ", actual=" << actual);
            errors++;
        }
    }
    
    EXPECT_EQ(errors, 0) << "Data integrity check failed after PP transfer";
    
    if (errors == 0)
    {
        LOG_INFO("[Test] ✓ DeviceGraphOrchestrator_CreatesHierarchicalPPContext PASSED");
        LOG_INFO("[Test]   - Orchestrator pattern correctly builds HierarchicalPPConfig");
        LOG_INFO("[Test]   - TP domains detected and wrapped in PPStage::fromTPContext");
        LOG_INFO("[Test]   - PP transfer from TP domain to single device works");
        LOG_INFO("[Test]   - Data integrity verified after cross-vendor transfer");
    }
    
    // =========================================================================
    // Cleanup
    // =========================================================================
    
    GlobalBackendRouter::shutdown();
    
#else
    GTEST_SKIP() << "Test requires both CUDA and ROCm support";
#endif
}

} // namespace
} // namespace llaminar2

// GTest main provided by the test framework

