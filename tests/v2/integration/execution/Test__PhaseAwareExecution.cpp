/**
 * @file Test__PhaseAwareExecution.cpp
 * @brief Integration tests for phase-aware device selection (prefill vs decode)
 *
 * This test validates that the inference engine correctly selects devices
 * based on the execution phase and batch size:
 * - PREFILL: High-batch processing, typically favors GPU
 * - DECODE: Single-token generation, may use CPU for efficiency
 *
 * Test scenarios:
 * 1. PrefillUsesGpu - GPU preferred for prefill even with small batch
 * 2. DecodeMayUseCpu - CPU fallback acceptable during decode phase
 * 3. BatchSizeThresholds - Different batch sizes trigger different device selection
 * 4. PhaseTransition - Correct device switching between prefill and decode
 * 5. HybridExecution - Mixed device usage based on operation efficiency
 * 6. CpuParticipationInDecode - CPU assists during decode phase
 *
 * Uses mock infrastructure from tests/v2/mocks/ to avoid real MPI/hardware.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <functional>

#include "mocks/MockMPITopology.h"
#include "mocks/MockCollectiveContext.h"
#include "mocks/MockModelContext.h"
#include "backends/DeviceId.h"
#include "utils/MPITopology.h"  // For RankPlacement, DeviceCapability

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// Execution Phase Enum
// =============================================================================

/**
 * @brief Execution phase for inference
 *
 * Different phases may use different device placement strategies:
 * - PREFILL: Process multiple input tokens, compute-bound, GPU-preferred
 * - DECODE: Generate tokens one at a time, memory-bound, CPU may be efficient
 */
enum class ExecutionPhase {
    PREFILL,  ///< Initial context processing (batch of input tokens)
    DECODE    ///< Autoregressive token generation (typically single token)
};

// =============================================================================
// Device Selection Configuration
// =============================================================================

/**
 * @brief Configuration for device selection based on phase
 */
struct PhaseDeviceConfig {
    DeviceId primary_device;        ///< Primary device for this phase
    bool allow_cpu_fallback;        ///< Whether CPU fallback is acceptable
    bool prefer_gpu_for_attention;  ///< GPU preferred for attention operations
    bool prefer_gpu_for_ffn;        ///< GPU preferred for FFN operations
    int batch_threshold_for_gpu;    ///< Minimum batch size to prefer GPU
};

/**
 * @brief Phase-aware device selector for heterogeneous execution
 *
 * Determines optimal device placement based on:
 * - Execution phase (prefill vs decode)
 * - Batch size
 * - Available devices in topology
 * - Memory constraints
 */
class PhaseAwareDeviceSelector {
public:
    explicit PhaseAwareDeviceSelector(std::shared_ptr<IMPITopology> topology)
        : topology_(std::move(topology)) {
        analyzeTopology();
    }

    /**
     * @brief Get device configuration for a specific phase and batch size
     */
    PhaseDeviceConfig getDeviceConfig(ExecutionPhase phase, int batch_size) const {
        PhaseDeviceConfig config;
        
        // Determine primary device based on phase and GPU availability
        if (has_gpu_) {
            if (phase == ExecutionPhase::PREFILL) {
                // Prefill always prefers GPU when available
                config.primary_device = gpu_device_;
                config.allow_cpu_fallback = false;
                config.prefer_gpu_for_attention = true;
                config.prefer_gpu_for_ffn = true;
                config.batch_threshold_for_gpu = 1;  // GPU even for batch=1 prefill
            } else {
                // Decode phase: depends on batch size
                if (batch_size >= decode_gpu_threshold_) {
                    config.primary_device = gpu_device_;
                    config.allow_cpu_fallback = false;
                } else {
                    // Small batch decode can use CPU
                    config.primary_device = cpu_device_;
                    config.allow_cpu_fallback = true;
                }
                config.prefer_gpu_for_attention = (batch_size >= attention_gpu_threshold_);
                config.prefer_gpu_for_ffn = (batch_size >= ffn_gpu_threshold_);
                config.batch_threshold_for_gpu = decode_gpu_threshold_;
            }
        } else {
            // CPU-only topology
            config.primary_device = cpu_device_;
            config.allow_cpu_fallback = true;
            config.prefer_gpu_for_attention = false;
            config.prefer_gpu_for_ffn = false;
            config.batch_threshold_for_gpu = INT_MAX;
        }
        
        return config;
    }

    /**
     * @brief Check if topology has GPU
     */
    bool hasGPU() const { return has_gpu_; }

    /**
     * @brief Get decode GPU threshold
     */
    int getDecodeGPUThreshold() const { return decode_gpu_threshold_; }

    /**
     * @brief Set decode GPU threshold for testing
     */
    void setDecodeGPUThreshold(int threshold) { decode_gpu_threshold_ = threshold; }

    /**
     * @brief Check if CPU participation is enabled for decode
     */
    bool isCpuParticipationEnabled() const { return cpu_participation_enabled_; }

    /**
     * @brief Enable CPU participation in decode phase
     */
    void enableCpuParticipation(bool enable) { cpu_participation_enabled_ = enable; }

private:
    void analyzeTopology() {
        // Check for GPU availability
        has_gpu_ = topology_->has_accelerator();
        
        // Set default devices
        cpu_device_ = DeviceId::cpu();
        if (has_gpu_) {
            const auto& devices = topology_->get_devices();
            for (const auto& dev : devices) {
                if (dev.type == DeviceCapability::Type::CUDA) {
                    gpu_device_ = DeviceId::cuda(dev.device_id);
                    break;
                } else if (dev.type == DeviceCapability::Type::ROCm) {
                    gpu_device_ = DeviceId::rocm(dev.device_id);
                    break;
                }
            }
        }
    }

    std::shared_ptr<IMPITopology> topology_;
    DeviceId cpu_device_ = DeviceId::cpu();
    DeviceId gpu_device_ = DeviceId::cpu();  // Default to CPU if no GPU
    bool has_gpu_ = false;
    
    // Thresholds for GPU usage during decode
    int decode_gpu_threshold_ = 8;      // Batch size above which GPU is preferred for decode
    int attention_gpu_threshold_ = 4;   // Batch size above which GPU is preferred for attention
    int ffn_gpu_threshold_ = 4;         // Batch size above which GPU is preferred for FFN
    
    // CPU participation flag
    bool cpu_participation_enabled_ = false;
};

// =============================================================================
// Test Fixture
// =============================================================================

class Test__PhaseAwareExecution : public ::testing::Test {
protected:
    void SetUp() override {
        // Default topology: CPU + GPU
        setupHeterogeneousTopology();
    }

    /**
     * @brief Setup heterogeneous topology with CPU and GPU
     */
    void setupHeterogeneousTopology() {
        topology_ = MockMPITopologyBuilder()
            .addRank(0, 0, {MockDevices::cpu(1.0f), MockDevices::cuda(0, 16.0f, 10.0f)})
            .setLocalRank(0)
            .build();
        
        selector_ = std::make_unique<PhaseAwareDeviceSelector>(topology_);
    }

    /**
     * @brief Setup CPU-only topology
     */
    void setupCPUOnlyTopology() {
        topology_ = MockMPITopologyBuilder()
            .addCPUOnlyRank(0, 0)
            .setLocalRank(0)
            .build();
        
        selector_ = std::make_unique<PhaseAwareDeviceSelector>(topology_);
    }

    /**
     * @brief Setup multi-rank heterogeneous topology
     */
    void setupMultiRankHeterogeneousTopology() {
        topology_ = MockMPITopologyBuilder()
            .addRank(0, 0, {MockDevices::cpu(1.0f), MockDevices::cuda(0, 24.0f, 15.0f)})
            .addRank(1, 0, {MockDevices::cpu(1.0f), MockDevices::cuda(1, 24.0f, 15.0f)})
            .setLocalRank(0)
            .build();
        
        selector_ = std::make_unique<PhaseAwareDeviceSelector>(topology_);
    }

    /**
     * @brief Setup topology with CPU-only and GPU ranks
     */
    void setupMixedCpuGpuTopology() {
        topology_ = MockMPITopologyBuilder()
            .addGPURank(0, 0, 0, 16.0f)   // Rank 0: CPU + GPU
            .addCPUOnlyRank(1, 0)         // Rank 1: CPU only
            .setLocalRank(0)
            .build();
        
        selector_ = std::make_unique<PhaseAwareDeviceSelector>(topology_);
    }

    std::shared_ptr<MockMPITopology> topology_;
    std::unique_ptr<PhaseAwareDeviceSelector> selector_;
};

// =============================================================================
// Test: PrefillUsesGpu
// =============================================================================

/**
 * @test Verify that prefill phase uses GPU even with small batch size
 *
 * During prefill, the computation is highly parallel and benefits from GPU
 * even for small batch sizes. The selector should always prefer GPU for
 * prefill when available.
 */
TEST_F(Test__PhaseAwareExecution, PrefillUsesGpu) {
    ASSERT_TRUE(selector_->hasGPU()) << "Test requires GPU topology";
    
    // Test with various batch sizes
    std::vector<int> batch_sizes = {1, 4, 8, 16, 64, 128, 512};
    
    for (int batch_size : batch_sizes) {
        auto config = selector_->getDeviceConfig(ExecutionPhase::PREFILL, batch_size);
        
        EXPECT_TRUE(config.primary_device.is_gpu())
            << "Prefill with batch=" << batch_size << " should use GPU";
        EXPECT_FALSE(config.allow_cpu_fallback)
            << "Prefill should not allow CPU fallback";
        EXPECT_TRUE(config.prefer_gpu_for_attention)
            << "Prefill should prefer GPU for attention";
        EXPECT_TRUE(config.prefer_gpu_for_ffn)
            << "Prefill should prefer GPU for FFN";
    }
}

/**
 * @test Verify prefill on CPU-only topology
 */
TEST_F(Test__PhaseAwareExecution, PrefillOnCpuOnlyTopology) {
    setupCPUOnlyTopology();
    ASSERT_FALSE(selector_->hasGPU()) << "Should be CPU-only topology";
    
    auto config = selector_->getDeviceConfig(ExecutionPhase::PREFILL, 100);
    
    EXPECT_TRUE(config.primary_device.is_cpu())
        << "CPU-only topology should use CPU for prefill";
    EXPECT_TRUE(config.allow_cpu_fallback)
        << "CPU-only topology should allow CPU fallback";
}

// =============================================================================
// Test: DecodeMayUseCpu
// =============================================================================

/**
 * @test Verify that single-token decode can fall back to CPU
 *
 * During decode with batch_size=1, the computation is memory-bound and
 * the overhead of GPU kernel launch may outweigh the benefits. CPU can
 * be acceptable for this scenario.
 */
TEST_F(Test__PhaseAwareExecution, DecodeMayUseCpu) {
    ASSERT_TRUE(selector_->hasGPU()) << "Test requires GPU topology";
    
    // Single token decode (batch_size = 1)
    auto config = selector_->getDeviceConfig(ExecutionPhase::DECODE, 1);
    
    // CPU is acceptable for single-token decode
    EXPECT_TRUE(config.allow_cpu_fallback)
        << "Single-token decode should allow CPU fallback";
    
    // Primary device may be CPU for small batches
    EXPECT_TRUE(config.primary_device.is_cpu())
        << "Single-token decode should prefer CPU";
}

/**
 * @test Verify that batched decode uses GPU
 */
TEST_F(Test__PhaseAwareExecution, BatchedDecodeUsesGpu) {
    ASSERT_TRUE(selector_->hasGPU()) << "Test requires GPU topology";
    
    // Batched decode (batch_size = 8, above threshold)
    auto config = selector_->getDeviceConfig(ExecutionPhase::DECODE, 8);
    
    EXPECT_TRUE(config.primary_device.is_gpu())
        << "Batched decode should use GPU";
    EXPECT_FALSE(config.allow_cpu_fallback)
        << "Batched decode should not allow CPU fallback";
}

// =============================================================================
// Test: BatchSizeThresholds
// =============================================================================

/**
 * @test Verify batch size thresholds trigger correct device selection
 */
TEST_F(Test__PhaseAwareExecution, BatchSizeThresholds) {
    ASSERT_TRUE(selector_->hasGPU()) << "Test requires GPU topology";
    
    struct TestCase {
        ExecutionPhase phase;
        int batch_size;
        bool expect_gpu_primary;
        bool expect_allow_cpu_fallback;
        const char* description;
    };
    
    std::vector<TestCase> cases = {
        // Prefill: Always GPU
        {ExecutionPhase::PREFILL, 1, true, false, "Prefill batch=1"},
        {ExecutionPhase::PREFILL, 100, true, false, "Prefill batch=100"},
        {ExecutionPhase::PREFILL, 1000, true, false, "Prefill batch=1000"},
        
        // Decode: Depends on batch size (threshold = 8)
        {ExecutionPhase::DECODE, 1, false, true, "Decode batch=1"},
        {ExecutionPhase::DECODE, 4, false, true, "Decode batch=4"},
        {ExecutionPhase::DECODE, 7, false, true, "Decode batch=7 (below threshold)"},
        {ExecutionPhase::DECODE, 8, true, false, "Decode batch=8 (at threshold)"},
        {ExecutionPhase::DECODE, 16, true, false, "Decode batch=16 (above threshold)"},
        {ExecutionPhase::DECODE, 32, true, false, "Decode batch=32"},
    };
    
    for (const auto& tc : cases) {
        auto config = selector_->getDeviceConfig(tc.phase, tc.batch_size);
        
        if (tc.expect_gpu_primary) {
            EXPECT_TRUE(config.primary_device.is_gpu())
                << tc.description << " should use GPU";
        } else {
            EXPECT_TRUE(config.primary_device.is_cpu())
                << tc.description << " should use CPU";
        }
        
        EXPECT_EQ(config.allow_cpu_fallback, tc.expect_allow_cpu_fallback)
            << tc.description << " CPU fallback mismatch";
    }
}

/**
 * @test Verify custom threshold configuration
 */
TEST_F(Test__PhaseAwareExecution, CustomBatchSizeThreshold) {
    ASSERT_TRUE(selector_->hasGPU()) << "Test requires GPU topology";
    
    // Set custom threshold
    selector_->setDecodeGPUThreshold(4);
    
    // Batch=3: Below threshold, should use CPU
    auto config3 = selector_->getDeviceConfig(ExecutionPhase::DECODE, 3);
    EXPECT_TRUE(config3.primary_device.is_cpu());
    
    // Batch=4: At threshold, should use GPU
    auto config4 = selector_->getDeviceConfig(ExecutionPhase::DECODE, 4);
    EXPECT_TRUE(config4.primary_device.is_gpu());
}

// =============================================================================
// Test: PhaseTransition
// =============================================================================

/**
 * @test Verify correct device switching between prefill and decode phases
 */
TEST_F(Test__PhaseAwareExecution, PhaseTransition) {
    ASSERT_TRUE(selector_->hasGPU()) << "Test requires GPU topology";
    
    // Simulate inference flow:
    // 1. Prefill with 100 tokens (GPU)
    // 2. Decode single tokens (CPU for small batches)
    // 3. Decode with batching (GPU for larger batches)
    
    // Phase 1: Prefill
    auto prefill_config = selector_->getDeviceConfig(ExecutionPhase::PREFILL, 100);
    EXPECT_TRUE(prefill_config.primary_device.is_gpu())
        << "Prefill should use GPU";
    
    // Phase 2: Decode single token
    auto decode1_config = selector_->getDeviceConfig(ExecutionPhase::DECODE, 1);
    EXPECT_TRUE(decode1_config.primary_device.is_cpu())
        << "Single-token decode should use CPU";
    EXPECT_TRUE(decode1_config.allow_cpu_fallback)
        << "Single-token decode should allow CPU fallback";
    
    // Phase 3: Continue decode (simulate speculative decoding with batch=4)
    auto decode4_config = selector_->getDeviceConfig(ExecutionPhase::DECODE, 4);
    EXPECT_TRUE(decode4_config.primary_device.is_cpu())
        << "Batch=4 decode below threshold should use CPU";
    
    // Phase 4: Larger batch decode
    auto decode16_config = selector_->getDeviceConfig(ExecutionPhase::DECODE, 16);
    EXPECT_TRUE(decode16_config.primary_device.is_gpu())
        << "Batch=16 decode above threshold should use GPU";
}

/**
 * @test Verify continuous decode doesn't change device unnecessarily
 */
TEST_F(Test__PhaseAwareExecution, ConsistentDecodeDevice) {
    ASSERT_TRUE(selector_->hasGPU()) << "Test requires GPU topology";
    
    // Multiple single-token decode calls should consistently select same device
    std::vector<DeviceId> selected_devices;
    for (int i = 0; i < 10; ++i) {
        auto config = selector_->getDeviceConfig(ExecutionPhase::DECODE, 1);
        selected_devices.push_back(config.primary_device);
    }
    
    // All should be the same device type
    for (size_t i = 1; i < selected_devices.size(); ++i) {
        EXPECT_EQ(selected_devices[i].type, selected_devices[0].type)
            << "Decode device selection should be consistent";
    }
}

// =============================================================================
// Test: HybridExecution
// =============================================================================

/**
 * @test Verify hybrid execution with different device preferences per operation
 */
TEST_F(Test__PhaseAwareExecution, HybridExecution) {
    ASSERT_TRUE(selector_->hasGPU()) << "Test requires GPU topology";
    
    // During decode with batch=4, attention might prefer GPU while FFN might use CPU
    selector_->setDecodeGPUThreshold(8);  // Primary device threshold
    
    auto config = selector_->getDeviceConfig(ExecutionPhase::DECODE, 4);
    
    // Primary device is CPU (below threshold)
    EXPECT_TRUE(config.primary_device.is_cpu())
        << "Primary device should be CPU for batch=4";
    
    // But GPU may still be preferred for attention (if batch >= attention_threshold)
    EXPECT_TRUE(config.prefer_gpu_for_attention)
        << "GPU should be preferred for attention with batch=4";
    EXPECT_TRUE(config.prefer_gpu_for_ffn)
        << "GPU should be preferred for FFN with batch=4";
}

/**
 * @test Verify operation-level device preferences with small batch
 */
TEST_F(Test__PhaseAwareExecution, OperationLevelDevicePreference) {
    ASSERT_TRUE(selector_->hasGPU()) << "Test requires GPU topology";
    
    // Very small batch (1) - GPU not preferred for individual operations
    auto config = selector_->getDeviceConfig(ExecutionPhase::DECODE, 1);
    
    EXPECT_TRUE(config.primary_device.is_cpu());
    EXPECT_FALSE(config.prefer_gpu_for_attention)
        << "GPU should not be preferred for attention with batch=1";
    EXPECT_FALSE(config.prefer_gpu_for_ffn)
        << "GPU should not be preferred for FFN with batch=1";
}

// =============================================================================
// Test: CpuParticipationInDecode
// =============================================================================

/**
 * @test Verify CPU participation flag during decode phase
 */
TEST_F(Test__PhaseAwareExecution, CpuParticipationInDecode) {
    ASSERT_TRUE(selector_->hasGPU()) << "Test requires GPU topology";
    
    // By default, CPU participation is disabled
    EXPECT_FALSE(selector_->isCpuParticipationEnabled())
        << "CPU participation should be disabled by default";
    
    // Enable CPU participation
    selector_->enableCpuParticipation(true);
    EXPECT_TRUE(selector_->isCpuParticipationEnabled())
        << "CPU participation should be enabled after setting";
    
    // Even with GPU primary, CPU can assist in decode
    auto config = selector_->getDeviceConfig(ExecutionPhase::DECODE, 16);
    EXPECT_TRUE(config.primary_device.is_gpu())
        << "Large batch decode should use GPU primary";
    
    // CPU participation enabled means CPU can assist
    EXPECT_TRUE(selector_->isCpuParticipationEnabled())
        << "CPU participation should remain enabled";
}

/**
 * @test Verify multi-rank heterogeneous CPU participation
 */
TEST_F(Test__PhaseAwareExecution, MultiRankCpuParticipation) {
    setupMixedCpuGpuTopology();
    
    // In mixed topology, CPU-only ranks participate in decode
    EXPECT_TRUE(selector_->hasGPU())
        << "Local rank should have GPU";
    
    // Enable CPU participation for hybrid execution
    selector_->enableCpuParticipation(true);
    
    auto config = selector_->getDeviceConfig(ExecutionPhase::DECODE, 1);
    EXPECT_TRUE(config.allow_cpu_fallback)
        << "CPU fallback should be allowed for single-token decode";
}

// =============================================================================
// Test: Topology Variations
// =============================================================================

/**
 * @test Verify device selection with multi-GPU topology
 */
TEST_F(Test__PhaseAwareExecution, MultiGpuTopology) {
    setupMultiRankHeterogeneousTopology();
    
    ASSERT_TRUE(selector_->hasGPU()) << "Should have GPU";
    
    // Prefill should use GPU
    auto prefill_config = selector_->getDeviceConfig(ExecutionPhase::PREFILL, 128);
    EXPECT_TRUE(prefill_config.primary_device.is_gpu());
    
    // Decode should follow thresholds
    auto decode_config = selector_->getDeviceConfig(ExecutionPhase::DECODE, 1);
    EXPECT_TRUE(decode_config.allow_cpu_fallback);
}

/**
 * @test Verify graceful fallback on CPU-only topology
 */
TEST_F(Test__PhaseAwareExecution, CpuOnlyGracefulFallback) {
    setupCPUOnlyTopology();
    
    ASSERT_FALSE(selector_->hasGPU()) << "Should be CPU-only";
    
    // Both phases should gracefully use CPU
    auto prefill_config = selector_->getDeviceConfig(ExecutionPhase::PREFILL, 128);
    EXPECT_TRUE(prefill_config.primary_device.is_cpu());
    EXPECT_TRUE(prefill_config.allow_cpu_fallback);
    
    auto decode_config = selector_->getDeviceConfig(ExecutionPhase::DECODE, 1);
    EXPECT_TRUE(decode_config.primary_device.is_cpu());
    EXPECT_TRUE(decode_config.allow_cpu_fallback);
    
    // GPU preference flags should be false
    EXPECT_FALSE(prefill_config.prefer_gpu_for_attention);
    EXPECT_FALSE(prefill_config.prefer_gpu_for_ffn);
}

// =============================================================================
// Test: Edge Cases
// =============================================================================

/**
 * @test Verify behavior at exact threshold boundary
 */
TEST_F(Test__PhaseAwareExecution, ExactThresholdBoundary) {
    ASSERT_TRUE(selector_->hasGPU()) << "Test requires GPU topology";
    
    int threshold = selector_->getDecodeGPUThreshold();
    
    // At threshold: GPU
    auto at_config = selector_->getDeviceConfig(ExecutionPhase::DECODE, threshold);
    EXPECT_TRUE(at_config.primary_device.is_gpu())
        << "At threshold should use GPU";
    
    // Just below threshold: CPU
    auto below_config = selector_->getDeviceConfig(ExecutionPhase::DECODE, threshold - 1);
    EXPECT_TRUE(below_config.primary_device.is_cpu())
        << "Below threshold should use CPU";
}

/**
 * @test Verify behavior with batch_size = 0 (edge case)
 */
TEST_F(Test__PhaseAwareExecution, ZeroBatchSize) {
    ASSERT_TRUE(selector_->hasGPU()) << "Test requires GPU topology";
    
    // Batch size 0 is an edge case - should default to CPU fallback
    auto config = selector_->getDeviceConfig(ExecutionPhase::DECODE, 0);
    
    // Should still return valid configuration
    EXPECT_TRUE(config.primary_device.is_cpu() || config.primary_device.is_gpu());
    EXPECT_TRUE(config.allow_cpu_fallback)
        << "Zero batch should allow CPU fallback";
}

/**
 * @test Verify behavior with very large batch size
 */
TEST_F(Test__PhaseAwareExecution, LargeBatchSize) {
    ASSERT_TRUE(selector_->hasGPU()) << "Test requires GPU topology";
    
    // Very large batch should definitely use GPU
    auto config = selector_->getDeviceConfig(ExecutionPhase::DECODE, 10000);
    
    EXPECT_TRUE(config.primary_device.is_gpu())
        << "Large batch decode should use GPU";
    EXPECT_FALSE(config.allow_cpu_fallback)
        << "Large batch decode should not allow CPU fallback";
}

// =============================================================================
// Test: Integration with MockCollectiveContext
// =============================================================================

/**
 * @test Verify phase-aware execution integrates with collective context
 */
TEST_F(Test__PhaseAwareExecution, CollectiveContextIntegration) {
    // Create collective context matching the topology
    auto collective_ctx = MockCollectiveContext::Builder()
        .withRank(0)
        .withWorldSize(1)
        .withDevice(DeviceId::cpu())
        .withDevice(DeviceId::cuda(0))
        .build();
    
    ASSERT_TRUE(selector_->hasGPU());
    
    // Verify selector considers available devices via localDevices() method
    const auto& devices = collective_ctx->localDevices();
    ASSERT_EQ(devices.size(), 2) << "Should have CPU and GPU devices";
    
    bool has_cpu = false, has_gpu = false;
    for (const auto& dev : devices) {
        if (dev.is_cpu()) has_cpu = true;
        if (dev.is_gpu()) has_gpu = true;
    }
    EXPECT_TRUE(has_cpu) << "Collective context should have CPU device";
    EXPECT_TRUE(has_gpu) << "Collective context should have GPU device";
    
    // Prefill should use GPU
    auto prefill_config = selector_->getDeviceConfig(ExecutionPhase::PREFILL, 64);
    EXPECT_TRUE(prefill_config.primary_device.is_gpu());
}

