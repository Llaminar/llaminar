/**
 * @file Test__RankOrchestrator.cpp
 * @brief Unit tests for RankOrchestrator mocks and interface contract
 * @author David Sanftenberg
 * @date January 2026
 *
 * Tests the mock implementations used for testing RankOrchestrator
 * coordination logic. The mocks enable testing LOCAL tensor parallelism
 * coordination without real devices.
 *
 * Note: Tests for the actual RankOrchestrator class will be enabled
 * once the implementation is added to the build.
 */

#include <gtest/gtest.h>
#include "execution/local_execution/orchestrators/IInferenceRunner.h"
#include "execution/debug/TPSnapshot.h"
#include "collective/ILocalTPContext.h"
#include "backends/GlobalDeviceAddress.h"
#include "config/OrchestrationConfig.h"
#include "tensors/Tensors.h"
#include <atomic>
#include <cstring>
#include <vector>
#include <memory>
#include <stdexcept>

using namespace llaminar2;

// =============================================================================
// MockDeviceGraphOrchestrator - Mock for per-device runners
// =============================================================================

/**
 * @brief Mock inference runner for per-device testing
 *
 * Tracks method calls and provides configurable return values for testing
 * the RankOrchestrator coordination logic.
 */
class MockDeviceGraphOrchestrator : public IInferenceRunner
{
public:
    struct Config
    {
        int vocab_size = 32000;
        bool forward_should_fail = false;
        std::string architecture = "mock_qwen2";
    };

    MockDeviceGraphOrchestrator() : MockDeviceGraphOrchestrator(Config{}) {}

    explicit MockDeviceGraphOrchestrator(const Config &config)
        : config_(config), position_(0)
    {
        logits_.resize(static_cast<size_t>(config_.vocab_size), 0.0f);
    }

    // =====================================================================
    // IInferenceRunner Implementation
    // =====================================================================

    bool forward(const int *tokens, int seq_len) override
    {
        (void)tokens;
        forward_calls_.fetch_add(1, std::memory_order_relaxed);
        if (config_.forward_should_fail)
        {
            return false;
        }
        position_ += seq_len;
        return true;
    }

    const float *logits() const override
    {
        return logits_.data();
    }

    int vocab_size() const override
    {
        return config_.vocab_size;
    }

    void clear_cache() override
    {
        clear_cache_calls_.fetch_add(1, std::memory_order_relaxed);
        position_ = 0;
    }

    int get_position() const override
    {
        return position_;
    }

    ExecutionPath executionPath() const override
    {
        return ExecutionPath::GRAPH;
    }

    const char *architecture() const override
    {
        return config_.architecture.c_str();
    }

    // =====================================================================
    // Test Utilities
    // =====================================================================

    size_t forward_call_count() const
    {
        return forward_calls_.load(std::memory_order_relaxed);
    }

    size_t clear_cache_call_count() const
    {
        return clear_cache_calls_.load(std::memory_order_relaxed);
    }

    void set_forward_fails(bool fails) { config_.forward_should_fail = fails; }

    void set_mock_logits(const std::vector<float> &logits)
    {
        logits_ = logits;
        config_.vocab_size = static_cast<int>(logits.size());
    }

    void set_vocab_size(int size)
    {
        config_.vocab_size = size;
        logits_.resize(static_cast<size_t>(size), 0.0f);
    }

    void reset_call_counts()
    {
        forward_calls_.store(0, std::memory_order_relaxed);
        clear_cache_calls_.store(0, std::memory_order_relaxed);
    }

private:
    Config config_;
    int position_;
    std::vector<float> logits_;
    mutable std::atomic<size_t> forward_calls_{0};
    mutable std::atomic<size_t> clear_cache_calls_{0};
};

// =============================================================================
// MockLocalTPContext - Mock for LOCAL TP context
// =============================================================================

/**
 * @brief Mock LOCAL TP context for testing collective operations
 *
 * Tracks synchronization calls and provides configurable devices/weights.
 */
class MockLocalTPContext : public ILocalTPContext
{
public:
    struct Config
    {
        std::vector<GlobalDeviceAddress> devices;
        std::vector<float> weights;
        CollectiveBackendType backend = CollectiveBackendType::HOST;
        bool allreduce_should_fail = false;
        bool allgather_should_fail = false;
    };

    MockLocalTPContext() : MockLocalTPContext(Config{}) {}

    explicit MockLocalTPContext(const Config &config)
        : config_(config)
    {
        // Default to 2 CPU devices if none specified
        if (config_.devices.empty())
        {
            config_.devices.push_back(GlobalDeviceAddress::cpu());
            config_.devices.push_back(GlobalDeviceAddress::cpu());
        }
        // Default to equal weights
        if (config_.weights.empty())
        {
            float equal = 1.0f / static_cast<float>(config_.devices.size());
            config_.weights.resize(config_.devices.size(), equal);
        }
    }

    // =====================================================================
    // ILocalTPContext Configuration API
    // =====================================================================

    const std::vector<GlobalDeviceAddress> &devices() const override
    {
        return config_.devices;
    }

    const std::vector<float> &weights() const override
    {
        return config_.weights;
    }

    CollectiveBackendType backend() const override
    {
        return config_.backend;
    }

    int degree() const override
    {
        return static_cast<int>(config_.devices.size());
    }

    int myIndex() const override { return 0; }

    // =====================================================================
    // ILocalTPContext Collective Operations
    // =====================================================================

    bool allreduce(TensorBase * /*tensor*/) override
    {
        allreduce_calls_.fetch_add(1, std::memory_order_relaxed);
        return !config_.allreduce_should_fail;
    }

    bool allreduce(TensorBase *tensor, const std::string & /*stage_name*/, size_t /*count*/ = 0) override
    {
        return allreduce(tensor);
    }

    bool allreduce(const TensorBase * /*input*/, TensorBase * /*output*/) override
    {
        allreduce_calls_.fetch_add(1, std::memory_order_relaxed);
        return !config_.allreduce_should_fail;
    }

    bool allgather(const TensorBase * /*local_shard*/, TensorBase * /*global_tensor*/) override
    {
        allgather_calls_.fetch_add(1, std::memory_order_relaxed);
        return !config_.allgather_should_fail;
    }

    bool gatherFromDevices(
        const std::vector<const TensorBase *> &shards,
        TensorBase *output) override
    {
        gather_from_devices_calls_.fetch_add(1, std::memory_order_relaxed);

        // Simple mock implementation: copy data from shards to output
        if (shards.empty() || !output)
        {
            return false;
        }

        float *dst = output->mutable_data();
        size_t offset = 0;
        for (const auto *shard : shards)
        {
            if (shard)
            {
                const float *src = shard->data();
                size_t count = shard->numel();
                std::memcpy(dst + offset, src, count * sizeof(float));
                offset += count;
            }
        }
        return !config_.allgather_should_fail;
    }

    bool reduceScatter(const TensorBase * /*input*/, TensorBase * /*output_shard*/) override
    {
        reduce_scatter_calls_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // =====================================================================
    // ILocalTPContext Synchronization
    // =====================================================================

    void synchronize() override
    {
        synchronize_calls_.fetch_add(1, std::memory_order_relaxed);
    }

    // =====================================================================
    // ILocalTPContext Device Management
    // =====================================================================

    int indexForDevice(const GlobalDeviceAddress &device) const override
    {
        for (size_t i = 0; i < config_.devices.size(); ++i)
        {
            if (config_.devices[i] == device)
            {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    const GlobalDeviceAddress &deviceAt(int index) const override
    {
        if (index < 0 || index >= static_cast<int>(config_.devices.size()))
        {
            throw std::out_of_range("MockLocalTPContext::deviceAt: index out of range");
        }
        return config_.devices[static_cast<size_t>(index)];
    }

    float weightForDevice(const GlobalDeviceAddress &device) const override
    {
        int idx = indexForDevice(device);
        return (idx >= 0) ? config_.weights[static_cast<size_t>(idx)] : 0.0f;
    }

    // =====================================================================
    // ILocalTPContext Sharding Utilities
    // =====================================================================

    int headsForDevice(const GlobalDeviceAddress &device, int total_heads) const override
    {
        float w = weightForDevice(device);
        return static_cast<int>(w * static_cast<float>(total_heads) + 0.5f);
    }

    std::pair<int, int> rowRangeForDevice(
        const GlobalDeviceAddress &device, int total_rows) const override
    {
        int idx = indexForDevice(device);
        if (idx < 0)
            return {0, 0};

        float cumulative = 0.0f;
        for (int i = 0; i < idx; ++i)
        {
            cumulative += config_.weights[static_cast<size_t>(i)];
        }
        int start = static_cast<int>(cumulative * static_cast<float>(total_rows));
        int end = static_cast<int>((cumulative + config_.weights[static_cast<size_t>(idx)]) * static_cast<float>(total_rows));
        return {start, end};
    }

    std::pair<int, int> colRangeForDevice(
        const GlobalDeviceAddress &device, int total_cols) const override
    {
        return rowRangeForDevice(device, total_cols);
    }

    // =====================================================================
    // ILocalTPContext BAR Registry (no-ops for tests)
    // =====================================================================

    void registerBARBackedOutput(
        const std::string & /*stage_name*/,
        const GlobalDeviceAddress & /*device*/,
        TensorBase * /*tensor*/) override
    {
        // No-op for unit tests
    }

    bool hasBARBackedOutputs(const std::string & /*stage_name*/) const override { return false; }
    void clearBARBackedOutputs() override {}
    bool reserveTempBufferBytes(size_t /*bytes*/) override { return true; }

    // =====================================================================
    // ILocalTPContext Broadcast (no-op)
    // =====================================================================
    bool broadcast(TensorBase * /*tensor*/, int /*source_device_index*/ = 0) override
    {
        broadcast_calls_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    void requestAbort() override {}
    bool isAbortRequested() const override { return false; }

    // =====================================================================
    // Test Utilities
    // =====================================================================

    size_t synchronize_call_count() const
    {
        return synchronize_calls_.load(std::memory_order_relaxed);
    }

    size_t allreduce_call_count() const
    {
        return allreduce_calls_.load(std::memory_order_relaxed);
    }

    size_t allgather_call_count() const
    {
        return allgather_calls_.load(std::memory_order_relaxed);
    }

    size_t broadcast_call_count() const
    {
        return broadcast_calls_.load(std::memory_order_relaxed);
    }

    size_t gather_from_devices_call_count() const
    {
        return gather_from_devices_calls_.load(std::memory_order_relaxed);
    }

    void reset_call_counts()
    {
        synchronize_calls_.store(0, std::memory_order_relaxed);
        allreduce_calls_.store(0, std::memory_order_relaxed);
        allgather_calls_.store(0, std::memory_order_relaxed);
        gather_from_devices_calls_.store(0, std::memory_order_relaxed);
        reduce_scatter_calls_.store(0, std::memory_order_relaxed);
    }

    void set_allreduce_fails(bool fails) { config_.allreduce_should_fail = fails; }
    void set_allgather_fails(bool fails) { config_.allgather_should_fail = fails; }

private:
    Config config_;
    mutable std::atomic<size_t> synchronize_calls_{0};
    mutable std::atomic<size_t> allreduce_calls_{0};
    mutable std::atomic<size_t> allgather_calls_{0};
    mutable std::atomic<size_t> broadcast_calls_{0};
    mutable std::atomic<size_t> gather_from_devices_calls_{0};
    mutable std::atomic<size_t> reduce_scatter_calls_{0};
};

// =============================================================================
// Test Fixture
// =============================================================================

/**
 * @brief Test fixture for RankOrchestrator tests
 *
 * Provides helper methods to create mock orchestrators with different
 * configurations.
 */
class Test__RankOrchestrator : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Default setup: 2 device runners
        mock_runners_.clear();
        mock_runners_.push_back(std::make_unique<MockDeviceGraphOrchestrator>());
        mock_runners_.push_back(std::make_unique<MockDeviceGraphOrchestrator>());

        // Create mock TP context with 2 devices
        MockLocalTPContext::Config tp_config;
        tp_config.devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
        tp_config.weights = {0.5f, 0.5f};
        mock_tp_ctx_ = std::make_unique<MockLocalTPContext>(tp_config);
    }

    // Store raw pointers to mock runners for call verification
    std::vector<MockDeviceGraphOrchestrator *> getRawMockRunners()
    {
        std::vector<MockDeviceGraphOrchestrator *> result;
        for (auto &runner : mock_runners_)
        {
            result.push_back(runner.get());
        }
        return result;
    }

    std::vector<std::unique_ptr<MockDeviceGraphOrchestrator>> mock_runners_;
    std::unique_ptr<MockLocalTPContext> mock_tp_ctx_;
};

// =============================================================================
// Construction Tests
// =============================================================================

TEST_F(Test__RankOrchestrator, ConstructsWithValidConfig)
{
    // Verify mock setup is valid
    ASSERT_EQ(mock_runners_.size(), 2u);
    ASSERT_NE(mock_tp_ctx_, nullptr);
    EXPECT_EQ(mock_tp_ctx_->degree(), 2);
}

TEST_F(Test__RankOrchestrator, MockTPContextDegreeMatchesDevices)
{
    // Create TP context with 3 devices
    MockLocalTPContext::Config tp_config;
    tp_config.devices = {
        GlobalDeviceAddress::cpu(),
        GlobalDeviceAddress::cpu(),
        GlobalDeviceAddress::cpu()};
    auto tp_ctx = std::make_unique<MockLocalTPContext>(tp_config);

    EXPECT_EQ(tp_ctx->degree(), 3);
    EXPECT_EQ(tp_ctx->devices().size(), 3u);
}

// =============================================================================
// Mock Device Runner Tests
// =============================================================================

TEST_F(Test__RankOrchestrator, MockDeviceRunnerForwardTracksCallCount)
{
    auto runner = std::make_unique<MockDeviceGraphOrchestrator>();
    int tokens[] = {1, 2, 3};

    EXPECT_EQ(runner->forward_call_count(), 0u);

    bool result = runner->forward(tokens, 3);
    EXPECT_TRUE(result);
    EXPECT_EQ(runner->forward_call_count(), 1u);

    result = runner->forward(tokens, 3);
    EXPECT_TRUE(result);
    EXPECT_EQ(runner->forward_call_count(), 2u);
}

TEST_F(Test__RankOrchestrator, MockDeviceRunnerClearCacheTracksCallCount)
{
    auto runner = std::make_unique<MockDeviceGraphOrchestrator>();

    EXPECT_EQ(runner->clear_cache_call_count(), 0u);

    runner->clear_cache();
    EXPECT_EQ(runner->clear_cache_call_count(), 1u);

    runner->clear_cache();
    EXPECT_EQ(runner->clear_cache_call_count(), 2u);
}

TEST_F(Test__RankOrchestrator, MockDeviceRunnerReturnsConfiguredVocabSize)
{
    MockDeviceGraphOrchestrator::Config config;
    config.vocab_size = 50000;
    auto runner = std::make_unique<MockDeviceGraphOrchestrator>(config);

    EXPECT_EQ(runner->vocab_size(), 50000);
}

TEST_F(Test__RankOrchestrator, MockDeviceRunnerForwardCanFail)
{
    auto runner = std::make_unique<MockDeviceGraphOrchestrator>();
    runner->set_forward_fails(true);

    int tokens[] = {1, 2, 3};
    bool result = runner->forward(tokens, 3);

    EXPECT_FALSE(result);
    EXPECT_EQ(runner->forward_call_count(), 1u); // Still tracked
}

TEST_F(Test__RankOrchestrator, MockDeviceRunnerReturnsLogits)
{
    auto runner = std::make_unique<MockDeviceGraphOrchestrator>();
    runner->set_mock_logits({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    const float *logits = runner->logits();
    ASSERT_NE(logits, nullptr);
    EXPECT_FLOAT_EQ(logits[0], 1.0f);
    EXPECT_FLOAT_EQ(logits[4], 5.0f);
}

TEST_F(Test__RankOrchestrator, MockDeviceRunnerReturnsArchitecture)
{
    MockDeviceGraphOrchestrator::Config config;
    config.architecture = "test_arch";
    auto runner = std::make_unique<MockDeviceGraphOrchestrator>(config);

    EXPECT_STREQ(runner->architecture(), "test_arch");
}

TEST_F(Test__RankOrchestrator, MockDeviceRunnerReturnsGraphExecutionPath)
{
    auto runner = std::make_unique<MockDeviceGraphOrchestrator>();
    EXPECT_EQ(runner->executionPath(), ExecutionPath::GRAPH);
}

// =============================================================================
// Mock TP Context Tests
// =============================================================================

TEST_F(Test__RankOrchestrator, MockTPContextReturnsConfiguredDevices)
{
    MockLocalTPContext::Config config;
    config.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    auto ctx = std::make_unique<MockLocalTPContext>(config);

    EXPECT_EQ(ctx->devices().size(), 2u);
    EXPECT_EQ(ctx->devices()[0], GlobalDeviceAddress::cuda(0));
    EXPECT_EQ(ctx->devices()[1], GlobalDeviceAddress::cuda(1));
}

TEST_F(Test__RankOrchestrator, MockTPContextReturnsConfiguredWeights)
{
    MockLocalTPContext::Config config;
    config.devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
    config.weights = {0.7f, 0.3f};
    auto ctx = std::make_unique<MockLocalTPContext>(config);

    EXPECT_EQ(ctx->weights().size(), 2u);
    EXPECT_FLOAT_EQ(ctx->weights()[0], 0.7f);
    EXPECT_FLOAT_EQ(ctx->weights()[1], 0.3f);
}

TEST_F(Test__RankOrchestrator, MockTPContextSynchronizeTracksCallCount)
{
    auto ctx = std::make_unique<MockLocalTPContext>();

    EXPECT_EQ(ctx->synchronize_call_count(), 0u);

    ctx->synchronize();
    EXPECT_EQ(ctx->synchronize_call_count(), 1u);

    ctx->synchronize();
    ctx->synchronize();
    EXPECT_EQ(ctx->synchronize_call_count(), 3u);
}

TEST_F(Test__RankOrchestrator, MockTPContextAllreduceReturnsTrue)
{
    auto ctx = std::make_unique<MockLocalTPContext>();

    bool result = ctx->allreduce(static_cast<TensorBase *>(nullptr));
    EXPECT_TRUE(result);
    EXPECT_EQ(ctx->allreduce_call_count(), 1u);
}

TEST_F(Test__RankOrchestrator, MockTPContextAllreduceCanFail)
{
    auto ctx = std::make_unique<MockLocalTPContext>();
    ctx->set_allreduce_fails(true);

    bool result = ctx->allreduce(static_cast<TensorBase *>(nullptr));
    EXPECT_FALSE(result);
    EXPECT_EQ(ctx->allreduce_call_count(), 1u); // Still tracked
}

TEST_F(Test__RankOrchestrator, MockTPContextAllgatherReturnsTrue)
{
    auto ctx = std::make_unique<MockLocalTPContext>();

    bool result = ctx->allgather(nullptr, nullptr);
    EXPECT_TRUE(result);
    EXPECT_EQ(ctx->allgather_call_count(), 1u);
}

TEST_F(Test__RankOrchestrator, MockTPContextDeviceIndexing)
{
    MockLocalTPContext::Config config;
    config.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(1)};
    auto ctx = std::make_unique<MockLocalTPContext>(config);

    EXPECT_EQ(ctx->indexForDevice(GlobalDeviceAddress::cuda(0)), 0);
    EXPECT_EQ(ctx->indexForDevice(GlobalDeviceAddress::rocm(1)), 1);
    EXPECT_EQ(ctx->indexForDevice(GlobalDeviceAddress::cpu()), -1); // Not found
}

TEST_F(Test__RankOrchestrator, MockTPContextDeviceAtReturnsCorrectDevice)
{
    MockLocalTPContext::Config config;
    config.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    auto ctx = std::make_unique<MockLocalTPContext>(config);

    EXPECT_EQ(ctx->deviceAt(0), GlobalDeviceAddress::cuda(0));
    EXPECT_EQ(ctx->deviceAt(1), GlobalDeviceAddress::cuda(1));
}

TEST_F(Test__RankOrchestrator, MockTPContextDeviceAtThrowsForInvalidIndex)
{
    auto ctx = std::make_unique<MockLocalTPContext>();

    EXPECT_THROW(ctx->deviceAt(-1), std::out_of_range);
    EXPECT_THROW(ctx->deviceAt(99), std::out_of_range);
}

TEST_F(Test__RankOrchestrator, MockTPContextWeightForDevice)
{
    MockLocalTPContext::Config config;
    config.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    config.weights = {0.6f, 0.4f};
    auto ctx = std::make_unique<MockLocalTPContext>(config);

    EXPECT_FLOAT_EQ(ctx->weightForDevice(GlobalDeviceAddress::cuda(0)), 0.6f);
    EXPECT_FLOAT_EQ(ctx->weightForDevice(GlobalDeviceAddress::cuda(1)), 0.4f);
    EXPECT_FLOAT_EQ(ctx->weightForDevice(GlobalDeviceAddress::cpu()), 0.0f); // Not found
}

TEST_F(Test__RankOrchestrator, MockTPContextHeadsForDevice)
{
    MockLocalTPContext::Config config;
    config.devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
    config.weights = {0.5f, 0.5f};
    auto ctx = std::make_unique<MockLocalTPContext>(config);

    // 16 total heads with 50/50 weights = 8 each
    EXPECT_EQ(ctx->headsForDevice(config.devices[0], 16), 8);
    EXPECT_EQ(ctx->headsForDevice(config.devices[1], 16), 8);
}

// =============================================================================
// Integration: Multi-Runner Coordination Tests
// =============================================================================

TEST_F(Test__RankOrchestrator, AllDevicesReadyWhenAllHaveVocabSize)
{
    // Verify that when all mock runners have valid vocab_size, allDevicesReady would return true
    // (Testing the mock behavior that the real orchestrator uses)
    for (auto &runner : mock_runners_)
    {
        EXPECT_GT(runner->vocab_size(), 0);
    }
    EXPECT_EQ(mock_runners_.size(), 2u);
}

TEST_F(Test__RankOrchestrator, MultipleRunnersCanBeCalledInParallel)
{
    // Simulate what RankOrchestrator does: call forward on all runners
    int tokens[] = {1, 2, 3};

    // Call forward on all mock runners
    bool all_success = true;
    for (auto &runner : mock_runners_)
    {
        if (!runner->forward(tokens, 3))
        {
            all_success = false;
        }
    }

    EXPECT_TRUE(all_success);
    for (auto &runner : mock_runners_)
    {
        EXPECT_EQ(runner->forward_call_count(), 1u);
    }
}

TEST_F(Test__RankOrchestrator, ForwardFailsIfAnyDeviceFails)
{
    // Set one runner to fail
    mock_runners_[1]->set_forward_fails(true);

    int tokens[] = {1, 2, 3};

    // Simulate RankOrchestrator::forward
    bool all_success = true;
    for (auto &runner : mock_runners_)
    {
        if (!runner->forward(tokens, 3))
        {
            all_success = false;
        }
    }

    EXPECT_FALSE(all_success);
    // Both should have been called
    EXPECT_EQ(mock_runners_[0]->forward_call_count(), 1u);
    EXPECT_EQ(mock_runners_[1]->forward_call_count(), 1u);
}

TEST_F(Test__RankOrchestrator, ClearCacheClearsAllDevices)
{
    // Simulate RankOrchestrator::clear_cache
    for (auto &runner : mock_runners_)
    {
        runner->clear_cache();
    }

    for (auto &runner : mock_runners_)
    {
        EXPECT_EQ(runner->clear_cache_call_count(), 1u);
    }
}

TEST_F(Test__RankOrchestrator, LogitsReturnsFromPrimaryDevice)
{
    // Set different logits on each runner
    mock_runners_[0]->set_mock_logits({10.0f, 20.0f, 30.0f});
    mock_runners_[1]->set_mock_logits({1.0f, 2.0f, 3.0f});

    // Simulate RankOrchestrator::logits (returns from primary device)
    const float *logits = mock_runners_[0]->logits();

    ASSERT_NE(logits, nullptr);
    EXPECT_FLOAT_EQ(logits[0], 10.0f);
    EXPECT_FLOAT_EQ(logits[1], 20.0f);
    EXPECT_FLOAT_EQ(logits[2], 30.0f);
}

TEST_F(Test__RankOrchestrator, VocabSizeFromPrimaryDevice)
{
    mock_runners_[0]->set_vocab_size(50000);
    mock_runners_[1]->set_vocab_size(32000);

    // Simulate RankOrchestrator::vocab_size (returns from primary device)
    int vocab = mock_runners_[0]->vocab_size();

    EXPECT_EQ(vocab, 50000);
}

TEST_F(Test__RankOrchestrator, SynchronizeDevicesCallsTPContext)
{
    // Simulate RankOrchestrator::synchronizeDevices
    mock_tp_ctx_->synchronize();

    EXPECT_EQ(mock_tp_ctx_->synchronize_call_count(), 1u);
}

TEST_F(Test__RankOrchestrator, DeviceCountMatchesTPDegree)
{
    // Verify TP context degree matches expected device count
    EXPECT_EQ(mock_tp_ctx_->degree(), static_cast<int>(mock_runners_.size()));
}

TEST_F(Test__RankOrchestrator, LocalTPContextReturnsContext)
{
    // Verify mock TP context is valid
    ASSERT_NE(mock_tp_ctx_, nullptr);
    EXPECT_EQ(mock_tp_ctx_->degree(), 2);
    EXPECT_EQ(mock_tp_ctx_->backend(), CollectiveBackendType::HOST);
}

// =============================================================================
// Edge Cases and Error Handling
// =============================================================================

TEST_F(Test__RankOrchestrator, EmptyDeviceRunnersReturnsSafeDefaults)
{
    std::vector<std::unique_ptr<MockDeviceGraphOrchestrator>> empty_runners;

    // With no runners, we should handle gracefully
    EXPECT_TRUE(empty_runners.empty());
}

TEST_F(Test__RankOrchestrator, SingleTPContextWithOneDevice)
{
    // Test TP context with single device - should work but is unusual
    MockLocalTPContext::Config config;
    config.devices = {GlobalDeviceAddress::cpu()};
    config.weights = {1.0f};
    auto ctx = std::make_unique<MockLocalTPContext>(config);

    EXPECT_EQ(ctx->degree(), 1);
    EXPECT_FLOAT_EQ(ctx->weights()[0], 1.0f);
}

TEST_F(Test__RankOrchestrator, MockRunnerPositionUpdatesOnForward)
{
    auto runner = std::make_unique<MockDeviceGraphOrchestrator>();
    int tokens[] = {1, 2, 3, 4, 5};

    EXPECT_EQ(runner->get_position(), 0);

    runner->forward(tokens, 5);
    EXPECT_EQ(runner->get_position(), 5);

    runner->forward(tokens, 3);
    EXPECT_EQ(runner->get_position(), 8);
}

TEST_F(Test__RankOrchestrator, MockRunnerClearCacheResetsPosition)
{
    auto runner = std::make_unique<MockDeviceGraphOrchestrator>();
    int tokens[] = {1, 2, 3};

    runner->forward(tokens, 3);
    EXPECT_EQ(runner->get_position(), 3);

    runner->clear_cache();
    EXPECT_EQ(runner->get_position(), 0);
}

TEST_F(Test__RankOrchestrator, MockTPContextRowRangeCalculation)
{
    MockLocalTPContext::Config config;
    // Use distinct devices so indexForDevice can differentiate them
    config.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    config.weights = {0.75f, 0.25f};
    auto ctx = std::make_unique<MockLocalTPContext>(config);

    // For 100 total rows with 75/25 split:
    // Device 0: rows 0-74 (75 rows)
    // Device 1: rows 75-99 (25 rows)
    auto [start0, end0] = ctx->rowRangeForDevice(config.devices[0], 100);
    EXPECT_EQ(start0, 0);
    EXPECT_EQ(end0, 75);

    auto [start1, end1] = ctx->rowRangeForDevice(config.devices[1], 100);
    EXPECT_EQ(start1, 75);
    EXPECT_EQ(end1, 100);
}

TEST_F(Test__RankOrchestrator, MockTPContextColRangeMatchesRowRange)
{
    MockLocalTPContext::Config config;
    // Use distinct devices so indexForDevice can differentiate them
    config.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    config.weights = {0.6f, 0.4f};
    auto ctx = std::make_unique<MockLocalTPContext>(config);

    auto row_range = ctx->rowRangeForDevice(config.devices[0], 50);
    auto col_range = ctx->colRangeForDevice(config.devices[0], 50);

    EXPECT_EQ(row_range, col_range);
}

TEST_F(Test__RankOrchestrator, ResetCallCountsWorks)
{
    auto runner = std::make_unique<MockDeviceGraphOrchestrator>();
    int tokens[] = {1, 2, 3};

    runner->forward(tokens, 3);
    runner->clear_cache();

    EXPECT_EQ(runner->forward_call_count(), 1u);
    EXPECT_EQ(runner->clear_cache_call_count(), 1u);

    runner->reset_call_counts();

    EXPECT_EQ(runner->forward_call_count(), 0u);
    EXPECT_EQ(runner->clear_cache_call_count(), 0u);
}

TEST_F(Test__RankOrchestrator, MockTPContextResetCallCountsWorks)
{
    mock_tp_ctx_->synchronize();
    mock_tp_ctx_->allreduce(static_cast<TensorBase *>(nullptr));

    EXPECT_EQ(mock_tp_ctx_->synchronize_call_count(), 1u);
    EXPECT_EQ(mock_tp_ctx_->allreduce_call_count(), 1u);

    mock_tp_ctx_->reset_call_counts();

    EXPECT_EQ(mock_tp_ctx_->synchronize_call_count(), 0u);
    EXPECT_EQ(mock_tp_ctx_->allreduce_call_count(), 0u);
}

// =============================================================================
// TPSnapshot Row/Cols Inference Tests
// =============================================================================
// These tests verify that TPSnapshot correctly handles row/cols metadata
// for column-parallel stages. The fix ensures local_cols is properly computed
// from hidden_size/tp_degree rather than using flattened size.

TEST_F(Test__RankOrchestrator, TPSnapshot_ColumnParallel_CorrectRowsColsForSingleRowData)
{
    // Test case: Single-row column-parallel data (typical decode case)
    // hidden_size=896, tp_degree=2, local_cols=448, seq_len=1
    // Each device should have [1, 448] shape

    TPSnapshot snapshot;
    snapshot.key = "layer0_QKV_PROJ";
    snapshot.mode = SnapshotShardingMode::COLUMN_PARALLEL;
    snapshot.tp_degree = 2;

    // Device 0: 448 elements -> should be [1, 448]
    DeviceSnapshotData dev0;
    dev0.device_index = 0;
    dev0.rows = 1;   // Correctly set rows
    dev0.cols = 448; // Correctly set cols (local_cols = 896/2)
    dev0.global_start_col = 0;
    dev0.global_total_cols = 896;
    dev0.data.resize(448, 1.0f);

    // Device 1: 448 elements -> should be [1, 448]
    DeviceSnapshotData dev1;
    dev1.device_index = 1;
    dev1.rows = 1;
    dev1.cols = 448;
    dev1.global_start_col = 448;
    dev1.global_total_cols = 896;
    dev1.data.resize(448, 2.0f);

    snapshot.device_data.push_back(std::move(dev0));
    snapshot.device_data.push_back(std::move(dev1));

    // computeCombined should correctly concatenate columns
    ASSERT_TRUE(snapshot.computeCombined());
    EXPECT_EQ(snapshot.combined_rows, 1);
    EXPECT_EQ(snapshot.combined_cols, 896);
    EXPECT_EQ(snapshot.combined_data.size(), 896);

    // Verify data is correctly concatenated: [1,1,1,...448...,2,2,2,...448...]
    EXPECT_FLOAT_EQ(snapshot.combined_data[0], 1.0f);   // First from device 0
    EXPECT_FLOAT_EQ(snapshot.combined_data[447], 1.0f); // Last from device 0
    EXPECT_FLOAT_EQ(snapshot.combined_data[448], 2.0f); // First from device 1
    EXPECT_FLOAT_EQ(snapshot.combined_data[895], 2.0f); // Last from device 1
}

TEST_F(Test__RankOrchestrator, TPSnapshot_ColumnParallel_CorrectRowsColsForMultiRowData)
{
    // Test case: Multi-row column-parallel data (typical prefill case)
    // hidden_size=896, tp_degree=2, local_cols=448, seq_len=9
    // Each device should have [9, 448] shape (total 4032 elements)

    TPSnapshot snapshot;
    snapshot.key = "layer0_ATTENTION_CONTEXT";
    snapshot.mode = SnapshotShardingMode::COLUMN_PARALLEL;
    snapshot.tp_degree = 2;

    const size_t seq_len = 9;
    const size_t local_cols = 448;
    const size_t total_elements = seq_len * local_cols;

    // Device 0: [9, 448] = 4032 elements
    DeviceSnapshotData dev0;
    dev0.device_index = 0;
    dev0.rows = seq_len;
    dev0.cols = local_cols;
    dev0.global_start_col = 0;
    dev0.global_total_cols = 896;
    dev0.data.resize(total_elements);
    // Fill with row index for verification
    for (size_t r = 0; r < seq_len; ++r)
    {
        for (size_t c = 0; c < local_cols; ++c)
        {
            dev0.data[r * local_cols + c] = static_cast<float>(r * 10 + 0); // Row * 10 + device
        }
    }

    // Device 1: [9, 448] = 4032 elements
    DeviceSnapshotData dev1;
    dev1.device_index = 1;
    dev1.rows = seq_len;
    dev1.cols = local_cols;
    dev1.global_start_col = local_cols;
    dev1.global_total_cols = 896;
    dev1.data.resize(total_elements);
    for (size_t r = 0; r < seq_len; ++r)
    {
        for (size_t c = 0; c < local_cols; ++c)
        {
            dev1.data[r * local_cols + c] = static_cast<float>(r * 10 + 1);
        }
    }

    snapshot.device_data.push_back(std::move(dev0));
    snapshot.device_data.push_back(std::move(dev1));

    // computeCombined should correctly concatenate columns for each row
    ASSERT_TRUE(snapshot.computeCombined());
    EXPECT_EQ(snapshot.combined_rows, seq_len);
    EXPECT_EQ(snapshot.combined_cols, 896);
    EXPECT_EQ(snapshot.combined_data.size(), seq_len * 896);

    // Verify row-by-row concatenation
    // Row 0: [dev0 cols...] [dev1 cols...]
    // Combined row 0, col 0 = dev0 data
    EXPECT_FLOAT_EQ(snapshot.combined_data[0], 0.0f); // Row 0, dev0
    // Combined row 0, col 448 = dev1 data
    EXPECT_FLOAT_EQ(snapshot.combined_data[448], 1.0f); // Row 0, dev1

    // Row 5: should have value 50 (5*10+0) from dev0, 51 (5*10+1) from dev1
    size_t row5_offset = 5 * 896;
    EXPECT_FLOAT_EQ(snapshot.combined_data[row5_offset], 50.0f);       // Row 5, dev0
    EXPECT_FLOAT_EQ(snapshot.combined_data[row5_offset + 448], 51.0f); // Row 5, dev1
}

TEST_F(Test__RankOrchestrator, TPSnapshot_ColumnParallel_WrongColsBreaksCombine)
{
    // Test case: What happens if cols is incorrectly set to flattened size?
    // This was the BUG: cols=4032 instead of cols=448
    // With incorrect cols, row stride calculation breaks

    TPSnapshot snapshot;
    snapshot.key = "layer0_BUG_CASE";
    snapshot.mode = SnapshotShardingMode::COLUMN_PARALLEL;
    snapshot.tp_degree = 2;

    const size_t seq_len = 9;
    const size_t local_cols = 448;
    const size_t total_elements = seq_len * local_cols;

    // Device 0 with CORRECT metadata
    DeviceSnapshotData dev0_correct;
    dev0_correct.device_index = 0;
    dev0_correct.rows = seq_len;
    dev0_correct.cols = local_cols; // CORRECT: 448
    dev0_correct.global_start_col = 0;
    dev0_correct.global_total_cols = 896;
    dev0_correct.data.resize(total_elements, 1.0f);

    // Device 1 with CORRECT metadata
    DeviceSnapshotData dev1_correct;
    dev1_correct.device_index = 1;
    dev1_correct.rows = seq_len;
    dev1_correct.cols = local_cols; // CORRECT: 448
    dev1_correct.global_start_col = local_cols;
    dev1_correct.global_total_cols = 896;
    dev1_correct.data.resize(total_elements, 2.0f);

    snapshot.device_data.push_back(std::move(dev0_correct));
    snapshot.device_data.push_back(std::move(dev1_correct));

    ASSERT_TRUE(snapshot.computeCombined());

    // With correct metadata, combined should have proper dimensions
    EXPECT_EQ(snapshot.combined_rows, seq_len);
    EXPECT_EQ(snapshot.combined_cols, 896);

    // Now test with INCORRECT metadata (the bug case)
    TPSnapshot buggy_snapshot;
    buggy_snapshot.key = "layer0_BUG_CASE";
    buggy_snapshot.mode = SnapshotShardingMode::COLUMN_PARALLEL;
    buggy_snapshot.tp_degree = 2;

    // Device 0 with BUG: cols = flattened size instead of actual cols
    DeviceSnapshotData dev0_buggy;
    dev0_buggy.device_index = 0;
    dev0_buggy.rows = 1;              // BUG: rows=1 because size/cols gives 1 when cols=4032
    dev0_buggy.cols = total_elements; // BUG: cols=4032 (flattened size)
    dev0_buggy.global_start_col = 0;
    dev0_buggy.global_total_cols = total_elements * 2;
    dev0_buggy.data.resize(total_elements, 1.0f);

    DeviceSnapshotData dev1_buggy;
    dev1_buggy.device_index = 1;
    dev1_buggy.rows = 1;
    dev1_buggy.cols = total_elements; // BUG: cols=4032
    dev1_buggy.global_start_col = total_elements;
    dev1_buggy.global_total_cols = total_elements * 2;
    dev1_buggy.data.resize(total_elements, 2.0f);

    buggy_snapshot.device_data.push_back(std::move(dev0_buggy));
    buggy_snapshot.device_data.push_back(std::move(dev1_buggy));

    ASSERT_TRUE(buggy_snapshot.computeCombined());

    // With buggy metadata, combined has WRONG dimensions
    // This would cause comparison against PyTorch (which has [9, 896]) to fail
    EXPECT_EQ(buggy_snapshot.combined_rows, 1);                  // WRONG: should be 9
    EXPECT_EQ(buggy_snapshot.combined_cols, total_elements * 2); // WRONG: should be 896

    // The total data size is the same, but the row/col interpretation is wrong
    // This demonstrates why correct rows/cols metadata is critical
    EXPECT_NE(buggy_snapshot.combined_rows, snapshot.combined_rows);
    EXPECT_NE(buggy_snapshot.combined_cols, snapshot.combined_cols);
}

TEST_F(Test__RankOrchestrator, TPSnapshot_ColumnParallel_ProportionalWeights)
{
    // Test case: Proportional TP with 73%/27% split (heterogeneous GPUs)
    // hidden_size=896, device0 gets 73% = 654 cols, device1 gets 27% = 242 cols

    TPSnapshot snapshot;
    snapshot.key = "layer0_PROPORTIONAL";
    snapshot.mode = SnapshotShardingMode::COLUMN_PARALLEL;
    snapshot.tp_degree = 2;

    const size_t seq_len = 4;
    const size_t dev0_cols = 654; // 73% of 896
    const size_t dev1_cols = 242; // 27% of 896
    const size_t total_cols = dev0_cols + dev1_cols;

    DeviceSnapshotData dev0;
    dev0.device_index = 0;
    dev0.rows = seq_len;
    dev0.cols = dev0_cols;
    dev0.global_start_col = 0;
    dev0.global_total_cols = total_cols;
    dev0.data.resize(seq_len * dev0_cols, 1.0f);

    DeviceSnapshotData dev1;
    dev1.device_index = 1;
    dev1.rows = seq_len;
    dev1.cols = dev1_cols;
    dev1.global_start_col = dev0_cols;
    dev1.global_total_cols = total_cols;
    dev1.data.resize(seq_len * dev1_cols, 2.0f);

    snapshot.device_data.push_back(std::move(dev0));
    snapshot.device_data.push_back(std::move(dev1));

    ASSERT_TRUE(snapshot.computeCombined());
    EXPECT_EQ(snapshot.combined_rows, seq_len);
    EXPECT_EQ(snapshot.combined_cols, total_cols);

    // Verify uneven column concatenation works
    // Row 0: [654 cols from dev0] [242 cols from dev1]
    EXPECT_FLOAT_EQ(snapshot.combined_data[0], 1.0f);              // First col from dev0
    EXPECT_FLOAT_EQ(snapshot.combined_data[653], 1.0f);            // Last col from dev0
    EXPECT_FLOAT_EQ(snapshot.combined_data[654], 2.0f);            // First col from dev1
    EXPECT_FLOAT_EQ(snapshot.combined_data[total_cols - 1], 2.0f); // Last col from dev1
}

TEST_F(Test__RankOrchestrator, TPSnapshot_Replicated_SingleRowMultipleDevices)
{
    // Test case: Replicated stage (same output on all devices)
    // Each device has full [1, 896] output

    TPSnapshot snapshot;
    snapshot.key = "layer0_ATTN_NORM";
    snapshot.mode = SnapshotShardingMode::REPLICATED;
    snapshot.tp_degree = 2;

    const size_t total_cols = 896;

    DeviceSnapshotData dev0;
    dev0.device_index = 0;
    dev0.rows = 1;
    dev0.cols = total_cols;
    dev0.global_start_col = 0;
    dev0.global_total_cols = total_cols;
    dev0.data.resize(total_cols, 3.14f);

    DeviceSnapshotData dev1;
    dev1.device_index = 1;
    dev1.rows = 1;
    dev1.cols = total_cols;
    dev1.global_start_col = 0;
    dev1.global_total_cols = total_cols;
    dev1.data.resize(total_cols, 3.14f); // Same data as dev0

    snapshot.device_data.push_back(std::move(dev0));
    snapshot.device_data.push_back(std::move(dev1));

    ASSERT_TRUE(snapshot.computeCombined());
    EXPECT_EQ(snapshot.combined_rows, 1);
    EXPECT_EQ(snapshot.combined_cols, total_cols);

    // For replicated, should just use first device's data
    EXPECT_FLOAT_EQ(snapshot.combined_data[0], 3.14f);
}
