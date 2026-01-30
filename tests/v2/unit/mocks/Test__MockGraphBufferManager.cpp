/**
 * @file Test__MockGraphBufferManager.cpp
 * @brief Unit tests for MockGraphBufferManager
 *
 * Tests the mock graph buffer manager implementation including:
 * - Buffer registration and retrieval
 * - Allocation behavior configuration
 * - Call tracking
 * - Builder pattern
 * - Failure injection
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "mocks/MockGraphBufferManager.h"
#include "utils/TestTensorFactory.h"
#include "execution/local_execution/graph/LivenessAnalyzer.h" // For AliasingGroup
#include "execution/local_execution/graph/GraphExecutor.h"    // For ComputeGraph
#include <memory>
#include <vector>
#include <thread>

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__MockGraphBufferManager : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create some test tensors
        tensor1_ = TestTensorFactory::createFP32Random({32, 896});
        tensor2_ = TestTensorFactory::createFP32Random({32, 3584});
        tensor3_ = TestTensorFactory::createFP32Random({1, 896});

        // Default mock
        mock_ = std::make_shared<MockGraphBufferManager>();
    }

    std::unique_ptr<FP32Tensor> tensor1_;
    std::unique_ptr<FP32Tensor> tensor2_;
    std::unique_ptr<FP32Tensor> tensor3_;
    std::shared_ptr<MockGraphBufferManager> mock_;
};

// =============================================================================
// Construction Tests
// =============================================================================

TEST_F(Test__MockGraphBufferManager, DefaultConstruction)
{
    MockGraphBufferManager mock;
    EXPECT_EQ(mock.bufferCount(), 0);
    EXPECT_EQ(mock.totalAllocatedBytes(), 0);
    EXPECT_TRUE(mock.config().track_calls);
    EXPECT_TRUE(mock.config().allocate_for_graph_succeeds);
}

TEST_F(Test__MockGraphBufferManager, ConfigConstruction)
{
    MockGraphBufferManager::Config config;
    config.track_calls = false;
    config.allocate_for_graph_succeeds = false;
    config.aliasing_savings_percent = 25.0;

    MockGraphBufferManager mock(config);

    EXPECT_FALSE(mock.config().track_calls);
    EXPECT_FALSE(mock.config().allocate_for_graph_succeeds);
    EXPECT_DOUBLE_EQ(mock.config().aliasing_savings_percent, 25.0);
}

TEST_F(Test__MockGraphBufferManager, CreateEmptyPreset)
{
    auto mock = MockGraphBufferManager::createEmpty();

    EXPECT_NE(mock, nullptr);
    EXPECT_EQ(mock->bufferCount(), 0);
    EXPECT_TRUE(mock->config().allocate_for_graph_succeeds);
}

TEST_F(Test__MockGraphBufferManager, CreateSuccessfulPreset)
{
    auto mock = MockGraphBufferManager::createSuccessful();

    EXPECT_TRUE(mock->config().allocate_for_graph_succeeds);
    EXPECT_TRUE(mock->config().allocate_buffer_succeeds);
    EXPECT_TRUE(mock->config().allocate_with_aliasing_succeeds);
    EXPECT_TRUE(mock->config().bind_buffer_succeeds);
}

TEST_F(Test__MockGraphBufferManager, CreateFailingPreset)
{
    auto mock = MockGraphBufferManager::createFailing();

    EXPECT_FALSE(mock->config().allocate_for_graph_succeeds);
    EXPECT_FALSE(mock->config().allocate_buffer_succeeds);
    EXPECT_FALSE(mock->config().allocate_with_aliasing_succeeds);
    EXPECT_FALSE(mock->config().bind_buffer_succeeds);
}

// =============================================================================
// Buffer Registration Tests
// =============================================================================

TEST_F(Test__MockGraphBufferManager, RegisterSingleBuffer)
{
    mock_->registerBuffer("layer0", "output", tensor1_.get());

    EXPECT_EQ(mock_->bufferCount(), 1);
    EXPECT_TRUE(mock_->hasBuffer("layer0", "output"));
    EXPECT_EQ(mock_->getBuffer("layer0", "output"), tensor1_.get());
}

TEST_F(Test__MockGraphBufferManager, RegisterMultipleBuffers)
{
    mock_->registerBuffer("layer0", "output", tensor1_.get());
    mock_->registerBuffer("layer0", "hidden", tensor2_.get());
    mock_->registerBuffer("layer1", "output", tensor3_.get());

    EXPECT_EQ(mock_->bufferCount(), 3);
    EXPECT_TRUE(mock_->hasBuffer("layer0", "output"));
    EXPECT_TRUE(mock_->hasBuffer("layer0", "hidden"));
    EXPECT_TRUE(mock_->hasBuffer("layer1", "output"));
}

TEST_F(Test__MockGraphBufferManager, RegisterBufferWithDescriptor)
{
    BufferDescriptor desc;
    desc.name = "output";
    desc.role = BufferRole::OUTPUT;
    desc.tensor_type = BufferTensorType::FP32;
    desc.shape = {32, 896};

    mock_->registerBuffer("layer0", desc, tensor1_.get());

    EXPECT_TRUE(mock_->hasBuffer("layer0", "output"));
    EXPECT_EQ(mock_->getBuffer("layer0", "output"), tensor1_.get());
}

TEST_F(Test__MockGraphBufferManager, UnregisterBuffer)
{
    mock_->registerBuffer("layer0", "output", tensor1_.get());
    EXPECT_TRUE(mock_->hasBuffer("layer0", "output"));

    mock_->unregisterBuffer("layer0", "output");

    EXPECT_FALSE(mock_->hasBuffer("layer0", "output"));
    EXPECT_EQ(mock_->bufferCount(), 0);
}

TEST_F(Test__MockGraphBufferManager, ClearBuffers)
{
    mock_->registerBuffer("layer0", "output", tensor1_.get());
    mock_->registerBuffer("layer1", "output", tensor2_.get());
    EXPECT_EQ(mock_->bufferCount(), 2);

    mock_->clearBuffers();

    EXPECT_EQ(mock_->bufferCount(), 0);
    EXPECT_FALSE(mock_->hasBuffer("layer0", "output"));
}

TEST_F(Test__MockGraphBufferManager, RegisterNullBuffer)
{
    mock_->registerBuffer("layer0", "output", nullptr);

    EXPECT_TRUE(mock_->hasBuffer("layer0", "output"));
    EXPECT_EQ(mock_->getBuffer("layer0", "output"), nullptr);
}

// =============================================================================
// Buffer Retrieval Tests
// =============================================================================

TEST_F(Test__MockGraphBufferManager, GetBufferByNameReturnsCorrectTensor)
{
    mock_->registerBuffer("layer0", "output", tensor1_.get());
    mock_->registerBuffer("layer0", "hidden", tensor2_.get());

    EXPECT_EQ(mock_->getBuffer("layer0", "output"), tensor1_.get());
    EXPECT_EQ(mock_->getBuffer("layer0", "hidden"), tensor2_.get());
}

TEST_F(Test__MockGraphBufferManager, GetBufferByNameReturnsNullIfNotFound)
{
    EXPECT_EQ(mock_->getBuffer("nonexistent", "buffer"), nullptr);
}

TEST_F(Test__MockGraphBufferManager, GetBufferByKey)
{
    mock_->registerBuffer("layer0", "output", tensor1_.get());

    BufferKey key{"layer0", "output"};
    EXPECT_EQ(mock_->getBuffer(key), tensor1_.get());
}

TEST_F(Test__MockGraphBufferManager, GetAllBufferKeys)
{
    mock_->registerBuffer("layer0", "output", tensor1_.get());
    mock_->registerBuffer("layer1", "hidden", tensor2_.get());

    auto keys = mock_->getAllBufferKeys();

    EXPECT_EQ(keys.size(), 2);
    // Keys order is not guaranteed (unordered_map)
    bool found_layer0 = false;
    bool found_layer1 = false;
    for (const auto &key : keys)
    {
        if (key.node_name == "layer0" && key.buffer_name == "output")
        {
            found_layer0 = true;
        }
        if (key.node_name == "layer1" && key.buffer_name == "hidden")
        {
            found_layer1 = true;
        }
    }
    EXPECT_TRUE(found_layer0);
    EXPECT_TRUE(found_layer1);
}

// =============================================================================
// Allocation Behavior Tests
// =============================================================================

TEST_F(Test__MockGraphBufferManager, AllocateBufferSucceedsByDefault)
{
    BufferDescriptor desc;
    desc.name = "test";

    EXPECT_TRUE(mock_->allocateBuffer("node", desc));
}

TEST_F(Test__MockGraphBufferManager, AllocateBufferCanBeMadeFailing)
{
    mock_->setAllocateBufferSucceeds(false);

    BufferDescriptor desc;
    desc.name = "test";

    EXPECT_FALSE(mock_->allocateBuffer("node", desc));
}

TEST_F(Test__MockGraphBufferManager, AllocateWithAliasingSucceedsByDefault)
{
    ComputeGraph graph; // Empty graph
    EXPECT_TRUE(mock_->allocateWithAliasing(graph));
}

TEST_F(Test__MockGraphBufferManager, AllocateWithAliasingCanBeMadeFailing)
{
    mock_->setAllocateWithAliasingSucceeds(false);

    ComputeGraph graph;
    EXPECT_FALSE(mock_->allocateWithAliasing(graph));
}

TEST_F(Test__MockGraphBufferManager, BindBufferSucceedsByDefault)
{
    mock_->registerBuffer("layer0", "output", tensor1_.get());

    TensorBase *bound = nullptr;
    EXPECT_TRUE(mock_->bindBuffer("layer0", "output", &bound));
    EXPECT_EQ(bound, tensor1_.get());
}

TEST_F(Test__MockGraphBufferManager, BindBufferCanBeMadeFailing)
{
    mock_->setBindBufferSucceeds(false);
    mock_->registerBuffer("layer0", "output", tensor1_.get());

    TensorBase *bound = nullptr;
    EXPECT_FALSE(mock_->bindBuffer("layer0", "output", &bound));
}

TEST_F(Test__MockGraphBufferManager, BindBufferWithNullTarget)
{
    mock_->registerBuffer("layer0", "output", tensor1_.get());

    // Should not crash when target_ptr is null
    EXPECT_TRUE(mock_->bindBuffer("layer0", "output", nullptr));
}

// =============================================================================
// Callback Tests
// =============================================================================

TEST_F(Test__MockGraphBufferManager, AllocateBufferCallback)
{
    std::string captured_node;
    std::string captured_name;

    mock_->setAllocateBufferCallback(
        [&](const std::string &node, const BufferDescriptor &desc)
        {
            captured_node = node;
            captured_name = desc.name;
            return false; // Custom return value
        });

    BufferDescriptor desc;
    desc.name = "test_buffer";

    EXPECT_FALSE(mock_->allocateBuffer("test_node", desc));
    EXPECT_EQ(captured_node, "test_node");
    EXPECT_EQ(captured_name, "test_buffer");
}

TEST_F(Test__MockGraphBufferManager, AllocateWithAliasingCallback)
{
    bool callback_called = false;

    mock_->setAllocateWithAliasingCallback([&](ComputeGraph &)
                                           {
        callback_called = true;
        return true; });

    ComputeGraph graph;
    mock_->allocateWithAliasing(graph);

    EXPECT_TRUE(callback_called);
}

// =============================================================================
// Statistics Tests
// =============================================================================

TEST_F(Test__MockGraphBufferManager, StatsTrackTotalBytes)
{
    mock_->registerBuffer("layer0", "output", tensor1_.get());

    EXPECT_EQ(mock_->totalAllocatedBytes(), tensor1_->size_bytes());

    mock_->registerBuffer("layer1", "output", tensor2_.get());

    EXPECT_EQ(mock_->totalAllocatedBytes(),
              tensor1_->size_bytes() + tensor2_->size_bytes());
}

TEST_F(Test__MockGraphBufferManager, StatsTrackBufferCount)
{
    EXPECT_EQ(mock_->stats().total_buffers, 0);

    mock_->registerBuffer("layer0", "output", tensor1_.get());
    EXPECT_EQ(mock_->stats().total_buffers, 1);

    mock_->registerBuffer("layer1", "output", tensor2_.get());
    EXPECT_EQ(mock_->stats().total_buffers, 2);
}

TEST_F(Test__MockGraphBufferManager, ResetStats)
{
    mock_->registerBuffer("layer0", "output", tensor1_.get());
    EXPECT_GT(mock_->stats().total_buffers, 0);

    mock_->resetStats();

    EXPECT_EQ(mock_->stats().total_buffers, 0);
    EXPECT_EQ(mock_->stats().total_bytes, 0);
}

TEST_F(Test__MockGraphBufferManager, AliasingSavingsPercent)
{
    EXPECT_DOUBLE_EQ(mock_->aliasingSavingsPercent(), 0.0);

    mock_->setAliasingSavingsPercent(35.5);

    EXPECT_DOUBLE_EQ(mock_->aliasingSavingsPercent(), 35.5);
}

TEST_F(Test__MockGraphBufferManager, AliasingGroupCount)
{
    EXPECT_EQ(mock_->aliasingGroupCount(), 0);

    mock_->setAliasingGroupCount(3);

    EXPECT_EQ(mock_->aliasingGroupCount(), 3);
}

TEST_F(Test__MockGraphBufferManager, AddAliasingGroup)
{
    AliasingGroup group;
    group.max_size_bytes = 1024;

    mock_->addAliasingGroup(group);

    EXPECT_EQ(mock_->aliasingGroupCount(), 1);
    EXPECT_EQ(mock_->aliasingGroups().size(), 1);
    EXPECT_EQ(mock_->aliasingGroups()[0].max_size_bytes, 1024);
}

TEST_F(Test__MockGraphBufferManager, ClearAliasingGroups)
{
    AliasingGroup group;
    mock_->addAliasingGroup(group);
    EXPECT_EQ(mock_->aliasingGroupCount(), 1);

    mock_->clearAliasingGroups();

    EXPECT_EQ(mock_->aliasingGroupCount(), 0);
}

// =============================================================================
// Call Tracking Tests
// =============================================================================

TEST_F(Test__MockGraphBufferManager, CallCountsInitiallyZero)
{
    EXPECT_EQ(mock_->allocateForGraph_call_count(), 0);
    EXPECT_EQ(mock_->allocateBuffer_call_count(), 0);
    EXPECT_EQ(mock_->allocateWithAliasing_call_count(), 0);
    EXPECT_EQ(mock_->releaseAll_call_count(), 0);
    EXPECT_EQ(mock_->getBuffer_call_count(), 0);
    EXPECT_EQ(mock_->hasBuffer_call_count(), 0);
    EXPECT_EQ(mock_->bindBuffer_call_count(), 0);
    EXPECT_EQ(mock_->total_call_count(), 0);
}

TEST_F(Test__MockGraphBufferManager, TrackAllocateBufferCalls)
{
    BufferDescriptor desc;
    desc.name = "test";

    mock_->allocateBuffer("node1", desc);
    mock_->allocateBuffer("node2", desc);

    EXPECT_EQ(mock_->allocateBuffer_call_count(), 2);
}

TEST_F(Test__MockGraphBufferManager, TrackGetBufferCalls)
{
    mock_->registerBuffer("layer0", "output", tensor1_.get());

    mock_->getBuffer("layer0", "output");
    mock_->getBuffer("layer0", "nonexistent");
    mock_->getBuffer("nonexistent", "buffer");

    EXPECT_EQ(mock_->getBuffer_call_count(), 3);
}

TEST_F(Test__MockGraphBufferManager, TrackHasBufferCalls)
{
    mock_->hasBuffer("layer0", "output");
    mock_->hasBuffer("layer1", "hidden");

    EXPECT_EQ(mock_->hasBuffer_call_count(), 2);
}

TEST_F(Test__MockGraphBufferManager, TrackReleaseAllCalls)
{
    mock_->releaseAll();
    mock_->releaseAll();
    mock_->releaseAll();

    EXPECT_EQ(mock_->releaseAll_call_count(), 3);
}

TEST_F(Test__MockGraphBufferManager, TrackBindBufferCalls)
{
    TensorBase *bound = nullptr;

    mock_->bindBuffer("layer0", "output", &bound);
    mock_->bindBuffer("layer1", "hidden", &bound);

    EXPECT_EQ(mock_->bindBuffer_call_count(), 2);
}

TEST_F(Test__MockGraphBufferManager, TotalCallCount)
{
    BufferDescriptor desc;
    desc.name = "test";
    TensorBase *bound = nullptr;

    mock_->allocateBuffer("node", desc);           // 1 allocateBuffer call
    mock_->getBuffer("layer0", "output");          // 1 getBuffer + 1 getBufferByKey (internal)
    mock_->hasBuffer("layer0", "output");          // 1 hasBuffer call
    mock_->bindBuffer("layer0", "output", &bound); // 1 bindBuffer + 1 getBuffer + 1 getBufferByKey (internal)
    mock_->releaseAll();                           // 1 releaseAll call

    // Total: 1 + 2 + 1 + 3 + 1 = 8 calls
    EXPECT_EQ(mock_->total_call_count(), 8);
}

TEST_F(Test__MockGraphBufferManager, ResetCallCounts)
{
    BufferDescriptor desc;
    desc.name = "test";

    mock_->allocateBuffer("node", desc);
    mock_->getBuffer("layer0", "output");

    EXPECT_GT(mock_->total_call_count(), 0);

    mock_->reset_call_counts();

    EXPECT_EQ(mock_->allocateBuffer_call_count(), 0);
    EXPECT_EQ(mock_->getBuffer_call_count(), 0);
    EXPECT_EQ(mock_->total_call_count(), 0);
}

// =============================================================================
// ReleaseAll Tests
// =============================================================================

TEST_F(Test__MockGraphBufferManager, ReleaseAllClearsBuffers)
{
    mock_->registerBuffer("layer0", "output", tensor1_.get());
    mock_->registerBuffer("layer1", "output", tensor2_.get());

    EXPECT_EQ(mock_->bufferCount(), 2);

    mock_->releaseAll();

    EXPECT_EQ(mock_->bufferCount(), 0);
    EXPECT_FALSE(mock_->hasBuffer("layer0", "output"));
}

TEST_F(Test__MockGraphBufferManager, ReleaseAllResetsStats)
{
    mock_->registerBuffer("layer0", "output", tensor1_.get());
    EXPECT_GT(mock_->totalAllocatedBytes(), 0);

    mock_->releaseAll();

    EXPECT_EQ(mock_->totalAllocatedBytes(), 0);
    EXPECT_EQ(mock_->stats().total_buffers, 0);
}

// =============================================================================
// Builder Pattern Tests
// =============================================================================

TEST_F(Test__MockGraphBufferManager, BuilderBasicConfiguration)
{
    auto mock = MockGraphBufferManagerBuilder()
                    .setTrackCalls(true)
                    .setAllocateForGraphSucceeds(false)
                    .build();

    EXPECT_TRUE(mock->config().track_calls);
    EXPECT_FALSE(mock->config().allocate_for_graph_succeeds);
}

TEST_F(Test__MockGraphBufferManager, BuilderAddBuffer)
{
    auto mock = MockGraphBufferManagerBuilder()
                    .addBuffer("layer0", "output", tensor1_.get())
                    .build();

    EXPECT_EQ(mock->bufferCount(), 1);
    EXPECT_EQ(mock->getBuffer("layer0", "output"), tensor1_.get());
}

TEST_F(Test__MockGraphBufferManager, BuilderAddMultipleBuffers)
{
    auto mock = MockGraphBufferManagerBuilder()
                    .addBuffer("layer0", "output", tensor1_.get())
                    .addBuffer("layer0", "hidden", tensor2_.get())
                    .addBuffer("layer1", "output", tensor3_.get())
                    .build();

    EXPECT_EQ(mock->bufferCount(), 3);
}

TEST_F(Test__MockGraphBufferManager, BuilderSetAliasing)
{
    auto mock = MockGraphBufferManagerBuilder()
                    .setAliasingSavingsPercent(40.0)
                    .setAliasingGroupCount(5)
                    .build();

    EXPECT_DOUBLE_EQ(mock->aliasingSavingsPercent(), 40.0);
    EXPECT_EQ(mock->aliasingGroupCount(), 5);
}

TEST_F(Test__MockGraphBufferManager, BuilderAddAliasingGroup)
{
    AliasingGroup group;
    group.max_size_bytes = 2048;

    auto mock = MockGraphBufferManagerBuilder()
                    .addAliasingGroup(group)
                    .build();

    EXPECT_EQ(mock->aliasingGroups().size(), 1);
    EXPECT_EQ(mock->aliasingGroups()[0].max_size_bytes, 2048);
}

TEST_F(Test__MockGraphBufferManager, BuilderAllOptions)
{
    AliasingGroup group;
    group.max_size_bytes = 1024;

    auto mock = MockGraphBufferManagerBuilder()
                    .setTrackCalls(true)
                    .setAllocateForGraphSucceeds(true)
                    .setAllocateBufferSucceeds(true)
                    .setAllocateWithAliasingSucceeds(false)
                    .setBindBufferSucceeds(true)
                    .setAliasingSavingsPercent(30.0)
                    .addBuffer("stage1", "buf", tensor1_.get())
                    .addAliasingGroup(group)
                    .build();

    EXPECT_TRUE(mock->config().track_calls);
    EXPECT_TRUE(mock->config().allocate_for_graph_succeeds);
    EXPECT_FALSE(mock->config().allocate_with_aliasing_succeeds);
    EXPECT_DOUBLE_EQ(mock->aliasingSavingsPercent(), 30.0);
    EXPECT_EQ(mock->bufferCount(), 1);
    EXPECT_EQ(mock->aliasingGroupCount(), 1);
}

// =============================================================================
// Interface Compliance Tests
// =============================================================================

TEST_F(Test__MockGraphBufferManager, InterfaceCompliance_CanUseAsIGraphBufferManager)
{
    std::shared_ptr<IGraphBufferManager> interface_ptr = mock_;

    interface_ptr->releaseAll();
    EXPECT_EQ(mock_->releaseAll_call_count(), 1);
}

TEST_F(Test__MockGraphBufferManager, InterfaceCompliance_AllMethodsAccessible)
{
    std::shared_ptr<IGraphBufferManager> iface = mock_;

    // Test all interface methods are accessible
    ComputeGraph graph;
    iface->allocateWithAliasing(graph);

    BufferDescriptor desc;
    desc.name = "test";
    iface->allocateBuffer("node", desc);

    iface->hasBuffer("node", "buffer");
    iface->getBuffer("node", "buffer");
    iface->getAllBufferKeys();

    TensorBase *ptr = nullptr;
    iface->bindBuffer("node", "buffer", &ptr);

    [[maybe_unused]] auto &s = iface->stats();
    iface->resetStats();
    [[maybe_unused]] auto bc = iface->bufferCount();
    [[maybe_unused]] auto tb = iface->totalAllocatedBytes();
    [[maybe_unused]] auto ap = iface->aliasingSavingsPercent();
    [[maybe_unused]] auto agc = iface->aliasingGroupCount();
    [[maybe_unused]] const auto &ag = iface->aliasingGroups();

    iface->dumpBufferInventory();
    iface->releaseAll();
}

// =============================================================================
// Thread Safety Tests (Basic)
// =============================================================================

TEST_F(Test__MockGraphBufferManager, ThreadSafety_ConcurrentCallTracking)
{
    const int num_threads = 4;
    const int calls_per_thread = 100;

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t)
    {
        threads.emplace_back([this, calls_per_thread]()
                             {
            for (int i = 0; i < calls_per_thread; ++i) {
                mock_->hasBuffer("layer0", "output");
            } });
    }

    for (auto &t : threads)
    {
        t.join();
    }

    EXPECT_EQ(mock_->hasBuffer_call_count(),
              static_cast<size_t>(num_threads * calls_per_thread));
}

TEST_F(Test__MockGraphBufferManager, ThreadSafety_ConcurrentGetBuffer)
{
    mock_->registerBuffer("layer0", "output", tensor1_.get());

    const int num_threads = 4;
    const int calls_per_thread = 50;

    std::vector<std::thread> threads;
    std::atomic<int> successful_gets{0};

    for (int t = 0; t < num_threads; ++t)
    {
        threads.emplace_back([this, calls_per_thread, &successful_gets]()
                             {
            for (int i = 0; i < calls_per_thread; ++i) {
                auto* buf = mock_->getBuffer("layer0", "output");
                if (buf == tensor1_.get()) {
                    successful_gets.fetch_add(1, std::memory_order_relaxed);
                }
            } });
    }

    for (auto &t : threads)
    {
        t.join();
    }

    EXPECT_EQ(successful_gets.load(), num_threads * calls_per_thread);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(Test__MockGraphBufferManager, EmptyNodeName)
{
    mock_->registerBuffer("", "output", tensor1_.get());

    EXPECT_TRUE(mock_->hasBuffer("", "output"));
    EXPECT_EQ(mock_->getBuffer("", "output"), tensor1_.get());
}

TEST_F(Test__MockGraphBufferManager, EmptyBufferName)
{
    mock_->registerBuffer("layer0", "", tensor1_.get());

    EXPECT_TRUE(mock_->hasBuffer("layer0", ""));
    EXPECT_EQ(mock_->getBuffer("layer0", ""), tensor1_.get());
}

TEST_F(Test__MockGraphBufferManager, OverwriteExistingBuffer)
{
    mock_->registerBuffer("layer0", "output", tensor1_.get());
    EXPECT_EQ(mock_->getBuffer("layer0", "output"), tensor1_.get());

    // Register same key with different tensor
    mock_->registerBuffer("layer0", "output", tensor2_.get());

    EXPECT_EQ(mock_->getBuffer("layer0", "output"), tensor2_.get());
    // Buffer count should still be 1 (overwritten, not added)
    EXPECT_EQ(mock_->bufferCount(), 1);
}

TEST_F(Test__MockGraphBufferManager, UnregisterNonexistentBuffer)
{
    // Should not crash
    mock_->unregisterBuffer("nonexistent", "buffer");
    EXPECT_EQ(mock_->bufferCount(), 0);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
