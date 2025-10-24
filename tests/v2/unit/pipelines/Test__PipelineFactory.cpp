/**
 * @file Test__PipelineFactory.cpp
 * @brief Unit tests for PipelineFactory
 * @author David Sanftenberg
 *
 * Tests:
 * - Singleton instance
 * - Registration (normal, duplicate, null)
 * - Creation (supported, unsupported)
 * - Architecture queries (isSupported, supportedArchitectures)
 * - Qwen2 auto-registration
 */

#include "../../src/v2/pipelines/PipelineFactory.h"
#include "../../src/v2/pipelines/PipelineBase.h"
#include "../../src/v2/pipelines/qwen/Qwen2Pipeline.h"
#include "../../src/v2/utils/MPIContext.h"
#include "../../src/v2/loaders/ModelContext.h"
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <algorithm>

using namespace llaminar2;

// =============================================================================
// Mock Pipeline for Testing
// =============================================================================

/**
 * @brief Mock pipeline for testing factory registration
 */
class MockPipeline : public PipelineBase
{
public:
    MockPipeline(std::shared_ptr<ModelContext> model_ctx,
                 std::shared_ptr<MPIContext> mpi_ctx,
                 int device_idx)
        : PipelineBase(model_ctx, mpi_ctx, device_idx)
    {
        n_layers_ = 12;
        d_model_ = 768;
        vocab_size_ = 50000;
    }

    bool forward(const int *, int) override { return true; }
    const float *logits() const override { return nullptr; }
    const char *architecture() const override { return "mock"; }

    std::string getModelPath() const { return model_path_; }
    int getDeviceIdx() const { return device_idx_; }

protected:
    bool load_weights(const std::string &) override { return true; }
    bool transformer_layer(int, int) override { return true; }
};

// =============================================================================
// Test Fixture
// =============================================================================

class Test__PipelineFactory : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Force Qwen2 registration (static constructors in libraries may not run)
        ensureQwen2Registration();
        
        // Track initial count for tests that check cumulative state
        initial_count_ = PipelineFactory::instance().registeredCount();
    }

    void TearDown() override
    {
        // Note: We can't unregister creators from the singleton
        // Tests must be designed to work with cumulative state
    }

    size_t initial_count_ = 0;
};

// =============================================================================
// Singleton Tests
// =============================================================================

TEST_F(Test__PipelineFactory, SingletonInstance)
{
    // Get instance twice and verify it's the same object
    auto &factory1 = PipelineFactory::instance();
    auto &factory2 = PipelineFactory::instance();

    EXPECT_EQ(&factory1, &factory2);
}

// =============================================================================
// Registration Tests
// =============================================================================

TEST_F(Test__PipelineFactory, RegisterCreator)
{
    auto creator = [](std::shared_ptr<ModelContext> model_ctx,
                      std::shared_ptr<MPIContext> mpi_ctx,
                      int device_idx) -> std::unique_ptr<PipelineBase>
    {
        return std::make_unique<MockPipeline>(model_ctx, mpi_ctx, device_idx);
    };

    PipelineFactory::instance().registerCreator("test_arch", creator);

    EXPECT_TRUE(PipelineFactory::instance().isSupported("test_arch"));
}

TEST_F(Test__PipelineFactory, RegisterDuplicateCreator)
{
    // Register first time
    auto creator1 = [](std::shared_ptr<ModelContext> model_ctx,
                       std::shared_ptr<MPIContext> mpi_ctx,
                       int device_idx) -> std::unique_ptr<PipelineBase>
    {
        return std::make_unique<MockPipeline>(model_ctx, mpi_ctx, device_idx);
    };

    PipelineFactory::instance().registerCreator("test_duplicate", creator1);
    size_t count_after_first = PipelineFactory::instance().registeredCount();

    // Try to register again (should be ignored)
    auto creator2 = [](std::shared_ptr<ModelContext> model_ctx,
                       std::shared_ptr<MPIContext> mpi_ctx,
                       int device_idx) -> std::unique_ptr<PipelineBase>
    {
        return std::make_unique<MockPipeline>(model_ctx, mpi_ctx, device_idx);
    };

    PipelineFactory::instance().registerCreator("test_duplicate", creator2);
    size_t count_after_second = PipelineFactory::instance().registeredCount();

    // Count should not increase
    EXPECT_EQ(count_after_first, count_after_second);
}

TEST_F(Test__PipelineFactory, RegisterNullCreator)
{
    size_t count_before = PipelineFactory::instance().registeredCount();

    // Try to register null creator (should be rejected)
    PipelineFactory::instance().registerCreator("test_null", nullptr);

    size_t count_after = PipelineFactory::instance().registeredCount();

    // Count should not change
    EXPECT_EQ(count_before, count_after);
    EXPECT_FALSE(PipelineFactory::instance().isSupported("test_null"));
}

// =============================================================================
// Creation Tests
// =============================================================================

TEST_F(Test__PipelineFactory, CreateSupportedArchitecture)
{
    // Register a mock architecture
    auto creator = [](std::shared_ptr<ModelContext> model_ctx,
                      std::shared_ptr<MPIContext> mpi_ctx,
                      int device_idx) -> std::unique_ptr<PipelineBase>
    {
        return std::make_unique<MockPipeline>(model_ctx, mpi_ctx, device_idx);
    };

    PipelineFactory::instance().registerCreator("test_create", creator);

    // Create model context (use test-only factory - doesn't actually load)
    auto model_ctx = ModelContext::createForTesting("test.gguf");

    // Create pipeline
    auto pipeline = PipelineFactory::instance().create("test_create", model_ctx, nullptr, -1);

    ASSERT_NE(pipeline, nullptr);
    EXPECT_STREQ(pipeline->architecture(), "mock");
}

TEST_F(Test__PipelineFactory, CreateUnsupportedArchitecture)
{
    // Create a test model context
    auto model_ctx = ModelContext::createForTesting("test.gguf");

    // Try to create pipeline for unsupported architecture
    auto pipeline = PipelineFactory::instance().create("nonexistent", model_ctx, nullptr, -1);

    EXPECT_EQ(pipeline, nullptr);
}

TEST_F(Test__PipelineFactory, CreateWithParameters)
{
    // Register a mock architecture
    auto creator = [](std::shared_ptr<ModelContext> model_ctx,
                      std::shared_ptr<MPIContext> mpi_ctx,
                      int device_idx) -> std::unique_ptr<PipelineBase>
    {
        return std::make_unique<MockPipeline>(model_ctx, mpi_ctx, device_idx);
    };

    PipelineFactory::instance().registerCreator("test_params", creator);

    // Create model context
    std::string test_path = "models/test.gguf";
    auto model_ctx = ModelContext::createForTesting(test_path);
    int test_device = 0;

    auto pipeline = PipelineFactory::instance().create("test_params", model_ctx, nullptr, test_device);

    ASSERT_NE(pipeline, nullptr);

    // Verify parameters were passed correctly
    auto mock = dynamic_cast<MockPipeline *>(pipeline.get());
    ASSERT_NE(mock, nullptr);
    EXPECT_EQ(mock->getModelPath(), model_ctx->path());
    EXPECT_EQ(mock->getDeviceIdx(), test_device);
}

// =============================================================================
// Query Tests
// =============================================================================

TEST_F(Test__PipelineFactory, IsSupported)
{
    // Check a supported architecture (Qwen2 should be auto-registered)
    EXPECT_TRUE(PipelineFactory::instance().isSupported("qwen2"));

    // Check an unsupported architecture
    EXPECT_FALSE(PipelineFactory::instance().isSupported("nonexistent"));
}

TEST_F(Test__PipelineFactory, SupportedArchitectures)
{
    auto architectures = PipelineFactory::instance().supportedArchitectures();

    // Should contain at least Qwen2 (auto-registered)
    EXPECT_FALSE(architectures.empty());

    auto it = std::find(architectures.begin(), architectures.end(), "qwen2");
    EXPECT_NE(it, architectures.end());
}

TEST_F(Test__PipelineFactory, RegisteredCount)
{
    size_t count = PipelineFactory::instance().registeredCount();

    // Should have at least Qwen2 registered
    EXPECT_GE(count, 1);
}

// =============================================================================
// Qwen2 Auto-Registration Test
// =============================================================================

TEST_F(Test__PipelineFactory, Qwen2AutoRegistered)
{
    // Qwen2 should be automatically registered (via ensureQwen2Registration in SetUp)
    EXPECT_TRUE(PipelineFactory::instance().isSupported("qwen2"));

    // Verify it appears in the supported architectures list
    auto architectures = PipelineFactory::instance().supportedArchitectures();
    auto it = std::find(architectures.begin(), architectures.end(), "qwen2");
    EXPECT_NE(it, architectures.end()) << "qwen2 not found in supported architectures";

    // Verify the count includes qwen2
    EXPECT_GE(PipelineFactory::instance().registeredCount(), 1);
    
    // Note: We don't actually create a Qwen2Pipeline here because:
    // 1. It requires a valid GGUF model file
    // 2. We're only testing the factory registration, not pipeline functionality
    // The CreateSupportedArchitecture test already covers the creation path with MockPipeline
}

// =============================================================================
// Integration Test
// =============================================================================

TEST_F(Test__PipelineFactory, FullWorkflow)
{
    // Register a test architecture
    auto creator = [](std::shared_ptr<ModelContext> model_ctx,
                      std::shared_ptr<MPIContext> mpi_ctx,
                      int device_idx) -> std::unique_ptr<PipelineBase>
    {
        return std::make_unique<MockPipeline>(model_ctx, mpi_ctx, device_idx);
    };

    const std::string arch_name = "test_workflow";
    PipelineFactory::instance().registerCreator(arch_name, creator);

    // Verify registration
    EXPECT_TRUE(PipelineFactory::instance().isSupported(arch_name));

    // Verify it appears in supported list
    auto supported = PipelineFactory::instance().supportedArchitectures();
    auto it = std::find(supported.begin(), supported.end(), arch_name);
    EXPECT_NE(it, supported.end());

    // Create model context for testing
    auto model_ctx = ModelContext::createForTesting("test.gguf");

    // Create pipeline
    auto pipeline = PipelineFactory::instance().create(arch_name, model_ctx, nullptr, -1);
    ASSERT_NE(pipeline, nullptr);

    // Verify architecture
    EXPECT_STREQ(pipeline->architecture(), "mock");

    // Verify basic functionality
    EXPECT_TRUE(pipeline->forward(nullptr, 0));
}
