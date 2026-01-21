/**
 * @file Test__MultiDomainOrchestrator.cpp
 * @brief Unit tests for MultiDomainOrchestrator
 * @author David Sanftenberg
 * @date January 2026
 *
 * Tests the MultiDomainOrchestrator wrapper class without MPI dependencies.
 * Uses mock/stub implementations for unit testing in isolation.
 */

#include <gtest/gtest.h>
#include "execution/MultiDomainOrchestrator.h"
#include "execution/IInferenceRunner.h"
#include "config/TPDomain.h"
#include "backends/DeviceId.h"

using namespace llaminar2;

// =============================================================================
// Mock GraphOrchestrator for Unit Testing
// =============================================================================

/**
 * @brief Minimal mock of GraphOrchestrator for testing MultiDomainOrchestrator
 *
 * This mock allows testing the wrapper without loading actual models.
 * It tracks method calls and provides configurable return values.
 */
class MockGraphOrchestrator : public IInferenceRunner
{
public:
    MockGraphOrchestrator()
        : ready_(true), vocab_size_(32000), position_(0)
    {
        // Initialize with dummy logits
        logits_.resize(vocab_size_, 0.0f);
    }

    // IInferenceRunner interface
    bool forward(const int *tokens, int seq_len) override
    {
        (void)tokens;
        forward_call_count_++;
        position_ += seq_len;
        return forward_return_value_;
    }

    const float *logits() const override
    {
        return logits_.data();
    }

    int vocab_size() const override
    {
        return vocab_size_;
    }

    void clear_cache() override
    {
        clear_cache_call_count_++;
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
        return "mock_qwen2";
    }

    // Test configuration
    void setReady(bool ready) { ready_ = ready; }
    void setForwardReturnValue(bool value) { forward_return_value_ = value; }
    void setVocabSize(int size)
    {
        vocab_size_ = size;
        logits_.resize(size, 0.0f);
    }

    // Call tracking
    int getForwardCallCount() const { return forward_call_count_; }
    int getClearCacheCallCount() const { return clear_cache_call_count_; }

private:
    bool ready_;
    bool forward_return_value_ = true;
    int vocab_size_;
    int position_;
    int forward_call_count_ = 0;
    int clear_cache_call_count_ = 0;
    std::vector<float> logits_;
};

// =============================================================================
// Factory Method Tests
// =============================================================================

TEST(Test__MultiDomainOrchestrator, CreateForTestReturnsValid)
{
    // Create mock inner orchestrator
    auto mock_inner = std::make_unique<MockGraphOrchestrator>();

    // Create TP config for testing (no MPI)
    auto tp_config = std::make_unique<MultiDomainTPConfig>();

    // Create wrapper with injected dependencies
    auto orchestrator = MultiDomainOrchestrator::createForTest(
        std::move(mock_inner),
        std::move(tp_config));

    ASSERT_NE(orchestrator, nullptr);
    EXPECT_TRUE(orchestrator->isReady());
}

TEST(Test__MultiDomainOrchestrator, CreateForTestWithNullInnerFails)
{
    // Null inner orchestrator should fail
    auto tp_config = std::make_unique<MultiDomainTPConfig>();

    auto orchestrator = MultiDomainOrchestrator::createForTest(
        nullptr,
        std::move(tp_config));

    EXPECT_EQ(orchestrator, nullptr);
}

TEST(Test__MultiDomainOrchestrator, CreateForTestWithNullTPConfigSucceeds)
{
    // Null TP config is allowed (single-domain operation)
    auto mock_inner = std::make_unique<MockGraphOrchestrator>();

    auto orchestrator = MultiDomainOrchestrator::createForTest(
        std::move(mock_inner),
        nullptr); // Null TP config

    ASSERT_NE(orchestrator, nullptr);
    EXPECT_TRUE(orchestrator->isReady());
    EXPECT_EQ(orchestrator->getTPConfig(), nullptr);
}

// =============================================================================
// Domain Access Tests
// =============================================================================

TEST(Test__MultiDomainOrchestrator, GetTPConfigReturnsConfig)
{
    auto mock_inner = std::make_unique<MockGraphOrchestrator>();
    auto tp_config = std::make_unique<MultiDomainTPConfig>();
    MultiDomainTPConfig *config_ptr = tp_config.get();

    auto orchestrator = MultiDomainOrchestrator::createForTest(
        std::move(mock_inner),
        std::move(tp_config));

    ASSERT_NE(orchestrator, nullptr);
    EXPECT_EQ(orchestrator->getTPConfig(), config_ptr);
}

TEST(Test__MultiDomainOrchestrator, GetGPUDomainReturnsCorrectDomain)
{
    auto mock_inner = std::make_unique<MockGraphOrchestrator>();

    // Create TP config with a GPU domain
    std::vector<TPDomain> domains;
    TPDomain gpu_domain;
    gpu_domain.type = TPDomainType::GPU_INTRA_RANK;
    gpu_domain.domain_size = 2;
    gpu_domain.local_rank_in_domain = 0;
    gpu_domain.name = "test_gpu_domain";
    gpu_domain.devices = {DeviceId::cuda(0), DeviceId::rocm(0)};
    domains.push_back(gpu_domain);

    auto tp_config = std::make_unique<MultiDomainTPConfig>(
        MultiDomainTPConfig::createForTest(domains));

    auto orchestrator = MultiDomainOrchestrator::createForTest(
        std::move(mock_inner),
        std::move(tp_config));

    ASSERT_NE(orchestrator, nullptr);
    const TPDomain *retrieved_domain = orchestrator->getGPUDomain();
    ASSERT_NE(retrieved_domain, nullptr);
    EXPECT_EQ(retrieved_domain->type, TPDomainType::GPU_INTRA_RANK);
    EXPECT_EQ(retrieved_domain->domain_size, 2);
    EXPECT_EQ(retrieved_domain->name, "test_gpu_domain");
}

TEST(Test__MultiDomainOrchestrator, GetCPUDomainReturnsCorrectDomain)
{
    auto mock_inner = std::make_unique<MockGraphOrchestrator>();

    // Create TP config with a CPU domain
    std::vector<TPDomain> domains;
    TPDomain cpu_domain;
    cpu_domain.type = TPDomainType::CPU_CROSS_RANK;
    cpu_domain.domain_size = 4;
    cpu_domain.local_rank_in_domain = 1;
    cpu_domain.name = "test_cpu_domain";
    cpu_domain.devices = {DeviceId::cpu(), DeviceId::cpu(), DeviceId::cpu(), DeviceId::cpu()};
    domains.push_back(cpu_domain);

    auto tp_config = std::make_unique<MultiDomainTPConfig>(
        MultiDomainTPConfig::createForTest(domains));

    auto orchestrator = MultiDomainOrchestrator::createForTest(
        std::move(mock_inner),
        std::move(tp_config));

    ASSERT_NE(orchestrator, nullptr);
    const TPDomain *retrieved_domain = orchestrator->getCPUDomain();
    ASSERT_NE(retrieved_domain, nullptr);
    EXPECT_EQ(retrieved_domain->type, TPDomainType::CPU_CROSS_RANK);
    EXPECT_EQ(retrieved_domain->domain_size, 4);
    EXPECT_EQ(retrieved_domain->name, "test_cpu_domain");
}

TEST(Test__MultiDomainOrchestrator, GetGPUDomainReturnsNullWhenNoGPUDomain)
{
    auto mock_inner = std::make_unique<MockGraphOrchestrator>();

    // Create TP config with only CPU domain
    std::vector<TPDomain> domains;
    TPDomain cpu_domain;
    cpu_domain.type = TPDomainType::CPU_CROSS_RANK;
    cpu_domain.domain_size = 2;
    cpu_domain.name = "cpu_only";
    cpu_domain.devices = {DeviceId::cpu(), DeviceId::cpu()};
    domains.push_back(cpu_domain);

    auto tp_config = std::make_unique<MultiDomainTPConfig>(
        MultiDomainTPConfig::createForTest(domains));

    auto orchestrator = MultiDomainOrchestrator::createForTest(
        std::move(mock_inner),
        std::move(tp_config));

    ASSERT_NE(orchestrator, nullptr);
    EXPECT_EQ(orchestrator->getGPUDomain(), nullptr);
    EXPECT_NE(orchestrator->getCPUDomain(), nullptr);
}

// =============================================================================
// Delegation Tests
// =============================================================================

TEST(Test__MultiDomainOrchestrator, IsReadyDelegatesToInner)
{
    auto mock_inner = std::make_unique<MockGraphOrchestrator>();
    mock_inner->setReady(true);

    auto orchestrator = MultiDomainOrchestrator::createForTest(
        std::move(mock_inner),
        nullptr);

    ASSERT_NE(orchestrator, nullptr);
    EXPECT_TRUE(orchestrator->isReady());
}

TEST(Test__MultiDomainOrchestrator, ResetDelegatesToInner)
{
    auto mock_inner = std::make_unique<MockGraphOrchestrator>();
    MockGraphOrchestrator *mock_ptr = mock_inner.get();

    // Simulate some position advancement
    int tokens[] = {1, 2, 3, 4, 5};
    mock_inner->forward(tokens, 5);
    EXPECT_EQ(mock_ptr->get_position(), 5);

    auto orchestrator = MultiDomainOrchestrator::createForTest(
        std::move(mock_inner),
        nullptr);

    ASSERT_NE(orchestrator, nullptr);

    // Clear cache via wrapper
    orchestrator->clear_cache();

    // Verify delegation
    EXPECT_EQ(mock_ptr->getClearCacheCallCount(), 1);
    EXPECT_EQ(orchestrator->get_position(), 0);
}

TEST(Test__MultiDomainOrchestrator, ForwardDelegatesToInner)
{
    auto mock_inner = std::make_unique<MockGraphOrchestrator>();
    MockGraphOrchestrator *mock_ptr = mock_inner.get();

    auto orchestrator = MultiDomainOrchestrator::createForTest(
        std::move(mock_inner),
        nullptr);

    ASSERT_NE(orchestrator, nullptr);

    int tokens[] = {1, 2, 3};
    bool result = orchestrator->forward(tokens, 3);

    EXPECT_TRUE(result);
    EXPECT_EQ(mock_ptr->getForwardCallCount(), 1);
    EXPECT_EQ(orchestrator->get_position(), 3);
}

TEST(Test__MultiDomainOrchestrator, VocabSizeDelegatesToInner)
{
    auto mock_inner = std::make_unique<MockGraphOrchestrator>();
    mock_inner->setVocabSize(65536);

    auto orchestrator = MultiDomainOrchestrator::createForTest(
        std::move(mock_inner),
        nullptr);

    ASSERT_NE(orchestrator, nullptr);
    EXPECT_EQ(orchestrator->vocab_size(), 65536);
}

TEST(Test__MultiDomainOrchestrator, LogitsDelegatesToInner)
{
    auto mock_inner = std::make_unique<MockGraphOrchestrator>();

    auto orchestrator = MultiDomainOrchestrator::createForTest(
        std::move(mock_inner),
        nullptr);

    ASSERT_NE(orchestrator, nullptr);
    EXPECT_NE(orchestrator->logits(), nullptr);
}

TEST(Test__MultiDomainOrchestrator, ArchitectureDelegatesToInner)
{
    auto mock_inner = std::make_unique<MockGraphOrchestrator>();

    auto orchestrator = MultiDomainOrchestrator::createForTest(
        std::move(mock_inner),
        nullptr);

    ASSERT_NE(orchestrator, nullptr);
    EXPECT_STREQ(orchestrator->architecture(), "mock_qwen2");
}

TEST(Test__MultiDomainOrchestrator, ExecutionPathReturnsGraph)
{
    auto mock_inner = std::make_unique<MockGraphOrchestrator>();

    auto orchestrator = MultiDomainOrchestrator::createForTest(
        std::move(mock_inner),
        nullptr);

    ASSERT_NE(orchestrator, nullptr);
    EXPECT_EQ(orchestrator->executionPath(), ExecutionPath::GRAPH);
}

// =============================================================================
// Statistics Tests
// =============================================================================

TEST(Test__MultiDomainOrchestrator, DomainStatsInitiallyZero)
{
    auto mock_inner = std::make_unique<MockGraphOrchestrator>();

    auto orchestrator = MultiDomainOrchestrator::createForTest(
        std::move(mock_inner),
        nullptr);

    ASSERT_NE(orchestrator, nullptr);

    auto stats = orchestrator->getDomainStats();
    EXPECT_EQ(stats.gpu_domain_collective_count, 0);
    EXPECT_EQ(stats.cpu_domain_collective_count, 0);
}

TEST(Test__MultiDomainOrchestrator, ResetStatsClearsStats)
{
    auto mock_inner = std::make_unique<MockGraphOrchestrator>();

    auto orchestrator = MultiDomainOrchestrator::createForTest(
        std::move(mock_inner),
        nullptr);

    ASSERT_NE(orchestrator, nullptr);

    // Stats start at zero
    auto stats_before = orchestrator->getDomainStats();
    EXPECT_EQ(stats_before.gpu_domain_collective_count, 0);

    // Reset stats (still zero, but verifies method works)
    orchestrator->resetStats();

    auto stats_after = orchestrator->getDomainStats();
    EXPECT_EQ(stats_after.gpu_domain_collective_count, 0);
    EXPECT_EQ(stats_after.cpu_domain_collective_count, 0);
}

// =============================================================================
// Model Info Tests
// =============================================================================

TEST(Test__MultiDomainOrchestrator, GetModelInfoReturnsDescription)
{
    auto mock_inner = std::make_unique<MockGraphOrchestrator>();

    auto orchestrator = MultiDomainOrchestrator::createForTest(
        std::move(mock_inner),
        nullptr);

    ASSERT_NE(orchestrator, nullptr);

    std::string info = orchestrator->getModelInfo();
    EXPECT_FALSE(info.empty());
    EXPECT_NE(info.find("MultiDomainOrchestrator"), std::string::npos);
}

TEST(Test__MultiDomainOrchestrator, GetInnerRunnerReturnsInner)
{
    auto mock_inner = std::make_unique<MockGraphOrchestrator>();
    MockGraphOrchestrator *mock_ptr = mock_inner.get();

    auto orchestrator = MultiDomainOrchestrator::createForTest(
        std::move(mock_inner),
        nullptr);

    ASSERT_NE(orchestrator, nullptr);

    // getInnerRunner returns the IInferenceRunner interface
    IInferenceRunner *inner = orchestrator->getInnerRunner();
    EXPECT_EQ(inner, mock_ptr);

    // getInnerOrchestrator returns nullptr for mocks (not actual GraphOrchestrator)
    GraphOrchestrator *inner_orch = orchestrator->getInnerOrchestrator();
    EXPECT_EQ(inner_orch, nullptr);
}
