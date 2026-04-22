/**
 * @file Test__MDO_InterfaceDecoupling.cpp
 * @brief Tests that RankOrchestrator works through IInferenceRunner interface
 *
 * Verifies the P0 interface decoupling: MDO stores device_runners_ as
 * vector<unique_ptr<IInferenceRunner>> and accesses device state through
 * primaryDeviceId(), hasLogitsLocal(), getLogitsLocalInfo() instead of
 * DGO-specific inferenceState().
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#include <gtest/gtest.h>

#include "execution/local_execution/orchestrators/RankOrchestrator.h"
#include "execution/local_execution/orchestrators/IInferenceRunner.h"
#include "collective/ILocalTPContext.h"
#include "backends/GlobalDeviceAddress.h"
#include "tensors/Tensors.h"
#include "mocks/MockModelContext.h"
#include "mocks/MockLocalTPContext.h"
#include <atomic>
#include <cstring>
#include <vector>
#include <memory>

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// MockRunner — IInferenceRunner with configurable logits and device
// =============================================================================

class MockRunnerWithLogitsLocal : public IInferenceRunner
{
public:
    struct Config
    {
        DeviceId device = DeviceId::cpu();
        int vocab_size = 32000;
        int vocab_local = 0; // 0 means no column-parallel (replicated)
        bool forward_fails = false;
    };

    explicit MockRunnerWithLogitsLocal(const Config &cfg)
        : config_(cfg), position_(0)
    {
        logits_.resize(static_cast<size_t>(config_.vocab_size), 0.0f);

        if (config_.vocab_local > 0)
        {
            logits_local_ = std::make_shared<FP32Tensor>(
                std::vector<size_t>{1, static_cast<size_t>(config_.vocab_local)},
                config_.device);
        }
    }

    // IInferenceRunner core
    bool forward(const int *tokens, int seq_len) override
    {
        (void)tokens;
        forward_calls_.fetch_add(1, std::memory_order_relaxed);
        if (config_.forward_fails)
            return false;
        position_ += seq_len;
        return true;
    }

    const float *logits() const override { return logits_.data(); }
    int vocab_size() const override { return config_.vocab_size; }
    void clear_cache() override { position_ = 0; }
    int get_position() const override { return position_; }
    ExecutionPath executionPath() const override { return ExecutionPath::GRAPH; }
    const char *architecture() const override { return "mock"; }

    // New P0 interface methods
    DeviceId primaryDeviceId() const override { return config_.device; }
    bool hasLogitsLocal() const override { return logits_local_ != nullptr; }

    LogitsLocalInfo getLogitsLocalInfo() const override
    {
        if (!logits_local_)
            return {};
        const auto &shape = logits_local_->shape();
        return LogitsLocalInfo{
            nullptr,
            std::nullopt,
            shape.size() >= 2 ? shape[1] : 0,
            logits_local_.get()};
    }

    // Test utilities
    void setLogitsLocalData(const std::vector<float> &data)
    {
        if (!logits_local_)
            return;
        float *dst = logits_local_->mutable_data();
        std::memcpy(dst, data.data(), data.size() * sizeof(float));
    }

    void setLogitsData(const std::vector<float> &data)
    {
        logits_ = data;
        config_.vocab_size = static_cast<int>(data.size());
    }

    size_t forward_call_count() const
    {
        return forward_calls_.load(std::memory_order_relaxed);
    }

private:
    Config config_;
    int position_;
    std::vector<float> logits_;
    std::shared_ptr<FP32Tensor> logits_local_;
    mutable std::atomic<size_t> forward_calls_{0};
};

// =============================================================================
// Test Fixture
// =============================================================================

class Test__MDO_InterfaceDecoupling : public ::testing::Test
{
protected:
    std::unique_ptr<RankOrchestrator> createMDO(
        std::vector<std::unique_ptr<IInferenceRunner>> runners,
        std::unique_ptr<ILocalTPContext> tp_ctx = nullptr)
    {
        RankOrchestrator::Config config;
        for (size_t i = 0; i < runners.size(); ++i)
        {
            config.devices.push_back(GlobalDeviceAddress::cpu());
        }

        return RankOrchestrator::createForTest(
            model_ctx_, std::move(runners), std::move(tp_ctx), config);
    }

    std::unique_ptr<MockLocalTPContext> createTPCtx(int n = 2)
    {
        auto ctx = std::make_unique<MockLocalTPContext>();
        std::vector<GlobalDeviceAddress> devs;
        std::vector<float> wts;
        for (int i = 0; i < n; ++i)
        {
            devs.push_back(GlobalDeviceAddress::cpu());
            wts.push_back(1.0f / static_cast<float>(n));
        }
        ctx->setDevices(std::move(devs));
        ctx->setWeights(std::move(wts));
        return ctx;
    }

    std::shared_ptr<IModelContext> model_ctx_ =
        MockModelContext::createMinimal();
};

// =============================================================================
// Interface Contract — primaryDeviceId()
// =============================================================================

TEST_F(Test__MDO_InterfaceDecoupling, PrimaryDeviceIdDefaultsCPU)
{
    MockRunnerWithLogitsLocal::Config cfg;
    auto runner = std::make_unique<MockRunnerWithLogitsLocal>(cfg);
    EXPECT_EQ(runner->primaryDeviceId(), DeviceId::cpu());
}

TEST_F(Test__MDO_InterfaceDecoupling, PrimaryDeviceIdReturnsCUDA)
{
    MockRunnerWithLogitsLocal::Config cfg;
    cfg.device = DeviceId::cuda(0);
    auto runner = std::make_unique<MockRunnerWithLogitsLocal>(cfg);
    EXPECT_EQ(runner->primaryDeviceId(), DeviceId::cuda(0));
}

TEST_F(Test__MDO_InterfaceDecoupling, PrimaryDeviceIdReturnsROCm)
{
    MockRunnerWithLogitsLocal::Config cfg;
    cfg.device = DeviceId::rocm(1);
    auto runner = std::make_unique<MockRunnerWithLogitsLocal>(cfg);
    EXPECT_EQ(runner->primaryDeviceId(), DeviceId::rocm(1));
}

// =============================================================================
// Interface Contract — hasLogitsLocal()
// =============================================================================

TEST_F(Test__MDO_InterfaceDecoupling, HasLogitsLocalFalseByDefault)
{
    MockRunnerWithLogitsLocal::Config cfg;
    auto runner = std::make_unique<MockRunnerWithLogitsLocal>(cfg);
    EXPECT_FALSE(runner->hasLogitsLocal());
}

TEST_F(Test__MDO_InterfaceDecoupling, HasLogitsLocalTrueWhenConfigured)
{
    MockRunnerWithLogitsLocal::Config cfg;
    cfg.vocab_local = 16000;
    auto runner = std::make_unique<MockRunnerWithLogitsLocal>(cfg);
    EXPECT_TRUE(runner->hasLogitsLocal());
}

// =============================================================================
// Interface Contract — getLogitsLocalInfo()
// =============================================================================

TEST_F(Test__MDO_InterfaceDecoupling, LogitsLocalInfoEmptyWhenNoLocal)
{
    MockRunnerWithLogitsLocal::Config cfg;
    auto runner = std::make_unique<MockRunnerWithLogitsLocal>(cfg);
    auto info = runner->getLogitsLocalInfo();
    EXPECT_FALSE(static_cast<bool>(info));
    EXPECT_EQ(info.tensor, nullptr);
    EXPECT_EQ(info.vocab_local, 0u);
}

TEST_F(Test__MDO_InterfaceDecoupling, LogitsLocalInfoValidWhenConfigured)
{
    MockRunnerWithLogitsLocal::Config cfg;
    cfg.vocab_local = 16000;
    auto runner = std::make_unique<MockRunnerWithLogitsLocal>(cfg);

    auto info = runner->getLogitsLocalInfo();
    EXPECT_TRUE(static_cast<bool>(info));
    EXPECT_NE(info.tensor, nullptr);
    EXPECT_EQ(info.vocab_local, 16000u);
    EXPECT_EQ(info.gpu_ptr, nullptr);
}

TEST_F(Test__MDO_InterfaceDecoupling, LogitsLocalInfoBoolConversion)
{
    LogitsLocalInfo empty;
    EXPECT_FALSE(static_cast<bool>(empty));

    FP32Tensor dummy({1, 4}, DeviceId::cpu());
    LogitsLocalInfo valid{nullptr, std::nullopt, 4, &dummy};
    EXPECT_TRUE(static_cast<bool>(valid));
}

// =============================================================================
// MDO Construction with IInferenceRunner
// =============================================================================

TEST_F(Test__MDO_InterfaceDecoupling, CreateForTestAcceptsIInferenceRunners)
{
    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    MockRunnerWithLogitsLocal::Config cfg;
    cfg.vocab_size = 32000;
    runners.push_back(std::make_unique<MockRunnerWithLogitsLocal>(cfg));
    runners.push_back(std::make_unique<MockRunnerWithLogitsLocal>(cfg));

    auto tp = createTPCtx(2);
    auto mdo = createMDO(std::move(runners), std::move(tp));
    ASSERT_NE(mdo, nullptr);
    EXPECT_EQ(mdo->device_count(), 2);
}

TEST_F(Test__MDO_InterfaceDecoupling, CreateForTestWithSingleRunner)
{
    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    MockRunnerWithLogitsLocal::Config cfg;
    cfg.vocab_size = 32000;
    runners.push_back(std::make_unique<MockRunnerWithLogitsLocal>(cfg));

    auto mdo = createMDO(std::move(runners));
    ASSERT_NE(mdo, nullptr);
    EXPECT_EQ(mdo->device_count(), 1);
}

// =============================================================================
// MDO Forward Via IInferenceRunner
// =============================================================================

TEST_F(Test__MDO_InterfaceDecoupling, ForwardDelegatesToRunners)
{
    MockRunnerWithLogitsLocal::Config cfg;
    cfg.vocab_size = 100;

    auto runner0 = std::make_unique<MockRunnerWithLogitsLocal>(cfg);
    auto *raw0 = runner0.get();

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));

    auto mdo = createMDO(std::move(runners));
    ASSERT_NE(mdo, nullptr);

    // forward() delegates to runners, but gatherLogits will fail because
    // combined_logits_ is not allocated by createForTest.
    // We verify the runner WAS called even though overall forward returns false.
    int tokens[] = {42};
    mdo->forward(tokens, 1);
    EXPECT_GE(raw0->forward_call_count(), 1u);
}

TEST_F(Test__MDO_InterfaceDecoupling, ForwardFailsWhenRunnerFails)
{
    MockRunnerWithLogitsLocal::Config cfg;
    cfg.vocab_size = 100;
    cfg.forward_fails = true;

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::make_unique<MockRunnerWithLogitsLocal>(cfg));

    auto mdo = createMDO(std::move(runners));
    ASSERT_NE(mdo, nullptr);

    int tokens[] = {42};
    bool ok = mdo->forward(tokens, 1);
    EXPECT_FALSE(ok);
}

// =============================================================================
// MDO Logits & Vocab Via IInferenceRunner
// =============================================================================

TEST_F(Test__MDO_InterfaceDecoupling, LogitsFromPrimaryRunner)
{
    MockRunnerWithLogitsLocal::Config cfg;
    cfg.vocab_size = 5;

    auto runner = std::make_unique<MockRunnerWithLogitsLocal>(cfg);
    runner->setLogitsData({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner));

    auto mdo = createMDO(std::move(runners));
    ASSERT_NE(mdo, nullptr);

    const float *logits = mdo->logits();
    ASSERT_NE(logits, nullptr);
    EXPECT_FLOAT_EQ(logits[0], 1.0f);
    EXPECT_FLOAT_EQ(logits[4], 5.0f);
}

TEST_F(Test__MDO_InterfaceDecoupling, VocabSizeFromModelContext)
{
    // MDO reads vocab_size from model_ctx_, not from runners.
    // MINIMAL preset has vocab_size=1000.
    MockRunnerWithLogitsLocal::Config cfg;
    cfg.vocab_size = 151936; // Runner's value is ignored by MDO

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::make_unique<MockRunnerWithLogitsLocal>(cfg));

    auto mdo = createMDO(std::move(runners));
    ASSERT_NE(mdo, nullptr);
    EXPECT_EQ(mdo->vocab_size(), 1000); // From MockModelContext::MINIMAL
}

// =============================================================================
// MDO ClearCache Via IInferenceRunner
// =============================================================================

TEST_F(Test__MDO_InterfaceDecoupling, ClearCacheDelegatesToAllRunners)
{
    MockRunnerWithLogitsLocal::Config cfg;
    cfg.vocab_size = 100;

    auto runner0 = std::make_unique<MockRunnerWithLogitsLocal>(cfg);
    auto runner1 = std::make_unique<MockRunnerWithLogitsLocal>(cfg);

    int tokens[] = {42};
    runner0->forward(tokens, 5);
    runner1->forward(tokens, 3);
    EXPECT_EQ(runner0->get_position(), 5);
    EXPECT_EQ(runner1->get_position(), 3);

    auto *raw0 = runner0.get();
    auto *raw1 = runner1.get();

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto tp = createTPCtx(2);
    auto mdo = createMDO(std::move(runners), std::move(tp));
    ASSERT_NE(mdo, nullptr);

    mdo->clear_cache();
    EXPECT_EQ(raw0->get_position(), 0);
    EXPECT_EQ(raw1->get_position(), 0);
}

// =============================================================================
// Mixed Device Types Through Interface
// =============================================================================

TEST_F(Test__MDO_InterfaceDecoupling, MixedDeviceTypeRunnersConstruct)
{
    MockRunnerWithLogitsLocal::Config cfg0;
    cfg0.device = DeviceId::cuda(0);
    cfg0.vocab_size = 32000;

    MockRunnerWithLogitsLocal::Config cfg1;
    cfg1.device = DeviceId::rocm(0);
    cfg1.vocab_size = 32000;

    auto runner0 = std::make_unique<MockRunnerWithLogitsLocal>(cfg0);
    auto runner1 = std::make_unique<MockRunnerWithLogitsLocal>(cfg1);

    EXPECT_EQ(runner0->primaryDeviceId(), DeviceId::cuda(0));
    EXPECT_EQ(runner1->primaryDeviceId(), DeviceId::rocm(0));

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto tp = createTPCtx(2);
    auto mdo = createMDO(std::move(runners), std::move(tp));
    ASSERT_NE(mdo, nullptr);
    EXPECT_EQ(mdo->device_count(), 2);
}

// =============================================================================
// Architecture delegated through IInferenceRunner
// =============================================================================

TEST_F(Test__MDO_InterfaceDecoupling, ArchitectureFromPrimaryRunner)
{
    MockRunnerWithLogitsLocal::Config cfg;
    cfg.vocab_size = 100;

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::make_unique<MockRunnerWithLogitsLocal>(cfg));

    auto mdo = createMDO(std::move(runners));
    ASSERT_NE(mdo, nullptr);
    EXPECT_STREQ(mdo->architecture(), "mock");
}

// =============================================================================
// Device Runner Access Via IRankOrchestrator
// =============================================================================

TEST_F(Test__MDO_InterfaceDecoupling, DeviceRunnerReturnsIInferenceRunner)
{
    MockRunnerWithLogitsLocal::Config cfg;
    cfg.vocab_size = 100;

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::make_unique<MockRunnerWithLogitsLocal>(cfg));
    runners.push_back(std::make_unique<MockRunnerWithLogitsLocal>(cfg));

    auto tp = createTPCtx(2);
    auto mdo = createMDO(std::move(runners), std::move(tp));
    ASSERT_NE(mdo, nullptr);

    IInferenceRunner *r0 = mdo->deviceRunner(0);
    IInferenceRunner *r1 = mdo->deviceRunner(1);
    ASSERT_NE(r0, nullptr);
    ASSERT_NE(r1, nullptr);
    EXPECT_EQ(r0->vocab_size(), 100);
    EXPECT_EQ(r1->vocab_size(), 100);
}

// =============================================================================
// Logits Local Detection Through Interface
// =============================================================================

TEST_F(Test__MDO_InterfaceDecoupling, RunnersWithoutLogitsLocalAreReplicated)
{
    MockRunnerWithLogitsLocal::Config cfg;
    cfg.vocab_size = 4;
    cfg.vocab_local = 0;

    auto runner0 = std::make_unique<MockRunnerWithLogitsLocal>(cfg);
    auto runner1 = std::make_unique<MockRunnerWithLogitsLocal>(cfg);

    EXPECT_FALSE(runner0->hasLogitsLocal());
    EXPECT_FALSE(runner1->hasLogitsLocal());

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto tp = createTPCtx(2);
    auto mdo = createMDO(std::move(runners), std::move(tp));
    ASSERT_NE(mdo, nullptr);
}

TEST_F(Test__MDO_InterfaceDecoupling, RunnersWithLogitsLocalAreColumnParallel)
{
    MockRunnerWithLogitsLocal::Config cfg;
    cfg.vocab_size = 4;
    cfg.vocab_local = 2;

    auto runner0 = std::make_unique<MockRunnerWithLogitsLocal>(cfg);
    auto runner1 = std::make_unique<MockRunnerWithLogitsLocal>(cfg);

    EXPECT_TRUE(runner0->hasLogitsLocal());
    EXPECT_TRUE(runner1->hasLogitsLocal());

    auto info0 = runner0->getLogitsLocalInfo();
    auto info1 = runner1->getLogitsLocalInfo();
    EXPECT_EQ(info0.vocab_local, 2u);
    EXPECT_EQ(info1.vocab_local, 2u);

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto tp = createTPCtx(2);
    auto mdo = createMDO(std::move(runners), std::move(tp));
    ASSERT_NE(mdo, nullptr);
}
