/**
 * @file Test__DeviceSampler.cpp
 * @brief Unit tests for DeviceSampler extracted from RankOrchestrator
 *
 * Tests the CPU-only fallback paths and edge cases. GPU-side sampling
 * (argmax, topK) is tested via integration tests.
 */

#include <gtest/gtest.h>

#include "execution/local_execution/orchestrators/DeviceSampler.h"
#include "execution/local_execution/orchestrators/IInferenceRunner.h"
#include "tensors/Tensors.h"
#include "utils/Sampler.h"
#include <array>
#include <cstring>
#include <memory>
#include <optional>
#include <vector>

using namespace llaminar2;

// =============================================================================
// Minimal IInferenceRunner mock for DeviceSampler tests
// =============================================================================

class DeviceSamplerMockRunner : public IInferenceRunner
{
public:
    explicit DeviceSamplerMockRunner(int vocab = 32000, bool has_logits_local = false)
        : vocab_(vocab), has_logits_local_(has_logits_local)
    {
        logits_.resize(static_cast<size_t>(vocab), 0.0f);
    }

    bool forward(const int *, int seq_len) override
    {
        position_ += seq_len;
        return true;
    }
    const float *logits() const override { return logits_.data(); }
    int vocab_size() const override { return vocab_; }
    void clear_cache() override { position_ = 0; }
    int get_position() const override { return position_; }
    ExecutionPath executionPath() const override { return ExecutionPath::GRAPH; }
    const char *architecture() const override { return "mock"; }

    // Batch stubs
    bool forward_batch(const std::vector<std::vector<int>> &) override { return false; }
    const float *getLogits(int) const override { return logits_.data(); }
    int batch_size() const override { return 1; }
    int padded_seq_len() const override { return 0; }
    const std::vector<int> &sequence_lengths() const override { return seq_lens_; }

    // Timeline/snapshot stubs
    void setSuppressTimeline(bool) override {}
    void setAccumulatePrefill(bool) override {}
    void flushStageTimeline() override {}
    void enableSnapshotCapture(const std::string &) override {}
    void disableSnapshotCapture() override {}
    void clearSnapshots() override {}
    std::vector<std::string> getSnapshotKeys() const override { return {}; }
    const float *getSnapshot(const std::string &, size_t &out_size) const override
    {
        out_size = 0;
        return nullptr;
    }
    SnapshotInfo getSnapshotWithShape(const std::string &) const override { return {}; }
    DeviceId primaryDeviceId() const override { return DeviceId::cpu(); }
    const GraphExecutorStats *executorStats() const override { return nullptr; }
    void resetExecutorStats() override {}

    // Override hasLogitsLocal for test control
    bool hasLogitsLocal() const override { return has_logits_local_; }

private:
    int vocab_;
    int position_ = 0;
    bool has_logits_local_;
    std::vector<float> logits_;
    std::vector<int> seq_lens_;
};

// =============================================================================
// Tests
// =============================================================================

class Test__DeviceSampler : public ::testing::Test
{
protected:
    std::vector<std::unique_ptr<IInferenceRunner>> makeSingleRunner()
    {
        std::vector<std::unique_ptr<IInferenceRunner>> runners;
        runners.push_back(std::make_unique<DeviceSamplerMockRunner>());
        return runners;
    }

    std::vector<std::unique_ptr<IInferenceRunner>> makeMultiRunner(int n, bool with_logits_local = false)
    {
        std::vector<std::unique_ptr<IInferenceRunner>> runners;
        for (int i = 0; i < n; ++i)
            runners.push_back(std::make_unique<DeviceSamplerMockRunner>(32000, with_logits_local));
        return runners;
    }
};

// =============================================================================
// 1. sampleGreedy edge cases
// =============================================================================

TEST_F(Test__DeviceSampler, SampleGreedy_EmptyRunners_ReturnsNeg1)
{
    std::vector<std::unique_ptr<IInferenceRunner>> empty;
    EXPECT_EQ(DeviceSampler::sampleGreedy(empty), -1);
}

TEST_F(Test__DeviceSampler, SampleGreedy_SingleRunner_ReturnsNeg1)
{
    // Single runner is not supported (caller uses host-side sampling)
    auto runners = makeSingleRunner();
    EXPECT_EQ(DeviceSampler::sampleGreedy(runners), -1);
}

TEST_F(Test__DeviceSampler, SampleGreedy_MultiRunnerNoLogitsLocal_ReturnsNeg1)
{
    // Multi-runner but no logits_local → unsupported
    auto runners = makeMultiRunner(2, false);
    EXPECT_EQ(DeviceSampler::sampleGreedy(runners), -1);
}

TEST_F(Test__DeviceSampler, SampleGreedy_MultiRunnerWithLogitsLocal_ReturnsNeg1_NoCPUBackend)
{
    // Multi-runner with logits_local but CPU device (no GPU backend) → -1
    auto runners = makeMultiRunner(2, true);
    // The implementation requires GPU pointers; CPU runners return null gpu_ptr
    EXPECT_EQ(DeviceSampler::sampleGreedy(runners), -1);
}

TEST_F(Test__DeviceSampler, SampleGreedyRowsFromLocalInfos_PreservesCrossShardTieBreak)
{
    auto shard0 = std::make_shared<FP32Tensor>(
        std::vector<size_t>{3, 2},
        DeviceId::cpu());
    auto shard1 = std::make_shared<FP32Tensor>(
        std::vector<size_t>{3, 3},
        DeviceId::cpu());

    const std::vector<float> shard0_data = {
        0.0f, 1.0f,
        4.0f, 10.0f,
        7.0f, 1.0f};
    const std::vector<float> shard1_data = {
        2.0f, 3.0f, 4.0f,
        10.0f, 3.0f, 2.0f,
        0.0f, 7.0f, 8.0f};
    std::memcpy(
        shard0->mutable_data(),
        shard0_data.data(),
        shard0_data.size() * sizeof(float));
    std::memcpy(
        shard1->mutable_data(),
        shard1_data.data(),
        shard1_data.size() * sizeof(float));

    std::vector<LogitsLocalInfo> infos;
    infos.push_back(LogitsLocalInfo{
        nullptr,
        std::nullopt,
        2,
        0,
        shard0.get(),
        nullptr,
        nullptr,
        nullptr,
        0});
    infos.push_back(LogitsLocalInfo{
        nullptr,
        std::nullopt,
        3,
        0,
        shard1.get(),
        nullptr,
        nullptr,
        nullptr,
        0});

    std::array<int32_t, 2> tokens = {-1, -1};
    ASSERT_TRUE(DeviceSampler::sampleGreedyRowsFromLocalInfos(
        infos,
        /*start_row=*/1,
        static_cast<int>(tokens.size()),
        tokens.data()));

    EXPECT_EQ(tokens[0], 1)
        << "row 1 ties at value 10 across shards, so the lower global token wins";
    EXPECT_EQ(tokens[1], 4)
        << "row 2 should pick shard 1 local column 2 with shard offset";
}

TEST_F(Test__DeviceSampler, SampleGreedyRowsFromLocalInfos_UsesExplicitVocabOffsets)
{
    auto high_offset_shard = std::make_shared<FP32Tensor>(
        std::vector<size_t>{2, 2},
        DeviceId::cpu());
    auto zero_offset_shard = std::make_shared<FP32Tensor>(
        std::vector<size_t>{2, 3},
        DeviceId::cpu());

    /*
     * The high-offset shard is intentionally listed first.  Production
     * LocalTP runners are not allowed to rely on participant vector order for
     * token ids; the graph's TP config owns the vocabulary range.
     */
    const std::vector<float> high_offset_data = {
        1.0f, 9.0f,
        10.0f, 2.0f};
    const std::vector<float> zero_offset_data = {
        0.0f, 10.0f, 3.0f,
        1.0f, 2.0f, 8.0f};
    std::memcpy(
        high_offset_shard->mutable_data(),
        high_offset_data.data(),
        high_offset_data.size() * sizeof(float));
    std::memcpy(
        zero_offset_shard->mutable_data(),
        zero_offset_data.data(),
        zero_offset_data.size() * sizeof(float));

    std::vector<LogitsLocalInfo> infos;
    infos.push_back(LogitsLocalInfo{
        nullptr,
        std::nullopt,
        2,
        5,
        high_offset_shard.get(),
        nullptr,
        nullptr,
        nullptr,
        0});
    infos.push_back(LogitsLocalInfo{
        nullptr,
        std::nullopt,
        3,
        0,
        zero_offset_shard.get(),
        nullptr,
        nullptr,
        nullptr,
        0});

    std::array<int32_t, 2> tokens = {-1, -1};
    ASSERT_TRUE(DeviceSampler::sampleGreedyRowsFromLocalInfos(
        infos,
        /*start_row=*/0,
        static_cast<int>(tokens.size()),
        tokens.data()));

    EXPECT_EQ(tokens[0], 1)
        << "row 0 should pick the zero-offset shard despite it being second";
    EXPECT_EQ(tokens[1], 5)
        << "row 1 should preserve the first shard's explicit vocab offset";
}

// =============================================================================
// 2. sample() edge cases
// =============================================================================

TEST_F(Test__DeviceSampler, Sample_Greedy_DelegatesToSampleGreedy)
{
    auto runners = makeSingleRunner();
    SamplingParams params;
    params.temperature = 0.0f; // is_greedy() should be true
    // Single runner → sampleGreedy returns -1
    EXPECT_EQ(DeviceSampler::sample(runners, params), -1);
}

TEST_F(Test__DeviceSampler, Sample_EmptyRunners_ReturnsNeg1)
{
    std::vector<std::unique_ptr<IInferenceRunner>> empty;
    SamplingParams params;
    params.temperature = 0.8f;
    params.top_k = 40;
    EXPECT_EQ(DeviceSampler::sample(empty, params), -1);
}

TEST_F(Test__DeviceSampler, Sample_SingleRunner_ReturnsNeg1)
{
    auto runners = makeSingleRunner();
    SamplingParams params;
    params.temperature = 0.8f;
    params.top_k = 40;
    EXPECT_EQ(DeviceSampler::sample(runners, params), -1);
}

TEST_F(Test__DeviceSampler, Sample_MultiRunnerNoLogitsLocal_ReturnsNeg1)
{
    auto runners = makeMultiRunner(2, false);
    SamplingParams params;
    params.temperature = 0.8f;
    params.top_k = 40;
    EXPECT_EQ(DeviceSampler::sample(runners, params), -1);
}
