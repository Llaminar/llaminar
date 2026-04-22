/**
 * @file Test__LogitsGatherer.cpp
 * @brief Unit tests for LogitsGatherer extracted from RankOrchestrator
 *
 * Tests buffer allocation, skip-gather control, needsGather() logic,
 * single-device gather path, and copyFromStage delegation.
 */

#include <gtest/gtest.h>

#include "execution/local_execution/orchestrators/LogitsGatherer.h"
#include "execution/local_execution/orchestrators/IInferenceRunner.h"
#include "tensors/Tensors.h"
#include <cstring>
#include <memory>
#include <vector>

using namespace llaminar2;

// =============================================================================
// Minimal IInferenceRunner mock for LogitsGatherer tests
// =============================================================================

class LogitsGathererMockRunner : public IInferenceRunner
{
public:
    explicit LogitsGathererMockRunner(int vocab = 32000)
        : vocab_(vocab)
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

    // --- Helpers for test control ---
    void setLogitsData(const std::vector<float> &data)
    {
        logits_ = data;
        vocab_ = static_cast<int>(data.size());
    }

    void fillLogits(float value)
    {
        std::fill(logits_.begin(), logits_.end(), value);
    }

private:
    int vocab_;
    int position_ = 0;
    std::vector<float> logits_;
    std::vector<int> seq_lens_;
};

// =============================================================================
// Test Fixture
// =============================================================================

class Test__LogitsGatherer : public ::testing::Test
{
protected:
    static constexpr int VOCAB = 128;
    static constexpr size_t MAX_TOKENS = 16;

    std::unique_ptr<LogitsGatherer> createGatherer(int vocab = VOCAB, size_t max_tokens = MAX_TOKENS)
    {
        return std::make_unique<LogitsGatherer>(vocab, max_tokens);
    }

    std::vector<std::unique_ptr<IInferenceRunner>> makeSingleRunner(int vocab = VOCAB)
    {
        std::vector<std::unique_ptr<IInferenceRunner>> runners;
        auto runner = std::make_unique<LogitsGathererMockRunner>(vocab);
        runners.push_back(std::move(runner));
        return runners;
    }
};

// =============================================================================
// 1. Buffer Allocation
// =============================================================================

TEST_F(Test__LogitsGatherer, ConstructorAllocatesBuffer)
{
    auto g = createGatherer();
    EXPECT_TRUE(g->isAllocated());
    EXPECT_EQ(g->bufferNumel(), static_cast<size_t>(VOCAB) * MAX_TOKENS);
}

TEST_F(Test__LogitsGatherer, ZeroVocabProducesEmptyGatherer)
{
    auto g = createGatherer(0, 0);
    // Buffer may or may not be allocated for zero size, but shouldn't crash
    EXPECT_EQ(g->lastGatheredSize(), 0u);
}

TEST_F(Test__LogitsGatherer, DataAccessible)
{
    auto g = createGatherer();
    ASSERT_NE(g->data(), nullptr);
    ASSERT_NE(g->mutableData(), nullptr);
    EXPECT_EQ(g->data(), g->mutableData());
}

// =============================================================================
// 2. Skip-Gather Control
// =============================================================================

TEST_F(Test__LogitsGatherer, SkipDecodeDefault)
{
    auto g = createGatherer();
    EXPECT_FALSE(g->skipDecode());
    EXPECT_FALSE(g->skipPrefill());
}

TEST_F(Test__LogitsGatherer, SetSkipDecode)
{
    auto g = createGatherer();
    g->setSkipDecode(true);
    EXPECT_TRUE(g->skipDecode());
    g->setSkipDecode(false);
    EXPECT_FALSE(g->skipDecode());
}

TEST_F(Test__LogitsGatherer, SetSkipPrefill)
{
    auto g = createGatherer();
    g->setSkipPrefill(true);
    EXPECT_TRUE(g->skipPrefill());
    g->setSkipPrefill(false);
    EXPECT_FALSE(g->skipPrefill());
}

// =============================================================================
// 3. needsGather() Logic
// =============================================================================

TEST_F(Test__LogitsGatherer, NeedsGatherDecode_DefaultTrue)
{
    auto g = createGatherer();
    EXPECT_TRUE(g->needsGather(1)); // decode: seq_len=1
}

TEST_F(Test__LogitsGatherer, NeedsGatherDecode_SkipSetFalse)
{
    auto g = createGatherer();
    g->setSkipDecode(true);
    EXPECT_FALSE(g->needsGather(1)); // decode skipped
}

TEST_F(Test__LogitsGatherer, NeedsGatherPrefill_DefaultTrue)
{
    auto g = createGatherer();
    EXPECT_TRUE(g->needsGather(10)); // prefill: seq_len > 1
}

TEST_F(Test__LogitsGatherer, NeedsGatherPrefill_SkipSetFalse)
{
    auto g = createGatherer();
    g->setSkipPrefill(true);
    EXPECT_FALSE(g->needsGather(10)); // prefill skipped
}

TEST_F(Test__LogitsGatherer, NeedsGatherDecodeAndPrefillIndependent)
{
    auto g = createGatherer();
    g->setSkipDecode(true);
    g->setSkipPrefill(false);
    EXPECT_FALSE(g->needsGather(1)); // decode skipped
    EXPECT_TRUE(g->needsGather(10)); // prefill not skipped

    g->setSkipDecode(false);
    g->setSkipPrefill(true);
    EXPECT_TRUE(g->needsGather(1));   // decode not skipped
    EXPECT_FALSE(g->needsGather(10)); // prefill skipped
}

// =============================================================================
// 4. Single-Device Gather (CPU path, no GPU needed)
// =============================================================================

TEST_F(Test__LogitsGatherer, GatherSingleDevice_DecodeCopiesLogits)
{
    auto g = createGatherer();
    auto runners = makeSingleRunner();

    // Set known logits on the runner
    auto *mock = static_cast<LogitsGathererMockRunner *>(runners[0].get());
    std::vector<float> expected(VOCAB);
    for (int i = 0; i < VOCAB; ++i)
        expected[i] = static_cast<float>(i) * 0.1f;
    mock->setLogitsData(expected);

    ASSERT_TRUE(g->gather(runners, 1, VOCAB));
    EXPECT_EQ(g->lastGatheredSize(), static_cast<size_t>(VOCAB));

    // Verify data was copied
    const float *result = g->data();
    ASSERT_NE(result, nullptr);
    for (int i = 0; i < VOCAB; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST_F(Test__LogitsGatherer, GatherSingleDevice_PrefillCopiesLogits)
{
    auto g = createGatherer();
    auto runners = makeSingleRunner();

    auto *mock = static_cast<LogitsGathererMockRunner *>(runners[0].get());
    mock->fillLogits(42.0f);

    ASSERT_TRUE(g->gather(runners, 1, VOCAB));
    EXPECT_EQ(g->lastGatheredSize(), static_cast<size_t>(VOCAB));

    const float *result = g->data();
    ASSERT_NE(result, nullptr);
    EXPECT_FLOAT_EQ(result[0], 42.0f);
}

TEST_F(Test__LogitsGatherer, GatherEmptyRunners_ReturnsFalse)
{
    auto g = createGatherer();
    std::vector<std::unique_ptr<IInferenceRunner>> empty;
    EXPECT_FALSE(g->gather(empty, 1, VOCAB));
}

// =============================================================================
// 5. copyFromStage
// =============================================================================

TEST_F(Test__LogitsGatherer, CopyFromStage_CopiesLogits)
{
    auto g = createGatherer();
    auto runner = std::make_unique<LogitsGathererMockRunner>(VOCAB);
    std::vector<float> expected(VOCAB);
    for (int i = 0; i < VOCAB; ++i)
        expected[i] = static_cast<float>(i) * -0.5f;
    runner->setLogitsData(expected);

    g->copyFromStage(*runner, 0, 1, 16);

    // Verify data was copied
    const float *result = g->data();
    ASSERT_NE(result, nullptr);
    for (int i = 0; i < VOCAB; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST_F(Test__LogitsGatherer, CopyFromStage_AllocatesIfNull)
{
    // Create a gatherer with 0 size (empty buffer)
    auto g = std::make_unique<LogitsGatherer>(0, 0);
    auto runner = std::make_unique<LogitsGathererMockRunner>(VOCAB);
    runner->fillLogits(7.0f);

    // copyFromStage should allocate the buffer on demand
    g->copyFromStage(*runner, 0, 1, 16);

    EXPECT_TRUE(g->isAllocated());
    const float *result = g->data();
    ASSERT_NE(result, nullptr);
    EXPECT_FLOAT_EQ(result[0], 7.0f);
}

// =============================================================================
// 6. lastGatheredSize tracking
// =============================================================================

TEST_F(Test__LogitsGatherer, LastGatheredSize_InitiallyZero)
{
    auto g = createGatherer();
    EXPECT_EQ(g->lastGatheredSize(), 0u);
}

TEST_F(Test__LogitsGatherer, LastGatheredSize_UpdatedAfterGather)
{
    auto g = createGatherer();
    auto runners = makeSingleRunner();
    auto *mock = static_cast<LogitsGathererMockRunner *>(runners[0].get());
    mock->fillLogits(1.0f);

    g->gather(runners, 1, VOCAB);
    EXPECT_EQ(g->lastGatheredSize(), static_cast<size_t>(VOCAB));
}

// =============================================================================
// 7. Move semantics
// =============================================================================

TEST_F(Test__LogitsGatherer, MoveConstructor)
{
    auto g1 = createGatherer();
    g1->setSkipDecode(true);

    LogitsGatherer g2(std::move(*g1));
    EXPECT_TRUE(g2.isAllocated());
    EXPECT_TRUE(g2.skipDecode());
}

TEST_F(Test__LogitsGatherer, MoveAssignment)
{
    auto g1 = createGatherer();
    auto runners = makeSingleRunner();
    auto *mock = static_cast<LogitsGathererMockRunner *>(runners[0].get());
    mock->fillLogits(99.0f);
    g1->gather(runners, 1, VOCAB);

    auto g2 = createGatherer(64, 8);
    *g2 = std::move(*g1);
    EXPECT_EQ(g2->lastGatheredSize(), static_cast<size_t>(VOCAB));
}

// =============================================================================
// 8. Multi-device column-parallel gather
// =============================================================================

/// Mock runner that advertises column-parallel local logits (CPU path)
class ColumnParallelMockRunner : public LogitsGathererMockRunner
{
public:
    ColumnParallelMockRunner(int local_vocab, const std::vector<float> &data)
        : LogitsGathererMockRunner(local_vocab), local_vocab_(local_vocab)
    {
        tensor_ = std::make_shared<FP32Tensor>(
            std::vector<size_t>{1, static_cast<size_t>(local_vocab)},
            DeviceId::cpu());
        std::memcpy(tensor_->mutable_data(), data.data(),
                    data.size() * sizeof(float));
    }

    bool hasLogitsLocal() const override { return true; }
    LogitsLocalInfo getLogitsLocalInfo() const override
    {
        LogitsLocalInfo info;
        info.gpu_ptr = nullptr; // CPU path
        info.vocab_local = static_cast<size_t>(local_vocab_);
        info.tensor = tensor_.get();
        return info;
    }

private:
    int local_vocab_;
    std::shared_ptr<FP32Tensor> tensor_;
};

TEST_F(Test__LogitsGatherer, GatherColumnParallel_TwoDevices_Decode)
{
    // Device 0 has vocab [0..63], device 1 has vocab [64..127]
    constexpr int LOCAL_V = 64;
    constexpr int FULL_V = 128;

    std::vector<float> data0(LOCAL_V), data1(LOCAL_V);
    for (int i = 0; i < LOCAL_V; ++i)
    {
        data0[i] = static_cast<float>(i);           // 0..63
        data1[i] = static_cast<float>(i + LOCAL_V); // 64..127
    }

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::make_unique<ColumnParallelMockRunner>(LOCAL_V, data0));
    runners.push_back(std::make_unique<ColumnParallelMockRunner>(LOCAL_V, data1));

    auto g = std::make_unique<LogitsGatherer>(FULL_V, 4);
    EXPECT_TRUE(g->gather(runners, 1, FULL_V));

    const float *out = g->data();
    ASSERT_NE(out, nullptr);

    // Verify interleaved output: [0, 1, ..., 63, 64, 65, ..., 127]
    for (int i = 0; i < FULL_V; ++i)
        EXPECT_FLOAT_EQ(out[i], static_cast<float>(i)) << "Mismatch at index " << i;

    EXPECT_EQ(g->lastGatheredSize(), static_cast<size_t>(FULL_V));
}

TEST_F(Test__LogitsGatherer, GatherColumnParallel_TwoDevices_Prefill)
{
    // 2 devices, 2 sequence positions, local vocab = 3 each
    constexpr int LOCAL_V = 3;
    constexpr int FULL_V = 6;
    constexpr size_t SEQ = 2;

    // Device 0: row0=[1,2,3], row1=[4,5,6]
    // Device 1: row0=[10,20,30], row1=[40,50,60]
    std::vector<float> data0 = {1, 2, 3, 4, 5, 6};
    std::vector<float> data1 = {10, 20, 30, 40, 50, 60};

    // Create mock runners with multi-row tensors
    auto r0 = std::make_unique<ColumnParallelMockRunner>(LOCAL_V, std::vector<float>(LOCAL_V));
    auto r1 = std::make_unique<ColumnParallelMockRunner>(LOCAL_V, std::vector<float>(LOCAL_V));

    // Override tensor data with multi-row data (seq_len=2)
    // We need larger tensors — create them directly
    auto tensor0 = std::make_shared<FP32Tensor>(
        std::vector<size_t>{SEQ, static_cast<size_t>(LOCAL_V)}, DeviceId::cpu());
    std::memcpy(tensor0->mutable_data(), data0.data(), data0.size() * sizeof(float));

    auto tensor1 = std::make_shared<FP32Tensor>(
        std::vector<size_t>{SEQ, static_cast<size_t>(LOCAL_V)}, DeviceId::cpu());
    std::memcpy(tensor1->mutable_data(), data1.data(), data1.size() * sizeof(float));

    // Need a mock that uses these larger tensors. Create a simple extension.
    class PrefillMockRunner : public LogitsGathererMockRunner
    {
    public:
        PrefillMockRunner(int local_v, std::shared_ptr<FP32Tensor> t)
            : LogitsGathererMockRunner(local_v), tensor_(std::move(t)), local_v_(local_v) {}
        bool hasLogitsLocal() const override { return true; }
        LogitsLocalInfo getLogitsLocalInfo() const override
        {
            LogitsLocalInfo info;
            info.gpu_ptr = nullptr;
            info.vocab_local = static_cast<size_t>(local_v_);
            info.tensor = tensor_.get();
            return info;
        }

    private:
        std::shared_ptr<FP32Tensor> tensor_;
        int local_v_;
    };

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::make_unique<PrefillMockRunner>(LOCAL_V, tensor0));
    runners.push_back(std::make_unique<PrefillMockRunner>(LOCAL_V, tensor1));

    auto g = std::make_unique<LogitsGatherer>(FULL_V, SEQ);
    EXPECT_TRUE(g->gather(runners, SEQ, FULL_V));

    const float *out = g->data();
    ASSERT_NE(out, nullptr);

    // Expected interleave:
    // Row 0: [1, 2, 3, 10, 20, 30]
    // Row 1: [4, 5, 6, 40, 50, 60]
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 2.0f);
    EXPECT_FLOAT_EQ(out[2], 3.0f);
    EXPECT_FLOAT_EQ(out[3], 10.0f);
    EXPECT_FLOAT_EQ(out[4], 20.0f);
    EXPECT_FLOAT_EQ(out[5], 30.0f);
    EXPECT_FLOAT_EQ(out[6], 4.0f);
    EXPECT_FLOAT_EQ(out[7], 5.0f);
    EXPECT_FLOAT_EQ(out[8], 6.0f);
    EXPECT_FLOAT_EQ(out[9], 40.0f);
    EXPECT_FLOAT_EQ(out[10], 50.0f);
    EXPECT_FLOAT_EQ(out[11], 60.0f);

    EXPECT_EQ(g->lastGatheredSize(), SEQ * FULL_V);
}

TEST_F(Test__LogitsGatherer, GatherColumnParallel_NullRunnerInList_Fails)
{
    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::make_unique<ColumnParallelMockRunner>(
        32, std::vector<float>(32, 1.0f)));
    runners.push_back(nullptr);

    auto g = std::make_unique<LogitsGatherer>(64, 4);
    EXPECT_FALSE(g->gather(runners, 1, 64));
}

TEST_F(Test__LogitsGatherer, GatherColumnParallel_UnequalVocabSlices)
{
    // Device 0: 80 vocab, Device 1: 48 vocab → total 128
    constexpr int V0 = 80, V1 = 48, FULL_V = 128;

    std::vector<float> data0(V0), data1(V1);
    for (int i = 0; i < V0; ++i)
        data0[i] = static_cast<float>(i);
    for (int i = 0; i < V1; ++i)
        data1[i] = static_cast<float>(V0 + i);

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::make_unique<ColumnParallelMockRunner>(V0, data0));
    runners.push_back(std::make_unique<ColumnParallelMockRunner>(V1, data1));

    auto g = std::make_unique<LogitsGatherer>(FULL_V, 4);
    EXPECT_TRUE(g->gather(runners, 1, FULL_V));

    const float *out = g->data();
    for (int i = 0; i < FULL_V; ++i)
        EXPECT_FLOAT_EQ(out[i], static_cast<float>(i));
}
