/**
 * @file Test__LogitsGatherer.cpp
 * @brief Unit tests for LogitsGatherer extracted from RankOrchestrator
 *
 * Tests buffer allocation, skip-gather control, needsGather() logic,
 * single-device gather path, copyFromStage delegation, and host pinning cleanup.
 * Bounded-output regressions reject oversized reads before touching storage and
 * prove that lazy terminal PP output has only one vocabulary row of capacity.
 * Captured-pipeline domain tests invoke the real rank completion to prove that
 * retained prefill bytes cannot masquerade as a newly completed decode row.
 */

#include <gtest/gtest.h>

#include "execution/local_execution/orchestrators/LogitsGatherer.h"
#include "execution/local_execution/orchestrators/IInferenceRunner.h"
#include "execution/local_execution/orchestrators/RankOrchestrator.h"
#include "mocks/MockBackend.h"
#include "mocks/MockModelContext.h"
#include "tensors/Tensors.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <tuple>
#include <vector>

using namespace llaminar2;

namespace
{
    /**
     * @brief Mock backend that records host pin/unpin calls for resolver-path tests.
     */
    class PinTrackingBackend : public test::MockBackend
    {
    public:
        explicit PinTrackingBackend(DeviceType device_type)
            : test::MockBackend(device_type) {}

        bool pinHostMemory(void *ptr, size_t bytes, int device_id) override
        {
            ++pin_count_;
            last_pinned_ptr_ = ptr;
            last_pinned_bytes_ = bytes;
            last_pinned_device_id_ = device_id;
            return true;
        }

        bool unpinHostMemory(void *ptr, int device_id) override
        {
            ++unpin_count_;
            last_unpinned_ptr_ = ptr;
            last_unpinned_device_id_ = device_id;
            return true;
        }

        size_t pinCount() const { return pin_count_; }
        size_t unpinCount() const { return unpin_count_; }
        void *lastPinnedPtr() const { return last_pinned_ptr_; }
        void *lastUnpinnedPtr() const { return last_unpinned_ptr_; }
        size_t lastPinnedBytes() const { return last_pinned_bytes_; }
        int lastPinnedDeviceId() const { return last_pinned_device_id_; }
        int lastUnpinnedDeviceId() const { return last_unpinned_device_id_; }

    private:
        size_t pin_count_ = 0;
        size_t unpin_count_ = 0;
        void *last_pinned_ptr_ = nullptr;
        void *last_unpinned_ptr_ = nullptr;
        size_t last_pinned_bytes_ = 0;
        int last_pinned_device_id_ = -1;
        int last_unpinned_device_id_ = -1;
    };

    PinTrackingBackend *g_cuda_pin_backend = nullptr;
    PinTrackingBackend *g_rocm_pin_backend = nullptr;

    /// @brief Routes backend lookups to test-owned CUDA/ROCm tracking backends.
    IBackend *resolvePinTrackingBackend(DeviceId device)
    {
        if (device.is_cuda())
            return g_cuda_pin_backend;
        if (device.is_rocm())
            return g_rocm_pin_backend;
        return nullptr;
    }

    /**
     * @brief Temporarily installs tracking backends for LogitsGatherer resolver tests.
     */
    class PinTrackingResolverScope
    {
    public:
        PinTrackingResolverScope(PinTrackingBackend &cuda_backend, PinTrackingBackend &rocm_backend)
        {
            g_cuda_pin_backend = &cuda_backend;
            g_rocm_pin_backend = &rocm_backend;
        }

        ~PinTrackingResolverScope()
        {
            g_cuda_pin_backend = nullptr;
            g_rocm_pin_backend = nullptr;
        }
    };

    /**
     * @brief CPU-only backend double for ordered GPU-logits D2H tests.
     *
     * Test vectors stand in for device storage, so these tests exercise only
     * LogitsGatherer's ownership and error propagation. No CUDA or ROCm runtime
     * is loaded by the unit suite.
     */
    class OrderedD2HBackend : public test::MockBackend
    {
    public:
        explicit OrderedD2HBackend(DeviceType device_type = DeviceType::CUDA)
            : test::MockBackend(device_type)
        {
        }

        bool deviceToHost(
            void *dst,
            const void *src,
            size_t bytes,
            int device_id,
            void *stream) override
        {
            streams_.push_back(stream);
            devices_.push_back(device_id);
            ++copy_count_;
            if (fail_copies_)
                return false;
            if (bytes > 0 && dst && src)
                std::memcpy(dst, src, bytes);
            return true;
        }

        bool deviceToHostFast(
            void *dst,
            const void *src,
            size_t bytes,
            int device_id,
            void *stream) override
        {
            ++fast_copy_count_;
            return deviceToHost(dst, src, bytes, device_id, stream);
        }

        void setFailCopies(bool fail) { fail_copies_ = fail; }
        size_t copyCount() const { return copy_count_; }
        size_t fastCopyCount() const { return fast_copy_count_; }
        const std::vector<void *> &streams() const { return streams_; }
        /// @return Original shard device ordinals supplied with each exact-stream copy.
        const std::vector<int> &devices() const { return devices_; }

    private:
        bool fail_copies_ = false;
        size_t copy_count_ = 0;
        size_t fast_copy_count_ = 0;
        std::vector<void *> streams_;
        std::vector<int> devices_;
    };

    OrderedD2HBackend *g_ordered_d2h_backend = nullptr;

    /// @brief Resolve the test-owned GPU backend without any native runtime registration.
    IBackend *resolveOrderedD2HBackend(DeviceId device)
    {
        return device.is_gpu() ? g_ordered_d2h_backend : nullptr;
    }

    /**
     * @brief Install one ordered-D2H backend for a lexical test scope.
     */
    class OrderedD2HResolverScope
    {
    public:
        explicit OrderedD2HResolverScope(OrderedD2HBackend &backend)
        {
            g_ordered_d2h_backend = &backend;
        }

        ~OrderedD2HResolverScope()
        {
            g_ordered_d2h_backend = nullptr;
        }
    };
} // namespace

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

TEST_F(Test__LogitsGatherer, HostDataIsInvalidUntilGathered)
{
    auto g = createGatherer();
    EXPECT_EQ(g->data(), nullptr);
    ASSERT_NE(g->mutableData(), nullptr);
}

TEST_F(Test__LogitsGatherer, DestructorUnpinsWithPinnedDeviceBackend)
{
    PinTrackingBackend cuda_backend(DeviceType::CUDA);
    PinTrackingBackend rocm_backend(DeviceType::ROCm);
    PinTrackingResolverScope resolver_scope(cuda_backend, rocm_backend);

    const void *buffer_ptr = nullptr;
    const size_t expected_bytes = static_cast<size_t>(VOCAB) * sizeof(float);

    {
        auto g = std::make_unique<LogitsGatherer>(VOCAB, MAX_TOKENS, resolvePinTrackingBackend);

        // Pin via a non-zero ROCm ordinal while CUDA is also available;
        // destruction must retain both backend type and exact registration
        // device instead of reconstructing ROCm:0 from a type-only flag.
        g->pinForDevice(DeviceId::rocm(3));
        buffer_ptr = g->mutableData();

        ASSERT_NE(buffer_ptr, nullptr);
        EXPECT_EQ(rocm_backend.pinCount(), 1u);
        EXPECT_EQ(rocm_backend.lastPinnedPtr(), buffer_ptr);
        EXPECT_EQ(rocm_backend.lastPinnedBytes(), expected_bytes);
        EXPECT_EQ(rocm_backend.lastPinnedDeviceId(), 3);
        EXPECT_EQ(cuda_backend.pinCount(), 0u);
    }

    EXPECT_EQ(rocm_backend.unpinCount(), 1u);
    EXPECT_EQ(rocm_backend.lastUnpinnedPtr(), buffer_ptr);
    EXPECT_EQ(rocm_backend.lastUnpinnedDeviceId(), 3);
    EXPECT_EQ(cuda_backend.unpinCount(), 0u);
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
    EXPECT_TRUE(g->needsGather(LogitsForwardPhase::Decode));
}

TEST_F(Test__LogitsGatherer, NeedsGatherDecode_SkipSetFalse)
{
    auto g = createGatherer();
    g->setSkipDecode(true);
    EXPECT_FALSE(g->needsGather(LogitsForwardPhase::Decode));
}

TEST_F(Test__LogitsGatherer, NeedsGatherPrefill_DefaultTrue)
{
    auto g = createGatherer();
    EXPECT_TRUE(g->needsGather(LogitsForwardPhase::Prefill));
}

TEST_F(Test__LogitsGatherer, NeedsGatherPrefill_SkipSetFalse)
{
    auto g = createGatherer();
    g->setSkipPrefill(true);
    EXPECT_FALSE(g->needsGather(LogitsForwardPhase::Prefill));
}

TEST_F(Test__LogitsGatherer, NeedsGatherDecodeAndPrefillIndependent)
{
    auto g = createGatherer();
    g->setSkipDecode(true);
    g->setSkipPrefill(false);
    EXPECT_FALSE(g->needsGather(LogitsForwardPhase::Decode));
    EXPECT_TRUE(g->needsGather(LogitsForwardPhase::Prefill));

    g->setSkipDecode(false);
    g->setSkipPrefill(true);
    EXPECT_TRUE(g->needsGather(LogitsForwardPhase::Decode));
    EXPECT_FALSE(g->needsGather(LogitsForwardPhase::Prefill));
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

/**
 * @brief Prove persistent allocation cannot expose data from an older forward.
 *
 * The gatherer deliberately retains its allocated buffer for economy, but the
 * allocation is not evidence that its bytes belong to the active request. A
 * forward boundary invalidates the publication until a new gather completes.
 */
TEST_F(Test__LogitsGatherer, InvalidateRetiresPreviouslyGatheredHostData)
{
    auto g = createGatherer();
    auto runners = makeSingleRunner();
    auto *mock = static_cast<LogitsGathererMockRunner *>(runners[0].get());
    mock->fillLogits(7.0f);

    ASSERT_TRUE(g->gather(runners, 1, VOCAB));
    ASSERT_NE(g->data(), nullptr);
    ASSERT_EQ(g->lastGatheredSize(), static_cast<size_t>(VOCAB));

    g->invalidate();

    EXPECT_TRUE(g->isAllocated());
    EXPECT_EQ(g->lastGatheredSize(), 0u);
    EXPECT_EQ(g->data(), nullptr);
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
    auto one_token_runner = std::make_unique<LogitsGathererMockRunner>(1);
    one_token_runner->setLogitsData({123.0f});
    g->copyFromStage(*one_token_runner, 1);
    ASSERT_EQ(g->lastGatheredSize(), 1u);

    auto runner = std::make_unique<LogitsGathererMockRunner>(VOCAB);
    std::vector<float> expected(VOCAB);
    for (int i = 0; i < VOCAB; ++i)
        expected[i] = static_cast<float>(i) * -0.5f;
    runner->setLogitsData(expected);

    g->copyFromStage(*runner);

    // Verify data was copied
    const float *result = g->data();
    ASSERT_NE(result, nullptr);
    for (int i = 0; i < VOCAB; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
    }
    EXPECT_EQ(g->lastGatheredSize(), static_cast<size_t>(VOCAB));
}

TEST_F(Test__LogitsGatherer, CopyFromStage_AllocatesIfNull)
{
    // Create a gatherer with 0 size (empty buffer)
    auto g = std::make_unique<LogitsGatherer>(0, 0);
    auto runner = std::make_unique<LogitsGathererMockRunner>(VOCAB);
    runner->fillLogits(7.0f);

    // copyFromStage should allocate the buffer on demand
    g->copyFromStage(*runner);

    EXPECT_TRUE(g->isAllocated());
    EXPECT_EQ(g->bufferNumel(), static_cast<size_t>(VOCAB));
    const float *result = g->data();
    ASSERT_NE(result, nullptr);
    EXPECT_FLOAT_EQ(result[0], 7.0f);
}

/** @brief Both replicated gather paths reject capacity overruns before copying. */
TEST_F(Test__LogitsGatherer, ReplicatedGatherRejectsRowsOutsideRetainedOutput)
{
    for (const size_t participants : {1u, 2u})
    {
        SCOPED_TRACE(participants);
        auto g = createGatherer(VOCAB, 1);
        auto runners = makeSingleRunner();
        if (participants == 2)
            runners.push_back(std::make_unique<LogitsGathererMockRunner>(VOCAB));
        ASSERT_TRUE(g->gather(runners, 1, VOCAB));
        EXPECT_FALSE(g->gather(runners, 2, VOCAB));
        EXPECT_EQ(g->data(), nullptr);
        EXPECT_EQ(g->lastGatheredSize(), 0u);
        EXPECT_FALSE(g->gather(runners, size_t(-1), VOCAB));
        EXPECT_FALSE(g->gather(runners, 1, 0));
        EXPECT_FALSE(g->gather(runners, 1, -1));
        EXPECT_FALSE(g->gather(runners, 0, VOCAB));
        EXPECT_TRUE(g->gather(runners, 1, VOCAB));
        EXPECT_EQ(g->bufferNumel(), static_cast<size_t>(VOCAB));
    }
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

/**
 * @brief GPU-shaped runner double with separate metadata and consuming views.
 *
 * getLogitsLocalInfo() deliberately returns no stream. The host-gather API is
 * the only operation allowed to attach the producer-ordered bridge stream.
 */
class OrderedGPUColumnParallelMockRunner : public LogitsGathererMockRunner
{
public:
    OrderedGPUColumnParallelMockRunner(
        int local_vocab,
        std::vector<float> device_data,
        void *host_bridge_stream,
        DeviceId device = DeviceId::cuda(0))
        : LogitsGathererMockRunner(local_vocab),
          local_vocab_(local_vocab),
          device_data_(std::move(device_data)),
          host_bridge_stream_(host_bridge_stream),
          device_(device),
          tensor_(std::make_shared<FP32Tensor>(
              std::vector<size_t>{1, static_cast<size_t>(local_vocab)},
              DeviceId::cpu()))
    {
        std::fill(
            tensor_->mutable_data(),
            tensor_->mutable_data() + local_vocab_,
            -999.0f);
    }

    bool hasLogitsLocal() const override { return true; }

    /** @return Authored placement metadata; no physical device is initialized. */
    DeviceId primaryDeviceId() const override { return device_; }

    /** @brief Count unexpected model dispatch; completion must never replay a child. */
    bool forward(const int *tokens, int rows) override
    {
        ++model_forwards_;
        return LogitsGathererMockRunner::forward(tokens, rows);
    }

    /** @return The single-child diagnostic view after its consuming publication boundary. */
    const float *logits() const override
    {
        const auto info = const_cast<OrderedGPUColumnParallelMockRunner *>(this)
                              ->consumeLogitsLocalInfoForHostGather();
        if (!info.stream)
            throw std::logic_error("Authored child host view has no producer-ordered stream");
        return device_data_.data();
    }

    LogitsLocalInfo getLogitsLocalInfo() const override
    {
        return LogitsLocalInfo{
            device_data_.data(),
            device_,
            static_cast<size_t>(local_vocab_),
            0,
            tensor_.get(),
            nullptr};
    }

    LogitsLocalInfo consumeLogitsLocalInfoForHostGather() override
    {
        ++host_gather_consumes_;
        LogitsLocalInfo info = getLogitsLocalInfo();
        info.stream = host_bridge_stream_;
        return info;
    }

    size_t hostGatherConsumes() const { return host_gather_consumes_; }

    /** @brief Replace a completed authored row and its exact next producer bridge. */
    void publishCompletedRow(const std::vector<float> &row, void *stream)
    {
        if (row.size() != device_data_.size())
            throw std::invalid_argument("Authored logits row changed retained geometry");
        std::copy(row.begin(), row.end(), device_data_.begin());
        host_bridge_stream_ = stream;
    }

    /** @return Number of model forwards; host publication must leave this zero. */
    size_t modelForwards() const { return model_forwards_; }

private:
    int local_vocab_;
    std::vector<float> device_data_;
    void *host_bridge_stream_ = nullptr;
    DeviceId device_;
    std::shared_ptr<FP32Tensor> tensor_;
    size_t host_gather_consumes_ = 0;
    size_t model_forwards_ = 0;
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

TEST_F(Test__LogitsGatherer, GatherGPUShardsConsumesExactHostBridgeStreams)
{
    constexpr int LOCAL_V = 4;
    constexpr int FULL_V = 8;
    void *stream0 = reinterpret_cast<void *>(0x1010);
    void *stream1 = reinterpret_cast<void *>(0x2020);

    OrderedD2HBackend backend;
    OrderedD2HResolverScope resolver_scope(backend);

    auto runner0 = std::make_unique<OrderedGPUColumnParallelMockRunner>(
        LOCAL_V,
        std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f},
        stream0);
    auto runner1 = std::make_unique<OrderedGPUColumnParallelMockRunner>(
        LOCAL_V,
        std::vector<float>{4.0f, 5.0f, 6.0f, 7.0f},
        stream1);
    auto *runner0_view = runner0.get();
    auto *runner1_view = runner1.get();

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    LogitsGatherer gatherer(
        FULL_V,
        /*max_tokens=*/1,
        resolveOrderedD2HBackend);
    ASSERT_TRUE(gatherer.gather(runners, /*seq_len=*/1, FULL_V));

    EXPECT_EQ(runner0_view->hostGatherConsumes(), 1U);
    EXPECT_EQ(runner1_view->hostGatherConsumes(), 1U);
    ASSERT_EQ(backend.streams().size(), 2U);
    EXPECT_EQ(backend.streams()[0], stream0);
    EXPECT_EQ(backend.streams()[1], stream1);
    EXPECT_EQ(backend.fastCopyCount(), 2U);
    for (int token = 0; token < FULL_V; ++token)
    {
        EXPECT_FLOAT_EQ(
            gatherer.data()[token],
            static_cast<float>(token));
    }
}

TEST_F(Test__LogitsGatherer, GatherGPUShardRejectsNullHostBridgeStream)
{
    constexpr int LOCAL_V = 4;
    constexpr int FULL_V = 8;

    OrderedD2HBackend backend;
    OrderedD2HResolverScope resolver_scope(backend);

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::make_unique<OrderedGPUColumnParallelMockRunner>(
        LOCAL_V,
        std::vector<float>(LOCAL_V, 1.0f),
        /*host_bridge_stream=*/nullptr));
    runners.push_back(std::make_unique<OrderedGPUColumnParallelMockRunner>(
        LOCAL_V,
        std::vector<float>(LOCAL_V, 2.0f),
        reinterpret_cast<void *>(0x3030)));

    LogitsGatherer gatherer(
        FULL_V,
        /*max_tokens=*/1,
        resolveOrderedD2HBackend);
    EXPECT_FALSE(gatherer.gather(runners, /*seq_len=*/1, FULL_V));
    EXPECT_EQ(backend.copyCount(), 0U);
    EXPECT_EQ(gatherer.lastGatheredSize(), 0U);
}

TEST_F(Test__LogitsGatherer, GatherGPUShardPropagatesOrderedD2HFailure)
{
    constexpr int LOCAL_V = 4;
    constexpr int FULL_V = 8;

    OrderedD2HBackend backend;
    backend.setFailCopies(true);
    OrderedD2HResolverScope resolver_scope(backend);

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::make_unique<OrderedGPUColumnParallelMockRunner>(
        LOCAL_V,
        std::vector<float>(LOCAL_V, 1.0f),
        reinterpret_cast<void *>(0x4040)));
    runners.push_back(std::make_unique<OrderedGPUColumnParallelMockRunner>(
        LOCAL_V,
        std::vector<float>(LOCAL_V, 2.0f),
        reinterpret_cast<void *>(0x5050)));

    LogitsGatherer gatherer(
        FULL_V,
        /*max_tokens=*/1,
        resolveOrderedD2HBackend);
    EXPECT_FALSE(gatherer.gather(runners, /*seq_len=*/1, FULL_V));
    EXPECT_EQ(backend.copyCount(), 1U);
    EXPECT_EQ(gatherer.lastGatheredSize(), 0U);
}

TEST_F(Test__LogitsGatherer, GatherLocalInfos_ReplicatedFullVocabUsesPrimary)
{
    // LocalTP expert-overlay runs can replicate dense weights, including the LM
    // head. In that mode every participant exposes a full-vocab logits row; the
    // gatherer must not concatenate those replicas into a bogus 2x vocab row.
    constexpr int FULL_V = 128;

    auto tensor0 = std::make_shared<FP32Tensor>(
        std::vector<size_t>{1, static_cast<size_t>(FULL_V)}, DeviceId::cpu());
    auto tensor1 = std::make_shared<FP32Tensor>(
        std::vector<size_t>{1, static_cast<size_t>(FULL_V)}, DeviceId::cpu());

    for (int i = 0; i < FULL_V; ++i)
    {
        tensor0->mutable_data()[i] = static_cast<float>(i);
        tensor1->mutable_data()[i] = static_cast<float>(1000 + i);
    }

    LogitsLocalInfo info0;
    info0.vocab_local = FULL_V;
    info0.tensor = tensor0.get();

    LogitsLocalInfo info1;
    info1.vocab_local = FULL_V;
    info1.tensor = tensor1.get();

    auto g = std::make_unique<LogitsGatherer>(FULL_V, 1);
    EXPECT_TRUE(g->gatherLocalInfos({info0, info1}, 1, FULL_V));

    const float *out = g->data();
    ASSERT_NE(out, nullptr);
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

TEST_F(Test__LogitsGatherer, GatherLocalInfos_ColumnParallelFullStorageUsesSemanticStride)
{
    constexpr int LOCAL_V = 4;
    constexpr int FULL_V = 8;
    constexpr size_t SEQ = 2;

    auto tensor0 = std::make_shared<FP32Tensor>(
        std::vector<size_t>{SEQ, static_cast<size_t>(FULL_V)}, DeviceId::cpu());
    auto tensor1 = std::make_shared<FP32Tensor>(
        std::vector<size_t>{SEQ, static_cast<size_t>(FULL_V)}, DeviceId::cpu());

    std::fill(tensor0->mutable_data(), tensor0->mutable_data() + SEQ * FULL_V, -1000.0f);
    std::fill(tensor1->mutable_data(), tensor1->mutable_data() + SEQ * FULL_V, -2000.0f);

    const std::array<float, LOCAL_V> row0_dev0 = {1, 2, 3, 4};
    const std::array<float, LOCAL_V> row1_dev0 = {5, 6, 7, 8};
    const std::array<float, LOCAL_V> row0_dev1 = {10, 20, 30, 40};
    const std::array<float, LOCAL_V> row1_dev1 = {50, 60, 70, 80};

    std::memcpy(tensor0->mutable_data(), row0_dev0.data(), LOCAL_V * sizeof(float));
    std::memcpy(tensor0->mutable_data() + FULL_V, row1_dev0.data(), LOCAL_V * sizeof(float));
    std::memcpy(tensor1->mutable_data(), row0_dev1.data(), LOCAL_V * sizeof(float));
    std::memcpy(tensor1->mutable_data() + FULL_V, row1_dev1.data(), LOCAL_V * sizeof(float));

    LogitsLocalInfo info0;
    info0.vocab_local = LOCAL_V;
    info0.tensor = tensor0.get();
    info0.row_stride = FULL_V;

    LogitsLocalInfo info1;
    info1.vocab_local = LOCAL_V;
    info1.tensor = tensor1.get();
    info1.row_stride = FULL_V;

    auto g = std::make_unique<LogitsGatherer>(FULL_V, SEQ);
    EXPECT_TRUE(g->gatherLocalInfos({info0, info1}, SEQ, FULL_V));

    const float *out = g->data();
    ASSERT_NE(out, nullptr);
    const std::array<float, SEQ * FULL_V> expected = {
        1, 2, 3, 4, 10, 20, 30, 40,
        5, 6, 7, 8, 50, 60, 70, 80};
    for (size_t i = 0; i < expected.size(); ++i)
        EXPECT_FLOAT_EQ(out[i], expected[i]) << "Mismatch at index " << i;
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

namespace llaminar2::test
{
    /**
     * @brief Seed authored retained host state and invoke the actual private rank completion.
     *
     * Only the consuming logits boundary is under test. The fixture does not
     * emulate a pipeline, run model forwards, or initialize a GPU runtime.
     */
    struct RankPipelineHostPublicationTestAccess
    {
        /** @brief Bind the existing deterministic backend resolver to the real gatherer. */
        static void installObservationOwner(RankOrchestrator &rank)
        {
            rank.logits_gatherer_ = std::make_unique<LogitsGatherer>(
                rank.vocab_size(), 1, resolveOrderedD2HBackend);
            rank.applyLogitsGatherSkipFlags();
        }

        /** @brief Retain a prefill row through the real child-consuming gather API. */
        static bool primePrefill(RankOrchestrator &rank)
        {
            rank.current_position_ = 9;
            rank.current_sequence_lengths_ = {9};
            rank.current_padded_seq_len_ = 9;
            rank.last_logits_forward_phase_ = LogitsForwardPhase::Prefill;
            return rank.logits_gatherer_->gather(rank.device_runners_, 1, rank.vocab_size());
        }

        /** @brief Invoke the same completion called by PipelineDeviceGeneration after its real output events. */
        static void complete(RankOrchestrator &rank) { rank.retirePipelineForwardOnlyDomain(); }

        /** @return Actual rank metadata that a failed host publication must leave unchanged. */
        static auto metadata(const RankOrchestrator &rank)
        {
            return std::tuple{rank.current_position_, rank.current_sequence_lengths_,
                rank.current_padded_seq_len_, rank.last_logits_forward_phase_,
                rank.rank_resident_logical_state_epoch_};
        }

        /** @return Host data only while the real gatherer certifies a current row. */
        static const float *hostData(const RankOrchestrator &rank)
        {
            return rank.logits_gatherer_ ? rank.logits_gatherer_->data() : nullptr;
        }
    };
}

namespace
{
    using PipelinePublicationAccess = test::RankPipelineHostPublicationTestAccess;

    /** @brief Test-owned completed shards and their real RankOrchestrator host publication boundary. */
    struct CompletedPipelineDomain
    {
        std::vector<OrderedGPUColumnParallelMockRunner *> children;
        std::unique_ptr<RankOrchestrator> rank;
        static constexpr int local_vocab = 4;

        /** @return Distinct authored exact-stream identities, not physical GPU observations. */
        static void *stream(size_t participant, int generation)
        {
            return reinterpret_cast<void *>(uintptr_t(0x1000 + 0x100 * generation + 0x10 * participant));
        }

        /** @brief Create any positive participant width without querying an accelerator. */
        CompletedPipelineDomain(size_t width, DeviceType backend, bool owns_head = true)
        {
            std::vector<std::unique_ptr<IInferenceRunner>> runners;
            for (size_t index = 0; index < width; ++index)
            {
                const auto device = backend == DeviceType::CUDA
                    ? DeviceId::cuda(static_cast<int>(index)) : DeviceId::rocm(static_cast<int>(index));
                auto child = std::make_unique<OrderedGPUColumnParallelMockRunner>(
                    local_vocab, std::vector<float>(local_vocab, -1.0f), stream(index, 0), device);
                children.push_back(child.get());
                runners.push_back(std::move(child));
            }
            auto model = test::MockModelContext::createMinimal();
            model->setVocabSize(width * local_vocab);
            RankOrchestrator::Config config;
            config.nested_pp_stage_config = FactoryPPStageConfig{
                .first_layer = 0, .last_layer = 1, .has_embedding = false, .has_lm_head = owns_head};
            rank = RankOrchestrator::createForTest(std::move(model), std::move(runners), nullptr, config);
            if (owns_head) PipelinePublicationAccess::installObservationOwner(*rank);
        }

        /** @brief Publish one changed completed row per original shard without executing inference. */
        void publish(float base, int generation, bool missing_first_stream = false)
        {
            for (size_t participant = 0; participant < children.size(); ++participant)
            {
                std::vector<float> row(local_vocab);
                for (int column = 0; column < local_vocab; ++column)
                    row[column] = base + static_cast<float>(participant * local_vocab + column);
                children[participant]->publishCompletedRow(row,
                    missing_first_stream && participant == 0 ? nullptr : stream(participant, generation));
            }
        }

        /** @brief Check complete byte freshness and that observation never submitted another model row. */
        void expectPublished(float base) const
        {
            std::vector<float> expected(children.size() * local_vocab);
            for (size_t column = 0; column < expected.size(); ++column)
                expected[column] = base + static_cast<float>(column);
            const float *published = PipelinePublicationAccess::hostData(*rank);
            ASSERT_NE(published, nullptr);
            EXPECT_EQ(std::memcmp(published, expected.data(), expected.size() * sizeof(float)), 0);
            const float *actual = rank->logits();
            ASSERT_NE(actual, nullptr);
            EXPECT_EQ(std::memcmp(actual, expected.data(), expected.size() * sizeof(float)), 0);
            for (const auto *child : children) EXPECT_EQ(child->modelForwards(), 0u);
        }
    };
}

/** @test Actual domain completion refreshes original shards across reset, replay and arbitrary widths. */
TEST_F(Test__LogitsGatherer, PipelineForwardOnlyRefreshesCurrentHostRows)
{
    for (const auto backend_type : {DeviceType::CUDA, DeviceType::ROCm})
    {
        OrderedD2HBackend backend(backend_type);
        OrderedD2HResolverScope resolver(backend);
        for (const size_t width : {1u, 2u, 3u, 8u})
        {
            SCOPED_TRACE(::testing::Message() << "backend=" << static_cast<int>(backend_type) << " width=" << width);
            CompletedPipelineDomain domain(width, backend_type);
            for (int request = 0; request < 3; ++request)
            {
                if (request) domain.rank->clear_cache();
                const float initial = static_cast<float>(100 * request);
                domain.publish(initial, 0);
                ASSERT_TRUE(PipelinePublicationAccess::primePrefill(*domain.rank));
                domain.expectPublished(initial);
                for (int step = 0; step < 3; ++step)
                {
                    const float next = initial + static_cast<float>(10 * (step + 1));
                    std::vector<size_t> consumed;
                    for (const auto *child : domain.children) consumed.push_back(child->hostGatherConsumes());
                    const auto before_copies = backend.copyCount();
                    domain.publish(next, step + 1);
                    PipelinePublicationAccess::complete(*domain.rank);
                    // Public get_position() delegates to the actual child KV
                    // owner. This unit boundary authors completed child rows,
                    // so inspect only the rank metadata this method publishes.
                    const auto metadata = PipelinePublicationAccess::metadata(*domain.rank);
                    EXPECT_EQ(std::get<0>(metadata), 10 + step);
                    EXPECT_EQ(std::get<1>(metadata), std::vector<int>({10 + step}));
                    EXPECT_EQ(std::get<2>(metadata), 1);
                    EXPECT_EQ(std::get<3>(metadata), std::optional{LogitsForwardPhase::Decode});
                    EXPECT_EQ(backend.copyCount() - before_copies, width == 1 ? 0u : width);
                    if (width > 1)
                        ASSERT_EQ(backend.streams().size(), before_copies + width);
                    for (size_t participant = 0; participant < width; ++participant)
                    {
                        EXPECT_EQ(domain.children[participant]->hostGatherConsumes(), consumed[participant] + 1);
                        if (width > 1)
                        {
                            EXPECT_EQ(backend.streams()[before_copies + participant],
                                CompletedPipelineDomain::stream(participant, step + 1));
                            EXPECT_EQ(backend.devices()[before_copies + participant], static_cast<int>(participant));
                        }
                    }
                    domain.expectPublished(next);
                }
            }
        }
    }
}

/** @test Device-only and hidden-stage completion cannot retain or gather a host logits row. */
TEST_F(Test__LogitsGatherer, PipelineForwardOnlyDisabledAndHiddenDomainsDoNotGather)
{
    for (const auto backend_type : {DeviceType::CUDA, DeviceType::ROCm})
    {
        OrderedD2HBackend backend(backend_type);
        OrderedD2HResolverScope resolver(backend);
        CompletedPipelineDomain domain(3, backend_type);
        domain.publish(0.0f, 0);
        ASSERT_TRUE(PipelinePublicationAccess::primePrefill(*domain.rank));
        domain.rank->setSkipLogitsGatherDecode(true);
        domain.publish(100.0f, 1);
        const auto before = backend.copyCount();
        PipelinePublicationAccess::complete(*domain.rank);
        EXPECT_EQ(PipelinePublicationAccess::hostData(*domain.rank), nullptr);
        EXPECT_EQ(backend.copyCount(), before);
        for (const auto *child : domain.children) EXPECT_EQ(child->hostGatherConsumes(), 1u);
        EXPECT_THROW(domain.rank->logits(), std::logic_error);
        domain.rank->setSkipLogitsGatherDecode(false);
        PipelinePublicationAccess::complete(*domain.rank);
        domain.expectPublished(100.0f);

        CompletedPipelineDomain hidden(3, backend_type, false);
        hidden.publish(200.0f, 2);
        const auto before_hidden = backend.copyCount();
        PipelinePublicationAccess::complete(*hidden.rank);
        EXPECT_EQ(backend.copyCount(), before_hidden);
        for (const auto *child : hidden.children) EXPECT_EQ(child->hostGatherConsumes(), 0u);
        EXPECT_THROW(hidden.rank->logits(), std::logic_error);
    }
}

/** @test Rejected exact-stream or D2H publication invalidates data before any rank metadata advances. */
TEST_F(Test__LogitsGatherer, PipelineForwardOnlyFailedGatherCannotPublishMetadata)
{
    for (const auto backend_type : {DeviceType::CUDA, DeviceType::ROCm})
    {
        for (const bool missing_stream : {false, true})
        {
            OrderedD2HBackend backend(backend_type);
            OrderedD2HResolverScope resolver(backend);
            CompletedPipelineDomain domain(3, backend_type);
            domain.publish(0.0f, 0);
            ASSERT_TRUE(PipelinePublicationAccess::primePrefill(*domain.rank));
            const auto before = PipelinePublicationAccess::metadata(*domain.rank);
            domain.publish(100.0f, 1, missing_stream);
            backend.setFailCopies(!missing_stream);
            EXPECT_THROW(PipelinePublicationAccess::complete(*domain.rank), std::runtime_error);
            EXPECT_EQ(PipelinePublicationAccess::hostData(*domain.rank), nullptr);
            EXPECT_EQ(PipelinePublicationAccess::metadata(*domain.rank), before);
            for (const auto *child : domain.children) EXPECT_EQ(child->modelForwards(), 0u);
        }
    }
}
