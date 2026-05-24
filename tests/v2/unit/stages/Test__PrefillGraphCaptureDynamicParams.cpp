/**
 * @file Test__PrefillGraphCaptureDynamicParams.cpp
 * @brief Phase 3 tests: Dynamic parameter mechanism for prefill graph capture.
 *
 * Validates:
 * 1. EmbeddingStage::isPrefillGraphCaptureReady() gating
 * 2. EmbeddingStage::setStableTokenPointer() contract
 * 3. Prelaunch token upload succeeds under graph capture (ROCm)
 * 4. Missing preload fails under graph capture (ROCm)
 * 5. KVCacheAppendStage replay callback contract
 * 6. ForwardGraphCache replay_callback_stages caching
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "execution/compute_stages/stages/EmbeddingStage.h"
#include "execution/compute_stages/stages/KVCacheAppendStage.h"
#include "execution/compute_stages/stages/LMHeadStage.h"
#include "execution/local_execution/engine/ForwardGraphTypes.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "tensors/Tensors.h"
#include "backends/DeviceId.h"
#include "mocks/MockComputeStage.h"
#include "utils/PreparedWeightTestHarness.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#include "kernels/KernelFactory.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "interfaces/IWorkspaceConsumer.h"
#endif

using namespace llaminar2;
using namespace llaminar2::test;
using namespace llaminar2::testing;

namespace
{

    /**
     * @brief Minimal KV cache that records host-side replay advancement.
     *
     * Prefill graph replay tests do not need real K/V storage; they only need to
     * verify that KVCacheAppendStage calls advanceHead() with the count selected by
     * its dynamic replay metadata.
     */
    class RecordingKVCache final : public IKVCache
    {
    public:
        ActivationPrecision k_precision() const override { return ActivationPrecision::FP32; }
        int get_cached_tokens(int layer, int seq_idx = 0) const override
        {
            (void)layer;
            (void)seq_idx;
            return cached_tokens_;
        }
        int max_seq_len() const override { return 4096; }
        int n_layers() const override { return 1; }

        bool get_kv(int layer, int seq_idx, ITensor **out_k, ITensor **out_v, int *out_kv_len = nullptr) override
        {
            (void)layer;
            (void)seq_idx;
            if (out_k)
                *out_k = nullptr;
            if (out_v)
                *out_v = nullptr;
            if (out_kv_len)
                *out_kv_len = cached_tokens_;
            return true;
        }

        bool get_kv(int layer, int seq_idx, const ITensor **out_k, const ITensor **out_v, int *out_kv_len = nullptr) const override
        {
            (void)layer;
            (void)seq_idx;
            if (out_k)
                *out_k = nullptr;
            if (out_v)
                *out_v = nullptr;
            if (out_kv_len)
                *out_kv_len = cached_tokens_;
            return true;
        }

        bool append(int layer, int seq_idx, const ITensor *K, const ITensor *V, int num_tokens) override
        {
            (void)layer;
            (void)seq_idx;
            (void)K;
            (void)V;
            if (!isGraphCaptureActive())
                cached_tokens_ += num_tokens;
            return true;
        }

        bool isGraphCaptureReady() const override { return true; }

        void advanceHead(int layer, int seq_idx, int num_tokens) override
        {
            last_layer_ = layer;
            last_seq_idx_ = seq_idx;
            last_advance_tokens_ = num_tokens;
            advance_calls_++;
            cached_tokens_ += num_tokens;
        }

        void clear() override { cached_tokens_ = 0; }
        void clear_sequence(int layer, int seq_idx) override
        {
            (void)layer;
            (void)seq_idx;
            cached_tokens_ = 0;
        }
        void clear_layer(int layer) override
        {
            (void)layer;
            cached_tokens_ = 0;
        }

        int lastLayer() const { return last_layer_; }
        int lastSeqIdx() const { return last_seq_idx_; }
        int lastAdvanceTokens() const { return last_advance_tokens_; }
        int advanceCalls() const { return advance_calls_; }

    private:
        int cached_tokens_ = 0;
        int last_layer_ = -1;
        int last_seq_idx_ = -1;
        int last_advance_tokens_ = 0;
        int advance_calls_ = 0;
    };

    // =========================================================================
    // Mock stage with configurable needsOnGraphReplayed()
    // =========================================================================

    class MockReplayCallbackStage : public MockComputeStage
    {
    public:
        explicit MockReplayCallbackStage(bool needs_replay, std::string name = "MockReplay")
            : MockComputeStage(ComputeStageType::GEMM, std::move(name), DeviceId::cpu()),
              needs_replay_(needs_replay) {}

        bool needsOnGraphReplayed() const override { return needs_replay_; }

        void onGraphReplayed() override { replay_count_++; }

        int replayCount() const { return replay_count_; }

    private:
        bool needs_replay_ = false;
        int replay_count_ = 0;
    };

    /**
     * @brief Mock stage that records fixed-bucket prefill replay metadata.
     */
    class MockPrefillReplayParamStage : public MockComputeStage
    {
    public:
        explicit MockPrefillReplayParamStage(std::string name = "MockPrefillReplay")
            : MockComputeStage(ComputeStageType::KV_CACHE_APPEND, std::move(name), DeviceId::cpu()) {}

        bool hasPrefillReplayParams() const override { return true; }

        void updatePrefillReplayParams(const PrefillReplayParams &params) override
        {
            last_params_ = params;
            update_count_++;
        }

        int updateCount() const { return update_count_; }
        PrefillReplayParams lastParams() const { return last_params_; }

    private:
        PrefillReplayParams last_params_{};
        int update_count_ = 0;
    };

    /**
     * @brief Creates hidden states whose real-token rows are modest and whose
     *        padded bucket tail is intentionally easy to distinguish.
     */
    std::unique_ptr<FP32Tensor> createDeterministicHiddenStates(
        int bucket_seq_len,
        int d_model,
        int hostile_tail_start_row)
    {
        auto hidden_states = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(bucket_seq_len), static_cast<size_t>(d_model)},
            DeviceId::cpu());
        float *hidden_data = hidden_states->mutable_data();

        for (int hidden_row = 0; hidden_row < bucket_seq_len; ++hidden_row)
        {
            const bool is_hostile_tail = hidden_row >= hostile_tail_start_row;
            for (int feature_idx = 0; feature_idx < d_model; ++feature_idx)
            {
                const size_t element_idx = static_cast<size_t>(hidden_row) * d_model + feature_idx;
                const float row_term = 0.03125f * static_cast<float>(hidden_row + 1);
                const float feature_term = 0.001f * static_cast<float>((feature_idx % 17) - 8);
                hidden_data[element_idx] = is_hostile_tail
                                               ? 25.0f + 3.0f * static_cast<float>(hidden_row) + 0.125f * static_cast<float>(feature_idx)
                                               : row_term + feature_term;
            }
        }

        return hidden_states;
    }

    /**
     * @brief Creates deterministic FP32 LM-head weights in [vocab_size, d_model] layout.
     */
    std::unique_ptr<FP32Tensor> createDeterministicLmHeadWeights(int vocab_size, int d_model)
    {
        auto lm_head_weight = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(vocab_size), static_cast<size_t>(d_model)},
            DeviceId::cpu());
        float *weight_data = lm_head_weight->mutable_data();

        for (int vocab_idx = 0; vocab_idx < vocab_size; ++vocab_idx)
        {
            for (int feature_idx = 0; feature_idx < d_model; ++feature_idx)
            {
                const size_t element_idx = static_cast<size_t>(vocab_idx) * d_model + feature_idx;
                const float vocab_term = 0.0025f * static_cast<float>((vocab_idx % 13) - 6);
                const float feature_term = 0.00075f * static_cast<float>((feature_idx % 11) - 5);
                weight_data[element_idx] = vocab_term + feature_term;
            }
        }

        return lm_head_weight;
    }

    /**
     * @brief Computes the CPU reference for one selected hidden row projected by LM head.
     */
    std::vector<float> computeLmHeadReference(
        const FP32Tensor &hidden_states,
        const FP32Tensor &lm_head_weight,
        int hidden_row,
        int vocab_size,
        int d_model)
    {
        const float *hidden_data = hidden_states.data();
        const float *weight_data = lm_head_weight.data();
        std::vector<float> reference(static_cast<size_t>(vocab_size), 0.0f);

        for (int vocab_idx = 0; vocab_idx < vocab_size; ++vocab_idx)
        {
            float dot_product = 0.0f;
            for (int feature_idx = 0; feature_idx < d_model; ++feature_idx)
            {
                const float activation = hidden_data[static_cast<size_t>(hidden_row) * d_model + feature_idx];
                const float weight = weight_data[static_cast<size_t>(vocab_idx) * d_model + feature_idx];
                dot_product += activation * weight;
            }
            reference[static_cast<size_t>(vocab_idx)] = dot_product;
        }

        return reference;
    }

    /**
     * @brief Returns the maximum absolute difference between logits row 0 and a reference vector.
     */
    float maxLogitDifference(const FP32Tensor &logits, const std::vector<float> &reference)
    {
        const float *logit_data = logits.data();
        float max_difference = 0.0f;
        for (size_t vocab_idx = 0; vocab_idx < reference.size(); ++vocab_idx)
        {
            max_difference = std::max(max_difference, std::fabs(logit_data[vocab_idx] - reference[vocab_idx]));
        }
        return max_difference;
    }

    /**
     * @brief Asserts that logits row 0 matches a CPU reference within FP32 tolerance.
     */
    void expectLogitsNearReference(const FP32Tensor &logits, const std::vector<float> &reference)
    {
        const float *logit_data = logits.data();
        for (size_t vocab_idx = 0; vocab_idx < reference.size(); ++vocab_idx)
        {
            EXPECT_NEAR(logit_data[vocab_idx], reference[vocab_idx], 1e-5f)
                << "vocab_idx=" << vocab_idx;
        }
    }

    // =========================================================================
    // Test Fixture
    // =========================================================================

    class Test__PrefillGraphCaptureDynamicParams : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            vocab_size_ = 512;
            d_model_ = 64;
        }

        std::unique_ptr<FP32Tensor> createEmbeddingTable() const
        {
            auto table = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(vocab_size_), static_cast<size_t>(d_model_)});
            float *data = table->mutable_data();
            for (int i = 0; i < vocab_size_ * d_model_; ++i)
                data[i] = static_cast<float>(i % 100) * 0.01f;
            return table;
        }

        EmbeddingStage::Params makeParams(const ITensor *embed_table,
                                          const int *token_ids,
                                          ITensor *output,
                                          int num_tokens,
                                          DeviceId device) const
        {
            EmbeddingStage::Params p{};
            p.embed_table = embed_table;
            p.token_ids = token_ids;
            p.output = output;
            p.num_tokens = num_tokens;
            p.d_model = d_model_;
            p.vocab_size = vocab_size_;
            p.device_id = device;
            return p;
        }

        int vocab_size_ = 0;
        int d_model_ = 0;
    };

    // =========================================================================
    // Test 1: isPrefillGraphCaptureReady returns false on CPU
    // =========================================================================

    TEST_F(Test__PrefillGraphCaptureDynamicParams, EmbeddingReadiness_CPU_ReturnsFalse)
    {
        auto embed_table = createEmbeddingTable();
        auto output = std::make_unique<FP32Tensor>(
            std::vector<size_t>{4, static_cast<size_t>(d_model_)});
        std::vector<int> tokens = {1, 2, 3, 4};

        auto params = makeParams(embed_table.get(), tokens.data(), output.get(), 4, DeviceId::cpu());
        EmbeddingStage stage(params);

        // CPU device → not ready for graph capture
        EXPECT_FALSE(stage.isPrefillGraphCaptureReady());
    }

    // =========================================================================
    // Test 2: isPrefillGraphCaptureReady returns true on ROCm with workspace
    // =========================================================================

    TEST_F(Test__PrefillGraphCaptureDynamicParams, EmbeddingReadiness_ROCm_ReturnsTrue)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support compiled";
#else
        int device_count = 0;
        hipGetDeviceCount(&device_count);
        if (device_count <= 0)
            GTEST_SKIP() << "No ROCm device available";
        hipSetDevice(0);

        auto embed_table = createEmbeddingTable();
        auto output = std::make_unique<FP32Tensor>(
            std::vector<size_t>{4, static_cast<size_t>(d_model_)});
        std::vector<int> tokens = {1, 2, 3, 4};

        embed_table->ensureOnDevice(DeviceId::rocm(0));
        output->ensureOnDevice(DeviceId::rocm(0));

        auto params = makeParams(embed_table.get(), tokens.data(), output.get(), 4, DeviceId::rocm(0));
        EmbeddingStage stage(params);

        // getKernelAsWorkspaceConsumer triggers kernel creation without executing
        auto *ws_consumer = stage.getKernelAsWorkspaceConsumer();
        if (!ws_consumer)
            GTEST_SKIP() << "Embedding kernel doesn't implement IWorkspaceConsumer";

        auto reqs = ws_consumer->getWorkspaceRequirements(4, d_model_, 0);
        DeviceWorkspaceManager wsm(DeviceId::rocm(0), 64 * 1024 * 1024);
        ASSERT_TRUE(wsm.allocate(reqs));
        ws_consumer->bindWorkspace(&wsm);

        EXPECT_TRUE(stage.isPrefillGraphCaptureReady());
#endif
    }

    // =========================================================================
    // Test 3: setStableTokenPointer updates the params token_ids pointer
    // =========================================================================

    TEST_F(Test__PrefillGraphCaptureDynamicParams, StableTokenPointer_Updates)
    {
        auto embed_table = createEmbeddingTable();
        auto output = std::make_unique<FP32Tensor>(
            std::vector<size_t>{4, static_cast<size_t>(d_model_)});
        std::vector<int> tokens_a = {1, 2, 3, 4};
        std::vector<int> tokens_b = {10, 20, 30, 40};

        auto params = makeParams(embed_table.get(), tokens_a.data(), output.get(), 4, DeviceId::cpu());
        EmbeddingStage stage(params);

        // Execute with original pointer
        ASSERT_TRUE(stage.execute(nullptr));
        float first_val = output->data()[0];

        // Update to stable pointer
        stage.setStableTokenPointer(tokens_b.data());

        // Execute with new pointer — should use tokens_b
        ASSERT_TRUE(stage.execute(nullptr));
        float second_val = output->data()[0];

        // Verify different tokens produce different outputs
        // tokens_b[0]=10, tokens_a[0]=1, so row 10 vs row 1 of embed table
        const float *row1 = embed_table->data() + 1 * d_model_;
        const float *row10 = embed_table->data() + 10 * d_model_;
        EXPECT_FLOAT_EQ(first_val, row1[0]);
        EXPECT_FLOAT_EQ(second_val, row10[0]);
    }

    // =========================================================================
    // Test 4: Prelaunch token upload succeeds under graph capture (ROCm)
    // =========================================================================

    TEST_F(Test__PrefillGraphCaptureDynamicParams, PrelaunchTokenUpload_SucceedsUnderCapture)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support compiled";
#else
        int device_count = 0;
        hipGetDeviceCount(&device_count);
        if (device_count <= 0)
            GTEST_SKIP() << "No ROCm device available";
        hipSetDevice(0);

        // Prefill with multiple tokens (seq_len > 1)
        const int seq_len = 8;
        auto embed_table = createEmbeddingTable();
        auto output = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model_)});
        std::vector<int> tokens = {5, 10, 15, 20, 25, 30, 35, 40};

        embed_table->ensureOnDevice(DeviceId::rocm(0));
        output->ensureOnDevice(DeviceId::rocm(0));

        auto params = makeParams(embed_table.get(), tokens.data(), output.get(), seq_len, DeviceId::rocm(0));
        EmbeddingStage stage(params);

        // Create kernel and bind workspace BEFORE first execute
        auto *ws_consumer = stage.getKernelAsWorkspaceConsumer();
        if (!ws_consumer)
            GTEST_SKIP() << "Embedding kernel doesn't implement IWorkspaceConsumer";

        auto reqs = ws_consumer->getWorkspaceRequirements(seq_len, d_model_, 0);
        DeviceWorkspaceManager wsm(DeviceId::rocm(0), 64 * 1024 * 1024);
        ASSERT_TRUE(wsm.allocate(reqs));
        ws_consumer->bindWorkspace(&wsm);

        // Verify basic execution works
        ASSERT_TRUE(stage.execute(nullptr));

        // Prelaunch: updateDynamicParams uploads token_ids to workspace buffer
        stage.updateDynamicParams(/*pos_offset=*/0, /*seq_len=*/seq_len);

        // Now execute under graph capture guard — should succeed because tokens are preloaded
        {
            GraphCaptureGuard guard;
            EXPECT_TRUE(stage.execute(nullptr));
        }
#endif
    }

    // =========================================================================
    // Test 5: Missing preload fails under graph capture (ROCm)
    // =========================================================================

    TEST_F(Test__PrefillGraphCaptureDynamicParams, NoPreload_FailsUnderCapture)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support compiled";
#else
        int device_count = 0;
        hipGetDeviceCount(&device_count);
        if (device_count <= 0)
            GTEST_SKIP() << "No ROCm device available";
        hipSetDevice(0);

        const int seq_len = 4;
        auto embed_table = createEmbeddingTable();
        auto output = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model_)});
        std::vector<int> tokens = {1, 2, 3, 4};

        embed_table->ensureOnDevice(DeviceId::rocm(0));
        output->ensureOnDevice(DeviceId::rocm(0));

        auto params = makeParams(embed_table.get(), tokens.data(), output.get(), seq_len, DeviceId::rocm(0));
        EmbeddingStage stage(params);

        // Create kernel and bind workspace BEFORE execution
        auto *ws_consumer = stage.getKernelAsWorkspaceConsumer();
        if (!ws_consumer)
            GTEST_SKIP() << "Embedding kernel doesn't implement IWorkspaceConsumer";

        auto reqs = ws_consumer->getWorkspaceRequirements(seq_len, d_model_, 0);
        DeviceWorkspaceManager wsm(DeviceId::rocm(0), 64 * 1024 * 1024);
        ASSERT_TRUE(wsm.allocate(reqs));
        ws_consumer->bindWorkspace(&wsm);

        // Execute once outside capture (succeeds with inline H2D)
        ASSERT_TRUE(stage.execute(nullptr));

        // Change tokens WITHOUT calling updateDynamicParams — preloaded data won't match
        std::vector<int> new_tokens = {99, 98, 97, 96};
        stage.setStableTokenPointer(new_tokens.data());

        // Under graph capture: tokens not preloaded → should fail with guard error
        {
            GraphCaptureGuard guard;
            EXPECT_FALSE(stage.execute(nullptr));
        }
#endif
    }

    // =========================================================================
    // Test 6: KVCacheAppendStage advertises replay callback need
    // =========================================================================

    TEST_F(Test__PrefillGraphCaptureDynamicParams, KVCacheAppend_NeedsReplayCallback)
    {
        // KVCacheAppendStage must report needsOnGraphReplayed() == true
        KVCacheAppendStage::Params kv_params{};
        kv_params.device_id = DeviceId::cpu();
        kv_params.layer_idx = 0;
        kv_params.num_tokens = 1;

        KVCacheAppendStage stage(kv_params);
        EXPECT_TRUE(stage.needsOnGraphReplayed());
    }

    TEST_F(Test__PrefillGraphCaptureDynamicParams, KVCacheAppend_ReplayAdvancesByRealPrefillTokens)
    {
        RecordingKVCache kv_cache;
        KVCacheAppendStage::Params kv_params{};
        kv_params.device_id = DeviceId::cpu();
        kv_params.kv_cache = &kv_cache;
        kv_params.layer_idx = 0;
        kv_params.seq_idx = 0;
        kv_params.num_tokens = 128;

        KVCacheAppendStage stage(kv_params);
        ASSERT_TRUE(stage.hasPrefillReplayParams());

        stage.updatePrefillReplayParams(IComputeStage::PrefillReplayParams{
            /*real_seq_len=*/37,
            /*bucket_seq_len=*/128,
            /*token_offset=*/0});
        stage.onGraphReplayed();

        EXPECT_EQ(kv_cache.advanceCalls(), 1);
        EXPECT_EQ(kv_cache.lastLayer(), 0);
        EXPECT_EQ(kv_cache.lastSeqIdx(), 0);
        EXPECT_EQ(kv_cache.lastAdvanceTokens(), 37);
        EXPECT_EQ(kv_cache.get_cached_tokens(0), 37);
    }

    TEST_F(Test__PrefillGraphCaptureDynamicParams, KVCacheAppend_ReplayFallsBackToCapturedTokenCount)
    {
        RecordingKVCache kv_cache;
        KVCacheAppendStage::Params kv_params{};
        kv_params.device_id = DeviceId::cpu();
        kv_params.kv_cache = &kv_cache;
        kv_params.layer_idx = 0;
        kv_params.seq_idx = 0;
        kv_params.num_tokens = 4;

        KVCacheAppendStage stage(kv_params);
        stage.onGraphReplayed();

        EXPECT_EQ(kv_cache.advanceCalls(), 1);
        EXPECT_EQ(kv_cache.lastAdvanceTokens(), 4);
        EXPECT_EQ(kv_cache.get_cached_tokens(0), 4);
    }

    TEST_F(Test__PrefillGraphCaptureDynamicParams, KVCacheAppend_CaptureWithoutHostBookkeepingWaitsForReplay)
    {
        RecordingKVCache kv_cache;
        auto K = std::make_unique<FP32Tensor>(std::vector<size_t>{2, 4});
        auto V = std::make_unique<FP32Tensor>(std::vector<size_t>{2, 4});

        KVCacheAppendStage::Params kv_params{};
        kv_params.device_id = DeviceId::cpu();
        kv_params.kv_cache = &kv_cache;
        kv_params.K = K.get();
        kv_params.V = V.get();
        kv_params.layer_idx = 0;
        kv_params.seq_idx = 0;
        kv_params.num_tokens = 2;

        KVCacheAppendStage stage(kv_params);

        {
            GraphCaptureGuard guard;
            ASSERT_TRUE(stage.execute(nullptr));
        }

        EXPECT_EQ(kv_cache.advanceCalls(), 0);
        EXPECT_EQ(kv_cache.get_cached_tokens(0), 0)
            << "plain capture (prefill-style) must rely on onGraphReplayed()";
    }

    TEST_F(Test__PrefillGraphCaptureDynamicParams, KVCacheAppend_SegmentedCaptureHostBookkeepingAdvancesImmediately)
    {
        RecordingKVCache kv_cache;
        auto K = std::make_unique<FP32Tensor>(std::vector<size_t>{2, 4});
        auto V = std::make_unique<FP32Tensor>(std::vector<size_t>{2, 4});

        KVCacheAppendStage::Params kv_params{};
        kv_params.device_id = DeviceId::cpu();
        kv_params.kv_cache = &kv_cache;
        kv_params.K = K.get();
        kv_params.V = V.get();
        kv_params.layer_idx = 0;
        kv_params.seq_idx = 0;
        kv_params.num_tokens = 2;

        KVCacheAppendStage stage(kv_params);

        {
            GraphCaptureGuard guard(/*host_bookkeeping=*/true);
            ASSERT_TRUE(stage.execute(nullptr));
        }

        EXPECT_EQ(kv_cache.advanceCalls(), 1);
        EXPECT_EQ(kv_cache.lastLayer(), 0);
        EXPECT_EQ(kv_cache.lastSeqIdx(), 0);
        EXPECT_EQ(kv_cache.lastAdvanceTokens(), 2);
        EXPECT_EQ(kv_cache.get_cached_tokens(0), 2)
            << "segmented capture needs logical metadata before downstream captured attention";
    }

    TEST_F(Test__PrefillGraphCaptureDynamicParams, LMHead_ReplayUsesLastRealTokenForPaddedBucket)
    {
        LMHeadStage::Params params{};
        params.device_id = DeviceId::cpu();
        params.seq_len = 768;
        params.d_model = d_model_;
        params.vocab_size = vocab_size_;

        LMHeadStage stage(params);
        ASSERT_TRUE(stage.hasPrefillReplayParams());
        EXPECT_EQ(stage.activationRowOffsetForLogits(), 767);

        stage.updatePrefillReplayParams(IComputeStage::PrefillReplayParams{
            /*real_seq_len=*/513,
            /*bucket_seq_len=*/768,
            /*token_offset=*/0});

        EXPECT_EQ(stage.activationRowOffsetForLogits(), 512);
    }

    TEST_F(Test__PrefillGraphCaptureDynamicParams, LMHead_ReplayClearsLastRealTokenForExactBucket)
    {
        LMHeadStage::Params params{};
        params.device_id = DeviceId::cpu();
        params.seq_len = 768;
        params.d_model = d_model_;
        params.vocab_size = vocab_size_;

        LMHeadStage stage(params);
        stage.updatePrefillReplayParams(IComputeStage::PrefillReplayParams{
            /*real_seq_len=*/513,
            /*bucket_seq_len=*/768,
            /*token_offset=*/0});
        ASSERT_EQ(stage.activationRowOffsetForLogits(), 512);

        stage.updatePrefillReplayParams(IComputeStage::PrefillReplayParams{
            /*real_seq_len=*/768,
            /*bucket_seq_len=*/768,
            /*token_offset=*/0});

        EXPECT_EQ(stage.activationRowOffsetForLogits(), 767);
    }

    TEST_F(Test__PrefillGraphCaptureDynamicParams, LMHead_ExecuteUsesLastRealRowForPaddedBucket)
    {
        const int bucket_seq_len = 8;
        const int real_seq_len = 5;
        const int token_offset = 128;
        auto hidden_states = createDeterministicHiddenStates(bucket_seq_len, d_model_, real_seq_len);
        auto lm_head_weight = createDeterministicLmHeadWeights(vocab_size_, d_model_);
        auto logits = std::make_unique<FP32Tensor>(
            std::vector<size_t>{1, static_cast<size_t>(vocab_size_)},
            DeviceId::cpu());
        auto prepared_lm_head = makePreparedGemmFixture(
            lm_head_weight.get(),
            DeviceId::cpu(),
            "output.weight");

        LMHeadStage::Params params{};
        params.device_id = DeviceId::cpu();
        params.hidden_states = hidden_states.get();
        params.lm_head_weight = lm_head_weight.get();
        params.logits = logits.get();
        params.seq_len = bucket_seq_len;
        params.d_model = d_model_;
        params.vocab_size = vocab_size_;
        params.prepared_ref = prepared_lm_head.ref;
        params.prepared_store = prepared_lm_head.store.get();

        LMHeadStage stage(params);
        stage.updatePrefillReplayParams(IComputeStage::PrefillReplayParams{
            /*real_seq_len=*/real_seq_len,
            /*bucket_seq_len=*/bucket_seq_len,
            /*token_offset=*/token_offset});
        ASSERT_EQ(stage.activationRowOffsetForLogits(), real_seq_len - 1);
        ASSERT_TRUE(stage.execute(nullptr));

        const auto last_real_reference = computeLmHeadReference(
            *hidden_states,
            *lm_head_weight,
            real_seq_len - 1,
            vocab_size_,
            d_model_);
        const auto bucket_tail_reference = computeLmHeadReference(
            *hidden_states,
            *lm_head_weight,
            bucket_seq_len - 1,
            vocab_size_,
            d_model_);

        expectLogitsNearReference(*logits, last_real_reference);
        EXPECT_GT(maxLogitDifference(*logits, bucket_tail_reference), 1e-3f);
    }

    TEST_F(Test__PrefillGraphCaptureDynamicParams, LMHead_ExecuteSwitchesSelectedRowWhenReplayMetadataChanges)
    {
        const int bucket_seq_len = 8;
        const int first_real_seq_len = 3;
        const int second_real_seq_len = 6;
        const int token_offset = 256;
        auto hidden_states = createDeterministicHiddenStates(bucket_seq_len, d_model_, second_real_seq_len);
        auto lm_head_weight = createDeterministicLmHeadWeights(vocab_size_, d_model_);
        auto logits = std::make_unique<FP32Tensor>(
            std::vector<size_t>{1, static_cast<size_t>(vocab_size_)},
            DeviceId::cpu());
        auto prepared_lm_head = makePreparedGemmFixture(
            lm_head_weight.get(),
            DeviceId::cpu(),
            "output.weight");

        LMHeadStage::Params params{};
        params.device_id = DeviceId::cpu();
        params.hidden_states = hidden_states.get();
        params.lm_head_weight = lm_head_weight.get();
        params.logits = logits.get();
        params.seq_len = bucket_seq_len;
        params.d_model = d_model_;
        params.vocab_size = vocab_size_;
        params.prepared_ref = prepared_lm_head.ref;
        params.prepared_store = prepared_lm_head.store.get();

        LMHeadStage stage(params);
        stage.updatePrefillReplayParams(IComputeStage::PrefillReplayParams{
            /*real_seq_len=*/first_real_seq_len,
            /*bucket_seq_len=*/bucket_seq_len,
            /*token_offset=*/token_offset});
        ASSERT_EQ(stage.activationRowOffsetForLogits(), first_real_seq_len - 1);
        ASSERT_TRUE(stage.execute(nullptr));

        const auto first_reference = computeLmHeadReference(
            *hidden_states,
            *lm_head_weight,
            first_real_seq_len - 1,
            vocab_size_,
            d_model_);
        expectLogitsNearReference(*logits, first_reference);
        const std::vector<float> first_logits(
            logits->data(),
            logits->data() + static_cast<size_t>(vocab_size_));

        stage.updatePrefillReplayParams(IComputeStage::PrefillReplayParams{
            /*real_seq_len=*/second_real_seq_len,
            /*bucket_seq_len=*/bucket_seq_len,
            /*token_offset=*/token_offset});
        ASSERT_EQ(stage.activationRowOffsetForLogits(), second_real_seq_len - 1);
        ASSERT_TRUE(stage.execute(nullptr));

        const auto second_reference = computeLmHeadReference(
            *hidden_states,
            *lm_head_weight,
            second_real_seq_len - 1,
            vocab_size_,
            d_model_);
        expectLogitsNearReference(*logits, second_reference);
        EXPECT_GT(maxLogitDifference(*logits, first_logits), 1e-3f);
    }

    // =========================================================================
    // Test 7: ForwardGraphCache caches replay callback stages
    // =========================================================================

    TEST_F(Test__PrefillGraphCaptureDynamicParams, ForwardGraphCache_CachesReplayCallbacks)
    {
        // Build a compute graph with a mix of stages:
        // - 2 stages that need replay callbacks
        // - 2 stages that don't
        ComputeGraph graph;

        auto replay_stage_1 = std::make_unique<MockReplayCallbackStage>(true, "kv_append_0");
        auto replay_stage_2 = std::make_unique<MockReplayCallbackStage>(true, "kv_append_1");
        auto normal_stage_1 = std::make_unique<MockReplayCallbackStage>(false, "gemm_0");
        auto normal_stage_2 = std::make_unique<MockReplayCallbackStage>(false, "norm_0");

        graph.addNode("kv_append_0", std::move(replay_stage_1));
        graph.addNode("gemm_0", std::move(normal_stage_1));
        graph.addNode("kv_append_1", std::move(replay_stage_2));
        graph.addNode("norm_0", std::move(normal_stage_2));

        // Create ForwardGraphCache and populate replay_callback_stages
        ForwardGraphCache cache;
        cache.graph = std::make_unique<ComputeGraph>(std::move(graph));
        cache.valid = true;

        ASSERT_FALSE(cache.replay_callback_stages_cached);

        // Simulate caching logic (same as ForwardExecutionEngine)
        {
            cache.replay_callback_stages.clear();
            const auto &order = cache.graph->getExecutionOrder();
            for (const auto &node_name : order)
            {
                ComputeNode *node = cache.graph->getNode(node_name);
                if (node && node->stage && node->stage->needsOnGraphReplayed())
                    cache.replay_callback_stages.push_back(node->stage.get());
            }
            cache.replay_callback_stages_cached = true;
        }

        EXPECT_TRUE(cache.replay_callback_stages_cached);
        EXPECT_EQ(cache.replay_callback_stages.size(), 2u);

        // Verify the correct stages were collected
        for (auto *stage : cache.replay_callback_stages)
        {
            EXPECT_TRUE(stage->needsOnGraphReplayed());
        }

        // Verify onGraphReplayed() can be called
        for (auto *stage : cache.replay_callback_stages)
        {
            stage->onGraphReplayed();
        }

        // Verify invalidation clears the cache
        cache.invalidate();
        EXPECT_FALSE(cache.replay_callback_stages_cached);
        EXPECT_TRUE(cache.replay_callback_stages.empty());
    }

    TEST_F(Test__PrefillGraphCaptureDynamicParams, ForwardGraphCache_CachesPrefillReplayParamStages)
    {
        ComputeGraph graph;

        auto param_stage_1 = std::make_unique<MockPrefillReplayParamStage>("kv_append_0");
        auto param_stage_2 = std::make_unique<MockPrefillReplayParamStage>("kv_append_1");
        auto normal_stage = std::make_unique<MockReplayCallbackStage>(false, "norm_0");
        auto *raw_param_stage_1 = param_stage_1.get();
        auto *raw_param_stage_2 = param_stage_2.get();

        graph.addNode("kv_append_0", std::move(param_stage_1));
        graph.addNode("norm_0", std::move(normal_stage));
        graph.addNode("kv_append_1", std::move(param_stage_2));

        ForwardGraphCache cache;
        cache.graph = std::make_unique<ComputeGraph>(std::move(graph));
        cache.valid = true;

        ASSERT_FALSE(cache.prefill_replay_param_stages_cached);

        // Simulate the ForwardExecutionEngine stage-list cache, then update all
        // consumers with one padded-bucket replay decision.
        cache.prefill_replay_param_stages.clear();
        const auto &order = cache.graph->getExecutionOrder();
        for (const auto &node_name : order)
        {
            ComputeNode *node = cache.graph->getNode(node_name);
            if (node && node->stage && node->stage->hasPrefillReplayParams())
                cache.prefill_replay_param_stages.push_back(node->stage.get());
        }
        cache.prefill_replay_param_stages_cached = true;

        const IComputeStage::PrefillReplayParams replay_params{
            /*real_seq_len=*/37,
            /*bucket_seq_len=*/128,
            /*token_offset=*/256};
        for (auto *stage : cache.prefill_replay_param_stages)
            stage->updatePrefillReplayParams(replay_params);

        EXPECT_TRUE(cache.prefill_replay_param_stages_cached);
        EXPECT_EQ(cache.prefill_replay_param_stages.size(), 2u);
        EXPECT_EQ(raw_param_stage_1->updateCount(), 1);
        EXPECT_EQ(raw_param_stage_2->updateCount(), 1);
        EXPECT_EQ(raw_param_stage_1->lastParams().real_seq_len, 37);
        EXPECT_EQ(raw_param_stage_1->lastParams().bucket_seq_len, 128);
        EXPECT_EQ(raw_param_stage_1->lastParams().token_offset, 256);

        cache.invalidate();
        EXPECT_FALSE(cache.prefill_replay_param_stages_cached);
        EXPECT_TRUE(cache.prefill_replay_param_stages.empty());
    }

} // anonymous namespace
