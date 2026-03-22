/**
 * @file Test__AttentionStage_DecodeKVOverride.cpp
 * @brief Unit tests locking in the decode-mode KV cache override fix
 *
 * Regression tests for the bug where the forward graph cache reuses a graph
 * built during prefill (cached_tokens=0, K/V wired to activation scratch
 * buffers). During decode (effective_kv_len > seq_len), those buffers only
 * hold the current token's projection — the full KV history lives in the
 * KV cache. Both AttentionComputeStage and FusedAttentionWoStage must
 * override K/V from the KV cache at execute time in decode mode.
 *
 * Bug: short→clear→long→clear→short produces different decode tokens.
 * Fix: Override K/V from cache when effective_kv_len > seq_len.
 */

#include <gtest/gtest.h>
#include <memory>
#include <cmath>
#include <numeric>

#include "execution/compute_stages/ComputeStages.h"
#include "tensors/Tensors.h"
#include "kernels/cpu/CPUKVCache.h"
#include "backends/DeviceId.h"

namespace llaminar2
{
    namespace
    {

        // =====================================================================
        // Mock KV cache — stores FP32 K/V tensors with controllable contents
        // =====================================================================
        class MockKVCache : public ICPUKVCache
        {
        public:
            MockKVCache(int num_layers, int max_seq_len, int kv_dim)
                : num_layers_(num_layers), max_seq_len_(max_seq_len), kv_dim_(kv_dim)
            {
                cached_tokens_.resize(num_layers, 0);
                for (int i = 0; i < num_layers; ++i)
                {
                    k_tensors_.push_back(std::make_unique<FP32Tensor>(
                        std::vector<size_t>{static_cast<size_t>(max_seq_len), static_cast<size_t>(kv_dim)}));
                    v_tensors_.push_back(std::make_unique<FP32Tensor>(
                        std::vector<size_t>{static_cast<size_t>(max_seq_len), static_cast<size_t>(kv_dim)}));
                }
            }

            ActivationPrecision precision() const override { return ActivationPrecision::FP32; }
            int num_layers() const override { return num_layers_; }
            int batch_size() const override { return 1; }
            int max_seq_len() const override { return max_seq_len_; }
            KVCacheLayoutMode layout_mode() const override { return KVCacheLayoutMode::POSITION_MAJOR; }
            TensorLayout kv_layout() const override { return TensorLayout::KV_POS_HEAD_DIM; }

            int get_cached_tokens(int layer, int seq_idx = 0) const override
            {
                (void)seq_idx;
                if (layer < 0 || layer >= num_layers_)
                    return 0;
                return cached_tokens_[layer];
            }

            bool get_kv(int layer, int seq_idx, ITensor **out_k, ITensor **out_v, int *out_kv_len = nullptr) override
            {
                (void)seq_idx;
                if (layer < 0 || layer >= num_layers_)
                {
                    if (out_k)
                        *out_k = nullptr;
                    if (out_v)
                        *out_v = nullptr;
                    if (out_kv_len)
                        *out_kv_len = 0;
                    return false;
                }
                if (out_k)
                    *out_k = k_tensors_[layer].get();
                if (out_v)
                    *out_v = v_tensors_[layer].get();
                if (out_kv_len)
                    *out_kv_len = cached_tokens_[layer];
                return true;
            }
            bool get_kv(int layer, int seq_idx, const ITensor **out_k, const ITensor **out_v, int *out_kv_len = nullptr) const override
            {
                (void)seq_idx;
                if (layer < 0 || layer >= num_layers_)
                {
                    if (out_k)
                        *out_k = nullptr;
                    if (out_v)
                        *out_v = nullptr;
                    if (out_kv_len)
                        *out_kv_len = 0;
                    return false;
                }
                if (out_k)
                    *out_k = k_tensors_[layer].get();
                if (out_v)
                    *out_v = v_tensors_[layer].get();
                if (out_kv_len)
                    *out_kv_len = cached_tokens_[layer];
                return true;
            }

            ITensor *get_k(int layer, int seq_idx = 0) override
            {
                (void)seq_idx;
                return (layer >= 0 && layer < num_layers_) ? k_tensors_[layer].get() : nullptr;
            }
            const ITensor *get_k(int layer, int seq_idx = 0) const override
            {
                (void)seq_idx;
                return (layer >= 0 && layer < num_layers_) ? k_tensors_[layer].get() : nullptr;
            }
            ITensor *get_v(int layer, int seq_idx = 0) override
            {
                (void)seq_idx;
                return (layer >= 0 && layer < num_layers_) ? v_tensors_[layer].get() : nullptr;
            }
            const ITensor *get_v(int layer, int seq_idx = 0) const override
            {
                (void)seq_idx;
                return (layer >= 0 && layer < num_layers_) ? v_tensors_[layer].get() : nullptr;
            }

            bool append_kv(int layer, int seq_idx, const TensorBase *new_k, const TensorBase *new_v) override
            {
                (void)seq_idx;
                (void)new_k;
                (void)new_v;
                if (layer < 0 || layer >= num_layers_)
                    return false;
                cached_tokens_[layer]++;
                return true;
            }
            bool append_kv(int layer, int seq_idx, const TensorBase *new_k, const TensorBase *new_v, int num_tokens) override
            {
                (void)seq_idx;
                (void)new_k;
                (void)new_v;
                if (layer < 0 || layer >= num_layers_)
                    return false;
                cached_tokens_[layer] += num_tokens;
                return true;
            }
            void clear() override
            {
                for (auto &c : cached_tokens_)
                    c = 0;
            }
            void clear_sequence(int, int) override {}
            void clear_layer(int) override {}
            void evict_oldest(int) override {}
            void evict_oldest_from_sequence(int, int) override {}
            DeviceId get_layer_device(int) const override { return DeviceId::cpu(); }
            int get_total_evicted() const override { return 0; }
            void reset_eviction_counter() override {}
            int gather_kv_batched(int, int, TensorBase *, TensorBase *, std::vector<int> &) override { return 0; }
            bool is_sharded() const override { return false; }
            int n_kv_heads() const override { return kv_dim_ / 32; }
            int local_n_kv_heads() const override { return kv_dim_ / 32; }
            int kv_head_start() const override { return 0; }
            int local_kv_dim() const override { return kv_dim_; }

            // Test helpers
            void setCachedTokens(int layer, int count)
            {
                if (layer >= 0 && layer < num_layers_)
                    cached_tokens_[layer] = count;
            }
            FP32Tensor *k_tensor(int layer) { return k_tensors_[layer].get(); }
            FP32Tensor *v_tensor(int layer) { return v_tensors_[layer].get(); }

        private:
            int num_layers_, max_seq_len_, kv_dim_;
            std::vector<int> cached_tokens_;
            std::vector<std::unique_ptr<FP32Tensor>> k_tensors_;
            std::vector<std::unique_ptr<FP32Tensor>> v_tensors_;
        };

        // =====================================================================
        // Fixture
        // =====================================================================
        class Test__AttentionStage_DecodeKVOverride : public ::testing::Test
        {
        protected:
            static constexpr int kNumLayers = 2;
            static constexpr int kMaxSeqLen = 64;
            static constexpr int kNumHeads = 4;
            static constexpr int kNumKVHeads = 2;
            static constexpr int kHeadDim = 32;
            static constexpr int kKVDim = kNumKVHeads * kHeadDim; // 64
            static constexpr int kQDim = kNumHeads * kHeadDim;    // 128

            void SetUp() override
            {
                kv_cache_ = std::make_unique<MockKVCache>(kNumLayers, kMaxSeqLen, kKVDim);

                // Decode-mode Q: [1, n_heads * head_dim]
                Q_ = std::make_unique<FP32Tensor>(std::vector<size_t>{1, static_cast<size_t>(kQDim)});
                output_ = std::make_unique<FP32Tensor>(std::vector<size_t>{1, static_cast<size_t>(kQDim)});

                // Workspace buffers (sized for max)
                workspace_scores_ = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{static_cast<size_t>(kNumHeads * kMaxSeqLen)});
                workspace_context_ = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{static_cast<size_t>(kMaxSeqLen), static_cast<size_t>(kHeadDim)});
                workspace_mask_ = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{static_cast<size_t>(kMaxSeqLen * kMaxSeqLen)});

                // "Stale" activation buffers — these simulate the scratch buffers
                // that would be wired during graph build at prefill time.
                // Fill with zeros to make it detectable when the stage uses them
                // instead of the cache.
                stale_K_ = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{1, static_cast<size_t>(kKVDim)});
                stale_V_ = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{1, static_cast<size_t>(kKVDim)});
                std::memset(stale_K_->mutable_data(), 0, stale_K_->numel() * sizeof(float));
                std::memset(stale_V_->mutable_data(), 0, stale_V_->numel() * sizeof(float));

                // Fill Q with non-zero pattern
                float *q_data = Q_->mutable_data();
                for (size_t i = 0; i < Q_->numel(); ++i)
                    q_data[i] = 0.1f * static_cast<float>((i % 10) + 1);
            }

            /// Fill cache K/V for layer 0 with known non-zero pattern
            void fillCacheWithGoodData(int layer, int num_tokens)
            {
                kv_cache_->setCachedTokens(layer, num_tokens);
                float *k = kv_cache_->k_tensor(layer)->mutable_data();
                float *v = kv_cache_->v_tensor(layer)->mutable_data();
                for (int i = 0; i < num_tokens * kKVDim; ++i)
                {
                    k[i] = 0.05f * static_cast<float>((i % 20) + 1);
                    v[i] = 0.02f * static_cast<float>((i % 20) + 1);
                }
            }

            float outputAbsSum() const
            {
                const float *d = output_->data();
                float sum = 0.0f;
                for (size_t i = 0; i < output_->numel(); ++i)
                    sum += std::abs(d[i]);
                return sum;
            }

            std::unique_ptr<MockKVCache> kv_cache_;
            std::unique_ptr<FP32Tensor> Q_, output_;
            std::unique_ptr<FP32Tensor> stale_K_, stale_V_;
            std::unique_ptr<FP32Tensor> workspace_scores_, workspace_context_, workspace_mask_;
        };

        // =============================================================================
        // AttentionComputeStage Tests
        // =============================================================================

        /**
         * @test Decode mode: K/V overridden from cache even with read_kv_from_cache=false
         *
         * Scenario: Graph was built during prefill (cached_tokens=0).
         * params.K/V point to stale activation scratch buffers (all zeros).
         * Cache has 5 tokens of real data. In decode mode (seq_len=1, kv_len=5),
         * the stage must use cache K/V → output is non-zero.
         *
         * Before the fix: stage used stale K/V (all zeros) → output was garbage/zeros.
         */
        TEST_F(Test__AttentionStage_DecodeKVOverride, AttentionCompute_DecodeOverridesStaleKV)
        {
            const int layer_idx = 0;
            const int cached_tokens = 5;

            fillCacheWithGoodData(layer_idx, cached_tokens);

            // Build params as the graph builder would — K/V point to stale activation buffers
            AttentionComputeStage::Params params;
            params.Q = Q_.get();
            params.K = stale_K_.get(); // Stale! (all zeros)
            params.V = stale_V_.get(); // Stale! (all zeros)
            params.output = output_.get();
            params.batch_size = 1;
            params.seq_len = 1; // Decode: single token
            params.kv_len = 1;  // Static hint from graph build (stale)
            params.n_heads = kNumHeads;
            params.n_kv_heads = kNumKVHeads;
            params.head_dim = kHeadDim;
            params.causal = true;
            params.auto_detect_mode = true;
            params.workspace_scores = workspace_scores_.get();
            params.workspace_context = workspace_context_.get();
            params.workspace_mask = workspace_mask_.get();
            params.kv_cache = kv_cache_.get();
            params.layer_idx = layer_idx;
            params.read_kv_from_cache = false; // NOT explicitly set — the fix must still override
            params.position_offset = cached_tokens - 1;

            auto stage = ComputeStageFactory::createAttentionCompute(params);
            ASSERT_NE(stage, nullptr);

            bool success = stage->execute(nullptr);
            ASSERT_TRUE(success) << "Attention stage should succeed in decode mode";

            // If the fix works, output is non-zero (used cache K/V with real data).
            // If the fix is broken, the stage would use stale_K_/stale_V_ (all zeros)
            // which means softmax(Q*0) = uniform → V*0 = 0 → output is all zeros.
            EXPECT_GT(outputAbsSum(), 0.0f)
                << "Decode output must be non-zero (K/V should come from cache, not stale buffers)";
        }

        /**
         * @test Decode with read_kv_from_cache=true (GPU optimization path)
         *
         * Even when the explicit flag is set, the decode override should still work.
         * This is the standard GPU path.
         */
        TEST_F(Test__AttentionStage_DecodeKVOverride, AttentionCompute_ExplicitReadFromCache)
        {
            const int layer_idx = 0;
            const int cached_tokens = 5;

            fillCacheWithGoodData(layer_idx, cached_tokens);

            AttentionComputeStage::Params params;
            params.Q = Q_.get();
            params.K = stale_K_.get();
            params.V = stale_V_.get();
            params.output = output_.get();
            params.batch_size = 1;
            params.seq_len = 1;
            params.kv_len = 1;
            params.n_heads = kNumHeads;
            params.n_kv_heads = kNumKVHeads;
            params.head_dim = kHeadDim;
            params.causal = true;
            params.auto_detect_mode = true;
            params.workspace_scores = workspace_scores_.get();
            params.workspace_context = workspace_context_.get();
            params.workspace_mask = workspace_mask_.get();
            params.kv_cache = kv_cache_.get();
            params.layer_idx = layer_idx;
            params.read_kv_from_cache = true; // Explicit flag (GPU path)
            params.position_offset = cached_tokens - 1;

            auto stage = ComputeStageFactory::createAttentionCompute(params);
            ASSERT_NE(stage, nullptr);

            bool success = stage->execute(nullptr);
            ASSERT_TRUE(success);
            EXPECT_GT(outputAbsSum(), 0.0f)
                << "Explicit read_kv_from_cache should use cache K/V";
        }

        /**
         * @test Prefill mode: K/V are NOT overridden (effective_kv_len == seq_len)
         *
         * During prefill, the wired K/V are correct (they're the fresh projections
         * for all tokens in the prompt). The override should NOT trigger.
         */
        TEST_F(Test__AttentionStage_DecodeKVOverride, AttentionCompute_PrefillUsesWiredKV)
        {
            const int layer_idx = 0;
            const int seq_len = 4;

            // Cache returns 0 tokens (empty, prefill hasn't finished yet)
            kv_cache_->setCachedTokens(layer_idx, 0);

            // Create properly-sized prefill tensors with non-zero data
            auto Q_prefill = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(kQDim)});
            auto K_wired = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(kKVDim)});
            auto V_wired = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(kKVDim)});
            auto output_prefill = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(kQDim)});

            // Fill with non-zero test data
            for (size_t i = 0; i < Q_prefill->numel(); ++i)
                Q_prefill->mutable_data()[i] = 0.1f * static_cast<float>((i % 10) + 1);
            for (size_t i = 0; i < K_wired->numel(); ++i)
                K_wired->mutable_data()[i] = 0.05f * static_cast<float>((i % 20) + 1);
            for (size_t i = 0; i < V_wired->numel(); ++i)
                V_wired->mutable_data()[i] = 0.02f * static_cast<float>((i % 20) + 1);

            AttentionComputeStage::Params params;
            params.Q = Q_prefill.get();
            params.K = K_wired.get(); // These ARE the correct K/V for prefill
            params.V = V_wired.get();
            params.output = output_prefill.get();
            params.batch_size = 1;
            params.seq_len = seq_len; // Prefill: seq_len == kv_len
            params.kv_len = seq_len;
            params.n_heads = kNumHeads;
            params.n_kv_heads = kNumKVHeads;
            params.head_dim = kHeadDim;
            params.causal = true;
            params.auto_detect_mode = true;
            params.workspace_scores = workspace_scores_.get();
            params.workspace_context = workspace_context_.get();
            params.workspace_mask = workspace_mask_.get();
            params.kv_cache = kv_cache_.get();
            params.layer_idx = layer_idx;
            params.read_kv_from_cache = false;
            params.position_offset = 0;

            auto stage = ComputeStageFactory::createAttentionCompute(params);
            ASSERT_NE(stage, nullptr);

            bool success = stage->execute(nullptr);
            ASSERT_TRUE(success) << "Prefill should succeed using wired K/V";

            // Output should be non-zero since wired K/V have data
            const float *out = output_prefill->data();
            float sum = 0.0f;
            for (size_t i = 0; i < output_prefill->numel(); ++i)
                sum += std::abs(out[i]);
            EXPECT_GT(sum, 0.0f) << "Prefill output should use wired K/V (non-zero)";
        }

        /**
         * @test Determinism: same decode scenario produces identical results
         *
         * Simulates the exact bug scenario: two decode executions with the same
         * inputs must produce bit-identical output.
         */
        TEST_F(Test__AttentionStage_DecodeKVOverride, AttentionCompute_DecodeIsDeterministic)
        {
            const int layer_idx = 0;
            const int cached_tokens = 5;

            fillCacheWithGoodData(layer_idx, cached_tokens);

            auto makeParams = [&]()
            {
                AttentionComputeStage::Params params;
                params.Q = Q_.get();
                params.K = stale_K_.get();
                params.V = stale_V_.get();
                params.output = output_.get();
                params.batch_size = 1;
                params.seq_len = 1;
                params.kv_len = 1;
                params.n_heads = kNumHeads;
                params.n_kv_heads = kNumKVHeads;
                params.head_dim = kHeadDim;
                params.causal = true;
                params.auto_detect_mode = true;
                params.workspace_scores = workspace_scores_.get();
                params.workspace_context = workspace_context_.get();
                params.workspace_mask = workspace_mask_.get();
                params.kv_cache = kv_cache_.get();
                params.layer_idx = layer_idx;
                params.read_kv_from_cache = false;
                params.position_offset = cached_tokens - 1;
                return params;
            };

            // Run 1
            auto stage1 = ComputeStageFactory::createAttentionCompute(makeParams());
            ASSERT_TRUE(stage1->execute(nullptr));
            std::vector<float> result1(output_->data(), output_->data() + output_->numel());

            // Reset output
            std::memset(output_->mutable_data(), 0, output_->numel() * sizeof(float));

            // Run 2 — same inputs, same cache state
            auto stage2 = ComputeStageFactory::createAttentionCompute(makeParams());
            ASSERT_TRUE(stage2->execute(nullptr));
            std::vector<float> result2(output_->data(), output_->data() + output_->numel());

            EXPECT_EQ(result1, result2) << "Two decode executions with same state must be bit-identical";
        }

        /**
         * @test Cache override produces different output than stale K/V
         *
         * Directly verifies that the override changes the computation:
         * run once with KV cache (decode override active), once without.
         * Results must differ because cache has data but stale buffers are zeros.
         */
        TEST_F(Test__AttentionStage_DecodeKVOverride, AttentionCompute_CacheOverrideChangesOutput)
        {
            const int layer_idx = 0;
            const int cached_tokens = 5;

            fillCacheWithGoodData(layer_idx, cached_tokens);

            // Run WITH cache (decode override activates)
            {
                AttentionComputeStage::Params params;
                params.Q = Q_.get();
                params.K = stale_K_.get();
                params.V = stale_V_.get();
                params.output = output_.get();
                params.batch_size = 1;
                params.seq_len = 1;
                params.kv_len = 1;
                params.n_heads = kNumHeads;
                params.n_kv_heads = kNumKVHeads;
                params.head_dim = kHeadDim;
                params.causal = true;
                params.auto_detect_mode = true;
                params.workspace_scores = workspace_scores_.get();
                params.workspace_context = workspace_context_.get();
                params.workspace_mask = workspace_mask_.get();
                params.kv_cache = kv_cache_.get();
                params.layer_idx = layer_idx;
                params.read_kv_from_cache = false;
                params.position_offset = cached_tokens - 1;

                auto stage = ComputeStageFactory::createAttentionCompute(params);
                ASSERT_TRUE(stage->execute(nullptr));
            }
            std::vector<float> with_cache(output_->data(), output_->data() + output_->numel());

            // Reset output
            std::memset(output_->mutable_data(), 0, output_->numel() * sizeof(float));

            // Run WITHOUT cache (uses stale K/V directly)
            {
                AttentionComputeStage::Params params;
                params.Q = Q_.get();
                params.K = stale_K_.get();
                params.V = stale_V_.get();
                params.output = output_.get();
                params.batch_size = 1;
                params.seq_len = 1;
                params.kv_len = 1; // Static: attend to just 1 token (the stale zero buffer)
                params.n_heads = kNumHeads;
                params.n_kv_heads = kNumKVHeads;
                params.head_dim = kHeadDim;
                params.causal = true;
                params.auto_detect_mode = true;
                params.workspace_scores = workspace_scores_.get();
                params.workspace_context = workspace_context_.get();
                params.workspace_mask = workspace_mask_.get();
                params.kv_cache = nullptr; // No cache → uses stale K/V
                params.layer_idx = -1;
                params.position_offset = 0;

                auto stage = ComputeStageFactory::createAttentionCompute(params);
                ASSERT_TRUE(stage->execute(nullptr));
            }
            std::vector<float> without_cache(output_->data(), output_->data() + output_->numel());

            // The outputs MUST be different — cache has real data, stale buffers are zeros
            EXPECT_NE(with_cache, without_cache)
                << "Decode with cache override must produce different output than using stale zero K/V";

            // With cache should have non-zero output, without may be zero or near-zero
            float cache_sum = 0.0f, stale_sum = 0.0f;
            for (size_t i = 0; i < with_cache.size(); ++i)
            {
                cache_sum += std::abs(with_cache[i]);
                stale_sum += std::abs(without_cache[i]);
            }
            EXPECT_GT(cache_sum, stale_sum)
                << "Cache-sourced K/V should produce larger-magnitude output than all-zero K/V";
        }

        /**
         * @test No cache provided: decode uses wired K/V fallback
         *
         * When kv_cache is nullptr, the stage must use whatever K/V was wired
         * at graph construction time.
         */
        TEST_F(Test__AttentionStage_DecodeKVOverride, AttentionCompute_NoCacheFallsBackToWiredKV)
        {
            // Create non-zero K/V (simulating activation buffers with real data)
            auto K_good = std::make_unique<FP32Tensor>(
                std::vector<size_t>{1, static_cast<size_t>(kKVDim)});
            auto V_good = std::make_unique<FP32Tensor>(
                std::vector<size_t>{1, static_cast<size_t>(kKVDim)});
            for (size_t i = 0; i < K_good->numel(); ++i)
                K_good->mutable_data()[i] = 0.1f * static_cast<float>((i % 10) + 1);
            for (size_t i = 0; i < V_good->numel(); ++i)
                V_good->mutable_data()[i] = 0.05f * static_cast<float>((i % 10) + 1);

            AttentionComputeStage::Params params;
            params.Q = Q_.get();
            params.K = K_good.get();
            params.V = V_good.get();
            params.output = output_.get();
            params.batch_size = 1;
            params.seq_len = 1;
            params.kv_len = 1;
            params.n_heads = kNumHeads;
            params.n_kv_heads = kNumKVHeads;
            params.head_dim = kHeadDim;
            params.causal = true;
            params.auto_detect_mode = true;
            params.workspace_scores = workspace_scores_.get();
            params.workspace_context = workspace_context_.get();
            params.workspace_mask = workspace_mask_.get();
            params.kv_cache = nullptr; // No cache
            params.layer_idx = -1;
            params.position_offset = 0;

            auto stage = ComputeStageFactory::createAttentionCompute(params);
            ASSERT_NE(stage, nullptr);

            ASSERT_TRUE(stage->execute(nullptr));
            EXPECT_GT(outputAbsSum(), 0.0f)
                << "Without cache, wired K/V should be used directly";
        }

        // =============================================================================
        // Edge Cases
        // =============================================================================

        /**
         * @test Cache has tokens but get_k/get_v returns nullptr
         *
         * Safety: if the cache reports tokens but the K/V pointers are bad,
         * the stage should fall back to using wired K/V gracefully.
         */
        TEST_F(Test__AttentionStage_DecodeKVOverride, AttentionCompute_CacheReturnsNullFallsBack)
        {
            const int layer_idx = 0;

            // Set cached tokens to trigger override, but use out-of-range layer
            // to cause get_k/get_v to return nullptr
            kv_cache_->setCachedTokens(layer_idx, 5);

            // Create good K/V for the wired path (non-zero)
            auto K_good = std::make_unique<FP32Tensor>(
                std::vector<size_t>{5, static_cast<size_t>(kKVDim)});
            auto V_good = std::make_unique<FP32Tensor>(
                std::vector<size_t>{5, static_cast<size_t>(kKVDim)});
            for (size_t i = 0; i < K_good->numel(); ++i)
                K_good->mutable_data()[i] = 0.1f;
            for (size_t i = 0; i < V_good->numel(); ++i)
                V_good->mutable_data()[i] = 0.05f;

            AttentionComputeStage::Params params;
            params.Q = Q_.get();
            params.K = K_good.get();
            params.V = V_good.get();
            params.output = output_.get();
            params.batch_size = 1;
            params.seq_len = 1;
            params.kv_len = 5;
            params.n_heads = kNumHeads;
            params.n_kv_heads = kNumKVHeads;
            params.head_dim = kHeadDim;
            params.causal = true;
            params.auto_detect_mode = true;
            params.workspace_scores = workspace_scores_.get();
            params.workspace_context = workspace_context_.get();
            params.workspace_mask = workspace_mask_.get();
            params.kv_cache = kv_cache_.get();
            params.layer_idx = 99; // Out of range → cache returns nullptr for K/V
            params.read_kv_from_cache = false;
            params.position_offset = 4;

            auto stage = ComputeStageFactory::createAttentionCompute(params);
            ASSERT_NE(stage, nullptr);

            // The stage should handle this gracefully: NULL from cache means
            // effective_kv_len defaults to seq_len (1), and wired K/V are used.
            // The out-of-range layer also means get_cached_tokens returns 0,
            // so effective_kv_len = params.seq_len = 1, which means no override.
            bool success = stage->execute(nullptr);
            EXPECT_TRUE(success) << "Stage should handle out-of-range cache layer gracefully";
        }

    } // namespace
} // namespace llaminar2
