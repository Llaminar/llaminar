/**
 * @file Test__AttentionComputeStage_DynamicKVLen.cpp
 * @brief Unit tests for AttentionComputeStage dynamic kv_len from KV cache
 * @author David Sanftenberg
 * @date December 2025
 *
 * Tests the declarative graph feature where AttentionComputeStage queries
 * the KV cache at execution time to get the actual kv_len, rather than
 * using a static value baked in at graph construction time.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "execution/compute_stages/ComputeStages.h"
#include "tensors/Tensors.h"
#include "kernels/cpu/CPUKVCache.h"
#include "backends/DeviceId.h"

namespace llaminar2
{
    namespace
    {
        namespace fs = std::filesystem;

        fs::path findRepoRoot()
        {
            std::vector<fs::path> starts;
            starts.push_back(fs::current_path());
            starts.push_back(fs::path(__FILE__));

            for (auto start : starts)
            {
                if (fs::is_regular_file(start))
                    start = start.parent_path();

                for (fs::path candidate = start; !candidate.empty(); candidate = candidate.parent_path())
                {
                    if (fs::exists(candidate / "src/v2/kernels/rocm/attention/ROCmFlashAttentionKernelT.cpp") &&
                        fs::exists(candidate / "tests/v2/CMakeLists.txt"))
                    {
                        return candidate;
                    }

                    if (candidate == candidate.root_path())
                        break;
                }
            }

            return fs::current_path();
        }

        std::string readFile(const fs::path &path)
        {
            std::ifstream input(path);
            if (!input)
                return {};

            std::ostringstream buffer;
            buffer << input.rdbuf();
            return buffer.str();
        }

        /**
         * @brief Mock KV cache for testing dynamic kv_len queries
         *
         * This mock allows us to control what get_cached_tokens() returns
         * to verify AttentionComputeStage queries it at execution time.
         */
        class MockKVCache : public ICPUKVCache
        {
        public:
            MockKVCache(int num_layers, int max_seq_len, int kv_dim)
                : num_layers_(num_layers), max_seq_len_(max_seq_len), kv_dim_(kv_dim)
            {
                // Initialize per-layer token counts to 0
                cached_tokens_.resize(num_layers, 0);

                // Create K/V tensors for each layer
                for (int i = 0; i < num_layers; ++i)
                {
                    k_tensors_.push_back(std::make_unique<FP32Tensor>(
                        std::vector<size_t>{static_cast<size_t>(max_seq_len), static_cast<size_t>(kv_dim)}));
                    v_tensors_.push_back(std::make_unique<FP32Tensor>(
                        std::vector<size_t>{static_cast<size_t>(max_seq_len), static_cast<size_t>(kv_dim)}));
                }
            }

            // ICPUKVCache interface
            ActivationPrecision k_precision() const override { return ActivationPrecision::FP32; }
            int num_layers() const override { return num_layers_; }
            int batch_size() const override { return 1; }
            int max_seq_len() const override { return max_seq_len_; }

            // Layout mode (mock returns POSITION_MAJOR)
            KVCacheLayoutMode layout_mode() const override { return KVCacheLayoutMode::POSITION_MAJOR; }
            TensorLayout kv_layout() const override { return TensorLayout::KV_POS_HEAD_DIM; }

            int get_cached_tokens(int layer, int seq_idx = 0) const override
            {
                (void)seq_idx;
                if (layer < 0 || layer >= num_layers_)
                    return 0;
                return cached_tokens_[layer];
            }

            // Unified KV access (new interface)
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

            bool get_kv_converted(int layer, int seq_idx,
                                  ActivationPrecision target,
                                  ITensor **out_k, ITensor **out_v,
                                  int *out_kv_len = nullptr,
                                  const KVReadParams *rope = nullptr) override
            {
                (void)target;
                (void)rope;
                return get_kv(layer, seq_idx, out_k, out_v, out_kv_len);
            }

            // Legacy individual accessors (deprecated)
            ITensor *get_k(int layer, int seq_idx = 0) override
            {
                (void)seq_idx;
                if (layer < 0 || layer >= num_layers_)
                    return nullptr;
                return k_tensors_[layer].get();
            }

            const ITensor *get_k(int layer, int seq_idx = 0) const override
            {
                (void)seq_idx;
                if (layer < 0 || layer >= num_layers_)
                    return nullptr;
                return k_tensors_[layer].get();
            }

            ITensor *get_v(int layer, int seq_idx = 0) override
            {
                (void)seq_idx;
                if (layer < 0 || layer >= num_layers_)
                    return nullptr;
                return v_tensors_[layer].get();
            }

            const ITensor *get_v(int layer, int seq_idx = 0) const override
            {
                (void)seq_idx;
                if (layer < 0 || layer >= num_layers_)
                    return nullptr;
                return v_tensors_[layer].get();
            }

            bool append_kv(int layer, int seq_idx, const TensorBase *new_k, const TensorBase *new_v) override
            {
                (void)seq_idx;
                (void)new_k;
                (void)new_v;
                if (layer < 0 || layer >= num_layers_)
                    return false;
                // Simulate appending 1 token
                cached_tokens_[layer]++;
                return true;
            }

            bool append_kv(int layer, int seq_idx, const TensorBase *new_k, const TensorBase *new_v,
                           int num_tokens) override
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
                for (auto &count : cached_tokens_)
                {
                    count = 0;
                }
            }

            void clear_sequence(int layer, int seq_idx) override
            {
                (void)layer;
                (void)seq_idx;
            }
            void clear_layer(int layer) override { (void)layer; }
            void evict_oldest(int tokens_to_evict) override { (void)tokens_to_evict; }
            void evict_oldest_from_sequence(int seq_idx, int tokens_to_evict) override
            {
                (void)seq_idx;
                (void)tokens_to_evict;
            }

            DeviceId get_layer_device(int layer) const override
            {
                (void)layer;
                return DeviceId::cpu();
            }

            int get_total_evicted() const override { return 0; }
            void reset_eviction_counter() override {}

            int gather_kv_batched(int layer, int num_sequences, TensorBase *out_k,
                                  TensorBase *out_v, std::vector<int> &out_kv_lens) override
            {
                (void)layer;
                (void)num_sequences;
                (void)out_k;
                (void)out_v;
                (void)out_kv_lens;
                return 0; // Not needed for these tests
            }

            // Test helper: set cached tokens directly
            void setCachedTokens(int layer, int count)
            {
                if (layer >= 0 && layer < num_layers_)
                {
                    cached_tokens_[layer] = count;
                }
            }

            // Sharding interface (mock returns non-sharded)
            bool is_sharded() const override { return false; }
            int n_kv_heads() const override { return kv_dim_ / 32; } // Approximate head dim
            int local_n_kv_heads() const override { return kv_dim_ / 32; }
            int kv_head_start() const override { return 0; }
            int local_kv_dim() const override { return kv_dim_; }

        private:
            int num_layers_;
            int max_seq_len_;
            int kv_dim_;
            std::vector<int> cached_tokens_;
            std::vector<std::unique_ptr<FP32Tensor>> k_tensors_;
            std::vector<std::unique_ptr<FP32Tensor>> v_tensors_;
        };

        /**
         * @brief Test fixture for AttentionComputeStage dynamic kv_len tests
         */
        class Test__AttentionComputeStage_DynamicKVLen : public ::testing::Test
        {
        protected:
            static constexpr int kNumLayers = 4;
            static constexpr int kMaxSeqLen = 256;
            static constexpr int kNumHeads = 4;
            static constexpr int kNumKVHeads = 2;
            static constexpr int kHeadDim = 32;
            static constexpr int kKVDim = kNumKVHeads * kHeadDim; // 64

            void SetUp() override
            {
                // Create mock KV cache
                kv_cache_ = std::make_unique<MockKVCache>(kNumLayers, kMaxSeqLen, kKVDim);

                // Pre-allocate tensors for attention computation
                // Q: [seq_len, n_heads * head_dim]
                // K/V: [kv_len, n_kv_heads * head_dim]
                // For decode: seq_len=1, kv_len=cached_tokens
                const size_t q_dim = kNumHeads * kHeadDim;
                const size_t kv_dim = kNumKVHeads * kHeadDim;

                Q_ = std::make_unique<FP32Tensor>(std::vector<size_t>{1, q_dim});
                output_ = std::make_unique<FP32Tensor>(std::vector<size_t>{1, q_dim});

                // Workspace buffers (sized for max)
                workspace_scores_ = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{static_cast<size_t>(kNumHeads * kMaxSeqLen)});
                workspace_context_ = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{static_cast<size_t>(kMaxSeqLen), static_cast<size_t>(kHeadDim)});
                workspace_mask_ = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{static_cast<size_t>(kMaxSeqLen * kMaxSeqLen)});

                // Initialize Q with some values
                float *q_data = Q_->mutable_data();
                for (size_t i = 0; i < Q_->numel(); ++i)
                {
                    q_data[i] = 0.1f * static_cast<float>(i % 10);
                }
            }

            std::unique_ptr<MockKVCache> kv_cache_;
            std::unique_ptr<FP32Tensor> Q_;
            std::unique_ptr<FP32Tensor> output_;
            std::unique_ptr<FP32Tensor> workspace_scores_;
            std::unique_ptr<FP32Tensor> workspace_context_;
            std::unique_ptr<FP32Tensor> workspace_mask_;
        };

        TEST_F(Test__AttentionComputeStage_DynamicKVLen, ROCmDynamicDecodeAttentionIsGraphCapturable)
        {
            AttentionComputeStage::Params params;
            params.Q = Q_.get();
            params.K = kv_cache_->get_k(0, 0);
            params.V = kv_cache_->get_v(0, 0);
            params.output = output_.get();
            params.batch_size = 1;
            params.seq_len = 1;
            params.kv_len = 127;
            params.n_heads = kNumHeads;
            params.n_kv_heads = kNumKVHeads;
            params.head_dim = kHeadDim;
            params.auto_detect_mode = true;
            params.kv_cache = kv_cache_.get();
            params.layer_idx = 0;
            params.device_id = DeviceId::rocm(0);

            AttentionComputeStage stage(params);
            EXPECT_TRUE(stage.isGraphCapturable());
            EXPECT_FALSE(stage.requiresGraphCaptureSegmentBoundaryBefore());
            EXPECT_FALSE(stage.requiresGraphCaptureSegmentBoundaryAfter());
        }

        TEST_F(Test__AttentionComputeStage_DynamicKVLen, ROCmDynamicDecodeAttentionVariantChangesAtSplitBucket)
        {
            AttentionComputeStage::Params params;
            params.Q = Q_.get();
            params.K = kv_cache_->get_k(0, 0);
            params.V = kv_cache_->get_v(0, 0);
            params.output = output_.get();
            params.batch_size = 1;
            params.seq_len = 1;
            params.kv_len = 64;
            params.n_heads = kNumHeads;
            params.n_kv_heads = kNumKVHeads;
            params.head_dim = kHeadDim;
            params.auto_detect_mode = true;
            params.kv_cache = kv_cache_.get();
            params.layer_idx = 0;
            params.device_id = DeviceId::rocm(0);

            AttentionComputeStage stage(params);

            kv_cache_->setCachedTokens(0, 62); // post-append kv_len=63
            const uint64_t bucket_a = stage.graphCaptureVariantSignature();
            ASSERT_NE(bucket_a, 0u);

            kv_cache_->setCachedTokens(0, 63); // post-append kv_len=64
            const uint64_t bucket_a_edge = stage.graphCaptureVariantSignature();
            EXPECT_EQ(bucket_a_edge, bucket_a);

            kv_cache_->setCachedTokens(0, 64); // post-append kv_len=65
            const uint64_t bucket_b = stage.graphCaptureVariantSignature();
            EXPECT_NE(bucket_b, 0u);
            EXPECT_NE(bucket_b, bucket_a)
                << "ROCm split-K decode must recapture when crossing the 64-token launch bucket";
        }

        TEST_F(Test__AttentionComputeStage_DynamicKVLen, CPUAndROCmPrefillGraphCaptureContractsRemainUnchanged)
        {
            AttentionComputeStage::Params params;
            params.Q = Q_.get();
            params.K = kv_cache_->get_k(0, 0);
            params.V = kv_cache_->get_v(0, 0);
            params.output = output_.get();
            params.batch_size = 1;
            params.seq_len = 8;
            params.kv_len = 8;
            params.n_heads = kNumHeads;
            params.n_kv_heads = kNumKVHeads;
            params.head_dim = kHeadDim;
            params.auto_detect_mode = true;
            params.kv_cache = kv_cache_.get();
            params.layer_idx = 0;

            params.device_id = DeviceId::cpu();
            AttentionComputeStage cpu_stage(params);
            EXPECT_TRUE(cpu_stage.isGraphCapturable());

            params.device_id = DeviceId::rocm(0);
            AttentionComputeStage rocm_prefill_stage(params);
            EXPECT_TRUE(rocm_prefill_stage.isGraphCapturable());
            EXPECT_FALSE(rocm_prefill_stage.requiresGraphCaptureSegmentBoundaryBefore());
            EXPECT_FALSE(rocm_prefill_stage.requiresGraphCaptureSegmentBoundaryAfter());
        }

        /// @brief Reference FP32 single-query GQA attention over a caller-selected KV prefix.
        std::vector<float> referenceSingleQueryGQAAttention(
            const float *q_data,
            const float *k_data,
            const float *v_data,
            int kv_len,
            int num_heads,
            int num_kv_heads,
            int head_dim)
        {
            std::vector<float> output(static_cast<size_t>(num_heads * head_dim), 0.0f);
            std::vector<float> scores(static_cast<size_t>(kv_len), 0.0f);

            const int kv_row_stride = num_kv_heads * head_dim;
            const int gqa_group_size = num_heads / num_kv_heads;
            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

            for (int head = 0; head < num_heads; ++head)
            {
                const int kv_head = head / gqa_group_size;
                const float *q_head = q_data + head * head_dim;

                float max_score = -std::numeric_limits<float>::infinity();
                for (int pos = 0; pos < kv_len; ++pos)
                {
                    const float *k_head = k_data + pos * kv_row_stride + kv_head * head_dim;
                    float dot = 0.0f;
                    for (int dim = 0; dim < head_dim; ++dim)
                    {
                        dot += q_head[dim] * k_head[dim];
                    }
                    scores[static_cast<size_t>(pos)] = dot * scale;
                    max_score = std::max(max_score, scores[static_cast<size_t>(pos)]);
                }

                float denom = 0.0f;
                for (int pos = 0; pos < kv_len; ++pos)
                {
                    const float weight = std::exp(scores[static_cast<size_t>(pos)] - max_score);
                    scores[static_cast<size_t>(pos)] = weight;
                    denom += weight;
                }

                for (int dim = 0; dim < head_dim; ++dim)
                {
                    float value = 0.0f;
                    for (int pos = 0; pos < kv_len; ++pos)
                    {
                        const float *v_head = v_data + pos * kv_row_stride + kv_head * head_dim;
                        value += (scores[static_cast<size_t>(pos)] / denom) * v_head[dim];
                    }
                    output[static_cast<size_t>(head * head_dim + dim)] = value;
                }
            }

            return output;
        }

        /// @brief Returns the largest absolute elementwise difference for equal-sized buffers.
        float maxAbsDiff(const float *actual, const std::vector<float> &expected)
        {
            float max_diff = 0.0f;
            for (size_t i = 0; i < expected.size(); ++i)
            {
                max_diff = std::max(max_diff, std::abs(actual[i] - expected[i]));
            }
            return max_diff;
        }

        /**
         * @test Verify that when kv_cache is provided, execute() queries it dynamically
         *
         * Scenario: Graph is built with static kv_len=4 (before KVCacheAppendStage runs),
         * but by execution time the cache has 5 tokens. AttentionComputeStage should
         * use 5, not 4.
         */
        TEST_F(Test__AttentionComputeStage_DynamicKVLen, UsesKVCacheForDynamicLength)
        {
            const int layer_idx = 0;
            const int static_kv_len = 4; // What graph builder would see
            const int actual_kv_len = 5; // After KVCacheAppendStage runs

            // Simulate KVCacheAppendStage having already run
            kv_cache_->setCachedTokens(layer_idx, actual_kv_len);

            // Initialize K/V in cache with some values
            ITensor *K = kv_cache_->get_k(layer_idx, 0);
            ITensor *V = kv_cache_->get_v(layer_idx, 0);
            float *k_data = dynamic_cast<FP32Tensor *>(K)->mutable_data();
            float *v_data = dynamic_cast<FP32Tensor *>(V)->mutable_data();
            for (int i = 0; i < actual_kv_len * kKVDim; ++i)
            {
                k_data[i] = 0.05f * static_cast<float>(i % 20);
                v_data[i] = 0.02f * static_cast<float>(i % 20);
            }

            // Build params as graph builder would (with STALE kv_len)
            AttentionComputeStage::Params params;
            params.Q = Q_.get();
            params.K = K;
            params.V = V;
            params.output = output_.get();
            params.batch_size = 1;
            params.seq_len = 1;            // Decode mode
            params.kv_len = static_kv_len; // Static hint (stale by execution time)
            params.n_heads = kNumHeads;
            params.n_kv_heads = kNumKVHeads;
            params.head_dim = kHeadDim;
            params.causal = true;
            params.auto_detect_mode = true; // Re-detect with actual kv_len
            params.workspace_scores = workspace_scores_.get();
            params.workspace_context = workspace_context_.get();
            params.workspace_mask = workspace_mask_.get();
            params.kv_cache = kv_cache_.get(); // Enable dynamic query
            params.layer_idx = layer_idx;
            params.position_offset = actual_kv_len - 1; // Query at position 4

            auto stage = ComputeStageFactory::createAttentionCompute(params);
            ASSERT_NE(stage, nullptr);

            // Execute should succeed using actual_kv_len=5, not static_kv_len=4
            bool success = stage->execute(nullptr);
            EXPECT_TRUE(success) << "AttentionComputeStage should succeed with dynamic kv_len";

            // Verify output is not all zeros (attention computed something)
            const float *out_data = output_->data();
            float sum = 0.0f;
            for (size_t i = 0; i < output_->numel(); ++i)
            {
                sum += std::abs(out_data[i]);
            }
            EXPECT_GT(sum, 0.0f) << "Output should have non-zero values after attention";
        }

        /**
         * @test Verify decode uses real cached tokens, not padded KV storage rows
         */
        TEST_F(Test__AttentionComputeStage_DynamicKVLen, DecodeIgnoresHostilePaddedKVRows)
        {
            const int layer_idx = 1;
            const int real_kv_len = 5;
            const int bucket_kv_len = 9;

            kv_cache_->setCachedTokens(layer_idx, real_kv_len);

            ITensor *K = kv_cache_->get_k(layer_idx, 0);
            ITensor *V = kv_cache_->get_v(layer_idx, 0);
            auto *K_fp32 = dynamic_cast<FP32Tensor *>(K);
            auto *V_fp32 = dynamic_cast<FP32Tensor *>(V);
            ASSERT_NE(K_fp32, nullptr);
            ASSERT_NE(V_fp32, nullptr);

            float *q_data = Q_->mutable_data();
            for (int head = 0; head < kNumHeads; ++head)
            {
                for (int dim = 0; dim < kHeadDim; ++dim)
                {
                    q_data[head * kHeadDim + dim] = 0.02f * static_cast<float>((head + 1) * (dim % 7 + 1));
                }
            }

            float *k_data = K_fp32->mutable_data();
            float *v_data = V_fp32->mutable_data();
            for (int pos = 0; pos < real_kv_len; ++pos)
            {
                for (int kv_head = 0; kv_head < kNumKVHeads; ++kv_head)
                {
                    for (int dim = 0; dim < kHeadDim; ++dim)
                    {
                        const int index = pos * kKVDim + kv_head * kHeadDim + dim;
                        const float sign = ((pos + kv_head + dim) % 2 == 0) ? 1.0f : -1.0f;
                        k_data[index] = sign * 0.015f * static_cast<float>(1 + ((pos * 11 + kv_head * 5 + dim) % 9));
                        v_data[index] = 0.025f * static_cast<float>((pos + 1) * (kv_head + 1)) +
                                        0.001f * static_cast<float>(dim);
                    }
                }
            }

            // Padded rows are deliberately attractive to the query and carry large values.
            // A stale bucket-length decode would include them and diverge from the prefix reference.
            for (int pos = real_kv_len; pos < bucket_kv_len; ++pos)
            {
                for (int kv_head = 0; kv_head < kNumKVHeads; ++kv_head)
                {
                    for (int dim = 0; dim < kHeadDim; ++dim)
                    {
                        const int index = pos * kKVDim + kv_head * kHeadDim + dim;
                        k_data[index] = 4.0f;
                        v_data[index] = 25.0f + static_cast<float>(pos - real_kv_len) * 3.0f +
                                        static_cast<float>(kv_head) + 0.1f * static_cast<float>(dim);
                    }
                }
            }

            const std::vector<float> expected_real_prefix = referenceSingleQueryGQAAttention(
                q_data, k_data, v_data,
                real_kv_len, kNumHeads, kNumKVHeads, kHeadDim);
            const std::vector<float> expected_full_bucket = referenceSingleQueryGQAAttention(
                q_data, k_data, v_data,
                bucket_kv_len, kNumHeads, kNumKVHeads, kHeadDim);

            AttentionComputeStage::Params params;
            params.Q = Q_.get();
            params.K = K;
            params.V = V;
            params.output = output_.get();
            params.batch_size = 1;
            params.seq_len = 1;
            params.kv_len = bucket_kv_len; // Intentionally stale padded-bucket hint.
            params.n_heads = kNumHeads;
            params.n_kv_heads = kNumKVHeads;
            params.head_dim = kHeadDim;
            params.causal = false;
            params.auto_detect_mode = true;
            params.workspace_scores = workspace_scores_.get();
            params.workspace_context = workspace_context_.get();
            params.workspace_mask = workspace_mask_.get();
            params.kv_cache = kv_cache_.get();
            params.layer_idx = layer_idx;
            params.position_offset = real_kv_len - 1;

            auto stage = ComputeStageFactory::createAttentionCompute(params);
            ASSERT_NE(stage, nullptr);
            ASSERT_TRUE(stage->execute(nullptr));

            const float *out_data = output_->data();
            const float diff_from_real_prefix = maxAbsDiff(out_data, expected_real_prefix);
            const float diff_from_full_bucket = maxAbsDiff(out_data, expected_full_bucket);

            EXPECT_LT(diff_from_real_prefix, 5e-3f)
                << "Decode should attend only the real KV-cache prefix";
            EXPECT_GT(diff_from_full_bucket, 5.0f)
                << "Hostile padded rows should materially change a stale bucket-length reference";
        }

        /**
         * @test Verify that when kv_cache is nullptr, static kv_len is used
         */
        TEST_F(Test__AttentionComputeStage_DynamicKVLen, FallsBackToStaticKVLen)
        {
            const int static_kv_len = 4;

            // Create standalone K/V tensors (not from cache)
            auto K = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(static_kv_len), static_cast<size_t>(kKVDim)});
            auto V = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(static_kv_len), static_cast<size_t>(kKVDim)});

            // Initialize with values
            float *k_data = K->mutable_data();
            float *v_data = V->mutable_data();
            for (int i = 0; i < static_kv_len * kKVDim; ++i)
            {
                k_data[i] = 0.05f * static_cast<float>(i % 20);
                v_data[i] = 0.02f * static_cast<float>(i % 20);
            }

            // Build params WITHOUT kv_cache
            AttentionComputeStage::Params params;
            params.Q = Q_.get();
            params.K = K.get();
            params.V = V.get();
            params.output = output_.get();
            params.batch_size = 1;
            params.seq_len = 1;
            params.kv_len = static_kv_len; // This should be used
            params.n_heads = kNumHeads;
            params.n_kv_heads = kNumKVHeads;
            params.head_dim = kHeadDim;
            params.causal = true;
            params.auto_detect_mode = true;
            params.workspace_scores = workspace_scores_.get();
            params.workspace_context = workspace_context_.get();
            params.workspace_mask = workspace_mask_.get();
            params.kv_cache = nullptr; // No dynamic query
            params.layer_idx = -1;
            params.position_offset = static_kv_len - 1;

            auto stage = ComputeStageFactory::createAttentionCompute(params);
            ASSERT_NE(stage, nullptr);

            bool success = stage->execute(nullptr);
            EXPECT_TRUE(success) << "AttentionComputeStage should succeed with static kv_len";

            // Verify output is not all zeros
            const float *out_data = output_->data();
            float sum = 0.0f;
            for (size_t i = 0; i < output_->numel(); ++i)
            {
                sum += std::abs(out_data[i]);
            }
            EXPECT_GT(sum, 0.0f) << "Output should have non-zero values";
        }

        /**
         * @test Verify prefill case: when cache has 0 tokens, use seq_len
         *
         * For prefill, the cache starts empty. The stage should use seq_len as kv_len.
         */
        TEST_F(Test__AttentionComputeStage_DynamicKVLen, PrefillWithEmptyCache)
        {
            const int layer_idx = 0;
            const int seq_len = 4;

            // Cache is empty (prefill scenario)
            kv_cache_->setCachedTokens(layer_idx, 0);

            // Create Q sized for prefill
            auto Q_prefill = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(kNumHeads * kHeadDim)});
            auto output_prefill = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(kNumHeads * kHeadDim)});

            // Initialize Q
            float *q_data = Q_prefill->mutable_data();
            for (size_t i = 0; i < Q_prefill->numel(); ++i)
            {
                q_data[i] = 0.1f * static_cast<float>(i % 10);
            }

            // Get K/V from cache and initialize
            ITensor *K = kv_cache_->get_k(layer_idx, 0);
            ITensor *V = kv_cache_->get_v(layer_idx, 0);
            float *k_data = dynamic_cast<FP32Tensor *>(K)->mutable_data();
            float *v_data = dynamic_cast<FP32Tensor *>(V)->mutable_data();
            for (int i = 0; i < seq_len * kKVDim; ++i)
            {
                k_data[i] = 0.05f * static_cast<float>(i % 20);
                v_data[i] = 0.02f * static_cast<float>(i % 20);
            }

            AttentionComputeStage::Params params;
            params.Q = Q_prefill.get();
            params.K = K;
            params.V = V;
            params.output = output_prefill.get();
            params.batch_size = 1;
            params.seq_len = seq_len;
            params.kv_len = seq_len; // For prefill, kv_len == seq_len
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
            params.position_offset = 0;

            auto stage = ComputeStageFactory::createAttentionCompute(params);
            ASSERT_NE(stage, nullptr);

            // With empty cache, dynamic query returns 0, so stage falls back to seq_len
            bool success = stage->execute(nullptr);
            EXPECT_TRUE(success) << "Prefill with empty cache should succeed";

            // Verify output
            const float *out_data = output_prefill->data();
            float sum = 0.0f;
            for (size_t i = 0; i < output_prefill->numel(); ++i)
            {
                sum += std::abs(out_data[i]);
            }
            EXPECT_GT(sum, 0.0f) << "Prefill output should have non-zero values";
        }

        /**
         * @test Verify multi-step decode where kv_len grows each step
         *
         * Simulates the scenario where the graph is reused across decode steps,
         * and the stage queries the cache each time to get the current kv_len.
         */
        TEST_F(Test__AttentionComputeStage_DynamicKVLen, MultiStepDecodeGrowingKVLen)
        {
            const int layer_idx = 0;
            const int initial_tokens = 4; // After prefill

            // Initialize K/V in cache
            ITensor *K = kv_cache_->get_k(layer_idx, 0);
            ITensor *V = kv_cache_->get_v(layer_idx, 0);
            float *k_data = dynamic_cast<FP32Tensor *>(K)->mutable_data();
            float *v_data = dynamic_cast<FP32Tensor *>(V)->mutable_data();
            for (size_t i = 0; i < K->numel(); ++i)
            {
                k_data[i] = 0.05f * static_cast<float>(i % 20);
                v_data[i] = 0.02f * static_cast<float>(i % 20);
            }

            // Create stage with static kv_len hint (doesn't matter, will query dynamically)
            AttentionComputeStage::Params params;
            params.Q = Q_.get();
            params.K = K;
            params.V = V;
            params.output = output_.get();
            params.batch_size = 1;
            params.seq_len = 1;
            params.kv_len = initial_tokens; // Static hint
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

            auto stage = ComputeStageFactory::createAttentionCompute(params);
            ASSERT_NE(stage, nullptr);

            // Simulate 3 decode steps, each time cache grows by 1
            std::vector<float> output_sums;
            for (int step = 0; step < 3; ++step)
            {
                int current_kv_len = initial_tokens + step + 1; // After append
                kv_cache_->setCachedTokens(layer_idx, current_kv_len);

                // Update position_offset to match current position
                // (In real code, this would be done by recreating params or having mutable position)
                // For this test, we verify the stage queries the cache

                bool success = stage->execute(nullptr);
                EXPECT_TRUE(success) << "Decode step " << step << " should succeed";

                // Record output sum for comparison
                const float *out_data = output_->data();
                float sum = 0.0f;
                for (size_t i = 0; i < output_->numel(); ++i)
                {
                    sum += out_data[i]; // Not abs, to detect sign changes
                }
                output_sums.push_back(sum);
            }

            // Different kv_len should produce different outputs
            // (This is a weak check, but better than nothing)
            // At minimum, all steps should produce non-zero output
            for (size_t i = 0; i < output_sums.size(); ++i)
            {
                EXPECT_NE(output_sums[i], 0.0f) << "Decode step " << i << " should produce non-zero output";
            }
        }

        /**
         * @test Verify invalid layer_idx is handled gracefully
         */
        TEST_F(Test__AttentionComputeStage_DynamicKVLen, InvalidLayerIndexFallsBackToStatic)
        {
            const int invalid_layer_idx = 999; // Out of range
            const int static_kv_len = 4;

            // Create standalone K/V tensors
            auto K = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(static_kv_len), static_cast<size_t>(kKVDim)});
            auto V = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(static_kv_len), static_cast<size_t>(kKVDim)});

            float *k_data = K->mutable_data();
            float *v_data = V->mutable_data();
            for (int i = 0; i < static_kv_len * kKVDim; ++i)
            {
                k_data[i] = 0.05f;
                v_data[i] = 0.02f;
            }

            AttentionComputeStage::Params params;
            params.Q = Q_.get();
            params.K = K.get();
            params.V = V.get();
            params.output = output_.get();
            params.batch_size = 1;
            params.seq_len = 1;
            params.kv_len = static_kv_len;
            params.n_heads = kNumHeads;
            params.n_kv_heads = kNumKVHeads;
            params.head_dim = kHeadDim;
            params.causal = true;
            params.auto_detect_mode = true;
            params.workspace_scores = workspace_scores_.get();
            params.workspace_context = workspace_context_.get();
            params.workspace_mask = workspace_mask_.get();
            params.kv_cache = kv_cache_.get();
            params.layer_idx = invalid_layer_idx; // Invalid!
            params.position_offset = static_kv_len - 1;

            auto stage = ComputeStageFactory::createAttentionCompute(params);
            ASSERT_NE(stage, nullptr);

            // Should fall back to static kv_len since cache query returns 0
            bool success = stage->execute(nullptr);
            EXPECT_TRUE(success) << "Should succeed with fallback to static kv_len";
        }

        /**
         * @test Verify that causal mask is built correctly with dynamic kv_len
         *
         * For decode at position P (0-indexed), the query should attend to [0, P].
         * If we have kv_len=5 and position_offset=4, the mask should allow
         * attending to all 5 cached positions.
         */
        TEST_F(Test__AttentionComputeStage_DynamicKVLen, CausalMaskWithDynamicKVLen)
        {
            const int layer_idx = 0;
            const int kv_len = 5;

            kv_cache_->setCachedTokens(layer_idx, kv_len);

            ITensor *K = kv_cache_->get_k(layer_idx, 0);
            ITensor *V = kv_cache_->get_v(layer_idx, 0);

            // Initialize K with distinct values per position so we can verify
            // attention is over all positions
            float *k_data = dynamic_cast<FP32Tensor *>(K)->mutable_data();
            float *v_data = dynamic_cast<FP32Tensor *>(V)->mutable_data();
            for (int pos = 0; pos < kv_len; ++pos)
            {
                for (int d = 0; d < kKVDim; ++d)
                {
                    k_data[pos * kKVDim + d] = 0.1f * static_cast<float>(pos + 1); // pos 0 -> 0.1, pos 4 -> 0.5
                    v_data[pos * kKVDim + d] = 0.2f * static_cast<float>(pos + 1);
                }
            }

            AttentionComputeStage::Params params;
            params.Q = Q_.get();
            params.K = K;
            params.V = V;
            params.output = output_.get();
            params.batch_size = 1;
            params.seq_len = 1;
            params.kv_len = 1; // Wrong static hint (what graph builder might have)
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
            params.position_offset = kv_len - 1; // Position 4 (can attend to 0-4)

            auto stage = ComputeStageFactory::createAttentionCompute(params);
            ASSERT_NE(stage, nullptr);

            bool success = stage->execute(nullptr);
            EXPECT_TRUE(success);

            // Output should reflect attention over all 5 positions
            // (since we're at position 4 with causal mask allowing [0,4])
            const float *out_data = output_->data();
            float sum = 0.0f;
            for (size_t i = 0; i < output_->numel(); ++i)
            {
                sum += std::abs(out_data[i]);
            }
            EXPECT_GT(sum, 0.0f) << "Output should have contributions from all attended positions";
        }

        TEST(Test__AttentionComputeStage_ROCmCaptureContract, CapturedAttentionUsesPreuploadedDeviceParams)
        {
            const fs::path root = findRepoRoot();
            const std::string source =
                readFile(root / "src/v2/kernels/rocm/attention/ROCmFlashAttentionKernelT.cpp");
            ASSERT_FALSE(source.empty());

            const size_t params_start =
                source.find("// Wire device_params for graph-capture replay.");
            ASSERT_NE(params_start, std::string::npos);
            const size_t params_end =
                source.find("void *d_buf = workspace_->getBuffer", params_start);
            ASSERT_NE(params_end, std::string::npos);
            const std::string params_section =
                source.substr(params_start, params_end - params_start);

            const size_t active_capture =
                params_section.find("if (cap_status == hipStreamCaptureStatusActive)");
            ASSERT_NE(active_capture, std::string::npos);
            const size_t non_capture_else = params_section.find("else", active_capture);
            ASSERT_NE(non_capture_else, std::string::npos);

            const std::string active_capture_section =
                params_section.substr(active_capture, non_capture_else - active_capture);
            EXPECT_EQ(active_capture_section.find("setDynamicAttnParams("), std::string::npos)
                << "Captured ROCm attention execution must not mutate or upload dynamic params";
            EXPECT_NE(active_capture_section.find("Attention device params were not ready before HIP graph capture"),
                      std::string::npos);
            EXPECT_NE(active_capture_section.find("dynamic_attn_query_rows_ != query_rows_for_params"),
                      std::string::npos)
                << "Captured graphs may bucket kv_len, but verifier row count must match the prepared params";

            const std::string non_capture_section = params_section.substr(non_capture_else);
            EXPECT_NE(non_capture_section.find("setDynamicAttnParams("), std::string::npos)
                << "Non-captured ROCm attention still prepares params lazily for eager execution";
        }

    } // namespace
} // namespace llaminar2
