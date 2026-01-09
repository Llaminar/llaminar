/**
 * @file Test__HybridQ16_Layer0_Dataflow.cpp
 * @brief Diagnostic test for HybridQ16 layer 0 data flow
 *
 * This test traces the exact data flow through layer 0 in HybridQ16 mode:
 * 1. Embedding → Q16_1 residual
 * 2. Q16_1 residual → RMSNorm → FP32 normalized
 * 3. FP32 normalized → QKV projection → Q8_1 Q/K/V
 * 4. Q8_1 Q/K/V → Fused Attention + Wo → output + Q16_1 residual fusion
 *
 * At each stage, we dump tensor statistics and compare with FP32 reference.
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>
#include <memory>
#include <iomanip>

#include "backends/DeviceId.h"
#include "execution/compute_stages/ComputeStages.h"
#include "execution/DeviceContext.h"
#include "backends/ComputeBackend.h"
#include "tensors/Tensors.h"
#include "kernels/KernelFactory.h"
#include "utils/Logger.h"
#include "../../mocks/MockComputeStage.h"

namespace
{
    using namespace llaminar2;
    using namespace llaminar::v2::kernels;
    using llaminar2::testing::MockDeviceContext;

    // ============================================================================
    // Tensor Statistics Helper
    // ============================================================================

    struct TensorStats
    {
        double min_val = 0.0;
        double max_val = 0.0;
        double mean = 0.0;
        double std_dev = 0.0;
        bool has_nan = false;
        bool has_inf = false;
        size_t nan_count = 0;
        size_t inf_count = 0;

        void print(const std::string &name) const
        {
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "[" << name << "] "
                      << "min=" << min_val
                      << " max=" << max_val
                      << " mean=" << mean
                      << " std=" << std_dev;
            if (has_nan)
                std::cout << " NaN=" << nan_count;
            if (has_inf)
                std::cout << " Inf=" << inf_count;
            std::cout << std::endl;
        }
    };

    TensorStats computeStats(const float *data, size_t n)
    {
        TensorStats stats;
        if (!data || n == 0)
            return stats;

        stats.min_val = data[0];
        stats.max_val = data[0];
        double sum = 0.0;
        double sum_sq = 0.0;

        for (size_t i = 0; i < n; ++i)
        {
            float val = data[i];
            if (std::isnan(val))
            {
                stats.has_nan = true;
                stats.nan_count++;
                continue;
            }
            if (std::isinf(val))
            {
                stats.has_inf = true;
                stats.inf_count++;
                continue;
            }
            stats.min_val = std::min(stats.min_val, static_cast<double>(val));
            stats.max_val = std::max(stats.max_val, static_cast<double>(val));
            sum += val;
            sum_sq += static_cast<double>(val) * val;
        }

        size_t valid_count = n - stats.nan_count - stats.inf_count;
        if (valid_count > 0)
        {
            stats.mean = sum / valid_count;
            double variance = (sum_sq / valid_count) - (stats.mean * stats.mean);
            stats.std_dev = variance > 0 ? std::sqrt(variance) : 0.0;
        }

        return stats;
    }

    double cosineSimilarity(const float *a, const float *b, size_t n)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            if (std::isnan(a[i]) || std::isnan(b[i]) ||
                std::isinf(a[i]) || std::isinf(b[i]))
                continue;
            dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        if (norm_a < 1e-12 || norm_b < 1e-12)
            return 0.0;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    // ============================================================================
    // Test Fixture
    // ============================================================================

    class Test__HybridQ16_Layer0_Dataflow : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Qwen2 0.5B dimensions
            vocab_size_ = 1000; // Smaller for testing
            d_model_ = 896;
            n_heads_ = 14;
            n_kv_heads_ = 2;
            head_dim_ = 64;
            rms_norm_eps_ = 1e-6f;

            // Initialize DeviceManager to enumerate devices (required by KernelFactory)
            DeviceManager::instance().initialize(-1); // -1 = no NUMA filtering

            // Create device context for stages that require it
            ctx_ = std::make_unique<CPUDeviceContext>(DeviceId::cpu(), 4);
        }

        // Create random FP32 embedding table
        std::unique_ptr<FP32Tensor> createEmbeddingTable()
        {
            auto table = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(vocab_size_), static_cast<size_t>(d_model_)});

            float *data = table->mutable_data();
            std::mt19937 gen(42);
            std::normal_distribution<float> dist(0.0f, 0.02f);
            for (size_t i = 0; i < static_cast<size_t>(vocab_size_ * d_model_); ++i)
            {
                data[i] = dist(gen);
            }
            return table;
        }

        // Create random FP32 RMSNorm weights (gamma)
        std::unique_ptr<FP32Tensor> createRMSNormWeights()
        {
            auto weights = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(d_model_)});

            float *data = weights->mutable_data();
            std::mt19937 gen(123);
            std::normal_distribution<float> dist(1.0f, 0.1f);
            for (int i = 0; i < d_model_; ++i)
            {
                data[i] = dist(gen);
            }
            return weights;
        }

        int vocab_size_;
        int d_model_;
        int n_heads_;
        int n_kv_heads_;
        int head_dim_;
        float rms_norm_eps_;
        std::unique_ptr<CPUDeviceContext> ctx_;
    };

    // ============================================================================
    // Test: Embedding → Q16_1 → RMSNorm → FP32
    // ============================================================================

    TEST_F(Test__HybridQ16_Layer0_Dataflow, Embedding_To_RMSNorm_DataIntegrity)
    {
        const int seq_len = 9; // Same as parity test
        std::vector<int> token_ids = {1, 42, 100, 200, 300, 400, 500, 600, 700};

        std::cout << "\n=== Testing Embedding → Q16_1 → RMSNorm → FP32 ===\n";

        // Create embedding table
        auto embed_table = createEmbeddingTable();
        auto rms_weights = createRMSNormWeights();

        // Create buffers
        auto q16_residual = std::make_unique<Q16_1Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model_)});
        auto fp32_hidden = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model_)});
        auto fp32_normalized = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model_)});

        // ========================================
        // Step 1: Embedding → Q16_1 (HybridQ16 path)
        // ========================================
        std::cout << "\n--- Step 1: Embedding → Q16_1 ---\n";
        {
            EmbeddingStage::Params params;
            params.embed_table = embed_table.get();
            params.token_ids = token_ids.data();
            params.output = q16_residual.get();
            params.num_tokens = seq_len;
            params.d_model = d_model_;
            params.vocab_size = vocab_size_;
            params.device_id = DeviceId::cpu();

            auto stage = ComputeStageFactory::createEmbedding(params);
            ASSERT_TRUE(stage->execute(ctx_.get()));
        }

        // Dequantize Q16_1 to FP32 for analysis
        std::vector<float> q16_dequant(seq_len * d_model_);
        for (int t = 0; t < seq_len; ++t)
        {
            q16_residual->to_fp32_row(t, q16_dequant.data() + t * d_model_);
        }
        auto q16_stats = computeStats(q16_dequant.data(), q16_dequant.size());
        q16_stats.print("Q16_1 Residual (after embedding)");

        // Compare with direct FP32 embedding
        {
            EmbeddingStage::Params params;
            params.embed_table = embed_table.get();
            params.token_ids = token_ids.data();
            params.output = fp32_hidden.get();
            params.num_tokens = seq_len;
            params.d_model = d_model_;
            params.vocab_size = vocab_size_;
            params.device_id = DeviceId::cpu();

            auto stage = ComputeStageFactory::createEmbedding(params);
            ASSERT_TRUE(stage->execute(ctx_.get()));
        }
        auto fp32_embed_stats = computeStats(fp32_hidden->data(), seq_len * d_model_);
        fp32_embed_stats.print("FP32 Embedding (reference)");

        double embed_cosine = cosineSimilarity(q16_dequant.data(), fp32_hidden->data(), seq_len * d_model_);
        std::cout << "Embedding Q16_1 vs FP32 cosine: " << embed_cosine << std::endl;
        EXPECT_GT(embed_cosine, 0.9999) << "Embedding should have near-perfect fidelity";

        // ========================================
        // Step 2: Q16_1 Residual → RMSNorm → FP32
        // ========================================
        std::cout << "\n--- Step 2: Q16_1 → RMSNorm → FP32 ---\n";
        {
            RMSNormStage::Params params;
            params.input = q16_residual.get();
            params.output = fp32_normalized.get();
            params.gamma = rms_weights.get();
            params.eps = rms_norm_eps_;
            params.seq_len = seq_len;
            params.device_id = DeviceId::cpu();

            auto stage = ComputeStageFactory::createRMSNorm(params);
            ASSERT_TRUE(stage->execute(ctx_.get()));
        }

        auto norm_stats = computeStats(fp32_normalized->data(), seq_len * d_model_);
        norm_stats.print("FP32 Normalized (from Q16_1)");

        EXPECT_FALSE(norm_stats.has_nan) << "RMSNorm output should not have NaN";
        EXPECT_FALSE(norm_stats.has_inf) << "RMSNorm output should not have Inf";

        // Compare with FP32 → FP32 RMSNorm
        auto fp32_norm_ref = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model_)});
        {
            RMSNormStage::Params params;
            params.input = fp32_hidden.get();
            params.output = fp32_norm_ref.get();
            params.gamma = rms_weights.get();
            params.eps = rms_norm_eps_;
            params.seq_len = seq_len;
            params.device_id = DeviceId::cpu();

            auto stage = ComputeStageFactory::createRMSNorm(params);
            ASSERT_TRUE(stage->execute(ctx_.get()));
        }

        auto fp32_norm_stats = computeStats(fp32_norm_ref->data(), seq_len * d_model_);
        fp32_norm_stats.print("FP32 Normalized (reference)");

        double norm_cosine = cosineSimilarity(fp32_normalized->data(), fp32_norm_ref->data(), seq_len * d_model_);
        std::cout << "RMSNorm Q16_1→FP32 vs FP32→FP32 cosine: " << norm_cosine << std::endl;
        EXPECT_GT(norm_cosine, 0.999) << "RMSNorm should have high fidelity";

        // ========================================
        // Step 3: Print first few values for manual inspection
        // ========================================
        std::cout << "\n--- First 8 values comparison ---\n";
        std::cout << std::setw(10) << "Index"
                  << std::setw(15) << "Q16_1→Norm"
                  << std::setw(15) << "FP32→Norm"
                  << std::setw(15) << "Diff" << std::endl;
        for (int i = 0; i < 8; ++i)
        {
            float q16_val = fp32_normalized->data()[i];
            float fp32_val = fp32_norm_ref->data()[i];
            std::cout << std::setw(10) << i
                      << std::setw(15) << q16_val
                      << std::setw(15) << fp32_val
                      << std::setw(15) << (q16_val - fp32_val) << std::endl;
        }
    }

    // ============================================================================
    // Test: Verify Q16_1 blocks are correctly populated
    // ============================================================================

    TEST_F(Test__HybridQ16_Layer0_Dataflow, Q16_1_Block_Structure_Verification)
    {
        const int seq_len = 2;
        std::vector<int> token_ids = {42, 100};

        std::cout << "\n=== Q16_1 Block Structure Verification ===\n";

        auto embed_table = createEmbeddingTable();
        auto q16_residual = std::make_unique<Q16_1Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model_)});

        // Run embedding
        {
            EmbeddingStage::Params params;
            params.embed_table = embed_table.get();
            params.token_ids = token_ids.data();
            params.output = q16_residual.get();
            params.num_tokens = seq_len;
            params.d_model = d_model_;
            params.vocab_size = vocab_size_;
            params.device_id = DeviceId::cpu();

            auto stage = ComputeStageFactory::createEmbedding(params);
            ASSERT_TRUE(stage->execute(ctx_.get()));
        }

        // Get Q16_1 blocks and inspect
        const Q16_1Block *blocks = q16_residual->as_block_32();
        ASSERT_NE(blocks, nullptr);

        const size_t blocks_per_row = (d_model_ + 31) / 32;
        std::cout << "blocks_per_row: " << blocks_per_row << std::endl;

        // Check first block of first row
        const Q16_1Block &block0 = blocks[0];
        std::cout << "\n--- Block 0 (row 0, col 0-31) ---\n";
        std::cout << "d (scale): " << block0.d << std::endl;
        std::cout << "sum_qs: " << block0.sum_qs << std::endl;
        std::cout << "First 8 qs: ";
        for (int i = 0; i < 8; ++i)
        {
            std::cout << block0.qs[i] << " ";
        }
        std::cout << std::endl;

        // Verify d is not zero (would indicate all-zero block)
        EXPECT_NE(block0.d, 0.0f) << "Scale should not be zero for non-zero embeddings";

        // Verify some qs values are non-zero
        int nonzero_qs = 0;
        for (int i = 0; i < 32; ++i)
        {
            if (block0.qs[i] != 0)
                nonzero_qs++;
        }
        EXPECT_GT(nonzero_qs, 0) << "Some quantized values should be non-zero";

        // Dequantize and compare with expected
        std::vector<float> dequant(32);
        float *expected = embed_table->mutable_data() + token_ids[0] * d_model_;

        // Manual dequant for block 0
        for (int i = 0; i < 32; ++i)
        {
            dequant[i] = static_cast<float>(block0.qs[i]) * block0.d;
        }

        std::cout << "\n--- Dequantized vs Expected (first 8) ---\n";
        std::cout << std::setw(10) << "Index"
                  << std::setw(15) << "Dequant"
                  << std::setw(15) << "Expected"
                  << std::setw(15) << "Diff" << std::endl;
        for (int i = 0; i < 8; ++i)
        {
            std::cout << std::setw(10) << i
                      << std::setw(15) << dequant[i]
                      << std::setw(15) << expected[i]
                      << std::setw(15) << (dequant[i] - expected[i]) << std::endl;
        }

        double cosine = cosineSimilarity(dequant.data(), expected, 32);
        std::cout << "Block 0 dequant vs expected cosine: " << cosine << std::endl;
        EXPECT_GT(cosine, 0.999) << "Dequantized values should match expected";
    }

} // anonymous namespace
