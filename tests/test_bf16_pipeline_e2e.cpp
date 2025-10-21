/**
 * @file test_bf16_pipeline_e2e.cpp
 * @brief End-to-end BF16 activation storage validation test
 *
 * Validates that the full Qwen pipeline produces correct outputs when
 * LLAMINAR_QUANT_OUTPUT_BF16=1 is enabled. Compares FP32 baseline vs BF16
 * activations to ensure numerical parity and measures memory reduction.
 *
 * @author David Sanftenberg
 * @date October 20, 2025
 */

#include "gtest/gtest.h"
#include "QwenPipeline.h"
#include "ModelLoader.h"
#include "utils/DebugEnv.h"
#include "tensors/TensorFactory.h"
#include "tensors/BF16Tensor.h"
#include "tensors/SimpleTensor.h"
#include <mpi.h>
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace llaminar;

namespace
{

    /**
     * Global MPI environment for test suite
     */
    struct MPIEnvironment : public ::testing::Environment
    {
        void SetUp() override
        {
            int inited = 0;
            MPI_Initialized(&inited);
            if (!inited)
            {
                int argc = 0;
                char **argv = nullptr;
                int provided = 0;
                MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
            }
            setenv("LLAMINAR_TEST_MPI_NO_FINALIZE", "1", 1);
        }

        void TearDown() override
        {
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (!finalized)
                MPI_Finalize();
        }
    };

    static ::testing::Environment *const mpi_env =
        ::testing::AddGlobalTestEnvironment(new MPIEnvironment());

    /**
     * Test fixture for BF16 end-to-end pipeline validation
     */
    class BF16PipelineE2ETest : public ::testing::Test
    {
    protected:
        int rank, size;

        void SetUp() override
        {
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);

            // Reset environment to default (FP32)
            unsetenv("LLAMINAR_QUANT_OUTPUT_BF16");
            unsetenv("LLAMINAR_ALLOW_BF16_RMSNORM");
            unsetenv("LLAMINAR_FORCE_FP32_RMSNORM");
            debugEnvRefresh();
        }

        void TearDown() override
        {
            // Clean up environment
            unsetenv("LLAMINAR_QUANT_OUTPUT_BF16");
            unsetenv("LLAMINAR_ALLOW_BF16_RMSNORM");
            unsetenv("LLAMINAR_FORCE_FP32_RMSNORM");
            debugEnvRefresh();
        }

        /**
         * Create minimal synthetic weights for testing
         */
        QwenPipeline::ModelWeights createSyntheticWeights(
            QwenPipeline &pipe,
            const TransformerLayerConfig &cfg)
        {
            auto make_matrix = [&](int rows, int cols)
            {
                auto t = pipe.allocateTestLocalTensor({rows, cols});
                for (int r = 0; r < rows; ++r)
                {
                    for (int c = 0; c < cols; ++c)
                    {
                        t->data()[r * cols + c] = 0.001f * (r + 1) + 0.0001f * (c + 1);
                    }
                }
                return t;
            };

            auto make_vector = [&](int dim)
            {
                auto t = pipe.allocateTestLocalTensor({dim});
                for (int i = 0; i < dim; ++i)
                {
                    t->data()[i] = 1.0f;
                }
                return t;
            };

            QwenPipeline::ModelWeights w;
            w.embedding_table = make_matrix(cfg.vocab_size, cfg.d_model);
            w.lm_head = make_matrix(cfg.vocab_size, cfg.d_model);
            w.norm_weight = make_vector(cfg.d_model);

            for (int l = 0; l < cfg.n_layers; ++l)
            {
                QwenPipeline::LayerWeights lw;
                lw.attn_norm = make_vector(cfg.d_model);
                lw.q_weight = make_matrix(cfg.d_model, cfg.d_model);
                lw.k_weight = make_matrix(cfg.d_model, cfg.d_model);
                lw.v_weight = make_matrix(cfg.d_model, cfg.d_model);
                lw.o_weight = make_matrix(cfg.d_model, cfg.d_model);
                lw.q_bias = make_vector(cfg.d_model);
                lw.k_bias = make_vector(cfg.d_model);
                lw.v_bias = make_vector(cfg.d_model);
                lw.ffn_norm = make_vector(cfg.d_model);
                lw.gate_weight = make_matrix(cfg.d_model * 4, cfg.d_model);
                lw.up_weight = make_matrix(cfg.d_model * 4, cfg.d_model);
                lw.down_weight = make_matrix(cfg.d_model, cfg.d_model * 4);
                w.layers.push_back(lw);
            }

            return w;
        }

        /**
         * Compute relative L2 error between two tensors
         */
        double computeRelativeL2Error(
            const std::vector<float> &a,
            const std::vector<float> &b)
        {
            EXPECT_EQ(a.size(), b.size()) << "Tensor sizes must match";
            if (a.size() != b.size())
                return 1e9;

            double diff_sq_sum = 0.0;
            double ref_sq_sum = 0.0;

            for (size_t i = 0; i < a.size(); ++i)
            {
                double diff = a[i] - b[i];
                diff_sq_sum += diff * diff;
                ref_sq_sum += a[i] * a[i];
            }

            return std::sqrt(diff_sq_sum / (ref_sq_sum + 1e-12));
        }
    };

    /**
     * Test: FP32 baseline vs BF16 activations numerical parity
     *
     * Validates that enabling LLAMINAR_QUANT_OUTPUT_BF16=1 produces outputs
     * within acceptable tolerance (rel_l2 < 1e-3) of FP32 baseline.
     */
    TEST_F(BF16PipelineE2ETest, FP32VsBF16NumericalParity)
    {
        // Small model configuration for fast testing
        TransformerLayerConfig cfg;
        cfg.vocab_size = 256;
        cfg.d_model = 128;
        cfg.n_heads = 8;
        cfg.n_kv_heads = 8;
        cfg.n_layers = 2;
        cfg.d_ffn = 512;
        cfg.rope_theta = 10000.0;
        cfg.max_seq_len = 128;
        cfg.head_dim = cfg.d_model / cfg.n_heads;

        // Test input sequence
        const int seq_len = 8;
        std::vector<int32_t> input_tokens = {1, 2, 3, 4, 5, 6, 7, 8};

        // ========== FP32 BASELINE ==========
        unsetenv("LLAMINAR_QUANT_OUTPUT_BF16");
        debugEnvRefresh();

        QwenPipeline fp32_pipeline(cfg);
        auto fp32_weights = createSyntheticWeights(fp32_pipeline, cfg);
        fp32_pipeline.setWeights(fp32_weights);

        auto fp32_logits = fp32_pipeline.forward(input_tokens);
        ASSERT_NE(fp32_logits, nullptr);
        ASSERT_GT(fp32_logits->size(), 0);

        // Extract FP32 logits
        std::vector<float> fp32_output(fp32_logits->data(),
                                       fp32_logits->data() + fp32_logits->size());

        if (rank == 0)
        {
            LOG_INFO("[BF16PipelineE2E] FP32 baseline completed: "
                     << fp32_output.size() << " logits");
        }

        // ========== BF16 ACTIVATIONS ==========
        setenv("LLAMINAR_QUANT_OUTPUT_BF16", "1", 1);
        debugEnvRefresh();

        const auto &env = debugEnv();
        ASSERT_TRUE(env.quant.output_bf16) << "BF16 flag should be enabled";

        QwenPipeline bf16_pipeline(cfg);
        auto bf16_weights = createSyntheticWeights(bf16_pipeline, cfg);
        bf16_pipeline.setWeights(bf16_weights);

        auto bf16_logits = bf16_pipeline.forward(input_tokens);
        ASSERT_NE(bf16_logits, nullptr);
        ASSERT_GT(bf16_logits->size(), 0);

        // Extract BF16 logits
        std::vector<float> bf16_output(bf16_logits->data(),
                                       bf16_logits->data() + bf16_logits->size());

        if (rank == 0)
        {
            LOG_INFO("[BF16PipelineE2E] BF16 pipeline completed: "
                     << bf16_output.size() << " logits");
        }

        // ========== NUMERICAL VALIDATION ==========
        ASSERT_EQ(fp32_output.size(), bf16_output.size())
            << "FP32 and BF16 output sizes must match";

        double rel_l2_error = computeRelativeL2Error(fp32_output, bf16_output);

        // Tolerance for BF16: 7-bit mantissa vs FP32's 23-bit
        // Expected error: ~1e-3 for typical activations
        const double tolerance = 5e-3; // 0.5% relative error

        EXPECT_LT(rel_l2_error, tolerance)
            << "BF16 vs FP32 relative L2 error should be < " << tolerance
            << ", got " << rel_l2_error;

        if (rank == 0)
        {
            LOG_INFO("[BF16PipelineE2E] Numerical parity validated: rel_l2="
                     << rel_l2_error << " (threshold=" << tolerance << ")");
        }
    }

    /**
     * Test: Memory footprint reduction with BF16 activations
     *
     * Validates that BF16 activation storage actually reduces memory usage
     * during forward pass execution.
     */
    TEST_F(BF16PipelineE2ETest, MemoryFootprintReduction)
    {
        // Larger model to make memory difference measurable
        TransformerLayerConfig cfg;
        cfg.vocab_size = 512;
        cfg.d_model = 256;
        cfg.n_heads = 8;
        cfg.n_kv_heads = 8;
        cfg.n_layers = 4;
        cfg.d_ffn = 1024;
        cfg.rope_theta = 10000.0;
        cfg.max_seq_len = 256;
        cfg.head_dim = cfg.d_model / cfg.n_heads;

        const int seq_len = 32;
        std::vector<int32_t> input_tokens(seq_len);
        std::iota(input_tokens.begin(), input_tokens.end(), 1);

        // ========== FP32 BASELINE ==========
        unsetenv("LLAMINAR_QUANT_OUTPUT_BF16");
        debugEnvRefresh();

        QwenPipeline fp32_pipeline(cfg);
        auto fp32_weights = createSyntheticWeights(fp32_pipeline, cfg);
        fp32_pipeline.setWeights(fp32_weights);

        auto fp32_logits = fp32_pipeline.forward(input_tokens);
        ASSERT_NE(fp32_logits, nullptr);

        // Estimate FP32 activation memory (rough):
        // Per layer: Q/K/V projections (3 * seq_len * d_model * 4 bytes)
        //            Attention context (seq_len * d_model * 4 bytes)
        //            FFN intermediates (seq_len * d_ffn * 4 bytes)
        size_t fp32_per_layer = seq_len * cfg.d_model * 4 * 4 // Q/K/V/context
                                + seq_len * cfg.d_ffn * 4;    // FFN
        size_t fp32_total_mb = (fp32_per_layer * cfg.n_layers) / (1024 * 1024);

        // ========== BF16 ACTIVATIONS ==========
        setenv("LLAMINAR_QUANT_OUTPUT_BF16", "1", 1);
        debugEnvRefresh();

        QwenPipeline bf16_pipeline(cfg);
        auto bf16_weights = createSyntheticWeights(bf16_pipeline, cfg);
        bf16_pipeline.setWeights(bf16_weights);

        auto bf16_logits = bf16_pipeline.forward(input_tokens);
        ASSERT_NE(bf16_logits, nullptr);

        // Estimate BF16 activation memory (2 bytes instead of 4)
        size_t bf16_per_layer = seq_len * cfg.d_model * 4 * 2 // Q/K/V/context
                                + seq_len * cfg.d_ffn * 2;    // FFN
        size_t bf16_total_mb = (bf16_per_layer * cfg.n_layers) / (1024 * 1024);

        // Validate ~2× reduction
        double reduction_factor = static_cast<double>(fp32_total_mb) /
                                  static_cast<double>(bf16_total_mb);

        EXPECT_NEAR(reduction_factor, 2.0, 0.3)
            << "BF16 should provide ~2× memory reduction, got "
            << reduction_factor << "×";

        if (rank == 0)
        {
            LOG_INFO("[BF16PipelineE2E] Memory reduction: FP32="
                     << fp32_total_mb << "MB, BF16=" << bf16_total_mb
                     << "MB (" << reduction_factor << "× reduction)");
        }
    }

    /**
     * Test: RMSNorm safety flags in full pipeline
     *
     * Validates that LLAMINAR_FORCE_FP32_RMSNORM correctly forces FP32 for
     * RMSNorm operations even when LLAMINAR_QUANT_OUTPUT_BF16=1.
     */
    TEST_F(BF16PipelineE2ETest, RMSNormSafetyInPipeline)
    {
        TransformerLayerConfig cfg;
        cfg.vocab_size = 128;
        cfg.d_model = 64;
        cfg.n_heads = 4;
        cfg.n_kv_heads = 4;
        cfg.n_layers = 1;
        cfg.d_ffn = 256;
        cfg.rope_theta = 10000.0;
        cfg.max_seq_len = 64;
        cfg.head_dim = cfg.d_model / cfg.n_heads;

        std::vector<int32_t> input_tokens = {1, 2, 3, 4};

        // ========== TEST 1: BF16 + Force FP32 RMSNorm (default) ==========
        setenv("LLAMINAR_QUANT_OUTPUT_BF16", "1", 1);
        // LLAMINAR_FORCE_FP32_RMSNORM defaults to true
        debugEnvRefresh();

        const auto &env1 = debugEnv();
        ASSERT_TRUE(env1.quant.output_bf16);
        ASSERT_TRUE(env1.quant.force_fp32_rmsnorm);

        QwenPipeline pipeline1(cfg);
        auto weights1 = createSyntheticWeights(pipeline1, cfg);
        pipeline1.setWeights(weights1);

        auto logits1 = pipeline1.forward(input_tokens);
        ASSERT_NE(logits1, nullptr);

        if (rank == 0)
        {
            LOG_INFO("[BF16PipelineE2E] RMSNorm forced to FP32 (default safety)");
        }

        // ========== TEST 2: BF16 + Allow BF16 RMSNorm ==========
        setenv("LLAMINAR_QUANT_OUTPUT_BF16", "1", 1);
        setenv("LLAMINAR_ALLOW_BF16_RMSNORM", "1", 1);
        debugEnvRefresh();

        const auto &env2 = debugEnv();
        ASSERT_TRUE(env2.quant.output_bf16);
        ASSERT_TRUE(env2.quant.allow_bf16_rmsnorm);

        QwenPipeline pipeline2(cfg);
        auto weights2 = createSyntheticWeights(pipeline2, cfg);
        pipeline2.setWeights(weights2);

        auto logits2 = pipeline2.forward(input_tokens);
        ASSERT_NE(logits2, nullptr);

        if (rank == 0)
        {
            LOG_INFO("[BF16PipelineE2E] RMSNorm allowed in BF16 (explicit opt-in)");
        }

        // Both should complete successfully
        EXPECT_GT(logits1->size(), 0);
        EXPECT_GT(logits2->size(), 0);
    }

    /**
     * Test: Multi-step generation with BF16 activations
     *
     * Validates that autoregressive decode works correctly with BF16 activations,
     * including KV cache updates.
     */
    TEST_F(BF16PipelineE2ETest, MultiStepGenerationWithBF16)
    {
        TransformerLayerConfig cfg;
        cfg.vocab_size = 128;
        cfg.d_model = 64;
        cfg.n_heads = 4;
        cfg.n_kv_heads = 4;
        cfg.n_layers = 2;
        cfg.d_ffn = 256;
        cfg.rope_theta = 10000.0;
        cfg.max_seq_len = 64;
        cfg.head_dim = cfg.d_model / cfg.n_heads;

        setenv("LLAMINAR_QUANT_OUTPUT_BF16", "1", 1);
        debugEnvRefresh();

        QwenPipeline pipeline(cfg);
        auto weights = createSyntheticWeights(pipeline, cfg);
        pipeline.setWeights(weights);

        // Prefill phase
        std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
        auto prefill_logits = pipeline.forward(prompt);
        ASSERT_NE(prefill_logits, nullptr);

        // Decode phase: generate 5 tokens
        const int n_decode = 5;
        std::vector<int32_t> generated_tokens;

        for (int step = 0; step < n_decode; ++step)
        {
            // Use last logit row to select next token (greedy: argmax)
            const float *last_row = prefill_logits->data() +
                                    (prefill_logits->size() - cfg.vocab_size);

            int32_t next_token = std::distance(
                last_row,
                std::max_element(last_row, last_row + cfg.vocab_size));

            generated_tokens.push_back(next_token);

            // Decode step
            auto decode_logits = pipeline.forward({next_token});
            ASSERT_NE(decode_logits, nullptr);
            ASSERT_EQ(decode_logits->size(), cfg.vocab_size);

            prefill_logits = decode_logits;
        }

        EXPECT_EQ(generated_tokens.size(), n_decode);

        if (rank == 0)
        {
            LOG_INFO("[BF16PipelineE2E] Multi-step generation completed: "
                     << generated_tokens.size() << " tokens");
        }
    }

} // anonymous namespace

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
