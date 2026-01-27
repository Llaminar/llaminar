/**
 * @file Test__WeightManager_LocalTPSlicing.cpp
 * @brief Unit tests for WeightManager LOCAL TP slicing helper methods
 *
 * Tests the static helper methods in WeightManager that support device-aware
 * weight slicing for LOCAL tensor parallelism:
 * - Weight category detection (isQKVWeight, isFFNGateUpWeight, etc.)
 * - Row slicing utility (sliceRowRange)
 *
 * Note: Full WeightManager integration tests that require model loading
 * are in tests/v2/integration/.
 */

#include <gtest/gtest.h>
#include "loaders/WeightManager.h"
#include "tensors/Tensors.h"
#include "TestTensorFactory.h"
#include <memory>
#include <cstring>

namespace llaminar2
{
    namespace test
    {

        // =============================================================================
        // Test Fixture
        // =============================================================================

        class Test__WeightManager_LocalTPSlicing : public ::testing::Test
        {
        protected:
            // Common layer name patterns used in GGUF models
            static constexpr const char *ATTN_Q_LAYER0 = "blk.0.attn_q.weight";
            static constexpr const char *ATTN_K_LAYER0 = "blk.0.attn_k.weight";
            static constexpr const char *ATTN_V_LAYER0 = "blk.0.attn_v.weight";
            static constexpr const char *ATTN_QKV_FUSED = "blk.0.attn_qkv.weight";
            static constexpr const char *ATTN_OUTPUT_LAYER0 = "blk.0.attn_output.weight";

            static constexpr const char *FFN_GATE_LAYER0 = "blk.0.ffn_gate.weight";
            static constexpr const char *FFN_UP_LAYER0 = "blk.0.ffn_up.weight";
            static constexpr const char *FFN_GATE_UP_FUSED = "blk.0.ffn_gate_up.weight";
            static constexpr const char *FFN_DOWN_LAYER0 = "blk.0.ffn_down.weight";

            static constexpr const char *LM_HEAD = "output.weight";
            static constexpr const char *EMBEDDING = "token_embd.weight";
            static constexpr const char *OUTPUT_NORM = "output_norm.weight";
            static constexpr const char *ATTN_NORM_LAYER0 = "blk.0.attn_norm.weight";
            static constexpr const char *FFN_NORM_LAYER0 = "blk.0.ffn_norm.weight";
        };

        // =============================================================================
        // Weight Category Detection Tests
        // =============================================================================

        TEST_F(Test__WeightManager_LocalTPSlicing, IsQKVWeight_MatchesQKV)
        {
            // Should match Q, K, V projections
            EXPECT_TRUE(WeightManager::isQKVWeight(ATTN_Q_LAYER0));
            EXPECT_TRUE(WeightManager::isQKVWeight(ATTN_K_LAYER0));
            EXPECT_TRUE(WeightManager::isQKVWeight(ATTN_V_LAYER0));
            EXPECT_TRUE(WeightManager::isQKVWeight(ATTN_QKV_FUSED));

            // Other layer indices should also match
            EXPECT_TRUE(WeightManager::isQKVWeight("blk.15.attn_q.weight"));
            EXPECT_TRUE(WeightManager::isQKVWeight("blk.27.attn_k.weight"));
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, IsQKVWeight_DoesNotMatchOther)
        {
            // Should NOT match output projection
            EXPECT_FALSE(WeightManager::isQKVWeight(ATTN_OUTPUT_LAYER0));

            // Should NOT match FFN weights
            EXPECT_FALSE(WeightManager::isQKVWeight(FFN_GATE_LAYER0));
            EXPECT_FALSE(WeightManager::isQKVWeight(FFN_UP_LAYER0));
            EXPECT_FALSE(WeightManager::isQKVWeight(FFN_DOWN_LAYER0));

            // Should NOT match norms or embedding
            EXPECT_FALSE(WeightManager::isQKVWeight(ATTN_NORM_LAYER0));
            EXPECT_FALSE(WeightManager::isQKVWeight(EMBEDDING));
            EXPECT_FALSE(WeightManager::isQKVWeight(LM_HEAD));
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, IsFFNGateUpWeight_MatchesGateUp)
        {
            // Should match gate and up projections
            EXPECT_TRUE(WeightManager::isFFNGateUpWeight(FFN_GATE_LAYER0));
            EXPECT_TRUE(WeightManager::isFFNGateUpWeight(FFN_UP_LAYER0));
            EXPECT_TRUE(WeightManager::isFFNGateUpWeight(FFN_GATE_UP_FUSED));

            // Other layers
            EXPECT_TRUE(WeightManager::isFFNGateUpWeight("blk.10.ffn_gate.weight"));
            EXPECT_TRUE(WeightManager::isFFNGateUpWeight("blk.23.ffn_up.weight"));
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, IsFFNGateUpWeight_DoesNotMatchOther)
        {
            // Should NOT match down projection
            EXPECT_FALSE(WeightManager::isFFNGateUpWeight(FFN_DOWN_LAYER0));

            // Should NOT match attention
            EXPECT_FALSE(WeightManager::isFFNGateUpWeight(ATTN_Q_LAYER0));
            EXPECT_FALSE(WeightManager::isFFNGateUpWeight(ATTN_OUTPUT_LAYER0));

            // Should NOT match norms
            EXPECT_FALSE(WeightManager::isFFNGateUpWeight(FFN_NORM_LAYER0));
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, IsFFNDownWeight_MatchesDown)
        {
            EXPECT_TRUE(WeightManager::isFFNDownWeight(FFN_DOWN_LAYER0));
            EXPECT_TRUE(WeightManager::isFFNDownWeight("blk.5.ffn_down.weight"));
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, IsFFNDownWeight_DoesNotMatchOther)
        {
            EXPECT_FALSE(WeightManager::isFFNDownWeight(FFN_GATE_LAYER0));
            EXPECT_FALSE(WeightManager::isFFNDownWeight(FFN_UP_LAYER0));
            EXPECT_FALSE(WeightManager::isFFNDownWeight(ATTN_Q_LAYER0));
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, IsLMHeadWeight_MatchesOutput)
        {
            EXPECT_TRUE(WeightManager::isLMHeadWeight(LM_HEAD));
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, IsLMHeadWeight_DoesNotMatchOther)
        {
            // Should NOT match layer outputs
            EXPECT_FALSE(WeightManager::isLMHeadWeight(ATTN_OUTPUT_LAYER0));

            // Should NOT match embedding (different purpose, even if sometimes tied)
            EXPECT_FALSE(WeightManager::isLMHeadWeight(EMBEDDING));

            // Should NOT match output norm
            EXPECT_FALSE(WeightManager::isLMHeadWeight(OUTPUT_NORM));
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, IsWoWeight_MatchesAttnOutput)
        {
            EXPECT_TRUE(WeightManager::isWoWeight(ATTN_OUTPUT_LAYER0));
            EXPECT_TRUE(WeightManager::isWoWeight("blk.20.attn_output.weight"));
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, IsWoWeight_DoesNotMatchOther)
        {
            // Should NOT match Q/K/V
            EXPECT_FALSE(WeightManager::isWoWeight(ATTN_Q_LAYER0));
            EXPECT_FALSE(WeightManager::isWoWeight(ATTN_K_LAYER0));

            // Should NOT match LM head (different output)
            EXPECT_FALSE(WeightManager::isWoWeight(LM_HEAD));
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, IsEmbeddingWeight_Matches)
        {
            EXPECT_TRUE(WeightManager::isEmbeddingWeight(EMBEDDING));
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, IsEmbeddingWeight_DoesNotMatchOther)
        {
            EXPECT_FALSE(WeightManager::isEmbeddingWeight(LM_HEAD));
            EXPECT_FALSE(WeightManager::isEmbeddingWeight(ATTN_Q_LAYER0));
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, IsOutputNormWeight_Matches)
        {
            EXPECT_TRUE(WeightManager::isOutputNormWeight(OUTPUT_NORM));
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, IsOutputNormWeight_DoesNotMatchOther)
        {
            EXPECT_FALSE(WeightManager::isOutputNormWeight(ATTN_NORM_LAYER0));
            EXPECT_FALSE(WeightManager::isOutputNormWeight(FFN_NORM_LAYER0));
        }

        // =============================================================================
        // sliceRowRange Tests - FP32 Tensors
        // =============================================================================

        TEST_F(Test__WeightManager_LocalTPSlicing, SliceRowRange_FP32_BasicSlice)
        {
            // Create 10x8 FP32 tensor with sequential values for easy verification
            auto tensor = TestTensorFactory::createFP32({10, 8});
            float *data = tensor->mutable_data();
            for (size_t i = 0; i < 80; ++i)
            {
                data[i] = static_cast<float>(i);
            }

            // Slice rows 3-5 (3 rows: indices 3, 4, 5)
            auto shared_tensor = std::shared_ptr<TensorBase>(tensor.release());
            auto sliced = WeightManager::sliceRowRange(shared_tensor, 3, 3);

            ASSERT_NE(sliced, nullptr);
            EXPECT_EQ(sliced->shape().size(), 2);
            EXPECT_EQ(sliced->shape()[0], 3);
            EXPECT_EQ(sliced->shape()[1], 8);

            // Verify values
            // Row 3 starts at index 3*8=24, contains values 24-31
            // Row 4 starts at index 4*8=32, contains values 32-39
            // Row 5 starts at index 5*8=40, contains values 40-47
            const float *sliced_data = sliced->data();
            EXPECT_FLOAT_EQ(sliced_data[0], 24.0f);  // First element of row 3
            EXPECT_FLOAT_EQ(sliced_data[7], 31.0f);  // Last element of row 3
            EXPECT_FLOAT_EQ(sliced_data[8], 32.0f);  // First element of row 4
            EXPECT_FLOAT_EQ(sliced_data[23], 47.0f); // Last element of row 5
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, SliceRowRange_FP32_FirstRows)
        {
            auto tensor = TestTensorFactory::createFP32({10, 8});
            float *data = tensor->mutable_data();
            for (size_t i = 0; i < 80; ++i)
            {
                data[i] = static_cast<float>(i);
            }

            auto shared_tensor = std::shared_ptr<TensorBase>(tensor.release());
            auto sliced = WeightManager::sliceRowRange(shared_tensor, 0, 4);

            ASSERT_NE(sliced, nullptr);
            EXPECT_EQ(sliced->shape()[0], 4);
            EXPECT_EQ(sliced->shape()[1], 8);

            const float *sliced_data = sliced->data();
            EXPECT_FLOAT_EQ(sliced_data[0], 0.0f);
            EXPECT_FLOAT_EQ(sliced_data[31], 31.0f);
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, SliceRowRange_FP32_LastRows)
        {
            auto tensor = TestTensorFactory::createFP32({10, 8});
            float *data = tensor->mutable_data();
            for (size_t i = 0; i < 80; ++i)
            {
                data[i] = static_cast<float>(i);
            }

            auto shared_tensor = std::shared_ptr<TensorBase>(tensor.release());
            auto sliced = WeightManager::sliceRowRange(shared_tensor, 7, 3); // Rows 7, 8, 9

            ASSERT_NE(sliced, nullptr);
            EXPECT_EQ(sliced->shape()[0], 3);
            EXPECT_EQ(sliced->shape()[1], 8);

            const float *sliced_data = sliced->data();
            // Row 7 starts at 7*8=56
            EXPECT_FLOAT_EQ(sliced_data[0], 56.0f);
            // Row 9 ends at 9*8+7=79
            EXPECT_FLOAT_EQ(sliced_data[23], 79.0f);
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, SliceRowRange_FP32_SingleRow)
        {
            auto tensor = TestTensorFactory::createFP32({10, 8});
            float *data = tensor->mutable_data();
            for (size_t i = 0; i < 80; ++i)
            {
                data[i] = static_cast<float>(i);
            }

            auto shared_tensor = std::shared_ptr<TensorBase>(tensor.release());
            auto sliced = WeightManager::sliceRowRange(shared_tensor, 5, 1);

            ASSERT_NE(sliced, nullptr);
            EXPECT_EQ(sliced->shape()[0], 1);
            EXPECT_EQ(sliced->shape()[1], 8);

            const float *sliced_data = sliced->data();
            EXPECT_FLOAT_EQ(sliced_data[0], 40.0f); // First of row 5
            EXPECT_FLOAT_EQ(sliced_data[7], 47.0f); // Last of row 5
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, SliceRowRange_FP32_AllRows)
        {
            auto tensor = TestTensorFactory::createFP32({10, 8});
            float *data = tensor->mutable_data();
            for (size_t i = 0; i < 80; ++i)
            {
                data[i] = static_cast<float>(i);
            }

            auto shared_tensor = std::shared_ptr<TensorBase>(tensor.release());
            auto sliced = WeightManager::sliceRowRange(shared_tensor, 0, 10);

            ASSERT_NE(sliced, nullptr);
            EXPECT_EQ(sliced->shape()[0], 10);
            EXPECT_EQ(sliced->shape()[1], 8);

            // Should be a complete copy
            const float *sliced_data = sliced->data();
            for (size_t i = 0; i < 80; ++i)
            {
                EXPECT_FLOAT_EQ(sliced_data[i], static_cast<float>(i));
            }
        }

        // =============================================================================
        // sliceRowRange Error Cases
        // =============================================================================

        TEST_F(Test__WeightManager_LocalTPSlicing, SliceRowRange_NullTensor_ReturnsNull)
        {
            std::shared_ptr<TensorBase> null_tensor = nullptr;
            auto sliced = WeightManager::sliceRowRange(null_tensor, 0, 1);
            EXPECT_EQ(sliced, nullptr);
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, SliceRowRange_OutOfBounds_ReturnsNull)
        {
            auto tensor = TestTensorFactory::createFP32({10, 8});
            auto shared_tensor = std::shared_ptr<TensorBase>(tensor.release());

            // Start + count exceeds tensor dimensions
            auto sliced = WeightManager::sliceRowRange(shared_tensor, 8, 5);
            EXPECT_EQ(sliced, nullptr);
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, SliceRowRange_StartAtEnd_ReturnsNull)
        {
            auto tensor = TestTensorFactory::createFP32({10, 8});
            auto shared_tensor = std::shared_ptr<TensorBase>(tensor.release());

            // Start at row 10 (out of bounds)
            auto sliced = WeightManager::sliceRowRange(shared_tensor, 10, 1);
            EXPECT_EQ(sliced, nullptr);
        }

        // =============================================================================
        // sliceRowRange - Larger Realistic Sizes
        // =============================================================================

        TEST_F(Test__WeightManager_LocalTPSlicing, SliceRowRange_FP32_LargeMatrix_ProportionalSplit)
        {
            // Simulate splitting a 1024x896 weight matrix for 73%/27% TP split
            const size_t out_dim = 1024;
            const size_t in_dim = 896;

            auto tensor = TestTensorFactory::createFP32Random({out_dim, in_dim});
            auto shared_tensor = std::shared_ptr<TensorBase>(tensor.release());

            // Device 0 gets ~73% = 747 rows (rounded down for alignment)
            // Device 1 gets ~27% = 277 rows
            size_t cuda_rows = 747;
            size_t rocm_rows = out_dim - cuda_rows;

            auto cuda_slice = WeightManager::sliceRowRange(shared_tensor, 0, cuda_rows);
            auto rocm_slice = WeightManager::sliceRowRange(shared_tensor, cuda_rows, rocm_rows);

            ASSERT_NE(cuda_slice, nullptr);
            ASSERT_NE(rocm_slice, nullptr);

            EXPECT_EQ(cuda_slice->shape()[0], cuda_rows);
            EXPECT_EQ(cuda_slice->shape()[1], in_dim);
            EXPECT_EQ(rocm_slice->shape()[0], rocm_rows);
            EXPECT_EQ(rocm_slice->shape()[1], in_dim);

            // Slices should be independent (not sharing memory with original)
            // Verify by checking that all values are valid floats
            auto *cuda_fp32 = dynamic_cast<FP32Tensor *>(cuda_slice.get());
            auto *rocm_fp32 = dynamic_cast<FP32Tensor *>(rocm_slice.get());
            ASSERT_NE(cuda_fp32, nullptr) << "CUDA slice should be FP32";
            ASSERT_NE(rocm_fp32, nullptr) << "ROCm slice should be FP32";
            EXPECT_FALSE(TestTensorFactory::hasNaNOrInf(cuda_fp32));
            EXPECT_FALSE(TestTensorFactory::hasNaNOrInf(rocm_fp32));
        }

        // =============================================================================
        // Weight Category Comprehensive Tests
        // =============================================================================

        TEST_F(Test__WeightManager_LocalTPSlicing, CategoryDetection_AllLayers)
        {
            // Test that category detection works for all layer indices
            for (int layer = 0; layer < 28; ++layer)
            {
                std::string q_name = "blk." + std::to_string(layer) + ".attn_q.weight";
                std::string k_name = "blk." + std::to_string(layer) + ".attn_k.weight";
                std::string v_name = "blk." + std::to_string(layer) + ".attn_v.weight";
                std::string wo_name = "blk." + std::to_string(layer) + ".attn_output.weight";
                std::string gate_name = "blk." + std::to_string(layer) + ".ffn_gate.weight";
                std::string up_name = "blk." + std::to_string(layer) + ".ffn_up.weight";
                std::string down_name = "blk." + std::to_string(layer) + ".ffn_down.weight";
                std::string attn_norm = "blk." + std::to_string(layer) + ".attn_norm.weight";
                std::string ffn_norm = "blk." + std::to_string(layer) + ".ffn_norm.weight";

                EXPECT_TRUE(WeightManager::isQKVWeight(q_name)) << "Layer " << layer << " Q";
                EXPECT_TRUE(WeightManager::isQKVWeight(k_name)) << "Layer " << layer << " K";
                EXPECT_TRUE(WeightManager::isQKVWeight(v_name)) << "Layer " << layer << " V";
                EXPECT_TRUE(WeightManager::isWoWeight(wo_name)) << "Layer " << layer << " Wo";
                EXPECT_TRUE(WeightManager::isFFNGateUpWeight(gate_name)) << "Layer " << layer << " gate";
                EXPECT_TRUE(WeightManager::isFFNGateUpWeight(up_name)) << "Layer " << layer << " up";
                EXPECT_TRUE(WeightManager::isFFNDownWeight(down_name)) << "Layer " << layer << " down";

                // Norms should NOT match any sharded category
                EXPECT_FALSE(WeightManager::isQKVWeight(attn_norm)) << "attn_norm shouldn't match QKV";
                EXPECT_FALSE(WeightManager::isQKVWeight(ffn_norm)) << "ffn_norm shouldn't match QKV";
                EXPECT_FALSE(WeightManager::isFFNGateUpWeight(attn_norm)) << "attn_norm shouldn't match FFN";
                EXPECT_FALSE(WeightManager::isFFNDownWeight(attn_norm)) << "attn_norm shouldn't match down";
            }
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, CategoryDetection_BiasWeights)
        {
            // Bias weights should not match any weight sharding category
            // (biases use isQKVBias, not isQKVWeight)
            EXPECT_FALSE(WeightManager::isQKVWeight("blk.0.attn_q.bias"));
            EXPECT_FALSE(WeightManager::isFFNGateUpWeight("blk.0.ffn_gate.bias"));
            EXPECT_FALSE(WeightManager::isWoWeight("blk.0.attn_output.bias"));
        }

        // =============================================================================
        // isQKVBias Tests - Bias Detection for Column-Parallel 1D Tensors
        // =============================================================================

        TEST_F(Test__WeightManager_LocalTPSlicing, IsQKVBias_MatchesQKVBiases)
        {
            // Q, K, V biases should match
            EXPECT_TRUE(WeightManager::isQKVBias("blk.0.attn_q.bias"));
            EXPECT_TRUE(WeightManager::isQKVBias("blk.0.attn_k.bias"));
            EXPECT_TRUE(WeightManager::isQKVBias("blk.0.attn_v.bias"));

            // Other layers should also match
            EXPECT_TRUE(WeightManager::isQKVBias("blk.15.attn_q.bias"));
            EXPECT_TRUE(WeightManager::isQKVBias("blk.23.attn_k.bias"));
            EXPECT_TRUE(WeightManager::isQKVBias("blk.27.attn_v.bias"));
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, IsQKVBias_DoesNotMatchWeights)
        {
            // Weights should NOT match (2D tensors handled differently)
            EXPECT_FALSE(WeightManager::isQKVBias("blk.0.attn_q.weight"));
            EXPECT_FALSE(WeightManager::isQKVBias("blk.0.attn_k.weight"));
            EXPECT_FALSE(WeightManager::isQKVBias("blk.0.attn_v.weight"));
            EXPECT_FALSE(WeightManager::isQKVBias("blk.0.attn_qkv.weight"));
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, IsQKVBias_DoesNotMatchOtherBiases)
        {
            // Output bias should NOT match (row-parallel, not column-parallel)
            EXPECT_FALSE(WeightManager::isQKVBias("blk.0.attn_output.bias"));

            // FFN biases should NOT match
            EXPECT_FALSE(WeightManager::isQKVBias("blk.0.ffn_gate.bias"));
            EXPECT_FALSE(WeightManager::isQKVBias("blk.0.ffn_up.bias"));
            EXPECT_FALSE(WeightManager::isQKVBias("blk.0.ffn_down.bias"));
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, IsQKVBias_DoesNotMatchNorms)
        {
            // Norms should NOT match (replicated, not sharded)
            EXPECT_FALSE(WeightManager::isQKVBias("blk.0.attn_norm.weight"));
            EXPECT_FALSE(WeightManager::isQKVBias("blk.0.ffn_norm.weight"));
            EXPECT_FALSE(WeightManager::isQKVBias("output_norm.weight"));
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, IsQKVBias_DoesNotMatchEmbeddingOrLMHead)
        {
            EXPECT_FALSE(WeightManager::isQKVBias("token_embd.weight"));
            EXPECT_FALSE(WeightManager::isQKVBias("output.weight"));
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, IsQKVBias_AllLayers)
        {
            // Test bias detection across all layer indices
            for (int layer = 0; layer < 28; ++layer)
            {
                std::string q_bias = "blk." + std::to_string(layer) + ".attn_q.bias";
                std::string k_bias = "blk." + std::to_string(layer) + ".attn_k.bias";
                std::string v_bias = "blk." + std::to_string(layer) + ".attn_v.bias";
                std::string wo_bias = "blk." + std::to_string(layer) + ".attn_output.bias";
                std::string gate_bias = "blk." + std::to_string(layer) + ".ffn_gate.bias";

                EXPECT_TRUE(WeightManager::isQKVBias(q_bias)) << "Layer " << layer << " Q bias";
                EXPECT_TRUE(WeightManager::isQKVBias(k_bias)) << "Layer " << layer << " K bias";
                EXPECT_TRUE(WeightManager::isQKVBias(v_bias)) << "Layer " << layer << " V bias";

                // Wo and FFN biases should NOT match
                EXPECT_FALSE(WeightManager::isQKVBias(wo_bias)) << "Layer " << layer << " Wo bias should NOT match";
                EXPECT_FALSE(WeightManager::isQKVBias(gate_bias)) << "Layer " << layer << " gate bias should NOT match";
            }
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, IsQKVBias_AndIsQKVWeight_AreMutuallyExclusive)
        {
            // Biases should ONLY match isQKVBias, not isQKVWeight
            // Weights should ONLY match isQKVWeight, not isQKVBias
            for (int layer = 0; layer < 5; ++layer)
            {
                std::string q_weight = "blk." + std::to_string(layer) + ".attn_q.weight";
                std::string q_bias = "blk." + std::to_string(layer) + ".attn_q.bias";

                // Weight: matches isQKVWeight, not isQKVBias
                EXPECT_TRUE(WeightManager::isQKVWeight(q_weight));
                EXPECT_FALSE(WeightManager::isQKVBias(q_weight));

                // Bias: matches isQKVBias, not isQKVWeight
                EXPECT_TRUE(WeightManager::isQKVBias(q_bias));
                EXPECT_FALSE(WeightManager::isQKVWeight(q_bias));
            }
        }

    } // namespace test
} // namespace llaminar2
