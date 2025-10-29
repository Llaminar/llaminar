/**
 * @file Test__DequantEquivalency.cpp
 * @brief Integration tests verifying bit-exact equivalence between Llaminar IBlockDecoder
 *        and llama.cpp dequantization routines
 *
 * Purpose: Ensure Llaminar's quantized tensor dequantization produces identical FP32
 *          outputs to llama.cpp's reference implementations (GGML library).
 *
 * Test Strategy:
 * 1. Load quantized weights from GGUF files (known-good source)
 * 2. Dequantize using llama.cpp's dequantize_row_* functions
 * 3. Dequantize using Llaminar's IBlockDecoder::decode_block_at
 * 4. Compare outputs element-by-element (bit-exact or tolerance-based)
 *
 * Note: User mentioned potential transpose in IBlockDecoder - tests will verify/account for this.
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "../../../src/v2/tensors/Tensors.h"
#include "../../../src/v2/loaders/ModelLoader.h"
#include <vector>
#include <cmath>
#include <memory>
#include <iostream>

// Include llama.cpp dequant headers
extern "C"
{
#include "ggml-quants.h"
}

namespace llaminar2
{
    namespace test
    {

        /**
         * @brief Test fixture for dequant equivalency tests
         */
        class DequantEquivalencyTest : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
                // Default model path (can be overridden by test)
                model_path_ = "/workspaces/llaminar/models/qwen2.5-0.5b-instruct-q8_0.gguf";
            }

            /**
             * @brief Compare two float arrays with configurable tolerance
             * @param llaminar Output from Llaminar IBlockDecoder
             * @param llamacpp Output from llama.cpp dequantize_row_*
             * @param count Number of elements
             * @param tolerance Absolute tolerance (default 1e-6 for FP32)
             * @param check_transpose If true, also check transposed comparison
             * @return true if arrays match within tolerance
             */
            bool compareOutputs(const float *llaminar, const float *llamacpp, size_t count,
                                float tolerance = 1e-6f, bool check_transpose = false)
            {
                size_t mismatch_count = 0;
                float max_abs_diff = 0.0f;
                float max_rel_diff = 0.0f;

                for (size_t i = 0; i < count; ++i)
                {
                    float abs_diff = std::fabs(llaminar[i] - llamacpp[i]);
                    max_abs_diff = std::max(max_abs_diff, abs_diff);

                    if (std::fabs(llamacpp[i]) > 1e-9f)
                    {
                        float rel_diff = abs_diff / std::fabs(llamacpp[i]);
                        max_rel_diff = std::max(max_rel_diff, rel_diff);
                    }

                    if (abs_diff > tolerance)
                    {
                        ++mismatch_count;
                        if (mismatch_count <= 5)
                        { // Print first 5 mismatches
                            std::cout << "Mismatch at [" << i << "]: "
                                      << "Llaminar=" << llaminar[i] << ", "
                                      << "llama.cpp=" << llamacpp[i] << ", "
                                      << "abs_diff=" << abs_diff << std::endl;
                        }
                    }
                }

                std::cout << "Comparison results:" << std::endl;
                std::cout << "  Total elements: " << count << std::endl;
                std::cout << "  Mismatches: " << mismatch_count << std::endl;
                std::cout << "  Max abs diff: " << max_abs_diff << std::endl;
                std::cout << "  Max rel diff: " << max_rel_diff << std::endl;

                // If direct comparison fails and transpose check requested
                if (mismatch_count > 0 && check_transpose)
                {
                    std::cout << "\nDirect comparison failed. Checking transposed layout..." << std::endl;
                    // TODO: Implement transpose check if IBlockDecoder actually transposes
                    // This would require knowing the matrix dimensions
                }

                return mismatch_count == 0;
            }

            /**
             * @brief Load a specific layer's weight tensor from GGUF
             * @param weight_name Name of weight tensor (e.g., "token_embd.weight")
             * @return Shared pointer to loaded tensor
             */
            std::shared_ptr<TensorBase> loadWeight(const std::string &weight_name)
            {
                ModelLoader loader;
                if (!loader.loadModel(model_path_))
                {
                    std::cerr << "Failed to load model: " << model_path_ << std::endl;
                    return nullptr;
                }

                auto weight = loader.loadTensor(weight_name);
                if (!weight)
                {
                    std::cerr << "Failed to load tensor: " << weight_name << std::endl;
                }
                return weight;
            }

            std::string model_path_;
        };

        /**
         * @brief Test Q8_0 dequantization equivalency
         */
        TEST_F(DequantEquivalencyTest, Q8_0_Equivalency)
        {
            // Load Q8_0 weight tensor from model
            auto weight = loadWeight("token_embd.weight");
            ASSERT_NE(weight, nullptr) << "Failed to load embedding weight";

            // Cast to Q8_0Tensor
            auto q8_tensor = std::dynamic_pointer_cast<Q8_0Tensor>(weight);
            ASSERT_NE(q8_tensor, nullptr) << "Weight is not Q8_0 type";

            const auto &shape = q8_tensor->shape();
            ASSERT_EQ(shape.size(), 2) << "Expected 2D tensor";

            size_t rows = shape[0];
            size_t cols = shape[1];

            std::cout << "Testing Q8_0 tensor: [" << rows << " × " << cols << "]" << std::endl;

            // Allocate output buffers
            std::vector<float> llaminar_output(cols);
            std::vector<float> llamacpp_output(cols);

            // Test first row dequantization
            size_t row_idx = 0;

            // Dequantize with Llaminar IBlockDecoder
            size_t num_blocks = (cols + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE;
            for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
            {
                size_t offset = block_idx * Q8_0Block::BLOCK_SIZE;
                size_t elements = std::min(Q8_0Block::BLOCK_SIZE, cols - offset);
                q8_tensor->decode_block_at(row_idx, block_idx, llaminar_output.data() + offset);
            }

            // Dequantize with llama.cpp
            const void *raw_block = q8_tensor->get_raw_block_at(row_idx, 0);
            const block_q8_0 *ggml_blocks = static_cast<const block_q8_0 *>(raw_block);
            dequantize_row_q8_0(ggml_blocks, llamacpp_output.data(), cols);

            // Compare outputs
            EXPECT_TRUE(compareOutputs(llaminar_output.data(), llamacpp_output.data(), cols,
                                       1e-6f, true /* check transpose */))
                << "Q8_0 dequantization does not match llama.cpp";
        }

        /**
         * @brief Test IQ4_NL dequantization equivalency
         */
        TEST_F(DequantEquivalencyTest, IQ4_NL_Equivalency)
        {
            // Override model path for IQ4_NL model
            model_path_ = "/workspaces/llaminar/models/qwen2.5-0.5b-instruct-iq4_nl.gguf";

            // Load a layer weight which should be IQ4_NL (embeddings might be kept at higher precision)
            auto weight = loadWeight("blk.0.attn_q.weight");
            ASSERT_NE(weight, nullptr) << "Failed to load IQ4_NL Q projection weight";

            // Debug: Print actual tensor type
            std::cout << "Loaded tensor type: " << static_cast<int>(weight->native_type()) << std::endl;
            std::cout << "Expected IQ4_NL type: " << static_cast<int>(TensorType::IQ4_NL) << std::endl;

            auto iq4nl_tensor = std::dynamic_pointer_cast<IQ4_NLTensor>(weight);
            ASSERT_NE(iq4nl_tensor, nullptr) << "Weight is not IQ4_NL type (trying layer weight instead of embedding)";

            const auto &shape = iq4nl_tensor->shape();
            size_t rows = shape[0];
            size_t cols = shape[1];

            std::cout << "Testing IQ4_NL tensor: [" << rows << " × " << cols << "]" << std::endl;
            std::cout << "Block size: " << IQ4_NLBlock::BLOCK_SIZE << std::endl;

            std::vector<float> llaminar_output(cols);
            std::vector<float> llamacpp_output(cols);

            size_t row_idx = 0;

            // Dequantize with Llaminar
            size_t num_blocks = (cols + IQ4_NLBlock::BLOCK_SIZE - 1) / IQ4_NLBlock::BLOCK_SIZE;
            std::cout << "Number of blocks in row: " << num_blocks << std::endl;
            for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
            {
                size_t offset = block_idx * IQ4_NLBlock::BLOCK_SIZE;
                iq4nl_tensor->decode_block_at(row_idx, block_idx, llaminar_output.data() + offset);
            }

            // Dequantize with llama.cpp
            const void *raw_block = iq4nl_tensor->get_raw_block_at(row_idx, 0);
            const block_iq4_nl *ggml_blocks = static_cast<const block_iq4_nl *>(raw_block);
            dequantize_row_iq4_nl(ggml_blocks, llamacpp_output.data(), cols);

            // Print first few values from each
            std::cout << "\nFirst 10 Llaminar values: ";
            for (int i = 0; i < 10 && i < cols; ++i)
            {
                std::cout << llaminar_output[i] << " ";
            }
            std::cout << "\n\nFirst 10 llama.cpp values: ";
            for (int i = 0; i < 10 && i < cols; ++i)
            {
                std::cout << llamacpp_output[i] << " ";
            }
            std::cout << "\n"
                      << std::endl;

            EXPECT_TRUE(compareOutputs(llaminar_output.data(), llamacpp_output.data(), cols,
                                       1e-5f, true /* IQ4_NL may have slightly larger error */))
                << "IQ4_NL dequantization does not match llama.cpp";
        }

        /**
         * @brief Test Q4_0 dequantization equivalency
         */
        TEST_F(DequantEquivalencyTest, Q4_0_Equivalency)
        {
            model_path_ = "/workspaces/llaminar/models/qwen2.5-0.5b-instruct-q4_0.gguf";

            // Use layer weight (embedding often kept at higher precision)
            auto weight = loadWeight("blk.0.attn_q.weight");
            ASSERT_NE(weight, nullptr) << "Failed to load Q4_0 Q projection weight";

            auto q4_tensor = std::dynamic_pointer_cast<Q4_0Tensor>(weight);
            ASSERT_NE(q4_tensor, nullptr) << "Weight is not Q4_0 type";

            const auto &shape = q4_tensor->shape();
            size_t rows = shape[0];
            size_t cols = shape[1];

            std::cout << "Testing Q4_0 tensor: [" << rows << " × " << cols << "]" << std::endl;

            std::vector<float> llaminar_output(cols);
            std::vector<float> llamacpp_output(cols);

            size_t row_idx = 0;

            size_t num_blocks = (cols + Q4_0Block::BLOCK_SIZE - 1) / Q4_0Block::BLOCK_SIZE;
            for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
            {
                size_t offset = block_idx * Q4_0Block::BLOCK_SIZE;
                q4_tensor->decode_block_at(row_idx, block_idx, llaminar_output.data() + offset);
            }

            const void *raw_block = q4_tensor->get_raw_block_at(row_idx, 0);
            const block_q4_0 *ggml_blocks = static_cast<const block_q4_0 *>(raw_block);
            dequantize_row_q4_0(ggml_blocks, llamacpp_output.data(), cols);

            EXPECT_TRUE(compareOutputs(llaminar_output.data(), llamacpp_output.data(), cols))
                << "Q4_0 dequantization does not match llama.cpp";
        }

        /**
         * @brief Test Q4_1 dequantization equivalency
         */
        TEST_F(DequantEquivalencyTest, Q4_1_Equivalency)
        {
            model_path_ = "/workspaces/llaminar/models/Qwen2.5-7B-Instruct-Q4_0.gguf";

            // Load model to inspect tensor types
            ModelLoader loader;
            if (!loader.loadModel(model_path_))
            {
                GTEST_SKIP() << "Q4_1 model not available";
                return;
            }

            // This model uses mixed quantization - ffn_down.weight is Q4_1
            auto weight = loadWeight("blk.0.ffn_down.weight");
            if (!weight)
            {
                GTEST_SKIP() << "Q4_1 model or weight not available";
                return;
            }

            auto q41_tensor = std::dynamic_pointer_cast<Q4_1Tensor>(weight);
            if (!q41_tensor)
            {
                // List all blk.0 tensors to find a Q4_1 one
                std::cerr << "\nDEBUG: Listing all blk.0 tensors in Q4_0 model:\n";
                const auto &model = loader.getModel();
                for (const auto &tensor_info : model.tensors)
                {
                    if (tensor_info.name.find("blk.0.") != std::string::npos)
                    {
                        std::cerr << "  " << tensor_info.name << " (" << ggml_type_name(static_cast<ggml_type>(tensor_info.type)) << ")\n";
                    }
                }
                GTEST_SKIP() << "blk.0.attn_q.weight is not Q4_1 type in this model";
                return;
            }

            const auto &shape = q41_tensor->shape();
            size_t cols = shape[1];

            std::vector<float> llaminar_output(cols);
            std::vector<float> llamacpp_output(cols);

            size_t row_idx = 0;
            size_t num_blocks = (cols + Q4_1Block::BLOCK_SIZE - 1) / Q4_1Block::BLOCK_SIZE;

            for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
            {
                size_t offset = block_idx * Q4_1Block::BLOCK_SIZE;
                q41_tensor->decode_block_at(row_idx, block_idx, llaminar_output.data() + offset);
            }

            const void *raw_block = q41_tensor->get_raw_block_at(row_idx, 0);
            const block_q4_1 *ggml_blocks = static_cast<const block_q4_1 *>(raw_block);
            dequantize_row_q4_1(ggml_blocks, llamacpp_output.data(), cols);

            EXPECT_TRUE(compareOutputs(llaminar_output.data(), llamacpp_output.data(), cols))
                << "Q4_1 dequantization does not match llama.cpp";
        }

        /**
         * @brief Test Q5_0 dequantization equivalency
         */
        TEST_F(DequantEquivalencyTest, Q5_0_Equivalency)
        {
            model_path_ = "/workspaces/llaminar/models/qwen2.5-0.5b-instruct-q5_0.gguf";

            // Use layer weight (embedding often kept at higher precision)
            auto weight = loadWeight("blk.0.attn_q.weight");
            if (!weight)
            {
                GTEST_SKIP() << "Q5_0 model or weight not available";
                return;
            }

            auto q5_tensor = std::dynamic_pointer_cast<Q5_0Tensor>(weight);
            ASSERT_NE(q5_tensor, nullptr) << "Weight is not Q5_0 type";

            const auto &shape = q5_tensor->shape();
            size_t cols = shape[1];

            std::vector<float> llaminar_output(cols);
            std::vector<float> llamacpp_output(cols);

            size_t row_idx = 0;
            size_t num_blocks = (cols + Q5_0Block::BLOCK_SIZE - 1) / Q5_0Block::BLOCK_SIZE;

            for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
            {
                size_t offset = block_idx * Q5_0Block::BLOCK_SIZE;
                q5_tensor->decode_block_at(row_idx, block_idx, llaminar_output.data() + offset);
            }

            const void *raw_block = q5_tensor->get_raw_block_at(row_idx, 0);
            const block_q5_0 *ggml_blocks = static_cast<const block_q5_0 *>(raw_block);
            dequantize_row_q5_0(ggml_blocks, llamacpp_output.data(), cols);

            EXPECT_TRUE(compareOutputs(llaminar_output.data(), llamacpp_output.data(), cols))
                << "Q5_0 dequantization does not match llama.cpp";
        }

        /**
         * @brief Test Q5_1 dequantization equivalency
         */
        TEST_F(DequantEquivalencyTest, Q5_1_Equivalency)
        {
            model_path_ = "/workspaces/llaminar/models/qwen2.5-0.5b-instruct-q5_k_m.gguf";

            // Q5_K_M model uses Q5_1 for attention weights (133 Q5_1 tensors total)
            auto weight = loadWeight("blk.0.attn_q.weight");
            if (!weight)
            {
                GTEST_SKIP() << "Q5_1 model or weight not available";
                return;
            }

            auto q5_tensor = std::dynamic_pointer_cast<Q5_1Tensor>(weight);
            ASSERT_NE(q5_tensor, nullptr) << "Weight is not Q5_1 type";

            const auto &shape = q5_tensor->shape();
            size_t cols = shape[1];

            std::vector<float> llaminar_output(cols);
            std::vector<float> llamacpp_output(cols);

            size_t row_idx = 0;
            size_t num_blocks = (cols + Q5_1Block::BLOCK_SIZE - 1) / Q5_1Block::BLOCK_SIZE;

            for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
            {
                size_t offset = block_idx * Q5_1Block::BLOCK_SIZE;
                q5_tensor->decode_block_at(row_idx, block_idx, llaminar_output.data() + offset);
            }

            const void *raw_block = q5_tensor->get_raw_block_at(row_idx, 0);
            const block_q5_1 *ggml_blocks = static_cast<const block_q5_1 *>(raw_block);
            dequantize_row_q5_1(ggml_blocks, llamacpp_output.data(), cols);

            EXPECT_TRUE(compareOutputs(llaminar_output.data(), llamacpp_output.data(), cols))
                << "Q5_1 dequantization does not match llama.cpp";
        }

        /**
         * @brief Test Q6_K dequantization equivalency
         */
        TEST_F(DequantEquivalencyTest, Q6_K_Equivalency)
        {
            model_path_ = "/workspaces/llaminar/models/qwen2.5-0.5b-instruct-q6_k.gguf";

            // Q6_K model stores FFN weights in Q6_K format (attention uses Q8_0)
            auto weight = loadWeight("blk.0.ffn_down.weight");
            if (!weight)
            {
                GTEST_SKIP() << "Q6_K model not available";
                return;
            }

            auto q6k_tensor = std::dynamic_pointer_cast<Q6_KTensor>(weight);
            ASSERT_NE(q6k_tensor, nullptr) << "Weight is not Q6_K type";

            const auto &shape = q6k_tensor->shape();
            size_t cols = shape[1];

            std::vector<float> llaminar_output(cols);
            std::vector<float> llamacpp_output(cols);

            size_t row_idx = 0;
            size_t num_blocks = (cols + Q6_KBlock::BLOCK_SIZE - 1) / Q6_KBlock::BLOCK_SIZE;

            for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
            {
                size_t offset = block_idx * Q6_KBlock::BLOCK_SIZE;
                q6k_tensor->decode_block_at(row_idx, block_idx, llaminar_output.data() + offset);
            }

            const void *raw_block = q6k_tensor->get_raw_block_at(row_idx, 0);
            const block_q6_K *ggml_blocks = static_cast<const block_q6_K *>(raw_block);
            dequantize_row_q6_K(ggml_blocks, llamacpp_output.data(), cols);

            EXPECT_TRUE(compareOutputs(llaminar_output.data(), llamacpp_output.data(), cols, 1e-5f))
                << "Q6_K dequantization does not match llama.cpp";
        }

        /**
         * @brief Test Q2_K dequantization equivalency
         */
        TEST_F(DequantEquivalencyTest, Q2_K_Equivalency)
        {
            model_path_ = "/workspaces/llaminar/models/Qwen2.5-7B-Instruct-Q2_K.gguf";

            // Load model to inspect tensor types
            ModelLoader loader;
            if (!loader.loadModel(model_path_))
            {
                GTEST_SKIP() << "Q2_K model not available";
                return;
            }

            // 7B Q2_K model uses mixed quantization - use gate/up or Q weights which are Q2_K
            auto weight = loadWeight("blk.0.ffn_gate.weight");
            if (!weight)
            {
                GTEST_SKIP() << "Q2_K model or weight not available";
                return;
            }

            auto q2k_tensor = std::dynamic_pointer_cast<Q2_KTensor>(weight);
            if (!q2k_tensor)
            {
                // List all blk.0 tensors to find a Q2_K one
                std::cerr << "\nDEBUG: Listing all blk.0 tensors in Q2_K model:\n";
                const auto &model = loader.getModel();
                for (const auto &tensor_info : model.tensors)
                {
                    if (tensor_info.name.find("blk.0.") != std::string::npos)
                    {
                        std::cerr << "  " << tensor_info.name << " (" << ggml_type_name(static_cast<ggml_type>(tensor_info.type)) << ")\n";
                    }
                }
                GTEST_SKIP() << "blk.0.ffn_down.weight is not Q2_K type in this model";
                return;
            }

            const auto &shape = q2k_tensor->shape();
            size_t cols = shape[1];

            std::vector<float> llaminar_output(cols);
            std::vector<float> llamacpp_output(cols);

            size_t row_idx = 0;
            size_t num_blocks = (cols + Q2_KBlock::BLOCK_SIZE - 1) / Q2_KBlock::BLOCK_SIZE;

            for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
            {
                size_t offset = block_idx * Q2_KBlock::BLOCK_SIZE;
                q2k_tensor->decode_block_at(row_idx, block_idx, llaminar_output.data() + offset);
            }

            const void *raw_block = q2k_tensor->get_raw_block_at(row_idx, 0);
            const block_q2_K *ggml_blocks = static_cast<const block_q2_K *>(raw_block);
            dequantize_row_q2_K(ggml_blocks, llamacpp_output.data(), cols);

            EXPECT_TRUE(compareOutputs(llaminar_output.data(), llamacpp_output.data(), cols, 1e-5f))
                << "Q2_K dequantization does not match llama.cpp";
        }

        /**
         * @brief Test Q3_K dequantization equivalency
         */
        TEST_F(DequantEquivalencyTest, Q3_K_Equivalency)
        {
            model_path_ = "/workspaces/llaminar/models/Qwen2.5-7B-Instruct-Q3_K_M.gguf";

            // Load model to inspect tensor types
            ModelLoader loader;
            if (!loader.loadModel(model_path_))
            {
                GTEST_SKIP() << "Q3_K model not available";
                return;
            }

            // 7B Q3_K_M model uses mixed quantization - use gate/up weights which are Q3_K
            auto weight = loadWeight("blk.0.ffn_gate.weight");
            if (!weight)
            {
                GTEST_SKIP() << "Q3_K model or weight not available";
                return;
            }

            auto q3k_tensor = std::dynamic_pointer_cast<Q3_KTensor>(weight);
            if (!q3k_tensor)
            {
                // List all blk.0 tensors to find a Q3_K one
                std::cerr << "\nDEBUG: Listing all blk.0 tensors in Q3_K_M model:\n";
                const auto &model = loader.getModel();
                for (const auto &tensor_info : model.tensors)
                {
                    if (tensor_info.name.find("blk.0.") != std::string::npos)
                    {
                        std::cerr << "  " << tensor_info.name << " (" << ggml_type_name(static_cast<ggml_type>(tensor_info.type)) << ")\n";
                    }
                }
                GTEST_SKIP() << "blk.0.ffn_down.weight is not Q3_K type in this model";
                return;
            }

            const auto &shape = q3k_tensor->shape();
            size_t cols = shape[1];

            std::vector<float> llaminar_output(cols);
            std::vector<float> llamacpp_output(cols);

            size_t row_idx = 0;
            size_t num_blocks = (cols + Q3_KBlock::BLOCK_SIZE - 1) / Q3_KBlock::BLOCK_SIZE;

            for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
            {
                size_t offset = block_idx * Q3_KBlock::BLOCK_SIZE;
                q3k_tensor->decode_block_at(row_idx, block_idx, llaminar_output.data() + offset);
            }

            const void *raw_block = q3k_tensor->get_raw_block_at(row_idx, 0);
            const block_q3_K *ggml_blocks = static_cast<const block_q3_K *>(raw_block);
            dequantize_row_q3_K(ggml_blocks, llamacpp_output.data(), cols);

            EXPECT_TRUE(compareOutputs(llaminar_output.data(), llamacpp_output.data(), cols, 1e-5f))
                << "Q3_K dequantization does not match llama.cpp";
        }

        /**
         * @brief Test Q4_K dequantization equivalency
         */
        TEST_F(DequantEquivalencyTest, Q4_K_Equivalency)
        {
            model_path_ = "/workspaces/llaminar/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

            // Q4_K_M model stores some FFN weights in Q4_K format (attention uses Q5_0)
            auto weight = loadWeight("blk.11.ffn_down.weight");
            if (!weight)
            {
                GTEST_SKIP() << "Q4_K model or tensor not available";
                return;
            }

            auto q4k_tensor = std::dynamic_pointer_cast<Q4_KTensor>(weight);
            ASSERT_NE(q4k_tensor, nullptr) << "Weight is not Q4_K type";

            const auto &shape = q4k_tensor->shape();
            size_t cols = shape[1];

            std::vector<float> llaminar_output(cols);
            std::vector<float> llamacpp_output(cols);

            size_t row_idx = 0;
            size_t num_blocks = (cols + Q4_KBlock::BLOCK_SIZE - 1) / Q4_KBlock::BLOCK_SIZE;

            for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
            {
                size_t offset = block_idx * Q4_KBlock::BLOCK_SIZE;
                q4k_tensor->decode_block_at(row_idx, block_idx, llaminar_output.data() + offset);
            }

            const void *raw_block = q4k_tensor->get_raw_block_at(row_idx, 0);
            const block_q4_K *ggml_blocks = static_cast<const block_q4_K *>(raw_block);
            dequantize_row_q4_K(ggml_blocks, llamacpp_output.data(), cols);

            EXPECT_TRUE(compareOutputs(llaminar_output.data(), llamacpp_output.data(), cols, 1e-5f))
                << "Q4_K dequantization does not match llama.cpp";
        }

        /**
         * @brief Test Q5_K dequantization equivalency
         */
        TEST_F(DequantEquivalencyTest, Q5_K_Equivalency)
        {
            model_path_ = "/workspaces/llaminar/models/qwen2.5-0.5b-instruct-q5_k_m.gguf";

            // Q5_K_M model stores some FFN weights in Q5_K format (attention uses Q5_1)
            auto weight = loadWeight("blk.11.ffn_down.weight");
            if (!weight)
            {
                GTEST_SKIP() << "Q5_K model or tensor not available";
                return;
            }

            auto q5k_tensor = std::dynamic_pointer_cast<Q5_KTensor>(weight);
            ASSERT_NE(q5k_tensor, nullptr) << "Weight is not Q5_K type";

            const auto &shape = q5k_tensor->shape();
            size_t cols = shape[1];

            std::vector<float> llaminar_output(cols);
            std::vector<float> llamacpp_output(cols);

            size_t row_idx = 0;
            size_t num_blocks = (cols + Q5_KBlock::BLOCK_SIZE - 1) / Q5_KBlock::BLOCK_SIZE;

            for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
            {
                size_t offset = block_idx * Q5_KBlock::BLOCK_SIZE;
                q5k_tensor->decode_block_at(row_idx, block_idx, llaminar_output.data() + offset);
            }

            const void *raw_block = q5k_tensor->get_raw_block_at(row_idx, 0);
            const block_q5_K *ggml_blocks = static_cast<const block_q5_K *>(raw_block);
            dequantize_row_q5_K(ggml_blocks, llamacpp_output.data(), cols);

            EXPECT_TRUE(compareOutputs(llaminar_output.data(), llamacpp_output.data(), cols, 1e-5f))
                << "Q5_K dequantization does not match llama.cpp";
        }

        /**
         * @brief Test Q8_K dequantization equivalency
         */
        TEST_F(DequantEquivalencyTest, Q8_K_Equivalency)
        {
            // Q8_K model not commonly available - skip if missing
            model_path_ = "/workspaces/llaminar/models/qwen2.5-0.5b-instruct-q8_k.gguf";

            auto weight = loadWeight("blk.0.attn_q.weight");
            if (!weight)
            {
                GTEST_SKIP() << "Q8_K model not available";
                return;
            }

            auto q8k_tensor = std::dynamic_pointer_cast<Q8_KTensor>(weight);
            ASSERT_NE(q8k_tensor, nullptr) << "Weight is not Q8_K type";

            const auto &shape = q8k_tensor->shape();
            size_t cols = shape[1];

            std::vector<float> llaminar_output(cols);
            std::vector<float> llamacpp_output(cols);

            size_t row_idx = 0;
            size_t num_blocks = (cols + Q8_KBlock::BLOCK_SIZE - 1) / Q8_KBlock::BLOCK_SIZE;

            for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
            {
                size_t offset = block_idx * Q8_KBlock::BLOCK_SIZE;
                q8k_tensor->decode_block_at(row_idx, block_idx, llaminar_output.data() + offset);
            }

            const void *raw_block = q8k_tensor->get_raw_block_at(row_idx, 0);
            const block_q8_K *ggml_blocks = static_cast<const block_q8_K *>(raw_block);
            dequantize_row_q8_K(ggml_blocks, llamacpp_output.data(), cols);

            EXPECT_TRUE(compareOutputs(llaminar_output.data(), llamacpp_output.data(), cols, 1e-5f))
                << "Q8_K dequantization does not match llama.cpp";
        }

        /**
         * @brief Test IQ4_XS dequantization equivalency
         */
        TEST_F(DequantEquivalencyTest, IQ4_XS_Equivalency)
        {
            model_path_ = "/workspaces/llaminar/models/ReWiz-Qwen-2.5-14B-IQ4_XS.gguf";

            ModelLoader loader;
            if (!loader.loadModel(model_path_))
            {
                GTEST_SKIP() << "IQ4_XS model not available";
                return;
            }

            // Use layer weight
            auto weight = loadWeight("blk.0.attn_q.weight");
            if (!weight)
            {
                GTEST_SKIP() << "IQ4_XS model or weight not available";
                return;
            }

            auto iq4xs_tensor = std::dynamic_pointer_cast<IQ4_XSTensor>(weight);
            if (!iq4xs_tensor)
            {
                // List all blk.0 tensors to find an IQ4_XS one
                std::cerr << "\nDEBUG: Listing all blk.0 tensors in IQ4_XS model:\n";
                const auto &model = loader.getModel();
                for (const auto &tensor_info : model.tensors)
                {
                    if (tensor_info.name.find("blk.0.") != std::string::npos)
                    {
                        std::cerr << "  " << tensor_info.name << " (" << ggml_type_name(static_cast<ggml_type>(tensor_info.type)) << ")\n";
                    }
                }
                GTEST_SKIP() << "blk.0.attn_q.weight is not IQ4_XS type in this model";
                return;
            }

            const auto &shape = iq4xs_tensor->shape();
            size_t cols = shape[1];

            std::vector<float> llaminar_output(cols);
            std::vector<float> llamacpp_output(cols);

            size_t row_idx = 0;
            size_t num_blocks = (cols + IQ4_XSBlock::BLOCK_SIZE - 1) / IQ4_XSBlock::BLOCK_SIZE;

            for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
            {
                size_t offset = block_idx * IQ4_XSBlock::BLOCK_SIZE;
                iq4xs_tensor->decode_block_at(row_idx, block_idx, llaminar_output.data() + offset);
            }

            const void *raw_block = iq4xs_tensor->get_raw_block_at(row_idx, 0);
            const block_iq4_xs *ggml_blocks = static_cast<const block_iq4_xs *>(raw_block);
            dequantize_row_iq4_xs(ggml_blocks, llamacpp_output.data(), cols);

            EXPECT_TRUE(compareOutputs(llaminar_output.data(), llamacpp_output.data(), cols, 1e-5f))
                << "IQ4_XS dequantization does not match llama.cpp";
        }

        /**
         * @brief Test IQ3_XXS dequantization equivalency
         */
        TEST_F(DequantEquivalencyTest, IQ3_XXS_Equivalency)
        {
            model_path_ = "/workspaces/llaminar/models/Qwen2.5-VL-7B-Instruct-UD-IQ3_XXS.gguf";

            ModelLoader loader;
            if (!loader.loadModel(model_path_))
            {
                GTEST_SKIP() << "IQ3_XXS model not available";
                return;
            }

            // VL model has IQ3_XXS tensors
            auto weight = loadWeight("blk.0.attn_q.weight");
            if (!weight)
            {
                GTEST_SKIP() << "IQ3_XXS model or weight not available";
                return;
            }

            auto iq3xxs_tensor = std::dynamic_pointer_cast<IQ3_XXSTensor>(weight);
            if (!iq3xxs_tensor)
            {
                // List all blk.0 tensors to find an IQ3_XXS one
                std::cerr << "\nDEBUG: Listing all blk.0 tensors in IQ3_XXS model:\n";
                const auto &model = loader.getModel();
                for (const auto &tensor_info : model.tensors)
                {
                    if (tensor_info.name.find("blk.0.") != std::string::npos)
                    {
                        std::cerr << "  " << tensor_info.name << " (" << ggml_type_name(static_cast<ggml_type>(tensor_info.type)) << ")\n";
                    }
                }
                GTEST_SKIP() << "blk.0.attn_q.weight is not IQ3_XXS type in this model";
                return;
            }

            const auto &shape = iq3xxs_tensor->shape();
            size_t cols = shape[1];

            std::vector<float> llaminar_output(cols);
            std::vector<float> llamacpp_output(cols);

            size_t row_idx = 0;
            size_t num_blocks = (cols + IQ3_XXSBlock::BLOCK_SIZE - 1) / IQ3_XXSBlock::BLOCK_SIZE;

            for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
            {
                size_t offset = block_idx * IQ3_XXSBlock::BLOCK_SIZE;
                iq3xxs_tensor->decode_block_at(row_idx, block_idx, llaminar_output.data() + offset);
            }

            const void *raw_block = iq3xxs_tensor->get_raw_block_at(row_idx, 0);
            const block_iq3_xxs *ggml_blocks = static_cast<const block_iq3_xxs *>(raw_block);
            dequantize_row_iq3_xxs(ggml_blocks, llamacpp_output.data(), cols);

            EXPECT_TRUE(compareOutputs(llaminar_output.data(), llamacpp_output.data(), cols, 1e-5f))
                << "IQ3_XXS dequantization does not match llama.cpp";
        }

        /**
         * @brief Test IQ3_S dequantization equivalency
         */
        TEST_F(DequantEquivalencyTest, IQ3_S_Equivalency)
        {
            model_path_ = "/workspaces/llaminar/models/Qwen2-0.5B.IQ3_S.gguf";

            ModelLoader loader;
            if (!loader.loadModel(model_path_))
            {
                GTEST_SKIP() << "IQ3_S model not available";
                return;
            }

            // Use ffn_down which is IQ3_S in this model
            auto weight = loadWeight("blk.0.ffn_down.weight");
            if (!weight)
            {
                GTEST_SKIP() << "IQ3_S model or weight not available";
                return;
            }

            auto iq3s_tensor = std::dynamic_pointer_cast<IQ3_STensor>(weight);
            if (!iq3s_tensor)
            {
                // List all blk.0 tensors to find an IQ3_S one
                std::cerr << "\nDEBUG: Listing all blk.0 tensors in IQ3_S model:\n";
                const auto &model = loader.getModel();
                for (const auto &tensor_info : model.tensors)
                {
                    if (tensor_info.name.find("blk.0.") != std::string::npos)
                    {
                        std::cerr << "  " << tensor_info.name << " (" << ggml_type_name(static_cast<ggml_type>(tensor_info.type)) << ")\n";
                    }
                }
                GTEST_SKIP() << "blk.0.attn_q.weight is not IQ3_S type in this model";
                return;
            }

            const auto &shape = iq3s_tensor->shape();
            size_t cols = shape[1];

            std::vector<float> llaminar_output(cols);
            std::vector<float> llamacpp_output(cols);

            size_t row_idx = 0;
            size_t num_blocks = (cols + IQ3_SBlock::BLOCK_SIZE - 1) / IQ3_SBlock::BLOCK_SIZE;

            for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
            {
                size_t offset = block_idx * IQ3_SBlock::BLOCK_SIZE;
                iq3s_tensor->decode_block_at(row_idx, block_idx, llaminar_output.data() + offset);
            }

            const void *raw_block = iq3s_tensor->get_raw_block_at(row_idx, 0);
            const block_iq3_s *ggml_blocks = static_cast<const block_iq3_s *>(raw_block);
            dequantize_row_iq3_s(ggml_blocks, llamacpp_output.data(), cols);

            EXPECT_TRUE(compareOutputs(llaminar_output.data(), llamacpp_output.data(), cols, 1e-5f))
                << "IQ3_S dequantization does not match llama.cpp";
        }

        /**
         * @brief Test IQ2_XXS dequantization equivalency
         */
        TEST_F(DequantEquivalencyTest, IQ2_XXS_Equivalency)
        {
            // VL model has IQ2_XXS tensors
            model_path_ = "/workspaces/llaminar/models/Qwen2.5-VL-7B-Instruct-UD-IQ2_XXS.gguf";

            ModelLoader loader;
            if (!loader.loadModel(model_path_))
            {
                GTEST_SKIP() << "IQ2_XXS model not available";
                return;
            }

            // Try various weights
            auto weight = loadWeight("blk.0.attn_q.weight");
            if (!weight)
            {
                weight = loadWeight("blk.0.ffn_gate.weight");
            }
            if (!weight)
            {
                GTEST_SKIP() << "IQ2_XXS model or weight not available";
                return;
            }

            auto iq2xxs_tensor = std::dynamic_pointer_cast<IQ2_XXSTensor>(weight);
            if (!iq2xxs_tensor)
            {
                // List all blk.0 tensors to find an IQ2_XXS one
                std::cerr << "\nDEBUG: Listing all blk.0 tensors in IQ2_XXS model:\n";
                const auto &model = loader.getModel();
                for (const auto &tensor_info : model.tensors)
                {
                    if (tensor_info.name.find("blk.0.") != std::string::npos)
                    {
                        std::cerr << "  " << tensor_info.name << " (" << ggml_type_name(static_cast<ggml_type>(tensor_info.type)) << ")\n";
                    }
                }
                GTEST_SKIP() << "No IQ2_XXS tensors found in this model";
                return;
            }

            const auto &shape = iq2xxs_tensor->shape();
            size_t cols = shape[1];

            std::vector<float> llaminar_output(cols);
            std::vector<float> llamacpp_output(cols);

            size_t row_idx = 0;
            size_t num_blocks = (cols + IQ2_XXSBlock::BLOCK_SIZE - 1) / IQ2_XXSBlock::BLOCK_SIZE;

            for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
            {
                size_t offset = block_idx * IQ2_XXSBlock::BLOCK_SIZE;
                iq2xxs_tensor->decode_block_at(row_idx, block_idx, llaminar_output.data() + offset);
            }

            const void *raw_block = iq2xxs_tensor->get_raw_block_at(row_idx, 0);
            const block_iq2_xxs *ggml_blocks = static_cast<const block_iq2_xxs *>(raw_block);
            dequantize_row_iq2_xxs(ggml_blocks, llamacpp_output.data(), cols);

            EXPECT_TRUE(compareOutputs(llaminar_output.data(), llamacpp_output.data(), cols, 1e-5f))
                << "IQ2_XXS dequantization does not match llama.cpp";
        }

        /**
         * @brief Test IQ2_XS dequantization equivalency
         */
        TEST_F(DequantEquivalencyTest, IQ2_XS_Equivalency)
        {
            // ReWiz 14B IQ2_XS model
            model_path_ = "/workspaces/llaminar/models/ReWiz-Qwen-2.5-14B-IQ2_XS.gguf";

            ModelLoader loader;
            if (!loader.loadModel(model_path_))
            {
                GTEST_SKIP() << "IQ2_XS model not available";
                return;
            }

            // Try various weights to find IQ2_XS
            auto weight = loadWeight("blk.0.attn_q.weight");
            if (!weight)
            {
                weight = loadWeight("blk.0.ffn_up.weight");
            }
            if (!weight)
            {
                weight = loadWeight("blk.0.ffn_down.weight");
            }
            if (!weight)
            {
                GTEST_SKIP() << "IQ2_XS model or weight not available";
                return;
            }

            auto iq2xs_tensor = std::dynamic_pointer_cast<IQ2_XSTensor>(weight);
            if (!iq2xs_tensor)
            {
                // List all blk.0 tensors to find an IQ2_XS one
                std::cerr << "\nDEBUG: Listing all blk.0 tensors in IQ2_XS model:\n";
                const auto &model = loader.getModel();
                for (const auto &tensor_info : model.tensors)
                {
                    if (tensor_info.name.find("blk.0.") != std::string::npos)
                    {
                        std::cerr << "  " << tensor_info.name << " (" << ggml_type_name(static_cast<ggml_type>(tensor_info.type)) << ")\n";
                    }
                }
                GTEST_SKIP() << "No IQ2_XS tensors found in this model";
                return;
            }

            const auto &shape = iq2xs_tensor->shape();
            size_t cols = shape[1];

            std::vector<float> llaminar_output(cols);
            std::vector<float> llamacpp_output(cols);

            size_t row_idx = 0;
            size_t num_blocks = (cols + IQ2_XSBlock::BLOCK_SIZE - 1) / IQ2_XSBlock::BLOCK_SIZE;

            for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
            {
                size_t offset = block_idx * IQ2_XSBlock::BLOCK_SIZE;
                iq2xs_tensor->decode_block_at(row_idx, block_idx, llaminar_output.data() + offset);
            }

            const void *raw_block = iq2xs_tensor->get_raw_block_at(row_idx, 0);
            const block_iq2_xs *ggml_blocks = static_cast<const block_iq2_xs *>(raw_block);
            dequantize_row_iq2_xs(ggml_blocks, llamacpp_output.data(), cols);

            EXPECT_TRUE(compareOutputs(llaminar_output.data(), llamacpp_output.data(), cols, 1e-5f))
                << "IQ2_XS dequantization does not match llama.cpp";
        }

        /**
         * @brief Test IQ2_S dequantization equivalency
         */
        TEST_F(DequantEquivalencyTest, IQ2_S_Equivalency)
        {
            // 7B IQ2_M model may contain IQ2_S tensors
            model_path_ = "/workspaces/llaminar/models/Qwen2.5-7B-Instruct-IQ2_M.gguf";

            ModelLoader loader;
            if (!loader.loadModel(model_path_))
            {
                GTEST_SKIP() << "IQ2_S model not available";
                return;
            }

            // Try various weights to find IQ2_S
            auto weight = loadWeight("blk.0.attn_k.weight");
            if (!weight)
            {
                weight = loadWeight("blk.0.attn_v.weight");
            }
            if (!weight)
            {
                GTEST_SKIP() << "IQ2_S model or weight not available";
                return;
            }

            auto iq2s_tensor = std::dynamic_pointer_cast<IQ2_STensor>(weight);
            if (!iq2s_tensor)
            {
                // List all blk.0 tensors to find an IQ2_S one
                std::cerr << "\nDEBUG: Listing all blk.0 tensors in IQ2_M model:\n";
                const auto &model = loader.getModel();
                for (const auto &tensor_info : model.tensors)
                {
                    if (tensor_info.name.find("blk.0.") != std::string::npos)
                    {
                        std::cerr << "  " << tensor_info.name << " (" << ggml_type_name(static_cast<ggml_type>(tensor_info.type)) << ")\n";
                    }
                }
                GTEST_SKIP() << "No IQ2_S tensors found in this model";
                return;
            }

            const auto &shape = iq2s_tensor->shape();
            size_t cols = shape[1];

            std::vector<float> llaminar_output(cols);
            std::vector<float> llamacpp_output(cols);

            size_t row_idx = 0;
            size_t num_blocks = (cols + IQ2_SBlock::BLOCK_SIZE - 1) / IQ2_SBlock::BLOCK_SIZE;

            for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
            {
                size_t offset = block_idx * IQ2_SBlock::BLOCK_SIZE;
                iq2s_tensor->decode_block_at(row_idx, block_idx, llaminar_output.data() + offset);
            }

            const void *raw_block = iq2s_tensor->get_raw_block_at(row_idx, 0);
            const block_iq2_s *ggml_blocks = static_cast<const block_iq2_s *>(raw_block);
            dequantize_row_iq2_s(ggml_blocks, llamacpp_output.data(), cols);

            EXPECT_TRUE(compareOutputs(llaminar_output.data(), llamacpp_output.data(), cols, 1e-5f))
                << "IQ2_S dequantization does not match llama.cpp";
        }

        /**
         * @brief Test IQ1_S (1.5625 bpw) dequantization equivalency
         * IQ1_S: Extremely aggressive 1-bit quantization with grid lookup
         * Block: d(fp16) + qs[32] + qh[8] = 42 bytes for 256 elements
         * Model: Qwen3-Coder-30B-A3B-Instruct-UD-IQ1_S.gguf
         */
        TEST_F(DequantEquivalencyTest, IQ1_S_Equivalency)
        {
            model_path_ = "/workspaces/llaminar/models/Qwen3-Coder-30B-A3B-Instruct-UD-IQ1_S.gguf";

            ModelLoader loader;
            if (!loader.loadModel(model_path_))
            {
                GTEST_SKIP() << "IQ1_S model not available";
                return;
            }

            // Try to find an IQ1_S tensor (qwen3moe uses different naming: ffn_gate_exps, ffn_up_exps)
            auto weight = loadWeight("blk.0.ffn_gate_exps.weight");
            if (!weight)
            {
                weight = loadWeight("blk.0.ffn_up_exps.weight");
            }
            if (!weight)
            {
                weight = loadWeight("blk.0.attn_k.weight");
            }
            if (!weight)
            {
                GTEST_SKIP() << "IQ1_S weight not available";
                return;
            }

            auto iq1s_tensor = std::dynamic_pointer_cast<IQ1_STensor>(weight);
            if (!iq1s_tensor)
            {
                // List all blk.0 tensors to find IQ1_S ones
                std::cerr << "\nDEBUG: Listing all blk.0 tensors in IQ1_S model:\n";
                const auto &model = loader.getModel();
                for (const auto &tensor_info : model.tensors)
                {
                    if (tensor_info.name.find("blk.0.") != std::string::npos)
                    {
                        std::cerr << "  " << tensor_info.name << " (" << ggml_type_name(static_cast<ggml_type>(tensor_info.type)) << ")\n";
                    }
                }
                GTEST_SKIP() << "No IQ1_S tensors found in this model";
                return;
            }

            const auto &shape = iq1s_tensor->shape();
            size_t cols = shape[1];

            std::vector<float> llaminar_output(cols);
            std::vector<float> llamacpp_output(cols);

            size_t row_idx = 0;
            size_t num_blocks = (cols + IQ1_SBlock::BLOCK_SIZE - 1) / IQ1_SBlock::BLOCK_SIZE;

            for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
            {
                size_t offset = block_idx * IQ1_SBlock::BLOCK_SIZE;
                iq1s_tensor->decode_block_at(row_idx, block_idx, llaminar_output.data() + offset);
            }

            const void *raw_block = iq1s_tensor->get_raw_block_at(row_idx, 0);
            const block_iq1_s *ggml_blocks = static_cast<const block_iq1_s *>(raw_block);
            dequantize_row_iq1_s(ggml_blocks, llamacpp_output.data(), cols);

            EXPECT_TRUE(compareOutputs(llaminar_output.data(), llamacpp_output.data(), cols, 1e-5f))
                << "IQ1_S dequantization does not match llama.cpp";
        }

        /**
         * @brief Test IQ1_M (1.75 bpw) dequantization equivalency
         * IQ1_M: Extremely aggressive 1-bit quantization with more precision than IQ1_S
         * Block: qs[32] + qh[16] + scales[8] = 56 bytes for 256 elements
         * Model: Qwen3-Coder-30B-A3B-Instruct-UD-IQ1_M.gguf
         */
        TEST_F(DequantEquivalencyTest, IQ1_M_Equivalency)
        {
            model_path_ = "/workspaces/llaminar/models/Qwen3-Coder-30B-A3B-Instruct-UD-IQ1_M.gguf";

            ModelLoader loader;
            if (!loader.loadModel(model_path_))
            {
                GTEST_SKIP() << "IQ1_M model not available";
                return;
            }

            // Try to find an IQ1_M tensor (qwen3moe uses different naming: ffn_gate_exps, ffn_up_exps)
            auto weight = loadWeight("blk.0.ffn_gate_exps.weight");
            if (!weight)
            {
                weight = loadWeight("blk.0.ffn_up_exps.weight");
            }
            if (!weight)
            {
                weight = loadWeight("blk.0.attn_k.weight");
            }
            if (!weight)
            {
                GTEST_SKIP() << "IQ1_M weight not available";
                return;
            }

            auto iq1m_tensor = std::dynamic_pointer_cast<IQ1_MTensor>(weight);
            if (!iq1m_tensor)
            {
                // List all blk.0 tensors to find IQ1_M ones
                std::cerr << "\nDEBUG: Listing all blk.0 tensors in IQ1_M model:\n";
                const auto &model = loader.getModel();
                for (const auto &tensor_info : model.tensors)
                {
                    if (tensor_info.name.find("blk.0.") != std::string::npos)
                    {
                        std::cerr << "  " << tensor_info.name << " (" << ggml_type_name(static_cast<ggml_type>(tensor_info.type)) << ")\n";
                    }
                }
                GTEST_SKIP() << "No IQ1_M tensors found in this model";
                return;
            }

            const auto &shape = iq1m_tensor->shape();
            size_t cols = shape[1];

            std::vector<float> llaminar_output(cols);
            std::vector<float> llamacpp_output(cols);

            size_t row_idx = 0;
            size_t num_blocks = (cols + IQ1_MBlock::BLOCK_SIZE - 1) / IQ1_MBlock::BLOCK_SIZE;

            for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
            {
                size_t offset = block_idx * IQ1_MBlock::BLOCK_SIZE;
                iq1m_tensor->decode_block_at(row_idx, block_idx, llaminar_output.data() + offset);
            }

            const void *raw_block = iq1m_tensor->get_raw_block_at(row_idx, 0);
            const block_iq1_m *ggml_blocks = static_cast<const block_iq1_m *>(raw_block);
            dequantize_row_iq1_m(ggml_blocks, llamacpp_output.data(), cols);

            EXPECT_TRUE(compareOutputs(llaminar_output.data(), llamacpp_output.data(), cols, 1e-5f))
                << "IQ1_M dequantization does not match llama.cpp";
        }

        /**
         * @brief Test BF16 dequantization equivalency
         *
         * BF16 is not a quantization format - it's a reduced precision float format.
         * This test validates that our BF16 to FP32 conversion matches llama.cpp's.
         */
        TEST_F(DequantEquivalencyTest, BF16_Equivalency)
        {
            model_path_ = "/workspaces/llaminar/models/Qwen2.5-1.5B-Instruct-bf16.gguf";

            ModelLoader loader;
            if (!loader.loadModel(model_path_))
            {
                GTEST_SKIP() << "BF16 model not available";
                return;
            }

            // Try to find a BF16 tensor (usually model weights are in BF16)
            auto weight = loadWeight("blk.0.attn_q.weight");
            if (!weight)
            {
                weight = loadWeight("token_embd.weight");
            }
            if (!weight)
            {
                GTEST_SKIP() << "BF16 weight not available";
                return;
            }

            auto bf16_tensor = std::dynamic_pointer_cast<BF16Tensor>(weight);
            if (!bf16_tensor)
            {
                GTEST_SKIP() << "No BF16 tensors found in this model";
                return;
            }

            const auto &shape = bf16_tensor->shape();
            size_t rows = shape[0];
            size_t cols = shape[1];

            std::cout << "Testing BF16 tensor with shape: " << rows << " × " << cols << std::endl;

            // Test a single row
            size_t row_idx = 0;
            std::vector<float> llaminar_output(cols);
            std::vector<float> llamacpp_output(cols);

            // Dequantize with Llaminar using to_fp32 method
            // Note: to_fp32 needs to convert the entire tensor, then we extract the row we want
            const uint16_t *bf16_data = bf16_tensor->bf16_data();
            size_t total_elements = rows * cols;
            std::vector<float> full_llaminar_output(total_elements);
            bf16_tensor->to_fp32(full_llaminar_output.data(), total_elements);

            // Extract the row we want to test
            std::copy(full_llaminar_output.begin() + row_idx * cols,
                      full_llaminar_output.begin() + (row_idx + 1) * cols,
                      llaminar_output.begin());

            // Dequantize with llama.cpp
            // BF16 is just uint16_t values that need to be reinterpreted as the upper 16 bits of FP32
            for (size_t i = 0; i < cols; ++i)
            {
                // llama.cpp's BF16 to FP32: shift BF16 bits to upper 16 bits of FP32
                uint32_t fp32_bits = static_cast<uint32_t>(bf16_data[row_idx * cols + i]) << 16;
                float fp32_value;
                std::memcpy(&fp32_value, &fp32_bits, sizeof(float));
                llamacpp_output[i] = fp32_value;
            }

            // Print first few values from each
            std::cout << "\nFirst 10 Llaminar values: ";
            for (int i = 0; i < 10 && i < cols; ++i)
            {
                std::cout << llaminar_output[i] << " ";
            }
            std::cout << "\n\nFirst 10 llama.cpp values: ";
            for (int i = 0; i < 10 && i < cols; ++i)
            {
                std::cout << llamacpp_output[i] << " ";
            }
            std::cout << "\n"
                      << std::endl;

            // BF16 should have exact conversion (no quantization error, just precision reduction)
            EXPECT_TRUE(compareOutputs(llaminar_output.data(), llamacpp_output.data(), cols, 1e-7f))
                << "BF16 conversion does not match llama.cpp";
        }

        /**
         * @brief Test FP32 model loading
         *
         * FP32 is the native float format - no conversion needed.
         * This test validates that FP32 tensors load correctly and data() returns proper values.
         */
        TEST_F(DequantEquivalencyTest, FP32_Loading)
        {
            model_path_ = "/workspaces/llaminar/models/Gemini-Distill-Qwen2.5-0.5B-ead-fp32.gguf";

            ModelLoader loader;
            if (!loader.loadModel(model_path_))
            {
                GTEST_SKIP() << "FP32 model not available";
                return;
            }

            // Try to find an FP32 tensor (usually all weights are in FP32)
            auto weight = loadWeight("blk.0.attn_q.weight");
            if (!weight)
            {
                weight = loadWeight("token_embd.weight");
            }
            if (!weight)
            {
                GTEST_SKIP() << "FP32 weight not available";
                return;
            }

            auto fp32_tensor = std::dynamic_pointer_cast<FP32Tensor>(weight);
            if (!fp32_tensor)
            {
                GTEST_SKIP() << "No FP32 tensors found in this model";
                return;
            }

            const auto &shape = fp32_tensor->shape();
            size_t rows = shape[0];
            size_t cols = shape[1];

            std::cout << "Testing FP32 tensor with shape: " << rows << " × " << cols << std::endl;

            // FP32 doesn't need conversion - data() returns float* directly
            const float *fp32_data = fp32_tensor->data();
            ASSERT_NE(fp32_data, nullptr) << "FP32 data() returned nullptr";

            // Just verify we can read values without crashes
            std::cout << "\nFirst 10 FP32 values: ";
            for (int i = 0; i < 10 && i < cols; ++i)
            {
                std::cout << fp32_data[i] << " ";
            }
            std::cout << "\n"
                      << std::endl;

            // Verify values are reasonable (not NaN, not Inf for typical weights)
            size_t nan_count = 0;
            size_t inf_count = 0;
            for (size_t i = 0; i < cols && i < 1000; ++i)
            {
                if (std::isnan(fp32_data[i]))
                    nan_count++;
                if (std::isinf(fp32_data[i]))
                    inf_count++;
            }

            std::cout << "Sanity check (first 1000 elements): NaN=" << nan_count << ", Inf=" << inf_count << std::endl;
            EXPECT_EQ(nan_count, 0) << "FP32 tensor contains NaN values";
            EXPECT_EQ(inf_count, 0) << "FP32 tensor contains Inf values";

            std::cout << "FP32 model loading test passed!" << std::endl;
        }

        /**
         * @brief Test FP16 dequantization equivalency
         *
         * FP16 is IEEE 754 half-precision float format.
         * This test validates that our FP16 to FP32 conversion matches llama.cpp's.
         */
        TEST_F(DequantEquivalencyTest, FP16_Equivalency)
        {
            model_path_ = "/workspaces/llaminar/models/qwen2.5-0.5b-instruct-fp16.gguf";

            ModelLoader loader;
            if (!loader.loadModel(model_path_))
            {
                GTEST_SKIP() << "FP16 model not available";
                return;
            }

            // Try to find an FP16 tensor (usually model weights are in FP16)
            auto weight = loadWeight("blk.0.attn_q.weight");
            if (!weight)
            {
                weight = loadWeight("token_embd.weight");
            }
            if (!weight)
            {
                GTEST_SKIP() << "FP16 weight not available";
                return;
            }

            auto fp16_tensor = std::dynamic_pointer_cast<FP16Tensor>(weight);
            if (!fp16_tensor)
            {
                GTEST_SKIP() << "No FP16 tensors found in this model";
                return;
            }

            const auto &shape = fp16_tensor->shape();
            size_t rows = shape[0];
            size_t cols = shape[1];

            std::cout << "Testing FP16 tensor with shape: " << rows << " × " << cols << std::endl;

            // Test a single row
            size_t row_idx = 0;
            std::vector<float> llaminar_output(cols);
            std::vector<float> llamacpp_output(cols);

            // Dequantize with Llaminar using to_fp32 method
            // Note: to_fp32 needs to convert the entire tensor, then we extract the row we want
            const uint16_t *fp16_data = fp16_tensor->fp16_data();
            size_t total_elements = rows * cols;
            std::vector<float> full_llaminar_output(total_elements);
            fp16_tensor->to_fp32(full_llaminar_output.data(), total_elements);

            // Extract the row we want to test
            std::copy(full_llaminar_output.begin() + row_idx * cols,
                      full_llaminar_output.begin() + (row_idx + 1) * cols,
                      llaminar_output.begin());

            // Dequantize with llama.cpp using ggml_fp16_to_fp32
            // FP16 uses IEEE 754 half-precision format
            for (size_t i = 0; i < cols; ++i)
            {
                llamacpp_output[i] = ggml_fp16_to_fp32(fp16_data[row_idx * cols + i]);
            }

            // Print first few values from each
            std::cout << "\nFirst 10 Llaminar values: ";
            for (int i = 0; i < 10 && i < cols; ++i)
            {
                std::cout << llaminar_output[i] << " ";
            }
            std::cout << "\n\nFirst 10 llama.cpp values: ";
            for (int i = 0; i < 10 && i < cols; ++i)
            {
                std::cout << llamacpp_output[i] << " ";
            }
            std::cout << "\n"
                      << std::endl;

            // FP16 should have exact conversion (IEEE 754 standard)
            EXPECT_TRUE(compareOutputs(llaminar_output.data(), llamacpp_output.data(), cols, 1e-7f))
                << "FP16 conversion does not match llama.cpp";
        }

    } // namespace test
} // namespace llaminar2
