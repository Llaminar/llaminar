/**
 * @file test_dequant_accuracy.cpp
 * @brief Accuracy tests comparing Llaminar vs llama.cpp dequantization
 *
 * This test suite verifies that Llaminar's quantized tensor implementations
 * produce identical results to llama.cpp's reference dequantization functions.
 *
 * Tests cover all supported quantization types:
 * - Q formats: Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_0, Q5_K, Q6_K, Q8_0
 * - IQ formats: IQ1_S, IQ1_M, IQ2_XXS, IQ2_M, IQ3_XXS, IQ4_NL, IQ4_XS
 *
 * @author David Sanftenberg
 * @date October 2025
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <cmath>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>

// Llaminar headers
#include "ModelLoader.h"
#include "tensors/QuantizedTensorBase.h"

// Note: Not all quantization types have dedicated tensor classes.
// Q4_K uses Q4_KTensor, Q5_K uses Q5_KTensor (part of *_K family).
// We use ModelLoader to get tensors which handles all types correctly.

// llama.cpp headers
extern "C"
{
#include "ggml-quants.h"
}

namespace
{

    /**
     * @brief Test configuration for each quantization type
     */
    struct QuantTestConfig
    {
        std::string name;          // Test name
        std::string model_pattern; // Glob pattern to find model
        GGUFTensorType quant_type; // Llaminar quantization type
        std::string tensor_name;   // Tensor to test (e.g., "blk.0.attn_q.weight")
        size_t max_rows;           // Max rows to test (for speed)
    };

    // Test configurations for all supported formats
    std::vector<QuantTestConfig> getTestConfigs()
    {
        return {
            // Q formats (0.5B models)
            {"Q2_K", "qwen2.5-0.5b-instruct-q2_k.gguf", GGUFTensorType::Q2_K, "blk.0.attn_q.weight", 100},
            {"Q3_K", "qwen2.5-0.5b-instruct-q3_k_m.gguf", GGUFTensorType::Q3_K, "blk.0.attn_q.weight", 100},
            {"Q4_0", "qwen2.5-0.5b-instruct-q4_0.gguf", GGUFTensorType::Q4_0, "blk.0.attn_q.weight", 100},
            {"Q4_K", "qwen2.5-0.5b-instruct-q4_k_m.gguf", GGUFTensorType::Q4_K, "blk.0.attn_q.weight", 100},
            {"Q5_0", "qwen2.5-0.5b-instruct-q5_0.gguf", GGUFTensorType::Q5_0, "blk.0.attn_q.weight", 100},
            {"Q5_K", "qwen2.5-0.5b-instruct-q5_k_m.gguf", GGUFTensorType::Q5_K, "blk.0.attn_q.weight", 100},
            {"Q6_K", "qwen2.5-0.5b-instruct-q6_k.gguf", GGUFTensorType::Q6_K, "blk.0.attn_q.weight", 100},
            {"Q8_0", "qwen2.5-0.5b-instruct-q8_0.gguf", GGUFTensorType::Q8_0, "blk.0.attn_q.weight", 100},

            // IQ formats (7B models - test fewer rows for speed)
            {"IQ1_S", "Qwen2.5-VL-7B-Instruct-UD-IQ1_S.gguf", GGUFTensorType::IQ1_S, "blk.0.attn_q.weight", 50},
            {"IQ1_M", "Qwen2.5-VL-7B-Instruct-UD-IQ1_M.gguf", GGUFTensorType::IQ1_M, "blk.0.attn_q.weight", 50},
            {"IQ2_XXS", "Qwen2.5-VL-7B-Instruct-UD-IQ2_XXS.gguf", GGUFTensorType::IQ2_XXS, "blk.0.attn_q.weight", 50},
            {"IQ2_M", "Qwen2.5-VL-7B-Instruct-UD-IQ2_M.gguf", GGUFTensorType::IQ2_M, "blk.0.attn_q.weight", 50},
            {"IQ3_XXS", "Qwen2.5-VL-7B-Instruct-UD-IQ3_XXS.gguf", GGUFTensorType::IQ3_XXS, "blk.0.attn_q.weight", 50},
            {"IQ4_NL", "Qwen2.5-VL-7B-Instruct-IQ4_NL.gguf", GGUFTensorType::IQ4_NL, "blk.0.attn_q.weight", 50},
            {"IQ4_XS", "Qwen2.5-VL-7B-Instruct-IQ4_XS.gguf", GGUFTensorType::IQ4_XS, "blk.0.attn_q.weight", 50},
        };
    }

    /**
     * @brief Compare two float arrays with detailed error reporting
     */
    struct ComparisonResult
    {
        bool passed;
        double max_abs_diff;
        double mean_abs_diff;
        double rel_l2_error;
        size_t num_mismatches;
        size_t total_elements;
    };

    ComparisonResult compareArrays(const float *llaminar, const float *llamacpp, size_t n,
                                   double abs_tol = 1e-4, double rel_tol = 1e-3)
    {
        ComparisonResult result = {true, 0.0, 0.0, 0.0, 0, n};

        double sum_abs_diff = 0.0;
        double sum_sq_llaminar = 0.0;
        double sum_sq_diff = 0.0;

        for (size_t i = 0; i < n; ++i)
        {
            double diff = std::abs(llaminar[i] - llamacpp[i]);
            sum_abs_diff += diff;
            sum_sq_diff += diff * diff;
            sum_sq_llaminar += llaminar[i] * llaminar[i];

            if (diff > result.max_abs_diff)
            {
                result.max_abs_diff = diff;
            }

            // Check tolerance
            double abs_threshold = abs_tol;
            double rel_threshold = rel_tol * std::max(std::abs(llaminar[i]), std::abs(llamacpp[i]));
            if (diff > abs_threshold && diff > rel_threshold)
            {
                result.num_mismatches++;
            }
        }

        result.mean_abs_diff = sum_abs_diff / n;
        result.rel_l2_error = std::sqrt(sum_sq_diff) / (std::sqrt(sum_sq_llaminar) + 1e-10);
        result.passed = (result.num_mismatches == 0);

        return result;
    }

    /**
     * @brief Call appropriate llama.cpp dequantization function
     */
    void llamacppDequantize(GGUFTensorType type, const void *quantized, float *output, int64_t k)
    {
        switch (type)
        {
        case GGUFTensorType::Q2_K:
            dequantize_row_q2_K(static_cast<const block_q2_K *>(quantized), output, k);
            break;
        case GGUFTensorType::Q3_K:
            dequantize_row_q3_K(static_cast<const block_q3_K *>(quantized), output, k);
            break;
        case GGUFTensorType::Q4_0:
            dequantize_row_q4_0(static_cast<const block_q4_0 *>(quantized), output, k);
            break;
        case GGUFTensorType::Q4_K:
            dequantize_row_q4_K(static_cast<const block_q4_K *>(quantized), output, k);
            break;
        case GGUFTensorType::Q5_0:
            dequantize_row_q5_0(static_cast<const block_q5_0 *>(quantized), output, k);
            break;
        case GGUFTensorType::Q5_K:
            dequantize_row_q5_K(static_cast<const block_q5_K *>(quantized), output, k);
            break;
        case GGUFTensorType::Q6_K:
            dequantize_row_q6_K(static_cast<const block_q6_K *>(quantized), output, k);
            break;
        case GGUFTensorType::Q8_0:
            dequantize_row_q8_0(static_cast<const block_q8_0 *>(quantized), output, k);
            break;
        case GGUFTensorType::IQ1_S:
            dequantize_row_iq1_s(static_cast<const block_iq1_s *>(quantized), output, k);
            break;
        case GGUFTensorType::IQ1_M:
            dequantize_row_iq1_m(static_cast<const block_iq1_m *>(quantized), output, k);
            break;
        case GGUFTensorType::IQ2_XXS:
            dequantize_row_iq2_xxs(static_cast<const block_iq2_xxs *>(quantized), output, k);
            break;
        case GGUFTensorType::IQ2_M:
            iq2xs_init_impl(GGML_TYPE_IQ2_M); // Initialize lookup tables
            dequantize_row_iq2_xs(static_cast<const block_iq2_xs *>(quantized), output, k);
            break;
        case GGUFTensorType::IQ3_XXS:
            dequantize_row_iq3_xxs(static_cast<const block_iq3_xxs *>(quantized), output, k);
            break;
        case GGUFTensorType::IQ4_NL:
            dequantize_row_iq4_nl(static_cast<const block_iq4_nl *>(quantized), output, k);
            break;
        case GGUFTensorType::IQ4_XS:
            dequantize_row_iq4_xs(static_cast<const block_iq4_xs *>(quantized), output, k);
            break;
        default:
            throw std::runtime_error("Unsupported quantization type for llama.cpp comparison");
        }
    }

    /**
     * @brief Accuracy test fixture
     */
    class DequantAccuracyTest : public ::testing::TestWithParam<QuantTestConfig>
    {
    protected:
        void SetUp() override
        {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);

            // Only rank 0 performs tests
            if (rank != 0)
            {
                GTEST_SKIP() << "Non-rank-0 process skipping accuracy tests";
            }
        }
    };

    TEST_P(DequantAccuracyTest, CompareWithLlamaCpp)
    {
        const auto &config = GetParam();

        // Load model
        std::string model_path = "models/" + config.model_pattern;
        ModelLoader loader(model_path);

        // Get quantized tensor
        auto tensor_opt = loader.getTensor(config.tensor_name, config.quant_type);
        ASSERT_TRUE(tensor_opt.has_value()) << "Failed to load tensor: " << config.tensor_name;

        auto quant_tensor = std::dynamic_pointer_cast<QuantizedTensorBase>(tensor_opt.value());
        ASSERT_NE(quant_tensor, nullptr) << "Tensor is not quantized";

        // Get tensor dimensions
        const auto &shape = quant_tensor->shape();
        ASSERT_EQ(shape.size(), 2) << "Expected 2D tensor";

        size_t rows = shape[0];
        size_t cols = shape[1];

        // Limit rows for performance
        rows = std::min(rows, config.max_rows);

        std::cout << "\n"
                  << config.name << " Test:\n";
        std::cout << "  Model: " << config.model_pattern << "\n";
        std::cout << "  Tensor: " << config.tensor_name << "\n";
        std::cout << "  Shape: " << rows << " × " << cols << "\n";

        // Allocate output buffers
        std::vector<float> llaminar_output(rows * cols);
        std::vector<float> llamacpp_output(rows * cols);

        // Test row-by-row for detailed diagnostics
        size_t total_mismatches = 0;
        double max_error = 0.0;

        for (size_t row = 0; row < rows; ++row)
        {
            // Llaminar dequantization
            quant_tensor->decodeRow(row, llaminar_output.data() + row * cols);

            // llama.cpp dequantization
            const void *quantized_row = quant_tensor->getQuantizedRow(row);
            llamacppDequantize(config.quant_type, quantized_row,
                               llamacpp_output.data() + row * cols, cols);

            // Compare this row
            auto result = compareArrays(
                llaminar_output.data() + row * cols,
                llamacpp_output.data() + row * cols,
                cols);

            if (!result.passed)
            {
                total_mismatches += result.num_mismatches;
                max_error = std::max(max_error, result.max_abs_diff);

                if (total_mismatches < 10)
                { // Print first few failures
                    std::cout << "  Row " << row << " MISMATCH: "
                              << result.num_mismatches << " / " << cols << " elements, "
                              << "max_diff=" << result.max_abs_diff << "\n";
                }
            }
        }

        // Overall comparison
        auto overall = compareArrays(
            llaminar_output.data(),
            llamacpp_output.data(),
            rows * cols);

        std::cout << "  Results:\n";
        std::cout << "    Total elements: " << (rows * cols) << "\n";
        std::cout << "    Mismatches: " << overall.num_mismatches << "\n";
        std::cout << "    Max abs diff: " << overall.max_abs_diff << "\n";
        std::cout << "    Mean abs diff: " << overall.mean_abs_diff << "\n";
        std::cout << "    Rel L2 error: " << overall.rel_l2_error << "\n";

        // Strict accuracy requirement
        EXPECT_TRUE(overall.passed)
            << config.name << " dequantization differs from llama.cpp reference\n"
            << "  Mismatches: " << overall.num_mismatches << " / " << (rows * cols) << "\n"
            << "  Max abs diff: " << overall.max_abs_diff << "\n"
            << "  Rel L2 error: " << overall.rel_l2_error;
    }

    INSTANTIATE_TEST_SUITE_P(
        AllQuantTypes,
        DequantAccuracyTest,
        ::testing::ValuesIn(getTestConfigs()),
        [](const ::testing::TestParamInfo<QuantTestConfig> &info)
        {
            return info.param.name;
        });

} // namespace

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Suppress output from non-rank-0 processes
    if (rank != 0)
    {
        ::testing::TestEventListeners &listeners =
            ::testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());
    }

    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
