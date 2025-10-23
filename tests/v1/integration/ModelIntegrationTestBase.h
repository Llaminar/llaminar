/**
 * @file ModelIntegrationTestBase.h
 * @brief Base class for model integration tests with PyTorch layer-by-layer parity
 * @author David Sanftenberg
 *
 * This framework provides automated layer-by-layer correctness verification for
 * all models (Qwen, LLaMA, etc.) across both prefill and decode stages.
 */

#ifndef MODEL_INTEGRATION_TEST_BASE_H
#define MODEL_INTEGRATION_TEST_BASE_H

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include "ParityTestFramework.h"
#include "AbstractPipeline.h"

namespace llaminar
{
    namespace testing
    {

        // Parity namespace types
        using llaminar::parity::ComparisonResult;
        using llaminar::parity::PytorchSnapshotLoader;
        using llaminar::parity::SnapshotRegistry;
        using llaminar::parity::TensorSnapshot;

        /**
         * @brief Tolerance configuration for layer-by-layer comparisons
         */
        struct ToleranceConfig
        {
            float max_abs; ///< Maximum absolute error
            float rel_l2;  ///< Relative L2 error

            ToleranceConfig(float abs = 0.1f, float l2 = 0.01f)
                : max_abs(abs), rel_l2(l2) {}

            /// Get tolerances for specific model precision
            static ToleranceConfig for_precision(const std::string &precision)
            {
                if (precision == "fp32")
                {
                    return ToleranceConfig(0.001f, 0.0001f); // Tight tolerance
                }
                else if (precision == "q6_k")
                {
                    return ToleranceConfig(0.01f, 0.001f); // Moderate tolerance
                }
                else if (precision == "q4_0")
                {
                    return ToleranceConfig(0.1f, 0.01f); // Relaxed tolerance
                }
                else
                {
                    return ToleranceConfig(0.1f, 0.01f); // Default conservative
                }
            }
        };

        /**
         * @brief Test configuration for a specific model/stage/precision combination
         */
        struct TestConfig
        {
            std::string model_name;       ///< e.g., "qwen2.5-0.5b-instruct"
            std::string precision;        ///< e.g., "q4_0", "q6_k", "fp32"
            std::string stage;            ///< "prefill" or "decode"
            std::vector<int> tokens;      ///< Token sequence to test
            std::string golden_reference; ///< Path to PyTorch NPZ snapshot
            ToleranceConfig tolerances;   ///< Comparison tolerances

            /// Construct full model path
            std::string get_model_path() const
            {
                return "models/" + model_name + "-" + precision + ".gguf";
            }

            /// Construct golden reference path relative to tests/golden_references/
            std::string get_golden_path() const
            {
                if (!golden_reference.empty())
                {
                    return golden_reference;
                }
                // Auto-generate path
                std::string tokens_str;
                for (size_t i = 0; i < tokens.size(); ++i)
                {
                    if (i > 0)
                        tokens_str += "_";
                    tokens_str += std::to_string(tokens[i]);
                }
                return "tests/golden_references/" + model_name + "-" + precision + "/" +
                       stage + "_tokens_" + tokens_str + ".npz";
            }
        };

        /**
         * @brief Base test fixture for model integration tests
         *
         * Provides common infrastructure for:
         * - Loading PyTorch golden references
         * - Running Llaminar pipeline
         * - Layer-by-layer comparison
         * - Detailed divergence reporting
         */
        class ModelIntegrationTestBase : public ::testing::Test
        {
        protected:
            void TearDown() override;

            /**
             * @brief Run integration test for given configuration
             * @param config Test configuration (model, precision, stage, tokens)
             * @return true if all layers match within tolerance
             */
            bool run_integration_test(const TestConfig &config);

            /**
             * @brief Load PyTorch golden reference from NPZ
             * @param npz_path Path to NPZ file
             * @param source_name Source identifier for registry
             * @return Number of snapshots loaded
             */
            size_t load_golden_reference(const std::string &npz_path,
                                         const std::string &source_name = "pytorch");

            /**
             * @brief Run Llaminar pipeline and capture snapshots
             * @param model_path Path to GGUF model
             * @param tokens Token sequence to process
             * @param is_prefill Whether this is prefill (vs decode)
             * @return true if pipeline executed successfully
             */
            bool run_llaminar_with_capture(const std::string &model_path,
                                           const std::vector<int> &tokens,
                                           bool is_prefill);

            /**
             * @brief Compare Llaminar vs PyTorch snapshots layer-by-layer
             * @param tolerances Comparison tolerances
             * @param first_diverging_layer [out] First layer that diverges
             * @return true if all layers match
             */
            bool compare_snapshots(const ToleranceConfig &tolerances,
                                   std::string &first_diverging_layer);

            /**
             * @brief Check if golden reference file exists
             * @param path Path to golden reference
             * @return true if file exists and is readable
             */
            bool golden_reference_exists(const std::string &path) const;

            /**
             * @brief Print detailed divergence report
             * @param diverging_layer Layer identifier that diverged
             * @param pytorch_snapshot PyTorch snapshot
             * @param llaminar_snapshot Llaminar snapshot
             * @param tolerances Tolerances used
             */
            void print_divergence_report(const std::string &diverging_layer,
                                         const TensorSnapshot &pytorch_snapshot,
                                         const TensorSnapshot &llaminar_snapshot,
                                         const ToleranceConfig &tolerances);
        };

        /**
         * @brief Parameterized test fixture for model/precision combinations
         */
        class ParameterizedModelTest : public ModelIntegrationTestBase,
                                       public ::testing::WithParamInterface<TestConfig>
        {
        };

        /**
         * @brief Parse PyTorch snapshot key to extract layer index and stage name
         * @param key PyTorch key (e.g., "pytorch:layer_0_attn_out")
         * @param layer_index [out] Layer index (-1 for global)
         * @param stage_name [out] Stage name
         * @return true if successfully parsed
         */
        bool parse_pytorch_key(const std::string &key, int &layer_index, std::string &stage_name);

        /**
         * @brief Map PyTorch stage name to Llaminar PipelineStage enum name
         * @param pytorch_stage PyTorch stage name (e.g., "input_norm_out")
         * @return Llaminar stage name (e.g., "ATTENTION_NORM")
         */
        std::string map_pytorch_stage_to_llaminar(const std::string &pytorch_stage);

    } // namespace testing
} // namespace llaminar

#endif // MODEL_INTEGRATION_TEST_BASE_H
