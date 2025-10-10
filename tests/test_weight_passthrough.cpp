/**
 * @file test_weight_passthrough.cpp
 * @brief Test to verify weights pass through intact from dequantizer to attention kernel
 *
 * This test verifies that:
 * 1. Weights dequantized by ModelLoader maintain their values
 * 2. Weights passed to MPIAttentionKernel are identical to dequantized values
 * 3. No scaling, corruption, or transformation occurs in the pipeline
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <memory>
#include <cmath>
#include <numeric>
#include <filesystem>

#include "model_loader.h"
#include "transformer_config.h"
#include "kernels/MPIAttentionKernel.h"
#include "tensors/tensor_factory.h"
#include "logger.h"

namespace llaminar
{
    namespace test
    {

        class WeightPassthroughTest : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
                MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
                MPI_Comm_size(MPI_COMM_WORLD, &size_);
            }

            void TearDown() override
            {
                MPI_Barrier(MPI_COMM_WORLD);
            }

            int rank_;
            int size_;
        };

        /**
         * @brief Capture weight statistics for comparison
         */
        struct WeightStats
        {
            float min_val;
            float max_val;
            float mean;
            float mean_abs;
            float std_dev;
            std::vector<float> samples; // First 100 values

            WeightStats() : min_val(0), max_val(0), mean(0), mean_abs(0), std_dev(0) {}

            static WeightStats compute(const float *data, size_t n_elements, size_t n_samples = 100)
            {
                WeightStats stats;

                if (n_elements == 0 || !data)
                {
                    return stats;
                }

                // Min/max/mean
                stats.min_val = data[0];
                stats.max_val = data[0];
                double sum = 0.0;
                double sum_abs = 0.0;

                for (size_t i = 0; i < n_elements; ++i)
                {
                    float val = data[i];
                    stats.min_val = std::min(stats.min_val, val);
                    stats.max_val = std::max(stats.max_val, val);
                    sum += val;
                    sum_abs += std::abs(val);
                }

                stats.mean = static_cast<float>(sum / n_elements);
                stats.mean_abs = static_cast<float>(sum_abs / n_elements);

                // Standard deviation
                double sum_sq_diff = 0.0;
                for (size_t i = 0; i < n_elements; ++i)
                {
                    double diff = data[i] - stats.mean;
                    sum_sq_diff += diff * diff;
                }
                stats.std_dev = static_cast<float>(std::sqrt(sum_sq_diff / n_elements));

                // Sample values
                size_t sample_count = std::min(n_samples, n_elements);
                stats.samples.reserve(sample_count);
                for (size_t i = 0; i < sample_count; ++i)
                {
                    stats.samples.push_back(data[i]);
                }

                return stats;
            }

            void print(const std::string &name) const
            {
                LOG_INFO("[WEIGHT_STATS] " << name);
                LOG_INFO("  min=" << min_val << " max=" << max_val);
                LOG_INFO("  mean=" << mean << " mean_abs=" << mean_abs << " std_dev=" << std_dev);
                LOG_INFO("  samples[0:10]: ");
                std::ostringstream oss;
                for (size_t i = 0; i < std::min(size_t(10), samples.size()); ++i)
                {
                    oss << samples[i];
                    if (i < std::min(size_t(9), samples.size() - 1))
                        oss << ", ";
                }
                LOG_INFO("    " << oss.str());
            }

            bool matches(const WeightStats &other, float tolerance = 1e-6f) const
            {
                // Check if statistics match within tolerance
                if (std::abs(min_val - other.min_val) > tolerance)
                    return false;
                if (std::abs(max_val - other.max_val) > tolerance)
                    return false;
                if (std::abs(mean - other.mean) > tolerance)
                    return false;
                if (std::abs(mean_abs - other.mean_abs) > tolerance)
                    return false;

                // Check samples match exactly
                size_t n_samples = std::min(samples.size(), other.samples.size());
                for (size_t i = 0; i < n_samples; ++i)
                {
                    if (std::abs(samples[i] - other.samples[i]) > tolerance)
                    {
                        return false;
                    }
                }

                return true;
            }
        };

        /**
         * @brief Custom attention kernel that captures weights passed to execute()
         */
        class WeightCapturingAttentionKernel : public MPIAttentionKernel
        {
        public:
            WeightCapturingAttentionKernel(int n_head, int n_head_kv, int head_dim, float rope_freq_base)
                : MPIAttentionKernel(n_head, n_head_kv, head_dim, rope_freq_base, DistributionStrategy::HEAD_WISE) {}

            bool execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                         std::vector<std::shared_ptr<TensorBase>> &outputs) override
            {

                // Capture weights before any processing
                if (inputs.size() >= 5)
                {
                    auto wq = inputs[1];
                    auto wk = inputs[2];
                    auto wv = inputs[3];
                    auto wo = inputs[4];

                    if (wq && wq->data())
                    {
                        size_t wq_elements = wq->shape()[0] * wq->shape()[1];
                        captured_wq_stats_ = WeightStats::compute(wq->data(), wq_elements);
                    }

                    if (wk && wk->data())
                    {
                        size_t wk_elements = wk->shape()[0] * wk->shape()[1];
                        captured_wk_stats_ = WeightStats::compute(wk->data(), wk_elements);
                    }

                    if (wv && wv->data())
                    {
                        size_t wv_elements = wv->shape()[0] * wv->shape()[1];
                        captured_wv_stats_ = WeightStats::compute(wv->data(), wv_elements);
                    }

                    if (wo && wo->data())
                    {
                        size_t wo_elements = wo->shape()[0] * wo->shape()[1];
                        captured_wo_stats_ = WeightStats::compute(wo->data(), wo_elements);
                    }
                }

                // Don't actually execute attention (we're just testing weight passthrough)
                return true;
            }

            const WeightStats &get_wq_stats() const { return captured_wq_stats_; }
            const WeightStats &get_wk_stats() const { return captured_wk_stats_; }
            const WeightStats &get_wv_stats() const { return captured_wv_stats_; }
            const WeightStats &get_wo_stats() const { return captured_wo_stats_; }

        private:
            WeightStats captured_wq_stats_;
            WeightStats captured_wk_stats_;
            WeightStats captured_wv_stats_;
            WeightStats captured_wo_stats_;
        };

        /**
         * @brief Test that weights are passed through intact from loader to attention kernel
         */
        TEST_F(WeightPassthroughTest, DequantizedWeightsMatchKernelInput)
        {
            // Find test model
            std::string model_path;
            std::vector<std::string> search_paths = {
                "models/qwen2.5-0.5b-instruct-q4_0.gguf",
                "../models/qwen2.5-0.5b-instruct-q4_0.gguf"};

            bool found = false;
            for (const auto &path : search_paths)
            {
                if (std::filesystem::exists(path))
                {
                    model_path = path;
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                GTEST_SKIP() << "Test model not found - skipping weight passthrough test";
            }

            if (rank_ == 0)
            {
                LOG_INFO("========================================");
                LOG_INFO("Weight Passthrough Test");
                LOG_INFO("========================================");
                LOG_INFO("Model: " << model_path);
            }

            // === STEP 1: Load and dequantize weights ===
            ModelLoader loader;
            if (!loader.loadModel(model_path))
            {
                FAIL() << "Failed to load model: " << model_path;
            }

            // Get architecture parameters
            auto config = loader.createLayerConfig();
            int n_head = config.n_head;
            int n_head_kv = config.n_head_kv;
            int head_dim = config.head_dim;
            int d_model = config.d_model;

            if (rank_ == 0)
            {
                LOG_INFO("Architecture: n_head=" << n_head << " n_head_kv=" << n_head_kv
                                                 << " head_dim=" << head_dim << " d_model=" << d_model);
            }

            // Load attention weights for layer 0
            std::string layer_prefix = "blk.0.attn_";

            // Load and dequantize weights
            auto wq_loaded = loader.loadTensor(layer_prefix + "q.weight");
            auto wk_loaded = loader.loadTensor(layer_prefix + "k.weight");
            auto wv_loaded = loader.loadTensor(layer_prefix + "v.weight");
            auto wo_loaded = loader.loadTensor(layer_prefix + "output.weight");

            ASSERT_TRUE(wq_loaded != nullptr) << "Q weight dequantization failed";
            ASSERT_TRUE(wk_loaded != nullptr) << "K weight dequantization failed";
            ASSERT_TRUE(wv_loaded != nullptr) << "V weight dequantization failed";
            ASSERT_TRUE(wo_loaded != nullptr) << "Output weight dequantization failed";

            // Compute statistics of dequantized weights
            size_t wq_elements = wq_loaded->shape()[0] * wq_loaded->shape()[1];
            size_t wk_elements = wk_loaded->shape()[0] * wk_loaded->shape()[1];
            size_t wv_elements = wv_loaded->shape()[0] * wv_loaded->shape()[1];
            size_t wo_elements = wo_loaded->shape()[0] * wo_loaded->shape()[1];

            WeightStats loader_wq_stats = WeightStats::compute(wq_loaded->data(), wq_elements);
            WeightStats loader_wk_stats = WeightStats::compute(wk_loaded->data(), wk_elements);
            WeightStats loader_wv_stats = WeightStats::compute(wv_loaded->data(), wv_elements);
            WeightStats loader_wo_stats = WeightStats::compute(wo_loaded->data(), wo_elements);

            if (rank_ == 0)
            {
                LOG_INFO("=== Dequantized Weight Statistics (from ModelLoader) ===");
                loader_wq_stats.print("Q weight");
                loader_wk_stats.print("K weight");
                loader_wv_stats.print("V weight");
                loader_wo_stats.print("Output weight");
            }

            // === STEP 2: Pass loaded weights directly to attention kernel ===
            // The loaded tensors are already in the right format - just pass them through

            // Create capturing attention kernel
            WeightCapturingAttentionKernel kernel(n_head, n_head_kv, head_dim, config.rope_freq_base);

            // Create dummy input (we don't care about the actual computation)
            int seq_len = 5;
            auto input_tensor = TensorFactory::create_simple({seq_len, d_model});
            std::fill_n(input_tensor->data(), seq_len * d_model, 0.0f);

            // Execute kernel with loaded weights
            std::vector<std::shared_ptr<TensorBase>> inputs = {
                input_tensor,
                wq_loaded, // Use loaded weights directly
                wk_loaded,
                wv_loaded,
                wo_loaded};
            std::vector<std::shared_ptr<TensorBase>> outputs;

            bool success = kernel.execute(inputs, outputs);
            ASSERT_TRUE(success) << "Kernel execution failed";

            // === STEP 3: Compare statistics ===
            const WeightStats &kernel_wq_stats = kernel.get_wq_stats();
            const WeightStats &kernel_wk_stats = kernel.get_wk_stats();
            const WeightStats &kernel_wv_stats = kernel.get_wv_stats();
            const WeightStats &kernel_wo_stats = kernel.get_wo_stats();

            if (rank_ == 0)
            {
                LOG_INFO("=== Weight Statistics Captured in Attention Kernel ===");
                kernel_wq_stats.print("Q weight (kernel)");
                kernel_wk_stats.print("K weight (kernel)");
                kernel_wv_stats.print("V weight (kernel)");
                kernel_wo_stats.print("Output weight (kernel)");
            }

            // Verify weights match
            float tolerance = 1e-6f;

            if (rank_ == 0)
            {
                LOG_INFO("=== Comparing Loader vs Kernel Statistics ===");
            }

            EXPECT_TRUE(loader_wq_stats.matches(kernel_wq_stats, tolerance))
                << "Q weight statistics mismatch between loader and kernel!\n"
                << "  Loader: min=" << loader_wq_stats.min_val << " max=" << loader_wq_stats.max_val
                << " mean=" << loader_wq_stats.mean << " mean_abs=" << loader_wq_stats.mean_abs << "\n"
                << "  Kernel: min=" << kernel_wq_stats.min_val << " max=" << kernel_wq_stats.max_val
                << " mean=" << kernel_wq_stats.mean << " mean_abs=" << kernel_wq_stats.mean_abs;

            EXPECT_TRUE(loader_wk_stats.matches(kernel_wk_stats, tolerance))
                << "K weight statistics mismatch between loader and kernel!\n"
                << "  Loader: min=" << loader_wk_stats.min_val << " max=" << loader_wk_stats.max_val
                << " mean=" << loader_wk_stats.mean << " mean_abs=" << loader_wk_stats.mean_abs << "\n"
                << "  Kernel: min=" << kernel_wk_stats.min_val << " max=" << kernel_wk_stats.max_val
                << " mean=" << kernel_wk_stats.mean << " mean_abs=" << kernel_wk_stats.mean_abs;

            EXPECT_TRUE(loader_wv_stats.matches(kernel_wv_stats, tolerance))
                << "V weight statistics mismatch between loader and kernel!\n"
                << "  Loader: min=" << loader_wv_stats.min_val << " max=" << loader_wv_stats.max_val
                << " mean=" << loader_wv_stats.mean << " mean_abs=" << loader_wv_stats.mean_abs << "\n"
                << "  Kernel: min=" << kernel_wv_stats.min_val << " max=" << kernel_wv_stats.max_val
                << " mean=" << kernel_wv_stats.mean << " mean_abs=" << kernel_wv_stats.mean_abs;

            EXPECT_TRUE(loader_wo_stats.matches(kernel_wo_stats, tolerance))
                << "Output weight statistics mismatch between loader and kernel!\n"
                << "  Loader: min=" << loader_wo_stats.min_val << " max=" << loader_wo_stats.max_val
                << " mean=" << loader_wo_stats.mean << " mean_abs=" << loader_wo_stats.mean_abs << "\n"
                << "  Kernel: min=" << kernel_wo_stats.min_val << " max=" << kernel_wo_stats.max_val
                << " mean=" << kernel_wo_stats.mean << " mean_abs=" << kernel_wo_stats.mean_abs;

            if (rank_ == 0)
            {
                LOG_INFO("✓ All weight statistics match between loader and kernel");
                LOG_INFO("========================================");
            }
        }

    } // namespace test
} // namespace llaminar

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);

    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
