/**
 * @file test_dequant_performance.cpp
 * @brief Performance benchmarks comparing Llaminar vs llama.cpp dequantization
 *
 * This benchmark suite measures and compares the performance of Llaminar's
 * optimized SIMD dequantization against llama.cpp's reference implementation.
 *
 * Benchmarks cover:
 * - All Q* formats: Q2_K through Q8_0
 * - All IQ* formats: IQ1_S through IQ4_XS
 * - Various tensor sizes (small, medium, large)
 * - OMP parallelization for both implementations
 *
 * @author David Sanftenberg
 * @date October 2025
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <iomanip>
#include <sstream>

// Llaminar headers
#include "ModelLoader.h"
#include "tensors/QuantizedTensorBase.h"

// Note: Not all quantization types have dedicated tensor classes.
// We use ModelLoader to get tensors which handles all types correctly.

// llama.cpp headers
extern "C"
{
#include "ggml-quants.h"
}

namespace
{

    /**
     * @brief Benchmark configuration
     */
    struct BenchConfig
    {
        std::string name;
        std::string model_pattern;
        GGUFTensorType quant_type;
        std::string tensor_name;
        size_t warmup_iters;
        size_t bench_iters;
        size_t max_rows; // For limiting large models
    };

    std::vector<BenchConfig> getBenchConfigs()
    {
        return {
            // Q formats - 0.5B models (manageable size)
            {"Q2_K", "qwen2.5-0.5b-instruct-q2_k.gguf", GGUFTensorType::Q2_K, "blk.0.attn_q.weight", 3, 10, 500},
            {"Q3_K", "qwen2.5-0.5b-instruct-q3_k_m.gguf", GGUFTensorType::Q3_K, "blk.0.attn_q.weight", 3, 10, 500},
            {"Q4_0", "qwen2.5-0.5b-instruct-q4_0.gguf", GGUFTensorType::Q4_0, "blk.0.attn_q.weight", 3, 10, 500},
            {"Q4_K", "qwen2.5-0.5b-instruct-q4_k_m.gguf", GGUFTensorType::Q4_K, "blk.0.attn_q.weight", 3, 10, 500},
            {"Q5_0", "qwen2.5-0.5b-instruct-q5_0.gguf", GGUFTensorType::Q5_0, "blk.0.attn_q.weight", 3, 10, 500},
            {"Q5_K", "qwen2.5-0.5b-instruct-q5_k_m.gguf", GGUFTensorType::Q5_K, "blk.0.attn_q.weight", 3, 10, 500},
            {"Q6_K", "qwen2.5-0.5b-instruct-q6_k.gguf", GGUFTensorType::Q6_K, "blk.0.attn_q.weight", 3, 10, 500},
            {"Q8_0", "qwen2.5-0.5b-instruct-q8_0.gguf", GGUFTensorType::Q8_0, "blk.0.attn_q.weight", 3, 10, 500},

            // IQ formats - 7B models (test fewer rows for reasonable runtime)
            {"IQ1_S", "Qwen2.5-VL-7B-Instruct-UD-IQ1_S.gguf", GGUFTensorType::IQ1_S, "blk.0.attn_q.weight", 3, 10, 200},
            {"IQ1_M", "Qwen2.5-VL-7B-Instruct-UD-IQ1_M.gguf", GGUFTensorType::IQ1_M, "blk.0.attn_q.weight", 3, 10, 200},
            {"IQ2_XXS", "Qwen2.5-VL-7B-Instruct-UD-IQ2_XXS.gguf", GGUFTensorType::IQ2_XXS, "blk.0.attn_q.weight", 3, 10, 200},
            {"IQ2_M", "Qwen2.5-VL-7B-Instruct-UD-IQ2_M.gguf", GGUFTensorType::IQ2_M, "blk.0.attn_q.weight", 3, 10, 200},
            {"IQ3_XXS", "Qwen2.5-VL-7B-Instruct-UD-IQ3_XXS.gguf", GGUFTensorType::IQ3_XXS, "blk.0.attn_q.weight", 3, 10, 200},
            {"IQ4_NL", "Qwen2.5-VL-7B-Instruct-IQ4_NL.gguf", GGUFTensorType::IQ4_NL, "blk.0.attn_q.weight", 3, 10, 200},
            {"IQ4_XS", "Qwen2.5-VL-7B-Instruct-IQ4_XS.gguf", GGUFTensorType::IQ4_XS, "blk.0.attn_q.weight", 3, 10, 200},
        };
    }

    /**
     * @brief High-resolution timer for benchmarking
     */
    class Timer
    {
    public:
        void start() { start_ = std::chrono::high_resolution_clock::now(); }

        double elapsed_ms() const
        {
            auto end = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(end - start_).count();
        }

    private:
        std::chrono::high_resolution_clock::time_point start_;
    };

    /**
     * @brief Benchmark result
     */
    struct BenchResult
    {
        std::string name;
        size_t rows;
        size_t cols;
        size_t total_elements;
        double llaminar_ms;
        double llamacpp_ms;
        double speedup;
        double llaminar_gb_per_sec;
        double llamacpp_gb_per_sec;
    };

    /**
     * @brief Call llama.cpp dequantization (must enable OMP parallelization)
     */
    void llamacppDequantizeParallel(GGUFTensorType type, const void *quantized,
                                    float *output, int64_t rows, int64_t cols)
    {
// llama.cpp functions operate on rows sequentially, so we parallelize the loop
#pragma omp parallel for schedule(static)
        for (int64_t row = 0; row < rows; ++row)
        {
            const char *row_ptr = static_cast<const char *>(quantized);
            size_t row_offset = 0;

            // Calculate offset to this row based on block size
            switch (type)
            {
            case GGUFTensorType::Q2_K:
                row_offset = row * ((cols / QK_K) * sizeof(block_q2_K));
                dequantize_row_q2_K(
                    reinterpret_cast<const block_q2_K *>(row_ptr + row_offset),
                    output + row * cols, cols);
                break;
            case GGUFTensorType::Q3_K:
                row_offset = row * ((cols / QK_K) * sizeof(block_q3_K));
                dequantize_row_q3_K(
                    reinterpret_cast<const block_q3_K *>(row_ptr + row_offset),
                    output + row * cols, cols);
                break;
            case GGUFTensorType::Q4_0:
                row_offset = row * ((cols / QK4_0) * sizeof(block_q4_0));
                dequantize_row_q4_0(
                    reinterpret_cast<const block_q4_0 *>(row_ptr + row_offset),
                    output + row * cols, cols);
                break;
            case GGUFTensorType::Q4_K:
                row_offset = row * ((cols / QK_K) * sizeof(block_q4_K));
                dequantize_row_q4_K(
                    reinterpret_cast<const block_q4_K *>(row_ptr + row_offset),
                    output + row * cols, cols);
                break;
            case GGUFTensorType::Q5_0:
                row_offset = row * ((cols / QK5_0) * sizeof(block_q5_0));
                dequantize_row_q5_0(
                    reinterpret_cast<const block_q5_0 *>(row_ptr + row_offset),
                    output + row * cols, cols);
                break;
            case GGUFTensorType::Q5_K:
                row_offset = row * ((cols / QK_K) * sizeof(block_q5_K));
                dequantize_row_q5_K(
                    reinterpret_cast<const block_q5_K *>(row_ptr + row_offset),
                    output + row * cols, cols);
                break;
            case GGUFTensorType::Q6_K:
                row_offset = row * ((cols / QK_K) * sizeof(block_q6_K));
                dequantize_row_q6_K(
                    reinterpret_cast<const block_q6_K *>(row_ptr + row_offset),
                    output + row * cols, cols);
                break;
            case GGUFTensorType::Q8_0:
                row_offset = row * ((cols / QK8_0) * sizeof(block_q8_0));
                dequantize_row_q8_0(
                    reinterpret_cast<const block_q8_0 *>(row_ptr + row_offset),
                    output + row * cols, cols);
                break;
            case GGUFTensorType::IQ1_S:
                row_offset = row * ((cols / QK_K) * sizeof(block_iq1_s));
                dequantize_row_iq1_s(
                    reinterpret_cast<const block_iq1_s *>(row_ptr + row_offset),
                    output + row * cols, cols);
                break;
            case GGUFTensorType::IQ1_M:
                row_offset = row * ((cols / QK_K) * sizeof(block_iq1_m));
                dequantize_row_iq1_m(
                    reinterpret_cast<const block_iq1_m *>(row_ptr + row_offset),
                    output + row * cols, cols);
                break;
            case GGUFTensorType::IQ2_XXS:
                row_offset = row * ((cols / QK_K) * sizeof(block_iq2_xxs));
                dequantize_row_iq2_xxs(
                    reinterpret_cast<const block_iq2_xxs *>(row_ptr + row_offset),
                    output + row * cols, cols);
                break;
            case GGUFTensorType::IQ2_M:
                row_offset = row * ((cols / QK_K) * sizeof(block_iq2_xs));
                dequantize_row_iq2_xs(
                    reinterpret_cast<const block_iq2_xs *>(row_ptr + row_offset),
                    output + row * cols, cols);
                break;
            case GGUFTensorType::IQ3_XXS:
                row_offset = row * ((cols / QK_K) * sizeof(block_iq3_xxs));
                dequantize_row_iq3_xxs(
                    reinterpret_cast<const block_iq3_xxs *>(row_ptr + row_offset),
                    output + row * cols, cols);
                break;
            case GGUFTensorType::IQ4_NL:
                row_offset = row * ((cols / QK4_NL) * sizeof(block_iq4_nl));
                dequantize_row_iq4_nl(
                    reinterpret_cast<const block_iq4_nl *>(row_ptr + row_offset),
                    output + row * cols, cols);
                break;
            case GGUFTensorType::IQ4_XS:
                row_offset = row * ((cols / QK_K) * sizeof(block_iq4_xs));
                dequantize_row_iq4_xs(
                    reinterpret_cast<const block_iq4_xs *>(row_ptr + row_offset),
                    output + row * cols, cols);
                break;
            default:
                throw std::runtime_error("Unsupported quantization type");
            }
        }
    }

    /**
     * @brief Format throughput numbers nicely
     */
    std::string formatThroughput(double gb_per_sec)
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << gb_per_sec << " GB/s";
        return oss.str();
    }

    /**
     * @brief Print benchmark results table
     */
    void printResults(const std::vector<BenchResult> &results)
    {
        std::cout << "\n╔════════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                     DEQUANTIZATION PERFORMANCE BENCHMARK                   ║\n";
        std::cout << "╠════════════╦══════════════╦═══════════════╦═══════════════╦════════════════╣\n";
        std::cout << "║   Format   ║  Tensor Size ║  Llaminar     ║  llama.cpp    ║    Speedup     ║\n";
        std::cout << "╠════════════╬══════════════╬═══════════════╬═══════════════╬════════════════╣\n";

        for (const auto &r : results)
        {
            std::ostringstream size_str;
            size_str << r.rows << "×" << r.cols;

            std::ostringstream llaminar_str;
            llaminar_str << std::fixed << std::setprecision(2) << r.llaminar_gb_per_sec << " GB/s";

            std::ostringstream llamacpp_str;
            llamacpp_str << std::fixed << std::setprecision(2) << r.llamacpp_gb_per_sec << " GB/s";

            std::ostringstream speedup_str;
            speedup_str << std::fixed << std::setprecision(2) << r.speedup << "×";

            std::string speedup_indicator = (r.speedup >= 1.0) ? "✓" : "⚠";

            printf("║ %-10s ║ %12s ║ %13s ║ %13s ║ %13s %s ║\n",
                   r.name.c_str(),
                   size_str.str().c_str(),
                   llaminar_str.str().c_str(),
                   llamacpp_str.str().c_str(),
                   speedup_str.str().c_str(),
                   speedup_indicator.c_str());
        }

        std::cout << "╚════════════╩══════════════╩═══════════════╩═══════════════╩════════════════╝\n\n";

        // Summary statistics
        double total_speedup = 0.0;
        double min_speedup = std::numeric_limits<double>::max();
        double max_speedup = 0.0;

        for (const auto &r : results)
        {
            total_speedup += r.speedup;
            min_speedup = std::min(min_speedup, r.speedup);
            max_speedup = std::max(max_speedup, r.speedup);
        }

        double avg_speedup = total_speedup / results.size();

        std::cout << "Summary:\n";
        std::cout << "  Average speedup: " << std::fixed << std::setprecision(2) << avg_speedup << "×\n";
        std::cout << "  Min speedup:     " << std::fixed << std::setprecision(2) << min_speedup << "×\n";
        std::cout << "  Max speedup:     " << std::fixed << std::setprecision(2) << max_speedup << "×\n";
        std::cout << "  OMP threads:     " << omp_get_max_threads() << "\n\n";
    }

    /**
     * @brief Performance test fixture
     */
    class DequantPerformanceTest : public ::testing::TestWithParam<BenchConfig>
    {
    protected:
        void SetUp() override
        {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);

            if (rank != 0)
            {
                GTEST_SKIP() << "Non-rank-0 process skipping performance tests";
            }
        }
    };

    TEST_P(DequantPerformanceTest, BenchmarkVsLlamaCpp)
    {
        const auto &config = GetParam();

        // Initialize IQ lookup tables if needed
        if (config.quant_type == GGUFTensorType::IQ2_M)
        {
            iq2xs_init_impl(GGML_TYPE_IQ2_M);
        }

        // Load model
        std::string model_path = "models/" + config.model_pattern;
        ModelLoader loader(model_path);

        auto tensor_opt = loader.getTensor(config.tensor_name, config.quant_type);
        ASSERT_TRUE(tensor_opt.has_value());

        auto quant_tensor = std::dynamic_pointer_cast<QuantizedTensorBase>(tensor_opt.value());
        ASSERT_NE(quant_tensor, nullptr);

        const auto &shape = quant_tensor->shape();
        ASSERT_EQ(shape.size(), 2);

        size_t rows = std::min(shape[0], config.max_rows);
        size_t cols = shape[1];

        std::cout << "\nBenchmarking " << config.name << ":\n";
        std::cout << "  Tensor: " << config.tensor_name << " (" << rows << " × " << cols << ")\n";
        std::cout << "  Warmup: " << config.warmup_iters << " iterations\n";
        std::cout << "  Benchmark: " << config.bench_iters << " iterations\n";

        // Allocate buffers
        std::vector<float> llaminar_output(rows * cols);
        std::vector<float> llamacpp_output(rows * cols);

        Timer timer;

        // === WARMUP ===
        for (size_t i = 0; i < config.warmup_iters; ++i)
        {
            quant_tensor->decode_to_fp32(llaminar_output.data(), 0, rows);
            llamacppDequantizeParallel(config.quant_type, quant_tensor->data(),
                                       llamacpp_output.data(), rows, cols);
        }

        // === LLAMINAR BENCHMARK ===
        timer.start();
        for (size_t i = 0; i < config.bench_iters; ++i)
        {
            quant_tensor->decode_to_fp32(llaminar_output.data(), 0, rows);
        }
        double llaminar_ms = timer.elapsed_ms() / config.bench_iters;

        // === LLAMA.CPP BENCHMARK ===
        timer.start();
        for (size_t i = 0; i < config.bench_iters; ++i)
        {
            llamacppDequantizeParallel(config.quant_type, quant_tensor->data(),
                                       llamacpp_output.data(), rows, cols);
        }
        double llamacpp_ms = timer.elapsed_ms() / config.bench_iters;

        // Calculate throughput
        size_t total_elements = rows * cols;
        double total_bytes = total_elements * sizeof(float); // Output size
        double llaminar_gb_per_sec = (total_bytes / 1e9) / (llaminar_ms / 1000.0);
        double llamacpp_gb_per_sec = (total_bytes / 1e9) / (llamacpp_ms / 1000.0);
        double speedup = llamacpp_ms / llaminar_ms;

        std::cout << "  Results:\n";
        std::cout << "    Llaminar: " << std::fixed << std::setprecision(3) << llaminar_ms << " ms ("
                  << formatThroughput(llaminar_gb_per_sec) << ")\n";
        std::cout << "    llama.cpp: " << std::fixed << std::setprecision(3) << llamacpp_ms << " ms ("
                  << formatThroughput(llamacpp_gb_per_sec) << ")\n";
        std::cout << "    Speedup: " << std::fixed << std::setprecision(2) << speedup << "×\n";

        // Store result for summary table
        static std::vector<BenchResult> all_results;
        all_results.push_back({config.name, rows, cols, total_elements,
                               llaminar_ms, llamacpp_ms, speedup,
                               llaminar_gb_per_sec, llamacpp_gb_per_sec});

        // Print summary table after last test
        static size_t test_count = 0;
        test_count++;
        if (test_count == getBenchConfigs().size())
        {
            printResults(all_results);
        }
    }

    INSTANTIATE_TEST_SUITE_P(
        AllQuantTypes,
        DequantPerformanceTest,
        ::testing::ValuesIn(getBenchConfigs()),
        [](const ::testing::TestParamInfo<BenchConfig> &info)
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

    if (rank != 0)
    {
        ::testing::TestEventListeners &listeners =
            ::testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());
    }

    // Print OMP configuration
    if (rank == 0)
    {
        std::cout << "Performance Benchmark Configuration:\n";
        std::cout << "  OMP threads: " << omp_get_max_threads() << "\n";
        std::cout << "  OMP places: " << (getenv("OMP_PLACES") ? getenv("OMP_PLACES") : "default") << "\n";
        std::cout << "  OMP proc bind: " << (getenv("OMP_PROC_BIND") ? getenv("OMP_PROC_BIND") : "default") << "\n\n";
    }

    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
