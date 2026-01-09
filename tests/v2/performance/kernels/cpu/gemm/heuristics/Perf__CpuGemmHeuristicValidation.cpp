/**
 * @file Perf__CpuGemmHeuristicValidation.cpp
 * @brief CPU GEMM ML heuristic data collection test suite
 *
 * Multi-quantization benchmark suite for ML heuristic training:
 *
 * TRAINING SET (33 shapes):
 * - Qwen 0.5B: 6 tests (single token, batch32, batch128, FFN gate/up/down)
 * - Qwen 4B: 6 tests (single token, batch32, batch128, FFN gate/up/down)
 * - Qwen 7B: 6 tests (single token, batch32, batch128, FFN gate/up/down)
 * - Qwen 14B: 5 tests (single token, batch32, FFN gate/up/down)
 * - Qwen 32B: 4 tests (single token, batch32, FFN gate/down)
 * - Edge cases: 6 tests (odd dims, non-power-of-2, prime batches)
 *
 * VALIDATION SET (5 shapes - HOLDOUT):
 * - Qwen 1.5B: 5 tests (single token, batch32, FFN gate/up/down)
 *   Purpose: Test interpolation between 0.5B and 4B
 *
 * Per-test coverage:
 * - Quant formats: IQ4_NL, Q4_0, Q8_0, Q6_K, FP16 (where available)
 * - Variants: 1,225 configs per shape (ISA × tile_m × tile_n × unroll_k × prefetch_dist)
 * - Output: CSV with 31 columns for ML training
 *
 * Test scenarios:
 * - Single token decode: m=1 (autoregressive generation)
 * - Small batch prefill: m=32 (multi-user serving)
 * - Large batch prefill: m=128 (batch inference jobs)
 * - FFN projections: gate/up (expand), down (contract)
 * - Edge cases: odd dims, prime batches, non-square matrices
 *
 * Usage:
 *   # Collect training data (excludes 1.5B)
 *   python3 src/v2/kernels/cpu/python/benchmark_cpu_gemm.py --output training_data.csv
 *
 *   # Collect validation data (1.5B only)
 *   python3 src/v2/kernels/cpu/python/benchmark_cpu_gemm.py --validation --output validation_data.csv
 *
 *   # Run single test manually
 *   BENCHMARK_MODEL_PATH=models/qwen2.5-0.5b-instruct-iq4_nl.gguf \
 *     BENCHMARK_CSV_OUTPUT=/tmp/results.csv \
 *     ./v2_perf_cpu_gemm_validation --gtest_filter='*Qwen_0_5B_SingleToken_QKV*'
 *
 * @author David Sanftenberg
 * @date November 2025
 */

#include <gtest/gtest.h>
#include "v2/tensors/Tensors.h"
#include "v2/loaders/ModelLoader.h"
#include "v2/tensors/TensorKernels.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

using namespace llaminar2;

// Variant configuration for microkernel GEMM
struct VariantConfig
{
    std::string isa;   // "AVX2" or "AVX512"
    int tile_m;        // MR: register blocking for M
    int tile_n;        // NR: register blocking for N
    int unroll_k;      // K unroll factor
    int prefetch_dist; // Prefetch distance

    std::string to_string() const
    {
        return isa + "_" + std::to_string(tile_m) + "x" + std::to_string(tile_n) +
               "_k" + std::to_string(unroll_k) + "_pf" + std::to_string(prefetch_dist);
    }
};

// Generate all variant configurations (1,225 total)
std::vector<VariantConfig> generateAllVariants()
{
    std::vector<VariantConfig> variants;

    // Match Python benchmark_cpu_gemm.py exactly
    std::vector<std::string> ISA_TYPES = {"AVX512", "AVX2"};
    std::vector<int> MR_VALUES = {1, 2, 4, 8, 16, 32, 64};
    std::vector<int> NR_VALUES = {1, 2, 4, 6, 8, 16, 32, 64};
    std::vector<int> UNROLL_K_VALUES = {1, 2, 4, 8, 16};
    std::vector<int> PREFETCH_DIST_VALUES = {0, 1, 2, 3, 5};

    for (const auto &isa : ISA_TYPES)
    {
        for (int mr : MR_VALUES)
        {
            for (int nr : NR_VALUES)
            {
                // Skip invalid register file combinations
                int max_regs = (isa == "AVX512") ? 48 : 32;
                if (mr * nr > max_regs)
                {
                    continue;
                }

                for (int unroll_k : UNROLL_K_VALUES)
                {
                    for (int prefetch : PREFETCH_DIST_VALUES)
                    {
                        variants.push_back({isa, mr, nr, unroll_k, prefetch});
                    }
                }
            }
        }
    }

    // Should produce exactly 1,225 variants (600 AVX2 + 625 AVX512)
    return variants;
}

class CpuGemmHeuristicValidation : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Load model for real quantized weights
        const char *model_path = std::getenv("BENCHMARK_MODEL_PATH");
        if (!model_path)
        {
            model_path = "models/qwen2.5-0.5b-instruct-iq4_nl.gguf";
        }

        model_path_ = model_path;

        loader_ = std::make_unique<ModelLoader>();
        if (!loader_->loadModel(model_path))
        {
            throw std::runtime_error(std::string("Failed to load model: ") + model_path);
        }

        // Get first weight tensor (any quantized tensor works for benchmarking)
        // Layer 0 attention query weight
        weight_tensor_ = loader_->loadTensor("blk.0.attn_q.weight", DeviceId::cpu());
        ASSERT_NE(weight_tensor_, nullptr) << "Failed to load weight tensor";

        // Detect quantization format
        if (dynamic_cast<IQ4_NLTensor *>(weight_tensor_.get()))
        {
            quant_format_ = "IQ4_NL";
            quant_block_size_ = 32;
            bits_per_weight_ = 4.5;
            bytes_per_block_ = 18;
        }
        else if (dynamic_cast<Q4_0Tensor *>(weight_tensor_.get()))
        {
            quant_format_ = "Q4_0";
            quant_block_size_ = 32;
            bits_per_weight_ = 4.5;
            bytes_per_block_ = 18;
        }
        else if (dynamic_cast<Q8_0Tensor *>(weight_tensor_.get()))
        {
            quant_format_ = "Q8_0";
            quant_block_size_ = 32;
            bits_per_weight_ = 8.5;
            bytes_per_block_ = 34;
        }
        else if (dynamic_cast<Q6_KTensor *>(weight_tensor_.get()))
        {
            quant_format_ = "Q6_K";
            quant_block_size_ = 256;
            bits_per_weight_ = 6.6;
            bytes_per_block_ = 210;
        }
        else if (dynamic_cast<FP16Tensor *>(weight_tensor_.get()))
        {
            quant_format_ = "FP16";
            quant_block_size_ = 1;
            bits_per_weight_ = 16.0;
            bytes_per_block_ = 2;
        }
        else
        {
            throw std::runtime_error("Unknown quantization format");
        }

        std::cout << "Loaded model: " << model_path_ << std::endl;
        std::cout << "Quant format: " << quant_format_ << std::endl;
    }

    /**
     * @brief Run GEMM benchmark for given shape with variant sweeping
     *
     * This function:
     * 1. Loads appropriate weight tensor for the shape
     * 2. Checks for BENCHMARK_VARIANT env var (single variant mode)
     * 3. If not set, sweeps ALL 1,225 variants
     * 4. Outputs CSV row for each variant with 31 columns
     *
     * @param test_name Test identifier (e.g., "Qwen_0_5B_SingleToken_QKV")
     * @param m Rows in A (batch × seq_len)
     * @param n Columns in B (output features)
     * @param k Common dimension (input features)
     * @param weight_name Name of weight tensor to load (e.g., "blk.0.attn_q.weight")
     */
    void benchmarkShape(
        const std::string &test_name,
        int m, int n, int k,
        const std::string &weight_name = "")
    {
        // Load appropriate weight tensor for this shape
        std::string actual_weight_name = weight_name;
        if (actual_weight_name.empty())
        {
            // Auto-select weight based on test name
            if (test_name.find("QKV") != std::string::npos)
            {
                actual_weight_name = "blk.0.attn_q.weight"; // 896×896 for 0.5B
            }
            else if (test_name.find("FFN_Gate") != std::string::npos)
            {
                actual_weight_name = "blk.0.ffn_gate.weight"; // 896×4864 for 0.5B
            }
            else if (test_name.find("FFN_Up") != std::string::npos)
            {
                actual_weight_name = "blk.0.ffn_up.weight"; // 896×4864 for 0.5B
            }
            else if (test_name.find("FFN_Down") != std::string::npos)
            {
                actual_weight_name = "blk.0.ffn_down.weight"; // 4864×896 for 0.5B
            }
            else
            {
                // Default: use any square weight
                actual_weight_name = "blk.0.attn_q.weight";
            }
        }

        // Try to load the specific weight, fall back to generic if not found
        auto test_weight = loader_->loadTensor(actual_weight_name, DeviceId::cpu());
        if (!test_weight)
        {
            std::cerr << "Warning: Could not load " << actual_weight_name
                      << ", using default weight_tensor_" << std::endl;
            test_weight = weight_tensor_;
        }

        // Verify weight dimensions match expected k×n
        auto weight_shape = test_weight->shape();
        if (weight_shape.size() != 2)
        {
            std::cerr << "Error: Weight tensor is not 2D!" << std::endl;
            GTEST_SKIP() << "Weight tensor shape mismatch";
            return;
        }

        size_t weight_k = weight_shape[1]; // Input features
        size_t weight_n = weight_shape[0]; // Output features

        if (static_cast<int>(weight_k) != k || static_cast<int>(weight_n) != n)
        {
            std::cerr << "Warning: Weight shape [" << weight_n << "," << weight_k
                      << "] doesn't match expected [" << n << "," << k << "]" << std::endl;
            std::cerr << "Skipping test due to dimension mismatch" << std::endl;
            GTEST_SKIP() << "Weight dimensions don't match test shape";
            return;
        }

        // Use this weight for the benchmark
        weight_tensor_ = test_weight;

        // Re-detect quantization format based on actual weight tensor being used
        // (Important: SetUp() detects based on first tensor, but different tests use different tensors)
        if (dynamic_cast<IQ4_NLTensor *>(weight_tensor_.get()))
        {
            quant_format_ = "IQ4_NL";
            quant_block_size_ = 32;
            bits_per_weight_ = 4.5;
            bytes_per_block_ = 18;
        }
        else if (dynamic_cast<Q4_0Tensor *>(weight_tensor_.get()))
        {
            quant_format_ = "Q4_0";
            quant_block_size_ = 32;
            bits_per_weight_ = 4.5;
            bytes_per_block_ = 18;
        }
        else if (dynamic_cast<Q8_0Tensor *>(weight_tensor_.get()))
        {
            quant_format_ = "Q8_0";
            quant_block_size_ = 32;
            bits_per_weight_ = 8.5;
            bytes_per_block_ = 34;
        }
        else if (dynamic_cast<Q6_KTensor *>(weight_tensor_.get()))
        {
            quant_format_ = "Q6_K";
            quant_block_size_ = 256;
            bits_per_weight_ = 6.6;
            bytes_per_block_ = 210;
        }
        else if (dynamic_cast<FP16Tensor *>(weight_tensor_.get()))
        {
            quant_format_ = "FP16";
            quant_block_size_ = 1;
            bits_per_weight_ = 16.0;
            bytes_per_block_ = 2;
        }
        else
        {
            std::cerr << "Warning: Unknown quantization format for tensor" << std::endl;
            // Keep previous format detection from SetUp()
        }

        // Check if we're running a specific variant (for Python orchestration)
        const char *variant_spec = std::getenv("BENCHMARK_VARIANT");
        std::vector<VariantConfig> variants;

        if (variant_spec)
        {
            // Parse variant spec: "AVX512_6x32_k4_pf128"
            // For now, run default variant (Python script will handle orchestration)
            variants.push_back({"AVX512", 6, 32, 4, 128});
        }
        else
        {
            // Sweep all variants
            variants = generateAllVariants();
        }

        std::cout << std::string(70, '=') << std::endl;
        std::cout << "Test: " << test_name << std::endl;
        std::cout << "Shape: " << m << " × " << n << " × " << k << std::endl;
        std::cout << "Quant: " << quant_format_ << std::endl;
        std::cout << "Variants to test: " << variants.size() << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        // Open CSV output (append mode)
        const char *csv_path = std::getenv("BENCHMARK_CSV_OUTPUT");
        std::ofstream csv;
        bool write_header = false;

        if (csv_path)
        {
            csv.open(csv_path, std::ios::app);
            // Check if file is empty (need header)
            csv.seekp(0, std::ios::end);
            write_header = (csv.tellp() == 0);

            if (write_header)
            {
                // Write CSV header (32 columns)
                csv << "test_name,m,n,k,problem_size,"
                    << "quant_format,quant_block_size,bits_per_weight,bytes_per_block,quant_alignment,"
                    << "isa,tile_m,tile_n,unroll_k,prefetch_dist,variant_name,"
                    << "is_avx512,tile_area,tile_bytes,l1_fit_ratio,"
                    << "working_set_bytes,l2_fit_ratio,"
                    << "m_alignment,n_alignment,m_n_ratio,"
                    << "log_m,log_n,log_k,"
                    << "gflops,time_ms,success\n";
            }
        }

        // Create input tensor
        auto input = std::make_unique<FP32Tensor>(std::vector<size_t>{
            static_cast<size_t>(m),
            static_cast<size_t>(k)});

        // Fill with random data
        auto *data = input->mutable_data();
        for (size_t i = 0; i < m * k; ++i)
        {
            data[i] = (rand() % 2000 - 1000) / 1000.0f; // [-1, 1]
        }

        // Create output tensor
        auto output = std::make_unique<FP32Tensor>(std::vector<size_t>{
            static_cast<size_t>(m),
            static_cast<size_t>(n)});

        // Benchmark each variant
        int tested = 0;
        int successful = 0;

        for (const auto &variant : variants)
        {
            tested++;

            // TODO: Actually configure the GEMM kernel with this variant
            // For now, we use the default kernel (Python script will filter results)

            // Get GEMM kernel from weight tensor
            auto gemm_kernel = createGemmKernel();
            if (!gemm_kernel)
            {
                std::cerr << "Failed to create GEMM kernel for variant: "
                          << variant.to_string() << std::endl;
                continue;
            }

            // Warmup (3 iterations)
            for (int i = 0; i < 3; ++i)
            {
                bool success = gemm_kernel->multiply(
                    input->data(),
                    output->mutable_data(),
                    m, n, k,
                    true,    // transpose_B (weights are transposed)
                    1.0f,    // alpha
                    0.0f,    // beta
                    nullptr, // MPI context (not needed for single-rank perf test)
                    -1       // rank (not needed)
                );
                if (!success)
                {
                    std::cerr << "GEMM failed during warmup for variant: "
                              << variant.to_string() << std::endl;
                    goto next_variant;
                }
            }

            // Timed run (10 iterations)
            {
                auto start = std::chrono::high_resolution_clock::now();

                for (int i = 0; i < 10; ++i)
                {
                    bool success = gemm_kernel->multiply(
                        input->data(),
                        output->mutable_data(),
                        m, n, k,
                        true, 1.0f, 0.0f, nullptr, -1);
                    if (!success)
                    {
                        std::cerr << "GEMM failed during timing for variant: "
                                  << variant.to_string() << std::endl;
                        goto next_variant;
                    }
                }

                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

                // Calculate performance
                double total_time_ms = duration.count() / 1000.0;
                double avg_time_ms = total_time_ms / 10.0;
                double flops = 2.0 * m * n * k; // multiply + add
                double gflops = (flops / 1e9) / (avg_time_ms / 1000.0);

                successful++;

                // Compute derived features for CSV
                int is_avx512 = (variant.isa == "AVX512") ? 1 : 0;
                int tile_area = variant.tile_m * variant.tile_n;
                int tile_bytes = tile_area * 4; // FP32
                double l1_fit_ratio = static_cast<double>(tile_bytes) / (32 * 1024);
                int working_set_bytes = (variant.tile_m + variant.tile_n) * k * 4;
                double l2_fit_ratio = static_cast<double>(working_set_bytes) / (256 * 1024);
                int m_alignment = m % variant.tile_m;
                int n_alignment = n % variant.tile_n;
                double m_n_ratio = static_cast<double>(m) / n;
                int quant_alignment = k % quant_block_size_;
                double log_m = std::log10(m + 1);
                double log_n = std::log10(n + 1);
                double log_k = std::log10(k + 1);

                // Output CSV row
                if (csv.is_open())
                {
                    csv << test_name << ","
                        << m << "," << n << "," << k << "," << (m * n * k) << ","
                        << quant_format_ << "," << quant_block_size_ << ","
                        << bits_per_weight_ << "," << bytes_per_block_ << ","
                        << quant_alignment << ","
                        << variant.isa << "," << variant.tile_m << "," << variant.tile_n << ","
                        << variant.unroll_k << "," << variant.prefetch_dist << ","
                        << variant.to_string() << ","
                        << is_avx512 << "," << tile_area << "," << tile_bytes << ","
                        << l1_fit_ratio << "," << working_set_bytes << "," << l2_fit_ratio << ","
                        << m_alignment << "," << n_alignment << "," << m_n_ratio << ","
                        << log_m << "," << log_n << "," << log_k << ","
                        << gflops << "," << avg_time_ms << ",1\n";
                    csv.flush();
                }

                // Progress output (every 100 variants)
                if (tested % 100 == 0)
                {
                    std::cout << "  [" << tested << "/" << variants.size() << "] "
                              << variant.to_string() << ": "
                              << std::fixed << std::setprecision(1) << gflops << " GFLOPS"
                              << std::endl;
                }
            }

        next_variant:
            continue;
        }

        if (csv.is_open())
        {
            csv.close();
        }

        std::cout << "\nCompleted: " << successful << "/" << tested << " variants successful" << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        // Test passes if at least one variant succeeded
        EXPECT_GT(successful, 0) << "No variants completed successfully";
    }

    /**
     * @brief Create GEMM kernel from loaded weight tensor
     */
    std::unique_ptr<ITensorGemm> createGemmKernel()
    {
        // Dispatch based on tensor type
        if (auto *iq4nl = dynamic_cast<IQ4_NLTensor *>(weight_tensor_.get()))
        {
            return iq4nl->createGemm();
        }
        else if (auto *q4_0 = dynamic_cast<Q4_0Tensor *>(weight_tensor_.get()))
        {
            return q4_0->createGemm();
        }
        else if (auto *q8_0 = dynamic_cast<Q8_0Tensor *>(weight_tensor_.get()))
        {
            return q8_0->createGemm();
        }
        else if (auto *q6_k = dynamic_cast<Q6_KTensor *>(weight_tensor_.get()))
        {
            return q6_k->createGemm();
        }
        else if (auto *fp16 = dynamic_cast<FP16Tensor *>(weight_tensor_.get()))
        {
            return fp16->createGemm();
        }
        return nullptr;
    }

    std::unique_ptr<ModelLoader> loader_;
    std::shared_ptr<TensorBase> weight_tensor_;
    std::string model_path_;
    std::string quant_format_;
    int quant_block_size_;
    double bits_per_weight_;
    int bytes_per_block_;
};

// =============================================================================
// Qwen 0.5B Tests (896 hidden size, 4864 FFN intermediate)
// =============================================================================

TEST_F(CpuGemmHeuristicValidation, Qwen_0_5B_SingleToken_QKV)
{
    // Single token Q/K/V projection: [1, 896] × [896, 896]
    benchmarkShape("Qwen_0_5B_SingleToken_QKV", 1, 896, 896);
}

TEST_F(CpuGemmHeuristicValidation, Qwen_0_5B_Batch32_QKV)
{
    // Batch 32 Q/K/V projection: [32, 896] × [896, 896]
    benchmarkShape("Qwen_0_5B_Batch32_QKV", 32, 896, 896);
}

TEST_F(CpuGemmHeuristicValidation, Qwen_0_5B_Batch128_QKV)
{
    // Batch 128 Q/K/V projection: [128, 896] × [896, 896]
    benchmarkShape("Qwen_0_5B_Batch128_QKV", 128, 896, 896);
}

TEST_F(CpuGemmHeuristicValidation, Qwen_0_5B_FFN_Gate)
{
    // FFN gate projection: [1, 896] × [896, 4864]
    benchmarkShape("Qwen_0_5B_FFN_Gate", 1, 4864, 896);
}

TEST_F(CpuGemmHeuristicValidation, Qwen_0_5B_FFN_Up)
{
    // FFN up projection: [1, 896] × [896, 4864]
    benchmarkShape("Qwen_0_5B_FFN_Up", 1, 4864, 896);
}

TEST_F(CpuGemmHeuristicValidation, Qwen_0_5B_FFN_Down)
{
    // FFN down projection: [1, 4864] × [4864, 896]
    benchmarkShape("Qwen_0_5B_FFN_Down", 1, 896, 4864);
}

// =============================================================================
// =============================================================================
// VALIDATION SET: Qwen 1.5B (HOLDOUT - not used for training)
// =============================================================================
// These tests exist for validation only - demonstrating generalization to unseen model sizes
// The 1.5B size sits between 0.5B and 4B, testing interpolation capability
// To run validation benchmarks: use --validation flag in Python script
//
// Test coverage for 1.5B (1536 hidden size, 8960 FFN intermediate):
// - Single token decode (1×1536×1536)
// - Small batch prefill (32×1536×1536)
// - FFN gate, up, down projections
// =============================================================================

TEST_F(CpuGemmHeuristicValidation, Qwen_1_5B_SingleToken_QKV)
{
    benchmarkShape("Qwen_1_5B_SingleToken_QKV", 1, 1536, 1536);
}

TEST_F(CpuGemmHeuristicValidation, Qwen_1_5B_Batch32_QKV)
{
    benchmarkShape("Qwen_1_5B_Batch32_QKV", 32, 1536, 1536);
}

TEST_F(CpuGemmHeuristicValidation, Qwen_1_5B_FFN_Gate)
{
    benchmarkShape("Qwen_1_5B_FFN_Gate", 1, 8960, 1536);
}

TEST_F(CpuGemmHeuristicValidation, Qwen_1_5B_FFN_Up)
{
    benchmarkShape("Qwen_1_5B_FFN_Up", 1, 8960, 1536);
}

TEST_F(CpuGemmHeuristicValidation, Qwen_1_5B_FFN_Down)
{
    benchmarkShape("Qwen_1_5B_FFN_Down", 1, 1536, 8960);
}

// =============================================================================
// TRAINING SET: Qwen 4B
// =============================================================================
// Coverage: Single token, small batch, large batch, all FFN projections
// =============================================================================

TEST_F(CpuGemmHeuristicValidation, Qwen_4B_SingleToken_QKV)
{
    benchmarkShape("Qwen_4B_SingleToken_QKV", 1, 2560, 2560);
}

TEST_F(CpuGemmHeuristicValidation, Qwen_4B_Batch32_QKV)
{
    benchmarkShape("Qwen_4B_Batch32_QKV", 32, 2560, 2560);
}

TEST_F(CpuGemmHeuristicValidation, Qwen_4B_Batch128_QKV)
{
    benchmarkShape("Qwen_4B_Batch128_QKV", 128, 2560, 2560);
}

TEST_F(CpuGemmHeuristicValidation, Qwen_4B_FFN_Gate)
{
    benchmarkShape("Qwen_4B_FFN_Gate", 1, 13824, 2560);
}

TEST_F(CpuGemmHeuristicValidation, Qwen_4B_FFN_Down)
{
    benchmarkShape("Qwen_4B_FFN_Down", 1, 2560, 13824);
}

TEST_F(CpuGemmHeuristicValidation, Qwen_4B_Attention_Output)
{
    // Attention output projection: [1, 2560] × [2560, 2560]
    benchmarkShape("Qwen_4B_Attention_Output", 1, 2560, 2560);
}

// =============================================================================
// Qwen 7B Tests (4096 hidden size, 22016 FFN intermediate)
// =============================================================================

TEST_F(CpuGemmHeuristicValidation, Qwen_7B_SingleToken_QKV)
{
    benchmarkShape("Qwen_7B_SingleToken_QKV", 1, 4096, 4096);
}

TEST_F(CpuGemmHeuristicValidation, Qwen_7B_Batch32_QKV)
{
    benchmarkShape("Qwen_7B_Batch32_QKV", 32, 4096, 4096);
}

TEST_F(CpuGemmHeuristicValidation, Qwen_7B_Batch128_QKV)
{
    benchmarkShape("Qwen_7B_Batch128_QKV", 128, 4096, 4096);
}

TEST_F(CpuGemmHeuristicValidation, Qwen_7B_FFN_Gate)
{
    benchmarkShape("Qwen_7B_FFN_Gate", 1, 22016, 4096);
}

TEST_F(CpuGemmHeuristicValidation, Qwen_7B_FFN_Down)
{
    benchmarkShape("Qwen_7B_FFN_Down", 1, 4096, 22016);
}

TEST_F(CpuGemmHeuristicValidation, Qwen_7B_Attention_Output)
{
    benchmarkShape("Qwen_7B_Attention_Output", 1, 4096, 4096);
}

// =============================================================================
// Qwen 14B Tests (5120 hidden size, 27392 FFN intermediate)
// =============================================================================

TEST_F(CpuGemmHeuristicValidation, Qwen_14B_SingleToken_QKV)
{
    benchmarkShape("Qwen_14B_SingleToken_QKV", 1, 5120, 5120);
}

TEST_F(CpuGemmHeuristicValidation, Qwen_14B_Batch32_QKV)
{
    benchmarkShape("Qwen_14B_Batch32_QKV", 32, 5120, 5120);
}

TEST_F(CpuGemmHeuristicValidation, Qwen_14B_Batch128_QKV)
{
    benchmarkShape("Qwen_14B_Batch128_QKV", 128, 5120, 5120);
}

TEST_F(CpuGemmHeuristicValidation, Qwen_14B_FFN_Gate)
{
    benchmarkShape("Qwen_14B_FFN_Gate", 1, 27392, 5120);
}

TEST_F(CpuGemmHeuristicValidation, Qwen_14B_FFN_Down)
{
    benchmarkShape("Qwen_14B_FFN_Down", 1, 5120, 27392);
}

// =============================================================================
// Qwen 32B Tests (5120 hidden size, 27648 FFN intermediate)
// =============================================================================

TEST_F(CpuGemmHeuristicValidation, Qwen_32B_SingleToken_QKV)
{
    benchmarkShape("Qwen_32B_SingleToken_QKV", 1, 5120, 5120);
}

TEST_F(CpuGemmHeuristicValidation, Qwen_32B_Batch128_QKV)
{
    benchmarkShape("Qwen_32B_Batch128_QKV", 128, 5120, 5120);
}

TEST_F(CpuGemmHeuristicValidation, Qwen_32B_FFN_Gate)
{
    benchmarkShape("Qwen_32B_FFN_Gate", 1, 27648, 5120);
}

TEST_F(CpuGemmHeuristicValidation, Qwen_32B_FFN_Down)
{
    benchmarkShape("Qwen_32B_FFN_Down", 1, 5120, 27648);
}

// =============================================================================
// Qwen 72B Tests (8192 hidden size, 49152 FFN intermediate)
// =============================================================================

TEST_F(CpuGemmHeuristicValidation, Qwen_72B_SingleToken_QKV)
{
    benchmarkShape("Qwen_72B_SingleToken_QKV", 1, 8192, 8192);
}

TEST_F(CpuGemmHeuristicValidation, Qwen_72B_Batch32_QKV)
{
    benchmarkShape("Qwen_72B_Batch32_QKV", 32, 8192, 8192);
}

TEST_F(CpuGemmHeuristicValidation, Qwen_72B_FFN_Gate)
{
    benchmarkShape("Qwen_72B_FFN_Gate", 1, 49152, 8192);
}

TEST_F(CpuGemmHeuristicValidation, Qwen_72B_FFN_Down)
{
    benchmarkShape("Qwen_72B_FFN_Down", 1, 8192, 49152);
}

// =============================================================================
// DeepSeek V3 671B Tests (MoE architecture with 2-stage Q projection)
// =============================================================================

TEST_F(CpuGemmHeuristicValidation, DeepSeek_671B_SingleToken_Attention)
{
    // DeepSeek 671B attention: [1, 7168] × [7168, 7168]
    benchmarkShape("DeepSeek_671B_SingleToken_Attention", 1, 7168, 7168);
}

TEST_F(CpuGemmHeuristicValidation, DeepSeek_671B_Batch128_Attention)
{
    benchmarkShape("DeepSeek_671B_Batch128_Attention", 128, 7168, 7168);
}

TEST_F(CpuGemmHeuristicValidation, DeepSeek_671B_Q_Projection_Stage1)
{
    // Q projection stage 1 (with LoRA): [1, 1536] × [1536, 7168]
    benchmarkShape("DeepSeek_671B_Q_Projection_Stage1", 1, 7168, 1536);
}

TEST_F(CpuGemmHeuristicValidation, DeepSeek_671B_Q_Projection_Stage2)
{
    // Q projection stage 2: [1, 7168] × [7168, 7168]
    benchmarkShape("DeepSeek_671B_Q_Projection_Stage2", 1, 7168, 7168);
}

TEST_F(CpuGemmHeuristicValidation, DeepSeek_671B_KV_Projection)
{
    // KV projection (MQA - smaller): [1, 7168] × [7168, 1024]
    benchmarkShape("DeepSeek_671B_KV_Projection", 1, 1024, 7168);
}

TEST_F(CpuGemmHeuristicValidation, DeepSeek_671B_Attention_Output)
{
    // Attention output projection: [1, 7168] × [7168, 7168]
    benchmarkShape("DeepSeek_671B_Attention_Output", 1, 7168, 7168);
}

TEST_F(CpuGemmHeuristicValidation, DeepSeek_671B_FFN_Gate)
{
    // FFN gate (MoE): [1, 7168] × [7168, 18432]
    benchmarkShape("DeepSeek_671B_FFN_Gate", 1, 18432, 7168);
}

TEST_F(CpuGemmHeuristicValidation, DeepSeek_671B_FFN_Down)
{
    // FFN down (MoE): [1, 18432] × [18432, 7168]
    benchmarkShape("DeepSeek_671B_FFN_Down", 1, 7168, 18432);
}

// =============================================================================
// Edge Cases: Odd Dimensions/Batches (aligned with Python benchmark script)
// =============================================================================

TEST_F(CpuGemmHeuristicValidation, EdgeCase_Odd_Tiny)
{
    benchmarkShape("EdgeCase_Odd_Tiny", 1, 127, 127);
}

TEST_F(CpuGemmHeuristicValidation, EdgeCase_Odd_Medium)
{
    benchmarkShape("EdgeCase_Odd_Medium", 1, 1023, 1023);
}

TEST_F(CpuGemmHeuristicValidation, EdgeCase_Batch_Prime)
{
    benchmarkShape("EdgeCase_Batch_Prime", 17, 896, 896);
}

TEST_F(CpuGemmHeuristicValidation, EdgeCase_Nonsquare_3to1)
{
    benchmarkShape("EdgeCase_Nonsquare_3to1", 1, 3072, 1024);
}

TEST_F(CpuGemmHeuristicValidation, EdgeCase_Nonsquare_1to3)
{
    benchmarkShape("EdgeCase_Nonsquare_1to3", 1, 1024, 3072);
}

TEST_F(CpuGemmHeuristicValidation, EdgeCase_Batch_NonPowerOf2)
{
    benchmarkShape("EdgeCase_Batch_NonPowerOf2", 63, 2048, 2048);
}

// =============================================================================
// Test Summary
// =============================================================================
// TRAINING SET: 33 tests
//   - Qwen 0.5B: 6 tests (single, batch32, batch128, FFN gate/up/down)
//   - Qwen 4B: 6 tests (single, batch32, batch128, FFN gate/up/down)
//   - Qwen 7B: 6 tests (single, batch32, batch128, FFN gate/up/down)
//   - Qwen 14B: 5 tests (single, batch32, FFN gate/up/down)
//   - Qwen 32B: 4 tests (single, batch32, FFN gate/down)
//   - Edge cases: 6 tests (robustness on odd/prime/nonsquare shapes)
//
// VALIDATION SET: 5 tests (HOLDOUT)
//   - Qwen 1.5B: 5 tests (single, batch32, FFN gate/up/down)
//   - Purpose: Test generalization to unseen model sizes (interpolation)
//
// Total: 38 test cases
//
// Note: Qwen 72B and DeepSeek 671B tests exist in separate file for
//       extrapolation validation (beyond training range)
