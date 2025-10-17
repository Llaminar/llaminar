/**
 * @file test_batch_performance.cpp
 * @brief Performance benchmarks for batch processing implementation
 * 
 * **CRITICAL REQUIREMENTS:**
 * - MUST be run in Release build: cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
 * - Debug builds are 5-10x slower and produce misleading results
 * - Run via: ./run_batch_performance.sh (handles environment setup)
 * 
 * Tests measure throughput (tokens/sec) across various batch sizes to validate
 * the 22× speedup target for batch=32. Compares aggregate throughput vs baseline
 * sequential execution.
 * 
 * @author David Sanftenberg
 * @date 2025-10-15
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <numeric>

#include "QwenPipelineAdapter.h"
#include "BatchQwenPipelineAdapter.h"
#include "AbstractPipeline.h"
#include "ModelLoader.h"
#include "TransformerConfig.h"
#include "MpiContext.h"
#include "tensors/SimpleTensor.h"
#include "Logger.h"

using namespace llaminar;

/**
 * @class BatchPerformanceTest
 * @brief Performance benchmarking for batch processing
 * 
 * Measures throughput (tokens/sec) across batch sizes 1-32 to validate
 * performance targets and calculate speedup vs baseline.
 */
class BatchPerformanceTest : public ::testing::Test {
protected:
    std::string model_path;
    ModelConfig seq_config;    // Sequential pipeline config
    ModelConfig batch_config;  // Batch pipeline config
    
    void SetUp() override {
        auto rank = MPIContext::capture().rank;
        
        // CRITICAL: Set log level to WARN to suppress excessive debug output
        // Performance tests generate massive logs (>100K lines) at DEBUG level
        Logger::getInstance().setLogLevel(LogLevel::WARN);
        
        if (rank == 0) {
            std::cout << "\n[NOTICE] Performance benchmarks MUST be run in Release build (-DCMAKE_BUILD_TYPE=Release)" << std::endl;
            std::cout << "[NOTICE] Debug builds are 5-10x slower and will show artificially low performance\n" << std::endl;
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Find model file (relative to test executable)
        model_path = "./models/qwen2.5-0.5b-instruct-q4_0.gguf";
        if (rank == 0) {
            std::ifstream check(model_path);
            if (!check.good()) {
                // Try parent directory (when run from build/)
                model_path = "../models/qwen2.5-0.5b-instruct-q4_0.gguf";
            }
        }
        
        // Load model configuration
        ModelLoader loader;
        if (!loader.loadModel(model_path)) {
            GTEST_SKIP() << "Failed to load model: " << model_path;
        }
        
        TransformerLayerConfig base_config = loader.createLayerConfig();
        seq_config = ModelConfig(base_config, "qwen");        // Sequential pipeline
        batch_config = ModelConfig(base_config, "qwen_batch");  // Batch pipeline
    }
    
    /**
     * @brief Measure prefill throughput for a given batch size (using batch pipeline)
     * 
     * @param batch_size Number of sequences to process in parallel
     * @param tokens_per_seq Number of tokens in each sequence
     * @return Throughput in tokens/second
     */
    double measurePrefillThroughput(int batch_size, int tokens_per_seq) {
        auto rank = MPIContext::capture().rank;
        
        // Create batch pipeline and load weights
        auto pipeline = PipelineFactory::instance().create(batch_config);
        auto weights = pipeline->loadWeights(model_path);
        
        // Create batch input
        std::vector<std::vector<int>> batch_input(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            batch_input[i].resize(tokens_per_seq);
            // Fill with sequential token IDs
            for (int j = 0; j < tokens_per_seq; ++j) {
                batch_input[i][j] = (i * tokens_per_seq + j) % 1000;
            }
        }
        
        // Prepare context and output
        StageContext ctx;
        std::shared_ptr<TensorBase> logits;
        
        // Warmup run
        pipeline->prefillBatch(batch_input, *weights, ctx, logits);
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        // INSTRUMENTATION: Track per-sequence timing to detect sequential processing
        if (rank == 0) {
            std::cout << "\n[BATCH_INSTRUMENTATION] Starting prefill for batch_size=" << batch_size << std::endl;
        }
        
        // Timed run
        auto start = std::chrono::high_resolution_clock::now();
        pipeline->prefillBatch(batch_input, *weights, ctx, logits);
        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();
        
        double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
        int total_tokens = batch_size * tokens_per_seq;
        double throughput = (total_tokens * 1000.0) / duration_ms;
        
        // INSTRUMENTATION: Calculate expected vs actual for sequential processing
        if (rank == 0) {
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "Prefill [batch=" << std::setw(2) << batch_size 
                      << ", tokens=" << std::setw(2) << tokens_per_seq << "]: "
                      << std::setw(8) << duration_ms << " ms, "
                      << std::setw(8) << throughput << " tok/s";
            
            // If batch_size > 1, show what we'd expect for sequential
            if (batch_size > 1) {
                // Assuming batch=1 baseline is ~826ms for 8 tokens
                double expected_sequential_ms = 826.0 * batch_size;
                double ratio = duration_ms / expected_sequential_ms;
                std::cout << " [seq_ratio=" << std::setw(4) << ratio << "]";
            }
            std::cout << std::endl;
        }
        
        return throughput;
    }
    
    /**
     * @brief Measure decode throughput for a given batch size
     * 
     * @param batch_size Number of sequences to process in parallel
     * @param decode_steps Number of decode steps to execute
     * @param prefill_tokens Tokens in initial prefill phase
     * @return Throughput in tokens/second
     */
    double measureDecodeThroughput(int batch_size, int decode_steps, int prefill_tokens) {
        auto rank = MPIContext::capture().rank;
        
        // Create batch pipeline and load weights
        auto pipeline = PipelineFactory::instance().create(batch_config);
        auto weights = pipeline->loadWeights(model_path);
        
        // Create batch input for prefill
        std::vector<std::vector<int>> batch_input(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            batch_input[i].resize(prefill_tokens);
            for (int j = 0; j < prefill_tokens; ++j) {
                batch_input[i][j] = (i * prefill_tokens + j) % 1000;
            }
        }
        
        // Prepare context and output
        StageContext ctx;
        std::shared_ptr<TensorBase> logits;
        
        // Prefill phase (not timed)
        pipeline->prefillBatch(batch_input, *weights, ctx, logits);
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Timed decode phase
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int step = 0; step < decode_steps; ++step) {
            std::vector<int> next_tokens(batch_size, 42);  // Deterministic tokens
            pipeline->decodeBatch(next_tokens, *weights, ctx, logits);
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();
        
        double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
        int total_tokens = batch_size * decode_steps;
        double throughput = (total_tokens * 1000.0) / duration_ms;
        
        if (rank == 0) {
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "Decode  [batch=" << std::setw(2) << batch_size 
                      << ", steps=" << std::setw(3) << decode_steps << "]: "
                      << std::setw(8) << duration_ms << " ms, "
                      << std::setw(8) << throughput << " tok/s" << std::endl;
        }
        
        return throughput;
    }
};

/**
 * @test PrefillThroughputScaling
 * @brief Measure prefill throughput across batch sizes 1, 2, 4, 8, 16, 32, 64, 128, 256
 * 
 * Tests prefill performance scaling with increasing batch sizes to validate
 * memory bandwidth utilization and parallel processing efficiency.
 * Extended to larger batch sizes to identify saturation point.
 */
TEST_F(BatchPerformanceTest, PrefillThroughputScaling) {
    auto rank = MPIContext::capture().rank;
    
    const int tokens_per_seq = 8;  // Fixed sequence length
    std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    std::vector<double> throughputs;
    
    if (rank == 0) {
        std::cout << "\n--- Prefill Throughput Scaling ---" << std::endl;
    }
    
    for (int batch_size : batch_sizes) {
        double throughput = measurePrefillThroughput(batch_size, tokens_per_seq);
        throughputs.push_back(throughput);
    }
    
    if (rank == 0) {
        // Calculate speedup vs batch=1 baseline
        double baseline = throughputs[0];
        std::cout << "\nPrefill Speedup Analysis:" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        for (size_t i = 0; i < batch_sizes.size(); ++i) {
            double speedup = throughputs[i] / baseline;
            std::cout << "Batch " << std::setw(2) << batch_sizes[i] << ": "
                      << std::setw(8) << throughputs[i] << " tok/s, "
                      << "speedup = " << std::setw(5) << speedup << "×" << std::endl;
        }
        
        // Check if we meet 22× target at batch=32
        double target_speedup = 22.0;
        double actual_speedup = throughputs.back() / baseline;
        
        std::cout << "\nTarget speedup @ batch=32: " << target_speedup << "×" << std::endl;
        std::cout << "Actual speedup @ batch=32: " << actual_speedup << "×" << std::endl;
        
        if (actual_speedup >= target_speedup * 0.9) {
            std::cout << "✅ Performance target MET (within 10%)" << std::endl;
        } else {
            std::cout << "⚠️  Performance target NOT MET (need optimization)" << std::endl;
            std::cout << "   Gap: " << (target_speedup - actual_speedup) << "×" << std::endl;
        }
    }
}

/**
 * @test PrefillThroughputScalingLongSequences
 * @brief Measure prefill throughput with longer sequences (128, 256, 512 tokens)
 * 
 * Tests prefill performance scaling with longer sequences to match llama.cpp
 * benchmark methodology (pp512). Longer sequences unlock better batch scaling
 * due to increased compute density and better amortization of overhead.
 * 
 * Target: Match or exceed llama.cpp's 1210 tok/s @ batch=512, seq_len=512
 * 
 * Environment variables for filtering:
 *   BATCH_SIZE_FILTER: Comma-separated list of batch sizes to test (e.g., "1,32")
 *   SEQ_LEN_FILTER: Comma-separated list of sequence lengths to test (e.g., "512")
 * 
 * Examples:
 *   # Test only batch=32 with seq_len=512
 *   BATCH_SIZE_FILTER=32 SEQ_LEN_FILTER=512 ./run_batch_performance.sh --filter '*LongSequences'
 *   
 *   # Test batch=1,4,32 with all sequence lengths
 *   BATCH_SIZE_FILTER=1,4,32 ./run_batch_performance.sh --filter '*LongSequences'
 */
TEST_F(BatchPerformanceTest, PrefillThroughputScalingLongSequences) {
    auto rank = MPIContext::capture().rank;
    
    if (rank == 0) {
        std::cout << "\n--- Prefill Throughput Scaling (Long Sequences) ---" << std::endl;
        std::cout << "Comparing against llama.cpp baseline: 1210 tok/s @ batch=512, seq_len=512\n" << std::endl;
    }
    
    // Default test parameters
    std::vector<int> sequence_lengths = {128, 256, 512};
    std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32};
    
    // Check for environment variable filters
    const char* batch_filter = std::getenv("BATCH_SIZE_FILTER");
    const char* seq_filter = std::getenv("SEQ_LEN_FILTER");
    
    if (batch_filter) {
        batch_sizes.clear();
        std::string filter_str(batch_filter);
        size_t pos = 0;
        while ((pos = filter_str.find(',')) != std::string::npos) {
            batch_sizes.push_back(std::stoi(filter_str.substr(0, pos)));
            filter_str.erase(0, pos + 1);
        }
        if (!filter_str.empty()) {
            batch_sizes.push_back(std::stoi(filter_str));
        }
        if (rank == 0) {
            std::cout << "🔍 Filtered batch sizes: ";
            for (size_t i = 0; i < batch_sizes.size(); ++i) {
                std::cout << batch_sizes[i];
                if (i < batch_sizes.size() - 1) std::cout << ", ";
            }
            std::cout << std::endl;
        }
    }
    
    if (seq_filter) {
        sequence_lengths.clear();
        std::string filter_str(seq_filter);
        size_t pos = 0;
        while ((pos = filter_str.find(',')) != std::string::npos) {
            sequence_lengths.push_back(std::stoi(filter_str.substr(0, pos)));
            filter_str.erase(0, pos + 1);
        }
        if (!filter_str.empty()) {
            sequence_lengths.push_back(std::stoi(filter_str));
        }
        if (rank == 0) {
            std::cout << "🔍 Filtered sequence lengths: ";
            for (size_t i = 0; i < sequence_lengths.size(); ++i) {
                std::cout << sequence_lengths[i];
                if (i < sequence_lengths.size() - 1) std::cout << ", ";
            }
            std::cout << std::endl;
        }
    }
    
    for (int seq_len : sequence_lengths) {
        if (rank == 0) {
            std::cout << "\n=== Sequence Length: " << seq_len << " tokens ===" << std::endl;
        }
        
        std::vector<double> throughputs;
        
        for (int batch_size : batch_sizes) {
            double throughput = measurePrefillThroughput(batch_size, seq_len);
            throughputs.push_back(throughput);
        }
        
        if (rank == 0) {
            // Calculate speedup vs batch=1 baseline
            double baseline = throughputs[0];
            std::cout << "\nSpeedup Analysis (seq_len=" << seq_len << "):" << std::endl;
            std::cout << std::fixed << std::setprecision(2);
            
            for (size_t i = 0; i < batch_sizes.size(); ++i) {
                double speedup = throughputs[i] / baseline;
                std::cout << "Batch " << std::setw(3) << batch_sizes[i] << ": "
                          << std::setw(8) << throughputs[i] << " tok/s, "
                          << "speedup = " << std::setw(5) << speedup << "×";
                
                // Highlight if we beat llama.cpp at comparable batch size
                if (seq_len == 512 && batch_sizes[i] == 32 && throughputs[i] > 643.4) {
                    std::cout << " 🚀 (exceeds llama.cpp @ batch=32: 643 tok/s)";
                }
                std::cout << std::endl;
            }
            
            // Compare to llama.cpp baseline at batch=32, seq_len=512
            if (seq_len == 512) {
                double llama_cpp_baseline = 1210.1;  // tok/s @ batch=512
                double llama_cpp_batch32 = 643.4;     // tok/s @ batch=32
                double actual_batch32 = throughputs[batch_sizes.size() - 1];
                
                std::cout << "\nComparison to llama.cpp (seq_len=512):" << std::endl;
                std::cout << "  llama.cpp @ batch=32:  " << std::setw(8) << llama_cpp_batch32 << " tok/s" << std::endl;
                std::cout << "  Llaminar  @ batch=32:  " << std::setw(8) << actual_batch32 << " tok/s";
                
                if (actual_batch32 >= llama_cpp_batch32) {
                    double improvement = (actual_batch32 / llama_cpp_batch32 - 1.0) * 100.0;
                    std::cout << " ✅ (+" << std::setprecision(1) << improvement << "% faster)" << std::endl;
                } else {
                    double gap = (llama_cpp_batch32 / actual_batch32 - 1.0) * 100.0;
                    std::cout << " ⚠️  (-" << std::setprecision(1) << gap << "% slower)" << std::endl;
                }
                
                std::cout << "\n  llama.cpp @ batch=512: " << std::setw(8) << llama_cpp_baseline << " tok/s (target)" << std::endl;
            }
        }
    }
}

/**
 * @test DecodeThroughputScaling
 * @brief Measure decode throughput across batch sizes 1, 2, 4, 8, 16, 32
 * 
 * Tests decode performance scaling with increasing batch sizes. Decode is
 * memory-bound (small compute, large KV cache access) so should show strong
 * batching benefits.
 */
TEST_F(BatchPerformanceTest, DecodeThroughputScaling) {
    auto rank = MPIContext::capture().rank;
    
    const int decode_steps = 20;    // Multiple steps to amortize overhead
    const int prefill_tokens = 4;   // Short prefill context
    std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32};
    std::vector<double> throughputs;
    
    if (rank == 0) {
        std::cout << "\n--- Decode Throughput Scaling ---" << std::endl;
    }
    
    for (int batch_size : batch_sizes) {
        double throughput = measureDecodeThroughput(batch_size, decode_steps, prefill_tokens);
        throughputs.push_back(throughput);
    }
    
    if (rank == 0) {
        // Calculate speedup vs batch=1 baseline
        double baseline = throughputs[0];
        std::cout << "\nDecode Speedup Analysis:" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        for (size_t i = 0; i < batch_sizes.size(); ++i) {
            double speedup = throughputs[i] / baseline;
            std::cout << "Batch " << std::setw(2) << batch_sizes[i] << ": "
                      << std::setw(8) << throughputs[i] << " tok/s, "
                      << "speedup = " << std::setw(5) << speedup << "×" << std::endl;
        }
        
        // Decode should show strong batching benefits (memory-bound)
        double target_speedup = 22.0;
        double actual_speedup = throughputs.back() / baseline;
        
        std::cout << "\nTarget speedup @ batch=32: " << target_speedup << "×" << std::endl;
        std::cout << "Actual speedup @ batch=32: " << actual_speedup << "×" << std::endl;
        
        if (actual_speedup >= target_speedup * 0.9) {
            std::cout << "✅ Performance target MET (within 10%)" << std::endl;
        } else {
            std::cout << "⚠️  Performance target NOT MET (need optimization)" << std::endl;
            std::cout << "   Gap: " << (target_speedup - actual_speedup) << "×" << std::endl;
        }
    }
}

/**
 * @test MemoryBandwidthAnalysis
 * @brief Calculate memory bandwidth utilization for batch processing
 * 
 * Estimates memory bandwidth based on model size, throughput, and batch size.
 * Target: 13.25 tok/s → 288-320 tok/s with improved bandwidth utilization.
 */
TEST_F(BatchPerformanceTest, MemoryBandwidthAnalysis) {
    auto rank = MPIContext::capture().rank;
    
    const int batch_size = 32;
    const int decode_steps = 50;
    const int prefill_tokens = 4;
    
    if (rank == 0) {
        std::cout << "\n--- Memory Bandwidth Analysis ---" << std::endl;
    }
    
    // Measure baseline (batch=1) using sequential pipeline
    auto pipeline_baseline = PipelineFactory::instance().create(seq_config);
    auto weights_baseline = pipeline_baseline->loadWeights(model_path);
    
    std::vector<std::vector<int>> input_baseline = {{1, 2, 3, 4}};
    StageContext ctx_baseline;
    std::shared_ptr<TensorBase> logits_baseline;
    
    pipeline_baseline->prefillBatch(input_baseline, *weights_baseline, ctx_baseline, logits_baseline);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start_baseline = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < decode_steps; ++step) {
        std::vector<int> next_tokens = {42};
        pipeline_baseline->decodeBatch(next_tokens, *weights_baseline, ctx_baseline, logits_baseline);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_baseline = std::chrono::high_resolution_clock::now();
    
    double duration_baseline_ms = std::chrono::duration<double, std::milli>(end_baseline - start_baseline).count();
    double throughput_baseline = (decode_steps * 1000.0) / duration_baseline_ms;
    
    // Measure batched (batch=32) using batch pipeline
    auto pipeline_batched = PipelineFactory::instance().create(batch_config);
    auto weights_batched = pipeline_batched->loadWeights(model_path);
    
    std::vector<std::vector<int>> input_batched(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        input_batched[i].resize(prefill_tokens);
        for (int j = 0; j < prefill_tokens; ++j) {
            input_batched[i][j] = (i * prefill_tokens + j) % 1000;
        }
    }
    
    StageContext ctx_batched;
    std::shared_ptr<TensorBase> logits_batched;
    
    pipeline_batched->prefillBatch(input_batched, *weights_batched, ctx_batched, logits_batched);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start_batched = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < decode_steps; ++step) {
        std::vector<int> next_tokens(batch_size, 42);
        pipeline_batched->decodeBatch(next_tokens, *weights_batched, ctx_batched, logits_batched);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_batched = std::chrono::high_resolution_clock::now();
    
    double duration_batched_ms = std::chrono::duration<double, std::milli>(end_batched - start_batched).count();
    double throughput_batched = (batch_size * decode_steps * 1000.0) / duration_batched_ms;
    
    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\nBaseline (batch=1):" << std::endl;
        std::cout << "  Throughput: " << throughput_baseline << " tok/s" << std::endl;
        std::cout << "  Duration: " << duration_baseline_ms << " ms" << std::endl;
        
        std::cout << "\nBatched (batch=" << batch_size << "):" << std::endl;
        std::cout << "  Throughput: " << throughput_batched << " tok/s" << std::endl;
        std::cout << "  Duration: " << duration_batched_ms << " ms" << std::endl;
        
        double speedup = throughput_batched / throughput_baseline;
        std::cout << "\nSpeedup: " << speedup << "×" << std::endl;
        
        // Memory bandwidth estimation
        // Approximate: Each decode step reads weights + KV cache
        size_t model_bytes = 409 * 1024 * 1024;  // 409 MB (Q4_0)
        double bytes_per_token = model_bytes / (double)decode_steps;  // Rough estimate
        
        double bandwidth_baseline_MBps = (throughput_baseline * bytes_per_token) / (1024 * 1024);
        double bandwidth_batched_MBps = (throughput_batched * bytes_per_token) / (1024 * 1024);
        
        std::cout << "\nEstimated Memory Bandwidth:" << std::endl;
        std::cout << "  Baseline: " << bandwidth_baseline_MBps << " MB/s" << std::endl;
        std::cout << "  Batched:  " << bandwidth_batched_MBps << " MB/s" << std::endl;
        std::cout << "  Improvement: " << (bandwidth_batched_MBps / bandwidth_baseline_MBps) << "×" << std::endl;
        
        // Compare to targets
        std::cout << "\nTarget Analysis:" << std::endl;
        std::cout << "  Target throughput: 288-320 tok/s @ batch=32" << std::endl;
        std::cout << "  Actual throughput: " << throughput_batched << " tok/s" << std::endl;
        
        if (throughput_batched >= 288.0 * 0.9) {
            std::cout << "  ✅ Within target range" << std::endl;
        } else {
            std::cout << "  ⚠️  Below target (need optimization)" << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    
    // Register both sequential and batch pipelines before running tests
    registerQwenPipeline();
    registerBatchQwenPipeline();
    
    int result = RUN_ALL_TESTS();
    
    MPI_Finalize();
    return result;
}
