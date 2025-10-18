/**
 * @file test_batch_benchmark.cpp
 * @brief Tests for batch benchmarking infrastructure and correctness validation
 * @author David Sanftenberg
 * @date 2025-10-15
 */

#include <gtest/gtest.h>
#include "BenchmarkRunner.h"
#include "QwenPipeline.h"
#include "QwenPipelineAdapter.h"
#include "ModelLoader.h"
#include "ArgumentParser.h"
#include "MpiContext.h"
#include "chat/TokenizerInterface.h"
#include <memory>
#include <vector>
#include <cmath>

using namespace llaminar;
using namespace llaminar::benchmark;

/**
 * @brief Simple mock tokenizer for testing
 */
class MockTokenizer : public chat::TokenizerInterface
{
public:
    std::vector<int> tokenize(const std::string &text) override
    {
        // Simple mock: return token IDs based on text length
        std::vector<int> tokens;
        for (size_t i = 0; i < text.length() && i < 10; ++i)
        {
            tokens.push_back(100 + static_cast<int>(text[i] % 50));
        }
        return tokens.empty() ? std::vector<int>{1} : tokens;
    }

    std::string detokenize(const std::vector<int> &tokens) override
    {
        // Simple mock: just return a placeholder
        return std::string(tokens.size(), 'x');
    }

    int32_t getSpecialToken(const std::string &token_name) override
    {
        if (token_name == "<|endoftext|>" || token_name == "<|im_end|>")
        {
            return 2; // EOS token
        }
        return -1;
    }
    
    std::string applyTemplate(const std::vector<chat::ChatMessage> &messages, bool add_generation_prompt) override
    {
        return "mock template";
    }
    
    bool loadVocabulary(const ModelLoader &model) override
    {
        return true;
    }
    
    size_t getVocabSize() const override
    {
        return 1000;
    }
    
    bool isReady() const override
    {
        return true;
    }
    
    std::string getTokenString(int32_t token_id) override
    {
        return "token" + std::to_string(token_id);
    }
};

/**
 * @brief Fixture for batch benchmark tests
 */
class BatchBenchmarkTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Get test model path from environment or use default
        const char *model_path_env = std::getenv("LLAMINAR_TEST_MODEL");
        if (model_path_env)
        {
            model_path = model_path_env;
        }
        else
        {
            model_path = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
        }
    }

    std::string model_path;
};

/**
 * @brief Test that BatchBenchmarkMetrics can be created and printed
 */
TEST_F(BatchBenchmarkTest, MetricsCreationAndPrint)
{
    BatchBenchmarkMetrics metrics;
    
    // Fill with sample data
    metrics.batch_size = 4;
    metrics.sequences_completed = 4;
    metrics.prefill_tokens_total = 32;
    metrics.prefill_time_ms = 100.0;
    metrics.prefill_throughput = 320.0;
    metrics.prefill_latency_per_token = 3.125;
    
    metrics.decode_tokens_total = 40;
    metrics.decode_time_ms = 200.0;
    metrics.decode_throughput = 200.0;
    metrics.decode_latency_per_token = 5.0;
    
    metrics.total_time_ms = 300.0;
    metrics.total_throughput = 240.0;
    
    metrics.memory_bandwidth_gbps = 5.5;
    metrics.efficiency_percent = 5.5;
    
    metrics.model_path = "test_model.gguf";
    metrics.backend = "OpenBLAS";
    
    // Should not crash
    ASSERT_NO_THROW(metrics.print());
    
    // Verify calculations are consistent
    EXPECT_EQ(metrics.prefill_tokens_total + metrics.decode_tokens_total, 72);
    EXPECT_NEAR(metrics.total_time_ms, metrics.prefill_time_ms + metrics.decode_time_ms, 0.01);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    
    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    
    int result = RUN_ALL_TESTS();
    
    MPI_Finalize();
    return result;
}
