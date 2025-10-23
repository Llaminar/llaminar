/**
 * @file TestInferenceParity.cpp
 * @brief Test inference parity between Llaminar and PyTorch for real prompts
 * @author David Sanftenberg
 *
 * This test generates text from simple English prompts and compares the output
 * tokens with PyTorch references to ensure identical behavior at temperature 0.0.
 *
 * CRITICAL: PyTorch loads the EXACT SAME GGUF file (with dequantization) as Llaminar,
 * ensuring a valid apples-to-apples comparison at the same quantization level.
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>

#include "QwenPipeline.h"
#include "QwenPipelineAdapter.h"
#include "ModelLoader.h"
#include "chat/GgufTokenizer.h"
#include "chat/TokenizerInterface.h"
#include "chat/ResponseGenerator.h"
#include "ArgumentParser.h"
#include "Logger.h"

#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <regex>

// OpenBLAS thread control
extern "C"
{
    void openblas_set_num_threads(int num_threads);
    int openblas_get_num_threads();
}

using namespace llaminar;
using namespace llaminar::chat;

namespace fs = std::filesystem;

#include <gtest/gtest.h>
#include <mpi.h>

#include "QwenPipeline.h"
#include "QwenPipelineAdapter.h"
#include "ModelLoader.h"
#include "chat/GgufTokenizer.h"
#include "chat/TokenizerInterface.h"
#include "chat/ResponseGenerator.h"
#include "ArgumentParser.h"
#include "Logger.h"

#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <regex>

using namespace llaminar;
using namespace llaminar::chat;

namespace fs = std::filesystem;

namespace
{
    /**
     * @brief Simple JSON parser for reading PyTorch reference files
     * Only parses the specific format we generate
     */
    class SimpleJsonParser
    {
    public:
        static std::vector<int32_t> parseIntArray(const std::string &json_content, const std::string &key)
        {
            std::vector<int32_t> result;

            // Find the key
            std::string search_key = "\"" + key + "\"";
            size_t key_pos = json_content.find(search_key);
            if (key_pos == std::string::npos)
            {
                return result;
            }

            // Find the opening bracket
            size_t bracket_start = json_content.find('[', key_pos);
            if (bracket_start == std::string::npos)
            {
                return result;
            }

            // Find the closing bracket
            size_t bracket_end = json_content.find(']', bracket_start);
            if (bracket_end == std::string::npos)
            {
                return result;
            }

            // Extract the array content
            std::string array_content = json_content.substr(bracket_start + 1, bracket_end - bracket_start - 1);

            // Parse comma-separated integers
            std::istringstream iss(array_content);
            std::string token;
            while (std::getline(iss, token, ','))
            {
                // Trim whitespace
                token.erase(0, token.find_first_not_of(" \t\n\r"));
                token.erase(token.find_last_not_of(" \t\n\r") + 1);

                if (!token.empty())
                {
                    try
                    {
                        result.push_back(std::stoi(token));
                    }
                    catch (...)
                    {
                        // Skip invalid numbers
                    }
                }
            }

            return result;
        }

        static std::string parseString(const std::string &json_content, const std::string &key)
        {
            // Find the key
            std::string search_key = "\"" + key + "\"";
            size_t key_pos = json_content.find(search_key);
            if (key_pos == std::string::npos)
            {
                return "";
            }

            // Find the opening quote for the value
            size_t quote_start = json_content.find('\"', key_pos + search_key.length());
            if (quote_start == std::string::npos)
            {
                return "";
            }

            // Find the closing quote
            size_t quote_end = quote_start + 1;
            while (quote_end < json_content.length())
            {
                if (json_content[quote_end] == '\"' && json_content[quote_end - 1] != '\\')
                {
                    break;
                }
                quote_end++;
            }

            if (quote_end >= json_content.length())
            {
                return "";
            }

            return json_content.substr(quote_start + 1, quote_end - quote_start - 1);
        }
    };
    /**
     * @brief Find all .gguf models in the models directory
     */
    std::vector<std::string> find_all_models()
    {
        std::vector<std::string> models;
        std::string models_dir = "models";

        if (!fs::exists(models_dir) || !fs::is_directory(models_dir))
        {
            return models;
        }

        for (const auto &entry : fs::recursive_directory_iterator(models_dir))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".gguf")
            {
                std::string filename = entry.path().filename().string();
                // Only test Qwen models for now (LLaMA adapter not fully implemented)
                if (filename.find("Qwen") != std::string::npos ||
                    filename.find("qwen") != std::string::npos ||
                    filename.find("Gemini-Distill") != std::string::npos)
                {
                    models.push_back(entry.path().string());
                }
            }
        }

        std::sort(models.begin(), models.end());
        return models;
    } /**
       * @brief Generate PyTorch reference for a given prompt
       */
    bool generate_pytorch_reference(
        const std::string &model_path,
        const std::string &prompt,
        int max_tokens,
        const std::string &output_path,
        int rank)
    {
        if (rank != 0)
        {
            // Only rank 0 generates PyTorch references
            return true;
        }

        std::cout << "\n[PYTORCH_REF] Generating reference for: '" << prompt << "'" << std::endl;

        // Build Python command
        std::ostringstream cmd;
        cmd << "python3 python/reference/generate_text_reference.py"
            << " --model \"" << model_path << "\""
            << " --prompt \"" << prompt << "\""
            << " --max-tokens " << max_tokens
            << " --temperature 0.0"
            << " --output \"" << output_path << "\""
            << " 2>&1";

        std::string command = cmd.str();
        std::cout << "[PYTORCH_REF] Running: " << command << std::endl;

        int ret = system(command.c_str());

        if (ret != 0)
        {
            std::cerr << "[PYTORCH_REF] ✗ Failed to generate reference (exit code: " << ret << ")" << std::endl;
            return false;
        }

        // Check that output file exists
        if (!std::filesystem::exists(output_path))
        {
            std::cerr << "[PYTORCH_REF] ✗ Output file not created: " << output_path << std::endl;
            return false;
        }

        std::cout << "[PYTORCH_REF] ✓ Reference generated successfully" << std::endl;
        return true;
    }

    /**
     * @brief Load PyTorch reference from JSON
     */
    struct PyTorchReference
    {
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> generated_tokens;
        std::string generated_text;

        static PyTorchReference load(const std::string &path)
        {
            std::ifstream file(path);
            if (!file.is_open())
            {
                throw std::runtime_error("Failed to open reference file: " + path);
            }

            // Read entire file
            std::string json_content((std::istreambuf_iterator<char>(file)),
                                     std::istreambuf_iterator<char>());
            file.close();

            PyTorchReference ref;
            ref.prompt_tokens = SimpleJsonParser::parseIntArray(json_content, "prompt_tokens");
            ref.generated_tokens = SimpleJsonParser::parseIntArray(json_content, "generated_tokens");
            ref.generated_text = SimpleJsonParser::parseString(json_content, "generated_text");

            return ref;
        }
    };

    /**
     * @brief Result from Llaminar text generation
     */
    struct LlaminarGenerationResult
    {
        std::vector<int32_t> tokens;
        std::string text;
    };

    /**
     * @brief Generate text using Llaminar
     */
    LlaminarGenerationResult generate_llaminar_text(
        std::shared_ptr<AbstractPipeline> pipeline,
        std::shared_ptr<TokenizerInterface> tokenizer,
        const QwenModelWeights &weights,
        const std::vector<int32_t> &prompt_tokens,
        int max_new_tokens,
        const std::string &model_path,
        int rank)
    {
        if (rank == 0)
        {
            std::cout << "\n[LLAMINAR] Generating with " << prompt_tokens.size()
                      << " prompt tokens, max " << max_new_tokens << " new tokens" << std::endl;
        }

        // Prompt tokens already broadcast by caller - no need to broadcast again
        std::vector<int32_t> local_prompt_tokens = prompt_tokens;

        // Create LlaminarParams for ResponseGenerator
        LlaminarParams params;
        params.model_file = model_path;
        params.n_predict = max_new_tokens;
        params.temperature = 0.0f; // Greedy/deterministic
        params.top_k = 0;
        params.top_p = 1.0f;

        // Create response generator with pre-loaded weights
        ResponseGenerator generator(tokenizer, pipeline, params, weights);

        // Generate response - this does prefill + decode internally
        std::string response = generator.generateResponse(local_prompt_tokens);

        // We need to extract the generated tokens
        // The ResponseGenerator doesn't currently expose them, so we'll need to
        // tokenize the response back to get the tokens
        // This is a workaround - ideally ResponseGenerator would expose generated tokens

        std::vector<int32_t> generated_tokens = tokenizer->tokenize(response);

        if (rank == 0)
        {
            std::cout << "[LLAMINAR] Generated " << generated_tokens.size() << " tokens" << std::endl;
            if (!generated_tokens.empty())
            {
                std::cout << "[LLAMINAR] First few tokens: ";
                for (size_t i = 0; i < std::min(generated_tokens.size(), size_t(10)); ++i)
                {
                    std::cout << generated_tokens[i];
                    if (i < std::min(generated_tokens.size(), size_t(10)) - 1)
                        std::cout << ", ";
                }
                if (generated_tokens.size() > 10)
                    std::cout << "...";
                std::cout << std::endl;
            }
            std::cout << "[LLAMINAR] Generated text: \"" << response << "\"" << std::endl;
        }

        return {generated_tokens, response};
    }

    /**
     * @brief Test a single model for inference parity
     */
    void test_model_inference_parity(const std::string &model_path, const std::string &prompt, int max_new_tokens)
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (rank == 0)
        {
            std::cout << "\n"
                      << std::string(80, '-') << std::endl;
            std::cout << "Testing model: " << model_path << std::endl;
            std::cout << std::string(80, '-') << std::endl;
        }

        // Create output path for PyTorch reference
        std::string model_name = std::filesystem::path(model_path).stem().string();
        std::string ref_path = "/tmp/inference_parity_" + model_name + ".json";

        // Clean up old reference
        if (rank == 0 && std::filesystem::exists(ref_path))
        {
            std::filesystem::remove(ref_path);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Generate PyTorch reference (rank 0 only)
        bool pytorch_success = true;
        if (rank == 0)
        {
            pytorch_success = generate_pytorch_reference(model_path, prompt, max_new_tokens, ref_path, rank);
        }

        // Broadcast success status to all ranks - if rank 0 fails, all ranks should abort
        int success_flag = pytorch_success ? 1 : 0;
        MPI_Bcast(&success_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // All ranks check the status together
        ASSERT_EQ(success_flag, 1)
            << "Failed to generate PyTorch reference for " << model_path
            << " (possibly unsupported quantization format)";

        MPI_Barrier(MPI_COMM_WORLD);

        // Load PyTorch reference (rank 0 only, will be broadcast if needed)
        PyTorchReference pytorch_ref;
        if (rank == 0)
        {
            pytorch_ref = PyTorchReference::load(ref_path);
            std::cout << "[PYTORCH] Generated tokens: ";
            for (size_t i = 0; i < std::min(pytorch_ref.generated_tokens.size(), size_t(10)); ++i)
            {
                std::cout << pytorch_ref.generated_tokens[i];
                if (i < std::min(pytorch_ref.generated_tokens.size(), size_t(10)) - 1)
                    std::cout << ", ";
            }
            if (pytorch_ref.generated_tokens.size() > 10)
                std::cout << "...";
            std::cout << std::endl;
            std::cout << "[PYTORCH] Text: \"" << pytorch_ref.generated_text << "\"" << std::endl;
        }

        // Load Llaminar model
        ModelLoader loader;
        ASSERT_TRUE(loader.loadModel(model_path)) << "Failed to load model: " << model_path;

        TransformerLayerConfig base_config = loader.createLayerConfig();
        ModelConfig model_cfg(base_config, "qwen");

        // Create Qwen pipeline via adapter
        auto pipeline = std::make_unique<QwenPipelineAdapter>(model_cfg);

        // Load weights (returns unique_ptr<IModelWeights>)
        auto weights = pipeline->loadWeights(model_path);
        ASSERT_NE(weights, nullptr) << "Failed to load weights for " << model_path;

        // Cast to QwenModelWeights
        auto *qwen_weights = dynamic_cast<QwenModelWeights *>(weights.get());
        ASSERT_NE(qwen_weights, nullptr) << "Failed to cast weights to QwenModelWeights";

        // Create tokenizer using factory
        auto tokenizer = chat::createTokenizer(loader);
        ASSERT_NE(tokenizer, nullptr) << "Failed to create tokenizer";

        // Share ownership for ResponseGenerator
        auto tokenizer_shared = std::shared_ptr<TokenizerInterface>(tokenizer.release());
        auto pipeline_shared = std::shared_ptr<AbstractPipeline>(pipeline.release());

        // Use PyTorch's prompt tokens for exact comparison
        // CRITICAL: Broadcast prompt tokens to all ranks before Llaminar generation
        std::vector<int32_t> prompt_tokens;
        if (rank == 0)
        {
            prompt_tokens = pytorch_ref.prompt_tokens;
        }

        // Broadcast prompt token count
        int prompt_token_count = (int)prompt_tokens.size();
        MPI_Bcast(&prompt_token_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Resize on non-root ranks
        if (rank != 0)
        {
            prompt_tokens.resize(prompt_token_count);
        }

        // Broadcast actual tokens
        MPI_Bcast(prompt_tokens.data(), prompt_token_count, MPI_INT, 0, MPI_COMM_WORLD);

        // Generate with Llaminar
        auto llaminar_result = generate_llaminar_text(
            pipeline_shared,
            tokenizer_shared,
            *qwen_weights,
            prompt_tokens,
            max_new_tokens,
            model_path,
            rank);

        // Compare token sequences (rank 0 only)
        // Declare match flag outside block so we can access it later for barrier synchronization
        bool all_match = true;

        if (rank == 0)
        {
            std::cout << "\n[COMPARISON] Comparing " << pytorch_ref.generated_tokens.size()
                      << " PyTorch tokens vs " << llaminar_result.tokens.size() << " Llaminar tokens" << std::endl;

            // Compare each token
            size_t min_len = std::min(llaminar_result.tokens.size(), pytorch_ref.generated_tokens.size());

            for (size_t i = 0; i < min_len; ++i)
            {
                if (llaminar_result.tokens[i] != pytorch_ref.generated_tokens[i])
                {
                    all_match = false;
                    break; // Exit early, we'll show full comparison in summary
                }
            }

            // Check if lengths also match
            if (llaminar_result.tokens.size() != pytorch_ref.generated_tokens.size())
            {
                all_match = false;
            }

            if (all_match)
            {
                std::cout << "[COMPARISON] ✓ All tokens match!" << std::endl;
            }
            else
            {
                std::cout << "[COMPARISON] ✗ Token mismatch detected" << std::endl;

                // Print summary table for easy comparison
                std::cout << "\n"
                          << std::string(80, '=') << std::endl;
                std::cout << "COMPARISON SUMMARY" << std::endl;
                std::cout << std::string(80, '=') << std::endl;

                // Show token arrays
                std::cout << "\nPyTorch Tokens (" << pytorch_ref.generated_tokens.size() << "):" << std::endl;
                std::cout << "  [";
                for (size_t i = 0; i < pytorch_ref.generated_tokens.size(); ++i)
                {
                    if (i > 0)
                        std::cout << ", ";
                    // Highlight mismatches
                    bool mismatch = (i >= llaminar_result.tokens.size() ||
                                     pytorch_ref.generated_tokens[i] != llaminar_result.tokens[i]);
                    if (mismatch)
                        std::cout << "**";
                    std::cout << pytorch_ref.generated_tokens[i];
                    if (mismatch)
                        std::cout << "**";
                }
                std::cout << "]" << std::endl;

                std::cout << "\nLlaminar Tokens (" << llaminar_result.tokens.size() << "):" << std::endl;
                std::cout << "  [";
                for (size_t i = 0; i < llaminar_result.tokens.size(); ++i)
                {
                    if (i > 0)
                        std::cout << ", ";
                    // Highlight mismatches
                    bool mismatch = (i >= pytorch_ref.generated_tokens.size() ||
                                     llaminar_result.tokens[i] != pytorch_ref.generated_tokens[i]);
                    if (mismatch)
                        std::cout << "**";
                    std::cout << llaminar_result.tokens[i];
                    if (mismatch)
                        std::cout << "**";
                }
                std::cout << "]" << std::endl;

                // Show text outputs
                std::cout << "\nPyTorch Text:" << std::endl;
                std::cout << "  \"" << pytorch_ref.generated_text << "\"" << std::endl;

                std::cout << "\nLlaminar Text:" << std::endl;
                std::cout << "  \"" << llaminar_result.text << "\"" << std::endl;

                // Show detailed mismatch positions
                std::cout << "\nMismatch Details:" << std::endl;
                size_t mismatch_count = 0;
                for (size_t i = 0; i < std::max(pytorch_ref.generated_tokens.size(), llaminar_result.tokens.size()); ++i)
                {
                    bool is_mismatch = false;
                    std::string reason;

                    if (i >= pytorch_ref.generated_tokens.size())
                    {
                        is_mismatch = true;
                        reason = "Extra token in Llaminar";
                    }
                    else if (i >= llaminar_result.tokens.size())
                    {
                        is_mismatch = true;
                        reason = "Missing token in Llaminar";
                    }
                    else if (pytorch_ref.generated_tokens[i] != llaminar_result.tokens[i])
                    {
                        is_mismatch = true;
                        reason = "Token value differs";
                    }

                    if (is_mismatch)
                    {
                        mismatch_count++;
                        std::cout << "  Position " << i << ": ";

                        if (i < pytorch_ref.generated_tokens.size())
                        {
                            std::cout << "PyTorch=" << pytorch_ref.generated_tokens[i];
                        }
                        else
                        {
                            std::cout << "PyTorch=<none>";
                        }

                        std::cout << ", ";

                        if (i < llaminar_result.tokens.size())
                        {
                            std::cout << "Llaminar=" << llaminar_result.tokens[i];
                        }
                        else
                        {
                            std::cout << "Llaminar=<none>";
                        }

                        std::cout << " (" << reason << ")" << std::endl;
                    }
                }

                std::cout << "\nTotal mismatches: " << mismatch_count << " / "
                          << std::max(pytorch_ref.generated_tokens.size(), llaminar_result.tokens.size())
                          << " positions" << std::endl;
                std::cout << std::string(80, '=') << std::endl;
            }
        }

        // CRITICAL: Broadcast match result to all ranks so they exit together
        // If we only ASSERT on rank 0, rank 1 would hang at barriers
        int match_flag = all_match ? 1 : 0;
        MPI_Bcast(&match_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Broadcast token counts for validation
        int pytorch_token_count = 0;
        int llaminar_token_count = 0;
        if (rank == 0)
        {
            pytorch_token_count = (int)pytorch_ref.generated_tokens.size();
            llaminar_token_count = (int)llaminar_result.tokens.size();
        }
        MPI_Bcast(&pytorch_token_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&llaminar_token_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // All ranks assert together based on broadcast results
        ASSERT_EQ(match_flag, 1) << "Token sequence mismatch for model: " << model_path;
        ASSERT_EQ(llaminar_token_count, pytorch_token_count)
            << "Generated token count mismatch";

        // Final barrier to ensure clean exit
        MPI_Barrier(MPI_COMM_WORLD);
    }
} // anonymous namespace

/**
 * @brief Parametrized test for inference parity across all GGUF models
 *
 * This test:
 * 1. For each GGUF model in models/ directory:
 *    a. Generates text with PyTorch (temperature 0.0 = greedy/deterministic)
 *    b. Generates text with Llaminar (greedy sampling)
 *    c. Compares generated token sequences - they should match exactly
 *
 * CRITICAL: PyTorch loads the EXACT SAME GGUF file (with dequantization) as Llaminar,
 * ensuring a valid apples-to-apples comparison at the same quantization level.
 */
class InferenceParityTest : public ::testing::TestWithParam<std::string>
{
protected:
    static void SetUpTestSuite()
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (rank == 0)
        {
            std::cout << "\n"
                      << std::string(80, '=') << std::endl;
            std::cout << "INFERENCE PARITY TEST SUITE" << std::endl;
            std::cout << std::string(80, '=') << std::endl;
            std::cout << "Testing all GGUF models in models/ directory" << std::endl;
            std::cout << "Prompt: \"The capital of France is\"" << std::endl;
            std::cout << "Max new tokens: 10" << std::endl;
            std::cout << std::string(80, '=') << std::endl;
        }
    }

    void SetUp() override
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        // Reset OpenBLAS thread state before each test to prevent thread pool corruption
        // This prevents intermittent hangs when running multiple heavy models sequentially
        int original_threads = openblas_get_num_threads();
        openblas_set_num_threads(1); // Reset to single-threaded

        // Ensure all MPI ranks are synchronized (use timeout to detect failures)
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if (mpi_initialized)
        {
            MPI_Barrier(MPI_COMM_WORLD);
        }

        // Restore multi-threading
        openblas_set_num_threads(original_threads > 0 ? original_threads : omp_get_max_threads());

        if (rank == 0)
        {
            LOG_DEBUG("Test SetUp: Reset OpenBLAS threads (now: " << openblas_get_num_threads() << ")");
        }
    }

    void TearDown() override
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        // Check if MPI is still valid before calling barrier
        int mpi_finalized = 0;
        MPI_Finalized(&mpi_finalized);

        if (!mpi_finalized)
        {
            // Use a non-blocking approach: signal completion and wait briefly
            // This prevents deadlock if one rank exits early due to assertion failure
            try
            {
                // Synchronize all ranks before tearing down
                MPI_Barrier(MPI_COMM_WORLD);

                // Reset OpenBLAS to single-threaded state to clean up thread pool
                openblas_set_num_threads(1);

                // Final barrier to ensure cleanup is synchronized
                MPI_Barrier(MPI_COMM_WORLD);

                if (rank == 0)
                {
                    LOG_DEBUG("Test TearDown: OpenBLAS thread cleanup complete");
                }
            }
            catch (...)
            {
                // If barrier fails, at least clean up locally
                openblas_set_num_threads(1);
                if (rank == 0)
                {
                    LOG_WARN("Test TearDown: MPI barrier failed, cleaned up locally only");
                }
            }
        }
        else
        {
            // MPI already finalized, just do local cleanup
            openblas_set_num_threads(1);
        }
    }

    static void TearDownTestSuite()
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (rank == 0)
        {
            std::cout << "\n"
                      << std::string(80, '=') << std::endl;
            std::cout << "INFERENCE PARITY TEST SUITE COMPLETE" << std::endl;
            std::cout << std::string(80, '=') << std::endl;
        }
    }
};

TEST_P(InferenceParityTest, SimpleEnglishPrompt)
{
    const std::string model_path = GetParam();
    const std::string prompt = "The capital of France is";
    const int max_new_tokens = 10;

    test_model_inference_parity(model_path, prompt, max_new_tokens);
}

// Instantiate the test suite with all GGUF models found in models/ directory
INSTANTIATE_TEST_SUITE_P(
    AllModels,
    InferenceParityTest,
    ::testing::ValuesIn([]()
                        {
        auto models = find_all_models();
        if (models.empty())
        {
            // Return dummy model to avoid instantiation errors
            // The test will skip if no models found
            return std::vector<std::string>{""};
        }
        return models; }()),
    [](const ::testing::TestParamInfo<std::string> &info)
    {
        // Generate test name from model filename
        std::string name = std::filesystem::path(info.param).stem().string();
        // Replace special characters with underscores for valid test names
        std::replace_if(name.begin(), name.end(), [](char c)
                        { return !std::isalnum(c); }, '_');
        return name;
    });

// Entry point
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
