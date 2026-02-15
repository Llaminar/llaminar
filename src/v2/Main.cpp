/**
 * @file Main.cpp
 * @brief Llaminar v2 entry point
 *
 * Clean implementation using OrchestrationRunner for:
 * - Device manager initialization
 * - Multi-GPU heterogeneous support
 * - Pipeline and tensor parallelism
 * - Direct kernel orchestration
 * - Self-bootstrap MPI support (auto-launches mpirun if not in MPI context)
 *
 * @author David Sanftenberg
 */

#include "utils/Logger.h"
#include "utils/MPIContext.h"
#include "utils/MPIBootstrap.h"
#include "utils/NUMATopology.h"
#include "utils/Sampler.h"
#include "utils/ChatUI.h"
#include "utils/ChatTemplate.h"
#include "utils/BenchmarkRunner.h"
#include "backends/ComputeBackend.h"
#include "config/OrchestrationConfigParser.h"
#include "execution/runner/IOrchestrationRunnerFactory.h"
#include "execution/local_execution/orchestrators/IInferenceRunner.h"
#include <mpi.h>
#include <iostream>
#include <climits>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <sstream>

using namespace llaminar2;

void list_devices()
{
    auto &dm = DeviceManager::instance();
    // List devices without NUMA filtering (show all available devices)
    dm.initialize(-1);

    const auto &devices = dm.devices();

    LOG_DEBUG("\n=== Available Devices ===\n\n");
    for (size_t i = 0; i < devices.size(); ++i)
    {
        const auto &dev = devices[i];
        LOG_DEBUG("Device " << i << ": ");

        switch (dev.type)
        {
        case ComputeBackendType::CPU:
            LOG_DEBUG("CPU");
            break;
        case ComputeBackendType::GPU_CUDA:
            LOG_DEBUG("GPU (CUDA) - " << dev.name);
            break;
        case ComputeBackendType::GPU_ROCM:
            LOG_DEBUG("GPU (ROCm) - " << dev.name);
            break;
        case ComputeBackendType::GPU_VULKAN:
            LOG_DEBUG("GPU (Vulkan) - " << dev.name);
            break;
        case ComputeBackendType::GPU_METAL:
            LOG_DEBUG("GPU (Metal) - " << dev.name);
            break;
        }

        if (dev.total_memory_bytes > 0)
        {
            double total_gb = dev.total_memory_bytes / (1024.0 * 1024.0 * 1024.0);
            double free_gb = dev.free_memory_bytes / (1024.0 * 1024.0 * 1024.0);
            LOG_DEBUG(" (" << total_gb << " GB total, " << free_gb << " GB free)");
        }

        LOG_DEBUG("\n");
    }

    LOG_DEBUG("\n");
}

/**
 * @brief Main entry point with MPI self-bootstrap support
 *
 * Execution flow:
 * 1. Parse arguments (lightweight, before MPI)
 * 2. Detect if running under MPI environment
 * 3. If not under MPI: self-launch via mpirun (replaces process)
 * 4. If under MPI: initialize MPI and proceed with inference
 */
int main(int argc, char *argv[])
{
    // Initialize logging from environment (LLAMINAR_LOG_LEVEL)
    // This happens before MPI so we can log bootstrap info
    initializeLogging();

    // Parse command-line arguments using OrchestrationConfigParser
    OrchestrationConfigParser parser;
    OrchestrationConfig config = parser.parseArgs(argc, argv);

    // Handle help early (no MPI needed)
    if (config.show_help)
    {
        std::cout << OrchestrationConfigParser::getHelpText() << std::endl;
        return 0;
    }

    // Handle list-devices early (no MPI needed)
    if (config.list_devices)
    {
        list_devices();
        return 0;
    }

    // ========================================================================
    // MPI Bootstrap Detection and Self-Launch
    // ========================================================================

    // Detect CPU topology (needed for both bootstrap and runtime config)
    CPUTopology cpu_topology = MPIBootstrap::detectCPUTopology();

    // Detect if we're already running under MPI
    MPIEnvironmentInfo mpi_env = MPIBootstrap::detectMPIEnvironment();

    // If NOT running under MPI and bootstrap is not disabled, self-launch via mpirun
    if (!mpi_env.is_mpi_process && !config.mpi_no_bootstrap)
    {
        // Build launch configuration from config and topology
        MPILaunchConfig launch_config = MPIBootstrap::getDefaultConfig(cpu_topology);

        // Override with user-specified values
        if (config.mpi_procs > 0)
        {
            launch_config.num_procs = config.mpi_procs;
        }
        if (!config.hostfile.empty())
        {
            launch_config.hostfile = config.hostfile;
        }
        launch_config.report_bindings = config.mpi_verbose || (config.verbose_level > 0);
        launch_config.verbose = config.mpi_verbose;
        launch_config.oversubscribe = config.mpi_oversubscribe;

        // Handle dry-run: print config and exit
        if (config.mpi_dry_run)
        {
            MPIBootstrap::printConfigurationSummary(cpu_topology, launch_config, mpi_env);
            std::cout << "Dry run requested - exiting without launching MPI.\n";
            return 0;
        }

        // Print config summary before launch
        MPIBootstrap::printConfigurationSummary(cpu_topology, launch_config, mpi_env);

        // Self-launch via mpirun (this replaces the current process)
        // If successful, this function does not return
        int result = MPIBootstrap::selfLaunchMPI(argc, argv, launch_config, cpu_topology);

        // If we get here, exec failed
        LOG_ERROR("Failed to self-launch via mpirun");
        return result;
    }

    // ========================================================================
    // MPI Runtime - We are running under mpirun
    // ========================================================================

    // OpenMPI vader's default CMA single-copy path can emit noisy
    // "Read -1, expected ..., errno = 1" warnings in containerized environments
    // where process_vm_readv is restricted. Keep user override if explicitly set.
    if (mpi_env.is_mpi_process && std::getenv("OMPI_MCA_btl_vader_single_copy_mechanism") == nullptr)
    {
        setenv("OMPI_MCA_btl_vader_single_copy_mechanism", "none", 0);
    }

    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    // Re-parse arguments (MPI_Init may modify argc/argv)
    config = parser.parseArgs(argc, argv);

    // Configure OpenMP environment for this rank
    MPILaunchConfig mpi_launch_config = MPIBootstrap::getDefaultConfig(cpu_topology);
    if (config.mpi_procs > 0)
    {
        mpi_launch_config.num_procs = config.mpi_procs;
    }
    MPIBootstrap::configureOpenMPEnvironment(cpu_topology, mpi_launch_config);

    // Detect NUMA node for MPI rank
    auto numa_info = NUMATopology::detectLocalNUMANode();
    auto mpi_ctx = MPIContextFactory::global();

    // Set logger rank for log output
    Logger::getInstance().setRank(mpi_ctx->rank());

    if (mpi_ctx->rank() == 0)
    {
        LOG_DEBUG("=== NUMA Topology Detection ===");
    }

    LOG_DEBUG("[Rank " << mpi_ctx->rank() << "] NUMA node: " << numa_info.local_numa_node
                       << " (detection: " << numa_info.detection_method << ")");

    if (!numa_info.detection_succeeded && mpi_ctx->rank() == 0)
    {
        LOG_WARN("NUMA detection failed, using fallback node 0. This may impact multi-socket performance.");
    }

    // Initialize device manager with NUMA-aware filtering
    auto &dm = DeviceManager::instance();
    dm.initialize(numa_info.local_numa_node);

    // Validate required arguments
    if (config.model_path.empty())
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_ERROR("Error: Model path required (-m)\n\n");
            std::cout << OrchestrationConfigParser::getHelpText() << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // ========================================================================
    // Create and Initialize OrchestrationRunner
    // ========================================================================

    auto factory = createOrchestrationRunnerFactory();
    auto runner = factory->createFromOrchestrationConfig(config);

    if (!runner)
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_ERROR("Error: Failed to create orchestration runner");
        }
        MPI_Finalize();
        return 1;
    }

    if (!runner->initialize())
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_ERROR("Failed to initialize: " << runner->lastError());
        }
        MPI_Finalize();
        return 1;
    }

    // Get tokenizer from runner
    auto tokenizer = runner->tokenizer();
    if (!tokenizer)
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_ERROR("Failed to get tokenizer from runner");
        }
        MPI_Finalize();
        return 1;
    }

    // Handle chat template override if specified
    if (!config.chat_template_override.empty())
    {
        // Parse template type from string
        ChatTemplateType override_type = ChatTemplateType::UNKNOWN;
        std::string tmpl_lower = config.chat_template_override;
        std::transform(tmpl_lower.begin(), tmpl_lower.end(), tmpl_lower.begin(), ::tolower);

        if (tmpl_lower == "chatml")
            override_type = ChatTemplateType::CHATML;
        else if (tmpl_lower == "llama3")
            override_type = ChatTemplateType::LLAMA3;
        else if (tmpl_lower == "llama2")
            override_type = ChatTemplateType::LLAMA2;
        else if (tmpl_lower == "mistral" || tmpl_lower == "mistral_v1")
            override_type = ChatTemplateType::MISTRAL_V1;
        else if (tmpl_lower == "mistral_v3")
            override_type = ChatTemplateType::MISTRAL_V3;
        else if (tmpl_lower == "mistral_v7")
            override_type = ChatTemplateType::MISTRAL_V7;
        else if (tmpl_lower == "phi3")
            override_type = ChatTemplateType::PHI3;
        else if (tmpl_lower == "phi4")
            override_type = ChatTemplateType::PHI4;
        else if (tmpl_lower == "gemma")
            override_type = ChatTemplateType::GEMMA;
        else if (tmpl_lower == "deepseek")
            override_type = ChatTemplateType::DEEPSEEK;
        else if (tmpl_lower == "deepseek2")
            override_type = ChatTemplateType::DEEPSEEK2;
        else if (tmpl_lower == "deepseek3")
            override_type = ChatTemplateType::DEEPSEEK3;
        else if (tmpl_lower == "zephyr")
            override_type = ChatTemplateType::ZEPHYR;
        else if (tmpl_lower == "vicuna")
            override_type = ChatTemplateType::VICUNA;
        else if (tmpl_lower == "command_r" || tmpl_lower == "command-r")
            override_type = ChatTemplateType::COMMAND_R;
        else
        {
            if (mpi_ctx->rank() == 0)
            {
                LOG_WARN("Unknown chat template '" << config.chat_template_override << "', using model's template");
            }
        }

        if (override_type != ChatTemplateType::UNKNOWN)
        {
            tokenizer->setChatTemplate(ChatTemplate::create(override_type));
            if (mpi_ctx->rank() == 0)
            {
                LOG_DEBUG("Using chat template override: " << config.chat_template_override);
            }
        }
    }

    // ========================================================================
    // Chat Mode Handling
    // ========================================================================

    // Interactive chat mode (--chat)
    if (config.chat_mode)
    {
        if (mpi_ctx->rank() == 0)
        {
            if (!tokenizer->hasChatTemplate())
            {
                LOG_ERROR("Chat mode requires a model with a chat template.");
                LOG_ERROR("Use --chat-template to specify one (e.g., --chat-template chatml)");
                MPI_Finalize();
                return 1;
            }

            LOG_INFO("Starting interactive chat mode...");

            ChatUIConfig chat_config;
            chat_config.system_prompt = config.system_prompt;
            chat_config.max_tokens = config.n_predict;
            chat_config.temperature = config.temperature;
            chat_config.top_k = config.top_k;
            chat_config.top_p = config.top_p;

            // Create an adapter to use OrchestrationRunner with ChatUI
            // ChatUI needs IInferenceRunner, so we wrap the runner calls
            class OrchestrationRunnerAdapter : public IInferenceRunner
            {
            public:
                OrchestrationRunnerAdapter(IOrchestrationRunner *orch_runner)
                    : orch_runner_(orch_runner), position_(0) {}

                bool forward(const int *tokens, int seq_len) override
                {
                    std::vector<int32_t> token_vec(tokens, tokens + seq_len);
                    bool result = orch_runner_->prefill(token_vec);
                    if (result)
                        position_ += seq_len;
                    return result;
                }

                const float *logits() const override
                {
                    return orch_runner_->lastLogits();
                }

                int vocab_size() const override
                {
                    return orch_runner_->vocabSize();
                }

                void clear_cache() override
                {
                    orch_runner_->clearCache();
                    position_ = 0;
                }

                int get_position() const override
                {
                    return position_;
                }

                ExecutionPath executionPath() const override
                {
                    return ExecutionPath::GRAPH;
                }

                const char *architecture() const override
                {
                    return "orchestrated";
                }

            private:
                IOrchestrationRunner *orch_runner_;
                int position_;
            };

            auto adapter = std::make_shared<OrchestrationRunnerAdapter>(runner.get());
            ChatUI chat_ui(tokenizer, adapter, chat_config);
            int result = chat_ui.run();

            runner->shutdown();
            MPI_Finalize();
            return result;
        }
        else
        {
            // Non-rank-0 processes wait for chat to complete
            // TODO: Implement proper multi-rank chat support
            MPI_Barrier(MPI_COMM_WORLD);
            runner->shutdown();
            MPI_Finalize();
            return 0;
        }
    }

    // Single-shot chat mode (--chat-single)
    // This mode applies the chat template to the prompt and generates a response.
    // All MPI ranks must participate in forward passes to avoid deadlocks.
    if (config.single_shot_chat)
    {
        if (!tokenizer->hasChatTemplate())
        {
            if (mpi_ctx->rank() == 0)
            {
                LOG_ERROR("Chat mode requires a model with a chat template.");
                LOG_ERROR("Use --chat-template to specify one (e.g., --chat-template chatml)");
            }
            MPI_Barrier(MPI_COMM_WORLD);
            runner->shutdown();
            MPI_Finalize();
            return 1;
        }

        if (mpi_ctx->rank() == 0)
        {
            LOG_INFO("Running single-shot chat...");
        }

        // Build conversation and encode with chat template (rank 0 only, then broadcast)
        std::vector<int32_t> token_ids;
        int token_count = 0;

        if (mpi_ctx->rank() == 0)
        {
            std::vector<ChatMessage> conversation;
            if (!config.system_prompt.empty())
            {
                conversation.push_back(ChatMessage("system", config.system_prompt));
            }
            conversation.push_back(ChatMessage("user", config.prompt));

            auto encoded = tokenizer->encodeChat(conversation, /*add_generation_prompt=*/true);
            token_ids.assign(encoded.begin(), encoded.end());
            token_count = static_cast<int>(token_ids.size());

            if (token_ids.empty())
            {
                LOG_ERROR("Failed to encode conversation with chat template");
                token_count = -1; // Signal error to other ranks
            }
            else
            {
                LOG_DEBUG("Encoded " << token_count << " tokens with chat template");
            }
        }

        // Broadcast token count first (to handle errors and allocate on other ranks)
        MPI_Bcast(&token_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (token_count <= 0)
        {
            runner->shutdown();
            MPI_Finalize();
            return 1;
        }

        // Resize token_ids on non-rank-0 processes and broadcast the tokens
        if (mpi_ctx->rank() != 0)
        {
            token_ids.resize(token_count);
        }
        MPI_Bcast(token_ids.data(), token_count, MPI_INT, 0, MPI_COMM_WORLD);

        // All ranks participate in prefill
        if (mpi_ctx->rank() == 0)
        {
            LOG_DEBUG("Running prefill (" << token_count << " tokens)...");
        }

        if (!runner->prefill(token_ids))
        {
            if (mpi_ctx->rank() == 0)
            {
                LOG_ERROR("Chat prefill failed: " << runner->lastError());
            }
            runner->shutdown();
            MPI_Finalize();
            return 1;
        }

        // Determine max tokens: -1 means unlimited (use max_seq_len as practical limit)
        int max_tokens = config.n_predict;
        if (max_tokens < 0)
        {
            max_tokens = config.max_seq_len - token_count;
        }

        if (mpi_ctx->rank() == 0)
        {
            LOG_INFO("Generating response (max " << max_tokens << " tokens)...\n");
        }

        // Decode loop - use decodeStep() which returns sampled token
        for (int i = 0; i < max_tokens; ++i)
        {
            // decodeStep() handles sampling internally based on config
            GenerationResult result = runner->decodeStep();

            if (!result.success())
            {
                if (mpi_ctx->rank() == 0)
                {
                    LOG_ERROR("Decode step failed: " << result.error);
                }
                runner->shutdown();
                MPI_Finalize();
                return 1;
            }

            if (result.tokens.empty())
            {
                // No token generated - shouldn't happen
                break;
            }

            int32_t next_token = result.tokens[0];

            // Output token text (streaming) on rank 0 - don't print stop tokens
            if (mpi_ctx->rank() == 0 && !tokenizer->is_stop_token(next_token))
            {
                std::string token_text = tokenizer->decode_token(next_token);
                std::cout << token_text << std::flush;
            }

            // Check for completion (EOS or stop token)
            if (result.is_complete || tokenizer->is_stop_token(next_token))
            {
                if (mpi_ctx->rank() == 0)
                {
                    LOG_DEBUG("Stop token encountered (" << next_token << "), stopping generation");
                }
                break;
            }
        }

        if (mpi_ctx->rank() == 0)
        {
            std::cout << std::endl;
            LOG_INFO("Chat generation complete.");
        }

        runner->shutdown();
        MPI_Finalize();
        return 0;
    }

    // ========================================================================
    // Benchmark Mode
    // ========================================================================
    if (config.benchmark_mode)
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_INFO("Running benchmark mode...");
        }

        // Create an adapter to use OrchestrationRunner with BenchmarkRunner
        class BenchmarkRunnerAdapter : public IInferenceRunner
        {
        public:
            BenchmarkRunnerAdapter(IOrchestrationRunner *orch_runner)
                : orch_runner_(orch_runner), position_(0) {}

            bool forward(const int *tokens, int seq_len) override
            {
                std::vector<int32_t> token_vec(tokens, tokens + seq_len);
                bool result = orch_runner_->prefill(token_vec);
                if (result)
                    position_ += seq_len;
                return result;
            }

            const float *logits() const override
            {
                return orch_runner_->lastLogits();
            }

            int vocab_size() const override
            {
                return orch_runner_->vocabSize();
            }

            void clear_cache() override
            {
                orch_runner_->clearCache();
                position_ = 0;
            }

            int get_position() const override
            {
                return position_;
            }

            ExecutionPath executionPath() const override
            {
                return ExecutionPath::GRAPH;
            }

            const char *architecture() const override
            {
                return "orchestrated";
            }

        private:
            IOrchestrationRunner *orch_runner_;
            int position_;
        };

        auto adapter = std::make_shared<BenchmarkRunnerAdapter>(runner.get());

        BenchmarkRunner benchmark(adapter, tokenizer, mpi_ctx);
        BenchmarkResult result = benchmark.run(config);
        benchmark.printResults(result);

        runner->shutdown();
        MPI_Finalize();
        return result.success ? 0 : 1;
    }

    // ========================================================================
    // Standard Inference Mode
    // ========================================================================

    // Tokenize prompt
    std::vector<int32_t> tokens;
    try
    {
        // Encode WITHOUT BOS token (E2E tests don't use BOS)
        auto encoded = tokenizer->encode(config.prompt, /*add_bos=*/false, /*add_eos=*/false);
        tokens.assign(encoded.begin(), encoded.end());

        if (tokens.empty())
        {
            if (mpi_ctx->rank() == 0)
            {
                LOG_ERROR("Tokenization resulted in empty token sequence");
            }
            runner->shutdown();
            MPI_Finalize();
            return 1;
        }

        if (mpi_ctx->rank() == 0)
        {
            LOG_INFO("Tokenized prompt: " << tokens.size() << " tokens");
            std::ostringstream token_ids_str;
            token_ids_str << "Token IDs: [";
            for (size_t i = 0; i < tokens.size(); ++i)
            {
                token_ids_str << tokens[i];
                if (i < tokens.size() - 1)
                    token_ids_str << ", ";
            }
            token_ids_str << "]";
            LOG_INFO(token_ids_str.str());
        }
    }
    catch (const std::exception &e)
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_ERROR("Error tokenizing prompt: " << e.what());
        }
        runner->shutdown();
        MPI_Finalize();
        return 1;
    }

    // Set up sampling parameters for logging
    SamplingParams sampling_params;
    sampling_params.temperature = config.temperature;
    sampling_params.top_k = config.top_k;
    sampling_params.top_p = config.top_p;
    sampling_params.seed = config.seed;

    if (mpi_ctx->rank() == 0)
    {
        LOG_DEBUG("Sampling parameters:");
        LOG_DEBUG("  temperature: " << sampling_params.temperature);
        LOG_DEBUG("  top_k: " << sampling_params.top_k);
        LOG_DEBUG("  top_p: " << sampling_params.top_p);
        LOG_DEBUG("  seed: " << sampling_params.seed);
    }

    // Run prefill inference
    if (mpi_ctx->rank() == 0)
    {
        LOG_INFO("Running prefill (" << tokens.size() << " tokens)...");
    }

    if (!runner->prefill(tokens))
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_ERROR("Error: Prefill forward pass failed: " << runner->lastError());
        }
        runner->shutdown();
        MPI_Finalize();
        return 1;
    }

    if (mpi_ctx->rank() == 0)
    {
        if (config.n_predict == -1)
        {
            LOG_DEBUG("Prefill complete. Generating tokens until EOS...\n");
        }
        else
        {
            LOG_DEBUG("Prefill complete. Generating " << config.n_predict << " tokens...\n");
        }
    }

    // Generate tokens autoregressively using decodeStep()
    // n_predict = -1 means unlimited (generate until EOS)
    int max_tokens = (config.n_predict == -1) ? INT_MAX : config.n_predict;
    for (int i = 0; i < max_tokens; ++i)
    {
        LOG_DEBUG("[Rank " << mpi_ctx->rank() << "] Starting decode iteration " << i);

        // decodeStep() returns the sampled token
        GenerationResult result = runner->decodeStep();

        if (!result.success())
        {
            if (mpi_ctx->rank() == 0)
            {
                LOG_ERROR("\nError: Decode step failed at token " << (i + 1) << ": " << result.error);
            }
            runner->shutdown();
            MPI_Finalize();
            return 1;
        }

        if (result.tokens.empty())
        {
            // No token generated - shouldn't happen
            LOG_DEBUG("[Rank " << mpi_ctx->rank() << "] No token generated at iteration " << i);
            break;
        }

        int32_t next_token = result.tokens[0];
        LOG_DEBUG("[Rank " << mpi_ctx->rank() << "] Generated token: " << next_token);

        // Output on rank 0 (streaming) - don't print stop tokens
        if (mpi_ctx->rank() == 0 && !tokenizer->is_stop_token(next_token))
        {
            std::string token_text = tokenizer->decode_token(next_token);
            std::cout << token_text << std::flush;
        }

        // Check for stop tokens (EOS, <|im_end|>, etc.)
        if (result.is_complete || tokenizer->is_stop_token(next_token))
        {
            if (mpi_ctx->rank() == 0 && config.verbose_level > 0)
            {
                LOG_DEBUG("\nGeneration stopped: stop token " << next_token << " encountered");
            }
            break;
        }
    }

    if (mpi_ctx->rank() == 0)
    {
        std::cout << "\n"
                  << std::endl; // Final newline
        LOG_DEBUG("Generation complete.");
    }

    if (mpi_ctx->world_size() > 1)
    {
        mpi_ctx->barrier();
    }

    runner->shutdown();

    if (mpi_ctx->world_size() > 1)
    {
        mpi_ctx->barrier();
    }

    MPI_Finalize();
    return 0;
}
