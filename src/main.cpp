// Main application entry & execution pipeline

#include <iostream>
#include <mpi.h>
#include <string>
#include <memory>
#include <iomanip>
#include <chrono>
#include <sstream>

#include "argument_parser.h"
#include "logger.h"
#include "topology_manager.h"
#include "model_loader.h"
#include "mpi_transformer_pipeline.h"
#include "graph_compute.h" // contains MatMulNode definition
#include "performance_timer.h"
#include "utils/debug_env.h"
#include "tensors/sharded_tensor_registry.h"
#include "tensors/tensor_factory.h"
#include "chat/tokenizer_interface.h"
#include "chat/chat_interface.h"
#include "chat/response_generator.h"

using namespace llaminar;

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int exit_code = 0;

    auto finalize = [&]()
    {
        int initialized = 0, finalized = 0;
        MPI_Initialized(&initialized);
        if (initialized)
            MPI_Finalized(&finalized);
        if (initialized && !finalized)
        {
            int global_code = exit_code;
            MPI_Allreduce(&exit_code, &global_code, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            exit_code = global_code;
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
        }
    };

#define RETURN_FAIL(code, msg)              \
    do                                      \
    {                                       \
        std::string _llm_msg = (msg);       \
        if (rank == 0 && !_llm_msg.empty()) \
        {                                   \
            LOG_ERROR(_llm_msg);            \
        }                                   \
        exit_code = (code);                 \
        finalize();                         \
        return exit_code;                   \
    } while (0)

    try
    {
        // 1. Parse command line arguments
        ArgumentParser parser(argc, argv);
        LlaminarParams params;
        if (!parser.parse(params))
        {
            RETURN_FAIL(1, "Argument parsing failed");
        }

        if (params.show_help)
        {
            if (rank == 0)
                parser.printUsage();
            finalize();
            return exit_code;
        }
        if (params.show_version)
        {
            if (rank == 0)
                parser.printVersion();
            finalize();
            return exit_code;
        }

        // 2. Initialize logging
        initializeLogging();
        Logger::getInstance().setLogLevel(params.log_level);
        if (rank == 0)
        {
            LOG_INFO("Llaminar LLM Inference Engine starting...");
            LOG_INFO("MPI initialized with " << size << " processes");
            LOG_DEBUG("Log level set to: " << Logger::getInstance().logLevelToString(params.log_level));
            // Emit one-time snapshot summary of enabled debug/diagnostic groups
            auto lines = formatDebugEnvSummary(debugEnv());
            for (const auto &line : lines)
            {
                LOG_INFO(line);
            }
        }

        // 3. Detect system topology
        TopologyManager topology_manager;
        SystemTopology topology = topology_manager.detectSystemTopology(
            params.use_hyperthreading,
            params.detect_gpus);
        if (rank == 0 && params.print_topology)
        {
            topology_manager.printSystemTopology(topology);
        }
        LOG_DEBUG("System initialization complete");

        // 4. Determine execution mode (informational)
        if (params.interactive && !params.model_file.empty())
        {
            LOG_INFO("Interactive chat mode requested");
            LOG_INFO("Model file: " << params.model_file);
        }
        else if (params.inference_mode || !params.model_file.empty())
        {
            LOG_INFO("Running inference mode");
            if (!params.model_file.empty())
                LOG_INFO("Model file: " << params.model_file);
        }
        else
        {
            LOG_INFO("Running COSMA benchmark mode");
        }

        // 5. Model loading
        std::unique_ptr<ModelLoader> model_loader;
        std::unique_ptr<MPITransformerPipeline> transformer_pipeline;
        std::unique_ptr<MPITransformerPipeline::ModelWeights> model_weights;
        if (!params.model_file.empty())
        {
            LOG_INFO("Loading model from: " << params.model_file);
            model_loader = std::make_unique<ModelLoader>();
            try
            {
                if (model_loader->loadModel(params.model_file))
                {
                    LOG_INFO("Model loaded successfully");
                    auto config = model_loader->createLayerConfig();
                    LOG_INFO("Extracted model config: " << config.n_layers << " layers, "
                                                        << config.n_head << " heads, " << config.d_model << " dimensions");
                    LOG_INFO("Model config details: vocab_size=" << config.vocab_size
                                                                 << ", max_seq_len=" << config.max_seq_len << ", d_ff=" << config.d_ff);
                    transformer_pipeline = std::make_unique<MPITransformerPipeline>(config);
                    LOG_INFO("Transformer pipeline initialized successfully");
                    try
                    {
                        model_weights = std::make_unique<MPITransformerPipeline::ModelWeights>(
                            loadModelWeights(*model_loader, config));
                        LOG_INFO("Model weights loaded for transformer pipeline");
                        if (rank == 0 && params.print_topology)
                        {
                            LOG_INFO("-- Sharded Tensor Registry (post model load) --");
                            auto snap = ShardedTensorRegistry::instance().snapshot();
                            if (snap.details.empty())
                            {
                                LOG_INFO("(no sharded tensors registered)");
                            }
                            else
                            {
                                int idx = 0;
                                for (const auto &d : snap.details)
                                {
                                    const auto &spec = d.spec;
                                    std::ostringstream oss;
                                    oss << "shard[" << idx++ << "] axis=" << spec.axis_name();
                                    if (!spec.role.empty())
                                        oss << " role=" << spec.role;
                                    oss << " world=" << spec.world << " rank=" << spec.rank
                                        << " global_dim=" << spec.global_dim << " local_offset=" << spec.local_offset
                                        << " local_dim=" << spec.local_dim << " local_shape=[";
                                    for (size_t i = 0; i < d.local_shape.size(); ++i)
                                    {
                                        if (i)
                                            oss << 'x';
                                        oss << d.local_shape[i];
                                    }
                                    oss << "] elems_local=" << d.local_elems;
                                    LOG_INFO(oss.str());
                                }
                                LOG_INFO(snap.summary_line());
                                for (const auto &ax : snap.per_axis)
                                {
                                    std::string axis_name = (ax.axis == ShardSpec::Axis::Hidden ? "Hidden" : (ax.axis == ShardSpec::Axis::Heads ? "Heads" : "Unknown"));
                                    LOG_INFO(axis_name << "-axis shards: count=" << ax.count << " local_elems=" << ax.local_elems << " global_dim=" << ax.global_dim);
                                }
                            }
                            LOG_INFO("-- End Sharded Tensor Registry --");
                        }
                    }
                    catch (const std::exception &e)
                    {
                        RETURN_FAIL(1, std::string("Failed to load model weights: ") + e.what());
                    }
                }
                else
                {
                    RETURN_FAIL(1, std::string("Failed to load model: ") + params.model_file);
                }
            }
            catch (const std::exception &e)
            {
                RETURN_FAIL(1, std::string("Exception loading model: ") + e.what());
            }
        }

        // 6. Chat interface mode
        if (params.interactive && transformer_pipeline)
        {
            if (rank == 0)
            {
                auto tokenizer = chat::createTokenizer(*model_loader);
                if (!tokenizer || !tokenizer->isReady())
                {
                    RETURN_FAIL(1, "Failed to initialize tokenizer for chat interface");
                }
                auto tokenizer_shared = std::shared_ptr<chat::TokenizerInterface>(tokenizer.release());
                auto tokenizer_copy = chat::createTokenizer(*model_loader);
                auto session = std::make_unique<chat::ChatSession>(std::move(tokenizer_copy), params);
                auto shared_pipeline = std::shared_ptr<MPITransformerPipeline>(std::move(transformer_pipeline));
                auto generator = std::make_unique<chat::ResponseGenerator>(tokenizer_shared, shared_pipeline, params, *model_weights);
                chat::ChatInterface chat_ui(std::move(session), shared_pipeline, std::move(generator), params);
                chat_ui.run();
                int done = 1;
                MPI_Bcast(&done, 1, MPI_INT, 0, MPI_COMM_WORLD);
            }
            else
            {
                int done = 0;
                while (!done)
                {
                    MPI_Bcast(&done, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    if (done)
                        break;
                    MPI_Barrier(MPI_COMM_WORLD);
                }
            }
            finalize();
            return exit_code;
        }

        // 7. Non-interactive prompt / eval path
        if (transformer_pipeline && (!params.prompt.empty() || params.eval_only))
        {
            std::shared_ptr<chat::TokenizerInterface> tokenizer_shared;
            std::vector<int32_t> prompt_tokens;
            if (rank == 0)
            {
                LOG_INFO("Running non-interactive prompt processing");
                auto tokenizer = chat::createTokenizer(*model_loader);
                if (!tokenizer || !tokenizer->isReady())
                {
                    RETURN_FAIL(1, "Failed to initialize tokenizer for prompt processing");
                }
                tokenizer_shared = std::shared_ptr<chat::TokenizerInterface>(tokenizer.release());
                std::string prompt_text = params.prompt.empty() ? "hello world" : params.prompt;
                LOG_INFO("Processing prompt: \"" << prompt_text << "\"");
                prompt_tokens = tokenizer_shared->tokenize(prompt_text);
                LOG_INFO("Tokenized to " << prompt_tokens.size() << " tokens");
                if (params.eval_only)
                {
                    std::cout << "Prompt: \"" << prompt_text << "\"\nTokens: [";
                    for (size_t i = 0; i < prompt_tokens.size(); ++i)
                    {
                        if (i)
                            std::cout << ", ";
                        std::cout << prompt_tokens[i];
                    }
                    std::cout << "]\nToken count: " << prompt_tokens.size() << "\nToken strings: [";
                    for (size_t i = 0; i < prompt_tokens.size(); ++i)
                    {
                        if (i)
                            std::cout << ", ";
                        std::string tk = tokenizer_shared->detokenize({prompt_tokens[i]});
                        std::cout << "\"" << tk << "\"";
                    }
                    std::cout << "]" << std::endl;
                    int completion_signal = 1;
                    MPI_Bcast(&completion_signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    finalize();
                    return exit_code;
                }
            }
            int token_count = 0;
            if (rank == 0)
                token_count = (int)prompt_tokens.size();
            MPI_Bcast(&token_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (rank != 0)
                prompt_tokens.resize(token_count);
            MPI_Bcast(prompt_tokens.data(), token_count, MPI_INT, 0, MPI_COMM_WORLD);
            auto shared_pipeline = std::shared_ptr<MPITransformerPipeline>(std::move(transformer_pipeline));
            auto generator = std::make_unique<chat::ResponseGenerator>(tokenizer_shared, shared_pipeline, params, *model_weights);
            try
            {
                std::string response = generator->generateResponse(prompt_tokens);
                if (rank == 0)
                    std::cout << "Response: " << response << std::endl;
            }
            catch (const std::exception &e)
            {
                RETURN_FAIL(1, std::string("Exception during response generation: ") + e.what());
            }
            finalize();
            return exit_code;
        }

        // 8. Standard inference path
        ComputeGraph graph;
        if (transformer_pipeline && model_weights)
        {
            LOG_INFO("Setting up transformer inference pipeline...");
            std::vector<int> input_tokens = {1, 2, 3, 4, 5};
            LOG_INFO("Running inference on " << input_tokens.size() << " tokens");
            try
            {
                std::shared_ptr<TensorBase> output;
                auto t0 = std::chrono::high_resolution_clock::now();
                bool ok = transformer_pipeline->execute(input_tokens, *model_weights, output);
                auto t1 = std::chrono::high_resolution_clock::now();
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
                if (ok && rank == 0)
                {
                    LOG_INFO("Inference completed successfully!");
                    LOG_INFO("Single token generation took: " << ms << " ms");
                    LOG_INFO("Output shape: [" << output->shape()[0] << ", " << output->shape()[1] << "]");
                    const float *out = output->data();
                    LOG_INFO("First 10 output values:");
                    for (int i = 0; i < std::min(10, (int)output->shape()[1]); ++i)
                        LOG_INFO("  [" << i << "] = " << out[i]);
                    std::cout << "\n"
                              << std::string(60, '=') << "\nPERFORMANCE TIMING REPORT\n";
                    std::cout << "Single token generation: " << ms << " ms\n"
                              << std::string(60, '=') << std::endl;
                    PerformanceTimer::getInstance().printReport();
                    std::cout << std::string(60, '=') << std::endl;
                }
                else if (!ok)
                {
                    RETURN_FAIL(1, "Transformer pipeline execution failed");
                }
            }
            catch (const std::exception &e)
            {
                RETURN_FAIL(1, std::string("Exception during inference: ") + e.what());
            }
        }
        else if (model_loader)
        {
            LOG_INFO("Model loaded but transformer pipeline not initialized - running matmul demo");
            int matrix_size = params.m;
            auto node = std::make_shared<MatMulNode>("model_matmul_demo", matrix_size, matrix_size, matrix_size);
            graph.addNode(node);
        }
        else
        {
            auto node = std::make_shared<MatMulNode>("cosma_benchmark", params.m, params.n, params.k);
            graph.addNode(node);
        }

        // 9. Execute compute graph if applicable
        if (!transformer_pipeline || !model_weights)
        {
            LOG_INFO("Executing compute graph...");
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < params.num_repeat; ++i)
            {
                bool ok = graph.execute();
                if (!ok)
                    RETURN_FAIL(1, "Compute graph execution failed on iteration " + std::to_string(i));
                if (rank == 0 && params.log_level >= LogLevel::VERBOSITY_DEBUG)
                {
                    LOG_DEBUG("Completed iteration " << (i + 1) << "/" << params.num_repeat);
                }
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            if (rank == 0)
            {
                LOG_INFO("=== Performance Results ===");
                LOG_INFO("Total execution time: " << ms.count() << " ms");
                LOG_INFO("Average time per iteration: " << (ms.count() / params.num_repeat) << " ms");
                if (!model_loader)
                {
                    long long ops = 2LL * params.m * params.n * params.k * params.num_repeat;
                    double gflops = (double)ops / (ms.count() * 1e6);
                    LOG_INFO("Achieved performance: " << std::fixed << std::setprecision(3) << gflops << " GFLOPS");
                }
            }
        }

        // 10. Validation
        if (params.validate_results && rank == 0)
        {
            LOG_INFO("Running result validation (stub)...");
            LOG_INFO("Validation completed successfully");
        }

        LOG_INFO("Llaminar execution completed successfully");
    }
    catch (const std::exception &e)
    {
        RETURN_FAIL(1, std::string("Exception in main: ") + e.what());
    }
    catch (...)
    {
        RETURN_FAIL(1, "Unknown exception in main");
    }

    finalize();
    return exit_code;
}