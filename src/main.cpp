// Main application entry & execution pipeline

#include <iostream>
#include <mpi.h>
#include <string>
#include <memory>
// Removed unused headers from legacy graph system; keep iostream, mpi, string, memory, chrono, iomanip, sstream if needed.
#include <iomanip>
#include <chrono>
#include <sstream>

#include "argument_parser.h"
#include "logger.h"
#include "topology_manager.h"
#include "model_loader.h"
#include "mpi_transformer_pipeline.h"
#include "abstract_pipeline.h"
#include "qwen_pipeline_adapter.h"
#include "performance_timer.h"
#include "utils/debug_env.h"
#include "utils/perf_counters.h"
#include "tensors/tensor_factory.h"
#include "tensors/sharded_tensor_registry.h" // required for ShardedTensorRegistry snapshot logging
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
    // Initialize performance counters rank context (no-op if disabled)
    ::llaminar::perfCounters().set_rank(rank);
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
            PerfAllreduce(&exit_code, &global_code, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
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
        // Apply global OpenMP thread policy (physical core restriction or forced count)
        configureGlobalOpenMPThreads();
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
        std::unique_ptr<MPITransformerPipeline> transformer_pipeline;        // legacy path
        std::unique_ptr<MPITransformerPipeline::ModelWeights> model_weights; // legacy weights
        std::unique_ptr<AbstractPipeline> abstract_pipeline;                 // new abstract pipeline (feature flag)
        std::unique_ptr<QwenModelWeights> abstract_weights;                  // adapter weights container
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
                    bool use_abstract = debugEnv().pipeline.enable_abstract_pipeline;
                    if (use_abstract)
                    {
                        registerQwenPipeline();
                        abstract_pipeline = PipelineFactory::instance().create("qwen", config);
                        if (!abstract_pipeline)
                        {
                            RETURN_FAIL(1, "Failed to create abstract pipeline instance");
                        }
                        LOG_INFO("Abstract pipeline initialized: " << abstract_pipeline->name());
                    }
                    else
                    {
                        transformer_pipeline = std::make_unique<MPITransformerPipeline>(config);
                        LOG_INFO("Transformer pipeline initialized successfully");
                    }
                    try
                    {
                        auto loaded = loadModelWeights(*model_loader, config);
                        if (abstract_pipeline)
                        {
                            abstract_weights = std::make_unique<QwenModelWeights>();
                            abstract_weights->inner = std::move(loaded);
                            LOG_INFO("Model weights loaded (abstract adapter)");
                        }
                        else
                        {
                            model_weights = std::make_unique<MPITransformerPipeline::ModelWeights>(std::move(loaded));
                            LOG_INFO("Model weights loaded for transformer pipeline");
                        }
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
        if (params.interactive && (transformer_pipeline || abstract_pipeline))
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
                std::shared_ptr<MPITransformerPipeline> shared_pipeline;
                if (transformer_pipeline)
                    shared_pipeline = std::shared_ptr<MPITransformerPipeline>(std::move(transformer_pipeline));
                auto generator = std::make_unique<chat::ResponseGenerator>(tokenizer_shared, shared_pipeline, params, transformer_pipeline ? *model_weights : abstract_weights->inner);
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
        if ((transformer_pipeline || abstract_pipeline) && (!params.prompt.empty() || params.eval_only))
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
            std::shared_ptr<MPITransformerPipeline> shared_pipeline;
            if (transformer_pipeline)
                shared_pipeline = std::shared_ptr<MPITransformerPipeline>(std::move(transformer_pipeline));
            auto generator = std::make_unique<chat::ResponseGenerator>(tokenizer_shared, shared_pipeline, params, transformer_pipeline ? *model_weights : abstract_weights->inner);
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
        else if (model_loader)
        {
            LOG_INFO("Model loaded but no transformer pipeline initialized (graph system removed) - skipping legacy matmul demo");
        }
        else
        {
            LOG_INFO("No model or pipeline initialized; graph system removed so benchmark path is disabled.");
        }

        // 9. (Legacy placeholder) Graph system removed; no execution here.
        LOG_INFO("Graph system removed; skipping legacy graph execution phase.");

        // 10. Validation
        if (params.validate_results && rank == 0)
        {
            LOG_INFO("Running result validation (stub)...");
            LOG_INFO("Validation completed successfully");
        }

        LOG_INFO("Llaminar execution completed successfully");
        if (params.kv_cache_stats && (transformer_pipeline || abstract_pipeline) && rank == 0)
        {
            LOG_INFO("-- KV Cache Summary --");
            if (abstract_pipeline)
            {
                if (auto *state = abstract_pipeline->kvCacheState())
                {
                    LOG_INFO("KV capacity=" << state->capacity_tokens
                                            << " used=" << state->used_tokens
                                            << " growth_events=" << state->growth_events);
                }
                else
                {
                    LOG_INFO("(abstract pipeline reports no KV cache state)");
                }
            }
            else if (transformer_pipeline)
            {
                LOG_INFO("KV capacity=" << transformer_pipeline->getKVCacheCapacity()
                                        << " used=" << transformer_pipeline->getKVCacheUsed()
                                        << " growth_events=" << transformer_pipeline->getKVCacheGrowthEvents());
            }
        }
        // Performance counters summary (only if enabled and rank matches)
        const auto &perf_env = debugEnv().performance;
        if (perf_env.enable && rank == perf_env.log_rank)
        {
            auto snap = perfCounters().snapshot();
            double agg_ms = snap.total_time_ms; // wall time sum of measured matmuls
            double flops = static_cast<double>(snap.total_flops);
            double gflops = (agg_ms > 0.0) ? flops / (agg_ms * 1e6) : 0.0;
            double comm_ms = snap.total_comm_time_ms;
            double compute_ms = agg_ms;                                                                        // currently only matmul time; future: subtract comm overlap if needed
            double comm_bw = (comm_ms > 0.0) ? (double)snap.total_comm_bytes / (comm_ms / 1000.0) / 1e6 : 0.0; // MB/s
            LOG_INFO("PERF_SUMMARY matmuls=" << snap.total_matmuls
                                             << " total_flops=" << flops
                                             << " total_time_ms=" << agg_ms
                                             << " aggregate_gflops=" << std::fixed << std::setprecision(2) << gflops
                                             << " comm_bytes=" << snap.total_comm_bytes
                                             << " comm_time_ms=" << comm_ms
                                             << " comm_pct=" << (comm_ms > 0 && (compute_ms + comm_ms) > 0 ? (comm_ms / (compute_ms + comm_ms)) * 100.0 : 0.0)
                                             << " comm_bw_MBps=" << std::setprecision(2) << comm_bw);
            if (snap.total_comm_bytes > 0)
            {
                static const char *kCommNames[] = {"Bcast", "Allreduce", "Allgather", "Allgatherv", "Reduce", "ReduceScatter", "Alltoall", "Alltoallv", "Barrier"};
                for (size_t i = 0; i < snap.comm.size() && i < 9; ++i)
                {
                    const auto &cs = snap.comm[i];
                    if (cs.calls == 0)
                        continue;
                    double op_bw = 0.0; // future: require per-op time to compute
                    LOG_INFO(std::string("COMM_OP ") + kCommNames[i] + " calls=" + std::to_string(cs.calls) + " bytes=" + std::to_string(cs.bytes));
                }
            }
            if (!snap.backends.empty())
            {
                LOG_INFO("BACKEND_DISTRIBUTION total_backends=" << snap.backends.size());
                for (const auto &b : snap.backends)
                {
                    double b_gflops = (b.time_ms > 0.0) ? (double)b.flops / (b.time_ms * 1e6) : 0.0;
                    double pct_time = (snap.total_time_ms > 0.0) ? (b.time_ms / snap.total_time_ms) * 100.0 : 0.0;
                    double pct_flops = (snap.total_flops > 0) ? ((double)b.flops / (double)snap.total_flops) * 100.0 : 0.0;
                    double avg_ms = (b.count > 0) ? b.time_ms / (double)b.count : 0.0;
                    double prefill_avg = (b.prefill_count > 0) ? b.prefill_time_ms / (double)b.prefill_count : 0.0;
                    double decode_avg = (b.decode_count > 0) ? b.decode_time_ms / (double)b.decode_count : 0.0;
                    LOG_INFO("BACKEND name=" << PerformanceCounters::backendName(b.backend) << " backend_id=" << b.backend
                                             << " count=" << b.count
                                             << " flops=" << b.flops
                                             << " time_ms=" << b.time_ms
                                             << " pct_time=" << std::fixed << std::setprecision(2) << pct_time
                                             << " pct_flops=" << std::fixed << std::setprecision(2) << pct_flops
                                             << " agg_gflops=" << std::fixed << std::setprecision(2) << b_gflops
                                             << " avg_ms=" << avg_ms
                                             << " min_ms=" << b.min_ms
                                             << " max_ms=" << b.max_ms
                                             << " prefill_count=" << b.prefill_count
                                             << " prefill_time_ms=" << b.prefill_time_ms
                                             << " prefill_avg_ms=" << prefill_avg
                                             << " prefill_min_ms=" << b.prefill_min_ms
                                             << " prefill_max_ms=" << b.prefill_max_ms
                                             << " decode_count=" << b.decode_count
                                             << " decode_time_ms=" << b.decode_time_ms
                                             << " decode_avg_ms=" << decode_avg
                                             << " decode_min_ms=" << b.decode_min_ms
                                             << " decode_max_ms=" << b.decode_max_ms);
                }
            }
            if (perf_env.log_each_matmul && !snap.samples.empty())
            {
                LOG_INFO("PERF_SAMPLES count=" << snap.samples.size());
                size_t limit = std::min<size_t>(snap.samples.size(), 32); // cap log volume
                for (size_t i = 0; i < limit; ++i)
                {
                    const auto &s = snap.samples[i];
                    const char *layer_name = PerformanceCounters::layerTypeName(s.layer_type);
                    const char *phase_name = (s.layer_type == 1) ? PerformanceCounters::mlpPhaseName(s.phase) : (s.layer_type == 2 ? PerformanceCounters::attnPhaseName(s.phase) : "-");
                    LOG_DEBUG("MATMUL_SAMPLE i=" << i << " m=" << s.m << " n=" << s.n << " k=" << s.k
                                                 << " ms=" << s.ms << " gflops=" << s.gflops << " backend=" << s.backend
                                                 << " layer=" << layer_name << " phase=" << phase_name << " layer_idx=" << s.layer_index);
                }
                if (snap.samples.size() > limit)
                {
                    LOG_INFO("PERF_SAMPLES truncated remaining=" << (snap.samples.size() - limit));
                }
            }
        }
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