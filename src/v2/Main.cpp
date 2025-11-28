/**
 * @file Main.cpp
 * @brief Llaminar v2 entry point
 *
 * Clean greenfield implementation with:
 * - Device manager initialization
 * - Multi-GPU heterogeneous support
 * - Direct kernel orchestration
 * - Architecture-agnostic pipeline creation via PipelineFactory
 *
 * @author David Sanftenberg
 */

#include "utils/Logger.h"
#include "utils/MPIContext.h"
#include "utils/ArgParser.h"
#include "utils/NUMATopology.h"
#include "utils/Tokenizer.h"
#include "utils/Sampler.h"
#include "utils/ChatUI.h"
#include "utils/ChatTemplate.h"
#include "backends/ComputeBackend.h"
#include "pipelines/PipelineFactory.h"
#include "pipelines/PipelineConfig.h"
#include "pipelines/qwen/Qwen2Pipeline.h"
#include "loaders/ModelLoader.h"
#include "loaders/ModelContext.h"
#include "loaders/DeviceOrchestrator.h"
#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>

using namespace llaminar2;

void list_devices()
{
    auto &dm = DeviceManager::instance();
    // List devices without NUMA filtering (show all available devices)
    dm.initialize(-1);

    const auto &devices = dm.devices();

    LOG_INFO("\n=== Available Devices ===\n\n");
    for (size_t i = 0; i < devices.size(); ++i)
    {
        const auto &dev = devices[i];
        LOG_INFO("Device " << i << ": ");

        switch (dev.type)
        {
        case ComputeBackendType::CPU_OPENBLAS:
            LOG_INFO("CPU (OpenBLAS)");
            break;
        case ComputeBackendType::CPU_MKL:
            LOG_INFO("CPU (Intel MKL)");
            break;
        case ComputeBackendType::GPU_CUDA:
            LOG_INFO("GPU (CUDA) - " << dev.name);
            break;
        case ComputeBackendType::GPU_ROCM:
            LOG_INFO("GPU (ROCm) - " << dev.name);
            break;
        case ComputeBackendType::GPU_VULKAN:
            LOG_INFO("GPU (Vulkan) - " << dev.name);
            break;
        case ComputeBackendType::GPU_METAL:
            LOG_INFO("GPU (Metal) - " << dev.name);
            break;
        }

        if (dev.total_memory_bytes > 0)
        {
            double total_gb = dev.total_memory_bytes / (1024.0 * 1024.0 * 1024.0);
            double free_gb = dev.free_memory_bytes / (1024.0 * 1024.0 * 1024.0);
            LOG_INFO(" (" << total_gb << " GB total, " << free_gb << " GB free)");
        }

        LOG_INFO("\n");
    }

    LOG_INFO("\n");
}

int parse_device(const std::string &device_str, DeviceManager &dm)
{
    if (device_str == "auto")
    {
        return dm.select_device();
    }

    if (device_str == "cpu")
    {
        return dm.find_device(ComputeBackendType::CPU_OPENBLAS, 0);
    }

    if (device_str.substr(0, 5) == "cuda:")
    {
        int device_id = std::stoi(device_str.substr(5));
        return dm.find_device(ComputeBackendType::GPU_CUDA, device_id);
    }

    if (device_str.substr(0, 5) == "rocm:")
    {
        int device_id = std::stoi(device_str.substr(5));
        return dm.find_device(ComputeBackendType::GPU_ROCM, device_id);
    }

    LOG_ERROR("Error: Unknown device format: " << device_str);
    return -1;
}

int main(int argc, char *argv[])
{
    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    // Initialize logging from environment (LLAMINAR_LOG_LEVEL)
    initializeLogging();

    // Ensure pipeline registrations (static constructors may not run in executables)
    ensureQwen2Registration();

    // Parse command-line arguments using centralized ArgParser
    ArgContext args = ArgParser::parse(argc, argv);

    // Handle help
    if (args.show_help)
    {
        if (MPIContextFactory::global()->rank() == 0)
        {
            ArgParser::printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    // Detect NUMA node for MPI rank
    auto numa_info = NUMATopology::detectLocalNUMANode();
    auto mpi_ctx = MPIContextFactory::global();

    if (mpi_ctx->rank() == 0)
    {
        LOG_INFO("=== NUMA Topology Detection ===");
    }

    LOG_INFO("[Rank " << mpi_ctx->rank() << "] NUMA node: " << numa_info.local_numa_node
                      << " (detection: " << numa_info.detection_method << ")");

    if (!numa_info.detection_succeeded && mpi_ctx->rank() == 0)
    {
        LOG_WARN("NUMA detection failed, using fallback node 0. This may impact multi-socket performance.");
    }

    // Initialize device manager with NUMA-aware filtering
    auto &dm = DeviceManager::instance();
    dm.initialize(numa_info.local_numa_node);

    // Handle list devices
    if (args.list_devices)
    {
        if (mpi_ctx->rank() == 0)
        {
            list_devices();
        }
        MPI_Finalize();
        return 0;
    }

    // Validate required arguments
    if (args.model_path.empty())
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_ERROR("Error: Model path required (-m)\n\n");
            ArgParser::printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // Parse device
    int device_idx = parse_device(args.device, dm);
    if (device_idx < 0)
    {
        MPI_Finalize();
        return 1;
    }

    // Parse placement strategy
    PlacementStrategy strategy = PlacementStrategy::AUTO;
    if (args.strategy == "all-gpu")
    {
        strategy = PlacementStrategy::ALL_GPU;
    }
    else if (args.strategy == "all-cpu")
    {
        strategy = PlacementStrategy::ALL_CPU;
    }
    else if (args.strategy == "layer-split")
    {
        strategy = PlacementStrategy::LAYER_SPLIT;
    }
    else if (args.strategy == "memory-aware")
    {
        strategy = PlacementStrategy::MEMORY_AWARE;
    }
    else if (args.strategy == "moe-optimized")
    {
        strategy = PlacementStrategy::MOE_OPTIMIZED;
    }
    else if (args.strategy == "custom")
    {
        strategy = PlacementStrategy::CUSTOM;
    }
    else if (args.strategy != "auto")
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_ERROR("Warning: Unknown strategy '" << args.strategy << "', using AUTO\n");
        }
    }

    // Create orchestration config from ArgContext
    OrchestrationConfig orch_config;
    orch_config.strategy = strategy;
    orch_config.gpu_device_idx = device_idx;
    orch_config.offload_layers = args.offload_layers;
    orch_config.verbose = args.verbose;
    // TODO: Add new fields to OrchestrationConfig for Phase 2:
    // orch_config.device_map = args.device_map;
    // orch_config.max_gpu_memory_mb = args.max_gpu_memory_mb;
    // orch_config.max_cpu_memory_mb = args.max_cpu_memory_mb;
    // orch_config.moe_shared_experts_gpu = args.moe_shared_experts_gpu;
    // orch_config.moe_sparse_experts_cpu = args.moe_sparse_experts_cpu;
    // orch_config.multi_gpu = args.multi_gpu;
    // orch_config.gpu_split = args.gpu_split;

    // Create device orchestrator
    auto device_mgr_shared = std::shared_ptr<DeviceManager>(&dm, [](DeviceManager *) {});
    auto orchestrator = std::make_shared<DeviceOrchestrator>(
        device_mgr_shared, mpi_ctx, orch_config);

    // Create runtime configuration from parsed arguments
    PipelineConfig pipeline_config;
    pipeline_config.max_seq_len = args.max_seq_len;
    pipeline_config.n_threads = args.n_threads;
    pipeline_config.batch_size = args.batch_size;
    pipeline_config.use_mmap = args.use_mmap;
    pipeline_config.seed = args.seed;

    // Parse weight precision mode
    if (args.weight_precision == "native")
    {
        pipeline_config.weight_precision = WeightPrecision::NATIVE;
    }
    else if (args.weight_precision == "fp32")
    {
        pipeline_config.weight_precision = WeightPrecision::CONVERT_TO_FP32;
    }
    else if (args.weight_precision == "bf16")
    {
        pipeline_config.weight_precision = WeightPrecision::CONVERT_TO_BF16;
    }
    else if (args.weight_precision == "fp16")
    {
        pipeline_config.weight_precision = WeightPrecision::CONVERT_TO_FP16;
    }
    else if (args.weight_precision == "int8")
    {
        pipeline_config.weight_precision = WeightPrecision::CONVERT_TO_INT8;
    }
    else
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_WARN("Unknown weight precision mode '" << args.weight_precision << "', defaulting to NATIVE");
        }
        pipeline_config.weight_precision = WeightPrecision::NATIVE;
    }

    // Parse activation precision mode
    if (args.activation_precision == "fp32")
    {
        pipeline_config.activation_precision = ActivationPrecision::FP32;
    }
    else if (args.activation_precision == "bf16")
    {
        pipeline_config.activation_precision = ActivationPrecision::BF16;
    }
    else if (args.activation_precision == "fp16")
    {
        pipeline_config.activation_precision = ActivationPrecision::FP16;
    }
    else if (args.activation_precision == "int32")
    {
        pipeline_config.activation_precision = ActivationPrecision::INT32;
    }
    else
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_WARN("Unknown activation precision mode '" << args.activation_precision << "', defaulting to FP32");
        }
        pipeline_config.activation_precision = ActivationPrecision::FP32;
    }

    // Create model context (loads metadata but not weights yet)
    auto model_ctx = ModelContext::create(args.model_path, mpi_ctx, nullptr, nullptr,
                                          WeightDistributionStrategy::REPLICATED,
                                          pipeline_config.weight_precision);
    if (!model_ctx)
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_ERROR("Error: Failed to load model: " << args.model_path);
        }
        MPI_Finalize();
        return 1;
    }

    // Create placement map from orchestrator
    auto placement_map = orchestrator->createPlacementMap(model_ctx);

    // Re-create model context with placement map (this creates WeightManager)
    model_ctx = ModelContext::create(args.model_path, mpi_ctx, placement_map, nullptr,
                                     WeightDistributionStrategy::REPLICATED,
                                     pipeline_config.weight_precision);
    if (!model_ctx)
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_ERROR("Error: Failed to load model with placement map: " << args.model_path);
        }
        MPI_Finalize();
        return 1;
    }

    const auto &model = model_ctx->model();
    std::string architecture = model_ctx->architecture();

    // Log selected precision modes
    if (mpi_ctx->rank() == 0)
    {
        const char *weight_prec_name = "Unknown";
        switch (pipeline_config.weight_precision)
        {
        case WeightPrecision::NATIVE:
            weight_prec_name = "NATIVE (keep in GGUF format)";
            break;
        case WeightPrecision::CONVERT_TO_FP32:
            weight_prec_name = "FP32 (dequantize to FP32 at load)";
            break;
        case WeightPrecision::CONVERT_TO_BF16:
            weight_prec_name = "BF16 (dequantize to BF16 at load)";
            break;
        case WeightPrecision::CONVERT_TO_FP16:
            weight_prec_name = "FP16 (dequantize to FP16 at load)";
            break;
        case WeightPrecision::CONVERT_TO_INT8:
            weight_prec_name = "INT8 (dequantize to INT8 at load)";
            break;
        }

        const char *activation_prec_name = "Unknown";
        switch (pipeline_config.activation_precision)
        {
        case ActivationPrecision::FP32:
            activation_prec_name = "FP32 (32-bit float)";
            break;
        case ActivationPrecision::BF16:
            activation_prec_name = "BF16 (bfloat16)";
            break;
        case ActivationPrecision::FP16:
            activation_prec_name = "FP16 (16-bit float)";
            break;
        case ActivationPrecision::INT32:
            activation_prec_name = "INT32 (32-bit integer)";
            break;
        }

        LOG_INFO("Weight precision: " << weight_prec_name);
        LOG_INFO("Activation precision: " << activation_prec_name);
    }

    // Create pipeline using factory
    auto pipeline = PipelineFactory::instance().create(architecture, model_ctx, mpi_ctx, device_idx, pipeline_config);
    if (!pipeline)
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_ERROR("Error: Failed to create pipeline for architecture: " << architecture);
            std::string supported_archs = "Supported architectures: ";
            auto supported = PipelineFactory::instance().supportedArchitectures();
            for (size_t i = 0; i < supported.size(); ++i)
            {
                supported_archs += supported[i];
                if (i + 1 < supported.size())
                    supported_archs += ", ";
            }
            LOG_ERROR(supported_archs);
        }
        MPI_Finalize();
        return 1;
    }

    // Create tokenizer from model context (avoids re-loading the model file)
    std::shared_ptr<ITokenizer> tokenizer;
    try
    {
        tokenizer = createTokenizer(model_ctx);
        if (!tokenizer)
        {
            if (mpi_ctx->rank() == 0)
            {
                LOG_ERROR("Failed to create tokenizer from model context");
            }
            MPI_Finalize();
            return 1;
        }
    }
    catch (const std::exception &e)
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_ERROR("Error creating tokenizer: " << e.what());
        }
        MPI_Finalize();
        return 1;
    }

    // Handle chat template override if specified
    if (!args.chat_template.empty())
    {
        // Parse template type from string
        ChatTemplateType override_type = ChatTemplateType::UNKNOWN;
        std::string tmpl_lower = args.chat_template;
        std::transform(tmpl_lower.begin(), tmpl_lower.end(), tmpl_lower.begin(), ::tolower);
        
        if (tmpl_lower == "chatml") override_type = ChatTemplateType::CHATML;
        else if (tmpl_lower == "llama3") override_type = ChatTemplateType::LLAMA3;
        else if (tmpl_lower == "llama2") override_type = ChatTemplateType::LLAMA2;
        else if (tmpl_lower == "mistral" || tmpl_lower == "mistral_v1") override_type = ChatTemplateType::MISTRAL_V1;
        else if (tmpl_lower == "mistral_v3") override_type = ChatTemplateType::MISTRAL_V3;
        else if (tmpl_lower == "mistral_v7") override_type = ChatTemplateType::MISTRAL_V7;
        else if (tmpl_lower == "phi3") override_type = ChatTemplateType::PHI3;
        else if (tmpl_lower == "phi4") override_type = ChatTemplateType::PHI4;
        else if (tmpl_lower == "gemma") override_type = ChatTemplateType::GEMMA;
        else if (tmpl_lower == "deepseek") override_type = ChatTemplateType::DEEPSEEK;
        else if (tmpl_lower == "deepseek2") override_type = ChatTemplateType::DEEPSEEK2;
        else if (tmpl_lower == "deepseek3") override_type = ChatTemplateType::DEEPSEEK3;
        else if (tmpl_lower == "zephyr") override_type = ChatTemplateType::ZEPHYR;
        else if (tmpl_lower == "vicuna") override_type = ChatTemplateType::VICUNA;
        else if (tmpl_lower == "command_r" || tmpl_lower == "command-r") override_type = ChatTemplateType::COMMAND_R;
        else
        {
            if (mpi_ctx->rank() == 0)
            {
                LOG_WARN("Unknown chat template '" << args.chat_template << "', using model's template");
            }
        }
        
        if (override_type != ChatTemplateType::UNKNOWN)
        {
            tokenizer->setChatTemplate(ChatTemplate::create(override_type));
            if (mpi_ctx->rank() == 0)
            {
                LOG_INFO("Using chat template override: " << args.chat_template);
            }
        }
    }

    // ========================================================================
    // Chat Mode Handling
    // ========================================================================
    
    // Interactive chat mode (--chat)
    if (args.chat_mode)
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
            chat_config.system_prompt = args.system_prompt;
            chat_config.max_tokens = args.n_predict;
            chat_config.temperature = args.temperature;
            chat_config.top_k = args.top_k;
            chat_config.top_p = args.top_p;
            
            // Convert unique_ptr to shared_ptr for ChatUI
            std::shared_ptr<PipelineBase> shared_pipeline(std::move(pipeline));
            
            ChatUI chat_ui(tokenizer, shared_pipeline, chat_config);
            int result = chat_ui.run();
            
            MPI_Finalize();
            return result;
        }
        else
        {
            // Non-rank-0 processes wait for chat to complete
            // TODO: Implement proper multi-rank chat support
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            return 0;
        }
    }
    
    // Single-shot chat mode (--chat-single)
    if (args.single_shot_chat)
    {
        if (!tokenizer->hasChatTemplate())
        {
            if (mpi_ctx->rank() == 0)
            {
                LOG_ERROR("Chat mode requires a model with a chat template.");
                LOG_ERROR("Use --chat-template to specify one (e.g., --chat-template chatml)");
            }
            MPI_Finalize();
            return 1;
        }

        if (mpi_ctx->rank() == 0)
        {
            LOG_INFO("Running single-shot chat...");
            
            ChatUIConfig chat_config;
            chat_config.max_tokens = args.n_predict;
            chat_config.temperature = args.temperature;
            chat_config.top_k = args.top_k;
            chat_config.top_p = args.top_p;
            
            // Convert unique_ptr to shared_ptr for runSingleShotChat
            std::shared_ptr<PipelineBase> shared_pipeline(std::move(pipeline));
            
            runSingleShotChat(tokenizer, shared_pipeline, args.prompt, args.system_prompt, chat_config);
        }
        
        MPI_Finalize();
        return 0;
    }

    // ========================================================================
    // Standard Inference Mode (original code path)
    // ========================================================================

    // Tokenize prompt
    std::vector<int> tokens;
    try
    {
        // Encode with BOS token for instruction-tuned models
        tokens = tokenizer->encode(args.prompt, /*add_bos=*/true, /*add_eos=*/false);

        if (tokens.empty())
        {
            if (mpi_ctx->rank() == 0)
            {
                LOG_ERROR("Tokenization resulted in empty token sequence");
            }
            MPI_Finalize();
            return 1;
        }

        if (mpi_ctx->rank() == 0)
        {
            LOG_INFO("Tokenized prompt: " << tokens.size() << " tokens");
        }
    }
    catch (const std::exception &e)
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_ERROR("Error tokenizing prompt: " << e.what());
        }
        MPI_Finalize();
        return 1;
    }

    // Create sampler with seed for reproducibility
    Sampler sampler(args.seed);
    SamplingParams sampling_params;
    sampling_params.temperature = args.temperature;
    sampling_params.top_k = args.top_k;
    sampling_params.top_p = args.top_p;
    sampling_params.seed = args.seed;

    if (mpi_ctx->rank() == 0)
    {
        LOG_INFO("Sampling parameters:");
        LOG_INFO("  temperature: " << sampling_params.temperature);
        LOG_INFO("  top_k: " << sampling_params.top_k);
        LOG_INFO("  top_p: " << sampling_params.top_p);
        LOG_INFO("  seed: " << sampling_params.seed);
    }

    // Run prefill inference
    if (mpi_ctx->rank() == 0)
    {
        LOG_INFO("Running prefill (" << tokens.size() << " tokens)...");
    }

    if (!pipeline->forward(tokens.data(), tokens.size()))
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_ERROR("Error: Prefill forward pass failed");
        }
        MPI_Finalize();
        return 1;
    }

    if (mpi_ctx->rank() == 0)
    {
        LOG_INFO("Prefill complete. Generating " << args.n_predict << " tokens...\n");
        // Print the prompt (decoded from tokens)
        std::string decoded_prompt = tokenizer->decode(tokens, /*remove_special=*/true);
        std::cout << decoded_prompt << std::flush;
    }

    // Get EOS token ID for early stopping
    int eos_token_id = tokenizer->eos_token();

    // Generate tokens autoregressively
    for (int i = 0; i < args.n_predict; ++i)
    {
        LOG_DEBUG("[Rank " << mpi_ctx->rank() << "] Starting decode iteration " << i);

        // Get logits from last forward pass
        LOG_DEBUG("[Rank " << mpi_ctx->rank() << "] Getting logits...");
        const float *logits = pipeline->logits();
        LOG_DEBUG("[Rank " << mpi_ctx->rank() << "] Logits retrieved");

        // Get vocabulary size from tokenizer
        size_t vocab_size = tokenizer->vocab_size();

        // Convert logits to vector for sampling (only on rank 0)
        int next_token = -1;
        if (mpi_ctx->rank() == 0)
        {
            LOG_DEBUG("[Rank 0] Sampling token...");
            std::vector<float> logits_vec(logits, logits + vocab_size);

            // Sample next token
            try
            {
                next_token = sampler.sample(logits_vec, sampling_params);
                LOG_DEBUG("[Rank 0] Sampled token: " << next_token);
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("Error sampling next token: " << e.what());
                MPI_Finalize();
                return 1;
            }

            // Decode and print the token immediately (streaming output)
            std::string token_text = tokenizer->decode_token(next_token);
            std::cout << token_text << std::flush;

            // Check for early stopping
            if (next_token == eos_token_id)
            {
                if (args.verbose)
                {
                    LOG_INFO("\nGeneration stopped: EOS token encountered");
                }
                break;
            }
        }

        // Broadcast next_token to all ranks for synchronized decode
        LOG_DEBUG("[Rank " << mpi_ctx->rank() << "] Broadcasting token...");
        MPI_Bcast(&next_token, 1, MPI_INT, 0, MPI_COMM_WORLD);
        LOG_DEBUG("[Rank " << mpi_ctx->rank() << "] Token broadcast complete: " << next_token);

        // Check if rank 0 hit EOS
        if (next_token == eos_token_id)
        {
            break;
        }

        // Forward next token through pipeline (single token decode)
        LOG_DEBUG("[Rank " << mpi_ctx->rank() << "] Entering decode forward...");
        if (!pipeline->forward(&next_token, 1))
        {
            LOG_ERROR("[Rank " << mpi_ctx->rank() << "] Decode forward FAILED at token " << (i + 1));
            if (mpi_ctx->rank() == 0)
            {
                LOG_ERROR("\nError: Decode forward pass failed at token " << (i + 1));
            }
            MPI_Finalize();
            return 1;
        }
        LOG_DEBUG("[Rank " << mpi_ctx->rank() << "] Decode forward SUCCESS");
    }

    if (mpi_ctx->rank() == 0)
    {
        std::cout << "\n"
                  << std::endl; // Final newline
        LOG_INFO("Generation complete.");
    }

    MPI_Finalize();
    return 0;
}
