/**
 * @file Main.cpp
 * @brief Llaminar v2 entry point
 *
 * Clean greenfield implementation with:
 * - Device manager initialization
 * - Multi-GPU heterogeneous support
 * - Direct kernel orchestration
 * - Architecture-agnostic pipeline creation via InferenceRunner factory
 * - Self-bootstrap MPI support (auto-launches mpirun if not in MPI context)
 *
 * @author David Sanftenberg
 */

#include "utils/Logger.h"
#include "utils/MPIContext.h"
#include "utils/MPIBootstrap.h"
#include "utils/ArgParser.h"
#include "utils/NUMATopology.h"
#include "utils/Tokenizer.h"
#include "utils/Sampler.h"
#include "utils/ChatUI.h"
#include "utils/ChatTemplate.h"
#include "utils/BenchmarkRunner.h"
#include "backends/ComputeBackend.h"
#include "execution/InferenceRunnerFactory.h"
#include "execution/IInferenceRunner.h"
#include "execution/RuntimeConfig.h"
#include "loaders/ModelLoader.h"
#include "loaders/ModelContext.h"
#include "loaders/DeviceOrchestrator.h"
#include <mpi.h>
#include <iostream>
#include <climits>
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

int parse_device(const std::string &device_str, DeviceManager &dm)
{
    if (device_str == "auto")
    {
        return dm.select_device();
    }

    if (device_str == "cpu")
    {
        return dm.find_device(ComputeBackendType::CPU, 0);
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

    // Parse command-line arguments (before MPI init for bootstrap decisions)
    ArgContext args = ArgParser::parse(argc, argv);

    // Handle parse errors early
    if (!args.error.empty())
    {
        std::cerr << "Error: " << args.error << std::endl;
        ArgParser::printUsage(argv[0]);
        return 1;
    }

    // Handle help early (no MPI needed)
    if (args.show_help)
    {
        ArgParser::printUsage(argv[0]);
        return 0;
    }

    // Handle list-devices early (no MPI needed)
    if (args.list_devices)
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
    if (!mpi_env.is_mpi_process && !args.mpi_no_bootstrap)
    {
        // Build launch configuration from args and topology
        MPILaunchConfig launch_config = MPIBootstrap::getDefaultConfig(cpu_topology);

        // Override with user-specified values
        if (args.mpi_procs > 0)
        {
            launch_config.num_procs = args.mpi_procs;
        }
        if (!args.hostfile.empty())
        {
            launch_config.hostfile = args.hostfile;
        }
        launch_config.report_bindings = args.mpi_verbose || args.verbose;
        launch_config.verbose = args.mpi_verbose;
        launch_config.oversubscribe = args.mpi_oversubscribe;

        // Handle dry-run: print config and exit
        if (args.mpi_dry_run)
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

    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    // Re-parse arguments (MPI_Init may modify argc/argv)
    args = ArgParser::parse(argc, argv);

    // Handle parse errors (shouldn't happen if first parse succeeded, but be safe)
    if (!args.error.empty())
    {
        std::cerr << "Error: " << args.error << std::endl;
        MPI_Finalize();
        return 1;
    }

    // Configure OpenMP environment for this rank
    // (In self-launch case, this was done before exec; here we do it post-MPI init)
    MPILaunchConfig mpi_launch_config = MPIBootstrap::getDefaultConfig(cpu_topology);
    if (args.mpi_procs > 0)
    {
        mpi_launch_config.num_procs = args.mpi_procs;
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
    else if (args.strategy == "multi-gpu")
    {
        strategy = PlacementStrategy::MULTI_GPU;
    }
    else if (args.strategy != "auto")
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_ERROR("Warning: Unknown strategy '" << args.strategy << "', using AUTO\n");
        }
    }

    // Auto-enable multi-gpu strategy if --multi-gpu or --gpus specified
    if (args.multi_gpu && strategy == PlacementStrategy::AUTO)
    {
        strategy = PlacementStrategy::MULTI_GPU;
    }

    // Create orchestration config from ArgContext
    OrchestrationConfig orch_config;
    orch_config.strategy = strategy;
    orch_config.gpu_device_idx = device_idx;
    orch_config.offload_layers = args.offload_layers;
    orch_config.verbose = args.verbose;
    orch_config.device_map = args.device_map;
    orch_config.max_gpu_memory_mb = args.max_gpu_memory_mb;
    orch_config.max_cpu_memory_mb = args.max_cpu_memory_mb;
    orch_config.moe_shared_experts_gpu = args.moe_shared_experts_gpu;
    orch_config.moe_sparse_experts_cpu = args.moe_sparse_experts_cpu;
    orch_config.multi_gpu = args.multi_gpu;
    orch_config.gpu_split = args.gpu_split;
    orch_config.gpu_devices = args.gpu_devices;

    // Create device orchestrator
    auto device_mgr_shared = std::shared_ptr<DeviceManager>(&dm, [](DeviceManager *) {});
    auto orchestrator = std::make_shared<DeviceOrchestrator>(
        device_mgr_shared, mpi_ctx, orch_config);

    // Create runtime configuration from parsed arguments
    RuntimeConfig runtime_config;
    runtime_config.max_seq_len = args.max_seq_len;
    runtime_config.n_threads = args.n_threads;
    runtime_config.batch_size = args.batch_size;
    runtime_config.use_mmap = args.use_mmap;
    runtime_config.seed = args.seed;

    // Parse weight precision mode
    if (args.weight_precision == "native")
    {
        runtime_config.weight_precision = WeightPrecision::NATIVE;
    }
    else if (args.weight_precision == "fp32")
    {
        runtime_config.weight_precision = WeightPrecision::CONVERT_TO_FP32;
    }
    else if (args.weight_precision == "bf16")
    {
        runtime_config.weight_precision = WeightPrecision::CONVERT_TO_BF16;
    }
    else if (args.weight_precision == "fp16")
    {
        runtime_config.weight_precision = WeightPrecision::CONVERT_TO_FP16;
    }
    else if (args.weight_precision == "int8")
    {
        runtime_config.weight_precision = WeightPrecision::CONVERT_TO_INT8;
    }
    else
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_WARN("Unknown weight precision mode '" << args.weight_precision << "', defaulting to NATIVE");
        }
        runtime_config.weight_precision = WeightPrecision::NATIVE;
    }

    // Parse activation precision mode
    if (args.activation_precision == "fp32")
    {
        runtime_config.activation_precision = ActivationPrecision::FP32;
    }
    else if (args.activation_precision == "bf16")
    {
        runtime_config.activation_precision = ActivationPrecision::BF16;
    }
    else if (args.activation_precision == "fp16")
    {
        runtime_config.activation_precision = ActivationPrecision::FP16;
    }
    else if (args.activation_precision == "q8_1")
    {
        runtime_config.activation_precision = ActivationPrecision::Q8_1;
    }
    else if (args.activation_precision == "hybrid")
    {
        runtime_config.activation_precision = ActivationPrecision::Hybrid;
    }
    else if (args.activation_precision == "hybridq16")
    {
        runtime_config.activation_precision = ActivationPrecision::HybridQ16;
    }
    else
    {
        // This branch should never be reached due to ArgParser validation,
        // but keep as defensive fallback
        if (mpi_ctx->rank() == 0)
        {
            LOG_ERROR("Invalid activation precision mode '" << args.activation_precision
                                                            << "' - this should have been caught by ArgParser");
        }
        runtime_config.activation_precision = ActivationPrecision::Hybrid; // Use default
    }

    // Fused attention + Wo kernel
    runtime_config.use_fused_attention = args.use_fused_attention;
    if (!args.fused_attention_backend_str.empty())
    {
        runtime_config.fused_attention_backend = parseFusedAttentionBackend(args.fused_attention_backend_str);
    }
    if (args.use_fused_attention && mpi_ctx->rank() == 0)
    {
        LOG_INFO("Fused attention+Wo kernel enabled (backend="
                 << fusedAttentionBackendToString(runtime_config.fused_attention_backend) << ")");
        // Fused attention supports Q8_1, Hybrid, and HybridQ16 modes
        // Q16_INTEGER backend specifically requires HybridQ16 mode
        if (runtime_config.activation_precision != ActivationPrecision::Q8_1 &&
            runtime_config.activation_precision != ActivationPrecision::Hybrid &&
            runtime_config.activation_precision != ActivationPrecision::HybridQ16)
        {
            LOG_WARN("Fused attention requires Q8_1, Hybrid, or HybridQ16 activation precision, current: "
                     << args.activation_precision << ". Will use unfused path.");
        }
        if (runtime_config.fused_attention_backend == FusedAttentionBackend::Q16_INTEGER &&
            runtime_config.activation_precision != ActivationPrecision::HybridQ16)
        {
            LOG_WARN("Q16_INTEGER backend requires HybridQ16 activation precision, current: "
                     << args.activation_precision << ". May not work correctly.");
        }
    }

    // Determine weight distribution strategy
    // Default: sharding enabled when world_size > 1 (unless explicitly disabled)
    WeightDistributionStrategy weight_strategy = WeightDistributionStrategy::REPLICATED;
    bool use_sharding = args.shard_weights ||
                        (mpi_ctx->world_size() > 1 && !args.disable_weight_sharding);

    if (use_sharding)
    {
        weight_strategy = WeightDistributionStrategy::SHARDED;

        // Row-parallel sharding now uses TensorSlice which preserves quantized format!
        // Column-parallel still requires FP32 for slicing (TODO: implement column-parallel TensorSlice)
        if (mpi_ctx->rank() == 0)
        {
            if (args.shard_weights)
            {
                LOG_INFO("Weight sharding enabled (--shard-weights) - row-parallel weights use TensorSlice");
            }
            else
            {
                LOG_INFO("Weight sharding auto-enabled for " << mpi_ctx->world_size()
                                                             << " MPI ranks (use --no-shard to disable)");
            }
        }
    }
    else if (mpi_ctx->world_size() > 1 && args.disable_weight_sharding)
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_INFO("Weight sharding disabled (--no-shard) - using replicated weights across "
                     << mpi_ctx->world_size() << " ranks");
        }
    }

    // Create model context (loads metadata but not weights yet)
    auto model_ctx = ModelContext::create(args.model_path, mpi_ctx, nullptr, nullptr,
                                          weight_strategy,
                                          runtime_config.weight_precision);
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
                                     weight_strategy,
                                     runtime_config.weight_precision);
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

    // Validate max_seq_len against model's context length
    if (model.context_length > 0)
    {
        if (runtime_config.max_seq_len > static_cast<int>(model.context_length))
        {
            if (mpi_ctx->rank() == 0)
            {
                LOG_WARN("Requested max_seq_len (" << runtime_config.max_seq_len
                                                   << ") exceeds model's context_length ("
                                                   << model.context_length << "). Clamping to model limit.");
            }
            runtime_config.max_seq_len = static_cast<int>(model.context_length);
        }

        if (mpi_ctx->rank() == 0)
        {
            LOG_INFO("KV cache size: " << runtime_config.max_seq_len
                                       << " tokens (model max: " << model.context_length << ")");
        }
    }

    // Log selected precision modes
    if (mpi_ctx->rank() == 0)
    {
        const char *weight_prec_name = "Unknown";
        switch (runtime_config.weight_precision)
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
        switch (runtime_config.activation_precision)
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
        case ActivationPrecision::Q8_1:
            activation_prec_name = "Q8_1 (quantized 8-bit with per-block scaling)";
            break;
        }

        LOG_DEBUG("Weight precision: " << weight_prec_name);
        LOG_DEBUG("Activation precision: " << activation_prec_name);
    }

    // Create inference runner using factory (auto-selects graph path by default)
    InferenceRunnerConfig runner_config;
    runner_config.max_seq_len = runtime_config.max_seq_len;
    runner_config.activation_precision = runtime_config.activation_precision;
    runner_config.fused_attention_backend = runtime_config.fused_attention_backend;

    auto runner = createInferenceRunner(model_ctx, mpi_ctx, device_idx, runner_config);
    if (!runner)
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_ERROR("Error: Failed to create inference runner for architecture: " << architecture);
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
                LOG_WARN("Unknown chat template '" << args.chat_template << "', using model's template");
            }
        }

        if (override_type != ChatTemplateType::UNKNOWN)
        {
            tokenizer->setChatTemplate(ChatTemplate::create(override_type));
            if (mpi_ctx->rank() == 0)
            {
                LOG_DEBUG("Using chat template override: " << args.chat_template);
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
            std::shared_ptr<IInferenceRunner> shared_runner(std::move(runner));

            ChatUI chat_ui(tokenizer, shared_runner, chat_config);
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
    // This mode applies the chat template to the prompt and generates a response.
    // All MPI ranks must participate in forward passes to avoid deadlocks.
    if (args.single_shot_chat)
    {
        if (!tokenizer->hasChatTemplate())
        {
            if (mpi_ctx->rank() == 0)
            {
                LOG_ERROR("Chat mode requires a model with a chat template.");
                LOG_ERROR("Use --chat-template to specify one (e.g., --chat-template chatml)");
            }
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            return 1;
        }

        if (mpi_ctx->rank() == 0)
        {
            LOG_INFO("Running single-shot chat...");
        }

        // Build conversation and encode with chat template (rank 0 only, then broadcast)
        std::vector<int> token_ids;
        int token_count = 0;

        if (mpi_ctx->rank() == 0)
        {
            std::vector<ChatMessage> conversation;
            if (!args.system_prompt.empty())
            {
                conversation.push_back(ChatMessage("system", args.system_prompt));
            }
            conversation.push_back(ChatMessage("user", args.prompt));

            token_ids = tokenizer->encodeChat(conversation, /*add_generation_prompt=*/true);
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

        if (!runner->forward(token_ids.data(), token_count))
        {
            if (mpi_ctx->rank() == 0)
            {
                LOG_ERROR("Chat prefill failed");
            }
            MPI_Finalize();
            return 1;
        }

        // Set up sampling
        int eos_token_id = tokenizer->eos_token();
        Sampler sampler(args.seed);
        SamplingParams sampling_params;
        sampling_params.temperature = args.temperature;
        sampling_params.top_k = args.top_k;
        sampling_params.top_p = args.top_p;
        sampling_params.seed = args.seed;

        // Determine max tokens: -1 means unlimited (use context size as practical limit)
        int max_tokens = args.n_predict;
        if (max_tokens < 0)
        {
            max_tokens = args.max_seq_len - token_count;
        }

        if (mpi_ctx->rank() == 0)
        {
            LOG_INFO("Generating response (max " << max_tokens << " tokens)...\n");
        }

        // Decode loop - all ranks participate in forward, rank 0 samples and outputs
        for (int i = 0; i < max_tokens; ++i)
        {
            int next_token = -1;

            // Rank 0: sample next token
            if (mpi_ctx->rank() == 0)
            {
                const float *logits = runner->logits();
                size_t vocab_size = tokenizer->vocab_size();
                std::vector<float> logits_vec(logits, logits + vocab_size);

                // Sample
                if (sampling_params.temperature < 0.01f)
                {
                    next_token = sampler.sample_greedy(logits_vec);
                }
                else
                {
                    next_token = sampler.sample(logits_vec, sampling_params);
                }

                // Output token text (streaming) - don't print stop tokens
                if (!tokenizer->is_stop_token(next_token))
                {
                    std::string token_text = tokenizer->decode_token(next_token);
                    std::cout << token_text << std::flush;
                }
            }

            // Broadcast next token to all ranks
            MPI_Bcast(&next_token, 1, MPI_INT, 0, MPI_COMM_WORLD);

            // Check for stop tokens (EOS, <|im_end|>, etc.)
            if (tokenizer->is_stop_token(next_token))
            {
                if (mpi_ctx->rank() == 0)
                {
                    LOG_DEBUG("Stop token encountered (" << next_token << "), stopping generation");
                }
                break;
            }

            // All ranks forward the next token
            if (!runner->forward(&next_token, 1))
            {
                if (mpi_ctx->rank() == 0)
                {
                    LOG_ERROR("Decode forward failed at token " << i);
                }
                MPI_Finalize();
                return 1;
            }
        }

        if (mpi_ctx->rank() == 0)
        {
            std::cout << std::endl;
            LOG_INFO("Chat generation complete.");
        }

        MPI_Finalize();
        return 0;
    }

    // ========================================================================
    // Benchmark Mode
    // ========================================================================
    if (args.benchmark_mode)
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_INFO("Running benchmark mode...");
        }

        // Convert unique_ptr to shared_ptr for BenchmarkRunner
        std::shared_ptr<IInferenceRunner> shared_runner(std::move(runner));

        BenchmarkRunner benchmark(shared_runner, tokenizer, mpi_ctx);
        BenchmarkResult result = benchmark.run(args);
        benchmark.printResults(result);

        MPI_Finalize();
        return result.success ? 0 : 1;
    }

    // ========================================================================
    // Standard Inference Mode (original code path)
    // ========================================================================

    // Tokenize prompt
    std::vector<int> tokens;
    try
    {
        // Encode WITHOUT BOS token (E2E tests don't use BOS)
        tokens = tokenizer->encode(args.prompt, /*add_bos=*/false, /*add_eos=*/false);

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
        LOG_DEBUG("Sampling parameters:");
        LOG_DEBUG("  temperature: " << sampling_params.temperature);
        LOG_DEBUG("  top_k: " << sampling_params.top_k);
        LOG_DEBUG("  top_p: " << sampling_params.top_p);
        LOG_DEBUG("  seed: " << sampling_params.seed);
    }

    // Run prefill inference
    if (mpi_ctx->rank() == 0)
    {
        LOG_DEBUG("Running prefill (" << tokens.size() << " tokens)...");
    }

    if (!runner->forward(tokens.data(), tokens.size()))
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
        if (args.n_predict == -1)
        {
            LOG_DEBUG("Prefill complete. Generating tokens until EOS...\n");
        }
        else
        {
            LOG_DEBUG("Prefill complete. Generating " << args.n_predict << " tokens...\n");
        }
    }

    // Get EOS token ID for early stopping
    int eos_token_id = tokenizer->eos_token();

    // Generate tokens autoregressively
    // n_predict = -1 means unlimited (generate until EOS)
    int max_tokens = (args.n_predict == -1) ? INT_MAX : args.n_predict;
    for (int i = 0; i < max_tokens; ++i)
    {
        LOG_DEBUG("[Rank " << mpi_ctx->rank() << "] Starting decode iteration " << i);

        // Get logits from last forward pass
        LOG_DEBUG("[Rank " << mpi_ctx->rank() << "] Getting logits...");
        const float *logits = runner->logits();
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
            // Don't print stop tokens
            if (!tokenizer->is_stop_token(next_token))
            {
                std::string token_text = tokenizer->decode_token(next_token);
                std::cout << token_text << std::flush;
            }

            // Check for early stopping (EOS, <|im_end|>, etc.)
            if (tokenizer->is_stop_token(next_token))
            {
                if (args.verbose)
                {
                    LOG_DEBUG("\nGeneration stopped: stop token " << next_token << " encountered");
                }
                break;
            }
        }

        // Broadcast next_token to all ranks for synchronized decode
        LOG_DEBUG("[Rank " << mpi_ctx->rank() << "] Broadcasting token...");
        MPI_Bcast(&next_token, 1, MPI_INT, 0, MPI_COMM_WORLD);
        LOG_DEBUG("[Rank " << mpi_ctx->rank() << "] Token broadcast complete: " << next_token);

        // Check if rank 0 hit stop token
        if (tokenizer->is_stop_token(next_token))
        {
            break;
        }

        // Forward next token through pipeline (single token decode)
        LOG_DEBUG("[Rank " << mpi_ctx->rank() << "] Entering decode forward...");
        if (!runner->forward(&next_token, 1))
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
        LOG_DEBUG("Generation complete.");
    }

    MPI_Finalize();
    return 0;
}
