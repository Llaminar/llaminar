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

using namespace llaminar2;

void list_devices()
{
    auto &dm = DeviceManager::instance();
    dm.initialize();

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

    // Initialize device manager
    auto &dm = DeviceManager::instance();
    dm.initialize();

    // Handle list devices
    if (args.list_devices)
    {
        if (MPIContextFactory::global()->rank() == 0)
        {
            list_devices();
        }
        MPI_Finalize();
        return 0;
    }

    // Validate required arguments
    if (args.model_path.empty())
    {
        if (MPIContextFactory::global()->rank() == 0)
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

    // Get MPI context
    auto mpi_ctx = MPIContextFactory::global();

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

    // Create model context (loads metadata but not weights yet)
    auto model_ctx = ModelContext::create(args.model_path, mpi_ctx, nullptr);
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
    model_ctx = ModelContext::create(args.model_path, mpi_ctx, placement_map);
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

    if (mpi_ctx->rank() == 0)
    {
        const auto &devices = dm.devices();
        LOG_INFO("\n=== Llaminar v2 ===");
        LOG_INFO("Model: " << args.model_path);
        LOG_INFO("Architecture: " << architecture);
        LOG_INFO("Device: " << device_idx << " (" << devices[device_idx].name << ")");
        LOG_INFO("Strategy: " << args.strategy);
        LOG_INFO("MPI ranks: " << mpi_ctx->world_size());
        LOG_INFO("");
    }

    // Create runtime configuration from parsed arguments
    PipelineConfig pipeline_config;
    pipeline_config.max_seq_len = args.max_seq_len;
    pipeline_config.n_threads = args.n_threads;
    pipeline_config.batch_size = args.batch_size;
    pipeline_config.use_mmap = args.use_mmap;
    pipeline_config.seed = args.seed;

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

    // TODO: Tokenize prompt (for now, use dummy tokens)
    std::vector<int> tokens = {1, 2, 3, 4, 5, 6, 7, 8}; // Placeholder

    // Run inference
    if (mpi_ctx->rank() == 0)
    {
        LOG_INFO("Running inference...\n");
        LOG_INFO("Prompt: \"" << args.prompt << "\"\n");
        LOG_INFO("Generating " << args.n_predict << " tokens...\n\n");
    }

    if (!pipeline->forward(tokens.data(), tokens.size()))
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_ERROR("Error: Forward pass failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Generate tokens
    for (int i = 0; i < args.n_predict; ++i)
    {
        const float *logits = pipeline->logits();

        // TODO: Sample next token (for now, greedy argmax)
        // tokens.push_back(next_token);

        // Decode single token
        // pipeline->forward(tokens.data() + tokens.size() - 1, 1);
    }

    if (mpi_ctx->rank() == 0)
    {
        LOG_INFO("\nInference complete.\n");
    }

    MPI_Finalize();
    return 0;
}
