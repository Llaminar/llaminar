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

#include "utils/MPIContext.h"
#include "utils/ArgParser.h"
#include "backends/ComputeBackend.h"
#include "pipelines/PipelineFactory.h"
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

    std::cout << "\n=== Available Devices ===\n\n";
    for (size_t i = 0; i < devices.size(); ++i)
    {
        const auto &dev = devices[i];
        std::cout << "Device " << i << ": ";

        switch (dev.type)
        {
        case ComputeBackendType::CPU_OPENBLAS:
            std::cout << "CPU (OpenBLAS)";
            break;
        case ComputeBackendType::CPU_MKL:
            std::cout << "CPU (Intel MKL)";
            break;
        case ComputeBackendType::GPU_CUDA:
            std::cout << "GPU (CUDA) - " << dev.name;
            break;
        case ComputeBackendType::GPU_ROCM:
            std::cout << "GPU (ROCm) - " << dev.name;
            break;
        case ComputeBackendType::GPU_VULKAN:
            std::cout << "GPU (Vulkan) - " << dev.name;
            break;
        case ComputeBackendType::GPU_METAL:
            std::cout << "GPU (Metal) - " << dev.name;
            break;
        }

        if (dev.total_memory_bytes > 0)
        {
            double total_gb = dev.total_memory_bytes / (1024.0 * 1024.0 * 1024.0);
            double free_gb = dev.free_memory_bytes / (1024.0 * 1024.0 * 1024.0);
            std::cout << " (" << total_gb << " GB total, " << free_gb << " GB free)";
        }

        std::cout << "\n";
    }

    std::cout << "\n";
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

    std::cerr << "Error: Unknown device format: " << device_str << "\n";
    return -1;
}

int main(int argc, char *argv[])
{
    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

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
            std::cerr << "Error: Model path required (-m)\n\n";
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
    if (args.strategy == "all-gpu") {
        strategy = PlacementStrategy::ALL_GPU;
    } else if (args.strategy == "all-cpu") {
        strategy = PlacementStrategy::ALL_CPU;
    } else if (args.strategy == "layer-split") {
        strategy = PlacementStrategy::LAYER_SPLIT;
    } else if (args.strategy == "memory-aware") {
        strategy = PlacementStrategy::MEMORY_AWARE;
    } else if (args.strategy == "moe-optimized") {
        strategy = PlacementStrategy::MOE_OPTIMIZED;
    } else if (args.strategy == "custom") {
        strategy = PlacementStrategy::CUSTOM;
    } else if (args.strategy != "auto") {
        if (mpi_ctx->rank() == 0) {
            std::cerr << "Warning: Unknown strategy '" << args.strategy << "', using AUTO\n";
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
    auto device_mgr_shared = std::shared_ptr<DeviceManager>(&dm, [](DeviceManager*){});
    auto orchestrator = std::make_shared<DeviceOrchestrator>(
        device_mgr_shared, mpi_ctx, orch_config);

    // Create model context (loads metadata but not weights yet)
    auto model_ctx = ModelContext::create(args.model_path, mpi_ctx, nullptr);
    if (!model_ctx)
    {
        if (mpi_ctx->rank() == 0)
        {
            std::cerr << "Error: Failed to load model: " << args.model_path << "\n";
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
            std::cerr << "Error: Failed to load model with placement map: " << args.model_path << "\n";
        }
        MPI_Finalize();
        return 1;
    }

    const auto &model = model_ctx->model();
    std::string architecture = model_ctx->architecture();

    if (mpi_ctx->rank() == 0)
    {
        const auto &devices = dm.devices();
        std::cout << "\n=== Llaminar v2 ===\n"
                  << "Model: " << args.model_path << "\n"
                  << "Architecture: " << architecture << "\n"
                  << "Device: " << device_idx << " (" << devices[device_idx].name << ")\n"
                  << "Strategy: " << args.strategy << "\n"
                  << "MPI ranks: " << mpi_ctx->world_size() << "\n"
                  << "\n";
    }

    // Create pipeline using factory
    auto pipeline = PipelineFactory::instance().create(architecture, model_ctx, mpi_ctx, device_idx);
    if (!pipeline)
    {
        if (mpi_ctx->rank() == 0)
        {
            std::cerr << "Error: Failed to create pipeline for architecture: " << architecture << "\n";
            std::cerr << "Supported architectures: ";
            auto supported = PipelineFactory::instance().supportedArchitectures();
            for (size_t i = 0; i < supported.size(); ++i)
            {
                std::cerr << supported[i];
                if (i + 1 < supported.size())
                    std::cerr << ", ";
            }
            std::cerr << "\n";
        }
        MPI_Finalize();
        return 1;
    }

    // TODO: Tokenize prompt (for now, use dummy tokens)
    std::vector<int> tokens = {1, 2, 3, 4, 5, 6, 7, 8}; // Placeholder

    // Run inference
    if (mpi_ctx->rank() == 0)
    {
        std::cout << "Running inference...\n";
        std::cout << "Prompt: \"" << args.prompt << "\"\n";
        std::cout << "Generating " << args.n_predict << " tokens...\n\n";
    }

    if (!pipeline->forward(tokens.data(), tokens.size()))
    {
        if (mpi_ctx->rank() == 0)
        {
            std::cerr << "Error: Forward pass failed\n";
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
        std::cout << "\nInference complete.\n";
    }

    MPI_Finalize();
    return 0;
}
