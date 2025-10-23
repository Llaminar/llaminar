/**
 * @file Main.cpp
 * @brief Llaminar v2 entry point
 *
 * Clean greenfield implementation with:
 * - Device manager initialization
 * - Multi-GPU heterogeneous support
 * - Direct kernel orchestration
 *
 * @author David Sanftenberg
 */

#include "utils/MPIContext.h"
#include "backends/ComputeBackend.h"
#include "pipelines/QwenPipeline.h"
#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>

using namespace llaminar2;

void print_usage(const char *prog_name)
{
    std::cout << "Usage: " << prog_name << " [options]\n"
              << "\n"
              << "Options:\n"
              << "  -m, --model PATH        Model file path (GGUF format)\n"
              << "  -p, --prompt TEXT       Prompt text\n"
              << "  -n, --n-predict N       Number of tokens to generate (default: 128)\n"
              << "  --device DEVICE         Device to use (auto, cpu, cuda:N, rocm:N)\n"
              << "  --list-devices          List available devices and exit\n"
              << "  -h, --help              Show this help\n"
              << "\n"
              << "Examples:\n"
              << "  " << prog_name << " -m model.gguf -p \"Hello\" -n 50\n"
              << "  " << prog_name << " --device cuda:0 -m model.gguf\n"
              << "  " << prog_name << " --list-devices\n";
}

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

    // Parse arguments
    std::string model_path;
    std::string prompt = "Hello, my name is";
    int n_predict = 128;
    std::string device_str = "auto";
    bool list_devices_only = false;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help")
        {
            print_usage(argv[0]);
            MPI_Finalize();
            return 0;
        }
        else if (arg == "--list-devices")
        {
            list_devices_only = true;
        }
        else if (arg == "-m" || arg == "--model")
        {
            if (i + 1 < argc)
                model_path = argv[++i];
        }
        else if (arg == "-p" || arg == "--prompt")
        {
            if (i + 1 < argc)
                prompt = argv[++i];
        }
        else if (arg == "-n" || arg == "--n-predict")
        {
            if (i + 1 < argc)
                n_predict = std::stoi(argv[++i]);
        }
        else if (arg == "--device")
        {
            if (i + 1 < argc)
                device_str = argv[++i];
        }
    }

    // Initialize device manager
    auto &dm = DeviceManager::instance();
    dm.initialize();

    // List devices and exit if requested
    if (list_devices_only)
    {
        list_devices();
        MPI_Finalize();
        return 0;
    }

    // Validate required arguments
    if (model_path.empty())
    {
        std::cerr << "Error: Model path required (-m)\n\n";
        print_usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    // Parse device
    int device_idx = parse_device(device_str, dm);
    if (device_idx < 0)
    {
        MPI_Finalize();
        return 1;
    }

    // Get MPI context
    auto mpi_ctx = MPIContextFactory::global();

    if (mpi_ctx->rank() == 0)
    {
        const auto &devices = dm.devices();
        std::cout << "\n=== Llaminar v2 ===\n"
                  << "Model: " << model_path << "\n"
                  << "Device: " << device_idx << " (" << devices[device_idx].name << ")\n"
                  << "MPI ranks: " << mpi_ctx->world_size() << "\n"
                  << "\n";
    }

    // Create pipeline
    auto pipeline = std::make_unique<QwenPipeline>(model_path, mpi_ctx, device_idx);

    // TODO: Tokenize prompt (for now, use dummy tokens)
    std::vector<int> tokens = {1, 2, 3, 4, 5, 6, 7, 8}; // Placeholder

    // Run inference
    if (mpi_ctx->rank() == 0)
    {
        std::cout << "Running inference...\n";
    }

    if (!pipeline->forward(tokens.data(), tokens.size()))
    {
        std::cerr << "Error: Forward pass failed\n";
        MPI_Finalize();
        return 1;
    }

    // Generate tokens
    for (int i = 0; i < n_predict; ++i)
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
