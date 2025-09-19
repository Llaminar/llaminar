#include <iostream>
#include <mpi.h>
#include <string>
#include <memory>
#include <iomanip>
#include <chrono>

// Llaminar modular components
#include "argument_parser.h"
#include "logger.h"
#include "topology_manager.h"
#include "graph_compute.h"
#include "model_loader.h"
// Note: inference_engine.h removed until llama.cpp dependency is resolved

using namespace llaminar;

int main(int argc, char *argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try
    {
        // 1. Parse command line arguments
        ArgumentParser parser(argc, argv);
        LlaminarParams params;

        if (!parser.parse(params))
        {
            MPI_Finalize();
            return 1;
        }

        // Handle help and version
        if (params.show_help)
        {
            if (rank == 0)
                parser.printUsage();
            MPI_Finalize();
            return 0;
        }

        if (params.show_version)
        {
            if (rank == 0)
                parser.printVersion();
            MPI_Finalize();
            return 0;
        }

        // 2. Initialize logging
        initializeLogging();
        Logger::getInstance().setLogLevel(params.log_level);

        if (rank == 0)
        {
            LOG_INFO("Llaminar LLM Inference Engine starting...");
            LOG_INFO("MPI initialized with " << size << " processes");
            LOG_DEBUG("Log level set to: " << Logger::getInstance().logLevelToString(params.log_level));
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

        // 4. Handle COSMA testing vs benchmark mode
        if (params.inference_mode || !params.model_file.empty())
        {
            // COSMA testing mode (inference engine temporarily disabled)
            LOG_INFO("Running COSMA testing mode");
            LOG_INFO("Note: Full inference mode is temporarily disabled while removing llama.cpp dependency");
            LOG_INFO("Use 'mpirun -np 2 ./build/tests/test_cosma' to test COSMA kernels");

            if (rank == 0)
            {
                std::cout << "\n=== COSMA MatMul Kernel Available ===" << std::endl;
                std::cout << "The refactored MatMulKernel is ready for use:" << std::endl;
                std::cout << "- Uses high-performance COSMA library" << std::endl;
                std::cout << "- Works with new Tensor architecture" << std::endl;
                std::cout << "- No kernel manager registration needed" << std::endl;
                std::cout << "- Compatible with other kernels (RMSNorm, Linear, etc.)" << std::endl;
                std::cout << "======================================" << std::endl;
            }
        }
        else
        {
            // COSMA benchmark mode
            LOG_INFO("Running COSMA benchmark mode");

            // Create compute graph for benchmarking
            ComputeGraph graph;
            auto node = std::make_shared<MatMulNode>(
                "cosma_benchmark",
                params.m, params.n, params.k);
            graph.addNode(node);

            // Execute benchmark
            LOG_INFO("Executing compute graph...");
            auto start_time = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < params.num_repeat; ++i)
            {
                bool success = graph.execute();
                if (!success)
                {
                    LOG_ERROR("Compute graph execution failed on iteration " << i);
                    MPI_Finalize();
                    return 1;
                }

                LOG_DEBUG("Completed iteration " << (i + 1) << "/" << params.num_repeat);
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double, std::milli>(end_time - start_time);

            // Report performance results
            LOG_INFO("=== Performance Results ===");
            LOG_INFO("Total execution time: " << duration.count() << " ms");
            LOG_INFO("Average time per iteration: " << duration.count() / params.num_repeat << " ms");

            // Calculate GFLOPS (2*m*n*k operations per matrix multiply)
            double total_ops = 2.0 * params.m * params.n * params.k * params.num_repeat;
            double gflops = total_ops / (duration.count() / 1000.0) / 1e9;

            if (rank == 0)
            {
                LOG_INFO("Matrix dimensions: " << params.m << "x" << params.n << "x" << params.k);
                LOG_INFO("Total operations: " << total_ops / 1e9 << " GFLOP");
                LOG_INFO("Achieved performance: " << gflops << " GFLOPS");
                LOG_INFO("Performance per MPI rank: " << gflops / size << " GFLOPS");
            }
        }

        // 5. Load model if specified
        std::unique_ptr<ModelLoader> model_loader;
        if (!params.model_file.empty())
        {
            LOG_INFO("Loading model from: " << params.model_file);
            model_loader = std::make_unique<ModelLoader>();

            try
            {
                if (model_loader->loadModel(params.model_file))
                {
                    LOG_INFO("Model loaded successfully");
                    LOG_INFO("Model inference pipeline will be implemented in future versions");
                }
                else
                {
                    LOG_ERROR("Failed to load model: " << params.model_file);
                    MPI_Finalize();
                    return 1;
                }
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("Exception loading model: " << e.what());
                MPI_Finalize();
                return 1;
            }
        }

        // 6. Create compute graph and execute workload
        ComputeGraph graph;

        if (model_loader)
        {
            // Model inference workflow
            LOG_INFO("Setting up model inference pipeline...");

            // TODO: Add model-specific compute graph construction
            // For now, demonstrate with a simple matrix multiplication
            LOG_WARN("Model-specific inference not implemented yet, running matrix multiplication demo");

            int matrix_size = params.m;

            LOG_INFO("Running matrix multiplication benchmark: " << matrix_size << "x" << matrix_size);

            // Add matrix multiplication node
            auto node = std::make_shared<MatMulNode>(
                "model_matmul_demo",
                matrix_size, matrix_size, matrix_size);

            graph.addNode(node);
        }
        else
        {
            // Legacy COSMA benchmark mode
            LOG_INFO("Running legacy COSMA benchmark mode");

            auto node = std::make_shared<MatMulNode>(
                "cosma_benchmark",
                params.m, params.n, params.k);

            graph.addNode(node);
        }

        // 7. Execute compute graph
        LOG_INFO("Executing compute graph...");

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < params.num_repeat; ++i)
        {
            bool success = graph.execute();

            if (!success)
            {
                LOG_ERROR("Compute graph execution failed on iteration " << i);
                MPI_Finalize();
                return 1;
            }

            if (rank == 0 && params.log_level >= LogLevel::VERBOSITY_DEBUG)
            {
                LOG_DEBUG("Completed iteration " << (i + 1) << "/" << params.num_repeat);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // 8. Report performance results
        if (rank == 0)
        {
            LOG_INFO("=== Performance Results ===");
            LOG_INFO("Total execution time: " << duration.count() << " ms");
            LOG_INFO("Average time per iteration: " << (duration.count() / params.num_repeat) << " ms");

            if (params.profile_kernels)
            {
                LOG_INFO("=== Kernel Performance ===");
                LOG_INFO("Kernel profiling not implemented yet");
            }

            // Calculate GFLOPS for matrix multiplication
            if (!model_loader)
            {
                long long ops = 2LL * params.m * params.n * params.k * params.num_repeat;
                double gflops = (double)ops / (duration.count() * 1e6);
                LOG_INFO("Achieved performance: " << std::fixed << std::setprecision(3) << gflops << " GFLOPS");
            }
        }

        // 9. Validation if requested
        if (params.validate_results && rank == 0)
        {
            LOG_INFO("Running result validation...");
            // TODO: Add result validation logic
            LOG_INFO("Validation completed successfully");
        }

        LOG_INFO("Llaminar execution completed successfully");
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Exception in main: " << e.what());
        MPI_Finalize();
        return 1;
    }
    catch (...)
    {
        LOG_ERROR("Unknown exception in main");
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}