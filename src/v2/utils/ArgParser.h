/**
 * @file ArgParser.h
 * @brief Command-line argument parser for Llaminar v2
 * @author David Sanftenberg
 * @date 2025-10-24
 */

#pragma once

#include <string>
#include <vector>
#include <optional>

namespace llaminar2
{

    /**
     * @brief Parsed command-line arguments context
     *
     * This structure contains all parsed CLI arguments in a structured format
     * that can be consumed by various components (DeviceOrchestrator, ModelContext, etc.)
     */
    struct ArgContext
    {
        // Model configuration
        std::string model_path;

        // Inference parameters
        std::string prompt = "Hello, my name is";
        int n_predict = 128;
        int max_seq_len = 2048; // Maximum sequence length for KV cache and activations
        float temperature = 0.8f;
        int top_k = 40;
        float top_p = 0.9f;
        int seed = -1;               // -1 = random        // Device configuration
        std::string device = "auto"; // "auto", "cpu", "cuda:N", "rocm:N"

        // Placement strategy
        std::string strategy = "auto"; // "auto", "all-gpu", "all-cpu", "layer-split", "memory-aware", "moe-optimized", "custom"
        int offload_layers = 0;        // For layer-split strategy
        std::string device_map;        // For custom strategy (e.g., "0-11:gpu:0,12-23:cpu")

        // Memory constraints
        std::optional<size_t> max_gpu_memory_mb; // Max GPU memory to use (MB)
        std::optional<size_t> max_cpu_memory_mb; // Max CPU memory to use (MB)

        // MoE-specific
        bool moe_shared_experts_gpu = true; // Shared experts on GPU (for moe-optimized)
        bool moe_sparse_experts_cpu = true; // Sparse experts on CPU (for moe-optimized)

        // Multi-GPU
        bool multi_gpu = false; // Enable multi-GPU mode
        std::string gpu_split;  // GPU split strategy: "even", "weighted", or custom ratios

        // Debugging/logging
        bool verbose = false;  // Deprecated: use verbose_level instead
        int verbose_level = 0; // 0 = default (INFO), 1 = DEBUG (-v), 2 = TRACE (-vv)
        bool list_devices = false;
        bool show_help = false;

        // Performance
        int batch_size = 1;
        bool use_mmap = true;
        int n_threads = -1; // -1 = auto-detect
    };

    /**
     * @brief Command-line argument parser
     *
     * Parses argc/argv into a structured ArgContext that can be consumed
     * by DeviceOrchestrator, ModelContext, and other components.
     *
     * Usage:
     *   auto arg_ctx = ArgParser::parse(argc, argv);
     *   if (arg_ctx.show_help) { ArgParser::printUsage(argv[0]); return 0; }
     *   if (arg_ctx.list_devices) { listDevices(); return 0; }
     *   // ... use arg_ctx to configure components
     */
    class ArgParser
    {
    public:
        /**
         * @brief Parse command-line arguments
         * @param argc Argument count
         * @param argv Argument vector
         * @return Parsed argument context
         */
        static ArgContext parse(int argc, char *argv[]);

        /**
         * @brief Print usage information
         * @param prog_name Program name (argv[0])
         */
        static void printUsage(const char *prog_name);

    private:
        /**
         * @brief Check if argument matches a flag (short or long form)
         * @param arg Argument string
         * @param short_flag Short flag (e.g., "-m")
         * @param long_flag Long flag (e.g., "--model")
         * @return true if matches
         */
        static bool matchesFlag(const std::string &arg,
                                const std::string &short_flag,
                                const std::string &long_flag);

        /**
         * @brief Get next argument value (and increment index)
         * @param argv Argument vector
         * @param argc Argument count
         * @param i Current index (will be incremented)
         * @param flag_name Flag name for error reporting
         * @return Argument value, or empty string if missing
         */
        static std::string getNextArg(char *argv[], int argc, int &i, const std::string &flag_name);
    };

} // namespace llaminar2
