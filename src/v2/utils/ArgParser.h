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
#include <functional>
#include <initializer_list>

namespace llaminar2
{

    // ========================================================================
    // Argument Definition Schema
    // ========================================================================

    /**
     * @brief Defines a CLI argument with its validation rules
     *
     * Used to systematically validate all CLI arguments against their
     * allowed values, types, and constraints.
     */
    struct ArgDef
    {
        std::string name;                      // Primary flag name (e.g., "--activation-precision")
        std::vector<std::string> aliases;      // Alternative names (e.g., {"--act-prec", "--activation-prec"})
        std::vector<std::string> valid_values; // For enum types: list of valid string values
        std::string default_value;             // Default value (for documentation)
        bool allow_empty = false;              // Whether empty string is valid
        bool is_prefix_match = false;          // For args like --device=cuda:0 where value has prefix

        // Convenience constructors
        ArgDef(const std::string &n,
               std::initializer_list<std::string> aliases_init,
               std::initializer_list<std::string> valid_init,
               const std::string &def = "",
               bool empty_ok = false,
               bool prefix = false)
            : name(n), aliases(aliases_init), valid_values(valid_init),
              default_value(def), allow_empty(empty_ok), is_prefix_match(prefix) {}
    };

    /**
     * @brief Registry of all CLI arguments with validation rules
     *
     * This is the single source of truth for argument validation.
     * Add new arguments here to automatically get validation.
     */
    class ArgRegistry
    {
    public:
        static const std::vector<ArgDef> &getDefinitions();

        /**
         * @brief Validate a single argument value against its definition
         * @param arg_name The argument name (e.g., "--activation-precision")
         * @param value The value to validate
         * @param error_out Output parameter for error message if validation fails
         * @return true if valid, false otherwise
         */
        static bool validateArg(const std::string &arg_name,
                                const std::string &value,
                                std::string &error_out);

        /**
         * @brief Get the ArgDef for a given argument name or alias
         * @return pointer to ArgDef if found, nullptr otherwise
         */
        static const ArgDef *findDef(const std::string &arg_name);
    };

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
        int n_predict = -1;     // -1 = unlimited (until EOS or context full)
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

        // Multi-GPU (Phase 6)
        bool multi_gpu = false;       // Enable multi-GPU mode
        std::string gpu_split;        // GPU split strategy: "even", "weighted", or custom ratios
        std::vector<int> gpu_devices; // Specific GPU device indices to use (empty = all)

        // Debugging/logging
        bool verbose = false;  // Deprecated: use verbose_level instead
        int verbose_level = 0; // 0 = default (INFO), 1 = DEBUG (-v), 2 = TRACE (-vv)
        bool list_devices = false;
        bool show_help = false;

        // Performance
        int batch_size = 1;
        bool use_mmap = true;
        int n_threads = -1; // -1 = auto-detect

        // Weight loading precision
        std::string weight_precision = "native"; // "native", "fp32", "bf16", "fp16", "int8"

        // Activation/accumulation precision
        std::string activation_precision = "fp32"; // "fp32", "bf16", "fp16", "q8_1"

        // Weight sharding for tensor parallelism
        bool shard_weights = false;           // Explicitly enable weight sharding (legacy)
        bool disable_weight_sharding = false; // Explicitly disable weight sharding (default is auto: enabled when world_size > 1)

        // Chat mode
        bool chat_mode = false;         // Enable interactive chat (FTXUI UI)
        bool single_shot_chat = false;  // Single prompt with chat template formatting
        std::string system_prompt = ""; // System message for chat
        std::string chat_template = ""; // Override template: "chatml", "llama3", "mistral", etc.

        // Benchmark mode
        bool benchmark_mode = false; // Run benchmark (prefill + decode timing)

        // Deterministic mode (for reproducible outputs and comparison testing)
        bool deterministic = false; // Forces temperature=0, seed=42, top_p=1.0, top_k=0

        // Fused attention kernel
        bool use_fused_attention = false;        // Use fused attention+Wo kernel
        std::string fused_attention_backend_str; // Backend: "jit" (default), "reference", "tiled"

        // MPI Bootstrap Options
        int mpi_procs = 0;              // Number of MPI processes (0 = auto: one per socket)
        std::string hostfile = "";      // Path to hostfile for multi-machine MPI
        bool mpi_dry_run = false;       // Print MPI configuration and exit (without launching)
        bool mpi_verbose = false;       // Verbose MPI output (report bindings)
        bool mpi_no_bootstrap = false;  // Disable auto-bootstrap (assume already under MPI)
        bool mpi_oversubscribe = false; // Allow more MPI ranks than available slots

        // Validation errors (set during parse if invalid arguments detected)
        std::string error; // Non-empty if parse failed - caller should check and exit
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
