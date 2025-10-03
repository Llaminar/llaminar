#pragma once

#include "common.h"
#include "log_level.h"
#include <string>

// Extended parameters structure for LLM inference engine
struct LlaminarParams
{
    // Matrix computation parameters (legacy COSMA)
    int m = 1024;
    int n = 1024;
    int k = 1024;
    int num_repeat = 10;
    bool use_cosma = true;

    // LLM inference parameters
    std::string model_file = "";         // -m, --model
    LogLevel log_level = LogLevel::INFO; // -v, -vv, -vvv
    bool inference_mode = false;         // --inference, -i
    std::string prompt = "";             // -p, --prompt
    bool eval_only = false;              // --eval (prompt evaluation only)
    int32_t ctx_size = 2048;             // --ctx-size (renamed from n_ctx)
    int32_t n_predict = 128;             // --predict
    float temperature = 0.7f;            // --temperature
    int32_t top_k = 40;                  // --top-k
    float top_p = 0.9f;                  // --top-p
    int32_t max_tokens = 128;            // --predict (alias)
    bool interactive = false;            // --interactive

    // Chat-specific parameters
    std::string system_prompt = "";     // --system
    std::string chat_template = "";     // --chat-template
    bool save_conversation = false;     // --save-chat
    bool streaming_output = true;       // --stream / --no-stream
    std::string conversation_file = ""; // --load-conversation

    // System configuration
    bool use_hyperthreading = false;
    bool detect_gpus = false;
    bool print_topology = false;

    // MPI parameters
    bool use_mpi = true;

    // Performance parameters
    bool profile_kernels = false;
    bool validate_results = false;
    bool kv_cache_stats = false; // --kv-stats : print KV cache capacity/usage summary at end

    // Help/version
    bool show_help = false;
    bool show_version = false;
};

// Argument parser class for command line processing
class ArgumentParser
{
public:
    ArgumentParser(int argc, char *argv[]);

    // Parse command line arguments into params structure
    bool parse(LlaminarParams &params);

    // Print usage information
    void printUsage() const;

    // Print version information
    void printVersion() const;

private:
    int argc_;
    char **argv_;

    // Helper functions
    bool hasArgument(const std::string &arg) const;
    std::string getArgumentValue(const std::string &arg) const;
    bool getBoolArgument(const std::string &arg, const std::string &short_arg = "") const;
    LogLevel parseLogLevel(const std::string &verbosity_args) const;
    std::string logLevelToString(LogLevel level) const;
};