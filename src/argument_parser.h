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
    int32_t n_ctx = 2048;                // --ctx-size
    int32_t n_predict = 128;             // --predict
    float temperature = 0.7f;            // --temperature
    int32_t top_k = 40;                  // --top-k
    float top_p = 0.9f;                  // --top-p
    bool interactive = false;            // --interactive

    // System configuration
    bool use_hyperthreading = false;
    bool detect_gpus = false;
    bool print_topology = false;

    // MPI parameters
    bool use_mpi = true;

    // Performance parameters
    bool profile_kernels = false;
    bool validate_results = false;

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