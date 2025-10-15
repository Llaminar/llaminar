#include "ArgumentParser.h"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

ArgumentParser::ArgumentParser(int argc, char *argv[]) : argc_(argc), argv_(argv) {}

bool ArgumentParser::parse(LlaminarParams &params)
{
    // Count verbosity flags
    int verbosity_count = 0;

    // Parse command line arguments
    for (int i = 1; i < argc_; ++i)
    {
        std::string arg = argv_[i];

        // Model file parameter
        if ((arg == "--model" || arg == "-m") && i + 1 < argc_)
        {
            params.model_file = argv_[i + 1];
            i++; // Skip next argument as it's the value
        }
        // Verbosity levels
        else if (arg == "-v")
        {
            verbosity_count = std::max(verbosity_count, 1);
        }
        else if (arg == "-vv")
        {
            verbosity_count = std::max(verbosity_count, 2);
        }
        else if (arg == "-vvv")
        {
            verbosity_count = std::max(verbosity_count, 3);
        }
        else if (arg == "--verbose")
        {
            verbosity_count = std::max(verbosity_count, 1);
        }
        else if (arg == "--debug")
        {
            verbosity_count = std::max(verbosity_count, 2);
        }
        else if (arg == "--trace")
        {
            verbosity_count = std::max(verbosity_count, 3);
        }
        // System configuration
        else if (arg == "--enable-hyperthreading" || arg == "--ht")
        {
            params.use_hyperthreading = true;
        }
        else if (arg == "--detect-gpus")
        {
            params.detect_gpus = true;
        }
        else if (arg == "--print-topology")
        {
            params.print_topology = true;
        }
        // Legacy COSMA parameters
        else if (arg == "--matrix-size" && i + 1 < argc_)
        {
            int size = std::stoi(argv_[i + 1]);
            params.m = params.n = params.k = size;
            i++;
        }
        else if (arg == "-m" && i + 1 < argc_ && params.model_file.empty())
        {
            // If --model wasn't specified, this might be matrix size
            try
            {
                int size = std::stoi(argv_[i + 1]);
                params.m = params.n = params.k = size;
                i++;
            }
            catch (std::exception &)
            {
                // Not a number, treat as model file
                params.model_file = argv_[i + 1];
                i++;
            }
        }
        else if (arg == "--repeat" && i + 1 < argc_)
        {
            params.num_repeat = std::stoi(argv_[i + 1]);
            i++;
        }
        else if (arg == "--no-cosma")
        {
            params.use_cosma = false;
        }
        // Inference parameters
        else if (arg == "--inference" || arg == "-i")
        {
            params.inference_mode = true;
        }
        else if ((arg == "--prompt" || arg == "-p") && i + 1 < argc_)
        {
            params.prompt = argv_[i + 1];
            params.prompts.push_back(argv_[i + 1]); // Also add to batch prompts
            params.inference_mode = true;           // Auto-enable inference mode
            i++;
        }
        else if (arg == "--batch-size" && i + 1 < argc_)
        {
            params.batch_size = std::atoi(argv_[i + 1]);
            if (params.batch_size < 1)
            {
                std::cerr << "Error: batch size must be >= 1" << std::endl;
                return false;
            }
            i++;
        }
        else if (arg == "--eval")
        {
            params.eval_only = true;
            params.inference_mode = true; // Auto-enable inference mode
        }
        else if (arg == "--ctx-size" && i + 1 < argc_)
        {
            params.ctx_size = std::stoi(argv_[i + 1]);
            i++;
        }
        else if ((arg == "--predict" || arg == "-n") && i + 1 < argc_)
        {
            params.n_predict = std::stoi(argv_[i + 1]);
            i++;
        }
        else if (arg == "--temperature" && i + 1 < argc_)
        {
            params.temperature = std::stof(argv_[i + 1]);
            i++;
        }
        else if (arg == "--top-k" && i + 1 < argc_)
        {
            params.top_k = std::stoi(argv_[i + 1]);
            i++;
        }
        else if (arg == "--top-p" && i + 1 < argc_)
        {
            params.top_p = std::stof(argv_[i + 1]);
            i++;
        }
        else if (arg == "--interactive")
        {
            params.interactive = true;
            params.inference_mode = true; // Auto-enable inference mode
        }
        // Chat-specific parameters
        else if (arg == "--system" && i + 1 < argc_)
        {
            params.system_prompt = argv_[i + 1];
            i++;
        }
        else if (arg == "--chat-template" && i + 1 < argc_)
        {
            params.chat_template = argv_[i + 1];
            i++;
        }
        else if (arg == "--save-chat")
        {
            params.save_conversation = true;
        }
        else if (arg == "--no-stream")
        {
            params.streaming_output = false;
        }
        else if (arg == "--stream")
        {
            params.streaming_output = true;
        }
        else if (arg == "--load-conversation" && i + 1 < argc_)
        {
            params.conversation_file = argv_[i + 1];
            i++;
        }
        // Performance options
        else if (arg == "--profile")
        {
            params.profile_kernels = true;
        }
        else if (arg == "--validate")
        {
            params.validate_results = true;
        }
        else if (arg == "--kv-stats")
        {
            params.kv_cache_stats = true;
        }
        // Output options
        else if (arg == "--output-json" && i + 1 < argc_)
        {
            params.output_json_file = argv_[i + 1];
            i++;
        }
        // Help and version
        else if (arg == "--benchmark")
        {
            params.benchmark_mode = true;
            params.inference_mode = true; // benchmark implies inference

            // Set default benchmark parameters if not already specified
            // Default: 512-token prefill, 128-token decode
            if (params.prompt.empty())
            {
                // Generate a prompt that will tokenize to ~512 tokens
                // Using repetitive text to ensure consistent tokenization
                params.prompt = "The quick brown fox jumps over the lazy dog. "
                                "This is a benchmark test to measure inference performance. "
                                "We need to generate approximately 512 tokens for the prefill phase "
                                "to properly stress test the model's throughput and latency characteristics. ";

                // Repeat the base text to reach ~512 tokens (each repetition is ~20-25 tokens)
                std::string base_prompt = params.prompt;
                for (int i = 0; i < 18; ++i) // 20 repetitions total = ~400-500 tokens
                {
                    params.prompt += base_prompt;
                }
            }

            // n_predict already defaults to 128 in LlaminarParams, which is correct for decode phase
        }
        else if (arg == "--help" || arg == "-h")
        {
            params.show_help = true;
            return true;
        }
        else if (arg == "--version")
        {
            params.show_version = true;
            return true;
        }
        else if (arg.substr(0, 2) == "--" || (arg.substr(0, 1) == "-" && arg.length() > 1))
        {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage();
            return false;
        }
    }

    // Set log level based on verbosity count
    switch (verbosity_count)
    {
    case 0:
        params.log_level = LogLevel::WARN;
        break;
    case 1:
        params.log_level = LogLevel::INFO;
        break;
    case 2:
        params.log_level = LogLevel::VERBOSITY_DEBUG;
        break;
    case 3:
    default:
        params.log_level = LogLevel::TRACE;
        break;
    }

    return true;
}

void ArgumentParser::printUsage() const
{
    std::cout << "\nLlaminar - High-Performance LLM Inference Engine\n"
              << std::endl;
    std::cout << "Usage: " << argv_[0] << " [options]\n"
              << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -m, --model <file>       Path to GGUF model file" << std::endl;
    std::cout << "  -v                       Verbose (INFO level)" << std::endl;
    std::cout << "  -vv                      More verbose (DEBUG level)" << std::endl;
    std::cout << "  -vvv                     Very verbose (TRACE level)" << std::endl;
    std::cout << "  --verbose                Same as -v" << std::endl;
    std::cout << "  --debug                  Same as -vv" << std::endl;
    std::cout << "  --trace                  Same as -vvv" << std::endl;
    std::cout << "\nInference Mode:" << std::endl;
    std::cout << "  -i, --inference          Enable inference mode" << std::endl;
    std::cout << "  -p, --prompt <text>      Text prompt to process (can be specified multiple times for batching)" << std::endl;
    std::cout << "  --batch-size <n>         Batch size for parallel sequence processing (default: 1)" << std::endl;
    std::cout << "  --eval                   Evaluate prompt and exit (no generation)" << std::endl;
    std::cout << "  --interactive            Interactive chat mode" << std::endl;
    std::cout << "  --ctx-size <size>        Context window size (default: 2048)" << std::endl;
    std::cout << "  -n, --predict <tokens>   Max tokens to generate (default: 128)" << std::endl;
    std::cout << "  --temperature <float>    Sampling temperature (default: 0.7)" << std::endl;
    std::cout << "  --top-k <int>            Top-K sampling (default: 40)" << std::endl;
    std::cout << "  --top-p <float>          Top-P sampling (default: 0.9)" << std::endl;
    std::cout << "\nChat Options:" << std::endl;
    std::cout << "  --system <text>          Set system prompt for chat" << std::endl;
    std::cout << "  --chat-template <name>   Chat template to use" << std::endl;
    std::cout << "  --save-chat              Save conversation to file" << std::endl;
    std::cout << "  --load-conversation <f>  Load previous conversation" << std::endl;
    std::cout << "  --stream                 Enable streaming output (default)" << std::endl;
    std::cout << "  --no-stream              Disable streaming output" << std::endl;
    std::cout << "\nBenchmark Mode:" << std::endl;
    std::cout << "  --benchmark              Run inference benchmark with performance metrics" << std::endl;
    std::cout << "                           (minimal logging, shows prefill/decode tok/s)" << std::endl;
    std::cout << "\nSystem Configuration:" << std::endl;
    std::cout << "  --enable-hyperthreading  Use hyperthreaded cores (default: physical cores only)" << std::endl;
    std::cout << "  --ht                     Short form of --enable-hyperthreading" << std::endl;
    std::cout << "  --detect-gpus            Enable GPU device detection" << std::endl;
    std::cout << "  --print-topology         Print system topology information" << std::endl;
    std::cout << "\nLegacy COSMA Parameters:" << std::endl;
    std::cout << "  --matrix-size <size>     Matrix size for benchmarks (default: 1024)" << std::endl;
    std::cout << "  --repeat <num>           Number of benchmark iterations (default: 10)" << std::endl;
    std::cout << "  --no-cosma               Disable COSMA optimizations" << std::endl;
    std::cout << "\nPerformance Options:" << std::endl;
    std::cout << "  --profile                Enable kernel profiling" << std::endl;
    std::cout << "  --validate               Enable result validation" << std::endl;
    std::cout << "  --kv-stats               Print KV cache capacity/usage summary at exit" << std::endl;
    std::cout << "\nOutput Options:" << std::endl;
    std::cout << "  --output-json <file>     Write logits and tokens to JSON file" << std::endl;
    std::cout << "\nInformation:" << std::endl;
    std::cout << "  --help, -h               Show this help message" << std::endl;
    std::cout << "  --version                Show version information" << std::endl;
    std::cout << "\nLog Levels:" << std::endl;
    std::cout << "  ERROR (0) - Only critical errors" << std::endl;
    std::cout << "  WARN  (1) - Warnings and errors (default)" << std::endl;
    std::cout << "  INFO  (2) - General information (-v)" << std::endl;
    std::cout << "  DEBUG (3) - Debug information (-vv)" << std::endl;
    std::cout << "  TRACE (4) - All debug information (-vvv)" << std::endl;
    std::cout << std::endl;
}

void ArgumentParser::printVersion() const
{
    std::cout << "Llaminar LLM Inference Engine v0.1.0" << std::endl;
    std::cout << "Built with COSMA matrix multiplication support" << std::endl;
    std::cout << "MPI distributed computing enabled" << std::endl;
    std::cout << "GGUF model format support" << std::endl;
}

bool ArgumentParser::hasArgument(const std::string &arg) const
{
    for (int i = 1; i < argc_; ++i)
    {
        if (std::string(argv_[i]) == arg)
        {
            return true;
        }
    }
    return false;
}

std::string ArgumentParser::getArgumentValue(const std::string &arg) const
{
    for (int i = 1; i < argc_ - 1; ++i)
    {
        if (std::string(argv_[i]) == arg)
        {
            return std::string(argv_[i + 1]);
        }
    }
    return "";
}

bool ArgumentParser::getBoolArgument(const std::string &arg, const std::string &short_arg) const
{
    if (hasArgument(arg))
        return true;
    if (!short_arg.empty() && hasArgument(short_arg))
        return true;
    return false;
}

LogLevel ArgumentParser::parseLogLevel(const std::string &verbosity_args) const
{
    // This is handled in the main parse function
    return LogLevel::INFO;
}

std::string ArgumentParser::logLevelToString(LogLevel level) const
{
    switch (level)
    {
    case LogLevel::ERROR:
        return "ERROR";
    case LogLevel::WARN:
        return "WARN";
    case LogLevel::INFO:
        return "INFO";
    case LogLevel::VERBOSITY_DEBUG:
        return "DEBUG";
    case LogLevel::TRACE:
        return "TRACE";
    default:
        return "INFO";
    }
}