/**
 * @file ArgParser.cpp
 * @brief Command-line argument parser implementation
 * @author David Sanftenberg
 * @date 2025-10-24
 */

#include "ArgParser.h"
#include "Logger.h"
#include <iostream>
#include <cstring>
#include <cstdlib>

namespace llaminar2
{

    ArgContext ArgParser::parse(int argc, char *argv[])
    {
        ArgContext ctx;

        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];

            // Help
            if (matchesFlag(arg, "-h", "--help"))
            {
                ctx.show_help = true;
                return ctx; // Early return, ignore other args
            }

            // List devices
            else if (arg == "--list-devices")
            {
                ctx.list_devices = true;
                return ctx; // Early return
            }

            // Model path
            else if (matchesFlag(arg, "-m", "--model"))
            {
                ctx.model_path = getNextArg(argv, argc, i, "model");
            }

            // Prompt
            else if (matchesFlag(arg, "-p", "--prompt"))
            {
                ctx.prompt = getNextArg(argv, argc, i, "prompt");
            }

            // Number of tokens to generate
            else if (matchesFlag(arg, "-n", "--n-predict"))
            {
                std::string val = getNextArg(argv, argc, i, "n-predict");
                if (!val.empty())
                    ctx.n_predict = std::stoi(val);
            }

            // Maximum sequence length
            else if (matchesFlag(arg, "-c", "--ctx-size"))
            {
                std::string val = getNextArg(argv, argc, i, "ctx-size");
                if (!val.empty())
                    ctx.max_seq_len = std::stoi(val);
            }

            // Temperature
            else if (matchesFlag(arg, "-t", "--temperature"))
            {
                std::string val = getNextArg(argv, argc, i, "temperature");
                if (!val.empty())
                    ctx.temperature = std::stof(val);
            }

            // Top-k
            else if (arg == "--top-k")
            {
                std::string val = getNextArg(argv, argc, i, "top-k");
                if (!val.empty())
                    ctx.top_k = std::stoi(val);
            }

            // Top-p
            else if (arg == "--top-p")
            {
                std::string val = getNextArg(argv, argc, i, "top-p");
                if (!val.empty())
                    ctx.top_p = std::stof(val);
            }

            // Seed
            else if (matchesFlag(arg, "-s", "--seed"))
            {
                std::string val = getNextArg(argv, argc, i, "seed");
                if (!val.empty())
                    ctx.seed = std::stoi(val);
            }

            // Device
            else if (arg == "--device")
            {
                ctx.device = getNextArg(argv, argc, i, "device");
            }

            // Placement strategy
            else if (arg == "--strategy")
            {
                ctx.strategy = getNextArg(argv, argc, i, "strategy");
            }

            // Offload layers
            else if (arg == "--offload-layers")
            {
                std::string val = getNextArg(argv, argc, i, "offload-layers");
                if (!val.empty())
                    ctx.offload_layers = std::stoi(val);
            }

            // Device map (custom strategy)
            else if (arg == "--device-map")
            {
                ctx.device_map = getNextArg(argv, argc, i, "device-map");
                if (!ctx.device_map.empty())
                {
                    ctx.strategy = "custom"; // Automatically set strategy
                }
            }

            // Memory constraints
            else if (arg == "--max-gpu-memory")
            {
                std::string val = getNextArg(argv, argc, i, "max-gpu-memory");
                if (!val.empty())
                    ctx.max_gpu_memory_mb = std::stoull(val);
            }

            else if (arg == "--max-cpu-memory")
            {
                std::string val = getNextArg(argv, argc, i, "max-cpu-memory");
                if (!val.empty())
                    ctx.max_cpu_memory_mb = std::stoull(val);
            }

            // MoE-specific
            else if (arg == "--moe-shared-gpu")
            {
                ctx.moe_shared_experts_gpu = true;
            }

            else if (arg == "--moe-shared-cpu")
            {
                ctx.moe_shared_experts_gpu = false;
            }

            else if (arg == "--moe-sparse-gpu")
            {
                ctx.moe_sparse_experts_cpu = false;
            }

            else if (arg == "--moe-sparse-cpu")
            {
                ctx.moe_sparse_experts_cpu = true;
            }

            // Multi-GPU
            else if (arg == "--multi-gpu")
            {
                ctx.multi_gpu = true;
            }

            else if (arg == "--gpu-split")
            {
                ctx.gpu_split = getNextArg(argv, argc, i, "gpu-split");
                ctx.multi_gpu = true; // Automatically enable
            }

            // Batch size
            else if (matchesFlag(arg, "-b", "--batch-size"))
            {
                std::string val = getNextArg(argv, argc, i, "batch-size");
                if (!val.empty())
                    ctx.batch_size = std::stoi(val);
            }

            // Threads
            else if (arg == "--threads")
            {
                std::string val = getNextArg(argv, argc, i, "threads");
                if (!val.empty())
                    ctx.n_threads = std::stoi(val);
            }

            // Weight loading precision
            else if (arg == "--weight-precision" || arg == "--weight-prec")
            {
                ctx.weight_precision = getNextArg(argv, argc, i, "weight-precision");
            }

            // Activation/accumulation precision
            else if (arg == "--activation-precision" || arg == "--activation-prec" || arg == "--act-prec")
            {
                ctx.activation_precision = getNextArg(argv, argc, i, "activation-precision");
            }

            // Weight sharding for tensor parallelism
            else if (arg == "--shard-weights")
            {
                ctx.shard_weights = true;
            }
            else if (arg == "--no-shard-weights")
            {
                ctx.disable_weight_sharding = true;
            }

            // Memory mapping
            else if (arg == "--no-mmap")
            {
                ctx.use_mmap = false;
            }

            // Chat mode
            else if (arg == "--chat")
            {
                ctx.chat_mode = true;
            }

            else if (arg == "--chat-single")
            {
                ctx.single_shot_chat = true;
            }

            else if (arg == "--system")
            {
                ctx.system_prompt = getNextArg(argv, argc, i, "system");
            }

            else if (arg == "--chat-template")
            {
                ctx.chat_template = getNextArg(argv, argc, i, "chat-template");
            }

            // Benchmark mode
            else if (arg == "--benchmark")
            {
                ctx.benchmark_mode = true;
            }

            // Fused attention kernel
            else if (arg == "--fused-attention")
            {
                ctx.use_fused_attention = true;
            }
            // Fused attention backend selection
            else if (arg.rfind("--fused-attention-backend=", 0) == 0)
            {
                ctx.fused_attention_backend_str = arg.substr(26);
                ctx.use_fused_attention = true; // Implicitly enable fused attention
            }

            // Verbose logging levels
            else if (arg == "-vv" || arg == "--vverbose")
            {
                ctx.verbose_level = 2; // TRACE
                ctx.verbose = true;    // Backward compat
            }
            else if (matchesFlag(arg, "-v", "--verbose"))
            {
                ctx.verbose_level = 1; // DEBUG
                ctx.verbose = true;    // Backward compat
            }

            else
            {
                std::cerr << "Warning: Unknown argument '" << arg << "'" << std::endl;
            }
        }

        // Apply verbose level to Logger (if specified via CLI)
        // This overrides environment variable LLAMINAR_LOG_LEVEL
        if (ctx.verbose_level == 2)
        {
            Logger::getInstance().setLogLevel(LogLevel::TRACE);
        }
        else if (ctx.verbose_level == 1)
        {
            Logger::getInstance().setLogLevel(LogLevel::VERBOSITY_DEBUG);
        }
        // else: keep default or environment-configured level

        return ctx;
    }

    void ArgParser::printUsage(const char *prog_name)
    {
        std::cout << "Llaminar v2 - High-performance LLM inference\n\n";
        std::cout << "Usage: " << prog_name << " [options]\n\n";

        std::cout << "Required:\n";
        std::cout << "  -m, --model PATH          Model file (GGUF format)\n\n";

        std::cout << "Inference:\n";
        std::cout << "  -p, --prompt TEXT         Prompt text (default: \"Hello, my name is\")\n";
        std::cout << "  -n, --n-predict N         Tokens to generate (default: 128)\n";
        std::cout << "  -c, --ctx-size N          Maximum context size (default: 2048)\n";
        std::cout << "  -t, --temperature T       Sampling temperature (default: 0.8)\n";
        std::cout << "  --top-k K                 Top-k sampling (default: 40)\n";
        std::cout << "  --top-p P                 Top-p sampling (default: 0.9)\n";
        std::cout << "  -s, --seed N              Random seed (-1 = random)\n";
        std::cout << "  -b, --batch-size N        Batch size (default: 1)\n\n";

        std::cout << "Device Placement:\n";
        std::cout << "  --device DEVICE           Device: auto, cpu, cuda:N, rocm:N (default: auto)\n";
        std::cout << "  --strategy STRATEGY       Placement strategy:\n";
        std::cout << "                              auto         - Auto-detect best strategy\n";
        std::cout << "                              all-gpu      - All weights on GPU\n";
        std::cout << "                              all-cpu      - All weights on CPU\n";
        std::cout << "                              layer-split  - First N layers on GPU\n";
        std::cout << "                              memory-aware - Fit within memory budget\n";
        std::cout << "                              moe-optimized - MoE-aware placement\n";
        std::cout << "                              custom       - Use --device-map\n";
        std::cout << "  --offload-layers N        Layers on GPU (layer-split strategy)\n";
        std::cout << "  --device-map MAP          Custom device map (e.g., \"0-11:gpu:0,12-23:cpu\")\n\n";

        std::cout << "Memory Management:\n";
        std::cout << "  --max-gpu-memory MB       Max GPU memory (MB)\n";
        std::cout << "  --max-cpu-memory MB       Max CPU memory (MB)\n";
        std::cout << "  --no-mmap                 Disable memory mapping\n\n";

        std::cout << "MoE Models:\n";
        std::cout << "  --moe-shared-gpu          Shared experts on GPU (default)\n";
        std::cout << "  --moe-shared-cpu          Shared experts on CPU\n";
        std::cout << "  --moe-sparse-gpu          Sparse experts on GPU\n";
        std::cout << "  --moe-sparse-cpu          Sparse experts on CPU (default)\n\n";

        std::cout << "Multi-GPU:\n";
        std::cout << "  --multi-gpu               Enable multi-GPU mode\n";
        std::cout << "  --gpu-split STRATEGY      GPU split: even, weighted, or custom ratios\n\n";

        std::cout << "Performance:\n";
        std::cout << "  --threads N               Thread count (-1 = auto)\n";
        std::cout << "  --weight-precision MODE   How weights are loaded (default: native)\n";
        std::cout << "  --weight-prec MODE          native - Keep in GGUF format (memory-efficient)\n";
        std::cout << "                              fp32   - Dequantize to FP32 at load\n";
        std::cout << "                              bf16   - Dequantize to BF16 at load\n";
        std::cout << "                              fp16   - Dequantize to FP16 at load\n";
        std::cout << "                              int8   - Dequantize to INT8 at load\n";
        std::cout << "  --activation-precision M  Precision for activations (default: fp32)\n";
        std::cout << "  --activation-prec M         fp32  - 32-bit float (highest accuracy)\n";
        std::cout << "  --act-prec M                bf16  - bfloat16 (Intel AMX, 2× faster)\n";
        std::cout << "                              fp16  - 16-bit float (ARM/GPU)\n";
        std::cout << "                              q8_1  - 8-bit quantized with per-block scaling\n\n";

        std::cout << "Tensor Parallelism:\n";
        std::cout << "  --no-shard-weights        Disable weight sharding (use replicated weights)\n";
        std::cout << "                              By default, sharding is enabled when MPI ranks > 1\n";
        std::cout << "                              Sharding reduces memory per rank by ~25-40%%\n\n";

        std::cout << "Chat Mode:\n";
        std::cout << "  --chat                    Interactive chat mode (FTXUI terminal UI)\n";
        std::cout << "  --chat-single             Single prompt with chat template formatting\n";
        std::cout << "  --system TEXT             System prompt for chat context\n";
        std::cout << "  --chat-template TEMPLATE  Override chat template:\n";
        std::cout << "                              chatml, llama3, llama2, mistral, phi3, gemma, deepseek, etc.\n\n";

        std::cout << "Benchmark:\n";
        std::cout << "  --benchmark               Run benchmark mode (separate prefill/decode timing)\n";
        std::cout << "                            Uses greedy sampling for deterministic results\n";
        std::cout << "                            Default: 128 tokens if -n not specified\n\n";

        std::cout << "Fused Attention:\n";
        std::cout << "  --fused-attention         Enable fused attention+Wo kernel (requires Q8_1 activations)\n";
        std::cout << "  --fused-attention-backend=B  Select backend: jit (default), reference, tiled\n";
        std::cout << "                              jit       - AVX-512 VNNI optimized (fastest)\n";
        std::cout << "                              reference - Pure C++ (for debugging)\n";
        std::cout << "                              tiled     - Cache-blocked (balanced)\n\n";

        std::cout << "Other:\n";
        std::cout << "  --list-devices            List available devices and exit\n";
        std::cout << "  -v, --verbose             Verbose logging (DEBUG level)\n";
        std::cout << "  -vv, --vverbose           Very verbose logging (TRACE level)\n";
        std::cout << "  -h, --help                Show this help\n\n";

        std::cout << "Examples:\n";
        std::cout << "  # Simple inference\n";
        std::cout << "  " << prog_name << " -m model.gguf -p \"Hello\" -n 50\n\n";

        std::cout << "  # Layer split: 16 layers on GPU, rest on CPU\n";
        std::cout << "  " << prog_name << " -m model.gguf --strategy layer-split --offload-layers 16\n\n";

        std::cout << "  # Custom device map\n";
        std::cout << "  " << prog_name << " -m model.gguf --device-map \"0-11:gpu:0,12-23:cpu\"\n\n";

        std::cout << "  # Memory-aware auto-fit\n";
        std::cout << "  " << prog_name << " -m model.gguf --strategy memory-aware --max-gpu-memory 8192\n\n";

        std::cout << "  # Multi-GPU balanced\n";
        std::cout << "  " << prog_name << " -m model.gguf --multi-gpu --gpu-split even\n\n";

        std::cout << "  # Interactive chat mode\n";
        std::cout << "  " << prog_name << " -m model.gguf --chat\n\n";

        std::cout << "  # Chat with system prompt\n";
        std::cout << "  " << prog_name << " -m model.gguf --chat --system \"You are a helpful assistant.\"\n\n";

        std::cout << "  # Single-shot with chat template\n";
        std::cout << "  " << prog_name << " -m model.gguf --chat-single -p \"What is the capital of France?\"\n\n";

        std::cout << "  # Benchmark mode (prefill + decode timing)\n";
        std::cout << "  " << prog_name << " -m model.gguf --benchmark -n 50\n\n";

        std::cout << "  # Benchmark with custom prompt\n";
        std::cout << "  " << prog_name << " -m model.gguf --benchmark -p \"Your prompt here\" -n 100\n\n";
    }

    bool ArgParser::matchesFlag(const std::string &arg,
                                const std::string &short_flag,
                                const std::string &long_flag)
    {
        return arg == short_flag || arg == long_flag;
    }

    std::string ArgParser::getNextArg(char *argv[], int argc, int &i, const std::string &flag_name)
    {
        if (i + 1 < argc)
        {
            return argv[++i];
        }
        else
        {
            std::cerr << "Error: Missing value for --" << flag_name << std::endl;
            return "";
        }
    }

} // namespace llaminar2
