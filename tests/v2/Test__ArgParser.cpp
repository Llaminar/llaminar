/**
 * @file Test__ArgParser.cpp
 * @brief Unit tests for ArgParser
 * @author David Sanftenberg
 * @date 2025-10-24
 */

#include "utils/ArgParser.h"
#include <gtest/gtest.h>
#include <vector>
#include <cstring>

using namespace llaminar2;

/**
 * @brief Helper to create argc/argv from string array
 */
class ArgHelper {
public:
    ArgHelper(std::vector<std::string> args) {
        args_.push_back("llaminar2");  // argv[0] = program name
        args_.insert(args_.end(), args.begin(), args.end());
        
        argv_ = new char*[args_.size()];
        for (size_t i = 0; i < args_.size(); ++i) {
            argv_[i] = new char[args_[i].size() + 1];
            std::strcpy(argv_[i], args_[i].c_str());
        }
        argc_ = static_cast<int>(args_.size());
    }
    
    ~ArgHelper() {
        for (int i = 0; i < argc_; ++i) {
            delete[] argv_[i];
        }
        delete[] argv_;
    }
    
    int argc() const { return argc_; }
    char** argv() const { return argv_; }
    
private:
    std::vector<std::string> args_;
    int argc_;
    char** argv_;
};

// ============================================================================
// Basic Argument Parsing
// ============================================================================

TEST(Test__ArgParser, ParsesModelPath) {
    ArgHelper args({"-m", "model.gguf"});
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_EQ(ctx.model_path, "model.gguf");
}

TEST(Test__ArgParser, ParsesPrompt) {
    ArgHelper args({"-p", "Hello world"});
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_EQ(ctx.prompt, "Hello world");
}

TEST(Test__ArgParser, ParsesNPredict) {
    ArgHelper args({"-n", "256"});
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_EQ(ctx.n_predict, 256);
}

TEST(Test__ArgParser, ParsesTemperature) {
    ArgHelper args({"-t", "0.7"});
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_FLOAT_EQ(ctx.temperature, 0.7f);
}

TEST(Test__ArgParser, ParsesTopK) {
    ArgHelper args({"--top-k", "50"});
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_EQ(ctx.top_k, 50);
}

TEST(Test__ArgParser, ParsesTopP) {
    ArgHelper args({"--top-p", "0.95"});
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_FLOAT_EQ(ctx.top_p, 0.95f);
}

TEST(Test__ArgParser, ParsesSeed) {
    ArgHelper args({"-s", "42"});
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_EQ(ctx.seed, 42);
}

TEST(Test__ArgParser, ParsesBatchSize) {
    ArgHelper args({"-b", "8"});
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_EQ(ctx.batch_size, 8);
}

// ============================================================================
// Device and Strategy Parsing
// ============================================================================

TEST(Test__ArgParser, ParsesDevice) {
    ArgHelper args({"--device", "cuda:0"});
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_EQ(ctx.device, "cuda:0");
}

TEST(Test__ArgParser, ParsesStrategy) {
    ArgHelper args({"--strategy", "layer-split"});
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_EQ(ctx.strategy, "layer-split");
}

TEST(Test__ArgParser, ParsesOffloadLayers) {
    ArgHelper args({"--offload-layers", "16"});
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_EQ(ctx.offload_layers, 16);
}

TEST(Test__ArgParser, ParsesDeviceMap) {
    ArgHelper args({"--device-map", "0-11:gpu:0,12-23:cpu"});
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_EQ(ctx.device_map, "0-11:gpu:0,12-23:cpu");
    EXPECT_EQ(ctx.strategy, "custom");  // Auto-set strategy
}

// ============================================================================
// Memory Management
// ============================================================================

TEST(Test__ArgParser, ParsesMaxGPUMemory) {
    ArgHelper args({"--max-gpu-memory", "8192"});
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    ASSERT_TRUE(ctx.max_gpu_memory_mb.has_value());
    EXPECT_EQ(ctx.max_gpu_memory_mb.value(), 8192);
}

TEST(Test__ArgParser, ParsesMaxCPUMemory) {
    ArgHelper args({"--max-cpu-memory", "16384"});
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    ASSERT_TRUE(ctx.max_cpu_memory_mb.has_value());
    EXPECT_EQ(ctx.max_cpu_memory_mb.value(), 16384);
}

TEST(Test__ArgParser, ParsesNoMmap) {
    ArgHelper args({"--no-mmap"});
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_FALSE(ctx.use_mmap);
}

// ============================================================================
// MoE-Specific Flags
// ============================================================================

TEST(Test__ArgParser, ParsesMoESharedGPU) {
    ArgHelper args({"--moe-shared-gpu"});
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_TRUE(ctx.moe_shared_experts_gpu);
}

TEST(Test__ArgParser, ParsesMoESharedCPU) {
    ArgHelper args({"--moe-shared-cpu"});
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_FALSE(ctx.moe_shared_experts_gpu);
}

TEST(Test__ArgParser, ParsesMoESparseGPU) {
    ArgHelper args({"--moe-sparse-gpu"});
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_FALSE(ctx.moe_sparse_experts_cpu);
}

TEST(Test__ArgParser, ParsesMoESparseCPU) {
    ArgHelper args({"--moe-sparse-cpu"});
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_TRUE(ctx.moe_sparse_experts_cpu);
}

// ============================================================================
// Multi-GPU Flags
// ============================================================================

TEST(Test__ArgParser, ParsesMultiGPU) {
    ArgHelper args({"--multi-gpu"});
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_TRUE(ctx.multi_gpu);
}

TEST(Test__ArgParser, ParsesGPUSplit) {
    ArgHelper args({"--gpu-split", "even"});
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_EQ(ctx.gpu_split, "even");
    EXPECT_TRUE(ctx.multi_gpu);  // Auto-enabled
}

// ============================================================================
// Performance Flags
// ============================================================================

TEST(Test__ArgParser, ParsesThreads) {
    ArgHelper args({"--threads", "8"});
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_EQ(ctx.n_threads, 8);
}

// ============================================================================
// Debug/Control Flags
// ============================================================================

TEST(Test__ArgParser, ParsesVerbose) {
    ArgHelper args({"-v"});
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_TRUE(ctx.verbose);
}

TEST(Test__ArgParser, ParsesVerboseLongFlag) {
    ArgHelper args({"--verbose"});
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_TRUE(ctx.verbose);
}

TEST(Test__ArgParser, ParsesListDevices) {
    ArgHelper args({"--list-devices"});
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_TRUE(ctx.list_devices);
}

TEST(Test__ArgParser, ParsesHelp) {
    ArgHelper args({"-h"});
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_TRUE(ctx.show_help);
}

TEST(Test__ArgParser, ParsesHelpLongFlag) {
    ArgHelper args({"--help"});
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_TRUE(ctx.show_help);
}

// ============================================================================
// Complex Argument Combinations
// ============================================================================

TEST(Test__ArgParser, ParsesFullInferenceConfig) {
    ArgHelper args({
        "-m", "model.gguf",
        "-p", "Test prompt",
        "-n", "100",
        "-t", "0.9",
        "--top-k", "30",
        "--top-p", "0.85",
        "-s", "123",
        "-b", "4"
    });
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_EQ(ctx.model_path, "model.gguf");
    EXPECT_EQ(ctx.prompt, "Test prompt");
    EXPECT_EQ(ctx.n_predict, 100);
    EXPECT_FLOAT_EQ(ctx.temperature, 0.9f);
    EXPECT_EQ(ctx.top_k, 30);
    EXPECT_FLOAT_EQ(ctx.top_p, 0.85f);
    EXPECT_EQ(ctx.seed, 123);
    EXPECT_EQ(ctx.batch_size, 4);
}

TEST(Test__ArgParser, ParsesLayerSplitStrategy) {
    ArgHelper args({
        "-m", "model.gguf",
        "--strategy", "layer-split",
        "--offload-layers", "20",
        "--device", "cuda:0"
    });
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_EQ(ctx.strategy, "layer-split");
    EXPECT_EQ(ctx.offload_layers, 20);
    EXPECT_EQ(ctx.device, "cuda:0");
}

TEST(Test__ArgParser, ParsesMemoryAwareStrategy) {
    ArgHelper args({
        "-m", "model.gguf",
        "--strategy", "memory-aware",
        "--max-gpu-memory", "8192",
        "--max-cpu-memory", "32768"
    });
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_EQ(ctx.strategy, "memory-aware");
    ASSERT_TRUE(ctx.max_gpu_memory_mb.has_value());
    EXPECT_EQ(ctx.max_gpu_memory_mb.value(), 8192);
    ASSERT_TRUE(ctx.max_cpu_memory_mb.has_value());
    EXPECT_EQ(ctx.max_cpu_memory_mb.value(), 32768);
}

TEST(Test__ArgParser, ParsesMoEOptimizedStrategy) {
    ArgHelper args({
        "-m", "model.gguf",
        "--strategy", "moe-optimized",
        "--moe-shared-gpu",
        "--moe-sparse-cpu"
    });
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_EQ(ctx.strategy, "moe-optimized");
    EXPECT_TRUE(ctx.moe_shared_experts_gpu);
    EXPECT_TRUE(ctx.moe_sparse_experts_cpu);
}

TEST(Test__ArgParser, ParsesCustomDeviceMap) {
    ArgHelper args({
        "-m", "model.gguf",
        "--device-map", "0-15:gpu:0,16-31:gpu:1,32-47:cpu"
    });
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_EQ(ctx.strategy, "custom");  // Auto-set
    EXPECT_EQ(ctx.device_map, "0-15:gpu:0,16-31:gpu:1,32-47:cpu");
}

TEST(Test__ArgParser, ParsesMultiGPUBalanced) {
    ArgHelper args({
        "-m", "model.gguf",
        "--multi-gpu",
        "--gpu-split", "0.6,0.4"
    });
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_TRUE(ctx.multi_gpu);
    EXPECT_EQ(ctx.gpu_split, "0.6,0.4");
}

// ============================================================================
// Default Values
// ============================================================================

TEST(Test__ArgParser, UsesDefaultValues) {
    ArgHelper args({});  // No arguments
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    
    EXPECT_EQ(ctx.prompt, "Hello, my name is");
    EXPECT_EQ(ctx.n_predict, 128);
    EXPECT_FLOAT_EQ(ctx.temperature, 0.8f);
    EXPECT_EQ(ctx.top_k, 40);
    EXPECT_FLOAT_EQ(ctx.top_p, 0.9f);
    EXPECT_EQ(ctx.seed, -1);
    EXPECT_EQ(ctx.batch_size, 1);
    EXPECT_EQ(ctx.device, "auto");
    EXPECT_EQ(ctx.strategy, "auto");
    EXPECT_EQ(ctx.offload_layers, 0);
    EXPECT_TRUE(ctx.use_mmap);
    EXPECT_TRUE(ctx.moe_shared_experts_gpu);
    EXPECT_TRUE(ctx.moe_sparse_experts_cpu);
    EXPECT_FALSE(ctx.multi_gpu);
    EXPECT_FALSE(ctx.verbose);
    EXPECT_FALSE(ctx.list_devices);
    EXPECT_FALSE(ctx.show_help);
    EXPECT_EQ(ctx.n_threads, -1);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
