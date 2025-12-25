/**
 * @file Test__ArgParser.cpp
 * @brief Unit tests for ArgParser command-line argument parsing and validation
 * @author David Sanftenberg
 * @date 2025
 */

#include <gtest/gtest.h>
#include "utils/ArgParser.h"

using namespace llaminar2;

// ============================================================================
// Helper to convert string array to argc/argv
// ============================================================================

class ArgvHelper
{
public:
    ArgvHelper(std::initializer_list<const char *> args)
    {
        for (const char *arg : args)
        {
            strings_.push_back(arg);
        }
        for (auto &s : strings_)
        {
            argv_.push_back(const_cast<char *>(s.c_str()));
        }
    }

    int argc() const { return static_cast<int>(argv_.size()); }
    char **argv() { return argv_.data(); }

private:
    std::vector<std::string> strings_;
    std::vector<char *> argv_;
};

// ============================================================================
// ArgRegistry Unit Tests - Test the validation system directly
// ============================================================================

TEST(Test__ArgRegistry, FindsDefinitionByPrimaryName)
{
    const ArgDef *def = ArgRegistry::findDef("--activation-precision");
    ASSERT_NE(def, nullptr);
    EXPECT_EQ(def->name, "--activation-precision");
}

TEST(Test__ArgRegistry, FindsDefinitionByAlias)
{
    const ArgDef *def = ArgRegistry::findDef("--act-prec");
    ASSERT_NE(def, nullptr);
    EXPECT_EQ(def->name, "--activation-precision");

    def = ArgRegistry::findDef("--weight-prec");
    ASSERT_NE(def, nullptr);
    EXPECT_EQ(def->name, "--weight-precision");
}

TEST(Test__ArgRegistry, ReturnsNullForUnknownArg)
{
    const ArgDef *def = ArgRegistry::findDef("--unknown-arg");
    EXPECT_EQ(def, nullptr);
}

TEST(Test__ArgRegistry, ValidatesActivationPrecision)
{
    std::string error;

    // Valid values
    EXPECT_TRUE(ArgRegistry::validateArg("--activation-precision", "fp32", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--activation-precision", "bf16", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--activation-precision", "fp16", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--activation-precision", "q8_1", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--activation-precision", "hybrid", error));

    // Invalid values
    EXPECT_FALSE(ArgRegistry::validateArg("--activation-precision", "INVALID", error));
    EXPECT_NE(error.find("Invalid"), std::string::npos);
    EXPECT_NE(error.find("--activation-precision"), std::string::npos);

    EXPECT_FALSE(ArgRegistry::validateArg("--activation-precision", "FP32", error)); // Wrong case
    EXPECT_FALSE(ArgRegistry::validateArg("--activation-precision", "", error));     // Empty
}

TEST(Test__ArgRegistry, ValidatesWeightPrecision)
{
    std::string error;

    // Valid values
    EXPECT_TRUE(ArgRegistry::validateArg("--weight-precision", "native", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--weight-precision", "fp32", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--weight-precision", "bf16", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--weight-precision", "fp16", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--weight-precision", "int8", error));

    // Invalid values
    EXPECT_FALSE(ArgRegistry::validateArg("--weight-precision", "q4_0", error));
    EXPECT_FALSE(ArgRegistry::validateArg("--weight-precision", "", error));
}

TEST(Test__ArgRegistry, ValidatesDevice)
{
    std::string error;

    // Valid exact values
    EXPECT_TRUE(ArgRegistry::validateArg("--device", "auto", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--device", "cpu", error));

    // Valid prefix patterns
    EXPECT_TRUE(ArgRegistry::validateArg("--device", "cuda:0", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--device", "cuda:1", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--device", "rocm:0", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--device", "rocm:2", error));

    // Invalid values
    EXPECT_FALSE(ArgRegistry::validateArg("--device", "gpu", error));
    EXPECT_FALSE(ArgRegistry::validateArg("--device", "nvidia:0", error));
    EXPECT_FALSE(ArgRegistry::validateArg("--device", "", error));
}

TEST(Test__ArgRegistry, ValidatesStrategy)
{
    std::string error;

    // Valid values
    EXPECT_TRUE(ArgRegistry::validateArg("--strategy", "auto", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--strategy", "all-gpu", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--strategy", "all-cpu", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--strategy", "layer-split", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--strategy", "memory-aware", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--strategy", "moe-optimized", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--strategy", "custom", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--strategy", "multi-gpu", error));

    // Invalid values
    EXPECT_FALSE(ArgRegistry::validateArg("--strategy", "balanced", error));
    EXPECT_FALSE(ArgRegistry::validateArg("--strategy", "", error));
}

TEST(Test__ArgRegistry, ValidatesGpuSplit)
{
    std::string error;

    // Valid values
    EXPECT_TRUE(ArgRegistry::validateArg("--gpu-split", "even", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--gpu-split", "0.6,0.4", error));     // Custom ratio
    EXPECT_TRUE(ArgRegistry::validateArg("--gpu-split", "0.5,0.3,0.2", error)); // 3-way split

    // Invalid values
    EXPECT_FALSE(ArgRegistry::validateArg("--gpu-split", "balanced", error));
    EXPECT_FALSE(ArgRegistry::validateArg("--gpu-split", "abc", error));
}

TEST(Test__ArgRegistry, ValidatesFusedAttentionBackend)
{
    std::string error;

    // Valid values
    EXPECT_TRUE(ArgRegistry::validateArg("--fused-attention-backend", "jit", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--fused-attention-backend", "reference", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--fused-attention-backend", "tiled", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--fused-attention-backend", "", error)); // Empty allowed

    // Invalid values
    EXPECT_FALSE(ArgRegistry::validateArg("--fused-attention-backend", "cuda", error));
}

TEST(Test__ArgRegistry, ValidatesChatTemplate)
{
    std::string error;

    // Valid values
    EXPECT_TRUE(ArgRegistry::validateArg("--chat-template", "chatml", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--chat-template", "llama3", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--chat-template", "llama2", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--chat-template", "mistral", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--chat-template", "phi3", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--chat-template", "gemma", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--chat-template", "deepseek", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--chat-template", "", error)); // Empty allowed (auto-detect)

    // Invalid values
    EXPECT_FALSE(ArgRegistry::validateArg("--chat-template", "gpt4", error));
    EXPECT_FALSE(ArgRegistry::validateArg("--chat-template", "CHATML", error)); // Wrong case
}

TEST(Test__ArgRegistry, UnknownArgsPassValidation)
{
    std::string error;

    // Arguments not in registry should pass (they're free-form like --prompt)
    EXPECT_TRUE(ArgRegistry::validateArg("--prompt", "anything goes here", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--model", "/path/to/model.gguf", error));
    EXPECT_TRUE(ArgRegistry::validateArg("--unknown-flag", "whatever", error));
}

TEST(Test__ArgRegistry, AliasValidationWorks)
{
    std::string error;

    // Validation should work via aliases too
    EXPECT_TRUE(ArgRegistry::validateArg("--act-prec", "hybrid", error));
    EXPECT_FALSE(ArgRegistry::validateArg("--act-prec", "INVALID", error));

    EXPECT_TRUE(ArgRegistry::validateArg("--weight-prec", "native", error));
    EXPECT_FALSE(ArgRegistry::validateArg("--weight-prec", "INVALID", error));
}

// ============================================================================
// ArgParser Integration Tests - End-to-end parsing and validation
// ============================================================================

TEST(Test__ArgParser, DefaultActivationPrecisionIsHybrid)
{
    ArgvHelper args{"llaminar2"};
    auto ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_TRUE(ctx.error.empty()) << "Parse should succeed with defaults: " << ctx.error;
    EXPECT_EQ(ctx.activation_precision, "hybrid");
}

TEST(Test__ArgParser, DefaultWeightPrecisionIsNative)
{
    ArgvHelper args{"llaminar2"};
    auto ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_TRUE(ctx.error.empty());
    EXPECT_EQ(ctx.weight_precision, "native");
}

TEST(Test__ArgParser, DefaultDeviceIsAuto)
{
    ArgvHelper args{"llaminar2"};
    auto ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_TRUE(ctx.error.empty());
    EXPECT_EQ(ctx.device, "auto");
}

TEST(Test__ArgParser, DefaultStrategyIsAuto)
{
    ArgvHelper args{"llaminar2"};
    auto ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_TRUE(ctx.error.empty());
    EXPECT_EQ(ctx.strategy, "auto");
}

TEST(Test__ArgParser, AcceptsValidActivationPrecision)
{
    for (const char *val : {"fp32", "bf16", "fp16", "q8_1", "hybrid"})
    {
        ArgvHelper args{"llaminar2", "--activation-precision", val};
        auto ctx = ArgParser::parse(args.argc(), args.argv());
        EXPECT_TRUE(ctx.error.empty()) << "Should accept " << val << ": " << ctx.error;
        EXPECT_EQ(ctx.activation_precision, val);
    }
}

TEST(Test__ArgParser, RejectsInvalidActivationPrecision)
{
    ArgvHelper args{"llaminar2", "--activation-precision", "INVALID"};
    auto ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_FALSE(ctx.error.empty());
    EXPECT_NE(ctx.error.find("Invalid"), std::string::npos);
    EXPECT_NE(ctx.error.find("--activation-precision"), std::string::npos);
}

TEST(Test__ArgParser, AcceptsValidDevice)
{
    for (const char *val : {"auto", "cpu", "cuda:0", "cuda:1", "rocm:0"})
    {
        ArgvHelper args{"llaminar2", "--device", val};
        auto ctx = ArgParser::parse(args.argc(), args.argv());
        EXPECT_TRUE(ctx.error.empty()) << "Should accept --device " << val << ": " << ctx.error;
        EXPECT_EQ(ctx.device, val);
    }
}

TEST(Test__ArgParser, RejectsInvalidDevice)
{
    ArgvHelper args{"llaminar2", "--device", "nvidia:0"};
    auto ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_FALSE(ctx.error.empty());
    EXPECT_NE(ctx.error.find("Invalid"), std::string::npos);
    EXPECT_NE(ctx.error.find("--device"), std::string::npos);
}

TEST(Test__ArgParser, AcceptsValidStrategy)
{
    for (const char *val : {"auto", "all-gpu", "all-cpu", "layer-split", "memory-aware", "custom"})
    {
        ArgvHelper args{"llaminar2", "--strategy", val};
        auto ctx = ArgParser::parse(args.argc(), args.argv());
        EXPECT_TRUE(ctx.error.empty()) << "Should accept --strategy " << val << ": " << ctx.error;
        EXPECT_EQ(ctx.strategy, val);
    }
}

TEST(Test__ArgParser, RejectsInvalidStrategy)
{
    ArgvHelper args{"llaminar2", "--strategy", "balanced"};
    auto ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_FALSE(ctx.error.empty());
    EXPECT_NE(ctx.error.find("Invalid"), std::string::npos);
    EXPECT_NE(ctx.error.find("--strategy"), std::string::npos);
}

TEST(Test__ArgParser, AcceptsValidFusedAttentionBackend)
{
    // Note: This flag uses --fused-attention-backend=VALUE syntax (with =)
    for (const char *val : {"jit", "reference", "tiled"})
    {
        std::string arg = std::string("--fused-attention-backend=") + val;
        ArgvHelper args{"llaminar2", arg.c_str()};
        auto ctx = ArgParser::parse(args.argc(), args.argv());
        EXPECT_TRUE(ctx.error.empty()) << "Should accept " << arg << ": " << ctx.error;
        EXPECT_EQ(ctx.fused_attention_backend_str, val);
    }
}

TEST(Test__ArgParser, RejectsInvalidFusedAttentionBackend)
{
    ArgvHelper args{"llaminar2", "--fused-attention-backend=cuda"};
    auto ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_FALSE(ctx.error.empty());
    EXPECT_NE(ctx.error.find("Invalid"), std::string::npos);
}

TEST(Test__ArgParser, AcceptsValidChatTemplate)
{
    for (const char *val : {"chatml", "llama3", "mistral", "phi3"})
    {
        ArgvHelper args{"llaminar2", "--chat-template", val};
        auto ctx = ArgParser::parse(args.argc(), args.argv());
        EXPECT_TRUE(ctx.error.empty()) << "Should accept --chat-template " << val << ": " << ctx.error;
        EXPECT_EQ(ctx.chat_template, val);
    }
}

TEST(Test__ArgParser, RejectsInvalidChatTemplate)
{
    ArgvHelper args{"llaminar2", "--chat-template", "gpt4"};
    auto ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_FALSE(ctx.error.empty());
    EXPECT_NE(ctx.error.find("Invalid"), std::string::npos);
}

TEST(Test__ArgParser, MultipleValidArguments)
{
    ArgvHelper args{
        "llaminar2",
        "-m", "model.gguf",
        "-p", "Hello world",
        "-n", "50",
        "--activation-precision", "hybrid",
        "--weight-precision", "native",
        "--device", "cuda:0",
        "--strategy", "all-gpu"};
    auto ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_TRUE(ctx.error.empty()) << "Error: " << ctx.error;
    EXPECT_EQ(ctx.model_path, "model.gguf");
    EXPECT_EQ(ctx.prompt, "Hello world");
    EXPECT_EQ(ctx.n_predict, 50);
    EXPECT_EQ(ctx.activation_precision, "hybrid");
    EXPECT_EQ(ctx.weight_precision, "native");
    EXPECT_EQ(ctx.device, "cuda:0");
    EXPECT_EQ(ctx.strategy, "all-gpu");
}

TEST(Test__ArgParser, FirstInvalidArgSetsError)
{
    ArgvHelper args{
        "llaminar2",
        "--activation-precision", "INVALID",
        "--strategy", "auto"}; // This is valid, but won't be checked
    auto ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_FALSE(ctx.error.empty());
    EXPECT_NE(ctx.error.find("activation-precision"), std::string::npos);
}

TEST(Test__ArgParser, HelpFlagReturnsEarly)
{
    ArgvHelper args{"llaminar2", "--help", "--activation-precision", "INVALID"};
    auto ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_TRUE(ctx.show_help);
    EXPECT_TRUE(ctx.error.empty()) << "Help flag should return before validation";
}

TEST(Test__ArgParser, ListDevicesFlagReturnsEarly)
{
    ArgvHelper args{"llaminar2", "--list-devices"};
    auto ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_TRUE(ctx.list_devices);
    EXPECT_TRUE(ctx.error.empty());
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(Test__ArgParser, CaseSensitiveValidation)
{
    // All enum values should be case-sensitive (lowercase only)
    ArgvHelper args1{"llaminar2", "--activation-precision", "FP32"};
    auto ctx1 = ArgParser::parse(args1.argc(), args1.argv());
    EXPECT_FALSE(ctx1.error.empty()) << "FP32 should be rejected (case sensitive)";

    ArgvHelper args2{"llaminar2", "--strategy", "ALL-GPU"};
    auto ctx2 = ArgParser::parse(args2.argc(), args2.argv());
    EXPECT_FALSE(ctx2.error.empty()) << "ALL-GPU should be rejected (case sensitive)";
}

TEST(Test__ArgParser, EmptyValueRejectedForRequiredFields)
{
    ArgvHelper args{"llaminar2", "--activation-precision", ""};
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    EXPECT_FALSE(ctx.error.empty()) << "Empty activation-precision should be rejected";
}

TEST(Test__ArgParser, EmptyValueAllowedForOptionalFields)
{
    // chat-template allows empty (auto-detect from model)
    ArgvHelper args{"llaminar2", "--chat-template", ""};
    auto ctx = ArgParser::parse(args.argc(), args.argv());
    EXPECT_TRUE(ctx.error.empty()) << "Empty chat-template should be allowed: " << ctx.error;
}
