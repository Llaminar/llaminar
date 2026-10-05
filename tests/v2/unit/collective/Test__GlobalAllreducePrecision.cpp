/**
 * @file Test__GlobalAllreducePrecision.cpp
 * @brief Device-free proof of the global dense/MoE native GPU sum policy.
 *
 * Production graph constructors must preserve the same FP16 default for early
 * layers, full attention, recurrent attention and experts. Explicit selectors
 * retain their precedence. Removing a diagnostic environment override must
 * restore both precision and threshold rather than leave stale arithmetic in
 * the next graph. The focused reset regression also belongs to preflight.
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include <optional>
#include <string>

#include "collective/AllreducePrecisionPolicy.h"
#include "models/GraphTypes.h"
#include "models/qwen/QwenStandardGraph.h"
#include "models/qwen35/Qwen35Graph.h"
#include "models/qwen35moe/Qwen35MoEGraph.h"
#include "utils/DebugEnv.h"

using namespace llaminar2;

namespace
{
    /** @brief Restore the exact previous environment and parsed snapshot. */
    class ScopedEnv final
    {
    public:
        /** @brief Set or remove one selector for the lifetime of this scope. */
        ScopedEnv(const char *name, const char *value) : name_(name)
        {
            if (const char *old = std::getenv(name)) previous_ = old;
            set(value);
        }

        /** @brief Reestablish the caller's selector on every exit path. */
        ~ScopedEnv() { set(previous_ ? previous_->c_str() : nullptr); }

        ScopedEnv(const ScopedEnv &) = delete;
        ScopedEnv &operator=(const ScopedEnv &) = delete;

        /** @brief Publish a new diagnostic selection and refresh its authority. */
        void set(const char *value) const
        {
            if (value) ::setenv(name_.c_str(), value, 1);
            else ::unsetenv(name_.c_str());
            mutableDebugEnv().reload();
        }

    private:
        std::string name_;
        std::optional<std::string> previous_;
    };

    /** @brief Construct real graph declarations without initializing a device. */
    GraphConfig declaration(int layers)
    {
        GraphConfig config;
        config.n_layers = layers;
        config.d_model = 5120;
        config.d_ff = 17408;
        config.layer_types.resize(layers, "gdn");
        for (int layer = 3; layer < layers; layer += 4)
            config.layer_types[layer] = "full_attention";
        return config;
    }
}

TEST(Test__GlobalAllreducePrecision, UnsetSelectorsUseFP16ForEveryRow)
{
    ScopedEnv precision("LLAMINAR_ALLREDUCE_PRECISION", nullptr);
    ScopedEnv minimum("LLAMINAR_ALLREDUCE_FP16_MIN_ELEMENTS", nullptr);
    EXPECT_EQ(debugEnv().allreduce_precision, "fp16");
    EXPECT_EQ(debugEnv().allreduce_fp16_min_elements, 0u);
    EXPECT_EQ(GraphConfig{}.getAllreducePrecision(), "fp16");
    for (const size_t rows : {1u, 2u, 3u, 16u, 64u, 512u})
        for (const size_t width : {1u, 5120u, 8192u, 17408u})
            EXPECT_TRUE(fp32SumUsesFP16Transport(
                debugEnv().allreduce_precision, rows * width, width,
                debugEnv().allreduce_fp16_min_elements));
}

TEST(Test__GlobalAllreducePrecision, DenseConstructorsPreserveGlobalDefault)
{
    ScopedEnv precision("LLAMINAR_ALLREDUCE_PRECISION", nullptr);
    for (const int layers : {24, 64})
    {
        const auto config = declaration(layers);
        QwenStandardGraph standard(config, nullptr);
        QwenStandardGraph standard_full(nullptr, nullptr, config);
        Qwen35Graph recurrent(config, nullptr);
        Qwen35Graph recurrent_full(nullptr, nullptr, config);
        EXPECT_EQ(standard.config().getAllreducePrecision(), "fp16");
        EXPECT_EQ(standard_full.config().getAllreducePrecision(), "fp16");
        EXPECT_EQ(recurrent.config().getAllreducePrecision(), "fp16");
        EXPECT_EQ(recurrent_full.config().getAllreducePrecision(), "fp16");
    }
}

TEST(Test__GlobalAllreducePrecision, MoEConstructorPreservesGlobalDefault)
{
    ScopedEnv precision("LLAMINAR_ALLREDUCE_PRECISION", nullptr);
    for (const int layers : {40, 48})
    {
        auto config = declaration(layers);
        config.moe.num_experts = 256;
        config.moe.top_k = 8;
        Qwen35MoEGraph graph(config, nullptr);
        EXPECT_EQ(graph.config().getAllreducePrecision(), "fp16");
    }
}

TEST(Test__GlobalAllreducePrecision, ExplicitConfigRemainsFirstClass)
{
    ScopedEnv precision("LLAMINAR_ALLREDUCE_PRECISION", nullptr);
    for (const auto *selection : {"fp32", "fp16", "bf16"})
    {
        auto config = declaration(64);
        config.tp_allreduce_precision_override = selection;
        Qwen35Graph graph(config, nullptr);
        Qwen35MoEGraph moe(config, nullptr);
        EXPECT_EQ(graph.config().getAllreducePrecision(), selection);
        EXPECT_EQ(moe.config().getAllreducePrecision(), selection);
    }
}

TEST(Test__GlobalAllreducePrecision, DefaultAliasesUseOneGlobalPolicy)
{
    ScopedEnv precision("LLAMINAR_ALLREDUCE_PRECISION", nullptr);
    for (const auto *selection : {"", "auto", "schema", "default", "off", "AUTO"})
    {
        auto config = declaration(64);
        config.tp_allreduce_precision_override = selection;
        EXPECT_EQ(config.getAllreducePrecision(), "fp16");
    }
}

TEST(Test__GlobalAllreducePrecision, ShortAliasesNormalizeExplicitPolicy)
{
    ScopedEnv precision("LLAMINAR_ALLREDUCE_PRECISION", nullptr);
    GraphConfig config;
    config.tp_allreduce_precision_override = "F32";
    EXPECT_EQ(config.getAllreducePrecision(), "fp32");
    config.tp_allreduce_precision_override = "f16";
    EXPECT_EQ(config.getAllreducePrecision(), "fp16");
}

TEST(Test__GlobalAllreducePrecision, DiagnosticEnvironmentWinsOverConfig)
{
    ScopedEnv precision("LLAMINAR_ALLREDUCE_PRECISION", "bf16");
    GraphConfig config;
    config.tp_allreduce_precision_override = "fp32";
    EXPECT_EQ(config.getAllreducePrecision(), "bf16");
    precision.set("auto");
    EXPECT_EQ(config.getAllreducePrecision(), "fp32");
}

TEST(Test__GlobalAllreducePrecision, RemovingOverridesRestoresBothDefaults)
{
    ScopedEnv precision("LLAMINAR_ALLREDUCE_PRECISION", "fp32");
    ScopedEnv minimum("LLAMINAR_ALLREDUCE_FP16_MIN_ELEMENTS", "8192");
    EXPECT_EQ(debugEnv().allreduce_precision, "fp32");
    EXPECT_EQ(debugEnv().allreduce_fp16_min_elements, 8192u);
    precision.set(nullptr);
    minimum.set(nullptr);
    EXPECT_FALSE(debugEnv().presence.has("LLAMINAR_ALLREDUCE_PRECISION"));
    EXPECT_EQ(debugEnv().allreduce_precision, "fp16");
    EXPECT_EQ(debugEnv().allreduce_fp16_min_elements, 0u);
    EXPECT_EQ(GraphConfig{}.getAllreducePrecision(), "fp16");
}

TEST(Test__GlobalAllreducePrecision, DiagnosticThresholdRetainsSerialRowArithmetic)
{
    ScopedEnv precision("LLAMINAR_ALLREDUCE_PRECISION", "fp16");
    ScopedEnv minimum("LLAMINAR_ALLREDUCE_FP16_MIN_ELEMENTS", "8192");
    for (const size_t rows : {1u, 2u, 3u, 16u, 64u, 512u})
    {
        EXPECT_FALSE(fp32SumUsesFP16Transport("fp16", rows * 5120, 5120,
            debugEnv().allreduce_fp16_min_elements));
        EXPECT_TRUE(fp32SumUsesFP16Transport("fp16", rows * 8192, 8192,
            debugEnv().allreduce_fp16_min_elements));
    }
}
