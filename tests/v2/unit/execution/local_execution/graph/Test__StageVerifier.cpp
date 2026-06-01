/**
 * @file Test__StageVerifier.cpp
 * @brief Regression tests for stage output validation.
 */

#include <gtest/gtest.h>

#include "execution/compute_stages/IComputeStage.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/StageVerifier.h"
#include "utils/DebugEnv.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace llaminar2;

#if LLAMINAR_ASSERTIONS_ACTIVE

namespace
{
    class RawFP32OutputStage final : public IComputeStage
    {
    public:
        explicit RawFP32OutputStage(std::vector<float> output)
            : IComputeStage(DeviceId::cpu()), output_(std::move(output)) {}

        bool execute(IDeviceContext *) override { return true; }
        ComputeStageType type() const override { return ComputeStageType::MOE_ROUTER; }
        bool supportsBackend(ComputeBackendType backend) const override
        {
            return backend == ComputeBackendType::CPU;
        }

    protected:
        StageDumpInfo buildDumpInfoImpl() const override
        {
            StageDumpInfo info;
            info.addOutput("router_logits", output_.data(), 1, output_.size());
            return info;
        }

    private:
        std::vector<float> output_;
    };

    class ScopedValidationEnv
    {
    public:
        ScopedValidationEnv()
            : validate_buffers_(std::getenv("LLAMINAR_VALIDATE_BUFFERS")),
              fail_on_zero_(std::getenv("LLAMINAR_FAIL_ON_ZERO")),
              fail_on_nan_(std::getenv("LLAMINAR_FAIL_ON_NAN")) {}

        ~ScopedValidationEnv()
        {
            restore("LLAMINAR_VALIDATE_BUFFERS", validate_buffers_);
            restore("LLAMINAR_FAIL_ON_ZERO", fail_on_zero_);
            restore("LLAMINAR_FAIL_ON_NAN", fail_on_nan_);
            mutableDebugEnv().reload();
        }

        void enableStrictOutputValidation()
        {
            setenv("LLAMINAR_VALIDATE_BUFFERS", "1", 1);
            setenv("LLAMINAR_FAIL_ON_ZERO", "1", 1);
            setenv("LLAMINAR_FAIL_ON_NAN", "1", 1);
            mutableDebugEnv().reload();
        }

    private:
        static void restore(const char *name, const char *value)
        {
            if (value)
                setenv(name, value, 1);
            else
                unsetenv(name);
        }

        const char *validate_buffers_;
        const char *fail_on_zero_;
        const char *fail_on_nan_;
    };
} // namespace

TEST(Test__StageVerifier, SparseOddNonzeroOutputDoesNotFailAllZeroValidation)
{
    ScopedValidationEnv env;
    env.enableStrictOutputValidation();

    std::vector<float> router_logits(256, 0.0f);
    router_logits[129] = 1.0f;

    ComputeNode node(
        "layer40_moe_routing",
        std::make_unique<RawFP32OutputStage>(std::move(router_logits)),
        DeviceId::cpu());

    EXPECT_TRUE(validateStageOutputs(node));
}

TEST(Test__StageVerifier, TrulyAllZeroOutputStillFailsValidation)
{
    ScopedValidationEnv env;
    env.enableStrictOutputValidation();

    std::vector<float> output(256, 0.0f);

    ComputeNode node(
        "zero_stage",
        std::make_unique<RawFP32OutputStage>(std::move(output)),
        DeviceId::cpu());

    EXPECT_FALSE(validateStageOutputs(node));
}

#endif // LLAMINAR_ASSERTIONS_ACTIVE
