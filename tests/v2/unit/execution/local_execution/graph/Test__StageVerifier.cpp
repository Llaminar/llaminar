/**
 * @file Test__StageVerifier.cpp
 * @brief Regression tests for stage output validation.
 */

#include <gtest/gtest.h>

#include "execution/compute_stages/IComputeStage.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/StageVerifier.h"
#include "tensors/TensorVerification.h"
#include "tensors/Tensors.h"
#include <limits>
#include "utils/DebugEnv.h"

#include <cstdlib>
#include <array>
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

    EXPECT_NO_THROW(verifyStageExit(node, 40));
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

    EXPECT_THROW(verifyStageExit(node, 0), verification::VerificationFailure);
}

/** @brief Host-owned stage exposes the same original work on input and output declarations. */
class AuthoredQueryStage final : public IComputeStage
{
public:
    /** @brief Retain original publications and numerical storage for all diagnostic borrows. */
    AuthoredQueryStage():IComputeStage(DeviceId::cpu()),payload_({8,2}),controls_({4}) {
        auto *c=static_cast<std::int32_t *>(controls_.raw_mutable_data());
        c[0]=2049;c[1]=4;c[2]=2051;c[3]=2;
        std::fill_n(payload_.mutable_data(),16,std::numeric_limits<float>::quiet_NaN());
        for(const int row:{2,3,4,5})for(int col=0;col<2;++col)payload_.mutable_data()[row*2+col]=float(row+col+1);
    }
    /** @brief Publish new host-owned original replay coordinates, without altering graph geometry. */
    void setControls(std::array<std::int32_t,4> values) {
        std::copy(values.begin(),values.end(),static_cast<std::int32_t *>(controls_.raw_mutable_data()));
    }
    /** @brief Corrupt one originally useful value to prove it cannot be skipped. */
    void poisonUseful() {payload_.mutable_data()[4]=std::numeric_limits<float>::quiet_NaN();}
    bool execute(IDeviceContext *) override{return true;}
    ComputeStageType type() const override{return ComputeStageType::MOE_ROUTER;}
    bool supportsBackend(ComputeBackendType b) const override{return b==ComputeBackendType::CPU;}
protected:
    /** @brief Declare original physical shape and per-request borrowed tensor subviews. */
    StageDumpInfo buildDumpInfoImpl() const override {
        auto *controls=const_cast<INT32Tensor *>(&controls_);
        auto *payload=const_cast<FP32Tensor *>(&payload_);
        const auto *c=static_cast<const std::int32_t *>(controls_.raw_data());
        const SnapshotAuthoredTokenRows extent{
            ProjectionRowWork::causalStrided(2,4,c,c+1,2,8192,2051),
            {{{controls,0},{controls,4},{}},{{controls,8},{controls,12},{}}}};
        StageDumpInfo dump;
        dump.inputs.push_back({.name="raw_query",.data=payload_.raw_data(),.rows=8,.cols=2,
            .dtype="FP32",.element_size=4,.byte_size=64,.tensor=payload,.semantic_extent=extent});
        dump.outputs.push_back({.name="prepared_query",.data=payload_.raw_data(),.rows=8,.cols=2,
            .dtype="FP32",.element_size=4,.byte_size=64,.tensor=payload,.semantic_extent=extent});
        return dump;
    }
private:
    FP32Tensor payload_;
    INT32Tensor controls_;
};

/** @test Both entry and exit skip only unauthored poison and reject required or malformed rows. */
TEST(Test__StageVerifier, AuthoredInputAndOutputUseOriginalRequestIntervals)
{
    ScopedValidationEnv env;env.enableStrictOutputValidation();
    auto stage=std::make_unique<AuthoredQueryStage>();auto *owned=stage.get();
    ComputeNode node("qsa_query",std::move(stage),DeviceId::cpu());
    EXPECT_NO_THROW(verifyStageEntry(node,1));
    EXPECT_NO_THROW(verifyStageExit(node,1));
    owned->setControls({0,4,2048,3});
    EXPECT_NO_THROW(verifyStageEntry(node,1));
    EXPECT_NO_THROW(verifyStageExit(node,1));
    owned->setControls({2049,4,2051,2});owned->poisonUseful();
    EXPECT_THROW(verifyStageEntry(node,1),verification::VerificationFailure);
    EXPECT_THROW(verifyStageExit(node,1),verification::VerificationFailure);
    owned->setControls({2049,4,8192,1});
    EXPECT_THROW(verifyStageEntry(node,1),verification::VerificationFailure);
}

#endif // LLAMINAR_ASSERTIONS_ACTIVE
