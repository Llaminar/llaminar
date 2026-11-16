/**
 * @file Test__VerifierProjectionBundle.cpp
 * @brief Device-free replay of original fused verifier identity and scope ownership.
 *
 * Borrowed descriptors must not authorize copied subgroups, another input/count,
 * or a nested unmeasured call. Counts are only pointer identities in this suite;
 * no accelerator or mutable inference state is required.
 */
#include "kernels/common/VerifierProjectionBundle.h"
#include "../../utils/TestTensorFactory.h"
#include <gtest/gtest.h>
#include <array>
#include <barrier>
#include <thread>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
/** @brief Prepared-owner identity only; no test executes arithmetic through this engine. */
class BindingEngine final : public ITensorGemm
{
public:
    /** @return No device execution is provided by this immutable test identity. */
    bool supports_device(int) const override { return false; }
    /** @return False because the test must never execute this identity as a kernel. */
    bool multiply_tensor(const TensorBase *, TensorBase *, int, int, int, bool,
        float, float, const TensorBase *, const IMPIContext *, int,
        DeviceWorkspaceManager *, int) override { return false; }
};

/** @brief Small shared host tensors keep all borrowed descriptor owners alive. */
struct BundleFixture
{
    BindingEngine engine;
    std::unique_ptr<FP32Tensor> input = TestTensorFactory::createFP32({16,32});
    std::unique_ptr<FP32Tensor> foreign_input = TestTensorFactory::createFP32({16,32});
    std::array<std::unique_ptr<FP32Tensor>,3> outputs{
        TestTensorFactory::createFP32({16,64}), TestTensorFactory::createFP32({16,32}),
        TestTensorFactory::createFP32({16,32})};
    std::array<ITensorGemm::TensorProjectionDesc,3> projections{{
        {&engine,outputs[0].get(),64}, {&engine,outputs[1].get(),32},
        {&engine,outputs[2].get(),32}}};
    int32_t count = 16, foreign_count = 16;
    DeviceRowRange rows = DeviceRowRange::deviceCounted(16,&count);
    /** @return Original QKV identity; its metadata does not initialize a GPU. */
    VerifierProjectionBundleBinding binding() const
    {
        return VerifierProjectionBundleBinding::bind(VerifierProjectionBundleKind::AttentionQKV,
            DeviceId::rocm(0),input.get(),rows,32,projections);
    }
};
}

/** @test Captured identity is invariant under count publication but rejects foreign owners. */
TEST(VerifierProjectionBundle, OriginalIdentitySurvivesCountResetAndRejectsSubgroups)
{
    BundleFixture f;
    const auto binding=f.binding();
    for(int live:{16,1,0,8,16})
    {
        f.count=live;
        EXPECT_TRUE(binding.matches(DeviceId::rocm(0),f.input.get(),f.rows,32,f.projections));
    }
    auto copied=f.projections;
    EXPECT_FALSE(binding.matches(DeviceId::rocm(0),f.input.get(),f.rows,32,copied));
    EXPECT_FALSE(binding.matches(DeviceId::rocm(0),f.input.get(),f.rows,32,
        std::span(f.projections).subspan(1)));
    EXPECT_FALSE(binding.matches(DeviceId::rocm(1),f.input.get(),f.rows,32,f.projections));
    EXPECT_FALSE(binding.matches(DeviceId::rocm(0),f.foreign_input.get(),f.rows,32,f.projections));
    EXPECT_FALSE(binding.matches(DeviceId::rocm(0),f.input.get(),
        DeviceRowRange::deviceCounted(16,&f.foreign_count),32,f.projections));
    EXPECT_FALSE(binding.matches(DeviceId::rocm(0),f.input.get(),f.rows.slice(1,15),32,f.projections));
    EXPECT_FALSE(binding.matches(DeviceId::rocm(0),f.input.get(),f.rows,64,f.projections));
}

/** @test Exact projection sets reject incomplete, null and aliased semantic members. */
TEST(VerifierProjectionBundle, InvalidBindingsFailBeforeScopeInstallation)
{
    BundleFixture f;
    const auto bind=[&](VerifierProjectionBundleKind kind,auto projections) {
        return VerifierProjectionBundleBinding::bind(kind,DeviceId::rocm(0),f.input.get(),f.rows,32,projections);
    };
    EXPECT_THROW(bind(VerifierProjectionBundleKind::AttentionKV,std::span(f.projections)),std::invalid_argument);
    EXPECT_NO_THROW(bind(VerifierProjectionBundleKind::AttentionKV,std::span(f.projections).subspan(1)));
    EXPECT_THROW(bind(static_cast<VerifierProjectionBundleKind>(99),std::span(f.projections)),std::invalid_argument);
    for(int mutation=0;mutation<4;++mutation)
    {
        auto invalid=f.projections;
        if(mutation==0)invalid[1].kernel=nullptr;
        if(mutation==1)invalid[1].output=nullptr;
        if(mutation==2)invalid[1].output=invalid[0].output;
        if(mutation==3)invalid[1].n=0;
        EXPECT_THROW(bind(VerifierProjectionBundleKind::AttentionQKV,std::span(invalid)),std::invalid_argument);
    }
    EXPECT_EQ(ScopedVerifierProjectionBundle::current(),nullptr);
}

/** @test Nested empty and exceptional operations cannot inherit a parent's bundle key. */
TEST(VerifierProjectionBundle, NestedExceptionAndEmptyScopesRestoreOriginalBinding)
{
    BundleFixture f;
    EXPECT_EQ(ScopedVerifierProjectionBundle::current(),nullptr);
    {
        ScopedVerifierProjectionBundle outer(f.binding());
        const auto *original=ScopedVerifierProjectionBundle::current();
        ASSERT_NE(original,nullptr);
        try
        {
            ScopedVerifierProjectionBundle unknown(std::nullopt);
            EXPECT_EQ(ScopedVerifierProjectionBundle::current(),nullptr);
            throw std::runtime_error("recording interrupted");
        }
        catch(const std::runtime_error&) {}
        EXPECT_EQ(ScopedVerifierProjectionBundle::current(),original);
        {
            ScopedVerifierProjectionBundle kv(VerifierProjectionBundleBinding::bind(
                VerifierProjectionBundleKind::AttentionKV,DeviceId::rocm(0),f.input.get(),f.rows,
                32,std::span(f.projections).subspan(1)));
            EXPECT_EQ(ScopedVerifierProjectionBundle::current()->kind(),VerifierProjectionBundleKind::AttentionKV);
        }
        EXPECT_EQ(ScopedVerifierProjectionBundle::current(),original);
    }
    EXPECT_EQ(ScopedVerifierProjectionBundle::current(),nullptr);
}

/** @test Simultaneous recording workers retain distinct descriptors and restore independently. */
TEST(VerifierProjectionBundle, RecordingWorkersDoNotShareAdmission)
{
    BundleFixture a,b;
    std::barrier boundary(2);
    const auto run=[&](BundleFixture &f) {
        EXPECT_EQ(ScopedVerifierProjectionBundle::current(),nullptr);
        {
            ScopedVerifierProjectionBundle scope(f.binding());
            boundary.arrive_and_wait();
            EXPECT_TRUE(ScopedVerifierProjectionBundle::current()->matches(
                DeviceId::rocm(0),f.input.get(),f.rows,32,f.projections));
            boundary.arrive_and_wait();
        }
        EXPECT_EQ(ScopedVerifierProjectionBundle::current(),nullptr);
    };
    std::thread first([&]{run(a);}),second([&]{run(b);});
    first.join();second.join();
}
