/**
 * @file Test__VerifierGDNBundle.cpp
 * @brief Replay copied GDN subgroup provenance without hardware or count reads.
 *
 * Genuine small host tensors retain every borrowed owner. Changed role order,
 * foreign copied storage, corrected-QKV reuse and stale nesting are rejected.
 */
#include "kernels/common/VerifierGDNBundle.h"
#include "../../utils/TestTensorFactory.h"
#include <gtest/gtest.h>
#include <array>
#include <barrier>
#include <thread>
using namespace llaminar2;
using namespace llaminar2::test;
namespace {
/** @brief Identity-only prepared engine; arithmetic must never execute here. */
class GDNBindingEngine final:public ITensorGemm {
public:
    /** @return False; this fixture initializes no devices. */
    bool supports_device(int) const override{return false;}
    /** @return False; only immutable descriptor identity is under test. */
    bool multiply_tensor(const TensorBase *,TensorBase *,int,int,int,bool,float,float,
        const TensorBase *,const IMPIContext *,int,DeviceWorkspaceManager *,int) override{return false;}
};
/** @brief Complete original storage and independently copied descriptors. */
struct Fixture {
    GDNBindingEngine engine;
    std::unique_ptr<FP32Tensor> input=TestTensorFactory::createFP32({16,32});
    std::array<std::unique_ptr<FP32Tensor>,4> output{TestTensorFactory::createFP32({16,32}),
        TestTensorFactory::createFP32({16,32}),TestTensorFactory::createFP32({16,32}),TestTensorFactory::createFP32({16,32})};
    std::array<ITensorGemm::TensorProjectionDesc,4> original{{{&engine,output[0].get(),32},
        {&engine,output[1].get(),32},{&engine,output[2].get(),32},{&engine,output[3].get(),32}}};
    int32_t count=16,foreign=16;
    DeviceRowRange rows=DeviceRowRange::deviceCounted(16,&count);
    /** @return Complete original identity; corrected mode retains the same row authority. */
    VerifierGDNBundleBinding bind(bool corrected=false) const {
        return VerifierGDNBundleBinding::bind(DeviceId::rocm(0),input.get(),rows,32,original,
            corrected?QuantizedActivationOperandPrecision::TwoTermQ8:QuantizedActivationOperandPrecision::SingleQ8,
            corrected?std::optional(rows):std::nullopt);
    }
};
}
/** @test Every ordered subgroup retains all four roles through partial, empty and reset replay. */
TEST(VerifierGDNBundle, OrderedCopiedSubgroupsKeepOriginalAuthority) {
    Fixture f;
    for(bool corrected:{false,true}) {
        ScopedVerifierGDNBundle owner(f.bind(corrected));
        for(unsigned mask=1;mask<16;++mask) {
            if(corrected && (mask&1))continue;
            std::vector<VerifierGDNRole> roles;
            std::vector<ITensorGemm::TensorProjectionDesc> copied;
            for(unsigned role=0;role<4;++role)if(mask&(1u<<role)){roles.push_back(static_cast<VerifierGDNRole>(role));copied.push_back(f.original[role]);}
            const auto view=ScopedVerifierGDNBundle::current()->subgroup(roles,copied);
            ScopedVerifierGDNSubgroup subgroup(view);
            for(int live:{16,1,0,7,16}){f.count=live;EXPECT_TRUE(view.matches(DeviceId::rocm(0),f.input.get(),f.rows,32,copied));}
            const auto foreign=copied;
            EXPECT_FALSE(view.matches(DeviceId::rocm(0),f.input.get(),f.rows,32,foreign));
            EXPECT_FALSE(view.matches(DeviceId::rocm(1),f.input.get(),f.rows,32,copied));
            EXPECT_FALSE(view.matches(DeviceId::rocm(0),f.input.get(),DeviceRowRange::deviceCounted(16,&f.foreign),32,copied));
            copied[0].n+=32;EXPECT_FALSE(view.matches(DeviceId::rocm(0),f.input.get(),f.rows,32,copied));
        }
    }
}
/** @test Corrected QKV, changed role order and incomplete original rows fail before publication. */
TEST(VerifierGDNBundle, InvalidRolesAndCorrectionIdentityFail) {
    Fixture f;
    ScopedVerifierGDNBundle owner(f.bind(true));
    const auto *binding=ScopedVerifierGDNBundle::current();
    const std::array qkv{VerifierGDNRole::QKV};
    EXPECT_THROW(binding->subgroup(qkv,std::span(f.original).first(1)),std::invalid_argument);
    const std::array repeated{VerifierGDNRole::Z,VerifierGDNRole::Z};
    EXPECT_THROW(binding->subgroup(repeated,std::span(f.original).subspan(1,2)),std::invalid_argument);
    const std::array reversed{VerifierGDNRole::Alpha,VerifierGDNRole::Z};
    EXPECT_THROW(binding->subgroup(reversed,std::span(f.original).subspan(1,2)),std::invalid_argument);
    EXPECT_THROW(VerifierGDNBundleBinding::bind(DeviceId::rocm(0),f.input.get(),f.rows,32,f.original,
        QuantizedActivationOperandPrecision::TwoTermQ8,DeviceRowRange::deviceCounted(8,&f.foreign)),std::invalid_argument);
    f.original[3].output=f.original[2].output;EXPECT_THROW(f.bind(),std::invalid_argument);
}
/** @test Empty nested scopes hide a parent's view and stale owner views cannot be reinstalled. */
TEST(VerifierGDNBundle, NestingAndExceptionalRetirementRestoreExactOwner) {
    Fixture f;std::optional<VerifierGDNSubgroup> stale;
    const std::array roles{VerifierGDNRole::Z};auto copied=std::array{f.original[1]};
    {
        ScopedVerifierGDNBundle owner(f.bind());
        stale=ScopedVerifierGDNBundle::current()->subgroup(roles,copied);
        ScopedVerifierGDNSubgroup subgroup(stale);
        try {ScopedVerifierGDNBundle empty(std::nullopt);ScopedVerifierGDNSubgroup hidden(std::nullopt);
            EXPECT_EQ(ScopedVerifierGDNSubgroup::current(),nullptr);
            EXPECT_FALSE(stale->matches(DeviceId::rocm(0),f.input.get(),f.rows,32,copied));
            EXPECT_THROW(ScopedVerifierGDNSubgroup invalid(stale),std::invalid_argument);
            throw std::runtime_error("interrupted recording");}catch(const std::runtime_error&){}
        EXPECT_TRUE(stale->matches(DeviceId::rocm(0),f.input.get(),f.rows,32,copied));
    }
    EXPECT_FALSE(stale->matches(DeviceId::rocm(0),f.input.get(),f.rows,32,copied));
    EXPECT_THROW(ScopedVerifierGDNSubgroup invalid(stale),std::invalid_argument);
    EXPECT_EQ(ScopedVerifierGDNBundle::current(),nullptr);EXPECT_EQ(ScopedVerifierGDNSubgroup::current(),nullptr);
}
/** @test Concurrent recording workers cannot borrow another original operation. */
TEST(VerifierGDNBundle, IndependentRecordingWorkers) {
    Fixture a,b;std::barrier boundary(2);
    const auto run=[&](Fixture &f){ScopedVerifierGDNBundle owner(f.bind());
        const std::array roles{VerifierGDNRole::Z};auto copied=std::array{f.original[1]};
        ScopedVerifierGDNSubgroup subgroup(ScopedVerifierGDNBundle::current()->subgroup(roles,copied));
        boundary.arrive_and_wait();EXPECT_TRUE(ScopedVerifierGDNSubgroup::current()->matches(DeviceId::rocm(0),f.input.get(),f.rows,32,copied));boundary.arrive_and_wait();};
    std::thread first([&]{run(a);}),second([&]{run(b);});first.join();second.join();
}

/** @test CPU compact corrected QKV retains its host count beside fully active siblings. */
TEST(VerifierGDNBundle, CpuCorrectionPreservesIndependentOriginalRows) {
    Fixture f;
    const auto binding=VerifierGDNBundleBinding::bind(DeviceId::cpu(),f.input.get(),
        DeviceRowRange::fullyActive(16),32,f.original,QuantizedActivationOperandPrecision::TwoTermQ8,f.rows);
    EXPECT_EQ(binding.rows().captureIdentity(),DeviceRowRange::fullyActive(16).captureIdentity());
    ASSERT_TRUE(binding.correctionRows());EXPECT_EQ(binding.correctionRows()->captureIdentity(),f.rows.captureIdentity());
}
