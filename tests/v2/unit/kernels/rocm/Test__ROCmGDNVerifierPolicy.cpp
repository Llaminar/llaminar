/**
 * @file Test__ROCmGDNVerifierPolicy.cpp
 * @brief Device-free GDN source, correction, control and native packet replay.
 *
 * Immutable host addresses stand for resident identities only. The complete
 * all-format width sweep rejects unmeasured shapes and arithmetic policies;
 * native integration separately authenticates actual launch choices and bytes.
 */
#include "kernels/rocm/gemm/ROCmNativeQuantisedGDNVerifier.h"
#include "../../../utils/QuantizedVerifierFormats.h"
#include <gtest/gtest.h>
using namespace llaminar2;
using namespace llaminar2::rocm;
using namespace llaminar2::test;
namespace {
/** @brief Identity-only engine for the original floating controls. */
class ControlEngine final:public ITensorGemm {
public:
    /** @return False; host admission tests do not execute arithmetic. */
    bool supports_device(int) const override{return false;}
    /** @return False; this test owns no accelerator state. */
    bool multiply_tensor(const TensorBase *,TensorBase *,int,int,int,bool,float,float,
        const TensorBase *,const IMPIContext *,int,DeviceWorkspaceManager *,int) override{return false;}
};
/** @brief Distinct source planes, original controls and one packet publication. */
struct Fixture {
    ControlEngine engine;
    std::unique_ptr<FP32Tensor> input=TestTensorFactory::createFP32({1});
    int32_t count=16,foreign_count=16,sum=0;int8_t quantized=0;
    float scale=1,outputs[5]{};uint8_t weights[3]{};uint16_t scales[3]{};
    DeviceRowRange rows=DeviceRowRange::deviceCounted(16,&count);
    NativeQuantisedGDNVerifierOperation operation{.device=0,.k=2560,.input=input.get(),.rows=rows,
        .precision=QuantizedActivationOperandPrecision::SingleQ8,.correction_rows=std::nullopt,
        .activations=&quantized,.activation_scales=&scale,.activation_sums=&sum,
        .qkv={.weights=&weights[0],.scales=&scales[0],.output=&outputs[0],.n=10240,.source={0,false,true}},
        .z={.weights=&weights[1],.scales=&scales[1],.output=&outputs[1],.n=6144,.source={0,false,true}},
        .alpha={.engine=&engine,.weights={.data=&weights[2],.type=TensorType::FP16,.n=48,.k=2560,.bytes=48*2560*2},.output=&outputs[2],.precision=FloatingActivationOperandPrecision::FP16RNE},
        .beta={.engine=&engine,.weights={.data=&weights[2],.type=TensorType::FP16,.n=48,.k=2560,.bytes=48*2560*2},.output=&outputs[3],.precision=FloatingActivationOperandPrecision::FP16RNE}};
    NativeQuantisedSmallMBatchedArgs packet{.rows=rows,.work=ProjectionRowWork::fromPrefix(rows)};
    NativeQuantisedVerifierWork work{rows,weights,10240,2560,2,0,NativeQuantisedVerifierPolicyMatch::CanonicalSerial,&packet,&quantized,&scale,&sum};
    /** @brief Initialize an original two-native-role packet. */
    Fixture(){subgroup(3);}
    /** @brief Emit any ordered nonempty native subset, including illegal corrected QKV controls. */
    void subgroup(unsigned mask) {
        packet.num_projections=0;
        const std::array members{operation.qkv,operation.z};
        for(unsigned role=0;role<2;++role)if(mask&(1u<<role)){
            const int p=packet.num_projections++;const auto &m=members[role];
            packet.payload[p]=m.weights;packet.scales[p]=m.scales;packet.mins[p]=m.mins;packet.emins[p]=m.emins;
            packet.output[p]=m.output;packet.N[p]=m.n;packet.codebook[p]=canonicalDeviceVnniCodebookId(m.source);
        }
        work.weights=packet.payload[0];work.n=packet.N[0];work.physical_codebook=packet.codebook[0];work.projections=packet.num_projections;
    }
};
}
/** @test Both SingleQ8 roles and only TwoTermQ8 Z can consume measured native work. */
TEST(ROCmGDNVerifierPolicy, OriginalRolesSurviveCountResetAndCorrection) {
    Fixture f;
    for(int live:{16,1,0,7,16})for(unsigned mask=1;mask<4;++mask){f.count=live;f.subgroup(mask);EXPECT_TRUE(matchesNativeQuantisedGDNVerifier(f.operation,f.work,0));}
    f.operation.precision=QuantizedActivationOperandPrecision::TwoTermQ8;f.operation.correction_rows=f.rows;
    for(unsigned mask=1;mask<4;++mask){f.subgroup(mask);EXPECT_EQ(matchesNativeQuantisedGDNVerifier(f.operation,f.work,0),mask==2);}
    f.subgroup(2);f.operation.correction_rows=DeviceRowRange::deviceCounted(16,&f.foreign_count);
    EXPECT_FALSE(matchesNativeQuantisedGDNVerifier(f.operation,f.work,0));
}
/** @test Native source identity cannot substitute for complete floating controls or packet provenance. */
TEST(ROCmGDNVerifierPolicy, ForeignRolesControlsAndPublicationsAreRejected) {
    for(int mutation=0;mutation<23;++mutation){SCOPED_TRACE(mutation);Fixture f;
        switch(mutation){
        case 0:f.operation.input=nullptr;break;
        case 1:f.operation.alpha.engine=nullptr;break;
        case 2:f.operation.alpha.weights.type=TensorType::BF16;break;
        case 3:f.operation.beta.precision=FloatingActivationOperandPrecision::Native;break;
        case 4:f.operation.alpha.weights.bytes-=2;break;
        case 5:f.operation.beta.output=f.operation.qkv.output;break;
        case 6:f.operation.qkv.source.present=false;break;
        case 7:f.operation.z.weights=nullptr;break;
        case 8:f.operation.device=1;break;
        case 9:f.operation.rows=DeviceRowRange::deviceCounted(16,&f.foreign_count);break;
        case 10:f.operation.activations=nullptr;break;
        case 11:f.work.activation_scales=&f.outputs[4];break;
        case 12:f.work.activation_sums=&f.foreign_count;break;
        case 13:f.work.policy=NativeQuantisedVerifierPolicyMatch::Unmeasured;break;
        case 14:f.packet.bias[1]=&f.outputs[4];break;
        case 15:f.packet.output[1]=&f.outputs[4];break;
        case 16:f.packet.scales[1]=&f.scales[2];break;
        case 17:f.packet.N[1]+=64;break;
        case 18:f.packet.codebook[1]=19;break;
        case 19:f.packet.work=ProjectionRowWork::fromPrefix(DeviceRowRange::deviceCounted(16,&f.foreign_count));break;
        case 20:f.packet.rows=DeviceRowRange::deviceCounted(16,&f.foreign_count);break;
        case 21:f.operation.correction_rows=f.rows;break;
        case 22:f.operation.precision=static_cast<QuantizedActivationOperandPrecision>(99);break;
        }
        EXPECT_FALSE(matchesNativeQuantisedGDNVerifier(f.operation,f.work,0));
    }
}
/** @test All source aliases and M1..33 remain total; only measured shapes enter the overlay. */
TEST(ROCmGDNVerifierPolicy, CompleteSourcePrecisionWidthInventory) {
    Fixture f;std::array<int,2> selected{};
    for(const auto &format:quantizedVerifierFormats())for(int mode=0;mode<2;++mode)for(int m=1;m<=33;++m){
        auto op=f.operation;op.rows=DeviceRowRange::deviceCounted(m,&f.count);
        op.precision=mode?QuantizedActivationOperandPrecision::TwoTermQ8:QuantizedActivationOperandPrecision::SingleQ8;
        op.correction_rows=mode?std::optional(op.rows):std::nullopt;
        op.qkv.source=op.z.source={format.source_codebook_id,format.source_is_superblock,true};
        const bool native=measuredNativeQuantisedGDNVerifier(op);selected[mode]+=native;
        if(m<2 || m>16)EXPECT_FALSE(native);
        auto foreign=op;foreign.qkv.source={19,false,true};if(foreign.qkv.source!=op.z.source)EXPECT_FALSE(measuredNativeQuantisedGDNVerifier(foreign));
        foreign=op;foreign.alpha.weights.n=24;foreign.alpha.weights.bytes/=2;EXPECT_FALSE(measuredNativeQuantisedGDNVerifier(foreign));
        foreign=op;foreign.k+=32;EXPECT_FALSE(measuredNativeQuantisedGDNVerifier(foreign));
    }
    EXPECT_EQ(selected[0],117);EXPECT_EQ(selected[1],44);
}
/** @test An unqualified nested backend call cannot inherit GDN admission. */
TEST(ROCmGDNVerifierPolicy, NativeNestedScopesRestoreIndependentBindings) {
    Fixture f;EXPECT_EQ(ScopedNativeQuantisedGDNVerifier::current(),nullptr);
    {ScopedNativeQuantisedGDNVerifier outer(f.operation);const auto *original=ScopedNativeQuantisedGDNVerifier::current();
        try {ScopedNativeQuantisedGDNVerifier empty(std::nullopt);EXPECT_EQ(ScopedNativeQuantisedGDNVerifier::current(),nullptr);throw std::runtime_error("interrupted capture");}catch(const std::runtime_error&){}
        EXPECT_EQ(ScopedNativeQuantisedGDNVerifier::current(),original);}
    EXPECT_EQ(ScopedNativeQuantisedGDNVerifier::current(),nullptr);
}
