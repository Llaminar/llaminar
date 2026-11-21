/**
 * @file Test__ROCmVerifierBundlePolicy.cpp
 * @brief Device-free replay of QKV/GU/head sources, original subgroups and measured-key admission.
 *
 * Host words stand only for immutable addresses. These tests never dereference
 * a device count or initialize hardware; native capture tests qualify the actual
 * chosen kernels separately against the independent serial stage.
 */
#include "kernels/common/VerifierProjectionBundle.h"
#include "kernels/rocm/gemm/ROCmNativeQuantisedVerifierLaunch.h"
#include "../../../utils/QuantizedVerifierFormats.h"
#include "../../../utils/ROCmSelectedHeadGolden.h"
#include <gtest/gtest.h>
#include <array>

using namespace llaminar2;
using namespace llaminar2::rocm;
using namespace llaminar2::test;

namespace
{
/** @brief Distinct original members, outputs and activation publication addresses. */
struct BundleFixture
{
    int32_t count=16,foreign_count=16,sum=0,foreign_sum=0;
    uint8_t weights[4]{};
    uint16_t scales[4]{};
    uint32_t emins[4]{};
    float outputs[4]{},activation_scale=1,foreign_scale=1;
    int8_t quantized=0,foreign_quantized=0;
    DeviceRowRange rows=DeviceRowRange::deviceCounted(16,&count);
    NativeQuantisedVerifierBundleOperation operation{
        .kind=VerifierProjectionBundleKind::AttentionQKV,.device=0,.k=5120,.count=3,.rows=rows,
        .activations=&quantized,.activation_scales=&activation_scale,.activation_sums=&sum};
    NativeQuantisedSmallMBatchedArgs packet{.rows=rows,.work=ProjectionRowWork::fromPrefix(rows)};
    NativeQuantisedVerifierWork work{rows,weights,12288,5120,3,0,
        NativeQuantisedVerifierPolicyMatch::CanonicalSerial,&packet,&quantized,&activation_scale,&sum};

    /** @brief Bind complete stage members; this constructs metadata only. */
    BundleFixture()
    {
        for(int member=0;member<3;++member)
            operation.members[member]={.weights=&weights[member],.scales=&scales[member],
                .output=&outputs[member],.n=member?1024:12288,.source={0,false,true}};
        subgroup(7);
    }
    /** @brief Project an ordered nonempty subset into one real low-level packet. */
    void subgroup(unsigned mask)
    {
        packet.num_projections=0;
        for(int member=0;member<3;++member)if(mask&(1u<<member))
        {
            const int p=packet.num_projections++;
            const auto &original=operation.members[member];
            packet.payload[p]=original.weights;packet.scales[p]=original.scales;
            packet.output[p]=original.output;packet.N[p]=original.n;
            packet.codebook[p]=canonicalDeviceVnniCodebookId(original.source);
        }
        work.weights=packet.payload[0];work.n=packet.N[0];work.projections=packet.num_projections;
        work.physical_codebook=packet.codebook[0];
    }
};
}

/** @test Every ordered policy subgroup retains the whole bundle's source/row authority. */
TEST(ROCmVerifierBundlePolicy, OrderedSubgroupsAndResetKeepOriginalIdentity)
{
    BundleFixture f;
    for(unsigned mask=1;mask<8;++mask)for(int live:{16,1,0,16})
    {
        f.subgroup(mask);f.count=live;
        EXPECT_TRUE(matchesNativeQuantisedVerifierBundle(f.operation,f.work,0));
    }
    EXPECT_TRUE(measuredNativeQuantisedVerifierBundle(f.operation));
}

/** @test A same-shaped leaf cannot borrow a foreign row, operand, member or arithmetic policy. */
TEST(ROCmVerifierBundlePolicy, ForeignAndAmbiguousBindingsAreRejected)
{
    for(int mutation=0;mutation<20;++mutation)
    {
        SCOPED_TRACE(mutation);
        BundleFixture f;
        switch(mutation)
        {
        case 0:f.work.activations=&f.foreign_quantized;break;
        case 1:f.work.activation_scales=&f.foreign_scale;break;
        case 2:f.work.activation_sums=&f.foreign_sum;break;
        case 3:f.work.rows=DeviceRowRange::deviceCounted(16,&f.foreign_count);break;
        case 4:f.packet.rows=DeviceRowRange::deviceCounted(16,&f.foreign_count);break;
        case 5:f.packet.work=ProjectionRowWork::fromPrefix(DeviceRowRange::deviceCounted(16,&f.foreign_count));break;
        case 6:f.packet.payload[1]=&f.weights[3];break;
        case 7:f.packet.scales[1]=&f.scales[3];break;
        case 8:f.packet.output[1]=&f.outputs[3];break;
        case 9:f.packet.N[1]+=64;break;
        case 10:f.packet.codebook[1]=19;break;
        case 11:f.packet.bias[1]=&f.outputs[3];break;
        case 12:f.packet.output[1]=f.packet.output[2];f.packet.payload[1]=f.packet.payload[2];f.packet.scales[1]=f.packet.scales[2];break;
        case 13:f.work.policy=NativeQuantisedVerifierPolicyMatch::Unmeasured;break;
        case 14:f.operation.device=1;break;
        case 15:f.operation.members[1].output=f.operation.members[0].output;break;
        case 16:f.packet.mins[1]=&f.scales[3];break;
        case 17:f.work.projections=4;break;
        case 18:f.operation.count=2;break;
        case 19:f.operation.k+=32;break;
        }
        EXPECT_FALSE(matchesNativeQuantisedVerifierBundle(f.operation,f.work,0));
    }
    BundleFixture reversed;
    std::swap(reversed.packet.payload[1],reversed.packet.payload[2]);
    std::swap(reversed.packet.scales[1],reversed.packet.scales[2]);
    std::swap(reversed.packet.output[1],reversed.packet.output[2]);
    EXPECT_FALSE(matchesNativeQuantisedVerifierBundle(reversed.operation,reversed.work,0));
}

/** @test Full-source masks are total beyond measured widths and retain independent source aliases. */
TEST(ROCmVerifierBundlePolicy, CompleteInventoryAndUnmeasuredOperations)
{
    BundleFixture f;
    struct Shape {int k,count;std::array<int,3> n;};
    const std::array<Shape,7> shapes{{{5120,3,{12288,1024,1024}},{5120,2,{1024,1024,0}},
        {5120,3,{6144,512,512}},{5120,2,{512,512,0}},{2048,3,{8192,512,512}},
        {2048,2,{512,512,0}},{2048,3,{2048,512,512}}}};
    int selected=0;
    for(const auto &format:quantizedVerifierFormats())for(const auto &shape:shapes)for(int m=1;m<=33;++m)
    {
        auto operation=f.operation;operation.k=shape.k;operation.count=shape.count;
        operation.kind=shape.count==3?VerifierProjectionBundleKind::AttentionQKV:VerifierProjectionBundleKind::AttentionKV;
        operation.rows=DeviceRowRange::deviceCounted(m,&f.count);
        for(int member=0;member<shape.count;++member)
        {
            operation.members[member].n=shape.n[member];
            operation.members[member].source={format.source_codebook_id,format.source_is_superblock,true};
        }
        const bool native=measuredNativeQuantisedVerifierBundle(operation);
        if(m<2 || m>16)EXPECT_FALSE(native);
        selected+=native;
        operation.k+=32;EXPECT_FALSE(measuredNativeQuantisedVerifierBundle(operation));
    }
    // Independent complete-operation evidence retains 1,217 of 2,205 keys:
    // duplicate full/local KV observations intersect, with a 5% qualification
    // margin across the candidate panel and both actual-Automatic panels.
    EXPECT_EQ(selected,1217);
    EXPECT_TRUE(measuredNativeQuantisedVerifierBundle(f.operation));
    auto mixed=f.operation;mixed.members[1].source={19,false,true};
    EXPECT_FALSE(measuredNativeQuantisedVerifierBundle(mixed));
    auto short_rows=f.operation;short_rows.rows=DeviceRowRange::deviceCounted(8,&f.count);
    EXPECT_FALSE(measuredNativeQuantisedVerifierBundle(short_rows));
}

/** @test Actual local-QKV crossover losses retain the measured ordered implementation. */
TEST(ROCmVerifierBundlePolicy, ActualAutomaticLocalCrossoversRemainOrdered)
{
    BundleFixture f;
    auto operation=f.operation;operation.k=2048;
    operation.members[0].n=2048;operation.members[1].n=operation.members[2].n=512;
    struct Loss {NativeQuantisedSourceIdentity source;int rows;};
    for(const auto &loss:std::array<Loss,7>{{{{10,true,true},9},{{5,false,true},7},{{5,true,true},7},
        {{16,true,true},9},{{14,true,true},7},{{9,true,true},8},{{5,true,true},9}}}) {
        for(auto &member:operation.members)member.source=loss.source;
        operation.rows=DeviceRowRange::deviceCounted(loss.rows,&f.count);
        EXPECT_FALSE(measuredNativeQuantisedVerifierBundle(operation));
        operation.rows=DeviceRowRange::deviceCounted(16,&f.count);
        EXPECT_TRUE(measuredNativeQuantisedVerifierBundle(operation));
    }
}

/** @test Empty nested APIs hide a qualified parent and restore it after failure. */
TEST(ROCmVerifierBundlePolicy, NativeScopeCannotLeakThroughUnmeasuredCalls)
{
    BundleFixture f;
    ASSERT_EQ(ScopedNativeQuantisedVerifierBundle::current(),nullptr);
    {
        ScopedNativeQuantisedVerifierBundle outer(f.operation);
        const auto *original=ScopedNativeQuantisedVerifierBundle::current();
        ASSERT_NE(original,nullptr);
        try {ScopedNativeQuantisedVerifierBundle inner(std::nullopt);EXPECT_EQ(ScopedNativeQuantisedVerifierBundle::current(),nullptr);throw std::runtime_error("test");}
        catch(const std::runtime_error&) {}
        EXPECT_EQ(ScopedNativeQuantisedVerifierBundle::current(),original);
    }
    EXPECT_EQ(ScopedNativeQuantisedVerifierBundle::current(),nullptr);
}

/** @test Equal reduction partitions never conceal another member's different wave target. */
TEST(ROCmVerifierBundlePolicy, EveryMemberOwnsPartitionsAndWaveTarget)
{
    using Policy=NativeQuantisedVerifierSerialPolicy;
    const Policy actual{4,4};
    std::array<std::optional<Policy>,3> policies{actual,actual,actual};
    const auto qualifies=[&](int count=3) {
        return matchesNativeQuantisedVerifierSerialPolicies(count,actual,
            [&](int member){return policies.at(member);});
    };
    ASSERT_TRUE(qualifies());
    for(int member=0;member<3;++member) {
        SCOPED_TRACE(member);
        policies[member]=Policy{4,8};EXPECT_FALSE(qualifies());
        policies[member]=Policy{8,4};EXPECT_FALSE(qualifies());
        policies[member]=std::nullopt;EXPECT_FALSE(qualifies());
        policies[member]=actual;EXPECT_TRUE(qualifies());
    }
    EXPECT_FALSE(qualifies(0));
    EXPECT_FALSE(qualifies(NATIVE_QUANTISED_SMALL_M_MAX_BATCH+1));
}

/** @test Valid backend GU singleton packets retain the complete original operation. */
TEST(ROCmVerifierBundlePolicy, GateUpSubgroupsAndOriginalAuthority)
{
    BundleFixture f;f.operation.kind=VerifierProjectionBundleKind::GateUp;f.operation.count=2;
    f.operation.members[0].n=f.operation.members[1].n=17408;
    for(unsigned mask:{1u,2u,3u})for(int live:{16,1,0,16}) {
        f.subgroup(mask);f.count=live;
        EXPECT_TRUE(matchesNativeQuantisedVerifierBundle(f.operation,f.work,0));
        EXPECT_TRUE(measuredNativeQuantisedVerifierBundle(f.operation));
    }
    f.subgroup(3);
    auto foreign=f.operation;foreign.kind=VerifierProjectionBundleKind::AttentionKV;
    EXPECT_FALSE(measuredNativeQuantisedVerifierBundle(foreign));
    foreign=f.operation;foreign.count=1;EXPECT_FALSE(matchesNativeQuantisedVerifierBundle(foreign,f.work,0));
    foreign=f.operation;foreign.count=3;EXPECT_FALSE(matchesNativeQuantisedVerifierBundle(foreign,f.work,0));
    foreign=f.operation;foreign.kind=static_cast<VerifierProjectionBundleKind>(255);
    EXPECT_FALSE(matchesNativeQuantisedVerifierBundle(foreign,f.work,0));EXPECT_FALSE(measuredNativeQuantisedVerifierBundle(foreign));
    f.work.activations=&f.foreign_quantized;EXPECT_FALSE(matchesNativeQuantisedVerifierBundle(f.operation,f.work,0));
    f.work.activations=&f.quantized;f.packet.rows=DeviceRowRange::deviceCounted(16,&f.foreign_count);
    EXPECT_FALSE(matchesNativeQuantisedVerifierBundle(f.operation,f.work,0));
}

/** @test Every original source/M key remains separate from mixed aliases and unmeasured geometry. */
TEST(ROCmVerifierBundlePolicy, GateUpSourceInventoryAndMeasuredHoles)
{
    BundleFixture f;auto operation=f.operation;operation.kind=VerifierProjectionBundleKind::GateUp;operation.count=2;
    operation.members[0].n=operation.members[1].n=17408;
    int selected=0;
    for(const auto &format:quantizedVerifierFormats())for(int m=1;m<=33;++m) {
        operation.rows=DeviceRowRange::deviceCounted(m,&f.count);
        const NativeQuantisedSourceIdentity source{format.source_codebook_id,format.source_is_superblock,true};
        operation.members[0].source=operation.members[1].source=source;
        const bool native=measuredNativeQuantisedVerifierBundle(operation);selected+=native;
        if(m<2 || m>16)EXPECT_FALSE(native);
        if(m==16)EXPECT_TRUE(native);
        auto unknown=operation;unknown.k+=32;EXPECT_FALSE(measuredNativeQuantisedVerifierBundle(unknown));
        unknown=operation;unknown.members[1].n+=64;EXPECT_FALSE(measuredNativeQuantisedVerifierBundle(unknown));
        unknown=operation;unknown.members[0].source.present=false;EXPECT_FALSE(measuredNativeQuantisedVerifierBundle(unknown));
        for(const auto &other:quantizedVerifierFormats()) {
            const NativeQuantisedSourceIdentity second{other.source_codebook_id,other.source_is_superblock,true};
            if(second==source)continue;
            auto mixed=operation;mixed.members[1].source=second;
            EXPECT_FALSE(measuredNativeQuantisedVerifierBundle(mixed));
        }
    }
    // The independent frozen complete-chain panel admits136 of315 source/M keys.
    EXPECT_EQ(selected,136);
    operation.members[0].source=operation.members[1].source={4,true,true};
    operation.rows=DeviceRowRange::deviceCounted(11,&f.count);EXPECT_TRUE(measuredNativeQuantisedVerifierBundle(operation));
    operation.rows=DeviceRowRange::deviceCounted(12,&f.count);EXPECT_FALSE(measuredNativeQuantisedVerifierBundle(operation));
    operation.rows=DeviceRowRange::deviceCounted(13,&f.count);EXPECT_TRUE(measuredNativeQuantisedVerifierBundle(operation));
}

/** @test Head admission is total across every source and widths outside the measured domain. */
TEST(ROCmVerifierBundlePolicy, MaterializedHeadCompleteInventoryReplay)
{
    BundleFixture f;
    auto operation=f.operation;
    operation.kind=VerifierProjectionBundleKind::MaterializedLMHead;operation.count=1;
    int selected=0;
    for (const auto &format:quantizedVerifierFormats()) for (int k:{2048,5120,1024,2560,5152})
        for (int n:{248320,124160,248384}) for (int m=1;m<=65;++m) {
            operation.k=k;operation.members[0].n=n;operation.rows=DeviceRowRange::fullyActive(m);
            operation.members[0].source={format.source_codebook_id,format.source_is_superblock,true};
            const bool expected=selectedHeadExpectsWmma(format.tensor_type,n,k,m);
            const bool actual=measuredNativeQuantisedVerifierBundle(operation);
            EXPECT_EQ(actual,expected)<<format.label<<" N="<<n<<" K="<<k<<" M="<<m;
            selected+=actual;
            operation.rows=DeviceRowRange::deviceCounted(m,&f.count);
            EXPECT_FALSE(measuredNativeQuantisedVerifierBundle(operation));
        }
    EXPECT_EQ(selected,490);
}

/** @test Single-member head packets retain source, full-row, bank and arithmetic identity. */
TEST(ROCmVerifierBundlePolicy, MaterializedHeadForeignBindingsRejectedReplay)
{
    for (int mutation=-1;mutation<23;++mutation) {
        SCOPED_TRACE(mutation);
        BundleFixture f;
        f.operation.kind=VerifierProjectionBundleKind::MaterializedLMHead;f.operation.count=1;
        f.operation.k=f.work.k=2048;
        f.operation.rows=f.packet.rows=f.work.rows=DeviceRowRange::fullyActive(16);
        f.packet.work=ProjectionRowWork::fromPrefix(f.work.rows);
        f.operation.members[0].n=248320;f.operation.members[0].source={8,true,true};f.subgroup(1);
        ASSERT_TRUE(matchesNativeQuantisedVerifierBundle(f.operation,f.work,0));
        ASSERT_TRUE(measuredNativeQuantisedVerifierBundle(f.operation));
        switch(mutation) {
        case 0:f.work.activations=&f.foreign_quantized;break;
        case 1:f.work.activation_scales=&f.foreign_scale;break;
        case 2:f.work.activation_sums=&f.foreign_sum;break;
        case 3:f.work.rows=DeviceRowRange::deviceCounted(16,&f.count);break;
        case 4:f.packet.rows=DeviceRowRange::deviceCounted(16,&f.count);break;
        case 5:f.packet.work=ProjectionRowWork::fromPrefix(DeviceRowRange::deviceCounted(16,&f.count));break;
        case 6:f.packet.payload[0]=&f.weights[3];break;
        case 7:f.packet.scales[0]=&f.scales[3];break;
        case 8:f.packet.output[0]=&f.outputs[3];break;
        case 9:f.packet.N[0]+=64;break;
        case 10:f.packet.codebook[0]=19;break;
        case 11:f.packet.bias[0]=&f.outputs[3];break;
        case 12:f.work.policy=NativeQuantisedVerifierPolicyMatch::Unmeasured;break;
        case 13:f.operation.device=1;break;
        case 14:f.packet.mins[0]=&f.scales[3];break;
        case 15:f.work.bundle=nullptr;break;
        case 16:f.operation.count=2;break;
        case 17:f.work.k+=32;break;
        case 18:f.operation.members[0].source.present=false;break;
        case 19:f.operation.rows=DeviceRowRange::deviceCounted(16,&f.count);break;
        case 20:f.operation.rows=DeviceRowRange::fullyActive(32).slice(8,16);break;
        case 21:f.operation.rows=DeviceRowRange::fullyActive(8).repeatEachRow(2);break;
        case 22:f.packet.emins[0]=&f.emins[3];break;
        }
        EXPECT_EQ(matchesNativeQuantisedVerifierBundle(f.operation,f.work,0),mutation==-1);
        if (mutation==16 || mutation==18 || (mutation>=19 && mutation<=21))
            EXPECT_FALSE(measuredNativeQuantisedVerifierBundle(f.operation));
    }
}

/** @test Singleton scalar packets are exposed only for a complete materialized head. */
TEST(ROCmVerifierBundlePolicy, MaterializedHeadSingletonScopeReplay)
{
    BundleFixture f;
    EXPECT_FALSE(hasNativeQuantisedMaterializedHead());
    for (auto kind:{VerifierProjectionBundleKind::AttentionQKV,VerifierProjectionBundleKind::AttentionKV,
                   VerifierProjectionBundleKind::GateUp,VerifierProjectionBundleKind::MaterializedLMHead}) {
        f.operation.kind=kind;
        ScopedNativeQuantisedVerifierBundle scope(f.operation);
        EXPECT_EQ(hasNativeQuantisedMaterializedHead(),kind==VerifierProjectionBundleKind::MaterializedLMHead);
        {ScopedNativeQuantisedVerifierBundle empty(std::nullopt);EXPECT_FALSE(hasNativeQuantisedMaterializedHead());}
        EXPECT_EQ(hasNativeQuantisedMaterializedHead(),kind==VerifierProjectionBundleKind::MaterializedLMHead);
    }
    EXPECT_FALSE(hasNativeQuantisedMaterializedHead());
}

/** @test Automatic verifier WMMA qualification excludes the unmeasured MXFP4 decoder. */
TEST(ROCmVerifierBundlePolicy, WmmaQualificationExcludesUnmeasuredMXFP4)
{
    using namespace llaminar2;
    using namespace llaminar2::rocm;
    for (const auto &format : test::quantizedVerifierFormats())
        EXPECT_EQ(nativeQuantisedVerifierWmmaQualified(format.device_execution_codebook_id),
                  format.device_execution_codebook_id != kNativeQuantisedMXFP4Codebook)
            << format.label;
}
