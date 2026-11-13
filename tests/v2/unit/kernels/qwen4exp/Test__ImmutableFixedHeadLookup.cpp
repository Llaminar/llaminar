/**
 * @file Test__ImmutableFixedHeadLookup.cpp
 * @brief Device-free fixed-head source projection and complete row/frame identity.
 *
 * Opaque addresses are metadata declarations only. Real kernel execution is in
 * the separate captured CUDA/HIP fixture; no device discovery occurs here.
 */
#include "kernels/qwen4exp/ImmutableFixedHeadLookupKernels.h"
#include <gtest/gtest.h>
#include <vector>
using namespace llaminar2;
using namespace llaminar2::qwen4exp;
namespace
{
/** @brief Small original source with unequal vocabularies, an inter-head gap and trailing rows. */
NgramHashPolicy policy()
{
    NgramHashPolicy p;p.multipliers={1,3,7};
    std::uint64_t end=0;
    for(int h=0;h<16;++h) {if(h==8)end+=2;p.offsets[h]=end;p.moduli[h]=h+3;end+=h+3;}
    return p;
}
/** @brief Distinct opaque addresses separated far beyond every tested capacity. */
template<class T>T *address(std::uintptr_t index) {return reinterpret_cast<T *>(index*0x10000000ULL);}
/** @brief Checked projection with exact original source range, without allocating a byte. */
ImmutableFixedHeadLookupPlan plan(int participants=2,int rank=0,TensorType type=TensorType::IQ4_NL,int width=160)
{
    const auto p=policy();const int heads=16/participants;
    const auto first=rank? p.offsets[rank*heads]:0;
    const auto end=rank+1==participants?178:p.offsets[(rank+1)*heads];
    const auto stride=type==TensorType::FP32?width*4u:type==TensorType::FP16||type==TensorType::BF16?width*2u:
        NativeEmbeddingLayout::forSource(type,1,width).row_stride_bytes;
    return ImmutableFixedHeadLookupPlan::bind(ImmutableResidentTableRangeView::bind(DeviceId::cuda(0),type,
        address<const std::byte>(1),178,first,end-first,width,stride,(end-first)*stride,7),p,participants,rank);
}
/** @brief Exact canonical query descriptor and all distinct persistent bank declarations. */
ImmutableFixedHeadLookupArgs args(const ImmutableFixedHeadLookupPlan &p,int rows=512)
{
    const auto local=static_cast<std::size_t>(rows)*p.heads()*p.table().rowBytes();
    return {DeviceRowRange::deviceCounted(rows,address<const int>(11)),address<const std::int64_t>(2),
        static_cast<std::size_t>(rows)*16,address<FixedHeadValidationHeader>(3),address<const FixedHeadValidationHeader>(4),
        static_cast<std::size_t>(p.participants()),address<int>(5),address<std::byte>(6),local,
        address<const std::byte>(7),local*p.participants(),address<float>(8),
        static_cast<std::size_t>(rows)*16*p.table().rowElements(),address<int>(9),address<int>(10)};
}
}
/** @test Unequal source allocation is complete once, while each live response has equal head count. */
TEST(ImmutableFixedHeadLookup, GapPreservingSourceProjectionAllDegrees)
{
    for(int degree:{2,4,8})
    {
        std::uint64_t rows=0;std::size_t previousEnd=0;
        for(int rank=0;rank<degree;++rank)
        {
            const auto p=plan(degree,rank);
            EXPECT_EQ(p.table().globalRowBegin(),previousEnd);previousEnd+=p.table().rowCount();
            rows+=p.table().rowCount();EXPECT_EQ(p.heads(),16/degree);EXPECT_EQ(p.firstHead(),rank*16/degree);
            EXPECT_NO_THROW((void)fixed_head_lookup_detail::validate(p,args(p),address<void>(12)));
        }
        EXPECT_EQ(rows,178u);EXPECT_EQ(previousEnd,178u);
    }
    EXPECT_EQ(plan(2,0).table().rowCount(),54u);EXPECT_EQ(plan(2,1).table().rowCount(),124u);
}
/** @test Every canonical original codec remains source-native, with no derived representation. */
TEST(ImmutableFixedHeadLookup, OriginalFormatInventory)
{
    std::vector<TensorType> formats{TensorType::FP32,TensorType::FP16,TensorType::BF16};
#define HEAD_FORMAT(F) formats.push_back(TensorType::F);
    QFN_LOOKUP_QUANT_FORMATS(HEAD_FORMAT)
#undef HEAD_FORMAT
    for(const auto format:formats)
    {
        const auto p=plan(4,2,format,256);
        EXPECT_EQ(p.table().format(),format);EXPECT_EQ(p.table().sourceRowCount(),178u);
        EXPECT_NO_THROW((void)fixed_head_lookup_detail::validate(p,args(p,15),address<void>(12)));
    }
}
/** @test Local row aliases cannot erase gaps or pretend their local allocation is the full source. */
TEST(ImmutableFixedHeadLookup, WrongSourceHeadOrCohortRejected)
{
    const auto p=plan();
    for(int degree:{1,3,16})
        EXPECT_THROW((void)ImmutableFixedHeadLookupPlan::bind(p.table(),policy(),degree,0),std::invalid_argument);
    EXPECT_THROW((void)ImmutableFixedHeadLookupPlan::bind(p.table(),policy(),2,2),std::invalid_argument);
    EXPECT_THROW((void)ImmutableFixedHeadLookupPlan::bind(plan(2,1).table(),policy(),2,0),std::invalid_argument);
    auto metadata=policy();metadata.offsets.back()=178;
    EXPECT_THROW((void)ImmutableFixedHeadLookupPlan::bind(p.table(),metadata,2,0),std::invalid_argument);
    const auto shortRange=ImmutableResidentTableRangeView::bind(DeviceId::cuda(0),TensorType::IQ4_NL,
        address<const std::byte>(1),178,0,52,160,90,52*90,7);
    EXPECT_THROW((void)ImmutableFixedHeadLookupPlan::bind(shortRange,policy(),2,0),std::invalid_argument);
}
/** @test Original logical count capacity/origin/repeat cannot be replaced by merely equal physical rows. */
TEST(ImmutableFixedHeadLookup, CompleteOriginalRowAuthorityRequired)
{
    const auto p=plan();auto a=args(p);
    const auto check=[&](DeviceRowRange rows) {a.tokenRows=rows;
        EXPECT_THROW((void)fixed_head_lookup_detail::validate(p,a,address<void>(12)),std::invalid_argument);};
    check(DeviceRowRange::deviceCounted(256,address<const int>(11)).repeatEachRow(2));
    check(DeviceRowRange::deviceCounted(1024,address<const int>(11)).slice(1,512));
    check(DeviceRowRange::deviceCounted(1024,address<const int>(11)).slice(0,512));
    for(int rows:{1,15,512})
    {
        auto exact=args(p,rows);const auto counted=fixed_head_lookup_detail::validate(p,exact,address<void>(12));
        exact.tokenRows=DeviceRowRange::fullyActive(rows);
        EXPECT_NE(counted,fixed_head_lookup_detail::validate(p,exact,address<void>(12)));
    }
}
/** @test Every embedded pointer/extent/stream and source generation participates in capture identity. */
TEST(ImmutableFixedHeadLookup, CaptureIdentityAndStorageExclusion)
{
    const auto p=plan();const auto original=args(p);
    const auto identity=fixed_head_lookup_detail::validate(p,original,address<void>(12));
    auto changed=original;changed.transportRows=address<int>(13);
    EXPECT_NE(identity,fixed_head_lookup_detail::validate(p,changed,address<void>(12)));
    EXPECT_NE(identity,fixed_head_lookup_detail::validate(p,original,address<void>(13)));
    const auto distinct=[&](auto mutate) {auto a=original;mutate(a);
        EXPECT_NE(identity,fixed_head_lookup_detail::validate(p,a,address<void>(12)));};
    distinct([](auto &a){a.ids=address<const std::int64_t>(13);});
    distinct([](auto &a){a.localHeader=address<FixedHeadValidationHeader>(13);});
    distinct([](auto &a){a.gatheredHeaders=address<const FixedHeadValidationHeader>(13);});
    distinct([](auto &a){a.localNative=address<std::byte>(13);});
    distinct([](auto &a){a.gatheredNative=address<const std::byte>(13);});
    distinct([](auto &a){a.output=address<float>(13);});
    distinct([](auto &a){a.healthy=address<int>(13);});distinct([](auto &a){a.error=address<int>(13);});
    distinct([](auto &a){a.tokenRows=DeviceRowRange::deviceCounted(512,address<const int>(13));});
    const auto &t=p.table();
    const auto newer=ImmutableFixedHeadLookupPlan::bind(ImmutableResidentTableRangeView::bind(t.device(),t.format(),
        t.nativeData(),t.sourceRowCount(),t.globalRowBegin(),t.rowCount(),t.rowElements(),t.rowBytes(),t.nativeBytes(),
        t.storageGeneration()+1),policy(),2,0);
    EXPECT_NE(identity,fixed_head_lookup_detail::validate(newer,original,address<void>(12)));
    auto newPolicy=policy();newPolicy.multipliers[0]=9;
    EXPECT_NE(identity,fixed_head_lookup_detail::validate(
        ImmutableFixedHeadLookupPlan::bind(t,newPolicy,2,0),original,address<void>(12)));
    auto codebooks=identity;codebooks.codebooks[0]=address<void>(13);EXPECT_NE(identity,codebooks);
    const auto reject=[&](auto mutate) {auto a=original;mutate(a);
        EXPECT_THROW((void)fixed_head_lookup_detail::validate(p,a,address<void>(12)),std::invalid_argument);};
    reject([](auto &a){--a.localNativeBytes;});reject([](auto &a){--a.gatheredNativeBytes;});
    reject([](auto &a){--a.idCapacity;});reject([](auto &a){--a.headerCapacity;});
    reject([](auto &a){--a.outputElements;});reject([](auto &a){a.transportRows=a.healthy;});
    reject([](auto &a){a.transportRows=const_cast<int *>(a.tokenRows.countOwner());});
    reject([](auto &a){a.gatheredNative=a.localNative;});
    reject([](auto &a){a.gatheredHeaders=a.localHeader;});
    for(std::uintptr_t stream:{0,1,2})
        EXPECT_THROW((void)fixed_head_lookup_detail::validate(p,original,reinterpret_cast<void *>(stream)),std::invalid_argument);
    EXPECT_THROW((void)fixed_head_lookup_detail::product(std::numeric_limits<std::size_t>::max(),2),std::overflow_error);
}
/** @test Native observations borrow exactly the same health/error request row and enter complete capture identity. */
TEST(ImmutableFixedHeadLookup, CounterBindingRequiresOriginalRequestAndCompleteStorageIdentity)
{
    const auto p=plan();auto original=args(p,15);
    original.counters={address<int>(13),sampling_math::kDeviceGenerationControlCount};
    original.healthy=original.counters.control+sampling_math::kDeviceGenerationControlOk;
    original.error=original.counters.control+sampling_math::kDeviceGenerationControlErrorCode;
    const auto identity=fixed_head_lookup_detail::validate(p,original,address<void>(12));
    auto newer=original;newer.counters.control=address<int>(14);
    newer.healthy=newer.counters.control+sampling_math::kDeviceGenerationControlOk;
    newer.error=newer.counters.control+sampling_math::kDeviceGenerationControlErrorCode;
    EXPECT_NE(identity,fixed_head_lookup_detail::validate(p,newer,address<void>(12)));
    const auto reject=[&](auto change){auto a=original;change(a);
        EXPECT_THROW((void)fixed_head_lookup_detail::validate(p,a,address<void>(12)),std::invalid_argument);};
    reject([](auto &a){a.counters.words=60;});
    reject([](auto &a){a.counters.control++;});
    reject([](auto &a){a.healthy=address<int>(9);});
    reject([](auto &a){a.error=address<int>(10);});
    reject([](auto &a){a.tokenRows=DeviceRowRange::deviceCounted(15,
        a.counters.control+sampling_math::kDeviceGenerationControlFixedHeadCallsLow);});
}

/** @test CPU-only linkage and explicitly disabled selections reject without native work. */
TEST(ImmutableFixedHeadLookup, CompiledBackendSelectionRejectsDisabledAndNonGPU)
{
    // Calling the real selector also retains its owning object in static links;
    // a CPU-only build must resolve no disabled CUDA/HIP factory symbols.
    const auto reject=[&](DeviceId device,const char *message) {
        try {
            (void)createImmutableFixedHeadLookupKernels(device,plan());
            FAIL()<<"Unsupported fixed-head backend was accepted";
        } catch(const std::invalid_argument &error) {
            EXPECT_STREQ(error.what(),message);
        }
    };
    reject(DeviceId::cpu(),"Fixed-head lookup requires an explicit CUDA or ROCm participant");
    reject(DeviceId::invalid(),"Fixed-head lookup requires an explicit CUDA or ROCm participant");
#ifndef HAVE_CUDA
    reject(DeviceId::cuda(0),"Fixed-head lookup CUDA backend is not compiled");
#endif
#ifndef HAVE_ROCM
    reject(DeviceId::rocm(0),"Fixed-head lookup ROCm backend is not compiled");
#endif
    // Compiled native branches execute only in their existing explicit device suites.
}
