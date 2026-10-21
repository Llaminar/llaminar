/**
 * @file Test__AdaptiveImmutableRowCacheGeometry.cpp
 * @brief Exact metadata/native BOM and ownership tests for the single-GPU component.
 *
 * Only pure source descriptors, typed layouts and canonical PMA admission run.
 * No source file is opened and no backend, model or allocator is initialized.
 * Capacity-sized GPU transfers and runtime kernel/source correctness remain
 * separate native obligations; this suite never claims serving qualification.
 */
#include "kernels/qwen4exp/AdaptiveImmutableRowCacheGeometry.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <vector>
namespace llaminar2::test {
namespace {
namespace a=qwen4exp::adaptive_lookup;
using Geometry=qwen4exp::AdaptiveImmutableRowCacheGeometry;
using Storage=ImmutableRowStorage;
using Id=PhysicalMemoryAllocatorIdentity;
/** @return Exact requester identity without opening a device. */
Id gpu(DeviceType backend=DeviceType::CUDA,int ordinal=0,int rank=0){return {rank,DeviceId(backend,ordinal)};}
/** @return Source metadata only; the deliberately nonexistent path must never be opened. */
ImmutableRowTableSource source(std::uint64_t rows=10000,GGUFTensorType type=GGUFTensorType::IQ4_NL,std::uint64_t columns=160){
    return {.path="/nonexistent/metadata-only-qfn-native.gguf",.tensor_name="ngram.embedding.weight",
        .file_offset=37,.rows=rows,.row_elements=columns,.format=type};
}
/** @return Checked pooled capacity and exact local SSD endpoint, not fixed source homes. */
ImmutableRowTableLayoutGeometry layout(const ImmutableRowTableSource &s,std::uint64_t slots,
    Id requester=gpu(),std::uint64_t maximumIds=8192,std::optional<std::size_t> ceiling=std::nullopt,
    int serviceRank=0){
    const std::array tiers{Storage::VRAM,Storage::MemoryMapped};
    return ImmutableRowTableLayoutGeometry::checked(ImmutableRowTableExtent::fromNative(s.format,s.rows,s.rowBytes()),
        ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,
        {{{requester,Storage::VRAM},0,slots,ceiling}},{requester},{},
        {{{serviceRank,DeviceId::cpu()},ImmutableRowTableStoragePolicy::forStorage(Storage::MemoryMapped,65536,2)}},
        {},slots?std::optional<Id>(requester):std::nullopt,maximumIds);
}
/** @return Explicit policy; this is an experimental candidate rather than a public serving default. */
a::Policy policy(){return {4096,3,4};}
/** @return An observation used by canonical PMA alone for full aggregate admission. */
PhysicalMemoryResource resource(Id requester,std::size_t available=1ull<<40){return {requester.world_rank,requester.device,1ull<<40,available};}
/** @return Exact source-owned owner charge, with a missing resource interpreted as zero. */
std::size_t bytes(const PhysicalMemoryPlan &p,Id id,PhysicalMemoryOwner owner){const auto *b=p.find(id);return b?b->bytes(owner):0;}
/** @brief Check disjoint named intervals within one admitted root; no allocator is invoked. */
void intervals(std::vector<std::pair<std::size_t,std::size_t>> spans,std::size_t root){
    std::erase_if(spans,[](const auto &p){return !p.second;});std::sort(spans.begin(),spans.end());
    std::size_t end=0;
    for(const auto &[offset,size]:spans){EXPECT_EQ(offset%64,0u);ASSERT_GE(offset,end);ASSERT_LE(offset,root);ASSERT_LE(size,root-offset);end=offset+size;}
    EXPECT_EQ(root%64,0u);
}
/** @brief Inspect exact record extents against their actual production types, including all frame arrays. */
void exactOffsets(const Geometry &g){
    const auto &c=g.cache();const auto &f=g.frame();const std::size_t n=g.slots(),k=g.maximumIds();
    if(!n){EXPECT_EQ(c,qwen4exp::AdaptiveCacheOffsets{});}
    else {
        const auto slotBytes=g.rowKeyWidth()==a::RowKeyWidth::UInt32?sizeof(a::Slot32):sizeof(a::Slot64);
        intervals({{c.owner,sizeof(a::OwnerControl)},{c.slots,n*slotBytes},{c.nodes,(n-1)*sizeof(a::RadixNode)},
            {c.epochs,g.laneWidth()*4*sizeof(std::uint32_t)},{c.counts,g.laneWidth()*4*sizeof(std::uint16_t)},
            {c.payload,n*g.slotStride()}},c.bytes);
    }
    intervals({{f.control,sizeof(a::FrameControl)},{f.keys_a,k*sizeof(a::KeyIndex)},{f.keys_b,k*sizeof(a::KeyIndex)},
        {f.first,k*sizeof(std::uint32_t)},{f.original_to_unique,k*sizeof(std::uint32_t)},
        {f.first_prefix,k*sizeof(std::uint32_t)},{f.unique,k*sizeof(a::UniqueRecord)},
        {f.miss_prefix,k*sizeof(std::uint32_t)},{f.acknowledgements,k*sizeof(ImmutableRowInvalidationAcknowledgement)},
        {f.scan,g.scanBytes()},{f.native_response,k*g.rowBytes()}},f.bytes);
    EXPECT_EQ(g.mandatoryDeviceBytes(),f.bytes+sizeof(llaminar2::qwen4exp::ImmutableLookupDeviceState));
}
}
TEST(AdaptiveImmutableRowCacheGeometry, ZeroSlotsKeepMandatoryProtocolResponseAndOriginalTier){
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm}){
        const auto s=source();const auto id=gpu(backend);const auto declaration=layout(s,0,id);
        const auto g=Geometry::fromLayout(declaration,id,s,512,policy());
        EXPECT_EQ(g.slots(),0u);EXPECT_EQ(g.laneWidth(),0u);EXPECT_EQ(g.protectedLimit(),0u);
        EXPECT_FALSE(declaration.gpuCoordinator());EXPECT_TRUE(declaration.tiers().contains(Storage::VRAM));
        EXPECT_EQ(g.maximumIds(),8192u);EXPECT_EQ(g.rowBytes(),90u);EXPECT_GT(g.frame().bytes,8192u*90u);
        EXPECT_EQ(g.cacheBOM(resource(id)).totalBytes(),0u);
        EXPECT_EQ(bytes(g.mandatoryBOM(resource(id)),id,PhysicalMemoryOwner::ImmutableLookupStaging),g.mandatoryDeviceBytes());
        EXPECT_EQ(g.channelLayout().protocol(),llaminar2::qwen4exp::kImmutableLookupSubordinateProtocol);
        EXPECT_EQ(g.channelLayout().maximumIds(),8192u);EXPECT_EQ(g.channelLayout().rowBytes(),90u);exactOffsets(g);
    }
}
TEST(AdaptiveImmutableRowCacheGeometry, EveryNativeFormatKeepsOriginalRowsAndAllRecordOffsets){
    // The exact23 original GGUF formats accepted by the resident source owner.
    // Internal Q8_1 has no original GGUF source discriminator and is not
    // invented here as a source-file format.
    const std::array types{GGUFTensorType::F32,GGUFTensorType::F16,GGUFTensorType::BF16,
        GGUFTensorType::Q4_0,GGUFTensorType::Q4_1,GGUFTensorType::Q5_0,GGUFTensorType::Q5_1,
        GGUFTensorType::Q8_0,GGUFTensorType::Q2_K,GGUFTensorType::Q3_K,GGUFTensorType::Q4_K,
        GGUFTensorType::Q5_K,GGUFTensorType::Q6_K,GGUFTensorType::Q8_K,GGUFTensorType::IQ1_S,
        GGUFTensorType::IQ1_M,GGUFTensorType::IQ2_XXS,GGUFTensorType::IQ2_XS,GGUFTensorType::IQ2_S,
        GGUFTensorType::IQ3_XXS,GGUFTensorType::IQ3_S,GGUFTensorType::IQ4_NL,GGUFTensorType::IQ4_XS};
    for(const auto type:types)for(const auto count:{0u,1u,2u,7u,64u}){
        SCOPED_TRACE(::testing::Message()<<int(type)<<":"<<count);
        const auto s=source(10000,type,256);const auto g=Geometry::fromLayout(layout(s,count),gpu(),s,15,policy());
        EXPECT_EQ(g.format(),ggufToTensorType(type));EXPECT_EQ(g.rowBytes(),s.rowBytes());EXPECT_EQ(g.rowElements(),256);
        EXPECT_GE(g.slotStride(),g.rowBytes());EXPECT_LT(g.slotStride()-g.rowBytes(),4u);
        EXPECT_EQ(g.maximumIds(),240u);EXPECT_EQ(g.slots(),count);exactOffsets(g);
        if(count){EXPECT_GE(g.laneWidth(),16u);EXPECT_LE(g.protectedLimit(),count-1);}
    }
}
TEST(AdaptiveImmutableRowCacheGeometry, C1MetadataCannotDegenerateAndByteCeilingCoversEntireCache){
    const auto s=source();const auto one=Geometry::fromLayout(layout(s,1),gpu(),s,1,policy());
    EXPECT_EQ(one.cache().nodes,0u);EXPECT_GE(one.laneWidth(),16u);EXPECT_EQ(one.protectedLimit(),0u);
    EXPECT_GT(one.cache().bytes,s.rowBytes());
    EXPECT_NO_THROW((void)Geometry::fromLayout(layout(s,1,gpu(),16,one.cache().bytes),gpu(),s,1,policy()));
    EXPECT_THROW((void)Geometry::fromLayout(layout(s,1,gpu(),16,one.cache().bytes-1),gpu(),s,1,policy()),std::invalid_argument);
    EXPECT_THROW((void)Geometry::fromLayout(layout(s,1,gpu(),16,s.rowBytes()),gpu(),s,1,policy()),std::invalid_argument);
}
TEST(AdaptiveImmutableRowCacheGeometry, DirectoryScalesWithSlotsNotOriginalRowsAndSupportsFullWidthIndices){
    const auto small=source(10000),large=source(320001536);
    const auto x=Geometry::fromLayout(layout(small,64),gpu(),small,512,policy());
    const auto y=Geometry::fromLayout(layout(large,64),gpu(),large,512,policy());
    EXPECT_EQ(x.cache(),y.cache());EXPECT_EQ(x.frame(),y.frame());EXPECT_EQ(x.laneWidth(),y.laneWidth());
    const auto source32=source(0x100000000ULL),source64=source(0x100000001ULL);
    const auto g32=Geometry::fromLayout(layout(source32,64),gpu(),source32,1,policy());
    const auto g64=Geometry::fromLayout(layout(source64,64),gpu(),source64,1,policy());
    EXPECT_EQ(g32.rowKeyWidth(),a::RowKeyWidth::UInt32);EXPECT_EQ(g64.rowKeyWidth(),a::RowKeyWidth::UInt64);
    EXPECT_GT(g64.cache().bytes,g32.cache().bytes);EXPECT_EQ(g32.frame(),g64.frame());
    constexpr auto beyond24Bits=(1u<<24)+1;
    const auto many=source(beyond24Bits+1ull);const auto huge=Geometry::fromLayout(layout(many,beyond24Bits),gpu(),many,1,policy());
    EXPECT_EQ(huge.slots(),beyond24Bits);EXPECT_GT(huge.cache().bytes,static_cast<std::size_t>(beyond24Bits)*sizeof(a::Slot32));
    exactOffsets(huge); // Arithmetic only; no capacity-sized vector or allocation.
}
TEST(AdaptiveImmutableRowCacheGeometry, CanonicalPMAAddsExactMandatoryCacheAndHigherPriorityOwners){
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})for(const auto count:{0u,1u,64u}){
        const auto id=gpu(backend);const auto s=source();const auto g=Geometry::fromLayout(layout(s,count,id),id,s,15,policy());
        const auto required=4096+g.mandatoryDeviceBytes()+g.cache().bytes;
        for(const bool fits:{true,false}){
            const auto observed=resource(id,required-(fits?0:1));
            PhysicalMemoryPlanBuilder total;total.add(observed,PhysicalMemoryOwner::PrimaryModelWeights,4096);
            const auto mandatory=g.mandatoryBOM(observed),cache=g.cacheBOM(observed);
            for(const auto &b:mandatory.resources())total.add(b);
            for(const auto &b:cache.resources())total.add(b);
            const auto plan=total.build();EXPECT_EQ(plan.fits(),fits);EXPECT_EQ(plan.totalBytes(),required);
            EXPECT_EQ(plan.resources().size(),1u);
            EXPECT_EQ(bytes(plan,id,PhysicalMemoryOwner::ImmutableTableStorage),g.cache().bytes);
            EXPECT_EQ(bytes(plan,id,PhysicalMemoryOwner::ImmutableLookupStaging),g.mandatoryDeviceBytes());
            EXPECT_EQ(bytes(plan,id,PhysicalMemoryOwner::PrimaryModelWeights),4096u);
            EXPECT_EQ(bytes(plan,{0,DeviceId::cpu()},PhysicalMemoryOwner::ImmutableLookupStaging),0u);
            if(fits){EXPECT_NO_THROW((void)PhysicalMemoryPlanAdmissionCertificate(plan));}
            else {EXPECT_THROW((void)PhysicalMemoryPlanAdmissionCertificate(plan),PhysicalMemoryCapacityExhausted);}
        }
    }
}
TEST(AdaptiveImmutableRowCacheGeometry, ExactAllocatorAndObservationConflictsRejectBeforeAdmission){
    const auto s=source();const auto g=Geometry::fromLayout(layout(s,2),gpu(),s,1,policy());
    for(const auto foreign:{gpu(DeviceType::CUDA,1),gpu(DeviceType::ROCm),gpu(DeviceType::CUDA,0,1),Id{0,DeviceId::cpu()}}){
        EXPECT_THROW((void)g.mandatoryBOM(resource(foreign)),std::invalid_argument);
        EXPECT_THROW((void)g.cacheBOM(resource(foreign)),std::invalid_argument);
    }
    PhysicalMemoryPlanBuilder total;
    const auto mandatory=g.mandatoryBOM(resource(gpu(),100000)),cache=g.cacheBOM(resource(gpu(),100001));
    for(const auto &b:mandatory.resources())total.add(b);
    for(const auto &b:cache.resources())EXPECT_THROW(total.add(b),std::invalid_argument);
}
TEST(AdaptiveImmutableRowCacheGeometry, WrongDomainSourcePolicyAndLiveEnvelopeRejectWithoutFallback){
    const auto s=source();const auto declaration=layout(s,2);auto changed=s;++changed.rows;
    EXPECT_THROW((void)Geometry::fromLayout(declaration,gpu(),changed,1,policy()),std::invalid_argument);
    changed=s;changed.row_elements=192;
    EXPECT_THROW((void)Geometry::fromLayout(declaration,gpu(),changed,1,policy()),std::invalid_argument);
    EXPECT_THROW((void)Geometry::fromLayout(declaration,gpu(DeviceType::CUDA,1),s,1,policy()),std::invalid_argument);
    EXPECT_THROW((void)Geometry::fromLayout(layout(s,2,gpu(),8192,std::nullopt,1),gpu(),s,1,policy()),std::invalid_argument);
    EXPECT_THROW((void)Geometry::fromLayout(declaration,gpu(),s,0,policy()),std::invalid_argument);
    EXPECT_THROW((void)Geometry::fromLayout(layout(s,2,gpu(),15),gpu(),s,1,policy()),std::invalid_argument);
    for(const auto invalid:{a::Policy{0,1,2},a::Policy{1,1,0},a::Policy{1,3,2}}){
        EXPECT_THROW((void)Geometry::fromLayout(declaration,gpu(),s,1,invalid),std::invalid_argument);
    }
    EXPECT_THROW((void)Geometry::fromLayout(declaration,gpu(),s,std::numeric_limits<std::size_t>::max(),policy()),std::invalid_argument);
}
TEST(AdaptiveImmutableRowCacheGeometry, CaptureGeometryIdentityIncludesSourcePolicyAndOriginalRows){
    const auto s=source();const auto declaration=layout(s,2);
    const auto g=Geometry::fromLayout(declaration,gpu(),s,15,policy());
    EXPECT_EQ(g,Geometry::fromLayout(declaration,gpu(),s,15,policy()));
    auto changed=policy();++changed.aging_accesses;
    EXPECT_NE(g,Geometry::fromLayout(declaration,gpu(),s,15,changed));
    changed=policy();--changed.protected_numerator;
    EXPECT_NE(g,Geometry::fromLayout(declaration,gpu(),s,15,changed));
    EXPECT_NE(g,Geometry::fromLayout(declaration,gpu(),s,16,policy()));
    EXPECT_NE(g,Geometry::fromLayout(layout(s,3),gpu(),s,15,policy()));
    EXPECT_NE(g,Geometry::fromLayout(layout(s,2,gpu(DeviceType::ROCm)),gpu(DeviceType::ROCm),s,15,policy()));
}
} // namespace llaminar2::test
