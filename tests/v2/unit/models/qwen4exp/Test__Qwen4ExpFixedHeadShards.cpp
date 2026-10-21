/**
 * @file Test__Qwen4ExpFixedHeadShards.cpp
 * @brief Device-free original-source range, exact ownership and byte-BOM proofs.
 *
 * No payload or backend is used. Unequal head vocabularies, internal/leading
 * gaps and trailing native rows must remain completely and uniquely owned.
 * Unsupported topology and every malformed interval reject before preparation.
 */
#include "models/qwen4exp/Qwen4ExpFixedHeadShards.h"
#include "kernels/qwen4exp/ImmutableResidentTableRangeView.h"
#include "loaders/ModelLoader.h"
#include <gtest/gtest.h>
#include <array>
#include <limits>
#include <type_traits>
using namespace llaminar2;
namespace {
/** @brief Source-owned16-head metadata with explicit unequal storage and two-row internal gap. */
struct Heads {
    std::array<std::uint64_t,16> offsets{},sizes{};std::uint64_t rows=0;
    /** @brief Preserve all leading/inter-head/trailing rows in the full extent. */
    explicit Heads(std::uint64_t leading=0) {
        rows=leading;
        for(std::size_t h=0;h<16;++h){if(h==8)rows+=2;offsets[h]=rows;sizes[h]=h+3;rows+=sizes[h];}
        rows+=8;
    }
    /** @return Original160-element IQ4_NL native row geometry. */
    ImmutableRowTableExtent extent()const{return ImmutableRowTableExtent::fromNative(GGUFTensorType::IQ4_NL,rows,90);}
};
/** @return Ordered real-owner metadata, deliberately not sorted by ordinal. */
std::vector<PhysicalMemoryAllocatorIdentity> cohort(std::size_t count,DeviceType backend=DeviceType::CUDA) {
    std::vector<PhysicalMemoryAllocatorIdentity> out;
    for(std::size_t p=0;p<count;++p)out.push_back({2,DeviceId(backend,static_cast<int>(count-p-1))});
    return out;
}
}
/** @test A range retains full identity while half-open row and native byte bounds stay exact. */
TEST(ImmutableRowTableSourceRange, OriginalExtentIsNeverReplacedByLocalRows) {
    Heads h;const auto extent=h.extent();const auto r=ImmutableRowTableSourceRange::checked(extent,54,124);
    EXPECT_EQ(r.sourceExtent(),extent);EXPECT_EQ(r.firstRow(),54u);EXPECT_EQ(r.rows(),124u);
    EXPECT_EQ(r.byteOffset(),4860u);EXPECT_EQ(r.nativeBytes(),11160u);EXPECT_FALSE(r.isWhole());
    EXPECT_FALSE(r.contains(53));EXPECT_TRUE(r.contains(54));EXPECT_TRUE(r.contains(177));EXPECT_FALSE(r.contains(178));
    EXPECT_TRUE(ImmutableRowTableSourceRange::whole(extent).isWhole());
    EXPECT_THROW((void)ImmutableRowTableSourceRange::checked(extent,0,0),std::invalid_argument);
    EXPECT_THROW((void)ImmutableRowTableSourceRange::checked(extent,178,1),std::invalid_argument);
    EXPECT_THROW((void)ImmutableRowTableSourceRange::checked(extent,54,125),std::invalid_argument);
    EXPECT_THROW((void)ImmutableRowTableSourceRange::checked(extent,1,std::numeric_limits<std::uint64_t>::max()),std::invalid_argument);
    EXPECT_THROW((void)ImmutableRowTableExtent::fromNative(GGUFTensorType::IQ4_NL,
        std::numeric_limits<std::uint64_t>::max(),90),std::overflow_error);
}
/** @test Equal eight-head workloads retain54 versus124 native rows, not equal halves of the file. */
TEST(Qwen4ExpFixedHeadShards, UnequalHeadStorageCoversInternalGapAndTrailingSourceExactlyOnce) {
    Heads h;const auto owners=cohort(2);const auto p=Qwen4ExpFixedHeadShards::declare(h.extent(),h.offsets,h.sizes,owners);
    ASSERT_TRUE(p->isFixedHeadShards());EXPECT_FALSE(p->isHostAdaptive());EXPECT_EQ(p->extent(),h.extent());
    const auto first=p->fixedHeadSourceRangeFor(owners[0]),second=p->fixedHeadSourceRangeFor(owners[1]);
    EXPECT_EQ(first.firstRow(),0u);EXPECT_EQ(first.rows(),54u);EXPECT_EQ(first.nativeBytes(),4860u);
    EXPECT_EQ(second.firstRow(),54u);EXPECT_EQ(second.rows(),124u);EXPECT_EQ(second.nativeBytes(),11160u);
    EXPECT_EQ(first.nativeBytes()+second.nativeBytes(),178u*90u);
    EXPECT_EQ(Qwen4ExpFixedHeadShards::assignmentFor(*p,owners[1]).first_head,8u);
    EXPECT_THROW((void)p->storage(),std::logic_error);
    EXPECT_THROW((void)p->storagePolicy(),std::logic_error);
    EXPECT_THROW((void)p->hostPlacement(),std::logic_error);
    EXPECT_THROW((void)p->storageForConsumer(owners[0]),std::logic_error);
    EXPECT_EQ(p->lookupForConsumer(owners[0]),ImmutableRowLookupExecution::FixedHeadShardGPU);
    EXPECT_NE(p->lookupForConsumer(owners[0]),ImmutableRowLookupExecution::ResidentGPU);
    EXPECT_TRUE(immutableRowLookupUsesNativeDeviceStorage(p->lookupForConsumer(owners[0])));
    EXPECT_FALSE(immutableRowLookupUsesNativeDeviceStorage(ImmutableRowLookupExecution::HostProvider));
    EXPECT_THROW((void)immutableRowLookupUsesNativeDeviceStorage(static_cast<ImmutableRowLookupExecution>(255)),std::invalid_argument);
    EXPECT_THROW(p->requireRequestedPolicy({ImmutableRowStoragePreference::VRAM}),std::logic_error);
}
/** @test Every supported cohort/backend uses one sole source range per actual owner, with leading gaps retained. */
TEST(Qwen4ExpFixedHeadShards, AllSupportedHomogeneousCohortsPreserveExactSourceCoverageAndBOM) {
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})for(const std::size_t degree:{2u,4u,8u}) {
        Heads h(5);const auto owners=cohort(degree,backend);const auto p=Qwen4ExpFixedHeadShards::declare(h.extent(),h.offsets,h.sizes,owners);
        std::vector<PhysicalMemoryResource> resources;
        for(const auto &o:owners)resources.push_back({o.world_rank,o.device,1u<<20,1u<<20});
        const auto bom=p->fixedHeadShardStorageBOM(resources);EXPECT_TRUE(bom.fits());
        std::size_t bytes=0;std::uint64_t frontier=0;
        for(std::size_t n=0;n<degree;++n) {
            const auto a=Qwen4ExpFixedHeadShards::assignmentFor(*p,owners[n]);
            EXPECT_EQ(a.participant,n);EXPECT_EQ(a.first_head,n*16/degree);EXPECT_EQ(a.head_count,16/degree);
            EXPECT_EQ(a.source.firstRow(),frontier);frontier+=a.source.rows();bytes+=a.source.nativeBytes();
            const auto *line=bom.find(owners[n]);ASSERT_NE(line,nullptr);
            EXPECT_EQ(line->bytes(PhysicalMemoryOwner::ImmutableTableStorage),a.source.nativeBytes());
            EXPECT_EQ(line->totalBytes(),a.source.nativeBytes());
            for(std::size_t head=a.first_head;head<a.first_head+a.head_count;++head) {
                EXPECT_TRUE(a.source.contains(h.offsets[head]));EXPECT_TRUE(a.source.contains(h.offsets[head]+h.sizes[head]-1));
            }
        }
        EXPECT_EQ(frontier,h.rows);EXPECT_EQ(bytes,h.extent().nativeBytes());EXPECT_EQ(bom.totalBytes(),h.extent().nativeBytes());
        // Pricing authors an exact BOM, never an allocation permit or a fit
        // exemption. Zero sampled availability must remain a canonical reject.
        auto unavailable=resources;
        for(auto &resource:unavailable){resource.admission_available_bytes=0;}
        EXPECT_FALSE(p->fixedHeadShardStorageBOM(unavailable).fits());
        EXPECT_EQ(p->fixedHeadShardLayout().fixedRanges().size(),degree);
        EXPECT_EQ(p->allocators().size(),degree);EXPECT_EQ(p->consumers().size(),degree);
        auto bad=resources;bad.pop_back();EXPECT_THROW((void)p->fixedHeadShardStorageBOM(bad),std::invalid_argument);
        bad=resources;bad.back()=bad.front();EXPECT_THROW((void)p->fixedHeadShardStorageBOM(bad),std::invalid_argument);
        bad=resources;bad.back().device=DeviceId(backend,99);EXPECT_THROW((void)p->fixedHeadShardStorageBOM(bad),std::invalid_argument);
        EXPECT_THROW((void)p->fixedHeadSourceRangeFor({3,owners[0].device}),std::invalid_argument);
    }
}
/** @test No unsupported participant set or malformed source-head contract can mint a shard declaration. */
TEST(Qwen4ExpFixedHeadShards, ForeignMixedDuplicateAndMalformedHeadInputsReject) {
    Heads h;auto owners=cohort(2);
    for(const std::size_t count:{0u,1u,3u,5u,16u})
        EXPECT_THROW((void)Qwen4ExpFixedHeadShards::declare(h.extent(),h.offsets,h.sizes,cohort(count)),std::invalid_argument);
    auto bad=owners;bad[1]=bad[0];EXPECT_THROW((void)Qwen4ExpFixedHeadShards::declare(h.extent(),h.offsets,h.sizes,bad),std::invalid_argument);
    bad=owners;bad[1].world_rank=3;EXPECT_THROW((void)Qwen4ExpFixedHeadShards::declare(h.extent(),h.offsets,h.sizes,bad),std::invalid_argument);
    bad=owners;bad[1].device=DeviceId::rocm(0);EXPECT_THROW((void)Qwen4ExpFixedHeadShards::declare(h.extent(),h.offsets,h.sizes,bad),std::invalid_argument);
    bad=owners;bad[1].device=DeviceId::cpu();EXPECT_THROW((void)Qwen4ExpFixedHeadShards::declare(h.extent(),h.offsets,h.sizes,bad),std::invalid_argument);
    EXPECT_THROW((void)Qwen4ExpFixedHeadShards::declare(h.extent(),std::span(h.offsets).first(15),h.sizes,owners),std::invalid_argument);
    for(int mutation=0;mutation<4;++mutation) {
        auto malformed=h;
        if(mutation==0)malformed.sizes[7]=0;
        if(mutation==1)malformed.offsets[8]=malformed.offsets[7];
        if(mutation==2)malformed.sizes[15]=std::numeric_limits<std::uint64_t>::max();
        if(mutation==3)malformed.offsets[15]=malformed.rows;
        EXPECT_THROW((void)Qwen4ExpFixedHeadShards::declare(h.extent(),malformed.offsets,malformed.sizes,owners),std::invalid_argument);
    }
}
/** @test Range view identity includes global origin/full extent and cannot become a fake whole-table view. */
TEST(ImmutableResidentTableRangeView, FullAndLocalCoordinatesAreDistinctAndChecked) {
    using View=llaminar2::qwen4exp::ImmutableResidentTableRangeView;
    using Whole=llaminar2::qwen4exp::ImmutableResidentTableView;
    static_assert(!std::is_convertible_v<View,Whole>);
    alignas(64)std::array<std::byte,16> memory{};
    const auto v=View::bind(DeviceId::cuda(0),TensorType::FP32,memory.data(),8,4,4,1,4,16,1);
    EXPECT_EQ(v.sourceRowCount(),8u);EXPECT_EQ(v.globalRowBegin(),4u);EXPECT_EQ(v.rowCount(),4u);EXPECT_EQ(v.nativeBytes(),16u);
    EXPECT_NE(v,View::bind(DeviceId::cuda(0),TensorType::FP32,memory.data(),9,4,4,1,4,16,1));
    EXPECT_NE(v,View::bind(DeviceId::cuda(0),TensorType::FP32,memory.data(),8,3,4,1,4,16,1));
    EXPECT_NE(v,View::bind(DeviceId::cuda(0),TensorType::FP32,memory.data(),8,4,4,1,4,16,2));
    EXPECT_THROW((void)View::bind(DeviceId::cuda(0),TensorType::FP32,memory.data(),8,5,4,1,4,16,1),std::invalid_argument);
    EXPECT_THROW((void)View::bind(DeviceId::cuda(0),TensorType::FP32,memory.data(),8,4,0,1,4,0,1),std::invalid_argument);
    EXPECT_THROW((void)View::bind(DeviceId::cuda(0),TensorType::FP32,memory.data(),8,4,4,1,4,32,1),std::invalid_argument);
    EXPECT_THROW((void)View::bind(DeviceId::cuda(0),TensorType::FP32,memory.data(),8,4,4,1,4,16,0),std::invalid_argument);
    EXPECT_THROW((void)View::bind(DeviceId::cpu(),TensorType::FP32,memory.data(),8,4,4,1,4,16,1),std::invalid_argument);
    EXPECT_THROW((void)View::bind(DeviceId::cuda(0),TensorType::FP32,memory.data(),
        std::numeric_limits<std::uint64_t>::max(),0,4,1,4,16,1),std::invalid_argument);
}
