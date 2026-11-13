/**
 * @file Test__SnapshotAuthoredRows.cpp
 * @brief Original-coordinate query snapshot proof without devices or model payloads.
 *
 * Unequal request histories, poison and explicit expected source rows distinguish
 * a packed authored interval from a misleading flattened prefix. Corrupt observed
 * coordinates are rejected before numeric comparison; original controls are never
 * replaced by diagnostic counts.
 */
#include "kernels/common/DiagnosticSnapshotCopy.h"
#include "snapshots/SnapshotSemanticObservation.h"
#include "tensors/Tensors.h"
#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <numeric>
#include <vector>
using namespace llaminar2;

/** @test Holes from short/empty requests pack only independently expected original rows. */
TEST(SnapshotAuthoredRows, IndependentRequestsRetainOriginalRowsAcrossDirtyReplays)
{
    constexpr int requests=3,capacity=4,columns=2;
    std::array<std::int32_t,6> publications{2049,4,0,4,2051,2};
    const auto work=ProjectionRowWork::causalStrided(requests,capacity,publications.data(),publications.data()+1,2,8192,2051);
    const DiagnosticSnapshotCopyGeometry geometry{DiagnosticSnapshotCopyKind::AuthoredTokenRows,requests*capacity,columns,4,0,1,requests};
    std::array<std::uint32_t,24> source{},destination;
    std::iota(source.begin(),source.end(),11u);
    std::vector<std::uint64_t> metadata(diagnosticSnapshotMetadataCapacity(geometry)/8,0xc3c3c3c3c3c3c3c3ull);
    auto *header=reinterpret_cast<SnapshotObservedHeader *>(metadata.data());
    DiagnosticSnapshotCopyArgs args{geometry,source.data(),destination.data(),header};args.work=work;
    std::fill(destination.begin(),destination.end(),0xa5a5a5a5u);
    copyDiagnosticSnapshotCPU(args);
    EXPECT_EQ(header->liveRows,4);EXPECT_EQ(header->payloadBytes,32u);
    const auto *records=reinterpret_cast<const SnapshotAuthoredTokenRowRecord *>(header+1);
    SnapshotAuthoredTokenRows extent{work,{}};
    const auto shape=validateSnapshotObservedHeader(extent,*header,12,2,4);
    const auto acquired=validateSnapshotAuthoredTokenRows(extent,{records,3},shape);
    ASSERT_EQ(acquired.requests.size(),3u);
    EXPECT_EQ(acquired.requests[0].firstRow,2);EXPECT_EQ(acquired.requests[0].endRow,4);
    EXPECT_EQ(acquired.requests[1].firstRow,8);EXPECT_EQ(acquired.requests[1].endRow,8);
    EXPECT_EQ(acquired.requests[1].emptyReason,SnapshotEmptyReason::RankingNotRequired);
    EXPECT_EQ(acquired.requests[2].firstRow,8);EXPECT_EQ(acquired.requests[2].endRow,10);
    const std::array<int,4> originalRows{2,3,8,9};
    for(std::size_t packed=0;packed<originalRows.size();++packed)
        for(int col=0;col<columns;++col)EXPECT_EQ(destination[packed*columns+col],source[originalRows[packed]*columns+col]);
    EXPECT_TRUE(std::all_of(destination.begin()+8,destination.end(),[](auto x){return x==0xa5a5a5a5u;}));
    // Same retained source/output storage, all requests short: no fabricated zero payload.
    publications={0,4,2048,3,2060,0};std::fill(destination.begin(),destination.end(),0xa5a5a5a5u);
    copyDiagnosticSnapshotCPU(args);EXPECT_EQ(header->liveRows,0);EXPECT_EQ(header->payloadBytes,0u);
    for(int request=0;request<3;++request)EXPECT_EQ(records[request].emptyReason,SnapshotEmptyReason::RankingNotRequired);
    EXPECT_TRUE(std::all_of(destination.begin(),destination.end(),[](auto x){return x==0xa5a5a5a5u;}));
    publications={2049,4,0,4,2051,2};copyDiagnosticSnapshotCPU(args);
    EXPECT_EQ(header->liveRows,4);
}

/** @test Clipped expanded views keep original coordinates rather than reindexing accepted rows. */
TEST(SnapshotAuthoredRows, ExpandedSliceHasExactPhysicalCoordinates)
{
    std::array<std::int32_t,4> controls{2049,4,2051,3};
    const auto work=ProjectionRowWork::causalStrided(2,4,controls.data(),controls.data()+1,2,8192,2051).repeatEachRow(3).slice(7,13);
    std::array<std::uint32_t,13> source{},destination;std::iota(source.begin(),source.end(),100u);
    const DiagnosticSnapshotCopyGeometry g{DiagnosticSnapshotCopyKind::AuthoredTokenRows,13,1,4,0,1,2};
    std::vector<std::uint64_t> metadata(diagnosticSnapshotMetadataCapacity(g)/8);
    auto *header=reinterpret_cast<SnapshotObservedHeader *>(metadata.data());
    DiagnosticSnapshotCopyArgs args{g,source.data(),destination.data(),header};args.work=work;
    std::fill(destination.begin(),destination.end(),0xa5a5a5a5u);copyDiagnosticSnapshotCPU(args);
    // Request0 physical[6,12) clipped to[7,12); request1 [12,21) clipped at20.
    EXPECT_EQ(header->liveRows,13);EXPECT_EQ(destination,source);
    const auto *records=reinterpret_cast<const SnapshotAuthoredTokenRowRecord *>(header+1);
    EXPECT_EQ(records[0].rowOffset,-7);EXPECT_EQ(records[0].firstRow,0);EXPECT_EQ(records[0].endRow,5);
    EXPECT_EQ(records[1].rowOffset,5);EXPECT_EQ(records[1].firstRow,5);EXPECT_EQ(records[1].endRow,13);
}

/** @test Negative/overflowing history and invented empty/offset records never become valid evidence. */
TEST(SnapshotAuthoredRows, RejectsMalformedOriginalAndAcquiredMetadataBeforePayloadUse)
{
    std::array<std::int32_t,4> controls{2051,1,2051,1};
    const auto work=ProjectionRowWork::causalStrided(2,2,controls.data(),controls.data()+1,2,8192,2051);
    std::array<std::uint32_t,4> source{1,2,3,4},destination;
    const DiagnosticSnapshotCopyGeometry g{DiagnosticSnapshotCopyKind::AuthoredTokenRows,4,1,4,0,1,2};
    std::vector<std::uint64_t> metadata(diagnosticSnapshotMetadataCapacity(g)/8);
    auto *header=reinterpret_cast<SnapshotObservedHeader *>(metadata.data());
    DiagnosticSnapshotCopyArgs args{g,source.data(),destination.data(),header};args.work=work;
    for(const auto bad:std::array{std::array{-1,1},std::array{8192,1},std::array{2051,3}}) {
        controls[2]=bad[0];controls[3]=bad[1];std::fill(destination.begin(),destination.end(),0xa5a5a5a5u);
        EXPECT_THROW(copyDiagnosticSnapshotCPU(args),std::invalid_argument);
        EXPECT_TRUE(std::all_of(destination.begin(),destination.end(),[](auto x){return x==0xa5a5a5a5u;}));
    }
    controls[2]=2051;controls[3]=1;copyDiagnosticSnapshotCPU(args);
    const auto shape=validateSnapshotObservedHeader(SnapshotAuthoredTokenRows{work,{}},*header,4,1,4);
    const auto *native=reinterpret_cast<const SnapshotAuthoredTokenRowRecord *>(header+1);
    for(const int corrupt:{0,1,2,3}) {
        std::array<SnapshotAuthoredTokenRowRecord,2> records{native[0],native[1]};
        if(corrupt==0)records[0].emptyReason=SnapshotEmptyReason::RankingNotRequired;
        if(corrupt==1)records[1].payloadRowOffset=0;
        if(corrupt==2)records[0].firstRow=records[0].endRow;
        if(corrupt==3)records[0].reserved=1;
        EXPECT_THROW(validateSnapshotAuthoredTokenRows(SnapshotAuthoredTokenRows{work,{}},records,shape),std::logic_error);
    }
}


/** @test Cold tensor/native ownership identity is exact while original live agreement stays execution-owned. */
TEST(SnapshotAuthoredRows, NativeInheritedBorrowRetainsOriginalOwnerWithoutHostMirror)
{
    INT32Tensor replay({2});auto *words=static_cast<std::int32_t *>(replay.raw_mutable_data());
    words[0]=2051;words[1]=2;std::int32_t originalControllerCount=2;
    const auto work=ProjectionRowWork::causalStrided(1,2,words,words+1,1,8192,2051)
        .withInheritedPrefix(DeviceRowRange::deviceCounted(2,&originalControllerCount));
    SnapshotAuthoredTokenRows authored{work,{{{&replay,0},{&replay,4},
        SnapshotNativeInt32Borrow{DeviceId::cpu(),&originalControllerCount,1,0}}}};
    EXPECT_EQ(snapshotProjectionWork(authored,false,DeviceId::cpu()).captureIdentity(),work.captureIdentity());
    auto bad=authored;bad.publications[0].frontier.byteOffset=8;
    EXPECT_THROW(snapshotProjectionWork(bad,false,DeviceId::cpu()),std::invalid_argument);
    bad=authored;bad.publications[0].inheritedLive=SnapshotNativeInt32Borrow{DeviceId::cuda(0),&originalControllerCount,1,0};
    EXPECT_THROW(snapshotProjectionWork(bad,false,DeviceId::cpu()),std::logic_error);
    bad=authored;bad.publications[0].inheritedLive=SnapshotNativeInt32Borrow{DeviceId::cpu(),&originalControllerCount,1,1};
    EXPECT_THROW(snapshotProjectionWork(bad,false,DeviceId::cpu()),std::logic_error);
    bad=authored;bad.publications[0].inheritedLive=std::monostate{};
    EXPECT_THROW(snapshotProjectionWork(bad,false,DeviceId::cpu()),std::logic_error);
    // Cold binding never reads the controller word or manufactures agreement.
    originalControllerCount=0;
    EXPECT_NO_THROW(snapshotProjectionWork(authored,false,DeviceId::cpu()));
    std::array<std::uint32_t,2> source{1,2},destination{0xa5a5a5a5u,0xa5a5a5a5u};
    std::array<std::uint64_t,8> metadata{};
    const DiagnosticSnapshotCopyGeometry g{DiagnosticSnapshotCopyKind::AuthoredTokenRows,2,1,4,0,1,1};
    DiagnosticSnapshotCopyArgs args{g,source.data(),destination.data(),reinterpret_cast<SnapshotObservedHeader *>(metadata.data())};args.work=work;
    EXPECT_THROW(copyDiagnosticSnapshotCPU(args),std::invalid_argument);
    EXPECT_EQ(destination,(std::array<std::uint32_t,2>{0xa5a5a5a5u,0xa5a5a5a5u}));
}


namespace
{
/**
 * @brief Original INT32 host storage with explicitly non-authoritative contents.
 *
 * TensorBase's concrete coherence state is the authority. In the original
 * defect, querying only ITensor's default isHostValid could certify these stale
 * bytes. This fixture changes no production API and performs no device work.
 */
class StaleHostQueryPublication final : public INT32Tensor
{
public:
    /** @brief Retain two original words in genuine test-owned CPU storage. */
    StaleHostQueryPublication():INT32Tensor({2}){}
    /** @return Invalid read authority without allocating or probing a GPU. */
    TensorCoherenceState coherenceState() const override{return TensorCoherenceState::INVALID;}
    /** @return Only an address for cold work construction, never an acquired live value. */
    const std::int32_t *originalAddress() const noexcept{return original.data();}
    mutable unsigned hostReads=0; ///< Detect any premature host pointer/value access.
protected:
    /** @return The original storage, counting attempted non-authoritative host access. */
    void *raw_host_data_ptr() override{++hostReads;return original.data();}
    /** @return The original storage, counting attempted non-authoritative host access. */
    const void *raw_host_data_ptr() const override{++hostReads;return original.data();}
private:
    std::array<std::int32_t,2> original{2051,1};
};
/**
 * @brief Allocation-shaped native metadata whose first captured producer has not executed.
 *
 * The fake native address is never dereferenced or sent to a runtime. A real
 * CPU tensor supplies dtype/extent metadata; only the test overrides allocation
 * identity and coherence, as the original cold setup contract permits.
 */
class ColdNativeQueryPublication final : public INT32Tensor
{
public:
    /** @brief Freeze one synthetic native allocation identity on the requested backend. */
    explicit ColdNativeQueryPublication(DeviceId device):INT32Tensor({2}),device_(device){}
    /** @return No live value is currently authored on either side. */
    TensorCoherenceState coherenceState() const override{return TensorCoherenceState::INVALID;}
    /** @return Exact cold native allocation participant, independent of value validity. */
    std::optional<DeviceId> current_device() const override{return device_;}
    /** @return An allocation-shaped address for static binding only. */
    void *gpu_data_ptr() override{return reinterpret_cast<void *>(std::uintptr_t{0x20000});}
    /** @return The identical cold allocation address without accessing native memory. */
    const void *gpu_data_ptr() const override{return reinterpret_cast<const void *>(std::uintptr_t{0x20000});}
protected:
    /** @brief Cold native binding must never acquire a host pointer or repair ownership. */
    void *raw_host_data_ptr() override{throw std::runtime_error("Cold native metadata has no host read authority");}
    /** @brief Cold native binding must never acquire a host pointer or repair ownership. */
    const void *raw_host_data_ptr() const override{throw std::runtime_error("Cold native metadata has no host read authority");}
private:
    const DeviceId device_;
};
}

/** @test Concrete stale-host metadata never becomes a valid snapshot or inherited agreement source. */
TEST(SnapshotAuthoredRows, RejectsOriginalNonAuthoritativeHostPublicationBeforeAnyRead)
{
    StaleHostQueryPublication original;EXPECT_FALSE(original.hostValid());
    const auto *words=original.originalAddress();
    const auto work=ProjectionRowWork::causalStrided(1,2,words,words+1,1,8192,2051);
    const SnapshotAuthoredTokenRows authored{work,{{{&original,0},{&original,4},{}}}};
    EXPECT_THROW(snapshotProjectionWork(authored,false,DeviceId::cpu()),std::logic_error);
    EXPECT_THROW(snapshotInheritedCountPointer(SnapshotInt32Binding{&original,4},false,DeviceId::cpu()),std::logic_error);
    FP32Tensor payload({2,2});std::fill_n(payload.mutable_data(),4,1.f);
    const StageDumpInfo::OutputBuffer output{.name="C",.data=payload.raw_data(),.rows=2,.cols=2,.dtype="FP32",
        .element_size=4,.byte_size=16,.tensor=&payload,.semantic_extent=authored};
    EXPECT_THROW((SnapshotSemanticObservation(output)),std::logic_error);
    EXPECT_EQ(original.hostReads,0u);
    EXPECT_FALSE(original.hostValid());
}

/** @test Both GPU cold binders authenticate storage without demanding future captured values already be valid. */
TEST(SnapshotAuthoredRows, ColdGPUOwnerBindingDoesNotReadOrRequirePublishedLiveWords)
{
    for(const auto device:{DeviceId::cuda(0),DeviceId::rocm(0)})
    {
        ColdNativeQueryPublication original(device);
        EXPECT_FALSE(original.hostValid());
        EXPECT_FALSE(original.isDeviceValid());
        const auto *words=static_cast<const std::int32_t *>(original.gpu_data_ptr());
        const auto work=ProjectionRowWork::causalStrided(1,2,words,words+1,1,8192,2051);
        const SnapshotAuthoredTokenRows authored{work,{{{&original,0},{&original,4},{}}}};
        EXPECT_EQ(snapshotProjectionWork(authored,true,device).captureIdentity(),work.captureIdentity());
        EXPECT_EQ(snapshotInheritedCountPointer(SnapshotInt32Binding{&original,4},true,device),words+1);
        const StageDumpInfo::OutputBuffer output{.name="C",.data=reinterpret_cast<void *>(std::uintptr_t{0x40000}),
            .rows=2,.cols=512,.dtype="FP32",.element_size=4,.byte_size=4096,.semantic_extent=authored};
        const auto copy=snapshotCopyArgs(output,nullptr,nullptr,true,device);
        EXPECT_EQ(copy.work.captureIdentity(),work.captureIdentity());EXPECT_EQ(copy.geometry.requests,1);
        const auto wrong=device.is_cuda()?DeviceId::rocm(0):DeviceId::cuda(0);
        EXPECT_THROW(snapshotProjectionWork(authored,true,wrong),std::logic_error);
        EXPECT_THROW(snapshotProjectionWork(authored,false,DeviceId::cpu()),std::logic_error);
        EXPECT_FALSE(original.hostValid());
        EXPECT_FALSE(original.isDeviceValid());
    }
}
