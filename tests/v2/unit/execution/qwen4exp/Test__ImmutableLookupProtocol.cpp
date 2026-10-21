/**
 * @file Test__ImmutableLookupProtocol.cpp
 * @brief Device-free adversarial retained immutable-row protocol and decoder proof.
 *
 * Explicit simulated GPU publications exercise real RAM/mmap native gathering,
 * generation authentication, acknowledgements, terminal close and exact wire
 * extents. Poisoned unused capacity proves large-to-small replay economy;
 * malformed identities fail before native payload bytes can become valid.
 */
#include "execution/qwen4exp/ImmutableLookupService.h"
#include "execution/local_execution/graph/RetainedStoragePerfStats.h"
#include "kernels/qwen4exp/ImmutableLookupDecode.h"
#include "loaders/ModelLoader.h"
#include "utils/OpenMPUtils.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <barrier>
#include <cstdio>
#include <exception>
#include <functional>
#include <thread>
#include <sys/resource.h>
#include <atomic>
#include <bit>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <unistd.h>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::qwen4exp;
namespace
{
/** @brief Own a tiny immutable source for both explicit storage modes. */
struct FixtureFile
{
    /** @brief Write finite distinguishable F32 rows once before immutable admission. */
    FixtureFile()
    {
        char pattern[]="/tmp/llaminar-immutable-protocol-XXXXXX";fd=::mkstemp(pattern);
        if(fd<0)throw std::runtime_error("Immutable lookup fixture file creation failed");path=pattern;
        for(std::size_t i=0;i<values.size();++i)values[i]=static_cast<float>(i-128.0)/17;
        if(::write(fd,values.data(),sizeof(values))!=sizeof(values))throw std::runtime_error("Immutable lookup fixture write failed");
    }
    /** @brief Remove only this private test fixture after source owners retire. */
    ~FixtureFile(){::close(fd);::unlink(path.c_str());}
    /** @brief Exact original source geometry, with no derived table conversion. */
    ImmutableRowTableSource source() const
    {
        return {.path=path,.tensor_name="fixture.ple.weight",.file_offset=0,.rows=37,.row_elements=8,.format=GGUFTensorType::F32};
    }
    int fd=-1;std::string path;std::array<float,37*8> values{};
};
/** @brief Exact PMA certificate for the native source, independent of protocol test bytes. */
std::shared_ptr<PhysicalMemoryAuthority> tableAuthority(std::size_t bytes)
{
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0,.device=DeviceId::cpu(),.total_bytes=1u<<20,.admission_available_bytes=1u<<20},
        PhysicalMemoryOwner::ImmutableTableStorage,bytes);
    return std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
}
/** @brief Real whole-provider ownership used by original RAM/mmap protocol controls. */
class WholeProtocolSource final
{
public:
    /** @brief Adopt the actual admitted native owner; no substitute source or protocol path. */
    WholeProtocolSource(ImmutableRowTableSource source,ImmutableRowTableStoragePolicy policy,DeviceId host,
        std::shared_ptr<PhysicalMemoryAuthority> memory,ImmutableRowTableHostPlacement locality)
        :provider_(HostImmutableRowProvider::fromWholeTable(std::make_unique<ImmutableRowTable>(
            std::move(source),policy,host,std::move(memory),locality))) {}
    /** @return The retained actual provider used by the production host processor. */
    const std::shared_ptr<HostImmutableRowProvider> &provider() const noexcept {return provider_;}
    /** @return Original whole-source completed mapping observation, preserving the old control assertions. */
    ImmutableRowGatherObservation gather(std::span<const std::int64_t> ids,std::span<std::byte> out)
    {return std::get<WholeImmutableRowGatherObservation>(provider_->gather(ids,out)).windows;}
    /** @return Coherent source-only lifetime observations for the original control comparison. */
    ImmutableRowTableStats stats() const {return provider_->sourceStats();}
private:
    std::shared_ptr<HostImmutableRowProvider> provider_;
};
/** @brief Matching aligned delete for caller-owned device-free protocol storage. */
struct AlignedDelete { /** @brief Retire admitted fixture bytes. */ void operator()(std::byte *p)const{::operator delete(p,std::align_val_t{64});} };
/** @brief Simulate only the documented GPU wire ownership, never a host token mirror. */
class Lane
{
public:
    /** @brief Bind one real source and a small independently admitted channel fixture. */
    Lane(WholeProtocolSource &table):Lane(table.provider()) {}
    /** @brief Retain the real shared provider for every simulated native channel. */
    Lane(std::shared_ptr<HostImmutableRowProvider> table):provider_(std::move(table)),layout(ImmutableLookupLayout::create(64,32)),
        bytes(static_cast<std::byte *>(::operator new(layout.bytes(),std::align_val_t{64}))),
        host(*provider_,layout,{bytes.get(),layout.bytes()},17,&fatalStatus)
    {std::fill_n(bytes.get()+layout.idsOffset(),layout.bytes()-layout.idsOffset(),std::byte{0xa5});}
    /** @brief Build exact live request metadata and release publication last. */
    void publish(std::uint64_t sequence,std::uint64_t generation,std::span<const std::int64_t> ids)
    {
        if(!ids.empty())std::memcpy(bytes.get()+layout.idsOffset(),ids.data(),ids.size_bytes());
        ImmutableLookupRequest record;record.sequenceComplement=~sequence;
        record.generation=generation;record.generationComplement=~generation;record.table=17;record.tableComplement=~record.table;
        record.idCount=ids.size();record.idCountComplement=~record.idCount;
        record.idBytes=ids.size_bytes();record.idBytesComplement=~record.idBytes;
        record.rowBytes=32;record.rowBytesComplement=~record.rowBytes;
        std::memcpy(reinterpret_cast<std::byte *>(&request())+8,reinterpret_cast<const std::byte *>(&record)+8,kLookupRequestLiveBytes-8);
        std::atomic_ref(request().publication).store(sequence,std::memory_order_release);
    }
    /** @brief Retire an acquired response exactly as the consumer's last reader does. */
    void ack(std::uint64_t sequence){std::atomic_ref(acknowledgement().consumedSequence).store(sequence,std::memory_order_release);}
    /** @brief Publish a complete-generation terminal close after the final ack. */
    void close(std::uint64_t generation){std::atomic_ref(acknowledgement().closedGeneration).store(generation,std::memory_order_release);}
    /** @brief Access CPU-side simulated wire metadata for adversarial tests. */
    ImmutableLookupRequest &request(){return *reinterpret_cast<ImmutableLookupRequest *>(bytes.get()+layout.requestOffset());}
    /** @brief Read the exact CPU response metadata. */
    ImmutableLookupCompletion &completion(){return *reinterpret_cast<ImmutableLookupCompletion *>(bytes.get()+layout.completionOffset());}
    /** @brief Access simulated GPU retirement metadata. */
    ImmutableLookupAcknowledgement &acknowledgement(){return *reinterpret_cast<ImmutableLookupAcknowledgement *>(bytes.get()+layout.acknowledgementOffset());}
    std::shared_ptr<HostImmutableRowProvider> provider_;
    ImmutableLookupLayout layout;std::unique_ptr<std::byte,AlignedDelete> bytes;
    std::atomic<ImmutableLookupStatus> fatalStatus{ImmutableLookupStatus::Success};
    ImmutableLookupHostProcessor host;
};
}

/** @test Native payload and wire counters are exact for empty, duplicate and changing live extents. */
TEST(ImmutableLookupProtocol, RAMAndMmapExactLiveBytesAcrossRetainedGenerations)
{
    FixtureFile file;
    for(auto mode:{ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped})
    {
        const auto policy=ImmutableRowTableStoragePolicy::forStorage(mode,4096,2);
        auto memory=tableAuthority(ImmutableRowTable::physicalBytes(file.source(),policy));
        WholeProtocolSource table(file.source(),policy, DeviceId::cpu(),memory,ImmutableRowTableHostPlacement::taskPolicy());Lane lane(table);
        std::uint64_t sequence=0,idsTotal=0;
        for(const auto generation:{1u,2u})
        {
            for(const std::size_t count:{64u,3u,0u,1u,16u})
            {
                std::vector<std::int64_t> ids(count);
                for(std::size_t i=0;i<count;++i)ids[i]=(i*7+generation)%37;
                std::fill_n(lane.bytes.get()+lane.layout.payloadOffset(),64*32,std::byte{0xa5});
                lane.publish(++sequence,generation,ids);
                ASSERT_EQ(lane.host.poll(generation),ImmutableLookupPoll::Completed);
                EXPECT_EQ(lane.host.poll(generation),ImmutableLookupPoll::Pending);
                const auto &response=lane.completion();
                EXPECT_EQ(ImmutableLookupProtocol::validateCompletion(response,response.publication,sequence,generation,17,count,32),ImmutableLookupStatus::Success);
                for(std::size_t i=0;i<count;++i)
                    EXPECT_EQ(std::memcmp(lane.bytes.get()+lane.layout.payloadOffset()+i*32,file.values.data()+ids[i]*8,32),0);
                EXPECT_TRUE(std::all_of(lane.bytes.get()+lane.layout.payloadOffset()+count*32,
                    lane.bytes.get()+lane.layout.payloadOffset()+64*32,[](std::byte b){return b==std::byte{0xa5};}));
                idsTotal+=count;lane.ack(sequence);
            }
            lane.close(generation);EXPECT_TRUE(lane.host.closed(generation));
        }
        const auto stats=lane.host.stats();EXPECT_EQ(stats.requests,sequence);EXPECT_EQ(stats.ids,idsTotal);
        EXPECT_EQ(stats.idBytes,idsTotal*8);EXPECT_EQ(stats.nativeBytes,idsTotal*32);
        EXPECT_EQ(stats.metadataBytes,sequence*(kLookupRequestLiveBytes+kLookupCompletionLiveBytes));
        EXPECT_EQ(stats.acknowledgementBytes,sequence*8);EXPECT_EQ(stats.terminalBytes,16u);EXPECT_EQ(stats.failures,0u);
    }
}

/** @test Every authenticated field rejects corruption before payload becomes consumable. */
TEST(ImmutableLookupProtocol, AuthenticationAndExactExtentRejectCorruption)
{
    FixtureFile file;const auto policy=ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::RAM,4096,2);
    auto memory=tableAuthority(ImmutableRowTable::physicalBytes(file.source(),policy));
    WholeProtocolSource table(file.source(),policy, DeviceId::cpu(),memory,ImmutableRowTableHostPlacement::taskPolicy());Lane lane(table);
    const std::array<std::int64_t,2> ids{1,2};lane.publish(1,7,ids);
    const auto valid=lane.request();
    auto check=[&](ImmutableLookupRequest bad,ImmutableLookupStatus status){
        EXPECT_EQ(ImmutableLookupProtocol::validateRequest(bad,1,7,17,0,64,32),status);
    };
    auto bad=valid;bad.protocolComplement^=1;check(bad,ImmutableLookupStatus::InvalidProtocol);
    bad=valid;bad.idCountComplement^=1;check(bad,ImmutableLookupStatus::InvalidProtocol);
    bad=valid;bad.generation=6;bad.generationComplement=~bad.generation;check(bad,ImmutableLookupStatus::StaleGeneration);
    bad=valid;bad.table=18;bad.tableComplement=~bad.table;check(bad,ImmutableLookupStatus::WrongTable);
    bad=valid;bad.idCount=65;bad.idCountComplement=~bad.idCount;check(bad,ImmutableLookupStatus::InvalidExtent);
    bad=valid;bad.idBytes=1;bad.idBytesComplement=~bad.idBytes;check(bad,ImmutableLookupStatus::InvalidExtent);
    EXPECT_EQ(ImmutableLookupProtocol::validateRequest(valid,1,7,17,1,64,32),ImmutableLookupStatus::InvalidSequence);
    lane.request().generation=6;lane.request().generationComplement=~std::uint64_t{6};
    EXPECT_EQ(lane.host.poll(7),ImmutableLookupPoll::Completed);EXPECT_EQ(lane.completion().nativeBytes,0u);
    EXPECT_EQ(lane.completion().status,static_cast<std::uint64_t>(ImmutableLookupStatus::StaleGeneration));
    EXPECT_EQ(lane.fatalStatus.load(std::memory_order_acquire),ImmutableLookupStatus::StaleGeneration);
    EXPECT_EQ(lane.host.stats().nativeBytes,0u);EXPECT_EQ(lane.host.stats().failures,1u);
    EXPECT_TRUE(std::all_of(lane.bytes.get()+lane.layout.payloadOffset(),lane.bytes.get()+lane.layout.payloadOffset()+64*32,
        [](std::byte b){return b==std::byte{0xa5};}));
}

/** @test Missing/future ack and premature close cannot free a still-live response slot. */
TEST(ImmutableLookupProtocol, AckCloseAndInvalidIdsFailClosed)
{
    FixtureFile file;const auto policy=ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::RAM,4096,2);
    auto memory=tableAuthority(ImmutableRowTable::physicalBytes(file.source(),policy));
    WholeProtocolSource table(file.source(),policy, DeviceId::cpu(),memory,ImmutableRowTableHostPlacement::taskPolicy());
    {
        Lane lane(table);const std::array<std::int64_t,1> id{0};lane.publish(1,1,id);ASSERT_EQ(lane.host.poll(1),ImmutableLookupPoll::Completed);
        lane.publish(2,1,id);EXPECT_THROW(lane.host.poll(1),std::runtime_error);
    }
    {
        Lane lane(table);const std::array<std::int64_t,1> id{0};lane.publish(1,1,id);ASSERT_EQ(lane.host.poll(1),ImmutableLookupPoll::Completed);
        lane.close(1);EXPECT_THROW(lane.host.closed(1),std::runtime_error);
        lane.ack(2);EXPECT_THROW(lane.host.closed(1),std::runtime_error);
    }
    {
        Lane lane(table);const std::array<std::int64_t,3> ids{0,37,1};lane.publish(1,1,ids);
        ASSERT_EQ(lane.host.poll(1),ImmutableLookupPoll::Completed);
        EXPECT_EQ(lane.completion().status,static_cast<std::uint64_t>(ImmutableLookupStatus::InvalidRowId));
        EXPECT_EQ(lane.completion().nativeBytes,0u);lane.ack(1);lane.close(1);EXPECT_TRUE(lane.host.closed(1));
        EXPECT_FALSE(lane.host.lastFailure().empty());
    }
    EXPECT_THROW(ImmutableLookupLayout::create(0,32),std::invalid_argument);
    EXPECT_THROW(ImmutableLookupLayout::create(std::numeric_limits<std::size_t>::max(),32),std::overflow_error);
}

/** @test Floating source bits and signed nonlinear native values decode without table conversion. */
TEST(ImmutableLookupProtocol, NativeDecoderPreservesOriginalValuesAndGroupedRows)
{
    const std::array<float,6> f32{-0.0f,1.25f,-9.5f,0.5f,17,-1};std::array<float,6> output{};
    decodeCPUImmutableLookupRows(TensorType::FP32,3,12,reinterpret_cast<const std::byte *>(f32.data()),2,output.data());
    EXPECT_EQ(std::memcmp(output.data(),f32.data(),sizeof(f32)),0);
    const std::array<std::uint16_t,4> half{0x8000,0x3c00,0xc000,0x0001};
    decodeCPUImmutableLookupRows(TensorType::FP16,2,4,reinterpret_cast<const std::byte *>(half.data()),2,output.data());
    EXPECT_TRUE(std::signbit(output[0]));EXPECT_EQ(output[1],1);EXPECT_EQ(output[2],-2);EXPECT_EQ(output[3],std::ldexp(1.0f,-24));
    const std::array<std::uint16_t,4> bf{0x8000,0x3f80,0xc000,0x0001};
    decodeCPUImmutableLookupRows(TensorType::BF16,2,4,reinterpret_cast<const std::byte *>(bf.data()),2,output.data());
    EXPECT_TRUE(std::signbit(output[0]));EXPECT_EQ(output[1],1);EXPECT_EQ(output[2],-2);EXPECT_EQ(output[3],std::ldexp(1.0f,-133));
    IQ4_NLBlock block{};block.d=0x3c00;
    for(int i=0;i<16;++i)block.qs[i]=static_cast<std::uint8_t>(i|(15-i)<<4);
    std::array<float,64> grouped{},serial{};std::array<IQ4_NLBlock,2> blocks{block,block};
    decodeCPUImmutableLookupRows(TensorType::IQ4_NL,32,sizeof(block),reinterpret_cast<const std::byte *>(blocks.data()),2,grouped.data());
    for(int row=0;row<2;++row)decodeCPUImmutableLookupRows(TensorType::IQ4_NL,32,sizeof(block),reinterpret_cast<const std::byte *>(&blocks[row]),1,serial.data()+row*32);
    EXPECT_EQ(std::memcmp(grouped.data(),serial.data(),sizeof(grouped)),0);
    constexpr std::array<int,16> expected{-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    for(int i=0;i<16;++i){EXPECT_EQ(grouped[i],expected[i]);EXPECT_EQ(grouped[16+i],expected[15-i]);}
    EXPECT_THROW(decodeCPUImmutableLookupRows(TensorType::FP32,3,11,nullptr,0,nullptr),std::invalid_argument);
}

/** @test SIMD widening preserves every original half/bfloat bit pattern, including NaN payloads. */
TEST(ImmutableLookupProtocol, FloatingSIMDExhaustsOriginalSixteenBitPatterns)
{
    std::vector<std::uint16_t> source(65536);std::vector<float> output(source.size());
    for(std::size_t i=0;i<source.size();++i)source[i]=static_cast<std::uint16_t>(i);
    for(const auto format:{TensorType::FP16,TensorType::BF16})
    {
        decodeCPUImmutableLookupRows(format,256,512,reinterpret_cast<const std::byte *>(source.data()),256,output.data());
        for(std::size_t i=0;i<source.size();++i)
        {
            const auto expected=format==TensorType::FP16?std::bit_cast<std::uint32_t>(native_embedding::half(source[i])):
                static_cast<std::uint32_t>(source[i])<<16;
            ASSERT_EQ(std::bit_cast<std::uint32_t>(output[i]),expected)<<"source="<<i<<" format="<<static_cast<int>(format);
        }
    }
}

/** @test Block SIMD retains all original IQ4 scales, signed source grid values and worker-independent bytes. */
TEST(ImmutableLookupProtocol, IQ4SIMDExhaustsScaleBitsAndSourceCodes)
{
    constexpr std::array<int,16> grid{-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    std::vector<IQ4_NLBlock> source(65536);std::vector<float> expected(source.size()*32),actual(expected.size());
    for(std::size_t block=0;block<source.size();++block)
    {
        source[block].d=static_cast<std::uint16_t>(block);
        const float scale=native_embedding::half(source[block].d);
        for(int c=0;c<16;++c)
        {
            source[block].qs[c]=static_cast<std::uint8_t>(c|((15-c)<<4));
            // Opaque register operands retain the real multiply for source
            // grid value +1 too; folding x*1 would leave signaling NaNs unquiet.
            expected[block*32+c]=device_fp32_contract::multiply(scale,device_fp32_contract::persistRounded(static_cast<float>(grid[c])));
            expected[block*32+16+c]=device_fp32_contract::multiply(scale,device_fp32_contract::persistRounded(static_cast<float>(grid[15-c])));
        }
    }
    const int previous=omp_get_max_threads();
    for(const int workers:{1,3})
    {
        omp_set_num_threads(workers);
        decodeCPUImmutableLookupRows(TensorType::IQ4_NL,32,18,reinterpret_cast<const std::byte *>(source.data()),source.size(),actual.data());
        EXPECT_EQ(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)),0)<<"workers="<<workers;
    }
    omp_set_num_threads(previous);
}

/** @test Terminal snapshots retain exact interval work and reject regressing authority counters. */
TEST(ImmutableLookupProtocol, TerminalStatisticsUseFreshIntervalExtents)
{
    const RetainedStorageCounters before{.requests=3,.ids=1000,.idBytes=8000,.nativeBytes=90000,
        .metadataBytes=672,.failures=0,.acknowledgementBytes=24,.terminalBytes=8};
    const RetainedStorageCounters after{.requests=5,.ids=1016,.idBytes=8128,.nativeBytes=91440,
        .metadataBytes=1120,.failures=1,.acknowledgementBytes=40,.terminalBytes=16};
    const auto observation=RetainedStorageTerminalObservation::between(2,HostImmutableRowMode::WholeMemoryMapped,before,after);
    EXPECT_EQ(observation.generation,2u);EXPECT_EQ(observation.mode,HostImmutableRowMode::WholeMemoryMapped);
    EXPECT_EQ(observation.counters.requests,2u);EXPECT_EQ(observation.counters.ids,16u);
    EXPECT_EQ(observation.counters.idBytes,128u);EXPECT_EQ(observation.counters.nativeBytes,1440u);
    EXPECT_EQ(observation.counters.metadataBytes,448u);EXPECT_EQ(observation.counters.failures,1u);
    EXPECT_EQ(observation.counters.acknowledgementBytes,16u);EXPECT_EQ(observation.counters.terminalBytes,8u);
    EXPECT_THROW(RetainedStorageTerminalObservation::between(3,HostImmutableRowMode::WholeRAM,after,before),std::logic_error);
    EXPECT_THROW(RetainedStorageTerminalObservation::between(0,HostImmutableRowMode::WholeRAM,before,after),std::invalid_argument);
    const auto empty=RetainedStorageTerminalObservation::between(3,HostImmutableRowMode::WholeRAM,after,after);
    EXPECT_EQ(empty.counters.nativeBytes,0u);EXPECT_EQ(empty.counters.requests,0u);
}

/** @test PerfStats reset cannot recount a lifetime total or replay a previously observed interval. */
TEST(ImmutableLookupProtocol, TerminalPerfStatsPreserveModeAndFreshnessAcrossReset)
{
    struct Environment
    {
        std::array<const char *,2> names{"LLAMINAR_PERF_STATS_JSON","LLAMINAR_PERF_STATS_FILTER"};
        std::array<std::optional<std::string>,2> previous;
        /** @brief Enable only this evidence family before test observations. */
        Environment()
        {
            for(std::size_t i=0;i<names.size();++i)if(const char *v=std::getenv(names[i]))previous[i]=v;
            ::setenv(names[0],"1",1);::setenv(names[1],"immutable_lookup",1);PerfStatsCollector::reset();
        }
        /** @brief Restore the process diagnostic configuration after this device-free proof. */
        ~Environment()
        {
            for(std::size_t i=0;i<names.size();++i)
                if(previous[i])::setenv(names[i],previous[i]->c_str(),1);else ::unsetenv(names[i]);
            PerfStatsCollector::reset();
        }
    } environment;
    RetainedStoragePerfStats evidence;
    const RetainedStorageCounters first{.requests=2,.ids=240,.idBytes=1920,.nativeBytes=21600,
        .metadataBytes=448,.failures=0,.acknowledgementBytes=16,.terminalBytes=8};
    evidence.publish(RetainedStorageTerminalObservation::between(1,HostImmutableRowMode::WholeRAM,{},first),"cuda:0");
    auto records=PerfStatsCollector::snapshot({"immutable_lookup"});ASSERT_EQ(records.size(),12u);
    const auto find=[&](const char *name)->const PerfStatRecord &
    {
        const auto p=std::find_if(records.begin(),records.end(),[&](const auto &r){return r.name==name;});
        if(p==records.end())throw std::runtime_error("Missing terminal storage evidence");return *p;
    };
    EXPECT_EQ(find("native_bytes").value,21600);EXPECT_EQ(find("ids").value,240);
    EXPECT_EQ(find("intervals").value,1);EXPECT_EQ(find("requests").tags.at("storage"),"ram");
    EXPECT_EQ(find("ids").sequence_minimum_words,(std::vector<std::uint64_t>{1,240}));
    PerfStatsCollector::reset();
    EXPECT_THROW(evidence.publish(RetainedStorageTerminalObservation::between(1,HostImmutableRowMode::WholeRAM,{},first),"cuda:0"),std::logic_error);
    EXPECT_TRUE(PerfStatsCollector::snapshot({"immutable_lookup"}).empty());
    auto total=first;total.requests+=1;total.ids+=16;total.idBytes+=128;total.nativeBytes+=1440;
    total.metadataBytes+=224;total.acknowledgementBytes+=8;total.terminalBytes+=8;
    evidence.publish(RetainedStorageTerminalObservation::between(2,HostImmutableRowMode::WholeMemoryMapped,first,total),"cuda:0");
    records=PerfStatsCollector::snapshot({"immutable_lookup"});ASSERT_EQ(records.size(),12u);
    EXPECT_EQ(find("native_bytes").value,1440);EXPECT_EQ(find("requests").value,1);
    EXPECT_EQ(find("requests").tags.at("storage"),"mmap");EXPECT_EQ(find("terminal_bytes").value,8);
    EXPECT_EQ(find("ids").sequence_minimum_words,(std::vector<std::uint64_t>{2,16}));
}

namespace
{
/** @brief Three distinct file windows; native rows never cross a page boundary. */
struct WindowFile
{
    /** @brief Publish finite original rows before their immutable source is opened. */
    WindowFile()
    {
        char pattern[]="/tmp/qfn-window-observation-XXXXXX";fd=::mkstemp(pattern);
        if(fd<0)throw std::runtime_error("Window observation fixture creation failed");path=pattern;
        for(std::size_t i=0;i<values.size();++i)values[i]=static_cast<float>(i)/32;
        if(::write(fd,values.data(),sizeof(values))!=sizeof(values))
        {::close(fd);::unlink(path.c_str());throw std::runtime_error("Window observation fixture write failed");}
    }
    /** @brief Remove this source after all ordinary table/mapping owners retire. */
    ~WindowFile(){::close(fd);::unlink(path.c_str());}
    /** @return Original32-byte F32 rows in three separate4KiB windows. */
    ImmutableRowTableSource source() const
    {return {.path=path,.tensor_name="window.fixture",.rows=257,.row_elements=8,.format=GGUFTensorType::F32};}
    int fd=-1;std::string path;std::array<float,257*8> values{};
};
/** @brief Scope only the existing bounded immutable-lookup PerfStats domain. */
struct WindowEvidence
{
    std::array<const char *,2> names{"LLAMINAR_PERF_STATS_JSON","LLAMINAR_PERF_STATS_FILTER"};
    std::array<std::optional<std::string>,2> previous;
    /** @brief Enable records before the actual completed table/service work. */
    WindowEvidence()
    {
        for(std::size_t i=0;i<names.size();++i)if(const char *v=std::getenv(names[i]))previous[i]=v;
        ::setenv(names[0],"1",1);::setenv(names[1],"immutable_lookup",1);PerfStatsCollector::reset();
    }
    /** @brief Restore configuration and discard only this fixture's observations. */
    ~WindowEvidence()
    {
        for(std::size_t i=0;i<names.size();++i)
            if(previous[i])::setenv(names[i],previous[i]->c_str(),1);else ::unsetenv(names[i]);
        PerfStatsCollector::reset();
    }
};
}

/** @test Per-call windows describe successful mapping resolutions, not lifetime totals or OS reads. */
TEST(ImmutableLookupProtocol, WindowGatherObservationsDistinguishRAMAndMmapWork)
{
    WindowFile file;
    for(auto mode:{ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped})
    {
        const auto policy=ImmutableRowTableStoragePolicy::forStorage(mode,4096,1);
        WholeProtocolSource table(file.source(),policy,DeviceId::cpu(),tableAuthority(ImmutableRowTable::physicalBytes(file.source(),policy)),ImmutableRowTableHostPlacement::taskPolicy());
        const std::array<std::int64_t,5> ids{0,0,128,128,256};std::array<std::byte,5*32> destination;
        const auto first=table.gather(ids,destination);
        EXPECT_EQ(first.windowHits,mode==ImmutableRowStorage::RAM?0u:2u);
        EXPECT_EQ(first.windowMisses,mode==ImmutableRowStorage::RAM?0u:3u);
        EXPECT_EQ(first.windowEvictions,mode==ImmutableRowStorage::RAM?0u:2u);
        const std::array<std::int64_t,1> last{256};std::array<std::byte,32> one;
        const auto next=table.gather(last,one);
        EXPECT_EQ(next.windowHits,mode==ImmutableRowStorage::RAM?0u:1u);
        EXPECT_EQ(next.windowMisses,0u);EXPECT_EQ(next.windowEvictions,0u);
        const auto empty=table.gather({},{});
        EXPECT_EQ(empty.windowHits,0u);EXPECT_EQ(empty.windowMisses,0u);EXPECT_EQ(empty.windowEvictions,0u);
        const auto lifetime=table.stats();
        EXPECT_EQ(lifetime.window_hits,first.windowHits+next.windowHits);
        EXPECT_EQ(lifetime.window_misses,first.windowMisses+next.windowMisses);
        EXPECT_EQ(lifetime.window_evictions,first.windowEvictions+next.windowEvictions);
        EXPECT_EQ(lifetime.rows,6u);EXPECT_EQ(lifetime.copied_bytes,6u*32);
    }
}

/** @test Concurrent independent host lanes charge each successful mapping resolution once. */
TEST(ImmutableLookupProtocol, WindowConcurrentSharedTableWorkBelongsToExactlyOneService)
{
    WindowFile file;const auto policy=ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped,4096,1);
    WholeProtocolSource table(file.source(),policy,DeviceId::cpu(),tableAuthority(ImmutableRowTable::physicalBytes(file.source(),policy)),ImmutableRowTableHostPlacement::taskPolicy());
    Lane a(table),b(table);std::barrier start(3);std::array<std::exception_ptr,2> failures{};
    const auto serve=[&](Lane &lane,int index,std::int64_t row)
    {
        start.arrive_and_wait();
        try
        {
            std::array<std::int64_t,8> ids;ids.fill(row);
            for(std::uint64_t sequence=1;sequence<=40;++sequence)
            {
                lane.publish(sequence,1,ids);
                if(lane.host.poll(1)!=ImmutableLookupPoll::Completed)throw std::runtime_error("Missing concurrent response");
                if(lane.host.poll(1)!=ImmutableLookupPoll::Pending)throw std::runtime_error("Response processed twice");
                lane.ack(sequence);
            }
            lane.close(1);
            if(!lane.host.closed(1))throw std::runtime_error("Missing concurrent terminal close");
        }
        catch(...){failures[index]=std::current_exception();}
    };
    std::thread first(serve,std::ref(a),0,0),second(serve,std::ref(b),1,128);
    start.arrive_and_wait();first.join();second.join();
    for(const auto &failure:failures)if(failure)std::rethrow_exception(failure);
    const auto sa=a.host.stats(),sb=b.host.stats();const auto global=table.stats();
    EXPECT_EQ(sa.requests,40u);EXPECT_EQ(sb.requests,40u);
    EXPECT_EQ(sa.windowHits+sa.windowMisses,320u);EXPECT_EQ(sb.windowHits+sb.windowMisses,320u);
    EXPECT_EQ(sa.windowHits+sb.windowHits,global.window_hits);
    EXPECT_EQ(sa.windowMisses+sb.windowMisses,global.window_misses);
    EXPECT_EQ(sa.windowEvictions+sb.windowEvictions,global.window_evictions);
    EXPECT_EQ(sa.nativeBytes+sb.nativeBytes,global.copied_bytes);
    EXPECT_EQ(global.requests,80u);EXPECT_GT(global.window_evictions,0u);
}

/** @test Reset and duplicate terminals cannot replay table lifetime or another lane's windows. */
TEST(ImmutableLookupProtocol, WindowTerminalIntervalsResetWithoutRecountingOtherServices)
{
    WindowEvidence environment;WindowFile file;
    EXPECT_THROW(recordCPUImmutableLookup(HostImmutableRowMode::WholeRAM,DeviceId::cpu(),1,32,
        RetainedStorageCPUOutcome::Completed,WholeImmutableRowGatherObservation{ImmutableRowStorage::RAM,1,32,{1,0,0}}),std::invalid_argument);
    EXPECT_THROW(recordCPUImmutableLookup(HostImmutableRowMode::WholeMemoryMapped,DeviceId::cpu(),0,0,
        RetainedStorageCPUOutcome::Failed,WholeImmutableRowGatherObservation{ImmutableRowStorage::MemoryMapped,0,0,{0,1,0}}),std::invalid_argument);
    const auto policy=ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped,4096,1);
    WholeProtocolSource table(file.source(),policy,DeviceId::cpu(),tableAuthority(ImmutableRowTable::physicalBytes(file.source(),policy)),ImmutableRowTableHostPlacement::taskPolicy());
    Lane a(table),b(table);const std::array<std::int64_t,2> zero{0,0},middle{128,128},last{256,256};
    a.publish(1,1,zero);ASSERT_EQ(a.host.poll(1),ImmutableLookupPoll::Completed);a.ack(1);a.close(1);ASSERT_TRUE(a.host.closed(1));
    b.publish(1,1,middle);ASSERT_EQ(b.host.poll(1),ImmutableLookupPoll::Completed);b.ack(1);b.close(1);ASSERT_TRUE(b.host.closed(1));
    RetainedStoragePerfStats pa,pb;
    const auto before=a.host.stats();
    const auto first=RetainedStorageTerminalObservation::between(1,HostImmutableRowMode::WholeMemoryMapped,{},before);
    pa.publish(first,"CUDA:0");
    pb.publish(RetainedStorageTerminalObservation::between(1,HostImmutableRowMode::WholeMemoryMapped,{},b.host.stats()),"ROCm:0");
    auto records=PerfStatsCollector::snapshot({"immutable_lookup"});
    const auto total=[&](const char *name)
    {double value=0;for(const auto &record:records)if(record.name==name)value+=record.value;return value;};
    EXPECT_EQ(total("window_hits"),2);EXPECT_EQ(total("window_misses"),2);EXPECT_EQ(total("window_evictions"),1);
    PerfStatsCollector::reset();EXPECT_THROW(pa.publish(first,"CUDA:0"),std::logic_error);
    EXPECT_TRUE(PerfStatsCollector::snapshot({"immutable_lookup"}).empty());
    a.publish(2,2,last);ASSERT_EQ(a.host.poll(2),ImmutableLookupPoll::Completed);a.ack(2);a.close(2);ASSERT_TRUE(a.host.closed(2));
    const auto second=RetainedStorageTerminalObservation::between(2,HostImmutableRowMode::WholeMemoryMapped,before,a.host.stats());
    pa.publish(second,"CUDA:0");records=PerfStatsCollector::snapshot({"immutable_lookup"});ASSERT_EQ(records.size(),12u);
    EXPECT_EQ(total("window_hits"),1);EXPECT_EQ(total("window_misses"),1);EXPECT_EQ(total("window_evictions"),1);
    EXPECT_EQ(total("native_bytes"),64);EXPECT_EQ(total("requests"),1);
    for(const auto &record:records)if(record.name=="window_hits")
        EXPECT_EQ(record.sequence_minimum_words,(std::vector<std::uint64_t>{2,1}));
    auto regressed=a.host.stats();regressed.windowHits=0;
    EXPECT_THROW(RetainedStorageTerminalObservation::between(3,HostImmutableRowMode::WholeMemoryMapped,a.host.stats(),regressed),std::logic_error);
    const auto prior=a.host.stats();
    a.publish(3,3,{});ASSERT_EQ(a.host.poll(3),ImmutableLookupPoll::Completed);a.ack(3);a.close(3);ASSERT_TRUE(a.host.closed(3));
    const auto empty=RetainedStorageTerminalObservation::between(3,HostImmutableRowMode::WholeMemoryMapped,prior,a.host.stats());
    EXPECT_EQ(empty.counters.windowHits,0u);EXPECT_EQ(empty.counters.windowMisses,0u);EXPECT_EQ(empty.counters.windowEvictions,0u);
    EXPECT_EQ(empty.counters.nativeBytes,0u);
    PerfStatsCollector::reset();pa.publish(empty,"CUDA:0");records=PerfStatsCollector::snapshot({"immutable_lookup"});
    EXPECT_EQ(total("window_hits"),0);EXPECT_EQ(total("window_misses"),0);EXPECT_EQ(total("window_evictions"),0);
    EXPECT_EQ(total("requests"),1);EXPECT_EQ(total("native_bytes"),0);
}

/**
 * @test A failed partial gather retains actual table work but authors no successful service observation.
 *
 * Only the death-test child limits its address space after warming one of two
 * slots and preallocating the channel. Cached row0 copies first; mapping the
 * next4KiB window must fail under RLIMIT_AS. The limit is restored before
 * assertions and ordinary owner destruction. No production fault hook or OS
 * physical-I/O claim is introduced; this is an actual mapping-failure boundary.
 */
TEST(ImmutableLookupProtocol, WindowFailedPartialGatherDoesNotClaimZeroTableWork)
{
    // Re-exec a fresh device-free test process rather than inheriting locks
    // from any earlier CPU worker/logging threads in this shared fixture.
    struct DeathStyle
    {
        std::string previous=GTEST_FLAG_GET(death_test_style);
        /** @brief Isolate the process-wide resource limit in a fresh child. */
        DeathStyle(){GTEST_FLAG_SET(death_test_style,"threadsafe");}
        /** @brief Restore the caller's death-test policy even after failure. */
        ~DeathStyle(){GTEST_FLAG_SET(death_test_style,previous);}
    } death_style;
    ASSERT_EXIT(([]
    {
        const auto verify=[]
        {
            WindowFile file;const auto policy=ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped,4096,2);
            WholeProtocolSource table(file.source(),policy,DeviceId::cpu(),tableAuthority(ImmutableRowTable::physicalBytes(file.source(),policy)),ImmutableRowTableHostPlacement::taskPolicy());
            std::array<std::byte,32> warm;const std::array<std::int64_t,1> first{0};
            (void)table.gather(first,warm);Lane lane(table);
            const std::array<std::int64_t,2> ids{0,128};lane.publish(1,1,ids);
            rlimit saved{};
            if(::getrlimit(RLIMIT_AS,&saved)!=0)throw std::runtime_error("Cannot inspect child RLIMIT_AS");
            auto *file_state=std::fopen("/proc/self/statm","r");unsigned long long pages=0;
            if(!file_state)throw std::runtime_error("Cannot inspect child virtual extent");
            const bool parsed=std::fscanf(file_state,"%llu",&pages)==1;std::fclose(file_state);
            const auto page_bytes=::sysconf(_SC_PAGESIZE);
            if(!parsed||page_bytes<=0||pages>std::numeric_limits<rlim_t>::max()/static_cast<unsigned long long>(page_bytes))
                throw std::runtime_error("Invalid child virtual extent");
            rlimit bounded=saved;bounded.rlim_cur=static_cast<rlim_t>(pages)*static_cast<rlim_t>(page_bytes);
            if(::setrlimit(RLIMIT_AS,&bounded)!=0)throw std::runtime_error("Cannot bound child virtual extent");
            ImmutableLookupPoll result;
            try{result=lane.host.poll(1);}
            catch(...){if(::setrlimit(RLIMIT_AS,&saved)!=0)std::terminate();throw;}
            if(::setrlimit(RLIMIT_AS,&saved)!=0)std::terminate();
            EXPECT_EQ(result,ImmutableLookupPoll::Completed);
            EXPECT_EQ(lane.host.lastStatus(),ImmutableLookupStatus::StorageFailure);
            const auto global=table.stats();const auto service=lane.host.stats();
            EXPECT_EQ(global.requests,1u);EXPECT_EQ(global.copied_bytes,32u); // warm success only
            EXPECT_EQ(global.window_hits,1u);EXPECT_EQ(global.window_misses,2u); // includes failed call
            EXPECT_EQ(service.requests,1u);EXPECT_EQ(service.failures,1u);EXPECT_EQ(service.nativeBytes,0u);
            EXPECT_EQ(service.windowHits,0u);EXPECT_EQ(service.windowMisses,0u);EXPECT_EQ(service.windowEvictions,0u);
            EXPECT_EQ(std::memcmp(lane.bytes.get()+lane.layout.payloadOffset(),file.values.data(),32),0);
            EXPECT_TRUE(std::all_of(lane.bytes.get()+lane.layout.payloadOffset()+32,
                lane.bytes.get()+lane.layout.payloadOffset()+64,[](std::byte b){return b==std::byte{0xa5};}));
        };
        verify();::_exit(::testing::Test::HasFailure()?1:0);
    }()),::testing::ExitedWithCode(0),"");
}

namespace {
/** @return One source-authorized provider shared across independent hosted lanes and a direct CPU client. */
std::shared_ptr<HostImmutableRowProvider> adaptiveProtocolProvider(const FixtureFile &file,std::size_t slots)
{
    const PhysicalMemoryResource host{0,DeviceId::cpu(),1u<<20,1u<<20};const PhysicalMemoryAllocatorIdentity id{0,DeviceId::cpu()};
    const std::array tiers{ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped};
    const auto src=file.source();const auto layout=ImmutableRowTableLayoutGeometry::checked(
        ImmutableRowTableExtent::fromNative(src.format,src.rows,src.rowBytes()),ImmutableRowTableTiers::of(tiers),
        ImmutableRowTableResidency::Adaptive,{{{id,ImmutableRowStorage::RAM},0,slots,{}}},
        {id,{0,DeviceId::cuda(0)},{0,DeviceId::rocm(0)}},{},
        {{id,ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped,4096,2)}},{},std::nullopt,64);
    const auto geometry=CpuAdaptiveImmutableRowCacheGeometry::fromLayout(layout,id,64);
    auto memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(geometry.bom(host)),0);
    return HostImmutableRowProvider::fromAdaptiveCache(CpuAdaptiveImmutableRowCache::create(src,geometry,memory));
}
}
/** @test Each hosted lane reports its own native cache work even when channel sequences coincide. */
TEST(ImmutableLookupProtocol, AdaptiveSharedProviderKeepsActualPerLaneAndCpuWorkSeparate)
{
    FixtureFile file;
    for(const std::size_t slots:{0u,2u})
    {
        auto provider=adaptiveProtocolProvider(file,slots);Lane a(provider),b(provider);
        const std::array<std::int64_t,4> ids{3,4,3,4};
        const auto before=provider->retainedOwnerObservation();
        a.publish(1,1,ids);ASSERT_EQ(a.host.poll(1),ImmutableLookupPoll::Completed);a.ack(1);
        b.publish(1,1,ids);ASSERT_EQ(b.host.poll(1),ImmutableLookupPoll::Completed);b.ack(1);
        const auto sa=a.host.stats(),sb=b.host.stats();
        EXPECT_EQ(sa.adaptiveBatches,1u);EXPECT_EQ(sb.adaptiveBatches,1u);
        EXPECT_EQ(sa.ids,4u);EXPECT_EQ(sb.ids,4u);EXPECT_EQ(sa.nativeBytes,128u);EXPECT_EQ(sb.nativeBytes,128u);
        EXPECT_EQ(sa.sourceRows,2u);EXPECT_EQ(sa.ramRows,0u);EXPECT_EQ(sa.duplicateRows,2u);
        EXPECT_EQ(sb.sourceRows,slots?0u:2u);EXPECT_EQ(sb.ramRows,slots?2u:0u);EXPECT_EQ(sb.duplicateRows,2u);
        for(auto *lane:{&a,&b})for(std::size_t i=0;i<ids.size();++i)
            EXPECT_EQ(std::memcmp(lane->bytes.get()+lane->layout.payloadOffset()+i*32,file.values.data()+ids[i]*8,32),0);
        std::array<std::byte,128> response;
        const auto cpu=std::get<CpuImmutableRowGatherObservation>(provider->gather(ids,response));
        EXPECT_EQ(cpu.policy_ordinal,2u);EXPECT_EQ(cpu.rows,4u);EXPECT_EQ(cpu.duplicate_rows,2u);
        // Neither the direct CPU client nor another lane can republish its work
        // into this lane's successful observation or terminal interval.
        EXPECT_EQ(a.host.stats().adaptiveBatches,1u);EXPECT_EQ(b.host.stats().adaptiveBatches,1u);
        a.close(1);b.close(1);ASSERT_TRUE(a.host.closed(1));ASSERT_TRUE(b.host.closed(1));
        const auto interval=RetainedStorageTerminalObservation::between(1,provider->mode(),{},a.host.stats());
        EXPECT_EQ(interval.mode,HostImmutableRowMode::AdaptiveRAMSSD);EXPECT_EQ(interval.counters.ids,4u);
        const auto beforeSecond=a.host.stats();
        a.publish(2,2,{});ASSERT_EQ(a.host.poll(2),ImmutableLookupPoll::Completed);a.ack(2);a.close(2);ASSERT_TRUE(a.host.closed(2));
        const auto empty=RetainedStorageTerminalObservation::between(2,provider->mode(),beforeSecond,a.host.stats());
        EXPECT_EQ(empty.counters.terminalBytes,8u);
        EXPECT_EQ(empty.counters.adaptiveBatches,1u);EXPECT_EQ(empty.counters.ids,0u);EXPECT_EQ(empty.counters.nativeBytes,0u);
        EXPECT_EQ(empty.counters.sourceRows+empty.counters.ramRows+empty.counters.duplicateRows,0u);
        const auto after=provider->retainedOwnerObservation();EXPECT_EQ(after.authority(),before.authority());
        EXPECT_EQ(after.tableStorageBytes(),before.tableStorageBytes());EXPECT_EQ(after.lookupStagingBytes(),before.lookupStagingBytes());
    }
}
/** @test Invalid mode/row work cannot be erased to whole mmap, partially accumulated or emitted as CPU evidence. */
TEST(ImmutableLookupProtocol, AdaptiveObservationModeAndPartitionRejectWithoutCounterMutation)
{
    RetainedStorageCounters total;CpuImmutableRowGatherObservation actual;
    actual.rows=4;actual.bytes=128;actual.source_rows=2;actual.duplicate_rows=2;
    accumulateHostImmutableRowWork(total,HostImmutableRowMode::AdaptiveRAMSSD,actual,4,128);
    EXPECT_EQ(total.sourceRows,2u);EXPECT_EQ(total.duplicateRows,2u);EXPECT_EQ(total.adaptiveBatches,1u);
    EXPECT_THROW(accumulateHostImmutableRowWork(total,HostImmutableRowMode::WholeMemoryMapped,actual,4,128),std::invalid_argument);
    ++actual.ram_rows;
    EXPECT_THROW(accumulateHostImmutableRowWork(total,HostImmutableRowMode::AdaptiveRAMSSD,actual,4,128),std::invalid_argument);
    EXPECT_EQ(total.sourceRows,2u);EXPECT_EQ(total.duplicateRows,2u);EXPECT_EQ(total.adaptiveBatches,1u);
    --actual.ram_rows;actual.bytes=64;
    EXPECT_THROW(accumulateHostImmutableRowWork(total,HostImmutableRowMode::AdaptiveRAMSSD,actual,4,128),std::invalid_argument);
    total.adaptiveBatches=std::numeric_limits<std::uint64_t>::max();actual.bytes=128;
    EXPECT_THROW(accumulateHostImmutableRowWork(total,HostImmutableRowMode::AdaptiveRAMSSD,actual,4,128),std::overflow_error);
    EXPECT_EQ(total.sourceRows,2u);EXPECT_EQ(total.duplicateRows,2u);
}

/** @test Protocol2 source evidence stays distinct and invalid projections do
 * not consume freshness. */
TEST(ImmutableLookupProtocol, AdaptiveSubordinateTransportIsExactAndDoesNotEraseSourceMode)
{
    struct Environment final
    {
        std::optional<std::string> json, filter;
        /** @brief Retain process diagnostics before enabling this one evidence
         * family. */
        Environment()
        {
            if (const char *value = std::getenv("LLAMINAR_PERF_STATS_JSON"))
                json = value;
            if (const char *value = std::getenv("LLAMINAR_PERF_STATS_FILTER"))
                filter = value;
            ::setenv("LLAMINAR_PERF_STATS_JSON", "1", 1);
            ::setenv("LLAMINAR_PERF_STATS_FILTER", "immutable_lookup", 1);
            PerfStatsCollector::reset();
        }
        /** @brief Restore original measurement policy after immutable observation
         * checks. */
        ~Environment()
        {
            if (json)
                ::setenv("LLAMINAR_PERF_STATS_JSON", json->c_str(), 1);
            else
                ::unsetenv("LLAMINAR_PERF_STATS_JSON");
            if (filter)
                ::setenv("LLAMINAR_PERF_STATS_FILTER", filter->c_str(), 1);
            else
                ::unsetenv("LLAMINAR_PERF_STATS_FILTER");
            PerfStatsCollector::reset();
        }
    } environment;
    RetainedStoragePerfStats stats;
    auto observed = RetainedStorageTerminalObservation::between(
        1, HostImmutableRowMode::AdaptiveRAMSSD, {},
        {.requests = 1, .ids = 2, .idBytes = 16, .nativeBytes = 180});
    observed.transport = static_cast<RetainedStorageTransport>(255);
    EXPECT_THROW(stats.publish(observed, "CUDA:0"), std::invalid_argument);
    observed.transport = RetainedStorageTransport::AdaptiveSubordinate;
    observed.mode = HostImmutableRowMode::WholeMemoryMapped;
    EXPECT_THROW(stats.publish(observed, "CUDA:0"), std::invalid_argument);
    EXPECT_TRUE(PerfStatsCollector::snapshot({"immutable_lookup"}).empty());
    observed.mode = HostImmutableRowMode::AdaptiveRAMSSD;
    stats.publish(observed, "CUDA:0");
    const auto records = PerfStatsCollector::snapshot({"immutable_lookup"});
    ASSERT_FALSE(records.empty());
    for (const auto &record : records)
    {
        EXPECT_EQ(record.tags.at("transport"), "adaptive_subordinate");
        EXPECT_EQ(record.tags.at("storage"), hostImmutableRowModeName(HostImmutableRowMode::AdaptiveRAMSSD));
    }
    PerfStatsCollector::reset();
    EXPECT_THROW(stats.publish(observed, "CUDA:0"), std::logic_error);
    EXPECT_TRUE(PerfStatsCollector::snapshot({"immutable_lookup"}).empty());
    observed.generation = 2;
    observed.counters = {};
    stats.publish(observed, "CUDA:0");
    for (const auto &record : PerfStatsCollector::snapshot({"immutable_lookup"}))
        if (record.name != "intervals")
        {
            EXPECT_EQ(record.value, 0.0);
        }
}
