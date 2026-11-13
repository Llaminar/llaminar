/**
 * @file Test__ImmutableLookupStage.cpp
 * @brief Real CPU RAM/mmap stage regression for live native row extents and replay.
 *
 * The fixture routes poisoned-capacity token batches through the production
 * stage and table authority. Large-to-small and empty replay must preserve
 * untouched destination bytes while retaining exact table identity and format.
 */
#include "execution/compute_stages/stages/qwen4exp/ImmutableLookupStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "tensors/Tensors.h"
#include "utils/OpenMPUtils.h"
#include "utils/PerfStatsCollector.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <optional>
#include <unistd.h>

using namespace llaminar2;
namespace
{
/** @brief Scope only immutable lookup evidence, preserving the caller's diagnostic environment. */
struct Evidence
{
    std::array<const char *,2> names{"LLAMINAR_PERF_STATS_JSON","LLAMINAR_PERF_STATS_FILTER"};
    std::array<std::optional<std::string>,2> previous;
    /** @brief Enable exact native I/O counters before executing the production CPU stage. */
    Evidence()
    {
        for(std::size_t i=0;i<names.size();++i)if(const char *value=std::getenv(names[i]))previous[i]=value;
        ::setenv(names[0],"1",1);::setenv(names[1],"immutable_lookup",1);PerfStatsCollector::reset();
    }
    /** @brief Restore process-wide collection policy and discard this fixture's observations. */
    ~Evidence()
    {
        for(std::size_t i=0;i<names.size();++i)
            if(previous[i])::setenv(names[i],previous[i]->c_str(),1);else ::unsetenv(names[i]);
        PerfStatsCollector::reset();
    }
};
/** @brief Small independently owned native table, removed after all table views retire. */
struct Source
{
    Source()
    {
        char pattern[]="/tmp/qfn-lookup-stage-XXXXXX";
        fd=::mkstemp(pattern);
        if(fd<0) throw std::runtime_error("Cannot create native table fixture");
        path=pattern;
        for(std::size_t i=0;i<values.size();++i) values[i]=(static_cast<int>(i)-100)/32.0f;
        if(::write(fd,values.data(),sizeof(values))!=sizeof(values))
        {
            ::close(fd);::unlink(path.c_str());
            throw std::runtime_error("Cannot write native table fixture");
        }
    }
    ~Source() { ::close(fd);::unlink(path.c_str()); }
    /** @brief Exact original F32 row coordinates; native format is independent of the file label. */
    ImmutableRowTableSource source() const
    {
        return {.path=path,.tensor_name="per_layer_token_embd.weight",.rows=37,.row_elements=8,.format=GGUFTensorType::F32};
    }
    /** @brief Admit only this table's exact physical demand. */
    std::shared_ptr<HostImmutableRowProvider> open(ImmutableRowStorage storage) const
    {
        const auto policy=ImmutableRowTableStoragePolicy::forStorage(storage,4096,2);
        PhysicalMemoryPlanBuilder plan;
        plan.add({.world_rank=0,.device=DeviceId::cpu(),.total_bytes=1u<<20,.admission_available_bytes=1u<<20},
            PhysicalMemoryOwner::ImmutableTableStorage,ImmutableRowTable::physicalBytes(source(),policy));
        auto authority=std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
        return HostImmutableRowProvider::fromWholeTable(std::make_unique<ImmutableRowTable>(
            source(),policy, DeviceId::cpu(),authority,ImmutableRowTableHostPlacement::taskPolicy()));
    }
    /** @return One actual CPU RAM/SSD owner; zero slots retains its exact mandatory SSD/scratch claims. */
    std::shared_ptr<HostImmutableRowProvider> openAdaptive(std::size_t slots,std::size_t maximum_ids) const
    {
        const PhysicalMemoryResource host{0,DeviceId::cpu(),1u<<20,1u<<20};
        const PhysicalMemoryAllocatorIdentity owner{0,DeviceId::cpu()};
        const std::array tiers{ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped};
        const auto layout=ImmutableRowTableLayoutGeometry::checked(
            ImmutableRowTableExtent::fromNative(source().format,source().rows,source().rowBytes()),
            ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,
            {{{owner,ImmutableRowStorage::RAM},0,slots,{}}},{owner},{},
            {{owner,ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped,4096,2)}},
            {},std::nullopt,maximum_ids);
        const auto geometry=CpuAdaptiveImmutableRowCacheGeometry::fromLayout(layout,owner,maximum_ids);
        auto memory=std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(geometry.bom(host)),0);
        return HostImmutableRowProvider::fromAdaptiveCache(CpuAdaptiveImmutableRowCache::create(source(),geometry,memory));
    }
    int fd;
    std::string path;
    std::array<float,37*8> values;
};
}
/** @test Native I/O/decode touches only live IDs and cannot alter inactive output rows. */
TEST(ImmutableLookupStage, RAMAndMmapRetainedLiveRowsUseTheSameNativeSource)
{
    llaminar::v2::ThreadCountGuard team(1);
    CPUDeviceContext context(DeviceId::cpu(),1);
    Source source;
    for(auto storage:{ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped})
    {
        auto table=source.open(storage);
        constexpr int rows=5,heads=16,width=8;
        INT32Tensor ids({rows*heads*2}),native({rows*heads*width}),control({1});
        FP32Tensor output({rows,heads*width},DeviceId::cpu());
        auto *indices=reinterpret_cast<std::int64_t *>(ids.raw_mutable_data());
        auto *live=static_cast<std::int32_t *>(control.raw_mutable_data());
        ImmutableLookupStage stage({.rows={rows,Qwen4ExpIntBinding(control,BufferId::PREFILL_CHUNK_GEOMETRY)},
            .ids={ids,BufferId::QFN_NGRAM_ROW_IDS},
            .output={output,BufferId::QFN_PLE_EMBEDDING},.format=TensorType::FP32,.rowElements=width,
            .rowBytes=width*sizeof(float),.storage=ImmutableLookupStage::CPUStorage{table,{native,BufferId::QFN_NGRAM_NATIVE_ROWS}}});
        std::uint64_t requested=0;
        for(int count:{5,1,0,3,2,5})
        {
            *live=count;
            std::fill_n(indices,rows*heads,-99);
            for(int i=0;i<count*heads;++i) indices[i]=(i*11+count)%37;
            std::fill_n(output.mutable_data(),output.numel(),-9999.0f);
            std::memset(native.raw_mutable_data(),0xa5,native.size_bytes());
            ASSERT_TRUE(stage.execute(&context));
            for(int i=0;i<count*heads;++i)
                for(int j=0;j<width;++j)
                    EXPECT_EQ(output.data()[i*width+j],source.values[indices[i]*width+j]);
            for(std::size_t i=count*heads*width;i<output.numel();++i) EXPECT_EQ(output.data()[i],-9999.0f);
            const auto *bytes=static_cast<const unsigned char *>(native.raw_data());
            EXPECT_TRUE(std::all_of(bytes+count*heads*width*sizeof(float),bytes+native.size_bytes(),
                [](unsigned char value){return value==0xa5;}));
            requested+=count*heads;
        }
        EXPECT_EQ(table->sourceStats().rows,requested);
        EXPECT_EQ(table->sourceStats().copied_bytes,requested*width*sizeof(float));
        *live=rows+1;
        EXPECT_THROW(stage.execute(&context),std::invalid_argument);
    }
}
/** @test CPU production counters prove native RAM/mmap work, live extents, failure and reset freshness. */
TEST(ImmutableLookupStage, CPUPerfStatsPublishExactNativeWorkAndStorageMode)
{
    Evidence evidence;
    llaminar::v2::ThreadCountGuard team(1);CPUDeviceContext context(DeviceId::cpu(),1);Source source;
    for(const auto mode:{ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped})
    {
        PerfStatsCollector::reset();auto table=source.open(mode);
        constexpr int capacity=5,heads=16,width=8;
        INT32Tensor ids({capacity*heads*2}),native({capacity*heads*width}),live({1});
        FP32Tensor output({capacity,heads*width},DeviceId::cpu());
        auto *indices=reinterpret_cast<std::int64_t *>(ids.raw_mutable_data());
        std::fill_n(indices,capacity*heads,3);
        auto *count=static_cast<std::int32_t *>(live.raw_mutable_data());
        ImmutableLookupStage stage({.rows={capacity,Qwen4ExpIntBinding(live,BufferId::PREFILL_CHUNK_GEOMETRY)},
            .ids={ids,BufferId::QFN_NGRAM_ROW_IDS},
            .output={output,BufferId::QFN_PLE_EMBEDDING},.format=TensorType::FP32,.rowElements=width,
            .rowBytes=width*sizeof(float),.storage=ImmutableLookupStage::CPUStorage{table,{native,BufferId::QFN_NGRAM_NATIVE_ROWS}}});
        for(const int rows:{5,1,0}){*count=rows;ASSERT_TRUE(stage.execute(&context));}
        auto records=PerfStatsCollector::snapshot({"immutable_lookup"});ASSERT_EQ(records.size(),7u);
        const auto value=[&](const char *name)
        {
            const auto p=std::find_if(records.begin(),records.end(),[&](const auto &r){return r.name==name;});
            if(p==records.end())throw std::runtime_error("Missing CPU native lookup evidence");return p->value;
        };
        EXPECT_EQ(value("requests"),3);EXPECT_EQ(value("ids"),96);EXPECT_EQ(value("native_bytes"),96*width*sizeof(float));
        EXPECT_EQ(value("failures"),0);
        EXPECT_EQ(value("window_hits"),mode==ImmutableRowStorage::RAM?0:95);
        EXPECT_EQ(value("window_misses"),mode==ImmutableRowStorage::RAM?0:1);
        EXPECT_EQ(value("window_evictions"),0);
        for(const auto &record:records)
        {
            EXPECT_EQ(record.tags.at("storage"),mode==ImmutableRowStorage::RAM?"ram":"mmap");
            EXPECT_EQ(record.tags.at("transport"),"cpu_direct");EXPECT_EQ(record.device,DeviceId::cpu().toString());
        }
        PerfStatsCollector::reset();*count=2;ASSERT_TRUE(stage.execute(&context));
        records=PerfStatsCollector::snapshot({"immutable_lookup"});EXPECT_EQ(value("requests"),1);
        EXPECT_EQ(value("ids"),32);EXPECT_EQ(value("native_bytes"),32*width*sizeof(float));
        EXPECT_EQ(value("window_hits"),mode==ImmutableRowStorage::RAM?0:32);
        EXPECT_EQ(value("window_misses"),0);EXPECT_EQ(value("window_evictions"),0);
        indices[0]=-1;EXPECT_THROW(stage.execute(&context),std::out_of_range);
        records=PerfStatsCollector::snapshot({"immutable_lookup"});EXPECT_EQ(value("requests"),2);
        EXPECT_EQ(value("failures"),1);EXPECT_EQ(value("ids"),32);EXPECT_EQ(value("native_bytes"),32*width*sizeof(float));
        EXPECT_EQ(value("window_hits"),mode==ImmutableRowStorage::RAM?0:32);
        EXPECT_EQ(value("window_misses"),0);EXPECT_EQ(value("window_evictions"),0);
    }
}

/** @test The real CPU stage accepts zero/nonzero native cache slots, exact original rows and truthful adaptive work. */
TEST(ImmutableLookupStage, AdaptiveProviderServesLiveRowsAndPreservesNativeSuffixAndMode)
{
    Evidence evidence;llaminar::v2::ThreadCountGuard team(1);CPUDeviceContext context(DeviceId::cpu(),1);Source source;
    for(const std::size_t slots:{0u,2u})
    {
        PerfStatsCollector::reset();constexpr int capacity=5,heads=16,width=8;
        auto provider=source.openAdaptive(slots,capacity*heads);
        const auto retained=provider->retainedOwnerObservation();
        INT32Tensor ids({capacity*heads*2}),native({capacity*heads*width}),control({1});FP32Tensor output({capacity,heads*width},DeviceId::cpu());
        auto *indices=reinterpret_cast<std::int64_t *>(ids.raw_mutable_data());auto *live=static_cast<std::int32_t *>(control.raw_mutable_data());
        ImmutableLookupStage stage({.rows={capacity,Qwen4ExpIntBinding(control,BufferId::PREFILL_CHUNK_GEOMETRY)},
            .ids={ids,BufferId::QFN_NGRAM_ROW_IDS},.output={output,BufferId::QFN_PLE_EMBEDDING},
            .format=TensorType::FP32,.rowElements=width,.rowBytes=width*sizeof(float),
            .storage=ImmutableLookupStage::CPUStorage{provider,{native,BufferId::QFN_NGRAM_NATIVE_ROWS}}});
        const auto identity=stage.graphCaptureVariantSignature();std::uint64_t total=0;
        for(const int count:{5,1,0,3,2,5})
        {
            *live=count;std::fill_n(indices,capacity*heads,-99);
            for(int i=0;i<count*heads;++i)indices[i]=(i%2)+3;
            std::fill_n(output.mutable_data(),output.numel(),-9999.0f);std::memset(native.raw_mutable_data(),0xa5,native.size_bytes());
            ASSERT_TRUE(stage.execute(&context));EXPECT_EQ(stage.graphCaptureVariantSignature(),identity);
            for(int i=0;i<count*heads;++i)for(int j=0;j<width;++j)
                EXPECT_EQ(output.data()[i*width+j],source.values[indices[i]*width+j]);
            for(std::size_t i=count*heads*width;i<output.numel();++i)EXPECT_EQ(output.data()[i],-9999.0f);
            const auto *bytes=static_cast<const unsigned char *>(native.raw_data());
            EXPECT_TRUE(std::all_of(bytes+count*heads*width*sizeof(float),bytes+native.size_bytes(),[](auto b){return b==0xa5;}));
            total+=count*heads;
        }
        const auto records=PerfStatsCollector::snapshot({"immutable_lookup"});
        const auto value=[&](const char *name){const auto r=std::find_if(records.begin(),records.end(),[&](const auto &x){return x.name==name;});
            if(r==records.end())throw std::runtime_error("Missing adaptive CPU work evidence");return r->value;};
        EXPECT_EQ(value("requests"),6);EXPECT_EQ(value("adaptive_batches"),6);EXPECT_EQ(value("ids"),total);
        EXPECT_EQ(value("native_bytes"),total*width*sizeof(float));
        EXPECT_EQ(value("ram_rows")+value("source_rows")+value("duplicate_rows"),total);
        EXPECT_GT(value("duplicate_rows"),0);
        if(!slots){EXPECT_EQ(value("ram_rows"),0);EXPECT_EQ(value("cache_admissions"),0);}
        else {EXPECT_GT(value("ram_rows"),0);EXPECT_GT(value("cache_admissions"),0);}
        for(const auto &r:records){EXPECT_EQ(r.tags.at("storage"),"host_adaptive_ram_ssd");EXPECT_EQ(r.tags.at("transport"),"cpu_direct");}
        const auto after=provider->retainedOwnerObservation();
        EXPECT_EQ(after.authority(),retained.authority());EXPECT_EQ(after.tableStorageBytes(),retained.tableStorageBytes());
        EXPECT_EQ(after.lookupStagingBytes(),retained.lookupStagingBytes());
        *live=1;indices[0]=-1;EXPECT_THROW(stage.execute(&context),std::out_of_range);
        *live=0;EXPECT_TRUE(stage.execute(&context)); // Invalid frame never poisons an intact owner.
    }
}
/** @test A stage cannot bind physical response capacity beyond its shared provider's admitted request arrays. */
TEST(ImmutableLookupStage, AdaptiveStageRejectsLargerResponseCapacityBeforeExecution)
{
    Source source;auto provider=source.openAdaptive(0,16);
    INT32Tensor ids({2*16*2}),native({2*16*8});FP32Tensor output({2,16*8},DeviceId::cpu());
    EXPECT_THROW((void)ImmutableLookupStage({.rows={2},.ids={ids,BufferId::QFN_NGRAM_ROW_IDS},
        .output={output,BufferId::QFN_PLE_EMBEDDING},.format=TensorType::FP32,.rowElements=8,.rowBytes=32,
        .storage=ImmutableLookupStage::CPUStorage{provider,{native,BufferId::QFN_NGRAM_NATIVE_ROWS}}}),std::invalid_argument);
}
