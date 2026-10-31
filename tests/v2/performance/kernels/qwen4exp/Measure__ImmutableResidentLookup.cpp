/**
 * @file Measure__ImmutableResidentLookup.cpp
 * @brief One-geometry captured IQ4_NL resident lookup timing without profiler attachment.
 *
 * Setup authors original native blocks once and admits exact persistent GPU
 * storage through PhysicalMemoryAuthority. The event-free graph contains only
 * production ID validation and direct native decode. Outer timing events cover
 * complete submissions; all CPU reference work and observations are outside
 * timing. A table larger than L2 is mandatory, but fixed replay IDs can have a
 * cache-resident live working set: both extents are reported without claiming
 * cold-DRAM bandwidth. This executable is diagnostic, never an inference path.
 */
#include "backends/GPUDeviceProcessLifetime.h"
#include "kernels/qwen4exp/ImmutableResidentLookupKernels.h"
#include "kernels/qwen4exp/ImmutableLookupKernels.h"
#include "backends/BackendManager.h"
#include "backends/ComputeBackend.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "transfer/TransferEngine.h"
#include "utils/Sha256.h"
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::qwen4exp;
namespace
{
constexpr int kWidth=160;
constexpr std::size_t kRowBytes=5*sizeof(IQ4_NLBlock);
/** @brief Reject failed infrastructure or invalid measurement declarations.
 * @param ok Required invariant.
 * @param why Precise failure diagnostic.
 */
void require(bool ok,const char *why){if(!ok)throw std::runtime_error(why);}
/** @brief Check exact byte/task arithmetic before admission or allocation.
 * @param a First nonnegative factor.
 * @param b Second nonnegative factor.
 * @return Product, or throw on overflow.
 */
std::size_t product(std::size_t a,std::size_t b)
{
    require(b==0||a<=std::numeric_limits<std::size_t>::max()/b,"Measurement geometry overflow");return a*b;
}
/** @brief Check exact additive storage arithmetic before physical admission.
 * @param a First byte extent.
 * @param b Second byte extent.
 * @return Sum, or throw on overflow.
 */
std::size_t sum(std::size_t a,std::size_t b)
{
    require(a<=std::numeric_limits<std::size_t>::max()-b,"Measurement storage sum overflow");return a+b;
}
/** @brief Immutable one-process measurement selection; no automatic fallback. */
struct Options
{
    DeviceId device=DeviceId::invalid();
    int rows=0,live=-1,iterations=100,batches=7,warmup=20;
    std::size_t tableRows=1048576;
    std::uint32_t seed=1701;
};
/** @brief Parse one complete unsigned decimal CLI scalar.
 * @param text Exact option value, without suffix/trailing whitespace.
 * @return Parsed value or throw before backend initialization.
 */
std::size_t number(std::string_view text)
{
    std::size_t result=0;const auto parsed=std::from_chars(text.data(),text.data()+text.size(),result);
    require(parsed.ec==std::errc{}&&parsed.ptr==text.data()+text.size(),"Invalid unsigned decimal option");return result;
}
/** @brief Parse one explicit backend/format/shape/live selection.
 * @param argc Number of command-line arguments.
 * @param argv Alternating option names and values.
 * @return Checked declaration; backend initialization has not occurred.
 */
Options options(int argc,char **argv)
{
    Options value;bool format=false;std::set<std::string_view> seen;
    for(int i=1;i<argc;i+=2)
    {
        require(i+1<argc,"Missing measurement option value");const std::string_view key=argv[i],text=argv[i+1];
        require(seen.insert(key).second,"Duplicate measurement option");
        if(key=="--backend")
        {
            require(text=="cuda"||text=="rocm","Backend must be cuda or rocm");
            value.device=text=="cuda"?DeviceId::cuda(0):DeviceId::rocm(0);
        }
        else if(key=="--format"){require(text=="IQ4_NL","This native measurement supports exactly IQ4_NL width160");format=true;}
        else if(key=="--table-rows")value.tableRows=number(text);
        else if(key=="--seed")
        {
            const auto n=number(text);require(n<=std::numeric_limits<std::uint32_t>::max(),"Seed exceeds uint32");value.seed=static_cast<std::uint32_t>(n);
        }
        else
        {
            const auto n=number(text);require(n<=static_cast<std::size_t>(std::numeric_limits<int>::max()),"Measurement scalar exceeds int");
            if(key=="--rows")value.rows=static_cast<int>(n);
            else if(key=="--live")value.live=static_cast<int>(n);
            else if(key=="--iterations")value.iterations=static_cast<int>(n);
            else if(key=="--batches")value.batches=static_cast<int>(n);
            else if(key=="--warmup")value.warmup=static_cast<int>(n);
            else throw std::invalid_argument("Unknown measurement option");
        }
    }
    require(value.device.is_gpu()&&format&&value.rows>0&&value.live>=0&&value.live<=value.rows,"Explicit backend, IQ4_NL, positive rows and bounded live rows required");
    require(value.tableRows>=product(static_cast<std::size_t>(value.rows),16),"Table must contain distinct admitted IDs");
    require(value.iterations>0&&value.batches>0&&value.warmup>0,"Positive iterations/batches/warmup required");return value;
}
/** @brief Obtain the startup-authorized endpoint observation without raw driver probes.
 * @param device Exact requested native endpoint.
 * @return Immutable startup hardware record, including observed L2 capacity.
 */
ComputeDevice observedDevice(DeviceId device)
{
    auto &manager=DeviceManager::instance();manager.initialize(-1,false);
    const auto type=device.is_cuda()?ComputeBackendType::GPU_CUDA:ComputeBackendType::GPU_ROCM;
    const auto &devices=manager.devices();const auto found=std::find_if(devices.begin(),devices.end(),[&](const auto &d){return d.type==type&&d.device_id==device.ordinal;});
    require(found!=devices.end(),"Requested endpoint excluded or unavailable in startup inventory");
    require(found->last_level_cache_bytes>0,"Observed last-level cache capacity is required");return *found;
}
/** @brief Author original IQ4_NL blocks directly, without floating source requantization.
 * @param rows Exact immutable source-table row count.
 * @param seed Stable native-byte fixture identity.
 * @return Complete row-major original blocks, 90 bytes per row.
 */
std::vector<std::byte> nativeTable(std::size_t rows,std::uint32_t seed)
{
    std::vector<std::byte> bytes(product(rows,kRowBytes));
#pragma omp parallel for schedule(static)
    for(std::size_t row=0;row<rows;++row)
        for(std::size_t b=0;b<5;++b)
        {
            IQ4_NLBlock block{};block.d=static_cast<std::uint16_t>(0x2000u+((row*5+b+seed)%32)*32);
            for(std::size_t q=0;q<16;++q)block.qs[q]=static_cast<std::uint8_t>((row*13+b*17+q*29+seed)&255u);
            std::memcpy(bytes.data()+row*kRowBytes+b*sizeof(block),&block,sizeof(block));
        }
    return bytes;
}
/** @brief Persistent graph storage with physical claims released after actual buffers. */
struct Session
{
    DeviceId device;
    IBackend *backend;
    void *stream;
    std::shared_ptr<PhysicalMemoryAuthority> authority;
    std::vector<PhysicalMemoryAllocationLease> claims;
    std::vector<std::shared_ptr<DeviceTransferBuffer>> owners;
    std::unique_ptr<IImmutableResidentLookupKernels> kernels;
    std::unique_ptr<IGPUGraphCapture> graph;
    std::vector<void *> events;
    /** @brief Complete diagnostic work before graph, event and allocation retirement. */
    ~Session() noexcept
    {
        if(!backend->synchronizeStream(stream,device.ordinal))std::terminate();
        graph.reset();for(auto *event:events)backend->destroyEvent(event,device.ordinal);
    }
    /** @brief Claim and materialize one exact persistent GPU allocation.
     * @param count Typed element capacity.
     * @param owner Sole admitted physical-byte owner.
     * @return Stable device pointer retained through graph retirement.
     */
    template<class T>T *allocate(std::size_t count,PhysicalMemoryOwner owner)
    {
        const auto bytes=product(count,sizeof(T));auto claim=authority->claimNewAllocation(device,owner,bytes);
        auto buffer=TransferEngine::instance().allocateDeviceTransferBuffer(bytes,device);
        require(buffer&&buffer->isBound(),"Measurement allocation failed");auto *pointer=static_cast<T *>(buffer->mutableDeviceData());
        claims.push_back(std::move(claim));owners.push_back(std::move(buffer));return pointer;
    }
    /** @brief Upload setup bytes only, before capture/timing.
     * @param target Exact persistent allocation.
     * @param source Original immutable host values.
     */
    template<class T>void upload(T *target,std::span<const T> source)
    {
        require(backend->hostToDeviceOnStream(target,source.data(),source.size_bytes(),device.ordinal,stream),"Setup publication failed");
        require(backend->synchronizeStream(stream,device.ordinal),"Setup publication completion failed");
    }
    /** @brief Observe exact required diagnostic evidence outside all timed work.
     * @param source Stable native source pointer.
     * @param count Required element count; zero transfers nothing.
     * @return Terminal host evidence.
     */
    template<class T>std::vector<T> download(const T *source,std::size_t count)
    {
        if(count==0)return {};std::vector<T> result(count);
        require(backend->deviceToHostOnStream(result.data(),source,product(count,sizeof(T)),device.ordinal,stream),"Terminal observation failed");
        require(backend->synchronizeStream(stream,device.ordinal),"Terminal observation completion failed");return result;
    }
    /** @brief Retain a native timing event outside the event-free graph.
     * @return Exact event, retired after submitted graph work completes.
     */
    void *timingEvent()
    {
        auto *event=backend->createTimingEvent(device.ordinal);require(event,"Native timing event unavailable");events.push_back(event);return event;
    }
};
/** @brief Admit original immutable table and exact direct-output/ID/control bytes once.
 * @param hardware Startup-authorized native physical capacity observation.
 * @param device Exact physical endpoint.
 * @param tableBytes Complete original native source storage.
 * @param workingBytes Exact persistent IDs/output/count/health/error storage.
 * @return Sole physical admission/materialization authority for these allocations.
 */
std::shared_ptr<PhysicalMemoryAuthority> admit(const ComputeDevice &hardware,DeviceId device,std::size_t tableBytes,std::size_t workingBytes)
{
    PhysicalMemoryResource resource{.world_rank=0,.device=device,.total_bytes=hardware.total_memory_bytes,.admission_available_bytes=hardware.free_memory_bytes};
    PhysicalMemoryPlanBuilder plan;plan.add(resource,PhysicalMemoryOwner::ImmutableTableStorage,tableBytes);
    plan.add(resource,PhysicalMemoryOwner::ActivationArena,workingBytes);
    return std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
}
/** @brief Assert exact live output, native controller health and inactive sentinels.
 * @param session Persistent graph/storage owner.
 * @param output Native output pointer.
 * @param controls Native live/healthy/error words.
 * @param expected Exact CPU reference for only live IDs.
 * @param capacity Admitted output element capacity.
 */
void validate(Session &session,const float *output,const int *controls,const std::vector<float> &expected,std::size_t capacity)
{
    const auto actual=session.download(output,expected.size());
    if(!actual.empty())require(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float))==0,"Native output differs from original-block CPU decode");
    const auto control=session.download(controls,3);require(control[0]==static_cast<int>(expected.size()/(16*kWidth))&&control[1]==1&&control[2]==0,"Resident lookup failed canonical live/health/error validation");
    if(expected.size()<capacity)
    {
        for(const auto index:std::array<std::size_t,2>{expected.size(),capacity-1})
            require(std::bit_cast<std::uint32_t>(session.download(output+index,1).front())==std::bit_cast<std::uint32_t>(-12345.0f),"Inactive output sentinel changed");
    }
}
/** @brief Time whole repeated two-kernel submissions using outer events only.
 * @param session Stable graph, stream and backend.
 * @param start Exact native start event outside capture.
 * @param stop Exact native stop event outside capture.
 * @param iterations Number of complete graph submissions.
 * @return Complete amortized submitted-transaction microseconds, including gaps.
 */
float elapsed(Session &session,void *start,void *stop,int iterations)
{
    require(session.backend->recordEvent(start,session.device.ordinal,session.stream),"Outer start event failed");
    for(int i=0;i<iterations;++i)require(session.graph->launch(),"Timed captured lookup submission failed");
    require(session.backend->recordEvent(stop,session.device.ordinal,session.stream),"Outer stop event failed");
    require(session.backend->waitForEvent(stop,session.device.ordinal),"Outer event completion failed");
    float ms=0;require(session.backend->eventElapsedTimeMs(start,stop,session.device.ordinal,&ms),"Outer elapsed time unavailable");return ms*1000.0f/iterations;
}
/** @brief Prepare and measure exactly one declared native geometry.
 * @param option Immutable CLI selection, printed with all evidence.
 */
void measure(const Options &option)
{
    const auto hardware=observedDevice(option.device);const auto tableBytes=product(option.tableRows,kRowBytes);
    require(tableBytes>hardware.last_level_cache_bytes,"Persistent original table must exceed observed L2 capacity");
    const auto idsCapacity=product(static_cast<std::size_t>(option.rows),16),outputCapacity=product(idsCapacity,kWidth);
    const auto idsBytes=product(idsCapacity,sizeof(std::int64_t)),outputBytes=product(outputCapacity,sizeof(float));
    const auto workingBytes=sum(sum(idsBytes,outputBytes),3*sizeof(int));
    auto authority=admit(hardware,option.device,tableBytes,workingBytes);
    const auto source=nativeTable(option.tableRows,option.seed);
    const auto sourceSha=sha256BytesHex({reinterpret_cast<const char *>(source.data()),source.size()});require(sourceSha.has_value(),"Native source SHA unavailable");
    std::vector<std::int64_t> hostIds(idsCapacity,-1);const auto liveIds=product(static_cast<std::size_t>(option.live),16);
    const auto step=option.tableRows/idsCapacity;
    const auto shift=static_cast<std::size_t>(option.seed)%option.tableRows;
    for(std::size_t i=0;i<liveIds;++i)
    {
        const auto base=i*step;
        hostIds[i]=static_cast<std::int64_t>(base>=option.tableRows-shift?base-(option.tableRows-shift):base+shift);
    }
    std::vector<std::byte> gathered(product(liveIds,kRowBytes));
    for(std::size_t i=0;i<liveIds;++i)std::memcpy(gathered.data()+i*kRowBytes,source.data()+static_cast<std::size_t>(hostIds[i])*kRowBytes,kRowBytes);
    std::vector<float> expected(product(liveIds,kWidth));decodeCPUImmutableLookupRows(TensorType::IQ4_NL,kWidth,kRowBytes,gathered.data(),liveIds,expected.data());
    auto *backend=getBackendFor(option.device);require(backend&&backend->deviceCount()>option.device.ordinal,"Requested native backend unavailable");
    auto &worker=GPUDeviceContextPool::instance().getContext(option.device);
    worker.submitAsync([&] {
        Session session{option.device,backend,ExplicitGPUStream(worker.defaultStream()).get(),authority};
        auto *table=session.allocate<std::byte>(tableBytes,PhysicalMemoryOwner::ImmutableTableStorage);session.upload<std::byte>(table,source);
        auto *ids=session.allocate<std::int64_t>(idsCapacity,PhysicalMemoryOwner::ActivationArena);session.upload<std::int64_t>(ids,hostIds);
        auto *output=session.allocate<float>(outputCapacity,PhysicalMemoryOwner::ActivationArena);
        const std::vector<float> poison(outputCapacity,-12345.0f);session.upload<float>(output,poison);
        auto *controls=session.allocate<int>(3,PhysicalMemoryOwner::ActivationArena);const std::array initial{option.live,1,0};session.upload<int>(controls,initial);
        const auto view=ImmutableResidentTableView::bind(option.device,TensorType::IQ4_NL,table,option.tableRows,kWidth,kRowBytes,tableBytes,1);
#ifdef HAVE_CUDA
        if(option.device.is_cuda())session.kernels=createCUDAImmutableResidentLookupKernels(view);
#endif
#ifdef HAVE_ROCM
        if(option.device.is_rocm())session.kernels=createROCmImmutableResidentLookupKernels(view);
#endif
        require(bool(session.kernels),"Selected resident lookup factory unavailable");
        session.events.reserve(2);auto *start=session.timingEvent(),*stop=session.timingEvent();
        const ImmutableResidentGatherArgs args{{option.rows,controls},ids,idsCapacity,output,outputCapacity,controls+1,controls+2};
        session.graph=worker.createGraphCapture(session.stream);require(bool(session.graph),"Native graph capture unavailable");
        {ScopedBackendGraphCapture capture(worker,*session.graph,"resident immutable lookup measurement");require(capture.begin(),"Lookup capture begin failed");session.kernels->gather(args,session.stream);capture.finish();}
        require(session.graph->nodeCount()==2,"Lookup graph must contain exactly two production kernels");
        std::vector<GPUGraphKernelNodeInfo> nodes;std::string detail;require(session.graph->inspectKernelNodes(nodes,&detail),detail.c_str());require(nodes.size()==2,"Incomplete native kernel inspection");
        for(const auto &node:nodes)
        {
            require(node.valid()&&node.local_memory_bytes_per_thread==0,"Native resource record invalid or local storage nonzero");
            std::cout<<"RESIDENT_LOOKUP_NODE name="<<std::quoted(node.name)<<" grid="<<node.grid_x<<'x'<<node.grid_y<<'x'<<node.grid_z
                <<" block="<<node.block_x<<'x'<<node.block_y<<'x'<<node.block_z<<" registers="<<node.registers_per_thread
                <<" local_bytes="<<node.local_memory_bytes_per_thread<<" static_shared="<<node.static_shared_memory_bytes
                <<" dynamic_shared="<<node.dynamic_shared_memory_bytes<<" max_active_blocks_per_sm="<<node.max_active_blocks_per_sm<<'\n';
        }
        require(session.graph->instantiate(),"Lookup graph instantiation failed");const auto owners=session.owners.size();
        for(int i=0;i<option.warmup;++i)require(session.graph->launch(),"Warm lookup replay failed");validate(session,output,controls,expected,outputCapacity);
        std::cout<<"RESIDENT_LOOKUP_IDENTITY backend="<<option.device.toString()<<" format=IQ4_NL width="<<kWidth
            <<" rows="<<option.rows<<" live="<<option.live<<" table_rows="<<option.tableRows<<" row_bytes="<<kRowBytes
            <<" table_bytes="<<tableBytes<<" observed_l2_bytes="<<hardware.last_level_cache_bytes<<" compute_units="<<hardware.compute_units
            <<" live_ids="<<liveIds<<" live_native_bytes="<<product(liveIds,kRowBytes)<<" live_output_bytes="<<expected.size()*sizeof(float)
            <<" seed="<<option.seed<<" source_sha256="<<*sourceSha<<" kernels_per_graph=2 table_gt_l2=1 cache_regime=fixed_replay_ids"
            <<" live_working_set_may_fit_l2="<<(product(liveIds,kRowBytes)<=hardware.last_level_cache_bytes)
            <<" source_requantization=0 lookup_service=0 timed_host_observations=0\n";
        std::vector<float> samples; samples.reserve(static_cast<std::size_t>(option.batches));
        for(int batch=0;batch<option.batches;++batch)
        {
            const auto us=elapsed(session,start,stop,option.iterations);validate(session,output,controls,expected,outputCapacity);
            require(session.owners.size()==owners&&session.kernels->table()==view,"Captured native allocation identity drift");samples.push_back(us);
            std::cout<<"RESIDENT_LOOKUP_TIMING batch="<<batch<<" submissions="<<option.iterations<<" complete_transaction_us="<<std::setprecision(9)<<us
                <<" validation=byte_exact_live_and_inactive_sentinels scope=outer_complete_two_kernel_graph includes_submission_gaps=1\n";
        }
        std::sort(samples.begin(),samples.end());std::cout<<"RESIDENT_LOOKUP_MEDIAN complete_transaction_us="<<samples[samples.size()/2]
            <<" samples="<<samples.size()<<" pointers_stable=1 native_owner_allocations="<<owners<<" validation=pass no_cold_dram_bandwidth_claim=1\n";
    }).get();
}
}
/** @brief Run one explicit diagnostic selection; exceptions fail the process.
 * @param argc Command-line argument count.
 * @param argv Explicit backend/format/shape/repetition values.
 * @return Zero only after native correctness and timing complete successfully.
 */
int main(int argc,char **argv)
{
    llaminar2::GPUDeviceProcessLifetime gpu_process_lifetime;

    try{measure(options(argc,argv));return 0;}
    catch(const std::exception &error){std::cerr<<"FATAL "<<error.what()<<'\n';return 1;}
}
