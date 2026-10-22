/**
 * @file RecordedProjectionRuntime.h
 * @brief Shared original recorded-projection ownership, native timing and cohort loop.
 *
 * The head and GDN tools use this one runtime for stat checks, native node
 * inventory, terminal collection, PMA admission, concurrent round submission
 * and participant retirement. Operation adapters own only their actual source
 * preparation, stage capture and original byte oracle.
 */
#pragma once
#include "RecordedHeadCohort.h"
#include "execution/diagnostics/PreparedGemmIsolationAdapter.h"
#include "backends/BackendManager.h"
#include "backends/GPUGraphMemoryContract.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "loaders/GPUVramPreflight.h"
#include <bit>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <set>
#include <sys/stat.h>
#include <thread>
namespace llaminar2::test::recorded_runtime
{
using recorded_head::FileIdentity;
using recorded_head::CohortRoundGate;
using recorded_head::CohortRankInterval;
using recorded_head::requireQueuedCohortAdmission;
using recorded_head::cohortMaximum;
using namespace diagnostics;
/** @brief Reject native failure without changing backend, precision, input or geometry. */
inline void checked(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
/** @brief Authenticate complete immutable file metadata; no binary/model payload hash. */
inline void checkFile(const FileIdentity &f)
{
    struct stat s{};checked(::stat(f.path.c_str(),&s)==0 && S_ISREG(s.st_mode),"Recorded input/source is not a regular file");
    checked(std::uint64_t(s.st_dev)==f.device && std::uint64_t(s.st_ino)==f.inode && std::uint64_t(s.st_size)==f.bytes &&
        std::int64_t(s.st_mtim.tv_sec)*1000000000+s.st_mtim.tv_nsec==f.mtime_ns &&
        std::int64_t(s.st_ctim.tv_sec)*1000000000+s.st_ctim.tv_nsec==f.ctime_ns,
        "Recorded input/source file identity changed");
}
/** @brief Read bounded metadata only; reject a payload-sized JSON/CSV request. */
inline std::string metadata(const std::string &path)
{
    std::ifstream f(path,std::ios::binary|std::ios::ate);checked(bool(f),"Recorded metadata unavailable");
    const auto n=f.tellg();checked(n>0 && n<=4*1024*1024,"Recorded metadata exceeds bounded input contract");
    std::string result(std::size_t(n),'\0');f.seekg(0);f.read(result.data(),std::streamsize(n));checked(bool(f),"Recorded metadata read failed");return result;
}
/** @brief Read only the selected exact FP32 row/slice during setup, preserving every bit. */
inline void readWords(const FileIdentity &f,std::size_t first,std::span<std::uint32_t> words)
{
    checked(first<=f.bytes/4 && words.size()<=f.bytes/4-first,"Recorded FP32 subrange exceeds original file");
    std::ifstream in(f.path,std::ios::binary);checked(bool(in),"Recorded FP32 file unavailable");
    in.seekg(std::streamoff(first*4));in.read(reinterpret_cast<char *>(words.data()),std::streamsize(words.size_bytes()));
    checked(bool(in),"Recorded FP32 subrange read failed");
}
/** @brief Poll one exact event with the standard bounded terminal retirement. */
inline void completed(IBackend &backend,DeviceId device,void *event)
{
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    for(;;)
    {
        bool ready=false;checked(backend.queryEvent(event,device.ordinal,&ready),"Recorded projection event query failed");
        if(ready)return;
        checked(std::chrono::steady_clock::now()<deadline,"Recorded projection exceeded30s retirement");
        std::this_thread::yield();
    }
}
/** @brief Timing events are native-owned; no profiler is initialized or attached. */
struct Events final
{
    IBackend &backend;DeviceId device;void *start=nullptr,*stop=nullptr;
    explicit Events(IBackend &b,DeviceId d):backend(b),device(d)
    {start=backend.createTimingEvent(device.ordinal);stop=backend.createTimingEvent(device.ordinal);
     if(!start||!stop){if(start)backend.destroyEvent(start,device.ordinal);if(stop)backend.destroyEvent(stop,device.ordinal);throw std::runtime_error("Recorded timing event creation failed");}}
    ~Events(){backend.destroyEvent(stop,device.ordinal);backend.destroyEvent(start,device.ordinal);}
};

/** @brief Compare the complete named workspace ABI, not only its byte sum. */
inline bool sameWorkspace(const WorkspaceRequirements &a,const WorkspaceRequirements &b)
{
    if(a.buffers.size()!=b.buffers.size())return false;
    for(std::size_t i=0;i<a.buffers.size();++i)
    {
        const auto &x=a.buffers[i],&y=b.buffers[i];
        if(std::tie(x.name,x.size_bytes,x.alignment,x.required,x.regime,x.content_lifetime)!=
            std::tie(y.name,y.size_bytes,y.alignment,y.required,y.regime,y.content_lifetime))return false;
    }
    return true;
}

/** @brief A typed single-node capture remains the original complete LM-head oracle. */
inline std::unique_ptr<IGPUGraphCapture> captureGraph(ComputeGraph &graph,DeviceGraphExecutor &executor,
    IDeviceContext &context,IWorkerGPUContext &worker,void *stream,const std::shared_ptr<PhysicalMemoryAuthority> &memory)
{
    checked(executor.prepareGraphStorageForCapture(graph,&context,stream,"recorded original LM head"),"Recorded head frontier preparation failed");
    auto result=worker.createGraphCapture(stream);checked(bool(result),"Recorded head graph creation failed");
    ScopedBackendGraphCapture capture(worker,*result,"recorded original LMHeadStage");
    checked(capture.begin(),"Recorded original head capture failed");
    checked(executor.recordSetupGraphMaterialization(graph,&context),"Recorded original head stage recording failed");capture.finish();
    checked(result->prepareRuntimeContextStorage(memory) && result->instantiate(),"Recorded original head instantiation failed");return result;
}
/** @brief The original head needs its named M1 native family; typed bundles authenticate engines separately. */
enum class InventoryPolicy { M1NativeProjection, PreparedStageBundle };
/** @brief Retain exact native nodes/resources; never call occupancy potential measured occupancy. */
inline nlohmann::json inventory(IGPUGraphCapture &graph,DeviceId device,InventoryPolicy policy=InventoryPolicy::M1NativeProjection)
{
    std::vector<GPUGraphKernelNodeInfo> nodes;std::string error;
    checked(graph.inspectKernelNodes(nodes,&error) && !nodes.empty(),"Recorded compute native inventory unavailable");
    nlohmann::json out=nlohmann::json::array();bool projection=false;
    for(const auto &n:nodes)
    {
        checked(n.name_resolved && n.valid(),"Recorded compute kernel identity/resources incomplete");
        checked(n.name.find("nccl")==std::string::npos && n.name.find("capturedTransfer")==std::string::npos,
            "Recorded compute contains unrelated communication");
        projection|=device.is_cuda()
            ? n.name.find("nativeVnniGemv_")!=std::string::npos || n.name.find("nativeVnniGemvTensorCoreSmallM")!=std::string::npos
            : n.name.find("gemv_native_vnni_kernel")!=std::string::npos;
        out.push_back({{"symbol",n.name},{"function",n.function_identity},{"grid",{n.grid_x,n.grid_y,n.grid_z}},
            {"block",{n.block_x,n.block_y,n.block_z}},{"dynamic_shared_bytes",n.dynamic_shared_memory_bytes},
            {"static_shared_bytes",n.static_shared_memory_bytes},{"local_bytes_per_thread",n.local_memory_bytes_per_thread},
            {"registers_per_thread",n.registers_per_thread},{"potential_max_active_blocks_per_sm",n.max_active_blocks_per_sm}});
    }
    checked(projection || policy==InventoryPolicy::PreparedStageBundle,"Recorded compute did not retain a NativeVNNI projection");return out;
}
/**
 * @brief Consume exactly one completed native restore/compute transaction timing.
 * @param parent Original retained ordered-timeline executable.
 * @return Compute and excluded restoration microseconds; no cross-device clock claim.
 */
inline std::pair<double,double> consumeTiming(IGPUGraphCapture &parent)
{
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
        for(;;)
        {
            auto value=parent.consumeOrderedTimelineTiming();
            if(value.state==GPUOrderedTimelineTimingState::Pending)
            {checked(std::chrono::steady_clock::now()<deadline,"Recorded cohort timing exceeded30s retirement");std::this_thread::yield();continue;}
            checked(value.state==GPUOrderedTimelineTimingState::Complete && value.samples.size()==2,
                "Recorded cohort timing is absent, failed, stale or incomplete");
            const auto &reset=value.samples[0],&projection=value.samples[1];
            checked(reset.name=="recorded_restore" && projection.name=="recorded_projection" &&
                reset.kind==GPUOrderedTimelineStepKind::CapturedFragment && projection.kind==GPUOrderedTimelineStepKind::CapturedFragment &&
                std::isfinite(reset.elapsed_ms) && reset.elapsed_ms>=0 && std::isfinite(projection.elapsed_ms) && projection.elapsed_ms>0,
                "Recorded cohort timing step identity or interval is invalid");
            return {projection.elapsed_ms*1000.0,reset.elapsed_ms*1000.0};
        }
}
/** @brief One exact canonical declaration, before any participant tensor materializes. */
struct ParticipantDeclaration
{
    DeviceId device;
    IBackend *backend=nullptr;
    WorkspaceRequirements requirements;
    PhysicalMemoryResource gpu;
    std::size_t tensor_bytes=0,host_bytes=0,raw=0,graph_bytes=0,persistent_bytes=0,staging_bytes=0,host_staging_bytes=0;
};

/**
 * @brief Add concurrent owners to the ONE canonical cohort ledger.
 * @param plan Canonical aggregate admission builder, never a second live ledger.
 * @param d One actual participant's source, storage and graph declaration.
 * @param cpu Same observed host allocator for all concurrently retained owners.
 */
inline void addDeclaration(PhysicalMemoryPlanBuilder &plan,const ParticipantDeclaration &d,const PhysicalMemoryResource &cpu)
{
    plan.add(d.gpu,PhysicalMemoryOwner::PrimaryModelWeights,d.persistent_bytes);
    plan.add(d.gpu,PhysicalMemoryOwner::WeightLoadStaging,d.staging_bytes);
    plan.add(cpu,PhysicalMemoryOwner::WeightLoadStaging,d.host_staging_bytes);
    plan.add(d.gpu,PhysicalMemoryOwner::ExecutionWorkspace,d.tensor_bytes+d.requirements.total_bytes_with_alignment());
    plan.add(cpu,PhysicalMemoryOwner::ExecutionWorkspace,d.host_bytes);
    plan.add(d.gpu,PhysicalMemoryOwner::NativeGraphExecutable,d.graph_bytes);
}

/**
 * @brief Read submission-skew metadata, never a cross-device event clock.
 * @return Monotonic host nanoseconds; no device is queried or synchronized.
 */
inline std::int64_t hostNow()
{return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
/**
 * @brief Compare complete node multisets while retaining original inventory records.
 * @param a Actual parent inventory copied solely for comparison.
 * @param b Original compute inventory including exact multiplicity and resources.
 * @return Whether composition retained every original compute kernel unchanged.
 */
inline bool sameKernels(nlohmann::json a,nlohmann::json b)
{
    const auto less=[](const auto &x,const auto &y){return x.dump()<y.dump();};
    std::sort(a.begin(),a.end(),less);std::sort(b.begin(),b.end(),less);return a==b;
}

/** @brief Retire every retained participant on its actual worker before common source/PMA destruction. */
template<class Participant>
struct ParticipantRetirement final
{
    std::vector<IWorkerGPUContext *> &workers;
    std::vector<std::unique_ptr<Participant>> &states;
    /** @brief Complete cleanup explicitly before final source-file revalidation; idempotent. */
    void retire() noexcept
    {
        for(std::size_t rank=0;rank<states.size();++rank)if(states[rank])
        {
            try{workers[rank]->submitAndWait([&,rank]{states[rank].reset();});}catch(...){std::terminate();}
        }
    }
    /** @brief Preserve every admitted owner on all exceptional paths until original-stream retirement. */
    ~ParticipantRetirement(){retire();}
};
/**
 * @brief Drain every queued task even when a participant or controller failed.
 * @param tasks Actual worker futures whose referenced state must remain alive.
 * @param error Retain the first error for rethrow only after every future is joined.
 */
inline void joinAll(std::vector<std::future<void>> &tasks,std::exception_ptr &error)
{
    for(auto &task:tasks)try{task.get();}catch(...){if(!error)error=std::current_exception();}
}

/**
 * @brief Reuse participant setup for one local or one closed concurrent cohort.
 * @param source Original local request or exact immutable cohort template.
 * @param cohort Validated concurrent membership; null preserves original local mode.
 * @return Passed source/byte/resource/sample evidence, after all graphs retire.
 * @note One host CPU backend/admission budget is initialized before any cohort
 *       materialization. No cross-device operation is hidden in a compute stage.
 */
template<class Operation>
nlohmann::json measureOperation(const typename Operation::Request &source,const typename Operation::Cohort *cohort)
{
    using Request=typename Operation::Request;
    using PreparedParticipant=typename Operation::Participant;
    typename Operation::Source observation(source);
    const std::vector<Request> requests=cohort?cohort->members():std::vector<Request>{source};
    std::vector<IWorkerGPUContext *> workers;std::vector<ParticipantDeclaration> declarations;
    // GPU access is lazy; the standalone CPU PMA owner is deliberately not.
    // Match canonical local startup once for the aggregate host-memory budget,
    // before participant setup or the shared CPU allocator observation.
    const auto first_device=source.backend=="rocm"?DeviceId::rocm(source.ordinal):DeviceId::cuda(source.ordinal);
    auto *first_backend=getBackendFor(first_device);
    checked(first_backend && first_device.ordinal<first_backend->deviceCount(),"Recorded head selected GPU backend/device unavailable");
    initCPUBackend(-1);
    auto *host=getBackendFor(DeviceId::cpu());checked(host,"Recorded head host-memory backend unavailable");
    for(const auto &r:requests)
    {
        const auto device=r.backend=="rocm"?DeviceId::rocm(r.ordinal):DeviceId::cuda(r.ordinal);
        auto *backend=getBackendFor(device);
        checked(backend && r.ordinal<backend->deviceCount(),"Recorded cohort backend/device is unavailable");
        auto *worker=&GPUDeviceContextPool::instance().getContext(device);
        checked(worker->deviceOrdinal()==r.ordinal && std::find(workers.begin(),workers.end(),worker)==workers.end(),
            "Recorded cohort worker does not uniquely own its physical member");
        workers.push_back(worker);
        workers.back()->submitAndWait([&]{declarations.push_back(Operation::declare(r,cohort!=nullptr));});
    }
    // Observe one shared CPU allocator AFTER every worker context is resident.
    // Reusing this exact observation makes concurrent host BOM sums canonical.
    const PhysicalMemoryResource cpu{0,DeviceId::cpu(),host->deviceMemoryTotal(0),host->deviceMemoryFree(0)};
    PhysicalMemoryPlanBuilder plan;
    for(const auto &declaration:declarations)addDeclaration(plan,declaration,cpu);
    auto memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
    std::vector<std::unique_ptr<PreparedParticipant>> states(requests.size());
    ParticipantRetirement<PreparedParticipant> retirement{workers,states};
    for(std::size_t rank=0;rank<requests.size();++rank)
        workers[rank]->submitAndWait([&,rank]
        {
            // If setup throws before publication, this local owner joins on
            // this same worker before any retained field can unwind.
            auto state=std::make_unique<PreparedParticipant>(requests[rank],declarations[rank],*workers[rank],memory);
            state->prepare(observation,cohort!=nullptr);states[rank]=std::move(state);
        });
    nlohmann::json result;
    if(!cohort)
        workers.front()->submitAndWait([&]{result=Operation::describe(*states.front(),observation,states.front()->measureLocal(),false);});
    else
    {
        nlohmann::json samples=nlohmann::json::array(),rounds=nlohmann::json::array();
        const auto workload_identity=Operation::workloadIdentity(source);const auto workload=workload_identity.dump();
        for(int iteration=0;iteration<source.warmup+source.samples;++iteration)
        {
            const bool measured=iteration>=source.warmup;const int round=measured?iteration-source.warmup:iteration;
            CohortRoundGate gate(source.degree);
            std::vector<nlohmann::json> raw(requests.size());std::vector<CohortRankInterval> intervals(requests.size());
            std::vector<std::future<void>> tasks;tasks.reserve(requests.size());std::exception_ptr error;
            const auto ready_ns=hostNow();
            try
            {
                for(std::size_t rank=0;rank<requests.size();++rank)
                {
                    auto task=workers[rank]->submitAsync([&,rank]
                    {
                        try
                        {
                            states[rank]->validateReplay();const auto admitted_ns=hostNow();gate.arriveAndWait(int(rank));
                            const auto submit_ns=hostNow();states[rank]->launchTimed();
                            const auto [us,reset_us]=states[rank]->consumeTiming();
                            intervals[rank]={round,int(rank),us};
                            raw[rank]={{"sample",round},{"rank",rank},{"device_ordinal",requests[rank].ordinal},{"degree",source.degree},
                                {"scope","actual_tp"},{"disposition","partitioned"},{"operation_boundary",Operation::boundary},
                                {"operation",Operation::operation},{"workload",workload},{"latency_us",us},{"restore_interval_us_excluded",reset_us},{"admitted_host_ns",admitted_ns},
                                {"submit_host_ns",submit_ns},{"complete_host_ns",hostNow()}};
                        }
                        catch(...){gate.abort();throw;}
                    });
                    requireQueuedCohortAdmission(task);tasks.push_back(std::move(task));
                }
                gate.releaseWhenReady();
            }
            catch(...){error=std::current_exception();gate.abort();}
            joinAll(tasks,error);
            // Completed values survive a later participant/byte failure in the
            // owned native journal. Never interleave partial worker JSON writes.
            for(const auto &value:raw)if(!value.is_null())
                std::cout<<Operation::sample_prefix<<nlohmann::json{{"measured",measured},{"interval",value}}.dump()<<'\n'<<std::flush;
            if(error)std::rethrow_exception(error);
            const double maximum=cohortMaximum(intervals,source.degree,round);
            tasks.clear();
            try{for(std::size_t rank=0;rank<requests.size();++rank)tasks.push_back(workers[rank]->submitAsync([&,rank]{states[rank]->observe();}));}
            catch(...){error=std::current_exception();}
            joinAll(tasks,error);if(error)std::rethrow_exception(error);
            if(measured)
            {
                for(auto &value:raw)samples.push_back(std::move(value));
                rounds.push_back({{"sample",round},{"maximum_rank_latency_us",maximum},{"host_round_begin_ns",ready_ns},{"release_host_ns",gate.releaseHostNanoseconds()},{"host_round_retired_ns",hostNow()}});
            }
        }
        nlohmann::json participants=nlohmann::json::array();
        for(std::size_t rank=0;rank<states.size();++rank)
            workers[rank]->submitAndWait([&,rank]{participants.push_back(Operation::describe(*states[rank],observation,nlohmann::json::array(),true));});
        result=Operation::describeCohort(*cohort,workload_identity,std::move(participants),std::move(samples),std::move(rounds));
    }
    retirement.retire();Operation::checkFiles(source);return result;
}
} // namespace llaminar2::test::recorded_runtime
