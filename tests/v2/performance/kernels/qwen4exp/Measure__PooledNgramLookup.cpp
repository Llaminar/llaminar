/**
 * @file Measure__PooledNgramLookup.cpp
 * @brief Release replay timing of actual local, cross-vendor and shared-source ConsumerLocal pooled fixtures.
 *
 * Fixed original IQ4_NL160/90B input traces and capture/validation code are shared
 * with native correctness tests. Preparation/admission/poison/oracles are outside
 * timing; native events cover the full source operation, including protocol waits.
 * Host wall covers submission through complete native cohort completion. The
 * original service remains armed across frames and terminally joins afterward. No observed device value selects a workload or repairs its state.
 */
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
#include "fixtures/execution/RetainedHostServiceFixture.h"
#include "fixtures/execution/RetainedHostServiceCalibrationAccess.h"
#endif
#include "execution/qwen4exp/PooledImmutableLocalLookupGraph.h"
#include "execution/qwen4exp/PooledImmutableParticipantReadGraph.h"
#include "execution/qwen4exp/PooledImmutableCompleteCohort.h"
#include "execution/qwen4exp/PooledImmutableParticipantRetirementGraph.h"
#include "PooledACKReadyCancellation.h"
#include "kernels/qwen4exp/PooledImmutableSourcePreparedStorage.h"
#include "kernels/qwen4exp/ImmutableLookupDecode.h"
#include "memory/CPUTensorStorageDeclaration.h"
#include "tensors/NativeTensorExtent.h"
#include "backends/BackendManager.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/GPUGraphMemoryContract.h"
#include "fixtures/backends/PreparedNativeLookupFixture.h"
#include "fixtures/backends/RetainedParticipantGraphExecutableFixtureAccess.h"
#include "backends/NativeOrderingEventGroup.h"
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
#include "fixtures/backends/PooledRemoteGraphCalibration.h"
#endif
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "transfer/TransferEngine.h"
#include "utils/EmbeddingVerifierFormats.h"
#include "utils/PooledNgramReplayTrace.h"
#include "utils/PooledNgramReplaySchedule.h"
#include <gtest/gtest.h>
#include <array>
#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <map>
#include <type_traits>
#include <unistd.h>
#include "backends/GPUDeviceProcessLifetime.h"
#include <charconv>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <string_view>
#include <set>
#include <future>
#include <vector>
#include <sys/mman.h>
#include <sys/statfs.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <fstream>
#include "loaders/AdaptiveImmutableLookupGeometry.h"

namespace llaminar2::test {
namespace {
namespace q=qwen4exp;namespace n=q::pooled_native;namespace p=n::p;
#include "integration/kernels/qwen4exp/PooledImmutableProviderReplay.inc"
#include "integration/kernels/qwen4exp/PooledCrossVendorReadReplay.inc"

/** @brief Explicit measurement declarations; no automatic backend or placement choice. */
struct Options {
    DeviceId continuation=DeviceId::invalid(),owner=DeviceId::invalid();
    std::string placement,scenario;
    ngram_replay::Shape shape;
    std::uint32_t gpu_slots=0,ram_slots=2,samples=9;
    std::vector<DeviceId> consumers; ///< Explicit actual ConsumerLocal membership, never storage-owner inference.
    std::uint32_t requests=1; ///< Explicit original request-reset replay count for grouped diagnostics only.
    std::size_t mmapWindowBytes=4096,mmapWindowCount=2; ///< Explicit original ConsumerLocal source policy; defaults preserve stress trace.
    bool mmapGeometryProvided=false; ///< Reject unsupported actor modes rather than silently ignoring authored geometry.
    bool coldAttempt=false; ///< Owned-file advice only; observations determine whether physical I/O occurred.
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
    ProviderReplayDiagnosticProgram diagnosticProgram=ProviderReplayDiagnosticProgram::OrdinaryLookup;
    std::uint32_t lifetimes=1;
    bool remoteHandshake=false;
#endif
};
/** @brief Parse a complete unsigned scalar before runtime initialization.
 * @param text Original CLI scalar. @return Exact representable value. */
std::uint64_t number(std::string_view text){std::uint64_t result=0;const auto parsed=std::from_chars(text.data(),text.data()+text.size(),result);
    requireProvider(parsed.ec==std::errc{}&&parsed.ptr==text.data()+text.size(),"Invalid complete unsigned measurement value");return result;}
/** @brief Parse one explicitly named native GPU. @param text cuda:N or rocm:N. @return Original GPU identity. */
DeviceId device(std::string_view text){const auto colon=text.find(':');requireProvider(colon!=std::string_view::npos,"Explicit GPU backend:ordinal required");
    const auto ordinal=number(text.substr(colon+1));requireProvider(ordinal<=std::numeric_limits<int>::max(),"GPU ordinal overflow");
    if(text.substr(0,colon)=="cuda")return DeviceId::cuda(static_cast<int>(ordinal));
    if(text.substr(0,colon)=="rocm")return DeviceId::rocm(static_cast<int>(ordinal));
    throw std::invalid_argument("Measurement supports only explicit CUDA/ROCm continuation devices");}
/** @brief Parse bounded cold declarations without creating runtime resources.
 * @param argc Original count. @param argv Original argument vector. @return Checked one-cell selection. */
Options options(int argc,char** argv){Options o;bool slots=false,tokens=false,capacity=false;
    for(int i=1;i<argc;++i){const std::string_view key=argv[i];requireProvider(i+1<argc,"Missing measurement argument value");const std::string_view value=argv[++i];
        if(key=="--continuation")o.continuation=device(value);else if(key=="--owner")o.owner=device(value);
        else if(key=="--consumers"){
            std::size_t begin=0;
            while(begin<value.size()){
                const auto end=value.find(',',begin);
                o.consumers.push_back(device(value.substr(begin,end==std::string::npos?end:end-begin)));
                if(end==std::string::npos)break;
                begin=end+1;
                requireProvider(begin<value.size(),"Trailing consumer delimiter");
            }
        }
        else if(key=="--mmap-window-bytes" || key=="--mmap-window-count"){
            const auto valueBytes=number(value);
            requireProvider(valueBytes>0 && valueBytes<=std::numeric_limits<std::size_t>::max(),"Invalid bounded mmap geometry");
            if(key=="--mmap-window-bytes")o.mmapWindowBytes=static_cast<std::size_t>(valueBytes);
            else o.mmapWindowCount=static_cast<std::size_t>(valueBytes);
            o.mmapGeometryProvided=true;
        }
        else if(key=="--requests"){
            const auto count=number(value);
            requireProvider(count&&count<=20,"Grouped requests must be1through20");
            o.requests=static_cast<std::uint32_t>(count);
        }
        else if(key=="--cold-attempt"){
            requireProvider(value=="0"||value=="1","Cold attempt must be exactly0or1");
            o.coldAttempt=value=="1";
        }
        else if(key=="--placement")o.placement=value;else if(key=="--scenario")o.scenario=value;
        else if(key=="--locality"){requireProvider(value=="scattered"||value=="contiguous","Unknown explicit source locality");o.shape.locality=value=="scattered"?ngram_replay::Locality::Scattered:ngram_replay::Locality::Contiguous;}
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        else if(key=="--diagnostic-program"){
            if(value=="ordinary")o.diagnosticProgram=ProviderReplayDiagnosticProgram::OrdinaryLookup;
            else if(value=="read-ahead")o.diagnosticProgram=ProviderReplayDiagnosticProgram::ReadAheadLookup;
            else if(value=="authored-remote")o.diagnosticProgram=ProviderReplayDiagnosticProgram::AuthoredRemoteLookup;
            else throw std::invalid_argument("Unknown explicit raw native program");
        }
        else if(key=="--remote-handshake"){requireProvider(value=="0" || value=="1","Remote handshake must be exactly 0 or 1");o.remoteHandshake=value=="1";}
        else if(key=="--lifetimes"){const auto n=number(value);requireProvider(n && n<=20,"Remote lifetimes must be 1 through 20");o.lifetimes=static_cast<std::uint32_t>(n);}
#endif
        else if(key=="--table-rows")o.shape.table_rows=number(value);
        else {const auto n=number(value);requireProvider(n<=std::numeric_limits<std::uint32_t>::max(),"Measurement uint32 geometry overflow");
            if(key=="--tokens"){o.shape.tokens=static_cast<std::uint32_t>(n);tokens=true;}
            else if(key=="--capacity-tokens"){o.shape.capacity_tokens=static_cast<std::uint32_t>(n);capacity=true;}
            else if(key=="--gpu-slots"){o.gpu_slots=static_cast<std::uint32_t>(n);slots=true;}
            else if(key=="--ram-slots")o.ram_slots=static_cast<std::uint32_t>(n);
            else if(key=="--samples")o.samples=static_cast<std::uint32_t>(n);
            else throw std::invalid_argument("Unknown measurement option");}}
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
    if(o.diagnosticProgram==ProviderReplayDiagnosticProgram::AuthoredRemoteLookup){
        requireProvider(o.placement=="cross_vendor","Authored remote calibration requires its explicit cross-vendor placement");
    }else{
        requireProvider(o.lifetimes==1 && !o.remoteHandshake,"Remote lifecycle observations require the explicit authored-remote diagnostic");
        if(o.placement=="cross_vendor" && o.diagnosticProgram!=ProviderReplayDiagnosticProgram::OrdinaryLookup)
            throw std::invalid_argument("Cross-vendor program requires ordinary or explicitly authored-remote diagnostic; local side-lane composition is separate");
    }
#endif
    if(!capacity)o.shape.capacity_tokens=o.shape.tokens;
    requireProvider(tokens&&o.shape.tokens&&slots&&o.continuation.is_gpu()&&o.samples&&o.samples<=200,"Explicit continuation/tokens/gpu-slots and bounded positive samples required");
    requireProvider(o.placement=="local"||o.placement=="cross_vendor"||o.placement=="consumer_local","Placement must be local, cross_vendor or consumer_local");
    requireProvider(!o.mmapGeometryProvided || o.placement=="consumer_local",
        "Explicit mmap geometry is supported only by the ConsumerLocal actor");
    (void)ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped,o.mmapWindowBytes,o.mmapWindowCount);
    if(o.placement=="consumer_local"){
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        requireProvider(o.diagnosticProgram==ProviderReplayDiagnosticProgram::OrdinaryLookup,
            "Grouped read-ahead/remote diagnostic composition is not bound; ordinary ConsumerLocal timing is explicit");
#endif
        requireProvider(o.consumers.size()==2&&o.consumers.front()==o.continuation&&
            o.consumers[0]!=o.consumers[1]&&o.consumers[0].type==o.consumers[1].type,
            "ConsumerLocal timing requires two distinct homogeneous GPU consumers with continuation first");
        requireProvider(!o.owner.is_valid()||o.owner==o.continuation,"ConsumerLocal owners are each original consumer, not one shared storage owner");
        o.owner=o.continuation;
    }else requireProvider(o.consumers.empty()&&!o.coldAttempt&&o.requests==1,"Grouped consumer/cold options require consumer_local placement");
    if(o.placement=="local"){requireProvider(!o.owner.is_valid()||o.owner==o.continuation,"Local pooled owner must be the continuation");o.owner=o.continuation;}
    else if(o.placement=="cross_vendor")requireProvider(o.owner.is_gpu()&&o.owner.type!=o.continuation.type,"Cross-vendor requires its actual opposite-vendor owner; native-peer substitution is forbidden");
    const auto ids=ngram_replay::ids(o.shape,ngram_replay::Pattern::DistinctA);const auto unique=ids.size();
    requireProvider(o.gpu_slots<=o.shape.table_rows,"Cache cannot exceed its declared original table rows");
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
    if(o.diagnosticProgram==ProviderReplayDiagnosticProgram::AuthoredRemoteLookup){
        requireProvider(o.scenario=="authored_lifecycle" && o.samples>=4,"Authored remote calibration requires its complete large/small/empty/duplicate lifecycle");return o;
    }
#endif
    if(o.scenario=="warm_hits")requireProvider(o.gpu_slots>=unique,"Warm-hit trace requires all distinct live head IDs resident");
    else if(o.scenario=="alternating_sets")requireProvider(o.gpu_slots<=unique,"Alternating trajectory requires disjoint sets larger than or equal to cache capacity");
    else if(o.scenario=="mixed")requireProvider(o.gpu_slots==unique,"Mixed trace retains exactly12hot heads and alternates4cold heads in one request-sized cache");
    else if(o.scenario=="churn")requireProvider(o.gpu_slots&&o.gpu_slots<unique&&o.ram_slots<unique,"A/B/A churn working set must exceed both declared cache tiers");
    else if(o.scenario=="duplicate_control")requireProvider(o.gpu_slots>=ngram_replay::uniqueCount(ngram_replay::ids(o.shape,ngram_replay::Pattern::DuplicateHeavy)),"Duplicate control requires its declared hot set");
    else requireProvider(o.scenario=="empty","Unknown closed replay scenario");
    return o;
}
/** @brief Fail also on assertion failures reported by the shared GTest fixture outside RUN_ALL_TESTS. */
void requireAssertions(){requireProvider(!::testing::Test::HasFailure()&&!::testing::UnitTest::GetInstance()->ad_hoc_test_result().Failed(),"Shared production fixture byte/extent/counter assertion failed");}
/** @brief Exact trace selection; no runtime observation participates in this function. */
struct Frame {ngram_replay::Pattern pattern;std::string phase;std::optional<std::uint32_t> expected_hits;};
/** @brief Resolve explicit CLI spelling to one closed shared schedule. */
ngram_replay::Scenario scenario(const Options& o){using S=ngram_replay::Scenario;
    if(o.scenario=="warm_hits")return S::WarmHits;if(o.scenario=="alternating_sets")return S::AlternatingSets;
    if(o.scenario=="mixed")return S::Mixed;if(o.scenario=="churn")return S::Churn;
    if(o.scenario=="duplicate_control")return S::DuplicateControl;if(o.scenario=="empty")return S::Empty;
    throw std::invalid_argument("Unknown fixed pooled traffic scenario");
}
/** @brief Prescribe GPU hits from immutable scenario and original cache geometry. */
Frame frame(const Options& o,std::uint32_t sample){using S=ngram_replay::Scenario;const auto kind=scenario(o);
    const auto declared=ngram_replay::scheduledFrame(kind,false,sample);std::optional<std::uint32_t> hits;
    if(kind==S::WarmHits)hits=o.shape.tokens*16;
    else if(kind==S::Empty || !o.gpu_slots)hits=0;
    else if(kind==S::DuplicateControl)hits=static_cast<std::uint32_t>(ngram_replay::uniqueCount(ngram_replay::ids(o.shape,declared.pattern)));
    return {declared.pattern,std::string(declared.phase),hits};
}
/** @brief Actual local pooled fixture actor; original GPU-worker ownership is preserved. */
struct LocalActor {
    ProviderReplayPrograms programs;
    IWorkerGPUContext& worker;
    std::unique_ptr<ProviderReplayGraphs> graphs;
    std::unique_ptr<ProviderReplay> fixture;
    /** @brief Admit the same original source/payload program before any actual native fixture body. */
    static ProviderReplayPrograms prepare(std::shared_ptr<ProviderFile> file,const Options& o){
        ProviderReplayPrograms result({{std::move(file),o.shape.capacity_tokens*16,o.gpu_slots,o.ram_slots,n::OriginalIdWidth::UInt64,true,OwnerAdmission::Original,ACKInterleaving::None,ActivationPreparation::BeforeGraph,o.shape.capacity_tokens*16,true}});
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        result.setDiagnosticProgram(o.diagnosticProgram);
#endif
        result.prepare(o.continuation);return result;
    }
    /** @brief Materialize the unchanged original body on its exact preborn source worker. */
    LocalActor(std::shared_ptr<ProviderFile> file,const Options& o):programs(prepare(std::move(file),o)),worker(programs.worker()){
        const auto& program=programs.program(0);graphs=std::make_unique<ProviderReplayGraphs>(program.session->permit(o.continuation),program.geometry.graphCount
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
            ,program.spec.diagnosticProgram
#endif
        );
        worker.submitAndWait([&]{fixture=std::make_unique<ProviderReplay>(program,*graphs);});
    }
    /** @brief Retire parent-owned graph handles before original embedded root teardown and native reset. */
    ~LocalActor(){worker.submitAndWait([&]{if(fixture)fixture->join();});graphs->retire();worker.submitAndWait([&]{fixture.reset();});graphs.reset();programs.retire();}
    /** @return Exact original owning stream. */
    void* stream()const{return fixture->stream;}
    /** @brief Arm the one actual retained service interval outside lookup timing. */
    void admit(){worker.submitAndWait([&]{fixture->admit(1);});}
    /** @brief Upload declared input outside lookup timing. */
    void prepare(std::span<const std::uint64_t> ids){worker.submitAndWait([&]{fixture->prepareTiming(ids);});}
    /** @brief Execute the complete original retained graph without observation. */
    void launch(){worker.submitAndWait([&]{fixture->launchTiming();});}
    /** @brief Wait for complete original native work, with no oracle or diagnostic readback. */
    void finish(){worker.submitAndWait([&]{fixture->join();});}
    /** @brief Close the original request interval and join its service once, after all lookup samples. */
    void close(){worker.submitAndWait([&]{fixture->close();});requireAssertions();}
    /** @brief Validate original native/output bytes after timing. */
    void validate(std::span<const std::uint64_t> ids,std::optional<std::uint32_t> hits){worker.submitAndWait([&]{fixture->validateTiming(ids);if(hits)EXPECT_EQ(fixture->lastObservation.gpu_hits,*hits);});requireAssertions();}
    /** @return Last final copied diagnostics. */
    PooledNgramReplayObservation observation()const{return fixture->lastObservation;}
};
/** @brief Actual independent source/remote owner cohort actor. */
struct CrossActor {
    CrossVendorReadReplayPrograms programs;
    IWorkerGPUContext& worker;
    std::unique_ptr<CrossVendorReadReplayGraphs> graphs;
    std::unique_ptr<CrossVendorReadReplay> fixture;
    /** @brief Declare only the actual source plus remote storage population, never a pretend compute domain. */
    static CrossVendorReadReplayPrograms prepare(std::shared_ptr<ProviderFile> file,const Options& o){
        CrossVendorReadReplayPrograms result({{std::move(file),o.gpu_slots,o.shape.capacity_tokens*16,o.shape.capacity_tokens*16,o.ram_slots}});
        result.prepare(o.continuation,o.owner);return result;
    }
    /** @brief Bind sole original graph owners before the unchanged cold source/remote stage recording. */
    CrossActor(std::shared_ptr<ProviderFile> file,const Options& o):programs(prepare(std::move(file),o)),worker(*programs.program(0).session->permit(o.continuation).worker()){
        const auto& original=programs.program(0);graphs=std::make_unique<CrossVendorReadReplayGraphs>(original);
        fixture=std::make_unique<CrossVendorReadReplay>(original,*graphs);
    }
    /** @brief Native definitions close before actual embedded roots; source contexts retire last. */
    ~CrossActor(){if(fixture){fixture->timingJoin();fixture->retireCohort();}graphs->retire();fixture.reset();graphs.reset();programs.retire();}
    void* stream()const{return fixture->measurementStream();}
    void admit(){fixture->admit(1);}
    void prepare(std::span<const std::uint64_t> ids){fixture->prepareTiming(ids);fixture->timingJoin();}
    void launch(){fixture->timingLaunch();}
    void finish(){fixture->timingJoin();}
    void close(){fixture->close();requireAssertions();}
    void validate(std::span<const std::uint64_t> ids,std::optional<std::uint32_t> hits){fixture->validateTiming(ids,hits);requireAssertions();}
    PooledNgramReplayObservation observation()const{return fixture->lastObservation;}
};
/** @brief Fixture-owned outer events; created/destroyed outside captured or measured execution. */
class TimingEvents final {
    IWorkerGPUContext& worker_;IBackend* backend_;DeviceId device_;void* start_=nullptr;void* stop_=nullptr;void* stream_=nullptr;
public:
    /** @brief Admit both timing events on the actual source worker, retaining partial-failure cleanup. */
    TimingEvents(IWorkerGPUContext& worker,DeviceId device):worker_(worker),backend_(getBackendFor(device)),device_(device){
        worker_.submitAndWait([&]{try{start_=backend_->createTimingEvent(device_.ordinal);requireProvider(start_,"Native start event unavailable");stop_=backend_->createTimingEvent(device_.ordinal);requireProvider(stop_,"Native stop event unavailable");}
            catch(...){if(start_)backend_->destroyEvent(start_,device_.ordinal);start_=nullptr;throw;}});}
    /** @brief All measured work has terminally retired before event release. */
    ~TimingEvents(){worker_.submitAndWait([&]{if(stream_ && !backend_->synchronizeStream(stream_,device_.ordinal))std::terminate();if(stop_)backend_->destroyEvent(stop_,device_.ordinal);if(start_)backend_->destroyEvent(start_,device_.ordinal);});}
    /** @brief Record start after untimed input publication. @param stream Original source stream. */
    void begin(void* stream){requireProvider(stream && (!stream_ || stream_==stream),"Timing events require one original exact stream");stream_=stream;
        requireProvider(backend_->recordEvent(start_,device_.ordinal,stream),"Native timing start failed");}
    /** @brief Record end behind complete source physical retirement. @param stream Original source stream. */
    void end(void* stream){requireProvider(stream==stream_,"Native timing end replaced its original source stream");
        requireProvider(backend_->recordEvent(stop_,device_.ordinal,stream),"Native timing stop failed");}
    /** @return Positive actual same-device native elapsed microseconds after full terminal join. */
    double microseconds(){float ms=0;worker_.submitAndWait([&]{requireProvider(backend_->eventElapsedTimeMs(start_,stop_,device_.ordinal,&ms),"Native timing interval unavailable");});
        requireProvider(std::isfinite(ms)&&ms>=0,"Invalid native event duration");return static_cast<double>(ms)*1000;}
};
/**
 * @brief Complete one local request inside its original single worker submission.
 * @param actor Original local fixture. @param events Original outer event pair.
 * @return External submit-through-completion wall microseconds, including event API overhead.
 *
 * Native events bracket only the complete graph on the owning worker. No caller
 * round trip separates event/launch/event/wait, and no oracle is evaluated here.
 */
double timedRequest(LocalActor& actor,TimingEvents& events){
    const auto start=std::chrono::steady_clock::now();
    actor.worker.submitAndWait([&]{events.begin(actor.stream());actor.fixture->launchTiming();events.end(actor.stream());actor.fixture->join();});
    return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
}
/**
 * @brief Time the original off-worker cross-vendor cohort without extra event worker submissions.
 * @param actor Original source/owner cohort. @param events Original outer source event pair.
 * @return Complete external wall microseconds; native event extraction is later and untimed.
 *
 * Public IBackend external event publication selects the exact device/stream on
 * both backends and rejects capture. Original source+owner submissions and final
 * participant joins remain unchanged; no worker guard or source API is bypassed.
 */
double timedRequest(CrossActor& actor,TimingEvents& events){
    const auto start=std::chrono::steady_clock::now();events.begin(actor.stream());
    actor.launch();events.end(actor.stream());actor.finish();
    return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
}
/** @brief Execute one shared production actor with immutable trajectory and terminal byte proof.
 * @tparam Actor Local or actual cross-vendor placement. @param actor Exact fixture. @param file Original provider.
 * @param o Frozen cell. @param worker Original source worker. */
template<class Actor>void measure(Actor& actor,ProviderFile& file,const Options& o,IWorkerGPUContext& worker){
    using P=ngram_replay::Pattern;TimingEvents events(worker,o.continuation);actor.admit();
    std::vector<double> eventSamples,wallSamples;eventSamples.reserve(o.samples);wallSamples.reserve(o.samples);
    for(std::uint32_t index=0;index<ngram_replay::PreludeFrames;++index){const auto declared=ngram_replay::scheduledFrame(scenario(o),true,index);
        const auto warm=ngram_replay::ids(o.shape,declared.pattern);actor.prepare(warm);actor.launch();actor.finish();actor.validate(warm,index==1 && (scenario(o)==ngram_replay::Scenario::WarmHits || scenario(o)==ngram_replay::Scenario::DuplicateControl)?frame(o,0).expected_hits:std::nullopt);}
    for(std::uint32_t sample=0;sample<o.samples;++sample){const auto declared=frame(o,sample);const auto ids=ngram_replay::ids(o.shape,declared.pattern);
        actor.prepare(ids);const double wall=timedRequest(actor,events);const double event=events.microseconds();
        eventSamples.push_back(event);wallSamples.push_back(wall);
        actor.validate(ids,declared.expected_hits);const auto observed=actor.observation();
        requireProvider(observed.live_ids==ids.size()&&observed.unique_ids==ngram_replay::uniqueCount(ids)&&(!declared.expected_hits || observed.gpu_hits==*declared.expected_hits),"Observed original workload differs from fixed trajectory");
        requireProvider(observed.source_rows+observed.gpu_hits==observed.unique_ids&&observed.source_native_bytes==observed.source_rows*ngram_replay::RowBytes,"Native source live extent disagrees with actual unique work");
        nlohmann::json row={{"schema",1},{"production_program_qualified",!ProviderReplayRawDiagnostic},{"raw_native_program_diagnostic",ProviderReplayRawDiagnostic},{"kind",o.placement=="local"?"local_pooled":"cross_vendor_pooled"},{"continuation",o.continuation.toString()},{"owners",{o.owner.toString()}},
            {"format","IQ4_NL"},{"width",ngram_replay::Width},{"row_bytes",ngram_replay::RowBytes},{"tokens",observed.live_ids/16},{"capacity_tokens",o.shape.capacity_tokens},{"live_ids",observed.live_ids},{"unique_ids",observed.unique_ids},
            {"scenario",o.scenario},{"phase",declared.phase},{"gpu_slots",o.gpu_slots},{"ram_slots",o.ram_slots},{"sample",sample},{"event_us",event},{"wall_us",wall},
            {"gpu_hits",observed.gpu_hits},{"source_rows",observed.source_rows},{"promotions",observed.promotions},{"evictions",observed.evictions},{"source_native_bytes",observed.source_native_bytes},{"hit_wire_bytes",observed.hit_wire_bytes},
            {"demotion_copy_bytes",nullptr},{"physical_ssd_read_bytes",nullptr},{"ram_cache_hits",nullptr},{"mapped_source_rows",nullptr},{"ram_promotions",nullptr},{"ram_evictions",nullptr},{"byte_validation",true},{"cache_regime","source_page_warmed_fixed_trajectory"},
            {"native_pattern",ngram_replay::NativePattern},{"trace_schedule",ngram_replay::Schedule},{"prelude_frames",ngram_replay::PreludeFrames},{"trajectory_index",ngram_replay::PreludeFrames+sample},{"source_locality",o.shape.locality==ngram_replay::Locality::Scattered?"affine_scattered":"contiguous_locality_control"},{"table_rows",o.shape.table_rows},{"table_bytes",o.shape.table_rows*ngram_replay::RowBytes},{"full_qfn_table",false},
            {"event_scope","source_complete_lookup_protocol_including_remote_waits"},{"wall_scope","open_interval_lookup_submit_through_complete_cohort_wait_with_outer_event_overhead"},{"admission_uploads_oracles_timed",false},{"event_includes_host_submission_gaps",o.placement!="local"},{"timing_event_worker_roundtrips",0},
            {"lookup_submission_worker_calls",o.placement=="local"?1:2},{"completion_collection_worker_calls",o.placement=="local"?0:2},
            {"provider_file_parent",std::filesystem::canonical(file.path).parent_path().string()},{"provider_file_tmpdir",std::filesystem::canonical(std::filesystem::temp_directory_path()).string()},
            {"source_event_publication",o.placement=="local"?"inside_one_original_owner_task":"public_backend_external_exact_stream"},{"provider_transport_included",true},
            // An external observer can attach without any actor-visible switch.
            // The run receipt owns instrumentation provenance; never claim an
            // unobserved run merely because this executable did not attach it.
            {"instrumentation_authority","external_run_receipt"}};
        std::cout<<"NGRAM_PERF_SAMPLE "<<row.dump()<<'\n';
    }
    const auto percentile=[](std::vector<double> values,std::size_t percent){std::sort(values.begin(),values.end());return values[(values.size()*percent+99)/100-1];};
    std::cout<<"NGRAM_PERF_SUMMARY "<<nlohmann::json{{"schema",1},{"kind",o.placement=="local"?"local_pooled":"cross_vendor_pooled"},
        {"continuation",o.continuation.toString()},{"scenario",o.scenario},{"samples",o.samples},{"latency_unit","microseconds_per_completed_lookup_request"},
        {"wall_p50_us",percentile(wallSamples,50)},{"wall_p95_us",percentile(wallSamples,95)},{"wall_max_us",*std::max_element(wallSamples.begin(),wallSamples.end())},
        {"event_p50_us",percentile(eventSamples,50)},{"event_p95_us",percentile(eventSamples,95)},{"event_max_us",*std::max_element(eventSamples.begin(),eventSamples.end())},
        {"quantile_method","nearest_rank_small_sample_diagnostic_raw_samples_retained"},{"per_row_amortization_primary",false}}.dump()<<'\n';
    const auto closing=std::chrono::steady_clock::now();actor.close();const auto closed=std::chrono::steady_clock::now();
    std::cout<<"NGRAM_PERF_TERMINAL "<<nlohmann::json{{"schema",1},{"kind",o.placement=="local"?"local_pooled":"cross_vendor_pooled"},
        {"continuation",o.continuation.toString()},{"scenario",o.scenario},{"terminal_close_us",std::chrono::duration<double,std::micro>(closed-closing).count()},
        {"scope","captured_service_terminal_close_and_original_worker_join_outside_lookup_timing"}}.dump()<<'\n';
}
/** @brief Exact owned-file residency and process I/O observations, outside the timed native interval. */
struct GroupSourceObservation final {
    std::size_t livePages=0,residentLive=0,filePages=0,residentFile=0;
    std::optional<std::uint64_t> readBytes;
    rusage usage{};
    long filesystemType=0;
};
/** @brief Observe actual source pages including original native file offset, without touching row payloads. */
GroupSourceObservation observeGroupSource(const ProviderFile& file,std::span<const std::uint64_t> ids){
    GroupSourceObservation result;
    const auto page=::sysconf(_SC_PAGESIZE);
    struct stat info{};
    struct statfs filesystem{};
    requireProvider(page>0&&::fstat(file.fd,&info)==0&&info.st_size>0&&::fstatfs(file.fd,&filesystem)==0,
        "Grouped original source observation failed");
    const auto bytes=static_cast<std::size_t>(info.st_size);
    result.filePages=(bytes+static_cast<std::size_t>(page)-1)/static_cast<std::size_t>(page);
    std::vector<unsigned char> state(result.filePages);
    void* mapping=::mmap(nullptr,bytes,PROT_NONE,MAP_SHARED,file.fd,0);
    requireProvider(mapping!=MAP_FAILED,"Grouped source residency mapping failed");
    const auto status=::mincore(mapping,bytes,state.data());
    const auto unmap=::munmap(mapping,bytes);
    requireProvider(status==0&&unmap==0,"Grouped source residency observation failed");
    std::set<std::size_t> live;
    const auto source=file.descriptor();
    for(const auto row:ids){
        requireProvider(row<source.rows,"Grouped source observation exceeds authored extent");
        const auto first=source.file_offset+row*file.row_bytes;
        const auto last=first+file.row_bytes-1;
        for(auto index=first/static_cast<std::size_t>(page);index<=last/static_cast<std::size_t>(page);++index)live.insert(index);
    }
    result.livePages=live.size();
    for(const auto index:live)result.residentLive+=state.at(index)&1u;
    for(const auto value:state)result.residentFile+=value&1u;
    result.filesystemType=filesystem.f_type;
    return result;
}
/** @brief Capture process counters after pre-observation and before post-observation scratch can add faults. */
void observeGroupCounters(GroupSourceObservation& result){
    std::ifstream io("/proc/self/io");
    std::string key;
    std::uint64_t value;
    while(io>>key>>value)if(key=="read_bytes:")result.readBytes=value;
    requireProvider(::getrusage(RUSAGE_SELF,&result.usage)==0,"Grouped source fault observation failed");
}

/** @brief Declare the exact ConsumerLocal owner/consumer/source topology without native discovery. */
ImmutableRowTableLayoutGeometry consumerLocalLayout(const Options& o,const ImmutableRowTableSource& source){
    const PhysicalMemoryAllocatorIdentity host{0,DeviceId::cpu()};
    std::vector<PhysicalMemoryAllocatorIdentity> consumers;
    std::vector<ImmutableRowTableLayoutOwner> owners;
    for(const auto consumer:o.consumers){
        consumers.push_back({0,consumer});
        owners.push_back({{{0,consumer},ImmutableRowStorage::VRAM},0,o.gpu_slots,{}});
    }
    owners.push_back({{host,ImmutableRowStorage::RAM},0,o.ram_slots,{}});
    const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped};
    return ImmutableRowTableLayoutGeometry::checked(
        ImmutableRowTableExtent::fromNative(source.format,source.rows,source.rowBytes()),
        ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,std::move(owners),consumers,{},
        {{host,ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped,o.mmapWindowBytes,o.mmapWindowCount)}},{},std::nullopt,
        o.shape.capacity_tokens*16,ImmutableRowTableAdaptiveGPUDomain::ConsumerLocal);
}
/** @brief Preserve the existing fixture's original independent GPU and shared host admission policies. */
AdaptiveVRAMRAMSSDRequest consumerLocalRequest(const Options& o){
    AdaptiveVRAMRAMSSDRequest request;
    request.maximum_gpu_cache_rows=o.gpu_slots;
    request.gpu_policy={65536,1,2};
    request.host.maximum_cache_rows=o.ram_slots;
    return request;
}
/** @brief Replay actual CLI and plural geometry with no prepared context, backend probe or native creation. */
void proveConsumerLocalDeclarations(){
    const ImmutableRowTableSource source{.path="unused-pure-geometry",.tensor_name="per_layer_token_embd.weight",
        .rows=1048576,.row_elements=ngram_replay::Width,.format=GGUFTensorType::IQ4_NL};
    for(const auto backend:{std::string("cuda"),std::string("rocm")}){
        for(const auto slots:{std::string("0"),std::string("64")}){
            std::vector<std::string> args{"component","--placement","consumer_local","--continuation",backend+":0",
                "--consumers",backend+":0,"+backend+":1","--tokens","8","--capacity-tokens","64",
                "--gpu-slots",slots,"--ram-slots","8","--scenario","alternating_sets","--requests","2","--cold-attempt","1"};
            const auto parseArgs=[&]{
                std::vector<char*> argv;
                for(auto& arg:args)argv.push_back(arg.data());
                return options(static_cast<int>(argv.size()),argv.data());
            };
            const auto defaults=parseArgs();
            requireProvider(defaults.mmapWindowBytes==4096&&defaults.mmapWindowCount==2,
                "Default ConsumerLocal mmap stress geometry changed");
            args.insert(args.end(),{"--mmap-window-bytes","65536","--mmap-window-count","1024"});
            const auto parsed=parseArgs();
            const auto layout=consumerLocalLayout(parsed,source);
            const auto request=consumerLocalRequest(parsed);
            requireProvider(layout.sourceServices().size()==1&&
                layout.sourceServices().front().windows.windowBytes()==65536&&
                layout.sourceServices().front().windows.windowCount()==1024,
                "Explicit mmap geometry lost the original layout/source policy");

            const auto shared=AdaptiveImmutableLookupGeometry::sharedHostGeometry(layout,request);
            requireProvider(shared.sourceBytes()==ImmutableRowTable::physicalBytes(source,layout.sourceServices().front().windows),
                "Explicit mmap source BOM differs from original source storage authority");
            const auto defaultLayout=consumerLocalLayout(defaults,source);
            const auto defaultHost=AdaptiveImmutableLookupGeometry::sharedHostGeometry(defaultLayout,consumerLocalRequest(defaults));
            requireProvider(shared.sourceBytes()>defaultHost.sourceBytes(),"Larger mmap geometry did not reach original source BOM");

            requireProvider(layout.consumers().size()==2&&layout.adaptiveGPUDomain()==ImmutableRowTableAdaptiveGPUDomain::ConsumerLocal,
                "CLI lost actual plural ConsumerLocal membership");
            std::optional<q::ImmutableLookupLayout> channel;
            for(const auto consumer:layout.consumers()){
                const auto local=AdaptiveImmutableLookupGeometry::forRequester(layout,source,request,consumer);
                requireProvider(local.physical().owners().size()==1&&local.physical().owners().front().allocator==consumer,
                    "Consumer-local owner projection changed actual requester");
                requireProvider(shared.executionBytes()>local.host().executionBytes(),"Shared source lost its concurrent grant storage");
                channel=local.channelLayout();
            }
            const auto frames=q::ImmutableLookupServiceGeometry::checked(*channel,parsed.consumers);
            requireProvider(frames.channelCount()==2&&frames.offset(0)!=frames.offset(1)&&
                frames.offset(1)>=frames.offset(0)+channel->bytes(),"Consumer service frames alias");
            const auto rejectGeometry=[&](std::size_t index,const std::string& invalid){
                const auto saved=args[index];args[index]=invalid;bool bad=false;
                try{(void)parseArgs();}catch(const std::exception&){bad=true;}
                args[index]=saved;requireProvider(bad,"Invalid or unsupported explicit mmap geometry was accepted");
            };
            rejectGeometry(args.size()-3,"0");
            rejectGeometry(args.size()-3,"1"); // Original source policy owns page alignment.
            rejectGeometry(args.size()-1,"0");
            rejectGeometry(2,"local"); // Explicit geometry must not be ignored by another actor.
            args[6]=backend+":0,"+backend+":0";
            bool rejected=false;
            try{(void)parseArgs();}catch(const std::exception&){rejected=true;}
            requireProvider(rejected,"Duplicate consumer CLI was accepted");
            args[6]=backend+":0,"+(backend=="cuda"?"rocm:0":"cuda:0");
            rejected=false;
            try{(void)parseArgs();}catch(const std::exception&){rejected=true;}
            requireProvider(rejected,"Mixed-vendor compute cohort was silently changed");
            args[6]=backend+":1,"+backend+":0";
            rejected=false;
            try{(void)parseArgs();}catch(const std::exception&){rejected=true;}
            requireProvider(rejected,"Foreign first continuation was accepted");
        }
    }
    std::cout<<"POOLED_CONSUMER_LOCAL_DECLARATIONS_PASS backends=2 capacities=0,64 original_channels=2 native_creation=0\n";
}

/** @brief Actual concurrent ConsumerLocal native owners around one original subordinate source and I/O worker.
 * Uses the existing production plural layout projections and complete-group admission.
 * All graphs remain participant-local and the existing native byte oracle is unchanged.
 */
class ConsumerLocalActor final {
public:
    /** @brief Freeze complete concurrent payload/native BOM before binding any provider, channel or graph. */
    ConsumerLocalActor(std::shared_ptr<ProviderFile> file,const Options& options):o(options),file_(std::move(file)){
        session_=std::shared_ptr<PreparedNativeLookupFixture>(PreparedNativeLookupFixture::prepare(o.consumers));
        const auto source=file_->descriptor();
        const auto layout=consumerLocalLayout(o,source);
        const auto request=consumerLocalRequest(o);
        std::vector<PhysicalMemoryResource> resources;
        for(const auto consumer:o.consumers)resources.push_back(session_->resource(consumer));
        const auto shared=AdaptiveImmutableLookupGeometry::sharedHostGeometry(layout,request);
        const auto cpu=session_->resource(DeviceId::cpu());
        const auto ordering=NativeOrderingEventBudgetProjection::installed();
        PhysicalMemoryPlanBuilder bom;
        for(const auto& contribution:{
                AdaptiveImmutableLookupGeometry::consumerLocalMandatoryBOM(layout,source,request,resources,cpu,ordering),
                AdaptiveImmutableLookupGeometry::consumerLocalSlotDependentBOM(layout,source,request,resources,cpu,ordering)})
            for(const auto& row:contribution.resources())bom.add(row);
        const auto io=RetainedHostServiceProgram::original(RetainedHostServiceRole::ImmutableLookupIO);
        const RetainedHostServiceInstance instance{0,DeviceId::cpu(),RetainedHostServiceRole::ImmutableLookupIO,0};
        RetainedHostServiceDeclarations::Builder services(0);
        services.add(instance,io);
        const auto declarations=std::move(services).freeze();
        const auto ioBOM=declarations->bom(instance,cpu);
        for(const auto& row:ioBOM.resources())bom.add(row);
        const ProviderReplaySpec spec{file_,o.shape.capacity_tokens*16,o.gpu_slots,o.ram_slots,n::OriginalIdWidth::UInt64,
            true,OwnerAdmission::Original,ACKInterleaving::None,ActivationPreparation::BeforeGraph,o.shape.capacity_tokens*16,true};
        for(const auto consumer:o.consumers){
            auto geometry=ProviderReplayGeometry::create(spec,consumer,*session_,io);
            const auto local=AdaptiveImmutableLookupGeometry::forRequester(layout,source,request,{0,consumer});
            geometry.physical=local.physical();
            geometry.policy=local.coordinator();
            geometry.cpu=shared;
            geometry.channel=local.channelLayout();
            bom.add(session_->resource(consumer),PhysicalMemoryOwner::ActivationArena,geometry.activation);
            for(const auto& root:geometry.declarations)bom.add(cpu,PhysicalMemoryOwner::ActivationArena,root.physicalBytes());
            const auto graphBOM=RetainedParticipantGraphExecutable::generalBOM(cpu,session_->resource(consumer),geometry.graphCount);
            for(const auto& row:graphBOM.resources())bom.add(row);
            programs_.push_back({spec,std::move(geometry),session_,consumer,{},instance});
        }
        session_->admit(bom.build());
        auto binding=declarations->bind(session_->memory(),std::span{&instance,std::size_t{1}});
        provider_=HostImmutableRowProvider::fromSubordinateAdaptiveCache(CpuAdaptiveImmutableRowCache::create(source,shared,session_->memory()));
        service_=q::ImmutableLookupService::create(provider_,programs_.front().geometry.channel,o.consumers,
            session_->memory(),DeviceId::cpu(),binding->take(instance));
        try{
            for(auto& program:programs_){
                program.sharedProvider=provider_;
                program.sharedService=service_;
                graphs_.push_back(std::make_unique<ProviderReplayGraphs>(session_->permit(program.device),program.geometry.graphCount));
                fixtures_.push_back(nullptr);
                const auto index=fixtures_.size()-1;
                worker(index).submitAndWait([&,index]{fixtures_[index]=std::make_unique<ProviderReplay>(programs_[index],*graphs_[index]);});
                events_.push_back(std::make_unique<TimingEvents>(worker(index),program.device));
            }
        }catch(...){release();throw;}
    }
    /** @brief Retain one source authority through all graph, payload and native-reset teardown. */
    ~ConsumerLocalActor()noexcept{try{release();}catch(const std::exception& failure){std::cerr<<"FATAL grouped provider cleanup: "<<failure.what()<<'\n';std::terminate();}}
    /** @brief Arm all exact channel tickets before either participant publishes admission or lookup work. */
    void admit(){
        std::vector<q::ImmutableLookupExecutionBoundary::SourceAdmissionParticipant> members;
        ++requestEpoch_;
        for(const auto& fixture:fixtures_)members.push_back({fixture->boundary,requestEpoch_});
        group_=q::ImmutableLookupExecutionBoundary::armSourceGroup(members);
        for(std::size_t i=0;i<fixtures_.size();++i)worker(i).submitAndWait([&,i]{fixtures_[i]->admit(requestEpoch_);});
    }
    /** @brief Prepare both channels outside timing, then submit both exact original streams before any completion collection. */
    void sample(std::span<const std::uint64_t> ids,std::optional<std::uint32_t> expectedHits,bool report,std::uint32_t index){
        for(std::size_t i=0;i<fixtures_.size();++i)worker(i).submitAndWait([&,i]{fixtures_[i]->prepareTiming(ids);});
        const auto beforeAdvice=observeGroupSource(*file_,ids);
        const auto advice=o.coldAttempt?::posix_fadvise(file_->fd,0,0,POSIX_FADV_DONTNEED):0;
        auto before=observeGroupSource(*file_,ids);
        observeGroupCounters(before);
        std::array<double,2> completionUs{};
        std::array<double,2> eventUs{};
        std::vector<std::future<void>> pending;
        const auto start=std::chrono::steady_clock::now();
        for(std::size_t i=0;i<fixtures_.size();++i)pending.push_back(worker(i).submitAsync([&,i]{
            events_[i]->begin(fixtures_[i]->stream);
            fixtures_[i]->launchTiming();
            events_[i]->end(fixtures_[i]->stream);
            fixtures_[i]->join();
            completionUs[i]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
        }));
        std::exception_ptr primary;
        for(auto& future:pending)try{future.get();}catch(...){if(!primary)primary=std::current_exception();}
        if(primary)std::rethrow_exception(primary);
        const auto complete=std::chrono::steady_clock::now();
        GroupSourceObservation completedCounters;
        observeGroupCounters(completedCounters);
        auto after=observeGroupSource(*file_,ids);
        after.readBytes=completedCounters.readBytes;
        after.usage=completedCounters.usage;
        nlohmann::json channels=nlohmann::json::array();
        for(std::size_t i=0;i<fixtures_.size();++i){
            eventUs[i]=events_[i]->microseconds();
            worker(i).submitAndWait([&,i]{fixtures_[i]->validateTiming(ids);});
            const auto observed=fixtures_[i]->lastObservation;
            requireProvider(!expectedHits||observed.gpu_hits==*expectedHits,"Grouped GPU cache trajectory differs from its original trace");
            requireProvider(observed.live_ids==ids.size()&&observed.unique_ids==ngram_replay::uniqueCount(ids)&&
                observed.source_rows+observed.gpu_hits==observed.unique_ids&&observed.source_native_bytes==observed.source_rows*file_->row_bytes,
                "Grouped consumer source or byte extent changed");
            channels.push_back({{"consumer",programs_[i].device.toString()},{"completion_wall_us",completionUs[i]},
                {"event_us",eventUs[i]},{"live_ids",observed.live_ids},{"unique_ids",observed.unique_ids},
                {"gpu_hits",observed.gpu_hits},{"source_rows",observed.source_rows},{"source_native_bytes",observed.source_native_bytes},
                {"promotions",observed.promotions},{"evictions",observed.evictions},{"hit_wire_bytes",observed.hit_wire_bytes}});
        }
        requireAssertions();
        if(!report)return;
        const bool known=before.readBytes&&after.readBytes&&*after.readBytes>=*before.readBytes;
        const auto bytes=known?*after.readBytes-*before.readBytes:0;
        const bool cold=o.coldAttempt&&advice==0&&before.livePages&&before.residentLive==0&&known&&bytes&&before.filesystemType!=0x01021994;
        std::cout<<"NGRAM_GROUP_SAMPLE "<<nlohmann::json{{"schema",1},{"kind","consumer_local_pooled"},
            {"profile","component_only"},{"production_program_qualified",!ProviderReplayRawDiagnostic},
            {"raw_native_program_diagnostic",ProviderReplayRawDiagnostic},{"full_model_tp_qualified",false},{"request_epoch",requestEpoch_},{"sample",index},{"scenario",o.scenario},
            {"trace_schedule",ngram_replay::Schedule},{"trajectory_index",ngram_replay::PreludeFrames+index},
            {"tokens_per_consumer",ids.size()/16},{"capacity_tokens_per_consumer",o.shape.capacity_tokens},
            {"query_heads_per_consumer",16},{"format","IQ4_NL"},{"width",file_->columns},{"row_bytes",file_->row_bytes},
            {"mmap_window_bytes",o.mmapWindowBytes},{"mmap_window_count",o.mmapWindowCount},
            {"table_rows",file_->rows},{"full_qfn_table",false},{"gpu_slots_per_consumer",o.gpu_slots},{"shared_ram_slots",o.ram_slots},
            {"channels",channels},{"group_wall_us",std::chrono::duration<double,std::micro>(complete-start).count()},
            {"scope","both_original_enqueues_through_all_native_completion_excluding_inputs_oracles_and_terminal"},
            {"source_copies",1},{"io_workers",1},{"physical_io_scope","process_not_per_file_or_channel_attribution"},
            {"read_bytes",known?nlohmann::json(bytes):nlohmann::json(nullptr)},
            {"major_faults",after.usage.ru_majflt-before.usage.ru_majflt},{"minor_faults",after.usage.ru_minflt-before.usage.ru_minflt},
            {"cache_regime",cold?"os_cold_confirmed_block_io":o.coldAttempt?"file_cold_attempt_unconfirmed":before.residentLive==before.livePages?"os_hot_live_pages":"mixed_os_residency"},
            {"file_advice_result",o.coldAttempt?nlohmann::json(advice):nlohmann::json(nullptr)},
            {"live_pages",before.livePages},{"resident_live_before",before.residentLive},{"resident_live_after",after.residentLive},
            {"file_pages",before.filePages},{"resident_file_before_advice",beforeAdvice.residentFile},
            {"resident_file_before",before.residentFile},{"resident_file_after",after.residentFile},
            {"filesystem_type",before.filesystemType},{"source_is_sparse_authored_component",true},{"byte_validation",true}}.dump()<<'\n';
    }
    /** @brief Publish and join BOTH original native terminals before consuming either shared-service channel await. */
    void close(){
        if(!group_)return;
        std::exception_ptr primary;
        for(std::size_t i=0;i<fixtures_.size();++i)if(fixtures_[i]->boundary->hasActiveInterval())try{
            worker(i).submitAndWait([&,i]{fixtures_[i]->publishClose();graphs_[i]->publishRequestTerminal();fixtures_[i]->join();});
        }catch(...){if(!primary)primary=std::current_exception();}
        if(primary)group_.reset();
        for(auto& fixture:fixtures_)if(fixture->boundary->hasActiveInterval())try{fixture->boundary->awaitTerminal();}catch(...){if(!primary)primary=std::current_exception();}
        if(group_){try{requireProvider(group_->finish(),"Original grouped service terminal failed");}catch(...){if(!primary)primary=std::current_exception();}group_.reset();}
        if(primary)std::rethrow_exception(primary);
        nlohmann::json channels=nlohmann::json::array();
        for(std::size_t i=0;i<fixtures_.size();++i){
            graphs_[i]->observeRequestTerminal();
            fixtures_[i]->validateClose();
            const auto terminal=fixtures_[i]->boundary->terminalObservation();
            const auto& counters=terminal.counters;
            channels.push_back({{"consumer",programs_[i].device.toString()},{"requests",counters.requests},{"ids",counters.ids},
                {"native_bytes",counters.nativeBytes},{"ram_rows",counters.ramRows},{"source_rows",counters.sourceRows},
                {"cache_admissions",counters.cacheAdmissions},{"cache_evictions",counters.cacheEvictions},
                {"window_hits",counters.windowHits},{"window_misses",counters.windowMisses},{"window_evictions",counters.windowEvictions}});
        }
        requireAssertions();
        const auto source=provider_->sourceStats();
        std::cout<<"NGRAM_GROUP_TERMINAL "<<nlohmann::json{{"schema",1},{"channels",channels},{"shared_source",true},
            {"mmap_window_bytes",o.mmapWindowBytes},{"mmap_window_count",o.mmapWindowCount},
            {"shared_cache",true},{"same_pma",true},{"all_original_channel_terminals_retired",true},
            {"source_requests_cumulative",source.requests},{"source_rows_cumulative",source.rows},
            {"source_copied_bytes_cumulative",source.copied_bytes},{"source_window_hits_cumulative",source.window_hits},
            {"source_window_misses_cumulative",source.window_misses},{"source_window_evictions_cumulative",source.window_evictions}}.dump()<<'\n';
    }
private:
    /** @return Exact source-prepared worker for this declared consumer. */
    IWorkerGPUContext& worker(std::size_t index)const{return *session_->permit(programs_.at(index).device).worker();}
    /** @brief Graph handles precede embedded payloads; all shared owners precede original physical reset. */
    void release(){
        if(!session_)return;
        close();
        events_.clear();
        for(std::size_t i=0;i<fixtures_.size();++i){worker(i).submitAndWait([&,i]{if(fixtures_[i])fixtures_[i]->join();});graphs_[i]->retire();}
        for(std::size_t i=0;i<fixtures_.size();++i)worker(i).submitAndWait([&,i]{fixtures_[i].reset();});
        fixtures_.clear();graphs_.clear();programs_.clear();service_.reset();provider_.reset();
        session_->requirePayloadRetired();session_->retire();session_.reset();
    }
    Options o;
    std::shared_ptr<ProviderFile> file_;
    std::shared_ptr<PreparedNativeLookupFixture> session_;
    std::vector<PreparedProviderReplayProgram> programs_;
    std::shared_ptr<HostImmutableRowProvider> provider_;
    std::shared_ptr<q::ImmutableLookupService> service_;
    std::vector<std::unique_ptr<ProviderReplayGraphs>> graphs_;
    std::vector<std::unique_ptr<ProviderReplay>> fixtures_;
    std::vector<std::unique_ptr<TimingEvents>> events_;
    std::unique_ptr<q::ImmutableLookupExecutionBoundary::SourceGroupAdmission> group_;
    std::uint64_t requestEpoch_=0; ///< Original host-owned request admission identity, never a device-loop mirror.
};
/** @brief Replay the unchanged canonical prelude/samples through actual concurrent pooled consumers. */
void measureConsumerLocal(std::shared_ptr<ProviderFile> file,const Options& o){
    requireProvider(::fsync(file->fd)==0,"Authored grouped source fsync failed");
    ConsumerLocalActor actor(std::move(file),o);
    try{
      for(std::uint32_t request=0;request<o.requests;++request){
        actor.admit();
        for(std::uint32_t index=0;index<ngram_replay::PreludeFrames;++index){
            const auto declared=ngram_replay::scheduledFrame(scenario(o),true,index);
            actor.sample(ngram_replay::ids(o.shape,declared.pattern),std::nullopt,false,index);
        }
        for(std::uint32_t sample=0;sample<o.samples;++sample){
            const auto declared=frame(o,sample);
            actor.sample(ngram_replay::ids(o.shape,declared.pattern),declared.expected_hits,true,sample);
        }
        actor.close();
      }
    }catch(const std::exception& failure){std::cerr<<"PRIMARY grouped lookup failure: "<<failure.what()<<'\n';throw;}
}

#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
/** @brief Exercise actual authored owner definitions through one predeclared source/PMA and unchanged byte oracle.
 * @param file Original immutable provider. @param o Explicit diagnostic geometry/lifetimes.
 * No timing events or per-instance source re-admission are introduced.
 */
void calibrateRemote(std::shared_ptr<ProviderFile> file,const Options& o){
    std::vector<CrossVendorReadReplaySpec> specs;
    for(std::uint32_t instance=0;instance<o.lifetimes;++instance)specs.push_back({file,o.gpu_slots,o.shape.capacity_tokens*16,o.shape.capacity_tokens*16,o.ram_slots,true,o.remoteHandshake});
    CrossVendorReadReplayPrograms programs(std::move(specs));programs.prepare(o.continuation,o.owner);
    for(std::uint32_t instance=0;instance<o.lifetimes;++instance)programs.run(instance,[&](CrossVendorReadReplay& replay){
        const auto receipt=programs.program(instance).calibration;requireProvider(bool(receipt),"Authored remote instance lost its original diagnostic receipt");
        receipt->checkpoint("native_program_ready");
        struct OriginalRequestClosure {CrossVendorReadReplay& replay;bool active=false;
            ~OriginalRequestClosure()noexcept{if(active)try{replay.close(true);}catch(...){std::terminate();}}
        } originalClose{replay};
        for(std::uint32_t request=0;request<2;++request){replay.admit(std::uint64_t{instance}*2+request+1);originalClose.active=true;
            for(std::uint32_t index=0;index<o.samples;++index){
                const auto ids=ngram_replay::authoredRemoteLifecycleIds(o.shape,index);
                replay.prepareTiming(ids);const auto submitted=std::chrono::steady_clock::now();replay.timingLaunch();replay.timingJoin();const auto completed=std::chrono::steady_clock::now();
                replay.validateTiming(ids,std::nullopt);requireAssertions();
                const auto observed=replay.lastObservation;requireProvider(observed.live_ids==ids.size() && observed.unique_ids==ngram_replay::uniqueCount(ids) &&
                    observed.source_rows+observed.gpu_hits==observed.unique_ids && observed.source_native_bytes==observed.source_rows*ngram_replay::RowBytes,
                    "Authored remote live source/owner extent or byte oracle failed");
                receipt->checkpoint(index?"native_replay_complete":"native_first_use_complete",{{"request",request},{"replay",index},{"diagnostic_completed_request_wall_us",std::chrono::duration<double,std::micro>(completed-submitted).count()},
                    {"wall_scope","original_cohort_submit_through_both_native_stream_joins_instrumented"},{"live_ids",observed.live_ids},
                    {"unique_ids",observed.unique_ids},{"gpu_hits",observed.gpu_hits},{"source_rows",observed.source_rows},{"source_native_bytes",observed.source_native_bytes},{"hit_wire_bytes",observed.hit_wire_bytes},{"byte_validation",true}});
            }
            originalClose.active=false;replay.close(request==0);requireAssertions();receipt->checkpoint(request==0?"typed_request_reset_complete":"terminal_complete");
        }
    });
    programs.retire();requireAssertions();
}
#endif
/** @brief Initialize exact selected devices and run one declared Release component cell. */
void run(const Options& o){
#ifndef LLAMINAR_NGRAM_PERF_RELEASE
    throw std::logic_error("Pooled timing requires coherent Release core and executable");
#endif
    if(!hasCPUBackend())initCPUBackend(-1);
    const auto selected=o.consumers.empty()?std::vector{o.continuation,o.owner}:o.consumers;
    for(const auto d:selected){auto* backend=getBackendFor(d);requireProvider(backend&&backend->deviceCount()>d.ordinal,"Explicit original measurement GPU unavailable");}
    std::vector<std::uint64_t> authored;
    for(const auto pattern:{ngram_replay::Pattern::DistinctA,ngram_replay::Pattern::DistinctB,ngram_replay::Pattern::DistinctC}){const auto values=ngram_replay::ids(o.shape,pattern);authored.insert(authored.end(),values.begin(),values.end());}
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
    if(o.diagnosticProgram==ProviderReplayDiagnosticProgram::AuthoredRemoteLookup){
        auto shape=o.shape;shape.tokens=shape.capacity_tokens;for(const auto pattern:{ngram_replay::Pattern::DistinctA,ngram_replay::Pattern::DuplicateHeavy}){
            const auto values=ngram_replay::ids(shape,pattern);authored.insert(authored.end(),values.begin(),values.end());}
    }
#endif
    auto file=std::make_shared<ProviderFile>(o.shape.table_rows,authored);
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
    if(o.diagnosticProgram==ProviderReplayDiagnosticProgram::AuthoredRemoteLookup){calibrateRemote(file,o);return;}
#endif
    if(o.placement=="consumer_local"){measureConsumerLocal(file,o);}
    else if(o.placement=="local"){LocalActor actor(file,o);measure(actor,*file,o,actor.worker);}
    else {CrossActor actor(file,o);measure(actor,*file,o,actor.worker);}
    requireAssertions();
}
} // namespace
} // namespace llaminar2::test
/** @brief Own process-native retirement; every assertion/exception fails the measurement. */
int main(int argc,char** argv){
    if(argc==2&&std::string_view(argv[1])=="--self-test"){
        try{llaminar2::test::proveConsumerLocalDeclarations();return 0;}
        catch(const std::exception& failure){std::cerr<<"FATAL "<<failure.what()<<'\n';return 1;}
    }
    llaminar2::GPUDeviceProcessLifetime lifetime;
    try{::testing::InitGoogleTest(&argc,argv);const auto o=llaminar2::test::options(argc,argv);llaminar2::test::run(o);return 0;}
    catch(const std::exception& e){std::cerr<<"FATAL "<<e.what()<<'\n';return 1;}}
