/**
 * @file CPUQ8ResidualPreparedMeasurement.h
 * @brief Model-free prepared-source CPU timing in the existing projection harness.
 *
 * Source preparation, independent oracle checks and terminal observations are
 * outside elapsed intervals. Both policies retain one actual source/store and
 * admitted primary/residual workspace. This is a component experiment, never a
 * model/corpus certificate. Single-policy runs support isolated profiling;
 * paired runs alternate AB/BA without attributing setup cost to projection.
 */
#pragma once
#include "utils/Q8ResidualAllFormatCPUHarness.h"
#include "utils/CPUFeatures.h"
#include <chrono>
#include <cstdlib>
#include <set>
#include <nlohmann/json.hpp>
#include <sched.h>

namespace llaminar2::test::q8_all_formats
{
/** @brief Complete explicit CPU measurement declaration, separate from legacy GPU options. */
struct CPUMeasurementOptions
{
    std::string format,phase,policy;
    int m=0,n=0,k=0,live=-1,workers=0,iterations=0,batches=0,warmup=0;
};
/** @brief Reject malformed setup before a real source/engine is prepared. */
inline void cpuMeasurementRequire(bool value,const char *why)
{
    if(!value)throw std::invalid_argument(why);
}
/** @brief Select only an explicit CPU request; unknown backend still uses the original parser. */
inline bool cpuQ8ResidualMeasurementRequested(int argc,char **argv)
{
    for(int i=1;i+1<argc;++i)
        if(std::string_view(argv[i])=="--backend" && std::string_view(argv[i+1])=="cpu")return true;
    return false;
}
/** @brief Parse exact CPU-only options without changing legacy GPU invocation behavior. */
inline CPUMeasurementOptions cpuMeasurementOptions(int argc,char **argv)
{
    CPUMeasurementOptions o;
    std::set<std::string> seen;
    const auto number=[](const std::string &text){
        size_t end=0;int value=std::stoi(text,&end);
        cpuMeasurementRequire(end==text.size(),"CPU timing integer has trailing text");return value;
    };
    for(int i=1;i<argc;i+=2)
    {
        cpuMeasurementRequire(i+1<argc,"CPU timing option requires a value");
        const std::string key=argv[i],value=argv[i+1];
        cpuMeasurementRequire(seen.insert(key).second,"Duplicate CPU timing option");
        if(key=="--backend")cpuMeasurementRequire(value=="cpu","CPU timing backend differs");
        else if(key=="--source-format")o.format=value;
        else if(key=="--phase")o.phase=value;
        else if(key=="--policy")o.policy=value;
        else if(key=="--m")o.m=number(value);
        else if(key=="--n")o.n=number(value);
        else if(key=="--k")o.k=number(value);
        else if(key=="--live")o.live=number(value);
        else if(key=="--threads")o.workers=number(value);
        else if(key=="--iterations")o.iterations=number(value);
        else if(key=="--batches")o.batches=number(value);
        else if(key=="--warmup")o.warmup=number(value);
        else throw std::invalid_argument("Unknown CPU timing option");
    }
    cpuMeasurementRequire(seen.size()==12 && seen.contains("--backend"),"All12 CPU timing options are required");
    cpuMeasurementRequire(o.m>0&&o.m<=2048&&o.n>0&&o.n<=10240&&o.k>0&&o.k<=65536&&o.k%32==0,
        "CPU timing matrix exceeds bounded component geometry");
    cpuMeasurementRequire(size_t(o.n)*size_t(o.k)<=size_t{64}*1024*1024,
        "CPU timing source exceeds bounded prepared fixture extent");
    cpuMeasurementRequire(o.live>=0&&o.live<=o.m&&o.workers>0&&o.workers<=omp_get_max_threads(),
        "CPU timing live rows/team exceed the caller's admitted physical-core budget");
    cpuMeasurementRequire(o.phase=="ordinary"||o.phase=="verifier"||o.phase=="serial","Unknown CPU timing phase");
    cpuMeasurementRequire((o.phase!="ordinary"||o.live==o.m)&&(o.phase!="serial"||(o.m==1&&o.live==1)),
        "Ordinary CPU timing requires compact live rows; serial requires one complete row");
    cpuMeasurementRequire(o.policy=="single_q8"||o.policy=="two_term_q8"||o.policy=="paired","Unknown CPU timing policy");
    cpuMeasurementRequire(o.iterations>0&&o.iterations<=65536&&o.batches>0&&o.batches<=20&&o.batches%2==0&&o.warmup>0&&o.warmup<=64,
        "CPU timing needs bounded positive repetitions and balanced batches");
    return o;
}
/** @return Actual calling-thread CPU affinity, not an inferred physical socket claim. */
inline std::vector<int> cpuMeasurementAffinity()
{
    cpu_set_t mask;CPU_ZERO(&mask);
    if(sched_getaffinity(0,sizeof(mask),&mask)!=0)throw std::runtime_error("CPU timing cannot observe actual affinity");
    std::vector<int> result;
    for(int cpu=0;cpu<CPU_SETSIZE;++cpu)if(CPU_ISSET(cpu,&mask))result.push_back(cpu);
    return result;
}
/**
 * @brief Execute one real prepared-source CPU experiment with exact policy identity.
 * @return Zero only after all timed replay/source/guard checks complete.
 * @throws std::exception For missing source, malformed setup or changed output/source.
 */
inline int runCPUQ8ResidualMeasurement(int argc,char **argv)
{
    const auto o=cpuMeasurementOptions(argc,argv);
    const auto &formats=quantizedVerifierFormats();
    const auto entry=std::find_if(formats.begin(),formats.end(),[&](const auto &f){return o.format==f.label;});
    cpuMeasurementRequire(entry!=formats.end(),"CPU timing source is absent from the canonical21 catalog");
    // Raw source fixture construction must obey original native block geometry.
    // Prepared K32 slices are validated separately by source-aware loader tests.
    const auto *source_format=native_vnni_formats::forQuantType(o.format);
    cpuMeasurementRequire(source_format&&(!source_format->is_superblock||o.k%256==0),
        "CPU timing raw source requires complete original superblocks");
    Threads team(o.workers);
    int actual_workers=0;
    std::vector<std::vector<int>> worker_affinities(size_t(o.workers));
    const auto observe_team=[&]() {
        const int worker=omp_get_thread_num();
        if(worker<o.workers)worker_affinities[size_t(worker)]=cpuMeasurementAffinity();
#pragma omp single
        actual_workers=omp_get_num_threads();
    };
    OMP_WORKSHARE_REGION(observe_team);
    cpuMeasurementRequire(actual_workers==o.workers,"CPU timing did not receive its requested team");
    CPUFixture f(*entry,o.m,o.n,o.k,false);
    auto two_params=f.stage->getParams();
    two_params.force_decode_equivalent_verifier_prefill=o.phase=="verifier";
    f.stage->unbindWorkspace();
    f.stage=std::make_unique<GEMMStage>(two_params);
    f.stage->bindWorkspace(f.workspace->get());
    auto single_params=two_params;
    single_params.quantized_operand_precision=QuantizedActivationOperandPrecision::SingleQ8;
    single_params.C=&f.baseline;
    GEMMStage single(single_params);
    single.bindWorkspace(f.workspace->get());
    /** @brief Revoke the second stage's borrow before shared scratch retires. */
    struct SingleBorrow {GEMMStage &stage;~SingleBorrow(){stage.unbindWorkspace();}} borrow{single};
    CPUDeviceContext context(DeviceId::cpu(),o.workers);
    f.count=o.live;
    const auto execute=[&](GEMMStage &stage){if(!stage.execute(&context))throw std::runtime_error("CPU prepared stage submission failed");};
    std::fill_n(f.output.mutable_data(),f.output.numel(),-12345.f);
    std::fill_n(f.baseline.mutable_data(),f.baseline.numel(),-12345.f);
    std::fill(f.input.mutable_data()+size_t(o.live)*o.k,f.input.mutable_data()+f.input.numel(),
        std::numeric_limits<float>::quiet_NaN());
    for(int i=0;i<o.warmup;++i)
    {
        if(o.policy!="two_term_q8")execute(single);
        if(o.policy!="single_q8")execute(*f.stage);
    }
    const std::vector<float> expected_single=o.policy=="two_term_q8"?std::vector<float>{}:
        std::vector<float>(f.baseline.data(),f.baseline.data()+size_t(o.live)*o.n);
    const std::vector<float> expected_two=o.policy=="single_q8"?std::vector<float>{}:
        std::vector<float>(f.output.data(),f.output.data()+size_t(o.live)*o.n);
    const auto *packed=f.kernel->exportCPUNativeVNNIPackedWeights();
    cpuMeasurementRequire(packed!=nullptr,"CPU timing source export disappeared");
    const auto *source_pointer=packed->interleavedBase();
    const std::vector<uint8_t> prepared_before(source_pointer,source_pointer+cpu::native_vnni::interleavedWorkspaceSize(*packed));
    // A bounded independent prepared-source FP64 check gates timing. Full matrix
    // source/oracle and serial-byte coverage belongs to the correctness target.
    auto *primary=static_cast<Q8_1Block *>(f.workspace->get()->getBuffer(kCPUProjectionQ8));
    auto *residual=static_cast<Q8_1Block *>(f.workspace->get()->getBuffer(kCPUProjectionResidualQ8));
    const auto banks=cpuBanks(primary,residual,o.policy=="single_q8"?0:o.live,o.k);
    if(o.live && o.policy!="single_q8")for(const int row:{0,o.live/2,o.live-1})for(const int column:{0,o.n/2,o.n-1})
    {
        double expected=0;
        for(int k=0;k<o.k;++k)
        {
            const size_t value=size_t(row)*o.k+k,block=value/32;
            expected+=f.prepared[size_t(column)*o.k+k]*(double(banks.d0[block])*banks.primary[value]+double(banks.d1[block])*banks.residual[value]);
        }
        cpuMeasurementRequire(std::isfinite(f.output.data()[size_t(row)*o.n+column])&&
            std::abs(double(f.output.data()[size_t(row)*o.n+column])-expected)<=roundingBound(f.prepared,column,o.k,banks,row),
            "CPU timing source-derived preflight oracle failed");
    }
    const auto elapsed=[&](GEMMStage &stage){
        const auto start=std::chrono::steady_clock::now();
        for(int iteration=0;iteration<o.iterations;++iteration)execute(stage);
        return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/o.iterations;
    };
#if defined(__AVX512F__)
    constexpr const char *codegen="avx512";
#else
    constexpr const char *codegen="avx2";
#endif
    const nlohmann::json metadata={{"scope","synthetic_original_source_prepared_cpu_component_not_model"},
        {"source_format",o.format},{"source_seed",801u+unsigned(entry->tensor_type)},
        {"codegen_isa",codegen},{"runtime_isa",isaLevelName(activeISALevel())},
        {"threads_requested",o.workers},{"threads_actual",actual_workers},
        {"calling_thread_affinity",cpuMeasurementAffinity()},{"worker_affinities",worker_affinities},
        {"M",o.m},{"N",o.n},{"K",o.k},{"live_rows",o.live},{"phase",o.phase},
        {"workspace_bytes",f.workspace->get()->budget()},
        {"activation_bank_bytes",f.workspace->get()->getBufferSize(kCPUProjectionQ8)+f.workspace->get()->getBufferSize(kCPUProjectionResidualQ8)},
        {"raw_source_bytes",f.raw.size()},{"packed_source_bytes",prepared_before.size()},
        {"source_pointer",reinterpret_cast<uintptr_t>(source_pointer)},
        {"primary_pointer",reinterpret_cast<uintptr_t>(primary)},{"residual_pointer",reinterpret_cast<uintptr_t>(residual)},
        {"full_correctness_gate_required",true}};
    std::cout<<"CPU_PREPARED_SOURCE "<<metadata.dump()<<'\n';
    for(int batch=0;batch<o.batches;++batch)
    {
        double single_us=0,two_us=0;
        if(o.policy=="paired")
        {
            if(batch%2==0){single_us=elapsed(single);two_us=elapsed(*f.stage);}
            else{two_us=elapsed(*f.stage);single_us=elapsed(single);}
        }
        else if(o.policy=="single_q8")single_us=elapsed(single);
        else two_us=elapsed(*f.stage);
        cpuMeasurementRequire((expected_single.empty()||std::memcmp(expected_single.data(),f.baseline.data(),expected_single.size()*sizeof(float))==0)&&
            (expected_two.empty()||std::memcmp(expected_two.data(),f.output.data(),expected_two.size()*sizeof(float))==0),
            "CPU timed replay changed output bytes");
        for(size_t index=size_t(o.live)*o.n;index<f.output.numel();++index)
            cpuMeasurementRequire(f.output.data()[index]==-12345.f&&f.baseline.data()[index]==-12345.f,
                "CPU timed replay changed an inactive output row");
        cpuMeasurementRequire(f.workspace->get()->getBuffer(kCPUProjectionQ8)==primary&&
            f.workspace->get()->getBuffer(kCPUProjectionResidualQ8)==residual,"CPU timing rebound an activation bank");
        cpuMeasurementRequire(packed->interleavedBase()==source_pointer&&
            std::equal(prepared_before.begin(),prepared_before.end(),source_pointer)&&originalBytes(*f.weight)==f.raw,
            "CPU timed replay changed original prepared source");
        std::cout<<"CPU_PREPARED_TIMING "<<nlohmann::json{{"batch",batch},{"policy",o.policy},
            {"order",o.policy=="paired"?(batch%2?"BA":"AB"):"isolated"},{"iterations",o.iterations},
            {"single_us",o.policy=="two_term_q8"?nlohmann::json(nullptr):nlohmann::json(single_us)},
            {"two_term_us",o.policy=="single_q8"?nlohmann::json(nullptr):nlohmann::json(two_us)},
            {"ratio",single_us>0&&two_us>0?nlohmann::json(two_us/single_us):nlohmann::json(nullptr)},
            {"timer","steady_clock_complete_stage_including_openmp_submission"}}.dump()<<'\n';
    }
    // Outside every measured interval: expose the actual production observer,
    // not an inferred submission/replay count. This also proves a disabled arm
    // remained silent. Warmup submissions are explicitly included in records.
    nlohmann::json observations=nlohmann::json::array();
    for(const auto &record:PerfStatsCollector::snapshot({"kernel.hc_up_projection_submissions"}))
        if(record.domain=="kernel" && record.name=="hc_up_projection_submissions")
            observations.push_back({{"count",record.count},{"value",record.value},{"tags",record.tags}});
    std::cout<<"CPU_HC_SUBMISSION_OBSERVATION "<<nlohmann::json{
        {"scope","successful_host_execute_or_capture_not_replay_or_live_rows"},
        {"collection_enabled",PerfStatsCollector::isDomainEnabled("kernel")},
        {"records",std::move(observations)}}.dump()<<'\n';
    return 0;
}
} // namespace llaminar2::test::q8_all_formats
