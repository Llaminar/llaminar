/**
 * @file ROCmGDNPairedNormalizationPerf.h
 * @brief Opt-in retained-graph timing of production and literal historical GDN.
 *
 * This fixture measures only the FP32 recurrence boundary. Q/K/V and projected
 * gates are already prepared; convolution, deinterleave, gate projection and
 * output normalization are outside the graph. Initial state is immutable and
 * distinct from each output bank, so every captured repeat receives identical
 * inputs. M1 publishes terminal state; M16 publishes every verifier snapshot,
 * selecting the same scalar-order production kernel rather than long prefill.
 * No setup, transfer, reset or allocation is captured or event-timed.
 * This separate version2 trial authenticates production128-thread/32-column
 * geometry independently from the unchanged historical256-thread/64-column geometry.
 */
#pragma once
#include <cerrno>
#include <cstdlib>
#include <cxxabi.h>
#include <iostream>
#include <limits>

/** @namespace gdn_paired_perf
 * @brief Test-only owners and observers for one immutable captured recurrence workload.
 */
namespace gdn_paired_perf
{
/**
 * @brief Reject an incomplete admission or observation before accepting timing evidence.
 * @param condition Predicate evaluated by the owning fixture on the host.
 * @param message Borrowed non-null diagnostic copied into the exception on failure.
 * @throws std::runtime_error When condition is false; no alternate path is selected.
 * @details This helper neither owns native resources nor changes stream/device state.
 */
inline void require(bool condition,const char* message)
{ if(!condition)throw std::runtime_error(message); }

/**
 * @brief Read one bounded environment scalar before allocating or capturing GPU work.
 * @param name Borrowed non-null name of the test-only environment variable.
 * @param fallback Value returned when the variable is absent; the caller supplies an admitted default.
 * @param lower Inclusive minimum for an explicitly provided value.
 * @param upper Inclusive maximum for an explicitly provided value.
 * @return The complete base-10 value, or fallback if the variable is absent.
 * @throws std::runtime_error On an empty value, trailing characters, conversion
 * range error or value outside the inclusive bounds. Plan applies its closed
 * geometry constraints after this initial range check.
 * @details The returned scalar is copied into the immutable plan. No borrowed
 * environment pointer survives this function and no native operation occurs.
 */
inline int integer(const char* name,int fallback,int lower,int upper)
{
    const char* value=std::getenv(name);
    if(!value)return fallback;
    char* end=nullptr;errno=0;
    const long parsed=std::strtol(value,&end,10);
    require(errno==0 && end!=value && *end=='\0' && parsed>=lower && parsed<=upper,
        "Invalid GDN paired-performance integer environment value");
    return int(parsed);
}

/**
 * @brief Preserve an actual native symbol inside one comma-delimited marker field.
 * @param input Borrowed complete symbol name; its ownership remains with the caller.
 * @return An owned string with comma, percent and ASCII control bytes encoded
 * as uppercase percent-hex sequences; all other bytes retain their exact order.
 * @details Encoding performs host string work during capture inspection, outside
 * every measured graph interval. It never abbreviates the symbol or changes its identity.
 */
inline std::string escaped(const std::string& input)
{
    constexpr char hex[]="0123456789ABCDEF";
    std::string result;
    for(unsigned char c:input)
        if(c==',' || c=='%' || c<' ' || c==127){result+='%';result+=hex[c>>4];result+=hex[c&15];}
        else result+=char(c);
    return result;
}

/**
 * @brief Own the admitted geometry and sampling policy for one test invocation.
 * @details Construction reads only test environment values. run() then retains
 * this object as const through all allocations, captures, observations and joins.
 * Requests are fixed at one. This sampling policy does not alter production
 * dispatch: complete M16 snapshots deliberately select the scalar-order verifier
 * recurrence, while M1 owns only terminal state publication.
 */
struct Plan final
{
    int dk=integer("LLAMINAR_GDN_PERF_DK",128,64,128); ///< Immutable key width selecting the64/128 native specialization.
    int dv=integer("LLAMINAR_GDN_PERF_DV",128,64,128); ///< Immutable value width64/128, independent of key width.
    int heads=integer("LLAMINAR_GDN_PERF_HEADS",12,1,256); ///< Local value-head count;12 is the focused ordinary TP4 shape.
    int rows=integer("LLAMINAR_GDN_PERF_M",1,1,16); ///< Physical request rows; closed admission is M1 orM16.
    int repeats=integer("LLAMINAR_GDN_PERF_REPEATS",64,1,256); ///< Complete recurrence nodes captured in one retained graph.
    std::string side=std::getenv("LLAMINAR_GDN_PERF_SIDE")?std::getenv("LLAMINAR_GDN_PERF_SIDE"):"paired"; ///< Owned choice of paired events or one isolated native family.
    /**
     * @brief Admit the closed lane after the member initializers copy environment values.
     * @throws std::runtime_error If key/value width is not64/128, rows are not1/16,
     * side is unknown, or isolated profiling requests more than one captured node.
     * @details Profiling requires the caller to set repeats explicitly to one;
     * invalid geometry fails before any stream, event, buffer or graph exists.
     */
    Plan()
    {
        require((dk==64 || dk==128) && (dv==64 || dv==128) && (rows==1 || rows==16),
            "GDN performance supports DK/DV64|128 and M1|16");
        require(side=="paired" || side=="production" || side=="historical","Unknown GDN performance side");
        require(side=="paired" || repeats==1,"Isolated GDN profiling requires explicit REPEATS=1");
    }
    /**
     * @brief Identify which already-admitted state publication contract this lane uses.
     * @return True for M16 complete row snapshots; false for M1 terminal updated state.
     * @details This is an immutable capture choice, not a read of live device state.
     */
    bool snapshots()const{return rows==16;}
    /**
     * @brief Compute the exact FP32 state element count for the single request.
     * @return heads*dk*dv elements, at most256*128*128 and therefore representable in int.
     * @details This is the per-row snapshot stride and terminal-bank capacity;
     * the row-snapshot allocation separately multiplies it by the physical row count.
     */
    int stateWidth()const{return heads*dk*dv;}
    /**
     * @brief Select the compiled trial family's exact producer block size.
     * @param historical True for unchanged historical geometry, false for the128-thread trial.
     * @return256 historical threads or128 production threads; no production runtime mode is selected.
     * @details The native graph independently proves this source-owned fixture expectation.
     */
    int threads(bool historical)const{return historical?256:128;}
    /**
     * @brief Derive each family's independent value-column ownership from four K-row splits.
     * @param historical True for the literal historical family, false for production.
     * @return64 historical columns or32 production columns per CTA.
     */
    int columns(bool historical)const{return threads(historical)/4;}
    /**
     * @brief Compute one family's exact source-owned dynamic shared-memory extent.
     * @param historical True for historical256-thread storage, false for production128-thread storage.
     * @return At DK128,13328 historical bytes or7184 production bytes; at DK64,2576 or1552 bytes.
     * @details Q/K and four norm-wave words retain their extent; only the two
     * reduction banks and explicit state tail scale with this family's thread count.
     * This query allocates nothing and does not change production launch policy.
     */
    std::size_t sharedBytes(bool historical)const{return std::size_t(2*dk+4+2*threads(historical)+std::max(0,dk/4-22)*threads(historical))*sizeof(float);}
    /**
     * @brief Append the immutable selector prefix to the host observation stream.
     * @param kind Borrowed non-null marker kind supplied by the owning observation.
     * @details Writes no newline or side-specific fields; the caller appends those.
     * No GPU state is observed or modified, and callers emit markers outside
     * timed event intervals. Event callers subsequently append us-per-recurrence values.
     */
    void marker(const char* kind)const
    {
        std::cout<<kind<<",version=2,requests=1,M="<<rows<<",DK="<<dk<<",DV="<<dv<<",heads="<<heads
            <<",ownership="<<(snapshots()?"grouped_snapshots":"scalar_terminal")<<",repeats="<<repeats;
    }
};

/**
 * @brief Authenticate the retained graph's actual native identity, resources and linear DAG.
 * @param graph Borrowed fixture-owned, already-instantiated graph whose buffers remain alive.
 * @param stream Exact non-default capture/replay stream, retained by the outer fixture.
 * @param plan Borrowed immutable plan defining node count, specialization and launch geometry.
 * @param historical True for the literal historical namespace; false for the production namespace.
 * @throws std::runtime_error On a HIP query error, missing/extra/foreign node,
 * function mismatch, incorrect geometry, incomplete/branching/cyclic dependency
 * chain, unavailable symbol or invalid register/private-memory observation.
 * @details Host queries and temporary inspection vectors occur before warmup
 * and timing. This observer does not launch or mutate the graph, synchronize
 * it, or take ownership of any native handle. It prints one complete capture
 * marker only after all nodes pass. Every node must name the same exact family
 * specialization. The dynamic shared-memory extent is checked independently
 * from the returned native static-shared attribute. Native resource observation
 * does not replace the mandatory compiler allocation/spill certificate.
 */
inline void inspect(const HipCapturedGraph& graph,hipStream_t stream,const Plan& plan,bool historical)
{
    std::size_t count=0;checkHip(hipGraphGetNodes(graph.graph,nullptr,&count),"GDN graph inventory");
    require(count==std::size_t(plan.repeats),"GDN graph must contain precisely one recurrence per repeat");
    std::vector<hipGraphNode_t> nodes(count);checkHip(hipGraphGetNodes(graph.graph,nodes.data(),&count),"GDN nodes");
    std::vector<int> next(count,-1);std::size_t root=count;void* function=nullptr;std::string name;
    for(std::size_t i=0;i<count;++i)
    {
        hipGraphNodeType type{};checkHip(hipGraphNodeGetType(nodes[i],&type),"GDN node type");
        require(type==hipGraphNodeTypeKernel,"GDN graph contains work outside recurrence");
        hipKernelNodeParams native{};checkHip(hipGraphKernelNodeGetParams(nodes[i],&native),"GDN native parameters");
        require(native.func && native.gridDim.x==unsigned(plan.heads) && native.gridDim.y==unsigned((plan.dv+plan.columns(historical)-1)/plan.columns(historical))
            && native.gridDim.z==1 && native.blockDim.x==unsigned(plan.threads(historical)) && native.blockDim.y==1 && native.blockDim.z==1
            && native.sharedMemBytes==plan.sharedBytes(historical),"GDN native geometry differs from its immutable plan");
        const char* raw=hipKernelNameRefByPtr(native.func,stream);require(raw,"GDN kernel name unavailable");
        int status=-1;std::unique_ptr<char,decltype(&std::free)> demangled(abi::__cxa_demangle(raw,nullptr,nullptr,&status),&std::free);
        require(status==0 && bool(demangled),"GDN native name cannot be demangled");
        const std::string actual(demangled.get());
        const auto prefix=std::string("void ")+(historical?"gdn_historical_native::":"(anonymous namespace)::")
            +"rocm_gdn_recurrent_step_kernel<"+std::to_string(plan.dk)+">(";
        require(actual.rfind(prefix,0)==0,"GDN graph did not capture the requested historical/production specialization");
        if(!function){function=native.func;name=actual;}
        require(function==native.func && name==actual,"GDN native identity changes between captured repeats");
        std::size_t parents=0;checkHip(hipGraphNodeGetDependencies(nodes[i],nullptr,&parents),"GDN dependency count");
        require(parents<=1,"GDN graph has a multi-parent node");
        if(!parents){require(root==count,"GDN graph has multiple roots");root=i;}
        else
        {
            hipGraphNode_t parent=nullptr;checkHip(hipGraphNodeGetDependencies(nodes[i],&parent,&parents),"GDN parent");
            const auto found=std::find(nodes.begin(),nodes.end(),parent);require(found!=nodes.end(),"GDN graph has a foreign parent");
            const auto j=std::size_t(found-nodes.begin());require(next[j]==-1,"GDN graph branches");next[j]=int(i);
        }
    }
    require(root<count,"GDN graph has no unique root");
    std::vector<bool> visited(count,false);int cursor=int(root);
    for(std::size_t i=0;i<count;++i){require(cursor>=0 && !visited[std::size_t(cursor)],"GDN graph is incomplete or cyclic");
        visited[std::size_t(cursor)]=true;cursor=next[std::size_t(cursor)];}
    require(cursor==-1,"GDN graph has excess nodes");
    hipFuncAttributes attributes{};checkHip(hipFuncGetAttributes(&attributes,function),"GDN function attributes");
    require(attributes.localSizeBytes==0 && attributes.numRegs>0,"GDN native private/register observation invalid");
    plan.marker("GDN_QK_CAPTURE");std::cout<<",side="<<(historical?"historical":"production")
        <<",function_encoding=percent-comma,function="<<escaped(name)<<",native_function="<<function
        <<",grid="<<plan.heads<<':'<<(plan.dv+plan.columns(historical)-1)/plan.columns(historical)<<":1,block="<<plan.threads(historical)<<":1:1,dynamic_shared_bytes="<<plan.sharedBytes(historical)
        <<",native_attribute_num_regs="<<attributes.numRegs<<",native_attribute_shared_bytes="<<attributes.sharedSizeBytes
        <<",native_attribute_local_bytes="<<attributes.localSizeBytes<<",ordered_chain_nodes="<<count<<'\n';
}

/**
 * @brief Exclusively own the reusable native event pair for host-driven diagnostic timing.
 * @details run() constructs this owner before recording graphs or submitting work.
 * Its later-declared retirement guard joins the stream before this owner unwinds.
 * No event allocation or destruction belongs to a captured or timed graph.
 */
struct Events final
{
    hipEvent_t begin=nullptr,end=nullptr; ///< Exclusively owned start/end event handles.
    /**
     * @brief Allocate both timing events before any measured submission.
     * @throws std::runtime_error On HIP allocation failure; if only the first
     * allocation succeeded, that handle is destroyed before propagating the error.
     */
    Events(){checkHip(hipEventCreate(&begin),"GDN begin event");
        const auto status=hipEventCreate(&end);if(status!=hipSuccess){(void)hipEventDestroy(begin);checkHip(status,"GDN end event");}}
    /**
     * @brief Release both native handles after the owning fixture's retirement join.
     * @details Does not throw; destroy return codes are intentionally ignored
     * during teardown. Measurement failures already propagate through checkHip.
     */
    ~Events(){if(end)(void)hipEventDestroy(end);if(begin)(void)hipEventDestroy(begin);}
    /**
     * @brief Time one retained graph submission and report one complete recurrence's cost.
     * @param graph Borrowed graph already authenticated by inspect(); all its native
     * buffers and immutable input values remain alive and unchanged through the join.
     * @param stream Exact non-default stream used to record events and replay this graph.
     * @param repeats Positive immutable node count already proven against the graph by inspect().
     * @return Elapsed event milliseconds converted to microseconds, divided once by repeats.
     * @throws std::runtime_error On any HIP record/launch/join/elapsed-time failure,
     * or on a nonfinite/nonpositive elapsed duration. Outer retirement still owns cleanup.
     * @details Records begin, submits exactly one retained graph, then records and
     * joins end. No setup, transfers, resets or allocations are inserted between
     * the events. This explicit host join is diagnostic fixture work, not a
     * production hot-path synchronization. The event pair is reused sequentially.
     */
    double measure(const HipCapturedGraph& graph,hipStream_t stream,int repeats)
    {
        checkHip(hipEventRecord(begin,stream),"GDN begin");graph.launch(stream);
        checkHip(hipEventRecord(end,stream),"GDN end");checkHip(hipEventSynchronize(end),"GDN measurement join");
        float milliseconds=0;checkHip(hipEventElapsedTime(&milliseconds,begin,end),"GDN event elapsed");
        require(std::isfinite(milliseconds) && milliseconds>0,"GDN event duration is invalid");
        return double(milliseconds)*1000.0/repeats;
    }
};

/**
 * @brief Own and observe one admitted recurrence workload without advancing request state.
 * @throws std::runtime_error On plan admission, HIP setup/capture/inspection,
 * byte-sanity or timing failure. A terminal stream-retirement failure terminates
 * the process rather than releasing storage still referenced by device work.
 * @details Reads Plan's environment controls before native setup. The fixture owns
 * one non-default stream, persistent input and independent per-family output/state
 * banks, optional complete snapshot banks, events and retained graphs. The initial
 * state never aliases either updated-state bank, so every captured repeat receives
 * identical inputs rather than advancing a mutable recurrent request. All host
 * upload spans and native owners outlive the final retirement guard.
 *
 * Paired mode captures only complete recurrence nodes, validates output and all
 * publication bytes before timing, warms each family five times, records12 paired
 * samples in alternating order, then repeats the complete byte check. M16's full
 * snapshot matrix selects the scalar-order verifier route and leaves terminal
 * state poison intact; M1 publishes terminal state with no snapshots. Setup,
 * transfers, byte downloads, inspection and joins remain outside event intervals.
 *
 * Isolated mode captures only the requested family with one node, warms five
 * graph launches, flushes its armed marker, then submits launch six without
 * event timing. Its byte proof is supplied by the separate paired invocation
 * and registered historical oracle. This component boundary excludes upstream
 * projections/convolution/deinterleave and downstream normalization or model work.
 */
inline void run()
{
    const Plan plan;
    const std::size_t qsize=std::size_t(plan.rows)*plan.heads*plan.dk;
    const std::size_t vsize=std::size_t(plan.rows)*plan.heads*plan.dv;
    const std::size_t gates=std::size_t(plan.rows)*plan.heads,state=plan.stateWidth();
    std::vector<float> q(qsize),k(qsize),v(vsize),alpha(gates),beta(gates),initial(state);
    for(std::size_t i=0;i<qsize;++i){q[i]=float(int(i%29)-14)*0.03125f;k[i]=float(int(i%23)-11)*0.0625f;}
    for(std::size_t i=0;i<vsize;++i)v[i]=float(int(i%31)-15)*0.0625f;
    for(std::size_t i=0;i<gates;++i){alpha[i]=float(int(i%7)-3)*0.125f;beta[i]=float(int(i%9)-4)*0.25f;}
    for(std::size_t i=0;i<state;++i)initial[i]=float(int(i%17)-8)*0.00073f;
    const std::vector<float> log(plan.heads,-0.375f),bias(plan.heads,0.125f);
    HipStreamHandle stream;
    HipFloatBuffer dq(q),dk(k),dv(v),da(alpha),db(beta),dl(log),dt(bias),di(initial);
    HipFloatBuffer po(vsize,-77.f),ho(vsize,-77.f),ps(state,-91.f),hs(state,-91.f);
    HipFloatBuffer pp(plan.snapshots()?std::size_t(plan.rows)*state:0,-83.f),hp(pp.count,-83.f);
    HipIntBuffer live(plan.rows);
    Events events;
    std::unique_ptr<HipCapturedGraph> production,historical;
    /**
     * @brief Retire asynchronous references before earlier-declared native owners unwind.
     * @details This guard borrows the stream, owns no storage, and is declared
     * after all graph pointers, events, buffers and host upload spans.
     */
    struct Retirement final{
        hipStream_t stream; ///< Borrowed exact non-default stream, owned by the outer fixture.
        /**
         * @brief Join the stream before graph/event/buffer destruction on success or failure.
         * @details A failed join terminates; the fixture never frees potentially
         * live resources and then presents the error as a clean retirement.
         */
        ~Retirement(){if(hipStreamSynchronize(stream)!=hipSuccess)std::terminate();}} retirement{stream.stream};
    if(plan.side!="historical")
        production=std::make_unique<HipCapturedGraph>(stream.stream,[&]{
            for(int r=0;r<plan.repeats;++r)
                if(!rocmGDN_chunk_forward_batched_effective(dq.ptr,dk.ptr,dv.ptr,da.ptr,db.ptr,dl.ptr,dt.ptr,po.ptr,di.ptr,ps.ptr,
                    plan.rows,1,plan.rows,plan.heads,plan.dk,plan.dv,true,live.ptr,pp.ptr,int(state),plan.snapshots()?plan.rows:0,0,stream.stream))return false;
            return true;
        });
    if(plan.side!="production")
        historical=std::make_unique<HipCapturedGraph>(stream.stream,[&]{
            const gdn_historical_native::Arguments args{.q=dq.ptr,.k=dk.ptr,.v=dv.ptr,.alpha=da.ptr,.beta=db.ptr,.a_log=dl.ptr,.dt_bias=dt.ptr,
                .output=ho.ptr,.initial=di.ptr,.updated=hs.ptr,.requests=1,.rows=plan.rows,.heads=plan.heads,.key_width=plan.dk,.value_width=plan.dv,
                .normalize=true,.lengths=live.ptr,.row_offset=0,.snapshots=hp.ptr,.snapshot_stride=int(state),.snapshot_rows=plan.snapshots()?plan.rows:0,.stream=stream.stream};
            for(int r=0;r<plan.repeats;++r)if(!gdn_historical_native::launch(args))return false;
            return true;
        });
    if(production)inspect(*production,stream.stream,plan,false);
    if(historical)inspect(*historical,stream.stream,plan,true);
    if(plan.side!="paired")
    {
        const auto& graph=production?*production:*historical;
        for(int i=0;i<5;++i)graph.launch(stream.stream);
        checkHip(hipStreamSynchronize(stream.stream),"GDN isolated warmup join");
        plan.marker("GDN_QK_PROFILE_ARMED");std::cout<<",side="<<plan.side<<",warmups=5,measured_occurrence=6,byte_sanity=separate_paired_gate"<<std::endl;
        graph.launch(stream.stream);checkHip(hipStreamSynchronize(stream.stream),"GDN isolated sixth-launch join");
        plan.marker("GDN_QK_PROFILE_DONE");std::cout<<",side="<<plan.side<<",graph_launches=6,kernel_dispatches=6\n";
        return;
    }
    /**
     * @brief Replay both families and compare every authored publication byte outside timing.
     * @param phase Borrowed marker label distinguishing the before and after boundaries.
     * @throws std::runtime_error On a launch/join, download, byte mismatch or terminal-poison failure.
     * @details Borrows all enclosing owners. Each retained graph sees the same
     * immutable initial state; downloads occur only after their shared stream joins.
     */
    const auto sanity=[&](const char* phase){
        production->launch(stream.stream);historical->launch(stream.stream);
        checkHip(hipStreamSynchronize(stream.stream),"GDN paired sanity join");
        /**
         * @brief Compare complete physical FP32 bank bytes without numeric tolerance.
         * @param a Borrowed production bank, retired before the download.
         * @param b Borrowed historical bank of the same expected element count.
         * @throws std::runtime_error If sizes or bytes differ, or a download fails.
         * @details Empty optional snapshot banks compare without dereferencing data.
         */
        const auto compare=[](const HipFloatBuffer& a,const HipFloatBuffer& b){
            const auto x=a.toHost(),y=b.toHost();require(x.size()==y.size(),"GDN bank size differs");
            require(x.empty() || std::memcmp(x.data(),y.data(),x.size()*sizeof(float))==0,"GDN historical byte sanity failed");};
        compare(po,ho);compare(ps,hs);compare(pp,hp);
        if(plan.snapshots())for(float word:ps.toHost())require(word==-91.f,"GDN snapshot lane unexpectedly publishes terminal bank");
        plan.marker("GDN_QK_SANITY");std::cout<<",phase="<<phase<<",output_bytes="<<po.count*sizeof(float)
            <<",terminal_bytes="<<ps.count*sizeof(float)<<",snapshot_bytes="<<pp.count*sizeof(float)<<",byte_equal=1\n";
    };
    sanity("before");
    for(int i=0;i<5;++i){production->launch(stream.stream);historical->launch(stream.stream);}
    checkHip(hipStreamSynchronize(stream.stream),"GDN paired warmup join");
    for(int sample=0;sample<12;++sample)
    {
        double p=0,h=0;
        if(sample%2==0){p=events.measure(*production,stream.stream,plan.repeats);h=events.measure(*historical,stream.stream,plan.repeats);}
        else{h=events.measure(*historical,stream.stream,plan.repeats);p=events.measure(*production,stream.stream,plan.repeats);}
        plan.marker("GDN_QK_EVENT");std::cout<<std::setprecision(12)<<",sample="<<sample<<",order="<<(sample%2?"historical-production":"production-historical")
            <<",production_us="<<p<<",historical_us="<<h<<",unit=us_per_recurrence\n";
    }
    sanity("after");
}
} // namespace gdn_paired_perf

/**
 * @test Disabled opt-in retained-graph events or one-family sixth-launch profiling.
 * @details Explicit invocation requires a real ROCm device and selects device0.
 * It delegates immutable geometry, input/output ownership, event boundaries and
 * failure retirement to run(). It is not an automatic preflight registration
 * or evidence for whole-GDN/model economics; root owns the isolated process,
 * loader, driver window and bounded external retirement protocol.
 */
TEST(Test__ROCmGDNPaddedRealLength, DISABLED_PairedQKNormalizationEvents)
{
    ASSERT_TRUE(hasROCm())<<"Explicit GDN performance run requires a ROCm device";
    checkHip(hipSetDevice(0),"GDN paired performance device");
    gdn_paired_perf::run();
}
