/**
 * @file QwenSparseObservationPerfTest.h
 * @brief Captured fixed-attention observation ON/OFF events and exact profiling.
 *
 * This include follows QwenSparseDeviceTest.h in either native backend translation unit, after its metadata adapter.
 * Both policies borrow identical compact FP16 Q/K/V and authored selected IDs.
 * Captured function, geometry and dependency checks authenticate the production
 * producer/merge pair. Repeated complete transactions amortize host submission;
 * profiler launches retain one transaction and name its sixth warmed occurrence.
 * Allocation, state publication, events and terminal collection are fixture-only.
 */
#pragma once
#include <string>
#include <iomanip>
#include <fstream>
#include <filesystem>
namespace llaminar2::qwen4exp::QSA_SUITE
{
namespace observation115
{
/**
 * @brief Reject invalid fixture state before recording or publishing evidence.
 * @param ok Whether the required source, geometry or lifecycle invariant holds.
 * @param message Stable diagnostic identifying the failed invariant.
 * @throws std::runtime_error When @p ok is false; no alternate path is selected.
 */
inline void require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}

/** @brief Join a borrowed fixture stream before retiring its retained native graph. */
struct Capture final
{
    QSA_STREAM stream;const int repeats;QSA_GRAPH graph=nullptr;QSA_EXEC exec=nullptr;
    /**
     * @brief Borrow the fixture stream and admit a fixed retained transaction count.
     * @param s Non-default fixture stream; its owner must outlive this capture.
     * @param transactions Number of complete producer/merge pairs, from 1 to 256.
     * @throws std::runtime_error If the transaction count is outside the admitted range.
     * @details Construction creates no graph and submits no device work. The graph
     * and executable become exclusively owned after record() succeeds.
     */
    explicit Capture(QSA_STREAM s,int transactions=1):stream(s),repeats(transactions){
        require(repeats>0&&repeats<=256,"Captured attention repeats must be in1..256");}
    /** @brief Prevent copying ownership of the retained graph and executable. */
    Capture(const Capture&)=delete;
    /**
     * @brief Join terminal fixture work before destroying graph-owned handles.
     * @details This intentional host join is outside captured inference and timed
     * event intervals. A failed join terminates rather than releasing live storage.
     * The borrowed stream remains owned by Fixture and is never destroyed here.
     */
    ~Capture(){if(QSA_RT(StreamSynchronize)(stream)!=QSA_RT(Success))std::terminate();
        if(exec)(void)QSA_RT(GraphExecDestroy)(exec);if(graph)(void)QSA_RT(GraphDestroy)(graph);}
    /**
     * @brief Record and authenticate one retained chain of complete attention pairs.
     * @tparam F Callable that enqueues exactly one production producer/merge pair.
     * @param launch Enqueue callable; it must use this capture's borrowed stream.
     * @param producer Unique producer symbol fragment expected in native metadata.
     * @param merge Unique merge symbol fragment expected in native metadata.
     * @param producerGrid Expected producer CTA grid; a zero X disables this check.
     * @param mergeGrid Expected merge CTA grid; a zero X disables this check.
     * @param producerThreads Expected producer X dimension; Y and Z must equal one.
     * @param mergeThreads Expected merge X dimension; Y and Z must equal one.
     * @param observation ON/OFF metadata label; it does not affect captured operations.
     * @throws std::runtime_error For repeated recording, native API failure, foreign
     * kernels, changed functions/geometry, or a broken producer/merge dependency chain.
     * @details Every admitted repeat retains the same two function pointers and one
     * ordered graph chain. Failed recording ends capture and retires its abandoned
     * graph before rethrowing. All storage must already be bound and remain live
     * through the terminal join in this object's destructor.
     */
    template<class F>void record(F launch,const char* producer,const char* merge,
        dim3 producerGrid=dim3(0),dim3 mergeGrid=dim3(0),int producerThreads=256,int mergeThreads=256,const char* observation="unspecified"){
        require(!graph&&!exec,"Observation fixture capture may only be recorded once");
        check(QSA_RT(StreamBeginCapture)(stream,QSA_RT(StreamCaptureModeThreadLocal)));
        try{for(int repeat=0;repeat<repeats;++repeat)launch();}catch(...){QSA_GRAPH abandoned=nullptr;(void)QSA_RT(StreamEndCapture)(stream,&abandoned);
            if(abandoned)(void)QSA_RT(GraphDestroy)(abandoned);throw;}
        check(QSA_RT(StreamEndCapture)(stream,&graph));
        std::size_t n=0;check(QSA_RT(GraphGetNodes)(graph,nullptr,&n));
        require(n==std::size_t(2*repeats),"Attention graph must contain exactly two kernels per repeat");
        std::vector<ObservationNode> nodes(n);check(QSA_RT(GraphGetNodes)(graph,nodes.data(),&n));
        std::vector<int> roles(n),next(n,-1);std::size_t head=n;int producers=0,merges=0;
        void* producerFunction=nullptr;void* mergeFunction=nullptr;
        for(std::size_t index=0;index<n;++index){const auto node=nodes[index];
            ObservationNodeType kind;check(QSA_RT(GraphNodeGetType)(node,&kind));
            require(kind==QSA_RT(GraphNodeTypeKernel),"Attention-only capture contains a non-kernel node");
            ObservationKernelParams p{};check(QSA_RT(GraphKernelNodeGetParams)(node,&p));
            const auto metadata=inspectObservationKernel(nodes[index],stream);
            require(metadata.identity && metadata.registers>0 && metadata.local_bytes==0 && metadata.max_threads>=256 && metadata.resident_block_limit>0,
                "Captured native function has invalid resources or private storage");
            const std::string& name=metadata.name;const bool isProducer=name.find(producer)!=std::string::npos;
            const bool isMerge=name.find(merge)!=std::string::npos;
            require(isProducer!=isMerge,"Captured kernel differs from exact attention pair");
            roles[index]=isProducer?0:1;
            const auto expectedGrid=isProducer?producerGrid:mergeGrid;
            if(expectedGrid.x)require(p.gridDim.x==expectedGrid.x&&p.gridDim.y==expectedGrid.y&&p.gridDim.z==expectedGrid.z&&
                p.blockDim.x==unsigned(isProducer?producerThreads:mergeThreads)&&p.blockDim.y==1&&p.blockDim.z==1,"Captured schedule function/launch geometry differs");
            void*& expected=isProducer?producerFunction:mergeFunction;
            if(!expected){expected=metadata.identity;
                std::cout<<"OBSERVATION115_NATIVE,{\"backend\":"<<std::quoted(ObservationBackend)
                    <<",\"observation\":"<<std::quoted(observation)<<",\"role\":"<<std::quoted(isProducer?"producer":"merge")
                    <<",\"symbol\":"<<std::quoted(metadata.name)<<",\"identity\":"<<reinterpret_cast<std::uintptr_t>(metadata.identity)
                    <<",\"registers\":"<<metadata.registers<<",\"shared_bytes\":"<<metadata.shared_bytes
                    <<",\"local_bytes\":"<<metadata.local_bytes<<",\"max_threads\":"<<metadata.max_threads
                    <<",\"resident_block_limit\":"<<metadata.resident_block_limit<<",\"occupancy_basis\":\"theoretical_resource_limit\",\"grid\":["<<p.gridDim.x<<','<<p.gridDim.y<<','<<p.gridDim.z
                    <<"],\"block\":["<<p.blockDim.x<<','<<p.blockDim.y<<','<<p.blockDim.z<<"]}\n";
            }
            require(expected==metadata.identity,"Function changes between captured repeats");
            producers+=isProducer;merges+=isMerge;
            std::size_t dependencies=0;check(observationDependencies(node,nullptr,&dependencies));
            require(dependencies<=1,"Attention repeats require a single ordered native dependency chain");
            if(!dependencies){require(head==n,"Attention capture has multiple independent roots");head=index;}
            else{ObservationNode parent=nullptr;check(observationDependencies(node,&parent,&dependencies));
                const auto it=std::find(nodes.begin(),nodes.end(),parent);
                require(it!=nodes.end(),"Captured dependency names a foreign node");
                const auto previous=std::size_t(it-nodes.begin());
                require(next[previous]==-1,"Captured pair chain branches");next[previous]=int(index);}}
        require(producers==repeats&&merges==repeats&&head<n&&producerFunction!=mergeFunction,"Attention pair inventory/root/function mismatch");
        std::vector<bool> visited(n,false);int cursor=int(head);
        for(std::size_t step=0;step<n;++step){require(cursor>=0&&!visited[std::size_t(cursor)],"Captured pair chain is incomplete/cyclic");
            const auto index=std::size_t(cursor);visited[index]=true;
            require(roles[index]==int(step%2),"Captured producer/merge pairs are out of order");cursor=next[index];}
        require(cursor==-1,"Captured pair chain has excess nodes");
        check(QSA_RT(GraphInstantiate)(&exec,graph,nullptr,nullptr,0));
    }
    /**
     * @brief Submit the retained graph on its exact borrowed stream without joining.
     * @throws std::runtime_error If recording has not completed or submission fails.
     * @details The event harness owns terminal completion; this method performs no
     * allocation, state publication, graph mutation or blocking synchronization.
     */
    void launch(){require(exec,"Unrecorded attention-only graph");check(QSA_RT(GraphLaunch)(exec,stream));}
};

/** @brief Exact compact original-head inputs; no native conversion or full-cache expansion. */
struct Inputs final
{
    Fixture& f;int heads,first,kv_heads,first_kv;
    std::vector<float> host_query;
    std::vector<std::uint16_t> host_keys,host_values;
    Buffer<float> query;Buffer<std::uint16_t> keys,values;
    FixtureCopyRetirement retirement;
    /**
     * @brief Materialize an exact original-head compact native view before graph recording.
     * @param fixture Persistent FP16 source fixture supplying stream, rows and capacity.
     * @param firstHead Original query-head origin within the global 24-head interval.
     * @param headCount Positive local query-head count, at most 24.
     * @param firstKV Original KV-head origin within the global two-head bank.
     * @param kvCount One or two local KV heads covering every declared query consumer.
     * @throws std::runtime_error For unsupported compact ownership or native API failure.
     * @details Query rows keep their original FP32 bytes; selected FP16 K/V head bytes
     * are copied without conversion. The terminal setup join completes uploads before
     * capture. Copy retirement also joins outstanding uploads on constructor failure
     * before their host/device backing unwinds. Fixture must outlive these inputs,
     * which must outlive every capture borrowing base().
     */
    Inputs(Fixture& fixture,int firstHead,int headCount,int firstKV,int kvCount)
        :f(fixture),heads(headCount),first(firstHead),kv_heads(kvCount),first_kv(firstKV),
         host_query(std::size_t(f.rows)*heads*256),host_keys(std::size_t(f.capacity)*kv_heads*256),
         host_values(host_keys.size()),query(host_query.size()),keys(host_keys.size()),values(host_values.size()),
         retirement{&f.stream}
    {
        require(heads>0&&heads<=24&&kv_heads>0&&kv_heads<=2&&first>=0&&first<=24-heads&&
            first_kv>=0&&first_kv<=2-kv_heads&&first/12>=first_kv&&(first+heads-1)/12<first_kv+kv_heads,
            "Native fixture requires exact original compact GQA ownership");
        for(int row=0;row<f.rows;++row)std::copy_n(f.host_core_query.data()+std::size_t(row)*6144+first*256,
            heads*256,host_query.data()+std::size_t(row)*heads*256);
        for(int token=0;token<f.capacity;++token)for(int head=0;head<kv_heads;++head){
            const auto from=(std::size_t(token)*2+first_kv+head)*256,to=(std::size_t(token)*kv_heads+head)*256;
            std::copy_n(f.key16.data()+from,256,host_keys.data()+to);
            std::copy_n(f.value16.data()+from,256,host_values.data()+to);}
        query.upload(host_query.data(),f.stream);keys.upload(host_keys.data(),f.stream);values.upload(host_values.data(),f.stream);
        check(QSA_RT(StreamSynchronize)(f.stream));
    }
    /**
     * @brief Bind compact native input pointers to the original fixed-attention policy.
     * @return Borrowed arguments with the declared compact query/KV heads, original
     * replay/selection buffers, 512-byte head strides and no sigmoid gate.
     * @details No data moves here. Inputs and its source Fixture must remain live
     * until each borrowing FixedRun has completed its terminal capture retirement.
     * Output/partial ownership is subsequently bound by each FixedRun constructor.
     */
    QwenSparseAttention base()const{
        auto a=f.attention;a.query=query.data;a.gate=nullptr;a.query_heads=heads;a.query_head_start=first;
        a.kv_heads=kv_heads;a.kv_head_start=first_kv;
        a.kv.keys={.data=keys.data,.token_stride_bytes=std::size_t(kv_heads)*512,.head_stride_bytes=512,.codec=TensorType::FP16};
        a.kv.values=a.kv.keys;a.kv.values.data=values.data;return a;
    }
};

/**
 * @brief Exclusive fixed-attention outputs and partials over shared immutable inputs.
 * @details Test-only schedule adapters may inherit these storage owners on the
 * stack. Ownership is never transferred or deleted through a base pointer.
 */
struct FixedRun
{
    std::vector<float> poison;std::vector<QwenSparseAttentionPartial> initial_partials;Buffer<float> output;Buffer<QwenSparseAttentionPartial> partials;
    Buffer<QwenSparseAttentionObservation> observations;QwenSparseAttention args;Capture capture;
    /**
     * @brief Bind exclusive output, eight-partition scratch and optional observations.
     * @param a Immutable compact production inputs; their owners must outlive this run.
     * @param s Exact fixture stream used for setup, recording and replay.
     * @param observed Whether the captured arguments publish logical work observations.
     * @param repeats Number of complete retained producer/merge pairs, from 1 to 256.
     * @throws std::runtime_error For invalid repeats or native allocation/copy failure.
     * @details Each row/head owns 256 FP32 outputs and eight exclusive partials; a
     * trailing 16-float output guard retains the existing poison bytes. The setup
     * upload joins before capture and timing. Capture is declared last, so its
     * terminal destructor join precedes destruction of every borrowed output bank.
     */
    FixedRun(QwenSparseAttention a,QSA_STREAM s,bool observed=true,int repeats=1):poison(std::size_t(a.row_capacity)*a.query_heads*256+16,-817.25f),
        initial_partials(std::size_t(a.row_capacity)*a.query_heads*8),output(poison.size()),partials(std::size_t(a.row_capacity)*a.query_heads*8),observations(std::size_t(a.row_capacity)*a.query_heads),args(a),capture(s,repeats){
        args.output=output.data;args.gpu_partials=partials.data;args.observations=observations.data;
        if(!observed)args.observations=nullptr;
        // Equal dirty payload makes empty-partition preservation comparable across
        // policies and frozen cores; publication completes before any capture.
        // Its host backing is a member preceding Capture, so the terminal join
        // retires an outstanding upload before constructor-failure unwinding.
        for(auto& p:initial_partials){p.maximum=p.denominator=-817.25f;std::fill(std::begin(p.weighted),std::end(p.weighted),-817.25f);}
        output.upload(poison.data(),s);partials.upload(initial_partials.data(),s);check(QSA_RT(StreamSynchronize)(s));}
    /**
     * @brief Record the exact fixed-attention producer and merge for this run.
     * @throws std::runtime_error If native capture, function or geometry checks fail.
     * @details The producer grid is [rows,query_heads,8], the merge grid is
     * [rows,query_heads,1], and both use 256 threads per CTA. Input and output
     * addresses remain bound through all repeats and terminal graph retirement.
     */
    void record(){capture.record([&]{QSA_ATTENTION(args,capture.stream);},"selectedAttentionPartitions","reduceAttentionPartitions",
        dim3(args.row_capacity,args.query_heads,QwenSparsePartitions),dim3(args.row_capacity,args.query_heads,1),256,256,args.observations?"on":"off");}
};

/** @brief Explicit authored membership distribution; neither mode executes indexer ranking. */
enum class SelectedDistribution {CanonicalTies,CausalSpread512Groups};
/**
 * @brief Name an explicitly authored selected-token distribution.
 * @param d Closed membership-distribution policy used before capture replay.
 * @return Stable marker naming canonical ties or causal spread over complete groups.
 * @throws std::invalid_argument If @p d is outside the supported enum.
 */
inline const char* distributionName(SelectedDistribution d){
    switch(d){case SelectedDistribution::CanonicalTies:return "canonical-ties";
        case SelectedDistribution::CausalSpread512Groups:return "causal-spread-512groups";}
    throw std::invalid_argument("Unknown authored selected-attention distribution");}
/**
 * @brief Publish authored causal membership, excluding the indexer from attention capture.
 * @param f Persistent fixture owning device replay, selected IDs, counts and stream.
 * @param prefix Visible tokens preceding the first live row, within token capacity.
 * @param live Number of causal live rows, from zero through admitted row capacity.
 * @param distribution Explicit membership policy; it does not change attention math.
 * @return Per-capacity-row selected counts; inactive rows retain the -99 fixture sentinel.
 * @throws std::runtime_error For invalid geometry/membership or native publication failure.
 * @details Canonical ties retain the original first 512 complete groups. Spread
 * selects at most 512 distinct ordered four-token groups over the real complete
 * prefix, including its first and last groups when there are at least two.
 * Both append the causal tail and keep the fixed 2051 selected-token capacity.
 * The intentional terminal upload join completes publication before timing or
 * profiling. The local copy-retirement scope also joins outstanding uploads on
 * failure before local host vectors are destroyed; no upload enters capture.
 */
inline std::vector<std::int32_t> publish(Fixture& f,int prefix,int live,
    SelectedDistribution distribution=SelectedDistribution::CanonicalTies){
    (void)distributionName(distribution);
    require(prefix>=0&&live>=0&&live<=f.rows&&prefix<=f.capacity&&live<=f.capacity-prefix,
        "Authored selection exceeds actual native row/token capacity");
    std::vector<std::int32_t> selected(f.selected.size,-77),counts(f.rows,-99);
    FixtureCopyRetirement retire{&f.stream};
    for(int row=0;row<live;++row){const int visible=prefix+row+1,complete=std::min(visible/4,512)*4;
        counts[row]=complete+visible%4;auto* out=selected.data()+std::size_t(row)*2051;
        const int blocks=visible/4,groups=complete/4;
        for(int group=0;group<groups;++group){
            const int block=distribution==SelectedDistribution::CanonicalTies?group:
                (groups<=1?0:int(std::int64_t(group)*(blocks-1)/(groups-1)));
            for(int lane=0;lane<4;++lane)out[group*4+lane]=block*4+lane;}
        // Validate the authored distribution before copying it to device-owned state.
        for(int i=0;i<complete;++i)require(out[i]>=0&&out[i]<visible&&(i==0||out[i]>out[i-1]),
            "Authored complete groups are noncausal, duplicated or unordered");
        for(int i=0;i<visible%4;++i)out[complete+i]=(visible/4)*4+i;
        require(counts[row]<=2051,"Authored selection exceeds fixed attention membership");
        for(int i=0;i<counts[row];++i)require(out[i]>=0&&out[i]<visible&&(i==0||out[i]>out[i-1]),
            "Authored selection is noncausal, duplicated or unordered");}
    f.replay={prefix,live,17};f.state.upload(&f.replay,f.stream);f.selected.upload(selected.data(),f.stream);
    f.count.upload(counts.data(),f.stream);check(QSA_RT(StreamSynchronize)(f.stream));return counts;
}

/** @brief Native event bracket for one retained attention producer/merge transaction. */
struct Events final
{
    QSA_EVENT begin=nullptr,end=nullptr;QSA_STREAM stream;
    /**
     * @brief Allocate the two native event handles before any measured replay.
     * @param s Exact borrowed fixture stream used by every sampled capture.
     * @throws std::runtime_error If event creation fails; a created first event is
     * retired when creation of the second event fails.
     * @details The stream owner must outlive these handles. Events bracket retained
     * graph execution; creation is outside both capture and measured intervals.
     */
    explicit Events(QSA_STREAM s):stream(s){check(QSA_RT(EventCreate)(&begin));
        try{check(QSA_RT(EventCreate)(&end));}catch(...){(void)QSA_RT(EventDestroy)(begin);throw;}}
    /**
     * @brief Retire event handles after sample() has joined terminal event completion.
     * @details The stream remains borrowed. The last completed sample and fixture
     * capture retirement own completion; this destructor submits no inference work.
     */
    ~Events(){if(end)(void)QSA_RT(EventDestroy)(end);if(begin)(void)QSA_RT(EventDestroy)(begin);}
    /**
     * @brief Measure one retained replay and normalize by its captured transaction count.
     * @param c Recorded capture using this object's exact borrowed stream.
     * @return Native event duration in microseconds per complete attention transaction.
     * @throws std::runtime_error For unrecorded capture or event/launch API failure.
     * @details The intentional end-event join is a terminal benchmark observation,
     * outside the captured production policy. The device interval spans only the
     * retained producer/merge chain; setup uploads, host collection and correctness
     * checks occur outside it. The elapsed duration is divided by repeats exactly once.
     */
    float sample(Capture& c){check(QSA_RT(EventRecord)(begin,stream));c.launch();check(QSA_RT(EventRecord)(end,stream));
        check(QSA_RT(EventSynchronize)(end));float ms=0;check(QSA_RT(EventElapsedTime)(&ms,begin,end));return ms*1000.f/float(c.repeats);}
};
} // namespace observation115

/**
 * @test Exact Q6/TP4 attention-only event pairs or one selected-family profiler launch.
 * @details Observation sides retain the production FixedFP32 selected-attention
 * producer/merge pair and identical authored IDs. Paired event samples alternate
 * order to control drift; isolated profiler sides launch only the named policy.
 * Admitted rows 1..2048 share a bounded 8192 row-repeat work budget. Terminal
 * exports are capped at 128 MiB. All byte/count/dirty-guard checks occur after
 * timing and are fixture sanity,
 * independent of the separately registered native correctness regressions.
 */
TEST(QSA_SUITE,DISABLED_Observation115AttentionOnlyEvents){
    using namespace observation115;
    const auto read=[](const char* key){const char* text=std::getenv(key);require(text&&*text,"Exact observation115 selector is required");
        std::size_t used=0;const int value=std::stoi(text,&used);require(used==std::strlen(text),"Malformed observation115 selector");return value;};
    const int m=read("QFN_OBSERVATION115_M"),context=read("QFN_OBSERVATION115_CONTEXT"),heads=read("QFN_OBSERVATION115_HEADS");
    require(m>0&&m<=2048&&context>=m&&context<=65536&&heads==6,"Exact component requires M1..2048, valid context and Q6 compact heads");
    const char* selected=std::getenv("QFN_OBSERVATION115_PROFILE_SIDE");
    const std::string side=selected?selected:"observation-paired";
    const bool observationPair=side=="observation-paired";
    require(observationPair||side=="observation-on"||side=="observation-off","Unknown observation profile side");
    const std::string selectedDistribution=std::getenv("QFN_OBSERVATION115_SELECTION")?std::getenv("QFN_OBSERVATION115_SELECTION"):"canonical-ties";
    require(selectedDistribution=="canonical-ties"||selectedDistribution=="causal-spread-512groups","Unknown authored selection distribution");
    const auto distribution=selectedDistribution=="canonical-ties"?SelectedDistribution::CanonicalTies:SelectedDistribution::CausalSpread512Groups;
    // Timing records repeated complete pairs inside one retained graph; profiling
    // deliberately keeps a single pair and never normalizes unrelated work.
    const int defaultRepeats=observationPair?std::min(256,8192/m):1;
    const int repeats=std::getenv("QFN_OBSERVATION115_REPEATS")?read("QFN_OBSERVATION115_REPEATS"):defaultRepeats;
    require(observationPair?(repeats>=2&&repeats<=256):repeats==1,
        "Paired timing needs 2..256 captured repeats; isolated profiling needs exactly one");
    // Bound captured work independently of the live context. Default repeats
    // scale down to 16/8/4 for M512/1024/2048; explicit smaller counts stay exact.
    require(std::size_t(m)*std::size_t(repeats)<=8192,
        "Observation fixture requires M times captured repeats at most 8192");
    Fixture f(m,context-m,context,QwenSparseScalar::FP16);Inputs inputs(f,0,6,0,1);auto base=inputs.base();
    // The two graphs borrow precisely the same Q/K/V, replay and selected
    // buffers. Only the optional observation publication pointer differs;
    // each run retains its own output/partial authority before recording.
    FixedRun on(base,f.stream,true,repeats),off(base,f.stream,false,repeats);
    on.record();off.record();const auto counts=publish(f,context-m,m,distribution);Events events(f.stream);
    if(!observationPair){
        auto& chosen=side=="observation-on"?on.capture:off.capture;
        for(int i=0;i<5;++i)(void)events.sample(chosen);
        const float us=events.sample(chosen);
        std::cout<<"OBSERVATION115_SINGLE,"<<side<<",M="<<m<<",context="<<context
            <<",Q=6,KV=1,program=FixedFP32_8x257,selection="<<selectedDistribution
            <<",producer=selectedAttentionPartitions,merge=reduceAttentionPartitions,producer_threads=256,merge_threads=256"
            <<",repeats="<<repeats<<",occurrence=6,us="<<us<<'\n';
        const auto result=side=="observation-on"?on.output.download(f.stream):off.output.download(f.stream);
        for(std::size_t i=0;i<std::size_t(m)*6*256;++i)EXPECT_TRUE(std::isfinite(result[i]));
        return;
    }
    for(int i=0;i<5;++i){(void)events.sample(on.capture);(void)events.sample(off.capture);}
    for(int pair=0;pair<12;++pair){
        float enabled=0,disabled=0;
        if(pair&1){disabled=events.sample(off.capture);enabled=events.sample(on.capture);}
        else{enabled=events.sample(on.capture);disabled=events.sample(off.capture);}
        std::cout<<"OBSERVATION115_PAIR,"<<pair<<",M="<<m<<",context="<<context
            <<",Q=6,KV=1,program=FixedFP32_8x257,selection="<<selectedDistribution
            <<",producer=selectedAttentionPartitions,merge=reduceAttentionPartitions,producer_threads=256,merge_threads=256"
            <<",repeats="<<repeats<<",observation_on_us="<<enabled<<",observation_off_us="<<disabled<<'\n';
    }
    // Terminal fixture collection does not enter the measured graph/event
    // interval. Independent all-codec and retained-reset gates remain separate.
    const auto a=on.output.download(f.stream),b=off.output.download(f.stream);
    ASSERT_EQ(std::memcmp(a.data(),b.data(),a.size()*sizeof(float)),0);
    // These trailing fixture words have no arithmetic producer. Compare their
    // original poison as well as ON/OFF equality, including large prefill rows.
    const auto liveOutput=std::size_t(m)*6*QwenSparseHeadDim;
    ASSERT_EQ(std::memcmp(a.data()+liveOutput,on.poison.data()+liveOutput,16*sizeof(float)),0);
    ASSERT_EQ(std::memcmp(b.data()+liveOutput,off.poison.data()+liveOutput,16*sizeof(float)),0);
    const auto onPartials=on.partials.download(f.stream),offPartials=off.partials.download(f.stream);
    std::vector<float> canonicalPartials;canonicalPartials.reserve(onPartials.size()*258);
    for(std::size_t i=0;i<onPartials.size();++i){
        ASSERT_EQ(std::memcmp(&onPartials[i].maximum,&offPartials[i].maximum,sizeof(float)),0);
        ASSERT_EQ(std::memcmp(&onPartials[i].denominator,&offPartials[i].denominator,sizeof(float)),0);
        ASSERT_EQ(std::memcmp(onPartials[i].weighted,offPartials[i].weighted,sizeof(onPartials[i].weighted)),0);
        const auto row=i/(6*QwenSparsePartitions),partition=i%QwenSparsePartitions;
        if(counts[row]<=int(partition)*QwenSparsePartitionTokens){
            // Empty arithmetic partitions publish metadata and retain every
            // dirty FP32 weighted word; agreement alone would miss two writers.
            ASSERT_EQ(onPartials[i].maximum,-INFINITY);ASSERT_EQ(onPartials[i].denominator,0.0f);
            ASSERT_EQ(offPartials[i].maximum,-INFINITY);ASSERT_EQ(offPartials[i].denominator,0.0f);
            ASSERT_EQ(std::memcmp(onPartials[i].weighted,on.initial_partials[i].weighted,sizeof(onPartials[i].weighted)),0);
            ASSERT_EQ(std::memcmp(offPartials[i].weighted,off.initial_partials[i].weighted,sizeof(offPartials[i].weighted)),0);
            ASSERT_EQ(onPartials[i].key_elements,0u);ASSERT_EQ(onPartials[i].value_elements,0u);
            ASSERT_EQ(offPartials[i].key_elements,0u);ASSERT_EQ(offPartials[i].value_elements,0u);
        }
        canonicalPartials.push_back(onPartials[i].maximum);canonicalPartials.push_back(onPartials[i].denominator);
        canonicalPartials.insert(canonicalPartials.end(),std::begin(onPartials[i].weighted),std::end(onPartials[i].weighted));
    }
    // Optional bounded cross-core bytes are terminal fixture output only. Even
    // M2048 exports 108.75 MiB at most, with no counters, ABI padding or unused
    // rows. Keep this explicit artifact cap synchronized with the root reader.
    if(const char* destination=std::getenv("QFN_OBSERVATION115_EXPORT_DIR")){
        constexpr std::size_t MaxTerminalExportBytes=128ULL*1024*1024;
        const auto terminalBytes=(liveOutput+canonicalPartials.size())*sizeof(float);
        require(terminalBytes<=MaxTerminalExportBytes,"Observation terminal export exceeds 128 MiB");
        const std::filesystem::path directory(destination);
        require(*destination && !std::filesystem::exists(directory),"Observation export requires a fresh terminal directory");
        require(std::filesystem::create_directories(directory),"Observation export directory creation failed");
        const auto write=[&](const char* name,const float* data,std::size_t count){
            std::ofstream target(directory/name,std::ios::binary);target.exceptions(std::ios::failbit|std::ios::badbit);
            target.write(reinterpret_cast<const char*>(data),static_cast<std::streamsize>(count*sizeof(float)));target.close();};
        write("output-fp32.bin",a.data(),std::size_t(m)*6*256);
        write("partials-fp32.bin",canonicalPartials.data(),canonicalPartials.size());
        std::ofstream metadata(directory/"shape.json");metadata.exceptions(std::ios::failbit|std::ios::badbit);
        metadata<<"{\"M\":"<<m<<",\"context\":"<<context<<",\"Q\":6,\"KV\":1,\"partitions\":8,\"partial_coordinates\":258,\"selection\":"
            <<std::quoted(selectedDistribution)<<"}\n";metadata.close();
    }
    const auto observed=on.observations.download(f.stream);
    for(int row=0;row<m;++row)for(int head=0;head<6;++head){
        const auto& o=observed[row*6+head];
        ASSERT_EQ(o.selected_tokens,counts[row]);
        ASSERT_EQ(o.key_elements,std::uint64_t(counts[row])*256);
        ASSERT_EQ(o.value_elements,std::uint64_t(counts[row])*256);
    }
    return;
}
} // namespace llaminar2::qwen4exp::QSA_SUITE
