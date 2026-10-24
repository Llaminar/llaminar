/**
 * @file ROCmGDNMergedExtendedTrial.h
 * @brief Extended merged-input recurrence, independent historical and serial-row proof.
 *
 * The existing HIP fixture owns storage, graphs, streams and terminal retirement.
 * Control captures two native kernels and candidate one per repeat with immutable initial
 * state. This native integration fixture proves request-hole, tail-column and modular
 * address arithmetic without installing production policy or claiming timing.
 */
#pragma once
#include "kernels/gdn/GDNMergedRecurrence.h"
#include "config/GDNHeadAssignment.h"
#include <limits>

namespace gdn_merged_extended
{
/** @brief Reject a failed private admission or byte comparison.
 * @param yes Required condition.
 * @param message Borrowed diagnostic consumed before return.
 */
inline void require(bool yes,const char* message){gdn_paired_perf::require(yes,message);}

/**
 * @brief Own parameterized TP4-shaped merged/planar and independent oracle transactions.
 * @details Existing HipFloatBuffer/HipCapturedGraph owners provide allocation and
 * graph lifetime. Retirement is last-declared and joins before any native owner
 * or upload vector is destroyed, including exception paths. No graph advances
 * initial recurrent state; both sides always read the same immutable initial bank.
 */
struct Cell final
{
    static constexpr int heads=12,keys=4;
    int dk,dv,state,requests,per_request;
    bool normalize;
    std::vector<int> lengths;
    int rows,repeats,offset;
    std::vector<float> merged,alpha,beta,initial;
    HipStreamHandle stream;
    HipFloatBuffer dm,da,db,dl,dt,di;
    HipFloatBuffer cq,ck,cv;
    HipFloatBuffer co,po,cs,ps,cshots,pshots,ho,hs,hshots,so,ss;
    HipIntBuffer live;
    llaminar2::DeviceRequestRowRanges ranges;
    std::unique_ptr<HipCapturedGraph> control,candidate,historical,serial;
    /** @brief Last-declared owner performs the intentional diagnostic terminal join. */
    struct Retirement final
    {
        hipStream_t stream;
        /** @brief Fail fatally rather than release possibly referenced native storage. */
        ~Retirement(){if(hipStreamSynchronize(stream)!=hipSuccess)std::terminate();}
    } retirement;

    /**
     * @brief Capture one parameterized request matrix and its independent oracles.
     * @param m Physical rows per request,1..16.
     * @param r Independent request count,1 or3.
     * @param key_width Native DK specialization64 or128.
     * @param value_width Positive physical DV, including tail widths65/129.
     * @param norm Original raw Q/K normalization policy.
     * @param modular_offset Signed original global value-head offset.
     * @details Local heads12/source heads4 retain original TP4 coordinates.
     * Captured pointers/geometry are fixed across dirty length replay.
     */
    Cell(int m,int r,int key_width,int value_width,bool norm,int modular_offset):
        dk(key_width),dv(value_width),state(heads*dk*dv),requests(r),per_request(m),normalize(norm),lengths(r,m),
        rows(m*r),repeats(1),offset(modular_offset),
        merged(std::size_t(rows)*(2*keys*dk+heads*dv)),alpha(rows*heads),beta(rows*heads),initial(r*state),
        dm(merged.size()),da(alpha.size()),db(beta.size()),dl(std::vector<float>(heads,-0.375f)),
        dt(std::vector<float>(heads,0.125f)),di(r*state),
        cq(rows*heads*dk),ck(rows*heads*dk),cv(rows*heads*dv),
        co(rows*heads*dv),po(rows*heads*dv),cs(r*state),ps(r*state),
        cshots(rows*state),pshots(rows*state),ho(rows*heads*dv),hs(r*state),hshots(rows*state),so(rows*heads*dv),ss(r*state),live({m,m,m}),
        ranges(llaminar2::DeviceRequestRowRanges::deviceCounted(r,m,live.ptr)),retirement{stream.stream}
    {
        require(m>=1&&m<=16&&(r==1||r==3),"Invalid extended GDN request geometry");
        publish(lengths,0);
        control=std::make_unique<HipCapturedGraph>(stream.stream,[&]{
            for(int i=0;i<repeats;++i){
                if(!rocmGDN_deinterleave_qkv(dm.ptr,cq.ptr,ck.ptr,cv.ptr,ranges,keys,heads,dk,dv,offset,0,stream.stream))return false;
                // Exact public order: total physical rows, request count, physical per-request stride.
                if(!rocmGDN_chunk_forward_batched_effective(cq.ptr,ck.ptr,cv.ptr,da.ptr,db.ptr,dl.ptr,dt.ptr,
                    co.ptr,di.ptr,cs.ptr,rows,requests,per_request,heads,dk,dv,normalize,live.ptr,
                    cshots.ptr,state,rows,0,stream.stream))return false;
            }return true;
        });
        candidate=std::make_unique<HipCapturedGraph>(stream.stream,[&]{
            const auto trial=arguments();
            for(int i=0;i<repeats;++i)if(!llaminar2::gdn::launchROCmMergedRecurrence(trial,stream.stream))return false;
            return true;
        });
        inspect(false);inspect(true);
        const gdn_historical_native::Arguments oracle{
            .q=cq.ptr,.k=ck.ptr,.v=cv.ptr,.alpha=da.ptr,.beta=db.ptr,.a_log=dl.ptr,.dt_bias=dt.ptr,
            .output=ho.ptr,.initial=di.ptr,.updated=hs.ptr,.requests=requests,.rows=per_request,
            .heads=heads,.key_width=dk,.value_width=dv,.normalize=normalize,.lengths=live.ptr,.row_offset=0,
            .snapshots=hshots.ptr,.snapshot_stride=state,.snapshot_rows=rows,.stream=stream.stream};
        historical=std::make_unique<HipCapturedGraph>(stream.stream,[&]{return gdn_historical_native::launch(oracle);});
        serial=std::make_unique<HipCapturedGraph>(stream.stream,[&]{
            for(int request=0;request<requests;++request)for(int row=0;row<per_request;++row){
                auto one=oracle;const int flat=request*per_request+row;
                one.q=cq.ptr+flat*heads*dk;one.k=ck.ptr+flat*heads*dk;one.v=cv.ptr+flat*heads*dv;
                one.alpha=da.ptr+flat*heads;one.beta=db.ptr+flat*heads;one.output=so.ptr+flat*heads*dv;
                one.initial=ss.ptr+request*state;one.updated=ss.ptr+request*state;
                one.requests=1;one.rows=1;one.lengths=live.ptr+request;one.row_offset=row;
                one.snapshots=nullptr;one.snapshot_rows=0;
                if(!gdn_historical_native::launch(one))return false;
            }return true;
        });
    }

    /**
     * @brief Bind the original merged source and live publication owner for capture.
     * @return Borrowed immutable argument value; all pointers are Cell-owned.
     * @details No allocation or live value read occurs; grouped snapshots select
     * the original scalar-order recurrence rather than long-prefill arithmetic.
     */
    llaminar2::gdn::MergedRecurrenceInput arguments()
    {
        return {ranges,keys,heads,dk,dv,offset,normalize,dm.ptr,da.ptr,db.ptr,dl.ptr,dt.ptr,
            po.ptr,di.ptr,ps.ptr,pshots.ptr,state,rows};
    }

    /**
     * @brief Publish dirty deterministic source and poison every destination outside capture.
     * @param active One authored live prefix per request in[0,per_request].
     * @param seed Distinct deterministic contents for successive retained replays.
     * @details Prior work is terminally joined by check() before republishing.
     * Initial state and sources are copied through existing fixture owners; no
     * allocation, transfer, reset or upload is event-timed or graph-captured.
     */
    void publish(const std::vector<int>& active,int seed)
    {
        require(active.size()==std::size_t(requests),"Wrong length owner extent");
        for(int count:active)require(count>=0&&count<=per_request,"Invalid live prefix");
        lengths=active;
        for(std::size_t i=0;i<merged.size();++i)merged[i]=float(int((i*7+seed*11)%31)-15)*0.03125f;
        for(std::size_t i=0;i<alpha.size();++i){alpha[i]=float(int((i+seed)%6)-2)*0.125f;beta[i]=float(int((i+seed)%9)-4)*0.5f;}
        for(std::size_t i=0;i<initial.size();++i)initial[i]=float(int((i+seed)%17)-8)*0.00073f;
        // Exact signed-zero inputs and state must survive unchanged layout arithmetic.
        if(seed%2){for(std::size_t i=0;i<merged.size();i+=17)merged[i]=-0.0f;
            for(std::size_t i=0;i<initial.size();i+=19)initial[i]=-0.0f;}
        const std::array<float,4> gates{19.75f,19.875f,20.f,20.5f};
        for(std::size_t i=0;i<alpha.size();++i)alpha[i]=gates[i%gates.size()];
        dm.copyFrom(merged);da.copyFrom(alpha);db.copyFrom(beta);di.copyFrom(initial);ss.copyFrom(initial);
        for(auto* b:{&cq,&ck,&cv})b->fill(-73.f);
        co.fill(-77.f);po.fill(-77.f);cs.fill(-91.f);ps.fill(-91.f);cshots.fill(-83.f);pshots.fill(-83.f);ho.fill(-77.f);hs.fill(-91.f);hshots.fill(-83.f);so.fill(-77.f);
        checkHip(hipMemcpyAsync(live.ptr,lengths.data(),lengths.size()*sizeof(int),hipMemcpyHostToDevice,stream.stream),"GDN trial publish prefix");
        // Cell-owned length storage remains alive through the ordered publication join.
        checkHip(hipStreamSynchronize(stream.stream),"GDN trial setup publication join");
    }

    /**
     * @brief Check complete paired output/state/snapshots and read-only original source protection.
     * @details Downloads and byte comparisons are outside timing.
     * Whole output/state/snapshot native results require exact bytes.
     * The original raw recurrence is the arithmetic oracle; no prepared planes exist.
     */
    void check()
    {
        control->launch(stream.stream);candidate->launch(stream.stream);historical->launch(stream.stream);serial->launch(stream.stream);
        checkHip(hipStreamSynchronize(stream.stream),"GDN trial parity terminal join");
        // All four graphs have retired above. No device work or publication occurs
        // until check() returns, so each terminal image can serve every assertion.
        // Cache only the three reused candidate images; download each independent
        // partner once and release it after comparison to bound host staging.
        const auto output=po.toHost(),state_image=ps.toHost(),snapshots=pshots.toHost();
        for(const auto pair:{
            std::pair{&co,&output},std::pair{&cs,&state_image},
            std::pair{&cshots,&snapshots},std::pair{&ho,&output},
            std::pair{&so,&output},std::pair{&hs,&state_image},
            std::pair{&hshots,&snapshots}}){
            const auto a=pair.first->toHost();
            const auto& b=*pair.second;
            require(a.size()==b.size()&&std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0,"GDN complete-pair bytes differ");
        }
        const auto serial_state=ss.toHost();
        for(int request=0;request<requests;++request){
            const auto base=std::size_t(request)*state;
            const float* terminal=lengths[request]>0 ? snapshots.data()+std::size_t(request*per_request+lengths[request]-1)*state : initial.data()+base;
            require(std::memcmp(terminal,serial_state.data()+base,state*sizeof(float))==0,
                    "Merged accepted terminal differs from independent serial-row oracle");
        }
        const auto source=dm.toHost(),source_alpha=da.toHost(),source_beta=db.toHost(),source_state=di.toHost();
        require(std::memcmp(source.data(),merged.data(),merged.size()*sizeof(float))==0&&
            std::memcmp(source_alpha.data(),alpha.data(),alpha.size()*sizeof(float))==0&&
            std::memcmp(source_beta.data(),beta.data(),beta.size()*sizeof(float))==0&&
            std::memcmp(source_state.data(),initial.data(),initial.size()*sizeof(float))==0,"Read-only GDN source changed");

        const auto q=cq.toHost(),k=ck.toHost(),v=cv.toHost();
        for(int row=0;row<rows;++row)for(int h=0;h<heads;++h){
            int source_head=int((std::int64_t(h)+offset)%keys);if(source_head<0)source_head+=keys;
            const std::size_t from=std::size_t(row)*(2*keys*dk+heads*dv);
            for(int d=0;d<dk;++d){const auto to=(std::size_t(row)*heads+h)*dk+d;
                const float eq=row%per_request<lengths[row/per_request]?merged[from+source_head*dk+d]:-73.f;
                const float ek=row%per_request<lengths[row/per_request]?merged[from+keys*dk+source_head*dk+d]:-73.f;
                require(std::memcmp(&q[to],&eq,sizeof(float))==0&&std::memcmp(&k[to],&ek,sizeof(float))==0,
                        "Independent merged modular QK mapping differs");
            }
            for(int d=0;d<dv;++d){const auto to=(std::size_t(row)*heads+h)*dv+d;
                const float ev=row%per_request<lengths[row/per_request]?merged[from+2*keys*dk+h*dv+d]:-73.f;
                require(std::memcmp(&v[to],&ev,sizeof(float))==0,"Independent merged V mapping differs");
            }
        }

    }

    /**
     * @brief Authenticate complete2-node control or1-node merged DAG and argument identity.
     * @param prepared True for direct merged recurrence; false for the planar control.
     * @details Reports actual static resources without claiming achieved occupancy.
     * No graph is executed or mutated by this host-only native inspection.
     */
    void inspect(bool prepared)
    {
        auto& graph=prepared?*candidate:*control;
        std::size_t count=0;checkHip(hipGraphGetNodes(graph.graph,nullptr,&count),"GDN pair node count");
        require(count==std::size_t((prepared?1:2)*repeats),"GDN pair native node count differs");
        std::vector<hipGraphNode_t> nodes(count);checkHip(hipGraphGetNodes(graph.graph,nodes.data(),&count),"GDN pair nodes");
        std::vector<int> next(count,-1);std::size_t root=count;
        for(std::size_t i=0;i<count;++i){
            hipGraphNodeType type{};checkHip(hipGraphNodeGetType(nodes[i],&type),"GDN pair type");require(type==hipGraphNodeTypeKernel,"GDN pair has nonkernel work");
            std::size_t n=0;checkHip(hipGraphNodeGetDependencies(nodes[i],nullptr,&n),"GDN pair parent count");require(n<=1,"GDN pair multiparent");
            if(!n){require(root==count,"GDN pair multiple roots");root=i;}else{
                hipGraphNode_t parent{};checkHip(hipGraphNodeGetDependencies(nodes[i],&parent,&n),"GDN pair parent");
                const auto at=std::find(nodes.begin(),nodes.end(),parent);require(at!=nodes.end(),"GDN pair foreign parent");
                const auto j=std::size_t(at-nodes.begin());require(next[j]==-1,"GDN pair branch");next[j]=int(i);
            }
        }
        void* functions[2]{};int index=int(root);
        for(std::size_t step=0;step<count;++step){
            require(index>=0&&std::size_t(index)<count,"GDN pair broken chain");
            hipKernelNodeParams n{};checkHip(hipGraphKernelNodeGetParams(nodes[index],&n),"GDN pair params");
            const bool rec=prepared||step%2;const char* raw=hipKernelNameRefByPtr(n.func,stream.stream);require(raw,"GDN pair missing symbol");
            int status=-1;std::unique_ptr<char,decltype(&std::free)> name(abi::__cxa_demangle(raw,nullptr,nullptr,&status),&std::free);
            require(status==0&&bool(name),"GDN pair demangle failed");const std::string actual(name.get());
            const std::string expected=rec?(prepared?"rocm_gdn_merged_recurrent_step_kernel<"+std::to_string(dk)+">(":"rocm_gdn_recurrent_step_kernel<"+std::to_string(dk)+">("):
                "rocm_gdn_deinterleave_qkv_kernel(";
            require(actual.find("(anonymous namespace)::"+expected)!=std::string::npos,"GDN pair wrong native family");
            require(!functions[rec]||functions[rec]==n.func,"GDN pair role function changes");functions[rec]=n.func;
            const unsigned bx=rec?128:256;
            const unsigned gx=rec?requests*heads:((per_request*(2*heads*dk+heads*dv)+255)/256);
            const unsigned gy=rec?(dv+31)/32:requests,gz=1;
            const unsigned recurrent_lds=(2*dk+4+256+std::max(0,dk/4-22)*128)*sizeof(float);
            // Original deinterleave linear grid scales with physical rows.
            require(n.blockDim.x==bx&&n.blockDim.y==1&&n.blockDim.z==1&&n.gridDim.x==gx&&
                n.gridDim.y==gy&&n.gridDim.z==gz&&n.sharedMemBytes==(rec?recurrent_lds:0u),"GDN pair wrong native geometry");
            require(n.kernelParams,"GDN pair missing captured arguments");
            const auto pointer=[&](int arg,const void* expected_pointer){require(*static_cast<void*const*>(n.kernelParams[arg])==expected_pointer,"GDN pair captured pointer differs");};
            const auto integer=[&](int arg,int expected_value){require(*static_cast<const int*>(n.kernelParams[arg])==expected_value,"GDN pair captured scalar differs");};
            const auto rowargs=[&](int arg){const auto& r=*static_cast<const llaminar2::DeviceRequestRowRanges*>(n.kernelParams[arg]);require(r.requests()==requests&&r.rowsPerRequest()==per_request&&r.lengthOwner()==live.ptr,"GDN pair captured row owner differs");};
            if(rec){
                pointer(0,prepared?dm.ptr:cq.ptr);pointer(1,prepared?nullptr:ck.ptr);pointer(2,prepared?nullptr:cv.ptr);
                pointer(3,da.ptr);pointer(4,db.ptr);pointer(5,dl.ptr);pointer(6,dt.ptr);
                pointer(7,prepared?po.ptr:co.ptr);pointer(8,di.ptr);pointer(9,prepared?ps.ptr:cs.ptr);
                integer(10,requests);integer(11,per_request);integer(12,heads);integer(13,dv);
                require(*static_cast<const bool*>(n.kernelParams[14])==normalize,"GDN merged norm policy differs");pointer(15,live.ptr);integer(16,0);
                pointer(17,prepared?pshots.ptr:cshots.ptr);integer(18,state);integer(19,rows);
                if(prepared){integer(20,keys);integer(21,offset);}
            }else{
                pointer(0,dm.ptr);pointer(1,cq.ptr);pointer(2,ck.ptr);pointer(3,cv.ptr);rowargs(4);
                integer(5,keys);integer(6,heads);integer(7,dk);integer(8,dv);integer(9,offset);
            }
            if(step<(prepared?1u:2u)){hipFuncAttributes a{};checkHip(hipFuncGetAttributes(&a,n.func),"GDN pair resources");require(a.localSizeBytes==0,"GDN pair private memory");
                std::cout<<"GDN_MERGED_CAPTURE,side="<<(prepared?"candidate":"control")<<",role="<<(rec?"recurrence":"preparation")
                    <<",M="<<per_request<<",R="<<requests<<",offset="<<offset<<",norm="<<normalize<<",key_heads=4,heads=12,DK="<<dk<<",DV="<<dv<<",grid="<<n.gridDim.x<<'x'<<n.gridDim.y<<'x'<<n.gridDim.z
                    <<",block="<<bx<<",registers="<<a.numRegs<<",static_lds="<<a.sharedSizeBytes<<",dynamic_lds="<<n.sharedMemBytes
                    <<",private="<<a.localSizeBytes<<",symbol="<<gdn_paired_perf::escaped(actual)<<'\n';}
            index=next[index];
        }require(index==-1,"GDN pair cyclic chain");
    }
};

/**
 * @brief Compare request-hole/full/empty dirty replay against both independent native oracles.
 * @param cell Borrowed complete native owners; every replay is joined before reset.
 */
inline void parity(Cell& cell)
{
    const int m=cell.per_request;
    // Earliest request empty cannot suppress a later request; replay dirties all
    // rows, then shrinks and grows each independently with stable graph pointers.
    int seed=0;
    for(const auto& lengths:std::vector<std::vector<int>>{{0,m,m>1?m-1:0},{m,0,m},{0,0,0},{m,m,m}}){
        cell.publish(lengths,seed++);cell.check();
    }
}
}


/** @test Exact merged/deinterleaved recurrence over request holes and tail columns.
 * @details Positive DV admission and native tail geometry are exercised directly; this
 * fixture does not select a production execution policy.
 * This is an explicit integration test, never a device-free Unit test.
 */
TEST(Test__ROCmGDNPaddedRealLength, MergedExtendedGeometryBytes)
{
    ASSERT_TRUE(hasROCm());checkHip(hipSetDevice(0),"GDN extended device");
    for(int m=1;m<=16;++m)for(int dk:{64,128})for(int dv:{64,65,128,129})for(bool norm:{false,true}){
        const int offset=norm?5:-5;
        SCOPED_TRACE(::testing::Message()<<"M="<<m<<" R=3 DK="<<dk<<" DV="<<dv<<" norm="<<norm<<" offset="<<offset);
        gdn_merged_extended::Cell cell(m,3,dk,dv,norm,offset);
        gdn_merged_extended::parity(cell);
    }
    for(int offset:{0,1,-1,std::numeric_limits<int>::min(),std::numeric_limits<int>::max()}){
        SCOPED_TRACE(offset);gdn_merged_extended::Cell cell(3,3,128,65,true,offset);
        gdn_merged_extended::parity(cell);
    }
}

/** @test Captured merged M1 updates the same live bank repeatedly, including empty transitions.
 * @details Independent frozen historical scalar recurrence owns the oracle.
 * Neither graph resets state. Dirty data/length publication remains outside
 * capture; full terminal state and all output bytes compare after every step.
 */
TEST(Test__ROCmGDNPaddedRealLength, MergedInPlaceM1SerialHistoryBytes)
{
    ASSERT_TRUE(hasROCm());checkHip(hipSetDevice(0),"GDN in-place device");
    for(int dk:{64,128})for(int dv:{65,128,129})for(bool norm:{false,true}){
        gdn_merged_extended::Cell cell(1,1,dk,dv,norm,-5);
        HipFloatBuffer liveState(cell.initial),oracleState(cell.initial);
        HipFloatBuffer actual(cell.heads*dv),expected(cell.heads*dv);
        std::unique_ptr<HipCapturedGraph> merged,serial;
        // Last-declared owner joins all local borrows before their destruction.
        gdn_merged_extended::Cell::Retirement retirement{cell.stream.stream};
        auto args=cell.arguments();args.initial_state=liveState.ptr;args.updated_state=liveState.ptr;
        args.output=actual.ptr;args.snapshots=nullptr;args.snapshot_stride_floats=0;args.max_snapshot_rows=0;
        merged=std::make_unique<HipCapturedGraph>(cell.stream.stream,[&]{
            return llaminar2::gdn::launchROCmMergedRecurrence(args,cell.stream.stream);
        });
        const gdn_historical_native::Arguments reference{
            .q=cell.cq.ptr,.k=cell.ck.ptr,.v=cell.cv.ptr,.alpha=cell.da.ptr,.beta=cell.db.ptr,
            .a_log=cell.dl.ptr,.dt_bias=cell.dt.ptr,.output=expected.ptr,
            .initial=oracleState.ptr,.updated=oracleState.ptr,
            .requests=1,.rows=1,.heads=cell.heads,.key_width=dk,.value_width=dv,.normalize=norm,
            .lengths=cell.live.ptr,.row_offset=0,.snapshots=nullptr,.snapshot_stride=0,.snapshot_rows=0,
            .stream=cell.stream.stream};
        serial=std::make_unique<HipCapturedGraph>(cell.stream.stream,[&]{
            return rocmGDN_deinterleave_qkv(cell.dm.ptr,cell.cq.ptr,cell.ck.ptr,cell.cv.ptr,cell.ranges,
                       cell.keys,cell.heads,dk,dv,cell.offset,0,cell.stream.stream)&&
                   gdn_historical_native::launch(reference);
        });
        // Authenticate the merged live-state alias itself, not a test-only copy.
        std::size_t count=0;checkHip(hipGraphGetNodes(merged->graph,nullptr,&count),"in-place node count");
        gdn_merged_extended::require(count==1,"In-place merged graph must have one kernel");
        hipGraphNode_t node{};checkHip(hipGraphGetNodes(merged->graph,&node,&count),"in-place node");
        hipGraphNodeType kind{};checkHip(hipGraphNodeGetType(node,&kind),"in-place node kind");
        gdn_merged_extended::require(kind==hipGraphNodeTypeKernel,"In-place graph has nonkernel work");
        hipKernelNodeParams native{};checkHip(hipGraphKernelNodeGetParams(node,&native),"in-place params");
        std::size_t canonicalCount=1;hipGraphNode_t canonical{};
        checkHip(hipGraphGetNodes(cell.candidate->graph,&canonical,&canonicalCount),"reference merged node");
        hipKernelNodeParams original{};checkHip(hipGraphKernelNodeGetParams(canonical,&original),"reference merged params");
        gdn_merged_extended::require(native.func==original.func&&native.gridDim.x==cell.heads&&
            native.gridDim.y==unsigned(1+(dv-1)/32)&&native.gridDim.z==1&&
            native.blockDim.x==128&&native.blockDim.y==1&&native.blockDim.z==1&&
            native.sharedMemBytes==original.sharedMemBytes&&native.kernelParams,
            "In-place native function/geometry differs");
        const auto ptr=[&](int i,const void* p){gdn_merged_extended::require(
            *static_cast<void*const*>(native.kernelParams[i])==p,"In-place captured pointer differs");};
        ptr(0,cell.dm.ptr);ptr(1,nullptr);ptr(2,nullptr);ptr(3,cell.da.ptr);ptr(4,cell.db.ptr);
        ptr(5,cell.dl.ptr);ptr(6,cell.dt.ptr);ptr(7,actual.ptr);ptr(8,liveState.ptr);ptr(9,liveState.ptr);
        ptr(15,cell.live.ptr);ptr(17,nullptr);
        const auto integer=[&](int i,int value){gdn_merged_extended::require(
            *static_cast<const int*>(native.kernelParams[i])==value,"In-place captured scalar differs");};
        integer(10,1);integer(11,1);integer(12,cell.heads);integer(13,dv);integer(16,0);
        integer(18,0);integer(19,0);integer(20,cell.keys);integer(21,cell.offset);
        gdn_merged_extended::require(*static_cast<const bool*>(native.kernelParams[14])==norm,
            "In-place captured normalization policy differs");
        int seed=1;
        for(int active:{1,1,0,0,1,0,1,1,1,0,1,1}){
            cell.publish({active},seed++);actual.fill(-77.f);expected.fill(-77.f);
            merged->launch(cell.stream.stream);serial->launch(cell.stream.stream);
            checkHip(hipStreamSynchronize(cell.stream.stream),"in-place history terminal join");
            const auto out=actual.toHost(),want=expected.toHost(),state=liveState.toHost(),oracle=oracleState.toHost();
            gdn_merged_extended::require(std::memcmp(out.data(),want.data(),out.size()*sizeof(float))==0,
                "In-place merged output differs from independent historical serial");
            gdn_merged_extended::require(std::memcmp(state.data(),oracle.data(),state.size()*sizeof(float))==0,
                "In-place merged live state differs from independent historical serial");
        }
    }
}

/** @test Mandatory merged M1 backend API uses its bound persistent live bank.
 * @details Independent frozen historical scalar recurrence owns the oracle.
 * Neither graph resets state. Dirty data/length publication remains outside
 * capture; full terminal state and all output bytes compare after every step.
 */
TEST(Test__ROCmGDNPaddedRealLength, MergedMandatoryM1BackendStateHistoryBytes)
{
    ASSERT_TRUE(hasROCm());checkHip(hipSetDevice(0),"GDN in-place device");
    for(int dk:{64,128})for(int dv:{65,128,129})for(bool norm:{false,true}){
        gdn_merged_extended::Cell cell(1,1,dk,dv,norm,-5);
        HipFloatBuffer liveState(cell.initial),oracleState(cell.initial);
        HipFloatBuffer actual(cell.heads*dv),expected(cell.heads*dv);
        ROCmGatedDeltaNet engine(0);
        engine.setGPUStream(cell.stream.stream);
        gdn_merged_extended::require(engine.bindDeviceState({
            .primary_state=liveState.ptr,.primary_state_floats=cell.state,
            .request_state_bank=liveState.ptr,.request_state_bank_floats=std::size_t(cell.state),
            .request_capacity=1}),"Mandatory merged API rejected native state binding");
        std::unique_ptr<HipCapturedGraph> merged,serial;
        // Last-declared owner joins all local borrows before their destruction.
        gdn_merged_extended::Cell::Retirement retirement{cell.stream.stream};
        merged=std::make_unique<HipCapturedGraph>(cell.stream.stream,[&]{
            return engine.chunkForwardMergedQKV(cell.dm.ptr,2*cell.keys*dk+cell.heads*dv,
                cell.da.ptr,cell.db.ptr,cell.dl.ptr,cell.dt.ptr,actual.ptr,nullptr,
                1,cell.keys,cell.heads,dk,dv,cell.offset,64,norm);
        });
        const gdn_historical_native::Arguments reference{
            .q=cell.cq.ptr,.k=cell.ck.ptr,.v=cell.cv.ptr,.alpha=cell.da.ptr,.beta=cell.db.ptr,
            .a_log=cell.dl.ptr,.dt_bias=cell.dt.ptr,.output=expected.ptr,
            .initial=oracleState.ptr,.updated=oracleState.ptr,
            .requests=1,.rows=1,.heads=cell.heads,.key_width=dk,.value_width=dv,.normalize=norm,
            .lengths=cell.live.ptr,.row_offset=0,.snapshots=nullptr,.snapshot_stride=0,.snapshot_rows=0,
            .stream=cell.stream.stream};
        serial=std::make_unique<HipCapturedGraph>(cell.stream.stream,[&]{
            return rocmGDN_deinterleave_qkv(cell.dm.ptr,cell.cq.ptr,cell.ck.ptr,cell.cv.ptr,cell.ranges,
                       cell.keys,cell.heads,dk,dv,cell.offset,0,cell.stream.stream)&&
                   gdn_historical_native::launch(reference);
        });
        // Authenticate the merged live-state alias itself, not a test-only copy.
        std::size_t count=0;checkHip(hipGraphGetNodes(merged->graph,nullptr,&count),"in-place node count");
        gdn_merged_extended::require(count==1,"In-place merged graph must have one kernel");
        hipGraphNode_t node{};checkHip(hipGraphGetNodes(merged->graph,&node,&count),"in-place node");
        hipGraphNodeType kind{};checkHip(hipGraphNodeGetType(node,&kind),"in-place node kind");
        gdn_merged_extended::require(kind==hipGraphNodeTypeKernel,"In-place graph has nonkernel work");
        hipKernelNodeParams native{};checkHip(hipGraphKernelNodeGetParams(node,&native),"in-place params");
        std::size_t canonicalCount=1;hipGraphNode_t canonical{};
        checkHip(hipGraphGetNodes(cell.candidate->graph,&canonical,&canonicalCount),"reference merged node");
        hipKernelNodeParams original{};checkHip(hipGraphKernelNodeGetParams(canonical,&original),"reference merged params");
        gdn_merged_extended::require(native.func==original.func&&native.gridDim.x==cell.heads&&
            native.gridDim.y==unsigned(1+(dv-1)/32)&&native.gridDim.z==1&&
            native.blockDim.x==128&&native.blockDim.y==1&&native.blockDim.z==1&&
            native.sharedMemBytes==original.sharedMemBytes&&native.kernelParams,
            "In-place native function/geometry differs");
        const auto ptr=[&](int i,const void* p){gdn_merged_extended::require(
            *static_cast<void*const*>(native.kernelParams[i])==p,"In-place captured pointer differs");};
        ptr(0,cell.dm.ptr);ptr(1,nullptr);ptr(2,nullptr);ptr(3,cell.da.ptr);ptr(4,cell.db.ptr);
        ptr(5,cell.dl.ptr);ptr(6,cell.dt.ptr);ptr(7,actual.ptr);ptr(8,liveState.ptr);ptr(9,liveState.ptr);
        ptr(15,nullptr);ptr(17,nullptr);
        const auto integer=[&](int i,int value){gdn_merged_extended::require(
            *static_cast<const int*>(native.kernelParams[i])==value,"In-place captured scalar differs");};
        integer(10,1);integer(11,1);integer(12,cell.heads);integer(13,dv);integer(16,0);
        integer(18,0);integer(19,0);integer(20,cell.keys);integer(21,cell.offset);
        gdn_merged_extended::require(*static_cast<const bool*>(native.kernelParams[14])==norm,
            "In-place captured normalization policy differs");
        int seed=1;
        for(int active:{1,1,1,1,1,1,1,1,1,1,1,1}){
            cell.publish({active},seed++);actual.fill(-77.f);expected.fill(-77.f);
            merged->launch(cell.stream.stream);serial->launch(cell.stream.stream);
            checkHip(hipStreamSynchronize(cell.stream.stream),"in-place history terminal join");
            const auto out=actual.toHost(),want=expected.toHost(),state=liveState.toHost(),oracle=oracleState.toHost();
            gdn_merged_extended::require(std::memcmp(out.data(),want.data(),out.size()*sizeof(float))==0,
                "In-place merged output differs from independent historical serial");
            gdn_merged_extended::require(std::memcmp(state.data(),oracle.data(),state.size()*sizeof(float))==0,
                "In-place merged live state differs from independent historical serial");
        }
    }
}

/** @test Retained larger M1 verifier envelopes preserve original bank and restore authority.
 * @details The mandatory backend API binds exact state geometry and four retained
 * snapshot rows. Its captured recurrence reads the live bank, uses speculative
 * work as destination and publishes only row0; device-index restore commits row0
 * into the same live bank. The independent historical scalar kernel proves bytes.
 * A separate native padded-stride case tests the original low-level stride API;
 * the backend's state-size binding does not expose an independent padded stride.
 */
TEST(Test__ROCmGDNPaddedRealLength, MergedM1RetainedVerifierEnvelopeAndRestoreBytes)
{
    ASSERT_TRUE(hasROCm());checkHip(hipSetDevice(0),"GDN retained verifier envelope device");
    for(int dv:{65,128})for(int padding:{0,13}){
        constexpr int retained_rows=4;
        gdn_merged_extended::Cell cell(1,1,128,dv,true,-5);
        const int stride=cell.state+padding;
        HipFloatBuffer liveState(cell.initial),oracleState(cell.initial);
        HipFloatBuffer work(cell.state,-91.f),oracleWork(cell.state,-91.f);
        HipFloatBuffer snapshots(retained_rows*stride,-83.f),oracleSnapshots(retained_rows*stride,-83.f);
        HipFloatBuffer actual(cell.heads*dv,-77.f),expected(cell.heads*dv,-77.f);
        HipIntBuffer accepted({0});
        ROCmGatedDeltaNet engine(0);engine.setGPUStream(cell.stream.stream);
        gdn_merged_extended::require(engine.bindDeviceState({
            .primary_state=liveState.ptr,.primary_state_floats=cell.state,
            .request_state_bank=liveState.ptr,.request_state_bank_floats=std::size_t(cell.state),
            .request_capacity=1}),"Verifier live bank binding failed");
        engine.bindVerifierStateCaptureWorkspace(snapshots.ptr,retained_rows,cell.state);
        engine.bindSpeculativeStateWorkspace(work.ptr,cell.state);
        std::unique_ptr<HipCapturedGraph> candidate,reference,restore;
        gdn_merged_extended::Cell::Retirement retirement{cell.stream.stream};
        auto nativeInput=cell.arguments();nativeInput.output=actual.ptr;
        nativeInput.initial_state=liveState.ptr;nativeInput.updated_state=work.ptr;
        nativeInput.snapshots=snapshots.ptr;nativeInput.snapshot_stride_floats=stride;
        nativeInput.max_snapshot_rows=retained_rows;
        candidate=std::make_unique<HipCapturedGraph>(cell.stream.stream,[&]{
            if(padding)return llaminar2::gdn::launchROCmMergedRecurrence(nativeInput,cell.stream.stream);
            return engine.chunkForwardMergedQKV(cell.dm.ptr,2*cell.keys*cell.dk+cell.heads*dv,
                cell.da.ptr,cell.db.ptr,cell.dl.ptr,cell.dt.ptr,actual.ptr,nullptr,
                1,cell.keys,cell.heads,cell.dk,dv,cell.offset,64,true);
        });
        const gdn_historical_native::Arguments oracle{
            .q=cell.cq.ptr,.k=cell.ck.ptr,.v=cell.cv.ptr,.alpha=cell.da.ptr,.beta=cell.db.ptr,
            .a_log=cell.dl.ptr,.dt_bias=cell.dt.ptr,.output=expected.ptr,
            .initial=oracleState.ptr,.updated=oracleWork.ptr,
            .requests=1,.rows=1,.heads=cell.heads,.key_width=cell.dk,.value_width=dv,.normalize=true,
            .lengths=nullptr,.row_offset=0,.snapshots=oracleSnapshots.ptr,
            .snapshot_stride=stride,.snapshot_rows=retained_rows,.stream=cell.stream.stream};
        reference=std::make_unique<HipCapturedGraph>(cell.stream.stream,[&]{
            return rocmGDN_deinterleave_qkv(cell.dm.ptr,cell.cq.ptr,cell.ck.ptr,cell.cv.ptr,cell.ranges,
                cell.keys,cell.heads,cell.dk,dv,cell.offset,0,cell.stream.stream)&&gdn_historical_native::launch(oracle);
        });
        if(!padding)restore=std::make_unique<HipCapturedGraph>(cell.stream.stream,[&]{
            return engine.restoreVerifierStateCaptureRowFromDeviceIndex(nullptr,accepted.ptr,cell.stream.stream);
        });
        std::size_t count=0;checkHip(hipGraphGetNodes(candidate->graph,nullptr,&count),"verifier node count");
        gdn_merged_extended::require(count==1,"Retained verifier M1 must capture one native kernel");
        hipGraphNode_t node{};checkHip(hipGraphGetNodes(candidate->graph,&node,&count),"verifier node");
        hipGraphNodeType kind{};checkHip(hipGraphNodeGetType(node,&kind),"verifier node type");
        gdn_merged_extended::require(kind==hipGraphNodeTypeKernel,"Retained verifier has nonkernel work");
        hipKernelNodeParams params{};checkHip(hipGraphKernelNodeGetParams(node,&params),"verifier params");
        std::size_t baseCount=1;hipGraphNode_t base{};
        checkHip(hipGraphGetNodes(cell.candidate->graph,&base,&baseCount),"merged reference node");
        hipKernelNodeParams baseParams{};checkHip(hipGraphKernelNodeGetParams(base,&baseParams),"merged reference params");
        gdn_merged_extended::require(params.func==baseParams.func&&params.kernelParams&&
            params.gridDim.x==cell.heads&&params.gridDim.y==unsigned(1+(dv-1)/32)&&params.gridDim.z==1&&
            params.blockDim.x==128&&params.blockDim.y==1&&params.blockDim.z==1&&params.sharedMemBytes==baseParams.sharedMemBytes,
            "Retained verifier native function/geometry changed");
        const auto pointer=[&](int i,const void* value){gdn_merged_extended::require(
            *static_cast<void*const*>(params.kernelParams[i])==value,"Verifier captured owner pointer differs");};
        pointer(0,cell.dm.ptr);pointer(1,nullptr);pointer(2,nullptr);pointer(3,cell.da.ptr);pointer(4,cell.db.ptr);
        pointer(5,cell.dl.ptr);pointer(6,cell.dt.ptr);pointer(7,actual.ptr);pointer(8,liveState.ptr);pointer(9,work.ptr);
        pointer(15,padding?cell.live.ptr:nullptr);pointer(17,snapshots.ptr);
        const auto integer=[&](int i,int value){gdn_merged_extended::require(
            *static_cast<const int*>(params.kernelParams[i])==value,"Verifier captured envelope scalar differs");};
        integer(10,1);integer(11,1);integer(12,cell.heads);integer(13,dv);integer(16,0);
        integer(18,stride);integer(19,retained_rows);integer(20,cell.keys);integer(21,cell.offset);
        gdn_merged_extended::require(*static_cast<const bool*>(params.kernelParams[14]),"Verifier normalization changed");
        for(int seed=1;seed<=3;++seed){
            cell.publish({1},seed);actual.fill(-77.f);expected.fill(-77.f);
            snapshots.fill(-83.f);oracleSnapshots.fill(-83.f);work.fill(-91.f);oracleWork.fill(-91.f);
            candidate->launch(cell.stream.stream);reference->launch(cell.stream.stream);
            checkHip(hipStreamSynchronize(cell.stream.stream),"retained verifier oracle terminal join");
            for(auto pair:{std::pair{&actual,&expected},std::pair{&snapshots,&oracleSnapshots},
                          std::pair{&liveState,&oracleState},std::pair{&work,&oracleWork}}){
                const auto got=pair.first->toHost(),want=pair.second->toHost();
                gdn_merged_extended::require(std::memcmp(got.data(),want.data(),got.size()*sizeof(float))==0,
                    "Retained envelope output/snapshots/live/work bytes differ");
            }
            if(restore){
                restore->launch(cell.stream.stream);
                checkHip(hipMemcpyAsync(oracleState.ptr,oracleSnapshots.ptr,cell.state*sizeof(float),
                    hipMemcpyDeviceToDevice,cell.stream.stream),"independent oracle accepted state copy");
                checkHip(hipStreamSynchronize(cell.stream.stream),"verifier restore terminal join");
                const auto got=liveState.toHost(),want=oracleState.toHost();
                gdn_merged_extended::require(std::memcmp(got.data(),want.data(),got.size()*sizeof(float))==0,
                    "Device-index restore did not publish accepted state to original live/request0 bank");
            }
        }
    }
}
