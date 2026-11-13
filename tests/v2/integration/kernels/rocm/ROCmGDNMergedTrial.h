/**
 * @file ROCmGDNMergedTrial.h
 * @brief Opt-in complete deinterleave/direct merged-input recurrence component trial.
 *
 * The existing HIP fixture owns storage, graphs, streams and terminal retirement.
 * Control captures two native kernels and candidate one per repeat with immutable initial
 * state. This private TP4 trial proves complete-pair bytes/economics, not production
 * admission, the identity-M1 alias route, or complete format/head geometry coverage.
 */
#pragma once
#include "kernels/gdn/GDNMergedRecurrence.h"
#include "config/GDNHeadAssignment.h"
#include <limits>

namespace gdn_merged_trial
{
/** @brief Reject a failed private admission or byte comparison.
 * @param yes Required condition.
 * @param message Borrowed diagnostic consumed before return.
 */
inline void require(bool yes,const char* message){gdn_paired_perf::require(yes,message);}

/**
 * @brief Own one fixed TP4-shaped pair of retained native transactions.
 * @details Existing HipFloatBuffer/HipCapturedGraph owners provide allocation and
 * graph lifetime. Retirement is last-declared and joins before any native owner
 * or upload vector is destroyed, including exception paths. No graph advances
 * initial recurrent state; both sides always read the same immutable initial bank.
 */
struct Cell final
{
    static constexpr int heads=12,keys=4,dk=128,dv=128,state=heads*dk*dv;
    int rows,repeats,offset;
    std::vector<float> merged,alpha,beta,initial;
    HipStreamHandle stream;
    HipFloatBuffer dm,da,db,dl,dt,di;
    HipFloatBuffer cq,ck,cv;
    HipFloatBuffer co,po,cs,ps,cshots,pshots;
    HipIntBuffer live;
    llaminar2::DeviceRequestRowRanges ranges;
    std::unique_ptr<HipCapturedGraph> control,candidate;
    /** @brief Last-declared owner performs the intentional diagnostic terminal join. */
    struct Retirement final
    {
        hipStream_t stream;
        /** @brief Fail fatally rather than release possibly referenced native storage. */
        ~Retirement(){if(hipStreamSynchronize(stream)!=hipSuccess)std::terminate();}
    } retirement;

    /**
     * @brief Admit and capture one M1/M16 complete-pair geometry before timing.
     * @param m Physical rows in one request, exactly1 or16.
     * @param repeat_count Complete two-node pairs per captured submission,1..256.
     * @details Source dimensions follow canonical QFN16key/48value heads over TP4,
     * using head partition[0,6)/24. Local packed modular mapping has offset0.
     * Model-free synthetic FP32 values do not claim to be real-model activations.
     */
    Cell(int m,int repeat_count,int modular_offset=0):rows(m),repeats(repeat_count),offset(modular_offset),
        merged(std::size_t(m)*(2*keys*dk+heads*dv)),alpha(m*heads),beta(m*heads),initial(state),
        dm(merged.size()),da(alpha.size()),db(beta.size()),dl(std::vector<float>(heads,-0.375f)),
        dt(std::vector<float>(heads,0.125f)),di(state),
        cq(m*heads*dk),ck(m*heads*dk),cv(m*heads*dv),co(m*heads*dv),po(m*heads*dv),cs(state),ps(state),
        cshots(m*state),pshots(m*state),live(m),
        ranges(llaminar2::DeviceRequestRowRanges::deviceCounted(1,m,live.ptr)),retirement{stream.stream}
    {
        require((rows==1||rows==16)&&repeats>=1&&repeats<=256,"Invalid private GDN cell");
        const auto assignment=llaminar2::GDNHeadAssignment::fromPartition(16,48,0,6,24);
        require(assignment.localKeyHeads()==keys&&assignment.localValueHeads()==heads,"TP4 GDN head assignment changed");
        publish(rows,0);
        control=std::make_unique<HipCapturedGraph>(stream.stream,[&]{
            for(int i=0;i<repeats;++i){
                if(!rocmGDN_deinterleave_qkv(dm.ptr,cq.ptr,ck.ptr,cv.ptr,ranges,keys,heads,dk,dv,offset,0,stream.stream))return false;
                // Exact public order: total physical M, request count1, physical request strideM.
                if(!rocmGDN_chunk_forward_batched_effective(cq.ptr,ck.ptr,cv.ptr,da.ptr,db.ptr,dl.ptr,dt.ptr,
                    co.ptr,di.ptr,cs.ptr,rows,1,rows,heads,dk,dv,true,live.ptr,
                    rows==1?nullptr:cshots.ptr,state,rows==1?0:rows,0,stream.stream))return false;
            }return true;
        });
        candidate=std::make_unique<HipCapturedGraph>(stream.stream,[&]{
            const auto trial=arguments();
            for(int i=0;i<repeats;++i)if(!llaminar2::gdn::launchROCmMergedRecurrence(trial,stream.stream))return false;
            return true;
        });
        inspect(false);inspect(true);
    }

    /**
     * @brief Bind the original merged source and live publication owner for capture.
     * @return Borrowed immutable argument value; all pointers are Cell-owned.
     * @details No allocation or live value read occurs; grouped snapshots select
     * the original scalar-order recurrence rather than long-prefill arithmetic.
     */
    llaminar2::gdn::MergedRecurrenceInput arguments()
    {
        return {ranges,keys,heads,dk,dv,offset,true,dm.ptr,da.ptr,db.ptr,dl.ptr,dt.ptr,
            po.ptr,di.ptr,ps.ptr,rows==1?nullptr:pshots.ptr,state,rows==1?0:rows};
    }

    /**
     * @brief Publish dirty deterministic source and poison every destination outside capture.
     * @param active Original live prefix in[0,physicalM].
     * @param seed Distinct deterministic contents for successive retained replays.
     * @details Prior work is terminally joined by check() before republishing.
     * Initial state and sources are copied through existing fixture owners; no
     * allocation, transfer, reset or upload is event-timed or graph-captured.
     */
    void publish(int active,int seed)
    {
        require(active>=0&&active<=rows,"Invalid live prefix");
        for(std::size_t i=0;i<merged.size();++i)merged[i]=float(int((i*7+seed*11)%31)-15)*0.03125f;
        for(std::size_t i=0;i<alpha.size();++i){alpha[i]=float(int((i+seed)%6)-2)*0.125f;beta[i]=float(int((i+seed)%9)-4)*0.5f;}
        for(std::size_t i=0;i<initial.size();++i)initial[i]=float(int((i+seed)%17)-8)*0.00073f;
        dm.copyFrom(merged);da.copyFrom(alpha);db.copyFrom(beta);di.copyFrom(initial);
        for(auto* b:{&cq,&ck,&cv})b->fill(-73.f);
        co.fill(-77.f);po.fill(-77.f);cs.fill(-91.f);ps.fill(-91.f);cshots.fill(-83.f);pshots.fill(-83.f);
        checkHip(hipMemcpyAsync(live.ptr,&active,sizeof(active),hipMemcpyHostToDevice,stream.stream),"GDN trial publish prefix");
        // Stack publication storage must survive its asynchronous copy.
        checkHip(hipStreamSynchronize(stream.stream),"GDN trial setup publication join");
    }

    /**
     * @brief Check complete paired output/state/snapshots and read-only original source protection.
     * @param active Authored live prefix just published to the original device owner.
     * @details Downloads and byte comparisons are outside timing.
     * Whole output/state/snapshot native results require exact bytes.
     * The original raw recurrence is the arithmetic oracle; no prepared planes exist.
     */
    void check(int active)
    {
        control->launch(stream.stream);candidate->launch(stream.stream);
        checkHip(hipStreamSynchronize(stream.stream),"GDN trial parity terminal join");
        for(auto pair:{std::pair{&co,&po},std::pair{&cs,&ps},std::pair{&cshots,&pshots}}){
            const auto a=pair.first->toHost(),b=pair.second->toHost();
            require(a.size()==b.size()&&std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0,"GDN complete-pair bytes differ");
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
                const float eq=row<active?merged[from+source_head*dk+d]:-73.f;
                const float ek=row<active?merged[from+keys*dk+source_head*dk+d]:-73.f;
                require(std::memcmp(&q[to],&eq,sizeof(float))==0&&std::memcmp(&k[to],&ek,sizeof(float))==0,
                        "Independent merged modular QK mapping differs");
            }
            for(int d=0;d<dv;++d){const auto to=(std::size_t(row)*heads+h)*dv+d;
                const float ev=row<active?merged[from+2*keys*dk+h*dv+d]:-73.f;
                require(std::memcmp(&v[to],&ev,sizeof(float))==0,"Independent merged V mapping differs");
            }
        }

    }

    /**
     * @brief Authenticate exact two-role native DAG, geometry and retained argument identity.
     * @param prepared True for canonical producer/consumer; false for original raw pair.
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
            const std::string expected=rec?(prepared?"rocm_gdn_merged_recurrent_step_kernel<128>(":"rocm_gdn_recurrent_step_kernel<128>("):
                "rocm_gdn_deinterleave_qkv_kernel(";
            require(actual.find("(anonymous namespace)::"+expected)!=std::string::npos,"GDN pair wrong native family");
            require(!functions[rec]||functions[rec]==n.func,"GDN pair role function changes");functions[rec]=n.func;
            const unsigned bx=rec?128:256,gx=rec?heads:18,gy=rec?4:1,gz=1;
            // Original deinterleave linear grid scales with physical rows.
            require(n.blockDim.x==bx&&n.blockDim.y==1&&n.blockDim.z==1&&n.gridDim.x==(rec||prepared?gx:gx*unsigned(rows))&&
                n.gridDim.y==gy&&n.gridDim.z==gz&&n.sharedMemBytes==(rec?7184u:0u),"GDN pair wrong native geometry");
            require(n.kernelParams,"GDN pair missing captured arguments");
            const auto pointer=[&](int arg,const void* expected_pointer){require(*static_cast<void*const*>(n.kernelParams[arg])==expected_pointer,"GDN pair captured pointer differs");};
            const auto integer=[&](int arg,int expected_value){require(*static_cast<const int*>(n.kernelParams[arg])==expected_value,"GDN pair captured scalar differs");};
            const auto rowargs=[&](int arg){const auto& r=*static_cast<const llaminar2::DeviceRequestRowRanges*>(n.kernelParams[arg]);require(r.requests()==1&&r.rowsPerRequest()==rows&&r.lengthOwner()==live.ptr,"GDN pair captured row owner differs");};
            if(rec){
                pointer(0,prepared?dm.ptr:cq.ptr);pointer(1,prepared?nullptr:ck.ptr);pointer(2,prepared?nullptr:cv.ptr);
                pointer(3,da.ptr);pointer(4,db.ptr);pointer(5,dl.ptr);pointer(6,dt.ptr);
                pointer(7,prepared?po.ptr:co.ptr);pointer(8,di.ptr);pointer(9,prepared?ps.ptr:cs.ptr);
                integer(10,1);integer(11,rows);integer(12,heads);integer(13,dv);
                require(*static_cast<const bool*>(n.kernelParams[14]),"GDN merged norm policy differs");pointer(15,live.ptr);integer(16,0);
                pointer(17,rows==1?nullptr:(prepared?pshots.ptr:cshots.ptr));integer(18,state);integer(19,rows==1?0:rows);
                if(prepared){integer(20,keys);integer(21,offset);}
            }else{
                pointer(0,dm.ptr);pointer(1,cq.ptr);pointer(2,ck.ptr);pointer(3,cv.ptr);rowargs(4);
                integer(5,keys);integer(6,heads);integer(7,dk);integer(8,dv);integer(9,offset);
            }
            if(step<(prepared?1u:2u)){hipFuncAttributes a{};checkHip(hipFuncGetAttributes(&a,n.func),"GDN pair resources");require(a.localSizeBytes==0,"GDN pair private memory");
                std::cout<<"GDN_MERGED_CAPTURE,side="<<(prepared?"candidate":"control")<<",role="<<(rec?"recurrence":"preparation")
                    <<",M="<<rows<<",key_heads=4,heads=12,DK=128,DV=128,grid="<<n.gridDim.x<<'x'<<n.gridDim.y<<'x'<<n.gridDim.z
                    <<",block="<<bx<<",registers="<<a.numRegs<<",static_lds="<<a.sharedSizeBytes<<",dynamic_lds="<<n.sharedMemBytes
                    <<",private="<<a.localSizeBytes<<",symbol="<<gdn_paired_perf::escaped(actual)<<'\n';}
            index=next[index];
        }require(index==-1,"GDN pair cyclic chain");
    }
};

/**
 * @brief Run full/partial/empty dirty replay on one fixed cell before economics.
 * @param cell Borrowed complete native owners; every replay is joined before reset.
 */
inline void parity(Cell& cell)
{
    int seed=0;for(int live:{cell.rows,cell.rows>1?cell.rows-1:0,0,cell.rows}){
        cell.publish(live,seed++);cell.check(live);
    }
}
}

/** @test Exact complete-pair native results and original source protection.
 * @details This narrowly scoped private trial covers M1/M16 TP4, full/partial/
 * empty/large-to-small replay. It is not a production policy or admission gate.
 */
TEST(Test__ROCmGDNPaddedRealLength, MergedCompleteBoundaryTrialBytes)
{
    ASSERT_TRUE(hasROCm());checkHip(hipSetDevice(0),"GDN canonical device");
    for(int rows:{1,16})for(int offset:{0,5,-5}){
        SCOPED_TRACE(rows);gdn_merged_trial::Cell cell(rows,1,offset);
        const auto args=cell.arguments();
        EXPECT_FALSE(llaminar2::gdn::launchROCmMergedRecurrence(args,nullptr));
        EXPECT_FALSE(llaminar2::gdn::launchROCmMergedRecurrence(args,reinterpret_cast<void*>(std::uintptr_t{1})));
        EXPECT_FALSE(llaminar2::gdn::launchROCmMergedRecurrence(args,reinterpret_cast<void*>(std::uintptr_t{2})));
        auto missing=args;missing.merged=nullptr;
        EXPECT_FALSE(llaminar2::gdn::launchROCmMergedRecurrence(missing,cell.stream.stream));
        gdn_merged_trial::parity(cell);
    }
}

/** @test Disabled single-cell complete-pair event economics with exact native graph proof.
 * @details Explicit controls QFN_GDN177_M=1|16, QFN_GDN177_LIVE=0..M,
 * QFN_GDN177_REPEATS=1..256. Five warmups precede12 alternating paired samples.
 * All source publication, poison, byte checks and joins remain outside events.
 */
TEST(Test__ROCmGDNPaddedRealLength, DISABLED_MergedCompleteBoundaryEvents)
{
    ASSERT_TRUE(hasROCm());checkHip(hipSetDevice(0),"GDN canonical event device");
    const int m=gdn_paired_perf::integer("QFN_GDN177_M",1,1,16);
    const int live=gdn_paired_perf::integer("QFN_GDN177_LIVE",m,0,m);
    const int repeats=gdn_paired_perf::integer("QFN_GDN177_REPEATS",64,1,256);
    gdn_merged_trial::require(m==1||m==16,"Private complete-pair event M must be1 or16");
    const std::string profile_side=std::getenv("QFN_GDN177_PROFILE_SIDE")?std::getenv("QFN_GDN177_PROFILE_SIDE"):"";
    const std::string profile_role=std::getenv("QFN_GDN177_PROFILE_ROLE")?std::getenv("QFN_GDN177_PROFILE_ROLE"):"";
    gdn_merged_trial::require((profile_side.empty()&&profile_role.empty())||
        ((profile_side=="control"&&(profile_role=="preparation"||profile_role=="recurrence")||
          profile_side=="candidate"&&profile_role=="recurrence")&&repeats==1),
        "Profile requires existing explicit side/role and repeats1");
    // Events precede Cell so its terminal retirement also protects event teardown.
    gdn_paired_perf::Events events;
    gdn_merged_trial::Cell cell(m,repeats);
    if(!profile_side.empty()){
        // No parity replay executes here. One selected complete native graph,
        // five warmups and sixth occurrence only; other side is never launched.
        cell.publish(live,7);
        const auto& graph=profile_side=="control"?*cell.control:*cell.candidate;
        for(int i=0;i<5;++i)graph.launch(cell.stream.stream);
        checkHip(hipStreamSynchronize(cell.stream.stream),"GDN merged exact-role warmup terminal join");
        const auto marker=[&](const char* phase){
            std::cout<<"GDN_MERGED_SINGLE,phase="<<phase<<",side="<<profile_side<<",role="<<profile_role
                <<",M="<<m<<",live="<<live<<",parts=complete_boundary,repeats=1,warmups=5,selected_occurrence=6,graph_launches=6,kernel_dispatches="
                <<(profile_side=="control"?12:6)<<std::endl;
        };
        marker("armed");graph.launch(cell.stream.stream);
        checkHip(hipStreamSynchronize(cell.stream.stream),"GDN merged exact-role sixth transaction terminal join");
        marker("done");return;
    }
    gdn_merged_trial::parity(cell);
    cell.publish(live,7);cell.check(live);
    for(int i=0;i<5;++i){cell.control->launch(cell.stream.stream);cell.candidate->launch(cell.stream.stream);}
    checkHip(hipStreamSynchronize(cell.stream.stream),"GDN complete-pair warmup join");
    for(int i=0;i<12;++i){double control,candidate;
        if(i%2){candidate=events.measure(*cell.candidate,cell.stream.stream,repeats);control=events.measure(*cell.control,cell.stream.stream,repeats);}
        else{control=events.measure(*cell.control,cell.stream.stream,repeats);candidate=events.measure(*cell.candidate,cell.stream.stream,repeats);}
        std::cout<<"GDN_MERGED_PAIR,sample="<<i<<",M="<<m<<",live="<<live<<",key_heads=4,heads=12,DK=128,DV=128,repeats="<<repeats
            <<",warmups=5,control_us="<<control<<",candidate_us="<<candidate<<",unit=us_per_complete_boundary\n";
    }cell.check(live);
}

/** @test Direct merged launch rejects complete and partial source/output byte overlap.
 * @details No invalid native launch is submitted. Ordinary initial/updated state
 * alias is deliberately excluded from this source/output-only admission rule.
 */
TEST(Test__ROCmGDNPaddedRealLength, MergedSourceOutputOverlapRejectedBeforeLaunch)
{
    ASSERT_TRUE(hasROCm());checkHip(hipSetDevice(0),"GDN overlap admission device");
    gdn_merged_trial::Cell cell(1,1);const auto original=cell.arguments();
    auto overlap=original;overlap.output=cell.dm.ptr;
    EXPECT_FALSE(llaminar2::gdn::launchROCmMergedRecurrence(overlap,cell.stream.stream));
    overlap.output=cell.dm.ptr+1;
    EXPECT_FALSE(llaminar2::gdn::launchROCmMergedRecurrence(overlap,cell.stream.stream));
    overlap=original;overlap.merged=cell.po.ptr+1;
    EXPECT_FALSE(llaminar2::gdn::launchROCmMergedRecurrence(overlap,cell.stream.stream));
    overlap=original;overlap.merged=reinterpret_cast<const float*>(std::numeric_limits<std::uintptr_t>::max()-sizeof(float));
    EXPECT_FALSE(llaminar2::gdn::launchROCmMergedRecurrence(overlap,cell.stream.stream));
}
