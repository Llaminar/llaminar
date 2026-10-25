/**
 * @file ROCmGDNValueMajorProductionTest.h
 * @brief Captured production value-major programs versus the frozen native oracle.
 *
 * Reuses existing HIP storage/graph owners and the independent historical
 * recurrence. Original device rows select actual work; only host setup and
 * terminal assertions permute state words. No conversion or serial-row replay
 * is part of a production graph. Source, state and snapshot guard bytes remain
 * observable over dirty full/holes/empty/reset replay.
 */
#pragma once
#include "ROCmGDNHistoricalRecurrentOracle.h"
#include "ROCmGDNMergedExtendedTrial.h"
#include <array>
#include "kernels/gdn/GDNPlanarRecurrence.h"
#include "kernels/gdn/GDNMergedRecurrence.h"
#include "kernels/gdn/GDNPreparedQK.h"
#include "kernels/rocm/gdn/ROCmShortConvolution.h"
#include <cxxabi.h>

namespace
{
namespace gdn_value_major_production
{
using gdn_merged_extended::require;
/** @brief Immutable physical geometry and exact production source/publication policy. */
struct Geometry final
{
    int m,r,heads,dk,dv,keys,offset,snapshot_rows;
    bool normalize,merged,inplace,misaligned,null_counts;
};

/** @brief One production capture and an independent frozen native oracle. */
struct Cell final
{
    static constexpr int Guard=128;
    static constexpr float Poison=-83.f;
    const Geometry g;
    const int rows,state,width,stride,prefix;
    HipStreamHandle stream;
    HipFloatBuffer source,q,k,v,alpha,beta,log,dt,original,transposed,old_state,new_state,
        old_output,new_output,old_shots,new_shots;
    std::vector<int> lengths;
    HipIntBuffer live;
    llaminar2::DeviceRequestRowRanges ranges;
    std::vector<float> host_source,host_alpha,host_beta,host_state,host_transposed;
    std::unique_ptr<HipCapturedGraph> oracle,production;
    gdn_merged_extended::Cell::Retirement retirement;

    /**
     * @brief Capture the actual production format and an independent native reference.
     * @param geometry Immutable original row/head/source/alignment/snapshot policy.
     */
    explicit Cell(Geometry geometry)
        : g(geometry),rows(g.m*g.r),state(g.heads*g.dk*g.dv),width(2*g.keys*g.dk+g.heads*g.dv),
          stride(state+13),prefix(g.misaligned?1:0),source(std::size_t(rows)*width+prefix+Guard),
          q(std::size_t(rows)*g.heads*g.dk+prefix+Guard),k(q.count),v(std::size_t(rows)*g.heads*g.dv+prefix+Guard),
          alpha(std::size_t(rows)*g.heads),beta(alpha.count),log(std::vector<float>(g.heads,-.375f)),dt(std::vector<float>(g.heads,.125f)),
          original(std::size_t(g.r)*state+prefix+Guard),transposed(original.count),old_state(original.count),new_state(original.count),
          old_output(std::size_t(rows)*g.heads*g.dv+Guard),new_output(old_output.count),
          old_shots(std::size_t(g.snapshot_rows)*stride+Guard),new_shots(old_shots.count),
          lengths(g.r,g.m),live(lengths),
          ranges(g.null_counts?llaminar2::DeviceRequestRowRanges::fullyActiveRequests(g.r,g.m):
              llaminar2::DeviceRequestRowRanges::deviceCounted(g.r,g.m,live.ptr)),retirement{stream.stream}
    {
        require(g.m>0&&(g.dk==64||g.dk==128)&&g.dv>0&&g.heads>0&&g.keys>0,"Invalid production fixture geometry");
        require(!g.merged||(g.m<=16&&(g.m==1||g.snapshot_rows>=rows)),"Merged public entry retains original full-capture admission");
        require((reinterpret_cast<std::uintptr_t>(source.ptr)|reinterpret_cast<std::uintptr_t>(q.ptr)|
            reinterpret_cast<std::uintptr_t>(original.ptr)|reinterpret_cast<std::uintptr_t>(transposed.ptr))%16==0,"Fixture bases must authenticate vector/scalar offset geometry");
        publish(lengths,0);
        oracle=std::make_unique<HipCapturedGraph>(stream.stream,[&]
        {
            if(!prepare())return false;
            const gdn_historical_native::Arguments p{
                .q=q.ptr+prefix,.k=k.ptr+prefix,.v=v.ptr+prefix,.alpha=alpha.ptr,.beta=beta.ptr,.a_log=log.ptr,.dt_bias=dt.ptr,
                .output=old_output.ptr,.initial=original.ptr+prefix,.updated=g.inplace?original.ptr+prefix:old_state.ptr+prefix,
                .requests=g.r,.rows=g.m,.heads=g.heads,.key_width=g.dk,.value_width=g.dv,.normalize=g.normalize,
                .lengths=ranges.lengthOwner(),.row_offset=0,.snapshots=g.snapshot_rows?old_shots.ptr:nullptr,
                .snapshot_stride=stride,.snapshot_rows=g.snapshot_rows,.stream=stream.stream};
            return gdn_historical_native::launch(p);
        });
        production=std::make_unique<HipCapturedGraph>(stream.stream,[&]
        {
            if(g.merged)return llaminar2::gdn::launchROCmMergedRecurrence(mergedArguments(),stream.stream);
            if(!prepare())return false;
            return llaminar2::gdn::launchROCmPlanarRecurrence(planarArguments(),stream.stream);
        });
        inspect();
    }

    /** @brief Borrow original source mapping; this real native node owns expanded Q/K/V. */
    bool prepare()
    {
        return rocmGDN_deinterleave_qkv(source.ptr+prefix,q.ptr+prefix,k.ptr+prefix,v.ptr+prefix,ranges,
            g.keys,g.heads,g.dk,g.dv,g.offset,0,stream.stream);
    }

    /** @brief Return a complete canonical-layout planar production tuple. */
    llaminar2::gdn::PlanarRecurrenceInput planarArguments()
    {
        return {ranges,g.heads,g.dk,g.dv,g.normalize,q.ptr+prefix,k.ptr+prefix,v.ptr+prefix,alpha.ptr,beta.ptr,log.ptr,dt.ptr,
            new_output.ptr,transposed.ptr+prefix,g.inplace?transposed.ptr+prefix:new_state.ptr+prefix,
            g.snapshot_rows?new_shots.ptr:nullptr,stride,g.snapshot_rows,llaminar2::GDNRecurrenceStateLayout::ValueMajorKeyContiguous};
    }

    /** @brief Return the actual compact-source production tuple, never a private trial. */
    llaminar2::gdn::MergedRecurrenceInput mergedArguments()
    {
        return {ranges,g.keys,g.heads,g.dk,g.dv,g.offset,g.normalize,source.ptr+prefix,alpha.ptr,beta.ptr,log.ptr,dt.ptr,
            new_output.ptr,transposed.ptr+prefix,g.inplace?transposed.ptr+prefix:new_state.ptr+prefix,
            g.snapshot_rows?new_shots.ptr:nullptr,stride,g.snapshot_rows,llaminar2::GDNRecurrenceStateLayout::ValueMajorKeyContiguous};
    }

    /**
     * @brief Publish new original inputs/counts and reset existing banks before replay.
     * @param active Authored independent accepted prefix for each request.
     * @param seed Finite dirty source/state and gate-edge rotation.
     */
    void publish(const std::vector<int>& active,int seed)
    {
        require(active.size()==std::size_t(g.r),"Wrong original request count publication");
        lengths=active;for(int count:lengths)require(count>=0&&count<=g.m,"Invalid test prefix");
        if(g.null_counts)require(std::all_of(lengths.begin(),lengths.end(),[&](int n){return n==g.m;}),"Null count represents full rows only");
        host_source.assign(source.count,Poison);host_state.assign(original.count,Poison);host_transposed.assign(transposed.count,Poison);
        host_alpha.resize(alpha.count);host_beta.resize(beta.count);
        for(std::size_t i=0;i<std::size_t(rows)*width;++i)host_source[prefix+i]=float(int((i*7+seed*11)%31)-15)*.03125f;
        for(std::size_t i=0;i<std::size_t(g.r)*state;++i)host_state[prefix+i]=float(int((i+seed)%17)-8)*.00073f;
        if(seed%2)for(std::size_t i=0;i<std::size_t(g.r)*state;i+=19)host_state[prefix+i]=-0.f;
        const std::array<float,8> edges{-90.f,-20.f,-1.f,0.f,19.875f,20.f,20.125f,80.f};
        const std::array<float,8> betas{-100.f,-40.f,-1.f,-0.f,0.f,1.f,40.f,100.f};
        for(std::size_t i=0;i<host_alpha.size();++i){host_alpha[i]=edges[(i+seed)%8]-.125f;host_beta[i]=betas[(i+seed)%8];}
        for(int request=0;request<g.r;++request)for(int h=0;h<g.heads;++h)for(int key=0;key<g.dk;++key)for(int value=0;value<g.dv;++value)
        {
            const auto from=std::size_t(request)*state+(h*g.dk+key)*g.dv+value;
            const auto to=std::size_t(request)*state+(h*g.dv+value)*g.dk+key;
            std::memcpy(&host_transposed[prefix+to],&host_state[prefix+from],sizeof(float));
        }
        source.copyFrom(host_source);alpha.copyFrom(host_alpha);beta.copyFrom(host_beta);
        original.copyFrom(host_state);transposed.copyFrom(host_transposed);
        for(auto* b:{&q,&k,&v,&old_state,&new_state,&old_output,&new_output,&old_shots,&new_shots})b->fill(Poison);
        checkHip(hipMemcpyAsync(live.ptr,lengths.data(),lengths.size()*sizeof(int),hipMemcpyHostToDevice,stream.stream),"Production GDN prefix publication");
        checkHip(hipStreamSynchronize(stream.stream),"Production GDN test setup join");
    }

    /** @brief Run both actual graphs and compare every logical state/output/guard byte. */
    void check()
    {
        oracle->launch(stream.stream);checkHip(hipStreamSynchronize(stream.stream),"Historical oracle observation");
        const auto golden_output=old_output.toHost(),golden_state=(g.inplace?original:old_state).toHost(),golden_shots=old_shots.toHost();
        const auto expanded_q=q.toHost(),expanded_k=k.toHost(),expanded_v=v.toHost();
        production->launch(stream.stream);checkHip(hipStreamSynchronize(stream.stream),"Production value-major terminal observation");
        const auto output=new_output.toHost(),native=(g.inplace?transposed:new_state).toHost(),shots=new_shots.toHost();
        const auto oldout=old_output.toHost(),oldstate=(g.inplace?original:old_state).toHost(),oldshots=old_shots.toHost();
        require(std::memcmp(oldout.data(),golden_output.data(),oldout.size()*sizeof(float))==0&&
            std::memcmp(oldstate.data(),golden_state.data(),oldstate.size()*sizeof(float))==0&&
            std::memcmp(oldshots.data(),golden_shots.data(),oldshots.size()*sizeof(float))==0,"Production changed independent historical checkpoints");
        require(std::memcmp(output.data(),golden_output.data(),output.size()*sizeof(float))==0,"Production value-major activation bytes differ");
        const auto compare=[&](const float* from,const float* expected)
        {
            for(int h=0;h<g.heads;++h)for(int key=0;key<g.dk;++key)for(int value=0;value<g.dv;++value)
                require(std::memcmp(from+(h*g.dv+value)*g.dk+key,expected+(h*g.dk+key)*g.dv+value,sizeof(float))==0,
                    "Production value-major state/snapshot word differs");
        };
        for(int request=0;request<g.r;++request)compare(native.data()+prefix+std::size_t(request)*state,golden_state.data()+prefix+std::size_t(request)*state);
        for(int row=0;row<g.snapshot_rows;++row)
        {
            const auto base=std::size_t(row)*stride;compare(shots.data()+base,golden_shots.data()+base);
            for(int i=state;i<stride;++i)require(shots[base+i]==Poison&&golden_shots[base+i]==Poison,"Production snapshot padding changed");
        }
        const auto guard=[&](const std::vector<float>& a,std::size_t first,std::size_t end)
        {for(std::size_t i=first;i<end;++i)require(a[i]==Poison,"Production physical guard changed");};
        guard(native,0,prefix);guard(native,prefix+std::size_t(g.r)*state,native.size());
        guard(output,std::size_t(rows)*g.heads*g.dv,output.size());guard(shots,std::size_t(g.snapshot_rows)*stride,shots.size());
        for(auto pair:std::array<std::pair<HipFloatBuffer*,const std::vector<float>*>,6>{{{&q,&expanded_q},{&k,&expanded_k},{&v,&expanded_v},{&source,&host_source},{&alpha,&host_alpha},{&beta,&host_beta}}})
        {
            const auto observed=pair.first->toHost();require(std::memcmp(observed.data(),pair.second->data(),observed.size()*sizeof(float))==0,"Production changed original source bytes");
        }
        for(auto pair:{std::pair{&log,-.375f},std::pair{&dt,.125f}})
            for(float value:pair.first->toHost())require(std::memcmp(&value,&pair.second,sizeof(float))==0,"Production changed learned gates");
        std::vector<int> count(g.r);
        checkHip(hipMemcpyAsync(count.data(),live.ptr,count.size()*sizeof(int),hipMemcpyDeviceToHost,stream.stream),"Production count terminal observation");
        checkHip(hipStreamSynchronize(stream.stream),"Production count terminal join");require(count==lengths,"Production changed original live words");
        if(g.inplace)
        {
            for(const auto* bank:{&old_state,&new_state})for(float value:bank->toHost())require(value==Poison,"Production touched unused separate terminal bank");
        }
        if(!g.inplace)
        {
            const auto a=original.toHost(),b=transposed.toHost();
            require(std::memcmp(a.data(),host_state.data(),a.size()*sizeof(float))==0&&
                std::memcmp(b.data(),host_transposed.data(),b.size()*sizeof(float))==0,"Production changed immutable initial banks");
        }
    }

    /** @brief Authenticate production source mapping, exact native family/geometry and all arguments. */
    void inspect();
};

inline void Cell::inspect()
{
    const auto check_graph=[&](const HipCapturedGraph& graph,bool candidate)
    {
        const bool has_prep=!candidate||!g.merged;
        std::size_t count=0;checkHip(hipGraphGetNodes(graph.graph,nullptr,&count),"Production GDN native count");
        require(count==std::size_t(has_prep?2:1),"Production GDN has hidden/absent native work");
        std::vector<hipGraphNode_t> nodes(count);checkHip(hipGraphGetNodes(graph.graph,nodes.data(),&count),"Production GDN nodes");
        std::vector<int> next(count,-1);int root=-1;
        for(std::size_t i=0;i<count;++i)
        {
            hipGraphNodeType type{};checkHip(hipGraphNodeGetType(nodes[i],&type),"Production GDN node type");
            require(type==hipGraphNodeTypeKernel,"Production GDN contains a conversion or foreign node");
            std::size_t n=0;checkHip(hipGraphNodeGetDependencies(nodes[i],nullptr,&n),"Production GDN parent count");
            require(n<=1,"Production GDN has multiple parents");
            if(!n){require(root<0,"Production GDN has multiple roots");root=int(i);}
            else
            {
                hipGraphNode_t parent{};checkHip(hipGraphNodeGetDependencies(nodes[i],&parent,&n),"Production GDN parent");
                const auto at=std::find(nodes.begin(),nodes.end(),parent);require(at!=nodes.end(),"Production GDN has foreign parent");
                const auto index=std::size_t(at-nodes.begin());require(next[index]<0,"Production GDN branches");next[index]=int(i);
            }
        }
        int at=root;
        for(std::size_t step=0;step<count;++step)
        {
            require(at>=0&&std::size_t(at)<count,"Production GDN chain truncated");
            hipKernelNodeParams node{};checkHip(hipGraphKernelNodeGetParams(nodes[at],&node),"Production GDN native tuple");
            const char* raw=hipKernelNameRefByPtr(node.func,stream.stream);require(raw,"Production GDN native name missing");
            int status=-1;std::unique_ptr<char,decltype(&std::free)> name(abi::__cxa_demangle(raw,nullptr,nullptr,&status),&std::free);
            require(status==0&&bool(name),"Production GDN native demangle failed");const std::string actual(name.get());
            require(node.kernelParams&&!node.extra,"Production GDN native argument authority missing");
            const auto pointer=[&](int i,const void* expected)
            {
                const void* got=nullptr;std::memcpy(&got,node.kernelParams[i],sizeof(got));require(got==expected,"Production GDN pointer mismatch");
            };
            const auto integer=[&](int i,int expected)
            {
                int got=0;std::memcpy(&got,node.kernelParams[i],sizeof(got));require(got==expected,"Production GDN scalar mismatch");
            };
            const auto boolean=[&](int i,bool expected)
            {
                bool got=false;std::memcpy(&got,node.kernelParams[i],sizeof(got));require(got==expected,"Production GDN policy mismatch");
            };
            const bool preparation=has_prep&&step==0;
            unsigned gx,gy,bx;std::size_t shared;
            if(preparation)
            {
                require(actual.find("rocm_gdn_deinterleave_qkv_kernel(")!=std::string::npos,"Production preparation is not native original mapping");
                gx=unsigned((g.m*(2*g.heads*g.dk+g.heads*g.dv)+255)/256);gy=g.r;bx=256;shared=0;
                pointer(0,source.ptr+prefix);pointer(1,q.ptr+prefix);pointer(2,k.ptr+prefix);pointer(3,v.ptr+prefix);
                auto observed=ranges;std::memcpy(&observed,node.kernelParams[4],sizeof(observed));
                require(observed.requests()==g.r&&observed.rowsPerRequest()==g.m&&observed.lengthOwner()==ranges.lengthOwner(),"Production mapping lost original request authority");
                integer(5,g.keys);integer(6,g.heads);integer(7,g.dk);integer(8,g.dv);integer(9,g.offset);
            }
            else
            {
                const bool complete=g.snapshot_rows>=rows&&g.snapshot_rows>0;
                const bool grouped=g.merged||g.m<=8||complete;
                const bool scalar=candidate&&grouped&&g.m==1;
                const bool cached=!grouped&&g.dk==128&&g.dv==128&&g.m<=2048;
                gx=g.r*g.heads;gy=(g.dv+(candidate&&grouped?31:63))/(candidate&&grouped?32:64);
                bx=candidate?(scalar?512:grouped?128:g.dv<=64?128:256):256;
                if(candidate&&!grouped){gx=g.heads;gy=(g.dv+(bx/4)-1)/(bx/4);}
                const int tail=g.dk/4>22?g.dk/4-22:0;
                shared=candidate?(scalar?0:grouped?std::size_t(2*g.dk+4+256+tail*128)*4:
                    (std::size_t(cached?16*(2*g.dk+int(bx)/4):2*g.dk)+16+((cached?0:2)+(g.dk==128&&g.dv==128&&!cached?5:0))*bx+
                        std::size_t(cached?4:2)*g.m)*4):std::size_t(2*g.dk+4+512+tail*256)*4;
                if(!candidate)
                    require(actual.find("gdn_historical_native::rocm_gdn_recurrent_step_kernel<"+std::to_string(g.dk)+">(")!=std::string::npos,"Oracle is not frozen native arithmetic");
                else
                {
                    const std::string family=scalar?"rocm_gdn_value_major_scalar_kernel<":grouped?"rocm_gdn_value_major_recurrent_step_kernel<":"rocm_gdn_value_major_chunk_forward_kernel<";
                    require(actual.find("(anonymous namespace)::"+family+std::to_string(g.dk)+",")!=std::string::npos,"Actual production state-format program differs");
                    if(grouped)require(actual.find(std::string("GDNRecurrentInputLayout)")+(g.merged?"1":"0"))!=std::string::npos,"Production input coordinates differ");
                    if(scalar||!grouped)
                    {
                        const bool scalar_access=g.misaligned||(scalar&&g.merged&&g.r>1&&width%4!=0);
                        require(actual.find(std::string("GdnRecurrenceEndpointAccess)")+(scalar_access?"1":"0"))!=std::string::npos,"Production endpoint alignment program differs");
                    }
                    if(!grouped)require(actual.find(std::string("GdnPrefillSnapshots)")+(cached&&!g.snapshot_rows?"0":"1"))!=std::string::npos,"Production long snapshot policy differs");
                }
                const float* init=candidate?transposed.ptr+prefix:original.ptr+prefix;
                float* updated=candidate?(g.inplace?transposed.ptr+prefix:new_state.ptr+prefix):(g.inplace?original.ptr+prefix:old_state.ptr+prefix);
                pointer(0,candidate&&g.merged?source.ptr+prefix:q.ptr+prefix);
                pointer(1,candidate&&g.merged?nullptr:k.ptr+prefix);pointer(2,candidate&&g.merged?nullptr:v.ptr+prefix);
                pointer(3,alpha.ptr);pointer(4,beta.ptr);pointer(5,log.ptr);pointer(6,dt.ptr);
                pointer(7,candidate?new_output.ptr:old_output.ptr);pointer(8,init);pointer(9,updated);
                float* shots=g.snapshot_rows?(candidate?new_shots.ptr:old_shots.ptr):nullptr;
                if(candidate&&!grouped)
                {
                    pointer(10,ranges.lengthOwner());pointer(11,shots);integer(12,stride);integer(13,g.snapshot_rows);
                    integer(14,g.r);integer(15,g.m);integer(16,g.heads);integer(17,g.dv);boolean(18,g.normalize);
                }
                else
                {
                    integer(10,g.r);integer(11,g.m);integer(12,g.heads);integer(13,g.dv);boolean(14,g.normalize);
                    pointer(15,ranges.lengthOwner());integer(16,0);pointer(17,shots);integer(18,stride);integer(19,g.snapshot_rows);
                    if(candidate){integer(20,g.merged?g.keys:0);integer(21,g.merged?g.offset:0);}
                }
            }
            require(node.gridDim.x==gx&&node.gridDim.y==gy&&node.gridDim.z==unsigned(candidate&&!preparation&&!(g.merged||g.m<=8||(g.snapshot_rows>=rows&&g.snapshot_rows>0))?g.r:1)&&
                node.blockDim.x==bx&&node.blockDim.y==1&&node.blockDim.z==1&&node.sharedMemBytes==shared,"Production native geometry/LDS mismatch");
            hipFuncAttributes resources{};checkHip(hipFuncGetAttributes(&resources,node.func),"Production native resources");
            require(resources.localSizeBytes==0,"Production native owns private memory");
            std::cout<<"GDN_VALUE_MAJOR_PRODUCTION_CAPTURE,side="<<(candidate?"candidate":"historical")<<",role="<<(preparation?"preparation":"recurrence")
                <<",M="<<g.m<<",R="<<g.r<<",heads="<<g.heads<<",DK="<<g.dk<<",DV="<<g.dv<<",merged="<<g.merged
                <<",normalize="<<g.normalize<<",misaligned="<<g.misaligned<<",retained="<<g.snapshot_rows<<",null_counts="<<g.null_counts
                <<",grid="<<gx<<'x'<<gy<<'x'<<node.gridDim.z<<",block="<<bx<<",dynamic_lds="<<shared<<",static_lds="<<resources.sharedSizeBytes
                <<",registers="<<resources.numRegs<<",private="<<resources.localSizeBytes<<",symbol="<<gdn_paired_perf::escaped(actual)<<'\n';
            at=next[at];
        }
        require(at<0,"Production GDN graph cycle");
    };
    check_graph(*oracle,false);check_graph(*production,true);
}

/**
 * @brief Run four dirty original-request publications through the same retained graphs.
 * @param geometry Exact immutable production geometry/policy.
 */
inline void run(Geometry geometry)
{
    Cell cell(geometry);
    const std::vector<int> full(geometry.r,geometry.m),empty(geometry.r,0);
    auto holes=full;for(int request=0;request<geometry.r;++request)holes[request]=geometry.r==1?geometry.m/2:request%3==0?geometry.m:request%3==1?0:std::min(geometry.m,1);
    int seed=0;
    for(const auto& active:{full,geometry.null_counts?full:holes,geometry.null_counts?full:empty,full}){cell.publish(active,seed++);cell.check();}
    std::cout<<"GDN_VALUE_MAJOR_PRODUCTION_CASE,M="<<geometry.m<<",R="<<geometry.r<<",heads="<<geometry.heads<<",DK="<<geometry.dk<<",DV="<<geometry.dv
        <<",keys="<<geometry.keys<<",offset="<<geometry.offset<<",merged="<<geometry.merged<<",normalize="<<geometry.normalize
        <<",inplace="<<geometry.inplace<<",misaligned="<<geometry.misaligned<<",null_counts="<<geometry.null_counts<<",retained="<<geometry.snapshot_rows
        <<",states=4,passed=1\n";
}
} // namespace gdn_value_major_production
} // anonymous namespace

/** @test Every grouped M1..16 uses actual production value-major programs with the frozen oracle. */
TEST(Test__ROCmGDNPaddedRealLength, ValueMajorProductionGroupedAndAlignmentBytes)
{
    using namespace gdn_value_major_production;
    ASSERT_TRUE(hasROCm());checkHip(hipSetDevice(0),"Production value-major grouped device");
    const std::array<int,6> heads{1,3,6,12,24,48};
    const std::array<int,5> offsets{0,5,-5,std::numeric_limits<int>::min(),std::numeric_limits<int>::max()};
    int cases=0;
    for(int m=1;m<=16;++m)for(int dk:{64,128})for(int dv:{1,65,128,129})for(bool norm:{false,true})
    {
        const int r=m%4==0?3:1;const int h=m==1?heads[cases%heads.size()]:12;
        SCOPED_TRACE(::testing::Message()<<"M="<<m<<" DK="<<dk<<" DV="<<dv<<" norm="<<norm);
        run({m,r,h,dk,dv,4,offsets[cases%5],r*m+4,norm,(m+dk/64+dv)%2==0,cases%3==0,cases%4==0,false});++cases;
    }
    // Real TP shard degrees, null counts, terminal-only state and non-four-aligned
    // compact request widths supplement the complete snapshot grouped matrix.
    for(int h:{1,3,5,6,7,8,9,10,12,16,24,48})for(int dk:{64,128})for(bool norm:{false,true})
    {
        const int dv=cases%3==0?1:cases%3==1?65:129;
        run({1,3,h,dk,dv,4,offsets[cases%5],0,norm,cases%2==0,cases%3==0,cases%4==0,cases%2==1});++cases;
    }
    ASSERT_EQ(cases,304);
    std::cout<<"GDN_VALUE_MAJOR_PRODUCTION_GROUPED_SUMMARY,cases=304,states=1216,replays=2432,passed=1\n";
}

/** @test Cached and generic resident-prefill routes preserve original history and snapshot bytes. */
TEST(Test__ROCmGDNPaddedRealLength, ValueMajorProductionLongRouteBoundariesBytes)
{
    using namespace gdn_value_major_production;
    ASSERT_TRUE(hasROCm());checkHip(hipSetDevice(0),"Production value-major prefill device");
    struct Shape{int m,h,dk,dv,r;};
    const std::array<Shape,12> shapes{{{9,3,64,17,1},{16,3,128,65,3},{31,3,64,128,1},{64,3,128,129,3},
        {512,12,128,128,1},{1024,3,128,128,1},{2048,3,128,128,1},{2049,3,128,128,3},
        {4096,3,128,128,1},{512,3,64,1,3},{64,3,128,64,1},{31,12,128,128,3}}};
    int cases=0;
    for(const auto& s:shapes)for(bool norm:{false,true})for(bool snapshots:{false,true})
    {
        SCOPED_TRACE(::testing::Message()<<"M="<<s.m<<" DK="<<s.dk<<" DV="<<s.dv<<" norm="<<norm);
        run({s.m,s.r,s.h,s.dk,s.dv,4,cases%2?5:-5,snapshots?std::min(17,s.m-1):0,norm,false,cases%3==0,cases%4==0,false});++cases;
    }
    // Public float-only-aligned state banks with bounded cached-normalization
    // snapshots are a distinct supported representation, independent of timing.
    for(bool norm:{false,true})
    {
        run({512,3,3,128,128,4,norm?5:-5,17,norm,false,false,true,false});++cases;
    }
    ASSERT_EQ(cases,50);
    std::cout<<"GDN_VALUE_MAJOR_PRODUCTION_LONG_SUMMARY,cases=50,states=200,replays=400,passed=1\n";
}

/**
 * @brief Reuse the real owner restore transaction for raw and producer-prepared inputs.
 * @param prepared Select the explicit producer/consumer binding, never a serving mode.
 * @details Each retained graph survives all dirty state resets. The prepared
 * branch adds original convolution to the independent oracle and uses the real
 * convolution/recurrence owners with their original speculative banks.
 */
inline void proveValueMajorOwnerRestore(bool prepared)
{
    using Layout=llaminar2::GDNRecurrenceStateLayout;
    constexpr int Heads=3,Keys=4,DV=17,Guard=128;
    constexpr float Poison=-91.f;
    ASSERT_TRUE(hasROCm());checkHip(hipSetDevice(0),"Value-major owner restore device");
    for(int dk:{64,128})for(int offset:{0,5,-5})for(bool norm:{false,true})
    {
        SCOPED_TRACE(::testing::Message()<<"DK="<<dk<<" offset="<<offset<<" normalize="<<norm);
        const int state=Heads*dk*DV,width=2*Keys*dk+Heads*DV,conv_state=3*width;
        HipStreamHandle stream;
        HipFloatBuffer raw(2*width),alpha(2*Heads),beta(2*Heads),log(std::vector<float>(Heads,-.375f)),dt(std::vector<float>(Heads,.125f));
        HipFloatBuffer q(2*Heads*dk),k(q.count),v(2*Heads*DV),reference_state(state),reference_output(2*Heads*DV+Guard),output(reference_output.count);
        HipFloatBuffer snapshots(2*state+Guard),work(state+Guard),wrong_snapshots(snapshots.count),wrong_work(work.count);
        HipFloatBuffer postconv(prepared?2*width+Guard:0),reference_raw(postconv.count),
            weights(prepared?4*width:0),bias(prepared?width:0),reference_conv(prepared?conv_state:0),
            conv_snapshots(prepared?2*conv_state+Guard:0),conv_work(prepared?conv_state+Guard:0),
            compact(prepared?gdn::ScalarPreparedQKLayout::fromHeads(Keys,Heads,dk,DV).elements+Guard:0);
        ROCmGatedDeltaNet kernel(0);
        HipGDNStateOwner owner(kernel,state,0,3,Layout::ValueMajorKeyContiguous);
        kernel.setGPUStream(stream.stream);
        ROCmShortConvolution convolution(0);
        std::unique_ptr<HipGDNStateOwner> conv_owner;
        if(prepared)
        {
            conv_owner=std::make_unique<HipGDNStateOwner>(convolution,conv_state);
            convolution.setGPUStream(stream.stream);
            convolution.bindVerifierStateCaptureWorkspace(conv_snapshots.ptr,2,conv_state);
            convolution.bindSpeculativeStateWorkspace(conv_work.ptr,conv_state);
        }
        HipIntBuffer accepted_row(0);
        std::unique_ptr<HipCapturedGraph> verifier,ordinary,reference_first,reference_next,restore;
        gdn_merged_extended::Cell::Retirement retirement{stream.stream};
        ASSERT_EQ(kernel.recurrenceStateLayout(),Layout::ValueMajorKeyContiguous);
        kernel.bindRecurrenceStateCaptureWorkspace(snapshots.ptr,2,state,Layout::ValueMajorKeyContiguous);
        kernel.bindRecurrenceSpeculativeWorkspace(work.ptr,state,Layout::ValueMajorKeyContiguous);
        EXPECT_THROW(kernel.bindRecurrenceStateCaptureWorkspace(wrong_snapshots.ptr,2,state,Layout::KeyMajorValueContiguous),std::invalid_argument);
        EXPECT_THROW(kernel.bindRecurrenceStateCaptureWorkspace(wrong_snapshots.ptr,2,state,static_cast<Layout>(255)),std::invalid_argument);
        EXPECT_THROW(kernel.bindRecurrenceSpeculativeWorkspace(wrong_work.ptr,state,Layout::KeyMajorValueContiguous),std::invalid_argument);
        const auto merged=[&](int row)
        {
            if(prepared)
            {
                if(!convolution.forwardPreparedQK({Keys,Heads,dk,DV,norm,raw.ptr+row*width,
                    weights.ptr,bias.ptr,postconv.ptr+row*width,compact.ptr}))return false;
                return kernel.chunkForwardPreparedQK({Keys,Heads,dk,DV,offset,norm,
                    postconv.ptr+row*width,compact.ptr,alpha.ptr+row*Heads,beta.ptr+row*Heads,
                    log.ptr,dt.ptr,output.ptr+row*Heads*DV});
            }
            return kernel.chunkForwardMergedQKV(raw.ptr+row*width,width,alpha.ptr+row*Heads,beta.ptr+row*Heads,
                log.ptr,dt.ptr,output.ptr+row*Heads*DV,nullptr,1,Keys,Heads,dk,DV,offset,64,norm);
        };
        verifier=std::make_unique<HipCapturedGraph>(stream.stream,[&]{return merged(0);});
        kernel.bindRecurrenceStateCaptureWorkspace(nullptr,0,state,Layout::ValueMajorKeyContiguous);
        kernel.bindRecurrenceSpeculativeWorkspace(nullptr,0,Layout::ValueMajorKeyContiguous);
        if(prepared)
        {
            convolution.bindVerifierStateCaptureWorkspace(nullptr,0,conv_state);
            convolution.bindSpeculativeStateWorkspace(nullptr,conv_state);
        }
        ordinary=std::make_unique<HipCapturedGraph>(stream.stream,[&]{return merged(1);});
        kernel.bindRecurrenceStateCaptureWorkspace(snapshots.ptr,2,state,Layout::ValueMajorKeyContiguous);
        kernel.bindRecurrenceSpeculativeWorkspace(work.ptr,state,Layout::ValueMajorKeyContiguous);
        if(prepared)
        {
            convolution.bindVerifierStateCaptureWorkspace(conv_snapshots.ptr,2,conv_state);
            convolution.bindSpeculativeStateWorkspace(conv_work.ptr,conv_state);
            restore=std::make_unique<HipCapturedGraph>(stream.stream,[&]
            {
                return convolution.restoreVerifierStateCaptureRowFromDeviceIndex(nullptr,accepted_row.ptr,stream.stream)&&
                    kernel.restoreVerifierStateCaptureRowFromDeviceIndex(nullptr,accepted_row.ptr,stream.stream);
            });
        }
        const auto reference=[&](int row)
        {
            if(prepared&&!rocmGDN_short_conv1d_effective(raw.ptr+row*width,weights.ptr,bias.ptr,
                reference_raw.ptr+row*width,reference_conv.ptr,reference_conv.ptr,1,width,4,true,
                nullptr,nullptr,conv_state,0,0,stream.stream))return false;
            if(!rocmGDN_deinterleave_qkv(prepared?reference_raw.ptr+row*width:raw.ptr+row*width,q.ptr+row*Heads*dk,k.ptr+row*Heads*dk,v.ptr+row*Heads*DV,
                DeviceRequestRowRanges::fullyActive(1),Keys,Heads,dk,DV,offset,0,stream.stream))return false;
            return gdn_historical_native::launch({q.ptr+row*Heads*dk,k.ptr+row*Heads*dk,v.ptr+row*Heads*DV,
                alpha.ptr+row*Heads,beta.ptr+row*Heads,log.ptr,dt.ptr,reference_output.ptr+row*Heads*DV,
                reference_state.ptr,reference_state.ptr,1,1,Heads,dk,DV,norm,nullptr,0,nullptr,state,0,stream.stream});
        };
        reference_first=std::make_unique<HipCapturedGraph>(stream.stream,[&]{return reference(0);});
        reference_next=std::make_unique<HipCapturedGraph>(stream.stream,[&]{return reference(1);});
        // Authenticate the real owner-selected workspace/live-bank bindings.
        const void* conv_live_bank=nullptr;
        const auto inspect=[&](const HipCapturedGraph& graph,bool capture,const void* live_bank)
        {
            std::size_t count=0;checkHip(hipGraphGetNodes(graph.graph,nullptr,&count),"Owner production node count");
            gdn_merged_extended::require(count==std::size_t(prepared?2:1),"Owner graph has hidden or missing work");
            std::vector<hipGraphNode_t> nodes(count);checkHip(hipGraphGetNodes(graph.graph,nodes.data(),&count),"Owner production nodes");
            hipGraphNode_t handle{},conv_node{};
            for(const auto node:nodes)
            {
                hipGraphNodeType type{};checkHip(hipGraphNodeGetType(node,&type),"Owner production node type");
                gdn_merged_extended::require(type==hipGraphNodeTypeKernel,"Owner graph contains foreign work");
                std::size_t parents=0;checkHip(hipGraphNodeGetDependencies(node,nullptr,&parents),"Owner parent count");
                if(prepared&&!parents)conv_node=node;else handle=node;
            }
            gdn_merged_extended::require(handle!=nullptr,"Owner recurrence node is missing");
            count=0;checkHip(hipGraphNodeGetDependencies(handle,nullptr,&count),"Owner production parent count");
            gdn_merged_extended::require(count==std::size_t(prepared?1:0),"Owner recurrence parent differs");
            if(prepared)
            {
                hipGraphNode_t parent{};checkHip(hipGraphNodeGetDependencies(handle,&parent,&count),"Owner convolution edge");
                gdn_merged_extended::require(parent==conv_node,"Owner producer/consumer edge differs");
                hipKernelNodeParams conv{};checkHip(hipGraphKernelNodeGetParams(conv_node,&conv),"Owner convolution tuple");
                const char* raw_name=hipKernelNameRefByPtr(conv.func,stream.stream);
                gdn_merged_extended::require(raw_name!=nullptr,"Owner convolution function unresolved");
                int status=-1;std::unique_ptr<char,decltype(&std::free)> name(abi::__cxa_demangle(raw_name,nullptr,nullptr,&status),&std::free);
                gdn_merged_extended::require(status==0&&bool(name)&&std::string(name.get()).find(
                    "rocm_gdn_scalar_prepared_qk_conv_kernel<"+std::to_string(dk)+">(")!=std::string::npos,"Owner convolution is not the production prepared program");
                gdn_merged_extended::require(conv.kernelParams&&!conv.extra,"Owner convolution tuple missing");
                const auto pointer=[&](int i,const void* expected){const void* got=nullptr;std::memcpy(&got,conv.kernelParams[i],sizeof(got));gdn_merged_extended::require(got==expected,"Owner convolution pointer differs");};
                const auto integer=[&](int i,int expected){int got=0;std::memcpy(&got,conv.kernelParams[i],sizeof(got));gdn_merged_extended::require(got==expected,"Owner convolution scalar differs");};
                const void* initial=nullptr;std::memcpy(&initial,conv.kernelParams[3],sizeof(initial));
                gdn_merged_extended::require(initial!=nullptr,"Owner convolution live bank missing");
                if(!conv_live_bank)conv_live_bank=initial;
                pointer(0,raw.ptr+(capture?0:1)*width);pointer(1,weights.ptr);pointer(2,bias.ptr);pointer(3,conv_live_bank);
                pointer(4,postconv.ptr+(capture?0:1)*width);pointer(5,capture?conv_work.ptr:conv_live_bank);
                pointer(6,capture?conv_snapshots.ptr:nullptr);pointer(7,compact.ptr);pointer(8,compact.ptr+Keys*dk);
                integer(9,1);integer(10,Keys);integer(11,Heads);integer(12,DV);pointer(13,nullptr);integer(14,conv_state);integer(15,capture?2:0);
                bool normalization=false;std::memcpy(&normalization,conv.kernelParams[16],sizeof(normalization));
                gdn_merged_extended::require(normalization==norm,"Owner convolution normalization differs");
                gdn_merged_extended::require(conv.blockDim.x==128&&conv.blockDim.y==1&&conv.blockDim.z==1&&
                    conv.gridDim.x==unsigned((width-1)/dk+1)&&conv.gridDim.y==1&&conv.gridDim.z==1&&conv.sharedMemBytes==0,"Owner convolution geometry differs");
                hipFuncAttributes attr{};checkHip(hipFuncGetAttributes(&attr,conv.func),"Owner convolution resources");
                gdn_merged_extended::require(attr.localSizeBytes==0&&attr.sharedSizeBytes==16,"Owner convolution has unexpected private/LDS storage");
            }
            hipKernelNodeParams node{};checkHip(hipGraphKernelNodeGetParams(handle,&node),"Owner production arguments");
            gdn_merged_extended::require(node.kernelParams&&!node.extra,"Owner native tuple missing");
            const char* name=hipKernelNameRefByPtr(node.func,stream.stream);gdn_merged_extended::require(name!=nullptr,"Owner native function unresolved");
            gdn_merged_extended::require(std::string(name).find("rocm_gdn_value_major_scalar_kernelILi"+std::to_string(dk))!=std::string::npos,"Owner native family/policy differs");
            gdn_merged_extended::require(std::string(name).find(std::string("GDNRecurrentInputLayoutE")+(prepared?"2E":"1E"))!=std::string::npos,"Owner native family/policy differs");
            gdn_merged_extended::require(std::string(name).find(std::string("GdnRecurrenceEndpointAccessE")+(prepared||capture?"0E":"1E"))!=std::string::npos,"Owner native family/policy differs");
            const auto pointer=[&](int i,const void* expected){const void* got=nullptr;std::memcpy(&got,node.kernelParams[i],sizeof(got));gdn_merged_extended::require(got==expected,"Owner native captured argument differs");};
            const auto integer=[&](int i,int expected){int got=0;std::memcpy(&got,node.kernelParams[i],sizeof(got));gdn_merged_extended::require(got==expected,"Owner native captured argument differs");};
            const int row=capture?0:1;
            pointer(0,prepared?postconv.ptr+row*width:raw.ptr+row*width);
            pointer(1,prepared?compact.ptr:nullptr);pointer(2,prepared?compact.ptr+Keys*dk:nullptr);
            pointer(3,alpha.ptr+row*Heads);pointer(4,beta.ptr+row*Heads);
            pointer(5,log.ptr);pointer(6,dt.ptr);pointer(7,output.ptr+row*Heads*DV);
            if(live_bank)pointer(8,live_bank);
            pointer(9,capture?static_cast<const void*>(work.ptr):live_bank);
            integer(10,1);integer(11,1);integer(12,Heads);integer(13,DV);
            bool normalization=false;std::memcpy(&normalization,node.kernelParams[14],sizeof(normalization));gdn_merged_extended::require(normalization==norm,"Owner native normalization differs");
            pointer(15,nullptr);integer(16,0);pointer(17,capture?snapshots.ptr:nullptr);integer(18,state);integer(19,capture?2:0);
            integer(20,Keys);integer(21,offset);
            gdn_merged_extended::require(node.gridDim.x==Heads,"Owner native geometry/resource differs");gdn_merged_extended::require(node.gridDim.y==1u,"Owner native geometry/resource differs");gdn_merged_extended::require(node.gridDim.z==1u,"Owner native geometry/resource differs");
            gdn_merged_extended::require(node.blockDim.x==512u,"Owner native geometry/resource differs");gdn_merged_extended::require(node.blockDim.y==1u,"Owner native geometry/resource differs");gdn_merged_extended::require(node.blockDim.z==1u,"Owner native geometry/resource differs");gdn_merged_extended::require(node.sharedMemBytes==0u,"Owner native geometry/resource differs");
            hipFuncAttributes resources{};checkHip(hipFuncGetAttributes(&resources,node.func),"Owner production resources");gdn_merged_extended::require(resources.localSizeBytes==0u,"Owner native geometry/resource differs");
        };
        inspect(*verifier,true,nullptr); // Authenticate known22arg family before borrowing its live-bank address.
        std::size_t n=0;checkHip(hipGraphGetNodes(verifier->graph,nullptr,&n),"Owner live-bank count");ASSERT_EQ(n,std::size_t(prepared?2:1));
        std::vector<hipGraphNode_t> native_nodes(n);checkHip(hipGraphGetNodes(verifier->graph,native_nodes.data(),&n),"Owner live-bank nodes");
        hipGraphNode_t node{};
        for(auto possible:native_nodes)
        {
            std::size_t parents=0;checkHip(hipGraphNodeGetDependencies(possible,nullptr,&parents),"Owner live-bank dependency");
            if(parents==std::size_t(prepared?1:0))node=possible;
        }
        ASSERT_NE(node,nullptr);
        hipGraphNodeType kind{};checkHip(hipGraphNodeGetType(node,&kind),"Owner live-bank node type");ASSERT_EQ(kind,hipGraphNodeTypeKernel);
        hipKernelNodeParams params{};checkHip(hipGraphKernelNodeGetParams(node,&params),"Owner live-bank binding");
        ASSERT_TRUE(params.kernelParams);
        const void* live_bank=nullptr;std::memcpy(&live_bank,params.kernelParams[8],sizeof(live_bank));ASSERT_NE(live_bank,nullptr);
        inspect(*verifier,true,live_bank);inspect(*ordinary,false,live_bank);
        const auto compare=[&](const float* transposed,const float* canonical)
        {
            for(int h=0;h<Heads;++h)for(int key=0;key<dk;++key)for(int value=0;value<DV;++value)
                ASSERT_EQ(std::memcmp(transposed+(h*DV+value)*dk+key,canonical+(h*dk+key)*DV+value,sizeof(float)),0);
        };
        for(int seed=0;seed<3;++seed)
        {
            std::vector<float> source(raw.count),a(alpha.count),b(beta.count),initial(state),native(state);
            for(std::size_t i=0;i<source.size();++i)source[i]=float(int((i*7+seed*11)%31)-15)*.03125f;
            for(std::size_t i=0;i<a.size();++i){a[i]=float(int((i+seed)%5)-2)*.25f;b[i]=float(int((i+seed)%7)-3)*.5f;}
            for(int i=0;i<state;++i)initial[i]=seed%2&&i%19==0?-0.f:float((i+seed)%17-8)*.00073f;
            for(int h=0;h<Heads;++h)for(int key=0;key<dk;++key)for(int value=0;value<DV;++value)
                std::memcpy(&native[(h*DV+value)*dk+key],&initial[(h*dk+key)*DV+value],sizeof(float));
            raw.copyFrom(source);alpha.copyFrom(a);beta.copyFrom(b);reference_state.copyFrom(initial);
            ASSERT_TRUE(kernel.importState(native.data(),nullptr,stream.stream));
            std::vector<float> initial_history(prepared?conv_state:0),host_weights(weights.count),host_bias(bias.count);
            if(prepared)
            {
                for(int i=0;i<conv_state;++i)initial_history[i]=float((i+seed)%17-8)*.03125f;
                for(std::size_t i=0;i<host_weights.size();++i)host_weights[i]=float(int((i*3+seed)%13)-6)*.0625f;
                for(std::size_t i=0;i<host_bias.size();++i)host_bias[i]=float(int((i+seed)%5)-2)*.015625f;
                reference_conv.copyFrom(initial_history);weights.copyFrom(host_weights);bias.copyFrom(host_bias);
                ASSERT_TRUE(convolution.importState(initial_history.data(),nullptr,stream.stream));
                for(auto* bank:{&postconv,&reference_raw,&conv_snapshots,&conv_work,&compact})bank->fill(Poison);
            }
            for(auto* bank:{&snapshots,&work,&wrong_snapshots,&wrong_work,&output,&reference_output})bank->fill(Poison);
            reference_first->launch(stream.stream);verifier->launch(stream.stream);
            checkHip(hipStreamSynchronize(stream.stream),"Owner verifier terminal observation");
            const auto golden=reference_state.toHost(),shot=snapshots.toHost(),actual=output.toHost(),expected=reference_output.toHost();
            ASSERT_EQ(std::memcmp(actual.data(),expected.data(),Heads*DV*sizeof(float)),0);compare(shot.data(),golden.data());
            for(std::size_t i=Heads*DV;i<actual.size();++i)EXPECT_EQ(actual[i],Poison);
            std::vector<float> live_state(state);ASSERT_TRUE(kernel.exportState(live_state.data(),nullptr,stream.stream));
            checkHip(hipStreamSynchronize(stream.stream),"Owner live state observation");ASSERT_EQ(std::memcmp(live_state.data(),native.data(),state*sizeof(float)),0);
            for(float x:work.toHost())EXPECT_EQ(x,Poison);
            if(prepared)
            {
                const auto golden_history=reference_conv.toHost(),actual_history=conv_snapshots.toHost();
                ASSERT_EQ(std::memcmp(actual_history.data(),golden_history.data(),conv_state*sizeof(float)),0);
                std::vector<float> live_history(conv_state);ASSERT_TRUE(convolution.exportState(live_history.data(),nullptr,stream.stream));
                checkHip(hipStreamSynchronize(stream.stream),"Prepared owner speculative history observation");
                ASSERT_EQ(std::memcmp(live_history.data(),initial_history.data(),conv_state*sizeof(float)),0);
                for(float x:conv_work.toHost())EXPECT_EQ(x,Poison);
                restore->launch(stream.stream);
            }
            else ASSERT_TRUE(kernel.restoreVerifierStateCaptureRow(nullptr,0,stream.stream));
            reference_next->launch(stream.stream);ordinary->launch(stream.stream);
            checkHip(hipStreamSynchronize(stream.stream),"Owner restored continuation observation");
            const auto final_expected=reference_output.toHost(),final_output=output.toHost(),final_state=reference_state.toHost();
            ASSERT_EQ(std::memcmp(final_expected.data(),final_output.data(),final_output.size()*sizeof(float)),0);
            ASSERT_TRUE(kernel.exportState(live_state.data(),nullptr,stream.stream));checkHip(hipStreamSynchronize(stream.stream),"Owner continuation state observation");compare(live_state.data(),final_state.data());
            for(auto* bank:{&wrong_snapshots,&wrong_work,&work})for(float x:bank->toHost())EXPECT_EQ(x,Poison);
            const auto final_shots=snapshots.toHost();for(std::size_t i=state;i<final_shots.size();++i)EXPECT_EQ(final_shots[i],Poison);
            const auto raw_after=raw.toHost(),alpha_after=alpha.toHost(),beta_after=beta.toHost();
            EXPECT_EQ(std::memcmp(raw_after.data(),source.data(),source.size()*sizeof(float)),0);
            EXPECT_EQ(std::memcmp(alpha_after.data(),a.data(),a.size()*sizeof(float)),0);
            EXPECT_EQ(std::memcmp(beta_after.data(),b.data(),b.size()*sizeof(float)),0);
            for(auto pair:{std::pair{&log,-.375f},std::pair{&dt,.125f}})
                for(float x:pair.first->toHost())EXPECT_EQ(std::memcmp(&x,&pair.second,sizeof(float)),0);
            if(prepared)
            {
                const auto want_raw=reference_raw.toHost(),got_raw=postconv.toHost();
                ASSERT_EQ(std::memcmp(want_raw.data(),got_raw.data(),want_raw.size()*sizeof(float)),0);
                const auto want_history=reference_conv.toHost();std::vector<float> got_history(conv_state);
                ASSERT_TRUE(convolution.exportState(got_history.data(),nullptr,stream.stream));
                checkHip(hipStreamSynchronize(stream.stream),"Prepared owner continued history observation");
                ASSERT_EQ(std::memcmp(want_history.data(),got_history.data(),conv_state*sizeof(float)),0);
                const auto cs=conv_snapshots.toHost(),cw=conv_work.toHost(),cq=compact.toHost();
                for(std::size_t i=conv_state;i<cs.size();++i)EXPECT_EQ(cs[i],Poison);
                for(float x:cw)EXPECT_EQ(x,Poison);
                for(std::size_t i=2*Keys*dk;i<cq.size();++i)EXPECT_EQ(cq[i],Poison);
                const auto actual_weights=weights.toHost(),actual_bias=bias.toHost();
                EXPECT_EQ(std::memcmp(actual_weights.data(),host_weights.data(),actual_weights.size()*sizeof(float)),0);
                EXPECT_EQ(std::memcmp(actual_bias.data(),host_bias.data(),actual_bias.size()*sizeof(float)),0);
            }
            ASSERT_EQ(kernel.recurrenceStateLayout(),Layout::ValueMajorKeyContiguous);
        }
    }
}


/**
 * @test Real value-major owner rejects foreign workspace formats and restores captured state.
 * @details Captures real merged verifier and ordinary graphs once. An independent
 * frozen planar oracle advances the same two rows; no reference state is used as
 * a candidate input. Rejected same-sized workspace binds must leave good bindings
 * intact, and the same restored live bank must drive ordinary continuation.
 */
TEST(Test__ROCmGDNPaddedRealLength, CapturedValueMajorMergedRestoreAndBindingsByteExact)
{
    proveValueMajorOwnerRestore(false);
}

/** @test Real producer/consumer owners capture, restore and reset without recapture or a host shadow. */
TEST(Test__ROCmGDNPaddedRealLength, CapturedPreparedQKOwnersRestoreAndResetByteExact)
{
    proveValueMajorOwnerRestore(true);
}
