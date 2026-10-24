/**
 * @file ROCmGDNValueMajorCompletePerf.h
 * @brief Real production KeyMajor/ValueMajor complete-boundary byte and event fixture.
 *
 * Uses the existing HIP test owners, immutable initial histories and original
 * public convolution/merged-recurrence control. No model policy is installed.
 * Raw convolution, terminal histories, all snapshots and recurrent results stay
 * independently observable. Host permutation is a diagnostic comparison only;
 * both captured graphs use their retained native physical state layouts.
 */
#pragma once
#include "kernels/gdn/GDNMergedRecurrence.h"
#include "kernels/rocm/gdn/ROCmShortConvolution.h"
#include "ROCmGDNPairedNormalizationPerf.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace { namespace gdn_value_major_complete
{
using namespace llaminar2;
using namespace llaminar2::gdn;
/** @brief Fail before accepting incomplete native or numerical evidence. */
inline void require(bool yes, const char* message)
{
    gdn_paired_perf::require(yes, message);
}

/** @brief One fixed TP4 complete pair; initial state is immutable across repeats. */
struct Cell final
{
    static constexpr int Keys=4, Heads=12, DK=128, DV=128;
    static constexpr int Channels=2*Keys*DK+Heads*DV, ConvState=3*Channels;
    static constexpr int State=Heads*DK*DV, Guard=128;
    static constexpr float Poison=-83.25f;
    int rows, repeats, offset;
    bool normalize;
    std::vector<float> input, weight, bias, initial_conv, alpha, beta, initial_state, transposed_state;
    HipStreamHandle stream;
    HipFloatBuffer x, w, b, ci, a, beta_raw, log_a, dt, si, ti;
    HipFloatBuffer raw_control, raw_candidate, conv_control, conv_candidate;
    HipFloatBuffer conv_shots_control, conv_shots_candidate;
    HipFloatBuffer output_control, output_candidate, state_control, state_candidate;
    HipFloatBuffer shots_control, shots_candidate;
    HipIntBuffer live;
    DeviceRequestRowRanges ranges;
    std::unique_ptr<HipCapturedGraph> control, candidate;
    /** @brief Last-declared terminal join protects graphs, native and host owners. */
    struct Retirement final
    {
        hipStream_t stream;
        /** @brief Failure is fatal before any borrowed allocation is released. */
        ~Retirement()
        {
            if(hipStreamSynchronize(stream)!=hipSuccess) std::terminate();
        }
    } retirement;

    /**
     * @brief Allocate/publish/capture two complete immutable-state transactions.
     * @param m Physical request rows, exactly one for this complete decode pair.
     * @param copies Complete conv/recurrence pairs per captured launch,1..256.
     * @param norm Original normalization-enabled or disabled arithmetic.
     * @param head_offset Original signed modular mapping to compact key heads.
     * @details Both sides call the original public convolution and production
     * typed merged recurrence. State layout is the only selected policy.
     */
    Cell(int m, int copies, bool norm=true, int head_offset=0)
        : rows(m), repeats(copies), offset(head_offset), normalize(norm),
          input((m+1)*Channels), weight(4*Channels), bias(Channels),
          initial_conv(ConvState+Guard), alpha(m*Heads), beta(m*Heads),
          initial_state(State+Guard), transposed_state(State+Guard),
          x(input.size()), w(weight.size()), b(bias.size()),
          ci(initial_conv.size()), a(alpha.size()), beta_raw(beta.size()),
          log_a(std::vector<float>(Heads,-0.375f)),dt(std::vector<float>(Heads,0.125f)),
          si(initial_state.size()), ti(transposed_state.size()),
          raw_control((m+1)*Channels),raw_candidate((m+1)*Channels),
          conv_control(ConvState+Guard),conv_candidate(ConvState+Guard),
          conv_shots_control((m+1)*ConvState),conv_shots_candidate((m+1)*ConvState),
          output_control((m+1)*Heads*DV),output_candidate((m+1)*Heads*DV),
          state_control(State+Guard),state_candidate(State+Guard),
          shots_control((m+1)*State),shots_candidate((m+1)*State),live(1),
          ranges(DeviceRequestRowRanges::deviceCounted(1,m,live.ptr)),retirement{stream.stream}
    {
        require(rows>=1&&rows<=16&&repeats>=1&&repeats<=256,"Invalid complete conv cell");
        require(rows==1,"Complete ValueMajor pair requires physical M1");
        require(reinterpret_cast<std::uintptr_t>(stream.stream)>2,
            "Complete conv requires an explicit nondefault stream");
        // Each region is a distinct existing fixture-owned physical allocation.
        // Prove complete declared extents, not just unequal starting pointers.
        const std::array buffers{&x,&w,&b,&ci,&a,&beta_raw,&log_a,&dt,&si,&ti,
            &raw_control,&raw_candidate,&conv_control,&conv_candidate,
            &conv_shots_control,&conv_shots_candidate,
            &output_control,&output_candidate,&state_control,&state_candidate,
            &shots_control,&shots_candidate};
        for(std::size_t i=0;i<buffers.size();++i)
            for(std::size_t j=0;j<i;++j)
            {
                const auto left=reinterpret_cast<std::uintptr_t>(buffers[i]->ptr);
                const auto right=reinterpret_cast<std::uintptr_t>(buffers[j]->ptr);
                require(left+buffers[i]->count*sizeof(float)<=right||
                    right+buffers[j]->count*sizeof(float)<=left,
                    "Complete conv fixture physical allocations overlap");
            }
        publish(rows,0);
        control=std::make_unique<HipCapturedGraph>(stream.stream,[&]
        {
            for(int i=0;i<repeats;++i)
            {
                if(!rocmGDN_short_conv1d_effective(x.ptr,w.ptr,b.ptr,raw_control.ptr,
                    ci.ptr,conv_control.ptr,rows,Channels,4,true,live.ptr,
                    rows==1?nullptr:conv_shots_control.ptr,ConvState,rows==1?0:rows,
                    0,stream.stream)) return false;
                if(!launchROCmMergedRecurrence(recurrence(false),stream.stream)) return false;
            }
            return true;
        });
        candidate=std::make_unique<HipCapturedGraph>(stream.stream,[&]
        {
            for(int i=0;i<repeats;++i)
            {
                if(!rocmGDN_short_conv1d_effective(x.ptr,w.ptr,b.ptr,raw_candidate.ptr,
                    ci.ptr,conv_candidate.ptr,rows,Channels,4,true,live.ptr,
                    nullptr,ConvState,0,0,stream.stream)) return false;
                if(!launchROCmMergedRecurrence(recurrence(true),stream.stream)) return false;
            }
            return true;
        });
        inspect(false);
        inspect(true);
    }

    /** @return Exact record family for the retained production complete boundary. */
    std::string marker(const char* role) const
    {
        return std::string("GDN_VALUE_MAJOR_COMPLETE_")+role;
    }

    /** @return Explicit native physical format and zero-preparation-storage proof. */
    const char* variantFields() const noexcept
    {
        return ",variant=production_value_major,prepared_storage=0,candidate_state_layout=head_value_key";
    }

    /**
     * @brief Bind exact original raw/gate/state owners for one retained side.
     * @param trial True selects candidate destinations; all immutable sources are shared.
     * @return Borrowed native input with the original live owner and snapshot stride.
     */
    MergedRecurrenceInput recurrence(bool trial)
    {
        return {ranges,Keys,Heads,DK,DV,offset,normalize,
            trial?raw_candidate.ptr:raw_control.ptr,a.ptr,beta_raw.ptr,log_a.ptr,dt.ptr,
            trial?output_candidate.ptr:output_control.ptr,
            trial?ti.ptr:si.ptr,
            trial?state_candidate.ptr:state_control.ptr,
            rows==1?nullptr:(trial?shots_candidate.ptr:shots_control.ptr),State,rows==1?0:rows,trial?GDNRecurrenceStateLayout::ValueMajorKeyContiguous:
                GDNRecurrenceStateLayout::KeyMajorValueContiguous};
    }

    /**
     * @brief Publish dirty original sources and poison every physical destination.
     * @param active True recurrence-live prefix, including zero.
     * @param seed Deterministic distinct raw/weight/history/gate values.
     * @details The original convolution clamps zero live to one, independently
     * of recurrence. All publication and its intentional terminal join precede capture/events.
     */
    void publish(int active,int seed)
    {
        require(active>=0&&active<=rows,"Invalid complete conv live prefix");
        for(std::size_t i=0;i<input.size();++i) input[i]=float(int((i*7+seed*11)%31)-15)*0.03125f;
        for(std::size_t i=0;i<weight.size();++i) weight[i]=float(int((i*3+seed)%13)-6)*0.0625f;
        for(std::size_t i=0;i<bias.size();++i) bias[i]=float(int((i+seed)%5)-2)*0.015625f;
        for(std::size_t i=0;i<initial_conv.size();++i) initial_conv[i]=float(int((i+seed)%17)-8)*0.03125f;
        for(std::size_t i=0;i<initial_state.size();++i) initial_state[i]=float(int((i+seed)%17)-8)*0.00073f;
        for(std::size_t i=0;i<alpha.size();++i)
        {
            alpha[i]=float(int((i+seed)%6)-2)*0.125f;
            beta[i]=float(int((i+seed)%9)-4)*0.5f;
        }
        x.copyFrom(input);w.copyFrom(weight);b.copyFrom(bias);ci.copyFrom(initial_conv);
        si.copyFrom(initial_state);a.copyFrom(alpha);beta_raw.copyFrom(beta);
        {
            // Test setup authors a distinct physical representation. Production
            // must retain this layout across phases; no transpose enters capture.
            transposed_state=initial_state;
            for(int h=0;h<Heads;++h)for(int v=0;v<DV;++v)for(int k=0;k<DK;++k)
                transposed_state[(h*DV+v)*DK+k]=initial_state[(h*DK+k)*DV+v];
            ti.copyFrom(transposed_state);
        }
        for(auto* buffer:{&raw_control,&raw_candidate,&conv_control,&conv_candidate,
            &conv_shots_control,&conv_shots_candidate,
            &output_control,&output_candidate,&state_control,&state_candidate,
            &shots_control,&shots_candidate}) buffer->fill(Poison);
        checkHip(hipMemcpyAsync(live.ptr,&active,sizeof(active),hipMemcpyHostToDevice,
            stream.stream),"Complete boundary publish live");
        checkHip(hipStreamSynchronize(stream.stream),"Complete boundary source publication join");
    }

    /**
     * @brief Prove complete bytes, independent history extents and physical guards.
     * @param active Original true recurrence-live prefix; convolution's clamp is separate.
     * @param execute False observes already completed timed output without replay.
     * @details Complete native control output/state/snapshot comparisons require
     * every logical byte after the diagnostic inverse physical permutation.
     */
    void check(int active,bool execute=true)
    {
        if(execute){control->launch(stream.stream);candidate->launch(stream.stream);}
        checkHip(hipStreamSynchronize(stream.stream),"Complete conv parity terminal join");
        for(auto pair:{std::pair{&raw_control,&raw_candidate},std::pair{&conv_control,&conv_candidate},
            std::pair{&conv_shots_control,&conv_shots_candidate},std::pair{&output_control,&output_candidate},
            std::pair{&state_control,&state_candidate},std::pair{&shots_control,&shots_candidate}})
        {
            const auto old=pair.first->toHost();
            auto trial=pair.second->toHost();
            if(pair.second==&state_candidate||pair.second==&shots_candidate)
            {
                // Compare semantic bytes after a diagnostic host permutation.
                // Guard suffixes and inactive poison rows remain observable.
                const auto physical=trial;
                for(std::size_t base=0;base+State<=trial.size();base+=State)
                    for(int h=0;h<Heads;++h)for(int k=0;k<DK;++k)for(int v=0;v<DV;++v)
                        trial[base+(h*DK+k)*DV+v]=physical[base+(h*DV+v)*DK+k];
            }
            require(old.size()==trial.size()&&std::memcmp(old.data(),trial.data(),old.size()*sizeof(float))==0,
                "Complete conv/raw/state/snapshot bytes differ");
        }
        const auto guard=[&](const std::vector<float>& v,std::size_t first)
        {
            for(std::size_t i=first;i<v.size();++i)
                require(std::memcmp(&v[i],&Poison,sizeof(float))==0,"Complete boundary physical guard changed");
        };
        const auto raw=raw_candidate.toHost();
        guard(raw,std::size_t(rows)*Channels);
        const int conv_live=std::max(1,std::min(rows,active));
        if(rows>1)
            for(std::size_t i=std::size_t(conv_live)*Channels;i<std::size_t(rows)*Channels;++i)
                require(raw[i]==0.f&&!std::signbit(raw[i]),"Clamped convolution inactive row not +0");
        const auto cv=conv_candidate.toHost(),snap=conv_shots_candidate.toHost();
        guard(cv,rows==1?ConvState:0);
        guard(snap,rows==1?0:std::size_t(conv_live)*ConvState);
        for(int row=0;row<(rows==1?1:conv_live);++row)
            for(int ch=0;ch<Channels;++ch)for(int j=0;j<3;++j)
            {
                const int from=row-2+j;
                const float expected=from>=0?input[std::size_t(from)*Channels+ch]:initial_conv[ch*3+3+from];
                const float actual=rows==1?cv[ch*3+j]:snap[std::size_t(row)*ConvState+ch*3+j];
                require(std::memcmp(&actual,&expected,sizeof(float))==0,"Independent convolution history differs");
            }
        const auto out=output_candidate.toHost();guard(out,std::size_t(rows)*Heads*DV);
        for(std::size_t i=std::size_t(active)*Heads*DV;i<std::size_t(rows)*Heads*DV;++i)
            require(out[i]==0.f&&!std::signbit(out[i]),"Inactive recurrence output not +0");
        guard(state_candidate.toHost(),State);guard(shots_candidate.toHost(),rows==1?0:std::size_t(active)*State);
        for(auto pair:{std::pair{&x,&input},std::pair{&w,&weight},std::pair{&b,&bias},
            std::pair{&ci,&initial_conv},std::pair{&si,&initial_state},std::pair{&a,&alpha},std::pair{&beta_raw,&beta}})
        {
            const auto observed=pair.first->toHost();
            require(std::memcmp(observed.data(),pair.second->data(),observed.size()*sizeof(float))==0,
                "Read-only complete conv source changed");
        }
        const auto transposed_observed=ti.toHost();
        require(transposed_observed.size()==transposed_state.size()&&
            (transposed_observed.empty()||std::memcmp(transposed_observed.data(),transposed_state.data(),
                transposed_state.size()*sizeof(float))==0),"Read-only transposed initial state changed");
        for(const float value:log_a.toHost())require(value==-0.375f,"Learned log gate changed");
        for(const float value:dt.toHost())require(value==0.125f,"Learned time bias changed");
        int observed_live=-1;
        checkHip(hipMemcpyAsync(&observed_live,live.ptr,sizeof(observed_live),hipMemcpyDeviceToHost,
            stream.stream),"Complete conv terminal count observation");
        checkHip(hipStreamSynchronize(stream.stream),"Complete conv terminal count join");
        require(observed_live==active,"Original live-count owner changed");
        std::cout<<marker("BYTES")<<variantFields()<<",M="<<rows<<",live="<<active<<",normalize="<<normalize
            <<",offset="<<offset<<",key_heads=4,heads=12,DK=128,DV=128,repeats="<<repeats
            <<",conv_live="<<conv_live<<",terminal_conv="<<(rows==1?"written":"snapshots")<<",passed=1\n";
    }

    /**
     * @brief Authenticate both functions, complete linear DAG and every argument.
     * @param trial True selects production ValueMajor; false selects original KeyMajor.
     * @details Inspection is cold and never executes or changes a captured node.
     */
    void inspect(bool trial);
};

/**
 * @brief Copy one cold native argument without typed-pointer aliasing.
 * @tparam T Exact original native argument type.
 * @param node Borrowed kernel parameter view owned by the retained graph.
 * @param index Original zero-based native parameter ordinal.
 * @return Exact copied native value; no borrowed host temporary escapes.
 */
template<class T> T argument(const hipKernelNodeParams& node,int index)
{
    require(node.kernelParams&&node.kernelParams[index]&&!node.extra,"Missing conv native argument");
    T value;
    std::memcpy(&value,node.kernelParams[index],sizeof(T));
    return value;
}

inline void Cell::inspect(bool trial)
{
    auto& graph=trial?*candidate:*control;
    std::size_t count=0;
    checkHip(hipGraphGetNodes(graph.graph,nullptr,&count),"Complete conv node count");
    require(count==std::size_t(2*repeats),"Complete conv graph must have two nodes per repeat");
    std::vector<hipGraphNode_t> nodes(count);
    checkHip(hipGraphGetNodes(graph.graph,nodes.data(),&count),"Complete conv nodes");
    std::vector<int> next(count,-1);
    int root=-1;
    for(std::size_t i=0;i<count;++i)
    {
        hipGraphNodeType type{};
        checkHip(hipGraphNodeGetType(nodes[i],&type),"Complete conv node type");
        require(type==hipGraphNodeTypeKernel,"Unexpected complete conv nonkernel work");
        std::size_t n=0;
        checkHip(hipGraphNodeGetDependencies(nodes[i],nullptr,&n),"Complete conv dependencies");
        require(n<=1,"Complete conv graph has multiple parents");
        if(!n)
        {
            require(root<0,"Complete conv graph has multiple roots");
            root=int(i);
        }
        else
        {
            hipGraphNode_t parent{};
            checkHip(hipGraphNodeGetDependencies(nodes[i],&parent,&n),"Complete conv parent");
            const auto at=std::find(nodes.begin(),nodes.end(),parent);
            require(at!=nodes.end(),"Foreign complete conv parent");
            const auto j=std::size_t(at-nodes.begin());
            require(next[j]<0,"Complete conv graph branches");
            next[j]=int(i);
        }
    }
    void* functions[2]{};
    int at=root;
    for(std::size_t step=0;step<count;++step)
    {
        require(at>=0&&std::size_t(at)<count,"Broken complete conv chain");
        hipKernelNodeParams node{};
        checkHip(hipGraphKernelNodeGetParams(nodes[at],&node),"Complete conv native tuple");
        const bool rec=step%2;
        const char* raw=hipKernelNameRefByPtr(node.func,stream.stream);
        require(raw,"Unresolved complete conv native function");
        int status=-1;
        std::unique_ptr<char,decltype(&std::free)> name(abi::__cxa_demangle(raw,nullptr,nullptr,&status),&std::free);
        require(status==0&&bool(name),"Complete conv native demangle failed");
        const std::string actual(name.get());
        const std::string family=rec
            ?(trial?"rocm_gdn_value_major_scalar_kernel<128,":"rocm_gdn_merged_recurrent_step_kernel<128>(")
            :"rocm_short_conv1d_decode_kernel(";
        require(actual.find("(anonymous namespace)::"+family)!=std::string::npos,
            "Wrong complete conv native family");
        if(rec&&trial)
        {
            require(actual.find("GDNRecurrentInputLayout)1")!=std::string::npos&&
                actual.find("GdnRecurrenceEndpointAccess)0")!=std::string::npos,
                "Production complete pair changed source/alignment program");
        }
        require(!functions[rec]||functions[rec]==node.func,"Complete conv role changed function");
        functions[rec]=node.func;
        const unsigned block=rec?(trial?512:128):256;
        require(node.blockDim.x==block&&node.blockDim.y==1&&node.blockDim.z==1&&
            node.gridDim.x==unsigned(rec?Heads:10)&&node.gridDim.y==unsigned(rec?4:1)&&
            node.gridDim.z==1&&node.sharedMemBytes==(rec?(trial?0u:7184u):0u),"Wrong complete conv geometry/LDS");
        const auto pointer=[&](int index,const void* value)
        {
            require(argument<const void*>(node,index)==value,"Complete conv captured pointer differs");
        };
        const auto integer=[&](int index,int value)
        {
            require(argument<int>(node,index)==value,"Complete conv captured scalar differs");
        };
        if(rec)
        {
            const auto p=recurrence(trial);
            pointer(0,p.merged);pointer(1,nullptr);
            pointer(2,nullptr);pointer(3,p.alpha);pointer(4,p.beta_raw);
            pointer(5,p.A_log);pointer(6,p.dt_bias);pointer(7,p.output);
            pointer(8,p.initial_state);pointer(9,p.updated_state);
            integer(10,1);integer(11,rows);integer(12,Heads);integer(13,DV);
            require(argument<bool>(node,14)==normalize,"Original QK policy changed");
            pointer(15,live.ptr);integer(16,0);pointer(17,p.snapshots);
            integer(18,State);integer(19,rows==1?0:rows);integer(20,Keys);integer(21,offset);
        }
        else
        {
            pointer(0,x.ptr);pointer(1,w.ptr);pointer(2,b.ptr);
            pointer(3,trial?raw_candidate.ptr:raw_control.ptr);
            pointer(4,ci.ptr);pointer(5,trial?conv_candidate.ptr:conv_control.ptr);
            if(rows==1)
            {
                integer(6,Channels);integer(7,4);
                require(argument<bool>(node,8),"Original convolution SiLU changed");
            }
            else
            {
                pointer(6,live.ptr);pointer(7,conv_shots_control.ptr);integer(8,ConvState);
                integer(9,rows);integer(10,rows);integer(11,Channels);integer(12,4);
                require(argument<bool>(node,13),"Original convolution SiLU changed");
            }
        }
        if(step<2)
        {
            hipFuncAttributes resources{};
            checkHip(hipFuncGetAttributes(&resources,node.func),"Complete conv resources");
            require(resources.localSizeBytes==0,"Complete conv native private memory");
            if(rec&&trial)
                require(resources.sharedSizeBytes==0,"Transposed scalar owns static shared memory");
            std::cout<<marker("CAPTURE")<<variantFields()<<",side="<<(trial?"candidate":"control")
                <<",role="<<(rec?"recurrence":"convolution")<<",M="<<rows
                <<",normalize="<<normalize<<",offset="<<offset<<",key_heads=4,heads=12,DK=128,DV=128"
                <<",nodes="<<count<<",repeats="<<repeats<<",grid="<<node.gridDim.x<<'x'
                <<node.gridDim.y<<'x'<<node.gridDim.z<<",block="<<block
                <<",registers="<<resources.numRegs<<",static_lds="<<resources.sharedSizeBytes
                <<",dynamic_lds="<<node.sharedMemBytes<<",private="<<resources.localSizeBytes
                <<",raw_symbol="<<raw<<",symbol="<<gdn_paired_perf::escaped(actual)<<'\n';
        }
        at=next[at];
    }
    require(at<0,"Cyclic complete conv native graph");
}

/**
 * @brief Full→partial→empty→full dirty replay on the existing fixed captures.
 * @param cell Borrowed exact owners; each check joins before the next publication.
 */
inline void parity(Cell& cell)
{
    int seed=0;
    for(int live:{cell.rows,cell.rows>1?cell.rows-1:0,0,cell.rows})
    {
        cell.publish(live,seed++);
        cell.check(live);
    }
}
/**
 * @brief Share the original complete-boundary event/profile schedule unchanged.
 */
inline void events()
{
    ASSERT_TRUE(hasROCm());
    checkHip(hipSetDevice(0),"Complete conv native event device");
    const int m=gdn_paired_perf::integer("QFN_GDN190_M",1,1,1);
    const int live=gdn_paired_perf::integer("QFN_GDN190_LIVE",m,0,m);
    const int repeats=gdn_paired_perf::integer("QFN_GDN190_REPEATS",64,1,256);
    const char* side_env="QFN_GDN190_PROFILE_SIDE";
    const char* role_env="QFN_GDN190_PROFILE_ROLE";
    const std::string side=std::getenv(side_env)?std::getenv(side_env):"";
    const std::string role=std::getenv(role_env)?std::getenv(role_env):"";
    gdn_value_major_complete::require((side.empty()&&role.empty())||
        ((side=="control"||side=="candidate")&&(role=="convolution"||role=="recurrence")&&repeats==1),
        "Complete conv profile requires exact side/role and repeats1");
    require(m==1,"Complete ValueMajor events require physical M1");
    gdn_paired_perf::Events events;
    gdn_value_major_complete::Cell cell(m,repeats,true,0);
    if(!side.empty())
    {
        cell.publish(live,7);
        const auto& graph=side=="control"?*cell.control:*cell.candidate;
        for(int i=0;i<5;++i)graph.launch(cell.stream.stream);
        checkHip(hipStreamSynchronize(cell.stream.stream),"Complete conv profile warmup join");
        const auto marker=[&](const char* phase)
        {
            std::cout<<cell.marker("SINGLE")<<cell.variantFields()<<",phase="<<phase<<",side="<<side<<",role="<<role
                <<",M="<<m<<",live="<<live<<",repeats=1,warmups=5,selected_occurrence=6"
                <<",graph_launches=6,kernel_dispatches=12\n";
        };
        marker("armed");graph.launch(cell.stream.stream);
        checkHip(hipStreamSynchronize(cell.stream.stream),"Complete conv profile sixth join");
        marker("done");return;
    }
    gdn_value_major_complete::parity(cell);
    cell.publish(live,7);cell.check(live);
    for(int i=0;i<5;++i)
    {
        cell.control->launch(cell.stream.stream);cell.candidate->launch(cell.stream.stream);
    }
    checkHip(hipStreamSynchronize(cell.stream.stream),"Complete conv warmup join");
    for(int sample=0;sample<12;++sample)
    {
        double control,candidate;
        if(sample%2)
        {
            candidate=events.measure(*cell.candidate,cell.stream.stream,repeats);
            control=events.measure(*cell.control,cell.stream.stream,repeats);
        }
        else
        {
            control=events.measure(*cell.control,cell.stream.stream,repeats);
            candidate=events.measure(*cell.candidate,cell.stream.stream,repeats);
        }
        std::cout<<cell.marker("PAIR")<<cell.variantFields()<<",sample="<<sample<<",M="<<m<<",live="<<live
            <<",key_heads=4,heads=12,DK=128,DV=128,repeats="<<repeats
            <<",warmups=5,control_us="<<control<<",candidate_us="<<candidate
            <<",unit=us_per_complete_conv_recurrence\n";
    }
    // Observe the actual final timed output before any new replay/reset.
    cell.check(live,false);
}
} } // anonymous namespace and gdn_value_major_complete

/** @test Production complete decode pair retains exact bytes through dirty replay. */
TEST(Test__ROCmGDNPaddedRealLength, ValueMajorProductionCompleteBoundaryBytes)
{
    ASSERT_TRUE(hasROCm());
    checkHip(hipSetDevice(0),"Production value-major complete conv native device");
    for(bool norm:{true,false})for(int offset:{0,5,-5})
    {
        SCOPED_TRACE(::testing::Message()<<"norm="<<norm<<" offset="<<offset);
        gdn_value_major_complete::Cell cell(1,1,norm,offset);
        gdn_value_major_complete::parity(cell);
    }
    std::cout<<"GDN_VALUE_MAJOR_COMPLETE_SUMMARY,variant=production_value_major,prepared_storage=0,candidate_state_layout=head_value_key,cases=6,states_per_case=4,replays=48,passed=1\n";
}

/** @test Complete original convolution plus transposed-state recurrence timing. */
TEST(Test__ROCmGDNPaddedRealLength, DISABLED_ValueMajorProductionCompleteBoundaryEvents)
{
    gdn_value_major_complete::events();
}
