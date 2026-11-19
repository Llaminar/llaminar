/**
 * @file CastagnaHcPersistExport.hip
 * @brief Thin diagnostic launch adapter around the byte-identical original device section.
 *
 * Original implementation: Castagna Veloce3483d462715d61f21ed5877836be35a724446359,
 * ggml/src/ggml-cuda/hc-persist.cu. See CASTAGNA-LICENSE. No kernel/helper/global
 * body is edited. The generated include excludes only subsequent host graph
 * matching/launch plumbing, replaced by semantic typed argument construction.
 */
#include "CastagnaHcPersistExport.h"
#include "CastagnaOriginalDeviceSection.inc"
#include <cstring>
#include <limits>

using namespace qfn_castagna_reference;
namespace {
/** @brief Validate pointers/alignment required by the same original source admission. */
bool validRequest(const Request& r) noexcept
{
    const auto& p=r.prefix;const auto& f=r.ffn;const auto& o=r.outputs;
    if(!p.residual||!p.previous_output||!p.previous_gate_logits||!p.norm_weights||
       !p.hc_down_q8_0||!p.hc_up_q8_0||!f.router||!f.shared_gate_q8_0||!f.shared_up_q8_0||
       !f.shared_down_q8_0||!f.shared_gate||!f.expert_gate||!f.expert_up||!f.expert_down||
       !o.post_residual||!o.normalized||!o.mixed||!o.router_logits||!o.ffn)return false;
    if(f.routed_columns<64||f.routed_columns>FFRMAX||f.routed_columns%64||f.routed_columns/32>16||
       f.shared_columns<64||f.shared_columns>FF||f.shared_columns%64||
       f.top_k<1||f.top_k>NUMAX)return false;
    if(f.gate_source!=GateSource::Q4_K&&f.gate_source!=GateSource::Q5_K&&f.gate_source!=GateSource::Q8_0)return false;
    if(f.down_source!=DownSource::Q5_1&&f.down_source!=DownSource::Q8_0)return false;
    const auto aligned=[](const void* pointer,std::uintptr_t bytes){return reinterpret_cast<std::uintptr_t>(pointer)%bytes==0;};
    if(!aligned(f.expert_gate,16)||!aligned(f.expert_up,16)||!aligned(f.expert_down,4)||
       !aligned(f.shared_gate_q8_0,4)||!aligned(f.shared_up_q8_0,4)||!aligned(f.shared_down_q8_0,4)||
       !aligned(p.hc_down_q8_0,4)||!aligned(p.hc_up_q8_0,4))return false;
    // Same original T1 arena capacity predicate, using its actual q81 type.
    return std::size_t(f.top_k*f.routed_columns+f.shared_columns)/32*sizeof(q81)<=
        std::size_t(E*4>NT*4?E*4:NT*4);
}
/** @brief Compare every captured pointer/scalar semantically, ignoring struct padding. */
bool sameRequest(const Request& a,const Request& b) noexcept
{
    const auto sameFloat=[](float x,float y){std::uint32_t xb,yb;std::memcpy(&xb,&x,sizeof(xb));std::memcpy(&yb,&y,sizeof(yb));return xb==yb;};
    const auto& x=a.prefix;const auto& y=b.prefix;const auto& f=a.ffn;const auto& g=b.ffn;const auto& o=a.outputs;const auto& p=b.outputs;
    return x.residual==y.residual&&x.previous_output==y.previous_output&&x.previous_gate_logits==y.previous_gate_logits&&
        x.norm_weights==y.norm_weights&&x.hc_down_q8_0==y.hc_down_q8_0&&x.hc_up_q8_0==y.hc_up_q8_0&&
        sameFloat(x.epsilon,y.epsilon)&&sameFloat(x.down_scale,y.down_scale)&&sameFloat(x.pre_scale,y.pre_scale)&&
        sameFloat(x.post_scale1,y.post_scale1)&&sameFloat(x.post_scale2,y.post_scale2)&&
        f.router==g.router&&f.shared_gate_q8_0==g.shared_gate_q8_0&&f.shared_up_q8_0==g.shared_up_q8_0&&
        f.shared_down_q8_0==g.shared_down_q8_0&&f.shared_gate==g.shared_gate&&f.expert_gate==g.expert_gate&&
        f.expert_up==g.expert_up&&f.expert_down==g.expert_down&&f.gate_source==g.gate_source&&f.down_source==g.down_source&&
        f.routed_columns==g.routed_columns&&f.shared_columns==g.shared_columns&&f.top_k==g.top_k&&
        sameFloat(f.selected_sum_clamp,g.selected_sum_clamp)&&o.post_residual==p.post_residual&&o.normalized==p.normalized&&
        o.mixed==p.mixed&&o.router_logits==p.router_logits&&o.ffn==p.ffn&&o.phase_timestamps==p.phase_timestamps;
}
/** @brief Derive source row bytes from original GGUF block definitions. */
std::size_t gateRowBytes(GateSource source) noexcept
{
    switch(source){case GateSource::Q4_K:return E/QK_K*sizeof(block_q4_K);
    case GateSource::Q5_K:return E/QK_K*sizeof(block_q5_K);
    case GateSource::Q8_0:return E/QK8_0*sizeof(block_q8_0);}
    return 0;
}
}
extern "C" const void* qfn_castagna_t1_post_ffn_kernel() noexcept
{
    return reinterpret_cast<const void*>(&hc_persist_kernel<1,true,false,false,false>);
}
extern "C" hipError_t qfn_castagna_t1_post_ffn_prepare(const Request* request,int device,Admission* admission) noexcept
{
    if(!request||!admission||!validRequest(*request))return hipErrorInvalidValue;
    int current=-1;auto error=hipGetDevice(&current);if(error!=hipSuccess)return error;
    if(current!=device)return hipErrorInvalidDevice;
    hipDeviceProp_t properties{};error=hipGetDeviceProperties(&properties,device);if(error!=hipSuccess)return error;
    if(std::strncmp(properties.gcnArchName,"gfx906",6)!=0||
       (properties.gcnArchName[6]!='\0'&&properties.gcnArchName[6]!=':')||properties.multiProcessorCount!=NB)return hipErrorInvalidDevice;
    Admission observed{};observed.function=qfn_castagna_t1_post_ffn_kernel();observed.device=device;
    observed.grid=NB;observed.threads=NT;observed.compute_units=properties.multiProcessorCount;
    error=hipFuncGetAttributes(&observed.resources,observed.function);if(error!=hipSuccess)return error;
    int active=0;error=hipOccupancyMaxActiveBlocksPerMultiprocessor(&active,observed.function,NT,0);if(error!=hipSuccess)return error;
    if(active<1||observed.resources.maxThreadsPerBlock<NT||observed.resources.localSizeBytes!=0)return hipErrorInvalidConfiguration;
    observed.active_blocks_per_cu=static_cast<unsigned>(active);observed.request=*request;*admission=observed;return hipSuccess;
}
extern "C" hipError_t qfn_castagna_t1_post_ffn_launch(const Request* request,const Admission* admission,hipStream_t stream) noexcept
{
    if(!request||!admission||reinterpret_cast<std::uintptr_t>(stream)<=2||
       !sameRequest(*request,admission->request)||admission->function!=qfn_castagna_t1_post_ffn_kernel()||admission->grid!=NB||admission->threads!=NT)
        return hipErrorInvalidValue;
    const auto& r=*request;const auto& p=r.prefix;const auto& f=r.ffn;const auto& o=r.outputs;
    hc_src src{};src.x=p.residual;src.out=p.previous_output;src.raw=p.previous_gate_logits;
    src.ps1=p.post_scale1;src.ps2=p.post_scale2;
    hc_ffn ffn{};ffn.w_logits=f.router;ffn.w_sgate=f.shared_gate_q8_0;ffn.w_sup=f.shared_up_q8_0;
    ffn.w_sdown=f.shared_down_q8_0;ffn.w_sg=f.shared_gate;ffn.logits=o.router_logits;ffn.on=1;ffn.ff=f.shared_columns;
    auto& mo=ffn.moe;mo.w_gate=f.expert_gate;mo.w_up=f.expert_up;mo.w_down=f.expert_down;
    mo.sg=static_cast<long long>(gateRowBytes(f.gate_source)*f.routed_columns);mo.su=mo.sg;
    mo.rd=static_cast<long long>(f.routed_columns/32*(f.down_source==DownSource::Q5_1?sizeof(block_q5_1):sizeof(block_q8_0)));
    mo.sd=mo.rd*E;mo.ffr=f.routed_columns;mo.dq8=f.down_source==DownSource::Q8_0;mo.nused=f.top_k;
    mo.clamp=f.selected_sum_clamp;mo.out=o.ffn;mo.on=1;mo.dedup=1;
    mo.gq5=f.gate_source==GateSource::Q5_K?1:f.gate_source==GateSource::Q8_0?2:0;
    const hc_early early{};const hc_ar ar{};
    hc_persist_kernel<1,true,false,false,false><<<NB,NT,0,stream>>>(src,p.norm_weights,p.hc_down_q8_0,p.hc_up_q8_0,
        p.epsilon,p.down_scale,p.pre_scale,o.post_residual,o.normalized,o.mixed,ffn,early,ar,o.phase_timestamps);
    return hipGetLastError();
}
extern "C" hipError_t qfn_castagna_t1_post_ffn_globals(GlobalRegions* regions) noexcept
{
    if(!regions)return hipErrorInvalidValue;
    GlobalRegions result{};unsigned i=0;
#define BORROW_GLOBAL(Name) do { void* address=nullptr;const auto status=hipGetSymbolAddress(&address,HIP_SYMBOL(Name)); \
    if(status!=hipSuccess)return status;result.regions[i++]={#Name,address,sizeof(Name)};result.total_bytes+=sizeof(Name); } while(false)
    BORROW_GLOBAL(g_hcp_acc);BORROW_GLOBAL(g_hcp_bar);BORROW_GLOBAL(g_hcp_inj);BORROW_GLOBAL(g_hcp_kern);
    BORROW_GLOBAL(g_hcp_h);BORROW_GLOBAL(g_hcp_sg);BORROW_GLOBAL(g_hcp_shexp);BORROW_GLOBAL(g_hcp_arseq);
    BORROW_GLOBAL(g_hcp_pushed);BORROW_GLOBAL(g_hcp_pushcnt);BORROW_GLOBAL(g_hcp_mh);
#undef BORROW_GLOBAL
    if(i!=global_region_count)return hipErrorInvalidValue;
    *regions=result;return hipSuccess;
}

namespace {
/**
 * @brief Untimed observer of the actual original64-lane routing helper.
 * @param logits Full original512logit result. @param top_k Original selected count.
 * @param clamp Original selected-weight sum clamp. @param ids Selected IDs output.
 * @param weights Selected weights output. Shared arrays match the original helper's
 * private selection publication; no new full-program work or state is introduced.
 */
__global__ __launch_bounds__(64) void castagnaOriginalSelectionObserver(
    const float* logits,int top_k,float clamp,int* ids,float* weights)
{
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__GFX9__)
    __shared__ int selected_ids[NUMAX];
    __shared__ float selected_weights[NUMAX];
    const int lane=threadIdx.x;
    float values[NR/64];
#pragma unroll
    for(int j=0;j<NR/64;++j)values[j]=logits[lane+64*j];
    moe_topk_wave(values,top_k,clamp,lane,selected_ids,selected_weights);
    // Original caller also retires the helper's LDS publication before consumers.
    lds_sync();
    if(lane<top_k){ids[lane]=selected_ids[lane];weights[lane]=selected_weights[lane];}
#endif
}
}
extern "C" const void* qfn_castagna_t1_selection_observer_kernel() noexcept
{
    return reinterpret_cast<const void*>(&castagnaOriginalSelectionObserver);
}
extern "C" hipError_t qfn_castagna_t1_observe_selection(
    const Admission* admission,int* ids,float* weights,hipStream_t stream) noexcept
{
    if(!admission||!ids||!weights||reinterpret_cast<std::uintptr_t>(stream)<=2||
       admission->function!=qfn_castagna_t1_post_ffn_kernel()||admission->grid!=NB||admission->threads!=NT)
        return hipErrorInvalidValue;
    const auto& request=admission->request;
    castagnaOriginalSelectionObserver<<<1,64,0,stream>>>(request.outputs.router_logits,
        request.ffn.top_k,request.ffn.selected_sum_clamp,ids,weights);
    return hipGetLastError();
}
