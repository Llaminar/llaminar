/**
 * @file CastagnaNativeWeightExport.hip
 * @brief Diagnostic B adapter retaining original full Castagna phases with native weight readers.
 * See the authenticated substitution manifest and original MIT license. A remains untouched.
 */
#include "CastagnaNativeWeightExport.h"
#include "CastagnaNativeDeviceSection.inc"
#include <cstring>
using namespace qfn_castagna_native_impl;
using qfn_castagna_native::Matrix;
using qfn_castagna_native::Request;
using qfn_castagna_native::Admission;
namespace {
bool sameMatrix(const Matrix& a,const Matrix& b) noexcept {
 return a.payload==b.payload&&a.scales==b.scales&&a.mins==b.mins&&a.emins==b.emins&&a.n==b.n&&a.k==b.k&&
 a.blocks_per_row==b.blocks_per_row&&a.codebook_id==b.codebook_id&&a.source_codebook_id==b.source_codebook_id&&
 a.source_is_superblock==b.source_is_superblock&&a.source_identity_present==b.source_identity_present&&
 a.allocation_payload_bytes_per_block==b.allocation_payload_bytes_per_block&&a.allocation_has_mins==b.allocation_has_mins&&
 a.allocation_has_emins==b.allocation_has_emins&&a.metadata_layout==b.metadata_layout&&a.reserved_tail==b.reserved_tail;
}
bool validMatrix(const Matrix& d,int n,int k,unsigned physical,unsigned source,bool superblock) noexcept {
 return d.valid()&&d.n==n&&d.k==k&&d.blocks_per_row==unsigned(k/32)&&d.codebook_id==physical&&
 d.source_identity_present==1&&d.source_codebook_id==source&&bool(d.source_is_superblock)==superblock&&
 !d.metadata_layout&&!d.reserved_tail&&
 (!d.allocation_payload_bytes_per_block||d.allocation_payload_bytes_per_block==(physical==19?32:physical==7?20:16))&&
 d.allocation_has_mins==(d.allocation_payload_bytes_per_block&&physical!=19)&&!d.allocation_has_emins&&
 d.payload&&d.scales&&!d.emins&&
 (physical==5||physical==7?d.mins!=nullptr:d.mins==nullptr)&&
 reinterpret_cast<uintptr_t>(d.payload)%16==0&&reinterpret_cast<uintptr_t>(d.scales)%2==0&&
 (!d.mins||reinterpret_cast<uintptr_t>(d.mins)%2==0)&&
 // The Q4 temporary panel reads eight adjacent halves per original plane.
 // N/first_row are multiples64, so aligned bases prove every int4 address.
 (physical!=5||(reinterpret_cast<uintptr_t>(d.scales)%16==0&&reinterpret_cast<uintptr_t>(d.mins)%16==0));
}
bool sameRequest(const Request& a,const Request& b) noexcept {
 const auto& x=a.original;const auto& y=b.original;
 const auto f=[](float x,float y){uint32_t u,v;memcpy(&u,&x,4);memcpy(&v,&y,4);return u==v;};
 return x.prefix.residual==y.prefix.residual&&x.prefix.previous_output==y.prefix.previous_output&&
 x.prefix.previous_gate_logits==y.prefix.previous_gate_logits&&x.prefix.norm_weights==y.prefix.norm_weights&&
 x.prefix.hc_down_q8_0==y.prefix.hc_down_q8_0&&x.prefix.hc_up_q8_0==y.prefix.hc_up_q8_0&&
 f(x.prefix.epsilon,y.prefix.epsilon)&&f(x.prefix.down_scale,y.prefix.down_scale)&&f(x.prefix.pre_scale,y.prefix.pre_scale)&&
 f(x.prefix.post_scale1,y.prefix.post_scale1)&&f(x.prefix.post_scale2,y.prefix.post_scale2)&&
 x.ffn.router==y.ffn.router&&x.ffn.shared_gate==y.ffn.shared_gate&&
 x.ffn.shared_gate_q8_0==y.ffn.shared_gate_q8_0&&x.ffn.shared_up_q8_0==y.ffn.shared_up_q8_0&&
 x.ffn.shared_down_q8_0==y.ffn.shared_down_q8_0&&x.ffn.expert_gate==y.ffn.expert_gate&&
 x.ffn.expert_up==y.ffn.expert_up&&x.ffn.expert_down==y.ffn.expert_down&&x.ffn.gate_source==y.ffn.gate_source&&
 x.ffn.down_source==y.ffn.down_source&&x.ffn.routed_columns==y.ffn.routed_columns&&x.ffn.shared_columns==y.ffn.shared_columns&&
 x.ffn.top_k==y.ffn.top_k&&f(x.ffn.selected_sum_clamp,y.ffn.selected_sum_clamp)&&
 x.outputs.post_residual==y.outputs.post_residual&&x.outputs.normalized==y.outputs.normalized&&x.outputs.mixed==y.outputs.mixed&&
 x.outputs.router_logits==y.outputs.router_logits&&x.outputs.ffn==y.outputs.ffn&&x.outputs.phase_timestamps==y.outputs.phase_timestamps&&
 sameMatrix(a.native.hc_down,b.native.hc_down)&&sameMatrix(a.native.hc_up,b.native.hc_up)&&
 sameMatrix(a.native.shared_gate,b.native.shared_gate)&&sameMatrix(a.native.shared_up,b.native.shared_up)&&
 sameMatrix(a.native.shared_down,b.native.shared_down)&&a.native.expert_gate==b.native.expert_gate&&
 a.native.expert_up==b.native.expert_up&&a.native.expert_down==b.native.expert_down;
}
}
extern "C" const void* qfn_castagna_native_weight_kernel() noexcept {
 return reinterpret_cast<const void*>(&hc_persist_kernel<1,true,false,false,false,qfn_castagna_reference::GateSource::Q4_K>);
}
extern "C" hipError_t qfn_castagna_native_weight_prepare(const Request* r,std::span<const Matrix> gate,
 std::span<const Matrix> up,std::span<const Matrix> down,int device,Admission* a) noexcept {
 if(!r||!a||gate.size()!=NR||up.size()!=NR||down.size()!=NR)return hipErrorInvalidValue;
 const auto& f=r->original.ffn;const auto& p=r->original.prefix;const auto& o=r->original.outputs;const auto& n=r->native;
 if(!p.residual||!p.previous_output||!p.previous_gate_logits||!p.norm_weights||!f.router||!f.shared_gate||
 !o.post_residual||!o.normalized||!o.mixed||!o.router_logits||!o.ffn||!n.expert_gate||!n.expert_up||!n.expert_down||
 f.routed_columns<64||f.routed_columns>FFRMAX||f.routed_columns%64||f.routed_columns/32>16||
 f.shared_columns<64||f.shared_columns>FF||f.shared_columns%64||!f.top_k||f.top_k>NUMAX||
 f.gate_source!=qfn_castagna_reference::GateSource::Q4_K||f.down_source!=qfn_castagna_reference::DownSource::Q5_1||
 !validMatrix(n.hc_down,LR,K,19,19,false)||reinterpret_cast<uintptr_t>(n.hc_down.scales)%4||
 !validMatrix(n.hc_up,K,LR,19,19,false)||
 !validMatrix(n.shared_gate,f.shared_columns,E,19,19,false)||!validMatrix(n.shared_up,f.shared_columns,E,19,19,false)||
 !validMatrix(n.shared_down,E,f.shared_columns,19,19,false)||
 p.hc_down_q8_0||p.hc_up_q8_0||f.shared_gate_q8_0||f.shared_up_q8_0||f.shared_down_q8_0||
 f.expert_gate||f.expert_up||f.expert_down)return hipErrorInvalidValue;
 for(unsigned i=0;i<NR;++i)if(!validMatrix(gate[i],f.routed_columns,E,5,5,true)||
 !validMatrix(up[i],f.routed_columns,E,5,5,true)||!validMatrix(down[i],E,f.routed_columns,7,7,false))return hipErrorInvalidValue;
 if(size_t(f.top_k*f.routed_columns+f.shared_columns)/32*sizeof(q81)>size_t(E*4>NT*4?E*4:NT*4))return hipErrorInvalidValue;
 int current=-1;auto error=hipGetDevice(&current);if(error!=hipSuccess)return error;if(current!=device)return hipErrorInvalidDevice;
 hipDeviceProp_t properties{};error=hipGetDeviceProperties(&properties,device);if(error!=hipSuccess)return error;
 if(strncmp(properties.gcnArchName,"gfx906",6)||(properties.gcnArchName[6]&&properties.gcnArchName[6]!=':')||properties.multiProcessorCount!=NB)return hipErrorInvalidDevice;
 Admission observed{};observed.function=qfn_castagna_native_weight_kernel();observed.device=device;observed.grid=NB;observed.threads=NT;
 error=hipFuncGetAttributes(&observed.resources,observed.function);if(error!=hipSuccess)return error;
 int active=0;error=hipOccupancyMaxActiveBlocksPerMultiprocessor(&active,observed.function,NT,0);if(error!=hipSuccess)return error;
 if(active<1||observed.resources.maxThreadsPerBlock<NT||observed.resources.localSizeBytes)return hipErrorInvalidConfiguration;
 observed.active_blocks_per_cu=active;observed.request=*r;*a=observed;return hipSuccess;
}
extern "C" hipError_t qfn_castagna_native_weight_launch(const Request* r,const Admission* a,hipStream_t stream) noexcept {
 if(!r||!a||reinterpret_cast<uintptr_t>(stream)<=2||!sameRequest(*r,a->request)||
 a->function!=qfn_castagna_native_weight_kernel()||a->grid!=NB||a->threads!=NT)return hipErrorInvalidValue;
 const auto& p=r->original.prefix;const auto& f=r->original.ffn;const auto& o=r->original.outputs;const auto& n=r->native;
 hc_src src{};src.x=p.residual;src.out=p.previous_output;src.raw=p.previous_gate_logits;src.ps1=p.post_scale1;src.ps2=p.post_scale2;
 hc_ffn ffn{};ffn.w_logits=f.router;ffn.w_sg=f.shared_gate;ffn.logits=o.router_logits;ffn.on=1;ffn.ff=f.shared_columns;
 ffn.native_sgate=n.shared_gate;ffn.native_sup=n.shared_up;ffn.native_sdown=n.shared_down;
 auto& mo=ffn.moe;mo.native_gate=n.expert_gate;mo.native_up=n.expert_up;mo.native_down=n.expert_down;
 mo.ffr=f.routed_columns;mo.dq8=0;mo.nused=f.top_k;mo.clamp=f.selected_sum_clamp;mo.out=o.ffn;mo.on=1;mo.dedup=1;mo.gq5=0;
 const hc_early early{};const hc_ar ar{};
 hc_persist_kernel<1,true,false,false,false,qfn_castagna_reference::GateSource::Q4_K><<<NB,NT,0,stream>>>(src,p.norm_weights,n.hc_down,n.hc_up,p.epsilon,p.down_scale,
 p.pre_scale,o.post_residual,o.normalized,o.mixed,ffn,early,ar,o.phase_timestamps);
 return hipGetLastError();
}
extern "C" hipError_t qfn_castagna_native_weight_globals(qfn_castagna_reference::GlobalRegions* regions) noexcept {
 if(!regions)return hipErrorInvalidValue;qfn_castagna_reference::GlobalRegions result{};unsigned i=0;
#define BORROW_GLOBAL(Name) do {void* address=nullptr;const auto status=hipGetSymbolAddress(&address,HIP_SYMBOL(Name)); \
 if(status!=hipSuccess)return status;result.regions[i++]={#Name,address,sizeof(Name)};result.total_bytes+=sizeof(Name);}while(false)
 BORROW_GLOBAL(g_hcp_acc);BORROW_GLOBAL(g_hcp_bar);BORROW_GLOBAL(g_hcp_inj);BORROW_GLOBAL(g_hcp_kern);
 BORROW_GLOBAL(g_hcp_h);BORROW_GLOBAL(g_hcp_sg);BORROW_GLOBAL(g_hcp_shexp);BORROW_GLOBAL(g_hcp_arseq);
 BORROW_GLOBAL(g_hcp_pushed);BORROW_GLOBAL(g_hcp_pushcnt);BORROW_GLOBAL(g_hcp_mh);
#undef BORROW_GLOBAL
 *regions=result;return i==qfn_castagna_reference::global_region_count?hipSuccess:hipErrorInvalidValue;
}
namespace {
/**
 * @brief Untimed observer of the actual original64-lane routing helper.
 * @param logits Full original512logit result. @param top_k Original selected count.
 * @param clamp Original selected-weight sum clamp. @param ids Selected IDs output.
 * @param weights Selected weights output. Shared arrays match the original helper's
 * private selection publication; no new full-program work or state is introduced.
 */
__global__ __launch_bounds__(64) void castagnaNativeSelectionObserver(
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
extern "C" const void* qfn_castagna_native_weight_selection_observer_kernel() noexcept
{
    return reinterpret_cast<const void*>(&castagnaNativeSelectionObserver);
}
extern "C" hipError_t qfn_castagna_native_weight_observe_selection(
    const Admission* admission,int* ids,float* weights,hipStream_t stream) noexcept
{
    if(!admission||!ids||!weights||reinterpret_cast<std::uintptr_t>(stream)<=2||
       admission->function!=qfn_castagna_native_weight_kernel()||admission->grid!=NB||admission->threads!=NT)
        return hipErrorInvalidValue;
    const auto& request=admission->request.original;
    castagnaNativeSelectionObserver<<<1,64,0,stream>>>(request.outputs.router_logits,
        request.ffn.top_k,request.ffn.selected_sum_clamp,ids,weights);
    return hipGetLastError();
}
