/**
 * @file ROCmCTAOwnedHCReadBody.hip.inc
 * @brief Wide CTA-owned original HC reducer/publication/up/read completion.
 *
 * Every CTA reads only immutable original down partials (or disjoint P1 raw)
 * and owns48 mixed channels. CTA0 uniquely exports original publication bytes.
 * Local activation and Q8 remove cross-CTA publication dependencies, including
 * the exact raw/activated alias. All K32 contributions and ordered RN folds use
 * original helpers. Include after their normative native definitions.
 */
#pragma once
#include "ROCmCTAOwnedHCRead.h"
#include "ROCmQuantisedActivationProducers.h"
namespace llaminar2::rocm {
/** @brief Phase-disjoint LDS: local activation retires before contribution planes. */
template<unsigned Groups> struct CTAOwnedHCShared {
 union {float activated[Groups*32];struct {float p[Groups][192],q[Groups][192];} terms;} phase;
 alignas(16) signed char primary[Groups*32],residual[Groups*32];
 float primaryScales[Groups],residualScales[Groups];
};
/** @brief Complete original HC tail;1024 physical lanes,48 independently owned mixed channels. */
template<unsigned Groups>
__global__ __launch_bounds__(1024) void ctaOwnedHCTailKernel(PersistentHCReadArgs a) {
 __shared__ CTAOwnedHCShared<Groups> s;
 const unsigned tid=threadIdx.x,H=a.norm.hidden,R=Groups*32;
 const bool live=a.publication.rows.activeRows()!=0;
 // P1 never authored partials. Its raw row is immutable for disjoint output;
 // exact alias is admitted only with one CTA, whose barrier retires all readers.
 if(tid<R) {
  const float raw=a.partitions==1?a.raw_down[tid]:nativeM1FoldColumnValue(a.down_partials,tid,R,a.partitions);
  if(blockIdx.x==0&&a.partitions>1)a.raw_down[tid]=raw;
  if(live) {
   GPUHCDownOutputView local=a.publication;
   local.activated=s.phase.activated;local.primary=s.primary;local.residual=s.residual;
   local.primaryScales=s.primaryScales;local.residualScales=s.residualScales;
   hc_down_publication::publish(raw,local,tid);
  }
 }
 __syncthreads();
 if(!live)return; // Uniform within each CTA, leaving inactive public banks untouched.
 if(blockIdx.x==0) {
  if(tid<R) {a.publication.activated[tid]=s.phase.activated[tid];
   a.publication.primary[tid]=s.primary[tid];a.publication.residual[tid]=s.residual[tid];}
  if(tid<Groups){a.publication.primaryScales[tid]=s.primaryScales[tid];a.publication.residualScales[tid]=s.residualScales[tid];}
 }
 __syncthreads(); // Last activation reader retires before the LDS union changes phase.
 GPUQ8ResidualCorrectionArgs local=a.up;
 local.primary=s.primary;local.residual=s.residual;
 local.primaryScales=s.primaryScales;local.residualScales=s.residualScales;
 for(unsigned logical=tid;logical<Groups*192;logical+=1024) {
  const unsigned group=logical/192,lane=logical%192,channel=blockIdx.x*48+lane%48;
  float p=0.f,q=0.f;
  if(channel<H)hc_short_k_native::contribution(local,group,static_cast<std::uint64_t>(lane/48)*H+channel,p,q);
  s.phase.terms.p[group][lane]=p;s.phase.terms.q[group][lane]=q;
 }
 __syncthreads();
 if(tid<192&&blockIdx.x*48+tid%48<H) {
  float p=0.f,q=0.f;
#pragma unroll
  for(unsigned group=0;group<Groups;++group) {
   p=device_fp32_contract::add(p,s.phase.terms.p[group][tid]);
   q=device_fp32_contract::add(q,s.phase.terms.q[group][tid]);
  }
  const float value=device_fp32_contract::add(p,q);
  a.up.output[static_cast<std::uint64_t>(tid/48)*H+blockIdx.x*48+tid%48]=value;
  s.phase.terms.p[0][tid]=value;
 }
 __syncthreads();
 if(tid<48&&blockIdx.x*48+tid<H) {
  const unsigned channel=blockIdx.x*48+tid;float mixed=0.f;
#pragma unroll
  for(unsigned branch=0;branch<4;++branch) {
   const float gate=1.f/(1.f+expf(-s.phase.terms.p[0][branch*48+tid]));
   mixed=fmaf(gate,a.norm.output[static_cast<std::uint64_t>(branch)*H+channel],mixed);
  }
  a.mixed[channel]=mixed*0.25f;
 }
}
/** @brief Typed exact-once original ticket consumer; never exposes mutable ticket state. */
struct CTAOwnedHCTicketConsumer {
 /** @brief Validate all original producer authorities before submitting the ordinary tail. */
 static void submit(NativeM1ReductionTicket& ticket,const CTAOwnedHCReadProgram& program,
  const PersistentHCReadArgs& args,int device,void* stream) {
  const auto& shape=ticket.producerShape();
  if((ticket.state_!=NativeM1ReductionTicket::State::Pending&&ticket.state_!=NativeM1ReductionTicket::State::Direct)||
     ticket.device_!=device||ticket.stream_!=stream||ticket.has_work_||ticket.output_!=args.raw_down||
     ticket.partials_!=args.down_partials||shape.columns!=args.down_weights.n||shape.partitions!=args.partitions||
     shape.tile_columns!=128||shape.columns_per_thread!=1)
   throw std::logic_error("CTA-owned HC lost its original fully-active deferred producer ticket");
  void* parameters[]{const_cast<PersistentHCReadArgs*>(&args)};
  if(hipLaunchKernel(program.resources().function,dim3((args.norm.hidden+47)/48),dim3(1024),parameters,0,
       static_cast<hipStream_t>(stream))!=hipSuccess)
   throw std::runtime_error("CTA-owned HC ordinary completion enqueue failed");
  ticket.state_=NativeM1ReductionTicket::State::Consumed;
 }
};
} // namespace llaminar2::rocm
