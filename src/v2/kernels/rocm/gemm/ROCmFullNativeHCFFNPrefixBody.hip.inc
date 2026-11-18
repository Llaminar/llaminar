/**
 * @file ROCmFullNativeHCFFNPrefixBody.hip.inc
 * @brief Production-owned HC write/norm/down/TwoTerm/up/mix resident phase.
 *
 * All pointers, live rows and arithmetic partitions come from the existing HC
 * declaration. Idle norm CTAs prefetch immutable native Q8 packets; complete
 * output-column owners retain ascending K32 and P accumulators. Three original
 * instance-owned grid publications retire normalized operands, raw down, and
 * mixed respectively. Bounded continuations cover larger admitted geometry;
 * no module state, auxiliary global allocation or host count mirror exists.
 */
#pragma once
#include "ROCmFullNativeHCFFNPrefix.h"
#include "kernels/qwen4exp/GPUBranchRMSNormBody.cuh"
#include "kernels/common/GPUHCDownPublication.cuh"
#include "kernels/common/GPUHCShortKBlockBody.cuh"
#include "kernels/common/DeviceQ8ActivationNumericalContract.h"
#include "ROCmSmallNFP32ProjectionBody.hip.inc"
namespace llaminar2::rocm::full_hc_native {
/** @brief Borrow immutable prepared bytes through their admitted global address space.
 * @tparam T Exact existing packet operand, without numeric conversion.
 * @param pointer Real prepared payload/metadata bytes. @return Identical loaded bytes.
 * @pre Only globally owned immutable weights may enter this reader.
 * @details The explicit AS1 load keeps weight completion independent of LDS
 * counters, as in the qualified issue path. It does not signal readiness early.
 */
template<class T>
__device__ __forceinline__ T weightRead(const void* pointer) {
 const auto* source=reinterpret_cast<const unsigned char __attribute__((address_space(1)))*>(
  reinterpret_cast<std::uintptr_t>(pointer));
 T value;__builtin_memcpy(&value,source,sizeof(T));return value;
}
/** @brief Qualified global-address weight acquisition for the sole canonical up body. */
struct GlobalWeightReader {
 /** @brief Load the existing operand without changing its numeric interpretation.
  * @tparam T Actual payload/half type. @param pointer Prepared global bytes.
  * @return Original operand, acquired through AS1 as in the qualified winner.
  */
 template<class T> __device__ __forceinline__ T read(const void* pointer) const {return weightRead<T>(pointer);}
};
/** @brief Immutable compiler-visible Q8 packet; raw scale bits remain unconsumed. */
struct Packet {int4 low,high;unsigned scale_bits;};
/** @brief Read only the actual prepared packet.
 * @param weights Original native19 matrix. @param column Real output column.
 * @param block Real K32 block. @return Eight payload words and original half bits.
 */
__device__ __forceinline__ Packet loadPacket(const DeviceNativeQuantisedMatrixDesc& weights,
 unsigned column,unsigned block) {
 const auto linear=std::size_t(block)*unsigned(weights.n)+column;
 const auto* p=weights.payload+linear*32;
 return {weightRead<int4>(p),weightRead<int4>(p+16),
  unsigned(weightRead<unsigned short>(static_cast<const unsigned short*>(weights.scales)+linear))};
}
/** @brief Consume one original signed dot8 and its sole original rounded commit.
 * @param packet Completed original packet. @param input Original32-code block.
 * @param scale Original stored activation scale. @return Original K32 contribution.
 */
__device__ __forceinline__ float consumePacket(const Packet& packet,const signed char* input,float scale) {
 const int w[8]{packet.low.x,packet.low.y,packet.low.z,packet.low.w,
  packet.high.x,packet.high.y,packet.high.z,packet.high.w};
 const auto* x=reinterpret_cast<const int*>(input);int dot=0;
#pragma unroll
 for(unsigned word=0;word<8;++word)dot=rocm_integer_contract::signedDot4(x[word],w[word],dot);
 const float ws=__half2float(__ushort_as_half(static_cast<unsigned short>(packet.scale_bits)));
 return device_native_quantised_contract::singleScaleBlock(dot,ws,scale);
}
/** @brief Bind the two original native MUL operand priorities, without extra FP work. */
struct OriginalNormalizedProducts {
 /** @brief Original inner multiply. @param inverse RMS inverse. @param post POST word. @return RN(inverse*POST). */
 __device__ __forceinline__ float inversePost(float inverse,float post) const {
  float value;asm volatile("v_mul_f32 %0, %1, %2":"=v"(value):"v"(inverse),"v"(post));return value;
 }
 /** @brief Original outer multiply. @param value Inner result. @param gain Original gain. @return RN(value*gain). */
 __device__ __forceinline__ float appendGain(float value,float gain) const {
  float result;asm volatile("v_mul_f32 %0, %1, %2":"=v"(result):"v"(value),"v"(gain));return result;
 }
};
/**
 * @brief Publish four original adjacent values through the exact lane0 K32 root.
 * @param value Rounded normalized quad. @param output Original code block.
 * @param scale_output Original scale. @param lane Complete width8 cohort lane.
 * @details Physical4/2/1 transfers virtualize original16/8/4 edges per component;
 * component0/2,1/3 then even/odd retain the original2/1 root operand orientation.
 * The unchanged native reciprocal feeds four original quantizeQ8 leaves.
 */
__device__ __forceinline__ void normQ8Four(float4 value,signed char* output,float* scale_output,unsigned lane) {
 float a=fabsf(value.x),b=fabsf(value.y),c=fabsf(value.z),d=fabsf(value.w);
#pragma unroll
 for(int offset=4;offset>0;offset>>=1) {
  a=fmaxf(a,__shfl_down(a,offset,8));b=fmaxf(b,__shfl_down(b,offset,8));
  c=fmaxf(c,__shfl_down(c,offset,8));d=fmaxf(d,__shfl_down(d,offset,8));
 }
 const float even=fmaxf(a,c),odd=fmaxf(b,d),root=fmaxf(even,odd);
 const float maximum=__shfl(root,0,8);
 const float scale=device_q8_activation_contract::publishPositiveFiniteHalfScale(
  device_q8_activation_contract::boundedScale(maximum));
 const float inverse=1.0f/scale;
 if(lane==0)*scale_output=scale;
 const unsigned q0=unsigned(decode_equivalent::quantizeQ8(value.x,inverse))&255u;
 const unsigned q1=unsigned(decode_equivalent::quantizeQ8(value.y,inverse))&255u;
 const unsigned q2=unsigned(decode_equivalent::quantizeQ8(value.z,inverse))&255u;
 const unsigned q3=unsigned(decode_equivalent::quantizeQ8(value.w,inverse))&255u;
 const unsigned packed=q0|(q1<<8)|(q2<<16)|(q3<<24);
 __builtin_memcpy(output+4*lane,&packed,sizeof(packed));
}
/** @brief Stable original POST view with a bounded phase-local cache. */
struct StagedBranchInput {
 const float *cached,*published;unsigned cached_channels;
 /** @brief Consume an already-published original POST channel.
  * @param channel Real channel. @return Original rounded POST bits.
  */
 __device__ __forceinline__ float retained(unsigned channel) const {
  return channel<cached_channels?cached[channel]:published[channel];
 }
 /** @brief Supply the same original value to the unchanged square/FMA body.
  * @param channel Real channel. @return Original rounded POST bits.
  */
 __device__ __forceinline__ float publish(unsigned channel) const {return retained(channel);}
};
/** @brief Wider normalized publication, sharing the sole original norm body. */
struct RegisterNormQ8Output {
 const float *gain_cache,*gain_public;unsigned cached_channels;
 signed char* codes;float* scales;
 /** @brief Consume completed total explicitly while preserving squareSum's member contract.
  * @tparam Input Original immutable POST source. @tparam Output Original normalized pointer.
  * @param input Stable staged/public POST. @param branch_gain Original gain pointer (bound above).
  * @param output Original public normalized row. @param hidden Original H.
  * @param epsilon Original RMS epsilon. @param base Original branch-major offset.
  * @param logical_sum Original member return (unused). @param completed_total Published lane0 total.
  * @param lane Physical output lane. @param member Original square membership (unused).
  */
 template<class Input,class Output>
 __device__ __forceinline__ void publish(const Input& input,const float* branch_gain,Output output,
  int hidden,float epsilon,std::uint64_t base,float logical_sum,const float* completed_total,
  unsigned lane,bool member) const {
  (void)branch_gain;(void)logical_sum;(void)member;
  using namespace qwen4exp::branch_norm_native;
  for(unsigned quad=lane;quad<unsigned(hidden/4);quad+=1024) {
   const unsigned first=4*quad;
   const float inverse=normalizationInverse(completed_total[0],hidden,epsilon);
   float4 value;
   value.x=normalizedElement(input.retained(first),inverse,
    first<cached_channels?gain_cache[first]:gain_public[first],OriginalNormalizedProducts{});
   value.y=normalizedElement(input.retained(first+1),inverse,
    first+1<cached_channels?gain_cache[first+1]:gain_public[first+1],OriginalNormalizedProducts{});
   value.z=normalizedElement(input.retained(first+2),inverse,
    first+2<cached_channels?gain_cache[first+2]:gain_public[first+2],OriginalNormalizedProducts{});
   value.w=normalizedElement(input.retained(first+3),inverse,
    first+3<cached_channels?gain_cache[first+3]:gain_public[first+3],OriginalNormalizedProducts{});
   output[base+first]=value.x;output[base+first+1]=value.y;
   output[base+first+2]=value.z;output[base+first+3]=value.w;
   normQ8Four(value,codes+(quad/8)*32,scales+quad/8,quad%8);
  }
 }
};
/** @brief One bounded complete-column down panel, sharing issue/consume coordinates. */
struct DownPanel {
 unsigned columns,parts,window,per,blocks;
};
/** @brief Derive actual bounded panel dimensions.
 * @param a Original arithmetic policy. @param owned Complete owner interval.
 * @param words Actual dynamic float capacity. @return Positive panel for a live owner.
 */
__device__ __forceinline__ DownPanel downPanel(const PersistentHCReadArgs& a,unsigned owned,unsigned words) {
 const unsigned blocks=unsigned(a.down_weights.k)/32,per=(blocks+unsigned(a.partitions)-1)/unsigned(a.partitions);
 const unsigned parts=unsigned(fullHCMin(unsigned(a.partitions),32));
 const unsigned window=unsigned(fullHCMin(per,32));
 const unsigned columns=unsigned(fullHCMin(owned,fullHCMin(1024/parts,words/(parts*(window+1)))));
 return {columns,parts,window,per,blocks};
}
/** @brief Issue the original first-window packet only for a valid real leaf.
 * @param a Original down matrix. @param owner Unique complete output interval.
 * @param panel Actual panel geometry. @param item Strided physical item.
 * @return Actual original packet, or an initialized unused value.
 */
__device__ __forceinline__ Packet firstPacket(const PersistentHCReadArgs& a,
 FullHCDownOwnership owner,DownPanel panel,unsigned item) {
 if(owner.columns&&item<panel.parts*panel.window*panel.columns) {
  const unsigned leaf=item/panel.columns,part=leaf/panel.window,offset=leaf%panel.window;
  const unsigned block=part*panel.per+offset;
  if(block<panel.blocks)return loadPacket(a.down_weights,owner.first+item%panel.columns,block);
 }
 return {};
}
/**
 * @brief Append a bounded load-ahead batch without changing any FP addition edge.
 * @param sum Original accumulator. @param values Published leaf plane.
 * @param stride Original columns. @param column Original local column.
 * @param count Exact live leaf count. @return Increasing-index RN fold.
 */
__device__ __forceinline__ float orderedAppend(float sum,const float* values,unsigned stride,
 unsigned column,unsigned count) {
 unsigned i=0;
 for(;i+8<=count;i+=8) {
  float a=values[(i+0)*stride+column],b=values[(i+1)*stride+column];
  float c=values[(i+2)*stride+column],d=values[(i+3)*stride+column];
  float e=values[(i+4)*stride+column],f=values[(i+5)*stride+column];
  float g=values[(i+6)*stride+column],h=values[(i+7)*stride+column];
  asm volatile("":"+v"(a),"+v"(b),"+v"(c),"+v"(d),"+v"(e),"+v"(f),"+v"(g),"+v"(h));
  sum=device_fp32_contract::add(sum,a);sum=device_fp32_contract::add(sum,b);
  sum=device_fp32_contract::add(sum,c);sum=device_fp32_contract::add(sum,d);
  sum=device_fp32_contract::add(sum,e);sum=device_fp32_contract::add(sum,f);
  sum=device_fp32_contract::add(sum,g);sum=device_fp32_contract::add(sum,h);
 }
 if(i+4<=count) {
  float a=values[(i+0)*stride+column],b=values[(i+1)*stride+column];
  float c=values[(i+2)*stride+column],d=values[(i+3)*stride+column];
  asm volatile("":"+v"(a),"+v"(b),"+v"(c),"+v"(d));
  sum=device_fp32_contract::add(sum,a);sum=device_fp32_contract::add(sum,b);
  sum=device_fp32_contract::add(sum,c);sum=device_fp32_contract::add(sum,d);i+=4;
 }
 for(;i<count;++i)sum=device_fp32_contract::add(sum,values[i*stride+column]);
 return sum;
}
/**
 * @brief Complete every original down leaf, partition and raw output for one owner.
 * @param a Original HC binding. @param owner Unique live column interval.
 * @param panel Bounded physical geometry. @param first Prefetched item0 packet.
 * @param second Prefetched item1024 packet. @param prefetched Whether idle issue occurred.
 * @param arena Exclusive dynamic phase arena. @param words Admitted float capacity.
 * @details One thread retains each partition accumulator across bounded K32
 * windows; one column thread retains the raw accumulator across partition
 * windows. Every append remains increasing, including original empty-partition
 * +0 and terminal multiply1. Only memory issue is batched. Initial two packets
 * are reused once; counted continuation loads all remaining real packets.
 */
__device__ __forceinline__ void down(const PersistentHCReadArgs& a,FullHCDownOwnership owner,
 DownPanel panel,const Packet& first,const Packet& second,bool prefetched,float* arena,unsigned words) {
 (void)words;
 const unsigned tid=threadIdx.x;
 if(!owner.columns) {__syncthreads();__syncthreads();return;}
 for(unsigned begin_col=0;begin_col<owner.columns;begin_col+=panel.columns) {
  const unsigned columns=unsigned(fullHCMin(panel.columns,owner.columns-begin_col));
  float raw=0.0f;
  for(unsigned begin_part=0;begin_part<unsigned(a.partitions);begin_part+=panel.parts) {
   const unsigned parts=unsigned(fullHCMin(panel.parts,unsigned(a.partitions)-begin_part));
   float sum=0.0f;
   for(unsigned begin_k=0;begin_k<panel.per;begin_k+=panel.window) {
    const unsigned window=unsigned(fullHCMin(panel.window,panel.per-begin_k));
    const unsigned items=parts*window*columns;
    for(unsigned item=tid;item<items;item+=1024) {
     const unsigned leaf=item/columns,part=begin_part+leaf/window;
     const unsigned offset=begin_k+leaf%window,block=part*panel.per+offset;
     // Invalid partition tails publish no padded leaf and are never consumed.
     if(block<panel.blocks) {
      Packet packet;
      if(prefetched&&begin_col==0&&begin_part==0&&begin_k==0&&item==tid)packet=first;
      else if(prefetched&&begin_col==0&&begin_part==0&&begin_k==0&&item==tid+1024)packet=second;
      else packet=loadPacket(a.down_weights,owner.first+begin_col+item%columns,block);
      arena[item]=consumePacket(packet,a.input_codes+block*32,a.input_scales[block]);
     }
    }
    __syncthreads();
    if(tid<parts*columns) {
     const unsigned part=begin_part+tid/columns,column=tid%columns;
     const unsigned first_block=part*panel.per+begin_k;
     const unsigned count=first_block<panel.blocks?
      unsigned(fullHCMin(window,panel.blocks-first_block)):0;
     sum=orderedAppend(sum,arena+(tid/columns)*window*columns,columns,column,count);
    }
    // All leaf consumers retire before the next window overwrites its arena.
    __syncthreads();
   }
   float* partition_values=arena+panel.parts*panel.window*panel.columns;
   if(tid<parts*columns) {
    // The old global P plane is retired internal scratch. Only the
    // owner-local terminal values feed the identical final fold.
    partition_values[tid]=__fmul_rn(sum,1.0f);
   }
   __syncthreads();
   if(tid<columns) {
    // P1 originally publishes its terminal product directly, without an
    // added +0 edge; preserve signed-zero and exceptional operand behavior.
    raw=a.partitions==1?partition_values[tid]:orderedAppend(raw,partition_values,columns,tid,parts);
   }
   // The completed P window is retired before any next K32 leaf publication.
   __syncthreads();
  }
  if(tid<columns)a.raw_down[owner.first+begin_col+tid]=raw;
  __syncthreads();
 }
}
/**
 * @brief Original independent FP32 four-output next-write injection.
 * @param a Original normalized input, source and explicit load-pair policy.
 * @param scratch Retired norm/down phase storage, at least256 floats.
 * @details All physical lanes enter the original logical256 helper. The first
 * global edge retires incoming write gates before a temporally aliased injection
 * output is authored. It is published by the second edge, and retained through
 * the final mixed-ready edge for original downstream consumers.
 */
__device__ __forceinline__ void injection(const PersistentHCReadArgs& a,float* scratch) {
 if(a.injection.weights&&a.injection.rows.activeRows())
  for(unsigned output=blockIdx.x;output<4;output+=gridDim.x) {
   const float* w=a.injection.weights+std::size_t(output)*a.down_weights.k;
   float value;
   if(a.injection.load_pairs==8)value=floating_projection_native::row<8>(a.norm.output,w,
    a.down_weights.k,scratch,threadIdx.x,threadIdx.x<256);
   else if(a.injection.load_pairs==4)value=floating_projection_native::row<4>(a.norm.output,w,
    a.down_weights.k,scratch,threadIdx.x,threadIdx.x<256);
   else value=floating_projection_native::row<1>(a.norm.output,w,
    a.down_weights.k,scratch,threadIdx.x,threadIdx.x<256);
   if(threadIdx.x==0)a.injection.output[output]=value;
   __syncthreads();
  }
}
/**
 * @brief Original TwoTerm up contributions and fixed increasing folds over bounded channels.
 * @param a Original immutable source/public output declaration.
 * @param local Published local activation banks, retired only at the final grid.
 * @param arena Exclusive up panel. @param words Admitted float capacity.
 * @param output_demand Frozen external/observation publications; local operands are unconditional.
 * @details Every native weight serves both original dot8 chains via the existing
 * canonical contribution helper. Output owners retain separate p/q chains, then
 * one p+q and original branch0..3 sigmoid/FMA mix. No cross-column reassociation.
 */
__device__ __forceinline__ void up(const PersistentHCReadArgs& a,
 const FullNativeHCFFNPrefixScratch& local,float* arena,unsigned words,FullNativeHCFFNOutputDemand output_demand) {
 const unsigned tid=threadIdx.x,H=unsigned(a.norm.hidden),groups=unsigned(a.up.weights.blocks_per_row);
 const unsigned first=unsigned(std::uint64_t(blockIdx.x)*H/gridDim.x);
 const unsigned end=unsigned(std::uint64_t(blockIdx.x+1)*H/gridDim.x);
 const unsigned panel_channels=unsigned(fullHCMax(1,fullHCMin(end-first,words/(8*groups))));
 auto operands=a.up;operands.primary=const_cast<signed char*>(local.primary);
 operands.residual=const_cast<signed char*>(local.residual);
 operands.primaryScales=const_cast<float*>(local.primary_scales);
 operands.residualScales=const_cast<float*>(local.residual_scales);
 for(unsigned begin=first;begin<end;begin+=panel_channels) {
  const unsigned n=unsigned(fullHCMin(panel_channels,end-begin)),columns=4*n;
  float* primary=arena;float* residual=arena+groups*columns;
  for(unsigned task=tid;task<groups*columns;task+=1024) {
   const unsigned block=task/columns,ci=task%columns;
   const unsigned column=(ci/n)*H+begin+ci%n;
   float p,q;hc_short_k_native::contribution(operands,block,column,p,q,GlobalWeightReader{});
   primary[task]=p;residual[task]=q;
  }
  __syncthreads();
  for(unsigned ci=tid;ci<columns;ci+=1024) {
   float p=0.0f,q=0.0f;
   for(unsigned block=0;block<groups;++block) {
    p=device_fp32_contract::add(p,primary[block*columns+ci]);
    q=device_fp32_contract::add(q,residual[block*columns+ci]);
   }
   const float value=device_fp32_contract::add(p,q);
   if(output_demand.contains(FullNativeHCFFNOutputRole::HCUp))
    a.up.output[std::size_t(ci/n)*H+begin+ci%n]=value;
   primary[ci]=value;
  }
  __syncthreads();
  for(unsigned channel=tid;channel<n;channel+=1024) {
   float mixed=0.0f;
#pragma unroll
   for(unsigned branch=0;branch<4;++branch) {
    const float gate=1.0f/(1.0f+expf(-primary[branch*n+channel]));
    mixed=fmaf(gate,a.norm.output[std::size_t(branch)*H+begin+channel],mixed);
   }
   a.mixed[begin+channel]=mixed*0.25f;
  }
  __syncthreads();
 }
}
} // namespace llaminar2::rocm::full_hc_native
namespace llaminar2::rocm {
/**
 * @brief Complete caller-owned production HC prefix with three instance-owned grid edges.
 * @tparam Grid Existing typed per-instance resident barrier, retaining the56B bank.
 * @param prefix Original source/count/write/injection declaration.
 * @param grid Exact admitted grid authority; every physical CTA joins three times.
 * @param local Small static bank, unioned with suffix scratch only after retirement.
 * @param dynamic_storage Borrowed aligned dynamic arena, overlaid with the suffix.
 * @param dynamic_bytes Original cold-derived prefix bound, sealed by the retained kernel node.
 * @pre Exactly1024 physical lanes; original norm/injection arithmetic uses256.
 * @details Counts remain device-owned and independent: inactive norm retains its
 * existing normalized bytes; inactive publication reads the already-bound Q8 bytes;
 * inactive up retains logits/mixed. Raw down is still authored exactly as in the
 * original operation. No host mirror, substitute source or geometry fallback is
 * introduced. Original public activation writes are delayed until the third edge
 * so exact raw/activated aliases cannot overwrite another CTA's raw input.
 */
template<class Grid>
__device__ __forceinline__ void fullNativeHCFFNPrefix(const FullNativeHCFFNPrefixArgs& prefix,
 Grid& grid,FullNativeHCFFNPrefixScratch& local,unsigned char* dynamic_storage,std::size_t dynamic_bytes) {
 using namespace full_hc_native;
 const auto& a=prefix.hc;
 const unsigned tid=threadIdx.x,H=unsigned(a.norm.hidden),R=unsigned(a.down_weights.n);
 // Cold admission and capture validation derive this bound from these exact
 // immutable source dimensions. Do not repeat that host geometry program in
 // every resident CTA; active row counts and all tensor data stay device-owned.
 if(blockDim.x!=1024||!dynamic_bytes||dynamic_bytes>32768||(dynamic_bytes&15u)||!dynamic_storage)__builtin_trap();
 float* arena=reinterpret_cast<float*>(dynamic_storage);const unsigned words=unsigned(dynamic_bytes/4);
 const auto owner=fullHCDownOwnership(blockIdx.x,gridDim.x,R);
 DownPanel panel{};
 if(owner.columns)panel=downPanel(a,owner.columns,words);
 Packet first{},second{};
 const bool early=gridDim.x>4&&blockIdx.x>=4;
 if(early) {first=firstPacket(a,owner,panel,tid);second=firstPacket(a,owner,panel,tid+1024);}
 const int norm_rows=a.norm.extent.liveRows?*a.norm.extent.liveRows:1;
 if(norm_rows<0||norm_rows>1)__builtin_trap();
 for(unsigned task=blockIdx.x;task<4;task+=gridDim.x) {
  signed char* codes=a.input_codes+std::size_t(task)*H;
  float* scales=a.input_scales+std::size_t(task)*(H/32);
  if(norm_rows) {
   // The bounded cache is complete32-value cohorts; excess channels retain
   // their original globally published POST/gain, with exactly the same math.
   const unsigned cache=unsigned(fullHCMin(H,((words-256)/2)/32*32));
   float* post_cache=arena+256;float* gain_cache=post_cache+cache;
   const auto base=std::size_t(task)*H;
   const float gate=qwen4exp::branch_norm_native::residualGate(prefix.write.gateLogits[task]);
   const qwen4exp::branch_norm_native::ResidualBranchInput original{
    prefix.write.residual+base,prefix.write.block,prefix.write.output+base,gate};
   for(unsigned channel=tid;channel<H;channel+=1024) {
    const float gain=a.norm.gain[base+channel],post=original.publish(channel);
    if(channel<cache){post_cache[channel]=post;gain_cache[channel]=gain;}
   }
   __syncthreads();
   const StagedBranchInput input{post_cache,prefix.write.output+base,cache};
   qwen4exp::branch_norm_native::taskFromBranchInput(input,a.norm.gain+base,a.norm.output,
    H,a.norm.epsilon,task,arena,tid,tid<256,
    RegisterNormQ8Output{gain_cache,a.norm.gain+base,cache,codes,scales});
   __syncthreads();
  } else {
   // Original count0 retains normalized FP32 data but still publishes its
   // complete original SingleQ8 down operand, including every real block.
   for(unsigned quad=tid;quad<H/4;quad+=1024) {
    const auto first_value=std::size_t(task)*H+4*quad;
    const float4 value{a.norm.output[first_value],a.norm.output[first_value+1],
     a.norm.output[first_value+2],a.norm.output[first_value+3]};
    normQ8Four(value,codes+(quad/8)*32,scales+quad/8,quad%8);
   }
   __syncthreads();
  }
 }
 grid.sync(); // All public normalized/Q8 words; old incoming write gates retired.
 injection(a,arena);
 down(a,owner,panel,first,second,early,arena,words);
 grid.sync(); // All raw outputs; input_codes/scales may now become the shorter up banks.
 const bool publish=a.publication.rows.activeRows()!=0;
 if(tid<R) {
  if(publish) {
   auto output=a.publication;
   output.activated=local.activated;output.primary=local.primary;output.residual=local.residual;
   output.primaryScales=local.primary_scales;output.residualScales=local.residual_scales;
   hc_down_publication::publish(a.raw_down[tid],output,tid,
    hc_down_publication::NativePositiveHalfScalePublication{});
  } else {local.primary[tid]=a.up.primary[tid];local.residual[tid]=a.up.residual[tid];}
 }
 if(!publish&&tid<R/32) {
  local.primary_scales[tid]=a.up.primaryScales[tid];
  local.residual_scales[tid]=a.up.residualScales[tid];
 }
 __syncthreads();
 if(publish&&blockIdx.x==0) {
  if(tid<R){a.publication.primary[tid]=local.primary[tid];a.publication.residual[tid]=local.residual[tid];}
  if(tid<R/32){a.publication.primaryScales[tid]=local.primary_scales[tid];
   a.publication.residualScales[tid]=local.residual_scales[tid];}
 }
 if(a.up.rows.activeRows())up(a,local,arena,words,prefix.output_demand);
 grid.sync(); // Mixed/injection ready; all replicated raw readers and local up users retired.
 if(prefix.output_demand.contains(FullNativeHCFFNOutputRole::HCActivated)&&publish&&blockIdx.x==0&&tid<R)
  a.publication.activated[tid]=local.activated[tid];
 // Static scratch is unioned with the suffix. Retire the final public copy
 // before this CTA can overwrite its activation source through that union.
 __syncthreads();
}
} // namespace llaminar2::rocm
