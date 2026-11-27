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
#include "kernels/rocm/common/ROCmWorkgroupJoin.h"
#include "kernels/rocm/common/ROCmWaveAddress.h"
#include "kernels/rocm/moe/ROCmMoEWaveReduction.h"
#include "ROCmSmallNFP32ProjectionBody.hip.inc"
#include <type_traits>
namespace llaminar2::rocm::full_hc_native {
#if defined(__HIP_DEVICE_COMPILE__)
template<class T> using PrefixLdsPtr = T __attribute__((opencl_local))*;
template<class T> using PrefixGlobalPtr = T __attribute__((address_space(1)))*;
#else
template<class T> using PrefixLdsPtr = T*;
template<class T> using PrefixGlobalPtr = T*;
#endif
using PrefixRawFloat4 = float __attribute__((ext_vector_type(4)));
using PrefixRawInt4 = int __attribute__((ext_vector_type(4)));
template<class T, class U>
__device__ __forceinline__ PrefixLdsPtr<T> prefixAsLds(U* ptr) {
 return (PrefixLdsPtr<T>)((PrefixLdsPtr<unsigned char>)ptr);
}
template<class T>
__device__ __forceinline__ PrefixGlobalPtr<T> prefixAsGlobal(T* ptr) {
 return reinterpret_cast<PrefixGlobalPtr<T>>(reinterpret_cast<std::uintptr_t>(ptr));
}
template<class T>
__device__ __forceinline__ T prefixWaveUniform(T value) {
 if constexpr (std::is_pointer_v<T>)
  return reinterpret_cast<T>(waveUniformAddress(reinterpret_cast<std::uintptr_t>(value)));
 else if constexpr (std::is_same_v<T, float>)
  return __builtin_bit_cast(float, static_cast<unsigned>(__builtin_amdgcn_readfirstlane(__builtin_bit_cast(unsigned, value))));
 else
  return static_cast<T>(__builtin_amdgcn_readfirstlane(static_cast<unsigned>(value)));
}
__device__ __forceinline__ unsigned prefixLane64() {
#if defined(__HIP_DEVICE_COMPILE__)
 unsigned lane;
 asm volatile("v_mbcnt_lo_u32_b32 %0, -1, 0\n\t"
              "v_mbcnt_hi_u32_b32 %0, -1, %0" : "=&v"(lane));
 return lane;
#else
 return static_cast<unsigned>(threadIdx.x) & 63u;
#endif
}
__device__ __forceinline__ float4 prefixLdsLoadFloat4(PrefixLdsPtr<const PrefixRawFloat4> ptr, unsigned idx) {
 const PrefixRawFloat4 v = ptr[idx];
 return make_float4(v[0], v[1], v[2], v[3]);
}
__device__ __forceinline__ void prefixLdsStoreFloat4(PrefixLdsPtr<PrefixRawFloat4> ptr, unsigned idx, float4 v) {
 PrefixRawFloat4 raw;
 raw[0] = v.x; raw[1] = v.y; raw[2] = v.z; raw[3] = v.w;
 ptr[idx] = raw;
}
__device__ __forceinline__ int4 prefixLdsLoadInt4(PrefixLdsPtr<const PrefixRawInt4> ptr, unsigned idx) {
 const PrefixRawInt4 v = ptr[idx];
 return make_int4(v[0], v[1], v[2], v[3]);
}
struct PrefixFastDiv16 {
 unsigned d, m;
 __device__ __forceinline__ static PrefixFastDiv16 make(unsigned divisor) {
  return {divisor, (divisor <= 1u) ? 65536u : ((65536u + divisor - 1u) / divisor)};
 }
 __device__ __forceinline__ unsigned div(unsigned n) const {
  return (d <= 1u) ? n : ((n * m) >> 16);
 }
 __device__ __forceinline__ void divmod(unsigned n, unsigned& q, unsigned& r) const {
  q = div(n);
  r = n - q * d;
 }
};
struct PrefixFastDiv32 {
 unsigned d, magic;
 __device__ __forceinline__ static PrefixFastDiv32 make(unsigned divisor) {
  return {divisor, (divisor <= 1u) ? 0xFFFFFFFFu : (0xFFFFFFFFu / divisor)};
 }
 __device__ __forceinline__ void divmod(unsigned n, unsigned& q, unsigned& r) const {
  if (d <= 1u) { q = n; r = 0u; return; }
  q = __umulhi(n, magic);
  r = n - q * d;
  if (r >= d) { ++q; r -= d; }
 }
};
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
 const int4 x0=weightRead<int4>(input);
 const int4 x1=weightRead<int4>(input+16u);
 int dot=0;
 dot=rocm_integer_contract::signedDot4(x0.x,packet.low.x,dot);
 dot=rocm_integer_contract::signedDot4(x0.y,packet.low.y,dot);
 dot=rocm_integer_contract::signedDot4(x0.z,packet.low.z,dot);
 dot=rocm_integer_contract::signedDot4(x0.w,packet.low.w,dot);
 dot=rocm_integer_contract::signedDot4(x1.x,packet.high.x,dot);
 dot=rocm_integer_contract::signedDot4(x1.y,packet.high.y,dot);
 dot=rocm_integer_contract::signedDot4(x1.z,packet.high.z,dot);
 dot=rocm_integer_contract::signedDot4(x1.w,packet.high.w,dot);
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
__device__ __forceinline__ float prefixOrderedAppendLds(
 float sum, PrefixLdsPtr<const float> values, unsigned stride, unsigned column, unsigned count);
/** @brief Consume one original signed dot8 using AS3 LDS-staged activation codes and scale. */
__device__ __forceinline__ float consumePacketLds(
 const Packet& packet,PrefixLdsPtr<const PrefixRawInt4> codes4_lds,unsigned block,float scale) {
 const int4 x0=prefixLdsLoadInt4(codes4_lds,block*2u);
 const int4 x1=prefixLdsLoadInt4(codes4_lds,block*2u+1u);
 int dot=0;
 dot=rocm_integer_contract::signedDot4(x0.x,packet.low.x,dot);
 dot=rocm_integer_contract::signedDot4(x0.y,packet.low.y,dot);
 dot=rocm_integer_contract::signedDot4(x0.z,packet.low.z,dot);
 dot=rocm_integer_contract::signedDot4(x0.w,packet.low.w,dot);
 dot=rocm_integer_contract::signedDot4(x1.x,packet.high.x,dot);
 dot=rocm_integer_contract::signedDot4(x1.y,packet.high.y,dot);
 dot=rocm_integer_contract::signedDot4(x1.z,packet.high.z,dot);
 dot=rocm_integer_contract::signedDot4(x1.w,packet.high.w,dot);
 const float ws=__half2float(__ushort_as_half(static_cast<unsigned short>(packet.scale_bits)));
 return device_native_quantised_contract::singleScaleBlock(dot,ws,scale);
}
/**
 * @brief Complete every original down leaf, partition and raw output for one owner.
 * @param a Original HC binding. @param owner Unique live column interval.
 * @param panel Bounded physical geometry. @param first Prefetched item0 packet.
 * @param second Prefetched item1024 packet. @param prefetched Whether idle issue occurred.
 * @param arena Exclusive dynamic phase arena. @param words Admitted float capacity.
 * @details One thread retains each partition accumulator across bounded K32
 * windows; one column thread retains the raw accumulator across partition
 * windows. Stages Q8 activation codes and scales in AS3 LDS when space allows,
 * uses wave-uniform FastDiv16 instead of integer division, and reads/writes
 * leaf planes via explicit AS3 LDS pointers while keeping idle-CTA early
 * prefetch and zero global down_partials round-trips.
 */
__device__ __forceinline__ void down(const PersistentHCReadArgs& a,FullHCDownOwnership owner,
 DownPanel panel,const Packet& first,const Packet& second,bool prefetched,float* arena,unsigned words) {
 const unsigned tid=threadIdx.x;
 if(!owner.columns) {__syncthreads();__syncthreads();return;}
 auto arena_lds=prefixAsLds<float>(arena);
 const unsigned panel_leaf_words=panel.parts*panel.window*panel.columns;
 const unsigned panel_total_words=(panel_leaf_words+panel.parts*panel.columns+3u)&~3u;
 const unsigned input_codes_words=panel.blocks*8u;
 const unsigned input_scales_words=panel.blocks;
 const bool stage_inputs=prefixWaveUniform(
  panel.blocks>0u&&(panel_total_words+input_codes_words+input_scales_words)<=words);
 PrefixLdsPtr<PrefixRawInt4> lds_codes4=nullptr;
 PrefixLdsPtr<float> lds_scales=nullptr;
 if(stage_inputs) {
  lds_codes4=prefixAsLds<PrefixRawInt4>(arena+panel_total_words);
  lds_scales=arena_lds+panel_total_words+input_codes_words;
  const auto* g_codes4=reinterpret_cast<const int4*>(prefixWaveUniform(a.input_codes));
  const auto* g_scales=prefixWaveUniform(a.input_scales);
  const unsigned num_int4=panel.blocks*2u;
  for(unsigned i=tid;i<num_int4;i+=1024u) {
   const int4 v=weightRead<int4>(g_codes4+i);
   PrefixRawInt4 raw;raw[0]=v.x;raw[1]=v.y;raw[2]=v.z;raw[3]=v.w;
   lds_codes4[i]=raw;
  }
  for(unsigned i=tid;i<panel.blocks;i+=1024u)
   lds_scales[i]=weightRead<float>(g_scales+i);
  workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
 }
 auto lds_codes4_ro=reinterpret_cast<PrefixLdsPtr<const PrefixRawInt4>>(lds_codes4);
 auto arena_lds_ro=reinterpret_cast<PrefixLdsPtr<const float>>(arena_lds);
 const unsigned a_partitions=prefixWaveUniform(unsigned(a.partitions));
 for(unsigned begin_col=0;begin_col<owner.columns;begin_col+=panel.columns) {
  const unsigned columns=prefixWaveUniform(unsigned(fullHCMin(panel.columns,owner.columns-begin_col)));
  const PrefixFastDiv16 col_div=PrefixFastDiv16::make(columns);
  float raw=0.0f;
  for(unsigned begin_part=0;begin_part<a_partitions;begin_part+=panel.parts) {
   const unsigned parts=prefixWaveUniform(unsigned(fullHCMin(panel.parts,a_partitions-begin_part)));
   float sum=0.0f;
   for(unsigned begin_k=0;begin_k<panel.per;begin_k+=panel.window) {
    const unsigned window=prefixWaveUniform(unsigned(fullHCMin(panel.window,panel.per-begin_k)));
    const PrefixFastDiv16 win_div=PrefixFastDiv16::make(window);
    const unsigned items=parts*window*columns;
    for(unsigned item=tid;item<items;item+=1024u) {
     unsigned leaf,col_off,part_off,k_off;
     col_div.divmod(item,leaf,col_off);
     win_div.divmod(leaf,part_off,k_off);
     const unsigned part=begin_part+part_off;
     const unsigned offset=begin_k+k_off;
     const unsigned block=part*panel.per+offset;
     // Invalid partition tails publish no padded leaf and are never consumed.
     if(block<panel.blocks) {
      Packet packet;
      if(prefetched&&begin_col==0&&begin_part==0&&begin_k==0&&item==tid)packet=first;
      else if(prefetched&&begin_col==0&&begin_part==0&&begin_k==0&&item==tid+1024u)packet=second;
      else packet=loadPacket(a.down_weights,owner.first+begin_col+col_off,block);
      arena_lds[item]=stage_inputs
       ?consumePacketLds(packet,lds_codes4_ro,block,lds_scales[block])
       :consumePacket(packet,a.input_codes+block*32u,a.input_scales[block]);
     }
    }
    workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
    if(tid<parts*columns) {
     unsigned part_off,column;
     col_div.divmod(tid,part_off,column);
     const unsigned part=begin_part+part_off;
     const unsigned first_block=part*panel.per+begin_k;
     const unsigned count=first_block<panel.blocks?
      unsigned(fullHCMin(window,panel.blocks-first_block)):0u;
     sum=prefixOrderedAppendLds(sum,arena_lds_ro+part_off*window*columns,columns,column,count);
    }
    // All leaf consumers retire before the next window overwrites its arena.
    workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
   }
   auto partition_values=arena_lds+panel_leaf_words;
   auto partition_values_ro=arena_lds_ro+panel_leaf_words;
   if(tid<parts*columns) {
    // The old global P plane is retired internal scratch. Only the
    // owner-local terminal values feed the identical final fold.
    partition_values[tid]=__fmul_rn(sum,1.0f);
   }
   workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
   if(tid<columns) {
    // P1 originally publishes its terminal product directly, without an
    // added +0 edge; preserve signed-zero and exceptional operand behavior.
    raw=a_partitions==1u?partition_values_ro[tid]:prefixOrderedAppendLds(raw,partition_values_ro,columns,tid,parts);
   }
   // The completed P window is retired before any next K32 leaf publication.
   workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
  }
  if(tid<columns)prefixAsGlobal(a.raw_down)[owner.first+begin_col+tid]=raw;
  workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
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
/**
 * @brief 1024B coalesced wave64 HC down with simultaneous payload+scale issue and double-buffered LDS.
 * @param a Original HC binding with aligned K32 blocks and down_partials plane.
 * @param dynamic_storage Admitted 16-byte-aligned phase LDS storage.
 * @param words Admitted float capacity.
 * @details All 16 waves issue contiguous 1024B Q8 payload bursts paired with simultaneous
 * scale loads while reading Q8 activations from LDS. Partition folds preserve exact ascending
 * K32 order and terminal multiply-by-1.0f semantics.
 */
__device__ __forceinline__ void fastCoalescedDown(const PersistentHCReadArgs& a,
 unsigned char* dynamic_storage,unsigned words) {
 (void)words;
 const unsigned tid=threadIdx.x,wave=tid>>6,lane=tid&63u;
 const unsigned R=unsigned(a.down_weights.n),down_blocks=unsigned(a.down_weights.k)>>5;
 const unsigned parts=unsigned(a.partitions),down_per=down_blocks/parts;
 int4* lds_codes4=reinterpret_cast<int4*>(dynamic_storage);
 float* lds_scales=reinterpret_cast<float*>(lds_codes4+down_blocks*2u);
 float* down_lds=lds_scales+down_blocks;
 {
  const int4* g_codes4=reinterpret_cast<const int4*>(a.input_codes);
  for(unsigned i=tid;i<down_blocks*2u;i+=1024u)lds_codes4[i]=g_codes4[i];
  for(unsigned i=tid;i<down_blocks;i+=1024u)lds_scales[i]=a.input_scales[i];
 }
 __syncthreads();
 const unsigned col_in_wave=lane>>1,half_part=lane&1u;
 const unsigned col_tiles=R>>5,cta_tasks=parts*col_tiles;
 const unsigned short* w_scales=static_cast<const unsigned short*>(a.down_weights.scales);
 const unsigned down_stride=down_per*32u;
 unsigned buf=0u;
#pragma unroll 1
 for(unsigned cta_task=blockIdx.x;cta_task<cta_tasks;cta_task+=gridDim.x) {
  float* cur_lds=down_lds+(buf&1u)*down_stride;
  const unsigned partition=cta_task/col_tiles;
  const unsigned col_base=(cta_task%col_tiles)<<5;
  const unsigned first=partition*down_per;
  const bool has0=wave<down_per;
  const bool has1=(wave+16u)<down_per;
  int4 w0{},w1{};
  unsigned short s0=0u,s1=0u;
  if(has0) {
   const std::size_t rb0=std::size_t(first+wave)*R+col_base;
   w0=weightRead<int4>(a.down_weights.payload+rb0*32u+lane*16u);
   if(half_part==0u)s0=weightRead<unsigned short>(w_scales+rb0+col_in_wave);
  }
  if(has1) {
   const std::size_t rb1=std::size_t(first+wave+16u)*R+col_base;
   w1=weightRead<int4>(a.down_weights.payload+rb1*32u+lane*16u);
   if(half_part==0u)s1=weightRead<unsigned short>(w_scales+rb1+col_in_wave);
  }
  if(has0) {
   const unsigned g0=first+wave;
   const int4 a0=lds_codes4[g0*2u+half_part];
   int d=0;
   d=rocm_integer_contract::signedDot4(a0.x,w0.x,d);
   d=rocm_integer_contract::signedDot4(a0.y,w0.y,d);
   d=rocm_integer_contract::signedDot4(a0.z,w0.z,d);
   d=rocm_integer_contract::signedDot4(a0.w,w0.w,d);
   const int full=d+__shfl_xor(d,1,64);
   if(half_part==0u) {
    const float ws=__half2float(__ushort_as_half(s0));
    cur_lds[wave*32u+col_in_wave]=device_native_quantised_contract::singleScaleBlock(full,ws,lds_scales[g0]);
   }
  }
  if(has1) {
   const unsigned g1=first+wave+16u;
   const int4 a1=lds_codes4[g1*2u+half_part];
   int d=0;
   d=rocm_integer_contract::signedDot4(a1.x,w1.x,d);
   d=rocm_integer_contract::signedDot4(a1.y,w1.y,d);
   d=rocm_integer_contract::signedDot4(a1.z,w1.z,d);
   d=rocm_integer_contract::signedDot4(a1.w,w1.w,d);
   const int full=d+__shfl_xor(d,1,64);
   if(half_part==0u) {
    const float ws=__half2float(__ushort_as_half(s1));
    cur_lds[(wave+16u)*32u+col_in_wave]=device_native_quantised_contract::singleScaleBlock(full,ws,lds_scales[g1]);
   }
  }
  __syncthreads();
  if(tid<32u) {
   float sum=0.0f;
   for(unsigned b=0u;b<down_per;++b)sum=device_fp32_contract::add(sum,cur_lds[b*32u+tid]);
   a.down_partials[std::size_t(partition)*R+col_base+tid]=__fmul_rn(sum,1.0f);
  }
  ++buf;
 }
}

/**
 * @brief 1024B coalesced wave64 HC up with simultaneous payload+scale issue and parallel 4-branch fold.
 * @param a Original immutable source/public output declaration.
 * @param local Published local TwoTerm activation banks.
 * @param arena Exclusive dynamic LDS float bank.
 * @param output_demand Sealed graph/observation demand for public HCUp logits.
 */
__device__ __forceinline__ void fastCoalescedUp(const PersistentHCReadArgs& a,
 const FullNativeHCFFNPrefixScratch& local,float* arena,FullNativeHCFFNOutputDemand output_demand) {
 const unsigned tid=threadIdx.x,wave=tid>>6,lane=tid&63u;
 const unsigned H=unsigned(a.norm.hidden),up_groups=unsigned(a.up.weights.blocks_per_row);
 const unsigned channel_tiles=H>>5,total_tasks=4u*up_groups,up_n=4u*H;
 const unsigned col_in_wave=lane>>1,half_part=lane&1u;
 const unsigned short* w_scales=static_cast<const unsigned short*>(a.up.weights.scales);
 const int4* p_codes4=reinterpret_cast<const int4*>(local.primary);
 const int4* r_codes4=reinterpret_cast<const int4*>(local.residual);
 const unsigned plane_floats=total_tasks*32u;
 unsigned buf=0u;
#pragma unroll 1
 for(unsigned tile=blockIdx.x;tile<channel_tiles;tile+=gridDim.x) {
  const unsigned col_base=tile<<5;
  float* primary_lds=arena+(buf&1u)*(2u*plane_floats);
  float* residual_lds=primary_lds+plane_floats;
  for(unsigned task=wave;task<total_tasks;task+=16u) {
   const unsigned br=task/up_groups,blk=task%up_groups;
   const std::size_t rb=std::size_t(blk)*up_n+std::size_t(br)*H+col_base;
   const int4 wv=weightRead<int4>(a.up.weights.payload+rb*32u+lane*16u);
   unsigned short sv=0u;
   if(half_part==0u)sv=weightRead<unsigned short>(w_scales+rb+col_in_wave);
   const int4 p0 = prefixLdsLoadInt4(prefixAsLds<const PrefixRawInt4>(p_codes4), blk*2u+half_part);
   const int4 r0 = prefixLdsLoadInt4(prefixAsLds<const PrefixRawInt4>(r_codes4), blk*2u+half_part);
   int dp=0,dr=0;
   dp=rocm_integer_contract::signedDot4(p0.x,wv.x,dp);
   dr=rocm_integer_contract::signedDot4(r0.x,wv.x,dr);
   dp=rocm_integer_contract::signedDot4(p0.y,wv.y,dp);
   dr=rocm_integer_contract::signedDot4(r0.y,wv.y,dr);
   dp=rocm_integer_contract::signedDot4(p0.z,wv.z,dp);
   dr=rocm_integer_contract::signedDot4(r0.z,wv.z,dr);
   dp=rocm_integer_contract::signedDot4(p0.w,wv.w,dp);
   dr=rocm_integer_contract::signedDot4(r0.w,wv.w,dr);
   const int fp=dp+__shfl_xor(dp,1,64),fr=dr+__shfl_xor(dr,1,64);
   if(half_part==0u) {
    const float ws=__half2float(__ushort_as_half(sv));
    primary_lds[task*32u+col_in_wave]=device_native_quantised_contract::singleScaleBlock(fp,ws,local.primary_scales[blk]);
    residual_lds[task*32u+col_in_wave]=device_native_quantised_contract::singleScaleBlock(fr,ws,local.residual_scales[blk]);
   }
  }
  __syncthreads();
  if(tid<128u) {
   const unsigned branch=tid>>5,col=tid&31u;
   float p=0.0f,q=0.0f;
   for(unsigned block=0u;block<up_groups;++block) {
    const unsigned idx=(branch*up_groups+block)*32u+col;
    p=device_fp32_contract::add(p,primary_lds[idx]);
    q=device_fp32_contract::add(q,residual_lds[idx]);
   }
   const float value=device_fp32_contract::add(p,q);
   if(output_demand.contains(FullNativeHCFFNOutputRole::HCUp))
    prefixAsGlobal(a.up.output)[std::size_t(branch)*H+col_base+col]=value;
   primary_lds[branch*32u+col]=1.0f/(1.0f+expf(-value));
   residual_lds[branch*32u+col]=weightRead<float>(a.norm.output+std::size_t(branch)*H+col_base+col);
  }
  __syncthreads();
  if(tid<32u) {
   float mixed=0.0f;
#pragma unroll
   for(unsigned branch=0u;branch<4u;++branch)
    mixed=fmaf(primary_lds[branch*32u+tid],residual_lds[branch*32u+tid],mixed);
   prefixAsGlobal(a.mixed)[col_base+tid]=mixed*0.25f;
  }
  ++buf;
 }
}
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx906__)
__device__ __forceinline__ float prefixQuietMax32(float4 values) {
 const auto qabs = [] __device__ (float x) {
  float out;
  asm("v_max_f32_e64 %0, |%1|, |%1|" : "=v"(out) : "v"(x));
  return out;
 };
 const auto compRoot = [] __device__ (float c) {
  float r;
  asm volatile("s_nop 1\n"
               "v_max_f32_dpp %0, %1, %1 row_shl:4 row_mask:0xf bank_mask:0x5\n"
               "v_max_f32_dpp %0, %1, %1 row_shr:4 row_mask:0xf bank_mask:0xa\n"
               "s_nop 1\n"
               "v_max_f32_dpp %0, %0, %0 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf\n"
               "s_nop 1\n"
               "v_max_f32_dpp %0, %0, %0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"
               : "=&v"(r) : "v"(c));
  return r;
 };
 const float x = compRoot(qabs(values.x)), y = compRoot(qabs(values.y));
 const float z = compRoot(qabs(values.z)), w = compRoot(qabs(values.w));
 return fmaxf(fmaxf(x, z), fmaxf(y, w));
}
#endif
__device__ __forceinline__ void prefixNormQ8Four(float4 value, signed char* output, float* scale_output, unsigned lane) {
 float maximum;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx906__)
 maximum = prefixQuietMax32(value);
#else
 float a=fabsf(value.x),b=fabsf(value.y),c=fabsf(value.z),d=fabsf(value.w);
#pragma unroll
 for(int offset=4;offset>0;offset>>=1) {
  a=fmaxf(a,__shfl_down(a,offset,8));b=fmaxf(b,__shfl_down(b,offset,8));
  c=fmaxf(c,__shfl_down(c,offset,8));d=fmaxf(d,__shfl_down(d,offset,8));
 }
 maximum=__shfl(fmaxf(fmaxf(a,c),fmaxf(b,d)),0,8);
#endif
 const float scale=device_q8_activation_contract::publishPositiveFiniteHalfScale(
  device_q8_activation_contract::boundedScale(maximum));
 const float inverse=1.0f/scale;
 if(lane==0u)*scale_output=scale;
 const unsigned q0=unsigned(decode_equivalent::quantizeQ8(value.x,inverse))&255u;
 const unsigned q1=unsigned(decode_equivalent::quantizeQ8(value.y,inverse))&255u;
 const unsigned q2=unsigned(decode_equivalent::quantizeQ8(value.z,inverse))&255u;
 const unsigned q3=unsigned(decode_equivalent::quantizeQ8(value.w,inverse))&255u;
 const unsigned packed=q0|(q1<<8)|(q2<<16)|(q3<<24);
 __builtin_memcpy(output+4u*lane,&packed,sizeof(packed));
}
struct PrefixLdsBranchInput {
 PrefixLdsPtr<const float> cached;
 __device__ __forceinline__ float retained(unsigned ch) const { return cached[ch]; }
 __device__ __forceinline__ float publish(unsigned ch) const { return cached[ch]; }
};
struct PrefixLdsRegisterNormQ8Output {
 PrefixLdsPtr<const PrefixRawFloat4> post_lds4;
 PrefixLdsPtr<const PrefixRawFloat4> gain_lds4;
 signed char* codes;
 float* scales;
 template<class Input, class Output>
 __device__ __forceinline__ void publish(
  const Input&, const float*, Output output,
  int hidden, float epsilon, std::uint64_t base, float,
  const float* completed_total, unsigned lane, bool) const {
  using namespace qwen4exp::branch_norm_native;
  auto* out_g = prefixAsGlobal(output);
  auto* codes_g = prefixAsGlobal(codes);
  auto* scales_g = prefixAsGlobal(scales);
  const float inverse = normalizationInverse(completed_total[0], hidden, epsilon);
  for (unsigned quad = lane; quad < unsigned(hidden / 4); quad += 1024u) {
   const float4 p4 = prefixLdsLoadFloat4(post_lds4, quad);
   const float4 g4 = prefixLdsLoadFloat4(gain_lds4, quad);
   float4 value;
   value.x = normalizedElement(p4.x, inverse, g4.x, OriginalNormalizedProducts{});
   value.y = normalizedElement(p4.y, inverse, g4.y, OriginalNormalizedProducts{});
   value.z = normalizedElement(p4.z, inverse, g4.z, OriginalNormalizedProducts{});
   value.w = normalizedElement(p4.w, inverse, g4.w, OriginalNormalizedProducts{});
   PrefixRawFloat4 out4;
   out4[0] = value.x; out4[1] = value.y; out4[2] = value.z; out4[3] = value.w;
   reinterpret_cast<PrefixGlobalPtr<PrefixRawFloat4>>(out_g + base)[quad] = out4;
   prefixNormQ8Four(value, reinterpret_cast<signed char*>(codes_g + (quad >> 3) * 32u),
                    reinterpret_cast<float*>(scales_g + (quad >> 3)), quad & 7u);
  }
 }
};
__device__ __forceinline__ float prefixConsumeHCDownPacket(
 const Packet& packet, const signed char* input, float scale) {
 const int4 x0 = weightRead<int4>(input);
 const int4 x1 = weightRead<int4>(input + 16u);
 int dot = 0;
 dot = rocm_integer_contract::signedDot4(x0.x, packet.low.x, dot);
 dot = rocm_integer_contract::signedDot4(x0.y, packet.low.y, dot);
 dot = rocm_integer_contract::signedDot4(x0.z, packet.low.z, dot);
 dot = rocm_integer_contract::signedDot4(x0.w, packet.low.w, dot);
 dot = rocm_integer_contract::signedDot4(x1.x, packet.high.x, dot);
 dot = rocm_integer_contract::signedDot4(x1.y, packet.high.y, dot);
 dot = rocm_integer_contract::signedDot4(x1.z, packet.high.z, dot);
 dot = rocm_integer_contract::signedDot4(x1.w, packet.high.w, dot);
 const float ws = __half2float(__ushort_as_half(static_cast<unsigned short>(packet.scale_bits)));
 return device_native_quantised_contract::singleScaleBlock(dot, ws, scale);
}
__device__ __forceinline__ float prefixOrderedAppendLds(
 float sum, PrefixLdsPtr<const float> values, unsigned stride, unsigned column, unsigned count) {
 unsigned i = 0u;
#pragma unroll 1
 for (; i + 8u <= count; i += 8u) {
  float a = values[(i + 0u) * stride + column], b = values[(i + 1u) * stride + column];
  float c = values[(i + 2u) * stride + column], d = values[(i + 3u) * stride + column];
  float e = values[(i + 4u) * stride + column], f = values[(i + 5u) * stride + column];
  float g = values[(i + 6u) * stride + column], h = values[(i + 7u) * stride + column];
#if defined(__HIP_DEVICE_COMPILE__)
  asm volatile("" : "+v"(a), "+v"(b), "+v"(c), "+v"(d), "+v"(e), "+v"(f), "+v"(g), "+v"(h));
#endif
  sum = device_fp32_contract::add(sum, a); sum = device_fp32_contract::add(sum, b);
  sum = device_fp32_contract::add(sum, c); sum = device_fp32_contract::add(sum, d);
  sum = device_fp32_contract::add(sum, e); sum = device_fp32_contract::add(sum, f);
  sum = device_fp32_contract::add(sum, g); sum = device_fp32_contract::add(sum, h);
 }
 if (i + 4u <= count) {
  float a = values[(i + 0u) * stride + column], b = values[(i + 1u) * stride + column];
  float c = values[(i + 2u) * stride + column], d = values[(i + 3u) * stride + column];
#if defined(__HIP_DEVICE_COMPILE__)
  asm volatile("" : "+v"(a), "+v"(b), "+v"(c), "+v"(d));
#endif
  sum = device_fp32_contract::add(sum, a); sum = device_fp32_contract::add(sum, b);
  sum = device_fp32_contract::add(sum, c); sum = device_fp32_contract::add(sum, d);
  i += 4u;
 }
 if (__builtin_expect(i < count, 0)) {
#pragma unroll 1
  for (; i < count; ++i)
   sum = device_fp32_contract::add(sum, values[i * stride + column]);
 }
 return sum;
}
struct PrefixHCUpPacket { int4 lo, hi; std::uint16_t s_bits; };
__device__ __forceinline__ PrefixHCUpPacket prefixHCUpLoadPacket(
 const std::uint8_t* up_pay, const std::uint16_t* up_sc, unsigned up_n, unsigned column, unsigned block) {
 const std::size_t linear = std::size_t(block) * up_n + column;
 const std::uint8_t* pay = up_pay + linear * 32u;
 return {weightRead<int4>(pay), weightRead<int4>(pay + 16u), weightRead<std::uint16_t>(up_sc + linear)};
}
__device__ __forceinline__ void prefixHCUpConsumePacket(
 const PrefixHCUpPacket& pkt, unsigned block,
 PrefixLdsPtr<const PrefixRawInt4> p_codes4, PrefixLdsPtr<const PrefixRawInt4> r_codes4,
 float ps, float rs, float& p_out, float& q_out) {
 const int4 pa0 = prefixLdsLoadInt4(p_codes4, block * 2u);
 const int4 ra0 = prefixLdsLoadInt4(r_codes4, block * 2u);
 int dp = 0, dq = 0;
 dp = rocm_integer_contract::signedDot4(pa0.x, pkt.lo.x, dp);
 dq = rocm_integer_contract::signedDot4(ra0.x, pkt.lo.x, dq);
 dp = rocm_integer_contract::signedDot4(pa0.y, pkt.lo.y, dp);
 dq = rocm_integer_contract::signedDot4(ra0.y, pkt.lo.y, dq);
 dp = rocm_integer_contract::signedDot4(pa0.z, pkt.lo.z, dp);
 dq = rocm_integer_contract::signedDot4(ra0.z, pkt.lo.z, dq);
 dp = rocm_integer_contract::signedDot4(pa0.w, pkt.lo.w, dp);
 dq = rocm_integer_contract::signedDot4(ra0.w, pkt.lo.w, dq);
 const int4 pa1 = prefixLdsLoadInt4(p_codes4, block * 2u + 1u);
 const int4 ra1 = prefixLdsLoadInt4(r_codes4, block * 2u + 1u);
 dp = rocm_integer_contract::signedDot4(pa1.x, pkt.hi.x, dp);
 dq = rocm_integer_contract::signedDot4(ra1.x, pkt.hi.x, dq);
 dp = rocm_integer_contract::signedDot4(pa1.y, pkt.hi.y, dp);
 dq = rocm_integer_contract::signedDot4(ra1.y, pkt.hi.y, dq);
 dp = rocm_integer_contract::signedDot4(pa1.z, pkt.hi.z, dp);
 dq = rocm_integer_contract::signedDot4(ra1.z, pkt.hi.z, dq);
 dp = rocm_integer_contract::signedDot4(pa1.w, pkt.hi.w, dp);
 dq = rocm_integer_contract::signedDot4(ra1.w, pkt.hi.w, dq);
 const float ws = __half2float(__ushort_as_half(pkt.s_bits));
 p_out = device_native_quantised_contract::singleScaleBlock(dp, ws, ps);
 q_out = device_native_quantised_contract::singleScaleBlock(dq, ws, rs);
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
 * @details Uses vectorized AS1/AS3 LDS BranchRMSNorm (WaveOwnsBothRoots + DPP max),
 * FastDiv16 idle-CTA-prefetched single-pass HC Down (zero down_partials round-trip),
 * and balanced 100% single-wave e0..e1 HC Up + Mix across arbitrary gridDim.x while
 * preserving general continuations for large geometries and exact OutputDemand semantics.
 */
template<class Grid>
__device__ __forceinline__ void fullNativeHCFFNPrefix(const FullNativeHCFFNPrefixArgs& prefix,
 Grid& grid,FullNativeHCFFNPrefixScratch& local,unsigned char* dynamic_storage,std::size_t dynamic_bytes) {
 using namespace full_hc_native;
 const auto& a=prefix.hc;
 const unsigned tid=threadIdx.x;
 const unsigned bid=blockIdx.x;
 const unsigned gdim=gridDim.x;
 const unsigned H=prefixWaveUniform(unsigned(a.norm.hidden));
 const unsigned R=prefixWaveUniform(unsigned(a.down_weights.n));
 if(blockDim.x!=1024||!dynamic_bytes||dynamic_bytes>32768||(dynamic_bytes&15u)||!dynamic_storage)__builtin_trap();
 float* arena=reinterpret_cast<float*>(dynamic_storage);
 auto arena_lds=prefixAsLds<float>(dynamic_storage);
 const unsigned words=prefixWaveUniform(unsigned(dynamic_bytes>>2));
 const unsigned total_parts=prefixWaveUniform(unsigned(a.partitions));
 const unsigned down_blocks=prefixWaveUniform(unsigned(a.down_weights.k)>>5);
 (void)down_blocks;
 // Preserve idle-CTA early prefetch of HC-down packets during vectorized fast_norm
 // and avoid the global down_partials write+reload round-trip (saves ~12 us vs fastCoalescedDown).
 const bool fast_down=false;
 {
  const auto owner=fullHCDownOwnership(bid,gdim,R);
  DownPanel panel{};
  Packet first{},second{};
  const bool early=!fast_down&&gdim>4u&&bid>=4u;
  if(!fast_down&&owner.columns)panel=downPanel(a,owner.columns,words);
  if(early) {first=firstPacket(a,owner,panel,tid);second=firstPacket(a,owner,panel,tid+1024u);}
  const int norm_rows=prefixWaveUniform(a.norm.extent.liveRows?*prefixAsGlobal(a.norm.extent.liveRows):1);
  if(norm_rows<0||norm_rows>1)__builtin_trap();
  const bool fast_norm=prefixWaveUniform((H&3u)==0u&&words>=256u+2u*H);
  if(fast_norm) {
   auto post_lds=arena_lds+256u;
   auto gain_lds=post_lds+H;
   auto post_lds4=prefixAsLds<PrefixRawFloat4>(post_lds);
   auto gain_lds4=prefixAsLds<PrefixRawFloat4>(gain_lds);
   const bool has_res=prefixWaveUniform(prefix.write.residual!=nullptr);
#pragma unroll 1
   for(unsigned task=bid;task<4u;task+=gdim) {
    signed char* codes=a.input_codes+std::size_t(task)*H;
    float* scales=a.input_scales+std::size_t(task)*(H>>5);
    if(__builtin_expect(norm_rows!=0,1)) {
     const auto base=std::size_t(task)*H;
     const float* gain_p=a.norm.gain+base;
     const float gate=has_res?qwen4exp::branch_norm_native::residualGate(
      weightRead<float>(prefix.write.gateLogits+task)):0.0f;
     const float* src_p=(has_res?prefix.write.residual:a.norm.input)+base;
     const float* blk_p=prefix.write.block;
     float* out_p=prefix.write.output+base;
     const unsigned quads=H>>2;
#pragma unroll 1
     for(unsigned q=tid;q<quads;q+=1024u) {
      const float4 g4=weightRead<float4>(gain_p+4u*q);
      float4 p4=weightRead<float4>(src_p+4u*q);
      if(has_res) {
       const float4 b4=weightRead<float4>(blk_p+4u*q);
       p4.x=qwen4exp::branch_norm_native::residualValue(p4.x,gate,b4.x);
       p4.y=qwen4exp::branch_norm_native::residualValue(p4.y,gate,b4.y);
       p4.z=qwen4exp::branch_norm_native::residualValue(p4.z,gate,b4.z);
       p4.w=qwen4exp::branch_norm_native::residualValue(p4.w,gate,b4.w);
       PrefixRawFloat4 ro;
       ro[0]=p4.x;ro[1]=p4.y;ro[2]=p4.z;ro[3]=p4.w;
       reinterpret_cast<PrefixGlobalPtr<PrefixRawFloat4>>(prefixAsGlobal(out_p))[q]=ro;
      }
      prefixLdsStoreFloat4(post_lds4,q,p4);
      prefixLdsStoreFloat4(gain_lds4,q,g4);
     }
     workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
     qwen4exp::branch_norm_native::taskFromBranchInput<
      PrefixLdsBranchInput,float*,PrefixLdsRegisterNormQ8Output,
      qwen4exp::branch_norm_native::SquareSumPublication::WaveOwnsBothRoots>(
      PrefixLdsBranchInput{post_lds},gain_p,a.norm.output,H,a.norm.epsilon,task,
      arena,tid,tid<256u,
      PrefixLdsRegisterNormQ8Output{post_lds4,gain_lds4,codes,scales});
    } else {
     for(unsigned quad=tid;quad<(H>>2);quad+=1024u) {
      const float4 val=weightRead<float4>(a.norm.output+std::size_t(task)*H+4u*quad);
      prefixNormQ8Four(val,codes+(quad>>3)*32u,scales+(quad>>3),quad&7u);
     }
    }
    if(task+gdim<4u)workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
   }
  } else {
   for(unsigned task=bid;task<4u;task+=gdim) {
    signed char* codes=a.input_codes+std::size_t(task)*H;
    float* scales=a.input_scales+std::size_t(task)*(H/32);
    if(norm_rows) {
     const unsigned cache=unsigned(fullHCMin(H,((words-256)/2)/32*32));
     float* post_cache=arena+256;float* gain_cache=post_cache+cache;
     const auto base=std::size_t(task)*H;
     const float* published_post=nullptr;
     if(prefix.write.residual) {
      const float gate=qwen4exp::branch_norm_native::residualGate(prefix.write.gateLogits[task]);
      const qwen4exp::branch_norm_native::ResidualBranchInput original{
       prefix.write.residual+base,prefix.write.block,prefix.write.output+base,gate};
      for(unsigned channel=tid;channel<H;channel+=1024) {
       const float gain=a.norm.gain[base+channel],post=original.publish(channel);
       if(channel<cache){post_cache[channel]=post;gain_cache[channel]=gain;}
      }
      published_post=prefix.write.output+base;
     } else {
      for(unsigned channel=tid;channel<H;channel+=1024) {
       const float gain=a.norm.gain[base+channel],post=a.norm.input[base+channel];
       if(channel<cache){post_cache[channel]=post;gain_cache[channel]=gain;}
      }
      published_post=a.norm.input+base;
     }
     __syncthreads();
     const StagedBranchInput input{post_cache,published_post,cache};
     qwen4exp::branch_norm_native::taskFromBranchInput(input,a.norm.gain+base,a.norm.output,
      H,a.norm.epsilon,task,arena,tid,tid<256,
      RegisterNormQ8Output{gain_cache,a.norm.gain+base,cache,codes,scales});
     __syncthreads();
    } else {
     for(unsigned quad=tid;quad<H/4;quad+=1024) {
      const auto first_value=std::size_t(task)*H+4*quad;
      const float4 value{a.norm.output[first_value],a.norm.output[first_value+1],
       a.norm.output[first_value+2],a.norm.output[first_value+3]};
      normQ8Four(value,codes+(quad/8)*32,scales+quad/8,quad%8);
     }
     __syncthreads();
    }
   }
  }
  grid.sync(); // All public normalized/Q8 words; old incoming write gates retired.
  if(__builtin_expect(bid<4u&&a.injection.weights&&a.injection.rows.activeRows(),0))
   injection(a,arena);
  if(fast_down)fastCoalescedDown(a,dynamic_storage,words);
  else down(a,owner,panel,first,second,early,arena,words);
 }
 grid.sync(); // All raw outputs; input_codes/scales may now become the shorter up banks.
 const bool publish=prefixWaveUniform(a.publication.rows.activeRows()!=0);
 if(tid<R) {
  float raw_val;
  if(fast_down) {
   const float* dp=a.down_partials+tid;
   if(total_parts==1u)raw_val=weightRead<float>(dp);
   else {
    raw_val=0.0f;
#pragma unroll 1
    for(unsigned p=0u;p<total_parts;++p)
     raw_val=device_fp32_contract::add(raw_val,weightRead<float>(dp+std::size_t(p)*R));
   }
   if(bid==0u)prefixAsGlobal(a.raw_down)[tid]=raw_val;
  } else {
   raw_val=weightRead<float>(a.raw_down+tid);
  }
  if(__builtin_expect(publish,1)) {
   auto output=a.publication;
   output.activated=local.activated;output.primary=local.primary;output.residual=local.residual;
   output.primaryScales=local.primary_scales;output.residualScales=local.residual_scales;
   hc_down_publication::publish(raw_val,output,tid,
    hc_down_publication::NativePositiveHalfScalePublication{});
  } else {local.primary[tid]=a.up.primary[tid];local.residual[tid]=a.up.residual[tid];}
 }
 if(__builtin_expect(!publish&&tid<(R>>5),0)) {
  local.primary_scales[tid]=a.up.primaryScales[tid];
  local.residual_scales[tid]=a.up.residualScales[tid];
 }
 workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
 if(__builtin_expect(publish&&bid==0u,0)) {
  if(tid<R){prefixAsGlobal(a.publication.primary)[tid]=local.primary[tid];prefixAsGlobal(a.publication.residual)[tid]=local.residual[tid];}
  if(tid<(R>>5)){prefixAsGlobal(a.publication.primaryScales)[tid]=local.primary_scales[tid];
   prefixAsGlobal(a.publication.residualScales)[tid]=local.residual_scales[tid];}
 }
 const bool up_active=prefixWaveUniform(
  (a.up.rows.countOwner()==a.publication.rows.countOwner())?publish:(a.up.rows.activeRows()!=0));
 if(__builtin_expect(up_active,1)) {
  const unsigned up_groups=prefixWaveUniform(unsigned(a.up.weights.blocks_per_row));
  const bool fast_up=prefixWaveUniform((H&31u)==0u&&up_groups>0u&&words>=2u*4u*up_groups*32u*2u);
  if(fast_up)fastCoalescedUp(a,local,arena,prefix.output_demand);
  else up(a,local,arena,words,prefix.output_demand);
 }
 grid.sync(); // Mixed/injection ready; all replicated raw readers and local up users retired.
 if(prefix.output_demand.contains(FullNativeHCFFNOutputRole::HCActivated)&&publish&&bid==0u&&tid<R)
  prefixAsGlobal(a.publication.activated)[tid]=local.activated[tid];
 workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
}
} // namespace llaminar2::rocm
