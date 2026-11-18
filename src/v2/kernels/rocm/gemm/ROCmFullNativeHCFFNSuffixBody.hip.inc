/**
 * @file ROCmFullNativeHCFFNSuffixBody.hip.inc
 * @brief Tuned native-column router/expert suffix over the original production owners.
 *
 * This body is composed by the existing full native entry after HC mixed-ready.
 * All live global bytes are borrowed from NativeLayoutRouterFFNArgs; there is no
 * module state, extra weight owner, allocation, launch or host routing mirror.
 * A bounded coefficient window preserves the measured full-H panel at H2560
 * while larger reductions retain the same increasing K32/P arithmetic. The
 * existing acquired runtime publisher and public FP32 route ABI remain intact.
 * Include after ROCmNativeLayoutRouterFFNBody.hip.inc, whose canonical decoder,
 * field, ordered-fold and publication helpers are the numerical authority.
 * Physical task/panel layouts derive from the Castagna adaptation; see
 * ROCmCastagnaT1.LICENSE for the original attribution.
 */
#pragma once
#include "ROCmNativeLayoutRouterFFN.h"
#include "kernels/common/DeviceQ8ActivationNumericalContract.h"
#include "kernels/common/DeviceOrderedRouteFold.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace llaminar2::rocm {
/** @brief Load a captured policy at its original first-kernarg field offsets. */
#define LLAMINAR_TUNED_SUFFIX_POLICY(Field) \
    native_layout_ffn::phasePolicy<offsetof(NativeLayoutRouterFFNArgs,ffn)+offsetof(ResidentMoEFFNArgs,Field)>()
/** @brief Load only this original role's sealed physical execution mask. */
#define LLAMINAR_TUNED_SUFFIX_MASK(Field) \
    phaseKernargField<NativeLayoutRouterFFNArgs,decltype(ResidentMoEFFNExecutionMasks::Field), \
        offsetof(NativeLayoutRouterFFNArgs,ffn)+offsetof(ResidentMoEFFNArgs,execution)+ \
        offsetof(ResidentMoEFFNExecutionMasks,Field)>()
/** @brief Pure extent result; admission remains owned by the original full launcher. */
struct FullNativeHCFFNSuffixGeometry {
    std::size_t dynamic_shared_bytes; ///< Maximum of disjoint hidden/GU and down lifetimes.
    unsigned panel_blocks; ///< Bounded K32 window; zero only for unions without Q4.
    unsigned panel_rows; ///< Native4..64 rows, or the minimum counted linear one-row panel.
    std::uint64_t useful_ctas; ///< Live task ceiling, never an occupancy estimate.
};
/**
 * @brief Size the bounded native suffix without reading device state or allocating.
 * @param args Original cold-resolved positive projection/source policies.
 * @param mask Exact original physical decoder union, not a source-type guess.
 * @param maximum_dynamic_bytes Cached device LDS limit minus the actual static union.
 * @return Complete dynamic LDS extent and the fixed panel window in K32 blocks.
 * @details Existing static NativeLayoutFFNPhaseScratch remains separate. The
 * hidden40-byte-per-block SoA prefix survives GU; its phase arena then reuses
 * the entire dynamic region for activated Q8 and at most64 completed columns.
 * The two original gate/up records follow the bounded Q4 panel. Non-Q4 unions
 * require only those records. Shape validation and overflow admission belong
 * to the existing cold owner before this pure helper is called.
 */
inline FullNativeHCFFNSuffixGeometry fullNativeHCFFNSuffixGeometry(
    const NativeLayoutRouterFFNArgs& args,std::uint32_t mask,std::size_t maximum_dynamic_bytes) noexcept {
    const auto& a=args.ffn;
    const std::size_t H=std::size_t(a.hidden_columns),F=std::size_t(a.routed_columns),
        S=std::size_t(a.shared_columns),routes=std::size_t(a.top_k);
    const unsigned blocks=unsigned(H/32);
    const unsigned wanted_blocks=std::min(80u,((blocks+15u)/16u)*16u);
    const bool panel=(mask&a.execution.routed_gate&(1u<<unsigned(NATIVE_QUANTISED_Q4_1)))!=0;
    const std::size_t hidden=(H+(H/32)*8+15u)&~std::size_t(15u);
    const std::size_t records=2*sizeof(native_layout_ffn::PhaseProjection);
    const std::size_t available=maximum_dynamic_bytes>hidden+records?maximum_dynamic_bytes-hidden-records:0;
    unsigned panel_rows=64,panel_blocks=0;
    if(panel) {
        // Preserve the native16-block bank permutation while reducing only
        // physical window/column capacity. The final one-row linear panel
        // retains the same counted publication/consume algorithm at minimum LDS.
        while(panel_rows>=4) {
            const auto fit=unsigned(std::min<std::size_t>(wanted_blocks,available/(2*panel_rows*sizeof(std::uint32_t))));
            panel_blocks=(fit/16)*16;
            if(panel_blocks)break;
            panel_rows/=2;
        }
        if(!panel_blocks) {
            panel_rows=1;
            const std::size_t raw_hidden=H+(H/32)*8;
            const std::size_t linear_available=maximum_dynamic_bytes>raw_hidden+records?
                maximum_dynamic_bytes-raw_hidden-records:0;
            // Returning a real one-block minimum makes insufficient LDS fail
            // cold admission precisely; it never skips or changes coefficient work.
            panel_blocks=unsigned(std::max<std::size_t>(1,std::min<std::size_t>(wanted_blocks,
                linear_available/(2*sizeof(std::uint32_t)))));
        }
    }
    const std::size_t panel_bytes=std::size_t(2)*panel_blocks*panel_rows*sizeof(std::uint32_t);
    const std::size_t gu=(panel_rows==1||!panel_blocks?H+(H/32)*8:hidden)+panel_bytes+records;
    const std::size_t values=routes*F+S;
    const std::size_t down=values+(values/32)*8+(routes+1)*64*sizeof(float);
    const std::uint64_t router=(std::uint64_t(a.experts)+15)/16;
    const std::uint64_t gu_work=(std::uint64_t(a.routed_columns)+3)/4*routes;
    const std::uint64_t outputs=(std::uint64_t(a.hidden_columns)+63)/64;
    return {std::max(gu,down),panel_blocks,panel_rows,std::max({router,gu_work,outputs,std::uint64_t{1}})};
}
namespace full_native_ffn {
using namespace native_layout_ffn;
/**
 * @brief View of the SAME immutable shared projections sealed by full preparation.
 * @details Carries only the original argument reference; no arbitrary descriptor
 * pointer, changed table, extra epoch or new allocation can enter this interface.
 */
class PreparedFullNativeSharedSource final {
    const ResidentMoEFFNArgs& args_;
public:
    /** @param args Original prepared full arguments with retained table publications. */
    __device__ explicit PreparedFullNativeSharedSource(const ResidentMoEFFNArgs& args):args_(args){}
    /** @param role Semantic projection role. @return SAME sealed original table slot. */
    __device__ __forceinline__ PublishedNativeProjection projection(PublishedNativeProjectionRole role) const {
        switch(role) {
        case PublishedNativeProjectionRole::Gate:return PublishedNativeProjection(args_.shared_gate);
        case PublishedNativeProjectionRole::Up:return PublishedNativeProjection(args_.shared_up);
        case PublishedNativeProjectionRole::Down:return PublishedNativeProjection(args_.shared_down);
        }
        __builtin_trap();
    }
};

/**
 * @brief Read immutable prepared-weight bytes in their admitted global address space.
 * @tparam T Original raw load type; memcpy preserves unaligned native block strides.
 * @param pointer Original table-owned global weight span. @return Identical bytes.
 * @pre This interface accepts only original prepared weights, never LDS/activation storage.
 */
template<class T>
__device__ __forceinline__ T globalWeight(const void* pointer) {
    const auto* source=reinterpret_cast<const unsigned char __attribute__((address_space(1)))*>(
        reinterpret_cast<std::uintptr_t>(pointer));
    T value;__builtin_memcpy(&value,source,sizeof(T));return value;
}
/** @brief Global-only weight issue injected into the existing canonical preload/role loop. */
struct GlobalWeightReader {
    /** @brief Borrow exactly the original scalar width from admitted global weight storage.
     * @tparam T Original type. @param pointer Original immutable source. @return Identical bits. */
    template<class T> __device__ __forceinline__ static T load(const T* pointer) {return globalWeight<T>(pointer);}
};
/** @brief Preserve the winner's fixed finite-half five-Newton reciprocal. */
__device__ __forceinline__ float preparedScaleReciprocal(float scale) {
    namespace fp=device_fp32_contract;
    const auto bits=fp::bitsFromFloat(scale);
    const int exponent=int((bits>>23)&255u)-127;
    const float normalized=fp::floatFromBits((127u<<23)|(bits&0x007fffffu));
    const auto inverse=fp::bitsFromFloat(fp::reciprocalOneToTwo(normalized));
    return fp::floatFromBits((unsigned(int((inverse>>23)&255u)-exponent)<<23)|(inverse&0x007fffffu));
}
/**
 * @brief Publish four original values with the qualified32-leaf max and exact sum.
 * @param value Four adjacent original rounded values. @param linear First scalar index.
 * @param codes Caller-owned aligned byte bank. @param scales Original FP32 scale bank.
 * @param sums Exact original int32 signed-code sums.
 * @pre Eight peers own one complete32-value block; caller strides are32-aligned.
 * @details Scalar conversion delegates to the one prepared-inverse contract.
 * Only integer code sums regroup, bounded by4064; no FP sum is reassociated.
 */
__device__ __forceinline__ void quantizeFour(float4 value,std::size_t linear,
    int8_t* codes,float* scales,int32_t* sums) {
    using namespace moe_wave_contract;
    float a=fabsf(value.x),b=fabsf(value.y),c=fabsf(value.z),d=fabsf(value.w);
#pragma unroll
    for(int mask=4;mask>0;mask>>=1) {
        a=fmaxf(a,quantizerXor32<QuantizerXorPolicy::TargetNative>(a,mask));
        b=fmaxf(b,quantizerXor32<QuantizerXorPolicy::TargetNative>(b,mask));
        c=fmaxf(c,quantizerXor32<QuantizerXorPolicy::TargetNative>(c,mask));
        d=fmaxf(d,quantizerXor32<QuantizerXorPolicy::TargetNative>(d,mask));
    }
    const float maximum=fmaxf(fmaxf(a,c),fmaxf(b,d));
    const float scale=device_q8_activation_contract::publishPositiveFiniteHalfScale(
        device_q8_activation_contract::boundedScale(maximum));
    const float inverse=preparedScaleReciprocal(scale);
    const int q0=device_q8_activation_contract::quantizePreparedInverse(value.x,inverse);
    const int q1=device_q8_activation_contract::quantizePreparedInverse(value.y,inverse);
    const int q2=device_q8_activation_contract::quantizePreparedInverse(value.z,inverse);
    const int q3=device_q8_activation_contract::quantizePreparedInverse(value.w,inverse);
    const auto packed=unsigned(uint8_t(q0))|(unsigned(uint8_t(q1))<<8)|
        (unsigned(uint8_t(q2))<<16)|(unsigned(uint8_t(q3))<<24);
    __builtin_memcpy(codes+linear,&packed,sizeof(packed));
    int sum=(q0+q2)+(q1+q3);
#pragma unroll
    for(int mask=4;mask>0;mask>>=1)sum+=quantizerXor32<QuantizerXorPolicy::TargetNative>(sum,mask);
    if((threadIdx.x&7u)==0){scales[linear/32]=scale;sums[linear/32]=sum;}
}
/** @brief Load only four real values; no stronger public-pointer alignment is imposed. */
__device__ __forceinline__ float4 valuesFour(const float* source,std::size_t first) {
    return {source[first],source[first+1],source[first+2],source[first+3]};
}
/** @brief Publish all real32-value blocks and complete their original CTA edge. */
__device__ __forceinline__ void quantizeRow(const float* source,unsigned count,
    int8_t* codes,float* scales,int32_t* sums) {
    for(unsigned quad=threadIdx.x;quad<count/4;quad+=1024)
        quantizeFour(valuesFour(source,std::size_t(quad)*4),std::size_t(quad)*4,codes,scales,sums);
    workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
}
/** @brief One immutable original router pair; no rounding or correction is cached. */
struct RouterPacket {float2 hidden;__half2 weights;};
/** @brief Borrow one live even-column pair from original published sources. */
__device__ __forceinline__ RouterPacket routerPacket(const float* hidden,const __half2* weights,unsigned pair) {
 return {{hidden[2*pair],hidden[2*pair+1]},weights[pair]};
}
/** @brief Commit one original pair using the actual operand conversion and original expression.
 * @param packet Exact immutable FP32 hidden/FP16 weight pair. @param sum Original lane accumulator.
 */
__device__ __forceinline__ void routerCommitPacket(const RouterPacket& packet,float& sum) {
 const float2 ww=__half22float2(packet.weights);
 sum+=::llaminar2::rocm::router_native::router_fp16_operand(packet.hidden.x)*ww.x+
      ::llaminar2::rocm::router_native::router_fp16_operand(packet.hidden.y)*ww.y;
}
/** @brief Evaluate the exact current logical128 router lane before its original wave tree. */
template<bool Lookahead=false>
__device__ __forceinline__ float routerPartial128(const float* hidden,const __half* gate_weights,
 int d_model,int num_experts,int expert,unsigned team_lane) {
 constexpr int TeamThreads=128;
 using llaminar2::rocm::router_native::router_fp16_operand;
 float sum=0.0f;
 if(expert<num_experts) {
  const __half* w=gate_weights+static_cast<size_t>(expert)*d_model;
  if((d_model&1)==0) {
   const int pair_count=d_model>>1;const __half2* w2=reinterpret_cast<const __half2*>(w);
   if constexpr(Lookahead) {
    for(int i=team_lane;i<pair_count;i+=2*TeamThreads) {
     const auto first=routerPacket(hidden,w2,unsigned(i));
     const bool second_live=i+TeamThreads<pair_count;
     RouterPacket second{};
     if(second_live)second=routerPacket(hidden,w2,unsigned(i+TeamThreads));
     routerCommitPacket(first,sum);
     if(second_live)routerCommitPacket(second,sum);
    }
   } else {
   for(int i=team_lane;i<pair_count;i+=TeamThreads) {
    const float2 ww=__half22float2(w2[i]);const int h=i<<1;
    sum+=::llaminar2::rocm::router_native::router_fp16_operand(hidden[h])*ww.x+::llaminar2::rocm::router_native::router_fp16_operand(hidden[h+1])*ww.y;
   }
   }
  } else for(int i=team_lane;i<d_model;i+=TeamThreads)
   sum+=::llaminar2::rocm::router_native::router_fp16_operand(hidden[i])*__half2float(w[i]);
 }
 return sum;
}
/** @brief Map two original logical router waves onto one physical wave, retaining all three original trees. */
template<bool Lookahead=false>
__device__ __forceinline__ float router128(const float* hidden,const uint16_t* gate_weights,
 int width,int experts,int expert,unsigned lane) {
    using llaminar2::rocm::moe_wave_contract::rocm_moe_wave_reduce_sum_f32;
    float first=0.f,second=0.f;
    // One current logical-wave address site, then its completed root. Do not
    // retain both future hidden/weight address chains through the first wave.
#pragma unroll 1
    for(unsigned bank=0;bank<2;++bank) {
        float value=routerPartial128<Lookahead>(hidden,reinterpret_cast<const __half*>(gate_weights),
            width,experts,expert,lane+64*bank);
        value=rocm_moe_wave_reduce_sum_f32(value);
        value=__builtin_bit_cast(float,__builtin_amdgcn_readlane(__builtin_bit_cast(int,value),0));
        if(bank==0)first=value;else second=value;
    }
    const float seed=lane==0?first:lane==1?second:0.f;
    return rocm_moe_wave_reduce_sum_f32(seed);
}
/** @brief Preserve the original256-logical-lane FP32 shared-gate dot through four virtual wave states. */
__device__ __forceinline__ float sharedGate256(const float* input,const float* weights,unsigned width,unsigned lane) {
    using llaminar2::rocm::moe_wave_contract::rocm_moe_wave_reduce_sum_f32;
    float first=0.f,second=0.f,third=0.f,fourth=0.f;
    // A common bank loop keeps only the current pair of float4 addresses live.
    // Each original logical thread and its original FMA sequence are unchanged.
#pragma unroll 1
    for(unsigned bank=0;bank<4;++bank) {
        float sum=0.f;
        const auto* a=reinterpret_cast<const float4*>(input);const auto* b=reinterpret_cast<const float4*>(weights);
        for(unsigned i=lane+bank*64;i<width/4;i+=256) {
            const float4 av=a[i],bv=b[i];sum+=av.x*bv.x+av.y*bv.y+av.z*bv.z+av.w*bv.w;
        }
        sum=rocm_moe_wave_reduce_sum_f32(sum);
        const float root=__builtin_bit_cast(float,__builtin_amdgcn_readlane(__builtin_bit_cast(int,sum),0));
        if(bank==0)first=root;else if(bank==1)second=root;else if(bank==2)third=root;else fourth=root;
    }
    const float seed=lane==0?first:lane==1?second:lane==2?third:lane==3?fourth:0.f;
    return rocm_moe_wave_reduce_sum_f32(seed);
}

/** @brief Publish one original role record fieldwise into caller-owned LDS. */
__device__ __forceinline__ void writeProjection(PhaseProjection& destination,const PhaseProjection& source) {
    destination.payload=source.payload;destination.scales=source.scales;
    destination.mins=source.mins;destination.emins=source.emins;
    destination.parts=source.parts;destination.partition_blocks=source.partition_blocks;
    destination.codebook=source.codebook;destination.ready=source.ready;
    destination.numeric_validity=source.numeric_validity;
}
/**
 * @brief Publish independent gate/up records from the same acquired selected route.
 * @param args Existing public source/policy bindings. @param scratch Same CTA's original selection/bank.
 * @param route Original selected slot. @param live Original device-owned row state.
 * @param records Existing two-record LDS span, disjoint from hidden/panel.
 * @details Whole-wave leaders 0 and 64 independently borrow the already issued
 * immutable bank and validate the same selected row; neither acquires an epoch
 * or republishes runtime routing state. Each leader publishes only its own
 * original role through the canonical policy/readiness/numeric classifier.
 * Nonlive, out-of-range, remote and non-native rows retain initialized unready
 * records without dereferencing a role descriptor. Every physical CTA lane
 * reaches the unchanged publication join before either record is consumed.
 * @pre The admitted ordered-wave64 entry has one active publisher lane in each
 * selected wave. Its first active lane therefore names exactly that wave's
 * original role; making this scalar avoids carrying a per-lane descriptor
 * offset through the route/validation loop without changing record ownership.
 */
__device__ __forceinline__ void publishRoutedPair(const NativeLayoutRouterFFNArgs& args,
    NativeLayoutFFNPhaseScratch& scratch,unsigned route,bool live,PhaseProjection* records) {
    static_assert(NativeLayoutDownGeometry::waves>=2);
    if(threadIdx.x==0||threadIdx.x==rocmOrderedWaveWidth) {
        const unsigned role=waveMetadataValue(unsigned(threadIdx.x)/rocmOrderedWaveWidth);
        const auto& a=args.ffn;
        const int expert=scratch.phase.selection.selected_ids[route];
        const auto acquired=scratch.gate_state.bank.borrow();
        // Nonlive/out-of-range routes never dereference a placement row.
        // Membership and epoch stay dynamic; only immutable checks move cold.
        if(!live||expert<0||expert>=a.experts) {
            PhaseProjection empty{};empty.parts=1;empty.partition_blocks=a.hidden_columns/32;
            writeProjection(records[role],empty);
        } else {
        const auto projection=acquired.localNativeProjection(expert,
            role==0?PublishedNativeProjectionRole::Gate:PublishedNativeProjectionRole::Up);
        const auto policy=LLAMINAR_TUNED_SUFFIX_POLICY(routed_gate_policy);
        const auto mask=LLAMINAR_TUNED_SUFFIX_MASK(routed_gate);
        // Read the same captured reduction only when this route publishes.
        // The existing volatile kernarg seam prevents a hoisted K32 ceil
        // numerator from occupying a vector register across the route loop.
        const auto reduction=phaseKernargField<NativeLayoutRouterFFNArgs,
            decltype(ResidentMoEFFNArgs::hidden_columns),
            offsetof(NativeLayoutRouterFFNArgs,ffn)+offsetof(ResidentMoEFFNArgs,hidden_columns)>();
        writeProjection(records[role],phaseProjection(projection,policy,mask,reduction));
        }
    }
    workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
}
/** @brief Original64-row coefficient permutation, with K local to the bounded window. */
__host__ __device__ constexpr unsigned panelIndex(unsigned row,unsigned block,unsigned capacity) {
    if(capacity==1)return block;
    return (block/16)*(capacity*16)+(row/4)*64+((block%16)^((row/4)>>1))*4+row%4;
}
/**
 * @brief Select aligned cardinal output grains without increasing maximum CTA work.
 * @param rows Real output width. @param owners Real CTA cohort size.
 * @return Eight or sixteen rows, derived from the native eight-half metadata packet.
 */
__host__ __device__ constexpr unsigned rowGrain(unsigned rows,unsigned owners) {
    const unsigned small=(rows+7u)/8u,large=(rows+15u)/16u;
    return rows%16==0&&((large+owners-1)/owners)*16<=((small+owners-1)/owners)*8?16:8;
}
/**
 * @brief Copy only real Q4 scale/min fields for one bounded K window and output tile.
 * @tparam Binding Original metadata type, retaining its caller-owned LDS address space.
 * @param record Wave-uniform original validated role; no descriptor is reinterpreted.
 * @param columns Original role N. @param first First real output row.
 * @param count Real output count, at most64. @param first_block Original first K32 block.
 * @param blocks Real K32 window count. @param capacity Cold-admitted real output capacity.
 * @param panel Existing capacity-row-per-block word bank.
 * @details Full aligned eight-half packets retain the winner's int4 issues;
 * scalar tails never read a padded/neighboring source element. Publication and
 * retirement joins are owned by the enclosing uniform CTA loop, not this helper.
 */
template<class Binding>
__device__ __forceinline__ void publishPanel(const Binding& record,unsigned columns,
    unsigned first,unsigned count,unsigned first_block,unsigned blocks,unsigned capacity,unsigned* panel) {
    if(!record.ready||record.numeric_validity!=PhaseProjectionValidation::State::Valid)return;
    const unsigned packets=(count+7u)/8u;
    for(unsigned item=threadIdx.x;item<blocks*packets;item+=1024) {
        const unsigned block=item/packets,row=(item%packets)*8;
        const unsigned real=min(8u,count-row);
        const std::size_t index=std::size_t(first_block+block)*columns+first+row;
        if(real==8&&(index&7u)==0&&
            (reinterpret_cast<std::uintptr_t>(record.scales+index)&15u)==0&&
            (reinterpret_cast<std::uintptr_t>(record.mins+index)&15u)==0) {
            const int4 s=globalWeight<int4>(record.scales+index);
            const int4 m=globalWeight<int4>(record.mins+index);
            const int4 low=make_int4(int((unsigned(s.x)&65535u)|(unsigned(m.x)<<16)),
                int((unsigned(s.x)>>16)|(unsigned(m.x)&0xffff0000u)),
                int((unsigned(s.y)&65535u)|(unsigned(m.y)<<16)),
                int((unsigned(s.y)>>16)|(unsigned(m.y)&0xffff0000u)));
            const int4 high=make_int4(int((unsigned(s.z)&65535u)|(unsigned(m.z)<<16)),
                int((unsigned(s.z)>>16)|(unsigned(m.z)&0xffff0000u)),
                int((unsigned(s.w)&65535u)|(unsigned(m.w)<<16)),
                int((unsigned(s.w)>>16)|(unsigned(m.w)&0xffff0000u)));
            *reinterpret_cast<int4*>(panel+panelIndex(row,block,capacity))=low;
            *reinterpret_cast<int4*>(panel+panelIndex(row+4,block,capacity))=high;
        } else for(unsigned offset=0;offset<real;++offset)
            panel[panelIndex(row+offset,block,capacity)]=unsigned(globalWeight<uint16_t>(record.scales+index+offset))|
                (unsigned(globalWeight<uint16_t>(record.mins+index+offset))<<16);
    }
}
/**
 * @brief Append actual Q4 K32 blocks using the existing paired integer/commit authority.
 * @tparam Binding Original metadata type, retaining native LDS provenance.
 * @param gate Original uniform gate record. @param up Original uniform up record.
 * @param n Original output. @param N Original output width. @param K Original K width.
 * @param part Original partition. @param first_block Current window start.
 * @param blocks Current real window count. @param local_row Panel-relative output.
 * @param capacity Cold-admitted real output capacity.
 * @param gate_panel Existing raw gate coefficients. @param up_panel Existing raw up coefficients.
 * @param codes Original hidden codes. @param scales Original hidden scales. @param sums Original sums.
 * @param g Original gate prefix, carried unchanged between windows.
 * @param u Original up prefix, carried unchanged between windows.
 * @pre Both roles have equal actual P/partition width and valid physical Q4_1.
 */
template<class Binding>
__device__ __forceinline__ void consumeQ4Window(const Binding& gate,const Binding& up,
    unsigned n,unsigned N,unsigned K,unsigned part,unsigned first_block,unsigned blocks,unsigned local_row,unsigned capacity,
    const unsigned* gate_panel,const unsigned* up_panel,const int8_t* codes,const float* scales,
    const int32_t* sums,float& g,float& u) {
    const unsigned width=waveMetadataValue(gate.partition_blocks);
    const unsigned first=max(first_block,part*width);
    const unsigned last=min(first_block+blocks,min(K/32,(part+1)*width));
    const auto* gp=waveMetadataValue(gate.payload);const auto* upayload=waveMetadataValue(up.payload);
#pragma unroll 1
    for(unsigned block=first;block<last;++block) {
        const std::size_t linear=std::size_t(block)*N+n;
        BlockOperand<NATIVE_QUANTISED_Q4_1> gw,uw;
        preloadBlockPayload<NATIVE_QUANTISED_Q4_1,GlobalWeightReader>(gw,gp,linear);
        preloadBlockPayload<NATIVE_QUANTISED_Q4_1,GlobalWeightReader>(uw,upayload,linear);
        gw.scale_pair=gate_panel[panelIndex(local_row,block-first_block,capacity)];
        uw.scale_pair=up_panel[panelIndex(local_row,block-first_block,capacity)];
        const auto activation=preloadActivationBlock<NATIVE_QUANTISED_Q4_1>(codes+std::size_t(block)*32,scales,sums,block);
        int32_t gd,ud;
        consumePairedQ4IntegerDots<NATIVE_QUANTISED_Q4_1,false>(gw,uw,activation.words,gd,ud);
        const auto* original=reinterpret_cast<const int8_t*>(activation.words);
        commitPackedIntegerDots<NATIVE_QUANTISED_Q4_1>(gw,original,gd,0,activation.scale,activation.sum,g);
        commitPackedIntegerDots<NATIVE_QUANTISED_Q4_1>(uw,original,ud,0,activation.scale,activation.sum,u);
    }
}
/**
 * @brief Balanced original selected-route GU with bounded, real-only Q4 coefficient windows.
 * @tparam Mask Complete admitted physical family; unrelated formats use the same canonical role helper.
 * @param args Original argument authority. @param scratch Original selection and acquired bank.
 * @param live Device-owned row activity. @param codes Completed hidden codes.
 * @param scales Completed hidden scales. @param sums Completed hidden exact sums.
 * @param arena Existing phase-exclusive panel/record region after hiddenQ8.
 * @param panel_blocks Cold-admitted K32 window, capped80.
 * @param panel_rows Native poweroftwo4..64 capacity or the minimum linear one-row panel.
 * @details Each CTA cohort owns an original selected slot; aligned cardinal
 * chunks cover every output once. Every K32 commit and each role's own P fold
 * remain ordered. Larger K only introduces local window-publication joins.
 * All physical lanes reach every panel join, including inactive columns/waves.
 */
template<std::uint32_t Mask>
__device__ __forceinline__ void routedGateUp(const NativeLayoutRouterFFNArgs& args,
    NativeLayoutFFNPhaseScratch& scratch,bool live,const int8_t* codes,const float* scales,
    const int32_t* sums,unsigned char* arena,unsigned panel_blocks,unsigned panel_rows) {
    const auto& a=args.ffn;
    const unsigned H=a.hidden_columns,F=a.routed_columns,routes=a.top_k;
    constexpr bool HasQ4=(Mask&(1u<<unsigned(NATIVE_QUANTISED_Q4_1)))!=0;
    const std::size_t panel_words=HasQ4?std::size_t(panel_blocks)*panel_rows:0;
    auto* panel=reinterpret_cast<unsigned*>(arena);
    auto* records=reinterpret_cast<PhaseProjection*>(panel+2*panel_words);
    const unsigned cohorts=min(routes,unsigned(gridDim.x));
    const unsigned bid=blockIdx.x;
    const unsigned cohort=unsigned((std::uint64_t(bid+1)*cohorts-1)/unsigned(gridDim.x));
    const unsigned begin=unsigned(std::uint64_t(cohort)*gridDim.x/cohorts);
    const unsigned end=unsigned(std::uint64_t(cohort+1)*gridDim.x/cohorts);
    const unsigned owners=end-begin,rank=bid-begin;
    const unsigned grain=rowGrain(F,owners),chunks=(F+grain-1)/grain;
    const unsigned first=unsigned(std::uint64_t(rank)*chunks/owners)*grain;
    const unsigned last=min(F,unsigned(std::uint64_t(rank+1)*chunks/owners)*grain);
    const unsigned wave=waveMetadataValue(unsigned(threadIdx.x)/64),lane=phasePhysicalLane64();
    if(first==last)return; // Uniform empty owner still reaches the caller's grid join.
    for(unsigned route=cohort;route<routes;route+=cohorts) {
        publishRoutedPair(args,scratch,route,live,records);
        const auto* current=currentFormatAddress((CurrentFormatLDSValue<const PhaseProjection>*)records);
        const auto& gate=current[0];const auto& up=current[1];
        bool paired=false;
        if constexpr(HasQ4)paired=waveMetadataValue(gate.ready)&&waveMetadataValue(up.ready)&&
            waveMetadataValue(gate.codebook)==NATIVE_QUANTISED_Q4_1&&waveMetadataValue(up.codebook)==NATIVE_QUANTISED_Q4_1&&
            waveMetadataValue(gate.numeric_validity)==PhaseProjectionValidation::State::Valid&&
            waveMetadataValue(up.numeric_validity)==PhaseProjectionValidation::State::Valid&&
            waveMetadataValue(gate.parts)==waveMetadataValue(up.parts)&&
            waveMetadataValue(gate.partition_blocks)==waveMetadataValue(up.partition_blocks);
        for(unsigned tile=first;tile<last;tile+=panel_rows) {
            const unsigned count=min(panel_rows,last-tile),local=wave*4+(lane&3u),n=tile+local,part=lane>>2;
            float g=0.f,u=0.f;
            if(paired&&panel_blocks) {
                for(unsigned block=0;block<H/32;block+=panel_blocks) {
                    const unsigned blocks=min(panel_blocks,H/32-block);
                    publishPanel(gate,F,tile,count,block,blocks,panel_rows,panel);
                    publishPanel(up,F,tile,count,block,blocks,panel_rows,panel+panel_words);
                    workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
                    if(local<count&&part<waveMetadataValue(gate.parts))
                        consumeQ4Window(gate,up,n,F,H,part,block,blocks,local,panel_rows,panel,panel+panel_words,
                            codes,scales,sums,g,u);
                    // A subsequent window overwrites the same raw coefficient words.
                    if(H/32-block>panel_blocks)workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
                }
            } else if(local<count) {
                g=phasePartial<Mask,ActivationIssue::GateUpBlockPreload,GlobalWeightReader>(gate,n,codes,scales,sums,F,H,part);
                u=phasePartial<Mask,ActivationIssue::GateUpBlockPreload,GlobalWeightReader>(up,n,codes,scales,sums,F,H,part);
            }
            const NativeLayoutDownWaveGeometry geometry{16,4};
            // Only waves owning a real output need the ordered P folds. The
            // predicate is uniform across all64 peers: a partial final wave
            // still supplies every original partition lane and publication
            // mask. Empty waves keep the surrounding CTA publication joins.
            if(wave*4u<count) {
                g=orderedWorkerZeroFold(g,waveMetadataValue(gate.parts),geometry,lane&3u,lane);
                u=orderedWorkerZeroFold(u,waveMetadataValue(up.parts),geometry,lane&3u,lane);
                if(lane<4&&local<count)
                    a.gate_partials[std::size_t(route)*F+n]=decode_equivalent::swigluValue(g,u);
            }
            // Covers record reuse for a later route as well as a later tile.
            if(last-tile>panel_rows||routes-route>cohorts)workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
        }
    }
}
/** @brief Per-lane Q8 fields selected only after both original roles were uniformly borrowed. */
struct SelectedQ8Fields {
    const uint8_t* payload; ///< This exact selected role's payload allocation.
    const uint16_t* scales; ///< This exact selected role's raw coefficient allocation.
    unsigned parts,width; ///< Original selected role's P and ceil(K32/P).
    bool valid; ///< Original numeric rejection remains independent of readiness.
};
/**
 * @brief Borrow one uniform role before any lane chooses gate versus up.
 * @tparam Binding Original generic/local metadata reference, without an address-space conversion.
 * @param role The same original record for every physical wave lane.
 * @return Original fields; no selected pointer is ever passed back through readfirstlane.
 */
template<class Binding>
__device__ __forceinline__ SelectedQ8Fields borrowQ8(const Binding& role) {
    return {waveMetadataValue(role.payload),waveMetadataValue(role.scales),
        waveMetadataValue(role.parts),waveMetadataValue(role.partition_blocks),
        waveMetadataValue(role.ready)&&waveMetadataValue(role.numeric_validity)==PhaseProjectionValidation::State::Valid};
}
/**
 * @brief Consume selected raw Q8 fields using the sole canonical block/commit helpers.
 * @param fields Already selected lane-local fields, never rebroadcast.
 * @param n Actual output row. @param N Original validated role width.
 * @param K Complete K. @param part Original partition. @param codes Original hidden codes.
 * @param scales Original hidden scales. @param sums Original exact hidden sums.
 * @return The unchanged ascending K32 partition or the original positive zero.
 */
__device__ __forceinline__ float selectedQ8Partial(const SelectedQ8Fields& fields,unsigned n,
    unsigned N,unsigned K,unsigned part,const int8_t* codes,const float* scales,const int32_t* sums) {
    if(!fields.valid||n>=N||part>=fields.parts)return 0.f;
    const unsigned first=part*fields.width,last=min(K/32,first+fields.width);
    float result=0.f;
#pragma unroll 1
    for(unsigned block=first;block<last;++block) {
        const auto operand=preloadBlock<NATIVE_QUANTISED_Q8_0,GlobalWeightReader>(fields.payload,fields.scales,nullptr,nullptr,
            std::size_t(block)*N+n);
        consumePackedBlock<NATIVE_QUANTISED_Q8_0,ActivationIssue::GateUpBlockPreload>(operand,
            codes+std::size_t(block)*32,scales,sums,block,result);
    }
    return result;
}
/** @brief Original arbitrary-stride P append; all64 peers execute before owner masks. */
__device__ __forceinline__ float foldStrided(float partial,unsigned parts,unsigned column,unsigned stride) {
    if(parts==1)return absoluteWaveValue(partial,column);
    float result=0.f;
#pragma unroll 1
    for(unsigned part=0;part<parts;++part)
        result=rocm_native_quantised_accumulate_rn(result,absoluteWaveValue(partial,stride*part+column));
    return result;
}
/**
 * @brief Author original shared public GU planes and activated values for one physical wave task.
 * @tparam Mask Original complete physical decoder family.
 * @param args Original shared descriptors/public outputs. @param records Both original published roles.
 * @param first First real row owned by this task. @param rows Real task rows (one/two for joint Q8).
 * @param joint True only for two actual physical Q8 roles. @param lane Physical wave64 lane.
 * @param codes Original hidden codes. @param scales Original scales. @param sums Original sums.
 * @param output_demand Cold consumer/observation demand for global mirror stores.
 * @details Both uniform borrows precede role selection. Different/missing format
 * roles use their existing canonical format decoder under a uniform record;
 * they are complete implementations of those formats, not a dispatch fallback.
 */
template<std::uint32_t Mask>
__device__ __forceinline__ void sharedItem(const NativeLayoutRouterFFNArgs& args,const PhaseProjection* records,
    unsigned first,unsigned rows,bool joint,unsigned lane,const int8_t* codes,const float* scales,const int32_t* sums,
    FullNativeHCFFNOutputDemand output_demand) {
    const auto& a=args.ffn;const unsigned S=a.shared_columns,H=a.hidden_columns;
    const auto* current=currentFormatAddress((CurrentFormatLDSValue<const PhaseProjection>*)records);
    const auto& gate=current[0];const auto& up=current[1];
    float g=0.f,u=0.f;
    if(joint) {
        const auto gate_fields=borrowQ8(gate),up_fields=borrowQ8(up);
        const unsigned row=lane&1u,part=lane>>2;
        const bool is_up=(lane&2u)!=0;
        const auto fields=is_up?up_fields:gate_fields;
        float partial=0.f;
        if(row<rows)partial=selectedQ8Partial(fields,first+row,S,H,part,codes,scales,sums);
        if(rows==2) {
            const NativeLayoutDownWaveGeometry geometry{16,4};
            g=orderedWorkerZeroFold(partial,gate_fields.parts,geometry,lane&3u,lane);
            const float up_roots=orderedWorkerZeroFold(partial,up_fields.parts,geometry,lane&3u,lane);
            u=moe_wave_contract::rowShiftNativeBits<2>(up_roots);
        } else {
            // Original odd-tail READLANE chain has its own native operand roles.
            g=foldStrided(partial,gate_fields.parts,0,4);
            u=foldStrided(partial,up_fields.parts,2,4);
        }
        if(lane<rows) {
            const unsigned n=first+lane;
            if(output_demand.contains(FullNativeHCFFNOutputRole::SharedGate))a.shared_gate_output[n]=g;
            if(output_demand.contains(FullNativeHCFFNOutputRole::SharedUp))a.shared_up_output[n]=u;
            a.gate_partials[std::size_t(a.top_k)*a.routed_columns+n]=decode_equivalent::swigluValue(g,u);
        }
    } else {
        const unsigned column=lane&3u,part=lane>>2,n=first+column;
        if(column<rows) {
            g=phasePartial<Mask,ActivationIssue::GateUpBlockPreload,GlobalWeightReader>(gate,n,codes,scales,sums,S,H,part);
            u=phasePartial<Mask,ActivationIssue::GateUpBlockPreload,GlobalWeightReader>(up,n,codes,scales,sums,S,H,part);
        }
        const NativeLayoutDownWaveGeometry geometry{16,4};
        g=orderedWorkerZeroFold(g,waveMetadataValue(gate.parts),geometry,column,lane);
        u=orderedWorkerZeroFold(u,waveMetadataValue(up.parts),geometry,column,lane);
        if(lane<rows) {
            if(output_demand.contains(FullNativeHCFFNOutputRole::SharedGate))a.shared_gate_output[n]=g;
            if(output_demand.contains(FullNativeHCFFNOutputRole::SharedUp))a.shared_up_output[n]=u;
            a.gate_partials[std::size_t(a.top_k)*a.routed_columns+n]=decode_equivalent::swigluValue(g,u);
        }
    }
}
/**
 * @brief Canonical router and shared GU run as independent whole-wave items.
 * @tparam Mask Original complete decoder union. @param args Original captured arguments.
 * @param scratch Original selection/gate-bank scratch, retained across both grid joins.
 * @param live Device row activity. @param codes Original hidden codes. @param scales Original scales.
 * @param sums Original sums. @param records Two existing phase records after the bounded panel.
 * @param output_demand Sealed graph demand; it does not suppress shared arithmetic or routing state.
 * @details Cardinal row ownership and the original128/256 logical arithmetic
 * are independent of resident CTA count. There is no extra grid or CTA join
 * inside the wave task loop; empty/partial work still reaches the caller's join.
 */
template<std::uint32_t Mask>
__device__ __forceinline__ void routerShared(const NativeLayoutRouterFFNArgs& args,
    NativeLayoutFFNPhaseScratch& scratch,bool live,const int8_t* codes,const float* scales,
    const int32_t* sums,PhaseProjection* records,FullNativeHCFFNOutputDemand output_demand) {
    const auto& a=args.ffn;
    // Independent wave leaders publish the two original immutable roles.
    // Each active publisher remains wave-uniform; canonical readiness, source
    // policy and numeric validation are unchanged. Both disjoint48B records
    // become readable only after the same complete1024-thread join below.
    static_assert(NativeLayoutDownGeometry::waves>=2);
    if(threadIdx.x==0||threadIdx.x==rocmOrderedWaveWidth) {
        const unsigned role=unsigned(threadIdx.x)/rocmOrderedWaveWidth;
        const PreparedFullNativeSharedSource source(a);
        const auto projection=source.projection(role==0?PublishedNativeProjectionRole::Gate:PublishedNativeProjectionRole::Up);
        const auto policy=LLAMINAR_TUNED_SUFFIX_POLICY(shared_gate_policy);
        const auto mask=LLAMINAR_TUNED_SUFFIX_MASK(shared_gate);
        writeProjection(records[role],phaseProjection(projection,policy,mask,a.hidden_columns));
    }
    workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
    const unsigned lane=phasePhysicalLane64(),wave=waveMetadataValue(unsigned(threadIdx.x)/64);
    const auto router_rows=NativeLayoutDownGeometry::rows(a.experts,blockIdx.x,gridDim.x);
    const auto shared_rows=NativeLayoutDownGeometry::rows(a.shared_columns,blockIdx.x,gridDim.x);
    bool joint=false;
    if constexpr(Mask&(1u<<unsigned(NATIVE_QUANTISED_Q8_0)))
        joint=waveMetadataValue(records[0].codebook)==NATIVE_QUANTISED_Q8_0&&
            waveMetadataValue(records[1].codebook)==NATIVE_QUANTISED_Q8_0;
    const unsigned rows_per_item=joint?2u:4u;
    const unsigned nl=router_rows.end-router_rows.begin,ns=shared_rows.end-shared_rows.begin;
    const unsigned shared_tasks=(ns+rows_per_item-1)/rows_per_item;
    const unsigned scalar=blockIdx.x+1u==gridDim.x?1u:0u;
    for(unsigned task=wave;task<nl+shared_tasks+scalar;task+=16) {
        if(task<nl) {
            const unsigned expert=router_rows.begin+task;
            const float value=router128<true>(a.hidden,static_cast<const uint16_t*>(args.router_weights),
                a.hidden_columns,a.experts,expert,lane);
            if(lane==0)args.router_logits[expert]=value;
        } else if(task<nl+shared_tasks) {
            const unsigned first=shared_rows.begin+(task-nl)*rows_per_item;
            sharedItem<Mask>(args,records,first,min(rows_per_item,shared_rows.end-first),joint,lane,codes,scales,sums,output_demand);
        } else {
            const float value=live?sharedGate256(a.hidden,a.shared_input_gate,a.hidden_columns,lane):0.f;
            if(lane==0)a.down_partials[0]=value;
        }
    }
}
/**
 * @brief Retain the original terminal and P fold when readiness/column policy rejects every block.
 * @param parts Exact original positive source P. @return Original positive-zero row result.
 * @details The original per-part phasePartial returns +0 on this same predicate.
 * Its multiply1 and P1/direct or +0-seeded increasing-P nodes remain explicit.
 */
__device__ __forceinline__ float completeZeroRow(unsigned parts) {
    float result=0.f;
#pragma unroll 1
    for(unsigned part=0;part<parts;++part) {
        const float value=device_fp32_contract::multiply(0.f,1.f);
        result=parts==1?value:rocm_native_quantised_accumulate_rn(result,value);
    }
    return result;
}
/**
 * @brief Complete a ready original row after one wave-uniform physical-format decision.
 * @tparam FMT Exact admitted original format. @tparam Binding Immutable acquired role/LDS reference.
 * @param role Original metadata, unchanged for the entire row. @param n Original output.
 * @param N Original N. @param K Original K. @param parts Original actual source P.
 * @param codes Original activation bytes. @param scales Original scales. @param sums Original sums.
 * @return Original K32/P result, with exactly the former terminal/fold nodes.
 * @pre The caller already checked the original ready and n<N predicates. Every
 * loop part is below the same parts field, so the former per-part eligibility
 * guard is redundant. Numeric validity and all payload/scale/decode behavior
 * still belong to the unchanged canonical phaseRolePartial body.
 */
template<NativeQuantisedFormat FMT,class Binding>
__device__ __forceinline__ float completeRow(const Binding& role,unsigned n,unsigned N,unsigned K,unsigned parts,
    const int8_t* codes,const float* scales,const int32_t* sums) {
    // The acquired record is immutable for this complete output row. Read its
    // original validity and planes once, before the P loop; canonical block
    // consumers contain compiler boundaries which otherwise reload the same
    // LDS fields and repeat their wave broadcast for every partition.
    if(waveMetadataValue(role.numeric_validity)!=PhaseProjectionValidation::State::Valid)
        return completeZeroRow(parts);
    using Traits=NATIVE_QUANTISEDTraits<FMT>;
    const PhaseProjection acquired{
        .payload=waveMetadataValue(role.payload),
        .scales=waveMetadataValue(role.scales),
        .mins=Traits::requires_secondary_metadata?waveMetadataValue(role.mins):nullptr,
        .emins=Traits::needs_emin_array?waveMetadataValue(role.emins):nullptr,
        .parts=parts,
        .partition_blocks=waveMetadataValue(role.partition_blocks),
        .codebook=static_cast<uint8_t>(FMT),
        .ready=true,
        .numeric_validity=PhaseProjectionValidation::State::Valid};
    float result=0.f;
#pragma unroll 1
    for(unsigned part=0;part<parts;++part) {
        float value=phaseRolePartial<FMT,ActivationIssue::DownWordsPreload,GlobalWeightReader>(
            acquired,n,codes,scales,sums,N,K,part);
        value=device_fp32_contract::multiply(value,1.f);
        result=parts==1?value:rocm_native_quantised_accumulate_rn(result,value);
    }
    return result;
}
/** @brief Typed down source borrowing only the current acquired and prepared owners. */
template<class Acquired>
class PublishedFullDownSource final {
    Acquired acquired_;
    PreparedFullNativeSharedSource shared_;
public:
    /** @param acquired Original issued epoch token. @param args SAME sealed full arguments. */
    __device__ PublishedFullDownSource(Acquired acquired,const ResidentMoEFFNArgs& args)
        :acquired_(acquired),shared_(args){}
    /** @param shared Original shared slot. @param expert Actual selected ID or explicit inactive sentinel.
     * @return SAME original native down projection; no mutable table is read through scalar-cache provenance. */
    __device__ __forceinline__ PublishedNativeProjection projection(bool shared,int expert) const {
        return shared?shared_.projection(PublishedNativeProjectionRole::Down):
            acquired_.localNativeProjectionOrInactive(expert,PublishedNativeProjectionRole::Down);
    }
};
/**
 * @brief Dispatch one acquired row once, outside its unchanged complete original-P sequence.
 * @tparam Mask Actual decoder envelope. @tparam Binding Immutable original metadata/LDS reference.
 * @param role Original role. @param n Original output. @param N Original N. @param K Original K.
 * @param codes Original activation bytes. @param scales Original scales. @param sums Original sums.
 * @return Canonical complete row; weighting/publication stay at the original caller.
 * @details This moves only the format and invariant eligibility decision. It
 * adds no decoder, partition accumulator array, source conversion or metadata
 * cache. Every format still uses the same canonical phaseRolePartial/consumer
 * and GlobalWeightReader; numeric rejection and unsupported-format traps remain.
 */
template<std::uint32_t Mask,class Binding>
__device__ __forceinline__ float downRow(const Binding& role,unsigned n,unsigned N,unsigned K,
    const int8_t* codes,const float* scales,const int32_t* sums) {
    const unsigned parts=waveMetadataValue(role.parts);
    if(!waveMetadataValue(role.ready)||n>=N)return completeZeroRow(parts);
#define LLAMINAR_COMPLETE_ROW(F) case F:if constexpr(Mask&(1u<<unsigned(F)))return completeRow<F>(role,n,N,K,parts,codes,scales,sums);else break;
    switch(waveMetadataValue(role.codebook)) {
    LLAMINAR_COMPLETE_ROW(NATIVE_QUANTISED_Q4_0) LLAMINAR_COMPLETE_ROW(NATIVE_QUANTISED_IQ4_NL)
    LLAMINAR_COMPLETE_ROW(NATIVE_QUANTISED_IQ4_XS_SOURCE) LLAMINAR_COMPLETE_ROW(NATIVE_QUANTISED_Q4_1)
    LLAMINAR_COMPLETE_ROW(NATIVE_QUANTISED_Q5_0) LLAMINAR_COMPLETE_ROW(NATIVE_QUANTISED_Q5_1)
    LLAMINAR_COMPLETE_ROW(NATIVE_QUANTISED_Q6_K) LLAMINAR_COMPLETE_ROW(NATIVE_QUANTISED_Q3_K)
    LLAMINAR_COMPLETE_ROW(NATIVE_QUANTISED_Q2_K) LLAMINAR_COMPLETE_ROW(NATIVE_QUANTISED_IQ3_S)
    LLAMINAR_COMPLETE_ROW(NATIVE_QUANTISED_IQ3_XXS) LLAMINAR_COMPLETE_ROW(NATIVE_QUANTISED_IQ2_S)
    LLAMINAR_COMPLETE_ROW(NATIVE_QUANTISED_IQ2_XS) LLAMINAR_COMPLETE_ROW(NATIVE_QUANTISED_IQ2_XXS)
    LLAMINAR_COMPLETE_ROW(NATIVE_QUANTISED_IQ1_S) LLAMINAR_COMPLETE_ROW(NATIVE_QUANTISED_IQ1_M)
    LLAMINAR_COMPLETE_ROW(NATIVE_QUANTISED_Q8_0) LLAMINAR_COMPLETE_ROW(NATIVE_QUANTISED_EXPANDED_INT8_MIN)
    default:break;
    }
#undef LLAMINAR_COMPLETE_ROW
    __builtin_trap();return 0.f;
}
/**
 * @brief Route-by-column output owners preserve the complete per-route FP32 ABI.
 * @tparam Mask Actual physical decoder family. @param args Original full suffix bindings.
 * @param scratch Original down metadata and weights after selector retirement.
 * @param live Device row state. @param codes Completed intermediate codes.
 * @param scales Completed intermediate scales. @param sums Completed exact sums.
 * @param completed One tile of finished routes, at most(topK+1)*64 floats.
 * @param output_demand Actual outside readers/observations; the local ordered fold remains unconditional.
 * @details Every independent wave owns one route and adjacent output columns.
 * Completed routes, never partial K leaves, enter the one local join. The
 * original selected-slot fold then authors routed/weighted-shared/combined planes.
 */
template<std::uint32_t Mask>
__device__ __forceinline__ void down(const NativeLayoutRouterFFNArgs& args,NativeLayoutFFNPhaseScratch& scratch,
    bool live,const int8_t* codes,const float* scales,const int32_t* sums,float* completed,
    FullNativeHCFFNOutputDemand output_demand) {
    const auto& a=args.ffn;const unsigned H=a.hidden_columns,F=a.routed_columns,S=a.shared_columns,routes=a.top_k;
    const auto owned=NativeLayoutDownGeometry::rows(H,blockIdx.x,gridDim.x);
    if(owned.begin==owned.end)return; // Both global publication joins already completed.
    auto* bindings=scratch.phase.down;
    const auto* runtime=reinterpret_cast<const DeviceMoELayerRuntimeView*>(a.runtime);
    publishDownPhase<DownMetadataPublication::LiveWaveLeaders>(H,F,S,routes,a.experts,live,args.router_publication.down,
        a.shared_down,runtime->topk_expert_ids,runtime->topk_weights,bindings,scratch.route_weights,
        PublishedFullDownSource(scratch.gate_state.bank.borrow(),a));
    const unsigned wave=waveMetadataValue(unsigned(threadIdx.x)/64),lane=phasePhysicalLane64();
    constexpr unsigned shared_slot=NativeLayoutDownGeometry::maximum_capacity-1;
    const auto* records=currentFormatAddress((CurrentFormatLDSValue<const PhaseProjection>*)bindings);
    for(unsigned first=owned.begin;first<owned.end;first+=64) {
        const unsigned count=min(64u,owned.end-first);
        for(unsigned route=wave;route<=routes;route+=16) {
            if(lane<count) {
                const unsigned n=first+lane;
                float value;
                if(route<routes) {
                    const float raw=downRow<Mask>(records[route],n,H,F,
                        codes+std::size_t(route)*F,scales+std::size_t(route)*(F/32),sums+std::size_t(route)*(F/32));
                    value=rocm_native_quantised_weight_route_rn(scratch.route_weights[route],raw);
                    if(output_demand.contains(FullNativeHCFFNOutputRole::RouteContributions))
                        a.route_contributions[std::size_t(route)*H+n]=value;
                } else {
                    const std::size_t offset=std::size_t(routes)*F;
                    const float raw=downRow<Mask>(records[shared_slot],n,H,S,
                        codes+offset,scales+offset/32,sums+offset/32);
                    value=rocm_native_quantised_accumulate_rn(0.f,rocm_native_quantised_weight_route_rn(1.f,raw));
                }
                completed[std::size_t(route)*64+lane]=value;
            }
        }
        workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
        if(threadIdx.x<count) {
            const unsigned local=threadIdx.x,n=first+local;
            const float routed=device_ordered_route_fold::sum(completed+local,std::size_t(64),routes);
            const float shared=live?device_fp32_contract::multiply(scratch.gate_state.reduction[0],
                completed[std::size_t(routes)*64+local]):0.f;
            if(output_demand.contains(FullNativeHCFFNOutputRole::Routed))a.routed_output[n]=routed;
            if(output_demand.contains(FullNativeHCFFNOutputRole::Shared))a.shared_output[n]=shared;
            a.combined_output[n]=live?device_fp32_contract::add(routed,shared):0.f;
        }
        if(owned.end-first>64)workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
    }
}
} // namespace full_native_ffn
/**
 * @brief Run the tuned complete production suffix after the HC mixed-ready edge.
 * @tparam Mask Exact original physical decoder union, retaining all15 compiled families.
 * @param args Actual FIRST by-value kernel argument, preserving field-loader authority.
 * @param grid Existing instance-owned two-phase suffix barrier, at phasezero.
 * @param scratch Existing CTA-local selector/bank/static metadata storage.
 * @param storage Admitted16-byte-aligned phase arena, extent from fullNativeHCFFNSuffixGeometry.
 * @param output_demand Immutable graph/observation materialization policy for this prepared generation.
 * @details args.full_entry_panel is the exact cold geometry result sealed in capture identity.
 * @pre Every1024-thread CTA enters. Original row/source/publication/stream and
 * workspace generations are sealed by the existing cold full-entry owner.
 * @post Original runtime routing state and all declared GU/route/final planes
 * are published; only kernel completion orders terminal global consumers.
 */
template<std::uint32_t Mask>
__device__ __forceinline__ void tunedFullNativeHCFFNSuffix(const NativeLayoutRouterFFNArgs args,
    ResidentGridBarrier& grid,NativeLayoutFFNPhaseScratch& scratch,unsigned char* storage,
    FullNativeHCFFNOutputDemand output_demand) {
    using namespace native_layout_ffn;
    const auto& a=args.ffn;const unsigned H=a.hidden_columns,F=a.routed_columns,S=a.shared_columns,routes=a.top_k;
    const bool live=a.rows.activeRows()!=0;
    const auto* runtime=reinterpret_cast<const DeviceMoELayerRuntimeView*>(a.runtime);
    if(!runtime||runtime->expert_count!=unsigned(a.experts)||runtime->top_k!=routes||
        (live&&!runtime->intermediate_shard.sameIdentity(a.original)))__builtin_trap();
    if(threadIdx.x==0)scratch.gate_state.bank.publish(
        router_native::AcquiredRuntimeDecodeBank<DeviceMoELayerRuntimeView>::acquire(
            reinterpret_cast<DeviceMoELayerRuntimeView*>(a.runtime)));
    auto* codes=reinterpret_cast<int8_t*>(storage);
    auto* scales=reinterpret_cast<float*>(codes+H);
    auto* sums=reinterpret_cast<int32_t*>(scales+H/32);
    constexpr bool HasQ4=(Mask&(1u<<unsigned(NATIVE_QUANTISED_Q4_1)))!=0;
    const bool use_panel=(LLAMINAR_TUNED_SUFFIX_MASK(routed_gate)&(1u<<unsigned(NATIVE_QUANTISED_Q4_1)))!=0;
    const unsigned panel_blocks=args.full_entry_panel.blocks,panel_rows=args.full_entry_panel.rows;
    if((panel_rows!=1&&(panel_rows<4||panel_rows>64||(panel_rows&(panel_rows-1))))||
        panel_blocks>80||(panel_rows!=1&&(panel_blocks&15u))||
        (use_panel&&(!HasQ4||!panel_blocks))||(!use_panel&&panel_blocks))__builtin_trap();
    const std::size_t raw_hidden=std::size_t(H)+std::size_t(H/32)*8;
    const std::size_t hidden_bytes=panel_rows==1||!panel_blocks?raw_hidden:(raw_hidden+15u)&~std::size_t(15u);
    auto* arena=storage+hidden_bytes;
    const std::size_t panel_words=use_panel?std::size_t(panel_blocks)*panel_rows:0;
    auto* records=reinterpret_cast<PhaseProjection*>(arena+2*panel_words*sizeof(unsigned));
    full_native_ffn::quantizeRow(a.hidden,H,codes,scales,sums);
    full_native_ffn::routerShared<Mask>(args,scratch,live,codes,scales,sums,records,output_demand);
    grid.sync(); // Original raw logits/shared GU publication; every CTA arrives.
    if(threadIdx.x==0)scratch.gate_state.reduction[0]=live?1.f/(1.f+expf(-a.down_partials[0])):0.f;
    auto& selection=scratch.phase.selection;
    router_native::selectProbabilities<false,256,false,WorkgroupJoinMemory::LDSOnly,
        router_native::TopKTransferPolicy::RowNativeRoot,
        router_native::SelectorIntermediateStorage::ThreadPair>(
        args.router_logits,nullptr,a.experts,routes,args.normalize_router_weights,selection);
    if(blockIdx.x==0&&threadIdx.x<routes)
        router_native::runtime_publish_independent_decode_dispatch_from_bank<true,
            RoutedExpertRowExecutionPolicy::IntermediateShardLocal,true>(scratch.gate_state.bank.borrow(),
            selection.selected_ids,selection.selected_weights,a.experts,routes,args.router_indices,
            args.router_weights_output,args.write_router_outputs,a.rows.countOwner(),args.router_publication);
    full_native_ffn::routedGateUp<Mask>(args,scratch,live,codes,scales,sums,arena,panel_blocks,panel_rows);
    grid.sync(); // Retire every raw-logit selector reader and publish routed activation/descriptors.
    if(output_demand.contains(FullNativeHCFFNOutputRole::RouterProbabilities)&&blockIdx.x==0)
      for(unsigned expert=threadIdx.x;expert<unsigned(a.experts);expert+=1024)
        args.router_logits[expert]=selection.expert_probs[expert];
    const std::size_t values=std::size_t(routes)*F+S;
    codes=reinterpret_cast<int8_t*>(storage);scales=reinterpret_cast<float*>(codes+values);
    sums=reinterpret_cast<int32_t*>(scales+values/32);
    full_native_ffn::quantizeRow(a.gate_partials,unsigned(values),codes,scales,sums);
    // Quantizer's unchanged all-CTA join retires probability/selection reads;
    // only now may the existing static union be reused for down metadata.
    auto* completed=reinterpret_cast<float*>(sums+values/32);
    full_native_ffn::down<Mask>(args,scratch,live,codes,scales,sums,completed,output_demand);
}
} // namespace llaminar2::rocm

#undef LLAMINAR_TUNED_SUFFIX_POLICY
#undef LLAMINAR_TUNED_SUFFIX_MASK
