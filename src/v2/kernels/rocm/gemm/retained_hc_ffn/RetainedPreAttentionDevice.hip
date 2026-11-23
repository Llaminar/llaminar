/**
 * @file RetainedPreAttentionDevice.hip
 * @brief Native retained fixed-shape POST/HC followed by original ordered Q8 projection.
 *
 * The original graph owns the tensors, row range, descriptor bank and persistent
 * scratch. This translation unit owns only the exact resident device entry
 * and its byte layout. Castagna ancestry and MIT terms accompany the body.
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "CastagnaNativeWeightExport.h"
#include "kernels/rocm/gemm/RetainedPreAttentionDevice.h"
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "ROCmRetainedM1HCFFNDevice.h"
#include "kernels/common/NativeCooperativeKernel.cuh"
#include "kernels/rocm/moe/ROCmNativeRuntimeBankView.h"
#include "RuntimeNativeExpertTableView.hip.inc"
#include "CanonicalFFNQ8.h"
#include "kernels/common/DeviceNativeQuantisedContributionContract.h"
#include "kernels/common/MoEProjectionNumericalContract.h"
#include "kernels/rocm/common/ROCmSignedDot4.h"
#include "kernels/rocm/gemm/ROCmNativeQuantisedBlockMetadata.h"
#include "kernels/rocm/common/ROCmQ4PairedDot.h"
#include "kernels/qwen4exp/GPUBranchRMSNormBody.cuh"
#include "kernels/common/OriginalSingleQ8Block32.cuh"
#include "kernels/common/GPUHCDownPublication.cuh"
#include "kernels/common/QuantizedActivationOperandPrecision.h"
#include "kernels/rocm/gemm/ROCmNativeQuantisedDecodeEquivalentMath.h"
#include "kernels/rocm/moe/ROCmOriginalRouterPhase.hip.inc"
#include "kernels/common/FullNativeHCFFNBarrierLayout.h"
#include <bit>
#include <type_traits>

#include "FastBankDeviceSection.inc"

namespace qfn_retained_m1_hc_ffn_device {
using ProbeProjection=llaminar2::rocm::pre_attention_probe::Projection;
/** @brief Bare device pointers from the already authenticated half owner. */
struct HalfPointers
{
    const std::uint16_t* alpha;
    const std::uint16_t* beta;
    float* raw_alpha;
    float* raw_beta;
};

/** @brief Each ordinary projection wave owns complete original K partitions for four columns. */
template<unsigned Columns>
__device__ __forceinline__ void completeProjectionGroup(
    const ProbeProjection& projection,unsigned first,const CanonicalFFNQ8Block* activated)
{
    namespace numeric=llaminar2::device_fp32_contract;
    const unsigned lane=threadIdx.x%64;
    const unsigned column=first+(lane&3u);
    const unsigned part=lane>>2;
    constexpr unsigned partitions=llaminar2::MoEProjectionNumericalContract::orderedKPartitionsForWidth(E);
    static_assert(partitions==16 && Columns%4==0);
    const auto fields=CanonicalSharedQ8Fields{Columns,projection.weights.payload,
        static_cast<const unsigned short*>(projection.weights.scales)};
    const float partial=canonicalSharedQ8Partial(fields,column,part,E,activated);
    // The original reducer begins at +0 and performs sixteen RN additions in
    // increasing partition order. All active wave lanes reach every shuffle.
    float result=0.0f;
#pragma unroll
    for(unsigned p=0;p<partitions;++p)
        result=numeric::add(result,__shfl(partial,(lane&3u)+4u*p,64));
    if(part==0)projection.output[column]=result;
}

/** @brief Widen with the original floating GEMM's NaN spelling and exact finite conversion. */
__device__ __forceinline__ float originalHalfWiden(std::uint16_t half)
{
    const auto native=__float_as_uint(__half2float(__ushort_as_half(half)));
    const auto nan=(std::uint32_t(half&0x8000u)<<16)|0x7f800000u|
        (std::uint32_t(half&0x03ffu)<<13);
    const auto mask=0u-std::uint32_t((half&0x7fffu)>0x7c00u);
    return __uint_as_float(native^((native^nan)&mask));
}

/** @brief One wave executes the original 256 logical partials for one prepared half result. */
__device__ __forceinline__ void completeHalfOutput(const float* mixed,const std::uint16_t* weight,float* out)
{
    const unsigned lane=threadIdx.x%64;
    float partial[4]{};
    // Preserve every fixed stride and the original +0 initialization despite
    // carrying four logical lanes per physical lane.
#pragma unroll
    for(unsigned step=0;step<E/256;++step) {
#pragma unroll
        for(unsigned group=0;group<4;++group) {
            const unsigned index=step*256+group*64+lane;
            const auto activation=__half_as_ushort(__float2half_rn(mixed[index]));
            partial[group]+=originalHalfWiden(activation)*originalHalfWiden(weight[index]);
        }
    }
    float value=(partial[0]+partial[2])+(partial[1]+partial[3]);
    for(unsigned stride=32;stride>0;stride>>=1)
        value+=__shfl_down(value,stride,64);
    if(lane==0)*out=value;
}

/** @brief Original once-published HC/mixed row and a proven local Q8 projection cohort. */
template<bool PublishedTrunk,bool FusedHalf>
__global__ __launch_bounds__(NT) void preAttentionKernel(
    const hc_src src,const float* gain,NativeMatrix down,NativeMatrix up,float epsilon,
    float* post,float* normalized,float* mixed,float* raw_down,const int* live_rows,
    ProbeProjection qkv,ProbeProjection z,HalfPointers half,void* state)
{
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx906__)
    constexpr unsigned prefixBytes=(256+2*E)*sizeof(float);
    alignas(16) __shared__ char arena[prefixBytes];
    __shared__ float lo[(LR/32)*33];
    __shared__ float gates[1][HC][64];
    __shared__ CanonicalFFNQ8Block activation[E/32];
    auto& prefix=*static_cast<CurrentHCPrefixState*>(state);
    auto* barrier=reinterpret_cast<unsigned*>(static_cast<char*>(state)+sizeof(CurrentHCPrefixState));
    ScalarInstanceWeightedJoin join{.words=barrier};
    const bool live=llaminar2::rocm::router_native::runtime_decode_row_active(live_rows);
    qfn_castagna_native::CanonicalPortObservation observation{};
    observation.hc_down=raw_down;
    current_llaminar_hc_prefix<1,llaminar2::QuantizedActivationOperandPrecision::TwoTermQ8,false,true,PublishedTrunk>(
        src,gain,down,up,epsilon,post,normalized,mixed,reinterpret_cast<float*>(arena),lo,gates,observation,prefix,join,live);
    if(!live)return;

    // Exact native v4 K32 quantization. Both original projections share the
    // local bank and preserve the original ordered 16-partition output fold.
    for(unsigned physical=threadIdx.x;physical<E;physical+=NT) {
        const unsigned block=physical/32,lane=physical%32;
        auto& q=activation[block];
        llaminar2::original_single_q8::publish(mixed[physical],true,
            reinterpret_cast<std::int8_t*>(q.qs),&q.scale,nullptr,lane);
    }
    lds_sync();
    constexpr unsigned qkv_groups=2560/4,z_groups=1536/4;
    constexpr unsigned total=qkv_groups+z_groups;
    const unsigned begin=blockIdx.x*total/NB,end=(blockIdx.x+1u)*total/NB;
    for(unsigned task=begin+threadIdx.x/64;task<end;task+=NT/64) {
        if(task<qkv_groups)completeProjectionGroup<2560>(qkv,4*task,activation);
        else completeProjectionGroup<1536>(z,4*(task-qkv_groups),activation);
    }
    if constexpr(FusedHalf) {
        constexpr unsigned heads=12;
        if(blockIdx.x<2*heads && threadIdx.x<64) {
            const bool beta=blockIdx.x>=heads;
            const auto row=blockIdx.x%heads;
            const auto* matrix=beta?half.beta:half.alpha;
            auto* output=beta?half.raw_beta:half.raw_alpha;
            completeHalfOutput(mixed,matrix+row*E,output+row);
        }
    }
#endif
}
} // namespace qfn_retained_m1_hc_ffn_device

namespace llaminar2::rocm::pre_attention_probe {
using namespace qfn_retained_m1_hc_ffn_device;
std::size_t scratchBytes() noexcept
{
    return sizeof(CurrentHCPrefixState)+FullNativeHCFFNBarrierLayout::bytes;
}

/** @brief Check the original native, block-major Q8 descriptor and immutable shape. */
static bool originalQ8(const DeviceNativeQuantisedMatrixDesc& weight,int n,int k) noexcept
{
    return weight.valid()&&weight.n==n&&weight.k==k&&weight.blocks_per_row==unsigned(k/32)&&
        weight.codebook_id==19&&weight.source_identity_present&&weight.source_codebook_id==19&&
        !weight.source_is_superblock&&!weight.mins&&!weight.emins;
}

/** @brief Select one exact gfx906 function from the validated immutable capture policy. */
template<bool PublishedTrunk>
static const void* selectFunction(HalfGDNPolicy policy) noexcept
{
    switch(policy) {
    case HalfGDNPolicy::SeparateOriginalSubgroup:
        return reinterpret_cast<const void*>(&preAttentionKernel<PublishedTrunk,false>);
    case HalfGDNPolicy::RetainedPreparedFP16RNE:
        return reinterpret_cast<const void*>(&preAttentionKernel<PublishedTrunk,true>);
    }
    return nullptr;
}

/** @brief Verify one complete prepared half matrix with the original GDN role extent. */
static bool validHalfMatrix(const PreparedHalfMatrix& descriptor) noexcept
{
    return descriptor.words && reinterpret_cast<std::uintptr_t>(descriptor.words)%alignof(std::uint16_t)==0 &&
        descriptor.bytes==std::size_t(12)*E*sizeof(std::uint16_t);
}

hipError_t prepare(const Invocation& invocation,int device,Admission& admission)
{
    const auto& prefix=invocation.prefix;
    const auto& hc=prefix.hc;
    if(!invocation.scratch||invocation.scratch_bytes!=scratchBytes()||
       reinterpret_cast<std::uintptr_t>(invocation.scratch)%16||hc.norm.hidden!=E||
       !hc.norm.gain||!hc.norm.output||!hc.mixed||!hc.raw_down||
       !hc.norm.input||hc.injection.weights||hc.injection.output||
       !originalQ8(hc.down_weights,LR,K)||!originalQ8(hc.up.weights,K,LR)||
       hc.norm.extent.rows!=1||hc.up.rows.physicalRows()!=1||hc.up.rows.capacity()!=1||
       hc.norm.extent.liveRows!=hc.up.rows.countOwner()||
       hc.up.quantizer!=Q8ResidualQuantizerPolicy::ROCmFiniteHalfNativeReciprocal||
       prefix.output_demand.bits())
        return hipErrorInvalidValue;
    if(invocation.geometry!=EarlyQ8Geometry::GDN||
       !invocation.projections[0].output||!originalQ8(invocation.projections[0].weights,2560,E)||
       !invocation.projections[1].output||!originalQ8(invocation.projections[1].weights,1536,E)||
       invocation.projections[2].output||invocation.projections[2].weights.payload)
        return hipErrorInvalidValue;

    const auto& half=invocation.half;
    if(invocation.half_policy==HalfGDNPolicy::RetainedPreparedFP16RNE) {
        if(!validHalfMatrix(half.alpha)||!validHalfMatrix(half.beta)||!half.raw_alpha||!half.raw_beta||
           half.raw_alpha==half.raw_beta)
            return hipErrorInvalidValue;
    } else if(half.alpha.words||half.alpha.bytes||half.beta.words||half.beta.bytes||half.raw_alpha||half.raw_beta) {
        return hipErrorInvalidValue;
    }

    const void* function=nullptr;
    switch(invocation.trunk_source) {
    case TrunkSource::PreviousWrite:
        if(!prefix.write.residual||!prefix.write.block||!prefix.write.gateLogits||
           !prefix.write.output||hc.norm.input!=prefix.write.output)
            return hipErrorInvalidValue;
        function=selectFunction<false>(invocation.half_policy);
        break;
    case TrunkSource::PublishedTrunk:
        if(prefix.write.residual||prefix.write.block||prefix.write.gateLogits||prefix.write.output)
            return hipErrorInvalidValue;
        function=selectFunction<true>(invocation.half_policy);
        break;
    }
    if(!function)return hipErrorInvalidValue;
    int current=-1;
    auto error=hipGetDevice(&current);
    if(error!=hipSuccess)return error;
    if(current!=device)return hipErrorInvalidDevice;
    const auto& hardware=rocmHardwareMetadata().at(device);
    if(hardware.compute_units!=NB || hardware.native_wave_width!=64 ||
       (hardware.isa_target!="gfx906" && !hardware.isa_target.starts_with("gfx906:")))
        return hipErrorInvalidDevice;
    hipFuncAttributes attributes{};
    error=hipFuncGetAttributes(&attributes,function);
    if(error!=hipSuccess)return error;
    int residents=0;
    error=hipOccupancyMaxActiveBlocksPerMultiprocessor(&residents,function,NT,0);
    if(error!=hipSuccess)return error;
    if(residents<1||attributes.localSizeBytes)return hipErrorInvalidConfiguration;
    admission={invocation,function,attributes,device};
    return hipSuccess;
}

hipError_t launch(const Admission& admission,hipStream_t stream) noexcept
{
    const auto& invocation=admission.invocation;
    const auto& prefix=invocation.prefix;
    const auto& hc=prefix.hc;
    if(reinterpret_cast<std::uintptr_t>(stream)<=2||!admission.function)return hipErrorInvalidValue;
    const bool published=invocation.trunk_source==TrunkSource::PublishedTrunk;
    hc_src source{.x=published?hc.norm.input:prefix.write.residual,.out=prefix.write.block,.raw=prefix.write.gateLogits,.ps1=.25f,.ps2=2.f};
    auto* gain=hc.norm.gain;
    auto down=hc.down_weights,up=hc.up.weights;
    auto epsilon=hc.norm.epsilon;
    auto* post=prefix.write.output;
    auto* normalized=hc.norm.output;
    auto* mixed=hc.mixed;
    auto* raw=hc.raw_down;
    const int* live=hc.norm.extent.liveRows;
    auto qkv=invocation.projections[0],z=invocation.projections[1];
    HalfPointers half{invocation.half.alpha.words,
        invocation.half.beta.words,
        invocation.half.raw_alpha,invocation.half.raw_beta};
    void* state=invocation.scratch;
    void* argv[]{&source,&gain,&down,&up,&epsilon,&post,&normalized,&mixed,&raw,&live,&qkv,&z,&half,&state};
    return hipLaunchKernel(admission.function,dim3(NB),dim3(NT),argv,0,stream);
}
}
