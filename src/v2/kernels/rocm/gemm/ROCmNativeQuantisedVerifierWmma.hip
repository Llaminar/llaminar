/**
 * @file ROCmNativeQuantisedVerifierWmma.hip
 * @brief Native M16/N64 verifier producers with canonical serial-M1 partitions.
 *
 * Original request intervals own row admission before decoding. One CTA writes
 * one physical K-partition plane; existing ordered reducers own the final fold.
 * Two-wave native tiles share decoded weights across sixteen rows. Exact hosted
 * construction selects this producer or the original Dot4 implementation from
 * measured public-operation keys. Mixed codebooks
 * share one LDS bank and the established compact decoder/contribution helpers.
 * No gfx906 device entry is emitted, and capture-time admission requires gfx1100.
 */
#include "ROCmNativeQuantisedVerifierLaunch.h"
#include "ROCmNativeQuantisedArchitecture.h"
#include "utils/PerfStatsCollector.h"
#include "ROCmNativeQuantisedDecodeEquivalentMath.h"
#include "kernels/rocm/common/ROCmSignedDot4.h"
#include "kernels/common/ProjectionWorkLaunch.h"
#include <hip/hip_fp16.h>
#include <stdexcept>
#include <type_traits>
#include "ROCmNativeQuantisedGemmDecode.hip.inc"

namespace llaminar2::rocm::verifier_wmma_detail {
#include "ROCmMoEGroupedWmmaMath.hip.inc"
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1100__)
/** @brief One shared bank for every decoder arm; capacity never multiplies by formats. */
struct Storage {
    I32x4 a[2][16][2], b[2][64][2]; ///< Two K32 stages with permuted vector banks.
    uint16_t scale[2][64], aux[2][64], qh[2][64]; ///< Original source metadata.
    uint32_t minima[2][64]; ///< Original dual-half minima.
    int sums[2][16][4]; ///< Disjoint scalar owners avoid vector-component write races.
    float activation_scale[2][16]; ///< Per-row/K32 scales, zero-work rows never read.
};
/** @brief One resolved bundle plane; pointer arrays die before decoder lifetimes. */
struct Projection {
    int n, column; ///< Physical width and CTA column origin.
    const uint8_t *payload; ///< Compact source blocks.
    const uint16_t *scales, *aux; ///< Original scale and correction planes.
    const uint32_t *minima; ///< Optional dual-half minima.
    const float *bias; ///< Only direct fused publication uses this pointer.
    float *destination; ///< Exact physical partition/output plane selected before decode.
};
/**
 * @brief Publish through the original standalone/fused/partial arithmetic contract.
 * @tparam Publication Captured output arithmetic. @param p Resolved destination.
 * @param row Original physical row. @param column Original column. @param value Ordered partition result.
 */
template<VerifierWmmaPublication Publication>
__device__ __forceinline__ void publish(Projection p,int row,int column,float value)
{
    if constexpr(Publication==VerifierWmmaPublication::Batched)
        value=__fadd_rn(value,p.bias?p.bias[column]:0.0f);
    p.destination[static_cast<size_t>(row)*p.n+column]=value;
}

/**
 * @brief Compute one original live row interval in one canonical partition.
 * @tparam FMT Shared physical decoder. @tparam Publication Exact destination arithmetic.
 * @param a Physical activation rows. @param activation_scales Per-K32 row scales.
 * @param activation_sums Original optional whole-K32 sums, authoritative when supplied.
 * @param p Resolved immutable projection. @param first Original first row.
 * @param end Original exclusive live interval end. @param k Physical contraction width.
 * @param begin_block First canonical K32 block. @param end_block Exclusive partition end.
 * @param storage CTA-owned bank reused only after all consumers retire.
 */
template<NativeQuantisedGemmFormat FMT, VerifierWmmaPublication Publication>
__device__ __forceinline__ void tile(const int8_t *a, const float *activation_scales,
    const int32_t *activation_sums, Projection p, int first, int end, int k, int begin_block, int end_block, Storage &storage)
{
#if defined(__gfx1100__)
    static_assert(__builtin_amdgcn_is_invocable(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w64));
    using Traits=NATIVE_QUANTISEDGemmTraits<FMT>;
    constexpr int Groups=FMT==NATIVE_QUANTISED_GEMM_IQ1_M ? 4 :
        Traits::is_dual_scale_asym ? 2 : Traits::is_asymmetric ? 1 : 0;
    const int tid=threadIdx.x, lane=tid%64, wave=tid/64, blocks=k/32;
    float acc[2][4] = {};
    for (int block=begin_block; block<end_block; block+=2) {
        // One lane owns each decoded weight/K32 block. Metadata preserves the
        // same source factors and rounding as ordinary production serial M1.
        const int stage=tid/64, column=tid%64, kb=block+stage;
        const int n=p.column+column;
        uint32_t raw[Traits::payload_dwords]={};
        uint16_t primary=0, secondary=0; uint32_t minima=0;
        if (n<p.n && kb<end_block) {
            const size_t index=static_cast<size_t>(kb)*p.n+n;
            load_payload<Traits::payload_bytes>(p.payload+index*Traits::payload_bytes,raw);
            primary=::llaminar2::native_quantised_scale::loadStagedPrimary<static_cast<uint8_t>(FMT)>(p.scales,index);
            if constexpr(Traits::needs_aux_compute) secondary=p.aux[index];
            if constexpr(Traits::is_dual_scale_asym) minima=p.minima[index];
        }
        int decoded[8]; decode_block<FMT>(raw,decoded);
        I32x4 low,high;
        #pragma unroll
        for(int g=0;g<4;++g){low[g]=decoded[g];high[g]=decoded[g+4];}
        storage.b[stage][column^stage][(column>>2)&1]=low;
        storage.b[stage][column^stage][((column>>2)&1)^1]=high;
        storage.scale[stage][column]=primary;
        if constexpr(Traits::needs_aux_compute) storage.aux[stage][column]=secondary;
        if constexpr(Traits::is_dual_scale_asym) storage.minima[stage][column]=minima;
        if constexpr(FMT==NATIVE_QUANTISED_GEMM_IQ1_M) storage.qh[stage][column]=raw[1];

        if(tid<64) {
            const int row=tid/4, as=(tid%4)/2, half=tid%2, ak=block+as;
            I32x4 values={};
            if(first+row<end && ak<end_block)
                values=reinterpret_cast<const I32x4*>(a+static_cast<size_t>(first+row)*k+ak*32)[half];
            storage.a[as][row^as][half^((row>>2)&1)]=values;
            if constexpr(Groups==1) {
                if(activation_sums) {
                    if(half==0) storage.sums[as][row][0]=first+row<end && ak<end_block
                        ? activation_sums[static_cast<size_t>(first+row)*blocks+ak] : 0;
                } else {
                    // The standalone API also admits quantization without a
                    // published sum plane; retain its exact integer reduction.
                    int sum=0;
                    #pragma unroll
                    for(int g=0;g<4;++g) sum=rocm_integer_contract::signedDot4(values[g],0x01010101,sum);
                    sum+=__shfl_xor(sum,1,32);
                    if(half==0) storage.sums[as][row][0]=sum;
                }
            } else if constexpr(Groups>1) {
                #pragma unroll
                for(int group=0;group<Groups/2;++group) {
                    int sum=0;
                    #pragma unroll
                    for(int g=0;g<8/Groups;++g)
                        sum=rocm_integer_contract::signedDot4(values[group*(8/Groups)+g],0x01010101,sum);
                    storage.sums[as][row][half*(Groups/2)+group]=sum;
                }
            }
            if(half==0) storage.activation_scale[as][row]=first+row<end && ak<end_block
                ? activation_scales[static_cast<size_t>(first+row)*blocks+ak] : 1.0f;
        }
        __syncthreads();
        #pragma unroll 1
        for(int as=0;as<2 && block+as<end_block;++as) {
            const int row=lane%16;
            const auto al=storage.a[as][row^as][(row>>2)&1];
            const auto ah=storage.a[as][row^as][((row>>2)&1)^1];
            #pragma unroll
            for(int ni=0;ni<2;++ni) {
                const int col=wave*32+ni*16+lane%16;
                const auto bl=storage.b[as][col^as][(col>>2)&1];
                const auto bh=storage.b[as][col^as][((col>>2)&1)^1];
                const I32x4 zero={};
                const auto lo=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w64(true,al,true,bl,zero,false);
                const auto hi=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w64(true,ah,true,bh,Traits::is_dual_scale?zero:lo,false);
                const auto primary=storage.scale[as][col];
                const uint16_t secondary=Traits::needs_aux_compute?storage.aux[as][col]:0;
                const auto weight=[&] {
                    if constexpr(Traits::is_dual_scale)
                        return rocm_native_quantised_load_dual_weight_scales<FMT>(primary,secondary);
                    else return ROCmNativeQuantisedDualWeightScales{
                        rocm_native_quantised_load_single_weight_scale<FMT>(primary,secondary),0.0f};
                }();
                const uint32_t minima=Traits::is_dual_scale_asym?storage.minima[as][col]:0;
                const uint16_t qh=FMT==NATIVE_QUANTISED_GEMM_IQ1_M?storage.qh[as][col]:0;
                #pragma unroll
                for(int e=0;e<4;++e) {
                    const int output_row=e*4+lane/16; I32x4 sums={};
                    #pragma unroll
                    for(int group=0;group<Groups;++group) sums[group]=storage.sums[as][output_row][group];
                    acc[ni][e]=wideCommit<FMT>(acc[ni][e],lo[e],hi[e],weight,secondary,minima,qh,
                        storage.activation_scale[as][output_row],sums);
                }
            }
        }
        __syncthreads();
    }
    // Empty trailing canonical partitions still publish +0 for live rows.
    // Their poisoned old scratch must never reach the fixed ordered reducer.
    #pragma unroll
    for(int ni=0;ni<2;++ni) {
        const int n=p.column+wave*32+ni*16+lane%16;
        #pragma unroll
        for(int e=0;e<4;++e) {
            const int row=first+e*4+lane/16;
            if(row<end && n<p.n) {
                publish<Publication>(p,row,n,acc[ni][e]);
            }
        }
    }
#endif
}
/**
 * @brief Inline original interval resolution before the decoder lifetime begins.
 * @tparam Cursor Existing prefix/request cursor, retaining its sole row authority.
 * @param cursor Borrowed per-CTA traversal. @param first Original first row.
 * @param end Original interval end. @return Whether an authored tile remains.
 * @details Statement inlining is local to this new producer. Outlining this
 * small reference-returning helper introduces an avoidable device call-frame
 * spill in the decoder-heavy caller.
 */
template<class Cursor>
__device__ __forceinline__ bool nextTile(Cursor &cursor,int &first,int &end)
{
    [[clang::always_inline]] return cursor.next(first,end);
}


/**
 * @brief Resolve original intervals before decoding native matrix operands.
 * @tparam Publication Captured publication contract. @tparam HasWork Original request ABI.
 * @param a Quantized physical input. @param scales Physical activation scales.
 * @param sums Original optional whole-K32 sum plane.
 * @param args Captured bundle. @param k Physical contraction width.
 * @param span Canonical ceil-divided serial partition width. @param storage Native CTA bank.
 */
template<VerifierWmmaPublication Publication,bool HasWork>
__device__ __forceinline__ void produce(
    const int8_t *a,const float *scales,const int32_t *sums,NativeQuantisedSmallMBatchedArgs args,int k,int span,Storage *storage)
{
#if defined(__gfx1100__)
    int proj=0;
    #pragma unroll
    for(int i=1;i<NATIVE_QUANTISED_SMALL_M_MAX_BATCH;++i)
        if(i<args.num_projections && blockIdx.x>=static_cast<unsigned>(args.grid_offsets[i]))proj=i;
    const int n=args.N[proj], physical=args.rows.physicalRows();
    float *dest;
    const float *bias=nullptr;
    if constexpr(Publication==VerifierWmmaPublication::OrderedPartial)
        dest=args.partial[proj]+static_cast<size_t>(blockIdx.y)*physical*n;
    else {
        dest=args.output[proj];
        if constexpr(Publication==VerifierWmmaPublication::Batched)bias=args.bias[proj];
    }
    Projection p{n,static_cast<int>(blockIdx.x-args.grid_offsets[proj])*64,
        args.payload[proj],args.scales[proj],args.mins[proj],args.emins[proj],bias,dest};
    const auto codebook=static_cast<NativeQuantisedGemmFormat>(args.codebook[proj]);
    auto cursor=projectionCursorFor<HasWork>(args.work,args.rows,blockIdx.z,gridDim.z,16);
    const int begin=blockIdx.y*span,end_block=min(k/32,begin+span);
    int first,end;
    while(nextTile(cursor,first,end)) {
        end=min(end,first+16);
        switch(codebook) {
#define LLAMINAR_VERIFIER_WMMA(F) case F: \
            tile<F,Publication>(a,scales,sums,p,first,end,k,begin,end_block,*storage); \
            break
        LLAMINAR_VERIFIER_WMMA(NATIVE_QUANTISED_GEMM_Q4_0);
        LLAMINAR_VERIFIER_WMMA(NATIVE_QUANTISED_GEMM_IQ4_NL);
        LLAMINAR_VERIFIER_WMMA(NATIVE_QUANTISED_GEMM_IQ4_XS_SOURCE);
        LLAMINAR_VERIFIER_WMMA(NATIVE_QUANTISED_GEMM_MXFP4);
        LLAMINAR_VERIFIER_WMMA(NATIVE_QUANTISED_GEMM_Q4_1);
        LLAMINAR_VERIFIER_WMMA(NATIVE_QUANTISED_GEMM_Q5_0);
        LLAMINAR_VERIFIER_WMMA(NATIVE_QUANTISED_GEMM_Q5_1);
        LLAMINAR_VERIFIER_WMMA(NATIVE_QUANTISED_GEMM_Q6_K);
        LLAMINAR_VERIFIER_WMMA(NATIVE_QUANTISED_GEMM_Q3_K);
        LLAMINAR_VERIFIER_WMMA(NATIVE_QUANTISED_GEMM_Q2_K);
        LLAMINAR_VERIFIER_WMMA(NATIVE_QUANTISED_GEMM_IQ3_S);
        LLAMINAR_VERIFIER_WMMA(NATIVE_QUANTISED_GEMM_IQ3_XXS);
        LLAMINAR_VERIFIER_WMMA(NATIVE_QUANTISED_GEMM_IQ2_S);
        LLAMINAR_VERIFIER_WMMA(NATIVE_QUANTISED_GEMM_IQ2_XS);
        LLAMINAR_VERIFIER_WMMA(NATIVE_QUANTISED_GEMM_IQ2_XXS);
        LLAMINAR_VERIFIER_WMMA(NATIVE_QUANTISED_GEMM_IQ1_S);
        LLAMINAR_VERIFIER_WMMA(NATIVE_QUANTISED_GEMM_IQ1_M);
        LLAMINAR_VERIFIER_WMMA(NATIVE_QUANTISED_GEMM_Q8_0);
        LLAMINAR_VERIFIER_WMMA(NATIVE_QUANTISED_GEMM_EXPANDED_INT8_MIN);
#undef LLAMINAR_VERIFIER_WMMA
        default: __builtin_trap();
        }
        __syncthreads();
    }
#endif
}
/** @brief Native matrix packet; template parameters and operands follow produce(). */
template<VerifierWmmaPublication Publication,bool HasWork>
__global__ __launch_bounds__(128,1) void native_quantised_verifier_wmma_m16n64(
    const int8_t *a,const float *scales,const int32_t *sums,NativeQuantisedSmallMBatchedArgs args,int k,int span)
{
#if defined(__gfx1100__)
    __shared__ Storage storage;
    produce<Publication,HasWork>(a,scales,sums,args,k,span,&storage);
#endif
}


#endif
}

namespace llaminar2::rocm {
void submitNativeQuantisedVerifierWmma(const NativeQuantisedVerifierWmmaArguments &args,void *stream)
{
    const auto hs=static_cast<hipStream_t>(stream);
    if(!hs || hs==hipStreamLegacy || hs==hipStreamPerThread || !args.activations ||
        !args.activation_scales || args.k<=0 || args.k%32 || args.partitions<=0 ||
        args.partitions>args.k/32 || args.row_workers<=0 || args.projections.rows.physicalRows()<=0 ||
        args.projections.num_projections<=0 || args.projections.num_projections>NATIVE_QUANTISED_SMALL_M_MAX_BATCH)
        throw std::invalid_argument("Incomplete native verifier WMMA launch");
    int stream_device=-1;
    if(args.device<0 || hipStreamGetDevice(hs,&stream_device)!=hipSuccess || stream_device!=args.device ||
        nativeQuantisedHardwareForDevice(args.device).architecture!=native_quantised::NativeQuantisedArchitecture::rocm(11,0,0))
        throw std::invalid_argument("Native verifier WMMA requires its actual gfx1100 stream/device");
    const auto &bundle=args.projections;
    if(bundle.work.physicalRows()!=bundle.rows.physicalRows() ||
        (args.partitions>1 && args.publication!=VerifierWmmaPublication::OrderedPartial) ||
        (args.publication==VerifierWmmaPublication::Standalone && bundle.num_projections!=1))
        throw std::invalid_argument("Native verifier WMMA row/publication contract disagrees");
    int64_t tiles=0;
    for(int p=0;p<bundle.num_projections;++p) {
        if(bundle.N[p]<=0 || bundle.grid_offsets[p]!=tiles || !bundle.payload[p] || !bundle.scales[p] ||
            (args.publication==VerifierWmmaPublication::OrderedPartial ? !bundle.partial[p] : !bundle.output[p]))
            throw std::invalid_argument("Native verifier WMMA projection is incomplete");
        tiles+=1+(bundle.N[p]-1)/64;
        auto validate=[&]<NativeQuantisedGemmFormat FMT>() {
            using T=NATIVE_QUANTISEDGemmTraits<FMT>;
            if((T::needs_aux_compute && !bundle.mins[p]) || (T::is_dual_scale_asym && !bundle.emins[p]))
                throw std::invalid_argument("Native verifier WMMA source metadata is incomplete");
        };
        switch(bundle.codebook[p]) {
#define LLAMINAR_VALIDATE_VERIFIER(F) case F: validate.template operator()<F>();break
        LLAMINAR_VALIDATE_VERIFIER(NATIVE_QUANTISED_GEMM_Q4_0);
        LLAMINAR_VALIDATE_VERIFIER(NATIVE_QUANTISED_GEMM_IQ4_NL);
        LLAMINAR_VALIDATE_VERIFIER(NATIVE_QUANTISED_GEMM_IQ4_XS_SOURCE);
        LLAMINAR_VALIDATE_VERIFIER(NATIVE_QUANTISED_GEMM_MXFP4);
        LLAMINAR_VALIDATE_VERIFIER(NATIVE_QUANTISED_GEMM_Q4_1);
        LLAMINAR_VALIDATE_VERIFIER(NATIVE_QUANTISED_GEMM_Q5_0);
        LLAMINAR_VALIDATE_VERIFIER(NATIVE_QUANTISED_GEMM_Q5_1);
        LLAMINAR_VALIDATE_VERIFIER(NATIVE_QUANTISED_GEMM_Q6_K);
        LLAMINAR_VALIDATE_VERIFIER(NATIVE_QUANTISED_GEMM_Q3_K);
        LLAMINAR_VALIDATE_VERIFIER(NATIVE_QUANTISED_GEMM_Q2_K);
        LLAMINAR_VALIDATE_VERIFIER(NATIVE_QUANTISED_GEMM_IQ3_S);
        LLAMINAR_VALIDATE_VERIFIER(NATIVE_QUANTISED_GEMM_IQ3_XXS);
        LLAMINAR_VALIDATE_VERIFIER(NATIVE_QUANTISED_GEMM_IQ2_S);
        LLAMINAR_VALIDATE_VERIFIER(NATIVE_QUANTISED_GEMM_IQ2_XS);
        LLAMINAR_VALIDATE_VERIFIER(NATIVE_QUANTISED_GEMM_IQ2_XXS);
        LLAMINAR_VALIDATE_VERIFIER(NATIVE_QUANTISED_GEMM_IQ1_S);
        LLAMINAR_VALIDATE_VERIFIER(NATIVE_QUANTISED_GEMM_IQ1_M);
        LLAMINAR_VALIDATE_VERIFIER(NATIVE_QUANTISED_GEMM_Q8_0);
        LLAMINAR_VALIDATE_VERIFIER(NATIVE_QUANTISED_GEMM_EXPANDED_INT8_MIN);
#undef LLAMINAR_VALIDATE_VERIFIER
        default: throw std::invalid_argument("Unknown native verifier WMMA physical codebook");
        }
    }
    if(tiles!=bundle.total_grid_n || tiles!=bundle.grid_offsets[bundle.num_projections])
        throw std::invalid_argument("Native verifier WMMA column grid disagrees with bundle");
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1100__)
    auto submit=[&]<VerifierWmmaPublication Publication,bool HasWork>() {
        const auto grid=dim3(args.projections.total_grid_n,args.partitions,args.row_workers);
        const int span=(args.k/32+args.partitions-1)/args.partitions;
        verifier_wmma_detail::native_quantised_verifier_wmma_m16n64<Publication,HasWork><<<grid,128,0,hs>>>(
            args.activations,args.activation_scales,args.activation_sums,args.projections,args.k,span);
        const auto status=hipGetLastError();
        if(status!=hipSuccess)throw std::runtime_error(hipGetErrorString(status));
        // Construction evidence names the actual submitted native function;
        // it is not a count of live device replays or active rows.
        if(PerfStatsCollector::isDomainEnabled("kernel"))
            PerfStatsCollector::addCounter("kernel","rocm_native_quantised_verifier_wmma_submission",1.0,
                "gemm",{},{{"engine","wmma_m16n64"},
                {"function",std::to_string(reinterpret_cast<uintptr_t>(
                    verifier_wmma_detail::native_quantised_verifier_wmma_m16n64<Publication,HasWork>))},
                {"block_threads","128"},{"device",std::to_string(args.device)},
                {"physical_m",std::to_string(args.projections.rows.physicalRows())},
                {"grid_n",std::to_string(args.projections.total_grid_n)},
                {"partitions",std::to_string(args.partitions)},{"row_workers",std::to_string(args.row_workers)},
                {"request_work",HasWork?"true":"false"}});
    };
    auto work=[&]<VerifierWmmaPublication Publication>() {
        if(args.has_request_work)submit.template operator()<Publication,true>();
        else submit.template operator()<Publication,false>();
    };
    switch(args.publication) {
    case VerifierWmmaPublication::Standalone: work.template operator()<VerifierWmmaPublication::Standalone>();return;
    case VerifierWmmaPublication::Batched: work.template operator()<VerifierWmmaPublication::Batched>();return;
    case VerifierWmmaPublication::OrderedPartial: work.template operator()<VerifierWmmaPublication::OrderedPartial>();return;
    }
#endif
    throw std::invalid_argument("Unsupported native verifier WMMA publication");
}
bool initializeNativeQuantisedVerifierWmmaGrids(int device,const void *iq3s,
    const void *iq3xxs,const void *iq2s,const void *iq2xs,const void *iq2xxs,const void *iq1s)
{return initialize_native_quantised_gemm_grids(device,iq3s,iq3xxs,iq2s,iq2xs,iq2xxs,iq1s);}
}
