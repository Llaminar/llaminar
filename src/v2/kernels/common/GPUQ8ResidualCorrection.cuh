/**
 * @file GPUQ8ResidualCorrection.cuh
 * @brief Captured CUDA/HIP two-term Q8 publication and native prepared-source dots.
 *
 * One publication launch forms two signed-byte activation banks. One projection
 * launch shares every native weight load across bounded real rows and both
 * terms. Independent increasing-K FP32 folds are added exactly once. All
 * pointers and physical geometry are immutable captured operands; live counts
 * remain on device and inactive input/output/scratch bytes are untouched.
 * HC two-term input publication retains the original activated FP32 value,
 * backend nonlinear arithmetic and prepared projection/workspace authority.
 * The native gfx906 cooperative program stages K32 pairs through this same
 * boundary; other native programs retain the original single-group schedule.
 * An explicit HIP reference uses original single-group staging on every target.
 * Native signed INT8 dot products share the exact unsaturated arithmetic
 * contract across gfx906 and gfx1100; lane ownership and reductions are unchanged.
 */
#pragma once
#if defined(Q8_RESIDUAL_CUDA)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define Q8R_STREAM cudaStream_t
#define Q8R_FACTORY cudaQ8ResidualCorrectedProjection
#define Q8R_HC_FACTORY cudaHCResidualDownQ8Projection
#define Q8R_HC_READ_FACTORY cudaHCResidualReadQ8Projection
#define Q8R_HC_SHORT_READ_FACTORY cudaHCResidualShortKReadQ8Projection
#define Q8R_COLUMN_REFERENCE cudaQ8ResidualOriginalColumnReference
#define Q8R_NAMESPACE cuda_q8_residual
#define Q8R_ERROR cudaGetLastError
#define Q8R_SUCCESS cudaSuccess
#define Q8R_ERROR_TEXT cudaGetErrorString
#define Q8R_SHFL(x,o) __shfl_xor_sync(0xffffffffu,x,o,32)
#else
#include "kernels/rocm/common/ROCmSignedDot4.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "kernels/common/VerifierDispatchTable.h"
#include "kernels/rocm/gemm/ROCmNativeQuantisedArchitecture.h"
#include "utils/DebugEnv.h"
#include "kernels/rocm/ops/ROCmSwiGLUDecodeEquivalentMath.h"
#define Q8R_STREAM hipStream_t
#define Q8R_FACTORY rocmQ8ResidualCorrectedProjection
#define Q8R_HC_FACTORY rocmHCResidualDownQ8Projection
#define Q8R_HC_READ_FACTORY rocmHCResidualReadQ8Projection
#define Q8R_HC_SHORT_READ_FACTORY rocmHCResidualShortKReadQ8Projection
#define Q8R_COLUMN_REFERENCE rocmQ8ResidualOriginalColumnReference
#define Q8R_NAMESPACE rocm_q8_residual
#define Q8R_ERROR hipGetLastError
#define Q8R_SUCCESS hipSuccess
#define Q8R_ERROR_TEXT hipGetErrorString
#define Q8R_SHFL(x,o) __shfl_xor(x,o,32)
#endif
#include "Q8ResidualGroupedDiagnostic.h"
#include "utils/PerfStatsCollector.h"
#include "DeviceFP32NumericalContract.h"
#include "DeviceNativeQuantisedContributionContract.h"
#include "Q8ResidualGroupedGeometry.h"
#include "Q8ResidualShortKGeometry.h"
#include "GPUQ8ResidualNative.h"
#include "GPUQ8ResidualNativeDevice.cuh"
#if defined(Q8_RESIDUAL_CUDA)
#include "kernels/cuda/gemm/CUDAQ8ResidualGrouped.cuh"
#else
#include "kernels/rocm/gemm/ROCmQ8ResidualGrouped.cuh"
#endif
#include "GPUHCDownPublication.cuh"
#include <cstdint>
#include <algorithm>
#include <climits>
#include <string>
#include <type_traits>
#include "GPUHCShortKBlockBody.cuh"
namespace llaminar2 {
namespace Q8R_NAMESPACE {
/** @brief Native signed four-byte integer dot without any floating operand. */
__device__ __forceinline__ int dot4(int a,int b,int c) {
#if defined(Q8_RESIDUAL_CUDA)
 return __dp4a(a,b,c);
#else
 return ::llaminar2::rocm_integer_contract::signedDot4(a,b,c);
#endif
}
/** @brief Derive the exact standalone/grouped/prefill HC-up blockwise scale.
 * CUDA's production quantizer is inline in CUDAQuantisedGemmKernel_CUTLASS.cu
 * and exposes no device helper: its FP32 division is retained verbatim here.
 * ROCm directly borrows its source-owned scale helper; the native prepared API
 * fixture authenticates both policies against all actual production dispatches.
 */
__device__ __forceinline__ float blockScale(float maximum) {
#if defined(Q8_RESIDUAL_CUDA)
 return maximum>0.f?maximum/127.f:1.f;
#else
 return rocm::decode_equivalent::q8Scale(maximum);
#endif
}
/** @brief Preserve the backend's actual reciprocal/multiply/RNE/saturation program. */
__device__ __forceinline__ int quantizedValue(float value,float scale) {
 // The actual HC-up blockwise producers use this native reciprocal on both
 // backends. ROCm quantizeQ8FromScale belongs to the heterogeneous MoE path
 // and intentionally derives a different reciprocal; it is not this contract.
 const float inverse=1.f/scale;
#if defined(Q8_RESIDUAL_CUDA)
 const float scaled=value*inverse;
 return static_cast<int>(rintf(fminf(127.f,fmaxf(-127.f,scaled))));
#else
 return rocm::decode_equivalent::quantizeQ8(value,inverse);
#endif
}
/** @brief Fatal nonfinite activation detection never substitutes another precision. */
__device__ __forceinline__ void invalidActivation() {
#if defined(Q8_RESIDUAL_CUDA)
 __trap();
#else
 __builtin_trap();
#endif
}
/** @brief Quantize a primary block and its exact rounded representation residual. */
__global__ __launch_bounds__(256,2) void publish(GPUQ8ResidualCorrectionArgs a) {
 const std::uint64_t task=static_cast<std::uint64_t>(blockIdx.x)*8+threadIdx.x/32;
 const auto bpr=a.weights.blocks_per_row;
 const int live=a.rows.activeRows();
 if(task>=static_cast<std::uint64_t>(live)*bpr)return;
 const unsigned lane=threadIdx.x%32;const auto offset=task*32+lane;
 const float value=a.input[offset];if(!isfinite(value))invalidActivation();float maximum=fabsf(value);
 for(int distance=16;distance;distance/=2)maximum=fmaxf(maximum,Q8R_SHFL(maximum,distance));
 const float stored=blockScale(maximum);
 const int primary=quantizedValue(value,stored);
 a.primary[offset]=static_cast<signed char>(primary);
 const float residual=device_fp32_contract::add(value,-device_fp32_contract::multiply(stored,static_cast<float>(primary)));
 if(!isfinite(residual))invalidActivation();
 float residualMaximum=fabsf(residual);
 for(int distance=16;distance;distance/=2)residualMaximum=fmaxf(residualMaximum,Q8R_SHFL(residualMaximum,distance));
 const float residualScale=blockScale(residualMaximum);
 a.residual[offset]=static_cast<signed char>(quantizedValue(residual,residualScale));
 if(lane==0){a.primaryScales[task]=stored;a.residualScales[task]=residualScale;}
}
/** @brief Original HC /4 SiLU and two-term publication with one input load.
 * @param hc Complete original projection and retained activated FP32 destination.
 * @details Every32-lane subgroup loads its own source values before its same-index store;
 * exact input/activated alias needs no cross-subgroup ordering. Inactive groups
 * leave every FP32 and Q8 bank byte untouched.
 */
__global__ __launch_bounds__(256,2) void publishHCResidualDown(GPUHCResidualDownPublication hc) {
 const auto a=hc.projection;
 const std::uint64_t task=static_cast<std::uint64_t>(blockIdx.x)*8+threadIdx.x/32;
 const auto bpr=a.weights.blocks_per_row;
 const int live=a.rows.activeRows();
 if(task>=static_cast<std::uint64_t>(live)*bpr)return;
 const unsigned lane=threadIdx.x%32;const auto offset=task*32+lane;
 const GPUHCDownOutputView output{a.rows,a.weights.k,hc.activated,a.primary,a.residual,a.primaryScales,a.residualScales};
 hc_down_publication::publish(a.input[offset],output,offset);
}
/** @brief Bounded real-row reuse with independent Q8 terms and one final output store.
 * @tparam Rows Fixed maximum real rows per lane; never changes a row's K order.
 */
template<int Rows>__global__ __launch_bounds__(128,2) void project(GPUQ8ResidualCorrectionArgs a) {
 const auto columns=(static_cast<unsigned>(a.weights.n)+127)/128;
 const auto rowBase=static_cast<std::uint64_t>(blockIdx.x/columns)*Rows;
 const unsigned column=(blockIdx.x%columns)*128+threadIdx.x;
 const int live=a.rows.activeRows();
 if(column>=static_cast<unsigned>(a.weights.n)||rowBase>=static_cast<unsigned>(live))return;
 float primary[Rows]{},residual[Rows]{};
 for(unsigned block=0;block<a.weights.blocks_per_row;++block) {
  const auto linear=static_cast<std::uint64_t>(block)*a.weights.n+column;
  const auto*payload=reinterpret_cast<const int4*>(a.weights.payload+linear*32);
  const int4 low=payload[0],high=payload[1];const int weights[8]{low.x,low.y,low.z,low.w,high.x,high.y,high.z,high.w};
  const auto scaleBits=static_cast<const unsigned short*>(a.weights.scales)[linear];
  const float weightScale=__half2float(__ushort_as_half(scaleBits));
#pragma unroll
  for(int r=0;r<Rows;++r) {
   const auto row=rowBase+r;if(row>=static_cast<unsigned>(live))break;
   const auto activationBlock=row*a.weights.blocks_per_row+block;
   const auto*p0=reinterpret_cast<const int*>(a.primary+activationBlock*32);
   const auto*p1=reinterpret_cast<const int*>(a.residual+activationBlock*32);
   int dot0=0,dot1=0;
#pragma unroll
   for(int group=0;group<8;++group){dot0=dot4(p0[group],weights[group],dot0);dot1=dot4(p1[group],weights[group],dot1);}
   primary[r]=device_fp32_contract::add(primary[r],device_native_quantised_contract::singleScaleBlock(dot0,weightScale,a.primaryScales[activationBlock]));
   residual[r]=device_fp32_contract::add(residual[r],device_native_quantised_contract::singleScaleBlock(dot1,weightScale,a.residualScales[activationBlock]));
  }
 }
#pragma unroll
 for(int r=0;r<Rows;++r)if(rowBase+r<static_cast<unsigned>(live))
  a.output[(rowBase+r)*a.weights.n+column]=device_fp32_contract::add(primary[r],residual[r]);
}
#if !defined(Q8_RESIDUAL_CUDA)
/** @brief Each lane computes both native Q8 terms for one K32 and multiple actual rows. */
template<unsigned Columns>
__device__ __forceinline__ void projectHCGroupedShortKBody(
    GPUQ8ResidualCorrectionArgs a, float (&primary)[4][10][Columns], float (&residual)[4][10][Columns])
{
    constexpr unsigned Groups=10, Rows=4, OutputWidth=10240;
    static_assert(Columns==32 || Columns==64);
    const unsigned ci=threadIdx.x%Columns;
    const unsigned kBlock=Columns==64
        ? __builtin_amdgcn_readfirstlane(threadIdx.x/Columns) : threadIdx.x/Columns;
    const unsigned column=blockIdx.x*Columns+ci;
    const unsigned rowBase=blockIdx.y*Rows;
    const unsigned live=a.rows.activeRows();
    if(rowBase>=live)return;
    const unsigned active=live-rowBase<Rows?live-rowBase:Rows;
    const auto location=kBlock*OutputWidth+column;
    const auto* payload=reinterpret_cast<const int4*>(a.weights.payload+location*32u);
    const int4 low=payload[0],high=payload[1];
    const int words[8]{low.x,low.y,low.z,low.w,high.x,high.y,high.z,high.w};
    const auto scale=__half2float(__ushort_as_half(static_cast<const unsigned short*>(a.weights.scales)[location]));
#pragma unroll
    for(unsigned r=0;r<Rows;++r) {
        if(r>=active)break;
        const unsigned activation=(rowBase+r)*Groups+kBlock;
        const auto* a0=reinterpret_cast<const int*>(a.primary+activation*32);
        const auto* a1=reinterpret_cast<const int*>(a.residual+activation*32);
        int dot0=0,dot1=0;
#pragma unroll
        for(unsigned word=0;word<8;++word) {
            dot0=hc_short_k_native::dot4(a0[word],words[word],dot0);
            dot1=hc_short_k_native::dot4(a1[word],words[word],dot1);
        }
        primary[r][kBlock][ci]=device_native_quantised_contract::singleScaleBlock(dot0,scale,a.primaryScales[activation]);
        residual[r][kBlock][ci]=device_native_quantised_contract::singleScaleBlock(dot1,scale,a.residualScales[activation]);
    }
    __syncthreads();

    // Each of the live row groups has its own fold wave. Original +0/K32
    // additions stay in ascending order and the two banks combine only once.
    if(kBlock<active) {
        float sum0=0.f,sum1=0.f;
#pragma unroll
        for(unsigned group=0;group<Groups;++group) {
            sum0=device_fp32_contract::add(sum0,primary[kBlock][group][ci]);
            sum1=device_fp32_contract::add(sum1,residual[kBlock][group][ci]);
        }
        a.output[(rowBase+kBlock)*OutputWidth+column]=device_fp32_contract::add(sum0,sum1);
    }
}

/** @brief Original half-wave K32 layout for unmeasured row capacities. */
__global__ __launch_bounds__(320) void projectHCGroupedShortK(GPUQ8ResidualCorrectionArgs a)
{
    __shared__ float primary[4][10][32],residual[4][10][32];
    projectHCGroupedShortKBody(a,primary,residual);
}

/** @brief One physical MI50 wave per K32 group, measured for four retained rows. */
__global__ __launch_bounds__(640) void projectHCGroupedShortKWide(GPUQ8ResidualCorrectionArgs a)
{
    __shared__ float primary[4][10][64],residual[4][10][64];
    projectHCGroupedShortKBody(a,primary,residual);
}

/** @brief Prepared ROCm grouped HC-up schedule for this original native tuple. */
enum class HCGroupedShortKSchedule { Disabled, Column32, Column64 };

/**
 * @brief Choose the schedule for the participant recording this grouped Q8 projection.
 * @param a Prepared original weights and retained row geometry.
 * @param stream Exact non-null stream from operand and publication validation.
 * @return The admitted native schedule for this tuple and stream owner.
 */
inline HCGroupedShortKSchedule selectHCGroupedShortK(const GPUQ8ResidualCorrectionArgs& a, hipStream_t stream)
{
    const auto physical=a.rows.physicalRows();
    if(physical<2||physical>8||a.weights.n!=10240||a.weights.k!=320||
       a.weights.blocks_per_row!=10||a.weights.codebook_id!=19||!a.weights.source_identity_present||
       a.weights.source_codebook_id!=19||a.weights.source_is_superblock||
       !qfnGroupedHCUpShortKExperiment())
        return HCGroupedShortKSchedule::Disabled;
    int ordinal=-1;
    if(const auto error=hipStreamGetDevice(stream, &ordinal);error!=hipSuccess)
        throw std::runtime_error(hipGetErrorString(error));
    const auto hardware=rocm::nativeQuantisedHardwareForDevice(ordinal);
    if(native_quantised::selectVerifierDispatchTable(hardware)!=native_quantised::VerifierDispatchTable::MI50)
        return HCGroupedShortKSchedule::Disabled;
    // A full native wave shares each activation K32 across 64 output columns.
    // Only M4 was measured; preserve the original schedule for other row buckets.
    return physical==4 && hardware.admits(640,20480)
        ? HCGroupedShortKSchedule::Column64 : HCGroupedShortKSchedule::Column32;
}
#endif

/** @brief Shared original signed-Q8 block arithmetic for one unchanged output owner.
 * @param a Exact native matrix, published activation banks and scales.
 * @param group Original K32 contribution index.
 * @param column Original branch-major projection output index.
 * @param p Primary contribution; rounded exactly as the standalone short-K program.
 * @param q Residual contribution; folded independently by the output owner.
 */
__device__ __forceinline__ void shortKBlockContribution(const GPUQ8ResidualCorrectionArgs& a,
 unsigned group,std::uint64_t column,float& p,float& q) {
 hc_short_k_native::contribution(a,group,column,p,q);
}
/**
 * @brief Compute independent K32 contributions, then retain the original ordered fold.
 * @tparam Groups Actual K32 group count; no inactive reduction capacity is admitted.
 * @param a Original native source, publication banks and M1 device-row authority.
 *
 * Each K32 group contributes 32 adjacent columns: one CUDA warp, or half a
 * gfx906 wave. Every contributor writes both FP32 terms into distinct LDS cells.
 * The group0 output owners fold group0..Groups-1 with the exact scalar add
 * contract, and add the two complete terms once.
 * The static storage and single workgroup barrier introduce no global workspace
 * or extra publication. Even zero-live or final-tile lanes reach the same barrier.
 */
template<int Groups>
__global__ __launch_bounds__(Groups*kQ8ResidualShortKColumns,1) void projectShortK(GPUQ8ResidualCorrectionArgs a) {
 static_assert(Groups>=2&&Groups<=16);
 constexpr unsigned Columns=kQ8ResidualShortKColumns;
 __shared__ float primary[Groups][Columns],residual[Groups][Columns];
 const unsigned group=threadIdx.x/Columns,localColumn=threadIdx.x%Columns;
 const auto column=static_cast<std::uint64_t>(blockIdx.x)*Columns+localColumn;
 const int live=a.rows.activeRows();
 float p=0.f,q=0.f;
 if(live>0&&column<static_cast<unsigned>(a.weights.n)) {
  shortKBlockContribution(a,group,column,p,q);
 }
 primary[group][localColumn]=p;
 residual[group][localColumn]=q;
 __syncthreads();
 if(group==0&&live>0&&column<static_cast<unsigned>(a.weights.n)) {
  float sum0=0.f,sum1=0.f;
#pragma unroll
  for(int block=0;block<Groups;++block) {
   sum0=device_fp32_contract::add(sum0,primary[block][localColumn]);
   sum1=device_fp32_contract::add(sum1,residual[block][localColumn]);
  }
  a.output[column]=device_fp32_contract::add(sum0,sum1);
 }
}
/**
 * @brief Finish the four-branch read with exactly the original short-K producer grid.
 * @tparam Groups Complete original K32 count; no padded groups or branch loops.
 * @param read Original Q8 banks, retained logits and normalized/mixed arena operands.
 *
 * Each original 32-column tile becomes four contiguous eight-column branch
 * segments. The same Groups*32 lanes and 2*Groups*32 floats of LDS author the
 * same number of native weight blocks as projectShortK. Each output owner folds
 * primary/residual in increasing K order and adds them once. Its now-dead primary
 * LDS column carries that exact logit across one additional CTA barrier; eight
 * owners then execute the original branch-order sigmoid/FMA read. Gate logits
 * remain public outputs, including the final partial tile. No inter-CTA state,
 * redundant dot, normalization, global barrier or allocation is introduced.
 */
template<int Groups>
__global__ __launch_bounds__(Groups*kQ8ResidualShortKColumns,1)
void projectHCShortKRead(GPUHCResidualReadProjection read) {
 static_assert(Groups>=2&&Groups<=16);
 constexpr unsigned Columns=32,MixedColumns=8;
 __shared__ float primary[Groups][Columns],residual[Groups][Columns];
 const auto a=read.down.projection;
 const unsigned group=threadIdx.x/Columns,lane=threadIdx.x%Columns;
 const unsigned branch=lane/MixedColumns,local=lane%MixedColumns;
 const unsigned hidden=static_cast<unsigned>(a.weights.n/4);
 const auto mixedColumn=static_cast<std::uint64_t>(blockIdx.x)*MixedColumns+local;
 const auto column=static_cast<std::uint64_t>(branch)*hidden+mixedColumn;
 const int live=a.rows.activeRows();
 float p=0.f,q=0.f;
 if(live>0&&mixedColumn<hidden)shortKBlockContribution(a,group,column,p,q);
 primary[group][lane]=p;residual[group][lane]=q;
 __syncthreads();
 if(group==0&&live>0&&mixedColumn<hidden) {
  float sum0=0.f,sum1=0.f;
#pragma unroll
  for(int block=0;block<Groups;++block) {
   sum0=device_fp32_contract::add(sum0,primary[block][lane]);
   sum1=device_fp32_contract::add(sum1,residual[block][lane]);
  }
  const float value=device_fp32_contract::add(sum0,sum1);
  a.output[column]=value;
  // Every output owner reads only its own LDS column, so this overwrite cannot
  // race another owner still completing the unchanged ascending-K fold.
  primary[0][lane]=value;
 }
 __syncthreads();
 if(group==0&&lane<MixedColumns&&live>0&&mixedColumn<hidden) {
  float mixed=0.f;
#pragma unroll
  for(unsigned b=0;b<4;++b) {
   const float gate=1.0f/(1.0f+expf(-primary[0][b*MixedColumns+lane]));
   mixed=fmaf(gate,read.normalized[static_cast<std::uint64_t>(b)*hidden+mixedColumn],mixed);
  }
  read.mixed[mixedColumn]=mixed*0.25f;
 }
}
/**
 * @brief Native small-row paired dots sharing one compact block across real rows.
 * @tparam C Actual physical execution decoder; original source identity is validated cold.
 * @tparam Rows Bounded real-row reuse; each output keeps one ascending K32 fold per term.
 */
template <std::uint8_t C, int Rows>
__global__ __launch_bounds__(128, 2) void projectNative(GPUQ8ResidualCorrectionArgs a,
                                                        NativeEmbeddingIQTables tables)
{
    using Traits = NativeDecoder::CodebookTraits<C>;
    const auto columnTiles = (static_cast<unsigned>(a.weights.n) + 127) / 128;
    const unsigned column = (blockIdx.x % columnTiles) * 128 + threadIdx.x;
    const auto rowBase = static_cast<std::uint64_t>(blockIdx.x / columnTiles) * Rows;
    const int live = a.rows.activeRows();
    if (column >= static_cast<unsigned>(a.weights.n) || rowBase >= static_cast<unsigned>(live))
        return;
    NativeDecoder decoder(tables);
    float primary[Rows]{}, residual[Rows]{};
    for (unsigned group = 0; group < a.weights.blocks_per_row; ++group)
    {
        const auto block = static_cast<std::uint64_t>(group) * a.weights.n + column;
        const auto *payload = a.weights.payload + block * Traits::payload_bytes;
        int weights[8];
        decoder.decode_groups<C>(payload, weights);
        const NativeBlock<C> meta(a.weights, block, payload);
#pragma unroll
        for (int r = 0; r < Rows; ++r)
        {
            const auto row = rowBase + r;
            if (row >= static_cast<unsigned>(live))
                break;
            const auto index = row * a.weights.blocks_per_row + group;
            const auto *p = reinterpret_cast<const int *>(a.primary + index * 32);
            const auto *q = reinterpret_cast<const int *>(a.residual + index * 32);
            int dp[2]{}, dq[2]{}, sp[4]{}, sq[4]{};
#pragma unroll
            for (int word = 0; word < 8; ++word)
            {
                constexpr int ones = 0x01010101;
                const int half = Traits::is_dual_scale ? word / 4 : 0;
                dp[half] = dot4(p[word], weights[word], dp[half]);
                dq[half] = dot4(q[word], weights[word], dq[half]);
                if constexpr (Traits::is_asymmetric || Traits::is_dual_scale_asym || Traits::is_iq1_m)
                {
                    sp[word / 2] = dot4(p[word], ones, sp[word / 2]);
                    sq[word / 2] = dot4(q[word], ones, sq[word / 2]);
                }
            }
            primary[r] = device_fp32_contract::add(
                primary[r], nativeContribution<C>(decoder, meta, dp[0], dp[1], a.primaryScales[index], sp));
            residual[r] = device_fp32_contract::add(
                residual[r], nativeContribution<C>(decoder, meta, dq[0], dq[1], a.residualScales[index], sq));
        }
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r)
        if (rowBase + r < static_cast<unsigned>(live))
            a.output[(rowBase + r) * a.weights.n + column] =
                device_fp32_contract::add(primary[r], residual[r]);
}
/** @brief Validate only static prepared identities and admitted pointer geometry. */
inline Q8R_STREAM validate(const GPUQ8ResidualCorrectionArgs &a, void *stream, NativeEmbeddingIQTables tables,
                           const GPUHCResidualDownPublication* hc=nullptr,
                           const GPUHCResidualReadProjection* read=nullptr)
{
    validateQ8ResidualCorrectionGeometry(a.rows.physicalRows(), a.weights.n, a.weights.k);
    const auto native = static_cast<Q8R_STREAM>(stream);
#if defined(Q8_RESIDUAL_CUDA)
    const bool exact = native && native != cudaStreamLegacy && native != cudaStreamPerThread;
    const bool quantizer = a.quantizer == Q8ResidualQuantizerPolicy::CUDAFloatScaleNativeReciprocal;
#else
    const bool exact = native && native != hipStreamLegacy && native != hipStreamPerThread;
    const bool quantizer = a.quantizer == Q8ResidualQuantizerPolicy::ROCmFiniteHalfNativeReciprocal;
#endif
    const auto physical =
        requireQ8ResidualNativeMatrix(a.weights, a.rows.physicalRows(), a.weights.n, a.weights.k);
    if (!exact || !quantizer || !a.input || !a.output || a.input == a.output || !a.primary || !a.residual ||
        a.primary == a.residual || !a.primaryScales || !a.residualScales ||
        a.primaryScales == a.residualScales || reinterpret_cast<std::uintptr_t>(a.weights.payload) % 16 ||
        reinterpret_cast<std::uintptr_t>(a.primary) % 16 || reinterpret_cast<std::uintptr_t>(a.residual) % 16)
        throw std::invalid_argument(
            "TwoTerm GPU requires exact non-default stream and distinct aligned admitted banks");
    const auto values = static_cast<std::size_t>(a.rows.physicalRows()) * a.weights.k;
    const auto outputValues = static_cast<std::size_t>(a.rows.physicalRows()) * a.weights.n;
    const auto weightBlocks = static_cast<std::size_t>(a.weights.n) * a.weights.blocks_per_row;
    const std::array writes{
        Q8ResidualStorageSpan{a.output, outputValues * sizeof(float), alignof(float)},
        Q8ResidualStorageSpan{a.primary, values, 16}, Q8ResidualStorageSpan{a.residual, values, 16},
        Q8ResidualStorageSpan{a.primaryScales, values / 32 * sizeof(float), alignof(float)},
        Q8ResidualStorageSpan{a.residualScales, values / 32 * sizeof(float), alignof(float)},
        Q8ResidualStorageSpan{hc?hc->activated:nullptr,hc?values*sizeof(float):0,alignof(float)},
        Q8ResidualStorageSpan{read?read->mixed:nullptr,read?outputValues/4*sizeof(float):0,alignof(float)}};
    const bool grids = q8ResidualNeedsIQTables(a.weights.codebook_id);
    if (grids &&
        (!tables.iq3s || !tables.iq3xxs || !tables.iq2s || !tables.iq2xs || !tables.iq2xxs || !tables.iq1s))
        throw std::invalid_argument(
            "TwoTerm native IQ projection requires its cold generation-bound original table view");
    const std::array reads{
        Q8ResidualStorageSpan{a.input,hc&&hc->activated==a.input?0:values*sizeof(float),alignof(float)},
        // A reusable weight slot owns its full declared payload capacity even
        // when this invocation decodes a compact prefix. Scratch may not alias it.
        Q8ResidualStorageSpan{a.weights.payload,
                              weightBlocks * (a.weights.allocation_payload_bytes_per_block
                                                  ? a.weights.allocation_payload_bytes_per_block
                                                  : physical.payload_bytes_per_block),
                              16},
        Q8ResidualStorageSpan{a.weights.scales, weightBlocks * sizeof(std::uint16_t), alignof(std::uint16_t)},
        Q8ResidualStorageSpan{a.weights.mins, a.weights.mins ? weightBlocks * sizeof(std::uint16_t) : 0,
                              alignof(std::uint16_t)},
        Q8ResidualStorageSpan{a.weights.emins, a.weights.emins ? weightBlocks * sizeof(std::uint32_t) : 0,
                              alignof(std::uint32_t)},
        Q8ResidualStorageSpan{a.rows.countOwner(), a.rows.countOwner() ? sizeof(std::int32_t) : 0,
                              alignof(std::int32_t)},
        Q8ResidualStorageSpan{tables.iq3s, grids ? 512 * sizeof(std::uint32_t) : 0, alignof(std::uint32_t)},
        Q8ResidualStorageSpan{tables.iq3xxs, grids ? 256 * sizeof(std::uint32_t) : 0, alignof(std::uint32_t)},
        Q8ResidualStorageSpan{tables.iq2s, grids ? 1024 * sizeof(std::uint64_t) : 0, alignof(std::uint64_t)},
        Q8ResidualStorageSpan{tables.iq2xs, grids ? 512 * sizeof(std::uint64_t) : 0, alignof(std::uint64_t)},
        Q8ResidualStorageSpan{tables.iq2xxs, grids ? 256 * sizeof(std::uint64_t) : 0, alignof(std::uint64_t)},
        Q8ResidualStorageSpan{tables.iq1s, grids ? 2048 * sizeof(std::uint64_t) : 0, alignof(std::uint64_t)},
        Q8ResidualStorageSpan{read?read->normalized:nullptr,read?outputValues*sizeof(float):0,alignof(float)}};
    if(hc&&!hc->activated)throw std::invalid_argument("HC GPU requires its retained FP32 activation output");
    if(read&&(!hc||!read->normalized||!read->mixed||a.weights.n%4))
        throw std::invalid_argument("HC fused read requires four complete branches and distinct normalized/mixed operands");
    validateQ8ResidualStorage(writes, reads);
    return native;
}

/**
 * @brief Own four branch projections and their exact mixed-channel epilogue.
 * @tparam C Actual unchanged prepared codebook, including native metadata planes.
 * @param read Complete retained HC publication, original branches and outputs.
 * @param tables Original immutable IQ decoder tables.
 *
 * One K32 group/branch/column tuple owns each contribution. Sixteen mixed columns
 * per CTA give each branch a contiguous 64-byte weight segment; two branch
 * workers each visit two branches, retaining more decode CTAs without splitting
 * a warp's accesses into four short segments. A bounded LDS tile carries
 * those values to the original ascending-K fold; long K visits consecutive tiles
 * without changing that fold. The four final terms are added once per branch,
 * stored to the original logits, then consumed by the original four ordered FMAs.
 * Rows own independent CTAs, so grouped and serial arithmetic are identical.
 */
template<std::uint8_t C>
__global__ __launch_bounds__(512,1) void projectHCRead(GPUHCResidualReadProjection read,
                                                      NativeEmbeddingIQTables tables)
{
    using Traits=NativeDecoder::CodebookTraits<C>;
    constexpr unsigned Columns=16,Groups=16,Branches=4,BranchWorkers=2;
    const auto a=read.down.projection;
    const unsigned hidden=static_cast<unsigned>(a.weights.n/4);
    const unsigned tiles=(hidden+Columns-1)/Columns;
    const unsigned row=blockIdx.x/tiles,column=(blockIdx.x%tiles)*Columns+threadIdx.x%Columns;
    if(row>=static_cast<unsigned>(a.rows.activeRows()))return;
    const unsigned groupLane=threadIdx.x/(Columns*BranchWorkers),localColumn=threadIdx.x%Columns;
    const unsigned branchLane=(threadIdx.x/Columns)%BranchWorkers;
    // Each simultaneous branch pair spans distinct bank lanes. Consecutive K
    // groups retain the same physical ordering as the complete fixed-order fold.
    __shared__ float primary[Groups][Branches][Columns],residual[Groups][Branches][Columns];
    float sum0[4]{},sum1[4]{};
    NativeDecoder decoder(tables);
    for(unsigned start=0;start<a.weights.blocks_per_row;start+=Groups)
    {
        const unsigned group=start+groupLane;
#pragma unroll
        for(unsigned branch=branchLane;branch<Branches;branch+=BranchWorkers)
        {
            float p=0.f,q=0.f;
            if(group<a.weights.blocks_per_row&&column<hidden)
            {
                const auto block=static_cast<std::uint64_t>(group)*a.weights.n+branch*hidden+column;
                const auto* payload=a.weights.payload+block*Traits::payload_bytes;
                int weights[8];decoder.decode_groups<C>(payload,weights);
                const NativeBlock<C> meta(a.weights,block,payload);
                const auto activation=static_cast<std::uint64_t>(row)*a.weights.blocks_per_row+group;
                const auto* p0=reinterpret_cast<const int*>(a.primary+activation*32);
                const auto* p1=reinterpret_cast<const int*>(a.residual+activation*32);
                int dp[2]{},dq[2]{},sp[4]{},sq[4]{};
#pragma unroll
                for(int word=0;word<8;++word)
                {
                    const int half=Traits::is_dual_scale?word/4:0;
                    dp[half]=dot4(p0[word],weights[word],dp[half]);
                    dq[half]=dot4(p1[word],weights[word],dq[half]);
                    if constexpr(Traits::is_asymmetric||Traits::is_dual_scale_asym||Traits::is_iq1_m)
                    {
                        sp[word/2]=dot4(p0[word],0x01010101,sp[word/2]);
                        sq[word/2]=dot4(p1[word],0x01010101,sq[word/2]);
                    }
                }
                p=nativeContribution<C>(decoder,meta,dp[0],dp[1],a.primaryScales[activation],sp);
                q=nativeContribution<C>(decoder,meta,dq[0],dq[1],a.residualScales[activation],sq);
            }
            primary[groupLane][branch][localColumn]=p;
            residual[groupLane][branch][localColumn]=q;
        }
        __syncthreads();
        if(groupLane==0&&branchLane==0&&column<hidden)
        {
            const unsigned count=min(Groups,a.weights.blocks_per_row-start);
#pragma unroll
            for(unsigned branch=0;branch<4;++branch)
                for(unsigned group=0;group<count;++group)
                {
                    sum0[branch]=device_fp32_contract::add(sum0[branch],primary[group][branch][localColumn]);
                    sum1[branch]=device_fp32_contract::add(sum1[branch],residual[group][branch][localColumn]);
                }
        }
        // Original owners finish their reads before a later K tile reuses LDS.
        if(start+Groups<a.weights.blocks_per_row)__syncthreads();
    }
    if(groupLane==0&&branchLane==0&&column<hidden)
    {
        float mixed=0.f;
#pragma unroll
        for(unsigned branch=0;branch<4;++branch)
        {
            const auto offset=(static_cast<std::uint64_t>(row)*4+branch)*hidden+column;
            const float value=device_fp32_contract::add(sum0[branch],sum1[branch]);
            a.output[offset]=value;
            const float gate=1.0f/(1.0f+expf(-value));
            mixed=fmaf(gate,read.normalized[offset],mixed);
        }
        read.mixed[static_cast<std::uint64_t>(row)*hidden+column]=mixed*0.25f;
    }
}
/**
 * @brief Launch the exact specialization selected from validated immutable geometry.
 * @param a Original prepared source, live authority and published activation banks.
 * @param geometry Already validated production geometry, with groups2..16.
 * @param stream Exact non-default CUDA/HIP stream used by the publisher.
 * @throws std::logic_error If an internal validated group lacks a specialization.
 * @note The caller validates all metadata before publication; this helper never
 * reads live state, publishes banks, allocates storage or selects another family.
 */
inline void launchOrderedShortK(const GPUQ8ResidualCorrectionArgs &a,
                               const Q8ResidualShortKGeometry &geometry, Q8R_STREAM stream)
{
#define Q8R_SHORT_K_CASE(G) case G: projectShortK<G><<<static_cast<unsigned>(geometry.blocks),geometry.threads,0,stream>>>(a); break
    switch (geometry.groups) {
        Q8R_SHORT_K_CASE(2); Q8R_SHORT_K_CASE(3); Q8R_SHORT_K_CASE(4); Q8R_SHORT_K_CASE(5);
        Q8R_SHORT_K_CASE(6); Q8R_SHORT_K_CASE(7); Q8R_SHORT_K_CASE(8); Q8R_SHORT_K_CASE(9);
        Q8R_SHORT_K_CASE(10); Q8R_SHORT_K_CASE(11); Q8R_SHORT_K_CASE(12); Q8R_SHORT_K_CASE(13);
        Q8R_SHORT_K_CASE(14); Q8R_SHORT_K_CASE(15); Q8R_SHORT_K_CASE(16);
        default: throw std::logic_error("Validated ordered short-K group has no native specialization");
    }
#undef Q8R_SHORT_K_CASE
}

}
/**
 * @brief Submit complete paired-Q8 arithmetic with passive host-submission evidence.
 * @param a Original source/banks, immutable physical shape and device-owned live extent.
 * @param stream Exact non-default native stream retaining all captured owners.
 * @param tables Already cold-declared immutable IQ tables when the source needs them.
 * @param publication Typed already-activated or HC producer declaration.
 * @tparam SingleGroupReference False selects production; true selects the explicit HIP-only original grouped reference.
 * @tparam Publication Compile-time semantic operation; no prepublished-bank mode exists.
 * @throws std::invalid_argument Before publication for invalid source, storage or geometry.
 * @throws std::runtime_error After a native submission error; failed calls are never counted.
 * @note The optional PerfStats record counts accepted host calls, including capture
 * and zero-live calls, not device executions or tokens. Only ordered short-K
 * submissions issue this record, using the same immutable launch/storage geometry.
 * Disabled collection does not construct tags or inspect device state.
 */
template<bool SingleGroupReference=false,class Publication>
static void submitQ8ResidualProjection(const GPUQ8ResidualCorrectionArgs &a, void *stream,
                                       NativeEmbeddingIQTables tables,const Publication& publication)
{
#if defined(Q8_RESIDUAL_CUDA)
    static_assert(!SingleGroupReference,"The original grouped reference belongs only to HIP");
#endif
    if constexpr(SingleGroupReference)
        if(a.weights.codebook_id!=19||a.rows.physicalRows()<32||a.weights.n<64)
            throw std::invalid_argument("HIP original grouped reference requires codebook19, physical rows>=32 and N>=64");
    const GPUHCResidualDownPublication* hc=nullptr;
    if constexpr(std::is_same_v<Publication,GPUHCResidualDownPublication>)hc=&publication;
    const auto native = Q8R_NAMESPACE::validate(a, stream, tables,hc);
    const auto tasks = static_cast<std::uint64_t>(a.rows.physicalRows()) * a.weights.blocks_per_row;
#if defined(Q8_RESIDUAL_CUDA)
    const auto geometry =
        q8ResidualGroupedGeometry(Q8ResidualGroupedBackend::CUDA, a.rows.physicalRows(), a.weights.n, a.weights.codebook_id);
#else
    const auto geometry =
        q8ResidualGroupedGeometry(Q8ResidualGroupedBackend::ROCm, a.rows.physicalRows(), a.weights.n, a.weights.codebook_id);
#endif
    const auto shortGeometry = q8ResidualShortKGeometry(
#if defined(Q8_RESIDUAL_CUDA)
        Q8ResidualGroupedBackend::CUDA,
#else
        Q8ResidualGroupedBackend::ROCm,
#endif
        a.rows.physicalRows(), a.weights.n, a.weights.k, a.weights.codebook_id);
    if (shortGeometry && a.weights.blocks_per_row != static_cast<unsigned>(shortGeometry->groups))
        throw std::invalid_argument("Ordered short-K source groups differ from retained geometry");
#if !defined(Q8_RESIDUAL_CUDA)
    using Q8R_NAMESPACE::HCGroupedShortKSchedule;
    auto groupedHCShortK=HCGroupedShortKSchedule::Disabled;
    if constexpr (!SingleGroupReference && std::is_same_v<Publication,GPUHCResidualDownPublication>)
        groupedHCShortK=Q8R_NAMESPACE::selectHCGroupedShortK(a,native);
#endif
    const auto grid = shortGeometry ? shortGeometry->blocks : geometry.blocks;
    if ((tasks + 7) / 8 > INT_MAX || grid > INT_MAX)
        throw std::invalid_argument("Two-term GPU Q8 retained launch grid exceeds native capacity");
    if constexpr(std::is_same_v<Publication,GPUHCResidualDownPublication>) {
        if(publication.source==HCDownQ8PublicationSource::Standalone)
            Q8R_NAMESPACE::publishHCResidualDown<<<static_cast<unsigned>((tasks + 7) / 8),256,0,native>>>(publication);
        else if(publication.source!=HCDownQ8PublicationSource::OriginalReducer)
            throw std::invalid_argument("Unknown HC down publication source");
    } else
        Q8R_NAMESPACE::publish<<<static_cast<unsigned>((tasks + 7) / 8), 256, 0, native>>>(a);
    // The consumer is submitted only after the exact producer enqueue succeeds.
    // Both native nodes retain the same immutable bank/row tuple and stream.
    const auto publicationError=Q8R_ERROR();
    if(publicationError!=Q8R_SUCCESS)
        throw std::runtime_error(std::string("Two-term Q8 publication failed: ")+Q8R_ERROR_TEXT(publicationError));
    if constexpr(SingleGroupReference)
    {
#if !defined(Q8_RESIDUAL_CUDA)
        Q8R_NAMESPACE::projectGrouped<1><<<static_cast<unsigned>(grid),geometry.threads,0,native>>>(a);
#endif
    }
#if !defined(Q8_RESIDUAL_CUDA)
    else if(groupedHCShortK==HCGroupedShortKSchedule::Column64)
        Q8R_NAMESPACE::projectHCGroupedShortKWide<<<160,640,0,native>>>(a);
    else if(groupedHCShortK==HCGroupedShortKSchedule::Column32)
        Q8R_NAMESPACE::projectHCGroupedShortK<<<dim3(320,(a.rows.physicalRows()+3)/4),320,0,native>>>(a);
#endif
    else if (shortGeometry)
        Q8R_NAMESPACE::launchOrderedShortK(a, *shortGeometry, native);
    else visitQ8ResidualNativeCodebook(a.weights.codebook_id, [&]<std::uint8_t C>() {
        if constexpr (C == 19)
        {
            if (geometry.strategy == Q8ResidualGroupedStrategy::CooperativeMatrix)
#if defined(Q8_RESIDUAL_CUDA)
                Q8R_NAMESPACE::projectGrouped<<<static_cast<unsigned>(grid), 256, 0, native>>>(a);
#else
                Q8R_NAMESPACE::projectGrouped<0><<<static_cast<unsigned>(grid), 256, 0, native>>>(a);
#endif
            else if (geometry.rowTile == 1)
                Q8R_NAMESPACE::project<1><<<static_cast<unsigned>(grid), 128, 0, native>>>(a);
            else
                Q8R_NAMESPACE::project<4><<<static_cast<unsigned>(grid), 128, 0, native>>>(a);
        }
        else
        {
            if (geometry.strategy == Q8ResidualGroupedStrategy::CooperativeMatrix)
                Q8R_NAMESPACE::projectNativeGrouped<C>
                    <<<static_cast<unsigned>(grid), 256, 0, native>>>(a, tables);
            else if (geometry.rowTile == 1)
                Q8R_NAMESPACE::projectNative<C, 1>
                    <<<static_cast<unsigned>(grid), 128, 0, native>>>(a, tables);
            else
                Q8R_NAMESPACE::projectNative<C, 4>
                    <<<static_cast<unsigned>(grid), 128, 0, native>>>(a, tables);
        }
    });
    const auto error = Q8R_ERROR();
    if (error != Q8R_SUCCESS)
        throw std::runtime_error(std::string("Two-term Q8 native launch failed: ") + Q8R_ERROR_TEXT(error));
    // This is accepted host submission/capture evidence only. Retained native
    // replay does not revisit this factory or increment this counter. No live
    // count, device ordinal or compiler resource query is performed here.
#if !defined(Q8_RESIDUAL_CUDA)
    if (groupedHCShortK!=HCGroupedShortKSchedule::Disabled && PerfStatsCollector::isDomainEnabled("kernel"))
        PerfStatsCollector::addCounter("kernel","q8_residual_grouped_hc_short_k_submissions",
            1.0,"submission",{},{{"physical_rows",std::to_string(a.rows.physicalRows())}});
#endif
    if (shortGeometry && PerfStatsCollector::isDomainEnabled("kernel"))
    {
#if defined(Q8_RESIDUAL_CUDA)
        constexpr const char *backend = "cuda";
#else
        constexpr const char *backend = "rocm";
#endif
        PerfStatsCollector::addCounter("kernel", "q8_residual_ordered_short_k_submissions", 1.0,
            "submission", {}, {{"backend", backend}, {"observation", "host_execute_or_capture"},
            {"kernel_family", "ordered_short_k"},
            {"physical_rows", std::to_string(a.rows.physicalRows())},
            {"n", std::to_string(a.weights.n)}, {"k", std::to_string(a.weights.k)},
            {"execution_codebook", std::to_string(a.weights.codebook_id)},
            {"launch_blocks", std::to_string(shortGeometry->blocks)},
            {"launch_threads", std::to_string(shortGeometry->threads)},
            {"static_shared_bytes", std::to_string(shortGeometry->shared_bytes)}});
    }
}
/** @brief Original activated input retains its original publication and projection. */
void Q8R_FACTORY(const GPUQ8ResidualCorrectionArgs& a,void* stream,NativeEmbeddingIQTables tables) {
 submitQ8ResidualProjection(a,stream,tables,a);
}
/** @brief HC input earns publication in the same invocation before the unchanged projection. */
void Q8R_HC_FACTORY(const GPUHCResidualDownPublication& hc,void* stream,NativeEmbeddingIQTables tables) {
 submitQ8ResidualProjection(hc.projection,stream,tables,hc);
}
/** @brief Submit only the sealed M1 signed-Q8 producer-shaped read candidate. */
void Q8R_HC_SHORT_READ_FACTORY(const GPUHCResidualReadProjection& read,void* stream) {
 const auto& a=read.down.projection;
 const auto native=Q8R_NAMESPACE::validate(a,stream,{},&read.down,&read);
 const auto geometry=q8ResidualShortKGeometry(
#if defined(Q8_RESIDUAL_CUDA)
     Q8ResidualGroupedBackend::CUDA,
#else
     Q8ResidualGroupedBackend::ROCm,
#endif
     a.rows.physicalRows(),a.weights.n,
     a.weights.k,a.weights.codebook_id);
 if(!geometry||a.weights.n%4)
  throw std::invalid_argument("HC short-K read requires original M1 signed-Q8 short-K and four branches");
 // ceil((4H)/32)==ceil(H/8): retain the original producer's actual grid even
 // for the partial mixed-channel tile. Device-owned zero-live work stays empty.
 if(read.down.source==HCDownQ8PublicationSource::Standalone)
  Q8R_NAMESPACE::publishHCResidualDown<<<(a.weights.blocks_per_row+7)/8,256,0,native>>>(read.down);
 else if(read.down.source!=HCDownQ8PublicationSource::OriginalReducer)
  throw std::invalid_argument("Unknown HC read publication source");
 const auto published=Q8R_ERROR();
 if(published!=Q8R_SUCCESS)
  throw std::runtime_error(std::string("HC short-K read publication failed: ")+Q8R_ERROR_TEXT(published));
#define Q8R_HC_SHORT_READ_CASE(G) case G: Q8R_NAMESPACE::projectHCShortKRead<G><<<static_cast<unsigned>(geometry->blocks),geometry->threads,0,native>>>(read); break
 switch(geometry->groups) {
  Q8R_HC_SHORT_READ_CASE(2); Q8R_HC_SHORT_READ_CASE(3); Q8R_HC_SHORT_READ_CASE(4);
  Q8R_HC_SHORT_READ_CASE(5); Q8R_HC_SHORT_READ_CASE(6); Q8R_HC_SHORT_READ_CASE(7);
  Q8R_HC_SHORT_READ_CASE(8); Q8R_HC_SHORT_READ_CASE(9); Q8R_HC_SHORT_READ_CASE(10);
  Q8R_HC_SHORT_READ_CASE(11); Q8R_HC_SHORT_READ_CASE(12); Q8R_HC_SHORT_READ_CASE(13);
  Q8R_HC_SHORT_READ_CASE(14); Q8R_HC_SHORT_READ_CASE(15); Q8R_HC_SHORT_READ_CASE(16);
  default: throw std::logic_error("Validated HC short-K read group has no specialization");
 }
#undef Q8R_HC_SHORT_READ_CASE
 const auto projected=Q8R_ERROR();
 if(projected!=Q8R_SUCCESS)
  throw std::runtime_error(std::string("HC short-K read projection failed: ")+Q8R_ERROR_TEXT(projected));
}
/** @brief Submit a shared original publisher and one complete four-branch read consumer. */
void Q8R_HC_READ_FACTORY(const GPUHCResidualReadProjection& read,void* stream,NativeEmbeddingIQTables tables)
{
    const auto& a=read.down.projection;
    const auto native=Q8R_NAMESPACE::validate(a,stream,tables,&read.down,&read);
    const auto groups=a.weights.blocks_per_row;
    const auto tasks=static_cast<std::uint64_t>(a.rows.physicalRows())*groups;
    const auto tiles=(static_cast<std::uint64_t>(a.weights.n/4)+15)/16;
    const auto blocks=static_cast<std::uint64_t>(a.rows.physicalRows())*tiles;
    if((tasks+7)/8>INT_MAX||blocks>INT_MAX)
        throw std::invalid_argument("HC fused read launch geometry exceeds native capacity");
    if(read.down.source==HCDownQ8PublicationSource::Standalone)
    Q8R_NAMESPACE::publishHCResidualDown<<<static_cast<unsigned>((tasks+7)/8),256,0,native>>>(read.down);
 else if(read.down.source!=HCDownQ8PublicationSource::OriginalReducer)
    throw std::invalid_argument("Unknown HC read publication source");
    const auto published=Q8R_ERROR();
    if(published!=Q8R_SUCCESS)
        throw std::runtime_error(std::string("HC fused read publication failed: ")+Q8R_ERROR_TEXT(published));
    const unsigned threads=std::min(groups,16u)*32;
    visitQ8ResidualNativeCodebook(a.weights.codebook_id,[&]<std::uint8_t C>() {
        Q8R_NAMESPACE::projectHCRead<C><<<static_cast<unsigned>(blocks),threads,0,native>>>(read,tables);
    });
    const auto projected=Q8R_ERROR();
    if(projected!=Q8R_SUCCESS)
        throw std::runtime_error(std::string("HC fused read projection failed: ")+Q8R_ERROR_TEXT(projected));
}
#if !defined(Q8_RESIDUAL_CUDA)
/**
 * @brief Submit original single-group ordinary arithmetic through the shared publication boundary.
 * @param a Original source, row authority, inputs and admitted output/scratch banks.
 * @param stream Exact non-default publication/projection stream.
 * @throws std::exception For any shared admission or native submission failure.
 */
void rocmQ8ResidualGroupedSingleGroupReference(const GPUQ8ResidualCorrectionArgs &a,void *stream)
{
    submitQ8ResidualProjection<true>(a,stream,{},a);
}
/**
 * @brief Submit original single-group HC arithmetic with complete publication and retained FP32 output.
 * @param hc Original projection tuple and activated FP32 destination, including exact-alias policy.
 * @param stream Exact non-default publication/projection stream.
 * @throws std::exception For any shared admission or native submission failure.
 */
void rocmHCResidualDownGroupedSingleGroupReference(const GPUHCResidualDownPublication &hc,void *stream)
{
    submitQ8ResidualProjection<true>(hc.projection,stream,{},hc);
}
#endif
/**
 * @brief Submit the immutable original-column M1 arithmetic as a diagnostic reference.
 * @param a Original signed-Q8 prepared source, physicalM1 row authority and admitted banks.
 * @param stream Exact native stream retaining the source, scratch and output owners.
 * @throws std::invalid_argument Before publication for non-M1 or non-signed-Q8 sources.
 * @throws std::runtime_error When either original native launch fails admission.
 * @note This named oracle never consults production geometry selection. It is
 * not a public arithmetic policy, automatic alternative or inference fallback.
 */
void Q8R_COLUMN_REFERENCE(const GPUQ8ResidualCorrectionArgs &a, void *stream)
{
    if (a.rows.physicalRows() != 1 || a.weights.codebook_id != 19)
        throw std::invalid_argument("Original-column reference requires physicalM1 and native signed-Q8");
    const auto native = Q8R_NAMESPACE::validate(a, stream, {});
    const auto tasks = static_cast<std::uint64_t>(a.weights.blocks_per_row);
    const auto blocks = (static_cast<std::uint64_t>(a.weights.n) + 127) / 128;
    if ((tasks + 7) / 8 > INT_MAX || blocks > INT_MAX)
        throw std::invalid_argument("Original-column reference grid exceeds native capacity");
    Q8R_NAMESPACE::publish<<<static_cast<unsigned>((tasks + 7) / 8), 256, 0, native>>>(a);
    Q8R_NAMESPACE::project<1><<<static_cast<unsigned>(blocks), 128, 0, native>>>(a);
    const auto error = Q8R_ERROR();
    if (error != Q8R_SUCCESS)
        throw std::runtime_error(std::string("Original-column reference launch failed: ") + Q8R_ERROR_TEXT(error));
}
/** @brief Preserve the original table-free external ABI through the same launcher. */
void Q8R_FACTORY(const GPUQ8ResidualCorrectionArgs &a, void *stream)
{
    Q8R_FACTORY(a, stream, {});
}
}
#undef Q8R_STREAM
#undef Q8R_FACTORY
#undef Q8R_HC_FACTORY
#undef Q8R_HC_READ_FACTORY
#undef Q8R_HC_SHORT_READ_FACTORY
#undef Q8R_COLUMN_REFERENCE
#undef Q8R_NAMESPACE
#undef Q8R_ERROR
#undef Q8R_SUCCESS
#undef Q8R_ERROR_TEXT
#undef Q8R_SHFL
