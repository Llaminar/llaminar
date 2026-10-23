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
 */
#pragma once
#if defined(Q8_RESIDUAL_CUDA)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define Q8R_STREAM cudaStream_t
#define Q8R_FACTORY cudaQ8ResidualCorrectedProjection
#define Q8R_HC_FACTORY cudaHCResidualDownQ8Projection
#define Q8R_COLUMN_REFERENCE cudaQ8ResidualOriginalColumnReference
#define Q8R_NAMESPACE cuda_q8_residual
#define Q8R_ERROR cudaGetLastError
#define Q8R_SUCCESS cudaSuccess
#define Q8R_ERROR_TEXT cudaGetErrorString
#define Q8R_SHFL(x,o) __shfl_xor_sync(0xffffffffu,x,o,32)
#else
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "kernels/rocm/ops/ROCmSwiGLUDecodeEquivalentMath.h"
#define Q8R_STREAM hipStream_t
#define Q8R_FACTORY rocmQ8ResidualCorrectedProjection
#define Q8R_HC_FACTORY rocmHCResidualDownQ8Projection
#define Q8R_COLUMN_REFERENCE rocmQ8ResidualOriginalColumnReference
#define Q8R_NAMESPACE rocm_q8_residual
#define Q8R_ERROR hipGetLastError
#define Q8R_SUCCESS hipSuccess
#define Q8R_ERROR_TEXT hipGetErrorString
#define Q8R_SHFL(x,o) __shfl_xor(x,o,32)
#endif
#include "Q8ResidualCorrection.h"
#include "utils/PerfStatsCollector.h"
#include "DeviceFP32NumericalContract.h"
#include "DeviceNativeVNNIContributionContract.h"
#include "Q8ResidualGroupedGeometry.h"
#include "Q8ResidualShortKGeometry.h"
#include "GPUQ8ResidualNative.h"
#include "GPUQ8ResidualNativeDevice.cuh"
#if defined(Q8_RESIDUAL_CUDA)
#include "kernels/cuda/gemm/CUDAQ8ResidualGrouped.cuh"
#else
#include "kernels/rocm/gemm/ROCmQ8ResidualGrouped.cuh"
#endif
#include <cstdint>
#include <climits>
#include <string>
#include <type_traits>
namespace llaminar2 {
namespace Q8R_NAMESPACE {
/** @brief Native signed four-byte integer dot without any floating operand. */
__device__ __forceinline__ int dot4(int a,int b,int c) {
#if defined(Q8_RESIDUAL_CUDA)
 return __dp4a(a,b,c);
#else
 return __builtin_amdgcn_sdot4(a,b,c,false);
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
 const float scaled=a.input[offset]*0.25f;
 const float value=scaled*(1.0f/(1.0f+expf(-scaled)));
 hc.activated[offset]=value;
 if(!isfinite(value))invalidActivation();float maximum=fabsf(value);
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
   primary[r]=device_fp32_contract::add(primary[r],device_native_vnni_contract::singleScaleBlock(dot0,weightScale,a.primaryScales[activationBlock]));
   residual[r]=device_fp32_contract::add(residual[r],device_native_vnni_contract::singleScaleBlock(dot1,weightScale,a.residualScales[activationBlock]));
  }
 }
#pragma unroll
 for(int r=0;r<Rows;++r)if(rowBase+r<static_cast<unsigned>(live))
  a.output[(rowBase+r)*a.weights.n+column]=device_fp32_contract::add(primary[r],residual[r]);
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
  const auto linear=static_cast<std::uint64_t>(group)*a.weights.n+column;
  const auto*payload=reinterpret_cast<const int4*>(a.weights.payload+linear*32);
  const int4 low=payload[0],high=payload[1];
  const int weights[8]{low.x,low.y,low.z,low.w,high.x,high.y,high.z,high.w};
  const auto bits=static_cast<const unsigned short*>(a.weights.scales)[linear];
  const float weightScale=__half2float(__ushort_as_half(bits));
  const auto*p0=reinterpret_cast<const int*>(a.primary+group*32);
  const auto*p1=reinterpret_cast<const int*>(a.residual+group*32);
  int dot0=0,dot1=0;
#pragma unroll
  for(int word=0;word<8;++word) {
   dot0=dot4(p0[word],weights[word],dot0);
   dot1=dot4(p1[word],weights[word],dot1);
  }
  p=device_native_vnni_contract::singleScaleBlock(dot0,weightScale,a.primaryScales[group]);
  q=device_native_vnni_contract::singleScaleBlock(dot1,weightScale,a.residualScales[group]);
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
                           const GPUHCResidualDownPublication* hc=nullptr)
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
        Q8ResidualStorageSpan{hc?hc->activated:nullptr,hc?values*sizeof(float):0,alignof(float)}};
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
        Q8ResidualStorageSpan{tables.iq1s, grids ? 2048 * sizeof(std::uint64_t) : 0, alignof(std::uint64_t)}};
    if(hc&&!hc->activated)throw std::invalid_argument("HC GPU requires its retained FP32 activation output");
    validateQ8ResidualStorage(writes, reads);
    return native;
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
 * @tparam Publication Compile-time semantic operation; no prepublished-bank mode exists.
 * @throws std::invalid_argument Before publication for invalid source, storage or geometry.
 * @throws std::runtime_error After a native submission error; failed calls are never counted.
 * @note The optional PerfStats record counts accepted host calls, including capture
 * and zero-live calls, not device executions or tokens. Only ordered short-K
 * submissions issue this record, using the same immutable launch/storage geometry.
 * Disabled collection does not construct tags or inspect device state.
 */
template<class Publication>
static void submitQ8ResidualProjection(const GPUQ8ResidualCorrectionArgs &a, void *stream,
                                       NativeEmbeddingIQTables tables,const Publication& publication)
{
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
    const auto grid = shortGeometry ? shortGeometry->blocks : geometry.blocks;
    if ((tasks + 7) / 8 > INT_MAX || grid > INT_MAX)
        throw std::invalid_argument("Two-term GPU Q8 retained launch grid exceeds native capacity");
    if constexpr(std::is_same_v<Publication,GPUHCResidualDownPublication>)
        Q8R_NAMESPACE::publishHCResidualDown<<<static_cast<unsigned>((tasks + 7) / 8),256,0,native>>>(publication);
    else
        Q8R_NAMESPACE::publish<<<static_cast<unsigned>((tasks + 7) / 8), 256, 0, native>>>(a);
    // The consumer is submitted only after the exact producer enqueue succeeds.
    // Both native nodes retain the same immutable bank/row tuple and stream.
    const auto publicationError=Q8R_ERROR();
    if(publicationError!=Q8R_SUCCESS)
        throw std::runtime_error(std::string("Two-term Q8 publication failed: ")+Q8R_ERROR_TEXT(publicationError));
    if (shortGeometry)
        Q8R_NAMESPACE::launchOrderedShortK(a, *shortGeometry, native);
    else visitQ8ResidualNativeCodebook(a.weights.codebook_id, [&]<std::uint8_t C>() {
        if constexpr (C == 19)
        {
            if (geometry.strategy == Q8ResidualGroupedStrategy::CooperativeMatrix)
                Q8R_NAMESPACE::projectGrouped<<<static_cast<unsigned>(grid), 256, 0, native>>>(a);
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
#undef Q8R_COLUMN_REFERENCE
#undef Q8R_NAMESPACE
#undef Q8R_ERROR
#undef Q8R_SUCCESS
#undef Q8R_ERROR_TEXT
#undef Q8R_SHFL
