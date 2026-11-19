/**
 * @file ROCmSharedExpertGateAddBody.cuh
 * @brief Original shared-input gate arithmetic indexed independently of native grids.
 *
 * Ordinary kernels and resident transactions share this exact 256-lane dot/tree,
 * sigmoid and separately rounded product/add publication. Only logical token
 * ownership is explicit; no source, row count or output role is reconstructed.
 */
#pragma once
#include "ROCmMoEWaveReduction.h"
#include "kernels/common/DeviceFP32NumericalContract.h"
#include <hip/hip_runtime.h>
namespace llaminar2::rocm::moe_shared_gate
{
/**
 * @brief Original physical256-lane shared-input gate dot and sigmoid, without output publication.
 * @param x Original full-H row. @param gate_inp Original FP32 input-gate source.
 * @param d_model Original full width. @param warp_sums CTA-owned16-float scratch.
 * @return Original sigmoid value, identically available to all CTA lanes.
 * @pre All256 lanes call; scratch is reused only after this helper's final CTA barrier.
 * @details This is a literal extraction of gateAdd's original lane loops and
 * wave/block tree. No scalar, partitioned or reassociated replacement is used.
 */
__device__ __forceinline__ float gateValue256(const float* x,const float* gate_inp,int d_model,float* warp_sums)
{
    float sum = 0.0f;
    if ((d_model & 3) == 0)
    {
        const int d4 = d_model >> 2;
        const float4 *__restrict__ x4 = reinterpret_cast<const float4 *>(x);
        const float4 *__restrict__ gate4 = reinterpret_cast<const float4 *>(gate_inp);
        for (int i = threadIdx.x; i < d4; i += blockDim.x)
        {
            const float4 xv = x4[i];
            const float4 gv = gate4[i];
            sum += xv.x * gv.x + xv.y * gv.y + xv.z * gv.z + xv.w * gv.w;
        }
    }
    else
    {
        for (int i = threadIdx.x; i < d_model; i += blockDim.x)
            sum += x[i] * gate_inp[i];
    }

    sum = llaminar2::rocm::moe_wave_contract::rocm_moe_wave_reduce_sum_f32(sum);

    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        float block_sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        block_sum = llaminar2::rocm::moe_wave_contract::rocm_moe_wave_reduce_sum_f32(block_sum);
        if (lane == 0)
            warp_sums[0] = 1.0f / (1.0f + expf(-block_sum));
    }
    __syncthreads();

    return warp_sums[0];
}

/**
 * @brief Execute one original gate/add logical row using the whole physical CTA.
 * @tparam ColumnSlice Original independent-column publication policy.
 * @param input Original full-H FP32 row.
 * @param gate_inp Original replicated FP32 input-gate source.
 * @param shared_output Original local shared contribution, gated in place.
 * @param routed_residual Original slot-ordered local routed output.
 * @param combined_output Distinct original local combined FFN output.
 * @param seq_len Physical original row capacity.
 * @param d_model Complete original model width.
 * @param token Logical row assigned to this CTA.
 * @param effective_seq_len Original already-clamped live prefix.
 * @param output_columns Original column-slice width or zero for complete output.
 * @pre All CTA lanes call; block width is the original 256-thread arithmetic geometry.
 */
template<bool ColumnSlice=false>
__device__ __forceinline__ void gateAdd(const float* input,const float* gate_inp,float* shared_output,
    const float* routed_residual,float* combined_output,int seq_len,int d_model,int token,
    int effective_seq_len,int output_columns=0)
{
    if (token >= seq_len) return;

    const int columns = ColumnSlice ? output_columns : d_model;
    const size_t row_offset = static_cast<size_t>(token) * columns;
    if (token >= effective_seq_len)
    {
        if ((columns & 3) == 0)
        {
            const int d4 = columns >> 2;
            float4 *__restrict__ shared4 =
                reinterpret_cast<float4 *>(shared_output + row_offset);
            float4 *__restrict__ combined4 =
                reinterpret_cast<float4 *>(combined_output + row_offset);
            const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int i = threadIdx.x; i < d4; i += blockDim.x)
            {
                shared4[i] = zero;
                combined4[i] = zero;
            }
        }
        else
        {
            for (int i = threadIdx.x; i < columns; i += blockDim.x)
            {
                shared_output[row_offset + i] = 0.0f;
                combined_output[row_offset + i] = 0.0f;
            }
        }
        return;
    }
    const float *x = input + static_cast<size_t>(token) * d_model;

    __shared__ float warp_sums[16];
    const float gate = gateValue256(x,gate_inp,d_model,warp_sums);
    if ((columns & 3) == 0)
    {
        const int d4 = columns >> 2;
        float4 *__restrict__ shared4 =
            reinterpret_cast<float4 *>(shared_output + row_offset);
        const float4 *__restrict__ routed4 =
            reinterpret_cast<const float4 *>(routed_residual + row_offset);
        float4 *__restrict__ combined4 =
            reinterpret_cast<float4 *>(combined_output + row_offset);
        for (int i = threadIdx.x; i < d4; i += blockDim.x)
        {
            const float4 s = shared4[i];
            const float4 r = routed4[i];
            const float4 gated = make_float4(
                llaminar2::device_fp32_contract::multiply(gate, s.x),
                llaminar2::device_fp32_contract::multiply(gate, s.y),
                llaminar2::device_fp32_contract::multiply(gate, s.z),
                llaminar2::device_fp32_contract::multiply(gate, s.w));
            shared4[i] = gated;
            // This specialization replaces two previously separate launches.
            // Persist the product with the shared numerical contract: HIP's
            // __fadd_rn alone is plain C++ addition in this toolchain and cannot
            // prevent LLVM contracting the producer into an FMA. The retained
            // register dependency adds no memory access or device barrier.
            combined4[i] = make_float4(
                llaminar2::device_fp32_contract::add(r.x, gated.x),
                llaminar2::device_fp32_contract::add(r.y, gated.y),
                llaminar2::device_fp32_contract::add(r.z, gated.z),
                llaminar2::device_fp32_contract::add(r.w, gated.w));
        }
    }
    else
    {
        for (int i = threadIdx.x; i < columns; i += blockDim.x)
        {
            const float gated_shared = llaminar2::device_fp32_contract::multiply(
                gate, shared_output[row_offset + i]);
            shared_output[row_offset + i] = gated_shared;
            combined_output[row_offset + i] = llaminar2::device_fp32_contract::add(
                routed_residual[row_offset + i], gated_shared);
        }
    }
}
}
