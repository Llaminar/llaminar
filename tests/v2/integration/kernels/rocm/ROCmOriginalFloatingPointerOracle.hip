/**
 * @file ROCmOriginalFloatingPointerOracle.hip
 * @brief Frozen immediate-pre102 staged-array native pipeline for calibration.
 *
 * Literal GPU arithmetic was copied from the pre102 production projection TU.
 * Every symbol is isolated/prefixed. No candidate GPU math/helper is imported.
 * Test setup owns matrices/tables; both staging and consumer launch are captured.
 */
#include "ROCmOriginalFloatingPointerOracle.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
namespace qfn_pointer102_original {
constexpr int pointer102Original_SMALL_N_FP32_BLOCK=256;
constexpr int pointer102Original_SMALL_N_FP32_MAX_BATCH=8;
enum pointer102Original_SmallProjectionWeightDtype {
 pointer102Original_SMALL_PROJECTION_WEIGHT_FP16=0,
 pointer102Original_SMALL_PROJECTION_WEIGHT_BF16=1
};
struct pointer102Original_Fp32BatchedProjectionPointers {
 const float *A[pointer102Original_SMALL_N_FP32_MAX_BATCH]; const float *B[pointer102Original_SMALL_N_FP32_MAX_BATCH]; float *C[pointer102Original_SMALL_N_FP32_MAX_BATCH];
};
/** @brief Frozen immediate-pre102 native body; never share candidate GPU math. */
    __device__ __forceinline__ float pointer102Original_uint32_bits_to_float(uint32_t bits)
    {
        union
        {
            uint32_t u;
            float f;
        } value{bits};
        return value.f;
    }

/** @brief Frozen immediate-pre102 native body; never share candidate GPU math. */
    __device__ __forceinline__ float pointer102Original_fp16_bits_to_float(uint16_t h)
    {
        const uint32_t native = __float_as_uint(__half2float(__ushort_as_half(h)));
        const uint32_t original_nan = (uint32_t(h & 0x8000u) << 16) |
            0x7F800000u | (uint32_t(h & 0x03FFu) << 13);
        const uint32_t nan_mask = 0u - uint32_t((h & 0x7FFFu) > 0x7C00u);
        // A bit select retains every NaN payload without divergent EXEC masks.
        return pointer102Original_uint32_bits_to_float(native ^ ((native ^ original_nan) & nan_mask));
    }

/** @brief Frozen immediate-pre102 native body; never share candidate GPU math. */
    __device__ __forceinline__ float pointer102Original_bf16_bits_to_float(uint16_t bf16)
    {
        return pointer102Original_uint32_bits_to_float(static_cast<uint32_t>(bf16) << 16);
    }

/** @brief Frozen immediate-pre102 native body; never share candidate GPU math. */
    __device__ __forceinline__ float pointer102Original_reduceFloating256(float value, float *scratch)
    {
        const unsigned lane = threadIdx.x;
        scratch[lane] = value;
        __syncthreads();
        for (unsigned stride = 128; stride >= 64; stride >>= 1)
        {
            if (lane < stride) scratch[lane] += scratch[lane + stride];
            __syncthreads();
        }
        if (lane < 64)
        {
            value = scratch[lane];
#pragma unroll
            for (unsigned stride = 32; stride; stride >>= 1)
            {
                const float partner = __shfl_down(value, stride, 64);
                if (lane < stride) value += partner;
            }
        }
        return value;
    }

/** @brief Frozen immediate-pre102 native body; never share candidate GPU math. */
    __global__ void pointer102Original_fp32_stage_batched_projection_pointers_kernel(
        const pointer102Original_Fp32BatchedProjectionPointers ptrs,
        const float **__restrict__ d_A_array,
        const float **__restrict__ d_B_array,
        float **__restrict__ d_C_array,
        int batch_count)
    {
        const int batch = static_cast<int>(threadIdx.x);
        if (batch >= batch_count)
            return;

        d_A_array[batch] = ptrs.A[batch];
        d_B_array[batch] = ptrs.B[batch];
        d_C_array[batch] = ptrs.C[batch];
    }

/** @brief Frozen immediate-pre102 native body; never share candidate GPU math. */
    __global__ __launch_bounds__(pointer102Original_SMALL_N_FP32_BLOCK)
        void pointer102Original_fp32_small_n_batched_projection_kernel(
            const float *const *__restrict__ d_A_array,
            const float *const *__restrict__ d_B_array,
            float *const *__restrict__ d_C_array,
            int M,
            int N,
            int K,
            llaminar2::FloatingProjectionRows row_range)
    {
        const int n = static_cast<int>(blockIdx.x);
        const int m = static_cast<int>(blockIdx.y);
        const int batch = static_cast<int>(blockIdx.z);
        if (!row_range.rowIsActive(m) || n >= N)
            return;

        const float *A = d_A_array[batch];
        const float *B = d_B_array[batch];
        float *C = d_C_array[batch];
        if (!A || !B || !C)
            return;

        float sum = 0.0f;
        for (int k = static_cast<int>(threadIdx.x); k < K; k += pointer102Original_SMALL_N_FP32_BLOCK)
        {
            sum += A[static_cast<size_t>(m) * K + k] *
                   B[static_cast<size_t>(n) * K + k];
        }

        __shared__ float scratch[pointer102Original_SMALL_N_FP32_BLOCK];
        const float reduced = pointer102Original_reduceFloating256(sum, scratch);

        if (threadIdx.x == 0)
            C[static_cast<size_t>(m) * N + n] = reduced;
    }

/** @brief Frozen immediate-pre102 native body; never share candidate GPU math. */
    template<pointer102Original_SmallProjectionWeightDtype WeightDtype>
    __global__ __launch_bounds__(pointer102Original_SMALL_N_FP32_BLOCK)
        void pointer102Original_fp32x16_batched_projection_kernel(
            const float *const *__restrict__ d_A_array,
            const float *const *__restrict__ d_B_array,
            float *const *__restrict__ d_C_array,
            int M,
            int N,
            int K,
            llaminar2::FloatingProjectionRows row_range)
    {
        static_assert(WeightDtype == pointer102Original_SMALL_PROJECTION_WEIGHT_FP16 ||
                      WeightDtype == pointer102Original_SMALL_PROJECTION_WEIGHT_BF16);
        const int n = static_cast<int>(blockIdx.x);
        const int m = static_cast<int>(blockIdx.y);
        const int batch = static_cast<int>(blockIdx.z);
        if (!row_range.rowIsActive(m) || n >= N)
            return;

        const float *A = d_A_array[batch];
        const uint16_t *B = reinterpret_cast<const uint16_t *>(d_B_array[batch]);
        float *C = d_C_array[batch];
        if (!A || !B || !C)
            return;

        float sum = 0.0f;
        for (int k = static_cast<int>(threadIdx.x); k < K; k += pointer102Original_SMALL_N_FP32_BLOCK)
        {
            const uint16_t weight_bits =
                B[static_cast<size_t>(n) * static_cast<size_t>(K) + static_cast<size_t>(k)];
            float weight;
            if constexpr (WeightDtype == pointer102Original_SMALL_PROJECTION_WEIGHT_BF16)
                weight = pointer102Original_bf16_bits_to_float(weight_bits);
            else
                weight = pointer102Original_fp16_bits_to_float(weight_bits);
            const size_t activation_index =
                static_cast<size_t>(m) * static_cast<size_t>(K) + static_cast<size_t>(k);
            // Preserve the original FP32 products, strided lane ownership
            // and fixed 256-lane reduction tree.
            const float activation = A[activation_index];
            sum += activation * weight;
        }

        __shared__ float scratch[pointer102Original_SMALL_N_FP32_BLOCK];
        const float reduced = pointer102Original_reduceFloating256(sum, scratch);

        if (threadIdx.x == 0)
            C[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)] = reduced;
    }

} // namespace qfn_pointer102_original
/** @copydoc qfnPointer102OriginalStaged */
bool qfnPointer102OriginalStaged(const float *const *a,const void *const *b,float *const *c,
 const float **da,const float **db,float **dc,int m,int n,int k,int count,int type,
 void *stream,const llaminar2::DeviceRowRange *prefix,const llaminar2::DeviceRequestRowRanges *requests)
{
 using namespace qfn_pointer102_original;
 if(!a||!b||!c||!da||!db||!dc||!stream||count<1||count>8||m<1||n<1||k<1||type<0||type>2)
  throw std::invalid_argument("Original pointer pipeline requires complete native bindings");
 auto native=static_cast<hipStream_t>(stream);
 if(native==hipStreamLegacy||native==hipStreamPerThread)throw std::invalid_argument("Original pointer pipeline rejects default streams");
 pointer102Original_Fp32BatchedProjectionPointers pointers{};
 for(int p=0;p<count;++p) {
  if(!a[p]||!b[p]||!c[p])throw std::invalid_argument("Original pointer pipeline null live matrix");
  pointers.A[p]=a[p];pointers.B[p]=static_cast<const float *>(b[p]);pointers.C[p]=c[p];
 }
 const auto rows=llaminar2::FloatingProjectionRows::bind(m,prefix,requests);
 hipLaunchKernelGGL(pointer102Original_fp32_stage_batched_projection_pointers_kernel,
  dim3(1),dim3(8),0,native,pointers,da,db,dc,count);
 if(hipGetLastError()!=hipSuccess)return false;
 const dim3 grid(n,m,count),block(256);
 if(type==0)hipLaunchKernelGGL(pointer102Original_fp32_small_n_batched_projection_kernel,grid,block,0,native,da,db,dc,m,n,k,rows);
 else if(type==1)hipLaunchKernelGGL((pointer102Original_fp32x16_batched_projection_kernel<pointer102Original_SMALL_PROJECTION_WEIGHT_FP16>),grid,block,0,native,da,db,dc,m,n,k,rows);
 else hipLaunchKernelGGL((pointer102Original_fp32x16_batched_projection_kernel<pointer102Original_SMALL_PROJECTION_WEIGHT_BF16>),grid,block,0,native,da,db,dc,m,n,k,rows);
 return hipGetLastError()==hipSuccess;
}
