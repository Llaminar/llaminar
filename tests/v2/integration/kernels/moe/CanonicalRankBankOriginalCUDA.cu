/**
 * @file CanonicalRankBankOriginalCUDA.cu
 * @brief Test-only frozen pre-repair shared rank-bank publication oracle.
 *
 * The original production entry body is retained verbatim apart from its
 * symbol name. In particular its lvalue conditional/zero temporary remains;
 * candidate code and runtime selection never call this diagnostic oracle.
 * Original production source: src/v2/kernels/cuda/moe/CUDAMoEKernels.cu
 * Frozen entry-body SHA256: 5841064ccb5ce64216b46e4dfc0186168b543c9b8f0f4183b9a8bae954db6ece.
 */
#include <cuda_runtime.h>
#include <algorithm>
namespace
{
/** @brief Frozen original clamp, including null/full and invalid-count boundaries. */
__device__ __forceinline__ int clamp_effective_seq_len(int seq_len, const int *device_effective_seq_len)
{
    if (!device_effective_seq_len)
        return seq_len;
    const int raw = *device_effective_seq_len;
    return raw<0?0:(raw>seq_len?seq_len:raw);
}
/** @brief Frozen original CUDA row reconstruction across gridY/Z. */
__device__ __forceinline__ int logical_row_from_grid_yz()
{return static_cast<int>(blockIdx.y+blockIdx.z*static_cast<unsigned int>(gridDim.y));
}
__global__
void originalSharedRankBankKernel(
        const float *__restrict__ shared_output,
        float *__restrict__ canonical_publication,
        int seq_len,
        int top_k,
        int d_model,
        int participant_index,
        int participant_count,
        const int *__restrict__ device_effective_seq_len)
    {
        const int bank_row = logical_row_from_grid_yz();
        const int participant = bank_row / seq_len;
        const int token = bank_row - participant * seq_len;
        if (participant >= participant_count || token >= seq_len)
            return;

        const int effective_seq_len =
            clamp_effective_seq_len(seq_len, device_effective_seq_len);
        const bool copy_local =
            participant == participant_index && token < effective_seq_len;
        const size_t row_elements =
            static_cast<size_t>(seq_len) * static_cast<size_t>(d_model);
        const size_t route_elements =
            row_elements * static_cast<size_t>(top_k);
        const size_t source_offset =
            static_cast<size_t>(token) * static_cast<size_t>(d_model);
        const size_t destination_offset =
            route_elements +
            static_cast<size_t>(participant) * row_elements + source_offset;

        if ((d_model & 3) == 0)
        {
            const int n4 = d_model >> 2;
            const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const float4 *source4 =
                reinterpret_cast<const float4 *>(shared_output + source_offset);
            float4 *destination4 =
                reinterpret_cast<float4 *>(
                    canonical_publication + destination_offset);
            for (int index = blockIdx.x * blockDim.x + threadIdx.x;
                 index < n4;
                 index += gridDim.x * blockDim.x)
            {
                destination4[index] = copy_local ? source4[index] : zero;
            }
            return;
        }

        /*
         * A ragged row gives every token after row zero a potentially
         * non-16-byte-aligned base. Reinterpreting that base as float4 is not
         * a legal vector access, so width totality requires a scalar row path.
         */
        for (int column = blockIdx.x * blockDim.x + threadIdx.x;
             column < d_model;
             column += gridDim.x * blockDim.x)
        {
            canonical_publication[destination_offset + column] =
                copy_local ? shared_output[source_offset + column] : 0.0f;
        }
    }
}
/** @brief Launch exactly the frozen native entry/geometry on a retained test stream.
 * @param source Original shared FP32 input rows.
 * @param destination Complete canonical publication allocation.
 * @param rows Positive physical row count.
 * @param topK Number of routed prefix slots.
 * @param hidden Native FP32 row width.
 * @param participant Actual local participant bank.
 * @param participants Complete participant-bank count.
 * @param live Original device-owned live prefix, optionally null/full.
 * @param ordinal Exact native device ordinal.
 * @param stream Explicit non-default original test stream.
 * @return Whether native submission succeeded; never synchronizes or allocates.
 */
extern "C" bool testOriginalCUDASharedRankBank(const float *source, float *destination,
    int rows, int topK, int hidden, int participant, int participants, const int *live, int ordinal, void *stream)
{
    if (!source || !destination || !stream || rows <= 0 || topK <= 0 || hidden <= 0 ||
        participants <= 0 || participant < 0 || participant >= participants)
        return false;
    if (cudaSetDevice(ordinal) != cudaSuccess)
        return false;
    const unsigned columns = std::max(1, (((hidden+3)/4)+255)/256);
    const unsigned rowCount = static_cast<unsigned>(rows*participants), rowY = std::min(rowCount, 65535u);
    const dim3 grid(columns, rowY, (rowCount+rowY-1)/rowY);
    originalSharedRankBankKernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        source, destination, rows, topK, hidden, participant, participants, live);
    return cudaGetLastError() == cudaSuccess;
}
