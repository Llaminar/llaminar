/**
 * @file CanonicalRankBankOriginalROCm.hip
 * @brief Test-only frozen pre-repair shared rank-bank publication oracle.
 *
 * The original production entry body is retained verbatim apart from its
 * symbol name. In particular its lvalue conditional/zero temporary remains;
 * candidate code and runtime selection never call this diagnostic oracle.
 * Original production source: src/v2/kernels/rocm/moe/ROCmMoEKernels.hip
 * Frozen entry-body SHA256: 5fc1d689ebd315bed026742321bb97a1a4993756b9493fb5a1403acc61e1f82e.
 */
#include <hip/hip_runtime.h>
#include <algorithm>
namespace
{
/** @brief Frozen original clamp, including null/full and invalid-count boundaries. */
__device__ __forceinline__ int rocm_moe_clamp_effective_seq_len(int seq_len, const int *device_effective_seq_len)
{
    if (!device_effective_seq_len)
        return seq_len;
    const int raw = *device_effective_seq_len;
    return raw<0?0:(raw>seq_len?seq_len:raw);
}
__global__ __launch_bounds__(256)
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
    const int bank_row = blockIdx.y;
    const int participant = bank_row / seq_len;
    const int token = bank_row - participant * seq_len;
    if (participant >= participant_count || token >= seq_len)
        return;

    const int effective_seq_len =
        rocm_moe_clamp_effective_seq_len(
            seq_len, device_effective_seq_len);
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
        const int d4 = d_model >> 2;
        const float4 *__restrict__ source4 =
            reinterpret_cast<const float4 *>(shared_output + source_offset);
        float4 *__restrict__ destination4 =
            reinterpret_cast<float4 *>(
                canonical_publication + destination_offset);
        const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int index = blockIdx.x * blockDim.x + threadIdx.x;
             index < d4;
             index += gridDim.x * blockDim.x)
        {
            destination4[index] = copy_local ? source4[index] : zero;
        }
        return;
    }

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
extern "C" bool testOriginalROCmSharedRankBank(const float *source, float *destination,
    int rows, int topK, int hidden, int participant, int participants, const int *live, int ordinal, void *stream)
{
    if (!source || !destination || !stream || rows <= 0 || topK <= 0 || hidden <= 0 ||
        participants <= 0 || participant < 0 || participant >= participants)
        return false;
    if (hipSetDevice(ordinal) != hipSuccess)
        return false;
    (void)hipGetLastError();
    const int work = (hidden&3) == 0?hidden/4:hidden;
    const dim3 grid(std::max(1, (work+255)/256), rows*participants);
    originalSharedRankBankKernel<<<grid, 256, 0, static_cast<hipStream_t>(stream)>>>(
        source, destination, rows, topK, hidden, participant, participants, live);
    return hipGetLastError() == hipSuccess;
}
