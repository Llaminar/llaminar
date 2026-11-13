/**
 * @file ROCmQwenSparseReplay.hip
 * @brief Allocation-free HIP publication of sparse transaction replay geometry.
 *
 * The captured root executes before all native appends and consumes only
 * canonical device pointers; it never observes counts on the host.
 */
#include <hip/hip_runtime.h>
#define QWEN_REPLAY_GPU_NAMESPACE rocm_replay_detail
#include "../../qwen4exp/QwenSparseReplayGPU.cuh"
#include <stdexcept>
namespace llaminar2::qwen4exp
{
void rocmQwenSparsePrepareReplay(const QwenSparsePrepareReplay &p,void *exactStream)
{
    validateQwenSparsePrepareReplay(p);
    auto stream=static_cast<hipStream_t>(exactStream);
    if(!p.row_parameters || !stream || stream==hipStreamLegacy || stream==hipStreamPerThread || p.sequence.encoding!=QwenSparseRingCursor::NextWriteRow)
        throw std::invalid_argument("ROCm QSA replay requires exact non-default stream and canonical next-write cursor");
    hipLaunchKernelGGL(rocm_replay_detail::prepareReplay,dim3((p.row_capacity+255)/256),dim3(256),0,stream,p);
    if(hipGetLastError()!=hipSuccess)throw std::runtime_error("ROCm QSA replay publication launch failed");
}
}
