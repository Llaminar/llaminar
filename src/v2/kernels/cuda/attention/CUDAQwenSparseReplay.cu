/**
 * @file CUDAQwenSparseReplay.cu
 * @brief Allocation-free CUDA publication of sparse transaction replay geometry.
 *
 * The captured root executes before all native appends and consumes only
 * canonical device pointers; it never observes counts on the host.
 */
#include <cuda_runtime.h>
#define QWEN_REPLAY_GPU_NAMESPACE cuda_replay_detail
#include "../../qwen4exp/QwenSparseReplayGPU.cuh"
#include <stdexcept>
namespace llaminar2::qwen4exp
{
void cudaQwenSparsePrepareReplay(const QwenSparsePrepareReplay &p,void *exactStream)
{
    validateQwenSparsePrepareReplay(p);
    auto stream=static_cast<cudaStream_t>(exactStream);
    if(!p.row_parameters || !stream || stream==cudaStreamLegacy || stream==cudaStreamPerThread || p.sequence.encoding!=QwenSparseRingCursor::NextWriteRow)
        throw std::invalid_argument("CUDA QSA replay requires exact non-default stream and canonical next-write cursor");
    cuda_replay_detail::prepareReplay<<<(p.row_capacity+255)/256,256,0,stream>>>(p);
    if(cudaGetLastError()!=cudaSuccess)throw std::runtime_error("CUDA QSA replay publication launch failed");
}
}
