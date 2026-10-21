/**
 * @file CUDAQwen4ExpStateIO.cu
 * @brief Exact-stream CUDA live-tail checkpoint and native reset submission.
 *
 * All pointers and capacity belong to admitted persistent state; no native
 * allocation, transfer, synchronization or count readback occurs here.
 */
#include <cuda_runtime.h>
#define QWEN_STATE_GPU_NAMESPACE cuda_state_detail
#include "../../qwen4exp/Qwen4ExpStateIOGPU.cuh"
#include <stdexcept>
namespace llaminar2::qwen4exp
{
namespace {
/** @brief Reject every default-stream spelling before submitting state work. */
cudaStream_t exact(void *value)
{
    auto stream=static_cast<cudaStream_t>(value);
    if(!stream || stream==cudaStreamLegacy || stream==cudaStreamPerThread)
        throw std::invalid_argument("CUDA QFN auxiliary state requires an exact non-default stream");
    return stream;
}
}
void cudaQwenStateTransfer(const QwenStateTransfer &p,void *stream)
{
    validateQwenStateTransfer(p);
    const auto native=exact(stream);
    const auto words=std::size_t(p.capacityRows)*(p.rowBytes/4);
    cuda_state_detail::transfer<<<(words+255)/256,256,0,native>>>(p);
    if(cudaGetLastError()!=cudaSuccess)throw std::runtime_error("CUDA QFN auxiliary transfer launch failed");
}
void cudaQwenStateFill(const QwenStateFill &p,void *stream)
{
    validateQwenStateFill(p);
    const auto native=exact(stream);
    cuda_state_detail::fill<<<(p.words+255)/256,256,0,native>>>(p);
    if(cudaGetLastError()!=cudaSuccess)throw std::runtime_error("CUDA QFN auxiliary reset launch failed");
}
}
