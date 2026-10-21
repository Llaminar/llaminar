/**
 * @file ROCmQwen4ExpStateIO.hip
 * @brief Exact-stream ROCm live-tail checkpoint and native reset submission.
 *
 * All pointers and capacity belong to admitted persistent state; no native
 * allocation, transfer, synchronization or count readback occurs here.
 */
#include <hip/hip_runtime.h>
#define QWEN_STATE_GPU_NAMESPACE rocm_state_detail
#include "../../qwen4exp/Qwen4ExpStateIOGPU.cuh"
#include <stdexcept>
namespace llaminar2::qwen4exp
{
namespace {
/** @brief Reject every default-stream spelling before submitting state work. */
hipStream_t exact(void *value)
{
    auto stream=static_cast<hipStream_t>(value);
    if(!stream || stream==hipStreamLegacy || stream==hipStreamPerThread)
        throw std::invalid_argument("ROCm QFN auxiliary state requires an exact non-default stream");
    return stream;
}
}
void rocmQwenStateTransfer(const QwenStateTransfer &p,void *stream)
{
    validateQwenStateTransfer(p);
    const auto native=exact(stream);
    const auto words=std::size_t(p.capacityRows)*(p.rowBytes/4);
    hipLaunchKernelGGL(rocm_state_detail::transfer,dim3((words+255)/256),dim3(256),0,native,p);
    if(hipGetLastError()!=hipSuccess)throw std::runtime_error("ROCm QFN auxiliary transfer launch failed");
}
void rocmQwenStateFill(const QwenStateFill &p,void *stream)
{
    validateQwenStateFill(p);
    const auto native=exact(stream);
    hipLaunchKernelGGL(rocm_state_detail::fill,dim3((p.words+255)/256),dim3(256),0,native,p);
    if(hipGetLastError()!=hipSuccess)throw std::runtime_error("ROCm QFN auxiliary reset launch failed");
}
}
