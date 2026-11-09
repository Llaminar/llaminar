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
#include <sstream>
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
/** @brief Describe every immutable transfer address/extent without reading device state. */
void describe(std::ostringstream &out,const QwenStateTransfer &p)
{
    out<<" source="<<p.source<<" destination="<<p.destination
       <<" row_bytes="<<p.rowBytes<<" capacity_rows="<<p.capacityRows
       <<" row_source="<<static_cast<unsigned>(p.rowSource)<<" count="<<p.count
       <<" fixed_rows="<<p.fixedRows<<" published_rows="<<p.publishedRows
       <<" copied_bytes="<<p.copiedBytes<<" aggregate_copied_bytes="<<p.aggregateCopiedBytes.words;
}
/** @brief Describe the exact reset destination, word count and native fill bits. */
void describe(std::ostringstream &out,const QwenStateFill &p)
{
    out<<" destination="<<p.destination<<" words="<<p.words<<" bits="<<p.bits;
}
/**
 * @brief Fail with retained native status and immutable launch geometry.
 * @param status Original non-success error, saved before diagnostic queries.
 * @param operation Transfer or reset operation that owns the observation.
 * @param before True only when the nonclearing pre-launch peek observed it.
 * @param stream Exact submitted stream; never inferred from ambient state.
 * @param words Immutable submitted/attempted extent in native 32-bit words.
 * @param params Original operation addresses/metadata; never dereferenced here.
 *
 * Device identity is queried only on this fatal path. An error first observed
 * after launch may originate from earlier asynchronous work; this diagnostic
 * records temporal observation, not an invented causal attribution.
 */
template<class Params>[[noreturn]] void failNative(hipError_t status,const char *operation,
    bool before,void *stream,std::size_t words,const Params &params)
{
    int device=-1;
    const auto deviceStatus=hipGetDevice(&device);
    std::ostringstream out;
    out<<"ROCm QFN auxiliary "<<operation<<" launch failed: origin="
       <<(before?"observed_before_launch":"first_observed_after_launch")
       <<" launch_submitted="<<(before?"false":"true")
       <<" status="<<static_cast<int>(status)<<" ("<<hipGetErrorString(status)<<")"
       <<" current_device="<<device<<" device_query_status="<<static_cast<int>(deviceStatus)
       <<" stream="<<stream<<" grid_x="<<(words+255)/256<<" block_x=256";
    describe(out,params);
    throw std::runtime_error(out.str());
}
}
void rocmQwenStateTransfer(const QwenStateTransfer &p,void *stream)
{
    validateQwenStateTransfer(p);
    const auto native=exact(stream);
    const auto words=std::size_t(p.capacityRows)*(p.rowBytes/4);
    const auto prior=hipPeekAtLastError();
    if(prior!=hipSuccess)failNative(prior,"transfer",true,stream,words,p);
    hipLaunchKernelGGL(rocm_state_detail::transfer,dim3((words+255)/256),dim3(256),0,native,p);
    const auto status=hipGetLastError();
    if(status!=hipSuccess)failNative(status,"transfer",false,stream,words,p);
}
void rocmQwenStateFill(const QwenStateFill &p,void *stream)
{
    validateQwenStateFill(p);
    const auto native=exact(stream);
    const auto prior=hipPeekAtLastError();
    if(prior!=hipSuccess)failNative(prior,"reset",true,stream,p.words,p);
    hipLaunchKernelGGL(rocm_state_detail::fill,dim3((p.words+255)/256),dim3(256),0,native,p);
    const auto status=hipGetLastError();
    if(status!=hipSuccess)failNative(status,"reset",false,stream,p.words,p);
}
}
