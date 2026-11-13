/**
 * @file Qwen4ExpBranchConcat.cpp
 * @brief Exact CPU Dense0 branch concatenation and setup-time native dispatch.
 *
 * Each task copies contiguous half-rows so vectorized memcpy retains every FP32
 * bit, including NaN payloads and signed zero. The four-branch row order never
 * depends on worker count and the original token authority is read once.
 */
#include "Qwen4ExpBranchConcat.h"
#include <cstring>
#include <omp.h>
#include <limits>
#include <stdexcept>
namespace llaminar2::qwen4exp {
namespace {
/** @brief Prove an aligned nonempty native interval without dereferencing it. */
std::uintptr_t span(const void *p,std::size_t bytes) {
    auto begin=reinterpret_cast<std::uintptr_t>(p);
    if(!p||begin%4||bytes>std::numeric_limits<std::uintptr_t>::max()-begin)
        throw std::invalid_argument("BranchConcat requires aligned, bounded native intervals");
    return begin;
}
}
void validateBranchConcat(const BranchConcatArgs &a) {
    const auto id=a.rows.captureIdentity();
    if(a.hidden<=0||id.repeats_per_row!=1||a.rows.physicalRows()>std::numeric_limits<int>::max()/4||
       a.hidden>std::numeric_limits<int>::max()/8||
       std::size_t(a.rows.physicalRows())>std::numeric_limits<std::size_t>::max()/32/std::size_t(a.hidden))
        throw std::invalid_argument("BranchConcat requires original token rows and bounded four-branch geometry");
    const auto bytes=std::size_t(a.rows.physicalRows())*std::size_t(a.hidden)*4;
    auto h=span(a.hiddenBranches,4*bytes),e=span(a.embedding,bytes),o=span(a.output,8*bytes);
    if(!(o+8*bytes<=h||h+4*bytes<=o)||!(o+8*bytes<=e||e+bytes<=o))
        throw std::invalid_argument("BranchConcat output must be distinct from both source intervals");
    if(id.count){auto c=span(id.count,4);if(!(o+8*bytes<=c||c+4<=o))throw std::invalid_argument("BranchConcat count must not alias output");}
}
void cpuBranchConcat(const BranchConcatArgs &a,void *stream) {
    if(stream)throw std::invalid_argument("CPU BranchConcat rejects GPU streams");
    validateBranchConcat(a);
    const int live=a.rows.activeRowsFor(a.rows.countOwner()?*a.rows.countOwner():a.rows.countCapacity());
    if(live<0)throw std::invalid_argument("BranchConcat live token count is corrupt");
    if(!live)return;
    const std::size_t h=a.hidden;
    const auto copy=[&](int branchRow){
        const auto row=std::size_t(branchRow/4),branch=std::size_t(branchRow%4);
        float *target=a.output+std::size_t(branchRow)*2*h;
        std::memcpy(target,a.embedding+row*h,h*4);
        std::memcpy(target+h,a.hiddenBranches+(row*4+branch)*h,h*4);
    };
    // GNU OpenMP allocates a serialized team even for a one-worker parallel
    // region. One worker executes the identical copy loop directly instead.
    if(omp_get_max_threads()==1){for(int row=0;row<live*4;++row)copy(row);return;}
    #pragma omp parallel for schedule(static)
    for(int row=0;row<live*4;++row)copy(row);
}
BranchConcatOperation createBranchConcatOperation(DeviceId device) {
    if(device.is_cpu())return cpuBranchConcat;
#ifdef HAVE_CUDA
    if(device.is_cuda())return cudaBranchConcat;
#endif
#ifdef HAVE_ROCM
    if(device.is_rocm())return rocmBranchConcat;
#endif
    throw std::invalid_argument("BranchConcat backend is not built");
}
}
