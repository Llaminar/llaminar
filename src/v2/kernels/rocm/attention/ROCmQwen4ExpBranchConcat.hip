/**
 * @file ROCmQwen4ExpBranchConcat.hip
 * @brief ROCm exact-stream Dense0 branch-concatenation submission.
 *
 * Setup validation never reads device counts; the retained kernel gates live
 * words from the original count owner and leaves all inactive rows untouched.
 */
#include <hip/hip_runtime.h>
#include "../../qwen4exp/Qwen4ExpBranchConcatGPU.cuh"
#include <stdexcept>
namespace llaminar2::qwen4exp {
void rocmBranchConcat(const BranchConcatArgs &a,void *stream) {
    if(!stream||stream==reinterpret_cast<void*>(hipStreamLegacy)||stream==reinterpret_cast<void*>(hipStreamPerThread))
        throw std::invalid_argument("ROCm BranchConcat requires an exact non-default stream");
    validateBranchConcat(a);
    // Row-striding retains exact geometry beyond the native grid.y limit.
    const dim3 grid(unsigned((std::size_t(a.hidden)+255)/256),
        unsigned(a.rows.physicalRows()<65535?a.rows.physicalRows():65535),8);
    const auto native=static_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(branchConcatKernel,grid,dim3(256),0,native,a);
    if(hipGetLastError()!=hipSuccess)throw std::runtime_error("ROCm BranchConcat launch failed");
}
}
