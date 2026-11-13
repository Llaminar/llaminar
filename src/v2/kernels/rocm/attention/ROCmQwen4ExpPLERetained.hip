/**
 * @file ROCmQwen4ExpPLERetained.hip
 * @brief Exact-stream native retained PLE transaction submission without hot allocation.
 *
 * Setup validates every pointer extent. Publication first launches a complete
 * native decision-vector validator, then one parallel fixed-tail splice. There
 * is no host metadata mirror, transfer, synchronization or fallback execution.
 */
#include <hip/hip_runtime.h>
#define PLE_RETAINED_GPU_NAMESPACE rocm_ple_retained
#include "../../qwen4exp/Qwen4ExpPLERetainedGPU.cuh"
#include <algorithm>
#include <stdexcept>
namespace llaminar2::qwen4exp {
namespace {
/** @brief Reject every default-stream spelling. */
hipStream_t exact(void *p){auto s=static_cast<hipStream_t>(p);if(!s||s==hipStreamLegacy||s==hipStreamPerThread)throw std::invalid_argument("PLE retention requires exact native stream");return s;}
/** @brief Bound physical CTAs independently of the admitted row capacity. */
unsigned blocks(std::size_t words){return static_cast<unsigned>(std::min<std::size_t>(256,(words+255)/256));}
/** @brief Preserve native submission failures as fatal errors. */
void check(){if(hipGetLastError()!=hipSuccess)throw std::runtime_error("PLE retained native launch failed");}
}
void rocmPLESnapshotPrior(const PLESnapshotPrior &p,void *stream){validatePLESnapshot(p);auto s=exact(stream);rocm_ple_retained::snapshot<<<blocks(2+std::size_t(p.channels)*9),256,0,s>>>(p);check();}
void rocmPLERetainCandidates(const PLERetainCandidates &p,void *stream){validatePLERetention(p);auto s=exact(stream);rocm_ple_retained::retain<<<blocks(std::size_t(p.transactionRows.rowsPerRequest())*(std::size_t(p.channels)+1)),256,0,s>>>(p);check();}
void rocmPLEPublishAccepted(const PLEPublishAccepted &p,void *stream){validatePLEPublication(p);auto s=exact(stream);rocm_ple_retained::validate<<<blocks(p.acceptance.candidates().requests()),256,0,s>>>(p.acceptance);check();rocm_ple_retained::publish<<<blocks(2+std::size_t(p.channels)*9),256,0,s>>>(p);check();}
}
