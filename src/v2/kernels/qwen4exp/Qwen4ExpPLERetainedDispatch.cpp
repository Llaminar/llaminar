/**
 * @file Qwen4ExpPLERetainedDispatch.cpp
 * @brief Single setup selection authority for CPU/CUDA/ROCm retained PLE operations.
 */
#include "Qwen4ExpPLERetainedDispatch.h"
#include <stdexcept>
namespace llaminar2::qwen4exp {
namespace {
/** @brief Reject accidental GPU stream use by a CPU operation. */
void cpuStream(void *s){if(s)throw std::invalid_argument("CPU PLE operation received a GPU stream");}
/** @brief Submit one host-owned immutable prestate capture. */
void snapshot(const PLESnapshotPrior &p,void*s){cpuStream(s);cpuPLESnapshotPrior(p);}
/** @brief Submit one host-owned exact-live candidate retention. */
void retain(const PLERetainCandidates &p,void*s){cpuStream(s);cpuPLERetainCandidates(p);}
/** @brief Submit one host-owned validated accepted-tail splice. */
void publish(const PLEPublishAccepted &p,void*s){cpuStream(s);cpuPLEPublishAccepted(p);}
}
PLETransactionOperations createPLETransactionOperations(DeviceId device){
 if(device.is_cpu())return {snapshot,retain,publish};
#ifdef HAVE_CUDA
 if(device.is_cuda())return {cudaPLESnapshotPrior,cudaPLERetainCandidates,cudaPLEPublishAccepted};
#endif
#ifdef HAVE_ROCM
 if(device.is_rocm())return {rocmPLESnapshotPrior,rocmPLERetainCandidates,rocmPLEPublishAccepted};
#endif
 throw std::invalid_argument("PLE retained transaction backend is not built");
}
}
