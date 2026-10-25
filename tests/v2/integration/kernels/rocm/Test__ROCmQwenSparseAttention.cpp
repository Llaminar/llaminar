/**
 * @file Test__ROCmQwenSparseAttention.cpp
 * @brief HIP retained-graph QSA correctness, sparse-read accounting, and profiling.
 *
 * The common conformance fixture exercises production launch entrypoints; the
 * explicit backend mappings preserve native stream and graph types.
 */
#include "kernels/rocm/attention/ROCmQwenSparseAttention.h"
extern "C" bool hipOps_rope_populate_inv_freq(float *,int,float,int,void *);
extern "C" bool hipOps_rope_fp32_decode(float *,float *,const float *,int,int,int,int,int,int,void *);
#define QSA_APPLY_ROPE hipOps_rope_fp32_decode
#define QSA_POPULATE_ROPE hipOps_rope_populate_inv_freq
#define QSA_RT(name) hip##name
#define QSA_STREAM hipStream_t
#define QSA_GRAPH hipGraph_t
#define QSA_EXEC hipGraphExec_t
#define QSA_EVENT hipEvent_t
#define QSA_INDEXER llaminar2::qwen4exp::rocmQwenSparseIndexer
#define QSA_ATTENTION llaminar2::qwen4exp::rocmQwenSparseAttention
#define QSA_COMMIT llaminar2::qwen4exp::rocmQwenSparseCommitPartial
#define QSA_PREPARE llaminar2::qwen4exp::rocmQwenSparsePrepareReplay
#define QSA_TRANSFER llaminar2::qwen4exp::rocmQwenStateTransfer
#define QSA_SUITE ROCmQwenSparseAttention
#include "../qwen4exp/QwenSparseDeviceTest.h"

#include "../qwen4exp/QwenSparseObservationROCmAdapter.h"
#include "../qwen4exp/QwenSparseObservationPerfTest.h"

#include "../qwen4exp/QwenSparseIndexerPolicyPerfROCmTest.h"

#include "../qwen4exp/QwenSparseFP16ROCmTest.h"

#include "../qwen4exp/QwenSparseFP16Parts33ROCmTest.h"

#ifdef HAVE_ROCM
#include "../qwen4exp/QwenSparseFP16OnlineRescaleROCmTest.h"
#endif
