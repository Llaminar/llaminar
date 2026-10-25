/**
 * @file Test__CUDAQwenSparseAttention.cpp
 * @brief CUDA retained-graph QSA correctness, sparse-read accounting, and profiling.
 *
 * The common conformance fixture exercises production launch entrypoints; the
 * explicit backend mappings preserve native stream and graph types.
 */
#include "kernels/cuda/attention/CUDAQwenSparseAttention.h"
extern "C" bool cudaOps_rope_populate_inv_freq(float *,int,float,int,cudaStream_t);
extern "C" bool cudaOps_rope_fp32_decode_v3(float *,float *,const float *,int,int,int,int,int,int,cudaStream_t);
#define QSA_APPLY_ROPE cudaOps_rope_fp32_decode_v3
#define QSA_POPULATE_ROPE cudaOps_rope_populate_inv_freq
#define QSA_CUDA_RUNTIME 1
#define QSA_RT(name) cuda##name
#define QSA_STREAM cudaStream_t
#define QSA_GRAPH cudaGraph_t
#define QSA_EXEC cudaGraphExec_t
#define QSA_EVENT cudaEvent_t
#define QSA_INDEXER llaminar2::qwen4exp::cudaQwenSparseIndexer
#define QSA_ATTENTION llaminar2::qwen4exp::cudaQwenSparseAttention
#define QSA_COMMIT llaminar2::qwen4exp::cudaQwenSparseCommitPartial
#define QSA_PREPARE llaminar2::qwen4exp::cudaQwenSparsePrepareReplay
#define QSA_TRANSFER llaminar2::qwen4exp::cudaQwenStateTransfer
#define QSA_SUITE CUDAQwenSparseAttention
#include "../qwen4exp/QwenSparseDeviceTest.h"

#include "../qwen4exp/QwenSparseObservationCUDAAdapter.h"
#include "../qwen4exp/QwenSparseObservationPerfTest.h"
