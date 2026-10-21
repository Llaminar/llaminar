/**
 * @file Test__CUDAQwenSparseBoundedSelection.cpp
 * @brief CUDA actual captured bounded-selector correctness and paired timing.
 */
#include "kernels/cuda/attention/CUDAQwenSparseAttention.h"
#define QSB_RT(name) cuda##name
#define QSB_STREAM cudaStream_t
#define QSB_GRAPH cudaGraph_t
#define QSB_EXEC cudaGraphExec_t
#define QSB_EVENT cudaEvent_t
#define QSB_NODE cudaGraphNode_t
#define QSB_NODETYPE cudaGraphNodeType
#define QSB_PARAMS cudaKernelNodeParams
#define QSB_ATTRIBUTES cudaFuncAttributes
#define QSB_SELECT llaminar2::qwen4exp::cudaQwenSparseSelectPools
#define QSB_SUITE CUDAQwenSparseBoundedSelection
#include "../qwen4exp/QwenSparseBoundedDeviceTest.h"
