/**
 * @file Test__ROCmQwenSparseBoundedSelection.cpp
 * @brief ROCm actual captured bounded-selector correctness and paired timing.
 */
#include "kernels/rocm/attention/ROCmQwenSparseAttention.h"
#define QSB_RT(name) hip##name
#define QSB_STREAM hipStream_t
#define QSB_GRAPH hipGraph_t
#define QSB_EXEC hipGraphExec_t
#define QSB_EVENT hipEvent_t
#define QSB_NODE hipGraphNode_t
#define QSB_NODETYPE hipGraphNodeType
#define QSB_PARAMS hipKernelNodeParams
#define QSB_ATTRIBUTES hipFuncAttributes
#define QSB_SELECT llaminar2::qwen4exp::rocmQwenSparseSelectPools
#define QSB_SUITE ROCmQwenSparseBoundedSelection
#include "../qwen4exp/QwenSparseBoundedDeviceTest.h"
