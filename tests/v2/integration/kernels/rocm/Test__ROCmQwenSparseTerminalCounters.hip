/**
 * @file Test__ROCmQwenSparseTerminalCounters.hip
 * @brief ROCm exact native atomic and complete-terminal-frontier counter regressions.
 */
#include "backends/rocm/HIPGraphCapture.h"
#include "kernels/qwen4exp/QwenSparseAttention.h"
#define QSC_DEPENDENTS(n,d,c) hipGraphNodeGetDependentNodes(n,d,c)
#define QSC_REMOVE(g,f,t,c) hipGraphRemoveDependencies(g,f,t,c)
#define QSC_RT(name) hip##name
#define QSC_STREAM hipStream_t
#define QSC_GRAPH hipGraph_t
#define QSC_NODE hipGraphNode_t
#define QSC_NODE_TYPE hipGraphNodeType
#define QSC_KERNEL_PARAMS hipKernelNodeParams
#define QSC_CAPTURE llaminar2::HIPGraphCapture
#define QSC_FACTORY llaminar2::qwen4exp::createROCmQwenSparseAttentionKernels
#define QSC_SUITE ROCmQwenSparseTerminalCounters
#include "../qwen4exp/QwenSparseTerminalCounterDeviceTest.h"

#include "../qwen4exp/QwenSparseFollowerCounterDeviceTest.h"
