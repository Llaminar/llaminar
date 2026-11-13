/**
 * @file Test__CUDAQwenSparseTerminalCounters.cu
 * @brief CUDA exact native atomic and complete-terminal-frontier counter regressions.
 */
#include "backends/cuda/CUDAGraphCapture.h"
#include "kernels/qwen4exp/QwenSparseAttention.h"
#define QSC_DEPENDENTS(n,d,c) cudaGraphNodeGetDependentNodes(n,d,nullptr,c)
#define QSC_REMOVE(g,f,t,c) cudaGraphRemoveDependencies(g,f,t,nullptr,c)
#define QSC_RT(name) cuda##name
#define QSC_STREAM cudaStream_t
#define QSC_GRAPH cudaGraph_t
#define QSC_NODE cudaGraphNode_t
#define QSC_NODE_TYPE cudaGraphNodeType
#define QSC_KERNEL_PARAMS cudaKernelNodeParams
#define QSC_CAPTURE llaminar2::CUDAGraphCapture
#define QSC_FACTORY llaminar2::qwen4exp::createCUDAQwenSparseAttentionKernels
#define QSC_SUITE CUDAQwenSparseTerminalCounters
#include "../qwen4exp/QwenSparseTerminalCounterDeviceTest.h"

#include "../qwen4exp/QwenSparseFollowerCounterDeviceTest.h"
