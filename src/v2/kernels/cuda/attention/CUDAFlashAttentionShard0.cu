/**
 * @file CUDAFlashAttentionShard0.cu
 * @brief CUDA flash-attention shard 0: ordinary pipelined prefill with FP32 native KV.
 */
#define LLAMINAR_CUDA_FLASH_ATTENTION_SHARD 0
#include "CUDAFlashAttentionShardImpl.cu.inc"
