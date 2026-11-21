/**
 * @file CUDAFlashAttentionShard3.cu
 * @brief CUDA flash-attention shard 3: context-transaction prefill with FP32 native KV.
 */
#define LLAMINAR_CUDA_FLASH_ATTENTION_SHARD 3
#include "CUDAFlashAttentionShardImpl.cu.inc"
