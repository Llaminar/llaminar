/**
 * @file CUDAFlashAttentionShard3_PVW1.cu
 * @brief CUDA flash-attention shard 3 HD=256 PVW=1 context-transaction prefill with FP32 native KV.
 */
#define LLAMINAR_CUDA_FLASH_ATTENTION_SHARD 3
#define LLAMINAR_CUDA_FLASH_ATTENTION_SUBSHARD 1
#include "CUDAFlashAttentionShardImpl.cu.inc"
