/**
 * @file CUDAFlashAttentionShard5_PVW1.cu
 * @brief CUDA flash-attention shard 5 HD=256 PVW=1 context-transaction prefill with BF16 native KV.
 */
#define LLAMINAR_CUDA_FLASH_ATTENTION_SHARD 5
#define LLAMINAR_CUDA_FLASH_ATTENTION_SUBSHARD 1
#include "CUDAFlashAttentionShardImpl.cu.inc"
