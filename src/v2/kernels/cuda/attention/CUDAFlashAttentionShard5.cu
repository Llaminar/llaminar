/**
 * @file CUDAFlashAttentionShard5.cu
 * @brief CUDA flash-attention shard 5: context-transaction prefill with BF16 native KV.
 */
#define LLAMINAR_CUDA_FLASH_ATTENTION_SHARD 5
#include "CUDAFlashAttentionShardImpl.cu.inc"
