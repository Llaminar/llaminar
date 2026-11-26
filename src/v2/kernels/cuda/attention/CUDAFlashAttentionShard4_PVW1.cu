/**
 * @file CUDAFlashAttentionShard4_PVW1.cu
 * @brief CUDA flash-attention shard 4 HD=256 PVW=1 context-transaction prefill with FP16 native KV.
 */
#define LLAMINAR_CUDA_FLASH_ATTENTION_SHARD 4
#define LLAMINAR_CUDA_FLASH_ATTENTION_SUBSHARD 1
#include "CUDAFlashAttentionShardImpl.cu.inc"
