/**
 * @file CUDAFlashAttentionShard4_PVW2.cu
 * @brief CUDA flash-attention shard 4 HD=256 PVW=2 context-transaction prefill with FP16 native KV.
 */
#define LLAMINAR_CUDA_FLASH_ATTENTION_SHARD 4
#define LLAMINAR_CUDA_FLASH_ATTENTION_SUBSHARD 2
#include "CUDAFlashAttentionShardImpl.cu.inc"
