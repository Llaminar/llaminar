/**
 * @file CUDAFlashAttentionShard1_HighPVW.cu
 * @brief CUDA flash-attention shard 1 high-PVW (HD=256 PVW=2/4) prefill with FP16 native KV.
 */
#define LLAMINAR_CUDA_FLASH_ATTENTION_SHARD 1
#define LLAMINAR_CUDA_FLASH_ATTENTION_SUBSHARD 1
#include "CUDAFlashAttentionShardImpl.cu.inc"
