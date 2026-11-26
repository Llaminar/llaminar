/**
 * @file CUDAFlashAttentionShard0_HighPVW.cu
 * @brief CUDA flash-attention shard 0 high-PVW (HD=256 PVW=2/4) prefill with FP32 native KV.
 */
#define LLAMINAR_CUDA_FLASH_ATTENTION_SHARD 0
#define LLAMINAR_CUDA_FLASH_ATTENTION_SUBSHARD 1
#include "CUDAFlashAttentionShardImpl.cu.inc"
