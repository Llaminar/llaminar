/**
 * @file CUDAFlashAttentionShard2_HighPVW.cu
 * @brief CUDA flash-attention shard 2 high-PVW (HD=256 PVW=2/4) prefill with BF16 native KV.
 */
#define LLAMINAR_CUDA_FLASH_ATTENTION_SHARD 2
#define LLAMINAR_CUDA_FLASH_ATTENTION_SUBSHARD 1
#include "CUDAFlashAttentionShardImpl.cu.inc"
