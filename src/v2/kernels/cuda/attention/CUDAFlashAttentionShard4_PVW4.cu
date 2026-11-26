/**
 * @file CUDAFlashAttentionShard4_PVW4.cu
 * @brief CUDA flash-attention shard 4 HD=256 PVW=4 context-transaction prefill with FP16 native KV.
 */
#define LLAMINAR_CUDA_FLASH_ATTENTION_SHARD 4
#define LLAMINAR_CUDA_FLASH_ATTENTION_SUBSHARD 3
#include "CUDAFlashAttentionShardImpl.cu.inc"
