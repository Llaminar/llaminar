/**
 * @file CUDAFlashAttentionShard2.cu
 * @brief CUDA flash-attention shard 2: ordinary pipelined prefill with BF16 native KV.
 */
#define LLAMINAR_CUDA_FLASH_ATTENTION_SHARD 2
#include "CUDAFlashAttentionShardImpl.cu.inc"
