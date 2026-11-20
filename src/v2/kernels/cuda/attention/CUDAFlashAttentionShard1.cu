/**
 * @file CUDAFlashAttentionShard1.cu
 * @brief CUDA flash-attention shard 1: ordinary pipelined prefill with FP16 native KV.
 */
#define LLAMINAR_CUDA_FLASH_ATTENTION_SHARD 1
#include "CUDAFlashAttentionShardImpl.cu.inc"
