/**
 * @file CUDAFlashAttentionShard4.cu
 * @brief CUDA flash-attention shard 4: context-transaction prefill with FP16 native KV, including partitioned sequences.
 */
#define LLAMINAR_CUDA_FLASH_ATTENTION_SHARD 4
#include "CUDAFlashAttentionShardImpl.cu.inc"
