/**
 * @file ComputeStages.h
 * @brief Convenience header that includes all compute stage types
 *
 * This header provides backward compatibility and a single include for
 * all compute stage definitions. Include this file to get access to all
 * stage types and the ComputeStageFactory.
 *
 * Individual stages can also be included directly from compute_stages/ folder:
 *   #include "execution/compute_stages/GEMMStage.h"
 *   #include "execution/compute_stages/AttentionWithKVCacheStage.h"
 */

#pragma once

// Base interface and types
#include "compute_stages/IComputeStage.h"

// GEMM stages
#include "compute_stages/GEMMStage.h"
#include "compute_stages/FusedQKVGEMMStage.h"
#include "compute_stages/FusedGateUpGEMMStage.h"

// Normalization and position encoding
#include "compute_stages/RMSNormStage.h"
#include "compute_stages/RoPEStage.h"

// Attention stages
#include "compute_stages/AttentionWithKVCacheStage.h"
#include "compute_stages/KVCacheAppendStage.h"
#include "compute_stages/KVCacheGatherStage.h"
#include "compute_stages/AttentionComputeStage.h"
#include "compute_stages/FusedAttentionWoStage.h"

// FFN and residual
#include "compute_stages/SwiGLUStage.h"
#include "compute_stages/ResidualAddStage.h"

// Model-level stages
#include "compute_stages/EmbeddingStage.h"
#include "compute_stages/LMHeadStage.h"
#include "compute_stages/QuantizeToQ16_1Stage.h"

// MPI communication stages
#include "compute_stages/AllreduceStage.h"
#include "compute_stages/AllGatherStage.h"

// MoE stages
#include "compute_stages/MoEStages.h"

// Factory
#include "compute_stages/ComputeStageFactory.h"
