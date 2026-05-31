/**
 * @file ComputeStages.h
 * @brief Convenience header that includes all compute stage types
 *
 * This header provides backward compatibility and a single include for
 * all compute stage definitions. Include this file to get access to all
 * stage types and the ComputeStageFactory.
 *
 * Individual stages can also be included directly from stages/ subfolder:
 *   #include "execution/compute_stages/stages/GEMMStage.h"
 */

#pragma once

// Base interface and types
#include "IComputeStage.h"

// GEMM stages
#include "stages/GEMMStage.h"
#include "stages/FusedQKVGEMMStage.h"
#include "stages/FusedGateUpGEMMStage.h"

// Normalization and position encoding
#include "stages/RMSNormStage.h"
#include "stages/RoPEStage.h"

// Attention stages
#include "stages/KVCacheAppendStage.h"
#include "stages/KVCacheGatherStage.h"
#include "stages/AttentionComputeStage.h"

// FFN and residual
#include "stages/ResidualAddStage.h"

// Model-level stages
#include "stages/EmbeddingStage.h"
#include "stages/HiddenStateRowSelectStage.h"
#include "stages/LMHeadStage.h"

// MPI communication stages
#include "stages/AllreduceStage.h"
#include "stages/AllGatherStage.h"

// MoE stages
#include "stages/MoEExpertDispatchStage.h"
#include "stages/MoEExpertParallelReduceStage.h"
#include "stages/MoELocalExpertStage.h"
#include "stages/MoESparseDispatchStage.h"
#include "stages/MoESparseReturnReduceStage.h"

// Qwen 3.5 FA stages
#include "stages/QGateSplitStage.h"

// MTP sidecar stages
#include "stages/MTPConcatStage.h"

// Factory
#include "ComputeStageFactory.h"
