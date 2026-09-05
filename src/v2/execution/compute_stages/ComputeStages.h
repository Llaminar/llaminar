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
#include "stages/TPKVCacheStateAllGatherStage.h"
#include "stages/AttentionComputeStage.h"

// FFN and residual
#include "stages/ResidualAddStage.h"

// Model-level stages
#include "stages/EmbeddingStage.h"
#include "stages/HiddenStateRowSelectStage.h"
#include "stages/HiddenStateRowsSelectStage.h"
#include "stages/LMHeadStage.h"

// MPI communication stages
#include "stages/AllreduceStage.h"
#include "stages/AllGatherStage.h"

// MoE stages
#include "stages/MoEOverlayTicketPublishStage.h"
#include "stages/MoEOverlayTicketConsumeStage.h"
#include "stages/MoEOverlayActivationPacketStages.h"
#include "stages/MoEExpertDispatchStage.h"
#include "stages/MoELocalExpertStage.h"
#include "stages/MoESparseDispatchStage.h"
#include "stages/MoESparseReturnReduceStage.h"
#include "stages/MoERankBatchSparseStages.h"
#include "stages/MoEDeviceRebalanceStage.h"
#include "stages/MoEDeviceDecodeCommitBoundaryStage.h"

// Qwen 3.5 FA stages
#include "stages/QGateSplitStage.h"
#include "stages/GDNLiveStateAllGatherStage.h"

// MTP sidecar stages
#include "stages/MTPConcatStage.h"
#include "stages/MTPDraftTokenPublicationStage.h"
#include "stages/MoEOverlayEpochBoundaryStage.h"
#include "stages/MoEOverlayDeviceControllerStage.h"
#include "stages/MTPVerifierPreparationStage.h"
#include "stages/MTPVerifierOutcomeStage.h"
#include "stages/MTPStochasticSerialOutcomeStage.h"
#include "stages/MTPStochasticTargetDistributionStage.h"
#include "stages/MTPSpeculativeStatePublicationStage.h"

// Factory
#include "ComputeStageFactory.h"
