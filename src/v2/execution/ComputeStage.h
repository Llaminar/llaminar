/**
 * @file ComputeStage.h
 * @brief Backward-compatibility header - includes ComputeStages.h
 * @author David Sanftenberg
 * @date December 2025
 *
 * @deprecated Use ComputeStages.h or include individual stage headers directly.
 *
 * This header now forwards to the modular compute_stages/ headers.
 * For new code, prefer including specific headers:
 *   #include "execution/compute_stages/GEMMStage.h"
 *   #include "execution/compute_stages/AttentionWithKVCacheStage.h"
 *
 * Or use the convenience header:
 *   #include "execution/ComputeStages.h"
 */

#pragma once

// Forward to the new modular headers
#include "ComputeStages.h"

// Legacy includes that some files may depend on
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include "DeviceContext.h"
#include "BufferRole.h"                 // For buffer requirements
#include "RuntimeConfig.h"              // For ActivationPrecision, FusedAttentionBackend
#include "../tensors/BlockStructures.h" // For Q8_1Block in StageDumpInfo
#include "../tensors/TensorKernels.h"   // For AttentionMode enum
#include "../utils/MPITopology.h"       // For WorkRange (tensor parallelism)
