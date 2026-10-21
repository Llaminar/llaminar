/**
 * @file Qwen4ExpPLERetainedDispatch.h
 * @brief Setup-only native PLE operation selection, independent of graph stages.
 */
#pragma once
#include "Qwen4ExpPLERetained.h"
#include "../../backends/DeviceId.h"
namespace llaminar2::qwen4exp {
/** @brief Immutable typed submission bundle; no owned workspace or mutable policy. */
struct PLETransactionOperations {
 void (*snapshot)(const PLESnapshotPrior &,void *);
 void (*retain)(const PLERetainCandidates &,void *);
 void (*publish)(const PLEPublishAccepted &,void *);
};
/** @brief Select supported native operations once; unsupported placement fails at setup. */
PLETransactionOperations createPLETransactionOperations(DeviceId device);
}
