/**
 * @file DeviceGenerationControlLayout.h
 * @brief Canonical fixed-width request controller and terminal observation word layout.
 *
 * Existing execution fields retain their ordinals. QSA work observations use
 * explicit low/high UINT32 pairs; they never decide generation or cache state.
 * The reversible depth comparison owns five learner words and three terminal
 * counters. Only learner words survive an authenticated response continuation;
 * counters restart so terminal consumers never add an earlier window twice.
 * Admission and response-window reset, arena BOM and the existing terminal copy
 * all use this one control-word count. No uint64 pointer/alignment is assumed.
 */
#pragma once
#include <cstdint>
namespace llaminar2::sampling_math {
enum DeviceGenerationControlIndex : int
{
    kDeviceGenerationControlOk = 0,
    kDeviceGenerationControlResponseTokenCount = 1,
    kDeviceGenerationControlRemainingTokenCount = 2,
    kDeviceGenerationControlModelStopped = 3,
    kDeviceGenerationControlRequestComplete = 4,
    kDeviceGenerationControlTransactionCount = 5,
    kDeviceGenerationControlNextLeadingCommittedOutputCount = 6,
    kDeviceGenerationControlTransactionCommitBudget = 7,
    kDeviceGenerationControlAcceptedSpeculativeTokenCount = 8,
    kDeviceGenerationControlRejectedTransactionCount = 9,
    kDeviceGenerationControlConsumedVerifierRowCount = 10,
    kDeviceGenerationControlErrorCode = 11,
    /** Total main-graph state rows committed across every transaction. */
    kDeviceGenerationControlPublishedStateCommitCount = 12,
    /** Active SWITCH selector and number of speculative comparison rows. */
    kDeviceGenerationControlCurrentDraftDepth = 13,
    /** Logical verifier width, including the leading condition row. */
    kDeviceGenerationControlActiveVerifierRowCount = 14,
    kDeviceGenerationControlMinimumDraftDepth = 15,
    kDeviceGenerationControlMaximumDraftDepth = 16,
    kDeviceGenerationControlDepthPolicyMode = 17,
    kDeviceGenerationControlDepthWindowSize = 18,
    kDeviceGenerationControlDepthMinimumSamples = 19,
    kDeviceGenerationControlDepthCooldownSteps = 20,
    kDeviceGenerationControlDepthPromoteConsecutiveWindows = 21,
    kDeviceGenerationControlDepthPromoteFullAcceptRatePPM = 22,
    kDeviceGenerationControlDepthDemoteZeroAcceptRatePPM = 23,
    kDeviceGenerationControlDepthDemoteAcceptanceRatePPM = 24,
    kDeviceGenerationControlDepthStepsSinceChange = 25,
    kDeviceGenerationControlDepthPromotionStreak = 26,
    kDeviceGenerationControlDepthWindowVerifierRuns = 27,
    kDeviceGenerationControlDepthWindowAttemptedTokens = 28,
    kDeviceGenerationControlDepthWindowAcceptedTokens = 29,
    kDeviceGenerationControlDepthWindowRejectedTokens = 30,
    kDeviceGenerationControlDepthWindowRollbacks = 31,
    kDeviceGenerationControlDepthWindowFullAccepts = 32,
    kDeviceGenerationControlDepthWindowZeroAccepts = 33,
    kDeviceGenerationControlDepthWindowAcceptedPrefixSum = 34,
    kDeviceGenerationControlDepthEvaluatedWindows = 35,
    kDeviceGenerationControlDepthUpdates = 36,
    kDeviceGenerationControlDepthPromotions = 37,
    kDeviceGenerationControlDepthDemotions = 38,
    kDeviceGenerationControlDepthLastRecommendedDepth = 39,
    /** Sum of device-selected speculative widths across transactions. */
    kDeviceGenerationControlAttemptedDraftTokenCount = 40,
    /** Sum of logical verifier widths, including each condition row. */
    kDeviceGenerationControlVerifierTokenCount = 41,
    /** Device-selected draft width of the most recently committed transaction. */
    kDeviceGenerationControlLastTransactionDraftDepth = 42,
    /** Response tokens appended by the most recently committed transaction. */
    kDeviceGenerationControlLastTransactionEmittedTokenCount = 43,
    /** Number of MoE layers that moved current-batch LLEP payloads. */
    kDeviceGenerationControlCurrentBatchLLEPMovementLayerCount = 44,
    /** Number of MoE layers that executed rows away from their static owner. */
    kDeviceGenerationControlCurrentBatchLLEPNonOwnerAssignmentLayerCount = 45,
    /** Immutable learned policy keys; observations remain device-owned. */
    kDeviceGenerationControlLearnedDepthEnabled = 46,
    kDeviceGenerationControlLearnedDepthBackend = 47,
    kDeviceGenerationControlLearnedDepthModelClass = 48,
    kDeviceGenerationControlLearnedDepthVerifyMode = 49,
    /** Proof that generated rules, including holds, controlled live windows. */
    kDeviceGenerationControlLearnedDepthMatchedWindows = 50,
    /** Immutable initial selector, authenticated across response windows. */
    kDeviceGenerationControlInitialDraftDepth = 51,
    /** Required original-state header bytes actually published by QSA. */
    kDeviceGenerationControlQsaPriorHeaderBytesLow = 52,
    kDeviceGenerationControlQsaPriorHeaderBytesHigh = 53,
    /** Live original-tail payload bytes actually copied by QSA. */
    kDeviceGenerationControlQsaPriorPayloadBytesLow = 54,
    kDeviceGenerationControlQsaPriorPayloadBytesHigh = 55,
    /** Live raw candidate payload bytes retained by QSA. */
    kDeviceGenerationControlQsaCandidatePayloadBytesLow = 56,
    kDeviceGenerationControlQsaCandidatePayloadBytesHigh = 57,
    /** Live accepted partial-tail bytes written, including zero acceptance restores. */
    kDeviceGenerationControlQsaAcceptedPayloadBytesLow = 58,
    kDeviceGenerationControlQsaAcceptedPayloadBytesHigh = 59,
    /** Fixed-head prefill observations survive initial response admission; continuation windows reset them.
     */
    kDeviceGenerationControlFixedHeadCallsLow = 60,
    kDeviceGenerationControlFixedHeadCallsHigh = 61,
    kDeviceGenerationControlFixedHeadLiveIdsLow = 62,
    kDeviceGenerationControlFixedHeadLiveIdsHigh = 63,
    kDeviceGenerationControlFixedHeadNativeBytesLow = 64,
    kDeviceGenerationControlFixedHeadNativeBytesHigh = 65,
    kDeviceGenerationControlFixedHeadFailuresLow = 66,
    kDeviceGenerationControlFixedHeadFailuresHigh = 67,
    /** Actual outgoing native ring-send bytes; terminal projection authenticates P-1 multiplicity. */
    kDeviceGenerationControlFixedHeadHeaderNativeSendBytesLow = 68,
    kDeviceGenerationControlFixedHeadHeaderNativeSendBytesHigh = 69,
    kDeviceGenerationControlFixedHeadPayloadNativeSendBytesLow = 70,
    kDeviceGenerationControlFixedHeadPayloadNativeSendBytesHigh = 71,
    /** Actual adaptive owner work; prefill survives only the initial response admission. */
    kDeviceGenerationControlAdaptiveCallsLow = 72,
    kDeviceGenerationControlAdaptiveCallsHigh = 73,
    kDeviceGenerationControlAdaptiveCompletedLiveIdsLow = 74,
    kDeviceGenerationControlAdaptiveCompletedLiveIdsHigh = 75,
    kDeviceGenerationControlAdaptiveHitRowsLow = 76,
    kDeviceGenerationControlAdaptiveHitRowsHigh = 77,
    kDeviceGenerationControlAdaptiveMissRowsLow = 78,
    kDeviceGenerationControlAdaptiveMissRowsHigh = 79,
    kDeviceGenerationControlAdaptiveSourceRowsLow = 80,
    kDeviceGenerationControlAdaptiveSourceRowsHigh = 81,
    kDeviceGenerationControlAdaptiveSourceBytesLow = 82,
    kDeviceGenerationControlAdaptiveSourceBytesHigh = 83,
    kDeviceGenerationControlAdaptivePromotionsLow = 84,
    kDeviceGenerationControlAdaptivePromotionsHigh = 85,
    kDeviceGenerationControlAdaptiveEvictionsLow = 86,
    kDeviceGenerationControlAdaptiveEvictionsHigh = 87,
    kDeviceGenerationControlAdaptiveFailedFramesLow = 88,
    kDeviceGenerationControlAdaptiveFailedFramesHigh = 89,
    /** Immutable measured scope; not live device observations. */
    kDeviceGenerationControlLearnedScopeWidthPolicy = 90,
    kDeviceGenerationControlLearnedScopeTPWidth = 91,
    kDeviceGenerationControlLearnedScopePPStages = 92,
    kDeviceGenerationControlLearnedScopeRetainedDepth = 93,
    kDeviceGenerationControlLearnedScopeHiddenDim = 94,
    kDeviceGenerationControlLearnedScopeMainLayers = 95,
    kDeviceGenerationControlLearnedScopeExpertCount = 96,
    kDeviceGenerationControlLearnedScopeExpertsPerToken = 97,
    kDeviceGenerationControlLearnedScopeComputeCapabilityMajor = 98,
    kDeviceGenerationControlLearnedScopeComputeCapabilityMinor = 99,
    /** Typed reversible comparison phase; zero is an empty comparison. */
    kDeviceGenerationControlDepthUtilityPhase = 100,
    /** Incumbent while probing, or failed target while retaining a rejection. */
    kDeviceGenerationControlDepthUtilityReferenceDepth = 101,
    /** Eligible physical transactions in the remembered comparison sample. */
    kDeviceGenerationControlDepthUtilityReferenceRuns = 102,
    /** Actual verified-state progress in the saved eligible sample, distinct from emitted outputs. */
    kDeviceGenerationControlDepthUtilityReferenceVerifiedProgress = 103,
    /** Actual emitted tokens in the current eligible controller window. */
    kDeviceGenerationControlDepthUtilityWindowEmitted = 104,
    /** Real depth trials begun in this response publication window. */
    kDeviceGenerationControlDepthUtilityProbes = 105,
    /** Trials retained after a strict estimated-utility improvement. */
    kDeviceGenerationControlDepthUtilityKeeps = 106,
    /** Trials returned to their incumbent, including estimated-utility ties. */
    kDeviceGenerationControlDepthUtilityReturns = 107,
    kDeviceGenerationControlCount = 108,
};
/** The mutable comparison and its response-local reporting have disjoint ranges. */
static_assert(kDeviceGenerationControlDepthUtilityWindowEmitted + 1 ==
              kDeviceGenerationControlDepthUtilityProbes);
static_assert(kDeviceGenerationControlDepthUtilityReturns + 1 ==
              kDeviceGenerationControlCount);
} // namespace llaminar2::sampling_math
