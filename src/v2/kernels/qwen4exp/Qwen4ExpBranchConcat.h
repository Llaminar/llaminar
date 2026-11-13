/**
 * @file Qwen4ExpBranchConcat.h
 * @brief Bit-preserving Dense0 four-branch input layout with token-owned counts.
 *
 * Native source/target intervals must be resident on the selected participant.
 * A borrowed original token count gates all four branches; this operation never
 * creates an expanded count or touches inactive admitted rows.
 */
#pragma once
#include "../common/DeviceRowRange.h"
#include <cstddef>
#include "../../backends/DeviceId.h"
namespace llaminar2::qwen4exp {
/** @brief Exact [M,4H]+[M,H] -> [M,4,2H] native layout. */
struct BranchConcatArgs {
    DeviceRowRange rows; ///< Original non-repeated token-row authority.
    int hidden; ///< Per-branch width H.
    const float *hiddenBranches; ///< [M,4H].
    const float *embedding; ///< [M,H].
    float *output; ///< [M,4,2H], [embedding|hidden_branch].
};
/** @brief Validate setup geometry, alignment, pointer spans and non-overlap. */
void validateBranchConcat(const BranchConcatArgs &args);
/** @brief CPU bit copy with original host-owned count and exact inactive preservation. */
void cpuBranchConcat(const BranchConcatArgs &args, void *stream);
/** @brief CUDA bit copy on one explicitly supplied non-default native stream. */
void cudaBranchConcat(const BranchConcatArgs &args, void *stream);
/** @brief HIP bit copy on one explicitly supplied non-default native stream. */
void rocmBranchConcat(const BranchConcatArgs &args, void *stream);
/** @brief Immutable setup-selected native submission function. */
using BranchConcatOperation = void (*)(const BranchConcatArgs &, void *);
/** @brief Select one built backend at setup, rejecting missing implementations. */
BranchConcatOperation createBranchConcatOperation(DeviceId device);
}
