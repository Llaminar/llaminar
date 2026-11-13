/**
 * @file TerminalInputWeights.h
 * @brief Model-owned terminal input validation shared by all runner topologies.
 *
 * A final norm and a branch mixer are different computations. Required mixer
 * operands cannot be replaced by an ordinary norm, and PP stages without the
 * head do not need either. Frozen planning still owns source materialization.
 */
#pragma once
#include "GraphTypes.h"

namespace llaminar2
{
/**
 * @brief Check the complete terminal transformation selected by authenticated model metadata.
 * @param config Model graph contract, including its explicit residual representation.
 * @param weights Participant-owned tensor views resolved from native source identities.
 * @return Whether every operand of that transformation is present.
 */
inline bool hasRequiredTerminalInputWeights(const GraphConfig &config, const ModelWeights &weights)
{
    if (config.flash_next)
        return weights.hyper_connection_head.norm && weights.hyper_connection_head.down &&
               weights.hyper_connection_head.up;
    return weights.final_norm != nullptr;
}
/**
 * @brief Check frozen terminal bindings without requiring an ordinary norm for a mixer.
 * @param config Model-owned terminal transformation, unchanged across primary and replica sources.
 * @param weights Selected immutable source bindings, whose tensor owners remain external.
 * @return Whether each required semantic operand has an actual tensor.
 */
inline bool hasRequiredTerminalInputWeights(const GraphConfig &config, const ModelWeightBindings &weights)
{
    if (config.flash_next)
        return legacyTensor(weights.hyper_connection_head.norm) &&
               legacyTensor(weights.hyper_connection_head.down) &&
               legacyTensor(weights.hyper_connection_head.up);
    return legacyTensor(weights.final_norm) != nullptr;
}
} // namespace llaminar2
