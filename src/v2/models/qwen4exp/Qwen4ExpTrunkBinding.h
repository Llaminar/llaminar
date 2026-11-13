/**
 * @file Qwen4ExpTrunkBinding.h
 * @brief Resolve the native residual trunk from its canonical activation role.
 *
 * The residual is a core activation field, not an extension-map entry. Graph
 * fragments may remap its arena identity while retaining the exact tensor view;
 * lexical and branch stages must consume that view without a second alias map.
 */
#pragma once
#include "../GraphTypes.h"
#include "../../execution/compute_stages/stages/qwen4exp/Qwen4ExpPrimitiveStage.h"
#include <stdexcept>

namespace llaminar2
{
/**
 * @brief Borrow the graph fragment's native residual tensor and physical arena identity.
 * @param buffers Canonical core fields and optional fragment identity remaps.
 * @return Exact FP32 trunk binding; no extension lookup or storage change occurs.
 * @throws std::invalid_argument If the fragment has no declared residual tensor.
 */
inline Qwen4ExpFloatBinding qwen4ExpTrunkBinding(const ActivationBuffers &buffers)
{
    if (!buffers.current_hidden)
        throw std::invalid_argument("Flash Next requires its canonical residual trunk binding");
    return {*buffers.current_hidden, buffers.idFor(BufferId::HIDDEN_STATE)};
}
} // namespace llaminar2
