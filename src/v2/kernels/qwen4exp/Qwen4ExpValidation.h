/**
 * @file Qwen4ExpValidation.h
 * @brief Shared launch-time binding validation without reading device state.
 *
 * All host checks depend only on immutable capture identity. Live and accepted
 * counts belong to the execution owner and are checked where they reside.
 */
#pragma once
#include "Qwen4ExpKernels.h"
#include <climits>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace llaminar2::qwen4exp::detail
{
/** @brief Fail a malformed binding before any work is submitted. */
inline void require(bool condition, const char *message)
{
    if (!condition) throw std::invalid_argument(message);
}
/** @brief Check admitted geometry without touching the optional live pointer. */
inline void validateExtent(RowExtent extent, int width)
{
    require(extent.rows > 0 && width > 0 && width <= INT_MAX / 4,
            "Qwen4Exp requires positive admitted rows and a valid positive width");
    require(static_cast<std::size_t>(extent.rows) <=
                std::numeric_limits<std::size_t>::max() / (static_cast<std::size_t>(width) * 4 * sizeof(float)),
            "Qwen4Exp admitted geometry overflows addressable storage");
}
/** @brief Validate immutable acceptance policy; dynamic values remain device-owned. */
inline void validatePrefix(const AcceptedPrefix &prefix)
{
    validateExtent(prefix.candidates, 1);
    require(prefix.acceptedRows || (prefix.hostAcceptedRows >= 0 &&
                                      prefix.hostAcceptedRows <= prefix.candidates.rows),
            "Qwen4Exp fixed accepted prefix is outside candidate capacity");
}
/** @brief Require distinct branch gains and a finite positive RMS epsilon. */
inline void validate(const BranchRMSNormArgs &a)
{
    validateExtent(a.extent, a.hidden);
    require(std::isfinite(a.epsilon) && a.epsilon > 0 && a.input && a.gain && a.output && a.output != a.gain,
            "Qwen4Exp branch RMSNorm requires positive finite epsilon and complete distinct gain binding");
}
/** @brief Reject absent embedding expansion bindings. */
inline void validate(const ResidualExpandArgs &a)
{
    validateExtent(a.extent, a.hidden);
    require(a.input && a.output && a.input != a.output,
            "Qwen4Exp residual expansion requires distinct input and output");
}
/** @brief Reject absent low-rank activation bindings. */
inline void validate(const ResidualDownArgs &a)
{
    validateExtent(a.extent, a.lowRank);
    require(a.input && a.output, "Qwen4Exp residual bottleneck requires input and output");
}
/** @brief Reject absent or aliasing branch-read bindings. */
inline void validate(const ResidualReadArgs &a)
{
    validateExtent(a.extent, a.hidden);
    require(a.normalized && a.gateLogits && a.output && a.output != a.normalized && a.output != a.gateLogits,
            "Qwen4Exp residual read requires distinct output and complete input bindings");
}
/** @brief Reject absent branch-write bindings. */
inline void validate(const ResidualWriteArgs &a)
{
    validateExtent(a.extent, a.hidden);
    require(a.residual && a.block && a.gateLogits && a.output && a.output != a.block && a.output != a.gateLogits,
            "Qwen4Exp residual write requires complete bindings and non-broadcast output");
}
/** @brief Reject malformed hash policies before entering captured execution. */
inline void validate(const NgramHashArgs &a)
{
    validateExtent(a.extent, kNgramHeads);
    a.policy.validate();
    require(a.tokens && a.priorTokens && a.output && static_cast<const void *>(a.output) != a.tokens && static_cast<const void *>(a.output) != a.priorTokens,
            "Qwen4Exp hashing requires tokens, immutable prior tokens and distinct output");
}
/** @brief Reject incomplete normalized similarity bindings. */
inline void validate(const PLEGateArgs &a)
{
    validateExtent(a.extent, a.hidden);
    require(a.normalizedKey && a.normalizedQuery && a.value && a.gated &&
                a.gated != a.normalizedKey && a.gated != a.normalizedQuery && a.gated != a.value,
            "Qwen4Exp PLE gate requires complete inputs and distinct gated output");
}
/** @brief Reject incomplete or destructive convolution bindings. */
inline void validate(const PLEConvArgs &a)
{
    validateExtent(a.extent, a.channels);
    require(a.channels % kResidualBranches == 0 && a.normalized && a.priorHistory && a.weights &&
                a.gated && a.residual && a.output && a.output != a.normalized && a.output != a.priorHistory &&
                a.output != a.weights,
            "Qwen4Exp PLE convolution requires complete bindings and immutable normalized history");
}
/** @brief Reject history publication that could overwrite its own source. */
inline void validate(const ConvHistoryCommitArgs &a)
{
    validatePrefix(a.prefix);
    validateExtent(a.prefix.candidates, a.channels);
    require(a.channels % kResidualBranches == 0 && a.priorHistory && a.candidates && a.outputHistory &&
                a.outputHistory != a.priorHistory && a.outputHistory != a.candidates,
            "Qwen4Exp convolution-history commit requires distinct immutable source and destination banks");
}
/** @brief Reject token publication that could overwrite its own source. */
inline void validate(const TokenHistoryCommitArgs &a)
{
    validatePrefix(a.prefix);
    require(a.priorTokens && a.candidates && a.outputTokens &&
                a.outputTokens != a.priorTokens && a.outputTokens != a.candidates,
            "Qwen4Exp token-history commit requires distinct immutable source and destination banks");
}
} // namespace llaminar2::qwen4exp::detail
