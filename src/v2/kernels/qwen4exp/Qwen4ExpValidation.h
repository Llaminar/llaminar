/**
 * @file Qwen4ExpValidation.h
 * @brief Shared launch-time binding validation without reading device state.
 *
 * All host checks depend only on immutable capture identity. Live and accepted
 * counts belong to the execution owner and are checked where they reside.
 */
#pragma once
#include "Qwen4ExpKernels.h"
#include "kernels/common/Q8ResidualCorrection.h"
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
/** @brief Validate original fully-active M1 norm and distinct preadmitted consumer banks. */
inline void validate(const BranchRMSNormSingleQ8Args& a) {
    validate(a.norm);
    require(a.norm.extent.rows==1&&!a.norm.extent.liveRows&&a.norm.hidden%32==0&&a.codes&&a.scales,
        "HC norm SingleQ8 requires original fully-active M1 with complete branch K32 blocks");
    const auto values=std::size_t(4)*a.norm.hidden;
    const std::array writes{Q8ResidualStorageSpan{a.norm.output,values*sizeof(float),alignof(float)},
        Q8ResidualStorageSpan{a.codes,values,16},Q8ResidualStorageSpan{a.scales,values/32*sizeof(float),alignof(float)}};
    const std::array reads{Q8ResidualStorageSpan{a.norm.input,a.norm.input==a.norm.output?0:values*sizeof(float),alignof(float)},
        Q8ResidualStorageSpan{a.norm.gain,values*sizeof(float),alignof(float)}};
    validateQ8ResidualStorage(writes,reads);
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
/**
 * @brief Reject overlap and overflow in the explicit two-output write/next-norm contract.
 * @param a Complete static binding; live rows remain on their execution owner.
 * @throws std::invalid_argument For incomplete bindings, partial overlap or output aliases.
 */
inline void validate(const ResidualWriteBranchRMSNormArgs &a)
{
    validate(ResidualWriteArgs{a.extent,a.hidden,a.residual,a.block,a.gateLogits,a.output});
    validate(BranchRMSNormArgs{a.extent,a.hidden,a.epsilon,a.output,a.gain,a.normalized});
    const auto trunk = static_cast<std::size_t>(a.extent.rows) * 4 * a.hidden * sizeof(float);
    const auto block = static_cast<std::size_t>(a.extent.rows) * a.hidden * sizeof(float);
    const auto logits = static_cast<std::size_t>(a.extent.rows) * 4 * sizeof(float);
    const auto gain = static_cast<std::size_t>(4) * a.hidden * sizeof(float);
    const auto disjoint = [](const void *left, std::size_t leftBytes, const void *right, std::size_t rightBytes) {
        const auto l = reinterpret_cast<std::uintptr_t>(left), r = reinterpret_cast<std::uintptr_t>(right);
        require(l <= std::numeric_limits<std::uintptr_t>::max() - leftBytes &&
                r <= std::numeric_limits<std::uintptr_t>::max() - rightBytes,
                "Qwen4Exp fused write/branch norm pointer extent overflows");
        return l + leftBytes <= r || r + rightBytes <= l;
    };
    require((a.output == a.residual || disjoint(a.output,trunk,a.residual,trunk)) &&
            disjoint(a.output,trunk,a.block,block) && disjoint(a.output,trunk,a.gateLogits,logits) &&
            disjoint(a.output,trunk,a.gain,gain) && disjoint(a.normalized,trunk,a.residual,trunk) &&
            disjoint(a.normalized,trunk,a.block,block) && disjoint(a.normalized,trunk,a.gateLogits,logits) &&
            disjoint(a.normalized,trunk,a.output,trunk) && disjoint(a.normalized,trunk,a.gain,gain) &&
            (!a.extent.liveRows || (disjoint(a.output,trunk,a.extent.liveRows,sizeof(int)) &&
                                   disjoint(a.normalized,trunk,a.extent.liveRows,sizeof(int)))),
            "Qwen4Exp fused write/branch norm requires distinct normalized bytes and exact residual alias only");
}
/** @brief Reject malformed hash policies before entering captured execution. */
inline void validate(const NgramHashArgs &a)
{
    validateExtent(a.extent, kNgramHeads);
    a.policy.validate();
    require(a.tokens && a.priorTokens && a.output && static_cast<const void *>(a.output) != a.tokens && static_cast<const void *>(a.output) != a.priorTokens,
            "Qwen4Exp hashing requires tokens, immutable prior tokens and distinct output");
}
/** @brief Validate only immutable next-window bindings; live values remain on their execution owner. */
inline void validate(const NgramNextWindowHashArgs& a)
{
    validate(NgramHashArgs{a.extent,a.tokens,a.priorTokens,a.output,a.policy});
    require((a.nextWindow.declared() &&
        a.tokens==a.nextWindow.request_tokens && !a.extent.liveRows && a.extent.rows==a.nextWindow.next_bucket),
        "Qwen4Exp next-window hash requires one complete original token/count alternative");
    if(!a.nextWindow.absent()){
        const auto out=reinterpret_cast<std::uintptr_t>(a.output);
        const auto bytes=std::size_t(a.extent.rows)*kNgramHeads*sizeof(std::int64_t);
        require(out<=std::numeric_limits<std::uintptr_t>::max()-bytes,"QFN next-window output extent overflows");
        const auto disjoint=[&](const std::int32_t* p,std::size_t count){
            const auto input=reinterpret_cast<std::uintptr_t>(p);
            return input+count*sizeof(std::int32_t)<=out || out+bytes<=input;
        };
        require(disjoint(a.tokens,a.nextWindow.request_capacity)&&disjoint(a.priorTokens,2)&&
            disjoint(a.nextWindow.request_positions,1)&&disjoint(a.nextWindow.request_total,1)&&
            disjoint(a.nextWindow.current_positions,1)&&disjoint(a.nextWindow.current_rows,1),
            "QFN next-window hash inputs must not alias its output");
    }
    require(a.nextWindow.declared(),"QFN next-window Hash cannot use an absent source");
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
