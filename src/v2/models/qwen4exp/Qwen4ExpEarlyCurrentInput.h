/**
 * @file Qwen4ExpEarlyCurrentInput.h
 * @brief Select accepted-state or genuine verifier-prior history for the original early-current invocation.
 *
 * Scalar MTP conditions share serial accepted history. Their sequence-length
 * mailbox is a position, never a query-row count. This role check does not
 * grant input/source readiness: the existing original PLE query and native
 * protocol still authenticate those authorities at graph construction.
 */
#pragma once
namespace llaminar2 {
struct GraphConfig;
struct ForwardInput;
/** @brief The existing history owner consumed by this one lexical graph build. */
enum class Qwen4ExpEarlyCurrentHistory { AcceptedState, VerifierPrior };
/**
 * @brief Resolve only supported original invocation roles before early Hash construction.
 * @param config Current immutable graph policy installed by the original invocation scope.
 * @param input Original forward input; no device value is observed by this check.
 * @return Accepted history for ordinary/scalar condition work, or the genuine grouped prior.
 * @throws std::invalid_argument For a mismatched role/policy or nonscalar/nondecode condition.
 */
[[nodiscard]] Qwen4ExpEarlyCurrentHistory resolveQwen4ExpEarlyCurrentHistory(
    const GraphConfig& config, const ForwardInput& input);
} // namespace llaminar2
