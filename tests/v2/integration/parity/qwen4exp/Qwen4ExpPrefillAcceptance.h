/**
 * @file Qwen4ExpPrefillAcceptance.h
 * @brief QFN-only complete main-decoder numerical acceptance before prefix reuse.
 *
 * The shared parity framework historically counts embedding alongside early
 * decoders. QFN requires all 48 original main layers independently, without
 * changing that legacy counting policy or the thresholds also used by decode.
 */
#pragma once
#include <algorithm>
#include <gtest/gtest.h>

namespace llaminar2::test::parity::qwen4exp
{
/** @brief Numerical policy never changes which original observations must exist. */
enum class Qwen4ExpMainEvidencePolicy { StrictLayerNumerics, FiniteExactObservations };

/**
 * @brief Authenticate each actual raw witness without requiring its diagnostic cosine verdict.
 * @tparam Layer Ordinary or combined TP layer record.
 * @param layer Original source-indexed record with its real comparison bank.
 * @return First missing/invalid stage, or success. No default-valid synthetic evidence exists.
 */
template <typename Layer>
::testing::AssertionResult acceptsQwen4ExpLayerEvidence(const Layer &layer)
{
    if (layer.stages_compared <= 0 || layer.stage_results.empty())
        return ::testing::AssertionFailure() << "QFN structural evidence: no actual stages for decoder " << layer.layer_idx;
    for (const auto &record : layer.stage_results)
    {
        const auto &stage = [&]() -> const auto & {
            if constexpr (requires { record.combined_result; }) return record.combined_result;
            else return record;
        }();
        if (!stage.evidence_valid || (stage.integer_mismatch_count && *stage.integer_mismatch_count != 0))
            return ::testing::AssertionFailure() << "QFN structural evidence: missing, nonfinite, malformed or non-exact stage "
                << stage.stage_name << " in decoder " << layer.layer_idx;
    }
    return ::testing::AssertionSuccess();
}

/**
 * @brief Accept only complete original QFN main-model prefill evidence.
 * @tparam Layers Ordinary or TP layer-record collection, retaining source indices.
 * @param layers Original comparison records; neither metrics nor verdicts are changed.
 * @param embedding_passed Original strict embedding verdict, or actual finite/count witness under the explicit ordinary-logit policy.
 * @param logits_passed Strict logit verdict, or independent actual final-row witness when numerical KL is judged separately.
 * @param policy Strict legacy numerical layers or independently valid finite/exact observations.
 * @return Success only for good embedding/logits and exactly 48 original main records, or a precise diagnostic.
 *
 * A missing, duplicated or out-of-main record cannot replace any decoder.
 * Vector order is diagnostic only. Sidecar layers cannot substitute for a
 * failed main layer; this main-model contract makes no MTP qualification claim.
 */
template <typename Layers>
::testing::AssertionResult acceptsQwen4ExpCompletePrefill(const Layers &layers, bool embedding_passed, bool logits_passed,
    Qwen4ExpMainEvidencePolicy policy = Qwen4ExpMainEvidencePolicy::StrictLayerNumerics)
{
    if (!embedding_passed)
        return ::testing::AssertionFailure() << "QFN complete prefill: failed or missing EMBEDDING";
    if (!logits_passed)
        return ::testing::AssertionFailure() << "QFN complete prefill: failed or missing LM_HEAD";
    for (const auto &layer : layers)
    {
        if (layer.layer_idx < 0 || layer.layer_idx >= 48)
            return ::testing::AssertionFailure()
                << "QFN complete prefill: foreign decoder index " << layer.layer_idx
                << "; expected exactly 48 original decoder records 0..47, observed " << layers.size();
    }
    for (int index = 0; index < 48; ++index)
    {
        const auto matches = [index](const auto &layer) { return layer.layer_idx == index; };
        const auto count = std::count_if(layers.begin(), layers.end(), matches);
        if (count != 1)
            return ::testing::AssertionFailure()
                << "QFN complete prefill: " << (count == 0 ? "missing" : "duplicate")
                << " decoder index " << index << " (records=" << count << "); expected exactly 48, observed "
                << layers.size();
        const auto layer = std::find_if(layers.begin(), layers.end(), matches);
        if (policy == Qwen4ExpMainEvidencePolicy::StrictLayerNumerics)
        {
            if (!layer->passed)
                return ::testing::AssertionFailure() << "QFN complete prefill: failed decoder index " << index;
        }
        else if (const auto evidence = acceptsQwen4ExpLayerEvidence(*layer); !evidence)
            return evidence;
    }
    // All identities must occur exactly once, with no foreign record. Keep the
    // physical record cardinality explicit instead of accepting a valid subset.
    if (layers.size() != 48)
        return ::testing::AssertionFailure()
            << "QFN complete prefill: expected exactly 48 decoder records, observed " << layers.size();
    return ::testing::AssertionSuccess();
}
} // namespace llaminar2::test::parity::qwen4exp
