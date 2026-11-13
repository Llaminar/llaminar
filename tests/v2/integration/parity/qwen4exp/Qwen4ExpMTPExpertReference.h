/**
 * @file Qwen4ExpMTPExpertReference.h
 * @brief Canonical original-source expert-only reference for QFN's4H HC predictor.
 *
 * Only selected expert IDs cross into Python. Its named input is the authenticated
 * HF compute-width FFN_NORM, with the separate wide terminal-input schema retained.
 * This bank cannot grant an ordinary-RMS suffix authority or rewrite HC/logits.
 */
#pragma once
#include "../MoERouteConditionedReference.h"
#include "../../../utils/TypedNpySnapshot.h"

namespace llaminar2::test::parity::qwen4exp
{
/** @brief Exact paired sources and immutable canonical checkpoint consumed by this bank. */
struct Qwen4ExpMTPExpertReferenceIdentity
{
    std::filesystem::path main;
    std::filesystem::path predictor;
    std::filesystem::path snapshots;
    std::filesystem::path artifacts;
    std::string checkpoint_prefix;
};
/**
 * @brief Evaluate/reuse original-source selected experts on an independently authenticated HF row.
 * @param identity Explicit main/source48 pair and exact canonical or branch-qualified checkpoint.
 * @param expert_ids Sorted unique union of validated original top10 native/HF selections (at most20).
 * @param rows Exact scalar R1 predictor row count; grouped/independent requests are not advertised.
 * @param hidden Original compute-width expert input/output dimension.
 * @return Immutable FP32 bank with explicit CanonicalComparisons suffix contract.
 * @throws std::runtime_error On incomplete source/schema/extent, malformed cache, or failed reconstruction.
 * @note ReferenceGenerationLease serializes validation/publication; no inference or native state enters Python.
 */
inline MoEIndependentReference loadQwen4ExpMTPExpertReference(
    const Qwen4ExpMTPExpertReferenceIdentity &identity, const std::vector<int> &expert_ids,
    std::size_t rows, std::size_t hidden)
{
    if (identity.main.empty() || identity.predictor.empty() || identity.checkpoint_prefix.empty() ||
        rows != 1 || !hidden || expert_ids.empty() || expert_ids.size() > 20 ||
        identity.checkpoint_prefix.find_first_not_of(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_") != std::string::npos)
        throw std::runtime_error("QFN independent expert reference requires exact paired-source scalar identity");
    std::ostringstream ids;
    std::string basename;
    for (std::size_t index = 0; index < expert_ids.size(); ++index)
    {
        if (expert_ids[index] < 0 || expert_ids[index] >= 512 ||
            (index && expert_ids[index - 1] >= expert_ids[index]))
            throw std::runtime_error("QFN independent expert IDs must be sorted unique original coordinates");
        if (index) ids << ',';
        ids << expert_ids[index];
        basename += std::to_string(expert_ids[index]) + "_";
    }
    const auto output = identity.snapshots / "route_conditioned" / "qwen4exp" /
        identity.checkpoint_prefix / (basename + ".npy");
    ReferenceGenerationLease lease;
    std::ostringstream script;
    script << "unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS OMP_PROC_BIND OMP_PLACES KMP_AFFINITY; "
           << "if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi; "
           << "python3 python/reference/generate_qwen4exp_mtp_expert_reference.py --model "
           << moeReferenceShellQuote(identity.main.string())
           << " --shared-predictor " << moeReferenceShellQuote(identity.predictor.string())
           << " --reference-dir " << moeReferenceShellQuote(identity.snapshots.string())
           << " --reference-prefix " << moeReferenceShellQuote(identity.checkpoint_prefix)
           << " --expert-ids " << moeReferenceShellQuote(ids.str())
           << " --output " << moeReferenceShellQuote(output.string());
    const auto command = "bash -c " + moeReferenceShellQuote(script.str()) + " 2>&1";
    FILE *pipe = popen(command.c_str(), "r");
    if (!pipe) throw std::runtime_error("Cannot launch canonical QFN independent expert reference");
    std::string diagnostics;
    std::array<char, 512> buffer{};
    while (fgets(buffer.data(), buffer.size(), pipe)) diagnostics += buffer.data();
    const int status = pclose(pipe);
    std::filesystem::create_directories(identity.artifacts);
    std::ofstream log(identity.artifacts / "qwen4exp_independent_expert_reference.log", std::ios::app);
    log << diagnostics;
    if (status != 0)
        throw std::runtime_error("Canonical QFN independent expert generation failed: " + diagnostics);
    const auto descriptor = ::llaminar2::test::inspectNpySnapshot(output.string());
    if (descriptor.scalar != ::llaminar2::test::NpySnapshotScalar::FP32 ||
        descriptor.shape != std::vector<std::size_t>{expert_ids.size(), rows, hidden})
        throw std::runtime_error("QFN independent expert bank has a different original scalar/row geometry");
    const auto bank = cnpy::npy_load(output.string());
    const auto *values = bank.data<float>();
    return {.experts = {values, values + bank.num_vals}, .suffix = MoECanonicalComparisons{}};
}
} // namespace llaminar2::test::parity::qwen4exp
