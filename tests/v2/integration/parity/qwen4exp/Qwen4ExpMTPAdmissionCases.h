/**
 * @file Qwen4ExpMTPAdmissionCases.h
 * @brief Gold-free R1 sidecar admission inputs derived from canonical main topology declarations.
 *
 * Admission does not declare numerical MTP support. Eight independent cases
 * retain the main source, topology, precision and RAM/mmap policy while explicitly
 * authoring a fixed-one execution and retained envelope before production planning.
 * No existing main MTPOff cell or standard depth15 matrix is modified.
 */
#pragma once
#include "integration/parity/qwen4exp/Qwen4ExpProductionParityCases.h"
#include "loaders/SharedPredictorGGUFSource.h"

namespace llaminar2::test::parity::qwen4exp
{
/** @brief Explicit paired-source execution intent for admission and its inactive control. */
enum class Qwen4ExpMTPAdmissionExecution { FixedOne, Inactive };

/** @brief A main geometry/topology declaration plus a separate, explicit sidecar source. */
struct Qwen4ExpMTPAdmissionCase
{
    ModelParityCase main;
    SharedPredictorGGUFSource predictor{std::string(kPredictorModelPath)};

    /** @return A separate admission identity; never the main MTPOff certificate name. */
    std::string testName() const
    {
        auto name = main.testName();
        constexpr std::string_view suffix = "MTPOff";
        if (!name.ends_with(suffix))
            throw std::logic_error("QFN sidecar admission requires the unchanged main-only declaration");
        name.resize(name.size() - suffix.size());
        return name + "MTPR1Fixed1Retained1";
    }

    /**
     * @brief Resolve this fixed admission input through the public configuration projection.
     * @param execution Explicit active or admitted-but-inactive paired source.
     * @return Full48/context4096/R1 configuration; no source or resource substitution.
     * @throws std::invalid_argument For a foreign mode or declaration without explicit hosted storage.
     */
    OrchestrationConfig config(Qwen4ExpMTPAdmissionExecution execution) const
    {
        if (execution != Qwen4ExpMTPAdmissionExecution::FixedOne &&
            execution != Qwen4ExpMTPAdmissionExecution::Inactive)
            throw std::invalid_argument("Unknown QFN MTP admission execution intent");
        if (!main.ngram_table_policy ||
            (main.ngram_table_policy->storage != ImmutableRowStoragePreference::RAM &&
             main.ngram_table_policy->storage != ImmutableRowStoragePreference::MemoryMapped))
            throw std::invalid_argument("QFN MTP admission requires its exact RAM/mmap source policy");
        auto result = main.makeOrchestrationConfig(main.model.model_path, 0);
        result.shared_predictor = predictor;
        result.batch_size = 1;
        result.mtp_activation_policy = execution == Qwen4ExpMTPAdmissionExecution::FixedOne
            ? MTPActivationPolicy::Enabled : MTPActivationPolicy::Disabled;
        result.mtp.enabled = execution == Qwen4ExpMTPAdmissionExecution::FixedOne;
        result.mtp.draft_tokens = 1;
        result.mtp.graph_capacity_draft_tokens = result.mtp.enabled ? 1 : 0;
        result.mtp.max_request_batch = 1;
        result.mtp.depth_policy = MTPDepthPolicyConfig{};
        result.mtp.depth_policy.mode = MTPDepthPolicyMode::Fixed;
        // Match the existing canonical main admission's upfront prefix geometry.
        result.prefix_cache.block_size = productionParityPrefixRestoreProofBlockSize(main.model.token_ids.size());
        result.deterministic = true;
        return result;
    }
};

/** @return Four existing topologies ×RAM/mmap, selecting only Static/Ordinal overlay ownership. */
inline const std::vector<Qwen4ExpMTPAdmissionCase> &qwen4ExpMTPAdmissionCases()
{
    static const auto cases = [] {
        std::vector<Qwen4ExpMTPAdmissionCase> result;
        for (const auto mode : {Qwen4ExpParityTopology::CPU, Qwen4ExpParityTopology::CUDACPU,
                               Qwen4ExpParityTopology::ROCmCPU, Qwen4ExpParityTopology::CUDAROCmCPU})
            for (auto &cell : expandModelParityDefinition(qwen4ExpProductionDefinition(mode)))
            {
                if (cell.expert_overlay && *cell.expert_overlay != ModelParityExpertOverlayPolicy{}) continue;
                result.push_back({.main = std::move(cell)});
            }
        return result;
    }();
    return cases;
}
} // namespace llaminar2::test::parity::qwen4exp
