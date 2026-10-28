/**
 * @file Qwen4ExpMTPAdmissionCases.h
 * @brief Gold-free R1 sidecar admission inputs derived from canonical main topology declarations.
 *
 * Admission does not declare numerical MTP support. Original admission cases and explicit A1 fixed/dynamic declarations
 * retain the main source, topology, precision and RAM/mmap policy while explicitly
 * authoring runtime execution and retained envelopes before production planning.
 * No existing main MTPOff cell or standard depth15 matrix is modified.
 */
#pragma once
#include "integration/parity/qwen4exp/Qwen4ExpProductionParityCases.h"
#include "loaders/SharedPredictorGGUFSource.h"
#include "Qwen4ExpFullROCmTP4ParityCases.h"
#include "integration/parity/ModelParityRuntimeExport.h"

namespace llaminar2::test::parity::qwen4exp
{
/** @brief Explicit paired-source execution intent for admission and its inactive control. */
enum class Qwen4ExpMTPAdmissionExecution { Active, Inactive };

/** @brief Separate admission proof, explicit HTTP qualification and production workload tuning. */
enum class Qwen4ExpMTPAdmissionPurpose
{
    AdmissionDiagnostic,
    RuntimeQualification,
    ProductionServing,
};

/**
 * @brief Name only an explicit runtime-export purpose before opening any artifact owner.
 * @param purpose Qualification or serving projection, never numerical admission diagnostics.
 * @return Stable source-owned purpose name carried by the paired runtime record.
 * @throws std::invalid_argument For admission diagnostics or an unknown purpose.
 */
inline const char *qwen4ExpMTPRuntimePurposeName(Qwen4ExpMTPAdmissionPurpose purpose)
{
    switch (purpose)
    {
    case Qwen4ExpMTPAdmissionPurpose::RuntimeQualification: return "runtime_qualification";
    case Qwen4ExpMTPAdmissionPurpose::ProductionServing: return "production_serving";
    default: throw std::invalid_argument("QFN runtime export requires an explicit runtime purpose");
    }
}

/** @brief A main geometry/topology declaration plus a separate, explicit sidecar source. */
struct Qwen4ExpMTPAdmissionCase
{
    ModelParityCase main;
    SharedPredictorGGUFSource predictor{std::string(kPredictorModelPath)};
    /** Runtime-only policy; never changes main.model reference capacity. */
    ModelParityMTP policy = ModelParityMTP::Depth1;
    /** Explicit physical draft envelope owned by this R1 admission. */
    int retained_depth = 1;
    /** Explicit terminal ownership; original admissions retain their automatic policy. */
    MTPTerminalHeadPolicy terminal_head_policy = MTPTerminalHeadPolicy::Automatic;

    /** @return Validated runtime depth without claiming sidecar reference coverage. */
    int requestedDepth() const
    {
        switch (policy)
        {
        case ModelParityMTP::Depth1: return 1;
        case ModelParityMTP::Depth2: return 2;
        case ModelParityMTP::Depth3: return 3;
        case ModelParityMTP::Depth5: return 5;
        case ModelParityMTP::Depth6: return 6;
        case ModelParityMTP::Depth15:
        case ModelParityMTP::DynamicDepth: return kModelParityRequiredMaximumMTPDepth;
        default: throw std::invalid_argument("Unknown active QFN admission policy");
        }
    }

    /** @return A separate admission identity; never the main MTPOff certificate name. */
    std::string testName() const
    {
        auto name = main.testName();
        constexpr std::string_view suffix = "MTPOff";
        if (!name.ends_with(suffix))
            throw std::logic_error("QFN sidecar admission requires the unchanged main-only declaration");
        name.resize(name.size() - suffix.size());
        return name + "MTPR1" +
            (policy == ModelParityMTP::DynamicDepth ? std::string("Dynamic") :
             "Fixed" + std::to_string(requestedDepth())) + "Retained" + std::to_string(retained_depth);
    }

    /**
     * @brief Resolve this explicit runtime admission input through the public configuration projection.
     * @param execution Explicit active or admitted-but-inactive paired source.
     * @param purpose Admission proof, explicit production HTTP qualification, or production workload tuning.
     * @return Full48/context4096/R1 configuration; no source or resource substitution.
     * @throws std::invalid_argument For a foreign mode or declaration without explicit hosted storage.
     */
    OrchestrationConfig config(Qwen4ExpMTPAdmissionExecution execution,
        Qwen4ExpMTPAdmissionPurpose purpose = Qwen4ExpMTPAdmissionPurpose::AdmissionDiagnostic) const
    {
        if (execution != Qwen4ExpMTPAdmissionExecution::Active &&
            execution != Qwen4ExpMTPAdmissionExecution::Inactive)
            throw std::invalid_argument("Unknown QFN MTP admission execution intent");
        if (!main.ngram_table_policy ||
            (main.ngram_table_policy->storage != ImmutableRowStoragePreference::RAM &&
             main.ngram_table_policy->storage != ImmutableRowStoragePreference::MemoryMapped))
            throw std::invalid_argument("QFN MTP admission requires its exact RAM/mmap source policy");
        if (purpose != Qwen4ExpMTPAdmissionPurpose::AdmissionDiagnostic &&
            purpose != Qwen4ExpMTPAdmissionPurpose::RuntimeQualification &&
            purpose != Qwen4ExpMTPAdmissionPurpose::ProductionServing)
            throw std::invalid_argument("Unknown QFN runtime projection purpose");
        const int depth = requestedDepth();
        if (retained_depth < depth || retained_depth > kModelParityRequiredMaximumMTPDepth)
            throw std::invalid_argument("QFN admission depth exceeds its explicit retained envelope");
        auto result = main.makeOrchestrationConfig(main.model.model_path, 0);
        if (purpose != Qwen4ExpMTPAdmissionPurpose::AdmissionDiagnostic)
        {
            // Production retains declared residency semantics while retiring
            // short numerical-parity cache geometry and forced maintenance
            // cadence. Runtime defaults remain code-owned, independent of the
            // client and of this model's diagnostic token count.
            const auto mode = result.moe_rebalance.mode;
            const auto defaults = OrchestrationConfig::defaults();
            result.prefix_cache = defaults.prefix_cache;
            result.moe_rebalance = defaults.moe_rebalance;
            result.moe_rebalance.mode = mode;
        }
        result.shared_predictor = predictor;
        result.batch_size = 1;
        result.mtp_activation_policy = execution == Qwen4ExpMTPAdmissionExecution::Active
            ? MTPActivationPolicy::Enabled : MTPActivationPolicy::Disabled;
        result.mtp.enabled = execution == Qwen4ExpMTPAdmissionExecution::Active;
        result.mtp.draft_tokens = depth;
        result.mtp.graph_capacity_draft_tokens = result.mtp.enabled ? retained_depth : 0;
        // Production HTTP sampling owns the public stochastic-capable verifier,
        // including its greedy specialization; admission-only diagnostics stay greedy.
        if (purpose != Qwen4ExpMTPAdmissionPurpose::AdmissionDiagnostic)
            result.mtp.verify_mode = modelParityHttpMTPVerifyMode();
        result.mtp.max_request_batch = 1;
        result.mtp.terminal_head_policy = terminal_head_policy;
        result.mtp.depth_policy = MTPDepthPolicyConfig{};
        result.mtp.depth_policy.mode = policy == ModelParityMTP::DynamicDepth
            ? MTPDepthPolicyMode::Dynamic : MTPDepthPolicyMode::Fixed;
        if (policy == ModelParityMTP::DynamicDepth)
        {
            result.mtp.depth_policy.min_depth = 1;
            result.mtp.depth_policy.max_depth = retained_depth;
            // HTTP qualification explicitly starts deep to exercise retained
            // execution. The full workload delegates initial depth to the
            // native trained backend/model resolver with the same bounds.
            if (purpose != Qwen4ExpMTPAdmissionPurpose::ProductionServing)
                result.mtp.depth_policy.initial_depth = depth;
        }
        // The short admission proof alone inherits numerical-prompt geometry.
        if (purpose == Qwen4ExpMTPAdmissionPurpose::AdmissionDiagnostic)
            result.prefix_cache.block_size = productionParityPrefixRestoreProofBlockSize(main.model.token_ids.size());
        result.deterministic = purpose == Qwen4ExpMTPAdmissionPurpose::AdmissionDiagnostic;
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
/**
 * @return Original A1 topology/source with fixed1/2/3/15 and dynamic plus named fixed5/6 R1 runtime admission.
 * @note These runtime declarations do not enter mathematical parity discovery:
 *       the main reference has no sidecar gold. Model admission is still PMA-owned.
 */
inline const std::vector<Qwen4ExpMTPAdmissionCase> &qwen4ExpFullROCmTP4MTPAdmissionCases()
{
    static const auto cases = [] {
        std::vector<Qwen4ExpMTPAdmissionCase> result;
        // The measured cold serve needs about 146 seconds before ServerReady.
        // This runtime-only copy owns a bounded startup allowance; request
        // duration and the original numerical Off declaration stay unchanged.
        auto runtime_main = qwen4ExpFullROCmTP4ProductionCases().front();
        runtime_main.generation_workload = runtime_main.generation_workload.withReadiness(300);
        for (const auto policy : {ModelParityMTP::Depth1, ModelParityMTP::Depth2, ModelParityMTP::Depth3,
                                  ModelParityMTP::Depth15, ModelParityMTP::DynamicDepth,
                                  ModelParityMTP::Depth5, ModelParityMTP::Depth6})
            result.push_back({.main = runtime_main,
                              .policy = policy,
                              .retained_depth = kModelParityRequiredMaximumMTPDepth,
                              .terminal_head_policy = MTPTerminalHeadPolicy::VocabularySharded});
        return result;
    }();
    return cases;
}
} // namespace llaminar2::test::parity::qwen4exp
