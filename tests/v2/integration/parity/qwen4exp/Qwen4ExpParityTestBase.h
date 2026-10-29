/**
 * @file Qwen4ExpParityTestBase.h
 * @brief Exact Flash Next model binding and native row-storage path evidence.
 *
 * The canonical production campaign owns runner setup, complete checkpoint
 * CSVs, prefix restoration, capture certification, and expert movement. This
 * specialization supplies only the independent model schema/reference and
 * verifies authority-authored native row extents for the selected hosted or
 * fixed-head native cell. Missing terminal evidence never becomes zero traffic.
 * An explicit frozen-history descriptor may project only prompt/token/reference
 * fields before admission, retaining the canonical topology, prefix/path policy,
 * all48 numerical stages and every original CSV writer. History publication
 * follows the original artifact-directory reset and precedes model admission.
 */
#pragma once
#include "config/OrchestrationConfigParser.h"
#include "Qwen4ExpProductionParityCases.h"
#include "Qwen4ExpSourceAdmission.h"
#include "Qwen4ExpPrefillAcceptance.h"
#include "Qwen4ExpMTPExpertReference.h"
#include "Qwen4ExpLookupRecordEvidence.h"
#include "Qwen4ExpGPUAdaptiveLookupEvidence.h"
#include "Qwen4ExpGDNProjectionEvidence.h"
#include "loaders/AdaptiveImmutableLookupGeometry.h"
#include "execution/qwen4exp/ImmutableLookupExecutionBoundary.h"
#include "../qwen2/Qwen2ParityTestBase.h"
#include "models/qwen4exp/Qwen4ExpSchema.h"
#include "loaders/ImmutableRowTable.h"
#include "loaders/ResidentImmutableRowTable.h"
#include <array>
#include <limits>

namespace llaminar2::test::parity::qwen4exp
{
/** @return Concrete execution-tier label, never inferred from the requested preference. */
inline const char *ngramStorageLabel(ImmutableRowStorage storage)
{
    switch(storage)
    {
    case ImmutableRowStorage::RAM: return "ram";
    case ImmutableRowStorage::MemoryMapped: return "mmap";
    case ImmutableRowStorage::VRAM: return "vram";
    }
    throw std::invalid_argument("Unknown concrete ngram storage in parity evidence");
}
/** @return Whether a concrete admission honors this lossless user preference. */
inline bool ngramStorageHonors(ImmutableRowStoragePreference requested, ImmutableRowStorage actual)
{
    switch(requested)
    {
    case ImmutableRowStoragePreference::Automatic: return true;
    case ImmutableRowStoragePreference::RAM: return actual==ImmutableRowStorage::RAM;
    case ImmutableRowStoragePreference::MemoryMapped: return actual==ImmutableRowStorage::MemoryMapped;
    case ImmutableRowStoragePreference::VRAM: return actual==ImmutableRowStorage::VRAM;
    case ImmutableRowStoragePreference::HostAdaptiveRAMSSD:
    case ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD:
        throw std::invalid_argument("Adaptive parity evidence requires the resolved layout, not a whole concrete tier");
    }
    throw std::invalid_argument("Unknown requested ngram storage in parity evidence");
}
/** @brief Bind Flash Next without substituting an ordinary Qwen graph. */
template<class Derived>
class Qwen4ExpConfigDrivenParityTest : public qwen2::ConfigDrivenParityTest<Derived>
{
protected:
    using Base = qwen2::ConfigDrivenParityTest<Derived>;
    /**
     * @brief Apply only an explicit exact-cell frozen history before base admission.
     * @throws std::invalid_argument For foreign source/cell, invalid diagnostic
     *         depth, LM-only reference, missing stages or mismatched teachers.
     *
     * The canonical case still owns topology, arithmetic, every threshold and
     * the full stage list. Throwing here aborts SetUp before ParityTestBase can
     * admit a model or regenerate a different greedy trajectory implicitly.
     */
    void applyModelOverrides() override
    {
        Base::applyModelOverrides();
        const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
        const std::string cell = info ? std::string(info->test_suite_name()) + "." + info->name() : std::string{};
        auto replay = FrozenParityHistoryReplay::fromDriver(cell);
        if (!replay) return;
        // Read the same setup selectors that base SetUp will reload before
        // runner admission; no arithmetic environment or policy is changed.
        mutableDebugEnv().reload();
        const auto &original = static_cast<const Derived *>(this)->modelParityCase();
        accuracy::requireQFN(original.model.transformer_layers == 48 &&
                             Base::config_.mtp_expectation == ParityMTPExpectation::Disabled,
                             "Frozen history replay requires the original QFN main48 MTP-off cell");
        auto arguments = modelParityDiagnosticBenchmarkArguments(original);
        arguments.insert(arguments.begin(), "canonical-history-replay");
        std::vector<char *> argv;
        for (auto &argument : arguments) argv.push_back(argument.data());
        const auto canonical = OrchestrationConfigParser{}.parseArgs(static_cast<int>(argv.size()), argv.data());
        replay->requireCanonicalConfiguration(canonical);
        replay->configureCPUExecution();
        replay->requireCanonicalSource(Base::config_.model_path, Base::config_.shared_predictor,
                                       Base::config_.decode_steps);
        replay->applyObservationDepth(Base::config_.decode_steps, Base::config_.final_logit_mean_kl);
        (void)replay->servingPrefillCapture(); // Validate the declared family before any model admission.
        Base::config_.prompt = replay->prompt();
        Base::config_.token_ids = replay->promptIds();
        Base::config_.snapshot_dir = replay->referenceDirectory();
        Base::frozen_history_replay_ = std::move(replay);
        const auto metadata = std::filesystem::path(Base::config_.snapshot_dir) / "metadata.txt";
        const auto validation = Base::validateReferenceSnapshotMetadata(metadata, true);
        if (!validation.usable)
            throw std::invalid_argument("Frozen full-stage reference failed before model admission: " + validation.reason);
        Base::frozen_history_replay_->requireSnapshotInventory(
            Base::config_.graph_snapshot_policy.requiredSnapshotKeys(ParityForwardPhase::Prefill), -1);
        for (int step = 0; step < Base::config_.decode_steps; ++step)
            Base::frozen_history_replay_->requireSnapshotInventory(
                Base::config_.graph_snapshot_policy.requiredSnapshotKeys(ParityForwardPhase::Decode), step);
    }
    /**
     * @brief Publish the retained original history only after its output directory is fresh.
     * @throws std::exception For mismatched installed history or failed output close.
     *
     * All descriptor/config/reference checks remain in applyModelOverrides.
     * The base calls this once on the artifact owner before model admission and
     * shares failure with peers; no rank races to recreate or overwrite metadata.
     */
    void onParityArtifactDirectoryReady() override
    {
        Base::onParityArtifactDirectoryReady();
        if (!Base::frozen_history_replay_) return;
        Base::frozen_history_replay_->publishBinding(Base::getResultsDir() / "history_replay_binding.json",
            Base::config_.prompt, Base::config_.token_ids, Base::config_.snapshot_dir, Base::config_.decode_steps);
    }
    /**
     * @brief Retain actual production-bucket evidence at the original compared checkpoint.
     * @param phase Original published phase; decode leaves the prefill proof unchanged.
     * @param step Original diagnostic decode offset, negative for prefill.
     * @param layers Original all-layer comparison records, left untouched.
     * @details Missing/wrong native geometry fails the test without suppressing raw CSVs.
     */
    void observeComparedParityCheckpoint(ParityForwardPhase phase, int step,
                                        const std::vector<LayerStats> &layers) override
    {
        Base::observeComparedParityCheckpoint(phase, step, layers);
        if (!Base::frozen_history_replay_ || phase != ParityForwardPhase::Prefill) return;
        const bool native = Base::productionParityExecutionTopology() != ProductionParityExecutionTopology::CPUOnly;
        nlohmann::json evidence;
        try
        {
            evidence = Base::frozen_history_replay_->servingPrefillEvidence(
                PerfStatsCollector::snapshot({"forward_graph"}), native);
            evidence["cpu_execution"] = Base::frozen_history_replay_->cpuExecutionEvidence(
                PerfStatsCollector::snapshot({"kernel.cpu_moe_router_q8_worker_rows"}),
                FrozenParityHistoryReplay::observeCPUExecution());
        }
        catch (const std::exception &error)
        {
            // Retain the existing raw comparisons/CSVs even when geometry is
            // structurally wrong. The driver rejects this incomplete handoff.
            evidence = {{"schema", "llaminar.model-kld-history-prefill-geometry.v1"},
                        {"error", error.what()}};
            ADD_FAILURE() << "Frozen history physical prefill proof failed: " << error.what();
        }
        std::ofstream output(Base::ensureResultsDir() / "history_replay_prefill_geometry.json", std::ios::trunc);
        output << evidence.dump(2) << '\n';
        output.close();
        if (!output) throw std::runtime_error("Cannot close frozen-history physical prefill geometry");
    }

    /** @brief Require every original main decoder and embedding before prefix certification. */
    ::testing::AssertionResult acceptsAdditionalPrefillEvidence(const ParityTestSummary &summary) const override
    {
        if (static_cast<const Derived *>(this)->modelParityCase().model.transformer_layers != 48)
            return Base::acceptsAdditionalPrefillEvidence(summary);
        const bool final_logits = Base::config_.final_logit_mean_kl.has_value();
        const auto complete = acceptsQwen4ExpCompletePrefill(summary.layer_stats,
            final_logits ? summary.embedding_evidence_valid : summary.embedding_passed,
            final_logits ? summary.lm_head_evidence_valid : summary.lm_head_passed,
            final_logits ? Qwen4ExpMainEvidencePolicy::FiniteExactObservations : Qwen4ExpMainEvidencePolicy::StrictLayerNumerics);
        if (!complete || !final_logits) return complete;
        return Base::acceptsRequiredRawParityStages(summary.layer_stats,
            Base::parityGraphSnapshotPolicy(ParityForwardPhase::Prefill).requiredSnapshotKeys(ParityForwardPhase::Prefill));
    }

    /** @brief Apply the identical complete-decoder contract to combined TP evidence. */
    ::testing::AssertionResult acceptsAdditionalPrefillEvidence(const TPParityTestSummary &summary) const override
    {
        if (static_cast<const Derived *>(this)->modelParityCase().model.transformer_layers != 48)
            return Base::acceptsAdditionalPrefillEvidence(summary);
        const bool final_logits = Base::config_.final_logit_mean_kl.has_value();
        const auto complete = acceptsQwen4ExpCompletePrefill(summary.layer_stats,
            final_logits ? summary.embedding_evidence_valid : summary.embedding_result.passed,
            final_logits ? summary.lm_head_evidence_valid : summary.lm_head_passed,
            final_logits ? Qwen4ExpMainEvidencePolicy::FiniteExactObservations : Qwen4ExpMainEvidencePolicy::StrictLayerNumerics);
        if (!complete || !final_logits) return complete;
        return Base::acceptsRequiredRawParityStages(summary.layer_stats,
            Base::parityGraphSnapshotPolicy(ParityForwardPhase::Prefill).requiredSnapshotKeys(ParityForwardPhase::Prefill));
    }

    /** @brief Every requested main decode step retains all original finite/exact decoder observations. */
    ::testing::AssertionResult acceptsAdditionalDecodeEvidence(const DecodeParitySummary &summary) const override
    {
        if (!Base::config_.final_logit_mean_kl ||
            static_cast<const Derived *>(this)->modelParityCase().model.transformer_layers != 48)
            return Base::acceptsAdditionalDecodeEvidence(summary);
        for (const auto &step : summary.step_stats)
        {
            if (const auto evidence = acceptsQwen4ExpCompletePrefill(step.layer_stats, true, true,
                    Qwen4ExpMainEvidencePolicy::FiniteExactObservations); !evidence)
                return ::testing::AssertionFailure() << "QFN decode step " << step.step_idx << ": " << evidence.message();
            if (const auto required = Base::acceptsRequiredRawParityStages(step.layer_stats,
                    Base::parityGraphSnapshotPolicy(ParityForwardPhase::Decode).requiredSnapshotKeys(ParityForwardPhase::Decode)); !required)
                return required;
        }
        return ::testing::AssertionSuccess();
    }

    /** @brief Bind only exact main-QSA producers from the frozen canonical capture selection. */
    std::vector<GraphSnapshotSourceOutput> nativeSnapshotSourceOutputs(
        const std::vector<std::string> &keys) const override
    {
        std::vector<GraphSnapshotSourceOutput> selected;
        for (const auto &key : keys)
            if (auto source = GraphSnapshotSourceOutput::fromQwenSparseSemanticKey(key))
                selected.push_back(*source);
        return selected;
    }

    /** @return Main phase payloads follow actual retained serving lanes, independent of decode observation count. */
    ParitySnapshotPayloadProjection referenceSnapshotPayloadProjection() const override
    {
        return ParitySnapshotPayloadProjection::RetainedMainServingPhases;
    }

    /**
     * @brief Bind the canonical HF-input selected-expert equation to the original source48 pair.
     * @param prefix Exact independently authenticated canonical or consumed-input-qualified checkpoint.
     * @param expert_ids Validated sorted union of native/HF routed experts.
     * @param rows Scalar predictor row count, independent of retained graph depth/capacity.
     * @param hidden Original compute-width expert input/output extent.
     * @return Expert-only authority; canonical combined/HC/final/logit verdicts remain untouched.
     */
    MoEIndependentReference independentMoEMTPExpertReference(
        const std::string &prefix, const std::vector<int> &expert_ids,
        std::size_t rows, std::size_t hidden) override
    {
        if (!Base::config_.shared_predictor)
            throw std::runtime_error("QFN independent MTP expert reference requires its explicit original predictor");
        return loadQwen4ExpMTPExpertReference({
            .main = Base::config_.model_path,
            .predictor = Base::config_.shared_predictor->path(),
            .snapshots = Base::config_.snapshot_dir,
            .artifacts = Base::ensureResultsDir(),
            .checkpoint_prefix = prefix}, expert_ids, rows, hidden);
    }

    /** @return All routed multi-device modes use the canonical ExpertOverlay authority. */
    bool requiresUniversalMoEAuthority() const override { return true; }

    /** @brief Install the exact architecture's immutable sharding policy before preparation. */
    void configureModel(std::shared_ptr<ModelContext> context) override
    {
        Qwen4ExpSchemaFactory schema;
        context->weightManager()->setWeightShardingConfig(schema.getWeightShardingConfig());
    }

    /**
     * @brief Schema1 consumes original GGUF rows and captures their native tiled value-head order.
     * @return Native ordering authenticated by the schema/converter pins below.
     */
    ParityGDNReferenceLayout referenceGDNSnapshotLayout() const override
    {
        return ParityGDNReferenceLayout::NativeGGUF;
    }

    /** @brief Authenticate pinned equations and integer/fusion contracts in the canonical pack. */
    typename Base::ReferenceSnapshotValidation validateModelSpecificReferenceSnapshotMetadata(
        const std::filesystem::path &path) const override
    {
        const std::array<std::pair<const char *, const char *>, 8> fields{{
            {"qwen4exp_reference_schema", "1"},
            {"qwen4exp_checkpoint_revision", "de4b8e4d43b917e7706784d8bb445c9af86a3540"},
            {"qwen4exp_transformers_revision", "28034545c35acac05793695bd5c878f66e6b544e"},
            {"qwen4exp_converter_revision", "1b70350c92563449576bcbbf552994b1d1ed847d"},
            {"qwen4exp_integer_checkpoint_dtype", "int64"},
            {"qwen4exp_gdn_output_gate", "sigmoid"},
            {"qwen4exp_residual_width", "10240"},
            {"qwen4exp_terminal_head_width", "2560"},
        }};
        for (const auto &[name, expected] : fields)
            if (Base::readSnapshotMetadataValue(path, name) != std::optional<std::string>(expected))
                return {false, std::string("Flash Next reference contract mismatch: ") + name};
        if (Base::config_.mtp_expectation == ParityMTPExpectation::Disabled)
            return {true, {}};
        if (!Base::config_.shared_predictor || Base::config_.mtp_expected_graph_capacity <= 0)
            return {false, "QFN MTP reference requires its explicit original predictor and retained depth"};
        const std::filesystem::path predictor(Base::config_.shared_predictor->path());
        std::error_code error;
        const auto bytes = std::filesystem::file_size(predictor, error);
        if (error || bytes == 0)
            return {false, "QFN MTP reference predictor source is unavailable"};
        const std::array<std::pair<std::string, std::string>, 9> mtp_fields{{
            {"qwen4exp_mtp_reference_schema", "1"},
            {"qwen4exp_mtp_source_layer", "48"},
            {"qwen4exp_mtp_cache_layer", "0"},
            {"qwen4exp_mtp_attention", "dense0"},
            {"qwen4exp_mtp_hidden_norm", "per_branch"},
            {"qwen4exp_mtp_shifted_origin", "visible_zero_hidden_token_zero"},
            {"qwen4exp_mtp_recursive_width", "10240"},
            {"qwen4exp_mtp_predictor_filename", predictor.filename().string()},
            {"qwen4exp_mtp_predictor_size_bytes", std::to_string(bytes)},
        }};
        for (const auto &[name, expected] : mtp_fields)
            if (Base::readSnapshotMetadataValue(path, name) != std::optional<std::string>(expected))
                return {false, "QFN paired MTP reference mismatch: " + name};
        const auto steps_text = Base::readSnapshotMetadataValue(path, "qwen4exp_mtp_decode_steps");
        try
        {
            std::size_t consumed = 0;
            const int steps = steps_text ? std::stoi(*steps_text, &consumed) : 0;
            if (!steps_text || consumed != steps_text->size() || steps < Base::config_.decode_steps)
                return {false, "QFN paired MTP reference has insufficient canonical trajectory"};
        }
        catch (const std::exception &)
        {
            return {false, "QFN paired MTP reference has invalid canonical trajectory"};
        }
        const auto depth_text = Base::readSnapshotMetadataValue(path, "mtp_sidecar_max_draft_depth");
        int depth = 0;
        try
        {
            std::size_t consumed = 0;
            depth = depth_text ? std::stoi(*depth_text, &consumed) : 0;
            if (!depth_text || consumed != depth_text->size()) depth = 0;
        }
        catch (const std::exception &) { depth = 0; }
        if (depth < Base::config_.mtp_expected_graph_capacity || depth > 15)
            return {false, "QFN paired MTP reference has insufficient original depth capacity"};
        for (const auto &name : {std::string("decode_step0_MTP0_EMBEDDING.npy"),
                 "decode_step0_MTP" + std::to_string(depth - 1) + "_LM_HEAD.npy"})
            if (!std::filesystem::is_regular_file(path.parent_path() / name))
                return {false, "QFN paired MTP reference is missing " + name};
        return {true, {}};
    }

    /** @brief Regenerate through the shared exact-GGUF CPU/FP32 reference entrypoint. */
    bool regeneratePyTorchSnapshots() override
    {
        if (Base::frozen_history_replay_)
        {
            LOG_ERROR("[Parity] Frozen full-stage history must be regenerated explicitly before admission; refusing implicit alternate trajectory");
            return false;
        }
        // The test process intentionally limits its own OpenMP team. Reference
        // generation has a separate CPU workload and must not inherit that limit.
        std::ostringstream script;
        script << "unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS OMP_PROC_BIND OMP_PLACES KMP_AFFINITY; "
               << "if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi; "
               << "python3 python/reference/generate_qwen4exp_pipeline_snapshots.py --model "
               << Base::parityShellQuote(Base::config_.model_path)
               << " --prompt " << Base::parityShellQuote(Base::config_.prompt)
               << " --output " << Base::parityShellQuote(Base::config_.snapshot_dir)
               << " --decode-steps " << Base::config_.decode_steps;
        if (Base::config_.mtp_expectation != ParityMTPExpectation::Disabled)
        {
            if (!Base::config_.shared_predictor || Base::config_.mtp_expected_graph_capacity <= 0)
                return false;
            script << " --shared-predictor " << Base::parityShellQuote(Base::config_.shared_predictor->path())
                   << " --mtp-sidecar-snapshots --mtp-max-draft-depth " << Base::config_.mtp_expected_graph_capacity;
            const auto metadata = std::filesystem::path(Base::config_.snapshot_dir) / "metadata.txt";
            if (std::filesystem::is_regular_file(metadata) &&
                Base::validateReferenceSnapshotMetadata(metadata, Base::productionParityCampaignEnabled(), false).usable)
                script << " --mtp-sidecar-only";
        }
        const auto command = "bash -c " + Base::parityShellQuote(script.str()) + " 2>&1";
        FILE *pipe = popen(command.c_str(), "r");
        if (!pipe) return false;
        char buffer[512];
        std::string output;
        while (fgets(buffer, sizeof(buffer), pipe)) output += buffer;
        const int status = pclose(pipe);
        if (status != 0) LOG_ERROR("[Parity] Flash Next reference generation failed:\n" << output);
        return status == 0;
    }

    /**
     * @brief Extend the authenticated QFN pack for one actual recursive branch.
     * @param reference_step Original canonical main decode step.
     * @param condition_tokens Exact recursive draft inputs consumed after MTP0.
     * @param condition_token_override Actual noncanonical MTP0 input, when present.
     * @return True only after the existing generator atomically publishes the additive branch.
     *
     * The shared framework owns generation leasing, freshness validation and
     * numerical comparison. This hook supplies only model-specific native
     * source/shape/origin binding; it does not execute another main trajectory.
     */
    bool regeneratePyTorchMTPBranchSnapshots(int reference_step,
        const std::vector<int32_t> &condition_tokens,
        std::optional<int32_t> condition_token_override) override
    {
        if (!Base::config_.shared_predictor || reference_step < 0 ||
            reference_step >= Base::config_.decode_steps ||
            (condition_tokens.empty() && !condition_token_override) || condition_tokens.size() >= 15 ||
            (condition_token_override && *condition_token_override < 0) ||
            std::any_of(condition_tokens.begin(), condition_tokens.end(), [](int32_t token) { return token < 0; }))
            return false;
        const auto directory = Base::ensureResultsDir();
        const auto request_path = directory / "mtp_hf_branch_request.json";
        std::ofstream request(request_path, std::ios::trunc);
        if (!request) return false;
        request << "{\"" << reference_step << "\": {\"condition_token\": ";
        if (condition_token_override) request << *condition_token_override;
        else request << "null";
        request << ", \"draft_tokens\": [";
        for (std::size_t index = 0; index < condition_tokens.size(); ++index)
        {
            if (index) request << ", ";
            request << condition_tokens[index];
        }
        request << "]}}\n";
        request.flush();
        if (!request.good()) return false;
        request.close();
        std::ostringstream script;
        script << "unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS OMP_PROC_BIND OMP_PLACES KMP_AFFINITY; "
               << "if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi; "
               << "python3 python/reference/generate_qwen4exp_pipeline_snapshots.py --model "
               << Base::parityShellQuote(Base::config_.model_path)
               << " --shared-predictor " << Base::parityShellQuote(Base::config_.shared_predictor->path())
               << " --prompt " << Base::parityShellQuote(Base::config_.prompt)
               << " --output " << Base::parityShellQuote(Base::config_.snapshot_dir)
               << " --decode-steps " << Base::config_.decode_steps
               << " --mtp-sidecar-snapshots --mtp-sidecar-only --mtp-max-draft-depth "
               << condition_tokens.size() + 1 << " --mtp-branch-overrides "
               << Base::parityShellQuote(request_path.string());
        const auto command = "bash -c " + Base::parityShellQuote(script.str()) + " 2>&1";
        FILE *pipe = popen(command.c_str(), "r");
        if (!pipe) return false;
        std::string output;
        std::array<char, 512> buffer{};
        while (fgets(buffer.data(), buffer.size(), pipe)) output += buffer.data();
        const int status = pclose(pipe);
        std::ofstream log(directory / "mtp_hf_branch_generation.log", std::ios::trunc);
        log << output;
        log.flush();
        if (status != 0) LOG_ERROR("[Parity] QFN additive MTP reference failed:\n" << output);
        return status == 0 && log.good();
    }

    /**
     * @brief Prove the one actual mixed source/root/PMA owner after complete runner initialization.
     *
     * Observations come from retained live leases. Geometry is used only as an
     * independent expected extent, never as live accounting or allocation
     * authority. No gather, policy touch, payload read or GPU state probe occurs.
     */
    void assertGPUAdaptiveNativeOwners()
    {
        // The setup-only and numerical fixtures consume the same retained
        // native owners/PMA attestation; neither reconstructs a live ledger.
        ASSERT_NO_FATAL_FAILURE(expectQwen4ExpSourceGPUAdaptiveOwners(
            static_cast<const Derived *>(this)->modelParityCase(), *Base::orch_runner_));
    }

    /**
     * @brief Retain exact native submission/capture routes at the completed cell boundary.
     * @details Raw records are exported before selected-policy validation, including
     * numerical-red cells. No global /tmp export or fabricated replay count is used.
     * Existing adaptive reporting-window reset preserves this unrelated domain.
     */
    void exportGDNProjectionRouteEvidence()
    {
        if (!Base::productionParityCampaignEnabled() || !Base::isRank0()) return;
        const auto records = PerfStatsCollector::snapshot({kGDNProjectionRouteFilter});
        EXPECT_TRUE(PerfStatsCollector::writeCsv(
            (Base::ensureResultsDir() / "gdn_projection_routes.csv").string(), {kGDNProjectionRouteFilter}))
            << "Cannot retain original GDN projection route evidence";
        const auto &test_case = static_cast<const Derived *>(this)->modelParityCase();
        if (test_case.model.transformer_layers != 48 ||
            debugEnv().gemm.qfn_gdn_qkv_activation != QuantizedActivationOperandPrecision::TwoTermQ8)
            return; // Default arithmetic has no new acceptance requirement.
        const auto *context = Base::activeModelContextForDiagnostics();
        ASSERT_NE(context, nullptr);
        ASSERT_NO_THROW(requireQwen4ExpGDNQKVRouteEvidence(records, context->model()));
    }

    /**
     * @brief Retain all canonical numerical gates and prove exact native storage traffic.
     *
     * CPU completion records describe synchronous native lookup. GPU records
     * are published only after the captured terminal event and mapped-worker
     * join. Neither path infers an executed mode from the requested CLI policy.
     */
    void runQwen4ExpProductionParityCampaign()
    {
        Base::runProductionParityCampaign();
        exportGDNProjectionRouteEvidence();
        if (::testing::Test::HasFatalFailure()) return;
        const auto &test_case = static_cast<const Derived *>(this)->modelParityCase();
        ASSERT_TRUE(test_case.ngram_table_policy.has_value());
        const bool cpu = test_case.topology.kind == ModelParityTopologyKind::SingleDevice;
        const std::string transport = cpu ? "cpu_direct" : "mapped_channel";
        const auto *context = Base::activeModelContextForDiagnostics();
        ASSERT_NE(context, nullptr);
        const auto placement = context->resolvedImmutableRowTablePlacement();
        ASSERT_NE(placement, nullptr);
        placement->requireRequestedPolicy(*test_case.ngram_table_policy);
        if(placement->isGPUAdaptive())
        {
            ASSERT_EQ(test_case.mtp, ModelParityMTP::Off);
            ASSERT_FALSE(Base::orch_runner_->config().mtp.enabled);
            // The partial-prefix proof intentionally ends at a standalone
            // suffix-prefill boundary. Retire it through the actual public
            // request reset before asking for idle retained-source credit.
            // Nonfatal HF numerical failures do not waive this lifecycle duty.
            ASSERT_NO_THROW(Base::activeClearCache());
            ASSERT_NO_FATAL_FAILURE(assertGPUAdaptiveNativeOwners());
            const auto boundary=context->adaptiveImmutableLookupBoundary(
                placement->gpuAdaptiveLayout().consumers().front().device);
            ASSERT_TRUE(boundary);
            const auto source_before=boundary->retainedSourceOwnerObservation();
            const auto provider_before=context->hostImmutableRowProvider();
            const auto roots_before=boundary->adaptiveRoots();
            const auto identity_before=boundary->captureIdentity();
            const auto original_path=Base::ensureResultsDir()/"production_path.csv";
            ASSERT_TRUE(std::filesystem::copy_file(original_path,
                Base::ensureResultsDir()/"production_path_forward.csv",
                std::filesystem::copy_options::overwrite_existing));
            const auto forward=PerfStatsCollector::snapshot({"immutable_lookup"});
            std::ofstream forward_csv(Base::ensureResultsDir()/"ngram_storage_forward.csv");
            ASSERT_TRUE(forward_csv.is_open());
            writeQwen4ExpLookupCounterCSV(forward_csv,forward);forward_csv.flush();ASSERT_TRUE(forward_csv.good());
            (void)beginQwen4ExpGPUAdaptiveGenerationEvidenceWindow();

            // Forced numerical rows never certify the captured generation
            // controller. Use the existing public ordinary request path on
            // this same runner, original prompt and installed sampling law.
            // Two outputs require one real model forward after the initial
            // sample; an early EOS remains a failed witness, never suppressed.
            const std::vector<int32_t> prompt(Base::config_.token_ids.begin(),Base::config_.token_ids.end());
            const auto generation=Base::orch_runner_->generate(
                prompt,2,Base::productionParitySamplingParams());
            std::ofstream generation_record(Base::ensureResultsDir()/"ngram_generation.txt");
            ASSERT_TRUE(generation_record.is_open());
            generation_record<<"response_budget=2\nreturned_tokens="<<generation.tokens.size()
                <<"\ncomplete="<<generation.is_complete<<"\nerror="<<generation.error<<"\ntoken_ids=";
            for(const auto token:generation.tokens)generation_record<<token<<' ';
            generation_record<<'\n';generation_record.flush();ASSERT_TRUE(generation_record.good());
            const auto records=PerfStatsCollector::snapshot({"immutable_lookup"});
            std::ofstream csv(Base::ensureResultsDir()/"ngram_storage.csv");ASSERT_TRUE(csv.is_open());
            writeQwen4ExpLookupCounterCSV(csv,records);csv.flush();ASSERT_TRUE(csv.good());
            ASSERT_TRUE(generation.success())<<generation.error;
            EXPECT_EQ(generation.tokens.size(),2u)<<"Adaptive terminal witness ended before one complete ordinary model forward";
            ASSERT_FALSE(boundary->hasActiveInterval());
            const auto source_after=boundary->retainedSourceOwnerObservation();
            EXPECT_EQ(source_after.authority().get(),source_before.authority().get());
            EXPECT_EQ(source_after.allocator(),source_before.allocator());
            EXPECT_EQ(source_after.hostPlacement(),source_before.hostPlacement());
            EXPECT_EQ(source_after.tableStorageBytes(),source_before.tableStorageBytes());
            EXPECT_EQ(source_after.lookupStagingBytes(),source_before.lookupStagingBytes());
            EXPECT_EQ(boundary->adaptiveRoots().cache.get(),roots_before.cache.get());
            EXPECT_EQ(boundary->adaptiveRoots().frame.get(),roots_before.frame.get());
            EXPECT_EQ(boundary->adaptiveRoots().protocol.get(),roots_before.protocol.get());
            EXPECT_EQ(boundary->captureIdentity(),identity_before);
            ASSERT_EQ(Base::activeModelContextForDiagnostics(),context);
            EXPECT_EQ(context->adaptiveImmutableLookupBoundary(boundary->device()).get(),boundary.get());
            EXPECT_EQ(context->hostImmutableRowProvider().get(),provider_before.get());
            ASSERT_NO_FATAL_FAILURE(assertGPUAdaptiveNativeOwners());
            const auto &layout=placement->gpuAdaptiveLayout();
            const auto provider=context->hostImmutableRowProvider();ASSERT_TRUE(provider);
            const auto geometry=AdaptiveImmutableLookupGeometry::fromLayout(layout,provider->source(),placement->gpuAdaptiveRequest());
            ASSERT_NO_THROW((void)requireQwen4ExpGPUAdaptiveLookupEvidence(records,layout.consumers().front().device,
                placement->extent().rowBytes(),placement->gpuAdaptiveMaximumLiveRows(),geometry.gpu().slots(),geometry.host().slots()));
            // Refresh from actual work while preserving every inherited raw
            // numerical/prefix result. No ModelContext reuse claim is inferred.
            Base::finishProductionParityEvidence();
            exportGDNProjectionRouteEvidence();
            return; // Every inherited numerical/prefix/capture gate already ran above.
        }
        if(placement->isHostAdaptive())
        {
            EXPECT_TRUE(test_case.ngram_table_policy->isHostAdaptive());
            const auto &request=placement->hostAdaptiveRequest();const auto slots=placement->hostAdaptiveLayout().owners().front().cache_rows;
            EXPECT_LE(slots,request.maximum_cache_rows);
            if(!request.maximum_cache_rows)EXPECT_EQ(slots,0u);
            else if(request.cache_physical_byte_limit)EXPECT_GT(slots,0u)
                << "Positive-cache model qualification requires actual admitted RAM slots";
        }
        else EXPECT_TRUE(ngramStorageHonors(test_case.ngram_table_policy->storage,placement->storage()));
        std::vector<DeviceId> lookup_consumers;
        for (const auto &consumer : placement->consumers()) lookup_consumers.push_back(consumer.device);
        const auto storage = placement->isHostAdaptive()?std::string(hostImmutableRowModeName(HostImmutableRowMode::AdaptiveRAMSSD)):
            ngramStorageLabel(placement->storage());
        if (placement->isFixedHeadShards())
        {
            ASSERT_TRUE(test_case.ngram_table_policy->isFixedHeadSharded());
            ASSERT_EQ(context->hostImmutableRowProvider(), nullptr);
            std::vector<DeviceId> owners;
            for (const auto &owner : placement->allocators())
            {
                const auto native = context->residentImmutableRowTable(owner.device);
                ASSERT_NE(native, nullptr);
                EXPECT_EQ(native->sourceRange(), placement->fixedHeadSourceRangeFor(owner));
                EXPECT_EQ(native->nativeBytes(), native->sourceRange().nativeBytes());
                owners.push_back(owner.device);
            }
            const auto records = PerfStatsCollector::snapshot({"immutable_lookup"});
            std::ofstream csv(Base::ensureResultsDir() / "ngram_storage.csv");
            ASSERT_TRUE(csv.is_open());
            writeQwen4ExpLookupCounterCSV(csv, records);
            csv.flush();
            ASSERT_TRUE(csv.good());
            for (const auto &record : records)
            {
                if (record.kind == PerfStatRecord::Kind::Counter)
                {
                    ASSERT_NO_THROW(classifyQwen4ExpLookupCounter(record, lookup_consumers, std::nullopt));
                }
            }
            ASSERT_NO_THROW(requireQwen4ExpFixedHeadLookupEvidence(records, owners, placement->extent().rowBytes()));
            return;
        }
        const auto table = context->hostImmutableRowProvider();
        ASSERT_NE(table, nullptr) << "Hosted numerical cell requires its concrete native table owner";
        if(placement->isHostAdaptive())EXPECT_EQ(table->mode(),HostImmutableRowMode::AdaptiveRAMSSD);
        else EXPECT_EQ(table->wholeStorage(),placement->storage());
        const auto records = PerfStatsCollector::snapshot({"immutable_lookup"});
        std::ofstream csv(Base::ensureResultsDir() / "ngram_storage.csv");
        ASSERT_TRUE(csv.is_open());
        writeQwen4ExpLookupCounterCSV(csv, records);
        csv.flush();
        ASSERT_TRUE(csv.good());
        std::map<std::string, double> totals;
        bool observed = false;
        for (const auto &record : records)
        {
            if (record.kind != PerfStatRecord::Kind::Counter) continue;
            std::optional<Qwen4ExpLookupCounterRole> role;
            ASSERT_NO_THROW(role = classifyQwen4ExpLookupCounter(record, lookup_consumers, table->mode()));
            ASSERT_TRUE(role.has_value());
            if (*role == Qwen4ExpLookupCounterRole::RequestResetLifecycle) continue;
            const auto mode = record.tags.find("storage");
            const auto wire = record.tags.find("transport");
            EXPECT_EQ(mode->second, storage);
            if (wire->second != transport) continue;
            EXPECT_EQ(record.phase, cpu ? "lookup_completion" : "generation_terminal");
            observed = true;
            totals[record.name] += record.value;
        }
        ASSERT_TRUE(observed) << "No native immutable row lookup evidence for " << transport;
        EXPECT_GT(totals["requests"], 0);
        EXPECT_GT(totals["ids"], 0);
        EXPECT_EQ(totals["native_bytes"], totals["ids"] * table->source().rowBytes());
        EXPECT_EQ(totals["failures"], 0);
        if (!cpu) EXPECT_EQ(totals["id_bytes"], totals["ids"] * sizeof(std::int64_t));
        if(placement->isHostAdaptive())
        {
            EXPECT_GT(totals["adaptive_batches"],0);
            EXPECT_EQ(totals["ram_rows"]+totals["source_rows"]+totals["duplicate_rows"],totals["ids"]);
            if(!placement->hostAdaptiveLayout().owners().front().cache_rows)
            {
                EXPECT_EQ(totals["ram_rows"],0);EXPECT_EQ(totals["cache_admissions"],0);
            }
        }
    }
};
} // namespace llaminar2::test::parity::qwen4exp
