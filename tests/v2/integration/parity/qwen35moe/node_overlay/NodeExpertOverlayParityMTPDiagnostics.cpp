/**
 * @file NodeExpertOverlayParityMTPDiagnostics.cpp
 * @brief MTPDiagnostics implementation of the shared node-overlay parity fixture.
 *
 * Shared by the 35B and 122B generated node ExpertOverlay matrices. This
 * translation-unit boundary does not change request ownership, reference
 * mathematics, graph execution, or evidence publication ordering.
 */
#include "NodeExpertOverlayParityFixture.h"

namespace llaminar2::test::parity::qwen35moe::node_overlay
{

    /** @return Whether @p key is a grouped main-verifier checkpoint. */
    auto Qwen35MoENodeExpertOverlayParityTest::isMTPMainVerifierDiagnosticKey(std::string_view key) -> bool
    {
        static constexpr std::array<std::string_view, 21> kStageSuffixes = {
            "ATTENTION_NORM",
            "QKV_PROJECTION",
            "Q_PROJECTION",
            "GDN_Z_PROJECTION",
            "ATTENTION_CONTEXT",
            "ATTENTION_OUTPUT",
            "ATTENTION_OUTPUT_ALLREDUCED",
            "FFN_NORM_RESIDUAL_OUT",
            "FFN_NORM",
            "MOE_ROUTER_OUTPUT",
            "MOE_ROUTING_INDICES",
            "MOE_ROUTING_WEIGHTS",
            "MOE_EXPERT_OUTPUT",
            "MOE_SHARED_EXPERT_OUTPUT",
            "MOE_SHARED_GATE_OUTPUT",
            "MOE_COMBINED_OUTPUT",
            "FFN_RESIDUAL",
            "GDN_CONV1D_OUTPUT",
            "GDN_DELTA_RULE_OUTPUT",
            "GDN_NORM_GATE_OUTPUT",
            "GDN_OUTPUT",
        };
        if (key == "EMBEDDING" || key == "FINAL_NORM" ||
            key == "LM_HEAD" || key == "LM_HEAD_ROWS_SELECT")
        {
            return true;
        }
        if (!key.starts_with("layer"))
            return false;
        return std::any_of(
            kStageSuffixes.begin(),
            kStageSuffixes.end(),
            [key](std::string_view suffix) { return key.ends_with(suffix); });
    }

    /**
     * @brief Copy the bounded main-verifier diagnostic surface now live.
     * @return Keyed production snapshots, excluding unrelated graph outputs.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::captureMTPMainVerifierDiagnostics() const -> std::map<std::string, std::vector<float>>
    {
        std::map<std::string, std::vector<float>> snapshots;
        for (const std::string &key : activeSnapshotKeys())
        {
            if (!isMTPMainVerifierDiagnosticKey(key) ||
                key == "LM_HEAD_ROWS_SELECT")
            {
                continue;
            }
            size_t elements = 0u;
            const float *const data = activeSnapshot(key, elements);
            if (data && elements > 0u)
            {
                snapshots.emplace(
                    key, std::vector<float>(data, data + elements));
            }
        }
        return snapshots;
    }

    /** @return Semantic stage suffix from a layer-qualified snapshot key. */
    auto Qwen35MoENodeExpertOverlayParityTest::mtpMainVerifierStage(std::string_view key) -> std::string
    {
        if (!key.starts_with("layer"))
            return std::string(key);
        const size_t delimiter = key.find('_');
        return delimiter == std::string_view::npos
                   ? std::string(key)
                   : std::string(key.substr(delimiter + 1u));
    }

    /** @return Whether all tensor statistics prove finite input and output. */
    auto Qwen35MoENodeExpertOverlayParityTest::mtpComparisonIsFinite(const StageComparisonResult &comparison) -> bool
    {
        return comparison.llaminar_stats.nan_count == 0u &&
               comparison.llaminar_stats.inf_count == 0u &&
               comparison.pytorch_stats.nan_count == 0u &&
               comparison.pytorch_stats.inf_count == 0u;
    }

    /**
     * @brief Preserve full values only for a failed numerical diagnostic.
     *
     * Successful cells retain compact scalar rows.  Failure-only copying keeps
     * the artifact as actionable as the historical long-horizon proof without
     * making every matrix cell duplicate hundreds of megabytes of tensors.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::retainMTPFailureValues(
        const MTPNumericalDiagnostic &diagnostic,
        std::span<const float> production,
        std::span<const float> reference) -> void
    {
        if (diagnostic.passed)
            return;
        ASSERT_EQ(production.size(), reference.size());
        mtp_failure_diagnostics_.push_back(MTPFailureDiagnostic{
            .call = diagnostic.call,
            .reference_step = diagnostic.reference_step,
            .reference_depth = diagnostic.reference_depth,
            .stage = diagnostic.stage,
            .production = std::vector<float>(
                production.begin(), production.end()),
            .reference = std::vector<float>(
                reference.begin(), reference.end()),
        });
    }

    /**
     * @brief Observe an ordinary main-model routed checkpoint.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::observeComparedParityCheckpoint(
        ParityForwardPhase phase,
        int step,
        const std::vector<LayerStats> &layers) -> void
    {
        observeComparedRoutedExpertCheckpoint(
            RoutedExpertCheckpointContext{
                .phase = phase,
                .step = step,
                .mtp = std::nullopt,
            },
            layers);
    }

    /**
     * @brief Retain one serial main-model row under its exact placement epoch.
     *
     * A later live-oracle extension may revisit the same logical row after an
     * ExpertOverlay epoch change. In that case the newer row replaces the old
     * diagnostic so grouped row zero is never compared with a stale physical
     * placement. Re-observing the same epoch must carry the same token identity.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::retainMTPSerialVerifierDiagnostic(
        const ProductionParityMTPSerialOracleBoundary &boundary) -> void
    {
        ASSERT_TRUE(boundary.valid());
        const auto placement = pinnedDevicePlacementEpochEvidence();
        ASSERT_TRUE(placement.has_value())
            << "Serial MTP oracle omitted device-authenticated placement";
        if (!placement.has_value())
            return;

        if (mtp_serial_verifier_diagnostic_.has_value() &&
            mtp_serial_verifier_diagnostic_->execution_epoch == placement->epoch)
        {
            EXPECT_EQ(
                mtp_serial_verifier_diagnostic_->boundary.output_index,
                boundary.output_index);
            EXPECT_EQ(
                mtp_serial_verifier_diagnostic_->boundary.token,
                boundary.token);
            return;
        }

        mtp_serial_verifier_diagnostic_ = MTPSerialVerifierDiagnostic{
            .boundary = boundary,
            .snapshots = captureMTPMainVerifierDiagnostics(),
            .execution_epoch = placement->epoch,
        };
        ASSERT_FALSE(mtp_serial_verifier_diagnostic_->snapshots.empty())
            << "Serial MTP oracle exposed no main-model diagnostic snapshots";
    }

    /**
     * @brief Reuse the compared captured M=1 row before placement maintenance.
     *
     * GPU fixed-depth cells need no serial replay: ordinary parity already
     * executed and compared the exact row selected for grouped verification.
     * Dynamic-depth and CPU cells can require a longer oracle than the compact
     * checkpoint corpus, so their explicit live-oracle callback remains the
     * authority.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::observeProductionParityDecodeBoundary(
        const ProductionParityDecodeBoundary &boundary) -> void
    {
        if (!isQwen122ProductionTest() || !activeMTPEnabled() ||
            !isRootParityRank() || !activePrimaryDevice().is_gpu() ||
            config_.mtp_expectation == ParityMTPExpectation::DynamicDepth)
        {
            return;
        }

        const auto accepted = productionParityMTPAcceptedDraftReference();
        if (!accepted.has_value() ||
            accepted->reference_step != boundary.reference_step)
        {
            return;
        }

        const auto &boundaries = productionParityDecodeBoundaries();
        ASSERT_FALSE(boundaries.empty());
        ASSERT_EQ(&boundaries.back(), &boundary)
            << "Decode observer did not receive the newly published boundary";
        const PrefixRuntimeStateSnapshot *before = nullptr;
        if (boundaries.size() == 1u)
        {
            ASSERT_TRUE(productionParityDecodePrefillState().has_value());
            before = &*productionParityDecodePrefillState();
        }
        else
        {
            before = &boundaries[boundaries.size() - 2u].runtime_state;
        }

        const ProductionParityMTPSerialOracleBoundary serial_boundary{
            .output_index = boundary.reference_step,
            .token = boundary.committed_token,
            .checkpoint_reference = true,
            .before = *before,
            .after = boundary.runtime_state,
        };
        retainMTPSerialVerifierDiagnostic(serial_boundary);
    }

    /**
     * @brief Observe the primary sidecar bank under its exact live namespace.
     *
     * A deeper transaction overwrites one reusable chained bank repeatedly.
     * Its terminal numerical row remains certified by the generic MTP gate,
     * while moved-expert provenance is attributed once from the primary bank
     * whose input lineage is canonical and independently named.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::observeComparedMTPParityCheckpoint(
        const ComparedMTPParityCheckpoint &checkpoint,
        const std::vector<StageComparisonResult> &stages) -> void
    {
        ASSERT_TRUE(checkpoint.valid());
        mtp_checkpoint_diagnostics_.push_back(MTPCheckpointDiagnostic{
            .checkpoint = checkpoint,
            .stages = stages,
        });

        for (const StageComparisonResult &comparison : stages)
        {
            const size_t separator = comparison.stage_name.find('_');
            ASSERT_NE(separator, std::string::npos)
                << "MTP comparison has no semantic stage suffix: "
                << comparison.stage_name;
            const std::string suffix =
                comparison.stage_name.substr(separator + 1u);
            std::string production_key;
            if (suffix == "TERMINAL_HIDDEN_ROW_SELECT")
            {
                production_key = checkpoint.production_stage_prefix;
                const std::string_view sidecar_prefix = "MTP0_";
                ASSERT_TRUE(production_key.ends_with(sidecar_prefix));
                production_key.resize(
                    production_key.size() - sidecar_prefix.size());
                production_key += "MTP_TERMINAL_HIDDEN_ROW_SELECT";
            }
            else
            {
                production_key =
                    checkpoint.production_stage_prefix + suffix;
            }
            const std::string reference_key =
                checkpoint.reference_stage_prefix + suffix;

            if (suffix == "LM_HEAD" &&
                checkpoint.identity.role == MTPParityCheckpointRole::Primary)
            {
                size_t production_elements = 0u;
                const float *const production =
                    activeSnapshot(production_key, production_elements);
                const std::vector<float> reference =
                    loadPyTorchSnapshot(reference_key);
                ASSERT_NE(production, nullptr);
                ASSERT_EQ(production_elements, reference.size());
                mtp_primary_production_top1_ = static_cast<int>(
                    std::distance(
                        production,
                        std::max_element(
                            production,
                            production + production_elements)));
                mtp_primary_reference_top1_ = static_cast<int>(
                    std::distance(
                        reference.begin(),
                        std::max_element(reference.begin(), reference.end())));
            }

            if (!comparison.passed)
            {
                size_t production_elements = 0u;
                const float *const production =
                    activeSnapshot(production_key, production_elements);
                const std::vector<float> reference =
                    loadPyTorchSnapshot(reference_key);
                ASSERT_NE(production, nullptr);
                ASSERT_EQ(production_elements, reference.size());
                const MTPNumericalDiagnostic diagnostic{
                    .reference_step = checkpoint.reference_step,
                    .reference_depth = checkpoint.identity.reference_depth,
                    .stage = suffix,
                    .production_key = production_key,
                    .reference_key = reference_key,
                    .comparison = comparison,
                    .exact_indices =
                        !comparison.is_routing_stage ||
                        comparison.routing_overlap >= 1.0f - 1.0e-6f,
                    .finite = mtpComparisonIsFinite(comparison),
                    .passed = comparison.passed,
                };
                retainMTPFailureValues(
                    diagnostic,
                    std::span<const float>(production, production_elements),
                    reference);
            }
        }

        if (checkpoint.identity.role != MTPParityCheckpointRole::Primary)
            return;

        LayerStats layer;
        layer.layer_idx = checkpoint.model_layer;
        layer.stage_results = stages;
        const std::vector<LayerStats> layers{std::move(layer)};
        observeComparedRoutedExpertCheckpoint(
            RoutedExpertCheckpointContext{
                .phase = ParityForwardPhase::Decode,
                .step = checkpoint.reference_step,
                .mtp = checkpoint,
            },
            layers);
    }

    /**
     * @brief Retain the selected serial row from the generic production oracle.
     *
     * The grouped verifier comparison consumes its first physical row, whose
     * logical decode-step identity is selected before serial execution.
     * Capturing any other row would reproduce the historical multi-request
     * diagnostic cost without increasing the batch-invariance proof surface.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::observeProductionParityMTPSerialOracleBoundary(
        const ProductionParityMTPSerialOracleBoundary &boundary) -> void
    {
        if (!isQwen122ProductionTest() || !activeMTPEnabled() ||
            !isRootParityRank() || !boundary.checkpoint_reference)
        {
            return;
        }
        retainMTPSerialVerifierDiagnostic(boundary);
    }

    /**
     * @brief Compare grouped verifier row zero with HF and serial M=1 rows.
     *
     * This is the Qwen-specific batch-invariance proof previously implemented
     * by launching a second serial/grouped campaign.  The generic production
     * proof has already produced both authorities, so this method only reads
     * their live diagnostic banks and emits scalar evidence.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::compareReusedMTPMainVerifierRows(
        const ProductionParityMTPTransactionBoundary &boundary) -> void
    {
        ASSERT_TRUE(mtp_serial_verifier_diagnostic_.has_value())
            << "Grouped MTP diagnostics have no selected serial-row oracle";
        const size_t verifier_rows =
            static_cast<size_t>(activeMTPPhysicalVerifierRows());
        ASSERT_GT(verifier_rows, 0u);

        const auto gdn = getGDNHeadConfig();
        const auto moe = getMoEConfig();
        size_t compared_main_stages = 0u;
        size_t compared_main_lm_heads = 0u;
        size_t failed_main_lm_heads = 0u;
        double numerical_cosine_sum = 0.0;
        size_t numerical_cosine_count = 0u;

        for (const std::string &key : activeSnapshotKeys())
        {
            if (!isMTPMainVerifierDiagnosticKey(key) ||
                key == "LM_HEAD_ROWS_SELECT")
            {
                continue;
            }

            size_t grouped_elements = 0u;
            const float *const grouped = activeSnapshot(key, grouped_elements);
            if (!grouped || grouped_elements == 0u ||
                grouped_elements % verifier_rows != 0u)
            {
                continue;
            }
            const size_t row_elements = grouped_elements / verifier_rows;
            const std::string reference_key =
                "decode_step" + std::to_string(boundary.reference_step) +
                "_" + key;
            std::vector<float> reference =
                loadPyTorchSnapshot(reference_key);
            if (reference.empty() || reference.size() < row_elements ||
                reference.size() % row_elements != 0u)
            {
                continue;
            }
            if (reference.size() > row_elements)
            {
                reference.erase(
                    reference.begin(),
                    reference.end() - static_cast<ptrdiff_t>(row_elements));
            }

            const std::string stage = mtpMainVerifierStage(key);
            const std::vector<float> permuted =
                applyGDNHeadPermutation(grouped, row_elements, stage, gdn);
            const float *const actual =
                permuted.empty() ? grouped : permuted.data();

            StageComparisonResult comparison;
            if (stage == "MOE_ROUTING_INDICES")
            {
                comparison = compareRoutingIndices(
                    actual, reference, row_elements, moe.top_k, stage);
            }
            else if (stage == "MOE_ROUTING_WEIGHTS")
            {
                const std::string index_key =
                    key.substr(
                        0,
                        key.size() -
                            std::string("MOE_ROUTING_WEIGHTS").size()) +
                    "MOE_ROUTING_INDICES";
                size_t grouped_index_elements = 0u;
                const float *const grouped_indices =
                    activeSnapshot(index_key, grouped_index_elements);
                std::vector<float> reference_indices = loadPyTorchSnapshot(
                    "decode_step" +
                    std::to_string(boundary.reference_step) + "_" +
                    index_key);
                if (reference_indices.size() > row_elements &&
                    reference_indices.size() % row_elements == 0u)
                {
                    reference_indices.erase(
                        reference_indices.begin(),
                        reference_indices.end() -
                            static_cast<ptrdiff_t>(row_elements));
                }
                if (grouped_indices &&
                    grouped_index_elements == verifier_rows * row_elements &&
                    reference_indices.size() == row_elements)
                {
                    comparison = compareRoutingWeights(
                        actual,
                        reference,
                        grouped_indices,
                        reference_indices,
                        row_elements,
                        moe.top_k,
                        moe.num_experts,
                        stage);
                }
                else
                {
                    comparison =
                        compareTensors(actual, reference, row_elements, stage);
                }
            }
            else
            {
                comparison =
                    compareTensors(actual, reference, row_elements, stage);
            }

            bool finite = true;
            bool exact_indices = true;
            for (size_t index = 0u; index < row_elements; ++index)
            {
                finite = finite && std::isfinite(actual[index]) &&
                         std::isfinite(reference[index]);
                exact_indices =
                    exact_indices && actual[index] == reference[index];
            }
            float kl = 0.0f;
            bool passed = finite && comparison.passed;
            if (stage == "MOE_ROUTING_INDICES")
            {
                const float minimum_overlap =
                    row_elements > 0u
                        ? 1.0f - 1.0f / static_cast<float>(row_elements)
                        : 1.0f;
                passed = finite && comparison.routing_top1_match >= 1.0f &&
                         comparison.routing_overlap >= minimum_overlap;
            }
            if (key == "LM_HEAD")
            {
                ++compared_main_lm_heads;
                kl = computeKLDivergence(
                    actual,
                    reference.data(),
                    row_elements,
                    static_cast<size_t>(orch_runner_->vocabSize()));
                const ReferenceTopKContainmentResult topk =
                    evaluateReferenceTopKContainment(
                        actual,
                        reference.data(),
                        row_elements,
                        row_elements,
                        config_.pytorch_top1_in_topk);
                passed = finite &&
                         comparison.cosine_similarity >=
                             config_.decode_cosine_threshold &&
                         kl < config_.kl_threshold && topk.passed;
                if (!passed)
                    ++failed_main_lm_heads;
            }
            comparison.kl_divergence = kl;
            comparison.passed = passed;
            if (!comparison.is_routing_stage)
            {
                numerical_cosine_sum += comparison.cosine_similarity;
                ++numerical_cosine_count;
            }

            MTPNumericalDiagnostic diagnostic{
                .reference_step = boundary.reference_step,
                .reference_depth = -2,
                .stage = stage,
                .production_key = key,
                .reference_key = reference_key,
                .comparison = comparison,
                .exact_indices = exact_indices,
                .finite = finite,
                .passed = passed,
            };
            retainMTPFailureValues(
                diagnostic,
                std::span<const float>(actual, row_elements),
                reference);
            mtp_numerical_diagnostics_.push_back(std::move(diagnostic));
            ++compared_main_stages;
        }

        ASSERT_GT(compared_main_stages, 0u)
            << "No reused grouped main-model checkpoint matched Hugging Face";
        ASSERT_GT(numerical_cosine_count, 0u)
            << "Grouped main-model comparison produced no numerical rows";
        EXPECT_GE(
            numerical_cosine_sum /
                static_cast<double>(numerical_cosine_count),
            static_cast<double>(config_.decode_cosine_threshold))
            << "Grouped main-model aggregate cosine failed against Hugging Face";
        EXPECT_GT(compared_main_lm_heads, 0u)
            << "Grouped main-model comparison omitted LM_HEAD";
        EXPECT_EQ(failed_main_lm_heads, 0u)
            << "Grouped main-model LM_HEAD failed cosine/KL/top-k parity";

        const auto &serial = *mtp_serial_verifier_diagnostic_;
        size_t compared_serial_stages = 0u;
        for (const auto &[key, serial_values] : serial.snapshots)
        {
            size_t grouped_elements = 0u;
            const float *const grouped = activeSnapshot(key, grouped_elements);
            if (!grouped || serial_values.empty() ||
                grouped_elements != verifier_rows * serial_values.size())
            {
                continue;
            }

            bool finite = true;
            bool exact = true;
            double maximum_absolute_error = 0.0;
            for (size_t index = 0u; index < serial_values.size(); ++index)
            {
                finite = finite && std::isfinite(grouped[index]) &&
                         std::isfinite(serial_values[index]);
                exact = exact && grouped[index] == serial_values[index];
                maximum_absolute_error = std::max(
                    maximum_absolute_error,
                    std::abs(
                        static_cast<double>(grouped[index]) -
                        static_cast<double>(serial_values[index])));
            }
            StageComparisonResult comparison;
            comparison.stage_name = mtpMainVerifierStage(key);
            comparison.total_elements = serial_values.size();
            comparison.cosine_similarity = computeCosineSimilarity(
                grouped, serial_values.data(), serial_values.size());
            comparison.max_abs_diff =
                static_cast<float>(maximum_absolute_error);
            comparison.is_routing_stage =
                std::string_view(key).ends_with("MOE_ROUTING_INDICES");
            if (comparison.is_routing_stage)
            {
                comparison.routing_overlap = exact ? 1.0f : 0.0f;
                comparison.routing_top1_match = exact ? 1.0f : 0.0f;
            }
            bool passed = finite &&
                          (comparison.is_routing_stage
                               ? exact
                               : comparison.cosine_similarity >=
                                     config_.decode_cosine_threshold);
            if (key == "LM_HEAD")
            {
                comparison.kl_divergence = computeKLDivergence(
                    grouped,
                    serial_values.data(),
                    serial_values.size(),
                    static_cast<size_t>(orch_runner_->vocabSize()));
                passed = passed &&
                         comparison.kl_divergence < config_.kl_threshold;
            }
            comparison.passed = passed;
            MTPNumericalDiagnostic diagnostic{
                .reference_step = boundary.reference_step,
                .reference_depth = -1,
                .stage = comparison.stage_name,
                .production_key = key,
                .reference_key =
                    "serial_reference_step" +
                    std::to_string(boundary.reference_step) + "_" + key,
                .comparison = comparison,
                .exact_indices = exact,
                .finite = finite,
                .passed = passed,
            };
            retainMTPFailureValues(
                diagnostic,
                std::span<const float>(grouped, serial_values.size()),
                serial_values);
            mtp_numerical_diagnostics_.push_back(std::move(diagnostic));
            ++compared_serial_stages;
            EXPECT_TRUE(passed)
                << "Grouped verifier row zero diverged from its selected serial decode at "
                << key << " cosine=" << comparison.cosine_similarity
                << " max_abs_diff=" << comparison.max_abs_diff
                << " kl=" << comparison.kl_divergence;
        }
        ASSERT_GT(compared_serial_stages, 0u)
            << "Grouped verifier exposed no row-zero batch-invariance surface";
    }

    /**
     * @brief Consume the generic transaction as the Qwen MTP diagnostic source.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::observeComparedProductionParityMTPTransaction(
        const ProductionParityMTPTransactionBoundary &boundary,
        std::span<const int32_t> serial_oracle) -> void
    {
        if (!isQwen122ProductionTest() || !activeMTPEnabled() ||
            !isRootParityRank())
        {
            return;
        }
        ASSERT_FALSE(mtp_transaction_diagnostic_.has_value())
            << "A typed parity cell published more than one MTP transaction";
        ASSERT_FALSE(serial_oracle.empty());
        ASSERT_TRUE(boundary.serial_token_exact);
        ASSERT_EQ(boundary.emitted_tokens, boundary.serial_oracle_tokens);
        EXPECT_GT(
            boundary.after.mtp_transaction_commits,
            boundary.before.mtp_transaction_commits);
        EXPECT_EQ(
            boundary.after.mtp_transaction_rollbacks,
            boundary.before.mtp_transaction_rollbacks);
        EXPECT_EQ(
            boundary.after.mtp_transaction_validation_failures,
            boundary.before.mtp_transaction_validation_failures);
        EXPECT_EQ(boundary.after.mtp_max_depth, activeMTPDraftDepth());

        const auto placement = pinnedDevicePlacementEpochEvidence();
        ASSERT_TRUE(placement.has_value())
            << "Grouped MTP transaction omitted device-authenticated placement";
        mtp_grouped_execution_epoch_ =
            placement.has_value() ? placement->epoch : 0u;
        compareReusedMTPMainVerifierRows(boundary);
        mtp_transaction_diagnostic_ = boundary;
    }

    /** @return Semicolon-delimited token identity for a diagnostic CSV field. */
    auto Qwen35MoENodeExpertOverlayParityTest::joinMTPDiagnosticTokens(
        std::span<const int32_t> tokens) -> std::string
    {
        std::ostringstream out;
        for (size_t index = 0u; index < tokens.size(); ++index)
        {
            if (index != 0u)
                out << ';';
            out << tokens[index];
        }
        return out.str();
    }

    /**
     * @brief Write Qwen-specific diagnostics from the canonical MTP proof.
     *
     * No inference is legal here.  Sidecar rows were compared by the generic
     * production campaign, while grouped-main and serial-row diagnostics were
     * compared by the typed transaction observer before the reusable graph
     * banks could be overwritten.  This method only projects those immutable
     * results into the historical CSV schemas.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::writeReusedMTPHuggingFaceCheckpointEvidence() -> void
    {
        if (!isQwen122ProductionTest() || !activeMTPEnabled() ||
            !isRootParityRank())
        {
            return;
        }

        const auto &transactions =
            productionParityMTPTransactionBoundaries();
        ASSERT_EQ(transactions.size(), 1u)
            << "Canonical MTP proof published the wrong transaction count";
        ASSERT_TRUE(mtp_serial_verifier_diagnostic_.has_value())
            << "Canonical MTP proof published no selected serial-row diagnostic";
        ASSERT_FALSE(mtp_checkpoint_diagnostics_.empty())
            << "Canonical MTP proof compared no recursive checkpoint bank";
        const auto &boundary = transactions.front();
        const auto &serial = *mtp_serial_verifier_diagnostic_;

        const auto result_dir = ensureResultsDir();
        const auto token_csv_path =
            result_dir / "mtp_sidecar_token_trace.csv";
        const auto snapshot_csv_path =
            result_dir / "mtp_sidecar_snapshot_breakdown.csv";
        const auto failure_values_csv_path =
            result_dir / "mtp_sidecar_failure_values.csv";
        std::ofstream token_csv(token_csv_path, std::ios::trunc);
        std::ofstream snapshot_csv(snapshot_csv_path, std::ios::trunc);
        std::ofstream failure_values_csv(
            failure_values_csv_path, std::ios::trunc);
        ASSERT_TRUE(token_csv.is_open()) << token_csv_path;
        ASSERT_TRUE(snapshot_csv.is_open()) << snapshot_csv_path;
        ASSERT_TRUE(failure_values_csv.is_open()) << failure_values_csv_path;

        token_csv
            << "call,reference_step,selected_depth,emitted_tokens,"
               "serial_expected_tokens,hf_expected_tokens,hf_branch_compatible,"
               "serial_epoch_compatible,serial_movement_epoch,"
               "grouped_movement_epoch_begin,grouped_movement_epoch_end,"
               "serial_execution_epoch,grouped_execution_epoch,"
               "serial_trajectory_epoch,grouped_trajectory_epoch,"
               "production_mtp0_top1,hf_mtp0_top1,recursive_branch_compatible,"
               "verifier_identity_transaction_count,verifier_identity_depth,"
               "production_verifier_draft_tokens,"
               "draft_steps,verifier_runs,accepted,rejected,commits,"
               "rollbacks,validation_failures,current_position\n";
        snapshot_csv
            << "call,reference_step,reference_depth,production_key,reference_key,"
               "elements,cosine,max_abs_diff,kl,exact_indices,routing_overlap,"
               "routing_top1_match,finite,passed\n";
        failure_values_csv
            << "call,reference_step,reference_depth,stage,index,production,reference\n"
            << std::setprecision(std::numeric_limits<float>::max_digits10);

        const std::vector<int> expected_tokens =
            readDecodeTokensFromMetadata();
        ASSERT_LE(
            static_cast<size_t>(boundary.reference_step) +
                boundary.emitted_tokens.size(),
            expected_tokens.size());
        const std::vector<int32_t> hf_expected(
            expected_tokens.begin() + boundary.reference_step,
            expected_tokens.begin() + boundary.reference_step +
                static_cast<ptrdiff_t>(boundary.emitted_tokens.size()));
        const bool hf_branch_compatible =
            std::equal(
                boundary.emitted_tokens.begin(),
                boundary.emitted_tokens.end(),
                hf_expected.begin());

        bool recursive_branch_compatible = true;
        for (int depth = 0;
             depth < boundary.snapshot_execution_draft_depth;
             ++depth)
        {
            ASSERT_LT(
                static_cast<size_t>(depth),
                boundary.after.mtp_observed_verifier_draft_tokens.size());
            const std::vector<float> canonical_logits =
                loadPyTorchSnapshot(
                    "decode_step" +
                    std::to_string(boundary.reference_step) + "_MTP" +
                    std::to_string(depth) + "_LM_HEAD");
            ASSERT_FALSE(canonical_logits.empty());
            const int32_t canonical_token = static_cast<int32_t>(
                std::distance(
                    canonical_logits.begin(),
                    std::max_element(
                        canonical_logits.begin(), canonical_logits.end())));
            recursive_branch_compatible = recursive_branch_compatible &&
                boundary.after.mtp_observed_verifier_draft_tokens[
                    static_cast<size_t>(depth)] == canonical_token;
        }

        const bool serial_epoch_compatible =
            serial.boundary.before.moe_runtime_movement_epoch ==
                serial.boundary.after.moe_runtime_movement_epoch &&
            boundary.before.moe_runtime_movement_epoch ==
                boundary.after.moe_runtime_movement_epoch &&
            serial.execution_epoch == mtp_grouped_execution_epoch_;
        const uint64_t serial_trajectory_epoch =
            serial.boundary.before.moe_runtime_movement_epoch ==
                    serial.execution_epoch &&
                serial.boundary.after.moe_runtime_movement_epoch ==
                    serial.execution_epoch
                ? serial.execution_epoch
                : 0u;
        const uint64_t grouped_trajectory_epoch =
            boundary.before.moe_runtime_movement_epoch ==
                    mtp_grouped_execution_epoch_ &&
                boundary.after.moe_runtime_movement_epoch ==
                    mtp_grouped_execution_epoch_
                ? mtp_grouped_execution_epoch_
                : 0u;

        token_csv
            << "0," << boundary.reference_step << ','
            << boundary.snapshot_execution_draft_depth << ','
            << joinMTPDiagnosticTokens(boundary.emitted_tokens) << ','
            << joinMTPDiagnosticTokens(boundary.serial_oracle_tokens) << ','
            << joinMTPDiagnosticTokens(hf_expected) << ','
            << (hf_branch_compatible ? 1 : 0) << ','
            << (serial_epoch_compatible ? 1 : 0) << ','
            << serial.boundary.before.moe_runtime_movement_epoch << ','
            << boundary.before.moe_runtime_movement_epoch << ','
            << boundary.after.moe_runtime_movement_epoch << ','
            << serial.execution_epoch << ',' << mtp_grouped_execution_epoch_
            << ',' << serial_trajectory_epoch << ','
            << grouped_trajectory_epoch << ','
            << mtp_primary_production_top1_.value_or(-1) << ','
            << mtp_primary_reference_top1_.value_or(-1) << ','
            << (recursive_branch_compatible ? 1 : 0) << ','
            << boundary.after.mtp_observed_verifier_transaction_count << ','
            << boundary.after.mtp_observed_verifier_draft_depth << ','
            << joinMTPDiagnosticTokens(
                   boundary.after.mtp_observed_verifier_draft_tokens)
            << ',' << boundary.after.mtp_draft_steps << ','
            << boundary.after.mtp_verifier_runs << ','
            << boundary.after.mtp_accepted_tokens << ','
            << boundary.after.mtp_rejected_tokens << ','
            << boundary.after.mtp_transaction_commits << ','
            << boundary.after.mtp_transaction_rollbacks << ','
            << boundary.after.mtp_transaction_validation_failures << ','
            << boundary.after.current_position << '\n';

        const auto write_row = [&](const MTPNumericalDiagnostic &row)
        {
            snapshot_csv
                << row.call << ',' << row.reference_step << ','
                << row.reference_depth << ',' << row.production_key << ','
                << row.reference_key << ','
                << row.comparison.total_elements << ','
                << row.comparison.cosine_similarity << ','
                << row.comparison.max_abs_diff << ','
                << row.comparison.kl_divergence << ','
                << (row.exact_indices ? 1 : 0) << ','
                << (row.comparison.is_routing_stage
                        ? row.comparison.routing_overlap
                        : 1.0f)
                << ','
                << (row.comparison.is_routing_stage
                        ? row.comparison.routing_top1_match
                        : 1.0f)
                << ',' << (row.finite ? 1 : 0) << ','
                << (row.passed ? 1 : 0) << '\n';
        };

        for (const MTPCheckpointDiagnostic &bank :
             mtp_checkpoint_diagnostics_)
        {
            for (const StageComparisonResult &comparison : bank.stages)
            {
                const size_t separator = comparison.stage_name.find('_');
                ASSERT_NE(separator, std::string::npos);
                const std::string suffix =
                    comparison.stage_name.substr(separator + 1u);
                std::string production_key;
                if (suffix == "TERMINAL_HIDDEN_ROW_SELECT")
                {
                    production_key = bank.checkpoint.production_stage_prefix;
                    ASSERT_TRUE(production_key.ends_with("MTP0_"));
                    production_key.resize(production_key.size() - 5u);
                    production_key += "MTP_TERMINAL_HIDDEN_ROW_SELECT";
                }
                else
                {
                    production_key =
                        bank.checkpoint.production_stage_prefix + suffix;
                }
                write_row(MTPNumericalDiagnostic{
                    .reference_step = bank.checkpoint.reference_step,
                    .reference_depth =
                        bank.checkpoint.identity.reference_depth,
                    .stage = suffix,
                    .production_key = production_key,
                    .reference_key =
                        bank.checkpoint.reference_stage_prefix + suffix,
                    .comparison = comparison,
                    .exact_indices =
                        !comparison.is_routing_stage ||
                        comparison.routing_overlap >= 1.0f - 1.0e-6f,
                    .finite = mtpComparisonIsFinite(comparison),
                    .passed = comparison.passed,
                });
            }
        }
        for (const MTPNumericalDiagnostic &row : mtp_numerical_diagnostics_)
            write_row(row);

        for (const MTPFailureDiagnostic &failure : mtp_failure_diagnostics_)
        {
            ASSERT_EQ(failure.production.size(), failure.reference.size());
            for (size_t index = 0u; index < failure.production.size(); ++index)
            {
                failure_values_csv
                    << failure.call << ',' << failure.reference_step << ','
                    << failure.reference_depth << ',' << failure.stage << ','
                    << index << ',' << failure.production[index] << ','
                    << failure.reference[index] << '\n';
            }
        }

        token_csv.flush();
        snapshot_csv.flush();
        failure_values_csv.flush();
        EXPECT_TRUE(token_csv.good()) << token_csv_path;
        EXPECT_TRUE(snapshot_csv.good()) << snapshot_csv_path;
        EXPECT_TRUE(failure_values_csv.good()) << failure_values_csv_path;
    }

}
