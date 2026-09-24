/**
 * @file Qwen35MoEParityTestBase.h
 * @brief Base class for Qwen3.5 MoE PyTorch parity tests
 *
 * Extends the Qwen3.5 (dense) parity infrastructure for the MoE variant.
 * Qwen3.5 MoE differs from dense Qwen3.5 in the FFN block:
 *   - Dense SwiGLU FFN is replaced by SparseMoeBlock:
 *     Router → 256 experts (top-8) + shared expert + sigmoid gate
 *   - Attention architecture (GDN + FA hybrid) is identical to dense Qwen3.5
 *
 * The test base overrides:
 *   - configureModel() → uses Qwen35MoESchemaFactory for weight sharding
 *   - regeneratePyTorchSnapshots() → uses MoE-specific snapshot generator
 *
 * @author David Sanftenberg
 * @date 2026
 */

#pragma once

#include "../qwen35/Qwen35ParityTestBase.h"
#include "../MoERouteConditionedReference.h"
#include "models/qwen35moe/Qwen35MoESchema.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

namespace llaminar2::test::parity::qwen35moe
{

    // Import all Qwen3.5 parity utilities — MoE reuses the same test infrastructure
    using namespace llaminar2::test::parity::qwen35;
    using namespace llaminar2::test::parity::qwen2;

    /**
     * @brief Config-driven parity test specialized for Qwen3.5 MoE models.
     *
     * Inherits from the Qwen3.5 Qwen35ConfigDrivenParityTest but overrides:
     * 1. configureModel() — uses Qwen35MoESchemaFactory for expert weight sharding
     * 2. regeneratePyTorchSnapshots() — uses Qwen3.5 MoE-specific snapshot generator
     *    that handles SparseMoeBlock (router, experts, shared expert, sigmoid gate)
     */
    template <typename Derived>
    class Qwen35MoEConfigDrivenParityTest : public Qwen35ConfigDrivenParityTest<Derived>
    {
    protected:
        using Base = Qwen35ConfigDrivenParityTest<Derived>;

        /**
         * @brief Evaluate Qwen MoE SwiGLU experts independently on the named HF input.
         * @param prefix Authenticated canonical or branch-qualified checkpoint.
         * @param expert_ids Union of validated native and HF selected IDs.
         * @param rows Number of reference input rows.
         * @param hidden Reference hidden width.
         * @return CPU FP32 expert bank and HF RMS parameters, independent of backend/format.
         */
        MoEIndependentReference independentMoEMTPExpertReference(
            const std::string &prefix, const std::vector<int> &expert_ids,
            size_t rows, size_t hidden) override
        {
            return loadMoERouteConditionedReference({
                .model = Base::config_.model_path,
                .snapshot_directory = Base::config_.snapshot_dir,
                .artifacts = Base::ensureResultsDir(),
                .checkpoint_prefix = prefix,
                .gguf_block = Base::parityLayerCount(),
            }, expert_ids, rows, hidden);
        }

        /** @return true because every multi-device MoE uses ExpertOverlay. */
        bool requiresUniversalMoEAuthority() const override
        {
            return true;
        }

        /**
         * @brief Classify the unsummed expert oracle as reference-only.
         *
         * Direct single-participant MoE kernels preserve serial route order
         * while accumulating straight into the final routed row. They do not
         * materialize a `[token, route, hidden]` bank. ExpertOverlay does own
         * that bank as part of its sparse publication protocol and its focused
         * movement proof compares it explicitly. Keeping this classification
         * here prevents generic MTP discovery from demanding a tensor that the
         * optimized standalone graph never produces.
         */
        bool productionParityMTPCheckpointIsLive(
            std::string_view suffix) const override
        {
            return !suffix.ends_with("MOE_ROUTE_CONTRIBUTIONS");
        }

        /**
         * @brief Return whether this fixture compares recursive MTP sidecars.
         *
         * Ordinary Qwen3.5 MoE parity packs do not pay the additional
         * recursive predictor forwards. Production MTP campaigns override
         * this hook so reference authentication, regeneration, and comparison
         * all agree that the schema-5 sidecar corpus is part of the case.
         *
         * @return `true` when the reference pack must contain MTP sidecars.
         */
        virtual bool requiresMTPSidecarReferenceSnapshots() const
        {
            if constexpr (requires(const Derived &fixture)
                          {
                              fixture.modelParityCase().mtpEnabled();
                          })
            {
                return static_cast<const Derived *>(this)
                    ->modelParityCase()
                    .mtpEnabled();
            }
            return false;
        }

        /**
         * @brief Return the deepest MTP draft geometry the reference must own.
         *
         * The value is an admitted transaction capacity, not merely the first
         * selected depth. Dynamic policies must return their maximum so an old
         * shallow pack cannot certify a larger retained graph family.
         *
         * @return Required recursive sidecar count, or zero when MTP is absent.
         */
        virtual int requiredMTPSidecarReferenceDraftDepth() const
        {
            if constexpr (requires(const Derived &fixture)
                          {
                              fixture.modelParityCase()
                                  .model.maximum_mtp_draft_depth;
                          })
            {
                const auto &test_case =
                    static_cast<const Derived *>(this)->modelParityCase();
                return test_case.mtpEnabled()
                           ? test_case.model.maximum_mtp_draft_depth
                           : 0;
            }
            return requiresMTPSidecarReferenceSnapshots() ? 3 : 0;
        }

        void SetUp() override
        {
            Base::SetUp();
        }

        void configureModel(std::shared_ptr<ModelContext> model_ctx) override
        {
            if (Base::cfg().is_local_tp() || Base::cfg().is_cross_rank_tp() || Base::cfg().is_hybrid_pp_tp())
            {
                // Use Qwen3.5 MoE schema factory for proper weight sharding
                // (expert weights are replicated; attention weights shard like dense Qwen3.5)
                Qwen35MoESchemaFactory schema_factory;
                model_ctx->weightManager()->setWeightShardingConfig(
                    schema_factory.getWeightShardingConfig());
            }
        }

        /**
         * @brief Authenticate router semantics and per-route expert evidence.
         *
         * Production CUDA and ROCm routing reuse their full probability
         * workspace after softmax. The CPU PyTorch oracle must therefore bind
         * `MOE_ROUTER_OUTPUT` to that same distribution, not reconstruct the
         * pre-softmax projection under an identical filename. Movement parity
         * additionally requires an unsummed Hugging Face contribution for
         * every selected route, so a moved expert remains comparable when an
         * unrelated low-weight top-k member differs.
         */
        typename Base::ReferenceSnapshotValidation
        validateMoEMainReferenceSnapshotMetadata(
            const std::filesystem::path &metadata_path) const
        {
            constexpr int kMoERouterSnapshotSchema = 1;
            const auto observed = Base::readSnapshotMetadataValue(
                metadata_path,
                "moe_router_snapshot_schema");
            if (!observed || *observed != std::to_string(kMoERouterSnapshotSchema))
            {
                return {
                    false,
                    "moe_router_snapshot_schema is missing or does not publish "
                    "the post-softmax production boundary"};
            }

            constexpr int kMoERouteContributionSnapshotSchema = 1;
            const auto contribution_schema =
                Base::readSnapshotMetadataValue(
                    metadata_path,
                    "moe_route_contribution_snapshot_schema");
            if (!contribution_schema ||
                *contribution_schema !=
                    std::to_string(kMoERouteContributionSnapshotSchema) ||
                !std::filesystem::is_regular_file(
                    metadata_path.parent_path() /
                    "layer0_MOE_ROUTE_CONTRIBUTIONS.npy"))
            {
                return {
                    false,
                    "moe_route_contribution_snapshot_schema is missing or "
                    "does not publish per-route expert contributions"};
            }

            return {true, {}};
        }

        /**
         * @brief Authenticate MoE main checkpoints and optional MTP extension.
         *
         * Main-model provenance and router semantics remain independently
         * usable when a typed MTP cell asks for a deeper recursive sidecar
         * corpus.  Keeping the two contracts separate lets regeneration repair
         * only the bounded predictor context without weakening either one.
         */
        typename Base::ReferenceSnapshotValidation
        validateModelSpecificReferenceSnapshotMetadata(
            const std::filesystem::path &metadata_path) const override
        {
            const auto main_validation =
                validateMoEMainReferenceSnapshotMetadata(metadata_path);
            if (!main_validation.usable)
                return main_validation;

            if (requiresMTPSidecarReferenceSnapshots())
            {
                constexpr int kMTPSidecarSnapshotSchema = 5;
                const int required_draft_depth =
                    requiredMTPSidecarReferenceDraftDepth();
                if (required_draft_depth <= 0)
                {
                    return {
                        false,
                        "MTP sidecar reference depth must be positive"};
                }
                std::ifstream schema_file(
                    metadata_path.parent_path() /
                    "mtp_sidecar_snapshot_schema.txt");
                int observed_schema = 0;
                if (!(schema_file >> observed_schema) ||
                    observed_schema != kMTPSidecarSnapshotSchema)
                {
                    return {
                        false,
                        "MTP sidecar reference schema is missing or is not "
                        "schema 5 recursive chaining"};
                }

                const auto available_depth =
                    Base::readSnapshotMetadataValue(
                        metadata_path,
                        "mtp_sidecar_max_draft_depth");
                int parsed_available_depth = 0;
                try
                {
                    size_t parsed = 0;
                    parsed_available_depth = available_depth
                                                 ? std::stoi(
                                                       *available_depth,
                                                       &parsed)
                                                 : 0;
                    if (!available_depth ||
                        parsed != available_depth->size())
                    {
                        parsed_available_depth = 0;
                    }
                }
                catch (...)
                {
                    parsed_available_depth = 0;
                }
                if (parsed_available_depth < required_draft_depth)
                {
                    return {
                        false,
                        "MTP sidecar reference pack admits depth " +
                            std::to_string(parsed_available_depth) +
                            ", but this cell requires depth " +
                            std::to_string(required_draft_depth)};
                }

                /*
                 * These two endpoints prove that regeneration completed both
                 * the externally admitted predictor and the deepest recursive
                 * predictor.  The numerical recorder later requires every
                 * intermediate stage; this lightweight validation prevents an
                 * interrupted corpus from being accepted before setup.
                 */
                const auto snapshot_dir = metadata_path.parent_path();
                const std::array<std::string, 2> required_snapshots = {
                    "decode_step0_MTP0_EMBEDDING.npy",
                    "decode_step0_MTP" +
                        std::to_string(required_draft_depth - 1) +
                        "_LM_HEAD.npy",
                };
                for (const auto &required : required_snapshots)
                {
                    if (!std::filesystem::exists(snapshot_dir / required))
                    {
                        return {
                            false,
                            std::string("MTP sidecar reference pack is missing ") +
                                required};
                    }
                }
            }
            return {true, {}};
        }

        /**
         * @brief Decide whether an existing main trajectory can be extended.
         *
         * This method authenticates the complete shared reference identity and
         * MoE router boundary while intentionally excluding only the separately
         * repairable MTP sidecar contract.  A stale model, prompt, token stream,
         * decode depth, or router schema therefore forces full regeneration.
         */
        bool canExtendAuthenticatedMainReferenceWithMTPSidecars() const
        {
            if (!requiresMTPSidecarReferenceSnapshots())
                return false;

            const auto metadata_path =
                std::filesystem::path(Base::config_.snapshot_dir) /
                "metadata.txt";
            if (!std::filesystem::is_regular_file(metadata_path))
                return false;

            const auto common_validation =
                Base::validateReferenceSnapshotMetadata(
                    metadata_path,
                    Base::productionParityCampaignEnabled(),
                    false);
            if (!common_validation.usable)
                return false;

            return validateMoEMainReferenceSnapshotMetadata(metadata_path).usable;
        }

        /**
         * @brief Regenerate PyTorch snapshots using Qwen3.5 MoE-specific generator.
         *
         * The standard Qwen3.5 snapshot generator only handles dense SwiGLU FFN.
         * MoE models require a dedicated generator that uses the
         * Qwen35MoEReferenceModel from the Python registry, which hooks:
         *   - MOE_ROUTER_OUTPUT (full post-softmax router distribution)
         *   - MOE_EXPERT_OUTPUT (combined routed expert output)
         *   - MOE_SHARED_EXPERT_OUTPUT (shared expert before gate)
         *   - MOE_SHARED_GATE_OUTPUT (after sigmoid gating)
         *   - MOE_COMBINED_OUTPUT (routed + shared)
         */
        bool regeneratePyTorchSnapshots() override
        {
            const bool extend_mtp_sidecars =
                canExtendAuthenticatedMainReferenceWithMTPSidecars();
            LOG_INFO("[" << Base::getBackendName()
                         << " Parity] "
                         << (extend_mtp_sidecars
                                 ? "Extending authenticated Qwen3.5 MoE MTP sidecars from GGUF: "
                                 : "Regenerating Qwen3.5 MoE PyTorch snapshots from GGUF: ")
                         << Base::config_.model_path);

            std::ostringstream script;
            // Source devcontainer venv if present, else fall back to system
            // python3 (CI builder image installs deps to system site-packages).
            //
            // IMPORTANT: CTest sets OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1 for the
            // Llaminar test process. Unset them so PyTorch uses all available cores
            // (Qwen3.5 MoE 35B prefill is especially slow single-threaded).
            script << "unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS OMP_PROC_BIND OMP_PLACES KMP_AFFINITY; "
                   << "if [ -f /workspaces/llaminar/.venv/bin/activate ]; then "
                   << "source /workspaces/llaminar/.venv/bin/activate; fi; "
                   << "python3 python/reference/generate_qwen35_moe_pipeline_snapshots.py"
                   << " --model " << Base::parityShellQuote(Base::config_.model_path)
                   << " --prompt " << Base::parityShellQuote(Base::config_.prompt)
                   << " --output " << Base::parityShellQuote(Base::config_.snapshot_dir)
                   << " --decode-steps " << Base::config_.decode_steps;
            if (requiresMTPSidecarReferenceSnapshots())
            {
                script << " --mtp-sidecar-snapshots"
                       << " --mtp-max-draft-depth "
                       << requiredMTPSidecarReferenceDraftDepth();
                if (extend_mtp_sidecars)
                    script << " --mtp-sidecar-only";
            }
            const std::string command =
                "bash -c " + Base::parityShellQuote(script.str()) + " 2>&1";

            FILE *pipe = popen(command.c_str(), "r");
            if (!pipe)
            {
                LOG_ERROR("[Parity] Failed to execute Qwen3.5 MoE snapshot generator");
                return false;
            }

            char buffer[256];
            std::string output;
            while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
            {
                output += buffer;
            }

            int exit_code = pclose(pipe);
            if (exit_code != 0)
            {
                LOG_ERROR("[Parity] Qwen3.5 MoE snapshot generation failed:\n"
                          << output);
                return false;
            }

            LOG_INFO("[Parity] Qwen3.5 MoE snapshots regenerated successfully");
            return true;
        }

        /**
         * @brief Generate an exact additive oracle for one observed MTP branch.
         *
         * The MoE generator's branch mode constructs only embeddings, LM head,
         * rotary state, and the single trailing predictor layer. It reuses the
         * authenticated main-model hidden/token trajectory already present in
         * the canonical pack, so a quantized proposal divergence costs one
         * bounded sidecar replay rather than a second complete 35B model load.
         * The caller owns node-wide serialization and validates every resulting
         * NPY before comparison.
         * @param reference_step Main-model position in the authenticated pack.
         * @param condition_tokens Inputs consumed by recursive MTP1..N rows.
         * @param condition_token_override Actual MTP0 input when noncanonical.
         * @return True after the additive FP32 branch is durably generated.
         */
        bool regeneratePyTorchMTPBranchSnapshots(
            int reference_step,
            const std::vector<int32_t> &condition_tokens,
            std::optional<int32_t> condition_token_override) override
        {
            if (reference_step < 0 ||
                (condition_tokens.empty() && !condition_token_override) ||
                (condition_token_override && *condition_token_override < 0) ||
                condition_tokens.size() >= 15u ||
                std::any_of(
                    condition_tokens.begin(),
                    condition_tokens.end(),
                    [](int32_t token) { return token < 0; }))
            {
                LOG_ERROR(
                    "[Parity] Invalid Qwen3.5 MoE MTP branch identity");
                return false;
            }

            const auto artifact_dir = Base::ensureResultsDir();
            const auto request_path =
                artifact_dir / "mtp_hf_branch_request.json";
            std::ofstream request(request_path, std::ios::trunc);
            if (!request.is_open())
            {
                LOG_ERROR(
                    "[Parity] Cannot write MTP branch request "
                    << request_path);
                return false;
            }
            request << "{\"" << reference_step << "\": {\"condition_token\": ";
            if (condition_token_override)
                request << *condition_token_override;
            else
                request << "null";
            request << ", \"draft_tokens\": [";
            for (size_t index = 0; index < condition_tokens.size(); ++index)
            {
                if (index != 0u)
                    request << ", ";
                request << condition_tokens[index];
            }
            request << "]}}\n";
            request.flush();
            if (!request.good())
            {
                LOG_ERROR(
                    "[Parity] Failed while writing MTP branch request "
                    << request_path);
                return false;
            }
            request.close();

            std::ostringstream script;
            script
                << "unset OMP_NUM_THREADS MKL_NUM_THREADS "
                   "OPENBLAS_NUM_THREADS OMP_PROC_BIND OMP_PLACES "
                   "KMP_AFFINITY; "
                << "if [ -f /workspaces/llaminar/.venv/bin/activate ]; then "
                   "source /workspaces/llaminar/.venv/bin/activate; fi; "
                << "python3 python/reference/"
                   "generate_qwen35_moe_pipeline_snapshots.py"
                << " --model "
                << Base::parityShellQuote(Base::config_.model_path)
                << " --prompt "
                << Base::parityShellQuote(Base::config_.prompt)
                << " --output "
                << Base::parityShellQuote(Base::config_.snapshot_dir)
                << " --decode-steps " << Base::config_.decode_steps
                << " --mtp-sidecar-snapshots --mtp-max-draft-depth "
                << condition_tokens.size() + 1u
                << " --mtp-branch-overrides "
                << Base::parityShellQuote(request_path.string());
            const std::string command =
                "bash -c " + Base::parityShellQuote(script.str()) + " 2>&1";

            LOG_INFO(
                "[Parity] Generating Qwen3.5 MoE forced-branch MTP oracle "
                "step="
                << reference_step << " depth=" << condition_tokens.size());
            FILE *pipe = popen(command.c_str(), "r");
            if (!pipe)
            {
                LOG_ERROR(
                    "[Parity] Failed to start Qwen3.5 MoE MTP branch generator");
                return false;
            }

            std::string output;
            std::array<char, 512> buffer{};
            while (fgets(buffer.data(), buffer.size(), pipe) != nullptr)
                output += buffer.data();
            const int exit_code = pclose(pipe);

            const auto log_path =
                artifact_dir / "mtp_hf_branch_generation.log";
            std::ofstream log(log_path, std::ios::trunc);
            if (log.is_open())
                log << output;
            if (exit_code != 0)
            {
                LOG_ERROR(
                    "[Parity] Qwen3.5 MoE MTP branch generation failed:\n"
                    << output);
                return false;
            }

            LOG_INFO(
                "[Parity] Qwen3.5 MoE forced-branch snapshots are durable");
            return true;
        }
    };

} // namespace llaminar2::test::parity::qwen35moe
