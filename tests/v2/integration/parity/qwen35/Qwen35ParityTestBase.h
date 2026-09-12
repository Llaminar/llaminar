/**
 * @file Qwen35ParityTestBase.h
 * @brief Base class for Qwen3.5 PyTorch parity tests
 *
 * Extends the Qwen2 parity infrastructure for Qwen3.5 architecture.
 * Qwen3.5 differs from Qwen2/Qwen3 in:
 *   - Hybrid GDN (Gated Delta Network) + Full Attention layers (75%/25%)
 *   - GDN layers: causal conv1d, delta-rule recurrence, gated RMSNorm, output gate
 *   - Full attention layers: GQA with per-head QK norms, partial RoPE, output gate
 *   - No KV cache for GDN layers (fixed recurrent state)
 *
 * The underlying graph/schema machinery handles these differences
 * automatically via Qwen35SchemaFactory and Qwen35Graph.
 * This test base overrides model paths, snapshot dirs, and the
 * snapshot generation script (Qwen3.5 requires a dedicated generator).
 *
 * @author David Sanftenberg
 * @date 2026
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../qwen2/Qwen2ParityTestBase.h"
#include "models/qwen35/Qwen35Schema.h"

namespace llaminar2::test::parity::qwen35
{

    // Import all Qwen2 parity utilities — Qwen3.5 reuses the same test infrastructure
    using namespace llaminar2::test::parity::qwen2;

    /**
     * @brief Config-driven parity test specialized for Qwen3.5 models.
     *
     * Inherits from the Qwen2 ConfigDrivenParityTest but overrides:
     * 1. configureModel() — uses Qwen35SchemaFactory for TP weight sharding
     * 2. regeneratePyTorchSnapshots() — uses Qwen3.5-specific snapshot generator
     *    that handles heterogeneous GDN + FA layers
     */
    template <typename Derived>
    class Qwen35ConfigDrivenParityTest : public ConfigDrivenParityTest<Derived>
    {
    protected:
        using Base = ConfigDrivenParityTest<Derived>;

        void applyModelOverrides() override
        {
            Base::applyModelOverrides();
        }

        void configureModel(std::shared_ptr<ModelContext> model_ctx) override
        {
            if (Base::cfg().is_local_tp() || Base::cfg().is_cross_rank_tp())
            {
                // Use Qwen3.5 schema factory for proper weight sharding
                // (GDN-specific weights: attn_qkv, attn_gate, ssm_out, ssm_alpha, etc.)
                Qwen35SchemaFactory schema_factory;
                model_ctx->weightManager()->setWeightShardingConfig(
                    schema_factory.getWeightShardingConfig());
            }
        }

        /** @return Whether this typed case requires recursive MTP checkpoints. */
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
         * @brief Return the admitted recursive predictor capacity of this case.
         * @return Maximum required draft depth, or zero when MTP is disabled.
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

        /**
         * @brief Authenticate the independently repairable recursive sidecar.
         *
         * Main-model identity is authenticated by ParityTestBase.  This layer
         * validates the sidecar schema, admitted maximum depth, and both endpoint
         * tensors so an interrupted expansion cannot masquerade as a complete
         * predictor corpus.
         */
        typename Base::ReferenceSnapshotValidation
        validateModelSpecificReferenceSnapshotMetadata(
            const std::filesystem::path &metadata_path) const override
        {
            if (!requiresMTPSidecarReferenceSnapshots())
                return {true, {}};

            constexpr int kMTPSidecarSnapshotSchema = 5;
            const int required_draft_depth =
                requiredMTPSidecarReferenceDraftDepth();
            if (required_draft_depth <= 0)
                return {false, "MTP sidecar reference depth must be positive"};

            std::ifstream schema_file(
                metadata_path.parent_path() /
                "mtp_sidecar_snapshot_schema.txt");
            int observed_schema = 0;
            if (!(schema_file >> observed_schema) ||
                observed_schema != kMTPSidecarSnapshotSchema)
            {
                return {
                    false,
                    "MTP sidecar reference schema is missing or is not schema "
                    "5 recursive chaining"};
            }

            const auto available_depth = Base::readSnapshotMetadataValue(
                metadata_path,
                "mtp_sidecar_max_draft_depth");
            int parsed_available_depth = 0;
            try
            {
                size_t parsed = 0;
                parsed_available_depth = available_depth
                                             ? std::stoi(*available_depth, &parsed)
                                             : 0;
                if (!available_depth || parsed != available_depth->size())
                    parsed_available_depth = 0;
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
                        "MTP sidecar reference pack is missing " + required};
                }
            }
            return {true, {}};
        }

        /**
         * @brief Determine whether only the bounded MTP extension needs repair.
         * @return true when the complete main trajectory remains authenticated.
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
            return Base::validateReferenceSnapshotMetadata(
                       metadata_path,
                       Base::productionParityCampaignEnabled(),
                       false)
                .usable;
        }

        /**
         * @brief Regenerate PyTorch snapshots using Qwen3.5-specific generator.
         *
         * The standard generate_qwen_pipeline_snapshots.py only supports
         * Qwen2/Qwen3 (homogeneous transformer layers). Qwen3.5 has
         * heterogeneous layers (GDN + FA) requiring a dedicated generator
         * that uses the Qwen35ReferenceModel from the Python registry.
         */
        bool regeneratePyTorchSnapshots() override
        {
            const bool extend_mtp_sidecars =
                canExtendAuthenticatedMainReferenceWithMTPSidecars();
            LOG_INFO("[" << Base::getBackendName()
                         << " Parity] "
                         << (extend_mtp_sidecars
                                 ? "Extending authenticated Qwen3.5 dense MTP sidecars from GGUF: "
                                 : "Regenerating Qwen3.5 PyTorch snapshots from GGUF: ")
                         << Base::config_.model_path);

            std::ostringstream script;
            // Source devcontainer venv if present, else fall back to system
            // python3 (CI builder image installs deps to system site-packages).
            //
            // IMPORTANT: CTest sets OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1 for the
            // Llaminar test process (intentional: mpirun -np 1 handles affinity itself).
            // Those env vars are inherited by this python3 subprocess, which would pin
            // PyTorch's CPU forward pass to a single thread — catastrophic for large
            // models (e.g., 27B Qwen3.5 prefill takes ~14 min on 1 thread). We unset
            // them and let PyTorch/OpenMP/MKL use all available cores.
            script << "unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS OMP_PROC_BIND OMP_PLACES KMP_AFFINITY; "
                   << "if [ -f /workspaces/llaminar/.venv/bin/activate ]; then "
                   << "source /workspaces/llaminar/.venv/bin/activate; fi; "
                   << "python3 python/reference/generate_qwen35_pipeline_snapshots.py"
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
                LOG_ERROR("[Parity] Failed to execute Qwen3.5 snapshot generator");
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
                LOG_ERROR("[Parity] Qwen3.5 snapshot generation failed:\n"
                          << output);
                return false;
            }

            LOG_INFO("[Parity] Qwen3.5 snapshots regenerated successfully");
            return true;
        }

        /**
         * @brief Generate an additive dense HF oracle for one observed branch.
         *
         * The generator loads only the dense sidecar allocation and reuses the
         * authenticated main-model trajectory. The node-wide lease and final
         * NPY validation are owned by ParityTestBase.
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
                LOG_ERROR("[Parity] Invalid dense MTP branch identity");
                return false;
            }

            const auto artifact_dir = Base::ensureResultsDir();
            const auto request_path =
                artifact_dir / "mtp_hf_branch_request.json";
            std::ofstream request(request_path, std::ios::trunc);
            if (!request.is_open())
            {
                LOG_ERROR("[Parity] Cannot write MTP branch request "
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
                LOG_ERROR("[Parity] Failed while writing MTP branch request "
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
                   "generate_qwen35_pipeline_snapshots.py"
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

            LOG_INFO("[Parity] Generating dense forced-branch MTP oracle step="
                     << reference_step << " depth=" << condition_tokens.size());
            FILE *pipe = popen(command.c_str(), "r");
            if (!pipe)
            {
                LOG_ERROR("[Parity] Failed to start dense MTP branch generator");
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
                LOG_ERROR("[Parity] Dense MTP branch generation failed:\n"
                          << output);
                return false;
            }

            LOG_INFO("[Parity] Dense forced-branch snapshots are durable");
            return true;
        }
    };

} // namespace llaminar2::test::parity::qwen35
