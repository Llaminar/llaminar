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
#include "models/qwen35moe/Qwen35MoESchema.h"

#include <filesystem>
#include <fstream>

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

        /** @return true because every multi-device MoE uses ExpertOverlay. */
        bool requiresUniversalMoEAuthority() const override
        {
            return true;
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
         * @brief Reject authenticated packs with the retired raw-logit key.
         *
         * Production CUDA and ROCm routing reuse their full probability
         * workspace after softmax. The CPU PyTorch oracle must therefore bind
         * `MOE_ROUTER_OUTPUT` to that same distribution, not reconstruct the
         * pre-softmax projection under an identical filename.
         */
        typename Base::ReferenceSnapshotValidation
        validateModelSpecificReferenceSnapshotMetadata(
            const std::filesystem::path &metadata_path) const override
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
            LOG_INFO("[" << Base::getBackendName()
                         << " Parity] Regenerating Qwen3.5 MoE PyTorch snapshots from GGUF: "
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
    };

} // namespace llaminar2::test::parity::qwen35moe
