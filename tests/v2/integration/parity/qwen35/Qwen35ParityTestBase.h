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
            LOG_INFO("[" << Base::getBackendName()
                         << " Parity] Regenerating Qwen3.5 PyTorch snapshots from GGUF: "
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
    };

} // namespace llaminar2::test::parity::qwen35
