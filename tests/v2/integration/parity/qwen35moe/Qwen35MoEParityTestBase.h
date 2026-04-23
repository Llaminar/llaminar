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

        void SetUp() override
        {
            Base::SetUp();
        }

        void configureModel(std::shared_ptr<ModelContext> model_ctx) override
        {
            if (Base::cfg().is_local_tp() || Base::cfg().is_cross_rank_tp())
            {
                // Use Qwen3.5 MoE schema factory for proper weight sharding
                // (expert weights are replicated; attention weights shard like dense Qwen3.5)
                Qwen35MoESchemaFactory schema_factory;
                model_ctx->weightManager()->setWeightShardingConfig(
                    schema_factory.getWeightShardingConfig());
            }
        }

        /**
         * @brief Regenerate PyTorch snapshots using Qwen3.5 MoE-specific generator.
         *
         * The standard Qwen3.5 snapshot generator only handles dense SwiGLU FFN.
         * MoE models require a dedicated generator that uses the
         * Qwen35MoEReferenceModel from the Python registry, which hooks:
         *   - MOE_ROUTER_OUTPUT (router logits)
         *   - MOE_EXPERT_OUTPUT (combined routed expert output)
         *   - MOE_SHARED_EXPERT_OUTPUT (shared expert before gate)
         *   - MOE_SHARED_GATE_OUTPUT (after sigmoid gating)
         *   - MOE_COMBINED_OUTPUT (routed + shared)
         */
        bool regeneratePyTorchSnapshots()
        {
            LOG_INFO("[" << Base::getBackendName()
                         << " Parity] Regenerating Qwen3.5 MoE PyTorch snapshots from GGUF: "
                         << Base::config_.model_path);

            std::ostringstream cmd;
            // Source devcontainer venv if present, else fall back to system
            // python3 (CI builder image installs deps to system site-packages).
            //
            // IMPORTANT: CTest sets OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1 for the
            // Llaminar test process. Unset them so PyTorch uses all available cores
            // (Qwen3.5 MoE 35B prefill is especially slow single-threaded).
            cmd << "bash -c 'unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS OMP_PROC_BIND OMP_PLACES KMP_AFFINITY; "
                << "[ -f /workspaces/llaminar/.venv/bin/activate ] && source /workspaces/llaminar/.venv/bin/activate; python3"
                << " python/reference/generate_qwen35_moe_pipeline_snapshots.py"
                << " --model " << Base::config_.model_path
                << " --prompt \"" << Base::config_.prompt << "\""
                << " --output " << Base::config_.snapshot_dir
                << " --decode-steps " << Base::config_.decode_steps
                << "' 2>&1";

            FILE *pipe = popen(cmd.str().c_str(), "r");
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
