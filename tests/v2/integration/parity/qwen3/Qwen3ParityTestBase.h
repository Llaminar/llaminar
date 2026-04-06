/**
 * @file Qwen3ParityTestBase.h
 * @brief Base class for Qwen3 PyTorch parity tests
 *
 * Extends the Qwen2 parity infrastructure for Qwen3 architecture.
 * Qwen3 differs from Qwen2 in:
 *   - Per-head QK RMSNorm (attn_q_norm.weight, attn_k_norm.weight)
 *   - No QKV biases
 *   - Different metadata prefix (qwen3. vs qwen2.)
 *
 * The underlying graph/schema machinery handles these differences
 * automatically via Qwen3SchemaFactory and the QKNormStage.
 * This test base only needs to override model paths and snapshot dirs.
 *
 * @author David Sanftenberg
 * @date 2026
 */

#pragma once

#include "../qwen2/Qwen2ParityTestBase.h"
#include "models/qwen3/Qwen3Schema.h"

namespace llaminar2::test::parity::qwen3
{

    // Import all Qwen2 parity utilities — Qwen3 is architecturally similar
    using namespace llaminar2::test::parity::qwen2;

    /**
     * @brief Config-driven parity test specialized for Qwen3 models.
     *
     * Inherits from the Qwen2 ConfigDrivenParityTest but overrides
     * the schema factory for TP weight sharding configuration.
     * Model path and snapshot dir come from TestConfig.
     */
    template <typename Derived>
    class Qwen3ConfigDrivenParityTest : public ConfigDrivenParityTest<Derived>
    {
    protected:
        using Base = ConfigDrivenParityTest<Derived>;

        void configureModel(std::shared_ptr<ModelContext> model_ctx) override
        {
            if (Base::cfg().is_local_tp() || Base::cfg().is_cross_rank_tp())
            {
                // Use Qwen3 schema factory for proper weight sharding
                // (no QKV biases, adds QK norm weights as replicated)
                Qwen3SchemaFactory schema_factory;
                model_ctx->weightManager()->setWeightShardingConfig(
                    schema_factory.getWeightShardingConfig());
            }
        }
    };

} // namespace llaminar2::test::parity::qwen3
